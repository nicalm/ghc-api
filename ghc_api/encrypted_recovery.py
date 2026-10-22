"""Bounded, process-local memory of successfully recovered encrypted histories."""
import hashlib
import json
import threading
import time
from collections import OrderedDict

from .counters import counters
from .utils import encrypted_content_fingerprint, remove_encrypted_content_items


def _fingerprints(value, depth=0):
    if depth > 32:
        return set()
    if isinstance(value, dict):
        result = set()
        for key, child in value.items():
            if key == "encrypted_content":
                result.add(encrypted_content_fingerprint(child))
            else:
                result.update(_fingerprints(child, depth + 1))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_fingerprints(child, depth + 1))
        return result
    return set()


class RecoveryMemory:
    """Retain only scoped SHA-256 values, for at most 24h and 8192 entries."""
    def __init__(self, capacity=8192, ttl=86400, clock=time.monotonic):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self):
        now = self.clock()
        for key in list(self._entries):
            if self._entries[key] <= now:
                del self._entries[key]

    def lookup(self, scope):
        if scope is None:
            return set()
        with self._lock:
            self._expire()
            return {fingerprint for (owner, fingerprint) in self._entries if owner == scope}

    def remember(self, scope, fingerprints):
        if scope is None:
            return
        with self._lock:
            self._expire()
            for fingerprint in sorted(fingerprints):
                key = (scope, fingerprint)
                self._entries[key] = self.clock() + self.ttl
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()


recovery_memory = RecoveryMemory()


class EncryptionRecovery:
    """One request's preclean/recovery transaction and redacted diagnostics."""
    def __init__(self, payload, user_id, upstream, account_type, enabled):
        self.enabled = enabled
        metadata = payload.get("client_metadata") or {}
        thread = metadata.get("thread_id") if isinstance(metadata, dict) else None
        # A provider cache key alone need not be a session identifier. Without a
        # stable explicit thread ID, retain the ordinary per-request behavior.
        self.scope = None
        if enabled and isinstance(thread, str) and 0 < len(thread) <= 512:
            identity = [user_id, upstream, account_type, payload.get("model"), thread]
            self.scope = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.pending = set()
        self.attempted = False
        self.completed = False
        self.precleaned_items = 0
        self.events = []

    def diagnostic(self, event):
        # The stream handler commits completion only after consuming the event;
        # the wrapper also observes it, but must not learn speculatively.
        if event.get("stage") == "completed":
            return
        if event.get("stage") == "original_rejected" and self.precleaned_items:
            event = {**event, "stage": "precleaned_rejected"}
        if len(self.events) < 16:
            self.events.append({k: v for k, v in event.items()
                                if k in {"stage", "error_type", "status_code", "error_code", "after_recovery"}})

    def prepare(self, payload):
        known = recovery_memory.lookup(self.scope) if self.enabled else set()
        if not known:
            return payload
        cleaned, count = remove_encrypted_content_items(payload.get("input"), fingerprints=known)
        if count:
            self.precleaned_items = count
            self.diagnostic({"stage": "precleaned"})
            counters.incr("mod.encrypted_content_preclean")
            return {**payload, "input": cleaned}
        return payload

    def recover(self, payload):
        cleaned, count = remove_encrypted_content_items(payload.get("input"))
        if count:
            self.attempted = True
            self.pending.update(_fingerprints(payload.get("input")))
            if not any(event.get("stage") == "original_rejected" for event in self.events):
                self.diagnostic({"stage": "original_rejected"})
            self.diagnostic({"stage": "cleanup_applied"})
        return {**payload, "input": cleaned}, count

    def complete(self):
        if self.completed:
            return
        self.completed = True
        if self.enabled and self.pending:
            recovery_memory.remember(self.scope, self.pending)
            counters.incr("mod.encrypted_content_recovery_completed")
        if len(self.events) < 16:
            self.events.append({"stage": "completed", "after_recovery": self.attempted})

    def cache_fields(self):
        return {"encrypted_recovery": {
            "precleaned_items": self.precleaned_items, "attempted": self.attempted,
            "completed": self.completed, "events": list(self.events),
        }}
