"""Bounded diagnostics for an otherwise unexplained image-stream rejection.

This never transforms images or accepts generated output. Only the already
observed, exact image-policy HTTP 400 is authoritative. Memo entries contain
only scoped request hashes and that error response, and expire after a minute.
"""
from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import requests

POLICY_MESSAGE = "Image processing blocked due to content policy violation."
DIAGNOSTIC_MAX_SECONDS = 10.0
DIAGNOSTIC_MAX_BYTES = 16 * 1024


def confirmed_image_policy_error(status_code, body):
    if status_code != 400 or not isinstance(body, dict):
        return None
    error = body.get("error")
    if (not isinstance(error, dict)
            or error.get("code") != "invalid_request_body"
            or error.get("message") != POLICY_MESSAGE):
        return None
    return {"code": error["code"], "message": error["message"]}


@dataclass
class ProbeResult:
    status_code: int | None = None
    body: Any = None
    response: Any = None
    error_type: str | None = None

    def close(self):
        if self.response is not None:
            self.response.close()


class BoundedErrorDiagnostics:
    """No queue; at most two live daemon workers, including socket cleanup.

    A timed-out/cancelled result cannot be accepted later. Its worker retains
    its slot until it really exits, so a stalled transport cannot cause an
    unbounded accumulation of threads. The caller's wait is capped at 10 s.
    """
    def __init__(self, capacity=2):
        self._slots = threading.BoundedSemaphore(capacity)

    def run(self, factory: Callable[[], ProbeResult], *, timeout=10.0,
            cancelled: Callable[[], bool] = lambda: False):
        if cancelled():
            return {"attempts": 0, "outcome": "cancelled"}
        if not self._slots.acquire(blocking=False):
            return {"attempts": 0, "outcome": "busy"}
        abandoned = threading.Event()
        results = queue.Queue(maxsize=1)
        deadline = time.monotonic() + min(max(float(timeout), 0.0), DIAGNOSTIC_MAX_SECONDS)

        def worker():
            probe = None
            try:
                if abandoned.is_set():
                    return
                try:
                    probe = factory()
                    if isinstance(probe, ProbeResult):
                        result = {"attempts": 1, "outcome": "returned",
                                  "status_code": probe.status_code, "body": probe.body}
                        if probe.error_type:
                            result.update(outcome="exception", error_type=probe.error_type)
                    else:
                        result = {"attempts": 1, "outcome": "ignored"}
                except Exception as exc:
                    result = {"attempts": 1, "outcome": "exception",
                              "error_type": type(exc).__name__}
                if not abandoned.is_set():
                    results.put_nowait(result)
            finally:
                # Publish before close, but hold the slot throughout close. A
                # buffered-reader lock must not block delivery of a known 400.
                try:
                    if isinstance(probe, ProbeResult):
                        probe.close()
                except Exception:
                    pass
                finally:
                    self._slots.release()

        try:
            threading.Thread(target=worker, daemon=True,
                             name="responses-error-diagnostic").start()
        except Exception:
            self._slots.release()
            return {"attempts": 0, "outcome": "worker_unavailable"}
        while True:
            if cancelled():
                abandoned.set()
                return {"attempts": 1, "outcome": "cancelled"}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                abandoned.set()
                return {"attempts": 1, "outcome": "timeout"}
            try:
                result = results.get(timeout=min(remaining, 0.05))
                if cancelled() or time.monotonic() >= deadline:
                    abandoned.set()
                    return {"attempts": 1, "outcome": "cancelled" if cancelled() else "timeout"}
                return result
            except queue.Empty:
                continue


def probe_image_failure(payload, headers, upstream_url, *, post=None):
    """Inspect a small nonstream error on the same input/provider/account.

    Do not route through the proxy again: diagnostics cannot enter recovery,
    cache successful output, invoke client tools, or teach schema memory.
    """
    diagnostic = copy.deepcopy(payload)
    diagnostic["stream"] = False
    diagnostic["tool_choice"] = "none"
    limit = diagnostic.get("max_output_tokens")
    diagnostic["max_output_tokens"] = min(limit, 16) if type(limit) is int and limit > 0 else 16
    response = None
    try:
        response = (post or requests.post)(
            upstream_url, headers=dict(headers), json=diagnostic,
            # The API payload is nonstreaming; stream=True here only bounds
            # how much HTTP response data this diagnostic reader retains.
            stream=True, timeout=(2.0, 8.0),
        )
        if response.status_code != 400:
            return ProbeResult(status_code=response.status_code, response=response)
        raw = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if len(raw) + len(chunk) > DIAGNOSTIC_MAX_BYTES:
                return ProbeResult(status_code=400, response=response)
            raw.extend(chunk)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            body = None
        # Probe-specific parameter/model errors cannot describe the original
        # failed stream. The allowlist is deliberately this one proven error.
        error = confirmed_image_policy_error(400, body)
        body = {"error": error} if error is not None else None
        return ProbeResult(status_code=400, body=body, response=response)
    except Exception as exc:
        return ProbeResult(response=response, error_type=type(exc).__name__)


def rejection_memo_key(payload, user_id, upstream, account_type, actual_model, headers):
    """Hash the complete original payload plus the authenticated request scope."""
    if not isinstance(payload, dict):
        return None
    client = payload.get("client_metadata")
    thread = headers.get("Thread-Id") or (client.get("thread_id") if isinstance(client, dict) else None)
    if not all(isinstance(x, str) and x for x in (user_id, upstream, actual_model, thread)):
        return None
    if len(thread) > 512:
        return None
    try:
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        for part in encoder.iterencode([user_id, upstream, account_type, actual_model, thread, payload]):
            digest.update(part.encode("utf-8"))
        return digest.hexdigest()
    except (TypeError, ValueError, UnicodeError):
        return None


class PermanentRejectionMemo:
    def __init__(self, ttl=60.0, capacity=128, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now):
        for key in [k for k, (until, _) in self._entries.items() if until <= now]:
            del self._entries[key]

    def get(self, key):
        if key is None:
            return None
        with self._lock:
            self._expire(self.clock())
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(value[1])

    def put(self, key, body):
        error = confirmed_image_policy_error(400, body)
        if key is None or error is None:
            return False
        with self._lock:
            now = self.clock()
            self._expire(now)
            self._entries[key] = (now + self.ttl, {"error": error})
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return True


error_diagnostics = BoundedErrorDiagnostics()
permanent_rejections = PermanentRejectionMemo()
