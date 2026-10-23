"""Preserve portable encrypted tool arguments by restoring proven declarations."""
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict

from . import utils


_DEFAULT_PATH = object()
_ERROR = ("Portable encrypted function call arguments require their original tool "
          "declaration or saved encryption schema.")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def is_portable_schema_error(error):
    return (isinstance(error, dict) and error.get("code") == "invalid_request_body"
            and error.get("message") == _ERROR)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _encrypted(value, depth=0):
    if depth > 32:
        return False
    if isinstance(value, dict):
        return value.get("encrypted") is True or any(_encrypted(v, depth + 1) for v in value.values())
    return isinstance(value, list) and any(_encrypted(v, depth + 1) for v in value)


def _declarations(roots):
    """Index exact declaration paths, retaining conflicting versions separately."""
    result = {}

    def visit(tools, prefix="", parents=()):
        if not isinstance(tools, list) or len(parents) > 16:
            return
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            name = prefix + tool["name"]
            if tool.get("type") == "namespace":
                visit(tool.get("tools"), name + ".", parents + (tool,))
            elif tool.get("type") == "function":
                root = copy.deepcopy(tool)
                for parent in reversed(parents):
                    root = {**copy.deepcopy(parent), "tools": [root]}
                result.setdefault(name, {})[_digest(root)] = root

    visit(roots)
    return result


def _tools(payload):
    roots = list(payload.get("tools") or []) if isinstance(payload.get("tools"), list) else []
    for item in payload.get("input", []) if isinstance(payload.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools" and isinstance(item.get("tools"), list):
            roots.extend(item["tools"])
    return roots


def _calls(items):
    for call in items if isinstance(items, list) else []:
        if not isinstance(call, dict) or call.get("type") != "function_call":
            continue
        name, namespace = call.get("name"), call.get("namespace", "")
        if (not isinstance(name, str) or not name or not isinstance(namespace, str)
                or not isinstance(call.get("call_id"), str) or not isinstance(call.get("arguments"), str)):
            continue
        qualified = (namespace + "." if namespace else "") + name
        yield qualified, _digest([name, namespace, call["call_id"], call["arguments"]])


def _merge(roots):
    """Combine namespace paths only when all existing metadata agrees exactly."""
    result = []
    for root in roots:
        old = next((t for t in result if t.get("name") == root.get("name")), None)
        if old is None:
            result.append(copy.deepcopy(root))
        elif old != root:
            if (old.get("type") != "namespace" or root.get("type") != "namespace"
                    or {k: v for k, v in old.items() if k != "tools"}
                    != {k: v for k, v in root.items() if k != "tools"}):
                raise ValueError("conflicting declaration")
            old["tools"] = _merge(old["tools"] + root["tools"])
    return result


class PortableSchemaMemory:
    """Bounded persistent declaration mappings; no call bodies or credentials."""
    def __init__(self, path=_DEFAULT_PATH, capacity=8192, max_bytes=8 * 1024 * 1024,
                 ttl=86400, clock=time.time):
        self.path, self.capacity, self.max_bytes, self.ttl, self.clock = path, capacity, max_bytes, ttl, clock
        self._loaded_path = object()
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def _load(self):
        path = (os.path.join(utils.get_config_dir(), "portable-schema-memory.json")
                if self.path is _DEFAULT_PATH else self.path)
        if path == self._loaded_path:
            return
        self._loaded_path, self._entries = path, OrderedDict()
        if path is None:
            return
        try:
            with open(path, "rb") as source:
                raw = source.read(self.max_bytes + 1)
            data = json.loads(raw)
            if len(raw) > self.max_bytes or data.get("version") != 1 or not isinstance(data.get("bindings"), list):
                raise ValueError("invalid memory")
            if len(data["bindings"]) > self.capacity:
                raise ValueError("oversized memory")
            for entry in data["bindings"]:
                if (set(entry) != {"scope", "call", "expires", "declaration"}
                        or not all(isinstance(entry[k], str) and _HASH.fullmatch(entry[k]) for k in ("scope", "call"))
                        or not isinstance(entry["expires"], (float, int))
                        or not 0 < entry["expires"] <= self.clock() + self.ttl + 1
                        or (entry["declaration"] is not None and len(_declarations([entry["declaration"]])) != 1)):
                    raise ValueError("invalid binding")
                key = (entry["scope"], entry["call"])
                if key in self._entries:
                    raise ValueError("duplicate binding")
                self._entries[key] = entry
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            self._entries.clear()
            print("[Portable Schema] Memory load declined: " + type(exc).__name__)

    def _trim(self):
        for key, entry in list(self._entries.items()):
            if entry["expires"] <= self.clock():
                del self._entries[key]
        sizes = {key: len(_json(entry).encode("utf-8")) + 1 for key, entry in self._entries.items()}
        size = 64 + sum(sizes.values())
        while self._entries and (len(self._entries) > self.capacity or size > self.max_bytes):
            key, _ = self._entries.popitem(last=False)
            size -= sizes[key]

    def _save(self):
        if self._loaded_path is None:
            return
        temporary = None
        try:
            directory = os.path.dirname(os.path.abspath(self._loaded_path))
            os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=".portable-schema-", delete=False) as output:
                temporary = output.name
                output.write(_json({"version": 1, "bindings": list(self._entries.values())}))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self._loaded_path)
        except OSError as exc:
            print("[Portable Schema] Memory save declined: " + type(exc).__name__)
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def lookup(self, scope):
        if scope is None:
            return {}
        with self._lock:
            self._load()
            self._trim()
            return {call: copy.deepcopy(entry["declaration"])
                    for (owner, call), entry in self._entries.items()
                    if owner == scope and entry["declaration"] is not None}

    def remember(self, scope, bindings):
        if scope is None or not bindings:
            return
        with self._lock:
            self._load()
            self._trim()
            for call, root in bindings.items():
                key = (scope, call)
                old = self._entries.get(key)
                # Once a call maps to different schemas, retain a bounded
                # tombstone instead of guessing which version produced it.
                if old is not None and old["declaration"] != root:
                    root = None
                self._entries[key] = {"scope": scope, "call": call, "declaration": copy.deepcopy(root),
                                      "expires": self.clock() + self.ttl}
                self._entries.move_to_end(key)
            self._trim()
            self._save()

    def clear(self):
        with self._lock:
            self._load()
            self._entries.clear()
            self._save()


portable_schema_memory = PortableSchemaMemory()


class PortableSchemaRecovery:
    def __init__(self, payload, user_id, upstream, account_type, enabled,
                 request_kind=None, memory=None):
        self.enabled, self.request_kind = enabled, request_kind
        self.memory = memory if memory is not None else portable_schema_memory
        self._payload = copy.deepcopy(payload)
        metadata = payload.get("client_metadata")
        thread = metadata.get("thread_id") if isinstance(metadata, dict) else None
        self.scope = (_digest([user_id, upstream, account_type, payload.get("model"), thread])
                      if enabled and isinstance(thread, str) and 0 < len(thread) <= 512 else None)
        self.attempted, self.completed, self.restored_declarations = False, False, 0
        self.events = []

    def diagnostic(self, event):
        if len(self.events) < 16:
            self.events.append({k: v for k, v in event.items() if k in {
                "stage", "error_type", "status_code", "error_code", "after_recovery"}})

    def track_payload(self, payload):
        """Track the actual sent copy after any independent history cleanup."""
        self._payload = copy.deepcopy(payload)

    def _restore(self, payload):
        if (not self.enabled or self.scope is None or
                (self.request_kind != "compaction" and payload.get("tool_choice") != "none")):
            return payload, 0
        known, current = self.memory.lookup(self.scope), _declarations(_tools(payload))
        missing = {}
        for name, fingerprint in _calls(payload.get("input")):
            root = known.get(fingerprint)
            if root is None:
                continue
            variants = _declarations([root]).get(name, {})
            combined = {**current.get(name, {}), **missing.get(name, {}), **variants}
            if len(combined) != 1:
                self.diagnostic({"stage": "schema_conflict"})
                return payload, 0
            if name not in current:
                missing[name] = combined
        if not missing:
            return payload, 0
        try:
            roots = _merge([next(iter(versions.values())) for versions in missing.values()])
            # Validate namespace metadata against all declarations already in
            # the request. No existing input item is modified or replaced.
            _merge(_tools(payload) + roots)
        except ValueError:
            self.diagnostic({"stage": "schema_conflict"})
            return payload, 0
        item = {"type": "additional_tools", "role": "developer", "tools": roots,
                "id": "at_ghc_schema_" + _digest(roots)[:24]}
        result = {**payload, "input": [item] + copy.deepcopy(payload["input"]), "tool_choice": "none"}
        self.track_payload(result)
        self.restored_declarations += len(missing)
        return result, len(missing)

    def prepare(self, payload):
        result, count = self._restore(payload)
        if count:
            self.diagnostic({"stage": "schema_prepared"})
        return result

    def recover(self, payload, error):
        if self.attempted or not is_portable_schema_error(error):
            return payload, 0
        result, count = self._restore(payload)
        if count:
            self.attempted = True
        self.diagnostic({"stage": "schema_restored" if count else "schema_unavailable"})
        return result, count

    def complete(self, response=None):
        if self.completed or (isinstance(response, dict) and response.get("status", "completed") != "completed"):
            return
        self.completed = True
        if self.enabled and self.scope is not None:
            declarations, bindings = _declarations(_tools(self._payload)), {}
            items = self._payload.get("input")
            items = list(items) if isinstance(items, list) else []
            if isinstance(response, dict) and isinstance(response.get("output"), list):
                items.extend(response["output"])
            for name, fingerprint in _calls(items):
                versions = declarations.get(name, {})
                if len(versions) == 1:
                    root = next(iter(versions.values()))
                    if _encrypted(root):
                        bindings[fingerprint] = root
            self.memory.remember(self.scope, bindings)
        self.diagnostic({"stage": "completed", "after_recovery": self.attempted})

    def cache_fields(self):
        return {"portable_schema_recovery": {
            "attempted": self.attempted, "completed": self.completed,
            "restored_declarations": self.restored_declarations, "events": list(self.events)}}
