"""The portable declaration repair must not opt callers into history cleanup."""

import copy
import json
from unittest import mock

import pytest

from ghc_api.encrypted_recovery import EncryptionRecovery, RecoveryMemory
from ghc_api.portable_schema import PortableSchemaMemory


SCHEMA_ERROR = {"code": "invalid_request_body", "message": (
    "Portable encrypted function call arguments require their original tool "
    "declaration or saved encryption schema.")}
CONTENT_ERROR = {"code": "invalid_request_body", "message":
                 "Encrypted function output content could not be decrypted or decoded."}
CALL = {"type": "function_call", "namespace": "collaboration", "name": "send_message",
        "call_id": "original-call", "arguments": '{"target":"child","message":"opaque"}'}
OUTPUT = {"type": "function_call_output", "call_id": "original-call", "output": "completed"}
ATTACHMENT = {"type": "reasoning", "encrypted_content": "preserve-exact-opaque-history"}
DECLARATION = {"type": "additional_tools", "role": "developer", "id": "original", "tools": [
    {"type": "namespace", "name": "collaboration", "description": "Agent tools", "tools": [
        {"type": "function", "name": "send_message", "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}, "message": {"type": "string", "encrypted": True}}}}]}]}


def payload(declared=False):
    return {"model": "test-model", "client_metadata": {"thread_id": "original-thread"},
            "tool_choice": "none", "input": copy.deepcopy(
                ([DECLARATION] if declared else []) + [CALL, OUTPUT, ATTACHMENT])}


def transaction(body, enabled=False, portable_enabled=True):
    return EncryptionRecovery(body, "user", "https://upstream.invalid", "individual", enabled,
                              request_kind="compaction", portable_enabled=portable_enabled)


@pytest.fixture(autouse=True)
def isolated_memories(monkeypatch):
    monkeypatch.setattr("ghc_api.portable_schema.portable_schema_memory", PortableSchemaMemory(path=None))
    monkeypatch.setattr("ghc_api.encrypted_recovery.recovery_memory", RecoveryMemory())


@pytest.mark.parametrize("enabled", [False, True])
def test_omitted_new_flag_preserves_upstream_enable_behavior(enabled):
    item = EncryptionRecovery(payload(True), "user", "https://upstream.invalid", "individual", enabled)
    assert item.enabled is enabled
    assert item.portable.enabled is enabled


def test_schema_can_learn_and_prepare_while_content_cleanup_is_disabled():
    transaction(payload(True)).complete({"status": "completed", "output": []})
    original = payload()
    before = copy.deepcopy(original)
    item = transaction(original)
    with mock.patch("ghc_api.encrypted_recovery.remove_encrypted_content_items",
                    side_effect=AssertionError("history cleanup must remain disabled")):
        restored = item.prepare(original)
    assert restored["input"][0]["tools"] == DECLARATION["tools"]
    assert restored["input"][1:] == before["input"]
    assert restored["tool_choice"] == "none"
    assert original == before
    assert not item.attempted and item.precleaned_items == 0


def test_schema_can_recover_http_error_once_without_content_cleanup():
    transaction(payload(True)).complete({"status": "completed", "output": []})
    original = payload()
    item = transaction(original)
    with mock.patch("ghc_api.encrypted_recovery.remove_encrypted_content_items",
                    side_effect=AssertionError("history cleanup must remain disabled")):
        restored, count = item.recover_http_error(original, 400, json.dumps({"error": SCHEMA_ERROR}))
        assert count == 1
        assert restored["input"][1:] == original["input"]
        assert item.has_attempted and item.portable.attempted and not item.attempted
        assert item.recover_error(original, SCHEMA_ERROR) == (original, 0)


def test_cleanup_disabled_blocks_direct_recovery_http_errors_and_precleaning():
    original = payload()
    item = transaction(original)
    with mock.patch("ghc_api.encrypted_recovery.remove_encrypted_content_items",
                    side_effect=AssertionError("history cleanup must remain disabled")), \
         mock.patch("ghc_api.encrypted_recovery.recovery_memory.lookup",
                    side_effect=AssertionError("cleanup memory must not be read")):
        assert item.prepare(original) is original
        assert item.recover(original) == (original, 0)
        assert item.recover_error(original, CONTENT_ERROR) == (original, 0)
        assert item.recover_http_error(original, 400, json.dumps({"error": CONTENT_ERROR})) == (original, 0)
    assert item.pending == set()
    assert ATTACHMENT in original["input"]


def test_portable_disabled_is_independent_of_enabled_legacy_content_cleanup():
    item = transaction(payload(True), enabled=True, portable_enabled=False)
    item.complete({"status": "completed", "output": []})
    original = payload()
    next_item = transaction(original, enabled=True, portable_enabled=False)
    assert next_item.prepare(original) is original
    assert next_item.recover_error(original, SCHEMA_ERROR) == (original, 0)
    cleaned, count = next_item.recover_error(original, CONTENT_ERROR)
    assert count == 1
    assert cleaned["input"] == [CALL, OUTPUT]


def test_failed_response_never_teaches_enabled_portable_memory():
    transaction(payload(True)).complete({"status": "failed", "output": []})
    original = payload()
    assert transaction(original).prepare(original) is original
