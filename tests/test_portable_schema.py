import copy
import json
import os

import pytest

from ghc_api.portable_schema import (
    PortableSchemaMemory, PortableSchemaRecovery, is_portable_schema_error,
)


ERROR = {"code": "invalid_request_body", "message": (
    "Portable encrypted function call arguments require their original tool "
    "declaration or saved encryption schema."
)}
TOOL = {"type": "namespace", "name": "collaboration", "description": "Agent tools.", "tools": [
    {"type": "function", "name": "send_message", "strict": False,
     "parameters": {"type": "object", "properties": {
         "message": {"type": "string", "encrypted": True},
         "target": {"type": "string"}}, "required": ["message", "target"]}},
]}
CALL = {"type": "function_call", "namespace": "collaboration", "name": "send_message",
        "call_id": "call-1", "arguments": '{"message":"PRIVATE-CIPHER","target":"child"}'}
OUTPUT = {"type": "function_call_output", "call_id": "call-1", "output": "KEEP PLAINTEXT"}


def payload(declare=False, **fields):
    result = {"model": "test", "client_metadata": {"thread_id": "thread-1"},
              "input": [copy.deepcopy(CALL), copy.deepcopy(OUTPUT)]}
    if declare:
        result["input"].insert(0, {"type": "additional_tools", "role": "developer",
                                   "id": "original", "tools": [copy.deepcopy(TOOL)]})
    result.update(fields)
    return result


def recovery(body, memory, **kwargs):
    return PortableSchemaRecovery(body, "user", "https://upstream", "individual", True,
                                  memory=memory, **kwargs)


def learned(memory):
    source = recovery(payload(True), memory)
    source.complete({"status": "completed", "output": []})
    return source


def test_exact_error_match_does_not_capture_generic_encryption_failures():
    assert is_portable_schema_error(ERROR)
    for error in (None, {}, {**ERROR, "code": "invalid_request_error"},
                  {**ERROR, "message": ERROR["message"] + " extra"},
                  {"code": "invalid_request_body", "message": "cannot decrypt output"}):
        assert not is_portable_schema_error(error)


def test_compaction_restores_exact_declaration_without_changing_history():
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload()
    original = copy.deepcopy(body)
    transaction = recovery(body, memory, request_kind="compaction")
    prepared = transaction.prepare(body)
    assert prepared["tool_choice"] == "none"
    assert prepared["input"][0]["tools"] == [TOOL]
    assert prepared["input"][1:] == body["input"]
    assert body == original
    assert not transaction.attempted
    assert transaction.cache_fields()["portable_schema_recovery"]["restored_declarations"] == 1


def test_failed_or_uncompleted_request_never_teaches_memory():
    memory = PortableSchemaMemory(path=None)
    transaction = recovery(payload(True), memory)
    transaction.complete({"status": "failed", "output": []})
    assert not transaction.completed
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) == body


def test_completed_output_calls_are_bound_even_when_not_yet_in_input():
    memory = PortableSchemaMemory(path=None)
    body = payload(True)
    body["input"] = body["input"][:1]
    recovery(body, memory).complete({"status": "completed", "output": [CALL]})
    compact = payload(tool_choice="none")
    assert recovery(compact, memory).prepare(compact) != compact


@pytest.mark.parametrize("field,value", [("name", "other"), ("namespace", "other"),
                                         ("call_id", "new"), ("arguments", "fresh-cipher")])
def test_call_identity_and_argument_bytes_must_match(field, value):
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload(tool_choice="none")
    body["input"][0][field] = value
    assert recovery(body, memory).prepare(body) == body


@pytest.mark.parametrize("change", ["user", "upstream", "account", "model", "thread", "missing-thread"])
def test_no_cross_scope_restore(change):
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload(tool_choice="none")
    args = ["user", "https://upstream", "individual"]
    if change in ("user", "upstream", "account"):
        args[("user", "upstream", "account").index(change)] = "different"
    elif change == "model":
        body["model"] = "other"
    elif change == "thread":
        body["client_metadata"]["thread_id"] = "other"
    else:
        body.pop("client_metadata")
        body["prompt_cache_key"] = "thread-1"
    assert PortableSchemaRecovery(body, *args, True, memory=memory).prepare(body) == body


def test_only_no_tools_requests_can_restore_and_retry_is_single_use():
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload()
    transaction = recovery(body, memory)
    assert transaction.prepare(body) == body
    assert transaction.recover(body, ERROR) == (body, 0)
    body["tool_choice"] = "none"
    transaction = recovery(body, memory)
    repaired, count = transaction.recover(body, ERROR)
    assert count == 1 and transaction.attempted
    assert repaired["input"][1:] == body["input"]
    assert transaction.recover(body, ERROR) == (body, 0)


def test_disabled_mode_does_not_load_or_remember():
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload(tool_choice="none")
    transaction = PortableSchemaRecovery(body, "user", "https://upstream", "individual", False,
                                         memory=memory)
    assert transaction.prepare(body) == body
    assert transaction.recover(body, ERROR) == (body, 0)


def test_conflicting_declaration_is_never_overwritten_or_learned():
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    body = payload(True, tool_choice="none")
    body["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"]["message"]["type"] = "number"
    assert recovery(body, memory).prepare(body) == body
    body["tools"] = [copy.deepcopy(TOOL)]
    second = PortableSchemaMemory(path=None)
    recovery(body, second).complete()
    compact = payload(tool_choice="none")
    assert recovery(compact, second).prepare(compact) == compact


def test_top_level_tools_and_nested_namespaces_are_supported():
    memory = PortableSchemaMemory(path=None)
    tool = {"type": "namespace", "name": "outer", "tools": [copy.deepcopy(TOOL)]}
    body = payload(tools=[tool])
    body["input"][0]["namespace"] = "outer.collaboration"
    recovery(body, memory).complete()
    compact = copy.deepcopy(body)
    compact.pop("tools")
    compact["tool_choice"] = "none"
    assert recovery(compact, memory).prepare(compact)["input"][0]["tools"] == [tool]


def test_plaintext_schema_is_not_remembered():
    memory = PortableSchemaMemory(path=None)
    body = payload(True)
    body["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"]["message"].pop("encrypted")
    recovery(body, memory).complete()
    compact = payload(tool_choice="none")
    assert recovery(compact, memory).prepare(compact) == compact


def test_persistence_is_hash_only_and_survives_restart(tmp_path):
    path = str(tmp_path / "memory.json")
    learned(PortableSchemaMemory(path=path))
    raw = (tmp_path / "memory.json").read_text(encoding="utf-8")
    for private in ("PRIVATE-CIPHER", "KEEP PLAINTEXT", "call-1", "thread-1", "https://upstream"):
        assert private not in raw
    body = payload(tool_choice="none")
    assert recovery(body, PortableSchemaMemory(path=path)).prepare(body) != body


def test_expiration_capacity_and_byte_budget(tmp_path):
    now = [1000]
    path = str(tmp_path / "memory.json")
    memory = PortableSchemaMemory(path=path, capacity=1, max_bytes=2048, ttl=10, clock=lambda: now[0])
    learned(memory)
    source = payload(True)
    source["input"][1]["call_id"] = "call-2"
    recovery(source, memory).complete()
    old = payload(tool_choice="none")
    assert recovery(old, memory).prepare(old) == old
    source["input"].pop(0)
    source["tool_choice"] = "none"
    assert recovery(source, memory).prepare(source) != source
    now[0] += 11
    assert recovery(source, memory).prepare(source) == source
    assert os.path.getsize(path) <= 2048
    tiny = PortableSchemaMemory(path=None, max_bytes=100)
    learned(tiny)
    assert recovery(old, tiny).prepare(old) == old


@pytest.mark.parametrize("contents", ["not json", '{"version":2,"bindings":[]}',
                                     '{"version":1,"bindings":[{}]}'])
def test_corrupt_file_fails_closed(tmp_path, contents):
    path = tmp_path / "memory.json"
    path.write_text(contents, encoding="utf-8")
    body = payload(tool_choice="none")
    assert recovery(body, PortableSchemaMemory(path=str(path))).prepare(body) == body


def test_default_memory_resolves_config_lazily_and_tracks_path_changes(tmp_path, monkeypatch):
    memory = PortableSchemaMemory()
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path / "first"))
    learned(memory)
    assert (tmp_path / "first" / "portable-schema-memory.json").exists()
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path / "second"))
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) == body


def test_diagnostics_never_include_upstream_message_or_exception_details():
    transaction = recovery(payload(), PortableSchemaMemory(path=None))
    transaction.diagnostic({"stage": "recovery_exception", "error_type": "ValueError",
                            "message": "SECRET", "arguments": "SECRET"})
    assert "SECRET" not in json.dumps(transaction.cache_fields())


def test_final_payload_tracking_does_not_bind_calls_removed_before_send():
    memory = PortableSchemaMemory(path=None)
    original = payload(True)
    transaction = recovery(original, memory)
    final = copy.deepcopy(original)
    final["input"].pop(1)
    transaction.track_payload(final)
    transaction.complete()
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) == body


def test_string_input_is_supported_and_completed_output_still_binds():
    memory = PortableSchemaMemory(path=None)
    source = payload(input="plain user input", tools=[TOOL])
    recovery(source, memory).complete({"status": "completed", "output": [CALL]})
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) != body


def test_same_call_with_conflicting_schemas_stays_unbound_until_expiry():
    now = [1000]
    memory = PortableSchemaMemory(path=None, ttl=10, clock=lambda: now[0])
    learned(memory)
    other = payload(True)
    other["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"]["message"]["description"] = "Different version"
    recovery(other, memory).complete()
    learned(memory)
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) == body
    now[0] += 11
    learned(memory)
    assert recovery(body, memory).prepare(body) != body


def test_two_same_name_schema_versions_in_history_decline_restoration():
    memory = PortableSchemaMemory(path=None)
    learned(memory)
    other = payload(True)
    other["input"][1]["call_id"] = "other-call"
    other["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"]["message"]["description"] = "Other version"
    recovery(other, memory).complete()
    body = payload(tool_choice="none")
    body["input"].append(other["input"][1])
    assert recovery(body, memory).prepare(body) == body


def test_multiple_tools_in_one_namespace_are_restored_together():
    memory = PortableSchemaMemory(path=None)
    source = payload(True)
    namespace = source["input"][0]["tools"][0]
    second_tool = copy.deepcopy(namespace["tools"][0])
    second_tool["name"] = "followup_task"
    namespace["tools"].append(second_tool)
    second_call = {**CALL, "name": "followup_task", "call_id": "second-call"}
    source["input"].append(second_call)
    recovery(source, memory).complete()
    body = payload(tool_choice="none")
    body["input"].append(second_call)
    assert recovery(body, memory).prepare(body)["input"][0]["tools"] == [namespace]


def test_failed_atomic_write_preserves_prior_file_and_does_not_fail_request(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    memory = PortableSchemaMemory(path=str(path))
    learned(memory)
    previous = path.read_bytes()
    def fail_replace(*args):
        raise PermissionError("PRIVATE ERROR MESSAGE")
    monkeypatch.setattr("ghc_api.portable_schema.os.replace", fail_replace)
    source = payload(True)
    source["input"][1]["call_id"] = "new-call"
    recovery(source, memory).complete()
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_schemas_with_escaped_surrogates_do_not_crash_persistence(tmp_path):
    memory = PortableSchemaMemory(path=str(tmp_path / "memory.json"))
    source = payload(True)
    source["input"][0]["tools"][0]["description"] = "Escaped surrogate: \ud800"
    recovery(source, memory).complete()
    body = payload(tool_choice="none")
    assert recovery(body, memory).prepare(body) != body
