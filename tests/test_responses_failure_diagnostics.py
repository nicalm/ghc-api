"""Offline error-diagnostic and original-request memo regressions."""
import copy
import json
import threading
import time
from collections import OrderedDict
from unittest import mock

import pytest
import requests

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.portable_schema import PortableSchemaMemory
from ghc_api.responses_failure_diagnostics import (
    BoundedErrorDiagnostics, PermanentRejectionMemo, ProbeResult,
    POLICY_MESSAGE, confirmed_image_policy_error, probe_image_failure,
    rejection_memo_key,
)
from ghc_api.sse.openai_responses import RetryingResponsesResponse
from ghc_api.state import state

POLICY = {"error": {"code": "invalid_request_body", "message": POLICY_MESSAGE}}
EMPTY = {"type": "response.failed", "response": {"error": None, "output": []}}
CREATED = {"type": "response.created", "response": {"status": "in_progress"}}
SUCCESS = {"type": "response.completed", "response": {"status": "completed", "output": [], "usage": {}}}


class FakeResponse:
    def __init__(self, events=(), status=200, body=None, close_event=None):
        self.events, self.status_code, self.ok = events, status, status < 400
        self.body = body
        self.closed = False
        self.close_event = close_event
        self.text = json.dumps(body or {})

    def iter_lines(self):
        for event in self.events:
            yield ("event: " + event["type"]).encode()
            yield ("data: " + json.dumps(event)).encode()
            yield b""
        yield b"data: [DONE]"

    def iter_content(self, chunk_size):
        yield self.text.encode()

    def json(self):
        return self.body

    def close(self):
        self.closed = True
        if self.close_event:
            self.close_event.set()


def events(lines):
    return [json.loads(line[6:]) for line in lines
            if line.startswith(b"data: ") and line != b"data: [DONE]"]


def image_payload(image="rejected", prompt="check", stream=True):
    return {
        "model": "alias", "stream": stream,
        "max_output_tokens": 128, "tool_choice": "auto",
        "client_metadata": {"thread_id": "image-diagnostic-thread"},
        "input": [{"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "diagnostic", "description": "", "tools": []}]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + image, "detail": "high"}]}],
    }


@pytest.fixture
def isolated_pool(monkeypatch):
    pool = BoundedErrorDiagnostics(2)
    monkeypatch.setattr("ghc_api.sse.openai_responses.error_diagnostics", pool)
    return pool


def wrapper(probe, *, source=None, accept=None, retries=0, factory=None, timeout=0.3):
    return RetryingResponsesResponse(
        source or FakeResponse([CREATED, EMPTY]), factory or mock.Mock(), retries, "offline",
        error_diagnostic_factory=probe, on_error_diagnostic=accept,
        error_diagnostic_timeout=timeout, failure_diagnostic_timeout=0.1,
    )


def test_probe_preserves_input_model_history_and_only_changes_three_fields():
    original = image_payload()
    before = copy.deepcopy(original)
    response = FakeResponse(status=400, body=POLICY)
    post = mock.Mock(return_value=response)
    result = probe_image_failure(original, {"Authorization": "test-only"}, "https://same.invalid/v1/responses", post=post)
    assert original == before
    sent = post.call_args.kwargs["json"]
    expected = copy.deepcopy(before)
    expected.update(stream=False, tool_choice="none", max_output_tokens=16)
    assert sent == expected
    assert sent["input"] == before["input"]
    assert post.call_args.args[0] == "https://same.invalid/v1/responses"
    assert post.call_args.kwargs["headers"] == {"Authorization": "test-only"}
    assert result.status_code == 400 and result.body == POLICY
    result.close()
    assert response.closed


def test_probe_keeps_a_smaller_existing_output_limit():
    payload = image_payload()
    payload["max_output_tokens"] = 3
    post = mock.Mock(return_value=FakeResponse(status=200))
    result = probe_image_failure(payload, {}, "https://same.invalid", post=post)
    assert post.call_args.kwargs["json"]["max_output_tokens"] == 3
    assert result.body is None
    result.close()


@pytest.mark.parametrize("status,body", [
    (200, {"status": "completed", "output": [{"type": "function_call", "name": "never_execute"}]}),
    (400, {"error": {"code": "invalid_request_body", "message": "max_output_tokens too small"}}),
    (400, {"error": {"code": "invalid_request_body", "message": "tool_choice is unsupported"}}),
    (400, {"error": {"code": "invalid_request_body", "message": "model not supported"}}),
    (403, POLICY), (500, POLICY),
])
def test_probe_only_accepts_the_observed_image_policy_error(status, body):
    response = FakeResponse(status=status, body=body)
    result = probe_image_failure(image_payload(), {}, "https://same.invalid", post=mock.Mock(return_value=response))
    assert result.body is None
    result.close()
    assert response.closed


def test_confirmed_error_and_memo_never_retain_extra_payload_fields():
    body = copy.deepcopy(POLICY)
    body["private_debug"] = "PRIVATE"
    body["error"]["input_echo"] = "PRIVATE"
    memo = PermanentRejectionMemo()
    assert confirmed_image_policy_error(400, body) == POLICY["error"]
    assert memo.put("hash", body)
    assert memo.get("hash") == POLICY
    assert "PRIVATE" not in repr(memo._entries)


def test_probe_response_body_budget():
    response = FakeResponse(status=400, body=POLICY)
    response.text = "x" * (16 * 1024 + 1)
    result = probe_image_failure(image_payload(), {}, "https://same.invalid", post=mock.Mock(return_value=response))
    assert result.body is None
    result.close()


def test_confirmed_probe_enriches_first_failure_and_commits_once(isolated_pool):
    probe = mock.Mock(return_value=ProbeResult(400, POLICY))
    commit, generic, repair = mock.Mock(return_value=True), mock.Mock(), mock.Mock()
    stream = wrapper(probe, accept=commit, retries=3, factory=generic)
    stream._error_response_factory = repair
    output = events(list(stream.iter_lines()))
    assert output == [CREATED, {"type": "response.failed", "response": {"error": POLICY["error"], "output": []}}]
    probe.assert_called_once_with()
    commit.assert_called_once_with(POLICY)
    generic.assert_not_called()
    repair.assert_not_called()
    assert stream.error_diagnostic_fields()["failure_diagnostic"] == {
        "eligible": True, "attempts": 1, "outcome": "confirmed", "memo_hit": False,
        "status_code": 400, "memoized": True,
    }


@pytest.mark.parametrize("source", [
    [CREATED, SUCCESS],
    [{"type": "response.output_text.delta", "delta": "visible"}, EMPTY],
    [CREATED, {"type": "response.failed", "response": {"error": None, "output": [{"type": "message"}]}}],
    [CREATED, {"type": "response.failed", "response": {"error": POLICY["error"], "output": []}}],
    [CREATED, EMPTY, {"type": "error", **POLICY["error"]}],
    [{"type": "error", **POLICY["error"]}],
    [CREATED, EMPTY, {"type": "response.output_item.added", "item": {"type": "function_call"}}],
])
def test_no_diagnostic_with_output_or_any_existing_error(source, isolated_pool):
    probe, commit = mock.Mock(), mock.Mock()
    list(wrapper(probe, source=FakeResponse(source), accept=commit).iter_lines())
    probe.assert_not_called()
    commit.assert_not_called()


@pytest.mark.parametrize("probe_result", [ProbeResult(200, {"status": "completed"}), ProbeResult(400, {"error": {"code": "invalid_request_body", "message": "max_output_tokens"}}), ProbeResult(error_type="ReadTimeout")])
def test_unhelpful_diagnostic_runs_once_across_generic_retries(probe_result, isolated_pool):
    probe, commit = mock.Mock(return_value=probe_result), mock.Mock()
    generic = mock.Mock(side_effect=lambda: FakeResponse([CREATED, EMPTY]))
    stream = wrapper(probe, accept=commit, retries=3, factory=generic)
    assert events(list(stream.iter_lines())) == [CREATED, EMPTY]
    probe.assert_called_once_with()
    commit.assert_not_called()
    assert generic.call_count == 3


def test_timeout_has_no_late_commit_and_retains_original_error(isolated_pool):
    release, closed = threading.Event(), threading.Event()
    def probe():
        release.wait(2)
        return ProbeResult(400, POLICY, FakeResponse(close_event=closed))
    commit = mock.Mock()
    stream = wrapper(probe, accept=commit, timeout=0.025)
    start = time.monotonic()
    try:
        assert events(list(stream.iter_lines())) == [CREATED, EMPTY]
        assert time.monotonic() - start < 0.25
        assert stream.error_diagnostic_fields()["failure_diagnostic"]["outcome"] == "timeout"
    finally:
        release.set()
    assert closed.wait(1)
    commit.assert_not_called()


def test_close_during_diagnostic_has_no_late_commit(isolated_pool):
    release, entered, finished = threading.Event(), threading.Event(), threading.Event()
    commit, result = mock.Mock(), []
    def probe():
        entered.set()
        release.wait(2)
        return ProbeResult(400, POLICY)
    stream = wrapper(probe, accept=commit, timeout=1)
    def consume():
        result.extend(stream.iter_lines())
        finished.set()
    threading.Thread(target=consume, daemon=True).start()
    assert entered.wait(1)
    stream.close()
    assert finished.wait(0.3)
    release.set()
    commit.assert_not_called()
    assert result == []


def test_two_slots_remain_occupied_after_timeout_until_workers_exit():
    pool = BoundedErrorDiagnostics(2)
    release = threading.Event()
    entered = [threading.Event(), threading.Event()]
    calls = []
    def factory(index):
        entered[index].set()
        release.wait(2)
        return ProbeResult(400, POLICY)
    result = []
    runners = [threading.Thread(target=lambda i=i: result.append(pool.run(lambda: factory(i), timeout=0.03))) for i in range(2)]
    for runner in runners: runner.start()
    for item in entered: assert item.wait(1)
    for runner in runners: runner.join(1)
    assert all(r["outcome"] == "timeout" for r in result)
    before = time.monotonic()
    assert pool.run(lambda: calls.append(True), timeout=1)["outcome"] == "busy"
    assert time.monotonic() - before < 0.1 and not calls
    release.set()


def test_worker_holds_slot_while_close_blocks_but_delivers_result():
    release, closing = threading.Event(), threading.Event()
    class SlowClose(FakeResponse):
        def close(self):
            closing.set()
            release.wait(2)
    pool = BoundedErrorDiagnostics(1)
    try:
        result = pool.run(lambda: ProbeResult(400, POLICY, SlowClose()), timeout=0.2)
        assert result["body"] == POLICY and closing.wait(1)
        assert pool.run(lambda: ProbeResult(400, POLICY))["outcome"] == "busy"
    finally:
        release.set()


def test_pool_caps_a_requested_timeout_and_cancellation(monkeypatch):
    from ghc_api import responses_failure_diagnostics as module
    monkeypatch.setattr(module, "DIAGNOSTIC_MAX_SECONDS", 0.025)
    release = threading.Event()
    pool = BoundedErrorDiagnostics(1)
    start = time.monotonic()
    try:
        result = pool.run(lambda: (release.wait(2), ProbeResult())[1], timeout=100)
        assert result["outcome"] == "timeout"
        assert time.monotonic() - start < 0.25
        assert pool.run(lambda: ProbeResult(), cancelled=lambda: True)["outcome"] == "cancelled"
    finally:
        release.set()


def key(payload, user="u", upstream="https://up", account="individual", model="actual", thread="t"):
    return rejection_memo_key(payload, user, upstream, account, model, {"Thread-Id": thread})


def test_memo_scope_full_payload_ttl_and_capacity():
    now = [100.0]
    memo = PermanentRejectionMemo(ttl=60, capacity=2, clock=lambda: now[0])
    body = image_payload()
    original = key(body)
    assert memo.put(original, POLICY)
    assert memo.get(original) == POLICY
    changes = [key(body, user="other"), key(body, upstream="https://other"), key(body, account="other"), key(body, model="other"), key(body, thread="other")]
    for field, value in [("model", "new"), ("temperature", 0.3), ("tool_choice", "none")]:
        changed = copy.deepcopy(body)
        changed[field] = value
        changes.append(key(changed))
    changed = copy.deepcopy(body)
    changed["input"][-1]["content"][-1]["image_url"] += "changed"
    changes.append(key(changed))
    assert all(k != original and memo.get(k) is None for k in changes)
    now[0] = 160.01
    assert memo.get(original) is None
    for k in ["one", "two", "three"]: assert memo.put(k, POLICY)
    assert memo.get("one") is None and len(memo._entries) == 2
    assert not memo.put("bad", {"error": {"code": "invalid_request_body", "message": "different"}})


@pytest.fixture
def route_context(monkeypatch):
    for field, value in {"auto_remove_encrypted_content_on_parse_error": False,
                         "enable_portable_schema_recovery": True,
                         "sse_keepalive_interval": 0, "max_connection_retries": 0,
                         "enable_responses_early_failure_retry": True,
                         "save_request_to_file": False}.items():
        monkeypatch.setattr(state, field, value)
    monkeypatch.setattr(cache, "cache", OrderedDict())
    memory = PortableSchemaMemory(path=None)
    monkeypatch.setattr("ghc_api.portable_schema.portable_schema_memory", memory)
    monkeypatch.setattr("ghc_api.routes.openai.permanent_rejections", PermanentRejectionMemo())
    monkeypatch.setattr("ghc_api.sse.openai_responses.error_diagnostics", BoundedErrorDiagnostics(2))
    monkeypatch.setattr("ghc_api.routes.openai.ensure_copilot_token", lambda: None)
    monkeypatch.setattr("ghc_api.routes.openai.supports_responses_api", lambda model: True)
    monkeypatch.setattr("ghc_api.routes.openai.translate_model_name", lambda model: model + "-translated")
    monkeypatch.setattr("ghc_api.routes.openai.get_copilot_base_url", lambda: "https://synthetic-upstream.invalid")
    monkeypatch.setattr("ghc_api.routes.openai.get_copilot_headers", lambda vision: {"test": "header"})
    monkeypatch.setattr("ghc_api.routes.openai.log_error_request", lambda *args: None)
    return create_app().test_client(), memory


def test_route_first_failure_diagnoses_second_exact_request_is_http400(monkeypatch, route_context):
    client, memory = route_context
    body = image_payload()
    original = copy.deepcopy(body)
    post = mock.Mock(side_effect=[FakeResponse([CREATED, EMPTY]), FakeResponse(status=400, body=POLICY), FakeResponse([CREATED, SUCCESS])])
    monkeypatch.setattr("ghc_api.routes.openai.requests.post", post)
    with mock.patch.object(memory, "remember", wraps=memory.remember) as remember:
        first = client.post("/v1/responses", json=body, buffered=True)
        assert first.status_code == 200
        event_rows = [json.loads(line[6:]) for line in first.data.splitlines() if line.startswith(b"data: ") and line != b"data: [DONE]"]
        assert event_rows[1]["response"]["error"] == POLICY["error"]
        assert post.call_count == 2
        remember.assert_not_called()
        first_record = cache.get_recent_requests(1)[0]
        assert first_record["state"] == "error"
        assert first_record["failure_diagnostic"]["memoized"] is True
        assert first_record["failure_diagnostic"]["attempts"] == 1
        assert first_record["input_tokens"] == first_record["output_tokens"] == 0
        assert not first_record["portable_schema_recovery"]["completed"]
        second = client.post("/responses", json=body, buffered=True)
        assert second.status_code == 400 and second.json == POLICY
        assert second.headers["X-GHC-Error-Memo"] == "hit"
        assert post.call_count == 2
        second_record = cache.get_recent_requests(1)[0]
        assert second_record["failure_diagnostic"]["memo_hit"] is True
        assert second_record["original_request_body"] == original
    normal = client.post("/v1/responses", json=image_payload(image="different"), buffered=True)
    assert normal.status_code == 200 and b"response.completed" in normal.data
    assert post.call_count == 3
    assert body == original
    actual, diagnostic = [call.kwargs["json"] for call in post.call_args_list[:2]]
    assert actual["model"] == diagnostic["model"] == "alias-translated"
    assert actual["input"] == diagnostic["input"]
    assert diagnostic["stream"] is False and diagnostic["tool_choice"] == "none"
    assert diagnostic["max_output_tokens"] == 16
    assert "failure_diagnostic" in first_record and "data:image" not in json.dumps(first_record["failure_diagnostic"])


def test_route_successful_probe_does_not_complete_or_learn_original(monkeypatch, route_context):
    client, memory = route_context
    post = mock.Mock(side_effect=[FakeResponse([CREATED, EMPTY]), FakeResponse(status=200, body={"status": "completed", "output": [{"type": "function_call", "name": "do_not_run"}], "usage": {"input_tokens": 9}})])
    monkeypatch.setattr("ghc_api.routes.openai.requests.post", post)
    with mock.patch.object(memory, "remember", wraps=memory.remember) as remember:
        response = client.post("/v1/responses", json=image_payload(), buffered=True)
        remember.assert_not_called()
    assert b"response.failed" in response.data and b"response.completed" not in response.data
    assert b"do_not_run" not in response.data
    record = cache.get_recent_requests(1)[0]
    assert record["failure_diagnostic"]["outcome"] == "ignored"
    assert record["failure_diagnostic"]["status_code"] == 200
    assert record["input_tokens"] == record["output_tokens"] == 0
    assert not record["portable_schema_recovery"]["completed"]


def test_route_text_only_failure_has_no_diagnostic(monkeypatch, route_context):
    client, _ = route_context
    body = image_payload()
    body["input"][-1]["content"] = [{"type": "input_text", "text": "only text"}]
    post = mock.Mock(return_value=FakeResponse([CREATED, EMPTY]))
    monkeypatch.setattr("ghc_api.routes.openai.requests.post", post)
    response = client.post("/v1/responses", json=body, buffered=True)
    assert b"response.failed" in response.data
    assert post.call_count == 1
    assert "failure_diagnostic" not in cache.get_recent_requests(1)[0]
