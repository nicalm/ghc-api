"""Only a streaming compaction receives an image-limit SSE failure."""

import copy
import json
from collections import OrderedDict
from contextlib import ExitStack
from unittest import mock

import pytest

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.portable_schema import PortableSchemaMemory
from ghc_api.responses_image_guard import image_limit_compaction_sse, image_limit_error
from ghc_api.state import state


def payload(images=52, stream=True, output_kind="custom_tool_call_output"):
    return {
        "model": "gpt-test",
        "stream": stream,
        "input": [{
            "type": output_kind,
            "call_id": "synthetic-call",
            "output": [
                {"type": "input_image", "image_url": "data:image/png;base64,synthetic"}
                for _ in range(images)
            ],
        }],
    }


def events(wire):
    return [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]


def test_streaming_compaction_gets_a_recognizable_terminal_error():
    value = payload()
    error = image_limit_error(value)
    original, original_error = copy.deepcopy(value), copy.deepcopy(error)
    result = events(image_limit_compaction_sse(
        value, error, '{"request_kind":"compaction"}', "synthetic-request"
    ))
    assert [item["type"] for item in result] == ["response.created", "response.failed"]
    assert [item["sequence_number"] for item in result] == [0, 1]
    created, failure = [item["response"] for item in result]
    assert created["status"] == "in_progress" and created["error"] is None
    assert failure["id"] == created["id"]
    assert failure["status"] == "failed"
    assert failure["error"] == error["error"]
    assert failure["error"]["code"] == "context_length_exceeded"
    assert failure["output"] == [] and failure["usage"] is None
    assert value == original and error == original_error


@pytest.mark.parametrize("stream,metadata", [
    (True, None),
    (True, "not-json"),
    (True, "[]"),
    (True, "null"),
    (True, {"request_kind": "turn"}),
    (False, {"request_kind": "compaction"}),
])
def test_ordinary_nonstream_and_unknown_metadata_keep_http_error(stream, metadata):
    value = payload(stream=stream)
    assert image_limit_compaction_sse(value, image_limit_error(value), metadata) is None


def test_policy_rejection_is_not_reclassified_as_context_pressure():
    policy = {"error": {
        "code": "invalid_request_body",
        "message": "Image processing blocked due to content policy violation.",
    }}
    snapshot = copy.deepcopy(policy)
    assert image_limit_compaction_sse(payload(), policy, {"request_kind": "compaction"}) is None
    assert policy == snapshot


class FakeResponse:
    def __init__(self, code=200, body=None):
        self.status_code = code
        self.ok = 200 <= code < 300
        self.body = body if body is not None else {"status": "completed", "output": [], "usage": {}}
        self.text = json.dumps(self.body)

    def json(self):
        return self.body

    def close(self):
        pass


@pytest.fixture
def route():
    with ExitStack() as stack:
        for key, value in {
            "auto_remove_encrypted_content_on_parse_error": False,
            "enable_portable_schema_recovery": True,
            "enable_responses_early_failure_retry": False,
            "sse_keepalive_interval": 0,
            "max_connection_retries": 0,
            "save_request_to_file": False,
        }.items():
            stack.enter_context(mock.patch.object(state, key, value))
        stack.enter_context(mock.patch.object(cache, "cache", OrderedDict()))
        memory = PortableSchemaMemory(path=None)
        stack.enter_context(mock.patch("ghc_api.portable_schema.portable_schema_memory", memory))
        stack.enter_context(mock.patch("ghc_api.routes.openai.ensure_copilot_token"))
        stack.enter_context(mock.patch("ghc_api.routes.openai.supports_responses_api", return_value=True))
        stack.enter_context(mock.patch("ghc_api.routes.openai.translate_model_name", side_effect=lambda value: value))
        stack.enter_context(mock.patch("ghc_api.routes.openai.get_copilot_base_url", return_value="https://upstream.invalid"))
        stack.enter_context(mock.patch("ghc_api.routes.openai.log_error_request"))
        headers = stack.enter_context(mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={}))
        post = stack.enter_context(mock.patch("ghc_api.routes.openai.requests.post", return_value=FakeResponse()))
        yield create_app().test_client(), post, headers, memory


@pytest.mark.parametrize("stream,kind,expected_http", [
    (False, "turn", 400),
    (True, "turn", 400),
    (False, "compaction", 400),
    (True, "compaction", 200),
])
def test_guard_is_local_and_uses_sse_only_inside_compaction(route, stream, kind, expected_http):
    client, post, headers, memory = route
    value = payload(stream=stream)
    original = copy.deepcopy(value)
    with mock.patch.object(memory, "remember", wraps=memory.remember) as remember:
        response = client.post(
            "/v1/responses", json=value,
            headers={"X-Codex-Turn-Metadata": json.dumps({"request_kind": kind})},
        )
    assert response.status_code == expected_http
    if expected_http == 200:
        assert response.mimetype == "text/event-stream"
        failure = events(response.get_data(as_text=True))[-1]
        assert failure["type"] == "response.failed"
        assert failure["response"]["error"]["code"] == "context_length_exceeded"
    else:
        assert response.mimetype == "application/json"
        assert "52 images (maximum 50)" in response.json["error"]["message"]
        assert response.json["error"]["code"] == "context_length_exceeded"
    post.assert_not_called()
    headers.assert_not_called()
    remember.assert_not_called()
    assert value == original
    record = cache.get_recent_requests(1)[0]
    assert record["status_code"] == 400
    assert record["original_request_body"] == original


@pytest.mark.parametrize("output_kind", ["function_call_output", "custom_tool_call_output"])
def test_exactly_50_typed_images_enable_vision_and_forward_unchanged(route, output_kind):
    client, post, headers, _ = route
    value = payload(images=50, stream=False, output_kind=output_kind)
    response = client.post("/v1/responses", json=value)
    assert response.status_code == 200
    headers.assert_called_once_with(True)
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["input"] == value["input"]


def test_plain_text_keeps_vision_disabled(route):
    client, post, headers, _ = route
    value = {"model": "gpt-test", "stream": False, "input": "Synthetic text-only request"}
    response = client.post("/v1/responses", json=value)
    assert response.status_code == 200
    headers.assert_called_once_with(False)
    assert post.call_args.kwargs["json"]["input"] == value["input"]


@pytest.mark.parametrize("stream,kind", [(False, "turn"), (True, "compaction")])
def test_under_limit_policy_error_is_preserved_without_cleanup(route, stream, kind):
    client, post, _, _ = route
    policy = {"error": {
        "code": "invalid_request_body",
        "message": "Image processing blocked due to content policy violation.",
    }}
    post.return_value = FakeResponse(400, policy)
    value = payload(images=3, stream=stream)
    with mock.patch(
        "ghc_api.encrypted_recovery.remove_encrypted_content_items",
        side_effect=AssertionError("Policy errors must not trigger history cleanup"),
    ):
        response = client.post(
            "/v1/responses", json=value,
            headers={"X-Codex-Turn-Metadata": json.dumps({"request_kind": kind})},
        )
    assert response.status_code == 400
    assert response.json == policy
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["input"] == value["input"]
