"""Preserve real reasoning accounting without fabricating upstream declarations."""

import json
import time
from unittest import mock

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.routes import openai as openai_routes
from ghc_api.sse import base as base_module
from ghc_api.sse import AnthropicDirectStreamHandler, OpenAIResponsesStreamHandler
from ghc_api.sse.proxy_responses import ProxyResponsesStreamHandler
from ghc_api.state import state


MODEL = "gpt-5"
CODEX_HEADERS = {"X-Codex-Turn-Metadata": '{"request_kind":"sampling"}'}
USAGE = {"input_tokens": 351063, "output_tokens": 254, "total_tokens": 351317}


def event(kind, **fields):
    return ("data: " + json.dumps({"type": kind, **fields}, separators=(",", ":"))).encode()


def completed_lines():
    return [
        b"event: response.completed",
        event("response.completed", response={"status": "completed", "usage": USAGE}),
        b"data: [DONE]",
    ]


class Upstream:
    def __init__(self, headers=None, lines=None, status=200, text="", body_delay=0):
        self.headers = CaseInsensitiveDict(headers or {})
        self.lines = completed_lines() if lines is None else lines
        self.status_code = status
        self.ok = status < 400
        self.error_text = text
        self.body_delay = body_delay
        self.text_accessed = False
        self.closed = False

    @property
    def text(self):
        self.text_accessed = True
        return self.error_text

    def json(self):
        return json.loads(self.error_text)

    def iter_lines(self):
        if self.body_delay:
            time.sleep(self.body_delay)
        yield from self.lines

    def close(self):
        self.closed = True


class Completed:
    def __init__(self, response):
        self.response = response

    def get(self, timeout=None):
        return self.response


@pytest.fixture
def isolated_routes(monkeypatch):
    cache = RequestCache()
    for name, value in {
        "models": {"data": [{"id": MODEL, "supported_endpoints": ["/responses"]}]},
        "enable_responses_early_failure_retry": False,
        "auto_remove_encrypted_content_on_parse_error": False,
        "max_connection_retries": 1,
        "sse_keepalive_interval": 0.01,
        "responses_pre_header_grace": 0.001,
    }.items():
        monkeypatch.setattr(state, name, value)
    monkeypatch.setattr(openai_routes, "cache", cache)
    monkeypatch.setattr(base_module, "cache", cache)
    monkeypatch.setattr(openai_routes, "ensure_copilot_token", lambda: None)
    monkeypatch.setattr(openai_routes, "get_copilot_headers", lambda *_: {})
    monkeypatch.setattr(openai_routes, "get_copilot_base_url", lambda: "https://upstream.test")
    return create_app(), cache


@pytest.mark.parametrize("handler_type", [OpenAIResponsesStreamHandler, ProxyResponsesStreamHandler])
@pytest.mark.parametrize("upstream_headers,expected", [
    ({"x-reasoning-included": "true"}, "true"),
    ({"X-Reasoning-Included": "TRUE"}, "TRUE"),
    ({}, None),
    ({"x-reasoning-included": "false"}, None),
    ({"x-reasoning-included": ""}, None),
    ({"x-reasoning-included": "unknown"}, None),
])
def test_response_handlers_forward_only_a_real_positive_declaration(
    isolated_routes, handler_type, upstream_headers, expected
):
    app, _ = isolated_routes
    upstream = Upstream({**upstream_headers, "Set-Cookie": "synthetic", "Authorization": "synthetic"})
    options = dict(
        response=upstream, request_id="headers", request_size=1, start_time=time.time(),
        original_model=MODEL, translated_model=MODEL, request_body_for_cache={},
    )
    if handler_type is ProxyResponsesStreamHandler:
        options.update(endpoint="/v1/responses", profile_name="test", public_model=MODEL, rewrite_model=False)
    with app.test_request_context("/v1/responses"):
        response = handler_type(**options).stream()
        assert response.headers.get("x-reasoning-included") == expected
        assert "Set-Cookie" not in response.headers
        assert "Authorization" not in response.headers
        body = response.get_data()
    assert event("response.completed", response={"status": "completed", "usage": USAGE}) in body
    assert not upstream.text_accessed
    assert upstream.closed


def test_messages_translation_does_not_inherit_responses_accounting(isolated_routes):
    app, _ = isolated_routes
    upstream = Upstream({"x-reasoning-included": "true"}, lines=[])
    with app.test_request_context("/v1/messages"):
        response = AnthropicDirectStreamHandler(
            response=upstream, request_id="messages", request_size=1, start_time=time.time(),
            original_model="claude-test", translated_model="claude-test", request_body_for_cache={},
        ).stream()
        assert "x-reasoning-included" not in response.headers
        response.get_data()


def test_fast_generic_stream_also_preserves_the_header(isolated_routes):
    app, _ = isolated_routes
    upstream = Upstream({"x-reasoning-included": "true"})
    with mock.patch.object(openai_routes, "_start_responses_post", return_value=Completed(upstream)):
        with app.test_client() as client:
            response = client.post("/v1/responses", json={"model": MODEL, "stream": True, "input": []})
            assert response.headers.get("x-reasoning-included") == "true"
            assert b"response.completed" in response.get_data()


@pytest.mark.parametrize("declared", [True, False])
def test_codex_delayed_headers_wait_for_truth_and_still_keep_the_body_alive(isolated_routes, declared):
    app, _ = isolated_routes
    upstream = Upstream({"x-reasoning-included": "true"} if declared else {}, body_delay=0.04)
    headers_received = False

    def delayed_post(*args, **kwargs):
        nonlocal headers_received
        time.sleep(0.02)  # Deliberately exceeds the old 0.001-second pre-header grace.
        headers_received = True
        return upstream

    with mock.patch.object(openai_routes.requests, "post", side_effect=delayed_post), \
            mock.patch.object(openai_routes, "_start_responses_post", side_effect=AssertionError("speculative headers")):
        with app.test_client() as client:
            response = client.post("/v1/responses", headers=CODEX_HEADERS,
                                   json={"model": MODEL, "stream": True, "input": []}, buffered=False)
            assert headers_received
            assert response.headers.get("x-reasoning-included") == ("true" if declared else None)
            chunks = iter(response.response)
            first = next(chunks)
            assert first == b": keepalive\n\n"
            remaining = b"".join(chunks)
            assert b"response.completed" in remaining
    assert not upstream.text_accessed
    assert upstream.closed


def test_codex_pre_header_timeout_returns_a_bounded_error_without_a_false_flag(isolated_routes):
    app, _ = isolated_routes
    state.max_connection_retries = 0
    with mock.patch.object(openai_routes.requests, "post", side_effect=requests.ReadTimeout("synthetic timeout")), \
            mock.patch.object(openai_routes, "log_connection_retry"):
        with app.test_client() as client:
            response = client.post("/v1/responses", headers=CODEX_HEADERS,
                                   json={"model": MODEL, "stream": True, "input": []})
            assert response.status_code == 504
            assert "x-reasoning-included" not in response.headers


def test_codex_http_error_is_not_mislabeled_as_successful_reasoning_accounting(isolated_routes):
    app, _ = isolated_routes
    upstream = Upstream({"x-reasoning-included": "true"}, status=429,
                        text=json.dumps({"error": {"message": "synthetic rate limit"}}))
    with mock.patch.object(openai_routes.requests, "post", return_value=upstream), \
            mock.patch.object(openai_routes, "log_error_request"):
        with app.test_client() as client:
            response = client.post("/v1/responses", headers=CODEX_HEADERS,
                                   json={"model": MODEL, "stream": True, "input": []})
            assert response.status_code == 429
            assert "x-reasoning-included" not in response.headers


@pytest.mark.parametrize("first_declared,second_declared", [(True, True), (False, False), (True, False), (False, True)])
def test_early_retry_cannot_silently_change_the_committed_accounting_contract(
    isolated_routes, first_declared, second_declared
):
    app, cache = isolated_routes
    state.enable_responses_early_failure_retry = True
    first = Upstream({"x-reasoning-included": "true"} if first_declared else {}, lines=[
        event("response.created"), event("response.failed", response={"error": None}),
    ])
    second = Upstream({"x-reasoning-included": "true"} if second_declared else {})
    with mock.patch.object(openai_routes.requests, "post", side_effect=[first, second]) as post:
        with app.test_client() as client:
            response = client.post("/v1/responses", headers=CODEX_HEADERS,
                                   json={"model": MODEL, "stream": True, "input": []})
            body = response.get_data()
    assert post.call_count == 2
    assert response.headers.get("x-reasoning-included") == ("true" if first_declared else None)
    if first_declared == second_declared:
        assert b"response.completed" in body
        assert b"response.failed" not in body
        assert next(iter(cache.cache.values()))["input_tokens"] == USAGE["input_tokens"]
    else:
        assert b"response.failed" in body
        assert b"response.completed" not in body
    assert first.closed and second.closed


def test_codex_http_recovery_uses_the_successful_attempt_header(isolated_routes):
    app, _ = isolated_routes
    state.auto_remove_encrypted_content_on_parse_error = True
    failed = Upstream(status=400, text=json.dumps({"error": {
        "message": "Encrypted function output content could not be decrypted or decoded.",
        "code": "invalid_request_body",
    }}))
    succeeded = Upstream({"x-reasoning-included": "true"})
    payload = {"model": MODEL, "stream": True, "input": [
        {"type": "message", "role": "user", "content": "synthetic retained instruction"},
        {"type": "encrypted_content", "encrypted_content": "synthetic-invalid"},
    ]}
    with mock.patch.object(openai_routes.requests, "post", side_effect=[failed, succeeded]) as post, \
            mock.patch.object(openai_routes, "log_error_request"):
        with app.test_client() as client:
            response = client.post("/v1/responses", headers=CODEX_HEADERS, json=payload)
            assert response.headers.get("x-reasoning-included") == "true"
            assert b"response.completed" in response.get_data()
    assert post.call_count == 2
    assert post.call_args.kwargs["json"]["input"][0] == payload["input"][0]
    assert failed.closed and succeeded.closed
