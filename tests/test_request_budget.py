"""Exact wire bytes, permanent failure recovery and preservation of Responses."""

import copy
import json
from dataclasses import replace
from unittest import mock

import pytest
import requests

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.counters import counters
from ghc_api.media import budget as budget_module
from ghc_api.media.budget import ResponsesBudget, RequestBudgetError, image_statistics
from ghc_api.media.config import RequestBudgetConfig
from ghc_api.routes import openai as routes
from ghc_api.sse import base as sse_base
from ghc_api.state import state


HEADERS = {'X-Codex-Turn-Metadata': '{"request_kind":"sampling"}'}


def upstream(status=200, text=None):
    response = requests.Response()
    response.status_code = status
    response._content = (text or json.dumps({'status': 'completed', 'output': [], 'usage': {}})).encode()
    response.headers['content-type'] = 'application/json'
    return response


@pytest.fixture
def guarded(monkeypatch):
    for name, value in {
        'enable_auth': False, 'enable_responses_early_failure_retry': False,
        'auto_remove_encrypted_content_on_parse_error': False,
        'max_connection_retries': 0, 'sse_keepalive_interval': 0,
        'models': {'data': [{'id': 'gpt-5', 'supported_endpoints': ['/responses']}]},
        'codex_request_budget': RequestBudgetConfig(enabled=True, warn_bytes=1024, max_bytes=2048),
    }.items():
        monkeypatch.setattr(state, name, value)
    monkeypatch.setattr(routes, 'ensure_copilot_token', lambda: None)
    monkeypatch.setattr(routes, 'get_copilot_headers', lambda *_: {})
    monkeypatch.setattr(routes, 'get_copilot_base_url', lambda: 'https://synthetic.invalid')
    cache = RequestCache()
    monkeypatch.setattr(routes, 'cache', cache)
    monkeypatch.setattr(sse_base, 'cache', cache)
    with budget_module._LOCK:
        budget_module._REJECTED.clear()
    return create_app().test_client(), cache


def payload(text='synthetic'):
    return {'model': 'gpt-5', 'stream': False, 'prompt_cache_key': 'task-1',
            'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}]}


def test_exact_wire_body_handles_unicode_and_preserves_every_field(guarded):
    client, _ = guarded
    value = payload('中文与 emoji 🦊')
    value['input'].extend([
        {'type': 'reasoning', 'encrypted_content': 'opaque-synthetic'},
        {'type': 'function_call', 'call_id': 'call_1', 'name': 'test', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'call_1', 'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA'}]},
    ])
    before = copy.deepcopy(value)
    with mock.patch.object(routes.requests, 'post', return_value=upstream()) as post:
        response = client.post('/v1/responses', headers=HEADERS, json=value)
    assert response.status_code == 200
    body = post.call_args.kwargs['data']
    assert isinstance(body, bytes) and 'json' not in post.call_args.kwargs
    assert json.loads(body) == before
    assert int(response.headers['X-GHC-Request-Bytes']) == len(body)
    assert int(post.call_args.kwargs['headers']['Content-Length']) == len(body)
    assert value == before


@pytest.mark.parametrize('path', ['/v1/responses', '/responses'])
def test_36mb_nested_image_request_is_blocked_without_upload(guarded, monkeypatch, path):
    client, _ = guarded
    monkeypatch.setattr(state, 'codex_request_budget', RequestBudgetConfig(enabled=True))
    value = payload()
    value['input'].append({'type': 'custom_tool_call_output', 'call_id': 'image', 'output': [
        {'type': 'input_image', 'image_url': 'data:image/png;base64,' + 'A' * (36 * 1024 * 1024)}]})
    with mock.patch.object(routes.requests, 'post') as post, mock.patch.object(routes, 'log_error_request') as log:
        response = client.post(path, headers=HEADERS, json=value)
    assert response.status_code == 400
    assert response.json['error']['code'] == 'ghc_request_budget_exceeded'
    assert response.json['error']['details']['image_count'] == 1
    assert response.json['error']['retryable'] is False
    assert len(response.data) < 2000
    post.assert_not_called()
    log.assert_not_called()


@pytest.mark.parametrize('native,enabled', [(False, True), (True, False)])
def test_other_clients_and_disabled_setting_use_existing_transport(guarded, monkeypatch, native, enabled):
    client, _ = guarded
    monkeypatch.setattr(state, 'codex_request_budget', replace(state.codex_request_budget, enabled=enabled))
    with mock.patch.object(routes.requests, 'post', return_value=upstream()) as post:
        response = client.post('/v1/responses', headers=HEADERS if native else {}, json=payload('a' * 3000))
    assert response.status_code == 200
    assert 'json' in post.call_args.kwargs
    assert 'X-GHC-Request-Bytes' not in response.headers


def test_warning_is_metadata_only_and_does_not_block(guarded, tmp_path, monkeypatch):
    client, _ = guarded
    monkeypatch.setenv('GHC_API_CONFIG_DIR', str(tmp_path))
    with mock.patch.object(routes.requests, 'post', return_value=upstream()):
        response = client.post('/v1/responses', headers=HEADERS, json=payload('private-synthetic' * 70))
    assert response.status_code == 200 and response.headers['X-GHC-Budget-Warning'] == 'true'
    audit = (tmp_path / 'request-budget.jl').read_text()
    assert 'private-synthetic' not in audit and 'task-1' not in audit
    assert 'request_budget_warning' in audit and 'request_bytes' in audit


def test_only_identical_413_payload_is_suppressed(guarded):
    client, cache = guarded
    with mock.patch.object(routes.requests, 'post', side_effect=[upstream(413, '{"error":{"message":"failed to parse request"}}'), upstream()]) as post, \
            mock.patch.object(routes, 'log_error_request') as log:
        first = client.post('/v1/responses', headers=HEADERS, json=payload())
        second = client.post('/v1/responses', headers=HEADERS, json=payload())
        corrected = client.post('/v1/responses', headers=HEADERS, json=payload('corrected'))
    assert first.status_code == 413 and second.status_code == 400 and corrected.status_code == 200
    assert second.json['error']['code'] == 'ghc_request_previous_413'
    assert post.call_count == 2
    log.assert_not_called()
    failure = next(row for row in cache.cache.values() if row['status_code'] == 413)
    assert failure['request_body'] == {'omitted': 'oversized request'}


def test_boundary_bytes_and_failed_request_memory_are_scoped(guarded):
    config = RequestBudgetConfig(enabled=True, warn_bytes=1000, max_bytes=2000)
    value = payload()
    expected = requests.Request('POST', 'https://synthetic.invalid', json=value).prepare().body
    exact = replace(config, warn_bytes=len(expected), max_bytes=len(expected))
    with mock.patch.object(budget_module.requests, 'post', return_value=upstream()) as post:
        guard = ResponsesBudget(exact, 'alice', 'task', 'endpoint', 'req')
        guard.post('https://synthetic.invalid', {}, value, False)
        assert len(post.call_args.kwargs['data']) == len(expected)
        with pytest.raises(RequestBudgetError):
            ResponsesBudget(replace(exact, max_bytes=len(expected)-1), 'alice', 'task', 'endpoint', 'req').post('https://synthetic.invalid', {}, value, False)
    with mock.patch.object(budget_module.requests, 'post', side_effect=[upstream(413), upstream(), upstream()]) as post:
        ResponsesBudget(config, 'alice', 'one', 'endpoint', 'r').post('https://synthetic.invalid', {}, value, False)
        ResponsesBudget(config, 'bob', 'one', 'endpoint', 'r').post('https://synthetic.invalid', {}, value, False)
        ResponsesBudget(config, 'alice', 'two', 'endpoint', 'r').post('https://synthetic.invalid', {}, value, False)
    assert post.call_count == 3


def test_image_metrics_do_not_download_urls_or_scan_quoted_text():
    value = {'input': [
        {'type': 'input_image', 'image_url': 'https://private.invalid/image'},
        {'type': 'image', 'source': {'type': 'base64', 'data': 'AAAA'}},
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,BBBB'}},
        {'type': 'input_text', 'text': 'data:image/png;base64,CCCC'},
    ]}
    with mock.patch.object(requests, 'get', side_effect=AssertionError('unexpected fetch')):
        result = image_statistics(value)
    assert result['image_count'] == 3
    assert result['inline_image_bytes'] == 4 + len('data:image/jpeg;base64,BBBB')


def test_413_memory_tolerates_key_order_but_not_changed_metadata(guarded):
    value = payload()
    value['client_metadata'] = {'turn': 'one', 'retry': 0}
    reordered = dict(reversed(list(value.items())))
    reordered['client_metadata'] = {'retry': 0, 'turn': 'one'}
    config = RequestBudgetConfig(enabled=True)
    guard = ResponsesBudget(config, 'alice', 'task', 'endpoint', 'r')
    with mock.patch.object(budget_module.requests, 'post', side_effect=[upstream(413), upstream()]) as post:
        guard.post('https://synthetic.invalid', {}, value, False)
        with pytest.raises(RequestBudgetError) as exc:
            guard.post('https://synthetic.invalid', {}, reordered, False)
        assert exc.value.code == 'ghc_request_previous_413'
        corrected = copy.deepcopy(value)
        corrected['client_metadata']['turn'] = 'two'
        guard.post('https://synthetic.invalid', {}, corrected, False)
    assert post.call_count == 2


def test_http_recovery_rechecks_wire_body_and_retains_accounting(guarded, monkeypatch):
    client, _ = guarded
    monkeypatch.setattr(state, 'auto_remove_encrypted_content_on_parse_error', True)
    value = payload()
    value['input'].append({'type': 'encrypted_content', 'encrypted_content': 'synthetic-invalid'})
    failed = upstream(400, json.dumps({'error': {'message': 'Encrypted function output content could not be decrypted or decoded.', 'code': 'invalid_request_body'}}))
    with mock.patch.object(routes.requests, 'post', side_effect=[failed, upstream()]) as post, mock.patch.object(routes, 'log_error_request'):
        response = client.post('/v1/responses', headers=HEADERS, json=value)
    assert response.status_code == 200 and post.call_count == 2
    bodies = [entry.kwargs['data'] for entry in post.call_args_list]
    assert len(bodies[1]) < len(bodies[0])
    assert json.loads(bodies[1])['input'] == value['input'][:1]
    assert int(response.headers['X-GHC-Request-Bytes']) == len(bodies[1])


def test_native_preheader_activity_returns_to_zero_even_after_errors(guarded):
    guard = ResponsesBudget(RequestBudgetConfig(enabled=True), 'alice', 'activity', 'endpoint', 'r')
    before = counters.snapshot().get('codex_budget.inflight', 0)
    def fail(*args, **kwargs):
        assert counters.snapshot()['codex_budget.inflight'] == before + 1
        raise requests.ReadTimeout('synthetic')
    with mock.patch.object(budget_module.requests, 'post', side_effect=fail), pytest.raises(requests.ReadTimeout):
        guard.post('https://synthetic.invalid', {}, payload(), False)
    assert counters.snapshot()['codex_budget.inflight'] == before
