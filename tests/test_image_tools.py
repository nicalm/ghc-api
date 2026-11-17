"""Image/output limits, caller isolation, source preservation and MCP transport."""

import base64
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest import mock

import pytest
from PIL import Image

from ghc_api.app import create_app
from ghc_api.auth import AuthResult
from ghc_api.media.assets import AssetStore
from ghc_api.media.config import ImageToolsConfig, parse_image_config, parse_budget_config
from ghc_api.media.previews import ImageToolError, make_preview
from ghc_api.state import state


@pytest.fixture
def images(tmp_path, monkeypatch):
    source = tmp_path / 'sources'
    source.mkdir()
    config = ImageToolsConfig(enabled=True, allowed_roots=(str(source),), data_dir=str(tmp_path / 'snapshots'))
    monkeypatch.setattr(state, 'image_tools', config)
    monkeypatch.setattr(state, 'enable_auth', False)
    picture = Image.new('RGBA', (1200, 900), (40, 100, 180, 110))
    path = source / 'synthetic.png'
    picture.save(path)
    return config, path, create_app().test_client()


def rpc(client, method, params=None, headers=None, **options):
    return client.post('/mcp/images', json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {}},
                       headers={'Accept': 'application/json, text/event-stream', **(headers or {})}, **options)


def call(client, name, arguments):
    response = rpc(client, 'tools/call', {'name': name, 'arguments': arguments})
    assert response.status_code == 200
    result = response.json['result']
    return result, json.loads(result['content'][0]['text'])


def register(client, path):
    result, metadata = call(client, 'register_image', {'path': str(path)})
    assert not result['isError'], metadata
    assert len(result['content']) == 1
    return metadata


def test_initialize_negotiate_discover_notification_and_no_stream(images):
    _, _, client = images
    response = rpc(client, 'initialize', {'protocolVersion': 'future-version', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}})
    assert response.json['result']['protocolVersion'] == '2025-06-18'
    assert response.json['result']['capabilities'] == {'tools': {'listChanged': False}}
    listed = rpc(client, 'tools/list').json['result']['tools']
    assert {tool['name'] for tool in listed} == {'register_image', 'describe_image', 'preview_images'}
    assert client.post('/mcp/images', json={'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                       headers={'Accept': 'application/json, text/event-stream'}).status_code == 202
    for method in ('GET', 'DELETE'):
        response = client.open('/mcp/images', method=method)
        assert response.status_code == 405 and response.headers['Allow'] == 'POST'
    assert rpc(client, 'unknown').json['error']['code'] == -32601


@pytest.mark.parametrize('headers,status', [
    ({'Origin': 'https://untrusted.invalid'}, 403),
    ({'Host': 'rebind.invalid'}, 403),
    ({'MCP-Protocol-Version': 'unknown'}, 400),
    ({'Accept': 'application/json'}, 406),
])
def test_transport_rejects_rebinding_origin_and_invalid_protocol(images, headers, status):
    assert rpc(images[2], 'tools/list', headers=headers).status_code == status


def test_remote_noauth_and_disabled_feature_are_denied(images, monkeypatch):
    config, _, client = images
    assert rpc(client, 'tools/list', environ_overrides={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403
    monkeypatch.setattr(state, 'image_tools', replace(config, enabled=False))
    assert rpc(client, 'tools/list').status_code == 404


def test_auth_applies_to_image_mcp_and_assets_belong_to_caller(images, monkeypatch):
    _, path, client = images
    monkeypatch.setattr(state, 'enable_auth', True)
    with mock.patch('ghc_api.app.require_auth', return_value=AuthResult(None, 'missing_token', 'missing', 401)):
        assert rpc(client, 'tools/list').status_code == 401
    with mock.patch('ghc_api.app.require_auth', return_value=AuthResult('alice')):
        asset = register(client, path)
    with mock.patch('ghc_api.app.require_auth', return_value=AuthResult('bob')):
        result, error = call(client, 'describe_image', {'asset_id': asset['asset_id']})
        assert result['isError'] and error['code'] == 'asset_not_found'


def test_registration_snapshot_and_preview_preserve_original_and_alpha(images):
    config, path, client = images
    original = path.read_bytes()
    asset = register(client, path)
    assert asset['has_transparency'] and asset['width'] == 1200
    assert asset['original_sha256'] == hashlib.sha256(original).hexdigest()
    assert register(client, path)['asset_id'] == asset['asset_id']
    result, metadata = call(client, 'preview_images', {'context_id': 'task-1', 'items': [
        {'asset_id': asset['asset_id'], 'background': 'checkerboard'},
        {'asset_id': asset['asset_id'], 'crop': [10, 20, 200, 300]},
    ]})
    assert not result['isError']
    output_bytes = 0
    for block, details in zip(result['content'][1:], metadata['previews']):
        data = base64.b64decode(block['data'])
        output_bytes += len(data)
        assert len(data) <= config.preview_max_bytes
        assert block['mimeType'] == 'image/jpeg'
        assert hashlib.sha256(data).hexdigest() == details['preview_sha256']
        with Image.open(io.BytesIO(data)) as preview:
            assert max(preview.size) <= config.preview_max_edge
            assert 'exif' not in preview.info
    assert metadata['budget']['used_bytes'] == output_bytes
    assert path.read_bytes() == original
    # Once registered, later source changes do not silently change its image ID.
    path.write_bytes(b'changed source')
    assert AssetStore(config).read('anonymous', asset['asset_id']) == original


def test_noisy_image_is_really_byte_bounded(images):
    config, path, _ = images
    Image.frombytes('RGB', (1600, 1600), os.urandom(1600 * 1600 * 3)).save(path)
    data = path.read_bytes()
    assert len(data) > config.preview_max_bytes
    preview, metadata = make_preview(data, config)
    assert len(preview) <= config.preview_max_bytes
    assert metadata['width'] >= 320


@pytest.mark.parametrize('crop', [[-1, 0, 10, 10], [0, 0, 9999, 10], [10, 0, 2, 4], [True, 0, 2, 4], 'bad'])
def test_invalid_crop_never_falls_back_to_full_original(images, crop):
    config, path, _ = images
    with pytest.raises(ImageToolError, match='crop'):
        make_preview(path.read_bytes(), config, crop)


def test_readable_preview_failure_does_not_return_original(images):
    config, path, _ = images
    data = io.BytesIO()
    Image.frombytes('RGB', (600, 600), os.urandom(600 * 600 * 3)).save(data, format='PNG')
    with pytest.raises(ImageToolError) as exc:
        make_preview(data.getvalue(), replace(config, preview_max_bytes=10))
    assert exc.value.code == 'preview_budget_exceeded'


@pytest.mark.parametrize('what', ['relative', 'outside', 'traversal', 'corrupt', 'large', 'pixels', 'animated'])
def test_source_limits(images, tmp_path, what):
    config, path, _ = images
    source = str(path)
    if what == 'relative':
        source = path.name
    elif what in ('outside', 'traversal'):
        outside = tmp_path / 'outside.png'
        outside.write_bytes(path.read_bytes())
        source = str(outside) if what == 'outside' else str(path.parent / '..' / outside.name)
    elif what == 'corrupt':
        path.write_bytes(b'not an image')
    elif what == 'large':
        config = replace(config, max_source_bytes=10)
    elif what == 'pixels':
        config = replace(config, max_pixels=100)
    elif what == 'animated':
        with Image.open(path) as first:
            other = Image.new('RGBA', first.size, 'red')
            first.save(path, format='PNG', save_all=True, append_images=[other])
    with pytest.raises(ImageToolError):
        AssetStore(config).register('alice', source)


def test_symlink_cannot_escape_allowed_root(images, tmp_path):
    config, path, _ = images
    outside = tmp_path / 'outside.png'
    outside.write_bytes(path.read_bytes())
    link = path.parent / 'link.png'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('Creating symlinks requires platform permission')
    with pytest.raises(ImageToolError) as exc:
        AssetStore(config).register('alice', str(link))
    assert exc.value.code == 'path_not_allowed'


def test_repeated_reads_and_concurrent_calls_share_durable_budget(images):
    config, path, _ = images
    store = AssetStore(replace(config, context_budget_bytes=200_000))
    with ThreadPoolExecutor(4) as executor:
        def consume(_):
            try:
                return store.consume('alice', 'same-task', 100_000)['used_bytes']
            except ImageToolError as exc:
                return exc.code
        results = list(executor.map(consume, range(4)))
    assert sorted(x for x in results if isinstance(x, int)) == [100_000, 200_000]
    assert results.count('context_preview_budget_exceeded') == 2
    with pytest.raises(ImageToolError):
        AssetStore(store.config).consume('alice', 'same-task', 1)
    assert store.consume('bob', 'same-task', 1)['used_bytes'] == 1


def test_expiry_only_removes_snapshots_and_preserves_original(images):
    config, path, _ = images
    store = AssetStore(config)
    asset = store.register('alice', str(path))
    with store.connection() as db:
        db.execute('UPDATE assets SET created=0')
        db.commit()
    store.consume('alice', 'cleanup', 1)
    assert not store._path('alice', asset['asset_id']).exists()
    assert path.is_file()
    with pytest.raises(ImageToolError):
        store.describe('alice', asset['asset_id'])


def test_snapshot_capacity_and_tamper_are_explicit(images):
    config, path, _ = images
    with pytest.raises(ImageToolError) as exc:
        AssetStore(replace(config, store_max_bytes=1)).register('alice', str(path))
    assert exc.value.code == 'asset_store_full'
    store = AssetStore(config)
    asset = store.register('alice', str(path))
    store._path('alice', asset['asset_id']).write_bytes(b'tampered')
    with pytest.raises(ImageToolError) as exc:
        store.read('alice', asset['asset_id'])
    assert exc.value.code == 'asset_changed'
    assert store.register('alice', str(path))['asset_id'] == asset['asset_id']
    assert store.read('alice', asset['asset_id']) == path.read_bytes()


def test_tool_count_and_request_body_limits(images):
    _, path, client = images
    asset = register(client, path)
    result, error = call(client, 'preview_images', {'context_id': 'test', 'items': [{'asset_id': asset['asset_id']}] * 3})
    assert result['isError'] and error['code'] == 'preview_count_exceeded'
    assert rpc(client, 'ping', {'huge': 'a' * 65536}).status_code == 413
    result, error = call(client, 'register_image', {'path': str(path), 'unexpected': True})
    assert result['isError'] and error['code'] == 'invalid_arguments'


@pytest.mark.parametrize('raw', [{'enabled': 'true'}, {'enabled': True}, {'allowed_roots': ['relative']},
                                  {'preview_max_bytes': True}, {'unknown_setting': 1}])
def test_invalid_image_config_fails_closed(raw):
    with pytest.raises(ValueError):
        parse_image_config(raw)


@pytest.mark.parametrize('raw', [{'enabled': 1}, {'warn_bytes': 4000, 'max_bytes': 3000},
                                  {'max_bytes': -1}, {'unknown_setting': True}])
def test_invalid_budget_config_fails_closed(raw):
    with pytest.raises(ValueError):
        parse_budget_config(raw)
