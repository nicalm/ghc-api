"""Stateless Streamable HTTP MCP: JSON responses, no server-initiated stream."""

import base64
import hashlib
import ipaddress
import json
import logging
import threading
from urllib.parse import urlsplit

from flask import Blueprint, g, jsonify, request

from .. import __version__
from ..counters import counters
from ..media.assets import AssetStore
from ..media.previews import ImageToolError, make_preview
from ..state import state


image_tools_bp = Blueprint('image_tools', __name__)
MCP_PATH = '/mcp/images'
MAX_BODY_BYTES = 64 * 1024
VERSIONS = ('2025-03-26', '2025-06-18', '2025-11-25')
_SLOTS = threading.BoundedSemaphore(2)
_LOG = logging.getLogger(__name__)


def _object(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}


TOOLS = [
    {'name': 'register_image',
     'description': 'Register an existing local PNG/JPEG/WebP under the configured image directories. Returns image ID, dimensions, transparency metadata and SHA-256, never image bytes. Keeps the original unchanged.',
     'inputSchema': _object({'path': {'type': 'string', 'maxLength': 4096}}, ['path']),
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False}},
    {'name': 'describe_image',
     'description': 'Read metadata for an image ID owned by this caller, without loading a preview into conversation history.',
     'inputSchema': _object({'asset_id': {'type': 'string'}}, ['asset_id']),
     'annotations': {'readOnlyHint': True, 'idempotentHint': True, 'openWorldHint': False}},
    {'name': 'preview_images',
     'description': 'Inspect at most two byte-bounded JPEG previews or crops. Use one stable context_id for the current task; repeated reads count toward its cumulative budget. Save observations as text instead of reloading all images. JPEG previews are not proof of original transparency.',
     'inputSchema': _object({
         'context_id': {'type': 'string', 'pattern': '^[A-Za-z0-9_.:-]{1,128}$'},
         'items': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                   'items': _object({'asset_id': {'type': 'string'},
                                     'crop': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4},
                                     'background': {'type': 'string', 'enum': ['white', 'checkerboard']}}, ['asset_id'])},
     }, ['context_id', 'items']),
     'annotations': {'readOnlyHint': True, 'idempotentHint': False, 'openWorldHint': False}},
]


def _text(value):
    return {'type': 'text', 'text': json.dumps(value, ensure_ascii=False, separators=(',', ':'))}


def _exact_keys(value, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ImageToolError('invalid_arguments', 'Arguments do not match the advertised tool schema.')


def _call_tool(name, arguments):
    store = AssetStore(state.image_tools)
    owner = getattr(g, 'user_id', 'anonymous')
    if name == 'register_image':
        _exact_keys(arguments, ['path'], ['path'])
        return {'content': [_text(store.register(owner, arguments['path']))], 'isError': False}
    if name == 'describe_image':
        _exact_keys(arguments, ['asset_id'], ['asset_id'])
        return {'content': [_text(store.describe(owner, arguments['asset_id']))], 'isError': False}
    if name == 'preview_images':
        _exact_keys(arguments, ['context_id', 'items'], ['context_id', 'items'])
        items = arguments['items']
        if not isinstance(items, list) or not 1 <= len(items) <= 2:
            raise ImageToolError('preview_count_exceeded', 'Request one or two previews at a time.')
        previews, total = [], 0
        for item in items:
            _exact_keys(item, ['asset_id', 'crop', 'background'], ['asset_id'])
            data, metadata = make_preview(store.read(owner, item['asset_id']), state.image_tools,
                                          item.get('crop'), item.get('background', 'white'))
            total += len(data)
            previews.append((data, {**metadata, 'asset_id': item['asset_id'],
                                    'preview_sha256': hashlib.sha256(data).hexdigest()}))
        budget = store.consume(owner, arguments['context_id'], total)
        content = [_text({'budget': budget, 'previews': [meta for _, meta in previews],
                          'note': 'JPEG derivatives only. Originals and alpha channels remain unchanged.'})]
        for data, _ in previews:
            content.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(data).decode('ascii')})
        return {'content': content, 'isError': False}
    raise ImageToolError('unknown_tool', 'Unknown image tool.')


def _rpc_error(rpc_id, code, message, status=200):
    return jsonify({'jsonrpc': '2.0', 'id': rpc_id, 'error': {'code': code, 'message': message}}), status


def _local_without_auth():
    try:
        host = urlsplit(request.host_url).hostname
        address = ipaddress.ip_address(request.remote_addr or '')
        return address.is_loopback and host in ('127.0.0.1', 'localhost', '::1')
    except ValueError:
        return False


@image_tools_bp.route(MCP_PATH, methods=['POST', 'GET', 'DELETE', 'OPTIONS'])
def image_tools():
    if not state.image_tools.enabled:
        return jsonify({'error': 'image_tools_disabled'}), 404
    # No CORS. Without auth, neither remote callers nor rebinding Host headers
    # may expose files. Authenticated calls use the app's existing auth gate.
    if not state.enable_auth and not _local_without_auth():
        return jsonify({'error': 'image_tools_require_localhost_or_auth'}), 403
    origin = request.headers.get('Origin')
    if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
        return jsonify({'error': 'invalid_origin'}), 403
    if request.method != 'POST':
        return jsonify({'error': 'This stateless MCP endpoint supports POST; no standalone SSE stream or session deletion.'}), 405, {'Allow': 'POST'}
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        return jsonify({'error': 'image_tool_request_too_large'}), 413
    if not request.is_json:
        return _rpc_error(None, -32700, 'Content-Type must be application/json.', 415)
    accept = request.headers.get('Accept', '')
    if 'application/json' not in accept or 'text/event-stream' not in accept:
        return _rpc_error(None, -32600, 'Accept must include application/json and text/event-stream.', 406)
    protocol = request.headers.get('MCP-Protocol-Version')
    if protocol and protocol not in VERSIONS:
        return _rpc_error(None, -32600, 'Unsupported MCP-Protocol-Version.', 400)
    rpc = request.get_json(silent=True)
    if not isinstance(rpc, dict) or rpc.get('jsonrpc') != '2.0' or not isinstance(rpc.get('method'), str):
        return _rpc_error(None, -32600, 'Expected one JSON-RPC 2.0 request.', 400)
    rpc_id = rpc.get('id')
    if 'id' not in rpc:
        return '', 202
    if type(rpc_id) not in (int, str):
        return _rpc_error(None, -32600, 'Invalid request ID.', 400)
    method, params = rpc['method'], rpc.get('params', {})
    if not isinstance(params, dict):
        return _rpc_error(rpc_id, -32602, 'params must be an object.')
    if method == 'initialize':
        if (not isinstance(params.get('protocolVersion'), str)
                or not isinstance(params.get('capabilities'), dict)
                or not isinstance(params.get('clientInfo'), dict)):
            return _rpc_error(rpc_id, -32602, 'Missing initialization parameters.')
        result = {'protocolVersion': params['protocolVersion'] if params['protocolVersion'] in VERSIONS else '2025-06-18',
                  'capabilities': {'tools': {'listChanged': False}},
                  'serverInfo': {'name': 'ghc-api-images', 'version': __version__},
                  'instructions': 'Register local images, then inspect only necessary small previews. Use one stable context_id per Codex task and record observations as text. Never bypass preview limits by reading the whole image set with another tool. Originals are retained outside the conversation.'}
    elif method == 'ping':
        result = {}
    elif method == 'tools/list':
        result = {'tools': TOOLS}
    elif method == 'tools/call':
        if not isinstance(params.get('name'), str) or not isinstance(params.get('arguments', {}), dict):
            return _rpc_error(rpc_id, -32602, 'Invalid tool call.')
        if not _SLOTS.acquire(blocking=False):
            result = {'content': [_text({'code': 'image_tools_busy', 'message': 'Image workers are busy; retry this preview later.'})], 'isError': True}
        else:
            counters.incr('image_tools.inflight')
            try:
                result = _call_tool(params['name'], params.get('arguments', {}))
            except ImageToolError as exc:
                result = {'content': [_text({'code': exc.code, 'message': str(exc)})], 'isError': True}
            except Exception as exc:
                _LOG.error('Image tool failed (%s)', type(exc).__name__)
                result = {'content': [_text({'code': 'image_tool_failed', 'message': 'Local image tool failed; no original was changed.'})], 'isError': True}
            finally:
                counters.incr('image_tools.inflight', -1)
                _SLOTS.release()
    else:
        return _rpc_error(rpc_id, -32601, 'Method not found.')
    return jsonify({'jsonrpc': '2.0', 'id': rpc_id, 'result': result})
