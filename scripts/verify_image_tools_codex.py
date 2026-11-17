"""Synthetic-only MCP/Responses acceptance with an installed Codex CLI.

Run from the repository virtualenv with pytest, Pillow and the optional MCP SDK.
No GitHub/Azure endpoint, real session or credential is used.
"""

import argparse
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image
from werkzeug.serving import WSGIRequestHandler, make_server

def event(kind, **fields):
    return f'event: {kind}\ndata: {json.dumps({"type": kind, **fields})}\n\n'.encode()


def response_events(item, number):
    response = {'id': f'resp_fixture_{number}', 'object': 'response', 'created_at': 1,
                'model': 'gpt-6-astra', 'status': 'completed', 'output': [item],
                'usage': {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}}
    pieces = [event('response.created', response={**response, 'status': 'in_progress', 'output': []})]
    if item['type'] == 'function_call':
        pieces.extend([
            event('response.output_item.added', output_index=0, item={**item, 'status': 'in_progress', 'arguments': ''}),
            event('response.function_call_arguments.delta', output_index=0, item_id=item['id'], delta=item['arguments']),
            event('response.function_call_arguments.done', output_index=0, item_id=item['id'], arguments=item['arguments']),
        ])
    else:
        pieces.extend([
            event('response.output_item.added', output_index=0, item={**item, 'status': 'in_progress', 'content': []}),
            event('response.output_text.delta', output_index=0, content_index=0, item_id=item['id'], delta='OK'),
        ])
    pieces.extend([event('response.output_item.done', output_index=0, item=item),
                   event('response.completed', response=response), b'data: [DONE]\n\n'])
    return b''.join(pieces)


class QuietWSGI(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex', default=shutil.which('codex'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Source or installed site-packages directory to verify')
    args = parser.parse_args()
    assert args.codex, 'Codex CLI not found'
    sys.path.insert(0, str(args.package_root))
    from ghc_api.app import create_app
    from ghc_api.media.assets import AssetStore
    from ghc_api.media.budget import _REJECTED, image_statistics
    from ghc_api.media.config import ImageToolsConfig, RequestBudgetConfig
    from ghc_api.routes import openai as routes
    from ghc_api.state import state
    assert Path(routes.__file__).resolve().is_relative_to(args.package_root.resolve())
    observed = {'model_posts': [], 'mcp_calls': [], 'wire': []}
    mode = ['normal']
    with tempfile.TemporaryDirectory(prefix='ghc-image-acceptance-') as temporary:
        root = Path(temporary).resolve()
        source = root / 'source'
        source.mkdir()
        picture_path = source / 'synthetic.png'
        Image.new('RGBA', (1200, 900), (40, 100, 160, 128)).save(picture_path)
        source_hash = hashlib.sha256(picture_path.read_bytes()).hexdigest()
        os.environ['GHC_API_CONFIG_DIR'] = str(root / 'proxy-config')
        state.image_tools = ImageToolsConfig(enabled=True, allowed_roots=(str(source),), data_dir=str(root / 'assets'))
        state.codex_request_budget = RequestBudgetConfig(enabled=True)
        state.enable_auth = False
        state.disable_onedrive_access = True
        state.save_request_to_file = False
        state.models = {'data': [{'id': 'gpt-6-astra', 'supported_endpoints': ['/responses']}]}
        state.sse_keepalive_interval = 0.01
        state.enable_responses_early_failure_retry = False
        state.auto_remove_encrypted_content_on_parse_error = False
        state.max_connection_retries = 0
        state.upstream_read_timeout = 5
        asset = AssetStore(state.image_tools).register('anonymous', str(picture_path))

        class FakeUpstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                payload = json.loads(raw)
                names, functions = [], []
                for tool in payload.get('tools', []):
                    if not isinstance(tool, dict):
                        continue
                    names.append(tool.get('name', ''))
                    if tool.get('type') == 'namespace':
                        functions.extend((child.get('name', ''), tool.get('name')) for child in tool.get('tools', []))
                    else:
                        functions.append((tool.get('name', ''), None))
                observed['model_posts'].append({'mode': mode[0], 'bytes': len(raw),
                                                'images': image_statistics(payload), 'tools': names,
                                                'field_hashes': {key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12] for key, value in payload.items()},
                                                'client_metadata': payload.get('client_metadata')})
                count = sum(row['mode'] == mode[0] for row in observed['model_posts'])
                if mode[0] == 'upstream413':
                    body = b'{"error":{"code":"","message":"failed to parse request"}}'
                    self.send_response(413)
                    self.send_header('Content-Type', 'application/json')
                else:
                    if count in (1, 2):
                        suffix = 'register_image' if count == 1 else 'preview_images'
                        candidates = [(name, namespace) for name, namespace in functions if name.endswith(suffix)]
                        if not candidates:
                            self.send_error(400, 'Image MCP tools missing from Codex catalog')
                            return
                        arguments = {'path': str(picture_path)} if count == 1 else {
                            'context_id': 'synthetic-codex-task', 'items': [{'asset_id': asset['asset_id']}]}
                        item = {'id': f'fc_{count}', 'type': 'function_call', 'status': 'completed',
                                'call_id': f'call_{count}', 'name': candidates[0][0], 'arguments': json.dumps(arguments)}
                        if candidates[0][1]:
                            item['namespace'] = candidates[0][1]
                    else:
                        item = {'id': 'msg_ok', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                                'content': [{'type': 'output_text', 'text': 'OK', 'annotations': []}]}
                    body = response_events(item, count)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('x-reasoning-included', 'true')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), FakeUpstream)
        app = create_app()

        def capture(environ, start_response):
            path = environ['PATH_INFO']
            if path == '/mcp/images' and environ['REQUEST_METHOD'] == 'POST':
                raw = environ['wsgi.input'].read(int(environ.get('CONTENT_LENGTH') or 0))
                environ['wsgi.input'] = io.BytesIO(raw)
                rpc = json.loads(raw)
                observed['mcp_calls'].append({'method': rpc.get('method'), 'tool': rpc.get('params', {}).get('name')})

            def headers(status, entries, exc_info=None):
                if path.endswith('/responses'):
                    mapped = {k.lower(): v for k, v in entries}
                    observed['wire'].append({'mode': mode[0], 'status': status,
                                             'reasoning': mapped.get('x-reasoning-included'),
                                             'request_bytes': mapped.get('x-ghc-request-bytes')})
                return start_response(status, entries, exc_info)
            return app(environ, headers)

        proxy = make_server('127.0.0.1', 0, capture, threaded=True, request_handler=QuietWSGI)
        servers = [upstream, proxy]
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
        cli_results = []
        try:
            with mock.patch.object(routes, 'ensure_copilot_token'), \
                    mock.patch.object(routes, 'get_copilot_headers', return_value={}), \
                    mock.patch.object(routes, 'get_copilot_base_url', return_value=f'http://127.0.0.1:{upstream.server_port}'):
                for thread in threads:
                    thread.start()

                async def sdk_check():
                    from mcp import ClientSession
                    from mcp.client.streamable_http import streamable_http_client
                    async with streamable_http_client(f'http://127.0.0.1:{proxy.server_port}/mcp/images') as streams:
                        async with ClientSession(streams[0], streams[1], read_timeout_seconds=10) as session:
                            initialized = await session.initialize()
                            listed = await session.list_tools()
                            result = await session.call_tool('preview_images', {'context_id': 'sdk-task', 'items': [{'asset_id': asset['asset_id']}]})
                            result = result.model_dump(by_alias=True)
                            assert not result.get('isError') and len(result['content']) == 2
                            image = result['content'][1]
                            assert image['type'] == 'image' and len(base64.b64decode(image['data'])) <= 200_000
                            return {'server': initialized.model_dump(by_alias=True)['serverInfo']['name'],
                                    'tools': len(listed.tools), 'preview_bytes': len(base64.b64decode(image['data']))}

                sdk_result = asyncio.run(sdk_check())
                for scenario in ('normal', 'local_limit', 'upstream413'):
                    mode[0] = scenario
                    _REJECTED.clear()
                    state.codex_request_budget = (RequestBudgetConfig(enabled=True, warn_bytes=1024, max_bytes=1024)
                                                  if scenario == 'local_limit' else RequestBudgetConfig(enabled=True))
                    cli_home = root / scenario
                    cli_home.mkdir()
                    environment = {k: v for k, v in os.environ.items() if not (
                        k.upper().startswith('CODEX_') or k.upper().endswith(('_TOKEN', '_API_KEY'))
                        or k.upper() in ('OPENAI_API_KEY', 'CHATGPT_ACCESS_TOKEN', 'GH_TOKEN'))}
                    environment['CODEX_HOME'] = str(cli_home)
                    environment['NO_PROXY'] = '127.0.0.1,localhost,::1'
                    command = [str(args.codex), 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                               '--skip-git-repo-check', '--sandbox', 'read-only', '--json', '-m', 'gpt-6-astra',
                               '-c', 'model_provider="fixture"', '-c', 'model_providers.fixture.name="Local synthetic fixture"',
                               '-c', f'model_providers.fixture.base_url="http://127.0.0.1:{proxy.server_port}/v1"',
                               '-c', 'model_providers.fixture.wire_api="responses"', '-c', 'model_providers.fixture.requires_openai_auth=false',
                               '-c', 'model_providers.fixture.stream_max_retries=1',
                               '-c', f'mcp_servers.ghc_images.url="http://127.0.0.1:{proxy.server_port}/mcp/images"',
                               '-c', 'model_reasoning_effort="low"', '-c', 'features.plugins=false', '-c', 'features.apps=false',
                               '-c', 'features.code_mode=false',
                               'Use the image MCP to register the synthetic local image and preview it once. These local snapshot writes are authorized. Reply OK.']
                    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True,
                                             encoding='utf-8', errors='replace', timeout=55,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    posts = [row for row in observed['model_posts'] if row['mode'] == scenario]
                    wire = [row for row in observed['wire'] if row['mode'] == scenario]
                    result = {'scenario': scenario, 'exit_code': process.returncode,
                              'upstream_calls': len(posts), 'client_attempts': len(wire)}
                    cli_results.append(result)
                    if scenario == 'normal':
                        assert process.returncode == 0, process.stderr[-1500:] + process.stdout[-3000:] + json.dumps(observed)
                        assert len(posts) == 3, json.dumps(observed)
                        assert posts[-1]['images']['image_count'] >= 1, json.dumps(posts)
                        assert all(row['reasoning'] == 'true' for row in wire)
                        assert any(row['tool'] == 'register_image' for row in observed['mcp_calls'])
                        assert any(row['tool'] == 'preview_images' for row in observed['mcp_calls'])
                    elif scenario == 'local_limit':
                        assert process.returncode != 0 and not posts and len(wire) == 1, json.dumps(result)
                        assert 'ghc_request_budget_exceeded' in process.stdout + process.stderr
                    else:
                        assert process.returncode != 0 and len(posts) == 1, json.dumps({'result': result, 'posts': posts})
                        assert len(wire) == 2, json.dumps(result)
                        assert 'ghc_request_previous_413' in process.stdout + process.stderr
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(3)
        assert all(not thread.is_alive() for thread in threads)
        assert hashlib.sha256(picture_path.read_bytes()).hexdigest() == source_hash
        report = {'passed': True, 'synthetic_only': True, 'real_model_calls': 0,
                  'sdk': sdk_result, 'codex': cli_results, 'original_unchanged': True,
                  'temporary_servers_stopped': True}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
