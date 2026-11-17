"""Measure the exact outbound JSON bytes, without mutating conversation history."""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import requests

from ..counters import counters
from ..utils import get_config_dir


_LOCK = threading.RLock()
_REJECTED = OrderedDict()
_MAX_REJECTED = 128
_LOG = logging.getLogger(__name__)


def image_statistics(payload):
    """Only inspect typed image fields. Do not fetch URLs or parse quoted JSON."""
    count = encoded = largest = 0
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            kind = value.get('type')
            image = None
            if kind in ('input_image', 'image_url'):
                image = value.get('image_url')
                if isinstance(image, dict):
                    image = image.get('url')
            elif kind == 'image':
                source = value.get('source')
                image = value.get('data')
                if isinstance(source, dict):
                    image = source.get('data') if source.get('type') == 'base64' else source.get('url')
            if kind in ('input_image', 'image_url', 'image'):
                count += 1
                size = len(image.encode('utf-8')) if isinstance(image, str) and (
                    image.startswith('data:image/') or kind == 'image' and not image.startswith(('http:', 'https:'))
                ) else 0
                encoded += size
                largest = max(largest, size)
            # Traverse nested tool outputs, but never expand strings (possibly private text).
            pending.extend(v for k, v in value.items() if isinstance(v, (dict, list)) and k != 'source')
        elif isinstance(value, list):
            pending.extend(value)
    return {'image_count': count, 'inline_image_bytes': encoded, 'largest_image_bytes': largest}


def _audit(entry):
    """Small rotating metadata-only log. Logging must not break model requests."""
    try:
        path = Path(get_config_dir()) / 'request-budget.jl'
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size >= 2 * 1024 * 1024:
                backup = path.with_suffix('.jl.1')
                backup.unlink(missing_ok=True)
                path.replace(backup)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'timestamp': int(time.time()), **entry}, separators=(',', ':')) + '\n')
    except OSError:
        _LOG.warning('Could not write request budget metadata')


class RequestBudgetError(ValueError):
    def __init__(self, code, metrics, limit):
        self.code = code
        self.metrics = metrics
        self.limit = limit
        why = 'exceeds the local request budget' if code == 'ghc_request_budget_exceeded' else 'was already rejected by upstream with HTTP 413'
        super().__init__(
            f"Request ({metrics['request_bytes'] / 1048576:.2f} MiB, "
            f"images {metrics['inline_image_bytes'] / 1048576:.2f} MiB) {why}. "
            f"Local limit: {limit / 1048576:.2f} MiB. Use the image preview tools and a text progress file; "
            'continue in a clean task if its existing image history cannot be reduced. '
            'The same payload will not be uploaded again.'
        )

    def response_body(self):
        return {'error': {'type': 'invalid_request_error', 'code': self.code,
                          'message': str(self), 'retryable': False,
                          'details': {**self.metrics, 'limit_bytes': self.limit}}}


class ResponsesBudget:
    def __init__(self, policy, owner, task_id, endpoint, request_id):
        self.policy = policy
        self.scope = hashlib.sha256((owner + '\0' + task_id + '\0' + endpoint).encode()).hexdigest()
        self.request_id = request_id
        self.metrics = None

    def _prepare(self, url, headers, payload):
        # Requests prepares JSON with its own encoder (including Unicode escaping).
        # Send these very bytes using data= so counting and transport cannot diverge.
        prepared = requests.Request('POST', url, headers=headers, json=payload).prepare()
        body = prepared.body
        if isinstance(body, str):
            body = body.encode('utf-8')
        metrics = {'request_bytes': len(body), **image_statistics(payload)}
        self.metrics = metrics
        # Codex can reorder JSON object keys on retry. Canonicalize only for
        # identity; count/send the exact prepared bytes and retain every field.
        fingerprint = hashlib.sha256(json.dumps(
            payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        key = (self.scope, fingerprint)
        now = time.monotonic()
        with _LOCK:
            for old_key in [k for k, expiry in _REJECTED.items() if expiry <= now]:
                _REJECTED.pop(old_key, None)
            remembered = key in _REJECTED
        code = None
        if metrics['request_bytes'] > self.policy.max_bytes:
            code = 'ghc_request_budget_exceeded'
        elif remembered:
            code = 'ghc_request_previous_413'
        if code or metrics['request_bytes'] >= self.policy.warn_bytes:
            _audit({'event': code or 'request_budget_warning', 'request_id': self.request_id,
                    'scope_sha256': self.scope, 'payload_sha256': fingerprint,
                    **metrics, 'limit_bytes': self.policy.max_bytes})
        if code:
            raise RequestBudgetError(code, metrics, self.policy.max_bytes)
        return body, prepared.headers, key

    def post(self, url, headers, payload, stream):
        # Native requests wait for truthful upstream headers before entering the
        # stream cache. Keep that pre-header work visible to shared-service drains.
        counters.incr('codex_budget.inflight')
        try:
            return self._post(url, headers, payload, stream)
        finally:
            counters.incr('codex_budget.inflight', -1)

    def _post(self, url, headers, payload, stream):
        body, final_headers, key = self._prepare(url, headers, payload)
        from ..state import state
        response = requests.post(url, headers=dict(final_headers), data=body, stream=stream,
                                 timeout=state.upstream_read_timeout)
        if response.status_code == 413:
            with _LOCK:
                _REJECTED[key] = time.monotonic() + self.policy.rejected_ttl_seconds
                _REJECTED.move_to_end(key)
                while len(_REJECTED) > _MAX_REJECTED:
                    _REJECTED.popitem(last=False)
            _audit({'event': 'upstream_413', 'request_id': self.request_id,
                    'scope_sha256': self.scope, 'payload_sha256': key[1], **self.metrics})
        return response

    def headers(self):
        if not self.metrics:
            return {}
        return {'X-GHC-Request-Bytes': str(self.metrics['request_bytes']),
                'X-GHC-Request-Budget': str(self.policy.max_bytes),
                'X-GHC-Budget-Warning': str(self.metrics['request_bytes'] >= self.policy.warn_bytes).lower()}
