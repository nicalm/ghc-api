"""Validated, opt-in settings; importing this module never opens local files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageToolsConfig:
    enabled: bool = False
    allowed_roots: tuple = ()
    data_dir: str = ""
    max_source_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 20_000_000
    preview_max_bytes: int = 200_000
    preview_max_edge: int = 1024
    context_budget_bytes: int = 4 * 1024 * 1024
    store_max_bytes: int = 256 * 1024 * 1024
    asset_ttl_seconds: int = 86400


@dataclass(frozen=True)
class RequestBudgetConfig:
    enabled: bool = False
    warn_bytes: int = 12 * 1024 * 1024
    max_bytes: int = 20 * 1024 * 1024
    rejected_ttl_seconds: int = 300


def _settings(raw, kind, ranges):
    if not isinstance(raw, dict):
        raise ValueError(f"{kind} must be a mapping")
    result = dict(raw)
    if 'enabled' in result and not isinstance(result['enabled'], bool):
        raise ValueError(f"{kind}.enabled must be a boolean")
    for key, (lower, upper) in ranges.items():
        if key in result and (type(result[key]) is not int or not lower <= result[key] <= upper):
            raise ValueError(f"{kind}.{key} must be an integer in {lower}..{upper}")
    return result


def parse_image_config(raw):
    values = _settings(raw, 'image_tools', {
        'max_source_bytes': (1024, 128 * 1024 * 1024),
        'max_pixels': (1024, 50_000_000),
        'preview_max_bytes': (16_384, 1024 * 1024),
        'preview_max_edge': (320, 2048),
        'context_budget_bytes': (16_384, 16 * 1024 * 1024),
        'store_max_bytes': (1024, 4 * 1024 * 1024 * 1024),
        'asset_ttl_seconds': (60, 30 * 86400),
    })
    roots = values.get('allowed_roots', [])
    if not isinstance(roots, (list, tuple)) or not all(isinstance(p, str) and p for p in roots):
        raise ValueError('image_tools.allowed_roots must be a list of absolute directories')
    resolved = []
    for name in roots:
        path = Path(name)
        if not path.is_absolute():
            raise ValueError('image_tools.allowed_roots must be absolute')
        path = path.resolve(strict=True)
        if not path.is_dir() or path == Path(path.anchor):
            raise ValueError('image_tools.allowed_roots must be specific directories, not drive roots')
        resolved.append(str(path))
    values['allowed_roots'] = tuple(resolved)
    if values.get('enabled') and not resolved:
        raise ValueError('image_tools.enabled requires explicit allowed_roots')
    directory = values.get('data_dir', '')
    if not isinstance(directory, str) or (directory and not Path(directory).is_absolute()):
        raise ValueError('image_tools.data_dir must be empty or an absolute path')
    try:
        result = ImageToolsConfig(**values)
    except TypeError as exc:
        raise ValueError('Unknown image_tools setting') from exc
    if result.context_budget_bytes < result.preview_max_bytes:
        raise ValueError('image_tools.context_budget_bytes must cover one preview')
    if result.store_max_bytes < result.max_source_bytes:
        raise ValueError('image_tools.store_max_bytes must cover one source image')
    return result


def parse_budget_config(raw):
    values = _settings(raw, 'codex_request_budget', {
        'warn_bytes': (1024, 256 * 1024 * 1024),
        'max_bytes': (1024, 256 * 1024 * 1024),
        'rejected_ttl_seconds': (1, 3600),
    })
    try:
        result = RequestBudgetConfig(**values)
    except TypeError as exc:
        raise ValueError('Unknown codex_request_budget setting') from exc
    if result.warn_bytes > result.max_bytes:
        raise ValueError('codex_request_budget.warn_bytes cannot exceed max_bytes')
    return result


def apply_media_config(config, target):
    """Validate both sections before changing either setting."""
    images = parse_image_config(config.get('image_tools', {}))
    budget = parse_budget_config(config.get('codex_request_budget', {}))
    target.image_tools = images
    target.codex_request_budget = budget
