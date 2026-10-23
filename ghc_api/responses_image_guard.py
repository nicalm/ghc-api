"""Non-mutating image limits for Responses requests.

Responses images can be carried by message content, or by typed tool outputs.
Tool declarations and opaque argument strings are not input image content.
"""

import copy
import json
import time
import uuid
from typing import Any, Dict, Iterator, Mapping, Optional


DEFAULT_MAX_INPUT_IMAGES = 50
_NON_CONTENT_ITEMS = {
    "additional_tools", "function_call", "custom_tool_call", "reasoning",
    "compaction", "configuration_update",
}
_TOOL_OUTPUT_ITEMS = {"function_call_output", "custom_tool_call_output"}


def _input_image_blocks(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    items = payload.get("input")
    if not isinstance(items, (list, dict)):
        return
    stack = [items]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
            continue
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind is not None and not isinstance(kind, str):
            # Leave malformed item validation to the API instead of raising
            # while looking up an unhashable user-supplied type value.
            continue
        if kind == "input_image":
            yield item
        elif kind not in _NON_CONTENT_ITEMS:
            field = "output" if kind in _TOOL_OUTPUT_ITEMS else "content"
            parts = item.get(field)
            if isinstance(parts, (list, dict)):
                stack.append(parts)


def count_input_images(payload: Mapping[str, Any]) -> int:
    """Count image blocks in input history, including typed tool outputs."""
    return sum(1 for _ in _input_image_blocks(payload))


def has_input_images(payload: Mapping[str, Any]) -> bool:
    """Whether upstream vision support is needed for this request."""
    return next(_input_image_blocks(payload), None) is not None


def image_limit_error(
    payload: Mapping[str, Any],
    max_images: int = DEFAULT_MAX_INPUT_IMAGES,
) -> Optional[Dict[str, Any]]:
    """Return a structured image-count error without changing the history.

    The caller must run this before committing response headers. The payload,
    image content, and conversation history are never modified here.
    An ordinary HTTP 400 does not automatically initiate Codex compaction.
    """
    if type(max_images) is not int or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    image_count = count_input_images(payload)
    if image_count <= max_images:
        return None
    return {
        "error": {
            "message": (
                f"Image history exceeds the upstream request limit: "
                f"{image_count} images (maximum {max_images}). "
                "Compact earlier image history or split the images into "
                "smaller requests."
            ),
            "type": "invalid_request_error",
            "param": "input",
            "code": "context_length_exceeded",
        }
    }


def image_limit_compaction_sse(
    payload: Mapping[str, Any],
    limit_error: Mapping[str, Any],
    request_metadata: Any,
    request_id: Optional[str] = None,
) -> Optional[str]:
    """Let an already-running Codex compaction shrink image-heavy history.

    The native compaction loop recognizes this specific terminal SSE error and
    removes an oldest history item before retrying. Ordinary requests keep the
    explicit HTTP 400 image-count explanation; this helper does not start a
    compaction or recover a business turn on its own. Policy errors are never
    reclassified as context errors.
    """
    if payload.get("stream") is not True:
        return None
    if isinstance(request_metadata, str):
        try:
            request_metadata = json.loads(request_metadata)
        except (ValueError, TypeError):
            return None
    if (
        not isinstance(request_metadata, Mapping)
        or request_metadata.get("request_kind") != "compaction"
    ):
        return None
    error = limit_error.get("error") if isinstance(limit_error, Mapping) else None
    if not isinstance(error, Mapping) or error.get("code") != "context_length_exceeded":
        return None
    response = {
        "id": "resp_ghc_image_limit_" + (request_id or uuid.uuid4().hex),
        "object": "response",
        "created_at": int(time.time()),
        "model": payload.get("model", "unknown"),
        "status": "in_progress",
        "output": [],
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }
    events = [
        {"type": "response.created", "sequence_number": 0, "response": response},
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": {**response, "status": "failed", "error": copy.deepcopy(dict(error))},
        },
    ]
    return "".join(
        "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
        for event in events
    )
