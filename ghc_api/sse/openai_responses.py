"""OpenAI Responses (``/v1/responses``) SSE stream handler.

The upstream stream uses both ``event:`` and ``data:`` SSE lines. Preserve
their wire format, but keep output item IDs stable when Copilot rotates its
opaque IDs between events. Unchanged events remain byte-for-byte passthrough;
the base handler retains the original upstream payloads in the request cache.
"""

import json
import re
import threading
from typing import Callable, Dict, Iterator, Optional

import requests

from ..utils import is_encrypted_content_error
from .base import SSEStreamHandler


class RetryingResponsesResponse:
    """Replay a Responses request when it fails before producing output.

    Copilot occasionally returns a successful HTTP response whose SSE body is
    just ``response.created`` followed by ``response.failed``. Buffer only that
    non-output preamble so a fresh attempt can replace it transparently. Once
    any substantive event is seen, lines pass through immediately and retries
    are disabled so text and tool calls cannot be duplicated.

    An optional error_response_factory can recover a specific upstream error
    independently of the transient retry budget. It must bound its own retries.
    Keep a trailing error diagnostic with the failed preamble so invalid input
    is never blindly retried, and the original error survives failed recovery.

    on_diagnostic receives stage/code/status metadata without upstream content.
    Set initial_after_recovery if the caller already recovered an HTTP error.
    A completed diagnostic requires the actual response.completed event.
    """

    _PRE_OUTPUT_EVENTS = {
        "response.created",
        "response.in_progress",
        "response.queued",
    }
    _INVALID_REQUEST_CODES = ("invalid_request_body", "invalid_request_error")
    _MAX_BUFFERED_BYTES = 256 * 1024

    def __init__(
        self,
        response: requests.Response,
        response_factory: Callable[[], requests.Response],
        max_retries: int,
        request_id: str,
        error_response_factory: Optional[Callable[[Dict], Optional[requests.Response]]] = None,
        on_diagnostic: Optional[Callable[[Dict], None]] = None,
        initial_after_recovery: bool = False,
    ) -> None:
        self._response = response
        self._response_factory = response_factory
        self._max_retries = max(0, max_retries)
        self._request_id = request_id
        self._error_response_factory = error_response_factory
        self._on_diagnostic = on_diagnostic
        self._after_recovery = initial_after_recovery
        # Serialize callbacks with close(), allowing a callback to close the
        # wrapper itself without deadlocking.
        self._lock = threading.RLock()
        self._closed = False

        # SSEStreamHandler reads these attributes before iterating. ``text`` is a
        # property on purpose: on a ``stream=True`` response ``Response.text``
        # goes through ``Response.content``, which drains the socket and blocks
        # until upstream is done. Reading it here would turn every streaming
        # request into a buffered one.
        self.status_code = response.status_code
        self.ok = response.ok

    def _diagnostic(self, stage: str, **fields) -> None:
        """Report only bounded metadata; diagnostics cannot affect the wire."""
        with self._lock:
            if self._closed or self._on_diagnostic is None:
                return
            try:
                self._on_diagnostic({"stage": stage, **fields})
            except Exception:
                pass

    @staticmethod
    def _error_code(error) -> str:
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            return code
        return "unknown"

    def _recover_from_error(self, error: Dict) -> Optional[requests.Response]:
        if self._error_response_factory is None:
            return None
        try:
            recovered = self._error_response_factory(error)
        except Exception as exc:
            self._diagnostic("recovery_started")
            self._diagnostic("recovery_exception", error_type=type(exc).__name__)
            # Exception messages can include tokens, URLs or upstream bodies.
            print(
                f"[Stream Responses] Error recovery failed for request "
                f"{self._request_id}: {type(exc).__name__}"
            )
            return None
        # Returning None means the factory declined recovery, not an attempt.
        if recovered is not None:
            self._diagnostic("recovery_started")
            if not recovered.ok:
                self._diagnostic("recovery_http_error", status_code=recovered.status_code)
        return recovered

    @property
    def text(self) -> str:
        with self._lock:
            response = self._response
        return response.text

    def _current_response(self):
        with self._lock:
            if self._closed:
                return None
            return self._response

    def _replace_response(self, expected, replacement) -> bool:
        with self._lock:
            if self._closed or self._response is not expected:
                accepted = False
            else:
                self._response = replacement
                accepted = True

        if accepted:
            expected.close()
        else:
            replacement.close()
        return accepted

    @staticmethod
    def _event(line) -> Optional[Dict]:
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(line, str) or not line.startswith("data:"):
            return None
        data = line[5:].lstrip(" ")
        if data == "[DONE]":
            return {"type": "response.done"}
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # A malformed data payload is still downstream-visible output and
            # therefore commits the stream; represent it as an unknown event.
            return {}
        return event if isinstance(event, dict) else {}

    @staticmethod
    def _error_payload(event: Dict) -> Dict:
        """Return the error object of a standalone ``error`` event.

        The API reference documents a flat event, but the live service wraps
        the error in a nested ``error`` object instead. Both shapes occur in
        practice, which is why ``anthropic_error_from_responses`` has unwrapped
        them the same way since the Anthropic bridge shipped. Matching only the
        flat shape here would silently skip recovery on the nested one.
        """
        nested = event.get("error")
        return nested if isinstance(nested, dict) else event

    def _recover_from_http_error(self, response, after_recovery: bool) -> Optional[requests.Response]:
        """Offer error recovery a replay that upstream rejected in HTTP form.

        The same rejection arrives either as a pre-output SSE failure or as an
        HTTP 4xx, and the two can interleave across attempts. Dropping the HTTP
        body here would leave recovery unattempted and send the client the
        earlier, diagnostic-free ``response.failed`` instead.
        """
        if self._error_response_factory is None:
            return None
        try:
            error = response.json().get("error")
        except (AttributeError, TypeError, ValueError):
            return None
        if not isinstance(error, dict):
            return None
        if not after_recovery and is_encrypted_content_error(error):
            self._diagnostic("original_rejected", error_code=self._error_code(error))
        return self._recover_from_error(error)

    def iter_lines(self) -> Iterator[bytes]:
        retries = 0
        after_recovery = self._after_recovery

        while True:
            response = self._current_response()
            if response is None:
                return
            lines = iter(response.iter_lines())
            buffered = []
            buffered_bytes = 0
            output_started = False
            early_failure = False
            terminal_error = False
            error = None
            terminal_seen = False
            failure_seen = False
            rejection_reported = False

            while True:
                try:
                    line = next(lines)
                except StopIteration:
                    break
                except requests.exceptions.RequestException as exc:
                    if after_recovery:
                        self._diagnostic("recovery_exception", error_type=type(exc).__name__)
                    if not early_failure:
                        raise
                    # A broken connection while reading the optional diagnostic
                    # must not discard the terminal failure already received.
                    # Mid-stream breaks usually surface as ChunkedEncodingError
                    # rather than ConnectionError, so catch the common base.
                    lines = iter(())
                    break
                if self._current_response() is None:
                    return
                # The normal fast path need not parse output. When observed,
                # completion/failure must also be noticed after the first delta.
                event = self._event(line) if not output_started or self._on_diagnostic is not None else None
                event_type = event.get("type", "") if event is not None else None
                if event_type == "response.completed":
                    if not terminal_seen:
                        self._diagnostic("completed", after_recovery=after_recovery)
                    terminal_seen = True
                elif event_type in ("response.failed", "error"):
                    terminal_seen = failure_seen = True
                    failed = event.get("response")
                    event_error = (
                        self._error_payload(event) if event_type == "error"
                        else failed.get("error") if isinstance(failed, dict) else None
                    )
                    if not rejection_reported and isinstance(event_error, dict) and event_error:
                        if after_recovery or is_encrypted_content_error(event_error):
                            self._diagnostic(
                                "recovery_rejected" if after_recovery else "original_rejected",
                                error_code=self._error_code(event_error),
                            )
                            rejection_reported = True
                elif event_type == "response.incomplete":
                    terminal_seen = True
                if self._current_response() is None:
                    return
                if output_started:
                    yield line
                    continue

                buffered.append(line)
                buffered_bytes += len(line) + 1
                # Bound the preamble, including comments/diagnostics after a
                # failure. Once anything is forwarded it cannot be replaced.
                if buffered_bytes > self._MAX_BUFFERED_BYTES:
                    output_started = True
                    early_failure = False
                    yield from buffered
                    buffered.clear()
                    continue
                if event_type == "error":
                    error = self._error_payload(event)
                    terminal_error = True
                    break
                if early_failure:
                    # Copilot can put error=null on response.failed and send
                    # the actual rejection in a following standalone error.
                    # Read that diagnostic before deciding to replay history.
                    if event_type is None:
                        continue
                    if event_type == "response.done":
                        break
                    # Unexpected content after failure must remain visible.
                    early_failure = False
                    output_started = True
                    yield from buffered
                    buffered.clear()
                    continue
                if event_type == "response.failed":
                    failed = event.get("response") or {}
                    if isinstance(failed, dict) and not failed.get("output"):
                        early_failure = True
                        error = failed.get("error")
                        if isinstance(error, dict) and error:
                            break
                        continue
                if event_type is not None and event_type not in self._PRE_OUTPUT_EVENTS:
                    output_started = True
                    yield from buffered
                    buffered.clear()

            retry_response = None
            if self._current_response() is None:
                return
            if after_recovery and failure_seen and not rejection_reported:
                self._diagnostic("recovery_rejected", error_code="unknown")
            if isinstance(error, dict) and self._error_response_factory is not None:
                # Recovery is best effort; preserve the buffered diagnostic if
                # token refresh or the replacement request fails.
                retry_response = self._recover_from_error(error)
            recovery_response = retry_response is not None

            invalid_request = (
                isinstance(error, dict)
                and (
                    error.get("code") in self._INVALID_REQUEST_CODES
                    or error.get("type") in self._INVALID_REQUEST_CODES
                )
            )
            # A diagnostic means upstream stated a reason, so replaying the same
            # input mostly burns quota on a request that cannot succeed; only an
            # unexplained failure (``error: null`` and no diagnostic) is replayed.
            generic_retry = (
                retry_response is None and early_failure and not terminal_error
                and not invalid_request and retries < self._max_retries
            )
            if generic_retry:
                try:
                    retry_response = self._response_factory()
                except Exception as exc:
                    if after_recovery:
                        self._diagnostic("recovery_exception", error_type=type(exc).__name__)
                    # The original response.failed is more useful to the
                    # downstream client than replacing it with an empty 504.
                    print(
                        f"[Stream Responses] Early-failure retry failed for request "
                        f"{self._request_id}: {type(exc).__name__}"
                    )
                    retry_response = None

                if retry_response is not None and not retry_response.ok:
                    if after_recovery:
                        self._diagnostic("recovery_http_error", status_code=retry_response.status_code)
                    recovered = self._recover_from_http_error(retry_response, after_recovery)
                    if recovered is not None:
                        retry_response.close()
                        retry_response = recovered
                        # Recovery has its own one-shot budget; this attempt
                        # must not be charged to the connection retries.
                        generic_retry = False
                        recovery_response = True

            if retry_response is not None and retry_response.ok:
                if not self._replace_response(response, retry_response):
                    return
                after_recovery = after_recovery or recovery_response
                if generic_retry:
                    retries += 1
                    print(
                        f"[Stream Responses] Retrying request {self._request_id} "
                        f"after an early stream failure ({retries}/{self._max_retries})"
                    )
                continue

            # The retry could not establish a valid SSE stream. Preserve the
            # original failure and diagnostic for the downstream client.
            if retry_response is not None:
                retry_response.close()

            yield from buffered
            if early_failure or terminal_error:
                # Preserve any sentinel or diagnostic lines following the
                # terminal failure on the final attempt.
                yield from lines
            return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            response = self._response
        response.close()


class OpenAIResponsesStreamHandler(SSEStreamHandler):
    endpoint = "/v1/responses"
    log_prefix = "[Stream Responses]"
    # /v1/responses sends an ``event: TYPE`` line *before* each ``data:`` line.
    # We pass the event header through verbatim and emit only the data line
    # ourselves (the original handler's convention).
    emit_event_header = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._item_ids: Dict[int, str] = {}

    def _normalize_id(self, output_index, value: Dict, field: str) -> Dict:
        """Keep the first upstream ID for an output slot, without inventing IDs.

        The index correlates Copilot's independently wrapped IDs. Retain a real
        upstream ID rather than generating one: clients may replay output items
        in later requests. Do not touch call_id or encrypted reasoning content.
        """
        if type(output_index) is not int or output_index < 0:
            return value
        item_id = value.get(field)
        if not isinstance(item_id, str) or not item_id:
            return value
        canonical_id = self._item_ids.setdefault(output_index, item_id)
        if item_id == canonical_id:
            return value
        return {**value, field: canonical_id}

    def forward_event(self, event_type: str, event: Dict, raw_data: str) -> Iterator[tuple]:
        normalized = event
        if event_type.startswith("response."):
            output_index = event.get("output_index")
            if event_type in ("response.output_item.added", "response.output_item.done"):
                item = event.get("item")
                if isinstance(item, dict):
                    normalized_item = self._normalize_id(output_index, item, "id")
                    if normalized_item is not item:
                        normalized = {**normalized, "item": normalized_item}
            normalized = self._normalize_id(output_index, normalized, "item_id")

        if event_type in ("response.completed", "response.incomplete", "response.failed"):
            response = event.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                normalized_output = [
                    self._normalize_id(index, item, "id") if isinstance(item, dict) else item
                    for index, item in enumerate(output)
                ]
                if any(new is not old for new, old in zip(normalized_output, output)):
                    normalized = {
                        **normalized,
                        "response": {**response, "output": normalized_output},
                    }

        # Yield immediately, including deltas. Copy-on-write preserves both the
        # parsed upstream event and the exact wire bytes for healthy streams.
        yield (event_type, raw_data if normalized is event else json.dumps(normalized, ensure_ascii=False))

    def on_event(self, event_type: str, event: Dict) -> None:
        if event_type in ("response.completed", "response.incomplete"):
            resp = event.get("response", {}) or {}
            usage = resp.get("usage", {}) or {}
            self.input_tokens = usage.get("input_tokens", 0)
            self.output_tokens = usage.get("output_tokens", 0)
            details = usage.get("input_tokens_details", {}) or {}
            self.cache_creation_input_tokens = details.get("cached_tokens", 0)
        elif event_type == "response.failed":
            # The HTTP status is already committed as 200, but request history
            # should still identify the terminal SSE failure.
            self.error_occurred = True
            self.status_code = 502
        elif event_type == "error":
            # Same for the standard Responses streaming ``error`` event, which
            # the proxy itself emits when an upstream error arrives after the
            # response headers were already committed. Without this a chained
            # ghc-api would record the failure as 200/completed.
            self.error_occurred = True
            self.status_code = 502
