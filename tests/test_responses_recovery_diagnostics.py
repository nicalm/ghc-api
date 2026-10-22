"""Recovery diagnostics describe attempts without changing the Responses wire."""

import json
import threading
from unittest import mock

import pytest
import requests

from ghc_api.sse import RetryingResponsesResponse


ENCRYPTED_ERROR = {
    "code": "invalid_request_body",
    "message": "The encrypted content SECRET_CIPHERTEXT could not be verified. "
    "Reason: Encrypted content could not be decrypted or parsed.",
}


def event(kind, **fields):
    return ("data: " + json.dumps({"type": kind, **fields})).encode()


def rejected_lines(error=ENCRYPTED_ERROR):
    return [
        b"event: response.created",
        event("response.created"),
        event("response.failed", response={"error": None}),
        b"event: error",
        event("error", error=error),
        b"data: [DONE]",
    ]


class FakeResponse:
    def __init__(self, lines=(), status_code=200, body=None):
        self.lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.body = body
        self.closed = False

    def iter_lines(self):
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def json(self):
        return self.body

    def close(self):
        self.closed = True


def wrapper(response, diagnostics, recover=None, retries=0, factory=None, **kwargs):
    return RetryingResponsesResponse(
        response, factory or mock.Mock(), retries, "diagnostics-test",
        error_response_factory=recover, on_diagnostic=diagnostics.append, **kwargs,
    )


def test_completed_recovery_is_byte_preserving_and_has_safe_diagnostics():
    success = [b": keepalive", event("response.created"),
               b'data:{"type":"response.output_text.delta","delta":"SECRET_OUTPUT"}',
               event("response.completed", response={"output": [{"encrypted_content": "SECRET"}]}),
               b"data: [DONE]"]
    first = FakeResponse(rejected_lines())
    diagnostics = []
    recover = mock.Mock(return_value=FakeResponse(success))

    assert list(wrapper(first, diagnostics, recover).iter_lines()) == success

    recover.assert_called_once_with(ENCRYPTED_ERROR)
    assert first.closed
    assert diagnostics == [
        {"stage": "original_rejected", "error_code": "invalid_request_body"},
        {"stage": "recovery_started"},
        {"stage": "completed", "after_recovery": True},
    ]
    assert "SECRET" not in json.dumps(diagnostics)


@pytest.mark.parametrize("terminal", [None, "response.incomplete", "response.failed"])
def test_incomplete_failed_or_eof_stream_never_reports_success(terminal):
    lines = [event("response.output_text.delta", delta="partial")]
    if terminal:
        lines.append(event(terminal, response={"error": None}))
    lines.append(b"data: [DONE]")
    diagnostics = []
    assert list(wrapper(FakeResponse(lines), diagnostics).iter_lines()) == lines
    assert diagnostics == []


def test_completion_after_output_is_reported_without_recovery():
    lines = [event("response.output_text.delta", delta="hello"),
             event("response.completed", response={}), b"data: [DONE]"]
    diagnostics = []
    assert list(wrapper(FakeResponse(lines), diagnostics).iter_lines()) == lines
    assert diagnostics == [{"stage": "completed", "after_recovery": False}]


@pytest.mark.parametrize("failure", [requests.ConnectionError("SECRET_TOKEN"), RuntimeError("SECRET_TOKEN")])
def test_recovery_creation_exception_retains_original_wire_without_leaking_message(failure, capsys):
    diagnostics = []
    lines = rejected_lines()
    assert list(wrapper(FakeResponse(lines), diagnostics, mock.Mock(side_effect=failure)).iter_lines()) == lines
    assert diagnostics == [
        {"stage": "original_rejected", "error_code": "invalid_request_body"},
        {"stage": "recovery_started"},
        {"stage": "recovery_exception", "error_type": type(failure).__name__},
    ]
    assert "SECRET" not in json.dumps(diagnostics)
    assert "SECRET_TOKEN" not in capsys.readouterr().out


def test_recovery_http_failure_reports_status_and_retains_original_wire():
    lines = rejected_lines()
    diagnostics = []
    failed = FakeResponse(status_code=503, body={"error": {"message": "SECRET"}})
    assert list(wrapper(FakeResponse(lines), diagnostics, mock.Mock(return_value=failed)).iter_lines()) == lines
    assert failed.closed
    assert diagnostics[-1] == {"stage": "recovery_http_error", "status_code": 503}
    assert not any(item["stage"] == "completed" for item in diagnostics)


@pytest.mark.parametrize("with_output", [False, True])
def test_failure_of_recovery_stream_is_visible_and_never_retried_after_output(with_output):
    failure = {"code": "upstream_error", "message": "SECRET_BODY"}
    lines = ([event("response.output_text.delta", delta="partial")] if with_output else [])
    lines += [event("response.failed", response={"error": failure}), b"data: [DONE]"]
    diagnostics = []
    recover = mock.Mock(side_effect=[FakeResponse(lines), None])
    retry = mock.Mock()
    assert list(wrapper(FakeResponse(rejected_lines()), diagnostics, recover, factory=retry).iter_lines()) == lines
    assert diagnostics[-1] == {"stage": "recovery_rejected", "error_code": "upstream_error"}
    assert recover.call_count == (1 if with_output else 2)
    retry.assert_not_called()
    assert "SECRET" not in json.dumps(diagnostics)


def test_recovery_stream_transport_exception_keeps_existing_exception_behavior():
    diagnostics = []
    broken = FakeResponse([requests.ConnectionError("SECRET_URL")])
    stream = wrapper(FakeResponse(rejected_lines()), diagnostics, mock.Mock(return_value=broken))
    with pytest.raises(requests.ConnectionError, match="SECRET_URL"):
        list(stream.iter_lines())
    assert diagnostics[-1] == {"stage": "recovery_exception", "error_type": "ConnectionError"}
    assert "SECRET" not in json.dumps(diagnostics)


def test_encrypted_error_after_output_is_reported_without_retry():
    lines = [event("response.output_text.delta", delta="partial"), event("error", error=ENCRYPTED_ERROR)]
    recover, retry, diagnostics = mock.Mock(), mock.Mock(), []
    assert list(wrapper(FakeResponse(lines), diagnostics, recover, 3, retry).iter_lines()) == lines
    recover.assert_not_called()
    retry.assert_not_called()
    assert diagnostics == [{"stage": "original_rejected", "error_code": "invalid_request_body"}]


def test_http_rejection_on_generic_retry_can_start_recovery():
    first = FakeResponse([event("response.failed", response={"error": None})])
    rejected = FakeResponse(status_code=400, body={"error": ENCRYPTED_ERROR})
    success = [event("response.completed", response={})]
    diagnostics = []
    assert list(wrapper(first, diagnostics, mock.Mock(return_value=FakeResponse(success)),
                        1, mock.Mock(return_value=rejected)).iter_lines()) == success
    assert rejected.closed
    assert diagnostics == [
        {"stage": "original_rejected", "error_code": "invalid_request_body"},
        {"stage": "recovery_started"},
        {"stage": "completed", "after_recovery": True},
    ]


def test_http_first_recovery_can_mark_initial_stream_as_recovered():
    diagnostics = []
    lines = [event("error", error={"code": "upstream_error", "message": "SECRET"})]
    assert list(wrapper(FakeResponse(lines), diagnostics, initial_after_recovery=True).iter_lines()) == lines
    assert diagnostics == [{"stage": "recovery_rejected", "error_code": "upstream_error"}]


def test_declined_recovery_is_not_reported_as_started():
    diagnostics = []
    lines = rejected_lines()
    assert list(wrapper(FakeResponse(lines), diagnostics, mock.Mock(return_value=None)).iter_lines()) == lines
    assert diagnostics == [{"stage": "original_rejected", "error_code": "invalid_request_body"}]


@pytest.mark.parametrize("error", [None, {}])
def test_recovered_failure_uses_trailing_error_code_or_reports_unknown(error):
    for trailing in ([], [event("error", error={"code": "upstream_error", "message": "SECRET"})]):
        diagnostics = []
        lines = [event("response.failed", response={"error": error}), *trailing, b"data: [DONE]"]
        assert list(wrapper(FakeResponse(lines), diagnostics, initial_after_recovery=True).iter_lines()) == lines
        assert diagnostics == [{"stage": "recovery_rejected", "error_code": "upstream_error" if trailing else "unknown"}]


def test_error_code_is_bounded_metadata_and_cannot_include_payload_object():
    for code in ({"message": "SECRET"}, "SECRET\nCiphertext", "x" * 1000):
        diagnostics = []
        lines = [event("error", error={"code": code, "message": "SECRET"})]
        assert list(wrapper(FakeResponse(lines), diagnostics, initial_after_recovery=True).iter_lines()) == lines
        assert diagnostics == [{"stage": "recovery_rejected", "error_code": "unknown"}]


def test_callback_errors_cannot_break_wire_or_recovery():
    lines = [event("response.completed", response={})]
    stream = RetryingResponsesResponse(FakeResponse(lines), mock.Mock(), 0, "callback-failure",
                                      on_diagnostic=mock.Mock(side_effect=RuntimeError("SECRET")))
    assert list(stream.iter_lines()) == lines


def test_close_during_recovery_suppresses_late_diagnostics_and_closes_replacement():
    diagnostics = []
    recovery_started, release_recovery = threading.Event(), threading.Event()
    replacement = FakeResponse([event("response.completed", response={})])

    def recover(error):
        recovery_started.set()
        assert release_recovery.wait(2)
        return replacement

    stream = wrapper(FakeResponse(rejected_lines()), diagnostics, recover)
    output = []
    thread = threading.Thread(target=lambda: output.extend(stream.iter_lines()))
    thread.start()
    try:
        assert recovery_started.wait(2)
        stream.close()
        at_close = list(diagnostics)
    finally:
        release_recovery.set()
        thread.join(2)
    assert not thread.is_alive()
    assert diagnostics == at_close
    assert output == []
    assert replacement.closed


def test_callback_may_close_wrapper_without_deadlocking():
    lines = [event("response.completed", response={})]
    stream = RetryingResponsesResponse(FakeResponse(lines), mock.Mock(), 0, "close-callback",
                                      on_diagnostic=lambda diagnostic: stream.close())
    result = []
    thread = threading.Thread(target=lambda: result.extend(stream.iter_lines()), daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
