"""Responses failure-tail regressions; no network or credentials."""
import contextlib
import io
import json
import threading
import time
import unittest

import requests

from ghc_api.sse.openai_responses import RetryingResponsesResponse

CREATED = {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}
FAILED = {"type": "response.failed", "response": {"id": "r1", "status": "failed", "error": None, "output": []}}
SCHEMA = {"code": "invalid_request_body", "message": "Portable encrypted function call arguments require their original tool declaration or saved encryption schema."}
ATTACHMENT = {"code": "invalid_request_body", "message": "Failed to parse function_output: invalid encrypted_content"}
TEXT = {"type": "response.output_text.delta", "delta": "visible text"}
TOOL = {"type": "response.output_item.added", "item": {"type": "function_call", "name": "write"}}
COMPLETED = {"type": "response.completed", "response": {"id": "r2", "status": "completed", "output": [], "usage": {}}}


def wire(*events, sentinel=False):
    lines = []
    for event in events:
        lines.extend([("event: " + event["type"]).encode(),
                      ("data: " + json.dumps(event)).encode(), b""])
    if sentinel:
        lines.extend([b"data: [DONE]", b""])
    return lines


def decoded(lines):
    return [json.loads(x[5:].lstrip()) for x in lines if x.startswith(b"data:")
            and x[5:].strip() != b"[DONE]"]


def failure(error=SCHEMA):
    return Response(wire(CREATED, FAILED, {"type": "error", **error}, sentinel=True))


def success():
    return Response(wire(CREATED, TEXT, COMPLETED, sentinel=True))


class Response:
    status_code = 200
    ok = True

    def __init__(self, lines, *, status=200, body=None):
        self.lines = lines
        self.status_code = status
        self.ok = status < 400
        self.body = body
        self.closed = 0
        self.yielded = 0
        self.text_reads = 0

    @property
    def text(self):
        self.text_reads += 1
        raise AssertionError("Reading text drains a streaming response")

    def json(self):
        return self.body

    def iter_lines(self):
        for line in self.lines:
            self.yielded += 1
            yield line

    def close(self):
        self.closed += 1


class Factory:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result

    def __call__(self):
        self.calls += 1
        if self.result is None:
            raise AssertionError("Unexpected generic replay")
        return self.result


class ResponsesFailureTailTests(unittest.TestCase):
    def make(self, response, generic=None, **kwargs):
        return RetryingResponsesResponse(
            response, generic or Factory(), kwargs.pop("max_retries", 3), "offline-test",
            failure_diagnostic_timeout=kwargs.pop("failure_diagnostic_timeout", 0.2),
            **kwargs,
        )

    def consume(self, wrapper):
        with contextlib.redirect_stdout(io.StringIO()):
            return list(wrapper.iter_lines())

    def test_real_schema_tail_is_visible_on_first_failed_and_not_blindly_retried(self):
        generic = Factory()
        out = self.consume(self.make(failure(), generic))
        events = decoded(out)
        self.assertEqual(events[1]["response"]["error"], SCHEMA)
        self.assertEqual(events[2], {"type": "error", **SCHEMA})
        self.assertEqual(generic.calls, 0)
        self.assertIn(b"data: [DONE]", out)

    def test_nested_error_is_enriched_without_replacing_existing_details(self):
        first = {"type": "response.failed", "response": {"error": {"code": "permission_denied", "message": "original"}, "output": []}}
        response = Response(wire(CREATED, first, {"type": "error", "error": SCHEMA}))
        generic = Factory()
        out = self.consume(self.make(response, generic))
        self.assertEqual(decoded(out)[1]["response"]["error"], first["response"]["error"])
        self.assertEqual(generic.calls, 0)
        nested = Response(wire(CREATED, FAILED, {"type": "error", "error": SCHEMA}))
        self.assertEqual(decoded(self.consume(self.make(nested)))[1]["response"]["error"], SCHEMA)

    def test_permanent_embedded_categories_do_not_use_generic_replay(self):
        for code in ["permission_denied", "invalid_api_key", "unsupported_model", "context_length_exceeded", "insufficient_quota"]:
            with self.subTest(code=code):
                event = {"type": "response.failed", "response": {"error": {"code": code, "message": "stop"}, "output": []}}
                generic = Factory()
                self.consume(self.make(Response(wire(CREATED, event)), generic))
                self.assertEqual(generic.calls, 0)

    def test_upstream_standalone_error_policy_remains_conservative(self):
        generic = Factory()
        out = self.consume(self.make(failure({"code": "server_error", "message": "temporary"}), generic))
        self.assertEqual(generic.calls, 0)
        self.assertEqual(decoded(out)[1]["response"]["error"]["code"], "server_error")

    def test_unexplained_early_failure_keeps_existing_retry(self):
        replacement = success()
        generic = Factory(replacement)
        initial = Response(wire(CREATED, FAILED))
        self.assertEqual(self.consume(self.make(initial, generic)), replacement.lines)
        self.assertEqual(generic.calls, 1)
        self.assertEqual(initial.closed, 1)

    def test_failed_embedded_output_is_never_replayed(self):
        embedded = {"type": "response.failed", "response": {"error": None, "output": [{"type": "message", "content": []}]}}
        generic, repairs = Factory(), []
        response = Response(wire(CREATED, embedded))
        out = self.consume(self.make(response, generic, error_response_factory=lambda error: repairs.append(error)))
        self.assertEqual(out, response.lines)
        self.assertEqual(generic.calls, 0)
        self.assertEqual(repairs, [])

    def test_visible_output_never_replays_and_error_still_gets_detail(self):
        for output in [TEXT, TOOL]:
            with self.subTest(output=output["type"]):
                generic, repairs = Factory(), []
                response = Response(wire(CREATED, output, FAILED, {"type": "error", **SCHEMA}))
                out = self.consume(self.make(response, generic, error_response_factory=lambda error: repairs.append(error)))
                events = decoded(out)
                self.assertEqual(events.count(output), 1)
                self.assertEqual(events[2]["response"]["error"], SCHEMA)
                self.assertEqual(generic.calls, 0)
                self.assertEqual(repairs, [])

    def test_output_after_failure_or_error_also_disables_repair(self):
        for prefix in [(CREATED, FAILED, {"type": "error", **SCHEMA}), (CREATED, {"type": "error", **SCHEMA})]:
            with self.subTest(prefix=len(prefix)):
                generic, repairs = Factory(), []
                response = Response(wire(*prefix, TOOL))
                out = self.consume(self.make(response, generic, error_response_factory=lambda error: repairs.append(error)))
                self.assertIn(TOOL, decoded(out))
                self.assertEqual(generic.calls, 0)
                self.assertEqual(repairs, [])

    def test_schema_and_attachment_repairs_each_once_in_both_orders(self):
        for order in [(SCHEMA, ATTACHMENT), (ATTACHMENT, SCHEMA)]:
            for second_transport in ["http", "sse"]:
                with self.subTest(first=order[0]["message"], second=second_transport):
                    generic, performed, callbacks, diagnostics = Factory(), [], [], []
                    def repair(error):
                        callbacks.append(error["message"])
                        if error["message"] in performed:
                            return None
                        performed.append(error["message"])
                        if len(performed) == 1:
                            return (Response([], status=400, body={"error": order[1]})
                                    if second_transport == "http" else failure(order[1]))
                        return success()
                    out = self.consume(self.make(failure(order[0]), generic, max_retries=0,
                                                 error_response_factory=repair, on_diagnostic=diagnostics.append))
                    self.assertEqual(performed, [e["message"] for e in order])
                    self.assertEqual(len(callbacks), 2)
                    self.assertEqual(generic.calls, 0)
                    self.assertNotIn("response.failed", [e["type"] for e in decoded(out)])
                    self.assertIn({"stage": "completed", "after_recovery": True}, diagnostics)

    def test_repeated_schema_rejection_gets_only_one_actual_repair(self):
        generic, callbacks, performed = Factory(), [], []
        def repair(error):
            callbacks.append(error)
            if performed:
                return None
            performed.append(error)
            return failure()
        out = self.consume(self.make(failure(), generic, error_response_factory=repair))
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(len(performed), 1)
        self.assertEqual(generic.calls, 0)
        self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)

    def test_http_400_after_generic_retry_enters_original_recovery_factory(self):
        generic = Factory(Response([], status=400, body={"error": SCHEMA}))
        repairs = []
        def repair(error):
            repairs.append(error)
            return success()
        out = self.consume(self.make(Response(wire(CREATED, FAILED)), generic, error_response_factory=repair))
        self.assertEqual(generic.calls, 1)
        self.assertEqual(repairs, [SCHEMA])
        self.assertIn(COMPLETED, decoded(out))

    def test_non_400_http_repair_result_preserves_original_diagnostic(self):
        generic, repairs = Factory(), []
        replacement = Response([], status=403, body={"error": ATTACHMENT})
        def repair(error):
            repairs.append(error)
            return replacement
        out = self.consume(self.make(failure(), generic, error_response_factory=repair))
        self.assertEqual(repairs, [SCHEMA])
        self.assertEqual(generic.calls, 0)
        self.assertEqual(replacement.closed, 1)
        self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)

    def test_initial_http_recovery_flag_and_completed_callback_survive(self):
        diagnostics = []
        response = success()
        out = self.consume(self.make(response, on_diagnostic=diagnostics.append, initial_after_recovery=True))
        self.assertEqual(out, response.lines)
        self.assertIn({"stage": "completed", "after_recovery": True}, diagnostics)

    def test_repair_exception_retains_error_and_redacts_exception_message(self):
        diagnostics, calls = [], []
        def repair(error):
            calls.append(error)
            raise RuntimeError("private-exception-detail")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            out = list(self.make(failure(), error_response_factory=repair, on_diagnostic=diagnostics.append).iter_lines())
        self.assertEqual(len(calls), 1)
        self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)
        self.assertNotIn("private-exception-detail", stdout.getvalue())
        self.assertNotIn("private-exception-detail", json.dumps(diagnostics))

    def test_healthy_stream_stays_lazy_and_byte_identical(self):
        response = Response([b": ping", b""] + wire(CREATED, TEXT, COMPLETED, sentinel=True))
        iterator = self.make(response).iter_lines()
        first = next(iterator)
        self.assertLess(response.yielded, len(response.lines))
        self.assertEqual([first] + list(iterator), response.lines)
        self.assertEqual(response.text_reads, 0)

    def test_upstream_preamble_cap_still_commits_before_replay(self):
        generic, repairs = Factory(), []
        response = Response([b": long comment before output"] * 12 + wire(CREATED, FAILED, {"type": "error", **SCHEMA}))
        wrapper = self.make(response, generic, error_response_factory=lambda error: repairs.append(error))
        wrapper._MAX_BUFFERED_BYTES = 64
        out = self.consume(wrapper)
        self.assertEqual(out[:12], response.lines[:12])
        self.assertEqual(generic.calls, 0)
        self.assertEqual(repairs, [])

    def test_terminal_tail_and_blocked_close_obey_deadline(self):
        release, reader_done, close_done = threading.Event(), threading.Event(), threading.Event()
        class Stalled(Response):
            def iter_lines(self):
                try:
                    yield from wire(CREATED, FAILED, {"type": "error", **SCHEMA})
                    release.wait(3)
                finally:
                    reader_done.set()
            def close(self):
                release.wait(3)
                super().close()
                close_done.set()
        response = Stalled([])
        wrapper = self.make(response, failure_diagnostic_timeout=0.03)
        start = time.monotonic()
        try:
            out = self.consume(wrapper)
            wrapper.close()
            self.assertLess(time.monotonic() - start, 0.4)
            self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)
        finally:
            release.set()
        self.assertTrue(reader_done.wait(1))
        self.assertTrue(close_done.wait(1))
        self.assertEqual(response.closed, 1)

    def test_client_close_while_diagnostic_reader_blocks_is_nonblocking(self):
        release, reading, closed, finished = [threading.Event() for _ in range(4)]
        errors = []
        class Stalled(Response):
            def iter_lines(self):
                yield from wire(CREATED, FAILED)
                reading.set()
                release.wait(3)
            def close(self):
                release.wait(3)
                super().close()
                closed.set()
        response = Stalled([])
        wrapper = self.make(response, failure_diagnostic_timeout=1)
        def consume():
            try:
                list(wrapper.iter_lines())
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        threading.Thread(target=consume, daemon=True).start()
        self.assertTrue(reading.wait(1))
        try:
            start = time.monotonic()
            wrapper.close()
            self.assertLess(time.monotonic() - start, 0.3)
        finally:
            release.set()
        self.assertTrue(finished.wait(1))
        self.assertTrue(closed.wait(1))
        self.assertEqual(errors, [])

    def test_tail_exception_preserves_error_and_recovery_diagnostic(self):
        diagnostics = []
        class Broken(Response):
            def iter_lines(self):
                yield from wire(CREATED, FAILED, {"type": "error", **SCHEMA})
                raise requests.exceptions.ChunkedEncodingError("private read detail")
        out = self.consume(self.make(Broken([]), initial_after_recovery=True, on_diagnostic=diagnostics.append))
        self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)
        self.assertIn({"stage": "recovery_exception", "error_type": "ChunkedEncodingError"}, diagnostics)
        self.assertNotIn("private read detail", json.dumps(diagnostics))

    def test_tail_line_and_byte_limits_preserve_terminal_failure(self):
        class Comments(Response):
            def iter_lines(self):
                yield from wire(CREATED, FAILED, {"type": "error", **SCHEMA})
                while True:
                    self.yielded += 1
                    yield b": keepalive"
        response = Comments([])
        wrapper = self.make(response, max_retries=0)
        out = self.consume(wrapper)
        wrapper.close()
        self.assertLessEqual(response.yielded, wrapper._FAILURE_TAIL_MAX_LINES)
        self.assertEqual(decoded(out)[1]["response"]["error"], SCHEMA)
        class Large(Response):
            consumed_extra = False
            def iter_lines(self):
                yield from wire(CREATED, FAILED)
                yield b"data: " + b"x" * (RetryingResponsesResponse._FAILURE_TAIL_MAX_BYTES + 1)
                self.consumed_extra = True
                yield b"never read"
        response = Large([])
        wrapper = self.make(response, max_retries=0)
        self.assertEqual(decoded(self.consume(wrapper)), [CREATED, FAILED])
        wrapper.close()
        self.assertFalse(response.consumed_extra)


if __name__ == "__main__":
    unittest.main(verbosity=2)
