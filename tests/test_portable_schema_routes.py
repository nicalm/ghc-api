import copy
import json
from unittest import mock

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.encrypted_recovery import recovery_memory
from ghc_api.portable_schema import PortableSchemaMemory, PortableSchemaRecovery
from ghc_api.state import state
from tests.test_encrypted_content_retry import (
    EncryptedContentTestCase, EncryptedContentStreamRetryTest,
    FakeResponse, FakeStreamResponse, FUNCTION_OUTPUT_ERROR,
)


ERROR = {"code": "invalid_request_body", "message":
         "Portable encrypted function call arguments require their original tool declaration or saved encryption schema."}
DECLARATION = {"type": "additional_tools", "role": "developer", "id": "at_test", "tools": [
    {"type": "namespace", "name": "collaboration", "description": "Synthetic namespace", "tools": [
        {"type": "function", "name": "send_message", "parameters": {"type": "object", "properties": {
            "message": {"type": "string", "encrypted": True}, "target": {"type": "string"}}}}]}]}
CALL = {"type": "function_call", "namespace": "collaboration", "name": "send_message",
        "call_id": "call_test", "arguments": '{"message":"opaque-secret-marker","target":"test"}'}
OUTPUT = {"type": "function_call_output", "call_id": "call_test", "output": "preserve the original plaintext"}


def success(output=None):
    return FakeStreamResponse([{"type": "response.completed", "response": {
        "status": "completed", "output": output or [], "usage": {"input_tokens": 1, "output_tokens": 1}}}])


def failure(shape="trailing"):
    if shape == "nested":
        return FakeStreamResponse([{"type": "response.failed", "response": {"output": [], "error": ERROR}}])
    if shape == "envelope":
        return FakeStreamResponse([{"type": "error", "error": ERROR}])
    return FakeStreamResponse([
        {"type": "response.created", "response": {}},
        {"type": "response.failed", "response": {"output": [], "error": None}},
        {"type": "error", **ERROR}])


class PortableSchemaRoutesTest(EncryptedContentTestCase):
    _run_stream = EncryptedContentStreamRetryTest._run_stream

    def setUp(self):
        super().setUp()
        recovery_memory.clear()
        self.addCleanup(recovery_memory.clear)
        self.memory = PortableSchemaMemory(path=None)
        self.patcher = mock.patch("ghc_api.portable_schema.portable_schema_memory", self.memory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def payload(self, declared=False):
        return {"model": "gpt-test", "stream": True, "client_metadata": {"thread_id": self.id()},
                "tool_choice": "auto" if declared else "none",
                "input": [copy.deepcopy(DECLARATION)] if declared else copy.deepcopy([CALL, OUTPUT])}

    def seed(self):
        response, post = self._run_stream([success([copy.deepcopy(CALL)])], self.payload(declared=True))
        self.assertIn(b"response.completed", response.data)
        self.assertEqual(post.call_count, 1)

    def assert_history(self, sent, original):
        self.assertEqual(sent["tool_choice"], "none")
        self.assertEqual(sent["input"][1:], original["input"])
        self.assertEqual(sent["input"][0]["type"], "additional_tools")
        self.assertEqual(sent["input"][0]["tools"], DECLARATION["tools"])

    def test_normal_turn_learns_output_call_and_compaction_restores_it(self):
        self.seed()
        payload = self.payload()
        payload["tool_choice"] = "auto"
        original = copy.deepcopy(payload)
        stack, post = self._patched_upstream([success()])
        with stack:
            response = create_app().test_client().post("/v1/responses", json=payload, buffered=True,
                headers={"X-Codex-Turn-Metadata": json.dumps({"request_kind": "compaction"})})
        self.assertIn(b"response.completed", response.data)
        self.assertEqual(post.call_count, 1)
        self.assert_history(post.call_args.kwargs["json"], original)
        self.assertEqual(payload, original)
        record = cache.get_recent_requests(1)[0]
        self.assertEqual(record["original_request_body"], original)
        self.assertEqual(record["portable_schema_recovery"]["restored_declarations"], 1)
        self.assertNotIn("opaque-secret-marker", json.dumps(record["portable_schema_recovery"]))

    def test_http_and_sse_recovery_across_all_entry_paths(self):
        for path in ("direct", "immediate", "pending", "nonstream"):
            for shape in ("http", "trailing", "nested", "envelope"):
                if path == "nonstream" and shape != "http":
                    continue
                with self.subTest(path=path, shape=shape):
                    self.seed()
                    payload = self.payload()
                    initial = FakeResponse(400, {"error": ERROR}) if shape == "http" else failure(shape)
                    state.sse_keepalive_interval = 1 if path in ("immediate", "pending") else 0
                    state.responses_pre_header_grace = 1
                    # Exercise reactive recovery separately from proactive preparation.
                    with mock.patch.object(PortableSchemaRecovery, "prepare", lambda self, value: value):
                        if path == "nonstream":
                            payload["stream"] = False
                            stack, post = self._patched_upstream([initial, FakeResponse(200, {
                                "status": "completed", "output": [], "usage": {}})])
                            with stack:
                                response = create_app().test_client().post("/v1/responses", json=payload)
                            self.assertEqual(response.status_code, 200)
                        else:
                            response, post = self._run_stream([initial, success()], payload, pending=path == "pending")
                            self.assertIn(b"response.completed", response.data)
                    self.assertEqual(post.call_count, 2)
                    self.assert_history(post.call_args.kwargs["json"], payload)
                    self.assertTrue(cache.get_recent_requests(1)[0]["portable_schema_recovery"]["attempted"])

    def test_schema_and_old_attachment_errors_each_have_their_own_recovery(self):
        for schema_first in (True, False):
            with self.subTest(schema_first=schema_first):
                recovery_memory.clear()
                self.seed()
                payload = self.payload()
                payload["input"].append({"type": "agent_message", "content": [
                    {"type": "input_text", "text": "keep mixed plaintext"},
                    {"type": "encrypted_content", "encrypted_content": "bad"}]})
                errors = [failure(), FakeResponse(400, FUNCTION_OUTPUT_ERROR)]
                if not schema_first:
                    errors.reverse()
                with mock.patch.object(PortableSchemaRecovery, "prepare", lambda self, value: value):
                    response, post = self._run_stream(errors + [success()], payload)
                self.assertIn(b"response.completed", response.data)
                self.assertEqual(post.call_count, 3)
                sent = post.call_args.kwargs["json"]
                self.assertIn(CALL, sent["input"])
                self.assertIn(OUTPUT, sent["input"])
                self.assertEqual(sent["input"][-1]["content"], [{"type": "input_text", "text": "keep mixed plaintext"}])

    def test_schema_recovery_is_once_and_never_replays_after_output(self):
        self.seed()
        with mock.patch.object(PortableSchemaRecovery, "prepare", lambda self, value: value):
            response, post = self._run_stream([failure(), failure()], self.payload())
        self.assertEqual(post.call_count, 2)
        self.assertIn(b"saved encryption schema", response.data)
        after_output = FakeStreamResponse([
            {"type": "response.output_text.delta", "delta": "already visible"},
            {"type": "error", **ERROR}])
        with mock.patch.object(PortableSchemaRecovery, "prepare", lambda self, value: value):
            response, post = self._run_stream([after_output], self.payload())
        self.assertEqual(post.call_count, 1)
        self.assertIn(b"already visible", response.data)

    def test_unknown_calls_are_not_repaired_using_another_call_or_task(self):
        self.seed()
        for change in ("thread", "arguments"):
            payload = self.payload()
            if change == "thread":
                payload["client_metadata"]["thread_id"] += "other"
            else:
                payload["input"][0]["arguments"] = '{"message":"different"}'
            response, post = self._run_stream([failure()], payload)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["json"], payload)
            self.assertIn(b"saved encryption schema", response.data)

    def test_recovery_does_not_treat_non_400_as_a_request_schema_failure(self):
        self.seed()
        for status in (403, 500):
            with self.subTest(status=status):
                payload = self.payload()
                payload["input"].append({"type": "reasoning", "encrypted_content": "bad"})
                with mock.patch.object(PortableSchemaRecovery, "prepare", lambda self, value: value):
                    response, post = self._run_stream([
                        failure(), FakeResponse(status, FUNCTION_OUTPUT_ERROR)], payload)
                self.assertEqual(post.call_count, 2)
                self.assertIn(b"saved encryption schema", response.data)
