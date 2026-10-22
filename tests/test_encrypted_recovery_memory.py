import copy
import json
import unittest
from unittest import mock

import requests

from ghc_api.cache import cache
from ghc_api.encrypted_recovery import EncryptionRecovery, RecoveryMemory, recovery_memory
from ghc_api.state import state
from ghc_api.utils import encrypted_content_fingerprint, remove_encrypted_content_items

from tests import test_encrypted_content_retry as cases
from tests.test_encrypted_content_retry import FakeResponse, FakeStreamResponse, FUNCTION_OUTPUT_ERROR


class EncryptedRecoveryMemoryTest(cases.EncryptedContentTestCase):
    """Exercise subsequent client requests through the real Responses route."""
    _payload = staticmethod(cases.EncryptedContentStreamRetryTest._payload)
    _failure = staticmethod(cases.EncryptedContentStreamRetryTest._failure)
    _success = staticmethod(cases.EncryptedContentStreamRetryTest._success)
    _run_stream = cases.EncryptedContentStreamRetryTest._run_stream

    def setUp(self):
        super().setUp()
        recovery_memory.clear()

    def tearDown(self):
        recovery_memory.clear()
        super().tearDown()

    def test_completed_recovery_precleans_only_old_ciphertext_on_next_request(self):
        payload = self._payload()
        payload["client_metadata"] = {"thread_id": self.id()}
        first, first_post = self._run_stream([self._failure(), self._success()], payload)
        self.assertIn(b"response.completed", first.data)
        self.assertEqual(first_post.call_count, 2)

        next_payload = copy.deepcopy(payload)
        fresh = {"type": "reasoning", "encrypted_content": "fresh-valid", "summary": []}
        next_payload["input"].append(fresh)
        original = copy.deepcopy(next_payload)
        second, second_post = self._run_stream([self._success()], next_payload)

        self.assertIn(b"response.completed", second.data)
        self.assertEqual(second_post.call_count, 1)
        sent = second_post.call_args.kwargs["json"]
        self.assertNotIn('"encrypted_content": "bad"', json.dumps(sent))
        self.assertEqual(sent["input"][-1], fresh)
        self.assertEqual(sent["input"][3]["content"], [
            {"type": "input_text", "text": "visible agent result"},
        ])
        self.assertEqual(next_payload, original)
        record = cache.get_recent_requests(1)[0]
        self.assertEqual(record["original_request_body"], original)
        self.assertEqual(record["encrypted_recovery"]["precleaned_items"], 3)
        self.assertFalse(record["encrypted_recovery"]["attempted"])

    def test_http_and_pending_stream_recoveries_learn_after_completion(self):
        for path in ("direct", "immediate", "pending"):
            with self.subTest(path=path):
                recovery_memory.clear()
                payload = self._payload()
                payload["client_metadata"] = {"thread_id": self.id() + path}
                state.sse_keepalive_interval = 0 if path == "direct" else 1
                state.responses_pre_header_grace = 1
                first, post = self._run_stream(
                    [FakeResponse(400, FUNCTION_OUTPUT_ERROR), self._success()],
                    payload, pending=path == "pending",
                )
                self.assertIn(b"response.completed", first.data)
                second, post = self._run_stream([self._success()], payload)
                self.assertEqual(post.call_count, 1)
                self.assertNotIn("encrypted_content", json.dumps(post.call_args.kwargs["json"]))
                self.assertIn(b"response.completed", second.data)

    def test_failed_incomplete_and_disconnected_recoveries_are_not_learned(self):
        for ending in (
            self._failure(),
            FakeStreamResponse([{"type": "response.incomplete", "response": {"status": "incomplete"}}]),
            FakeStreamResponse([]),
            requests.ConnectionError("synthetic secret must not appear in diagnostics"),
            FakeResponse(503, {"error": "synthetic private error"}),
        ):
            with self.subTest(ending=type(ending).__name__):
                recovery_memory.clear()
                payload = self._payload()
                payload["client_metadata"] = {"thread_id": self.id()}
                self._run_stream([self._failure(), ending], payload)
                record = cache.get_recent_requests(1)[0]
                diagnostic = record["encrypted_recovery"]
                self.assertTrue(diagnostic["attempted"])
                self.assertFalse(diagnostic["completed"])
                self.assertNotIn("synthetic", json.dumps(diagnostic))
                _, post = self._run_stream([self._success()], payload)
                self.assertIn("encrypted_content", json.dumps(post.call_args.kwargs["json"]))

    def test_diagnostics_separate_original_error_from_failed_recovery(self):
        for ending, expected_stage in (
            (self._failure(), "recovery_rejected"),
            (requests.ConnectionError("private detail"), "recovery_exception"),
            (FakeResponse(503, {"error": "private detail"}), "recovery_http_error"),
        ):
            with self.subTest(stage=expected_stage):
                self._run_stream([self._failure(), ending])
                record = cache.get_recent_requests(1)[0]
                stages = [event["stage"] for event in record["encrypted_recovery"]["events"]]
                self.assertIn("original_rejected", stages)
                self.assertIn(expected_stage, stages)
                self.assertNotIn("private detail", json.dumps(record["encrypted_recovery"]))
        self._run_stream([FakeResponse(400, FUNCTION_OUTPUT_ERROR), self._failure()])
        stages = [event["stage"] for event in cache.get_recent_requests(1)[0]["encrypted_recovery"]["events"]]
        self.assertIn("recovery_rejected", stages)

    def test_http_cleanup_failure_is_cached_on_all_entry_paths(self):
        for path in ("nonstream", "direct", "immediate", "pending"):
            with self.subTest(path=path):
                cache.cache.clear()
                payload = self._payload()
                payload["client_metadata"] = {"thread_id": self.id()}
                state.sse_keepalive_interval = 1 if path in ("immediate", "pending") else 0
                state.responses_pre_header_grace = 1
                upstream = [FakeResponse(400, FUNCTION_OUTPUT_ERROR), requests.ConnectionError("private")]
                if path == "nonstream":
                    from ghc_api.app import create_app
                    payload["stream"] = False
                    stack, post = self._patched_upstream(upstream)
                    with stack:
                        create_app().test_client().post("/v1/responses", json=payload)
                else:
                    self._run_stream(upstream, payload, pending=path == "pending")
                record = cache.get_recent_requests(1)[0]
                self.assertEqual(record["status_code"], 504)
                self.assertFalse(record["encrypted_recovery"]["completed"])
                self.assertIn({"stage": "recovery_exception", "error_type": "ConnectionError"},
                              record["encrypted_recovery"]["events"])
                self.assertEqual(record["original_request_body"], payload)

    def test_disabled_or_unidentified_requests_do_not_preclean(self):
        payload = self._payload()
        payload["client_metadata"] = {"thread_id": self.id()}
        self._run_stream([self._failure(), self._success()], payload)
        state.auto_remove_encrypted_content_on_parse_error = False
        _, post = self._run_stream([self._success()], payload)
        self.assertIn("encrypted_content", json.dumps(post.call_args.kwargs["json"]))
        state.auto_remove_encrypted_content_on_parse_error = True
        payload.pop("client_metadata")
        payload["prompt_cache_key"] = "shared-cache-key"
        self._run_stream([self._failure(), self._success()], payload)
        _, post = self._run_stream([self._success()], payload)
        self.assertIn("encrypted_content", json.dumps(post.call_args.kwargs["json"]))

    def test_nonstream_recovery_learns_only_completed_response(self):
        from ghc_api.app import create_app
        payload = self._payload()
        payload.update(stream=False, client_metadata={"thread_id": self.id()})
        success = {"status": "completed", "output": [], "usage": {}}
        stack, post = self._patched_upstream([
            FakeResponse(400, FUNCTION_OUTPUT_ERROR), FakeResponse(200, success), FakeResponse(200, success),
        ])
        with stack:
            client = create_app().test_client()
            self.assertEqual(client.post("/v1/responses", json=payload).status_code, 200)
            self.assertEqual(client.post("/v1/responses", json=payload).status_code, 200)
        self.assertEqual(post.call_count, 3)
        self.assertNotIn("encrypted_content", json.dumps(post.call_args.kwargs["json"]))


class RecoveryMemoryBoundsTest(unittest.TestCase):
    def tearDown(self):
        recovery_memory.clear()

    def test_scope_isolated_by_user_upstream_account_model_and_thread(self):
        base = {"model": "model-a", "client_metadata": {"thread_id": "thread-a"},
                "input": [{"type": "reasoning", "encrypted_content": "old", "summary": []}]}
        context = EncryptionRecovery(base, "user-a", "upstream-a", "individual", True)
        context.recover(base)
        context.complete()
        self.assertEqual(context.prepare(base)["input"], [])
        for user, upstream, account, model, thread in (
            ("user-b", "upstream-a", "individual", "model-a", "thread-a"),
            ("user-a", "upstream-b", "individual", "model-a", "thread-a"),
            ("user-a", "upstream-a", "business", "model-a", "thread-a"),
            ("user-a", "upstream-a", "individual", "model-b", "thread-a"),
            ("user-a", "upstream-a", "individual", "model-a", "thread-b"),
        ):
            payload = {**base, "model": model, "client_metadata": {"thread_id": thread}}
            other = EncryptionRecovery(payload, user, upstream, account, True)
            self.assertEqual(other.prepare(payload), payload)

    def test_expiry_and_capacity_are_bounded(self):
        clock = mock.Mock(return_value=0)
        memory = RecoveryMemory(capacity=2, ttl=10, clock=clock)
        memory.remember("scope", {"a", "b"})
        memory.remember("other", {"c"})
        self.assertEqual(memory.lookup("scope"), {"b"})
        self.assertEqual(memory.lookup("other"), {"c"})
        clock.return_value = 10
        self.assertEqual(memory.lookup("scope"), set())
        self.assertEqual(memory.lookup("other"), set())

    def test_selective_cleanup_keeps_new_ciphertext_inside_same_message(self):
        item = {"type": "agent_message", "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "encrypted_content", "encrypted_content": "old"},
            {"type": "encrypted_content", "encrypted_content": "new"},
        ]}
        cleaned, count = remove_encrypted_content_items([item], {encrypted_content_fingerprint("old")})
        self.assertEqual(count, 1)
        self.assertEqual(cleaned, [{**item, "content": [item["content"][0], item["content"][2]]}])
        self.assertEqual(len(item["content"]), 3)
