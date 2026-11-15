from tests import _isolate  # noqa: F401  (must come before any app import)
import http.client
import json
import os
import unittest
import urllib.error
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.features.ask import router as qa
from app.features.ask import llm


class FakeStream:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line


DELTA = b'{"message":{"content":"partial"},"done":false}\n'
DONE = b'{"done":true}\n'
STREAM_FAILURES = (
    TimeoutError("read timeout"), ConnectionResetError("reset"),
    urllib.error.URLError("offline"), http.client.IncompleteRead(b"truncated"),
    b'{"broken"', b"\xff\n", b"[]\n", b'{"message":"invalid"}\n',
    b'{"message":{"content":123}}\n', b'{"done":"true"}\n',
    b'{"error":"model unavailable"}\n',
)


class LLMReliabilityTests(unittest.TestCase):
    def test_transport_and_malformed_ndjson_errors_become_llm_unavailable(self):
        for failure in STREAM_FAILURES:
            response = FakeStream([DELTA, failure])
            pieces = []
            with self.subTest(failure=failure), patch.object(llm.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(llm.LLMUnavailable):
                    for piece in llm._ollama_stream("prompt"):
                        pieces.append(piece)
                self.assertEqual(pieces, ["partial"])
                self.assertTrue(response.closed)

    def test_connection_open_timeout_and_transport_errors_are_translated(self):
        for failure in (TimeoutError("connect timeout"), urllib.error.URLError("offline"),
                        ConnectionResetError("reset"), http.client.RemoteDisconnected("closed")):
            with self.subTest(failure=failure), patch.object(llm.urllib.request, "urlopen", side_effect=failure):
                with self.assertRaises(llm.LLMUnavailable):
                    list(llm._ollama_stream("prompt"))

    def test_premature_eof_and_empty_stream_are_errors(self):
        for lines in ([], [b"\n"], [DELTA], [DONE]):
            response = FakeStream(lines)
            with self.subTest(lines=lines), patch.object(llm.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(llm.LLMUnavailable):
                    list(llm._ollama_stream("prompt"))
            self.assertTrue(response.closed)

    def test_success_requires_done_and_keeps_final_message_content(self):
        response = FakeStream([b"\n", DELTA, b'{"message":{"content":" final"},"done":true}\n', b"invalid"])
        with patch.dict(os.environ, {"OLLAMA_TIMEOUT_SECONDS": "4"}), \
                patch.object(llm.urllib.request, "urlopen", return_value=response) as open_request:
            self.assertEqual(list(llm._ollama_stream("prompt")), ["partial", " final"])
        self.assertEqual(open_request.call_args.kwargs["timeout"], 4)
        self.assertTrue(response.closed)

    def test_gemini_rewrite_and_summary_forward_generation_limits_and_timeout(self):
        model = Mock()
        model.generate_content.return_value.text = "A complete standalone question?"
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                patch.object(llm.gemini, "_ensure_configured"), \
                patch.object(llm.gemini.genai, "GenerativeModel", return_value=model):
            self.assertEqual(llm.rewrite_question("What about it?", [{"question": "Prior?", "answer": "Answer"}], timeout=2.5),
                             "A complete standalone question?")
            self.assertIsNotNone(llm.summarize("", [{"question": "Prior?", "answer": "Answer"}], timeout=7))
        rewrite, summary = model.generate_content.call_args_list
        self.assertEqual(rewrite.kwargs["generation_config"], {"max_output_tokens": 60, "temperature": 0.0})
        self.assertEqual(rewrite.kwargs["request_options"], {"timeout": 2.5})
        self.assertEqual(summary.kwargs["generation_config"], {"max_output_tokens": 200, "temperature": 0.2})
        self.assertEqual(summary.kwargs["request_options"], {"timeout": 7})


class QAStreamTests(unittest.TestCase):
    def setUp(self):
        self.api = TestClient(app)
        self.payload = {"video_id": "abc123def456", "question": "Question?"}
        plan = {"conversation": None, "history": [], "query": "Question?", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        for patcher in (patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "ADMIN_TOKEN": "demo-admin"}),
                        patch.object(qa.retrieval, "has_index", return_value=True),
                        patch.object(qa, "prepare", return_value=plan),
                        patch.object(qa, "find_evidence", return_value=({"slides": []}, "evidence"))):
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(qa, "remember")
        self.remember = patcher.start()
        self.addCleanup(patcher.stop)

    def events(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/x-ndjson"))
        return [json.loads(line) for line in response.text.splitlines()]

    def test_all_ollama_read_and_format_failures_end_with_an_error_event(self):
        for failure in STREAM_FAILURES:
            response = FakeStream([DELTA, failure])
            with self.subTest(failure=failure), patch.object(llm.urllib.request, "urlopen", return_value=response):
                events = self.events(self.api.post("/api/qa/stream", json=self.payload))
            self.assertEqual([event["type"] for event in events], ["meta", "sources", "delta", "error"])
            self.assertTrue(events[-1]["detail"])
            self.assertTrue(response.closed)
        self.remember.assert_not_called()

    def test_premature_stream_termination_ends_with_error(self):
        with patch.object(llm.urllib.request, "urlopen", return_value=FakeStream([DELTA])):
            events = self.events(self.api.post("/api/qa/stream", json=self.payload))
        self.assertEqual(events[-1]["type"], "error")
        self.remember.assert_not_called()

    def test_success_stays_public_and_ends_with_done(self):
        with patch.object(llm.urllib.request, "urlopen", return_value=FakeStream([DELTA, DONE])):
            events = self.events(self.api.post("/api/qa/stream", json=self.payload))
        self.assertEqual([event["type"] for event in events], ["meta", "sources", "delta", "done"])
        self.assertEqual(events[2]["text"], "partial")
        self.remember.assert_called_once()

    def test_post_generation_history_failure_also_ends_with_error(self):
        self.remember.side_effect = OSError("history database down")
        with patch.object(llm.urllib.request, "urlopen", return_value=FakeStream([DELTA, DONE])), \
                self.assertLogs(qa.logger, level="ERROR"):
            events = self.events(self.api.post("/api/qa/stream", json=self.payload))
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("database down", events[-1]["detail"])
