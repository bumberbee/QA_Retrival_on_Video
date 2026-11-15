from tests import _isolate  # noqa: F401
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.shared.models import Transcript
from app.features.ask import router as qa
from app.features.ask import llm


SERVER_TIMINGS = {"load_duration": 0, "prompt_eval_count": 12, "prompt_eval_duration": 200_000_000,
                  "eval_count": 4, "eval_duration": 500_000_000}


class FakeResponse:
    def __init__(self, data=None, lines=(), on_read=lambda: None):
        self.data = data
        self.lines = lines
        self.on_read = on_read
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self):
        self.on_read()
        return json.dumps(self.data).encode("utf-8")

    def __iter__(self):
        for data in self.lines:
            if callable(data):
                data = data()
            if isinstance(data, Exception):
                raise data
            yield json.dumps(data).encode("utf-8") + b"\n"


class LLMTimingTests(unittest.TestCase):
    def test_ollama_buffered_response_keeps_native_fields_and_marks_observed_ttft(self):
        response = FakeResponse({"message": {"content": " Answer "}, **SERVER_TIMINGS})
        timings = {"retrieval_seconds": 2.0}
        with patch.object(llm.urllib.request, "urlopen", return_value=response) as request, \
                patch.object(llm.time, "perf_counter", side_effect=[10.0, 13.0, 14.0]):
            self.assertEqual(llm._ollama("prompt", timings=timings), "Answer")
        self.assertFalse(json.loads(request.call_args.args[0].data)["stream"])
        self.assertEqual(timings, {"retrieval_seconds": 2.0, "total_seconds": 4.0,
                                  "ttft_seconds": 3.0, "ttft_is_buffered": True, **SERVER_TIMINGS})
        self.assertTrue(response.closed)

    def test_ollama_stream_reads_metrics_from_done_and_measures_first_nonempty_token(self):
        response = FakeResponse(lines=[{}, {"message": {"content": ""}},
                                       {"message": {"content": "First"}, "eval_count": 999},
                                       {"message": {"content": " final"}, "done": True, **SERVER_TIMINGS}])
        timings = {}
        with patch.object(llm.urllib.request, "urlopen", return_value=response), \
                patch.object(llm.time, "perf_counter", side_effect=[10.0, 11.0, 16.0]):
            self.assertEqual(list(llm._ollama_stream("prompt", timings=timings)), ["First", " final"])
        self.assertEqual(timings, {"total_seconds": 6.0, "ttft_seconds": 1.0,
                                  "ttft_is_buffered": False, **SERVER_TIMINGS})
        self.assertTrue(response.closed)

    def test_missing_server_metrics_are_null_and_failed_stream_retains_client_timings(self):
        for lines, first in (([{"message": {"content": "First"}}, TimeoutError("offline")], 1.0),
                             ([TimeoutError("offline")], None)):
            ticks = [10.0, 11.0, 14.0] if first else [10.0, 14.0]
            timings = {}
            with self.subTest(first=first), \
                    patch.object(llm.urllib.request, "urlopen", return_value=FakeResponse(lines=lines)), \
                    patch.object(llm.time, "perf_counter", side_effect=ticks):
                with self.assertRaises(llm.LLMUnavailable):
                    list(llm._ollama_stream("prompt", timings=timings))
            self.assertEqual(timings["total_seconds"], 4.0)
            self.assertEqual(timings["ttft_seconds"], first)
            self.assertTrue(all(timings[field] is None for field in SERVER_TIMINGS))

    def test_gemini_stream_and_buffered_calls_measure_client_times(self):
        for stream, first in ((False, 3.0), (True, 1.0)):
            timings = {}
            with self.subTest(stream=stream), patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                    patch.object(llm.gemini, "complete_parts", return_value="Answer"), \
                    patch.object(llm.gemini, "stream_parts", return_value=iter(["", "An", "swer"])), \
                    patch.object(llm.time, "perf_counter", side_effect=[10.0, 10.0 + first, 14.0]):
                if stream:
                    answer = "".join(llm.stream_with_evidence("Question", None, {"slides": []}, "", timings=timings))
                else:
                    answer = llm.answer_with_evidence("Question", None, {"slides": []}, "", timings=timings)
            self.assertEqual(answer, "Answer")
            self.assertEqual(timings, {"total_seconds": 4.0, "ttft_seconds": first,
                                      "ttft_is_buffered": not stream})

    def test_transcript_answers_are_instrumented(self):
        timings = {}
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                patch.object(llm.gemini, "complete", return_value="Answer"), \
                patch.object(llm.time, "perf_counter", side_effect=[10.0, 12.0, 13.0]):
            self.assertEqual(llm.answer_question(Transcript(video_id="v", segments=[]), "Question", timings=timings), "Answer")
        self.assertEqual(timings["total_seconds"], 3.0)
        self.assertEqual(timings["ttft_seconds"], 2.0)


class APITimingTests(unittest.TestCase):
    def test_retrieval_and_model_timings_reach_response_done_and_one_structured_log(self):
        api = TestClient(app)
        plan = {"conversation": None, "history": [], "query": "Question", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        clock = [100.0]

        def advance(seconds, data):
            clock[0] += seconds
            return data

        def search(*args, **kwargs):
            return advance(2.0, {"slides": [], "best_semantic": 1.0})

        response = FakeResponse({"message": {"content": "Answer"}, **SERVER_TIMINGS},
                                on_read=lambda: advance(3.0, None),
                                lines=[lambda: advance(1.0, {"message": {"content": "Answer"}}),
                                       lambda: advance(4.0, {"done": True, **SERVER_TIMINGS})])
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}), \
                patch.object(qa.retrieval, "has_index", return_value=True), \
                patch.object(qa.retrieval, "search", side_effect=search), \
                patch.object(qa, "prepare", return_value=plan), patch.object(qa, "remember"), \
                patch.object(llm.urllib.request, "urlopen", return_value=response), \
                patch.object(llm.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch("builtins.print") as log:
            buffered = api.post("/api/qa", json={"video_id": "abc123def456", "question": "Question"})
            streamed = api.post("/api/qa/stream", json={"video_id": "abc123def456", "question": "Question"})
        self.assertEqual(buffered.status_code, 200)
        buffered_timings = buffered.json()["timings"]
        self.assertEqual(buffered_timings, {"retrieval_seconds": 2.0, "total_seconds": 3.0,
                                           "ttft_seconds": 3.0, "ttft_is_buffered": True,
                                           "request_total_seconds": 5.0, **SERVER_TIMINGS})
        done = json.loads(streamed.text.splitlines()[-1])
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["timings"], {"retrieval_seconds": 2.0, "total_seconds": 5.0,
                                          "ttft_seconds": 1.0, "ttft_is_buffered": False,
                                          "request_total_seconds": 7.0, **SERVER_TIMINGS})
        self.assertEqual(log.call_count, 2)
        for call, timings, stream in zip(log.call_args_list, [buffered_timings, done["timings"]], [False, True]):
            entry = json.loads(call.args[0])
            self.assertEqual(entry["event"], "qa_answer")
            self.assertEqual(entry["timings"], timings)
            self.assertEqual(entry["stream"], stream)
            self.assertEqual(entry["status"], "ok")

    def test_failed_answer_has_one_structured_error_log(self):
        plan = {"conversation": None, "history": [], "query": "Question", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}), \
                patch.object(qa.retrieval, "has_index", return_value=True), \
                patch.object(qa, "prepare", return_value=plan), \
                patch.object(qa, "find_evidence", return_value=({"slides": []}, "")), \
                patch.object(llm.urllib.request, "urlopen", side_effect=TimeoutError("offline")), \
                patch("builtins.print") as log:
            response = TestClient(app).post("/api/qa", json={"video_id": "abc123def456", "question": "Question"})
        self.assertEqual(response.status_code, 503)
        log.assert_called_once()
        entry = json.loads(log.call_args.args[0])
        self.assertEqual(entry["status"], "error")
        self.assertIsNone(entry["timings"]["ttft_seconds"])
        self.assertGreaterEqual(entry["timings"]["total_seconds"], 0)
