from tests import _isolate  # noqa: F401
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.features.agent import graph
from app.features.agent.tools import BudgetLimits
from app.features.ask import llm
from app.features.ask import router as qa
from app.features.decisions import decide

VIDEO = "abc123def456"


def search_result(best_semantic=0.9, *, temporal=False, title="Xác suất", best_rerank=None):
    chunk = {"chunk_id": "c1", "segment_id": f"{VIDEO}:s1", "start_sec": 65, "end_sec": 90,
             "text": "Xác suất có điều kiện là xác suất của A khi biết B.", "source_type": "speech",
             "title": "", "priority": 0, "selection_reason": "matched"}
    found = {"slides": [{"video_id": VIDEO, "segment_id": f"{VIDEO}:s1", "start_sec": 60, "end_sec": 120,
                         "title": title, "ocr_text": "", "speech_text": "", "figures": [], "score": 0.03,
                         "evidence_candidates": [chunk]}],
             "temporal": temporal, "best_semantic": best_semantic, "question": "q", "retrieval_options": {}}
    if best_rerank is not None:
        found["best_rerank"] = best_rerank
    return found


class Searcher:
    """Stands in for retrieval.search; one prepared result per query (or the same for all)."""

    def __init__(self, *results, by_query=None):
        self.results, self.by_query, self.calls = list(results), by_query or {}, []

    def __call__(self, video_id, question, *positional, **options):
        self.calls.append((question, options))
        if question in self.by_query:
            return self.by_query[question]
        return self.results[min(len(self.calls), len(self.results)) - 1]


def run(question, searcher=None, **options):
    return graph.run(question, video_id=VIDEO, searcher=searcher or Searcher(search_result()), **options)


class PathTests(unittest.TestCase):
    def test_answerable_question_reaches_generation_with_cited_context(self):
        searcher = Searcher(search_result(0.9))
        result = run("Xác suất có điều kiện là gì?", searcher, prefer=frozenset({"s9"}), current_time=30)
        self.assertEqual((result.action, result.refusal, result.text), ("answer", False, None))
        self.assertEqual(result.trace["states"],
                         ["S1_understand", "S2_plan", "S3_search", "S4_grade", "S5_answer"])
        self.assertEqual(result.trace["decision"]["qtype"], "definition")
        self.assertEqual(result.trace["decision"]["decided_by"], "rules")
        self.assertEqual(result.trace["q8"], [{"round": 1, "label": "enough", "p": 0.9, "backend": "similarity"}])
        self.assertTrue(result.trace["verified"])
        self.assertIn("[E1]", result.context)
        self.assertEqual(result.found["evidence_map"]["E1"]["t_start"], 65)
        self.assertEqual(result.found["slides"][0]["segment_id"], f"{VIDEO}:s1")
        question, options = searcher.calls[0]
        self.assertEqual(options["prefer_segments"], frozenset({"s9"}))
        self.assertEqual(options["current_time"], 30.0)
        self.assertEqual(result.trace["budget"]["search_rounds"], 1)
        self.assertEqual(result.trace["budget"]["llm_calls"], 0)

    def test_chitchat_and_clarification_answer_without_searching(self):
        searcher = Searcher(search_result())
        hello = run("Chào bạn", searcher)
        thanks = run("Cảm ơn nhé", searcher)
        pointer = run("Cái này là gì?", searcher)
        self.assertEqual((hello.action, hello.text), ("direct", graph.REPLIES["hello"]))
        self.assertEqual(thanks.text, graph.REPLIES["thanks"])
        self.assertEqual((pointer.action, pointer.trace["plan"]), ("direct", "clarify"))
        self.assertIn("slide nào", pointer.text)
        self.assertFalse(hello.refusal or thanks.refusal or pointer.refusal)
        self.assertEqual(searcher.calls, [])
        self.assertEqual(hello.trace["states"], ["S1_understand", "S2_plan"])

    def test_pointer_with_a_playing_video_is_searched(self):
        searcher = Searcher(search_result(0.5, temporal=True))
        result = run("Cái này là gì?", searcher, current_time=42)
        self.assertEqual(result.action, "answer")
        self.assertEqual(result.trace["q8"][0]["label"], "enough")   # the watched slide answers it

    def test_long_sentence_with_a_greeting_word_is_a_question(self):
        result = run("Chào hỏi trong văn hóa giao tiếp được thầy trình bày ra sao")
        self.assertEqual((result.action, result.trace["plan"]), ("answer", "plan_explanation"))

    def test_rule_out_of_scope_refuses_before_searching(self):
        searcher = Searcher(search_result())
        result = run("Giá bitcoin hôm nay", searcher)
        self.assertEqual((result.action, result.refusal, result.text), ("direct", True, llm.REFUSAL))
        self.assertEqual(result.trace["states"], ["S1_understand", "S2_plan", "S7_refuse"])
        self.assertEqual(searcher.calls, [])

    def test_irrelevant_evidence_refuses_with_the_closest_part(self):
        result = run("Thủ đô của nước Pháp nằm ở đâu vậy", Searcher(search_result(0.78)))
        self.assertEqual((result.action, result.refusal), ("direct", True))
        self.assertEqual(result.text, f"{llm.REFUSAL} Phần gần nhất trong bài là «Xác suất» tại [01:05].")
        self.assertEqual(result.trace["states"][-1], "S7_refuse")
        untitled = run("Thủ đô của nước Pháp nằm ở đâu vậy", Searcher(search_result(0.78, title="")))
        self.assertEqual(untitled.text, f"{llm.REFUSAL} Phần gần nhất mình tìm được ở [01:05].")

    def test_weak_evidence_still_answers_but_is_not_verified(self):
        result = run("Thầy giải thích vì sao dùng công thức đó", Searcher(search_result(0.83)))
        self.assertEqual(result.action, "answer")
        self.assertEqual(result.trace["q8"][0]["label"], "insufficient")
        self.assertFalse(result.trace["verified"])


class RetryAndBudgetTests(unittest.TestCase):
    def test_a_weak_first_round_retries_with_the_typed_question(self):
        searcher = Searcher(by_query={"đầy đủ hơn": search_result(0.82), "Vì sao vậy thầy": search_result(0.91)})
        result = run("Vì sao vậy thầy", searcher, query="đầy đủ hơn", previous_question="Bayes là gì?")
        self.assertEqual([call[0] for call in searcher.calls], ["đầy đủ hơn", "Vì sao vậy thầy"])
        self.assertEqual([entry["label"] for entry in result.trace["q8"]], ["insufficient", "enough"])
        self.assertEqual(result.found["best_semantic"], 0.91)
        self.assertEqual(result.trace["states"].count("S3_search"), 2)

    def test_the_better_round_is_kept_when_none_is_enough(self):
        searcher = Searcher(by_query={"a b c": search_result(0.84), "Vì sao vậy thầy": search_result(0.81)})
        result = run("Vì sao vậy thầy", searcher, query="a b c")
        self.assertEqual((result.action, result.found["best_semantic"]), ("answer", 0.84))

    def test_search_budget_and_failures_fall_back_to_the_baseline(self):
        searcher = Searcher(by_query={"a b c": search_result(0.82), "Vì sao vậy thầy": search_result(0.91)})
        limited = run("Vì sao vậy thầy", searcher, query="a b c",
                      limits=BudgetLimits(max_search_rounds=1))
        self.assertEqual(len(searcher.calls), 1)
        self.assertEqual(limited.found["best_semantic"], 0.82)
        self.assertEqual(limited.trace["tools"][1]["error"], "search_budget: maximum search rounds reached")

        def broken(*args, **kwargs):
            raise RuntimeError("index missing")

        failed = run("Xác suất có điều kiện là gì?", broken)
        self.assertEqual((failed.action, failed.found), ("answer", {}))
        self.assertIn("index missing", failed.trace["tools"][0]["error"])


class GradeTests(unittest.TestCase):
    def test_reranker_scores_take_precedence_over_similarity(self):
        self.assertEqual(graph.grade(search_result(0.5, best_rerank=0.9)),
                         {"label": "enough", "p": 0.9, "backend": "reranker"})
        self.assertEqual(graph.grade(search_result(0.95, best_rerank=0.1))["label"], "irrelevant")

    def test_similarity_thresholds_and_missing_scores(self):
        labels = [decide("Q8", {"best_semantic": score}, backend="similarity").label
                  for score in (0.85, 0.849, 0.801, 0.80, -0.2)]
        self.assertEqual(labels, ["enough", "insufficient", "insufficient", "irrelevant", "irrelevant"])
        self.assertEqual(decide("Q8", {"retrieval": {"best_semantic": 0.9}}, backend="similarity").label, "enough")
        missing = decide("Q8", {}, backend="similarity")
        self.assertEqual((missing.label, missing.backend), ("insufficient", "fallback"))


class AgentAPITests(unittest.TestCase):
    def ask(self, question, *, agent="on", result=None, video_ids=None, raw="Trả lời [E1]."):
        api = TestClient(app)
        plan = {"conversation": None, "history": [], "query": question, "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        payload = {"video_id": VIDEO, "question": question, "current_time": 12, "video_ids": video_ids}
        search = Searcher(result or search_result(0.9))
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "LECTURE_AGENT": agent}), \
                patch.object(qa.retrieval, "has_index", return_value=True), \
                patch.object(qa.retrieval, "search", search), \
                patch.object(qa.retrieval, "load_index", return_value=type("Index", (), {"model": "m"})()), \
                patch.object(qa.retrieval, "default_encoder", return_value=None), \
                patch.object(qa, "prepare", return_value=plan), patch.object(qa, "remember") as remember, \
                patch.object(llm.gemini, "complete_parts", return_value=raw) as complete, \
                patch.object(llm.gemini, "stream_parts", side_effect=lambda parts: iter([raw])) as stream:
            body = api.post("/api/qa", json=payload).json()
            events = [json.loads(line) for line in api.post("/api/qa/stream", json=payload).text.splitlines()]
        return body, events, complete.call_count + stream.call_count, remember, search

    def test_generated_answer_carries_the_trace(self):
        body, events, llm_calls, _, _ = self.ask("Xác suất có điều kiện là gì?")
        self.assertEqual(body["answer"], "Trả lời [01:05].")
        self.assertFalse(body["refusal"])
        self.assertEqual(body["agent"]["states"][-1], "S5_answer")
        self.assertEqual(body["sources"][0]["segment_id"], f"{VIDEO}:s1")
        self.assertEqual([s["id"] for s in body["cited_sources"]], ["E1"])
        self.assertEqual(llm_calls, 2)
        done = events[-1]
        self.assertEqual((done["type"], done["refusal"], done["agent"]["plan"]), ("done", False, "plan_definition"))
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), body["answer"])

    def test_refusal_and_template_skip_the_model(self):
        body, events, llm_calls, remember, _ = self.ask("Thủ đô của nước Pháp nằm ở đâu vậy",
                                                        result=search_result(0.7))
        self.assertTrue(body["refusal"])
        self.assertTrue(body["answer"].startswith(llm.REFUSAL))
        self.assertEqual(len(body["sources"]), 1)
        self.assertEqual(llm_calls, 0)
        self.assertEqual([e["type"] for e in events], ["meta", "sources", "delta", "done"])
        self.assertEqual(events[2]["text"], body["answer"])
        self.assertTrue(events[-1]["refusal"])
        self.assertEqual(remember.call_args.args[3], body["answer"])
        body, events, llm_calls, _, search = self.ask("Chào bạn")
        self.assertEqual((body["answer"], body["sources"], llm_calls), (graph.REPLIES["hello"], [], 0))
        self.assertEqual(search.calls, [])

    def test_switch_off_and_multi_video_use_the_baseline(self):
        body, events, llm_calls, _, _ = self.ask("Chào bạn", agent="")
        self.assertEqual((body["answer"], body["agent"], llm_calls), ("Trả lời [01:05].", None, 2))
        self.assertNotIn("agent", events[-1])
        with patch.object(qa.retrieval, "search_many", return_value=search_result(0.9)):
            body, _, _, _, _ = self.ask("Chào bạn", video_ids=[VIDEO, "abc123def457"])
        self.assertIsNone(body["agent"])

    def test_agent_error_falls_back_to_the_baseline(self):
        with patch.object(graph, "run", side_effect=RuntimeError("boom")), self.assertLogs(qa.logger, level="ERROR"):
            body, _, llm_calls, _, _ = self.ask("Xác suất có điều kiện là gì?")
        self.assertEqual((body["answer"], body["agent"], llm_calls), ("Trả lời [01:05].", None, 2))


if __name__ == "__main__":
    unittest.main()
