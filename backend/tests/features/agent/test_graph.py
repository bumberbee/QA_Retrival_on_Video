from tests import _isolate  # noqa: F401
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.features.agent import graph, verify
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
        result = run("Thầy giải thích vì sao dùng công thức đó", Searcher(search_result(0.845)))
        self.assertEqual(result.action, "answer")
        self.assertEqual(result.trace["q8"][0]["label"], "insufficient")
        self.assertFalse(result.trace["verified"])


class RetryAndBudgetTests(unittest.TestCase):
    def test_a_weak_first_round_retries_with_the_typed_question(self):
        searcher = Searcher(by_query={"đầy đủ hơn": search_result(0.845), "Vì sao vậy thầy": search_result(0.91)})
        result = run("Vì sao vậy thầy", searcher, query="đầy đủ hơn", previous_question="Bayes là gì?")
        self.assertEqual([call[0] for call in searcher.calls], ["đầy đủ hơn", "Vì sao vậy thầy"])
        self.assertEqual([entry["label"] for entry in result.trace["q8"]], ["insufficient", "enough"])
        self.assertEqual(result.found["best_semantic"], 0.91)
        self.assertEqual(result.trace["states"].count("S3_search"), 2)

    def test_the_better_round_is_kept_when_none_is_enough(self):
        searcher = Searcher(by_query={"a b c": search_result(0.848), "Vì sao vậy thầy": search_result(0.842)})
        result = run("Vì sao vậy thầy", searcher, query="a b c")
        self.assertEqual((result.action, result.found["best_semantic"]), ("answer", 0.848))

    def test_search_budget_and_failures_fall_back_to_the_baseline(self):
        searcher = Searcher(by_query={"a b c": search_result(0.845), "Vì sao vậy thầy": search_result(0.91)})
        limited = run("Vì sao vậy thầy", searcher, query="a b c",
                      limits=BudgetLimits(max_search_rounds=1))
        self.assertEqual(len(searcher.calls), 1)
        self.assertEqual(limited.found["best_semantic"], 0.845)
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
                  for score in (0.85, 0.849, 0.841, 0.84, -0.2)]
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
        self.assertEqual(body["agent"]["states"][-2:], ["S5_answer", "S6_check"])
        self.assertEqual(body["agent"]["verification"], {"label": "verified", "reasons": [],
                                                         "numeric_claims": 0, "numeric_checked": 0})
        self.assertEqual(body["agent"]["budget"]["llm_calls"], 1)
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


CARD = json.loads((Path(__file__).resolve().parents[1] / "ask/fixtures/card_probability.json")
                  .read_text(encoding="utf-8"))


class VerifyTests(unittest.TestCase):
    def test_real_card_answer_is_a_mismatch_and_its_correction_is_not(self):
        wrong = verify.check(CARD["wrong_explanation"], {})
        self.assertTrue(wrong["mismatches"])
        self.assertTrue(all(row["computed"] for row in wrong["mismatches"]))
        self.assertEqual(verify.check(CARD["corrected_explanation"], {})["mismatches"], [])
        text = verify.hint(wrong)
        self.assertIn("không phải", text)
        self.assertIn(wrong["mismatches"][0]["computed"], text)

    def test_verdict_names_every_reason(self):
        clean = {"claims": [{"status": "ok"}, {"status": "unparsed"}], "mismatches": []}
        cited = {"cited_sources": [{"id": "E1"}], "invalid_citations": []}
        self.assertEqual(verify.verdict(clean, cited, evidence_enough=True),
                         {"label": "verified", "reasons": [], "numeric_claims": 2, "numeric_checked": 1})
        bad = verify.verdict({"claims": [{"status": "mismatch"}], "mismatches": [{}]},
                             {"cited_sources": [], "invalid_citations": ["E9"]}, evidence_enough=False)
        self.assertEqual((bad["label"], bad["reasons"]), ("unverified", [
            "weak_evidence", "numeric_mismatch", "invalid_citation", "no_citation"]))
        inferred = verify.verdict(clean, {"cited_sources": [{"id": "E1", "inferred": True}]}, evidence_enough=True)
        self.assertEqual(inferred["reasons"], ["inferred_citation"])

    def test_citations_after_a_number_do_not_hide_the_claim(self):
        for answer in ("Ta có 2 + 2 = 5 [01:05].", "Ta có 2 + 2 = 5 [E1].", "Ta có 2 + 2 = 5 [1:01:05] nhé."):
            with self.subTest(answer=answer):
                report = verify.check(answer, {})
                self.assertEqual([(r["lhs"], r["rhs"], r["computed"]) for r in report["mismatches"]],
                                 [("2 + 2", "5", "4")])

    def test_checker_fault_is_reported_not_raised(self):
        with patch.object(verify.numcheck, "check_answer", side_effect=RuntimeError("bad")):
            self.assertEqual(verify.check("2 + 2 = 5", {}), {"claims": [], "mismatches": [],
                                                            "error": "RuntimeError: bad"})


class RegenerationAPITests(unittest.TestCase):
    WRONG, RIGHT = "Ta có 2 + 2 = 5 [E1].", "Ta có 2 + 2 = 4 [E1]."

    def ask(self, answers, limits=None):
        """answers: the model's first answer and, when given, its answer to a regeneration."""
        api = TestClient(app)
        plan = {"conversation": None, "history": [], "query": "q", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        payload = {"video_id": VIDEO, "question": "Hai cộng hai bằng bao nhiêu?"}
        prompts = []

        def complete(parts):
            prompts.append(parts[0])
            return answers[-1] if "phép tính sai" in parts[0] else answers[0]   # the hint marks a regeneration

        real_run = graph.run
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "LECTURE_AGENT": "on"}),                 patch.object(qa.retrieval, "has_index", return_value=True),                 patch.object(qa.retrieval, "search", Searcher(search_result(0.9))),                 patch.object(qa.retrieval, "load_index", return_value=type("Index", (), {"model": "m"})()),                 patch.object(qa.retrieval, "default_encoder", return_value=None),                 patch.object(graph, "run", lambda *a, **kw: real_run(*a, limits=limits, **kw)),                 patch.object(qa, "prepare", return_value=plan), patch.object(qa, "remember") as remember,                 patch.object(llm.gemini, "complete_parts", side_effect=complete),                 patch.object(llm.gemini, "stream_parts", side_effect=lambda parts: iter([answers[0]])):
            body = api.post("/api/qa", json=payload).json()
            first = list(prompts)
            del prompts[:]
            events = [json.loads(line) for line in api.post("/api/qa/stream", json=payload).text.splitlines()]
        return body, first, events, list(prompts), remember

    def test_wrong_arithmetic_is_regenerated_once_with_the_exact_value(self):
        body, prompts, events, stream_prompts, remember = self.ask([self.WRONG, self.RIGHT])
        self.assertEqual(body["answer"], "Ta có 2 + 2 = 4 [01:05].")
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("phép tính sai", prompts[0])
        self.assertIn("2 + 2 = 4, không phải 5", prompts[1])
        trace = body["agent"]
        self.assertEqual(trace["states"][-2:], ["S6_check", "S6_regenerate"])
        self.assertEqual([len(r["mismatches"]) for r in trace["s6"]], [1, 0])
        self.assertEqual(trace["verification"]["label"], "verified")
        self.assertEqual(trace["budget"]["llm_calls"], 2)
        # Stream: the wrong text was shown, then replaced through the done event.
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), "Ta có 2 + 2 = 5 [01:05].")
        self.assertEqual(events[-1]["answer"], body["answer"])
        self.assertEqual(len(stream_prompts), 1)          # only the regeneration is a buffered call
        self.assertEqual(remember.call_args.args[3], body["answer"])

    def test_a_still_wrong_regeneration_keeps_the_first_answer_unverified(self):
        body, prompts, events, _, _ = self.ask([self.WRONG, "Ta có 2 + 2 = 6 [E1]."])
        self.assertEqual(body["answer"], "Ta có 2 + 2 = 5 [01:05].")
        self.assertEqual(len(prompts), 2)
        self.assertEqual(body["agent"]["verification"]["reasons"], ["numeric_mismatch"])
        self.assertEqual([s["id"] for s in body["cited_sources"]], ["E1"])
        self.assertNotIn("answer", events[-1])

    def test_no_regeneration_without_llm_budget_or_when_the_answer_is_right(self):
        body, prompts, _, _, _ = self.ask([self.WRONG, self.RIGHT], limits=BudgetLimits(max_llm_calls=1))
        self.assertEqual((len(prompts), body["agent"]["verification"]["label"]), (1, "unverified"))
        self.assertEqual(body["agent"]["states"][-1], "S6_check")
        body, prompts, _, _, _ = self.ask([self.RIGHT])
        self.assertEqual((len(prompts), body["agent"]["verification"]["label"]), (1, "verified"))


if __name__ == "__main__":
    unittest.main()
