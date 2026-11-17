from tests import _isolate  # noqa: F401
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.features.ask import llm, postprocess
from app.features.ask import router as qa
from app.features.search import retrieval

REFUSAL = llm.REFUSAL
ANSWER = "Nhận định văn học là ý kiến đánh giá về một tác phẩm hoặc tác giả văn học."
VOCABULARY = ["nhận", "định", "văn", "học", "tác", "phẩm", "bayes", "xác", "suất", "công", "thức"]


class FakeEncoder:
    """Bag of words over a tiny vocabulary: similarity is word overlap, no model."""

    def encode_many(self, texts, prefix=None):
        rows = np.array([[text.casefold().count(word) for word in VOCABULARY] for text in texts], dtype=float)
        return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)


def found_fixture(best_semantic=0.9):
    def chunk(cid, start, text):
        return {"chunk_id": cid, "segment_id": "abc123def456:s1", "start_sec": start, "end_sec": start + 20,
                "text": text, "source_type": "speech", "title": "", "priority": 0,
                "selection_reason": "matched"}

    found = {"slides": [{"video_id": "abc123def456", "segment_id": "abc123def456:s1", "start_sec": 0, "end_sec": 600,
                         "title": "", "ocr_text": "", "speech_text": "", "figures": [],
                         "evidence_candidates": [
                             chunk("a", 65, "nhận định văn học là đánh giá về tác phẩm văn học"),
                             chunk("b", 300, "công thức bayes tính xác suất")]}],
             "best_semantic": best_semantic}
    retrieval.build_context_with_ids(found)
    return found


def strip(text, *, active=True, pieces=None):
    stripper = postprocess.RefusalStripper(REFUSAL, active)
    parts = pieces if pieces is not None else [text]
    out = "".join(stripper.feed(part) for part in parts) + stripper.feed("", final=True)
    return out, stripper.stripped


class RefusalStripperTests(unittest.TestCase):
    def test_refusal_followed_by_an_answer_is_dropped(self):
        self.assertEqual(strip(f"{REFUSAL} Tuy nhiên, {ANSWER[0].lower()}{ANSWER[1:]}"), (ANSWER, True))
        self.assertEqual(strip(f'"{REFUSAL}"\n\n{ANSWER}'), (ANSWER, True))

    def test_plain_refusal_and_short_hint_are_kept(self):
        for text in (REFUSAL, f"{REFUSAL} Xem phút 12.", f'"{REFUSAL}"'):
            with self.subTest(text=text):
                self.assertEqual(strip(text), (text, False))

    def test_other_answers_and_inactive_filter_pass_through(self):
        late = f"{ANSWER} {REFUSAL}"
        self.assertEqual(strip(ANSWER), (ANSWER, False))
        self.assertEqual(strip(late), (late, False))
        both = f"{REFUSAL} {ANSWER}"
        self.assertEqual(strip(both, active=False), (both, False))
        self.assertEqual(strip("Mình chưa tìm thấy slide đó, nhưng " + ANSWER)[1], False)

    def test_every_stream_split_matches_the_whole_text(self):
        for text in (f"{REFUSAL} {ANSWER}", REFUSAL, ANSWER, f"{REFUSAL} Xem phút 12."):
            expected = strip(text)
            for split in range(len(text) + 1):
                with self.subTest(text=text[:20], split=split):
                    self.assertEqual(strip(text, pieces=[text[:split], text[split:]]), expected)
            self.assertEqual(strip(text, pieces=list(text)), expected)

    def test_only_strong_evidence_activates_the_filter(self):
        both = f"{REFUSAL} {ANSWER}"
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "ASK_STRIP_REFUSAL": "on"}), \
                patch.object(llm.gemini, "complete_parts", return_value=both), \
                patch.object(llm.gemini, "stream_parts", side_effect=lambda parts: iter([both[:30], both[30:]])):
            strong, weak = found_fixture(0.9), found_fixture(0.8)
            self.assertEqual(llm.answer_with_evidence("Q", None, strong, "ctx"), ANSWER)
            self.assertTrue(strong["refusal_stripped"])
            self.assertEqual("".join(llm.stream_with_evidence("Q", None, found_fixture(0.9), "ctx")), ANSWER)
            self.assertEqual(llm.answer_with_evidence("Q", None, weak, "ctx"), both)
            self.assertFalse(weak["refusal_stripped"])
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "ASK_STRIP_REFUSAL": ""}), \
                patch.object(llm.gemini, "complete_parts", return_value=both):
            self.assertEqual(llm.answer_with_evidence("Q", None, found_fixture(0.9), "ctx"), both)


class AttributionTests(unittest.TestCase):
    def test_each_sentence_gets_its_closest_evidence(self):
        found = found_fixture()
        answer = "Nhận định văn học là đánh giá tác phẩm. Công thức Bayes tính xác suất!\nOk."
        text, sources = postprocess.attribute(answer, found, FakeEncoder(), refusal=REFUSAL, minimum=0.5)
        self.assertEqual(text, "Nhận định văn học là đánh giá tác phẩm [01:05]. "
                               "Công thức Bayes tính xác suất [05:00]!\nOk.")
        self.assertEqual([(s["id"], s["inferred"], s["t_start"]) for s in sources],
                         [("E1", True, 65), ("E2", True, 300)])

    def test_weak_matches_refusals_and_cited_answers_are_untouched(self):
        found = found_fixture()
        encoder = FakeEncoder()
        for answer in ("Thủ đô của nước Pháp là Paris.", REFUSAL, "",
                       "Nhận định văn học là đánh giá tác phẩm [01:05]."):
            with self.subTest(answer=answer):
                self.assertEqual(postprocess.attribute(answer, found, encoder, refusal=REFUSAL, minimum=0.5),
                                 (answer, []))
        self.assertEqual(postprocess.attribute(ANSWER, {"slides": []}, encoder), (ANSWER, []))

    def test_legacy_slide_blocks_supply_their_text(self):
        found = {"slides": [{"video_id": "abc123def456", "segment_id": "abc123def456:s1", "start_sec": 130, "end_sec": 200,
                             "title": "Bayes", "ocr_text": "công thức bayes", "speech_text": "xác suất",
                             "figures": []}], "best_semantic": 0.9}
        retrieval.build_context_with_ids(found)
        text, sources = postprocess.attribute("Công thức Bayes tính xác suất.", found, FakeEncoder(), minimum=0.5)
        self.assertEqual(text, "Công thức Bayes tính xác suất [02:10].")
        self.assertEqual(sources[0]["kind"], "slide")


class AttributionAPITests(unittest.TestCase):
    def ask(self, environment, raw):
        api = TestClient(app)
        plan = {"conversation": None, "history": [], "query": "Q", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "ASK_ATTRIBUTE_MIN": "0.5", **environment}), \
                patch.object(qa.retrieval, "has_index", return_value=True), \
                patch.object(qa.retrieval, "search", side_effect=lambda *a, **kw: found_fixture()), \
                patch.object(qa.retrieval, "load_index", return_value=type("Index", (), {"model": "m"})()), \
                patch.object(qa.retrieval, "default_encoder", return_value=FakeEncoder()), \
                patch.object(qa, "prepare", return_value=plan), patch.object(qa, "remember") as remember, \
                patch.object(llm.gemini, "complete_parts", return_value=raw), \
                patch.object(llm.gemini, "stream_parts", side_effect=lambda parts: iter([raw[:9], raw[9:]])):
            body = api.post("/api/qa", json={"video_id": "abc123def456", "question": "Q"}).json()
            stream = api.post("/api/qa/stream", json={"video_id": "abc123def456", "question": "Q"})
        events = [json.loads(line) for line in stream.text.splitlines()]
        return body, events, remember

    def test_uncited_answer_is_attributed_in_both_endpoints(self):
        raw = "Công thức Bayes tính xác suất."
        body, events, remember = self.ask({"ASK_ATTRIBUTE": "on"}, raw)
        self.assertEqual(body["answer"], "Công thức Bayes tính xác suất [05:00].")
        self.assertEqual([(s["id"], s["inferred"]) for s in body["cited_sources"]], [("E2", True)])
        self.assertIn("attribution_seconds", body["timings"])
        done = events[-1]
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), raw)
        self.assertEqual(done["answer"], body["answer"])
        self.assertEqual(done["cited_sources"], body["cited_sources"])
        self.assertEqual(remember.call_args.args[3], body["answer"])

    def test_model_citations_and_the_default_setting_leave_the_answer_alone(self):
        body, events, _ = self.ask({"ASK_ATTRIBUTE": "on"}, "Công thức Bayes tính xác suất [E1].")
        self.assertEqual(body["answer"], "Công thức Bayes tính xác suất [01:05].")
        self.assertNotIn("inferred", body["cited_sources"][0])
        self.assertNotIn("answer", events[-1])
        body, events, _ = self.ask({"ASK_ATTRIBUTE": ""}, "Công thức Bayes tính xác suất.")
        self.assertEqual(body["answer"], "Công thức Bayes tính xác suất.")
        self.assertEqual(body["cited_sources"], [])
        self.assertNotIn("answer", events[-1])
        self.assertNotIn("attribution_seconds", body["timings"])

    def test_encoder_failure_keeps_the_answer(self):
        with patch.object(postprocess, "attribute", side_effect=RuntimeError("no model")), \
                self.assertLogs(qa.logger, level="ERROR"):
            body, events, _ = self.ask({"ASK_ATTRIBUTE": "on"}, "Công thức Bayes tính xác suất.")
        self.assertEqual(body["answer"], "Công thức Bayes tính xác suất.")
        self.assertEqual(events[-1]["type"], "done")


class CiteReminderTests(unittest.TestCase):
    def test_reminder_follows_the_question_only_when_switched_on(self):
        found = found_fixture()
        with patch.dict(os.environ, {"ASK_CITE_REMINDER": ""}):
            plain = llm._final_prompt("Câu hỏi?", None, found, "ctx")
        with patch.dict(os.environ, {"ASK_CITE_REMINDER": "on"}):
            reminded = llm._final_prompt("Câu hỏi?", None, found, "ctx")
        self.assertEqual(plain, llm._evidence_prompt("Câu hỏi?", None, found, "ctx"))
        self.assertEqual(reminded, plain + llm.CITE_REMINDER + "\n")


if __name__ == "__main__":
    unittest.main()
