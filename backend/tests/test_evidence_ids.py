import _isolate  # noqa: F401
import copy
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers import qa
from app.services import llm, retrieval


def evidence_fixture():
    def chunk(cid, sid, start, end, text, kind="speech", priority=0):
        return {"chunk_id": cid, "segment_id": sid, "start_sec": start, "end_sec": end,
                "text": text, "source_type": kind, "title": "", "priority": priority,
                "selection_reason": "matched"}

    return {"slides": [
        {"video_id": "abc123def456", "segment_id": "abc123def456:s1", "start_sec": 0,
         "end_sec": 4000, "title": "First", "ocr_text": "OCR", "speech_text": "Speech",
         "figures": [], "matched_chunk_ids": ["late"], "evidence_candidates": [
             chunk("late", "abc123def456:s1", 3661.9, 3670, "Late answer", priority=0),
             chunk("early", "abc123def456:s1", 65.8, 75, "Early answer", "ocr", priority=1)]},
        {"video_id": "other", "segment_id": "other:s1", "start_sec": 0, "end_sec": 100,
         "title": "Other", "video_title": "Other video", "ocr_text": "OCR", "speech_text": "Speech",
         "figures": [], "evidence_candidates": [
             chunk("other", "other:s1", 65.8, 80, "Early answer")]}], "best_semantic": 1.0}


class EvidenceIDTests(unittest.TestCase):
    def test_ids_follow_final_order_and_snapshot_precise_spans(self):
        found = evidence_fixture()
        legacy = retrieval.build_context(copy.deepcopy(found))
        context, sources = retrieval.build_context_with_ids(found)
        self.assertEqual(list(sources), ["E1", "E2", "E3"])
        self.assertEqual(sources["E1"]["chunk_ids"], ["early"])
        self.assertEqual(sources["E2"]["chunk_ids"], ["late"])
        self.assertEqual(sources["E2"]["t_start"], 3661.9)
        self.assertEqual(sources["E2"]["t_end"], 3670)
        self.assertEqual(sources["E2"]["slide_id"], "abc123def456:s1")
        self.assertEqual(sources["E1"]["kind"], "ocr")
        self.assertEqual(sources["E3"]["video_id"], "other")
        self.assertIn("Video Other video", context)
        self.assertEqual(context.replace("[E1] ", "").replace("[E2] ", "").replace("[E3] ", ""), legacy)
        self.assertEqual((context, sources), retrieval.build_context_with_ids(found))
        found["slides"][0]["evidence_candidates"][0]["start_sec"] = 999
        found["slides"][0]["matched_chunk_ids"].append("changed")
        self.assertEqual(sources["E2"]["t_start"], 3661.9)
        self.assertEqual(sources["E2"]["chunk_ids"], ["late"])

    def test_ids_account_for_budget_and_only_map_kept_blocks(self):
        found = evidence_fixture()
        budget = len(retrieval.evidence_block(found["slides"][0]["evidence_candidates"][0], True, "E1"))
        context, sources = retrieval.build_context_with_ids(found, max_chars=budget)
        self.assertLessEqual(len(context), budget)
        self.assertEqual(list(sources), ["E1"])
        self.assertEqual(sources["E1"]["chunk_ids"], ["late"])
        self.assertEqual([c["evidence_id"] for c in found["context_evidence"]], ["E1"])
        self.assertEqual(retrieval.build_context_with_ids(found, max_chars=1), ("", {}))
        with self.assertRaises(ValueError):
            retrieval.build_context_with_ids(found, max_chars=0)

    def test_legacy_and_neighbor_only_modes_assign_ids_without_truncated_headers(self):
        for candidates, options in ((False, retrieval.RetrievalOptions()),
                                    (True, retrieval.RetrievalOptions(ranked_context=False, neighbors=False)),
                                    (True, retrieval.RetrievalOptions(ranked_context=False, neighbors=True))):
            found = evidence_fixture()
            if not candidates:
                for slide in found["slides"]:
                    slide.pop("evidence_candidates")
            else:
                found["slides"][0]["evidence_candidates"][0]["selection_reason"] = "next"
            with self.subTest(candidates=candidates, options=options):
                context, sources = retrieval.build_context_with_ids(found, options=options, max_chars=350)
                self.assertLessEqual(len(context), 350)
                self.assertTrue(sources)
                for eid, source in sources.items():
                    self.assertIn(f"[{eid}] ", context)
                    self.assertTrue(source["slide_id"])
                self.assertEqual(retrieval.build_context_with_ids(found, options=options, max_chars=1), ("", {}))

    def test_neighbor_source_uses_its_own_slide_and_dedup_occurrence_is_not_cited(self):
        found = evidence_fixture()
        neighbor = found["slides"][0]["evidence_candidates"][0]
        neighbor.update(segment_id="abc123def456:neighbor", selection_reason="next",
                        occurrences=[{"chunk_id": "copy", "start_sec": 10, "end_sec": 20}])
        _, sources = retrieval.build_context_with_ids(found)
        source = next(s for s in sources.values() if s["chunk_ids"] == ["late"])
        self.assertEqual(source["slide_id"], "abc123def456:neighbor")
        self.assertEqual(source["t_start"], 3661.9)

    def test_empty_evidence_is_empty(self):
        self.assertEqual(retrieval.build_context_with_ids({"slides": []}), ("", {}))


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.found = evidence_fixture()
        self.context, self.sources = retrieval.build_context_with_ids(self.found)

    def test_complete_tokens_render_hours_and_unknown_ids_are_dropped(self):
        renderer = llm.CitationRenderer(self.sources)
        text = renderer.feed("Early [E1], late [E2], other [E3], again [E1], bad [E99][E99]. [E2x]", final=True)
        self.assertEqual(text, "Early [01:05], late [1:01:01], other [01:05], again [01:05], bad . [E2x]")
        self.assertEqual([s["id"] for s in renderer.metadata()["cited_sources"]], ["E1", "E2", "E3"])
        self.assertEqual(renderer.invalid_citations, ["E99"])

    def test_every_stream_split_renders_identically_without_leaking_id_prefixes(self):
        raw = "A [E1] B [E2] [E999] [E1] C [E3]."
        expected = "A [01:05] B [1:01:01]  [01:05] C [01:05]."
        for split in range(len(raw) + 1):
            renderer = llm.CitationRenderer(self.sources)
            pieces = [renderer.feed(raw[:split]), renderer.feed(raw[split:]), renderer.feed("", final=True)]
            with self.subTest(split=split):
                self.assertEqual("".join(pieces), expected)
                self.assertFalse(any(piece.endswith(("[", "[E", "[E1", "[E2", "[E999")) for piece in pieces[:2]))
                self.assertEqual(renderer.invalid_citations, ["E999"])
        renderer = llm.CitationRenderer(self.sources)
        self.assertEqual("".join(renderer.feed(c) for c in raw) + renderer.feed("", final=True), expected)

    def test_incomplete_and_non_id_brackets_are_preserved(self):
        renderer = llm.CitationRenderer(self.sources)
        self.assertEqual(renderer.feed("[E12"), "")
        self.assertEqual(renderer.feed("", final=True), "[E12")
        self.assertEqual(renderer.feed("[Example] [01:00] [e1]", final=True), "[Example] [01:00] [e1]")
        self.assertEqual(renderer.invalid_citations, [])

    def test_nonstream_and_stream_adapters_render_and_record_the_same_citations(self):
        raw = "Answer [E2] [E99]."
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                patch.object(llm.gemini, "complete_parts", return_value=raw), \
                patch.object(llm.gemini, "stream_parts", return_value=iter(["Answer [", "E", "2] [E9", "9]."])):
            answer = llm.answer_with_evidence("Question", None, self.found, self.context)
            metadata = {key: copy.deepcopy(self.found[key]) for key in ("cited_sources", "invalid_citations")}
            stream = "".join(llm.stream_with_evidence("Question", None, self.found, self.context))
        self.assertEqual(answer, "Answer [1:01:01] .")
        self.assertEqual(stream, answer)
        self.assertEqual(self.found["invalid_citations"], ["E99"])
        self.assertEqual(self.found["cited_sources"], metadata["cited_sources"])
        self.assertEqual([s["id"] for s in self.found["cited_sources"]], ["E2"])


class EvidenceAPITests(unittest.TestCase):
    def test_response_and_done_keep_retrieved_sources_and_add_cited_sources(self):
        api = TestClient(app)
        plan = {"conversation": None, "history": [], "query": "Question", "prefer": frozenset(),
                "rewritten": None, "follow_up": False, "previous": None, "summary": ""}
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                patch.object(qa.retrieval, "has_index", return_value=True), \
                patch.object(qa.retrieval, "search", side_effect=lambda *a, **kw: evidence_fixture()), \
                patch.object(qa, "prepare", return_value=plan), patch.object(qa, "remember") as remember, \
                patch.object(llm.gemini, "complete_parts", return_value="Answer [E2] [E99]."), \
                patch.object(llm.gemini, "stream_parts", return_value=iter(["Answer [E", "2] [E99]."])):
            response = api.post("/api/qa", json={"video_id": "abc123def456", "question": "Question"})
            stream = api.post("/api/qa/stream", json={"video_id": "abc123def456", "question": "Question"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Answer [1:01:01] .")
        self.assertEqual(len(body["sources"]), 2)
        self.assertIn("used_segments", body)
        self.assertEqual(len(body["evidence_map"]), 3)
        self.assertEqual([s["id"] for s in body["cited_sources"]], ["E2"])
        self.assertEqual(body["invalid_citations"], ["E99"])
        events = [json.loads(line) for line in stream.text.splitlines()]
        self.assertEqual(events[-1]["type"], "done")
        for field in ("evidence_map", "cited_sources", "invalid_citations"):
            self.assertEqual(events[-1][field], body[field])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), body["answer"])
        self.assertEqual(events[1]["sources"], body["sources"])
        self.assertEqual(remember.call_args.args[3], body["answer"])

    def test_multi_video_search_uses_evidence_ids(self):
        req = qa.AskRequest(video_id="abc123def456", video_ids=["abc123def456", "other"], question="Question")
        with patch.object(retrieval, "search_many", return_value=evidence_fixture()) as search:
            found, context = qa.find_evidence(req, "Question")
        search.assert_called_once()
        self.assertEqual(len(found["evidence_map"]), 3)
        self.assertIn("[E1]", context)
        self.assertNotIn("khi trích mốc thời gian", context)


class TruthfulPromptTests(unittest.TestCase):
    def test_only_gemini_with_loaded_images_gets_image_instructions(self):
        found = {"slides": [{"image": "missing.jpg"}]}
        image = {"mime_type": "image/jpeg", "data": b"fake image"}
        for provider, loaded, with_images in (("ollama", [image], False), ("gemini", [], False), ("gemini", [image], True)):
            for stream in (False, True):
                with self.subTest(provider=provider, loaded=bool(loaded), stream=stream), \
                        patch.dict(os.environ, {"LLM_PROVIDER": provider}), \
                        patch.object(llm, "_slide_images", return_value=loaded) as slide_images, \
                        patch.object(llm, "_ollama", return_value="Answer") as complete, \
                        patch.object(llm, "_ollama_stream", return_value=iter(["Answer"])) as ollama_stream, \
                        patch.object(llm.gemini, "complete_parts", return_value="Answer") as gemini_complete, \
                        patch.object(llm.gemini, "stream_parts", return_value=iter(["Answer"])) as gemini_stream:
                    if stream:
                        list(llm.stream_with_evidence("Question", None, found, "evidence"))
                    else:
                        llm.answer_with_evidence("Question", None, found, "evidence")
                    call = (ollama_stream if stream else complete) if provider == "ollama" else (gemini_stream if stream else gemini_complete)
                    parts = call.call_args.args[0]
                    prompt = parts if provider == "ollama" else parts[0]
                    self.assertEqual(llm.IMAGE_NOTE in prompt, with_images)
                    self.assertEqual(llm.TEXT_ONLY_NOTE in prompt, not with_images)
                    self.assertIn("Mình chưa tìm thấy đủ bằng chứng trong bài giảng để trả lời câu này.", prompt)
                    self.assertIn(llm.CITE_RULE, prompt)
                    self.assertNotIn("Bài giảng không đề cập đến nội dung này", prompt)
                    if provider == "ollama":
                        slide_images.assert_not_called()
                    else:
                        self.assertEqual(parts[1:], loaded)

    def test_unreadable_image_uses_text_only_note(self):
        found = {"slides": [{"image": "missing.jpg"}]}
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}), \
                patch("builtins.open", side_effect=OSError("missing")), \
                patch.object(llm.gemini, "complete_parts", return_value="Answer") as complete:
            llm.answer_with_evidence("Question", None, found, "evidence")
        self.assertEqual(len(complete.call_args.args[0]), 1)
        self.assertIn(llm.TEXT_ONLY_NOTE, complete.call_args.args[0][0])
        self.assertNotIn(llm.IMAGE_NOTE, complete.call_args.args[0][0])
