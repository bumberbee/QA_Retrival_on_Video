from tests import _isolate  # noqa: F401
import copy
import unittest
from unittest.mock import Mock

from app.features.agent.tools import default_registry
from app.features.agent.tools.slide_tools import get_slide, get_window, neighbors
from app.features.search import retrieval
from tests.features.agent.fixtures import context, lecture_index


class SlideToolTests(unittest.TestCase):
    def test_get_slide_returns_full_ocr_timestamps_image_and_stable_evidence(self):
        index = lecture_index()
        original = copy.deepcopy(index.segments)
        result = get_slide("vid:s1", ctx=context(index))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.data["ocr_text"], "Full first OCR\nA = B + C")
        self.assertEqual((result.data["t_start"], result.data["t_end"]), (0, 10))
        self.assertTrue(result.data["image_path"].endswith("s1.jpg"))
        self.assertEqual(result.data["evidence"][0]["chunk_id"], "vid:s1:ocr")
        second = get_slide("s1", ctx=context(index))
        self.assertEqual(result.data["evidence"], second.data["evidence"])
        text, sources = retrieval.build_context_with_ids(result.data["found"])
        self.assertIn("A = B + C", text)
        self.assertEqual(sources["E1"]["slide_id"], "vid:s1")
        self.assertEqual(index.segments, original)

    def test_get_slide_falls_back_to_raw_ocr_then_chunks(self):
        index = lecture_index()
        segment = index.by_segment["vid:s1"]
        segment["visual"] = {"ocr_text": "Raw full OCR", "image_path": "raw.jpg"}
        self.assertEqual(get_slide("s1", ctx=context(index)).data["ocr_text"], "Raw full OCR")
        segment["visual"] = {}
        self.assertEqual(get_slide("s1", ctx=context(index)).data["ocr_text"], "New OCR only")

    def test_cloud_image_path_and_empty_slide_are_preserved(self):
        index = lecture_index()
        index.folder = None
        self.assertEqual(get_slide("s1", ctx=context(index)).data["image_path"], "s1.jpg")
        index.by_segment["vid:s3"]["visual"] = {}
        result = get_slide("s3", ctx=context(index))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["ocr_text"], "")
        self.assertEqual(result.data["evidence"], [])

    def test_neighbors_are_chronological_exclude_anchor_and_ads(self):
        result = neighbors("s2", ctx=context())
        self.assertTrue(result.ok)
        self.assertEqual([slide["slide_id"] for slide in result.data["slides"]], ["vid:s1", "vid:s3"])
        self.assertEqual([item["slide_id"] for item in result.data["evidence"]], ["vid:s1", "vid:s3"])
        _, sources = retrieval.build_context_with_ids(result.data["found"])
        self.assertEqual({source["slide_id"] for source in sources.values()}, {"vid:s1", "vid:s3"})

    def test_neighbors_clamp_at_edges_and_allow_zero(self):
        self.assertEqual([slide["slide_id"] for slide in neighbors("s1", k=10, ctx=context()).data["slides"]],
                         ["vid:s2", "vid:s3"])
        self.assertEqual(neighbors("s1", k=0, ctx=context()).data["slides"], [])

    def test_get_window_orders_overlapping_speech_and_preserves_source_times(self):
        index = lecture_index()
        original = copy.deepcopy(index.chunks)
        result = get_window("vid", 6, 16, ctx=context(index))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.data["speech_text"], "Early speech\nLater speech")
        self.assertEqual([item["chunk_id"] for item in result.data["evidence"]], ["sp1", "sp2"])
        self.assertEqual((result.data["evidence"][0]["t_start"], result.data["evidence"][0]["t_end"]), (4, 8))
        _, sources = retrieval.build_context_with_ids(result.data["found"])
        self.assertEqual(sources["E1"]["chunk_ids"], ["sp1"])
        self.assertEqual(index.chunks, original)

    def test_window_boundaries_empty_results_and_ads(self):
        self.assertEqual(get_window("vid", 8, 14, ctx=context()).data["evidence"], [])
        self.assertEqual(get_window("vid", 20, 30, ctx=context()).data["evidence"], [])
        self.assertEqual(get_window("vid", 99, 100, ctx=context()).data["speech_text"], "")

    def test_window_prefers_aligned_spans_over_coarse_chunks(self):
        index = lecture_index()
        index.by_segment["vid:s1"]["speech"] = {"spans": [
            {"span_id": "span1", "text": "Word-aligned speech", "start_sec": 5, "end_sec": 6},
            {"span_id": "span2", "text": "Outside window", "start_sec": 7, "end_sec": 8}]}
        result = get_window("vid", 5, 7, ctx=context(index))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["speech_text"], "Word-aligned speech")
        self.assertEqual(result.data["evidence"][0]["chunk_id"], "span1")

    def test_window_reads_segment_speech_when_chunks_are_missing(self):
        index = lecture_index()
        index.chunks = []
        index.by_segment["vid:s2"]["speech"] = {"text": "Complete segment speech"}
        result = get_window("vid", 12, 15, ctx=context(index))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["speech_text"], "Complete segment speech")
        self.assertEqual(result.data["evidence"][0]["chunk_id"], "vid:s2:speech")

    def test_window_uses_its_explicit_video_id(self):
        ctx = context()
        result = get_window("other", 0, 10, ctx=ctx)
        self.assertTrue(result.ok)
        ctx.index_loader.assert_called_once_with("other")
        self.assertEqual(result.data["evidence"][0]["video_id"], "other")

    def test_invalid_inputs_and_missing_slides_never_raise(self):
        registry = default_registry()
        calls = [("get_slide", {"slide_id": "../escape"}), ("get_slide", {"slide_id": 1}),
                 ("neighbors", {"slide_id": "s1", "k": -1}), ("neighbors", {"slide_id": "s1", "k": True}),
                 ("get_window", {"video_id": "../escape", "t_start": 0, "t_end": 1}),
                 ("get_window", {"video_id": "vid", "t_start": 2, "t_end": 1}),
                 ("get_window", {"video_id": "vid", "t_start": 1, "t_end": 1}),
                 ("get_window", {"video_id": "vid", "t_start": -1, "t_end": 1}),
                 ("get_window", {"video_id": "vid", "t_start": 0, "t_end": float("inf")})]
        for name, args in calls:
            with self.subTest(name=name, args=args):
                ctx = context()
                self.assertFalse(registry.run(name, args, ctx).ok)
                ctx.index_loader.assert_not_called()
        self.assertFalse(get_slide("absent", ctx=context()).ok)
        self.assertFalse(get_slide("ad", ctx=context()).ok)
        self.assertFalse(neighbors("absent", ctx=context()).ok)

    def test_loader_failure_is_a_tool_error(self):
        ctx = context()
        ctx.index_loader = Mock(side_effect=OSError("index unavailable"))
        self.assertIn("index unavailable", get_slide("s1", ctx=ctx).error)
