from tests import _isolate  # noqa: F401
import copy
import unittest
from unittest.mock import Mock, patch

from app.features.agent.tools import default_registry
from app.features.agent.tools.search_tools import search_hybrid, search_ocr, search_speech
from app.features.search import retrieval
from tests.features.agent.fixtures import context, found, lecture_index


class SearchToolTests(unittest.TestCase):
    def test_hybrid_delegates_and_keeps_rank_and_metadata(self):
        response, encoder = found(), object()
        original = copy.deepcopy(response)
        searcher = Mock(return_value=response)
        ctx = context(searcher=searcher, encoder=encoder, current_time=8)
        result = search_hybrid("question", top_k=2, ctx=ctx)
        self.assertTrue(result.ok, result.error)
        searcher.assert_called_once_with("vid", "question", current_time=8, encoder=encoder, sources=None)
        self.assertEqual([chunk["chunk_id"] for chunk in result.data["chunks"]], ["ocr1", "sp1"])
        self.assertEqual(result.data["evidence"][0]["id"], "vid:ocr1")
        self.assertEqual(response, original)
        self.assertEqual((ctx.budget.tool_calls, ctx.budget.search_rounds), (1, 1))

    def test_modality_tools_use_sources_and_exclude_other_evidence(self):
        for function, kind in ((search_ocr, "ocr"), (search_speech, "speech")):
            with self.subTest(kind=kind):
                searcher = Mock(return_value=found())
                result = function("question", ctx=context(searcher=searcher))
                self.assertTrue(result.ok)
                self.assertEqual(searcher.call_args.kwargs["sources"], frozenset({kind}))
                self.assertTrue(all(item["source_type"] == kind for item in result.data["evidence"]))
                text, sources = retrieval.build_context_with_ids(result.data["found"])
                self.assertTrue(text)
                self.assertTrue(all(item["kind"] == kind for item in sources.values()))

    def test_source_weights_prefer_speech_without_reimplementing_retrieval(self):
        searcher = Mock(return_value=found())
        result = search_hybrid("question", ocr_weight=0.5, speech_weight=2, ctx=context(searcher=searcher))
        self.assertTrue(result.ok)
        self.assertEqual([chunk["chunk_id"] for chunk in result.data["chunks"]], ["sp1", "sp2", "ocr1"])
        self.assertEqual(result.data["chunks"][0]["weighted_score"], 0.08)
        searcher.assert_called_once()

    def test_zero_weight_omits_that_source_from_citeable_context(self):
        result = search_hybrid("question", ocr_weight=0, ctx=context(searcher=Mock(return_value=found())))
        self.assertTrue(result.ok)
        self.assertTrue(all(chunk["source_type"] == "speech" for chunk in result.data["evidence"]))
        _, sources = retrieval.build_context_with_ids(result.data["found"])
        self.assertTrue(all(source["kind"] == "speech" for source in sources.values()))

    def test_default_callable_is_existing_retrieval_search(self):
        with patch.object(retrieval, "search", return_value=found()) as search:
            result = search_hybrid("question", ctx=context())
        self.assertTrue(result.ok)
        search.assert_called_once_with("vid", "question", current_time=None, encoder=None, sources=None)

    def test_actual_search_uses_fake_encoder_and_in_memory_index(self):
        import numpy as np
        index = lecture_index()
        encoder = Mock()
        encoder.encode.return_value = np.array([1.0])
        with patch.object(retrieval, "load_index", return_value=index), \
             patch.object(retrieval, "lexical_hits", return_value=[]), \
             patch.object(retrieval, "default_encoder", side_effect=AssertionError("model forbidden")):
            result = search_ocr("formula", ctx=context(index, encoder=encoder))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.data["chunks"][0]["chunk_id"], "ocr1")
        self.assertEqual(result.data["chunks"][0]["text"], "New OCR only")
        self.assertEqual(result.data["evidence"][0]["source_type"], "ocr")
        encoder.encode.assert_called_once_with("formula")

    def test_empty_search_is_successful_without_evidence(self):
        result = search_speech("question", ctx=context(searcher=Mock(return_value={"slides": []})))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["evidence"], [])
        self.assertEqual(retrieval.build_context_with_ids(result.data["found"]), ("", {}))

    def test_inputs_are_validated_before_search(self):
        invalid = [{"q": ""}, {"q": " "}, {"q": 3}, {"q": "x", "top_k": True},
                   {"q": "x", "top_k": 0}, {"q": "x", "top_k": 101}, {"q": "x", "ocr_weight": -1},
                   {"q": "x", "speech_weight": float("nan")}, {"q": "x", "ocr_weight": False},
                   {"q": "x", "ocr_weight": 0, "speech_weight": 0}, {"q": "x", "extra": 1}]
        registry, searcher = default_registry(), Mock(return_value=found())
        for args in invalid:
            with self.subTest(args=args):
                ctx = context(searcher=searcher)
                self.assertFalse(registry.run("search_hybrid", args, ctx).ok)
                self.assertEqual(ctx.budget.tool_calls, 0)
        searcher.assert_not_called()

    def test_search_exceptions_and_bad_context_are_failures(self):
        ctx = context(searcher=Mock(side_effect=RuntimeError("unavailable")))
        self.assertIn("unavailable", search_ocr("question", ctx=ctx).error)
        ctx.video_id = "../escape"
        self.assertFalse(search_speech("question", ctx=ctx).ok)
        self.assertFalse(search_hybrid("question").ok)
        ctx = context(searcher=Mock(return_value=found()), current_time=float("inf"))
        self.assertFalse(search_ocr("question", ctx=ctx).ok)
        ctx.searcher.assert_not_called()

    def test_guard_counts_searches_and_normalizes_defaults(self):
        ctx = context(searcher=Mock(return_value=found()))
        registry = default_registry()
        self.assertTrue(registry.run("search_hybrid", {"q": "first"}, ctx).ok)
        self.assertIn("duplicate_call", search_hybrid("first", ctx=ctx).error)
        self.assertIn("duplicate_call", search_hybrid("first", ocr_weight=1, speech_weight=1, ctx=ctx).error)
        self.assertTrue(search_ocr("second", ctx=ctx).ok)
        self.assertIn("search_budget", search_speech("third", ctx=ctx).error)
        self.assertEqual(ctx.searcher.call_count, 2)
        self.assertEqual(ctx.budget.tool_calls, 2)
