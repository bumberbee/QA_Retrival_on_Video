from tests import _isolate  # noqa: F401
import copy
import math
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from app.features.agent.tools import ToolContext, default_registry
from app.features.agent.tools.rerank_tool import rerank
from app.features.search import retrieval
from tests.features.agent.fixtures import context, lecture_index


def fake_module():
    module = ModuleType("app.features.search.rerank")
    def rank(question, candidates, *, scorer=None, top_k=None):
        logits = scorer([(question, candidate["text"]) for candidate in candidates])
        results = [{**candidate, "rerank_score": 1 / (1 + math.exp(-logit))}
                   for candidate, logit in zip(candidates, logits)]
        results.sort(key=lambda candidate: -candidate["rerank_score"])
        return [{**candidate, "rerank_rank": rank} for rank, candidate in enumerate(results[:top_k], 1)]
    module.rerank = Mock(side_effect=rank)
    return module


class RerankToolTests(unittest.TestCase):
    def test_delegates_exact_shared_interface_with_injected_logits(self):
        module = fake_module()
        chunks = [dict(chunk, marker="keep") for chunk in lecture_index().chunks[:2]]
        original, scorer = copy.deepcopy(chunks), Mock(return_value=[-2.0, 2.0])
        with patch.dict(sys.modules, {module.__name__: module}):
            result = rerank("question", chunks, ctx=context(scorer=scorer))
        self.assertTrue(result.ok, result.error)
        module.rerank.assert_called_once_with("question", original, scorer=scorer)
        scorer.assert_called_once_with([("question", chunk["text"]) for chunk in chunks])
        self.assertEqual([chunk["chunk_id"] for chunk in result.data["chunks"]], ["ocr1", "sp2"])
        self.assertEqual([chunk["rerank_rank"] for chunk in result.data["chunks"]], [1, 2])
        self.assertGreater(result.data["chunks"][0]["rerank_score"], 0.8)
        self.assertEqual(result.data["chunks"][0]["marker"], "keep")
        self.assertEqual(chunks, original)
        _, sources = retrieval.build_context_with_ids(result.data["found"])
        self.assertEqual({tuple(source["chunk_ids"]) for source in sources.values()}, {("ocr1",), ("sp2",)})

    def test_minimal_candidates_resolve_sources_with_fake_loader(self):
        module = fake_module()
        chunks = [{"chunk_id": "ocr1", "text": "New OCR only"}]
        ctx = context(scorer=Mock(return_value=[0.0]))
        with patch.dict(sys.modules, {module.__name__: module}):
            result = rerank("q", chunks, ctx=ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["chunks"][0]["rerank_score"], 0.5)
        self.assertEqual(result.data["evidence"][0]["id"], "vid:ocr1")
        self.assertEqual(result.data["evidence"][0]["slide_id"], "vid:s1")
        ctx.index_loader.assert_called_once_with("vid")

    def test_unknown_minimal_candidates_do_not_fabricate_citations(self):
        module = fake_module()
        with patch.dict(sys.modules, {module.__name__: module}):
            result = rerank("q", [{"chunk_id": "unknown", "text": "some text"}],
                            ctx=ToolContext(video_id="vid", scorer=Mock(return_value=[0.0])))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["evidence"], [])
        self.assertEqual(result.data["chunks"][0]["chunk_id"], "unknown")

    def test_unavailable_default_reranker_is_returned_as_error(self):
        module = ModuleType("app.features.search.rerank")
        class RerankerUnavailable(RuntimeError):
            pass
        module.rerank = Mock(side_effect=RerankerUnavailable("local model unavailable"))
        with patch.dict(sys.modules, {module.__name__: module}):
            result = rerank("q", [{"chunk_id": "x", "text": "x"}], ctx=context())
        self.assertFalse(result.ok)
        self.assertIn("RerankerUnavailable", result.error)
        module.rerank.assert_called_once_with("q", [{"chunk_id": "x", "text": "x"}], scorer=None)

    def test_missing_module_does_not_prevent_registry_creation(self):
        with patch.dict(sys.modules, {"app.features.search.rerank": None}):
            registry = default_registry()
            result = registry.run("rerank", {"q": "q", "chunks": []}, context())
            self.assertFalse(result.ok)
            self.assertIn("ModuleNotFoundError", result.error)
            self.assertTrue(registry.run("calc", {"expr": "1+1"}, context()).ok)

    def test_invalid_inputs_never_reach_reranker(self):
        module, registry = fake_module(), default_registry()
        with patch.dict(sys.modules, {module.__name__: module}):
            for args in ({"q": "", "chunks": []}, {"q": "q", "chunks": "text"},
                         {"q": "q", "chunks": [{}]}, {"q": "q", "chunks": [{"chunk_id": "x", "text": 2}]},
                         {"q": "q", "chunks": [{"chunk_id": "", "text": "x"}]},
                         {"q": "q", "chunks": [{"chunk_id": "x", "text": "x", "score": float("nan")}]}):
                self.assertFalse(registry.run("rerank", args, context()).ok)
        module.rerank.assert_not_called()

    def test_empty_candidates_delegate_without_any_real_model(self):
        module = fake_module()
        with patch.dict(sys.modules, {module.__name__: module}):
            result = rerank("q", [], ctx=context(scorer=Mock(return_value=[])))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["chunks"], [])
