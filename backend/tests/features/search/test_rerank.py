from tests import _isolate  # noqa: F401

import copy
import importlib
import math
from unittest.mock import Mock

import pytest

module = importlib.import_module("app.features.search.rerank")


def candidates(n=3):
    return [{"chunk_id": str(i), "text": f"Đoạn {i}", "source_type": "ocr"} for i in range(n)]


def test_sigmoid_order_new_dicts_and_stable_ties():
    source = candidates(4)
    original = copy.deepcopy(source)
    calls = []

    def scorer(pairs):
        calls.append(pairs)
        return [-1000, 2, 2, 1000]

    result = module.rerank("Câu hỏi", source, scorer=scorer)
    assert [c["chunk_id"] for c in result] == ["3", "1", "2", "0"]
    assert [c["rerank_rank"] for c in result] == [1, 2, 3, 4]
    assert result[0]["rerank_score"] == 1.0
    assert result[-1]["rerank_score"] == 0.0
    assert result[1]["rerank_score"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert all(type(c["rerank_score"]) is float for c in result)
    assert source == original
    assert all(c is not source[int(c["chunk_id"])] for c in result)
    assert all(c["source_type"] == "ocr" for c in result)
    assert calls == [[("Câu hỏi", c["text"]) for c in source]]


def test_batching_and_top_k_use_scores_across_all_batches():
    calls = []

    def scorer(pairs):
        calls.append(pairs)
        return [float(text.split()[-1]) for _, text in pairs]

    result = module.rerank("q", candidates(20), scorer=scorer, top_k=2)
    assert [len(pairs) for pairs in calls] == [8, 8, 4]
    assert [c["chunk_id"] for c in result] == ["19", "18"]
    assert [c["rerank_rank"] for c in result] == [1, 2]


@pytest.mark.parametrize("values", [[0], [0, 1, 2, 3], [0, float("nan"), 0],
                                    [0, float("inf"), 0], [0, -float("inf"), 0]])
def test_invalid_scorer_outputs_are_rejected(values):
    with pytest.raises(ValueError):
        module.rerank("q", candidates(), scorer=lambda pairs: values)


@pytest.mark.parametrize("top_k", [-1, 1.5, "2", True])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        module.rerank("q", candidates(), scorer=lambda pairs: [0] * len(pairs), top_k=top_k)


@pytest.mark.parametrize("source,top_k", [([], None), (candidates(), 0)])
def test_empty_results_do_not_construct_default_scorer(monkeypatch, source, top_k):
    loader = Mock(side_effect=AssertionError("Must remain lazy"))
    monkeypatch.setattr(module, "default_scorer", loader)
    assert module.rerank("q", source, top_k=top_k) == []
    loader.assert_not_called()


def test_default_scorer_is_lazy_and_unavailable_is_explicit(monkeypatch):
    loader = Mock(return_value=lambda pairs: [0.0] * len(pairs))
    monkeypatch.setattr(module, "default_scorer", loader)
    module.rerank("q", candidates(), scorer=lambda pairs: [0.0] * len(pairs))
    loader.assert_not_called()
    assert module.rerank("q", candidates())[0]["rerank_score"] == 0.5
    loader.assert_called_once_with()
    loader.side_effect = module.RerankerUnavailable("local weights absent")
    with pytest.raises(module.RerankerUnavailable):
        module.rerank("q", candidates())


@pytest.mark.parametrize("source", [[{"text": "a"}], [{"chunk_id": "a"}],
                                   [{"chunk_id": "a", "text": None}]])
def test_candidate_contract(source):
    with pytest.raises(ValueError):
        module.rerank("q", source, scorer=lambda pairs: [0.0])


def test_default_backend_prefers_onnx_and_falls_back_only_to_installed_backends(monkeypatch, tmp_path):
    module._load_default_scorer.cache_clear()
    fake = lambda pairs: [0.0] * len(pairs)
    onnx = Mock(return_value=fake)
    transformers = Mock(return_value=fake)
    sentence_transformers = Mock(return_value=fake)
    monkeypatch.setattr(module, "_model_directory", lambda: tmp_path)
    monkeypatch.setattr(module, "_installed", lambda name: True)
    monkeypatch.setattr(module, "_OnnxScorer", onnx)
    monkeypatch.setattr(module, "_TransformersScorer", transformers)
    monkeypatch.setattr(module, "_SentenceTransformersScorer", sentence_transformers)
    try:
        assert module.default_scorer() is fake
        assert module.default_scorer() is fake
        onnx.assert_called_once_with(tmp_path)
        transformers.assert_not_called()
        module._load_default_scorer.cache_clear()
        onnx.side_effect = OSError("No local ONNX export")
        assert module.default_scorer() is fake
        transformers.assert_called_once_with(tmp_path)
        sentence_transformers.assert_not_called()
        module._load_default_scorer.cache_clear()
        monkeypatch.setattr(module, "_installed", lambda name: False)
        with pytest.raises(module.RerankerUnavailable):
            module.default_scorer()
    finally:
        module._load_default_scorer.cache_clear()
