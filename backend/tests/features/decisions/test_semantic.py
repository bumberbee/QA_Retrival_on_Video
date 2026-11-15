from tests import _isolate  # noqa: F401

import json

import pytest

from app.features.decisions import NotConfigured, decide, route_question
from app.features.decisions.config import ROOT, load_config
from app.features.decisions.semantic import SemanticBackend
from app.features.decisions.types import QTYPES


def basis(index, dimension=8):
    return [float(i == index) for i in range(dimension)]


def test_example_bank_has_twenty_bilingual_examples_per_type_and_matching_json():
    examples = load_config("router_examples")
    assert set(examples) == set(QTYPES)
    assert examples == json.loads((ROOT / "router_examples.json").read_text(encoding="utf-8"))
    all_examples = [example for items in examples.values() for example in items]
    assert len(set(all_examples)) == len(all_examples)
    for items in examples.values():
        assert len(items) >= 20
        assert sum(not text.isascii() for text in items) >= 8
        assert sum(text.isascii() for text in items) >= 8


def test_semantic_uses_top_k_mean_cosines_and_caches_examples():
    examples = {q: [q + ":0", q + ":1", q + ":2"] for q in QTYPES}
    vectors = {text: basis(i) for i, items in enumerate(examples.values()) for text in items}
    # definition's first two exemplars score 1 and .8; its third is orthogonal.
    vectors["definition:1"] = [0.8, 0.6, 0, 0, 0, 0, 0, 0]
    vectors["definition:2"] = basis(7)
    calls = []
    def encode(text):
        calls.append(text)
        return vectors.get(text, basis(0))
    backend = SemanticBackend(encode, examples, top_k=2)
    result = backend.run("Q4", {"question": "query"})
    assert result.label == "definition"
    assert result.p == pytest.approx(0.9)
    assert result.scores["formula"] == 0
    assert len(calls) == 25
    backend.run("Q4", {"question": "query2"})
    assert len(calls) == 26


@pytest.mark.parametrize("label", QTYPES)
def test_all_semantic_labels_with_fake_encoder(label):
    examples = {q: [q] for q in QTYPES}
    vectors = {q: basis(i) for i, q in enumerate(QTYPES)}
    vectors["query"] = vectors[label]
    backend = SemanticBackend(vectors.__getitem__, examples)
    result = backend.run("Q4", {"question": "query"})
    assert result.label == label
    assert result.p == 1


def test_injected_encoder_object_works_in_router_and_decide_without_loading_model(monkeypatch):
    from app.features.search import retrieval
    def never(*args):
        raise AssertionError("Real model initialization attempted")
    monkeypatch.setattr(retrieval, "default_encoder", never)
    bank = load_config("router_examples")
    vectors = {text: basis(i) for i, items in enumerate(bank.values()) for text in items}
    class Encoder:
        def __init__(self):
            self.calls = 0
        def encode(self, text):
            self.calls += 1
            return vectors.get(text, basis(3))
    encoder = Encoder()
    routed = route_question("Nội dung cần tìm", {"encoder": encoder})
    assert routed.qtype == "comparison"
    assert routed.decided_by == "semantic"
    assert encoder.calls == 161
    result = decide("Q4", {"question": "Nội dung khác", "encoder": encoder}, backend="semantic")
    assert result.label == "comparison"
    assert result.backend == "semantic"
    assert encoder.calls == 162
    route_question("Một nội dung khác nữa", {}, encode=encoder.encode)
    assert encoder.calls == 163


def test_semantic_not_configured_and_zero_vectors():
    examples = {q: [q] for q in QTYPES}
    with pytest.raises(NotConfigured):
        SemanticBackend(examples=examples).run("Q4", {"question": "query"})
    backend = SemanticBackend(lambda text: [0, 0], examples)
    assert backend.run("Q4", {"question": "query"}).p == 0
    with pytest.raises(NotConfigured):
        backend.run("Q1", {})


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_semantic_invalid_vectors_surface(vector):
    examples = {q: [q] for q in QTYPES}
    with pytest.raises(ValueError):
        SemanticBackend(lambda text: vector, examples).run("Q4", {"question": "query"})


def test_semantic_dimension_mismatch_and_invalid_configuration():
    examples = {q: [q] for q in QTYPES}
    backend = SemanticBackend(lambda text: [1, 0] if text in QTYPES else [1], examples)
    with pytest.raises(ValueError, match="dimensions"):
        backend.run("Q4", {"question": "query"})
    with pytest.raises(ValueError):
        SemanticBackend(lambda text: [1], {"definition": ["one"]})
    with pytest.raises(ValueError):
        SemanticBackend(lambda text: [1], examples, top_k=0)
