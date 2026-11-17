from tests import _isolate  # noqa: F401

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.features.agent import nli


def found(texts=None):
    return nli.found_from_evidence_map({key: {"text": text} for key, text in
                                      (texts or {"E1": "Bằng chứng thứ nhất.", "E2": "Bằng chứng thứ hai."}).items()})


def test_vietnamese_sentence_splitting_and_unique_citations():
    answer = "Xác suất là 0.5 [E1] [E1]. Biến cố chắc chắn có xác suất bằng một!\nĐiều này có đúng không [E2]?"
    assert nli.split_claims(answer) == [
        {"claim_id": "C1", "text": "Xác suất là 0.5 [E1] [E1].", "evidence_ids": ["E1"]},
        {"claim_id": "C2", "text": "Biến cố chắc chắn có xác suất bằng một!", "evidence_ids": []},
        {"claim_id": "C3", "text": "Điều này có đúng không [E2]?", "evidence_ids": ["E2"]}]
    assert nli.split_claims(" \n ") == []


@pytest.mark.parametrize("triple,label", [((0.6, 0.4, 0), "supported"),
                                         ((0.59999, 0.40001, 0), "unsupported"),
                                         ((0.2, 0.3, 0.5), "contradicted"),
                                         ((0.2, 0.30001, 0.49999), "unsupported")])
def test_threshold_edges(triple, label):
    report = nli.check_claims("Mệnh đề [E1].", found(), scorer=lambda pairs: [triple] * len(pairs))
    row = report["claims"][0]
    assert row["status"] == label
    assert row["entail"] == triple[0] and row["contradict"] == triple[2]
    assert report["counts"] == {"total": 1, "supported": int(label == "supported"),
                                "unsupported": int(label == "unsupported"), "contradicted": int(label == "contradicted")}
    assert (report["accept"], report["reject_contradict"]) == (0.6, 0.5)


def test_premise_selection_uses_citations_or_all_sources_for_uncited_and_inferred():
    source = found()
    source["cited_sources"] = [{"id": "E2", "inferred": True}]
    original = copy.deepcopy(source)
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [(0.8, 0.1, 0.1)] * len(pairs)

    report = nli.check_claims("Có nguồn [E1]. Không có nguồn. Nguồn suy ra [E2]. Nguồn sai [E9].", source, scorer=scorer)
    assert [row["evidence_ids"] for row in report["claims"]] == [["E1"], ["E1", "E2"], ["E1", "E2"], []]
    assert calls == [("Bằng chứng thứ nhất.", "Có nguồn ."),
                     ("Bằng chứng thứ nhất.", "Không có nguồn."), ("Bằng chứng thứ hai.", "Không có nguồn."),
                     ("Bằng chứng thứ nhất.", "Nguồn suy ra ."), ("Bằng chứng thứ hai.", "Nguồn suy ra .")]
    assert [row["premise_chars"] for row in report["claims"]] == [len("Bằng chứng thứ nhất."),
        len("Bằng chứng thứ nhất.") + len("Bằng chứng thứ hai."),
        len("Bằng chứng thứ nhất.") + len("Bằng chứng thứ hai."), 0]
    assert report["claims"][-1]["status"] == "unsupported"
    assert source == original


def test_maximum_entailment_wins_over_conflicting_premise():
    report = nli.check_claims("Mệnh đề [E1] [E2].", found(),
                              scorer=lambda pairs: [(0.7, 0.2, 0.1), (0.05, 0.05, 0.9)])
    assert report["claims"][0]["status"] == "supported"
    assert report["claims"][0]["entail"] == 0.7
    assert report["claims"][0]["contradict"] == 0.9


def test_check_uses_exact_context_text_and_slide_fallback():
    source = {"evidence_map": {"E1": {"kind": "chunk"}, "E2": {"kind": "slide", "segment_id": "s"}},
              "context_evidence": [{"evidence_id": "E1", "text": "Retained prompt text"}],
              "slides": [{"segment_id": "s", "title": "Title", "ocr_text": "OCR", "speech_text": "Speech"}]}
    scorer = Mock(return_value=[(0.7, 0.2, 0.1), (0.8, 0.1, 0.1)])
    nli.check_claims("First [E1]. Second [E2].", source, scorer=scorer)
    assert scorer.call_args.args[0] == [("Retained prompt text", "First ."), ("Title\nOCR\nSpeech", "Second .")]


def test_batching_probabilities_and_no_load_on_empty_input(monkeypatch):
    loader = Mock(side_effect=AssertionError("Real model forbidden"))
    monkeypatch.setattr(nli, "default_scorer", loader)
    calls = []
    scorer = lambda pairs: calls.append(pairs) or [(0.6, 0.3, 0.1)] * len(pairs)
    assert len(nli.score_pairs([("p", "h")] * 20, scorer=scorer)) == 20
    assert [len(batch) for batch in calls] == [8, 8, 4]
    assert nli.score_pairs([]) == []
    loader.assert_not_called()


@pytest.mark.parametrize("values", [[], [(0.6, 0.4)], [(0.6, 0.4, 0.1)], [(-0.1, 1, 0.1)],
                                    [(float("nan"), 0, 1)], [(float("inf"), 0, 0)], [1]])
def test_invalid_scorer_outputs(values):
    with pytest.raises(ValueError):
        nli.score_pairs([("p", "h")], scorer=lambda pairs: values)


@pytest.mark.parametrize("pairs", [[("p",)], [("p", 1)], ["ph"]])
def test_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        nli.score_pairs(pairs, scorer=Mock())


def test_default_loader_is_lazy_cached_and_failures_remain_retryable(monkeypatch, tmp_path):
    nli._load_default_scorer.cache_clear()
    fake = Mock(return_value=[(0.7, 0.2, 0.1)])
    constructor = Mock(return_value=fake)
    monkeypatch.setattr(nli, "_OnnxScorer", constructor)
    monkeypatch.delenv("LECTURE_NLI_MODEL_DIR", raising=False)
    try:
        with pytest.raises(nli.NLIUnavailable):
            nli.default_scorer()
        constructor.assert_not_called()
        monkeypatch.setenv("LECTURE_NLI_MODEL_DIR", str(tmp_path))
        assert nli.default_scorer() is fake
        assert nli.default_scorer() is fake
        constructor.assert_called_once_with(tmp_path.resolve())
        nli._load_default_scorer.cache_clear()
        constructor.side_effect = ImportError("Missing inference library")
        with pytest.raises(nli.NLIUnavailable, match="no download"):
            nli.default_scorer()
    finally:
        nli._load_default_scorer.cache_clear()


def test_flag_is_read_at_startup_not_per_request(monkeypatch):
    before = nli.enabled()
    monkeypatch.setenv("LECTURE_NLI", "on" if not before else "off")
    assert nli.enabled() is before
    spec = importlib.util.spec_from_file_location("_fake_startup_nli", nli.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.enabled() is not before


@pytest.mark.parametrize("mapping,order", [({"0": "contradiction", "1": "neutral", "2": "entailment"}, (2, 1, 0)),
                                         ({"0": "ENTAILMENT", "1": "CONTRADICTION", "2": "NEUTRAL"}, (0, 2, 1))])
def test_label_order_is_read_from_config(tmp_path, mapping, order):
    (tmp_path / "config.json").write_text(json.dumps({"id2label": mapping}), encoding="utf-8")
    assert nli._label_order(tmp_path) == order


@pytest.mark.parametrize("mapping", [{}, {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},
                                      {"0": "entailment", "1": "neutral", "2": "neutral"}])
def test_ambiguous_labels_are_never_guessed(tmp_path, mapping):
    (tmp_path / "config.json").write_text(json.dumps({"id2label": mapping}), encoding="utf-8")
    with pytest.raises(nli.NLIUnavailable):
        nli._label_order(tmp_path)


def test_onnx_contract_with_fake_runtime_and_tokenizer(monkeypatch, tmp_path):
    # No weights or real tokenizer/model are loaded: these doubles expose only
    # the runtime methods needed to verify truncation, feeds and label softmax.
    encodings = []

    class Encoding:
        def __init__(self, ids, types=None):
            self.ids = list(ids)
            self.type_ids = types or [0] * len(ids)
            self.attention_mask = [1] * len(ids)

        def truncate(self, size):
            self.ids = self.ids[:size]

    class Tokenizer:
        def no_truncation(self): pass
        def no_padding(self): pass
        def num_special_tokens_to_add(self, is_pair): return 3
        def token_to_id(self, token): return 0

        def encode_batch(self, texts, add_special_tokens):
            return [Encoding([int(text[0])] * int(text[1:])) for text in texts]

        def post_process(self, premise, hypothesis, add_special_tokens):
            encodings.append((list(premise.ids), list(hypothesis.ids)))
            ids = [9] + premise.ids + [9] + hypothesis.ids + [9]
            return Encoding(ids, [0] * (len(premise.ids) + 2) + [1] * (len(hypothesis.ids) + 1))

    feeds = []

    class Session:
        def get_inputs(self): return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]
        def get_outputs(self): return [SimpleNamespace(name="logits")]
        def run(self, outputs, feed):
            feeds.append(feed)
            return [np.tile([0, 1, 2], (len(feed["input_ids"]), 1))]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Mock(return_value=Session())))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=Mock(return_value=Tokenizer()))))
    monkeypatch.setattr(nli, "_file", lambda directory, *names: directory / names[0])
    monkeypatch.setattr(nli, "_label_order", lambda directory: (2, 1, 0))
    scorer = nli._OnnxScorer(tmp_path)
    result = nli.score_pairs([("1600", "2100"), ("120", "2600"), ("11", "21")], scorer=scorer)
    assert [tuple(map(len, pair)) for pair in encodings] == [(409, 100), (0, 509), (1, 1)]
    feed = feeds[0]
    assert feed["input_ids"].shape == (3, 512)
    assert feed["attention_mask"][2].sum() == 5
    assert feed["input_ids"].dtype == np.int64
    expected = np.exp([2, 1, 0]) / np.exp([2, 1, 0]).sum()
    assert result[0] == pytest.approx(tuple(expected))
    assert sum(result[0]) == pytest.approx(1)


@pytest.mark.parametrize("accept,reject", [(float("nan"), 0.5), (0.6, -0.1), (1.1, 0.5)])
def test_invalid_thresholds(accept, reject):
    with pytest.raises(ValueError):
        nli.check_claims("Claim", found(), scorer=Mock(), accept=accept, reject=reject)
