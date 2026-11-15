from tests import _isolate  # noqa: F401

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.features.search import retrieval


def fixture_index(count=4):
    segments = [{"segment_id": f"v:s{i}", "start_sec": i * 100, "end_sec": i * 100 + 60,
                 "visual": {"title_clean": f"Slide {i}"}, "speech": {}}
                for i in range(count)]
    chunks = [{"chunk_id": f"c{i}", "segment_id": f"v:s{i}", "source_type": "ocr",
               "text": f"Unique evidence {i}", "start_sec": i * 100,
               "end_sec": i * 100 + 60, "title": f"Slide {i}"}
              for i in range(count)]
    vectors = np.column_stack([np.linspace(1.0, 0.1, count), np.zeros(count)])
    return retrieval.LectureIndex(Path("cached/index"), chunks, vectors, segments,
                                  {s["segment_id"]: s for s in segments})


class FixedEncoder:
    def encode(self, question):
        return np.array([1.0, 0.0])


def search_fixture(index=None, **kwargs):
    index = index or fixture_index()
    with patch.object(retrieval, "load_index", return_value=index), \
            patch.object(retrieval, "lexical_hits", return_value=["c0", "c1"]):
        return retrieval.search("v", "Which evidence?", encoder=FixedEncoder(), **kwargs)


def serialized_fixture(**kwargs):
    found = search_fixture(**kwargs)
    context, evidence_map = retrieval.build_context_with_ids(found)
    return json.dumps({"result": found, "context": context, "evidence_map": evidence_map},
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("flag", [None, "0", "off", "false"])
def test_disabled_rerank_is_byte_identical_to_pre_a3_snapshot(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("LECTURE_RERANK", raising=False)
    else:
        monkeypatch.setenv("LECTURE_RERANK", flag)
    monkeypatch.setattr(retrieval, "RERANK", False)
    with patch.object(retrieval, "rerank", side_effect=AssertionError("Reranker called while off")):
        actual = serialized_fixture()
    expected = Path(__file__).with_name("fixtures") / "retrieval_pre_a3.json"
    assert actual == expected.read_bytes()


def test_rerank_changes_slide_selection_and_preserves_scores(monkeypatch):
    monkeypatch.setattr(retrieval, "RERANK", True)
    index = fixture_index()
    original = copy.deepcopy(index.chunks)
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [4.0 if text.endswith("3") else -2.0 for _, text in pairs]

    found = search_fixture(index, rerank_scorer=scorer)
    assert found["slides"][0]["segment_id"] == "v:s3"
    assert found["best_rerank"] == pytest.approx(1 / (1 + np.exp(-4)))
    chunk = found["slides"][0]["evidence_candidates"][0]
    assert chunk["rerank_score"] == found["best_rerank"]
    assert chunk["rerank_rank"] == 1
    assert chunk["selection_reason"] == "matched"
    assert found["slides"][0]["score"] > 0
    assert index.chunks == original
    retrieval.build_context(found)
    retained = next(c for c in found["context_evidence"] if c["chunk_id"] == "c3")
    assert retained["rerank_score"] == found["best_rerank"]
    assert len(calls) == 4


def test_only_top_30_rrf_candidates_are_reranked(monkeypatch):
    monkeypatch.setattr(retrieval, "RERANK", True)
    index = fixture_index(45)
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [0.0] * len(pairs)

    with patch.object(retrieval, "load_index", return_value=index), \
            patch.object(retrieval, "lexical_hits", return_value=[f"c{i}" for i in range(20, 40)]):
        found = retrieval.search("v", "question", encoder=FixedEncoder(), rerank_scorer=scorer)
    assert len(calls) == retrieval.RERANK_CANDIDATES == 30
    assert found["best_rerank"] == 0.5


def test_temporal_pin_and_source_filter_survive_reranking(monkeypatch):
    monkeypatch.setattr(retrieval, "RERANK", True)
    index = fixture_index()
    index.chunks[1]["source_type"] = "speech"
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [4.0 if text.endswith("3") else -2.0 for _, text in pairs]

    with patch.object(retrieval, "load_index", return_value=index), \
            patch.object(retrieval, "lexical_hits", return_value=["c0", "c1"]):
        found = retrieval.search("v", "this slide", current_time=10, encoder=FixedEncoder(),
                                 sources=frozenset({"ocr"}), rerank_scorer=scorer)
    assert found["slides"][0]["segment_id"] == "v:s0"
    assert found["temporal"] is True
    assert all(text != "Unique evidence 1" for _, text in calls)


def test_empty_rerank_pool_never_loads_a_model(monkeypatch):
    monkeypatch.setattr(retrieval, "RERANK", True)
    found = search_fixture(sources=frozenset(), rerank_scorer=lambda pairs: pytest.fail("called"))
    assert found["slides"] == []
    assert "best_rerank" not in found
