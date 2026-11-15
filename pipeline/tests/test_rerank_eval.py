"""Paired metric and notebook contracts using only synthetic indexes/fake scorers."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from offline_common import load_backend
from rerank_eval import evaluate_pair, measure, paired_summary, query_vectors
from build_rerank_notebook import build


def cached_index(folder):
    folder.mkdir(parents=True)
    (folder / "lecture").mkdir()
    segments = [{"segment_id": f"v:s{i}", "start_sec": i * 100, "end_sec": i * 100 + 60,
                 "visual": {"title_clean": f"Title {i}"}, "speech": {}} for i in range(4)]
    chunks = [{"chunk_id": f"c{i}", "segment_id": f"v:s{i}", "source_type": "ocr",
               "text": "Bayes" if i == 3 else f"unrelated {i}", "start_sec": i * 100,
               "end_sec": i * 100 + 60, "title": f"Title {i}"} for i in range(4)]
    (folder / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    (folder / "lecture/segments.json").write_text(json.dumps(segments), encoding="utf-8")
    np.save(folder / "embeddings.npy", np.array([[1, 0], [0.8, 0], [0.6, 0], [0.1, 0]]))
    return chunks


def test_paired_rebuild_changes_retrieval_and_context_without_changing_indexes(tmp_path):
    backend = load_backend()
    folder = tmp_path / "v/05_index"
    cached_index(folder)
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    questions = [{"id": "q1", "video": "v", "type": "slide", "question": "Bayes?",
                  "gold_spans": [[310, 315]], "key_terms": ["Bayes"], "evidence_source": "ocr"},
                 {"id": "q2", "video": "v", "type": "unanswerable", "question": "Outside?",
                  "gold_spans": [], "key_terms": []}]
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [4.0 if text == "Bayes" else -4.0 for _, text in pairs]

    with patch.object(backend, "lexical_hits", return_value=["c0", "c1"]), \
            patch.object(backend, "RERANK", True):
        report = evaluate_pair(backend, {"v": folder}, questions, np.array([[1, 0], [1, 0]]), scorer=scorer)
        assert backend.RERANK is True
    summary = report["overall"]
    assert summary["answerable"] == summary["unanswerable"] == 1
    for key in ("hit1", "hit3", "mrr3", "context_success"):
        assert summary["baseline"][key] == 0
        assert summary["rerank"][key] == summary["paired_delta"][key] == 1
        assert summary["improved"][key] == 1
        assert summary["regressed"][key] == 0
    row = report["questions"][0]
    assert row["baseline"]["best_rerank"] is None
    assert row["rerank"]["best_rerank"] > 0.9
    assert row["rerank"]["slides"][0]["segment_id"] == "v:s3"
    assert "delta" not in report["questions"][1]
    assert len(calls) == 8  # Only the two reranked arms, never baseline.
    assert report["per_video"]["v"] == summary
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before


def test_mrr_stops_at_three_and_context_requires_retained_terms(tmp_path):
    backend = load_backend()
    chunks = cached_index(tmp_path / "index")
    index = backend.LectureIndex(tmp_path / "index", chunks, None, [], {})
    slides = [{"segment_id": f"v:s{i}", "rank": i, "start_sec": i * 100, "end_sec": i * 100 + 60,
               "matched_chunk_ids": [f"c{i}"], "title": "", "speech_text": "", "ocr_text": "",
               "figures": []} for i in range(4)]
    question = {"type": "slide", "gold_spans": [[310, 315]], "key_terms": ["Bayes"], "evidence_source": "ocr"}
    found = {"slides": slides, "best_semantic": 1.0}
    with patch.object(backend, "build_context", return_value=""):
        row = measure(backend, index, question, found)
    assert (row["hit1"], row["hit3"], row["mrr3"], row["context_success"]) == (0, 0, 0, 0)


def test_paired_summary_excludes_unanswerable_and_reports_negative_deltas():
    before = dict(hit1=1, hit3=1, mrr3=1.0, context_success=1)
    after = dict(hit1=0, hit3=1, mrr3=1 / 3, context_success=0)
    row = {"type": "slide", "baseline": before, "rerank": after,
           "delta": {key: after[key] - before[key] for key in before}}
    summary = paired_summary([row, {"type": "unanswerable"}])
    assert summary["paired_delta"]["mrr3"] == pytest.approx(-2 / 3)
    assert summary["regressed"]["context_success"] == 1
    assert paired_summary([{"type": "unanswerable"}])["baseline"]["hit1"] is None


@pytest.mark.parametrize("vectors", [np.array([[float("nan"), 0]]), np.ones(2), np.ones((2, 2))])
def test_invalid_vectors_fail_before_search(vectors):
    with pytest.raises(ValueError, match="query vector"):
        evaluate_pair(load_backend(), {}, [{"id": "q1"}], vectors, scorer=lambda pairs: [0] * len(pairs))


@pytest.mark.parametrize("field,value", [("question_ids", ["wrong"]), ("model", "wrong"),
                                       ("revision", "wrong"), ("questions_sha256", "wrong")])
def test_audited_query_cache_is_checked_without_loading_encoder(tmp_path, field, value):
    backend = load_backend()
    values = dict(vectors=np.array([[1, 0]]), question_ids=["q1"], model=backend.E5_REPO,
                  revision=backend.E5_REVISION, questions_sha256="checksum")
    cache = tmp_path / "queries.npz"
    np.savez(cache, **values)
    with patch.object(backend, "OnnxEncoder", side_effect=AssertionError("No model loading")):
        assert query_vectors(backend, [{"id": "q1"}], "checksum", cache).tolist() == [[1, 0]]
        values[field] = value
        np.savez(cache, **values)
        with pytest.raises(ValueError, match="mismatch"):
            query_vectors(backend, [{"id": "q1"}], "checksum", cache)


def test_notebook_builder_embeds_dependencies_without_executing_or_downloading(tmp_path):
    target = build(tmp_path / "rerank.ipynb")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) == 3
    for cell in data["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(source, "notebook-cell", "exec")
    config = "".join(data["cells"][1]["source"])
    code = "".join(data["cells"][2]["source"])
    assert "USE_GPU = False" in config and "GPU_RUN_APPROVED = False" in config
    assert "backend/app/features/search/rerank.py" in code
    assert "backend/app/features/search/retrieval.py" in code
    assert "local_files_only=True" in code
    assert "pip install" not in code
