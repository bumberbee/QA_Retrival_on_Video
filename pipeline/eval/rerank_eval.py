"""Offline paired dev-120 retrieval evaluation; never rebuilds indexes or downloads models.

Do not launch a GPU run without explicit user approval. See notes/RERANK_PLAN.md.
"""

import argparse
import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from offline_common import (ROOT, VIDEOS, encode_batches, experiment_dir, load_backend,
                            read_json, write_json)
from eval_retrieval import context_trace, overlap, validate_questions

METRICS = ("hit1", "hit3", "mrr3", "context_success")


def measure(backend, index, question, found):
    context = backend.build_context(found)
    row = {"slides": [{k: slide[k] for k in ("segment_id", "start_sec", "end_sec", "rank",
                                           "matched_chunk_ids")} for slide in found["slides"]],
           "best_semantic": found["best_semantic"], "best_rerank": found.get("best_rerank"),
           "context": context, "context_chars": len(context),
           "context_evidence": found.get("context_evidence", [])}
    if question["type"] != "unanswerable":
        hits = [overlap(s["start_sec"], s["end_sec"], question["gold_spans"])
                for s in found["slides"][:3]]
        trace = context_trace(found, index, question, context)
        row.update(hit1=int(any(hits[:1])), hit3=int(any(hits)),
                   mrr3=next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
                   context_success=int(trace["recalled"]), context_trace=trace)
    return row


def paired_summary(rows):
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    n = len(answerable)
    before = {key: sum(r["baseline"][key] for r in answerable) / n if n else None for key in METRICS}
    after = {key: sum(r["rerank"][key] for r in answerable) / n if n else None for key in METRICS}
    return {"answerable": n, "unanswerable": len(rows) - n,
            "baseline": before, "rerank": after,
            "paired_delta": {key: sum(r["delta"][key] for r in answerable) / n if n else None
                             for key in METRICS},
            "improved": {key: sum(r["delta"][key] > 0 for r in answerable) for key in METRICS},
            "regressed": {key: sum(r["delta"][key] < 0 for r in answerable) for key in METRICS}}


def evaluate_pair(backend, folders, questions, vectors, *, scorer=None):
    """Both arms use the same immutable indexes, E5 vectors and retrieval options."""
    import numpy as np

    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(questions) or not np.isfinite(vectors).all():
        raise ValueError("Expected one finite query vector per question")
    options = backend.RetrievalOptions()
    rows = []
    backend.load_index.cache_clear()
    try:
        with patch.object(backend, "index_dir", side_effect=lambda video: folders[video]):
            for question, vector in zip(questions, vectors):
                index = backend.load_index(question["video"])
                if index.model != backend.E5_REPO:
                    raise ValueError("dev-120 evaluator requires existing multilingual-e5-small indexes")
                if index.vectors.ndim != 2 or index.vectors.shape[1] != len(vector):
                    raise ValueError("Query/index embedding dimensions do not match")
                fixed = type("EncodedQuery", (), {"encode": lambda self, text, v=vector: v})()
                row = {key: question[key] for key in ("id", "video", "type", "question")}
                for label, enabled in (("baseline", False), ("rerank", True)):
                    started = perf_counter()
                    with patch.object(backend, "RERANK", enabled):
                        found = backend.search(question["video"], question["question"], encoder=fixed,
                                               options=options, rerank_scorer=scorer)
                    row[label] = measure(backend, index, question, found)
                    row[label]["elapsed_sec"] = perf_counter() - started
                if question["type"] != "unanswerable":
                    row["delta"] = {key: row["rerank"][key] - row["baseline"][key] for key in METRICS}
                rows.append(row)
    finally:
        backend.load_index.cache_clear()
    return {"questions": rows, "overall": paired_summary(rows),
            "per_video": {video: paired_summary([r for r in rows if r["video"] == video])
                          for video in sorted({r["video"] for r in rows})},
            "per_type": {kind: paired_summary([r for r in rows if r["type"] == kind])
                         for kind in sorted({r["type"] for r in rows})}}


def query_vectors(backend, questions, questions_hash, cache=None, encoder_dir=None):
    """Optional audited NPZ cache, otherwise local-only ONNX E5 query encoding."""
    import numpy as np

    if cache is not None:
        with np.load(cache, allow_pickle=False) as data:
            if (data["question_ids"].tolist() != [q["id"] for q in questions]
                    or str(data["model"].item()) != backend.E5_REPO
                    or str(data["revision"].item()) != backend.E5_REVISION
                    or str(data["questions_sha256"].item()) != questions_hash):
                raise ValueError("Query cache IDs/model/revision/questions checksum mismatch")
            return data["vectors"].copy()
    from huggingface_hub import hf_hub_download

    def local_file(repo, filename, **kwargs):
        if encoder_dir is not None:
            path = encoder_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing local E5 asset: {path}")
            return str(path)
        return hf_hub_download(repo, filename, **kwargs, local_files_only=True)

    with patch("huggingface_hub.hf_hub_download", side_effect=local_file):
        encoder = backend.OnnxEncoder()
    return encode_batches(encoder, [q["question"] for q in questions], "query: ")


def markdown(report):
    lines = ["# Paired dev-120 reranking", "", f"Questions: {report['question_status']}", "",
             "Metrics exclude unanswerable questions. MRR@3 stops at the third selected slide.",
             "Context success requires gold time overlap and all annotated terms in retained text",
             "from the requested evidence source (the existing context_trace definition).", "",
             "| Metric | Rerank off | Rerank on | Paired delta | Improved | Regressed |",
             "|---|---:|---:|---:|---:|---:|"]
    summary = report["overall"]
    for key in METRICS:
        if summary["baseline"][key] is not None:
            lines.append(f"| {key} | {summary['baseline'][key]:.4f} | {summary['rerank'][key]:.4f} | "
                         f"{summary['paired_delta'][key]:+.4f} | {summary['improved'][key]} | "
                         f"{summary['regressed'][key]} |")
    lines += ["", f"Answerable: {summary['answerable']}; unanswerable: {summary['unanswerable']}.",
              "Unanswerable rows retain best scores for later A8 calibration; no threshold was tuned.",
              "JSON includes per-question paired deltas, context traces, per-video and per-type metrics.",
              "Sigmoid relevance scores are uncalibrated; this run measures retrieval, not Q8 accuracy."]
    return "\n".join(lines) + "\n"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(*, scorer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index-root", type=Path, help="Cached export root containing <video>/05_index")
    group.add_argument("--index-dir", type=Path, help="One cached 05_index; requires one --videos value")
    parser.add_argument("--videos", nargs="+", choices=VIDEOS, default=VIDEOS)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions_dev.json")
    parser.add_argument("--query-vectors", type=Path, help="Audited NPZ cache; see RERANK_PLAN.md")
    parser.add_argument("--encoder-dir", type=Path, help="Local E5 snapshot with onnx assets")
    parser.add_argument("--reranker-dir", type=Path, help="Already provisioned local BGE snapshot/export")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.index_dir and len(args.videos) != 1:
        parser.error("--index-dir requires exactly one --videos value")
    data = read_json(args.questions)
    validate_questions(data)
    if len(data["questions"]) != 120:
        parser.error("Expected the full dev-120 question bank; do not use the held-out test set")
    videos = list(dict.fromkeys(args.videos))
    questions = [q for q in data["questions"] if q["video"] in videos]
    folders = {v: (args.index_dir or args.index_root / v / "05_index").resolve() for v in videos}
    inputs = {}
    for video, folder in folders.items():
        names = ["chunks.jsonl", "embeddings.npy", "lexical.sqlite3", "manifest.json", "lecture/segments.json"]
        if (folder / "lecture/words.json").exists():
            names.append("lecture/words.json")
        inputs[video] = {"path": str(folder), "sha256": {name: sha256(folder / name) for name in names}}
    backend = load_backend()
    reranker = importlib.import_module(backend.__package__ + ".rerank")
    if args.reranker_dir:
        os.environ["LECTURE_RERANK_MODEL_DIR"] = str(args.reranker_dir.resolve())
    questions_hash = sha256(args.questions)
    vectors = query_vectors(backend, questions, questions_hash, args.query_vectors, args.encoder_dir)
    report = evaluate_pair(backend, folders, questions, vectors, scorer=scorer)
    report.update(question_status=data["status"], questions_sha256=questions_hash,
                  measured_utc=datetime.now(timezone.utc).isoformat(), index_inputs=inputs,
                  sources_sha256={str(p.relative_to(ROOT.parent)): sha256(p) for p in
                                  (Path(__file__), ROOT / "eval/offline_common.py", ROOT / "eval/eval_retrieval.py",
                                   Path(backend.__file__), Path(reranker.__file__))},
                  retrieval={"options": backend.asdict(backend.RetrievalOptions()),
                             "max_slides": backend.MAX_SLIDES, "rerank_candidates": backend.RERANK_CANDIDATES,
                             "rrf_k": backend.RRF_K, "semantic_weight": backend.SEMANTIC_WEIGHT,
                             "lexical_weight": backend.LEXICAL_WEIGHT},
                  encoder={"model": backend.E5_REPO, "revision": backend.E5_REVISION,
                           "query_cache_sha256": sha256(args.query_vectors) if args.query_vectors else None},
                  reranker={"model": reranker.RERANK_REPO, "revision": reranker.RERANK_REVISION,
                            "local_dir": os.environ.get("LECTURE_RERANK_MODEL_DIR"),
                            "max_tokens": reranker.MAX_TOKENS, "batch_size": reranker.BATCH_SIZE,
                            "scorer": getattr(scorer, "backend_name", "local-only CPU (ONNX preferred)"),
                            "calibrated": False})
    output = experiment_dir(args.name, args.output_root)
    write_json(output / "rerank.json", report)
    (output / "rerank.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
