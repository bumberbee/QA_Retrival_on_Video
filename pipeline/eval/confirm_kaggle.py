"""CPU-only confirmation evaluation; run inside the generated private Kaggle kernel."""
import hashlib
import json
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

from offline_common import (ModelAdapter, ROOT, VIDEOS, encode_batches, load_backend,
                            load_notebook, read_json, write_json, module_path)
import eval_retrieval as dev
import legacy_offline as legacy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(index_root, backend, dev_questions, legacy_questions, vectors):
    folders = {v: index_root / v / "05_index" for v in VIDEOS}
    backend.load_index.cache_clear()
    dev_rows, legacy_rows, strict_rows = [], [], []
    with patch.object(backend, "index_dir", side_effect=lambda v: folders[v]):
        for q, vector in zip(dev_questions, vectors["dev"]):
            fixed = types.SimpleNamespace(encode=lambda text, v=vector: v)
            found = backend.search(q["video"], q["question"], encoder=fixed,
                                   options=backend.retrieval_options(None))
            context = backend.build_context(found)
            row = {"id": q["id"], "type": q["type"], "context": context,
                   "best_semantic": found["best_semantic"],
                   "top_rrf": found["slides"][0]["score"] if found["slides"] else 0}
            if q["type"] != "unanswerable":
                hits = [dev.overlap(s["start_sec"], s["end_sec"], q["gold_spans"])
                        for s in found["slides"]]
                trace = dev.context_trace(found, backend.load_index(q["video"]), q, context)
                row.update(hit1=int(any(hits[:1])), hit3=int(any(hits[:3])),
                           hit5=int(any(hits[:5])),
                           mrr=next((1 / (i + 1) for i, h in enumerate(hits) if h), 0),
                           context_recall=int(trace["recalled"]))
            dev_rows.append(row)
        for q, vector in zip(legacy_questions, vectors["legacy"]):
            fixed = types.SimpleNamespace(encode=lambda text, v=vector: v)
            found = backend.search(q["video"], q["q"], q.get("current_time"), encoder=fixed)
            row = legacy.score_one(q, found, backend.build_context(found))
            legacy_rows.append(row)
            strict = dict(row)
            if q["answer"] is not None:
                a, b = q["answer"]
                hits = [max(0, min(s["end_sec"], b) - max(s["start_sec"], a)) >= 0.5
                        for s in found["slides"]]
                strict.update(hit1=int(any(hits[:1])), hit3=int(any(hits[:3])),
                              mrr=next((1 / (i + 1) for i, h in enumerate(hits) if h), 0),
                              retained_time_overlap=int(any(e.get("text") and
                                  max(0, min(e["end_sec"], b) - max(e["start_sec"], a)) >= 0.5
                                  for e in found.get("context_evidence", []))))
            strict_rows.append(strict)
    backend.load_index.cache_clear()
    return {"dev120": dev.aggregate(dev_rows),
            "legacy49_any_positive": legacy.aggregate(legacy_rows),
            "legacy49_strict_0_5s": legacy.aggregate(strict_rows)}


def rebuild(export, target, encoder):
    import numpy as np
    indexing = load_notebook("indexing")
    counts = {}
    for video in VIDEOS:
        old = export / video / "05_index"
        before = [json.loads(line) for line in (old / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        removed = [c for c in before if c["source_type"] == "speech"
                   and not any(x.isalnum() for x in c["text"])]
        expected = [c for c in before if c not in removed]
        parameters = read_json(old / "manifest.json")["parameters"]
        if (parameters["embedding_model"] != "intfloat/multilingual-e5-small"
                or parameters["resolved_revision"] != load_backend().E5_REVISION):
            raise RuntimeError("Production model/config does not match the frozen query encoder")
        adapter = ModelAdapter(encoder, [old])
        # No new vectors are allowed: this experiment removes only junk evidence.
        adapter.encode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Unexpected changed embedding text"))
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = lambda *a, **k: adapter
        indexing._SHARED_MODELS = {}
        options = {k: parameters[k] for k in (
            "embedding_model", "revision", "chunk_tokens", "overlap_tokens",
            "speech_chunk_tokens", "speech_maximum_seconds", "generated_semantic_only",
            "visual_state_timing")}
        options["include_generated_descriptions"] = parameters["visual_description_policy"] != "human_verified_only"
        options["dedupe_previous"] = False
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            directory, manifest, after = indexing.run_indexing(
                export / video / "04_bundle", target / video, video, device="cpu", **options)
        if after != expected:
            changed = [c["chunk_id"] for c, e in zip(after, expected) if c != e]
            raise RuntimeError(f"Rebuild changed surviving evidence for {video}: {changed[:5]}")
        original_vectors = np.load(old / "embeddings.npy", allow_pickle=False)
        kept = [i for i, c in enumerate(before) if c not in removed]
        if not np.array_equal(np.load(directory / "embeddings.npy", allow_pickle=False), original_vectors[kept]):
            raise RuntimeError("Surviving production vectors changed")
        counts[video] = {"before": len(before), "after": len(after), "removed": len(removed),
                         "removed_ids": [c["chunk_id"] for c in removed]}
        print(video, counts[video]["before"], "->", counts[video]["after"], flush=True)
    return counts


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU-only environment required")
    work = ROOT.parent.parent
    export, fixed = work / "export", work / "fixed"
    archives = list(Path("/kaggle/input").rglob("fast_pipeline_export.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one attached confirmation export, found {len(archives)}")
    with zipfile.ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            destination = (export / item.filename).resolve()
            if not destination.is_relative_to(export.resolve()):
                raise RuntimeError("Unsafe export path")
        archive.extractall(export)
    # The compact export deliberately strips WAVs, but stage-04 manifests retain
    # their checksums. Restore only missing manifest assets from the same attached
    # run's checkpoint so run_indexing can perform its normal validation/copy.
    missing = []
    for video in VIDEOS:
        bundle = export / video / "04_bundle"
        missing.extend(f"{video}/04_bundle/{name}" for name in read_json(bundle / "manifest.json")["files"]
                       if not (bundle / name).is_file())
    if missing:
        checkpoints = list(Path("/kaggle/input").rglob("fast_pipeline_checkpoint.zip"))
        if len(checkpoints) != 1:
            raise RuntimeError("Missing stage-04 assets require the same attached confirm checkpoint")
        with zipfile.ZipFile(checkpoints[0]) as checkpoint:
            for name in missing:
                destination = (export / name).resolve()
                if not destination.is_relative_to(export.resolve()):
                    raise RuntimeError("Unsafe checkpoint asset path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open(name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
        print("Restored", len(missing), "manifest assets from attached checkpoint", flush=True)
    data = read_json(ROOT / "eval/questions_dev.json")
    dev.validate_questions(data)
    dev_questions = [q for q in data["questions"] if q["video"] in VIDEOS]
    legacy_path = ROOT / "eval/questions_legacy.json"
    legacy_questions = legacy.select_questions(read_json(legacy_path), VIDEOS)
    if len(dev_questions) != 120 or len(legacy_questions) != 49:
        raise RuntimeError("Frozen question scopes changed")
    backend = load_backend()
    # Existing offline evaluators disable downloads at import time; this remote CPU
    # run explicitly needs the frozen ONNX model from Hugging Face.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    encoder = backend.OnnxEncoder()
    vectors = {"dev": encode_batches(encoder, [q["question"] for q in dev_questions], "query: "),
               "legacy": encode_batches(encoder, [q["q"] for q in legacy_questions], "query: ")}
    summary = {"status": "RUNNING", "gpu_hours": 0, "kernel_source": "lecture-rag-fast-10-confirm-20261102",
               "encoder": {"model": backend.E5_REPO, "revision": backend.E5_REVISION, "device": "CPU ONNX"},
               "question_hashes": {"dev": digest(ROOT / "eval/questions_dev.json"), "legacy": digest(legacy_path)},
               "source_hashes": {"indexing": digest(module_path("indexing.py")),
                                 "backend": digest(backend.__file__)},
               "strict_definition": "Intersection duration >= 0.5 seconds; archived boundary diagnostic",
               "restored_stage04_assets_from_checkpoint": len(missing),
               "rebuild": "Stage 05 from unchanged stage 04; exact surviving chunks and production vectors verified"}
    summary["as_is"] = evaluate(export, backend, dev_questions, legacy_questions, vectors)
    write_json(Path("/kaggle/working/confirm-eval-summary.json"), summary)
    print("As-is evaluation finished", flush=True)
    summary["chunks"] = rebuild(export, fixed, encoder)
    summary["fixed"] = evaluate(fixed, backend, dev_questions, legacy_questions, vectors)
    summary["chunk_totals"] = {k: sum(c[k] for c in summary["chunks"].values()) for k in ("before", "after", "removed")}
    summary["status"] = "COMPLETE"
    write_json(Path("/kaggle/working/confirm-eval-summary.json"), summary)
    print("Complete", summary["chunk_totals"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = Path("/kaggle/working/confirm-eval-summary.json")
        summary = read_json(path) if path.exists() else {"gpu_hours": 0}
        summary.update(status="ERROR", error=str(error))
        write_json(path, summary)
        raise
