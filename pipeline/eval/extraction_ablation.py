"""CPU extraction ablations. Immutable exports, frozen gold, one video per rebuild.

All generated artifacts live on D:. No application/cloud/config imports. Stage-01
replay is the documented eval/segmentation/experiment.py protocol with fresh D: storage.
OCR swaps require an identical representative frame hash or timestamp; unmatched
records are retained and explicitly counted, never copied from a nearby state.
"""
import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

from offline_common import (ROOT, VIDEOS, encode_batches, load_backend, load_notebook,
                            production_config, read_json, write_json)
from rebuild_offline import copy_input_file, rebuild_video
from eval_retrieval import aggregate, context_trace, overlap
from legacy_offline import LEGACY, aggregate as legacy_aggregate, score_one, select_questions

BASE = Path("D:/thesis-data/run-20261102/ablation")
OLD = Path("D:/thesis-data/run-20261020/export")
NEW = Path("D:/thesis-data/run-20261102/full/lecture_rag")
MEDIUM = Path("D:/thesis-data/run-20261102/eval/medium")
OLD_INDEX = Path("D:/thesis-data/runs-exp/phase3-m-combined-subscript")
NEW_INDEX = MEDIUM


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpu_encoder():
    """Bound ONNX threads and its workspace on the shared local machine."""
    import onnxruntime
    original = onnxruntime.InferenceSession
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    with patch.object(onnxruntime, "InferenceSession",
                      side_effect=lambda *a, **kw: original(*a, sess_options=options, **kw)):
        encoder = load_backend().OnnxEncoder()
    return CachedEncoder(encoder)


class CachedEncoder:
    """Reuse exact-text vectors from checked CPU indexes, independent of chunk IDs."""
    def __init__(self, encoder):
        import numpy as np
        self.encoder, self.cache = encoder, {}
        self.hits = self.misses = 0
        backend = load_backend()
        for root in (OLD_INDEX, NEW_INDEX):
            for video in VIDEOS:
                folder = root / video / "05_index"
                manifest = read_json(folder / "manifest.json")
                parameters = manifest["parameters"]
                if (parameters["embedding_model"], parameters["revision"]) != (backend.E5_REPO, backend.E5_REVISION):
                    raise ValueError("Cached encoder model/revision mismatch")
                for filename in ("chunks.jsonl", "embeddings.npy"):
                    if sha(folder / filename) != manifest["files"][filename]:
                        raise ValueError("Cached encoder asset checksum mismatch")
                chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
                vectors = np.load(folder / "embeddings.npy", allow_pickle=False)
                if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all():
                    raise ValueError("Cached vector shape/finiteness mismatch")
                for chunk, vector in zip(chunks, vectors):
                    self.cache[chunk["embedding_text"]] = vector.copy()

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def encode_many(self, texts, prefix=None):
        import numpy as np
        prefix = self.encoder.prefix if prefix is None else prefix
        keys = [prefix + text for text in texts]
        unknown = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if unknown:
            vectors = self.encoder.encode_many(unknown, prefix="")
            self.cache.update(zip(unknown, vectors))
        self.misses += len(unknown)
        self.hits += len(keys) - len(unknown)
        return np.stack([self.cache[k] for k in keys])


def scoped_options(video):
    frozen = BASE / "original-stage01-options.json"
    if frozen.exists():
        return read_json(frozen).get(video, {})
    tree = ast.parse((ROOT / "builders/build_fast_pipeline.py").read_text(encoding="utf-8"))
    source = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
    options = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == "STAGE01_VIDEO_OPTIONS")
    write_json(frozen, options)
    return options.get(video, {})


def match_frame(target, donors, tolerance=0.001):
    """Same image first, otherwise same decoded frame time (not interval overlap)."""
    exact = [d for d in donors if d["image_sha256"] == target["image_sha256"]]
    if exact:
        return min(exact, key=lambda d: abs(d["keyframe_time_sec"] - target["keyframe_time_sec"])), "hash"
    exact = [d for d in donors if abs(d["keyframe_time_sec"] - target["keyframe_time_sec"]) <= tolerance]
    if exact:
        return exact[0], "time"
    return None, "unmatched"


def copy_tree(source, target):
    shutil.copytree(source, target, copy_function=copy_input_file)


def legacy_frame_plan(groups, legacy_groups):
    """Never reuse a deleted state: both full interval and final state must match."""
    by_interval = {(g["start_sec"], g["end_sec"]): g for g in legacy_groups}
    plan = {}
    for group in groups:
        old = by_interval.get((group["start_sec"], group["end_sec"]))
        state = group.get("reveal_states", [group])[-1]
        if old and state["start_sec"] <= old["keyframe_time_sec"] < state["end_sec"]:
            plan[group["segment_id"]] = old
    return plan


def reuse_unchanged_frames(directory, video, common):
    """Keep new partitions; reuse archived full-frame representatives where safe."""
    manifest = common.load_stage(directory, 1, video)
    if manifest["parameters"].get("legacy_frames_for_unchanged_groups"):
        return
    legacy_dir = OLD / video / "01_slides"
    legacy_manifest = common.load_stage(legacy_dir, 1, video)
    common.require_same_source(manifest, legacy_manifest)
    groups = read_json(directory / "intervals.json")
    plan = legacy_frame_plan(groups, read_json(legacy_dir / "intervals.json"))
    for group in groups:
        old = plan.get(group["segment_id"])
        if not old:
            continue
        path = "legacy_frames/" + group["segment_id"].split(":")[-1] + Path(old["image_path"]).suffix
        dest = directory / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_input_file(legacy_dir / old["image_path"], dest)
        group["superseded_roi_frame"] = {k: group.get(k) for k in
                                          ("image_path", "image_sha256", "keyframe_time_sec", "content_quality")}
        fields = {k: old[k] for k in ("image_sha256", "keyframe_time_sec", "sharpness_score")}
        fields["image_path"] = path
        group.update(fields)
        group.pop("content_quality", None)
        state = group.get("reveal_states", [group])[-1]
        state.update(fields)
        state.pop("content_quality", None)
        state["content_observed_sec"] = old["keyframe_time_sec"]
        state["content_onset_bounds_sec"] = [state["start_sec"], old["keyframe_time_sec"]]
        windows = group.get("visual_evidence_windows", [])
        if windows:
            windows[-1].update(image_path=path, observed_sec=old["keyframe_time_sec"])
        group["legacy_frame_source"] = old["segment_id"]
    common.write_json(directory / "intervals.json", groups)
    with (directory.parent / "frame-reuse.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        common.finish_stage(directory, 1, video, manifest["source"],
                            {**manifest["parameters"], "legacy_frames_for_unchanged_groups": True,
                             "native_stage01_artifact_id": manifest["artifact_id"],
                             "legacy_stage01_artifact_id": legacy_manifest["artifact_id"],
                             "reused_frames": len(plan)})


def detect(video):
    """Decode once, lossless signatures, release the entire cache before next video."""
    import cv2
    import numpy as np
    m = load_notebook("detection")
    cv2.setNumThreads(1)
    folder = BASE / "detection" / video
    if (folder / "report.json").exists():
        return read_json(folder / "report.json")
    folder.mkdir(parents=True, exist_ok=True)
    original_scan, original_detect = m.scan_frames, m.detect_stable_boundaries
    cached = []
    scope = scoped_options(video)
    content_options = {"roi": scope.get("board_options", scope.get("keyframe_options", {})).get("roi", (0, 0, 1, 1)),
                       "mode": "board" if "board_options" in scope else "generic"}

    def preserve_scan(path, *args, **kwargs):
        path = Path(path).resolve()
        if path.name == "_scan" and path.is_relative_to(folder.resolve()):
            return
        raise ValueError(f"Unexpected detection deletion: {path}")

    m.shutil = types.SimpleNamespace(**{n: getattr(m.shutil, n) for n in dir(m.shutil) if not n.startswith("_")})
    m.shutil.rmtree = preserve_scan

    def capture(*args, **kwargs):
        kwargs["content_options"] = content_options
        for sample in original_scan(*args, **kwargs):
            stored = {k: v for k, v in sample.items() if k not in ("signature", "fine")}
            stored["signature"] = (sample["signature"] * 255).round().astype("uint8")
            cached.append(stored)
            yield sample

    first = True

    def capture_detect(samples, *args, **kwargs):
        nonlocal first
        result = original_detect(samples, *args, **kwargs)
        if first:
            by_time = {s["time_sec"]: s for s in cached}
            for candidate in result[1]:
                by_time[candidate["time_sec"]]["fine"] = [
                    (t, (sig * 255).round().astype("uint8")) for t, sig in candidate["fine"]]
            first = False
        return result

    m.scan_frames, m.detect_stable_boundaries = capture, capture_detect
    started = time.monotonic()
    with (folder / "detection.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        baseline_dir, baseline_manifest, before = m.run_slide_detection(
            Path("D:/thesis-data/mp4-extra") / (video + ".mp4"), folder / "legacy", video)
    source = baseline_manifest["source"]
    archived = read_json(OLD / video / "01_slides/intervals.json")
    if before != archived:
        write_json(folder / "baseline-difference.json", {"actual": before, "archived": archived})
        raise ValueError(f"CPU legacy detector parity failed: {video}")

    def replay(video_path, source, directory, *args, records=None, **kwargs):
        for stored in cached:
            sample = {**stored, "signature": stored["signature"].astype("float32") / 255,
                      "fine": [(t, sig.astype("float32") / 255) for t, sig in stored.get("fine", [])]}
            sample["image_path"] = str((baseline_dir / stored["image_path"]).resolve())
            if records is not None:
                records.append({**stored, "image_path": sample["image_path"]})
            yield sample

    m.scan_frames, m.detect_stable_boundaries = replay, original_detect
    m.probe_video = lambda _: source
    rows = {"legacy": {"groups": len(before), "archive_exact": True}}
    configurations = {"onsets": {"export_state_onsets": True},
                      "merge": {"export_state_onsets": True, "merge_same_content": True},
                      "scoped": {"export_state_onsets": True, **scope},
                      "full": {"export_state_onsets": True, "merge_same_content": True, **scope}}
    for name, config in configurations.items():
        t = time.monotonic()
        with (folder / "detection.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            _, manifest, intervals = m.run_slide_detection(Path("D:/thesis-data/mp4-extra") / (video + ".mp4"),
                                                         folder / name, video, **config)
        new = read_json(NEW / video / "01_slides/intervals.json")
        rows[name] = {"groups": len(intervals), "seconds": round(time.monotonic() - t, 2),
                      "new_run_exact": intervals == new, "config": config}
    report = {"video": video, "source_sha256": source["sha256"], "variants": rows,
              "total_seconds": round(time.monotonic() - started, 2), "samples": len(cached)}
    write_json(folder / "report.json", report)
    del cached, m
    gc.collect()
    return report


def materialize(video, stage01, primary, donor=None, name="native", donor_segments=None,
                legacy_unchanged_frames=False):
    """Fresh checked 01/02 export; all OCR assets retain their source provenance."""
    target = BASE / "exports" / name / video
    if (target / "coverage.json").exists():
        return target.parent, read_json(target / "coverage.json")
    common = load_notebook("alignment")
    inputs = [primary] + ([donor] if donor else [])
    pools = []
    for root in inputs:
        groups = read_json(root / video / "01_slides/intervals.json")
        if root == donor and donor_segments is not None:
            groups = [g for g in groups if g["segment_id"] in donor_segments]
        visuals = {r["segment_id"]: r for r in read_json(root / video / "02_visual/visual.json")}
        pools.append((root, groups, visuals))
    if (target / "01_slides").exists():
        actual = common.load_stage(target / "01_slides", 1, video)
        expected = common.load_stage(stage01, 1, video)
        if actual["artifact_id"] != expected["artifact_id"] and actual["parameters"].get(
                "native_stage01_artifact_id") != expected["artifact_id"]:
            raise ValueError("Refuse to resume a different detector export")
    else:
        copy_tree(stage01, target / "01_slides")
    if legacy_unchanged_frames:
        reuse_unchanged_frames(target / "01_slides", video, common)
    intervals = read_json(target / "01_slides/intervals.json")
    m1 = common.load_stage(target / "01_slides", 1, video)
    directory = target / "02_visual"
    directory.mkdir(parents=True, exist_ok=True)
    records, mapping = [], []
    for group in intervals:
        selected = None
        for root, groups, visuals in reversed(pools):
            found, method = match_frame(group, groups)
            if found:
                selected = root, found, visuals[found["segment_id"]], method
                break
        if selected is None:
            mapping.append({"target": group["segment_id"], "method": "missing"})
            continue
        root, found, visual, method = selected
        item = copy.deepcopy(visual)
        item.update(segment_id=group["segment_id"], occurrence_id=group["segment_id"] + ":visual")
        # Rename ancillary crops/regions so donors' segment numbering cannot collide.
        for obj, field in [(item, "content_image_path"), *[(r, "image_path") for r in item.get("regions", [])]]:
            if obj.get(field):
                old_path = obj[field]
                new_path = "ablation_assets/" + group["segment_id"].split(":")[-1] + "/" + old_path
                dest = directory / new_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                copy_input_file(root / video / "02_visual" / old_path, dest)
                obj[field] = new_path
        records.append(item)
        mapping.append({"target": group["segment_id"], "donor": found["segment_id"],
                        "source": str(root), "method": method,
                        "target_frame_sec": group["keyframe_time_sec"],
                        "source_frame_sec": found["keyframe_time_sec"]})
    missing = [r for r in mapping if r["method"] == "missing"]
    if missing:
        write_json(target / "missing.json", missing)
        raise ValueError(f"{video}/{name}: {len(missing)} frames lack cached OCR")
    common.write_json(directory / "visual.json", records)
    with (target / "materialize.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        common.finish_stage(directory, 2, video, m1["source"],
                            {"ablation": name, "ocr_sources": [str(p) for p in inputs]}, [m1],
                            {"records": len(records)})
    coverage = {"video": video, "groups": len(intervals),
                "primary": sum(r.get("source") == str(primary) for r in mapping),
                "swapped": sum(r.get("source") == str(donor) for r in mapping) if donor else 0,
                "hash": sum(r["method"] == "hash" for r in mapping),
                "time": sum(r["method"] == "time" for r in mapping), "mapping": mapping}
    write_json(target / "coverage.json", coverage)
    return target.parent, coverage


def evaluate(index_root, name, encoder=None):
    backend = load_backend()
    encoder = encoder or cpu_encoder()
    dev_data = read_json(ROOT / "eval/questions_dev.json")
    questions = dev_data["questions"]
    legacy = select_questions(read_json(LEGACY), VIDEOS)
    dv = encode_batches(encoder, [q["question"] for q in questions], "query: ", 8)
    lv = encode_batches(encoder, [q["q"] for q in legacy], "query: ", 8)
    reports = {}
    backend.load_index.cache_clear()
    with patch.object(backend, "index_dir", side_effect=lambda v: index_root / v / "05_index"):
        for label, qs, vectors in [("dev", questions, dv), ("legacy", legacy, lv)]:
            rows = []
            for video in VIDEOS:
                backend.load_index.cache_clear()
                for q, vector in zip(qs, vectors):
                    if q["video"] != video:
                        continue
                    fixed = type("Query", (), {"encode": lambda self, text, v=vector: v})()
                    found = backend.search(video, q.get("question", q.get("q")),
                                           q.get("current_time"), encoder=fixed)
                    context = backend.build_context(found)
                    if label == "dev":
                        hits = [overlap(s["start_sec"], s["end_sec"], q["gold_spans"]) for s in found["slides"]]
                        row = {"id": q["id"], "video": video, "type": q["type"], "question": q["question"],
                               "context": context, "context_chars": len(context),
                               "best_semantic": found["best_semantic"],
                               "top_rrf": found["slides"][0]["score"] if found["slides"] else 0}
                        if q["type"] != "unanswerable":
                            trace = context_trace(found, backend.load_index(video), q, context)
                            row.update(hit1=int(any(hits[:1])), hit3=int(any(hits[:3])), hit5=int(any(hits)),
                                       mrr=next((1 / (i + 1) for i, h in enumerate(hits) if h), 0),
                                       context_recall=int(trace["recalled"]), context_trace=trace)
                    else:
                        row = score_one(q, found, context)
                    row.update(slides=[{k: v for k, v in s.items() if k not in ("speech_text", "ocr_text", "figures")}
                                       for s in found["slides"]], context_evidence=found.get("context_evidence", []))
                    rows.append(row)
            reports[label] = {"questions": rows, "overall": (aggregate(rows) if label == "dev" else legacy_aggregate(rows)),
                              "questions_sha256": sha(ROOT / "eval/questions_dev.json" if label == "dev" else LEGACY),
                              "index_root": str(index_root),
                              "backend_sha256": sha(ROOT.parent / "backend/app/features/search/retrieval.py")}
            write_json(BASE / "eval" / name / (label + ".json"), reports[label])
    backend.load_index.cache_clear()
    summary = {label: {k: r["overall"].get(k) for k in ("hit1", "hit3", "mrr", "context_recall", "retained_time_overlap")}
               for label, r in reports.items()}
    write_json(BASE / "eval" / name / "summary.json", summary)
    return reports


def build(name, specs, encoder, refresh_vectors=False):
    output = BASE / "indexes" / name
    config = production_config()
    config["batch_size"] = 8
    rows = {}
    for video in VIDEOS:
        spec = specs[video]
        if spec.get("reuse"):
            dest = output / video
            if not dest.exists():
                dest.mkdir(parents=True)
                for stage in ("04_bundle", "05_index"):
                    copy_tree(Path(spec["reuse"]) / video / stage, dest / stage)
            rows[video] = {"reuse": spec["reuse"]}
            continue
        if ((output / video / "05_index/manifest.json").exists() and
                not (output / video / "05_index/_RUNNING.json").exists() and not refresh_vectors):
            rows[video] = {"resumed": True}
            continue
        root, coverage = materialize(video, Path(spec["stage01"]), Path(spec["primary"]),
                                     Path(spec["donor"]) if spec.get("donor") else None, name,
                                     donor_segments=spec.get("donor_segments"),
                                     legacy_unchanged_frames=spec.get("legacy_unchanged_frames", False))
        cfg = {**config, "visual_state_timing": spec.get("timing", True)}
        reference = Path(spec.get("vector_reference", NEW_INDEX if Path(spec["primary"]) == NEW else OLD_INDEX))
        alternate = OLD_INDEX if reference == NEW_INDEX else NEW_INDEX
        with (BASE / "build.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            rows[video] = rebuild_video(root, output, video, cfg, encoder,
                                       asr_export_root=BASE / "medium-cache",
                                       chunk_references=[r / video / "05_index" for r in (reference, alternate)])
        rows[video]["ocr_coverage"] = {k: v for k, v in coverage.items() if k != "mapping"}
        print(name, video, "rebuilt", flush=True)
        gc.collect()
    write_json(output / "rebuild_report.json", {"config": config, "specs": specs, "videos": rows})
    return evaluate(output, name, encoder)


def prepare_medium():
    for video in VIDEOS:
        dest = BASE / "medium-cache" / video / "03_audio"
        if not dest.exists():
            copy_tree(MEDIUM / video / "_inputs/03_audio", dest)


def audit():
    dev = {q["id"]: q for q in read_json(ROOT / "eval/questions_dev.json")["questions"]}
    legacy = {q["id"]: q for q in select_questions(read_json(LEGACY), VIDEOS)}
    result = {}
    for label, gold in [("dev", dev), ("legacy", legacy)]:
        before = read_json(BASE / "eval/old-current" / (label + ".json"))["questions"]
        after = {q["id"]: q for q in read_json(BASE / "eval/new-current" / (label + ".json"))["questions"]}
        rows = []
        for old in before:
            new = after[old["id"]]
            losses = [k for k in ("hit1", "hit3", "mrr", "context_recall", "retained_time_overlap")
                      if k in old and old[k] > new[k]]
            if not losses:
                continue
            question = gold[old["id"]]
            spans = question["gold_spans"] if label == "dev" else [question["answer"]]
            row = {"id": old["id"], "video": old["video"], "question": question.get("question", question.get("q")),
                   "gold": spans, "key_terms": question.get("key_terms", []), "losses": losses}
            old_groups = read_json(OLD / old["video"] / "01_slides/intervals.json")
            for side, found, root in [("old", old, OLD), ("new", new, NEW)]:
                groups = {g["segment_id"]: g for g in read_json(root / old["video"] / "01_slides/intervals.json")}
                top = []
                for s in found["slides"]:
                    g = groups[s["segment_id"]]
                    matched = [c for c in s["evidence_candidates"] if c["chunk_id"] in s["matched_chunk_ids"]]
                    prior = next(g0 for g0 in old_groups if g0["start_sec"] <= g["keyframe_time_sec"] < g0["end_sec"])
                    top.append({"group": s["segment_id"].split(":")[-1], "start": s["start_sec"], "end": s["end_sec"],
                                "frame": g["keyframe_time_sec"], "onset": g.get("representative_content_onset_sec"),
                                "card_hit": overlap(s["start_sec"], s["end_sec"], spans),
                                "matched_chunk_hit": any(overlap(c["start_sec"], c["end_sec"], spans) for c in matched),
                                "frame_in_gold": any(lo <= g["keyframe_time_sec"] < hi for lo, hi in spans),
                                "old_group_projected_hit": overlap(prior["start_sec"], prior["end_sec"], spans),
                                "matched_chunks": [{k: c[k] for k in ("source_type", "start_sec", "end_sec", "text")}
                                                   for c in matched[:3]]})
                row[side] = {"top3": top, "metrics": {k: found.get(k) for k in ("hit1", "hit3", "mrr", "context_recall")},
                             "evidence": [{k: c[k] for k in ("source_type", "start_sec", "end_sec", "text")}
                                          for c in found["context_evidence"]],
                             "trace": found.get("context_trace")}
            row["group_definition_sensitive"] = (old.get("hit3", 0) > new.get("hit3", 0) and
                                                any(t["old_group_projected_hit"] for t in row["new"]["top3"]))
            rows.append(row)
        result[label] = rows
    write_json(BASE / "metric-audit.json", result)
    print("audit", {k: len(v) for k, v in result.items()},
          "group-definition-sensitive", {k: sum(r["group_definition_sensitive"] for r in v) for k, v in result.items()})
    return result


def stage01_specs(variant):
    specs = {}
    for video in VIDEOS:
        stage = BASE / "detection" / video / variant / "01_slides"
        if not (stage / "manifest.json").exists():
            old, new = [read_json(r / video / "01_slides/intervals.json") for r in (OLD, NEW)]
            keys = ("segment_id", "start_sec", "end_sec", "keyframe_time_sec", "image_sha256")
            if [{k: g[k] for k in keys} for g in old] != [{k: g[k] for k in keys} for g in new]:
                raise ValueError(f"Missing CPU detector variant: {video}/{variant}")
            if variant != "legacy" and any(g["representative_content_onset_sec"] !=
                    h.get("reveal_states", [h])[-1]["start_sec"] for g, h in zip(new, old)):
                raise ValueError(f"Unreplayed onset differs from legacy state: {video}")
            stage = NEW / video / "01_slides" if variant != "legacy" else OLD / video / "01_slides"
        specs[video] = {"stage01": str(stage), "primary": str(OLD), "timing": variant != "legacy"}
        if variant == "scoped":
            # Keep old OCR on exactly shared frames; new frames have their native OCR.
            specs[video].update(primary=str(NEW), donor=str(OLD), vector_reference=str(OLD_INDEX))
    return specs


def validate_index(root, name):
    import numpy as np
    common = load_notebook("alignment")
    rows = {}
    for video in VIDEOS:
        stages = [root / video / s for s in ("04_bundle", "05_index")]
        m4, m5 = [common.load_stage(p, i + 4, video) for i, p in enumerate(stages)]
        common.require_same_source(m4, m5)
        common.require_parent(m5, m4)
        duration = m5["source"]["duration_sec"]
        if sha(Path("D:/thesis-data/mp4-extra") / (video + ".mp4")) != m5["source"]["sha256"]:
            raise ValueError("Final index source MP4 mismatch")
        segments = read_json(stages[0] / "segments.json")
        common.verify_intervals(segments, duration)
        words = read_json(stages[0] / "words.json")
        owned = [w for s in segments for w in s["speech"]["word_ids"]]
        if len(owned) != len(set(owned)) or set(owned) != {w["word_id"] for w in words}:
            raise ValueError("Final word ownership mismatch")
        chunks = [json.loads(line) for line in (stages[1] / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        vectors = np.load(stages[1] / "embeddings.npy", allow_pickle=False)
        if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Final vectors failed shape/finiteness/normalization")
        for chunk in chunks:
            if not 0 <= chunk["start_sec"] < chunk["end_sec"] <= duration:
                raise ValueError("Final chunk time outside source")
            if not (stages[1] / "lecture" / chunk["image_path"]).is_file():
                raise ValueError("Final chunk image missing")
        for word in words:
            if not 0 <= word["start_sec"] < word["end_sec"] <= duration:
                raise ValueError("Final word time outside source")
        rows[video] = {"pass": True, "groups": len(segments), "chunks": len(chunks), "words": len(words),
                       "source_sha256": m5["source"]["sha256"], "index_artifact_id": m5["artifact_id"]}
    write_json(BASE / "validation" / (name + ".json"), {"pass": True, "videos": rows})
    print(name, "validated", len(rows), "/10", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("detect", "evaluate", "build", "audit"))
    p.add_argument("--videos", nargs="+", default=VIDEOS, choices=VIDEOS)
    p.add_argument("--name", default="old-current")
    p.add_argument("--index-root", type=Path, default=OLD_INDEX)
    p.add_argument("--specs", type=Path)
    args = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.action == "detect":
        for video in args.videos:
            row = detect(video)
            print(video, {k: v["groups"] for k, v in row["variants"].items()}, row["total_seconds"], flush=True)
    elif args.action == "audit":
        audit()
    elif args.action == "evaluate":
        reports = evaluate(args.index_root, args.name)
        print({k: {m: round(v["overall"][m], 5) for m in ("hit1", "hit3", "mrr")} for k, v in reports.items()})
    else:
        prepare_medium()
        encoder = cpu_encoder()
        reports = build(args.name, read_json(args.specs), encoder)
        print({k: {m: round(v["overall"][m], 5) for m in ("hit1", "hit3", "mrr")} for k, v in reports.items()})


if __name__ == "__main__":
    main()
