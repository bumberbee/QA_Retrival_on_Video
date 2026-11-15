"""Rebuild production stages 04/05 from exported raw OCR/ASR, on CPU without torch."""

import os

for _name in set(os.environ) | {"SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
                               "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}:
    if _name.startswith(("SUPABASE_", "R2_")):
        os.environ[_name] = ""

import argparse
import json
import shutil
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

from offline_common import (ModelAdapter, ROOT, VIDEOS, encode_batches, experiment_dir,
                            load_backend, load_notebook, production_config, read_json, write_json, module_path)
from shared.source_paths import MODULE_PATHS


def compare_indexes(rebuilt, exported):
    import numpy as np
    def chunks(folder):
        return {c["chunk_id"]: c for c in
                [json.loads(line) for line in (folder / "chunks.jsonl").read_text(
                    encoding="utf-8").splitlines() if line]}
    actual, expected = chunks(rebuilt), chunks(exported)
    common = sorted(actual.keys() & expected.keys())
    changed = [{"chunk_id": cid, "fields": [key for key in ("text", "embedding_text", "encoded_tokens")
                                            if actual[cid][key] != expected[cid][key]]}
               for cid in common if any(actual[cid][key] != expected[cid][key]
                                        for key in ("text", "embedding_text", "encoded_tokens"))]
    av, ev = [np.load(p / "embeddings.npy", allow_pickle=False) for p in (rebuilt, exported)]
    ai = {cid: i for i, cid in enumerate(actual)}
    ei = {cid: i for i, cid in enumerate(expected)}
    cosines = [float(np.dot(av[ai[cid]], ev[ei[cid]]) /
                     (np.linalg.norm(av[ai[cid]]) * np.linalg.norm(ev[ei[cid]]))) for cid in common]
    return {"rebuilt_chunks": len(actual), "exported_chunks": len(expected),
            "missing_ids": sorted(expected.keys() - actual.keys()),
            "extra_ids": sorted(actual.keys() - expected.keys()), "changed": changed,
            "id_order_equal": list(actual) == list(expected),
            "minimum_cosine": min(cosines) if cosines else None,
            "mean_cosine": float(np.mean(cosines)) if cosines else None,
            "cosine_at_or_below_0_999": sum(c <= 0.999 for c in cosines),
            "pass": list(actual) == list(expected) and not changed
                    and bool(cosines) and min(cosines) > 0.999}


def compare_bundles(rebuilt, exported):
    actual, expected = [read_json(p / "segments.json") for p in (rebuilt, exported)]
    result = {"segment_field_differences": {
        key: sum(a.get(key) != b.get(key) for a, b in zip(actual, expected))
        for key in ("speech", "visual", "audio", "speech_topic")},
        "words_equal": read_json(rebuilt / "words.json") == read_json(exported / "words.json")}
    windows = [p / "speech_relevance.json" for p in (rebuilt, exported)]
    if all(p.exists() for p in windows):
        a, b = map(read_json, windows)
        result["relevance_windows_different"] = sum(x != y for x, y in zip(a, b))
        result["relevance_window_counts"] = [len(a), len(b)]
        result["maximum_topic_score_difference"] = max(abs(x["topic_score"] - y["topic_score"])
                                                         for x, y in zip(a, b))
        result["maximum_topic_z_difference"] = max(abs(x["topic_z"] - y["topic_z"])
                                                     for x, y in zip(a, b))
    return result


def copy_input_file(source, destination):
    """Share immutable large assets; always copy metadata before any re-signing."""
    if Path(source).suffix.lower() not in {".json", ".jsonl", ".npy", ".sqlite", ".sqlite3", ".db"}:
        if Path(destination).exists():
            if os.path.samefile(source, destination):
                return str(destination)
            raise FileExistsError(f"Refuse to overwrite an existing immutable asset: {destination}")
        try:
            os.link(source, destination)
            return str(destination)
        except OSError:
            pass
    return shutil.copy2(source, destination)


def prepare_inputs(export, target, video, mp4_root, ffmpeg, asr_export=None, resume=False):
    common, audio = load_notebook("alignment"), load_notebook("transcription")
    provenance = []
    for number, stage in enumerate(("01_slides", "02_visual", "03_audio"), 1):
        stage_export = Path(asr_export) if number == 3 and asr_export is not None else export
        source, dest = stage_export / video / stage, target / "_inputs" / stage
        manifest = common.load_stage(source, number, video, verify=False)
        for name, expected in manifest["files"].items():
            path = common.relative_asset(source, name)
            if path.is_file():
                if common.file_sha256(path) != expected:
                    raise ValueError(f"Export checksum mismatch: {path}")
            elif not (number == 3 and name == "audio/lecture_16k.wav"):
                raise FileNotFoundError(f"Unexpected missing export asset: {path}")
        if dest.exists() and not resume:
            raise FileExistsError(f"Choose a fresh experiment name: {dest}")
        if dest.exists():
            existing = common.load_stage(dest, number, video)
            if existing["artifact_id"] != manifest["artifact_id"]:
                raise ValueError(f"Cannot resume different stage input: {dest}")
        else:
            shutil.copytree(source, dest, copy_function=copy_input_file)
        if number == 3 and 'audio/lecture_16k.wav' in manifest['files'] and not (dest / "audio/lecture_16k.wav").exists():
            mp4 = mp4_root / (video + ".mp4")
            if common.file_sha256(mp4) != manifest["source"]["sha256"]:
                raise ValueError(f"MP4 source mismatch: {mp4}")
            with patch.dict(os.environ, {"PATH": str(ffmpeg.parent) + os.pathsep + os.environ["PATH"]}):
                audio.normalize_audio(mp4, manifest["source"], dest / "audio/lecture_16k.wav")
            old = manifest["files"]["audio/lecture_16k.wav"]
            new = common.file_sha256(dest / "audio/lecture_16k.wav")
            manifest["files"]["audio/lecture_16k.wav"] = new
            manifest["artifact_id"] = common.fingerprint({k: v for k, v in manifest.items()
                                                         if k not in {"artifact_id", "created_utc"}})
            write_json(dest / "manifest.json", manifest)
            provenance.append({"asset": "audio/lecture_16k.wav", "original_sha256": old,
                               "restored_sha256": new, "byte_equal": old == new,
                               "note": "Real MP4 audio normalized by existing normalize_audio; copied manifest re-signed."})
        common.load_stage(dest, number, video)
    inputs = [target / "_inputs" / s for s in ("01_slides", "02_visual", "03_audio")]
    parents = [common.load_stage(p, i + 1, video) for i, p in enumerate(inputs)]
    common.require_same_source(*parents)
    common.require_parent(parents[1], parents[0])
    if '1' in parents[2]['parents']:
        common.require_parent(parents[2], parents[0])
    provenance.append({'stage_artifact_ids': {str(p['stage']): p['artifact_id'] for p in parents},
                       'source_sha256': parents[0]['source']['sha256'],
                       'asr_export': str(Path(asr_export or export).resolve())})
    return inputs, provenance


def rebuild_video(export_root, output, video, config, encoder, *, asr_export_root=None,
                  mp4_root=Path("D:/thesis-data/mp4-extra"),
                  ffmpeg=Path("D:/FilmTools/Shotcut/ffmpeg.exe"),
                  skip_reference_comparison=True, audio_reference=None, chunk_references=()):
    """Reusable one-video CPU rebuild, with the same manifest checks as the CLI."""
    started = time.monotonic()
    target = output / video
    alignment, indexing = load_notebook("alignment"), load_notebook("indexing")
    alignment_keys = {"ad_suspect_overlap", "slide_speech", "conservative_ocr", "heading_titles",
                      "asr_correction", "failure_guard", "speech_ad_filter"}
    stage04 = target / "04_bundle"
    if (stage04 / "manifest.json").exists():
        manifest = alignment.load_stage(stage04, 4, video)
        inputs = [target / "_inputs" / s for s in ("01_slides", "02_visual", "03_audio")]
        parents = [alignment.load_stage(p, i + 1, video) for i, p in enumerate(inputs)]
        for parent in parents:
            alignment.require_parent(manifest, parent)
        alignment.require_same_source(manifest, *parents)
        for key in alignment_keys:
            if key in manifest["parameters"] and manifest["parameters"][key] != config[key]:
                raise ValueError(f"Cannot resume stage 04 with changed {key}")
        restored = [{"resumed_verified_stage04": manifest["artifact_id"]}]
    else:
        inputs, restored = prepare_inputs(export_root, target, video, mp4_root, ffmpeg, asr_export_root, resume=True)
        speech_embed = (lambda texts: encode_batches(encoder, texts, "query: ", config["batch_size"])) if config["speech_relevance"] else None
        stage04, _, _ = alignment.run_alignment(*inputs, target, video, speech_embed=speech_embed,
                                               **{k: config[k] for k in alignment_keys})
    options = {k: v for k, v in config.items() if k not in alignment_keys | {"speech_relevance"}}
    adapter = types.ModuleType("sentence_transformers")
    model = ModelAdapter(encoder, chunk_references)
    adapter.SentenceTransformer = lambda *a, **kw: model
    with patch.dict(sys.modules, {"sentence_transformers": adapter}):
        stage05, _, _ = indexing.run_indexing(stage04, target, video, device="cpu", **options)
    row = {"parity": None if skip_reference_comparison else
           compare_indexes(stage05, export_root / video / "05_index"),
           "stage04_comparison": None if skip_reference_comparison else
           compare_bundles(stage04, export_root / video / "04_bundle"),
           "restored_assets": restored, "elapsed_seconds": round(time.monotonic() - started, 2)}
    if audio_reference:
        from dedupe_audio import dedupe_video_assets
        row["audio_sharing"] = {"linked_files": len(dedupe_video_assets(target, audio_reference / video)),
                                "policy": "identical_SHA256_hardlinks"}
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected torch import")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--asr-export-root", type=Path, help="Isolated stage-03 variant; stages 01/02 stay identical")
    parser.add_argument("--output-root", type=Path, help="Experiment storage root, e.g. D:/thesis-data/run-20261102/eval")
    parser.add_argument("--skip-reference-comparison", action="store_true", help="Rebuilding a new ASR variant rather than testing parity")
    parser.add_argument("--videos", nargs="+", default=VIDEOS, choices=VIDEOS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides of production_config")
    parser.add_argument("--audio-reference", type=Path, help="Completed experiment for identical-SHA256 WAV hard links")
    parser.add_argument("--mp4-root", type=Path, default=Path("D:/thesis-data/mp4-extra"))
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "D:/FilmTools/Shotcut/ffmpeg.exe"))
    args = parser.parse_args()
    output, config = experiment_dir(args.name, args.output_root), production_config()
    if args.config:
        overrides = read_json(args.config)
        if set(overrides) - set(config):
            parser.error(f"Unknown config keys: {sorted(set(overrides) - set(config))}")
        config.update(overrides)
    backend = load_backend()
    if (config["embedding_model"], config["revision"]) != (backend.E5_REPO, backend.E5_REVISION):
        parser.error("This harness only supports the pinned production E5 model/revision.")
    encoder = backend.OnnxEncoder()
    alignment = load_notebook("alignment")
    report = {"config": config, "export_root": str(args.export_root.resolve()),
              "execution": "CPU ONNX; offline HF cache; isolated storage; no torch",
              "code_sha256": {str(p.relative_to(ROOT)): alignment.file_sha256(p)
                              for p in [ROOT / "builders/build_fast_pipeline.py",
                                        *[module_path(name) for name in sorted(MODULE_PATHS)]]}, "videos": {}}
    write_json(output / "rebuild_report.json", report)
    reviewed = read_json(ROOT / "eval/phase3_gpu_review.json")
    write_json(output / "gpu_retry_review.json", [item for item in reviewed if item["video_id"] in args.videos])
    retry_segments = []
    for video in args.videos:
        row = rebuild_video(args.export_root, output, video, config, encoder,
                            asr_export_root=args.asr_export_root, mp4_root=args.mp4_root,
                            ffmpeg=args.ffmpeg,
                            skip_reference_comparison=args.skip_reference_comparison,
                            audio_reference=args.audio_reference)
        retry_segments.extend(read_json(output / video / "04_bundle/gpu_retry_segments.json"))
        write_json(output / "gpu_retry_segments.json", retry_segments)
        report["videos"][video] = row
        write_json(output / "rebuild_report.json", report)
        print(video, "REBUILT", flush=True)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected torch import")


if __name__ == "__main__":
    main()
