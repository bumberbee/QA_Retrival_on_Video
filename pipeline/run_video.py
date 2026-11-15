"""Run the notebook's production recipe sequentially on one CUDA device.

The stages are notebook fragments, deliberately executed in one namespace just
as build_fast_pipeline embeds them. Importing this CLI never loads a model.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from functools import partial
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import traceback
import types

from pipeline.shared.source_paths import module_path

ROOT = Path(__file__).resolve().parent
STAGES = ("s01_segment", "s02_gate", "s03_ocr", "s04_asr", "s05_align", "s06_index")


def load_runtime():
    """Reuse builder constants and queue functions without running the Kaggle queue."""
    builder_path = ROOT / "builders/build_fast_pipeline.py"
    builder = ast.parse(builder_path.read_text(encoding="utf-8"))
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in builder.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"CONFIG", "MODULES"}}
    runtime = types.ModuleType("pipeline_single_video_runtime")
    sources = []
    for name in constants["MODULES"]:
        path = module_path(name)
        source = path.read_text(encoding="utf-8")
        sources.append(source)
        exec(compile(source, str(path), "exec"), runtime.__dict__)
    exec(compile(constants["CONFIG"], str(builder_path), "exec"), runtime.__dict__)
    queue_path = module_path("production_queue.py")
    queue_source = queue_path.read_text(encoding="utf-8")
    functions = [node for node in ast.parse(queue_source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {"detect", "ocr", "asr", "speech_embed_model"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(queue_path), "exec"),
         runtime.__dict__)
    runtime.CODE_RECIPE_SHA256 = hashlib.sha256(
        ("".join(sources) + queue_source).encode("utf-8")).hexdigest()
    return runtime


def release_models(runtime):
    """Drop all notebook cache/closure owners, including CT2, before empty_cache."""
    for name in ("ocr_gate", "ad_scorer", "content_scorer", "speech_embed"):
        setattr(runtime, name, None)
    with runtime.MODEL_LOAD_LOCK:
        models = list(runtime._SHARED_MODELS.values())
        runtime._SHARED_MODELS.clear()
        del models
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class StageLogs(io.TextIOBase):
    """Forward notebook prints to stderr and translate their item counts to progress."""

    def __init__(self, stderr, emit, stage):
        self.stderr, self.emit, self.stage = stderr, emit, stage
        self.pending, self.progress = "", 0.0
        self.lock = threading.RLock()

    def write(self, value):
        with self.lock:
            self.stderr.write(value)
            self.pending += value
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                match = re.match(r"^(?:OCR|ASR|Qwen đọc slide) (\d+)/(\d+)", line)
                if match and int(match[2]):
                    progress = min(0.99, int(match[1]) / int(match[2]))
                    if progress > self.progress:
                        self.progress = progress
                        self.emit(event="stage", stage=self.stage, status="running", progress=progress)
        return len(value)

    def flush(self):
        self.stderr.flush()


def run_gate(runtime, stage01, video_id, emit):
    """Persist the existing frame+crops decisions, retaining no SigLIP owner."""
    from PIL import Image

    intervals = runtime.read_json(stage01 / "intervals.json")
    gate = runtime.load_ocr_gate(runtime.AD_DEVICE, threshold=runtime.OCR_GATE_THRESHOLD,
                                 views=runtime.OCR_GATE_VIEWS) if runtime.OCR_GATE else None
    decisions = {}
    for index, interval in enumerate(intervals):
        if interval.get("content_kind") != "ad":
            with Image.open(runtime.relative_asset(stage01, interval["image_path"])) as opened:
                image = opened.convert("RGB")
            # Use precisely the same ROI that run_visual_extraction passes to gate_keyframe.
            if runtime.OCR_ENGINE == "qwen" and runtime.QWEN_CONTENT_CROP:
                rois = runtime.QWEN_CONTENT_ROIS
                roi, _ = runtime.qwen_content_roi(image, interval, runtime.FRAME_HSV.get(video_id),
                                                  rois.get(interval["segment_id"], rois.get(video_id)))
            else:
                roi = runtime.detect_content_roi(image, runtime.FRAME_HSV.get(video_id))
            decisions[interval["segment_id"]] = runtime.gate_keyframe(gate, image, roi)
        emit(event="stage", stage="s02_gate", status="running",
             progress=min(0.99, (index + 1) / len(intervals)))
    info = gate[1] if gate else None
    runtime.write_json(stage01.parent / "gate.json", {
        "parent_artifact_id": runtime.load_stage(stage01, 1, video_id)["artifact_id"],
        "parameters": info, "decisions": decisions})
    return decisions, info


def preflight(args):
    if args.video_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.video_id):
        raise ValueError("--video-id must be a single safe directory name (letters, digits, _, -, .)")
    args.video = args.video.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.video}")
    if args.video.suffix.lower() != ".mp4":
        raise ValueError("--video must be an MP4 file")
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("ffmpeg and ffprobe must be available in PATH")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("--device cuda requires an available CUDA GPU; use --device cpu")
        torch.cuda.set_device(0)
    args.out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out).free < 2 * 2**30:
        raise RuntimeError("At least 2 GiB of working disk space is required")


def _execute_stage(runtime, fn):
    try:
        return fn()
    except BaseException as error:
        # Unwound stage frames can otherwise retain model/tensor locals until main exits.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        release_models(runtime)


def run(args, emit, stderr):
    """Return the video export path only after the production validator succeeds."""
    runtime = load_runtime()
    device = "cuda:0" if args.device == "cuda" else "cpu"
    runtime.OUTPUT_ROOT = args.out / "export"
    runtime.VIDEO_IDS = [args.video_id]
    # Whisper's None requests detection. Qwen's existing non-Vietnamese prompt is English.
    runtime.VIDEO_LANGUAGES = {args.video_id: None if args.language == "auto" else args.language}
    runtime.AD_DEVICE = runtime.EMBEDDING_DEVICE = device
    runtime.OCR_DEVICE = runtime.ASR_DEVICE = args.device
    runtime.ASR_DEVICE_INDEX = 0
    runtime.HYBRID_DEVICES = (device,)
    runtime.find_video_file = lambda video_id: args.video
    runtime.ad_scorer = runtime.content_scorer = runtime.ocr_gate = runtime.speech_embed = None
    recipe = runtime.fingerprint({
        "code": runtime.CODE_RECIPE_SHA256, "video_id": args.video_id,
        "language": args.language, "device": args.device,
        "skip_ocr": args.skip_ocr, "skip_asr": args.skip_asr})
    progress = runtime.ProductionProgress(args.out, [args.video_id], recipe)
    export_dir = runtime.OUTPUT_ROOT / args.video_id
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "completed.json").unlink(missing_ok=True)

    def stage(name, fn, skipped=False):
        emit(event="stage", stage=name, status="running", progress=0.0)
        progress.update(args.video_id, stage=name)
        try:
            with redirect_stdout(StageLogs(stderr, emit, name)):
                result = progress.timed(name, args.video_id, _execute_stage, runtime, fn)
        except BaseException:
            progress.update(args.video_id, status="failed")
            raise
        emit(event="stage", stage=name, status="skipped" if skipped else "completed", progress=1.0)
        return result

    # The production queue's ASR branch is independent of slide edges; retain that behavior.
    _, stage01, manifest01, _ = stage(STAGES[0], lambda: runtime.detect(args.video_id))
    decisions, gate_info = stage(STAGES[1],
        lambda: (None, None) if args.skip_ocr else run_gate(runtime, stage01, args.video_id, emit),
        skipped=args.skip_ocr)
    runtime.ocr_gate = (None, gate_info) if gate_info else None
    runtime.run_visual_extraction = partial(runtime.run_visual_extraction,
                                            gate_decisions=decisions, skip_ocr=args.skip_ocr)
    stage02, _ = stage(STAGES[2], lambda: runtime.ocr(args.video_id, stage01), skipped=args.skip_ocr)
    runtime.run_transcription = partial(runtime.run_transcription, skip_asr=args.skip_asr)
    stage03, _ = stage(STAGES[3], lambda: runtime.asr(args.video_id), skipped=args.skip_asr)
    visual = runtime.read_json(stage02 / "visual.json")
    manifest02 = runtime.read_json(stage02 / "manifest.json")
    progress.update(args.video_id, duration_sec=manifest01["source"]["duration_sec"],
                    slides=manifest01["interval_count"],
                    visual_states=manifest01["visual_state_count"],
                    ocr_read=manifest02.get("ocr_read_count"), ocr_failed=manifest02.get("ocr_failed_count", 0),
                    retries=sum(max(0, len(v.get("ocr_generation", {}).get("attempts", [])) - 1) for v in visual))

    def align():
        runtime.speech_embed = runtime.speech_embed_model() if runtime.SPEECH_RELEVANCE else None
        return runtime.run_alignment(stage01, stage02, stage03, export_dir, args.video_id,
            ad_suspect_overlap=runtime.AD_SUSPECT_OVERLAP, speech_embed=runtime.speech_embed,
            slide_speech=runtime.SLIDE_SPEECH if runtime.AD_FILTER == "slide_speech" else None,
            conservative_ocr=runtime.CONSERVATIVE_OCR, heading_titles=runtime.HEADING_TITLES,
            asr_correction=runtime.ASR_CORRECTION, failure_guard=runtime.FAILURE_GUARD,
            speech_ad_filter=runtime.SPEECH_AD_FILTER)

    stage04, _, _ = stage(STAGES[4], align)
    allow_empty = args.skip_ocr or args.skip_asr

    def index():
        runtime.run_indexing(stage04, export_dir, args.video_id,
            embedding_model=runtime.EMBEDDING_MODEL, revision=runtime.EMBEDDING_REVISION,
            device=runtime.EMBEDDING_DEVICE, dedupe_previous=runtime.OCR_DEDUPE_PREVIOUS,
            include_generated_descriptions=runtime.INCLUDE_GENERATED_DESCRIPTIONS,
            generated_semantic_only=runtime.GENERATED_SEMANTIC_ONLY,
            visual_state_timing=runtime.VISUAL_STATE_TIMING, allow_empty=allow_empty)
        validated = runtime.validate_video_output(export_dir, args.video_id, allow_empty_index=allow_empty)
        return validated

    validated = stage(STAGES[5], index)
    progress.update(args.video_id, status="completed", error=None, validation=validated)
    # Publish completion after validation, memory cleanup, and the final timing span.
    runtime.write_json(export_dir / "completed.json", {
        **validated, "recipe": recipe, "retries": progress.rows[args.video_id]["retries"],
        "stage_timings": progress.rows[args.video_id]["stage_timings"]})
    return export_dir


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    stdout, stderr = sys.stdout, sys.stderr
    current_stage = "preflight"

    def emit(**event):
        nonlocal current_stage
        if event.get("event") == "stage":
            current_stage = event["stage"]
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), file=stdout, flush=True)

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=("vi", "en", "auto"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    try:
        args = parser.parse_args(argv)
        preflight(args)
        with redirect_stdout(stderr):
            export_dir = run(args, emit, stderr)
        emit(event="done", export_dir=str(export_dir))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc(file=stderr)
        emit(event="error", stage=current_stage, message=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
