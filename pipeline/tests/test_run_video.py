"""Single-video contract tests: real media/stages, fake model loaders, no network."""

import ast
import gc
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import types
from unittest.mock import Mock
import weakref

import numpy as np
from PIL import Image, ImageDraw
import pytest

from pipeline import run_video

REPO = Path(__file__).resolve().parents[2]


class Tokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        tokens = [ord(c) for c in text]
        return [0, *tokens, 0] if add_special_tokens else tokens

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(c) for c in tokens if c)


def install_fakes(runtime, set_module):
    """Install models whose weak owners prove that successive stages never overlap."""
    live, calls = weakref.WeakSet(), []

    class Owner:
        def __init__(self, name):
            gc.collect()
            assert not live, f"previous model still resident before {name}"
            live.add(self)
            calls.append((name,))

    class Encoder(Owner):
        max_seq_length = 512
        tokenizer = Tokenizer()

        def __init__(self, model_id, *, revision, device):
            super().__init__("e5")
            calls.append(("embedding", model_id, revision, device))

        def get_sentence_embedding_dimension(self):
            return 384

        def _first_module(self):
            return types.SimpleNamespace(auto_model=types.SimpleNamespace(
                config=types.SimpleNamespace(_commit_hash=runtime.EMBEDDING_REVISION)))

        def encode(self, texts, **kwargs):
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            vectors[:, 0] = 1.0
            return vectors

    class ASR(Owner):
        def __init__(self):
            super().__init__("asr")

        def transcribe(self, audio, **kwargs):
            calls.append(("transcribe", kwargs))
            words = [types.SimpleNamespace(start=0.2 + i * .4, end=.5 + i * .4,
                                           word=word, probability=.95)
                     for i, word in enumerate(("A", "lecture", "about", "probability."))]
            return [types.SimpleNamespace(words=words, text="A lecture about probability.")], \
                types.SimpleNamespace(language=kwargs["language"] or "en", duration=len(audio)/16000,
                                      duration_after_vad=len(audio)/16000)

    def gate(device, *, threshold, views):
        owner = Owner("gate")
        calls.append(("gate_options", device, threshold, views))

        def check(images):
            assert owner in live
            # A low full-frame score is rescued by the real frame+crops logic.
            return [.05] if len(images) == 1 else [.9] * len(images)

        runtime._SHARED_MODELS[("fixture_gate",)] = owner
        return check, {"model": "google/siglip-base-patch16-224", "threshold": threshold, "views": views}

    def qwen(devices, **options):
        owner = Owner("qwen")
        calls.append(("qwen_options", devices, options))

        def read(images, language="vi", *, on_answer=None):
            assert owner in live
            calls.append(("ocr_language", language))
            answers = []
            for i, _ in enumerate(images):
                def generate(prompt, limit, penalty):
                    calls.append(("generation", limit, penalty))
                    return {"text": "loop " * 9 if limit == 1536 else "Probability theorem\nP(A) = 0.5",
                            "token_ids": [10, 99 if limit == 1536 else 2], "eos_token_id": [2]}
                answer = runtime.qwen_ocr_with_retry(generate, language,
                    validate=options["validate"], retry_invalid=options["retry_invalid"])
                answers.append(answer)
                if on_answer:
                    on_answer(i, answer)
                print(f"Qwen đọc slide {i+1}/{len(images)}", flush=True)
            return answers

        read.models = [owner]
        return read

    def asr(model_id, revision, device, device_index):
        calls.append(("asr_options", model_id, revision, device, device_index))
        return ASR(), revision, device, "float16" if device == "cuda" else "int8"

    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 set_device=Mock(), empty_cache=Mock())
    set_module("torch", types.SimpleNamespace(cuda=cuda))
    set_module("sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    runtime.load_ocr_gate, runtime.load_qwen_ocr, runtime.load_asr_model = gate, qwen, asr
    return calls, live, cuda


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = run_video.load_runtime()
    calls, live, cuda = install_fakes(runtime, lambda name, module: monkeypatch.setitem(sys.modules, name, module))
    monkeypatch.setattr(run_video, "load_runtime", lambda: runtime)
    return runtime, calls, live, cuda


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        pytest.skip("FFmpeg/FFprobe required for synthetic MP4 tests")
    root = tmp_path_factory.mktemp("cli-media")
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    for line in range(4):
        draw.rectangle((20, 25 + line*30, 260-line*25, 33+line*30), fill="black")
    image.save(root / "slide.png")
    path = root / "tiny.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
                    "-i", str(root / "slide.png"), "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=16000:duration=3", "-t", "3", "-r", "10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
    return path


def invoke(video, out, *extra):
    return run_video.main(["--video", str(video), "--video-id", "fixture", "--out", str(out),
                           "--language", "vi", *extra])


def events(capsys):
    captured = capsys.readouterr()
    return [json.loads(line) for line in captured.out.splitlines()], captured.err


def read_importer_layout(folder):
    """Replicate import_video's reads without importing backend storage or credentials."""
    intervals = json.loads((folder / "01_slides/intervals.json").read_text(encoding="utf8"))
    audit = json.loads((folder / "01_slides/reveal_audit.json").read_text(encoding="utf8"))
    split_reasons = {item["current_state"]: item["decision"] for item in audit}
    assert (folder / "01_slides/keyframes/states").is_dir()
    for interval in intervals:
        states = interval["reveal_states"]
        assert states
        split_reasons.get(states[0]["source_state"])
        for state in states:
            assert (folder / "01_slides" / state["image_path"]).is_file()
            float(state["start_sec"])
        assert interval["start_sec"] < interval["end_sec"]
    segments = json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf8"))
    transcript = json.loads((folder / "04_bundle/transcript_app.json").read_text(encoding="utf8"))
    assert transcript["video_id"] == folder.name
    assert set(transcript) == {"video_id", "language", "segments"}
    for segment in transcript["segments"]:
        assert set(segment) == {"start", "end", "text"}
    chunks = [json.loads(line) for line in (folder / "05_index/chunks.jsonl").read_text(
        encoding="utf8").splitlines() if line]
    ids = json.loads((folder / "05_index/chunk_ids.json").read_text(encoding="utf8"))
    assert ids == [chunk["chunk_id"] for chunk in chunks]
    assert np.load(folder / "05_index/embeddings.npy", allow_pickle=False).shape == (len(chunks), 384)
    for chunk in chunks:
        assert (folder / "05_index/lecture" / chunk["image_path"]).is_file()
    with sqlite3.connect(folder / "05_index/lexical.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == len(chunks)
    return intervals, segments, transcript, chunks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cli_progress_models_and_importer_layout(video, tmp_path, fake_runtime, capsys, device):
    runtime, calls, live, cuda = fake_runtime
    assert invoke(video, tmp_path, "--device", device) == 0
    rows, logs = events(capsys)
    assert rows[-1] == {"event": "done", "export_dir": str(tmp_path / "export/fixture")}
    starts = [r["stage"] for r in rows if r.get("status") == "running" and r["progress"] == 0]
    assert starts == list(run_video.STAGES)
    assert [r["stage"] for r in rows if r.get("status") == "completed"] == list(run_video.STAGES)
    for name in run_video.STAGES:
        values = [r["progress"] for r in rows if r.get("stage") == name]
        assert values == sorted(values) and values[0] == 0 and values[-1] == 1
    assert any(0 < row.get("progress", 0) < 1 for row in rows)
    assert "Hoàn tất" in logs and "Qwen đọc slide" in logs
    assert [call[0] for call in calls if len(call) == 1] == ["gate", "qwen", "asr", "e5", "e5"]
    assert not live and not runtime._SHARED_MODELS
    assert cuda.empty_cache.call_count == 6
    target_device = "cuda:0" if device == "cuda" else "cpu"
    assert ("gate_options", target_device, .15, "frame+crops") in calls
    qwen = next(call for call in calls if call[0] == "qwen_options")
    assert qwen[1] == (target_device,)
    assert qwen[2] == dict(model_id=runtime.QWEN_OCR_MODEL, revision=runtime.QWEN_OCR_REVISION,
                          min_pixels=256*32*32, max_pixels=1280*32*32,
                          validate=True, retry_invalid=True)
    assert [c for c in calls if c[0] == "generation"] == [("generation", 1536, 1.05), ("generation", 2048, 1.10)]
    asr = next(call for call in calls if call[0] == "asr_options")
    assert asr[1:] == (runtime.ASR_MODEL, runtime.ASR_REVISION, device, 0)
    options = next(call[1] for call in calls if call[0] == "transcribe")
    assert options["vad_filter"] and options["word_timestamps"]
    assert options["vad_parameters"] == {"min_silence_duration_ms": 500, "speech_pad_ms": 250}
    assert options["beam_size"] == 5 and not options["condition_on_previous_text"]
    folder = tmp_path / "export/fixture"
    assert runtime.validate_video_output(folder, "fixture")["passed"]
    _, _, transcript, chunks = read_importer_layout(folder)
    assert transcript["language"] == "vi" and transcript["segments"] and chunks
    gate = runtime.read_json(folder / "gate.json")
    assert next(iter(gate["decisions"].values()))[0] is False
    assert runtime.read_json(folder / "04_bundle/manifest.json")["parameters"]["asr_correction"] == "glossary_code"
    completed = runtime.read_json(folder / "completed.json")
    assert set(completed["stage_timings"]) == set(run_video.STAGES)
    assert runtime.read_json(tmp_path / "progress.json")["videos"][0]["status"] == "completed"


@pytest.mark.parametrize("skip", [(True, False), (False, True), (True, True)])
def test_skip_flags_keep_valid_exports_without_loading_skipped_models(video, tmp_path, fake_runtime, capsys, skip):
    runtime, calls, live, _ = fake_runtime
    flags = (["--skip-ocr"] if skip[0] else []) + (["--skip-asr"] if skip[1] else [])
    assert invoke(video, tmp_path, "--device", "cpu", *flags) == 0
    rows, _ = events(capsys)
    assert rows[-1]["event"] == "done" and not live
    loaded = [call[0] for call in calls if len(call) == 1]
    assert ("qwen" in loaded) != skip[0] and ("gate" in loaded) != skip[0]
    assert ("asr" in loaded) != skip[1]
    folder = tmp_path / "export/fixture"
    assert runtime.validate_video_output(folder, "fixture", allow_empty_index=True)["passed"]
    _, _, transcript, chunks = read_importer_layout(folder)
    if skip[0]:
        assert all(v["ocr_status"] == "skipped_requested" for v in runtime.read_json(folder / "02_visual/visual.json"))
    if skip[1]:
        assert not transcript["segments"]
        assert runtime.read_json(folder / "03_audio/transcript.json")["status"] == "skipped_requested"
    if all(skip):
        assert not chunks
        with pytest.raises(ValueError, match="chunk/vector"):
            runtime.validate_video_output(folder, "fixture")


def test_auto_detects_asr_language_with_existing_english_ocr_fallback(video, tmp_path, fake_runtime, capsys):
    runtime, calls, _, _ = fake_runtime
    assert invoke(video, tmp_path, "--device", "cpu", "--language", "auto") == 0
    assert events(capsys)[0][-1]["event"] == "done"
    assert ("ocr_language", None) in calls
    assert next(c[1] for c in calls if c[0] == "transcribe")["language"] is None
    assert runtime.read_json(tmp_path / "export/fixture/04_bundle/transcript_app.json")["language"] == "en"


def test_gate_replay_matches_in_stage_gate(video, tmp_path, fake_runtime, capsys):
    runtime, _, _, _ = fake_runtime
    assert invoke(video, tmp_path / "cli", "--device", "cpu") == 0
    events(capsys)
    folder = tmp_path / "cli/export/fixture"
    shutil.copytree(folder / "02_visual/_cache", tmp_path / "direct/02_visual/_cache")
    gate = runtime.load_ocr_gate("cpu", threshold=.15, views="frame+crops")
    runtime.run_visual_extraction.func(folder / "01_slides", tmp_path / "direct", "fixture",
        ocr_device="cpu", ocr_engine="qwen", hybrid_devices=("cpu",), ocr_gate=gate,
        reuse_near_duplicates=True, qwen_content_rois=runtime.QWEN_CONTENT_ROIS)
    direct = runtime.read_json(tmp_path / "direct/02_visual/visual.json")
    replayed = runtime.read_json(folder / "02_visual/visual.json")
    for a, b in zip(direct, replayed):
        assert {k: a.get(k) for k in ("ocr_text", "content_roi", "ocr_status", "ocr_gate_score", "ocr_gate_crop_scores")} == \
            {k: b.get(k) for k in ("ocr_text", "content_roi", "ocr_status", "ocr_gate_score", "ocr_gate_crop_scores")}
    run_video.release_models(runtime)


@pytest.mark.parametrize("failure", ["segment", "gate", "ocr", "asr", "align", "index", "validate"])
def test_stage_error_event_nonzero_and_cleanup(video, tmp_path, fake_runtime, capsys, monkeypatch, failure):
    runtime, _, live, _ = fake_runtime
    function, stage = {"segment": ("run_slide_detection", "s01_segment"),
                       "gate": ("load_ocr_gate", "s02_gate"), "ocr": ("load_qwen_ocr", "s03_ocr"),
                       "asr": ("load_asr_model", "s04_asr"),
                       "align": ("run_alignment", "s05_align"), "index": ("run_indexing", "s06_index"),
                       "validate": ("validate_video_output", "s06_index")}[failure]

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic stage failure")

    monkeypatch.setattr(runtime, function, fail)
    assert invoke(video, tmp_path, "--device", "cpu") == 1
    rows, logs = events(capsys)
    assert rows[-1] == {"event": "error", "stage": stage, "message": "RuntimeError: synthetic stage failure"}
    assert not any(row["event"] == "done" for row in rows)
    assert "synthetic stage failure" in logs
    assert not live and not runtime._SHARED_MODELS
    assert not (tmp_path / "export/fixture/completed.json").exists()
    assert runtime.read_json(tmp_path / "progress.json")["videos"][0]["status"] == "failed"


@pytest.mark.parametrize("arguments", [[], ["--language", "fr"], ["--video-id", "../escape"]])
def test_argument_errors_are_json(video, tmp_path, capsys, arguments):
    if not arguments:
        code = run_video.main([])
    else:
        code = invoke(video, tmp_path, "--device", "cpu", *arguments)
    assert code == 1
    rows, _ = events(capsys)
    assert len(rows) == 1 and rows[0]["event"] == "error" and rows[0]["stage"] == "preflight"


def test_module_process_missing_video_emits_error_and_exits_nonzero(tmp_path):
    result = subprocess.run([sys.executable, "-m", "pipeline.run_video", "--video", str(tmp_path / "absent.mp4"),
                             "--video-id", "fixture", "--out", str(tmp_path), "--language", "vi", "--device", "cpu"],
                            cwd=REPO, capture_output=True, text=True, encoding="utf8")
    assert result.returncode == 1
    assert json.loads(result.stdout)["event"] == "error"
    assert "FileNotFoundError" in result.stderr


def test_module_process_runs_six_stages_with_fake_models(video, tmp_path):
    code = """import sys
from pipeline import run_video
from pipeline.tests.test_run_video import install_fakes
runtime = run_video.load_runtime()
install_fakes(runtime, lambda name, module: sys.modules.__setitem__(name, module))
run_video.load_runtime = lambda: runtime
raise SystemExit(run_video.main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-c", code, "--video", str(video), "--video-id", "fixture",
                             "--out", str(tmp_path), "--language", "en", "--device", "cpu"], cwd=REPO,
                            capture_output=True, text=True, encoding="utf8")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[-1]["event"] == "done"
    assert len([r for r in rows if r.get("status") == "completed"]) == 6
    read_importer_layout(tmp_path / "export/fixture")


def test_pinned_requirements_satisfy_builder_specifiers():
    from packaging.requirements import Requirement
    import shlex
    builder = ast.parse((run_video.ROOT / "builders/build_fast_pipeline.py").read_text(encoding="utf8"))
    deps = next(ast.literal_eval(n.value) for n in builder.body if isinstance(n, ast.Assign)
                and n.targets[0].id == "DEPS")
    pinned = {r.name: r for line in (run_video.ROOT / "requirements-gpu.txt").read_text().splitlines()
              if line and not line.startswith(("#", "--")) for r in [Requirement(line)]}
    for dep in shlex.split(deps):
        requirement = Requirement(dep)
        pin = next(iter(pinned[requirement.name].specifier))
        assert pin.operator == "==" and pin.version in requirement.specifier


def test_stdout_count_translation_is_monotonic_and_logs_stay_on_stderr():
    logs, rows = io.StringIO(), []
    stream = run_video.StageLogs(logs, lambda **event: rows.append(event), "s03_ocr")
    stream.write("Qwen đọc slide 1/2\nOCR 1/10\nOCR 10/10\nordinary log\n")
    assert [row["progress"] for row in rows] == [.5, .99]
    assert "ordinary log" in logs.getvalue()


def test_stage_events_carry_whole_run_progress_and_loading_phase():
    assert sum(run_video.STAGE_WEIGHTS) == pytest.approx(1) and len(run_video.STAGE_WEIGHTS) == len(run_video.STAGES)
    start = run_video.with_overall({"event": "stage", "stage": "s03_ocr", "status": "running", "progress": 0.0})
    assert (start["overall"], start["phase"]) == (.15, "loading")
    half = run_video.with_overall({"event": "stage", "stage": "s03_ocr", "status": "running", "progress": .5})
    assert (half["overall"], half["phase"]) == (.35, "working")
    done = run_video.with_overall({"event": "stage", "stage": "s06_index", "status": "completed", "progress": 1.0})
    assert (done["overall"], done["phase"]) == (1.0, "working")
    skipped = run_video.with_overall({"event": "stage", "stage": "s02_gate", "status": "skipped", "progress": 1.0})
    assert skipped["phase"] == "working"


def test_keyframe_and_ocr_readahead_counts_report_progress():
    rows = []
    stream = run_video.StageLogs(io.StringIO(), lambda **event: rows.append(event), "s01_segment")
    stream.write("Keyframe 1/4: 0.00s\nOCR đọc trước 3/4\n")
    assert [row["progress"] for row in rows] == [.25, .75]
