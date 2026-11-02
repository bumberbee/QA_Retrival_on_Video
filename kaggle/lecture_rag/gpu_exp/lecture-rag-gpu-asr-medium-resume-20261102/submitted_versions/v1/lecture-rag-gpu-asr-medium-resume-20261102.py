"""Shared, credential-free helpers embedded in private Kaggle scripts."""
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

START = time.monotonic()
OUT = Path('/kaggle/working/results')
OUT.mkdir(parents=True, exist_ok=True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf8')


def install(*packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True, timeout=600)


def environment():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True)
    return {'python': sys.version, 'gpu': gpu.stdout.strip(),
            'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}}


def vram():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return [int(x) for x in r.stdout.splitlines() if x.strip().isdigit()]


def export_root():
    # Prefer mounted individual exports, otherwise extract only stages needed by the experiments.
    paths = list(Path('/kaggle/input').rglob('transcript.json'))
    for p in paths:
        if p.parent.name == '03_audio' and p.parent.parent.name == 'hVU5Mw19V_c':
            return p.parents[2]
    for z in Path('/kaggle/input').rglob('*.zip'):
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            if not any('hVU5Mw19V_c/03_audio/transcript.json' in n for n in names):
                continue
            dest = Path('/kaggle/working/export_inputs')
            for n in names:
                if any('/' + s + '/' in '/' + n for s in ('01_slides', '02_visual', '03_audio')):
                    target = (dest / n).resolve()
                    if not target.is_relative_to(dest.resolve()):
                        raise ValueError('Unsafe archive path')
                    f.extract(n, dest)
            return next(dest.rglob('hVU5Mw19V_c/03_audio/transcript.json')).parents[2]
    raise FileNotFoundError('Attached fast-pipeline output has no stage-03 export')


def finish(extra=None):
    save(OUT / 'session.json', {'elapsed_seconds': time.monotonic() - START,
                              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              **(extra or {})})
    # Keep outputs limited to measurements; intermediate model copies are reproducible from pins.
    working=Path('/kaggle/working').resolve()
    for name in ('audio','export_inputs','phowhisper_ct2','ollama.tgz','ollama','ollama_models'):
        path=(working/name).resolve()
        if not path.is_relative_to(working) or path==working:raise ValueError('Unsafe scratch cleanup')
        if path.is_dir():shutil.rmtree(path)
        elif path.is_file():path.unlink()


def failure(label, error):
    detail = {'error': str(error), 'traceback': traceback.format_exc()}
    save(OUT / ('failure_' + label + '.json'), detail)
    print(label, 'FAILED', str(error), flush=True)

"""Shared artifact contract. Embedded into every notebook by build_notebooks.py."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import threading
import unicodedata
from datetime import datetime, timezone

# transformers' from_pretrained builds models under a process-wide "meta device" context
# that is not thread-safe: two loads at once (OCR thread and main thread) fail with
# "Cannot copy out of meta tensor". Every model is loaded once per session, one at a time.
MODEL_LOAD_LOCK = threading.RLock()
_SHARED_MODELS = {}


def shared_model(key, loader):
    """Load a model once per session (reused by every video) and never two at once."""
    with MODEL_LOAD_LOCK:
        if key not in _SHARED_MODELS:
            _SHARED_MODELS[key] = loader()
        return _SHARED_MODELS[key]


SCHEMA_VERSION = "lecture-rag-v1"
PIPELINE_VERSION = "l25-v1"
STAGE_NAMES = {
    1: "01_slides", 2: "02_visual", 3: "03_audio",
    4: "04_bundle", 5: "05_index", 6: "06_retrieval",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def fingerprint(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_asset(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset nằm ngoài thư mục kết quả: {name}")
    return path


def run_command(arguments):
    process = subprocess.run(
        [str(value) for value in arguments], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if process.returncode:
        raise RuntimeError(
            f"{arguments[0]} thất bại ({process.returncode}):\n{process.stderr[-3000:]}"
        )
    return process.stdout


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def find_video(override=None, input_root="/kaggle/input"):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {path}")
        return path
    candidates = sorted(
        path for path in Path(input_root).rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}
        and path.stem.lower() in {"l25_v001", "tl25_v001"}
    )
    if len(candidates) != 1:
        raise ValueError(
            "Cần đúng một video L25_V001/TL25_V001. Add dataset "
            "thnhtrungnguynmif/l25a-lecture-videos-media hoặc đặt VIDEO_PATH. "
            f"Tìm thấy: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def probe_video(path):
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("Cần FFmpeg và FFprobe trong PATH.")
    data = json.loads(run_command([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(path),
    ]))
    videos = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    if not videos:
        raise ValueError("Tệp không có luồng video.")
    info = data.get("format", {})
    origin = float(info.get("start_time", 0) or 0)
    duration = float(info.get("duration") or videos[0].get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Không đọc được thời lượng video hợp lệ.")
    return {
        "filename": Path(path).name,
        "sha256": file_sha256(path),
        "duration_sec": duration,
        "timeline_origin_sec": origin,
        "video_start_sec": float(videos[0].get("start_time", origin)),
        "audio_start_sec": float(audios[0].get("start_time", origin)) if audios else None,
        "has_audio": bool(audios),
        "width": int(videos[0]["width"]),
        "height": int(videos[0]["height"]),
        "clock": "original_presentation_timestamp_minus_format_start_time",
    }


def begin_stage(output_root, stage):
    directory = Path(output_root) / STAGE_NAMES[stage]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "_RUNNING.json", {
        "stage": stage, "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    return directory


def finish_stage(directory, stage, video_id, source, parameters, parents=(), extra=None):
    directory = Path(directory)
    files = {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not any(part.startswith("_") for part in path.relative_to(directory).parts)
        and path.suffix != ".tmp"
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage,
        "video_id": video_id,
        "source": source,
        "parameters": parameters,
        "parents": {str(parent["stage"]): parent["artifact_id"] for parent in parents},
        "files": files,
        **(extra or {}),
    }
    payload["artifact_id"] = fingerprint(payload)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "manifest.json", payload)
    (directory / "_RUNNING.json").unlink(missing_ok=True)
    print(f"Hoàn tất {STAGE_NAMES[stage]}: {directory}")
    return payload


def load_stage(directory, stage=None, video_id="L25_V001", verify=True):
    directory = Path(directory)
    if (directory / "_RUNNING.json").exists():
        raise ValueError(f"Bước xử lý chưa hoàn tất; cần chạy lại: {directory}")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Schema không tương thích: {directory}")
    if stage is not None and manifest.get("stage") != stage:
        raise ValueError(f"Cần stage {stage}, nhận {manifest.get('stage')}: {directory}")
    if manifest.get("video_id") != video_id:
        raise ValueError(f"Sai video_id: {manifest.get('video_id')} != {video_id}")
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError(f"Sai pipeline_version: {directory}")
    signed = {key: value for key, value in manifest.items()
              if key not in {"artifact_id", "created_utc"}}
    if fingerprint(signed) != manifest["artifact_id"]:
        raise ValueError(f"Manifest đã thay đổi sau khi xuất: {directory}")
    if verify:
        for name, expected in manifest["files"].items():
            path = relative_asset(directory, name)
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Asset bị thiếu hoặc không khớp checksum: {path}")
    return manifest


def find_stage(stage, override=None, video_id="L25_V001",
               search_roots=("/kaggle/input", "/kaggle/working")):
    if override:
        directory = Path(override)
        return directory, load_stage(directory, stage, video_id)
    matches = []
    for root in map(Path, search_roots):
        if not root.exists():
            continue
        for path in root.rglob("manifest.json"):
            try:
                manifest = read_json(path)
            except (ValueError, OSError):
                continue
            if (manifest.get("schema_version") == SCHEMA_VERSION
                    and manifest.get("stage") == stage
                    and manifest.get("video_id") == video_id
                    and manifest.get("pipeline_version") == PIPELINE_VERSION):
                matches.append((path.parent, manifest))
    identities = {manifest.get("artifact_id") for _, manifest in matches}
    if not matches or len(identities) != 1:
        raise ValueError(
            f"Cần một phiên bản rõ ràng của {STAGE_NAMES[stage]}. "
            "Add Input → Notebook Output của bước trước, hoặc đặt STAGE_XX_DIR. "
            f"Các thư mục tìm được: {[str(path) for path, _ in matches]}"
        )
    directory = sorted(matches, key=lambda item: str(item[0]))[0][0]
    return directory, load_stage(directory, stage, video_id)


def require_same_source(*manifests):
    identities = {
        (item["video_id"], item["source"]["sha256"], item["pipeline_version"])
        for item in manifests
    }
    if len(identities) != 1:
        raise ValueError("Các nhánh không được tạo từ cùng video/pipeline.")


def require_parent(child, parent):
    require_same_source(child, parent)
    if child["parents"].get(str(parent["stage"])) != parent["artifact_id"]:
        raise ValueError(
            f"Stage {child['stage']} dùng phiên bản stage {parent['stage']} khác. "
            "Chạy lại nhánh với cùng kết quả phát hiện slide."
        )


def verify_intervals(intervals, duration, tolerance=0.03):
    if not intervals:
        raise ValueError("Danh sách interval rỗng.")
    ids = set()
    previous_end = 0.0
    for index, item in enumerate(intervals):
        start, end = float(item["start_sec"]), float(item["end_sec"])
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end <= duration + tolerance):
            raise ValueError(f"Interval không hợp lệ: {item}")
        if abs(start - previous_end) > tolerance:
            raise ValueError(f"Khoảng trống/chồng lấn ở interval {index}.")
        if item["segment_id"] in ids:
            raise ValueError("Trùng segment_id.")
        ids.add(item["segment_id"])
        previous_end = end
    if abs(previous_end - duration) > tolerance:
        raise ValueError("Interval cuối không bao phủ hết video.")


def owner_interval(intervals, time_sec):
    # Half-open intervals: a word exactly at a boundary belongs to the next one.
    for item in intervals:
        if item["start_sec"] <= time_sec < item["end_sec"]:
            return item
    if intervals and math.isclose(time_sec, intervals[-1]["end_sec"], abs_tol=1e-6):
        return intervals[-1]
    raise ValueError(f"Timestamp nằm ngoài lecture: {time_sec}")


def copy_stage_assets(source, destination, manifest):
    for name in manifest["files"]:
        target = relative_asset(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative_asset(source, name), target)


def normalize_search(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if not unicodedata.combining(char))


def timestamp_label(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def unique_records(records, key):
    result = {}
    for record in records:
        if record[key] in result:
            raise ValueError(f"Trùng {key}: {record[key]}")
        result[record[key]] = record
    return result


def read_pcm_audio(path, sample_rate=16000):
    import numpy as np
    import wave

    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate() != sample_rate or stream.getnchannels() != 1
                or stream.getsampwidth() != 2):
            raise ValueError("Audio phải là PCM signed 16-bit, mono, 16 kHz.")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


def save_pcm_clip(path, pcm, start_sample, end_sample, sample_rate=16000):
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm[start_sample:end_sample].astype("<i2").tobytes())

"""Audio clock normalization, pause-aware ASR windows, word timing, slide clips."""


def normalize_audio(video_path, source, output_path, sample_rate=16000):
    origin, duration = source["timeline_origin_sec"], source["duration_sec"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep original timestamps before subtracting the common video clock origin.
    # first_pts=0 fills an initial audio delay; apad retains silent video endings.
    filters = (
        f"asetpts=PTS-({origin})/TB,"
        f"aresample={sample_rate}:async=1:first_pts=0,"
        f"apad,atrim=start=0:end={duration}"
    )
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-copyts",
        "-i", str(video_path), "-map", "0:a:0", "-vn", "-af", filters,
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output_path),
    ])


def silence_midpoints(pcm, sample_rate=16000, threshold_db=-38.0,
                       minimum_silence_sec=0.35):
    import numpy as np

    frame_length = int(sample_rate * 0.02)
    usable = len(pcm) // frame_length * frame_length
    if not usable:
        return []
    frames = pcm[:usable].astype("float32").reshape(-1, frame_length) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    quiet = 20 * np.log10(np.maximum(rms, 1e-8)) <= threshold_db
    pauses, start = [], None
    for index, is_quiet in enumerate([*quiet, False]):
        if is_quiet and start is None:
            start = index
        if not is_quiet and start is not None:
            if (index - start) * 0.02 >= minimum_silence_sec:
                pauses.append((start + index) * 0.01)
            start = None
    return pauses


def make_asr_windows(duration, intervals, pauses=(), *, target_sec=45.0,
                     minimum_sec=20.0, maximum_sec=60.0, overlap_sec=1.25,
                     sample_rate=16000):
    if not (0 < minimum_sec <= target_sec <= maximum_sec
            and 0 <= overlap_sec < minimum_sec / 2 and duration > 0):
        raise ValueError("Thông số ASR window không hợp lệ.")
    slide_edges = [item["end_sec"] for item in intervals[:-1]]
    windows = []
    start = 0.0
    while start < duration - 1e-6:
        if duration - start <= maximum_sec:
            end = duration
        else:
            target = start + target_sec
            choices = [time for time in pauses
                       if start + minimum_sec <= time <= start + maximum_sec
                       and abs(time - target) <= 12.0]
            if choices:
                # Prefer nearby pauses; a coincident slide boundary is a small bonus.
                def cost(time):
                    slide_bonus = 2.0 if any(abs(time - edge) < 0.5
                                             for edge in slide_edges) else 0.0
                    return abs(time - target) - slide_bonus
                end = min(choices, key=cost)
            else:
                end = target
        clip_start_sample = max(0, math.floor((start - overlap_sec) * sample_rate))
        clip_end_sample = min(round(duration * sample_rate),
                              math.ceil((end + overlap_sec) * sample_rate))
        windows.append({
            "window_id": f"asr_{len(windows):05d}",
            "core_start_sec": start, "core_end_sec": end,
            "clip_start_sample": clip_start_sample, "clip_end_sample": clip_end_sample,
            "clip_start_sec": clip_start_sample / sample_rate,
            "clip_end_sec": clip_end_sample / sample_rate,
            "segment_ids": [item["segment_id"] for item in intervals
                            if item["start_sec"] < end and item["end_sec"] > start],
        })
        start = end
    return windows


def load_asr_model(model_id="Systran/faster-whisper-small", revision="main",
                   device="auto", device_index=0):
    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_path = snapshot_download(
        model_id, revision=revision,
        allow_patterns=["config.json", "model.bin", "tokenizer.json",
                        "vocabulary.*", "preprocessor_config.json"],
    )
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type,
                             device_index=device_index if device == "cuda" else 0)
    except RuntimeError as error:
        raise RuntimeError(
            f"Không khởi tạo được ASR trên {device}. Kiểm tra GPU/CUDA của Kaggle; "
            "có thể đặt ASR_DEVICE='cpu' để chạy int8. Lỗi gốc: " + str(error)
        ) from error
    return model, Path(model_path).name, device, compute_type


def transcribe_window(model, audio, language="vi", initial_prompt=None):
    segments, info = model.transcribe(
        audio, language=language, beam_size=5, temperature=0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250},
        word_timestamps=True, condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
    )
    return collect_asr_segments(segments, info)


def transcribe_audio_batched(model, audio, language="vi", initial_prompt=None, batch_size=8):
    from faster_whisper import BatchedInferencePipeline

    pipeline = BatchedInferencePipeline(model=model)
    segments, info = pipeline.transcribe(
        audio, language=language, beam_size=5, batch_size=batch_size,
        vad_filter=True, word_timestamps=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250},
        initial_prompt=initial_prompt or None,
    )
    return collect_asr_segments(segments, info)


def collect_asr_segments(segments, info):
    words, fallback_spans = [], []
    for segment in segments:
        if segment.words:
            words.extend({
                "start_sec": float(word.start), "end_sec": float(word.end),
                "text": word.word.strip(), "probability": float(word.probability),
            } for word in segment.words)
        elif segment.text.strip():
            # Never fabricate individual word timestamps from a segment.
            fallback_spans.append({
                "start_sec": float(segment.start), "end_sec": float(segment.end),
                "text": segment.text.strip(), "timing_precision": "segment",
            })
    return {"words": words, "fallback_spans": fallback_spans, "language": info.language,
            "duration_sec": getattr(info, "duration", None),
            "speech_duration_sec": getattr(info, "duration_after_vad", None)}


def globalize_window_words(raw_words, window, duration):
    result, invalid = [], 0
    offset = window["clip_start_sec"]
    for raw in raw_words:
        start, end = offset + raw["start_sec"], offset + raw["end_sec"]
        if (not math.isfinite(start) or not math.isfinite(end)
                or end < start or not raw["text"].strip()):
            invalid += 1
            continue
        start, end = max(0, start), min(duration, end)
        if start >= duration or end <= 0:
            invalid += 1
            continue
        end = min(duration, max(start + 1 / 16000, end))
        midpoint = (start + end) / 2
        if not window["core_start_sec"] <= midpoint < window["core_end_sec"]:
            continue
        probability = raw.get("probability")
        result.append({
            "start_sec": round(start, 6), "end_sec": round(end, 6),
            "text": raw["text"].strip(),
            "probability": probability if probability is not None
            and math.isfinite(probability) else None,
            "window_id": window["window_id"],
        })
    return result, invalid


def deduplicate_overlap_words(words):
    words = sorted(words, key=lambda word: (word["start_sec"], word["end_sec"]))
    kept, removed = [], 0
    for word in words:
        duplicate = False
        for previous in kept[-8:]:
            if previous["window_id"] == word["window_id"]:
                continue  # Repeated spoken words in one utterance are legitimate.
            if normalize_search(previous["text"]).strip(".,!?;:") != \
                    normalize_search(word["text"]).strip(".,!?;:"):
                continue
            overlap = max(0, min(word["end_sec"], previous["end_sec"])
                          - max(word["start_sec"], previous["start_sec"]))
            shorter = max(1e-6, min(word["end_sec"] - word["start_sec"],
                                   previous["end_sec"] - previous["start_sec"]))
            if overlap / shorter > 0.5:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(word)
    for index, word in enumerate(kept):
        word["word_id"] = f"word_{index:07d}"
    return kept, removed


def run_transcription(stage01_dir, video_path, output_root, video_id="L25_V001", *,
                      model_id="Systran/faster-whisper-small", revision="main",
                      language="vi", device="auto", initial_prompt=None,
                      target_sec=45.0, minimum_sec=20.0, maximum_sec=60.0,
                      overlap_sec=1.25, silence_threshold_db=-38.0,
                      transcription_mode="windowed", batch_size=8, device_index=0):
    import numpy as np

    if transcription_mode not in {"windowed", "batched"} or batch_size < 1:
        raise ValueError("ASR_MODE phải là windowed/batched; batch_size >= 1.")
    if stage01_dir is None:
        # ASR needs only the audio: it can run in parallel with stage 01. Slide
        # audio clips are then cut by stage 04, which has both branches.
        parent, intervals = None, []
        source = probe_video(video_path)
        duration = source["duration_sec"]
    else:
        parent = load_stage(stage01_dir, 1, video_id)
        if file_sha256(video_path) != parent["source"]["sha256"]:
            raise ValueError("Video ASR không khớp checksum video đã dùng để trích slide.")
        intervals = read_json(Path(stage01_dir) / "intervals.json")
        source, duration = parent["source"], parent["source"]["duration_sec"]
        verify_intervals(intervals, duration)
    directory = begin_stage(output_root, 3)
    parameters = {
        "model_id": model_id, "revision": revision, "language": language,
        "initial_prompt": initial_prompt, "target_sec": target_sec,
        "minimum_sec": minimum_sec, "maximum_sec": maximum_sec,
        "overlap_sec": overlap_sec, "silence_threshold_db": silence_threshold_db,
        "vad_filter": True, "word_timestamps": True, "condition_on_previous_text": False,
        "sample_rate": 16000,
        "transcription_mode": transcription_mode, "batch_size": batch_size,
        "window_controls_used": transcription_mode == "windowed",
        "packages": package_versions(["faster-whisper", "ctranslate2", "numpy"]),
    }
    all_words, fallback_spans, windows, slide_clips = [], [], [], []
    invalid_count, dedup_count = 0, 0
    if source["has_audio"]:
        audio_path = directory / "audio/lecture_16k.wav"
        normalize_audio(video_path, source, audio_path)
        pcm = read_pcm_audio(audio_path)
        if abs(len(pcm) / 16000 - duration) > 0.03:
            raise ValueError("Audio chuẩn hóa không còn khớp thời lượng video.")
        if transcription_mode == "batched":
            # VAD batching restores word times to the original normalized WAV.
            windows = [{
                "window_id": "asr_batched_full", "core_start_sec": 0.0,
                "core_end_sec": duration, "clip_start_sample": 0,
                "clip_end_sample": len(pcm), "clip_start_sec": 0.0,
                "clip_end_sec": len(pcm) / 16000,
                "segment_ids": [item["segment_id"] for item in intervals],
                "internal_segmentation": "faster_whisper_vad_with_restored_timestamps",
            }]
        else:
            pauses = silence_midpoints(pcm, threshold_db=silence_threshold_db)
            windows = make_asr_windows(
                duration, intervals, pauses, target_sec=target_sec, minimum_sec=minimum_sec,
                maximum_sec=maximum_sec, overlap_sec=overlap_sec,
            )
        model, resolved_revision, actual_device, compute_type = shared_model(
            ("asr", model_id, revision, device, device_index),
            lambda: load_asr_model(model_id, revision, device, device_index))
        parameters.update({
            "resolved_revision": resolved_revision,
            "device": actual_device, "compute_type": compute_type,
        })
        cache_directory = directory / "_cache" / fingerprint({
            "parameters": parameters,
            "parent": parent["artifact_id"] if parent else source["sha256"],
        })[:20]
        cache_directory.mkdir(parents=True, exist_ok=True)
        for index, window in enumerate(windows):
            cache = cache_directory / f"{window['window_id']}.json"
            if cache.exists():
                raw = read_json(cache)
            else:
                audio = pcm[window["clip_start_sample"]:window["clip_end_sample"]]
                normalized = audio.astype(np.float32) / 32768.0
                if transcription_mode == "batched":
                    raw = transcribe_audio_batched(model, normalized, language, initial_prompt, batch_size)
                else:
                    raw = transcribe_window(model, normalized, language, initial_prompt)
                write_json(cache, raw)
            if raw.get("speech_duration_sec") is not None:
                window["speech_duration_sec"] = raw["speech_duration_sec"]
            words, invalid = globalize_window_words(raw["words"], window, duration)
            invalid_count += invalid
            all_words.extend(words)
            for span in raw["fallback_spans"]:
                start = max(window["core_start_sec"],
                            window["clip_start_sec"] + span["start_sec"])
                end = min(window["core_end_sec"],
                          window["clip_start_sec"] + span["end_sec"])
                if start < end:
                    fallback_spans.append({
                        **span, "start_sec": start, "end_sec": end,
                        "window_id": window["window_id"],
                        "span_id": f"fallback_{len(fallback_spans):05d}",
                        "uncertainty": "Không có word timing; nội dung có thể chạm ranh giới.",
                    })
            print(f"ASR {index + 1}/{len(windows)}: "
                  f"{window['core_start_sec']:.1f}–{window['core_end_sec']:.1f}s")
        all_words, dedup_count = deduplicate_overlap_words(all_words)
        for interval in intervals:
            start_sample = round(interval["start_sec"] * 16000)
            end_sample = min(len(pcm), round(interval["end_sec"] * 16000))
            name = f"slide_audio/seg_{interval['sequence']:05d}.wav"
            save_pcm_clip(directory / name, pcm, start_sample, end_sample)
            slide_clips.append({
                "segment_id": interval["segment_id"], "path": name,
                "start_sec": start_sample / 16000, "end_sec": end_sample / 16000,
            })
    transcript = {
        "video_id": video_id, "language": language, "words": all_words,
        "fallback_spans": fallback_spans, "windows": windows,
        "slide_audio_clips": slide_clips,
        "status": "completed" if source["has_audio"] else "no_audio",
        "invalid_word_count": invalid_count, "overlap_duplicates_removed": dedup_count,
    }
    write_json(directory / "transcript.json", transcript)
    manifest = finish_stage(directory, 3, video_id, source, parameters,
                            parents=[parent] if parent else [],
                            extra={"word_count": len(all_words),
                                   "audio_status": transcript["status"]})
    return directory, manifest, transcript


CONFIG = {'videos': ['L25_V011', 'cs_0001'], 'glossaries': {'hVU5Mw19V_c': ['Bayes', 'xác suất', 'biến cố', 'công thức xác suất đầy đủ'], 'v2YTz8ToEag': ['PPF', 'đường giới hạn khả năng sản xuất', 'chi phí cơ hội', 'khan hiếm', 'kinh tế học'], 'y1CvB950VlU': ['Python', 'len', 'for', 'print', 'end', 'sep', 'chuỗi', 'string', 'a1', 'a2'], 'L25_V011': ['Liên Hiệp Quốc', 'Hội đồng Bảo an', 'quyền bình đẳng', 'chủ quyền'], 'cs_0001': ['FIFO', 'MSCs', 'RSC', 'synchronous', 'asynchronous', 'partial order', 'message passing', 'decidable', 'undecidable']}, 'soft_limit_seconds': 1100, 'models': [{'label': 'medium_hints', 'model_id': 'Systran/faster-whisper-medium', 'revision': '08e178d48790749d25932bbc082711ddcfdfbc4f', 'hints': True}]}
"""Run matched production windows; preserve raw segment diagnostics and timing."""
install('faster-whisper==1.2.1', 'ctranslate2==4.8.3', 'numpy==2.1.3',
        'huggingface-hub==0.36.0', 'av==15.1.0')
import numpy as np
import ctranslate2
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

save(OUT / 'environment.json', environment())
root = export_root()
save(OUT / 'config.json', CONFIG)


def prepare_audio(video):
    source = json.loads((root / video / '03_audio/manifest.json').read_text())['source']
    candidates = list(Path('/kaggle/input').rglob(video + '.mp4'))
    if not candidates:
        raise FileNotFoundError(video + '.mp4')
    mp4 = candidates[0]
    if file_sha256(mp4) != source['sha256']:
        raise ValueError('Video source hash mismatch: ' + video)
    wav = Path('/kaggle/working/audio') / (video + '.wav')
    wav.parent.mkdir(exist_ok=True)
    if not wav.exists():
        normalize_audio(mp4, source, wav)
    import wave
    with wave.open(str(wav), 'rb') as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
    return a, source, wav


def run_model(label, model_id, revision, hints=False, local_path=None):
    if hints:
        ctranslate2.set_random_seed(20261101)
    before = time.monotonic()
    if model_id=='vinai/PhoWhisper-large' and not local_path:
        install('transformers==4.57.1','accelerate==1.11.0','sentencepiece==0.2.1')
        from ctranslate2.converters import TransformersConverter
        original_path=snapshot_download(model_id,revision=revision)
        local_path='/kaggle/working/phowhisper_ct2'
        TransformersConverter(original_path,copy_files=['tokenizer.json','preprocessor_config.json']).convert(
            local_path,quantization='float16',force=True)
    model_path = local_path or snapshot_download(model_id, revision=revision,
        allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.*','preprocessor_config.json'])
    model = WhisperModel(model_path, device='cuda', device_index=0, compute_type='float16')
    loading = time.monotonic() - before
    for video in CONFIG['videos']:
        folder = OUT / label / video
        a, source, wav = prepare_audio(video)
        original = json.loads((root / video / '03_audio/transcript.json').read_text())
        windows = original['windows']
        language = 'en' if video == 'cs_0001' else 'vi'
        words, fallback, diagnostics, times, peak = [], [], [], [], 0
        invalid_total=0
        started = time.monotonic()
        for i, window in enumerate(windows):
            if time.monotonic() - START > CONFIG['soft_limit_seconds']:
                raise TimeoutError('Soft GPU budget reached; completed-video checkpoints retained')
            audio = a[window['clip_start_sample']:window['clip_end_sample']]
            kw = dict(language=language, beam_size=5, temperature=0, vad_filter=True,
                      vad_parameters={'min_silence_duration_ms':500,'speech_pad_ms':250},
                      word_timestamps=True, condition_on_previous_text=False, initial_prompt=None)
            if hints:
                glossary = ', '.join(CONFIG['glossaries'][video])
                kw.update(initial_prompt=glossary, hotwords=glossary,
                          temperature=[0.0,0.2,0.4,0.6], compression_ratio_threshold=2.4,
                          hallucination_silence_threshold=2.0)
            t = time.monotonic()
            segments, info = model.transcribe(audio, **kw)
            segments = list(segments)
            raw = collect_asr_segments(segments, info)
            global_words, invalid = globalize_window_words(raw['words'], window, source['duration_sec'])
            invalid_total+=invalid
            words.extend(global_words)
            for span in raw['fallback_spans']:
                span = dict(span)
                span['start_sec'] += window['clip_start_sec']; span['end_sec'] += window['clip_start_sec']
                span['window_id'] = window['window_id']
                if window['core_start_sec'] <= (span['start_sec']+span['end_sec'])/2 < window['core_end_sec']:
                    span.setdefault('span_id',f'fallback_{len(fallback):07d}')
                    fallback.append(span)
            for s in segments:
                diagnostics.append({'window_id':window['window_id'], 'start':s.start,'end':s.end,
                    'text':s.text,'avg_logprob':s.avg_logprob,'compression_ratio':s.compression_ratio,
                    'no_speech_prob':s.no_speech_prob,'temperature':s.temperature,
                    'raw_words':[{'start':w.start,'end':w.end,'text':w.word,'probability':w.probability}
                                 for w in (s.words or [])]})
            times.append(time.monotonic() - t)
            mem = vram(); peak = max(peak, mem[0] if mem else 0)
            if (i+1)%10 == 0 or i+1==len(windows):
                print(label, video, i+1, '/', len(windows), round(sum(times),1), 'seconds', flush=True)
        words, removed = deduplicate_overlap_words(words)
        transcript = dict(original, words=words, fallback_spans=fallback, status='completed',
                          invalid_word_count=invalid_total, overlap_duplicates_removed=removed)
        save(folder / 'transcript.json', transcript)
        save(folder / 'diagnostics.json', diagnostics)
        save(folder / 'metrics.json', {'model_id':model_id,'revision':revision,'settings':kw,
            'audio_sha256':file_sha256(wav),'duration_seconds':source['duration_sec'],
            'decode_seconds':sum(times),'video_elapsed_seconds':time.monotonic()-started,
            'rtf':sum(times)/source['duration_sec'],'peak_sampled_device_vram_mib':peak,
            'model_loading_seconds':loading,'window_seconds':times})
        if hints:
            save(folder / 'hint_provenance.json', {'random_seed':20261101,
                'glossary':CONFIG['glossaries'][video],
                'combined_ablation':'initial_prompt + hotwords + temperature fallback + hallucination silence threshold'})
    del model
    gc.collect()


for spec in CONFIG['models']:
    try:
        run_model(**spec)
    except Exception as e:
        failure(spec['label'], e)
if CONFIG['models']:finish()
