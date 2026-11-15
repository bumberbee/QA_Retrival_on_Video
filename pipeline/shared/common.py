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
