import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from .models import Transcript, VideoInfo


def cloud() -> bool:
    """Supabase holds the data (SUPABASE_DB_URL set); otherwise the files under DATA_DIR."""
    return bool(os.getenv("SUPABASE_DB_URL"))


def _supa():
    from .services import supa
    return supa

# LECTURE_RAG_DATA_DIR moves the app data off the default backend/data (e.g. to a bigger drive).
DATA_DIR = Path(os.getenv("LECTURE_RAG_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
VIDEOS_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
SLIDE_VARIANTS_DIR = DATA_DIR / "slide_variants"
SLIDE_REVIEWS_DIR = DATA_DIR / "slide_reviews"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)


def new_video_id() -> str:
    return uuid.uuid4().hex[:12]


class InvalidVideoId(ValueError):
    pass


class UnsafeStoragePath(ValueError):
    pass


class CloudWriteUnsupported(RuntimeError):
    pass


def validate_video_id(video_id: str) -> str:
    # Both uploads and import_fast_pipeline.py use new_video_id(), including imported lectures.
    if not isinstance(video_id, str) or not re.fullmatch(r"[0-9a-f]{12}", video_id):
        raise InvalidVideoId("Invalid video_id (expected 12 lowercase hex characters)")
    return video_id


def contained_path(base: Path, *parts: str) -> Path:
    """Keep paths under their fixed storage base, including through symlinks/junctions."""
    for part in parts:
        if ("\\" in part or ":" in part or Path(part).is_absolute()
                or PureWindowsPath(part).drive or any(ord(c) < 32 for c in part)):
            raise UnsafeStoragePath("Invalid storage path")
    root = base.resolve()
    path = root.joinpath(*parts).resolve()
    if root not in path.parents:
        raise UnsafeStoragePath("Storage path escapes its base directory")
    return path


def require_local_writes() -> None:
    if cloud():
        raise CloudWriteUnsupported("Video uploads, URL imports and transcript replacement are unsupported in cloud mode")


def _write_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".write-", suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            out.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def video_path(video_id: str, ext: str) -> Path:
    validate_video_id(video_id)
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", ext):
        raise UnsafeStoragePath("Invalid video extension")
    return contained_path(VIDEOS_DIR, f"{video_id}{ext}")


def video_meta_path(video_id: str) -> Path:
    return contained_path(VIDEOS_DIR, f"{validate_video_id(video_id)}.meta.json")


def probe_duration_seconds(video_file: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_file),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return round(float(result.stdout.strip()), 1)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def save_video_meta(
    video_id: str,
    original_filename: str,
    video_file: Path,
    *,
    source_url: str | None = None,
    source_platform: str | None = None,
) -> None:
    validate_video_id(video_id)
    require_local_writes()
    _write_text(video_meta_path(video_id),
        json.dumps(
            {
                "original_filename": original_filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": probe_duration_seconds(video_file),
                "source_url": source_url,
                "source_platform": source_platform,
            },
            ensure_ascii=False,
        ),
    )


def load_video_meta(video_id: str) -> dict | None:
    validate_video_id(video_id)
    if cloud():
        return _supa().video_meta(video_id)
    path = video_meta_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_video_id_by_original_filename(original_filename: str) -> str | None:
    if cloud():
        video_id = _supa().find_id_by_filename(original_filename)
        return validate_video_id(video_id) if video_id is not None else None
    for meta_file in VIDEOS_DIR.glob("*.meta.json"):
        video_id = meta_file.stem.removesuffix(".meta")
        try:
            meta_file = video_meta_path(video_id)
        except (InvalidVideoId, UnsafeStoragePath):
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("original_filename") == original_filename:
            return video_id
    return None


def transcript_path(video_id: str) -> Path:
    return contained_path(TRANSCRIPTS_DIR, f"{validate_video_id(video_id)}.json")


def save_transcript(transcript: Transcript) -> None:
    validate_video_id(transcript.video_id)
    require_local_writes()
    path = transcript_path(transcript.video_id)
    _write_text(path, transcript.model_dump_json(indent=2))


def load_transcript(video_id: str) -> Transcript | None:
    validate_video_id(video_id)
    if cloud():
        return _supa().transcript(video_id)
    path = transcript_path(video_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(**data)


def keyframes_dir(video_id: str) -> Path:
    return contained_path(KEYFRAMES_DIR, validate_video_id(video_id))


def list_keyframes(video_id: str) -> list[str]:
    validate_video_id(video_id)
    if cloud():
        return sorted(name for name in _supa().keyframe_keys(video_id) if "/" not in name)
    d = keyframes_dir(video_id)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if slide_file(video_id, p.name))


def slide_file(video_id: str, name: str) -> Path | None:
    """Resolve a slide image under the video's keyframe folder; None if outside or missing."""
    validate_video_id(video_id)
    try:
        path = contained_path(keyframes_dir(video_id), name)
    except UnsafeStoragePath:
        return None
    if not path.is_file():
        return None
    return path


def load_slide_variants(video_id: str) -> dict | None:
    """Alternative slide segmentations for evaluation (written by scripts/import_slide_variants.py)."""
    validate_video_id(video_id)
    if cloud():
        return _supa().slide_variants(video_id)
    path = contained_path(SLIDE_VARIANTS_DIR, f"{video_id}.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_slide_reviews(video_id: str) -> dict:
    validate_video_id(video_id)
    if cloud():
        return _supa().slide_reviews(video_id)
    path = contained_path(SLIDE_REVIEWS_DIR, f"{video_id}.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_slide_reviews(video_id: str, reviews: dict) -> None:
    validate_video_id(video_id)
    if cloud():
        return _supa().save_slide_reviews(video_id, reviews)
    SLIDE_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    path = contained_path(SLIDE_REVIEWS_DIR, f"{video_id}.json")
    _write_text(path, json.dumps(reviews, ensure_ascii=False, indent=2))


def list_videos() -> list[VideoInfo]:
    if cloud():
        return _supa().list_videos()
    videos = []
    for f in sorted(VIDEOS_DIR.iterdir()):
        if not f.is_file() or f.name.endswith(".meta.json"):
            continue
        video_id = f.stem
        try:
            validate_video_id(video_id)
            f = contained_path(VIDEOS_DIR, f.name)
        except (InvalidVideoId, UnsafeStoragePath):
            continue
        meta = load_video_meta(video_id)
        videos.append(
            VideoInfo(
                video_id=video_id,
                filename=meta["original_filename"] if meta else f.name,
                has_transcript=transcript_path(video_id).exists(),
                has_slides=len(list_keyframes(video_id)) > 0,
                duration_seconds=meta.get("duration_seconds") if meta else None,
                uploaded_at=meta.get("uploaded_at") if meta else None,
                source_url=meta.get("source_url") if meta else None,
                source_platform=meta.get("source_platform") if meta else None,
            )
        )
    return videos


def find_video_file(video_id: str) -> Path | None:
    validate_video_id(video_id)
    matches = [
        contained_path(VIDEOS_DIR, p.name) for p in sorted(VIDEOS_DIR.glob(f"{video_id}.*"))
        if p.is_file() and not p.name.endswith(".meta.json")
    ]
    return matches[0] if matches else None
