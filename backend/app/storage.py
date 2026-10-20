import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

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


def video_path(video_id: str, ext: str) -> Path:
    return VIDEOS_DIR / f"{video_id}{ext}"


def video_meta_path(video_id: str) -> Path:
    return VIDEOS_DIR / f"{video_id}.meta.json"


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
    video_meta_path(video_id).write_text(
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
        encoding="utf-8",
    )


def load_video_meta(video_id: str) -> dict | None:
    if cloud():
        return _supa().video_meta(video_id)
    path = video_meta_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_video_id_by_original_filename(original_filename: str) -> str | None:
    if cloud():
        return _supa().find_id_by_filename(original_filename)
    for meta_file in VIDEOS_DIR.glob("*.meta.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("original_filename") == original_filename:
            return meta_file.stem.removesuffix(".meta")
    return None


def transcript_path(video_id: str) -> Path:
    return TRANSCRIPTS_DIR / f"{video_id}.json"


def save_transcript(transcript: Transcript) -> None:
    path = transcript_path(transcript.video_id)
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def load_transcript(video_id: str) -> Transcript | None:
    if cloud():
        return _supa().transcript(video_id)
    path = transcript_path(video_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(**data)


def keyframes_dir(video_id: str) -> Path:
    return KEYFRAMES_DIR / video_id


def list_keyframes(video_id: str) -> list[str]:
    if cloud():
        return sorted(name for name in _supa().keyframe_keys(video_id) if "/" not in name)
    d = keyframes_dir(video_id)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file())


def slide_file(video_id: str, name: str) -> Path | None:
    """Resolve a slide image under the video's keyframe folder; None if outside or missing."""
    root = keyframes_dir(video_id).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path


def load_slide_variants(video_id: str) -> dict | None:
    """Alternative slide segmentations for evaluation (written by scripts/import_slide_variants.py)."""
    if cloud():
        return _supa().slide_variants(video_id)
    path = SLIDE_VARIANTS_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_slide_reviews(video_id: str) -> dict:
    if cloud():
        return _supa().slide_reviews(video_id)
    path = SLIDE_REVIEWS_DIR / f"{video_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_slide_reviews(video_id: str, reviews: dict) -> None:
    if cloud():
        return _supa().save_slide_reviews(video_id, reviews)
    SLIDE_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    path = SLIDE_REVIEWS_DIR / f"{video_id}.json"
    path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")


def list_videos() -> list[VideoInfo]:
    if cloud():
        return _supa().list_videos()
    videos = []
    for f in sorted(VIDEOS_DIR.iterdir()):
        if not f.is_file() or f.name.endswith(".meta.json"):
            continue
        video_id = f.stem
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
    matches = [
        p for p in VIDEOS_DIR.glob(f"{video_id}.*") if not p.name.endswith(".meta.json")
    ]
    return matches[0] if matches else None
