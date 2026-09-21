import json
import uuid
from pathlib import Path

from .models import Transcript, VideoInfo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VIDEOS_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def new_video_id() -> str:
    return uuid.uuid4().hex[:12]


def video_path(video_id: str, ext: str) -> Path:
    return VIDEOS_DIR / f"{video_id}{ext}"


def transcript_path(video_id: str) -> Path:
    return TRANSCRIPTS_DIR / f"{video_id}.json"


def save_transcript(transcript: Transcript) -> None:
    path = transcript_path(transcript.video_id)
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def load_transcript(video_id: str) -> Transcript | None:
    path = transcript_path(video_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(**data)


def list_videos() -> list[VideoInfo]:
    videos = []
    for f in sorted(VIDEOS_DIR.iterdir()):
        if not f.is_file():
            continue
        video_id = f.stem
        videos.append(
            VideoInfo(
                video_id=video_id,
                filename=f.name,
                has_transcript=transcript_path(video_id).exists(),
            )
        )
    return videos


def find_video_file(video_id: str) -> Path | None:
    matches = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    return matches[0] if matches else None
