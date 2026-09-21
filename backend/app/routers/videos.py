from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import storage
from ..models import Transcript, VideoInfo

router = APIRouter(prefix="/api/videos", tags=["videos"])

ALLOWED_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


@router.post("", response_model=VideoInfo)
async def upload_video(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {ext}")

    video_id = storage.new_video_id()
    dest = storage.video_path(video_id, ext)
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    return VideoInfo(video_id=video_id, filename=dest.name, has_transcript=False)


@router.get("", response_model=list[VideoInfo])
async def get_videos():
    return storage.list_videos()


@router.get("/{video_id}/file")
async def get_video_file(video_id: str):
    path = storage.find_video_file(video_id)
    if not path:
        raise HTTPException(404, "Không tìm thấy video")
    return FileResponse(path)


@router.get("/{video_id}/transcript", response_model=Transcript)
async def get_transcript(video_id: str):
    transcript = storage.load_transcript(video_id)
    if not transcript:
        raise HTTPException(404, "Video này chưa có transcript (chạy Whisper trên Kaggle rồi upload lên)")
    return transcript


@router.put("/{video_id}/transcript", response_model=Transcript)
async def put_transcript(video_id: str, transcript: Transcript):
    if transcript.video_id != video_id:
        raise HTTPException(400, "video_id không khớp")
    storage.save_transcript(transcript)
    return transcript
