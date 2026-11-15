from fastapi import APIRouter, Depends, HTTPException

from app.features.videos.router import check_video_id, require_admin
from app.shared import storage
from . import jobs
from .models import ExtractionStatus


def require_enabled() -> None:
    if not jobs.enabled():
        raise HTTPException(503, "Trích xuất video đang tắt. Đặt LECTURE_EXTRACT=on trên máy chủ GPU để bật.")


router = APIRouter(prefix="/api/videos", tags=["extract"],
                   dependencies=[Depends(require_enabled), Depends(check_video_id)])


@router.post("/{video_id}/extract", response_model=ExtractionStatus, status_code=202,
             dependencies=[Depends(require_admin), Depends(storage.require_local_writes)])
def start_extraction(video_id: str):
    try:
        return jobs.get_queue().enqueue(video_id)
    except FileNotFoundError:
        raise HTTPException(404, "Không tìm thấy video.") from None


@router.get("/{video_id}/extract", response_model=ExtractionStatus)
def get_extraction(video_id: str):
    storage.require_local_writes()
    job = jobs.get_queue().get(video_id)
    if job is not None:
        return job
    if not storage.find_video_file(video_id):
        raise HTTPException(404, "Không tìm thấy video.")
    # Preloaded notebook exports are already ready, although they have no queue snapshot.
    from app.features.search import retrieval
    if storage.load_transcript(video_id) is not None and retrieval.has_index(video_id):
        return ExtractionStatus(video_id=video_id, status="completed", progress=1, updated_at=jobs._now())
    raise HTTPException(404, "Video chưa được đưa vào hàng đợi trích xuất.")
