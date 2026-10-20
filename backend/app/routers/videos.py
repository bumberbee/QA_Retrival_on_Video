from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from .. import storage
from ..models import SlideReview, Transcript, VideoImportRequest, VideoImportStatus, VideoInfo
from ..services import objects, video_import
from ..services.media import video_response

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

    original_filename = file.filename or dest.name
    storage.save_video_meta(video_id, original_filename, dest)
    meta = storage.load_video_meta(video_id) or {}

    return VideoInfo(
        video_id=video_id,
        filename=original_filename,
        has_transcript=False,
        duration_seconds=meta.get("duration_seconds"),
        uploaded_at=meta.get("uploaded_at"),
    )


@router.get("", response_model=list[VideoInfo])
async def get_videos():
    return storage.list_videos()


@router.post("/import", response_model=VideoImportStatus, status_code=202)
async def import_video(payload: VideoImportRequest):
    try:
        return video_import.start_import(payload.url)
    except video_import.ImportValidationError as error:
        raise HTTPException(400, str(error)) from error
    except video_import.ImportBusyError as error:
        raise HTTPException(429, str(error)) from error


@router.get("/import/{import_id}", response_model=VideoImportStatus)
async def get_video_import(import_id: str):
    job = video_import.get_import(import_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy lượt nhập video. Máy chủ có thể đã khởi động lại.")
    return job


@router.head("/{video_id}/file", include_in_schema=False)
@router.get("/{video_id}/file")
async def get_video_file(video_id: str, request: Request):
    # the browser plays the mp4 straight from R2 (signed URL, range requests work); without R2 the
    # files are served from LECTURE_RAG_DATA_DIR on the backend machine
    if storage.cloud() and objects.enabled():
        key = (storage.load_video_meta(video_id) or {}).get("video_key")
        if not key:
            raise HTTPException(404, "Không tìm thấy video")
        return RedirectResponse(objects.signed_url(key), status_code=307)
    path = storage.find_video_file(video_id)
    if not path:
        raise HTTPException(404, "Không tìm thấy video")
    return video_response(path, request)


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


@router.get("/{video_id}/outline")
async def get_outline(video_id: str):
    """Lecture structure from the stage-05 index: one entry per slide with its time and title."""
    from ..services import retrieval

    if not retrieval.has_index(video_id):
        return []
    outline = []
    for segment in retrieval.load_index(video_id).segments:
        visual = segment.get("visual", {})
        ocr = (visual.get("ocr_text_clean") or visual.get("ocr_text") or "").strip()
        first_line = next((line.strip() for line in ocr.splitlines() if len(line.strip()) >= 4), "")
        outline.append({
            "segment_id": segment["segment_id"], "start": segment.get("start_sec", 0), "end": segment.get("end_sec", 0),
            "title": (visual.get("title_clean") or visual.get("title") or first_line)[:140],
            "kind": "ad" if retrieval.is_ad(segment) else segment.get("content_kind", "slide"),
            "has_text": bool(ocr), "words": len(segment.get("speech", {}).get("word_ids", [])),
        })
    return outline


@router.get("/{video_id}/slides", response_model=list[str])
async def get_slides(video_id: str):
    return storage.list_keyframes(video_id)


@router.get("/{video_id}/slide-variants")
async def get_slide_variants(video_id: str):
    variants = storage.load_slide_variants(video_id)
    if variants is None:
        raise HTTPException(404, "Video này chưa có các bản gộp slide để so sánh")
    return variants


@router.get("/{video_id}/slide-reviews")
async def get_slide_reviews(video_id: str):
    return storage.load_slide_reviews(video_id)


@router.put("/{video_id}/slide-reviews/{variant}/{slide_key}")
async def put_slide_review(video_id: str, variant: str, slide_key: str, review: SlideReview):
    variants = storage.load_slide_variants(video_id)
    if variants is None or variant not in variants.get("variants", {}):
        raise HTTPException(404, "Không tìm thấy bản gộp slide này")
    reviews = storage.load_slide_reviews(video_id)
    entries = reviews.setdefault(variant, {})
    if review.verdict is None and not review.note:
        entries.pop(slide_key, None)
    else:
        entries[slide_key] = review.model_dump()
    storage.save_slide_reviews(video_id, reviews)
    return reviews


@router.get("/{video_id}/slides/{filename:path}")
async def get_slide_file(video_id: str, filename: str):
    if storage.cloud() and objects.enabled():
        from ..services import supa
        key = supa.keyframe_keys(video_id).get(filename)
        if not key:
            raise HTTPException(404, "Không tìm thấy slide")
        return RedirectResponse(objects.signed_url(key), status_code=307)
    path = storage.slide_file(video_id, filename)
    if not path:
        raise HTTPException(404, "Không tìm thấy slide")
    return FileResponse(path)
