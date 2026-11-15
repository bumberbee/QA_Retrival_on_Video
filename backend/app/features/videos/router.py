import os
import secrets
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.formparsers import MultiPartException

from app.shared import storage
from app.shared.models import SlideReview, Transcript, VideoImportRequest, VideoImportStatus, VideoInfo
from app.shared import objects
from app.features.videos import importer as video_import
from app.features.videos.media import video_response
from app.features.extract.jobs import enqueue_after_import


def check_video_id(request: Request) -> None:
    if "video_id" in request.path_params:
        storage.validate_video_id(request.path_params["video_id"])


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = os.getenv("ADMIN_TOKEN")
    if token:
        if not x_admin_token or not secrets.compare_digest(token.encode(), x_admin_token.encode()):
            raise HTTPException(403, "A valid X-Admin-Token is required")
    elif storage.cloud():
        raise HTTPException(403, "Writes are disabled in cloud mode until ADMIN_TOKEN is configured")


def max_upload_bytes() -> int:
    limit = int(float(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
    if limit <= 0:
        raise ValueError("MAX_UPLOAD_MB must be positive")
    return limit


class UploadLimitMiddleware:
    """Bound incoming multipart data before FastAPI spools it, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/api/videos":
            return await self.app(scope, receive, send)
        request = Request(scope)
        try:
            require_admin(request.headers.get("x-admin-token"))
            storage.require_local_writes()
        except (HTTPException, storage.CloudWriteUnsupported) as error:
            return await JSONResponse({"detail": str(error.detail) if isinstance(error, HTTPException) else str(error)},
                                      status_code=403)(scope, receive, send)
        # The route enforces the exact file limit; allow a bounded multipart envelope here.
        limit = max_upload_bytes() + 64 * 1024
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > limit:
            return await JSONResponse({"detail": "Upload exceeds MAX_UPLOAD_MB"}, status_code=413)(scope, receive, send)
        size = 0
        exceeded = False

        async def bounded_receive():
            nonlocal size, exceeded
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                exceeded = True
                # Starlette closes partial spool files when this parser exception is raised.
                raise MultiPartException("Upload exceeds MAX_UPLOAD_MB")
            return message

        async def bounded_send(message):
            if not exceeded:
                await send(message)

        await self.app(scope, bounded_receive, bounded_send)
        if exceeded:
            await JSONResponse({"detail": "Upload exceeds MAX_UPLOAD_MB"}, status_code=413)(scope, receive, send)


router = APIRouter(prefix="/api/videos", tags=["videos"], dependencies=[Depends(check_video_id)])

ALLOWED_EXTS = {".mp4", ".webm", ".mov", ".mkv"}
_review_lock = threading.Lock()


@router.post("", response_model=VideoInfo, dependencies=[Depends(require_admin), Depends(storage.require_local_writes)])
def upload_video(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {ext}")

    video_id = storage.new_video_id()
    dest = storage.video_path(video_id, ext)
    limit = max_upload_bytes()
    temporary = None
    completed = False
    try:
        if file.size is not None and file.size > limit:
            raise HTTPException(413, "Upload exceeds MAX_UPLOAD_MB")
        size = 0
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".upload-", suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "Upload exceeds MAX_UPLOAD_MB")
                out.write(chunk)
        original_filename = file.filename or dest.name
        storage.save_video_meta(video_id, original_filename, temporary)
        temporary.replace(dest)
        meta = storage.load_video_meta(video_id) or {}
        completed = True
    finally:
        file.file.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            if not completed:
                dest.unlink(missing_ok=True)
                storage.video_meta_path(video_id).unlink(missing_ok=True)

    enqueue_after_import(video_id)
    return VideoInfo(
        video_id=video_id,
        filename=original_filename,
        has_transcript=False,
        duration_seconds=meta.get("duration_seconds"),
        uploaded_at=meta.get("uploaded_at"),
    )


@router.get("", response_model=list[VideoInfo])
def get_videos():
    return storage.list_videos()


@router.post("/import", response_model=VideoImportStatus, status_code=202,
             dependencies=[Depends(require_admin), Depends(storage.require_local_writes)])
def import_video(payload: VideoImportRequest):
    try:
        return video_import.start_import(payload.url)
    except video_import.ImportValidationError as error:
        raise HTTPException(400, str(error)) from error
    except video_import.ImportBusyError as error:
        raise HTTPException(429, str(error)) from error


@router.get("/import/{import_id}", response_model=VideoImportStatus)
def get_video_import(import_id: str):
    job = video_import.get_import(import_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy lượt nhập video. Máy chủ có thể đã khởi động lại.")
    return job


@router.head("/{video_id}/file", include_in_schema=False)
@router.get("/{video_id}/file")
def get_video_file(video_id: str, request: Request):
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
def get_transcript(video_id: str):
    transcript = storage.load_transcript(video_id)
    if not transcript:
        raise HTTPException(404, "Video này chưa có transcript (chạy Whisper trên Kaggle rồi upload lên)")
    return transcript


@router.put("/{video_id}/transcript", response_model=Transcript,
            dependencies=[Depends(require_admin), Depends(storage.require_local_writes)])
def put_transcript(video_id: str, transcript: Transcript):
    if transcript.video_id != video_id:
        raise HTTPException(400, "video_id không khớp")
    storage.save_transcript(transcript)
    return transcript


@router.get("/{video_id}/outline")
def get_outline(video_id: str):
    """Lecture structure from the stage-05 index: one entry per slide with its time and title."""
    from app.features.search import retrieval

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
def get_slides(video_id: str):
    return storage.list_keyframes(video_id)


@router.get("/{video_id}/slide-variants")
def get_slide_variants(video_id: str):
    variants = storage.load_slide_variants(video_id)
    if variants is None:
        raise HTTPException(404, "Video này chưa có các bản gộp slide để so sánh")
    return variants


@router.get("/{video_id}/slide-reviews")
def get_slide_reviews(video_id: str):
    return storage.load_slide_reviews(video_id)


@router.put("/{video_id}/slide-reviews/{variant}/{slide_key}", dependencies=[Depends(require_admin)])
def put_slide_review(video_id: str, variant: str, slide_key: str, review: SlideReview):
    variants = storage.load_slide_variants(video_id)
    if variants is None or variant not in variants.get("variants", {}):
        raise HTTPException(404, "Không tìm thấy bản gộp slide này")
    # Threadpool handlers must keep the review read/modify/write together.
    with _review_lock:
        reviews = storage.load_slide_reviews(video_id)
        entries = reviews.setdefault(variant, {})
        if review.verdict is None and not review.note:
            entries.pop(slide_key, None)
        else:
            entries[slide_key] = review.model_dump()
        storage.save_slide_reviews(video_id, reviews)
    return reviews


@router.get("/{video_id}/slides/{filename:path}")
def get_slide_file(video_id: str, filename: str):
    if storage.cloud() and objects.enabled():
        from app.shared import supabase as supa
        key = supa.keyframe_keys(video_id).get(filename)
        if not key:
            raise HTTPException(404, "Không tìm thấy slide")
        return RedirectResponse(objects.signed_url(key), status_code=307)
    path = storage.slide_file(video_id, filename)
    if not path:
        raise HTTPException(404, "Không tìm thấy slide")
    return FileResponse(path)
