from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    video_id: str
    language: str | None = None
    segments: list[TranscriptSegment]


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    has_transcript: bool
    has_slides: bool = False
    duration_seconds: float | None = None
    uploaded_at: str | None = None  # ISO 8601
    source_url: str | None = None
    source_platform: Literal["youtube", "facebook", "tiktok"] | None = None


class VideoImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class VideoImportStatus(BaseModel):
    import_id: str
    status: Literal["queued", "downloading", "completed", "failed"]
    progress: float | None = None
    video: VideoInfo | None = None
    error: str | None = None


class SlideReview(BaseModel):
    # good: slide đúng; wrong_merge: gộp nhầm nhiều slide; extra_split: tách thừa (giống slide trước);
    # bad_image: ảnh đại diện sai/mờ; not_slide: không phải slide (quảng cáo, cảnh giảng viên); other: khác
    verdict: Literal["good", "wrong_merge", "extra_split", "bad_image", "not_slide", "other"] | None = None
    note: str = Field(default="", max_length=2000)


class AskRequest(BaseModel):
    video_id: str
    question: str
    current_time: float | None = None  # thời điểm video đang phát, giây
    previous_question: str | None = None  # câu hỏi liền trước trong cuộc trò chuyện
    video_ids: list[str] | None = None  # hỏi trên các video được chọn (có thể không gồm video đang xem)


class AnswerSource(BaseModel):
    segment_id: str
    start: float
    end: float
    title: str = ""
    video_id: str | None = None
    video_title: str = ""


class AskResponse(BaseModel):
    answer: str
    used_segments: list[TranscriptSegment] = []
    sources: list[AnswerSource] = []
    mode: Literal["evidence", "transcript"] = "transcript"
