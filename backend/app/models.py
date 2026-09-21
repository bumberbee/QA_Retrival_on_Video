from pydantic import BaseModel


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


class AskRequest(BaseModel):
    video_id: str
    question: str
    current_time: float | None = None  # thời điểm video đang phát, giây


class AskResponse(BaseModel):
    answer: str
    used_segments: list[TranscriptSegment] = []
