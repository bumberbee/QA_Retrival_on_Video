from fastapi import APIRouter, HTTPException

from .. import storage
from ..models import AskRequest, AskResponse
from ..services import gemini

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("", response_model=AskResponse)
async def ask_question(req: AskRequest):
    transcript = storage.load_transcript(req.video_id)
    if not transcript:
        raise HTTPException(404, "Video này chưa có transcript, không thể trả lời câu hỏi")

    try:
        answer = gemini.answer_question(transcript, req.question, req.current_time)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return AskResponse(answer=answer)
