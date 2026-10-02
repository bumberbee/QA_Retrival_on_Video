import time

from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from fastapi import APIRouter, HTTPException

from .. import storage
from ..models import AnswerSource, AskRequest, AskResponse
from ..services import llm, retrieval

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("", response_model=AskResponse)
async def ask_question(req: AskRequest):
    # Videos imported from the extraction pipeline have a stage-05 index: answer from the
    # retrieved slide evidence (OCR + speech + slide images). Others: the whole transcript.
    use_index = retrieval.has_index(req.video_id)
    transcript = None if use_index else storage.load_transcript(req.video_id)
    if not use_index and not transcript:
        raise HTTPException(404, "Video này chưa có transcript, không thể trả lời câu hỏi")

    backend = llm.describe()
    started = time.perf_counter()
    sources: list[AnswerSource] = []
    try:
        if use_index:
            found = retrieval.search(req.video_id, req.question, req.current_time)
            context = retrieval.build_context(found)
            answer = llm.answer_with_evidence(req.question, req.current_time, found, context)
            sources = [AnswerSource(segment_id=s["segment_id"], start=s["start_sec"], end=s["end_sec"],
                                    title=s["title"] or "") for s in found["slides"]]
        else:
            answer = llm.answer_question(transcript, req.question, req.current_time)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except ResourceExhausted:
        raise HTTPException(
            429,
            "Đã hết quota Gemini API miễn phí cho hôm nay. Vui lòng thử lại sau hoặc "
            "nâng cấp gói Gemini API.",
        )
    except GoogleAPICallError as e:
        raise HTTPException(502, f"Gemini API lỗi: {e.message or str(e)}")
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    elapsed = time.perf_counter() - started
    print(
        f"[qa] {backend['provider']}:{backend['model']} mode={'evidence' if use_index else 'transcript'} "
        f"video={req.video_id} elapsed={elapsed:.1f}s"
    )
    return AskResponse(answer=answer, sources=sources, mode="evidence" if use_index else "transcript")
