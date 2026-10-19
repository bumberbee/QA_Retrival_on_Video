import json
import time

from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import storage
from ..models import AnswerSource, AskRequest, AskResponse
from ..services import llm, retrieval

router = APIRouter(prefix="/api/qa", tags=["qa"])


def find_evidence(req: AskRequest, query: str) -> tuple[dict, str]:
    """Search the videos ticked in the list (video_ids), or the video being watched when none is."""
    ids = list(dict.fromkeys(req.video_ids or [req.video_id]))
    if ids == [req.video_id]:
        found = retrieval.search(req.video_id, query, req.current_time)
        return found, retrieval.build_context(found)
    found = retrieval.search_many(ids, query, req.video_id, req.current_time)
    note = (f"(Bằng chứng lấy từ {found.get('videos', len(ids))} video; khi trích mốc thời gian, "
            "ghi kèm tên video.)\n\n")
    return found, note + retrieval.build_context(found)


def source_dict(slide: dict) -> dict:
    return {"segment_id": slide["segment_id"], "start": slide["start_sec"], "end": slide["end_sec"],
            "title": slide["title"] or "", "video_id": slide.get("video_id"),
            "video_title": slide.get("video_title", "")}


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
            query = retrieval.search_question(req.question, req.previous_question)
            found, context = find_evidence(req, query)
            answer = llm.answer_with_evidence(req.question, req.current_time, found, context,
                                              previous=req.previous_question if query != req.question else None)
            sources = [AnswerSource(**source_dict(s)) for s in found["slides"]]
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


@router.post("/stream")
def ask_question_stream(req: AskRequest):
    """The evidence answer as NDJSON lines: {"type": "sources"}, then {"type": "delta"} pieces,
    then {"type": "done"} or {"type": "error"}. Videos without an index use POST /api/qa."""
    if not retrieval.has_index(req.video_id):
        raise HTTPException(404, "Video này chưa có chỉ mục, dùng /api/qa")
    query = retrieval.search_question(req.question, req.previous_question)
    previous = req.previous_question if query != req.question else None
    found, context = find_evidence(req, query)
    sources = [source_dict(s) for s in found["slides"]]

    def lines():
        backend = llm.describe()
        started = time.perf_counter()
        first = None
        yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"
        try:
            for piece in llm.stream_with_evidence(req.question, req.current_time, found, context, previous):
                first = first or time.perf_counter() - started
                yield json.dumps({"type": "delta", "text": piece}, ensure_ascii=False) + "\n"
        except ResourceExhausted:
            yield json.dumps({"type": "error", "detail": "Đã hết quota Gemini API miễn phí cho hôm nay."},
                             ensure_ascii=False) + "\n"
            return
        except (llm.LLMUnavailable, GoogleAPICallError, RuntimeError) as e:
            yield json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
            return
        print(f"[qa-stream] {backend['provider']}:{backend['model']} video={req.video_id} "
              f"first={first or 0:.1f}s total={time.perf_counter() - started:.1f}s")
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
