import json
import logging
import threading
import time

from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from .. import storage
from ..models import AnswerSource, AskRequest, AskResponse
from ..services import llm, retrieval, sessions

router = APIRouter(prefix="/api/qa", tags=["qa"])
logger = logging.getLogger(__name__)


SUMMARY_EVERY = 4      # turns between two updates of the running summary
HISTORY_TURNS = 3      # last turns given to the answer model (older ones live in the summary)


class QAAnswerResponse(AskResponse):
    # Extend the response here to keep the existing shared/frontend fields intact.
    evidence_map: dict[str, dict] = Field(default_factory=dict)
    cited_sources: list[dict] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)


def citation_metadata(found: dict) -> dict:
    return {"evidence_map": found.get("evidence_map", {}),
            "cited_sources": found.get("cited_sources", []),
            "invalid_citations": found.get("invalid_citations", [])}


def log_answer(backend: dict, req: AskRequest, timings: dict, *, stream: bool,
               mode: str, success: bool) -> None:
    # Preserve the existing stdout logging without depending on a root INFO handler.
    print(json.dumps({"event": "qa_answer", "provider": backend["provider"],
                      "model": backend["model"], "video_id": req.video_id,
                      "stream": stream, "mode": mode,
                      "status": "ok" if success else "error", "timings": timings}), flush=True)


def find_evidence(req: AskRequest, query: str, prefer: frozenset = frozenset()) -> tuple[dict, str]:
    """Search the videos ticked in the list (video_ids), or the video being watched when none is."""
    ids = list(dict.fromkeys(req.video_ids or [req.video_id]))
    if ids == [req.video_id]:
        found = retrieval.search(req.video_id, query, req.current_time, prefer_segments=prefer)
        context, _ = retrieval.build_context_with_ids(found)
        return found, context
    found = retrieval.search_many(ids, query, req.video_id, req.current_time, prefer_segments=prefer)
    note = (f"(Bằng chứng lấy từ {found.get('videos', len(ids))} video; "
            "trích dẫn bằng mã nguồn của từng đoạn.)\n\n")
    context, _ = retrieval.build_context_with_ids(found)
    return found, note + context


def source_dict(slide: dict) -> dict:
    return {"segment_id": slide["segment_id"], "start": slide["start_sec"], "end": slide["end_sec"],
            "title": slide["title"] or "", "video_id": slide.get("video_id"),
            "video_title": slide.get("video_title", "")}


def prepare(req: AskRequest, client_id: str | None) -> dict:
    """The conversation, the query to search with and the memory for the prompt.

    A follow-up ("còn … thì sao", "nó") is rewritten into a standalone question from the last turns
    (LLM4CS); if that fails it falls back to appending the previous question. A standalone question
    is searched as it is (appending history hurts after a topic switch, TopiOCQA)."""
    conversation = None
    if client_id and sessions.valid_client(client_id):
        if req.conversation_id:
            conversation = sessions.get(req.conversation_id, client_id)
        if conversation is None:
            conversation = sessions.create(client_id, req.video_id, req.video_ids)
    history = sessions.recent(conversation, HISTORY_TURNS)
    previous = history[-1]["question"] if history else req.previous_question
    follow_up = bool(previous) and retrieval.is_follow_up(req.question)
    rewritten = llm.rewrite_question(req.question, history) if follow_up and history else None
    query = rewritten or retrieval.search_question(req.question, previous)
    last_sources = history[-1].get("sources", []) if history else []
    prefer = frozenset(s["segment_id"] for s in last_sources) if follow_up else frozenset()
    return {"conversation": conversation, "history": history, "query": query, "prefer": prefer,
            "rewritten": rewritten, "follow_up": follow_up,
            "previous": previous if (query != req.question and not history) else None,
            "summary": (conversation or {}).get("summary", "")}


def remember(req: AskRequest, client_id: str | None, plan: dict, answer: str, sources: list, seconds: float) -> None:
    """Save the turn; once SUMMARY_EVERY turns are older than the history window, fold them into
    the running summary in the background (recursive summarizing)."""
    conversation = plan["conversation"]
    if not conversation or not answer.strip():
        return
    saved = sessions.append_turn(conversation["id"], client_id, {
        "question": req.question, "rewritten": plan["rewritten"], "query": plan["query"],
        "answer": answer, "sources": sources, "seconds": round(seconds, 1),
        "video_id": req.video_id, "video_ids": req.video_ids}, req.video_id, req.video_ids)
    if not saved:
        return
    covered = saved.get("summarized_turns", 0)
    older = saved["turns"][covered:max(covered, len(saved["turns"]) - HISTORY_TURNS)]
    if len(older) >= SUMMARY_EVERY:
        def run():
            summary = llm.summarize(saved.get("summary", ""), older)
            if summary:
                sessions.set_summary(saved["id"], client_id, summary, covered + len(older))
        threading.Thread(target=run, daemon=True).start()


def memory(plan: dict) -> dict:
    return {"history": plan["history"], "summary": plan["summary"], "rewritten": plan["rewritten"]}


def validate_video_ids(req: AskRequest) -> None:
    for video_id in [req.video_id, *(req.video_ids or [])]:
        storage.validate_video_id(video_id)


@router.post("", response_model=QAAnswerResponse)
def ask_question(req: AskRequest, x_client_id: str | None = Header(default=None)):
    # Videos imported from the extraction pipeline have a stage-05 index: answer from the
    # retrieved slide evidence (OCR + speech + slide images). Others: the whole transcript.
    started = time.perf_counter()
    validate_video_ids(req)
    use_index = retrieval.has_index(req.video_id)
    retrieval_started = time.perf_counter()
    transcript = None if use_index else storage.load_transcript(req.video_id)
    retrieval_seconds = 0.0 if use_index else time.perf_counter() - retrieval_started
    if not use_index and not transcript:
        raise HTTPException(404, "Video này chưa có transcript, không thể trả lời câu hỏi")

    backend = llm.describe()
    timings = {"retrieval_seconds": retrieval_seconds, "total_seconds": 0.0,
               "ttft_seconds": None, "ttft_is_buffered": True}
    sources: list[AnswerSource] = []
    plan = None
    found = {}
    success = False
    try:
        if use_index:
            plan = prepare(req, x_client_id)
            retrieval_started = time.perf_counter()
            found, context = find_evidence(req, plan["query"], plan["prefer"])
            timings["retrieval_seconds"] = time.perf_counter() - retrieval_started
            answer = llm.answer_with_evidence(req.question, req.current_time, found, context,
                                              previous=plan["previous"], timings=timings, **memory(plan))
            sources = [AnswerSource(**source_dict(s)) for s in found["slides"]]
            remember(req, x_client_id, plan, answer, [s.model_dump() for s in sources],
                     time.perf_counter() - started)
        else:
            answer = llm.answer_question(transcript, req.question, req.current_time, timings=timings)
        success = True
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
    finally:
        timings["request_total_seconds"] = time.perf_counter() - started
        log_answer(backend, req, timings, stream=False,
                   mode="evidence" if use_index else "transcript", success=success)

    return QAAnswerResponse(answer=answer, sources=sources, mode="evidence" if use_index else "transcript",
                            conversation_id=((plan or {}).get("conversation") or {}).get("id"),
                            rewritten=(plan or {}).get("rewritten"), timings=timings, **citation_metadata(found))


@router.post("/stream")
def ask_question_stream(req: AskRequest, x_client_id: str | None = Header(default=None)):
    """The evidence answer as NDJSON lines: {"type": "sources"}, then {"type": "delta"} pieces,
    then {"type": "done"} or {"type": "error"}. Videos without an index use POST /api/qa."""
    started = time.perf_counter()
    validate_video_ids(req)
    if not retrieval.has_index(req.video_id):
        raise HTTPException(404, "Video này chưa có chỉ mục, dùng /api/qa")
    plan = prepare(req, x_client_id)
    retrieval_started = time.perf_counter()
    found, context = find_evidence(req, plan["query"], plan["prefer"])
    timings = {"retrieval_seconds": time.perf_counter() - retrieval_started,
               "total_seconds": 0.0, "ttft_seconds": None, "ttft_is_buffered": False}
    sources = [source_dict(s) for s in found["slides"]]

    def lines():
        backend = llm.describe()
        yield json.dumps({"type": "meta", "conversation_id": (plan["conversation"] or {}).get("id"),
                          "rewritten": plan["rewritten"], "follow_up": plan["follow_up"]},
                         ensure_ascii=False) + "\n"
        yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"
        answer = []
        success = False
        try:
            for piece in llm.stream_with_evidence(req.question, req.current_time, found, context,
                                                  plan["previous"], timings=timings, **memory(plan)):
                answer.append(piece)
                yield json.dumps({"type": "delta", "text": piece}, ensure_ascii=False) + "\n"
            remember(req, x_client_id, plan, "".join(answer), sources, time.perf_counter() - started)
            success = True
        except ResourceExhausted:
            yield json.dumps({"type": "error", "detail": "Đã hết quota Gemini API miễn phí cho hôm nay."},
                             ensure_ascii=False) + "\n"
            return
        except (llm.LLMUnavailable, GoogleAPICallError, RuntimeError) as e:
            yield json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
            return
        except Exception:
            logger.exception("Q&A stream failed")
            yield json.dumps({"type": "error", "detail": "Không thể hoàn tất câu trả lời. Vui lòng thử lại."},
                             ensure_ascii=False) + "\n"
            return
        finally:
            timings["request_total_seconds"] = time.perf_counter() - started
            log_answer(backend, req, timings, stream=True, mode="evidence", success=success)
        yield json.dumps({"type": "done", "timings": timings, **citation_metadata(found)}, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
