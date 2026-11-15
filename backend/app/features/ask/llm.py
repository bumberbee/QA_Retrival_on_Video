"""Answer generation adapter.

``LLM_PROVIDER`` selects the backend:

* ``ollama`` -- a local Ollama server, used to smoke-test the whole app on a
  machine without spending Gemini quota. Slow on CPU.
* ``gemini`` -- the Gemini API (needs ``GEMINI_API_KEY``).

Both share the evidence and citation rules. Image instructions are included only
when Gemini receives slide images; Ollama receives OCR text and speech.
"""

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

from app.shared.models import Transcript
from app.shared.llm import gemini

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

# Values from codex-strings.json, kept here so serving needs no external data file.
REFUSAL = "Mình chưa tìm thấy đủ bằng chứng trong bài giảng để trả lời câu này."
TEXT_ONLY_NOTE = "Bằng chứng gồm chữ trên slide (OCR, có thể sai số hoặc sai dấu) và lời giảng của giáo viên; bạn không xem được ảnh slide."
IMAGE_NOTE = "Với công thức và con số, ưu tiên đọc trên ảnh slide; khi OCR và ảnh khác nhau, theo ảnh."
CITE_RULE = "Trích dẫn bằng mã nguồn trong ngoặc vuông, ví dụ [E2], ngay sau ý dùng đoạn đó; không tự viết mốc thời gian."
OLLAMA_TIMING_FIELDS = ("load_duration", "prompt_eval_count", "prompt_eval_duration",
                        "eval_count", "eval_duration")

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý trả lời câu hỏi về nội dung một video bài giảng, dựa trên "
    "transcript có timestamp dưới đây.\n"
    "Chỉ trả lời dựa trên nội dung transcript; nếu không tìm thấy thông tin liên "
    f"quan hãy nói rõ: \"{REFUSAL}\"\n"
    "Nếu phù hợp, hãy trích dẫn mốc thời gian [mm:ss] liên quan trong câu trả lời."
)


def _finish_timings(timings: dict | None, started: float, first: float | None,
                    *, buffered: bool = False, ollama_data: dict | None = None) -> None:
    """Client times are seconds; Ollama durations retain the server's nanoseconds.

    A buffered API exposes text only with the complete response, so its observed
    TTFT is marked explicitly rather than presented as token-level latency.
    """
    if timings is None:
        return
    timings.update(total_seconds=time.perf_counter() - started,
                   ttft_seconds=first, ttft_is_buffered=buffered)
    if ollama_data is not None:
        timings.update({field: ollama_data.get(field) for field in OLLAMA_TIMING_FIELDS})


class CitationRenderer:
    """Render only complete evidence tokens, retaining possible split-token suffixes."""

    def __init__(self, evidence_map: dict):
        self.evidence_map = evidence_map
        self.cited_ids: list[str] = []
        self.invalid_citations: list[str] = []
        self.pending = ""

    def _replace(self, match) -> str:
        evidence_id = match.group(1)
        source = self.evidence_map.get(evidence_id)
        if source is None:
            if evidence_id not in self.invalid_citations:
                self.invalid_citations.append(evidence_id)
            return ""
        if evidence_id not in self.cited_ids:
            self.cited_ids.append(evidence_id)
        seconds = max(0, int(source["t_start"]))
        stamp = (f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
                 if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}")
        return f"[{stamp}]"

    def feed(self, piece: str, *, final: bool = False) -> str:
        text = self.pending + piece
        self.pending = ""
        if not final:
            suffix = re.search(r"\[(?:E[0-9]*)?$", text)
            if suffix:
                self.pending = text[suffix.start():]
                text = text[:suffix.start()]
        return re.sub(r"\[(E[0-9]+)\]", self._replace, text)

    def metadata(self) -> dict:
        return {"cited_sources": [{"id": eid, **self.evidence_map[eid]} for eid in self.cited_ids],
                "invalid_citations": list(self.invalid_citations)}


class LLMUnavailable(RuntimeError):
    """The configured generation backend is missing or unreachable."""


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()


def describe() -> dict:
    """Small summary for the health endpoint and for logging."""
    name = provider()
    if name == "ollama":
        return {
            "provider": "ollama",
            "model": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            "base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        }
    return {"provider": "gemini", "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest")}


def _format_transcript(transcript: Transcript) -> str:
    lines = []
    for seg in transcript.segments:
        start = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
        lines.append(f"[{start}] {seg.text}")
    return "\n".join(lines)


def _build_prompt(transcript: Transcript, question: str, current_time: float | None) -> str:
    time_hint = ""
    if current_time is not None:
        mm = int(current_time // 60)
        ss = int(current_time % 60)
        time_hint = f"\nNgười dùng đang xem tại thời điểm {mm:02d}:{ss:02d} của video."
    return (
        f"{SYSTEM_INSTRUCTION}\n{time_hint}\n\n"
        f"--- TRANSCRIPT ---\n{_format_transcript(transcript)}\n--- HẾT TRANSCRIPT ---\n\n"
        f"Câu hỏi của người dùng: {question}\n"
    )


def _ollama(prompt: str, max_tokens: int | None = None, temperature: float = 0.2,
            timeout: float | None = None, *, timings: dict | None = None) -> str:
    base = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1800"))
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
            # Ollama defaults to a 4096-token window and truncates silently; the
            # transcript of a 27-minute lecture is already ~3.4k tokens.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            # Keeps a CPU-only smoke test bounded instead of rambling for minutes.
            "num_predict": max_tokens or int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
        },
    }
    request = urllib.request.Request(
        f"{base}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    data = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("message"), dict) and data["message"].get("content"):
                first = time.perf_counter() - started
    except urllib.error.URLError as error:
        raise LLMUnavailable(
            f"Không kết nối được Ollama tại {base} (model {model}): {error.reason}. "
            "Kiểm tra 'ollama serve' đang chạy, hoặc đặt LLM_PROVIDER=gemini trong .env."
        ) from error
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as error:
        raise LLMUnavailable(f"Ollama không trả về kết quả hợp lệ: {error}") from error
    finally:
        _finish_timings(timings, started, first, buffered=True,
                        ollama_data=data if isinstance(data, dict) else {})

    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise LLMUnavailable("Ollama không trả về kết quả hợp lệ.")
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise LLMUnavailable("Ollama trả về câu trả lời rỗng.")
    return text


EVIDENCE_INSTRUCTION = (
    "Bạn là trợ lý trả lời câu hỏi về một video bài giảng tiếng Việt. Bên dưới là BẰNG CHỨNG lấy "
    "từ các phần liên quan của bài giảng.\n"
    "- Chỉ trả lời dựa trên bằng chứng. Nếu bằng chứng không chứa câu trả lời, nói rõ "
    f"\"{REFUSAL}\" và gợi ý phần gần nhất mà bài có giảng nếu có bằng chứng cho phần đó.\n"
    f"- {CITE_RULE}\n"
    "- Khi bằng chứng đã có câu trả lời, trả lời thẳng, không vừa từ chối vừa trả lời.\n"
    "- Trả lời đúng điều được hỏi, 1–3 câu, rõ ràng, bằng tiếng Việt."
)


def _ollama_stream(prompt: str, *, timings: dict | None = None):
    """Yields the answer piece by piece (Ollama /api/chat with stream=true, one JSON per line)."""
    base = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                    "num_predict": int(os.getenv("OLLAMA_MAX_TOKENS", "512"))},
    }
    request = urllib.request.Request(f"{base}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    final_data = {}
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1800"))) as response:
            has_text = False
            for line in response:
                if not line.strip():
                    continue
                data = json.loads(line.decode("utf-8"))
                if not isinstance(data, dict):
                    raise LLMUnavailable("Ollama trả về dòng JSON không hợp lệ.")
                if data.get("error"):
                    raise LLMUnavailable(f"Ollama lỗi: {data['error']}")
                message = data.get("message", {})
                if (not isinstance(message, dict) or not isinstance(message.get("content", ""), str)
                        or ("done" in data and not isinstance(data["done"], bool))):
                    raise LLMUnavailable("Ollama trả về dòng JSON không hợp lệ.")
                piece = message.get("content", "")
                if piece:
                    has_text = True
                    if first is None:
                        first = time.perf_counter() - started
                    yield piece
                if data.get("done") is True:
                    final_data = data
                    if not has_text:
                        raise LLMUnavailable("Ollama trả về câu trả lời rỗng.")
                    return
    except urllib.error.URLError as error:
        raise LLMUnavailable(
            f"Không kết nối được Ollama tại {base} (model {model}): {error.reason}. "
            "Kiểm tra 'ollama serve' đang chạy, hoặc đặt LLM_PROVIDER=gemini trong .env."
        ) from error
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as error:
        raise LLMUnavailable(f"Ollama stream không trả về kết quả hợp lệ: {error}") from error
    finally:
        _finish_timings(timings, started, first, ollama_data=final_data)
    raise LLMUnavailable("Ollama stream kết thúc trước tín hiệu done.")


def _history_block(history: list | None, summary: str = "", answer_chars: int = 600) -> str:
    """The last turns (answers cut to answer_chars) and the running summary of older ones."""
    lines = []
    if summary:
        lines.append(f"Tóm tắt phần trước của cuộc trò chuyện: {summary}")
    for turn in history or []:
        answer = " ".join((turn.get("answer") or "").split())
        lines.append(f"Người dùng: {turn['question']}\nTrợ lý: {answer[:answer_chars]}")
    if not lines:
        return ""
    return ("--- HỘI THOẠI TRƯỚC (chỉ để hiểu câu hỏi, không phải bằng chứng) ---\n"
            + "\n".join(lines) + "\n--- HẾT ---\n\n")


def _evidence_prompt(question: str, current_time: float | None, found: dict, context: str,
                     previous: str | None = None, history: list | None = None, summary: str = "",
                     rewritten: str | None = None, images: bool = False) -> str:
    time_hint = ""
    if current_time is not None:
        time_hint = f"\nNgười dùng đang xem tại {int(current_time // 60):02d}:{int(current_time % 60):02d}."
    weak = ""
    if found.get("best_semantic", 1.0) < 0.85 and not found.get("temporal"):
        # measured on a 20-min lecture: questions it answers score >= 0.89, others 0.81-0.87
        weak = "\nLưu ý: bằng chứng tìm được khớp yếu với câu hỏi, có thể chưa đủ để trả lời."
    follow = "" if history else (
        f"Câu hỏi trước đó của người dùng (câu hỏi dưới đây nối tiếp nó): {previous}\n" if previous else "")
    meant = f"(Nghĩa đầy đủ của câu hỏi: {rewritten})\n" if rewritten and rewritten != question else ""
    image_note = (f"Bằng chứng có ảnh chụp slide đính kèm.\n- {IMAGE_NOTE}" if images else TEXT_ONLY_NOTE)
    return (f"{EVIDENCE_INSTRUCTION}\n{image_note}{time_hint}{weak}\n\n{_history_block(history, summary)}"
            f"--- BẰNG CHỨNG ---\n{context}\n--- HẾT ---\n\n{follow}Câu hỏi: {question}\n{meant}")


def _slide_images(found: dict) -> list:
    parts = []
    for slide in found["slides"][:2]:
        image = slide.get("image")
        if image and image.lower().endswith((".jpg", ".jpeg")):
            try:
                with open(image, "rb") as handle:
                    parts.append({"mime_type": "image/jpeg", "data": handle.read()})
            except OSError:
                pass
    return parts


def answer_with_evidence(question: str, current_time: float | None, found: dict, context: str,
                         previous: str | None = None, *, timings: dict | None = None, **memory) -> str:
    """Answer from retrieved slide evidence (text + up to 2 slide images for Gemini).
    memory: history (last turns), summary, rewritten (see _evidence_prompt)."""
    name = provider()
    images = _slide_images(found) if name != "ollama" else []
    prompt = _evidence_prompt(question, current_time, found, context, previous, images=bool(images), **memory)
    if name == "ollama":
        answer = _ollama(prompt, timings=timings)
    else:
        answer = _gemini_answer(lambda: gemini.complete_parts([prompt] + images), timings)
    renderer = CitationRenderer(found.get("evidence_map", {}))
    answer = renderer.feed(answer, final=True)
    found.update(renderer.metadata())
    return answer


def stream_with_evidence(question: str, current_time: float | None, found: dict, context: str,
                         previous: str | None = None, *, timings: dict | None = None, **memory):
    """Same answer as answer_with_evidence, yielded piece by piece as the model writes it."""
    name = provider()
    images = _slide_images(found) if name != "ollama" else []
    prompt = _evidence_prompt(question, current_time, found, context, previous, images=bool(images), **memory)
    pieces = (_ollama_stream(prompt, timings=timings) if name == "ollama" else
              _gemini_stream([prompt] + images, timings))
    renderer = CitationRenderer(found.get("evidence_map", {}))
    for piece in pieces:
        rendered = renderer.feed(piece)
        if rendered:
            yield rendered
    tail = renderer.feed("", final=True)
    if tail:
        yield tail
    found.update(renderer.metadata())


def _gemini_answer(complete, timings: dict | None) -> str:
    started = time.perf_counter()
    first = None
    try:
        answer = complete()
        if answer:
            first = time.perf_counter() - started
        return answer
    finally:
        _finish_timings(timings, started, first, buffered=True)


def _gemini_stream(parts: list, timings: dict | None):
    started = time.perf_counter()
    first = None
    try:
        for piece in gemini.stream_parts(parts):
            if piece:
                if first is None:
                    first = time.perf_counter() - started
                yield piece
    finally:
        _finish_timings(timings, started, first)


def answer_question(transcript: Transcript, question: str, current_time: float | None = None,
                    *, timings: dict | None = None) -> str:
    prompt = _build_prompt(transcript, question, current_time)
    if provider() == "ollama":
        return _ollama(prompt, timings=timings)
    return _gemini_answer(lambda: gemini.complete(prompt), timings)


# ---- multi-turn: rewrite follow-ups (LLM4CS-style few-shot) and summarize older turns ----

REWRITE_INSTRUCTION = (
    "Viết lại CÂU HỎI CUỐI thành một câu hỏi độc lập, đọc riêng vẫn đủ nghĩa, dựa vào hội thoại phía trên "
    "(thay \"nó\", \"cái đó\", \"còn … thì sao\" bằng đúng đối tượng được nhắc). Giữ nguyên ngôn ngữ của câu "
    "hỏi. Nếu câu hỏi đã đủ nghĩa thì chép lại nguyên văn. Chỉ trả về đúng một câu hỏi, không giải thích.\n\n"
    "Ví dụ:\n"
    "Người dùng: Liên hợp quốc ra đời trong hoàn cảnh nào?\nTrợ lý: Sau Hội nghị Yalta tháng 2/1945 …\n"
    "Câu hỏi cuối: Còn ASEAN thì sao?\nViết lại: ASEAN ra đời trong hoàn cảnh nào?\n\n"
    "Người dùng: Công thức Bayes là gì?\nTrợ lý: P(Bk|A) = P(A|Bk)P(Bk) / tổng P(A|Bi)P(Bi) …\n"
    "Câu hỏi cuối: Cho ví dụ về nó\nViết lại: Cho ví dụ áp dụng công thức Bayes.\n\n"
    "User: What is an eigenvector?\nAssistant: A vector v with Tv = λv …\n"
    "Câu hỏi cuối: And how do I find them?\nViết lại: How do I find the eigenvectors of a linear map?\n"
)


def _complete(prompt: str, max_tokens: int, temperature: float, timeout: float) -> str:
    if provider() == "ollama":
        return _ollama(prompt, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    # The installed SDK supports both generation limits and per-request timeouts.
    gemini._ensure_configured()
    model = gemini.genai.GenerativeModel(gemini._MODEL_NAME)
    response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens,
                                                                "temperature": temperature},
                                      request_options={"timeout": timeout})
    return response.text or "Xin lỗi, tui không tạo được câu trả lời."


def _clip(turn: dict, chars: int) -> str:
    return " ".join((turn.get("answer") or "").split())[:chars]


def rewrite_question(question: str, history: list, timeout: float | None = None) -> str | None:
    """Standalone form of a follow-up question, or None when the model fails or is too slow
    (the caller then searches with the previous question appended, as before)."""
    if not history:
        return None
    turns = "\n".join(f"Người dùng: {turn['question']}\nTrợ lý: {_clip(turn, 300)}" for turn in history[-2:])
    prompt = f"{REWRITE_INSTRUCTION}\n--- HỘI THOẠI ---\n{turns}\nCâu hỏi cuối: {question}\nViết lại:"
    try:
        text = _complete(prompt, max_tokens=60, temperature=0.0,
                         timeout=timeout or float(os.getenv("REWRITE_TIMEOUT_SECONDS", "40")))
    except Exception:   # Ollama down or slow, Gemini quota: fall back, never fail the answer
        return None
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    line = line.removeprefix("Viết lại:").strip().strip('"“”').strip()
    return line if 4 <= len(line) <= 300 else None


SUMMARY_INSTRUCTION = (
    "Tóm tắt cuộc trò chuyện hỏi đáp về bài giảng dưới đây trong tối đa 4 câu: người dùng đã hỏi những gì, "
    "các ý chính đã trả lời (giữ mốc thời gian [mm:ss] quan trọng). Viết bằng ngôn ngữ của cuộc trò chuyện. "
    "Chỉ trả về bản tóm tắt.\n"
)


def summarize(previous_summary: str, turns: list, timeout: float | None = None) -> str | None:
    """Recursive summary: the previous summary plus the turns it does not cover yet."""
    body = "\n".join(f"Người dùng: {turn['question']}\nTrợ lý: {_clip(turn, 500)}" for turn in turns)
    prompt = (SUMMARY_INSTRUCTION + "\n"
              + (f"Tóm tắt trước đó: {previous_summary}\n\n" if previous_summary else "")
              + f"--- CÁC LƯỢT MỚI ---\n{body}\n--- HẾT ---\nTóm tắt:")
    try:
        text = _complete(prompt, max_tokens=200, temperature=0.2,
                         timeout=timeout or float(os.getenv("SUMMARY_TIMEOUT_SECONDS", "300")))
    except Exception:
        return None
    return text.strip() or None
