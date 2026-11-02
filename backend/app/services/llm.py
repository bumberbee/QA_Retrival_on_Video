"""Answer generation adapter.

``LLM_PROVIDER`` selects the backend:

* ``ollama`` -- a local Ollama server, used to smoke-test the whole app on a
  machine without spending Gemini quota. Slow on CPU.
* ``gemini`` -- the Gemini API (needs ``GEMINI_API_KEY``).

Both receive the exact same prompt, so a local run exercises the same code path
as the hosted one; only the transport differs.
"""

import http.client
import json
import os
import urllib.error
import urllib.request

from ..models import Transcript
from . import gemini

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý trả lời câu hỏi về nội dung một video bài giảng, dựa trên "
    "transcript có timestamp dưới đây.\n"
    "Chỉ trả lời dựa trên nội dung transcript; nếu không tìm thấy thông tin liên "
    "quan hãy nói rõ là transcript không đề cập.\n"
    "Nếu phù hợp, hãy trích dẫn mốc thời gian [mm:ss] liên quan trong câu trả lời."
)


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
            timeout: float | None = None) -> str:
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise LLMUnavailable(
            f"Không kết nối được Ollama tại {base} (model {model}): {error.reason}. "
            "Kiểm tra 'ollama serve' đang chạy, hoặc đặt LLM_PROVIDER=gemini trong .env."
        ) from error
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as error:
        raise LLMUnavailable(f"Ollama không trả về kết quả hợp lệ: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise LLMUnavailable("Ollama không trả về kết quả hợp lệ.")
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise LLMUnavailable("Ollama trả về câu trả lời rỗng.")
    return text


EVIDENCE_INSTRUCTION = (
    "Bạn là trợ lý trả lời câu hỏi về một video bài giảng tiếng Việt. Bên dưới là BẰNG CHỨNG lấy "
    "từ các slide liên quan: chữ trên slide (OCR, có thể sai số hoặc sai dấu), lời giảng của giáo "
    "viên và ảnh chụp các slide đó.\n"
    "- Chỉ trả lời dựa trên bằng chứng. Nếu bằng chứng không chứa câu trả lời, nói rõ "
    "\"Bài giảng không đề cập đến nội dung này\" và gợi ý phần gần nhất mà bài có giảng.\n"
    "- Với công thức và con số, ưu tiên đọc trên ảnh slide; khi OCR và ảnh khác nhau, theo ảnh.\n"
    "- Trích dẫn mốc thời gian [mm:ss] ghi ngay trên đoạn bằng chứng dùng để trả lời. "
    "Mỗi đoạn lời giảng có mốc riêng, có thể khác khoảng thời gian của slide.\n"
    "- Chỉ nói \"không đề cập\" khi bằng chứng thật sự không có câu trả lời; đã có thì trả lời "
    "thẳng, không vừa nói không đề cập vừa trả lời.\n"
    "- Trả lời đúng điều được hỏi, 1–3 câu, rõ ràng, bằng tiếng Việt."
)


def _ollama_stream(prompt: str):
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
                    yield piece
                if data.get("done") is True:
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
                     rewritten: str | None = None) -> str:
    time_hint = ""
    if current_time is not None:
        time_hint = f"\nNgười dùng đang xem tại {int(current_time // 60):02d}:{int(current_time % 60):02d}."
    weak = ""
    if found.get("best_semantic", 1.0) < 0.85 and not found.get("temporal"):
        # measured on a 20-min lecture: questions it answers score >= 0.89, others 0.81-0.87
        weak = "\nLưu ý: bằng chứng tìm được khớp yếu với câu hỏi, rất có thể bài giảng không đề cập."
    follow = "" if history else (
        f"Câu hỏi trước đó của người dùng (câu hỏi dưới đây nối tiếp nó): {previous}\n" if previous else "")
    meant = f"(Nghĩa đầy đủ của câu hỏi: {rewritten})\n" if rewritten and rewritten != question else ""
    return (f"{EVIDENCE_INSTRUCTION}{time_hint}{weak}\n\n{_history_block(history, summary)}"
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
                         previous: str | None = None, **memory) -> str:
    """Answer from retrieved slide evidence (text + up to 2 slide images for Gemini).
    memory: history (last turns), summary, rewritten (see _evidence_prompt)."""
    prompt = _evidence_prompt(question, current_time, found, context, previous, **memory)
    if provider() == "ollama":
        return _ollama(prompt)
    return gemini.complete_parts([prompt] + _slide_images(found))


def stream_with_evidence(question: str, current_time: float | None, found: dict, context: str,
                         previous: str | None = None, **memory):
    """Same answer as answer_with_evidence, yielded piece by piece as the model writes it."""
    prompt = _evidence_prompt(question, current_time, found, context, previous, **memory)
    if provider() == "ollama":
        yield from _ollama_stream(prompt)
    else:
        yield from gemini.stream_parts([prompt] + _slide_images(found))


def answer_question(transcript: Transcript, question: str, current_time: float | None = None) -> str:
    prompt = _build_prompt(transcript, question, current_time)
    if provider() == "ollama":
        return _ollama(prompt)
    return gemini.complete(prompt)


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
