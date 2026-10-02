"""Answer generation adapter.

``LLM_PROVIDER`` selects the backend:

* ``ollama`` -- a local Ollama server, used to smoke-test the whole app on a
  machine without spending Gemini quota. Slow on CPU.
* ``gemini`` -- the Gemini API (needs ``GEMINI_API_KEY``).

Both receive the exact same prompt, so a local run exercises the same code path
as the hosted one; only the transport differs.
"""

import json
import os
import urllib.error
import urllib.request

from ..models import Transcript
from . import gemini

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2:7b"

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


def _ollama(prompt: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1800"))
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": 0.2,
            # Ollama defaults to a 4096-token window and truncates silently; the
            # transcript of a 27-minute lecture is already ~3.4k tokens.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            # Keeps a CPU-only smoke test bounded instead of rambling for minutes.
            "num_predict": int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
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
    except (TimeoutError, json.JSONDecodeError) as error:
        raise LLMUnavailable(f"Ollama không trả về kết quả hợp lệ: {error}") from error

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
    "- Trích dẫn mốc thời gian [mm:ss] của slide hoặc lời giảng dùng để trả lời.\n"
    "- Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt."
)


def answer_with_evidence(question: str, current_time: float | None, found: dict, context: str) -> str:
    """Answer from retrieved slide evidence (text + up to 2 slide images for Gemini)."""
    time_hint = ""
    if current_time is not None:
        time_hint = f"\nNgười dùng đang xem tại {int(current_time // 60):02d}:{int(current_time % 60):02d}."
    weak = ""
    if found.get("best_semantic", 1.0) < 0.85 and not found.get("temporal"):
        # measured on a 20-min lecture: questions it answers score >= 0.89, others 0.81-0.87
        weak = "\nLưu ý: bằng chứng tìm được khớp yếu với câu hỏi, rất có thể bài giảng không đề cập."
    prompt = (f"{EVIDENCE_INSTRUCTION}{time_hint}{weak}\n\n--- BẰNG CHỨNG ---\n{context}\n--- HẾT ---\n\n"
              f"Câu hỏi: {question}\n")
    if provider() == "ollama":
        return _ollama(prompt)
    parts: list = [prompt]
    for slide in found["slides"][:2]:
        image = slide.get("image")
        if image and image.lower().endswith((".jpg", ".jpeg")):
            try:
                with open(image, "rb") as handle:
                    parts.append({"mime_type": "image/jpeg", "data": handle.read()})
            except OSError:
                pass
    return gemini.complete_parts(parts)


def answer_question(transcript: Transcript, question: str, current_time: float | None = None) -> str:
    prompt = _build_prompt(transcript, question, current_time)
    if provider() == "ollama":
        return _ollama(prompt)
    return gemini.complete(prompt)
