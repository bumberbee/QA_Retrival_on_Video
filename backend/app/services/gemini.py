import os

import google.generativeai as genai

from ..models import Transcript

_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong biến môi trường (.env)")
    genai.configure(api_key=api_key)
    _configured = True


def _format_transcript(transcript: Transcript) -> str:
    lines = []
    for seg in transcript.segments:
        start = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
        lines.append(f"[{start}] {seg.text}")
    return "\n".join(lines)


def answer_question(transcript: Transcript, question: str, current_time: float | None = None) -> str:
    _ensure_configured()
    model = genai.GenerativeModel(_MODEL_NAME)

    context = _format_transcript(transcript)
    time_hint = ""
    if current_time is not None:
        mm = int(current_time // 60)
        ss = int(current_time % 60)
        time_hint = f"\nNgười dùng đang xem tại thời điểm {mm:02d}:{ss:02d} của video."

    prompt = f"""Bạn là trợ lý trả lời câu hỏi về nội dung một video, dựa trên transcript có timestamp dưới đây.
Chỉ trả lời dựa trên nội dung transcript, nếu không tìm thấy thông tin liên quan hãy nói rõ là transcript không đề cập.
Nếu phù hợp, hãy trích dẫn mốc thời gian [mm:ss] liên quan trong câu trả lời.
{time_hint}

--- TRANSCRIPT ---
{context}
--- HẾT TRANSCRIPT ---

Câu hỏi của người dùng: {question}
"""

    response = model.generate_content(prompt)
    return response.text or "Xin lỗi, tui không tạo được câu trả lời."
