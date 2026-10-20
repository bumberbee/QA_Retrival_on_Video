"""Minimal Gemini API adapter.

Prompt construction lives in ``llm``; this module only owns the API call so the
provider switch stays in one place.
"""

import os

import google.generativeai as genai

_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
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


def complete(prompt: str) -> str:
    _ensure_configured()
    model = genai.GenerativeModel(_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text or "Xin lỗi, tui không tạo được câu trả lời."


def complete_parts(parts: list) -> str:
    """Text and images (``{"mime_type": "image/jpeg", "data": bytes}``) in one request."""
    _ensure_configured()
    model = genai.GenerativeModel(_MODEL_NAME)
    response = model.generate_content(parts)
    return response.text or "Xin lỗi, tui không tạo được câu trả lời."


def stream_parts(parts: list):
    """complete_parts, yielded piece by piece."""
    _ensure_configured()
    model = genai.GenerativeModel(_MODEL_NAME)
    for chunk in model.generate_content(parts, stream=True):
        if chunk.text:
            yield chunk.text
