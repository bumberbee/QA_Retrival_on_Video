"""Q&A sessions (conversations): stored per anonymous browser id so a chat can be reopened.

One JSON file per conversation under DATA_DIR/conversations. Each turn keeps the question, the
standalone rewrite used for search, the answer, its sources and the latency, so a reopened chat
shows the same answers and the model can carry on from the last turns plus a running summary.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import storage

_lock = threading.Lock()
TITLE_CHARS = 80


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def folder() -> Path:
    path = storage.DATA_DIR / "conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_client(value: str) -> bool:
    """Anonymous browser id (frontend: crypto.randomUUID kept in localStorage)."""
    return bool(re.fullmatch(r"[A-Za-z0-9-]{8,64}", value or ""))


def valid_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", value or ""))


def _path(conversation_id: str) -> Path:
    if not valid_id(conversation_id):
        raise KeyError(conversation_id)
    return folder() / f"{conversation_id}.json"


def _supa():
    from . import supa
    return supa if storage.cloud() else None


def _read(conversation_id: str) -> dict | None:
    if not valid_id(conversation_id):
        raise KeyError(conversation_id)
    if _supa():
        return _supa().conversation(conversation_id)
    path = _path(conversation_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(conversation: dict) -> None:
    if _supa():
        return _supa().save_conversation(conversation)
    path = _path(conversation["id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(conversation, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def summary_of(conversation: dict) -> dict:
    """What the history list shows (no turns)."""
    return {"id": conversation["id"], "title": conversation["title"], "video_ids": conversation["video_ids"],
            "video_id": conversation.get("video_id"), "turns": len(conversation["turns"]),
            "created_at": conversation["created_at"], "updated_at": conversation["updated_at"]}


def create(client_id: str, video_id: str | None, video_ids: list | None = None, title: str = "") -> dict:
    conversation = {"id": uuid.uuid4().hex, "client_id": client_id, "title": title.strip()[:TITLE_CHARS],
                    "video_id": video_id, "video_ids": list(video_ids or ([video_id] if video_id else [])),
                    "summary": "", "summarized_turns": 0, "turns": [],
                    "created_at": now(), "updated_at": now()}
    with _lock:
        _write(conversation)
    return conversation


def get(conversation_id: str, client_id: str) -> dict | None:
    try:
        conversation = _read(conversation_id)
    except KeyError:
        return None
    if not conversation or conversation.get("client_id") != client_id:
        return None
    return conversation


def _all(client_id: str):
    if _supa():
        yield from _supa().conversations_of(client_id)
        return
    for path in folder().glob("*.json"):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue


def list_for(client_id: str) -> list[dict]:
    items = []
    for conversation in _all(client_id):
        if conversation.get("client_id") == client_id and conversation["turns"]:
            items.append(summary_of(conversation))
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def rename(conversation_id: str, client_id: str, title: str) -> dict | None:
    with _lock:
        conversation = get(conversation_id, client_id)
        if conversation:
            conversation["title"] = title.strip()[:TITLE_CHARS]
            conversation["updated_at"] = now()
            _write(conversation)
    return conversation


def delete(conversation_id: str, client_id: str) -> bool:
    with _lock:
        if not get(conversation_id, client_id):
            return False
        if _supa():
            _supa().delete_conversation(conversation_id)
        else:
            _path(conversation_id).unlink(missing_ok=True)
    return True


def append_turn(conversation_id: str, client_id: str, turn: dict, video_id: str | None = None,
                video_ids: list | None = None) -> dict | None:
    with _lock:
        conversation = get(conversation_id, client_id)
        if not conversation:
            return None
        conversation["turns"].append({**turn, "at": now()})
        if not conversation["title"]:
            conversation["title"] = turn["question"].strip()[:TITLE_CHARS]
        if video_id:
            conversation["video_id"] = video_id
        if video_ids:
            conversation["video_ids"] = list(video_ids)
        conversation["updated_at"] = now()
        _write(conversation)
    return conversation


def set_summary(conversation_id: str, client_id: str, summary: str, covered_turns: int) -> None:
    with _lock:
        conversation = get(conversation_id, client_id)
        if conversation and covered_turns >= conversation.get("summarized_turns", 0):
            conversation["summary"] = summary.strip()
            conversation["summarized_turns"] = covered_turns
            _write(conversation)


def recent(conversation: dict | None, count: int = 3) -> list[dict]:
    return (conversation or {}).get("turns", [])[-count:]
