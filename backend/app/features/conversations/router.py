"""Q&A history of one browser (anonymous id in the X-Client-Id header): list, open, rename, delete."""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.features.conversations import sessions

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def client(x_client_id: str | None) -> str:
    if not x_client_id or not sessions.valid_client(x_client_id):
        raise HTTPException(400, "Thiếu hoặc sai header X-Client-Id")
    return x_client_id


@router.get("")
def list_conversations(x_client_id: str | None = Header(default=None)):
    return sessions.list_for(client(x_client_id))


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, x_client_id: str | None = Header(default=None)):
    conversation = sessions.get(conversation_id, client(x_client_id))
    if not conversation:
        raise HTTPException(404, "Không tìm thấy phiên hỏi đáp")
    return {**sessions.summary_of(conversation), "summary": conversation.get("summary", ""),
            "turns": conversation["turns"]}


@router.patch("/{conversation_id}")
def rename_conversation(conversation_id: str, body: Rename, x_client_id: str | None = Header(default=None)):
    conversation = sessions.rename(conversation_id, client(x_client_id), body.title)
    if not conversation:
        raise HTTPException(404, "Không tìm thấy phiên hỏi đáp")
    return sessions.summary_of(conversation)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, x_client_id: str | None = Header(default=None)):
    if not sessions.delete(conversation_id, client(x_client_id)):
        raise HTTPException(404, "Không tìm thấy phiên hỏi đáp")
