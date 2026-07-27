"""Pydantic request/response models for the API. Kept minimal and typed;
sqlite3.Row rows are mapped via ``model_validate(dict(row))`` at the edges."""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """POST /api/chat body. conversation_id None => start a new conversation;
    model None => the engine default (config.CHAT_MODEL)."""

    conversation_id: Optional[int] = None
    message: str
    model: Optional[str] = None


class NewConversationRequest(BaseModel):
    """POST /api/conversations body -- an optional title for an empty chat."""

    title: Optional[str] = None


class Turn(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str


class Conversation(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationSummary(Conversation):
    """A conversation row in the list view, carrying its turn count."""

    turn_count: int = 0


class ConversationDetail(Conversation):
    """A single conversation plus its full turn history."""

    turns: list[Turn] = []


class Health(BaseModel):
    ok: bool
    backend: str
    model: str
    reachable: bool
