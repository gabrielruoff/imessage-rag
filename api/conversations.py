"""REST endpoints over app.db: list / create / fetch / delete conversations.

Pure CRUD around api.store; no engine calls here. Rows come back as sqlite3.Row
and are mapped into the typed response models at the edge.
"""

from fastapi import APIRouter, HTTPException, Response

from . import store
from .models import (
    ConversationDetail,
    ConversationSummary,
    NewConversationRequest,
    Turn,
)

router = APIRouter()


@router.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations():
    """All conversations, newest activity first, each with its turn_count."""
    return [ConversationSummary.model_validate(dict(r)) for r in store.list_conversations()]


@router.post("/api/conversations", response_model=ConversationSummary)
def create_conversation(req: NewConversationRequest):
    """Create an empty conversation (optional title)."""
    row = store.create_conversation(req.title)
    data = dict(row)
    data["turn_count"] = 0
    return ConversationSummary.model_validate(data)


@router.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int):
    """One conversation plus its full turn history (404 if missing)."""
    row = store.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    data = dict(row)
    data["turns"] = [Turn.model_validate(dict(t)) for t in store.get_turns(conversation_id)]
    return ConversationDetail.model_validate(data)


@router.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int):
    """Delete a conversation (and its turns, via cascade). Idempotent 204."""
    store.delete_conversation(conversation_id)
    return Response(status_code=204)
