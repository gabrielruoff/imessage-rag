"""POST /api/chat -- the core streaming endpoint.

``chat.run_turn_events`` is a *blocking sync generator* (Ollama sockets, SQLite
reads, tool calls), so it can't run on the event loop directly. We drive it on a
worker thread and bridge each yielded event onto an ``asyncio.Queue`` via
``loop.call_soon_threadsafe``; the async publisher drains the queue and emits one
``ServerSentEvent`` per engine event.

Only ONE generation runs at a time (module-level lock): Ollama serializes anyway
and both DBs take write locks, so concurrent turns would only deadlock or error.
"""

import asyncio
import json
import threading

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

import chat
import config

from . import store
from .models import ChatRequest

router = APIRouter()

# Serialize generation across the whole process -- see module docstring.
_generation_lock = asyncio.Lock()

# Sentinel pushed onto the bridge queue when the worker thread is done.
_DONE = object()


def _drain_remaining(queue: asyncio.Queue):
    """Non-blocking pull of whatever is already on the queue (used during
    cleanup to recover a trailing done event after a client disconnect)."""
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if item is not _DONE:
            yield item


def _title_from(message: str) -> str:
    """First ~60 chars of the opening message, as a conversation title."""
    title = " ".join(message.split())
    return title[:60] if title else "New conversation"


@router.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    # 1. Resolve / create the conversation.
    if req.conversation_id is None:
        conv = store.create_conversation(_title_from(req.message))
        conv_id = conv["id"]
    else:
        conv_id = req.conversation_id

    # 2. Persist the user turn.
    store.add_turn(conv_id, "user", req.message)
    store.touch_conversation(conv_id)

    # 3. Rebuild the stateless messages list: system prompt + every stored turn
    #    (role/content only). The just-added user turn is already included, so we
    #    build straight from the stored turns -- no separate append.
    messages = [{"role": "system", "content": chat._build_system_prompt()}]
    for t in store.get_turns(conv_id):
        messages.append({"role": t["role"], "content": t["content"]})

    resolved_model = req.model or config.CHAT_MODEL

    async def publisher():
        # First frame: tell the client its conversation id + resolved model.
        yield ServerSentEvent(
            event="meta",
            data=json.dumps({"conversation_id": conv_id, "model": resolved_model}),
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            """Iterate the blocking generator, bridging events to the loop."""
            try:
                for event in chat.run_turn_events(messages, model=req.model):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # transport errors other than handled retry
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "message": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        # One generation at a time (Ollama + SQLite write locks forbid overlap).
        await _generation_lock.acquire()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        reply = None
        try:
            while True:
                ev = await queue.get()
                if ev is _DONE:
                    break
                if ev.get("type") == "done":
                    reply = ev.get("reply", "")
                etype = ev["type"]
                data = {k: v for k, v in ev.items() if k != "type"}
                yield ServerSentEvent(event=etype, data=json.dumps(data))
        finally:
            # The worker always terminates (it pushes _DONE from its own finally);
            # join off the event loop so we never release the lock while a
            # generation is still in flight, even if the client disconnected. If
            # we broke early (disconnect), the reply may still be sitting on the
            # queue -- capture it so the assistant turn is persisted regardless.
            try:
                if reply is None:
                    for item in _drain_remaining(queue):
                        if isinstance(item, dict) and item.get("type") == "done":
                            reply = item.get("reply", "")
                await loop.run_in_executor(None, thread.join)
                if reply is None:
                    for item in _drain_remaining(queue):
                        if isinstance(item, dict) and item.get("type") == "done":
                            reply = item.get("reply", "")
            finally:
                _generation_lock.release()
            # 4. Persist the assistant reply (done.reply is the canonical text).
            if reply is not None:
                store.add_turn(conv_id, "assistant", reply)
                store.touch_conversation(conv_id)

    return EventSourceResponse(publisher())
