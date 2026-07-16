"""Ollama chat-streaming HTTP (embedding calls live in embeddings.py). Every
knob (host, port, endpoint path, model, timeout) comes from config.py.
"""

import http.client
import json

from config import (
    CHAT_MODEL,
    CHAT_THINK,
    CHAT_TIMEOUT,
    OLLAMA_CHAT_PATH,
    OLLAMA_HOST,
    OLLAMA_PORT,
)


def chat_stream(messages, tools=None):
    """POST a streaming request to the chat endpoint (OLLAMA_CHAT_PATH) and
    yield each chunk's `message` dict as it arrives. Raises RuntimeError if
    the server streams an error chunk.

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one."""
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=CHAT_TIMEOUT)
    body = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    if CHAT_THINK is not None:
        body["think"] = CHAT_THINK
    payload = json.dumps(body).encode()
    try:
        conn.request(
            "POST", OLLAMA_CHAT_PATH, body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        for line in resp:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(f"Ollama error: {chunk['error']}")
            yield chunk.get("message", {})
            if chunk.get("done"):
                break
    finally:
        conn.close()
