"""All Ollama HTTP calls in one place: single-item embedding (query time),
batch embedding (ingest time), and streaming chat. Every knob (host, port,
endpoint paths, models, timeouts) comes from config.py.
"""

import http.client
import json
import threading

from config import (
    CHAT_MODEL,
    CHAT_TIMEOUT,
    EMBED_MODEL,
    OLLAMA_CHAT_PATH,
    OLLAMA_EMBED_PATH,
    OLLAMA_EMBEDDINGS_PATH,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OLLAMA_TIMEOUT,
)

_thread_local = threading.local()


def _persistent_connection() -> http.client.HTTPConnection:
    """One keep-alive connection per worker thread, for high-volume batch calls."""
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_TIMEOUT)
        _thread_local.conn = conn
    return conn


def embed_one(text: str) -> list:
    """Embed a single text via the single-item endpoint (OLLAMA_EMBEDDINGS_PATH).
    Raises RuntimeError on any connection/response failure, suitable for
    surfacing directly as an LLM tool error."""
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_TIMEOUT)
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    try:
        conn.request(
            "POST", OLLAMA_EMBEDDINGS_PATH, body=payload,
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(conn.getresponse().read())
        return body["embedding"]
    except (http.client.HTTPException, OSError, KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ollama embedding request failed: {e}") from e
    finally:
        conn.close()


def embed_batch(texts: list) -> list:
    """Embed a batch of texts in a single request via the batch endpoint
    (OLLAMA_EMBED_PATH). Returns one vector per input. Uses a persistent
    per-thread connection; connection errors propagate to the caller (which
    batches and counts them) after dropping the dead connection."""
    conn = _persistent_connection()
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    try:
        conn.request(
            "POST", OLLAMA_EMBED_PATH, body=payload,
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(conn.getresponse().read())
    except (http.client.HTTPException, OSError):
        # connection may be dead; drop it so the next call on this thread reopens one
        _thread_local.conn = None
        raise
    return body["embeddings"]


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
