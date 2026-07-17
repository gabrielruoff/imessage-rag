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


class OllamaResponseError(RuntimeError):
    """Error chunk streamed back by the Ollama server (e.g. a tool-call the
    model emitted that Ollama could not parse). Often transient: re-sending
    the same request lets the model re-sample."""


def chat_once(messages, model=None, format=None, timeout=None):
    """POST a non-streaming request to the chat endpoint and return the
    response's `message` dict. format='json' turns on Ollama JSON mode (the
    batch annotation/labeling path). Raises RuntimeError on an error body."""
    conn = http.client.HTTPConnection(
        OLLAMA_HOST, OLLAMA_PORT, timeout=timeout or CHAT_TIMEOUT
    )
    body = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    if format:
        body["format"] = format
    if CHAT_THINK is not None:
        body["think"] = CHAT_THINK
    try:
        conn.request(
            "POST", OLLAMA_CHAT_PATH, body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status, raw = response.status, response.read()
    finally:
        conn.close()
    # Check the status before parsing: a non-200 body is often HTML or empty,
    # and json.loads would raise JSONDecodeError -- burying the actual cause
    # (wrong port, model not pulled) under a parse error.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        snippet = raw[:200].decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"Ollama returned unparseable HTTP {status} from {OLLAMA_CHAT_PATH}: "
            f"{snippet or '(empty body)'}"
        ) from None
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    if status != 200:
        raise RuntimeError(f"Ollama HTTP {status}: {data}")
    return data.get("message", {})


_JSON_RETRY_NUDGE = (
    "Respond with only the JSON object described in the instructions -- "
    "no prose, no code fences, nothing else."
)


def chat_once_json(messages, model=None, retry_nudge=_JSON_RETRY_NUDGE):
    """chat_once(format='json') -> parsed dict, retrying once with a
    corrective nudge on a malformed/non-object response; None if the retry
    also fails (the caller skips the row -- batch callers pick it up again on
    their next run). Callers with a task-specific output shape pass their own
    retry_nudge so the correction names the exact JSON expected."""
    for _ in range(2):
        # chat_once is resolved as a module global at call time (the tests'
        # patch seam).
        reply = chat_once(messages, model=model, format="json")
        content = reply.get("content") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": retry_nudge},
        ]
    return None


def chat_stream(messages, tools=None, model=None):
    """POST a streaming request to the chat endpoint (OLLAMA_CHAT_PATH) and
    yield each chunk's `message` dict as it arrives. Raises OllamaResponseError
    if the server streams an error chunk.

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one. model=None uses
    CHAT_MODEL from config."""
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=CHAT_TIMEOUT)
    body = {
        "model": model or CHAT_MODEL,
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
                raise OllamaResponseError(f"Ollama error: {chunk['error']}")
            yield chunk.get("message", {})
            if chunk.get("done"):
                break
    finally:
        conn.close()
