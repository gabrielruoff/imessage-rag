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
    is_apple_model,
)


class OllamaResponseError(RuntimeError):
    """Error chunk streamed back by the Ollama server (e.g. a tool-call the
    model emitted that Ollama could not parse). Often transient: re-sending
    the same request lets the model re-sample."""


# think= needs three states, so the default can't be None: None already means
# "omit the flag entirely" (for models that reject a think option), which is
# distinct from False ("send think: false"). _UNSET is the fourth state --
# "caller expressed no preference, fall back to CHAT_THINK".
_UNSET = object()


def _apply_think(body, think):
    """Resolve the think tri-state into the request body, in place."""
    if think is _UNSET:
        think = CHAT_THINK
    if think is not None:
        body["think"] = think


def chat_once(messages, model=None, format=None, timeout=None, think=_UNSET):
    """POST a non-streaming request to the chat endpoint and return the
    response's `message` dict. format='json' turns on Ollama JSON mode (the
    batch annotation/labeling path). Raises RuntimeError on an error body.

    think: True/False for toggle models (qwen3), or an effort level string
    ('low'/'medium'/'high') for reasoning-effort models (gpt-oss); False
    disables a thinking model's hidden reasoning pass for this call only;
    omitted uses CHAT_THINK. Per-call because the batch pipelines want it
    low/off (it is pure latency on a short JSON answer) while interactive
    chat may want it on."""
    # apple sentinel -> Foundation Models backend; imported lazily so
    # Ollama-only users never pay the import. Forward every arg unchanged.
    if is_apple_model(model or CHAT_MODEL):
        import applefm
        return applefm.chat_once(messages, model=model, format=format,
                                 timeout=timeout, think=think)
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
    _apply_think(body, think)
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


# Flipped when the model can't take the `think` value we send -- either it
# rejects the option outright (an error body) or it accepts think=false and
# answers with EMPTY content (gpt-oss's signature). Either way the run
# degrades to no reasoning control once instead of failing every request.
_think_unsupported = False


def chat_once_json(messages, model=None, retry_nudge=_JSON_RETRY_NUDGE,
                   timeout=None, think=_UNSET):
    """chat_once(format='json') -> parsed dict, retrying once with a
    corrective nudge on a malformed/non-object response; None if the retry
    also fails (the caller skips the row -- batch callers pick it up again on
    their next run). Callers with a task-specific output shape pass their own
    retry_nudge so the correction names the exact JSON expected.
    timeout/think forward to chat_once (per-pipeline overrides); models that
    can't take the `think` value we send -- by rejecting it, or by answering
    think=false with empty content -- degrade to no reasoning control for the
    rest of the process instead of failing every request."""
    # apple sentinel -> Foundation Models backend (no _think_unsupported logic
    # there; FM has no think flag). Dispatch before any think handling.
    if is_apple_model(model or CHAT_MODEL):
        import applefm
        try:
            return applefm.chat_once_json(messages, model=model,
                                          retry_nudge=retry_nudge,
                                          timeout=timeout, think=think)
        except (applefm.GenerationRejectedError,
                applefm.ContextOverflowError) as exc:
            # A per-request Apple rejection (unsupported language, guardrail, an
            # over-long single input) degrades to this function's documented None
            # contract -- the batch caller skips the row and retries next run --
            # instead of aborting the whole run. A fatal AppleFMError
            # (unavailable, or a bad_request protocol bug) still propagates.
            print(f"  Apple FM skipped an input ({exc}); retried next run")
            return None
    global _think_unsupported
    for _ in range(2):
        send_think = None if _think_unsupported else think
        # chat_once is resolved as a module global at call time (the tests'
        # patch seam).
        try:
            reply = chat_once(messages, model=model, format="json",
                              timeout=timeout, think=send_think)
        except RuntimeError as exc:
            explicit = send_think is not _UNSET and send_think is not None
            if not (explicit and "think" in str(exc).lower()):
                raise
            _think_unsupported = True
            print(f"  model rejected think={think!r}; continuing without reasoning control")
            reply = chat_once(messages, model=model, format="json",
                              timeout=timeout, think=None)
        content = reply.get("content") or ""
        if not content.strip() and send_think is False:
            # gpt-oss doesn't reject think=false, it honours it and returns
            # EMPTY content -- so the branch above never fires and every
            # caller would instead record a per-row failure for what is one
            # config mismatch. Same degrade, detected from the reply.
            _think_unsupported = True
            print("  model returned empty output for think=false; "
                  "continuing without reasoning control")
            reply = chat_once(messages, model=model, format="json",
                              timeout=timeout, think=None)
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


def chat_stream(messages, tools=None, model=None, timeout=None, think=_UNSET):
    """POST a streaming request to the chat endpoint (OLLAMA_CHAT_PATH) and
    yield each chunk's `message` dict as it arrives. Raises OllamaResponseError
    if the server streams an error chunk.

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one. model=None uses
    CHAT_MODEL from config; think= behaves as in chat_once."""
    # apple sentinel -> Foundation Models backend. This is a generator, so we
    # `yield from` the delegate (a bare `return applefm.chat_stream(...)` would
    # set StopIteration.value and yield the caller NOTHING) then stop.
    if is_apple_model(model or CHAT_MODEL):
        import applefm
        yield from applefm.chat_stream(messages, tools=tools, model=model,
                                       timeout=timeout, think=think)
        return
    conn = http.client.HTTPConnection(
        OLLAMA_HOST, OLLAMA_PORT, timeout=timeout or CHAT_TIMEOUT
    )
    body = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    _apply_think(body, think)
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
