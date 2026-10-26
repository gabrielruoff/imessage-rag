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
    is_openrouter_model,
)


def _refuse_hosted(model):
    """D2: openrouter/ models are annotate-only. This seam is the one path every
    other pipeline (and its --model flag) goes through, so refusing here keeps
    them all local; there is deliberately no OpenRouter dispatch."""
    if is_openrouter_model(model or CHAT_MODEL):
        raise RuntimeError(
            "openrouter/ models are annotate-only (annotate.py per-chunk tiers); "
            "set a local or apple model for this pipeline")


class OllamaResponseError(RuntimeError):
    """Error chunk streamed back by the Ollama server (e.g. a tool-call the
    model emitted that Ollama could not parse). Often transient: re-sending
    the same request lets the model re-sample."""


class OllamaUnreachable(ConnectionError):
    """No connection to the Ollama server at all -- refused, unresolvable host,
    unreachable network, connect timeout. Raised before anything is sent: a
    config error (server not running, wrong OLLAMA_HOST), not a failed request."""


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
    batch annotation/labeling path). Raises RuntimeError on an error body and
    OllamaUnreachable when no connection could be opened.

    think: True/False for toggle models (qwen3), or an effort level string
    ('low'/'medium'/'high') for reasoning-effort models (gpt-oss); False
    disables a thinking model's hidden reasoning pass for this call only;
    omitted uses CHAT_THINK. Per-call because the batch pipelines want it
    low/off (it is pure latency on a short JSON answer) while interactive
    chat may want it on."""
    _refuse_hosted(model)
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
        try:
            conn.connect()
        except OSError as exc:
            raise OllamaUnreachable(
                f"cannot reach Ollama at {OLLAMA_HOST}:{OLLAMA_PORT}: {exc}") from exc
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


def effective_think(think):
    """The `think` value to actually SEND, given what this process has already
    learned about the model. Returns `think` unchanged until some request has
    proved the model can't take it, and None -- omit the flag -- from then on.

    Public because the degradation belongs to the module, not to
    chat_once_json: a caller that needs chat_once directly (annotate's
    schema-constrained retry does, since chat_once_json hardcodes
    format='json' and so cannot carry a dict schema) would otherwise re-send
    the value the model has already rejected and fail every single time, with
    no seam to notice through. One home for the rule, both callers.

    Tri-state preserving, and it has to be: None stays None (omit), False
    stays False (send think: false), _UNSET stays _UNSET (chat_once resolves
    it to CHAT_THINK). Only the degraded case rewrites anything."""
    return None if _think_unsupported else think


def _extract_reply(reply, messages, retry_nudge):
    """Shared JSON-mode reply handling for chat_once_json here and in applefm
    (which returns the identical {"role","content"} shape): parse the assistant
    content as a JSON object and return (data, next_messages). On success `data`
    is the dict and next_messages is unchanged; on a missing/non-JSON/non-object
    reply `data` is None and next_messages is the conversation extended with the
    assistant turn plus the corrective nudge, ready for a retry pass. Single
    source so a change to the parse/retry shape can't miss the other backend."""
    content = reply.get("content") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data, messages
    return None, messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": retry_nudge},
    ]


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
    _refuse_hosted(model)
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
        send_think = effective_think(think)
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
        data, messages = _extract_reply(reply, messages, retry_nudge)
        if data is not None:
            return data
    return None


def chat_stream(messages, tools=None, model=None, timeout=None, think=_UNSET):
    """POST a streaming request to the chat endpoint (OLLAMA_CHAT_PATH) and
    yield each chunk's `message` dict as it arrives. Raises OllamaResponseError
    if the server streams an error chunk.

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one. model=None uses
    CHAT_MODEL from config; think= behaves as in chat_once."""
    _refuse_hosted(model)
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
