"""Apple Foundation Models backend client: an Ollama-shaped API
(chat_once / chat_once_json / chat_stream) plus an embedding entry point
(embed_texts_apple), all served by one persistent Swift subprocess (the
`applefm-shim` binary, built by `make apple-shim`) over JSON-lines/stdio.

Selection is via the "apple" sentinel in the existing *_MODEL env vars
(config.is_apple_model); ollama.py and embeddings.py dispatch here so every
other caller stays backend-agnostic. The chat *orchestration* for the fixed
4,096-token window lives in chat_apple.py, not here -- this module is only the
transport and the Ollama-API mirror.

============================== WIRE PROTOCOL ==============================
JSON-lines over stdin/stdout, one JSON object per line, `id`-correlated.
Requests (Python -> shim):

- {"op":"availability","id":N} ->
  {"id":N,"type":"availability","available":bool,"reason":str}
  (from SystemLanguageModel.default.availability; reason strings like
  "Apple Intelligence is not enabled").

- {"op":"respond","id":N,"system":str,"prompt":str,"schema":obj?,"stream":bool?,
   "max_tokens":int?}
  - schema present -> guided generation: the shim builds a
    DynamicGenerationSchema from the JSON-Schema subset (object with
    properties/required, string, integer, number, boolean, enum-of-strings,
    array of scalars). Response `content` is the generated JSON as a string.
  - max_tokens present and > 0 -> caps the reply via
    GenerationOptions.maximumResponseTokens; absent or <= 0 leaves generation
    unbounded (the FM has no reasoning-effort flag, so this is the one per-call
    speed lever). Omitted from the request when None.
  - stream:true -> {"id":N,"type":"chunk","content":str}* then
    {"id":N,"type":"done"}.
  - non-stream -> {"id":N,"type":"done","content":str}.
  - Every respond uses a FRESH LanguageModelSession (callers never want
    history). The shim prewarms the model at startup, so the first respond does
    not pay the cold-load latency.

- {"op":"embed","id":N,"texts":[str,...]} ->
  {"id":N,"type":"embed","vectors":[[f32 x 512],...]}
  NLContextualEmbedding (latin-script model; requestAssets on first use),
  token vectors mean-pooled + L2-normalized; texts over the model's token
  limit are segmented internally and segment vectors mean-pooled.

Errors (shim -> Python):
  {"id":N,"type":"error","message":str,"kind":str} with
  kind:"context_overflow" mapped from exceededContextWindowSize,
  "unavailable" for availability failures, "bad_request" otherwise. The
  process never exits on a per-request error.
==========================================================================

Track B implements the transport below against this frozen contract: a single
persistent, lock-serialized shim subprocess, id-correlated request/response
with respawn-and-retry-once on a crash and a watchdog per-call timeout.
"""

import contextlib
import json
import subprocess
import threading

import config

# The shim binary path, exposed as a module attribute so tests can patch it
# (mirrors the DB_PATH convention in annotate.py/topics.py); the persistent
# subprocess is spawned from this at call time.
SHIM_PATH = config.APPLE_SHIM_PATH

# Mirror ollama._UNSET so the think tri-state passes through unchanged from
# ollama.py's dispatch (applefm ignores it -- Foundation Models has no think
# flag -- but the parameter must exist to accept the forwarded value).
_UNSET = object()

# chat_once_json's default corrective nudge (mirrors ollama._JSON_RETRY_NUDGE).
_JSON_RETRY_NUDGE = (
    "Respond with only the JSON object described in the instructions -- "
    "no prose, no code fences, nothing else."
)

class AppleFMError(RuntimeError):
    """Base for every caller-facing error raised by the Apple FM transport
    (missing shim, unavailable, timeout, crash, bad_request, and -- via the
    subclass below -- context overflow). A RuntimeError subclass so existing
    `except RuntimeError` handlers still catch it, but a distinct type so
    chat.py can catch it precisely and surface a clean, backend-labeled message
    instead of letting a raw traceback escape the turn."""


class ContextOverflowError(AppleFMError):
    """A respond() call whose prompt exceeded the FM's fixed 4,096-token window
    (mapped from the shim's exceededContextWindowSize). An AppleFMError subclass,
    so existing `except RuntimeError` handlers still catch it, but a distinct
    type so chat_apple can catch it precisely and shrink its prompt (compact the
    scratchpad, drop the catalog) rather than let the whole turn crash."""


class GenerationRejectedError(AppleFMError):
    """A per-request, content-driven rejection from the on-device model (mapped
    from the shim's generation_error kind: an unsupported language/locale, a
    guardrail violation, a decoding failure, ...). Recoverable by SKIPPING this
    input -- unlike ContextOverflowError, shrinking the prompt won't help. A
    sibling of ContextOverflowError and still an AppleFMError/RuntimeError, so
    broad handlers keep working, but a distinct type so batch callers (e.g.
    ollama.chat_once_json) can degrade one bad row to a skip instead of aborting
    the whole run, while a fatal bad_request/unavailable still propagates."""


# --- Transport state ---------------------------------------------------------
# One persistent shim subprocess, lazily spawned and shared by every call. The
# single lock serializes writes + the id counter + spawn/respawn, so at most one
# request is ever in flight; that also makes id-correlation trivial (the reply
# we read next is necessarily ours). Never touch _proc / _id_counter without
# holding _lock.
_proc = None
_lock = threading.Lock()
_id_counter = 0

# Role prefixes used when flattening a message list into the shim's single
# `prompt` string (the shim session is single-turn; the labels carry the turn
# structure the model would otherwise get from a chat template).
_ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
    "tool": "Tool",
}

# format="json" has NO direct DynamicGenerationSchema equivalent. Guided
# generation needs known fields, and an object schema with empty `properties`
# pins the whole reply to literally "{}" -- so a permissive guided schema would
# silently return an empty object for every batch caller (annotate/topics).
# Instead JSON mode folds this directive into the system message and generates
# free text, exactly the way Ollama's format="json" leans on the prompt; the
# callers' own system prompts already spell out the JSON shape, and
# chat_once_json parses + nudge-retries. A caller that needs constrained
# decoding to a SPECIFIC shape uses respond() with its own schema (chat_apple's
# route/fill stages do -- those get real guided generation).
_JSON_DIRECTIVE = (
    "Respond with ONLY a single JSON object -- no prose, no explanation, "
    "no code fences."
)


# --- Internal transport control-flow signals --------------------------------
# Kept off the public surface: they only travel from a reader up to _transact,
# which turns them into a respawn+retry or a RuntimeError.
class _ShimTimeout(Exception):
    """The shim went silent past the per-call timeout and was killed."""


class _ShimCrashed(Exception):
    """The shim closed stdout (EOF) before answering -- it crashed."""


# --- Subprocess lifecycle ----------------------------------------------------
def _spawn():
    """Launch the persistent shim subprocess from SHIM_PATH. A missing binary
    is the common first-run failure (the shim is built by `make apple-shim`,
    deliberately kept out of `make install`), so it raises an actionable
    AppleFMError rather than surfacing a bare FileNotFoundError from Popen.
    Caller holds _lock."""
    if not SHIM_PATH.exists():
        raise AppleFMError(
            f"Apple FM shim not found at {SHIM_PATH}. Build it with "
            "'make apple-shim' (needs macOS 26+, Apple Silicon, and Apple "
            "Intelligence enabled in System Settings)."
        )
    # Text mode + line buffering: the protocol is one JSON object per line, so
    # we write a line and flush, and read responses one line at a time.
    return subprocess.Popen(
        [str(SHIM_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )


def _ensure_proc():
    """Return the live shim, spawning it on first use or after a crash (a
    finished proc has a non-None poll()). Caller holds _lock."""
    global _proc
    if _proc is None or _proc.poll() is not None:
        _proc = _spawn()
    return _proc


def _kill_proc():
    """Tear down the current shim and clear the module handle so the next
    request respawns a fresh one. Best-effort throughout -- the proc may already
    be dead. Caller holds _lock."""
    global _proc
    proc, _proc = _proc, None
    if proc is None:
        return
    try:
        proc.kill()
    except (OSError, ValueError):
        pass  # already exited / never fully started
    for stream in (proc.stdin, proc.stdout):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    try:
        proc.wait(timeout=5)  # reap so we don't leak a zombie
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass


# --- Read side (watchdog + id-correlated line reads) -------------------------
@contextlib.contextmanager
def _watchdog(proc, timeout):
    """Kill the shim if it stays silent longer than `timeout` seconds, so the
    blocking stdout readline unblocks at EOF instead of hanging forever. Yields
    an Event set iff the watchdog fired, letting the reader tell a timeout-kill
    from a genuine crash."""
    fired = threading.Event()

    def _fire():
        fired.set()
        try:
            proc.kill()
        except (OSError, ValueError):
            pass

    timer = threading.Timer(timeout, _fire)
    timer.start()
    try:
        yield fired
    finally:
        timer.cancel()


def _read_matching(proc, want_id, fired):
    """Block until the next stdout line whose `id` matches want_id, returning
    the parsed object. Blank lines are skipped; a non-matching id can only be
    stale output from a prior (timed-out) request, so it is skipped too. EOF
    raises _ShimTimeout when our watchdog caused it, else _ShimCrashed."""
    while True:
        line = proc.stdout.readline()
        if line == "":
            if fired.is_set():
                raise _ShimTimeout()
            raise _ShimCrashed()
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("id") == want_id:
            return msg


def _reader_single(proc, want_id, timeout):
    """Reader for a non-stream response: one matching-id line. A shim error
    line becomes a RuntimeError, raised after the watchdog is torn down."""
    with _watchdog(proc, timeout) as fired:
        msg = _read_matching(proc, want_id, fired)
    if msg.get("type") == "error":
        _raise_shim_error(msg)
    return msg


def _reader_stream(proc, want_id, timeout):
    """Reader for a streamed respond: drain every chunk line under one watchdog
    until the matching-id `done`, returning the collected content list. It
    buffers the whole stream rather than yielding lazily on purpose -- see
    _stream_respond for why."""
    chunks = []
    with _watchdog(proc, timeout) as fired:
        while True:
            msg = _read_matching(proc, want_id, fired)
            mtype = msg.get("type")
            if mtype == "chunk":
                chunks.append(msg.get("content", ""))
            elif mtype == "done":
                return chunks
            elif mtype == "error":
                _raise_shim_error(msg)


def _raise_shim_error(resp):
    """Turn a shim {"type":"error"} response into an AppleFMError carrying both
    kind and message. Three kinds are actionable and phrased to stand out:
    'unavailable' (Apple Intelligence off/unsupported -- the reason string IS
    the fix), 'context_overflow' (the fixed 4,096-token window was exceeded),
    and 'generation_error' (a per-request content rejection -- recoverable by
    skipping this input); anything else is a generic, fatal bad_request."""
    kind = resp.get("kind", "error")
    message = resp.get("message", "")
    if kind == "unavailable":
        raise AppleFMError(f"Apple Foundation Models unavailable: {message}")
    if kind == "context_overflow":
        raise ContextOverflowError(
            f"Apple FM context_overflow (4,096-token window exceeded): {message}"
        )
    if kind == "generation_error":
        raise GenerationRejectedError(f"Apple FM rejected the request: {message}")
    raise AppleFMError(f"Apple FM shim error [{kind}]: {message}")


def _transact(payload, timeout, reader):
    """Run one serialized request/response against the persistent shim. _lock is
    held across the write and the entire read (the shim answers one request at a
    time), which also makes the `id` correlation trivial. On a broken-pipe/EOF
    crash the dead shim is torn down and the request retried ONCE against a
    respawn; a second death -- or a timeout -- raises. `reader(proc, id,
    timeout)` consumes stdout and returns the result (or raises _ShimTimeout /
    _ShimCrashed). timeout=None defaults to CHAT_TIMEOUT, mirroring ollama.py."""
    if timeout is None:
        timeout = config.CHAT_TIMEOUT
    global _id_counter
    with _lock:
        for attempt in range(2):
            proc = _ensure_proc()
            _id_counter += 1
            payload["id"] = _id_counter
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                return reader(proc, payload["id"], timeout)
            except _ShimTimeout:
                _kill_proc()
                raise AppleFMError(
                    f"Apple FM shim timed out after {timeout}s (killed)"
                ) from None
            except (BrokenPipeError, _ShimCrashed):
                # The shim died mid-request. Drop it; on the first attempt
                # respawn and retry once -- a second failure is a real fault,
                # not a transient restart.
                _kill_proc()
                if attempt == 0:
                    continue
                raise AppleFMError(
                    "Apple FM shim crashed without answering"
                ) from None


def _flatten(messages):
    """Collapse an Ollama-style message list into the shim's (system, prompt)
    pair. The FIRST system message becomes `system`; every other message is
    rendered as its own role-labeled block ("User: ...", "Assistant: ...")
    joined by blank lines, so a single-turn shim session still sees the whole
    exchange. A second system message (rare) folds in as a labeled turn --
    gemma4's renderer also honors only the first system message (see CLAUDE.md),
    so nothing that runs on Ollama relies on more than one."""
    system = ""
    have_system = False
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system" and not have_system:
            system = content
            have_system = True
            continue
        parts.append(f"{_ROLE_LABELS.get(role, role.capitalize())}: {content}")
    return system, "\n\n".join(parts)


def chat_once(messages, model=None, format=None, timeout=None, think=_UNSET):
    """Ollama-shape chat_once for the apple backend: returns
    {"role": "assistant", "content": str}. The first system message maps to the
    shim `system`; remaining messages are rendered role-labeled into `prompt`.
    format="json" folds a JSON directive into the system and generates free text
    (see _JSON_DIRECTIVE -- an empty guided schema would pin the reply to "{}").
    `think` is accepted and IGNORED -- Foundation Models has no think flag, so it
    is never forwarded."""
    system, prompt = _flatten(messages)
    if format == "json":
        system = f"{system}\n\n{_JSON_DIRECTIVE}".strip() if system else _JSON_DIRECTIVE
    content = respond(system, prompt, timeout=timeout)
    return {"role": "assistant", "content": content}


def chat_once_json(messages, model=None, retry_nudge=_JSON_RETRY_NUDGE,
                   timeout=None, think=_UNSET):
    """chat_once, JSON-mode, parsed -> dict or None, with the same
    retry-nudge-once contract as ollama.chat_once_json."""
    # Mirrors ollama.chat_once_json's retry-nudge-once contract, minus the
    # _think_unsupported degrade: Foundation Models has no think flag, so there
    # is nothing to detect or turn off.
    for _ in range(2):
        reply = chat_once(messages, model=model, format="json",
                          timeout=timeout, think=think)
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
    """Generator yielding {"role", "content"} chunks (a streamed respond).
    `tools` is accepted but unused -- the apple chat path goes through
    chat_apple.py, never native tool calling. `think` accepted and ignored."""
    system, prompt = _flatten(messages)
    for chunk in respond(system, prompt, stream=True, timeout=timeout):
        yield {"role": "assistant", "content": chunk}


def embed_texts_apple(texts, kind="document"):
    """Embed `texts` via the shim's NLContextualEmbedding op; returns a list of
    512-dim float vectors, one per input. Task prefixes are NOT applied for
    apple (nomic-style prefixes are model-specific); `kind` is accepted for a
    signature match with embeddings.embed_texts and is otherwise ignored."""
    resp = _transact({"op": "embed", "texts": list(texts)}, None, _reader_single)
    # Swift's JSON encoder can emit an exact-zero Float as `0` (an int after
    # json.loads), so coerce every component to float -- the sqlite-vec writer
    # and _fit_dim downstream expect a clean list[list[float]].
    return [[float(x) for x in vec] for vec in resp.get("vectors", [])]


# --- Low-level shim seams (used directly by chat_apple.py) -------------------
# chat_apple's staged orchestration needs guided generation against ARBITRARY
# JSON schemas (a route enum, a tool's full input_schema) -- something the
# Ollama-shape chat_once (which only offers format="json") cannot express. So
# the transport exposes the shim `respond` op directly here; chat_once /
# chat_once_json / chat_stream are thin wrappers over it. chat_apple._respond
# delegates to this, and is the single seam evalkit and the tests patch.


def respond(system, prompt, schema=None, stream=False, timeout=None,
            max_tokens=None):
    """The shim `respond` op, low-level. Non-stream (`stream=False`) returns the
    generated `content` as a str -- the guided JSON string when `schema` is
    given, else free text. `stream=True` returns a generator yielding content
    deltas (str) until the shim's `done`. `schema` is the JSON-Schema-subset
    object for guided generation (omitted from the request when None).
    `max_tokens`, when set and > 0, caps the reply length via the FM's
    GenerationOptions.maximumResponseTokens (the one per-call speed lever);
    None/<=0 leaves generation unbounded and is omitted from the request. Raises
    RuntimeError on a shim error (kind preserved in the message; a
    context-overflow surfaces distinctly)."""
    payload = {"op": "respond", "system": system, "prompt": prompt}
    if schema is not None:
        payload["schema"] = schema
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    if stream:
        payload["stream"] = True
        return _stream_respond(payload, timeout)
    resp = _transact(payload, timeout, _reader_single)
    return resp.get("content", "")


def _stream_respond(payload, timeout):
    """Generator backing respond(stream=True). Correctness-over-laziness lock
    discipline: _transact drains every chunk under _lock into a list, and only
    then do we yield from it with the lock released. Yielding lazily *while*
    holding the lock would pin the single serialized shim for as long as the
    caller takes to consume the stream -- and deadlock outright if that caller
    made another shim call mid-iteration. Streamed replies here are single short
    answers, so the buffering costs nothing that matters."""
    yield from _transact(payload, timeout, _reader_stream)


def availability():
    """The shim `availability` op -> {"available": bool, "reason": str} from
    SystemLanguageModel.default.availability. Used for actionable errors when
    the apple backend is selected but Apple Intelligence is off/unsupported."""
    resp = _transact({"op": "availability"}, None, _reader_single)
    return {
        "available": bool(resp.get("available", False)),
        "reason": resp.get("reason", ""),
    }
