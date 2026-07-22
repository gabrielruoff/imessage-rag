"""Staged chat orchestration for the Apple Foundation Models backend.

The FM window is a fixed 4,096 tokens and chat.py's default overhead (its
system prompt + the 21 tool schemas) is ~8.3K tokens on its own -- native tool
calling simply does not fit. So the apple chat path does NOT hand the model the
whole tool loop at once; it stages many small, fresh, single-purpose sessions
per user turn and carries state between them in Python, never in the model's
context:

  ROUTE  -> pick the single next tool (or "answer") from a compact catalog
  FILL   -> generate that tool's arguments against its (simplified) schema
  EXEC   -> run the query_tools function locally, reusing chat.py's registry
  DIGEST -> compress the (often oversized) result down to question-relevant
            facts, slicing at row boundaries so a result far bigger than the
            window is still read whole -- the RAW result never re-enters a prompt
  ANSWER -> answer from the accumulated facts alone, streamed like chat.py

Every stage is one guided-generation call through the single _respond seam
(applefm.respond); the tests and evalkit patch exactly that. Multi-turn REPL
history is kept as a rolling one-line-per-turn digest, never the raw transcript
(the window cannot hold it). chat.run_turn dispatches here when
config.is_apple_model(model) -- see chat.py.
"""

import json

import applefm
import config
import tool_schemas

# Patchable hook (mirrors the DB_PATH/SHIM_PATH module-attribute convention):
# when evalkit sets this to a callable, it fires at each tool EXECUTE with the
# executed (name, args) so the sweep can record round trips on a path that never
# calls chat.ollama_chat_stream. None in normal use -- zero overhead.
_recorder = None

# --- Stage prompts (each a fresh session's system message) -------------------
ROUTE_SYSTEM = (
    "You route one step of an assistant that answers questions about the user's "
    "iMessage history. Given the question, the facts gathered so far, and a "
    "catalog of retrieval tools, choose the SINGLE best next action: the name of "
    "the one tool to call next, or \"answer\" when the gathered facts already "
    "answer the question. Resolve a named person or group to ids first "
    "(search_contacts_by_name, then list_chats). Never guess message content -- "
    "gather it with a tool. Reply with only the JSON object."
)
ARG_SYSTEM = (
    "You produce the arguments for one tool call of an iMessage-history "
    "assistant. Given the question and the facts gathered so far, fill in the "
    "tool's JSON arguments. Reuse ids (chat_id, handle_id, contact_id) that "
    "already appear in the facts rather than inventing them; leave out any "
    "argument you do not need. Reply with only the JSON object."
)
DIGEST_SYSTEM = (
    "You extract the facts relevant to a question from one chunk of a tool "
    "result. Reply with a terse plain-text list of only the specific facts that "
    "bear on the question -- names, ids, dates, counts, exact message text. Omit "
    "everything else. If nothing in the chunk is relevant, reply exactly "
    "\"nothing relevant\". Add no commentary."
)
ANSWER_SYSTEM = (
    "You answer a question about the user's iMessage history using ONLY the "
    "facts gathered below. Do not invent messages, names, dates, or counts. If "
    "the facts are incomplete or do not answer the question, say so plainly "
    "instead of guessing. Cite dates when relevant, and be concise."
)
_NOTHING_RELEVANT = "nothing relevant"


# --- Tool-schema simplification (done once, at catalog-build time) -----------
# The shim's DynamicGenerationSchema understands only a subset of JSON Schema
# (see applefm.py's wire protocol): an object with scalar / enum-of-strings /
# array-of-scalar properties. Union `type` lists (the _ID_OR_LIST_SCHEMA idiom)
# have no equivalent, so each collapses to the single supported type a caller is
# most likely to want; a string carries the widest range of inputs and an array
# the narrowest, giving the plan's collapse rules: ["integer","array"]->integer,
# ["string","array"]->string, ["string","integer"]->string.
_TYPE_PREFERENCE = ("string", "integer", "number", "boolean", "array")


def _pick_type(type_field):
    """Collapse a JSON-Schema `type` (a bare string, or a union list) to the one
    type the shim supports, preferring the widest-input type present; None when
    nothing in it is supported (the property is then dropped)."""
    if isinstance(type_field, list):
        for cand in _TYPE_PREFERENCE:
            if cand in type_field:
                return cand
        return None
    return type_field if type_field in _TYPE_PREFERENCE else None


def _simplify_property(prop):
    """Map one JSON-Schema property to the shim's supported subset, or None to
    drop it. A union collapses to a single type (_pick_type); an enum survives
    only as an enum-of-strings on a string; an array keeps a scalar `items`
    type (array-of-object / array-of-array are unsupported and dropped)."""
    if not isinstance(prop, dict):
        return None
    t = _pick_type(prop.get("type"))
    if t is None:
        return None
    if t == "array":
        item_t = _pick_type((prop.get("items") or {}).get("type"))
        if item_t is None or item_t == "array":
            return None
        return {"type": "array", "items": {"type": item_t}}
    out = {"type": t}
    enum = prop.get("enum")
    if t == "string" and isinstance(enum, list) and all(isinstance(e, str) for e in enum):
        out["enum"] = list(enum)
    return out


def _simplify_schema(input_schema):
    """Map a tool's JSON-Schema input into the subset the shim's guided
    generation understands: an object of scalar / enum-of-strings /
    array-of-scalar properties plus a `required` list narrowed to the kept
    properties. Unsupported or union-typed bits are simplified or dropped rather
    than raised -- a dropped property just leaves the arg-filler unconstrained on
    it, never a failed request."""
    props = {}
    for name, prop in (input_schema.get("properties") or {}).items():
        simplified = _simplify_property(prop)
        if simplified is not None:
            props[name] = simplified
    out = {"type": "object", "properties": props}
    required = [r for r in input_schema.get("required", []) if r in props]
    if required:
        out["required"] = required
    return out


_CATALOG_LINE_MAX = 150


def _summary(description):
    """A compact catalog line for the router: the tool description's first
    sentence (split on '. '), hard-capped at _CATALOG_LINE_MAX chars. The full
    descriptions -- and even some first "sentences", which run 300+ chars when a
    parenthetical swallows the first period -- would eat the router's slice of
    the 4,096-token window (21 lines share it with the question and scratchpad)."""
    first = description.split(". ")[0].strip()
    return first if len(first) <= _CATALOG_LINE_MAX else first[:_CATALOG_LINE_MAX - 1] + "…"


# Built once at import: the compact catalog + a simplified schema per tool, plus
# the route enum. Keyed by the same names chat.TOOL_FUNCTIONS dispatches on.
_TOOL_SPECS = {
    t["name"]: {
        "summary": _summary(t["description"]),
        "schema": _simplify_schema(t["input_schema"]),
    }
    for t in tool_schemas.TOOL_DEFINITIONS
}
_TOOL_NAMES = list(_TOOL_SPECS)
CATALOG = "\n".join(f"- {name}: {spec['summary']}" for name, spec in _TOOL_SPECS.items())
_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [*_TOOL_NAMES, "answer"]},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}


# --- The one model seam ------------------------------------------------------
def _respond(system, prompt, schema=None, stream=False):
    """The single model entry point -- every ROUTE/FILL/DIGEST/ANSWER call goes
    through here, so the tests and evalkit patch exactly this one function.
    Delegates to the FM transport; importing applefm at module load does not
    spawn the shim (that happens lazily on the first real call)."""
    return applefm.respond(system, prompt, schema=schema, stream=stream,
                           timeout=config.CHAT_TIMEOUT)


def _respond_json(system, prompt, schema):
    """A guided _respond call whose `content` is a JSON string (the ROUTE/FILL
    stages), parsed to a dict. A missing or malformed reply yields None so the
    caller can fall back rather than raise."""
    raw = _respond(system, prompt, schema=schema)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


# --- Prompt-context helpers --------------------------------------------------
_SUMMARY_Q_CHARS = 200
_SUMMARY_A_CHARS = 300


def _clip(text, n):
    """Truncate to n chars with an ellipsis -- keeps the rolling summary short."""
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _current_question(messages):
    """The current turn's question: the content of the last user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _prior_turns_summary(messages):
    """Compress every COMPLETED prior turn (a user question and the assistant
    reply that answered it) into one short line apiece. The FM window cannot
    hold a raw multi-turn transcript, so earlier turns enter later prompts only
    through this rolling digest; the final user message (the current question) is
    excluded and handled separately."""
    last_user = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    lines = []
    pending_q = None
    for i, m in enumerate(messages):
        if i == last_user:
            break
        role = m.get("role")
        if role == "user":
            pending_q = _clip(m.get("content", ""), _SUMMARY_Q_CHARS)
        elif role == "assistant" and pending_q is not None:
            answer = _clip(m.get("content", ""), _SUMMARY_A_CHARS)
            lines.append(f"- Asked: {pending_q}\n  Answered: {answer}")
            pending_q = None
    return "\n".join(lines)


def _context_block(question, summary, scratchpad):
    """The shared prompt body for the route/fill/answer stages: the current
    question, the rolling summary of earlier turns (if any), and the facts
    gathered so far this turn. Only DIGESTED facts appear here -- never a raw
    tool result -- so the body stays inside the FM window."""
    parts = []
    if summary:
        parts.append("Earlier in this conversation:\n" + summary)
    parts.append("Question: " + question)
    if scratchpad:
        parts.append("Facts gathered so far:\n" + "\n".join(scratchpad))
    else:
        parts.append("Facts gathered so far: (none yet)")
    return "\n\n".join(parts)


# The scratchpad shares the 4,096-token window with the ~3K-char catalog and the
# stage system prompt, so its accumulated facts must stay well under the window
# or a later ROUTE/FILL/ANSWER prompt overflows. A single tool result is already
# capped (APPLE_TOOL_RESULT_MAX_CHARS), but several verbatim results stack up
# past the window -- so after each tool the scratchpad is compacted back under
# this budget. Sized so ROUTE (catalog + system + question + this) fits with
# headroom.
_SCRATCHPAD_BUDGET = 5000


def _tail_under_budget(scratchpad):
    """Model-free fallback compaction: keep the most recent facts whose combined
    length fits _SCRATCHPAD_BUDGET (the newest are likeliest relevant to the next
    step). Always keeps at least the last fact."""
    kept, total = [], 0
    for fact in reversed(scratchpad):
        if kept and total + len(fact) > _SCRATCHPAD_BUDGET:
            break
        kept.append(fact)
        total += len(fact)
    kept.reverse()
    return kept


def _compact(question, scratchpad):
    """Keep the running scratchpad within _SCRATCHPAD_BUDGET so no later prompt
    overflows the FM window. Under budget, it is returned untouched; over budget,
    one digest pass reduces every fact to just what bears on the question, and
    that single compaction (clipped to the budget) replaces the facts. If the
    digest overflows or comes back empty, fall back to keeping the most recent
    facts under budget -- never lose everything, never crash."""
    if sum(len(f) for f in scratchpad) <= _SCRATCHPAD_BUDGET:
        return scratchpad
    joined = "\n".join(scratchpad)
    try:
        reply = _respond(
            DIGEST_SYSTEM,
            f"Extract facts relevant to this question:\n{question}\n\nFrom:\n{joined}",
        )
    except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
        return _tail_under_budget(scratchpad)
    reply = (reply or "").strip()
    if reply and reply.lower() != _NOTHING_RELEVANT:
        return [_clip(reply, _SCRATCHPAD_BUDGET)]
    return _tail_under_budget(scratchpad)


# --- The staged loop ---------------------------------------------------------
def _route(question, summary, scratchpad):
    """ROUTE stage: pick the single next tool by name, or "answer" when the
    gathered facts already suffice. Guided against _ROUTE_SCHEMA, so a well-formed
    reply is one of the 21 tool names or "answer"; a missing/garbled reply falls
    back to "answer" (a premature but honest answer beats a crash or a loop)."""
    prompt = (
        _context_block(question, summary, scratchpad)
        + "\n\nAvailable tools:\n" + CATALOG
        + "\n\nPick the ONE tool to call next, or \"answer\" if the facts above "
          "are already enough to answer the question."
    )
    try:
        data = _respond_json(ROUTE_SYSTEM, prompt, _ROUTE_SCHEMA)
    except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
        return "answer"  # can't fit/generate the router step -- answer from facts
    action = (data or {}).get("action")
    return action if isinstance(action, str) else "answer"


def _fill_args(name, question, summary, scratchpad):
    """FILL stage: generate `name`'s arguments as JSON, guided against its
    simplified input schema, reusing ids already in the scratchpad. A malformed
    reply yields {} (the tool then runs with its defaults, or errors into a note
    the next digest can read)."""
    spec = _TOOL_SPECS[name]
    prompt = (
        _context_block(question, summary, scratchpad)
        + f"\n\nProduce the JSON arguments for `{name}` -- {spec['summary']}."
    )
    try:
        data = _respond_json(ARG_SYSTEM, prompt, spec["schema"])
    except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
        return {}  # fall back to the tool's own argument defaults
    return data if isinstance(data, dict) else {}


def _execute(chat, name, args, seen_calls):
    """EXEC stage: run the query_tools function behind `name` locally, reusing
    chat.py's registry. Mirrors chat.run_turn's guards -- an exact-repeat
    (name, args) call is short-circuited to a note rather than re-run, an unknown
    name and any exception both become a digestible {"error": ...}. The executed
    call is echoed to chat.console and offered to the _recorder hook."""
    call_key = (name, json.dumps(args, sort_keys=True, default=str))
    fn = chat.TOOL_FUNCTIONS.get(name)
    if call_key in seen_calls:
        result = {
            "duplicate_call": True,
            "note": f"already called {name} with these arguments this turn",
        }
    elif fn is None:
        result = {"error": f"unknown tool {name}"}
    else:
        seen_calls.add(call_key)
        if _recorder is not None:
            _recorder(name, args)
        try:
            result = fn(**args)
        except Exception as e:
            result = {"error": str(e)}
    if isinstance(result, list):
        count = len(result)
    elif isinstance(result, dict) and isinstance(result.get("rows"), list):
        count = len(result["rows"])
    else:
        count = "ok"
    # markup=False: the literal "[tool call]" label and any brackets in the
    # arguments must print verbatim -- rich's markup parser would eat them.
    chat.console.print(f"  [tool call] {name}({args}) -> {count}", style="dim cyan",
                       markup=False)
    return result


def _char_windows(text, budget):
    """Last-resort splitter for a shape with no rows to cut on: fixed-size
    character windows over the raw JSON. The digester reads them as text, so a
    mid-token cut costs a little fidelity but never drops content."""
    return [text[i:i + budget] for i in range(0, len(text), budget)] or [text]


def _slice_result(result, result_json, budget):
    """Cut an oversized result into <=budget JSON windows at ROW boundaries so
    every row is digested and none is split mid-object. List and {rows:[...]}
    envelope results slice on their rows; any other shape -- or a single row that
    alone busts the budget -- falls back to character windows over the raw JSON."""
    envelope = (
        result if isinstance(result, dict) and isinstance(result.get("rows"), list)
        else None
    )
    rows = envelope["rows"] if envelope is not None else result
    if not (isinstance(rows, list) and rows):
        return _char_windows(result_json, budget)
    windows = []
    cur = []
    for row in rows:
        row_json = json.dumps(row, default=str)
        if len(row_json) > budget:
            # One row bigger than the whole window: flush what we have, then
            # char-slice this row so it still gets read.
            if cur:
                windows.append(json.dumps(cur, default=str))
                cur = []
            windows.extend(_char_windows(row_json, budget))
            continue
        cur.append(row)
        if len(json.dumps(cur, default=str)) > budget:
            cur.pop()
            windows.append(json.dumps(cur, default=str))
            cur = [row]
    if cur:
        windows.append(json.dumps(cur, default=str))
    return windows


def _digest(question, name, result):
    """DIGEST stage: reduce one tool result to question-relevant facts that fit
    the FM window. A result whose JSON already fits APPLE_TOOL_RESULT_MAX_CHARS
    is kept verbatim as the fact; a larger one is sliced at row boundaries into
    <=budget windows, each digested by its own fresh session, and the returned
    facts merged. The raw oversized JSON never enters a route/fill/answer
    prompt -- only these digested facts do."""
    result_json = json.dumps(result, default=str)
    budget = config.APPLE_TOOL_RESULT_MAX_CHARS
    if len(result_json) <= budget:
        return result_json
    facts = []
    # _slice_result walks an envelope's `rows` only, so its scalar metadata
    # (total_matching / returned / order / truncated) would be lost on exactly
    # the large/truncated results where "how many matched" matters most. Keep it
    # verbatim as the first fact -- a handful of scalars, safely under budget --
    # so the ANSWER stage can still report the true total and the capped flag.
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        meta = {k: v for k, v in result.items() if k != "rows"}
        if meta:
            facts.append(json.dumps(meta, default=str))
    for window in _slice_result(result, result_json, budget):
        try:
            reply = _respond(
                DIGEST_SYSTEM,
                f"Extract facts relevant to this question:\n{question}\n\nFrom:\n{window}",
            )
        except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
            continue  # a stubborn/rejected window -- skip it, don't crash the turn
        reply = (reply or "").strip()
        if reply and reply.lower() != _NOTHING_RELEVANT:
            facts.append(reply)
    return " ".join(facts)


def _model_label(model):
    """The label above the streamed reply, mirroring chat.ollama_chat_stream's
    "model>" header -- the apple sentinel ("apple"/"apple:variant") reads fine."""
    return model or config.CHAT_MODEL


def _render_stream(chat, stream, show_model_header, model):
    """Render the streamed answer to chat.console like chat.ollama_chat_stream: a
    live markdown render on a terminal, one faithful render at the end on a
    non-terminal (a file/pipe, where Live emits only its first wrapped line).
    Returns the assembled reply string."""
    from rich.live import Live
    from rich.markdown import Markdown

    label = _model_label(model)
    content = ""
    live = None
    use_live = chat.console.is_terminal
    try:
        for piece in stream:
            if not piece:
                continue
            content += piece
            if use_live:
                if live is None:
                    if show_model_header:
                        chat.console.print(f"[bold]{label}>[/bold]")
                    live = Live(Markdown(""), console=chat.console, refresh_per_second=12)
                    live.start()
                live.update(Markdown(content))
    finally:
        if live is not None:
            live.stop()
    if content and not use_live:
        if show_model_header:
            chat.console.print(f"[bold]{label}>[/bold]")
        chat.console.print(Markdown(content))
    return content


def _answer(chat, question, summary, scratchpad, show_model_header, model):
    """ANSWER stage: answer from the gathered facts ALONE, streamed to
    chat.console. No tools and no raw results reach here -- if the facts are
    incomplete the answerer is told to say so plainly rather than guess. On a
    window overflow (facts still too big for the answer prompt) the scratchpad is
    compacted and the summary dropped, then finally the question is answered
    alone -- a degraded answer beats a crash on the final step."""
    attempts = [
        (summary, scratchpad),
        ("", _compact(question, scratchpad)),
        ("", []),
    ]
    for summ, sp in attempts:
        try:
            prompt = _context_block(question, summ, sp)
            stream = _respond(ANSWER_SYSTEM, prompt, stream=True)
            return _render_stream(chat, stream, show_model_header, model)
        except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
            continue
    return ""


def run_turn(messages, show_model_header=True, model=None):
    """Run one user turn to a final answer via staged FM sessions, mirroring
    chat.run_turn's signature and contract: the final reply is returned AND
    appended to `messages` (so the REPL, `chat.py -p`, and evalkit all see it).
    Earlier turns in `messages` fold into a compressed summary; the last user
    message is the current question. Up to config.MAX_TOOL_ITERATIONS
    route/fill/exec/digest cycles run before the loop falls through to answer."""
    import chat  # lazy: chat.py imports this module lazily in its dispatch

    question = _current_question(messages)
    summary = _prior_turns_summary(messages)
    scratchpad = []
    seen_calls = set()
    for _ in range(config.MAX_TOOL_ITERATIONS):
        action = _route(question, summary, scratchpad)
        if action not in _TOOL_SPECS:  # "answer", or any off-catalog value
            break
        args = _fill_args(action, question, summary, scratchpad)
        result = _execute(chat, action, args, seen_calls)
        fact = _digest(question, action, result)
        if fact:
            scratchpad.append(f"[{action}] {fact}")
            scratchpad = _compact(question, scratchpad)
    reply = _answer(chat, question, summary, scratchpad, show_model_header, model)
    messages.append({"role": "assistant", "content": reply})
    return reply
