"""Staged chat orchestration for the Apple Foundation Models backend.

The FM window is a fixed 4,096 tokens and chat.py's default overhead (its
system prompt + every tool schema) is roughly double the window on its own --
native tool calling simply does not fit. So the apple chat path does NOT hand the model the
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
import re

import applefm
import chat_floor  # feature: wellbeing -- the single-sourced floor (arm/strip/dedupe/fold)
import config
import tool_schemas
import wellbeing  # feature: wellbeing -- dependency-free (stats + re), no config/Ollama pulled

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
    "assistant. Most arguments are OPTIONAL filters -- read each argument's "
    "description. Include an id (chat_id, handle_id, contact_id) or a date "
    "(date_from, date_to) ONLY if its value appears in the question or the facts "
    "gathered so far; NEVER invent one. To count or analyze the whole corpus, "
    "pass no filters at all. Reply with only the JSON object."
)
DIGEST_SYSTEM = (
    "You extract the facts relevant to a question from one chunk of a tool "
    "result. Reply with a terse plain-text list of only the specific facts that "
    "bear on the question -- names, ids, dates, counts, exact message text. "
    "Preserve every id with its field name exactly as written (chat_id=42, "
    "handle_id=5, contact_id=168) so it can be reused -- never merge or relabel "
    "them. Omit everything else. If nothing in the chunk is relevant, reply "
    "exactly \"nothing relevant\". Add no commentary."
)
ANSWER_SYSTEM = (
    "You answer a question about the user's iMessage history using ONLY the "
    "facts gathered below. Do not invent messages, names, dates, or counts. When "
    "a fact states a number (a count, total, id, or date), use it EXACTLY as "
    "written -- never recompute, sum, round, or estimate it. If the facts are "
    "incomplete or do not answer the question, say so plainly instead of "
    "guessing. Cite dates when relevant, and be concise."
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

# Per-property descriptions are the shim's generation guides (main.swift threads
# them into DynamicGenerationSchema.Property) and ride out-of-band as the
# `schema` payload, so they cost nothing against the 4,096-token PROMPT budget --
# but a cap keeps the constrained-decoding grammar small and keeps the first,
# load-bearing sentence (e.g. the id/date-format and include-section guidance).
_ARG_DESC_MAX = 200


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
    type (array-of-object / array-of-array are unsupported and dropped). The
    property's `description` is preserved (capped at _ARG_DESC_MAX) -- the shim
    forwards it to the model as a generation guide, and without it the arg-filler
    is blind to what each id/date/enum field means (the classic fabricated-arg
    failure)."""
    if not isinstance(prop, dict):
        return None
    t = _pick_type(prop.get("type"))
    if t is None:
        return None
    if t == "array":
        item_t = _pick_type((prop.get("items") or {}).get("type"))
        if item_t is None or item_t == "array":
            return None
        out = {"type": "array", "items": {"type": item_t}}
    else:
        out = {"type": t}
        enum = prop.get("enum")
        if t == "string" and isinstance(enum, list) and all(isinstance(e, str) for e in enum):
            out["enum"] = list(enum)
    desc = prop.get("description")
    if isinstance(desc, str) and desc.strip():
        out["description"] = desc.strip()[:_ARG_DESC_MAX]
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
    the 4,096-token window (one line per tool shares it with the question and
    scratchpad)."""
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
def _respond(system, prompt, schema=None, stream=False, max_tokens=None):
    """The single model entry point -- every ROUTE/FILL/DIGEST/ANSWER call goes
    through here, so the tests and evalkit patch exactly this one function.
    Delegates to the FM transport; importing applefm at module load does not
    spawn the shim (that happens lazily on the first real call). `max_tokens`
    (per-stage APPLE_*_MAX_TOKENS; 0/None = uncapped) caps the reply length."""
    return applefm.respond(system, prompt, schema=schema, stream=stream,
                           timeout=config.CHAT_TIMEOUT, max_tokens=max_tokens)


def _respond_json(system, prompt, schema, max_tokens=None):
    """A guided _respond call whose `content` is a JSON string (the ROUTE/FILL
    stages), parsed to a dict. A missing or malformed reply yields None so the
    caller can fall back rather than raise."""
    raw = _respond(system, prompt, schema=schema, max_tokens=max_tokens)
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
            max_tokens=config.APPLE_DIGEST_MAX_TOKENS,
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
        data = _respond_json(ROUTE_SYSTEM, prompt, _ROUTE_SCHEMA,
                             max_tokens=config.APPLE_ROUTE_MAX_TOKENS)
    except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
        return "answer"  # can't fit/generate the router step -- answer from facts
    action = (data or {}).get("action")
    return action if isinstance(action, str) else "answer"


# Id / date filter args the small FM model fabricates most (see _ground_args).
_ID_ARG_KEYS = ("chat_id", "handle_id", "contact_id")
_DATE_ARG_KEYS = ("date_from", "date_to")


def _ground_args(args, question, summary, scratchpad):
    """Code enforcement of ARG_SYSTEM's "never invent an id or date" rule: drop
    any id/date filter whose value is not grounded in the facts gathered so far
    (or the question). The FM arg-filler routinely fabricates precise-looking
    chat_id/handle_id/contact_id values and date windows that were never
    retrieved, zeroing out the query; prompting alone does not stop it.

    An id member survives only on a WORD-BOUNDARY match against the corpus -- a
    substring test ("1" in "1234") matched everything and let bad ids through. A
    date survives only if its leading 4-digit year appears in the corpus. Every
    other argument (is_from_me, limit, granularity, include, query, ...) passes
    untouched. On an empty scratchpad nothing grounds, so the first call runs
    filter-free (the whole-corpus aggregate) and name->id resolution must precede
    any filtering -- exactly the intended flow."""
    if not isinstance(args, dict):
        return {}
    corpus = "\n".join(scratchpad) + "\n" + (summary or "") + "\n" + (question or "")
    cleaned = {}
    for key, value in args.items():
        if key in _ID_ARG_KEYS:
            members = value if isinstance(value, list) else [value]
            kept = [x for x in members
                    if re.search(rf"\b{re.escape(str(x))}\b", corpus)]
            if not kept:
                continue  # every id fabricated -- drop the filter
            cleaned[key] = kept if isinstance(value, list) else kept[0]
        elif key in _DATE_ARG_KEYS:
            year = re.match(r"\s*(\d{4})", str(value))
            if year and year.group(1) in corpus:
                cleaned[key] = value
            # else: fabricated date window -- drop it
        else:
            cleaned[key] = value
    return cleaned


def _fill_args(name, question, summary, scratchpad):
    """FILL stage: generate `name`'s arguments as JSON, guided against its
    simplified input schema, reusing ids already in the scratchpad. A malformed
    reply yields {} (the tool then runs with its defaults, or errors into a note
    the next digest can read). Filled args pass through _ground_args, which drops
    any id/date filter the model fabricated rather than took from the facts."""
    spec = _TOOL_SPECS[name]
    prompt = (
        _context_block(question, summary, scratchpad)
        + f"\n\nProduce the JSON arguments for `{name}` -- {spec['summary']}."
    )
    try:
        data = _respond_json(ARG_SYSTEM, prompt, spec["schema"],
                             max_tokens=config.APPLE_FILL_MAX_TOKENS)
    except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
        return {}  # fall back to the tool's own argument defaults
    data = data if isinstance(data, dict) else {}
    return _ground_args(data, question, summary, scratchpad)


def _execute(chat, name, args, seen_calls, floor):
    """EXEC stage: run the query_tools function behind `name` locally, reusing
    chat.py's registry. Mirrors chat.run_turn's guards -- an exact-repeat
    (name, args) call is short-circuited to a note rather than re-run, an unknown
    name and any exception both become a digestible {"error": ...}. The executed
    call is echoed to chat.console and offered to the _recorder hook.

    `floor` is the per-turn chat_floor.TurnFloor (see run_turn); this stage applies
    chat.run_turn's code-level wellbeing floor before the result is digested via
    the SAME shared helper the Ollama path uses -- it arms the whole-reply sanitize
    on get_language_trajectory, collects any reserved `_epilogue`, and strips both
    `_epilogue` and the duplicate `support_block` from the result the model sees
    (code-delivery directives, never data to narrate)."""
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
    # feature: wellbeing (parity with chat.run_turn) -- arm sanitize + strip the
    # epilogue/support_block via the single-sourced floor, BEFORE digest.
    result = floor.note_tool(name, result)
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
                max_tokens=config.APPLE_DIGEST_MAX_TOKENS,
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


def _render_stream(chat, stream, show_model_header, model, sanitize=False):
    """Render the streamed answer to chat.console like chat.ollama_chat_stream: a
    live markdown render on a terminal, one faithful render at the end on a
    non-terminal (a file/pipe, where Live emits only its first wrapped line).
    Returns the assembled reply string.

    On a wellbeing turn (`sanitize=True`) the whole assembled reply is run
    through wellbeing.sanitize_condition_names before it ever reaches the screen
    -- so partial, unsanitized content is never live-rendered, matching the
    invariant chat.ollama_chat_stream holds (chat.py:350-366)."""
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
                # Hold every frame on a wellbeing turn until the whole reply has
                # been sanitized below; never live-render partial content.
                if not sanitize:
                    live.update(Markdown(content))
        if sanitize and content:
            content, _redacted = wellbeing.sanitize_condition_names(content)
            if use_live and live is not None:
                live.update(Markdown(content))
    finally:
        if live is not None:
            live.stop()
    if content and not use_live:
        if show_model_header:
            chat.console.print(f"[bold]{label}>[/bold]")
        chat.console.print(Markdown(content))
    return content


def _answer(chat, question, summary, scratchpad, show_model_header, model,
            sanitize=False):
    """ANSWER stage: answer from the gathered facts ALONE, streamed to
    chat.console. No tools and no raw results reach here -- if the facts are
    incomplete the answerer is told to say so plainly rather than guess. On a
    window overflow (facts still too big for the answer prompt) the scratchpad is
    compacted and the summary dropped, then finally the question is answered
    alone -- a degraded answer beats a crash on the final step. `sanitize` (a
    wellbeing turn) is honored on EVERY attempt, degraded ones included."""
    attempts = [
        (summary, scratchpad),
        ("", _compact(question, scratchpad)),
        ("", []),
    ]
    for summ, sp in attempts:
        try:
            prompt = _context_block(question, summ, sp)
            stream = _respond(ANSWER_SYSTEM, prompt, stream=True,
                              max_tokens=config.APPLE_ANSWER_MAX_TOKENS)
            return _render_stream(chat, stream, show_model_header, model,
                                  sanitize=sanitize)
        except (applefm.ContextOverflowError, applefm.GenerationRejectedError):
            continue
    return ""


def _emit_epilogues(chat, reply, floor):
    """Print the collected epilogue block verbatim after the reply and fold it
    into the returned string (so `chat.py -p` and evalkit transcripts capture the
    code-emitted support text), mirroring chat.run_turn's epilogue handling. A
    no-op when nothing was collected, so non-wellbeing turns are byte-for-byte
    unchanged. Dedupe + folding live in the shared chat_floor.TurnFloor."""
    block = floor.epilogue_block()
    if block:
        chat.console.print()
        chat.console.print(block, markup=False)
    return floor.fold_epilogues(reply)


# --- Deterministic name->id pre-resolution (before the staged loop) ----------
# Question words / imperative verbs that a capitalized-token scan picks up but
# that must never be looked up as a contact.
_NAME_STOPWORDS = frozenset({
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "did", "do", "does", "is", "are", "was", "were", "has", "have", "had",
    "draft", "find", "break", "based", "help", "looking", "if", "the", "a",
    "an", "and", "or", "but", "my", "me", "i", "you", "we", "us", "show",
    "tell", "give", "list", "count", "compare",
})
_NAME_CANDIDATE_RE = re.compile(r"[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)*")
_MAX_PRERESOLVE_NAMES = 4
_MAX_PRERESOLVE_CONTACTS = 2
_MAX_PRERESOLVE_CHATS = 3


def _extract_candidate_names(question):
    """Pull the capitalized name-shaped spans out of the question (single or
    multi-word), dropping the sentence-initial question words / verbs a naive
    scan also catches. Deterministic and model-free -- a false candidate simply
    resolves to no contact and is skipped. Order-preserving, de-duplicated."""
    out, seen = [], set()
    for span in _NAME_CANDIDATE_RE.findall(question or ""):
        # Trim leading/trailing question-words the scan swept in ("And Kyle" ->
        # "Kyle", "Who" -> "") so a real name is kept clean and dedups with its
        # bare form; an all-stopword span trims to nothing and is dropped.
        words = span.split()
        while words and words[0].lower() in _NAME_STOPWORDS:
            words.pop(0)
        while words and words[-1].lower() in _NAME_STOPWORDS:
            words.pop()
        name = " ".join(words)
        if len(name) < 2:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out[:_MAX_PRERESOLVE_NAMES]


def _preresolve_entities(chat, question):
    """Resolve person names in the question to labeled id facts BEFORE the staged
    loop, using only the public directory tools. This hands the FM model the
    chat_id / handle_id / contact_id it otherwise fabricates (or resolves
    unreliably across several loop turns), each clearly labeled so it cannot
    conflate a contact_id with a chat_id. Best-effort: any failure degrades to no
    seed. Deterministic infra, so it does not consume APPLE_MAX_TOOL_ITERATIONS
    and is not offered to the _recorder (which counts model-initiated calls).

    Returns (facts, resolved): `facts` are the [resolved] scratchpad seed strings;
    `resolved` is the structured list ({"contact_id","chats"}) that lets a forced
    contact-scoped intent be filled deterministically (_primary_contact) rather
    than trusting the FM to reuse the right labeled id."""
    find_contacts = chat.TOOL_FUNCTIONS.get("search_contacts_by_name")
    find_chats = chat.TOOL_FUNCTIONS.get("list_chats")
    if find_contacts is None:
        return [], []
    facts, resolved = [], []
    for name in _extract_candidate_names(question):
        try:
            contacts = find_contacts(name=name, limit=5)
        except Exception:
            continue
        if not isinstance(contacts, list):
            continue
        for c in contacts[:_MAX_PRERESOLVE_CONTACTS]:
            if not isinstance(c, dict) or c.get("id") is None:
                continue
            cid = c["id"]
            handle_ids = [h.get("id") for h in (c.get("handles") or [])
                          if isinstance(h, dict) and h.get("id") is not None]
            label = " ".join(str(c.get(k) or "").strip()
                             for k in ("first_name", "last_name")).strip() or name
            direct_chats = []
            if find_chats is not None:
                try:
                    rows = find_chats(contact_id=cid, limit=50)
                    direct_chats = [r["id"] for r in (rows or [])
                                    if isinstance(r, dict) and not r.get("is_group")
                                    and r.get("id") is not None][:_MAX_PRERESOLVE_CHATS]
                except Exception:
                    direct_chats = []
            fact = (f'"{name}" -> {label}: contact_id={cid}'
                    + (f", handle_id={handle_ids}" if handle_ids else "")
                    + (f", 1:1 chat_id={direct_chats}" if direct_chats else ""))
            facts.append(f"[resolved] {fact}")
            resolved.append({"contact_id": cid, "chats": direct_chats})
            chat.console.print(f"  [resolve] {fact}", style="dim cyan", markup=False)
    return facts, resolved


# Forced-intent tools whose scope is naturally a contact_id -- when the question
# named exactly one clear person, we fill that scope deterministically instead of
# trusting the FM (which conflates contact_id with chat_id even when the seed
# labels both). No-name questions leave `resolved` empty and fall through to the
# normal FILL (whole-corpus for the aggregates).
_CONTACT_SCOPED_INTENT_TOOLS = frozenset({
    "get_milestones", "get_response_dynamics", "get_chat_summary_stats",
    "get_activity_patterns",
})


def _primary_contact(resolved):
    """The single unambiguous contact to scope a forced aggregate by: the one
    resolved contact that has a known 1:1 chat (a real conversation), or the sole
    resolved contact if only one matched. None when ambiguous or empty -- then the
    FM fills the scope itself."""
    if not resolved:
        return None
    with_chat = [r for r in resolved if r.get("chats")]
    if len(with_chat) == 1:
        return with_chat[0]["contact_id"]
    if not with_chat and len(resolved) == 1:
        return resolved[0]["contact_id"]
    return None


# --- Deterministic intent -> tool routing (the loop's forced first action) ---
# The small FM model routes fine once pointed at the right tool but reliably
# MIS-picks the canonical aggregate/milestone tools -- reaching for
# search_conversations / search_messages_sql on "how many messages", "what hour",
# or "when did we first text" (which search_messages_sql cannot answer: it is
# newest-first only). Each rule maps a high-precision question pattern to the
# canonical tool and forces it as iteration 0's action, bypassing _route (also
# saving one model round-trip). Arg-filling still goes through _fill_args, so a
# rule decides only the tool NAME. A miss falls back to model routing -- no harm.
_ROUTE_RULES = [
    (re.compile(r"\bfirst (text|message|messaged|texted)\b", re.I), "get_milestones"),
    (re.compile(r"\bwho (texted|messaged|reached out|contacted) .*first\b", re.I),
     "get_milestones"),
    (re.compile(r"\bwhen did .* first\b", re.I), "get_milestones"),
    (re.compile(r"\b\d[\d,]*(st|nd|rd|th) message\b", re.I), "get_milestones"),
    (re.compile(r"\banniversar", re.I), "get_milestones"),
    (re.compile(r"\blongest (silence|gap|stretch)\b", re.I), "get_milestones"),
    (re.compile(r"\bwho (replies|responds|texts|gets back) .*(faster|fastest|first)\b",
                re.I), "get_response_dynamics"),
    (re.compile(r"\bhow (long|fast|quick).*(reply|respond|get back|text back)\b", re.I),
     "get_response_dynamics"),
    (re.compile(r"\bresponse time\b", re.I), "get_response_dynamics"),
    (re.compile(r"\b(leave|left|leaves) .* on read\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bghost", re.I), "get_response_dynamics"),
    (re.compile(r"\b(what|which) (hour|time of day)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bbusiest (day|hour|time|night)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bwhen do (i|we) (text|message)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bhow often do (i|we)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bhow many (total )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\btotal (number of )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\bmessage count\b", re.I), "get_chat_summary_stats"),
]


def _suggest_first_action(question):
    """The canonical tool for a recognized question intent, or None. First rule
    wins; only fires for a tool the FM model reliably mis-picks. Returns a name in
    _TOOL_SPECS or None (guarded so a rule can never name an unregistered tool)."""
    for pattern, tool in _ROUTE_RULES:
        if tool in _TOOL_SPECS and pattern.search(question or ""):
            return tool
    return None


_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday")


_FIRST_MSG_RE = re.compile(
    r"first (text|message|messaged|texted)|who .*(texted|messaged|reached out) .*first"
    r"|when did .* first", re.I)


def _derive_summary_facts(name, result, question=""):
    """Pre-compute the arithmetic a small model cannot do over a structured
    aggregate result, returning clean plain-text facts to seed alongside the
    digest (a non-empty return also ends the loop -- the exact answer is in hand).
    Cases the FM otherwise fails on: reducing the get_activity_patterns 7x24
    heatmap to the busiest hour/day; comparing the nested get_response_dynamics
    reply-latency medians to say who replies faster; and reading who sent the
    first message from get_milestones (a two-timestamp comparison), gated to an
    actual first/who-first question so nth/anniversary queries are untouched.
    Returns [] for anything it does not special-case."""
    if not isinstance(result, dict):
        return []
    if name == "get_activity_patterns":
        return _derive_activity_facts(result)
    if name == "get_response_dynamics":
        return _derive_response_facts(result)
    if name == "get_milestones" and _FIRST_MSG_RE.search(question or ""):
        return _derive_milestone_facts(result)
    return []


def _derive_milestone_facts(result):
    firsts = result.get("firsts")
    if not isinstance(firsts, dict):
        return []
    first = firsts.get("first") if isinstance(firsts.get("first"), dict) else {}
    fm = firsts.get("first_from_me") if isinstance(firsts.get("first_from_me"), dict) else {}
    ft = firsts.get("first_from_them") if isinstance(firsts.get("first_from_them"), dict) \
        else {}
    first_date = first.get("date")
    fm_date, ft_date = fm.get("date"), ft.get("date")
    if not first_date:
        return []
    # Keep the who-first verdict a single unambiguous clause -- including the two
    # near-identical raw timestamps confused the small model into inverting it.
    day = str(first_date)[:10]
    facts = [f"[derived] The first message between you and the other person was on {day}."]
    if fm_date and ft_date:  # ISO strings compare lexicographically = chronologically
        who = "You" if fm_date <= ft_date else "The other person (not you)"
        facts.append(f"[derived] {who} sent that very first message.")
    return facts


def _derive_activity_facts(result):
    heatmap = result.get("heatmap")
    counts = heatmap.get("counts") if isinstance(heatmap, dict) else None
    if not (isinstance(counts, list) and len(counts) == 7
            and all(isinstance(row, list) and len(row) == 24 for row in counts)):
        return []
    hour_totals = [sum(counts[d][h] for d in range(7)) for h in range(24)]
    day_totals = [sum(counts[d]) for d in range(7)]
    if not any(hour_totals):
        return []
    tz = result.get("timezone", "local")
    busiest_hour = max(range(24), key=lambda h: hour_totals[h])
    busiest_day = max(range(7), key=lambda d: day_totals[d])
    return [
        f"[derived] busiest hour of day = {busiest_hour:02d}:00 {tz} "
        f"({hour_totals[busiest_hour]} messages); busiest weekday = "
        f"{_WEEKDAY_NAMES[busiest_day]} ({day_totals[busiest_day]} messages)"
    ]


def _derive_response_facts(result):
    lat = result.get("reply_latency")
    if not isinstance(lat, dict):
        return []
    me, them = lat.get("me"), lat.get("them")
    if not (isinstance(me, dict) and isinstance(them, dict)):
        return []
    me_med, them_med = me.get("median_seconds"), them.get("median_seconds")
    if not (isinstance(me_med, (int, float)) and isinstance(them_med, (int, float))):
        return []
    if me_med < them_med:
        verdict = "you reply faster (lower median reply time)"
    elif them_med < me_med:
        verdict = "the other person replies faster (lower median reply time)"
    else:
        verdict = "you both reply about equally fast"
    return [
        f"[derived] median reply time: you = {me_med}s, the other person = "
        f"{them_med}s -> {verdict}"
    ]


# --- Loop-progress signals (deterministic backoff) ---------------------------
def _is_dud(result):
    """A result that carries no new information: the _execute dedup note or an
    error dict. Its content must not pollute the scratchpad."""
    return isinstance(result, dict) and bool(
        result.get("duplicate_call") or result.get("error"))


def _has_signal(result):
    """Whether a result advanced the turn -- used to detect a spinning loop. A
    dud, an empty list, or an all-empty envelope (total_matching == 0) is
    no-progress; a populated list, a matching envelope, or an aggregate stats
    dict is progress."""
    if _is_dud(result):
        return False
    if isinstance(result, list):
        return len(result) > 0
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        return (result.get("total_matching") or 0) > 0
    if isinstance(result, dict):
        return True  # an aggregate/stats dict (get_chat_summary_stats, etc.)
    return bool(result)


def run_turn(messages, show_model_header=True, model=None):
    """Run one user turn to a final answer via staged FM sessions, mirroring
    chat.run_turn's signature and contract: the final reply is returned AND
    appended to `messages` (so the REPL, `chat.py -p`, and evalkit all see it).
    Earlier turns in `messages` fold into a compressed summary; the last user
    message is the current question. Named people in the question are resolved to
    labeled id facts up front (_preresolve_entities); a recognized intent forces
    the first tool (_suggest_first_action). Up to config.APPLE_MAX_TOOL_ITERATIONS
    route/fill/exec/digest cycles run, stopping early after
    config.APPLE_NO_PROGRESS_LIMIT consecutive no-signal iterations, before the
    loop falls through to answer."""
    import chat  # lazy: chat.py imports this module lazily in its dispatch

    question = _current_question(messages)
    summary = _prior_turns_summary(messages)
    scratchpad, resolved = _preresolve_entities(chat, question)
    seen_calls = set()
    first_action = _suggest_first_action(question)
    if first_action is not None:
        chat.console.print(f"  [route] intent -> {first_action}", style="dim cyan",
                           markup=False)
    # Per-turn wellbeing floor, carried through _execute (parity with
    # chat.run_turn, same shared helper): arms the whole-reply condition-name
    # scrub once get_language_trajectory has run, and collects code-emitted
    # support blocks to print verbatim after the reply.
    floor = chat_floor.TurnFloor()
    no_progress = 0
    for _ in range(config.APPLE_MAX_TOOL_ITERATIONS):
        forced = first_action is not None
        if forced:  # deterministic first step, no route round-trip
            action, first_action = first_action, None
        else:
            action = _route(question, summary, scratchpad)
        if action not in _TOOL_SPECS:  # "answer", or any off-catalog value
            break
        scope = _primary_contact(resolved) if forced else None
        if forced and action in _CONTACT_SCOPED_INTENT_TOOLS and scope is not None:
            args = {"contact_id": scope}  # fill the scope in code, not via the FM
        else:
            args = _fill_args(action, question, summary, scratchpad)
        result = _execute(chat, action, args, seen_calls, floor)
        derived = []
        if not _is_dud(result):  # never let a duplicate/error note enter the scratchpad
            derived = _derive_summary_facts(action, result, question)
            if derived:
                # The exact answer was computed in code. Seed ONLY the clean
                # derived fact (the raw result's digest is noise that the small
                # model has been observed to answer from, contradicting the
                # verdict) and stop before it makes noisy follow-up calls.
                scratchpad.extend(derived)
                break
            fact = _digest(question, action, result)
            if fact:
                scratchpad.append(f"[{action}] {fact}")
                scratchpad = _compact(question, scratchpad)
        no_progress = 0 if _has_signal(result) else no_progress + 1
        if no_progress >= config.APPLE_NO_PROGRESS_LIMIT:
            break  # the loop is spinning (duplicates/empties) -- answer from facts
    reply = _answer(chat, question, summary, scratchpad, show_model_header, model,
                    sanitize=floor.sanitize)
    # Support resources must survive even a degraded/empty answer, so emit the
    # epilogue regardless of `reply` (fold into the returned string too).
    reply = _emit_epilogues(chat, reply, floor)
    messages.append({"role": "assistant", "content": reply})
    return reply
