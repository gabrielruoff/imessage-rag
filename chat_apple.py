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

# Companion hook: fires AFTER a recorded tool executes, with (name, result) --
# the digest-stripped result the model will see. evalkit sets it to attach the
# raw result to the round trip so its numeric/faithfulness checks have a source
# (the staged path keeps no tool-role messages). None in normal use.
_result_recorder = None

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
    "written -- never recompute, sum, round, or estimate it. "
    "A fact tagged [derived] is an EXACT value computed directly from the user's "
    "own data by trusted code -- it IS the answer: state it plainly and never say "
    "the information is unavailable, missing, or cannot be determined when a "
    "[derived] fact addresses the question. Only if NO fact bears on the question "
    "at all should you say you could not find it. Cite dates when relevant, and be "
    "concise -- when a fact lists many periods or items, summarize the headline "
    "rather than reciting every one."
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
# Enum/scalar NARROWING filters the arg-filler invents that silently mis-scope a
# tool without erroring (a spurious status='dismissed' empties get_loose_threads;
# subject='them' errors get_voiceprint; reaction_type='laugh' scopes it to laughs).
# Kept only when the value's text appears in the question -- so a genuinely-asked
# narrowing ("dismissed nudges", "laugh reactions") survives. NOT applied to
# granularity/ngram/include/is_from_me/limit (legitimate shape args, and a value
# like 2 for "two-word" would not match literally).
_FILTER_ARG_KEYS = ("status", "kind", "reaction_type", "subject", "phase_filter")
# Args a forced-tool error retry may NOT shed: everything above survived
# _ground_args, so it is grounded (id/date) or literally asked for (filter) --
# dropping one changes WHICH question gets answered (a shed contact_id computes
# whole-corpus stats misattributed to the named person). Ungated shape args
# (is_from_me/granularity/ngram/...) pass grounding unchecked, so they are the
# error causes a retry can safely shed.
_RETRY_KEEP_KEYS = frozenset(_ID_ARG_KEYS + _DATE_ARG_KEYS + _FILTER_ARG_KEYS)


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
        elif key in _FILTER_ARG_KEYS:
            # Keep an enum narrowing filter only if its value was actually asked
            # for (case-insensitive substring of the question); else drop it so the
            # tool runs unscoped/default rather than on an invented slice.
            if value is not None and str(value).lower() in (question or "").lower():
                cleaned[key] = value
            # else: invented narrowing filter -- drop it
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
    recorded = False
    if call_key in seen_calls:
        result = {
            "duplicate_call": True,
            "note": f"already called {name} with these arguments this turn",
        }
    elif fn is None:
        result = {"error": f"unknown tool {name}"}
    else:
        seen_calls.add(call_key)
        recorded = True
        if _recorder is not None:
            _recorder(name, args)
        try:
            result = fn(**args)
        except Exception as e:
            result = {"error": str(e)}
    # feature: wellbeing (parity with chat.run_turn) -- arm sanitize + strip the
    # epilogue/support_block via the single-sourced floor, BEFORE digest.
    result = floor.note_tool(name, result)
    # Offer the (wellbeing-stripped) result to the eval capture hook, only for a
    # genuinely executed call (a duplicate/unknown made no round trip to attach
    # to). Mirrors the pre-exec _recorder so the two align 1:1 by order.
    if recorded and _result_recorder is not None:
        _result_recorder(name, result)
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


# Internal provenance tags the scratchpad prefixes onto its facts. The small model
# sometimes echoes a whole fact line verbatim into its answer (the S11.3 leak),
# exposing "[resolved]"/"[derived]"/"[<tool>]" scaffolding to the user -- strip any
# line that is one of those seeded facts before the reply is rendered/returned.
_SCRATCHPAD_TAG_RE = re.compile(r"^\s*[*_`\-\s]*\[(resolved|derived)\]", re.I)


def _strip_scratchpad_tags(text):
    """Drop lines the model copied straight from the seeded scratchpad facts. Only
    removes lines that OPEN with an internal tag, so ordinary prose is untouched;
    collapses the blank gaps the removal leaves."""
    if not text or ("[resolved]" not in text and "[derived]" not in text):
        return text
    kept = [ln for ln in text.splitlines() if not _SCRATCHPAD_TAG_RE.match(ln)]
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


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
        if content:
            stripped = _strip_scratchpad_tags(content)
            if sanitize:
                stripped, _redacted = wellbeing.sanitize_condition_names(stripped)
            if stripped != content:
                content = stripped
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
    # Sentence-initial verbs / auxiliaries / imperatives a capitalized-token scan
    # sweeps in ("Am I texting…" -> "Am", "Show me…" -> "Show"). Each would
    # otherwise fuzzy-resolve to a real contact (Am -> Durham/Graham), poison the
    # scratchpad, and -- with the unresolved-name guard -- suppress a correct
    # forced route. See RC-5 in APPLE_FM_EVAL_REPORT.md.
    "am", "could", "would", "should", "shall", "might",
    "must", "been", "being", "be", "get", "got", "make", "made", "let", "see",
    "look", "there", "here", "any", "some", "no", "not", "over", "about",
    "around", "between", "with", "for", "from", "of", "on", "in", "at", "to",
    "does", "whats", "hows", "wheres", "im", "ive", "id",
})
# Auxiliaries that are ALSO real first names (Will, May, Can). Unconditionally
# stopwording them swallows the contact ("How often does Will text me?" -> no
# candidate), which ALSO disarms the unresolved-name guard (no candidates -> no
# guard) so a forced contact-scoped intent runs unscoped and misattributes
# whole-corpus stats to the person -- the S7.6 confabulation. Capitalization
# disambiguates: mid-sentence these are capitalized only as names, while an
# auxiliary is capitalized only sentence-initially ("Will I ever…"). So they are
# trimmed ONLY as a span's first token at a sentence start.
_POSITIONAL_STOPWORDS = frozenset({"will", "may", "can"})
# Nothing but whitespace / an opening quote/paren between the span and either the
# start of the question or the previous sentence's terminator.
_SENTENCE_INITIAL_RE = re.compile(r"(?:^|[.?!;])[\s\"'“”‘’(\[]*$")
_NAME_CANDIDATE_RE = re.compile(r"[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)*")
_MAX_PRERESOLVE_NAMES = 4
_MAX_PRERESOLVE_CONTACTS = 2
_MAX_PRERESOLVE_CHATS = 3


def _extract_candidate_names(question):
    """Pull the capitalized name-shaped spans out of the question (single or
    multi-word), dropping the sentence-initial question words / verbs a naive
    scan also catches. Deterministic and model-free -- a false candidate simply
    resolves to no contact and is skipped. Order-preserving, de-duplicated."""
    question = question or ""
    out, seen = [], set()
    for m in _NAME_CANDIDATE_RE.finditer(question):
        # Trim leading/trailing question-words the scan swept in ("And Kyle" ->
        # "Kyle", "Who" -> "") so a real name is kept clean and dedups with its
        # bare form; an all-stopword span trims to nothing and is dropped. A
        # name-colliding auxiliary (_POSITIONAL_STOPWORDS) is trimmed only as the
        # span's opening token at a sentence start -- mid-sentence, or after a
        # trimmed opener ("Can Will text back?"), it is a name and is kept; the
        # trailing trim never touches it ("Jenna May" stays whole).
        words = m.group(0).split()
        sentence_initial = bool(_SENTENCE_INITIAL_RE.search(question[:m.start()]))
        lead = 0
        while lead < len(words):
            head = words[lead].lower()
            if head in _NAME_STOPWORDS or (
                    lead == 0 and sentence_initial and head in _POSITIONAL_STOPWORDS):
                lead += 1
            else:
                break
        words = words[lead:]
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


# Forced-intent tools whose scope is a GROUP chat_id. get_group_dynamics /
# get_chat_participant_stats error without one, and _preresolve_entities only
# resolves PEOPLE -- so a group named in the question ("Real Tahoe") went
# unscoped and the tool errored, then the model confabulated from search. These
# get their chat_id filled from a group pre-resolution instead.
_GROUP_SCOPED_INTENT_TOOLS = frozenset({
    "get_group_dynamics", "get_chat_participant_stats",
})


def _preresolve_groups(chat, question):
    """Resolve a group chat named in the question to its chat_id BEFORE the loop,
    via the public list_chats(query=...) directory search -- the group analogue of
    _preresolve_entities. Returns (facts, group_ids): the [resolved] scratchpad
    seed lines and the list of matched group chat_ids (deduped). Best-effort: any
    failure degrades to no seed. Named after the same candidate-name scan people
    use, so a person name that matches no group simply yields nothing."""
    find_chats = chat.TOOL_FUNCTIONS.get("list_chats")
    if find_chats is None:
        return [], []
    facts, group_ids, seen = [], [], set()
    for name in _extract_candidate_names(question):
        try:
            rows = find_chats(query=name, limit=5)
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows[:_MAX_PRERESOLVE_CHATS]:
            if not (isinstance(r, dict) and r.get("is_group") and r.get("id") is not None):
                continue
            # list_chats(query=) matches on PARTICIPANTS too, so a person's name
            # returns every group they are in (mostly unnamed). Require the group's
            # own display_name to match the candidate -- that is the case where the
            # question actually named the GROUP (e.g. "Real Tahoe"), not a member.
            dn = (r.get("display_name") or "").strip()
            nl, dl = name.lower(), dn.lower()
            if not dn or (nl not in dl and dl not in nl):
                continue
            gid = r["id"]
            if gid in seen:
                continue
            seen.add(gid)
            group_ids.append(gid)
            fact = f'"{name}" -> group "{dn}": chat_id={gid}'
            facts.append(f"[resolved] {fact}")
            chat.console.print(f"  [resolve] {fact}", style="dim cyan", markup=False)
    return facts, group_ids


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
    # --- Specific intents FIRST. First rule wins: any specific intent whose
    # phrasing can also match a generic frequency/volume rule below ("how often
    # do i", "when do i text", ...) must sit ABOVE the activity block, or the
    # generic rule steals it ("how often do I ghost people?" is a ghost ask). ---
    (re.compile(r"\bdouble.?text", re.I), "get_response_dynamics"),
    (re.compile(r"\bghost", re.I), "get_response_dynamics"),
    (re.compile(r"\b(leave|left|leaves) .* on read\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bnot replied to a conversation\b", re.I), "get_response_dynamics"),
    # --- Milestones (firsts / nth / anniversaries / gaps / history bounds) ---
    (re.compile(r"\bfirst (text|message|messaged|texted)\b", re.I), "get_milestones"),
    (re.compile(r"\bwho (texted|messaged|reached out|contacted) .*first\b", re.I),
     "get_milestones"),
    (re.compile(r"\bwhen did .* first\b", re.I), "get_milestones"),
    (re.compile(r"\b\d[\d,]*(st|nd|rd|th) message\b", re.I), "get_milestones"),
    (re.compile(r"\banniversar", re.I), "get_milestones"),
    (re.compile(r"\bhow far back\b", re.I), "get_milestones"),
    (re.compile(r"\blast (thing|message) .*(sent|texted|said)\b", re.I), "get_milestones"),
    (re.compile(r"\blongest (silence|gap|stretch)\b", re.I), "get_milestones"),
    (re.compile(r"\bwithout (talking|texting)\b", re.I), "get_milestones"),
    # --- Activity patterns (heatmap / streaks / late-night / volume) ---
    (re.compile(r"\blongest streak\b", re.I), "get_activity_patterns"),
    (re.compile(r"\b(after midnight|3\s?am|late at night)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\b(what|which) (hour|time of day)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bbusiest (\w+ )?(day|hour|time|night)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bwhen (am|do) (i|we) .*(active|text|message)\b", re.I),
     "get_activity_patterns"),
    (re.compile(r"\bwhen do (i|we) (text|message)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bhow often do (i|we)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\btext(ing)? (more|less)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\btext(ed|ing)? (much|a lot) (with|in)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\beach (month|year|week) of \d{4}\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bevolved\b", re.I), "get_activity_patterns"),
    # --- Response dynamics ---
    (re.compile(r"\bwho (replies|responds|texts|gets back) .*(faster|fastest|first)\b",
                re.I), "get_response_dynamics"),
    (re.compile(r"\bquickest to respond\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bhow (long|fast|quick).*(reply|respond|get back|text back)\b", re.I),
     "get_response_dynamics"),
    (re.compile(r"\bresponse time\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bwho (starts|initiates) .*(me or|\bmore\b)", re.I),
     "get_response_dynamics"),
    (re.compile(r"\b(long conversations|quick bursts)\b", re.I), "get_response_dynamics"),
    # --- Per-member (chat_participant_stats) BEFORE reactions/group (more specific) ---
    (re.compile(r"\breactions? per message\b", re.I), "get_chat_participant_stats"),
    (re.compile(r"\bat least \d+ messages\b", re.I), "get_chat_participant_stats"),
    # --- Reactions ---
    (re.compile(r"\btapback", re.I), "get_reaction_stats"),
    (re.compile(r"\bwhat reactions do\b", re.I), "get_reaction_stats"),
    (re.compile(r"\bfunniest\b", re.I), "get_reaction_stats"),
    (re.compile(r"\bfinds? me funn", re.I), "get_reaction_stats"),
    (re.compile(r"\breact(s|ed)? to (my|your|me|them|their|his|her)\b", re.I),
     "get_reaction_stats"),
    (re.compile(r"\bmost reactions\b", re.I), "get_reaction_stats"),
    (re.compile(r"\b(love|loved|like|liked|laugh|laughed) (at )?my (message|text)", re.I),
     "get_reaction_stats"),
    (re.compile(r"\bhow (quickly|fast) .*react\b", re.I), "get_reaction_stats"),
    # --- Group dynamics ---
    (re.compile(r"\bwho (carries|lurks?|dominates)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\b(dominated by|balanced)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\bwho (starts|talks|texts) .* (in|inside)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\btalks? to whom\b", re.I), "get_group_dynamics"),
    (re.compile(r"\bstands? out\b", re.I), "get_group_dynamics"),
    # --- Words / language (trends) ---
    (re.compile(r"\binside jokes?\b", re.I), "get_word_trends"),
    (re.compile(r"\bunique to (me|us)\b", re.I), "get_word_trends"),
    (re.compile(r"\bemoji\b", re.I), "get_word_trends"),
    (re.compile(r"\buse of the (word|phrase)\b", re.I), "get_word_trends"),
    (re.compile(r"\bstart(ed)? saying\b", re.I), "get_word_trends"),
    (re.compile(r"\bhave i ever (said|used|texted)\b", re.I), "get_word_trends"),
    (re.compile(r"\b(laugh a lot|we laugh)\b", re.I), "get_word_trends"),
    (re.compile(r"\bmood of (my|our|the) (chats?|texts?)\b", re.I), "get_word_trends"),
    (re.compile(r"\bemotions\b", re.I), "get_word_trends"),
    (re.compile(r"\bhow similar .* style\b", re.I), "get_word_trends"),
    # --- Word frequency ---
    (re.compile(r"\bmost common words\b", re.I), "get_word_frequency"),
    (re.compile(r"\b(two.word|2.word|two word) phrase", re.I), "get_word_frequency"),
    (re.compile(r"\bphrases? (do )?(i|you) (say|use)\b", re.I), "get_word_frequency"),
    # --- Voiceprint ---
    (re.compile(r"\bvoiceprint\b", re.I), "get_voiceprint"),
    (re.compile(r"\b(describe|what.?s) my .*(texting )?(style|voice)\b", re.I),
     "get_voiceprint"),
    # --- Summary stats (counts / totals / attachments / who-most leaderboard) ---
    (re.compile(r"\bhow many (photos|pictures|images|attachments)\b", re.I),
     "get_chat_summary_stats"),
    (re.compile(r"\bhow many (total )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\btotal (number of )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\bmessage count\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\bwho do (i|you) (text|message|talk to) (the )?most\b", re.I),
     "get_chat_summary_stats"),
    # --- Nudges / loose ends ---
    (re.compile(r"\bowe .*(a )?(reply|text|response)\b", re.I), "get_loose_threads"),
    (re.compile(r"\bleft hanging\b", re.I), "get_loose_threads"),
    (re.compile(r"\bnever (got back|replied|responded)\b", re.I), "get_loose_threads"),
    (re.compile(r"\bnudges?\b", re.I), "get_loose_threads"),
    (re.compile(r"\bunanswered\b", re.I), "get_loose_threads"),
    # --- Drift / reconnection (fadeout gated behind DRIFT_ENABLED) ---
    (re.compile(r"\b(fad(e|ing|ed)|drift(ing|ed)?|growing apart)\b", re.I),
     "get_fadeout_watchlist"),
    (re.compile(r"\b(lost touch|reconnect|break the ice)\b", re.I), "get_loose_threads"),
    # --- Conflict (find_conflicts; apology/draft guarded out in _suggest_first_action) ---
    (re.compile(r"\b(rough patch(es)?|tense stretch(es)?|tensest|falling out)\b", re.I),
     "find_conflicts"),
    (re.compile(r"\b(fight|argument|arguments|conflict)\b", re.I), "find_conflicts"),
    # --- Persona / wellbeing ---
    (re.compile(r"\bsound like (something )?(i|me) would write\b", re.I),
     "check_style_match"),
    (re.compile(r"\bwhat (might|would) .*(text|say|respond) back\b", re.I),
     "simulate_contact"),
    (re.compile(r"\b(mood or mental health|mental health)\b", re.I),
     "get_language_trajectory"),
    (re.compile(r"\b(changed|shifted).*(way (i|she|he|they) (write|text)|sustained)\b",
                re.I), "get_language_trajectory"),
    (re.compile(r"\bsustained (way|change|departure)\b", re.I), "get_language_trajectory"),
]

# Intents whose canonical tool takes an id only a PRIOR call can supply
# (draft_apology/diagnose_conflict need a conflict_id) or that the model already
# routes well on its own (draft_reply needs the inbound message first) -- forcing
# a first action here would either misfire or cut the loop short before the
# follow-up call. When matched, skip deterministic routing and let the model
# stage the chain (it lands the second tool once find_conflicts/retrieval seeds
# the id/context).
_ROUTE_SKIP_RE = re.compile(r"\bapolog|\b(draft|write) (a |me a )?reply", re.I)


def _suggest_first_action(question):
    """The canonical tool for a recognized question intent, or None. First rule
    wins; only fires for a tool the FM model reliably mis-picks. Returns a name in
    _TOOL_SPECS or None (guarded so a rule can never name an unregistered tool --
    a flag-gated tool absent from the catalog simply falls through to model
    routing). Apology/draft-reply intents are skipped (see _ROUTE_SKIP_RE)."""
    question = question or ""
    if _ROUTE_SKIP_RE.search(question):
        return None
    for pattern, tool in _ROUTE_RULES:
        if tool in _TOOL_SPECS and pattern.search(question):
            return tool
    return None


_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday")


_FIRST_MSG_RE = re.compile(
    r"first (text|message|messaged|texted)|who .*(texted|messaged|reached out) .*first"
    r"|when did .* first", re.I)


def _derive_summary_facts(name, result, question=""):
    """Pre-compute the arithmetic/reading a small model cannot do reliably over a
    structured aggregate, returning clean plain-text facts to seed alongside the
    digest. A NON-EMPTY return also ends the loop -- the exact answer is in hand,
    and the raw result's digest is noise the small model has been observed to
    confabulate from (RC-2 in APPLE_FM_EVAL_REPORT.md). Every specialized
    analytics tool routes through a deriver here so the reply narrates a
    code-computed fact rather than inventing one; anything not special-cased (or a
    deriver that cannot read its result) returns [] and falls back to DIGEST."""
    q = question or ""
    if isinstance(result, list):
        # List-shaped results (loose threads, participant stats, and the
        # group_by_chat=True per-chat leaderboard of get_chat_summary_stats).
        if name == "get_loose_threads":
            return _derive_loose_threads_facts(result)
        if name == "get_chat_participant_stats":
            return _derive_participant_facts(result, q)
        if name == "get_chat_summary_stats":
            return _derive_summary_stats_facts(result, q)
        return []
    if not isinstance(result, dict):
        return []
    if result.get("error") or result.get("duplicate_call"):
        return []
    dispatch = {
        "get_activity_patterns": _derive_activity_facts,
        "get_response_dynamics": _derive_response_facts,
        "get_milestones": _derive_milestone_facts,
        "get_reaction_stats": _derive_reaction_facts,
        "get_word_trends": _derive_word_trends_facts,
        "get_word_frequency": _derive_word_frequency_facts,
        "get_group_dynamics": _derive_group_facts,
        "get_voiceprint": _derive_voiceprint_facts,
        "get_chat_summary_stats": _derive_summary_stats_facts,
        "get_fadeout_watchlist": _derive_fadeout_facts,
        "find_conflicts": _derive_conflict_facts,
    }
    fn = dispatch.get(name)
    if fn is None:
        return []
    try:
        return fn(result, q)
    except Exception:
        return []  # a shape surprise must degrade to DIGEST, never crash the turn


def _msg_snippet(m):
    """A one-line 'sender: "text" (day)' from a milestone/message dict."""
    if not isinstance(m, dict):
        return ""
    sender = m.get("sender") or ("You" if m.get("is_from_me") else "the other person")
    text = _clip((m.get("text") or "").strip(), 90)
    day = str(m.get("date") or "")[:10]
    return f'{sender}: "{text}" ({day})' if text else f"{sender} ({day})"


def _derive_milestone_facts(result, question=""):
    """get_milestones has four sub-sections; pick the one the question asks for so
    a 'when did we first / who first / 10,000th message / longest silence / how
    far back' answer quotes the right landmark instead of the model conflating
    two near-identical timestamps (RC-4)."""
    q = question.lower()
    firsts = result.get("firsts") if isinstance(result.get("firsts"), dict) else {}
    facts = []
    # Each specific-intent branch commits: it returns its fact, or [] when the
    # section is absent -- never the first/who-first default, which would answer a
    # different question (a 10,000th-message ask must not get the first-text date).
    # Longest silence / gone without talking.
    if re.search(r"\b(longest|silence|gap|without (talking|texting)|gone without)\b", q):
        sil = result.get("longest_silence")
        if isinstance(sil, dict) and sil.get("available") and sil.get("days") is not None:
            return [f"[derived] The longest silence was {sil['days']} days "
                    f"({str(sil.get('from'))[:10]} to {str(sil.get('to'))[:10]}), "
                    f"broken by {sil.get('broken_by', 'someone')}."]
        return []
    # Nth-message landmark (10,000th / 1,000th ...).
    if re.search(r"\b\d[\d,]*(st|nd|rd|th)\b", q):
        nth = result.get("nth")
        rows = nth.get("landmarks") if isinstance(nth, dict) else None
        if isinstance(rows, list) and rows:
            asked = None
            mnum = re.search(r"(\d[\d,]*)(?:st|nd|rd|th)", q)
            if mnum:
                asked = int(mnum.group(1).replace(",", ""))
            pick = None
            for row in rows:  # exact match if present, else the largest landmark
                if asked is not None and row.get("ordinal") == asked:
                    pick = row
                    break
            pick = pick or rows[-1]
            return [f"[derived] The {pick.get('ordinal')}th message was "
                    f"{_msg_snippet(pick.get('message'))}."]
        return []
    # Anniversaries.
    if "anniversar" in q:
        ann = result.get("anniversaries")
        if isinstance(ann, dict) and ann.get("next_anniversary"):
            return [f"[derived] Next anniversary: {ann['next_anniversary']} "
                    f"({ann.get('years', '?')} years since first contact, "
                    f"{ann.get('days_until', '?')} days away)."]
        return []
    # How far back / earliest.
    if re.search(r"\bhow far back|earliest|history go\b", q):
        first = firsts.get("first")
        if isinstance(first, dict) and first.get("date"):
            return [f"[derived] Message history goes back to {str(first['date'])[:10]} "
                    f"({_msg_snippet(first)})."]
        return []
    # Last message: ONLY firsts.last (the overall most recent message). Milestones
    # has no last-inbound field, and any fallback to a "first_*" receipt would
    # present the oldest message as the newest -- degrade to DIGEST instead.
    if re.search(r"\blast (thing|message)\b", q):
        last = firsts.get("last")
        if not (isinstance(last, dict) and last.get("date")):
            return []
        # "Last thing THEY sent": if the most recent message is mine, the
        # their-side answer is not in this result -- do not present my own
        # message as theirs.
        if last.get("is_from_me") and re.search(r"\b(they|she|he|them)\b", q):
            return []
        return [f"[derived] The last message was {_msg_snippet(last)}."]
    # Default: first / who-first.
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
    facts = [f'[derived] The first message between you and the other person was on {day}'
             f' ({_msg_snippet(first)}).']
    if fm_date and ft_date:  # ISO strings compare lexicographically = chronologically
        who = "You" if fm_date <= ft_date else "The other person (not you)"
        facts.append(f"[derived] {who} sent that very first message.")
    return facts


def _derive_activity_facts(result, question=""):
    """get_activity_patterns carries the heatmap plus late_night / streaks /
    busiest_days / volume_series; pick the section the question asks for."""
    q = question.lower()
    tz = result.get("timezone", "local")
    # Late-night / 3am: name the share and the top late-night contact (RC-3 fix
    # for the 'you text Chungus 10 times' confabulation).
    if re.search(r"\b(after midnight|3\s?am|late at night|at night)\b", q):
        ln = result.get("late_night")
        if isinstance(ln, dict) and ln.get("n") is not None:
            # Skip business/opaque handles (urn:...) so the "who" is a real person.
            top = [c for c in (ln.get("top_contacts") or [])
                   if isinstance(c, dict) and not str(c.get("contact", "")).startswith("urn:")]
            who = ""
            if top:
                names = ", ".join(f"{c.get('contact')} ({c.get('late_night')} late-night "
                                  f"messages)" for c in top[:3])
                who = (f" The people on the other end, most first: {names}.")
            return [f"[derived] You send {ln['n']} messages between "
                    f"{ln.get('start_hour', 0)}:00 and {ln.get('end_hour', 5)}:00 "
                    f"({round((ln.get('share') or 0) * 100, 1)}% of all your messages)."
                    + who]
    # Longest streak of consecutive days.
    if "streak" in q:
        st = result.get("streaks")
        ls = st.get("longest_streak") if isinstance(st, dict) else None
        if isinstance(ls, dict) and ls.get("days") is not None:
            return [f"[derived] The longest streak of consecutive days talking was "
                    f"{ls['days']} days ({ls.get('start')} to {ls.get('end')})."]
    # Single busiest day ever.
    if re.search(r"\bbusiest .*(day|date) (ever|of all)|single busiest\b", q):
        bd = result.get("busiest_days")
        if isinstance(bd, list) and bd and isinstance(bd[0], dict):
            top = bd[0]
            return [f"[derived] Your single busiest day was {top.get('day')} "
                    f"with {top.get('count')} messages."]
    # More/less this year vs last (year volume compare).
    if re.search(r"\b(more|less)\b.*\b(year|month)\b", q):
        vs = result.get("volume_series")
        rows = vs.get("rows") if isinstance(vs, dict) else None
        if isinstance(rows, list) and len(rows) >= 2:
            def tot(r):
                return (r.get("sent") or 0) + (r.get("received") or 0)
            last, prev = rows[-1], rows[-2]
            direction = ("more" if tot(last) > tot(prev)
                         else "less" if tot(last) < tot(prev) else "about the same")
            return [f"[derived] {last.get('period')}: {tot(last)} messages vs "
                    f"{prev.get('period')}: {tot(prev)} -> you are texting {direction}."]
    # Per-period volume series ("how many each month of 2025").
    if re.search(r"\beach (month|year|week)|per (month|year|week)|how many .*each\b", q):
        vs = result.get("volume_series")
        rows = vs.get("rows") if isinstance(vs, dict) else None
        if isinstance(rows, list) and rows:
            grand = sum((r.get("sent") or 0) + (r.get("received") or 0) for r in rows)
            per = "; ".join(
                f"{r.get('period')}: {r.get('sent')} sent / {r.get('received')} received"
                for r in rows if isinstance(r, dict))
            return [f"[derived] Volume by {vs.get('granularity', 'period')} "
                    f"({grand} total): {per}."]
    # Default: busiest hour + weekday from the 7x24 heatmap.
    heatmap = result.get("heatmap")
    counts = heatmap.get("counts") if isinstance(heatmap, dict) else None
    if not (isinstance(counts, list) and len(counts) == 7
            and all(isinstance(row, list) and len(row) == 24 for row in counts)):
        return []
    hour_totals = [sum(counts[d][h] for d in range(7)) for h in range(24)]
    day_totals = [sum(counts[d]) for d in range(7)]
    if not any(hour_totals):
        return []
    busiest_hour = max(range(24), key=lambda h: hour_totals[h])
    busiest_day = max(range(7), key=lambda d: day_totals[d])
    return [
        f"[derived] busiest hour of day = {busiest_hour:02d}:00 {tz} "
        f"({hour_totals[busiest_hour]} messages); busiest weekday = "
        f"{_WEEKDAY_NAMES[busiest_day]} ({day_totals[busiest_day]} messages)"
    ]


# Reply-speed cues that justify the get_response_dynamics reply-latency verdict (who
# replies faster / response time / quickest). Gates the default so a non-speed
# question routed here does not get a spurious latency fact.
_REPLY_SPEED_RE = re.compile(
    r"\b(fast(er|est)?|quick(er|est)?|slow(er|est)?|repl(y|ies|ied)|respond|responds"
    r"|response|get(s)? back|text(s)? back|first)\b", re.I)

# Read-receipt coverage below this share carries no usable "left on read" signal
# (the caveat is the honest answer); at/above it, the measured read->reply
# figures are worth stating -- still never a left-on-read COUNT, because the
# tool records read-and-replied pairs only.
_READ_COVERAGE_MIN = 0.5


def _derive_response_facts(result, question=""):
    q = question.lower()
    # Initiation: who starts more.
    if re.search(r"\b(start|starts|initiat)", q):
        init = result.get("initiation")
        if isinstance(init, dict) and isinstance(init.get("me"), (int, float)) \
                and isinstance(init.get("them"), (int, float)):
            who = ("You" if init["me"] > init["them"]
                   else "The other person" if init["them"] > init["me"] else "Both equally")
            return [f"[derived] Conversation starts: you initiated {init['me']}, the "
                    f"other person {init['them']} -> {who} start(s) more."]
    # Double-texting (a follow-up you send before the other person replies). Routed
    # here by the \bdouble.?text rule -- without this branch it fell to the reply-
    # latency default and answered a different question.
    if re.search(r"\bdouble.?text", q):
        dt = result.get("double_text")
        if isinstance(dt, dict) and isinstance(dt.get("me"), (int, float)) \
                and isinstance(dt.get("them"), (int, float)):
            who = ("You double-text more" if dt["me"] > dt["them"]
                   else "The other person double-texts more" if dt["them"] > dt["me"]
                   else "You both double-text about equally")
            return [f"[derived] Double-texts (a follow-up sent before a reply): you "
                    f"{dt['me']}, the other person {dt['them']} -> {who}."]
    # Left on read. The tool has no read-but-never-replied count -- only my
    # read->reply medians plus per-side read-receipt coverage. Pick the side the
    # question asks about: "leaves ME on read" is about THEIR receipts of my
    # outbound messages; anything else is about my receipts of their inbound.
    if re.search(r"\bon read\b", q):
        r2r = result.get("read_to_reply")
        if isinstance(r2r, dict):
            cov = r2r.get("coverage") or {}
            about_me = re.search(r"\b(leave|left|leaves)\s+me\b", q)
            if about_me:
                got = cov.get("outbound_with_date_read") or 0
                tot = cov.get("outbound_total") or 0
                side = "your messages to them"
            else:
                got = cov.get("inbound_with_date_read") or 0
                tot = cov.get("inbound_total") or 0
                side = "their messages to you"
            pct = round(got / tot * 100, 1) if tot else 0
            if not tot or got / tot < _READ_COVERAGE_MIN:
                return [f"[derived] Read receipts cover only {pct}% of {side} "
                        f"({got}/{tot}), so 'left on read' cannot be measured "
                        f"reliably -- state this caveat rather than a verdict."]
            fact = f"[derived] Read receipts cover {pct}% of {side} ({got}/{tot})"
            med = r2r.get("median_deliberation_seconds")
            # The read->reply medians measure MY replies to messages I read, so
            # they only bear on the "do I leave them on read" side.
            if not about_me and isinstance(med, (int, float)):
                fact += (f"; you reply a median of {med}s after reading "
                         f"({r2r.get('pairs', 0)} read->reply pairs)")
            return [fact + ". Only read-and-replied pairs are recorded -- a "
                    "read-but-never-answered count is not available, so report "
                    "what is measured, not a left-on-read verdict."]
    # Long conversations vs quick bursts.
    if re.search(r"\b(long conversations|quick bursts|bursts)\b", q):
        sess = result.get("sessions")
        if isinstance(sess, dict) and sess.get("median_messages") is not None:
            med = sess.get("median_messages") or 0
            shape = "longer conversations" if med >= 8 else "quick bursts"
            return [f"[derived] {sess.get('count')} conversation sessions, median "
                    f"{med} messages each -> {shape}."]
    # Ghosted / never replied. only_me lives under the "ghosted" section (a count of
    # one-sided sessions), NOT "sessions" (which has no such field).
    if re.search(r"\b(ghost|not repl|never repl)", q):
        ghost = result.get("ghosted")
        only_me = ghost.get("only_me") if isinstance(ghost, dict) else None
        if only_me is not None:
            sess = result.get("sessions") or {}
            return [f"[derived] {only_me} conversation(s) you started got no reply at "
                    f"all (out of {sess.get('count', '?')} sessions)."]
    # Default: reply-latency comparison.
    lat = result.get("reply_latency")
    if not isinstance(lat, dict):
        return []
    me, them = lat.get("me"), lat.get("them")
    if not (isinstance(me, dict) and isinstance(them, dict)):
        return []
    me_med, them_med = me.get("median_seconds"), them.get("median_seconds")
    if not (isinstance(me_med, (int, float)) and isinstance(them_med, (int, float))):
        return []
    # "How long does <them> take to reply" asks for THEIR median specifically --
    # lead with it so the small model does not quote your side by mistake (S2.2).
    if re.search(r"\bhow (long|fast|quick)\b", q) and not re.search(r"\bwho\b", q):
        them_p90 = them.get("p90_seconds")
        p90 = f", 90th percentile {them_p90}s" if isinstance(them_p90, (int, float)) else ""
        # State ONLY their median -- naming your own number here made the small
        # model quote the wrong side (S2.2). The answer is a single figure.
        return [f"[derived] The other person takes a median of {them_med}s to reply "
                f"to you{p90}."]
    # The reply-latency verdict must answer only a reply-SPEED question -- a question
    # that reached this tool for another reason (double-text, a future intent) would
    # otherwise get a bogus "who replies faster" fact. Gate it, but only when a
    # question is present (a no-question call keeps the default -- production always
    # supplies one, so this affects tests/edge callers only).
    if q and not _REPLY_SPEED_RE.search(q):
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


# --- Tapback / reaction labels for friendly narration -----------------------
_REACT_LABEL = {
    "love": "Loved (heart)", "like": "Liked (thumbs up)", "dislike": "Disliked",
    "laugh": "Laughed (haha)", "emphasize": "Emphasized (!!)",
    "question": "Questioned (?)", "custom-emoji": "custom emoji", "sticker": "sticker",
}


def _top_by_type(by_type):
    """(label, count) of the highest-count reaction type, or None."""
    if not isinstance(by_type, dict) or not by_type:
        return None
    top = max(by_type.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
    return _REACT_LABEL.get(top[0], top[0]), top[1]


def _derive_reaction_facts(result, question=""):
    q = question.lower()
    # Funniest leaderboard.
    if "funn" in q:
        fun = result.get("funniest")
        if isinstance(fun, list) and fun and isinstance(fun[0], dict):
            f = fun[0]
            return [f"[derived] {f.get('participant')} laughs at your messages most: "
                    f"{f.get('laughs')} laughs ({f.get('laughs_per_100_my_messages')} "
                    f"per 100 of your messages)."]
    # Most-reacted single message (hall of fame).
    if re.search(r"\bmost reactions|which .*message\b", q):
        hof = result.get("hall_of_fame")
        if isinstance(hof, list) and hof and isinstance(hof[0], dict):
            h = hof[0]
            return [f"[derived] Your most-reacted message: {_msg_snippet(h)} — "
                    f"{h.get('reaction_count')} reactions."]
    # How quickly people react.
    if re.search(r"\bhow (quickly|fast)|react.*quick", q):
        ttr = result.get("time_to_react")
        them = ttr.get("them_reacting") if isinstance(ttr, dict) else None
        if isinstance(them, dict) and them.get("median_seconds") is not None:
            return [f"[derived] People react to your messages in a median of "
                    f"{them['median_seconds']} seconds."]
    # Loved my messages (received love count).
    if re.search(r"\b(love|loved|heart)\b", q):
        rec = result.get("received")
        love = (rec.get("by_type") or {}).get("love") if isinstance(rec, dict) else None
        if love is not None:
            return [f"[derived] You received {love} 'Loved' reactions on your messages "
                    f"in this scope."]
    # React more than / reciprocity.
    if re.search(r"\bmore than|reciprocit|react.*more\b", q):
        rec = result.get("reciprocity")
        if isinstance(rec, dict) and rec.get("ratio") is not None:
            return [f"[derived] You gave {rec.get('given')} reactions and received "
                    f"{rec.get('received')} (ratio {rec.get('ratio')}): you react "
                    f"{'more' if (rec.get('ratio') or 0) > 1 else 'less'} than they do."]
    # Different reactions between us (given vs received breakdown).
    if re.search(r"\bdifferent|what reactions do\b", q):
        given = _top_by_type((result.get("given") or {}).get("by_type"))
        recv = _top_by_type((result.get("received") or {}).get("by_type"))
        if given or recv:
            parts = []
            if given:
                parts.append(f"you most often send '{given[0]}' ({given[1]})")
            if recv:
                parts.append(f"you most often receive '{recv[0]}' ({recv[1]})")
            return [f"[derived] Reaction use: {'; '.join(parts)}."]
    # Default: top tapback given.
    given = _top_by_type((result.get("given") or {}).get("by_type"))
    if given:
        return [f"[derived] Your most-used tapback is '{given[0]}' ({given[1]} times)."]
    return []


def _derive_word_trends_facts(result, question=""):
    facts = []
    pt = result.get("phrase_trend")
    if isinstance(pt, dict) and isinstance(pt.get("series"), list) and pt["series"]:
        series = pt["series"]

        def tot(r):
            return (r.get("me") or 0) + (r.get("them") or 0)
        first, last = series[0], series[-1]
        direction = ("rising" if tot(last) > tot(first)
                     else "falling" if tot(last) < tot(first) else "flat")
        facts.append(
            f"[derived] '{pt.get('phrase')}' appears in {pt.get('total_messages')} "
            f"messages; {first.get('period')}={tot(first)} -> {last.get('period')}="
            f"{tot(last)} ({direction}). First used by "
            f"{pt.get('coined_by', 'unknown')} on {str(pt.get('date') or '')[:10]}."
            if pt.get("coined_by") else
            f"[derived] '{pt.get('phrase')}' appears in {pt.get('total_messages')} "
            f"messages; {first.get('period')}={tot(first)} -> {last.get('period')}="
            f"{tot(last)} ({direction}).")
    emoji = result.get("emoji")
    if isinstance(emoji, dict) and isinstance(emoji.get("top"), list) and emoji["top"]:
        top = ", ".join(f"{e.get('emoji')} ({e.get('count')})"
                        for e in emoji["top"][:3] if isinstance(e, dict))
        facts.append(f"[derived] Top emoji: {top}.")
    dt = result.get("distinctive_terms")
    if isinstance(dt, dict) and isinstance(dt.get("terms"), list) and dt["terms"]:
        terms = "; ".join(
            f"{t.get('term')} ({t.get('verdict')})"
            + ("" if t.get("curated") else " [uncurated]")
            for t in dt["terms"][:5] if isinstance(t, dict))
        facts.append(f"[derived] Distinctive shared terms: {terms}.")
    lsm = result.get("lsm")
    if isinstance(lsm, dict) and lsm.get("score") is not None:
        s = lsm["score"]
        reading = ("very similar" if s >= 0.9 else "fairly similar" if s >= 0.8
                   else "moderately similar" if s >= 0.6 else "not very similar")
        facts.append(f"[derived] Language style matching score = {s} ({reading}); "
                     f"1.0 is identical style.")
    emo = result.get("emotions")
    if isinstance(emo, dict) and isinstance(emo.get("top"), list) and emo["top"]:
        top = ", ".join(f"{e.get('emotion')} ({e.get('count')})"
                        for e in emo["top"][:3] if isinstance(e, dict))
        facts.append(f"[derived] Top emotions: {top}.")
    elif isinstance(emo, dict) and not emo.get("top"):
        facts.append("[derived] Emotion analysis is not available for this scope.")
    val = result.get("valence")
    if isinstance(val, dict) and isinstance(val.get("series"), list) and val["series"]:
        s = val["series"]
        facts.append(f"[derived] Mood (mean valence) {s[0].get('period')}="
                     f"{s[0].get('mean_valence')} -> {s[-1].get('period')}="
                     f"{s[-1].get('mean_valence')}.")
    elif isinstance(val, dict) and val.get("available") is False:
        facts.append("[derived] Mood/valence analysis is not available for this scope.")
    rates = result.get("rates")
    if isinstance(rates, dict) and isinstance(rates.get("series"), list) and rates["series"]:
        s = rates["series"]
        first_me = (s[0].get("me") or {}) if isinstance(s[0], dict) else {}
        last_me = (s[-1].get("me") or {}) if isinstance(s[-1], dict) else {}
        facts.append(f"[derived] Your laughs per 100 messages: {s[0].get('period')}="
                     f"{first_me.get('laugh_per_100')} -> {s[-1].get('period')}="
                     f"{last_me.get('laugh_per_100')}.")
    return facts


def _derive_word_frequency_facts(result, question=""):
    rows = result.get("rows")
    if not (isinstance(rows, list) and rows):
        return []
    top = ", ".join(f"{r.get('ngram')} ({r.get('count')})"
                    for r in rows[:10] if isinstance(r, dict))
    kind = "phrases" if (result.get("ngram") or 1) >= 2 else "words"
    return [f"[derived] Most common {kind} (scanned {result.get('messages_scanned', '?')} "
            f"messages): {top}."]


def _derive_group_facts(result, question=""):
    q = question.lower()
    if re.search(r"\bcarr|dominates|dominated by|balanced|most active\b", q):
        members = result.get("members")
        if isinstance(members, list) and members and isinstance(members[0], dict):
            m = members[0]
            gini = (result.get("gini") or {}).get("gini")
            lead = (f"{m.get('member')} carries the group with "
                    f"{round((m.get('share') or 0) * 100, 1)}% of messages")
            if gini is not None:
                bal = "fairly balanced" if gini < 0.3 else "dominated by a few"
                lead += f"; Gini {gini} ({bal})"
            return [f"[derived] {lead}."]
    if "lurk" in q:
        lurkers = result.get("lurkers")
        if isinstance(lurkers, list):
            names = ", ".join(str(x.get("member") if isinstance(x, dict) else x)
                              for x in lurkers) if lurkers else "none"
            return [f"[derived] Lurkers (members who mostly read, rarely post): {names}."]
    if re.search(r"\b(starts|start|igniter|initiat)", q):
        ig = result.get("igniters")
        lb = ig.get("leaderboard") if isinstance(ig, dict) else None
        if isinstance(lb, list) and lb and isinstance(lb[0], dict):
            top = lb[0]
            return [f"[derived] {top.get('member')} starts the most conversations "
                    f"({round((top.get('share') or 0) * 100, 1)}%, "
                    f"{top.get('sessions_started')} of {ig.get('n_sessions')} sessions)."]
    if re.search(r"\btalks? to whom|who .*talks?\b", q):
        adj = result.get("adjacency")
        pairs = adj.get("pairs") if isinstance(adj, dict) else None
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict):
            p = pairs[0]
            return [f"[derived] The most active pair is {p.get('replier')} <-> "
                    f"{p.get('replied_to')} ({p.get('count')} exchanges)."]
    if re.search(r"\bstands? out\b", q):
        gl = result.get("group_lsm")
        members = gl.get("members") if isinstance(gl, dict) else None
        if isinstance(members, list) and members:
            odd = min(members, key=lambda m: m.get("lsm_score", 1)
                      if isinstance(m, dict) else 1)
            return [f"[derived] {odd.get('member')}'s style stands out most from the "
                    f"group (lowest LSM {odd.get('lsm_score')})."]
    # Default: top member by share.
    members = result.get("members")
    if isinstance(members, list) and members and isinstance(members[0], dict):
        m = members[0]
        return [f"[derived] Most active member: {m.get('member')} "
                f"({round((m.get('share') or 0) * 100, 1)}% of messages)."]
    return []


def _derive_participant_facts(result, question=""):
    """get_chat_participant_stats: a per-member list. Answer the 'who gets the most
    reactions per message, among members with >= N sent' question by applying the
    floor the prompt states, then ranking reactions_received / messages_sent."""
    if not (isinstance(result, list) and result):
        return []
    floor = 0
    m = re.search(r"at least (\d+)|(\d+)\+? messages", question.lower())
    if m:
        floor = int(m.group(1) or m.group(2))
    eligible = [r for r in result if isinstance(r, dict)
                and (r.get("messages_sent") or 0) >= floor]
    if not eligible:
        return [f"[derived] No members sent at least {floor} messages."]

    def rate(r):
        sent = r.get("messages_sent") or 0
        return (r.get("reactions_received") or 0) / sent if sent else 0
    top = max(eligible, key=rate)
    return [f"[derived] Among members with >= {floor} messages, {top.get('participant')} "
            f"gets the most reactions per message: {top.get('reactions_received')} "
            f"reactions / {top.get('messages_sent')} sent = {round(rate(top), 3)}."]


def _derive_voiceprint_facts(result, question=""):
    feats = result.get("features")
    if not isinstance(feats, dict):
        return []
    conf = result.get("confidence", "unknown")
    n = result.get("n_messages", "?")
    bits = []
    lc = feats.get("lowercase_ratio")
    if isinstance(lc, (int, float)):
        bits.append(f"{'mostly lowercase' if lc >= 0.6 else 'mixed case'} "
                    f"({round(lc * 100)}% lowercase)")
    burst = feats.get("burst_size")
    if isinstance(burst, (int, float)):
        bits.append(f"{'tends to double-text' if burst >= 1.8 else 'one message at a time'} "
                    f"(avg burst {round(burst, 1)})")
    ex = feats.get("exclaim_rate")
    if isinstance(ex, (int, float)) and ex:
        bits.append(f"exclamation rate {round(ex, 3)}")

    def phrases(key):
        items = result.get(key)
        if isinstance(items, list) and items:
            return ", ".join(f"'{p.get('phrase')}'" for p in items[:3]
                             if isinstance(p, dict) and p.get("phrase"))
        return ""
    openers, closers = phrases("openers"), phrases("closers")
    if openers:
        bits.append(f"common openers {openers}")
    if closers:
        bits.append(f"common closers {closers}")
    caveat = (" This rests on few messages (LOW confidence) -- describe it tentatively."
              if conf == "low" else "")
    return [f"[derived] Texting voice ({conf} confidence, {n} messages): "
            f"{'; '.join(bits)}.{caveat}"]


def _derive_summary_stats_facts(result, question=""):
    q = question.lower()
    # group_by_chat leaderboard (list): the top thread by volume.
    if isinstance(result, list):
        rows = [r for r in result if isinstance(r, dict)]
        if not rows:
            return []
        top = max(rows, key=lambda r: r.get("message_count") or 0)
        # A derived fact ends the loop, so no follow-up call can resolve the id
        # to a name -- present the id honestly instead of asking the model to.
        return [f"[derived] Your highest-volume thread is chat_id={top.get('chat_id')} "
                f"with {top.get('message_count')} messages (a chat id -- the thread's "
                f"name is not in this result)."]
    if not isinstance(result, dict):
        return []
    if re.search(r"\b(photo|picture|image|attachment)", q):
        ac = result.get("attachment_count")
        if ac is not None:
            return [f"[derived] {ac} attachments have been sent in this scope "
                    f"(individual attachments, not messages)."]
    mc = result.get("message_count")
    if mc is not None:
        return [f"[derived] {mc} messages in this scope "
                f"({result.get('from_me', '?')} from you, "
                f"{result.get('from_them', '?')} from them)."]
    return []


def _derive_fadeout_facts(result, question=""):
    wl = result.get("watchlist")
    if not (isinstance(wl, list) and wl):
        return []
    items = []
    for r in wl[:6]:
        if not isinstance(r, dict):
            continue
        driver = (r.get("top_driver") or {}).get("signal", "?")
        items.append(f"{r.get('contact')} ({r.get('phase')}, driver: {driver})")
    return [f"[derived] Drifting/fading ties (iMessage-only signal, not a verdict): "
            f"{'; '.join(items)}."]


def _derive_loose_threads_facts(result, question=""):
    if not (isinstance(result, list) and result):
        return ["[derived] No open loose threads / nudges were found."]
    items = []
    for r in result[:8]:
        if not isinstance(r, dict):
            continue
        kind = r.get("kind", "?")
        txt = _clip((r.get("nudge_text") or "").strip(), 70)
        items.append(f"[{kind}] {txt}" if txt else f"[{kind}]")
    return [f"[derived] {len(result)} open loose thread(s): " + " | ".join(items)]


# Defer (no derive/stop) ONLY for an apology/draft ask -- draft_apology needs a
# conflict_id the model must carry forward from find_conflicts, so stopping on a
# derived summary would prevent the draft. A diagnosis ask ("break down the tensest
# stretch") does NOT defer: the model does not reliably chain to diagnose_conflict,
# so a grounded find_conflicts summary (dates/signals, no-fault) beats letting the
# digest confabulate an episode.
_CONFLICT_DEFER_RE = re.compile(r"\bapolog|\b(draft|write)\b", re.I)


def _derive_conflict_facts(result, question=""):
    if _CONFLICT_DEFER_RE.search(question or ""):
        return []
    total = result.get("total_matching")
    rows = result.get("rows")
    if total is None or not isinstance(rows, list):
        return []
    if total == 0 or not rows:
        return ["[derived] No tense stretches / conflicts were detected in this thread."]
    top = rows[0] if isinstance(rows[0], dict) else {}
    signals = top.get("signals")
    sig = ", ".join(map(str, signals)) if isinstance(signals, list) else "?"
    # NOTE: deliberately no who-started / aggressor field exists -- do not infer one.
    return [f"[derived] {total} tense stretch(es) detected (deterministic text-only "
            f"signals, not confirmed fights). The strongest: around "
            f"{str(top.get('first_date') or '')[:10]}, signals: {sig}, corroboration: "
            f"{top.get('corroboration')}. These signals do NOT identify who was at "
            f"fault or who started it."]


# A who-started / whose-fault / who's-to-blame question about a conflict. The
# finder has NO aggressor field, so any fault verdict is a confabulation the
# no-aggressor design (Master Spec C2) forbids -- yet the small model reliably
# states one anyway when asked directly. So on this question the answer is
# COMPOSED IN CODE from the derived facts, never generated, mirroring how the DV
# floor and humility disclaimer are code-enforced rather than prompted.
_FAULT_QUESTION_RE = re.compile(
    r"\bwho (started|caused|instigated|is to blame|was at fault|escalated)\b"
    r"|\bwhose fault\b|\bwho'?s to blame\b|\bwho (was|is) the aggressor\b", re.I)


def _compose_no_fault_answer(scratchpad):
    """A code-composed reply for a who-started/fault conflict question: it relays
    the deterministic count/signals but never assigns blame. None when no conflict
    fact was gathered (then the normal answer path runs)."""
    conflict = next((f for f in scratchpad if "tense stretch" in f.lower()), None)
    if conflict is None:
        return None
    # No conflict was found: nothing to assign a start/fault to. Compose a clean
    # reply rather than splicing the negative fact into "what the signals show".
    if "no tense stretch" in conflict.lower():
        return (
            "I couldn't find any tense stretches or conflicts in this thread, so "
            "there's nothing to assign a start or fault to — and I wouldn't guess "
            "even if there were. These signals are a text-only read (reply gaps, "
            "one-sided sessions, tapbacks) that can't see calls, tone, or in-person "
            "context, so treat their absence as reassuring but not definitive.")
    body = conflict.split("]", 1)[-1].strip() if conflict.startswith("[") else conflict
    body = body.split("These signals do NOT")[0].strip().rstrip(".")
    return (
        "I can't say who started more of them, and I won't guess. The conflict "
        "signals are a text-only read (reply gaps, one-sided sessions, tapbacks) "
        "with no 'who started it' or fault information — it can't see calls, tone, "
        "or in-person context. "
        + (f"What the signals do show: {body}. " if body else "")
        + "Treat this as a prompt to reflect on the pattern, not a verdict on either "
        "person.")


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
    group_facts, group_ids = _preresolve_groups(chat, question)
    scratchpad.extend(group_facts)
    seen_calls = set()
    first_action = _suggest_first_action(question)
    # A named person the directory could not resolve (candidates found, none
    # matched) means the contact does not exist -- forcing a contact-scoped
    # aggregate would compute whole-corpus stats and misattribute them to that
    # name (the confabulation S7.6 guards). Drop the forced action so the loop
    # routes/answers honestly ("I can't find them") instead.
    if (first_action in _CONTACT_SCOPED_INTENT_TOOLS
            and not resolved and _extract_candidate_names(question)):
        first_action = None
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
        group_scope = group_ids[0] if (forced and len(group_ids) == 1) else None
        code_filled_scope = False
        if forced and action in _CONTACT_SCOPED_INTENT_TOOLS and scope is not None:
            args = {"contact_id": scope}  # fill the scope in code, not via the FM
            code_filled_scope = True
        elif forced and action in _GROUP_SCOPED_INTENT_TOOLS and group_scope is not None:
            args = {"chat_id": group_scope}  # fill the group chat_id deterministically
            code_filled_scope = True
        else:
            args = _fill_args(action, question, summary, scratchpad)
        result = _execute(chat, action, args, seen_calls, floor)
        # A forced (deterministically-routed) tool is the RIGHT tool; when the FM
        # arg-filler decorates it with a spurious shape arg that errors it (an
        # unsupported is_from_me/granularity value), retry once with only the
        # grounded/asked-for args (_RETRY_KEEP_KEYS) so the correct tool still
        # answers rather than the loop falling through to search/refusal. Never
        # shed a kept arg: a grounded contact_id or an asked-for date/filter is
        # part of the question, and dropping it would answer a different one
        # (whole-corpus stats misattributed to the named person -- the same
        # confabulation the code-filled-scope and unresolved-name guards prevent).
        # NOT for a code-filled scope (contact_id/chat_id): that call carried the
        # scope alone, so there is nothing sheddable by construction.
        if (forced and args and not code_filled_scope
                and isinstance(result, dict) and result.get("error")):
            kept = {k: v for k, v in args.items() if k in _RETRY_KEEP_KEYS}
            if kept != args:  # only retry when a shape arg can actually be shed
                result = _execute(chat, action, kept, seen_calls, floor)
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
    # Aggressor/fault question about a conflict: compose the reply in code so the
    # small model cannot verbalize a who-started verdict the tool has no data for
    # (Master Spec C2, enforced here the way the DV floor / disclaimer are).
    fault_answer = (_compose_no_fault_answer(scratchpad)
                    if _FAULT_QUESTION_RE.search(question) else None)
    if fault_answer is not None:
        reply = _render_stream(chat, iter([fault_answer]), show_model_header, model,
                               sanitize=floor.sanitize)
    else:
        reply = _answer(chat, question, summary, scratchpad, show_model_header, model,
                        sanitize=floor.sanitize)
    # Support resources must survive even a degraded/empty answer, so emit the
    # epilogue regardless of `reply` (fold into the returned string too).
    reply = _emit_epilogues(chat, reply, floor)
    messages.append({"role": "assistant", "content": reply})
    return reply
