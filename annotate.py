"""Batch LLM annotation CLI: sentiment/emotion/depth/event annotation of
conversation chunks into chunk_annotations, plus dyad-term curation
(--terms). Sequential Ollama JSON-mode requests, resumable (only rows
without an annotation are processed; Ctrl-C commits partial progress).
Mixing annotation models in one table is refused via the ingest_state
`annotate_model` guard, mirroring the embedding-model guard.

--quality picks the speed/label-fidelity tradeoff. 'distilled' (the default)
LLM-labels a random seed sample, then kNN-classifies every remaining chunk
from its chunk_vectors embedding: full corpus in ~1 min once seeds exist
(valence/depth at LLM fidelity, emotions rougher, events left empty --
source='distilled' marks the rows). The LLM-per-chunk tiers: on a reasoning
model the hidden thinking pass dominates wall time (~8.6s/chunk on
gpt-oss:20b), so 'balanced' asks for low reasoning effort (~3x faster,
agreement with 'max' within the model's own run-to-run noise) and 'fast'
additionally annotates several chunks per request (~4-8x faster, slightly
noisier); 'max' is the original one-chunk-per-request, model-default-
reasoning path. --upgrade-distilled lets an LLM tier re-annotate distilled
rows (the way to fill in events for chats that matter). LLM passes fan out
over ANNOTATE_WORKERS concurrent requests, which buys ~1.24x -- NOT the ~2x
this line used to claim. Measured 2026-07-28 (qwen3.5:4b, batch 8) with
OLLAMA_NUM_PARALLEL raised to 4: this hardware still effectively serializes
generation, so the concurrency mostly overlaps request overhead. Recorded so
nobody re-litigates the hope; OLLAMA_NUM_PARALLEL is deliberately left alone,
and the levers that DID pay are batch packing and the schema retry below.

'hybrid' sits between the two, and is budgeted in WALL CLOCK rather than
chunks (ANNOTATE_BUDGET_MINUTES; default 25 min on Ollama, 50 on apple),
which is what makes its runtime promise hold on any machine and corpus: a
stratified seed top-up (capped at its own share of the budget) and a
corpus-wide kNN pass first, then LLM upgrades of what kNN can't or probably
didn't get right -- event candidates first (a distilled row's events are
always empty), then the lowest-confidence rows -- until the deadline, which it
self-terminates at and resumes from on a rerun. EVERY phase is gated, seeding
and the O(rows x seeds) kNN pass included; each stops dispatching a pool-drain
early rather than at the deadline itself, and each starts from the per-request
time an earlier run banked in ingest_state so nothing goes out blind. It is
itself a distilled-upgrader, so --upgrade-distilled does not apply to it (see
annotate_hybrid.py).

--terms is a separate pipeline, not a --quality tier: it curates dyad_terms
verdicts, batches ANNOTATE_TERMS_BATCH_SIZE terms per request, and runs with
its own think (ANNOTATE_TERMS_THINK, default false -- reasoning about a
one-word verdict is ~11x latency for the same answer). A term that fails
curation ANNOTATE_TERMS_MAX_FAILURES times is parked in
dyad_terms.curate_failures rather than retried by every later run;
--retry-failed unparks.

Usage (chunks): annotate.py [--all | --top-dyads N] [--limit N]
                   [--model NAME]
                   [--quality distilled|hybrid|fast|balanced|max]
                   [--upgrade-distilled] [--dry-run]
Usage (terms):  annotate.py --terms [--limit N] [--model NAME]
                   [--retry-failed] [--dry-run]
--terms and --quality/--upgrade-distilled are mutually exclusive.
"""

import argparse
import json

import annotate_distill
import annotate_hybrid
import config
import ollama
import progress
import schema
from config import (
    ANNOTATE_BATCH_SIZE,
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_MODEL,
    ANNOTATE_QUALITY,
    ANNOTATE_TERMS_BATCH_SIZE,
    ANNOTATE_TERMS_THINK,
    ANNOTATE_THINK,
    ANNOTATE_TIMEOUT,
    ANNOTATE_TOP_DYADS,
    ANNOTATE_WORKERS,
    TARGET_DB,
)
from db import (
    check_model_guard,
    connect_default,
    run_cli,
)
from function_words import EMOTION_LABELS

# The dispatch/budget machinery lives in annotate_rate; re-exported so
# annotate._pooled / ._GroupRate / ._DispatchDeadline (annotate_distill,
# annotate_hybrid, the tests) keep resolving.
from annotate_rate import (  # noqa: F401
    _COLD_GROUP_SECONDS_PER_CHUNK,
    _DispatchDeadline,
    _GroupRate,
    _group_rate_key,
    _load_group_rate,
    _mean_group_rows,
    _pooled,
    _save_group_rate,
)

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# --quality presets: (think, batch_size). think is Ollama's reasoning-effort
# control, written against an EFFORT-LEVEL model: 'low' asks gpt-oss for a
# cheap reasoning pass, which is what makes balanced ~3x faster than max
# (measured on gpt-oss:20b: max ~8.6s/chunk, balanced ~3s, fast ~1-2s). To a
# model that merely toggles thinking -- which the default CHAT_MODEL is -- the
# same string is only truthy and buys a FULL pass, so _resolve_quality
# translates it to False there via config.has_effort_levels. Read these tuples
# as the gpt-oss values; what a toggle model actually gets is the translation.
# None omits the flag (the model's own default reasoning) and is NEVER
# translated, so 'max' keeps meaning what it says on every backend.
# 'distilled' uses its tuple only for the LLM seed pass; the rest of the
# corpus is kNN-classified.
# 'hybrid' is the one preset measured on a TOGGLE model (qwen3.5:4b,
# 2026-07-27), and both of its numbers come out differently: think=False, not
# 'low', because an effort level is not portable -- 'low' is merely *truthy*
# there, buying a full reasoning pass that times out the request; and batch 8,
# because 16/24 measured WORSE per chunk (1.9-2.6s vs 1.1-1.4s), the bigger
# prompt costing more than the saved round-trips. Its tuple covers the seed
# pass and the phase-3 LLM upgrades; everything else is kNN-classified.
QUALITY_PRESETS = {
    "max": (None, 1),
    "balanced": ("low", 1),
    "fast": ("low", 8),
    "distilled": ("low", 1),
    "hybrid": (False, 8),
}

# Apple backend: batch by chunk-text chars, not a fixed row count. The FM
# window is a fixed 4,096 tokens covering PROMPT AND RESPONSE, and a request
# over it raises rather than answering -- so the batched-annotation prompt has
# to be packed under a char budget rather than the Ollama batch_size.
#
# 2,000 = one CHUNK_MAX_CHARS: never two full-size chunks in one request, while
# genuinely short chunks still pair up. Smaller is strictly CHEAPER here, which
# is the counter-intuitive part: the dominant batched-call failure on this
# backend is malformed JSON, not overflow, so a batched apple request usually
# fails whatever its size and each failure costs its wall time PLUS the
# single-chunk retries. (docs/DECISIONS.md has the budget-vs-failure table and
# why the earlier 12,000 and 8,000 values did not fix what they were chosen for.)
#
# Residual, not fixable here: _char_budget_groups always emits at least one row,
# so a chunk bigger than the budget is still sent alone and can still overflow
# (~3% of chunks, mostly emoji/non-Latin text that tokenizes far worse than the
# 4-chars-per-token rule of thumb). Such a row is lost for the run and retried
# the next.
#
# Shared by every tier (_annotation_groups), so the reduction is global. A
# module constant, not a config knob: a property of the FM window, not a user
# tunable.
_APPLE_ANNOTATE_CHAR_BUDGET = 2000

# Ollama backend: the same packing idea for a different reason. Nothing here
# has to fit a fixed window -- batch_size already bounds the request -- but a
# fixed ROW count lets a group's cost swing with chunk length. Measured
# 2026-07-28 (qwen3.5:4b, batch 8): ~4.6 s/chunk on typical event chunks
# (~1,550 avg chars) against 13.1-28.3 s/chunk on the longest ones (~2,150),
# because a long, event-dense group is exactly where the model starts dropping
# ids from the batched reply and _annotate_group has to buy them back.
#
# That swing does not just cost time, it POISONS the budget: _save_group_rate
# banks the WORST group of a run, so one 28 s/chunk group inflates
# _DispatchDeadline.reserve() for the whole next run and the gate declines
# dispatches it could have afforded. Capping the worst case is therefore worth
# budget twice -- faster groups AND a smaller reserve. 10,000 chars is ~4-6 of
# the long chunks (whose latency then clusters instead of tailing) while short
# chunks still fill a whole batch of 8, so the fast case is untouched.
#
# A module constant, not a config knob, exactly like the apple budget above: it
# is a property of the backend, not something a user tunes. And like it, it is
# applied in _annotation_groups, so it covers EVERY non-apple tier at once
# (fast/balanced/max, the distilled seed pass, hybrid's upgrades). Apple's
# packing is untouched -- it has its own, much smaller budget and no row cap.
_OLLAMA_ANNOTATE_CHAR_BUDGET = 10_000

EVENT_KINDS = ("good_news", "gratitude")
RESPONSE_CLASSES = (
    "active_constructive",
    "passive_constructive",
    "active_destructive",
    "passive_destructive",
)
TERM_VERDICTS = ("inside_joke", "nickname", "ordinary", "noise")

_ANNOTATE_SYSTEM = f"""You annotate excerpts of a personal text-message history.
Respond with ONLY one JSON object, no prose, with exactly these keys:
{{"valence": <float between -1 (very negative) and 1 (very positive): the overall emotional tone>,
 "emotions": [<up to 3 labels, each one of: {", ".join(EMOTION_LABELS)}>],
 "depth": <integer 1-5; 1 = pure logistics/scheduling, 5 = heart-to-heart>,
 "events": [{{"kind": <"good_news" or "gratitude">,
             "excerpt": <verbatim quote of the message sharing the news or thanks>,
             "response_class": <how the other party responded: "active_constructive",
              "passive_constructive", "active_destructive", "passive_destructive",
              or null if there was no response>}}]}}
"events" lists moments where someone shares good news or expresses gratitude;
use an empty list when there are none."""

_BATCH_SYSTEM_SUFFIX = """
You will receive SEVERAL excerpts, each starting with a line '### chunk <id>'.
Annotate each excerpt independently of the others. Respond with ONLY one JSON
object mapping each chunk id (as a string) to its annotation object:
{"<id>": {...}, "<id>": {...}, ...}."""

_TERM_VERDICT_LIST = ", ".join(f'"{v}"' for v in TERM_VERDICTS)

_TERM_DEFS = """"inside_joke" = a private/coined expression between these people;
"nickname" = a name they call a person, pet, or place; "ordinary" = a normal
word that just happens to be frequent; "noise" = a typo, artifact, or
meaningless fragment."""

_TERMS_SYSTEM = f"""You classify a term that appears unusually often in one text-message chat.
{_TERM_DEFS}
Respond with ONLY one JSON object: {{"verdict": <one of {_TERM_VERDICT_LIST}>}}."""

_TERMS_BATCH_SYSTEM = f"""You classify terms that appear unusually often in one text-message chat.
{_TERM_DEFS}
You will receive SEVERAL terms, each starting with a line '### term <n>'.
Classify each term independently of the others. Respond with ONLY one JSON
object mapping each term's number (as a string) to its verdict, one of
{_TERM_VERDICT_LIST}: {{"0": "ordinary", "1": "inside_joke", ...}}."""

# chat_once_json's default nudge only says "one JSON object"; a correction
# that names the exact shape recovers replies that would otherwise cost the
# caller a fallback request per term in the batch.
_TERMS_NUDGE = (
    f'Respond with only the JSON object {{"verdict": <one of '
    f'{_TERM_VERDICT_LIST}>}} -- no prose, no code fences, nothing else.'
)
_TERMS_BATCH_NUDGE = (
    f'Respond with only one JSON object mapping each term number to its '
    f'verdict, one of {_TERM_VERDICT_LIST}, like {{"0": "ordinary", '
    f'"1": "inside_joke"}} -- no prose, no code fences, nothing else.'
)

# Give up on a group's single-term fallbacks after this many in a row fail.
# A batch that comes back unusable otherwise becomes len(group) sequential
# requests (each retried once) on one worker slot -- up to ~50 requests at
# the default batch size, i.e. slower than not batching at all -- and charges
# a failure to every term in it.
_GROUP_GIVEUP = 5
_ABANDONED = "group abandoned after repeated failures (left pending)"

# A run in which the model has answered nothing at all is a configuration
# failure (wrong model, an unsupported think, Ollama serving something else),
# not this many independent bad terms -- and parking the whole corpus over it
# is exactly the sticky state curate_failures exists to prevent.
_TERMS_ABORT_AFTER = 20

def _connect():
    return connect_default(DB_PATH, rows=False)


# ---------------------------------------------------------------------------
# Chunk annotation
# ---------------------------------------------------------------------------

def _pending_chunks(con, all_chats, top_dyads, limit, upgrade_distilled=False):
    # With upgrade_distilled, kNN-classified rows count as pending too, so an
    # LLM tier replaces them (fills in events) for the chats in scope.
    cond = ("(a.chunk_id IS NULL OR a.source = 'distilled')"
            if upgrade_distilled else "a.chunk_id IS NULL")
    sql = f"""
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN chunk_annotations a ON a.chunk_id = c.id
        WHERE {cond}
    """
    params = []
    if not all_chats:
        sql += """  AND c.chat_id IN (
            SELECT chat_id FROM messages WHERE chat_id IS NOT NULL
            GROUP BY chat_id ORDER BY COUNT(*) DESC LIMIT ?)
        """
        params.append(top_dyads)
    sql += " ORDER BY c.id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return con.execute(sql, params).fetchall()


def _check_model_guard(con, model, table="chunk_annotations", predicate="1"):
    """Refuse to mix annotation models within one artifact -- db.check_model_guard
    (the shared shape, also chunking.build_vec_table's embedding guard) with this
    pipeline's key, table and remedy sentence. `predicate` narrows the count to
    the rows this model actually produced (dyad_terms holds uncurated rows too).
    Records the chosen model when proceeding."""
    check_model_guard(
        con, key="annotate_model", model=model, table=table, predicate=predicate,
        message=(
            "{table} holds {n} rows annotated by {prev}, but the "
            "model is now {model}. Mixing annotation models skews every "
            "downstream aggregate -- clear {table} (or reset_db.py) "
            "to switch, or rerun with --model {prev}."
        ),
    )


def _ensure_column(con, table, column, dry_run=False):
    """Self-heal a column that postdates DBs already in the wild, returning
    whether `column` is present afterwards. annotate.py is reachable without
    re-running ingest.py -- which is what normally applies migrations -- so a
    DB from before dyad_terms.curate_failures (--terms) or
    chunk_annotations.knn_confidence (the distilled/hybrid tiers) would die on
    'no such column' instead. Run the canonical migration (idempotent) rather
    than keeping a second, drifting copy of it here.

    Scope note: create_schema does more than the one ALTER -- it re-runs the
    DDL scripts, recreates the FTS triggers and rescans reactions for the
    old guid prefix. That's a one-off (the PRAGMA gate means the next run
    skips it entirely) and worth not having a second, drifting copy of the
    migration here; schema.py stays the only place that DDL lives.

    `dry_run=True` skips the migration and returns False instead. --dry-run
    promises no writes, and that reaction rescan is a data write on a path
    that would otherwise have none -- so a previewing caller degrades its own
    reads (the column simply has no values yet) rather than migrating behind
    the user's back."""
    cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        return True
    if dry_run:
        return False
    schema.create_schema(con)
    con.commit()
    return True


def _validate_annotation(data):
    """Clamp/filter a parsed annotation object to the storage contract."""
    try:
        valence = float(data.get("valence", 0.0))
    except (TypeError, ValueError):
        valence = 0.0
    valence = max(-1.0, min(1.0, valence))

    emotions = [
        e for e in (data.get("emotions") or [])
        if isinstance(e, str) and e in EMOTION_LABELS
    ][:3]

    try:
        depth = int(data.get("depth", 1))
    except (TypeError, ValueError):
        depth = 1
    depth = max(1, min(5, depth))

    events = []
    for ev in data.get("events") or []:
        if not isinstance(ev, dict) or ev.get("kind") not in EVENT_KINDS:
            continue
        response_class = ev.get("response_class")
        if response_class not in RESPONSE_CLASSES:
            response_class = None
        excerpt = ev.get("excerpt")
        events.append({
            "kind": ev["kind"],
            "excerpt": excerpt if isinstance(excerpt, str) else "",
            "response_class": response_class,
        })
    return valence, emotions, depth, events


def _annotation_object_schema():
    """JSON Schema for ONE annotation object, built from the very constants
    _validate_annotation checks against (EMOTION_LABELS, EVENT_KINDS,
    RESPONSE_CLASSES) so a schema and the validator can never drift apart.

    A fresh dict per call, not a shared module constant: the result is embedded
    once per chunk id in _annotation_schema, and aliasing one mutable dict into
    N properties is a footgun for no saving."""
    return {
        "type": "object",
        "properties": {
            # The numeric bounds are as much documentation as constraint:
            # Ollama's grammar converter enforces an integer range but not a
            # float's. That costs nothing, because _validate_annotation clamps
            # both regardless -- the schema is an optimization for COVERAGE,
            # never the validation gate.
            "valence": {"type": "number", "minimum": -1, "maximum": 1},
            "emotions": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": list(EMOTION_LABELS)},
            },
            "depth": {"type": "integer", "minimum": 1, "maximum": 5},
            "events": {
                "type": "array",
                # No measured basis, unlike emotions' 3 (which _validate_
                # annotation also truncates to) -- a ceiling so a grammar
                # cannot run away. This is the one field whose shape is
                # unbounded: an array of objects each carrying a free-form
                # excerpt, i.e. the only thing here that can generate long
                # enough to hit ANNOTATE_TIMEOUT, which per the ladder costs
                # the group its retry and drops it to per-chunk singles. A
                # conversation chunk with more than 8 distinct good-news or
                # gratitude events is not a case worth keeping the tail open
                # for; _validate_annotation does not cap events either way, so
                # nothing downstream depends on this number.
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(EVENT_KINDS)},
                        "excerpt": {"type": "string"},
                        # "one of the four classes, or null when there was no
                        # response" -- as anyOf rather than a union
                        # "type": ["string", "null"] or an enum with a null in
                        # it. All three express it, but anyOf is the oldest and
                        # most portable construct in the JSON-schema-to-grammar
                        # converter Ollama's structured outputs run on, while
                        # union types and mixed-type enums are version-
                        # dependent there. _validate_annotation already coerces
                        # anything outside RESPONSE_CLASSES to None, so this is
                        # about the model emitting a usable null, not about
                        # trusting it.
                        "response_class": {
                            "anyOf": [
                                {"type": "string",
                                 "enum": list(RESPONSE_CLASSES)},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["kind", "excerpt", "response_class"],
                },
            },
        },
        "required": ["valence", "emotions", "depth", "events"],
    }


def _annotation_schema(chunk_ids):
    """JSON Schema (a dict, for Ollama structured outputs via `format=`)
    requiring exactly one annotation object per id in `chunk_ids`, keyed by the
    same str(id) the batched prompt asks for.

    This is what the schema-constrained retry in _annotate_group spends its one
    request on: coverage of the named ids is guaranteed BY CONSTRUCTION, where
    a free-JSON batch merely asks for it. Kept to the missed ids only, so the
    grammar stays small and so does its overhead. Pure -- no I/O, no config
    reads, deterministic for a given id list -- hence safe to build inside a
    worker thread."""
    keys = [str(cid) for cid in chunk_ids]
    return {
        "type": "object",
        "properties": {key: _annotation_object_schema() for key in keys},
        "required": keys,
    }


def _resolve_event_message_ids(con, chunk_id, events):
    """Attach message_id to each event by exact-substring match of its
    excerpt against the chunk's messages; only an unambiguous (single)
    match counts."""
    if not events:
        return
    rows = con.execute(
        "SELECT id, text FROM messages WHERE chunk_id = ? AND text IS NOT NULL",
        (chunk_id,),
    ).fetchall()
    for ev in events:
        matches = [mid for mid, text in rows if ev["excerpt"] and ev["excerpt"] in text]
        ev["message_id"] = matches[0] if len(matches) == 1 else None


def _annotate_batch(group, model, think):
    """One JSON-mode request annotating several chunks at once; returns
    {chunk_id: annotation dict} for the ids the reply actually covered --
    the caller retries any missing/mangled id with a single-chunk request."""
    user = "\n\n".join(f"### chunk {cid}\n{text}" for cid, text in group)
    data = ollama.chat_once_json(
        [{"role": "system", "content": _ANNOTATE_SYSTEM + _BATCH_SYSTEM_SUFFIX},
         {"role": "user", "content": user}],
        model=model, think=think, timeout=ANNOTATE_TIMEOUT,
    )
    if not isinstance(data, dict):
        return {}
    return {cid: data[str(cid)] for cid, _ in group
            if isinstance(data.get(str(cid)), dict)}


def _annotate_schema_retry(rows, model, think):
    """One SCHEMA-CONSTRAINED request re-asking for exactly the ids a batched
    free-JSON reply missed; returns {chunk_id: annotation dict} for whatever
    came back, and {} on any failure at all.

    Why a second batched request instead of going straight to singles: measured
    2026-07-28 (qwen3.5:4b), free-JSON is bimodal. With full id coverage it is
    the fastest option there is (~4.6 s/chunk on typical event chunks); when it
    drops ids -- correlated with long, event-dense chunks -- every miss buys its
    own single-chunk request, each with chat_once_json's own nudge retry, and
    the group lands at 13-28 s/chunk. Constrained decoding is NOT universally
    faster (the grammar costs a constant: 7.0 s/chunk against free-JSON's 4.6
    on typical chunks), so it does not belong on the fast path -- but its
    coverage is guaranteed by construction, which is exactly what the retry
    needs. Worst case becomes roughly one extra ~40-55 s request rather than up
    to batch_size sequential singles.

    This is the one annotation request that reaches ollama.chat_once directly
    -- chat_once_json hardcodes format='json' and so cannot carry a dict schema
    -- which means it also has to resolve `think` the way that seam does, via
    ollama.effective_think. Skipping it would be silently fatal on a model that
    can't take the value: chat_once_json flips ollama._think_unsupported the
    first time a request proves that (gpt-oss answers think=false with EMPTY
    content rather than erroring), and re-sending the rejected value here would
    fail EVERY schema retry for the rest of the process -- turning a rung that
    exists to remove a request per miss into one that adds one. The flag is
    always already resolved by the time this runs: the rung is reachable only
    behind a batched request, which went through chat_once_json.

    Failure is total and silent by design. The caller's per-chunk fallback is
    still behind this, so a transport error, a timeout, a server that rejects
    the schema, a non-dict reply, unparseable content -- all of it has to
    degrade to "covered nothing" rather than propagate. Not because a raise
    would kill the worker: _pooled hands a worker exception to the future and it
    surfaces on the CALLER's thread at done.result(), where it aborts the whole
    PHASE (the group included) and _run_upgrades' finally can only commit what
    already landed. An Ollama build that will not take a dict `format` -- or a
    grammar that runs past ANNOTATE_TIMEOUT -- must cost this group its retry,
    never the run. The rung above (_annotate_batch) is deliberately unguarded by
    contrast: chat_once_json already returns None rather than raising on a
    malformed reply, so its failures are values, not exceptions.
    KeyboardInterrupt still propagates (it is not an Exception), which is what
    keeps the tier Ctrl-C safe."""
    ids = [cid for cid, _text in rows]
    user = "\n\n".join(f"### chunk {cid}\n{text}" for cid, text in rows)
    try:
        reply = ollama.chat_once(
            [{"role": "system", "content": _ANNOTATE_SYSTEM + _BATCH_SYSTEM_SUFFIX},
             {"role": "user", "content": user}],
            model=model, format=_annotation_schema(ids),
            timeout=ANNOTATE_TIMEOUT, think=ollama.effective_think(think),
        )
        # chat_once returns the response's `message` dict (possibly {}), so the
        # parse is ollama._extract_reply's: json.loads of the assistant
        # content, nothing cleverer. Deliberately WITHOUT its corrective-nudge
        # pass -- a grammar-constrained reply cannot come back fenced or with a
        # preamble, and the per-chunk fallback already is the retry for
        # anything that somehow does.
        data = json.loads((reply or {}).get("content") or "")
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {cid: data[str(cid)] for cid in ids
            if isinstance(data.get(str(cid)), dict)}


def _annotate_group(group, model, think):
    """All LLM work for one group of (chunk_id, text) rows: a batched request
    for a multi-chunk group -- then one schema-constrained request naming
    exactly the ids that reply missed, then a single-chunk request for anything
    STILL missing -- or just the single request for a singleton. Runs on worker
    threads; returns {chunk_id: annotation dict | None}.

    The ladder is ordered by what each rung costs when it works. Free-JSON with
    full coverage is the cheapest thing available and stays the fast path: a
    fully covered batch issues exactly ONE request and nothing else. The schema
    retry (see _annotate_schema_retry) caps the failure mode at one extra
    request instead of one per miss. The per-chunk singles stay underneath as
    belt and braces for a schema request that errored, timed out or came back
    unusable -- expected ~never, but the group must still complete.

    Apple FM never reaches the middle rung: its `format` seam is not a guided
    schema (an empty-properties DynamicGenerationSchema pins the reply to {}),
    so handing applefm a dict format is undefined. It keeps today's behavior --
    batch, then a single-chunk request per miss."""
    batched = _annotate_batch(group, model, think) if len(group) > 1 else {}
    # Only ids a batch actually attempted and missed are worth a schema retry;
    # a singleton group has had no batched request at all, so its "miss" is
    # just its first request and belongs on the single-chunk path below.
    missed = [(cid, text) for cid, text in group if cid not in batched]
    # `model or config.CHAT_MODEL` is how ollama.chat_once itself resolves the
    # backend, and this guard has to ask the same question the dispatch will:
    # today they cannot diverge (annotate always passes a resolved model, and
    # ANNOTATE_MODEL falls back to CHAT_MODEL), but "apple never receives a dict
    # format" is a boundary, and a boundary that agrees with its seam by
    # coincidence is one bug away from not being one.
    if (len(group) > 1 and missed
            and not config.is_apple_model(model or config.CHAT_MODEL)):
        batched.update(_annotate_schema_retry(missed, model, think))
    out = {}
    for chunk_id, text in group:
        data = batched.get(chunk_id)
        if data is None:
            data = ollama.chat_once_json(
                [{"role": "system", "content": _ANNOTATE_SYSTEM},
                 {"role": "user", "content": text}],
                model=model, think=think, timeout=ANNOTATE_TIMEOUT,
            )
        out[chunk_id] = data if isinstance(data, dict) else None
    return out


def _write_annotation(con, chunk_id, data, source="llm"):
    """Validate + store one parsed annotation (the INSERT every path --
    LLM tiers, seed pass, kNN classify -- goes through)."""
    valence, emotions, depth, events = _validate_annotation(data)
    _resolve_event_message_ids(con, chunk_id, events)
    con.execute(
        "INSERT OR REPLACE INTO chunk_annotations"
        "(chunk_id, valence, emotions_json, depth, events_json, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chunk_id, valence, json.dumps(emotions), depth,
         json.dumps(events), source),
    )


def _char_budget_groups(pending, budget, max_rows=None):
    """Greedily pack (chunk_id, text) rows into groups whose combined chunk
    text stays under `budget` chars and -- when `max_rows` is given -- whose row
    count stays at or under it. Accumulate until the next row would breach
    either limit, then start a new group; a single chunk bigger than the budget
    still gets its own group (>=1 row always), which _annotate_group sends as a
    singleton, exactly like batch_size 1.

    Shared by both backends, for two different reasons. Apple passes no
    max_rows, because a row count is the wrong measure there at all: its fixed
    4,096-token window covers prompt AND response, so chars are the only unit
    that bounds a request. Ollama passes its batch_size, keeping the row cap it
    has always had, and uses the budget only to stop a group of long chunks
    from costing several times what a group of short ones does (rationale and
    numbers: _OLLAMA_ANNOTATE_CHAR_BUDGET)."""
    groups = []
    group = []
    used = 0
    for row in pending:
        text_len = len(row[1] or "")
        full = max_rows is not None and len(group) >= max_rows
        if group and (full or used + text_len > budget):
            groups.append(group)
            group, used = [], 0
        group.append(row)
        used += text_len
    if group:
        groups.append(group)
    return groups


def _annotation_groups(pending, model, batch_size):
    """Split (chunk_id, text) rows into request-sized groups. Both backends
    pack by char budget; what differs is the row cap and the reason. Apple has
    no row cap -- its fixed window makes a row count the wrong unit entirely --
    while every other backend caps at batch_size and treats the (much larger)
    budget purely as a ceiling on a group of long chunks. One home for the
    branch so a tier can never batch a backend by the wrong unit by accident --
    and the backend is resolved exactly as ollama.chat_once resolves it, so the
    packing and the dispatch can never disagree about which one this is."""
    if config.is_apple_model(model or config.CHAT_MODEL):
        return _char_budget_groups(pending, _APPLE_ANNOTATE_CHAR_BUDGET)
    return _char_budget_groups(pending, _OLLAMA_ANNOTATE_CHAR_BUDGET,
                               max_rows=max(1, batch_size))


def annotate_chunks(con, model, all_chats=False, top_dyads=ANNOTATE_TOP_DYADS,
                    limit=None, dry_run=False, think=None, batch_size=1,
                    workers=1, upgrade_distilled=False):
    # `top_dyads` defaults to ANNOTATE_TOP_DYADS, bound at import, so a direct
    # caller can reach _pending_chunks' scope `LIMIT ?` with a negative env
    # value that main's own check never saw. Same rule and same gate as there:
    # refused where it is resolved, and only where the run binds it (`all_chats`
    # omits the scope clause entirely).
    if top_dyads is not None and top_dyads < 0 and not all_chats:
        raise SystemExit(
            f"ANNOTATE_TOP_DYADS cannot be negative (got {top_dyads}); SQLite "
            f"reads it as no limit at all, so a scoped run silently becomes an "
            f"--all one. Pass all_chats=True if that is what you want.")
    pending = _pending_chunks(con, all_chats, top_dyads, limit,
                              upgrade_distilled=upgrade_distilled)
    scope = "all chats" if all_chats else f"top {top_dyads} chats by message count"
    settings = ""
    if think is not None:
        settings += f", think {think}"
    if batch_size > 1:
        settings += f", batch {batch_size}"
    if workers > 1:
        settings += f", workers {workers}"
    if upgrade_distilled:
        settings += ", upgrading distilled rows"
    print(f"{len(pending)} chunks pending annotation ({scope}, model {model}{settings})")
    if dry_run:
        preview = ", ".join(str(cid) for cid, _ in pending[:10])
        print(f"--dry-run: first chunk ids: {preview or '(none)'}")
        return 0
    if not pending:
        return 0
    _check_model_guard(con, model)

    groups = _annotation_groups(pending, model, batch_size)
    n_done = n_skipped = 0
    try:
        with progress.track("Annotating chunks", total=len(pending),
                            header=False) as p:
            for group, results in _pooled(
                    groups, lambda g: _annotate_group(g, model, think), workers):
                for chunk_id, _text in group:
                    data = results.get(chunk_id)
                    if data is None:
                        n_skipped += 1
                        print(f"  chunk {chunk_id}: unparseable response, skipped "
                              f"(retried next run)")
                        p.advance()
                        continue
                    _write_annotation(con, chunk_id, data)
                    n_done += 1
                    p.advance()
                    if n_done % ANNOTATE_COMMIT_EVERY == 0:
                        con.commit()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- committed {n_done} annotations "
              f"({n_skipped} skipped); rerun to resume.")
        return n_done
    con.commit()
    print(f"Annotated {n_done} chunks ({n_skipped} skipped)")
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_quality(quality, model):
    """Preset name -> (think, batch_size), with the preset's effort level
    translated for `model` and the ANNOTATE_THINK / ANNOTATE_BATCH_SIZE env
    overrides applied on top.

    The presets were written against an effort-level model (gpt-oss:20b), where
    'low' asks for a CHEAP reasoning pass and is what makes balanced ~3x faster
    than max. To a model that merely toggles thinking -- which includes the
    default CHAT_MODEL -- that same string is only truthy and buys a FULL pass:
    measured 282.6s vs 26.2s per dyad term on gemma4, and a timed-out request
    on qwen3.5:4b at batch 8. So on those models an effort level resolves to
    False, which preserves the ladder the tiers actually mean (max = the
    model's own default reasoning, balanced/fast = none) instead of inverting
    it into "the fast tier is the slow one".

    Only the preset is translated, and only when the value is an effort STRING:
    None (omit the flag) and an explicit True/False pass through untouched, so
    the tri-state survives. An explicit ANNOTATE_THINK always wins -- a user who
    names a think has said something this cannot second-guess."""
    try:
        think, batch_size = QUALITY_PRESETS[quality]
    except KeyError:
        raise SystemExit(
            f"unknown ANNOTATE_QUALITY {quality!r} "
            f"(choose from {', '.join(sorted(QUALITY_PRESETS))})"
        ) from None
    if ANNOTATE_THINK:
        think = _parse_think(ANNOTATE_THINK)
    elif isinstance(think, str) and not config.has_effort_levels(model):
        think = False
    if ANNOTATE_BATCH_SIZE:
        batch_size = ANNOTATE_BATCH_SIZE
    return think, max(1, batch_size)


def _parse_think(value):
    """Env think string -> the tri-state ollama.chat_once wants: True/False
    for toggle models, an effort string for effort models, None to omit the
    flag entirely. 'omit'/'none' reach that third state, which is NOT False
    (which sends `think: false`) and is the right setting for at least one
    real model family -- without a spelling for it, 'none' would forward the
    literal string "none" to Ollama."""
    return {"true": True, "false": False,
            "omit": None, "none": None}.get(value, value)


def _resolve_terms_think():
    """Terms get their own think, NOT --quality's: classifying a term is a
    four-way choice answered in one word, so reasoning is pure latency
    (~11x on gemma4 -- see config.ANNOTATE_TERMS_THINK). Default false;
    ANNOTATE_TERMS_THINK overrides, including 'omit' for the flagless
    middle state (ANNOTATE_THINK deliberately does not --
    it tunes the chunk tiers, and having it reach in here is the bug this
    replaces)."""
    return _parse_think(ANNOTATE_TERMS_THINK) if ANNOTATE_TERMS_THINK else False


def _build_parser():
    """The CLI's flag surface, extracted from main() so it is greppable and
    testable on its own."""
    parser = argparse.ArgumentParser(
        description="Batch LLM annotation of conversation chunks (or dyad-term "
                    "curation with --terms). Resumable; Ctrl-C commits progress.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true",
                       help="annotate every chunk, not just the top dyads")
    scope.add_argument("--top-dyads", type=int, metavar="N", default=None,
                       help=f"annotate the top N chats by message count "
                            f"(default {ANNOTATE_TOP_DYADS})")
    parser.add_argument("--terms", action="store_true",
                        help="curate dyad_terms verdicts instead of annotating chunks")
    parser.add_argument("--limit", type=int, metavar="N", default=None,
                        help="process at most N rows this run (hybrid: at most N "
                             "upgrades WRITTEN, not dispatched)")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model (default {ANNOTATE_MODEL})")
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default=None,
                        help="speed/quality tradeoff: distilled = LLM seed sample + "
                             "kNN classify the rest from embeddings (~1 min corpus-"
                             "wide once seeded; no events); hybrid = distilled from a "
                             "wider stratified seed pool, then LLM-upgrade the event "
                             "candidates and the least-confident rows until a "
                             "wall-clock budget runs out (ANNOTATE_BUDGET_MINUTES); "
                             "max = one chunk per request at the model's default "
                             "reasoning effort; balanced = low reasoning effort (~3x "
                             "faster than max); fast = low effort + several chunks per "
                             f"request (~4-8x faster) (default {ANNOTATE_QUALITY})")
    parser.add_argument("--upgrade-distilled", action="store_true",
                        help="LLM tiers only: also re-annotate kNN-classified "
                             "(source='distilled') rows in scope, filling in events")
    parser.add_argument("--retry-failed", action="store_true",
                        help="--terms only: reset the failure counter on parked "
                             "terms so this run retries them")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pending scope and exit without LLM calls or writes")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    model = args.model or ANNOTATE_MODEL
    quality = args.quality or ANNOTATE_QUALITY
    # A negative row cap is never a smaller run, whichever tier reads it. On the
    # per-chunk tiers and on --terms it reaches a SQL `LIMIT ?`, where -1 means
    # NO limit; in hybrid's phase 3a it reaches a list slice, where
    # `event_ids[:-5]` is not a prefix of the density ranking at all -- it drops
    # the five LEAST dense candidates and dispatches everything else. Both
    # directions turn a flag whose whole job is to bound a run into an unbounded
    # one, so the check belongs here, at the one place every tier passes
    # through, rather than once per consumer (hybrid_chunks keeps its own for
    # callers that reach it directly). 0 is legal: it is the empty cap. What
    # each tier then does with an empty cap is the tier's own business and is
    # documented there -- distilled ignores --limit outright, and hybrid's
    # phase 1 can still spend LLM calls on seeds -- so this message promises
    # nothing about the run, only about the cap.
    if args.limit is not None and args.limit < 0:
        parser.error(f"--limit cannot be negative (got {args.limit}); use 0 for "
                     f"an empty cap, or omit it for no cap")
    # --top-dyads is the same shape against the scope subquery's own LIMIT,
    # where a negative widens the run to every chat -- which is what --all is
    # for. Resolved with `is not None` rather than `or` so an explicit 0 means
    # "no chats in scope" instead of silently falling through to the 20-chat
    # default; hybrid_chunks has always read its own 0 that way, and main was
    # the half that disagreed.
    if args.top_dyads is not None and args.top_dyads < 0:
        parser.error(f"--top-dyads cannot be negative (got {args.top_dyads}); "
                     f"use --all to annotate every chat")
    top_dyads = args.top_dyads if args.top_dyads is not None else ANNOTATE_TOP_DYADS
    # The env half of the same check, for the reason conflict_detect gives for
    # CONFLICT_MAX_LABEL_TURNS: a knob is validated where it is RESOLVED, not
    # only where a flag supplies it, or the unbounded run stays reachable
    # through the quieter of the two doors.
    #
    # Gated on the run actually BINDING it, which is the other half of that
    # same precedent (`--no-llm` does not resolve CONFLICT_BUDGET_MINUTES at
    # all, so a bad value cannot refuse a run it could not have affected).
    # `--terms` has no scope at all, `--all` omits the scope clause, and
    # distilled hardcodes `top_dyads=0` because it is corpus-wide by
    # construction -- none of the three can reach that LIMIT. Refusing them
    # would take out bare `make annotate` over a knob those runs never read,
    # and would make the message below name a remedy it had itself refused.
    if top_dyads < 0 and not args.terms and not args.all and quality != "distilled":
        raise SystemExit(
            f"ANNOTATE_TOP_DYADS cannot be negative (got {top_dyads}); SQLite "
            f"reads it as no limit at all, so a scoped run silently becomes an "
            f"--all one. Pass --all (or --top-dyads N) if that is what you want.")
    # --quality is a chunk-annotation tier and never described term curation;
    # accepting it silently here is what let its `think` cost ~11x per term.
    # Checked before --upgrade-distilled so `--terms --quality max
    # --upgrade-distilled` names the flag the user has to drop, not a
    # consequence of it.
    if args.quality and args.terms:
        parser.error("--quality does not apply to --terms (term curation has "
                     "no fidelity tiers); use ANNOTATE_TERMS_THINK / "
                     "ANNOTATE_TERMS_BATCH_SIZE to tune it")
    # hybrid is rejected alongside distilled because it IS a distilled-upgrader:
    # its phase 3 re-annotates distilled rows by construction, so the flag would
    # be asking for behavior it already has.
    if args.upgrade_distilled and (quality in ("distilled", "hybrid") or args.terms):
        parser.error("--upgrade-distilled only applies to the per-chunk LLM tiers "
                     "(--quality fast/balanced/max); distilled does no LLM upgrade "
                     "pass at all, and hybrid already upgrades distilled rows")
    if args.retry_failed and not args.terms:
        parser.error("--retry-failed only applies to --terms")
    con = _connect()
    try:
        if args.terms:
            curate_terms(con, model, limit=args.limit, dry_run=args.dry_run,
                         think=_resolve_terms_think(), workers=ANNOTATE_WORKERS,
                         batch_size=max(1, ANNOTATE_TERMS_BATCH_SIZE),
                         retry_failed=args.retry_failed)
            return
        # Chunk tiers only: --quality is theirs, and resolving it for --terms
        # above is exactly the coupling that made term curation ~11x slow.
        think, batch_size = _resolve_quality(quality, model)
        if quality == "distilled":
            # Corpus-wide by design; --all/--top-dyads/--limit don't apply.
            annotate_distill.distill_chunks(con, model, think=think,
                                            workers=ANNOTATE_WORKERS,
                                            dry_run=args.dry_run)
        elif quality == "hybrid":
            # Its kNN phases are corpus-wide like distilled's; the scope flags
            # (and --limit) reach only the deadline-gated LLM upgrade phases.
            annotate_hybrid.hybrid_chunks(
                con, model,
                think=think,
                workers=ANNOTATE_WORKERS,
                all_chats=args.all,
                top_dyads=top_dyads,
                limit=args.limit,
                dry_run=args.dry_run,
                batch_size=batch_size,
            )
        else:
            annotate_chunks(
                con, model,
                all_chats=args.all,
                top_dyads=top_dyads,
                limit=args.limit,
                dry_run=args.dry_run,
                think=think,
                batch_size=batch_size,
                workers=ANNOTATE_WORKERS,
                upgrade_distilled=args.upgrade_distilled,
            )
    finally:
        con.close()


# Above the bottom import on purpose; db.run_cli re-enters via the real module.
if __name__ == "__main__":
    run_cli(main)


# --terms is its own pipeline and lives in annotate_terms; re-exported so
# annotate.curate_terms keeps resolving. Bottom import: annotate_terms
# late-binds the prompts and helpers defined above.
from annotate_terms import curate_terms  # noqa: E402,F401
