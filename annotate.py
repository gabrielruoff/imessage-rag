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
over ANNOTATE_WORKERS concurrent requests (~2x with a parallel Ollama).

--terms is a separate pipeline, not a --quality tier: it curates dyad_terms
verdicts, batches ANNOTATE_TERMS_BATCH_SIZE terms per request, and runs with
its own think (ANNOTATE_TERMS_THINK, default false -- reasoning about a
one-word verdict is ~11x latency for the same answer). A term that fails
curation ANNOTATE_TERMS_MAX_FAILURES times is parked in
dyad_terms.curate_failures rather than retried by every later run;
--retry-failed unparks.

Usage (chunks): annotate.py [--all | --top-dyads N] [--limit N]
                   [--model NAME] [--quality distilled|fast|balanced|max]
                   [--upgrade-distilled] [--dry-run]
Usage (terms):  annotate.py --terms [--limit N] [--model NAME]
                   [--retry-failed] [--dry-run]
--terms and --quality/--upgrade-distilled are mutually exclusive.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import annotate_distill
import ollama
import progress
import schema
from config import (
    ANNOTATE_BATCH_SIZE,
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_MODEL,
    ANNOTATE_QUALITY,
    ANNOTATE_TERMS_BATCH_SIZE,
    ANNOTATE_TERMS_MAX_FAILURES,
    ANNOTATE_TERMS_THINK,
    ANNOTATE_THINK,
    ANNOTATE_TIMEOUT,
    ANNOTATE_TOP_DYADS,
    ANNOTATE_WORKERS,
    TARGET_DB,
)
from db import connect_default, get_state, run_cli, set_state
from function_words import EMOTION_LABELS

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# --quality presets: (think, batch_size). think is Ollama's reasoning-effort
# control ('low' works on effort-level models like gpt-oss and is tolerated
# by gemma4; models that reject it fall back to no control mid-run). None
# omits the flag: the model's default reasoning. Measured on gpt-oss:20b:
# max ~8.6s/chunk, balanced ~3s, fast ~1-2s. 'distilled' uses its tuple only
# for the LLM seed pass; the rest of the corpus is kNN-classified.
QUALITY_PRESETS = {
    "max": (None, 1),
    "balanced": ("low", 1),
    "fast": ("low", 8),
    "distilled": ("low", 1),
}

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


def _like_escape(value):
    r"""Escape LIKE wildcards for use with ESCAPE '\'."""
    return (value.replace("\\", "\\\\")
                 .replace("%", "\\%")
                 .replace("_", "\\_"))


def _pooled(items, request_fn, workers):
    """Yield (item, request_fn(item)) with up to `workers` requests in
    flight, in completion order. Only request_fn runs on worker threads --
    callers keep validation, SQLite writes and progress on the main thread
    (mirrors chunking.embed_chunks). By contract `items` is consumed on the
    caller's thread (the driving loop below iterates it), so a generator that
    runs SQL per item -- _term_groups does -- keeps its sqlite affinity.
    workers<=1 is a plain sequential loop.
    On interrupt, not-yet-started requests are cancelled; in-flight ones
    finish server-side but their results are dropped."""
    if workers <= 1:
        for item in items:
            yield item, request_fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        try:
            for item in items:
                futures[pool.submit(request_fn, item)] = item
                if len(futures) >= workers * 2:
                    done = next(as_completed(futures))
                    yield futures.pop(done), done.result()
            while futures:
                done = next(as_completed(futures))
                yield futures.pop(done), done.result()
        finally:
            for f in futures:
                f.cancel()


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
    """Refuse to mix annotation models within one artifact (same shape as
    chunking.build_vec_table's embedding guard). `predicate` narrows the count
    to the rows this model actually produced (dyad_terms holds uncurated rows
    too). Records the chosen model when proceeding."""
    prev = get_state(con, "annotate_model")
    if prev and prev != model:
        n = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {predicate}").fetchone()[0]
        if n:
            print(
                f"{table} holds {n} rows annotated by {prev}, but the "
                f"model is now {model}. Mixing annotation models skews every "
                f"downstream aggregate -- clear {table} (or reset_db.py) "
                f"to switch, or rerun with --model {prev}."
            )
            raise SystemExit(1)
    set_state(con, "annotate_model", model)
    con.commit()


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


def _annotate_group(group, model, think):
    """All LLM work for one group of (chunk_id, text) rows: a batched request
    for a multi-chunk group -- retrying any id it missed with a single-chunk
    request -- or just the single request for a singleton. Runs on worker
    threads; returns {chunk_id: annotation dict | None}."""
    batched = _annotate_batch(group, model, think) if len(group) > 1 else {}
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


def annotate_chunks(con, model, all_chats=False, top_dyads=ANNOTATE_TOP_DYADS,
                    limit=None, dry_run=False, think=None, batch_size=1,
                    workers=1, upgrade_distilled=False):
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

    groups = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
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
# Dyad-term curation (--terms)
# ---------------------------------------------------------------------------

def _term_prompt(con, chat_id, term):
    """Build the curation prompt for one term. Main-thread SQL because sqlite
    connections aren't shared across threads; that costs nothing -- the LIKE
    rides idx_messages_chat_date and measures ~2ms at 350k messages, against
    seconds for the LLM call it feeds. ESCAPE: a term is tokenizer output so
    '%'/'_' are vanishingly unlikely, but an unescaped one would silently
    match everything."""
    examples = [r[0] for r in con.execute(
        "SELECT text FROM messages "
        "WHERE chat_id = ? AND text LIKE '%' || ? || '%' ESCAPE '\\' LIMIT 3",
        (chat_id, _like_escape(term)),
    )]
    return f'Term: "{term}"\n\nExample messages from this chat:\n' + (
        "\n".join(f"- {t[:200]}" for t in examples) or "(none found)"
    )


def _curate_one(prompt, model, think):
    """One single-term request -> (verdict, reason). reason is None on success
    and otherwise says why there is no verdict: an unparseable reply and a
    well-formed reply naming a bogus verdict are different failures (a bad
    prompt vs a bad model) and used to be reported identically as `None`."""
    data = ollama.chat_once_json(
        [{"role": "system", "content": _TERMS_SYSTEM},
         {"role": "user", "content": prompt}],
        model=model, think=think, timeout=ANNOTATE_TIMEOUT,
        retry_nudge=_TERMS_NUDGE,
    )
    if not isinstance(data, dict):
        return None, "no JSON object in reply (and the retry nudge also failed)"
    verdict = data.get("verdict")
    if verdict not in TERM_VERDICTS:
        return None, f"reply had no valid verdict (got {verdict!r})"
    return verdict, None


def _curate_batch(group, model, think):
    """One JSON-mode request classifying several terms at once; returns
    {index: verdict} for the entries the reply actually covered -- the caller
    retries the rest individually. Keyed by position, not by the term text:
    positions can't collide or need escaping."""
    user = "\n\n".join(f"### term {i}\n{prompt}"
                       for i, (_chat_id, _term, prompt) in enumerate(group))
    data = ollama.chat_once_json(
        [{"role": "system", "content": _TERMS_BATCH_SYSTEM},
         {"role": "user", "content": user}],
        model=model, think=think, timeout=ANNOTATE_TIMEOUT,
        retry_nudge=_TERMS_BATCH_NUDGE,
    )
    if not isinstance(data, dict):
        return {}
    return {i: data.get(str(i)) for i in range(len(group))
            if data.get(str(i)) in TERM_VERDICTS}


def _curate_group(group, model, think):
    """All LLM work for one group of (chat_id, term, prompt) items: a batched
    request for a multi-term group -- falling back to a single-term request
    for any entry it missed or mangled -- or just the single request for a
    singleton. Once _GROUP_GIVEUP fallbacks in a row have failed, the rest of
    the group is abandoned unattempted (reason `_ABANDONED`) rather than
    fanned out into more doomed requests. Runs on worker threads; returns
    {index: (verdict, reason)}."""
    batched = _curate_batch(group, model, think) if len(group) > 1 else {}
    out = {}
    misses = 0
    for i, (_chat_id, _term, prompt) in enumerate(group):
        if i in batched:
            out[i] = (batched[i], None)
            misses = 0
        elif misses >= _GROUP_GIVEUP:
            out[i] = (None, _ABANDONED)
        else:
            verdict, reason = _curate_one(prompt, model, think)
            out[i] = (verdict, reason)
            misses = 0 if verdict is not None else misses + 1
    return out


def _term_groups(con, rows, batch_size):
    """Lazily yield batches of (chat_id, term, prompt), never spanning chats:
    a batch shares one system prompt that says 'in one text-message chat', and
    mixing chats would make the examples contradict it. Lazy so the per-term
    prompt SQL stays interleaved with the LLM calls (rows arrive ordered by
    chat_id, so a plain run-length walk groups them)."""
    group = []
    for chat_id, term in rows:
        if group and group[0][0] != chat_id:
            yield group
            group = []
        group.append((chat_id, term, _term_prompt(con, chat_id, term)))
        if len(group) >= batch_size:
            yield group
            group = []
    if group:
        yield group


def _ensure_curate_failures(con):
    """dyad_terms.curate_failures postdates DBs already in the wild, and
    --terms is reachable without re-running ingest.py -- which is what
    normally applies migrations. Self-heal through the canonical migration
    (idempotent) instead of dying on 'no such column'.

    Scope note: create_schema does more than the one ALTER -- it re-runs the
    DDL scripts, recreates the FTS triggers and rescans reactions for the
    old guid prefix. That's a one-off (the PRAGMA gate means the next run
    skips it entirely) and worth not having a second, drifting copy of the
    migration here; schema.py stays the only place dyad_terms' DDL lives."""
    cols = {row[1] for row in con.execute("PRAGMA table_info(dyad_terms)")}
    if "curate_failures" not in cols:
        schema.create_schema(con)
        con.commit()


def _report_parked(parked, max_failures):
    """Name the terms this run parked. The header printed the pre-existing
    parked count, but a long run scrolls it away -- and a term going quiet
    forever should not be something you have to query SQL to discover."""
    if not parked:
        return
    shown = ", ".join(parked[:10]) + (" ..." if len(parked) > 10 else "")
    print(f"  parked {len(parked)} terms at {max_failures} failures "
          f"({shown}); --retry-failed to retry them")


def curate_terms(con, model, limit=None, dry_run=False, think=False, workers=1,
                 batch_size=1, max_failures=ANNOTATE_TERMS_MAX_FAILURES,
                 retry_failed=False):
    _ensure_curate_failures(con)
    if retry_failed and dry_run:
        # --dry-run promises no writes, and unparking during a *preview* is
        # worse than a stray write: the preview spends the parking state, so
        # the next bare --terms run re-attempts those terms at full price --
        # the loop this pipeline exists to close.
        (n_would,) = con.execute(
            "SELECT COUNT(*) FROM dyad_terms "
            "WHERE llm_verdict IS NULL AND curate_failures > 0").fetchone()
        print(f"--retry-failed: would unpark {n_would} terms (--dry-run: not written)")
    elif retry_failed:
        n_reset = con.execute(
            "UPDATE dyad_terms SET curate_failures = 0 "
            "WHERE llm_verdict IS NULL AND curate_failures > 0").rowcount
        con.commit()
        print(f"--retry-failed: unparked {n_reset} terms")
    # Parked terms (curate_failures at the cap) are deliberately NOT pending:
    # a term that has already failed max_failures times -- each failure costing
    # two requests, since chat_once_json retries once -- would otherwise be
    # re-attempted at full price by every run forever.
    n_parked = con.execute(
        "SELECT COUNT(*) FROM dyad_terms "
        "WHERE llm_verdict IS NULL AND curate_failures >= ?", (max_failures,)
    ).fetchone()[0]
    # --retry-failed lifts the ceiling: after the unpark above every term sits
    # at 0 anyway, and under --dry-run (which skipped the unpark) the preview
    # must still show the scope the real run would take.
    ceiling = float("inf") if retry_failed else max_failures
    sql = ("SELECT chat_id, term FROM dyad_terms "
           "WHERE llm_verdict IS NULL AND curate_failures < ? "
           "ORDER BY chat_id, term")
    params = [ceiling]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = con.execute(sql, params).fetchall()
    settings = f"model {model}, think {think!r}"
    if batch_size > 1:
        settings += f", batch {batch_size}"
    if workers > 1:
        settings += f", workers {workers}"
    print(f"{len(rows)} dyad terms pending curation ({settings})")
    if n_parked and not retry_failed:
        print(f"  ({n_parked} parked after {max_failures} failed attempts; "
              f"--retry-failed to retry them)")
    if dry_run:
        preview = ", ".join(f"{chat_id}:{term}" for chat_id, term in rows[:10])
        print(f"--dry-run: first terms: {preview or '(none)'}")
        return 0
    if not rows:
        return 0
    # Verdicts are model-derived and survive dyad_terms rebuilds, exactly like
    # chunk_annotations rows -- so they get the same anti-mixing guard.
    _check_model_guard(con, model, "dyad_terms", "llm_verdict IS NOT NULL")

    n_done = n_skipped = 0
    parked = []
    try:
        with progress.track("Curating terms", total=len(rows), header=False) as p:
            for group, results in _pooled(
                    _term_groups(con, rows, batch_size),
                    lambda g: _curate_group(g, model, think), workers):
                for i, (chat_id, term, _prompt) in enumerate(group):
                    verdict, reason = results[i]
                    if verdict is None:
                        n_skipped += 1
                        # An abandoned term never got its own request, so it
                        # is still pending, not failing -- don't charge it.
                        # Every other miss did get a dedicated request (the
                        # batch fallback), so the counter only ever records
                        # failures a single term actually earned.
                        if reason is not _ABANDONED:
                            con.execute(
                                "UPDATE dyad_terms SET curate_failures = "
                                "curate_failures + 1 WHERE chat_id = ? AND term = ?",
                                (chat_id, term),
                            )
                            (n_fail,) = con.execute(
                                "SELECT curate_failures FROM dyad_terms "
                                "WHERE chat_id = ? AND term = ?",
                                (chat_id, term)).fetchone()
                            if n_fail >= max_failures:
                                parked.append(f"{chat_id}:{term}")
                        print(f"  term {chat_id}:{term}: {reason}, skipped")
                        if n_done == 0 and n_skipped >= _TERMS_ABORT_AFTER:
                            # Nothing committed yet (the cadence below waits
                            # for a first success), so this really does undo
                            # every counter this run wrote.
                            con.rollback()
                            raise SystemExit(
                                f"Aborting: the first {n_skipped} terms all failed "
                                f"and none succeeded -- that is a configuration "
                                f"problem, not {n_skipped} bad terms, so no failure "
                                f"counters were recorded. Check that model {model!r} "
                                f"is pulled and answers JSON; if it is gpt-oss, "
                                f"think={think!r} yields empty replies -- set "
                                f"ANNOTATE_TERMS_THINK=low."
                            )
                    else:
                        con.execute(
                            "UPDATE dyad_terms SET llm_verdict = ? "
                            "WHERE chat_id = ? AND term = ?",
                            (verdict, chat_id, term),
                        )
                        n_done += 1
                    p.advance()
                    # Both branches write, so the cadence counts both -- but
                    # not before a first success, so the abort above can roll
                    # back cleanly whatever ANNOTATE_COMMIT_EVERY is set to.
                    if n_done and (n_done + n_skipped) % ANNOTATE_COMMIT_EVERY == 0:
                        con.commit()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- committed {n_done} verdicts "
              f"({n_skipped} skipped); rerun to resume.")
        _report_parked(parked, max_failures)
        return n_done
    con.commit()
    print(f"Curated {n_done} terms ({n_skipped} skipped)")
    _report_parked(parked, max_failures)
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_quality(quality):
    """Preset name -> (think, batch_size), with ANNOTATE_THINK /
    ANNOTATE_BATCH_SIZE env overrides applied on top."""
    try:
        think, batch_size = QUALITY_PRESETS[quality]
    except KeyError:
        raise SystemExit(
            f"unknown ANNOTATE_QUALITY {quality!r} "
            f"(choose from {', '.join(sorted(QUALITY_PRESETS))})"
        ) from None
    if ANNOTATE_THINK:
        think = _parse_think(ANNOTATE_THINK)
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


def main(argv=None):
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
                        help="process at most N rows this run")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model (default {ANNOTATE_MODEL})")
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default=None,
                        help="speed/quality tradeoff: distilled = LLM seed sample + "
                             "kNN classify the rest from embeddings (~1 min corpus-"
                             "wide once seeded; no events); max = one chunk per "
                             "request at the model's default reasoning effort; "
                             "balanced = low reasoning effort (~3x faster than max); "
                             "fast = low effort + several chunks per request "
                             f"(~4-8x faster) (default {ANNOTATE_QUALITY})")
    parser.add_argument("--upgrade-distilled", action="store_true",
                        help="LLM tiers only: also re-annotate kNN-classified "
                             "(source='distilled') rows in scope, filling in events")
    parser.add_argument("--retry-failed", action="store_true",
                        help="--terms only: reset the failure counter on parked "
                             "terms so this run retries them")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pending scope and exit without LLM calls or writes")
    args = parser.parse_args(argv)

    model = args.model or ANNOTATE_MODEL
    quality = args.quality or ANNOTATE_QUALITY
    # --quality is a chunk-annotation tier and never described term curation;
    # accepting it silently here is what let its `think` cost ~11x per term.
    # Checked before --upgrade-distilled so `--terms --quality max
    # --upgrade-distilled` names the flag the user has to drop, not a
    # consequence of it.
    if args.quality and args.terms:
        parser.error("--quality does not apply to --terms (term curation has "
                     "no fidelity tiers); use ANNOTATE_TERMS_THINK / "
                     "ANNOTATE_TERMS_BATCH_SIZE to tune it")
    if args.upgrade_distilled and (quality == "distilled" or args.terms):
        parser.error("--upgrade-distilled only applies to the LLM chunk tiers "
                     "(--quality fast/balanced/max)")
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
        think, batch_size = _resolve_quality(quality)
        if quality == "distilled":
            # Corpus-wide by design; --all/--top-dyads/--limit don't apply.
            annotate_distill.distill_chunks(con, model, think=think,
                                            workers=ANNOTATE_WORKERS,
                                            dry_run=args.dry_run)
        else:
            annotate_chunks(
                con, model,
                all_chats=args.all,
                top_dyads=args.top_dyads or ANNOTATE_TOP_DYADS,
                limit=args.limit,
                dry_run=args.dry_run,
                think=think,
                batch_size=batch_size,
                workers=ANNOTATE_WORKERS,
                upgrade_distilled=args.upgrade_distilled,
            )
    finally:
        con.close()


if __name__ == "__main__":
    run_cli(main)
