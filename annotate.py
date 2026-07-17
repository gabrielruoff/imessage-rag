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

Usage: annotate.py [--all | --top-dyads N] [--terms] [--limit N]
                   [--model NAME] [--quality distilled|fast|balanced|max]
                   [--upgrade-distilled] [--dry-run]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import annotate_distill
import ollama
import progress
from config import (
    ANNOTATE_BATCH_SIZE,
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_MODEL,
    ANNOTATE_QUALITY,
    ANNOTATE_THINK,
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

_TERMS_SYSTEM = """You classify a term that appears unusually often in one text-message chat.
Respond with ONLY one JSON object: {"verdict": <one of "inside_joke", "nickname",
"ordinary", "noise">}. "inside_joke" = a private/coined expression between these
people; "nickname" = a name they call a person, pet, or place; "ordinary" = a
normal word that just happens to be frequent; "noise" = a typo, artifact, or
meaningless fragment."""

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
    (mirrors chunking.embed_chunks). workers<=1 is a plain sequential loop.
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
        model=model, think=think,
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
                model=model, think=think,
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
    """Build the curation prompt for one term (main-thread SQL: the LIKE
    scan is the expensive bit, but sqlite connections aren't shared across
    threads). ESCAPE: a term is tokenizer output so '%'/'_' are vanishingly
    unlikely, but an unescaped one would silently match everything."""
    examples = [r[0] for r in con.execute(
        "SELECT text FROM messages "
        "WHERE chat_id = ? AND text LIKE '%' || ? || '%' ESCAPE '\\' LIMIT 3",
        (chat_id, _like_escape(term)),
    )]
    return f'Term: "{term}"\n\nExample messages from this chat:\n' + (
        "\n".join(f"- {t[:200]}" for t in examples) or "(none found)"
    )


def curate_terms(con, model, limit=None, dry_run=False, think=None, workers=1):
    sql = ("SELECT chat_id, term FROM dyad_terms WHERE llm_verdict IS NULL "
           "ORDER BY chat_id, term")
    params = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = con.execute(sql, params).fetchall()
    print(f"{len(rows)} dyad terms pending curation (model {model})")
    if dry_run:
        preview = ", ".join(f"{chat_id}:{term}" for chat_id, term in rows[:10])
        print(f"--dry-run: first terms: {preview or '(none)'}")
        return 0
    if not rows:
        return 0
    # Verdicts are model-derived and survive dyad_terms rebuilds, exactly like
    # chunk_annotations rows -- so they get the same anti-mixing guard.
    _check_model_guard(con, model, "dyad_terms", "llm_verdict IS NOT NULL")

    # Lazy: prompts are built on the main thread as _pooled pulls items, so
    # the LIKE scans stay interleaved with the LLM calls instead of running
    # all up front.
    items = ((chat_id, term, _term_prompt(con, chat_id, term))
             for chat_id, term in rows)

    n_done = n_skipped = 0
    try:
        with progress.track("Curating terms", total=len(rows), header=False) as p:
            for (chat_id, term, _user), data in _pooled(
                    items,
                    lambda it: ollama.chat_once_json(
                        [{"role": "system", "content": _TERMS_SYSTEM},
                         {"role": "user", "content": it[2]}],
                        model=model, think=think),
                    workers):
                verdict = data.get("verdict") if isinstance(data, dict) else None
                if verdict not in TERM_VERDICTS:
                    n_skipped += 1
                    print(f"  term {chat_id}:{term}: invalid verdict {verdict!r}, skipped")
                    p.advance()
                    continue
                con.execute(
                    "UPDATE dyad_terms SET llm_verdict = ? WHERE chat_id = ? AND term = ?",
                    (verdict, chat_id, term),
                )
                n_done += 1
                p.advance()
                if n_done % ANNOTATE_COMMIT_EVERY == 0:
                    con.commit()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- committed {n_done} verdicts "
              f"({n_skipped} skipped); rerun to resume.")
        return n_done
    con.commit()
    print(f"Curated {n_done} terms ({n_skipped} skipped)")
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
        think = {"true": True, "false": False}.get(ANNOTATE_THINK, ANNOTATE_THINK)
    if ANNOTATE_BATCH_SIZE:
        batch_size = ANNOTATE_BATCH_SIZE
    return think, max(1, batch_size)


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
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pending scope and exit without LLM calls or writes")
    args = parser.parse_args(argv)

    model = args.model or ANNOTATE_MODEL
    quality = args.quality or ANNOTATE_QUALITY
    think, batch_size = _resolve_quality(quality)
    if args.upgrade_distilled and (quality == "distilled" or args.terms):
        parser.error("--upgrade-distilled only applies to the LLM chunk tiers "
                     "(--quality fast/balanced/max)")
    con = _connect()
    try:
        if args.terms:
            curate_terms(con, model, limit=args.limit, dry_run=args.dry_run,
                         think=think, workers=ANNOTATE_WORKERS)
        elif quality == "distilled":
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
