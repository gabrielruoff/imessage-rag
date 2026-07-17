"""Batch LLM annotation CLI: sentiment/emotion/depth/event annotation of
conversation chunks into chunk_annotations, plus dyad-term curation
(--terms). One chunk per Ollama JSON-mode request, sequential, resumable
(only rows without an annotation are processed; Ctrl-C commits partial
progress). Mixing annotation models in one table is refused via the
ingest_state `annotate_model` guard, mirroring the embedding-model guard.

Usage: annotate.py [--all | --top-dyads N] [--terms] [--limit N]
                   [--model NAME] [--dry-run]
"""

import argparse
import json
import sqlite3

import ollama
import progress
from config import (
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_MODEL,
    ANNOTATE_TOP_DYADS,
    TARGET_DB,
)
from db import get_state, load_vec_extension, set_state
from function_words import EMOTION_LABELS

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

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

_TERMS_SYSTEM = """You classify a term that appears unusually often in one text-message chat.
Respond with ONLY one JSON object: {"verdict": <one of "inside_joke", "nickname",
"ordinary", "noise">}. "inside_joke" = a private/coined expression between these
people; "nickname" = a name they call a person, pet, or place; "ordinary" = a
normal word that just happens to be frequent; "noise" = a typo, artifact, or
meaningless fragment."""

_RETRY_NUDGE = (
    "Respond with only the JSON object described in the instructions -- "
    "no prose, no code fences, nothing else."
)


def _connect():
    con = sqlite3.connect(DB_PATH)
    load_vec_extension(con)
    return con


def _like_escape(value):
    r"""Escape LIKE wildcards for use with ESCAPE '\'."""
    return (value.replace("\\", "\\\\")
                 .replace("%", "\\%")
                 .replace("_", "\\_"))


def _ask_json(messages, model):
    """chat_once(format='json') -> parsed dict, retrying once with a
    corrective nudge on a malformed/non-object response; None if the retry
    also fails (caller skips -- the row is retried on the next run)."""
    for _ in range(2):
        # Resolved through the ollama module at call time (patch seam).
        reply = ollama.chat_once(messages, model=model, format="json")
        content = reply.get("content") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": _RETRY_NUDGE},
        ]
    return None


# ---------------------------------------------------------------------------
# Chunk annotation
# ---------------------------------------------------------------------------

def _pending_chunks(con, all_chats, top_dyads, limit):
    sql = """
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN chunk_annotations a ON a.chunk_id = c.id
        WHERE a.chunk_id IS NULL
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


def annotate_chunks(con, model, all_chats=False, top_dyads=ANNOTATE_TOP_DYADS,
                    limit=None, dry_run=False):
    pending = _pending_chunks(con, all_chats, top_dyads, limit)
    scope = "all chats" if all_chats else f"top {top_dyads} chats by message count"
    print(f"{len(pending)} chunks pending annotation ({scope}, model {model})")
    if dry_run:
        preview = ", ".join(str(cid) for cid, _ in pending[:10])
        print(f"--dry-run: first chunk ids: {preview or '(none)'}")
        return 0
    if not pending:
        return 0
    _check_model_guard(con, model)

    n_done = n_skipped = 0
    try:
        with progress.track("Annotating chunks", total=len(pending),
                            header=False) as p:
            for chunk_id, text in pending:
                data = _ask_json(
                    [{"role": "system", "content": _ANNOTATE_SYSTEM},
                     {"role": "user", "content": text}],
                    model,
                )
                if data is None:
                    n_skipped += 1
                    print(f"  chunk {chunk_id}: unparseable response, skipped "
                          f"(retried next run)")
                    p.advance()
                    continue
                valence, emotions, depth, events = _validate_annotation(data)
                _resolve_event_message_ids(con, chunk_id, events)
                con.execute(
                    "INSERT OR REPLACE INTO chunk_annotations"
                    "(chunk_id, valence, emotions_json, depth, events_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, valence, json.dumps(emotions), depth, json.dumps(events)),
                )
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

def curate_terms(con, model, limit=None, dry_run=False):
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

    n_done = n_skipped = 0
    try:
        with progress.track("Curating terms", total=len(rows), header=False) as p:
            for chat_id, term in rows:
                # ESCAPE: a term is tokenizer output so '%'/'_' are vanishingly
                # unlikely, but an unescaped one would silently match everything.
                examples = [r[0] for r in con.execute(
                    "SELECT text FROM messages "
                    "WHERE chat_id = ? AND text LIKE '%' || ? || '%' ESCAPE '\\' LIMIT 3",
                    (chat_id, _like_escape(term)),
                )]
                user = f'Term: "{term}"\n\nExample messages from this chat:\n' + (
                    "\n".join(f"- {t[:200]}" for t in examples) or "(none found)"
                )
                data = _ask_json(
                    [{"role": "system", "content": _TERMS_SYSTEM},
                     {"role": "user", "content": user}],
                    model,
                )
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
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pending scope and exit without LLM calls or writes")
    args = parser.parse_args(argv)

    model = args.model or ANNOTATE_MODEL
    con = _connect()
    try:
        if args.terms:
            curate_terms(con, model, limit=args.limit, dry_run=args.dry_run)
        else:
            annotate_chunks(
                con, model,
                all_chats=args.all,
                top_dyads=args.top_dyads or ANNOTATE_TOP_DYADS,
                limit=args.limit,
                dry_run=args.dry_run,
            )
    finally:
        con.close()


if __name__ == "__main__":
    main()
