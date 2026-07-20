"""
Derived-artifact stages of the ingest pipeline: group ingested messages into
conversation chunks, then embed the chunks via local Ollama into the
sqlite-vec chunk_vectors table.

Everything here operates only on imessage_analysis.db (the target DB), never
the chat.db snapshot. Chunk building and chunk embedding live together
deliberately: they share the chunks lifecycle (an incremental run's
tail-chunk rebuild deletes rows from both `chunks` and `chunk_vectors`).
ingest.py runs both stages after source ingestion and re-exports them, so
`ingest.build_chunks` etc. keep working.
"""

import datetime
import http.client
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import sqlite_vec

from config import (
    CHUNK_GAP_MINUTES,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_MESSAGES,
    EMBED_BATCH_SIZE,
    EMBED_COMMIT_EVERY,
    EMBED_MAX_WORKERS,
    EMBED_MODEL,
)
import progress
from db import CONTACT_LABEL_SQL, get_state, iso_to_day, load_vec_extension, set_state

# ---------------------------------------------------------------------------
# Conversation chunks
# ---------------------------------------------------------------------------

# A message contributes a line to a chunk if it has text or an attachment;
# system events with neither stay chunk_id NULL and never trigger rebuilds.
_RENDERABLE = "(m.text IS NOT NULL OR m.has_attachment = 1)"

_MAX_HEADER_NAMES = 6


def _chunk_header(who, first_date):
    return f"Conversation with {who} on {first_date[:10]}:"


def _render_line(labels, att_types, msg):
    msg_id, is_from_me, handle_id, text, _date, _has_attachment = msg
    sender = "Me" if is_from_me else (labels.get(handle_id) or "Unknown")
    if text:
        return f"{sender}: {text}"
    kinds = att_types.get(msg_id)
    return f"{sender}: [attachment: {', '.join(kinds)}]" if kinds else f"{sender}: [attachment]"


def _chat_label(tgt, labels, chat_id):
    """Header name for a chat: its display_name, else its participants."""
    row = tgt.execute("SELECT display_name FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row and row[0]:
        return row[0]
    names = [labels[r[0]] for r in tgt.execute(
        "SELECT handle_id FROM chat_participants WHERE chat_id = ? ORDER BY handle_id",
        (chat_id,)) if labels.get(r[0])]
    if not names:
        return "unknown"
    label = ", ".join(names[:_MAX_HEADER_NAMES])
    if len(names) > _MAX_HEADER_NAMES:
        label += f" +{len(names) - _MAX_HEADER_NAMES} others"
    return label


def _flush_chunk(tgt, chat_id, who, chunk_msgs):
    """Write one chunk row and stamp chunk_id on its messages. A message
    carried into the next chunk by the size-split overlap keeps the first
    chunk_id it was assigned (the `chunk_id IS NULL` guard)."""
    first, last = chunk_msgs[0][0], chunk_msgs[-1][0]
    text = "\n".join([_chunk_header(who, first[4])] + [line for _, line in chunk_msgs])
    cur = tgt.execute(
        "INSERT INTO chunks(chat_id, first_message_id, last_message_id, "
        " first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, first[0], last[0], first[4], last[4], len(chunk_msgs), text),
    )
    ids = [m[0] for m, _ in chunk_msgs]
    placeholders = ",".join("?" * len(ids))
    tgt.execute(
        f"UPDATE messages SET chunk_id = ? WHERE chunk_id IS NULL AND id IN ({placeholders})",
        [cur.lastrowid] + ids,
    )


def build_chunks(tgt: sqlite3.Connection, gap_minutes=CHUNK_GAP_MINUTES,
                 max_chars=CHUNK_MAX_CHARS, overlap_messages=CHUNK_OVERLAP_MESSAGES):
    """Group not-yet-chunked messages into conversation chunks: per chat,
    ordered by (date, id), split when the gap between two messages exceeds
    gap_minutes (a session boundary) or the composed text would exceed
    max_chars (a size split, which carries the last overlap_messages messages
    into the next chunk for continuity -- unless the carried text alone would
    already overflow max_chars, in which case no overlap is carried so a
    chunk can never consist solely of repeated messages). Chunk text is a
    context header plus
    one "{sender}: {text}" line per message, so speakers and participants are
    part of what gets embedded.

    Incremental behavior: a chat that received new messages has its final
    chunk deleted and rebuilt from that chunk's first message, so a resumed
    conversation extends one coherent chunk instead of splitting at the
    ingest boundary. Chunks and their vectors are derived artifacts -- this
    is the one deletion in the pipeline, and it never touches source-derived
    rows. (If messages ever arrive with dates *earlier* than already-chunked
    ones, they form their own chunk out of id order -- still a valid chunk,
    just not merged into its older neighbors.)
    """
    pending_chats = [r[0] for r in tgt.execute(f"""
        SELECT DISTINCT m.chat_id FROM messages m
        WHERE m.chunk_id IS NULL AND m.chat_id IS NOT NULL
          AND m.date IS NOT NULL AND {_RENDERABLE}
    """)]
    if not pending_chats:
        print("Chunks up to date (0 built)")
        return 0

    labels = {row[0]: row[1] for row in tgt.execute(f"""
        SELECT h.id, {CONTACT_LABEL_SQL} FROM handles h
        LEFT JOIN contacts c ON c.id = h.contact_id
    """)}
    has_vec_table = tgt.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone() is not None

    gap = datetime.timedelta(minutes=gap_minutes)
    n_chunks_before = tgt.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_msgs = 0
    with progress.track("Building chunks", total=len(pending_chats)) as p:
        for chat_id in pending_chats:
            # Rebuild the chat's tail chunk so the pending messages extend it.
            # The rebuild must re-include the tail's overlap prefix -- messages
            # carried into it by a size split keep their *first* chunk's id, so
            # `chunk_id IS NULL` alone would miss them; `id >= first_message_id`
            # picks up exactly those (they stay stamped to their original chunk
            # via _flush_chunk's NULL guard).
            tail = tgt.execute(
                "SELECT id, first_message_id FROM chunks "
                "WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
                (chat_id,)).fetchone()
            rebuild_cond, rebuild_params = "m.chunk_id IS NULL", []
            if tail:
                if has_vec_table:
                    tgt.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (tail[0],))
                tgt.execute("UPDATE messages SET chunk_id = NULL WHERE chunk_id = ?",
                            (tail[0],))
                tgt.execute("DELETE FROM chunks WHERE id = ?", (tail[0],))
                rebuild_cond = "(m.chunk_id IS NULL OR m.id >= ?)"
                rebuild_params = [tail[1]]

            rows = tgt.execute(f"""
                SELECT m.id, m.is_from_me, m.handle_id, m.text, m.date, m.has_attachment
                FROM messages m
                WHERE m.chat_id = ? AND {rebuild_cond} AND m.date IS NOT NULL
                  AND {_RENDERABLE}
                ORDER BY m.date, m.id
            """, [chat_id] + rebuild_params).fetchall()
            if not rows:
                p.advance()
                continue

            att_types = {}
            for msg_id, kind in tgt.execute(f"""
                SELECT a.message_id, COALESCE(a.mime_type, a.transfer_name, 'file')
                FROM attachments a JOIN messages m ON m.id = a.message_id
                WHERE m.chat_id = ? AND {rebuild_cond}
            """, [chat_id] + rebuild_params):
                att_types.setdefault(msg_id, []).append(kind)

            who = _chat_label(tgt, labels, chat_id)
            chunk_msgs, cur_len, prev_dt = [], 0, None
            for msg in rows:
                line = _render_line(labels, att_types, msg)
                dt = datetime.datetime.fromisoformat(msg[4])
                if chunk_msgs and dt - prev_dt > gap:
                    _flush_chunk(tgt, chat_id, who, chunk_msgs)
                    chunk_msgs, cur_len = [], 0  # session boundary: no overlap carried
                elif chunk_msgs and cur_len + len(line) + 1 > max_chars:
                    _flush_chunk(tgt, chat_id, who, chunk_msgs)
                    chunk_msgs = chunk_msgs[-overlap_messages:] if overlap_messages else []
                    cur_len = sum(len(li) + 1 for _, li in chunk_msgs)
                    if cur_len + len(line) + 1 > max_chars:
                        # Oversized overlap (very long messages): carrying it would
                        # immediately overflow again and flush a chunk made only of
                        # already-chunked repeats -- start clean instead.
                        chunk_msgs, cur_len = [], 0
                chunk_msgs.append((msg, line))
                cur_len += len(line) + 1
                prev_dt = dt
            if chunk_msgs:
                _flush_chunk(tgt, chat_id, who, chunk_msgs)
            n_msgs += len(rows)
            p.advance()

    tgt.commit()
    n_built = tgt.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] - n_chunks_before
    print(f"Built {n_built} conversation chunks covering {n_msgs} messages "
          f"({len(pending_chats)} chats)")
    return n_built


# ---------------------------------------------------------------------------
# Embeddings (sqlite-vec + Ollama)
# ---------------------------------------------------------------------------

def build_vec_table(tgt: sqlite3.Connection, dim: int):
    """Create chunk_vectors sized to the model's (detected) embedding
    dimension. Refuses to proceed if existing vectors were built by a
    different model or at a different dimension -- mixing embedding spaces
    silently corrupts search. Also drops the legacy per-message
    message_vectors table (replaced by chunk-level embeddings)."""
    load_vec_extension(tgt)
    tgt.execute("DROP TABLE IF EXISTS message_vectors")

    existing = tgt.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone()
    prev_model = get_state(tgt, "embed_model")
    if existing:
        match = re.search(r"FLOAT\[(\d+)\]", existing[0], re.IGNORECASE)
        existing_dim = int(match.group(1)) if match else None
        if existing_dim != dim or (prev_model and prev_model != EMBED_MODEL):
            n_vectors = tgt.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            if n_vectors:
                raise SystemExit(
                    f"chunk_vectors holds {n_vectors} vectors from "
                    f"{prev_model or 'an unknown model'} (dim {existing_dim}), but "
                    f"EMBED_MODEL is now {EMBED_MODEL} (dim {dim}). Mixing embedding "
                    f"spaces corrupts search -- run reset_db.py and re-ingest."
                )
            tgt.execute("DROP TABLE chunk_vectors")

    # chat_id / first_day / last_day are vec0 *metadata columns*: sqlite-vec
    # (>= 0.1.6) applies WHERE constraints on them inside the KNN scan, so
    # scoped/date-bounded semantic search needs no over-fetch + post-filter.
    # Days are integers (db.iso_to_day) -- TEXT metadata longer than 12 bytes
    # is slow to filter on. cosine, not the L2 default: gemma4's mean-pooled
    # embeddings aren't length-normalized.
    tgt.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}] distance_metric=cosine,
            chat_id INTEGER,
            first_day INTEGER,
            last_day INTEGER
        )
    """)
    set_state(tgt, "embed_model", EMBED_MODEL)
    set_state(tgt, "embed_dim", dim)


def embed_chunks(
    tgt: sqlite3.Connection,
    min_chars=CHUNK_MIN_CHARS,
    batch_size=EMBED_BATCH_SIZE,
    max_workers=EMBED_MAX_WORKERS,
    batch_commit_every=EMBED_COMMIT_EVERY,
):
    rows = tgt.execute("""
        SELECT c.id, c.chat_id, c.first_date, c.last_date, c.text
        FROM chunks c
        LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
        WHERE v.chunk_id IS NULL
          AND length(c.text) >= ?
    """, (min_chars,)).fetchall()

    print(f"Embedding {len(rows)} conversation chunks via Ollama ({EMBED_MODEL})...")
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def process(batch):
        # embed_texts is resolved through the ingest module at call time --
        # that keeps ingest.embed_texts the one patchable seam (tests stub
        # it there). Lazy import: cycle-safe once ingest is initialized.
        import ingest

        return batch, ingest.embed_texts([r[4] for r in batch], kind="document")

    n_done, n_err, n_processed, n_committed = 0, 0, 0, 0
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        with progress.track("Embedding chunks", total=len(rows), header=False) as p:
            future_to_batch = {pool.submit(process, batch): batch for batch in batches}
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch, vecs = future.result()
                except (http.client.HTTPException, OSError, RuntimeError,
                        json.JSONDecodeError):
                    n_err += len(batch)
                    # A failed batch is still a settled batch: without this the
                    # bar stalls short of its total and reads as a hang.
                    p.advance(len(batch))
                    continue
                tgt.executemany(
                    "INSERT OR REPLACE INTO chunk_vectors"
                    "(chunk_id, embedding, chat_id, first_day, last_day) VALUES (?, ?, ?, ?, ?)",
                    [(chunk_id, sqlite_vec.serialize_float32(vec), chat_id,
                      iso_to_day(first_date), iso_to_day(last_date))
                     for (chunk_id, chat_id, first_date, last_date, _), vec in zip(batch, vecs)],
                )
                n_done += len(batch)
                n_processed += len(batch)
                p.advance(len(batch))
                if n_processed - n_committed >= batch_commit_every:
                    tgt.commit()
                    n_committed = n_processed
    except KeyboardInterrupt:
        # Don't wait on in-flight worker threads (each blocked on an Ollama
        # HTTP call) -- drop anything not yet started and commit whatever
        # finished so it isn't lost.
        print("\nInterrupted -- cancelling remaining embedding batches...")
        pool.shutdown(wait=False, cancel_futures=True)
        tgt.commit()
        print(f"Embedded {n_done} chunks before interrupt ({n_err} errors)")
        raise
    pool.shutdown(wait=True)
    tgt.commit()
    print(f"Embedded {n_done} chunks ({n_err} errors)")

    # Per-batch errors are swallowed above so one bad batch can't abandon the
    # rest, but that leaves the total-failure case exiting 0 with a database
    # the caller believes is complete -- semantic search then returns nothing,
    # forever, with no error to go looking for. Ollama dying mid-run is the
    # way this happens, so no preflight can catch it; only this can.
    if n_err:
        if n_done == 0:
            raise SystemExit(
                f"error: every embedding batch failed ({n_err} chunks) -- is Ollama running?\n"
                f"  Nothing was embedded, so semantic search would return nothing.\n"
                f"  Check with 'make doctor'; start Ollama with 'make ollama-start',\n"
                f"  then re-run -- embedding resumes from where it stopped."
            )
        print(
            f"warning: {n_err} chunks failed to embed and have no vector. Semantic\n"
            f"  search is incomplete until you re-run ingest (it retries only those)."
        )
