"""
Derived-artifact stages of the ingest pipeline: group ingested messages into
conversation chunks, then embed the chunks (local Ollama, or the Voyage Batch
API for a voyage/ EMBED_MODEL) into the sqlite-vec chunk_vectors table.

Everything here operates only on imessage_analysis.db (the target DB), never
the chat.db snapshot. Chunk building and chunk embedding live together
deliberately: they share the chunks lifecycle (an incremental run's
tail-chunk rebuild deletes rows from both `chunks` and `chunk_vectors`).
ingest.py runs both stages after source ingestion and re-exports them, so
`ingest.build_chunks` etc. keep working.
"""

import asyncio
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
import config
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


def unembedded_chunks(tgt: sqlite3.Connection, min_chars=CHUNK_MIN_CHARS):
    """(id, chat_id, first_date, last_date, text) of every chunk that should
    have a vector and doesn't -- the one "pending" predicate both backends
    and their end-of-run checks share."""
    return tgt.execute("""
        SELECT c.id, c.chat_id, c.first_date, c.last_date, c.text
        FROM chunks c
        LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
        WHERE v.chunk_id IS NULL
          AND length(c.text) >= ?
    """, (min_chars,)).fetchall()


def _insert_vectors(tgt, rows):
    """rows: (chunk_id, chat_id, first_date, last_date, vector)."""
    tgt.executemany(
        "INSERT OR REPLACE INTO chunk_vectors"
        "(chunk_id, embedding, chat_id, first_day, last_day) VALUES (?, ?, ?, ?, ?)",
        [(chunk_id, sqlite_vec.serialize_float32(vec), chat_id,
          iso_to_day(first_date), iso_to_day(last_date))
         for chunk_id, chat_id, first_date, last_date, vec in rows],
    )


def embed_chunks(
    tgt: sqlite3.Connection,
    min_chars=CHUNK_MIN_CHARS,
    batch_size=EMBED_BATCH_SIZE,
    max_workers=EMBED_MAX_WORKERS,
    batch_commit_every=EMBED_COMMIT_EVERY,
):
    rows = unembedded_chunks(tgt, min_chars)

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
                _insert_vectors(tgt, [(*row[:4], vec) for row, vec in zip(batch, vecs)])
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


# ---------------------------------------------------------------------------
# Embeddings via the Voyage Batch API (EMBED_MODEL="voyage/<id>")
# ---------------------------------------------------------------------------
# ingest_state keys: voyage_batch_ids (JSON list of batches not yet collected),
# voyage_batch_space ("<EMBED_MODEL>|<EMBED_DIM>" they were submitted under),
# voyage_pending_input_file (an uploaded input file whose batch id is not yet
# saved; "" when none), voyage_pending_input_space (the space that file was
# uploaded under).


class _DiscardBatch(Exception):
    """A batch whose results must not be stored (gone, or another embedding
    space). Its id is dropped; its chunks stay missing."""


def _voyage_space():
    return f"{EMBED_MODEL}|{config.EMBED_DIM}"


def _saved_batch_ids(tgt):
    return json.loads(get_state(tgt, "voyage_batch_ids", "[]"))


def _save_batch_ids(tgt, ids):
    set_state(tgt, "voyage_batch_ids", json.dumps(ids))
    tgt.commit()


def _track_batch(tgt, batch_id):
    """Save a new batch id (and its space) and clear the pending input file,
    in one commit."""
    set_state(tgt, "voyage_batch_space", _voyage_space())
    set_state(tgt, "voyage_pending_input_file", "")
    _save_batch_ids(tgt, _saved_batch_ids(tgt) + [batch_id])


def _voyage_row(tgt, custom_id, vec):
    """The insert row for one output line, or None when its chunk is gone, its
    text changed since submit (the vector is stale; the chunk stays missing and
    is resubmitted), or it already has a vector (a resumed batch re-streams
    lines an interrupted collect committed; vec0 rejects a second insert)."""
    import voyage

    chunk_id, digest = voyage.parse_custom_id(custom_id)
    if tgt.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)).fetchone():
        return None
    row = tgt.execute("SELECT chat_id, first_date, last_date, text FROM chunks WHERE id = ?",
                      (chunk_id,)).fetchone()
    if row is None or voyage.text_hash(row[3]) != digest:
        return None
    return (chunk_id, *row[:3], vec)


async def _reconcile_pending_input(tgt, client):
    """A previous run uploaded an input file but never saved its batch id:
    adopt the batch Voyage created from it -- unless the file was uploaded
    under another embedding space, when the batch and its files are deleted
    and nothing is stored -- else delete the file."""
    file_id = get_state(tgt, "voyage_pending_input_file")
    if not file_id:
        return
    batch = await client.find_batch(file_id)
    if batch and get_state(tgt, "voyage_pending_input_space", _voyage_space()) == _voyage_space():
        print(f"Adopting Voyage batch {batch['id']} from an interrupted submit")
        _track_batch(tgt, batch["id"])
        return
    if batch:
        print(f"warning: not adopting Voyage batch {batch['id']}: it was submitted under "
              f"another EMBED_MODEL/EMBED_DIM; its chunks are resubmitted")
        await _delete_batch_files(client, batch)
    else:
        await client.delete_file(file_id)
    set_state(tgt, "voyage_pending_input_file", "")
    tgt.commit()


async def _submit_voyage_batch(tgt, client, budget, batch_rows, lines):
    """Reserve the worst case, upload, create, and persist the batch id before
    any polling. Returns (batch_id, reserved_usd)."""
    import hosted
    import voyage

    reserved = voyage.worst_case_usd([text for _, text in batch_rows])
    if not budget.reserve(reserved):
        raise hosted.HostedStop(f"budget cap reached: the next batch needs up to "
                                f"${reserved:.4f} ({budget.report()})")
    file_id = await client.upload(b"".join(lines))
    # Saved first: if create_batch's outcome is lost, the next run reconciles.
    set_state(tgt, "voyage_pending_input_file", file_id)
    set_state(tgt, "voyage_pending_input_space", _voyage_space())
    tgt.commit()
    batch = await client.create_batch(file_id)
    _track_batch(tgt, batch["id"])
    print(f"Submitted Voyage batch {batch['id']} ({len(lines)} chunks)")
    return batch["id"], reserved


async def _collect_voyage_batch(tgt, client, budget, batch_id, reserved, commit_every,
                                resumed):
    """Poll to a terminal state, stream the output into chunk_vectors, count
    the error file, settle the budget, forget the id, then delete its files
    (only after the results are committed)."""
    import voyage

    batch = await client.get_batch(batch_id)
    if resumed and (
            get_state(tgt, "voyage_batch_space", _voyage_space()) != _voyage_space()
            or batch.get("model", voyage.model_id()) != voyage.model_id()):
        raise _DiscardBatch(batch, "it was submitted under another EMBED_MODEL/EMBED_DIM")
    total = (batch.get("request_counts") or {}).get("total") or None
    with progress.track(f"Waiting on Voyage batch {batch_id}", total=total) as p:
        seen = 0
        while True:
            counts = batch.get("request_counts") or {}
            done = counts.get("completed", 0) + counts.get("failed", 0)
            p.advance(max(0, done - seen))
            seen = max(seen, done)
            if batch["status"] in voyage.TERMINAL:
                break
            await asyncio.sleep(config.VOYAGE_POLL_SECONDS)
            batch = await client.get_batch(batch_id)

    n_stored, n_stale, n_failed, n_errored, tokens, pending = 0, 0, 0, 0, 0, []
    if batch.get("output_file_id"):
        async for line in client.stream_file(batch["output_file_id"]):
            rec = json.loads(line)
            response = rec.get("response") or {}
            if response.get("status_code") != 200:
                n_failed += 1  # stays missing; the next round resubmits it
                continue
            body = response["body"]
            tokens += (body.get("usage") or {}).get("total_tokens", 0)
            vec = body["data"][0]["embedding"]
            if len(vec) != config.EMBED_DIM:
                raise _DiscardBatch(batch, f"it returned {len(vec)}-dim vectors, "
                                           f"EMBED_DIM is {config.EMBED_DIM}")
            row = _voyage_row(tgt, rec["custom_id"], vec)
            if row is None:
                n_stale += 1
                continue
            pending.append(row)
            if len(pending) >= commit_every:
                _insert_vectors(tgt, pending)
                tgt.commit()
                n_stored += len(pending)
                pending = []
    _insert_vectors(tgt, pending)
    tgt.commit()
    n_stored += len(pending)
    budget.settle(reserved, tokens * config.VOYAGE_BATCH_PRICE_PER_M / 1e6)
    if batch.get("error_file_id"):
        async for _ in client.stream_file(batch["error_file_id"]):
            n_errored += 1  # their chunks stay missing and join the next round
    summary = (f"Voyage batch {batch_id} {batch['status']}: {n_stored} vectors stored, "
               f"{n_stale} stale discarded, {n_failed} failed, {n_errored} errored")
    if batch["status"] == "failed":
        # codes and line numbers only: an error message may quote input text
        errors = (batch.get("errors") or {}).get("data") or []
        summary += f" ({[(e.get('code'), e.get('line')) for e in errors]})"
    print(summary)

    _save_batch_ids(tgt, [b for b in _saved_batch_ids(tgt) if b != batch_id])
    await _delete_batch_files(client, batch)


async def _delete_batch_files(client, batch):
    for file_id in (batch.get("input_file_id"), batch.get("output_file_id"),
                    batch.get("error_file_id")):
        if file_id:
            await client.delete_file(file_id)


async def _finish_voyage_batch(tgt, client, budget, batch_id, reserved, commit_every,
                               resumed=False):
    """_collect_voyage_batch, dropping a batch that must not be stored: one
    from another embedding space, or (on resume only) one Voyage no longer
    serves. Its chunks stay missing, so nothing is paid twice."""
    import voyage

    try:
        await _collect_voyage_batch(tgt, client, budget, batch_id, reserved, commit_every,
                                    resumed)
    except (_DiscardBatch, voyage.NotFound) as e:
        if isinstance(e, voyage.NotFound) and not resumed:
            raise
        why = e.args[1] if isinstance(e, _DiscardBatch) else "Voyage no longer has it"
        print(f"warning: dropping Voyage batch {batch_id}: {why}; its chunks are resubmitted")
        budget.settle(reserved, reserved)
        _save_batch_ids(tgt, [b for b in _saved_batch_ids(tgt) if b != batch_id])
        if isinstance(e, _DiscardBatch):
            await _delete_batch_files(client, e.args[0])


async def _embed_voyage(tgt, budget, max_parallel, min_chars, commit_every):
    import hosted
    import voyage

    # One Limiter per event loop; every batch of this run shares it.
    async with voyage.BatchClient(hosted.Limiter(max_parallel)) as client:
        # Saved batches finish before anything new is submitted: never pay twice.
        await _reconcile_pending_input(tgt, client)
        await asyncio.gather(*(
            _finish_voyage_batch(tgt, client, budget, batch_id, 0.0, commit_every,
                                 resumed=True)
            for batch_id in _saved_batch_ids(tgt)))
        stalled = 0
        while True:
            rows = unembedded_chunks(tgt, min_chars)
            if not rows:
                return
            print(f"Embedding {len(rows)} conversation chunks via the Voyage Batch API "
                  f"({EMBED_MODEL})...")
            submitted = [
                await _submit_voyage_batch(tgt, client, budget, batch_rows, lines)
                for batch_rows, lines in voyage.pack_lines([(r[0], r[4]) for r in rows])]
            await asyncio.gather(*(
                _finish_voyage_batch(tgt, client, budget, batch_id, reserved, commit_every)
                for batch_id, reserved in submitted))
            if len(unembedded_chunks(tgt, min_chars)) < len(rows):
                stalled = 0
                continue
            stalled += 1
            if stalled >= config.VOYAGE_MAX_STALLED_ROUNDS:
                raise hosted.HostedStop(f"{stalled} Voyage rounds in a row embedded nothing")


def embed_chunks_voyage(tgt: sqlite3.Connection, budget, max_parallel,
                        min_chars=CHUNK_MIN_CHARS, commit_every=EMBED_COMMIT_EVERY):
    """Embed every pending chunk through the Voyage Batch API: finish batches a
    previous run saved in ingest_state.voyage_batch_ids, then submit, poll and
    collect until nothing is missing. Raises hosted.HostedStop on a
    non-transient stop; SystemExit(1) when the final re-count from the DB is
    not 0."""
    try:
        asyncio.run(_embed_voyage(tgt, budget, max_parallel, min_chars, commit_every))
    except KeyboardInterrupt:
        print("\nInterrupted -- submitted Voyage batches stay saved; the next "
              "ingest collects them before submitting anything new.")
        raise
    finally:
        tgt.commit()
        print(budget.report())
    missing = len(unembedded_chunks(tgt, min_chars))
    print(f"{missing} chunks without a vector")
    if missing:
        raise SystemExit(f"error: {missing} chunks still have no vector -- re-run ingest "
                         f"to resubmit them")
