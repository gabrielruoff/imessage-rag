"""Conversation-chunk retrieval tools: semantic search over chunk vectors
(sqlite-vec KNN), hybrid keyword+semantic search fused with reciprocal-rank
fusion, more-like-this, and chunk -> full-messages expansion."""

import contextlib
import datetime
import re

import sqlite_vec

from db import CONTACT_LABEL_SQL

from ._shared import (
    CHUNK_PREVIEW_CHARS,
    HYBRID_POOL,
    MAX_CHUNK_MESSAGES,
    MAX_VEC_K,
    RRF_K,
    _MSG_COLS,
    _MSG_JOIN,
    _connect,
    _date_clauses,
    _id_filter_clause,
    _normalize_date_bound,
    _resolve_chat_scope,
    _truncate_text_field,
)


def _embed_query(query):
    """Embed a search query into a serialized vec0 vector. embed_texts is
    resolved through the package attribute so tests can patch
    query_tools.embed_texts (lazy import -- cycle-safe after package init)."""
    import query_tools

    return sqlite_vec.serialize_float32(query_tools.embed_texts([query], kind="query")[0])


# Label for the chat a chunk belongs to: its display_name, else the resolved
# participant names. The subquery's h/c aliases are what CONTACT_LABEL_SQL
# expects; enclosing queries must alias chunks AS ck and chats AS ch to stay
# out of its way.
_CHUNK_PARTICIPANTS = (
    "COALESCE(NULLIF(ch.display_name, ''), "
    "(SELECT GROUP_CONCAT(" + CONTACT_LABEL_SQL + ", ', ') "
    " FROM chat_participants cp "
    " JOIN handles h ON h.id = cp.handle_id "
    " LEFT JOIN contacts c ON c.id = h.contact_id "
    " WHERE cp.chat_id = ck.chat_id))"
)

_CHUNK_SELECT = (
    f"SELECT ck.id AS chunk_id, ck.chat_id, {_CHUNK_PARTICIPANTS} AS participants, "
    "ck.first_date, ck.last_date, ck.n_messages, "
    "ck.first_message_id, ck.last_message_id"
)


def _day_bounds(date_from, date_to):
    """Convert optional ISO date bounds into inclusive day numbers matching
    chunk_vectors' first_day/last_day metadata (see db.iso_to_day). date_to
    reuses _normalize_date_bound's half-open upper bound, stepped back one
    microsecond so e.g. '2026-01-02' maps to Jan 2's day number, not Jan 3's."""
    from_day = to_day = None
    if date_from is not None:
        from_day = datetime.datetime.fromisoformat(
            _normalize_date_bound(date_from)).toordinal()
    if date_to is not None:
        upper = datetime.datetime.fromisoformat(_normalize_date_bound(date_to, upper=True))
        to_day = (upper - datetime.timedelta(microseconds=1)).toordinal()
    return from_day, to_day


def _chunk_knn(con, query_vec, k, chat_ids, from_day, to_day, exclude_chunk_id=None):
    """KNN over chunk_vectors with scope/date pushed down as vec0 metadata
    constraints (applied inside the scan -- no over-fetch needed). A chunk
    overlaps the date range when last_day >= from AND first_day <= to."""
    fetch_k = k + 1 if exclude_chunk_id is not None else k
    clauses, params = ["v.embedding MATCH ?", "k = ?"], [query_vec, fetch_k]
    if chat_ids:
        placeholders = ",".join("?" * len(chat_ids))
        clauses.append(f"v.chat_id IN ({placeholders})")
        params.extend(chat_ids)
    if from_day is not None:
        clauses.append("v.last_day >= ?")
        params.append(from_day)
    if to_day is not None:
        clauses.append("v.first_day <= ?")
        params.append(to_day)

    rows = con.execute(f"""
        {_CHUNK_SELECT}, v.distance, ck.text
        FROM chunk_vectors v
        JOIN chunks ck ON ck.id = v.chunk_id
        LEFT JOIN chats ch ON ch.id = ck.chat_id
        WHERE {' AND '.join(clauses)}
        ORDER BY v.distance
    """, params).fetchall()
    rows = [dict(r) for r in rows]
    if exclude_chunk_id is not None:
        rows = [r for r in rows if r["chunk_id"] != exclude_chunk_id][:k]
    return rows


def search_conversations(query: str, k=10, chat_id=None, handle_id=None, contact_id=None,
                         date_from=None, date_to=None):
    """Semantic search over conversation chunks: embed the query and find the
    nearest chunk vectors. chat_id/handle_id/contact_id (single id or list
    each) scope to chats / a person's chats; date_from/date_to accept the
    same ISO forms as search_messages_sql and match any chunk whose time span
    overlaps the range. Results carry a text preview -- expand one with
    get_conversation(chunk_id)."""
    k = max(1, min(k, MAX_VEC_K))
    query_vec = _embed_query(query)

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []
        from_day, to_day = _day_bounds(date_from, date_to)
        rows = _chunk_knn(con, query_vec, k, chat_ids, from_day, to_day)
    return _truncate_text_field(rows, CHUNK_PREVIEW_CHARS)


def find_similar_conversations(chunk_id: int, k=10, chat_id=None, handle_id=None,
                               contact_id=None, date_from=None, date_to=None):
    """More-like-this: find conversation chunks semantically similar to an
    existing one, using its stored vector (no embedding request needed). The
    source chunk itself is excluded. Same scope/date filters as
    search_conversations."""
    k = max(1, min(k, MAX_VEC_K))
    with contextlib.closing(_connect()) as con:
        row = con.execute(
            "SELECT embedding FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"chunk {chunk_id} has no embedding (does it exist / "
                             f"has embedding finished?)")
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []
        from_day, to_day = _day_bounds(date_from, date_to)
        rows = _chunk_knn(con, row["embedding"], k, chat_ids, from_day, to_day,
                          exclude_chunk_id=chunk_id)
    return _truncate_text_field(rows, CHUNK_PREVIEW_CHARS)


def get_conversation(chunk_id: int):
    """Expand a chunk into its full messages: every message in the chunk's
    chat within the chunk's time window (so overlap-carried messages and
    system events that didn't render into the chunk text appear too).
    Returns the chunk's metadata plus a `messages` list."""
    with contextlib.closing(_connect()) as con:
        chunk = con.execute(f"""
            {_CHUNK_SELECT}
            FROM chunks ck
            LEFT JOIN chats ch ON ch.id = ck.chat_id
            WHERE ck.id = ?
        """, (chunk_id,)).fetchone()
        if chunk is None:
            return None
        chunk = dict(chunk)

        rows = con.execute(f"""
            SELECT {_MSG_COLS}
            FROM messages m
            {_MSG_JOIN}
            WHERE m.chat_id = ? AND m.date >= ? AND m.date <= ?
            ORDER BY m.date, m.id
            LIMIT ?
        """, (chunk["chat_id"], chunk["first_date"], chunk["last_date"],
              MAX_CHUNK_MESSAGES)).fetchall()
    chunk["messages"] = _truncate_text_field([dict(r) for r in rows])
    return chunk


def _fts_or_query(query: str) -> str:
    """Turn free text into a forgiving FTS5 query: each term quoted (so
    natural-language punctuation can't break FTS5 syntax) and OR-joined --
    any-term match, with rank still rewarding more/denser matches."""
    terms = re.findall(r"\w+", query)
    return " OR ".join(f'"{t}"' for t in terms)


def _fts_chunk_ranks(con, query, chat_ids, date_from, date_to):
    """The FTS arm of hybrid search: {chunk_id: rank} for the chunks holding
    the best keyword hits, {} when the query yields no FTS terms. Takes raw
    date bounds -- unlike the vec arm's day-number metadata filters, FTS hits
    are individual message rows, filtered on the full date string."""
    fts_query = _fts_or_query(query)
    if not fts_query:
        return {}
    clauses = ["messages_fts MATCH ?", "m.chunk_id IS NOT NULL"]
    params = [fts_query]
    clause, cparams = _id_filter_clause("m.chat_id", chat_ids)
    if clause:
        clauses.append(clause)
        params.extend(cparams)
    _date_clauses(clauses, params, date_from, date_to)
    # Aggregate to chunks in SQL (best-ranked message hit per chunk)
    # so a few chatty chunks can't consume the candidate pool.
    hits = con.execute(f"""
        SELECT m.chunk_id
        FROM messages_fts f
        JOIN messages m ON m.id = f.rowid
        WHERE {' AND '.join(clauses)}
        GROUP BY m.chunk_id
        ORDER BY MIN(rank)
        LIMIT ?
    """, params + [HYBRID_POOL]).fetchall()
    return {row["chunk_id"]: rank for rank, row in enumerate(hits)}


def search_messages_hybrid(query: str, k=10, chat_id=None, handle_id=None, contact_id=None,
                           date_from=None, date_to=None):
    """Keyword (FTS5 over messages) and semantic (chunk vectors) search run
    together and fused with reciprocal-rank fusion; FTS hits vote for the
    chunk containing them. Returns ranked conversation chunks, each tagged
    with matched_by ('keyword' and/or 'semantic') and rrf_score. Both arms
    apply the same chat/person/date scope."""
    k = max(1, min(k, MAX_VEC_K))
    query_vec = _embed_query(query)

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []
        from_day, to_day = _day_bounds(date_from, date_to)

        vec_rows = _chunk_knn(con, query_vec, HYBRID_POOL, chat_ids, from_day, to_day)
        vec_ranks = {r["chunk_id"]: rank for rank, r in enumerate(vec_rows)}

        fts_ranks = _fts_chunk_ranks(con, query, chat_ids, date_from, date_to)

        scores = {}
        for ranks in (vec_ranks, fts_ranks):
            for cid, rank in ranks.items():
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        if not top:
            return []

        by_id = {r["chunk_id"]: r for r in vec_rows}
        missing = [cid for cid in top if cid not in by_id]
        if missing:
            placeholders = ",".join("?" * len(missing))
            for r in con.execute(f"""
                {_CHUNK_SELECT}, NULL AS distance, ck.text
                FROM chunks ck
                LEFT JOIN chats ch ON ch.id = ck.chat_id
                WHERE ck.id IN ({placeholders})
            """, missing).fetchall():
                by_id[r["chunk_id"]] = dict(r)

    results = []
    for cid in top:
        r = by_id.get(cid)
        if r is None:
            continue
        r["rrf_score"] = round(scores[cid], 6)
        r["matched_by"] = [name for name, ranks in
                           (("semantic", vec_ranks), ("keyword", fts_ranks)) if cid in ranks]
        results.append(r)
    return _truncate_text_field(results, CHUNK_PREVIEW_CHARS)
