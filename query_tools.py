"""
Query layer over imessage_analysis.db, meant to be exposed as LLM tool calls
(Claude tool_use / function-calling) rather than pre-loading data into context.

Retrieval modes:
  - search_messages_sql     structured filters (chat, sender, date range)
  - search_messages_fts     keyword / phrase search via FTS5
  - search_conversations    semantic search over conversation chunks
                            (sqlite-vec + Ollama embeddings)
  - search_messages_hybrid  keyword + semantic together, fused with
                            reciprocal-rank fusion

plus expansion helpers: get_conversation (a chunk's messages),
get_message_context (messages around any hit), find_similar_conversations
(more-like-this from a chunk's stored vector), get_full_messages.

Each returns a small list of dicts, meant to be fed back to the LLM as tool
results -- never the whole table.
"""

import contextlib
import datetime
import re
import sqlite3

import sqlite_vec

from config import TARGET_DB
from db import CONTACT_LABEL_SQL, load_vec_extension
from embeddings import embed_texts

DB_PATH = str(TARGET_DB)


def _connect():
    con = sqlite3.connect(DB_PATH)
    load_vec_extension(con)
    con.row_factory = sqlite3.Row
    return con


# Human-readable label for a contact/handle: nickname > "First Last" > raw
# handle_identifier (shared with ingest.py's chunk builder via db.py).
# Assumes the query LEFT JOINs handles AS h and contacts AS c.
_CONTACT_LABEL = CONTACT_LABEL_SQL

# Resolve a message's sender: "Me" when is_from_me, else the contact label.
# Requires the join below.
_SENDER_SELECT = f"CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE {_CONTACT_LABEL} END AS sender"
_MSG_JOIN = (
    "LEFT JOIN handles h ON h.id = m.handle_id "
    "LEFT JOIN contacts c ON c.id = h.contact_id"
)

# Ceilings on model-requested result sizes -- defaults above are just what's
# used when the model omits the arg; these are enforced regardless of what
# the model asks for, so a tool call can't blow up prompt size.
MAX_MESSAGE_LIMIT = 50   # search_messages_sql / search_messages_fts
MAX_VEC_K = 20           # chunk-search tools (search_conversations etc.)
MAX_CONTACTS_LIMIT = 20  # search_contacts_by_name
MAX_CHATS_LIMIT = 50     # list_chats
MAX_ID_LIST = 50         # cap on chat_id/handle_id/contact_id/guid list params
MAX_CONTEXT_MESSAGES = 25   # get_message_context before/after ceiling
MAX_SCOPE_CHATS = 500       # cap on the person -> chats expansion for chunk search
MAX_CHUNK_MESSAGES = 200    # get_conversation row ceiling
HYBRID_POOL = 30            # per-arm candidate depth fed into RRF
RRF_K = 60                  # standard reciprocal-rank-fusion constant

TEXT_PREVIEW_CHARS = 300    # message rows
CHUNK_PREVIEW_CHARS = 500   # chunk rows (get_conversation returns the messages)


def _truncate_text_field(rows, limit=TEXT_PREVIEW_CHARS):
    """Cap each row's `text` field for prompt size; full content is available
    via get_full_messages(ids) / get_conversation(chunk_id)."""
    for r in rows:
        text = r.get("text")
        if text and len(text) > limit:
            r["text"] = text[:limit] + "…"
    return rows


def _as_id_list(value):
    """Normalize a param that may be None, a scalar, or a list into None or
    a capped list -- lets every id-scoped tool param accept either shape."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)[:MAX_ID_LIST]
    return [value]


def _resolve_contact_handle_ids(con, contact_ids):
    """Given a list of contact_ids, return the handle_ids they own."""
    placeholders = ",".join("?" * len(contact_ids))
    rows = con.execute(
        f"SELECT id FROM handles WHERE contact_id IN ({placeholders})",
        contact_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _id_filter_clause(column, values):
    """Build an `column IN (?,...)` fragment + params for a normalized id
    list, or ("", []) if values is None."""
    if not values:
        return "", []
    placeholders = ",".join("?" * len(values))
    return f"{column} IN ({placeholders})", list(values)


def _append_id_filters(clauses, params, chat_ids, handle_ids, prefix="m."):
    """Append chat_id/handle_id IN-clauses (if non-empty) to a shared
    clauses/params list -- the common tail of every scoped message query."""
    for column, ids in ((f"{prefix}chat_id", chat_ids), (f"{prefix}handle_id", handle_ids)):
        clause, cparams = _id_filter_clause(column, ids)
        if clause:
            clauses.append(clause)
            params.extend(cparams)


def _date_bound_step(value):
    """How far to advance an upper date bound: one unit of the smallest field
    the caller actually wrote. Read off the raw string because fromisoformat
    parses '2026-01-02' and '2026-01-02T00:00:00' into the same datetime.

    fromisoformat accepts any single character as the date/time separator (it
    is isoformat(sep=...)'s inverse -- '2026-01-02 13:46:50' parses fine), so
    the time part is found positionally, after the 10-char extended date or
    8-char compact date, not by splitting on 'T'."""
    time_part = value[11:] if "-" in value[:5] else value[9:]
    for sep in ("+", "-", "Z", "z"):          # drop any UTC offset suffix
        time_part = time_part.partition(sep)[0]
    if not time_part:
        return datetime.timedelta(days=1)
    if "." in time_part or "," in time_part:  # ISO allows a comma decimal mark
        return datetime.timedelta(microseconds=1)
    return {
        2: datetime.timedelta(hours=1),       # '13' / compact or extended
        4: datetime.timedelta(minutes=1),     # '13:46' / '1346'
        6: datetime.timedelta(seconds=1),     # '13:46:50' / '134650'
    }.get(len(time_part.replace(":", "")), datetime.timedelta(seconds=1))


def _normalize_date_bound(value, upper=False):
    """Parse an ISO date/datetime into the unlabeled-UTC string form the `date`
    columns are stored in (see the module docstring on storage format).

    An upper bound comes back half-open -- advanced by one unit of whatever
    granularity the caller wrote -- so date_to='2026-01-02' means "through the
    end of Jan 2" and pairs with `date < ?`. Compared as a plain string, a bare
    date would otherwise match only exactly midnight and silently drop the day.

    A tz-aware bound ('...-05:00', '...Z') is converted to UTC here, so a caller
    that skips the conversion still gets a correct answer. An unparseable bound
    raises ValueError rather than quietly matching nothing.
    """
    if value is None:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"invalid ISO date {value!r}: expected '2026-01-02' or "
            f"'2026-01-02T13:00:00' (optionally with a UTC offset) -- {e}"
        ) from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    if upper:
        parsed += _date_bound_step(value)
    return parsed.isoformat()


def _resolve_handle_filter(con, handle_id, contact_id):
    """Merge handle_id and contact_id (each scalar-or-list-or-None) into a
    single normalized handle_id list, resolving contact_id through handles."""
    handle_ids = _as_id_list(handle_id) or []
    contact_ids = _as_id_list(contact_id)
    if contact_ids:
        handle_ids = list(handle_ids) + _resolve_contact_handle_ids(con, contact_ids)
    return handle_ids[:MAX_ID_LIST] or None


def _resolve_scope(con, chat_id, handle_id, contact_id):
    """Resolve scope params into (chat_ids, handle_ids, empty). `empty` is
    True when a scope param was provided but matched no ids -- the caller
    should then return [] rather than dropping the filter and over-matching."""
    chat_requested = chat_id is not None
    handle_requested = handle_id is not None or contact_id is not None
    chat_ids = _as_id_list(chat_id)
    handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
    empty = (chat_requested and not chat_ids) or (handle_requested and not handle_ids)
    return chat_ids, handle_ids, empty


def search_messages_sql(chat_id=None, handle_id=None, contact_id=None, is_from_me=None,
                         date_from=None, date_to=None, limit=50):
    """Structured filter search -- no text matching, just facets. chat_id and
    handle_id each accept a single id or a list of ids. contact_id (single or
    list) scopes to a person across every handle they own -- the way to search
    one person's messages regardless of which phone/email/chat they used.
    date_from/date_to accept an ISO date ('2026-01-01') or full ISO datetime
    ('2026-01-01T13:00:00') -- a full datetime narrows to a time-of-day range,
    not just a day. date_to covers the whole unit named, so date_to='2026-01-01'
    includes all of Jan 1. Message dates are stored in UTC; a bound carrying an
    offset ('2026-01-01T13:00:00-05:00', '...Z') is converted to UTC for you,
    and a bound without one is taken as already-UTC."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))

    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []

        clauses, params = [], []
        _append_id_filters(clauses, params, chat_ids, handle_ids)
        if is_from_me is not None:
            clauses.append("m.is_from_me = ?")
            params.append(int(is_from_me))
        if date_from is not None:
            clauses.append("m.date >= ?")
            params.append(_normalize_date_bound(date_from))
        if date_to is not None:
            clauses.append("m.date < ?")
            params.append(_normalize_date_bound(date_to, upper=True))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages m
            {_MSG_JOIN}
            {where}
            ORDER BY m.date DESC
            LIMIT ?
        """
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
    return _truncate_text_field([dict(r) for r in rows])


def search_messages_fts(query: str, chat_id=None, handle_id=None, contact_id=None, limit=50):
    """Keyword/phrase search via FTS5 (supports FTS5 query syntax:
    'foo AND bar', '"exact phrase"', 'foo NOT bar', etc.). chat_id/handle_id
    accept a single id or a list; contact_id (single or list) scopes to a
    person across every handle they own."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))
    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []

        clauses, params = ["messages_fts MATCH ?"], [query]
        _append_id_filters(clauses, params, chat_ids, handle_ids)

        sql = f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            {_MSG_JOIN}
            WHERE {' AND '.join(clauses)}
            ORDER BY rank
            LIMIT ?
        """
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
    return _truncate_text_field([dict(r) for r in rows])


# --- Conversation-chunk search ----------------------------------------------

# Label for the chat a chunk belongs to: its display_name, else the resolved
# participant names. The subquery's h/c aliases are what _CONTACT_LABEL
# expects; enclosing queries must alias chunks AS ck and chats AS ch to stay
# out of its way.
_CHUNK_PARTICIPANTS = (
    "COALESCE(NULLIF(ch.display_name, ''), "
    "(SELECT GROUP_CONCAT(" + _CONTACT_LABEL + ", ', ') "
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


def _resolve_chat_scope(con, chat_id, handle_id, contact_id):
    """Resolve scope params into (chat_ids, empty) for chunk-level search.
    Chunks belong to whole chats, so handle_id/contact_id scope to the chats
    that person participates in (an exact chat_participants join); when chat
    and person params are both given, the scopes intersect, matching the AND
    semantics of the message-level tools. `empty` mirrors _resolve_scope:
    True when a scope was requested but matched nothing."""
    chat_ids = _as_id_list(chat_id)
    if chat_id is not None and not chat_ids:
        return None, True
    handle_requested = handle_id is not None or contact_id is not None
    if handle_requested:
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if not handle_ids:
            return None, True
        placeholders = ",".join("?" * len(handle_ids))
        # ORDER BY makes the MAX_SCOPE_CHATS truncation deterministic
        person_chats = [r[0] for r in con.execute(
            f"SELECT DISTINCT chat_id FROM chat_participants "
            f"WHERE handle_id IN ({placeholders}) ORDER BY chat_id", handle_ids,
        ).fetchall()][:MAX_SCOPE_CHATS]
        if chat_ids:
            person_set = set(person_chats)
            chat_ids = [c for c in chat_ids if c in person_set]
        else:
            chat_ids = person_chats
        if not chat_ids:
            return None, True
    return chat_ids, False


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
    query_vec = sqlite_vec.serialize_float32(embed_texts([query], kind="query")[0])

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
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages m
            {_MSG_JOIN}
            WHERE m.chat_id = ? AND m.date >= ? AND m.date <= ?
            ORDER BY m.date, m.id
            LIMIT ?
        """, (chunk["chat_id"], chunk["first_date"], chunk["last_date"],
              MAX_CHUNK_MESSAGES)).fetchall()
    chunk["messages"] = _truncate_text_field([dict(r) for r in rows])
    return chunk


def get_message_context(message_id: int, before=10, after=10):
    """The messages surrounding one message in its chat, in time order --
    turns any single search hit (FTS, SQL, or a message inside a chunk) into
    a readable stretch of conversation. The target row is flagged
    is_target=True. Returns [] if the message id doesn't exist."""
    before = max(0, min(int(before), MAX_CONTEXT_MESSAGES))
    after = max(0, min(int(after), MAX_CONTEXT_MESSAGES))
    with contextlib.closing(_connect()) as con:
        target = con.execute(
            "SELECT id, chat_id, date FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if target is None:
            return []

        select = f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages m
            {_MSG_JOIN}
        """
        # (date, id) row-value comparisons: same total order as the ORDER BY,
        # id as tiebreak for identical timestamps.
        prev_rows = con.execute(select + """
            WHERE m.chat_id = ? AND (m.date, m.id) < (?, ?)
            ORDER BY m.date DESC, m.id DESC
            LIMIT ?
        """, (target["chat_id"], target["date"], target["id"], before)).fetchall()
        next_rows = con.execute(select + """
            WHERE m.chat_id = ? AND (m.date, m.id) >= (?, ?)
            ORDER BY m.date, m.id
            LIMIT ?
        """, (target["chat_id"], target["date"], target["id"], after + 1)).fetchall()

    rows = [dict(r) for r in reversed(prev_rows)] + [dict(r) for r in next_rows]
    for r in rows:
        r["is_target"] = r["id"] == message_id
    return _truncate_text_field(rows)


def _fts_or_query(query: str) -> str:
    """Turn free text into a forgiving FTS5 query: each term quoted (so
    natural-language punctuation can't break FTS5 syntax) and OR-joined --
    any-term match, with rank still rewarding more/denser matches."""
    terms = re.findall(r"\w+", query)
    return " OR ".join(f'"{t}"' for t in terms)


def search_messages_hybrid(query: str, k=10, chat_id=None, handle_id=None, contact_id=None,
                           date_from=None, date_to=None):
    """Keyword (FTS5 over messages) and semantic (chunk vectors) search run
    together and fused with reciprocal-rank fusion; FTS hits vote for the
    chunk containing them. Returns ranked conversation chunks, each tagged
    with matched_by ('keyword' and/or 'semantic') and rrf_score. Both arms
    apply the same chat/person/date scope."""
    k = max(1, min(k, MAX_VEC_K))
    query_vec = sqlite_vec.serialize_float32(embed_texts([query], kind="query")[0])

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []
        from_day, to_day = _day_bounds(date_from, date_to)

        vec_rows = _chunk_knn(con, query_vec, HYBRID_POOL, chat_ids, from_day, to_day)
        vec_ranks = {r["chunk_id"]: rank for rank, r in enumerate(vec_rows)}

        fts_ranks = {}
        fts_query = _fts_or_query(query)
        if fts_query:
            clauses = ["messages_fts MATCH ?", "m.chunk_id IS NOT NULL"]
            params = [fts_query]
            clause, cparams = _id_filter_clause("m.chat_id", chat_ids)
            if clause:
                clauses.append(clause)
                params.extend(cparams)
            if date_from is not None:
                clauses.append("m.date >= ?")
                params.append(_normalize_date_bound(date_from))
            if date_to is not None:
                clauses.append("m.date < ?")
                params.append(_normalize_date_bound(date_to, upper=True))
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
            fts_ranks = {row["chunk_id"]: rank for rank, row in enumerate(hits)}

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


def search_contact_by_handle(handle_id=None, handle_identifier=None):
    """Look up contact info for a handle, plus every other handle the same contact owns."""
    if handle_id is None and handle_identifier is None:
        raise ValueError("must provide handle_id or handle_identifier")

    handle_cols = "id, handle_identifier, contact_id, service"
    with contextlib.closing(_connect()) as con:
        if handle_id is not None:
            handle = con.execute(
                f"SELECT {handle_cols} FROM handles WHERE id = ?", (handle_id,)
            ).fetchone()
        else:
            handle = con.execute(
                f"SELECT {handle_cols} FROM handles WHERE handle_identifier = ?",
                (handle_identifier,),
            ).fetchone()

        if handle is None:
            return None

        handle = dict(handle)
        contact = None
        other_handles = []
        if handle["contact_id"] is not None:
            contact = con.execute(
                "SELECT id, source_uniqueid, first_name, last_name, nickname, organization "
                "FROM contacts WHERE id = ?", (handle["contact_id"],)
            ).fetchone()
            contact = dict(contact) if contact else None

            other_handles = con.execute(
                "SELECT id, handle_identifier, service FROM handles "
                "WHERE contact_id = ? AND id != ?",
                (handle["contact_id"], handle["id"]),
            ).fetchall()
            other_handles = [dict(r) for r in other_handles]

    return {
        "handle": handle,
        "contact": contact,
        "other_handles": other_handles,
    }


def search_contacts_by_name(name: str, limit=20):
    """Find contacts by name or organization -- the bridge from a human name to
    the handle_id / chat_id values the other tools need. Returns each matched
    contact plus all handles (phone numbers/emails) it owns."""
    limit = max(1, min(limit, MAX_CONTACTS_LIMIT))
    like = f"%{name}%"
    with contextlib.closing(_connect()) as con:
        contacts = con.execute(
            """
            SELECT id, source_uniqueid, first_name, last_name, nickname, organization
            FROM contacts
            WHERE first_name LIKE ? OR last_name LIKE ? OR nickname LIKE ?
               OR organization LIKE ?
               OR TRIM(COALESCE(first_name,'') || ' ' || COALESCE(last_name,'')) LIKE ?
            ORDER BY last_name, first_name
            LIMIT ?
            """,
            (like, like, like, like, like, limit),
        ).fetchall()

        results = []
        for c in contacts:
            c = dict(c)
            handles = con.execute(
                "SELECT id, handle_identifier, service FROM handles WHERE contact_id = ?",
                (c["id"],),
            ).fetchall()
            c["handles"] = [dict(h) for h in handles]
            results.append(c)
    return results


def list_chats(query=None, handle_id=None, contact_id=None, limit=50):
    """List chats (conversations) with resolved participant names -- the bridge
    from a person/chat name to a chat_id, and the way to label a chat_id. Pass
    `query` to filter by participant name or chat_identifier (substring match,
    fuzzy). Pass handle_id/contact_id (single id or list) for an exact match on
    every chat that person actually participates in -- prefer this over `query`
    once you already have a contact_id from search_contacts_by_name, since a
    name substring can over- or under-match."""
    limit = max(1, min(limit, MAX_CHATS_LIMIT))
    like = f"%{query}%" if query else None
    with contextlib.closing(_connect()) as con:
        handle_requested = handle_id is not None or contact_id is not None
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if handle_requested and not handle_ids:
            return []
        params = {"like": like, "limit": limit}
        id_filter_sql = ""
        if handle_ids:
            placeholders = ",".join(f":hid{i}" for i in range(len(handle_ids)))
            id_filter_sql = f"""
                AND ch.id IN (
                    SELECT cp2.chat_id FROM chat_participants cp2
                    WHERE cp2.handle_id IN ({placeholders})
                )
            """
            params.update({f"hid{i}": v for i, v in enumerate(handle_ids)})

        # Per-chat participant label list, resolved through handles/contacts.
        # The substring filter and limit are applied in SQL: the HAVING clause
        # is a no-op when `like` is NULL, else it matches the chat_identifier,
        # display_name, or the GROUP_CONCAT'd participant labels. LIKE is
        # case-insensitive for ASCII, matching the prior Python behavior.
        # id_filter_sql (when handle_ids/contact_id given) restricts to chats
        # that person actually participates in -- an exact join, not a guess.
        rows = con.execute(
            f"""
            SELECT ch.id, ch.chat_identifier, ch.display_name, ch.is_group, ch.service,
                   GROUP_CONCAT({_CONTACT_LABEL}, ', ') AS participants
            FROM chats ch
            LEFT JOIN chat_participants cp ON cp.chat_id = ch.id
            LEFT JOIN handles h ON h.id = cp.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE 1=1 {id_filter_sql}
            GROUP BY ch.id
            HAVING :like IS NULL
                OR ch.chat_identifier LIKE :like
                OR ch.display_name LIKE :like
                OR participants LIKE :like
            ORDER BY ch.id
            LIMIT :limit
            """,
            params,
        ).fetchall()

    return [dict(r) for r in rows]


def get_message_reactions(target_message_guid):
    """List the reactions/tapbacks on one or more messages (by guid), with who
    reacted. Accepts a single guid or a list of guids -- each returned row
    carries its `target_message_guid` so a batch result stays attributable.
    Message search results include each message's `guid`."""
    guids = _as_id_list(target_message_guid) or []
    if not guids:
        return []
    placeholders = ",".join("?" * len(guids))
    with contextlib.closing(_connect()) as con:
        rows = con.execute(
            f"""
            SELECT r.id, r.target_message_guid, r.is_from_me,
                   CASE WHEN r.is_from_me = 1 THEN 'Me' ELSE {_CONTACT_LABEL} END AS reactor,
                   r.associated_message_type, r.date
            FROM reactions r
            LEFT JOIN handles h ON h.id = r.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE r.target_message_guid IN ({placeholders})
            ORDER BY r.date
            """,
            guids,
        ).fetchall()
    return [dict(r) for r in rows]


def get_full_messages(ids):
    """Return the full, untruncated text for one or more message ids.
    Use after a search result's `text` was truncated and the full content
    is needed."""
    if isinstance(ids, int):
        ids = [ids]
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with contextlib.closing(_connect()) as con:
        rows = con.execute(f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages m
            {_MSG_JOIN}
            WHERE m.id IN ({placeholders})
        """, ids).fetchall()
    return [dict(r) for r in rows]


_ZERO_STATS = {"message_count": 0, "first_message": None, "last_message": None,
               "from_me": 0, "from_them": 0, "attachment_count": 0, "reaction_count": 0}


def _stats_filters(prefix, chat_ids, handle_ids, is_from_me, date_from, date_to):
    """(clauses, params) scoping one table of the stats query. messages,
    reactions, and attachments-joined-through-messages all expose the same
    chat_id/handle_id/is_from_me/date columns, so building each table's filter
    from here keeps every count on a row describing the same slice."""
    clauses, params = [], []
    _append_id_filters(clauses, params, chat_ids, handle_ids, prefix=prefix)
    if is_from_me is not None:
        clauses.append(f"{prefix}is_from_me = ?")
        params.append(int(is_from_me))
    if date_from is not None:
        clauses.append(f"{prefix}date >= ?")
        params.append(_normalize_date_bound(date_from))
    if date_to is not None:
        clauses.append(f"{prefix}date < ?")
        params.append(_normalize_date_bound(date_to, upper=True))
    return clauses, params


def get_chat_summary_stats(chat_id=None, handle_id=None, contact_id=None,
                            is_from_me=None, date_from=None, date_to=None,
                            group_by_chat=False):
    """Cheap aggregate stats -- no LLM call needed. chat_id (single id or
    list) aggregates across one or more chats. handle_id/contact_id (single
    or list) scopes to a person instead -- e.g. total messages exchanged with
    someone across every chat/handle they use. At least one of chat_id,
    handle_id, contact_id must be given. Pass is_from_me to narrow to only
    sent or only received messages (e.g. 'how many messages have I sent X').
    date_from/date_to accept an ISO date ('2026-01-01') or full ISO datetime
    ('2026-01-01T13:00:00'); date_to covers the whole unit named, so
    date_to='2026-01-01' includes all of Jan 1. Dates are stored in UTC and a
    bound carrying an offset ('...-05:00', '...Z') is converted for you. Pass
    group_by_chat=True to get one row per chat_id instead of a single pooled
    total -- useful for comparing activity across chats. Each row also includes
    attachment_count (individual attachments, so a message with three photos
    counts three) and reaction_count. Every filter applies to all three counts
    alike -- is_from_me=True means messages, attachments, and reactions you
    sent."""
    scope_requested = chat_id is not None or handle_id is not None or contact_id is not None
    if not scope_requested:
        raise ValueError("must provide chat_id, handle_id, or contact_id")
    with contextlib.closing(_connect()) as con:
        chat_ids = _as_id_list(chat_id)
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if not chat_ids and not handle_ids:
            # scope was requested but resolved to nothing -- zeroed stats,
            # not an unfiltered aggregate over the whole corpus.
            return [] if group_by_chat else dict(_ZERO_STATS)

        filters = (is_from_me, date_from, date_to)
        clauses, params = _stats_filters("", chat_ids, handle_ids, *filters)
        where = " AND ".join(clauses)

        if group_by_chat:
            # Both subqueries correlate on the outer chat_id, which already
            # carries the chat scope -- so they only re-apply the handle/
            # is_from_me/date filters, not chat_ids.
            att_clauses, att_params = _stats_filters("am.", None, handle_ids, *filters)
            att_sql = (
                "SELECT COUNT(*) FROM attachments a JOIN messages am ON a.message_id = am.id "
                f"WHERE {' AND '.join(['am.chat_id = messages.chat_id'] + att_clauses)}"
            )
            reaction_clauses, reaction_params = _stats_filters("r.", None, handle_ids, *filters)
            reaction_sql = (
                "SELECT COUNT(*) FROM reactions r WHERE "
                f"{' AND '.join(['r.chat_id = messages.chat_id'] + reaction_clauses)}"
            )
            rows = con.execute(f"""
                SELECT
                    chat_id,
                    COUNT(*) AS message_count,
                    MIN(date) AS first_message,
                    MAX(date) AS last_message,
                    SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) AS from_me,
                    SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) AS from_them,
                    ({att_sql}) AS attachment_count,
                    ({reaction_sql}) AS reaction_count
                FROM messages WHERE {where}
                GROUP BY chat_id
                ORDER BY chat_id
            """, att_params + reaction_params + params).fetchall()
            return [dict(r) for r in rows]

        att_clauses, att_params = _stats_filters("am.", chat_ids, handle_ids, *filters)
        att_sql = (
            "SELECT COUNT(*) FROM attachments a JOIN messages am ON a.message_id = am.id "
            f"WHERE {' AND '.join(att_clauses)}"
        )
        reaction_clauses, reaction_params = _stats_filters("r.", chat_ids, handle_ids, *filters)
        reaction_where = " AND ".join(reaction_clauses)

        row = con.execute(f"""
            SELECT
                COUNT(*) AS message_count,
                MIN(date) AS first_message,
                MAX(date) AS last_message,
                SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) AS from_me,
                SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) AS from_them,
                ({att_sql}) AS attachment_count,
                (SELECT COUNT(*) FROM reactions r WHERE {reaction_where}) AS reaction_count
            FROM messages WHERE {where}
        """, att_params + reaction_params + params).fetchone()
    return dict(row)

