"""Message-level retrieval tools: structured (SQL-facet) search, FTS5
keyword search, context expansion around a hit, reactions/tapbacks, and
full-text recovery for truncated results."""

import contextlib

from db import CONTACT_LABEL_SQL, tapback_name

from ._shared import (
    MAX_CONTEXT_MESSAGES,
    MAX_MESSAGE_LIMIT,
    _MSG_COLS,
    _MSG_JOIN,
    _append_id_filters,
    _as_id_list,
    _connect,
    _date_clauses,
    _envelope,
    _resolve_scope,
    _truncate_text_field,
)


def search_messages_sql(chat_id=None, handle_id=None, contact_id=None, is_from_me=None,
                         date_from=None, date_to=None, limit=50) -> dict:
    """Structured filter search -- no text matching, just facets. chat_id and
    handle_id each accept a single id or a list of ids. contact_id (single or
    list) scopes to a person across every handle they own -- the way to search
    one person's messages regardless of which phone/email/chat they used.
    date_from/date_to accept an ISO date ('2026-01-01') or full ISO datetime
    ('2026-01-01T13:00:00') -- a full datetime narrows to a time-of-day range,
    not just a day. date_to covers the whole unit named, so date_to='2026-01-01'
    includes all of Jan 1. Message dates are stored in UTC; a bound carrying an
    offset ('2026-01-01T13:00:00-05:00', '...Z') is converted to UTC for you,
    and a bound without one is taken as already-UTC.

    Returns an envelope, not a bare list: rows carries the newest
    `limit` matching messages, total_matching the full match count -- when
    truncated is true, rows is only a sample of the population."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))

    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return _envelope([], 0, "newest-first")

        clauses, params = [], []
        _append_id_filters(clauses, params, chat_ids, handle_ids)
        if is_from_me is not None:
            clauses.append("m.is_from_me = ?")
            params.append(int(is_from_me))
        _date_clauses(clauses, params, date_from, date_to)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = con.execute(
            f"SELECT COUNT(*) FROM messages m {where}", params
        ).fetchone()[0]
        sql = f"""
            SELECT {_MSG_COLS}
            FROM messages m
            {_MSG_JOIN}
            {where}
            ORDER BY m.date DESC
            LIMIT ?
        """
        rows = con.execute(sql, params + [limit]).fetchall()
    return _envelope(_truncate_text_field([dict(r) for r in rows]), total, "newest-first")


def search_messages_fts(query: str, chat_id=None, handle_id=None, contact_id=None,
                        limit=50) -> dict:
    """Keyword/phrase search via FTS5 (supports FTS5 query syntax:
    'foo AND bar', '"exact phrase"', 'foo NOT bar', etc.). chat_id/handle_id
    accept a single id or a list; contact_id (single or list) scopes to a
    person across every handle they own.

    Returns an envelope, not a bare list: rows carries the `limit`
    best-ranked hits, total_matching the full match count -- when truncated
    is true, rows is only a sample of the matches."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))
    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return _envelope([], 0, "best-match-first")

        clauses, params = ["messages_fts MATCH ?"], [query]
        _append_id_filters(clauses, params, chat_ids, handle_ids)
        where = " AND ".join(clauses)

        total = con.execute(
            f"SELECT COUNT(*) FROM messages_fts f JOIN messages m ON m.id = f.rowid "
            f"WHERE {where}",
            params,
        ).fetchone()[0]
        sql = f"""
            SELECT {_MSG_COLS}
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            {_MSG_JOIN}
            WHERE {where}
            ORDER BY rank
            LIMIT ?
        """
        rows = con.execute(sql, params + [limit]).fetchall()
    return _envelope(
        _truncate_text_field([dict(r) for r in rows]), total, "best-match-first"
    )


def get_message_context(message_id: int, before=10, after=10) -> list:
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
            SELECT {_MSG_COLS}
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


def get_message_reactions(target_message_guid) -> list:
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
                   CASE WHEN r.is_from_me = 1 THEN 'Me' ELSE {CONTACT_LABEL_SQL} END AS reactor,
                   r.associated_message_type, r.date
            FROM reactions r
            LEFT JOIN handles h ON h.id = r.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE r.target_message_guid IN ({placeholders})
            ORDER BY r.date
            """,
            guids,
        ).fetchall()
    results = [dict(r) for r in rows]
    for r in results:
        # Human-readable tapback name next to the raw Apple type code.
        r["reaction"] = tapback_name(r["associated_message_type"])
    return results


def get_full_messages(ids) -> list:
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
            SELECT {_MSG_COLS}
            FROM messages m
            {_MSG_JOIN}
            WHERE m.id IN ({placeholders})
        """, ids).fetchall()
    return [dict(r) for r in rows]
