"""Directory and stats tools: the bridges from human names to the
handle_id/contact_id/chat_id values the search tools need, plus cheap
aggregate message counts."""

import contextlib

from db import CONTACT_LABEL_SQL

from ._shared import (
    MAX_CHATS_LIMIT,
    MAX_CONTACTS_LIMIT,
    MAX_STATS_ROWS,
    _append_id_filters,
    _as_id_list,
    _connect,
    _normalize_date_bound,
    _resolve_handle_filter,
    _scope_requested,
)


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


def list_chats(query=None, chat_id=None, handle_id=None, contact_id=None, limit=50):
    """List chats (conversations) with resolved participant names -- the bridge
    from a person/chat name to a chat_id, and the way to label a chat_id. Pass
    chat_id (single id or list) to label specific chats you already have ids
    for (e.g. rows from get_chat_summary_stats group_by_chat=True). Pass
    `query` to filter by participant name or chat_identifier (substring match,
    fuzzy). Pass handle_id/contact_id (single id or list) for an exact match on
    every chat that person actually participates in -- prefer this over `query`
    once you already have a contact_id from search_contacts_by_name, since a
    name substring can over- or under-match."""
    limit = max(1, min(limit, MAX_CHATS_LIMIT))
    like = f"%{query}%" if query else None
    with contextlib.closing(_connect()) as con:
        handle_requested = _scope_requested(handle_id, contact_id)
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if handle_requested and not handle_ids:
            return []
        params = {"like": like, "limit": limit}
        id_filter_sql = ""
        chat_ids = _as_id_list(chat_id)
        if chat_ids:
            placeholders = ",".join(f":cid{i}" for i in range(len(chat_ids)))
            id_filter_sql += f" AND ch.id IN ({placeholders})"
            params.update({f"cid{i}": v for i, v in enumerate(chat_ids)})
        if handle_ids:
            placeholders = ",".join(f":hid{i}" for i in range(len(handle_ids)))
            id_filter_sql += f"""
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
                   GROUP_CONCAT({CONTACT_LABEL_SQL}, ', ') AS participants
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
                            group_by_chat=False, limit=25):
    """Cheap aggregate stats -- no LLM call needed. chat_id (single id or
    list) aggregates across one or more chats. handle_id/contact_id (single
    or list) scopes to a person instead -- e.g. total messages exchanged with
    someone across every chat/handle they use. With no scope at all the whole
    database is aggregated (e.g. 'how many messages total'); group_by_chat
    then ranks every chat. Pass is_from_me to narrow to only
    sent or only received messages (e.g. 'how many messages have I sent X').
    date_from/date_to accept an ISO date ('2026-01-01') or full ISO datetime
    ('2026-01-01T13:00:00'); date_to covers the whole unit named, so
    date_to='2026-01-01' includes all of Jan 1. Dates are stored in UTC and a
    bound carrying an offset ('...-05:00', '...Z') is converted for you. Pass
    group_by_chat=True to get one row per chat_id instead of a single pooled
    total -- useful for comparing activity across chats; rows are the top
    `limit` chats by message_count (an unlimited dump of every chat would
    overflow the chat model's context window). Each row also includes
    attachment_count (individual attachments, so a message with three photos
    counts three) and reaction_count. Every filter applies to all three counts
    alike -- is_from_me=True means messages, attachments, and reactions you
    sent."""
    with contextlib.closing(_connect()) as con:
        chat_ids = _as_id_list(chat_id)
        handle_requested = _scope_requested(handle_id, contact_id)
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if handle_requested and not handle_ids:
            # a person scope was requested but resolved to nothing -- zeroed
            # stats, not an unfiltered aggregate over the whole corpus.
            return [] if group_by_chat else dict(_ZERO_STATS)

        filters = (is_from_me, date_from, date_to)
        clauses, params = _stats_filters("", chat_ids, handle_ids, *filters)
        where = " AND ".join(clauses) or "1=1"

        if group_by_chat:
            # The CTE ranks chats on the messages table alone, so the
            # correlated attachment/reaction subqueries run only for the
            # <= MAX_STATS_ROWS rows that survive the limit -- not for every
            # chat in the corpus. Both subqueries correlate on the surviving
            # row's chat_id, which already carries the chat scope, so they
            # only re-apply the handle/is_from_me/date filters, not chat_ids.
            att_clauses, att_params = _stats_filters("am.", None, handle_ids, *filters)
            att_sql = (
                "SELECT COUNT(*) FROM attachments a JOIN messages am ON a.message_id = am.id "
                f"WHERE {' AND '.join(['am.chat_id = t.chat_id'] + att_clauses)}"
            )
            reaction_clauses, reaction_params = _stats_filters("r.", None, handle_ids, *filters)
            reaction_sql = (
                "SELECT COUNT(*) FROM reactions r WHERE "
                f"{' AND '.join(['r.chat_id = t.chat_id'] + reaction_clauses)}"
            )
            rows = con.execute(f"""
                WITH top_chats AS (
                    SELECT
                        chat_id,
                        COUNT(*) AS message_count,
                        MIN(date) AS first_message,
                        MAX(date) AS last_message,
                        SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) AS from_me,
                        SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) AS from_them
                    FROM messages WHERE {where}
                    GROUP BY chat_id
                    ORDER BY message_count DESC
                    LIMIT ?
                )
                SELECT t.*,
                    ({att_sql}) AS attachment_count,
                    ({reaction_sql}) AS reaction_count
                FROM top_chats t
                ORDER BY t.message_count DESC
            """, params + [max(1, min(limit, MAX_STATS_ROWS))]
                 + att_params + reaction_params).fetchall()
            return [dict(r) for r in rows]

        att_clauses, att_params = _stats_filters("am.", chat_ids, handle_ids, *filters)
        att_sql = (
            "SELECT COUNT(*) FROM attachments a JOIN messages am ON a.message_id = am.id "
            f"WHERE {' AND '.join(att_clauses) or '1=1'}"
        )
        reaction_clauses, reaction_params = _stats_filters("r.", chat_ids, handle_ids, *filters)
        reaction_where = " AND ".join(reaction_clauses) or "1=1"

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
