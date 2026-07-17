"""Sessions pipeline: group each chat's renderable messages into engagement
sessions -- contiguous runs split when the silence between two messages
exceeds SESSION_GAP_MINUTES -- persisted in the `sessions` table with per-side
counts and initiator/ender attribution. This is the derived table behind
get_response_dynamics' initiation/session/ghosting stats (and group-chat
igniter leaderboards): consumers read session rows, there is no
messages.session_id column.

Like chunks, sessions are a rebuildable artifact (reset_db deletes them; the
append-only rule's carve-out applies). Incremental behavior is deliberately
coarse: any chat that gained renderable messages since the last run has its
sessions deleted and rebuilt whole -- the largest chat rebuilds in
milliseconds, nothing expensive depends on session rows, and a whole-chat
rebuild is immune to tail-surgery subtleties. A changed gap setting (recorded
in ingest_state) forces a rebuild of every chat, so the table can never mix
splitting rules.
"""

import datetime
import sqlite3

from config import SESSION_GAP_MINUTES
from db import get_state, set_state
from query_tools._shared import _REAL_MESSAGE_CLAUSES

# A message belongs in a session when it is renderable conversation content:
# dated, carrying text or an attachment, and a real message (not a legacy
# emote or a system row -- the shared _REAL_MESSAGE_CLAUSES).
#
# This is deliberately one gate STRICTER than the analytics aggregates in
# query_tools: they keep every dated real message, while a session also
# requires text or an attachment. The difference is the handful of rows that
# are dated and real but carry no renderable content (~32 of 350k on the
# development corpus) -- they cannot start, end, or extend a conversation, so
# they must not open a session or claim an initiation. Consequently a
# session's n_messages and get_response_dynamics' message_count are counts of
# *near*-identical populations, not identical ones; don't assert equality.
_RENDERABLE_WHERE = " AND ".join((
    "m.date IS NOT NULL",
    "(m.text IS NOT NULL OR m.has_attachment = 1)",
    *_REAL_MESSAGE_CLAUSES,
))


def _insert_session(tgt, chat_id, msgs):
    """Insert one session row for a run of (id, handle_id, is_from_me, date)
    tuples. initiator/ender handle_id is NULL when that side is me -- from-me
    rows carry handle_id=0 (the counterparty's slot is meaningless there)."""
    first_id, first_handle, first_me, first_date = msgs[0]
    last_id, last_handle, last_me, last_date = msgs[-1]
    tgt.execute(
        "INSERT INTO sessions(chat_id, first_message_id, last_message_id, "
        "first_date, last_date, n_messages, n_from_me, initiator_handle_id, "
        "initiator_is_me, ender_handle_id, ender_is_me) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chat_id, first_id, last_id, first_date, last_date,
            len(msgs), sum(1 for m in msgs if m[2]),
            None if first_me else first_handle, int(bool(first_me)),
            None if last_me else last_handle, int(bool(last_me)),
        ),
    )


def build_sessions(tgt: sqlite3.Connection, gap_minutes=None) -> int:
    """Rebuild the sessions of every chat with renderable messages newer than
    the last run (all chats when the recorded gap setting changed), splitting
    each chat's (date, id)-ordered stream on gaps > gap_minutes (default
    SESSION_GAP_MINUTES). Returns the number of sessions built."""
    if gap_minutes is None:
        gap_minutes = SESSION_GAP_MINUTES
    gap = datetime.timedelta(minutes=float(gap_minutes))

    last_seen = int(get_state(tgt, "sessions_max_message_id") or 0)
    prev_gap = get_state(tgt, "sessions_gap_minutes")
    # A gap change invalidates every chat's splits; None (first run) rebuilds
    # everything anyway since last_seen is 0.
    force_all = prev_gap is None or float(prev_gap) != float(gap_minutes)

    id_clause = "" if force_all else " AND m.id > ?"
    params = [] if force_all else [last_seen]
    chat_ids = [r[0] for r in tgt.execute(
        f"SELECT DISTINCT m.chat_id FROM messages m "
        f"WHERE {_RENDERABLE_WHERE} AND m.chat_id IS NOT NULL{id_clause}",
        params,
    ).fetchall()]

    n_built = 0
    for cid in chat_ids:
        tgt.execute("DELETE FROM sessions WHERE chat_id = ?", (cid,))
        msgs = tgt.execute(
            f"SELECT m.id, m.handle_id, m.is_from_me, m.date FROM messages m "
            f"WHERE m.chat_id = ? AND {_RENDERABLE_WHERE} ORDER BY m.date, m.id",
            (cid,),
        ).fetchall()
        current = []
        prev_dt = None
        for row in msgs:
            cur_dt = datetime.datetime.fromisoformat(row[3])
            if current and cur_dt - prev_dt > gap:
                _insert_session(tgt, cid, current)
                n_built += 1
                current = []
            current.append(row)
            prev_dt = cur_dt
        if current:
            _insert_session(tgt, cid, current)
            n_built += 1

    max_id = tgt.execute(
        f"SELECT MAX(m.id) FROM messages m WHERE {_RENDERABLE_WHERE}"
    ).fetchone()[0]
    set_state(tgt, "sessions_max_message_id", int(max_id or 0))
    set_state(tgt, "sessions_gap_minutes", gap_minutes)
    tgt.commit()

    if n_built:
        print(f"Built {n_built} sessions across {len(chat_ids)} chats "
              f"(gap {gap_minutes} min)")
    else:
        print("Sessions up to date (0 built)")
    return n_built
