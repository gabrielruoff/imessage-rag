"""Conflict read tools (RES-46 / C1): find_conflicts (ranked conflict episodes
in scope) and get_conflict (one episode's details + its labeled signal turns).

These read the deterministic finder's output (conflicts / conflict_signals,
written by conflict_detect.py) -- no Ollama, no re-detection. Conflicts exist
only for 1:1 chats by construction; a person/chat scope resolves the same way
as the other analytics tools and is then restricted to direct chats
(_restrict_to_direct_chats -- F3). Escalation labels are eval-gated BANDS
(tentative/likely), never certainties -- surface them as such.

conflicts.first_date/last_date are unlabeled-UTC ISO strings: plain-string
compares only, never SQL date functions in WHERE (see CLAUDE.md gotchas)."""

import contextlib
import json

from ._shared import (
    _connect,
    _date_clauses,
    _envelope,
    _resolve_chat_scope,
    _restrict_to_direct_chats,
)

# Result caps (contract-fixed; this tool's own, like analytics_social's
# eligibility gates). Kept well under TOOL_RESULT_MAX_CHARS: <=15 signal rows,
# each excerpt <=200 chars, is a few KB.
MAX_CONFLICT_ROWS = 25       # find_conflicts page size
MAX_SIGNAL_ROWS = 15         # get_conflict signal rows per page
EXCERPT_CAP = 200            # defensive re-cap on stored excerpts


def _has_conflict_tables(con):
    """True only when both conflict tables exist -- a DB built before this
    feature has neither, and the read tools must return empty, not raise."""
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


def _signals_json(peak_signal_json):
    try:
        data = json.loads(peak_signal_json) if peak_signal_json else {}
    except (TypeError, ValueError):
        data = {}
    return data


def _conflict_summary(con, row):
    """Compact conflict row for find_conflicts (no per-signal excerpts)."""
    peak = _signals_json(row["peak_signal_json"])
    n_signal_rows = con.execute(
        "SELECT COUNT(*) FROM conflict_signals WHERE conflict_id = ? "
        "AND label_class != 'abstain'", (row["id"],)).fetchone()[0]
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "topic": row["topic"],
        "signals": peak.get("signals", []),
        "signal_count": peak.get("signal_count", 0),
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "labeled_signal_turns": n_signal_rows,
        "peak_gap_hours": peak.get("peak_gap_hours"),
    }


def find_conflicts(chat_id=None, handle_id=None, contact_id=None,
                   date_from=None, date_to=None, limit=10):
    """Detected conflict episodes in scope, ranked by deterministic signal
    strength then recency. Each is a tense stretch found from deterministic
    signals WITH RECEIPTS (reply-latency withdrawal, one-sided sessions,
    dislike-tapback runs, volume collapse); `signals` names which fired and
    `signal_count` how many. 1:1 chats only. handle_id/contact_id scope to that
    person's direct chats; date bounds filter on the episode start. corroboration
    is 'none' unless source='llm' annotations backed the tone drop. Returns an
    envelope {rows, total_matching, returned, order, truncated}; expand one with
    get_conflict(conflict_id)."""
    limit = max(1, min(int(limit), MAX_CONFLICT_ROWS))
    order = "signal_count-desc, most-recent-first"
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return _envelope([], 0, order)
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return _envelope([], 0, order)
        if chat_ids is not None:
            chat_ids = _restrict_to_direct_chats(con, chat_ids)
            if not chat_ids:
                return _envelope([], 0, order)

        clauses, params = [], []
        if chat_ids is not None:
            clauses.append(f"chat_id IN ({','.join('?' * len(chat_ids))})")
            params.extend(chat_ids)
        _date_clauses(clauses, params, date_from, date_to, column="first_date")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = con.execute(
            f"SELECT id, chat_id, first_date, last_date, topic, peak_signal_json, "
            f"corroboration, off_text_flag FROM conflicts{where}", params,
        ).fetchall()
        summaries = [_conflict_summary(con, r) for r in rows]

    # Strongest (most distinct deterministic signals) first, then most recent.
    summaries.sort(key=lambda s: (s["signal_count"], s["last_date"] or ""),
                   reverse=True)
    total = len(summaries)
    return _envelope(summaries[:limit], total, order)


def _signal_row(row):
    try:
        top2 = json.loads(row["top2_labels_json"]) if row["top2_labels_json"] else []
    except (TypeError, ValueError):
        top2 = []
    excerpt = row["excerpt"] or ""
    return {
        "message_id": row["message_id"],
        "actor": "me" if row["actor_is_me"] else "them",
        "label_class": row["label_class"],
        "top2": top2,
        "band": row["band"],
        "excerpt": excerpt[:EXCERPT_CAP],
    }


def get_conflict(conflict_id, offset=0, limit=15):
    """One conflict episode: its deterministic evidence (peak_signal), tone
    corroboration, off_text_flag, and up to 15 labeled signal turns (paginated;
    excerpts capped at 200 chars). Labels are eval-gated BANDS (band
    'tentative'/'likely'), never raw scores -- present them as such, and never
    read a resolution/apology into an episode whose off_text_flag is set. Pass
    offset to page further signals."""
    conflict_id = int(conflict_id)
    limit = max(1, min(int(limit), MAX_SIGNAL_ROWS))
    offset = max(0, int(offset))
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, start_session_id, "
            "end_session_id, topic, peak_signal_json, corroboration, "
            "off_text_flag, source FROM conflicts WHERE id = ?", (conflict_id,),
        ).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}
        total = con.execute(
            "SELECT COUNT(*) FROM conflict_signals WHERE conflict_id = ?",
            (conflict_id,)).fetchone()[0]
        sig_rows = con.execute(
            "SELECT message_id, actor_is_me, label_class, top2_labels_json, "
            "band, excerpt FROM conflict_signals WHERE conflict_id = ? "
            "ORDER BY id LIMIT ? OFFSET ?", (conflict_id, limit, offset),
        ).fetchall()

    signals = [_signal_row(r) for r in sig_rows]
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "start_session_id": row["start_session_id"],
        "end_session_id": row["end_session_id"],
        "topic": row["topic"],
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "peak_signal": _signals_json(row["peak_signal_json"]),
        "signals": signals,
        "signals_total": total,
        "returned": len(signals),
        "offset": offset,
        "limit": limit,
        "truncated": total > offset + len(signals),
        "note": "labels are eval-gated bands (tentative/likely), not certainties",
    }
