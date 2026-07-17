"""Group-chat structural analytics: get_group_dynamics -- per-member message
share, participation inequality (gini), lurkers, session igniters, reply
adjacency, and per-member one-vs-rest language style matching. Split out of
analytics_language.py; the lexicon/LSM primitives the two share
(_lexicon_rows, _weighted_rates, _lsm_score, _TEXT_MESSAGE_CLAUSES) still
live there.

The tool returns one dict of small sections (leaderboards capped at
MAX_LEADERBOARD_ROWS, series at MAX_SERIES_BUCKETS), so a single call
answers a whole family of questions without paging raw rows. chat_id is
required, accepts a list (split threads), and must be group chats."""

import collections
import contextlib
import datetime

from config import ADJACENCY_MINUTES
from db import CONTACT_LABEL_SQL
from function_words import tokenize

from ._shared import (
    MAX_LEADERBOARD_ROWS,
    MAX_SERIES_BUCKETS,
    _append_id_filters,
    _as_id_list,
    _connect,
    _date_clauses,
    _id_filter_clause,
    _participant_roster,
    _validate_include,
    member_key,
)
from .analytics_language import (
    _TEXT_MESSAGE_CLAUSES,
    _lexicon_rows,
    _lsm_score,
    _weighted_rates,
)

_MAX_MEMBER_ROWS = 25
_GROUP_SECTIONS = ("members", "gini", "lurkers", "igniters", "adjacency", "group_lsm")


def _member_key(is_from_me, contact_id, handle_id):
    """Merge key for a group member (thin wrapper over _shared.member_key):
    'me' for my side, else the handle's contact (duplicate handles collapse),
    else the bare handle, else the "unknown" bucket -- the same contract as
    analytics.get_chat_participant_stats."""
    return member_key(is_from_me, contact_id, handle_id)


def _gini(values):
    """Gini coefficient of a list of non-negative counts (0 = perfectly
    equal shares, ->1 = one member sends everything)."""
    values = sorted(values)
    n, total = len(values), sum(values)
    if n == 0 or total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(values))
    return round((2 * cumulative) / (n * total) - (n + 1) / n, 4)


def _group_chat_error(con, chat_ids):
    """Error dict when the chat list isn't purely group chats, else None.
    Every chat in the list is validated, not just the first: the sections
    merge the rows into one population, so a 1:1 chat smuggled into the list
    would silently skew every member share."""
    placeholders = ",".join("?" * len(chat_ids))
    found = {r["id"]: r["is_group"] for r in con.execute(
        f"SELECT id, is_group FROM chats WHERE id IN ({placeholders})", chat_ids)}
    missing = [c for c in chat_ids if c not in found]
    if missing:
        return {"error": f"chat {missing[0]} not found"}
    direct = [c for c in chat_ids if not found[c]]
    if direct:
        return {"error": f"chat {direct[0]} is a 1:1 chat -- get_group_dynamics "
                         "covers group chats; use get_chat_participant_stats or "
                         "get_word_trends for direct chats"}
    return None


def _group_messages(con, chat_ids, date_from, date_to):
    """Text messages in scope with sender identity columns, date-ordered."""
    clauses, params = list(_TEXT_MESSAGE_CLAUSES), []
    _append_id_filters(clauses, params, chat_ids, None)
    _date_clauses(clauses, params, date_from, date_to)
    return con.execute(
        f"""
        SELECT m.date, m.text, m.is_from_me, m.handle_id, h.contact_id,
               CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE {CONTACT_LABEL_SQL} END AS sender
        FROM messages m
        LEFT JOIN handles h ON h.id = m.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE {' AND '.join(clauses)}
        ORDER BY m.date, m.id
        """,
        params,
    ).fetchall()


def _member_tallies(msgs):
    """Per-member message/word counts, labels, and per-year counts."""
    counts, words, labels = collections.Counter(), collections.Counter(), {}
    yearly = collections.defaultdict(collections.Counter)  # year -> member counts
    for m in msgs:
        key = _member_key(m["is_from_me"], m["contact_id"], m["handle_id"])
        counts[key] += 1
        words[key] += len(tokenize(m["text"]))
        labels[key] = m["sender"]
        yearly[m["date"][:4]][key] += 1
    return counts, words, labels, yearly


def _members_section(counts, words, labels, total):
    return [
        {
            "member": labels[key],
            "is_me": key == "me",
            "messages": counts[key],
            "share": round(counts[key] / total, 3) if total else 0.0,
            "words": words[key],
        }
        for key in sorted(counts, key=lambda k: -counts[k])[:_MAX_MEMBER_ROWS]
    ]


def _gini_section(counts, yearly):
    # Inequality among members who spoke at all; silent participants
    # are the lurkers section's job, not a gini inflator.
    return {
        "gini": _gini(list(counts.values())),
        "n_members": len(counts),
        "drift": [
            {"period": year, "gini": _gini(list(members.values())),
             "n_members": len(members)}
            for year, members in sorted(yearly.items())
        ][:MAX_SERIES_BUCKETS],
    }


def _lurkers_section(con, chat_ids, counts, labels):
    roster = {}
    for r in _participant_roster(con, chat_ids):
        roster[_member_key(False, r["contact_id"], r["handle_id"])] = r["label"]
    lurkers = [{"member": label, "messages": 0}
               for key, label in sorted(roster.items()) if key not in counts]
    if not lurkers and len(counts) >= 10:
        # No fully silent member: surface the bottom decile instead.
        cutoff = sorted(counts.values())[max(0, len(counts) // 10 - 1)]
        lurkers = [{"member": labels[k], "messages": n}
                   for k, n in counts.items() if n <= cutoff]
    return sorted(lurkers, key=lambda r: r["messages"])[:MAX_LEADERBOARD_ROWS]


def _igniters_section(con, chat_ids, date_from, date_to):
    s_clauses, s_params = [], []
    clause, cparams = _id_filter_clause("s.chat_id", chat_ids)
    s_clauses.append(clause)
    s_params.extend(cparams)
    _date_clauses(s_clauses, s_params, date_from, date_to, column="s.first_date")
    starts, start_labels = collections.Counter(), {}
    for r in con.execute(
        f"""
        SELECT s.initiator_is_me AS is_me, s.initiator_handle_id AS handle_id,
               h.contact_id,
               CASE WHEN s.initiator_is_me = 1 THEN 'Me'
                    ELSE {CONTACT_LABEL_SQL} END AS who
        FROM sessions s
        LEFT JOIN handles h ON h.id = s.initiator_handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE {' AND '.join(s_clauses)}
        """,
        s_params,
    ):
        key = _member_key(r["is_me"], r["contact_id"], r["handle_id"])
        starts[key] += 1
        start_labels[key] = r["who"]
    if not starts:
        return {"available": False,
                "note": "no sessions for this chat -- sessions are built by ingest"}
    n_sessions = sum(starts.values())
    return {
        "n_sessions": n_sessions,
        "leaderboard": [
            {"member": start_labels[k], "sessions_started": n,
             "share": round(n / n_sessions, 3)}
            for k, n in starts.most_common(MAX_LEADERBOARD_ROWS)
        ],
    }


def _adjacency_section(msgs, labels):
    gap = datetime.timedelta(minutes=ADJACENCY_MINUTES)
    pairs = collections.Counter()
    prev_key = prev_dt = None
    for m in msgs:
        key = _member_key(m["is_from_me"], m["contact_id"], m["handle_id"])
        dt = datetime.datetime.fromisoformat(m["date"])
        if prev_key is not None and key != prev_key and dt - prev_dt <= gap:
            pairs[(key, prev_key)] += 1
        prev_key, prev_dt = key, dt
    return {
        "gap_minutes": ADJACENCY_MINUTES,
        "pairs": [
            {"replier": labels.get(a, a), "replied_to": labels.get(b, b),
             "count": n}
            for (a, b), n in pairs.most_common(MAX_LEADERBOARD_ROWS)
        ],
    }


def _group_lsm_section(con, chat_ids, labels, date_from, date_to):
    lex = _lexicon_rows(con, chat_ids, date_from, date_to)
    handle_contact = dict(con.execute("SELECT id, contact_id FROM handles"))
    by_member = collections.defaultdict(list)
    for r in lex:
        if r["handle_id"] is None:
            by_member["me"].append(r)
        else:
            by_member[_member_key(
                False, handle_contact.get(r["handle_id"]), r["handle_id"])].append(r)
    scores = []
    for key, rows in by_member.items():
        rest = [r for k, other in by_member.items() if k != key for r in other]
        mine, others = _weighted_rates(rows), _weighted_rates(rest)
        if mine is None or others is None:
            continue
        scores.append({"member": labels.get(key, key),
                       "lsm_score": round(_lsm_score(mine, others)[0], 4)})
    if not scores:
        return {"available": False,
                "note": "no lexicon rows for this chat -- "
                        "person_period_lexicon is built by ingest"}
    scores.sort(key=lambda r: -r["lsm_score"])
    return {"members": scores[:_MAX_MEMBER_ROWS]}


def get_group_dynamics(chat_id, date_from=None, date_to=None, include=None):
    """Structural analytics for one GROUP chat. Sections (include=[...]
    selects): members (per-member message share and word counts, 'Me'
    included), gini (participation inequality, 0=equal, plus per-year
    drift), lurkers (participants with no -- or bottom-decile -- messages),
    igniters (who starts sessions, from the sessions table), adjacency (who
    replies to whom: consecutive different-sender messages within
    ADJACENCY_MINUTES), group_lsm (per-member one-vs-rest language style
    matching). chat_id is required and accepts a list (pass every chat row
    of a split thread); a 1:1 chat returns an error -- use
    get_chat_participant_stats / get_word_trends for those."""
    chat_ids = _as_id_list(chat_id)
    if not chat_ids:
        raise ValueError("get_group_dynamics requires a chat_id (or list)")
    wanted = _validate_include(include, _GROUP_SECTIONS, "get_group_dynamics")

    def want(name):
        return name in wanted

    with contextlib.closing(_connect()) as con:
        error = _group_chat_error(con, chat_ids)
        if error:
            return error
        result = {"chat_ids": chat_ids}

        msgs = _group_messages(con, chat_ids, date_from, date_to)
        counts, words, labels, yearly = _member_tallies(msgs)
        total = sum(counts.values())

        if want("members"):
            result["members"] = _members_section(counts, words, labels, total)
        if want("gini"):
            result["gini"] = _gini_section(counts, yearly)
        if want("lurkers"):
            result["lurkers"] = _lurkers_section(con, chat_ids, counts, labels)
        if want("igniters"):
            result["igniters"] = _igniters_section(con, chat_ids, date_from, date_to)
        if want("adjacency"):
            result["adjacency"] = _adjacency_section(msgs, labels)
        if want("group_lsm"):
            result["group_lsm"] = _group_lsm_section(con, chat_ids, labels,
                                                     date_from, date_to)

    return result
