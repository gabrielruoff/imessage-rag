"""Social analytics tools: get_reaction_stats (tapbacks given/received with
per-type breakdowns and rates, reciprocity, time-to-react medians, the
funniest-contact leaderboard, and a hall-of-fame of most-reacted messages)
and get_milestones (first/last/nth messages, landmark ordinals,
anniversaries, the longest silence and who broke it, earliest phrase use).
Each returns one compact dict of pre-aggregated sections instead of forcing
the model to page raw rows.

Scope semantics: handle_id/contact_id resolve to that person's *chats*
(_resolve_chat_scope) so both sides of the conversation are counted --
from-me rows carry handle_id=0, so a sender-level handle filter would drop
the "me" side entirely.

Reaction counting always filters associated_message_type BETWEEN 2000 AND
2999: 3000-range rows are tapback *removals* and other codes are non-tapback
associated messages -- neither may inflate a count (see db.TAPBACK_NAMES).
"""

import collections
import contextlib
import datetime
import statistics

import stats
from db import CONTACT_LABEL_SQL, TAPBACK_NAMES, tapback_name

from ._shared import (
    MAX_LEADERBOARD_ROWS,
    MAX_MESSAGE_LIMIT,
    _MSG_COLS,
    _MSG_JOIN,
    _REAL_MESSAGE_CLAUSES,
    _append_id_filters,
    _as_exclusion_list,
    _connect,
    _date_clauses,
    _exclude_id_filters,
    _id_filter_clause,
    _resolve_chat_scope,
    _truncate_text_field,
    _validate_include,
    member_key,
)

# Only tapback *additions*; removals (3000s) and strays (2, 3, 1000, ...) are
# never counted anywhere in this module.
_TAPBACK_ADD_CLAUSE = "r.associated_message_type BETWEEN 2000 AND 2999"
_LAUGH_CODE = 2003

# Contract-fixed eligibility gates, not env tunables: they define what the
# sections *mean* (a "funniest contact" rate over 3 reactions is noise), so
# they are deliberately not overridable per deployment.
_FUNNIEST_MIN_REACTIONS = 20     # reactions on my messages to enter `funniest`
_ANNIVERSARY_MIN_MESSAGES = 100  # lifetime messages to enter unscoped anniversaries
_ANNIVERSARY_WINDOW_DAYS = 30    # look-ahead window after reference_date
_LANDMARK_ORDINALS = (1000, 5000, 10000, 25000, 50000)

_REACTION_SECTIONS = ("given", "received", "reciprocity", "time_to_react",
                      "funniest", "hall_of_fame")


def _resolve_reaction_type(reaction_type):
    """Accept a tapback name ('laugh') or raw int code; return the int code."""
    if isinstance(reaction_type, int):
        return reaction_type
    by_name = {name: code for code, name in TAPBACK_NAMES.items()}
    code = by_name.get(str(reaction_type).strip().lower())
    if code is None:
        raise ValueError(
            f"unknown reaction_type {reaction_type!r}: expected one of "
            f"{', '.join(sorted(by_name))} or a raw tapback code"
        )
    return code


def _person_key(contact_id, handle_id):
    """Merge key for grouping per-person rows (thin wrapper over
    _shared.member_key): duplicate handles of one contact collapse;
    from-me/system rows (handle_id 0/None, no contact) have no person and
    return None."""
    return member_key(False, contact_id, handle_id, unknown=None)


def _receipt(row):
    """One message row -> a truncated receipt dict (or None)."""
    return _truncate_text_field([dict(row)])[0] if row else None


def _reaction_clauses(chat_ids, date_from, date_to, type_code,
                      exclude_chat_ids=None):
    """Shared WHERE tail for reaction-counting queries: tapback-adds only, on
    a real target message, optional single type, chat scope on the target
    message, and date bounds on the *reaction* date (a reaction is an event of
    its own).

    _REAL_MESSAGE_CLAUSES applies to the joined target message so these
    numerators are drawn from the same population as the _message_counts
    denominators they are divided by."""
    clauses, params = [_TAPBACK_ADD_CLAUSE, *_REAL_MESSAGE_CLAUSES], []
    if type_code is not None:
        clauses.append("r.associated_message_type = ?")
        params.append(type_code)
    _append_id_filters(clauses, params, chat_ids, None)
    _exclude_id_filters(clauses, params, exclude_chat_ids)
    _date_clauses(clauses, params, date_from, date_to, column="r.date")
    return clauses, params


def _message_counts(con, chat_ids, date_from, date_to, exclude_chat_ids=None):
    """(sent, received) real-message counts in scope -- the rate denominators."""
    clauses = ["m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES]
    params = []
    _append_id_filters(clauses, params, chat_ids, None)
    _exclude_id_filters(clauses, params, exclude_chat_ids)
    _date_clauses(clauses, params, date_from, date_to)
    total, sent = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(m.is_from_me), 0) "
        f"FROM messages m WHERE {' AND '.join(clauses)}",
        params,
    ).fetchone()
    return sent, total - sent


def _direction_stats(con, chat_ids, date_from, date_to, type_code,
                     direction_clause, denominator, rate_key,
                     exclude_chat_ids=None):
    """Total + per-type counts for one direction of reactions, with a
    per-100-messages rate against the given denominator."""
    clauses, params = _reaction_clauses(chat_ids, date_from, date_to, type_code,
                                        exclude_chat_ids)
    clauses.append(direction_clause)
    by_type = collections.Counter()
    for code, count in con.execute(
        f"""
        SELECT r.associated_message_type, COUNT(*)
        FROM reactions r
        JOIN messages m ON m.guid = r.target_message_guid
        WHERE {' AND '.join(clauses)}
        GROUP BY 1
        """,
        params,
    ).fetchall():
        by_type[tapback_name(code)] += count
    total = sum(by_type.values())
    rate = round(total * 100.0 / denominator, 2) if denominator else None
    return {"total": total, "by_type": dict(by_type), rate_key: rate}


def _zero_direction(rate_key):
    return {"total": 0, "by_type": {}, rate_key: None}


def _time_to_react(con, chat_ids, date_from, date_to, exclude_chat_ids=None):
    """Median seconds from message to reaction, split by who reacted --
    medians computed in Python (never SQL date math in predicates)."""
    clauses, params = _reaction_clauses(chat_ids, date_from, date_to, None,
                                        exclude_chat_ids)
    clauses += ["m.date IS NOT NULL", "r.date IS NOT NULL"]
    samples = {"me_reacting": [], "them_reacting": []}
    for m_date, r_date, from_me in con.execute(
        f"""
        SELECT m.date, r.date, r.is_from_me
        FROM reactions r
        JOIN messages m ON m.guid = r.target_message_guid
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchall():
        delta = (datetime.datetime.fromisoformat(r_date)
                 - datetime.datetime.fromisoformat(m_date))
        samples["me_reacting" if from_me else "them_reacting"].append(delta.total_seconds())
    return {
        side: {
            "median_seconds": round(statistics.median(vals), 1) if vals else None,
            "count": len(vals),
        }
        for side, vals in samples.items()
    }


def _funniest(con, chat_ids, date_from, date_to, exclude_chat_ids=None):
    """Contacts ranked by laugh-tapbacks per 100 of MY messages in their
    chats (min _FUNNIEST_MIN_REACTIONS reactions on my messages to enter)."""
    clauses, params = _reaction_clauses(chat_ids, date_from, date_to, None,
                                        exclude_chat_ids)
    clauses += ["m.is_from_me = 1", "r.is_from_me = 0"]
    reactors = {}
    for row in con.execute(
        f"""
        SELECT h.contact_id AS contact_id, r.handle_id AS handle_id,
               {CONTACT_LABEL_SQL} AS label,
               SUM(CASE WHEN r.associated_message_type = {_LAUGH_CODE}
                        THEN 1 ELSE 0 END) AS laughs,
               COUNT(*) AS total
        FROM reactions r
        JOIN messages m ON m.guid = r.target_message_guid
        LEFT JOIN handles h ON h.id = r.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE {' AND '.join(clauses)}
        GROUP BY h.contact_id, r.handle_id
        """,
        params,
    ).fetchall():
        key = _person_key(row["contact_id"], row["handle_id"])
        if key is None:
            continue
        entry = reactors.setdefault(key, {
            "participant": row["label"], "contact_id": row["contact_id"],
            "handle_ids": set(), "laughs": 0, "reactions_on_my_messages": 0,
        })
        entry["handle_ids"].add(row["handle_id"])
        entry["laughs"] += row["laughs"]
        entry["reactions_on_my_messages"] += row["total"]

    # "Their chats" spans every handle the contact owns, not just the ones
    # they happened to react from -- expand contact-keyed entries.
    contact_ids = [e["contact_id"] for e in reactors.values() if e["contact_id"] is not None]
    if contact_ids:
        clause, cparams = _id_filter_clause("contact_id", contact_ids)
        by_contact = collections.defaultdict(set)
        for handle_id, cid in con.execute(
            f"SELECT id, contact_id FROM handles WHERE {clause}", cparams
        ).fetchall():
            by_contact[cid].add(handle_id)
        for entry in reactors.values():
            if entry["contact_id"] is not None:
                entry["handle_ids"] |= by_contact[entry["contact_id"]]

    # Denominator: my real messages per chat (scoped + date-bounded), summed
    # over the chats each reactor participates in.
    clauses = ["m.date IS NOT NULL", "m.is_from_me = 1", *_REAL_MESSAGE_CLAUSES]
    params = []
    _append_id_filters(clauses, params, chat_ids, None)
    _exclude_id_filters(clauses, params, exclude_chat_ids)
    _date_clauses(clauses, params, date_from, date_to)
    my_msgs_by_chat = dict(con.execute(
        f"SELECT m.chat_id, COUNT(*) FROM messages m "
        f"WHERE {' AND '.join(clauses)} GROUP BY 1",
        params,
    ).fetchall())
    chats_by_handle = collections.defaultdict(set)
    for handle_id, cid in con.execute(
        "SELECT handle_id, chat_id FROM chat_participants"
    ).fetchall():
        chats_by_handle[handle_id].add(cid)

    rows = []
    for entry in reactors.values():
        if entry["reactions_on_my_messages"] < _FUNNIEST_MIN_REACTIONS:
            continue
        their_chats = set().union(*(chats_by_handle[h] for h in entry["handle_ids"]))
        my_messages = sum(my_msgs_by_chat.get(c, 0) for c in their_chats)
        if not my_messages:
            continue
        rows.append({
            "participant": entry["participant"],
            "contact_id": entry["contact_id"],
            "handle_ids": sorted(entry["handle_ids"]),
            "laughs": entry["laughs"],
            "reactions_on_my_messages": entry["reactions_on_my_messages"],
            "my_messages": my_messages,
            "laughs_per_100_my_messages": round(entry["laughs"] * 100.0 / my_messages, 2),
        })
    rows.sort(key=lambda r: (-r["laughs_per_100_my_messages"], -r["laughs"]))
    return rows[:MAX_LEADERBOARD_ROWS]


def _hall_of_fame(con, chat_ids, date_from, date_to, type_code, limit,
                  exclude_chat_ids=None):
    """Top-`limit` most-reacted messages with receipts and a per-type
    breakdown. Date bounds apply to the *message* date here (the question is
    'best message of that period'), unlike the reaction-event sections."""
    clauses = [_TAPBACK_ADD_CLAUSE, "m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES]
    params = []
    if type_code is not None:
        clauses.append("r.associated_message_type = ?")
        params.append(type_code)
    _append_id_filters(clauses, params, chat_ids, None)
    _exclude_id_filters(clauses, params, exclude_chat_ids)
    _date_clauses(clauses, params, date_from, date_to)
    rows = con.execute(
        f"""
        SELECT {_MSG_COLS}, COUNT(r.id) AS reaction_count
        FROM reactions r
        JOIN messages m ON m.guid = r.target_message_guid
        {_MSG_JOIN}
        WHERE {' AND '.join(clauses)}
        GROUP BY m.id
        ORDER BY reaction_count DESC, m.date DESC
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    results = [dict(r) for r in rows]
    if results:
        # Breakdown is per-type over ALL tapback adds on these messages,
        # even under a reaction_type filter -- ranking narrows, context doesn't.
        guids = [r["guid"] for r in results]
        placeholders = ",".join("?" * len(guids))
        breakdown = collections.defaultdict(collections.Counter)
        for guid, code, n in con.execute(
            f"""
            SELECT target_message_guid, associated_message_type, COUNT(*)
            FROM reactions
            WHERE target_message_guid IN ({placeholders})
              AND associated_message_type BETWEEN 2000 AND 2999
            GROUP BY 1, 2
            """,
            guids,
        ).fetchall():
            breakdown[guid][tapback_name(code)] = n
        for r in results:
            r["reactions_by_type"] = dict(breakdown.get(r["guid"], {}))
    return _truncate_text_field(results)


def get_reaction_stats(chat_id=None, handle_id=None, contact_id=None,
                       date_from=None, date_to=None, include=None,
                       reaction_type=None, limit=10, exclude_chat_id=None) -> dict:
    """Tapback statistics in one call: `given` (reactions I sent: total,
    per-type, rate per 100 messages received), `received` (reactions on my
    messages: total, per-type, rate per 100 messages sent), `reciprocity`
    (given/received ratio), `time_to_react` (median seconds from message to
    reaction, me vs them), `funniest` (contacts ranked by laughs per 100 of
    my messages in their chats), and `hall_of_fame` (top-`limit`
    most-reacted messages, text truncated -- expand via get_full_messages).
    include selects sections (default all); reaction_type (name like
    'laugh'/'love' or raw code) narrows given/received/hall_of_fame;
    handle_id/contact_id scope to that person's chats; date bounds apply to
    the reaction date (message date for hall_of_fame). exclude_chat_id
    (scalar or list; not in the LLM schema -- wrapped.py's carve-out) drops
    those chats from every section."""
    sections = _validate_include(include, _REACTION_SECTIONS, "get_reaction_stats")
    limit = max(1, min(int(limit), MAX_MESSAGE_LIMIT))
    type_code = None if reaction_type is None else _resolve_reaction_type(reaction_type)
    exclude_chat_ids = _as_exclusion_list(exclude_chat_id)

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        result = {"chat_ids": chat_ids}
        if empty:
            zeros = {
                "given": _zero_direction("per_100_received_messages"),
                "received": _zero_direction("per_100_sent_messages"),
                "reciprocity": {"given": 0, "received": 0, "ratio": None},
                "time_to_react": {
                    side: {"median_seconds": None, "count": 0}
                    for side in ("me_reacting", "them_reacting")
                },
                "funniest": [],
                "hall_of_fame": [],
            }
            result.update({name: zeros[name] for name in _REACTION_SECTIONS
                           if name in sections})
            return result

        given = received = None
        if sections & {"given", "received", "reciprocity"}:
            sent_msgs, received_msgs = _message_counts(con, chat_ids, date_from,
                                                       date_to, exclude_chat_ids)
            given = _direction_stats(con, chat_ids, date_from, date_to, type_code,
                                     "r.is_from_me = 1", received_msgs,
                                     "per_100_received_messages", exclude_chat_ids)
            received = _direction_stats(con, chat_ids, date_from, date_to, type_code,
                                        "m.is_from_me = 1", sent_msgs,
                                        "per_100_sent_messages", exclude_chat_ids)
        if "given" in sections:
            result["given"] = given
        if "received" in sections:
            result["received"] = received
        if "reciprocity" in sections:
            ratio = (round(given["total"] / received["total"], 2)
                     if received["total"] else None)
            result["reciprocity"] = {"given": given["total"],
                                     "received": received["total"], "ratio": ratio}
        if "time_to_react" in sections:
            result["time_to_react"] = _time_to_react(con, chat_ids, date_from,
                                                     date_to, exclude_chat_ids)
        if "funniest" in sections:
            result["funniest"] = _funniest(con, chat_ids, date_from, date_to,
                                           exclude_chat_ids)
        if "hall_of_fame" in sections:
            result["hall_of_fame"] = _hall_of_fame(con, chat_ids, date_from, date_to,
                                                   type_code, limit, exclude_chat_ids)
    return result


def _real_message_where(chat_ids):
    """Base clauses/params every milestone query shares: dated, real
    conversation messages in scope."""
    clauses = ["m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES]
    params = []
    _append_id_filters(clauses, params, chat_ids, None)
    return " AND ".join(clauses), params


def _fetch_message(con, where, params, direction="ASC", offset=0):
    """One receipt row at a given rank of the scoped, dated message order."""
    row = con.execute(
        f"SELECT {_MSG_COLS} FROM messages m {_MSG_JOIN} WHERE {where} "
        f"ORDER BY m.date {direction}, m.id {direction} LIMIT 1 OFFSET ?",
        params + [offset],
    ).fetchone()
    return _receipt(row)


def _anniversary_fields(first_iso, ref):
    first_date = datetime.date.fromisoformat(first_iso[:10])
    candidate = stats.next_occurrence(first_date, ref)
    return {
        "next_anniversary": candidate.isoformat(),
        "years": candidate.year - first_date.year,
        "days_until": (candidate - ref).days,
    }


def _upcoming_anniversaries(con, ref, limit):
    """Unscoped anniversaries: contacts whose first-ever message month-day
    falls within _ANNIVERSARY_WINDOW_DAYS after ref (year wrap handled by
    stats.next_occurrence), min _ANNIVERSARY_MIN_MESSAGES lifetime messages
    *from them* -- from-me rows carry no handle, so my side can't be
    attributed to a person without a per-chat expansion."""
    people = {}
    for row in con.execute(
        f"""
        SELECT h.contact_id AS contact_id, m.handle_id AS handle_id,
               {CONTACT_LABEL_SQL} AS label, MIN(m.date) AS first, COUNT(*) AS n
        FROM messages m
        LEFT JOIN handles h ON h.id = m.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE m.date IS NOT NULL AND m.is_from_me = 0
          AND {' AND '.join(_REAL_MESSAGE_CLAUSES)}
        GROUP BY h.contact_id, m.handle_id
        """
    ).fetchall():
        key = _person_key(row["contact_id"], row["handle_id"])
        if key is None:
            continue
        entry = people.setdefault(key, {
            "participant": row["label"], "contact_id": row["contact_id"],
            "first_date": row["first"], "lifetime_messages": 0,
        })
        entry["first_date"] = min(entry["first_date"], row["first"])
        entry["lifetime_messages"] += row["n"]

    rows = []
    for entry in people.values():
        if entry["lifetime_messages"] < _ANNIVERSARY_MIN_MESSAGES:
            continue
        fields = _anniversary_fields(entry["first_date"], ref)
        if fields["days_until"] > _ANNIVERSARY_WINDOW_DAYS:
            continue
        rows.append({**entry, **fields})
    rows.sort(key=lambda r: (r["days_until"], -r["lifetime_messages"]))
    return rows[:limit]


def _longest_silence(con, where, params):
    """Largest gap between consecutive dated real messages in scope, plus the
    reconnection message that broke it. One ordered scan, gap math in Python."""
    rows = con.execute(
        f"SELECT m.id, m.date FROM messages m WHERE {where} ORDER BY m.date, m.id",
        params,
    ).fetchall()
    if len(rows) < 2:
        return {"available": False, "note": "fewer than two messages in scope"}
    best = None  # (seconds, prev_date, cur_date, cur_id)
    _, prev_date = rows[0]
    prev_dt = datetime.datetime.fromisoformat(prev_date)
    for cur_id, cur_date in rows[1:]:
        cur_dt = datetime.datetime.fromisoformat(cur_date)
        seconds = (cur_dt - prev_dt).total_seconds()
        if best is None or seconds > best[0]:
            best = (seconds, prev_date, cur_date, cur_id)
        prev_dt, prev_date = cur_dt, cur_date
    seconds, from_date, to_date, breaker_id = best
    receipt = _fetch_message(con, f"{where} AND m.id = ?", params + [breaker_id])
    return {
        "available": True,
        "from": from_date,
        "to": to_date,
        "days": round(seconds / 86400, 1),
        "broken_by": receipt["sender"] if receipt else None,
        "reconnection_message": receipt,
    }


def _earliest_phrase(con, where, params, phrase):
    """Earliest FTS hit for the phrase in scope. The phrase is always issued
    as one quoted FTS phrase query (embedded double quotes stripped), so user
    text can't inject FTS syntax."""
    fts_query = '"' + str(phrase).replace('"', "") + '"'
    row = con.execute(
        f"""
        SELECT {_MSG_COLS}
        FROM messages_fts f
        JOIN messages m ON m.id = f.rowid
        {_MSG_JOIN}
        WHERE messages_fts MATCH ? AND {where}
        ORDER BY m.date, m.id LIMIT 1
        """,
        [fts_query] + params,
    ).fetchone()
    return {"phrase": phrase, "message": _receipt(row)}


def get_milestones(chat_id=None, handle_id=None, contact_id=None, phrase=None,
                   n=None, reference_date=None, limit=10) -> dict:
    """Relationship milestones in one call: `firsts` (first/last message in
    scope plus each side's first, with text), `nth` (the nth message when n
    is given, else landmark messages at 1000/5000/10000/25000/50000),
    `anniversaries` (scoped: first-contact date + the next anniversary;
    unscoped: contacts whose first-message anniversary falls in the next 30
    days after reference_date, default today), `longest_silence` (largest
    gap between consecutive messages + the reconnection message -- requires
    scope), and `earliest_phrase` (first FTS match, when phrase is given).
    handle_id/contact_id scope to that person's chats."""
    limit = max(1, min(int(limit), MAX_LEADERBOARD_ROWS))
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
    if reference_date is not None:
        ref = datetime.date.fromisoformat(str(reference_date)[:10])
    else:
        ref = datetime.datetime.now(datetime.timezone.utc).date()

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        result = {"chat_ids": chat_ids}
        if empty:
            result["firsts"] = {"first": None, "last": None,
                                "first_from_me": None, "first_from_them": None}
            result["nth"] = ({"total_messages": 0, "n": n, "message": None} if n
                             else {"total_messages": 0, "landmarks": []})
            result["anniversaries"] = {"first_contact": None}
            result["longest_silence"] = {"available": False,
                                         "note": "scope matched nothing"}
            if phrase:
                result["earliest_phrase"] = {"phrase": phrase, "message": None}
            return result

        where, params = _real_message_where(chat_ids)
        result["firsts"] = {
            "first": _fetch_message(con, where, params),
            "last": _fetch_message(con, where, params, direction="DESC"),
            "first_from_me": _fetch_message(con, f"{where} AND m.is_from_me = 1", params),
            "first_from_them": _fetch_message(con, f"{where} AND m.is_from_me = 0", params),
        }

        total = con.execute(
            f"SELECT COUNT(*) FROM messages m WHERE {where}", params
        ).fetchone()[0]
        if n is not None:
            message = _fetch_message(con, where, params, offset=n - 1) if n <= total else None
            result["nth"] = {"total_messages": total, "n": n, "message": message}
        else:
            result["nth"] = {
                "total_messages": total,
                "landmarks": [
                    {"ordinal": k,
                     "message": _fetch_message(con, where, params, offset=k - 1)}
                    for k in _LANDMARK_ORDINALS if k <= total
                ],
            }

        if chat_ids is not None:
            first = result["firsts"]["first"]
            if first is None:
                result["anniversaries"] = {"first_contact": None}
            else:
                result["anniversaries"] = {
                    "first_contact": first["date"],
                    **_anniversary_fields(first["date"], ref),
                }
            result["longest_silence"] = _longest_silence(con, where, params)
        else:
            result["anniversaries"] = {
                "reference_date": ref.isoformat(),
                "window_days": _ANNIVERSARY_WINDOW_DAYS,
                "upcoming": _upcoming_anniversaries(con, ref, limit),
            }
            # A corpus-wide "longest silence" would just measure vacations
            # and dead threads across unrelated chats -- scope required.
            result["longest_silence"] = {
                "available": False,
                "note": "longest_silence needs a chat or person scope",
            }

        if phrase:
            result["earliest_phrase"] = _earliest_phrase(con, where, params, phrase)
    return result
