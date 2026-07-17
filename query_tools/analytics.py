"""Aggregate analytics tools: conversation dynamics (initiations, response
times, silences), time-bucketed activity histograms, reaction leaderboards,
per-participant group-chat stats, and word/phrase frequency. Added for the
quantitative/superlative questions the retrieval tools can't answer from a
50-row sample (see docs/eval_report_2026-07-16.md) -- each returns one
compact, pre-aggregated result instead of forcing the model to page raw rows.

Scope semantics differ deliberately between tools: dynamics/histogram/
reactions/participant tools resolve a person to their *chats*
(_resolve_chat_scope) so both sides of the conversation are counted --
from-me rows carry handle_id=0, so a sender-level handle filter would drop
the "me" side entirely. get_word_frequency keeps *sender* scope
(_resolve_scope): "their most-used words" means messages they sent.
"""

import collections
import contextlib
import datetime
import re
import statistics

from config import DOUBLE_TEXT_MIN_MINUTES, DYNAMICS_GAP_MINUTES
from db import CONTACT_LABEL_SQL, TAPBACK_NAMES, tapback_name

from ._shared import (
    MAX_HISTOGRAM_ROWS,
    MAX_MESSAGE_LIMIT,
    MAX_SILENCES,
    MAX_STATS_ROWS,
    MAX_WORD_FREQ_TOP,
    _MSG_COLS,
    _MSG_JOIN,
    _append_id_filters,
    _as_id_list,
    _connect,
    _id_filter_clause,
    _normalize_date_bound,
    _resolve_chat_scope,
    _resolve_scope,
    _truncate_text_field,
)


def _date_clauses(clauses, params, date_from, date_to, column="m.date"):
    """Append normalized date-range clauses (plain-string compares -- never
    wrap the column in SQL date functions, that kills idx_messages_chat_date)."""
    if date_from is not None:
        clauses.append(f"{column} >= ?")
        params.append(_normalize_date_bound(date_from))
    if date_to is not None:
        clauses.append(f"{column} < ?")
        params.append(_normalize_date_bound(date_to, upper=True))


def _restrict_to_direct_chats(con, chat_ids):
    """Keep only 1:1 chats (is_group = 0) from a chat_id list."""
    if not chat_ids:
        return chat_ids
    placeholders = ",".join("?" * len(chat_ids))
    rows = con.execute(
        f"SELECT id FROM chats WHERE id IN ({placeholders}) AND is_group = 0",
        chat_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _side(is_from_me):
    return "me" if is_from_me else "them"


def _response_stats(samples):
    """Median/p90 seconds over a list of response gaps (guarding tiny lists)."""
    if not samples:
        return {"median_seconds": None, "p90_seconds": None, "count": 0}
    p90 = statistics.quantiles(samples, n=10)[8] if len(samples) >= 2 else samples[0]
    return {
        "median_seconds": round(statistics.median(samples), 1),
        "p90_seconds": round(p90, 1),
        "count": len(samples),
    }


def _zero_dynamics():
    return {
        "message_count": 0,
        "from_me": 0,
        "from_them": 0,
        "first_message": None,
        "last_message": None,
        "initiations": {"me": 0, "them": 0},
        "response_seconds": {"me": _response_stats([]), "them": _response_stats([])},
        "double_texts": {"me": 0, "them": 0},
        "longest_silences": [],
    }


def get_conversation_dynamics(chat_id=None, handle_id=None, contact_id=None,
                              date_from=None, date_to=None,
                              gap_minutes=None, top_silences=5):
    """Who-texts-first analytics for a conversation: per-side message counts,
    initiations (first message after a silence longer than gap_minutes,
    default DYNAMICS_GAP_MINUTES), median/p90 response times per side (a
    response = the other side replying within gap_minutes), double-text
    counts (same sender again after >= DOUBLE_TEXT_MIN_MINUTES), and the
    longest silences with who broke each. Some scope is required: chat_id
    (single or list -- pass every chat row of a split iMessage/SMS thread),
    or handle_id/contact_id for a person, which covers that person's *1:1*
    chats only -- initiation/response attribution is two-sided, so group
    chats are included only via an explicit chat_id."""
    gap_minutes = float(gap_minutes) if gap_minutes else DYNAMICS_GAP_MINUTES
    gap = datetime.timedelta(minutes=gap_minutes)
    double_gap = datetime.timedelta(minutes=DOUBLE_TEXT_MIN_MINUTES)
    top_silences = max(1, min(int(top_silences), MAX_SILENCES))

    if not _as_id_list(chat_id) and not _as_id_list(handle_id) and not _as_id_list(contact_id):
        raise ValueError(
            "get_conversation_dynamics needs a scope: chat_id, handle_id, or contact_id"
        )

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if not empty and chat_ids and _as_id_list(chat_id) is None:
            # Person scope without an explicit chat_id: 1:1 chats only.
            chat_ids = _restrict_to_direct_chats(con, chat_ids)
            empty = not chat_ids
        result = {"chat_ids": chat_ids, "gap_minutes": gap_minutes}
        if empty:
            result.update(_zero_dynamics())
            return result

        clauses, params = ["m.date IS NOT NULL"], []
        _append_id_filters(clauses, params, chat_ids, None)
        _date_clauses(clauses, params, date_from, date_to)
        rows = con.execute(
            f"SELECT m.date, m.is_from_me FROM messages m "
            f"WHERE {' AND '.join(clauses)} ORDER BY m.date, m.id",
            params,
        ).fetchall()

    counts = {"me": 0, "them": 0}
    initiations = {"me": 0, "them": 0}
    double_texts = {"me": 0, "them": 0}
    responses = {"me": [], "them": []}
    silences = []  # (gap_timedelta, prev_date, cur_date, broken_by)

    prev_date = prev_side = None
    for date_str, is_from_me in rows:
        side = _side(is_from_me)
        counts[side] += 1
        cur = datetime.datetime.fromisoformat(date_str)
        if prev_date is None:
            initiations[side] += 1
        else:
            delta = cur - prev_date
            silences.append((delta, prev_date, cur, side))
            if delta > gap:
                initiations[side] += 1
            elif side != prev_side:
                responses[side].append(delta.total_seconds())
            elif delta >= double_gap:
                double_texts[side] += 1
        prev_date, prev_side = cur, side

    silences.sort(key=lambda s: s[0], reverse=True)
    longest = [
        {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "days": round(delta.total_seconds() / 86400, 1),
            "broken_by": side,
        }
        for delta, start, end, side in silences[:top_silences]
    ]

    result.update(
        message_count=len(rows),
        from_me=counts["me"],
        from_them=counts["them"],
        first_message=rows[0][0] if rows else None,
        last_message=rows[-1][0] if rows else None,
        initiations=initiations,
        response_seconds={s: _response_stats(responses[s]) for s in ("me", "them")},
        double_texts=double_texts,
        longest_silences=longest,
    )
    return result


_BUCKET_EXPRS = {
    # Bucket expressions live in SELECT/GROUP BY only; WHERE stays plain-string
    # date compares so idx_messages_chat_date keeps working.
    "month": "substr(m.date, 1, 7)",
    "day": "substr(m.date, 1, 10)",
    "hour": "substr(m.date, 12, 2)",
    "dow": "strftime('%w', m.date)",
}
_DOW_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def get_activity_histogram(chat_id=None, handle_id=None, contact_id=None,
                           bucket="month", date_from=None, date_to=None):
    """Message counts bucketed by 'month', 'day', 'dow' (day of week), or
    'hour', each split into from_me/from_them. handle_id/contact_id scope to
    that person's chats (both sides counted); no scope aggregates the whole
    corpus. All buckets are UTC. Rows are capped at MAX_HISTOGRAM_ROWS with a
    truncation flag -- narrow the date range or coarsen the bucket to see
    the rest."""
    expr = _BUCKET_EXPRS.get(bucket)
    if expr is None:
        raise ValueError(f"unknown bucket {bucket!r}: expected one of "
                         f"{', '.join(sorted(_BUCKET_EXPRS))}")

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return {"bucket": bucket, "timezone": "UTC", "rows": [], "truncated": False}

        clauses, params = ["m.date IS NOT NULL"], []
        _append_id_filters(clauses, params, chat_ids, None)
        _date_clauses(clauses, params, date_from, date_to)
        rows = con.execute(
            f"""
            SELECT {expr} AS bucket, COUNT(*) AS total,
                   SUM(m.is_from_me) AS from_me,
                   COUNT(*) - SUM(m.is_from_me) AS from_them
            FROM messages m
            WHERE {' AND '.join(clauses)}
            GROUP BY 1 ORDER BY 1
            LIMIT ?
            """,
            params + [MAX_HISTOGRAM_ROWS + 1],
        ).fetchall()

    truncated = len(rows) > MAX_HISTOGRAM_ROWS
    rows = [dict(r) for r in rows[:MAX_HISTOGRAM_ROWS]]
    if bucket == "dow":
        for r in rows:
            r["bucket"] = _DOW_NAMES[int(r["bucket"])]
    result = {"bucket": bucket, "timezone": "UTC", "rows": rows, "truncated": truncated}
    if truncated:
        result["note"] = "more buckets exist -- narrow the date range or use a coarser bucket"
    return result


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


def get_top_reacted_messages(chat_id=None, handle_id=None, contact_id=None,
                             date_from=None, date_to=None, reaction_type=None,
                             is_from_me=None, limit=10):
    """Messages ranked by tapbacks RECEIVED (most first) -- 'my most-loved
    message', 'whose texts get the most laughs'. reaction_type filters to one
    tapback (name like 'laugh'/'love' or raw code); is_from_me filters whose
    messages were reacted to (true = mine). handle_id/contact_id scope to
    that person's chats. Each row is a message (text truncated -- expand via
    get_full_messages) plus reaction_count and a reactions_by_type breakdown.
    Tapback removals and non-tapback associated messages are excluded."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return []

        clauses = ["r.associated_message_type BETWEEN 2000 AND 2999"]
        params = []
        if reaction_type is not None:
            clauses.append("r.associated_message_type = ?")
            params.append(_resolve_reaction_type(reaction_type))
        _append_id_filters(clauses, params, chat_ids, None)
        if is_from_me is not None:
            clauses.append("m.is_from_me = ?")
            params.append(int(is_from_me))
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


def get_chat_participant_stats(chat_id, date_from=None, date_to=None):
    """Per-participant stats for a chat (group-chat 'awards show' numbers):
    one row per member -- including 'Me' and zero-message members -- with
    messages_sent, first/last message date, reactions_given, and
    reactions_received broken down by tapback type. chat_id is required and
    accepts a list: pass every chat row of a split iMessage/SMS thread to
    get one merged table. Rows are sorted by messages_sent."""
    chat_ids = _as_id_list(chat_id)
    if not chat_ids:
        raise ValueError("get_chat_participant_stats requires a chat_id (or list)")

    chat_clause, chat_params = _id_filter_clause("m.chat_id", chat_ids)

    def scoped(extra_clauses, extra_params, date_column="m.date"):
        clauses, params = [chat_clause] + extra_clauses, chat_params + list(extra_params)
        _date_clauses(clauses, params, date_from, date_to, column=date_column)
        return " AND ".join(clauses), params

    # Participants are merged in Python, not SQL: from-me rows carry
    # handle_id=0 (no handles row), and a contact may own several duplicate
    # handle rows -- so the merge key is "me" for is_from_me, else the
    # handle's contact_id, else the bare handle. The rare is_from_me=0 +
    # handle_id=0 rows (system noise) land in an "unknown" bucket.
    def merge_key(is_from_me, contact_id_, handle_id_):
        if is_from_me:
            return "me"
        if contact_id_ is not None:
            return f"contact:{contact_id_}"
        if handle_id_:
            return f"handle:{handle_id_}"
        return "unknown"

    participants = {}

    def row_for(key):
        return participants.setdefault(key, {
            "participant": "Me" if key == "me" else "(unknown)",
            "is_me": key == "me",
            "contact_id": None,
            "handle_ids": set(),
            "messages_sent": 0,
            "first_message": None,
            "last_message": None,
            "reactions_given": 0,
            "reactions_received": 0,
            "reactions_received_by_type": collections.Counter(),
        })

    with contextlib.closing(_connect()) as con:
        # Seed from chat_participants so silent members still get a row.
        placeholders = ",".join("?" * len(chat_ids))
        seed = con.execute(
            f"""
            SELECT DISTINCT h.id AS handle_id, h.contact_id,
                   {CONTACT_LABEL_SQL} AS label
            FROM chat_participants cp
            JOIN handles h ON h.id = cp.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE cp.chat_id IN ({placeholders})
            """,
            chat_ids,
        ).fetchall()
        for r in seed:
            row = row_for(merge_key(False, r["contact_id"], r["handle_id"]))
            row["participant"] = r["label"]
            row["contact_id"] = r["contact_id"]
            row["handle_ids"].add(r["handle_id"])
        row_for("me")

        where, params = scoped(["m.date IS NOT NULL"], [])
        for r in con.execute(
            f"""
            SELECT m.is_from_me, h.contact_id, m.handle_id,
                   COUNT(*) AS n, MIN(m.date) AS first, MAX(m.date) AS last
            FROM messages m LEFT JOIN handles h ON h.id = m.handle_id
            WHERE {where} GROUP BY 1, 2, 3
            """,
            params,
        ).fetchall():
            row = row_for(merge_key(r["is_from_me"], r["contact_id"], r["handle_id"]))
            row["messages_sent"] += r["n"]
            if row["first_message"] is None or r["first"] < row["first_message"]:
                row["first_message"] = r["first"]
            if row["last_message"] is None or r["last"] > row["last_message"]:
                row["last_message"] = r["last"]

        # Reactions received: attributed to the *target message's* sender.
        where, params = scoped(["r.associated_message_type BETWEEN 2000 AND 2999"], [])
        for r in con.execute(
            f"""
            SELECT m.is_from_me, h.contact_id, m.handle_id,
                   r.associated_message_type AS code, COUNT(*) AS n
            FROM reactions r
            JOIN messages m ON m.guid = r.target_message_guid
            LEFT JOIN handles h ON h.id = m.handle_id
            WHERE {where} GROUP BY 1, 2, 3, 4
            """,
            params,
        ).fetchall():
            row = row_for(merge_key(r["is_from_me"], r["contact_id"], r["handle_id"]))
            row["reactions_received"] += r["n"]
            row["reactions_received_by_type"][tapback_name(r["code"])] += r["n"]

        # Reactions given: attributed to the *reactor* (r.is_from_me=1 rows
        # carry the counterparty's handle_id, so is_from_me decides first).
        r_chat_clause, r_chat_params = _id_filter_clause("r.chat_id", chat_ids)
        clauses = [r_chat_clause, "r.associated_message_type BETWEEN 2000 AND 2999"]
        params = list(r_chat_params)
        _date_clauses(clauses, params, date_from, date_to, column="r.date")
        where = " AND ".join(clauses)
        for r in con.execute(
            f"""
            SELECT r.is_from_me, h.contact_id, r.handle_id, COUNT(*) AS n
            FROM reactions r LEFT JOIN handles h ON h.id = r.handle_id
            WHERE {where} GROUP BY 1, 2, 3
            """,
            params,
        ).fetchall():
            row = row_for(merge_key(r["is_from_me"], r["contact_id"], r["handle_id"]))
            row["reactions_given"] += r["n"]

    rows = sorted(participants.values(), key=lambda r: -r["messages_sent"])[:MAX_STATS_ROWS]
    for row in rows:
        row["handle_ids"] = sorted(row["handle_ids"])
        row["reactions_received_by_type"] = dict(row["reactions_received_by_type"])
    return rows


_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset("""
a about after all am an and any are as at be been but by can can't could did
do don't for from get got had has have he her him his how i i'll i'm i've if
in is it it's just like me my no not of oh on one or our out say she so some
than that that's the their them then there they this to u ur up us was we
we're were what when who will with would you you'll you're your
""".split())


def _tokenize(text):
    """Lowercased word tokens. Curly apostrophes (what iOS actually types)
    are normalized to straight ones first, so contractions stay whole
    ("don't") instead of shedding fragments ("don", "t")."""
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def get_word_frequency(chat_id=None, handle_id=None, contact_id=None, is_from_me=None,
                       date_from=None, date_to=None, ngram=1, top=25, min_count=2,
                       include_stopwords=False):
    """Most frequent words (ngram=1) or phrases (ngram=2..3) in message text,
    with counts. Unlike the other analytics tools this scopes by SENDER:
    handle_id/contact_id means messages that person sent, is_from_me=true
    means messages you sent -- 'what words do I overuse' is
    is_from_me=true with no other scope. Common English stopwords are
    excluded unless include_stopwords=true (phrases are dropped only when
    every word is a stopword). Scans every matching message -- corpus-wide
    calls take a few seconds."""
    ngram = max(1, min(int(ngram), 3))
    top = max(1, min(int(top), MAX_WORD_FREQ_TOP))

    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return {"rows": [], "ngram": ngram, "messages_scanned": 0,
                    "stopwords_excluded": not include_stopwords}

        # is_emote/item_type exclude tapback-ish and system rows (belt and
        # braces -- ingest already splits tapbacks into reactions).
        clauses = ["m.text IS NOT NULL", "m.text != ''",
                   "COALESCE(m.is_emote, 0) = 0", "COALESCE(m.item_type, 0) = 0"]
        params = []
        _append_id_filters(clauses, params, chat_ids, handle_ids)
        if is_from_me is not None:
            clauses.append("m.is_from_me = ?")
            params.append(int(is_from_me))
        _date_clauses(clauses, params, date_from, date_to)

        counter = collections.Counter()
        scanned = 0
        for (text,) in con.execute(
            f"SELECT m.text FROM messages m WHERE {' AND '.join(clauses)}", params
        ):
            scanned += 1
            tokens = _tokenize(text)
            if ngram == 1:
                counter.update(
                    t for t in tokens if include_stopwords or t not in _STOPWORDS
                )
            else:
                for i in range(len(tokens) - ngram + 1):
                    gram = tokens[i:i + ngram]
                    if include_stopwords or not all(t in _STOPWORDS for t in gram):
                        counter[" ".join(gram)] += 1

    rows = [
        {"ngram": gram, "count": n}
        for gram, n in counter.most_common()
        if n >= min_count
    ][:top]
    return {"rows": rows, "ngram": ngram, "messages_scanned": scanned,
            "stopwords_excluded": not include_stopwords}
