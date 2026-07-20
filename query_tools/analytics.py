"""Aggregate analytics tools over the base schema: per-participant chat
stats and word/phrase frequency. Added for the quantitative/superlative
questions the retrieval tools can't answer from a 50-row sample (see
docs/eval_report_2026-07-16.md) -- each returns one compact, pre-aggregated
result instead of forcing the model to page raw rows. The richer
time/social/language aggregates live in analytics_time.py /
analytics_social.py / analytics_language.py.

Scope semantics differ deliberately between tools: get_chat_participant_stats
covers whole chats so both sides of the conversation are counted -- from-me
rows carry handle_id=0, so a sender-level handle filter would drop the "me"
side entirely. get_word_frequency keeps *sender* scope (_resolve_scope):
"their most-used words" means messages they sent.
"""

import collections
import contextlib
import re

from db import tapback_name

from ._shared import (
    MAX_STATS_ROWS,
    MAX_WORD_FREQ_TOP,
    _REAL_MESSAGE_CLAUSES,
    _append_id_filters,
    _as_exclusion_list,
    _as_id_list,
    _connect,
    _date_clauses,
    _exclude_id_filters,
    _id_filter_clause,
    _participant_roster,
    _resolve_scope,
    member_key,
)


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
    # handle rows -- _shared.member_key is the single source of that merge
    # key. This module's contract keeps the rare is_from_me=0 + handle_id=0
    # rows (system noise) in an "unknown" bucket (member_key's default).
    merge_key = member_key

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
        for r in _participant_roster(con, chat_ids):
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
                       include_stopwords=False, exclude_chat_id=None):
    """Most frequent words (ngram=1) or phrases (ngram=2..3) in message text,
    with counts. Unlike the other analytics tools this scopes by SENDER:
    handle_id/contact_id means messages that person sent, is_from_me=true
    means messages you sent -- 'what words do I overuse' is
    is_from_me=true with no other scope. Common English stopwords are
    excluded unless include_stopwords=true (phrases are dropped only when
    every word is a stopword). Scans every matching message -- corpus-wide
    calls take a few seconds. exclude_chat_id (scalar or list; not in the
    LLM schema -- wrapped.py's carve-out) drops those chats from the scan."""
    ngram = max(1, min(int(ngram), 3))
    top = max(1, min(int(top), MAX_WORD_FREQ_TOP))

    with contextlib.closing(_connect()) as con:
        chat_ids, handle_ids, empty = _resolve_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return {"rows": [], "ngram": ngram, "messages_scanned": 0,
                    "stopwords_excluded": not include_stopwords}

        clauses = ["m.text IS NOT NULL", "m.text != ''", *_REAL_MESSAGE_CLAUSES]
        params = []
        _append_id_filters(clauses, params, chat_ids, handle_ids)
        _exclude_id_filters(clauses, params, _as_exclusion_list(exclude_chat_id))
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
