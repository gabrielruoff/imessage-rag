"""Language analytics tools: phrase-over-time trends, emoji habits,
we/question/laugh rates, each chat's distinctive terms (inside-joke
candidates with birth certificates), language style matching (LSM), and
LLM-annotated emotion/valence trends -- plus the group-chat structural view
(message share, gini, lurkers, igniters, reply adjacency, per-member LSM).

Both tools return one dict of small sections (leaderboards capped at
MAX_LEADERBOARD_ROWS, series at MAX_SERIES_BUCKETS with month->year
auto-coarsening), so a single call answers a whole family of questions
without paging raw rows. Scope semantics match analytics.py: a person
(handle_id/contact_id) resolves to *that person's chats* via
_resolve_chat_scope, so both sides of the conversation are counted."""

import collections
import contextlib
import datetime
import json

from config import ADJACENCY_MINUTES
from db import CONTACT_LABEL_SQL
from function_words import FUNCTION_WORDS, tokenize

from ._shared import (
    MAX_LEADERBOARD_ROWS,
    MAX_SERIES_BUCKETS,
    _REAL_MESSAGE_CLAUSES,
    _append_id_filters,
    _as_id_list,
    _connect,
    _date_clauses,
    _id_filter_clause,
    _normalize_date_bound,
    _resolve_chat_scope,
    _truncate_text_field,
    _validate_include,
)

_FW_CATEGORIES = tuple(FUNCTION_WORDS)
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in _FW_CATEGORIES)
_TOP_EMOJI = 15
_MAX_MEMBER_ROWS = 25
_EPS = 1e-9  # LSM denominator guard: both-zero rates score a perfect 1.0

_TREND_SECTIONS = ("phrase_trend", "emoji", "rates", "distinctive_terms", "lsm",
                   "emotions", "valence")
_GROUP_SECTIONS = ("members", "gini", "lurkers", "igniters", "adjacency", "group_lsm")

# The population every section counts: real conversation messages with text
# (see _shared._REAL_MESSAGE_CLAUSES).
_TEXT_MESSAGE_CLAUSES = (
    "m.date IS NOT NULL", "m.text IS NOT NULL", "m.text != ''", *_REAL_MESSAGE_CLAUSES,
)


def _month_bounds(date_from, date_to):
    """Inclusive 'YYYY-MM' bounds for filtering person_period_lexicon.month.
    The upper bound backs off one microsecond from the half-open datetime
    bound so date_to='2026-04-30' includes April but not May."""
    lo = hi = None
    if date_from is not None:
        lo = _normalize_date_bound(date_from)[:7]
    if date_to is not None:
        upper = datetime.datetime.fromisoformat(_normalize_date_bound(date_to, upper=True))
        hi = (upper - datetime.timedelta(microseconds=1)).strftime("%Y-%m")
    return lo, hi


def _period_len(granularity):
    if granularity not in ("month", "year"):
        raise ValueError(f"unknown granularity {granularity!r}: expected 'month' or 'year'")
    return 7 if granularity == "month" else 4


def _lexicon_rows(con, chat_ids, date_from, date_to):
    """person_period_lexicon rows in scope (handle_id NULL = my side)."""
    clauses, params = [], []
    clause, cparams = _id_filter_clause("chat_id", chat_ids)
    if clause:
        clauses.append(clause)
        params.extend(cparams)
    lo, hi = _month_bounds(date_from, date_to)
    if lo is not None:
        clauses.append("month >= ?")
        params.append(lo)
    if hi is not None:
        clauses.append("month <= ?")
        params.append(hi)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return con.execute(f"SELECT * FROM person_period_lexicon {where}", params).fetchall()


def _weighted_rates(rows):
    """Word-count-weighted mean of the 9 category rates over lexicon rows,
    or None when the rows carry no words at all."""
    total_words = sum(r["n_words"] for r in rows)
    if not total_words:
        return None
    return {
        cat: sum(r[col] * r["n_words"] for r in rows) / total_words
        for cat, col in zip(_FW_CATEGORIES, _FW_COLUMNS)
    }


def _lsm_score(rates_a, rates_b):
    """Mean over categories of LSM_c = 1 - |ra-rb| / (ra+rb+eps)."""
    per_cat = {
        cat: 1 - abs(rates_a[cat] - rates_b[cat]) / (rates_a[cat] + rates_b[cat] + _EPS)
        for cat in _FW_CATEGORIES
    }
    return sum(per_cat.values()) / len(per_cat), per_cat


# ---------------------------------------------------------------------------
# get_word_trends sections
# ---------------------------------------------------------------------------

def _phrase_trend(con, phrase, chat_ids, date_from, date_to, granularity):
    n = _period_len(granularity)
    clauses, params = list(_TEXT_MESSAGE_CLAUSES), []
    _append_id_filters(clauses, params, chat_ids, None)
    _date_clauses(clauses, params, date_from, date_to)
    # Phrase-quote the FTS query so the model's raw words never reach FTS5's
    # query parser (a stray AND/NEAR/quote would change or break the match).
    match = '"' + phrase.replace('"', '""') + '"'
    rows = con.execute(
        f"""
        SELECT substr(m.date, 1, {n}) AS period,
               SUM(m.is_from_me) AS me, COUNT(*) - SUM(m.is_from_me) AS them
        FROM messages_fts f JOIN messages m ON m.id = f.rowid
        WHERE messages_fts MATCH ? AND {' AND '.join(clauses)}
        GROUP BY 1 ORDER BY 1
        """,
        [match] + params,
    ).fetchall()
    series = [{"period": r["period"], "me": r["me"], "them": r["them"]} for r in rows]
    note = None
    if len(series) > MAX_SERIES_BUCKETS and granularity == "month":
        by_year = collections.defaultdict(lambda: {"me": 0, "them": 0})
        for row in series:
            year = by_year[row["period"][:4]]
            year["me"] += row["me"]
            year["them"] += row["them"]
        series = [{"period": p, **v} for p, v in sorted(by_year.items())]
        note = "coarsened month -> year to fit the series cap"
    if len(series) > MAX_SERIES_BUCKETS:
        series, note = series[-MAX_SERIES_BUCKETS:], "series truncated to the newest buckets"
    section = {
        "phrase": phrase,
        "total_messages": sum(r["me"] + r["them"] for r in series),
        "series": series,
    }
    if note:
        section["note"] = note
    return section


def _emoji_section(lex):
    counter = collections.Counter()
    for r in lex:
        if r["emoji_json"]:
            counter.update(json.loads(r["emoji_json"]))
    return {
        "top": [{"emoji": e, "count": n} for e, n in counter.most_common(_TOP_EMOJI)],
        "total_emoji": sum(counter.values()),
    }


def _rates_section(lex, granularity):
    def build(n):
        agg = {}
        for r in lex:
            period = r["month"][:n]
            side = "me" if r["handle_id"] is None else "them"
            sums = agg.setdefault(period, {"me": [0] * 4, "them": [0] * 4})[side]
            sums[0] += r["n_messages"]
            sums[1] += r["we_count"]
            sums[2] += r["question_count"]
            sums[3] += r["laugh_count"]
        series = []
        for period in sorted(agg):
            row = {"period": period}
            for side in ("me", "them"):
                n_msgs, we, question, laugh = agg[period][side]
                row[side] = {
                    "n_messages": n_msgs,
                    "we_per_100": round(100 * we / n_msgs, 2) if n_msgs else None,
                    "question_per_100": round(100 * question / n_msgs, 2) if n_msgs else None,
                    "laugh_per_100": round(100 * laugh / n_msgs, 2) if n_msgs else None,
                }
            series.append(row)
        return series

    series, note = build(_period_len(granularity)), None
    if len(series) > MAX_SERIES_BUCKETS and granularity == "month":
        series, note = build(4), "coarsened month -> year to fit the series cap"
    if len(series) > MAX_SERIES_BUCKETS:
        series, note = series[-MAX_SERIES_BUCKETS:], "series truncated to the newest buckets"
    section = {"series": series}
    if note:
        section["note"] = note
    return section


def _distinctive_terms(con, chat_ids):
    if not chat_ids:
        return {"available": False,
                "note": "distinctive terms are per chat -- pass chat_id or a person scope"}
    clause, params = _id_filter_clause("chat_id", chat_ids)
    # Curated keepers (inside_joke/nickname) outrank uncurated candidates;
    # LLM-rejected terms (ordinary/noise) are dropped entirely.
    rows = con.execute(
        f"""
        SELECT chat_id, term, lift, n_uses, first_message_id,
               coined_by_handle_id, coined_is_me, llm_verdict
        FROM dyad_terms
        WHERE {clause} AND (llm_verdict IS NULL
                            OR llm_verdict IN ('inside_joke', 'nickname'))
        ORDER BY (llm_verdict IS NOT NULL) DESC, lift DESC
        LIMIT ?
        """,
        params + [MAX_LEADERBOARD_ROWS],
    ).fetchall()
    if not rows:
        return {"available": False,
                "note": "no distinctive terms for this scope (dyad_terms is built by ingest)"}

    # Birth certificates: the coining message, fetched in one batch.
    msg_ids = [r["first_message_id"] for r in rows if r["first_message_id"] is not None]
    births = {}
    if msg_ids:
        placeholders = ",".join("?" * len(msg_ids))
        births = {m["id"]: m for m in con.execute(
            f"""
            SELECT m.id, m.date, m.text,
                   CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE {CONTACT_LABEL_SQL} END AS sender
            FROM messages m
            LEFT JOIN handles h ON h.id = m.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE m.id IN ({placeholders})
            """,
            msg_ids,
        )}
    terms = []
    for r in rows:
        birth = births.get(r["first_message_id"])
        terms.append({
            "term": r["term"],
            "chat_id": r["chat_id"],
            "lift": round(r["lift"], 2),
            "n_uses": r["n_uses"],
            "verdict": r["llm_verdict"],
            "curated": r["llm_verdict"] is not None,
            "first_message_id": r["first_message_id"],
            "date": birth["date"] if birth else None,
            "text": birth["text"] if birth else None,
            "coined_by": "Me" if r["coined_is_me"] else (birth["sender"] if birth else None),
        })
    return {"terms": _truncate_text_field(terms)}


def _lsm_section(lex, granularity):
    me_rows = [r for r in lex if r["handle_id"] is None]
    them_rows = [r for r in lex if r["handle_id"] is not None]
    rates_me, rates_them = _weighted_rates(me_rows), _weighted_rates(them_rows)
    if rates_me is None or rates_them is None:
        return {"available": False,
                "note": "LSM needs lexicon rows with words on both sides "
                        "(person_period_lexicon is built by ingest)"}
    score, per_cat = _lsm_score(rates_me, rates_them)

    def build(n):
        series = []
        for period in sorted({r["month"][:n] for r in lex}):
            wm = _weighted_rates([r for r in me_rows if r["month"][:n] == period])
            wt = _weighted_rates([r for r in them_rows if r["month"][:n] == period])
            if wm is None or wt is None:
                continue  # one-sided period: no matching to measure
            series.append({"period": period, "score": round(_lsm_score(wm, wt)[0], 4)})
        return series

    # Trajectory is monthly by contract; coarsen to years only past the cap.
    trajectory, note = build(7), None
    if len(trajectory) > MAX_SERIES_BUCKETS:
        trajectory, note = build(4), "coarsened month -> year to fit the series cap"
    section = {
        "score": round(score, 4),
        "per_category": {cat: round(v, 4) for cat, v in per_cat.items()},
        "trajectory": trajectory,
    }
    if note:
        section["note"] = note
    return section


def _annotation_rows(con, chat_ids, date_from, date_to):
    """chunk_annotations joined to chunks for scope/date (chunks.first_date
    is the same unlabeled-UTC string form as messages.date)."""
    clauses, params = [], []
    clause, cparams = _id_filter_clause("c.chat_id", chat_ids)
    if clause:
        clauses.append(clause)
        params.extend(cparams)
    _date_clauses(clauses, params, date_from, date_to, column="c.first_date")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return con.execute(
        f"""
        SELECT a.valence, a.emotions_json, c.first_date
        FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id
        {where}
        """,
        params,
    ).fetchall()


_NO_ANNOTATIONS = {"available": False,
                   "note": "no chunk annotations in scope -- run annotate.py first"}


def _emotions_section(ann):
    counter = collections.Counter()
    for r in ann:
        if r["emotions_json"]:
            counter.update(json.loads(r["emotions_json"]))
    if not counter:
        return dict(_NO_ANNOTATIONS)
    return {
        "top": [{"emotion": e, "count": n}
                for e, n in counter.most_common(MAX_LEADERBOARD_ROWS)],
        "chunks_annotated": len(ann),
    }


def _valence_section(ann, granularity):
    samples = [(r["first_date"], r["valence"]) for r in ann if r["valence"] is not None]
    if not samples:
        return dict(_NO_ANNOTATIONS)

    def build(n):
        agg = collections.defaultdict(list)
        for date, valence in samples:
            agg[date[:n]].append(valence)
        return [
            {"period": p, "mean_valence": round(sum(v) / len(v), 3), "n_chunks": len(v)}
            for p, v in sorted(agg.items())
        ]

    series, note = build(_period_len(granularity)), None
    if len(series) > MAX_SERIES_BUCKETS and granularity == "month":
        series, note = build(4), "coarsened month -> year to fit the series cap"
    section = {"series": series}
    if note:
        section["note"] = note
    return section


def get_word_trends(phrase=None, chat_id=None, handle_id=None, contact_id=None,
                    date_from=None, date_to=None, granularity="year", include=None):
    """Language-over-time analytics in one call. Sections (include=[...]
    selects; default is every applicable one): phrase_trend (how often
    `phrase` appears per period, split me/them -- only when phrase is given),
    emoji (top emoji with counts, both sides), rates (we/question/laugh
    messages per 100 per period, me vs them), distinctive_terms (the chat's
    inside-joke/nickname candidates with lift, uses, and the coining message
    -- needs a chat scope), lsm (language style matching me-vs-them: score,
    per-category, monthly trajectory), emotions/valence (LLM-annotated, only
    after annotate.py has run). granularity is 'month' or 'year'.
    handle_id/contact_id scope to that person's chats (both sides counted);
    no scope means the whole corpus."""
    _period_len(granularity)  # validate early
    wanted = _validate_include(include, _TREND_SECTIONS, "get_word_trends")

    def want(name):
        return name in wanted

    result = {"granularity": granularity}
    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        result["chat_ids"] = chat_ids
        if empty:
            result["note"] = "scope matched nothing"
            return result

        if phrase and want("phrase_trend"):
            result["phrase_trend"] = _phrase_trend(
                con, phrase, chat_ids, date_from, date_to, granularity)
        lex = None
        if want("emoji") or want("rates") or want("lsm"):
            lex = _lexicon_rows(con, chat_ids, date_from, date_to)
        if want("emoji"):
            result["emoji"] = _emoji_section(lex)
        if want("rates"):
            result["rates"] = _rates_section(lex, granularity)
        if want("distinctive_terms"):
            result["distinctive_terms"] = _distinctive_terms(con, chat_ids)
        if want("lsm"):
            result["lsm"] = _lsm_section(lex, granularity)
        if want("emotions") or want("valence"):
            ann = _annotation_rows(con, chat_ids, date_from, date_to)
            if want("emotions"):
                result["emotions"] = _emotions_section(ann)
            if want("valence"):
                result["valence"] = _valence_section(ann, granularity)
    return result


# ---------------------------------------------------------------------------
# get_group_dynamics
# ---------------------------------------------------------------------------

def _member_key(is_from_me, contact_id, handle_id):
    """Merge key for a group member: 'me' for my side, else the handle's
    contact (duplicate handles collapse), else the bare handle. Mirrors
    analytics.get_chat_participant_stats."""
    if is_from_me:
        return "me"
    if contact_id is not None:
        return f"contact:{contact_id}"
    if handle_id:
        return f"handle:{handle_id}"
    return "unknown"


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
    placeholders = ",".join("?" * len(chat_ids))
    roster = {}
    for r in con.execute(
        f"""
        SELECT DISTINCT h.id AS handle_id, h.contact_id,
               {CONTACT_LABEL_SQL} AS label
        FROM chat_participants cp
        JOIN handles h ON h.id = cp.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE cp.chat_id IN ({placeholders})
        """,
        chat_ids,
    ):
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
