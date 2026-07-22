"""Language analytics: phrase-over-time trends, emoji habits,
we/question/laugh rates, each chat's distinctive terms (inside-joke
candidates with birth certificates), language style matching (LSM), and
LLM-annotated emotion/valence trends. The group-chat structural view lives
in analytics_group.py, which shares this module's lexicon/LSM primitives.

The tool returns one dict of small sections (leaderboards capped at
MAX_LEADERBOARD_ROWS, series at MAX_SERIES_BUCKETS with month->year
auto-coarsening), so a single call answers a whole family of questions
without paging raw rows. Scope semantics match analytics.py: a person
(handle_id/contact_id) resolves to *that person's chats* via
_resolve_chat_scope, so both sides of the conversation are counted."""

import collections
import contextlib
import datetime
import json

from db import CONTACT_LABEL_SQL
from function_words import FUNCTION_WORDS

from ._shared import (
    MAX_LEADERBOARD_ROWS,
    MAX_SERIES_BUCKETS,
    _REAL_MESSAGE_CLAUSES,
    _append_id_filters,
    _as_exclusion_list,
    _connect,
    _date_clauses,
    _exclude_id_filters,
    _id_filter_clause,
    _normalize_date_bound,
    _resolve_chat_scope,
    _truncate_text_field,
    _validate_include,
)

_FW_CATEGORIES = tuple(FUNCTION_WORDS)
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in _FW_CATEGORIES)
_TOP_EMOJI = 15
_EPS = 1e-9  # LSM denominator guard: both-zero rates score a perfect 1.0

_TREND_SECTIONS = ("phrase_trend", "emoji", "rates", "distinctive_terms", "lsm",
                   "emotions", "valence")

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


def _lexicon_rows(con, chat_ids, date_from, date_to, exclude_chat_ids=None):
    """person_period_lexicon rows in scope (handle_id NULL = my side)."""
    clauses, params = [], []
    clause, cparams = _id_filter_clause("chat_id", chat_ids)
    if clause:
        clauses.append(clause)
        params.extend(cparams)
    _exclude_id_filters(clauses, params, exclude_chat_ids, prefix="")
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

def _phrase_trend(con, phrase, chat_ids, date_from, date_to, granularity,
                  exclude_chat_ids=None):
    n = _period_len(granularity)
    clauses, params = list(_TEXT_MESSAGE_CLAUSES), []
    _append_id_filters(clauses, params, chat_ids, None)
    _exclude_id_filters(clauses, params, exclude_chat_ids)
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


def _distinctive_terms(con, chat_ids, exclude_chat_ids=None):
    if not chat_ids:
        return {"available": False,
                "note": "distinctive terms are per chat -- pass chat_id or a person scope"}
    clauses, params = [], []
    clause, cparams = _id_filter_clause("chat_id", chat_ids)
    clauses.append(clause)
    params.extend(cparams)
    _exclude_id_filters(clauses, params, exclude_chat_ids, prefix="")
    # Curated keepers (inside_joke/nickname) outrank uncurated candidates;
    # LLM-rejected terms (ordinary/noise) are dropped entirely.
    rows = con.execute(
        f"""
        SELECT chat_id, term, lift, n_uses, first_message_id,
               coined_by_handle_id, coined_is_me, llm_verdict
        FROM dyad_terms
        WHERE {' AND '.join(clauses)} AND (llm_verdict IS NULL
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


def _annotation_rows(con, chat_ids, date_from, date_to, exclude_chat_ids=None):
    """chunk_annotations joined to chunks for scope/date (chunks.first_date
    is the same unlabeled-UTC string form as messages.date)."""
    clauses, params = [], []
    clause, cparams = _id_filter_clause("c.chat_id", chat_ids)
    if clause:
        clauses.append(clause)
        params.extend(cparams)
    _exclude_id_filters(clauses, params, exclude_chat_ids, prefix="c.")
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
                    date_from=None, date_to=None, granularity="year", include=None,
                    exclude_chat_id=None):
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
    no scope means the whole corpus. exclude_chat_id (scalar or list; not in
    the LLM schema -- wrapped.py's carve-out) drops those chats from every
    section."""
    _period_len(granularity)  # validate early
    wanted = _validate_include(include, _TREND_SECTIONS, "get_word_trends")
    exclude_chat_ids = _as_exclusion_list(exclude_chat_id)

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
                con, phrase, chat_ids, date_from, date_to, granularity,
                exclude_chat_ids)
        lex = None
        if want("emoji") or want("rates") or want("lsm"):
            lex = _lexicon_rows(con, chat_ids, date_from, date_to, exclude_chat_ids)
        if want("emoji"):
            result["emoji"] = _emoji_section(lex)
        if want("rates"):
            result["rates"] = _rates_section(lex, granularity)
        if want("distinctive_terms"):
            result["distinctive_terms"] = _distinctive_terms(con, chat_ids,
                                                             exclude_chat_ids)
        if want("lsm"):
            result["lsm"] = _lsm_section(lex, granularity)
        if want("emotions") or want("valence"):
            ann = _annotation_rows(con, chat_ids, date_from, date_to,
                                   exclude_chat_ids)
            if want("emotions"):
                result["emotions"] = _emotions_section(ann)
            if want("valence"):
                result["valence"] = _valence_section(ann, granularity)
    return result


# --- feature: nudges ---
def lsm_score(lexicon_rows):
    """Public, thin re-export of the language-style-matching primitive for the
    nudges pipeline (loose_ends.py dormant_tie detector) -- so callers outside
    the package never reach into the package-private _lsm_score / _weighted_rates.

    Given a dyad's person_period_lexicon rows (handle_id NULL = my side,
    non-NULL = the counterparty), return the scalar mean LSM between the two
    sides' word-count-weighted category rates, or None when either side carries
    no words in the given rows (nothing to match). Mirrors _lsm_section's
    me/them split; drops the per-category breakdown the internal callers need."""
    me_rows = [r for r in lexicon_rows if r["handle_id"] is None]
    them_rows = [r for r in lexicon_rows if r["handle_id"] is not None]
    rates_me, rates_them = _weighted_rates(me_rows), _weighted_rates(them_rows)
    if rates_me is None or rates_them is None:
        return None
    return _lsm_score(rates_me, rates_them)[0]
# --- end feature: nudges ---
