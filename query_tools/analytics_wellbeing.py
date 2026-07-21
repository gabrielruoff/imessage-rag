"""Wellbeing trajectory query tool (RES-61 / W2): get_language_trajectory.

Wire the deterministic, scope-agnostic change-point core (wellbeing.py, W1)
to the analysis DB. This module owns ALL of the SQL and the consent gate; the
math lives in wellbeing.py and never touches sqlite/config.

Two modes:
  - SELF (default, `contact=None`): the user's own trajectory -- every
    `handle_id IS NULL` lexicon row and every `is_from_me = 1` message/session,
    pooled across all chats.
  - PER-CONTACT (`contact=<contacts.id>`, Master Spec A12): accepted ONLY when
    that id is on the WELLBEING_CONSENT_IDS allowlist (consent.is_consented).
    A non-consented target is refused IN CODE with a fixed message and never
    analyzed. Otherwise the SAME machinery runs against that contact's own
    identity (their handle rows / their `is_from_me = 0` lines).

Marker construction is COUNT-BASED (never a rate of rates). For each marker we
sum raw counts across the scoped `(chat_id, <target handle>, month)` rows and
divide by the summed `n_words` once, at the aggregate level -- reconstructing
the negation category from its stored rate as `round(fw_negation_rate *
n_words)` and the emoji count by summing each row's `emoji_json` values. A
marker with fewer than WELLBEING_MIN_MARKER_HITS lifetime raw hits is dropped
BEFORE calling W1 (W1 accepts already-filtered series). Reply latency is NOT a
v1 marker (not in W1's set) -- the omission is documented in the envelope note.

Late-night detection uses the local timezone (config.LOCAL_TZ) for the hour, as
the time-analytics tools do; every month bucket is the UTC `date[:7]` so the
behavioral markers share one axis with the UTC-month lexicon.
"""

import collections
import contextlib
import datetime
import json
import statistics
import zoneinfo

import config
import wellbeing
from consent import is_consented
from function_words import ABSOLUTIST, FUNCTION_WORDS, PRONOUN_I, tokenize

from ._shared import (
    MAX_SERIES_BUCKETS,
    _REAL_MESSAGE_CLAUSES,
    _connect,
    _normalize_date_bound,
    _validate_include,
    member_key,
)

# Envelope sections a caller may select via `include` (default: all). The other
# envelope keys (scope, markers_available, dropped_markers, note, support_block)
# are always present.
_INCLUDE_SECTIONS = ("series", "windows", "examples", "baseline")

# Lexicon-derived markers (person_period_lexicon) and behavioral markers
# (messages / sessions). Together these are exactly wellbeing.MARKERS' keys.
_LEXICON_MARKERS = ("i_count", "absolutist_count", "fw_negation_rate",
                    "laugh_rate", "emoji_rate")
_BEHAVIORAL_MARKERS = ("late_night_share", "initiation_share")

# Grounding: which chunks/lines to score, and how tightly to trim an excerpt.
_MAX_GROUNDING_CHUNKS = 500   # chunks scored per firing window (bounds tokenizing)
_EXCERPT_LINE_CAP = 200       # chars per target line
_EXCERPT_MAX_LINES = 6        # target lines shown per example

# The fired-family composite tokens for grounding: the concerning-language
# vocabulary of each lexical family (behavioral markers are not per-line text
# properties, so grounding always keys off a lexical family).
_FAMILY_TOKENS = {
    "absolutist_negation": frozenset(ABSOLUTIST) | frozenset(FUNCTION_WORDS["negation"]),
    "self_focus": frozenset(PRONOUN_I),
}

# The fixed refusal shown for a non-consented per-contact target -- never an
# analysis, never a hint about who is on the list.
_REFUSAL_MESSAGE = (
    "Per-contact wellbeing analysis is not enabled for this contact. It runs "
    "only for contacts explicitly opted in via WELLBEING_CONSENT_IDS."
)

_REPLY_LATENCY_NOTE = (
    "reply-latency is intentionally not a v1 marker (not in the wellbeing "
    "marker set)"
)


# --------------------------------------------------------------------------
# local-tz late-night bucketing (mirrors analytics_time; never SQL date funcs)
# --------------------------------------------------------------------------

def _local_tz():
    name = config.LOCAL_TZ
    return zoneinfo.ZoneInfo(name) if name else None


def _is_late_night(hour):
    """Local hour inside [LATE_NIGHT_START_HOUR, LATE_NIGHT_END_HOUR); a
    start > end window wraps midnight."""
    start, end = config.LATE_NIGHT_START_HOUR, config.LATE_NIGHT_END_HOUR
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


# --------------------------------------------------------------------------
# scope resolution
# --------------------------------------------------------------------------

def _contact_handle_ids(con, contact_id):
    return [r[0] for r in con.execute(
        "SELECT id FROM handles WHERE contact_id = ?", (contact_id,)).fetchall()]


def _contact_chat_ids(con, handle_ids):
    placeholders = ",".join("?" * len(handle_ids))
    return [r[0] for r in con.execute(
        f"SELECT DISTINCT chat_id FROM chat_participants "
        f"WHERE handle_id IN ({placeholders})", handle_ids).fetchall()]


def _contact_display_name(con, contact_id):
    row = con.execute(
        "SELECT COALESCE("
        "NULLIF(nickname, ''), "
        "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), '')"
        ") FROM contacts WHERE id = ?", (contact_id,)).fetchone()
    if row and row[0]:
        return row[0]
    handle = con.execute(
        "SELECT handle_identifier FROM handles WHERE contact_id = ? "
        "ORDER BY id LIMIT 1", (contact_id,)).fetchone()
    return handle[0] if handle else f"contact {contact_id}"


# --------------------------------------------------------------------------
# date-range helpers
# --------------------------------------------------------------------------

def _month_bounds(date_from, date_to):
    """Inclusive 'YYYY-MM' bounds for the lexicon month filter (mirrors
    analytics_language._month_bounds)."""
    lo = hi = None
    if date_from is not None:
        lo = _normalize_date_bound(date_from)[:7]
    if date_to is not None:
        upper = datetime.datetime.fromisoformat(_normalize_date_bound(date_to, upper=True))
        hi = (upper - datetime.timedelta(microseconds=1)).strftime("%Y-%m")
    return lo, hi


def _date_bounds(date_from, date_to):
    lo = _normalize_date_bound(date_from) if date_from is not None else None
    hi = _normalize_date_bound(date_to, upper=True) if date_to is not None else None
    return lo, hi


def _next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m > 12:
        m, y = 1, y + 1
    return f"{y:04d}-{m:02d}"


# --------------------------------------------------------------------------
# count-based marker aggregation
# --------------------------------------------------------------------------

def _new_counts():
    # month -> [numerator, denominator]
    return collections.defaultdict(lambda: [0, 0])


def _lexicon_markers(con, target_handles, lo_month, hi_month):
    """Sum raw counts per month across the scoped lexicon rows and return
    {marker: {month: [num, den]}} plus the scope's total words.

    NEVER averages the stored fw_*_rate: the negation category is reconstructed
    as round(rate * n_words), so the month rate is formed once from summed
    counts. Emoji is the summed integer values of each row's emoji_json (a NULL
    / empty dict contributes 0 numerator, its n_words still counts)."""
    clauses = ["handle_id IS NULL"] if target_handles is None else [
        f"handle_id IN ({','.join('?' * len(target_handles))})"]
    params = [] if target_handles is None else list(target_handles)
    if lo_month is not None:
        clauses.append("month >= ?")
        params.append(lo_month)
    if hi_month is not None:
        clauses.append("month <= ?")
        params.append(hi_month)
    rows = con.execute(
        "SELECT month, n_words, fw_negation_rate, laugh_count, emoji_json, "
        "i_count, absolutist_count FROM person_period_lexicon "
        f"WHERE {' AND '.join(clauses)}", params).fetchall()

    markers = {name: _new_counts() for name in _LEXICON_MARKERS}
    total_words = 0
    for r in rows:
        words = r["n_words"] or 0
        total_words += words
        month = r["month"]
        markers["i_count"][month][0] += r["i_count"] or 0
        markers["absolutist_count"][month][0] += r["absolutist_count"] or 0
        # Reconstruct the category count from its stored rate -- count-then-
        # divide, never a rate of rates.
        markers["fw_negation_rate"][month][0] += round((r["fw_negation_rate"] or 0.0) * words)
        markers["laugh_rate"][month][0] += r["laugh_count"] or 0
        emoji_n = 0
        if r["emoji_json"]:
            try:
                emoji_n = sum(int(v) for v in json.loads(r["emoji_json"]).values())
            except (TypeError, ValueError):
                emoji_n = 0
        markers["emoji_rate"][month][0] += emoji_n
        for name in _LEXICON_MARKERS:
            markers[name][month][1] += words
    return markers, total_words


def _late_night_marker(con, target_handles, date_lo, date_hi):
    """{month: [late_night_msgs, total_msgs]} for the target's own messages,
    late-night detected on the LOCAL hour, bucketed by UTC month."""
    clauses = ["m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES]
    params = []
    if target_handles is None:
        clauses.append("m.is_from_me = 1")
    else:
        clauses.append("m.is_from_me = 0")
        clauses.append(f"m.handle_id IN ({','.join('?' * len(target_handles))})")
        params.extend(target_handles)
    if date_lo is not None:
        clauses.append("m.date >= ?")
        params.append(date_lo)
    if date_hi is not None:
        clauses.append("m.date < ?")
        params.append(date_hi)
    tz = _local_tz()
    counts = _new_counts()
    for (date_str,) in con.execute(
        f"SELECT m.date FROM messages m WHERE {' AND '.join(clauses)}", params):
        utc = datetime.datetime.fromisoformat(date_str).replace(
            tzinfo=datetime.timezone.utc)
        month = date_str[:7]
        counts[month][1] += 1
        if _is_late_night(utc.astimezone(tz).hour):
            counts[month][0] += 1
    return counts


def _initiation_marker(con, target_handles, contact_chat_ids, date_lo, date_hi):
    """{month: [target_initiated_sessions, total_sessions]} over the sessions in
    scope (all sessions in SELF mode; the contact's chats in per-contact mode).
    Session initiator identity is the member_key merge in SQL: initiator_is_me
    for self, else the contact's own handle rows."""
    clauses, params = [], []
    if contact_chat_ids is not None:
        clauses.append(f"chat_id IN ({','.join('?' * len(contact_chat_ids))})")
        params.extend(contact_chat_ids)
    if date_lo is not None:
        clauses.append("first_date >= ?")
        params.append(date_lo)
    if date_hi is not None:
        clauses.append("first_date < ?")
        params.append(date_hi)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    handle_set = set(target_handles) if target_handles is not None else None
    counts = _new_counts()
    for r in con.execute(
        f"SELECT first_date, initiator_is_me, initiator_handle_id "
        f"FROM sessions {where}", params):
        month = r["first_date"][:7]
        counts[month][1] += 1
        if handle_set is None:
            initiated = bool(r["initiator_is_me"])
        else:
            initiated = (not r["initiator_is_me"]
                         and r["initiator_handle_id"] in handle_set)
        if initiated:
            counts[month][0] += 1
    return counts


def _filter_by_hit_floor(marker_counts, floor):
    """Split {marker: {month: [num, den]}} into the W1 input series (markers
    clearing the lifetime raw-hit floor) and a {marker: reason} drop map."""
    series, dropped = {}, {}
    for name, month_map in marker_counts.items():
        hits = sum(num for num, _ in month_map.values())
        points = [(m, num / den) for m, (num, den) in sorted(month_map.items())
                  if den > 0]
        if hits < floor or not points:
            dropped[name] = f"below_hit_floor ({hits} < {floor})"
            continue
        series[name] = points
    return series, dropped


# --------------------------------------------------------------------------
# time-series envelope shaping (compact by construction)
# --------------------------------------------------------------------------

def _series_points(month_map):
    """Compact [[period, rate], ...] for one marker, coarsening month->year past
    MAX_SERIES_BUCKETS (summed counts, so the coarser rate stays count-based)."""
    months = sorted(m for m, (_, den) in month_map.items() if den > 0)
    if not months:
        return [], False
    pairs = [[m, round(month_map[m][0] / month_map[m][1], 5)] for m in months]
    coarsened = False
    if len(pairs) > MAX_SERIES_BUCKETS:
        by_year = collections.defaultdict(lambda: [0, 0])
        for m in months:
            num, den = month_map[m]
            by_year[m[:4]][0] += num
            by_year[m[:4]][1] += den
        pairs = [[y, round(v[0] / v[1], 5)] for y, v in sorted(by_year.items())
                 if v[1] > 0]
        coarsened = True
    if len(pairs) > MAX_SERIES_BUCKETS:
        pairs = pairs[-MAX_SERIES_BUCKETS:]
    return pairs, coarsened


# --------------------------------------------------------------------------
# grounding: median-scoring, target-lines-only excerpts for a firing window
# --------------------------------------------------------------------------

def _fired_family(families):
    """The lexical family to ground on: prefer absolutist_negation, else
    self_focus, whichever departed (a flagged window always has a departed
    lexical family, by W1's concordance rule)."""
    for name in ("absolutist_negation", "self_focus"):
        if families.get(name, {}).get("departed"):
            return name
    return "absolutist_negation"


def _grounding(con, target_handles, contact_chat_ids, window, families, n_examples):
    """Return up to `n_examples` chunks overlapping the firing window whose
    fired-family composite (over ONLY the target's own lines) is NEAREST the
    window median -- never the max-negative chunk. Excerpts carry the target's
    own lines only (other-speaker lines are already excluded by the scan)."""
    family = _fired_family(families)
    tokens_of = _FAMILY_TOKENS[family]
    start_month, end_month = window
    end_bound = _next_month(end_month)

    chunk_clauses = ["last_date >= ?", "first_date < ?"]
    chunk_params = [start_month, end_bound]
    if contact_chat_ids is not None:
        if not contact_chat_ids:
            return [], family
        chunk_clauses.append(f"chat_id IN ({','.join('?' * len(contact_chat_ids))})")
        chunk_params.extend(contact_chat_ids)
    chunk_ids = [r[0] for r in con.execute(
        f"SELECT id FROM chunks WHERE {' AND '.join(chunk_clauses)} "
        f"ORDER BY first_date DESC LIMIT {_MAX_GROUNDING_CHUNKS}", chunk_params)]
    if not chunk_ids:
        return [], family

    msg_clauses = [f"m.chunk_id IN ({','.join('?' * len(chunk_ids))})",
                   "m.text IS NOT NULL", "m.text != ''", *_REAL_MESSAGE_CLAUSES]
    msg_params = list(chunk_ids)
    if target_handles is None:
        msg_clauses.append("m.is_from_me = 1")
    else:
        msg_clauses.append("m.is_from_me = 0")
        msg_clauses.append(f"m.handle_id IN ({','.join('?' * len(target_handles))})")
        msg_params.extend(target_handles)

    per_chunk = collections.defaultdict(lambda: {"lines": [], "hits": 0, "words": 0})
    for r in con.execute(
        f"SELECT m.chunk_id, m.text FROM messages m "
        f"WHERE {' AND '.join(msg_clauses)} ORDER BY m.id", msg_params):
        agg = per_chunk[r["chunk_id"]]
        agg["lines"].append(r["text"])
        toks = tokenize(r["text"])
        agg["words"] += len(toks)
        agg["hits"] += sum(1 for t in toks if t in tokens_of)

    scored = [(cid, agg, agg["hits"] / agg["words"])
              for cid, agg in per_chunk.items() if agg["words"] > 0]
    if not scored:
        return [], family
    median = statistics.median(c for _, _, c in scored)
    # Nearest the window median first (max-negative chunk sorts last); chunk_id
    # tiebreak keeps it deterministic.
    scored.sort(key=lambda s: (abs(s[2] - median), s[0]))
    examples = []
    for cid, agg, composite in scored[:n_examples]:
        lines = [ln[:_EXCERPT_LINE_CAP] for ln in agg["lines"][:_EXCERPT_MAX_LINES]]
        examples.append({
            "chunk_id": cid,
            "composite": round(composite, 5),
            "n_lines": len(agg["lines"]),
            "excerpt_lines": lines,
        })
    return examples, family


# --------------------------------------------------------------------------
# envelope builders
# --------------------------------------------------------------------------

def _stub_envelope(scope, note, *, refused=False, baseline=None):
    env = {
        "scope": scope,
        "markers_available": [],
        "dropped_markers": {},
        "baseline": baseline,
        "series": None,
        "windows": None,
        "examples": None,
        "note": note,
        "support_block": None,
    }
    if refused:
        env["refused"] = True
    env["note"] = _sanitize_text(env["note"])  # W3: source-level scrub of tool prose
    return env


# --------------------------------------------------------------------------
# W3: source-level (non-clinical) sanitization of tool-authored free text
# --------------------------------------------------------------------------

def _sanitize_text(text):
    """Strip any clinical-condition name from tool-AUTHORED prose before it can
    seed the model's narration (defense in depth with chat.py's final-sentence
    scrub). Applied to the envelope's framing/note fields. The grounding
    excerpts are the target's OWN messages -- evidence, not narration -- so they
    are returned verbatim and never redacted here."""
    return wellbeing.sanitize_condition_names(text)[0]


def get_language_trajectory(contact=None, date_from=None, date_to=None,
                            include=None, min_words=None):
    """Detect a sustained, cross-family shift in someone's texting language --
    self by default, or a consent-gated contact.

    SELF (contact=None) analyzes the user's own history (all your `handle_id
    IS NULL` lexicon rows and your sent messages/sessions, pooled across chats).
    Naming a `contact` (a contacts.id) analyzes THAT person's language instead,
    but ONLY when they are on the WELLBEING_CONSENT_IDS allowlist -- a
    non-consented contact is refused, never analyzed.

    Markers (all count-based, NEVER a rate of rates): first-person-singular
    `i`, absolutist words, negation, laugh rate, emoji rate (from the lexicon),
    plus late-night share and outbound session-initiation share. A marker with
    too few lifetime hits is dropped before analysis. Reply latency is not a v1
    marker. `include` selects envelope sections: series, windows, examples,
    baseline (default all). `min_words` is the corpus floor below which no
    analysis runs.

    Returns an envelope: scope, markers_available, dropped_markers, baseline,
    series, windows, examples, note, and (only when a departure fires)
    support_block. Below the word floor it returns an insufficient-data
    envelope with no analysis."""
    wanted = _validate_include(include, _INCLUDE_SECTIONS, "get_language_trajectory")
    if min_words is None:
        min_words = config.MIN_WELLBEING_WORDS

    with contextlib.closing(_connect()) as con:
        # --- resolve target + consent gate ---
        target_handles = None      # None = self (handle_id IS NULL rows)
        contact_chat_ids = None    # None = all chats (self); else the contact's chats
        if contact is None:
            scope, framing_noun = "self", "your language"
        else:
            contact_id = int(contact)
            scope = member_key(False, contact_id, None)  # "contact:<id>"
            if not is_consented(contact_id, config.WELLBEING_CONSENT_IDS):
                return _stub_envelope(scope, _REFUSAL_MESSAGE, refused=True)
            target_handles = _contact_handle_ids(con, contact_id)
            if not target_handles:
                return _stub_envelope(
                    scope, f"no handles on file for contact {contact_id}")
            contact_chat_ids = _contact_chat_ids(con, target_handles)
            framing_noun = f"{_contact_display_name(con, contact_id)}'s language"

        lo_month, hi_month = _month_bounds(date_from, date_to)
        date_lo, date_hi = _date_bounds(date_from, date_to)

        # --- corpus floor (cheap; short-circuit before the heavy passes) ---
        marker_counts, total_words = _lexicon_markers(
            con, target_handles, lo_month, hi_month)
        if total_words < min_words:
            note = (f"insufficient data: {total_words} words in scope is below "
                    f"the {min_words}-word floor; no analysis was run. "
                    f"{_REPLY_LATENCY_NOTE}.")
            return _stub_envelope(
                scope, note,
                baseline={"total_words": total_words, "min_words": min_words})

        # --- behavioral markers ---
        marker_counts["late_night_share"] = _late_night_marker(
            con, target_handles, date_lo, date_hi)
        marker_counts["initiation_share"] = _initiation_marker(
            con, target_handles, contact_chat_ids, date_lo, date_hi)

        # --- hit floor, then the deterministic core ---
        marker_series, dropped = _filter_by_hit_floor(
            marker_counts, config.WELLBEING_MIN_MARKER_HITS)
        result = wellbeing.analyze_trajectory(
            marker_series,
            z_threshold=config.WELLBEING_Z_THRESHOLD,
            min_run=config.WELLBEING_MIN_RUN_MONTHS,
            min_baseline_months=config.WELLBEING_MIN_BASELINE_MONTHS,
            family_cap=config.WELLBEING_FAMILY_CAP,
            cusum_k=config.WELLBEING_CUSUM_K,
            cusum_h=config.WELLBEING_CUSUM_H,
        )
        for name in result["dropped_markers"]:
            dropped.setdefault(name, "degenerate (empty or MAD==0)")

        flagged = result["flagged"]
        markers_available = [m for m in wellbeing.MARKERS if m in marker_series]
        history_months = sorted({m for pts in marker_series.values() for m, _ in pts})

        # --- envelope sections ---
        notes = [_REPLY_LATENCY_NOTE]
        env = {
            "scope": scope,
            "markers_available": markers_available,
            "dropped_markers": dropped,
            "support_block": wellbeing.support_block() if flagged else None,
        }
        # --- feature: wellbeing (W3) ---
        # Code-emitted, direction-independent support text. chat.run_turn prints
        # a tool's reserved `_epilogue` verbatim AFTER the reply, so support is
        # ALWAYS present on a flagged result regardless of which way the window
        # moved -- delivery is code's job here, not the model's narration.
        if flagged:
            env["_epilogue"] = wellbeing.support_block()
        # --- end feature: wellbeing (W3) ---

        if "baseline" in wanted:
            env["baseline"] = {
                "total_words": total_words,
                "n_history_months": len(history_months),
                "baseline_months": result["baseline_months"],
            }
        else:
            env["baseline"] = None

        if "series" in wanted:
            series, any_coarsened = {}, False
            for name in markers_available:
                pts, coarsened = _series_points(marker_counts[name])
                series[name] = pts
                any_coarsened = any_coarsened or coarsened
            env["series"] = series
            if any_coarsened:
                notes.append("series coarsened month->year past the bucket cap")
        else:
            env["series"] = None

        if "windows" in wanted:
            if flagged:
                start_m, end_m = result["window"]
                departed = [f for f, blk in result["families"].items()
                            if blk["departed"]]
                env["windows"] = [{
                    "start": start_m,
                    "end": end_m,
                    "peak_z": round(result["peak_z"], 3),
                    "baseline_months": result["baseline_months"],
                    "departed_families": departed,
                    "framing": _sanitize_text(
                        f"A sustained shift in {framing_noun} from "
                        f"{start_m} to {end_m}."),
                }]
            else:
                env["windows"] = []
        else:
            env["windows"] = None

        if "examples" in wanted:
            if flagged:
                examples, family = _grounding(
                    con, target_handles, contact_chat_ids, result["window"],
                    result["families"], config.WELLBEING_N_EXAMPLES)
                env["examples"] = examples
                if examples:
                    notes.append(f"examples score the '{family}' family over the "
                                 "target's own lines, nearest the window median")
            else:
                env["examples"] = []
        else:
            env["examples"] = None

        if not flagged:
            notes.append("no sustained cross-family departure detected")
        env["note"] = _sanitize_text("; ".join(notes))  # W3: source-level scrub
        return env
