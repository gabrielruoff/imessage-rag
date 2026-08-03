"""Loose Ends / nudge detectors (the six "# --- detector: X ---" families).

Split out of loose_ends.py: the deterministic (plus one eval-gated LLM)
detectors that yield CandidateRow objects. Each registers itself into the shared
DETECTORS registry on import. Reads the shared dyad primitives, CandidateRow, and
the DETECTORS registry from nudge_shared (a neutral module both this file and
loose_ends import, so there is no import cycle); loose_ends imports this module
(running the registrations) and re-exports the detectors. Thresholds arrive via
the cfg dict (built by loose_ends._build_cfg), so no config vars are read here.
"""

import datetime
import json
import math
import re
import statistics
from collections import defaultdict
from typing import Iterable

import ollama
import stats
from eval.evalkit.gates import eval_gate_passed
from query_tools._shared import TEXT_PREVIEW_CHARS, _restrict_to_direct_chats
from query_tools.analytics_language import lsm_score

from nudge_shared import (
    DETECTORS,
    CandidateRow,
    _REAL,
    _counterparty_handle,
    _dyad_gap_p90,
    _last_message,
)


def detect_faded_initiator(con, cfg) -> Iterable[CandidateRow]:
    """Fire when the counterparty's share of session initiations over the
    trailing NUDGE_FADED_INIT_WINDOW_DAYS has fallen to <= NUDGE_FADED_INIT_DROP
    of their all-time share (both from sessions.initiator_is_me), for dyads with
    at least NUDGE_FADED_INIT_MIN_SESSIONS lifetime sessions. Pure SQL over the
    derived `sessions` table; group chats are excluded via F3
    _restrict_to_direct_chats. Skips a dyad the other person never initiated
    (nothing to fade) or with no recent sessions at all (that is a dormant tie,
    a different detector's job)."""
    window_start = (
        cfg["now"] - datetime.timedelta(days=cfg["faded_window_days"])
    ).isoformat()
    rows = con.execute(
        """
        SELECT s.chat_id,
               COUNT(*) AS life_total,
               SUM(CASE WHEN s.initiator_is_me = 0 THEN 1 ELSE 0 END) AS life_theirs,
               SUM(CASE WHEN s.first_date >= ? THEN 1 ELSE 0 END) AS trail_total,
               SUM(CASE WHEN s.first_date >= ? AND s.initiator_is_me = 0
                        THEN 1 ELSE 0 END) AS trail_theirs
        FROM sessions s
        GROUP BY s.chat_id
        HAVING life_total >= ?
        """,
        (window_start, window_start, cfg["faded_min_sessions"]),
    ).fetchall()
    if not rows:
        return
    direct = set(_restrict_to_direct_chats(con, [r["chat_id"] for r in rows]))
    for r in rows:
        chat_id = r["chat_id"]
        if chat_id not in direct:
            continue
        life_total, life_theirs = r["life_total"], r["life_theirs"]
        trail_total, trail_theirs = r["trail_total"], r["trail_theirs"]
        if not life_theirs or not trail_total:
            continue
        all_time_share = life_theirs / life_total
        trailing_share = trail_theirs / trail_total
        if trailing_share > cfg["faded_drop"] * all_time_share:
            continue
        anchor_id, anchor_date = _last_message(con, chat_id)
        yield CandidateRow(
            kind="faded_initiator",
            chat_id=chat_id,
            handle_id=_counterparty_handle(con, chat_id),
            anchor_message_id=anchor_id,
            anchor_date=anchor_date,
            evidence={
                "all_time_initiation_share": round(all_time_share, 4),
                "trailing_initiation_share": round(trailing_share, 4),
                "lifetime_sessions": life_total,
                "trailing_sessions": trail_total,
                "window_start": window_start,
            },
        )


DETECTORS.append(("faded_initiator", detect_faded_initiator))


# ---------------------------------------------------------------------------
# Detector: dormant_tie (RES-55 / N2)
# ---------------------------------------------------------------------------
# --- detector: dormant_tie ---
# A once-close 1:1 tie now overdue for contact. Both halves are defined against
# the trailing 12 months; the pinned knobs live in config.NUDGE_DORMANT_*. Like
# every detector this is PURE detection -- the shared driver runs the Stage-0
# exclusion gate first, so an excluded / negative-valence / zero-outbound
# contact is never surfaced here (that spine is N1's, relied on, not re-done).

_DORMANT_WINDOW_DAYS = 365


def _dormant_window_start(now):
    """ISO lower bound of the trailing-12-month closeness/eligibility window."""
    return (now - datetime.timedelta(days=_DORMANT_WINDOW_DAYS)).isoformat()


def _dormant_mean_depth(con, chat_id, window_start):
    """Mean chunk_annotations.depth (ANY source -- depth distills fine) over the
    chat's chunks whose last_date is in the trailing window; None when no
    annotated chunk is in range (that dyad contributes 0 to the depth z-term)."""
    row = con.execute(
        "SELECT AVG(a.depth) FROM chunks c "
        "JOIN chunk_annotations a ON a.chunk_id = c.id "
        "WHERE c.chat_id = ? AND c.last_date >= ? AND a.depth IS NOT NULL",
        (chat_id, window_start),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _dormant_lsm(con, chat_id, window_start):
    """The dyad's language-style-matching score over its person_period_lexicon
    rows in the trailing window, via query_tools' public lsm_score. None when
    either side has no words in range (one-sided lexicon -> nothing to match)."""
    rows = con.execute(
        "SELECT * FROM person_period_lexicon WHERE chat_id = ? AND month >= ?",
        (chat_id, window_start[:7]),
    ).fetchall()
    return lsm_score(rows) if rows else None


def _dormant_eligible(con, cfg, window_start):
    """Every eligible direct dyad with its three raw closeness components over
    the trailing window: dicts {chat_id, messages, depth, lsm}. Eligible = a 1:1
    chat with >= dormant_min_messages real messages in the window."""
    counts = con.execute(
        f"SELECT m.chat_id AS chat_id, COUNT(*) AS n FROM messages m "
        f"WHERE m.date IS NOT NULL AND m.date >= ? AND {_REAL} "
        f"GROUP BY m.chat_id HAVING n >= ?",
        (window_start, cfg["dormant_min_messages"]),
    ).fetchall()
    direct = set(_restrict_to_direct_chats(con, [r["chat_id"] for r in counts]))
    out = []
    for r in counts:
        cid = r["chat_id"]
        if cid not in direct:
            continue
        out.append({
            "chat_id": cid,
            "messages": r["n"],
            "depth": _dormant_mean_depth(con, cid, window_start),
            "lsm": _dormant_lsm(con, cid, window_start),
        })
    return out


def _zscores(values):
    """z-score a list whose entries may be None. Mean/stdev are taken over the
    PRESENT values only; a None (missing component) maps to 0.0, and a
    degenerate spread (<2 present, or zero stdev) maps every entry to 0.0 -- so a
    missing or non-discriminating component neither helps nor hurts the score."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0] * len(values)
    mean = statistics.fmean(present)
    sd = statistics.pstdev(present)
    if sd == 0:
        return [0.0] * len(values)
    return [((v - mean) / sd if v is not None else 0.0) for v in values]


def detect_dormant_tie(con, cfg) -> Iterable[CandidateRow]:
    """Fire for a once-close 1:1 tie that has now gone quiet for anomalously
    long. Once-close = the top cfg['dormant_top_frac'] of eligible dyads by a
    z-scored closeness composite over the trailing 12 months:
        closeness = z(messages) + z(mean chunk depth) + z(LSM),
    each term z-scored across all eligible dyads (a dyad missing depth/LSM
    contributes 0 for that term; depth from chunk_annotations of ANY source, LSM
    via query_tools' public lsm_score). Overdue = the current silence (corpus
    now - the dyad's last message) exceeds BOTH the dyad's own p90 inter-message
    gap (_dyad_gap_p90 over the FULL gap distribution -- not reply latency, which
    drops long silences and would fire on every pair) AND the absolute
    cfg['dormant_min_gap_days'] floor. Pure detection; the driver's Stage-0 gate
    suppresses any excluded counterparty before a row is written."""
    now = cfg["now"]
    window_start = _dormant_window_start(now)
    eligible = _dormant_eligible(con, cfg, window_start)
    if not eligible:
        return
    z_msgs = _zscores([e["messages"] for e in eligible])
    z_depth = _zscores([e["depth"] for e in eligible])
    z_lsm = _zscores([e["lsm"] for e in eligible])
    for e, zm, zd, zl in zip(eligible, z_msgs, z_depth, z_lsm):
        e["closeness"] = zm + zd + zl
        e["z"] = {"messages": round(zm, 4), "depth": round(zd, 4), "lsm": round(zl, 4)}
    ranked = sorted(eligible, key=lambda e: e["closeness"], reverse=True)
    n_top = math.ceil(cfg["dormant_top_frac"] * len(ranked))
    min_gap_seconds = cfg["dormant_min_gap_days"] * 86400
    for e in ranked[:n_top]:
        cid = e["chat_id"]
        anchor_id, anchor_date = _last_message(con, cid)
        if not anchor_date:
            continue
        current_gap = (
            now - datetime.datetime.fromisoformat(anchor_date)
        ).total_seconds()
        p90 = _dyad_gap_p90(con, cid)
        if p90 is None or current_gap <= p90 or current_gap <= min_gap_seconds:
            continue
        yield CandidateRow(
            kind="dormant_tie",
            chat_id=cid,
            handle_id=_counterparty_handle(con, cid),
            anchor_message_id=anchor_id,
            anchor_date=anchor_date,
            evidence={
                "closeness": round(e["closeness"], 4),
                "closeness_z": e["z"],
                "trailing_messages": e["messages"],
                "mean_depth": round(e["depth"], 3) if e["depth"] is not None else None,
                "lsm": round(e["lsm"], 4) if e["lsm"] is not None else None,
                "current_gap_days": round(current_gap / 86400, 1),
                "p90_gap_days": round(p90 / 86400, 1),
                "top_frac": cfg["dormant_top_frac"],
                "window_start": window_start,
            },
        )


DETECTORS.append(("dormant_tie", detect_dormant_tie))
# --- end detector: dormant_tie ---


# ---------------------------------------------------------------------------
# Detector: unanswered_question (RES-56 / N3)
# ---------------------------------------------------------------------------
# --- detector: unanswered_question ---
# A 1:1 session that ended on the OTHER person's genuine, unanswered question.
# Two stages, both PURE detection (the shared driver runs the Stage-0 exclusion
# gate and urgency scoring OUTSIDE this function, exactly as for every
# detector, so an excluded / negative-valence / zero-outbound contact never
# reaches here):
#   Stage 1  -- candidate = a session whose LAST message is a TRAILING,
#               length-gated, non-rhetorical question the counterparty sent.
#   Stage 1b -- MANDATORY cross-session answered-check: the question is a loose
#               end only if it went unanswered for longer than the dyad's own
#               cadence. Suppress iff any is_from_me real message lands in
#               (anchor, deadline], where deadline = anchor + max(the dyad's p90
#               inter-message gap via _dyad_gap_p90, NUDGE_QUESTION_GRACE_HOURS).
#               A self-reply AFTER the deadline does NOT suppress -- the question
#               sat past the pair's own rhythm, which is the signal -- and that
#               late reply is recorded in evidence for narration.
#
# The question test here is deliberately UNLIKE lexicon_build's bare `"?" in
# text` (which powers person_period_lexicon.question_count and MUST NOT change):
# it requires a trailing '?', a length floor, and a rhetorical/pleasantry
# stoplist, so "lol?" / "u up?" / "how are you?" never become a nudge.

# PINNED module-level stoplist of rhetorical / pleasantry questions, matched
# against the lowercased final sentence of the message (both with and without a
# trailing '?', so an entry may be spelled either way). Extending it is expected
# -- it lives here, next to the detector, on purpose.
_RHETORICAL_STOPLIST = frozenset({
    "how are you", "how r u", "hru", "how are you doing", "how's it going",
    "hows it going", "how you doing", "you know?", "right?", "ok?", "okay?",
    "k?", "lol?", "yeah?", "really?", "seriously?", "wanna bet?", "u up?",
    "you up?", "make sense?", "sound good?", "you know what i mean?",
    "isn't it?", "or what?", "guess what?", "wyd?", "sup?", "wassup?",
})

# Single-token questions whose entire content is one filler / pleasantry word
# are rhetorical regardless of the phrase list above (e.g. "really?", "huh?").
_QUESTION_STOPWORDS = frozenset({
    "really", "seriously", "srsly", "huh", "what", "ok", "okay", "k", "sure",
    "yeah", "yep", "yup", "lol", "lmao", "right", "oh", "hm", "hmm", "wait",
    "no", "yes", "nah", "meh", "eh", "sup",
})


def _final_sentence(text):
    """The message's final sentence, lowercased and stripped, keeping its
    trailing terminator (so 'right?' matches the stoplist form). Splits on a
    sentence terminator followed by whitespace, so 'Hey. How are you?' -> 'how
    are you?'.

    Shares a name with eval.evalkit.checks._final_sentence and is NOT the same
    function: that one strips the terminator, does not lowercase, and splits on
    newlines too, because it reads an LLM reply's closing line. This one's
    output is matched against _RHETORICAL_STOPLIST, whose entries carry their
    '?'. Kept separate deliberately -- one implementation would silently change
    one of the two matchers."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[-1].strip().lower() if parts else ""


def _is_rhetorical_question(text):
    """True when a trailing-question message is a rhetorical / pleasantry
    verdict rather than a genuine ask -- matched on the lowercased final
    sentence against _RHETORICAL_STOPLIST (with and without the '?') plus the
    single-stopword-token rule. Length gating is the detector's job, not this."""
    final = _final_sentence(text)
    if not final:
        return True
    bare = final.rstrip("?!. ").strip()
    if final in _RHETORICAL_STOPLIST or bare in _RHETORICAL_STOPLIST:
        return True
    tokens = bare.split()
    return len(tokens) == 1 and tokens[0] in _QUESTION_STOPWORDS


def _self_reply_in_window(con, chat_id, after, until):
    """True iff any is_from_me real message lands in the window (after, until]
    -- the Stage-1b suppression predicate. Plain UTC-ISO string compares."""
    row = con.execute(
        f"SELECT 1 FROM messages m "
        f"WHERE m.chat_id = ? AND m.is_from_me = 1 AND m.date IS NOT NULL "
        f"AND m.date > ? AND m.date <= ? AND {_REAL} LIMIT 1",
        (chat_id, after, until),
    ).fetchone()
    return row is not None


def _first_self_reply_after(con, chat_id, after):
    """(id, date) of the earliest is_from_me real message strictly after the ISO
    bound `after`, or (None, None) -- the late reply recorded past the deadline
    when the question went unanswered within the dyad's cadence."""
    row = con.execute(
        f"SELECT m.id, m.date FROM messages m "
        f"WHERE m.chat_id = ? AND m.is_from_me = 1 AND m.date IS NOT NULL "
        f"AND m.date > ? AND {_REAL} ORDER BY m.date, m.id LIMIT 1",
        (chat_id, after),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def detect_unanswered_question(con, cfg) -> Iterable[CandidateRow]:
    """Fire for a 1:1 session that ended on the counterparty's genuine,
    unanswered question (see the fence header for the two stages). One candidate
    per chat: the MOST RECENT session-ending question owns the verdict -- if it
    is short/rhetorical the chat yields nothing, and an older question never
    resurrects a chat whose latest ask was a pleasantry or was answered in time.
    Ranked by how out-of-character the silence is (silence / p90), recorded in
    evidence for the driver's urgency pass."""
    now = cfg["now"]
    min_chars = cfg["question_min_chars"]
    grace_seconds = cfg["question_grace_hours"] * 3600
    rows = con.execute(
        f"SELECT s.chat_id AS chat_id, s.last_message_id AS mid, "
        f"m.text AS text, m.date AS date "
        f"FROM sessions s JOIN messages m ON m.id = s.last_message_id "
        f"WHERE m.is_from_me = 0 AND m.date IS NOT NULL AND m.text IS NOT NULL "
        f"AND TRIM(m.text) LIKE '%?' AND {_REAL} "
        f"ORDER BY m.date DESC, m.id DESC"
    ).fetchall()
    if not rows:
        return
    direct = set(_restrict_to_direct_chats(con, list({r["chat_id"] for r in rows})))
    seen = set()
    for r in rows:
        chat_id = r["chat_id"]
        if chat_id in seen or chat_id not in direct:
            continue
        # The most-recent session-ending question owns this chat's verdict,
        # whether or not it survives the filters below.
        seen.add(chat_id)
        text = (r["text"] or "").strip()
        if len(text) < min_chars or _is_rhetorical_question(text):
            continue
        anchor_date = r["date"]
        anchor_dt = datetime.datetime.fromisoformat(anchor_date)
        p90 = _dyad_gap_p90(con, chat_id)
        grace = max(p90 or 0.0, grace_seconds)
        deadline = (anchor_dt + datetime.timedelta(seconds=grace)).isoformat()
        if _self_reply_in_window(con, chat_id, anchor_date, deadline):
            continue  # answered within the dyad's own cadence -- not a loose end
        late_id, late_date = _first_self_reply_after(con, chat_id, deadline)
        if late_date is not None and not cfg["question_include_answered_late"]:
            continue  # answered, just late -- resolved, not an open loose end
        silence_days = max(0.0, (now - anchor_dt).total_seconds()) / 86400
        max_age = cfg.get("question_max_age_days")
        if max_age is not None and silence_days > max_age:
            continue  # a years-old question is not an actionable loose end
        p90_days = (p90 / 86400) if p90 else None
        evidence = {
            "question": text[:TEXT_PREVIEW_CHARS],
            "silence_days": round(silence_days, 1),
            "p90_gap_days": round(p90_days, 1) if p90_days is not None else None,
            "silence_ratio": round(silence_days / p90_days, 2) if p90_days else None,
            "grace_hours": round(grace / 3600, 1),
            "deadline": deadline,
            "answered_late": late_date is not None,
        }
        if late_date is not None:
            late_dt = datetime.datetime.fromisoformat(late_date)
            evidence["late_reply_date"] = late_date
            evidence["late_reply_days_later"] = round(
                (late_dt - anchor_dt).total_seconds() / 86400, 1)
        yield CandidateRow(
            kind="unanswered_question",
            chat_id=chat_id,
            handle_id=_counterparty_handle(con, chat_id),
            anchor_message_id=r["mid"],
            anchor_date=anchor_date,
            evidence=evidence,
        )


DETECTORS.append(("unanswered_question", detect_unanswered_question))


# --- detector: inferred_occasion ---
# A recurring annual occasion a 1:1 dyad marks -- a birthday or anniversary --
# inferred from CONGRATULATORY messages that land on the SAME CALENDAR DATE
# across multiple years. Alone among the detectors it also fills `due_by`: the
# next upcoming occurrence of that month-day, so the digest can surface the
# reminder BEFORE the date instead of nagging after it. Pure detection -- the
# shared driver runs the Stage-0 gate first, so a muted / negative-valence /
# zero-outbound counterparty never reaches here.
#
# LOAD-BEARING date rule: no SQL date function ever wraps the date column in a
# WHERE clause. Candidate congratulatory messages are fetched with a plain text
# LIKE filter (the date column is only ever compared, never wrapped); the
# (month, day) and year are then sliced off the ISO string in PYTHON and
# bucketed here -- the same discipline analytics.py uses.

# Substrings (matched case-insensitively against message text) that mark a
# congratulatory / milestone message. Deliberately permissive: a one-off
# "congrats on the job" never fires on its own -- only the SAME (month, day)
# recurring across NUDGE_OCCASION_MIN_YEARS distinct years becomes an occasion,
# which a non-anniversary congratulation cannot satisfy.
_OCCASION_PHRASES = (
    "happy birthday", "happy bday", "hbd", "happy anniversary",
    "happy anniv", "congrats", "congratulations",
)


def _next_occurrence(now, month_day):
    """The next calendar date on or after `now` that falls on month_day
    ('MM-DD'), as an ISO 'YYYY-MM-DD' string, or None. Parses the 'MM-DD' form
    here and defers the roll-forward (and the Feb 29 -> Feb 28 rule) to
    stats.next_occurrence, which is the single home for both. 2000 is the
    anchor year purely because it is a leap year, so a Feb 29 occasion parses."""
    try:
        anchor = datetime.date(2000, int(month_day[:2]), int(month_day[3:5]))
    except ValueError:
        return None
    candidate = stats.next_occurrence(anchor, now.date())
    return candidate.isoformat() if candidate else None


def detect_inferred_occasion(con, cfg) -> Iterable[CandidateRow]:
    """Fire once per 1:1 dyad that congratulates on a recurring calendar date.
    Candidate congratulatory messages are pulled with a text LIKE filter (never
    a date function in WHERE); their (month, day)/year are sliced in Python, and
    a (month, day) seen in >= cfg['occasion_min_years'] distinct years becomes
    an occasion. The most recent occurrence is the anchor; due_by is the next
    upcoming date, so the nudge lands ahead of it. A dyad emits at most one
    candidate (the soonest-due occasion); any others ride in
    evidence['other_occasions']."""
    min_years = cfg["occasion_min_years"]
    like = " OR ".join("LOWER(m.text) LIKE ?" for _ in _OCCASION_PHRASES)
    params = [f"%{p}%" for p in _OCCASION_PHRASES]
    rows = con.execute(
        f"SELECT m.chat_id AS chat_id, m.id AS id, m.date AS date, m.text AS text "
        f"FROM messages m "
        f"WHERE m.text IS NOT NULL AND m.date IS NOT NULL AND ({like}) AND {_REAL} "
        f"ORDER BY m.date, m.id",
        params,
    ).fetchall()
    if not rows:
        return
    direct = set(_restrict_to_direct_chats(con, list({r["chat_id"] for r in rows})))
    # chat_id -> 'MM-DD' -> {year: (message_id, date, text)}; the latest message
    # per (chat, month-day, year) wins because rows arrive in ascending date.
    buckets = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cid = r["chat_id"]
        if cid not in direct:
            continue
        d = r["date"]
        buckets[cid][d[5:10]][d[:4]] = (r["id"], d, r["text"])
    # loose_threads holds one open row per (kind, chat_id) -- the upsert keys on
    # exactly that -- so a dyad with several recurring occasions must collapse to
    # ONE candidate; picking the soonest due_by surfaces the most imminent one
    # (which the proximity urgency in _score_urgency then ranks up), while the
    # runners-up are preserved in evidence rather than silently dropped. due_by
    # rolls off real today (cfg['today'], injected by detect_all) when present,
    # falling back to corpus-now for direct-call determinism.
    today = cfg.get("today") or cfg["now"]
    for cid, by_md in buckets.items():
        occasions = []
        for month_day, by_year in by_md.items():
            if len(by_year) < min_years:
                continue
            years = sorted(by_year)
            anchor_id, anchor_date, sample = by_year[years[-1]]
            occasions.append({
                "month_day": month_day,
                "years": years,
                "anchor_id": anchor_id,
                "anchor_date": anchor_date,
                "sample": sample,
                "due_by": _next_occurrence(today, month_day),
            })
        if not occasions:
            continue
        # Soonest due_by wins; tie-break most years observed, then month_day. A
        # None due_by (neither this nor next year resolved) sorts last.
        occasions.sort(key=lambda o: (o["due_by"] or "9999-99-99",
                                      -len(o["years"]), o["month_day"]))
        primary = occasions[0]
        others = [{"month_day": o["month_day"], "next_occurrence": o["due_by"],
                   "n_years": len(o["years"])} for o in occasions[1:]]
        yield CandidateRow(
            kind="inferred_occasion",
            chat_id=cid,
            handle_id=_counterparty_handle(con, cid),
            anchor_message_id=primary["anchor_id"],
            anchor_date=primary["anchor_date"],
            due_by=primary["due_by"],
            evidence={
                "month_day": primary["month_day"],
                "observed_years": primary["years"],
                "n_years": len(primary["years"]),
                "next_occurrence": primary["due_by"],
                "sample_message": (primary["sample"] or "")[:TEXT_PREVIEW_CHARS],
                "other_occasions": others,
            },
        )


DETECTORS.append(("inferred_occasion", detect_inferred_occasion))
# --- end detector: inferred_occasion ---
# --- end detector: unanswered_question ---


# ---------------------------------------------------------------------------
# Detector: dropped_plan / unmet_commitment (RES-57 / N4)
# ---------------------------------------------------------------------------
# --- detector: dropped_plan ---
# A plan or commitment raised in a 1:1 chat that the following conversation gives
# no sign was ever carried out. HARD-GATED and OFF BY DEFAULT
# (NUDGE_DROPPED_PLAN_ENABLED): the off path returns immediately -- it runs no
# LLM and never consults the eval gate, so turning the detector off does NOT
# depend on the owner having run the F4 measurement gate. Two stages once
# enabled:
#   Stage 1 -- an FTS cue-phrase pre-filter over messages_fts finds the
#              most-recent message that reads like a plan/commitment (the
#              anchor). One anchor per chat bounds the verifier to a single LLM
#              call per candidate chat.
#   Stage 2 -- ollama.chat_once_json asks Searle-style questions over the anchor
#              + up to NUDGE_PLAN_CONTEXT_MSGS following messages and returns
#              fulfilled | unfulfilled | UNKNOWN. Only "unfulfilled" is a loose
#              end; UNKNOWN -- and "fulfilled", and any unparseable reply --
#              SUPPRESS.
# Precision gate (F4): even enabled, nothing is written unless the dropped_plan
# class is cleared at >= 0.85 -- eval_gate_passed(con, 'dropped_plan',
# 'dropped_plan') is checked UP FRONT (so a failing/absent gate also skips the
# expensive verification). dropped_plan ships no seed, so an absent verdict
# suppresses. The Stage-0 exclusion gate + urgency run OUTSIDE this function in
# the shared driver, exactly as for every detector.

# PINNED module-level cue phrases (a frozenset, next to the detector) OR-joined
# into one FTS5 MATCH query. These are commissive / plan markers -- a message
# matching one is only a CANDIDATE anchor; the Stage-2 verifier decides whether
# it is a genuine, unfulfilled plan. Extending the list is expected.
_PLAN_CUE_PHRASES = frozenset({
    "let's grab", "lets grab", "let's get", "lets get", "let's do", "lets do",
    "we should", "we need to", "we have to", "we gotta", "we ought to",
    "rain check", "raincheck", "reschedule", "next week", "next time",
    "catch up soon", "grab dinner", "grab lunch", "grab coffee", "grab drinks",
    "get coffee", "get dinner", "get drinks", "come visit", "come over",
    "i'll send", "i'll get you", "i'll let you know", "i'll text you",
    "i promise", "let me know when", "circle back", "follow up", "we'll figure",
})

_PLAN_VERIFY_SYSTEM = (
    "You judge whether a plan or commitment made in a personal text conversation "
    "was followed through. You are given an ANCHOR message that may propose a "
    "plan or promise an action, followed by the messages that came AFTER it in "
    "the same chat. Answer two Searle-style questions: (1) Does the ANCHOR "
    "actually commit the speakers to a specific future action or plan -- a "
    "commissive or a concrete directive -- rather than small talk? (2) If so, do "
    "the following messages show it was carried out? Respond with ONLY one JSON "
    'object: {"verdict": "fulfilled" | "unfulfilled" | "UNKNOWN"}. Use '
    '"unfulfilled" ONLY when the ANCHOR is a genuine plan/commitment AND the '
    'following messages give no sign it happened. Use "UNKNOWN" whenever you are '
    "unsure, the ANCHOR is not a real plan, or there is too little context to "
    "tell."
)

_PLAN_VERIFY_NUDGE = (
    'Respond with only the JSON object {"verdict": "fulfilled" | "unfulfilled" '
    '| "UNKNOWN"} -- no prose, no code fences, nothing else.'
)


def _plan_fts_query():
    """The cue-phrase frozenset OR-joined into one FTS5 MATCH query. Each phrase
    is double-quoted so FTS5 treats it as a consecutive-token phrase -- an
    apostrophe in "let's" is a token separator, so it still matches "let's grab"."""
    return " OR ".join(f'"{p}"' for p in sorted(_PLAN_CUE_PHRASES))


def _plan_following_context(con, chat_id, anchor_id, anchor_date, limit):
    """Up to `limit` real messages strictly after the anchor, in time order --
    the hindsight the verifier reads to decide whether the plan happened."""
    return con.execute(
        f"SELECT m.is_from_me AS is_from_me, m.text AS text FROM messages m "
        f"WHERE m.chat_id = ? AND m.date IS NOT NULL "
        f"AND (m.date > ? OR (m.date = ? AND m.id > ?)) AND {_REAL} "
        f"ORDER BY m.date, m.id LIMIT ?",
        (chat_id, anchor_date, anchor_date, anchor_id, limit),
    ).fetchall()


def _render_plan_context(anchor_text, anchor_is_me, following):
    """The anchor + following messages as speaker-labelled lines for the
    verifier ('Me' / 'Them'), each capped at TEXT_PREVIEW_CHARS."""
    def who(is_me):
        return "Me" if is_me else "Them"
    lines = [f"ANCHOR {who(anchor_is_me)}: {(anchor_text or '')[:TEXT_PREVIEW_CHARS]}"]
    for row in following:
        lines.append(
            f"{who(row['is_from_me'])}: {(row['text'] or '')[:TEXT_PREVIEW_CHARS]}")
    return "\n".join(lines)


def _verify_plan(anchor_text, anchor_is_me, following, cfg):
    """Stage-2 verdict from ollama.chat_once_json: 'fulfilled' / 'unfulfilled' /
    'unknown' (lowercased), or None when the model is REACHABLE but returns no
    usable JSON. An unreachable / erroring backend (Ollama down, model not
    pulled, HTTP error) raises OSError/RuntimeError out of chat_once_json rather
    than returning None -- detect_dropped_plan catches that at the call site and
    degrades the pass. think is passed per the tri-state via cfg['nudges_think'];
    with gemma the effective value must be false -- a one-word verdict is not
    worth a reasoning pass, and NUDGES_THINK is only ever True/False/None (never
    an effort-level string)."""
    context = _render_plan_context(anchor_text, anchor_is_me, following)
    user = ("ANCHOR message and the conversation that followed it:\n\n"
            f"{context}\n\nWas the plan or commitment in the ANCHOR carried out?")
    result = ollama.chat_once_json(
        [{"role": "system", "content": _PLAN_VERIFY_SYSTEM},
         {"role": "user", "content": user}],
        model=cfg["nudges_model"], retry_nudge=_PLAN_VERIFY_NUDGE,
        timeout=cfg["nudges_timeout"], think=cfg["nudges_think"],
    )
    if not result:
        return None
    return str(result.get("verdict", "")).strip().lower()


def detect_dropped_plan(con, cfg) -> Iterable[CandidateRow]:
    """Fire for a plan/commitment a 1:1 chat raised and then let drop (see the
    fence header). OFF unless cfg['dropped_plan_enabled'] AND the F4 dropped_plan
    precision gate is cleared -- BOTH checked before any FTS or LLM work, so the
    disabled / ungated path writes nothing and calls no model. One candidate per
    chat: the most-recent cue-phrase message owns the verdict; UNKNOWN suppresses."""
    if not cfg["dropped_plan_enabled"]:
        return
    # Precision gate BEFORE the FTS/LLM work: a failing or absent verdict means
    # zero rows AND zero verification calls (dropped_plan ships no seed).
    if not eval_gate_passed(con, "dropped_plan", "dropped_plan"):
        return
    # --dry-run is a read-only preview that must call no model (see main's help).
    # dropped_plan is the ONLY model-backed detector, and its verdict cannot be
    # produced offline, so a dry run necessarily omits its candidates from scope.
    if cfg.get("dry_run"):
        return
    rows = con.execute(
        f"SELECT m.chat_id AS chat_id, m.id AS id, m.date AS date, "
        f"m.text AS text, m.is_from_me AS is_from_me "
        f"FROM messages_fts f JOIN messages m ON m.id = f.rowid "
        f"WHERE messages_fts MATCH ? AND m.date IS NOT NULL AND {_REAL} "
        f"ORDER BY m.date DESC, m.id DESC",
        (_plan_fts_query(),),
    ).fetchall()
    if not rows:
        return
    direct = set(_restrict_to_direct_chats(con, list({r["chat_id"] for r in rows})))
    seen = set()
    for r in rows:
        chat_id = r["chat_id"]
        if chat_id in seen or chat_id not in direct:
            continue
        seen.add(chat_id)  # the most-recent cue message owns this chat's verdict
        following = _plan_following_context(
            con, chat_id, r["id"], r["date"], cfg["plan_context_msgs"])
        try:
            verdict = _verify_plan(r["text"], r["is_from_me"], following, cfg)
        except (OSError, RuntimeError) as exc:
            # Verify backend unusable (Ollama down / model not pulled / HTTP
            # error). Unlike a reachable-but-unparseable reply (None, handled
            # below), retrying a dead backend per candidate is pointless -- each
            # can burn a full NUDGES_TIMEOUT. Mark the pass degraded so detect_all
            # exempts dropped_plan from absence-based auto-resolve (a delayed
            # close, never a wrongful one -- NONDETERMINISTIC_KINDS) and stop this
            # detector for the pass.
            cfg["dropped_plan_degraded"] = True
            print(f"  dropped_plan verify unavailable ({exc}); degrading this pass")
            return
        if verdict in (None, ""):
            # Verify ran but produced no usable verdict (unparseable/empty JSON
            # after retry, or a dict missing 'verdict'). Mark the run degraded so
            # the driver skips auto-resolving dropped_plan threads this pass. A
            # clean 'fulfilled'/'unknown' below is a legitimate resolve, NOT this.
            cfg["dropped_plan_degraded"] = True
            continue
        if verdict != "unfulfilled":
            continue  # clean UNKNOWN / fulfilled -> legitimate SUPPRESS
        yield CandidateRow(
            kind="dropped_plan",
            chat_id=chat_id,
            handle_id=_counterparty_handle(con, chat_id),
            anchor_message_id=r["id"],
            anchor_date=r["date"],
            evidence={
                "anchor_text": (r["text"] or "")[:TEXT_PREVIEW_CHARS],
                "anchor_from_me": bool(r["is_from_me"]),
                "verdict": verdict,
                "context_messages": len(following),
                "verified_by": cfg["nudges_model"],
                "gate": "dropped_plan (F4 precision gate cleared >= 0.85)",
            },
        )


DETECTORS.append(("dropped_plan", detect_dropped_plan))
# --- end detector: dropped_plan ---


# ---------------------------------------------------------------------------
# Detector: unreciprocated_goodnews (RES-57 / N4)
# ---------------------------------------------------------------------------
# --- detector: unreciprocated_goodnews ---
# Someone shared good news in a 1:1 chat and it drew anything other than an
# active-constructive response -- a moment worth circling back on. OPT-IN and OFF
# BY DEFAULT (NUDGE_GOODNEWS_ENABLED); the off path returns immediately. The
# signal is read straight out of the gemma chunk annotations: response_class
# inside chunk_annotations.events_json (a JSON FIELD, not a column), for
# source='llm' rows ONLY (distilled rows carry empty events). Because this leans
# on the annotator's judgement, its precision is INHERITED (gemma), not
# SQL-exact -- evidence_json says so in a precision_note, and also reports
# llm-annotation COVERAGE for the dyad so "no good news" is never confused with
# "the dyad was barely annotated". Pure detection; the shared driver runs Stage-0
# exclusion + urgency outside this function, as for every detector.

# Gable's capitalization taxonomy: active_constructive is the ideal (reciprocated)
# response; the three below are genuinely UNDER-received. A null/absent
# response_class means the annotator did NOT classify the response -- that is
# "unknown", not "under-received", so it must NOT fire (it was a false-positive
# source: unclassified good-news events with urgency ~0).
_GOODNEWS_RECIPROCATED = "active_constructive"
_GOODNEWS_UNDER_RECEIVED = frozenset({
    "passive_constructive", "passive_destructive", "active_destructive"})


def _goodnews_coverage(con, chat_id):
    """The dyad's llm-annotation coverage: how many of its chunks carry a
    source='llm' annotation, out of all its chunks. Reported in evidence so a
    firing dyad's numbers can be read in context -- the good-news signal only
    ever sees llm rows, so low coverage means a soft, thin signal."""
    row = con.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN a.source = 'llm' THEN 1 ELSE 0 END) AS llm "
        "FROM chunks c LEFT JOIN chunk_annotations a ON a.chunk_id = c.id "
        "WHERE c.chat_id = ?",
        (chat_id,),
    ).fetchone()
    total = row["total"] or 0
    llm = row["llm"] or 0
    return {
        "llm_annotated_chunks": llm,
        "total_chunks": total,
        "fraction": round(llm / total, 4) if total else 0.0,
    }


def _unreciprocated_good_news(events_json):
    """The good_news events in one chunk whose response_class is a genuine
    UNDER-RECEIVED class (passive_constructive / passive_destructive /
    active_destructive). A null/absent or reciprocated (active_constructive)
    class does NOT fire. [] on malformed JSON or when there is no such event."""
    try:
        events = json.loads(events_json or "[]")
    except (ValueError, TypeError):
        return []
    if not isinstance(events, list):
        return []
    return [e for e in events
            if isinstance(e, dict) and e.get("kind") == "good_news"
            and e.get("response_class") in _GOODNEWS_UNDER_RECEIVED]


def detect_unreciprocated_goodnews(con, cfg) -> Iterable[CandidateRow]:
    """Fire for a 1:1 chat whose most-recent llm-annotated good-news moment drew
    a non active-constructive response (see the fence header). OFF unless
    cfg['goodnews_enabled']; reads events_json for source='llm' rows only and
    reports llm-annotation coverage in evidence. One candidate per chat: the
    newest chunk carrying an unreciprocated good-news event owns it."""
    if not cfg["goodnews_enabled"]:
        return
    rows = con.execute(
        "SELECT c.chat_id AS chat_id, c.last_message_id AS mid, "
        "c.last_date AS last_date, a.events_json AS events_json "
        "FROM chunks c JOIN chunk_annotations a ON a.chunk_id = c.id "
        "WHERE a.source = 'llm' AND a.events_json IS NOT NULL "
        "AND a.events_json NOT IN ('', '[]') "
        "ORDER BY c.chat_id, c.last_date DESC, c.id DESC"
    ).fetchall()
    if not rows:
        return
    direct = set(_restrict_to_direct_chats(con, list({r["chat_id"] for r in rows})))
    max_age_days = cfg.get("goodnews_max_age_days")
    now = cfg["now"]
    seen = set()
    for r in rows:
        chat_id = r["chat_id"]
        if chat_id in seen or chat_id not in direct:
            continue
        unrecip = _unreciprocated_good_news(r["events_json"])
        if not unrecip:
            continue  # this chunk had no under-received good news; try older ones
        seen.add(chat_id)  # newest unreciprocated good-news chunk owns the chat
        # Recency floor: the newest unreciprocated good-news is the freshest one
        # this dyad has -- if even that is stale, belatedly nudging about it is
        # noise (and its urgency decays to ~0), so the dyad is skipped entirely.
        if max_age_days is not None and r["last_date"]:
            age_days = (now - datetime.datetime.fromisoformat(r["last_date"])).days
            if age_days > max_age_days:
                continue
        first = unrecip[0]
        yield CandidateRow(
            kind="unreciprocated_goodnews",
            chat_id=chat_id,
            handle_id=_counterparty_handle(con, chat_id),
            anchor_message_id=r["mid"],
            anchor_date=r["last_date"],
            evidence={
                "good_news_excerpt": (first.get("excerpt") or "")[:TEXT_PREVIEW_CHARS],
                "response_class": first.get("response_class"),
                "n_unreciprocated": len(unrecip),
                "llm_annotation_coverage": _goodnews_coverage(con, chat_id),
                "precision_note": "Signal inherited from gemma chunk annotations "
                                  "(events_json response_class), not an SQL-exact "
                                  "measure; treat as a soft hint.",
            },
        )


DETECTORS.append(("unreciprocated_goodnews", detect_unreciprocated_goodnews))
# --- end detector: unreciprocated_goodnews ---
