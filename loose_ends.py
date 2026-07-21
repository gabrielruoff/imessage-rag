"""Loose Ends detector CLI: find conversational threads the user let drift and
persist them as `loose_threads` rows to nudge about later. Mirrors annotate.py's
batch-CLI ergonomics (resumable, commits as it goes, Ctrl-C safe).

Pipeline shape, in order:

  Stage 0 -- exclusion gate (ALWAYS first): load nudge_exclusions, then auto-mute
             counterparties whose most-recent llm-annotated exchange is negative
             (mean valence <= NUDGE_VALENCE_EXCLUDE) or who have never received an
             outbound message. NO candidate is EVER generated for an excluded
             contact.
  Stage 1 -- detectors: each registered detector is a pure function
             detect_fn(con, cfg) -> Iterable[CandidateRow]. A detector only
             DETECTS; it never writes rows and never applies exclusions or
             scoring -- those run once, outside detectors, in the shared driver
             loop below. N1 registers `faded_initiator` (pure SQL over the
             derived `sessions` table). Follow-ups (N2/N3/N4) append to DETECTORS
             in their own fence.
  Stage 3 -- urgency scoring (Python): urgency = kind_weight *
             exp(-age_days/halflife * ln 2), age measured from the corpus's
             latest message so the freshest threads score highest.
  Narration -- one short LLM line per open thread into nudge_text (skippable with
             --no-llm). This is the ONLY place a model is used; the loose_ends_model
             ingest_state key guards ONLY the LLM-derived columns (nudge_text,
             llm_verdict, llm_reason) -- on a model change it clears/regenerates
             those columns and never blocks or deletes the SQL-detected rows.

Every dyad detector is confined to 1:1 chats (F3 _restrict_to_direct_chats,
is_group=0) intersected with the shared _REAL_MESSAGE_CLAUSES. This is a
STANDALONE CLI (invoked from the Makefile `pipeline` target AFTER annotate, since
Stage 0 reads chunk_annotations), not an ingest stage.

Usage: loose_ends.py [--model NAME] [--limit N] [--no-llm] [--dry-run]
"""

import argparse
import datetime
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable

import ollama
import progress
import schema
from config import (
    NUDGE_FADED_INIT_DROP,
    NUDGE_FADED_INIT_MIN_SESSIONS,
    NUDGE_FADED_INIT_WINDOW_DAYS,
    NUDGE_KIND_WEIGHTS,
    NUDGE_URGENCY_HALFLIFE_DAYS,
    NUDGE_VALENCE_EXCLUDE,
    NUDGES_MODEL,
    NUDGES_THINK,
    NUDGES_TIMEOUT,
    TARGET_DB,
)
from db import CONTACT_LABEL_SQL, connect_default, get_state, run_cli, set_state
from query_tools._shared import _REAL_MESSAGE_CLAUSES, _restrict_to_direct_chats
from query_tools.nudges import (
    _add_exclusion,
    _is_excluded,
    _load_exclusions,
    _utcnow_iso,
)

# --- detector: dormant_tie ---
# Additive per-detector imports (RES-55 / N2). Kept in their own fenced block so
# concurrent N-wave detectors never contend for the same import statement, and
# so the public lsm_score re-export is used rather than the package-private name.
from config import (
    NUDGE_DORMANT_MIN_GAP_DAYS,
    NUDGE_DORMANT_MIN_MESSAGES,
    NUDGE_DORMANT_TOP_FRAC,
)
from query_tools.analytics_language import lsm_score
# --- end detector: dormant_tie ---

# --- detector: unanswered_question ---
# Additive per-detector imports (RES-56 / N3), fenced so concurrent N-wave
# detectors never contend for the shared import block. `re` powers the
# final-sentence split in the rhetorical test; TEXT_PREVIEW_CHARS caps the
# question text stored in evidence_json for narration.
import re

from config import (
    NUDGE_QUESTION_GRACE_HOURS,
    NUDGE_QUESTION_INCLUDE_ANSWERED_LATE,
    NUDGE_QUESTION_MIN_CHARS,
)
# --- detector: inferred_occasion ---
# Additive per-detector imports (RES-58 / N5), fenced so concurrent N-wave
# detectors never contend for the shared import block. defaultdict powers the
# (month, day) -> {year: message} bucketing that turns recurring congratulatory
# messages into an occasion; NUDGE_OCCASION_MIN_YEARS is the recurrence floor.
from collections import defaultdict

from config import NUDGE_OCCASION_MIN_YEARS
# --- end detector: inferred_occasion ---
from query_tools._shared import TEXT_PREVIEW_CHARS
# --- end detector: unanswered_question ---

# --- detector: dropped_plan ---
# Additive per-detector imports (RES-57 / N4), fenced so concurrent N-wave
# detectors never contend for the shared import block. The F4 precision gate is
# CONSUMED here (never built here): eval.evalkit.gates is stdlib-only to import
# and pulls in no app module, so loading it costs no Ollama. TEXT_PREVIEW_CHARS
# (imported above) caps the anchor/context text this detector renders.
from config import NUDGE_DROPPED_PLAN_ENABLED, NUDGE_PLAN_CONTEXT_MSGS
from eval.evalkit.gates import eval_gate_passed
# --- end detector: dropped_plan ---

# --- detector: unreciprocated_goodnews ---
# Additive per-detector import (RES-57 / N4). Opt-in flag only; the signal is
# read out of chunk_annotations.events_json (source='llm') and its precision is
# inherited from the gemma annotations, not SQL-exact.
from config import NUDGE_GOODNEWS_ENABLED, NUDGE_GOODNEWS_MAX_AGE_DAYS
# --- end detector: unreciprocated_goodnews ---

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# The shared real-message predicate, spelled once for the string joins below.
_REAL = " AND ".join(_REAL_MESSAGE_CLAUSES)

# Stage-0 auto-exclusion reasons (the 'auto:' prefix distinguishes them from a
# user mute written by mute_contact_from_nudges).
_REASON_NEGATIVE = "auto:negative_valence"
_REASON_ZERO_OUTBOUND = "auto:zero_outbound"

# Fallback urgency weight for a kind absent from NUDGE_KIND_WEIGHTS.
_DEFAULT_KIND_WEIGHT = 0.5


def _connect():
    return connect_default(DB_PATH, rows=True)


# ---------------------------------------------------------------------------
# Candidate rows + the detector registry
# ---------------------------------------------------------------------------

@dataclass
class CandidateRow:
    """A detector's output: the loose_threads column fields it is responsible
    for (minus id/urgency/status, which the driver owns). A detector fills in
    the detection-relevant fields; the driver computes urgency, sets
    status='open'/detected_at, applies the Stage-0 exclusion gate, and later
    narrates nudge_text. A detector never writes a row itself."""

    kind: str
    chat_id: int
    handle_id: int | None = None
    anchor_message_id: int | None = None
    anchor_date: str | None = None
    evidence: dict = field(default_factory=dict)
    due_by: str | None = None
    source: str = "sql"


# Detector registry: (kind, detect_fn) pairs. N1 registers faded_initiator;
# follow-ups append their own (kind, detect_fn) in a per-detector fence. The
# driver loop below is the single place exclusion + urgency run.
DETECTORS: list[tuple[str, Callable[[object, dict], Iterable[CandidateRow]]]] = []


# ---------------------------------------------------------------------------
# Shared dyad primitives (consumed by detectors here and in N2/N3)
# ---------------------------------------------------------------------------

def _dyad_gap_p90(con, chat_id):
    """p90 of inter-message gaps (seconds) between consecutive real messages of
    a dyad, or None when the chat has fewer than two such messages. Single home
    for this primitive: N2/N3 detectors consume it (e.g. to decide a current
    silence is anomalously long for how this pair normally paces)."""
    dates = [
        datetime.datetime.fromisoformat(r[0])
        for r in con.execute(
            f"SELECT m.date FROM messages m "
            f"WHERE m.chat_id = ? AND m.date IS NOT NULL AND {_REAL} "
            f"ORDER BY m.date, m.id",
            (chat_id,),
        ).fetchall()
    ]
    if len(dates) < 2:
        return None
    gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
    if len(gaps) < 2:
        return gaps[0]
    return statistics.quantiles(gaps, n=10)[8]


def _counterparty_handle(con, chat_id):
    """The 1:1 chat's counterparty handle_id (the participant that is not me)."""
    row = con.execute(
        "SELECT handle_id FROM chat_participants WHERE chat_id = ? "
        "ORDER BY handle_id LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row[0] if row else None


def _last_message(con, chat_id):
    """(id, date) of the most recent real message in a chat -- the nudge anchor."""
    row = con.execute(
        f"SELECT m.id, m.date FROM messages m "
        f"WHERE m.chat_id = ? AND m.date IS NOT NULL AND {_REAL} "
        f"ORDER BY m.date DESC, m.id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _handle_contact(con, handle_id):
    """contact_id owning a handle (NULL when the handle is unlinked)."""
    if handle_id is None:
        return None
    row = con.execute(
        "SELECT contact_id FROM handles WHERE id = ?", (handle_id,)
    ).fetchone()
    return row[0] if row else None


def _corpus_now(con):
    """The data's "now": the latest real message date in the corpus (a naive
    datetime, matching the unlabeled-UTC storage). Falls back to wall-clock UTC
    on an empty corpus."""
    row = con.execute(
        f"SELECT MAX(m.date) FROM messages m WHERE m.date IS NOT NULL AND {_REAL}"
    ).fetchone()
    if row and row[0]:
        return datetime.datetime.fromisoformat(row[0])
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Detector: faded_initiator (pure SQL over the sessions table)
# ---------------------------------------------------------------------------

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
    are you?'."""
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
    ('MM-DD'), as an ISO 'YYYY-MM-DD' string. Rolls to next year once this
    year's date has already passed; a Feb 29 occasion in a non-leap target year
    falls back to Feb 28 so the reminder never silently vanishes."""
    month, day = int(month_day[:2]), int(month_day[3:5])
    today = now.date()
    for year in (today.year, today.year + 1):
        try:
            candidate = datetime.date(year, month, day)
        except ValueError:  # Feb 29 in a non-leap year
            candidate = datetime.date(year, month, 28)
        if candidate >= today:
            return candidate.isoformat()
    return None


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
    'unknown' (lowercased), or None when the model returns no usable JSON. think
    is passed per the tri-state via cfg['nudges_think']; with gemma the effective
    value must be false -- a one-word verdict is not worth a reasoning pass, and
    NUDGES_THINK is only ever True/False/None (never an effort-level string)."""
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
        verdict = _verify_plan(r["text"], r["is_from_me"], following, cfg)
        if verdict != "unfulfilled":
            continue  # UNKNOWN / fulfilled / unparseable -> SUPPRESS
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


# ---------------------------------------------------------------------------
# Stage 0: exclusion gate
# ---------------------------------------------------------------------------

def _direct_chat_counterparties(con):
    """(chat_id, handle_id, contact_id) for every 1:1 chat's counterparty."""
    return con.execute(
        """
        SELECT cp.chat_id, cp.handle_id, h.contact_id
        FROM chat_participants cp
        JOIN chats ch ON ch.id = cp.chat_id AND ch.is_group = 0
        JOIN handles h ON h.id = cp.handle_id
        """
    ).fetchall()


def _outbound_count(con, chat_id):
    """Number of real messages I sent in a chat (is_from_me = 1)."""
    return con.execute(
        f"SELECT COUNT(*) FROM messages m "
        f"WHERE m.chat_id = ? AND m.is_from_me = 1 AND {_REAL}",
        (chat_id,),
    ).fetchone()[0]


def _recent_llm_valence(con, chat_id):
    """Mean valence of the chat's most-recent llm-annotated exchange -- taken as
    the most-recent chunk (by last_date) carrying a source='llm' annotation.
    None when the chat has no llm-annotated chunk, so distilled-only chats never
    trip the valence gate."""
    row = con.execute(
        "SELECT a.valence FROM chunks c JOIN chunk_annotations a ON a.chunk_id = c.id "
        "WHERE c.chat_id = ? AND a.source = 'llm' AND a.valence IS NOT NULL "
        "ORDER BY c.last_date DESC, c.id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row[0] if row else None


def run_exclusion_gate(con, cfg, now, dry_run=False):
    """Stage 0, always first. Load stored exclusions, then auto-mute any 1:1
    counterparty whose most-recent llm-annotated exchange is negative
    (<= NUDGE_VALENCE_EXCLUDE) or who has never received an outbound message.
    Persists new auto-mutes to nudge_exclusions (dedup-safe) AND returns the
    in-memory (handles, contacts) sets, so no candidate is ever generated for
    them. Returns (excluded_handles, excluded_contacts, n_added).

    Under dry_run the auto-mutes are NOT written or committed (--dry-run is a
    read-only preview), but the in-memory sets and n_added still reflect what
    WOULD be muted -- the returned sets drive suppression for this same run."""
    excluded_handles, excluded_contacts = _load_exclusions(con)
    n_added = 0
    for chat_id, handle_id, contact_id in _direct_chat_counterparties(con):
        if _is_excluded(handle_id, contact_id, excluded_handles, excluded_contacts):
            continue
        reason = None
        if _outbound_count(con, chat_id) == 0:
            reason = _REASON_ZERO_OUTBOUND
        else:
            valence = _recent_llm_valence(con, chat_id)
            if valence is not None and valence <= cfg["valence_exclude"]:
                reason = _REASON_NEGATIVE
        if reason is None:
            continue
        if _add_exclusion(con, handle_id=handle_id, contact_id=contact_id,
                          reason=reason, now=now, persist=not dry_run):
            n_added += 1
        if handle_id is not None:
            excluded_handles.add(handle_id)
        if contact_id is not None:
            excluded_contacts.add(contact_id)
    if not dry_run:
        con.commit()
    return excluded_handles, excluded_contacts, n_added


# ---------------------------------------------------------------------------
# Stage 3: urgency + row persistence
# ---------------------------------------------------------------------------

def _score_urgency(kind, anchor_date, ref_now, cfg, due_by=None):
    """urgency = kind_weight * exp(-age_days/halflife * ln 2); age from ref_now
    (the corpus's latest message). A missing anchor scores the bare weight.

    inferred_occasion is forward-looking, not backward: its anchor is a past
    occurrence up to ~365d old, so the age formula would bury it (~0.0002). With
    a due_by it instead decays on days-*until*-due from real today, so a due-soon
    occasion scores near full weight and ranks up the digest ahead of the date."""
    weight = cfg["kind_weights"].get(kind, _DEFAULT_KIND_WEIGHT)
    if kind == "inferred_occasion" and due_by:
        today = (cfg.get("today") or ref_now).date()
        days_until = max(0.0, (datetime.date.fromisoformat(due_by) - today).days)
        return round(weight * math.exp(-days_until / cfg["halflife_days"] * math.log(2)), 6)
    if not anchor_date:
        return round(weight, 6)
    age_days = max(
        0.0,
        (ref_now - datetime.datetime.fromisoformat(anchor_date)).total_seconds() / 86400.0,
    )
    return round(weight * math.exp(-age_days / cfg["halflife_days"] * math.log(2)), 6)


def _upsert_thread(con, cand, urgency, now):
    """Insert a new open thread for (kind, chat_id), or refresh the existing
    open one (anchor/urgency/evidence move as the corpus grows). A dismissed
    thread for the same (kind, chat_id) is left untouched -- a detector never
    resurrects a dismissal. Returns the row id, or None when skipped."""
    existing = con.execute(
        "SELECT id, status FROM loose_threads WHERE kind = ? AND chat_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (cand.kind, cand.chat_id),
    ).fetchone()
    evidence_json = json.dumps(cand.evidence)
    if existing is not None:
        if existing["status"] == "dismissed":
            return None
        con.execute(
            "UPDATE loose_threads SET handle_id = ?, anchor_message_id = ?, "
            "anchor_date = ?, due_by = ?, urgency = ?, evidence_json = ?, "
            "source = ?, detected_at = ? WHERE id = ?",
            (cand.handle_id, cand.anchor_message_id, cand.anchor_date, cand.due_by,
             urgency, evidence_json, cand.source, now, existing["id"]),
        )
        return existing["id"]
    cur = con.execute(
        "INSERT INTO loose_threads(kind, chat_id, handle_id, anchor_message_id, "
        "anchor_date, due_by, urgency, status, evidence_json, source, detected_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
        (cand.kind, cand.chat_id, cand.handle_id, cand.anchor_message_id,
         cand.anchor_date, cand.due_by, urgency, evidence_json, cand.source, now),
    )
    return cur.lastrowid


def _build_cfg(con):
    return {
        "now": _corpus_now(con),
        "valence_exclude": NUDGE_VALENCE_EXCLUDE,
        "faded_drop": NUDGE_FADED_INIT_DROP,
        "faded_min_sessions": NUDGE_FADED_INIT_MIN_SESSIONS,
        "faded_window_days": NUDGE_FADED_INIT_WINDOW_DAYS,
        "kind_weights": NUDGE_KIND_WEIGHTS,
        "halflife_days": max(NUDGE_URGENCY_HALFLIFE_DAYS, 1e-9),
        # --- detector: dormant_tie ---
        "dormant_top_frac": NUDGE_DORMANT_TOP_FRAC,
        "dormant_min_messages": NUDGE_DORMANT_MIN_MESSAGES,
        "dormant_min_gap_days": NUDGE_DORMANT_MIN_GAP_DAYS,
        # --- end detector: dormant_tie ---
        # --- detector: unanswered_question ---
        "question_min_chars": NUDGE_QUESTION_MIN_CHARS,
        "question_grace_hours": NUDGE_QUESTION_GRACE_HOURS,
        "question_include_answered_late": NUDGE_QUESTION_INCLUDE_ANSWERED_LATE,
        # --- end detector: unanswered_question ---
        # --- detector: inferred_occasion ---
        "occasion_min_years": NUDGE_OCCASION_MIN_YEARS,
        # --- end detector: inferred_occasion ---
        # --- detector: dropped_plan ---
        # HARD-GATED: dropped_plan_enabled is the default-OFF flag; the verifier
        # reuses the narration model/timeout and passes think=nudges_think per
        # ollama's tri-state (false/omitted for gemma -- a one-word verdict).
        "dropped_plan_enabled": NUDGE_DROPPED_PLAN_ENABLED,
        "plan_context_msgs": NUDGE_PLAN_CONTEXT_MSGS,
        "nudges_model": NUDGES_MODEL,
        "nudges_think": NUDGES_THINK,
        "nudges_timeout": NUDGES_TIMEOUT,
        # --- end detector: dropped_plan ---
        # --- detector: unreciprocated_goodnews ---
        "goodnews_enabled": NUDGE_GOODNEWS_ENABLED,
        "goodnews_max_age_days": NUDGE_GOODNEWS_MAX_AGE_DAYS,
        # --- end detector: unreciprocated_goodnews ---
    }


def detect_all(con, cfg, event_now, dry_run=False):
    """The shared driver loop: Stage 0 gate, then every detector, with the
    exclusion gate and urgency scoring applied OUTSIDE the detectors. Returns a
    summary dict (and the written thread ids)."""
    cfg["dry_run"] = dry_run  # rides cfg so detectors (uniform detect_fn(con, cfg)) can read it
    # Real wall-clock "today" for forward-looking reads (inferred_occasion's
    # due_by / proximity urgency); backward age scoring keeps using corpus-now.
    try:
        cfg["today"] = datetime.datetime.fromisoformat(event_now.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        cfg["today"] = None
    excluded_handles, excluded_contacts, n_auto = run_exclusion_gate(
        con, cfg, event_now, dry_run=dry_run)
    n_candidates = n_written = n_suppressed = 0
    thread_ids = []
    for _kind, detect_fn in DETECTORS:
        for cand in detect_fn(con, cfg):
            n_candidates += 1
            contact_id = _handle_contact(con, cand.handle_id)
            if _is_excluded(cand.handle_id, contact_id,
                            excluded_handles, excluded_contacts):
                n_suppressed += 1
                continue
            urgency = _score_urgency(cand.kind, cand.anchor_date, cfg["now"], cfg,
                                     due_by=cand.due_by)
            if dry_run:
                n_written += 1
                continue
            tid = _upsert_thread(con, cand, urgency, event_now)
            if tid is not None:
                thread_ids.append(tid)
                n_written += 1
    if not dry_run:
        con.commit()
    return {"auto_excluded": n_auto, "candidates": n_candidates,
            "suppressed": n_suppressed, "written": n_written, "thread_ids": thread_ids}


# ---------------------------------------------------------------------------
# Narration (the only LLM stage; guarded by loose_ends_model)
# ---------------------------------------------------------------------------

_NARRATE_SYSTEM = (
    "You write a single short, warm one-line reminder nudging the user to "
    "reconnect with someone they have drifted from. No preamble, no quotes, "
    "one sentence, under 25 words."
)


def _guard_llm_model(con, model):
    """Guard ONLY the LLM-derived columns. On a model change with rows present,
    clear nudge_text/llm_verdict/llm_reason for regeneration -- never block or
    delete the SQL-detected rows (detectors are model-free, unlike annotate's
    refuse-outright guard)."""
    prev = get_state(con, "loose_ends_model")
    if prev and prev != model:
        con.execute(
            "UPDATE loose_threads SET nudge_text = NULL, llm_verdict = NULL, "
            "llm_reason = NULL WHERE nudge_text IS NOT NULL OR llm_verdict IS NOT NULL "
            "OR llm_reason IS NOT NULL"
        )
        print(f"loose_threads LLM columns were produced by {prev}, now {model}; "
              f"cleared them for regeneration (SQL-detected rows kept).")
    set_state(con, "loose_ends_model", model)
    con.commit()


def _threads_needing_narration(con):
    """Open threads with no nudge_text yet, urgency-first (resumable: a later
    run picks up whatever was left)."""
    return con.execute(
        f"""
        SELECT lt.id, lt.kind, lt.evidence_json,
               CASE WHEN lt.handle_id IS NULL THEN NULL ELSE {CONTACT_LABEL_SQL} END
                   AS label
        FROM loose_threads lt
        LEFT JOIN handles h ON h.id = lt.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE lt.status = 'open' AND lt.nudge_text IS NULL
        ORDER BY lt.urgency DESC, lt.id DESC
        """
    ).fetchall()


def _narrate_one(kind, label, evidence, model, think, timeout):
    """One short LLM line for a thread. think is passed through per the tri-state
    rule (True/False/None). Returns the first non-empty line, or None."""
    user = (f"Reason: {kind}. Person: {label or 'this contact'}. "
            f"Signals: {json.dumps(evidence)}. Write the one-line reminder.")
    msg = ollama.chat_once(
        [{"role": "system", "content": _NARRATE_SYSTEM},
         {"role": "user", "content": user}],
        model=model, think=think, timeout=timeout,
    )
    text = (msg.get("content") or "").strip()
    return text.splitlines()[0].strip() if text else None


def narrate_threads(con, model, think, timeout, limit=None):
    """Fill nudge_text on open, un-narrated threads (resumable, commits per row,
    Ctrl-C safe)."""
    _guard_llm_model(con, model)
    rows = _threads_needing_narration(con)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        print("No loose threads need narration.")
        return 0
    n = 0
    try:
        with progress.track("Narrating nudges", total=len(rows), header=False) as p:
            for r in rows:
                text = _narrate_one(
                    r["kind"], r["label"], json.loads(r["evidence_json"] or "null"),
                    model, think, timeout,
                )
                if text:
                    con.execute("UPDATE loose_threads SET nudge_text = ? WHERE id = ?",
                                (text, r["id"]))
                    n += 1
                    con.commit()
                p.advance()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- narrated {n}; rerun to resume.")
        return n
    print(f"Narrated {n} nudges")
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect loose threads (nudges): SQL detectors behind an "
                    "exclusion gate, urgency-scored, with an optional one-line "
                    "LLM narration. Resumable; Ctrl-C commits progress.")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model for narration (default {NUDGES_MODEL})")
    parser.add_argument("--limit", type=int, metavar="N", default=None,
                        help="narrate at most N threads this run")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the nudge_text narration (SQL detection only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect and print the scope without writing rows or "
                             "calling the model (the LLM-backed dropped_plan "
                             "detector is skipped under --dry-run)")
    # --- detector: dormant_tie ---
    parser.add_argument("--acknowledge-reconnect", action="store_true",
                        help="opt in (once, durably) to dormant-tie reconnect "
                             "nudges: sets the ingest_state nudges_reconnect_ack "
                             "key so get_loose_threads stops hiding them, then "
                             "exits without running detection")
    # --- end detector: dormant_tie ---
    args = parser.parse_args(argv)

    model = args.model or NUDGES_MODEL
    con = _connect()
    try:
        # Self-heal the tables (standalone runs / older DBs). Idempotent.
        schema._create_nudges_tables(con)
        con.commit()
        # --- detector: dormant_tie ---
        # A pure toggle: record the durable opt-in and exit. Kept non-detecting
        # so a batch run never has to be interactive to surface reconnect nudges.
        if args.acknowledge_reconnect:
            set_state(con, "nudges_reconnect_ack", _utcnow_iso())
            con.commit()
            print("Dormant-tie reconnect nudges enabled (nudges_reconnect_ack "
                  "set); they will now appear in get_loose_threads.")
            return 0
        # --- end detector: dormant_tie ---
        cfg = _build_cfg(con)
        event_now = _utcnow_iso()
        summary = detect_all(con, cfg, event_now, dry_run=args.dry_run)
        print(f"Loose threads: {summary['candidates']} candidates, "
              f"{summary['suppressed']} suppressed, {summary['written']} written "
              f"({summary['auto_excluded']} new auto-exclusions)"
              f"{' [--dry-run: nothing written]' if args.dry_run else ''}")
        if args.dry_run:
            return 0
        # --- detector: dormant_tie ---
        # First-run batch-hide setup: dormant-tie rows are written like any
        # other, but get_loose_threads keeps them hidden until the opt-in above.
        # Surface a one-line pointer when some were written and the key is unset.
        if get_state(con, "nudges_reconnect_ack") is None:
            n_dormant = con.execute(
                "SELECT COUNT(*) FROM loose_threads "
                "WHERE status = 'open' AND kind = 'dormant_tie'"
            ).fetchone()[0]
            if n_dormant:
                print(f"{n_dormant} dormant-tie reconnect nudge(s) written but "
                      "hidden from the digest. Run `loose_ends.py "
                      "--acknowledge-reconnect` to include them.")
        # --- end detector: dormant_tie ---
        if not args.no_llm:
            narrate_threads(con, model, NUDGES_THINK, NUDGES_TIMEOUT, limit=args.limit)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    run_cli(main)
