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
             --no-llm). NOT the only place a model is used: the dropped_plan
             detector (opt-in, F4-gated) verifies each candidate with the same
             model during DETECTION, which --no-llm does not skip. The
             loose_ends_model ingest_state key guards ONLY the LLM-derived
             columns (nudge_text, llm_verdict, llm_reason) and REFUSES a model
             change while they hold anything -- like the annotate_model and
             conflict_model guards -- never blocking or deleting the
             SQL-detected rows. Of the three, only nudge_text has a writer
             today; the other two are reserved.

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

import ollama
import progress
import schema
from config import (
    NUDGE_FADED_INIT_DROP,
    NUDGE_FADED_INIT_MIN_SESSIONS,
    NUDGE_FADED_INIT_WINDOW_DAYS,
    NUDGE_KIND_WEIGHTS,
    NUDGE_OWNER_CONTACT_IDS,
    NUDGE_OWNER_HANDLES,
    NUDGE_SHORTCODE_MAX_DIGITS,
    NUDGE_URGENCY_HALFLIFE_DAYS,
    NUDGE_VALENCE_EXCLUDE,
    NUDGES_MODEL,
    NUDGES_THINK,
    NUDGES_TIMEOUT,
    TARGET_DB,
)
from consent import parse_consent_ids
from db import (
    CONTACT_LABEL_SQL,
    check_model_guard,
    connect_default,
    get_state,
    run_cli,
    set_state,
)
from nudge_shared import (  # noqa: F401 -- re-exported for tests / nudge_detectors
    DETECTORS,
    CandidateRow,
    _REAL,
    _counterparty_handle,
    _dyad_gap_p90,
    _last_message,
)
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
    NUDGE_QUESTION_MAX_AGE_DAYS,
    NUDGE_QUESTION_MIN_CHARS,
)
# --- detector: inferred_occasion ---
# Additive per-detector imports (RES-58 / N5), fenced so concurrent N-wave
# detectors never contend for the shared import block. defaultdict powers the
# (month, day) -> {year: message} bucketing that turns recurring congratulatory
# messages into an occasion; NUDGE_OCCASION_MIN_YEARS is the recurrence floor.

from config import NUDGE_OCCASION_MIN_YEARS
# --- end detector: inferred_occasion ---
# --- end detector: unanswered_question ---

# --- detector: dropped_plan ---
# Additive per-detector imports (RES-57 / N4), fenced so concurrent N-wave
# detectors never contend for the shared import block. The F4 precision gate is
# CONSUMED here (never built here): eval.evalkit.gates is stdlib-only to import
# and pulls in no app module, so loading it costs no Ollama. TEXT_PREVIEW_CHARS
# (imported above) caps the anchor/context text this detector renders.
from config import NUDGE_DROPPED_PLAN_ENABLED, NUDGE_PLAN_CONTEXT_MSGS
# --- end detector: dropped_plan ---

# --- detector: unreciprocated_goodnews ---
# Additive per-detector import (RES-57 / N4). Opt-in flag only; the signal is
# read out of chunk_annotations.events_json (source='llm') and its precision is
# inherited from the gemma annotations, not SQL-exact.
from config import NUDGE_GOODNEWS_ENABLED, NUDGE_GOODNEWS_MAX_AGE_DAYS
# --- end detector: unreciprocated_goodnews ---

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# Stage-0 auto-exclusion reasons (the 'auto:' prefix distinguishes them from a
# user mute written by mute_contact_from_nudges).
_REASON_NEGATIVE = "auto:negative_valence"
_REASON_ZERO_OUTBOUND = "auto:zero_outbound"
_REASON_SELF = "auto:self_identity"          # the account owner's own number/card
_REASON_AUTOMATED = "auto:automated_sender"  # shortcode / business / bot sender

# Fallback urgency weight for a kind absent from NUDGE_KIND_WEIGHTS.
_DEFAULT_KIND_WEIGHT = 0.5


def _connect():
    return connect_default(DB_PATH, rows=True)


# Kinds whose detection re-runs a live, nondeterministic model verdict every
# pass (dropped_plan is the only one -- the rest are deterministic SQL / read
# precomputed annotations). Reconciliation auto-resolves any OPEN thread a run
# does NOT re-detect, which is sound for deterministic detectors but wrong for
# these: a degraded verify (model unreachable / unparseable JSON) suppresses the
# candidate and would silently resolve a still-open thread. When such a detector
# reports it "ran degraded" this pass (cfg['<kind>_degraded']), the driver
# exempts that kind from absence-based auto-resolve -- a delayed close on the
# next clean run, never a wrongful one.
NONDETERMINISTIC_KINDS = frozenset({"dropped_plan"})


# ---------------------------------------------------------------------------
# Shared dyad primitives (_dyad_gap_p90/_counterparty_handle/_last_message) live
# in nudge_shared, imported at the top of this module so loose_ends.<name> and
# nudge_detectors both resolve them from one home.
# ---------------------------------------------------------------------------

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
# Detectors (defined in nudge_detectors, split out of this file)
# ---------------------------------------------------------------------------
# Importing nudge_detectors defines the six detectors and registers each into
# DETECTORS (above) via its own DETECTORS.append; they are re-exported here so
# loose_ends.<detector> and the detector-internal helpers the tests reference
# keep resolving. nudge_detectors imports the shared primitives defined above,
# so this runs after them.
from nudge_detectors import detect_faded_initiator  # noqa: E402,F401
from nudge_detectors import detect_dormant_tie  # noqa: E402,F401
from nudge_detectors import detect_unanswered_question  # noqa: E402,F401
from nudge_detectors import detect_inferred_occasion  # noqa: E402,F401
from nudge_detectors import detect_dropped_plan  # noqa: E402,F401
from nudge_detectors import detect_unreciprocated_goodnews  # noqa: E402,F401
from nudge_detectors import _is_rhetorical_question  # noqa: E402,F401
from nudge_detectors import _PLAN_CUE_PHRASES  # noqa: E402,F401
from nudge_detectors import _unreciprocated_good_news  # noqa: E402,F401
from nudge_detectors import _next_occurrence  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Stage 0: exclusion gate
# ---------------------------------------------------------------------------

def _direct_chat_counterparties(con):
    """(chat_id, handle_id, contact_id, handle_identifier) for every 1:1 chat's
    counterparty. handle_identifier lets the gate recognize the owner's own
    number and automated/shortcode senders."""
    return con.execute(
        """
        SELECT cp.chat_id, cp.handle_id, h.contact_id, h.handle_identifier
        FROM chat_participants cp
        JOIN chats ch ON ch.id = cp.chat_id AND ch.is_group = 0
        JOIN handles h ON h.id = cp.handle_id
        """
    ).fetchall()


def _is_owner_counterparty(contact_id, handle_identifier, cfg):
    """True when a 1:1 counterparty IS the account owner -- a self-chat / the
    owner's own second number. The schema stores no owner identity (see
    member_key: 'me' is only is_from_me, and outbound rows carry handle_id=0), so
    this is an explicit owner allowlist the owner configures once."""
    if contact_id is not None and contact_id in cfg["owner_contact_ids"]:
        return True
    return bool(handle_identifier) and handle_identifier in cfg["owner_handles"]


def _is_automated_sender(handle_identifier, cfg):
    """True for an automated / non-human sender: an Apple business handle
    (urn:biz:*) or a short numeric SMS shortcode (e.g. 22022) -- a marketing /
    survey / 2FA sender, never a person the user owes a reply. A shortcode is a
    BARE (no country-code '+') all-digit sender at most shortcode_max_digits long;
    a real phone number carries a '+' or is 10-15 digits, so it is never a
    shortcode."""
    ident = (handle_identifier or "").strip()
    if not ident:
        return False
    if ident.lower().startswith("urn:biz:"):
        return True
    return (not ident.startswith("+") and ident.isdigit()
            and len(ident) <= cfg["shortcode_max_digits"])


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
    for chat_id, handle_id, contact_id, handle_identifier in \
            _direct_chat_counterparties(con):
        if _is_excluded(handle_id, contact_id, excluded_handles, excluded_contacts):
            continue
        # Identity gates first (cheap, no SQL): the owner's own self-chat and
        # automated/shortcode senders are never a loose end regardless of volume
        # or valence.
        if _is_owner_counterparty(contact_id, handle_identifier, cfg):
            reason = _REASON_SELF
        elif _is_automated_sender(handle_identifier, cfg):
            reason = _REASON_AUTOMATED
        elif _outbound_count(con, chat_id) == 0:
            reason = _REASON_ZERO_OUTBOUND
        else:
            reason = None
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
    one (anchor/urgency/evidence move as the corpus grows). A dismissed thread
    for the same (kind, chat_id) is left untouched -- a detector never resurrects
    a dismissal -- but a 'resolved' (auto-closed) thread that is detected again is
    RE-OPENED, since the underlying loose end has recurred. Returns the row id,
    or None when skipped."""
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
            "UPDATE loose_threads SET status = 'open', handle_id = ?, "
            "anchor_message_id = ?, anchor_date = ?, due_by = ?, urgency = ?, "
            "evidence_json = ?, source = ?, detected_at = ? WHERE id = ?",
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


def _build_cfg(con, model=None):
    """`model` is the run's resolved narration model (--model > NUDGES_MODEL).
    It reaches cfg because dropped_plan verifies with it and stamps it into
    evidence_json.verified_by, so --model has to scope detection too or the
    guard would clear one model while the detector spent another. None keeps
    the config default, for the callers (tests, detectors) that do not set it."""
    return {
        "now": _corpus_now(con),
        "valence_exclude": NUDGE_VALENCE_EXCLUDE,
        "faded_drop": NUDGE_FADED_INIT_DROP,
        "faded_min_sessions": NUDGE_FADED_INIT_MIN_SESSIONS,
        "faded_window_days": NUDGE_FADED_INIT_WINDOW_DAYS,
        "kind_weights": NUDGE_KIND_WEIGHTS,
        "halflife_days": max(NUDGE_URGENCY_HALFLIFE_DAYS, 1e-9),
        # Stage-0 identity gates: the owner's own contact/handles (self-chat) and
        # the shortcode-length ceiling for automated senders.
        "owner_contact_ids": parse_consent_ids(NUDGE_OWNER_CONTACT_IDS),
        "owner_handles": frozenset(
            t for t in re.split(r"[,\s]+", (NUDGE_OWNER_HANDLES or "").strip()) if t),
        "shortcode_max_digits": NUDGE_SHORTCODE_MAX_DIGITS,
        # --- detector: dormant_tie ---
        "dormant_top_frac": NUDGE_DORMANT_TOP_FRAC,
        "dormant_min_messages": NUDGE_DORMANT_MIN_MESSAGES,
        "dormant_min_gap_days": NUDGE_DORMANT_MIN_GAP_DAYS,
        # --- end detector: dormant_tie ---
        # --- detector: unanswered_question ---
        "question_min_chars": NUDGE_QUESTION_MIN_CHARS,
        "question_grace_hours": NUDGE_QUESTION_GRACE_HOURS,
        "question_include_answered_late": NUDGE_QUESTION_INCLUDE_ANSWERED_LATE,
        "question_max_age_days": NUDGE_QUESTION_MAX_AGE_DAYS,
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
        "nudges_model": model or NUDGES_MODEL,
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
    n_resolved = 0
    if not dry_run:
        # A nondeterministic detector that ran degraded this pass (e.g. the
        # model was unreachable for dropped_plan) is exempted from absence-based
        # auto-resolve -- otherwise a transient verify failure silently closes a
        # still-open thread.
        degraded_kinds = {k for k in NONDETERMINISTIC_KINDS
                          if cfg.get(f"{k}_degraded")}
        n_resolved = _resolve_stale_threads(
            con, thread_ids, event_now, skip_kinds=degraded_kinds)
        con.commit()
    return {"auto_excluded": n_auto, "candidates": n_candidates,
            "suppressed": n_suppressed, "written": n_written,
            "resolved": n_resolved, "thread_ids": thread_ids}


def _resolve_stale_threads(con, written_ids, now, skip_kinds=None):
    """Reconciliation: auto-close every OPEN thread this run did NOT re-detect --
    the loose end resolved (question answered, tie reconnected, occasion passed).
    Sets status='resolved' (distinct from a user 'dismissed'; hidden from both the
    open and the 'already dealt with' digests). Detectors run corpus-wide, so a
    thread absent from this run's candidates is genuinely no longer a loose end --
    EXCEPT for a kind in skip_kinds, whose (nondeterministic) detector ran
    degraded this pass and whose absence therefore proves nothing.
    Returns the number closed. Does NOT commit -- the caller owns the transaction."""
    skip_kinds = sorted(skip_kinds or ())
    keep_ph = ",".join("?" * len(written_ids))
    keep = f" AND id NOT IN ({keep_ph})" if written_ids else ""
    skip_ph = ",".join("?" * len(skip_kinds))
    skip = f" AND kind NOT IN ({skip_ph})" if skip_kinds else ""
    cur = con.execute(
        f"UPDATE loose_threads SET status = 'resolved', dismissed_at = ? "
        f"WHERE status = 'open'{keep}{skip}",
        [now, *written_ids, *skip_kinds],
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Narration (the only LLM stage; guarded by loose_ends_model)
# ---------------------------------------------------------------------------

_NARRATE_SYSTEM = (
    "You write a single short, warm one-line reminder nudging the user to "
    "reconnect with someone they have drifted from. No preamble, no quotes, "
    "one sentence, under 25 words."
)


_LLM_COLUMNS_SET = ("nudge_text IS NOT NULL OR llm_verdict IS NOT NULL "
                    "OR llm_reason IS NOT NULL")


def _guard_llm_model(con, model, record=True):
    """Guard ONLY the LLM-derived columns (nudge_text / llm_verdict /
    llm_reason). The SQL-detected rows are model-free and are never touched.

    REFUSES a model change while those columns hold anything -- the shared
    db.check_model_guard shape, and like annotate's only when there is something
    to protect. It used to CLEAR them and let the run regenerate; that was wrong
    because `--limit N` re-narrates N threads while the clear wiped all of them,
    and because it was unpreviewable (neither --dry-run nor --no-llm reaches
    narration). See docs/DECISIONS.md, "two guards that destroyed what they
    existed to protect".

    `record=False` runs the REFUSAL half only, for main()'s early call BEFORE
    detection: dropped_plan spends NUDGES_MODEL calls during DETECTION and
    stamps the model into evidence_json.verified_by, so a run that will be
    refused must be refused before it pays for that. Recording stays with the
    narration pass, so a run that never narrates leaves the key untouched."""
    check_model_guard(
        con, key="loose_ends_model", model=model, table="loose_threads",
        predicate=_LLM_COLUMNS_SET, record=record,
        message=(
            "loose_threads holds {n} rows whose LLM columns were produced "
            "by {prev}, but the model is now {model}. Mixing models in one "
            "digest makes the narration inconsistent -- rerun with "
            "--model {prev}, set NUDGES_MODEL={prev}, or clear the columns "
            "to switch:\n"
            "  UPDATE loose_threads SET nudge_text = NULL, "
            "llm_verdict = NULL, llm_reason = NULL;"
        ),
    )


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

def _build_parser():
    """The CLI's flag surface, extracted from main() so it is greppable and
    testable on its own."""
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
    return parser


def main(argv=None):
    parser = _build_parser()
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
        # Before detection, not just before narration: dropped_plan is the one
        # model-backed detector, and it spends NUDGES_MODEL LLM calls and
        # stamps evidence_json.verified_by during DETECTION -- so a run that is
        # going to be refused has to be refused before it pays for that. Same
        # ordering conflict_detect's guard needed, for the same reason.
        #
        # Refusal only; the narration pass owns the recording. Skipped under
        # --dry-run (no writes, and its contract is to neither guard nor
        # record), and under --no-llm UNLESS dropped_plan is actually enabled:
        # with it off -- the default -- that path makes zero LLM calls and can
        # touch no guarded column, and CLAUDE.md's own rule is that a guard
        # must not refuse a run it cannot affect. (--acknowledge-reconnect
        # returns above this: a pure toggle that never detects.)
        if not args.dry_run and (not args.no_llm or NUDGE_DROPPED_PLAN_ENABLED):
            _guard_llm_model(con, model, record=False)
        # `model`, not the config default: --model scopes the WHOLE run, so the
        # value the guard just cleared is the one dropped_plan verifies with
        # and stamps. Passing NUDGES_MODEL here instead would guard one model
        # and spend another.
        cfg = _build_cfg(con, model)
        event_now = _utcnow_iso()
        summary = detect_all(con, cfg, event_now, dry_run=args.dry_run)
        print(f"Loose threads: {summary['candidates']} candidates, "
              f"{summary['suppressed']} suppressed, {summary['written']} written, "
              f"{summary.get('resolved', 0)} auto-resolved "
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
