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


def run_exclusion_gate(con, cfg, now):
    """Stage 0, always first. Load stored exclusions, then auto-mute any 1:1
    counterparty whose most-recent llm-annotated exchange is negative
    (<= NUDGE_VALENCE_EXCLUDE) or who has never received an outbound message.
    Persists new auto-mutes to nudge_exclusions (dedup-safe) AND returns the
    in-memory (handles, contacts) sets, so no candidate is ever generated for
    them. Returns (excluded_handles, excluded_contacts, n_added)."""
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
                          reason=reason, now=now):
            n_added += 1
        if handle_id is not None:
            excluded_handles.add(handle_id)
        if contact_id is not None:
            excluded_contacts.add(contact_id)
    con.commit()
    return excluded_handles, excluded_contacts, n_added


# ---------------------------------------------------------------------------
# Stage 3: urgency + row persistence
# ---------------------------------------------------------------------------

def _score_urgency(kind, anchor_date, ref_now, cfg):
    """urgency = kind_weight * exp(-age_days/halflife * ln 2); age from ref_now
    (the corpus's latest message). A missing anchor scores the bare weight."""
    weight = cfg["kind_weights"].get(kind, _DEFAULT_KIND_WEIGHT)
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
    }


def detect_all(con, cfg, event_now, dry_run=False):
    """The shared driver loop: Stage 0 gate, then every detector, with the
    exclusion gate and urgency scoring applied OUTSIDE the detectors. Returns a
    summary dict (and the written thread ids)."""
    excluded_handles, excluded_contacts, n_auto = run_exclusion_gate(con, cfg, event_now)
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
            urgency = _score_urgency(cand.kind, cand.anchor_date, cfg["now"], cfg)
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
                             "calling the model")
    args = parser.parse_args(argv)

    model = args.model or NUDGES_MODEL
    con = _connect()
    try:
        # Self-heal the tables (standalone runs / older DBs). Idempotent.
        schema._create_nudges_tables(con)
        con.commit()
        cfg = _build_cfg(con)
        event_now = _utcnow_iso()
        summary = detect_all(con, cfg, event_now, dry_run=args.dry_run)
        print(f"Loose threads: {summary['candidates']} candidates, "
              f"{summary['suppressed']} suppressed, {summary['written']} written "
              f"({summary['auto_excluded']} new auto-exclusions)"
              f"{' [--dry-run: nothing written]' if args.dry_run else ''}")
        if args.dry_run:
            return 0
        if not args.no_llm:
            narrate_threads(con, model, NUDGES_THINK, NUDGES_TIMEOUT, limit=args.limit)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    run_cli(main)
