"""draft_apology scaffold (C4 / RES-53, Lewicki-6 -- NEVER ghostwrites).

Split out of conflict.py: the generative repair surface. Under the code-
enforced DV floor it withholds the ENTIRE scaffold on a role-suppress hit
(drafting an apology is itself the risky output). Reuses the single-source
_dv_screen from conflict_dv; reads conflict read-side helpers from .conflict.
"""

import contextlib

from ._shared import _connect
from .conflict import EXCERPT_CAP, _has_conflict_tables
from .conflict_dv import _dv_screen


# --- feature: conflict (C4 apology) ---
# ===========================================================================
# C4 (RES-53): draft_apology scaffold (Lewicki-6, NEVER ghostwrites).
#
# draft_apology(conflict_id) SCAFFOLDS a repair message; it does not write one.
# It returns the ranked Lewicki-6 apology checklist (Lewicki, Polin & Lount
# 2016, "An Exploration of the Structure of Effective Apologies" -- the six
# components ordered by MEASURED effectiveness: acknowledging responsibility
# first, offering repair second, requesting forgiveness last and flagged
# skippable), the user's REAL grievance excerpts pulled VERBATIM from the
# episode's labeled turns (never invented), and -- only where the `repair`
# escalation label has cleared its Cohen's-kappa eval gate
# (eval_gate_passed(con, 'conflict_labels', 'repair')) -- the ignored repair
# attempts the other person made. When `repair` is NOT eval-cleared that
# component is OMITTED and returned as {"status": "unavailable", ...} rather
# than guessed from raw text; the checklist + excerpts still return.
#
# It emits NO finished prose (emits_prose is always False). Any finished draft
# must sit behind a SECOND explicit confirmation and be labeled "draft template
# -- edit before sending" (guidance carried in the payload note).
#
# DV floor (Master Spec A10, reusing C2's _dv_screen / _DV_RESOURCE_TEXT):
# drafting an apology is ITSELF the risky output -- it can pressure a victim
# into apologizing to an abuser -- so UNLIKE diagnose_conflict there is NO
# partial scaffold. A 'role-suppress' screen (marker hit, screening on)
# WITHHOLDS the entire scaffold and returns ONLY the code-emitted resource
# pointer. A 'bypassed-with-footer' screen (screening off) returns the full
# scaffold with the resource footer appended. The pointer is never fully
# removed; the screen result is never persisted.
# ===========================================================================

# The Lewicki-6 checklist, ranked by measured effectiveness (rank 1 = highest).
# A module constant, returned as-is and never mutated; grounding excerpts (not
# generated prose) are attached separately per call.
_LEWICKI_CHECKLIST = (
    {"rank": 1, "component": "acknowledge_responsibility",
     "prompt": ("Name what you did and own it plainly -- no 'if', no 'but', no "
                "shifting it onto them."),
     "skippable": False,
     "note": "most effective component (Lewicki, Polin & Lount 2016)"},
    {"rank": 2, "component": "offer_of_repair",
     "prompt": ("Say concretely how you'll make it right or keep it from "
                "happening again."),
     "skippable": False,
     "note": "second most effective"},
    {"rank": 3, "component": "expression_of_regret",
     "prompt": "Say you're sorry and that you regret the hurt it caused.",
     "skippable": False},
    {"rank": 4, "component": "explanation",
     "prompt": ("Briefly explain what happened -- an honest account, NOT an "
                "excuse."),
     "skippable": False},
    {"rank": 5, "component": "declaration_of_repentance",
     "prompt": "State that you don't intend to repeat it.",
     "skippable": False},
    {"rank": 6, "component": "request_for_forgiveness",
     "prompt": ("Optionally ask for forgiveness -- never demand or expect it, "
                "and skip it if pressing for it would burden them."),
     "skippable": True,
     "note": "least effective; safe to omit"},
)

# Escalation labels that count as grievance turns (the horsemen); NOT 'repair'
# (its own eval-gated component) and NOT 'abstain'.
_GRIEVANCE_CLASSES = ("criticism", "contempt", "defensiveness", "stonewalling")
# Per-component excerpt cap (kept well under TOOL_RESULT_MAX_CHARS alongside the
# 200-char EXCERPT_CAP re-cap on each stored excerpt).
_APOLOGY_MAX_EXCERPTS = 15


def _grievance_excerpts(con, conflict_id):
    """The episode's REAL grievance turns: VERBATIM conflict_signals excerpts
    labeled with one of the horsemen classes (never invented, never guessed),
    each re-capped at EXCERPT_CAP and tagged with the actor. Ordered as stored;
    capped at _APOLOGY_MAX_EXCERPTS."""
    ph = ",".join("?" * len(_GRIEVANCE_CLASSES))
    rows = con.execute(
        f"SELECT message_id, actor_is_me, label_class, excerpt "
        f"FROM conflict_signals WHERE conflict_id = ? "
        f"AND label_class IN ({ph}) AND excerpt IS NOT NULL "
        f"ORDER BY id LIMIT ?",
        (conflict_id, *_GRIEVANCE_CLASSES, _APOLOGY_MAX_EXCERPTS)).fetchall()
    return [{
        "message_id": r["message_id"],
        "actor": "me" if r["actor_is_me"] else "them",
        "label_class": r["label_class"],
        "excerpt": (r["excerpt"] or "")[:EXCERPT_CAP],
    } for r in rows]


def _ignored_repair_attempts(con, conflict_id):
    """The repair BIDS the OTHER person made during the episode (label_class
    'repair', actor them): VERBATIM conflict_signals excerpts the user can
    acknowledge, never guessed. Called ONLY when the `repair` label class is
    eval-cleared -- its rows only exist at all when that class cleared the gate
    at label-write time (a suppressed class writes zero rows)."""
    rows = con.execute(
        "SELECT message_id, actor_is_me, excerpt FROM conflict_signals "
        "WHERE conflict_id = ? AND label_class = 'repair' AND actor_is_me = 0 "
        "AND excerpt IS NOT NULL ORDER BY id LIMIT ?",
        (conflict_id, _APOLOGY_MAX_EXCERPTS)).fetchall()
    return [{
        "message_id": r["message_id"],
        "actor": "them",
        "excerpt": (r["excerpt"] or "")[:EXCERPT_CAP],
    } for r in rows]


def draft_apology(conflict_id) -> dict:
    """SCAFFOLD (never ghostwrite) a repair message for one conflict episode (a
    conflict_id from find_conflicts). Returns the ranked Lewicki-6 apology
    checklist (Lewicki, Polin & Lount 2016 -- components ordered by measured
    effectiveness: acknowledging responsibility first, offering repair second,
    requesting forgiveness last and flagged skippable), the user's REAL
    grievance excerpts pulled VERBATIM from the episode (never invented), and --
    only where the `repair` escalation label has cleared its eval gate -- the
    ignored repair attempts the other person made; when `repair` is not
    eval-cleared that component is omitted and marked unavailable rather than
    guessed from raw text. It emits NO finished apology prose (emits_prose is
    False) -- do not ghostwrite the message; walk the user through the checklist
    grounded in their real excerpts.

    DV safety floor (Master Spec A10): the same code-enforced screen
    diagnose_conflict uses runs over the full conflict window. Because drafting
    an apology is ITSELF the risky output (it can pressure a person into
    apologizing to someone who is hurting them), a role-suppress result WITHHOLDS
    the ENTIRE scaffold and returns ONLY the fixed hotline resource pointer --
    there is NO partial scaffold here (unlike diagnose_conflict). With screening
    off (CONFLICT_DV_SCREEN=off or chat.py --no-dv-screen) the scaffold returns
    but STILL carries the appended footer on a marker hit. The screen result is
    never persisted."""
    conflict_id = int(conflict_id)
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, topic, off_text_flag "
            "FROM conflicts WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}

        screen = _dv_screen(con, conflict_id)
        # DV role-suppress: drafting the apology is itself the risk. WITHHOLD the
        # whole scaffold and return ONLY the code-emitted resource pointer -- no
        # partial scaffold, unlike diagnose_conflict's mechanical section.
        if screen["suppress_role"]:
            return {
                "conflict_id": row["id"],
                "chat_id": row["chat_id"],
                "dv_screen": screen["status"],
                "withheld": True,
                "reason": ("high-risk safety markers present -- the apology "
                           "scaffold is withheld entirely; drafting an apology "
                           "can pressure a person into apologizing to someone "
                           "who is hurting them"),
                "resource_footer": screen["footer"],
            }

        # Scaffold path (screen is 'clean' or 'bypassed-with-footer'). Lazy,
        # Ollama-free import (app modules are pulled in lazily by gates).
        from eval.evalkit.gates import eval_gate_passed

        result = {
            "conflict_id": row["id"],
            "chat_id": row["chat_id"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "topic": row["topic"],
            "off_text_flag": bool(row["off_text_flag"]),
            "dv_screen": screen["status"],
            "checklist": list(_LEWICKI_CHECKLIST),
            "grievances": _grievance_excerpts(con, conflict_id),
            "emits_prose": False,
            "note": ("this is a SCAFFOLD, not an apology -- it emits no finished "
                     "prose. Do NOT ghostwrite the message; walk the user "
                     "through the checklist grounded in their real grievance "
                     "excerpts. If they explicitly ask for drafted text AFTER "
                     "seeing this, produce it only on a second explicit "
                     "confirmation and label it 'draft template -- edit before "
                     "sending'."),
        }
        # Repair-attempts component is eval-gated: present only when the `repair`
        # label class has cleared its kappa gate (absent-key => suppressed).
        if eval_gate_passed(con, "conflict_labels", "repair"):
            result["repair_attempts"] = {
                "status": "available",
                "attempts": _ignored_repair_attempts(con, conflict_id),
                "note": ("repair bids the other person made -- acknowledging "
                         "these is part of an effective repair"),
            }
        else:
            result["repair_attempts"] = {
                "status": "unavailable",
                "reason": "label class not eval-cleared",
            }

    # The resource pointer is never fully removed: a 'bypassed-with-footer' hit
    # appends the footer even though role analysis was restored.
    if screen["footer"]:
        result["resource_footer"] = screen["footer"]
    return result
# --- end feature: conflict (C4 apology) ---
