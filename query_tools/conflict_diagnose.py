"""diagnose_conflict (RES-46 / C2): the deterministic scorecard for one
conflict episode, plus the softened DV-suppression floor around its one
interpretive section.

Split out of query_tools/conflict.py, which keeps the read tools
(find_conflicts / get_conflict) and re-exports everything here, so
`query_tools.conflict.diagnose_conflict` keeps resolving.

The scorecard composes DETERMINISTIC pieces ONLY -- lexicon I-vs-you / we-talk
ratios, LSM, session initiator/ender roles, reply-latency + volume collapse
read off C1's stored peak_signal survival stats -- and carries NO aggressor /
who-won verdict. The one interpretive-ish section (which side withdrew vs
pursued, a NON-BLAME demand-withdraw read) is the "role assignment" the DV
floor withholds.
"""

import contextlib

from ._shared import _connect
# LSM primitives are imported, never reimplemented (F1/analytics_language is the
# single home): _weighted_rates -> word-count-weighted category rates over
# lexicon rows, _lsm_score -> the me-vs-them style-matching score.
from .analytics_language import _lsm_score, _weighted_rates
# Bottom-imported by .conflict, so its core names are already bound here.
from .conflict import _ANALYSIS_DISCLAIMER, _has_conflict_tables, _signals_json
from .conflict_dv import _dv_screen

#
# The scorecard composes DETERMINISTIC pieces ONLY -- lexicon I-vs-you /
# we-talk ratios, LSM, session initiator/ender roles, reply-latency + volume
# collapse read off C1's stored peak_signal survival stats -- and carries NO
# aggressor / who-won verdict. The one interpretive-ish section (which side
# withdrew vs pursued, a NON-BLAME demand-withdraw read) is the "role
# assignment" the DV floor withholds.


# --- Deterministic scorecard sections ---------------------------------------

def _side_ratios(rows):
    """I-vs-you & we-talk ratios for one side (F1 contract: SUM raw counts then
    form the rate against summed n_words; COALESCE the nullable pronoun-split
    columns). Returns None when the side has no words."""
    n_msgs = sum(r["n_messages"] for r in rows)
    n_words = sum(r["n_words"] for r in rows)
    if not n_words or not n_msgs:
        return None
    i_c = sum((r["i_count"] or 0) for r in rows)
    you_c = sum((r["you_count"] or 0) for r in rows)
    we_c = sum(r["we_count"] for r in rows)
    return {
        "n_messages": n_msgs,
        "n_words": n_words,
        "i_rate": round(i_c / n_words, 4),
        "you_rate": round(you_c / n_words, 4),
        "i_to_you_ratio": round(i_c / you_c, 3) if you_c else None,
        "we_per_100_msgs": round(100 * we_c / n_msgs, 2),
    }


def _lexicon_section(con, chat_id, first_date, last_date):
    """Per-side I-vs-you & we-talk ratios + the LSM style-matching score over
    the window's months (person_period_lexicon, chat-scoped). Deterministic;
    'available' False when the lexicon has not been built for this window."""
    lo, hi = first_date[:7], last_date[:7]
    rows = con.execute(
        "SELECT * FROM person_period_lexicon WHERE chat_id = ? "
        "AND month >= ? AND month <= ?", (chat_id, lo, hi)).fetchall()
    me_rows = [r for r in rows if r["handle_id"] is None]
    them_rows = [r for r in rows if r["handle_id"] is not None]
    me, them = _side_ratios(me_rows), _side_ratios(them_rows)
    section = {"me": me, "them": them}
    if me is None or them is None:
        section["available"] = False
        section["note"] = ("lexicon not built for this window "
                           "(person_period_lexicon is built by ingest)")
    else:
        section["available"] = True
    # LSM reuses the F1 primitives (never reimplemented).
    rates_me = _weighted_rates(me_rows) if me_rows else None
    rates_them = _weighted_rates(them_rows) if them_rows else None
    if rates_me is not None and rates_them is not None:
        section["lsm"] = round(_lsm_score(rates_me, rates_them)[0], 4)
    else:
        section["lsm"] = None
    return section


def _session_roles(con, start_session_id, end_session_id):
    """Deterministic session roles: who opened the episode's first session
    (initiator) and who had the last word in its final session (ender). Raw
    is-me facts; NOT a verdict. Always returned, even under a DV hit."""
    out = {"opener": None, "last_word": None}
    if start_session_id is not None:
        s = con.execute("SELECT initiator_is_me FROM sessions WHERE id = ?",
                        (start_session_id,)).fetchone()
        if s is not None:
            out["opener"] = "me" if s["initiator_is_me"] else "them"
    if end_session_id is not None:
        e = con.execute("SELECT ender_is_me FROM sessions WHERE id = ?",
                        (end_session_id,)).fetchone()
        if e is not None:
            out["last_word"] = "me" if e["ender_is_me"] else "them"
    return out


def _interaction_pattern(con, peak, session_roles):
    """The demand-withdraw READ -- who withdrew vs who kept reaching out --
    derived DETERMINISTICALLY from C1's peak reply-gap (the actor of the
    message left hanging pursued; the other withdrew), falling back to who had
    the last word. A NON-BLAME interactional description with NO aggressor /
    who-won verdict -- and the one section the DV floor withholds (A10)."""
    pursuer = None
    peak_mid = peak.get("peak_gap_message_id")
    if peak_mid is not None:
        m = con.execute("SELECT is_from_me FROM messages WHERE id = ?",
                        (peak_mid,)).fetchone()
        if m is not None:
            pursuer = "me" if m["is_from_me"] else "them"
    if pursuer is None:
        pursuer = session_roles.get("last_word")
    withdrawer = {"me": "them", "them": "me"}.get(pursuer)
    return {
        "pursuer": pursuer,
        "withdrawer": withdrawer,
        "basis": ("deterministic demand-withdraw read from the peak reply gap; "
                  "a mechanical pattern, NOT a judgment of fault"),
    }


def diagnose_conflict(conflict_id) -> dict:
    """A deterministic scorecard for one conflict episode (by conflict_id from
    find_conflicts): lexicon I-vs-you & we-talk ratios and LSM style-matching
    per side, the session opener/last-word roles, reply-latency and
    volume-collapse read off the stored deterministic evidence, and a NON-BLAME
    demand-withdraw read (who withdrew vs pursued). It carries NO aggressor or
    who-won verdict. A code-enforced DV safety screen (Master Spec A10) scans
    the window's full text: on a high-risk marker it withholds ONLY the
    demand-withdraw read and appends a fixed hotline resource footer, while the
    mechanical scorecard still returns; CONFLICT_DV_SCREEN=off (or chat.py
    --no-dv-screen) restores the read but still appends the footer on a hit.
    The screen result is never persisted."""
    conflict_id = int(conflict_id)
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, start_session_id, "
            "end_session_id, topic, peak_signal_json, corroboration, "
            "off_text_flag FROM conflicts WHERE id = ?", (conflict_id,),
        ).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}
        peak = _signals_json(row["peak_signal_json"])
        signals = peak.get("signals", [])
        result = {
            "conflict_id": row["id"],
            "chat_id": row["chat_id"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "topic": row["topic"],
            "signals": signals,
            "corroboration": row["corroboration"],
            "off_text_flag": bool(row["off_text_flag"]),
            "lexicon": _lexicon_section(
                con, row["chat_id"], row["first_date"], row["last_date"]),
            "session_roles": _session_roles(
                con, row["start_session_id"], row["end_session_id"]),
            # Latency + volume come from C1's stored survival stats
            # (stats.frac_over_threshold); get_response_dynamics is NOT called.
            "latency": {
                "peak_gap_hours": peak.get("peak_gap_hours"),
                "withdrawal_fired": "latency_withdrawal" in signals,
                "source": "peak_signal survival stats (frac_over_threshold), C1",
            },
            "volume": {
                "volume_ratio": peak.get("volume_ratio"),
                "collapse_fired": "volume_collapse" in signals,
            },
            "note": ("deterministic scorecard -- no aggressor/who-won verdict; "
                     "labels elsewhere are eval-gated bands, not certainties"),
        }
        pattern = _interaction_pattern(con, peak, result["session_roles"])
        screen = _dv_screen(con, conflict_id)

    result["dv_screen"] = screen["status"]
    if screen["suppress_role"]:
        result["interaction_pattern"] = {
            "withheld": True,
            "reason": ("high-risk safety markers present -- role/aggressor read "
                       "withheld (DV floor); the mechanical scorecard still "
                       "applies"),
        }
    else:
        result["interaction_pattern"] = pattern
    if screen["footer"]:
        result["resource_footer"] = screen["footer"]
    # Code-printed humility disclaimer (out-of-band via chat.run_turn); distinct
    # from the in-payload resource_footer, which the model narrates.
    result["_epilogue"] = _ANALYSIS_DISCLAIMER
    return result


