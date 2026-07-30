"""Conflict read tools (RES-46 / C1): find_conflicts (ranked conflict episodes
in scope) and get_conflict (one episode's details + its labeled signal turns).

These read the deterministic finder's output (conflicts / conflict_signals,
written by conflict_detect.py) -- no Ollama, no re-detection. Conflicts exist
only for 1:1 chats by construction; a person/chat scope resolves the same way
as the other analytics tools and is then restricted to direct chats
(_restrict_to_direct_chats -- F3). Escalation labels are eval-gated BANDS
(tentative/likely), never certainties -- surface them as such.

conflicts.first_date/last_date are unlabeled-UTC ISO strings: plain-string
compares only, never SQL date functions in WHERE (see CLAUDE.md gotchas)."""

import contextlib
import json

import config

from ._shared import (
    _connect,
    _date_clauses,
    _envelope,
    _resolve_chat_scope,
    _restrict_to_direct_chats,
)
# LSM primitives are imported, never reimplemented (F1/analytics_language is the
# single home): _weighted_rates -> word-count-weighted category rates over
# lexicon rows, _lsm_score -> the me-vs-them style-matching score.
from .analytics_language import _lsm_score, _weighted_rates
from .conflict_dv import _dv_screen, _window_message_texts
# set_dv_screen_bypass re-exported so query_tools.conflict.set_dv_screen_bypass
# (chat.py --no-dv-screen, tests) keeps resolving after the DV split.
from .conflict_dv import set_dv_screen_bypass  # noqa: F401

# Result caps (contract-fixed; this tool's own, like analytics_social's
# eligibility gates). Kept well under TOOL_RESULT_MAX_CHARS: <=15 signal rows,
# each excerpt <=200 chars, is a few KB.
MAX_CONFLICT_ROWS = 25       # find_conflicts page size
MAX_SIGNAL_ROWS = 15         # get_conflict signal rows per page
EXCERPT_CAP = 200            # defensive re-cap on stored excerpts
# find_conflicts rank tie-break: tone-corroborated episodes surface first.
_CORROBORATION_RANK = {"llm": 2, "weak": 1, "none": 0}

# Fixed, code-emitted humility disclaimer for the conflict READ surfaces
# (find_conflicts / get_conflict / diagnose_conflict). Delivered out-of-band via
# the generic `_epilogue` channel (chat.run_turn strips it from the model's view
# and prints it verbatim after the reply -- the wellbeing support_block
# precedent), so it always reaches the user regardless of model compliance.
# Never LLM-authored, never persisted. It replaces a hard who-started boundary:
# the tool surfaces signals, not a fault verdict, and says so.
_ANALYSIS_DISCLAIMER = (
    "Heads up: this is an automated read of text-only signals (reply timing, "
    "message volume, tapbacks). It has limited information -- it can't see calls, "
    "in-person context, or tone -- so it can be wrong and does not determine who "
    "was at fault or how things actually resolved. Treat it as a prompt to "
    "reflect, not a verdict."
)
# Appended to that disclaimer (verbatim, never formatted with counts -- the
# epilogue stays fixed, code-authored text) when some in-scope window has not
# been through per-turn labeling. The counts live in the model-visible payload;
# this is the belt-and-braces copy the USER sees regardless of model compliance.
_PARTIAL_LABELING_EPILOGUE = (
    " Note: the escalation pass runs on a time budget (strongest stretches "
    "first) and has not reached every detected stretch yet, so a stretch with "
    "nothing flagged may simply not have been looked at."
)
# The same belt-and-braces copy for the OTHER way a stretch shows nothing: the
# pass did run and produced no shippable label. Its sibling above exists
# precisely because the model may not narrate a payload-only caveat, and this
# case licenses the identical misreading ("I checked those stretches and
# nothing escalatory came up"), so it cannot be left to the model either -- the
# repo's rule is that a framing floor is code, never prompt. Fixed text, never
# formatted with counts: the counts live in the model-visible payload.
_LABELED_EMPTY_EPILOGUE = (
    " Note: where the escalation pass did run and flagged nothing, that is not "
    "a clean bill of health -- it reads message text only, so a stretch carried "
    "by tapbacks, by tone, or by a conversation that moved elsewhere can be "
    "looked at and still come back empty."
)

# --- detected-but-unlabeled (the budgeted label pass) ------------------------
# conflict_detect.py detects every window but LLM-labels only as many as its
# wall-clock budget buys, strongest first; `conflicts.labeled_at` NULL means
# this window's per-turn pass never ran (a rerun extends coverage), non-NULL
# means it ran -- even if every label was neutral or eval-gate-suppressed, so
# zero conflict_signals rows is a legitimate LABELED outcome. Surfacing the
# marker is the same degrade-honestly discipline the llm-vs-distilled
# annotation readers follow: without it `labeled_signal_turns: 0` reads as
# evidence of absence for a window nothing ever looked at.
#
# One transitional state keeps both notes honest on a migrated DB: a window
# labeled by the pre-budget pipeline carries its signal rows but a NULL
# labeled_at (the migration deliberately does not invent a stamp -- see
# schema.py), so labeled=false with labeled_signal_turns > 0 is possible until
# a budgeted run re-reaches it. Those labels are real; the notes say so rather
# than asserting an equivalence the data briefly violates.
_UNLABELED_SCOPE_NOTE = (
    "{n_unlabeled} of {total} detected windows in scope have NOT been through "
    "per-turn escalation labeling yet (it runs on a wall-clock budget, "
    "strongest windows first; rerunning the conflict pipeline extends "
    "coverage). On a row with labeled=false, labeled_signal_turns=0 means NOT "
    "ANALYZED -- not 'analyzed and nothing escalatory found' -- and topic / "
    "reconciliation are unset for the same reason. (A labeled=false row that "
    "still shows labeled_signal_turns > 0 was labeled by an earlier version of "
    "the pipeline that recorded no completion stamp: those labels are real, "
    "and a future budgeted run will refresh them.) The deterministic evidence "
    "(signals, signal_count, corroboration) is complete for every row."
)
_LABELED_NOTE = "labels are eval-gated bands (tentative/likely), not certainties"
# A window CAN be stamped labeled_at having dispatched nothing: the per-turn
# pass reads message TEXT, so a window whose turns are all tapbacks/emoji has
# nothing to classify, and a turn that IS classified can still be dropped by the
# confidence floor or suppressed by the F4 eval gate. "Labeled with zero turns"
# is therefore not the same claim as "we read this and it was calm", and the
# note must not let it be read as one -- the same honesty the labeled/unlabeled
# split exists for, one level in.
_LABELED_EMPTY_NOTE = (
    "this episode has been through per-turn escalation labeling and NO turn "
    "produced a shippable label -- which is not a finding of calm: the pass "
    "reads message text only, so a stretch carried by tapbacks, by tone, or by "
    "a conversation that moved off-channel (off_text_flag) can be looked at and "
    "still yield nothing, and a turn can also be dropped by the confidence "
    "floor or suppressed by the eval gate. Read this one off its deterministic "
    "evidence (peak_signal, corroboration, off_text_flag), which is complete."
)
_LABELED_EMPTY_SCOPE_NOTE = (
    "{n_empty} of {total} detected windows in scope WERE labeled but produced "
    "zero signal turns (labeled=true, labeled_signal_turns=0). That is not "
    "'analyzed and calm': labeling reads message text only, so a stretch "
    "carried by tapbacks, by tone, or by a conversation that moved off-channel "
    "yields nothing to label, and a turn can also fall under the confidence "
    "floor or be suppressed by the eval gate. Those rows' deterministic "
    "evidence (signals, signal_count, corroboration) is what found them."
)
# Appended to whichever note is emitted on a DB whose `conflicts` table has no
# labeled_at column at all (see _labeled_at_select): there `labeled` is INFERRED
# from stored signal rows, so the two states the column exists to separate are
# only partly separable.
_PRE_MIGRATION_NOTE = (
    "(This database predates the labeling-completion stamp, so `labeled` is "
    "inferred from whether the window has stored signal turns: a window the old "
    "pipeline looked at and found nothing in is indistinguishable here from one "
    "it never reached. Rerunning the conflict pipeline stamps them properly.)"
)
_UNLABELED_NOTE = (
    "this episode has NOT been through per-turn escalation labeling yet (it "
    "runs on a wall-clock budget, strongest windows first; rerunning the "
    "conflict pipeline extends coverage), so zero signal turns here means NOT "
    "ANALYZED, not 'analyzed and nothing escalatory found' -- and topic is "
    "unset for the same reason. If signal turns ARE present despite this, they "
    "were written by an earlier version of the pipeline that recorded no "
    "completion stamp: real labels, refreshed by a future budgeted run. Its "
    "deterministic evidence (peak_signal, corroboration, off_text_flag) is "
    "complete either way."
)


def _has_conflict_tables(con):
    """True only when both conflict tables exist -- a DB built before this
    feature has neither, and the read tools must return empty, not raise."""
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


def _predates_labeling_stamps(con):
    """True when `conflicts` has no labeled_at column at all -- a DB neither
    re-ingested nor re-detected since the budgeted tier landed. Returning rows
    beats raising `no such column`, which is the same contract
    _has_conflict_tables keeps one migration earlier."""
    return "labeled_at" not in {r[1] for r in con.execute("PRAGMA table_info(conflicts)")}


def _labeled_at_select(con, alias):
    """The SELECT fragment yielding a `labeled_at` value, degrading on a
    `conflicts` table that predates the column.

    On that path the value is derived from EVIDENCE, not asserted: 'pre-budget'
    only where the window actually has conflict_signals rows, NULL otherwise. A
    flat non-NULL literal was wrong in the one direction that matters -- an old
    `--no-llm` run wrote windows nothing ever labeled, and marking them
    `labeled` made get_conflict report "we looked and found nothing" about a
    window nothing looked at, exactly the misreading the labeled/unlabeled split
    exists to prevent. The inference is one-way-safe: signals cannot exist
    without a labeling pass, while their absence stays ambiguous (a pass that
    found nothing looks like no pass) -- which is what _PRE_MIGRATION_NOTE says
    out loud rather than papering over.

    `alias` is the caller's table alias for `conflicts` and is REQUIRED: the
    fallback correlates a subquery on `<alias>.id`, and an unqualified `id`
    there would resolve to the subquery's own conflict_signals.id. The fragment
    stays a constant -- no user input is ever interpolated."""
    if not _predates_labeling_stamps(con):
        return f"{alias}.labeled_at"
    return ("CASE WHEN EXISTS (SELECT 1 FROM conflict_signals ps "
            f"WHERE ps.conflict_id = {alias}.id) THEN 'pre-budget' END "
            "AS labeled_at")


def _signals_json(peak_signal_json):
    try:
        data = json.loads(peak_signal_json) if peak_signal_json else {}
    except (TypeError, ValueError):
        data = {}
    return data


def _conflict_summary(row):
    """Compact conflict row for find_conflicts (no per-signal excerpts).

    `labeled_signal_turns` (non-abstain conflict_signals rows) is folded into the
    find_conflicts SELECT as a GROUP BY aggregate, so this is a pure row->dict
    shaper with no per-row query (was an N+1 COUNT).

    `labeled` mirrors conflicts.labeled_at (see the notes above): it separates
    "we never looked" (labeled false) from "the pass ran here" (labeled true).
    It does NOT license reading labeled true + 0 turns as "we looked and found
    nothing" -- labeling reads message text only, so a stretch carried by
    tapbacks, by tone, or moved off-channel can be looked at and yield nothing;
    _LABELED_EMPTY_NOTE / _LABELED_EMPTY_SCOPE_NOTE are what say so to the
    caller. This is the ONE place a find_conflicts row dict is built -- keep it
    that way."""
    peak = _signals_json(row["peak_signal_json"])
    n_signal_rows = row["labeled_signal_turns"]
    # A make-up window is one the LLM topic pass labeled 'repair'. Kept explicit
    # so the ranking can de-prioritize reconciliations (a make-up is not the
    # "tensest" stretch) and the model can narrate the arc faithfully. NOTE:
    # peak's `valence_boost` is NOT a make-up signal -- it flags a corroborated
    # valence DROP (tension), so it is surfaced for context but never drives
    # `reconciliation`.
    valence_boost = bool(peak.get("valence_boost"))
    reconciliation = row["topic"] == "repair"
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "topic": row["topic"],
        "signals": peak.get("signals", []),
        "signal_count": peak.get("signal_count", 0),
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "valence_boost": valence_boost,
        "reconciliation": reconciliation,
        "labeled": row["labeled_at"] is not None,
        "labeled_signal_turns": n_signal_rows,
        "peak_gap_hours": peak.get("peak_gap_hours"),
    }


def _empty_conflicts_envelope(order):
    """find_conflicts' no-rows result. The tool's docstring and its JSON schema
    promise n_corroborated and n_labeled on EVERY envelope, so the empty-scope
    early returns carry them too -- a caller that reads env["n_labeled"] must
    not have to special-case "the scope resolved to nothing"."""
    env = _envelope([], 0, order)
    env["n_corroborated"] = 0
    env["n_labeled"] = 0
    return env


def find_conflicts(chat_id=None, handle_id=None, contact_id=None,
                   date_from=None, date_to=None, limit=10):
    """Detected conflict episodes in scope, ranked so the best-substantiated
    surface first: tone-corroborated episodes (corroboration llm > weak > none),
    then deterministic signal strength, then recency. Each is a tense stretch
    found from deterministic
    signals WITH RECEIPTS (reply-latency withdrawal, one-sided sessions,
    dislike-tapback runs, volume collapse); `signals` names which fired and
    `signal_count` how many. 1:1 chats only. handle_id/contact_id scope to that
    person's direct chats; date bounds filter on the episode start. corroboration
    is 'none' unless source='llm' annotations backed the tone drop. Returns an
    envelope {rows, total_matching, n_corroborated, n_labeled, returned, order,
    truncated}; total_matching counts EVERY detected stretch in scope (mostly
    uncorroborated lulls -- not a count of real fights), while n_corroborated is
    the substantiated subset (weak+llm) -- prefer it as a headline count.
    n_labeled counts the ones whose per-turn escalation labeling has actually
    run: labeling is wall-clock-budgeted, so a row with labeled=false has
    labeled_signal_turns=0 because nothing looked, NOT because nothing was
    found. A labeled=true row with labeled_signal_turns=0 is not a finding of
    calm either -- labeling reads message text only. A `labeling_note` spells
    out whichever of the two is present. Expand one with
    get_conflict(conflict_id)."""
    limit = max(1, min(int(limit), MAX_CONFLICT_ROWS))
    order = ("corroboration-desc, non-reconciliation-first, signal_count-desc, "
             "most-recent-first")
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return _empty_conflicts_envelope(order)
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return _empty_conflicts_envelope(order)
        if chat_ids is not None:
            chat_ids = _restrict_to_direct_chats(con, chat_ids)
            if not chat_ids:
                return _empty_conflicts_envelope(order)

        clauses, params = [], []
        if chat_ids is not None:
            clauses.append(f"c.chat_id IN ({','.join('?' * len(chat_ids))})")
            params.extend(chat_ids)
        _date_clauses(clauses, params, date_from, date_to, column="c.first_date")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # Fold the per-conflict labeled-signal count into one query (was a per-row
        # COUNT -> N+1). LEFT JOIN keeps conflicts with zero signals; the CASE sums
        # only non-abstain rows (there is no index on label_class). No SQL LIMIT --
        # total_matching/n_corroborated below must count the full in-scope
        # population; the Python sort + [:limit] slice do the ranking.
        rows = con.execute(
            "SELECT c.id, c.chat_id, c.first_date, c.last_date, c.topic, "
            "c.peak_signal_json, c.corroboration, c.off_text_flag, "
            f"{_labeled_at_select(con, 'c')}, "
            "COALESCE(SUM(CASE WHEN s.label_class != 'abstain' THEN 1 ELSE 0 END), 0) "
            "AS labeled_signal_turns "
            "FROM conflicts c "
            "LEFT JOIN conflict_signals s ON s.conflict_id = c.id"
            f"{where} GROUP BY c.id", params,
        ).fetchall()
        summaries = [_conflict_summary(r) for r in rows]
        pre_migration = _predates_labeling_stamps(con)

    # Best-substantiated first: tone-corroborated episodes (llm > weak > none)
    # outrank raw signal count, so the ~5% with a corroborated valence drop
    # surface ahead of the majority of uncorroborated lulls. WITHIN a
    # corroboration tier, genuine tension outranks a reconciliation/make-up
    # window (`not reconciliation` sorts first under reverse=True), so an apology
    # exchange is never surfaced as the "tensest" stretch. Then most distinct
    # deterministic signals, then most recent. (Detection + total_matching are
    # unchanged -- this only reorders which episodes fill the top `limit`.)
    summaries.sort(
        key=lambda s: (_CORROBORATION_RANK.get(s["corroboration"], 0),
                       not s["reconciliation"],
                       s["signal_count"], s["last_date"] or ""),
        reverse=True)
    total = len(summaries)
    n_corroborated = sum(
        1 for s in summaries if _CORROBORATION_RANK.get(s["corroboration"], 0) >= 1)
    # Coverage is counted over the FULL in-scope population (like
    # total_matching/n_corroborated), not the returned page: a reader has to be
    # able to tell how much of what it just asked about was ever analyzed.
    # Labeling is NOT a ranking key -- an unlabeled window is not weaker
    # evidence, it is unread evidence, and demoting it would hide exactly the
    # rows the note is warning about.
    n_labeled = sum(1 for s in summaries if s["labeled"])
    # Counted apart from the unlabeled ones: both render as
    # `labeled_signal_turns: 0`, neither is a finding of calm, but they are
    # different facts (never looked at vs looked at with nothing readable to
    # look at) and the note has to say which one a row is.
    n_labeled_empty = sum(
        1 for s in summaries if s["labeled"] and not s["labeled_signal_turns"])
    env = _envelope(summaries[:limit], total, order)
    env["n_corroborated"] = n_corroborated
    env["n_labeled"] = n_labeled
    if total:
        env["_epilogue"] = _ANALYSIS_DISCLAIMER
    notes = []
    if total > n_labeled:
        notes.append(_UNLABELED_SCOPE_NOTE.format(
            n_unlabeled=total - n_labeled, total=total))
        env["_epilogue"] += _PARTIAL_LABELING_EPILOGUE
    if n_labeled_empty:
        notes.append(_LABELED_EMPTY_SCOPE_NOTE.format(
            n_empty=n_labeled_empty, total=total))
        # Out-of-band too, not payload-only: `total == n_labeled` (every window
        # labeled, all of them empty) is exactly the scope where the branch
        # above adds nothing and the model is most likely to report "I looked
        # and it was calm" with no code-emitted correction behind it.
        env["_epilogue"] += _LABELED_EMPTY_EPILOGUE
    if notes and pre_migration:
        # Only alongside a note: on a pre-migration DB where every row carries
        # signals nothing was inferred wrongly, and an unprompted caveat there
        # would spend the model's attention on a distinction that does not bite.
        notes.append(_PRE_MIGRATION_NOTE)
    if notes:
        env["labeling_note"] = " ".join(notes)
    return env


def _signal_row(row):
    try:
        top2 = json.loads(row["top2_labels_json"]) if row["top2_labels_json"] else []
    except (TypeError, ValueError):
        top2 = []
    excerpt = row["excerpt"] or ""
    return {
        "message_id": row["message_id"],
        "actor": "me" if row["actor_is_me"] else "them",
        "label_class": row["label_class"],
        "top2": top2,
        "band": row["band"],
        "excerpt": excerpt[:EXCERPT_CAP],
    }


def get_conflict(conflict_id, offset=0, limit=15):
    """One conflict episode: its deterministic evidence (peak_signal), tone
    corroboration, off_text_flag, and up to 15 labeled signal turns (paginated;
    excerpts capped at 200 chars). Labels are eval-gated BANDS (band
    'tentative'/'likely'), never raw scores -- present them as such, and never
    read a resolution/apology into an episode whose off_text_flag is set. Pass
    offset to page further signals.

    `labeled` false means the per-turn labeling never ran for this episode
    (wall-clock-budgeted, strongest windows first), so signals_total 0 is
    unanalyzed, not clean -- the `note` says so. `labeled` true with
    labeled_signal_turns 0 is not "clean" either: labeling reads message text,
    so a stretch carried by tapbacks or moved off-channel can be looked at and
    yield nothing. Both cases get their own `note` and their own code-printed
    epilogue; neither is a calm verdict. labeled_signal_turns counts non-abstain
    rows, the same quantity find_conflicts lists the episode with;
    signals_total counts every stored row (abstains included) and is what
    paginates."""
    conflict_id = int(conflict_id)
    limit = max(1, min(int(limit), MAX_SIGNAL_ROWS))
    offset = max(0, int(offset))
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        # Aliased `c` because _labeled_at_select's pre-migration fallback
        # correlates a subquery on the conflicts row: an unqualified `id` inside
        # it would bind to conflict_signals.id instead.
        row = con.execute(
            "SELECT c.id, c.chat_id, c.first_date, c.last_date, "
            "c.start_session_id, c.end_session_id, c.topic, c.peak_signal_json, "
            f"c.corroboration, c.off_text_flag, c.source, "
            f"{_labeled_at_select(con, 'c')} "
            "FROM conflicts c WHERE c.id = ?",
            (conflict_id,),
        ).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}
        # Two counts, deliberately: `total` paginates (every stored row is
        # returned, abstains included), while `n_shippable` -- non-abstain rows,
        # the SAME definition find_conflicts' labeled_signal_turns uses -- is
        # what decides which note this window gets. Keying the note off
        # COUNT(*) made the two tools contradict each other on an abstain-only
        # window: find_conflicts called it "labeled but produced zero signal
        # turns", and drilling into that very row -- the expansion the envelope
        # tells the caller to make -- answered with the bands caveat instead,
        # dropping the warning at the step it was meant to survive.
        total, n_shippable = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN label_class != 'abstain' "
            "THEN 1 ELSE 0 END), 0) FROM conflict_signals WHERE conflict_id = ?",
            (conflict_id,)).fetchone()
        sig_rows = con.execute(
            "SELECT message_id, actor_is_me, label_class, top2_labels_json, "
            "band, excerpt FROM conflict_signals WHERE conflict_id = ? "
            "ORDER BY id LIMIT ? OFFSET ?", (conflict_id, limit, offset),
        ).fetchall()
        pre_migration = _predates_labeling_stamps(con)

    signals = [_signal_row(r) for r in sig_rows]
    labeled = row["labeled_at"] is not None
    if not labeled:
        note = _UNLABELED_NOTE
    elif n_shippable:
        note = _LABELED_NOTE
    else:
        # Labeled with nothing to show: the analyzed-and-clean reading is the
        # one thing this must not be, so it gets its own wording rather than
        # the bands caveat (there are no bands to caveat).
        note = _LABELED_EMPTY_NOTE
    if pre_migration and not labeled:
        note += " " + _PRE_MIGRATION_NOTE
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "start_session_id": row["start_session_id"],
        "end_session_id": row["end_session_id"],
        "topic": row["topic"],
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "peak_signal": _signals_json(row["peak_signal_json"]),
        "labeled": labeled,
        "signals": signals,
        "signals_total": total,
        # Named and counted exactly as find_conflicts does, so the row a caller
        # expanded reports the same number it was listed with (signals_total
        # can exceed it: abstains are stored and returned, they just are not
        # findings).
        "labeled_signal_turns": n_shippable,
        "returned": len(signals),
        "offset": offset,
        "limit": limit,
        "truncated": total > offset + len(signals),
        # An unlabeled episode has no bands to caveat -- it has an absence to
        # explain, and that is the more dangerous of the two to leave unsaid.
        # A LABELED one with zero turns has the same absence for a different
        # reason (see _LABELED_EMPTY_NOTE), so it is not sold as a clean read.
        "note": note,
        # Code-printed humility disclaimer (stripped from the model's view by
        # chat.run_turn, appended verbatim after the reply). Both absences get
        # one: never looked at, and looked at with nothing shippable to show.
        "_epilogue": _ANALYSIS_DISCLAIMER + (
            "" if labeled and n_shippable
            else _PARTIAL_LABELING_EPILOGUE if not labeled
            else _LABELED_EMPTY_EPILOGUE),
    }


# ===========================================================================
# C2 (RES-51): diagnose_conflict scorecard + softened DV-suppression floor.
#
# The scorecard composes DETERMINISTIC pieces ONLY -- lexicon I-vs-you /
# we-talk ratios, LSM, session initiator/ender roles, reply-latency + volume
# collapse read off C1's stored peak_signal survival stats -- and carries NO
# aggressor / who-won verdict. The one interpretive-ish section (which side
# withdrew vs pursued, a NON-BLAME demand-withdraw read) is the "role
# assignment" the DV floor withholds.
# ===========================================================================


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


def diagnose_conflict(conflict_id):
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


# --- feature: conflict ---
# ===========================================================================
# C3 (RES-52): gated cross-chat conflict trace.
#
# trace_conflict_across_chats(conflict_id, k, consent) answers "did this one
# argument spill into my OTHER conversations?" -- chunks in a DIFFERENT chat,
# within a few days of the episode, that are BOTH semantically close to it AND
# literally name chat A's partner (their handle, first name, or one of the
# chat's distinctive dyad_terms). Two STRUCTURAL gates keep it safe:
#   1. Registration gate (the real boundary): the schema is registered in
#      chat.py's tool list ONLY when config.CONFLICT_TRACE_ENABLED. Off by
#      default -- withholding the schema is the repo's one reliable way to stop
#      the model calling a tool (see the CLAUDE.md gotcha), so a disabled
#      deployment never needs prompt-level "don't trace" text.
#   2. Per-episode consent (UX ONLY, NOT the boundary): without consent=True the
#      tool returns a fixed ask-the-user payload and reads no data.
# The name-reference gate filters topically-similar-but-unnamed chunks (a
# "weekend with my boss" chunk near a "weekend with Jordan" episode is dropped
# unless it actually names Jordan). Computed on demand and written to NO table
# (there is deliberately no conflict_links table): nothing here is persisted.
# ===========================================================================
import re  # noqa: E402 (local to the additive C3 block)

from db import iso_to_day  # noqa: E402

from ._shared import CHUNK_PREVIEW_CHARS, MAX_VEC_K  # noqa: E402

# Shortest token that counts as a literal partner reference -- a 1-2 char name
# or dyad_term ("i", "ok") would match half the corpus, defeating the gate.
_TRACE_MIN_REF_LEN = 3
# Query-text budget fed to the embedder (the episode's own window text).
_TRACE_QUERY_CHARS = 4000
# Over-fetch factor: the sim floor + name gate prune hard, so pull a wider KNN
# pool than k before filtering (still capped at MAX_VEC_K).
_TRACE_POOL_FACTOR = 5


def _trace_ask_payload(conflict_id):
    """The fixed no-data response returned when consent was NOT given: it
    carries zero retrieved content -- only the instruction to ask first. This
    is the UX layer, not the security boundary (registration is)."""
    return {
        "conflict_id": conflict_id,
        "status": "consent_required",
        "links": [],
        "ask": ("Tracing whether this argument spilled into your OTHER "
                "conversations reads across chats. Want me to check? I'll look "
                "only at messages that both match this episode AND name the "
                "person involved, and I won't save anything."),
        "note": ("no data was read -- call again with consent=true only after "
                 "the user agrees; nothing is ever persisted"),
    }


def _chat_partner_references(con, chat_id):
    """The literal strings a related chunk must contain to count as naming chat
    A's partner: each participant handle_identifier, the counterparty contact
    first_name(s), and chat A's distinctive dyad_terms. Returns an ordered,
    de-duplicated list of (kind, lowercased_value); blank/too-short values are
    dropped so the gate stays meaningful."""
    refs = []
    for hid, first_name in con.execute(
        "SELECT h.handle_identifier, c.first_name "
        "FROM chat_participants cp "
        "JOIN handles h ON h.id = cp.handle_id "
        "LEFT JOIN contacts c ON c.id = h.contact_id "
        "WHERE cp.chat_id = ?", (chat_id,),
    ):
        if hid and len(hid) >= _TRACE_MIN_REF_LEN:
            refs.append(("handle", hid.lower()))
        if first_name and len(first_name) >= _TRACE_MIN_REF_LEN:
            refs.append(("name", first_name.lower()))
    for (term,) in con.execute(
        "SELECT term FROM dyad_terms WHERE chat_id = ?", (chat_id,),
    ):
        if term and len(term) >= _TRACE_MIN_REF_LEN:
            refs.append(("dyad_term", term.lower()))
    seen, out = set(), []
    for kind, value in refs:
        if value not in seen:
            seen.add(value)
            out.append((kind, value))
    return out


def _mentions_reference(text, references):
    """The first partner reference `text` literally mentions ({kind, value}) or
    None. Handle identifiers (phones/emails) match as substrings; names and
    dyad_terms match on WORD BOUNDARIES, so 'Jordan' fires on 'weekend with
    Jordan' but not on 'Jordania', and a topically-similar chunk that names no
    one is filtered out."""
    if not text:
        return None
    lowered = text.lower()
    for kind, value in references:
        if kind == "handle":
            if value in lowered:
                return {"kind": kind, "value": value}
        elif re.search(rf"\b{re.escape(value)}\b", lowered):
            return {"kind": kind, "value": value}
    return None


def trace_conflict_across_chats(conflict_id, k=10, consent=False):
    """Trace whether ONE conflict episode (a conflict_id from find_conflicts)
    spilled into your OTHER conversations: conversation chunks in a DIFFERENT
    chat, within CONFLICT_TRACE_BUFFER_DAYS of the episode, that are BOTH
    semantically close to it AND literally name chat A's partner (their handle,
    first name, or one of the chat's distinctive terms). A link needs cosine
    similarity > CONFLICT_TRACE_SIM_THRESHOLD AND a name hit -- topically
    similar but unnamed chunks are dropped.

    Cross-chat by nature, so it is off by default and consent-gated. The real
    off-switch is REGISTRATION: the whole tool is withheld from the model unless
    config.CONFLICT_TRACE_ENABLED. The `consent` param is a UX prompt only, NOT
    the security boundary -- without consent=True this returns a fixed
    ask-the-user payload and reads no data. Computed on demand; NOTHING is
    persisted (there is deliberately no conflict_links table)."""
    conflict_id = int(conflict_id)
    if not consent:
        return _trace_ask_payload(conflict_id)

    # Local import keeps this additive block self-contained and cycle-safe (the
    # package is fully initialized by the time a tool is called).
    from .chunks import _chunk_knn, _embed_query

    k = max(1, min(int(k), MAX_VEC_K))
    buffer = max(0, int(config.CONFLICT_TRACE_BUFFER_DAYS))
    threshold = float(config.CONFLICT_TRACE_SIM_THRESHOLD)

    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"conflict_id": conflict_id, "links": [],
                    "error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT chat_id, first_date, last_date, topic FROM conflicts "
            "WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            return {"conflict_id": conflict_id, "links": [],
                    "error": f"no conflict with id {conflict_id}"}
        origin_chat = row["chat_id"]
        first_date, last_date = row["first_date"], row["last_date"]
        # Convert the episode's UTC-ISO message dates to vec0 day numbers with
        # db.iso_to_day -- the SAME converter the metadata writer uses; mixing
        # formats silently empties the scan (see the CLAUDE.md gotcha).
        from_day = iso_to_day(first_date) - buffer
        to_day = iso_to_day(last_date) + buffer

        references = _chat_partner_references(con, origin_chat)
        # Query text = the episode's own full window text (same source the DV
        # screen reads), falling back to its topic when the window has no text.
        window_texts = _window_message_texts(
            con, conflict_id, origin_chat, first_date, last_date)
        query_text = " ".join(t for t in window_texts if t)[:_TRACE_QUERY_CHARS]
        if not query_text.strip():
            query_text = (row["topic"] or "")
        if not query_text.strip():
            return {"conflict_id": conflict_id, "origin_chat_id": origin_chat,
                    "links": [], "returned": 0,
                    "note": "no episode text to trace; nothing persisted"}

        query_vec = _embed_query(query_text)
        pool = max(1, min(k * _TRACE_POOL_FACTOR, MAX_VEC_K))
        candidates = _chunk_knn(
            con, query_vec, pool, chat_ids=None, from_day=from_day, to_day=to_day,
            exclude_chat_ids=[origin_chat])

    links = []
    for r in candidates:
        distance = r.get("distance")
        if distance is None:
            continue
        similarity = 1.0 - float(distance)
        if similarity <= threshold:
            continue
        matched = _mentions_reference(r.get("text") or "", references)
        if matched is None:
            continue
        links.append({
            "chunk_id": r["chunk_id"],
            "chat_id": r["chat_id"],
            "participants": r.get("participants"),
            "first_date": r["first_date"],
            "last_date": r["last_date"],
            "similarity": round(similarity, 4),
            "matched_reference": matched,
            "text": (r.get("text") or "")[:CHUNK_PREVIEW_CHARS],
        })
        if len(links) >= k:
            break

    return {
        "conflict_id": conflict_id,
        "origin_chat_id": origin_chat,
        "window": {"first_date": first_date, "last_date": last_date,
                   "buffer_days": buffer},
        "sim_threshold": threshold,
        "n_references": len(references),
        "links": links,
        "returned": len(links),
        "note": ("cross-chat spillover -- each link BOTH exceeds the similarity "
                 "threshold AND literally names the partner; computed on demand "
                 "and NOT persisted (no conflict_links table exists)"),
    }
# --- end feature: conflict ---


