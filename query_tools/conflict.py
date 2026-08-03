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


from ._shared import (
    _connect,
    _date_clauses,
    _envelope,
    _resolve_chat_scope,
    _restrict_to_direct_chats,
)
# The DV floor lives in conflict_dv (its single home) and is applied by
# conflict_diagnose / conflict_apology. Re-exported here so
# query_tools.conflict._dv_screen and .set_dv_screen_bypass (chat.py
# --no-dv-screen, tests) keep resolving after the module splits.
from .conflict_dv import _dv_screen, set_dv_screen_bypass  # noqa: F401

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
# A window labeled by the pre-budget pipeline carries its signal rows but a
# NULL labeled_at (the migration deliberately does not invent a stamp -- see
# schema.py). _labeled_at_select infers those back to labeled=true, so
# `labeled: false` now means no stamp AND no signal rows, which forces
# labeled_signal_turns=0 -- the two travel together and neither note has to
# describe a mixed shape. What those rows DO need saying about them is where
# their `labeled` came from, which is _PRE_MIGRATION_NOTE's job.
_UNLABELED_SCOPE_NOTE = (
    "{n_unlabeled} of {total} detected windows in scope have NOT been through "
    "per-turn escalation labeling yet (it runs on a wall-clock budget, "
    "strongest windows first; rerunning the conflict pipeline extends "
    "coverage). On a row with labeled=false, labeled_signal_turns=0 means NOT "
    "ANALYZED -- not 'analyzed and nothing escalatory found' -- and topic / "
    "reconciliation are unset for the same reason. The deterministic evidence "
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
# Appended to whichever note is emitted when some window in this database was
# labeled by a pipeline that recorded no completion stamp (see
# _has_pre_budget_rows): for THOSE rows `labeled` is INFERRED from stored signal
# rows, so the two states the stamp exists to separate are only partly
# separable. Keyed on the data, not on whether the column exists -- a plain
# `make annotate` can add the column without changing a single row.
_PRE_MIGRATION_NOTE = (
    "(Some windows here were labeled by an earlier version of the pipeline that "
    "recorded no completion stamp, so for those `labeled` is inferred from "
    "whether the window has stored signal turns: one the old pipeline looked at "
    "and found nothing in is indistinguishable from one it never reached. "
    "Rerunning the conflict pipeline stamps them properly.)"
)
_UNLABELED_NOTE = (
    "this episode has NOT been through per-turn escalation labeling yet (it "
    "runs on a wall-clock budget, strongest windows first; rerunning the "
    "conflict pipeline extends coverage), so zero signal turns here means NOT "
    "ANALYZED, not 'analyzed and nothing escalatory found' -- and topic is "
    "unset for the same reason. Its deterministic evidence (peak_signal, "
    "corroboration, off_text_flag) is complete regardless."
)


def _has_conflict_tables(con):
    """True only when both conflict tables exist -- a DB built before this
    feature has neither, and the read tools must return empty, not raise."""
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


# Sentinel `labeled_at` for a window labeled by a pipeline that recorded no
# completion stamp. Distinguishable from a real stamp by construction -- those
# are ISO timestamps from conflict_detect._now_iso -- so a reader can tell an
# INFERRED "labeled" from a recorded one, which is what gates the caveat below.
_PRE_BUDGET_STAMP = "pre-budget"


def _has_labeled_at(con):
    """Whether `conflicts` carries the labeled_at column at all. False on a DB
    neither re-ingested nor re-detected since the budgeted tier landed;
    returning rows beats raising `no such column`, the same contract
    _has_conflict_tables keeps one migration earlier."""
    return "labeled_at" in {r[1] for r in con.execute("PRAGMA table_info(conflicts)")}


def _has_pre_budget_rows(con):
    """Whether any window was labeled by a pipeline that recorded no stamp --
    signal rows present, `labeled_at` absent or NULL. That, NOT the column's
    existence, is what makes an unstamped-and-empty window ambiguous, and it is
    the condition _PRE_MIGRATION_NOTE describes.

    Keyed on the DATA rather than the SCHEMA because the two come apart, and
    the way they come apart used to silently corrupt this read: `annotate.py`'s
    _ensure_column self-heal runs the whole of schema.create_schema, so a plain
    `make annotate` ADDS conflicts.labeled_at as a side effect. On the old
    column-existence test that single migration flipped every window the old
    pipeline had labeled from `labeled: true` to `labeled: false` -- and
    removed the caveat that would have explained it, because the caveat was
    gated on the column being missing. A user who ran annotate before conflict
    got the most confident version of the wrong answer."""
    if not _has_conflict_tables(con):
        return False
    stamp = "c.labeled_at IS NULL AND " if _has_labeled_at(con) else ""
    return bool(con.execute(
        f"SELECT 1 FROM conflicts c WHERE {stamp}EXISTS "
        "(SELECT 1 FROM conflict_signals s WHERE s.conflict_id = c.id) LIMIT 1"
    ).fetchone())


def _labeled_at_select(con, alias):
    """The SELECT fragment yielding a `labeled_at` value: the recorded stamp
    where there is one, else 'pre-budget' inferred from stored signal rows,
    else NULL.

    The inference is applied on EVERY database, not only one whose `conflicts`
    table predates the column. Gating it on the schema was wrong because the
    column can arrive without the DATA changing -- `annotate.py`'s
    _ensure_column runs create_schema, which ALTERs it on -- and at that moment
    every window the old pipeline had labeled silently went from `labeled: true`
    to `labeled: false`. The rows did not change; only the test did. Keyed on
    evidence, a migration is a no-op here, which is what it should always have
    been.

    The value is derived from evidence, never asserted: 'pre-budget' only where
    the window actually HAS conflict_signals rows. A flat non-NULL literal was
    wrong in the one direction that matters -- an old `--no-llm` run wrote
    windows nothing ever labeled, and marking them `labeled` made get_conflict
    report "we looked and found nothing" about a window nothing looked at,
    exactly the misreading the labeled/unlabeled split exists to prevent. The
    inference is one-way-safe: signals cannot exist without a labeling pass,
    while their absence stays ambiguous (a pass that found nothing looks like no
    pass) -- which is what _PRE_MIGRATION_NOTE says out loud rather than
    papering over.

    Read-side only, deliberately: the stored `labeled_at` really is NULL on
    those rows, so conflict_detect's budgeted pass still re-labels them (their
    turn selection changed, so it should). This changes what is REPORTED, never
    what is done.

    `alias` is the caller's table alias for `conflicts` and is REQUIRED: the
    fragment correlates a subquery on `<alias>.id`, and an unqualified `id`
    there would resolve to the subquery's own conflict_signals.id. The fragment
    stays a constant -- no user input is ever interpolated."""
    stamp = f"{alias}.labeled_at" if _has_labeled_at(con) else "NULL"
    return (f"COALESCE({stamp}, CASE WHEN EXISTS (SELECT 1 FROM conflict_signals ps "
            f"WHERE ps.conflict_id = {alias}.id) THEN '{_PRE_BUDGET_STAMP}' END) "
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

    `labeled` mirrors the recorded conflicts.labeled_at stamp OR, where there
    is none, the signals-based inference (_labeled_at_select): it separates "we
    never looked" (labeled false) from "the pass ran here" (labeled true).
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
                   date_from=None, date_to=None, limit=10) -> dict:
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
        # Whether any window in the DATABASE carries inferred-labeled state,
        # not whether the column exists: a plain `make annotate` adds the
        # column without touching a row, and gating on the schema made that
        # migration silently drop this caveat.
        # `or not _has_labeled_at(con)`: a table with no stamp column cannot
        # have a trustworthy `labeled` on ANY row, even if no signal row
        # survives to trigger the data test -- that is the one case the schema
        # question still dominates, so it is kept as a floor under it.
        pre_migration = _has_pre_budget_rows(con) or not _has_labeled_at(con)
        # ...and how many of THOSE are in this scope, so a page made entirely
        # of inferred rows still says its `labeled` values are inferred rather
        # than recorded. Counted off the raw rows: the sentinel is deliberately
        # not exposed in the summary dicts.
        n_inferred = sum(1 for r in rows if r["labeled_at"] == _PRE_BUDGET_STAMP)

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
    if pre_migration and (notes or n_inferred):
        # Alongside another note (where an unstamped-and-empty window really is
        # ambiguous), or whenever an inferred row is on this page -- a scope
        # made entirely of inferred-labeled windows produces no other note, and
        # saying nothing there would present an inferred `labeled: true` as a
        # recorded one. Still silent when neither holds, so a fully stamped DB
        # never spends the model's attention on a distinction that cannot bite.
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


def get_conflict(conflict_id, offset=0, limit=15) -> dict:
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
        pre_migration = _has_pre_budget_rows(con) or not _has_labeled_at(con)

    signals = [_signal_row(r) for r in sig_rows]
    labeled = row["labeled_at"] is not None
    # This episode's `labeled` came from the signals-based inference rather than
    # a recorded stamp -- real labels, but from a pass that recorded no
    # completion time, so a rerun will refresh them.
    inferred = row["labeled_at"] == _PRE_BUDGET_STAMP
    if not labeled:
        note = _UNLABELED_NOTE
    elif n_shippable:
        note = _LABELED_NOTE
    else:
        # Labeled with nothing to show: the analyzed-and-clean reading is the
        # one thing this must not be, so it gets its own wording rather than
        # the bands caveat (there are no bands to caveat).
        note = _LABELED_EMPTY_NOTE
    # On an unlabeled row the caveat explains the ambiguity the inference
    # leaves; on an inferred row it explains where the `labeled: true` came
    # from. Both are things a caller acting on this episode needs.
    if inferred or (pre_migration and not labeled):
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


# diagnose_conflict (C2) and trace_conflict_across_chats (C3) live in their own
# modules now; re-export them so query_tools.conflict.diagnose_conflict and
# .trace_conflict_across_chats keep resolving for callers and tests. Bottom
# imports: both modules read the notes/helpers defined above.
from .conflict_diagnose import diagnose_conflict  # noqa: E402,F401
from .conflict_trace import trace_conflict_across_chats  # noqa: E402,F401
