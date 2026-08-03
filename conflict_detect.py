"""Deterministic conflict finder + eval-gated escalation labels (RES-46 / C1).

Standalone batch CLI (cloned from annotate.py's ergonomics: resumable per
chat, commits as it goes, Ctrl-C safe). Invoked by the Makefile `pipeline`
after `annotate` -- it consumes chunk_annotations for valence corroboration.
No ingest.py edit.

What it does, per 1:1 chat (group chats are out for v1):

  1. Find TENSE STRETCHES from DETERMINISTIC signals only -- the LLM never
     originates a conflict:
       * latency_withdrawal -- reply-gap survival (stats.frac_over_threshold
         over RAW gaps, right-censored) elevated vs a trailing-90-day baseline
       * one_sided_session  -- a session where one side never answered
         (initiator/ender attribution from the `sessions` table)
       * dislike_run        -- >= 2 dislike tapbacks (associated_message_type
         2002 -- DISLIKE, not 2003=laugh) inside the window
       * volume_collapse    -- window daily rate <= (1-X) * trailing-8-week
         median rate
     A window is written when it trips >= CONFLICT_MIN_SIGNALS distinct
     signals AND at least one is an ACTIVE-TENSION signal (latency_withdrawal
     or dislike_run). The latter gate exists because one_sided_session and
     volume_collapse are not independent -- a single quiet, one-directional
     lull trips both -- so requiring them plus a genuine tension marker stops
     ordinary conversational ebbs from being labeled conflicts.
     Distilled/absent annotations NEVER block detection: valence only
     corroborates (source='llm' rows only) and can RAISE a window's rank, never
     originate it. A window with no llm-annotated chunks is written with
     corroboration='none' -- the finder is never dead on a default (all-
     distilled) corpus. Run `annotate.py --quality balanced --upgrade-distilled`
     over the chats you care about to light corroboration.

  2. Attach per-turn escalation labels ONLY where the F4 eval gate says they
     are earned. Each candidate turn is classified (Gottman's Four Horsemen +
     repair; the model may also say neutral/abstain) with a top-2 and an
     UNCALIBRATED confidence. At write time:
       * below CONFLICT_LABEL_CONF_FLOOR -> dropped,
       * horseman classes collapse to the binary harsh/critical umbrella
         ('criticism') unless the full 4-way is gate-cleared,
       * eval_gate_passed(con, 'conflict_labels', cls) must clear the class or
         the row is SUPPRESSED,
       * survivors render as BANDS (tentative/likely), never a raw float.

Step 2 is the only expensive part, so it runs on a WALL-CLOCK BUDGET
(CONFLICT_BUDGET_MINUTES, default 25) rather than a fixed call count -- the
same contract as annotate.py's --quality hybrid tier: detection covers every
chat, then the strongest windows are labeled first until the deadline, and a
rerun continues down the ranking (conflicts.labeled_at is the resume marker).
That is what makes the runtime promise hold on a bigger corpus or a slower
machine, where a tuned constant would not. `--budget-minutes 0` restores the
unbounded path.

The `conflict_model` ingest_state key guards against mixing label models in
one conflicts table (same shape as annotate_model / embed_model).
"""

import argparse
import datetime
import json
import statistics
import time

import ollama
import progress
import schema
# Several CONFLICT_* knobs below are read back through this module by
# conflict_signals / conflict_labeling (so a test patching one here moves
# them too), which is why they look unused from inside this file.
from config import (  # noqa: F401
    CONFLICT_BUDGET_MINUTES,
    CONFLICT_LABEL_CONF_FLOOR,
    CONFLICT_LATENCY_FRAC,
    CONFLICT_LATENCY_Q_HOURS,
    CONFLICT_MAX_LABEL_TURNS,
    CONFLICT_MIN_SIGNALS,
    CONFLICT_MIN_TURN_CHARS,
    CONFLICT_MIN_VALENCE_DROP,
    CONFLICT_MODEL,
    CONFLICT_OFF_TEXT_GAP_HOURS,
    CONFLICT_THINK,
    CONFLICT_TIMEOUT,
    CONFLICT_VOLUME_COLLAPSE,
    TARGET_DB,
    has_effort_levels,
)
from db import check_model_guard, connect_default, resolve_budget_minutes, run_cli
from eval.evalkit.gates import eval_gate_passed
from query_tools._shared import _REAL_MESSAGE_CLAUSES, _restrict_to_direct_chats

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# The eval-gate task whose cleared classes govern which labels may ship.
GATE_TASK = "conflict_labels"
# Gottman's Four Horsemen + repair. MUST match
# eval.evalkit.gates.TASKS['conflict_labels']['classes'] (order aside) -- the
# seed verdict pre-clears exactly these. neutral/abstain are model-only
# non-escalation verdicts, never gate classes.
HORSEMEN = ("criticism", "contempt", "defensiveness", "stonewalling")
GATE_CLASSES = HORSEMEN + ("repair",)
TURN_CLASSES = GATE_CLASSES + ("neutral", "abstain")
# The harsh/critical umbrella the 4-way collapses to when the horsemen are not
# all gate-cleared (criticism is itself a gate class, so it can still ship).
_COLLAPSE_TARGET = "criticism"

_REAL_WHERE = " AND ".join(("m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES))

_DISLIKE_TAPBACK = 2002  # DISLIKE. NOT 2003 (laugh); NOT the whole 2000-2999 range.

_TURN_CLASS_LIST = ", ".join(
    f'"{c}"' for c in ("criticism", "contempt", "defensiveness",
                       "stonewalling", "repair", "neutral"))
_TURN_SYSTEM = f"""You label ONE message from a tense text-message exchange for
signs of conflict escalation, using Gottman's Four Horsemen plus a repair class.
Classes:
- "criticism": attacking the other's character/personality ("you always").
- "contempt": mockery, sarcasm, insults, name-calling (the most corrosive).
- "defensiveness": deflecting blame, counter-attacking, playing the victim.
- "stonewalling": withdrawing, shutting down, terse/absent non-answers.
- "repair": de-escalating -- apology, taking responsibility, reaching back.
- "neutral": no escalation in this message.
Respond with ONLY one JSON object, no prose:
{{"class": <one of {_TURN_CLASS_LIST}>,
 "top2": [<class>, <class>],
 "confidence": <float 0..1, your certainty>}}
If you genuinely cannot tell, use "class": "abstain"."""

_TURN_NUDGE = (
    'Respond with only one JSON object {"class": <label>, "top2": [<label>, '
    '<label>], "confidence": <0..1>} -- no prose, no code fences.'
)

_GATE_CLASS_LIST = ", ".join(f'"{c}"' for c in GATE_CLASSES)
_GATE_SYSTEM = f"""You classify a short text-message excerpt for its dominant
conflict-escalation type, using Gottman's Four Horsemen plus repair. Pick the
SINGLE best-fitting class even if the fit is loose.
- {HORSEMEN[0]}: attacking character.
- {HORSEMEN[1]}: contempt / mockery / insult.
- {HORSEMEN[2]}: deflecting blame / counter-attack.
- {HORSEMEN[3]}: withdrawing / shutting down.
- repair: de-escalation / apology.
Respond with ONLY one JSON object: {{"class": <one of {_GATE_CLASS_LIST}>}}."""

_GATE_NUDGE = (
    f'Respond with only {{"class": <one of {", ".join(GATE_CLASSES)}>}} -- '
    "no prose, no code fences."
)


def _connect():
    return connect_default(DB_PATH, rows=True)


# ---------------------------------------------------------------------------
# Think resolution (the pipeline default, not a change to the tri-state)
# ---------------------------------------------------------------------------

# What an unset CONFLICT_THINK resolves to on a model that reads
# 'low'/'medium'/'high' as a reasoning-EFFORT level (config.has_effort_levels).
# To every other model an effort string is merely truthy and buys a FULL
# reasoning pass, which is the failure this whole resolver exists to prevent.
_THINK_EFFORT = "low"


def _resolve_think(model):
    """The think value this pipeline labels with. An explicit CONFLICT_THINK
    (or an explicit CHAT_THINK it inherits) always wins and passes through
    untouched; only a genuinely UNSET one is resolved here.

    Unset means None at the config seam, i.e. "omit the flag" -- and on a
    thinking-capable model omitting it buys a full reasoning pass on every
    single turn: measured 2026-07-28 on qwen3.5:4b, 197 s per labeling call
    against 2.2 s with think=False. Nothing about a one-JSON-object turn
    classification wants a reasoning pass, so the PIPELINE default becomes
    False (an effort level on the models that have them), exactly as
    annotate._resolve_quality does for its presets. The tri-state itself is
    untouched: None still means "omit" at the ollama.py seam, and
    config._think / CHAT_THINK keep their meanings."""
    if CONFLICT_THINK is not None:
        return CONFLICT_THINK
    return _THINK_EFFORT if has_effort_levels(model) else False


# ---------------------------------------------------------------------------
# LLM turn classification (mockable: the write path takes classify_fn)
# ---------------------------------------------------------------------------

def _validate_turn(data):
    """Coerce a parsed turn reply to the storage contract. Anything malformed
    becomes an abstain (confidence 0) rather than a spurious escalation."""
    if not isinstance(data, dict):
        return {"class": "abstain", "top2": [], "confidence": 0.0}
    cls = data.get("class")
    if cls not in TURN_CLASSES:
        cls = "abstain"
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    top2 = [c for c in (data.get("top2") or []) if c in TURN_CLASSES][:2]
    return {"class": cls, "top2": top2, "confidence": conf}


def _classify_turn(text, model, think):
    """One JSON-mode request labeling a single turn. Returns the validated
    dict (never raises on a bad reply)."""
    data = ollama.chat_once_json(
        [{"role": "system", "content": _TURN_SYSTEM},
         {"role": "user", "content": (text or "")[:1000]}],
        model=model, think=think, timeout=CONFLICT_TIMEOUT, retry_nudge=_TURN_NUDGE,
    )
    return _validate_turn(data)


def _is_labelable(text):
    """Whether a turn has any text for the classifier to read at all: it needs
    one alphanumeric character. An emoji/punctuation/tapback-ish turn carries
    no lexical signal, and on this corpus ~70% of labels come back `neutral`
    anyway, each costing a whole request to say nothing.

    Length is deliberately NOT a filter here. `_TURN_SYSTEM` defines
    stonewalling as "withdrawing, shutting down, terse/absent non-answers" and
    contempt covers "k.", "fine.", "whatever" -- every one of them under a
    15-character floor. Screening those out deleted the exact class the
    classifier is asked to find, and worse: a window whose escalation is
    ENTIRELY terse withdrawal then selected zero turns, was still stamped
    labeled_at (the stamp is a one-way latch), and read downstream as
    "analyzed and clean". CONFLICT_MIN_TURN_CHARS survives as a RANKING
    preference in _select_turns' no-anchor fallback -- it demotes, it no longer
    deletes."""
    return any(ch.isalnum() for ch in (text or ""))


def _select_turns(seg, max_turns, peak_id=None):
    """The <= max_turns turns of one window worth spending LLM calls on, as
    [(message_id, is_from_me, text)] in CHRONOLOGICAL order (the labels read as
    a sequence downstream, so they are never emitted peak-outward).

    Two rules, in order:
      * the lexical prescreen (_is_labelable) drops turns with no alphanumeric
        character -- nothing for the classifier to read -- and nothing else;
      * what is left is taken as a contiguous span CENTERED on the window's
        peak_gap_message_id -- the message left hanging, i.e. the escalation
        moment the deterministic finder actually keyed on. The old head slice
        (`[:max_turns]`) provably misses it: windows in this corpus run to
        1,940 candidate turns.
    A window with no peak id, or whose peak turn is not in this segment, has no
    escalation moment to bracket, so it falls back to a chronological slice with
    ONE ranking preference: turns at or over CONFLICT_MIN_TURN_CHARS are taken
    before shorter ones, and the chosen set is re-sorted chronologically before
    it is returned (the sequence contract holds on every path). That is all the
    floor does now -- a terse "whatever" is a candidate like any other, it just
    yields its slot to a longer turn when the window has more candidates than
    the cap. When the peak turn is dropped by the prescreen the span still
    centers on WHERE it was, so the neighbours that survive are the ones around
    the escalation."""
    kept = [(i, m) for i, m in enumerate(seg) if _is_labelable(m["text"])]
    turns = [(m["id"], m["is_from_me"], m["text"]) for _i, m in kept]
    if max_turns is not None and max_turns <= 0:
        return []
    if max_turns is None or len(turns) <= max_turns:
        return turns

    anchor = next((n for n, (_i, m) in enumerate(kept) if m["id"] == peak_id), None)
    if anchor is None and peak_id is not None:
        # The peak turn itself was prescreened out (a long silence is often
        # answered by a tapback or a bare emoji): anchor on its POSITION
        # instead, so the span still lands on the escalation rather than on the
        # window's head.
        seg_pos = next((i for i, m in enumerate(seg) if m["id"] == peak_id), None)
        if seg_pos is not None:
            anchor = sum(1 for i, _m in kept if i < seg_pos)
    if anchor is None:
        # No escalation moment to bracket: rank by the length preference, then
        # restore chronological order. `k` breaks length ties by position, so a
        # window of uniformly long turns still yields its head -- the behavior
        # this branch had when the floor was a filter.
        ranked = sorted(range(len(turns)),
                        key=lambda k: (len((turns[k][2] or "").strip())
                                       < CONFLICT_MIN_TURN_CHARS, k))
        return [turns[k] for k in sorted(ranked[:max_turns])]
    start = min(max(0, anchor - max_turns // 2), len(turns) - max_turns)
    return turns[start:start + max_turns]


def gate_classifier(con, items):
    """Production classifier wired into eval.evalkit.gates.TASKS for the
    owner-run `python -m eval.evalkit gate conflict_labels` path
    (signature classifier(con, items) -> {id: predicted_class}). Force-chooses
    one of the five gate classes for each chunk so kappa is well-defined;
    needs Ollama (owner-run, never in hermetic tests). Resolved lazily via its
    dotted-path string, so importing gates never pulls this module in."""
    out = {}
    # The resolved value, not the raw CONFLICT_THINK: the gate path is the same
    # one-JSON-object classification as the label pass, so leaving it on the
    # unset (= omit the flag) default would make owner gating 90x slow too.
    think = _resolve_think(CONFLICT_MODEL)
    for it in items:
        data = ollama.chat_once_json(
            [{"role": "system", "content": _GATE_SYSTEM},
             {"role": "user", "content": (it.get("text") or "")[:1500]}],
            model=CONFLICT_MODEL, think=think, timeout=CONFLICT_TIMEOUT,
            retry_nudge=_GATE_NUDGE,
        )
        cls = data.get("class") if isinstance(data, dict) else None
        out[it["id"]] = cls if cls in GATE_CLASSES else _COLLAPSE_TARGET
    return out


# ---------------------------------------------------------------------------
# Model guard (mirrors annotate._check_model_guard)
# ---------------------------------------------------------------------------

def _check_model_guard(con, model, record=True):
    """Refuse to mix label models within one conflicts table (same shape as
    the chunk_annotations annotate_model guard). Records the chosen model.

    `record=False` runs the REFUSAL half only. main() calls it that way before
    anything destructive happens -- the --relabel clear, and the detection pass,
    whose _drop_conflicts deletes a superseded window's conflict_signals rows
    along with it -- because both of those destroy labels the guard exists to
    protect, and both used to run BEFORE the only guard call (inside
    label_top_conflicts / detect_conflicts). Detection first was the worse of
    the two: it can drop the last conflicts row, and a guard that then reads
    COUNT(conflicts) == 0 accepts the new model and rewrites conflict_model
    having silently thrown the old labels away. Recording is left to the
    labeling pass so the key still means "the model that labeled these rows":
    an early record stamped conflict_model on a run that went on to label
    nothing (a Ctrl-C during detection, --no-llm), and the next run was refused
    in the name of a model that had never written a label."""
    check_model_guard(
        con, key="conflict_model", model=model, table="conflicts", record=record,
        message=(
            "conflicts holds {n} rows labeled by {prev}, but the model is "
            "now {model}. Mixing label models skews the escalation labels "
            "-- clear conflicts (or reset_db.py) to switch, or rerun with "
            "--model {prev}."
        ),
    )


# ---------------------------------------------------------------------------
# Per-chat detection
# ---------------------------------------------------------------------------

def _consecutive_runs(hot_flags):
    """Maximal runs of consecutive hot session indices -> [(start, end)]."""
    runs, i, n = [], 0, len(hot_flags)
    while i < n:
        if hot_flags[i]:
            j = i
            while j + 1 < n and hot_flags[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _now_iso():
    """Unlabeled-UTC ISO stamp, the same shape as every date column here (and
    the same tz-aware-then-dropped construction ingest.py/loose_ends.py use --
    datetime.utcnow() is deprecated)."""
    return (datetime.datetime.now(datetime.timezone.utc)
            .replace(tzinfo=None).isoformat(timespec="seconds"))


# Ids per `WHERE id IN (...)` slice, under SQLite's default 999-variable
# ceiling -- one long-running chat can hold more stale windows than that.
_ID_SLICE = 500


def _drop_conflicts(con, chat_id, keep=()):
    """Delete this chat's conflicts (and their signals) EXCEPT `keep`."""
    stale = [r[0] for r in con.execute(
        "SELECT id FROM conflicts WHERE chat_id = ?", (chat_id,)).fetchall()
        if r[0] not in keep]
    for i in range(0, len(stale), _ID_SLICE):
        block = stale[i:i + _ID_SLICE]
        ph = ",".join("?" * len(block))
        con.execute(f"DELETE FROM conflict_signals WHERE conflict_id IN ({ph})", block)
        con.execute(f"DELETE FROM conflicts WHERE id IN ({ph})", block)


def detect_chat(con, chat_id, corpus_end, classify_fn, model, label=True,
                max_label_turns=CONFLICT_MAX_LABEL_TURNS):
    """Detect + write every conflict in one 1:1 chat. Returns the number of
    conflicts written. With label=False the deterministic windows are still
    written but the per-turn escalation labeling (the only LLM work) is skipped
    -- no conflict_signals, no topic. max_label_turns caps the LLM-labeled
    turns per window (selected by _select_turns, peak-anchored).

    Re-runs are idempotent by RECONCILING rather than by deleting and
    rewriting: a window whose (first_date, last_date) is unchanged keeps its
    row id, its conflict_signals, its topic and its labeled_at stamp, and only
    its deterministic columns are refreshed; windows that no longer exist are
    dropped with their signals. The distinction matters because every budgeted
    run re-detects all 787 chats before labeling -- a delete-then-rewrite would
    throw away the labels the previous run's budget bought, and no rerun could
    ever extend coverage. An already-labeled window is therefore NOT re-labeled
    here; changing the window (new messages) is what makes it a candidate
    again, and clearing the table is what forces a wholesale re-label."""
    sessions = _load_sessions(con, chat_id)
    messages = _load_messages(con, chat_id)
    prior = {(r["first_date"], r["last_date"]): r for r in con.execute(
        "SELECT id, first_date, last_date, labeled_at FROM conflicts "
        "WHERE chat_id = ?", (chat_id,)).fetchall()}
    if len(sessions) == 0 or len(messages) < 2:
        _drop_conflicts(con, chat_id)
        return 0

    gap = _reply_gaps(messages, corpus_end)
    dislike_dts = _load_dislike_dts(con, chat_id)
    daily = _daily_counts(messages)
    chat_mean, llm_rows = _llm_valence(con, chat_id)
    four_way_cleared = all(eval_gate_passed(con, GATE_TASK, h) for h in HORSEMEN)

    per_session = [_session_signals(s, gap, dislike_dts) for s in sessions]
    hot_flags = [bool(fired) for fired, _meta in per_session]

    n_written = 0
    kept_ids = set()
    for i, j in _consecutive_runs(hot_flags):
        first_date = sessions[i]["first_date"]
        last_date = sessions[j]["last_date"]
        first_dt, last_dt = _dt(first_date), _dt(last_date)
        fired = set().union(*(per_session[k][0] for k in range(i, j + 1)))

        window_msgs = sum(sessions[k]["n_messages"] for k in range(i, j + 1))
        v_fired, v_ratio = _volume_collapse(daily, first_dt, last_dt, window_msgs)
        if v_fired:
            fired.add("volume_collapse")
        # Independence gate + min-signals: a window needs >= CONFLICT_MIN_SIGNALS
        # distinct signals AND at least one active-tension signal. Without the
        # latter, ordinary conversational ebbs (pure one_sided+volume lulls,
        # ~40% of raw windows) get labeled conflicts.
        if not _window_qualifies(fired):
            continue

        corrob, boost, valence_drop = _corroboration(
            chat_mean, llm_rows, first_date, last_date)
        off_flag, max_gap = _off_text(messages, first_dt, last_dt)

        peak_h = peak_id = None
        dislike_total = 0
        for k in range(i, j + 1):
            meta = per_session[k][1]
            dislike_total += meta["dislike_count"]
            if meta["peak_gap_hours"] is not None and (
                    peak_h is None or meta["peak_gap_hours"] > peak_h):
                peak_h, peak_id = meta["peak_gap_hours"], meta["peak_gap_message_id"]

        _, band_cut = _band(1.0, CONFLICT_LABEL_CONF_FLOOR)
        peak_signal = {
            "signals": sorted(fired),
            "signal_count": len(fired),
            "peak_gap_hours": round(peak_h, 1) if peak_h is not None else None,
            "peak_gap_message_id": peak_id,
            "dislike_count": dislike_total,
            "volume_ratio": round(v_ratio, 3) if v_ratio is not None else None,
            "off_text_max_gap_hours": max_gap,
            "valence_drop": valence_drop,
            "valence_boost": boost,
            "band_likely_cutpoint": band_cut,
        }

        existing = prior.get((first_date, last_date))
        if existing is None:
            cur = con.execute(
                "INSERT INTO conflicts(chat_id, first_date, last_date, "
                "start_session_id, end_session_id, topic, peak_signal_json, "
                "corroboration, off_text_flag, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, first_date, last_date, sessions[i]["id"], sessions[j]["id"],
                 None, json.dumps(peak_signal), corrob, int(bool(off_flag)), "llm"),
            )
            conflict_id = cur.lastrowid
            already_labeled = False
        else:
            # Same window, recomputed evidence: refresh the deterministic
            # columns in place and keep id/topic/labeled_at (and, through the
            # id, this window's conflict_signals).
            conflict_id = existing["id"]
            already_labeled = existing["labeled_at"] is not None
            con.execute(
                "UPDATE conflicts SET start_session_id = ?, end_session_id = ?, "
                "peak_signal_json = ?, corroboration = ?, off_text_flag = ?, "
                "source = ? WHERE id = ?",
                (sessions[i]["id"], sessions[j]["id"], json.dumps(peak_signal),
                 corrob, int(bool(off_flag)), "llm", conflict_id),
            )
        kept_ids.add(conflict_id)

        if label and not already_labeled:
            seg = [m for m in messages
                   if first_dt <= _dt(m["date"]) <= last_dt and m["text"]]
            turns = _select_turns(seg, max_label_turns, peak_id)
            # Belt and braces: an unlabeled window should have no signals, but
            # a re-label must never stack a second set on top of a first.
            con.execute("DELETE FROM conflict_signals WHERE conflict_id = ?",
                        (conflict_id,))
            try:
                _n_labels, survivors = _label_turns(con, conflict_id, turns,
                                                    classify_fn, four_way_cleared)
            except BaseException:
                # A window is all of its turns or none of them -- the contract
                # label_top_conflicts already keeps, now kept here too. Without
                # it a Ctrl-C mid-window left committed signal rows with no
                # stamp (detect_conflicts' handler COMMITS rather than rolling
                # back), and the read tools infer "labeled" from exactly that
                # shape: a window whose labeling covered 2 of 6 turns was
                # reported as analyzed and complete, and a later --no-llm run
                # never revisits it, so the misreport persisted. Deleting the
                # partial rows restores the premise the inference rests on --
                # signals imply a pass that finished.
                con.execute("DELETE FROM conflict_signals WHERE conflict_id = ?",
                            (conflict_id,))
                raise
            # labeled_at is stamped even with zero survivors: the window WAS
            # processed, and the signal rows are not the marker.
            con.execute("UPDATE conflicts SET topic = ?, labeled_at = ? WHERE id = ?",
                        (statistics.mode(survivors) if survivors else None,
                         _now_iso(), conflict_id))
        n_written += 1
    _drop_conflicts(con, chat_id, keep=kept_ids)
    return n_written


def detect_conflicts(con, model, classify_fn=None, limit=None, dry_run=False,
                     label=True, max_label_turns=CONFLICT_MAX_LABEL_TURNS,
                     think=None, stats=None):
    """Sweep every 1:1 chat (via _restrict_to_direct_chats -- F3). classify_fn
    is the injectable label path (default: the real LLM); tests pass a stub.
    label=False is the deterministic-only fast path: windows are written but the
    per-turn LLM labeling is skipped (zero Ollama calls), so the model guard and
    classifier are never touched. `think` None resolves through _resolve_think
    (an unset CONFLICT_THINK must not reach the request as "omit the flag").

    `stats`, when given, is filled with this pass's out-of-band facts (the
    annotate_hybrid convention -- a return value can't carry them without
    breaking every caller): `chat_ids`, the chats actually scanned, so a
    --limit run can scope the labeling phase to them; and `interrupted`, since
    a Ctrl-C is caught HERE and a two-phase caller that could not tell would
    walk straight into a full budget of LLM traffic the user just stopped."""
    if stats is None:
        stats = {}
    stats.setdefault("interrupted", False)
    all_ids = [r[0] for r in con.execute("SELECT id FROM chats ORDER BY id").fetchall()]
    direct = _restrict_to_direct_chats(con, all_ids)
    if limit is not None:
        direct = direct[:limit]
    stats["chat_ids"] = list(direct)
    scope = f"(model {model})" if label else "(deterministic only -- no labeling)"
    print(f"{len(direct)} direct (1:1) chats to scan for conflicts {scope}")
    if dry_run:
        print(f"--dry-run: first chat ids: "
              f"{', '.join(str(c) for c in direct[:10]) or '(none)'}")
        return 0
    if not direct:
        return 0
    if label:
        _check_model_guard(con, model)
        if classify_fn is None:
            think = _resolve_think(model) if think is None else think
            classify_fn = lambda text: _classify_turn(text, model, think)  # noqa: E731

    corpus_end = _corpus_end(con)
    total = 0
    try:
        with progress.track("Detecting conflicts", total=len(direct), header=False) as p:
            for chat_id in direct:
                total += detect_chat(con, chat_id, corpus_end, classify_fn, model,
                                     label=label, max_label_turns=max_label_turns)
                con.commit()
                p.advance()
    except KeyboardInterrupt:
        con.commit()
        stats["interrupted"] = True
        print(f"\nInterrupted -- committed {total} conflicts; rerun to resume.")
        return total
    print(f"Detected {total} conflicts across {len(direct)} chats")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_budget_minutes(budget_minutes=None):
    """(minutes, provenance) for the run's wall-clock budget. Precedence:
    --budget-minutes > CONFLICT_BUDGET_MINUTES > the config default, the same
    caller > env > default order annotate_hybrid._resolve_budget_minutes uses.

    0 means UNBUDGETED in both places, where annotate's hybrid tier reads 0 as
    "unset": there the budget IS the tier, here it is additive, so switching it
    off has to restore the unbounded path rather than fall back to a default.

    A NEGATIVE value is refused, not clamped, from either source. Clamping sent
    it to 0 = unbudgeted -- the unbounded run this tier exists to prevent, and
    the exact run the CLI already refuses `--budget-minutes -5` for, so an env
    typo silently bought what a flag typo is told off for. The message names the
    knob it came from, mirroring the parser's wording."""
    return resolve_budget_minutes(
        budget_minutes, CONFLICT_BUDGET_MINUTES, CONFLICT_BUDGET_MINUTES,
        zero_means="unbudgeted",
        flag_name="--budget-minutes", env_name="CONFLICT_BUDGET_MINUTES",
        default_name="CONFLICT_BUDGET_MINUTES",
        flag_error="--budget-minutes cannot be negative (0 disables the budget)",
        env_error=("CONFLICT_BUDGET_MINUTES cannot be negative "
                   "(0 disables the budget)"),
    )


def _clear_labels(con):
    """Drop every window's labels so the next pass starts over (--relabel).

    The resume stamp is otherwise a one-way latch, by design: a window is only
    re-labeled when its dates change. That is right for resuming a budget and
    wrong for the case where what labeling PRODUCES changed -- a different
    CONFLICT_MAX_LABEL_TURNS or CONFLICT_MIN_TURN_CHARS, a newly cleared eval
    gate -- where every stamped window would silently keep the old answer.

    Clears the signals and the topic too, not just the stamp: leaving the rows
    behind would give a reader a window that reads unlabeled while still
    carrying labels, and the labeling loop replaces them per window anyway."""
    n = con.execute(
        "SELECT COUNT(*) FROM conflicts WHERE labeled_at IS NOT NULL").fetchone()[0]
    con.execute("DELETE FROM conflict_signals")
    con.execute("UPDATE conflicts SET labeled_at = NULL, topic = NULL")
    con.commit()
    print(f"--relabel: cleared the labels on {n} conflict windows")


def _conflicts_ready(con):
    """Whether the conflict tables exist AND carry the resume column -- both
    are things create_schema self-heals, and conflict_detect is reachable
    without the ingest.py run that normally applies migrations."""
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflicts'"
    ).fetchone():
        return False
    return "labeled_at" in {row[1] for row in con.execute("PRAGMA table_info(conflicts)")}


def _build_parser():
    """The CLI's flag surface, extracted from main() so it is greppable and
    testable on its own."""
    parser = argparse.ArgumentParser(
        description="Deterministic conflict finder + eval-gated escalation "
                    "labels. Idempotent per chat; Ctrl-C commits progress.")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model for the label pass (default {CONFLICT_MODEL})")
    parser.add_argument("--limit", type=int, metavar="N", default=None,
                        help="scan at most N direct chats this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the scan scope and exit without LLM calls or writes")
    parser.add_argument("--max-label-turns", type=int, metavar="N", default=None,
                        help=f"cap LLM-labeled turns per conflict (default "
                             f"{CONFLICT_MAX_LABEL_TURNS}, peak-anchored); lower = "
                             f"fewer LLM calls, coarser labels")
    parser.add_argument("--budget-minutes", type=int, metavar="N", default=None,
                        help=f"wall-clock minutes this run may spend, detection "
                             f"included (default {CONFLICT_BUDGET_MINUTES} from "
                             f"CONFLICT_BUDGET_MINUTES). Labels the strongest windows "
                             f"first and stops at the deadline; rerun to continue down "
                             f"the ranking. 0 = no budget (label every window)")
    parser.add_argument("--relabel", action="store_true",
                        help="clear every window's labels (conflict_signals, topic and "
                             "the labeled_at resume stamp) before running, so the pass "
                             "starts over. Needed after changing what labeling produces "
                             "-- CONFLICT_MAX_LABEL_TURNS, CONFLICT_MIN_TURN_CHARS, the "
                             "eval gate -- since a stamped window is otherwise never "
                             "revisited. NOT the model-switch door: the conflict_model "
                             "guard runs BEFORE the clear (on every path, --no-llm "
                             "included), so --relabel --model <other> is refused with "
                             "the labels intact")
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--no-llm", action="store_true",
                       help="FAST path: write the deterministic conflict windows "
                            "but skip the per-turn escalation labeling (the only "
                            "LLM work) -- no conflict_signals, no topic, zero "
                            "Ollama calls")
    speed.add_argument("--label-top", type=int, metavar="N", default=None,
                       help="BUDGETED path: detect all windows, but label only the "
                            "N strongest (by signal count, corroboration, dislikes, "
                            "peak gap); weaker windows stay detected-but-unlabeled. "
                            "N counts UNLABELED windows, so a rerun labels the NEXT "
                            "N down the ranking rather than re-visiting the same "
                            "top N. The wall-clock budget still applies on top -- "
                            "pass --budget-minutes 0 for N as the only cap. "
                            "Bounds LLM calls to N * max-label-turns.")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.no_llm and args.budget_minutes is not None:
        parser.error("--budget-minutes has nothing to bound with --no-llm "
                     "(that path makes zero LLM calls)")
    # Rejected rather than clamped, both of them: a negative budget would clamp
    # to 0, which means UNBUDGETED here -- the unbounded run this tier exists to
    # prevent -- and a non-positive turn cap would select no turns at all while
    # still stamping every window processed, foreclosing them for good.
    if args.budget_minutes is not None and args.budget_minutes < 0:
        parser.error("--budget-minutes cannot be negative (0 disables the budget)")
    if args.max_label_turns is not None and args.max_label_turns < 1:
        parser.error("--max-label-turns must be at least 1 (use --no-llm to skip "
                     "labeling entirely)")

    # The clock starts here, not in the labeling loop: detection is part of what
    # the budget promises, so the deadline has to cover it.
    started = time.monotonic()
    model = args.model or CONFLICT_MODEL
    think = _resolve_think(model)
    # Not resolved at all under --no-llm: that path makes zero LLM calls, so
    # there is nothing for a budget to bound (the parser refuses
    # --budget-minutes there for exactly that reason) -- and resolving it would
    # let a negative CONFLICT_BUDGET_MINUTES refuse a run the knob cannot
    # affect. `minutes` stays 0 there, which the branch below ignores anyway.
    minutes, budget_source = ((0, None) if args.no_llm
                              else _resolve_budget_minutes(args.budget_minutes))
    max_turns = args.max_label_turns or CONFLICT_MAX_LABEL_TURNS
    # The env half of the same check the parser makes on --max-label-turns, and
    # for the same reason (Fix 6's rule: a knob is validated where it is
    # RESOLVED, not only where a flag supplies it). `int(os.getenv(...) or "6")`
    # reads "0" as 0, and a non-positive cap makes _select_turns return no turns
    # at all while label_top_conflicts still stamps every window it processes --
    # so one bad .env line latches the whole corpus as labeled having dispatched
    # nothing, and the summary blames the prescreen. Only reachable from the
    # env: the flag is already refused at parse time.
    if max_turns < 1:
        raise SystemExit(
            f"CONFLICT_MAX_LABEL_TURNS must be at least 1 (got {max_turns}); a "
            f"non-positive cap labels no turns while still stamping every "
            f"window as processed. Use --no-llm to skip labeling entirely.")
    con = _connect()
    try:
        # Self-heal a DB that predates the conflict tables or the labeled_at
        # resume column (create_schema is idempotent) -- conflict_detect is
        # reachable without re-running ingest. Skipped under --dry-run, which
        # promises no writes and never reads either (annotate._ensure_column
        # degrades the same way): create_schema is DDL plus a reaction rescan.
        if not args.dry_run and not _conflicts_ready(con):
            schema.create_schema(con)
            con.commit()
        # The guard runs BEFORE anything that can destroy a label, and both
        # things that can are below it: _clear_labels drops every
        # conflict_signals row in the table, and the detection pass's
        # _drop_conflicts deletes a superseded window's signals with the window
        # (committed per chat). The only guard call used to sit inside
        # label_top_conflicts / detect_conflicts, i.e. after both -- so
        # `--relabel --model <other>` destroyed every label the previous budgets
        # had bought and THEN refused to rebuild any of them, and a plain
        # mismatched-model run could drop the last conflicts row during
        # detection and leave the guard reading COUNT(conflicts) == 0, at which
        # point it accepted the new model over the wreckage.
        # --relabel is not the model-switch door -- the guard's own message
        # names clearing conflicts / reset_db.py for that -- it is the "what
        # labeling PRODUCES changed" door, so a mismatched model is refused
        # while the labels are still there, --no-llm included: that path skips
        # the labeling, not the deletion, and the model flag being inert there
        # is not a reason to let it wipe another model's labels.
        # record=False: the labeling pass still owns the conflict_model write,
        # so a run that labels nothing (interrupted detection, --no-llm) leaves
        # the key exactly as it found it. Idempotent -- the pass calls the guard
        # again with the same value.
        if not args.dry_run and (args.relabel or not args.no_llm):
            _check_model_guard(con, model, record=False)
        if args.relabel and not args.dry_run:
            _clear_labels(con)
        # Two-phase whenever there is LLM work to bound: fast deterministic
        # detection over all chats, then label the strongest windows under
        # whichever caps apply (a wall clock, a fixed N, or both). With neither
        # -- `--budget-minutes 0` and no --label-top -- the legacy single-phase
        # path labels every window as it detects it.
        if not args.no_llm and (minutes or args.label_top is not None):
            if not args.dry_run:
                budget = (f"{minutes:g} min budget ({budget_source})" if minutes
                          else "no budget (--budget-minutes 0)")
                top = f", top {args.label_top} windows" if args.label_top is not None else ""
                print(f"conflict labeling: {budget}, model {model}, think {think!r}, "
                      f"<= {max_turns} turns/window{top}")
            # Pre-seeded rather than read with .get(): these are the two facts
            # the labeling phase needs from the detection pass, and a stand-in
            # detect_conflicts that fills neither should degrade to the
            # unscoped, uninterrupted defaults rather than raise.
            detected = {"interrupted": False, "chat_ids": None}
            detect_conflicts(con, model, limit=args.limit, dry_run=args.dry_run,
                             label=False, max_label_turns=max_turns, stats=detected)
            if detected["interrupted"]:
                # detect_conflicts catches Ctrl-C and returns normally, so
                # without this the key that stopped the (cheap, deterministic)
                # detection pass would start a full budget of LLM traffic.
                print("Stopped before labeling; rerun to resume.")
            elif not args.dry_run:
                label_top_conflicts(con, model, n=args.label_top,
                                    max_label_turns=max_turns, think=think,
                                    deadline=started + minutes * 60 if minutes else None,
                                    # --limit scoped the detection pass, so it
                                    # has to scope the labeling pass too, or the
                                    # budget goes to chats this run excluded.
                                    # `is not None`, not truthiness: --limit 0
                                    # scans no chats, and falling back to the
                                    # corpus-wide default there would label the
                                    # whole table -- the exact failure this
                                    # scoping exists to prevent.
                                    chat_ids=(detected["chat_ids"]
                                              if args.limit is not None else None))
                print(f"  used {(time.monotonic() - started) / 60.0:.1f} minutes")
        else:
            detect_conflicts(con, model, limit=args.limit, dry_run=args.dry_run,
                             label=not args.no_llm, max_label_turns=max_turns,
                             think=think)
    finally:
        con.close()


if __name__ == "__main__":
    run_cli(main)


# The deterministic signal computation and the LLM labeling pass live in their
# own modules; re-exported so conflict_detect._window_qualifies /
# .label_top_conflicts (etc.) keep resolving for callers and the tests. Bottom
# imports: both modules read this module's knobs and helpers late, so the names
# above must already be bound.
from conflict_signals import (  # noqa: E402,F401
    _ACTIVE_TENSION_SIGNALS,
    _corpus_end,
    _corroboration,
    _daily_counts,
    _dt,
    _hours,
    _llm_valence,
    _load_dislike_dts,
    _load_messages,
    _load_sessions,
    _off_text,
    _reply_gaps,
    _session_latency,
    _session_signals,
    _volume_collapse,
    _window_qualifies,
)
from conflict_labeling import (  # noqa: E402,F401
    _band,
    _BudgetExhausted,
    _COLD_LOAD_SECONDS,
    _COLD_TURN_SECONDS,
    _CORROB_RANK,
    _conflict_strength,
    _insert_signal,
    _label_turns,
    _load_turn_rate,
    _MSG_CACHE_CHATS,
    _peak_message_id,
    _resolve_label,
    _save_turn_rate,
    _timed,
    _turn_rate_key,
    _TURN_RATE_SAFETY,
    _TurnRate,
    label_top_conflicts,
)

