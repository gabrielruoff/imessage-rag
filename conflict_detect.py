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
import bisect
import datetime
import json
import statistics
import time

import ollama
import progress
import schema
import stats
from config import (
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
from db import connect_default, get_state, run_cli, set_state
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
    """Whether a turn can carry an escalation signal at all. Under
    CONFLICT_MIN_TURN_CHARS (whitespace stripped) or with no alphanumeric
    character in it (emoji/punctuation/tapback-ish), the answer is always
    `neutral` -- and on this corpus ~70% of labels come back neutral anyway,
    each costing a whole request to say nothing. Read as a module global so a
    test can patch the threshold."""
    t = (text or "").strip()
    return len(t) >= CONFLICT_MIN_TURN_CHARS and any(ch.isalnum() for ch in t)


def _select_turns(seg, max_turns, peak_id=None):
    """The <= max_turns turns of one window worth spending LLM calls on, as
    [(message_id, is_from_me, text)] in CHRONOLOGICAL order (the labels read as
    a sequence downstream, so they are never emitted peak-outward).

    Two rules, in order:
      * the lexical prescreen (_is_labelable) drops turns that cannot carry a
        signal -- it mostly matters in the big windows, since the median window
        has only 4 candidate turns to begin with;
      * what is left is taken as a contiguous span CENTERED on the window's
        peak_gap_message_id -- the message left hanging, i.e. the escalation
        moment the deterministic finder actually keyed on. The old head slice
        (`[:max_turns]`) provably misses it: windows in this corpus run to
        1,940 candidate turns.
    A window with no peak id, or whose peak turn is not in this segment, falls
    back to the chronological head. When the peak turn is dropped by the
    prescreen the span still centers on WHERE it was, so the neighbours that
    survive are the ones around the escalation."""
    kept = [(i, m) for i, m in enumerate(seg) if _is_labelable(m["text"])]
    turns = [(m["id"], m["is_from_me"], m["text"]) for _i, m in kept]
    if max_turns is not None and max_turns <= 0:
        return []
    if max_turns is None or len(turns) <= max_turns:
        return turns

    anchor = next((n for n, (_i, m) in enumerate(kept) if m["id"] == peak_id), None)
    if anchor is None and peak_id is not None:
        # The peak turn itself was prescreened out (a long silence is often
        # answered by a one-word line): anchor on its POSITION instead, so the
        # span still lands on the escalation rather than on the window's head.
        seg_pos = next((i for i, m in enumerate(seg) if m["id"] == peak_id), None)
        if seg_pos is not None:
            anchor = sum(1 for i, _m in kept if i < seg_pos)
    if anchor is None:
        return turns[:max_turns]
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

def _check_model_guard(con, model):
    """Refuse to mix label models within one conflicts table (same shape as
    the chunk_annotations annotate_model guard). Records the chosen model."""
    prev = get_state(con, "conflict_model")
    if prev and prev != model:
        n = con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
        if n:
            print(
                f"conflicts holds {n} rows labeled by {prev}, but the model is "
                f"now {model}. Mixing label models skews the escalation labels "
                f"-- clear conflicts (or reset_db.py) to switch, or rerun with "
                f"--model {prev}."
            )
            raise SystemExit(1)
    set_state(con, "conflict_model", model)
    con.commit()


# ---------------------------------------------------------------------------
# Deterministic signal computation
# ---------------------------------------------------------------------------

# Signals that indicate active tension rather than mere quiet. one_sided_session
# and volume_collapse both just describe a low-activity stretch (and co-fire on
# any lull), so a window needs at least one of these to qualify -- see
# _window_qualifies.
_ACTIVE_TENSION_SIGNALS = frozenset({"latency_withdrawal", "dislike_run"})


def _window_qualifies(fired):
    """True iff a fired-signal set is a real conflict window: it must trip
    >= CONFLICT_MIN_SIGNALS distinct signals AND include at least one
    active-tension signal. one_sided_session and volume_collapse are not
    independent -- a single quiet, one-directional lull trips both -- so their
    pair is one phenomenon double-counted, not corroboration."""
    return len(fired) >= CONFLICT_MIN_SIGNALS and bool(fired & _ACTIVE_TENSION_SIGNALS)


def _dt(iso):
    return datetime.datetime.fromisoformat(iso)


def _hours(a, b):
    return (b - a).total_seconds() / 3600.0


def _corpus_end(con):
    row = con.execute("SELECT MAX(date) FROM messages WHERE date IS NOT NULL").fetchone()
    return _dt(row[0]) if row and row[0] else datetime.datetime.utcnow()


def _load_messages(con, chat_id):
    return con.execute(
        f"SELECT m.id, m.is_from_me, m.date, m.text FROM messages m "
        f"WHERE m.chat_id = ? AND {_REAL_WHERE} ORDER BY m.date, m.id",
        (chat_id,),
    ).fetchall()


def _load_sessions(con, chat_id):
    return con.execute(
        "SELECT id, first_date, last_date, n_messages, n_from_me, "
        "initiator_is_me, ender_is_me FROM sessions "
        "WHERE chat_id = ? ORDER BY first_date, id",
        (chat_id,),
    ).fetchall()


def _load_dislike_dts(con, chat_id):
    return [
        _dt(r[0]) for r in con.execute(
            "SELECT date FROM reactions WHERE chat_id = ? "
            "AND associated_message_type = ? AND date IS NOT NULL ORDER BY date",
            (chat_id, _DISLIKE_TAPBACK),
        ).fetchall()
    ]


def _reply_gaps(messages, corpus_end):
    """RAW reply gaps for the survival estimator. A reply is a sender flip;
    the gap (hours) is attributed to the message being answered (so a long
    inter-session withdrawal lands on the message left hanging). Adds one
    right-CENSORED observation for the final message, whose reply is still
    open at corpus end. Returns (dts, hours, censored, answered_ids), all
    date-sorted and parallel."""
    dts, hrs, cen, mids = [], [], [], []

    def _add(dt, hours, censored, mid):
        dts.append(dt)
        hrs.append(hours)
        cen.append(censored)
        mids.append(mid)

    for k in range(1, len(messages)):
        if messages[k]["is_from_me"] != messages[k - 1]["is_from_me"]:
            a = _dt(messages[k - 1]["date"])
            g = _hours(a, _dt(messages[k]["date"]))
            if g >= 0:
                _add(a, g, False, messages[k - 1]["id"])
    if messages:
        a = _dt(messages[-1]["date"])
        g = _hours(a, corpus_end)
        if g > 0:
            _add(a, g, True, messages[-1]["id"])
    return dts, hrs, cen, mids


def _session_latency(gap, first_dt, last_dt):
    """(fired, frac, baseline_frac, peak_gap_hours, peak_gap_message_id) for one
    session's attributed reply gaps vs its trailing-90-day baseline."""
    dts, hrs, cen, mids = gap
    q, frac_min = CONFLICT_LATENCY_Q_HOURS, CONFLICT_LATENCY_FRAC
    lo = bisect.bisect_left(dts, first_dt)
    hi = bisect.bisect_right(dts, last_dt)
    dur, c = hrs[lo:hi], cen[lo:hi]
    frac = stats.frac_over_threshold(dur, c, q) if dur else None
    peak_h = peak_id = None
    for i in range(lo, hi):
        if not cen[i] and (peak_h is None or hrs[i] > peak_h):
            peak_h, peak_id = hrs[i], mids[i]
    b0 = first_dt - datetime.timedelta(days=90)
    blo = bisect.bisect_left(dts, b0)
    bdur, bc = hrs[blo:lo], cen[blo:lo]
    bfrac = stats.frac_over_threshold(bdur, bc, q) if bdur else None
    baseline = bfrac or 0.0
    fired = frac is not None and frac >= frac_min and frac >= 3.0 * baseline
    return fired, frac, bfrac, peak_h, peak_id


def _session_signals(session, gap, dislike_dts):
    """The three per-session deterministic signals, as (fired_set, meta)."""
    first_dt, last_dt = _dt(session["first_date"]), _dt(session["last_date"])
    fired = set()
    l_fired, frac, bfrac, peak_h, peak_id = _session_latency(gap, first_dt, last_dt)
    if l_fired:
        fired.add("latency_withdrawal")

    n = bisect.bisect_right(dislike_dts, last_dt) - bisect.bisect_left(dislike_dts, first_dt)
    if n >= 2:
        fired.add("dislike_run")

    nm, nfm = session["n_messages"], session["n_from_me"]
    if nm >= 2 and (nfm == nm or nfm == 0):
        fired.add("one_sided_session")

    meta = {"latency_frac": frac, "latency_baseline_frac": bfrac,
            "peak_gap_hours": peak_h, "peak_gap_message_id": peak_id,
            "dislike_count": n}
    return fired, meta


def _daily_counts(messages):
    counts = {}
    for m in messages:
        day = m["date"][:10]
        counts[day] = counts.get(day, 0) + 1
    return counts


def _volume_collapse(daily, first_dt, last_dt, window_msgs):
    """(fired, ratio) -- window daily rate vs the trailing-8-week (56d) median
    active-day rate before the window start."""
    base_lo = (first_dt - datetime.timedelta(days=56)).date()
    base_hi = first_dt.date()
    base = [n for day, n in daily.items()
            if base_lo <= datetime.date.fromisoformat(day) < base_hi and n > 0]
    if not base:
        return False, None
    median = statistics.median(base)
    span_days = max(1.0, _hours(first_dt, last_dt) / 24.0)
    rate = window_msgs / span_days
    ratio = rate / median if median else None
    fired = median > 0 and rate <= (1.0 - CONFLICT_VOLUME_COLLAPSE) * median
    return fired, ratio


def _off_text(messages, first_dt, last_dt):
    """(flag, max_gap_hours) over the window's messages -- a gap this large
    means the exchange likely moved off-channel (forbids a resolution
    verdict downstream)."""
    seg = [m for m in messages if first_dt <= _dt(m["date"]) <= last_dt]
    max_gap = 0.0
    for k in range(1, len(seg)):
        max_gap = max(max_gap, _hours(_dt(seg[k - 1]["date"]), _dt(seg[k]["date"])))
    return max_gap > CONFLICT_OFF_TEXT_GAP_HOURS, round(max_gap, 1)


# ---------------------------------------------------------------------------
# Valence corroboration (source='llm' rows ONLY)
# ---------------------------------------------------------------------------

def _llm_valence(con, chat_id):
    """(chat_mean, rows) where rows = [(first_date, last_date, valence)] over
    chunk_annotations.source='llm' only. Distilled rows are ignored -- they
    can't corroborate (their emotions are rougher and their events empty)."""
    rows = con.execute(
        "SELECT ch.first_date, ch.last_date, a.valence "
        "FROM chunk_annotations a JOIN chunks ch ON ch.id = a.chunk_id "
        "WHERE ch.chat_id = ? AND a.source = 'llm' AND a.valence IS NOT NULL",
        (chat_id,),
    ).fetchall()
    vals = [r[2] for r in rows]
    return (statistics.fmean(vals) if vals else None), rows


def _corroboration(chat_mean, llm_rows, first_date, last_date):
    """'none' (no llm chunks overlap -> no valence rank boost), 'weak' (llm
    chunks overlap but no meaningful drop) or 'llm' (drop >= threshold), plus a
    boost flag. Overlap is a plain-string ISO compare."""
    overlap = [v for (cf, cl, v) in llm_rows if cf <= last_date and cl >= first_date]
    if not overlap:
        return "none", False, None
    drop = (chat_mean - statistics.fmean(overlap)) if chat_mean is not None else 0.0
    if drop >= CONFLICT_MIN_VALENCE_DROP:
        return "llm", True, round(drop, 3)
    return "weak", False, round(drop, 3)


# ---------------------------------------------------------------------------
# Label write path (eval-gated)
# ---------------------------------------------------------------------------

def _band(conf, floor):
    """Two confidence bands from the terciles of the admissible [floor, 1]
    range -- rendered so the model never sees a raw float. (The intended
    source is the cleared class's confidence terciles recorded in the gate
    verdict; the F4 seed verdict does not yet carry them, so we derive the
    cutpoints from the floor and record them in peak_signal_json.)"""
    span = max(1e-9, 1.0 - floor)
    upper = floor + 2.0 * span / 3.0
    return "likely" if conf >= upper else "tentative", round(upper, 3)


def _resolve_label(pred_class, four_way_cleared):
    """Collapse the four horsemen to the harsh/critical umbrella unless the
    full 4-way is gate-cleared; repair and the umbrella pass through."""
    if pred_class in HORSEMEN and not four_way_cleared:
        return _COLLAPSE_TARGET
    return pred_class


class _BudgetExhausted(Exception):
    """Raised by _label_turns when `should_stop` fires mid-window. A window is
    the unit that gets stamped labeled_at, so a half-labeled one must not be
    committed: the caller rolls back its partial signal rows and leaves the
    window unlabeled for the next run to buy whole."""


def _label_turns(con, conflict_id, turns, classify_fn, four_way_cleared,
                 should_stop=None):
    """Classify + gate each candidate turn, writing the survivors. Returns
    (n_written, survivor_label_classes). A class the F4 gate has not cleared
    writes ZERO rows (suppressed); a below-floor confidence is dropped; an
    explicit model abstain is stored (band NULL) but is not escalation.

    `should_stop` is the budgeted caller's per-call deadline check, consulted
    BEFORE each request; when it fires this raises _BudgetExhausted rather than
    returning a partial result, so the window-level contract (all of its turns
    or none of them) is not something each caller has to re-implement."""
    floor = CONFLICT_LABEL_CONF_FLOOR
    n_written = 0
    survivors = []
    for mid, is_from_me, text in turns:
        if should_stop is not None and should_stop():
            raise _BudgetExhausted
        pred = classify_fn(text)
        cls, top2, conf = pred["class"], pred["top2"], pred["confidence"]
        excerpt = (text or "")[:200]
        if cls == "neutral":
            continue
        if cls == "abstain":
            _insert_signal(con, conflict_id, mid, is_from_me, "abstain", top2, None, excerpt)
            n_written += 1
            continue
        if conf < floor:
            continue
        label = _resolve_label(cls, four_way_cleared)
        if not eval_gate_passed(con, GATE_TASK, label):
            continue
        band, _cut = _band(conf, floor)
        _insert_signal(con, conflict_id, mid, is_from_me, label, top2, band, excerpt)
        survivors.append(label)
        n_written += 1
    return n_written, survivors


def _insert_signal(con, conflict_id, message_id, actor_is_me, label_class, top2, band, excerpt):
    con.execute(
        "INSERT INTO conflict_signals(conflict_id, message_id, actor_is_me, "
        "label_class, top2_labels_json, band, excerpt) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conflict_id, message_id, int(bool(actor_is_me)), label_class,
         json.dumps(top2), band, excerpt[:200]),
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
            _n_labels, survivors = _label_turns(con, conflict_id, turns, classify_fn,
                                                four_way_cleared)
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
# Budgeted labeling: label only the N strongest already-detected windows
# ---------------------------------------------------------------------------

# corroboration -> rank, so an llm-corroborated window outranks a bare one at
# equal signal count (mirrors the corroboration ladder in _corroboration).
_CORROB_RANK = {"llm": 2, "weak": 1, "none": 0}


def _peak_message_id(peak_signal_json):
    """The window's peak_gap_message_id -- the message left hanging, which is
    what _select_turns anchors on -- read back out of the stored evidence."""
    try:
        return json.loads(peak_signal_json or "{}").get("peak_gap_message_id")
    except (ValueError, TypeError):
        return None


def _conflict_strength(peak_signal_json, corroboration):
    """Rank key for a detected window (higher = label first): signal count, then
    valence corroboration, then dislike volume, then peak reply gap. Reads the
    deterministic evidence the finder already stored in peak_signal_json."""
    try:
        ps = json.loads(peak_signal_json or "{}")
    except (ValueError, TypeError):
        ps = {}
    return (
        ps.get("signal_count", 0),
        _CORROB_RANK.get(corroboration, 0),
        ps.get("dislike_count", 0),
        ps.get("peak_gap_hours") or 0.0,
    )


# --- the wall-clock rate estimator -----------------------------------------
# Per-CALL seconds, not per-group: this loop dispatches one turn at a time and
# gates before every request, so the whole apparatus annotate needs for its
# pooled fan-out (drain reserve, thread-safe counters) has nothing to do here.
# Measured 2026-07-28 on qwen3.5:4b, real turns (median 30 chars): 2.2 s mean /
# 2.0 s median with think=False.
_COLD_TURN_SECONDS = 5.0
# The model's first request of a run pays its load: ~54 s measured for
# qwen3.5:4b. Reserved once, so a small budget declines the run instead of
# overrunning it by a minute, and never folded into the rate -- banking it
# would tell the NEXT run that every call costs a cold load.
_COLD_LOAD_SECONDS = 60.0
# Planning margin over the measured mean. Turn latency scales with turn length
# and the mean is drawn from turns far shorter than the longest ones a window
# holds. Applied on top of the mean rather than banking the worst call (which
# is what annotate does): one timed-out request would otherwise freeze
# CONFLICT_TIMEOUT seconds into ingest_state as "the rate" and every later run
# would decline to dispatch anything. The reason annotate cannot do this is the
# blind wave its pool commits before any timing exists; a sequential loop that
# re-checks the clock before every call has no such wave to protect against.
_TURN_RATE_SAFETY = 1.5


# Chats whose messages the labeling loop keeps resident at once. The ranking is
# strength-ordered, so consecutive windows often share a chat (the cache's whole
# purpose) but the run as a whole walks hundreds of them.
_MSG_CACHE_CHATS = 16


def _turn_rate_key(model):
    return f"conflict_turn_seconds:{model}"


class _TurnRate:
    """Running mean of per-call wall-clock, priming the deadline arithmetic.

    `prior` is what to plan with before this run has measured anything (what an
    earlier run banked, else _COLD_TURN_SECONDS). The first completed call is
    deliberately NOT recorded: it carries the model's cold load, and folding
    that into the mean would make one run's startup cost look like every
    future call's price."""

    def __init__(self, prior=None):
        self.prior = prior or _COLD_TURN_SECONDS
        self.warm = False
        self._n = 0
        self._total = 0.0

    def record(self, seconds):
        if not self.warm:
            self.warm = True        # the cold-load call: charged, never counted
            return
        self._n += 1
        self._total += seconds

    def measured(self):
        """Mean seconds per call, or None when nothing countable has run."""
        return self._total / self._n if self._n else None

    def per_call(self):
        return (self.measured() or self.prior) * _TURN_RATE_SAFETY

    def next_call(self):
        """per_call plus the one-off model load this run has not paid yet."""
        return self.per_call() + (0.0 if self.warm else _COLD_LOAD_SECONDS)

    def window(self, n_turns):
        """What labeling a whole window of `n_turns` should cost."""
        return self.next_call() + max(0, n_turns - 1) * self.per_call()


def _load_turn_rate(con, model):
    """A _TurnRate primed with what an earlier run measured for this model."""
    raw = get_state(con, _turn_rate_key(model))
    try:
        prior = float(raw) if raw else None
    except (TypeError, ValueError):
        prior = None
    return _TurnRate(prior=prior if prior and prior > 0 else None)


def _save_turn_rate(con, rate, model):
    """Bank what this run measured so the next one is not blind. Only a
    MEASURED value: writing the prior back would freeze a guess into the DB
    where nothing could ever correct it."""
    mean = rate.measured()
    if not mean:
        return
    set_state(con, _turn_rate_key(model), repr(mean))
    con.commit()


def _timed(classify_fn, rate, monotonic):
    """classify_fn wrapped so every SUCCESSFUL call feeds the rate estimator.

    A call that raises is deliberately not recorded. Its duration is not
    evidence about what a working call costs -- an ollama timeout contributes
    CONFLICT_TIMEOUT (300 s by default) -- and because the estimate is banked
    for the NEXT run, recording it would latch: a prior that large makes the
    gate decline every window, a run that dispatches nothing measures nothing,
    and a run that measures nothing never overwrites the value that stopped it.
    A malformed REPLY is a different thing and still counts: _classify_turn
    returns an abstain for it, so the request really did cost what it cost."""
    def run(text):
        started = monotonic()
        result = classify_fn(text)
        rate.record(monotonic() - started)
        return result
    return run


def label_top_conflicts(con, model, n=None, max_label_turns=CONFLICT_MAX_LABEL_TURNS,
                        classify_fn=None, think=None, deadline=None,
                        monotonic=time.monotonic, rate=None, chat_ids=None):
    """Second pass over already-detected windows: label the strongest first (by
    _conflict_strength), under a wall-clock deadline, a fixed count, or both.

      * `n` None labels every unlabeled window; an int stops after that many
        (the `--label-top N` contract: weaker windows stay detected-but-
        unlabeled).
      * `deadline` is an absolute time on the injected `monotonic` clock (None
        = unbounded). Before each window the loop asks whether the estimated
        cost of ITS turns still fits; before each call inside the window it
        asks again, and a window that runs out mid-way is rolled back rather
        than committed half-labeled. Nothing is truncated silently -- the
        summary says how many windows were left.
      * `chat_ids` restricts the candidates to those chats. None means the
        whole table, which is what a full run wants -- but a --limit run
        scanned only some chats, and without this it would spend its budget
        labeling windows in chats it was told not to touch.
      * Windows already carrying `labeled_at` are skipped, so a rerun continues
        down the ranking wherever the last budget ran out. Every processed
        window is stamped, including one where no label survived the gate.

    Commits per window (Ctrl-C safe) and banks the observed per-call seconds
    for the next run."""
    _check_model_guard(con, model)
    if think is None:
        think = _resolve_think(model)
    if classify_fn is None:
        classify_fn = lambda text: _classify_turn(text, model, think)  # noqa: E731
    if rate is None:
        rate = _load_turn_rate(con, model)
    four_way_cleared = all(eval_gate_passed(con, GATE_TASK, h) for h in HORSEMEN)
    rows = con.execute(
        "SELECT id, chat_id, first_date, last_date, peak_signal_json, "
        "corroboration, labeled_at FROM conflicts"
    ).fetchall()
    if chat_ids is not None:
        in_scope = set(chat_ids)
        rows = [r for r in rows if r["chat_id"] in in_scope]
    if not rows:
        print("No conflicts to label.")
        return 0
    ranked = sorted(
        (r for r in rows if r["labeled_at"] is None),
        key=lambda r: _conflict_strength(r["peak_signal_json"], r["corroboration"]),
        reverse=True,
    )
    if n is not None:
        ranked = ranked[:max(0, n)]
    if not ranked:
        print(f"All {len(rows)} conflict windows are already labeled.")
        return 0

    def _out_of_budget(cost):
        return deadline is not None and monotonic() + cost > deadline

    timed = _timed(classify_fn, rate, monotonic)
    # Requests actually issued by THIS call -- the honest subject of the
    # "dispatched nothing" line. Not rate.warm: a caller can hand in a rate
    # that already paid a cold load, and not `labeled`: a window rolled back
    # at its per-call gate leaves labeled == 0 having spent real requests.
    calls = {"n": 0}

    def classify(text):
        calls["n"] += 1
        return timed(text)

    msg_cache = {}
    labeled = 0
    free = 0
    stopped = False
    try:
        with progress.track("Labeling conflicts", total=len(ranked), header=False) as p:
            for r in ranked:
                if deadline is not None and monotonic() >= deadline:
                    # Before the segment build, not just before the request:
                    # a window whose turns all fail the prescreen costs no LLM
                    # call but still loads its chat's messages, and the tail of
                    # the ranking holds enough of them to matter.
                    stopped = True
                    break
                cid = r["chat_id"]
                if cid not in msg_cache:
                    # Bounded: the ranking is by strength, not by chat, so a
                    # long run touches hundreds of chats and an unbounded cache
                    # would hold most of the corpus's message rows resident.
                    # Evicting the oldest is enough -- consecutive windows from
                    # one chat still hit, which is all the cache ever bought.
                    if len(msg_cache) >= _MSG_CACHE_CHATS:
                        msg_cache.pop(next(iter(msg_cache)))
                    msg_cache[cid] = _load_messages(con, cid)
                first_dt, last_dt = _dt(r["first_date"]), _dt(r["last_date"])
                seg = [m for m in msg_cache[cid]
                       if first_dt <= _dt(m["date"]) <= last_dt and m["text"]]
                peak_id = _peak_message_id(r["peak_signal_json"])
                turns = _select_turns(seg, max_label_turns, peak_id)
                if turns and _out_of_budget(rate.window(len(turns))):
                    stopped = True
                    break
                # A window labeled before conflicts.labeled_at existed carries
                # signals but no stamp, so this pass will re-label it: replace
                # its rows rather than stacking a second set on top.
                con.execute("DELETE FROM conflict_signals WHERE conflict_id = ?",
                            (r["id"],))
                try:
                    _n, survivors = _label_turns(
                        con, r["id"], turns, classify, four_way_cleared,
                        should_stop=lambda: _out_of_budget(rate.next_call()))
                except _BudgetExhausted:
                    # Partial window: drop its signal rows (nothing else is
                    # uncommitted -- the loop commits per window) and leave it
                    # unlabeled for the next run to buy whole.
                    con.rollback()
                    stopped = True
                    break
                con.execute("UPDATE conflicts SET topic = ?, labeled_at = ? WHERE id = ?",
                            (statistics.mode(survivors) if survivors else None,
                             _now_iso(), r["id"]))
                labeled += 1
                if not turns:
                    # Every turn failed the prescreen, so the model never saw
                    # this window. It is legitimately processed -- a rerun would
                    # find the same nothing -- but counting it silently with the
                    # windows that cost requests would overstate what the budget
                    # actually bought, which is the one number this tier owes.
                    free += 1
                con.commit()
                p.advance()
    except KeyboardInterrupt:
        # Rollback, not commit: every completed window committed on its own, so
        # the only uncommitted rows are a half-labeled window's -- and stamping
        # nothing means the next run buys it whole.
        con.rollback()
        print(f"\nInterrupted -- labeled {labeled} of {len(ranked)}; rerun to resume.")
        return labeled
    except BaseException:
        # Same contract for an ollama timeout or a transport RuntimeError as
        # for the deadline stop: a window is all of its turns or none of them.
        # Without this the _save_turn_rate commit below would persist the
        # half-labeled window on its way out, unstamped -- signals a reader
        # would treat as this window's complete labeling.
        con.rollback()
        raise
    finally:
        _save_turn_rate(con, rate, model)

    # Counted from the rows this call was scoped to, not from a fresh
    # corpus-wide COUNT: under `chat_ids` the two differ, and reporting the
    # whole table's backlog for a scoped run would be a different number than
    # the one the sentence promises.
    left = sum(1 for r in rows if r["labeled_at"] is None) - labeled
    # Windows the prescreen emptied are stamped but never reached the model, so
    # they are named apart from the ones the budget paid for.
    free_note = f" ({free} of them had no labelable turn and cost no request)" if free else ""
    if stopped:
        if not calls["n"]:
            # Declining every window is the right answer to a budget smaller
            # than one window, but silence would read as "nothing to do".
            print(f"  dispatched nothing: one window of up to {max_label_turns} turns "
                  f"is worth ~{rate.window(max_label_turns):.0f}s at the measured "
                  f"rate, more than the budget left -- raise CONFLICT_BUDGET_MINUTES")
        print(f"Labeled {labeled} conflict windows{free_note}, then stopped at the "
              f"deadline -- {left} of {len(rows)} still unlabeled; rerun to resume.")
    else:
        print(f"Labeled {labeled} of {len(rows)} conflict windows{free_note} "
              f"({left} left unlabeled)")
    return labeled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_budget_minutes(budget_minutes=None):
    """(minutes, provenance) for the run's wall-clock budget. Precedence:
    --budget-minutes > CONFLICT_BUDGET_MINUTES > the config default, the same
    caller > env > default order annotate_hybrid._resolve_budget_minutes uses.

    0 means UNBUDGETED in both places, where annotate's hybrid tier reads 0 as
    "unset": there the budget IS the tier, here it is additive, so switching it
    off has to restore the unbounded path rather than fall back to a default."""
    if budget_minutes is not None:
        return max(0, budget_minutes), "--budget-minutes"
    return max(0, CONFLICT_BUDGET_MINUTES), "CONFLICT_BUDGET_MINUTES"


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


def main(argv=None):
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
                             "revisited")
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
                            "The wall-clock budget still applies on top -- pass "
                            "--budget-minutes 0 for N as the only cap. "
                            "Bounds LLM calls to N * max-label-turns.")
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
    minutes, budget_source = _resolve_budget_minutes(args.budget_minutes)
    max_turns = args.max_label_turns or CONFLICT_MAX_LABEL_TURNS
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
