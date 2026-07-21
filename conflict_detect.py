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
     signals. Distilled/absent annotations NEVER block detection: valence only
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

The `conflict_model` ingest_state key guards against mixing label models in
one conflicts table (same shape as annotate_model / embed_model).
"""

import argparse
import bisect
import datetime
import json
import statistics

import ollama
import progress
import schema
import stats
from config import (
    CONFLICT_LABEL_CONF_FLOOR,
    CONFLICT_LATENCY_FRAC,
    CONFLICT_LATENCY_Q_HOURS,
    CONFLICT_MAX_LABEL_TURNS,
    CONFLICT_MIN_SIGNALS,
    CONFLICT_MIN_VALENCE_DROP,
    CONFLICT_MODEL,
    CONFLICT_OFF_TEXT_GAP_HOURS,
    CONFLICT_THINK,
    CONFLICT_TIMEOUT,
    CONFLICT_VOLUME_COLLAPSE,
    TARGET_DB,
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


def gate_classifier(con, items):
    """Production classifier wired into eval.evalkit.gates.TASKS for the
    owner-run `python -m eval.evalkit gate conflict_labels` path
    (signature classifier(con, items) -> {id: predicted_class}). Force-chooses
    one of the five gate classes for each chunk so kappa is well-defined;
    needs Ollama (owner-run, never in hermetic tests). Resolved lazily via its
    dotted-path string, so importing gates never pulls this module in."""
    out = {}
    for it in items:
        data = ollama.chat_once_json(
            [{"role": "system", "content": _GATE_SYSTEM},
             {"role": "user", "content": (it.get("text") or "")[:1500]}],
            model=CONFLICT_MODEL, think=CONFLICT_THINK, timeout=CONFLICT_TIMEOUT,
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


def _label_turns(con, conflict_id, turns, classify_fn, four_way_cleared):
    """Classify + gate each candidate turn, writing the survivors. Returns
    (n_written, survivor_label_classes). A class the F4 gate has not cleared
    writes ZERO rows (suppressed); a below-floor confidence is dropped; an
    explicit model abstain is stored (band NULL) but is not escalation."""
    floor = CONFLICT_LABEL_CONF_FLOOR
    n_written = 0
    survivors = []
    for mid, is_from_me, text in turns:
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


def detect_chat(con, chat_id, corpus_end, classify_fn, model, label=True,
                max_label_turns=CONFLICT_MAX_LABEL_TURNS):
    """Detect + write every conflict in one 1:1 chat (delete-then-rewrite, so
    re-runs are idempotent). Returns the number of conflicts written. With
    label=False the deterministic windows are still written but the per-turn
    escalation labeling (the only LLM work) is skipped -- no conflict_signals,
    no topic. max_label_turns caps the LLM-labeled turns per window."""
    sessions = _load_sessions(con, chat_id)
    messages = _load_messages(con, chat_id)
    # Idempotent rewrite: drop this chat's prior conflicts (+ their signals).
    old = [r[0] for r in con.execute(
        "SELECT id FROM conflicts WHERE chat_id = ?", (chat_id,)).fetchall()]
    if old:
        ph = ",".join("?" * len(old))
        con.execute(f"DELETE FROM conflict_signals WHERE conflict_id IN ({ph})", old)
        con.execute("DELETE FROM conflicts WHERE chat_id = ?", (chat_id,))
    if len(sessions) == 0 or len(messages) < 2:
        return 0

    gap = _reply_gaps(messages, corpus_end)
    dislike_dts = _load_dislike_dts(con, chat_id)
    daily = _daily_counts(messages)
    chat_mean, llm_rows = _llm_valence(con, chat_id)
    four_way_cleared = all(eval_gate_passed(con, GATE_TASK, h) for h in HORSEMEN)

    per_session = [_session_signals(s, gap, dislike_dts) for s in sessions]
    hot_flags = [bool(fired) for fired, _meta in per_session]

    n_written = 0
    for i, j in _consecutive_runs(hot_flags):
        first_date = sessions[i]["first_date"]
        last_date = sessions[j]["last_date"]
        first_dt, last_dt = _dt(first_date), _dt(last_date)
        fired = set().union(*(per_session[k][0] for k in range(i, j + 1)))

        window_msgs = sum(sessions[k]["n_messages"] for k in range(i, j + 1))
        v_fired, v_ratio = _volume_collapse(daily, first_dt, last_dt, window_msgs)
        if v_fired:
            fired.add("volume_collapse")
        if len(fired) < CONFLICT_MIN_SIGNALS:
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

        cur = con.execute(
            "INSERT INTO conflicts(chat_id, first_date, last_date, "
            "start_session_id, end_session_id, topic, peak_signal_json, "
            "corroboration, off_text_flag, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, first_date, last_date, sessions[i]["id"], sessions[j]["id"],
             None, json.dumps(peak_signal), corrob, int(bool(off_flag)), "llm"),
        )
        conflict_id = cur.lastrowid

        if label:
            seg = [m for m in messages
                   if first_dt <= _dt(m["date"]) <= last_dt and m["text"]]
            turns = [(m["id"], m["is_from_me"], m["text"]) for m in seg][:max_label_turns]
            _n_labels, survivors = _label_turns(con, conflict_id, turns, classify_fn,
                                                four_way_cleared)
            if survivors:
                topic = statistics.mode(survivors)
                con.execute("UPDATE conflicts SET topic = ? WHERE id = ?",
                            (topic, conflict_id))
        n_written += 1
    return n_written


def detect_conflicts(con, model, classify_fn=None, limit=None, dry_run=False,
                     label=True, max_label_turns=CONFLICT_MAX_LABEL_TURNS):
    """Sweep every 1:1 chat (via _restrict_to_direct_chats -- F3). classify_fn
    is the injectable label path (default: the real LLM); tests pass a stub.
    label=False is the deterministic-only fast path: windows are written but the
    per-turn LLM labeling is skipped (zero Ollama calls), so the model guard and
    classifier are never touched."""
    all_ids = [r[0] for r in con.execute("SELECT id FROM chats ORDER BY id").fetchall()]
    direct = _restrict_to_direct_chats(con, all_ids)
    if limit is not None:
        direct = direct[:limit]
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
            classify_fn = lambda text: _classify_turn(text, model, CONFLICT_THINK)  # noqa: E731

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


def label_top_conflicts(con, model, n, max_label_turns, classify_fn=None):
    """Second pass over already-detected windows: label ONLY the N strongest
    (by _conflict_strength), leaving weaker windows detected-but-unlabeled. This
    is the budgeted middle tier -- LLM calls are bounded by N * max_label_turns.
    Assumes a prior deterministic detection pass cleared stale signals (the
    per-chat delete-then-rewrite in detect_chat), so re-runs stay idempotent.
    Resumable within a run (commits per window, Ctrl-C safe)."""
    _check_model_guard(con, model)
    if classify_fn is None:
        classify_fn = lambda text: _classify_turn(text, model, CONFLICT_THINK)  # noqa: E731
    four_way_cleared = all(eval_gate_passed(con, GATE_TASK, h) for h in HORSEMEN)
    rows = con.execute(
        "SELECT id, chat_id, first_date, last_date, peak_signal_json, corroboration "
        "FROM conflicts"
    ).fetchall()
    ranked = sorted(
        rows,
        key=lambda r: _conflict_strength(r["peak_signal_json"], r["corroboration"]),
        reverse=True,
    )[:max(0, n)]
    if not ranked:
        print("No conflicts to label.")
        return 0
    msg_cache = {}
    labeled = 0
    try:
        with progress.track("Labeling top conflicts", total=len(ranked), header=False) as p:
            for r in ranked:
                cid = r["chat_id"]
                if cid not in msg_cache:
                    msg_cache[cid] = _load_messages(con, cid)
                first_dt, last_dt = _dt(r["first_date"]), _dt(r["last_date"])
                seg = [m for m in msg_cache[cid]
                       if first_dt <= _dt(m["date"]) <= last_dt and m["text"]]
                turns = [(m["id"], m["is_from_me"], m["text"]) for m in seg][:max_label_turns]
                _n, survivors = _label_turns(con, r["id"], turns, classify_fn, four_way_cleared)
                if survivors:
                    con.execute("UPDATE conflicts SET topic = ? WHERE id = ?",
                                (statistics.mode(survivors), r["id"]))
                labeled += 1
                con.commit()
                p.advance()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- labeled {labeled} of top {len(ranked)}; rerun to resume.")
        return labeled
    print(f"Labeled top {labeled} of {len(rows)} conflict windows "
          f"(<= {len(ranked) * max_label_turns} LLM calls budgeted)")
    return labeled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
                             f"{CONFLICT_MAX_LABEL_TURNS}); lower = fewer LLM calls, "
                             f"coarser labels")
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
                            "Bounds LLM calls to N * max-label-turns.")
    args = parser.parse_args(argv)

    model = args.model or CONFLICT_MODEL
    con = _connect()
    try:
        # Self-heal a DB that predates the conflict tables (create_schema is
        # idempotent) -- conflict_detect is reachable without re-running ingest.
        if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflicts'"
        ).fetchone():
            schema.create_schema(con)
        max_turns = args.max_label_turns or CONFLICT_MAX_LABEL_TURNS
        if args.label_top is not None:
            # Two-phase: fast deterministic detection over all chats, then label
            # only the N strongest windows (the rest stay detected-but-unlabeled).
            detect_conflicts(con, model, limit=args.limit, dry_run=args.dry_run,
                             label=False, max_label_turns=max_turns)
            if not args.dry_run:
                label_top_conflicts(con, model, args.label_top, max_turns)
        else:
            detect_conflicts(con, model, limit=args.limit, dry_run=args.dry_run,
                             label=not args.no_llm, max_label_turns=max_turns)
    finally:
        con.close()


if __name__ == "__main__":
    run_cli(main)
