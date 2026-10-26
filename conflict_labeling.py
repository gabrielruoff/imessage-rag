"""The LLM half of conflict analysis: banding a per-turn label, writing the
eval-gated signal rows, and the budgeted pass that labels the strongest
already-detected windows until a wall-clock deadline.

Split out of conflict_detect.py, which owns the CLI and the deterministic
detection sweep, and re-exports everything here so
`conflict_detect.label_top_conflicts` / `._TurnRate` (etc.) keep resolving --
including for the tests that patch label_top_conflicts on that module.

Knobs, prompts and the helpers detection also uses are read back through
`conflict_detect` rather than imported by value, so a test that patches one on
that module still moves this code.
"""

import json
import statistics
import time

import progress
from db import get_float_state, set_float_state
from eval.evalkit.gates import eval_gate_passed


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
    if pred_class in conflict_detect.HORSEMEN and not four_way_cleared:
        return conflict_detect._COLLAPSE_TARGET
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
    floor = conflict_detect.CONFLICT_LABEL_CONF_FLOOR
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
        if not eval_gate_passed(con, conflict_detect.GATE_TASK, label):
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
# Budgeted labeling: label only the N strongest already-detected windows
# ---------------------------------------------------------------------------

# corroboration -> rank, so an llm-corroborated window outranks a bare one at
# equal signal count (mirrors the corroboration ladder in _corroboration).
_CORROB_RANK = {"llm": 2, "weak": 1, "none": 0}


def _peak_message_id(peak_signal_json):
    """The window's peak_gap_message_id -- the message left hanging, which is
    what conflict_detect._select_turns anchors on -- read back out of the stored evidence."""
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
    return _TurnRate(prior=get_float_state(con, _turn_rate_key(model)))


def _save_turn_rate(con, rate, model):
    """Bank what this run measured so the next one is not blind. Only a
    MEASURED value: writing the prior back would freeze a guess into the DB
    where nothing could ever correct it."""
    set_float_state(con, _turn_rate_key(model), rate.measured())


def _timed(classify_fn, rate, monotonic):
    """classify_fn wrapped so every SUCCESSFUL call feeds the rate estimator.

    A call that raises is deliberately not recorded. Its duration is not
    evidence about what a working call costs -- an ollama timeout contributes
    CONFLICT_TIMEOUT (300 s by default) -- and because the estimate is banked
    for the NEXT run, recording it would latch: a prior that large makes the
    gate decline every window, a run that dispatches nothing measures nothing,
    and a run that measures nothing never overwrites the value that stopped it.
    A malformed REPLY is a different thing and still counts: conflict_detect._classify_turn
    returns an abstain for it, so the request really did cost what it cost."""
    def run(text):
        started = monotonic()
        result = classify_fn(text)
        rate.record(monotonic() - started)
        return result
    return run


def _rank_unlabeled(con, chat_ids, n):
    """(all in-scope windows, the unlabeled ones strongest-first) for the
    labeling pass. `n` caps the ranking, not the scan -- the slice is taken over
    the UNLABELED windows, so a rerun's same N buys the NEXT N down the ranking
    instead of revisiting stamped ones. Prints the reason and returns an empty
    ranking when there is nothing to do."""
    rows = con.execute(
        "SELECT id, chat_id, first_date, last_date, peak_signal_json, "
        "corroboration, labeled_at FROM conflicts"
    ).fetchall()
    if chat_ids is not None:
        in_scope = set(chat_ids)
        rows = [r for r in rows if r["chat_id"] in in_scope]
    if not rows:
        print("No conflicts to label.")
        return rows, []
    ranked = sorted(
        (r for r in rows if r["labeled_at"] is None),
        key=lambda r: _conflict_strength(r["peak_signal_json"], r["corroboration"]),
        reverse=True,
    )
    if n is not None:
        ranked = ranked[:max(0, n)]
    if not ranked:
        print(f"All {len(rows)} conflict windows are already labeled.")
    return rows, ranked


def _window_turns(con, row, msg_cache, max_label_turns):
    """The turns of one window worth sending to the model: its chat's messages
    (through a bounded cache), narrowed to the window's dates, then
    peak-anchored and capped by _select_turns."""
    cid = row["chat_id"]
    if cid not in msg_cache:
        # Bounded: the ranking is by strength, not by chat, so a long run
        # touches hundreds of chats and an unbounded cache would hold most of
        # the corpus's message rows resident. Evicting the oldest is enough --
        # consecutive windows from one chat still hit, which is all the cache
        # ever bought.
        if len(msg_cache) >= _MSG_CACHE_CHATS:
            msg_cache.pop(next(iter(msg_cache)))
        msg_cache[cid] = conflict_detect._load_messages(con, cid)
    first_dt = conflict_detect._dt(row["first_date"])
    last_dt = conflict_detect._dt(row["last_date"])
    seg = [m for m in msg_cache[cid]
           if first_dt <= conflict_detect._dt(m["date"]) <= last_dt and m["text"]]
    peak_id = _peak_message_id(row["peak_signal_json"])
    return conflict_detect._select_turns(seg, max_label_turns, peak_id)


def _report_labeling(rows, labeled, free, stopped, n_calls, rate, max_label_turns):
    """The pass's closing line: what was labeled, what it cost nothing, and what
    is left -- counted from the rows this call was SCOPED to, not from a fresh
    corpus-wide COUNT. Under `chat_ids` the two differ, and reporting the whole
    table's backlog for a scoped run would be a different number than the one
    the sentence promises."""
    left = sum(1 for r in rows if r["labeled_at"] is None) - labeled
    # Windows the prescreen emptied are stamped but never reached the model, so
    # they are named apart from the ones the budget paid for.
    free_note = f" ({free} of them had no labelable turn and cost no request)" if free else ""
    if stopped:
        if not n_calls:
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


def label_top_conflicts(con, model, n=None, max_label_turns=None,
                        classify_fn=None, think=None, deadline=None,
                        monotonic=time.monotonic, rate=None, chat_ids=None):
    """Second pass over already-detected windows: label the strongest first (by
    _conflict_strength), under a wall-clock deadline, a fixed count, or both.

      * `n` None labels every unlabeled window; an int stops after that many
        (the `--label-top N` contract: weaker windows stay detected-but-
        unlabeled). The slice is taken over the UNLABELED windows, so on a
        rerun the same N buys the next N down the ranking rather than
        re-visiting stamped ones.
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
    conflict_detect._check_model_guard(con, model)
    # Resolved here rather than as a default, joining `think` / `classify_fn` /
    # `rate` below. A default expression evaluates at DEF time, which both
    # forced this module to import conflict_detect from the top -- the one
    # thing that made it unimportable on its own -- and quietly broke the
    # late-binding this module's docstring promises: the value would freeze at
    # import, so patching conflict_detect.CONFLICT_MAX_LABEL_TURNS moved every
    # other read here except this one.
    if max_label_turns is None:
        max_label_turns = conflict_detect.CONFLICT_MAX_LABEL_TURNS
    if think is None:
        think = conflict_detect._resolve_think(model)
    if classify_fn is None:
        classify_fn = lambda text: conflict_detect._classify_turn(text, model, think)  # noqa: E731
    if rate is None:
        rate = _load_turn_rate(con, model)
    four_way_cleared = all(eval_gate_passed(con, conflict_detect.GATE_TASK, h)
                           for h in conflict_detect.HORSEMEN)
    rows, ranked = _rank_unlabeled(con, chat_ids, n)
    if not ranked:
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
                turns = _window_turns(con, r, msg_cache, max_label_turns)
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
                             conflict_detect._now_iso(), r["id"]))
                labeled += 1
                if not turns:
                    # No turn in this window holds a single alphanumeric
                    # character (all tapbacks/emoji), so the model never saw it.
                    # It is legitimately processed -- a rerun would find the
                    # same nothing -- but counting it silently with the windows
                    # that cost requests would overstate what the budget
                    # actually bought, which is the one number this tier owes.
                    # The read tools say the same thing on their side: a stamped
                    # window with zero signal turns is never narrated as a
                    # positive finding of calm (query_tools.conflict's
                    # _LABELED_EMPTY_NOTE).
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

    _report_labeling(rows, labeled, free, stopped, calls["n"], rate,
                     max_label_turns)
    return labeled

# Bottom import: this module is one half of a cycle -- `conflict_detect` re-exports the
# names below, so importing it from the TOP here makes conflict_labeling unimportable
# on its own (the parent would run its own re-export against a still-empty
# conflict_labeling). Down here every name above is already bound, so the cycle
# resolves whichever module a caller reaches first. Only used inside functions,
# so the late binding costs nothing.
import conflict_detect  # noqa: E402
