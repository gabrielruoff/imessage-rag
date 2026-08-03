"""Dispatch and wall-clock-budget machinery shared by the annotate tiers: the
bounded request pool, the per-phase group-latency estimator, and the deadline
gate that decides whether the next group can still be dispatched.

Split out of annotate.py; every name here is re-exported from `annotate`, so
`annotate._pooled` / `annotate._DispatchDeadline` (annotate_distill,
annotate_hybrid, the tests) keep resolving. Deliberately free of any annotate
import -- it needs nothing from the tier layer, only stdlib and db.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import get_float_state, set_float_state


def _pooled(items, request_fn, workers):
    """Yield (item, request_fn(item)) with up to `workers` requests in
    flight, in completion order. Only request_fn runs on worker threads --
    callers keep validation, SQLite writes and progress on the main thread
    (mirrors chunking.embed_chunks). By contract `items` is consumed on the
    caller's thread (the driving loop below iterates it), so a generator that
    runs SQL per item -- _term_groups does -- keeps its sqlite affinity.
    workers<=1 is a plain sequential loop.
    On interrupt, not-yet-started requests are cancelled; in-flight ones
    finish server-side but their results are dropped."""
    if workers <= 1:
        for item in items:
            yield item, request_fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        try:
            for item in items:
                futures[pool.submit(request_fn, item)] = item
                if len(futures) >= workers * 2:
                    done = next(as_completed(futures))
                    yield futures.pop(done), done.result()
            while futures:
                done = next(as_completed(futures))
                yield futures.pop(done), done.result()
        finally:
            for f in futures:
                f.cancel()


class _GroupRate:
    """Thread-safe running mean of per-group wall-clock. Deliberately OUTSIDE
    _DispatchDeadline so the estimator outlives the gate that uses it:
    _load_group_rate primes one PER PHASE with what an earlier run banked and
    _save_group_rate writes back what this run measured, so only a phase with
    nothing banked has to start from the cold constant. Phases do not share an
    estimator -- their per-group costs differ; see _load_group_rate.

    `prior` is what to estimate with BEFORE anything has completed: a value
    _load_group_rate read back from an earlier run (or the cold constant). A
    measured mean always wins over it; `mean()` stays measurement-only, so a
    caller can still tell the two apart."""

    def __init__(self, prior=None, cold_per_chunk=None, workers=1):
        self._lock = threading.Lock()
        self._n = 0
        self._total = 0.0
        self._worst = 0.0
        self.prior = prior
        # Set ONLY when `prior` is the cold constant's arithmetic rather than a
        # banked measurement, so denominate() can redo that arithmetic against
        # the packing the caller actually built and can never touch a number an
        # earlier run really measured.
        self._cold_per_chunk = cold_per_chunk
        self._workers = max(1, workers)

    def denominate(self, rows_per_group):
        """Re-scale a COLD prior to the rows a group actually holds. A no-op on
        a prior an earlier run banked and on a rate with no prior at all -- and
        moot once anything has completed, since estimate() takes the measured
        mean over the prior regardless.

        _load_group_rate has only `batch_size` when it builds the prior, and
        batch_size is a MAXIMUM: _annotation_groups packs by char budget too, so
        the real group is ~5 rows on the ~1,900-char event candidates and 1-2 on
        Apple FM. The dispatch loops know the true number for free -- they built
        the groups -- and this is where they hand it back. See
        _COLD_GROUP_SECONDS_PER_CHUNK for what the mis-denomination cost."""
        if not self._cold_per_chunk or not rows_per_group:
            return
        with self._lock:
            self.prior = (self._cold_per_chunk * max(1.0, rows_per_group)
                          * self._workers)

    def record(self, seconds):
        with self._lock:                    # worker threads; a paired update
            self._n += 1
            self._total += seconds
            self._worst = max(self._worst, seconds)

    def mean(self):
        """Mean seconds per group, or None when nothing has completed yet."""
        with self._lock:
            return self._total / self._n if self._n else None

    def worst(self):
        """Slowest group seen, or None when nothing has completed yet. This is
        what gets carried to the NEXT run: group latency is noisy (a batched
        reply that covers only some ids costs an extra request -- the schema
        retry, or a single-chunk request per miss when that fails too -- so the
        same phase measured a 106.4 s mean one run and 148 s the next), and the
        failure this whole gate exists to prevent is one-sided -- under-buying
        costs coverage a rerun recovers, overrunning breaks the only promise
        the tier makes."""
        with self._lock:
            return self._worst or None

    def estimate(self):
        """What to plan with: this run's measurement if there is one, else the
        prior, else None (nothing to go on)."""
        return self.mean() or self.prior


# Seconds of group LATENCY per chunk to assume when nothing has been measured
# and nothing was stored -- i.e. the first budgeted run against a given DB.
# reserve() divides latency by `workers`, so the group estimate is
# rate x rows-per-group x workers. Measured 2026-07-28 on the 25,029-chunk
# corpus (qwen3.5:4b, batch 8, ANNOTATE_WORKERS=4), slowest group per phase:
# 160.3 s seeding, 152.9-198.1 s on the event upgrades, 77.7-107.3 s on the
# confidence upgrades. 5.3 x 8 x 4 = 169.6 s covers the slowest two.
#
# Sized to the worst phase on purpose. It is a prior, not a prediction: it
# governs only until a phase's first group completes, _save_group_rate then
# replaces it per phase with this machine's own number, and an over-estimate
# makes a run buy LESS rather than overrun -- the safe direction for something
# whose whole contract is a wall clock.
#
# ROWS-PER-GROUP, not batch_size, is the denominator. batch_size is a MAXIMUM:
# _annotation_groups packs by char budget too, closing an Ollama group at ~5
# rows on the ~1,900-char event candidates and an Apple FM group at 1-2, so
# charging the prior for 8 rows a group measured a ~3.6x over-estimate that made
# phase 3a stop dispatching with most of its share unused. The dispatch loops
# therefore call _GroupRate.denominate with the mean rows-per-group of the
# packing they just built -- free, since they hold the groups already. The
# margin stays deliberately one-sided (~2.3x conservative at ~5 rows). See
# docs/DECISIONS.md, "annotation throughput, round 2".
_COLD_GROUP_SECONDS_PER_CHUNK = 5.3


def _mean_group_rows(groups):
    """Mean rows per group of a packing that has already been built -- the
    denominator _GroupRate.denominate wants, since _annotation_groups' char
    budget can close a group well before batch_size. 0.0 for an empty packing,
    which denominate() treats as "nothing to say".

    The MEAN is deliberately the less conservative choice: a packing's
    remainder group dilutes it below the typical dispatched group, where
    _COLD_GROUP_SECONDS_PER_CHUNK was calibrated against WORST groups.
    Affordable because the rescaled prior still measured ~2.3x over the banked
    worst group; if a budgeted first run ever overruns its wall clock, making
    this max(len(group)) is the first knob to reach for."""
    if not groups:
        return 0.0
    return sum(len(group) for group in groups) / len(groups)


def _group_rate_key(phase, model, batch_size, workers):
    return f"annotate_group_seconds:{phase}:{model}:b{batch_size}:w{workers}"


def _load_group_rate(con, phase, model, batch_size, workers):
    """A _GroupRate primed with what an earlier run measured for this exact
    (phase, model, batch, workers) shape.

    Without a prior, reserve() returns 0.0 until the first group completes and
    _pooled has by then committed 2x workers groups the gate cannot take back:
    measured 2026-07-28, a 2-min budget landed at 4.4 min and phase 3b was
    starved to zero. The gate cannot shrink that wave without changing
    _pooled's backlog rule for every tier -- but it can stop dispatching into
    it, which is exactly what having an estimate at group #1 buys.

    The key carries batch and workers because the reserve is denominated in
    group LATENCY, and both change it (4 concurrent requests against a
    serializing Ollama each measure ~4x a lone one). It carries `phase` because
    a run's phases have measurably different per-group costs -- their chunks are
    different populations, the event candidates averaging ~1,900 chars against
    the confidence upgrades' ~360 median, so the same batch_size buys a ~5-row
    group in one phase and a full 8 in the other -- and one shared estimator
    would let whichever phase ran first anchor the others wrong in both
    directions.

    `batch_size` in the key is a MAXIMUM, not the size of any particular group:
    _annotation_groups also packs to _OLLAMA_ANNOTATE_CHAR_BUDGET, so a
    long-chunk phase runs 4-6-row groups under a b8 key. That is fine and is
    what the key is for -- it identifies the PACKING CONFIG, and every group a
    run makes under one config is drawn from the same distribution, which is
    all a per-group estimate needs. Rates banked under an older packing
    self-heal rather than needing a migration: _save_group_rate stores the
    worst group of each run unconditionally, so the first run under new packing
    overwrites whatever the old one left.

    The COLD fallback cannot be so relaxed about it, because it is arithmetic
    rather than a measurement: batch_size is the only size it has here, so the
    returned rate carries the constant along and a caller that has built its
    groups calls _GroupRate.denominate with their real mean row count."""
    prior = get_float_state(con, _group_rate_key(phase, model, batch_size, workers))
    if prior:
        return _GroupRate(prior=prior)
    return _GroupRate(
        prior=_COLD_GROUP_SECONDS_PER_CHUNK * max(1, batch_size) * max(1, workers),
        cold_per_chunk=_COLD_GROUP_SECONDS_PER_CHUNK, workers=workers)


def _save_group_rate(con, rate, phase, model, batch_size, workers):
    """Persist what this run actually measured, so the next one is not blind.
    Only a MEASURED value is stored -- writing the prior back would freeze a
    cold guess into the DB where nothing would ever correct it -- and the
    slowest group rather than the mean, for the reason _GroupRate.worst gives.
    Within a run the live mean still wins, so the conservative number only
    governs the dispatches made before this run has measured anything."""
    set_float_state(con, _group_rate_key(phase, model, batch_size, workers),
                    rate.worst())


class _DispatchDeadline:
    """Deadline gate for a _pooled dispatch loop, sized so the groups still in
    flight when it stops dispatching DRAIN inside the deadline.

    Testing `now >= deadline` before each dispatch is not enough. _pooled holds
    up to 2x workers groups (one when sequential) and every one of them still
    finishes, is written and is committed after the generator stops yielding.
    Measured 2026-07-27 at ANNOTATE_WORKERS=4, batch 8, qwen3.5:4b: +1.5 min
    past a 5-min budget, +2.0 min past 2 min, +3.7 min past a 1-min apple
    budget -- minutes, not the "one group, tens of seconds" the first cut
    assumed. So the gate reserves that drain: the pool retires its backlog
    `workers` at a time, i.e. (in_flight / workers) waves of the mean per-group
    wall-clock of the groups that have ALREADY completed.

    _pooled submits 2x workers groups before it yields the first result, and
    those submissions all happen in the same instant, so a phase with nothing
    to estimate from commits to that much work whatever the clock says. That
    is what _GroupRate's `prior` is for: _load_group_rate hands the gate the
    seconds-per-group an earlier run measured for the same model/batch/workers
    (falling back to _COLD_GROUP_SECONDS_PER_CHUNK, which the dispatch loop then
    denominates in its packing's real rows-per-group), so reserve() is honest at
    group #1 and the wave is never committed blind. Each phase brings its OWN
    `rate` under its own key -- 3a's event groups and 3b's confidence groups
    are different populations, and one shared estimator would let whichever
    phase ran first anchor the other wrong (see _load_group_rate).

    `monotonic` is injected because the deadline and the gate must read the
    SAME clock -- the tier tests patch their module's `time` with a fake one
    that only advances when a fake LLM charges it.
    """

    def __init__(self, deadline, workers, monotonic=time.monotonic, rate=None):
        self.deadline = deadline
        self.workers = max(1, workers)
        self.stopped = False
        self.rate = rate if rate is not None else _GroupRate()
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._dispatched = 0
        self._finished = 0

    def timed(self, request_fn):
        """Wrap the per-group request so completions feed both the shared rate
        and THIS gate's outstanding count."""
        def run(group):
            started = self._monotonic()
            try:
                return request_fn(group)
            finally:
                self.rate.record(self._monotonic() - started)
                with self._lock:
                    self._finished += 1
        return run

    def reserve(self):
        """Seconds to drain what this gate still has in flight plus the group
        about to join it -- 0 only when the rate has neither a measurement nor
        a prior to plan with.

        Counting the ACTUAL backlog rather than _pooled's 2x workers ceiling
        matters at both ends: a phase starting with an empty pool would
        otherwise reserve two full waves before its very first group and buy
        nothing, while a phase in steady state is holding roughly that ceiling
        anyway. The estimate is over group LATENCY as observed under the run's
        own concurrency, which is what makes this self-correcting on a
        serializing Ollama: 4 concurrent requests each measure ~4x a lone one,
        and dividing by `workers` puts the throughput back."""
        mean = self.rate.estimate()
        if not mean:
            return 0.0
        with self._lock:
            outstanding = max(0, self._dispatched - self._finished) + 1
        # At least one wave: a lone group takes a full latency however many
        # workers are idle.
        return max(1.0, outstanding / self.workers) * mean

    def dispatchable(self, groups):
        """Yield from `groups` until the reserved drain would cross the
        deadline. _pooled consumes its items on the CALLER's thread (its
        documented contract), so a generator that simply stops yielding is
        exactly the semantics wanted: the clock is read immediately before a
        group is handed to the pool, and everything already dispatched still
        completes and commits."""
        for group in groups:
            if self._monotonic() + self.reserve() >= self.deadline:
                self.stopped = True
                return
            with self._lock:
                self._dispatched += 1
            yield group
