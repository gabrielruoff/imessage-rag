"""Hybrid annotation tier (annotate.py --quality hybrid): spend a fixed
WALL-CLOCK budget on the most valuable annotations instead of a fixed chunk
count. That is what makes the tier's runtime promise hold by construction --
on a slower machine or a bigger corpus it buys fewer upgrades, never a blown
deadline, and on a faster one it buys more coverage.

Four phases under ONE deadline:

  1  seed top-up      -- widen the LLM-labeled seed pool with a stratified,
                         embedding-diverse draw (annotate_sampling) rather than
                         distilled's random sample, so the kNN vote generalizes
  2  kNN classify     -- label the whole corpus from chunk_vectors, then
                         backfill knn_confidence onto distilled rows written
                         before that column existed
  3a event upgrades   -- LLM-re-annotate event-candidate chunks (annotate_events),
                         densest lexicon match first: events are the one label
                         kNN structurally CANNOT produce, since every
                         source='distilled' row carries events_json='[]'
  3b confidence upgrades -- LLM-re-annotate the lowest-knn_confidence distilled
                         rows, i.e. where the kNN vote is least likely right

Upgraded rows are written source='llm', so every valence-trusting reader
(conflict corroboration, the nudge Stage-0 exclusion, voiceprint tone) lights
up for exactly the chunks that matter.

Phases 1-2 are corpus-wide by design (the kNN vote has to generalize, so the
scope flags deliberately don't reach them). EVERY phase is deadline-gated, not
just the LLM ones: phase 1 is LLM work like phase 3 (measured unbounded it
consumed a whole 8-minute budget on its own), so it gets its own
_SEED_PHASE_SHARE slice of the clock, and phase 2 is deterministic but not
cheap -- its cost is O(rows x seeds), and on the same corpus and machine it
measured 14.6 s at a 450-seed pool against 82.0 s at a 4,416-seed one -- so it
runs against the budget itself.
Both are resumable, so stopping short only defers work to the next run. Only
phase 3 also honors --all/--top-dyads/--limit, like the LLM tiers.

annotate.py owns the CLI and calls hybrid_chunks; the shared request/
validation/write helpers stay in annotate and are resolved late (module-
attribute access), so the two modules can import each other safely.
"""

import time

import annotate
import annotate_distill
import annotate_events
import annotate_sampling
import config
import db
import progress

# Config is read through the module (config.X) rather than imported by name:
# the budget default depends on the model, which is only known per call, and
# the knobs stay patchable by a test that drives hybrid_chunks directly.

# Default wall-clock budget per backend when ANNOTATE_BUDGET_MINUTES is unset.
# Measured 2026-07-27 (docs/DECISIONS.md, 25,029-chunk corpus):
# sustained ~1.5-2.5 s/chunk on qwen3.5:4b (Ollama, batch 8, think=False) and
# ~3-5 s/chunk on Apple FM, whose fixed 4,096-token window forces small
# char-budgeted batches. The tier's hard success criterion is < 30 min on qwen
# and < 1 h on apple, so each default leaves room for the kNN pass plus the
# overshoot annotate._DispatchDeadline cannot remove: _pooled commits a blind
# wave of 2x ANNOTATE_WORKERS groups before any timing exists, ~2-3 min at
# batch 8 on this machine. Module constants, not config knobs -- exactly like
# annotate._APPLE_ANNOTATE_CHAR_BUDGET, these are properties of the tier and
# the backend, and the user-facing lever (ANNOTATE_BUDGET_MINUTES) exists.
_DEFAULT_BUDGET_MINUTES_OLLAMA = 25
_DEFAULT_BUDGET_MINUTES_APPLE = 50

# Share of the WHOLE budget Phase 1 may spend topping up the seed pool.
# Measured 2026-07-27 on the 25,029-chunk corpus, qwen3.5:4b: unbounded and one
# chunk per request, a 350-seed top-up cost 641.9 s (1.83 s/chunk) -- 10.7 min
# of an 8-min budget, phases 3a/3b never reaching a single LLM call, and a
# fresh 800-seed pool extrapolating to ~24 min of the 25-min default. Batching
# it (batch 8, 4 workers) only buys ~20%: 1.46 s/chunk measured, because this
# Ollama serializes and the seeds are short (median 293 chars), so the bigger
# prompt saves round-trips it was never really spending. Batching alone
# therefore cannot fix this -- hence the cap.
#
# The tier exists for the UPGRADE phases, so seeds take a minority slice.
# 0.25 x 25 min = 6.2 min ~= 257 seeds per run at the measured rate: enough to
# clear annotate_distill._KNN_MIN_SEEDS (100) on the very first run of a fresh
# corpus, so the kNN vote works immediately, while a full 800-seed pool
# accumulates over ~3 runs and >= 75% of every budget stays on upgrades. Apple
# FM is slower per seed, so its 50-min budget's 12.5 min buys a partial top-up
# and the next run continues it -- an honest degrade, which is the whole point
# of budgeting in wall clock.
_SEED_PHASE_SHARE = 0.25

# Share of the time left after Phase 2 reserved for event upgrades (3a) before
# confidence upgrades (3b) take the rest. Events go first because they are the
# thing kNN cannot produce at all, so 3a buys coverage no amount of extra
# classification would; the remainder goes where kNN is most likely to be
# WRONG. 3a hands unused time back -- when the candidates run out early, 3b
# simply inherits the balance.
#
# The consequence to know about is the floor underneath both phases: a phase
# whose share is under one group's LATENCY buys exactly one group (the probe in
# annotate._DispatchDeadline.dispatchable) and overruns its share by that
# group. Measured 2026-07-28 (qwen3.5:4b, batch 8, 4 workers) after the char
# packing and the schema retry landed, an event group -- 4-6 chunks, not 8, at
# ~1,900 chars each -- banked at 46.7 s of latency. The 25-min default is far
# clear of the floor; the old behavior on a budget that small was to commit a
# blind wave anyway and take 4.4 min over a 2-min one.
_EVENT_PHASE_SHARE = 0.60

# Ids per `WHERE id IN (...)` slice. A corpus-wide event prefilter can name
# thousands of chunks, past SQLite's default 999-variable ceiling.
_ID_SLICE = 500

# Cap on the phase-3b candidate rows fetched in one run. The fetch carries
# every row's full chunk TEXT (measured 14,995 rows / 9.5 MB / ~26 MB resident
# on the reference corpus) and the budget can only ever dispatch a few hundred
# of them: at the fastest rate measured for this phase (2.2 s/chunk end to
# end) even an hour buys ~1,600. 5,000 is that with room to spare, and the rows are taken
# lowest-confidence-first, so the cap only ever drops rows the run was never
# going to reach.
_LOW_CONF_FETCH_CAP = 5000

# The same cap for phase 3a, sized on 3a's own arithmetic rather than copied
# from 3b's -- the two phases fetch very different rows. An event candidate
# averages ~1,900 chars (against the confidence rows' ~360 median), so
# _OLLAMA_ANNOTATE_CHAR_BUDGET closes a group at 4-6 of them, and measured
# 2026-07-28 (qwen3.5:4b, batch 8, ANNOTATE_WORKERS=4) that group banked a
# 46.7 s worst latency. Four workers retire a wave of 4 groups per 46.7 s, i.e.
# ~24 chunks / 46.7 s ~= 0.5 chunks/s. An hour spent ENTIRELY in this phase is
# therefore ~1,850 upgrades -- and no run spends an hour here at the DEFAULT
# budget, since 3a gets _EVENT_PHASE_SHARE of 25 min (~460). 2,000 covers even
# the whole-hour case. That sizing is against the default, not a law:
# ANNOTATE_BUDGET_MINUTES has no ceiling, so past ~111 budgeted minutes
# (2,000 / 0.5 chunks/s / 0.60) this cap rather than the clock is what stops
# 3a. Which is precisely why it prints what it left instead of letting the
# deadline take the credit -- the printed line names the cap as the cause.
# Why it has to exist at all: _chunk_rows materializes every id's full chunk
# TEXT, and the prefilter's ~19-20% hit rate means a corpus 10x the reference
# yields ~50k candidates -- ~95 MB of text held to dispatch a few hundred rows.
# The cap is a PREFIX, which is what makes it lossless in practice:
# annotate_events hands the ids over ranked by pattern-hit density, so the rows
# it drops are exactly the ones a deadline was never going to reach anyway --
# the same argument _LOW_CONF_FETCH_CAP's ordering makes. Dropped rows are
# still printed, never silently discarded.
# A module constant, not a config knob, for the same reason
# _DEFAULT_BUDGET_MINUTES_* are: it is a memory ceiling derived from this
# phase's own measured throughput, not a preference, and the user-facing lever
# over how much 3a fetches is ANNOTATE_BUDGET_MINUTES. Same for
# _LOW_CONF_FETCH_CAP.
_EVENT_FETCH_CAP = 2000

# ingest_state key: how many seeds THIS tier's sampler has drawn. Not
# COUNT(source='llm'), because phase 3 writes source='llm' too -- counting its
# own upgrades made the tier satisfy its own seed target, so on the reference
# corpus (4,416 'llm' rows) the target and the whole stratified sampler were
# permanently inert. An ingest_state counter rather than a new `source` value:
# every valence-trusting reader in the repo keys on source='llm', and a third
# value would mean touching all of them to keep seeds visible. reset_db.py
# clears ingest_state wholesale, so a reset zeroes this with the annotations.
_SEED_COUNTER_KEY = "hybrid_seeds"


def _resolve_budget_minutes(model, budget_minutes=None):
    """Wall-clock minutes this run may spend, plus a short provenance string
    for the header. Precedence: an explicit argument > ANNOTATE_BUDGET_MINUTES
    > the per-backend default. 0 means "unset" in both the argument and the env
    -- there is no such thing as a zero-minute run of this tier -- matching the
    empty-means-default convention config uses for its other int knobs.

    A NEGATIVE value is refused rather than passed through or clamped. Passed
    through it resolves to a deadline already in the past, which every phase
    then declines: the run reports "the budget was spent on phases 1-2" and
    exits 0, indistinguishable from a machine too slow to afford one request.
    Clamping to 0 would be worse still -- 0 means "unset" here, so a typo'd
    -25 would silently buy the 25-minute default. Refusing names the knob,
    mirroring conflict_detect's --budget-minutes check. The env is validated
    only where it is actually read, so an explicit argument still overrides a
    bad env exactly as the precedence above says it does."""
    apple = config.is_apple_model(model)
    return db.resolve_budget_minutes(
        budget_minutes, config.ANNOTATE_BUDGET_MINUTES,
        _DEFAULT_BUDGET_MINUTES_APPLE if apple else _DEFAULT_BUDGET_MINUTES_OLLAMA,
        zero_means="unset",
        flag_name="caller", env_name="ANNOTATE_BUDGET_MINUTES",
        default_name="apple default" if apple else "ollama default",
        flag_error=("budget_minutes cannot be negative (got {value}); 0 means "
                    "'unset' -- the per-backend default"),
        env_error=(f"ANNOTATE_BUDGET_MINUTES cannot be negative (got "
                   f"{{value}}); leave it empty or 0 for the "
                   f"per-backend default ({_DEFAULT_BUDGET_MINUTES_OLLAMA} min on "
                   f"Ollama, {_DEFAULT_BUDGET_MINUTES_APPLE} on Apple FM)"),
    )


def _scope_clause(all_chats, top_dyads):
    """The `AND c.chat_id IN (top dyads)` fragment and its params, or ('', [])
    for --all. Same subquery shape as annotate._pending_chunks, so "top N
    chats" means the same thing in every tier."""
    if all_chats:
        return "", []
    return ("""  AND c.chat_id IN (
                SELECT chat_id FROM messages WHERE chat_id IS NOT NULL
                GROUP BY chat_id ORDER BY COUNT(*) DESC LIMIT ?)
        """, [top_dyads])


# Phase 3b's candidate set: distilled rows are the only upgradable ones ('llm'
# rows are already at full fidelity, and re-sending them would spend the budget
# on work already done).
_LOW_CONF_FROM = ("FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id "
                  "WHERE a.source = 'distilled'")

# NULL confidence sorts LAST but stays eligible: a distilled row whose chunk has
# no embedding cannot be scored (backfill_confidence skips it), and excluding it
# would make those chunks permanently unreachable by this tier. They are the
# least informative choice, not an invalid one.
_LOW_CONF_ORDER = " ORDER BY a.knn_confidence IS NULL, a.knn_confidence"


def _low_confidence_rows(con, all_chats, top_dyads, limit=None,
                         by_confidence=True):
    """(chunk_id, text) for the in-scope distilled rows the kNN vote was least
    sure about, least-confident first. `by_confidence=False` drops to chunk-id
    order for the one caller that can meet a DB where the column does not exist
    yet (--dry-run, which no longer migrates)."""
    clause, params = _scope_clause(all_chats, top_dyads)
    order = _LOW_CONF_ORDER if by_confidence else " ORDER BY a.chunk_id"
    sql = f"SELECT c.id, c.text {_LOW_CONF_FROM}{clause}{order}"
    if limit is not None:
        sql += " LIMIT ?"
        params = params + [limit]
    return con.execute(sql, params).fetchall()


def _low_confidence_count(con, all_chats, top_dyads):
    """How many rows _low_confidence_rows could return -- for --dry-run, which
    wants the number without materializing every chunk's text."""
    clause, params = _scope_clause(all_chats, top_dyads)
    return con.execute(f"SELECT COUNT(*) {_LOW_CONF_FROM}{clause}", params).fetchone()[0]


def _chunk_rows(con, chunk_ids):
    """(chunk_id, text) for `chunk_ids`, in the CALLER'S order -- the row shape
    annotate._annotate_group wants, and a ranking this must not destroy:
    annotate_events hands over its candidates densest-first, and the phase
    deadline stops partway down, so a re-sort here would silently hand the
    budget back to chunk-id order. The fetch still goes out in parameter-safe
    _ID_SLICE blocks (a corpus-wide prefilter names thousands of ids, past
    SQLite's 999-variable ceiling), which means SQL row order is per-block and
    meaningless; the rows are put back into the input's sequence in python.

    Two edge cases, both deliberate. An id with no `chunks` row is omitted
    rather than faked: there is no text to annotate, and a None-text row would
    reach the LLM as an empty prompt and burn a request to come back
    unparseable. A repeated id yields ONE row, at its first position -- the
    tier's whole contract is a wall-clock budget, so paying twice for the same
    chunk is never the right reading of a duplicate."""
    by_id = {}
    for i in range(0, len(chunk_ids), _ID_SLICE):
        block = list(chunk_ids[i:i + _ID_SLICE])
        placeholders = ",".join("?" * len(block))
        for row in con.execute(
                f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", block):
            by_id[row[0]] = row
    # pop, not [], so a duplicate id contributes only its first occurrence.
    return [row for row in (by_id.pop(cid, None) for cid in chunk_ids)
            if row is not None]


def _n_seeds(con):
    """LLM seeds the kNN vote can actually use -- source='llm' rows that HAVE
    an embedding, i.e. exactly what annotate_distill._build_seed_index counts
    before refusing. 0 when chunk_vectors is missing, which is a different
    failure (no embeddings at all) and must not be mistaken for a short seed
    pool; the caller checks for the table itself."""
    if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone():
        return None
    return con.execute("""
        SELECT COUNT(*) FROM chunk_annotations a
        JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
        WHERE a.source = 'llm'
    """).fetchone()[0]


def _run_upgrades(con, rows, model, think, workers, batch_size, deadline,
                  description, seen, stats, rate, phase):
    """LLM-re-annotate `rows` ((chunk_id, text)) until they run out or
    `deadline` passes, writing source='llm'. Mirrors annotate_chunks' dispatch
    loop -- same grouping, same _pooled fan-out, same per-row write and commit
    cadence -- differing only in the deadline gate and the `seen` dedup set.

    The gate is annotate._DispatchDeadline, which stops dispatching a drain's
    worth of time BEFORE the deadline rather than at it: everything already
    handed to the pool still finishes and commits, and at ANNOTATE_WORKERS=4
    that backlog measured minutes, not "one group". `rate`/`phase` are this
    phase's estimator and its ingest_state key: annotate._load_group_rate
    primes it with what an earlier run measured for the same phase, so the
    first dispatch is not blind, and the finally banks what this one measured.
    Both are required, not defaulted -- an unprimed default _GroupRate would
    make reserve() 0.0 and let the gate commit a blind wave, which is exactly
    the failure the estimator exists to prevent.

    A COLD prior is re-denominated here, one line after the packing exists,
    because that is the first moment anyone knows how many rows a group of THIS
    phase's chunks actually holds -- _load_group_rate had only batch_size, which
    is a ceiling the char budget routinely undercuts (see
    annotate._COLD_GROUP_SECONDS_PER_CHUNK). It costs a sum over groups and
    touches nothing that was ever measured.

    Counters go into the caller-owned `stats` dict (done/skipped/stopped)
    rather than a return value, so a KeyboardInterrupt propagating out of here
    still leaves the caller able to print an accurate per-phase summary."""
    groups = annotate._annotation_groups(rows, model, batch_size)
    gate = annotate._DispatchDeadline(deadline, workers, time.monotonic, rate)
    gate.rate.denominate(annotate._mean_group_rows(groups))

    try:
        # total=None deliberately: the deadline, not the candidate list, decides
        # where this phase stops, and a total would promise an ETA for work that
        # was never going to run (a stopped phase left the bar at 72/2567). The
        # phase's own header line already printed the candidate count.
        with progress.track(description, total=None if rows else 0,
                            header=False) as p:
            for group, results in annotate._pooled(
                    gate.dispatchable(groups),
                    gate.timed(lambda g: annotate._annotate_group(
                                   g, model, think, fail_fast=True),
                               bankable=annotate._group_bankable),
                    workers):
                failed = getattr(results, "failed", False)
                if failed:
                    print(f"  group of {len(group)}: request failed "
                          f"(timeout/transport), rows retried next run")
                for chunk_id, _text in group:
                    # Dispatched, so never a candidate again this run -- even if
                    # it came back unusable, since a second identical request
                    # would just spend budget to fail the same way.
                    seen.add(chunk_id)
                    data = results.get(chunk_id)
                    if data is None:
                        stats["skipped"] += 1
                        if not failed:
                            print(f"  chunk {chunk_id}: unparseable response, "
                                  f"skipped (retried next run)")
                        p.advance()
                        continue
                    annotate._write_annotation(con, chunk_id, data)
                    stats["done"] += 1
                    p.advance()
                    if stats["done"] % config.ANNOTATE_COMMIT_EVERY == 0:
                        con.commit()
    finally:
        # Commit in the finally, not just on the KeyboardInterrupt path: an
        # ollama timeout propagating out of _pooled would otherwise roll back
        # up to ANNOTATE_COMMIT_EVERY - 1 upgrades the budget had already paid
        # for. Also where stats["stopped"] lands, so the caller's summary is
        # accurate on every exit -- the gate is the only thing that knows
        # whether candidates were left undispatched.
        con.commit()
        stats["stopped"] = gate.stopped
        annotate._save_group_rate(con, gate.rate, phase, model, batch_size,
                                  workers)
    if gate.stopped and not gate._dispatched and groups:
        # Keyed on groups DISPATCHED, not written: a probe that came back empty
        # did dispatch. A phase's rate has measured nothing when it starts, so
        # the probe always sends its first group unless the phase's deadline had
        # already passed -- the earlier phases spent its share. Say that rather
        # than "there was nothing to do".
        print(f"  dispatched nothing: this phase's share of the budget was spent "
              f"before its first request ({len(rows)} candidates left) -- raise "
              f"ANNOTATE_BUDGET_MINUTES")


def _preview(rows):
    return ", ".join(str(cid) for cid, *_ in rows[:10]) or "(none)"


def _fmt_left(seconds):
    """Remaining time in whichever unit carries information. The budget is
    denominated in minutes, but a small ANNOTATE_BUDGET_MINUTES (or a test
    driving seconds) would otherwise print '0.0 min left' immediately before
    the tier goes and annotates several hundred chunks."""
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f} min"


def _summary(counts, events, low_conf, minutes, started, note):
    elapsed = (time.monotonic() - started) / 60.0
    print(f"hybrid summary: seeded {counts['seeded']}, kNN-classified "
          f"{counts['classified']}, confidence backfilled {counts['backfilled']}, "
          f"event upgrades {events['done']} ({events['skipped']} skipped), "
          f"confidence upgrades {low_conf['done']} ({low_conf['skipped']} skipped)")
    print(f"  {note} -- used {elapsed:.1f} of {minutes:g} budgeted minutes")


class _Run:
    """The one mutable state object the four phases share, so each is a plain
    function against an explicit contract instead of 350 lines of interleaved
    locals.

    It carries three kinds of thing, and the distinction is what keeps the
    budget invariants readable: the run's SETTINGS (model/think/scope/limit,
    fixed for the run), its CLOCK (`started`, `deadline`, `minutes`, plus the
    per-phase rate estimators an earlier run banked), and its TALLIES
    (`counts`/`events`/`low_conf`/`seen`), which every phase adds to and the
    closing summary reads.

    A phase returns None to mean "carry on" and an int to mean "this is the
    tier's return value, stop here" -- exactly the early returns the single
    function used to do inline, so short-circuit behavior is unchanged.
    """

    def __init__(self, con, model, think, workers, batch_size, all_chats,
                 top_dyads, limit, dry_run, scope, has_conf, k, seed_target,
                 minutes, started, deadline):
        self.con, self.model, self.think = con, model, think
        self.workers, self.batch_size = workers, batch_size
        self.all_chats, self.top_dyads, self.limit = all_chats, top_dyads, limit
        self.dry_run, self.scope, self.has_conf, self.k = dry_run, scope, has_conf, k
        self.seed_target = seed_target
        self.minutes, self.started, self.deadline = minutes, started, deadline
        self.counts = {"seeded": 0, "classified": 0, "backfilled": 0}
        self.events = {"done": 0, "skipped": 0, "stopped": False}
        self.low_conf = {"done": 0, "skipped": 0, "stopped": False}
        self.seen = set()
        # One estimator per phase, each primed from what an earlier run measured
        # for this model/batch/workers -- so no phase has to commit a blind
        # _pooled wave, and none of them anchors another wrong. They really are
        # different populations, and it is the CHUNKS that differ: the event
        # candidates average ~1,900 chars, so
        # annotate._OLLAMA_ANNOTATE_CHAR_BUDGET closes a group at 4-6 of them,
        # while the confidence upgrades' ~360-char median fills a whole batch of
        # 8. Measured 2026-07-28 after the schema retry landed, the worst group
        # cost 14.1 s in 3a against 16.6 s in 3b single-stream -- similar per
        # GROUP over groups of quite different size, which is exactly why one
        # shared estimator would mis-plan both.
        self.seed_rate = annotate._load_group_rate(con, "seed", model, batch_size, workers)
        self.event_rate = annotate._load_group_rate(con, "event", model, batch_size, workers)
        self.low_conf_rate = annotate._load_group_rate(con, "lowconf", model,
                                                       batch_size, workers)
        # Phase 3a hands these to 3b and to the closing summary.
        self.event_left_behind = 0
        self.n_low_conf = 0
        self.low_conf_rows = []

    @property
    def total(self):
        """Rows written by the deterministic phases (the confidence backfill is
        deliberately excluded -- see hybrid_chunks)."""
        return self.counts["seeded"] + self.counts["classified"]

    @property
    def written(self):
        """Everything this run wrote, deterministic phases plus LLM upgrades."""
        return self.total + self.events["done"] + self.low_conf["done"]

    def summary(self, note):
        _summary(self.counts, self.events, self.low_conf, self.minutes,
                 self.started, note)


def _phase_seed(run):
    """Phase 1: top up the LLM-labeled seed pool with a stratified,
    embedding-diverse draw, corpus-wide, inside _SEED_PHASE_SHARE of the clock.
    Returns the tier's value on a Ctrl-C, else None."""
    seed_deadline = run.started + run.minutes * 60 * _SEED_PHASE_SHARE
    print(f"phase 1: seed top-up (corpus-wide, stratified draw, up to "
          f"{_SEED_PHASE_SHARE:.0%} of the budget)")
    seeds = {"done": 0}
    try:
        run.counts["seeded"] = annotate_distill._seed_chunks(
            run.con, run.model, run.think, run.workers, run.seed_target,
            dry_run=run.dry_run, sampler=annotate_sampling.stratified_sample,
            batch_size=run.batch_size, deadline=seed_deadline, rate=run.seed_rate,
            counter_key=_SEED_COUNTER_KEY, interrupt_exit=False, stats=seeds)
    except KeyboardInterrupt:
        # Ctrl-C ends every phase of this tier the same way -- commit, report,
        # exit 0 -- rather than phase 1 alone exiting 1 (db.run_cli never even
        # saw the interrupt, so the tier had three exit codes for one key).
        run.counts["seeded"] = seeds["done"]
        run.summary("interrupted during the seed pass; rerun to resume")
        return run.counts["seeded"]
    finally:
        if not run.dry_run:
            annotate._save_group_rate(run.con, run.seed_rate, "seed", run.model,
                                      run.batch_size, run.workers)
    return None


def _phase_knn(run):
    """Phase 2: kNN-classify the whole corpus, then backfill knn_confidence onto
    distilled rows written before that column existed. Deterministic, but
    O(rows x seeds), so it runs against the budget like everything else.
    Returns the tier's value when there is no point going on, else None."""
    print("phase 2: kNN classify + confidence backfill (corpus-wide)")
    knn = {"stopped": False, "interrupted": False}
    try:
        run.counts["classified"] = annotate_distill._classify_chunks(
            run.con, run.k, dry_run=run.dry_run, deadline=run.deadline, stats=knn)
        # Skipped outright when the classify pass already ran out of clock (or
        # was interrupted): it would only reconfirm that there is no time.
        if not (knn["stopped"] or knn["interrupted"]):
            run.counts["backfilled"] = annotate_distill.backfill_confidence(
                run.con, run.k, dry_run=run.dry_run, deadline=run.deadline,
                stats=knn, has_confidence=run.has_conf)
    except SystemExit:
        # A seed pass cut short by its share of the budget can leave the pool
        # under annotate_distill._KNN_MIN_SEEDS, where the vote really would be
        # noise and _build_seed_index refuses. The floor stays; what softens is
        # the OUTCOME, because a tier whose contract is "spend N minutes and
        # commit what you got" must not exit non-zero for having been given too
        # few minutes. Re-raised for anything else (no chunk_vectors, no
        # embed_dim) -- those are misconfigurations a rerun won't fix.
        n_seeds = _n_seeds(run.con)
        if n_seeds is None or n_seeds >= annotate_distill._KNN_MIN_SEEDS:
            raise
        run.summary(
            f"the seed pool is still under the {annotate_distill._KNN_MIN_SEEDS}"
            f"-seed kNN floor, so no classification or upgrades ran; rerun "
            f"(each run seeds another {_SEED_PHASE_SHARE:.0%} of the budget)")
        return run.counts["seeded"]

    # A Ctrl-C inside phase 2 is caught and RETURNED there, not raised, so
    # without this the tier would fall straight into the LLM phases the user
    # just asked it to stop.
    if knn["interrupted"]:
        run.summary("the kNN phases were interrupted, so no LLM upgrades ran; "
                    "rerun to resume")
        return run.total

    now = time.monotonic()
    if not run.dry_run and (knn["stopped"] or now >= run.deadline):
        run.summary("the budget was spent on phases 1-2, so no LLM upgrades ran; "
                    "rerun to resume")
        return run.total
    return None


def _phase_event_upgrades(run):
    """Phase 3a: LLM-upgrade the event-candidate chunks, densest lexicon match
    first, inside _EVENT_PHASE_SHARE of what the clock has left. Returns the
    tier's value on an interrupt, else None."""
    # The event prefilter already excludes chunks annotated by an LLM, so a
    # resumed run never re-buys an upgrade it has paid for -- and it hands back
    # what is left ranked by pattern-hit density, so all three ways this list
    # gets shortened (--limit's prefix, _EVENT_FETCH_CAP's prefix, the deadline's
    # cut) keep the densest candidates rather than the oldest.
    event_ids = annotate_events.event_candidate_chunk_ids(
        run.con, run.all_chats, run.top_dyads)
    # The TRUE candidate count, taken before either prefix shortens the list --
    # the same rule 3b's textless COUNT follows, and for the same reason:
    # reporting a count that a cap already shrank understates the scope. Both
    # prefixes are then measured against it, so "left for a later run" means
    # what it says.
    n_candidates = len(event_ids)
    if run.limit is not None:
        # An upper bound on what this sub-phase can deliver, not the cap itself:
        # --limit counts rows WRITTEN (see phase 3b), and a dispatch that comes
        # back unparseable writes nothing, so 3b makes up the shortfall out of
        # its own candidates WHEN IT HAS ANY -- with an empty 3b pool (a scope
        # whose pending chunks were never embedded, so phase 2 classified
        # nothing) a skipped 3a dispatch does leave the run short of N. --limit
        # is an "at most N" cap; the alternative is dispatching past N, which is
        # work the flag did not authorize. A rerun buys the rest.
        event_ids = event_ids[:run.limit]
    # Kept for the --dry-run preview, which must not be clipped to a memory
    # ceiling it is not paying (the fetch below never runs on that path).
    scoped_ids = event_ids
    capped = len(event_ids) > _EVENT_FETCH_CAP
    if capped:
        event_ids = event_ids[:_EVENT_FETCH_CAP]
    run.event_left_behind = n_candidates - len(event_ids)
    # A dry run never fetches: it reports the scope, and pulling every
    # candidate's text to print ten ids would be the run's most expensive step.
    event_rows = [] if run.dry_run else _chunk_rows(run.con, event_ids)
    # The clock is re-read HERE, after the prefilter and the fetch, not before
    # them. event_candidate_chunk_ids is the run's one unavoidable corpus-wide
    # `messages` scan and _chunk_rows carries every candidate's full text;
    # deriving this phase's share from the `now` in phase 2 charged 3a for both,
    # so on a big corpus the phase silently got a fraction of the 60% the header
    # line claims it did. (Phase 2's `now` still owns the early return there --
    # that check has to happen BEFORE paying for the scan.)
    now = time.monotonic()
    left = max(0.0, run.deadline - now)
    events_deadline = now + left * _EVENT_PHASE_SHARE
    print(f"phase 3a: {n_candidates} event-candidate chunks to upgrade "
          f"({run.scope}, up to {_EVENT_PHASE_SHARE:.0%} of the "
          f"{_fmt_left(left)} left)")
    if run.event_left_behind:
        # Named by cause, and counted against the true candidate total: with
        # --limit in play the cap's own drop is only part of what this run
        # leaves behind, and reporting the cap's share alone understated the
        # backlog by however much --limit had already cut. "would fetch" on a
        # dry run because that path provably fetches nothing (see above) -- a
        # preview reports what a real run WOULD do, never work it did not do.
        why = []
        if run.limit is not None and n_candidates > run.limit:
            why.append(f"--limit {run.limit}")
        if capped:
            why.append(f"the {_EVENT_FETCH_CAP:,}-row fetch cap")
        taken = (f"the densest {len(event_ids):,} of them" if event_ids
                 else "none of them")
        print(f"  {'would fetch' if run.dry_run else 'fetching'} {taken}; "
              f"{run.event_left_behind:,} left for a later run "
              f"({' and '.join(why)})")
    if run.dry_run:
        print(f"--dry-run: first chunk ids: "
              f"{', '.join(str(cid) for cid in scoped_ids[:10]) or '(none)'}")
        return None
    # Again, cheaply: the top-level call in hybrid_chunks already cleared this
    # run, so this is one ingest_state read, and it is what keeps the invariant
    # local to the writes for a future caller that reaches phase 3 without
    # coming through the top of hybrid_chunks.
    annotate._check_model_guard(run.con, run.model)
    try:
        _run_upgrades(run.con, event_rows, run.model, run.think, run.workers,
                      run.batch_size, events_deadline,
                      "Upgrading event candidates", run.seen, run.events,
                      run.event_rate, phase="event")
    except KeyboardInterrupt:
        print(f"\nInterrupted -- committed {run.events['done']} event upgrades "
              f"({run.events['skipped']} skipped); rerun to resume.")
        run.summary("interrupted")
        return run.total + run.events["done"]
    return None


def _phase_lowconf_upgrades(run):
    """Phase 3b: LLM-upgrade the lowest-knn_confidence distilled chunks with
    whatever budget is left. Always returns the tier's value -- it is last."""
    # `limit` counts UPGRADES WRITTEN across both sub-phases, so what 3a spends
    # of it is events["done"] and not len(seen): `seen` also holds the chunks 3a
    # dispatched and could not parse, which bought nothing, and charging them
    # against a work-product cap made --limit N deliver fewer than N (the wall
    # clock is what caps spending). The residual is one-sided and symmetrical: a
    # row THIS phase fails to parse leaves the run one short (it is the last
    # phase -- nothing follows to make it up), and so does a 3a skip when this
    # phase has no candidates to make it up FROM. Either way the fix would be to
    # dispatch past N, which is work the flag did not authorize; a rerun buys it,
    # and the summary line below says candidates were left.
    #
    # Fetching remaining+len(seen) rows and filtering afterwards keeps the cap
    # exact: 3a's writes drop out of this query on their own (they are 'llm'
    # now), but the rows it dispatched and skipped are still source='distilled'
    # and would come back around inside the same run. len(seen) is every row 3a
    # dispatched and so bounds how many of those this query can return -- fetch
    # that many extra and `remaining` fresh rows always survive the filter.
    remaining = None if run.limit is None else max(0, run.limit - run.events["done"])
    left = max(0.0, run.deadline - time.monotonic())
    if not run.dry_run and left <= 0:
        # Before the fetch, not after: the fetch carries every in-scope row's
        # full chunk TEXT (9.5 MB / ~26 MB resident on the reference corpus)
        # and there is nothing left to dispatch it into.
        print(f"phase 3b: skipped -- no budget left ({run.scope})")
        run.summary("stopped at a phase deadline with candidates left; rerun to "
                    "resume (rows already annotated by an LLM are skipped)")
        return run.total + run.events["done"]
    # The candidate count comes from the textless COUNT, not len(rows): the
    # fetch below is capped at what a budget could conceivably dispatch, and
    # reporting the cap would understate the scope.
    run.n_low_conf = _low_confidence_count(run.con, run.all_chats, run.top_dyads)
    if run.dry_run:
        run.low_conf_rows = _low_confidence_rows(
            run.con, run.all_chats, run.top_dyads, limit=10,
            by_confidence=run.has_conf)
    else:
        fetch = _LOW_CONF_FETCH_CAP if remaining is None else min(
            _LOW_CONF_FETCH_CAP, remaining + len(run.seen))
        run.low_conf_rows = [
            row for row in _low_confidence_rows(run.con, run.all_chats,
                                                run.top_dyads, limit=fetch)
            if row[0] not in run.seen
        ]
        if remaining is not None:
            run.low_conf_rows = run.low_conf_rows[:remaining]
    print(f"phase 3b: {run.n_low_conf} low-confidence distilled chunks to "
          f"upgrade ({run.scope}, {_fmt_left(left)} left)")
    if run.dry_run:
        # No _summary here: every phase above already printed the number it
        # WOULD process, and a summary of zeros-plus-would-be counts reads as a
        # report of work done.
        print(f"--dry-run: first chunk ids: {_preview(run.low_conf_rows)}")
        print("--dry-run: no LLM calls, no writes")
        return 0
    try:
        _run_upgrades(run.con, run.low_conf_rows, run.model, run.think,
                      run.workers, run.batch_size, run.deadline,
                      "Upgrading low-confidence chunks", run.seen, run.low_conf,
                      run.low_conf_rate, phase="lowconf")
    except KeyboardInterrupt:
        print(f"\nInterrupted -- committed {run.low_conf['done']} confidence "
              f"upgrades ({run.low_conf['skipped']} skipped); rerun to resume.")
        run.summary("interrupted")
        return run.written

    # "a deadline", not "the deadline": 3a stops at its own share of the clock,
    # 3b at the budget itself, and either leaving candidates undispatched is the
    # same news to the reader -- there is more to buy, so rerun.
    stopped = run.events["stopped"] or run.low_conf["stopped"]
    # Candidates this run never even fetched, which is a different sentence from
    # "the clock ran out": --limit (or the fetch caps) scoped them out while the
    # budget still had time. Saying "finished every candidate in scope" there
    # was flatly false, and it hid exactly the shortfall a --limit run wants to
    # see -- the run can end with minutes unspent and thousands of candidates
    # untouched.
    # events["skipped"] comes off the 3b side of the sum: those rows are still
    # source='distilled' (so n_low_conf counts them) and `seen` filtered them
    # out of the fetch, but they were DISPATCHED, not scoped out -- counting
    # them here would blame --limit for an unparseable reply.
    unfetched = run.event_left_behind + max(
        0, run.n_low_conf - run.events["skipped"] - len(run.low_conf_rows))
    if stopped:
        outcome = ("stopped at a phase deadline with candidates left; rerun to "
                   "resume (rows already annotated by an LLM are skipped)")
    elif unfetched:
        outcome = (f"finished everything this run was scoped to, with "
                   f"{unfetched:,} candidates left over (--limit / the fetch "
                   f"caps, not the clock); rerun to continue")
    else:
        outcome = "finished every candidate in scope with budget to spare"
    run.summary(outcome)
    return run.written


def hybrid_chunks(con, model, think=False, workers=1, all_chats=False,
                  top_dyads=None, limit=None, dry_run=False,
                  budget_minutes=None, seed_target=None, batch_size=None):
    """Run the hybrid tier's four phases against one wall-clock deadline (see
    the module docstring). Returns the number of annotation rows written:
    seeds + kNN classifications + Phase-3 upgrades. The confidence backfill is
    deliberately NOT counted -- it fills a column on rows that already existed,
    and its corpus-sized number would drown out the work the budget bought.

    budget_minutes/seed_target None (or 0) resolve from config, with the
    apple-vs-ollama split for the budget; a negative budget is refused rather
    than clamped (see _resolve_budget_minutes). top_dyads None means
    ANNOTATE_TOP_DYADS. `limit` caps the TOTAL Phase-3 upgrades WRITTEN across
    3a+3b -- successes, not dispatches: a chunk whose reply came back
    unparseable spent budget but produced no annotation, and charging the cap
    for it made --limit N deliver fewer than N. The budget is the spend cap;
    --limit is the deliverable cap -- an "at most N", with one one-sided
    residual: a dispatch that fails to parse is made up out of the OTHER
    sub-phase's candidates, so when those run out (3b's pool empty, or 3a's
    ranked list already exhausted) the run lands short of N. Making it up
    regardless would mean dispatching past N, which is spend the flag did not
    authorize; a rerun buys the rest. A NEGATIVE limit is refused outright.

    annotate._check_model_guard runs at the TOP, before the budget clock
    starts, because phases 1-2 can burn half the budget before Phase 3 reaches
    a guard: Phase 1 calls it too, but only once it actually has seeds to
    label, so a pool already at target skips it entirely and a switched
    ANNOTATE_MODEL used to surface minutes later. Phase 3 keeps its own call
    (below) rather than assuming this one ran.

    `batch_size` is appended at the END of the signature deliberately: the
    tier's original design froze the parameter list without it while also
    requiring that ANNOTATE_BATCH_SIZE and _resolve_quality keep working here,
    and re-deriving it inside would mean a second copy of _resolve_quality.
    Appending keeps every positional caller working. None falls back to the
    preset's value."""
    if limit is not None and limit < 0:
        # Refused, not clamped or passed through, on the same rule as a negative
        # budget: a negative limit is never a request for anything. Passed
        # through it silently inverts both of the prefixes it feeds --
        # `event_ids[:-1]` drops the LEAST dense candidate instead of taking a
        # densest-first prefix, and `max(0, limit - done)` then switches 3b off
        # entirely -- so a stray minus bought a run nobody asked for rather than
        # an error. 0 stays a legitimate (empty) run, as it is for the LLM tiers.
        raise SystemExit(f"--limit cannot be negative (got {limit}); 0 means "
                         f"'upgrade nothing', omit it for no cap")
    if top_dyads is None:
        top_dyads = config.ANNOTATE_TOP_DYADS
    if top_dyads < 0 and not all_chats:
        # This is the second place ANNOTATE_TOP_DYADS is resolved (annotate.main
        # is the first), and a knob is checked where it is resolved -- otherwise
        # a caller reaching this function directly gets the corpus-wide phase-3
        # scope that `LIMIT -1` means, which is the run --all exists to ask for
        # explicitly. Gated on `not all_chats` because that branch never binds
        # the value: a bad knob must not refuse a run it cannot affect.
        raise SystemExit(
            f"ANNOTATE_TOP_DYADS cannot be negative (got {top_dyads}); SQLite "
            f"reads it as no limit at all, so phase 3 would silently cover every "
            f"chat. Pass all_chats=True if that is what you want.")
    if not seed_target:
        seed_target = config.ANNOTATE_HYBRID_SEED_TARGET
    if batch_size is None:
        batch_size = annotate.QUALITY_PRESETS["hybrid"][1]
    # Before the clock, before the migration, before a single phase runs. Every
    # other tier reaches its guard within one dispatch of starting; this one has
    # two corpus-wide phases in front of it, and with the seed pool already at
    # target neither of them guards (Phase 1 returns before its own call), so a
    # switched ANNOTATE_MODEL was refused only after the kNN passes had spent up
    # to half the budget. Skipped under --dry-run for the same reason the
    # migration is: the guard RECORDS the model it cleared (set_state + commit),
    # and a preview that writes annotate_model is a preview that wrote.
    if not dry_run:
        annotate._check_model_guard(con, model)
    # Phases 2 and 3b read and write knn_confidence, which postdates DBs in the
    # wild -- and annotate.py is reachable without the ingest.py run that
    # normally applies migrations. Under --dry-run the migration is skipped
    # (it is a write, and create_schema also rescans reactions) and the two
    # preview reads below degrade instead.
    has_conf = annotate._ensure_column(con, "chunk_annotations", "knn_confidence",
                                       dry_run=dry_run)
    minutes, budget_source = _resolve_budget_minutes(model, budget_minutes)
    k = config.ANNOTATE_KNN_K
    started = time.monotonic()
    deadline = started + minutes * 60

    scope = "all chats" if all_chats else f"top {top_dyads} chats by message count"
    settings = f"model {model}, think {think!r}, batch {batch_size}"
    if workers > 1:
        settings += f", workers {workers}"
    if limit is not None:
        settings += f", limit {limit}"
    print(f"hybrid tier: {minutes:g} min budget ({budget_source}), seed target "
          f"{seed_target}, kNN k={k}, phase 3 scope {scope} ({settings})")

    run = _Run(con, model, think, workers, batch_size, all_chats, top_dyads,
               limit, dry_run, scope, has_conf, k, seed_target,
               minutes, started, deadline)
    # Each phase returns None to carry on and an int to end the run there --
    # the same short-circuits the one long function did inline.
    for phase in (_phase_seed, _phase_knn, _phase_event_upgrades):
        out = phase(run)
        if out is not None:
            return out
    return _phase_lowconf_upgrades(run)
