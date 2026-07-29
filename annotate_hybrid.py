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
import progress

# Config is read through the module (config.X) rather than imported by name:
# the budget default depends on the model, which is only known per call, and
# the knobs stay patchable by a test that drives hybrid_chunks directly.

# Default wall-clock budget per backend when ANNOTATE_BUDGET_MINUTES is unset.
# Measured 2026-07-27 (docs/PLAN_ANNOTATE_HYBRID.md §2, 25,029-chunk corpus):
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
# The consequence to know about is the floor underneath both phases: nothing
# smaller than one group can be dispatched, so a phase whose share is under one
# group's LATENCY buys nothing at all. Measured 2026-07-28 (qwen3.5:4b, batch
# 8, 4 workers) after the char packing and the schema retry landed, an event
# group -- 4-6 chunks, not 8, at ~1,900 chars each -- banked at 46.7 s of
# latency, so a small budget correctly declines both phases and _run_upgrades
# prints why rather than leaving that reading as "there was nothing to do". The
# 25-min default is far clear of the floor; the old behavior on a budget that
# small was to commit a blind wave anyway and take 4.4 min over a 2-min one.
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
    empty-means-default convention config uses for its other int knobs."""
    if budget_minutes:
        return budget_minutes, "caller"
    if config.ANNOTATE_BUDGET_MINUTES:
        return config.ANNOTATE_BUDGET_MINUTES, "ANNOTATE_BUDGET_MINUTES"
    if config.is_apple_model(model):
        return _DEFAULT_BUDGET_MINUTES_APPLE, "apple default"
    return _DEFAULT_BUDGET_MINUTES_OLLAMA, "ollama default"


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
                    gate.timed(lambda g: annotate._annotate_group(g, model, think)),
                    workers):
                for chunk_id, _text in group:
                    # Dispatched, so never a candidate again this run -- even if
                    # it came back unusable, since a second identical request
                    # would just spend budget to fail the same way.
                    seen.add(chunk_id)
                    data = results.get(chunk_id)
                    if data is None:
                        stats["skipped"] += 1
                        print(f"  chunk {chunk_id}: unparseable response, skipped "
                              f"(retried next run)")
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
    if gate.stopped and not stats["done"] and groups:
        # Declining every candidate is the correct answer to a budget smaller
        # than one request, but silence would read as "there was nothing to do".
        # len(groups[0]), not batch_size: apple packs by char budget instead.
        # The estimate is guarded even though every production rate carries a
        # prior (_load_group_rate falls back to the cold constant): a rate with
        # nothing to plan with reserves 0.0, so the only way here without one
        # is a deadline that had already passed -- say that, don't crash on
        # formatting None.
        est = gate.rate.estimate()
        cost = (f"worth ~{est:.0f}s at the measured rate, more than this "
                f"phase's share of the budget" if est
                else "more than the time this phase had left")
        print(f"  dispatched nothing: one request of {len(groups[0])} chunks is "
              f"{cost} -- raise ANNOTATE_BUDGET_MINUTES")


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


def hybrid_chunks(con, model, think=False, workers=1, all_chats=False,
                  top_dyads=None, limit=None, dry_run=False,
                  budget_minutes=None, seed_target=None, batch_size=None):
    """Run the hybrid tier's four phases against one wall-clock deadline (see
    the module docstring). Returns the number of annotation rows written:
    seeds + kNN classifications + Phase-3 upgrades. The confidence backfill is
    deliberately NOT counted -- it fills a column on rows that already existed,
    and its corpus-sized number would drown out the work the budget bought.

    budget_minutes/seed_target None (or 0) resolve from config, with the
    apple-vs-ollama split for the budget. top_dyads None means
    ANNOTATE_TOP_DYADS. `limit` caps the TOTAL Phase-3 upgrades across 3a+3b.
    annotate._check_model_guard runs before any Phase-3 write: Phase 1 calls it
    too, but only when it actually has seeds to label, so Phase 3 must not
    assume Phase 1 ran.

    Deviation from docs/PLAN_ANNOTATE_HYBRID.md §4: `batch_size` is appended to
    the frozen signature (backward-compatibly, at the end). The plan's
    signature omitted it while also requiring that ANNOTATE_BATCH_SIZE and
    _resolve_quality keep working for this tier; re-deriving it here would mean
    a second copy of _resolve_quality. None falls back to the preset's value."""
    if top_dyads is None:
        top_dyads = config.ANNOTATE_TOP_DYADS
    if not seed_target:
        seed_target = config.ANNOTATE_HYBRID_SEED_TARGET
    if batch_size is None:
        batch_size = annotate.QUALITY_PRESETS["hybrid"][1]
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

    counts = {"seeded": 0, "classified": 0, "backfilled": 0}
    events = {"done": 0, "skipped": 0, "stopped": False}
    low_conf = {"done": 0, "skipped": 0, "stopped": False}
    seen = set()
    # One estimator per phase, each primed from what an earlier run measured
    # for this model/batch/workers -- so no phase has to commit a blind _pooled
    # wave, and none of them anchors another wrong. They really are different
    # populations, and it is the CHUNKS that differ: the event candidates
    # average ~1,900 chars, so annotate._OLLAMA_ANNOTATE_CHAR_BUDGET closes a
    # group at 4-6 of them, while the confidence upgrades' ~360-char median
    # fills a whole batch of 8. Measured 2026-07-28 after the schema retry
    # landed, the worst group cost 14.1 s in 3a against 16.6 s in 3b
    # single-stream -- similar per GROUP over groups of quite different size,
    # which is exactly why one shared estimator would mis-plan both.
    seed_rate = annotate._load_group_rate(con, "seed", model, batch_size, workers)
    event_rate = annotate._load_group_rate(con, "event", model, batch_size, workers)
    low_conf_rate = annotate._load_group_rate(con, "lowconf", model, batch_size,
                                              workers)

    # Phases 1-2: corpus-wide (see the module docstring). Phase 1 gets a
    # bounded slice of the clock because it is LLM work; phase 2 runs against
    # the budget itself -- it is deterministic, but O(rows x seeds), which is
    # not the same thing as cheap.
    seed_deadline = started + minutes * 60 * _SEED_PHASE_SHARE
    print(f"phase 1: seed top-up (corpus-wide, stratified draw, up to "
          f"{_SEED_PHASE_SHARE:.0%} of the budget)")
    seeds = {"done": 0}
    try:
        counts["seeded"] = annotate_distill._seed_chunks(
            con, model, think, workers, seed_target, dry_run=dry_run,
            sampler=annotate_sampling.stratified_sample,
            batch_size=batch_size, deadline=seed_deadline, rate=seed_rate,
            counter_key=_SEED_COUNTER_KEY, interrupt_exit=False, stats=seeds)
    except KeyboardInterrupt:
        # Ctrl-C ends every phase of this tier the same way -- commit, report,
        # exit 0 -- rather than phase 1 alone exiting 1 (db.run_cli never even
        # saw the interrupt, so the tier had three exit codes for one key).
        counts["seeded"] = seeds["done"]
        _summary(counts, events, low_conf, minutes, started,
                 "interrupted during the seed pass; rerun to resume")
        return counts["seeded"]
    finally:
        if not dry_run:
            annotate._save_group_rate(con, seed_rate, "seed", model, batch_size,
                                      workers)
    print("phase 2: kNN classify + confidence backfill (corpus-wide)")
    knn = {"stopped": False, "interrupted": False}
    try:
        counts["classified"] = annotate_distill._classify_chunks(
            con, k, dry_run=dry_run, deadline=deadline, stats=knn)
        # Skipped outright when the classify pass already ran out of clock (or
        # was interrupted): it would only reconfirm that there is no time.
        if not (knn["stopped"] or knn["interrupted"]):
            counts["backfilled"] = annotate_distill.backfill_confidence(
                con, k, dry_run=dry_run, deadline=deadline, stats=knn,
                has_confidence=has_conf)
    except SystemExit:
        # A seed pass cut short by its share of the budget can leave the pool
        # under annotate_distill._KNN_MIN_SEEDS, where the vote really would be
        # noise and _build_seed_index refuses. The floor stays; what softens is
        # the OUTCOME, because a tier whose contract is "spend N minutes and
        # commit what you got" must not exit non-zero for having been given too
        # few minutes. Re-raised for anything else (no chunk_vectors, no
        # embed_dim) -- those are misconfigurations a rerun won't fix.
        n_seeds = _n_seeds(con)
        if n_seeds is None or n_seeds >= annotate_distill._KNN_MIN_SEEDS:
            raise
        _summary(counts, events, low_conf, minutes, started,
                 f"the seed pool is still under the {annotate_distill._KNN_MIN_SEEDS}"
                 f"-seed kNN floor, so no classification or upgrades ran; rerun "
                 f"(each run seeds another {_SEED_PHASE_SHARE:.0%} of the budget)")
        return counts["seeded"]
    total = counts["seeded"] + counts["classified"]

    # A Ctrl-C inside phase 2 is caught and RETURNED there, not raised, so
    # without this the tier would fall straight into the LLM phases the user
    # just asked it to stop.
    if knn["interrupted"]:
        _summary(counts, events, low_conf, minutes, started,
                 "the kNN phases were interrupted, so no LLM upgrades ran; "
                 "rerun to resume")
        return total

    now = time.monotonic()
    if not dry_run and (knn["stopped"] or now >= deadline):
        _summary(counts, events, low_conf, minutes, started,
                 "the budget was spent on phases 1-2, so no LLM upgrades ran; "
                 "rerun to resume")
        return total

    # Phase 3a. The event prefilter already excludes chunks annotated by an LLM,
    # so a resumed run never re-buys an upgrade it has paid for -- and it hands
    # back what is left ranked by pattern-hit density, so both --limit's prefix
    # and the deadline's cut take the densest candidates, not the oldest.
    event_ids = annotate_events.event_candidate_chunk_ids(con, all_chats, top_dyads)
    if limit is not None:
        event_ids = event_ids[:limit]
    left = max(0.0, deadline - now)
    events_deadline = now + left * _EVENT_PHASE_SHARE
    print(f"phase 3a: {len(event_ids)} event-candidate chunks to upgrade "
          f"({scope}, up to {_EVENT_PHASE_SHARE:.0%} of the {_fmt_left(left)} left)")
    if dry_run:
        # Ids only: a dry run reports the scope, and pulling every candidate's
        # text to print ten of them would be the run's most expensive step.
        print(f"--dry-run: first chunk ids: "
              f"{', '.join(str(cid) for cid in event_ids[:10]) or '(none)'}")
    else:
        event_rows = _chunk_rows(con, event_ids)
        # Once, before either sub-phase can write. Phase 1 guards its own
        # writes, but it returns early when the seed pool is already full.
        annotate._check_model_guard(con, model)
        try:
            _run_upgrades(con, event_rows, model, think, workers, batch_size,
                          events_deadline, "Upgrading event candidates", seen,
                          events, event_rate, phase="event")
        except KeyboardInterrupt:
            print(f"\nInterrupted -- committed {events['done']} event upgrades "
                  f"({events['skipped']} skipped); rerun to resume.")
            _summary(counts, events, low_conf, minutes, started, "interrupted")
            return total + events["done"]

    # Phase 3b. Fetching limit+len(seen) rows and filtering afterwards keeps the
    # cap exact: 3a's writes drop out of this query on their own (they are 'llm'
    # now), but a row it skipped, or one whose write is still uncommitted, would
    # otherwise come back around inside the same run.
    remaining = None if limit is None else max(0, limit - len(seen))
    left = max(0.0, deadline - time.monotonic())
    if not dry_run and left <= 0:
        # Before the fetch, not after: the fetch carries every in-scope row's
        # full chunk TEXT (9.5 MB / ~26 MB resident on the reference corpus)
        # and there is nothing left to dispatch it into.
        print(f"phase 3b: skipped -- no budget left ({scope})")
        _summary(counts, events, low_conf, minutes, started,
                 "stopped at a phase deadline with candidates left; rerun to "
                 "resume (rows already annotated by an LLM are skipped)")
        return total + events["done"]
    # The candidate count comes from the textless COUNT, not len(rows): the
    # fetch below is capped at what a budget could conceivably dispatch, and
    # reporting the cap would understate the scope.
    n_low_conf = _low_confidence_count(con, all_chats, top_dyads)
    if dry_run:
        low_conf_rows = _low_confidence_rows(con, all_chats, top_dyads, limit=10,
                                             by_confidence=has_conf)
    else:
        fetch = _LOW_CONF_FETCH_CAP if remaining is None else min(
            _LOW_CONF_FETCH_CAP, remaining + len(seen))
        low_conf_rows = [
            row for row in _low_confidence_rows(con, all_chats, top_dyads,
                                                limit=fetch)
            if row[0] not in seen
        ]
        if remaining is not None:
            low_conf_rows = low_conf_rows[:remaining]
    print(f"phase 3b: {n_low_conf} low-confidence distilled chunks to upgrade "
          f"({scope}, {_fmt_left(left)} left)")
    if dry_run:
        # No _summary here: every phase above already printed the number it
        # WOULD process, and a summary of zeros-plus-would-be counts reads as a
        # report of work done.
        print(f"--dry-run: first chunk ids: {_preview(low_conf_rows)}")
        print("--dry-run: no LLM calls, no writes")
        return 0
    try:
        _run_upgrades(con, low_conf_rows, model, think, workers, batch_size,
                      deadline, "Upgrading low-confidence chunks", seen,
                      low_conf, low_conf_rate, phase="lowconf")
    except KeyboardInterrupt:
        print(f"\nInterrupted -- committed {low_conf['done']} confidence upgrades "
              f"({low_conf['skipped']} skipped); rerun to resume.")
        _summary(counts, events, low_conf, minutes, started, "interrupted")
        return total + events["done"] + low_conf["done"]

    # "a deadline", not "the deadline": 3a stops at its own share of the clock,
    # 3b at the budget itself, and either leaving candidates undispatched is the
    # same news to the reader -- there is more to buy, so rerun.
    stopped = events["stopped"] or low_conf["stopped"]
    _summary(counts, events, low_conf, minutes, started,
             "stopped at a phase deadline with candidates left; rerun to resume "
             "(rows already annotated by an LLM are skipped)" if stopped
             else "finished every candidate in scope with budget to spare")
    return total + events["done"] + low_conf["done"]
