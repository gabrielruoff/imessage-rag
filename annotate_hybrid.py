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
  3a event upgrades   -- LLM-re-annotate event-candidate chunks (annotate_events):
                         events are the one label kNN structurally CANNOT
                         produce, since every source='distilled' row carries
                         events_json='[]'
  3b confidence upgrades -- LLM-re-annotate the lowest-knn_confidence distilled
                         rows, i.e. where the kNN vote is least likely right

Upgraded rows are written source='llm', so every valence-trusting reader
(conflict corroboration, the nudge Stage-0 exclusion, voiceprint tone) lights
up for exactly the chunks that matter.

Phases 1-2 are corpus-wide by design (the kNN vote has to generalize, so the
scope flags deliberately don't reach them). Phase 2 is genuinely cheap and runs
to completion; phase 1 is NOT -- it is LLM work like phase 3, and measured
unbounded it consumed a whole 8-minute budget on its own -- so it gets its own
_SEED_PHASE_SHARE slice of the clock and stops dispatching at it, leaving the
pool topped up on the next run. Every LLM phase is deadline-gated; only phase 3
also honors --all/--top-dyads/--limit, like the LLM tiers.

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
_EVENT_PHASE_SHARE = 0.60

# Ids per `WHERE id IN (...)` slice. A corpus-wide event prefilter can name
# thousands of chunks, past SQLite's default 999-variable ceiling.
_ID_SLICE = 500


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


def _low_confidence_rows(con, all_chats, top_dyads, limit=None):
    """(chunk_id, text) for the in-scope distilled rows the kNN vote was least
    sure about, least-confident first."""
    clause, params = _scope_clause(all_chats, top_dyads)
    sql = f"SELECT c.id, c.text {_LOW_CONF_FROM}{clause}{_LOW_CONF_ORDER}"
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
    """(chunk_id, text) for `chunk_ids`, id-ascending -- the row shape
    annotate._annotate_group wants. Fetched in parameter-safe slices; the
    caller's ids arrive ascending (annotate_events' contract), so the slices
    concatenate already ordered."""
    rows = []
    for i in range(0, len(chunk_ids), _ID_SLICE):
        block = list(chunk_ids[i:i + _ID_SLICE])
        placeholders = ",".join("?" * len(block))
        rows.extend(con.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders}) ORDER BY id",
            block).fetchall())
    return rows


def _knn_work_pending(con):
    """True when the Phase-2 passes left work behind -- which here means a
    Ctrl-C landed inside one of them. Both _classify_chunks and
    backfill_confidence catch KeyboardInterrupt internally and RETURN, without
    telling the caller, so without this check the tier would fall straight into
    the LLM phases the user just asked it to stop. Exact rather than heuristic:
    both passes drive off these same two todo queries, so a pass that ran to
    completion always leaves zero."""
    return bool(con.execute("""
        SELECT (SELECT COUNT(*) FROM chunk_vectors v
                  LEFT JOIN chunk_annotations a ON a.chunk_id = v.chunk_id
                 WHERE a.chunk_id IS NULL)
             + (SELECT COUNT(*) FROM chunk_annotations a
                  JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
                 WHERE a.source = 'distilled' AND a.knn_confidence IS NULL)
    """).fetchone()[0])


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
                  description, seen, stats, rate=None):
    """LLM-re-annotate `rows` ((chunk_id, text)) until they run out or
    `deadline` passes, writing source='llm'. Mirrors annotate_chunks' dispatch
    loop -- same grouping, same _pooled fan-out, same per-row write and commit
    cadence -- differing only in the deadline gate and the `seen` dedup set.

    The gate is annotate._DispatchDeadline, which stops dispatching a drain's
    worth of time BEFORE the deadline rather than at it: everything already
    handed to the pool still finishes and commits, and at ANNOTATE_WORKERS=4
    that backlog measured minutes, not "one group". `rate` carries the run's
    per-group timings across phases, so only the first LLM phase of a run has
    to dispatch without an estimate.

    Counters go into the caller-owned `stats` dict (done/skipped/stopped)
    rather than a return value, so a KeyboardInterrupt propagating out of here
    still leaves the caller able to print an accurate per-phase summary."""
    groups = annotate._annotation_groups(rows, model, batch_size)
    gate = annotate._DispatchDeadline(deadline, workers, time.monotonic, rate)

    try:
        with progress.track(description, total=len(rows), header=False) as p:
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
    except KeyboardInterrupt:
        con.commit()
        raise
    finally:
        # In a finally so the caller's summary is accurate on the interrupt
        # path too -- the gate is the only thing that knows whether candidates
        # were left undispatched.
        stats["stopped"] = gate.stopped
    con.commit()


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
    # normally applies migrations.
    annotate._ensure_column(con, "chunk_annotations", "knn_confidence")
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
    # One estimator for the whole run: phase 3 then starts from what phase 1
    # measured instead of committing a blind _pooled wave of its own.
    rate = annotate._GroupRate()

    # Phases 1-2: corpus-wide (see the module docstring). Phase 1 is LLM work
    # and gets a bounded slice of the clock; phase 2 is deterministic kNN and
    # runs to completion.
    seed_deadline = started + minutes * 60 * _SEED_PHASE_SHARE
    print(f"phase 1: seed top-up (corpus-wide, stratified draw, up to "
          f"{_SEED_PHASE_SHARE:.0%} of the budget)")
    counts["seeded"] = annotate_distill._seed_chunks(
        con, model, think, workers, seed_target, dry_run=dry_run,
        sampler=annotate_sampling.stratified_sample,
        batch_size=batch_size, deadline=seed_deadline, rate=rate)
    print("phase 2: kNN classify + confidence backfill (corpus-wide)")
    try:
        counts["classified"] = annotate_distill._classify_chunks(con, k, dry_run=dry_run)
        counts["backfilled"] = annotate_distill.backfill_confidence(con, k, dry_run=dry_run)
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

    if not dry_run and _knn_work_pending(con):
        _summary(counts, events, low_conf, minutes, started,
                 "the kNN phases were interrupted, so no LLM upgrades ran; "
                 "rerun to resume")
        return total

    now = time.monotonic()
    if not dry_run and now >= deadline:
        _summary(counts, events, low_conf, minutes, started,
                 "the budget was spent on phases 1-2, so no LLM upgrades ran; "
                 "rerun to resume")
        return total

    # Phase 3a. The event prefilter already excludes chunks annotated by an LLM,
    # so a resumed run never re-buys an upgrade it has paid for.
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
                          events, rate)
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
    if dry_run:
        n_low_conf = _low_confidence_count(con, all_chats, top_dyads)
        low_conf_rows = _low_confidence_rows(con, all_chats, top_dyads, limit=10)
    else:
        low_conf_rows = [
            row for row in _low_confidence_rows(
                con, all_chats, top_dyads,
                limit=None if remaining is None else remaining + len(seen))
            if row[0] not in seen
        ]
        if remaining is not None:
            low_conf_rows = low_conf_rows[:remaining]
        n_low_conf = len(low_conf_rows)
    left = max(0.0, deadline - time.monotonic())
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
                      low_conf, rate)
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
