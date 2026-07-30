"""Distilled annotation tier (annotate.py --quality distilled): top up an
LLM-labeled seed pool by annotating a random corpus-wide sample, then
kNN-classify every remaining embedded chunk from its chunk_vectors embedding
-- valence/emotions/depth only, events left empty (source='distilled' marks
the rows). annotate.py owns the CLI and calls distill_chunks; the shared
request/validation/write helpers stay in annotate and are resolved late
(module-attribute access), so the two modules can import each other safely.

Every kNN row also carries a deterministic 0..1 `knn_confidence` (how close
and how unanimous its neighbours were), which the hybrid tier uses to aim its
LLM budget at the least certain rows; backfill_confidence fills it in for
distilled rows written before the column existed. _seed_chunks takes an
optional `sampler` hook so a tier can substitute a smarter draw than the
default random sample without forking this module, plus `batch_size`/`deadline`
/`counter_key`/`interrupt_exit` so a wall-clock-budgeted tier can batch the
seed pass, cap what it spends on it, count only its OWN draws toward the
target and keep its own interrupt reporting. Both kNN passes take the same
optional `deadline`/`stats` -- neither is free (each is one vec0 MATCH per row
against every INDEXED seed), so a budgeted caller has to be able to stop them.
ANNOTATE_KNN_INDEX_CAP bounds that second factor: the seed POOL grows without
limit (the hybrid tier widens it every run) but the index built from it does
not. Every one of those defaults is today's distilled behavior exactly.
"""

import json
import math
import random
import time

import annotate
import progress
from config import (
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_KNN_INDEX_CAP,
    ANNOTATE_KNN_K,
    ANNOTATE_SEED_TARGET,
)
from db import get_state, set_state

# kNN classification: a candidate emotion label must win at least this share
# of the neighbours' distance-weighted vote. Chosen on the 2026-07-17
# held-out eval (see docs/DECISIONS.md).
_KNN_EMOTION_MIN_SCORE = 0.15
# Fewer LLM seeds than this and the kNN vote is mostly noise -- refuse.
_KNN_MIN_SEEDS = 100

# The (kept, pool) pair _build_seed_index last reported, so the two rebuilds
# inside one hybrid run (_classify_chunks, then backfill_confidence) don't print
# the same sentence twice, wedged between two different progress bars. Process
# state rather than a caller flag because it is a display concern only: nothing
# reads it, and a run where the pool really did change between builds still says
# so.
_last_index_report = None


def _seed_chunks(con, model, think, workers, target, dry_run=False, sampler=None,
                 batch_size=1, deadline=None, rate=None, counter_key=None,
                 interrupt_exit=True, stats=None):
    """Top up the LLM-labeled seed pool to `target` rows by annotating a
    random corpus-wide sample (random + corpus-wide so the kNN vote
    generalizes; scope flags deliberately don't apply). Same writes as an
    LLM tier, so seeds double as ordinary annotations. Caveat: pre-existing
    source='llm' rows count toward `target` regardless of how they were
    scoped, so a pool inherited from an earlier top-dyads LLM run is biased
    toward those chats rather than truly random -- the held-out agreement
    numbers in docs/DECISIONS.md assume a random pool. The sample draws from
    distilled rows as well as unannotated ones (relabeling them 'llm'), so
    raising ANNOTATE_SEED_TARGET grows the pool even on a fully classified
    corpus.

    `sampler` is Callable[(con, pending, need), list] and replaces the default
    random draw -- the hook the hybrid tier uses to substitute a stratified,
    embedding-diverse selection (which also blunts the inherited-pool bias
    above). It is read-only: it picks rows out of `pending`, it never writes.
    sampler=None keeps the historical random.sample behavior exactly, so the
    distilled tier is unaffected.

    `batch_size` and `deadline` are the two levers a budgeted caller needs and
    the distilled tier deliberately does not use. Measured 2026-07-27: a
    350-seed top-up one chunk per request cost 641.9 s (1.83 s/chunk) and ate a
    whole 8-minute hybrid budget before an upgrade could run. Batching it is
    worth ~20% (1.46 s/chunk at batch 8) -- real but not enough on its own, so
    hybrid passes BOTH its batch size and a share-of-the-budget deadline.
    `rate` is this phase's annotate._GroupRate, primed by annotate.
    _load_group_rate with what an earlier run measured.

    `counter_key` changes what counts toward `target`: with it, the target is a
    floor on the seeds THIS pipeline drew (tracked under that ingest_state key)
    rather than on every source='llm' row in the table. The hybrid tier needs
    that because its own phase-3 upgrades are 'llm' rows, so counting them made
    the tier's output satisfy its own seed target -- measured on the reference
    corpus, 4,416 'llm' rows against a target of 800 printed "labeling 0 more"
    on every run and the stratified sampler never executed again, which is
    exactly the inherited-pool bias above going unblunted. Clamped to the real
    'llm' count so an emptied chunk_annotations cannot leave a counter claiming
    seeds that no longer exist.

    `interrupt_exit=False` re-raises the KeyboardInterrupt (after committing
    and reporting) instead of turning it into SystemExit(1), for a caller that
    owns a multi-phase summary of its own -- and `stats`, a caller-owned dict,
    carries the committed count out along that path, where there is no return
    value to carry it.

    All the defaults are exactly today's behavior, so the distilled tier is
    untouched."""
    n_llm = con.execute(
        "SELECT COUNT(*) FROM chunk_annotations WHERE source = 'llm'"
    ).fetchone()[0]
    if counter_key is None:
        n_seeded = n_llm
        pool = f"{n_llm} LLM annotations"
    else:
        n_seeded = min(n_llm, int(get_state(con, counter_key) or 0))
        pool = f"{n_seeded} sampled seeds of {n_llm} LLM annotations"
    need = max(0, target - n_seeded)
    print(f"seed pool: {pool} (target {target}, labeling {need} more)")
    if dry_run or not need:
        return 0
    pending = annotate._pending_chunks(con, all_chats=True, top_dyads=0, limit=None,
                                       upgrade_distilled=True)
    sample = (random.sample(pending, min(need, len(pending))) if sampler is None
              else sampler(con, pending, need))
    if not sample:
        return 0
    annotate._check_model_guard(con, model)

    # batch_size 1 stays a literal one-row-per-request draw rather than going
    # through _annotation_groups: that helper's apple branch packs by char
    # budget, which would silently re-batch the DISTILLED tier's seed pass on
    # Apple FM. For every other backend the two are the same list.
    groups = (annotate._annotation_groups(sample, model, batch_size)
              if batch_size > 1 else [[row] for row in sample])
    gate = (annotate._DispatchDeadline(deadline, workers, time.monotonic, rate)
            if deadline is not None else None)
    if gate is not None:
        # A cold prior is arithmetic over batch_size, which is only a ceiling:
        # seeds are short so an Ollama group really does hold batch_size of
        # them, but an apple group holds 1-2 whatever the batch size says. The
        # packing is built, so hand the gate its real row count (no-op on a
        # prior an earlier run measured). See annotate._GroupRate.denominate.
        gate.rate.denominate(annotate._mean_group_rows(groups))

    def request(group):
        return annotate._annotate_group(group, model, think)

    items = groups if gate is None else gate.dispatchable(groups)
    n_done = n_skipped = 0
    # A gated pass stops where the clock says, not where the sample ends, so a
    # total would advertise an ETA for work that was never going to run.
    total = None if gate is not None else len(sample)
    try:
        try:
            with progress.track("Seeding annotations", total=total,
                                header=False) as p:
                for group, results in annotate._pooled(
                        items, request if gate is None else gate.timed(request),
                        workers):
                    for chunk_id, _text in group:
                        data = results.get(chunk_id)
                        if data is None:
                            n_skipped += 1
                        else:
                            annotate._write_annotation(con, chunk_id, data)
                            n_done += 1
                            if n_done % ANNOTATE_COMMIT_EVERY == 0:
                                con.commit()
                        p.advance()
        finally:
            # Every exit path, not just Ctrl-C and success -- the same rule
            # annotate_hybrid._run_upgrades' finally follows, and for a sharper
            # reason here. An ollama timeout propagating out of _pooled used to
            # roll back up to ANNOTATE_COMMIT_EVERY - 1 seed rows the run had
            # already paid for AND skip the counter update for the ones that had
            # committed, so the next run re-bought seeds it already owned and
            # sailed past ANNOTATE_HYBRID_SEED_TARGET -- spending the upgrade
            # phases' share of the budget on a pool that was already deep
            # enough. Committed rows and the counter that vouches for them have
            # to move together or neither number means anything. Runs exactly
            # once, so the success path below no longer records a second time;
            # the clamp stays where it always was, in `n_seeded` above.
            con.commit()
            _record_seeds(con, counter_key, n_seeded, n_done, stats)
    except KeyboardInterrupt:
        print(f"\nInterrupted -- committed {n_done} seed annotations "
              f"({n_skipped} skipped); rerun to resume.")
        if interrupt_exit:
            raise SystemExit(1) from None
        raise
    # A budgeted caller can stop this pass short, and the pool level is what
    # every later phase depends on -- so say the shortfall out loud rather than
    # letting "Seeded 412" read as "done".
    shortfall = ""
    if gate is not None and gate.stopped:
        shortfall = (f" -- stopped at the seed phase's share of the budget with "
                     f"{n_seeded + n_done} of {target} seeds; rerun to top up")
    print(f"Seeded {n_done} annotations ({n_skipped} skipped){shortfall}")
    return n_done


def _record_seeds(con, counter_key, n_seeded, n_done, stats=None):
    """Bank this pass's draws under `counter_key` and report the count out.
    Called from _seed_chunks' finally, so it runs on the interrupt and the
    error paths too, not just on success: those rows are committed, and a
    counter that forgot them would buy them a second time on the next run."""
    if stats is not None:
        stats["done"] = n_done
    if counter_key and n_done:
        set_state(con, counter_key, str(n_seeded + n_done))
        con.commit()


def _knn_predict(con, labels, emb, k):
    """Distance-weighted vote of the k nearest seed annotations (cosine, via
    the temp.seed_vectors vec0 table). Returns
    (valence, emotions, depth, confidence).

    `confidence` is the vote's own deterministic 0..1 self-assessment --
    mean(w) * (1 - clamp(weighted std of the neighbours' valence, 0, 1)) --
    so it is high only when the neighbours were BOTH close (large weights)
    and in agreement (small spread), and monotone in each. Bounded by
    construction rather than by clipping the result: cosine distance is 0..2
    so every w lands in [1e-6, 1] and their mean in [0, 1], and the second
    factor is 1 minus a value clamped to [0, 1]. The hybrid tier ranks its
    LLM upgrades by this, lowest first -- spend the budget where kNN is least
    likely to be right."""
    nn = con.execute(
        "SELECT chunk_id, distance FROM temp.seed_vectors "
        "WHERE embedding MATCH ? AND k = ?", (emb, k)).fetchall()
    if not nn:
        return 0.0, [], 1, 0.0
    tw = val = val_sq = dep = 0.0
    escore = {}
    for cid, d in nn:
        w = max(1e-6, 1.0 - d)  # cosine distance is 0..2
        lv, le, ld = labels[cid]
        tw += w
        val += w * lv
        val_sq += w * lv * lv
        dep += w * ld
        for e in le:
            escore[e] = escore.get(e, 0.0) + w
    emotions = sorted(escore, key=escore.get, reverse=True)
    emotions = ([e for e in emotions[:3] if escore[e] / tw >= _KNN_EMOTION_MIN_SCORE]
                or emotions[:1])
    valence = val / tw
    # Weighted variance in one pass (E[x^2] - E[x]^2); rounding can push a
    # unanimous vote a hair below zero, so floor it before the sqrt.
    spread = math.sqrt(max(0.0, val_sq / tw - valence * valence))
    confidence = (tw / len(nn)) * (1.0 - min(1.0, spread))
    return valence, emotions, max(1, min(5, round(dep / tw))), confidence


def _require_chunk_vectors(con):
    """The kNN passes all read chunk_vectors directly, so the missing-table
    error has to be raised before the first SELECT touches it -- earlier than
    _build_seed_index runs in _classify_chunks (which queries the table to size
    its work before deciding whether a seed index is needed at all). Hence its
    own helper; the repeated sqlite_master lookup is free."""
    if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone():
        raise SystemExit(
            "chunk_vectors is missing -- run ingest.py (with embedding) first, "
            "or annotate with an LLM tier (--quality fast/balanced/max).")


def _chat_sort_key(chat_id):
    """Total order over chat ids that tolerates the NULL chunks.chat_id allows
    (None sorts last), matching annotate_sampling's own tie-break."""
    return (chat_id is None, chat_id or 0)


def _corpus_quotas(pool_by_chat, corpus_sizes, n):
    """Split `n` index slots across the chats present in `pool_by_chat`,
    weighted by each chat's share of the CORPUS and capped by what the pool
    actually holds for it. Largest-remainder, with the leftover from
    pool-starved chats redistributed over the chats that still have room, so
    the quotas always sum to exactly n. Pure and deterministic.

    Weighting by corpus share rather than by pool share is the whole point:
    a pool-proportional draw cannot fix a lopsided pool, because it reproduces
    the lopsidedness exactly (measured: 0.58% WORSE than uniform, which is why
    the earlier "stratifying the index sample would be chasing noise" note was
    right about the thing it was actually about, and does not extend to this).
    Remainders are compared as exact integers, never floats, so the tie-break
    cannot drift."""
    quota = dict.fromkeys(pool_by_chat, 0)
    room = {c: len(ids) for c, ids in pool_by_chat.items() if ids}
    remaining = min(n, sum(room.values()))
    while remaining > 0 and room:
        # Weight is the corpus share; a chat the corpus no longer knows about
        # still gets a floor of 1 so it cannot become unreachable.
        weights = {c: max(1, corpus_sizes.get(c, 0)) for c in room}
        total = sum(weights.values())
        parts = []
        for chat, cap_c in room.items():
            num = remaining * weights[chat]
            parts.append([chat, min(num // total, cap_c), num % total])
        extra = max(0, remaining - sum(p[1] for p in parts))
        for part in sorted(parts, key=lambda p: (-p[2], _chat_sort_key(p[0])))[:extra]:
            if part[1] < room[part[0]]:
                part[1] += 1
        progressed = False
        for chat, add, _rem in parts:
            if not add:
                continue
            quota[chat] += add
            remaining -= add
            room[chat] -= add
            progressed = True
        room = {c: k for c, k in room.items() if k > 0}
        if not progressed:
            # A pathological weight/room combination could round every share to
            # zero; hand the rest out in a fixed order rather than spin.
            for chat in sorted(room, key=_chat_sort_key):
                if remaining <= 0:
                    break
                take = min(remaining, room[chat])
                quota[chat] += take
                remaining -= take
            break
    return quota


def _sample_seed_ids(seed_ids, cap, floor=_KNN_MIN_SEEDS, chat_of=None,
                     corpus_sizes=None):
    """Which of `seed_ids` belong in the temp kNN index: all of them when the
    cap is off (<= 0) or the pool already fits under it, otherwise a
    deterministic subset. Pure -- no DB, no writes, no global state.

    `cap` is raised to `floor` before it bites (and the result is never bigger
    than the pool), so a small cap can shrink the index but can never take it
    below the same minimum _build_seed_index's pool check enforces: the
    guarantee that every kNN vote is drawn from >= _KNN_MIN_SEEDS rows holds
    for any cap value.

    Deterministic AND stable across calls by construction, which the two
    callers depend on -- _classify_chunks and backfill_confidence each rebuild
    the index inside one run and must vote off the same one. Three things buy
    that: the ids are sorted before the draw, the per-chat draws happen in a
    fixed chat order, and the randomness comes from a fresh random.Random(0)
    rather than the module-global RNG (_seed_chunks samples from that one;
    consuming or reseeding it here would silently change the seed draw as a
    side effect, and would make this sample depend on how much randomness the
    process had already spent).

    `chat_of` ({chunk_id: chat_id}) and `corpus_sizes` ({chat_id: chunks in
    that chat}) turn the draw from uniform-over-the-pool into
    proportional-to-the-CORPUS, which is what keeps a lopsided pool from
    becoming a lopsided index. Omit either and the historical uniform draw is
    used unchanged -- that fallback is what every existing caller and test
    still exercises.

    Why it was needed, measured 2026-07-30 on the two real corpora: the pool is
    every source='llm' row -- sampler seeds, phase-3 upgrades and rows
    inherited from an earlier scoped LLM run alike -- so it is neither random
    nor stratified, and an earlier version of this docstring claiming "the pool
    a tier feeds this is already stratified by its own sampler" was wrong on
    any corpus with inherited rows. On the 25,029-chunk reference corpus ALL
    4,416 llm rows sit in ONE chat (38.2% of the corpus), so the shipped index
    represented 1 of 532 chats and the other 61.8% of the corpus was classified
    by a vote drawn entirely from one dyad -- a chunk in that chat sat 0.1120
    from its nearest seed against 0.2718 for a chunk anywhere else, 2.43x
    further. On the 74,233-chunk corpus that has actually run the hybrid tier:
    uniform gave 58 of 535 chats at 0.1554 mean nearest-seed distance and 548.0
    summed deviation from proportional; this draw gives 73 chats at 0.1478 and
    165.4, against a salt-to-salt noise floor of ~0.0005.

    The redistribution is deliberate and has a named cost: on that corpus the
    dominant chat's own chunks go from 0.1216 to 0.1283 (5.5% worse) to move
    the other 531 chats from 0.1766 to 0.1601 (9.3% better). A corpus that
    really is one person may not want that trade; ANNOTATE_KNN_INDEX_CAP = 0
    turns the cap -- and so this draw -- off entirely.

    Scope of the claim: this is a GEOMETRY result, not a demonstrated label
    win. Neither corpus holds held-out LLM labels outside the dominant chat, so
    accuracy off the corpus's other 61.8% cannot be measured, and the cap's own
    documented churn (below) means any new draw is a third sample of the same
    urn. What the cap costs the labels, measured 2026-07-28 (2,000 held-out
    chunks re-classified at a 4,416 pool, capped vs uncapped): valence r=0.90,
    depth 85.1%, top-1 emotion 59.1% -- against a control where a DIFFERENT
    800-sample disagrees by the same margins (r=0.90, top-1 58.0%), so that is
    the vote's own sensitivity to which seeds are in the urn and not a cost the
    cap adds. Those figures are sample-dependent: a contiguous chunk-id block
    (chat-correlated, so a harder case) measured r=0.738 / 65.0% / 39.6%. See
    config.ANNOTATE_KNN_INDEX_CAP; the earlier "~1% at k=10" claim was wrong
    and is retracted there."""
    n = min(len(seed_ids), max(cap, floor))
    if cap <= 0 or n >= len(seed_ids):
        return set(seed_ids)
    rng = random.Random(0)
    if not chat_of or not corpus_sizes:
        # No chat information: the historical uniform draw, unchanged.
        return set(rng.sample(sorted(seed_ids), n))
    pool_by_chat = {}
    for cid in sorted(seed_ids):
        pool_by_chat.setdefault(chat_of.get(cid), []).append(cid)
    quota = _corpus_quotas(pool_by_chat, corpus_sizes, n)
    keep = set()
    # Fixed chat order so the single RNG's stream cannot depend on dict order.
    for chat in sorted(pool_by_chat, key=_chat_sort_key):
        want = quota.get(chat, 0)
        ids = pool_by_chat[chat]
        if want >= len(ids):
            keep.update(ids)
        elif want:
            keep.update(rng.sample(ids, want))
    return keep


def _build_seed_index(con):
    """(Re)build temp.seed_vectors from the source='llm' annotations that have
    embeddings and return their {chunk_id: (valence, emotions, depth)} labels
    -- everything _knn_predict needs. Shared by _classify_chunks and
    backfill_confidence so the two passes can never vote off different
    indexes, which ANNOTATE_KNN_INDEX_CAP preserves: the draw is a pure
    function of the pool's ids (see _sample_seed_ids), so both callers build
    the same index from the same pool.

    That cap bounds the INDEX only, never the pool -- every source='llm' row
    stays written and every other reader still sees all of them. It exists
    because the kNN passes cost O(rows x seeds), one vec0 MATCH per row
    against every indexed seed, and the hybrid tier widens the pool on every
    run: uncapped, this phase gets slower forever (measured 2026-07-28, same
    corpus and machine: 14.6 s at a 450-seed index vs 82.0 s at 4,416).
    The labels are filtered WITH the index and from the same list, never
    separately -- _knn_predict looks its neighbours up in this dict, so an id
    in one and not the other is a KeyError or a dropped vote.

    WHICH rows the cap keeps is decided proportionally to each chat's share of
    the CORPUS, not uniformly over the pool: the pool is whatever earlier runs
    happened to label, and on both real corpora that is overwhelmingly one
    dyad, so a uniform draw handed most of the corpus a vote from a chat it has
    nothing to do with. See _sample_seed_ids for the measurements and for the
    redistribution's named cost. The two extra reads that buys -- chat_id on
    the pool rows, and one corpus-wide chunk count per chat -- are the same
    shape annotate_sampling already runs and measured ~80 ms on the larger
    corpus; the chunks join also drops any annotation whose chunk was removed
    by a tail re-chunk, which could not have been voted with anyway."""
    _require_chunk_vectors(con)
    # ORDER BY: the row order of an unordered SELECT is not guaranteed stable,
    # and both the sample and the vec0 insert order have to be reproducible
    # run to run. (_sample_seed_ids sorts its input again, so it is correct
    # whatever a caller hands it; this makes the whole function reproducible.)
    seeds = con.execute("""
        SELECT a.chunk_id, v.embedding, a.valence, a.emotions_json, a.depth,
               c.chat_id
        FROM chunk_annotations a
        JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
        JOIN chunks c ON c.id = a.chunk_id
        WHERE a.source = 'llm' ORDER BY a.chunk_id
    """).fetchall()
    # The floor is a question about the POOL -- "are there enough LLM seeds to
    # vote with at all" -- so it is checked BEFORE the cap and against the
    # uncapped count. Capping is a speed knob and must not be able to invent
    # this error on a healthy pool (nor could a cap hide it: _sample_seed_ids
    # never returns fewer than _KNN_MIN_SEEDS ids from a pool that has them).
    if len(seeds) < _KNN_MIN_SEEDS:
        raise SystemExit(
            f"only {len(seeds)} LLM seed annotations with embeddings "
            f"(need >= {_KNN_MIN_SEEDS}) -- run the seed pass first")
    # Read only when the cap can actually bite, so an uncapped run (or a pool
    # already under the cap) pays nothing for the corpus count.
    chat_of = corpus_sizes = None
    if 0 < ANNOTATE_KNN_INDEX_CAP < len(seeds):
        chat_of = {row[0]: row[5] for row in seeds}
        corpus_sizes = dict(con.execute(
            "SELECT chat_id, COUNT(*) FROM chunks GROUP BY chat_id").fetchall())
    keep = _sample_seed_ids([cid for cid, *_rest in seeds], ANNOTATE_KNN_INDEX_CAP,
                            chat_of=chat_of, corpus_sizes=corpus_sizes)
    if len(keep) < len(seeds):
        global _last_index_report
        report = (len(keep), len(seeds))
        if report != _last_index_report:
            print(f"using {len(keep):,} of {len(seeds):,} seeds for the kNN index")
            _last_index_report = report
        seeds = [row for row in seeds if row[0] in keep]
    dim = int(get_state(con, "embed_dim") or 0)
    if not dim:
        raise SystemExit("no embed_dim in ingest_state -- run ingest.py first")
    con.execute("DROP TABLE IF EXISTS temp.seed_vectors")
    con.execute(
        f"CREATE VIRTUAL TABLE temp.seed_vectors USING vec0("
        f"chunk_id integer PRIMARY KEY, "
        f"embedding FLOAT[{dim}] distance_metric=cosine)")
    con.executemany(
        "INSERT INTO temp.seed_vectors(chunk_id, embedding) VALUES (?, ?)",
        [(cid, emb) for cid, emb, *_ in seeds])
    return {cid: (val, json.loads(ej), dep)
            for cid, _emb, val, ej, dep, _chat in seeds}


# How often the kNN passes read the clock. Per row would be correct and free
# next to a ~3 ms vec0 MATCH, but these loops also run ungated (the distilled
# tier passes no deadline) and a stride keeps the branch off that hot path;
# 256 rows is well under a second of work at any measured rate.
_DEADLINE_CHECK_EVERY = 256


def _past(deadline, n_done):
    """True when a gated kNN loop should stop. Checked on a stride, so the
    overshoot is bounded by _DEADLINE_CHECK_EVERY rows, not by the pass."""
    return (deadline is not None and n_done % _DEADLINE_CHECK_EVERY == 0
            and time.monotonic() >= deadline)


def _classify_chunks(con, k, dry_run=False, deadline=None, stats=None):
    """kNN-classify every un-annotated chunk that has an embedding, writing
    source='distilled' rows (valence/emotions/depth only -- events need an
    LLM tier; see --upgrade-distilled).

    `deadline` stops the loop where the clock says (the pass is resumable --
    it only ever looks at chunks with no annotation at all -- so the next run
    picks the rest up). `stats`, when given, is a caller-owned dict this sets
    'stopped'/'interrupted' in: the two are different news to a budgeted
    caller, and neither is visible in the row count."""
    _require_chunk_vectors(con)
    todo = con.execute("""
        SELECT v.chunk_id, v.embedding FROM chunk_vectors v
        LEFT JOIN chunk_annotations a ON a.chunk_id = v.chunk_id
        WHERE a.chunk_id IS NULL ORDER BY v.chunk_id
    """).fetchall()
    print(f"{len(todo)} chunks to kNN-classify (k={k})")
    if dry_run or not todo:
        return 0

    labels = _build_seed_index(con)

    n_done = 0
    try:
        with progress.track("Classifying chunks (kNN)", total=len(todo),
                            header=False) as p:
            for chunk_id, emb in todo:
                if _past(deadline, n_done):
                    if stats is not None:
                        stats["stopped"] = True
                    break
                valence, emotions, depth, conf = _knn_predict(con, labels, emb, k)
                con.execute(
                    "INSERT OR REPLACE INTO chunk_annotations"
                    "(chunk_id, valence, emotions_json, depth, events_json, "
                    "source, knn_confidence) "
                    "VALUES (?, ?, ?, ?, '[]', 'distilled', ?)",
                    (chunk_id, valence, json.dumps(emotions), depth, conf),
                )
                n_done += 1
                p.advance()
                # Writes are ~ms each; commit far less often than LLM passes.
                if n_done % (ANNOTATE_COMMIT_EVERY * 20) == 0:
                    con.commit()
    except KeyboardInterrupt:
        con.commit()
        if stats is not None:
            stats["interrupted"] = True
        print(f"\nInterrupted -- committed {n_done} distilled annotations; "
              f"rerun to resume.")
        return n_done
    con.commit()
    tail = "" if n_done == len(todo) else f" of {len(todo)} (out of budget; rerun)"
    print(f"Classified {n_done} chunks{tail}")
    return n_done


def backfill_confidence(con, k, dry_run=False, deadline=None, stats=None,
                        has_confidence=True):
    """Fill knn_confidence in for source='distilled' rows that lack it, by
    re-running the same kNN vote (deterministic, no LLM call). Needed because
    _classify_chunks only touches chunks with NO annotation at all, so on a
    corpus that already had a distilled pass -- the normal case on the machines
    the hybrid tier targets -- every distilled row predates the column, reads
    NULL, and a lowest-confidence-first ranking would have nothing to rank.

    Deliberately an UPDATE of knn_confidence alone: valence/emotions/depth/
    events/source are already-committed labels and relabeling them here would
    be a different, unrequested change (and would silently move readers that
    trust the existing values). Rows with no chunk_vectors embedding can't be
    scored and are skipped. Returns the number of rows updated -- or, under
    dry_run, the number it would update.

    `deadline`/`stats` work as in _classify_chunks: the pass is resumable (it
    only looks at rows whose confidence is still NULL), and it needs the gate
    because its cost is O(rows x seeds) -- one vec0 MATCH per row against
    every INDEXED source='llm' seed. Measured 2026-07-28, same corpus and
    machine: 14.6 s for 24,579 rows against a 450-seed pool (1,688 rows/s) but
    82.0 s for 20,613 rows against a 4,416-seed one (251 rows/s), a 6.7x
    per-row slowdown for a 9.8x wider pool -- which is what
    ANNOTATE_KNN_INDEX_CAP now bounds (_build_seed_index), so the gate is a
    backstop rather than the only thing between this pass and the budget.
    `has_confidence=False` says the column does not exist yet (only reachable
    under --dry-run, which no longer migrates), in which case every distilled
    row is pending by definition."""
    _require_chunk_vectors(con)
    if dry_run:
        unscored = "a.knn_confidence IS NULL" if has_confidence else "1"
        n = con.execute(f"""
            SELECT COUNT(*) FROM chunk_annotations a
            JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
            WHERE a.source = 'distilled' AND {unscored}
        """).fetchone()[0]
        print(f"{n} distilled rows to backfill with kNN confidence (k={k})")
        return n
    todo = con.execute("""
        SELECT a.chunk_id, v.embedding FROM chunk_annotations a
        JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
        WHERE a.source = 'distilled' AND a.knn_confidence IS NULL
        ORDER BY a.chunk_id
    """).fetchall()
    print(f"{len(todo)} distilled rows to backfill with kNN confidence (k={k})")
    if not todo:
        return 0

    labels = _build_seed_index(con)

    n_done = 0
    try:
        with progress.track("Backfilling kNN confidence", total=len(todo),
                            header=False) as p:
            for chunk_id, emb in todo:
                if _past(deadline, n_done):
                    if stats is not None:
                        stats["stopped"] = True
                    break
                confidence = _knn_predict(con, labels, emb, k)[3]
                con.execute(
                    "UPDATE chunk_annotations SET knn_confidence = ? "
                    "WHERE chunk_id = ?", (confidence, chunk_id))
                n_done += 1
                p.advance()
                # Writes are ~ms each; commit far less often than LLM passes.
                if n_done % (ANNOTATE_COMMIT_EVERY * 20) == 0:
                    con.commit()
    except KeyboardInterrupt:
        con.commit()
        if stats is not None:
            stats["interrupted"] = True
        print(f"\nInterrupted -- committed {n_done} confidence backfills; "
              f"rerun to resume.")
        return n_done
    con.commit()
    tail = "" if n_done == len(todo) else f" of {len(todo)} (out of budget; rerun)"
    print(f"Backfilled confidence for {n_done}{tail} distilled rows")
    return n_done


def distill_chunks(con, model, think=None, workers=1,
                   seed_target=ANNOTATE_SEED_TARGET, k=ANNOTATE_KNN_K,
                   dry_run=False):
    """The distilled quality tier: ensure `seed_target` LLM-labeled seeds
    exist, then kNN-classify the rest of the corpus from chunk_vectors."""
    # _classify_chunks writes knn_confidence, which postdates DBs in the wild.
    # A dry run never reaches that write (it returns on the count line), so it
    # keeps its no-writes promise by skipping the migration entirely.
    annotate._ensure_column(con, "chunk_annotations", "knn_confidence",
                            dry_run=dry_run)
    _seed_chunks(con, model, think, workers, seed_target, dry_run=dry_run)
    return _classify_chunks(con, k, dry_run=dry_run)
