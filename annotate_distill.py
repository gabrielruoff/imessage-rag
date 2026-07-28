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
against every seed), so a budgeted caller has to be able to stop them. Every
one of those defaults is today's distilled behavior exactly.
"""

import json
import math
import random
import time

import annotate
import progress
from config import ANNOTATE_COMMIT_EVERY, ANNOTATE_KNN_K, ANNOTATE_SEED_TARGET
from db import get_state, set_state

# kNN classification: a candidate emotion label must win at least this share
# of the neighbours' distance-weighted vote. Chosen on the 2026-07-17
# held-out eval (see docs/DECISIONS.md).
_KNN_EMOTION_MIN_SCORE = 0.15
# Fewer LLM seeds than this and the kNN vote is mostly noise -- refuse.
_KNN_MIN_SEEDS = 100


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

    def request(group):
        return annotate._annotate_group(group, model, think)

    items = groups if gate is None else gate.dispatchable(groups)
    n_done = n_skipped = 0
    # A gated pass stops where the clock says, not where the sample ends, so a
    # total would advertise an ETA for work that was never going to run.
    total = None if gate is not None else len(sample)
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
    except KeyboardInterrupt:
        con.commit()
        _record_seeds(con, counter_key, n_seeded, n_done, stats)
        print(f"\nInterrupted -- committed {n_done} seed annotations "
              f"({n_skipped} skipped); rerun to resume.")
        if interrupt_exit:
            raise SystemExit(1) from None
        raise
    con.commit()
    _record_seeds(con, counter_key, n_seeded, n_done, stats)
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
    Run on the interrupt path too: the rows are committed, so a counter that
    forgot them would buy them a second time on the next run."""
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


def _build_seed_index(con):
    """(Re)build temp.seed_vectors from the source='llm' annotations that have
    embeddings and return their {chunk_id: (valence, emotions, depth)} labels
    -- everything _knn_predict needs. Shared by _classify_chunks and
    backfill_confidence so the two passes can never vote off different
    indexes."""
    _require_chunk_vectors(con)
    seeds = con.execute("""
        SELECT a.chunk_id, v.embedding, a.valence, a.emotions_json, a.depth
        FROM chunk_annotations a JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
        WHERE a.source = 'llm'
    """).fetchall()
    if len(seeds) < _KNN_MIN_SEEDS:
        raise SystemExit(
            f"only {len(seeds)} LLM seed annotations with embeddings "
            f"(need >= {_KNN_MIN_SEEDS}) -- run the seed pass first")
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
            for cid, _emb, val, ej, dep in seeds}


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
    every source='llm' seed. Measured 2026-07-28, same corpus and machine:
    14.6 s for 24,579 rows against a 450-seed pool (1,688 rows/s) but 82.0 s
    for 20,613 rows against a 4,416-seed one (251 rows/s), a 6.7x per-row
    slowdown for a 9.8x wider pool. `has_confidence=False` says the column does not
    exist yet (only reachable under --dry-run, which no longer migrates), in
    which case every distilled row is pending by definition."""
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
