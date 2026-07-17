"""Distilled annotation tier (annotate.py --quality distilled): top up an
LLM-labeled seed pool by annotating a random corpus-wide sample, then
kNN-classify every remaining embedded chunk from its chunk_vectors embedding
-- valence/emotions/depth only, events left empty (source='distilled' marks
the rows). annotate.py owns the CLI and calls distill_chunks; the shared
request/validation/write helpers stay in annotate and are resolved late
(module-attribute access), so the two modules can import each other safely.
"""

import json
import random

import annotate
import progress
from config import ANNOTATE_COMMIT_EVERY, ANNOTATE_KNN_K, ANNOTATE_SEED_TARGET
from db import get_state

# kNN classification: a candidate emotion label must win at least this share
# of the neighbours' distance-weighted vote. Chosen on the 2026-07-17
# held-out eval (see docs/DECISIONS.md).
_KNN_EMOTION_MIN_SCORE = 0.15
# Fewer LLM seeds than this and the kNN vote is mostly noise -- refuse.
_KNN_MIN_SEEDS = 100


def _seed_chunks(con, model, think, workers, target, dry_run=False):
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
    corpus."""
    n_llm = con.execute(
        "SELECT COUNT(*) FROM chunk_annotations WHERE source = 'llm'"
    ).fetchone()[0]
    need = max(0, target - n_llm)
    print(f"seed pool: {n_llm} LLM annotations (target {target}, "
          f"labeling {need} more)")
    if dry_run or not need:
        return 0
    pending = annotate._pending_chunks(con, all_chats=True, top_dyads=0, limit=None,
                                       upgrade_distilled=True)
    sample = random.sample(pending, min(need, len(pending)))
    if not sample:
        return 0
    annotate._check_model_guard(con, model)

    n_done = n_skipped = 0
    try:
        with progress.track("Seeding annotations", total=len(sample),
                            header=False) as p:
            for group, results in annotate._pooled(
                    [[row] for row in sample],
                    lambda g: annotate._annotate_group(g, model, think), workers):
                chunk_id = group[0][0]
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
        print(f"\nInterrupted -- committed {n_done} seed annotations "
              f"({n_skipped} skipped); rerun to resume.")
        raise SystemExit(1) from None
    con.commit()
    print(f"Seeded {n_done} annotations ({n_skipped} skipped)")
    return n_done


def _knn_predict(con, labels, emb, k):
    """Distance-weighted vote of the k nearest seed annotations (cosine, via
    the temp.seed_vectors vec0 table). Returns (valence, emotions, depth)."""
    nn = con.execute(
        "SELECT chunk_id, distance FROM temp.seed_vectors "
        "WHERE embedding MATCH ? AND k = ?", (emb, k)).fetchall()
    if not nn:
        return 0.0, [], 1
    tw = val = dep = 0.0
    escore = {}
    for cid, d in nn:
        w = max(1e-6, 1.0 - d)  # cosine distance is 0..2
        lv, le, ld = labels[cid]
        tw += w
        val += w * lv
        dep += w * ld
        for e in le:
            escore[e] = escore.get(e, 0.0) + w
    emotions = sorted(escore, key=escore.get, reverse=True)
    emotions = ([e for e in emotions[:3] if escore[e] / tw >= _KNN_EMOTION_MIN_SCORE]
                or emotions[:1])
    return val / tw, emotions, max(1, min(5, round(dep / tw)))


def _classify_chunks(con, k, dry_run=False):
    """kNN-classify every un-annotated chunk that has an embedding, writing
    source='distilled' rows (valence/emotions/depth only -- events need an
    LLM tier; see --upgrade-distilled)."""
    if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone():
        raise SystemExit(
            "chunk_vectors is missing -- run ingest.py (with embedding) first, "
            "or annotate with an LLM tier (--quality fast/balanced/max).")
    todo = con.execute("""
        SELECT v.chunk_id, v.embedding FROM chunk_vectors v
        LEFT JOIN chunk_annotations a ON a.chunk_id = v.chunk_id
        WHERE a.chunk_id IS NULL ORDER BY v.chunk_id
    """).fetchall()
    print(f"{len(todo)} chunks to kNN-classify (k={k})")
    if dry_run or not todo:
        return 0

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
    labels = {cid: (val, json.loads(ej), dep)
              for cid, _emb, val, ej, dep in seeds}

    n_done = 0
    try:
        with progress.track("Classifying chunks (kNN)", total=len(todo),
                            header=False) as p:
            for chunk_id, emb in todo:
                valence, emotions, depth = _knn_predict(con, labels, emb, k)
                con.execute(
                    "INSERT OR REPLACE INTO chunk_annotations"
                    "(chunk_id, valence, emotions_json, depth, events_json, "
                    "source) VALUES (?, ?, ?, ?, '[]', 'distilled')",
                    (chunk_id, valence, json.dumps(emotions), depth),
                )
                n_done += 1
                p.advance()
                # Writes are ~ms each; commit far less often than LLM passes.
                if n_done % (ANNOTATE_COMMIT_EVERY * 20) == 0:
                    con.commit()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- committed {n_done} distilled annotations; "
              f"rerun to resume.")
        return n_done
    con.commit()
    print(f"Classified {n_done} chunks")
    return n_done


def distill_chunks(con, model, think=None, workers=1,
                   seed_target=ANNOTATE_SEED_TARGET, k=ANNOTATE_KNN_K,
                   dry_run=False):
    """The distilled quality tier: ensure `seed_target` LLM-labeled seeds
    exist, then kNN-classify the rest of the corpus from chunk_vectors."""
    _seed_chunks(con, model, think, workers, seed_target, dry_run=dry_run)
    return _classify_chunks(con, k, dry_run=dry_run)
