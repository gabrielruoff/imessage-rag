"""Topic clustering CLI: k-means over chunk_vectors with the assignment step
pushed into sqlite-vec (scratch vec0 centroid table + per-chunk KNN), centroid
updates in plain Python via array('f') -- no numpy. After clustering, an LLM
label pass names each cluster from the chunks nearest its centroid.

Usage: topics.py [--k K] [--relabel]
  --relabel re-runs ONLY the label pass, for ALL clusters (default runs
  clustering, then labels clusters WHERE label IS NULL).
"""

import argparse
import json
import random
import sqlite3
from array import array
from collections import Counter
from math import sqrt

import sqlite_vec

import ollama
from config import TARGET_DB, TOPIC_K, TOPIC_KMEANS_ITERS, TOPIC_LABEL_SAMPLES
from db import get_state, load_vec_extension, set_state

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

_SCRATCH_TABLE = "_topic_centroids"
_EARLY_STOP_FRACTION = 0.005  # stop when <0.5% of chunks changed cluster
_LABEL_MAX_CHARS = 60
_SAMPLE_TEXT_CHARS = 1200

_LABEL_SYSTEM = """You name topics in a personal text-message archive. You are given a few
sample conversation excerpts that all belong to one topic cluster. Respond
with ONLY one JSON object: {"label": <a 2-5 word topic name>}."""

_RETRY_NUDGE = (
    'Respond with only the JSON object {"label": "<2-5 word topic name>"} -- '
    "no prose, nothing else."
)


def _connect():
    con = sqlite3.connect(DB_PATH)
    load_vec_extension(con)
    return con


def _ask_json(messages, model=None):
    """chat_once(format='json') -> parsed dict; one corrective retry on a
    malformed/non-object response, then None (caller skips)."""
    for _ in range(2):
        # Resolved through the ollama module at call time (patch seam).
        reply = ollama.chat_once(messages, model=model, format="json")
        content = reply.get("content") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": _RETRY_NUDGE},
        ]
    return None


def _load_vectors(con):
    """{chunk_id: array('f', ...)} for every embedded chunk, id-ordered so the
    seeded centroid init is deterministic."""
    return {
        chunk_id: array("f", blob)
        for chunk_id, blob in con.execute(
            "SELECT chunk_id, embedding FROM chunk_vectors ORDER BY chunk_id")
    }


def _normalized(vec):
    """L2-normalize (cosine space: only direction matters for a centroid)."""
    norm = sqrt(sum(x * x for x in vec))
    if norm == 0:
        return array("f", vec)
    return array("f", (x / norm for x in vec))


def _mean_vector(vectors, member_ids):
    dim = len(vectors[member_ids[0]])
    sums = [0.0] * dim
    for chunk_id in member_ids:
        vec = vectors[chunk_id]
        for j in range(dim):
            sums[j] += vec[j]
    n = len(member_ids)
    return _normalized(array("f", (s / n for s in sums)))


# ---------------------------------------------------------------------------
# K-means (assignment in sqlite-vec, updates in Python)
# ---------------------------------------------------------------------------

def cluster_chunks(con, k, iters=TOPIC_KMEANS_ITERS):
    """Run k-means over chunk_vectors; returns ({chunk_id: cluster_id}, dim, k)."""
    vectors = _load_vectors(con)
    n = len(vectors)
    ids = list(vectors)
    dim = len(vectors[ids[0]])
    k = min(k, n)
    seeds = random.Random(0).sample(ids, k)
    centroids = {i: vectors[chunk_id] for i, chunk_id in enumerate(seeds)}
    print(f"Clustering {n} chunks into k={k} topics (dim {dim})...")

    assignments = {}
    for iteration in range(iters):
        con.execute(f"DROP TABLE IF EXISTS {_SCRATCH_TABLE}")
        con.execute(f"""
            CREATE VIRTUAL TABLE {_SCRATCH_TABLE} USING vec0(
                centroid_id INTEGER PRIMARY KEY,
                vec FLOAT[{dim}] distance_metric=cosine
            )
        """)
        con.executemany(
            f"INSERT INTO {_SCRATCH_TABLE}(centroid_id, vec) VALUES (?, ?)",
            [(cluster_id, sqlite_vec.serialize_float32(list(vec)))
             for cluster_id, vec in centroids.items()],
        )

        changed = 0
        new_assignments = {}
        for chunk_id in ids:
            row = con.execute(
                f"SELECT centroid_id FROM {_SCRATCH_TABLE} WHERE vec MATCH ? AND k = 1",
                (sqlite_vec.serialize_float32(list(vectors[chunk_id])),),
            ).fetchone()
            new_assignments[chunk_id] = row[0]
            if assignments.get(chunk_id) != row[0]:
                changed += 1
        assignments = new_assignments
        print(f"  iteration {iteration + 1}: {changed}/{n} reassigned")
        if changed < _EARLY_STOP_FRACTION * n:
            break

        members = {}
        for chunk_id, cluster_id in assignments.items():
            members.setdefault(cluster_id, []).append(chunk_id)
        for cluster_id in centroids:
            if members.get(cluster_id):  # empty cluster keeps its old centroid
                centroids[cluster_id] = _mean_vector(vectors, members[cluster_id])

    con.execute(f"DROP TABLE IF EXISTS {_SCRATCH_TABLE}")
    return assignments, dim, k


def persist_clusters(con, assignments, embed_model, dim, k):
    """DELETE+INSERT topics/chunk_topics and record the model:dim:k state.
    Warn-only guard: a rebuild replacing a differently-parameterized run is
    legitimate -- the fix for a mismatch IS the rebuild."""
    new_state = f"{embed_model}:{dim}:{k}"
    prev_state = get_state(con, "topic_model_k")
    if prev_state and prev_state != new_state:
        print(f"warning: existing topics were built as {prev_state}; "
              f"rebuilding as {new_state} replaces them.")
    con.execute("DELETE FROM chunk_topics")
    con.execute("DELETE FROM topics")
    counts = Counter(assignments.values())
    con.executemany(
        "INSERT INTO topics(cluster_id, label, n_chunks) VALUES (?, NULL, ?)",
        sorted(counts.items()),
    )
    con.executemany(
        "INSERT INTO chunk_topics(chunk_id, cluster_id) VALUES (?, ?)",
        assignments.items(),
    )
    set_state(con, "topic_model_k", new_state)
    con.commit()
    print(f"Persisted {len(counts)} topics over {len(assignments)} chunks "
          f"({new_state})")


# ---------------------------------------------------------------------------
# LLM label pass
# ---------------------------------------------------------------------------

def label_clusters(con, relabel=False, samples=TOPIC_LABEL_SAMPLES):
    """Name clusters from their most-central chunks. Default labels only
    clusters WHERE label IS NULL (resumable); --relabel renames all."""
    where = "" if relabel else " WHERE label IS NULL"
    cluster_ids = [r[0] for r in con.execute(
        f"SELECT cluster_id FROM topics{where} ORDER BY cluster_id")]
    if not cluster_ids:
        print("No clusters need labels")
        return 0

    vectors = _load_vectors(con)
    n_vectors = len(vectors)
    members = {}
    for chunk_id, cluster_id in con.execute(
            "SELECT chunk_id, cluster_id FROM chunk_topics"):
        members.setdefault(cluster_id, []).append(chunk_id)

    print(f"Labeling {len(cluster_ids)} clusters "
          f"({samples} sample chunks each)...")
    n_done = 0
    for cluster_id in cluster_ids:
        member_ids = [m for m in members.get(cluster_id, []) if m in vectors]
        if not member_ids:
            continue
        centroid = _mean_vector(vectors, member_ids)
        # vec0 KNN needs an explicit k; rank everything, keep the members.
        ranked = [r[0] for r in con.execute(
            "SELECT chunk_id FROM chunk_vectors WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(list(centroid)), n_vectors),
        )]
        member_set = set(member_ids)
        sample_ids = [cid for cid in ranked if cid in member_set][:samples]
        if not sample_ids:
            # The KNN scan returned none of this cluster's members (it should
            # rank every vector, but an empty IN () is a syntax error -- never
            # let a surprise here take the whole label pass down).
            print(f"  cluster {cluster_id}: no sample chunks ranked, skipped")
            continue
        placeholders = ",".join("?" * len(sample_ids))
        texts = [r[0] for r in con.execute(
            f"SELECT text FROM chunks WHERE id IN ({placeholders})", sample_ids)]
        prompt = "\n\n---\n\n".join(t[:_SAMPLE_TEXT_CHARS] for t in texts if t)

        data = _ask_json([
            {"role": "system", "content": _LABEL_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        label = data.get("label") if isinstance(data, dict) else None
        if not isinstance(label, str) or not label.strip():
            print(f"  cluster {cluster_id}: unusable label response, skipped")
            continue
        label = label.strip()[:_LABEL_MAX_CHARS]
        con.execute("UPDATE topics SET label = ? WHERE cluster_id = ?",
                    (label, cluster_id))
        con.commit()
        n_done += 1
        print(f"  cluster {cluster_id} ({len(member_ids)} chunks): {label}")
    print(f"Labeled {n_done} clusters")
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="K-means topic clustering over chunk embeddings, plus an "
                    "LLM label pass naming each cluster.")
    parser.add_argument("--k", type=int, default=None,
                        help=f"number of clusters (default {TOPIC_K})")
    parser.add_argument("--relabel", action="store_true",
                        help="re-run only the label pass, for all clusters")
    args = parser.parse_args(argv)

    con = _connect()
    try:
        has_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone()
        n_vectors = (con.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
                     if has_table else 0)
        if not n_vectors:
            print("chunk_vectors is missing or empty -- run ingest.py (with "
                  "embedding) first; nothing to cluster.")
            return
        if args.relabel:
            label_clusters(con, relabel=True)
            return

        embed_model = get_state(con, "embed_model") or "unknown"
        k = args.k or TOPIC_K
        assignments, dim, k = cluster_chunks(con, k)
        state_dim = get_state(con, "embed_dim")
        persist_clusters(con, assignments, embed_model,
                         int(state_dim) if state_dim else dim, k)
        label_clusters(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
