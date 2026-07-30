"""Stratified, embedding-diverse seed sampler for the hybrid annotate tier
(annotate.py --quality hybrid, via annotate_distill._seed_chunks(sampler=...)).

The distilled tier draws its LLM seed pool with a flat random.sample over the
whole corpus. This module replaces that draw with two levels:

  1. proportional allocation across chats (largest-remainder rounding, so the
     quotas sum to exactly `need` and every chat with a pending chunk has a
     seat at the table rather than one it happened to win), then
  2. k-means++ D-squared selection over chunk_vectors embeddings inside each
     chat, so the picks avoid landing on top of each other in the chat's
     densest conversational mode.

Everything here is read-only SQL and deterministic: two calls with the same
(pending, need, DB) return the identical list, in the same order.

What it actually buys (measured, and smaller than you might assume)
------------------------------------------------------------------
Measured 2026-07-27 against the 25,029-chunk / 532-chat reference corpus
(nomic-embed-text, dim 768), need=800, probing with EVERY corpus chunk:

* Corpus *coverage* is a wash. Mean cosine distance from a chunk to its
  nearest seed: 0.1496 stratified vs 0.1503 for random.sample (-0.4%); p90
  0.2223 vs 0.2282 (-2.6%); worst-covered chunk 0.3547 vs 0.3675 (-3.5%);
  mean distance to the 10 nearest seeds (k = ANNOTATE_KNN_K) 0.1857 vs 0.1830,
  i.e. 1.4% *worse*; mean pairwise distance among the seeds 0.3522 vs 0.3482
  (+1.1%). At 768 dimensions, 800 seeds over 25k chunks are already spread
  near-uniformly by chance, and no selection rule moves the average much. An
  earlier 3,000-probe measurement showed a "-11% covering radius" win that
  did not survive probing the full corpus -- it was sampling noise. Do not
  claim a coverage win for this module; it does not have one.
* Redundant seeds drop ~4x, and that is the real win. Seeds whose nearest
  *other* seed is closer than 0.02 cosine (a near-duplicate: the LLM call
  teaches the kNN vote nothing new): 22/800 for random.sample, 5/800 here.
  Under 0.05: 32 vs 15. Under 0.10: 116 vs 81. In a tier whose whole premise
  is a wall-clock budget, ~2-4% of the seed LLM calls recovered from
  duplicated work is worth 4s of preamble.
* Chat representation stops being a lottery. Over 25 random.sample draws the
  number of chats represented ranged 92-117 and the summed deviation from
  proportional was 208.9 +- 12.0 rows, with a worst single-chat miss of 31.6
  rows on the dominant dyad. The allocation here is deterministic, hits 108
  chats, and its summed deviation is 77.5 rows -- the irreducible
  largest-remainder floor, with every chat within 1 row of proportional.

Performance, and why there is no numpy
--------------------------------------
numpy is NOT a dependency of this project (see requirements.txt / pyproject)
and this sampler is a preamble to a wall-clock-budgeted annotate run, so it has
to be fast in pure CPython. Selection at full embedding width is not: 768-dim
vectors over a 25k-chunk corpus cost `need_c * n_c * 768` multiply-adds per
chat, ~8e8 float ops corpus-wide -- minutes, not seconds.

Two things make it cheap instead. First, every embedding is projected ONCE
through a deterministic sparse Johnson-Lindenstrauss / Achlioptas matrix down
to _PROJ_DIMS coordinates, each row supported on only _PROJ_NNZ of the input
coordinates with random +/-1 signs, so a projection costs
`_PROJ_DIMS * _PROJ_NNZ` gathers done at C speed via operator.itemgetter
rather than `_PROJ_DIMS * dim`. Second, selection runs in that low-dimensional
space against a documented op budget (_SPREAD_DOT_BUDGET); a chat whose
candidate pool would blow its share of the budget is first subsampled with the
seeded RNG, which degrades that chat toward random-within-chat -- i.e. toward
exactly the status quo -- never toward a blown time budget.

Measured on the same corpus/machine (CPython 3.14.6), need=800:

| step                                             | wall     |
|---|---|
| decode 25,029 float32 blobs (array.array)        | 0.01s    |
| project 25,029 vectors (p=32, nnz=32)            | 0.40s    |
| selection dot products                           | ~2.0e6/s |
| stratified_sample(con, pending, 800), end to end | 4.2-4.6s |

random.sample is ~0.01s for the same draw, and ~4.5s against a 25-minute
annotate budget is 0.3% of the run. The op budget makes the cost flat rather
than growing in `need`: need=1200 also lands at 4.5s, need=5000 at 4.0s (the
per-chat pool shrinks as need grows), need == len(pending) at 0.01s (every chat
takes its whole pool, so no vector is read at all). Widening the projection
buys distance fidelity (Pearson r against full-dim cosine: 0.68 at p=32, 0.77
at p=64, 0.83 at p=96/nnz=128) but moved none of the selection metrics above,
while wall clock went 4.5s -> 6.3s -> 9.8s: p=32 it is.

Degradation is always graceful: a missing chunk_vectors table, an unloadable
sqlite-vec extension, a missing/zero embed_dim, a chunk with no vector row, or
a wrong-length blob each fall back to a seeded random draw *within the chat*.
The level-1 allocation survives every one of those paths, so the worst case is
still exactly stratified -- only level 2 is lost.
"""

import array
import random
import sqlite3
from bisect import bisect_right
from itertools import accumulate
from operator import itemgetter, mul

from db import get_state, load_vec_extension

# Width and per-row support of the sparse JL projection. The selection rule
# only ever compares distances against each other, so the projection has to
# preserve their *ordering*, not their values: measured Pearson r against
# full-dim cosine is 0.68 at (32, 32), and the wider settings that push it to
# 0.83 changed none of the selection metrics while costing 2.3x the wall clock
# (see the module docstring). Algorithmic properties of the projection, not
# user tunables -- they do not belong in config.py, same reasoning as
# annotate._APPLE_ANNOTATE_CHAR_BUDGET.
_PROJ_DIMS = 32
_PROJ_NNZ = 32
# Fixed seed for the projection basis: the matrix must be the same on every
# call so the sample is reproducible. Kept separate from the sampling RNG so
# that whether a projection was built at all cannot shift the draw stream.
_PROJ_SEED = 0

# Total selection dot products this module will spend, corpus-wide. At the
# measured ~2.0e6 dots/s (p=32, CPython 3.14) this is ~3s. Each chat's share of
# the budget is proportional to its quota and a chat spends quota_c * pool_c
# dots, so the per-chat candidate cap collapses to one number,
# _SPREAD_DOT_BUDGET // need (see _pool_cap). Sizing note: the reference corpus
# at need=800 wants ~3.2e6 dots un-capped, so the cap bites on the dominant
# dyad (9,554 chunks, thinned to 7,500 candidates) and nowhere else.
_SPREAD_DOT_BUDGET = 6_000_000

# Rows per `chunk_id IN (...)` embedding fetch. SQLite's host-parameter limit
# (SQLITE_MAX_VARIABLE_NUMBER, 999 on older builds) makes a single IN over
# thousands of ids an error, so the fetch is batched; 500 is comfortably under
# the oldest limit and 50 batches cost ~2s over the reference corpus.
_EMB_FETCH_BATCH = 500

# Every stochastic step in this module draws from random.Random(_SEED): the
# sample has to be reproducible so a rerun of an interrupted seed pass makes
# the same choices.
_SEED = 0


def stratified_sample(con, pending, need):
    """pending: list[(chunk_id, text)] as returned by annotate._pending_chunks.
    Returns a list of `need` rows drawn from pending (fewer if pending is
    smaller), stratified per the plan. Deterministic given (pending, need, DB).
    Pure read-only SQL on chunk_vectors/chunks; no writes, no LLM.

    The returned rows are the very objects handed in, so the caller can keep
    keying its annotation writes off them. They come back proportionally
    interleaved across chats rather than in chunk-id order: the caller
    (annotate_distill._seed_chunks) commits every ANNOTATE_COMMIT_EVERY rows
    and is Ctrl-C safe, so any *prefix* of the result being a stratified,
    diversity-ordered sample in its own right is worth the ten lines it costs.
    """
    if need <= 0 or not pending:
        return []
    rng = random.Random(_SEED)

    # Level 1: bucket the pending rows by chat. Chunk -> chat comes from one
    # `SELECT id, chat_id FROM chunks` read into a dict rather than feeding the
    # pending ids into an `IN (?, ?, ...)`: 25k two-column rows is a few ms and
    # a handful of MB, whereas the IN form would trip SQLite's host-parameter
    # limit (999 on older builds) and need batching for no gain -- we want the
    # whole mapping anyway.
    chat_of = dict(con.execute("SELECT id, chat_id FROM chunks").fetchall())
    buckets = {}
    for i, row in enumerate(pending):
        # Rows are indexed, never unpacked, so a duplicate chunk_id in
        # `pending` stays two distinct candidates instead of collapsing.
        buckets.setdefault(chat_of.get(row[0]), []).append(i)

    # Deterministic chat order: biggest pending pool first, chat id as the
    # tie-break (None -- a chunk whose chat row vanished -- sorts last).
    chats = sorted(buckets, key=lambda c: (-len(buckets[c]), c is None, c or 0))
    rank = {c: r for r, c in enumerate(chats)}
    quota = _allocate(chats, {c: len(buckets[c]) for c in chats}, rank, need)

    vectors = _load_vectors(con, pending, buckets, quota)

    picks = {}
    for chat in chats:
        want = quota[chat]
        if want:
            picks[chat] = _pick_within_chat(buckets[chat], want, vectors, rng, need)

    # Proportional (Bresenham-style) interleave: pick j of a chat that is
    # getting q rows is emitted at position (j + 0.5) / q of the stream, so
    # every chat's rows are spread evenly and any prefix is close to the
    # allocation's proportions. Ties break on chat rank then pick index, so
    # the order is fully determined.
    ordered = sorted(
        ((idx, (j + 0.5) / len(chat_picks), rank[chat], j)
         for chat, chat_picks in picks.items()
         for j, idx in enumerate(chat_picks)),
        key=lambda t: t[1:],
    )
    return [pending[idx] for idx, *_ in ordered]


# ---------------------------------------------------------------------------
# Level 1: proportional allocation across chats
# ---------------------------------------------------------------------------

def _allocate(chats, sizes, rank, need):
    """Largest-remainder allocation of `need` across `chats`, weighted by each
    chat's pending count, capped at that count. Returns {chat: quota} summing
    to min(need, total pending).

    Remainders are compared as exact integers (need*n_c divided by the total),
    never floats, so the tie-break can never drift. Weighting by *remaining*
    capacity makes the cap self-enforcing -- floor(R*cap/T) can only reach cap
    when R == T, and then every remainder is 0 so nothing is rounded up -- so
    the redistribution loop below provably settles in one pass. It is kept as a
    loop anyway because the invariant is worth enforcing rather than trusting.
    """
    quota = dict.fromkeys(chats, 0)
    room = {c: sizes[c] for c in chats if sizes[c] > 0}
    remaining = min(need, sum(room.values()))
    while remaining > 0 and room:
        total = sum(room.values())
        parts = []
        for chat, cap in room.items():
            num = remaining * cap
            parts.append([chat, min(num // total, cap), num % total])
        # max(0, ...) only so a negative slice can never silently hand the
        # remainder to the *smallest* fractions; the sum cannot exceed remaining.
        extra = max(0, remaining - sum(p[1] for p in parts))
        # Hand the rounding remainder to the biggest fractional parts first.
        for part in sorted(parts, key=lambda p: (-p[2], rank[p[0]]))[:extra]:
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
        room = {c: n for c, n in room.items() if n > 0}
        if not progressed:
            # Unreachable given the argument above; bail rather than spin.
            break
    return quota


# ---------------------------------------------------------------------------
# Level 2: embedding diversity within a chat
# ---------------------------------------------------------------------------

def _pool_cap(need):
    """Largest candidate pool _spread_select may scan for one chat. Each chat's
    share of _SPREAD_DOT_BUDGET is proportional to its quota, and a chat spends
    quota_c * pool_c dots, so the pool cap collapses to a single corpus-wide
    number and the total is bounded by _SPREAD_DOT_BUDGET by construction."""
    return max(1, _SPREAD_DOT_BUDGET // max(1, need))


def _pick_within_chat(indices, want, vectors, rng, need):
    """Choose `want` of one chat's pending row indices. Embedded candidates go
    through D-squared selection; candidates with no usable vector are a seeded
    random draw appended afterwards (the plan's random fallback)."""
    if want >= len(indices):
        return list(indices)

    pool = indices
    cap = max(want, _pool_cap(need))
    if len(pool) > cap:
        # Over budget for this chat: thin the candidate list first. This
        # degrades the chat toward random-within-chat -- which is exactly the
        # status quo the whole module is improving on -- rather than blowing
        # the wall-clock budget the hybrid tier is built around.
        pool = sorted(rng.sample(pool, cap))

    embedded = [i for i in pool if vectors.get(i) is not None]
    bare = [i for i in pool if vectors.get(i) is None]
    if len(embedded) <= want:
        # Not enough vectors to select on: take every embedded candidate and
        # top up randomly from the rest, preserving the level-1 quota.
        picked = list(embedded)
        if len(picked) < want:
            picked += rng.sample(bare, want - len(picked))
        return picked

    picked = _spread_select([vectors[i] for i in embedded], want, rng, embedded)
    if len(picked) < want:
        # _spread_select stops early only when every remaining candidate is a
        # duplicate of one already chosen (identical chunk text -> identical
        # embedding). Top up deterministically so the level-1 quota still
        # holds -- the caller's contract is an exact row count.
        seen = set(picked)
        picked += [i for i in embedded + bare if i not in seen][:want - len(picked)]
    return picked


def _spread_select(points, want, rng, labels):
    """Diversity selection over unit-norm low-dimensional points: k-means++
    D-squared sampling. Returns entries of `labels` (parallel to `points`) in
    selection order, so a *prefix* of the result is itself a spread-out sample
    -- which is what makes an interrupted seed pass still useful.

    Why D-squared and not the plan's greedy farthest-point (FPS): FPS takes the
    argmax every round, so it walks the convex hull and fills the seed pool with
    outliers. Measured on the reference corpus (800 seeds, full 768-dim cosine)
    FPS pushed the mean distance from a corpus chunk to its nearest seed the
    wrong way -- 0.1455 -> 0.1624, and 0.1830 -> 0.2102 for the mean over the 10
    nearest, which is the neighbourhood _knn_predict actually votes with. It
    bought spread by abandoning the dense middle where most chunks live.
    D-squared draws each pick with probability proportional to its distance from
    the already-chosen set, so it still reaches empty regions but stays
    density-aware, and it leaves those two numbers within 1.5% of random while
    cutting near-duplicate seeds 4x (module docstring). For unit vectors squared
    Euclidean distance is 2*(1 - cos), so weighting by cosine distance IS the
    k-means++ D-squared rule -- no squaring.

    The inner loop is written for CPython speed: one list comprehension of
    C-level dot products, one C-level elementwise min, and accumulate+bisect
    (both C) for the weighted draw -- no per-element Python arithmetic.
    """
    start = rng.randrange(len(points))
    chosen = [start]
    first = points[start]
    # best[i] = cosine distance from candidate i to the nearest chosen point.
    best = [1.0 - sum(map(mul, first, p)) for p in points]
    best[start] = 0.0
    while len(chosen) < want:
        cum = list(accumulate(best))
        if cum[-1] <= 0.0:
            break  # every remaining candidate duplicates a chosen one
        cur = min(bisect_right(cum, rng.random() * cum[-1]), len(best) - 1)
        if best[cur] <= 0.0:
            break
        chosen.append(cur)
        q = points[cur]
        best = list(map(min, best, (1.0 - sum(map(mul, q, p)) for p in points)))
        best[cur] = 0.0
    return [labels[i] for i in chosen]


# ---------------------------------------------------------------------------
# Embeddings: read chunk_vectors, project, normalize
# ---------------------------------------------------------------------------

def _load_vectors(con, pending, buckets, quota):
    """{pending index: projected unit vector} for every candidate a chat might
    actually run farthest-point over. Returns {} on any missing prerequisite
    (no chunk_vectors, no sqlite-vec, no embed_dim), which degrades level 2 to
    seeded random-within-chat while leaving level 1 intact."""
    wanted = []
    for chat, chat_indices in buckets.items():
        # Chats that take their whole pool, or none of it, never need vectors.
        if 0 < quota[chat] < len(chat_indices):
            wanted.extend(chat_indices)
    if not wanted:
        return {}
    dim = _embed_dim(con)
    if not dim or not _vec_readable(con):
        return {}

    by_id = {}
    for i in wanted:
        by_id.setdefault(pending[i][0], []).append(i)
    ids = sorted(by_id)
    project = _projection(dim)
    out = {}
    for start in range(0, len(ids), _EMB_FETCH_BATCH):
        block = ids[start:start + _EMB_FETCH_BATCH]
        placeholders = ",".join("?" * len(block))
        try:
            rows = con.execute(
                f"SELECT chunk_id, embedding FROM chunk_vectors "
                f"WHERE chunk_id IN ({placeholders})", block).fetchall()
        except sqlite3.Error:
            return {}
        for chunk_id, blob in rows:
            vec = _decode(blob, dim)
            if vec is None:
                continue  # wrong-length/NULL blob: treat as un-embedded
            point = _normalize(project(vec))
            if point is None:
                continue  # degenerate (all-zero) projection: no geometry to use
            for i in by_id.get(chunk_id, ()):
                out[i] = point
    return out


def _embed_dim(con):
    try:
        return int(get_state(con, "embed_dim") or 0)
    except (TypeError, ValueError, sqlite3.Error):
        return 0


def _vec_readable(con):
    """True if chunk_vectors can actually be SELECTed from. The caller's
    connection normally arrives with sqlite-vec already loaded (db.connect
    loads it by default), but this module is also reachable from a bare
    connection, so it probes and loads on demand -- and gives up quietly if the
    Python build has no loadable-extension support at all.

    The load arm catches exactly the ways that can fail: sqlite3.Error (the
    extension is absent or the vec0 table won't open), AttributeError (a
    python.org build with no enable_load_extension at all -- see CLAUDE.md's
    install note) and OSError (the .dylib is missing or unreadable). NOT a bare
    Exception: degrading level-2 selection to random is the right answer to a
    missing extension and the wrong one to a bug in this module."""
    if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone():
        return False
    try:
        con.execute("SELECT chunk_id FROM chunk_vectors LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        pass
    try:
        load_vec_extension(con)
        con.execute("SELECT chunk_id FROM chunk_vectors LIMIT 1").fetchone()
    except (sqlite3.Error, AttributeError, OSError):
        return False
    return True


def _decode(blob, dim):
    """chunk_vectors stores raw float32 arrays. array.frombytes decodes at C
    speed (a struct.unpack loop is ~40x slower and would dominate this module).
    """
    if blob is None or len(blob) != dim * 4:
        return None
    vec = array.array("f")
    try:
        vec.frombytes(bytes(blob))
    except (ValueError, TypeError):
        return None
    return vec


def _gatherer(idx):
    """itemgetter that always returns a tuple (it returns a bare scalar for a
    single index, and rejects an empty one)."""
    if not idx:
        return lambda _v: ()
    if len(idx) == 1:
        pos = idx[0]
        return lambda v: (v[pos],)
    return itemgetter(*idx)


def _projection(dim):
    """Build the sparse JL projection for `dim`-wide embeddings: _PROJ_DIMS
    rows, each with _PROJ_NNZ non-zeros split evenly between +1 and -1. The
    balanced signs kill the all-ones direction, which for mean-pooled
    embeddings is a large shared component carrying no discriminative signal.
    Achlioptas' 1/sqrt(s) scaling is omitted deliberately: the projected
    vectors are normalized before use, so a global scale is a no-op.
    """
    rng = random.Random(_PROJ_SEED)
    nnz = max(1, min(_PROJ_NNZ, dim))
    rows = []
    for _ in range(_PROJ_DIMS):
        # random.sample returns its picks in random order, so splitting it in
        # half assigns the signs at random. Sorting the whole list *before* the
        # split would tie a coordinate's sign to its position -- every row would
        # then be roughly "low dims minus high dims", the rows would collapse
        # onto one direction, and fidelity would fall off with nnz instead of
        # improving (measured Pearson r against full-dim cosine at nnz=128:
        # 0.78 with random signs, 0.52 with positional ones; this cost an hour
        # of the wrong conclusion, hence the comment). Each half is sorted after
        # the split, which is free and keeps the gather local.
        idx = rng.sample(range(dim), nnz)
        rows.append((_gatherer(sorted(idx[:nnz // 2])),
                     _gatherer(sorted(idx[nnz // 2:]))))

    def project(vec):
        return [sum(pos(vec)) - sum(neg(vec)) for pos, neg in rows]

    return project


def _normalize(point):
    norm = sum(map(mul, point, point)) ** 0.5
    if not norm:
        return None
    return [c / norm for c in point]
