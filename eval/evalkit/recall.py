"""Voyage A/B evals against the production EMBED_MODEL vectors.

Retrieval -- recall@10 of chunk_vectors vs a Voyage-embedded copy, three steps:

  recall-labels  local LLM writes one search query per sampled chunk -> gold set
                 (eval/labels/recall.jsonl, gitignored: it quotes private chats).
                 Hand-written lines {"query": ..., "chunk_ids": [...]} may be appended.
  recall-embed   embeds every chunk in chunk_vectors with Voyage into a separate
                 vec0 DB (eval/results/recall/<model>.db); --pool N embeds only the
                 gold chunks + N random distractors. UPLOADS chunk text to
                 Voyage, so it refuses without --confirm-upload. Resumable.
  recall         embeds each gold query with both models, top-10 KNN over each
                 index (the same unscoped vec0 MATCH search_conversations runs),
                 reports recall@1/@10 and MRR@10.

Gold labels are synthetic and single-chunk, so absolute scores are pessimistic
(an overlapping neighbour chunk counts as a miss); the A/B delta is the signal.

Distilled labels -- distill-ab replays annotate_distill's kNN vote under k-fold
CV over the source='llm' annotations, once per vector space, and reports
agreement with the held-out LLM labels (uploads those chunks to Voyage).
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request

from . import EVAL_DIR, RESULTS_DIR
from .manifest import resolve_db_path, write_json_atomic
from .runner import acquire_ollama_lock, release_ollama_lock

LABELS_PATH = EVAL_DIR / "labels" / "recall.jsonl"
RECALL_DIR = RESULTS_DIR / "recall"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_BATCH = 128  # texts per request; halved on a 400 (per-request token cap)
K = 10

_QUERY_PROMPT = (
    "Below is a stretch of the phone owner's iMessage history. Write the short "
    "search query (3-12 words) the owner might type months later to find this "
    "conversation again. Describe what was discussed in your own words; you may "
    "say who it was with, but do not copy distinctive phrases verbatim.\n\n"
    "{text}\n\nReply with JSON only: {{\"query\": \"...\"}}"
)


def _voyage_db_path(model, tag=""):
    """tag keys the cache to a source DB: chunk ids are only unique within one."""
    return RECALL_DIR / f"{model}{tag}.db"


def _open_voyage_db(model, dim=None, tag=""):
    import db

    RECALL_DIR.mkdir(parents=True, exist_ok=True)
    con = db.connect(str(_voyage_db_path(model, tag)), rows=False)
    if dim is not None:
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}] distance_metric=cosine)"
        )
    return con


def _voyage_embed(texts, model, input_type, api_key, retries=6):
    """One Voyage request -> vectors in input order. Retries 429/5xx with
    backoff; a 400 on a multi-text batch is split in half (token cap)."""
    body = json.dumps({"input": texts, "model": model, "input_type": input_type}).encode()
    req = urllib.request.Request(
        VOYAGE_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())["data"]
            return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        except urllib.error.HTTPError as e:
            if e.code == 400 and len(texts) > 1:
                mid = len(texts) // 2
                return (_voyage_embed(texts[:mid], model, input_type, api_key)
                        + _voyage_embed(texts[mid:], model, input_type, api_key))
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"Voyage {e.code}: {e.read().decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("Voyage request failed after retries")


def _api_key():
    """Call only after resolve_db_path: importing config loads .env (and freezes TARGET_DB)."""
    import os

    import config  # noqa: F401 -- loads .env

    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        print("VOYAGE_API_KEY is not set", file=sys.stderr)
        raise SystemExit(2)
    return key


def cmd_labels(args):
    db_path = resolve_db_path(args.db)
    import db
    import ollama
    import progress

    con = db.connect(str(db_path))
    ids = [r[0] for r in con.execute(
        "SELECT c.id FROM chunks c JOIN chunk_vectors v ON v.chunk_id = c.id "
        "WHERE c.n_messages >= ? AND length(c.text) BETWEEN ? AND ? ORDER BY c.id",
        (args.min_messages, args.min_chars, args.max_chars))]
    picked = random.Random(args.seed).sample(ids, min(args.n, len(ids)))
    LABELS_PATH.parent.mkdir(exist_ok=True)
    done = 0
    acquire_ollama_lock("recall-labels")
    try:
        with open(LABELS_PATH, "w") as out, \
                progress.track("Writing gold queries", total=len(picked), log_every=1) as p:
            for cid in picked:
                text = con.execute("SELECT text FROM chunks WHERE id = ?", (cid,)).fetchone()[0]
                try:
                    reply = ollama.chat_once_json(
                        [{"role": "user", "content": _QUERY_PROMPT.format(text=text)}],
                        think=False)
                    query = (reply or {}).get("query", "").strip()
                except Exception as e:  # one bad generation must not sink the set
                    print(f"  chunk {cid}: {e}", file=sys.stderr)
                    query = ""
                if query:
                    out.write(json.dumps({"query": query, "chunk_ids": [cid]}) + "\n")
                    done += 1
                p.advance()
    finally:
        release_ollama_lock()
    print(f"wrote {done} gold queries -> {LABELS_PATH}")
    return 0


def cmd_embed(args):
    if not args.confirm_upload:
        print("recall-embed sends every chunk's text to Voyage; pass --confirm-upload",
              file=sys.stderr)
        return 2
    db_path = resolve_db_path(args.db)
    api_key = _api_key()
    import db

    src = db.connect(str(db_path), rows=False)
    dst = _open_voyage_db(args.model, dim=args.dim)
    have = {r[0] for r in dst.execute("SELECT chunk_id FROM chunk_vectors")}
    wanted = [r[0] for r in src.execute("SELECT chunk_id FROM chunk_vectors ORDER BY chunk_id")]
    if args.pool:
        # gold chunks + seeded random distractors; `recall` scores both models on this pool
        gold = {c for lab in _load_labels() for c in lab["chunk_ids"]}
        rest = [c for c in wanted if c not in gold]
        sample = set(random.Random(args.seed).sample(rest, min(args.pool, len(rest))))
        wanted = sorted(gold | sample)
    print(f"{len(have)} already embedded, {len(set(wanted) - have)} to go")
    _embed_missing(src, dst, wanted, args.model, api_key)
    return 0


def _load_labels():
    if not LABELS_PATH.exists():
        return []
    return [json.loads(line) for line in open(LABELS_PATH) if line.strip()]


def _rank(con, table, vec, gold, k=K):
    """1-based rank of the first gold chunk in the top k, else None."""
    import sqlite_vec

    rows = con.execute(
        f"SELECT chunk_id FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(vec), k)).fetchall()
    return next((i + 1 for i, (cid,) in enumerate(rows) if cid in gold), None)


def _summary(ranks):
    n = len(ranks)
    return {
        "n": n,
        "recall@1": round(sum(r == 1 for r in ranks) / n, 4),
        f"recall@{K}": round(sum(r is not None for r in ranks) / n, 4),
        f"mrr@{K}": round(sum(1 / r for r in ranks if r) / n, 4),
    }


def cmd_recall(args):
    db_path = resolve_db_path(args.db)
    import config
    import db
    import embeddings

    labels = _load_labels()
    if not labels:
        print(f"no gold queries in {LABELS_PATH}; run recall-labels first", file=sys.stderr)
        return 2
    base = db.connect(str(db_path), rows=False)
    n_base = base.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    backends = {config.EMBED_MODEL: (base, "chunk_vectors")}
    scope = f"full corpus ({n_base} chunks)"
    vpath = _voyage_db_path(args.model)
    if vpath.exists():
        vcon = _open_voyage_db(args.model)
        pool = [r[0] for r in vcon.execute("SELECT chunk_id FROM chunk_vectors")]
        missing = {c for lab in labels for c in lab["chunk_ids"]} - set(pool)
        if missing:
            print(f"{len(missing)} gold chunks are not in {vpath}; rerun recall-embed",
                  file=sys.stderr)
            return 2
        if len(pool) < n_base:
            # Pool mode: restrict the baseline to the same chunks so both models
            # rank against identical distractors.
            dim = base.execute("SELECT value FROM ingest_state WHERE key = 'embed_dim'"
                               ).fetchone()[0]
            base.execute("CREATE VIRTUAL TABLE temp.pool_vectors USING vec0("
                         f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}] "
                         "distance_metric=cosine)")
            for i in range(0, len(pool), 500):
                ids = pool[i:i + 500]
                base.execute(
                    "INSERT INTO temp.pool_vectors SELECT chunk_id, embedding FROM "
                    f"chunk_vectors WHERE chunk_id IN ({','.join('?' * len(ids))})", ids)
            backends[config.EMBED_MODEL] = (base, "temp.pool_vectors")
            scope = f"pool of {len(pool)}/{n_base} chunks (gold + random distractors)"
        backends[args.model] = (vcon, "chunk_vectors")
    else:
        print(f"(no {vpath}; reporting the {config.EMBED_MODEL} baseline only)")
    print(f"scope: {scope}")

    queries = [lab["query"] for lab in labels]
    qvecs = {config.EMBED_MODEL: embeddings.embed_texts(queries, kind="query")}
    if args.model in backends:
        api_key = _api_key()
        qvecs[args.model] = [v for i in range(0, len(queries), VOYAGE_BATCH)
                             for v in _voyage_embed(queries[i:i + VOYAGE_BATCH],
                                                    args.model, "query", api_key)]

    per_query, report = [dict(lab) for lab in labels], {}
    for name, (con, table) in backends.items():
        ranks = [_rank(con, table, v, set(lab["chunk_ids"]))
                 for v, lab in zip(qvecs[name], labels)]
        for row, r in zip(per_query, ranks):
            row[f"rank:{name}"] = r
        report[name] = _summary(ranks)
    out = RECALL_DIR / f"recall-{time.strftime('%Y%m%d-%H%M%S')}.json"
    RECALL_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, {"scope": scope, "summary": report, "queries": per_query})
    for name, s in report.items():
        print(f"{name:>24}  " + "  ".join(f"{k}={v}" for k, v in s.items()))
    print(f"per-query ranks -> {out}")
    return 0


def _embed_missing(src, dst, ids, model, api_key):
    """Voyage-embed whichever of `ids` (chunk ids in src) dst lacks; commits per batch."""
    import progress
    import sqlite_vec

    have = {r[0] for r in dst.execute("SELECT chunk_id FROM chunk_vectors")}
    todo = [c for c in ids if c not in have]
    with progress.track("Voyage embedding", total=len(todo)) as p:
        for i in range(0, len(todo), VOYAGE_BATCH):
            batch = todo[i:i + VOYAGE_BATCH]
            marks = ",".join("?" * len(batch))
            texts = dict(src.execute(f"SELECT id, text FROM chunks WHERE id IN ({marks})", batch))
            vecs = _voyage_embed([texts[c] for c in batch], model, "document", api_key)
            dst.executemany(
                "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
                [(c, sqlite_vec.serialize_float32(v)) for c, v in zip(batch, vecs)])
            dst.commit()  # per batch: an interrupted run resumes without re-paying
            p.advance(len(batch))


def _distill_scores(con, vecs, labels, train, test, chat_of, corpus_sizes, dim):
    """Replay the production distilled vote: an ANNOTATE_KNN_INDEX_CAP-capped
    seed index drawn by annotate_distill._sample_seed_ids, then
    _knn_predict per test chunk. Returns one score dict per test chunk."""
    import annotate_distill
    import config

    keep = annotate_distill._sample_seed_ids(train, config.ANNOTATE_KNN_INDEX_CAP,
                                             chat_of=chat_of, corpus_sizes=corpus_sizes)
    con.execute("DROP TABLE IF EXISTS temp.seed_vectors")
    con.execute("CREATE VIRTUAL TABLE temp.seed_vectors USING vec0("
                f"chunk_id integer PRIMARY KEY, embedding FLOAT[{dim}] distance_metric=cosine)")
    con.executemany("INSERT INTO temp.seed_vectors(chunk_id, embedding) VALUES (?, ?)",
                    [(c, vecs[c]) for c in sorted(keep)])
    seed_labels = {c: labels[c] for c in keep}
    out = []
    for c in test:
        val, emos, dep, _conf = annotate_distill._knn_predict(
            con, seed_labels, vecs[c], config.ANNOTATE_KNN_K)
        tv, te, td = labels[c]
        union = set(emos) | set(te)
        out.append({
            "chunk_id": c,
            "abs_err": abs(val - tv),
            "depth_pm1": abs(dep - td) <= 1,
            "jaccard": len(set(emos) & set(te)) / len(union) if union else 1.0,
            "top1": bool(emos and te and emos[0] in te),
        })
    return out


def _agg(rows):
    n = len(rows)
    return {
        "n": n,
        "valence_mae": round(sum(r["abs_err"] for r in rows) / n, 4),
        "depth_pm1": round(sum(r["depth_pm1"] for r in rows) / n, 4),
        "emotion_jaccard": round(sum(r["jaccard"] for r in rows) / n, 4),
        "emotion_top1_in_set": round(sum(r["top1"] for r in rows) / n, 4),
    }


def _paired_ci(a, b, key, reps=2000):
    """95% paired-bootstrap CI of mean(b[key]) - mean(a[key])."""
    rng, n = random.Random(0), len(a)
    diffs = sorted(
        sum(float(b[i][key]) - float(a[i][key]) for i in idx) / n
        for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(reps)))
    return round(diffs[int(reps * .025)], 4), round(diffs[int(reps * .975)], 4)


def cmd_distill_ab(args):
    if not args.confirm_upload:
        print("distill-ab sends every LLM-labeled chunk's text to Voyage; pass --confirm-upload",
              file=sys.stderr)
        return 2
    db_path = resolve_db_path(args.db)
    api_key = _api_key()
    import config
    import db

    base = db.connect(str(db_path), rows=False)
    rows = base.execute("""
        SELECT a.chunk_id, v.embedding, a.valence, a.emotions_json, a.depth, c.chat_id
        FROM chunk_annotations a
        JOIN chunk_vectors v ON v.chunk_id = a.chunk_id
        JOIN chunks c ON c.id = a.chunk_id
        WHERE a.source = 'llm' ORDER BY a.chunk_id""").fetchall()
    labels = {r[0]: (r[2], json.loads(r[3]), r[4]) for r in rows}
    chat_of = {r[0]: r[5] for r in rows}
    corpus_sizes = dict(base.execute("SELECT chat_id, COUNT(*) FROM chunks GROUP BY chat_id"))
    ids = sorted(labels)
    tag = "-" + db_path.stem.replace(":", "_")
    vcon = _open_voyage_db(args.model, dim=args.dim, tag=tag)
    _embed_missing(base, vcon, ids, args.model, api_key)

    backends = {
        config.EMBED_MODEL: (base, {r[0]: r[1] for r in rows},
                             int(db.get_state(base, "embed_dim"))),
        args.model: (vcon, dict(vcon.execute("SELECT chunk_id, embedding FROM chunk_vectors")),
                     args.dim),
    }
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    per = {name: [] for name in backends}
    for f, test in enumerate(folds):
        train = sorted(set(ids) - set(test))
        for name, (con, vecs, dim) in backends.items():
            per[name] += _distill_scores(con, vecs, labels, train, test,
                                         chat_of, corpus_sizes, dim)
        print(f"fold {f + 1}/{args.folds} done")

    top_chat = max(set(chat_of.values()), key=list(chat_of.values()).count)
    nomic, voy = (sorted(per[n], key=lambda r: r["chunk_id"]) for n in backends)
    report = {}
    for scope, keep in (("all", lambda r: True),
                        (f"excluding dominant chat {top_chat}",
                         lambda r: chat_of[r["chunk_id"]] != top_chat)):
        a = [r for r in nomic if keep(r)]
        b = [r for r in voy if keep(r)]
        report[scope] = {
            config.EMBED_MODEL: _agg(a), args.model: _agg(b),
            f"delta {args.model} - {config.EMBED_MODEL} (95% CI)": {
                k: _paired_ci(a, b, k)
                for k in ("abs_err", "depth_pm1", "jaccard", "top1")},
        }
    out = RECALL_DIR / f"distill-ab-{time.strftime('%Y%m%d-%H%M%S')}.json"
    write_json_atomic(out, {"db": str(db_path), "folds": args.folds, "report": report})
    print(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0


def add_parser(subparsers):
    ab = subparsers.add_parser(
        "distill-ab", help="distilled-label agreement: EMBED_MODEL vs Voyage vectors (uploads)")
    ab.add_argument("--db")
    ab.add_argument("--model", default="voyage-4-large")
    ab.add_argument("--dim", type=int, default=1024)
    ab.add_argument("--folds", type=int, default=5)
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--confirm-upload", action="store_true")
    ab.set_defaults(func=cmd_distill_ab)

    lab = subparsers.add_parser("recall-labels", help="generate synthetic gold queries (Ollama)")
    lab.add_argument("--db")
    lab.add_argument("-n", type=int, default=200)
    lab.add_argument("--seed", type=int, default=0)
    lab.add_argument("--min-messages", type=int, default=4)
    lab.add_argument("--min-chars", type=int, default=300)
    lab.add_argument("--max-chars", type=int, default=4000)
    lab.set_defaults(func=cmd_labels)

    emb = subparsers.add_parser("recall-embed", help="embed all chunks with Voyage (uploads)")
    emb.add_argument("--db")
    emb.add_argument("--model", default="voyage-4-large")
    emb.add_argument("--dim", type=int, default=1024, help="Voyage default output dimension")
    emb.add_argument("--pool", type=int, default=0,
                     help="embed only the gold chunks + this many random distractors (0 = all)")
    emb.add_argument("--seed", type=int, default=0)
    emb.add_argument("--confirm-upload", action="store_true")
    emb.set_defaults(func=cmd_embed)

    rec = subparsers.add_parser("recall", help="recall@10 A/B over the gold queries")
    rec.add_argument("--db")
    rec.add_argument("--model", default="voyage-4-large")
    rec.set_defaults(func=cmd_recall)
    return 0
