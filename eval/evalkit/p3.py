"""P3: generator-output evals (plan §4.7) -- wrapped.py narrative
faithfulness, annotate.py and topics.py samplers. `wrapped` and `--judge`
take the Ollama lock; the check/sample paths are parallel-safe file I/O."""

import json
import os
import random
import re
import sqlite3
import subprocess
import sys

from . import EVAL_DIR, RESULTS_DIR
from .manifest import load_manifest, resolve_db_path, write_json_atomic
from .runner import acquire_ollama_lock, release_ollama_lock

NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _run_dir(args):
    run_dir = RESULTS_DIR / args.run
    if not run_dir.is_dir():
        print(f"no such run dir: {run_dir}", file=sys.stderr)
        raise SystemExit(2)
    return run_dir


# --- P3.1 wrapped ------------------------------------------------------------


def cmd_wrapped(args):
    run_dir = _run_dir(args)
    resolve_db_path(args.db)
    year = load_manifest()["roster"]["YEAR"]
    acquire_ollama_lock(f"p3-wrapped-{args.run}")
    try:
        for extra, out_name in (
            ([], "P3.1.wrapped_llm.md"),
            (["--no-llm"], "P3.1.wrapped_stats.md"),
        ):
            cmd = [
                sys.executable, "wrapped.py", "--year", str(year), "--yes",
                "--out", str(run_dir / out_name), *extra,
            ]
            print(f"$ {' '.join(cmd)}")
            proc = subprocess.run(cmd, cwd=EVAL_DIR.parent, env=os.environ.copy())
            if proc.returncode != 0:
                print(f"wrapped.py failed (exit {proc.returncode})", file=sys.stderr)
                return 1
    finally:
        release_ollama_lock()
    print(f"wrote {run_dir}/P3.1.wrapped_llm.md and P3.1.wrapped_stats.md")
    return 0


def _numeric_values(text):
    values = set()
    for token in NUMBER_TOKEN_RE.findall(text):
        try:
            values.add(float(token.rstrip("%").replace(",", "")))
        except ValueError:
            pass
    return values


def check_wrapped_numbers(llm_text, stats_text):
    """Every number/date token in the LLM narrative must appear in the
    stats-only version (rounding-tolerant; % also tried x100 / /100).
    Zero tolerance: any invented number fails the feature."""
    stats_values = _numeric_values(stats_text)

    def supported(value):
        for v in stats_values:
            if (round(v, 0) == round(value, 0) or round(v, 1) == round(value, 1)
                    or round(v * 100, 0) == round(value, 0)
                    or round(v, 0) == round(value * 100, 0)):
                return True
        return False

    invented, checked = [], 0
    for token in ISO_DATE_RE.findall(llm_text):
        checked += 1
        if token not in stats_text:
            invented.append(token)
    for token in NUMBER_TOKEN_RE.findall(ISO_DATE_RE.sub(" ", llm_text)):
        checked += 1
        if token.rstrip("%") in stats_text:
            continue
        try:
            value = float(token.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if not supported(value) and token not in invented:
            invented.append(token)
    return {"numbers_checked": checked, "invented": invented}


def cmd_check_wrapped(args):
    run_dir = _run_dir(args)
    llm_path = run_dir / "P3.1.wrapped_llm.md"
    stats_path = run_dir / "P3.1.wrapped_stats.md"
    if not llm_path.exists() or not stats_path.exists():
        print("prereq missing: run `p3 wrapped` first", file=sys.stderr)
        return 2
    result = check_wrapped_numbers(llm_path.read_text(), stats_path.read_text())
    result["auto_fail"] = bool(result["invented"])
    result["manual"] = [
        "tone (descriptive, never evaluative)",
        "verbatim excerpts",
        "silent-exclusions honored",
        "markdown sanity",
    ]
    write_json_atomic(run_dir / "P3.1.checks.json", result)
    print(json.dumps(result, indent=1))
    return 1 if result["auto_fail"] else 0


# --- P3.2 annotate sampler -----------------------------------------------------


_STRATA = (
    ("most_negative", "ORDER BY a.valence ASC"),
    ("most_positive", "ORDER BY a.valence DESC"),
    ("middle", "ORDER BY ABS(a.valence) ASC"),
    ("highest_depth", "ORDER BY a.depth DESC"),
    ("event_bearing",
     "WHERE a.events_json IS NOT NULL AND a.events_json NOT IN ('[]', '') "
     "ORDER BY a.chunk_id"),
)


def cmd_annotate_sample(args):
    run_dir = _run_dir(args)
    db_path = resolve_db_path(args.db)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    n_annotated = con.execute("SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]
    if not n_annotated:
        print("prereq missing: chunk_annotations is empty (run annotate.py first)",
              file=sys.stderr)
        return 2
    n_chunks = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE id IN (SELECT chunk_id FROM chunk_vectors)"
    ).fetchone()[0]
    samples, seen = [], set()
    for stratum, tail in _STRATA:
        rows = con.execute(
            "SELECT a.chunk_id, a.valence, a.emotions_json, a.depth, a.events_json, c.text "
            "FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id "
            f"{tail} LIMIT 25"
        ).fetchall()
        kept = 0
        for row in rows:
            if row["chunk_id"] in seen or kept >= 5:
                continue
            seen.add(row["chunk_id"])
            kept += 1
            samples.append({
                "stratum": stratum,
                "chunk_id": row["chunk_id"],
                "chunk_text": row["text"],
                "annotation": {
                    "valence": row["valence"],
                    "emotions": json.loads(row["emotions_json"] or "[]"),
                    "depth": row["depth"],
                    "events": json.loads(row["events_json"] or "[]"),
                },
                "judgment": {"valence_sign_agree": None, "emotions_agree": None,
                             "depth_rating": None, "notes": None},
            })
    con.close()

    if args.judge:
        _judge_annotations(samples, args.run)

    doc = {
        "n_annotated": n_annotated,
        "n_embeddable_chunks": n_chunks,
        "note": "skip rate ~ share of embeddable chunks in annotated dyads without a row; "
                "annotate.py skips chunks that failed JSON twice",
        "samples": samples,
        "success_criteria": "valence sign agreement >= 80%; emotions in EMOTION_LABELS; "
                            "event message_ids contain the excerpt; depth roughly monotone",
    }
    write_json_atomic(run_dir / "P3.2.samples.json", doc)
    print(f"wrote {run_dir}/P3.2.samples.json ({len(samples)} samples, "
          f"judged={bool(args.judge)})")
    return 0


def _judge_annotations(samples, run_id):
    import ollama

    acquire_ollama_lock(f"p3-annotate-judge-{run_id}")
    try:
        for sample in samples:
            prompt = (
                "You are auditing an automatic conversation annotation.\n\n"
                f"Conversation chunk:\n{sample['chunk_text']}\n\n"
                f"Stored annotation: {json.dumps(sample['annotation'])}\n\n"
                "Reply with JSON only: {\"valence_sign_agree\": true|false, "
                "\"emotions_agree\": true|false, \"depth_rating\": 0-3, "
                "\"notes\": \"one sentence\"}"
            )
            try:
                msg = ollama.chat_once([{"role": "user", "content": prompt}], format="json")
                sample["judgment"] = json.loads(msg.get("content") or "{}")
            except Exception as e:
                sample["judgment"] = {"error": str(e)}
    finally:
        release_ollama_lock()


# --- P3.3 topics sampler --------------------------------------------------------

# Chunks per cluster shown to the label-quality judge. Deliberately NOT
# config.TOPIC_LABEL_SAMPLES: that value is tuned for the label pass, whose
# samples are centroid-ranked (and so need to be many, to escape the short
# filler chunks nearest the centroid). These are picked ORDER BY c.id, so more
# of them buys no representativeness -- it only grows the judge prompt (each
# chunk is truncated to 800 chars and all are joined into one user message),
# which Ollama would silently drop past the context window rather than error.
_JUDGE_SAMPLES = 5


def cmd_topics_sample(args):
    run_dir = _run_dir(args)
    db_path = resolve_db_path(args.db)

    from db import get_state

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    clusters = con.execute("SELECT cluster_id, label, n_chunks FROM topics").fetchall()
    if not clusters:
        con.close()
        print("prereq missing: topics is empty (run topics.py first)", file=sys.stderr)
        return 2
    total = sum(r["n_chunks"] for r in clusters)
    degenerate = [
        {"cluster_id": r["cluster_id"], "label": r["label"],
         "share": round(r["n_chunks"] / total, 3)}
        for r in clusters
        if r["n_chunks"] / total > 0.4
    ]
    rng = random.Random(20260716)
    picked = rng.sample(list(clusters), min(10, len(clusters)))
    samples = []
    for cluster in picked:
        # Random members stand in for centroid-nearest (centroids aren't stored).
        chunks = con.execute(
            "SELECT c.id, c.text FROM chunk_topics ct JOIN chunks c ON c.id = ct.chunk_id "
            "WHERE ct.cluster_id = ? ORDER BY c.id LIMIT ?",
            (cluster["cluster_id"], _JUDGE_SAMPLES),
        ).fetchall()
        samples.append({
            "cluster_id": cluster["cluster_id"],
            "label": cluster["label"],
            "n_chunks": cluster["n_chunks"],
            "chunks": [{"chunk_id": r["id"], "text": r["text"][:800]} for r in chunks],
            "judgment": {"label_fits": None, "notes": None},
        })
    state_k = get_state(con, "topic_model_k")
    con.close()

    if args.judge:
        _judge_topics(samples, args.run)

    doc = {
        "topic_model_k": state_k,
        "n_clusters": len(clusters),
        "degenerate_clusters_over_40pct": degenerate,
        "samples": samples,
        "success_criteria": ">= 8/10 labels are 2-5 words and describe the sampled chunks; "
                            "no cluster > 40% of all chunks",
    }
    write_json_atomic(run_dir / "P3.3.samples.json", doc)
    print(f"wrote {run_dir}/P3.3.samples.json ({len(samples)} clusters, "
          f"degenerate={len(degenerate)}, judged={bool(args.judge)})")
    return 0


def _judge_topics(samples, run_id):
    import ollama

    acquire_ollama_lock(f"p3-topics-judge-{run_id}")
    try:
        for sample in samples:
            excerpts = "\n---\n".join(c["text"] for c in sample["chunks"])
            prompt = (
                f"Topic label: {sample['label']!r}\n\n"
                f"Sampled conversation chunks from this cluster:\n{excerpts}\n\n"
                "Does the label describe these chunks (and is it 2-5 words)? "
                "Reply with JSON only: {\"label_fits\": true|false, "
                "\"notes\": \"one sentence\"}"
            )
            try:
                msg = ollama.chat_once([{"role": "user", "content": prompt}], format="json")
                sample["judgment"] = json.loads(msg.get("content") or "{}")
            except Exception as e:
                sample["judgment"] = {"error": str(e)}
    finally:
        release_ollama_lock()


def add_parser(subparsers):
    p = subparsers.add_parser("p3", help="P3: wrapped/annotate/topics generator evals")
    sub = p.add_subparsers(dest="p3_cmd", required=True)

    w = sub.add_parser("wrapped", help="generate wrapped LLM + stats reports (Ollama lane)")
    w.add_argument("--run", required=True)
    w.add_argument("--db")
    w.set_defaults(func=cmd_wrapped)

    cw = sub.add_parser("check-wrapped", help="number-faithfulness check (parallel-safe)")
    cw.add_argument("--run", required=True)
    cw.set_defaults(func=cmd_check_wrapped)

    ann = sub.add_parser("annotate-sample", help="stratified chunk_annotations sample")
    ann.add_argument("--run", required=True)
    ann.add_argument("--db")
    ann.add_argument("--judge", action="store_true", help="LLM-judge the samples (Ollama lane)")
    ann.set_defaults(func=cmd_annotate_sample)

    top = sub.add_parser("topics-sample", help="topic-label quality sample")
    top.add_argument("--run", required=True)
    top.add_argument("--db")
    top.add_argument("--judge", action="store_true", help="LLM-judge the samples (Ollama lane)")
    top.set_defaults(func=cmd_topics_sample)
