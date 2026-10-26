"""Hosted-annotation pilot: annotate a seeded sample of LLM-labeled chunks with
OpenRouter models using annotate.py's own prompt, schema and validator, and
report pairwise agreement (each hosted model vs the reference model and vs the
stored local LLM labels).

Spend is HARD-bounded per invocation (--resume reuses results, not budget):
before each request a worst-case cost is reserved --
UTF-8 bytes of the prompt (byte-level BPE never yields more tokens than bytes)
x the live input price, plus max_tokens x the live output price -- and the
request is refused if spent + reservations would exceed --budget. Settled cost
is max(OpenRouter-reported cost, tokens x live price).
"""

import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import RESULTS_DIR
from .manifest import resolve_db_path, write_json_atomic

OR_URL = "https://openrouter.ai/api/v1"
PILOT_DIR = RESULTS_DIR / "pilot"
MAX_TOKENS = 1024


class Budget:
    def __init__(self, limit):
        self.limit, self.spent, self.reserved = limit, 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


def _prices(models):
    """{model: (usd per input token, usd per output token)} from the live list."""
    with urllib.request.urlopen(f"{OR_URL}/models", timeout=60) as resp:
        listing = {m["id"]: m["pricing"] for m in json.loads(resp.read())["data"]}
    missing = [m for m in models if m not in listing]
    if missing:
        raise SystemExit(f"not on OpenRouter: {', '.join(missing)}")
    return {m: (float(listing[m]["prompt"]), float(listing[m]["completion"])) for m in models}


def _call(model, system, text, schema, api_key, reasoning):
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "annotation", "strict": True, "schema": schema}},
        "reasoning": reasoning,
        "provider": {"require_parameters": True},  # never silently drop the schema
        "usage": {"include": True},
    }
    req = urllib.request.Request(
        f"{OR_URL}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            code = e.code
            if code != 429 and code < 500:
                return {"error": f"{code}: {e.read().decode(errors='replace')[:300]}",
                        "unbilled": True}
        except (urllib.error.URLError, TimeoutError):
            code = None
        time.sleep(2 ** (attempt + 1))
    # only a run of 429s is known unbilled; a 5xx or timeout may have been charged
    return {"error": f"retries exhausted (last: {code or 'timeout'})", "unbilled": code == 429}


def _annotate(model, chunk_id, text, ctx):
    import annotate

    in_price, out_price = ctx["prices"][model]
    system = annotate._ANNOTATE_SYSTEM
    worst = (len(system.encode()) + len(text.encode()) + 64) * in_price + MAX_TOKENS * out_price
    if not ctx["budget"].reserve(worst):
        return {"chunk_id": chunk_id, "model": model, "skipped": "budget"}
    actual = 0.0
    try:
        reasoning = ctx["reasoning"].get(model, {"enabled": False})
        reply = _call(model, system, text, ctx["schema"], ctx["api_key"], reasoning)
        if "Reasoning is mandatory" in reply.get("error", ""):
            # e.g. Sonnet 5.5 rejects disabled thinking: lowest effort instead
            ctx["reasoning"][model] = reasoning = {"effort": "low"}
            reply = _call(model, system, text, ctx["schema"], ctx["api_key"], reasoning)
        usage = reply.get("usage") or {}
        actual = max(float(usage.get("cost") or 0.0),
                     usage.get("prompt_tokens", 0) * in_price
                     + usage.get("completion_tokens", 0) * out_price)
        if "error" in reply:
            # 4xx/429 are never billed; anything else without usage counts as worst case
            actual = actual or (0.0 if reply.get("unbilled") else worst)
            return {"chunk_id": chunk_id, "model": model, "error": reply["error"],
                    "cost": actual}
        content = reply["choices"][0]["message"].get("content") or ""
        try:
            val, emos, dep, events = annotate._validate_annotation(json.loads(content))
            parsed = {"valence": val, "emotions": emos, "depth": dep, "events": events}
        except (json.JSONDecodeError, AttributeError):
            parsed = None
        return {
            "chunk_id": chunk_id, "model": model, "annotation": parsed,
            "raw": None if parsed else content[:500], "cost": actual,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"),
        }
    finally:
        ctx["budget"].settle(worst, actual)


def _compare(a, b):
    """Agreement of annotation dicts b vs a over the chunks both have."""
    common = [c for c in a if c in b]
    if not common:
        return {"n": 0}
    n = len(common)

    def jac(x, y):
        x, y = set(x), set(y)
        return len(x & y) / len(x | y) if x | y else 1.0

    return {
        "n": n,
        "valence_mae": round(sum(abs(a[c]["valence"] - b[c]["valence"]) for c in common) / n, 4),
        "valence_sign_agree": round(sum(
            (a[c]["valence"] > 0.1) == (b[c]["valence"] > 0.1)
            and (a[c]["valence"] < -0.1) == (b[c]["valence"] < -0.1) for c in common) / n, 4),
        "depth_pm1": round(sum(abs(a[c]["depth"] - b[c]["depth"]) <= 1 for c in common) / n, 4),
        "depth_exact": round(sum(a[c]["depth"] == b[c]["depth"] for c in common) / n, 4),
        "emotion_jaccard": round(sum(jac(a[c]["emotions"], b[c]["emotions"])
                                     for c in common) / n, 4),
        "has_event_agree": round(sum(bool(a[c]["events"]) == bool(b[c]["events"])
                                     for c in common) / n, 4),
        "event_kinds_jaccard": round(sum(
            jac({e["kind"] for e in a[c]["events"]}, {e["kind"] for e in b[c]["events"]})
            for c in common) / n, 4),
    }


def cmd_pilot(args):
    db_path = resolve_db_path(args.db)
    import os

    import annotate
    import config  # noqa: F401 -- loads .env
    import db

    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("OPEN_ROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    models = [args.model, args.reference]
    con = db.connect(str(db_path), rows=False)
    pool = con.execute(
        "SELECT a.chunk_id, a.valence, a.emotions_json, a.depth, a.events_json, c.text, c.chat_id "
        "FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id "
        "WHERE a.source = 'llm' ORDER BY a.chunk_id").fetchall()
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    local = {r[0]: {"valence": r[1], "emotions": json.loads(r[2]), "depth": r[3],
                    "events": json.loads(r[4])} for r in sample}
    ctx = {"prices": _prices(models), "budget": Budget(args.budget), "api_key": api_key,
           "schema": annotate._annotation_object_schema(), "reasoning": {}}
    print(f"{len(sample)} chunks x {len(models)} models, hard budget ${args.budget:.2f}")

    done = []
    if args.resume:
        done = [r for r in json.load(open(args.resume))["results"]
                if r.get("annotation") and r["chunk_id"] in local]
    have = {(r["model"], r["chunk_id"]) for r in done}
    jobs = [(m, r[0], r[5]) for m in models for r in sample if (m, r[0]) not in have]
    print(f"{len(have)} reused from --resume, {len(jobs)} to run")
    with ThreadPoolExecutor(args.workers) as pool_ex:
        results = done + list(pool_ex.map(lambda j: _annotate(*j, ctx), jobs))

    by_model = {m: {r["chunk_id"]: r["annotation"] for r in results
                    if r["model"] == m and r.get("annotation")} for m in models}
    local_name = f"local:{db.get_state(con, 'annotate_model')}"
    stats = {m: {
        "ok": len(by_model[m]),
        "unparseable": sum(1 for r in results if r["model"] == m and "annotation" in r
                           and r["annotation"] is None),
        "errors": sum(1 for r in results if r["model"] == m and "error" in r),
        "skipped_budget": sum(1 for r in results if r["model"] == m and "skipped" in r),
        "cost_usd": round(sum(r.get("cost", 0.0) for r in results if r["model"] == m), 4),
        "mean_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in results
                                            if r["model"] == m) / max(1, len(by_model[m])), 1),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in results
                                if r["model"] == m),
    } for m in models}
    report = {
        "db": str(db_path), "n": len(sample),
        "spent_usd": round(ctx["budget"].spent, 4), "budget_usd": args.budget,
        "per_model": stats, "reasoning_used": ctx["reasoning"],
        "agreement": {
            f"{args.model} vs {args.reference}": _compare(by_model[args.reference],
                                                          by_model[args.model]),
            f"{args.model} vs {local_name}": _compare(local, by_model[args.model]),
            f"{args.reference} vs {local_name}": _compare(local, by_model[args.reference]),
        },
    }
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PILOT_DIR / f"pilot-{time.strftime('%Y%m%d-%H%M%S')}.json"
    write_json_atomic(out, {"report": report, "results": results})
    print(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0


def add_parser(subparsers):
    p = subparsers.add_parser(
        "annotate-pilot", help="hosted annotation pilot via OpenRouter (uploads chunk text)")
    p.add_argument("--db")
    p.add_argument("-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--reference", default="anthropic/claude-sonnet-5.5")
    p.add_argument("--budget", type=float, default=2.0, help="hard USD cap for the whole run")
    p.add_argument("--workers", type=int, default=2, help="8 hit OpenRouter 429s on Sonnet")
    p.add_argument("--resume", help="earlier pilot JSON: reuse its successful annotations")
    p.set_defaults(func=cmd_pilot)
    return 0
