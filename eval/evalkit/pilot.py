"""Hosted-annotation pilot: annotate a seeded sample of LLM-labeled chunks with
OpenRouter models using annotate.py's own prompt, schema and validator, and
report pairwise agreement (each hosted model vs the reference model and vs the
stored local LLM labels).

Requests are production's: openrouter.request_body (cached system block, strict
schema, OPENROUTER_MAX_TOKENS), live prices from openrouter.fetch_prices (an
unlisted or unpriced model refuses the run), retries per hosted.classify /
retry_delay, and errors carry only hosted.error_message, never a raw body.

Spend is HARD-bounded per invocation (--resume reuses results, not budget):
before each request a worst-case cost is reserved exactly as
openrouter.Annotator._post does -- UTF-8 bytes of the whole request body,
schema included (byte-level BPE never yields more tokens than bytes) x the live
input price, plus max_tokens x the live output price -- and the request is
refused if spent + reservations would exceed --budget. Settled cost is
max(OpenRouter-reported cost, tokens x live price); a 4xx without usage is 0,
anything else without usage the worst case.
"""

import asyncio
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import RESULTS_DIR
from .manifest import resolve_db_path, write_json_atomic

PILOT_DIR = RESULTS_DIR / "pilot"
# Attempts per chunk on transient failures (production retries without limit;
# a pilot records the chunk as an error instead).
RETRIES = 4


async def _fetch_prices(models):
    import openrouter

    async with openrouter.client() as http:
        return await openrouter.fetch_prices(http, models)


def _post(body, prices, ctx):
    """One request under the shared budget -> (response | transport exception
    | None when the budget refuses, parsed JSON or None, settled cost)."""
    import config
    import hosted
    import openrouter

    in_price, out_price = prices
    worst = (len(json.dumps(body).encode()) * in_price
             + config.OPENROUTER_MAX_TOKENS * out_price)
    if not ctx["budget"].reserve(worst):
        return None, None, 0.0
    actual = worst
    try:
        resp = ctx["http"].post("/chat/completions", json=body)
        data = openrouter._json(resp)
        usage = data.get("usage") if isinstance(data, dict) else None
        if usage:
            actual = max(float(usage.get("cost") or 0.0),
                         (usage.get("prompt_tokens") or 0) * in_price
                         + (usage.get("completion_tokens") or 0) * out_price)
        elif resp.status_code >= 400:  # unbilled, as openrouter.Annotator._post
            actual = 0.0
        return resp, data, actual
    except hosted.REQUEST_ERRORS as exc:
        return exc, None, actual
    finally:
        ctx["budget"].settle(worst, actual)


def _annotate(model, chunk_id, text, ctx):
    import annotate
    import httpx

    import hosted
    import openrouter

    row = {"chunk_id": chunk_id, "model": model}
    cost, attempt = 0.0, 0
    while True:
        think = ctx["think"].get(model, False)
        body = openrouter.request_body(model, annotate._ANNOTATE_SYSTEM, ctx["schema"], text,
                                       think, ctx["session_id"])
        resp, data, spent = _post(body, ctx["prices"][model], ctx)
        cost += spent
        if resp is None:
            return {**row, "skipped": "budget", "cost": cost}
        kind = hosted.classify(resp)
        if kind == "ok":
            break
        response = resp if isinstance(resp, httpx.Response) else None
        if kind == "retry" and attempt + 1 < RETRIES:
            time.sleep(hosted.retry_delay(attempt, response))
            attempt += 1
            continue
        status, msg = openrouter._error(resp, data)
        if "reasoning is mandatory" in msg.lower() and think != "low":
            # e.g. Sonnet 5.5 rejects disabled thinking: lowest effort instead
            ctx["think"][model] = "low"
            continue
        return {**row, "error": f"{status}: {msg}", "cost": cost}
    usage = data.get("usage") or {}
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        content = ""
    try:
        val, emos, dep, events = annotate._validate_annotation(json.loads(content))
        parsed = {"valence": val, "emotions": emos, "depth": dep, "events": events}
    except (json.JSONDecodeError, AttributeError):
        parsed = None
    return {
        **row, "annotation": parsed,
        "raw": None if parsed else content[:500], "cost": cost,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"),
    }


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
    import annotate
    import config
    import db
    import httpx
    import openrouter
    from hosted import Budget, HostedStop, resolve_budget_usd

    if not config.OPEN_ROUTER_API_KEY.strip():
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
    try:
        prices = asyncio.run(_fetch_prices(models))
    except HostedStop as e:
        print(e.cause, file=sys.stderr)
        return 2
    http = httpx.Client(
        base_url=openrouter.BASE_URL, timeout=config.HOSTED_TIMEOUT,
        headers={"Authorization": f"Bearer {config.OPEN_ROUTER_API_KEY.strip()}"})
    ctx = {"prices": prices, "budget": Budget(resolve_budget_usd(args.budget)), "http": http,
           "schema": annotate._annotation_object_schema(), "think": {},
           "session_id": f"pilot-{uuid.uuid4().hex}"}
    print(f"{len(sample)} chunks x {len(models)} models, hard budget ${args.budget:.2f}")

    done = []
    if args.resume:
        done = [r for r in json.load(open(args.resume))["results"]
                if r.get("annotation") and r["chunk_id"] in local]
    have = {(r["model"], r["chunk_id"]) for r in done}
    jobs = [(m, r[0], r[5]) for m in models for r in sample if (m, r[0]) not in have]
    print(f"{len(have)} reused from --resume, {len(jobs)} to run")
    with http, ThreadPoolExecutor(args.workers) as pool_ex:
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
        "per_model": stats, "reasoning_used": {
            m: openrouter.reasoning_for(t) for m, t in ctx["think"].items()},
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
