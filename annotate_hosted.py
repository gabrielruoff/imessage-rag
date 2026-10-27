"""Hosted annotation (annotate.py's per-chunk tiers with an openrouter/ model):
one chunk per OpenRouter request, driven by asyncio on the main thread under
hosted.Limiter, with writes and commits on the loop thread.

Completeness: a chunk is done only once its annotation row is written. A
transient failure is retried until it succeeds; an unparseable reply or a
provider refusal is never written and keeps the chunk pending; a HostedStop or
an exhausted budget stops dispatch, drains what is in flight and commits; any
other exception still commits what was written, then propagates. The
run ends by recounting pending chunks from the DB and exits non-zero unless
none remain.

Shared helpers stay in annotate and resolve late (module-attribute access), the
arrangement annotate_distill and annotate_hybrid use.
"""

import asyncio
import json
import uuid

import annotate
import config
import hosted
import openrouter
import progress
from config import ANNOTATE_COMMIT_EVERY


def _parse(content):
    data = json.loads(content)
    annotate._validate_annotation(data)  # raises on a non-object reply
    return data


async def _drive(con, pending, model, think, budget, max_parallel, st, p):
    limiter = hosted.Limiter(max_parallel)  # bound to this run's loop
    async with openrouter.client() as http:
        model_id = config.hosted_model_id(model)
        prices = (await openrouter.fetch_prices(http, [model_id]))[model_id]
        ann = openrouter.Annotator(
            http, model_id, system=annotate._ANNOTATE_SYSTEM,
            schema=annotate._annotation_object_schema(), think=think,
            session_id=uuid.uuid4().hex, prices=prices, budget=budget,
            limiter=limiter, parse=_parse)
        rows = iter(pending)

        async def worker():
            for chunk_id, text in rows:
                if st["stop"]:
                    return  # taken but not sent: still pending
                try:
                    status, data = await ann.annotate_one(text)
                except hosted.HostedStop as exc:
                    st["stop"] = st["stop"] or exc.cause
                    return
                if status == "ok":
                    annotate._write_annotation(con, chunk_id, data, source="llm")
                    st["done"] += 1
                    if st["done"] % ANNOTATE_COMMIT_EVERY == 0:
                        con.commit()
                else:
                    st[status].append(chunk_id)
                    print(f"  chunk {chunk_id}: {status}, not written (retried next run)")
                p.advance()

        await asyncio.gather(*(worker() for _ in range(max_parallel)))


def annotate_chunks_hosted(con, model, *, all_chats, top_dyads, limit, dry_run,
                           think, upgrade_distilled, budget, max_parallel):
    """Annotate the pending chunks in scope with an openrouter/ `model`.
    `budget` is the --budget flag (None -> HOSTED_BUDGET_USD, required).
    Raises HostedStop (cause + pending count) on a non-transient stop and
    SystemExit when chunks remain pending at the end; returns the count
    written."""
    pending = annotate._pending_chunks(con, all_chats, top_dyads, limit,
                                       upgrade_distilled=upgrade_distilled)
    scope = "all chats" if all_chats else f"top {top_dyads} chats by message count"
    settings = f", think {think}" if think is not None else ""
    if max_parallel > 1:
        settings += f", max-parallel {max_parallel}"
    if upgrade_distilled:
        settings += ", upgrading distilled rows"
    print(f"{len(pending)} chunks pending annotation ({scope}, model {model}{settings})")
    if dry_run:
        preview = ", ".join(str(cid) for cid, _ in pending[:10])
        print(f"--dry-run: first chunk ids: {preview or '(none)'}")
        return 0
    if not pending:
        return 0
    spend = hosted.Budget(hosted.resolve_budget_usd(budget))
    if not config.OPEN_ROUTER_API_KEY.strip():
        raise SystemExit("OPEN_ROUTER_API_KEY is not set (needed for an openrouter/ model)")
    # Records the configured openrouter/ value, never a model id a response echoes.
    annotate._check_model_guard(con, model)
    hosted.upload_notice("OpenRouter", f"chunk text ({config.hosted_model_id(model)})")

    st = {"done": 0, "stop": None, "refused": [], "unparseable": []}
    try:
        try:
            with progress.track("Annotating chunks", total=len(pending), header=False) as p:
                asyncio.run(_drive(con, pending, model, think, spend, max_parallel, st, p))
        finally:  # every written (paid) annotation is kept, whatever ended the run
            con.commit()
    except KeyboardInterrupt:
        print(f"\nInterrupted -- committed {st['done']} annotations "
              f"({len(st['refused']) + len(st['unparseable'])} skipped); rerun to resume.")
        print(spend.report())
        return st["done"]
    except hosted.HostedStop as exc:  # before dispatch: price lookup
        st["stop"] = exc.cause

    # Recount from the DB, not from counters. --limit is not reapplied: the
    # question is how many of THIS run's chunks are still pending.
    ids = {cid for cid, _ in pending}
    left = sum(1 for cid, _ in annotate._pending_chunks(
        con, all_chats, top_dyads, None, upgrade_distilled=upgrade_distilled) if cid in ids)
    print(f"Annotated {st['done']} chunks")
    print(f"pending: {left}")
    print(spend.report())
    detail = ""
    if st["refused"]:
        detail += f"; refused by every provider: {', '.join(map(str, st['refused']))}"
    if st["unparseable"]:
        detail += (f"; unparseable after {openrouter.UNPARSEABLE_RETRIES} retries: "
                   f"{', '.join(map(str, st['unparseable']))}")
    if st["stop"]:
        raise hosted.HostedStop(f"{st['stop']} ({left} chunks still pending{detail})")
    if left:
        raise SystemExit(f"{left} chunks still pending{detail}")
    return st["done"]
