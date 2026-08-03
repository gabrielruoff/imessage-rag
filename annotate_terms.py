"""Dyad-term curation (annotate.py --terms): classify each chat-distinctive
term in dyad_terms as an inside joke, a nickname, ordinary vocabulary or noise.

Its OWN pipeline, not a --quality tier -- one four-way choice answered in one
word, so it has its own think default (ANNOTATE_TERMS_THINK) and its own
batching. Split out of annotate.py, which owns the CLI and re-exports
curate_terms, so `annotate.curate_terms` keeps resolving.

The shared prompts, validation and request/pool helpers stay in annotate and
are resolved late (module-attribute access), the same arrangement
annotate_distill and annotate_hybrid use, so the two modules can import each
other safely.
"""

import ollama
import progress
from config import (
    ANNOTATE_COMMIT_EVERY,
    ANNOTATE_TERMS_MAX_FAILURES,
    ANNOTATE_TIMEOUT,
)
from db import like_escape

import annotate


# ---------------------------------------------------------------------------
# Dyad-term curation (--terms)
# ---------------------------------------------------------------------------

def _term_prompt(con, chat_id, term):
    """Build the curation prompt for one term. Main-thread SQL because sqlite
    connections aren't shared across threads; that costs nothing -- the LIKE
    rides idx_messages_chat_date and measures ~2ms at 350k messages, against
    seconds for the LLM call it feeds. ESCAPE: a term is tokenizer output so
    '%'/'_' are vanishingly unlikely, but an unescaped one would silently
    match everything."""
    examples = [r[0] for r in con.execute(
        "SELECT text FROM messages "
        "WHERE chat_id = ? AND text LIKE '%' || ? || '%' ESCAPE '\\' LIMIT 3",
        (chat_id, like_escape(term)),
    )]
    return f'Term: "{term}"\n\nExample messages from this chat:\n' + (
        "\n".join(f"- {t[:200]}" for t in examples) or "(none found)"
    )


def _curate_one(prompt, model, think):
    """One single-term request -> (verdict, reason). reason is None on success
    and otherwise says why there is no verdict: an unparseable reply and a
    well-formed reply naming a bogus verdict are different failures (a bad
    prompt vs a bad model) and used to be reported identically as `None`."""
    data = ollama.chat_once_json(
        [{"role": "system", "content": annotate._TERMS_SYSTEM},
         {"role": "user", "content": prompt}],
        model=model, think=think, timeout=ANNOTATE_TIMEOUT,
        retry_nudge=annotate._TERMS_NUDGE,
    )
    if not isinstance(data, dict):
        return None, "no JSON object in reply (and the retry nudge also failed)"
    verdict = data.get("verdict")
    if verdict not in annotate.TERM_VERDICTS:
        return None, f"reply had no valid verdict (got {verdict!r})"
    return verdict, None


def _curate_batch(group, model, think):
    """One JSON-mode request classifying several terms at once; returns
    {index: verdict} for the entries the reply actually covered -- the caller
    retries the rest individually. Keyed by position, not by the term text:
    positions can't collide or need escaping."""
    user = "\n\n".join(f"### term {i}\n{prompt}"
                       for i, (_chat_id, _term, prompt) in enumerate(group))
    data = ollama.chat_once_json(
        [{"role": "system", "content": annotate._TERMS_BATCH_SYSTEM},
         {"role": "user", "content": user}],
        model=model, think=think, timeout=ANNOTATE_TIMEOUT,
        retry_nudge=annotate._TERMS_BATCH_NUDGE,
    )
    if not isinstance(data, dict):
        return {}
    return {i: data.get(str(i)) for i in range(len(group))
            if data.get(str(i)) in annotate.TERM_VERDICTS}


def _curate_group(group, model, think):
    """All LLM work for one group of (chat_id, term, prompt) items: a batched
    request for a multi-term group -- falling back to a single-term request
    for any entry it missed or mangled -- or just the single request for a
    singleton. Once annotate._GROUP_GIVEUP fallbacks in a row have failed, the rest of
    the group is abandoned unattempted (reason `annotate._ABANDONED`) rather than
    fanned out into more doomed requests. Runs on worker threads; returns
    {index: (verdict, reason)}."""
    batched = _curate_batch(group, model, think) if len(group) > 1 else {}
    out = {}
    misses = 0
    for i, (_chat_id, _term, prompt) in enumerate(group):
        if i in batched:
            out[i] = (batched[i], None)
            misses = 0
        elif misses >= annotate._GROUP_GIVEUP:
            out[i] = (None, annotate._ABANDONED)
        else:
            verdict, reason = _curate_one(prompt, model, think)
            out[i] = (verdict, reason)
            misses = 0 if verdict is not None else misses + 1
    return out


def _term_groups(con, rows, batch_size):
    """Lazily yield batches of (chat_id, term, prompt), never spanning chats:
    a batch shares one system prompt that says 'in one text-message chat', and
    mixing chats would make the examples contradict it. Lazy so the per-term
    prompt SQL stays interleaved with the LLM calls (rows arrive ordered by
    chat_id, so a plain run-length walk groups them)."""
    group = []
    for chat_id, term in rows:
        if group and group[0][0] != chat_id:
            yield group
            group = []
        group.append((chat_id, term, _term_prompt(con, chat_id, term)))
        if len(group) >= batch_size:
            yield group
            group = []
    if group:
        yield group


def _report_parked(parked, max_failures):
    """Name the terms this run parked. The header printed the pre-existing
    parked count, but a long run scrolls it away -- and a term going quiet
    forever should not be something you have to query SQL to discover."""
    if not parked:
        return
    shown = ", ".join(parked[:10]) + (" ..." if len(parked) > 10 else "")
    print(f"  parked {len(parked)} terms at {max_failures} failures "
          f"({shown}); --retry-failed to retry them")


def curate_terms(con, model, limit=None, dry_run=False, think=False, workers=1,
                 batch_size=1, max_failures=ANNOTATE_TERMS_MAX_FAILURES,
                 retry_failed=False):
    # dry_run is forwarded so a preview keeps the no-writes promise on a DB
    # from before the column (the migration is a write, and create_schema also
    # rescans reactions -- same contract as the distilled/hybrid callers).
    # has_failures is False only under --dry-run: the real path migrates, so
    # every read past the dry-run return below can rely on the column. A
    # pre-column DB has no recorded failures, which is what each degraded read
    # says: nothing parked, nothing to unpark, every unverdicted term pending.
    has_failures = annotate._ensure_column(con, "dyad_terms", "curate_failures",
                                  dry_run=dry_run)
    if retry_failed and dry_run:
        # --dry-run promises no writes, and unparking during a *preview* is
        # worse than a stray write: the preview spends the parking state, so
        # the next bare --terms run re-attempts those terms at full price --
        # the loop this pipeline exists to close.
        (n_would,) = con.execute(
            "SELECT COUNT(*) FROM dyad_terms "
            "WHERE llm_verdict IS NULL AND curate_failures > 0"
        ).fetchone() if has_failures else (0,)
        print(f"--retry-failed: would unpark {n_would} terms (--dry-run: not written)")
    elif retry_failed:
        n_reset = con.execute(
            "UPDATE dyad_terms SET curate_failures = 0 "
            "WHERE llm_verdict IS NULL AND curate_failures > 0").rowcount
        con.commit()
        print(f"--retry-failed: unparked {n_reset} terms")
    # Parked terms (curate_failures at the cap) are deliberately NOT pending:
    # a term that has already failed max_failures times -- each failure costing
    # two requests, since chat_once_json retries once -- would otherwise be
    # re-attempted at full price by every run forever.
    n_parked = con.execute(
        "SELECT COUNT(*) FROM dyad_terms "
        "WHERE llm_verdict IS NULL AND curate_failures >= ?", (max_failures,)
    ).fetchone()[0] if has_failures else 0
    # --retry-failed lifts the ceiling: after the unpark above every term sits
    # at 0 anyway, and under --dry-run (which skipped the unpark) the preview
    # must still show the scope the real run would take.
    ceiling = float("inf") if retry_failed else max_failures
    parked_filter = "AND curate_failures < ? " if has_failures else ""
    sql = ("SELECT chat_id, term FROM dyad_terms "
           f"WHERE llm_verdict IS NULL {parked_filter}"
           "ORDER BY chat_id, term")
    params = [ceiling] if has_failures else []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = con.execute(sql, params).fetchall()
    settings = f"model {model}, think {think!r}"
    if batch_size > 1:
        settings += f", batch {batch_size}"
    if workers > 1:
        settings += f", workers {workers}"
    print(f"{len(rows)} dyad terms pending curation ({settings})")
    if n_parked and not retry_failed:
        print(f"  ({n_parked} parked after {max_failures} failed attempts; "
              f"--retry-failed to retry them)")
    if dry_run:
        preview = ", ".join(f"{chat_id}:{term}" for chat_id, term in rows[:10])
        print(f"--dry-run: first terms: {preview or '(none)'}")
        return 0
    if not rows:
        return 0
    # Verdicts are model-derived and survive dyad_terms rebuilds, exactly like
    # chunk_annotations rows -- so they get the same anti-mixing guard.
    annotate._check_model_guard(con, model, "dyad_terms", "llm_verdict IS NOT NULL")

    n_done = n_skipped = 0
    parked = []
    try:
        with progress.track("Curating terms", total=len(rows), header=False) as p:
            for group, results in annotate._pooled(
                    _term_groups(con, rows, batch_size),
                    lambda g: _curate_group(g, model, think), workers):
                for i, (chat_id, term, _prompt) in enumerate(group):
                    verdict, reason = results[i]
                    if verdict is None:
                        n_skipped += 1
                        # An abandoned term never got its own request, so it
                        # is still pending, not failing -- don't charge it.
                        # Every other miss did get a dedicated request (the
                        # batch fallback), so the counter only ever records
                        # failures a single term actually earned.
                        if reason is not annotate._ABANDONED:
                            con.execute(
                                "UPDATE dyad_terms SET curate_failures = "
                                "curate_failures + 1 WHERE chat_id = ? AND term = ?",
                                (chat_id, term),
                            )
                            (n_fail,) = con.execute(
                                "SELECT curate_failures FROM dyad_terms "
                                "WHERE chat_id = ? AND term = ?",
                                (chat_id, term)).fetchone()
                            if n_fail >= max_failures:
                                parked.append(f"{chat_id}:{term}")
                        print(f"  term {chat_id}:{term}: {reason}, skipped")
                        if n_done == 0 and n_skipped >= annotate._TERMS_ABORT_AFTER:
                            # Nothing committed yet (the cadence below waits
                            # for a first success), so this really does undo
                            # every counter this run wrote.
                            con.rollback()
                            raise SystemExit(
                                f"Aborting: the first {n_skipped} terms all failed "
                                f"and none succeeded -- that is a configuration "
                                f"problem, not {n_skipped} bad terms, so no failure "
                                f"counters were recorded. Check that model {model!r} "
                                f"is pulled and answers JSON; if it is gpt-oss, "
                                f"think={think!r} yields empty replies -- set "
                                f"ANNOTATE_TERMS_THINK=low."
                            )
                    else:
                        con.execute(
                            "UPDATE dyad_terms SET llm_verdict = ? "
                            "WHERE chat_id = ? AND term = ?",
                            (verdict, chat_id, term),
                        )
                        n_done += 1
                    p.advance()
                    # Both branches write, so the cadence counts both -- but
                    # not before a first success, so the abort above can roll
                    # back cleanly whatever ANNOTATE_COMMIT_EVERY is set to.
                    if n_done and (n_done + n_skipped) % ANNOTATE_COMMIT_EVERY == 0:
                        con.commit()
    except KeyboardInterrupt:
        con.commit()
        print(f"\nInterrupted -- committed {n_done} verdicts "
              f"({n_skipped} skipped); rerun to resume.")
        _report_parked(parked, max_failures)
        return n_done
    con.commit()
    print(f"Curated {n_done} terms ({n_skipped} skipped)")
    _report_parked(parked, max_failures)
    return n_done
