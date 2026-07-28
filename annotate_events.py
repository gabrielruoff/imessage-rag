"""Event-candidate prefilter for the hybrid annotation tier (annotate.py
--quality hybrid, Phase 3a): which chunks are worth spending an LLM call on
because they probably contain a good_news/gratitude event.

Events are the one label kNN distillation structurally cannot produce -- a
source='distilled' row always has events_json='[]' -- so the hybrid tier's
whole reason to buy LLM time is to fill them in, and its budget covers only a
few hundred chunks of a 25k-chunk corpus. Asking the LLM which chunks to look
at would spend the exact resource being rationed, so the shortlist is drawn
deterministically from the EVENT_CANDIDATE_PATTERNS lexicon in
function_words.py (which documents the precision/recall trade behind each
substring). A miss here costs recall, never correctness: the LLM still decides
what is an event, and the chunks this skips are already kNN-labeled for
valence/emotions/depth.

Measured on the 350k-message / 25k-chunk corpus (2026-07-27): one SQL pass,
0.71 s scoped to the top 20 dyads and 0.76 s corpus-wide, selecting 20.5% /
19.0% of in-scope chunks at 83.3% precision against that corpus's existing LLM
event labels (1.72x their 48.4% base rate -- see function_words for the full
scoring). Matching the same rows in Python instead measured 0.6 s but
materializes every message body, so SQL wins on memory at equal speed; 38
separate per-pattern scans measured 4.9 s, which is why every pattern goes
into one OR'd predicate.
"""

from function_words import EVENT_CANDIDATE_PATTERNS

# Flattened once: the frozen contract returns a single kind-agnostic list, and
# one scan with all patterns OR'd together is ~2x cheaper than a scan per kind.
# (A future by-kind variant -- so a caller could interleave gratitude and
# good-news upgrades instead of taking a chunk-id-ordered prefix, which today
# is 63% gratitude-only, 22% good-news-only, 15% both -- is a 3-line addition:
# _candidate_ids already takes the pattern tuple as an argument.)
_ALL_PATTERNS = tuple(p for pats in EVENT_CANDIDATE_PATTERNS.values() for p in pats)


def _like_escape(value):
    r"""Escape LIKE wildcards for use with ESCAPE '\'. A deliberate 3-line
    duplicate of annotate._like_escape rather than an import: that is another
    module's private helper, and importing annotate here would create an
    import cycle for no gain. EVENT_CANDIDATE_PATTERNS is wildcard-free today
    (documented there), so this is purely insurance against a later pattern
    containing a literal % or _."""
    return (value.replace("\\", "\\\\")
                 .replace("%", "\\%")
                 .replace("_", "\\_"))


def _candidate_ids(con, patterns, all_chats, top_dyads):
    """Distinct chunk ids in scope with at least one message matching
    `patterns`, ascending. See event_candidate_chunk_ids for the contract."""
    # No LOWER() on either side: SQLite's LIKE already folds case, and only for
    # ASCII -- which is exactly right for these patterns (all ASCII) and is
    # why nobody should "fix" this by wrapping the column, which would cost a
    # per-row function call for zero behavior change. Messages with NULL text
    # drop out on their own (NULL LIKE x is NULL, never true).
    like = " OR ".join(["m.text LIKE ? ESCAPE '\\'"] * len(patterns))
    params = [f"%{_like_escape(p)}%" for p in patterns]
    # LEFT JOIN, not NOT EXISTS: a chunk is a candidate when it has no
    # annotation at all or only a distilled one. Rows already source='llm'
    # have their events and must never be paid for twice.
    sql = f"""
        SELECT DISTINCT m.chunk_id
        FROM messages m
        JOIN chunks c ON c.id = m.chunk_id
        LEFT JOIN chunk_annotations a ON a.chunk_id = m.chunk_id
        WHERE m.chunk_id IS NOT NULL
          AND ({like})
          AND (a.chunk_id IS NULL OR a.source = 'distilled')
    """
    if not all_chats:
        # Same subquery shape as annotate._pending_chunks, so hybrid's Phase 3
        # scope is bit-identical to what the LLM tiers annotate. top_dyads is
        # passed through verbatim (a NULL LIMIT means unlimited), matching
        # _pending_chunks rather than second-guessing the caller.
        sql += """  AND c.chat_id IN (
            SELECT chat_id FROM messages WHERE chat_id IS NOT NULL
            GROUP BY chat_id ORDER BY COUNT(*) DESC LIMIT ?)
        """
        params.append(top_dyads)
    sql += " ORDER BY m.chunk_id"
    return [row[0] for row in con.execute(sql, params)]


def event_candidate_chunk_ids(con, all_chats, top_dyads):
    """Returns an ordered list[int] of chunk ids in scope whose messages match
    EVENT_CANDIDATE_PATTERNS and whose annotation is absent or
    source='distilled'. Read-only. Order: chunk id ascending."""
    return _candidate_ids(con, _ALL_PATTERNS, all_chats, top_dyads)
