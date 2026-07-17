"""
Derived language artifacts of the ingest pipeline: the per-person-per-month
function-word lexicon (person_period_lexicon) and each chat's distinctive
vocabulary (dyad_terms).

Everything here operates only on imessage_analysis.db (the target DB). Both
tables are full rebuilds -- like chunks, they are derived artifacts and the
append-only rule's carve-out applies -- except that dyad_terms preserves
annotate.py's curation verdicts across rebuilds (an llm_verdict is expensive
LLM output; recomputed stats are not). ingest.py runs both stages after
chunking and re-exports them, so `ingest.build_dyad_terms` etc. keep working.
"""

import collections
import json
import sqlite3

import progress
from config import DYAD_TERM_MIN_LIFT, DYAD_TERM_MIN_USES, DYAD_TERM_TOP_K
from function_words import (
    EMOJI_RE,
    FUNCTION_WORDS,
    LAUGH_RE,
    STOPWORDS,
    WE_WORDS,
    tokenize,
)
from query_tools._shared import _REAL_MESSAGE_CLAUSES

# Both pipelines read the same population: real conversation messages with a
# date and non-empty text, in a chat. (Attachment-only rows carry no words;
# emote/system rows must never contribute vocabulary -- see _shared.py.)
_QUALIFYING_WHERE = " AND ".join((
    "m.chat_id IS NOT NULL",
    "m.date IS NOT NULL",
    "m.text IS NOT NULL",
    "m.text != ''",
    *_REAL_MESSAGE_CLAUSES,
))

# Category order is FUNCTION_WORDS declaration order; the fw_*_rate columns
# below are generated from it so the two can never disagree.
_FW_CATEGORIES = tuple(FUNCTION_WORDS)
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in _FW_CATEGORIES)

# How many of the corpus's most frequent tokens join STOPWORDS when filtering
# dyad-term candidates ("yeah", proper names everyone uses, ...). A module
# constant, not config: it's part of the candidate definition, and tests on
# tiny fixtures patch it down (in a small corpus every token is "top 200").
_CORPUS_TOP_N = 200


def build_person_period_lexicon(tgt: sqlite3.Connection) -> int:
    """Rebuild person_period_lexicon from scratch: one tokenizing pass over
    every qualifying message, accumulated per (chat_id, handle_id-or-None,
    month) -- handle_id is NULL for my side (is_from_me rows carry the
    counterparty's handle_id or 0, never mine). Each row carries message/word
    counts, mean message length, the 9 function-word category rates
    (category hits / words, 0.0 for wordless months), we/question/laugh
    message counts, and an {emoji: count} JSON dict.

    Full rebuild by design (DELETE + plain INSERTs, a few seconds at 350k
    messages): incremental upkeep would need read-modify-write on every
    affected month row for no measurable win."""
    tgt.execute("DELETE FROM person_period_lexicon")

    # Counted up front purely so the pass can report a percentage -- the loop
    # below streams its cursor and would otherwise have no denominator. Same
    # WHERE as the stream, so it stays on the same index.
    n_total = tgt.execute(
        f"SELECT COUNT(*) FROM messages m WHERE {_QUALIFYING_WHERE}"
    ).fetchone()[0]

    stats = {}
    with progress.track("Building lexicon", total=n_total) as p:
        for chat_id, handle_id, is_from_me, text, date in tgt.execute(
            f"SELECT m.chat_id, m.handle_id, m.is_from_me, m.text, m.date "
            f"FROM messages m WHERE {_QUALIFYING_WHERE}"
        ):
            key = (chat_id, None if is_from_me else handle_id, date[:7])
            s = stats.get(key)
            if s is None:
                s = stats[key] = {
                    "n_messages": 0, "n_words": 0, "chars": 0,
                    "fw": [0] * len(_FW_CATEGORIES),
                    "we": 0, "question": 0, "laugh": 0,
                    "emoji": collections.Counter(),
                }
            tokens = tokenize(text)
            s["n_messages"] += 1
            s["n_words"] += len(tokens)
            s["chars"] += len(text)
            fw = s["fw"]
            for tok in tokens:
                for i, cat in enumerate(_FW_CATEGORIES):
                    if tok in FUNCTION_WORDS[cat]:
                        fw[i] += 1
            if any(tok in WE_WORDS for tok in tokens):
                s["we"] += 1
            if "?" in text:
                s["question"] += 1
            if LAUGH_RE.search(text):
                s["laugh"] += 1
            s["emoji"].update(EMOJI_RE.findall(text))
            p.advance()

    rows = []
    for (chat_id, handle_id, month), s in stats.items():
        n_words = s["n_words"]
        rates = [hits / n_words if n_words else 0.0 for hits in s["fw"]]
        rows.append((
            chat_id, handle_id, month,
            s["n_messages"], n_words, s["chars"] / s["n_messages"],
            *rates,
            s["we"], s["question"], s["laugh"],
            json.dumps(dict(s["emoji"]), ensure_ascii=False),
        ))
    tgt.executemany(
        "INSERT INTO person_period_lexicon("
        "chat_id, handle_id, month, n_messages, n_words, avg_chars, "
        + ", ".join(_FW_COLUMNS)
        + ", we_count, question_count, laugh_count, emoji_json) "
        "VALUES (" + ", ".join("?" * (10 + len(_FW_COLUMNS))) + ")",
        rows,
    )
    tgt.commit()
    print(f"Built person_period_lexicon: {len(rows)} person-month rows")
    return len(rows)


def build_dyad_terms(tgt: sqlite3.Connection, min_uses=None, min_lift=None,
                     top_k=None) -> int:
    """Rebuild dyad_terms: per chat, the unigrams (>= 3 letters) and bigrams
    used disproportionately often *in that chat* versus the rest of the
    corpus -- inside-joke and nickname candidates for annotate.py --terms to
    curate. Candidates exclude stopwords (the 9 function-word categories plus
    the corpus's _CORPUS_TOP_N most frequent tokens, computed in the same
    pass); a bigram needs both tokens non-stopword, and its uses are counted
    as adjacent token pairs.

    lift = (uses_in_chat / tokens_in_chat) / ((uses_elsewhere + 1) /
    tokens_elsewhere) -- add-one smoothed so a chat-only term gets a finite
    lift. Kept: n_uses >= min_uses AND lift >= min_lift, top_k per chat by
    lift. Provenance (first occurrence by date,id -> first_message_id /
    coined_by_handle_id [NULL = me] / coined_is_me) is tracked in the pass.

    Full rebuild preserving curation: existing llm_verdict values are
    snapshotted per (chat_id, term) and reattached to recomputed rows."""
    min_uses = DYAD_TERM_MIN_USES if min_uses is None else int(min_uses)
    min_lift = DYAD_TERM_MIN_LIFT if min_lift is None else float(min_lift)
    top_k = DYAD_TERM_TOP_K if top_k is None else int(top_k)

    verdicts = {
        (chat_id, term): verdict
        for chat_id, term, verdict in tgt.execute(
            "SELECT chat_id, term, llm_verdict FROM dyad_terms "
            "WHERE llm_verdict IS NOT NULL"
        )
    }
    tgt.execute("DELETE FROM dyad_terms")

    chat_terms = collections.defaultdict(collections.Counter)  # chat -> term uses
    chat_tokens = collections.Counter()      # chat -> token total
    corpus_tokens = collections.Counter()    # token -> corpus frequency (top-N stoplist)
    term_totals = collections.Counter()      # term -> corpus-wide uses
    provenance = {}                          # (chat, term) -> (msg_id, handle-or-None, is_me)

    # Counted up front only to give the streaming pass below a denominator to
    # report progress against; same WHERE, so it stays on the same index.
    n_total = tgt.execute(
        f"SELECT COUNT(*) FROM messages m WHERE {_QUALIFYING_WHERE}"
    ).fetchone()[0]

    # ORDER BY date,id so the first time a (chat, term) pair is seen IS its
    # first occurrence -- provenance falls out of setdefault.
    with progress.track("Scanning messages for terms", total=n_total) as p:
        for chat_id, handle_id, is_from_me, text, msg_id in tgt.execute(
            f"SELECT m.chat_id, m.handle_id, m.is_from_me, m.text, m.id "
            f"FROM messages m WHERE {_QUALIFYING_WHERE} ORDER BY m.date, m.id"
        ):
            tokens = tokenize(text)
            chat_tokens[chat_id] += len(tokens)
            corpus_tokens.update(tokens)
            coiner = (msg_id, None if is_from_me else handle_id, 1 if is_from_me else 0)
            counter = chat_terms[chat_id]
            # Stopword filtering happens after the pass (the top-N list doesn't
            # exist yet); the length gate is applied now to bound the counters.
            terms = [t for t in tokens if len(t) >= 3]
            terms += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
            for term in terms:
                counter[term] += 1
                term_totals[term] += 1
                provenance.setdefault((chat_id, term), coiner)
            p.advance()

    stop = set(STOPWORDS)
    stop.update(tok for tok, _ in corpus_tokens.most_common(_CORPUS_TOP_N))
    total_tokens = sum(chat_tokens.values())

    rows = []
    n_chats = 0
    for chat_id, counter in chat_terms.items():
        tokens_in = chat_tokens[chat_id]
        tokens_elsewhere = total_tokens - tokens_in
        if not tokens_in or not tokens_elsewhere:
            # Single-chat corpus (or an empty one): no "elsewhere" baseline
            # exists, so no term can be distinctive -- skip sanely.
            continue
        kept = []
        for term, n_uses in counter.items():
            if n_uses < min_uses:
                continue
            if " " in term:
                first, second = term.split(" ", 1)
                if first in stop or second in stop:
                    continue
            elif term in stop:
                continue
            uses_elsewhere = term_totals[term] - n_uses
            lift = (n_uses / tokens_in) / ((uses_elsewhere + 1) / tokens_elsewhere)
            if lift >= min_lift:
                kept.append((term, lift, n_uses))
        if not kept:
            continue
        n_chats += 1
        kept.sort(key=lambda t: (-t[1], t[0]))
        for term, lift, n_uses in kept[:top_k]:
            msg_id, coined_by, coined_is_me = provenance[(chat_id, term)]
            rows.append((chat_id, term, lift, n_uses, msg_id, coined_by,
                         coined_is_me, verdicts.get((chat_id, term))))

    tgt.executemany(
        "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, first_message_id, "
        "coined_by_handle_id, coined_is_me, llm_verdict) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    tgt.commit()
    n_curated = sum(1 for r in rows if r[7] is not None)
    print(f"Built dyad_terms: {len(rows)} terms across {n_chats} chats "
          f"({n_curated} verdicts preserved)")
    return len(rows)
