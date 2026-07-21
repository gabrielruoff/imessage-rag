"""Voiceprint pipeline (V1): the deterministic, zero-LLM per-person voice
profile, persisted into person_voiceprint and read back by the get_voiceprint
tool (query_tools/persona.py).

Everything here is pure aggregation over imessage_analysis.db -- no Ollama.
A "voiceprint" answers "how does WHOSE voice sound, WITH WHOM":

  scope='dyad'   member_key's voice in the 1:1 chats with counterpart_key
                 (a real other member_key). Built both directions, so
                 me-with-Mom = ("me","contact:12","dyad") and
                 Mom-with-me = ("contact:12","me","dyad") are distinct rows.
  scope='group'  member_key's voice in ONE group chat; counterpart_key is
                 "chat:<chat_id>".
  scope='all'    member_key's overall voice across every chat;
                 counterpart_key is "*".

member_key/counterpart_key are query_tools._shared.member_key() strings -- the
single home for participant identity, so a contact's duplicate handles collapse
here exactly as in every live aggregate, and a full rebuild self-heals a
handle:N -> contact:M key flip. Rows are never built for the 'unknown' bucket.

Features (all deterministic):
  * per-message style, scanned from RAW messages.text (NOT the lexicon):
    avg_msg_len, lowercase_ratio, exclaim_rate, ellipsis_rate;
  * burst_size -- mean length of the sender's same-sender bursts (runs whose
    intra-run gaps stay under VOICEPRINT_BURST_GAP_SECONDS);
  * the 9 function-word rates, pulled from person_period_lexicon and
    re-aggregated n_words-WEIGHTED (never a naive mean of monthly rates);
  * openers/closers -- top lowercased first/last-3-token phrases of the
    sender's session-opening / -closing messages (from the sessions table);
  * tone_valence / tone_emotions -- LOW-TRUST, from chunk_annotations rows with
    source='llm' ONLY (distilled rows never contribute; a scope with zero llm
    rows gets no tone at all).

Coverage columns n_messages / n_reply_pairs / n_months feed the read-time
confidence gate. feature_std_json carries, per scope, the population std of
every numeric feature (std==0 omitted), replicated onto each row so a
downstream z-normalization (V3) reads it off the row without rescanning.

Rebuild + guard: a full rebuild (DELETE + reinsert) gated on ingest_state's
voiceprint_max_message_id -- unchanged corpus => skip, else rebuild. Like the
other derived tables, reset_db deletes it and the append-only rule's carve-out
applies.

Usage: voiceprint.py [--force] [--dry-run]
"""

import argparse
import collections
import datetime
import json
import statistics

import config
import progress
import schema
from config import TARGET_DB
from db import connect_default, get_state, run_cli, set_state
from function_words import FUNCTION_WORDS, tokenize
from query_tools._shared import (
    _REAL_MESSAGE_CLAUSES,
    _resolve_contact_handle_ids,
    _restrict_to_direct_chats,
    member_key,
)

# Module attribute so tests can patch it; _connect() reads it at call time.
DB_PATH = TARGET_DB

# Function-word categories/columns mirror lexicon_build (declaration order),
# so person_voiceprint's fw_*_rate columns can never disagree with the lexicon.
_FW_CATEGORIES = tuple(FUNCTION_WORDS)
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in _FW_CATEGORIES)

# Numeric style features carried on every row (order stable for feature_std).
_STYLE_FEATURES = ("avg_msg_len", "lowercase_ratio", "exclaim_rate",
                   "ellipsis_rate", "burst_size")
# Every numeric feature the per-scope std map covers (openers/closers/tone are
# not numeric and are excluded by construction).
_NUMERIC_FEATURES = _STYLE_FEATURES + _FW_COLUMNS

_REAL = " AND ".join(_REAL_MESSAGE_CLAUSES)

# INSERT column order == the person_voiceprint DDL in schema.py.
_INSERT_COLS = (
    ["member_key", "counterpart_key", "scope"]
    + list(_STYLE_FEATURES)
    + list(_FW_COLUMNS)
    + ["openers_json", "closers_json", "tone_valence", "tone_emotions_json",
       "n_messages", "n_reply_pairs", "n_months", "feature_std_json"]
)
_INSERT_SQL = (
    f"INSERT INTO person_voiceprint({', '.join(_INSERT_COLS)}) "
    f"VALUES ({', '.join('?' * len(_INSERT_COLS))})"
)

_SIGNATURE_TOKENS = 3   # first/last N tokens of a session-boundary message
_SIGNATURE_TOP = 5      # keep at most this many opener/closer phrases
_SIGNATURE_MIN_COUNT = 3
_TONE_TOP_EMOTIONS = 5


def _connect():
    return connect_default(DB_PATH, rows=False)


# ---------------------------------------------------------------------------
# Identity -> SQL fragments (member_key is the single home; these just translate
# a resolved key into the WHERE clauses the lexicon/sessions queries need)
# ---------------------------------------------------------------------------

def _member_handle_ids(con, member_key_str):
    """Handle ids the (non-me) member owns, or None for 'me'. contact:N
    collapses the contact's duplicate handles; handle:N is a single handle."""
    if member_key_str == "me":
        return None
    kind, _, raw = member_key_str.partition(":")
    if kind == "contact":
        return _resolve_contact_handle_ids(con, [int(raw)])
    if kind == "handle":
        return [int(raw)]
    return []


def _lexicon_member_where(con, member_key_str):
    """(clause, params) selecting the member's person_period_lexicon rows.
    The lexicon encodes 'me' as handle_id NULL (from-me rows carry the
    counterparty's handle, never mine), matching build_person_period_lexicon."""
    if member_key_str == "me":
        return "handle_id IS NULL", []
    ids = _member_handle_ids(con, member_key_str)
    if not ids:
        return "0", []
    return f"handle_id IN ({','.join('?' * len(ids))})", list(ids)


# ---------------------------------------------------------------------------
# Raw-message scan: per-message style, bursts, reply pairs
# ---------------------------------------------------------------------------

def _msg_style(text):
    """Per-message signals: (length, lowercase_ratio|None, has_exclaim,
    has_ellipsis). lowercase_ratio is over cased letters only (None when the
    message has no letters, so it never dilutes the average)."""
    letters = [c for c in text if c.isalpha()]
    lower = (sum(1 for c in letters if c.islower()) / len(letters)
             if letters else None)
    has_exclaim = 1 if "!" in text else 0
    has_ellipsis = 1 if ("..." in text or "…" in text) else 0
    return len(text), lower, has_exclaim, has_ellipsis


def _scan_chats(con, chat_ids, member_key_str, date_from, date_to, budget):
    """One (date, id)-ordered pass over the real messages of each chat,
    classifying every row as the member's or not via member_key. Accumulates
    the member's per-message style, burst-run count, reply pairs (a non-member
    message immediately followed by a member message), and active months.

    burst_size falls out as n_messages / n_runs: every member message belongs
    to exactly one maximal same-sender run, and a run breaks whenever the other
    party speaks OR the gap to the previous member message reaches
    VOICEPRINT_BURST_GAP_SECONDS. Streams are never joined across chats (bursts
    and reply pairs reset per chat)."""
    gap = float(config.VOICEPRINT_BURST_GAP_SECONDS)
    acc = {
        "n_messages": 0, "n_text": 0, "sum_len": 0,
        "sum_lower": 0.0, "n_lower": 0, "n_exclaim": 0, "n_ellipsis": 0,
        "n_runs": 0, "reply_pairs": 0, "months": set(),
    }
    scanned = 0
    capped = False
    for cid in chat_ids:
        if budget is not None and scanned >= budget:
            capped = True
            break
        clauses = ["m.chat_id = ?", "m.date IS NOT NULL", _REAL]
        params = [cid]
        if date_from is not None:
            clauses.append("m.date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("m.date < ?")
            params.append(date_to)
        sql = (f"SELECT m.is_from_me, m.handle_id, h.contact_id, m.date, m.text "
               f"FROM messages m LEFT JOIN handles h ON h.id = m.handle_id "
               f"WHERE {' AND '.join(clauses)} ORDER BY m.date, m.id")
        if budget is not None:
            sql += " LIMIT ?"
            params.append(budget - scanned)

        prev_member = False
        prev_dt = None
        in_run = False
        for is_from_me, handle_id, contact_id, date, text in con.execute(sql, params):
            scanned += 1
            is_member = member_key(is_from_me, contact_id, handle_id) == member_key_str
            cur_dt = datetime.datetime.fromisoformat(date)
            if is_member:
                # Reply pair: the immediately preceding real message was the
                # other side. Gap-agnostic -- this is a coverage measure.
                if prev_dt is not None and not prev_member:
                    acc["reply_pairs"] += 1
                # Burst run bookkeeping.
                if in_run and (cur_dt - prev_dt).total_seconds() < gap:
                    pass  # extends the current run
                else:
                    acc["n_runs"] += 1
                in_run = True
                acc["n_messages"] += 1
                acc["months"].add(date[:7])
                if text:
                    ln, lower, has_exclaim, has_ellipsis = _msg_style(text)
                    acc["n_text"] += 1
                    acc["sum_len"] += ln
                    if lower is not None:
                        acc["sum_lower"] += lower
                        acc["n_lower"] += 1
                    acc["n_exclaim"] += has_exclaim
                    acc["n_ellipsis"] += has_ellipsis
            else:
                in_run = False
            prev_member = is_member
            prev_dt = cur_dt
    return acc, scanned, capped


# ---------------------------------------------------------------------------
# Lexicon (function-word rates), sessions (openers/closers), tone
# ---------------------------------------------------------------------------

def _fw_rates(con, member_key_str, chat_ids, month_from=None, month_to=None):
    """The member's 9 function-word rates, aggregated from
    person_period_lexicon n_words-WEIGHTED across the scope's months -- a
    naive mean of monthly rates would over-weight quiet months. Missing lexicon
    (never built) yields all-zero rates."""
    where, params = _lexicon_member_where(con, member_key_str)
    clauses = [where]
    if chat_ids is not None:
        clauses.append(f"chat_id IN ({','.join('?' * len(chat_ids))})")
        params += list(chat_ids)
    if month_from is not None:
        clauses.append("month >= ?")
        params.append(month_from)
    if month_to is not None:
        clauses.append("month <= ?")
        params.append(month_to)
    cols = ", ".join(_FW_COLUMNS)
    hits = [0.0] * len(_FW_COLUMNS)
    total_words = 0
    for row in con.execute(
        f"SELECT n_words, {cols} FROM person_period_lexicon "
        f"WHERE {' AND '.join(clauses)}", params
    ):
        n_words = row[0] or 0
        total_words += n_words
        for i in range(len(_FW_COLUMNS)):
            hits[i] += (row[i + 1] or 0.0) * n_words
    return {col: (hits[i] / total_words if total_words else 0.0)
            for i, col in enumerate(_FW_COLUMNS)}


def _phrase_counts(con, member_key_str, chat_ids, msg_col, role, first):
    """Top lowercased first/last-3-token phrases of the member's session-
    opening (role='initiator', msg_col='first_message_id') or -closing
    (role='ender', 'last_message_id') messages. Kept: top _SIGNATURE_TOP with
    count >= _SIGNATURE_MIN_COUNT. Empty when the sessions table is unbuilt."""
    if member_key_str == "me":
        member_clause = f"s.{role}_is_me = 1"
        member_params = []
    else:
        ids = _member_handle_ids(con, member_key_str)
        if not ids:
            return []
        member_clause = (f"s.{role}_is_me = 0 AND "
                         f"s.{role}_handle_id IN ({','.join('?' * len(ids))})")
        member_params = list(ids)
    clauses = [member_clause]
    params = list(member_params)
    if chat_ids is not None:
        clauses.append(f"s.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += list(chat_ids)
    counts = collections.Counter()
    for (text,) in con.execute(
        f"SELECT mo.text FROM sessions s "
        f"JOIN messages mo ON mo.id = s.{msg_col} "
        f"WHERE {' AND '.join(clauses)} AND mo.text IS NOT NULL", params
    ):
        toks = tokenize(text)
        if not toks:
            continue
        phrase = " ".join(toks[:_SIGNATURE_TOKENS] if first
                          else toks[-_SIGNATURE_TOKENS:])
        counts[phrase] += 1
    return [{"phrase": p, "count": n}
            for p, n in counts.most_common(_SIGNATURE_TOP)
            if n >= _SIGNATURE_MIN_COUNT]


def _tone(con, chat_ids):
    """LOW-TRUST persona tone over the scope's chunk_annotations, source='llm'
    ONLY -- distilled rows carry the seed model's bias and empty events, so
    they never contribute. Returns (mean_valence|None, [emotions]|None); both
    None when the scope has zero llm annotations, so the caller omits tone
    entirely rather than reporting a spurious neutral."""
    clauses = ["ca.source = 'llm'"]
    params = []
    if chat_ids is not None:
        clauses.append(f"c.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += list(chat_ids)
    rows = con.execute(
        f"SELECT ca.valence, ca.emotions_json FROM chunk_annotations ca "
        f"JOIN chunks c ON c.id = ca.chunk_id "
        f"WHERE {' AND '.join(clauses)}", params
    ).fetchall()
    if not rows:
        return None, None
    valences = [r[0] for r in rows if r[0] is not None]
    valence = round(sum(valences) / len(valences), 4) if valences else None
    emotions = collections.Counter()
    for _valence, emotions_json in rows:
        try:
            labels = json.loads(emotions_json) if emotions_json else []
        except (TypeError, ValueError):
            labels = []
        emotions.update(e for e in labels if isinstance(e, str))
    top = [e for e, _ in emotions.most_common(_TONE_TOP_EMOTIONS)]
    return valence, (top or None)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def compute_voiceprint(con, member_key_str, chat_ids, *, date_from=None,
                       date_to=None, budget=None):
    """Assemble the full deterministic feature dict for one identity within
    chat_ids (order-independent style/tone/fw; per-chat bursts + reply pairs).
    Returns None when the member has no real messages in scope. `budget` caps
    the raw scan (the date-scoped read path passes VOICEPRINT_SCAN_MAX_MESSAGES);
    a truncated scan is flagged via the returned 'capped'."""
    if not chat_ids:
        return None
    acc, _scanned, capped = _scan_chats(con, chat_ids, member_key_str,
                                        date_from, date_to, budget)
    n_messages = acc["n_messages"]
    if n_messages == 0:
        return None
    n_text = acc["n_text"]
    month_from = date_from[:7] if date_from else None
    month_to = date_to[:7] if date_to else None
    openers = _phrase_counts(con, member_key_str, chat_ids,
                             "first_message_id", "initiator", first=True)
    closers = _phrase_counts(con, member_key_str, chat_ids,
                             "last_message_id", "ender", first=False)
    tone_valence, tone_emotions = _tone(con, chat_ids)
    return {
        "avg_msg_len": (acc["sum_len"] / n_text) if n_text else 0.0,
        "lowercase_ratio": (acc["sum_lower"] / acc["n_lower"]) if acc["n_lower"] else 0.0,
        "exclaim_rate": (acc["n_exclaim"] / n_text) if n_text else 0.0,
        "ellipsis_rate": (acc["n_ellipsis"] / n_text) if n_text else 0.0,
        "burst_size": n_messages / acc["n_runs"],  # n_runs >= 1 when n_messages > 0
        "fw": _fw_rates(con, member_key_str, chat_ids, month_from, month_to),
        "openers": openers,
        "closers": closers,
        "tone_valence": tone_valence,
        "tone_emotions": tone_emotions,
        "n_messages": n_messages,
        "n_reply_pairs": acc["reply_pairs"],
        "n_months": len(acc["months"]),
        "capped": capped,
    }


def _feature_value(result, feature):
    """A numeric feature's value out of a compute_voiceprint result (style
    features are top-level keys; fw_* live under 'fw')."""
    return result["fw"].get(feature) if feature.startswith("fw_") else result.get(feature)


# ---------------------------------------------------------------------------
# Scope helpers shared with the read path (query_tools/persona.py)
# ---------------------------------------------------------------------------

def member_chats(con, member_key_str, *, direct_only=False):
    """Distinct chats where the member sent a real message (all their chats
    for scope='all'; direct_only restricts to 1:1 chats for a dyad recompute)."""
    if member_key_str == "me":
        rows = con.execute(
            f"SELECT DISTINCT chat_id FROM messages m "
            f"WHERE m.is_from_me = 1 AND {_REAL} AND chat_id IS NOT NULL")
    else:
        ids = _member_handle_ids(con, member_key_str)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT DISTINCT chat_id FROM messages m "
            f"WHERE m.is_from_me = 0 AND m.handle_id IN ({placeholders}) "
            f"AND {_REAL} AND chat_id IS NOT NULL", ids)
    chats = [r[0] for r in rows]
    if direct_only:
        chats = _restrict_to_direct_chats(con, chats)
    return sorted(chats)


def scan_size(con, chat_ids, date_from=None, date_to=None):
    """Count of real messages the raw scan of chat_ids (optionally date-bound)
    would touch -- the read path's pre-check against VOICEPRINT_SCAN_MAX_MESSAGES."""
    if not chat_ids:
        return 0
    total = 0
    for cid in chat_ids:
        clauses = ["m.chat_id = ?", "m.date IS NOT NULL", _REAL]
        params = [cid]
        if date_from is not None:
            clauses.append("m.date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("m.date < ?")
            params.append(date_to)
        total += con.execute(
            f"SELECT COUNT(*) FROM messages m WHERE {' AND '.join(clauses)}",
            params).fetchone()[0]
    return total


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _enumerate_tasks(con):
    """Every (member_key, counterpart_key, scope, chat_ids) the corpus implies.
    dyad rows both directions per direct-chat counterpart; group rows per
    participant per group chat; one 'all' row per identity."""
    tasks = []

    # dyad: real counterpart -> the set of 1:1 chats they appear in.
    direct = {}
    for chat_id, is_from_me, handle_id, contact_id in con.execute(
        f"SELECT DISTINCT m.chat_id, m.is_from_me, m.handle_id, h.contact_id "
        f"FROM messages m JOIN chats ch ON ch.id = m.chat_id "
        f"LEFT JOIN handles h ON h.id = m.handle_id "
        f"WHERE ch.is_group = 0 AND m.is_from_me = 0 AND {_REAL} "
        f"AND m.chat_id IS NOT NULL"
    ):
        key = member_key(is_from_me, contact_id, handle_id)
        if key == "unknown":
            continue
        direct.setdefault(key, set()).add(chat_id)
    for ckey, chats in sorted(direct.items()):
        chat_list = sorted(chats)
        tasks.append(("me", ckey, "dyad", chat_list))
        tasks.append((ckey, "me", "dyad", chat_list))

    # group: per group chat, every participating identity (including me).
    group = {}
    for chat_id, is_from_me, handle_id, contact_id in con.execute(
        f"SELECT DISTINCT m.chat_id, m.is_from_me, m.handle_id, h.contact_id "
        f"FROM messages m JOIN chats ch ON ch.id = m.chat_id "
        f"LEFT JOIN handles h ON h.id = m.handle_id "
        f"WHERE ch.is_group = 1 AND {_REAL} AND m.chat_id IS NOT NULL"
    ):
        key = member_key(is_from_me, contact_id, handle_id)
        if key == "unknown":
            continue
        group.setdefault(chat_id, set()).add(key)
    for chat_id, keys in sorted(group.items()):
        for key in sorted(keys):
            tasks.append((key, f"chat:{chat_id}", "group", [chat_id]))

    # all: one global row per identity, over every chat they appear in.
    allkeys = {}
    for is_from_me, handle_id, contact_id, chat_id in con.execute(
        f"SELECT DISTINCT m.is_from_me, m.handle_id, h.contact_id, m.chat_id "
        f"FROM messages m LEFT JOIN handles h ON h.id = m.handle_id "
        f"WHERE {_REAL} AND m.chat_id IS NOT NULL"
    ):
        key = member_key(is_from_me, contact_id, handle_id)
        if key == "unknown":
            continue
        allkeys.setdefault(key, set()).add(chat_id)
    for key, chats in sorted(allkeys.items()):
        tasks.append((key, "*", "all", sorted(chats)))

    return tasks


def _std_map(results):
    """Per-scope population std of each numeric feature across the given
    compute results; features with std == 0 (constant, or a single row) are
    omitted so a downstream z-normalization simply skips them (no /0)."""
    std = {}
    for feature in _NUMERIC_FEATURES:
        values = [_feature_value(r, feature) for r in results]
        values = [v for v in values if v is not None]
        if not values:
            continue
        spread = statistics.pstdev(values)
        if spread > 0:
            std[feature] = round(spread, 6)
    return std


def _row_tuple(member_key_str, counterpart_key, scope, result, std_json):
    return (
        member_key_str, counterpart_key, scope,
        result["avg_msg_len"], result["lowercase_ratio"], result["exclaim_rate"],
        result["ellipsis_rate"], result["burst_size"],
        *[result["fw"][col] for col in _FW_COLUMNS],
        json.dumps(result["openers"], ensure_ascii=False),
        json.dumps(result["closers"], ensure_ascii=False),
        result["tone_valence"],
        (json.dumps(result["tone_emotions"], ensure_ascii=False)
         if result["tone_emotions"] is not None else None),
        result["n_messages"], result["n_reply_pairs"], result["n_months"],
        std_json,
    )


def _corpus_max_message_id(con):
    return int(con.execute("SELECT MAX(id) FROM messages").fetchone()[0] or 0)


def build_voiceprints(con, *, force=False, dry_run=False):
    """Full rebuild of person_voiceprint, guarded by ingest_state's
    voiceprint_max_message_id: an unchanged corpus is skipped (unless --force),
    otherwise every row is DELETEd and recomputed -- which self-heals a
    handle:N -> contact:M key flip, since keys are re-derived from the current
    contact_id each time. feature_std_json is recomputed per scope each rebuild.
    Returns the number of rows written."""
    # Self-heal a DB predating the table (voiceprint.py is reachable without a
    # fresh ingest, which is what normally applies migrations).
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_voiceprint'"
    ).fetchone():
        schema.create_schema(con)
        con.commit()

    current_max = _corpus_max_message_id(con)
    stored = get_state(con, "voiceprint_max_message_id")
    if not force and stored is not None and int(stored) == current_max:
        print(f"Voiceprints up to date (max message id {current_max}, 0 built)")
        return 0

    tasks = _enumerate_tasks(con)
    if dry_run:
        by_scope = collections.Counter(scope for _m, _c, scope, _ch in tasks)
        print(f"--dry-run: {len(tasks)} candidate rows "
              f"({dict(sorted(by_scope.items()))}); not written")
        return 0

    con.execute("DELETE FROM person_voiceprint")

    built = []  # (member_key, counterpart_key, scope, result)
    with progress.track("Building voiceprints", total=len(tasks)) as p:
        for member_key_str, counterpart_key, scope, chat_ids in tasks:
            result = compute_voiceprint(con, member_key_str, chat_ids)
            if result is not None:
                built.append((member_key_str, counterpart_key, scope, result))
            p.advance()

    # Per-scope corpus std, replicated onto each row of that scope.
    std_json_by_scope = {}
    by_scope = collections.defaultdict(list)
    for _m, _c, scope, result in built:
        by_scope[scope].append(result)
    for scope, results in by_scope.items():
        std_json_by_scope[scope] = json.dumps(_std_map(results))

    con.executemany(_INSERT_SQL, [
        _row_tuple(member_key_str, counterpart_key, scope, result,
                   std_json_by_scope[scope])
        for member_key_str, counterpart_key, scope, result in built
    ])
    set_state(con, "voiceprint_max_message_id", current_max)
    con.commit()

    n_scope = collections.Counter(scope for _m, _c, scope, _r in built)
    print(f"Built {len(built)} voiceprints "
          f"({dict(sorted(n_scope.items()))}; max message id {current_max})")
    return len(built)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build deterministic per-person voiceprints "
                    "(person_voiceprint). Pure SQL/Python, no Ollama.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the corpus is unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="print how many rows would be built and exit")
    args = parser.parse_args(argv)
    con = _connect()
    try:
        build_voiceprints(con, force=args.force, dry_run=args.dry_run)
    finally:
        con.close()


if __name__ == "__main__":
    run_cli(main)
