"""Persona query tools.

V1 exposes get_voiceprint ONLY: read a person's precomputed deterministic
voiceprint from person_voiceprint (built by voiceprint.py), or recompute it
over a date window on the fly (capped). Later issues add draft_reply (V2),
style-match (V3) and simulate (V5) here -- this module stays a thin read layer
over the voiceprint the batch CLI computes; it never runs an LLM.

The voiceprint answers "how does WHOSE voice sound, WITH WHOM". The subject
defaults to "me": name a person to get YOUR voice with them (a dyad), or omit
to get your overall voice (scope='all'). subject='them' returns the named
person's voice instead. Group chats surface only when scope='group' is asked
for explicitly, with a group chat_id.
"""

import contextlib
import json

import config
from function_words import FUNCTION_WORDS

from ._shared import (
    _connect,
    _normalize_date_bound,
    member_key,
)

# Feature key sets, kept in step with voiceprint.py / person_voiceprint.
_STYLE_FEATURES = ("avg_msg_len", "lowercase_ratio", "exclaim_rate",
                   "ellipsis_rate", "burst_size")
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in FUNCTION_WORDS)


def _table_exists(con):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_voiceprint'"
    ).fetchone() is not None


def _confidence(n_messages, n_reply_pairs, n_months):
    """The card confidence gate: 'high' only when volume, two-sided engagement,
    and breadth-over-time ALL clear their thresholds (read at call time so
    tests/.env can move them), else 'low'."""
    if (n_messages >= config.VOICEPRINT_MIN_MESSAGES
            and n_reply_pairs >= config.VOICEPRINT_MIN_REPLY_PAIRS
            and n_months >= config.VOICEPRINT_MIN_MONTHS):
        return "high"
    return "low"


def _contact_label(con, contact_id):
    row = con.execute(
        "SELECT COALESCE("
        "NULLIF(nickname, ''), "
        "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), '')"
        ") FROM contacts WHERE id = ?", (contact_id,)
    ).fetchone()
    if row and row[0]:
        return row[0]
    handle = con.execute(
        "SELECT handle_identifier FROM handles WHERE contact_id = ? "
        "ORDER BY id LIMIT 1", (contact_id,)
    ).fetchone()
    return handle[0] if handle else f"contact {contact_id}"


def _label_for_key(con, key):
    """Human-readable label for a member_key / counterpart_key string."""
    if key == "me":
        return "Me"
    if key == "*":
        return "everyone"
    kind, _, raw = key.partition(":")
    if kind == "contact":
        return _contact_label(con, int(raw))
    if kind == "handle":
        row = con.execute(
            "SELECT handle_identifier FROM handles WHERE id = ?", (int(raw),)
        ).fetchone()
        return row[0] if row else key
    if kind == "chat":
        row = con.execute(
            "SELECT COALESCE(NULLIF(display_name, ''), chat_identifier) "
            "FROM chats WHERE id = ?", (int(raw),)
        ).fetchone()
        return (row[0] if row and row[0] else f"group chat {raw}")
    return key


def _resolve_person(con, contact_id, handle_id, name):
    """Resolve a person param into (member_key, label, error_dict). All three
    None means no person was named. error_dict (non-None) is returned to the
    caller verbatim -- an unknown handle or an ambiguous name never guesses."""
    if contact_id is not None:
        return f"contact:{contact_id}", _contact_label(con, contact_id), None
    if handle_id is not None:
        row = con.execute(
            "SELECT contact_id, handle_identifier FROM handles WHERE id = ?",
            (handle_id,)
        ).fetchone()
        if row is None:
            return None, None, {"error": f"no handle with id {handle_id}"}
        cid, ident = row[0], row[1]
        key = member_key(False, cid, handle_id)
        label = _contact_label(con, cid) if cid is not None else ident
        return key, label, None
    if name:
        like = f"%{name}%"
        rows = con.execute(
            "SELECT id, COALESCE("
            "NULLIF(nickname, ''), "
            "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), ''), "
            "'') AS label FROM contacts "
            "WHERE first_name LIKE ? OR last_name LIKE ? OR nickname LIKE ? "
            "OR TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')) LIKE ? "
            "ORDER BY last_name, first_name LIMIT 10",
            (like, like, like, like)
        ).fetchall()
        if not rows:
            return None, None, {"error": f"no contact matches {name!r}"}
        if len(rows) > 1:
            return None, None, {
                "error": f"{len(rows)} contacts match {name!r}; pass contact_id",
                "matches": [{"contact_id": r[0], "label": r[1] or f"contact {r[0]}"}
                            for r in rows],
            }
        return f"contact:{rows[0][0]}", (rows[0][1] or f"contact {rows[0][0]}"), None
    return None, None, None


def _tone_block(valence, emotions_json):
    """The low-trust tone block, or None when the scope had no source='llm'
    annotations (both fields NULL) -- tone is then omitted from the card, never
    reported as a spurious neutral."""
    if valence is None and not emotions_json:
        return None
    try:
        emotions = json.loads(emotions_json) if emotions_json else []
    except (TypeError, ValueError):
        emotions = []
    return {"valence": valence, "emotions": emotions, "low_trust": True}


def _base_card(con, member_key_str, counterpart_key, scope, n_messages,
               n_reply_pairs, n_months, features, openers, closers,
               feature_std, tone):
    card = {
        "member_key": member_key_str,
        "counterpart_key": counterpart_key,
        "scope": scope,
        "member": _label_for_key(con, member_key_str),
        "counterpart": _label_for_key(con, counterpart_key),
        "confidence": _confidence(n_messages, n_reply_pairs, n_months),
        "n_messages": n_messages,
        "n_reply_pairs": n_reply_pairs,
        "n_months": n_months,
        "features": features,
        "openers": openers,
        "closers": closers,
        "feature_std": feature_std,
    }
    if tone is not None:
        card["tone"] = tone
    return card


def _card_from_row(con, row):
    features = {f: row[f] for f in _STYLE_FEATURES}
    features.update({c: row[c] for c in _FW_COLUMNS})
    return _base_card(
        con, row["member_key"], row["counterpart_key"], row["scope"],
        row["n_messages"], row["n_reply_pairs"], row["n_months"],
        features,
        json.loads(row["openers_json"] or "[]"),
        json.loads(row["closers_json"] or "[]"),
        json.loads(row["feature_std_json"] or "{}"),
        _tone_block(row["tone_valence"], row["tone_emotions_json"]),
    )


def _lookup(con, member_key_str, counterpart_key, scope):
    if not _table_exists(con):
        return {"note": "person_voiceprint not built yet -- run `make voiceprint`.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "found": False}
    row = con.execute(
        "SELECT * FROM person_voiceprint "
        "WHERE member_key = ? AND counterpart_key = ? AND scope = ?",
        (member_key_str, counterpart_key, scope)
    ).fetchone()
    if row is None:
        return {"note": f"No voiceprint on file for member={member_key_str}, "
                        f"counterpart={counterpart_key}, scope={scope} "
                        "(too little interaction, or run `make voiceprint`).",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "found": False}
    return _card_from_row(con, row)


def _precomputed_std(con, member_key_str, counterpart_key, scope):
    """The stored per-scope feature std for this key -- a corpus property, so a
    date-scoped recompute reuses it rather than rescanning the whole corpus."""
    if not _table_exists(con):
        return {}
    row = con.execute(
        "SELECT feature_std_json FROM person_voiceprint "
        "WHERE member_key = ? AND counterpart_key = ? AND scope = ?",
        (member_key_str, counterpart_key, scope)
    ).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _recompute(con, member_key_str, counterpart_key, scope, chat_id, person_key,
               date_from, date_to):
    """Date-scoped path: recompute the features over a capped raw scan rather
    than reading the (whole-corpus) precomputed row."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle

    df = _normalize_date_bound(date_from) if date_from is not None else None
    dt = _normalize_date_bound(date_to, upper=True) if date_to is not None else None

    if scope == "dyad":
        chats = voiceprint.member_chats(con, person_key, direct_only=True)
    elif scope == "group":
        chats = [chat_id]
    else:  # all
        chats = voiceprint.member_chats(con, member_key_str)

    cap = config.VOICEPRINT_SCAN_MAX_MESSAGES
    scan_n = voiceprint.scan_size(con, chats, df, dt)
    if scan_n > cap:
        return {"note": f"Date range too wide: {scan_n} messages exceed the "
                        f"{cap}-message scan cap -- narrow date_from/date_to.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "messages_in_range": scan_n, "recomputed": False}

    result = voiceprint.compute_voiceprint(
        con, member_key_str, chats, date_from=df, date_to=dt, budget=cap)
    if result is None:
        return {"note": "No messages in the requested date range.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "recomputed": True, "n_messages": 0}

    features = {f: result[f] for f in _STYLE_FEATURES}
    features.update(result["fw"])
    card = _base_card(
        con, member_key_str, counterpart_key, scope,
        result["n_messages"], result["n_reply_pairs"], result["n_months"],
        features, result["openers"], result["closers"],
        _precomputed_std(con, member_key_str, counterpart_key, scope),
        _tone_block(result["tone_valence"],
                    json.dumps(result["tone_emotions"])
                    if result["tone_emotions"] is not None else None),
    )
    card["recomputed"] = True
    card["date_from"] = date_from
    card["date_to"] = date_to
    if result["n_messages"] < config.VOICEPRINT_MIN_RANGE_MESSAGES:
        card["note"] = (f"Only {result['n_messages']} messages in range "
                        f"(< {config.VOICEPRINT_MIN_RANGE_MESSAGES}); this "
                        "voiceprint may be unreliable.")
    return card


def get_voiceprint(contact_id=None, handle_id=None, name=None, subject="me",
                   scope=None, chat_id=None, date_from=None, date_to=None):
    """A person's deterministic voiceprint -- how they text.

    Defaults to YOUR voice (subject='me'): name a person
    (contact_id/handle_id/name) to get your voice WITH them (scope='dyad'), or
    omit the person for your overall voice (scope='all'). subject='them'
    returns the named person's voice instead. scope='group' needs a group
    chat_id (group rows are never returned unless asked for). Passing
    date_from/date_to recomputes the card over that window from a capped raw
    scan; otherwise the precomputed person_voiceprint row is returned.

    The card carries the style/function-word features, signature
    openers/closers, a low-trust tone block (only when LLM chunk annotations
    exist for the scope), coverage counts, a confidence flag (high/low), and
    the per-feature corpus std (feature_std) for downstream normalization."""
    subject = (subject or "me").lower()
    if subject not in ("me", "them"):
        return {"error": "subject must be 'me' or 'them'"}

    with contextlib.closing(_connect()) as con:
        person_key, _label, err = _resolve_person(con, contact_id, handle_id, name)
        if err:
            return err

        if scope is None:
            scope = "dyad" if person_key is not None else "all"
        scope = str(scope).lower()
        if scope not in ("dyad", "group", "all"):
            return {"error": f"scope must be dyad, group, or all (got {scope!r})"}

        if scope == "dyad":
            if person_key is None:
                return {"error": "scope='dyad' needs a person "
                                 "(contact_id/handle_id/name)"}
            member_key_str, counterpart_key = (
                ("me", person_key) if subject == "me" else (person_key, "me"))
        elif scope == "all":
            counterpart_key = "*"
            if subject == "me":
                member_key_str = "me"
            elif person_key is None:
                return {"error": "scope='all' with subject='them' needs a person"}
            else:
                member_key_str = person_key
        else:  # group
            if chat_id is None:
                return {"error": "scope='group' needs a group chat_id"}
            counterpart_key = f"chat:{chat_id}"
            if subject == "me":
                member_key_str = "me"
            elif person_key is None:
                return {"error": "scope='group' with subject='them' needs a person"}
            else:
                member_key_str = person_key

        if date_from is not None or date_to is not None:
            return _recompute(con, member_key_str, counterpart_key, scope,
                              chat_id, person_key, date_from, date_to)
        return _lookup(con, member_key_str, counterpart_key, scope)


# --- feature: persona ---
# V2: draft_reply -- in-tool generation of a reply to an incoming message in the
# USER's voice with a named person, grounded ONLY in REAL reply-in-context
# exemplars retrieved from that dyad. Two load-bearing correctness points:
#   1. Reply-in-context pairing: within a retrieved chunk, an inbound message is
#      paired with the is_from_me=1 message that IMMEDIATELY follows it in
#      (date, id) order over REAL messages (emote/system rows can neither anchor
#      nor serve as the reply) -- NOT every inbound x self co-occurrence.
#   2. STRUCTURAL decline: below the per-call VOICEPRINT_MIN_CONTEXT_PAIRS floor
#      the tool makes NO generation call at all -- it returns the labeled "your
#      recent replies to this person" pack with draft=None. Decline is enforced
#      in code, not asked of the model.
# The dyad_terms post-check then strips ungrounded inside-jokes/nicknames from
# the generated draft before returning. See RES-62 / docs/DECISIONS.md.

_EXEMPLAR_CAP = 8   # exemplar-bundle ceiling (issue: <= 8, each at TEXT_PREVIEW_CHARS)


def _clip(text, limit):
    """Truncate a message body to `limit` chars for prompt/payload size, with a
    trailing ellipsis (the same marker the shared message truncator uses)."""
    return text if text is None or len(text) <= limit else text[:limit] + "…"


def _dyad_chats(con, person_key):
    """The 1:1 chats between me and this counterpart (direct_only), via the
    voiceprint scope helper -- the single home for a dyad's chat set, so the
    retrieval scope matches the voiceprint's exactly."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle

    return voiceprint.member_chats(con, person_key, direct_only=True)


def reply_pairs_in_chunk(con, chunk_id):
    """The ACTUAL reply-in-context pairs inside one chunk: each inbound
    (is_from_me=0) message paired with the is_from_me=1 message that IMMEDIATELY
    follows it, ordering by (date, id) over REAL messages only
    (_REAL_MESSAGE_CLAUSES -- emote/system rows can neither anchor nor serve as
    the reply). This is the load-bearing correctness point: adjacency in the
    time-ordered stream, NOT every inbound x self co-occurrence in the chunk.

    Rows are read positionally so the helper is row_factory-agnostic. Pairs
    whose either side has no text (an attachment-only message can't be a text
    exemplar) are skipped."""
    from ._shared import TEXT_PREVIEW_CHARS, _REAL_MESSAGE_CLAUSES

    real = " AND ".join(_REAL_MESSAGE_CLAUSES)
    rows = con.execute(
        f"SELECT m.id, m.is_from_me, m.text, m.date FROM messages m "
        f"WHERE m.chunk_id = ? AND {real} ORDER BY m.date, m.id",
        (chunk_id,),
    ).fetchall()
    pairs = []
    for i in range(len(rows) - 1):
        cur_id, cur_me, cur_text, _cur_date = rows[i][0], rows[i][1], rows[i][2], rows[i][3]
        nxt_id, nxt_me, nxt_text, nxt_date = rows[i + 1][0], rows[i + 1][1], \
            rows[i + 1][2], rows[i + 1][3]
        if cur_me == 0 and nxt_me == 1 and cur_text and nxt_text:
            pairs.append({
                "incoming": _clip(cur_text, TEXT_PREVIEW_CHARS),
                "reply": _clip(nxt_text, TEXT_PREVIEW_CHARS),
                "date": nxt_date,
                "incoming_id": cur_id,
                "reply_id": nxt_id,
            })
    return pairs


def _vectors_available(con):
    """chunk_vectors is a sqlite-vec vec0 table built by chunking (needs the
    extension). When it's absent (e.g. a not-yet-embedded corpus, or a hermetic
    test DB) the semantic arm is skipped -- the FTS arm alone still retrieves."""
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'"
    ).fetchone() is not None


def _similar_chunk_ids(con, incoming, chat_ids, pool):
    """Chunks in the dyad most relevant to `incoming`, fused (reciprocal-rank)
    from an FTS keyword arm (always available) and a chunk-vector semantic arm
    (skipped when chunk_vectors isn't built or embedding is unavailable),
    best-first. Reuses the chunk-retrieval primitives so scope/date handling
    stays identical to search_messages_hybrid."""
    from ._shared import RRF_K
    from .chunks import _chunk_knn, _embed_query, _fts_chunk_ranks

    fts_ranks = _fts_chunk_ranks(con, incoming, chat_ids, None, None)
    vec_ranks = {}
    if _vectors_available(con):
        try:
            vec_rows = _chunk_knn(con, _embed_query(incoming), pool, chat_ids, None, None)
            vec_ranks = {r["chunk_id"]: rank for rank, r in enumerate(vec_rows)}
        except Exception:
            # A missing embedder / unreachable Ollama must degrade to FTS-only,
            # never break the tool.
            vec_ranks = {}
    scores = {}
    for ranks in (vec_ranks, fts_ranks):
        for cid, rank in ranks.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _recent_chunk_ids(con, chat_ids, pool):
    """The dyad's most recent chunks (by last_date) -- the recency source for
    the honest-degrade 'your recent replies to this person' pack."""
    if not chat_ids:
        return []
    placeholders = ",".join("?" * len(chat_ids))
    rows = con.execute(
        f"SELECT id FROM chunks WHERE chat_id IN ({placeholders}) "
        f"ORDER BY last_date DESC, id DESC LIMIT ?",
        list(chat_ids) + [pool],
    ).fetchall()
    return [r[0] for r in rows]


def _collect_pairs(con, chunk_ids, cap):
    """Reply-in-context pairs across `chunk_ids` in order, deduped by
    (incoming_id, reply_id), capped at `cap`."""
    pairs, seen = [], set()
    for cid in chunk_ids:
        for pair in reply_pairs_in_chunk(con, cid):
            key = (pair["incoming_id"], pair["reply_id"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
            if len(pairs) >= cap:
                return pairs
    return pairs


def _public_exemplars(pairs):
    """The exemplar bundle as returned to the caller -- drops the private id
    fields used only for dedup."""
    return [{"incoming": p["incoming"], "reply": p["reply"], "date": p["date"]}
            for p in pairs]


def _style_summary(card):
    """A compact, bounded style line from the V1 voiceprint card, or '' when no
    card is on file (the exemplars still carry the voice)."""
    if not isinstance(card, dict) or "features" not in card:
        return ""
    feats = card["features"]

    def g(key):
        val = feats.get(key)
        return round(val, 3) if isinstance(val, (int, float)) else val

    bits = [
        f"avg message length ~{g('avg_msg_len')} chars",
        f"lowercase ratio {g('lowercase_ratio')}",
        f"exclaim rate {g('exclaim_rate')}, ellipsis rate {g('ellipsis_rate')}",
        f"typical burst {g('burst_size')} messages",
    ]
    openers = [o.get("phrase") for o in (card.get("openers") or []) if o.get("phrase")][:3]
    closers = [c.get("phrase") for c in (card.get("closers") or []) if c.get("phrase")][:3]
    if openers:
        bits.append("openers: " + ", ".join(openers))
    if closers:
        bits.append("closers: " + ", ".join(closers))
    return "; ".join(str(b) for b in bits)


def _draft_prompt(card, exemplars, incoming, label):
    """The bounded in-tool generation prompt: the V1 style card + the real
    exemplar pairs + the incoming message + quote-verbatim / no-invention rules.
    All grounding is inline; nothing else is in the model's context."""
    system = (
        "You draft a SINGLE text-message reply in the USER's own voice. You are "
        "given the user's style profile and REAL past exchanges (the other "
        "person's message, then how the USER actually replied). Match that "
        "voice: length, casing, punctuation, and phrasing. Rules: reply with "
        "ONLY the message text, nothing else -- no preamble, no quotes, no "
        "explanation. Invent NO facts, plans, names, or inside jokes; use only "
        "what appears in the exchanges or the incoming message. Do NOT introduce "
        "nicknames or slang the user has not been shown to use."
    )
    parts = []
    style = _style_summary(card)
    if style:
        parts.append(f"USER STYLE (with {label}):\n{style}")
    parts.append("PAST EXCHANGES (their message -> your actual reply):")
    for i, ex in enumerate(exemplars, 1):
        parts.append(f"{i}. THEM: {ex['incoming']}\n   YOU: {ex['reply']}")
    parts.append(f"INCOMING message from {label} to reply to:\n{incoming}")
    parts.append("Write the USER's reply now (message text only):")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _strip_ungrounded_terms(con, draft, chat_ids, pairs, incoming):
    """dyad_terms post-check, run IN TOOL CODE on the generated draft: any
    inside-joke/nickname token drawn from the corpus's distinctive-vocabulary
    universe (dyad_terms across all chats) that is NOT grounded -- neither in
    THIS dyad's own dyad_terms nor in the exemplars/incoming the model was shown
    -- is stripped from the draft. Returns (clean_draft, [stripped_tokens])."""
    import re

    from function_words import tokenize

    allowed = set(tokenize(incoming or ""))
    for pair in pairs:
        allowed |= set(tokenize(pair["incoming"]))
        allowed |= set(tokenize(pair["reply"]))
    if chat_ids:
        placeholders = ",".join("?" * len(chat_ids))
        for (term,) in con.execute(
            f"SELECT term FROM dyad_terms WHERE chat_id IN ({placeholders})",
            list(chat_ids),
        ):
            allowed |= set(tokenize(term or ""))
    suspect = set()
    for (term,) in con.execute("SELECT DISTINCT term FROM dyad_terms"):
        suspect |= set(tokenize(term or ""))

    strip = suspect - allowed
    if not strip or not draft:
        return draft, []

    stripped = []
    token_re = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")

    def _repl(match):
        low = match.group(0).lower().replace("’", "'")
        if low in strip:
            stripped.append(low)
            return ""
        return match.group(0)

    clean = token_re.sub(_repl, draft)
    clean = re.sub(r"\s+([,.!?;:])", r"\1", clean)   # tidy space before punctuation
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()  # collapse the gaps left behind
    return clean, sorted(set(stripped))


def draft_reply(incoming, contact_id=None, handle_id=None, name=None):
    """Draft a reply to an INCOMING message in the USER's own voice with a named
    person, grounded ONLY in real reply-in-context exemplars from that 1:1 chat.

    Name the person (contact_id preferred, else handle_id/name) and pass the
    `incoming` message text. Retrieval finds dyad chunks similar to `incoming`,
    then within each takes the reply the USER actually sent immediately after an
    inbound message -> real (their message -> your reply) exemplars.

    Below VOICEPRINT_MIN_CONTEXT_PAIRS such exemplars this DECLINES structurally:
    it makes no generation call and returns {draft: None, confidence: 'low'}
    with your recent replies to that person as the exemplar pack. At or above
    the floor it assembles a bounded prompt (the V1 voiceprint card + the
    exemplars + the incoming + quote-verbatim/no-invention rules), calls the
    chat model in tool, runs the dyad_terms post-check (stripping ungrounded
    inside-jokes/nicknames), and returns {draft, exemplars, confidence: 'high',
    notes}. 1:1 only; the outer chat loop merely relays the result."""
    if not incoming or not str(incoming).strip():
        return {"error": "draft_reply needs the incoming message text"}
    incoming = str(incoming)

    with contextlib.closing(_connect()) as con:
        person_key, label, err = _resolve_person(con, contact_id, handle_id, name)
        if err:
            return err
        if person_key is None:
            return {"error": "draft_reply needs a person (contact_id/handle_id/name)"}

        chat_ids = _dyad_chats(con, person_key)
        if not chat_ids:
            return {"error": f"no 1:1 chat history with {label} to draft a reply from",
                    "counterpart_key": person_key, "counterpart": label}

        from ._shared import HYBRID_POOL, TEXT_PREVIEW_CHARS

        card = _lookup(con, "me", person_key, "dyad")
        floor = config.VOICEPRINT_MIN_CONTEXT_PAIRS

        similar = _similar_chunk_ids(con, incoming, chat_ids, HYBRID_POOL)
        pairs = _collect_pairs(con, similar, _EXEMPLAR_CAP)
        n_pairs = len(pairs)

        base = {
            "member_key": "me",
            "counterpart_key": person_key,
            "counterpart": label,
            "incoming": _clip(incoming, TEXT_PREVIEW_CHARS),
            "n_context_pairs": n_pairs,
        }
        if isinstance(card, dict) and card.get("confidence"):
            base["voiceprint_confidence"] = card["confidence"]

        # STRUCTURAL decline: below the floor make NO generation call -- return
        # the labeled recent-replies pack instead.
        if n_pairs < floor:
            recent = _collect_pairs(
                con, _recent_chunk_ids(con, chat_ids, HYBRID_POOL), _EXEMPLAR_CAP)
            base.update({
                "confidence": "low",
                "draft": None,
                "exemplars": _public_exemplars(recent),
                "notes": [
                    f"Only {n_pairs} reply-in-context example(s) similar to the "
                    f"incoming message (< {floor} required to draft safely); no "
                    f"reply was generated. Returning your recent replies to "
                    f"{label} for reference."
                ],
            })
            return base

        # Confidence high: assemble the bounded prompt and generate IN TOOL,
        # inheriting CHAT_MODEL + the CHAT_THINK tri-state from chat_once.
        import ollama

        exemplars = _public_exemplars(pairs)
        reply = ollama.chat_once(_draft_prompt(card, exemplars, incoming, label))
        draft = (reply.get("content") or "").strip()

        clean, stripped = _strip_ungrounded_terms(con, draft, chat_ids, pairs, incoming)
        notes = []
        if stripped:
            notes.append(
                "stripped ungrounded inside-joke/nickname token(s) not found in "
                f"this dyad's terms or the exemplars: {stripped}"
            )
        base.update({
            "confidence": "high",
            "draft": clean,
            "exemplars": exemplars,
            "notes": notes,
        })
        return base
# --- end feature: persona ---
