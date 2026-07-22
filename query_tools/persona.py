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
import consent  # feature: persona (V5) -- shared opt-in id-list parser for PERSONA_SIM_BLOCK_IDS
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
    # HARD floor: below VOICEPRINT_MIN_RETURN_MESSAGES the member's own message
    # count is too small for the stylometry to mean anything, so decline the
    # card entirely (coverage note only) rather than emit precise-looking
    # numbers behind a soft 'low' flag -- mirrors draft_reply's structural
    # decline. Read at call time so tests/.env can move it.
    if (n_messages or 0) < config.VOICEPRINT_MIN_RETURN_MESSAGES:
        return {
            "member_key": member_key_str,
            "counterpart_key": counterpart_key,
            "scope": scope,
            "member": _label_for_key(con, member_key_str),
            "counterpart": _label_for_key(con, counterpart_key),
            "found": False,
            "note": (f"Too little interaction to characterize a voice "
                     f"({n_messages} messages; need "
                     f">= {config.VOICEPRINT_MIN_RETURN_MESSAGES}). Stylometry "
                     "omitted -- it would be noise at this volume."),
            "n_messages": n_messages,
            "n_reply_pairs": n_reply_pairs,
            "n_months": n_months,
        }
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

    # Never strip trivial short tokens (len < 3): a 1-2 char fragment like "s"
    # (from another dyad's tokenized term) is noise, and removing it strands
    # apostrophes/letters mid-word (the "It's okay" -> "It'" defect).
    strip = {t for t in (suspect - allowed) if len(t) >= 3}
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
        # DECLINE rather than emit a degenerate fragment: if the ungrounded-term
        # strip removed so much that too little of the draft survives (e.g. it
        # collapsed to "It'"), a broken stub is worse than no draft. Mirror the
        # below-floor structural decline instead.
        survived = len(clean) / max(1, len(draft))
        if draft and survived < config.DRAFT_MIN_SURVIVING_FRACTION:
            base.update({
                "confidence": "low",
                "draft": None,
                "exemplars": exemplars,
                "notes": [
                    f"The drafted reply leaned on inside-joke/nickname terms not "
                    f"grounded in your history with {label} ({stripped}); after "
                    f"removing them too little remained to send confidently, so no "
                    f"draft was produced. Use the examples above as a starting point."
                ],
            })
            return base
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


# V3: check_style_match -- turn a user-written DRAFT into a DETERMINISTIC style
# report against a target person_voiceprint row. No model is in the loop (the
# optional token-logprob score was cut as unverified): the draft's own features
# are computed with V1's EXACT definitions (reusing voiceprint._msg_style +
# function_words), then compared to the stored card. Distance is the mean
# absolute z-difference over the card's numeric features, EACH normalized by the
# corpus-wide std V1 already stored on the row (feature_std_json). The tool
# NEVER rescans the corpus, and features whose std was 0 (omitted from
# feature_std_json) are skipped -- no divide-by-zero, and a constant feature
# carries no matchable signal. The top-3 divergences get a RULE-BASED
# explanation (capitalization, punctuation, formality...), never an LLM critique.

# Per-feature rule-based framing: (human aspect, phrase when the draft sits
# HIGHER than the typical value, phrase when LOWER). Deterministic.
_FEATURE_ASPECT = {
    "avg_msg_len": ("message length", "longer messages", "shorter messages"),
    "lowercase_ratio": ("capitalization", "more lowercase text",
                        "more capitalized text"),
    "exclaim_rate": ("exclamation marks", "more exclamation marks",
                     "fewer exclamation marks"),
    "ellipsis_rate": ("ellipses", "more ellipses (...)", "fewer ellipses"),
    "burst_size": ("burst size", "more messages per burst",
                   "fewer messages per burst"),
    "fw_pronoun_rate": ("personal pronouns", "more personal pronouns",
                        "fewer personal pronouns"),
    "fw_ipron_rate": ("impersonal pronouns", "more impersonal pronouns",
                      "fewer impersonal pronouns"),
    "fw_article_rate": ("articles / formality", "more articles (a/an/the)",
                        "fewer articles (a/an/the)"),
    "fw_conj_rate": ("conjunctions", "more conjunctions", "fewer conjunctions"),
    "fw_prep_rate": ("prepositions / formality", "more prepositions",
                     "fewer prepositions"),
    "fw_aux_rate": ("auxiliary verbs / contractions", "more auxiliary verbs",
                    "fewer auxiliary verbs"),
    "fw_adverb_rate": ("adverbs / hedging", "more adverbs", "fewer adverbs"),
    "fw_negation_rate": ("negations", "more negations", "fewer negations"),
    "fw_quant_rate": ("quantifiers", "more quantifiers", "fewer quantifiers"),
}


def _draft_messages(draft):
    """Split a user-written draft into the 'messages' the voiceprint scan sees:
    one per non-empty line (a multi-text draft is several messages), falling
    back to the whole draft as a single message. Purely deterministic."""
    lines = [ln.strip() for ln in str(draft).splitlines()]
    lines = [ln for ln in lines if ln]
    return lines or [str(draft).strip()]


def _draft_features(draft):
    """The draft's numeric style + function-word features, computed with V1's
    EXACT definitions so each value is directly comparable to its
    person_voiceprint column: per-message style via voiceprint._msg_style, fw_*
    rates as category-hit / total-word counts (function_words), and burst_size
    as the message count (all draft lines are one same-sender burst)."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle
    from function_words import tokenize

    msgs = _draft_messages(draft)
    n = len(msgs)
    sum_len = n_lower = n_exclaim = n_ellipsis = 0
    sum_lower = 0.0
    fw_hits = {cat: 0 for cat in FUNCTION_WORDS}
    total_words = 0
    for text in msgs:
        ln, lower, has_exclaim, has_ellipsis = voiceprint._msg_style(text)
        sum_len += ln
        if lower is not None:
            sum_lower += lower
            n_lower += 1
        n_exclaim += has_exclaim
        n_ellipsis += has_ellipsis
        toks = tokenize(text)
        total_words += len(toks)
        for tok in toks:
            for cat in FUNCTION_WORDS:
                if tok in FUNCTION_WORDS[cat]:
                    fw_hits[cat] += 1

    feats = {
        "avg_msg_len": sum_len / n if n else 0.0,
        "lowercase_ratio": sum_lower / n_lower if n_lower else 0.0,
        "exclaim_rate": n_exclaim / n if n else 0.0,
        "ellipsis_rate": n_ellipsis / n if n else 0.0,
        "burst_size": float(n),  # every line is one same-sender burst
    }
    for cat in FUNCTION_WORDS:
        feats[f"fw_{cat}_rate"] = fw_hits[cat] / total_words if total_words else 0.0
    return feats


def _voice_label(card):
    """A readable name for whose voice the target card is, for the rule-based
    explanations ('your voice with Mom', "Al's voice")."""
    member = card.get("member") or "the target"
    counterpart = card.get("counterpart")
    base = "your voice" if member == "Me" else f"{member}'s voice"
    if card.get("scope") == "dyad" and counterpart:
        return f"{base} with {'you' if counterpart == 'Me' else counterpart}"
    return base


def _divergence_explanation(feature, card_val, draft_val, voice_label):
    """A deterministic, rule-based sentence for one diverging feature -- no LLM
    critique. Names the aspect and the direction of the gap."""
    aspect, higher, lower = _FEATURE_ASPECT.get(
        feature, (feature, f"a higher {feature}", f"a lower {feature}"))
    phrase = higher if draft_val > card_val else lower
    return (f"{aspect}: this draft uses {phrase} than {voice_label} "
            f"(draft {draft_val:.3g} vs typical {card_val:.3g})")


def _style_distance(card_features, draft_features, feature_std, voice_label):
    """(distance, divergences): the mean absolute z-difference over every
    numeric feature that has a NON-ZERO stored std, each gap normalized by that
    stored corpus std (feature_std) -- no rescan. Features absent from
    feature_std (std was 0, so omitted) are skipped. divergences are the
    per-feature records, sorted by z descending (stable: fixed feature order
    breaks ties). distance is None when nothing was comparable."""
    divergences = []
    for feature in _STYLE_FEATURES + _FW_COLUMNS:
        std = feature_std.get(feature)
        if not std:  # 0, None, or absent -> no signal / no divide-by-zero
            continue
        card_val = card_features.get(feature)
        draft_val = draft_features.get(feature)
        if card_val is None or draft_val is None:
            continue
        z = abs(card_val - draft_val) / std
        divergences.append({
            "feature": feature,
            "z": round(z, 4),
            "draft_value": round(draft_val, 6),
            "typical_value": round(card_val, 6),
            "corpus_std": std,
            "explanation": _divergence_explanation(
                feature, card_val, draft_val, voice_label),
        })
    divergences.sort(key=lambda d: d["z"], reverse=True)
    distance = (round(sum(d["z"] for d in divergences) / len(divergences), 4)
                if divergences else None)
    return distance, divergences


def check_style_match(draft, contact_id=None, handle_id=None, name=None,
                      subject="me", scope=None, chat_id=None):
    """Deterministically score how closely a user-written DRAFT matches a
    target voiceprint -- 'does this sound like me (or them)?'.

    Pass the `draft` text and name the person (contact_id/handle_id/name).
    Defaults to YOUR voice with that person (subject='me', scope='dyad');
    subject='them' checks whether the draft imitates the named person, and
    scope='all'/'group' target the overall/group voiceprint instead. The draft's
    own features are computed with the SAME definitions the voiceprint uses, then
    each numeric feature's gap is z-normalized by the corpus std stored on the
    row (feature_std) -- the corpus is NEVER rescanned, and features whose std
    was 0 are skipped. Returns {distance (mean absolute z-difference; 0 =
    identical voice, higher = more divergent), top_divergences (the 3 features
    the draft is furthest off on, each with a rule-based explanation),
    divergences (all compared features), draft_features}. No model in the loop."""
    if not draft or not str(draft).strip():
        return {"error": "check_style_match needs the draft text"}
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

        card = _lookup(con, member_key_str, counterpart_key, scope)
        if not isinstance(card, dict) or "features" not in card:
            return card  # the not-found / not-built note, returned verbatim

        from ._shared import TEXT_PREVIEW_CHARS

        feature_std = card.get("feature_std") or {}
        draft_features = _draft_features(draft)
        distance, divergences = _style_distance(
            card["features"], draft_features, feature_std, _voice_label(card))

        result = {
            "member_key": card["member_key"],
            "counterpart_key": card["counterpart_key"],
            "scope": card["scope"],
            "member": card["member"],
            "counterpart": card["counterpart"],
            "voiceprint_confidence": card.get("confidence"),
            "draft": _clip(str(draft), TEXT_PREVIEW_CHARS),
            "draft_messages": len(_draft_messages(draft)),
            "distance": distance,
            "n_features_compared": len(divergences),
            "top_divergences": divergences[:3],
            "divergences": divergences,
            "draft_features": {k: round(v, 6) for k, v in draft_features.items()},
        }
        if distance is None:
            result["note"] = (
                "No per-feature corpus std on this voiceprint (feature_std_json "
                "empty -- corpus too uniform, or a single-row scope), so a "
                "deterministic style distance could not be computed.")
        else:
            result["note"] = (
                "distance is the mean absolute z-difference across the compared "
                "features (0 = identical to the stored voice, higher = more "
                "divergent); each gap is normalized by the corpus std stored on "
                "the voiceprint row -- no rescan, and std==0 features are skipped.")
        return result
# --- end feature: persona ---


# --- feature: persona (V5 simulate_contact; RES-69) --------------------------
# The highest-risk persona leg: ESTIMATE how a contact might reply in their voice,
# or talk to your PAST SELF as of a date. Structural guardrails (RES-69):
#   * Registration gate (the SECURITY boundary): the schema is withheld unless
#     config.PERSONA_SIM_ENABLED. When off the model never sees the tool -- the
#     repo's one reliable stop. This function is ALWAYS importable; only its
#     schema (tool_schemas / chat SYSTEM_PROMPT) is conditional.
#   * In-tool generation (same architecture as V2 draft_reply): simulate_contact
#     calls ollama.chat_once ITSELF and returns text it has ALREADY wrapped -- a
#     per-line "estimate, not <person>" label applied IN CODE, post-generation,
#     before the payload leaves the tool. The outer chat loop only ever relays a
#     pre-labeled artifact; there is no unlabeled version in the conversation for
#     it to strip.
#   * Deceased/grief HARD block: config.PERSONA_SIM_BLOCK_IDS hard-refuses in code
#     (fixed compassionate message, no generation). A soft last-message-age +
#     one-sided heuristic WARNS (never blocks) before simulating a likely-dormant
#     contact -- a fail-open net over the manual list.
#   * Living-contact UX: acknowledged=true is required before any generation (a
#     documented UX flow, NOT the security boundary -- the registration gate is).
#   * Past-self: a bounded date-scoped raw rescan, gated at
#     VOICEPRINT_MIN_RANGE_MESSAGES per range and capped at
#     VOICEPRINT_SCAN_MAX_MESSAGES.
# Nothing here auto-sends (the app has no send path) and nothing is persisted.

_SIM_EXEMPLAR_CAP = 8   # exemplar-bundle ceiling (same as V2's _EXEMPLAR_CAP)

# Fixed, code-emitted refusal for a denylisted contact -- constant on purpose
# (never LLM-authored, never persisted): a block only ever returns THIS string.
_SIM_BLOCK_MESSAGE = (
    "You've marked this person as someone not to simulate, so this feature is "
    "turned off for them. If they've passed away, I'm sorry for your loss -- and "
    "estimating their voice isn't something I'll do here."
)


def _directed_reply_pairs(con, chunk_id, reply_from_me):
    """reply_pairs_in_chunk generalized to EITHER direction: each real message
    immediately followed (in (date, id) order over REAL rows -- emote/system rows
    can neither anchor nor serve) by a real message from the requested speaker
    becomes an (incoming -> reply) pair. reply_from_me selects WHOSE replies to
    collect: 1 = the user's (past-self grounding), 0 = the counterpart's
    (simulating a contact). Textless (attachment-only) rows on either side are
    skipped. This is the same load-bearing adjacency rule as V2's
    reply_pairs_in_chunk, just parameterized by direction."""
    from ._shared import TEXT_PREVIEW_CHARS, _REAL_MESSAGE_CLAUSES

    real = " AND ".join(_REAL_MESSAGE_CLAUSES)
    rows = con.execute(
        f"SELECT m.id, m.is_from_me, m.text, m.date FROM messages m "
        f"WHERE m.chunk_id = ? AND {real} ORDER BY m.date, m.id",
        (chunk_id,),
    ).fetchall()
    want = 1 if reply_from_me else 0
    pairs = []
    for i in range(len(rows) - 1):
        cur_id, cur_me, cur_text = rows[i][0], rows[i][1], rows[i][2]
        nxt_id, nxt_me, nxt_text = rows[i + 1][0], rows[i + 1][1], rows[i + 1][2]
        if cur_me == (1 - want) and nxt_me == want and cur_text and nxt_text:
            pairs.append({
                "incoming": _clip(cur_text, TEXT_PREVIEW_CHARS),
                "reply": _clip(nxt_text, TEXT_PREVIEW_CHARS),
                "date": rows[i + 1][3],
                "incoming_id": cur_id,
                "reply_id": nxt_id,
            })
    return pairs


def _collect_sim_pairs(con, chunk_ids, cap, reply_from_me):
    """Directed reply pairs across chunk_ids in order, deduped by
    (incoming_id, reply_id), capped at `cap`. The simulate-a-contact analogue of
    _collect_pairs, but for the counterpart's replies (reply_from_me=0)."""
    pairs, seen = [], set()
    for cid in chunk_ids:
        for pair in _directed_reply_pairs(con, cid, reply_from_me):
            key = (pair["incoming_id"], pair["reply_id"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
            if len(pairs) >= cap:
                return pairs
    return pairs


def _window_reply_pairs(con, chat_ids, date_from, date_to, reply_from_me, cap):
    """Directed reply pairs from a raw (date, id)-ordered scan of chat_ids within
    [date_from, date_to), independent of chunks/vectors -- the past-self path's
    bounded raw rescan, so it works even before embedding. Same adjacency rule as
    _directed_reply_pairs. Returns the MOST RECENT `cap` pairs (closest to the
    window's end), public-shaped (no private ids)."""
    from ._shared import TEXT_PREVIEW_CHARS, _REAL_MESSAGE_CLAUSES

    real = " AND ".join(_REAL_MESSAGE_CLAUSES)
    want = 1 if reply_from_me else 0
    collected = []
    for cid in chat_ids:
        clauses = ["m.chat_id = ?", "m.date IS NOT NULL", real]
        params = [cid]
        if date_from is not None:
            clauses.append("m.date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("m.date < ?")
            params.append(date_to)
        rows = con.execute(
            f"SELECT m.is_from_me, m.text, m.date FROM messages m "
            f"WHERE {' AND '.join(clauses)} ORDER BY m.date, m.id", params
        ).fetchall()
        for i in range(len(rows) - 1):
            cur_me, cur_text = rows[i][0], rows[i][1]
            nxt_me, nxt_text, nxt_date = rows[i + 1][0], rows[i + 1][1], rows[i + 1][2]
            if cur_me == (1 - want) and nxt_me == want and cur_text and nxt_text:
                collected.append({
                    "incoming": _clip(cur_text, TEXT_PREVIEW_CHARS),
                    "reply": _clip(nxt_text, TEXT_PREVIEW_CHARS),
                    "date": nxt_date,
                })
    collected.sort(key=lambda p: p["date"], reverse=True)
    return collected[:cap]


def _wrap_simulation(text, label):
    """Prefix EVERY non-empty line of the generated text with `label`, in code,
    post-generation. The outer model only ever sees the pre-labeled artifact, so
    there is no unlabeled draft in the conversation for it to strip. Blank lines
    are preserved untouched."""
    lines = (text or "").split("\n")
    return "\n".join(f"[{label}] {ln}" if ln.strip() else ln for ln in lines)


def _sim_prompt(style, exemplars, message, persona_label, reply_tag):
    """The bounded in-tool generation prompt: the target's style + REAL exchanges
    (a message -> how the target actually replied) + the incoming message +
    estimate-only / no-invention rules. All grounding is inline; nothing else is
    in the model's context."""
    system = (
        f"You ESTIMATE how {persona_label} might reply to a text message, in "
        "their voice, from a style profile and REAL past exchanges (the other "
        "side's message, then how they actually replied). This is a STATISTICAL "
        "ESTIMATE of their style, NOT their real words and NOT a prediction of "
        "what they would actually say. Match that voice: length, casing, "
        "punctuation, and phrasing. Rules: reply with ONLY the message text, "
        "nothing else -- no preamble, no quotes, no explanation. Invent NO facts, "
        "plans, names, or events; use only what appears in the exchanges or the "
        "incoming message."
    )
    parts = []
    if style:
        parts.append(f"{persona_label} STYLE:\n{style}")
    parts.append(f"PAST EXCHANGES (a message -> how {persona_label} actually replied):")
    for i, ex in enumerate(exemplars, 1):
        parts.append(f"{i}. IN: {ex['incoming']}\n   {reply_tag}: {ex['reply']}")
    parts.append(f"MESSAGE to reply to:\n{message}")
    parts.append(f"Write {persona_label}'s estimated reply now (message text only):")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _person_handle_ids(con, person_key):
    """The handle ids behind a (non-me) member_key -- reuses voiceprint's single
    home for that translation so a contact's duplicate handles collapse exactly
    as everywhere else."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle

    return voiceprint._member_handle_ids(con, person_key) or []


def _predict_tapback(con, chat_ids, handle_ids):
    """The contact's likely tapback, from a FULL-MESSAGE base rate (RES-69): each
    tapback type the contact SENT in the dyad divided by ALL real messages
    received FROM that contact -- NOT divided by only the messages that drew a
    reaction (that denominator conditions on a reaction and overstates every
    rate). Returns the modal tapback + per-type base rates, clearly an estimate,
    or None when there is nothing to normalize."""
    from db import tapback_name

    from ._shared import _REAL_MESSAGE_CLAUSES

    if not chat_ids or not handle_ids:
        return None
    real = " AND ".join(_REAL_MESSAGE_CLAUSES)
    cph = ",".join("?" * len(chat_ids))
    hph = ",".join("?" * len(handle_ids))
    scope_params = list(chat_ids) + list(handle_ids)
    denom = con.execute(
        f"SELECT COUNT(*) FROM messages m WHERE m.chat_id IN ({cph}) "
        f"AND m.is_from_me = 0 AND m.handle_id IN ({hph}) AND {real}",
        scope_params,
    ).fetchone()[0]
    if not denom:
        return None
    rows = con.execute(
        f"SELECT associated_message_type, COUNT(*) FROM reactions "
        f"WHERE chat_id IN ({cph}) AND is_from_me = 0 AND handle_id IN ({hph}) "
        f"AND associated_message_type BETWEEN 2000 AND 2999 "
        f"GROUP BY associated_message_type",
        scope_params,
    ).fetchall()
    base = {}
    for code, n in rows:
        base[tapback_name(code)] = round(n / denom, 4)
    if not base:
        return {"denominator_messages": denom, "base_rates": {},
                "most_likely": None,
                "note": "no tapback history from this contact -- they rarely react."}
    base = dict(sorted(base.items(), key=lambda kv: kv[1], reverse=True))
    return {"denominator_messages": denom, "base_rates": base,
            "most_likely": next(iter(base)),
            "note": "base rate = tapbacks the contact sent / all real messages "
                    "they sent (a full-message denominator, not reacted-only); a "
                    "rough estimate of their reaction habit, not a prediction."}


def _dormant_signal(con, chat_ids, handle_ids):
    """Soft, FAIL-OPEN deceased/dormant heuristic: fires when the contact's most
    recent message to the user is older than PERSONA_SIM_DORMANT_DAYS AND the user
    has since sent >= PERSONA_SIM_ONESIDED_MIN_MSGS messages (texting into
    silence). A WARNING, never a block -- it cannot tell deceased from long-lost.
    Returns an evidence dict when it fires, else None. Uses the corpus's latest
    message as 'now' (the data has no real clock) and plain-string date compares
    only (never a SQL date function -- the load-bearing date rule)."""
    import datetime

    from ._shared import _REAL_MESSAGE_CLAUSES

    if not chat_ids or not handle_ids:
        return None
    real = " AND ".join(_REAL_MESSAGE_CLAUSES)
    cph = ",".join("?" * len(chat_ids))
    hph = ",".join("?" * len(handle_ids))
    last_them = con.execute(
        f"SELECT MAX(m.date) FROM messages m WHERE m.chat_id IN ({cph}) "
        f"AND m.is_from_me = 0 AND m.handle_id IN ({hph}) AND {real}",
        list(chat_ids) + list(handle_ids),
    ).fetchone()[0]
    corpus_now = con.execute("SELECT MAX(date) FROM messages").fetchone()[0]
    if last_them is None or corpus_now is None:
        return None
    my_after = con.execute(
        f"SELECT COUNT(*) FROM messages m WHERE m.chat_id IN ({cph}) "
        f"AND m.is_from_me = 1 AND m.date > ? AND {real}",
        list(chat_ids) + [last_them],
    ).fetchone()[0]
    days_since = (datetime.date.fromisoformat(corpus_now[:10])
                  - datetime.date.fromisoformat(last_them[:10])).days
    if (days_since >= config.PERSONA_SIM_DORMANT_DAYS
            and my_after >= config.PERSONA_SIM_ONESIDED_MIN_MSGS):
        return {
            "last_message_from_them": last_them[:10],
            "days_since_their_last_message": days_since,
            "your_messages_since_with_no_reply": my_after,
            "warning": ("This contact went quiet long ago and you've kept "
                        "messaging with no reply since. If they've passed away, "
                        "consider whether estimating their voice is something you "
                        "want right now. Add them to PERSONA_SIM_BLOCK_IDS to stop "
                        "offering this."),
        }
    return None


def _simulate_them(con, message, contact_id, handle_id, name, acknowledged):
    """Simulate a named contact's likely reply. Block-list hard-refuse, then the
    acknowledged UX gate (carrying any soft dormant warning), then -- only once
    acknowledged -- topical exemplar retrieval + in-tool generation, wrapped."""
    person_key, label, err = _resolve_person(con, contact_id, handle_id, name)
    if err:
        return err
    if person_key is None:
        return {"error": "simulate_contact needs a person (contact_id/handle_id/name)"}

    # Deceased/grief HARD block. Only a resolved contact id can be on the list
    # (a bare handle can't be marked); consent membership here means "blocked".
    kind, _, raw = person_key.partition(":")
    blocked_id = int(raw) if kind == "contact" else None
    if blocked_id is not None and consent.is_consented(
            blocked_id, config.PERSONA_SIM_BLOCK_IDS):
        return {"refused": True, "reason": "blocked_contact", "generated": False,
                "person": label, "message": _SIM_BLOCK_MESSAGE}

    chat_ids = _dyad_chats(con, person_key)
    if not chat_ids:
        return {"error": f"no 1:1 chat history with {label} to simulate from",
                "counterpart_key": person_key, "counterpart": label}
    handle_ids = _person_handle_ids(con, person_key)
    dormant = _dormant_signal(con, chat_ids, handle_ids)

    # Living-contact UX gate: acknowledged=true required before ANY generation (a
    # documented UX flow, NOT the boundary -- the registration gate is). The soft
    # dormant/deceased warning rides in the same confirmation payload so the user
    # sees it BEFORE acknowledging.
    if not acknowledged:
        out = {
            "acknowledgment_required": True, "generated": False,
            "person": label, "counterpart_key": person_key,
            "instruction": (
                "Simulating a real person produces a STATISTICAL ESTIMATE of "
                "their texting style, never their real words. Confirm the user "
                "understands this"
                + (" AND relay the deceased_warning below" if dormant else "")
                + ", then call simulate_contact again with acknowledged=true."),
        }
        if dormant:
            out["deceased_warning"] = dormant
        return out

    from ._shared import HYBRID_POOL, TEXT_PREVIEW_CHARS

    base = {"member_key": person_key, "counterpart_key": "me", "person": label,
            "subject": "them", "message": _clip(message, TEXT_PREVIEW_CHARS),
            "generated": False}
    tapback = _predict_tapback(con, chat_ids, handle_ids)
    if tapback is not None:
        base["likely_tapback"] = tapback
    if dormant:
        base["deceased_warning"] = dormant

    # Topical grounding: dyad chunks similar to the incoming, then how the CONTACT
    # actually replied there (reply_from_me=0). Structural decline below the floor
    # -- no generation call at all.
    similar = _similar_chunk_ids(con, message, chat_ids, HYBRID_POOL)
    pairs = _collect_sim_pairs(con, similar, _SIM_EXEMPLAR_CAP, reply_from_me=0)
    base["n_context_pairs"] = len(pairs)
    floor = config.VOICEPRINT_MIN_CONTEXT_PAIRS
    if len(pairs) < floor:
        recent = _collect_sim_pairs(
            con, _recent_chunk_ids(con, chat_ids, HYBRID_POOL),
            _SIM_EXEMPLAR_CAP, reply_from_me=0)
        base.update({
            "confidence": "low", "simulation": None,
            "exemplars": _public_exemplars(recent),
            "notes": [f"Only {len(pairs)} example(s) of how {label} replies to "
                      f"similar messages (< {floor} required to simulate); no "
                      f"estimate was generated. Returning recent examples of how "
                      f"{label} texts for reference."],
        })
        return base

    import ollama

    card = _lookup(con, person_key, "me", "dyad")
    style = _style_summary(card) if isinstance(card, dict) else ""
    exemplars = _public_exemplars(pairs)
    reply = ollama.chat_once(_sim_prompt(style, exemplars, message, label, label))
    raw_text = (reply.get("content") or "").strip()
    if not raw_text:
        # Empty generation: decline honestly rather than relay a blank estimate.
        base.update({
            "confidence": "low", "simulation": None, "exemplars": exemplars,
            "notes": [f"The model returned no text for {label}; no estimate was "
                      "generated."],
        })
        return base
    label_tag = f"ESTIMATE of {label}'s voice — not their real words"
    base.update({
        "confidence": "high", "generated": True,
        "simulation": _wrap_simulation(raw_text, label_tag),
        "disclaimer": (f"This is a statistical estimate of how {label} texts, "
                       "generated from their style and real past replies. It is "
                       "NOT their actual words and not a prediction of what they "
                       "would really say."),
        "exemplars": exemplars,
    })
    return base


def _simulate_past_self(con, message, contact_id, handle_id, name, as_of_date):
    """Talk to your PAST SELF as of a date via a bounded date-scoped raw rescan,
    gated at VOICEPRINT_MIN_RANGE_MESSAGES per range and capped at
    VOICEPRINT_SCAN_MAX_MESSAGES. Optionally scoped to a person (your past voice
    WITH them); otherwise your overall past voice."""
    import voiceprint

    if not as_of_date or not str(as_of_date).strip():
        return {"error": "past-self simulation (subject='me') needs as_of_date"}
    person_key, label, err = _resolve_person(con, contact_id, handle_id, name)
    if err:
        return err

    dt = _normalize_date_bound(as_of_date, upper=True)
    if person_key is not None:
        chats = _dyad_chats(con, person_key)
        scope_label = f"with {label}"
    else:
        chats = voiceprint.member_chats(con, "me")
        scope_label = "overall"
    if not chats:
        return {"error": "no chat history to simulate your past self from"}

    cap = config.VOICEPRINT_SCAN_MAX_MESSAGES
    scan_n = voiceprint.scan_size(con, chats, None, dt)
    if scan_n > cap:
        return {"note": f"Too much history up to {as_of_date}: {scan_n} messages "
                        f"exceed the {cap}-message scan cap -- narrow the window "
                        "(pass a later as_of_date).",
                "subject": "me", "as_of_date": as_of_date, "scope": scope_label,
                "messages_in_range": scan_n, "generated": False}

    result = voiceprint.compute_voiceprint(con, "me", chats, date_to=dt, budget=cap)
    n = result["n_messages"] if result else 0
    if result is None or n < config.VOICEPRINT_MIN_RANGE_MESSAGES:
        return {"note": f"Only {n} of your messages on record up to {as_of_date} "
                        f"(< {config.VOICEPRINT_MIN_RANGE_MESSAGES}); too little to "
                        "simulate your past self reliably.",
                "subject": "me", "as_of_date": as_of_date, "scope": scope_label,
                "n_messages": n, "generated": False}

    from ._shared import TEXT_PREVIEW_CHARS

    # Grounding: how YOU actually replied in the window (reply_from_me=1).
    pairs = _window_reply_pairs(con, chats, None, dt, reply_from_me=1,
                               cap=_SIM_EXEMPLAR_CAP)
    base = {"subject": "me", "as_of_date": as_of_date, "scope": scope_label,
            "message": _clip(message, TEXT_PREVIEW_CHARS),
            "n_messages": n, "n_context_pairs": len(pairs), "generated": False}
    floor = config.VOICEPRINT_MIN_CONTEXT_PAIRS
    if len(pairs) < floor:
        base.update({
            "confidence": "low", "simulation": None, "exemplars": pairs,
            "notes": [f"Only {len(pairs)} reply example(s) up to {as_of_date} "
                      f"(< {floor} required to simulate); no estimate generated."],
        })
        return base

    import ollama

    feats = {f: result[f] for f in _STYLE_FEATURES}
    feats.update(result["fw"])
    card = {"features": feats, "openers": result["openers"],
            "closers": result["closers"]}
    style = _style_summary(card)
    persona_label = f"your past self (as of {as_of_date})"
    reply = ollama.chat_once(_sim_prompt(style, pairs, message, persona_label, "YOU"))
    raw_text = (reply.get("content") or "").strip()
    if not raw_text:
        # Empty generation: decline honestly rather than relay a blank estimate.
        base.update({
            "confidence": "low", "simulation": None, "exemplars": pairs,
            "notes": [f"The model returned no text up to {as_of_date}; no estimate "
                      "was generated."],
        })
        return base
    label_tag = f"ESTIMATE of your past self as of {as_of_date} — not a real message"
    base.update({
        "confidence": "high", "generated": True,
        "simulation": _wrap_simulation(raw_text, label_tag),
        "disclaimer": (f"This is a statistical estimate of how you texted around "
                       f"{as_of_date}, generated from your real messages then. It "
                       "is not a real message and not a memory."),
        "exemplars": pairs,
    })
    return base


def simulate_contact(message, contact_id=None, handle_id=None, name=None,
                     subject="them", as_of_date=None, acknowledged=False):
    """ESTIMATE how a contact might reply in their voice, or talk to your PAST
    SELF as of a date. In-tool generation: the returned `simulation` is ALREADY
    wrapped in code (every line labeled an estimate, post-generation), so the
    outer chat loop only ever relays a pre-labeled artifact -- never their real
    words.

    subject='them' (default) simulates the named person (contact_id preferred,
    else handle_id/name): a deny-listed contact (PERSONA_SIM_BLOCK_IDS) is
    hard-refused in code, and the living-contact path requires acknowledged=true
    (a UX flow -- confirm with the user first). A soft last-message-age +
    one-sided heuristic surfaces a deceased_warning to relay before acknowledging;
    it warns, never blocks. The payload also carries a reaction base-rate tapback
    prediction. Below VOICEPRINT_MIN_CONTEXT_PAIRS similar reply examples it
    DECLINES structurally (no generation), returning recent examples instead.

    subject='me' + as_of_date talks to your past self via a bounded date-scoped
    raw rescan, gated at VOICEPRINT_MIN_RANGE_MESSAGES per range and capped at
    VOICEPRINT_SCAN_MAX_MESSAGES; name a person to scope it to your past voice
    WITH them. 1:1 only; nothing auto-sends and nothing is persisted."""
    if not message or not str(message).strip():
        return {"error": "simulate_contact needs the message text to respond to"}
    message = str(message)
    subject = (subject or "them").lower()
    if subject not in ("them", "me"):
        return {"error": "subject must be 'them' (a contact) or 'me' (your past self)"}

    with contextlib.closing(_connect()) as con:
        if subject == "me":
            return _simulate_past_self(con, message, contact_id, handle_id, name,
                                       as_of_date)
        return _simulate_them(con, message, contact_id, handle_id, name, acknowledged)
# --- end feature: persona (V5 simulate_contact) ---
