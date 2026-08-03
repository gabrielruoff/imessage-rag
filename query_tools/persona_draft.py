"""Persona draft / style-match tools (V2 draft_reply, V3 check_style_match).

Split out of persona.py: in-tool reply drafting grounded only in real
reply-in-context exemplars, and the deterministic style-match report. Reads
the base voiceprint helpers and shared retrieval primitives from .persona.
See docs/DECISIONS.md (RES-62) for the grounding rationale.
"""

import contextlib
import re

import config
from function_words import FUNCTION_WORDS, TOKEN_RE, tokenize

from ._shared import HYBRID_POOL, RRF_K, TEXT_PREVIEW_CHARS, _connect
from .persona import (
    _FW_COLUMNS,
    _STYLE_FEATURES,
    _clip,
    _lookup,
    _resolve_person,
    collect_pairs as _collect_pairs,
    resolve_voice_scope as _resolve_voice_scope,
)


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


def _dyad_chats(con, person_key):
    """The 1:1 chats between me and this counterpart (direct_only), via the
    voiceprint scope helper -- the single home for a dyad's chat set, so the
    retrieval scope matches the voiceprint's exactly."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle

    return voiceprint.member_chats(con, person_key, direct_only=True)


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

    def _repl(match):
        # Normalize the curly apostrophe the same way tokenize() does, because
        # `strip` was built from tokenize() output: without it a draft's
        # "don’t" would never match the "don't" in the set. TOKEN_RE accepts
        # both apostrophes so the whole contraction is one match -- matching
        # only "don" and deleting it stranded the "’t".
        low = match.group(0).lower().replace("’", "'")
        if low in strip:
            stripped.append(low)
            return ""
        return match.group(0)

    clean = TOKEN_RE.sub(_repl, draft)
    clean = re.sub(r"\s+([,.!?;:])", r"\1", clean)   # tidy space before punctuation
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()  # collapse the gaps left behind
    return clean, sorted(set(stripped))


def draft_reply(incoming, contact_id=None, handle_id=None, name=None) -> dict:
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
                      subject="me", scope=None, chat_id=None) -> dict:
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

        member_key_str, counterpart_key, scope, err = _resolve_voice_scope(
            scope, subject, person_key, chat_id)
        if err:
            return err

        card = _lookup(con, member_key_str, counterpart_key, scope)
        if not isinstance(card, dict) or "features" not in card:
            return card  # the not-found / not-built note, returned verbatim

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
