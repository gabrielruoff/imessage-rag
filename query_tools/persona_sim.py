"""Persona simulation tool (V5 simulate_contact; RES-69).

Split out of persona.py: the highest-risk persona leg -- ESTIMATE how a
contact might reply in their voice, or talk to your past self as of a date.
The security boundary is the schema-registration gate (PERSONA_SIM_ENABLED,
withheld unless enabled). This module also enforces the deceased/grief
denylist, the soft dormant warning, and the acknowledged UX gate in code.
Reads base voiceprint helpers from .persona and shared retrieval/exemplar
helpers from .persona_draft.
"""

import contextlib
import datetime

import config
import consent
from db import tapback_name

from ._shared import (
    HYBRID_POOL,
    TEXT_PREVIEW_CHARS,
    _REAL_MESSAGE_CLAUSES,
    _connect,
    _normalize_date_bound,
)
from .persona import (
    _STYLE_FEATURES,
    _clip,
    _lookup,
    _resolve_person,
    collect_pairs as _collect_sim_pairs,
)
from .persona_draft import (
    _dyad_chats,
    _public_exemplars,
    _recent_chunk_ids,
    _similar_chunk_ids,
    _style_summary,
)


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


def _window_reply_pairs(con, chat_ids, date_from, date_to, reply_from_me, cap):
    """Directed reply pairs from a raw (date, id)-ordered scan of chat_ids within
    [date_from, date_to), independent of chunks/vectors -- the past-self path's
    bounded raw rescan, so it works even before embedding. Same adjacency rule as
    persona.directed_reply_pairs. Returns the MOST RECENT `cap` pairs (closest to the
    window's end), public-shaped (no private ids)."""
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

    # Deceased/grief HARD block. Matches the resolved member-key against the
    # denylist, which accepts bare contact ids (back-compat) AND explicit
    # contact:N / handle:N keys -- so a person who exists only as an unlinked
    # handle (no contacts row) can be protected too. Membership here == blocked.
    if consent.is_blocked_member(person_key, config.PERSONA_SIM_BLOCK_IDS):
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
                     subject="them", as_of_date=None, acknowledged=False) -> dict:
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
