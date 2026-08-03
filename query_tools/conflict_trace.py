"""trace_conflict_across_chats (RES-52 / C3): the gated cross-chat conflict
trace -- "did this one argument spill into my OTHER conversations?".

Split out of query_tools/conflict.py, which re-exports it so
`query_tools.conflict.trace_conflict_across_chats` keeps resolving.

Chunks in a DIFFERENT chat, within a few days of the episode, that are BOTH
semantically close to it AND literally name chat A's partner (their handle,
first name, or one of the chat's distinctive dyad_terms). Two STRUCTURAL gates
keep it safe:
  1. Registration gate (the real boundary): the schema reaches chat.py's tool
     list ONLY when config.CONFLICT_TRACE_ENABLED. Off by default -- withholding
     the schema is the repo's one reliable way to stop the model calling a tool,
     so a disabled deployment needs no prompt-level "don't trace" text.
  2. Per-episode consent (UX ONLY, NOT the boundary): without consent=True the
     tool returns a fixed ask-the-user payload and reads no data.
The name-reference gate filters topically-similar-but-unnamed chunks (a
"weekend with my boss" chunk near a "weekend with Jordan" episode is dropped
unless it actually names Jordan). Computed on demand and written to NO table:
nothing here is persisted.
"""

import contextlib
import re

import config
from db import iso_to_day

from ._shared import CHUNK_PREVIEW_CHARS, MAX_VEC_K, _connect
# Bottom-imported by .conflict, so its core names are already bound here.
from .conflict import _has_conflict_tables
from .conflict_dv import _window_message_texts

# --- feature: conflict ---
#
# trace_conflict_across_chats(conflict_id, k, consent) answers "did this one
# argument spill into my OTHER conversations?" -- chunks in a DIFFERENT chat,
# within a few days of the episode, that are BOTH semantically close to it AND
# literally name chat A's partner (their handle, first name, or one of the
# chat's distinctive dyad_terms). Two STRUCTURAL gates keep it safe:
#   1. Registration gate (the real boundary): the schema is registered in
#      chat.py's tool list ONLY when config.CONFLICT_TRACE_ENABLED. Off by
#      default -- withholding the schema is the repo's one reliable way to stop
#      the model calling a tool (see the CLAUDE.md gotcha), so a disabled
#      deployment never needs prompt-level "don't trace" text.
#   2. Per-episode consent (UX ONLY, NOT the boundary): without consent=True the
#      tool returns a fixed ask-the-user payload and reads no data.
# The name-reference gate filters topically-similar-but-unnamed chunks (a
# "weekend with my boss" chunk near a "weekend with Jordan" episode is dropped
# unless it actually names Jordan). Computed on demand and written to NO table
# (there is deliberately no conflict_links table): nothing here is persisted.



# Shortest token that counts as a literal partner reference -- a 1-2 char name
# or dyad_term ("i", "ok") would match half the corpus, defeating the gate.
_TRACE_MIN_REF_LEN = 3
# Query-text budget fed to the embedder (the episode's own window text).
_TRACE_QUERY_CHARS = 4000
# Over-fetch factor: the sim floor + name gate prune hard, so pull a wider KNN
# pool than k before filtering (still capped at MAX_VEC_K).
_TRACE_POOL_FACTOR = 5


def _trace_ask_payload(conflict_id):
    """The fixed no-data response returned when consent was NOT given: it
    carries zero retrieved content -- only the instruction to ask first. This
    is the UX layer, not the security boundary (registration is)."""
    return {
        "conflict_id": conflict_id,
        "status": "consent_required",
        "links": [],
        "ask": ("Tracing whether this argument spilled into your OTHER "
                "conversations reads across chats. Want me to check? I'll look "
                "only at messages that both match this episode AND name the "
                "person involved, and I won't save anything."),
        "note": ("no data was read -- call again with consent=true only after "
                 "the user agrees; nothing is ever persisted"),
    }


def _chat_partner_references(con, chat_id):
    """The literal strings a related chunk must contain to count as naming chat
    A's partner: each participant handle_identifier, the counterparty contact
    first_name(s), and chat A's distinctive dyad_terms. Returns an ordered,
    de-duplicated list of (kind, lowercased_value); blank/too-short values are
    dropped so the gate stays meaningful."""
    refs = []
    for hid, first_name in con.execute(
        "SELECT h.handle_identifier, c.first_name "
        "FROM chat_participants cp "
        "JOIN handles h ON h.id = cp.handle_id "
        "LEFT JOIN contacts c ON c.id = h.contact_id "
        "WHERE cp.chat_id = ?", (chat_id,),
    ):
        if hid and len(hid) >= _TRACE_MIN_REF_LEN:
            refs.append(("handle", hid.lower()))
        if first_name and len(first_name) >= _TRACE_MIN_REF_LEN:
            refs.append(("name", first_name.lower()))
    for (term,) in con.execute(
        "SELECT term FROM dyad_terms WHERE chat_id = ?", (chat_id,),
    ):
        if term and len(term) >= _TRACE_MIN_REF_LEN:
            refs.append(("dyad_term", term.lower()))
    seen, out = set(), []
    for kind, value in refs:
        if value not in seen:
            seen.add(value)
            out.append((kind, value))
    return out


def _mentions_reference(text, references):
    """The first partner reference `text` literally mentions ({kind, value}) or
    None. Handle identifiers (phones/emails) match as substrings; names and
    dyad_terms match on WORD BOUNDARIES, so 'Jordan' fires on 'weekend with
    Jordan' but not on 'Jordania', and a topically-similar chunk that names no
    one is filtered out."""
    if not text:
        return None
    lowered = text.lower()
    for kind, value in references:
        if kind == "handle":
            if value in lowered:
                return {"kind": kind, "value": value}
        elif re.search(rf"\b{re.escape(value)}\b", lowered):
            return {"kind": kind, "value": value}
    return None


def trace_conflict_across_chats(conflict_id, k=10, consent=False) -> dict:
    """Trace whether ONE conflict episode (a conflict_id from find_conflicts)
    spilled into your OTHER conversations: conversation chunks in a DIFFERENT
    chat, within CONFLICT_TRACE_BUFFER_DAYS of the episode, that are BOTH
    semantically close to it AND literally name chat A's partner (their handle,
    first name, or one of the chat's distinctive terms). A link needs cosine
    similarity > CONFLICT_TRACE_SIM_THRESHOLD AND a name hit -- topically
    similar but unnamed chunks are dropped.

    Cross-chat by nature, so it is off by default and consent-gated. The real
    off-switch is REGISTRATION: the whole tool is withheld from the model unless
    config.CONFLICT_TRACE_ENABLED. The `consent` param is a UX prompt only, NOT
    the security boundary -- without consent=True this returns a fixed
    ask-the-user payload and reads no data. Computed on demand; NOTHING is
    persisted (there is deliberately no conflict_links table)."""
    conflict_id = int(conflict_id)
    if not consent:
        return _trace_ask_payload(conflict_id)

    # Local import keeps this additive block self-contained and cycle-safe (the
    # package is fully initialized by the time a tool is called).
    from .chunks import _chunk_knn, _embed_query

    k = max(1, min(int(k), MAX_VEC_K))
    buffer = max(0, int(config.CONFLICT_TRACE_BUFFER_DAYS))
    threshold = float(config.CONFLICT_TRACE_SIM_THRESHOLD)

    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"conflict_id": conflict_id, "links": [],
                    "error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT chat_id, first_date, last_date, topic FROM conflicts "
            "WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            return {"conflict_id": conflict_id, "links": [],
                    "error": f"no conflict with id {conflict_id}"}
        origin_chat = row["chat_id"]
        first_date, last_date = row["first_date"], row["last_date"]
        # Convert the episode's UTC-ISO message dates to vec0 day numbers with
        # db.iso_to_day -- the SAME converter the metadata writer uses; mixing
        # formats silently empties the scan (see the CLAUDE.md gotcha).
        from_day = iso_to_day(first_date) - buffer
        to_day = iso_to_day(last_date) + buffer

        references = _chat_partner_references(con, origin_chat)
        # Query text = the episode's own full window text (same source the DV
        # screen reads), falling back to its topic when the window has no text.
        window_texts = _window_message_texts(
            con, conflict_id, origin_chat, first_date, last_date)
        query_text = " ".join(t for t in window_texts if t)[:_TRACE_QUERY_CHARS]
        if not query_text.strip():
            query_text = (row["topic"] or "")
        if not query_text.strip():
            return {"conflict_id": conflict_id, "origin_chat_id": origin_chat,
                    "links": [], "returned": 0,
                    "note": "no episode text to trace; nothing persisted"}

        query_vec = _embed_query(query_text)
        pool = max(1, min(k * _TRACE_POOL_FACTOR, MAX_VEC_K))
        candidates = _chunk_knn(
            con, query_vec, pool, chat_ids=None, from_day=from_day, to_day=to_day,
            exclude_chat_ids=[origin_chat])

    links = []
    for r in candidates:
        distance = r.get("distance")
        if distance is None:
            continue
        similarity = 1.0 - float(distance)
        if similarity <= threshold:
            continue
        matched = _mentions_reference(r.get("text") or "", references)
        if matched is None:
            continue
        links.append({
            "chunk_id": r["chunk_id"],
            "chat_id": r["chat_id"],
            "participants": r.get("participants"),
            "first_date": r["first_date"],
            "last_date": r["last_date"],
            "similarity": round(similarity, 4),
            "matched_reference": matched,
            "text": (r.get("text") or "")[:CHUNK_PREVIEW_CHARS],
        })
        if len(links) >= k:
            break

    return {
        "conflict_id": conflict_id,
        "origin_chat_id": origin_chat,
        "window": {"first_date": first_date, "last_date": last_date,
                   "buffer_days": buffer},
        "sim_threshold": threshold,
        "n_references": len(references),
        "links": links,
        "returned": len(links),
        "note": ("cross-chat spillover -- each link BOTH exceeds the similarity "
                 "threshold AND literally names the partner; computed on demand "
                 "and NOT persisted (no conflict_links table exists)"),
    }
# --- end feature: conflict ---


