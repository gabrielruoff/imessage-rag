"""S9 recipes: retrieval & directory plumbing (handle resolution, chat
listing, message reactions, full-text recovery)."""

from ._base import _qt, _v, recipe


@recipe("handle_owner")
def handle_owner(roster, handle):
    """search_contact_by_handle's contract: a bare phone/email resolves to a
    named person plus every other handle that person owns."""
    qt = _qt()
    r = qt.search_contact_by_handle(handle_identifier=handle)
    if not r or not r.get("contact"):
        raise ValueError(f"handle {handle!r} resolves to no contact")
    top1 = roster["TOP1"]
    if r["contact"]["id"] != top1["contact_id"]:
        raise ValueError(
            f"handle {handle!r} resolves to contact {r['contact']['id']}, not TOP1 "
            f"({top1['contact_id']}) -- the roster's HANDLE is meant to be TOP1's"
        )
    # One identifier can own several handle ROWS -- chat.db files the same phone
    # number once per service (iMessage/iMessageLite/SMS/RCS). A reply counts the
    # phone numbers and emails it can see, so GT counts distinct identifiers; the
    # row count stays in raw as advisory context.
    identifiers = {r["handle"]["handle_identifier"]} | {
        h["handle_identifier"] for h in r["other_handles"]
    }
    return {
        "values": {
            "owner": _v(top1["display_name"], "exact"),
            "n_handles": _v(len(identifiers), "exact"),
        },
        "raw": {**r, "n_handle_rows": 1 + len(r["other_handles"])},
    }


@recipe("chats_for_contact")
def chats_for_contact(roster, scope):
    """list_chats as the person -> chat_id bridge. n_chats is the count of
    1:1 threads (a split iMessage/SMS pair counts twice -- that IS the
    answer, and a reply that silently merges them loses the distinction)."""
    from query_tools._shared import MAX_CHATS_LIMIT

    qt = _qt()
    rows = qt.list_chats(contact_id=scope["contact_id"], limit=MAX_CHATS_LIMIT)
    if len(rows) >= MAX_CHATS_LIMIT:
        # The cap is hard (directory.py clamps any limit to it), so a contact in
        # more chats than this silently loses rows -- including dyad rows. Say so
        # rather than let the roster cross-check below misreport it as divergence.
        raise ValueError(
            f"list_chats hit its {MAX_CHATS_LIMIT}-row cap for this contact -- the dyad "
            "count may be truncated; S9.2 needs a contact in fewer chats"
        )
    # list_chats returns every chat the person is in, groups included; the
    # dyad threads are the is_group=0 rows. Cross-checked against the roster's
    # chat_ids so a divergence surfaces here rather than as a graded failure.
    dyad_rows = [row for row in rows if not row["is_group"]]
    if {row["id"] for row in dyad_rows} != set(scope["chat_ids"]):
        raise ValueError(
            f"list_chats 1:1 rows {[row['id'] for row in dyad_rows]} != roster chat_ids "
            f"{scope['chat_ids']}"
        )
    return {
        "values": {"n_dyad_chats": _v(len(dyad_rows), "exact")},
        "raw": {
            "list_chats": rows,
            "n_chats_returned": len(rows),
            "dyad_chat_ids": scope["chat_ids"],
        },
    }


@recipe("message_reactions")
def message_reactions(roster, message):
    """get_message_reactions on a known-reacted message: who reacted and with
    what. The reactor names and tapback names are the answer -- raw 2000-range
    type codes leaking into a reply are a communication failure."""
    qt = _qt()
    rows = qt.get_message_reactions(message["guid"])
    # A retracted tapback keeps its row (type >= 3000, rendered 'removed-...').
    # It is not a reaction anyone can see today, so it must not inflate GT.
    live = [row for row in rows if not str(row["reaction"]).startswith("removed-")]
    removed = [row for row in rows if str(row["reaction"]).startswith("removed-")]
    if not live:
        raise ValueError(f"message {message['message_id']} has no live reactions")
    values = {"n_reactions": _v(len(live), "exact")}
    for i, row in enumerate(live, 1):
        values[f"reaction_{i}_type"] = _v(row["reaction"], "exact")
    return {
        "values": values,
        "raw": {"message": message, "reactions": rows, "removed_reactions": removed},
    }


@recipe("long_message_full")
def long_message_full(roster, message):
    """The truncation-recovery contract: search rows cap text at
    TEXT_PREVIEW_CHARS, so the tail of this message exists only behind
    get_full_messages. `text_tail` is past the cap -- a reply containing it
    proves the model recovered the full text rather than paraphrasing the
    preview."""
    from query_tools._shared import TEXT_PREVIEW_CHARS

    text = message["text"]
    # Whole words from a single line: an arbitrary character slice can straddle a
    # newline or a run of spaces, and models reflow whitespace when they quote --
    # a correct reply would then fail an `exact` substring check.
    window = text[TEXT_PREVIEW_CHARS:TEXT_PREVIEW_CHARS + 120]
    words = window.split("\n", 1)[0].split()[1:]  # drop the (likely partial) first word
    tail = ""
    for word in words:
        candidate = f"{tail} {word}".strip()
        if len(candidate) > 60:
            break
        tail = candidate
    if not tail:
        raise ValueError(
            f"message {message['message_id']} has no whole-word tail past the "
            f"{TEXT_PREVIEW_CHARS}-char preview cap"
        )
    return {
        "values": {
            # text_head is the roster's collapsed form -- the same string the
            # prompt quotes back, so a reply echoing the prompt matches it.
            "text_head": _v(message["text_head"], "exact"),
            "text_tail": _v(tail, "exact"),
        },
        "raw": {
            "message_id": message["message_id"],
            "length": len(text),
            "preview_cap": TEXT_PREVIEW_CHARS,
            "text": text,
        },
    }
