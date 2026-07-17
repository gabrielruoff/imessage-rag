"""S5 recipes: milestones (firsts/lasts, landmarks, silences, anniversaries,
earliest phrase)."""

from ._base import _connect, _qt, _real, _scope_kwargs, _v, recipe

@recipe("milestone_firsts")
def milestone_firsts(roster, scope=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope) if scope else {}
    r = qt.get_milestones(**kwargs)  # get_milestones has no include param
    firsts = r["firsts"]
    values = {}
    if firsts.get("first"):
        values["first_date"] = _v(firsts["first"]["date"], "exact")
        values["first_sender"] = _v(firsts["first"]["sender"], "exact")
        values["first_text"] = _v(firsts["first"]["text"], "exact")
    if firsts.get("last"):
        values["last_date"] = _v(firsts["last"]["date"], "exact")
        values["last_text"] = _v(firsts["last"]["text"], "exact")
    return {"values": values, "raw": r}


@recipe("last_message_from_them")
def last_message_from_them(roster, scope):
    """The last message THEY sent (S5.6 asks 'the last thing <X> sent me').
    get_milestones' firsts.last is the last message regardless of sender --
    when that one is mine it would be wrong GT for this prompt -- so this
    queries their side directly."""
    chat_ids = scope["chat_ids"]
    placeholders = ",".join("?" * len(chat_ids))
    with _connect() as con:
        row = con.execute(
            f"SELECT m.date, m.text FROM messages m "
            f"WHERE m.chat_id IN ({placeholders}) AND m.is_from_me = 0 AND {_real()} "
            "ORDER BY m.date DESC LIMIT 1",
            chat_ids,
        ).fetchone()
    if row is None:
        raise ValueError(f"no inbound message in chats {chat_ids}")
    return {
        "values": {
            "last_date": _v(row["date"], "exact"),
            "last_text": _v(row["text"], "exact"),
        },
        "raw": {"date": row["date"], "text": row["text"]},
    }


@recipe("nth_landmark")
def nth_landmark(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    landmarks = r["nth"]["landmarks"]
    if not landmarks:
        raise ValueError("no landmark ordinals exist for this chat")
    top = max(landmarks, key=lambda row: row["ordinal"])
    return {
        "values": {
            "landmark_ordinal": _v(top["ordinal"], "exact"),
            "landmark_date": _v(top["message"]["date"], "exact"),
            "landmark_text": _v(top["message"]["text"], "exact"),
        },
        "raw": r,
    }


@recipe("longest_silence")
def longest_silence(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    s = r["longest_silence"]
    if not s.get("available", True):
        raise ValueError("longest_silence unavailable for scope")
    return {
        "values": {
            "silence_days": _v(s["days"], "round"),
            "silence_from": _v(s["from"], "exact"),
            "silence_to": _v(s["to"], "exact"),
            "broken_by": _v(s["broken_by"], "exact"),
        },
        "raw": r,
    }


@recipe("anniversaries_upcoming")
def anniversaries_upcoming(roster):
    qt = _qt()
    r = qt.get_milestones()
    values = {}
    for i, row in enumerate(r["anniversaries"]["upcoming"][:3], 1):
        values[f"upcoming_{i}_participant"] = _v(row["participant"], "exact")
        values[f"upcoming_{i}_date"] = _v(row["next_anniversary"], "exact")
    return {"values": values, "raw": r}


@recipe("earliest_phrase")
def earliest_phrase(roster, scope, phrase):
    qt = _qt()
    r = qt.get_milestones(phrase=phrase, **_scope_kwargs(scope))
    msg = r["earliest_phrase"]["message"]
    if not msg:
        raise ValueError(f"no hit for phrase {phrase!r} in scope")
    return {
        "values": {
            "first_mention_date": _v(msg["date"], "exact"),
            "first_mention_text": _v(msg["text"], "exact"),
        },
        "raw": r,
    }


@recipe("anniversary_first")
def anniversary_first(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    return {
        "values": {"first_contact": _v(r["anniversaries"]["first_contact"], "exact")},
        "raw": r,
    }
