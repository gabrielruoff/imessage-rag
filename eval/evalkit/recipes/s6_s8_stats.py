"""S4.7 / S6 / S7 / S8 recipes: summary stats, group dynamics, participant
leaderboards, cross-tool composites."""

from ._base import PARTICIPANT_REACTIONS_MIN_MESSAGES, _qt, _scope_kwargs, _v, recipe

@recipe("word_frequency_sender")
def word_frequency_sender(roster, scope):
    qt = _qt()
    r = qt.get_word_frequency(contact_id=scope["contact_id"], top=10)
    values = {}
    for i, row in enumerate(r["rows"][:5], 1):
        values[f"top_word_{i}"] = _v(row["ngram"], "exact")
    return {"values": values, "raw": r}


@recipe("group_dynamics")
def group_dynamics(roster, group, section):
    qt = _qt()
    r = qt.get_group_dynamics(chat_id=group["chat_ids"], include=[section])
    if "error" in r:
        raise ValueError(f"get_group_dynamics errored: {r['error']}")
    data = r[section]
    values = {}
    if section == "members":
        top = max(data, key=lambda m: m["messages"])
        values["top_member"] = _v(top["member"], "exact")
        values["top_member_share"] = _v(top["share"], "pct1")
    elif section == "lurkers":
        for i, row in enumerate(data, 1):
            values[f"lurker_{i}"] = _v(row["member"], "exact")
    elif section == "gini":
        values["gini"] = _v(data["gini"], "round")
    elif section == "igniters":
        if data.get("available") is False:
            raise ValueError(f"igniters unavailable: {data.get('note')}")
        values["top_igniter"] = _v(data["leaderboard"][0]["member"], "exact")
    elif section == "adjacency":
        top = data["pairs"][0]
        values["top_pair_replier"] = _v(top["replier"], "exact")
        values["top_pair_replied_to"] = _v(top["replied_to"], "exact")
    elif section == "group_lsm":
        if data.get("available") is False:
            raise ValueError(f"group_lsm unavailable: {data.get('note')}")
        scored = [m for m in data["members"] if m.get("lsm_score") is not None]
        low = min(scored, key=lambda m: m["lsm_score"])
        values["standout_member"] = _v(low["member"], "exact")
        values["standout_lsm"] = _v(low["lsm_score"], "round")
    return {"values": values, "raw": r}


@recipe("participant_reactions")
def participant_reactions(roster, group):
    """get_chat_participant_stats' own leaderboard: reactions RECEIVED per
    member, normalized by messages sent. get_group_dynamics has no such
    section, so this is the only tool that answers the question.

    MIN_MESSAGES matches the floor S6.8's prompt states out loud: a hidden
    threshold would be unguessable, and without one a member with a handful of
    messages wins on noise."""
    qt = _qt()
    rows = qt.get_chat_participant_stats(chat_id=group["chat_ids"])
    scored = [r for r in rows if r["messages_sent"] >= PARTICIPANT_REACTIONS_MIN_MESSAGES]
    if not scored:
        raise ValueError(
            f"no member of the group has >= {PARTICIPANT_REACTIONS_MIN_MESSAGES} messages"
        )
    rates = sorted(
        (
            {
                "participant": r["participant"],
                "messages_sent": r["messages_sent"],
                "reactions_received": r["reactions_received"],
                "rate": r["reactions_received"] / r["messages_sent"],
            }
            for r in scored
        ),
        key=lambda r: -r["rate"],
    )
    top = rates[0]
    return {
        "values": {
            "top_member": _v(top["participant"], "exact"),
            "top_member_reactions_received": _v(top["reactions_received"], "exact"),
            "top_member_messages_sent": _v(top["messages_sent"], "exact"),
        },
        # rates lets a grader adjudicate a near-miss (the top two can be close)
        "raw": {"rows": rows, "min_messages": PARTICIPANT_REACTIONS_MIN_MESSAGES, "rates": rates},
    }


@recipe("top_chats_leaderboard")
def top_chats_leaderboard(roster):
    """get_chat_summary_stats(group_by_chat=True): the leaderboard path. Its
    rows are bare chat_ids, so a correct answer also needs list_chats (or an
    equivalent) to put a name to the winner."""
    qt = _qt()
    rows = qt.get_chat_summary_stats(group_by_chat=True, limit=10)
    labelled = qt.list_chats(chat_id=[r["chat_id"] for r in rows])
    top1 = roster["TOP1"]
    dyad = qt.get_chat_summary_stats(**_scope_kwargs(top1))
    return {
        # TOP1 is the #1 1:1 contact by definition; the leaderboard's top row
        # may be a group chat or one half of a split thread, so the dyad total
        # -- not rows[0]["message_count"] -- is the number to expect.
        "values": {
            "top_contact": _v(top1["display_name"], "exact"),
            "top_contact_total": _v(dyad["message_count"], "exact"),
        },
        "raw": {"leaderboard": rows, "labels": labelled, "top_contact_stats": dyad},
    }


@recipe("attachments_pair")
def attachments_pair(roster, scope):
    """attachment_count is per individual attachment (a message with three
    photos counts three) and every filter applies to it alike."""
    qt = _qt()
    both = qt.get_chat_summary_stats(**_scope_kwargs(scope))
    mine = qt.get_chat_summary_stats(is_from_me=True, **_scope_kwargs(scope))
    return {
        "values": {
            "attachments_total": _v(both["attachment_count"], "exact"),
            "attachments_from_me": _v(mine["attachment_count"], "exact"),
        },
        "raw": {"both_sides": both, "from_me": mine},
    }


@recipe("total_messages_pair")
def total_messages_pair(roster, scope):
    qt = _qt()
    r = qt.get_chat_summary_stats(**_scope_kwargs(scope))
    # get_chat_summary_stats person scope counts messages *sent by* that
    # person; kept in raw so graders can recognize that number in a reply.
    sender = (
        qt.get_chat_summary_stats(contact_id=scope["contact_id"])
        if scope.get("contact_id") is not None else None
    )
    return {
        "values": {"total": _v(r["message_count"], "exact")},
        "raw": {"dyad_stats": r, "sender_scoped_stats": sender},
    }


@recipe("activity_and_latency")
def activity_and_latency(roster, scope):
    qt = _qt()
    ap = qt.get_activity_patterns(include=["heatmap"])
    counts = ap["heatmap"]["counts"]
    hour_totals = [sum(row[h] for row in counts) for h in range(24)]
    rd = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {
        "values": {
            "top_hour": _v(max(range(24), key=hour_totals.__getitem__), "exact"),
            "them_median_s": _v(rd["reply_latency"]["them"]["median_seconds"], "round"),
        },
        "raw": {"activity": ap, "dynamics": rd},
    }


@recipe("compare_pair")
def compare_pair(roster, a, b):
    qt = _qt()
    sa = qt.get_chat_summary_stats(**_scope_kwargs(a))
    sb = qt.get_chat_summary_stats(**_scope_kwargs(b))
    ra = qt.get_response_dynamics(**_scope_kwargs(a))
    rb = qt.get_response_dynamics(**_scope_kwargs(b))
    a_median = ra["reply_latency"]["them"]["median_seconds"]
    b_median = rb["reply_latency"]["them"]["median_seconds"]
    return {
        "values": {
            "a_total": _v(sa["message_count"], "exact"),
            "b_total": _v(sb["message_count"], "exact"),
            "a_them_median_s": _v(a_median, "round"),
            "b_them_median_s": _v(b_median, "round"),
            "more_side": _v(
                a["display_name"] if sa["message_count"] >= sb["message_count"]
                else b["display_name"],
                "exact",
            ),
            "faster_side": _v(
                a["display_name"] if a_median <= b_median else b["display_name"], "exact"
            ),
        },
        "raw": {"a_stats": sa, "b_stats": sb, "a_dynamics": ra, "b_dynamics": rb},
    }


@recipe("sparse_history")
def sparse_history(roster, scope):
    qt = _qt()
    stats = qt.get_chat_summary_stats(**_scope_kwargs(scope))
    ap = qt.get_activity_patterns(
        include=["volume_series"], granularity="year", **_scope_kwargs(scope)
    )
    values = {"lifetime_total": _v(stats["message_count"], "exact")}
    for bucket in ap["volume_series"]["rows"]:
        values[f"{bucket['period']}_total"] = _v(bucket["sent"] + bucket["received"], "exact")
    return {"values": values, "raw": {"stats": stats, "activity": ap}}


@recipe("top_contact_profile")
def top_contact_profile(roster):
    qt = _qt()
    top1 = roster["TOP1"]
    ap = qt.get_activity_patterns(include=["heatmap"], **_scope_kwargs(top1))
    counts = ap["heatmap"]["counts"]
    hour_totals = [sum(row[h] for row in counts) for h in range(24)]
    rd = qt.get_response_dynamics(**_scope_kwargs(top1))
    init = rd["initiation"]
    return {
        "values": {
            "top_contact": _v(top1["display_name"], "exact"),
            "top_hour": _v(max(range(24), key=hour_totals.__getitem__), "exact"),
            "initiation_me": _v(init["me"], "exact"),
            "initiation_them": _v(init["them"], "exact"),
        },
        "raw": {"activity": ap, "dynamics": rd},
    }
