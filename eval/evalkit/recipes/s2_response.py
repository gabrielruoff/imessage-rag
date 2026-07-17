"""S2 recipes: response dynamics (reply latency, initiation, double-texts,
ghosting, read-to-reply, sessions/burstiness, group initiators)."""

from ._base import _qt, _scope_kwargs, _v, recipe

@recipe("reply_latency")
def reply_latency(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    me, them = r["reply_latency"]["me"], r["reply_latency"]["them"]
    values = {
        "me_median_s": _v(me["median_seconds"], "round"),
        "them_median_s": _v(them["median_seconds"], "round"),
        "them_p90_s": _v(them["p90_seconds"], "round"),
        "faster_side": _v("me" if me["median_seconds"] < them["median_seconds"] else "them",
                          "exact"),
    }
    return {"values": values, "raw": r}


@recipe("reply_latency_pair")
def reply_latency_pair(roster, a, b):
    qt = _qt()
    ra = qt.get_response_dynamics(**_scope_kwargs(a))
    rb = qt.get_response_dynamics(**_scope_kwargs(b))
    return {
        "values": {
            "a_them_median_s": _v(ra["reply_latency"]["them"]["median_seconds"], "round"),
            "b_them_median_s": _v(rb["reply_latency"]["them"]["median_seconds"], "round"),
        },
        "raw": {"a": ra, "b": rb},
    }


@recipe("initiation")
def initiation(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    init = r["initiation"]
    more = "me" if init["me"] > init["them"] else "them" if init["them"] > init["me"] else "even"
    return {
        "values": {
            "initiation_me": _v(init["me"], "exact"),
            "initiation_them": _v(init["them"], "exact"),
            "more_side": _v(more, "exact"),
        },
        "raw": r,
    }


@recipe("double_text")
def double_text(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {"values": {"double_text_me": _v(r["double_text"]["me"], "exact")}, "raw": r}


@recipe("ghosted")
def ghosted(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {"values": {"ghosted_only_me": _v(r["ghosted"]["only_me"], "exact")}, "raw": r}


@recipe("read_to_reply")
def read_to_reply(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    rr = r["read_to_reply"]
    return {
        "values": {
            "pairs": _v(rr["pairs"], "exact"),
            "inbound_with_date_read": _v(rr["coverage"]["inbound_with_date_read"], "exact"),
            "inbound_total": _v(rr["coverage"]["inbound_total"], "exact"),
        },
        "raw": r,
    }


@recipe("sessions_burstiness")
def sessions_burstiness(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    s = r["sessions"]
    return {
        "values": {
            "session_count": _v(s["count"], "exact"),
            "median_messages": _v(s["median_messages"], "exact"),
            "burstiness": _v(r["burstiness"], "round"),
        },
        "raw": r,
    }


@recipe("group_response")
def group_response(roster, group):
    qt = _qt()
    r = qt.get_response_dynamics(chat_id=group["chat_ids"])
    by_member = r.get("initiations_by_member")
    if not by_member and len(group["chat_ids"]) > 1:
        r = qt.get_response_dynamics(chat_id=group["chat_ids"][0])
        by_member = r.get("initiations_by_member")
    values = {}
    if by_member:
        values["top_initiator"] = _v(by_member[0]["member"], "exact")
    return {"values": values, "raw": r}
