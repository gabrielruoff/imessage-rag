"""S3 recipes: reactions (tapback leaderboards, hall of fame, reciprocity,
funniest, time-to-react)."""

from ._base import _qt, _scope_kwargs, _v, _year_bounds, recipe

@recipe("reactions_top_type")
def reactions_top_type(roster, direction):
    qt = _qt()
    r = qt.get_reaction_stats(include=[direction])
    name, count = max(r[direction]["by_type"].items(), key=lambda kv: kv[1])
    return {
        "values": {"top_type": _v(name, "exact"), "top_type_count": _v(count, "exact")},
        "raw": r,
    }


@recipe("hall_of_fame_top")
def hall_of_fame_top(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["hall_of_fame"])
    top = r["hall_of_fame"][0]
    return {
        "values": {
            "text": _v(top["text"], "exact"),
            "date": _v(top["date"], "exact"),
            "reaction_count": _v(top["reaction_count"], "exact"),
        },
        "raw": r,
    }


@recipe("reciprocity")
def reciprocity(roster, scope):
    qt = _qt()
    r = qt.get_reaction_stats(**_scope_kwargs(scope))
    rec = r["reciprocity"]
    return {
        "values": {
            "given": _v(rec["given"], "exact"),
            "received": _v(rec["received"], "exact"),
            "ratio": _v(rec["ratio"], "round"),
        },
        "raw": r,
    }


@recipe("funniest")
def funniest(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["funniest"])
    top = r["funniest"][0]
    return {
        "values": {
            "funniest": _v(top["participant"], "exact"),
            "laughs_per_100": _v(top["laughs_per_100_my_messages"], "round"),
        },
        "raw": r,
    }


@recipe("time_to_react")
def time_to_react(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["time_to_react"])
    return {
        "values": {
            "them_median_s": _v(r["time_to_react"]["them_reacting"]["median_seconds"], "round")
        },
        "raw": r,
    }


@recipe("reaction_types_pair")
def reaction_types_pair(roster, scope):
    qt = _qt()
    r = qt.get_reaction_stats(**_scope_kwargs(scope))
    values = {}
    for direction in ("given", "received"):
        by_type = r[direction]["by_type"]
        if by_type:
            name, _ = max(by_type.items(), key=lambda kv: kv[1])
            values[f"{direction}_top_type"] = _v(name, "exact")
    return {"values": values, "raw": r}


@recipe("reactions_received_year")
def reactions_received_year(roster, type, year):
    qt = _qt()
    date_from, date_to = _year_bounds(year)
    r = qt.get_reaction_stats(
        include=["received"], reaction_type=type, date_from=date_from, date_to=date_to
    )
    return {"values": {"count": _v(r["received"]["total"], "exact")}, "raw": r}
