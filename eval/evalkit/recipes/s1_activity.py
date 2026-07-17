"""S1 recipes: activity patterns (heatmap argmax, volume series, streaks,
busiest day, message counts)."""

from ._base import _connect, _qt, _real, _scope_kwargs, _v, _year_bounds, recipe

@recipe("activity_argmax")
def activity_argmax(roster, axis, scope=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope) if scope else {}
    r = qt.get_activity_patterns(include=["heatmap"], **kwargs)
    counts = r["heatmap"]["counts"]
    raw = {"tool": r}
    values = {}
    if axis == "hour":
        hour_totals = [sum(row[h] for row in counts) for h in range(24)]
        values["top_hour"] = _v(max(range(24), key=hour_totals.__getitem__), "exact")
        if scope is None:
            # The classic failure is a UTC-shaped argmax (4-5h off); store it
            # so graders can recognize it in a wrong reply.
            with _connect() as con:
                rows = con.execute(
                    "SELECT CAST(substr(m.date, 12, 2) AS INTEGER) AS h, COUNT(*) AS n "
                    f"FROM messages m WHERE {_real()} GROUP BY h"
                ).fetchall()
            raw["utc_top_hour"] = max(rows, key=lambda r: r["n"])["h"]
    else:
        day_totals = [sum(row) for row in counts]
        top = max(range(len(day_totals)), key=day_totals.__getitem__)
        values["top_weekday"] = _v(r["heatmap"]["weekdays"][top], "exact")
    return {"values": values, "raw": raw}


@recipe("volume_series")
def volume_series(roster, scope, year):
    qt = _qt()
    date_from, date_to = _year_bounds(year)
    r = qt.get_activity_patterns(
        include=["volume_series"],
        granularity="month",
        date_from=date_from,
        date_to=date_to,
        **_scope_kwargs(scope),
    )
    values = {}
    for bucket in r["volume_series"]["rows"]:
        p = bucket["period"]
        values[f"{p}_sent"] = _v(bucket["sent"], "exact")
        values[f"{p}_received"] = _v(bucket["received"], "exact")
        values[f"{p}_total"] = _v(bucket["sent"] + bucket["received"], "exact")
    return {"values": values, "raw": r}


@recipe("year_compare")
def year_compare(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["volume_series"], granularity="year")
    series = sorted(r["volume_series"]["rows"], key=lambda b: b["period"])
    if len(series) < 2:
        raise ValueError("fewer than two year buckets in the corpus")
    last, this = series[-2], series[-1]
    this_total = this["sent"] + this["received"]
    last_total = last["sent"] + last["received"]
    direction = "more" if this_total > last_total else "less" if this_total < last_total else "same"
    return {
        "values": {
            "this_year_total": _v(this_total, "exact"),
            "last_year_total": _v(last_total, "exact"),
            "direction": _v(direction, "exact"),
        },
        "raw": r,
    }


@recipe("late_night")
def late_night(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["late_night"])
    ln = r["late_night"]
    values = {"late_night_share": _v(ln["share"], "pct1")}
    if ln.get("top_contacts"):
        values["top_contact"] = _v(ln["top_contacts"][0]["contact"], "exact")
    return {"values": values, "raw": r}


@recipe("streaks")
def streaks(roster, scope):
    qt = _qt()
    r = qt.get_activity_patterns(include=["streaks"], **_scope_kwargs(scope))
    s = r["streaks"]["longest_streak"]
    return {
        "values": {
            "streak_days": _v(s["days"], "exact"),
            "streak_start": _v(s["start"], "exact"),
            "streak_end": _v(s["end"], "exact"),
        },
        "raw": r,
    }


@recipe("busiest_day")
def busiest_day(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["busiest_days"])
    top = r["busiest_days"][0]
    return {
        "values": {"day": _v(top["day"], "exact"), "count": _v(top["count"], "exact")},
        "raw": r,
    }


@recipe("message_count")
def message_count(roster, scope, year=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope)
    if year:
        kwargs["date_from"], kwargs["date_to"] = _year_bounds(year)
    r = qt.get_chat_summary_stats(**kwargs)
    return {"values": {"total": _v(r["message_count"], "exact")}, "raw": r}
