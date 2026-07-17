"""Time-pattern analytics tools: get_activity_patterns (when do we talk --
local-tz weekday/hour heatmap, volume series, late-night share, busiest days,
streaks) and get_response_dynamics (how does the conversation move -- reply
latency, read-to-reply, initiations, session shape, ghosting, double-texting,
burstiness). Each returns one compact pre-aggregated dict so quantitative
"when / how fast / who first" questions never require paging raw rows.

All hour/weekday/day bucketing happens in Python in the local timezone
(config.LOCAL_TZ, or the system zone when unset): messages.date stores
unlabeled UTC, and UTC-bucketed "late night" would be off by hours. SQL
predicates stay plain-string date compares so idx_messages_chat_date keeps
working; substr()/strftime() never appear in WHERE.

Scope semantics match the other analytics tools: handle_id/contact_id resolve
to that person's *chats* (_resolve_chat_scope) so both sides are counted --
from-me rows carry handle_id=0, so a sender-level filter would drop the "me"
side. get_response_dynamics additionally restricts a person scope to that
person's 1:1 chats: response attribution is two-sided, so group chats join
only via an explicit chat_id.
"""

import collections
import contextlib
import datetime
import statistics
import zoneinfo

import config
from config import (
    DOUBLE_TEXT_MIN_MINUTES,
    INITIATION_GAP_HOURS,
    LATE_NIGHT_END_HOUR,
    LATE_NIGHT_START_HOUR,
)
from db import CONTACT_LABEL_SQL

from ._shared import (
    MAX_LEADERBOARD_ROWS,
    MAX_SERIES_BUCKETS,
    MIN_LEADERBOARD_MESSAGES,
    _append_id_filters,
    _as_id_list,
    _connect,
    _date_clauses,
    _REAL_MESSAGE_CLAUSES,
    _resolve_chat_scope,
    _scope_requested,
    _validate_include,
)

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"]  # datetime.weekday() order
_GRANULARITIES = ("day", "month", "year")
_ACTIVITY_SECTIONS = ("heatmap", "volume_series", "late_night", "busiest_days", "streaks")


def _local_tz():
    """Effective bucketing timezone: config.LOCAL_TZ (an IANA name) resolved
    through zoneinfo, or None -- which astimezone() treats as the system
    local zone. Read from config at call time (and kept as a function) so
    tests can pin a zone deterministically."""
    name = config.LOCAL_TZ
    return zoneinfo.ZoneInfo(name) if name else None


def _tz_label(tz):
    if tz is not None:
        return str(tz)
    return str(datetime.datetime.now().astimezone().tzinfo)


def _restrict_to_direct_chats(con, chat_ids):
    """Keep only 1:1 chats (is_group = 0) from a chat_id list."""
    if not chat_ids:
        return chat_ids
    placeholders = ",".join("?" * len(chat_ids))
    rows = con.execute(
        f"SELECT id FROM chats WHERE id IN ({placeholders}) AND is_group = 0",
        chat_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _side(is_from_me):
    return "me" if is_from_me else "them"


def _is_late_night(hour):
    """Local hour inside [LATE_NIGHT_START_HOUR, LATE_NIGHT_END_HOUR); a
    start > end window wraps midnight (e.g. 22..5)."""
    start, end = LATE_NIGHT_START_HOUR, LATE_NIGHT_END_HOUR
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _volume_section(series, granularity):
    """Render a {period: [sent, received]} dict as sorted rows, auto-coarsened
    day->month past MAX_SERIES_BUCKETS and hard-capped to the most recent
    MAX_SERIES_BUCKETS buckets either way."""
    note = None
    if granularity == "day" and len(series) > MAX_SERIES_BUCKETS:
        coarse = {}
        for period, counts in series.items():
            bucket = coarse.setdefault(period[:7], [0, 0])
            bucket[0] += counts[0]
            bucket[1] += counts[1]
        series = coarse
        granularity = "month"
        note = f"day series exceeded {MAX_SERIES_BUCKETS} buckets; coarsened to month"
    periods = sorted(series)
    if len(periods) > MAX_SERIES_BUCKETS:
        dropped = len(periods) - MAX_SERIES_BUCKETS
        periods = periods[-MAX_SERIES_BUCKETS:]
        extra = f"kept the {MAX_SERIES_BUCKETS} most recent buckets ({dropped} older dropped)"
        note = f"{note}; {extra}" if note else extra
    section = {
        "granularity": granularity,
        "rows": [{"period": p, "sent": series[p][0], "received": series[p][1]}
                 for p in periods],
    }
    if note:
        section["note"] = note
    return section


def get_activity_patterns(chat_id=None, handle_id=None, contact_id=None,
                          date_from=None, date_to=None, granularity="month",
                          include=None):
    """When-do-we-talk analytics, bucketed in the local timezone: `heatmap`
    (7x24 weekday-by-hour message counts), `volume_series` (per-period
    sent/received at granularity 'day'|'month'|'year'; a day series past
    MAX_SERIES_BUCKETS auto-coarsens to month with a note), `late_night`
    (share of messages in the configured late-night hour window; an unscoped
    call adds a top-contacts leaderboard over 1:1 chats), `busiest_days`
    (top 10 local dates), `streaks` (longest run of consecutive active days
    and the longest drought). Messages only, reactions excluded.
    include=["heatmap", ...] selects a subset of sections; no scope
    aggregates the whole corpus."""
    if granularity not in _GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}: expected one of "
                         f"{', '.join(_GRANULARITIES)}")
    sections = _validate_include(include, _ACTIVITY_SECTIONS, "get_activity_patterns")
    unscoped = not _scope_requested(chat_id, handle_id, contact_id)
    tz = _local_tz()
    want_leaderboard = unscoped and "late_night" in sections

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        rows = []
        if not empty:
            clauses, params = ["m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES], []
            _append_id_filters(clauses, params, chat_ids, None)
            _date_clauses(clauses, params, date_from, date_to)
            rows = con.execute(
                f"SELECT m.date, m.is_from_me, m.chat_id FROM messages m "
                f"WHERE {' AND '.join(clauses)} ORDER BY m.date, m.id",
                params,
            ).fetchall()
        contact_by_chat = {}
        if want_leaderboard:
            # Attribute each 1:1 chat's traffic (both sides) to its contact;
            # group chats are skipped -- a late-night group message has no
            # single counterparty to credit.
            contact_by_chat = {
                r[0]: (r[1], r[2])
                for r in con.execute(f"""
                    SELECT cp.chat_id, COALESCE(h.contact_id, -h.id) AS ckey,
                           {CONTACT_LABEL_SQL} AS label
                    FROM chat_participants cp
                    JOIN chats ch ON ch.id = cp.chat_id AND ch.is_group = 0
                    JOIN handles h ON h.id = cp.handle_id
                    LEFT JOIN contacts c ON c.id = h.contact_id
                """).fetchall()
            }

    heat = [[0] * 24 for _ in range(7)]
    series = {}                       # period -> [sent, received]
    day_counts = collections.Counter()  # local 'YYYY-MM-DD' -> count
    late_n = 0
    per_contact = {}                  # ckey -> {"label", "total", "late"}
    prev_utc = prev_local = None
    drought = None                    # (seconds, from_iso, to_iso)
    period_len = {"day": 10, "month": 7, "year": 4}[granularity]

    for date_str, is_from_me, cid in rows:
        utc = datetime.datetime.fromisoformat(date_str).replace(
            tzinfo=datetime.timezone.utc)
        loc = utc.astimezone(tz)
        day = loc.date().isoformat()
        heat[loc.weekday()][loc.hour] += 1
        day_counts[day] += 1
        bucket = series.setdefault(day[:period_len], [0, 0])
        bucket[0 if is_from_me else 1] += 1
        late = _is_late_night(loc.hour)
        if late:
            late_n += 1
        if want_leaderboard and cid in contact_by_chat:
            ckey, label = contact_by_chat[cid]
            stats = per_contact.setdefault(ckey, {"label": label, "total": 0, "late": 0})
            stats["total"] += 1
            stats["late"] += int(late)
        if prev_utc is not None:
            gap = (utc - prev_utc).total_seconds()
            if drought is None or gap > drought[0]:
                drought = (gap, prev_local.isoformat(), loc.isoformat())
        prev_utc, prev_local = utc, loc

    total = len(rows)
    result = {"timezone": _tz_label(tz), "message_count": total}

    if "heatmap" in sections:
        result["heatmap"] = {
            "weekdays": _WEEKDAY_NAMES,
            "counts": heat,
            "note": "rows are weekdays (Monday first), columns are local hours 0-23",
        }
    if "volume_series" in sections:
        result["volume_series"] = _volume_section(series, granularity)
    if "late_night" in sections:
        section = {
            "start_hour": LATE_NIGHT_START_HOUR,
            "end_hour": LATE_NIGHT_END_HOUR,
            "n": late_n,
            "total": total,
            "share": round(late_n / total, 4) if total else None,
        }
        if want_leaderboard:
            board = [
                {"contact": v["label"], "messages": v["total"],
                 "late_night": v["late"],
                 "share": round(v["late"] / v["total"], 4)}
                for v in per_contact.values() if v["total"] >= MIN_LEADERBOARD_MESSAGES
            ]
            board.sort(key=lambda r: (-r["share"], -r["messages"], r["contact"]))
            section["top_contacts"] = board[:MAX_LEADERBOARD_ROWS]
            section["top_contacts_note"] = (
                f"1:1 chats with >= {MIN_LEADERBOARD_MESSAGES} messages; "
                "both sides of each chat counted")
        result["late_night"] = section
    if "busiest_days" in sections:
        result["busiest_days"] = [
            {"day": day, "count": n}
            for day, n in sorted(day_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ]
    if "streaks" in sections:
        longest = None
        ordinals = sorted(datetime.date.fromisoformat(d).toordinal() for d in day_counts)
        if ordinals:
            best_len = run_len = 1
            best_end = run_start = ordinals[0]
            best_start = ordinals[0]
            for prev, cur in zip(ordinals, ordinals[1:]):
                if cur == prev + 1:
                    run_len += 1
                else:
                    run_start, run_len = cur, 1
                if run_len > best_len:
                    best_len, best_start, best_end = run_len, run_start, cur
            longest = {
                "days": best_len,
                "start": datetime.date.fromordinal(best_start).isoformat(),
                "end": datetime.date.fromordinal(best_end).isoformat(),
            }
        result["streaks"] = {
            "longest_streak": longest,
            "longest_drought": None if drought is None else {
                "days": round(drought[0] / 86400, 1),
                "from": drought[1],
                "to": drought[2],
            },
        }
    return result


def _latency_stats(samples):
    """Median/p90 seconds over a list of gaps (guarding tiny lists)."""
    if not samples:
        return {"median_seconds": None, "p90_seconds": None, "count": 0}
    p90 = statistics.quantiles(samples, n=10)[8] if len(samples) >= 2 else samples[0]
    return {
        "median_seconds": round(statistics.median(samples), 1),
        "p90_seconds": round(p90, 1),
        "count": len(samples),
    }


def _zero_response_sections():
    return {
        "message_count": 0,
        "from_me": 0,
        "from_them": 0,
        "reply_latency": {"me": _latency_stats([]), "them": _latency_stats([])},
        "read_to_reply": {
            "pairs": 0,
            "median_unseen_seconds": None,
            "median_deliberation_seconds": None,
            "coverage": {"inbound_with_date_read": 0, "inbound_total": 0,
                         "outbound_with_date_read": 0, "outbound_total": 0},
        },
        "initiation": {"me": 0, "them": 0},
        "sessions": {"count": 0, "median_messages": None,
                     "median_duration_seconds": None, "ended_by_me_share": None},
        "ghosted": {"only_me": 0, "only_them": 0},
        "double_text": {"me": 0, "them": 0},
        "burstiness": None,
    }


def _response_rows(con, chat_ids, date_from, date_to):
    """(id, is_from_me, date, date_read) for every real message in scope,
    date-ordered -- the single pass _scan_message_flow consumes."""
    clauses, params = ["m.date IS NOT NULL", *_REAL_MESSAGE_CLAUSES], []
    _append_id_filters(clauses, params, chat_ids, None)
    _date_clauses(clauses, params, date_from, date_to)
    return con.execute(
        f"SELECT m.id, m.is_from_me, m.date, m.date_read FROM messages m "
        f"WHERE {' AND '.join(clauses)} ORDER BY m.date, m.id",
        params,
    ).fetchall()


def _response_sessions(con, chat_ids, date_from, date_to):
    """Session rows for the same scope; a session is in-range when it
    *started* inside the date window (plain-string compares)."""
    sess_clauses = [f"s.chat_id IN ({','.join('?' * len(chat_ids))})"]
    sess_params = list(chat_ids)
    _date_clauses(sess_clauses, sess_params, date_from, date_to, column="s.first_date")
    return con.execute(
        "SELECT s.chat_id, s.n_messages, s.n_from_me, s.first_date, s.last_date, "
        "s.ender_is_me, s.initiator_is_me, s.initiator_handle_id "
        f"FROM sessions s WHERE {' AND '.join(sess_clauses)}",
        sess_params,
    ).fetchall()


def _group_chat_ids(con, chat_ids):
    """The subset of chat_ids that are group chats."""
    placeholders = ",".join("?" * len(chat_ids))
    return {r[0] for r in con.execute(
        f"SELECT id FROM chats WHERE id IN ({placeholders}) AND is_group = 1",
        chat_ids,
    ).fetchall()}


def _initiator_labels(con, sess, group_ids):
    """Contact labels for the group sessions' initiator handles."""
    handle_ids = sorted({r["initiator_handle_id"] for r in sess
                         if r["chat_id"] in group_ids
                         and r["initiator_handle_id"] is not None})
    if not handle_ids:
        return {}
    placeholders = ",".join("?" * len(handle_ids))
    return dict(con.execute(
        f"SELECT h.id, {CONTACT_LABEL_SQL} FROM handles h "
        f"LEFT JOIN contacts c ON c.id = h.contact_id "
        f"WHERE h.id IN ({placeholders})",
        handle_ids,
    ).fetchall())


def _scan_message_flow(rows, init_gap, double_gap):
    """One date-ordered pass over the message rows, accumulating every
    per-message section at once (the sections share prev-message state, so
    they can't be computed independently): side counts, initiations,
    per-side reply latencies, read-to-reply splits with date_read coverage,
    double-texts, and the raw inter-message gaps burstiness is computed
    from."""
    counts = {"me": 0, "them": 0}
    initiations = {"me": 0, "them": 0}
    double_texts = {"me": 0, "them": 0}
    latency = {"me": [], "them": []}
    unseen, deliberation = [], []
    coverage = {"inbound_with_date_read": 0, "inbound_total": 0,
                "outbound_with_date_read": 0, "outbound_total": 0}
    gaps = []
    prev_dt = prev_side = prev_read = None

    for _mid, is_from_me, date_str, date_read in rows:
        side = _side(is_from_me)
        counts[side] += 1
        direction = "outbound" if is_from_me else "inbound"
        coverage[f"{direction}_total"] += 1
        if date_read:
            coverage[f"{direction}_with_date_read"] += 1
        cur = datetime.datetime.fromisoformat(date_str)
        if prev_dt is None:
            initiations[side] += 1
        else:
            gap = (cur - prev_dt).total_seconds()
            gaps.append(gap)
            if gap >= init_gap:
                initiations[side] += 1
            elif side != prev_side:
                latency[side].append(gap)
                if is_from_me and prev_read:
                    # My reply to an inbound message I read: split the total
                    # into unseen (arrival -> read) + deliberation (read ->
                    # reply); out-of-order timestamps are dropped.
                    read_dt = datetime.datetime.fromisoformat(prev_read)
                    u = (read_dt - prev_dt).total_seconds()
                    d = (cur - read_dt).total_seconds()
                    if u >= 0 and d >= 0:
                        unseen.append(u)
                        deliberation.append(d)
            elif gap >= double_gap:
                double_texts[side] += 1
        prev_dt, prev_side, prev_read = cur, side, date_read

    return {
        "counts": counts,
        "initiations": initiations,
        "double_texts": double_texts,
        "latency": latency,
        "unseen": unseen,
        "deliberation": deliberation,
        "coverage": coverage,
        "gaps": gaps,
    }


def _burstiness(gaps):
    """Goh-Barabasi B over inter-message gaps (-1 regular .. +1 bursty)."""
    if len(gaps) < 2:
        return None
    mu = statistics.fmean(gaps)
    sigma = statistics.pstdev(gaps)
    if sigma + mu > 0:
        return round((sigma - mu) / (sigma + mu), 4)
    return None


def _sessions_section(sess):
    n_sess = len(sess)
    durations = [
        (datetime.datetime.fromisoformat(r["last_date"])
         - datetime.datetime.fromisoformat(r["first_date"])).total_seconds()
        for r in sess
    ]
    return {
        "count": n_sess,
        "median_messages": statistics.median([r["n_messages"] for r in sess])
        if n_sess else None,
        "median_duration_seconds": round(statistics.median(durations), 1)
        if n_sess else None,
        "ended_by_me_share": round(sum(r["ender_is_me"] for r in sess) / n_sess, 4)
        if n_sess else None,
    }


def _ghosted_section(sess):
    return {
        "only_me": sum(1 for r in sess
                       if r["n_messages"] > 0 and r["n_from_me"] == r["n_messages"]),
        "only_them": sum(1 for r in sess if r["n_messages"] > 0 and r["n_from_me"] == 0),
    }


def _read_to_reply_section(unseen, deliberation, coverage):
    return {
        "pairs": len(unseen),
        "median_unseen_seconds": round(statistics.median(unseen), 1)
        if unseen else None,
        "median_deliberation_seconds": round(statistics.median(deliberation), 1)
        if deliberation else None,
        "coverage": coverage,
    }


def _initiations_by_member(sess, group_ids, labels):
    """Session-starter leaderboard over the group chats in scope."""
    counter = collections.Counter()
    for r in sess:
        if r["chat_id"] not in group_ids:
            continue
        if r["initiator_is_me"]:
            counter["Me"] += 1
        else:
            hid = r["initiator_handle_id"]
            counter[labels.get(hid, f"handle:{hid}")] += 1
    return [
        {"member": member, "sessions_initiated": n}
        for member, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:MAX_LEADERBOARD_ROWS]


def get_response_dynamics(chat_id=None, handle_id=None, contact_id=None,
                          date_from=None, date_to=None):
    """How the conversation moves, for one scope (required -- chat_id, or
    handle_id/contact_id covering that person's 1:1 chats): `reply_latency`
    (median/p90 seconds per side; only gaps shorter than INITIATION_GAP_HOURS
    count as replies), `read_to_reply` (for my replies to read inbound
    messages: unseen = read - arrival, deliberation = reply - read; with
    honest per-direction date_read coverage counts), `initiation` (messages
    arriving after >= INITIATION_GAP_HOURS of silence, first message
    included), `sessions` (from the sessions table: count, median size and
    duration, ended_by_me_share), `ghosted` (fully one-sided sessions per
    side), `double_text` (same-sender follow-ups after >=
    DOUBLE_TEXT_MIN_MINUTES), and `burstiness` (Goh-Barabasi B over
    inter-message gaps, -1 regular .. +1 bursty). An explicit group chat_id
    adds `initiations_by_member` (session-starter leaderboard)."""
    if not _scope_requested(chat_id, handle_id, contact_id):
        return {"error": "get_response_dynamics needs a scope: chat_id, "
                         "handle_id, or contact_id"}
    init_gap = INITIATION_GAP_HOURS * 3600.0
    double_gap = DOUBLE_TEXT_MIN_MINUTES * 60.0

    with contextlib.closing(_connect()) as con:
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if not empty and chat_ids and _as_id_list(chat_id) is None:
            # Person scope without an explicit chat_id: 1:1 chats only.
            chat_ids = _restrict_to_direct_chats(con, chat_ids)
            empty = not chat_ids
        result = {"chat_ids": chat_ids, "initiation_gap_hours": INITIATION_GAP_HOURS}
        if empty:
            result.update(_zero_response_sections())
            return result

        rows = _response_rows(con, chat_ids, date_from, date_to)
        sess = _response_sessions(con, chat_ids, date_from, date_to)
        # initiations_by_member is offered only for an explicit group chat_id.
        group_ids = set()
        if _as_id_list(chat_id) is not None:
            group_ids = _group_chat_ids(con, chat_ids)
        labels = _initiator_labels(con, sess, group_ids)

    flow = _scan_message_flow(rows, init_gap, double_gap)

    result.update(
        message_count=len(rows),
        from_me=flow["counts"]["me"],
        from_them=flow["counts"]["them"],
        reply_latency={s: _latency_stats(flow["latency"][s]) for s in ("me", "them")},
        read_to_reply=_read_to_reply_section(
            flow["unseen"], flow["deliberation"], flow["coverage"]),
        initiation=flow["initiations"],
        sessions=_sessions_section(sess),
        ghosted=_ghosted_section(sess),
        double_text=flow["double_texts"],
        burstiness=_burstiness(flow["gaps"]),
    )

    if group_ids:
        result["initiations_by_member"] = _initiations_by_member(sess, group_ids, labels)
    return result
