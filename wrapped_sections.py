"""The Wrapped report's section builders: build_sections assembles the
ordered list of report sections, each `_sec_*` builder computing one
section's stats from a single analytics tool call (plus two direct SQL
helpers for group rosters and inside jokes). Split out of wrapped.py, which
keeps the CLI, roster/exclusion logic, rendering, and the `_tool` / scope
helpers these builders resolve late (module-attribute access), so the two
modules can import each other safely.

Sections are visual-first: each builder reshapes its tool's raw result into
reader-facing rows (plumbing dropped, jargon renamed, errors removed), then
renders them through wrapped_html's chart builders into `extra_html` --
bars, split bars, chat bubbles, a word cloud, joke cards -- keeping only
stat tiles in `stats`. The full shaped data goes in `narrate_stats` so the
LLM narrator sees everything the visuals show. Sections may also carry a
`caption` (a one-line methodology note rendered under the heading) and a
`hint` (context prepended to the narrator prompt, never rendered)."""

import datetime
import re

import config
import db
import progress
import wrapped
import wrapped_html
from function_words import COMMON_WORDS
from query_tools._shared import MIN_LEADERBOARD_MESSAGES

_URL_ONLY_RE = re.compile(r"^https?://\S+$")


def _sum_series(series, key):
    return sum(int(row.get(key) or 0) for row in series if isinstance(row, dict))


def _available(value):
    """True when a tool sub-result is real data, not an error/unavailable
    marker (those must never render as report content)."""
    if value is None:
        return False
    if isinstance(value, dict) and value.get("available") is False:
        return False
    return True


def _hour_range(start, end):
    """(0, 5) -> 'Midnight – 5 AM'."""
    lo = "Midnight" if int(start) % 24 == 0 else wrapped_html.format_hour(start)
    return f"{lo} – {wrapped_html.format_hour(end)}"


def _message_row(msg):
    """A raw message dict -> the three columns a keepsake needs.
    Attachment-only messages (text is just the ￼ placeholder) get labeled
    instead of rendering as an empty dash."""
    text = (msg.get("text") or "").replace("￼", "").strip()
    if not text and msg.get("has_attachment"):
        text = "(attachment)"
    return {"sender": msg.get("sender"), "text": text or None,
            "date": msg.get("date")}


def _your_people_stats(con, roster, year, exclude_chat_ids=(), exclude_ranges=(),
                       excluded_names=frozenset()):
    """Roster ranks with, when prior years exist, each contact's rank in the
    previous year's roster. The prior-year roster is built with the SAME
    exclusions as the current one -- otherwise an excluded chat inflates the
    ranks a 'previous_rank' is compared against; excluded persons are also
    dropped by name, so group-chat traffic can't keep one in a prior rank
    slot their 1:1 exclusion vacated."""
    prev_rank = {}
    have_prior = False
    if year is not None and con.execute(
        "SELECT 1 FROM messages WHERE date < ? LIMIT 1", (f"{year:04d}-01-01",)
    ).fetchone():
        have_prior = True
        prev = wrapped.build_roster(con, year - 1, exclude_chat_ids, exclude_ranges,
                                    limit=max(50, len(roster)))
        prev = [e for e in prev if e["name"].lower() not in excluded_names]
        prev_rank = {wrapped._entry_key(e): i + 1 for i, e in enumerate(prev)}
    people = []
    for i, entry in enumerate(roster, 1):
        row = {"rank": i, "name": entry["name"], "messages": entry["n_messages"]}
        pr = prev_rank.get(wrapped._entry_key(entry))
        if pr is not None:
            row["previous_rank"] = pr
        elif have_prior:
            row["previous_rank"] = None
        people.append(row)
    return people, have_prior


def _top_group_chats(con, year, exclude_chat_ids=(), limit=3):
    clauses = ["ch.is_group = 1", "m.date IS NOT NULL", wrapped._REAL_MSG]
    params = []
    if year is not None:
        clauses += ["m.date >= ?", "m.date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("m.chat_id != ?")
        params.append(cid)
    return con.execute(
        f"""
        SELECT m.chat_id, NULLIF(ch.display_name, '') AS display_name,
               ch.chat_identifier, COUNT(*) AS n_messages
        FROM messages m JOIN chats ch ON ch.id = m.chat_id
        WHERE {" AND ".join(clauses)}
        GROUP BY m.chat_id ORDER BY n_messages DESC LIMIT ?
        """,
        params + [limit],
    ).fetchall()


def _sec_year_in_numbers(patterns, patterns_note, year):
    """Built from the get_activity_patterns result shared with
    _sec_late_nights (one tool call serves both sections). Late-night detail
    lives in that later section only -- no duplicated block."""
    if patterns is None:
        return {"title": "Year in Numbers", "stats": None, "note": patterns_note}
    series = patterns.get("volume_series") or []
    if isinstance(series, dict):  # real tool wraps rows in {granularity, rows}
        series = series.get("rows") or []
    if year is not None:
        # Local-tz bucketing can spill a few edge messages into the
        # neighboring month; a 25-message bar for last December is noise.
        series = [r for r in series if str(r.get("period", "")).startswith(str(year))]
    stats = {
        "messages_sent": _sum_series(series, "sent"),
        "messages_received": _sum_series(series, "received"),
    }
    busiest = max(series, key=lambda r: (r.get("sent") or 0) + (r.get("received") or 0),
                  default=None)
    if busiest is not None:
        stats["busiest_month"] = busiest.get("period")
    stats["volume_series"] = series
    return {
        "title": "Year in Numbers",
        "stats": stats,
        "hint": ("volume_series is messages per month. Mention the totals and the "
                 "overall shape of the year (peaks, lulls), not every month."),
    }


def _sec_your_people(con, roster, year, exclude_chat_ids, exclude_ranges,
                     excluded_names=frozenset()):
    people, have_prior = _your_people_stats(con, roster, year, exclude_chat_ids,
                                            exclude_ranges, excluded_names)
    if not people:
        return {"title": "Your People", "stats": None, "note": "no contacts in scope"}
    rows = []
    for p in people:
        row = dict(p)
        if have_prior:
            pr = p.get("previous_rank")
            if pr is None:
                row["_badge"] = "new"
            elif pr > p["rank"]:
                row["_badge"], row["_badge_up"] = f"▲ {pr - p['rank']}", True
                row["_tip"] = f"up from #{pr} last year"
            elif pr < p["rank"]:
                row["_badge"] = f"▼ {p['rank'] - pr}"
                row["_tip"] = f"down from #{pr} last year"
        rows.append(row)
    return {
        "title": "Your People",
        "stats": {},
        "narrate_stats": {"people": people},
        "extra_html": wrapped_html.render_hbars(rows, "name", "messages"),
        "caption": ("Ranked by messages received from each person"
                    + (", with movement since last year." if have_prior else ".")),
        "hint": ("Ranked by messages received from each person. 'Previous rank' is "
                 "last year's rank -- call out the biggest climber or a notable "
                 "swap, not every row."),
    }


def _sec_tempo(tempo_entries, dates):
    """One comparable row per contact; the reply-speed comparison renders as
    opposed me/them bars. Returns (tempo_section, ghosts_section) -- one
    get_response_dynamics call per contact feeds both."""
    rows, ghost_rows, tempo_notes = [], [], []
    # Generic label: the roster's names must not reach stdout here.
    with progress.track("Analyzing tempo", total=len(tempo_entries)) as p:
        for entry in tempo_entries:
            res, note = wrapped._tool("get_response_dynamics",
                                      **wrapped._entry_scope(entry), **dates)
            if res is None:
                tempo_notes.append(f"{entry['name']}: {note}")
                p.advance()
                continue
            latency = res.get("reply_latency") or {}
            initiation = res.get("initiation") or {}
            sessions = res.get("sessions") or {}
            rows.append({
                "contact": entry["name"],
                "my_median_reply_seconds": (latency.get("me") or {}).get("median_seconds"),
                "their_median_reply_seconds": (latency.get("them") or {}).get("median_seconds"),
                "conversations": sessions.get("count"),
                # A 0 median means mostly single-message sessions; render the
                # em dash, not a meaningless "0 s".
                "typical_length_seconds": sessions.get("median_duration_seconds") or None,
                "started_by_me": initiation.get("me"),
                "started_by_them": initiation.get("them"),
                "ended_by_me_share": sessions.get("ended_by_me_share"),
            })
            ghosted = res.get("ghosted") or {}
            double = res.get("double_text") or {}
            if any((ghosted.get("only_me"), ghosted.get("only_them"),
                    double.get("me"), double.get("them"))):
                ghost_rows.append({
                    "contact": entry["name"],
                    "i_went_unanswered": ghosted.get("only_me"),
                    "they_went_unanswered": ghosted.get("only_them"),
                    "double_texts_by_me": double.get("me"),
                    "double_texts_by_them": double.get("them"),
                })
            p.advance()
    if not rows:
        tempo = {"title": "Tempo", "stats": None,
                 "note": "; ".join(tempo_notes[:2]) or "no contacts in scope"}
        return tempo, None
    # Fastest repliers first: the sorted bars tell the story on their own.
    rows.sort(key=lambda r: (r["my_median_reply_seconds"] is None,
                             r["my_median_reply_seconds"] or 0))
    visual = wrapped_html.render_split_bars(
        rows, "contact", "my_median_reply_seconds", "their_median_reply_seconds",
        left_label="My median reply", right_label="Their median reply",
        sqrt_scale=True,
    )
    if visual is None:  # nothing numeric to chart -- degrade to the table
        visual = wrapped_html._rows_table(rows, wrapped_html._columns(rows))
    tempo = {
        "title": "Tempo",
        "stats": {},
        "narrate_stats": {"per_contact": rows},
        "extra_html": visual,
        "caption": ("Reply times are medians, fastest first; bar length is "
                    "square-root scaled so seconds and hours share one picture. "
                    "The table view has conversation counts and who starts them."),
        "hint": ("Each row is one contact: median reply times for each side, how many "
                 "conversations you had, how long a typical one ran, and who starts "
                 "them. Contrast the extremes -- fastest vs slowest, who does the "
                 "starting -- in plain language. Never list every contact."),
    }
    return tempo, _sec_ghosts(ghost_rows)


def _sec_ghosts(rows):
    """Who talks into the void, and who double-texts: fully one-sided
    conversations and long-gap same-sender follow-ups, per contact."""
    if not rows:
        return None
    visuals = []
    unanswered = wrapped_html.render_split_bars(
        rows, "contact", "i_went_unanswered", "they_went_unanswered",
        left_label="I went unanswered", right_label="They went unanswered")
    if unanswered:
        visuals.append(wrapped_html.block("Unanswered conversations", unanswered))
    doubles = wrapped_html.render_split_bars(
        rows, "contact", "double_texts_by_me", "double_texts_by_them",
        left_label="By me", right_label="By them")
    if doubles:
        visuals.append(wrapped_html.block("Double texts", doubles))
    if not visuals:
        return None
    gap = config.DOUBLE_TEXT_MIN_MINUTES
    return {
        "title": "Ghosts & Double Texts",
        "stats": {},
        "narrate_stats": {"per_contact": rows},
        "extra_html": "".join(visuals),
        "caption": ("An unanswered conversation is one where every message came "
                    "from one side. A double text is following yourself up after "
                    f"{gap}+ minutes with no reply in between."),
        "hint": ("'I went unanswered' counts conversations where I sent every "
                 "message and never got a reply -- I was the one left hanging. "
                 "'They went unanswered' is the reverse: they talked, I never "
                 "replied. Double texts are follow-ups to your own message after "
                 "a long silence. Keep it playful, never judgmental; name at most "
                 "two contacts and never flip who was ignored."),
    }


def _person_chat_ids(con, contact_entry):
    """All chat ids a contact participates in (for direct-SQL sections that
    must honor the deep-dive scope)."""
    if contact_entry.get("contact_id") is not None:
        who, param = "h.contact_id = ?", contact_entry["contact_id"]
    else:
        who, param = "h.id = ?", contact_entry["handle_id"]
    return [r[0] for r in con.execute(
        f"""SELECT DISTINCT cp.chat_id FROM chat_participants cp
            JOIN handles h ON h.id = cp.handle_id WHERE {who}""", (param,))]


def _session_duration(first_date, last_date):
    try:
        delta = (datetime.datetime.fromisoformat(last_date)
                 - datetime.datetime.fromisoformat(first_date))
        return max(delta.total_seconds(), 0)
    except (TypeError, ValueError):
        return None


def _sec_marathons(con, contact_entry, year, exclude_chat_ids=()):
    """The year's longest sittings, straight from the sessions table."""
    clauses, params = ["s.n_messages >= 20"], []
    if year is not None:
        clauses += ["s.first_date >= ?", "s.first_date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("s.chat_id != ?")
        params.append(cid)
    if contact_entry is not None:
        chat_ids = _person_chat_ids(con, contact_entry)
        if not chat_ids:
            return None
        clauses.append(f"s.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += chat_ids
    rows = con.execute(
        f"""
        SELECT s.n_messages, s.first_date, s.last_date,
               NULLIF(ch.display_name, '') AS display_name, ch.is_group,
               (SELECT {db.CONTACT_LABEL_SQL}
                FROM chat_participants cp
                JOIN handles h ON h.id = cp.handle_id
                LEFT JOIN contacts c ON c.id = h.contact_id
                WHERE cp.chat_id = s.chat_id
                ORDER BY h.id LIMIT 1) AS partner
        FROM sessions s JOIN chats ch ON ch.id = s.chat_id
        WHERE {" AND ".join(clauses)}
        ORDER BY s.n_messages DESC LIMIT 5
        """,
        params,
    ).fetchall()
    if not rows:
        return None
    marathons = []
    for row in rows:
        if row["is_group"]:
            partner = row["display_name"] or "a group chat"
        else:
            partner = row["display_name"] or row["partner"]
        duration = _session_duration(row["first_date"], row["last_date"])
        marathons.append({
            "with": partner, "messages": row["n_messages"],
            "day": row["first_date"],
            "duration": wrapped_html.format_duration(duration)
            if duration else None,
            "_tip": " · ".join(p for p in (
                wrapped_html.display_scalar(row["first_date"], "date"),
                wrapped_html.format_duration(duration) if duration else "",
            ) if p),
        })
    top = marathons[0]
    stats = {"longest_sitting": f"{top['messages']:,} messages",
             "it_ran": top.get("duration"), "with_whom": top["with"],
             "on": top["day"]}
    chart = wrapped_html.render_hbars(marathons, "with", "messages", table=False)
    return {
        "title": "The Marathon",
        "stats": stats,
        "narrate_stats": {"top_sittings": [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in marathons]},
        "extra_html": wrapped_html.block("Top sittings", chart) if chart else "",
        "caption": ("A sitting is one unbroken conversation -- no long silences "
                    "from either side until it ends."),
        "hint": ("Lead with the single longest sitting: how many messages, with "
                 "whom, how long it ran, and the date. One sentence on the "
                 "runners-up at most."),
    }


def _reaction_type_rows(given, received):
    """Two {type: count} maps -> [{reaction, given, received}] rows, ordered
    by combined count, with an emoji-prefixed label for the chart."""
    g, r = given or {}, received or {}
    rows = []
    for kind in sorted(set(g) | set(r), key=lambda k: -(g.get(k, 0) + r.get(k, 0))):
        emoji = wrapped_html._REACTION_EMOJI.get(kind, "")
        name = kind.replace("-", " ").capitalize()
        rows.append({"reaction": f"{emoji} {name}".strip(),
                     "given": g.get(kind, 0), "received": r.get(kind, 0)})
    return rows


def _sec_reactions(scope, dates, excludes=None):
    reactions, note = wrapped._tool("get_reaction_stats", **scope, **dates,
                                    **(excludes or {}))
    if reactions is None:
        return {"title": "Reactions", "stats": None, "note": note}
    given = reactions.get("given") or {}
    received = reactions.get("received") or {}
    stats, narrate, visuals = {}, {}, []
    if given.get("total") is not None or received.get("total") is not None:
        stats["reactions_given"] = narrate["reactions_given"] = given.get("total")
        stats["reactions_received"] = narrate["reactions_received"] = received.get("total")
    by_type = _reaction_type_rows(given.get("by_type"), received.get("by_type"))
    if by_type:
        narrate["by_type"] = by_type
        chart = wrapped_html.render_split_bars(
            by_type, "reaction", "given", "received",
            left_label="Given", right_label="Received")
        if chart:
            visuals.append(wrapped_html.block("By type", chart))
    funniest = reactions.get("funniest")
    # A single-row leaderboard (the deep-dive) says nothing -- suppress it.
    if funniest and len(funniest) > 1:
        audience = [
            {"friend": row.get("participant"), "laughs": row.get("laughs"),
             "my_messages": row.get("my_messages"),
             "laughs_per_100_my_messages": row.get("laughs_per_100_my_messages"),
             "_tip": f"{row.get('laughs')} laugh reactions on "
                     f"{row.get('my_messages'):,} of my messages"
             if row.get("my_messages") else None}
            for row in funniest
        ]
        narrate["best_audience"] = [{k: v for k, v in r.items()
                                     if not k.startswith("_")} for r in audience]
        chart = wrapped_html.render_hbars(
            audience, "friend", "laughs_per_100_my_messages")
        if chart:
            visuals.append(wrapped_html.block("Best audience", chart))
    fame = reactions.get("hall_of_fame")
    if fame:
        bubbles = [
            {**_message_row(row), "reactions": row.get("reactions_by_type")}
            for row in fame
            # A bare link isn't a keepsake -- drop URL-only messages.
            if not _URL_ONLY_RE.match((row.get("text") or "").strip())
        ]
        if bubbles:
            narrate["hall_of_fame"] = bubbles
            visuals.append(wrapped_html.block(
                "Hall of fame", wrapped_html.render_bubbles(bubbles)))
    if not (stats or visuals):
        return {"title": "Reactions", "stats": None, "note": note}
    return {
        "title": "Reactions",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": ("Best audience ranks who laugh-reacts to your messages most "
                    "often, per 100 of your messages. The hall of fame is your "
                    "most-reacted messages."),
        "hint": ("Mention the totals, the favorite reaction type on each side, and "
                 "at most one hall-of-fame or best-audience standout."),
    }


def _sec_late_nights(patterns, patterns_note):
    """Second consumer of the shared get_activity_patterns result."""
    if patterns is None:
        return {"title": "Late Nights, Streaks & the Heatmap",
                "stats": None, "note": patterns_note}
    stats, narrate, visuals = {}, {}, []
    ln = patterns.get("late_night") or {}
    # Computed even when late_night is missing (defaults match the tool's):
    # the caption below must describe the configured window, not a literal.
    hours = _hour_range(ln.get("start_hour", 0), ln.get("end_hour", 5))
    if ln:
        if ln.get("n") is not None:
            stats["late_night_messages"] = ln.get("n")
            stats["late_night_share"] = ln.get("share")
        narrate["late_night"] = {"hours": hours, "messages": ln.get("n"),
                                 "share_of_messages": ln.get("share")}
        owls = [
            {"contact": row.get("contact"), "all_messages": row.get("messages"),
             "late_night": row.get("late_night"), "share": row.get("share"),
             "_tip": f"{row.get('late_night')} of {row.get('messages'):,} "
                     "messages after midnight"
             if row.get("messages") else None}
            for row in ln.get("top_contacts") or []
        ]
        if len(owls) > 1:
            narrate["late_night"]["night_owls"] = [
                {k: v for k, v in r.items() if not k.startswith("_")} for r in owls]
            chart = wrapped_html.render_hbars(owls, "contact", "share")
            if chart:
                visuals.append(wrapped_html.block("Night owls", chart))
    streaks = patterns.get("streaks") or {}
    longest = streaks.get("longest_streak") or {}
    drought = streaks.get("longest_drought") or {}
    if longest.get("days") is not None:
        stats["longest_streak"] = f"{longest['days']:,} days"
        narrate["longest_streak"] = longest
    if drought.get("days") is not None:
        stats["longest_quiet_spell"] = (
            wrapped_html.display_scalar(drought["days"]) + " days")
        narrate["longest_quiet_stretch"] = drought
    busiest = patterns.get("busiest_days") or []
    if busiest:
        days = [{"day": row.get("day"), "messages": row.get("count", row.get("n"))}
                for row in busiest]
        narrate["busiest_days"] = days
        chart = wrapped_html.render_hbars(days, "day", "messages", table=False)
        if chart:
            visuals.append(wrapped_html.block("Busiest days", chart))
    heatmap = wrapped_html.render_heatmap(patterns.get("heatmap"))
    if heatmap:
        visuals.append(heatmap)
    if not (stats or visuals):
        return {"title": "Late Nights, Streaks & the Heatmap",
                "stats": None, "note": patterns_note}
    return {
        "title": "Late Nights, Streaks & the Heatmap",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": (f"Late night is {hours[0].lower() + hours[1:]}. Night owls "
                    f"are 1:1 chats "
                    f"with at least {MIN_LEADERBOARD_MESSAGES} messages, ranked "
                    "by the share of the chat that happens then."),
        "hint": ("The streak is consecutive days with at least one message; the quiet "
                 "stretch is the longest gap without any; day counts include both "
                 "sides' messages ('you exchanged', not 'you sent'). Mention the "
                 "late-night share, the streak, and the single busiest day."),
    }


def _inside_jokes_all(con, roster, exclude_chat_ids=(), excluded_names=frozenset(),
                      limit=14, min_uses=10):
    """Featured-roster inside jokes straight from dyad_terms: the LLM-curated
    inside_joke/nickname verdicts, heaviest use first, with provenance (who
    coined it, in which chat, the first-use message). 1:1 chats are labeled
    by the other participant's contact name -- never the raw identifier.

    Scope matters here: dyad_terms is corpus-wide, so without a roster
    restriction this would resurface curated terms from chats the silent-
    contact guard (or --exclude) deliberately dropped -- and from bank/alert
    threads (a bank's shorthand for a savings balance). 1:1 chats must belong
    to a featured roster member; group chats pass (they're not what the guard
    protects, and their labels are group names). excluded_names drops terms
    that ARE an excluded person's name ('alice' coined in someone else's
    chat) -- a nickname card for the excluded defeats the exclusion."""
    clauses = ["d.llm_verdict IN ('inside_joke', 'nickname')", "d.n_uses >= ?"]
    params = [min_uses]
    for cid in exclude_chat_ids:
        clauses.append("d.chat_id != ?")
        params.append(cid)
    contact_ids = [e["contact_id"] for e in roster if e.get("contact_id") is not None]
    handle_ids = [e["handle_id"] for e in roster
                  if e.get("contact_id") is None and e.get("handle_id") is not None]
    member_clauses = []
    if contact_ids:
        member_clauses.append(
            f"h2.contact_id IN ({','.join('?' * len(contact_ids))})")
    if handle_ids:
        member_clauses.append(f"h2.id IN ({','.join('?' * len(handle_ids))})")
    if member_clauses:
        clauses.append(
            "(ch.is_group = 1 OR EXISTS ("
            "SELECT 1 FROM chat_participants cp2 "
            "JOIN handles h2 ON h2.id = cp2.handle_id "
            f"WHERE cp2.chat_id = d.chat_id AND ({' OR '.join(member_clauses)})))")
        params += contact_ids + handle_ids
    else:
        clauses.append("ch.is_group = 1")
    rows = con.execute(
        f"""
        SELECT d.term, d.n_uses, d.coined_is_me, d.llm_verdict,
               NULLIF(ch.display_name, '') AS display_name, ch.is_group,
               m.date AS first_date, m.text AS first_text,
               (SELECT {db.CONTACT_LABEL_SQL}
                FROM handles h LEFT JOIN contacts c ON c.id = h.contact_id
                WHERE h.id = d.coined_by_handle_id) AS coined_name,
               (SELECT {db.CONTACT_LABEL_SQL}
                FROM chat_participants cp
                JOIN handles h ON h.id = cp.handle_id
                LEFT JOIN contacts c ON c.id = h.contact_id
                WHERE cp.chat_id = d.chat_id
                ORDER BY h.id LIMIT 1) AS partner
        FROM dyad_terms d
        JOIN chats ch ON ch.id = d.chat_id
        LEFT JOIN messages m ON m.id = d.first_message_id
        WHERE {" AND ".join(clauses)}
        ORDER BY d.n_uses DESC LIMIT ?
        """,
        params + [limit * 2],
    ).fetchall()
    jokes, seen = [], set()
    for row in rows:
        if row["term"] in seen or row["term"].lower() in excluded_names:
            continue
        seen.add(row["term"])
        if row["is_group"]:
            with_label = row["display_name"] or "a group chat"
        else:
            with_label = row["display_name"] or row["partner"]
        jokes.append({
            "term": row["term"], "uses": row["n_uses"],
            "kind": row["llm_verdict"].replace("_", " "),
            "with": with_label,
            "coined_by": "Me" if row["coined_is_me"] else row["coined_name"],
            "first_used": row["first_date"], "quote": row["first_text"],
        })
        if len(jokes) >= limit:
            break
    return jokes


def _contact_jokes(trends, contact_name):
    """Inside jokes for the deep-dive: the contact-scoped curated
    distinctive terms, reshaped to the joke-card contract."""
    distinctive = trends.get("distinctive_terms") if trends else None
    if isinstance(distinctive, dict):
        terms = distinctive.get("terms") if _available(distinctive) else None
    else:
        terms = distinctive
    return [
        {"term": row.get("term"), "uses": row.get("n_uses"),
         "kind": str(row.get("verdict") or "").replace("_", " ") or None,
         "with": contact_name, "coined_by": row.get("coined_by"),
         "first_used": row.get("date"), "quote": row.get("text")}
        for row in terms or []
    ][:14]


def _sec_words(con, scope, dates, contact_entry, roster, exclude_chat_ids=(),
               excluded_names=frozenset()):
    # get_word_frequency is the one analytics tool that scopes by SENDER, not
    # by that person's chats: under a --contact scope it returns only THEIR
    # words, while every other key here covers both sides. Naming the key for
    # what it actually holds keeps the narrator from calling one side's
    # vocabulary "shared".
    excludes = {"exclude_chat_id": list(exclude_chat_ids)}
    freq, note_f = wrapped._tool("get_word_frequency", top=250,
                                 **scope, **dates, **excludes)
    trends, note_t = wrapped._tool("get_word_trends", **scope, **dates, **excludes)
    stats, narrate, visuals = {}, {}, []
    freq_rows = freq.get("rows") if isinstance(freq, dict) else freq
    words = []
    for row in freq_rows or []:
        word = str(row.get("ngram", row.get("word")) or "")
        # Skip everyday English plus URL debris/number tokens -- what's left
        # is the vocabulary that is actually this corpus's. Excluded persons'
        # names are skipped too: a cloud word naming them defeats the
        # exclusion just like a joke card would.
        if (word.lower() in COMMON_WORDS or word.lower() in excluded_names
                or not any(c.isalpha() for c in word)):
            continue
        words.append({"word": word, "count": row.get("count")})
    words = words[:18]
    if words:
        key = "their_top_words" if contact_entry else "top_words"
        narrate[key] = words
        cloud = wrapped_html.render_word_cloud(words)
        if cloud:
            visuals.append(wrapped_html.block(wrapped_html.humanize(key), cloud))
    if trends is not None:
        emoji = trends.get("emoji")
        top_emoji = None
        if isinstance(emoji, dict):
            if emoji.get("total_emoji"):
                stats["emoji_used"] = narrate["emoji_used"] = emoji["total_emoji"]
            top_emoji = emoji.get("top")
        elif emoji:
            top_emoji = emoji
        if top_emoji:
            narrate["top_emoji"] = top_emoji[:12]
            visuals.append(wrapped_html.block(
                "Top emoji", wrapped_html.render_emoji_row(top_emoji[:12])))
    if contact_entry is not None:
        jokes = _contact_jokes(trends, contact_entry["name"])
    else:
        jokes = _inside_jokes_all(con, roster, exclude_chat_ids, excluded_names)
    if jokes:
        narrate["inside_jokes"] = jokes
        visuals.append(wrapped_html.block(
            "Inside jokes", wrapped_html.render_joke_cards(jokes)))
    if not (stats or visuals):
        return {"title": "Words & Inside Jokes", "stats": None,
                "note": "; ".join(n for n in (note_f, note_t) if n)}
    scope_hint = ("This covers one relationship." if contact_entry else
                  "This covers ALL your chats -- never say 'this chat'.")
    return {
        "title": "Words & Inside Jokes",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": ("Top words exclude everyday English, so what's left is the "
                    "vocabulary that is actually yours. Inside jokes are curated "
                    "distinctive terms, counted over your whole history -- hover "
                    "one for its first use."),
        "hint": (f"{scope_hint} Mention two or three standout words or emoji, and "
                 "quote at most one inside joke (they are distinctive vocabulary "
                 "from these chats)."),
    }


def _group_label(row, members):
    """A group chat's display name, else the other members' names -- never
    the raw chat_identifier (an opaque 'chat<19-digit-id>' string)."""
    if row["display_name"]:
        return row["display_name"]
    names = [m.get("member") for m in members
             if m.get("member") and not m.get("is_me")]
    if not names:
        return str(row["chat_identifier"])
    if len(names) > 3:
        return f"{', '.join(names[:3])} + {len(names) - 3} more"
    return " & ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]


def _one_group(res):
    """Reshape and render one get_group_dynamics result. Returns
    (narrate_dict, visual_html)."""
    narrate, parts = {}, []
    members = [
        {"member": m.get("member"), "messages": m.get("messages"),
         "share": m.get("share"), "words": m.get("words")}
        for m in res.get("members") or []
    ]
    if members:
        narrate["members"] = members
        share = wrapped_html.render_share_bar(members)
        if share:
            parts.append(share)
    gini = (res.get("gini") or {}).get("gini")
    if gini is not None:
        narrate["balance_gini"] = gini
        parts.append(f'<p class="note">Balance (Gini, 0 = everyone equal): '
                     f"{wrapped_html.format_scalar(gini)}</p>")
    lurkers = res.get("lurkers")
    if lurkers:
        names = [m.get("member", m) if isinstance(m, dict) else m for m in lurkers]
        narrate["lurkers"] = names
        chips = "".join(f'<span class="chip">{wrapped_html.esc(str(n))}</span>'
                        for n in names)
        parts.append(wrapped_html.block(
            "Lurkers", f'<div class="chips">{chips}</div>', depth=1))
    leaderboard = (res.get("igniters") or {}).get("leaderboard")
    if leaderboard:
        starters = [
            {"member": row.get("member"), "started": row.get("sessions_started"),
             "share": row.get("share")}
            for row in leaderboard
        ]
        narrate["conversation_starters"] = starters
        chart = wrapped_html.render_hbars(starters, "member", "started", table=False)
        if chart:
            parts.append(wrapped_html.block("Conversation starters", chart, depth=1))
    pairs = (res.get("adjacency") or {}).get("pairs")
    if pairs:
        replies = [
            {"replier": row.get("replier"), "replies_to": row.get("replied_to"),
             "count": row.get("count")}
            for row in pairs[:6]
        ]
        narrate["who_replies_to_whom"] = replies
        parts.append(wrapped_html.block(
            "Who replies to whom",
            wrapped_html._rows_table(replies, wrapped_html._columns(replies)),
            depth=1))
    lsm = (res.get("group_lsm") or {}).get("members")
    if lsm:
        narrate["style_match"] = lsm
        parts.append(wrapped_html.block(
            "Style match (LSM)",
            wrapped_html._rows_table(lsm, wrapped_html._columns(lsm)),
            depth=1))
    return narrate, "".join(parts)


def _sec_groups(con, year, dates, exclude_chat_ids=()):
    narrate, visuals, group_notes = {}, [], []
    for row in _top_group_chats(con, year, exclude_chat_ids):
        res, note = wrapped._tool("get_group_dynamics", chat_id=row["chat_id"], **dates)
        if res is None:
            label = row["display_name"] or f"group chat {row['chat_id']}"
            group_notes.append(f"{label}: {note}")
            continue
        label = _group_label(row, res.get("members") or [])
        group_narrate, group_html = _one_group(res)
        narrate[label] = group_narrate
        visuals.append(wrapped_html.block(label, group_html))
    if not visuals:
        return {"title": "Groups", "stats": None,
                "note": "; ".join(group_notes[:2]) or "no group chats in scope"}
    return {
        "title": "Groups",
        "stats": {},
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": ("The bar is who talks how much. Style match is linguistic "
                    "style matching -- how closely each member's function-word "
                    "use tracks the group's."),
        "hint": ("Each top-level key is one group chat. Say who carries each group, "
                 "who starts the conversations ('started' counts conversations, not "
                 "messages), and one fun reply-pair fact. Don't recite whole "
                 "tables."),
    }


def _tone_by_period(con, contact_entry, year, exclude_chat_ids=()):
    """Average tone + leading emotion per period from chunk_annotations:
    months inside a report year, years for all-time. Direct SQL (the trends
    tool aggregates whole-scope); honors the deep-dive and chat exclusions.
    substr bucketing in GROUP BY over filtered rows is the allowed pattern
    (see the date-handling rule)."""
    width = 7 if year is not None else 4
    clauses, params = ["c.first_date IS NOT NULL"], []
    if year is not None:
        clauses += ["c.first_date >= ?", "c.first_date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("c.chat_id != ?")
        params.append(cid)
    if contact_entry is not None:
        chat_ids = _person_chat_ids(con, contact_entry)
        if not chat_ids:
            return []
        clauses.append(f"c.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += chat_ids
    where = " AND ".join(clauses)
    tone = {p: (t, n) for p, t, n in con.execute(
        f"""SELECT substr(c.first_date, 1, {width}) AS p, AVG(a.valence), COUNT(*)
            FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id
            WHERE {where} GROUP BY p""", params)}
    leader = {}
    for p, emotion, n in con.execute(
        f"""SELECT substr(c.first_date, 1, {width}) AS p, je.value, COUNT(*) AS n
            FROM chunk_annotations a JOIN chunks c ON c.id = a.chunk_id,
                 json_each(a.emotions_json) je
            WHERE {where} GROUP BY p, je.value ORDER BY n""", params):
        leader[p] = emotion  # last row per period wins = highest count
    rows = []
    for period in sorted(tone):
        avg, n = tone[period]
        rows.append({
            "period": period,
            "tone": round(avg, 3) if avg is not None else None,
            "top_emotion": str(leader.get(period, "")).capitalize() or None,
            "snippets": n,
        })
    return rows


def _sec_eras(con, scope, dates, contact_entry=None, year=None,
              exclude_chat_ids=()):
    """Phase-3 section: None (section omitted) unless the batch-LLM
    artifacts exist at all."""
    n_annotated = con.execute("SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]
    n_topics = con.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    if not (n_annotated or n_topics):
        return None
    arc, note = wrapped._tool(
        "get_word_trends", include=["emotions", "valence"], **scope, **dates,
        exclude_chat_id=list(exclude_chat_ids),
    )
    stats, narrate, visuals, n_windows = {}, {}, [], 0
    by_period = _tone_by_period(con, contact_entry, year, exclude_chat_ids)
    if len(by_period) > 1:
        narrate["by_period"] = by_period
        visuals.append(wrapped_html.block(
            "The arc", wrapped_html.render_tone_strip(by_period)))
    if arc is not None:
        emotions = arc.get("emotions") or {}
        n_windows = emotions.get("chunks_annotated") or 0
        if emotions.get("top"):
            rows = [
                {"emotion": str(row.get("emotion", "")).capitalize(),
                 "snippets": row.get("count")}
                for row in emotions["top"]
            ]
            narrate["top_emotions"] = rows
            chart = wrapped_html.render_hbars(rows, "emotion", "snippets")
            if chart:
                visuals.append(wrapped_html.block("Top emotions", chart))
        valence = (arc.get("valence") or {}).get("series") or []
        if len(valence) == 1:
            stats["average_tone"] = narrate["average_tone"] = \
                valence[0].get("mean_valence")
        elif valence:
            tone = [{"period": row.get("period"), "tone": row.get("mean_valence")}
                    for row in valence]
            narrate["tone_over_time"] = tone
            stats["tone_over_time"] = tone
    if not (stats or visuals):
        return {"title": "Eras & Emotional Arc", "stats": None, "note": note}
    caption = None
    if n_windows:
        caption = (f"A local model read {n_windows:,} conversation snippets and "
                   "tagged the feelings it saw -- a vibe check, not a diagnosis.")
    return {
        "title": "Eras & Emotional Arc",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "note": note,
        "caption": caption,
        "hint": ("Tone runs from −1 (rough) to +1 (glowing); snippets are short "
                 "stretches of conversation; by_period is the arc over time. Name "
                 "the two or three feelings that defined the year, and one turn "
                 "in the arc if there is one (a month whose leading feeling or "
                 "tone stands out)."),
    }


def build_sections(con, roster, contact_entry=None, year=None,
                   exclude_chat_ids=(), exclude_ranges=(),
                   excluded_names=frozenset(), wellbeing=False):
    """Build the ordered list of report sections. roster drives the default
    multi-contact report; contact_entry switches to the single-relationship
    deep-dive (people/groups sections are dropped -- they describe the whole
    corpus). exclude_chat_ids (already expanded by wrapped.main to cover
    excluded persons' 1:1 chats) reaches every section -- the direct-SQL ones
    and, via each tool's exclude_chat_id kwarg, the corpus-wide tool calls;
    it is also forwarded to the prior-year roster so previous_rank is
    computed on the same basis as the current ranks. excluded_names keeps
    terms named after an excluded person out of the joke cards and the
    person out of prior-year ranks."""
    dates = {}
    if year is not None:
        dates = {"date_from": f"{year:04d}-01-01", "date_to": f"{year:04d}-12-31"}
    scope = wrapped._entry_scope(contact_entry) if contact_entry else {}
    excludes = {"exclude_chat_id": list(exclude_chat_ids)}
    tempo_entries = [contact_entry] if contact_entry else roster

    # Year in Numbers + (later) Late Nights share one get_activity_patterns call.
    patterns, patterns_note = wrapped._tool("get_activity_patterns",
                                            **scope, **dates, **excludes)

    sections = [_sec_year_in_numbers(patterns, patterns_note, year)]
    if contact_entry is None:
        sections.append(_sec_your_people(con, roster, year, exclude_chat_ids,
                                         exclude_ranges, excluded_names))
    tempo, ghosts = _sec_tempo(tempo_entries, dates)
    sections.append(tempo)
    if ghosts is not None:
        sections.append(ghosts)
    marathons = _sec_marathons(con, contact_entry, year, exclude_chat_ids)
    if marathons is not None:
        sections.append(marathons)
    sections.append(_sec_reactions(scope, dates, excludes))
    sections.append(_sec_late_nights(patterns, patterns_note))
    sections.append(_sec_words(con, scope, dates, contact_entry, roster,
                               exclude_chat_ids, excluded_names))
    if contact_entry is None:
        sections.append(_sec_groups(con, year, dates, exclude_chat_ids))
    eras = _sec_eras(con, scope, dates, contact_entry, year, exclude_chat_ids)
    if eras is not None:
        sections.append(eras)
    # --- feature: drift ---
    # D4 (RES-66): opt-in, corpus-wide "Who's Drifting" surface. The builder
    # self-gates on config.DRIFT_ENABLED (default off); like Your People and
    # Groups it is a whole-corpus section, so it is dropped in the deep-dive.
    if contact_entry is None:
        drifting = _sec_drifting(con, excluded_names)
        if drifting is not None:
            sections.append(drifting)
    # --- end feature: drift ---
    sections.append(_sec_voiceprint(roster, contact_entry))  # feature: persona
    # --- feature: conflict ---  (C5 / RES-54; builder appended at EOF)
    sections += [s for s in [_sec_repair_report(
        con, contact_entry, year, exclude_chat_ids)] if s is not None]
    # --- end feature: conflict ---
    # --- feature: wellbeing ---
    if wellbeing:  # opt-in only (RES-68 / W4); omitted unless a window fires
        wb = _sec_wellbeing()
        if wb is not None:
            sections.append(wb)
    # --- end feature: wellbeing ---
    # --- feature: nudges ---
    if contact_entry is None:
        sections.append(_sec_open_loops(exclude_chat_ids, excluded_names))
    # --- end feature: nudges ---
    return sections


# --- feature: drift ---
def _sec_drifting(con, excluded_names=frozenset(), limit=8):
    """Opt-in "Who's Drifting" section (RES-66 / D4) -- a reflective surface,
    OFF by default. Renders ONLY when config.DRIFT_ENABLED, and only for
    phase_corroborated fading/dormant ties: a decline label that an
    intra-iMessage corroborator (reciprocity erosion, or reply-time inflation)
    actually backs. Every number is read verbatim from the drift model's
    dyad_drift table -- no re-scoring, no Ollama -- so the narrator only
    narrates already-computed values. Returns None (section omitted) when
    drift is disabled, the table is absent (a pre-feature DB), or nothing
    corroborated survives the report's person exclusions.

    Split bars contrast the two corroborating signals' magnitude per tie;
    open reconnection-window chips flag ties still inside the window before the
    drift model would count them dormant -- the same framing as
    query_tools.drift._reconnection_window, reading config.DRIFT_PHASE_THRESHOLDS
    at call time (the repo convention). excluded_names keeps a person the
    report deliberately dropped (a named --exclude, or the silent-contact
    guard) out of this section too: a 'who's drifting' card for an excluded ex
    would defeat the exclusion."""
    import json  # function-local: keeps this additive block off the top imports
    if not config.DRIFT_ENABLED:
        return None
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dyad_drift'"
    ).fetchone():
        return None
    rows = con.execute(
        """
        SELECT contact_key, label, phase, drift_score,
               reciprocity_value, latency_value, coverage_json,
               last_month, confidence_tier, n_messages
        FROM dyad_drift
        WHERE phase_corroborated = 1 AND phase IN ('fading', 'dormant')
        ORDER BY drift_score DESC
        """
    ).fetchall()

    dormant_days = float(config.DRIFT_PHASE_THRESHOLDS.get("dormant_days", 90))
    per_tie, windows, seen = [], [], set()
    for row in rows:
        label = row["label"] or row["contact_key"]
        if str(label).lower() in excluded_names or row["contact_key"] in seen:
            continue
        seen.add(row["contact_key"])
        try:
            coverage = json.loads(row["coverage_json"] or "{}")
        except (TypeError, ValueError):
            coverage = {}
        gap_days = coverage.get("last_message_days")
        per_tie.append({
            "contact": label,
            "reciprocity_erosion": round(row["reciprocity_value"] or 0.0, 3),
            "latency_inflation": round(row["latency_value"] or 0.0, 3),
            "phase": row["phase"],
            "quiet_for_days": gap_days,
            "confidence": row["confidence_tier"],
        })
        # Days left before the dormant threshold, from the stored gap -- same
        # open/closing/overdue framing as query_tools.drift._reconnection_window.
        if isinstance(gap_days, (int, float)) and not isinstance(gap_days, bool):
            days_until = round(dormant_days - gap_days)
            if row["phase"] == "dormant" or days_until <= 0:
                status = "overdue"
            elif days_until <= 14:
                status = "closing"
            else:
                status = "open"
            if status in ("open", "closing"):
                windows.append({"contact": label,
                                "days_until_dormant": days_until,
                                "status": status})
        if len(per_tie) >= limit:
            break

    if not per_tie:
        return None

    split = wrapped_html.render_split_bars(
        per_tie, "contact", "reciprocity_erosion", "latency_inflation",
        left_label="Reciprocity erosion", right_label="Reply-time drift")
    if split is None:  # nothing chartable -- degrade to the table view
        split = wrapped_html._rows_table(per_tie, wrapped_html._columns(per_tie))
    visuals = [wrapped_html.block("Cooling ties", split)]
    chips = wrapped_html.render_window_chips(windows)
    if chips:
        visuals.append(wrapped_html.block("Still time to reach out", chips))

    return {
        "title": "Who's Drifting",
        "stats": {},
        "narrate_stats": {"cooling_ties": per_tie,
                          "reconnection_windows": windows},
        "extra_html": "".join(visuals),
        "caption": ("iMessage only -- a thread that has gone quiet here may be "
                    "loud on calls, in person, or elsewhere. Only ties whose "
                    "cooling is corroborated by an eroding back-and-forth or "
                    "stretching reply times appear; both magnitudes are read "
                    "from the drift model, never re-scored."),
        "hint": ("A gentle, reflective look at 1:1 ties that have cooled on "
                 "iMessage ONLY -- never diagnose a relationship or give advice. "
                 "Reciprocity erosion is how lopsided the sending became; "
                 "reply-time drift is how much slower replies got (both are 0-1 "
                 "magnitudes). A reconnection window is the days left before a "
                 "tie is counted dormant. Note softly that iMessage is one "
                 "channel, mention at most one or two ties, and quote the "
                 "numbers exactly as given."),
    }
# --- end feature: drift ---


# --- feature: persona ---
def _sec_voiceprint(roster, contact_entry=None):
    """"Your Voiceprint" (RES-64 / V4): the reader's OWN deterministic style
    voiceprint -- the ("me", "*", "all") card from get_voiceprint -- rendered in
    iMessage bubbles, plus a consent-gated "guess who texts like this" game.

    PRIVACY (third-party content is default-redacted): a contact's verbatim
    signature phrases and stylometric card enter the generated file ONLY when
    that contact's `contacts.id` is in WRAPPED_VOICEPRINT_CONSENT_IDS (parsed by
    consent.py -- never re-implemented here). With the default empty list the
    section is self-only and the game is ABSENT. Consent ADDS a contact; it never
    overrides a roster exclusion -- the game iterates the already-exclusion-
    filtered featured roster (the silent-contact guard, --contact, and --exclude
    have all run before build_sections), so a consented-but-excluded contact is
    simply not present to feature. Nothing is ever sent; this only renders."""
    import consent

    me_card, note = wrapped._tool("get_voiceprint")  # ("me", "*", "all")
    if not (isinstance(me_card, dict) and "features" in me_card):
        reason = note or (me_card.get("note") if isinstance(me_card, dict) else None)
        return {"title": "Your Voiceprint", "stats": None,
                "note": reason or "no voiceprint on file yet -- run `make voiceprint`"}

    feats = me_card.get("features") or {}
    stats = {}
    if feats.get("avg_msg_len") is not None:
        stats["typical_message_length"] = f"{round(feats['avg_msg_len']):,} chars"
    if feats.get("lowercase_ratio") is not None:
        stats["lowercase_share"] = feats["lowercase_ratio"]  # 'share' -> percent
    if feats.get("burst_size") is not None:
        stats["typical_burst"] = f"{round(feats['burst_size']):,} messages"
    if me_card.get("confidence"):
        stats["confidence"] = me_card["confidence"]

    narrate = {"you": {
        "openers": wrapped_html._voiceprint_phrases(me_card.get("openers")),
        "closers": wrapped_html._voiceprint_phrases(me_card.get("closers")),
        "typical_message_length_chars": (round(feats["avg_msg_len"])
                                         if feats.get("avg_msg_len") is not None else None),
    }}

    visuals = []
    you = wrapped_html.render_voiceprint_card(
        me_card.get("openers"), me_card.get("closers"), outgoing=True)
    if you:
        visuals.append(wrapped_html.block("This is you", you))

    # Consent-gated third-party game. Default-redacted: a contact's card is
    # rendered ONLY when opted in AND present in the featured roster (which the
    # exclusions already filtered), so consent can never resurface an excluded
    # contact. Empty consent list -> the loop bodies never run -> no game.
    consent_raw = getattr(config, "WRAPPED_VOICEPRINT_CONSENT_IDS", "")
    mystery = []
    for entry in roster:
        cid = entry.get("contact_id")
        if cid is None or not consent.is_consented(cid, consent_raw):
            continue
        card, _card_note = wrapped._tool("get_voiceprint", contact_id=cid,
                                         subject="them", scope="all")
        if not (isinstance(card, dict) and "features" in card):
            continue
        clue = wrapped_html.render_voiceprint_card(
            card.get("openers"), card.get("closers"),
            outgoing=False, reveal=entry.get("name"))
        if clue:
            mystery.append(wrapped_html.block(f"Mystery #{len(mystery) + 1}",
                                              clue, depth=1))
    if mystery:
        visuals.append(wrapped_html.block("Guess Who Texts Like This?",
                                          "".join(mystery)))
        narrate["guess_who"] = {"players": len(mystery)}

    if not (stats or visuals):
        return {"title": "Your Voiceprint", "stats": None,
                "note": "no voiceprint on file yet"}
    caption = ("Your voiceprint is a deterministic fingerprint of how you text -- "
               "message length, casing, and the phrases you open and close with.")
    if mystery:
        caption += (" The guess-who game features only friends who explicitly "
                    "opted in; open a card to reveal who it is.")
    return {
        "title": "Your Voiceprint",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": caption,
        "hint": ("This is the reader's own texting style -- mention a couple of "
                 "signature traits (a favorite opener, message length, a lowercase "
                 "habit) warmly. If there is a guess-who game, tease it WITHOUT "
                 "naming any of the players."),
    }
# --- end feature: persona ---


# --- feature: conflict ---
# C5 (RES-54): the Wrapped "Repair Report" section. Deterministic aggregates
# over the conflict finder's tables (conflicts / conflict_signals, RES-46 / C1);
# the LLM only narrates the numbers, it never computes them.
#
# EVERY repair-derived stat is eval-gated (RES-70 / F4): a label class appears
# only when eval_gate_passed(con, 'conflict_labels', <class>) clears it, so a
# below-kappa class is SUPPRESSED, not softened. Gated stats:
#   * personal most-common horseman -- computed over ONLY the gate-cleared
#     Gottman horsemen, so the class it surfaces always cleared kappa;
#   * most-repaired relationship AND the recovery-time trend -- BOTH ride the
#     one 'repair' gate; if 'repair' hasn't cleared, BOTH vanish entirely.
# Recovery time is a resolution claim, so off_text_flag=1 windows (a large
# intra-window gap forbids a resolution verdict) are excluded from it.
# The deterministic-finder facts (window counts, cool-off gap, volume collapse)
# are NOT label-derived and are never gated -- they are the fallback the section
# renders when no class clears; with no conflict windows at all the builder
# returns None (skipped), never empty scaffolding.
import json  # noqa: E402 (local to the additive C5 block)

_CONFLICT_GATE_TASK = "conflict_labels"
_HORSEMEN = ("criticism", "contempt", "defensiveness", "stonewalling")
_HORSEMAN_LABELS = {
    "criticism": "Criticism", "contempt": "Contempt",
    "defensiveness": "Defensiveness", "stonewalling": "Stonewalling",
}


def _horseman_label(cls):
    return _HORSEMAN_LABELS.get(cls, str(cls).replace("_", " ").capitalize())


def _conflict_class_cleared(con, cls):
    """Per-class eval clearance for the conflict_labels gate (RES-70 / F4). A
    thin module-level seam: the import stays lazy (only runs once the section
    has conflicts to report) and tests can patch this one point to drive the
    suppressed path. eval_gate_passed treats an absent/failing verdict as
    False, so an uncleared class is suppressed."""
    from eval.evalkit.gates import eval_gate_passed
    return eval_gate_passed(con, _CONFLICT_GATE_TASK, cls)


def _conflict_tables_present(con):
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


def _conflict_scope(con, contact_entry, year, exclude_chat_ids):
    """(where, params) restricting `conflicts` (aliased `cf`) to the report
    scope: the report year (on first_date), the deep-dive contact's chats, and
    the excluded chats. Columns are qualified `cf.*` so the clause drops
    unchanged into the join queries. Returns None when a contact scope resolves
    to no chats. Plain-string date compares only (see the CLAUDE.md gotcha)."""
    clauses, params = ["1 = 1"], []
    if year is not None:
        clauses += ["cf.first_date >= ?", "cf.first_date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("cf.chat_id != ?")
        params.append(cid)
    if contact_entry is not None:
        chat_ids = _person_chat_ids(con, contact_entry)
        if not chat_ids:
            return None
        clauses.append(f"cf.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += chat_ids
    return " AND ".join(clauses), params


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _conflict_base(con, where, params, period_width):
    """Deterministic finder facts (NOT label-derived, never gated): the number
    of conflict windows in scope, the typical cool-off gap (median peak reply
    gap), how many windows saw a volume collapse, and the window count per
    period. None when scope has no windows."""
    rows = con.execute(
        f"SELECT cf.first_date, cf.peak_signal_json FROM conflicts cf "
        f"WHERE {where}", params).fetchall()
    if not rows:
        return None
    gaps, collapses, by_period = [], 0, {}
    for row in rows:
        try:
            peak = json.loads(row["peak_signal_json"]) if row["peak_signal_json"] else {}
        except (TypeError, ValueError):
            peak = {}
        gap = peak.get("peak_gap_hours")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool):
            gaps.append(float(gap))
        if "volume_collapse" in (peak.get("signals") or []):
            collapses += 1
        period = (row["first_date"] or "")[:period_width]
        if period:
            by_period[period] = by_period.get(period, 0) + 1
    return {
        "conflict_windows": len(rows),
        "typical_gap_hours": _median(gaps) if gaps else None,
        "volume_collapses": collapses,
        "windows_by_period": [{"period": p, "windows": by_period[p]}
                              for p in sorted(by_period)],
    }


def _personal_horseman(con, where, params):
    """The reader's own most-common Gottman horseman (actor_is_me = 1),
    computed over ONLY the horseman classes that cleared the eval gate -- so
    the surfaced class always cleared kappa. None when no cleared horseman
    fired in scope."""
    cleared = [h for h in _HORSEMEN if _conflict_class_cleared(con, h)]
    if not cleared:
        return None
    placeholders = ",".join("?" * len(cleared))
    rows = con.execute(
        f"SELECT cs.label_class AS label_class, COUNT(*) AS n "
        f"FROM conflict_signals cs JOIN conflicts cf ON cf.id = cs.conflict_id "
        f"WHERE {where} AND cs.actor_is_me = 1 "
        f"AND cs.label_class IN ({placeholders}) "
        f"GROUP BY cs.label_class ORDER BY n DESC, cs.label_class",
        params + cleared).fetchall()
    if not rows:
        return None
    breakdown = [{"horseman": _horseman_label(r["label_class"]), "your_turns": r["n"]}
                 for r in rows]
    return {"horseman": breakdown[0]["horseman"],
            "your_turns": breakdown[0]["your_turns"], "breakdown": breakdown}


def _most_repaired(con, where, params):
    """The relationship with the most repair turns in scope (either side), by
    chat, resolved to the counterpart's contact label. Caller gates this on the
    'repair' class clearing. None when no repair turns in scope."""
    row = con.execute(
        f"SELECT cf.chat_id AS chat_id, COUNT(*) AS n, "
        f"NULLIF(ch.display_name, '') AS display_name, "
        f"(SELECT {db.CONTACT_LABEL_SQL} FROM chat_participants cp "
        f"JOIN handles h ON h.id = cp.handle_id "
        f"LEFT JOIN contacts c ON c.id = h.contact_id "
        f"WHERE cp.chat_id = cf.chat_id ORDER BY h.id LIMIT 1) AS partner "
        f"FROM conflict_signals cs JOIN conflicts cf ON cf.id = cs.conflict_id "
        f"JOIN chats ch ON ch.id = cf.chat_id "
        f"WHERE {where} AND cs.label_class = 'repair' "
        f"GROUP BY cf.chat_id ORDER BY n DESC, cf.chat_id LIMIT 1",
        params).fetchone()
    if row is None:
        return None
    label = row["display_name"] or row["partner"] or f"chat {row['chat_id']}"
    return {"with": label, "repair_turns": row["n"]}


def _recovery_trend(con, where, params, period_width):
    """Average recovery time per period for repaired, RESOLVABLE conflicts:
    windows that (a) have off_text_flag = 0 -- off-text forbids a resolution
    verdict, so those windows are EXCLUDED -- AND (b) contain at least one
    'repair' turn (a conflict with no repair has no recovery to time). Recovery
    time is the window's own span (first_date -> last_date). Caller gates this
    on the 'repair' class clearing."""
    rows = con.execute(
        f"SELECT cf.first_date, cf.last_date FROM conflicts cf "
        f"WHERE {where} AND cf.off_text_flag = 0 AND EXISTS ("
        f"SELECT 1 FROM conflict_signals cs "
        f"WHERE cs.conflict_id = cf.id AND cs.label_class = 'repair')",
        params).fetchall()
    buckets = {}
    for row in rows:
        dur = _session_duration(row["first_date"], row["last_date"])
        if dur is None:
            continue
        period = (row["first_date"] or "")[:period_width]
        if period:
            buckets.setdefault(period, []).append(dur)
    return [{"period": p, "recovery_seconds": round(sum(v) / len(v)),
             "conflicts": len(v)} for p, v in sorted(buckets.items())]


def _sec_repair_report(con, contact_entry=None, year=None, exclude_chat_ids=()):
    """The Repair Report: deterministic conflict aggregates, LLM-narrated. The
    deterministic-finder facts always render (the fallback); the label-derived
    stats (horseman, most-repaired relationship, recovery-time trend) appear
    only for the eval-cleared classes. None (skipped) when there is no conflict
    data in scope -- never empty scaffolding."""
    if not _conflict_tables_present(con):
        return None
    scope = _conflict_scope(con, contact_entry, year, exclude_chat_ids)
    if scope is None:
        return None
    where, params = scope
    period_width = 7 if year is not None else 4
    base = _conflict_base(con, where, params, period_width)
    if base is None:  # no conflict windows in scope -> skip
        return None

    stats, narrate, visuals = {}, {}, []
    # Deterministic-finder fallback (never gated): always survives here.
    stats["conflict_windows"] = narrate["conflict_windows"] = base["conflict_windows"]
    if base["typical_gap_hours"] is not None:
        secs = round(base["typical_gap_hours"] * 3600)
        stats["typical_cool_off_seconds"] = secs
        narrate["typical_cool_off"] = wrapped_html.format_duration(secs)
    if base["volume_collapses"]:
        narrate["windows_with_volume_collapse"] = base["volume_collapses"]
    if len(base["windows_by_period"]) > 1:
        narrate["windows_by_period"] = base["windows_by_period"]
        chart = wrapped_html.render_hbars(
            base["windows_by_period"], "period", "windows", table=False)
        if chart:
            visuals.append(wrapped_html.block("Conflict windows over time", chart))

    # Gated stat 1: personal most-common horseman (per-class gate; the helper
    # already restricts to gate-cleared classes).
    horseman = _personal_horseman(con, where, params)
    if horseman is not None:
        stats["your_top_horseman"] = horseman["horseman"]
        narrate["your_most_common_horseman"] = horseman
        if len(horseman["breakdown"]) > 1:
            chart = wrapped_html.render_hbars(
                horseman["breakdown"], "horseman", "your_turns")
            if chart:
                visuals.append(wrapped_html.block("Your escalation moves", chart))

    # Gated stats 2 & 3: most-repaired relationship AND recovery-time trend both
    # ride the single 'repair' gate -- if it has not cleared, BOTH are omitted.
    if _conflict_class_cleared(con, "repair"):
        repaired = _most_repaired(con, where, params)
        if repaired is not None:
            stats["most_repaired_relationship"] = repaired["with"]
            narrate["most_repaired_relationship"] = repaired
        trend = _recovery_trend(con, where, params, period_width)
        if len(trend) > 1:
            narrate["recovery_time_trend"] = trend
            chart = wrapped_html.render_hbars(
                trend, "period", "recovery_seconds", table=False)
            if chart:
                visuals.append(wrapped_html.block("Recovery time over time", chart))
        elif trend:
            # One period is not a trend, but its recovery time is still a real,
            # gated number worth stating.
            narrate["typical_recovery"] = wrapped_html.format_duration(
                trend[0]["recovery_seconds"])

    return {
        "title": "Repair Report",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": ("Conflict windows are tense stretches a local model flagged "
                    "from deterministic signals (reply-latency withdrawal, "
                    "one-sided sessions, volume collapse) -- a read, not a "
                    "verdict. Escalation and repair labels appear only where "
                    "they cleared an agreement check, and off-the-text stretches "
                    "are left out of recovery times."),
        "hint": ("Conflict windows are deterministic tense-stretch flags, never "
                 "certainties -- describe them gently and never assign blame or "
                 "name a 'winner'. The horseman is the reader's OWN most-common "
                 "escalation move in their turns. The most-repaired relationship "
                 "and recovery times appear only when those labels passed their "
                 "gate; recovery time is how long a repaired window lasted. Keep "
                 "it warm and non-judgmental, and name at most one relationship."),
    }
# --- end feature: conflict ---


# --- feature: wellbeing ---
# W4 (RES-68): the opt-in wellbeing trajectory section. Built ONLY when
# build_sections is called with wellbeing=True (wrapped.py resolves the
# --wellbeing flag / WRAPPED_WELLBEING config), and even then it renders only
# when get_language_trajectory reports a flagged, sustained cross-family shift
# in the user's OWN language -- otherwise it returns None and the section is
# dropped. It is deliberately self-scoped (the user's history, pooled across
# chats): per-contact trajectories are consent-gated and are not surfaced in a
# shareable keepsake. Two safety invariants, both owned by CODE not the model:
#   * the fixed support footer (wellbeing.support_block) is rendered into the
#     section HTML, so it ships on every flagged window regardless of narration;
#   * every builder-authored string here passes wellbeing.sanitize_condition_names
#     before it can render, and the LLM blurb is scrubbed the same way in
#     wrapped.compose_report (via the sanitize_narrative flag below).

# Internal marker name -> a plain, non-clinical reader label. Never names a
# condition; describes the behavior a marker measures.
_WELLBEING_MARKER_LABELS = {
    "i_count": "“I” focus",
    "absolutist_count": "Absolute words",
    "fw_negation_rate": "Negations",
    "late_night_share": "Late-night messaging",
    "initiation_share": "Reaching out first",
    "laugh_rate": "Laughing",
    "emoji_rate": "Emoji",
}


def _sec_wellbeing():
    """Self trajectory with the flagged window shaded, plus the code-emitted
    support footer. Returns None (section omitted) when the tool is unavailable,
    below the data floor, or reports no sustained departure."""
    import wellbeing  # dependency-free; imported locally to keep the top clean

    result, _note = wrapped._tool("get_language_trajectory")
    if result is None:
        return None
    windows = result.get("windows") or []
    if not windows:  # no sustained cross-family shift -> nothing to shade
        return None
    window = windows[0]
    start, end = window.get("start"), window.get("end")
    if not (start and end):
        return None

    series_map = result.get("series") or {}
    departed = set(window.get("departed_families") or [])
    ordered = list(wellbeing.MARKERS)
    # Prefer the markers whose families actually departed; fall back to whatever
    # series exist so the chart is never empty when a window fired.
    chosen = [name for name in ordered
              if wellbeing.MARKERS[name][0] in departed and series_map.get(name)]
    if not chosen:
        chosen = [name for name in ordered if series_map.get(name)]
    markers = [{"label": _WELLBEING_MARKER_LABELS.get(name, name),
                "points": series_map[name]} for name in chosen]

    def clean(text):
        return wellbeing.sanitize_condition_names(text or "")[0]

    span = f"{wrapped_html.display_scalar(start)} to {wrapped_html.display_scalar(end)}"
    framing = clean(window.get("framing")
                    or "A stretch where several of your own signals moved together.")
    caption = clean(
        "This looks back at how your own texting shifted over your history; the "
        "shaded stretch is where several signals moved together at once. It is a "
        "gentle heads-up, never a diagnosis.")
    support = wellbeing.support_block()  # fixed, code-emitted, never the model's

    return {
        "title": "A Gentle Check-In",
        "stats": {},
        "narrate_stats": {
            "the_stretch": span,
            "what_this_is": framing,
            "signals_that_shifted": [m["label"] for m in markers],
        },
        "extra_html": wrapped_html.render_wellbeing(markers, (start, end), support),
        "caption": caption,
        "sanitize_narrative": True,  # wrapped.compose_report scrubs the LLM blurb
        "hint": clean(
            "This is a sensitive, opt-in section about the reader's own texting "
            "over time. Write two or three warm, plain second-person sentences. "
            "Name the stretch (the_stretch) and that a few signals shifted "
            "together -- never diagnose, never name any condition, never give "
            "medical advice, and stay gentle and non-alarming. Support "
            "information is shown separately below your blurb."),
    }
# --- end feature: wellbeing ---


# --- feature: nudges ---
# Human-facing labels for each loose-thread kind, so the board reads as a
# keepsake and never leaks a detector's snake_case name. Unknown kinds fall
# back to a de-underscored form.
_OPEN_LOOP_LABELS = {
    "faded_initiator": "they used to reach out first",
    "dormant_tie": "gone quiet",
    "unanswered_question": "a question left hanging",
    "inferred_occasion": "a date coming back around",
    "dropped_plan": "a plan that fizzled",
    "unreciprocated_goodnews": "good news that went unanswered",
}


def _sec_open_loops(exclude_chat_ids=(), excluded_names=frozenset()):
    """Open Loops board: the current loose-thread nudge digest rendered as a
    keepsake. Composes from get_loose_threads (already post-Stage-0, so muted
    nudge_exclusions contacts never appear) and ADDITIONALLY drops any contact
    Wrapped itself excluded -- the silent-contact roster guard or --exclude,
    expanded to their 1:1 chats in wrapped.main, plus the lowercase-name
    fallback for unsaved handles -- so an excluded person can never resurface
    here. The dormant-tie batch-hide notice (a non-thread marker) is skipped."""
    threads, note = wrapped._tool("get_loose_threads", status="open",
                                  limit=config.NUDGE_MAX_RESULTS)
    if threads is None:
        return {"title": "Open Loops", "stats": None, "note": note}
    exclude_chat_ids = set(exclude_chat_ids)
    rows = []
    for t in threads:
        if not isinstance(t, dict) or "notice" in t:
            continue  # the dormant-tie batch-hide notice is not a board item
        if t.get("chat_id") in exclude_chat_ids:
            continue
        name = t.get("contact")
        if name and str(name).lower() in excluded_names:
            continue
        rows.append({
            "who": name or "Someone",
            "kind": _OPEN_LOOP_LABELS.get(
                t.get("kind"), str(t.get("kind") or "").replace("_", " ")),
            "nudge": t.get("nudge_text"),
            "since": t.get("anchor_date"),
            "due": t.get("due_by"),
        })
    if not rows:
        return {"title": "Open Loops", "stats": None,
                "note": "no open loops in scope"}
    return {
        "title": "Open Loops",
        "stats": {"open_loops": len(rows)},
        "narrate_stats": {"open_loops": rows},
        "extra_html": wrapped_html.render_open_loops(rows),
        "caption": ("Threads you let drift -- someone who went quiet, a question "
                    "left hanging, a birthday coming back around. Muted contacts "
                    "and anyone kept out of this report never appear here."),
        "hint": ("Each item is one open loop: the person, why it's open, and for "
                 "a recurring occasion when it next comes up. Nudge gently toward "
                 "one or two; never scold, and don't list them all."),
    }
# --- end feature: nudges ---
