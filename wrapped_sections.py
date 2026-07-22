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


# The feature-suite sections were split into wrapped_sections_suite; re-export
# them so build_sections (above) and the tests keep resolving
# wrapped_sections._sec_*. Bottom import: the suite module imports the core
# helpers defined above, so this must run after they are defined.
from wrapped_sections_suite import (  # noqa: E402
    _sec_drifting,
    _sec_open_loops,
    _sec_repair_report,
    _sec_voiceprint,
    _sec_wellbeing,
)

# Re-exported so tests can patch wrapped_sections._conflict_class_cleared and the
# suite's _sec_repair_report (which looks it up via this module) honors the
# patch; not otherwise referenced in this file.
from wrapped_sections_suite import _conflict_class_cleared  # noqa: E402,F401
