"""The Wrapped report's section builders: build_sections assembles the
ordered list of report sections, each `_sec_*` builder computing one
section's stats dict from a single analytics tool call. Split out of
wrapped.py, which keeps the CLI, roster/exclusion logic, rendering, and the
`_tool` / scope helpers these builders resolve late (module-attribute
access), so the two modules can import each other safely."""

import wrapped
import progress


def _sum_series(series, key):
    return sum(int(row.get(key) or 0) for row in series if isinstance(row, dict))


def _your_people_stats(con, roster, year, exclude_chat_ids=(), exclude_ranges=()):
    """Roster ranks with, when prior years exist, each contact's rank in the
    previous year's roster. The prior-year roster is built with the SAME
    exclusions as the current one -- otherwise an excluded chat inflates the
    ranks a 'previous_rank' is compared against."""
    prev_rank = {}
    if year is not None and con.execute(
        "SELECT 1 FROM messages WHERE date < ? LIMIT 1", (f"{year:04d}-01-01",)
    ).fetchone():
        prev = wrapped.build_roster(con, year - 1, exclude_chat_ids, exclude_ranges,
                                    limit=max(50, len(roster)))
        prev_rank = {wrapped._entry_key(e): i + 1 for i, e in enumerate(prev)}
    people = []
    for i, entry in enumerate(roster, 1):
        row = {"rank": i, "name": entry["name"], "messages": entry["n_messages"]}
        pr = prev_rank.get(wrapped._entry_key(entry))
        if pr is not None:
            row["previous_rank"] = pr
        people.append(row)
    return {"people": people}


def _top_group_chats(con, year, limit=3):
    clauses = ["ch.is_group = 1", "m.date IS NOT NULL", wrapped._REAL_MSG]
    params = []
    if year is not None:
        clauses += ["m.date >= ?", "m.date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    return con.execute(
        f"""
        SELECT m.chat_id,
               COALESCE(NULLIF(ch.display_name, ''), ch.chat_identifier) AS label,
               COUNT(*) AS n_messages
        FROM messages m JOIN chats ch ON ch.id = m.chat_id
        WHERE {" AND ".join(clauses)}
        GROUP BY m.chat_id ORDER BY n_messages DESC LIMIT ?
        """,
        params + [limit],
    ).fetchall()


def _sec_year_in_numbers(patterns, patterns_note):
    """Built from the get_activity_patterns result shared with
    _sec_late_nights (one tool call serves both sections)."""
    if patterns is None:
        return {"title": "Year in Numbers", "stats": None, "note": patterns_note}
    series = patterns.get("volume_series") or []
    if isinstance(series, dict):  # real tool wraps rows in {granularity, rows}
        series = series.get("rows") or []
    return {
        "title": "Year in Numbers",
        "stats": {
            "messages_sent": _sum_series(series, "sent"),
            "messages_received": _sum_series(series, "received"),
            "late_night": patterns.get("late_night"),
            "volume_series": series,
        },
    }


def _sec_tempo(tempo_entries, dates):
    per_contact, tempo_notes = {}, []
    # Generic label: the roster's names must not reach stdout here.
    with progress.track("Analyzing tempo", total=len(tempo_entries)) as p:
        for entry in tempo_entries:
            res, note = wrapped._tool("get_response_dynamics",
                                      **wrapped._entry_scope(entry), **dates)
            if res is None:
                tempo_notes.append(f"{entry['name']}: {note}")
                p.advance()
                continue
            picked = {k: res[k] for k in ("reply_latency", "initiation", "sessions")
                      if k in res}
            per_contact[entry["name"]] = picked or res
            p.advance()
    return {"title": "Tempo",
            "stats": {"per_contact": per_contact} if per_contact else None,
            "note": "; ".join(tempo_notes[:2]) or "no contacts in scope"}


def _sec_reactions(scope, dates):
    reactions, note = wrapped._tool("get_reaction_stats", **scope, **dates)
    if reactions is None:
        return {"title": "Reactions", "stats": None, "note": note}
    keys = ("given", "received", "reciprocity", "funniest", "hall_of_fame")
    picked = {k: reactions[k] for k in keys if k in reactions}
    return {"title": "Reactions", "stats": picked or reactions}


def _sec_late_nights(patterns, patterns_note):
    """Second consumer of the shared get_activity_patterns result."""
    if patterns is None:
        return {"title": "Late Nights, Streaks & the Heatmap",
                "stats": None, "note": patterns_note}
    art = wrapped.render_heatmap_ascii(patterns.get("heatmap"))
    return {
        "title": "Late Nights, Streaks & the Heatmap",
        "stats": {"late_night": patterns.get("late_night"),
                  "streaks": patterns.get("streaks"),
                  "busiest_days": patterns.get("busiest_days")},
        "extra_md": f"```\n{art}\n```" if art else None,
    }


def _sec_milestones(scope, year):
    # get_milestones takes no date bounds by contract; reference_date anchors
    # the anniversary window to the report year's end when a year is given.
    milestone_kwargs = dict(scope)
    if year is not None:
        milestone_kwargs["reference_date"] = f"{year:04d}-12-31"
    milestones, note = wrapped._tool("get_milestones", **milestone_kwargs)
    return {"title": "Milestones & Anniversaries", "stats": milestones, "note": note}


def _sec_words(scope, dates, contact_entry):
    # get_word_frequency is the one analytics tool that scopes by SENDER, not
    # by that person's chats: under a --contact scope it returns only THEIR
    # words, while every other key here covers both sides. Naming the key for
    # what it actually holds keeps the narrator from calling one side's
    # vocabulary "shared".
    freq, note_f = wrapped._tool("get_word_frequency", **scope, **dates)
    trends, note_t = wrapped._tool("get_word_trends", **scope, **dates)
    words = {}
    if freq is not None:
        words["their_top_words" if contact_entry else "top_words"] = freq
    if trends is not None:
        for k in ("distinctive_terms", "emoji", "rates"):
            if k in trends:
                words[k] = trends[k]
    return {"title": "Words & Inside Jokes", "stats": words or None,
            "note": "; ".join(n for n in (note_f, note_t) if n)}


def _sec_groups(con, year, dates):
    groups, group_notes = {}, []
    for row in _top_group_chats(con, year):
        res, note = wrapped._tool("get_group_dynamics", chat_id=row["chat_id"], **dates)
        if res is None:
            group_notes.append(f"{row['label']}: {note}")
        else:
            groups[row["label"]] = res
    return {"title": "Groups", "stats": groups or None,
            "note": "; ".join(group_notes[:2]) or "no group chats in scope"}


def _sec_eras(con, scope, dates):
    """Phase-3 section: None (section omitted) unless the batch-LLM
    artifacts exist at all."""
    n_annotated = con.execute("SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]
    n_topics = con.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    if not (n_annotated or n_topics):
        return None
    arc, note = wrapped._tool(
        "get_word_trends", include=["emotions", "valence"], **scope, **dates
    )
    stats = None
    if arc is not None:
        stats = {k: arc[k] for k in ("emotions", "valence") if k in arc} or None
    return {"title": "Eras & Emotional Arc", "stats": stats, "note": note}


def build_sections(con, roster, contact_entry=None, year=None,
                   exclude_chat_ids=(), exclude_ranges=()):
    """Build the ordered list of report sections. roster drives the default
    multi-contact report; contact_entry switches to the single-relationship
    deep-dive (people/groups sections are dropped -- they describe the whole
    corpus). The exclusions are forwarded to the prior-year roster so
    previous_rank is computed on the same basis as the current ranks."""
    dates = {}
    if year is not None:
        dates = {"date_from": f"{year:04d}-01-01", "date_to": f"{year:04d}-12-31"}
    scope = wrapped._entry_scope(contact_entry) if contact_entry else {}
    tempo_entries = [contact_entry] if contact_entry else roster

    # Year in Numbers + (later) Late Nights share one get_activity_patterns call.
    patterns, patterns_note = wrapped._tool("get_activity_patterns", **scope, **dates)

    sections = [_sec_year_in_numbers(patterns, patterns_note)]
    if contact_entry is None:
        sections.append({
            "title": "Your People",
            "stats": _your_people_stats(con, roster, year, exclude_chat_ids, exclude_ranges),
        })
    sections.append(_sec_tempo(tempo_entries, dates))
    sections.append(_sec_reactions(scope, dates))
    sections.append(_sec_late_nights(patterns, patterns_note))
    sections.append(_sec_milestones(scope, year))
    sections.append(_sec_words(scope, dates, contact_entry))
    if contact_entry is None:
        sections.append(_sec_groups(con, year, dates))
    eras = _sec_eras(con, scope, dates)
    if eras is not None:
        sections.append(eras)
    return sections
