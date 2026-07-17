"""Year-in-review ("Wrapped") markdown report generator.

Composes a report from deterministic, direct query_tools calls (no LLM tool
loop): roster ranking and scoping run as direct SQL against the analysis DB,
each feature section calls one analytics tool, and (unless --no-llm) a short
LLM narrative is written per section from the computed stats alone.

Safety-first flow (the the silent-contact guard): before generating anything, the
would-be-featured roster is printed, abruptly-silent contacts are
auto-excluded (resurfacing an ex or the estranged uninvited is the classic
nostalgia-product failure), and the run aborts unless confirmed (--yes skips
the prompt; --include "Name" re-admits an excluded contact).

Usage:
    wrapped.py [--year YYYY] [--contact NAME] [--exclude SPEC ...]
               [--include NAME ...] [--out PATH] [--no-llm] [--yes]

--exclude SPEC: an integer excludes that chat_id, 'A..B' (ISO dates) drops
that date range from scope, anything else is a contact name (resolved via
search_contacts_by_name).

Every tool is resolved as a query_tools.<name> attribute at call time, so a
tool that is not integrated yet (AttributeError) or that returns an error /
"available": false dict just skips its section with a note -- the report
never crashes on a missing section.
"""

import argparse
import contextlib
import datetime
import json
import re
import sqlite3
import sys
from pathlib import Path

import config
import db
import ollama
import progress
import query_tools

# Patchable by tests, resolved at call time (mirrors query_tools.DB_PATH).
DB_PATH = str(config.TARGET_DB)

# Aggregate over real conversation messages only (same convention as the
# analytics tools -- see query_tools._shared._REAL_MESSAGE_CLAUSES).
_REAL_MSG = "COALESCE(m.is_emote, 0) = 0 AND COALESCE(m.item_type, 0) = 0"

_DENSITY_CHARS = " .:-=+*#%@"
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_NARRATIVE_SYSTEM = (
    "You write short narrative blurbs for sections of a personal year-in-review "
    "messaging report. Write 2-4 sentences from the stats JSON you are given. "
    "Rules: use ONLY numbers that appear in the input -- never invent, derive, or "
    "round differently; be descriptive, never evaluative -- no judgments about "
    "relationship health, no advice; no headings or bullet lists, plain prose only."
)


def _connect():
    con = sqlite3.connect(DB_PATH)  # module attribute, patchable by tests
    db.load_vec_extension(con)
    con.row_factory = sqlite3.Row
    return con


def _tool(name, /, **kwargs):
    """Call query_tools.<name>, late-bound. Returns (result, note): a missing
    tool (not yet integrated), a raised error, or an error/unavailable dict
    all come back as (None, reason) so sections degrade instead of crashing."""
    try:
        fn = getattr(query_tools, name)
    except AttributeError:
        return None, f"{name} is not available yet"
    try:
        result = fn(**kwargs)
    except Exception as e:  # a broken section must never kill the report
        return None, f"{name} failed: {e}"
    if isinstance(result, dict):
        if result.get("error"):
            return None, f"{name}: {result['error']}"
        if result.get("available") is False:
            return None, f"{name}: data not available"
    return result, None


def _month_index(ym):
    """'YYYY-MM...' -> monotonic month number (year*12 + month)."""
    return int(ym[:4]) * 12 + int(ym[5:7])


def parse_exclude_spec(spec):
    """Classify one --exclude SPEC: int-like -> ('chat', chat_id); 'A..B' with
    both sides ISO dates -> ('range', (A, B)); anything else -> ('name', spec)."""
    spec = spec.strip()
    try:
        return "chat", int(spec)
    except ValueError:
        pass
    if ".." in spec:
        a, _, b = spec.partition("..")
        try:
            datetime.date.fromisoformat(a[:10])
            datetime.date.fromisoformat(b[:10])
            return "range", (a, b)
        except ValueError:
            pass
    return "name", spec


def _entry_key(entry):
    """Stable identity for a roster entry across years: contact when known,
    else the bare handle."""
    if entry["contact_id"] is not None:
        return "c", entry["contact_id"]
    return "h", entry["handle_id"]


def _entry_scope(entry):
    """Tool-call scope kwargs for one roster entry."""
    if entry["contact_id"] is not None:
        return {"contact_id": entry["contact_id"]}
    return {"handle_id": entry["handle_id"]}


def build_roster(con, year=None, exclude_chat_ids=(), exclude_ranges=(), limit=None):
    """Top contacts by message volume (messages they sent) in the scope year
    (all time when year is None), grouped by contact -- unsaved numbers group
    by handle. Returns [{contact_id, handle_id, name, n_messages}] ranked."""
    if limit is None:
        limit = config.WRAPPED_TOP_CONTACTS
    clauses = ["m.date IS NOT NULL", "m.is_from_me = 0", _REAL_MSG]
    params = []
    if year is not None:
        clauses += ["m.date >= ?", "m.date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("m.chat_id != ?")
        params.append(cid)
    for lo, hi in exclude_ranges:
        clauses.append("NOT (m.date >= ? AND m.date < ?)")
        params += [
            query_tools._normalize_date_bound(lo),
            query_tools._normalize_date_bound(hi, upper=True),
        ]
    rows = con.execute(
        f"""
        SELECT c.id AS contact_id, MIN(h.id) AS handle_id,
               {db.CONTACT_LABEL_SQL} AS name, COUNT(*) AS n_messages
        FROM messages m
        JOIN handles h ON h.id = m.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE {" AND ".join(clauses)}
        GROUP BY COALESCE('c' || c.id, 'h' || h.id)
        ORDER BY n_messages DESC, name
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def silent_exclusions(con, roster, min_messages=None, silent_months=None):
    """Split a roster into (kept, excluded): a contact is auto-excluded when
    they have >= min_messages lifetime messages, their last message is >=
    silent_months months before the corpus end (MAX(messages.date)), and they
    had traffic in >= 3 of the 6 months before going silent -- i.e. an active
    relationship that stopped abruptly, not one that faded out."""
    if min_messages is None:
        min_messages = config.WRAPPED_SILENT_MIN_MESSAGES
    if silent_months is None:
        silent_months = config.WRAPPED_SILENT_MONTHS
    corpus_end = con.execute("SELECT MAX(date) FROM messages").fetchone()[0]
    if not corpus_end:
        return list(roster), []
    end_idx = _month_index(corpus_end)
    kept, excluded = [], []
    for entry in roster:
        if entry["contact_id"] is not None:
            who, params = "h.contact_id = ?", [entry["contact_id"]]
        else:
            who, params = "m.handle_id = ?", [entry["handle_id"]]
        # _REAL_MSG here too: the auto-exclusion threshold must be measured
        # against the same population build_roster ranks on, or a contact can
        # clear WRAPPED_SILENT_MIN_MESSAGES on system/emote rows alone.
        lifetime, last_date = con.execute(
            f"""
            SELECT COUNT(*), MAX(m.date) FROM messages m
            JOIN handles h ON h.id = m.handle_id
            WHERE m.date IS NOT NULL AND {_REAL_MSG} AND {who}
            """,
            params,
        ).fetchone()
        if last_date is None or lifetime < min_messages:
            kept.append(entry)
            continue
        last_idx = _month_index(last_date)
        if end_idx - last_idx < silent_months:
            kept.append(entry)
            continue
        # substr in projection over already-filtered rows: allowed by the
        # date-handling rule (never in WHERE predicates).
        months = con.execute(
            f"""
            SELECT DISTINCT substr(m.date, 1, 7) FROM messages m
            JOIN handles h ON h.id = m.handle_id
            WHERE m.date IS NOT NULL AND {_REAL_MSG} AND {who}
            """,
            params,
        ).fetchall()
        active = sum(1 for (ym,) in months if last_idx - 5 <= _month_index(ym) <= last_idx)
        if active >= 3:
            excluded.append({**entry, "last_month": last_date[:7]})
        else:
            kept.append(entry)
    return kept, excluded


def render_heatmap_ascii(heatmap):
    """Render a 7x24 weekday-major count grid as ASCII art using
    ' .:-=+*#%@' density characters. Accepts the raw grid or a dict wrapping
    it; returns None when no grid is recognizable."""
    grid = heatmap
    if isinstance(heatmap, dict):
        grid = next(
            (heatmap[k] for k in ("grid", "counts", "cells", "heatmap") if k in heatmap),
            None,
        )
    if not grid:
        return None
    try:
        peak = max(v for row in grid for v in row)
    except (TypeError, ValueError):
        return None
    header = [" "] * 24
    for hour in (0, 6, 12, 18):
        for i, ch in enumerate(str(hour)):
            header[hour + i] = ch
    lines = ["     " + "".join(header)]
    top = len(_DENSITY_CHARS) - 1
    for day, row in zip(_WEEKDAYS, grid):
        cells = []
        for v in row:
            if not peak or not v:
                cells.append(_DENSITY_CHARS[0])
            else:
                cells.append(_DENSITY_CHARS[max(1, round(v / peak * top))])
        lines.append(f"{day}  " + "".join(cells))
    return "\n".join(lines)


def narrate_section(title, stats):
    """One 2-4 sentence LLM narrative for a section's stats dict. Raises on
    LLM failure -- the caller falls back to stats-only."""
    payload = json.dumps(stats, ensure_ascii=False, default=str)
    reply = ollama.chat_once(
        [
            {"role": "system", "content": _NARRATIVE_SYSTEM},
            {"role": "user", "content": f"Section: {title}\nStats JSON:\n{payload}"},
        ],
        model=config.WRAPPED_MODEL,
    )
    return (reply.get("content") or "").strip() or None


def compose_report(sections, title, use_llm=False):
    """Assemble the markdown report. Each section dict: {title, stats,
    note, extra_md} -- stats None renders the heading plus a one-line skip
    note; a narrative failure keeps the stats block and notes the fallback."""
    parts = [f"# {title}"]
    # Only the narration pass is slow enough to be worth reporting (one Ollama
    # call per section); --no-llm assembles in milliseconds. The label stays
    # generic -- section titles can carry a contact's name, and progress output
    # is not the place to leak one.
    with progress.track("Narrating sections", total=len(sections),
                        enabled=use_llm) as p:
        for sec in sections:
            parts.append(f"## {sec['title']}")
            stats = sec.get("stats")
            if stats is None:
                parts.append(f"_Skipped: {sec.get('note') or 'no data'}_")
                p.advance()
                continue
            if use_llm:
                try:
                    narrative = narrate_section(sec["title"], stats)
                except Exception as e:
                    parts.append(f"_Narrative unavailable ({e}); stats only._")
                else:
                    if narrative:
                        parts.append(narrative)
            parts.append(
                "```json\n"
                + json.dumps(stats, indent=2, ensure_ascii=False, default=str)
                + "\n```"
            )
            if sec.get("extra_md"):
                parts.append(sec["extra_md"])
            p.advance()
    return "\n\n".join(parts) + "\n"


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
        prev = build_roster(con, year - 1, exclude_chat_ids, exclude_ranges,
                            limit=max(50, len(roster)))
        prev_rank = {_entry_key(e): i + 1 for i, e in enumerate(prev)}
    people = []
    for i, entry in enumerate(roster, 1):
        row = {"rank": i, "name": entry["name"], "messages": entry["n_messages"]}
        pr = prev_rank.get(_entry_key(entry))
        if pr is not None:
            row["previous_rank"] = pr
        people.append(row)
    return {"people": people}


def _top_group_chats(con, year, limit=3):
    clauses = ["ch.is_group = 1", "m.date IS NOT NULL", _REAL_MSG]
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
    scope = _entry_scope(contact_entry) if contact_entry else {}
    tempo_entries = [contact_entry] if contact_entry else roster
    sections = []

    # Year in Numbers + (later) Late Nights share one get_activity_patterns call.
    patterns, patterns_note = _tool("get_activity_patterns", **scope, **dates)
    if patterns is None:
        sections.append({"title": "Year in Numbers", "stats": None, "note": patterns_note})
    else:
        series = patterns.get("volume_series") or []
        if isinstance(series, dict):  # real tool wraps rows in {granularity, rows}
            series = series.get("rows") or []
        sections.append({
            "title": "Year in Numbers",
            "stats": {
                "messages_sent": _sum_series(series, "sent"),
                "messages_received": _sum_series(series, "received"),
                "late_night": patterns.get("late_night"),
                "volume_series": series,
            },
        })

    if contact_entry is None:
        sections.append({
            "title": "Your People",
            "stats": _your_people_stats(con, roster, year, exclude_chat_ids, exclude_ranges),
        })

    per_contact, tempo_notes = {}, []
    # Generic label: the roster's names must not reach stdout here.
    with progress.track("Analyzing tempo", total=len(tempo_entries)) as p:
        for entry in tempo_entries:
            res, note = _tool("get_response_dynamics", **_entry_scope(entry), **dates)
            if res is None:
                tempo_notes.append(f"{entry['name']}: {note}")
                p.advance()
                continue
            picked = {k: res[k] for k in ("reply_latency", "initiation", "sessions")
                      if k in res}
            per_contact[entry["name"]] = picked or res
            p.advance()
    sections.append(
        {"title": "Tempo", "stats": {"per_contact": per_contact} if per_contact else None,
         "note": "; ".join(tempo_notes[:2]) or "no contacts in scope"}
    )

    reactions, note = _tool("get_reaction_stats", **scope, **dates)
    if reactions is None:
        sections.append({"title": "Reactions", "stats": None, "note": note})
    else:
        keys = ("given", "received", "reciprocity", "funniest", "hall_of_fame")
        picked = {k: reactions[k] for k in keys if k in reactions}
        sections.append({"title": "Reactions", "stats": picked or reactions})

    if patterns is None:
        sections.append({"title": "Late Nights, Streaks & the Heatmap",
                         "stats": None, "note": patterns_note})
    else:
        art = render_heatmap_ascii(patterns.get("heatmap"))
        sections.append({
            "title": "Late Nights, Streaks & the Heatmap",
            "stats": {"late_night": patterns.get("late_night"),
                      "streaks": patterns.get("streaks"),
                      "busiest_days": patterns.get("busiest_days")},
            "extra_md": f"```\n{art}\n```" if art else None,
        })

    # get_milestones takes no date bounds by contract; reference_date anchors
    # the anniversary window to the report year's end when a year is given.
    milestone_kwargs = dict(scope)
    if year is not None:
        milestone_kwargs["reference_date"] = f"{year:04d}-12-31"
    milestones, note = _tool("get_milestones", **milestone_kwargs)
    sections.append({"title": "Milestones & Anniversaries", "stats": milestones, "note": note})

    # get_word_frequency is the one analytics tool that scopes by SENDER, not
    # by that person's chats: under a --contact scope it returns only THEIR
    # words, while every other key here covers both sides. Naming the key for
    # what it actually holds keeps the narrator from calling one side's
    # vocabulary "shared".
    freq, note_f = _tool("get_word_frequency", **scope, **dates)
    trends, note_t = _tool("get_word_trends", **scope, **dates)
    words = {}
    if freq is not None:
        words["their_top_words" if contact_entry else "top_words"] = freq
    if trends is not None:
        for k in ("distinctive_terms", "emoji", "rates"):
            if k in trends:
                words[k] = trends[k]
    sections.append(
        {"title": "Words & Inside Jokes", "stats": words or None,
         "note": "; ".join(n for n in (note_f, note_t) if n)}
    )

    if contact_entry is None:
        groups, group_notes = {}, []
        for row in _top_group_chats(con, year):
            res, note = _tool("get_group_dynamics", chat_id=row["chat_id"], **dates)
            if res is None:
                group_notes.append(f"{row['label']}: {note}")
            else:
                groups[row["label"]] = res
        sections.append(
            {"title": "Groups", "stats": groups or None,
             "note": "; ".join(group_notes[:2]) or "no group chats in scope"}
        )

    # Phase-3 section: only when the batch-LLM artifacts exist at all.
    n_annotated = con.execute("SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]
    n_topics = con.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    if n_annotated or n_topics:
        arc, note = _tool(
            "get_word_trends", include=["emotions", "valence"], **scope, **dates
        )
        stats = None
        if arc is not None:
            stats = {k: arc[k] for k in ("emotions", "valence") if k in arc} or None
        sections.append({"title": "Eras & Emotional Arc", "stats": stats, "note": note})

    return sections


def _resolve_contact(name):
    """Resolve --contact NAME to exactly one contact dict, or None (after
    printing why): no match or an ambiguous match both abort the run."""
    matches, note = _tool("search_contacts_by_name", name=name)
    if matches is None:
        print(f"could not resolve contact {name!r}: {note}")
        return None
    if not matches:
        print(f"no contact matches {name!r}")
        return None
    if len(matches) > 1:
        print(f"ambiguous contact {name!r}; candidates:")
        for c in matches:
            label = c.get("nickname") or " ".join(
                p for p in (c.get("first_name"), c.get("last_name")) if p
            )
            print(f"  {c['id']}: {label}")
        return None
    return matches[0]


def _contact_label(contact):
    return (
        contact.get("nickname")
        or " ".join(p for p in (contact.get("first_name"), contact.get("last_name")) if p)
        or str(contact.get("id"))
    )


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "all"


def _resolve_name_exclusions(names):
    """--exclude names -> contact-id set (via search_contacts_by_name) plus a
    lowercase-name fallback set for roster entries with no saved contact."""
    contact_ids, lowered = set(), set()
    for name in names:
        lowered.add(name.lower())
        matches, _ = _tool("search_contacts_by_name", name=name)
        for c in matches or []:
            contact_ids.add(c["id"])
    return contact_ids, lowered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, help="scope the report to one calendar year")
    parser.add_argument("--contact", help="single-relationship deep-dive for this contact name")
    parser.add_argument("--exclude", action="append", default=[], metavar="SPEC",
                        help="chat_id, ISO date range A..B, or contact name to exclude")
    parser.add_argument("--include", action="append", default=[], metavar="NAME",
                        help="re-admit an auto-excluded contact by name")
    parser.add_argument("--out", help="output path (default wrapped_<scope>.md)")
    parser.add_argument("--no-llm", action="store_true", help="stats-only, no narratives")
    parser.add_argument("--yes", action="store_true", help="skip the roster confirmation prompt")
    args = parser.parse_args(argv)

    with contextlib.closing(_connect()) as con:
        exclude_chats, exclude_ranges = [], []
        if args.contact:
            contact = _resolve_contact(args.contact)
            if contact is None:
                return 1
            entry = {"contact_id": contact["id"], "handle_id": None,
                     "name": _contact_label(contact), "n_messages": None}
            print(f"Featured: {entry['name']} (contact_id {contact['id']})")
            roster, contact_entry, scope_slug = [entry], entry, _slug(entry["name"])
        else:
            exclude_names = []
            for spec in args.exclude:
                kind, value = parse_exclude_spec(spec)
                if kind == "chat":
                    exclude_chats.append(value)
                elif kind == "range":
                    exclude_ranges.append(value)
                else:
                    exclude_names.append(value)
            roster = build_roster(con, args.year, exclude_chats, exclude_ranges)
            drop_ids, drop_names = _resolve_name_exclusions(exclude_names)
            roster = [e for e in roster
                      if e["contact_id"] not in drop_ids and e["name"].lower() not in drop_names]
            kept, silenced = silent_exclusions(con, roster)
            include_lower = {n.lower() for n in args.include}
            readmitted = {_entry_key(e) for e in silenced if e["name"].lower() in include_lower}
            keep_keys = {_entry_key(e) for e in kept} | readmitted
            featured = [e for e in roster if _entry_key(e) in keep_keys]

            print("Featured roster:")
            for i, e in enumerate(featured, 1):
                print(f"  {i}. {e['name']} — {e['n_messages']} messages")
            for e in silenced:
                if _entry_key(e) in readmitted:
                    continue
                print(f'excluded (went quiet {e["last_month"]}): {e["name"]} — '
                      f'pass --include "{e["name"]}" to feature them')
            roster, contact_entry, scope_slug = featured, None, "all"

        if not args.yes:
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1

        sections = build_sections(con, roster, contact_entry, args.year,
                                  exclude_chats, exclude_ranges)

    year_label = str(args.year) if args.year else "All Time"
    title = f"Wrapped — {year_label}"
    if contact_entry is not None:
        title += f" — {contact_entry['name']}"
    report = compose_report(sections, title, use_llm=not args.no_llm)

    out = Path(args.out) if args.out else Path(f"wrapped_{args.year or 'all'}_{scope_slug}.md")
    out.write_text(report, encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
