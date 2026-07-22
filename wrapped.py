"""Year-in-review ("Wrapped") HTML report generator.

Composes a report from deterministic, direct query_tools calls (no LLM tool
loop): roster ranking and scoping run as direct SQL against the analysis DB,
each feature section calls one analytics tool, and (unless --no-llm) a short
LLM narrative is written per section from the computed stats alone.

Safety-first flow (the silent-contact guard): before generating anything, the
would-be-featured roster is printed, abruptly-silent contacts are
auto-excluded (resurfacing an ex or the estranged uninvited is the classic
nostalgia-product failure), and the run aborts unless confirmed (--yes skips
the prompt; --include "Name" re-admits an excluded contact).

Usage:
    wrapped.py [--year YYYY] [--contact NAME] [--exclude SPEC ...]
               [--include NAME ...] [--out PATH] [--no-llm] [--yes]

--exclude SPEC: an integer excludes that chat_id, 'A..B' (ISO dates) drops
that date range from roster ranking, anything else is a contact name
(resolved via search_contacts_by_name). Excluding a person -- by name or via
the silent-contact guard -- removes them from the roster AND drops their 1:1
chats from every section (totals, marathons, leaderboards, jokes, tone...).
Group chats they belong to stay in the report and may still name them, and
free-text mentions of them inside other people's kept messages are not
scrubbed.

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
from pathlib import Path

import config
import db
import ollama
import progress
import query_tools
import wellbeing  # --- feature: wellbeing --- (RES-68 / W4): narrative scrub
import wrapped_html
import wrapped_sections

# Patchable by tests, resolved at call time (mirrors query_tools.DB_PATH).
DB_PATH = str(config.TARGET_DB)

# Aggregate over real conversation messages only (same convention as the
# analytics tools -- see query_tools._shared._REAL_MESSAGE_CLAUSES).
_REAL_MSG = "COALESCE(m.is_emote, 0) = 0 AND COALESCE(m.item_type, 0) = 0"

# Apple backend: cap the stats-JSON prompt at this many chars. The FM window is
# a fixed 4,096 tokens (~16k chars) and an oversized message is silently
# dropped, so a big section's payload is truncated (with a marker so the
# narrator knows it was cut) rather than lost whole. A module constant, not a
# config knob -- it is a property of the FM window, not a user dial.
_APPLE_NARRATE_CHAR_BUDGET = 6000

_NARRATIVE_SYSTEM = (
    "You write the short blurb that opens each section of a personal year-in-review "
    "texting report (think Spotify Wrapped, for iMessage). You get one section's "
    "stats as JSON; the reader sees the same numbers in tables right below your "
    "blurb.\n"
    "Rules:\n"
    "- Write 2-3 sentences of plain, warm, second-person prose ('you'). No "
    "headings, bullets, or emoji.\n"
    "- Pick only the two or three most interesting facts; never walk through every "
    "row or restate a whole table.\n"
    "- Quote numbers, percentages, durations, and dates EXACTLY as they appear in "
    "the JSON -- they are pre-formatted for display; never invent, recompute, "
    "combine, or re-round anything.\n"
    "- Reply times and conversation lengths are MEDIANS: say 'typically' or "
    "'usually', never 'average' or 'on average'.\n"
    "- Descriptive, never evaluative: no judgments about relationship health, no "
    "advice.\n"
    "- Never mention JSON, stats, data, sections, or this report's machinery."
)


def _connect():
    return db.connect_default(DB_PATH)  # DB_PATH: module attribute, patchable by tests


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
    clauses = ["m.date IS NOT NULL", "m.is_from_me = 0", _REAL_MSG,
               "h.handle_identifier NOT LIKE 'urn:biz:%'"]
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


def narrate_section(title, stats, hint=None, report_title=""):
    """One short LLM narrative for a section's stats dict. The narrator gets
    the humanized copy of the stats -- hidden keys stripped, every value
    formatted exactly as the page renders it -- so it can only quote numbers
    in the form the reader sees -- plus the report title, so an all-time
    report isn't narrated as "this year". Raises on LLM failure -- the
    caller falls back to stats-only."""
    payload = json.dumps(wrapped_html.humanized(stats), ensure_ascii=False, default=str)
    if config.is_apple_model(config.WRAPPED_MODEL) and len(payload) > _APPLE_NARRATE_CHAR_BUDGET:
        # The FM window can't take a full section's stats JSON; cut it and mark
        # the cut -- payload is prompt text, so the marker just tells the
        # narrator the numbers are partial (better a short, honest prompt than
        # a whole message Ollama's apple path would silently drop).
        payload = payload[:_APPLE_NARRATE_CHAR_BUDGET] + " ...(truncated)"
    content = ""
    if report_title:
        content += (f"Report: {report_title} (match your wording to this scope -- "
                    "never say 'this year' unless the report covers one year)\n")
    content += f"Section: {title}\n"
    if hint:
        content += f"What this section shows: {hint}\n"
    content += f"Stats JSON:\n{payload}"
    reply = ollama.chat_once(
        [
            {"role": "system", "content": _NARRATIVE_SYSTEM},
            {"role": "user", "content": content},
        ],
        model=config.WRAPPED_MODEL,
        timeout=config.WRAPPED_TIMEOUT,
        think=config.WRAPPED_THINK,
    )
    return (reply.get("content") or "").strip() or None


def compose_report(sections, title, subtitle="", use_llm=False):
    """Assemble the HTML report. Each section dict: {title, stats, note,
    caption, hint, narrate_stats, extra_html}. A section renders visuals
    from extra_html plus any residual stats (tiles/tables via wrapped_html);
    it is skipped with a note only when it has neither. narrate_stats, when
    present, is the full shaped data the narrator sees -- sections that move
    their rows into visuals keep narrating the complete picture. A narrative
    failure keeps the rendered content and notes the fallback; caption is a
    one-line methodology note under the narrative; no JSON reaches the
    page."""
    cards = []
    # Only the narration pass is slow enough to be worth reporting (one Ollama
    # call per section); --no-llm assembles in milliseconds. The label stays
    # generic -- section titles can carry a contact's name, and progress output
    # is not the place to leak one.
    with progress.track("Narrating sections", total=len(sections),
                        enabled=use_llm) as p:
        for sec in sections:
            parts = [f"<h2>{wrapped_html.esc(sec['title'])}</h2>"]
            stats = sec.get("stats")
            extra = sec.get("extra_html")
            if stats is None and not extra:
                note = sec.get("note") or "no data"
                parts.append(f'<p class="note">Skipped: {wrapped_html.esc(note)}</p>')
            else:
                narrate_stats = sec.get("narrate_stats") or stats
                if use_llm and narrate_stats:
                    try:
                        narrative = narrate_section(sec["title"], narrate_stats,
                                                    sec.get("hint"), title)
                    except Exception as e:
                        parts.append(
                            '<p class="note">Narrative unavailable '
                            f"({wrapped_html.esc(e)}); stats only.</p>"
                        )
                    else:
                        # --- feature: wellbeing ---
                        # A wellbeing section's LLM blurb is scrubbed of any
                        # clinical-condition name before it reaches the page
                        # (defense in depth with the tool's source-level scrub);
                        # the support footer is code-emitted in extra_html and
                        # never depends on this narration.
                        if narrative and sec.get("sanitize_narrative"):
                            narrative = wellbeing.sanitize_condition_names(narrative)[0]
                        # --- end feature: wellbeing ---
                        if narrative:
                            parts.append(
                                f'<p class="narrative">{wrapped_html.esc(narrative)}</p>'
                            )
                if sec.get("caption"):
                    parts.append(
                        f'<p class="note">{wrapped_html.esc(sec["caption"])}</p>'
                    )
                if stats:
                    parts.append(wrapped_html.render_value(stats))
                if extra:
                    parts.append(extra)
            cards.append('<section class="card">' + "".join(parts) + "</section>")
            p.advance()
    cards.append(_outro_card(title))
    footer = "Generated locally from your message history. Nothing left this machine."
    hero = _hero(sections, subtitle)
    return wrapped_html.render_page(title, subtitle, "\n".join(cards), footer,
                                    hero=hero)


def _outro_card(title):
    """The closing beat: the report signs off as one received message --
    bookending the hero, which opened as a sent one."""
    year = re.search(r"\b(\d{4})\b", title)
    text = f"that’s a wrap on {year.group(1)} 💙" if year else "that’s a wrap 💙"
    return (
        '<section class="card outro"><div class="bubbles">'
        '<figure class="msg inc"><div class="bubble">'
        f"{wrapped_html.esc(text)}</div></figure></div></section>"
    )


def _hero(sections, subtitle):
    """The opener bubble: total messages in scope (from Year in Numbers) with
    reactions received as its tapback. Empty when the totals are missing --
    render_page then falls back to the plain subtitle."""
    totals = reactions = None
    for sec in sections:
        stats = sec.get("narrate_stats") or sec.get("stats")
        if not isinstance(stats, dict):
            continue
        if totals is None and stats.get("messages_sent") is not None:
            totals = (stats.get("messages_sent") or 0) + (stats.get("messages_received") or 0)
        if reactions is None and stats.get("reactions_received"):
            reactions = stats["reactions_received"]
    if not totals:
        return ""
    caption = subtitle.replace("Generated", "Delivered") or subtitle
    return wrapped_html.render_hero(totals, caption, reactions)


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


def _person_direct_chat_ids(con, contact_ids, handle_ids):
    """1:1 (non-group) chat ids the given contacts/handles participate in --
    what excluding a person carves out of every section. Group chats stay:
    they are shared history with everyone else in them (same call
    _inside_jokes_all makes)."""
    who = []
    params = []
    if contact_ids:
        who.append(f"h.contact_id IN ({','.join('?' * len(contact_ids))})")
        params += sorted(contact_ids)
    if handle_ids:
        who.append(f"h.id IN ({','.join('?' * len(handle_ids))})")
        params += sorted(handle_ids)
    if not who:
        return set()
    return {r[0] for r in con.execute(
        f"""
        SELECT DISTINCT cp.chat_id
        FROM chat_participants cp
        JOIN handles h ON h.id = cp.handle_id
        JOIN chats ch ON ch.id = cp.chat_id AND ch.is_group = 0
        WHERE {" OR ".join(who)}
        """,
        params,
    )}


def _business_chat_ids(con):
    """Chat ids with a business (urn:biz:) participant. build_roster keeps
    business handles off the roster, but corpus-wide sections don't go through
    the roster -- their chats must be carved out via exclude_chat_ids or the
    raw urn:biz identifier surfaces as a 'contact' (e.g. the night-owls
    leaderboard). No is_group guard: a business in a chat makes the whole chat
    business traffic."""
    return {r[0] for r in con.execute(
        """
        SELECT DISTINCT cp.chat_id
        FROM chat_participants cp
        JOIN handles h ON h.id = cp.handle_id
        WHERE h.handle_identifier LIKE 'urn:biz:%'
        """
    )}


def _excluded_name_set(con, contact_ids, entries, name_specs):
    """Lowercase names identifying excluded persons -- used to keep a term
    named after them ('alice' coined in someone else's chat) out of the joke
    cards, and them out of the prior-year ranks. Covers the raw --exclude
    specs, each excluded contact's first name / nickname / full label, and
    the dropped roster entries' display labels. Bare first names mean
    collisions over-drop (excluding one Alice hides every Alice from the word
    cloud, joke cards, and prior ranks) -- deliberate: for a privacy guard,
    over-exclusion is the right failure direction."""
    names = {s.lower() for s in name_specs}
    names |= {e["name"].lower() for e in entries if e.get("name")}
    if contact_ids:
        placeholders = ",".join("?" * len(contact_ids))
        for first, last, nick in con.execute(
            f"SELECT first_name, last_name, nickname FROM contacts "
            f"WHERE id IN ({placeholders})", sorted(contact_ids),
        ):
            for name in (first, nick, " ".join(p for p in (first, last) if p)):
                if name:
                    names.add(name.lower())
    names.discard("")
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, help="scope the report to one calendar year")
    parser.add_argument("--contact", help="single-relationship deep-dive for this contact name")
    parser.add_argument("--exclude", action="append", default=[], metavar="SPEC",
                        help="chat_id, ISO date range A..B, or contact name to exclude")
    parser.add_argument("--include", action="append", default=[], metavar="NAME",
                        help="re-admit an auto-excluded contact by name")
    parser.add_argument("--out", help="output path (default wrapped_<scope>.html)")
    parser.add_argument("--no-llm", action="store_true", help="stats-only, no narratives")
    parser.add_argument("--yes", action="store_true", help="skip the roster confirmation prompt")
    # --- feature: wellbeing ---
    # Opt-in wellbeing trajectory section (RES-68 / W4). The flag overrides the
    # WRAPPED_WELLBEING config for this run; off by default either way.
    parser.add_argument("--wellbeing", action="store_true",
                        help="include the opt-in wellbeing trajectory section")
    # --- end feature: wellbeing ---
    args = parser.parse_args(argv)

    # --- feature: wellbeing ---
    wellbeing_on = args.wellbeing or config.WRAPPED_WELLBEING
    # --- end feature: wellbeing ---

    with contextlib.closing(_connect()) as con:
        exclude_chats, exclude_ranges, exclude_names = [], [], []
        for spec in args.exclude:
            kind, value = parse_exclude_spec(spec)
            if kind == "chat":
                exclude_chats.append(value)
            elif kind == "range":
                exclude_ranges.append(value)
            else:
                exclude_names.append(value)
        drop_ids, drop_names = _resolve_name_exclusions(exclude_names)
        excluded_entries = []  # roster entries dropped by name or silence

        if args.contact:
            contact = _resolve_contact(args.contact)
            if contact is None:
                return 1
            entry = {"contact_id": contact["id"], "handle_id": None,
                     "name": _contact_label(contact), "n_messages": None}
            print(f"Featured: {entry['name']} (contact_id {contact['id']})")
            roster, contact_entry, scope_slug = [entry], entry, _slug(entry["name"])
        else:
            roster = build_roster(con, args.year, exclude_chats, exclude_ranges)
            dropped = [e for e in roster
                       if e["contact_id"] in drop_ids or e["name"].lower() in drop_names]
            roster = [e for e in roster if e not in dropped]
            kept, silenced = silent_exclusions(con, roster)
            include_lower = {n.lower() for n in args.include}
            readmitted = {_entry_key(e) for e in silenced if e["name"].lower() in include_lower}
            keep_keys = {_entry_key(e) for e in kept} | readmitted
            featured = [e for e in roster if _entry_key(e) in keep_keys]
            excluded_entries = dropped + [e for e in silenced
                                          if _entry_key(e) not in readmitted]

            print("Featured roster:")
            for i, e in enumerate(featured, 1):
                print(f"  {i}. {e['name']} — {e['n_messages']} messages")
            for e in silenced:
                if _entry_key(e) in readmitted:
                    continue
                print(f'excluded (went quiet {e["last_month"]}): {e["name"]} — '
                      f'pass --include "{e["name"]}" to feature them')
            roster, contact_entry, scope_slug = featured, None, "all"

        # An excluded person's data must not leak back in through the
        # corpus-wide sections: expand every excluded person (named or
        # silence-guarded) into their 1:1 chats and carve those out of
        # everything. Group chats stay -- shared history.
        excl_contact_ids = set(drop_ids)
        excl_handle_ids = set()
        for e in excluded_entries:
            if e.get("contact_id") is not None:
                excl_contact_ids.add(e["contact_id"])
            elif e.get("handle_id") is not None:
                excl_handle_ids.add(e["handle_id"])
        person_chats = _person_direct_chat_ids(con, excl_contact_ids, excl_handle_ids)
        if person_chats:
            print(f"(also excluding {len(person_chats)} direct chat(s) of "
                  "excluded contacts from every section)")
        biz_chats = _business_chat_ids(con)
        if biz_chats:
            print(f"(also excluding {len(biz_chats)} business chat(s) -- "
                  "urn:biz handles -- from every section)")
        exclude_chats = sorted(set(exclude_chats) | person_chats | biz_chats)
        excluded_names = _excluded_name_set(con, excl_contact_ids,
                                            excluded_entries, drop_names)

        # --- feature: wellbeing ---
        # Discoverability: the section is opt-in, so surface it in the preview
        # when it is off rather than leaving it invisible.
        if not wellbeing_on:
            print("wellbeing trajectory section available but off — pass "
                  "--wellbeing (or set WRAPPED_WELLBEING=true) to include it")
        # --- end feature: wellbeing ---

        if not args.yes:
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1

        sections = wrapped_sections.build_sections(con, roster, contact_entry,
                                                   args.year, exclude_chats,
                                                   exclude_ranges, excluded_names,
                                                   wellbeing=wellbeing_on)

    year_label = str(args.year) if args.year else "All Time"
    title = f"Wrapped — {year_label}"
    if contact_entry is not None:
        title += f" — {contact_entry['name']}"
    today = datetime.date.today()
    subtitle = f"Generated {today:%B} {today.day}, {today.year}"
    report = compose_report(sections, title, subtitle, use_llm=not args.no_llm)

    out = (Path(args.out) if args.out
           else Path(f"wrapped_{args.year or 'all'}_{scope_slug}.html"))
    out.write_text(report, encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    db.run_cli(main)
