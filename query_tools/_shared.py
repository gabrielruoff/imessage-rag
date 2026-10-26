"""Cross-cutting pieces of the query_tools package: the DB connection,
result-size ceilings, id-list/scope normalization, date-bound parsing, and
the message SELECT fragments every message-returning tool shares."""

import datetime
import json
import zoneinfo

import config
from config import TARGET_DB
from db import CONTACT_LABEL_SQL, connect

DB_PATH = str(TARGET_DB)


def _connect():
    # The path is resolved through the package attribute (not this module's
    # global) so tests can patch query_tools.DB_PATH; the lazy import is
    # cycle-safe because _connect only runs after the package is initialized.
    import query_tools

    return connect(query_tools.DB_PATH)


# Resolve a message's sender: "Me" when is_from_me, else the contact label
# (CONTACT_LABEL_SQL, shared with ingest's chunk builder via db.py).
# Requires the join below.
_SENDER_SELECT = f"CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE {CONTACT_LABEL_SQL} END AS sender"
_MSG_JOIN = (
    "LEFT JOIN handles h ON h.id = m.handle_id "
    "LEFT JOIN contacts c ON c.id = h.contact_id"
)
# The one column list every message-returning tool selects -- add new result
# columns here, not in the individual queries.
_MSG_COLS = (
    f"m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT}, "
    "m.text, m.date, m.has_attachment"
)

# Ceilings on model-requested result sizes -- defaults above are just what's
# used when the model omits the arg; these are enforced regardless of what
# the model asks for, so a tool call can't blow up prompt size.
MAX_MESSAGE_LIMIT = 50   # search_messages_sql / search_messages_fts
MAX_VEC_K = 20           # chunk-search tools (search_conversations etc.)
MAX_CONTACTS_LIMIT = 20  # search_contacts_by_name
MAX_CHATS_LIMIT = 50     # list_chats
MAX_STATS_ROWS = 50      # get_chat_summary_stats group_by_chat=True
MAX_ID_LIST = 50         # cap on chat_id/handle_id/contact_id/guid list params
MAX_CONTEXT_MESSAGES = 25   # get_message_context before/after ceiling
MAX_SCOPE_CHATS = 500       # cap on the person -> chats expansion (chunk + analytics tools)
MAX_WORD_FREQ_TOP = 250     # get_word_frequency `top` ceiling (~5KB of rows;
                            # wrapped over-fetches then filters COMMON_WORDS)
MAX_LEADERBOARD_ROWS = 10   # leaderboard sections in the analytics dict tools
MAX_SERIES_BUCKETS = 120    # time-series sections (auto-coarsen day->month past this)
MIN_LEADERBOARD_MESSAGES = 50  # min messages for a contact to enter unscoped leaderboards
MAX_CHUNK_MESSAGES = 200    # get_conversation row ceiling
HYBRID_POOL = 30            # per-arm candidate depth fed into RRF
RRF_K = 60                  # standard reciprocal-rank-fusion constant

TEXT_PREVIEW_CHARS = 300    # message rows
CHUNK_PREVIEW_CHARS = 500   # chunk rows (get_conversation returns the messages)

# Every aggregate counts only real conversation messages: legacy emote rows
# and system rows (group renames, member joins, ...) must never claim an
# initiation, break a silence, pad a histogram bucket, or count as a
# participant's sent message. (Belt and braces -- ingest already splits
# tapbacks into reactions.) Shared by the analytics tool modules and the
# sessions pipeline.
_REAL_MESSAGE_CLAUSES = ("COALESCE(m.is_emote, 0) = 0", "COALESCE(m.item_type, 0) = 0")


def member_key(is_from_me, contact_id, handle_id, *, unknown="unknown"):
    """The one home for participant identity when merging per-person rows in
    Python (SQL can't: from-me rows carry handle_id=0 with no handles row, and
    a contact may own several duplicate handle rows). 'me' for my side, else
    the handle's contact (duplicate handles collapse), else the bare handle.
    `unknown` is what a row with no identity (is_from_me=0 + handle_id=0
    system noise) maps to: the default keeps such rows in an "unknown"
    bucket; analytics_social passes None to drop them instead. Callers keep
    thin named wrappers preserving their exact contracts (analytics_social's
    `_person_key` maps no-identity rows to None; analytics_group's
    `_member_key` keeps the "unknown" bucket).

    Storage-format contract: the return is a stable, serializable string --
    "me", "contact:<contacts.id>", "handle:<handles.id>", or the `unknown`
    bucket -- and these strings ARE the storage format for derived-table
    identity keys. Features that persist per-person rows keyed by RESOLVED
    CONTACT IDENTITY (the dyad-scoped drift/persona tables) key on member_key's
    output, never on a raw chat_id/handle_id, and never re-implement the merge:
    member_key is the single home, so a contact's duplicate handles collapse to
    the same key in the store as in every live aggregate."""
    if is_from_me:
        return "me"
    if contact_id is not None:
        return f"contact:{contact_id}"
    if handle_id:
        return f"handle:{handle_id}"
    return unknown


def contact_label(con, contact_id):
    """Display name for a contacts row: nickname, else "First Last", else the
    contact's lowest-id handle identifier, else a "contact <id>" placeholder.
    The other direction of member_key -- key to label is `label_for_key`."""
    row = con.execute(
        "SELECT COALESCE("
        "NULLIF(nickname, ''), "
        "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), '')"
        ") FROM contacts WHERE id = ?", (contact_id,)
    ).fetchone()
    if row and row[0]:
        return row[0]
    handle = con.execute(
        "SELECT handle_identifier FROM handles WHERE contact_id = ? "
        "ORDER BY id LIMIT 1", (contact_id,)
    ).fetchone()
    return handle[0] if handle else f"contact {contact_id}"


def label_for_key(con, key, stored=None):
    """Human-readable label for a member_key / counterpart_key string -- the
    inverse of `member_key`, and its single home for the same reason.

    `stored` is a label a derived table already resolved for this key
    (`dyad_drift.label` is CONTACT_LABEL_SQL, the resolution every live
    aggregate uses). It wins when non-empty, so a read tool always agrees with
    the row it read the key from; pass None for a fresh lookup."""
    if stored:
        return stored
    if key == "me":
        return "Me"
    if key == "*":
        return "everyone"
    kind, _, raw = key.partition(":")
    if kind == "contact" and raw.isdigit():
        return contact_label(con, int(raw))
    if kind == "handle" and raw.isdigit():
        row = con.execute(
            "SELECT handle_identifier FROM handles WHERE id = ?", (int(raw),)
        ).fetchone()
        return row[0] if row and row[0] else key
    if kind == "chat" and raw.isdigit():
        row = con.execute(
            "SELECT COALESCE(NULLIF(display_name, ''), chat_identifier) "
            "FROM chats WHERE id = ?", (int(raw),)
        ).fetchone()
        return row[0] if row and row[0] else f"group chat {raw}"
    return key


def local_tz():
    """Effective bucketing timezone: config.LOCAL_TZ (an IANA name) resolved
    through zoneinfo, or None -- which astimezone() treats as the system local
    zone. Read from config at call time (and kept as a function) so tests can
    pin a zone deterministically."""
    name = config.LOCAL_TZ
    return zoneinfo.ZoneInfo(name) if name else None


def is_late_night(hour):
    """Local hour inside [LATE_NIGHT_START_HOUR, LATE_NIGHT_END_HOUR); a
    start > end window wraps midnight (e.g. 22..5)."""
    start, end = config.LATE_NIGHT_START_HOUR, config.LATE_NIGHT_END_HOUR
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def month_bounds(date_from, date_to):
    """Inclusive 'YYYY-MM' bounds for filtering person_period_lexicon.month.
    The upper bound backs off one microsecond from the half-open datetime bound
    so date_to='2026-04-30' includes April but not May."""
    lo = hi = None
    if date_from is not None:
        lo = _normalize_date_bound(date_from)[:7]
    if date_to is not None:
        upper = datetime.datetime.fromisoformat(_normalize_date_bound(date_to, upper=True))
        hi = (upper - datetime.timedelta(microseconds=1)).strftime("%Y-%m")
    return lo, hi


def _truncate_text_field(rows, limit=TEXT_PREVIEW_CHARS):
    """Cap each row's `text` field for prompt size; full content is available
    via get_full_messages(ids) / get_conversation(chunk_id)."""
    for r in rows:
        text = r.get("text")
        if text and len(text) > limit:
            r["text"] = text[:limit] + "…"
    return rows


def _validate_include(include, known, tool_name):
    """Normalize a section-selecting `include` param into a set of section
    names (None/[] = every section). Unknown names always raise rather than
    being silently dropped: a typo'd section would otherwise return a
    plausible-looking partial result the model can't tell from the real
    thing. Shared by every section-based analytics tool so they agree."""
    if not include:
        return set(known)
    names = set(include) if isinstance(include, (list, tuple)) else {include}
    unknown = names - set(known)
    if unknown:
        raise ValueError(
            f"{tool_name}: unknown include section(s) {sorted(unknown)!r}: "
            f"expected any of {', '.join(known)}"
        )
    return names


def _as_id_list(value):
    """Normalize a param that may be None, a scalar, or a list into None or
    a capped non-empty list -- lets every id-scoped tool param accept either
    shape. An explicit empty list means "no filter", same as omitting the
    param (LLMs pass [] to mean that), so it normalizes to None too."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)[:MAX_ID_LIST] or None
    return [value]


def _as_exclusion_list(value):
    """Normalize an exclude param (None, scalar, or list) into None or a
    non-empty list. Unlike _as_id_list this never truncates: an exclusion is
    a privacy guard, and silently dropping entries past MAX_ID_LIST would
    quietly re-admit the excluded -- the opposite failure mode of an
    oversized scope list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value) or None
    return [value]


def _exclude_id_filters(clauses, params, exclude_chat_ids, prefix="m."):
    """Append a `{prefix}chat_id NOT IN (?,...)` clause for a normalized
    exclusion list (no-op when None/empty) -- the mirror of
    _append_id_filters for carving chats *out* of a scope. NULL chat_id rows
    (messages not attached to any chat) survive: NULL fails NOT IN, so
    without the IS NULL arm an exclusion would silently drop them from
    otherwise-unscoped aggregates."""
    if not exclude_chat_ids:
        return
    placeholders = ",".join("?" * len(exclude_chat_ids))
    clauses.append(f"({prefix}chat_id IS NULL OR "
                   f"{prefix}chat_id NOT IN ({placeholders}))")
    params.extend(exclude_chat_ids)


def _scope_requested(*values):
    """True when any of the given id params is a real (non-empty) scope --
    an explicit [] counts as omitted, matching _as_id_list."""
    return any(_as_id_list(v) is not None for v in values)


def _participant_roster(con, chat_ids):
    """Roster seed rows for a chat list: one row per distinct handle in
    chat_participants -- (handle_id, contact_id, label) -- so silent members
    still appear. Shared by analytics.get_chat_participant_stats and the
    group-dynamics lurkers section; each caller does its own post-processing."""
    placeholders = ",".join("?" * len(chat_ids))
    return con.execute(
        f"""
        SELECT DISTINCT h.id AS handle_id, h.contact_id,
               {CONTACT_LABEL_SQL} AS label
        FROM chat_participants cp
        JOIN handles h ON h.id = cp.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
        WHERE cp.chat_id IN ({placeholders})
        """,
        chat_ids,
    ).fetchall()


def _resolve_contact_handle_ids(con, contact_ids):
    """Given a list of contact_ids, return the handle_ids they own."""
    placeholders = ",".join("?" * len(contact_ids))
    rows = con.execute(
        f"SELECT id FROM handles WHERE contact_id IN ({placeholders})",
        contact_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _envelope(rows, total, order):
    """Result envelope for the row-capped ranking/search tools: makes
    truncation visible to the model so a capped page is never mistaken for
    the whole population (see docs/eval_report_2026-07-16.md)."""
    return {
        "rows": rows,
        "total_matching": total,
        "returned": len(rows),
        "order": order,
        "truncated": total > len(rows),
    }


def _id_filter_clause(column, values):
    """Build an `column IN (?,...)` fragment + params for a normalized id
    list, or ("", []) if values is None."""
    if not values:
        return "", []
    placeholders = ",".join("?" * len(values))
    return f"{column} IN ({placeholders})", list(values)


def _append_id_filters(clauses, params, chat_ids, handle_ids, prefix="m."):
    """Append chat_id/handle_id IN-clauses (if non-empty) to a shared
    clauses/params list -- the common tail of every scoped message query."""
    for column, ids in ((f"{prefix}chat_id", chat_ids), (f"{prefix}handle_id", handle_ids)):
        clause, cparams = _id_filter_clause(column, ids)
        if clause:
            clauses.append(clause)
            params.extend(cparams)


def _date_clauses(clauses, params, date_from, date_to, column="m.date"):
    """Append normalized date-range clauses (plain-string compares -- never
    wrap the column in SQL date functions, that kills idx_messages_chat_date)."""
    if date_from is not None:
        clauses.append(f"{column} >= ?")
        params.append(_normalize_date_bound(date_from))
    if date_to is not None:
        clauses.append(f"{column} < ?")
        params.append(_normalize_date_bound(date_to, upper=True))


def _date_bound_step(value):
    """How far to advance an upper date bound: one unit of the smallest field
    the caller actually wrote. Read off the raw string because fromisoformat
    parses '2026-01-02' and '2026-01-02T00:00:00' into the same datetime.

    fromisoformat accepts any single character as the date/time separator (it
    is isoformat(sep=...)'s inverse -- '2026-01-02 13:46:50' parses fine), so
    the time part is found positionally, after the 10-char extended date or
    8-char compact date, not by splitting on 'T'."""
    time_part = value[11:] if "-" in value[:5] else value[9:]
    for sep in ("+", "-", "Z", "z"):          # drop any UTC offset suffix
        time_part = time_part.partition(sep)[0]
    if not time_part:
        return datetime.timedelta(days=1)
    if "." in time_part or "," in time_part:  # ISO allows a comma decimal mark
        return datetime.timedelta(microseconds=1)
    return {
        2: datetime.timedelta(hours=1),       # '13' / compact or extended
        4: datetime.timedelta(minutes=1),     # '13:46' / '1346'
        6: datetime.timedelta(seconds=1),     # '13:46:50' / '134650'
    }.get(len(time_part.replace(":", "")), datetime.timedelta(seconds=1))


def _normalize_date_bound(value, upper=False):
    """Parse an ISO date/datetime into the unlabeled-UTC string form the `date`
    columns are stored in (see the package docstring on storage format).

    An upper bound comes back half-open -- advanced by one unit of whatever
    granularity the caller wrote -- so date_to='2026-01-02' means "through the
    end of Jan 2" and pairs with `date < ?`. Compared as a plain string, a bare
    date would otherwise match only exactly midnight and silently drop the day.

    A tz-aware bound ('...-05:00', '...Z') is converted to UTC here, so a caller
    that skips the conversion still gets a correct answer. An unparseable bound
    raises ValueError rather than quietly matching nothing.
    """
    if value is None:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"invalid ISO date {value!r}: expected '2026-01-02' or "
            f"'2026-01-02T13:00:00' (optionally with a UTC offset) -- {e}"
        ) from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    if upper:
        parsed += _date_bound_step(value)
    return parsed.isoformat()


def _resolve_handle_filter(con, handle_id, contact_id):
    """Merge handle_id and contact_id (each scalar-or-list-or-None) into a
    single normalized handle_id list, resolving contact_id through handles."""
    handle_ids = _as_id_list(handle_id) or []
    contact_ids = _as_id_list(contact_id)
    if contact_ids:
        handle_ids = list(handle_ids) + _resolve_contact_handle_ids(con, contact_ids)
    return handle_ids[:MAX_ID_LIST] or None


def _resolve_chat_scope(con, chat_id, handle_id, contact_id):
    """Resolve scope params into (chat_ids, empty) for chat-level tools.
    Chunks and aggregate analytics cover whole chats, so handle_id/contact_id
    scope to the chats that person participates in (an exact
    chat_participants join); when chat and person params are both given, the
    scopes intersect, matching the AND semantics of the message-level tools.
    `empty` mirrors _resolve_scope: True when a scope was requested but
    matched nothing."""
    # chat_ids needs no emptiness check: a chat_id never resolves through a
    # lookup, so a requested chat scope is always non-empty after
    # normalization (and an explicit [] normalizes to "no filter").
    chat_ids = _as_id_list(chat_id)
    handle_requested = _scope_requested(handle_id, contact_id)
    if handle_requested:
        handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
        if not handle_ids:
            return None, True
        placeholders = ",".join("?" * len(handle_ids))
        # ORDER BY makes the MAX_SCOPE_CHATS truncation deterministic
        person_chats = [r[0] for r in con.execute(
            f"SELECT DISTINCT chat_id FROM chat_participants "
            f"WHERE handle_id IN ({placeholders}) ORDER BY chat_id", handle_ids,
        ).fetchall()][:MAX_SCOPE_CHATS]
        if chat_ids:
            person_set = set(person_chats)
            chat_ids = [c for c in chat_ids if c in person_set]
        else:
            chat_ids = person_chats
        if not chat_ids:
            return None, True
    return chat_ids, False


def _restrict_to_direct_chats(con, chat_ids):
    """Keep only 1:1 chats (chats.is_group = 0) from a resolved chat_id list,
    returning the filtered list (or the input unchanged when None/empty).

    Composes with the scope resolution above: feed it the chat_ids that
    _resolve_chat_scope / _resolve_scope produced, and every downstream query
    (built with _append_id_filters + _REAL_MESSAGE_CLAUSES) is then confined to
    direct chats. Two-sided per-dyad analytics use this so a person scope never
    silently folds in group traffic that has no single counterparty:
    get_response_dynamics restricts a bare person scope to that person's 1:1
    chats, and the dyad-scoped nudges/drift/persona features share the same
    home rather than re-deriving is_group = 0. Opt-in -- existing tools that
    don't call it keep their behavior unchanged."""
    if not chat_ids:
        return chat_ids
    placeholders = ",".join("?" * len(chat_ids))
    rows = con.execute(
        f"SELECT id FROM chats WHERE id IN ({placeholders}) AND is_group = 0",
        chat_ids,
    ).fetchall()
    return [r[0] for r in rows]


def _resolve_scope(con, chat_id, handle_id, contact_id):
    """Resolve scope params into (chat_ids, handle_ids, empty). `empty` is
    True when a non-empty scope param was provided but matched no ids -- the
    caller should then return [] rather than dropping the filter and
    over-matching. An explicit empty list is treated as an omitted param."""
    chat_ids = _as_id_list(chat_id)
    handle_requested = _scope_requested(handle_id, contact_id)
    handle_ids = _resolve_handle_filter(con, handle_id, contact_id)
    # chat_ids needs no such check: a chat_id never resolves through a lookup,
    # so a requested chat scope is always non-empty after normalization.
    empty = handle_requested and not handle_ids
    return chat_ids, handle_ids, empty


def cap_tool_result_json(result, result_json, max_chars):
    """Cap one tool result's JSON at ``max_chars`` while keeping it
    valid JSON. An oversized tool message makes Ollama silently drop it AND
    every message before it (only the system prompt survives), so the model
    would answer with no user question in context. List results (the only
    shape that gets big in practice) are cut at whole-row boundaries so every
    kept row stays parseable; anything else keeps a prefix explicitly labeled
    as a fragment."""
    if len(result_json) <= max_chars:
        return result_json
    note = (
        f"result was {len(result_json)} chars, over the {max_chars}-char "
        "cap; re-call with narrower filters or a smaller limit"
    )
    # Enveloped results ({"rows": [...], ...}) get their rows cut the same
    # way as bare lists, with the other envelope keys preserved.
    envelope = (
        result if isinstance(result, dict) and isinstance(result.get("rows"), list)
        else None
    )
    rows = envelope["rows"] if envelope is not None else result
    if isinstance(rows, list) and rows:
        # Proportional first guess, then halve until the wrapper fits.
        keep = max(1, len(rows) * max_chars // len(result_json))
        while True:
            capped_result = {
                "truncated": True, "rows_total": len(rows), "rows_returned": keep,
                "note": note, "rows": rows[:keep],
            }
            if envelope is not None:
                # Carry the envelope's context keys; "returned" would now lie
                # (it counted the pre-cap rows), so it is recomputed.
                capped_result.update(
                    {k: v for k, v in envelope.items()
                     if k not in capped_result and k != "returned"}
                )
                capped_result["returned"] = keep
            capped = json.dumps(capped_result)
            if len(capped) <= max_chars:
                return capped
            if keep == 1:
                break  # a single row alone busts the cap -- fragment it below
            keep = max(1, keep // 2)
    # JSON-escaping the fragment can inflate it, so leave envelope headroom
    # and halve on the (rare) overshoot.
    cut = max_chars - 200
    while True:
        capped = json.dumps({"truncated": True, "note": note,
                             "partial_json": result_json[:cut]})
        if len(capped) <= max_chars or cut == 0:
            return capped
        cut //= 2
