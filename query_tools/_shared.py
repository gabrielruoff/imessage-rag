"""Cross-cutting pieces of the query_tools package: the DB connection,
result-size ceilings, id-list/scope normalization, date-bound parsing, and
the message SELECT fragments every message-returning tool shares."""

import datetime
import sqlite3

from config import TARGET_DB
from db import CONTACT_LABEL_SQL, load_vec_extension

DB_PATH = str(TARGET_DB)


def _connect():
    # The path is resolved through the package attribute (not this module's
    # global) so tests can patch query_tools.DB_PATH; the lazy import is
    # cycle-safe because _connect only runs after the package is initialized.
    import query_tools

    con = sqlite3.connect(query_tools.DB_PATH)
    load_vec_extension(con)
    con.row_factory = sqlite3.Row
    return con


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
MAX_WORD_FREQ_TOP = 100     # get_word_frequency `top` ceiling
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


def _scope_requested(*values):
    """True when any of the given id params is a real (non-empty) scope --
    an explicit [] counts as omitted, matching _as_id_list."""
    return any(_as_id_list(v) is not None for v in values)


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
