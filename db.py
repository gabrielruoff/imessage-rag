"""Small helpers shared between ingest.py, chunking.py, query_tools, and
reset_db.py: sqlite-vec extension loading, the contact-label SQL fragment,
the ISO-date -> day-number conversion used by chunk_vectors' metadata
columns, and the ingest_state key/value accessors."""

import datetime
import re
import sqlite3
import sys

import sqlite_vec


def load_vec_extension(con):
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)


def connect(path, vec=True, rows=True):
    """Open the analysis DB the one shared way. Callers keep the path in a
    module-level attribute (their test patch seam) and pass it in; vec/rows
    opt out of the sqlite-vec extension and sqlite3.Row for the callers that
    don't use them."""
    con = sqlite3.connect(path)
    if vec:
        load_vec_extension(con)
    if rows:
        con.row_factory = sqlite3.Row
    return con


def connect_default(path, *, vec=True, rows=True):
    """The one home for the batch CLIs' connection defaults. The per-module
    ``_connect()`` wrappers (annotate.py, topics.py, wrapped.py,
    eval/evalkit recipes) are documented test-patch seams: they keep their
    module-level DB path attribute and delegate here, so the defaults are
    defined once."""
    return connect(path, vec=vec, rows=rows)


def run_cli(main):
    """Shared ``__main__`` tail for the batch CLIs (annotate.py, topics.py,
    wrapped.py): exit with main()'s return code (None -> 0) and turn
    Ctrl-C / EOF into the conventional exit 130."""
    try:
        sys.exit(main() or 0)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted, exiting.")
        sys.exit(130)


# Human-readable label for a contact/handle: nickname > "First Last" > raw
# handle_identifier. Assumes the enclosing query joins handles AS h and
# contacts AS c (on h.contact_id).
CONTACT_LABEL_SQL = (
    "COALESCE("
    "NULLIF(c.nickname, ''), "
    "NULLIF(TRIM(COALESCE(c.first_name, '') || ' ' || COALESCE(c.last_name, '')), ''), "
    "h.handle_identifier)"
)


# chat.db's associated_message_guid points at a message *part*
# ('p:0/<guid>', 'bp:<guid>'); messages.guid is always bare. Reactions are
# stored with the prefix stripped -- at insert time by ingest, and for old
# rows by create_schema's migration -- so they join messages.guid. Both
# sites use this one function so they can never disagree.
_REACTION_GUID_PREFIX = re.compile(r"^(?:p:\d+/|bp:)")


def strip_reaction_guid_prefix(guid):
    return _REACTION_GUID_PREFIX.sub("", guid) if guid else guid


# Apple tapback type codes as stored in reactions.associated_message_type.
# 2000-2007 are tapbacks being added; 3000-3007 remove the corresponding
# tapback. Other codes (2, 3, 1000, ...) are non-tapback associated messages;
# counting queries filter to BETWEEN 2000 AND 2999 so removals and strays
# never inflate reaction counts.
TAPBACK_NAMES = {
    2000: "love",
    2001: "like",
    2002: "dislike",
    2003: "laugh",
    2004: "emphasize",
    2005: "question",
    2006: "custom-emoji",
    2007: "sticker",
}


def tapback_name(code):
    """Human-readable name for a reactions.associated_message_type code."""
    if code in TAPBACK_NAMES:
        return TAPBACK_NAMES[code]
    if code is not None and 3000 <= code <= 3007 and code - 1000 in TAPBACK_NAMES:
        return "removed-" + TAPBACK_NAMES[code - 1000]
    return f"unknown-{code}"


def get_state(con, key, default=None):
    """Read one ingest_state value (last_message_rowid, embed_model, ...)."""
    row = con.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con, key, value):
    con.execute(
        "INSERT INTO ingest_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def iso_to_day(iso_date: str) -> int:
    """Calendar day number (proleptic Gregorian ordinal) of an ISO date or
    datetime string. Written into chunk_vectors' first_day/last_day INTEGER
    metadata columns at ingest time and compared against by query_tools'
    date-range filters -- both sides must use this same function."""
    return datetime.date.fromisoformat(iso_date[:10]).toordinal()
