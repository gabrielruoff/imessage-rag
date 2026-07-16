"""Small helpers shared between ingest.py, chunking.py, query_tools, and
reset_db.py: sqlite-vec extension loading, the contact-label SQL fragment,
the ISO-date -> day-number conversion used by chunk_vectors' metadata
columns, and the ingest_state key/value accessors."""

import datetime

import sqlite_vec


def load_vec_extension(con):
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)


# Human-readable label for a contact/handle: nickname > "First Last" > raw
# handle_identifier. Assumes the enclosing query joins handles AS h and
# contacts AS c (on h.contact_id).
CONTACT_LABEL_SQL = (
    "COALESCE("
    "NULLIF(c.nickname, ''), "
    "NULLIF(TRIM(COALESCE(c.first_name, '') || ' ' || COALESCE(c.last_name, '')), ''), "
    "h.handle_identifier)"
)


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
