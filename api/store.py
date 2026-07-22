"""app.db persistence for the web UI: conversations and their turns.

Plain sqlite3 -- NO sqlite-vec, its own connection helper, deliberately kept
apart from the analysis DB (db.py). This DB holds only the UI's chat history;
the chat engine stays stateless and callers rebuild the messages list from the
stored turns. Timestamps are ISO-8601 UTC strings.

DB_PATH is a module attribute (defaulting to config.APP_DB) so tests can point
it at a temp file before calling init_db().
"""

import contextlib
import datetime
import sqlite3

import config

DB_PATH = config.APP_DB


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextlib.contextmanager
def _connect():
    """Open app.db the one shared way and always close it: Row factory + foreign
    keys enforced (needed for the ON DELETE CASCADE from conversations to turns),
    committing on success. (A bare ``with sqlite3.connect()`` manages the
    transaction but never closes the handle -- a per-request leak on a server.)"""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        with con:
            yield con
    finally:
        con.close()


def init_db():
    """Create the schema if absent (idempotent). Safe to call on every start."""
    with _connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "title TEXT, created_at TEXT, updated_at TEXT)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "conversation_id INTEGER NOT NULL "
            "REFERENCES conversations(id) ON DELETE CASCADE, "
            "role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id)")


def create_conversation(title):
    """Insert an empty conversation, returning its row."""
    now = _now()
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return con.execute(
            "SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)
        ).fetchone()


def list_conversations():
    """All conversations, newest activity first, each with a turn_count."""
    with _connect() as con:
        return con.execute(
            "SELECT c.*, "
            "(SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) AS turn_count "
            "FROM conversations c ORDER BY c.updated_at DESC, c.id DESC"
        ).fetchall()


def get_conversation(conversation_id):
    """One conversation row (or None)."""
    with _connect() as con:
        return con.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()


def get_turns(conversation_id):
    """All turns for a conversation in insertion order."""
    with _connect() as con:
        return con.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()


def add_turn(conversation_id, role, content):
    """Append one turn (role/content), returning its row."""
    now = _now()
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO turns(conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        return con.execute("SELECT * FROM turns WHERE id = ?", (cur.lastrowid,)).fetchone()


def touch_conversation(conversation_id):
    """Bump updated_at to now (called after each new turn)."""
    with _connect() as con:
        con.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def delete_conversation(conversation_id):
    """Delete a conversation; its turns go via ON DELETE CASCADE."""
    with _connect() as con:
        con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
