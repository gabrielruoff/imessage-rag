"""
iMessage chat.db -> normalized SQLite analysis database.

Pipeline:
  1. Snapshot chat.db safely (read-only backup, never touches the live file)
  2. Decode attributedBody blobs (streamtyped fallback for NULL `text` rows)
  3. Join message/handle/chat/attachment tables into a normalized schema,
     enriching handles with contact names from ./address_book.abcddb if present
  4. Build an FTS5 index over message text
  5. Group messages into conversation chunks (session-gap + size splitting,
     speaker labels, context header) and embed the chunks via local Ollama
     (nomic-embed-text) into a sqlite-vec table

Usage:
    python3 ingest.py                  # full pipeline
    python3 ingest.py --no-embed       # skip embedding step (e.g. Ollama not running)
    python3 ingest.py --since-last     # incremental: only ingest new messages since last run
    python3 ingest.py --senders="+12015551234,alice@example.com"
                                        # only ingest chats (direct or group) containing
                                        # at least one of the given phone numbers/emails
"""

import argparse
import datetime
import http.client
import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sqlite_vec

from config import (
    ADDRESS_BOOK_PATH,
    CHUNK_GAP_MINUTES,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_MESSAGES,
    EMBED_BATCH_SIZE,
    EMBED_MAX_WORKERS,
    EMBED_MODEL,
    SNAPSHOT_DB,
    SOURCE_DB,
    TARGET_DB,
)
from db import CONTACT_LABEL_SQL, iso_to_day, load_vec_extension
from embeddings import detect_embed_dim, embed_texts

APPLE_EPOCH = datetime.datetime(2001, 1, 1)

START_PATTERN = b"\x01\x2b"
END_PATTERN = b"\x86\x84"


# ---------------------------------------------------------------------------
# Timestamp + text decoding
# ---------------------------------------------------------------------------

def apple_ns_to_iso(ns):
    """Apple's message.date is nanoseconds since 2001-01-01 (Mac absolute time)."""
    if ns is None or ns == 0:
        return None
    return (APPLE_EPOCH + datetime.timedelta(seconds=ns / 1e9)).isoformat()


def decode_streamtyped(blob: bytes):
    """
    Port of imessage-exporter's `streamtyped.rs` fallback decoder.
    Extracts plain text from an NSAttributedString typedstream blob by
    locating the start/end byte markers and decoding the UTF-8 payload
    between them. Does not reconstruct styling/attachment ranges.
    Returns None if the markers aren't found (caller should flag as failed).
    """
    if not blob:
        return None

    start = blob.find(START_PATTERN)
    if start == -1:
        return None
    data = blob[start + 2:]

    end = data.find(END_PATTERN)
    if end != -1:
        data = data[:end]

    try:
        s = data.decode("utf-8")
        offset = 1
    except UnicodeDecodeError:
        s = data.decode("utf-8", errors="replace")
        offset = 3

    return s[offset:] if len(s) > offset else s


def resolve_text(raw_text, attributed_body):
    """Returns (text, decode_failed)."""
    if raw_text:
        return raw_text, 0
    if attributed_body:
        decoded = decode_streamtyped(bytes(attributed_body))
        if decoded is not None:
            return decoded, 0
        return None, 1
    return None, 0  # legitimately empty (e.g. pure attachment / reaction)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_source(source: Path, dest: Path):
    """Copy the live chat.db (plus -wal/-shm sidecars, if present) to a
    standalone snapshot, then checkpoint its WAL into the main file. The
    rest of the pipeline only ever reads from this copy, never the original
    (which may still be written to by Messages.app)."""
    import shutil

    for suffix in ("", "-wal", "-shm"):
        src_path = source.with_name(source.name + suffix)
        dest_path = dest.with_name(dest.name + suffix)
        if not src_path.exists():
            try:
                dest_path.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
            continue
        # Overwrite in place rather than unlinking first -- some mounted
        # filesystems disallow unlinking a file that's still referenced
        # elsewhere (e.g. FUSE mounts of a host folder).
        shutil.copy2(src_path, dest_path)

    # Force a WAL checkpoint so the snapshot is fully self-contained in the
    # main db file (safe to delete the -wal/-shm copies afterward).
    con = sqlite3.connect(dest)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()

    for suffix in ("-wal", "-shm"):
        p = dest.with_name(dest.name + suffix)
        if p.exists():
            p.unlink()

    print(f"Snapshotted {source} -> {dest}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY,
    chat_identifier TEXT,
    display_name TEXT,
    is_group INTEGER,
    service TEXT
);

CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY,      -- ZABCDRECORD.Z_PK from address_book.abcddb
    source_uniqueid TEXT,        -- ZABCDRECORD.ZUNIQUEID
    first_name TEXT,
    last_name TEXT,
    nickname TEXT,
    organization TEXT
);

CREATE TABLE IF NOT EXISTS handles (
    id INTEGER PRIMARY KEY,
    handle_identifier TEXT,      -- raw phone number/email from chat.db's handle.id
    contact_id INTEGER REFERENCES contacts(id),
    service TEXT
    -- name/org intentionally NOT stored here: they live on contacts and are
    -- resolved via contact_id. See build_contact_lookup / ingest_handles.
);

CREATE TABLE IF NOT EXISTS chat_participants (
    chat_id INTEGER,
    handle_id INTEGER,
    PRIMARY KEY (chat_id, handle_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,           -- source message ROWID
    guid TEXT UNIQUE,
    chat_id INTEGER,
    handle_id INTEGER,
    is_from_me INTEGER,
    text TEXT,
    decode_failed INTEGER DEFAULT 0,
    date TEXT,
    date_read TEXT,
    date_delivered TEXT,
    item_type INTEGER,
    is_emote INTEGER,
    has_attachment INTEGER DEFAULT 0,
    chunk_id INTEGER              -- chunks.id whose text this message was rendered into
);

CREATE TABLE IF NOT EXISTS chunks (
    -- AUTOINCREMENT so ids are never reused: tail chunks get deleted and
    -- rebuilt on incremental runs, and a reused id could alias a stale
    -- chunk_vectors row onto a different chunk.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    first_message_id INTEGER,
    last_message_id INTEGER,
    first_date TEXT,
    last_date TEXT,
    n_messages INTEGER,
    text TEXT                     -- composed conversation text, exactly as embedded
);

CREATE TABLE IF NOT EXISTS reactions (
    id INTEGER PRIMARY KEY,          -- source message ROWID of the reaction event
    target_message_guid TEXT,
    chat_id INTEGER,
    handle_id INTEGER,
    is_from_me INTEGER,
    associated_message_type INTEGER, -- tapback type code
    date TEXT
);

CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY,
    message_id INTEGER,
    filename TEXT,
    mime_type TEXT,
    transfer_name TEXT
);

CREATE TABLE IF NOT EXISTS ingest_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_date ON messages(chat_id, date);
CREATE INDEX IF NOT EXISTS idx_messages_handle ON messages(handle_id);
CREATE INDEX IF NOT EXISTS idx_messages_chunk ON messages(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chat ON chunks(chat_id);
CREATE INDEX IF NOT EXISTS idx_reactions_target ON reactions(target_message_guid);
CREATE INDEX IF NOT EXISTS idx_handles_identifier ON handles(handle_identifier);
CREATE INDEX IF NOT EXISTS idx_handles_contact_id ON handles(contact_id);
CREATE INDEX IF NOT EXISTS idx_chat_participants_handle ON chat_participants(handle_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

-- UPDATE OF text (not any column): chunk building stamps chunk_id on every
-- message, and an unscoped trigger would pointlessly rewrite the FTS entry
-- for each of those updates.
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF text ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def create_schema(con: sqlite3.Connection):
    cols_before = {row[1]: row[2] for row in con.execute("PRAGMA table_info(handles)")}
    if cols_before.get("contact_id") == "TEXT":
        # Old schema: `contact_id TEXT` held the raw phone/email string.
        con.execute("ALTER TABLE handles RENAME COLUMN contact_id TO handle_identifier")

    # Migrate an existing (old-layout) handles table to the new column set BEFORE
    # running SCHEMA -- SCHEMA creates idx_handles_contact_id ON handles(contact_id),
    # which would fail if that column weren't present yet. (On a fresh DB cols_before
    # is empty and SCHEMA creates handles complete, so this block is skipped.) The
    # REFERENCES contacts(id) target need not exist yet; FKs resolve at DML time.
    if cols_before:
        existing_cols = {row[1] for row in con.execute("PRAGMA table_info(handles)")}
        if "contact_id" not in existing_cols:
            con.execute("ALTER TABLE handles ADD COLUMN contact_id INTEGER REFERENCES contacts(id)")
        # Drop the old denormalized name/org columns: they were a pure copy of the
        # linked contacts row (contact_id) and are now resolved via join instead.
        for col in ("first_name", "last_name", "nickname", "organization"):
            if col in existing_cols:
                con.execute(f"ALTER TABLE handles DROP COLUMN {col}")

    # Pre-chunking messages table: add chunk_id before SCHEMA runs, since
    # SCHEMA creates idx_messages_chunk ON messages(chunk_id).
    msg_cols = {row[1] for row in con.execute("PRAGMA table_info(messages)")}
    if msg_cols and "chunk_id" not in msg_cols:
        con.execute("ALTER TABLE messages ADD COLUMN chunk_id INTEGER")

    # The update trigger used to fire on ANY messages update; drop it so
    # FTS_SCHEMA recreates it scoped to UPDATE OF text (see comment there).
    con.execute("DROP TRIGGER IF EXISTS messages_au")

    con.executescript(SCHEMA)
    con.executescript(FTS_SCHEMA)
    con.commit()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def get_state(con, key, default=None):
    row = con.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con, key, value):
    con.execute(
        "INSERT INTO ingest_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def resolve_allowed_chat_ids(src, senders):
    """Given a list of phone numbers / emails (matched against handle.id),
    return the set of chat_ids for every chat (direct or group) that any of
    them participates in. Returns None if `senders` is falsy, meaning "no
    filter" -- caller should ingest everything."""
    if not senders:
        return None

    placeholders = ",".join("?" for _ in senders)
    rows = src.execute(f"""
        SELECT DISTINCT chj.chat_id
        FROM chat_handle_join chj
        JOIN handle h ON h.ROWID = chj.handle_id
        WHERE h.id IN ({placeholders})
    """, senders).fetchall()
    return {r[0] for r in rows}


def _chat_id_filter_sql(column, allowed_chat_ids, params):
    """Build an 'AND <column> IN (...)' fragment for an optional chat_id
    allowlist, extending params in place. Returns '' if no filter."""
    if allowed_chat_ids is None:
        return ""
    placeholders = ",".join("?" for _ in allowed_chat_ids)
    params.extend(allowed_chat_ids)
    return f"AND {column} IN ({placeholders})"


def ingest_chats(src, tgt, allowed_chat_ids=None):
    rows = src.execute("""
        SELECT ROWID, chat_identifier, display_name, service_name, room_name
        FROM chat
    """).fetchall()
    if allowed_chat_ids is not None:
        rows = [r for r in rows if r[0] in allowed_chat_ids]
    for rowid, chat_identifier, display_name, service_name, room_name in rows:
        is_group = 1 if room_name else 0
        tgt.execute(
            "INSERT OR REPLACE INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, ?)",
            (rowid, chat_identifier, display_name, is_group, service_name),
        )

    participants = src.execute("SELECT chat_id, handle_id FROM chat_handle_join").fetchall()
    if allowed_chat_ids is not None:
        participants = [p for p in participants if p[0] in allowed_chat_ids]
    for chat_id, handle_id in participants:
        tgt.execute(
            "INSERT OR IGNORE INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            (chat_id, handle_id),
        )
    print(f"Ingested {len(rows)} chats")


def normalize_handle_id(raw_id):
    """Key used to join a chat.db handle.id against an AddressBook entry:
    digits-only, last-10 for phone-shaped ids; lowercased as-is for emails."""
    if raw_id is None:
        return None
    if "@" in raw_id:
        return raw_id.strip().lower()
    digits = "".join(ch for ch in raw_id if ch.isdigit())
    return digits[-10:] if digits else None


def build_contact_lookup(db_path=ADDRESS_BOOK_PATH):
    """Best-effort: read contacts out of a local macOS AddressBook
    (ZABCDRECORD/ZABCDPHONENUMBER/ZABCDEMAILADDRESS) sqlite file. Returns
    ([], {}) if db_path doesn't exist -- callers should treat that as "no
    contacts available".

    Note: some address books contain genuine duplicate ZABCDRECORD rows for
    the same person (same name, same phone number, distinct Z_PK) -- these
    are unmerged contact cards in the source data, not a query artifact, and
    Apple's own ZLINKID merge-linking column is NULL on them. There's no
    reliable ID to auto-merge them, so each ZABCDRECORD is kept as its own
    contact row; a handle maps to whichever contact record actually owns
    that phone number/email.

    Returns:
        contacts: list of (z_pk, source_uniqueid, first_name, last_name, nickname, organization)
                  -- one row per ZABCDRECORD contact.
        handle_to_contact: dict of normalized handle id -> owning z_pk.
    """
    if not Path(db_path).exists():
        return [], {}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        contacts = con.execute("""
            SELECT Z_PK, ZUNIQUEID, ZFIRSTNAME, ZLASTNAME, ZNICKNAME, ZORGANIZATION
            FROM ZABCDRECORD
            WHERE Z_ENT = (SELECT Z_ENT FROM Z_PRIMARYKEY WHERE Z_NAME = 'ABCDContact')
        """).fetchall()

        owner_rows = con.execute("""
            SELECT p.ZOWNER, p.ZFULLNUMBER, NULL AS email
            FROM ZABCDPHONENUMBER p
            UNION ALL
            SELECT e.ZOWNER, NULL AS phone, e.ZADDRESS
            FROM ZABCDEMAILADDRESS e
        """).fetchall()
    finally:
        con.close()

    handle_to_contact = {}
    for owner, phone, email in owner_rows:
        key = normalize_handle_id(phone if phone is not None else email)
        if key and key not in handle_to_contact:
            handle_to_contact[key] = owner
    return contacts, handle_to_contact


def ingest_contacts(tgt, contacts):
    for z_pk, source_uniqueid, first, last, nick, org in contacts:
        tgt.execute(
            "INSERT OR REPLACE INTO contacts"
            "(id, source_uniqueid, first_name, last_name, nickname, organization) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (z_pk, source_uniqueid, first, last, nick, org),
        )
    print(f"Ingested {len(contacts)} contacts")


def ingest_handles(src, tgt, contacts, handle_to_contact):
    rows = src.execute("SELECT ROWID, id, service FROM handle").fetchall()
    contacts_by_id = {c[0]: c for c in contacts}
    matched = 0
    for rowid, handle_identifier, service in rows:
        owner_id = handle_to_contact.get(normalize_handle_id(handle_identifier))
        contact = contacts_by_id.get(owner_id)
        if contact:
            matched += 1
            contact_id = contact[0]
        else:
            # No ingested contact owns this handle -- leave contact_id NULL so
            # the FK to contacts(id) never dangles (owner_id may point at a
            # ZABCDRECORD that wasn't ingested as an ABCDContact). Names/org are
            # resolved from contacts via contact_id, not stored on the handle.
            contact_id = None
        tgt.execute(
            "INSERT OR REPLACE INTO handles"
            "(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, ?)",
            (rowid, handle_identifier, contact_id, service),
        )
    print(f"Ingested {len(rows)} handles ({matched} matched to a contact)")


def ingest_messages_and_reactions(src, tgt, since_rowid=0, allowed_chat_ids=None):
    params = [since_rowid]
    chat_filter = _chat_id_filter_sql("cmj.chat_id", allowed_chat_ids, params)

    query = f"""
        SELECT
            m.ROWID, m.guid, cmj.chat_id, m.handle_id, m.is_from_me,
            m.text, m.attributedBody, m.date, m.date_read, m.date_delivered,
            m.item_type, m.is_emote, m.cache_has_attachments,
            m.associated_message_type, m.associated_message_guid
        FROM message m
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE m.ROWID > ? {chat_filter}
        ORDER BY m.ROWID
    """
    n_msg = 0
    n_react = 0
    n_failed = 0
    max_rowid = since_rowid

    for row in src.execute(query, params):
        (rowid, guid, chat_id, handle_id, is_from_me,
         raw_text, attributed_body, date, date_read, date_delivered,
         item_type, is_emote, has_attachment,
         assoc_type, assoc_guid) = row

        max_rowid = max(max_rowid, rowid)
        text, decode_failed = resolve_text(raw_text, attributed_body)
        if decode_failed:
            n_failed += 1

        if assoc_type and assoc_type != 0:
            # This row is a tapback/reaction on another message, not a message itself.
            tgt.execute(
                "INSERT OR REPLACE INTO reactions"
                "(id, target_message_guid, chat_id, handle_id, is_from_me, "
                " associated_message_type, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rowid, assoc_guid, chat_id, handle_id, is_from_me,
                 assoc_type, apple_ns_to_iso(date)),
            )
            n_react += 1
        else:
            tgt.execute(
                "INSERT OR REPLACE INTO messages"
                "(id, guid, chat_id, handle_id, is_from_me, text, decode_failed, "
                " date, date_read, date_delivered, item_type, is_emote, has_attachment) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rowid, guid, chat_id, handle_id, is_from_me, text, decode_failed,
                 apple_ns_to_iso(date), apple_ns_to_iso(date_read),
                 apple_ns_to_iso(date_delivered), item_type, is_emote,
                 1 if has_attachment else 0),
            )
            n_msg += 1

    set_state(tgt, "last_message_rowid", max_rowid)
    print(f"Ingested {n_msg} messages, {n_react} reactions, {n_failed} decode failures")
    return n_msg


def ingest_attachments(src, tgt, since_rowid=0, allowed_chat_ids=None):
    params = [since_rowid]
    join_clause = ""
    if allowed_chat_ids is not None:
        join_clause = "JOIN chat_message_join cmj ON cmj.message_id = maj.message_id"
    chat_filter = _chat_id_filter_sql("cmj.chat_id", allowed_chat_ids, params)

    rows = src.execute(f"""
        SELECT a.ROWID, maj.message_id, a.filename, a.mime_type, a.transfer_name
        FROM attachment a
        JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
        {join_clause}
        WHERE maj.message_id > ? {chat_filter}
    """, params).fetchall()
    for rowid, message_id, filename, mime_type, transfer_name in rows:
        tgt.execute(
            "INSERT OR REPLACE INTO attachments"
            "(id, message_id, filename, mime_type, transfer_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (rowid, message_id, filename, mime_type, transfer_name),
        )
    print(f"Ingested {len(rows)} attachments")


# ---------------------------------------------------------------------------
# Conversation chunks
# ---------------------------------------------------------------------------

# A message contributes a line to a chunk if it has text or an attachment;
# system events with neither stay chunk_id NULL and never trigger rebuilds.
_RENDERABLE = "(m.text IS NOT NULL OR m.has_attachment = 1)"

_MAX_HEADER_NAMES = 6


def _chunk_header(who, first_date):
    return f"Conversation with {who} on {first_date[:10]}:"


def _render_line(labels, att_types, msg):
    msg_id, is_from_me, handle_id, text, _date, _has_attachment = msg
    sender = "Me" if is_from_me else (labels.get(handle_id) or "Unknown")
    if text:
        return f"{sender}: {text}"
    kinds = att_types.get(msg_id)
    return f"{sender}: [attachment: {', '.join(kinds)}]" if kinds else f"{sender}: [attachment]"


def _chat_label(tgt, labels, chat_id):
    """Header name for a chat: its display_name, else its participants."""
    row = tgt.execute("SELECT display_name FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row and row[0]:
        return row[0]
    names = [labels[r[0]] for r in tgt.execute(
        "SELECT handle_id FROM chat_participants WHERE chat_id = ? ORDER BY handle_id",
        (chat_id,)) if labels.get(r[0])]
    if not names:
        return "unknown"
    label = ", ".join(names[:_MAX_HEADER_NAMES])
    if len(names) > _MAX_HEADER_NAMES:
        label += f" +{len(names) - _MAX_HEADER_NAMES} others"
    return label


def _flush_chunk(tgt, chat_id, who, chunk_msgs):
    """Write one chunk row and stamp chunk_id on its messages. A message
    carried into the next chunk by the size-split overlap keeps the first
    chunk_id it was assigned (the `chunk_id IS NULL` guard)."""
    first, last = chunk_msgs[0][0], chunk_msgs[-1][0]
    text = "\n".join([_chunk_header(who, first[4])] + [line for _, line in chunk_msgs])
    cur = tgt.execute(
        "INSERT INTO chunks(chat_id, first_message_id, last_message_id, "
        " first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, first[0], last[0], first[4], last[4], len(chunk_msgs), text),
    )
    ids = [m[0] for m, _ in chunk_msgs]
    placeholders = ",".join("?" * len(ids))
    tgt.execute(
        f"UPDATE messages SET chunk_id = ? WHERE chunk_id IS NULL AND id IN ({placeholders})",
        [cur.lastrowid] + ids,
    )


def build_chunks(tgt: sqlite3.Connection, gap_minutes=CHUNK_GAP_MINUTES,
                 max_chars=CHUNK_MAX_CHARS, overlap_messages=CHUNK_OVERLAP_MESSAGES):
    """Group not-yet-chunked messages into conversation chunks: per chat,
    ordered by (date, id), split when the gap between two messages exceeds
    gap_minutes (a session boundary) or the composed text would exceed
    max_chars (a size split, which carries the last overlap_messages messages
    into the next chunk for continuity -- unless the carried text alone would
    already overflow max_chars, in which case no overlap is carried so a
    chunk can never consist solely of repeated messages). Chunk text is a
    context header plus
    one "{sender}: {text}" line per message, so speakers and participants are
    part of what gets embedded.

    Incremental behavior: a chat that received new messages has its final
    chunk deleted and rebuilt from that chunk's first message, so a resumed
    conversation extends one coherent chunk instead of splitting at the
    ingest boundary. Chunks and their vectors are derived artifacts -- this
    is the one deletion in the pipeline, and it never touches source-derived
    rows. (If messages ever arrive with dates *earlier* than already-chunked
    ones, they form their own chunk out of id order -- still a valid chunk,
    just not merged into its older neighbors.)
    """
    pending_chats = [r[0] for r in tgt.execute(f"""
        SELECT DISTINCT m.chat_id FROM messages m
        WHERE m.chunk_id IS NULL AND m.chat_id IS NOT NULL
          AND m.date IS NOT NULL AND {_RENDERABLE}
    """)]
    if not pending_chats:
        print("Chunks up to date (0 built)")
        return 0

    labels = {row[0]: row[1] for row in tgt.execute(f"""
        SELECT h.id, {CONTACT_LABEL_SQL} FROM handles h
        LEFT JOIN contacts c ON c.id = h.contact_id
    """)}
    has_vec_table = tgt.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone() is not None

    gap = datetime.timedelta(minutes=gap_minutes)
    n_chunks_before = tgt.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_msgs = 0
    for chat_id in pending_chats:
        # Rebuild the chat's tail chunk so the pending messages extend it.
        # The rebuild must re-include the tail's overlap prefix -- messages
        # carried into it by a size split keep their *first* chunk's id, so
        # `chunk_id IS NULL` alone would miss them; `id >= first_message_id`
        # picks up exactly those (they stay stamped to their original chunk
        # via _flush_chunk's NULL guard).
        tail = tgt.execute(
            "SELECT id, first_message_id FROM chunks "
            "WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id,)).fetchone()
        rebuild_cond, rebuild_params = "m.chunk_id IS NULL", []
        if tail:
            if has_vec_table:
                tgt.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (tail[0],))
            tgt.execute("UPDATE messages SET chunk_id = NULL WHERE chunk_id = ?", (tail[0],))
            tgt.execute("DELETE FROM chunks WHERE id = ?", (tail[0],))
            rebuild_cond = "(m.chunk_id IS NULL OR m.id >= ?)"
            rebuild_params = [tail[1]]

        rows = tgt.execute(f"""
            SELECT m.id, m.is_from_me, m.handle_id, m.text, m.date, m.has_attachment
            FROM messages m
            WHERE m.chat_id = ? AND {rebuild_cond} AND m.date IS NOT NULL
              AND {_RENDERABLE}
            ORDER BY m.date, m.id
        """, [chat_id] + rebuild_params).fetchall()
        if not rows:
            continue

        att_types = {}
        for msg_id, kind in tgt.execute(f"""
            SELECT a.message_id, COALESCE(a.mime_type, a.transfer_name, 'file')
            FROM attachments a JOIN messages m ON m.id = a.message_id
            WHERE m.chat_id = ? AND {rebuild_cond}
        """, [chat_id] + rebuild_params):
            att_types.setdefault(msg_id, []).append(kind)

        who = _chat_label(tgt, labels, chat_id)
        chunk_msgs, cur_len, prev_dt = [], 0, None
        for msg in rows:
            line = _render_line(labels, att_types, msg)
            dt = datetime.datetime.fromisoformat(msg[4])
            if chunk_msgs and dt - prev_dt > gap:
                _flush_chunk(tgt, chat_id, who, chunk_msgs)
                chunk_msgs, cur_len = [], 0  # session boundary: no overlap carried
            elif chunk_msgs and cur_len + len(line) + 1 > max_chars:
                _flush_chunk(tgt, chat_id, who, chunk_msgs)
                chunk_msgs = chunk_msgs[-overlap_messages:] if overlap_messages else []
                cur_len = sum(len(li) + 1 for _, li in chunk_msgs)
                if cur_len + len(line) + 1 > max_chars:
                    # Oversized overlap (very long messages): carrying it would
                    # immediately overflow again and flush a chunk made only of
                    # already-chunked repeats -- start clean instead.
                    chunk_msgs, cur_len = [], 0
            chunk_msgs.append((msg, line))
            cur_len += len(line) + 1
            prev_dt = dt
        if chunk_msgs:
            _flush_chunk(tgt, chat_id, who, chunk_msgs)
        n_msgs += len(rows)

    tgt.commit()
    n_built = tgt.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] - n_chunks_before
    print(f"Built {n_built} conversation chunks covering {n_msgs} messages "
          f"({len(pending_chats)} chats)")
    return n_built


# ---------------------------------------------------------------------------
# Embeddings (sqlite-vec + Ollama)
# ---------------------------------------------------------------------------

def build_vec_table(tgt: sqlite3.Connection, dim: int):
    """Create chunk_vectors sized to the model's (detected) embedding
    dimension. Refuses to proceed if existing vectors were built by a
    different model or at a different dimension -- mixing embedding spaces
    silently corrupts search. Also drops the legacy per-message
    message_vectors table (replaced by chunk-level embeddings)."""
    load_vec_extension(tgt)
    tgt.execute("DROP TABLE IF EXISTS message_vectors")

    existing = tgt.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_vectors'").fetchone()
    prev_model = get_state(tgt, "embed_model")
    if existing:
        match = re.search(r"FLOAT\[(\d+)\]", existing[0], re.IGNORECASE)
        existing_dim = int(match.group(1)) if match else None
        if existing_dim != dim or (prev_model and prev_model != EMBED_MODEL):
            n_vectors = tgt.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            if n_vectors:
                raise SystemExit(
                    f"chunk_vectors holds {n_vectors} vectors from "
                    f"{prev_model or 'an unknown model'} (dim {existing_dim}), but "
                    f"EMBED_MODEL is now {EMBED_MODEL} (dim {dim}). Mixing embedding "
                    f"spaces corrupts search -- run reset_db.py and re-ingest."
                )
            tgt.execute("DROP TABLE chunk_vectors")

    # chat_id / first_day / last_day are vec0 *metadata columns*: sqlite-vec
    # (>= 0.1.6) applies WHERE constraints on them inside the KNN scan, so
    # scoped/date-bounded semantic search needs no over-fetch + post-filter.
    # Days are integers (db.iso_to_day) -- TEXT metadata longer than 12 bytes
    # is slow to filter on. cosine, not the L2 default: gemma4's mean-pooled
    # embeddings aren't length-normalized.
    tgt.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}] distance_metric=cosine,
            chat_id INTEGER,
            first_day INTEGER,
            last_day INTEGER
        )
    """)
    set_state(tgt, "embed_model", EMBED_MODEL)
    set_state(tgt, "embed_dim", dim)


def embed_chunks(
    tgt: sqlite3.Connection,
    min_chars=CHUNK_MIN_CHARS,
    batch_size=EMBED_BATCH_SIZE,
    max_workers=EMBED_MAX_WORKERS,
    batch_log_every=200,
):
    rows = tgt.execute("""
        SELECT c.id, c.chat_id, c.first_date, c.last_date, c.text
        FROM chunks c
        LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
        WHERE v.chunk_id IS NULL
          AND length(c.text) >= ?
    """, (min_chars,)).fetchall()

    print(f"Embedding {len(rows)} conversation chunks via Ollama ({EMBED_MODEL})...")
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def process(batch):
        return batch, embed_texts([r[4] for r in batch], kind="document")

    n_done, n_err, n_processed, n_logged = 0, 0, 0, 0
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_batch = {pool.submit(process, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch, vecs = future.result()
            except (http.client.HTTPException, OSError, RuntimeError, json.JSONDecodeError):
                n_err += len(batch)
                continue
            tgt.executemany(
                "INSERT OR REPLACE INTO chunk_vectors"
                "(chunk_id, embedding, chat_id, first_day, last_day) VALUES (?, ?, ?, ?, ?)",
                [(chunk_id, sqlite_vec.serialize_float32(vec), chat_id,
                  iso_to_day(first_date), iso_to_day(last_date))
                 for (chunk_id, chat_id, first_date, last_date, _), vec in zip(batch, vecs)],
            )
            n_done += len(batch)
            n_processed += len(batch)
            if n_processed - n_logged >= batch_log_every:
                tgt.commit()
                print(f"  ...{n_processed}/{len(rows)}")
                n_logged = n_processed
    except KeyboardInterrupt:
        # Don't wait on in-flight worker threads (each blocked on an Ollama
        # HTTP call) -- drop anything not yet started and commit whatever
        # finished so it isn't lost.
        print("\nInterrupted -- cancelling remaining embedding batches...")
        pool.shutdown(wait=False, cancel_futures=True)
        tgt.commit()
        print(f"Embedded {n_done} chunks before interrupt ({n_err} errors)")
        raise
    pool.shutdown(wait=True)
    tgt.commit()
    print(f"Embedded {n_done} chunks ({n_err} errors)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding step")
    ap.add_argument("--since-last", action="store_true",
                     help="only ingest messages newer than the last recorded run")
    ap.add_argument("-s", "--senders", default=None,
                     help="comma-separated list of phone numbers/emails; only ingest "
                          "chats (direct or group) that include at least one of them")
    args = ap.parse_args()

    senders = None
    if args.senders:
        senders = [s.strip() for s in args.senders.split(",") if s.strip()]

    snapshot_source(SOURCE_DB, SNAPSHOT_DB)

    src = sqlite3.connect(f"file:{SNAPSHOT_DB}?mode=ro", uri=True)
    try:
        tgt = sqlite3.connect(TARGET_DB)
        try:
            # Loaded unconditionally (not just before embedding): the chunk
            # builder deletes a rebuilt tail chunk's row from chunk_vectors,
            # which needs the vec0 module registered even under --no-embed.
            load_vec_extension(tgt)
            # Enforce the handles.contact_id -> contacts(id) FK. Must run
            # before any transaction is opened, so it's issued right after
            # connecting and before any DML.
            tgt.execute("PRAGMA foreign_keys = ON")
            create_schema(tgt)

            since_rowid = 0
            if args.since_last:
                since_rowid = int(get_state(tgt, "last_message_rowid", 0))
                print(f"Incremental run: ingesting messages with ROWID > {since_rowid}")

            allowed_chat_ids = resolve_allowed_chat_ids(src, senders)
            if senders is not None:
                if allowed_chat_ids:
                    print(f"Sender filter: {len(allowed_chat_ids)} chat(s) match {senders}")
                else:
                    print(f"Sender filter: no chats found matching {senders} -- check the "
                          f"phone/email format (e.g. +12015551234)")

            ingest_chats(src, tgt, allowed_chat_ids)
            contacts, handle_to_contact = build_contact_lookup()
            ingest_contacts(tgt, contacts)
            ingest_handles(src, tgt, contacts, handle_to_contact)
            ingest_messages_and_reactions(src, tgt, since_rowid, allowed_chat_ids)
            ingest_attachments(src, tgt, since_rowid, allowed_chat_ids)
            tgt.commit()

            # Chunk building is cheap and local -- always runs. Only the
            # Ollama embedding pass is gated by --no-embed.
            build_chunks(tgt)

            if not args.no_embed:
                try:
                    dim = detect_embed_dim()
                except (http.client.HTTPException, OSError, RuntimeError) as e:
                    raise SystemExit(
                        f"Cannot reach Ollama to embed ({e}) -- is it running? "
                        f"Re-run with --no-embed to skip embedding."
                    ) from None
                build_vec_table(tgt, dim)
                embed_chunks(tgt)
                tgt.commit()
            else:
                print("Skipping embedding step (--no-embed)")
        finally:
            tgt.close()
    finally:
        src.close()
    print(f"Done. Analysis DB at {TARGET_DB}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
        sys.exit(130)
