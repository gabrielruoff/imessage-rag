"""
iMessage chat.db -> normalized SQLite analysis database.

Pipeline:
  1. Snapshot chat.db safely (read-only backup, never touches the live file)
  2. Decode attributedBody blobs (streamtyped fallback for NULL `text` rows)
  3. Join message/handle/chat/attachment tables into a normalized schema,
     enriching handles with contact names from ./address_book.abcddb if present
  4. Build an FTS5 index over message text
  5. Build a sqlite-vec table and embed messages via local Ollama (nomic-embed-text)

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
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sqlite_vec

SOURCE_DB = Path("chat.db")
SNAPSHOT_DB = Path("chat_snapshot.db")
TARGET_DB = Path("imessage_analysis.db")
ADDRESS_BOOK_PATH = Path("address_book.abcddb")

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_EMBED_PATH = "/api/embed"  # batch endpoint: {"input": [...]} -> {"embeddings": [...]}
OLLAMA_TIMEOUT = 60
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text output dimension
EMBED_BATCH_SIZE = 64   # texts per Ollama request
EMBED_MAX_WORKERS = 4   # concurrent requests, matches Ollama's default OLLAMA_NUM_PARALLEL

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

CREATE TABLE IF NOT EXISTS handles (
    id INTEGER PRIMARY KEY,
    contact_id TEXT,
    service TEXT,
    first_name TEXT,
    last_name TEXT,
    nickname TEXT,
    organization TEXT
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
    has_attachment INTEGER DEFAULT 0
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
CREATE INDEX IF NOT EXISTS idx_reactions_target ON reactions(target_message_guid);
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

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def create_schema(con: sqlite3.Connection):
    con.executescript(SCHEMA)
    con.executescript(FTS_SCHEMA)
    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(handles)")}
    for col in ("first_name", "last_name", "nickname", "organization"):
        if col not in existing_cols:
            con.execute(f"ALTER TABLE handles ADD COLUMN {col} TEXT")
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
    """Best-effort: read contact names out of a local macOS AddressBook
    (ZABCDRECORD/ZABCDPHONENUMBER/ZABCDEMAILADDRESS) sqlite file. Returns {}
    if db_path doesn't exist -- callers should treat that as "no names available"."""
    if not Path(db_path).exists():
        return {}

    lookup = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("""
            SELECT p.ZFULLNUMBER, NULL AS email,
                   r.ZFIRSTNAME, r.ZLASTNAME, r.ZNICKNAME, r.ZORGANIZATION
            FROM ZABCDPHONENUMBER p
            JOIN ZABCDRECORD r ON r.Z_PK = p.ZOWNER
            UNION ALL
            SELECT NULL AS phone, e.ZADDRESS,
                   r.ZFIRSTNAME, r.ZLASTNAME, r.ZNICKNAME, r.ZORGANIZATION
            FROM ZABCDEMAILADDRESS e
            JOIN ZABCDRECORD r ON r.Z_PK = e.ZOWNER
        """).fetchall()
    finally:
        con.close()

    for phone, email, first, last, nick, org in rows:
        key = normalize_handle_id(phone if phone is not None else email)
        if key and key not in lookup:
            lookup[key] = (first, last, nick, org)
    return lookup


def ingest_handles(src, tgt):
    rows = src.execute("SELECT ROWID, id, service FROM handle").fetchall()
    contacts = build_contact_lookup()
    matched = 0
    for rowid, contact_id, service in rows:
        match = contacts.get(normalize_handle_id(contact_id))
        if match:
            matched += 1
            first_name, last_name, nickname, organization = match
        else:
            first_name = last_name = nickname = organization = None
        tgt.execute(
            "INSERT OR REPLACE INTO handles"
            "(id, contact_id, service, first_name, last_name, nickname, organization) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rowid, contact_id, service, first_name, last_name, nickname, organization),
        )
    print(f"Ingested {len(rows)} handles ({matched} matched to a contact name)")


def ingest_messages_and_reactions(src, tgt, since_rowid=0, allowed_chat_ids=None):
    params = [since_rowid]
    chat_filter = ""
    if allowed_chat_ids is not None:
        placeholders = ",".join("?" for _ in allowed_chat_ids)
        chat_filter = f"AND cmj.chat_id IN ({placeholders})"
        params.extend(allowed_chat_ids)

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
    chat_filter = ""
    if allowed_chat_ids is not None:
        placeholders = ",".join("?" for _ in allowed_chat_ids)
        join_clause = "JOIN chat_message_join cmj ON cmj.message_id = maj.message_id"
        chat_filter = f"AND cmj.chat_id IN ({placeholders})"
        params.extend(allowed_chat_ids)

    rows = src.execute(f"""
        SELECT a.ROWID, maj.message_id, a.filename, a.mime_type, a.transfer_name
        FROM attachment a
        JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
        {join_clause}
        WHERE maj.message_id > ? {chat_filter}
    """, params).fetchall()
    for rowid, message_id, filename, mime_type, transfer_name in rows:
        tgt.execute(
            "INSERT OR REPLACE INTO attachments(id, message_id, filename, mime_type, transfer_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (rowid, message_id, filename, mime_type, transfer_name),
        )
    print(f"Ingested {len(rows)} attachments")


def prune_out_of_scope(tgt: sqlite3.Connection, allowed_chat_ids):
    """Remove rows for chats outside the current --senders filter that were
    left behind by an earlier (unfiltered or differently-filtered) run.
    Ensures the target DB always reflects only the current filter, not just
    the rows touched by this particular ingest call. No-op if `senders`
    matched zero chats -- that's almost certainly a typo, and wiping the
    whole DB on a typo would be far worse than leaving stale data."""
    if not allowed_chat_ids:
        return

    ids = list(allowed_chat_ids)
    placeholders = ",".join("?" for _ in ids)

    tgt.execute(
        f"DELETE FROM attachments WHERE message_id IN "
        f"(SELECT id FROM messages WHERE chat_id NOT IN ({placeholders}) OR chat_id IS NULL)",
        ids,
    )
    tgt.execute(
        f"DELETE FROM reactions WHERE chat_id NOT IN ({placeholders}) OR chat_id IS NULL", ids
    )
    n_before = tgt.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    tgt.execute(
        f"DELETE FROM messages WHERE chat_id NOT IN ({placeholders}) OR chat_id IS NULL", ids
    )
    n_pruned = n_before - tgt.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    if tgt.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vectors'"
    ).fetchone():
        tgt.enable_load_extension(True)
        sqlite_vec.load(tgt)
        tgt.enable_load_extension(False)
        tgt.execute("DELETE FROM message_vectors WHERE message_id NOT IN (SELECT id FROM messages)")

    tgt.execute(f"DELETE FROM chat_participants WHERE chat_id NOT IN ({placeholders})", ids)
    tgt.execute(f"DELETE FROM chats WHERE id NOT IN ({placeholders})", ids)

    if n_pruned:
        print(f"Pruned {n_pruned} out-of-scope message(s) left over from a prior run")


# ---------------------------------------------------------------------------
# Embeddings (sqlite-vec + Ollama)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _ollama_connection() -> http.client.HTTPConnection:
    """One persistent keep-alive connection per worker thread."""
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_TIMEOUT)
        _thread_local.conn = conn
    return conn


def ollama_embed_batch(texts: list) -> list:
    """Embed a batch of texts in a single Ollama request. Returns one vector per input."""
    conn = _ollama_connection()
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    try:
        conn.request(
            "POST", OLLAMA_EMBED_PATH, body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read())
    except (http.client.HTTPException, OSError):
        # connection may be dead; drop it so the next call on this thread reopens one
        _thread_local.conn = None
        raise
    return body["embeddings"]


def build_vec_table(tgt: sqlite3.Connection):
    tgt.enable_load_extension(True)
    sqlite_vec.load(tgt)
    tgt.enable_load_extension(False)
    tgt.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS message_vectors USING vec0(
            message_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBED_DIM}]
        )
    """)


def embed_messages(
    tgt: sqlite3.Connection,
    min_chars=3,
    batch_size=EMBED_BATCH_SIZE,
    max_workers=EMBED_MAX_WORKERS,
    batch_log_every=200,
):
    rows = tgt.execute("""
        SELECT m.id, m.text FROM messages m
        LEFT JOIN message_vectors v ON v.message_id = m.id
        WHERE v.message_id IS NULL
          AND m.text IS NOT NULL
          AND length(m.text) >= ?
    """, (min_chars,)).fetchall()

    print(f"Embedding {len(rows)} messages via Ollama ({EMBED_MODEL})...")
    chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def process(chunk):
        texts = [text for _, text in chunk]
        vecs = ollama_embed_batch(texts)
        return chunk, vecs

    n_done, n_err, n_processed, n_logged = 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_chunk = {pool.submit(process, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk, vecs = future.result()
            except (http.client.HTTPException, OSError, KeyError, json.JSONDecodeError):
                n_err += len(chunk)
                continue
            tgt.executemany(
                "INSERT OR REPLACE INTO message_vectors(message_id, embedding) VALUES (?, ?)",
                [(msg_id, sqlite_vec.serialize_float32(vec))
                 for (msg_id, _), vec in zip(chunk, vecs)],
            )
            n_done += len(chunk)
            n_processed += len(chunk)
            if n_processed - n_logged >= batch_log_every:
                tgt.commit()
                print(f"  ...{n_processed}/{len(rows)}")
                n_logged = n_processed
    tgt.commit()
    print(f"Embedded {n_done} messages ({n_err} errors)")


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
            ingest_handles(src, tgt)
            ingest_messages_and_reactions(src, tgt, since_rowid, allowed_chat_ids)
            ingest_attachments(src, tgt, since_rowid, allowed_chat_ids)
            if senders is not None:
                prune_out_of_scope(tgt, allowed_chat_ids)
            tgt.commit()

            if not args.no_embed:
                build_vec_table(tgt)
                embed_messages(tgt)
                tgt.commit()
            else:
                print("Skipping embedding step (--no-embed)")
        finally:
            tgt.close()
    finally:
        src.close()
    print(f"Done. Analysis DB at {TARGET_DB}")


if __name__ == "__main__":
    main()
