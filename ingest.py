"""
iMessage chat.db -> normalized SQLite analysis database.

Pipeline:
  1. Snapshot chat.db safely (read-only backup, never touches the live file)
  2. Decode attributedBody blobs (streamtyped fallback for NULL `text` rows)
  3. Join message/handle/chat/attachment tables into a normalized schema,
     enriching handles with contact names from ./address_book.abcddb if present
  4. Build an FTS5 index over message text
  5. Group messages into conversation chunks and embed them into a sqlite-vec
     table -- via local Ollama (nomic-embed-text), or for EMBED_MODEL=voyage/<id>
     via the Voyage Batch API (submit, save the batch ids in ingest_state, poll,
     collect; an interrupted run's batches are finished first next time). Both
     stages live in chunking.py; this module runs them (and re-exports them)

Usage:
    python3 ingest.py                  # full pipeline
    python3 ingest.py --no-embed       # skip embedding step (e.g. Ollama not running)
    python3 ingest.py --since-last     # incremental: only ingest new messages since last run
    python3 ingest.py --senders="+12015551234,alice@example.com"
                                        # only ingest chats (direct or group) containing
                                        # at least one of the given phone numbers/emails
    python3 ingest.py --exclude="+12015551234,alice@example.com"
                                        # ingest everything EXCEPT chats (direct or group)
                                        # containing any of the given phone numbers/emails
    python3 ingest.py --lookback=3mo    # only ingest messages from the last 3 months
                                        # (units d/w/mo/y); an extra lower bound on
                                        # message.date, combines with the flags above
    python3 ingest.py --budget 2 --max-parallel 4
                                        # voyage/ EMBED_MODEL only: USD spend cap
                                        # (else HOSTED_BUDGET_USD) and requests in flight
"""

import argparse
import calendar
import datetime
import http.client
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import progress
from chunking import (
    build_chunks,
    build_vec_table,
    embed_chunks,
    embed_chunks_voyage,
    unembedded_chunks,
)
from config import (
    ADDRESS_BOOK_PATH,
    EMBED_MODEL,
    HOSTED_MAX_PARALLEL,
    SNAPSHOT_DB,
    SOURCE_DB,
    TARGET_DB,
    VOYAGE_API_KEY,
    hosted_model_id,
    is_voyage_model,
)
from db import get_state, load_vec_extension, set_state, strip_reaction_guid_prefix
from lexicon_build import build_dyad_terms, build_person_period_lexicon
from sessions import build_sessions
# embed_texts is otherwise unused here, but it must stay an attribute of
# this module: chunking.embed_chunks resolves `ingest.embed_texts` at call
# time, keeping it the one patchable seam (tests stub it here).
from embeddings import detect_embed_dim, embed_texts  # noqa: F401
# create_schema must stay importable as ingest.create_schema -- tests and
# callers reach it through this module.
from schema import create_schema

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


# ---------------------------------------------------------------------------
# Lookback window (--lookback)
# ---------------------------------------------------------------------------

_PERIOD_RE = re.compile(r"^\s*(\d+)\s*(d|w|mo|y)\s*$", re.IGNORECASE)


def _parse_lookback(value):
    """'1d'/'3mo'/'2y' -> (amount, unit); None passes through as "no filter".
    Units: d=days, w=weeks, mo=months, y=years (bare 'm' is rejected to avoid
    the minutes/months ambiguity). Raises SystemExit on malformed input."""
    if value is None:
        return None
    m = _PERIOD_RE.match(value)
    if not m:
        raise SystemExit(
            f"error: --lookback expects a period like 7d, 3mo, or 2y "
            f"(units: d=days, w=weeks, mo=months, y=years); got {value!r}"
        )
    amount = int(m.group(1))
    if amount <= 0:
        raise SystemExit("error: --lookback period must be a positive number")
    return amount, m.group(2).lower()


def _subtract_period(when, amount, unit):
    """`when` minus (amount, unit). d/w use timedelta; mo/y use calendar math,
    clamping the day to the target month's last valid day (so Mar 31 - 1mo ->
    Feb 28/29 rather than an invalid date)."""
    if unit == "d":
        return when - datetime.timedelta(days=amount)
    if unit == "w":
        return when - datetime.timedelta(weeks=amount)
    months = amount * (12 if unit == "y" else 1)
    total = when.year * 12 + (when.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(when.day, calendar.monthrange(year, month)[1])
    return when.replace(year=year, month=month, day=day)


def _now_utc_naive():
    """Wall-clock UTC as a naive datetime -- the space APPLE_EPOCH and
    apple_ns_to_iso live in (messages.date is unlabeled UTC)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _lookback_cutoff_ns(period, now=None):
    """(amount, unit) -> Apple-epoch-ns lower bound for message.date; None ->
    None. `now` defaults to wall-clock UTC; passed in only by tests."""
    if period is None:
        return None
    cutoff = _subtract_period(now or _now_utc_naive(), *period)
    delta = cutoff - APPLE_EPOCH
    # Integer ns to avoid float rounding at the ~1e18 scale (>2**53).
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


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
    # Checked before the copy loop, because that loop reads a missing path as
    # "delete the stale destination sidecar" -- right for -wal/-shm, silently
    # wrong for the main file. Without this, sqlite3.connect(dest) below
    # CREATES an empty snapshot, we print a success line for a database with
    # no messages in it, and the run dies four stages later in ingest_chats
    # with "no such table: chat" -- an error naming neither this file nor
    # source_data/.
    if not source.exists():
        raise SystemExit(
            f"error: no source database at {source}\n"
            f"  Copy your Messages database in with:  make source-data\n"
            f"  (or point SOURCE_DB at an existing copy -- see .env.example)"
        )

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
# Ingestion
# ---------------------------------------------------------------------------

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


def all_chat_ids(src):
    """Every chat ROWID in the source db -- used to turn a bare --exclude
    (no --senders) into an explicit allowlist of everything-but-excluded."""
    return {r[0] for r in src.execute("SELECT ROWID FROM chat")}


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


# A true upsert (never INSERT OR REPLACE -- see docs/DECISIONS.md, "message
# upsert"): source columns are refreshed in place, derived columns (chunk_id)
# survive. Contract: an edited message keeps its chunk_id, so its chunk text may
# be stale. ON CONFLICT covers only `id`; a `guid` clash is resolved first by
# _claim_rowid.
_MESSAGE_SOURCE_COLS = (
    "guid", "chat_id", "handle_id", "is_from_me", "text", "decode_failed",
    "date", "date_read", "date_delivered", "item_type", "is_emote", "has_attachment")
_MESSAGE_UPSERT_SQL = (
    f"INSERT INTO messages(id, {', '.join(_MESSAGE_SOURCE_COLS)}) "
    f"VALUES ({', '.join('?' * (len(_MESSAGE_SOURCE_COLS) + 1))}) "
    "ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _MESSAGE_SOURCE_COLS))


# Every column outside `messages` that stores a messages.id. A re-key must
# carry all of them, or the row's chunk/session/nudge/attachment links dangle.
_MESSAGE_ID_REFS = (
    ("attachments", "message_id"),
    ("chunks", "first_message_id"), ("chunks", "last_message_id"),
    ("sessions", "first_message_id"), ("sessions", "last_message_id"),
    ("dyad_terms", "first_message_id"),
    ("loose_threads", "anchor_message_id"),
    ("conflict_signals", "message_id"),
)


def _move_message(tgt, old_id, new_id):
    """Re-key one messages row and every stored reference to it. messages_au
    moves its FTS entry; chunk_id and all other columns travel with the row."""
    tgt.execute("UPDATE messages SET id = ? WHERE id = ?", (new_id, old_id))
    for table, col in _MESSAGE_ID_REFS:
        tgt.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (new_id, old_id))
    # Ids inside JSON: an event cites a message of its own chunk only.
    for chunk_id, events_json in tgt.execute(
            "SELECT a.chunk_id, a.events_json FROM chunk_annotations a "
            "JOIN messages m ON m.chunk_id = a.chunk_id WHERE m.id = ?",
            (new_id,)).fetchall():
        events = json.loads(events_json or "[]")
        if any(ev.get("message_id") == old_id for ev in events):
            for ev in events:
                if ev.get("message_id") == old_id:
                    ev["message_id"] = new_id
            tgt.execute("UPDATE chunk_annotations SET events_json = ? WHERE chunk_id = ?",
                        (json.dumps(events), chunk_id))
    tgt.execute(
        "UPDATE conflicts SET peak_signal_json = "
        "json_set(peak_signal_json, '$.peak_gap_message_id', ?) "
        "WHERE json_extract(peak_signal_json, '$.peak_gap_message_id') = ?",
        (new_id, old_id))


def _claim_rowid(tgt, rowid, guid):
    """Make (rowid, guid) safe for _MESSAGE_UPSERT_SQL when chat.db has moved
    messages to new ROWIDs (iCloud resync, restore onto a new Mac): guid is the
    message's identity, ROWID is only its current address.

    - Another message holds `rowid`: park it at a fresh negative id (a row with
      no current source ROWID) so the upsert cannot overwrite it. Its guid
      re-arriving later moves it to its new ROWID; until then it stays, since
      ingest never deletes a message.
    - This guid sits at another id: move that row to `rowid`, keeping its
      chunk_id and references, so the newest source row wins without a re-chunk.
    """
    held = tgt.execute("SELECT guid FROM messages WHERE id = ?", (rowid,)).fetchone()
    if held and held[0] != guid:
        park = tgt.execute(
            "SELECT MIN(COALESCE(MIN(id), 0), 0) - 1 FROM messages").fetchone()[0]
        _move_message(tgt, rowid, park)
    prev = tgt.execute("SELECT id FROM messages WHERE guid = ?", (guid,)).fetchone()
    if prev and prev[0] != rowid:
        _move_message(tgt, prev[0], rowid)


def ingest_messages_and_reactions(src, tgt, since_rowid=0, allowed_chat_ids=None,
                                  since_date_ns=None):
    params = [since_rowid]
    chat_filter = _chat_id_filter_sql("cmj.chat_id", allowed_chat_ids, params)
    # Appended AFTER the chat filter so the placeholder order stays
    # [since_rowid, *chat_ids, since_date_ns] for both queries below.
    date_filter = ""
    if since_date_ns is not None:
        date_filter = "AND m.date >= ?"
        params.append(since_date_ns)

    query = f"""
        SELECT
            m.ROWID, m.guid, cmj.chat_id, m.handle_id, m.is_from_me,
            m.text, m.attributedBody, m.date, m.date_read, m.date_delivered,
            m.item_type, m.is_emote, m.cache_has_attachments,
            m.associated_message_type, m.associated_message_guid
        FROM message m
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE m.ROWID > ? {chat_filter} {date_filter}
        ORDER BY m.ROWID
    """
    n_msg = 0
    n_react = 0
    n_failed = 0
    max_rowid = since_rowid

    # The row loop streams its cursor (the corpus is too big to fetchall), so
    # the total has to be asked for separately to report a percentage. Same
    # FROM/WHERE as the stream so the count reflects exactly what follows.
    n_total = src.execute(f"""
        SELECT COUNT(*)
        FROM message m
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE m.ROWID > ? {chat_filter} {date_filter}
    """, params).fetchone()[0]

    with progress.track("Ingesting messages", total=n_total) as p:
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
                    (rowid, strip_reaction_guid_prefix(assoc_guid), chat_id, handle_id,
                     is_from_me, assoc_type, apple_ns_to_iso(date)),
                )
                n_react += 1
            else:
                _claim_rowid(tgt, rowid, guid)
                tgt.execute(_MESSAGE_UPSERT_SQL,
                    (rowid, guid, chat_id, handle_id, is_from_me, text, decode_failed,
                     apple_ns_to_iso(date), apple_ns_to_iso(date_read),
                     apple_ns_to_iso(date_delivered), item_type, is_emote,
                     1 if has_attachment else 0),
                )
                n_msg += 1
            p.advance()

    set_state(tgt, "last_message_rowid", max_rowid)
    print(f"Ingested {n_msg} messages, {n_react} reactions, {n_failed} decode failures")
    return n_msg


def ingest_attachments(src, tgt, since_rowid=0, allowed_chat_ids=None,
                       since_date_ns=None):
    params = [since_rowid]
    join_clause = ""
    if allowed_chat_ids is not None:
        join_clause = "JOIN chat_message_join cmj ON cmj.message_id = maj.message_id"
    chat_filter = _chat_id_filter_sql("cmj.chat_id", allowed_chat_ids, params)
    # Attachments carry no date of their own; join `message` to reach it so the
    # attachment set stays aligned with the (date-filtered) message set.
    date_join = date_filter = ""
    if since_date_ns is not None:
        date_join = "JOIN message m ON m.ROWID = maj.message_id"
        date_filter = "AND m.date >= ?"
        params.append(since_date_ns)

    rows = src.execute(f"""
        SELECT a.ROWID, maj.message_id, a.filename, a.mime_type, a.transfer_name
        FROM attachment a
        JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
        {join_clause}
        {date_join}
        WHERE maj.message_id > ? {chat_filter} {date_filter}
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
# Main
# ---------------------------------------------------------------------------

def _split_csv(value):
    """Comma-separated phone/email string -> list of trimmed non-empty items,
    or None when the flag was not given (the "no filter" sentinel)."""
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def _build_parser():
    """The CLI's flag surface, extracted from main() so it is greppable and
    testable on its own."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding step")
    ap.add_argument("--since-last", action="store_true",
                     help="only ingest messages newer than the last recorded run")
    ap.add_argument("-s", "--senders", default=None,
                     help="comma-separated list of phone numbers/emails; only ingest "
                          "chats (direct or group) that include at least one of them")
    ap.add_argument("-x", "--exclude", default=None,
                     help="comma-separated list of phone numbers/emails; skip any "
                          "chat (direct or group) that includes at least one of them")
    ap.add_argument("-l", "--lookback", default=None, metavar="PERIOD",
                     help="only ingest messages from the last PERIOD, e.g. 7d, 3mo, 2y "
                          "(units d=days/w=weeks/mo=months/y=years); an extra lower "
                          "bound on message.date, combines with --since-last/--senders/"
                          "--exclude")
    ap.add_argument("--budget", type=float, default=None, metavar="USD",
                     help="voyage/ EMBED_MODEL only: spend cap for this run "
                          "(else HOSTED_BUDGET_USD; one of the two is required)")
    ap.add_argument("--max-parallel", type=int, default=None, metavar="N",
                     help="voyage/ EMBED_MODEL only: max Voyage requests in flight "
                          f"(default HOSTED_MAX_PARALLEL={HOSTED_MAX_PARALLEL})")
    return ap


def _embed_voyage(tgt, budget, max_parallel):
    """The voyage/ embedding step. A HostedStop becomes an exit naming the
    cause and how many chunks are still pending."""
    import hosted

    hosted.upload_notice(f"Voyage AI ({hosted_model_id(EMBED_MODEL)})", "chunk text")
    build_vec_table(tgt, detect_embed_dim())
    tgt.commit()
    try:
        embed_chunks_voyage(tgt, budget, max_parallel)
    except hosted.HostedStop as e:
        pending = len(unembedded_chunks(tgt))
        raise SystemExit(f"error: Voyage run stopped: {e.cause}\n"
                         f"  {pending} chunks still have no vector; re-run ingest to "
                         f"resume.") from None


def main():
    ap = _build_parser()
    args = ap.parse_args()
    use_voyage = is_voyage_model(EMBED_MODEL)
    if not use_voyage and (args.budget is not None or args.max_parallel is not None):
        ap.error("--budget / --max-parallel only apply to hosted models "
                 f"(EMBED_MODEL={EMBED_MODEL})")
    if args.no_embed and (args.budget is not None or args.max_parallel is not None):
        ap.error("--budget / --max-parallel have no effect with --no-embed")

    senders = _split_csv(args.senders)
    excluded = _split_csv(args.exclude)
    # Parse-and-validate before the (potentially multi-GB) snapshot copy so a
    # typo fails instantly rather than minutes in.
    lookback = _parse_lookback(args.lookback)
    if use_voyage and not args.no_embed:
        import hosted

        if not VOYAGE_API_KEY.strip():
            raise SystemExit(f"EMBED_MODEL={EMBED_MODEL} needs VOYAGE_API_KEY (set it in "
                             f".env), or re-run with --no-embed")
        budget = hosted.Budget(hosted.resolve_budget_usd(args.budget))
        max_parallel = hosted.check_max_parallel(
            HOSTED_MAX_PARALLEL if args.max_parallel is None else args.max_parallel)

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

            since_date_ns = None
            if lookback:
                # Sample "now" once so the printed date matches the applied cutoff.
                now = _now_utc_naive()
                since_date_ns = _lookback_cutoff_ns(lookback, now=now)
                cutoff = _subtract_period(now, *lookback)
                print(f"Lookback filter: ingesting messages on/after "
                      f"{cutoff.date().isoformat()} (last {args.lookback})")

            allowed_chat_ids = resolve_allowed_chat_ids(src, senders)
            if senders is not None:
                if allowed_chat_ids:
                    print(f"Sender filter: {len(allowed_chat_ids)} chat(s) match {senders}")
                else:
                    print(f"Sender filter: no chats found matching {senders} -- check the "
                          f"phone/email format (e.g. +12015551234)")

            if excluded is not None:
                excluded_chat_ids = resolve_allowed_chat_ids(src, excluded) or set()
                if excluded_chat_ids:
                    # No --senders means "ingest everything"; realize that as an
                    # explicit allowlist so exclusion threads through the same
                    # chat_id plumbing (subtraction otherwise, when --senders set).
                    if allowed_chat_ids is None:
                        allowed_chat_ids = all_chat_ids(src) - excluded_chat_ids
                    else:
                        allowed_chat_ids = allowed_chat_ids - excluded_chat_ids
                    print(f"Exclude filter: dropping {len(excluded_chat_ids)} chat(s) "
                          f"matching {excluded}")
                else:
                    print(f"Exclude filter: no chats found matching {excluded} -- check "
                          f"the phone/email format (e.g. +12015551234)")

            ingest_chats(src, tgt, allowed_chat_ids)
            contacts, handle_to_contact = build_contact_lookup()
            ingest_contacts(tgt, contacts)
            ingest_handles(src, tgt, contacts, handle_to_contact)
            ingest_messages_and_reactions(src, tgt, since_rowid, allowed_chat_ids,
                                          since_date_ns)
            ingest_attachments(src, tgt, since_rowid, allowed_chat_ids, since_date_ns)
            tgt.commit()

            # Chunk building is cheap and local -- always runs. Only the
            # Ollama embedding pass is gated by --no-embed.
            build_chunks(tgt)

            # Derived analytics artifacts, all cheap and local: sessions is
            # incremental like chunks; the lexicon/dyad-term tables are full
            # rebuilds (a few seconds, and dyad_terms preserves curation).
            build_sessions(tgt)
            build_person_period_lexicon(tgt)
            build_dyad_terms(tgt)

            if not args.no_embed and use_voyage:
                _embed_voyage(tgt, budget, max_parallel)
            elif not args.no_embed:
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
