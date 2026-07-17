"""
DDL for imessage_analysis.db (tables, indexes, FTS5 triggers) plus the
in-place migration of pre-chunking / old-layout databases. ingest.py runs
create_schema at the start of every pipeline run and re-exports it, so
`ingest.create_schema` keeps working for callers and tests.
"""

import sqlite3

from db import strip_reaction_guid_prefix

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

-- Derived analytics artifacts (like chunks: rebuildable, reset_db deletes
-- them, the append-only rule's carve-out applies). sessions/-lexicon/
-- dyad_terms are built by ingest; chunk_annotations by annotate.py;
-- topics/chunk_topics by topics.py.

CREATE TABLE IF NOT EXISTS sessions (
    -- AUTOINCREMENT: rebuilt per chat, ids must never be reused
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    first_message_id INTEGER,
    last_message_id INTEGER,
    first_date TEXT,
    last_date TEXT,
    n_messages INTEGER,
    n_from_me INTEGER,
    initiator_handle_id INTEGER,     -- NULL = me (paired with initiator_is_me)
    initiator_is_me INTEGER,
    ender_handle_id INTEGER,         -- NULL = me
    ender_is_me INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sessions_chat ON sessions(chat_id, first_date);

CREATE TABLE IF NOT EXISTS person_period_lexicon (
    chat_id INTEGER,
    handle_id INTEGER,               -- NULL = me
    month TEXT,                      -- 'YYYY-MM' (date[:7])
    n_messages INTEGER,
    n_words INTEGER,
    avg_chars REAL,
    fw_pronoun_rate REAL,
    fw_ipron_rate REAL,
    fw_article_rate REAL,
    fw_conj_rate REAL,
    fw_prep_rate REAL,
    fw_aux_rate REAL,
    fw_adverb_rate REAL,
    fw_negation_rate REAL,
    fw_quant_rate REAL,
    we_count INTEGER,
    question_count INTEGER,
    laugh_count INTEGER,
    emoji_json TEXT,
    -- NULL handle_id in the PK is fine: the table is full-rebuild only
    -- (plain INSERTs, never upserts), so PK uniqueness is never relied on.
    PRIMARY KEY (chat_id, handle_id, month)
);

CREATE TABLE IF NOT EXISTS dyad_terms (
    chat_id INTEGER,
    term TEXT,
    lift REAL,
    n_uses INTEGER,
    first_message_id INTEGER,
    coined_by_handle_id INTEGER,     -- NULL = me
    coined_is_me INTEGER,
    llm_verdict TEXT,                -- NULL until annotate.py --terms curates it
    curate_failures INTEGER NOT NULL DEFAULT 0,
                                     -- failed curation attempts. At
                                     -- ANNOTATE_TERMS_MAX_FAILURES the term is
                                     -- parked: llm_verdict stays NULL but runs
                                     -- stop retrying it (--retry-failed
                                     -- unparks). Preserved across rebuilds
                                     -- alongside llm_verdict.
    PRIMARY KEY (chat_id, term)
);

CREATE TABLE IF NOT EXISTS chunk_annotations (
    chunk_id INTEGER PRIMARY KEY,
    valence REAL,                    -- -1..1
    emotions_json TEXT,              -- JSON list of GoEmotions labels
    depth INTEGER,                   -- 1 (logistics) .. 5 (heart-to-heart)
    events_json TEXT,                -- JSON list of good-news/gratitude events
    source TEXT DEFAULT 'llm'        -- 'llm' or 'distilled' (annotate.py provenance;
                                     -- distilled rows have empty events and can be
                                     -- re-annotated by an LLM tier)
);

CREATE TABLE IF NOT EXISTS topics (
    cluster_id INTEGER PRIMARY KEY,
    label TEXT,                      -- NULL until the label pass names it
    n_chunks INTEGER
);
CREATE TABLE IF NOT EXISTS chunk_topics (
    chunk_id INTEGER PRIMARY KEY,
    cluster_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunk_topics_cluster ON chunk_topics(cluster_id);

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

    # Pre-distillation chunk_annotations: add the provenance column. The
    # DEFAULT stamps existing rows 'llm', which is correct -- every row
    # predating the column was LLM-produced.
    ann_cols = {row[1] for row in con.execute("PRAGMA table_info(chunk_annotations)")}
    if ann_cols and "source" not in ann_cols:
        con.execute("ALTER TABLE chunk_annotations ADD COLUMN source TEXT DEFAULT 'llm'")

    # Pre-parking dyad_terms: add the failure counter. The DEFAULT stamps
    # existing rows 0, which is correct -- an uncurated row predating the
    # column has no recorded failures, so it stays eligible.
    term_cols = {row[1] for row in con.execute("PRAGMA table_info(dyad_terms)")}
    if term_cols and "curate_failures" not in term_cols:
        con.execute("ALTER TABLE dyad_terms "
                    "ADD COLUMN curate_failures INTEGER NOT NULL DEFAULT 0")

    # The update trigger used to fire on ANY messages update; drop it so
    # FTS_SCHEMA recreates it scoped to UPDATE OF text (see comment there).
    con.execute("DROP TRIGGER IF EXISTS messages_au")

    con.executescript(SCHEMA)
    con.executescript(FTS_SCHEMA)

    # Reactions used to be stored with chat.db's raw associated_message_guid,
    # which carries a message-part prefix ('p:0/<guid>' -- tapback on part N
    # -- or 'bp:<guid>' for balloon parts) that messages.guid never has, so
    # reaction -> message joins missed ~99% of rows. Normalize in place with
    # the exact function ingest applies at insert time (the SQL LIKEs are
    # only a loose prefilter; the regex decides). Idempotent: a normalized
    # guid matches no prefix.
    prefixed = con.execute(
        "SELECT id, target_message_guid FROM reactions "
        "WHERE target_message_guid LIKE 'p:%/%' OR target_message_guid LIKE 'bp:%'"
    ).fetchall()
    con.executemany(
        "UPDATE reactions SET target_message_guid = ? WHERE id = ?",
        [(stripped, rowid) for rowid, guid in prefixed
         if (stripped := strip_reaction_guid_prefix(guid)) != guid],
    )
    con.commit()
