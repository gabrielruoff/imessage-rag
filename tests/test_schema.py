"""Tests for schema.py -- the DDL blob plus the in-place migrations
create_schema runs against an OLD imessage_analysis.db.

Hermetic: every test runs against sqlite3.connect(":memory:"). No Ollama, no
real chat.db, no sqlite-vec (chunk_vectors is built by ingest.py, not here),
no temp files. schema.create_schema COMMITs and also WRITES data (the
reaction-guid normalization rescan), so each test owns its own connection.

The ~71 existing create_schema call sites across tests/ already smoke the
fresh-DB path; what is untested is everything that can only fail on a
pre-existing database -- migration ORDER, the documented column defaults, the
declared-vs-migrated column order, and reset_db parity.

KNOWN GAP (documented here, deliberately not fixed): only conflicts.labeled_at
has a PRAGMA-table_info back-fill. dyad_drift, person_voiceprint,
loose_threads, nudge_exclusions and conflict_signals are pure
`CREATE ... IF NOT EXISTS` with no ALTER path at all, so adding a column to any
of them ships DDL that a fresh DB satisfies and an existing DB does not -- the
next run dies with "no such column" at the first read of the new column. The
tables that DO have a back-fill (handles, messages, chunk_annotations,
dyad_terms, person_period_lexicon, conflicts) are the precedent; the five above
are the ones a future column add would trip over.
"""

import sqlite3
import unittest

import reset_db
import schema

# Pre-chunking messages layout: everything the current SCHEMA declares EXCEPT
# chunk_id. It must carry chat_id/date/handle_id, because CREATE TABLE IF NOT
# EXISTS will not widen an existing table and SCHEMA indexes those columns.
_OLD_MESSAGES_DDL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
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
)
"""

# Pre-distillation chunk_annotations: no `source`, no `knn_confidence`.
_OLD_ANNOTATIONS_DDL = """
CREATE TABLE chunk_annotations (
    chunk_id INTEGER PRIMARY KEY,
    valence REAL,
    emotions_json TEXT,
    depth INTEGER,
    events_json TEXT
)
"""

# Pre-parking dyad_terms: no `curate_failures`.
_OLD_DYAD_TERMS_DDL = """
CREATE TABLE dyad_terms (
    chat_id INTEGER,
    term TEXT,
    lift REAL,
    n_uses INTEGER,
    first_message_id INTEGER,
    coined_by_handle_id INTEGER,
    coined_is_me INTEGER,
    llm_verdict TEXT,
    PRIMARY KEY (chat_id, term)
)
"""

# Pre-budget conflicts: no `labeled_at` resume marker.
_OLD_CONFLICTS_DDL = """
CREATE TABLE conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    first_date TEXT,
    last_date TEXT,
    start_session_id INTEGER,
    end_session_id INTEGER,
    topic TEXT,
    peak_signal_json TEXT,
    corroboration TEXT,
    off_text_flag INTEGER DEFAULT 0,
    source TEXT DEFAULT 'llm'
)
"""

# Pre-F1 person_period_lexicon: no i_count/you_count/absolutist_count.
_OLD_LEXICON_DDL = """
CREATE TABLE person_period_lexicon (
    chat_id INTEGER,
    handle_id INTEGER,
    month TEXT,
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
    PRIMARY KEY (chat_id, handle_id, month)
)
"""

# Every table create_schema leaves behind on an empty database. Pinned in full
# so a feature's DDL block silently lost in a cross-worktree merge fails here:
# schema.py:366-369 states the file is organized for exactly those additive
# merges, which is precisely the edit that loses a block without a conflict.
_FRESH_TABLES = {
    "attachments",
    "chat_participants",
    "chats",
    "chunk_annotations",
    "chunk_topics",
    "chunks",
    "conflict_signals",
    "conflicts",
    "contacts",
    "dyad_drift",
    "dyad_terms",
    "handles",
    "ingest_state",
    "loose_threads",
    "messages",
    "messages_fts",
    "messages_fts_config",
    "messages_fts_data",
    "messages_fts_docsize",
    "messages_fts_idx",
    "nudge_exclusions",
    "person_period_lexicon",
    "person_voiceprint",
    "reactions",
    "sessions",
    "sqlite_sequence",
    "topics",
}

# messages_fts and its four FTS5 shadow tables are kept in sync by the
# messages_ai/ad/au triggers and re-synced by reset_db's 'rebuild' command, so
# they are never listed as tables to DELETE from.
_FTS_OBJECTS = {
    "messages_fts",
    "messages_fts_config",
    "messages_fts_data",
    "messages_fts_docsize",
    "messages_fts_idx",
}


def _tables(con):
    """Table names only -- indexes and triggers are asserted separately."""
    return {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _columns(con, table):
    """Declared column names, IN DECLARATION ORDER (PRAGMA table_info's cid
    order), because positional INSERTs depend on it."""
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def _column_info(con, table, column):
    """The full PRAGMA table_info row for one column: (cid, name, type,
    notnull, dflt_value, pk)."""
    for row in con.execute(f"PRAGMA table_info({table})"):
        if row[1] == column:
            return row
    return None


def _object_sql(con, name):
    """sqlite_master.sql for a named object (trigger/index/table)."""
    row = con.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def _traced_create_schema(con):
    """Run create_schema while recording every SQL statement sqlite executes,
    including the individual statements inside an executescript. Lets a test
    assert on migration ORDER rather than only on the end state."""
    seen = []
    con.set_trace_callback(seen.append)
    try:
        schema.create_schema(con)
    finally:
        con.set_trace_callback(None)
    return seen


def _memory_db(testcase):
    con = sqlite3.connect(":memory:")
    testcase.addCleanup(con.close)
    return con


class FreshSchemaTest(unittest.TestCase):
    def test_create_schema_twice_is_a_no_op(self):
        # ingest.py calls create_schema at the top of EVERY pipeline run, so the
        # second call is the normal case, not the edge case. Every migration is
        # guarded ("add only if absent"), so a re-run must not duplicate a
        # column or change any declaration order.
        con = _memory_db(self)
        schema.create_schema(con)
        first_tables = _tables(con)
        first_columns = {name: _columns(con, name) for name in first_tables}

        schema.create_schema(con)

        self.assertEqual(_tables(con), first_tables)
        for name in sorted(first_tables):
            self.assertEqual(_columns(con, name), first_columns[name], name)
            # A duplicated ALTER would show up as a repeated name here even if
            # the list length happened to match some other table's.
            self.assertEqual(len(set(first_columns[name])), len(first_columns[name]), name)

    def test_a_fresh_database_has_the_expected_table_set(self):
        # The whole point of pinning the SET (not just spot-checking one table)
        # is that a lost feature DDL block is invisible otherwise: the pipeline
        # that reads it fails far away, at query time.
        con = _memory_db(self)
        schema.create_schema(con)
        self.assertEqual(_tables(con), _FRESH_TABLES)

    def test_no_alter_runs_against_a_fresh_database(self):
        # Every migration is behind an `if <cols>:` emptiness guard, so on an
        # empty DB the CREATE statements are the ONLY writers of structure. If
        # an ALTER ever fires here, some guard was dropped and a fresh DB is
        # being "migrated" against columns SCHEMA already declared.
        #
        # The one exception by construction: the conflicts.labeled_at check at
        # schema.py:552-553 has NO emptiness guard -- it reads
        # PRAGMA table_info(conflicts) unconditionally. It is safe only because
        # _create_conflict_tables' CREATE TABLE immediately precedes it, so on a
        # fresh DB the column is already there and the `if` is False. Moving
        # that check above the CREATE would break the invariant this asserts.
        con = _memory_db(self)
        statements = _traced_create_schema(con)
        alters = [s for s in statements if s.lstrip().upper().startswith("ALTER")]
        self.assertEqual(alters, [])

    def test_person_period_lexicon_column_order_matches_the_migration_order(self):
        # schema.py:143-145 declares i_count/you_count/absolutist_count LAST
        # explicitly "to match the ALTER migration's append order". Nothing
        # checked it. A mismatch is silent: SELECT * and positional INSERTs
        # would map counts onto the wrong columns on a migrated DB while a
        # fresh DB stayed correct -- exactly the class of bug that only shows up
        # on a user's real database.
        fresh = _memory_db(self)
        schema.create_schema(fresh)
        fresh_order = _columns(fresh, "person_period_lexicon")

        migrated = _memory_db(self)
        migrated.execute(_OLD_LEXICON_DDL)
        migrated.commit()
        schema.create_schema(migrated)

        self.assertEqual(_columns(migrated, "person_period_lexicon"), fresh_order)
        # And the contract the ordering exists for: the three counters are the
        # tail, in this order.
        self.assertEqual(fresh_order[-3:], ["i_count", "you_count", "absolutist_count"])


class MigrationOrderTest(unittest.TestCase):
    """These can only fail against an OLD database -- on a fresh one SCHEMA
    creates every table complete and no migration runs at all. That is exactly
    why a suite built on fresh-start fixtures cannot catch a reordering."""

    def test_handles_migrate_before_the_index_that_needs_the_column(self):
        # schema.py:289-293: the handles migration MUST precede the SCHEMA
        # executescript, because SCHEMA creates
        # idx_handles_contact_id ON handles(contact_id). Reorder them and
        # create_schema raises "no such column: contact_id" on every old DB.
        # (This is the ADD COLUMN branch -- a handles table with no contact_id
        # at all -- not the TEXT-rename branch.)
        con = _memory_db(self)
        con.execute(
            "CREATE TABLE handles (id INTEGER PRIMARY KEY, handle_identifier TEXT, service TEXT)"
        )
        con.commit()

        statements = _traced_create_schema(con)

        self.assertIn("contact_id", _columns(con, "handles"))
        index_sql = _object_sql(con, "idx_handles_contact_id")
        self.assertIsNotNone(index_sql)
        self.assertIn("contact_id", index_sql)
        # Assert the ORDER directly, not just the end state: a reordering that
        # happened to still work would otherwise pass.
        alter_at = next(
            i for i, s in enumerate(statements)
            if "ALTER TABLE handles ADD COLUMN contact_id" in s
        )
        index_at = next(i for i, s in enumerate(statements) if "idx_handles_contact_id" in s)
        self.assertLess(alter_at, index_at)

    def test_messages_chunk_id_is_added_before_its_index(self):
        # schema.py:304-305, same shape: SCHEMA creates
        # idx_messages_chunk ON messages(chunk_id), so the pre-chunking
        # messages table must gain chunk_id first.
        con = _memory_db(self)
        con.execute(_OLD_MESSAGES_DDL)
        con.commit()

        statements = _traced_create_schema(con)

        self.assertIn("chunk_id", _columns(con, "messages"))
        index_sql = _object_sql(con, "idx_messages_chunk")
        self.assertIsNotNone(index_sql)
        self.assertIn("chunk_id", index_sql)
        alter_at = next(
            i for i, s in enumerate(statements)
            if "ALTER TABLE messages ADD COLUMN chunk_id" in s
        )
        index_at = next(i for i, s in enumerate(statements) if "idx_messages_chunk" in s)
        self.assertLess(alter_at, index_at)


class ColumnSemanticsTest(unittest.TestCase):
    """Each added column's nullability/DEFAULT encodes a documented claim about
    what a pre-column row MEANS. These pin the claim, not just the column."""

    def _annotations_db(self):
        con = _memory_db(self)
        con.execute(_OLD_ANNOTATIONS_DDL)
        con.execute("INSERT INTO chunk_annotations VALUES (1, 0.5, '[]', 3, '[]')")
        con.commit()
        return con

    def test_source_defaults_to_llm_for_rows_that_predate_the_column(self):
        # schema.py:310-315: the DEFAULT is not a convenience -- it is a factual
        # claim. Every annotation written before the distilled tier existed was
        # LLM-produced, so stamping them 'llm' is correct, and stamping them
        # 'distilled' (or NULL) would make valence-dependent reads either
        # discard real LLM labels or trust kNN ones.
        con = self._annotations_db()
        schema.create_schema(con)

        info = _column_info(con, "chunk_annotations", "source")
        self.assertIsNotNone(info)
        self.assertEqual(info[4], "'llm'")
        self.assertEqual(
            con.execute("SELECT source FROM chunk_annotations WHERE chunk_id = 1").fetchone()[0],
            "llm",
        )

    def test_knn_confidence_is_added_nullable_with_no_default(self):
        # schema.py:317-322: deliberately UNLIKE `source`. A distilled row
        # written before the column genuinely has no recorded confidence; NULL
        # says so, whereas "any numeric default would invent a ranking signal"
        # that the hybrid tier's lowest-confidence-first upgrade order reads.
        con = self._annotations_db()
        schema.create_schema(con)

        info = _column_info(con, "chunk_annotations", "knn_confidence")
        self.assertIsNotNone(info)
        self.assertEqual(info[2], "REAL")
        self.assertEqual(info[3], 0)  # notnull == 0
        self.assertIsNone(info[4])  # no DEFAULT
        self.assertIsNone(
            con.execute(
                "SELECT knn_confidence FROM chunk_annotations WHERE chunk_id = 1"
            ).fetchone()[0]
        )

    def test_curate_failures_is_added_not_null_defaulting_to_zero(self):
        # schema.py:326-332: an uncurated term predating the counter has no
        # recorded failures, so 0 keeps it ELIGIBLE. NULL would compare
        # false against the ANNOTATE_TERMS_MAX_FAILURES threshold in SQL and
        # silently park -- or unpark -- every legacy term.
        con = _memory_db(self)
        con.execute(_OLD_DYAD_TERMS_DDL)
        con.execute("INSERT INTO dyad_terms VALUES (1, 'bruh', 4.0, 9, 100, NULL, 1, NULL)")
        con.commit()

        schema.create_schema(con)

        info = _column_info(con, "dyad_terms", "curate_failures")
        self.assertIsNotNone(info)
        self.assertEqual(info[3], 1)  # NOT NULL
        self.assertEqual(info[4], "0")
        self.assertEqual(
            con.execute("SELECT curate_failures FROM dyad_terms WHERE term = 'bruh'").fetchone()[0],
            0,
        )

    def test_labeled_at_is_added_nullable_so_null_means_label_it(self):
        # schema.py:545-554: a window written before the column WAS labeled,
        # under the old unbounded path, but nothing recorded when. Stamping a
        # made-up timestamp would tell the budgeted labeling loop to skip work
        # it never verified; NULL means "label it", the safe direction.
        con = _memory_db(self)
        con.execute(_OLD_CONFLICTS_DDL)
        con.execute(
            "INSERT INTO conflicts (id, chat_id, first_date, last_date) "
            "VALUES (1, 7, '2024-01-01 00:00:00', '2024-01-02 00:00:00')"
        )
        con.commit()

        schema.create_schema(con)

        info = _column_info(con, "conflicts", "labeled_at")
        self.assertIsNotNone(info)
        self.assertEqual(info[3], 0)  # nullable
        self.assertIsNone(info[4])  # no DEFAULT
        self.assertIsNone(
            con.execute("SELECT labeled_at FROM conflicts WHERE id = 1").fetchone()[0]
        )


class FtsTriggerTest(unittest.TestCase):
    def _snapshot_fts(self, con):
        """The FTS5 shadow-table state. Sensitive enough to distinguish "the
        update trigger did not fire" from "it fired and rewrote the same text":
        an external-content delete+insert writes new segment blocks even when
        the text is unchanged."""
        return (
            con.execute("SELECT id, block FROM messages_fts_data ORDER BY id").fetchall(),
            con.execute("SELECT * FROM messages_fts_docsize").fetchall(),
        )

    def test_the_update_trigger_is_scoped_to_the_text_column(self):
        # schema.py:227-229 + :345-347. The trigger USED to fire on any messages
        # UPDATE, so create_schema unconditionally DROPs messages_au and lets
        # FTS_SCHEMA recreate it scoped -- CREATE TRIGGER IF NOT EXISTS alone
        # would leave an old unscoped trigger in place forever.
        con = _memory_db(self)
        schema.create_schema(con)
        # Reinstate the legacy unscoped trigger, exactly as an old DB carries it.
        con.execute("DROP TRIGGER messages_au")
        con.executescript(
            "CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN\n"
            "    INSERT INTO messages_fts(messages_fts, rowid, text) "
            "VALUES ('delete', old.id, old.text);\n"
            "    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);\n"
            "END;"
        )
        self.assertNotIn("UPDATE OF text", _object_sql(con, "messages_au"))

        schema.create_schema(con)

        self.assertIn("UPDATE OF text", _object_sql(con, "messages_au"))

    def test_stamping_chunk_id_does_not_rewrite_the_fts_entry(self):
        # The behavioral half of the scoping contract: chunk building stamps
        # chunk_id on EVERY message, and an unscoped trigger would re-index the
        # whole corpus for those writes. The probe is validated in-test -- a
        # real text UPDATE must move the same shadow tables -- so a passing
        # assertion cannot mean "the probe sees nothing".
        con = _memory_db(self)
        schema.create_schema(con)
        con.execute(
            "INSERT INTO messages (id, guid, chat_id, text, date) "
            "VALUES (1, 'g1', 7, 'dinner at eight', '2024-01-01 00:00:00')"
        )
        con.commit()
        before = self._snapshot_fts(con)

        con.execute("UPDATE messages SET chunk_id = 42 WHERE id = 1")
        con.commit()

        self.assertEqual(self._snapshot_fts(con), before)
        self.assertEqual(
            con.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'dinner'"
            ).fetchall(),
            [(1,)],
        )

        con.execute("UPDATE messages SET text = 'breakfast at nine' WHERE id = 1")
        con.commit()
        self.assertNotEqual(self._snapshot_fts(con), before)


class ResetParityTest(unittest.TestCase):
    def test_every_derived_table_is_listed_in_reset_db(self):
        # reset_db.py:16-69 enumerates the tables to clear BY NAME, so a table
        # added to schema.py is silently omitted: `make reset-db` reports
        # success and leaves the new table's rows behind, which then read as
        # live data against a freshly re-ingested corpus. Nothing checked this
        # parity, so pin it -- along with each deliberate exclusion, so the
        # exclusions have to be argued for rather than accumulated.
        con = _memory_db(self)
        schema.create_schema(con)

        # nudge_exclusions is durable USER INTENT (mutes), preserved across a
        # reset by design (reset_db.py:49-56); the committed
        # eval/seed_gates/conflict_labels.json survives for the same reason,
        # but is a repo file, not a table. sqlite_sequence is sqlite's own
        # AUTOINCREMENT bookkeeping and has no reset_db entry.
        expected = _tables(con) - _FTS_OBJECTS - {"nudge_exclusions", "sqlite_sequence"}

        self.assertEqual(set(reset_db.TABLES), expected)
        self.assertNotIn("nudge_exclusions", reset_db.TABLES)
        # chunk_vectors/message_vectors are DROPPED, not emptied (their vector
        # dimension is bound to EMBED_MODEL), and neither is created by
        # create_schema -- so they must not appear in the parity set above.
        self.assertEqual(reset_db.DROP_TABLES, ["chunk_vectors", "message_vectors"])
        self.assertEqual(set(reset_db.DROP_TABLES) & _tables(con), set())


if __name__ == "__main__":
    unittest.main()
