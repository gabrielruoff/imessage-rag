"""Lightweight smoke tests for the contacts refactor.

No Ollama and no real chat.db required -- everything runs against temporary /
in-memory SQLite databases. Covers the delicate bits of PR #1:
  - the in-place handles-table migration in ingest.create_schema
  - FK-safe, idempotent re-ingest of contacts + handles
  - the two query_tools payload/behavior fixes (source_uniqueid, SQL-side
    list_chats filter+limit)

Run from the repo root:  python3 -m unittest discover tests
"""

import os
import sqlite3
import tempfile
import unittest

import ingest
import query_tools


class MigrationTest(unittest.TestCase):
    def test_old_handles_layout_migrates_in_place(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        # Old schema: contact_id held the raw phone/email; name/org denormalized.
        con.execute(
            """CREATE TABLE handles (
                id INTEGER PRIMARY KEY, contact_id TEXT, service TEXT,
                first_name TEXT, last_name TEXT, nickname TEXT, organization TEXT)"""
        )
        con.execute(
            "INSERT INTO handles VALUES (5, '+12015551234', 'iMessage', "
            "'Jane', 'Doe', NULL, 'Acme')"
        )
        con.commit()

        ingest.create_schema(con)

        cols = {row[1]: row[2] for row in con.execute("PRAGMA table_info(handles)")}
        # Raw handle string preserved under its new column name.
        self.assertIn("handle_identifier", cols)
        # contact_id is now an integer FK, no longer TEXT.
        self.assertEqual(cols.get("contact_id"), "INTEGER")
        # Denormalized name/org columns dropped.
        for gone in ("first_name", "last_name", "nickname", "organization"):
            self.assertNotIn(gone, cols)

        row = con.execute(
            "SELECT handle_identifier, contact_id FROM handles WHERE id = 5"
        ).fetchone()
        self.assertEqual(row[0], "+12015551234")
        self.assertIsNone(row[1])  # reset to NULL, to be repopulated by ingest_handles


class ReingestTest(unittest.TestCase):
    def _fake_src(self):
        src = sqlite3.connect(":memory:")
        self.addCleanup(src.close)
        src.execute("CREATE TABLE handle (id TEXT, service TEXT)")  # ROWID implicit
        src.executemany(
            "INSERT INTO handle(id, service) VALUES (?, ?)",
            [("+12015551234", "iMessage"), ("friend@example.com", "iMessage")],
        )
        src.commit()
        return src

    def test_reingest_is_fk_safe_and_idempotent(self):
        tgt = sqlite3.connect(":memory:")
        self.addCleanup(tgt.close)
        tgt.execute("PRAGMA foreign_keys = ON")
        ingest.create_schema(tgt)

        contacts = [(1, "uid-1", "Jane", "Doe", None, "Acme")]
        # Only the phone maps to a contact; the email handle stays contact_id NULL.
        handle_to_contact = {ingest.normalize_handle_id("+12015551234"): 1}

        def run_once():
            src = self._fake_src()
            ingest.ingest_contacts(tgt, contacts)
            ingest.ingest_handles(tgt=tgt, src=src, contacts=contacts,
                                  handle_to_contact=handle_to_contact)
            tgt.commit()

        run_once()
        counts_1 = (
            tgt.execute("SELECT COUNT(*) FROM contacts").fetchone()[0],
            tgt.execute("SELECT COUNT(*) FROM handles").fetchone()[0],
        )
        # Second run must not raise (INSERT OR REPLACE on the integer-PK parent
        # is an in-place replace, so the handles.contact_id FK never dangles).
        run_once()
        counts_2 = (
            tgt.execute("SELECT COUNT(*) FROM contacts").fetchone()[0],
            tgt.execute("SELECT COUNT(*) FROM handles").fetchone()[0],
        )
        self.assertEqual(counts_1, (1, 2))
        self.assertEqual(counts_1, counts_2)  # stable, purely additive

        linked = tgt.execute(
            "SELECT contact_id FROM handles WHERE handle_identifier = '+12015551234'"
        ).fetchone()[0]
        unlinked = tgt.execute(
            "SELECT contact_id FROM handles WHERE handle_identifier = 'friend@example.com'"
        ).fetchone()[0]
        self.assertEqual(linked, 1)
        self.assertIsNone(unlinked)


class QueryToolsTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            ingest.create_schema(con)
            con.execute(
                "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
                "nickname, organization) VALUES (2, 'uid-2', 'Alice', 'Smith', NULL, 'Globex')"
            )
            # Alice owns two handles (phone + email) and appears in both a
            # direct chat (100, via her phone) and a group chat (101, via her
            # email) -- this is the "mom has multiple handles/chats" shape
            # that contact_id-scoped filtering needs to collapse into one call.
            con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (50, '+19995551234', 2, 'iMessage')"
            )
            con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (51, 'alice@example.com', 2, 'iMessage')"
            )
            con.executemany(
                "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (100, "chat100", None, 0, "iMessage"),
                    (101, "chat101", None, 1, "iMessage"),
                    (102, "special-id", None, 0, "iMessage"),
                ],
            )
            con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (100, 50)")
            con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (101, 51)")
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                [
                    (1000, "guid-1000", 100, 50, 0, "hi from phone", "2026-01-01T00:00:00"),
                    (1001, "guid-1001", 100, 50, 1, "hi back", "2026-01-02T00:00:00"),
                    (1002, "guid-1002", 101, 51, 0, "hi from email in group", "2026-01-03T00:00:00"),
                ],
            )
            con.execute(
                "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
                "is_from_me, associated_message_type, date) VALUES "
                "(2000, 'guid-1000', 100, 50, 0, 2000, '2026-01-01T00:05:00')"
            )
            con.execute(
                "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
                "is_from_me, associated_message_type, date) VALUES "
                "(2001, 'guid-1002', 101, 51, 0, 2000, '2026-01-03T00:05:00')"
            )
            con.commit()
        finally:
            con.close()

        # Point the query layer at this temp DB (_connect reads the module global).
        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)

    def test_search_contacts_by_name_includes_source_uniqueid(self):
        results = query_tools.search_contacts_by_name("Alice")
        self.assertEqual(len(results), 1)
        self.assertIn("source_uniqueid", results[0])
        self.assertEqual(results[0]["source_uniqueid"], "uid-2")

    def test_list_chats_respects_limit(self):
        self.assertLessEqual(len(query_tools.list_chats(limit=2)), 2)

    def test_list_chats_filters_by_participant_name(self):
        # Alice now owns two handles (see setUp) and appears in both chat 100
        # (direct) and chat 101 (group) via her phone and email respectively.
        chats = query_tools.list_chats(query="Alice")
        self.assertEqual({c["id"] for c in chats}, {100, 101})

    def test_list_chats_filters_by_chat_identifier(self):
        chats = query_tools.list_chats(query="special")
        self.assertEqual([c["id"] for c in chats], [102])

    def test_search_messages_sql_handle_id_list_unions_handles(self):
        rows = query_tools.search_messages_sql(handle_id=[50, 51])
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_contact_id_resolves_all_handles(self):
        rows = query_tools.search_messages_sql(contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_chat_id_list(self):
        rows = query_tools.search_messages_sql(chat_id=[100, 101])
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_single_chat_id_still_works(self):
        rows = query_tools.search_messages_sql(chat_id=100)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001})

    def test_list_chats_filters_by_contact_id(self):
        chats = query_tools.list_chats(contact_id=2)
        self.assertEqual({c["id"] for c in chats}, {100, 101})

    def test_list_chats_filters_by_handle_id(self):
        chats = query_tools.list_chats(handle_id=50)
        self.assertEqual({c["id"] for c in chats}, {100})

    def test_get_chat_summary_stats_with_chat_id_list(self):
        stats = query_tools.get_chat_summary_stats(chat_id=[100, 101])
        self.assertEqual(stats["message_count"], 3)

    def test_get_chat_summary_stats_with_contact_id(self):
        stats = query_tools.get_chat_summary_stats(contact_id=2)
        self.assertEqual(stats["message_count"], 3)
        self.assertEqual(stats["from_me"], 1)
        self.assertEqual(stats["from_them"], 2)

    def test_get_chat_summary_stats_requires_a_scope(self):
        with self.assertRaises(ValueError):
            query_tools.get_chat_summary_stats()

    def test_get_message_reactions_single_guid_still_works(self):
        rows = query_tools.get_message_reactions("guid-1000")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_message_guid"], "guid-1000")

    def test_get_message_reactions_batch_guids(self):
        rows = query_tools.get_message_reactions(["guid-1000", "guid-1002"])
        self.assertEqual(
            {(r["target_message_guid"]) for r in rows}, {"guid-1000", "guid-1002"}
        )

    def test_search_messages_fts_scopes_by_contact_id(self):
        rows = query_tools.search_messages_fts("hi", contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

        rows = query_tools.search_messages_fts("email", contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1002})


if __name__ == "__main__":
    unittest.main()
