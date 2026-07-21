"""Pronoun-split + absolutist lexicon extension (RES-43 / F1).

Covers the three columns build_person_period_lexicon now emits --
i_count / you_count / absolutist_count -- and the idempotent schema migration
that adds them. Hermetic: a hand-scripted temp DB with known token counts
(contractions like "i'm" and texting variants like "ur"/"u"/"im" included),
no Ollama, no real chat.db.

These are RAW token counts (every matching token, repeats included), not the
per-message flags we/question/laugh carry -- the assertions below pin that.
"""

import contextlib
import io
import os
import sqlite3
import tempfile
import unittest

import lexicon_build
import schema
from function_words import ABSOLUTIST, PRONOUN_I, PRONOUN_YOU


def _msg(mid, chat, handle, from_me, text, date):
    return (mid, f"guid-{mid}", chat, handle, from_me, text, date, 0)


class PronounAbsolutistLexiconTest(unittest.TestCase):
    """Chat 100: a 1:1 (handle 70). One month, so exactly one me-row and one
    them-row -- each column has a single hand-computed right answer."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            schema.create_schema(con)
            con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (70, '+15550009999', NULL, 'iMessage')"
            )
            con.execute(
                "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (100, 'chat100', NULL, 0, 'iMessage')"
            )
            con.execute(
                "INSERT INTO chat_participants(chat_id, handle_id) VALUES (100, 70)"
            )
            # Me (2026-01):
            #   m1 "i think i'm right"  -> I: i, i'm (2)
            #   m2 "you always say that" -> YOU: you (1); ABS: always (1)
            # Them (2026-01):
            #   m3 "ur totally wrong u never listen" -> YOU: ur, u (2);
            #                                           ABS: totally, never (2)
            #   m4 "im done myself and me"           -> I: im, myself, me (3)
            rows = [
                _msg(1, 100, 0, 1, "i think i'm right", "2026-01-01T10:00:00"),
                _msg(2, 100, 0, 1, "you always say that", "2026-01-01T10:01:00"),
                _msg(3, 100, 70, 0, "ur totally wrong u never listen",
                     "2026-01-02T09:00:00"),
                _msg(4, 100, 70, 0, "im done myself and me", "2026-01-02T09:01:00"),
            ]
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        finally:
            con.close()

    def _build(self):
        con = sqlite3.connect(self.db_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return lexicon_build.build_person_period_lexicon(con)
        finally:
            con.close()

    def _row(self, sql, params=()):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            (row,) = con.execute(sql, params).fetchall()
            return row
        finally:
            con.close()

    def test_me_side_counts(self):
        self._build()
        row = self._row(
            "SELECT * FROM person_period_lexicon "
            "WHERE chat_id = 100 AND handle_id IS NULL AND month = '2026-01'"
        )
        self.assertEqual(row["i_count"], 2)          # i, i'm
        self.assertEqual(row["you_count"], 1)        # you
        self.assertEqual(row["absolutist_count"], 1)  # always

    def test_them_side_counts_texting_variants(self):
        self._build()
        row = self._row(
            "SELECT * FROM person_period_lexicon "
            "WHERE chat_id = 100 AND handle_id = 70 AND month = '2026-01'"
        )
        self.assertEqual(row["i_count"], 3)          # im, myself, me
        self.assertEqual(row["you_count"], 2)        # ur, u
        self.assertEqual(row["absolutist_count"], 2)  # totally, never

    def test_full_rebuild_populates_all_three_columns(self):
        # Acceptance: re-running the lexicon stage fills every new column
        # (no NULLs left behind), and the rebuild is idempotent.
        self._build()
        self._build()
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT i_count, you_count, absolutist_count "
                "FROM person_period_lexicon"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIsNotNone(r["i_count"])
            self.assertIsNotNone(r["you_count"])
            self.assertIsNotNone(r["absolutist_count"])

    def test_frozensets_have_expected_members(self):
        # Canonical Al-Mosaiwi & Johnstone (2018) 19 absolutist words, at least.
        canonical = {
            "absolutely", "all", "always", "complete", "completely", "constant",
            "constantly", "definitely", "entire", "ever", "every", "everyone",
            "everything", "full", "must", "never", "nothing", "totally", "whole",
        }
        self.assertTrue(canonical <= ABSOLUTIST)
        # First-person SINGULAR only -- no plural "we", clear texting variants.
        self.assertLessEqual({"i", "me", "my", "i'm", "im", "myself"}, PRONOUN_I)
        self.assertNotIn("we", PRONOUN_I)
        self.assertLessEqual({"you", "your", "u", "ur", "youre"}, PRONOUN_YOU)


class LexiconMigrationTest(unittest.TestCase):
    """The idempotent ALTER that adds the three columns to a pre-F1 DB."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

    def _cols(self, con):
        return {row[1] for row in con.execute(
            "PRAGMA table_info(person_period_lexicon)"
        )}

    def test_fresh_schema_has_columns_and_is_reentrant(self):
        con = sqlite3.connect(self.db_path)
        try:
            schema.create_schema(con)
            cols = self._cols(con)
            self.assertLessEqual({"i_count", "you_count", "absolutist_count"}, cols)
            # Re-running create_schema must be a no-op (columns already present).
            schema.create_schema(con)
            self.assertEqual(self._cols(con), cols)
        finally:
            con.close()

    def test_migrates_pre_f1_table(self):
        con = sqlite3.connect(self.db_path)
        try:
            # An old-layout person_period_lexicon: pre-F1 columns only.
            con.execute(
                "CREATE TABLE person_period_lexicon ("
                "chat_id INTEGER, handle_id INTEGER, month TEXT, "
                "n_messages INTEGER, n_words INTEGER, avg_chars REAL, "
                "we_count INTEGER, question_count INTEGER, laugh_count INTEGER, "
                "emoji_json TEXT, PRIMARY KEY (chat_id, handle_id, month))"
            )
            con.execute(
                "INSERT INTO person_period_lexicon(chat_id, handle_id, month, "
                "n_messages, n_words, we_count) VALUES (1, NULL, '2026-01', 3, 9, 1)"
            )
            con.commit()
            self.assertNotIn("i_count", self._cols(con))

            schema.create_schema(con)

            cols = self._cols(con)
            self.assertLessEqual({"i_count", "you_count", "absolutist_count"}, cols)
            # Pre-existing rows survive and read NULL for the new counts until
            # the next full rebuild.
            row = con.execute(
                "SELECT we_count, i_count, you_count, absolutist_count "
                "FROM person_period_lexicon WHERE chat_id = 1"
            ).fetchone()
            self.assertEqual(row[0], 1)   # we_count preserved
            self.assertIsNone(row[1])     # i_count NULL
            self.assertIsNone(row[2])
            self.assertIsNone(row[3])
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
