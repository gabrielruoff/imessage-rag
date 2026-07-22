"""query_tools/_shared.py: the shared _restrict_to_direct_chats helper the
dyad-scoped analytics tools use to confine a person scope to that person's 1:1
chats. Fixture has mixed 1:1 and group chats so the group rows are provably
excluded. Hermetic: a temp sqlite file, no Ollama, no real chat.db."""

import os
import sqlite3
import tempfile
import unittest

from db import load_vec_extension
from query_tools._shared import _restrict_to_direct_chats
from schema import create_schema


class RestrictToDirectChatsTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        self.con = sqlite3.connect(self.db_path)
        load_vec_extension(self.con)
        self.addCleanup(self.con.close)
        create_schema(self.con)
        # chats 10/12 are 1:1 (is_group=0); 11/13 are group (is_group=1).
        self.con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, 'iMessage')",
            [(10, "chat10", None, 0),
             (11, "chat11", "The Group", 1),
             (12, "chat12", None, 0),
             (13, "chat13", "Another Group", 1)],
        )
        self.con.commit()

    def test_drops_group_chats(self):
        # Mixed list: only the two 1:1 chats survive, in id order.
        self.assertEqual(_restrict_to_direct_chats(self.con, [10, 11, 12, 13]), [10, 12])

    def test_all_direct_pass_through(self):
        self.assertEqual(_restrict_to_direct_chats(self.con, [10, 12]), [10, 12])

    def test_all_group_returns_empty_list(self):
        self.assertEqual(_restrict_to_direct_chats(self.con, [11, 13]), [])

    def test_none_and_empty_pass_through_unchanged(self):
        # A falsy scope short-circuits: the input is returned as-is, so an
        # unscoped caller never runs the query and never over-restricts.
        self.assertIsNone(_restrict_to_direct_chats(self.con, None))
        self.assertEqual(_restrict_to_direct_chats(self.con, []), [])

    def test_ids_absent_from_chats_are_excluded(self):
        # An id with no chats row is simply absent from the result (it isn't a
        # 1:1 chat), while the real direct chat still passes.
        self.assertEqual(_restrict_to_direct_chats(self.con, [10, 999]), [10])


if __name__ == "__main__":
    unittest.main()
