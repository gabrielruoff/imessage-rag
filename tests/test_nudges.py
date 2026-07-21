"""Hermetic tests for the Loose Ends / nudges core (RES-47 / N1): the
schema + reset_db behavior, the faded_initiator detector, the Stage-0 exclusion
spine, and the three query tools. Temp SQLite only -- no Ollama, no real chat.db
(narration is never exercised here; the CLI runs with --no-llm)."""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import loose_ends
import query_tools
import reset_db
import schema

# corpus "now" is the latest real message date; keep the fixture's anchor here.
_CORPUS_NOW = "2026-07-01T09:00:00"


def _faded_sessions(chat_id, handle_id):
    """13 sessions shaped so the counterparty is a 'faded initiator': 8 old
    sessions they started, 2 old ones I started, 3 recent ones (inside the 90d
    window) I started -- all-time their-share 8/13 (~0.62), trailing 0/3."""
    rows = []
    for _ in range(8):
        rows.append((chat_id, "2024-06-01T10:00:00", "2024-06-01T11:00:00",
                     5, 2, 0, handle_id))       # them
    for _ in range(2):
        rows.append((chat_id, "2024-07-01T10:00:00", "2024-07-01T11:00:00",
                     5, 2, 1, None))            # me, old
    for _ in range(3):
        rows.append((chat_id, "2026-05-15T10:00:00", "2026-05-15T11:00:00",
                     5, 2, 1, None))            # me, recent (in window)
    return rows


def _build_fixture(db_path, *, dan_pre_excluded=True):
    """Alice (faded, survives), Bob (negative valence -> auto-muted), Carol
    (zero outbound -> auto-muted), Dan (pre-muted), and a group chat (excluded
    from the dyad detector). Only Alice should yield a written loose thread."""
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
            "VALUES (?, ?, ?, ?)",
            [(1, "u1", "Alice", "A"), (2, "u2", "Bob", "B"),
             (3, "u3", "Carol", "C"), (4, "u4", "Dan", "D"),
             (5, "u5", "Grace", "G")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+1000", 1), (20, "+2000", 2), (30, "+3000", 3),
             (40, "+4000", 4), (50, "+5000", 5)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, ?, 'iMessage')",
            [(100, "c100", 0), (200, "c200", 0), (300, "c300", 0),
             (400, "c400", 0), (500, "c500", 1)],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50)],
        )
        # Messages: Alice's outbound at corpus-now is the global max date.
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1000, "g1000", 100, 10, 0, "hi", "2026-06-01T08:00:00"),
                (1001, "g1001", 100, 0, 1, "you around?", _CORPUS_NOW),
                (2000, "g2000", 200, 20, 0, "whatever", "2026-05-01T08:00:00"),
                (2001, "g2001", 200, 0, 1, "fine", "2026-05-02T09:00:00"),
                (3000, "g3000", 300, 30, 0, "hello?", "2026-05-01T08:00:00"),
                (4000, "g4000", 400, 40, 0, "hey", "2026-05-01T08:00:00"),
                (4001, "g4001", 400, 0, 1, "hi", "2026-05-02T09:00:00"),
                (5000, "g5000", 500, 50, 0, "group", "2026-05-01T08:00:00"),
                (5001, "g5001", 500, 0, 1, "yo", "2026-05-02T09:00:00"),
            ],
        )
        # Bob's most-recent llm-annotated exchange is negative -> Stage-0 mute.
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(1, 200, 2000, 2001, '2026-05-01T08:00:00', '2026-05-02T09:00:00', 2, 'x')"
        )
        con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (1, -0.8, '[]', 1, '[]', 'llm')"
        )
        # Faded sessions for the direct chats and the group (group is filtered).
        for cid, hid in [(100, 10), (200, 20), (400, 40), (500, 50)]:
            con.executemany(
                "INSERT INTO sessions(chat_id, first_date, last_date, n_messages, "
                "n_from_me, initiator_is_me, initiator_handle_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                _faded_sessions(cid, hid),
            )
        if dan_pre_excluded:
            con.execute(
                "INSERT INTO nudge_exclusions(handle_id, contact_id, reason, created_at) "
                "VALUES (40, 4, 'muted', '2026-01-01T00:00:00')"
            )
        con.commit()
    finally:
        con.close()


class SchemaAndResetTest(unittest.TestCase):
    def test_ddl_is_idempotent(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        schema.create_schema(con)
        schema.create_schema(con)  # second run must be a no-op, not an error
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("loose_threads", tables)
        self.assertIn("nudge_exclusions", tables)

    def test_nudge_exclusions_dedups_on_unique(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        schema.create_schema(con)
        con.execute("INSERT INTO nudge_exclusions(handle_id, contact_id, reason) "
                    "VALUES (7, 3, 'muted')")
        with self.assertRaises(sqlite3.IntegrityError):
            con.execute("INSERT INTO nudge_exclusions(handle_id, contact_id, reason) "
                        "VALUES (7, 3, 'muted')")

    def test_reset_clears_loose_threads_keeps_exclusions(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        con = sqlite3.connect(path)
        schema.create_schema(con)
        con.execute("INSERT INTO loose_threads(kind, chat_id, status) "
                    "VALUES ('faded_initiator', 100, 'open')")
        con.execute("INSERT INTO nudge_exclusions(handle_id, contact_id, reason) "
                    "VALUES (10, 1, 'muted')")
        con.commit()
        con.close()

        with mock.patch.object(reset_db, "DB_PATH", Path(path)):
            with mock.patch.object(sys, "argv", ["reset_db.py", "-f"]):
                reset_db.main()

        con = sqlite3.connect(path)
        self.addCleanup(con.close)
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM loose_threads").fetchone()[0], 0)
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM nudge_exclusions").fetchone()[0], 1)


class _FixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def _threads(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(
                "SELECT id, kind, chat_id, handle_id, status, urgency, evidence_json "
                "FROM loose_threads").fetchall()
        finally:
            con.close()


class FadedInitiatorTest(_FixtureTest):
    def test_detector_yields_direct_faded_chats_only(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cands = list(loose_ends.detect_faded_initiator(con, cfg))
        by_chat = {c.chat_id: c for c in cands}
        # 100/200/400 are direct + faded; 300 has no faded sessions; 500 is a
        # group chat and must be excluded (F3 _restrict_to_direct_chats).
        self.assertEqual(set(by_chat), {100, 200, 400})
        self.assertNotIn(500, by_chat)
        alice = by_chat[100]
        self.assertEqual(alice.kind, "faded_initiator")
        self.assertEqual(alice.handle_id, 10)
        self.assertEqual(alice.anchor_message_id, 1001)
        self.assertEqual(alice.anchor_date, _CORPUS_NOW)
        self.assertEqual(alice.evidence["trailing_initiation_share"], 0.0)
        self.assertEqual(alice.evidence["lifetime_sessions"], 13)

    def test_min_sessions_threshold_excludes_thin_dyads(self):
        # Raise the lifetime-sessions floor above the fixture's 13 -> nothing.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["faded_min_sessions"] = 99
        self.assertEqual(list(loose_ends.detect_faded_initiator(con, cfg)), [])


class ExclusionGateTest(_FixtureTest):
    def test_full_run_writes_only_the_unexcluded_dyad(self):
        loose_ends.main(["--no-llm"])
        rows = self._threads()
        # Alice alone survives: Bob (negative valence) and Dan (pre-muted) are
        # suppressed before any row is written; Carol (zero outbound) and the
        # group are never candidates.
        self.assertEqual([r["chat_id"] for r in rows], [100])
        self.assertEqual(rows[0]["kind"], "faded_initiator")
        self.assertEqual(rows[0]["status"], "open")

        reasons = {
            (r[0], r[1]): r[2] for r in sqlite3.connect(self.db_path).execute(
                "SELECT contact_id, handle_id, reason FROM nudge_exclusions")
        }
        self.assertEqual(reasons[(2, 20)], "auto:negative_valence")  # Bob
        self.assertEqual(reasons[(3, 30)], "auto:zero_outbound")     # Carol
        self.assertEqual(reasons[(4, 40)], "muted")                  # Dan (pre-existing)

    def test_dry_run_writes_nothing(self):
        loose_ends.main(["--no-llm", "--dry-run"])
        self.assertEqual(self._threads(), [])

    def test_rerun_is_idempotent_and_respects_dismissal(self):
        loose_ends.main(["--no-llm"])
        (tid,) = sqlite3.connect(self.db_path).execute(
            "SELECT id FROM loose_threads").fetchone()
        query_tools.resolve_loose_thread(tid)
        loose_ends.main(["--no-llm"])  # must NOT re-open the dismissed thread
        rows = self._threads()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "dismissed")


class DyadGapAndRegistryTest(_FixtureTest):
    def test_registry_and_helper_are_exported_with_docstrings(self):
        self.assertTrue(loose_ends.DETECTORS)
        kinds = [k for k, _ in loose_ends.DETECTORS]
        self.assertIn("faded_initiator", kinds)
        for _kind, fn in loose_ends.DETECTORS:
            self.assertTrue(callable(fn))
            self.assertTrue(fn.__doc__)
        self.assertTrue(loose_ends._dyad_gap_p90.__doc__)

    def test_dyad_gap_p90(self):
        con = self._open()
        # Alice's chat 100 has two messages -> exactly one gap; p90 of one gap
        # is that gap. Chat 300 has a single message -> None.
        self.assertIsNotNone(loose_ends._dyad_gap_p90(con, 100))
        self.assertIsNone(loose_ends._dyad_gap_p90(con, 300))


class ToolsTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (1, 'u1', 'Alice')")
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, 1, 'iMessage')", [(10, "+1000"), (11, "+1001")])
        con.commit()
        con.close()
        patcher = mock.patch.object(query_tools, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _insert_thread(self, urgency, handle_id=10, status="open"):
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.execute(
                "INSERT INTO loose_threads(kind, chat_id, handle_id, urgency, "
                "status, evidence_json) VALUES ('faded_initiator', ?, ?, ?, ?, '{}')",
                (100 + int(urgency * 100), handle_id, urgency, status))
            con.commit()
            return cur.lastrowid
        finally:
            con.close()

    def test_get_loose_threads_is_capped_and_ranked(self):
        for u in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            self._insert_thread(u)
        rows = query_tools.get_loose_threads(limit=100)
        self.assertEqual(len(rows), 5)  # hard-capped at NUDGE_MAX_RESULTS
        urgencies = [r["urgency"] for r in rows]
        self.assertEqual(urgencies, sorted(urgencies, reverse=True))
        self.assertEqual(urgencies[0], 0.8)
        # evidence_json is parsed into `evidence`; the raw key is gone.
        self.assertIn("evidence", rows[0])
        self.assertNotIn("evidence_json", rows[0])

    def test_get_loose_threads_status_filter(self):
        self._insert_thread(0.5, status="open")
        self._insert_thread(0.9, status="dismissed")
        self.assertEqual(len(query_tools.get_loose_threads(status="open")), 1)
        self.assertEqual(len(query_tools.get_loose_threads(status="dismissed")), 1)

    def test_resolve_sets_dismissed_at(self):
        tid = self._insert_thread(0.5)
        result = query_tools.resolve_loose_thread(tid)
        self.assertTrue(result["dismissed"])
        row = sqlite3.connect(self.db_path).execute(
            "SELECT status, dismissed_at FROM loose_threads WHERE id = ?", (tid,)
        ).fetchone()
        self.assertEqual(row[0], "dismissed")
        self.assertIsNotNone(row[1])

    def test_resolve_unknown_id_reports_error(self):
        result = query_tools.resolve_loose_thread(999999)
        self.assertFalse(result["dismissed"])
        self.assertIn("error", result)

    def test_mute_writes_exclusion_and_dismisses_open_threads(self):
        t1 = self._insert_thread(0.5, handle_id=10)
        t2 = self._insert_thread(0.6, handle_id=11)  # same contact, other handle
        result = query_tools.mute_contact_from_nudges(contact_id=1)
        self.assertTrue(result["muted"])
        self.assertTrue(result["exclusion_added"])
        self.assertEqual(result["dismissed_threads"], 2)

        con = sqlite3.connect(self.db_path)
        statuses = {r[0] for r in con.execute(
            "SELECT status FROM loose_threads WHERE id IN (?, ?)", (t1, t2))}
        self.assertEqual(statuses, {"dismissed"})
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM nudge_exclusions WHERE contact_id = 1")
            .fetchone()[0], 1)

    def test_mute_twice_does_not_stack_rows(self):
        query_tools.mute_contact_from_nudges(contact_id=1)
        second = query_tools.mute_contact_from_nudges(contact_id=1)
        self.assertFalse(second["exclusion_added"])
        self.assertEqual(
            sqlite3.connect(self.db_path).execute(
                "SELECT COUNT(*) FROM nudge_exclusions").fetchone()[0], 1)

    def test_mute_needs_a_target(self):
        with self.assertRaises(ValueError):
            query_tools.mute_contact_from_nudges()


class RegistrationSyncTest(unittest.TestCase):
    """The 3-place registration must stay in sync: every schema name is a
    query_tools function of the same name, and appears in chat.py's tool list."""

    def test_nudge_tools_registered_everywhere(self):
        import chat
        from tool_schemas import TOOL_DEFINITIONS
        names = {"get_loose_threads", "resolve_loose_thread", "mute_contact_from_nudges"}
        schema_names = {t["name"] for t in TOOL_DEFINITIONS}
        self.assertTrue(names <= schema_names)
        for name in names:
            self.assertTrue(callable(getattr(query_tools, name)))
            self.assertIn(name, chat.TOOL_FUNCTIONS)
            self.assertIn(name, chat.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
