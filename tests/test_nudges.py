"""Hermetic tests for the Loose Ends / nudges core (RES-47 / N1): the
schema + reset_db behavior, the faded_initiator detector, the Stage-0 exclusion
spine, and the three query tools. Temp SQLite only -- no Ollama, no real chat.db
(narration is never exercised here; the CLI runs with --no-llm)."""

import datetime
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import db
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
        # --dry-run must not persist the Stage-0 auto-mutes either: only Dan's
        # pre-existing 'muted' row survives. A non-dry run would additionally add
        # auto:negative_valence (Bob) and auto:zero_outbound (Carol), and those
        # rows are preserved across reset_db -- so a "preview" writing them would
        # permanently mute contacts.
        rows = sqlite3.connect(self.db_path).execute(
            "SELECT contact_id, handle_id, reason FROM nudge_exclusions "
            "ORDER BY contact_id").fetchall()
        self.assertEqual(rows, [(4, 40, "muted")])

    def test_rerun_is_idempotent_and_respects_dismissal(self):
        loose_ends.main(["--no-llm"])
        (tid,) = sqlite3.connect(self.db_path).execute(
            "SELECT id FROM loose_threads").fetchone()
        query_tools.resolve_loose_thread(tid)
        loose_ends.main(["--no-llm"])  # must NOT re-open the dismissed thread
        rows = self._threads()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "dismissed")

    def test_owner_self_contact_is_excluded(self):
        # F4: mark Alice's contact (the only surviving dyad) as the account
        # owner -> her self-chat is suppressed with reason auto:self_identity,
        # and no thread is written.
        with mock.patch.object(loose_ends, "NUDGE_OWNER_CONTACT_IDS", "1"):
            loose_ends.main(["--no-llm"])
        self.assertEqual(self._threads(), [])
        reason = sqlite3.connect(self.db_path).execute(
            "SELECT reason FROM nudge_exclusions WHERE contact_id = 1"
        ).fetchone()
        self.assertEqual(reason[0], "auto:self_identity")

    def test_reconciliation_resolves_undetected_open_thread(self):
        # A stale open thread whose (kind, chat_id) no detector re-emits is
        # auto-closed to 'resolved' on the next run (never left dangling open).
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
            "VALUES ('faded_initiator', 555, 'open', '2020-01-01')")
        con.commit()
        con.close()
        loose_ends.main(["--no-llm"])
        statuses = dict(sqlite3.connect(self.db_path).execute(
            "SELECT chat_id, status FROM loose_threads").fetchall())
        self.assertEqual(statuses[555], "resolved")   # stale -> auto-closed
        self.assertEqual(statuses[100], "open")        # Alice still detected


class LlmModelGuardTest(_FixtureTest):
    """loose_ends_model guards the LLM-derived columns. It REFUSES a model
    change while they hold anything -- it used to clear them instead, which the
    CHAT_MODEL default flip turned into a silent wipe on every upgrade, since
    NUDGES_MODEL inherits it."""

    def _narrated(self, model="model-a", verdict=None):
        loose_ends.main(["--no-llm", "--model", model])
        con = sqlite3.connect(self.db_path)
        con.execute("UPDATE loose_threads SET nudge_text = 'go say hi'")
        if verdict is not None:
            con.execute("UPDATE loose_threads SET llm_verdict = ?, llm_reason = 'r'",
                        (verdict,))
        con.execute("INSERT OR REPLACE INTO ingest_state(key, value) "
                    "VALUES ('loose_ends_model', ?)", (model,))
        con.commit()
        con.close()

    def _llm_columns(self):
        con = sqlite3.connect(self.db_path)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM loose_threads WHERE nudge_text IS NOT NULL "
                "OR llm_verdict IS NOT NULL OR llm_reason IS NOT NULL").fetchone()[0]
        finally:
            con.close()

    def test_a_model_change_is_refused_with_the_columns_intact(self):
        self._narrated(verdict="plan")
        before = self._llm_columns()
        self.assertGreater(before, 0)
        with self.assertRaises(SystemExit):
            loose_ends.main(["--no-llm", "--model", "model-b"])
        self.assertEqual(self._llm_columns(), before,
                         "the guard destroyed the columns it exists to protect")

    def test_a_refused_run_does_not_record_the_new_model(self):
        # record=False on the early call: the narration pass owns the write, so
        # a run refused before it gets there leaves the key as it found it.
        self._narrated(verdict="plan")
        with self.assertRaises(SystemExit):
            loose_ends.main(["--model", "model-b"])
        con = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(
                con.execute("SELECT value FROM ingest_state "
                            "WHERE key = 'loose_ends_model'").fetchone()[0],
                "model-a", "a refused run must not record the new model")
        finally:
            con.close()

    def test_no_llm_is_refused_only_when_dropped_plan_can_actually_run(self):
        # --no-llm makes zero LLM calls with dropped_plan off (the default), so
        # it can touch no guarded column and must not be refused -- the repo's
        # rule that a guard never refuses a run it cannot affect. With the
        # detector ON it DOES spend NUDGES_MODEL calls during detection and
        # stamps evidence_json.verified_by, so it is guarded.
        self._narrated()
        with mock.patch.object(loose_ends, "NUDGE_DROPPED_PLAN_ENABLED", False):
            loose_ends.main(["--no-llm", "--model", "model-b"])   # allowed
        with mock.patch.object(loose_ends, "NUDGE_DROPPED_PLAN_ENABLED", True):
            with self.assertRaises(SystemExit):
                loose_ends.main(["--no-llm", "--model", "model-c"])

    def test_model_scopes_detection_not_just_narration(self):
        # dropped_plan verifies with cfg["nudges_model"] and stamps it into
        # evidence_json.verified_by, so --model has to reach _build_cfg or the
        # guard would clear one model while the detector spent another.
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(loose_ends._build_cfg(con, "model-x")["nudges_model"],
                         "model-x")
        self.assertEqual(loose_ends._build_cfg(con)["nudges_model"],
                         loose_ends.NUDGES_MODEL)

    def test_the_same_model_runs_normally(self):
        self._narrated()
        loose_ends.main(["--no-llm", "--model", "model-a"])
        self.assertGreater(self._llm_columns(), 0)

    def test_a_model_change_is_allowed_when_nothing_is_narrated(self):
        # Nothing to protect: the guard must not block a run over a digest that
        # has no LLM output yet (annotate's guard has the same carve-out). The
        # key is seeded WITHOUT writing an LLM column, so this really does
        # reach `prev and prev != model` with n == 0 -- seeding it via a run
        # would leave prev unset and the test would pass vacuously.
        loose_ends.main(["--no-llm", "--model", "model-a"])
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT OR REPLACE INTO ingest_state(key, value) "
                    "VALUES ('loose_ends_model', 'model-a')")
        con.commit()
        con.close()
        self.assertEqual(self._llm_columns(), 0)
        loose_ends.main(["--no-llm", "--model", "model-b"])
        self.assertEqual(self._llm_columns(), 0)

    def test_dry_run_neither_guards_nor_records(self):
        self._narrated()
        loose_ends.main(["--no-llm", "--dry-run", "--model", "model-b"])
        con = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(
                con.execute("SELECT value FROM ingest_state "
                            "WHERE key = 'loose_ends_model'").fetchone()[0],
                "model-a")
        finally:
            con.close()


class SenderClassifierTest(unittest.TestCase):
    CFG = {"owner_contact_ids": frozenset({42}),
           "owner_handles": frozenset({"+14438480968"}),
           "shortcode_max_digits": 6}

    def test_owner_by_contact_id_and_handle(self):
        self.assertTrue(loose_ends._is_owner_counterparty(42, "+1555", self.CFG))
        self.assertTrue(loose_ends._is_owner_counterparty(None, "+14438480968", self.CFG))
        self.assertFalse(loose_ends._is_owner_counterparty(7, "+1555", self.CFG))

    def test_shortcode_is_automated_real_number_is_not(self):
        self.assertTrue(loose_ends._is_automated_sender("22022", self.CFG))
        self.assertTrue(loose_ends._is_automated_sender("urn:biz:xyz", self.CFG))
        self.assertFalse(loose_ends._is_automated_sender("+16074345333", self.CFG))
        self.assertFalse(loose_ends._is_automated_sender("+1000", self.CFG))  # has '+'
        self.assertFalse(loose_ends._is_automated_sender("6074345333", self.CFG))  # long


class ThreadSummaryTest(unittest.TestCase):
    """The deterministic per-kind summary that grounds the model's narration."""
    _summary = staticmethod(query_tools.nudges._thread_summary)

    def test_faded_initiator_states_the_counterparty_went_quiet(self):
        # F11: the initiation share is the COUNTERPARTY's -- the summary must say
        # THEY went quiet, never that the user stopped initiating.
        s = self._summary("faded_initiator", "Alex",
                          {"all_time_initiation_share": "0.62",
                           "trailing_initiation_share": "0.0"})
        self.assertIn("Alex used to start conversations with you", s)
        self.assertIn("62%", s)
        self.assertIn("gone quiet", s)

    def test_inferred_occasion_does_not_guess_a_type(self):
        # F12: a 'Congrats women' occasion must not be narrated as a 'birthday';
        # the summary stays generic and quotes the real past message.
        s = self._summary("inferred_occasion", "Jill",
                          {"month_day": "08-06", "observed_years": "['2024', '2025']",
                           "sample_message": "Congrats women"})
        self.assertIn("recurring date", s)
        self.assertIn("Congrats women", s)
        self.assertNotIn("birthday", s.lower())

    def test_unanswered_question_quotes_the_question(self):
        s = self._summary("unanswered_question", "Sam", {"question": "you free sat?"})
        self.assertIn("Sam asked", s)
        self.assertIn("you free sat?", s)

    def test_unknown_kind_returns_none(self):
        self.assertIsNone(self._summary("mystery_kind", "X", {}))


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
        threads = [r for r in rows if not r.get("digest_meta")]
        self.assertEqual(len(threads), 5)  # hard-capped at NUDGE_MAX_RESULTS
        urgencies = [r["urgency"] for r in threads]
        self.assertEqual(urgencies, sorted(urgencies, reverse=True))
        self.assertEqual(urgencies[0], 0.8)
        # evidence_json is parsed into `evidence`; the raw key is gone.
        self.assertIn("evidence", threads[0])
        self.assertNotIn("evidence_json", threads[0])
        self.assertIn("summary", threads[0])  # F11/F12: grounded per-kind blurb
        # F8: a completeness meta row exposes the un-capped total (8) so the
        # caller never mistakes the capped sample for the whole list.
        (meta,) = [r for r in rows if r.get("digest_meta")]
        self.assertEqual(meta["n_total"], 8)
        self.assertEqual(meta["returned"], 5)
        self.assertTrue(meta["truncated"])

    def test_get_loose_threads_no_meta_when_not_truncated(self):
        # Below the cap: no digest_meta row (nothing is hidden).
        for u in (0.5, 0.6):
            self._insert_thread(u)
        rows = query_tools.get_loose_threads(limit=100)
        self.assertEqual(len(rows), 2)
        self.assertFalse([r for r in rows if r.get("digest_meta")])

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


# ---------------------------------------------------------------------------
# dormant_tie detector (RES-55 / N2)
# ---------------------------------------------------------------------------

def _dense(chat_id, handle_id, start_id, start_dt, n, step_hours, *, all_inbound=False):
    """n messages alternating sender (or all inbound), spaced step_hours apart
    from start_dt. is_from_me=1 rows carry handle_id 0 (matching the N1 fixture);
    inbound rows carry the counterparty handle_id. Returns (rows, last_dt)."""
    rows, last = [], start_dt
    for i in range(n):
        dt = start_dt + datetime.timedelta(hours=step_hours * i)
        is_me = 0 if all_inbound else (i % 2)
        hid = 0 if is_me else handle_id
        rows.append((start_id + i, f"g{start_id + i}", chat_id, hid, is_me,
                     "msg", dt.strftime("%Y-%m-%dT%H:%M:%S")))
        last = dt
    return rows, last


_LEX_COLS = ("chat_id, handle_id, month, n_words, fw_pronoun_rate, fw_ipron_rate, "
             "fw_article_rate, fw_conj_rate, fw_prep_rate, fw_aux_rate, "
             "fw_adverb_rate, fw_negation_rate, fw_quant_rate")


def _build_dormant_fixture(db_path):
    """Direct dyads for the dormant_tie detector, anchored so corpus-now is
    Carol's recent message:
      Alice (100)  once-close (300 msgs, depth 5, LSM~1) + DORMANT + OVERDUE -> fires
      Beth  (200)  eligible but less close (200 msgs, depth 2, LSM~0.4) + OVERDUE
      Carol (300)  close (240 msgs) but RECENT -> not overdue (anchors corpus-now)
      Bob   (400)  once-close + OVERDUE but NEGATIVE valence -> Stage-0 muted
      Ed    (500)  once-close + OVERDUE but ZERO outbound   -> Stage-0 muted
      Frank (600)  bursty history (15 msgs, months apart) -> the p90 fixture
    """
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (?, ?, ?)",
            [(i, f"u{i}", n) for i, n in
             [(1, "Alice"), (2, "Beth"), (3, "Carol"), (4, "Bob"),
              (5, "Ed"), (6, "Frank")]],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+10", 1), (20, "+20", 2), (30, "+30", 3),
             (40, "+40", 4), (50, "+50", 5), (60, "+60", 6)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, 0, 'iMessage')",
            [(c, f"c{c}") for c in (100, 200, 300, 400, 500, 600)],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50), (600, 60)],
        )

        d = datetime.datetime
        msgs = []
        msgs += _dense(100, 10, 1000, d(2025, 8, 1), 300, 8)[0]        # Alice dormant
        msgs += _dense(200, 20, 2000, d(2025, 8, 1), 200, 8)[0]        # Beth dormant
        msgs += _dense(300, 30, 3000, d(2026, 4, 1), 240, 8)[0]        # Carol recent (max)
        msgs += _dense(400, 40, 4000, d(2025, 8, 1), 240, 8)[0]        # Bob dormant
        msgs += _dense(500, 50, 5000, d(2025, 8, 1), 240, 8,
                       all_inbound=True)[0]                            # Ed dormant, 0 out
        # Frank: 5 bursts of 3 quick messages, 30 days apart (the p90 fixture).
        mid = 6000
        for b in range(5):
            base = d(2025, 1, 1) + datetime.timedelta(days=30 * b)
            for k in range(3):
                t = base + datetime.timedelta(minutes=5 * k)
                msgs.append((mid, f"g{mid}", 600, (0 if k % 2 else 60), k % 2,
                             "hi", t.strftime("%Y-%m-%dT%H:%M:%S")))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            msgs,
        )

        # Depth chunks (source='distilled' -> depth counts, valence gate does not).
        con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, 2, 'x')",
            [(900, 100, 1000, 1001, "2025-08-01T00:00:00", "2025-09-01T00:00:00"),
             (901, 200, 2000, 2001, "2025-08-01T00:00:00", "2025-09-01T00:00:00"),
             (902, 400, 4000, 4001, "2025-08-01T00:00:00", "2025-09-01T00:00:00")],
        )
        con.executemany(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (?, ?, '[]', ?, '[]', ?)",
            [(900, None, 5, "distilled"),   # Alice: high depth, no valence gate
             (901, None, 2, "distilled"),   # Beth: lower depth
             (902, -0.8, 3, "llm")],        # Bob: negative llm valence -> muted
        )

        # LSM lexicon: Alice near-identical both sides (~1.0), Beth divergent (~0.4).
        same = (0.1,) * 9
        beth_me = (0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2)
        beth_them = (0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05)
        con.executemany(
            f"INSERT INTO person_period_lexicon({_LEX_COLS}) "
            f"VALUES ({','.join('?' * 13)})",
            [(100, None, "2025-09", 100, *same),
             (100, 10, "2025-09", 100, *same),
             (200, None, "2025-09", 100, *beth_me),
             (200, 20, "2025-09", 100, *beth_them)],
        )
        con.commit()
    finally:
        con.close()


class _DormantFixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_dormant_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def _dormant_chats(self):
        con = sqlite3.connect(self.db_path)
        try:
            return [r[0] for r in con.execute(
                "SELECT chat_id FROM loose_threads WHERE kind = 'dormant_tie' "
                "AND status = 'open' ORDER BY chat_id")]
        finally:
            con.close()


class DormantTieDetectorTest(_DormantFixtureTest):
    def test_p90_uses_full_gap_distribution_not_reply_latency(self):
        # Frank (600) messages in bursts 30 days apart. The FULL-distribution p90
        # reflects the long silences; a reply-latency median (within-burst gaps
        # only) would be minutes and is circular -- it excludes the silences.
        con = self._open()
        p90 = loose_ends._dyad_gap_p90(con, 600)
        dates = [datetime.datetime.fromisoformat(r[0]) for r in con.execute(
            "SELECT date FROM messages WHERE chat_id = 600 ORDER BY date, id")]
        gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
        reply_latency_median = statistics.median(g for g in gaps if g < 86400)
        self.assertLess(reply_latency_median, 3600)          # minutes
        self.assertGreater(p90, 10 * 86400)                  # p90 is days, not minutes
        self.assertGreater(p90, reply_latency_median * 100)  # not circular

    def test_top_frac_selects_only_the_closest_overdue_dyad(self):
        # Default top_frac (0.2) over 5 eligible dyads -> ceil = 1 slot. Alice is
        # the closest (most messages + highest depth + highest LSM), so only she
        # is a candidate even though Beth/Bob/Ed are also overdue and eligible.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cands = list(loose_ends.detect_dormant_tie(con, cfg))
        self.assertEqual([c.chat_id for c in cands], [100])
        ev = cands[0].evidence
        self.assertEqual(ev["trailing_messages"], 300)
        self.assertEqual(ev["mean_depth"], 5.0)
        self.assertIsNotNone(ev["lsm"])
        self.assertGreater(ev["current_gap_days"], ev["p90_gap_days"])
        self.assertGreater(ev["current_gap_days"], 21)

    def test_recent_and_thin_dyads_do_not_fire(self):
        # With top_frac=1.0 every eligible dyad is "once-close"; the overdue gate
        # must still drop Carol (recent) and never invent Frank (< min messages).
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["dormant_top_frac"] = 1.0
        chats = {c.chat_id for c in loose_ends.detect_dormant_tie(con, cfg)}
        self.assertIn(100, chats)
        self.assertIn(200, chats)
        self.assertNotIn(300, chats)   # Carol: recent -> not overdue
        self.assertNotIn(600, chats)   # Frank: 15 msgs -> not eligible


class DormantTieDriverExclusionTest(_DormantFixtureTest):
    def test_excluded_contacts_never_surface_through_the_driver(self):
        # top_frac=1.0 makes Bob (negative valence) and Ed (zero outbound)
        # candidates; the Stage-0 gate must suppress them before any row lands.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["dormant_top_frac"] = 1.0
        summary = loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        self.assertEqual(self._dormant_chats(), [100, 200])
        self.assertGreaterEqual(summary["suppressed"], 2)   # Bob + Ed
        reasons = dict(con.execute(
            "SELECT chat_id, reason FROM nudge_exclusions ne "
            "JOIN chat_participants cp ON cp.handle_id = ne.handle_id"))
        self.assertEqual(reasons.get(400), "auto:negative_valence")
        self.assertEqual(reasons.get(500), "auto:zero_outbound")


class DormantTieBatchHideTest(_DormantFixtureTest):
    def _write_dormant(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["dormant_top_frac"] = 1.0
        loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")

    def test_hidden_until_acknowledged_then_visible(self):
        self._write_dormant()
        # Absent nudges_reconnect_ack: no dormant rows, plus the one-line notice.
        rows = query_tools.get_loose_threads(kind="dormant_tie", limit=5)
        self.assertTrue(all(r.get("kind") != "dormant_tie" or "notice" in r
                            for r in rows))
        self.assertFalse([r for r in rows if r.get("status") == "open"
                          and "notice" not in r])
        notices = [r for r in rows if "notice" in r]
        self.assertEqual(len(notices), 1)
        self.assertIn("--acknowledge-reconnect", notices[0]["notice"])

        # The unscoped digest also hides dormant rows but still carries the notice.
        allrows = query_tools.get_loose_threads(limit=5)
        self.assertFalse([r for r in allrows if r.get("kind") == "dormant_tie"
                          and "notice" not in r])
        self.assertTrue(any("notice" in r for r in allrows))

        # Opt in via the CLI flag, then the rows appear and the notice is gone.
        loose_ends.main(["--acknowledge-reconnect"])
        after = query_tools.get_loose_threads(kind="dormant_tie", limit=5)
        self.assertEqual(sorted(r["chat_id"] for r in after), [100, 200])
        self.assertFalse([r for r in after if "notice" in r])


class DormantTieConfigTest(unittest.TestCase):
    def test_knobs_are_named_config_vars(self):
        self.assertEqual(config.NUDGE_DORMANT_TOP_FRAC, 0.2)
        self.assertEqual(config.NUDGE_DORMANT_MIN_MESSAGES, 200)
        self.assertEqual(config.NUDGE_DORMANT_MIN_GAP_DAYS, 21)

    def test_dormant_tie_is_registered(self):
        kinds = [k for k, _ in loose_ends.DETECTORS]
        self.assertIn("dormant_tie", kinds)


# ---------------------------------------------------------------------------
# unanswered_question detector (RES-56 / N3)
# ---------------------------------------------------------------------------

def _build_unanswered_fixture(db_path):
    """Direct dyads for the unanswered_question detector; corpus-now is Carol's
    late self-reply (the global max date):
      Alice (100)  genuine question, never answered           -> fires
      Bob   (200)  genuine question, answered 90 min later     -> Stage-1b suppresses
      Carol (300)  genuine question, answered 10 days later     -> fires (late reply kept)
      Dave  (400)  rhetorical last question ("How are you?")    -> stoplist excludes
      Eve   (500)  short last question ("you around?", 11 ch)   -> length excludes
      group (600)  genuine question but is_group = 1            -> direct-chat filter excludes
    """
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (?, ?, ?)",
            [(1, "u1", "Alice"), (2, "u2", "Bob"), (3, "u3", "Carol"),
             (4, "u4", "Dave"), (5, "u5", "Eve"), (6, "u6", "Grace")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+10", 1), (20, "+20", 2), (30, "+30", 3),
             (40, "+40", 4), (50, "+50", 5), (60, "+60", 6)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, ?, 'iMessage')",
            [(100, "c100", 0), (200, "c200", 0), (300, "c300", 0),
             (400, "c400", 0), (500, "c500", 0), (600, "c600", 1)],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50), (600, 60)],
        )
        # is_from_me=1 rows carry handle_id 0 (matching the other fixtures).
        msgs = [
            # Alice: genuine question, never answered.
            (1000, "g1000", 100, 0, 1, "hey how was your trip", "2026-05-01T10:00:00"),
            (1001, "g1001", 100, 10, 0, "it was great thanks", "2026-05-01T10:30:00"),
            (1002, "g1002", 100, 10, 0,
             "Do you want to grab dinner this weekend?", "2026-05-05T09:00:00"),
            # Bob: genuine question, answered 90 minutes later (a new session).
            (2000, "g2000", 200, 0, 1, "hey", "2026-05-01T10:00:00"),
            (2001, "g2001", 200, 20, 0,
             "Are we still on for lunch tomorrow?", "2026-05-02T10:00:00"),
            (2002, "g2002", 200, 0, 1, "yes definitely", "2026-05-02T11:30:00"),
            # Dave: rhetorical last question (12 chars -> passes length, stoplist drops).
            (4000, "g4000", 400, 0, 1, "hey", "2026-05-01T10:00:00"),
            (4001, "g4001", 400, 40, 0, "How are you?", "2026-05-03T10:00:00"),
            # Eve: short last question (11 chars -> length drops).
            (5000, "g5000", 500, 0, 1, "yo", "2026-05-01T10:00:00"),
            (5001, "g5001", 500, 50, 0, "you around?", "2026-05-04T10:00:00"),
            # Group: a genuine question, but is_group -> filtered.
            (6000, "g6000", 600, 60, 0,
             "Are we all set for the group trip?", "2026-05-05T10:00:00"),
        ]
        # Carol: 20 dense messages 1h apart (keeps p90 ~ 1h, under the 72h
        # floor), then the question, then a self-reply 10 days later (past the
        # deadline). The late reply is the global max date -> corpus-now.
        base = datetime.datetime(2026, 5, 10, 9, 0, 0)
        for i in range(20):
            dt = base + datetime.timedelta(hours=i)
            is_me = 1 if i % 2 == 0 else 0
            msgs.append((3000 + i, f"g{3000 + i}", 300, 0 if is_me else 30, is_me,
                         "chatting", dt.strftime("%Y-%m-%dT%H:%M:%S")))
        q_dt = base + datetime.timedelta(hours=20)
        reply_dt = q_dt + datetime.timedelta(days=10)
        msgs.append((3020, "g3020", 300, 30, 0,
                     "What did you think of the movie we saw?",
                     q_dt.strftime("%Y-%m-%dT%H:%M:%S")))
        msgs.append((3021, "g3021", 300, 0, 1, "sorry just saw this, loved it",
                     reply_dt.strftime("%Y-%m-%dT%H:%M:%S")))
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            msgs,
        )
        # One session per chat, ending on the question. Carol's late reply (3021)
        # sits in a later session the detector never treats as a candidate.
        con.executemany(
            "INSERT INTO sessions(chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, n_from_me) "
            "VALUES (?, ?, ?, '2026-05-01T00:00:00', '2026-05-11T05:00:00', 2, 1)",
            [(100, 1000, 1002), (200, 2000, 2001), (300, 3000, 3020),
             (400, 4000, 4001), (500, 5000, 5001), (600, 6000, 6000)],
        )
        con.commit()
    finally:
        con.close()


class _UnansweredFixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_unanswered_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con


class UnansweredQuestionDetectorTest(_UnansweredFixtureTest):
    def test_trailing_length_and_stoplist_filter(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cands = {c.chat_id: c for c in loose_ends.detect_unanswered_question(con, cfg)}
        # Alice (genuine, unanswered) fires. Carol was answered after the
        # deadline (just late) -> RESOLVED, suppressed by default. Bob is
        # answered inside the deadline, Dave is rhetorical, Eve is under the
        # length floor, the group chat is filtered out.
        self.assertEqual(set(cands), {100})
        self.assertNotIn(300, cands)   # answered late -> resolved, not open
        self.assertNotIn(200, cands)   # answered within the deadline
        self.assertNotIn(400, cands)   # rhetorical ("How are you?")
        self.assertNotIn(500, cands)   # short ("you around?", 11 chars)
        self.assertNotIn(600, cands)   # group chat
        alice = cands[100]
        self.assertEqual(alice.kind, "unanswered_question")
        self.assertEqual(alice.handle_id, 10)
        self.assertEqual(alice.anchor_message_id, 1002)
        self.assertEqual(alice.anchor_date, "2026-05-05T09:00:00")
        self.assertTrue(alice.evidence["question"].startswith("Do you want"))
        self.assertFalse(alice.evidence["answered_late"])
        self.assertNotIn("late_reply_date", alice.evidence)

    def test_stale_question_dropped_by_max_age(self):
        # F6: a question unanswered longer than question_max_age_days is too stale
        # to be an actionable loose end. Alice fires by default; with a zero-day
        # ceiling her (positive) silence exceeds it and she is dropped.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        self.assertIn(100, {c.chat_id
                            for c in loose_ends.detect_unanswered_question(con, cfg)})
        cfg["question_max_age_days"] = 0
        self.assertNotIn(100, {c.chat_id
                               for c in loose_ends.detect_unanswered_question(con, cfg)})

    def test_stage1b_suppresses_reply_within_deadline(self):
        # Bob's question is answered 90 minutes later (a new session) -- well
        # inside the 72h deadline -- so Stage-1b suppresses it: no candidate.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        chats = {c.chat_id for c in loose_ends.detect_unanswered_question(con, cfg)}
        self.assertNotIn(200, chats)

    def test_answered_late_suppressed_by_default(self):
        # Carol's question is answered 10 days later, past the 72h deadline. A
        # late reply still RESOLVES the question, so by default it is not an open
        # loose end (surfacing it as "unanswered" would mislead).
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        chats = {c.chat_id for c in loose_ends.detect_unanswered_question(con, cfg)}
        self.assertNotIn(300, chats)

    def test_answered_late_fires_and_records_when_flag_set(self):
        # Opt-in NUDGE_QUESTION_INCLUDE_ANSWERED_LATE restores the "you replied
        # slowly" signal: Carol fires and the late reply is recorded.
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["question_include_answered_late"] = True
        carol = next(c for c in loose_ends.detect_unanswered_question(con, cfg)
                     if c.chat_id == 300)
        self.assertTrue(carol.evidence["answered_late"])
        self.assertEqual(carol.evidence["late_reply_days_later"], 10.0)
        self.assertEqual(carol.evidence["grace_hours"], 72.0)
        self.assertIn("late_reply_date", carol.evidence)

    def test_grace_floor_beats_a_tiny_p90(self):
        # With the dense history Carol's p90 is well under the 72h floor, so the
        # deadline is anchor + 72h (the floor), not anchor + p90.
        con = self._open()
        p90 = loose_ends._dyad_gap_p90(con, 300)
        self.assertIsNotNone(p90)
        self.assertLess(p90, 72 * 3600)

    def test_full_run_writes_only_the_unanswered_dyads(self):
        # Only Alice (100) is a genuinely open loose end; Carol (300) was
        # answered late and is resolved (suppressed by default).
        loose_ends.main(["--no-llm"])
        con = sqlite3.connect(self.db_path)
        try:
            rows = [r[0] for r in con.execute(
                "SELECT chat_id FROM loose_threads "
                "WHERE kind = 'unanswered_question' AND status = 'open' "
                "ORDER BY chat_id")]
        finally:
            con.close()
        self.assertEqual(rows, [100])


class RhetoricalQuestionTest(unittest.TestCase):
    def test_pleasantries_and_single_tokens_are_rhetorical(self):
        for q in ("How are you?", "how r u?", "hru?", "You know?", "Right?",
                  "ok?", "really?", "seriously?", "u up?", "What?", "Huh?",
                  "Hey! How are you?", "wanna bet?"):
            self.assertTrue(loose_ends._is_rhetorical_question(q), q)

    def test_genuine_questions_are_not_rhetorical(self):
        for q in ("Do you want to grab dinner this weekend?",
                  "Are we still on for lunch tomorrow?",
                  "What did you think of the movie we saw?",
                  "Can you send me the address when you get a chance?"):
            self.assertFalse(loose_ends._is_rhetorical_question(q), q)


class UnansweredQuestionConfigTest(unittest.TestCase):
    def test_knobs_are_named_config_vars(self):
        self.assertEqual(config.NUDGE_QUESTION_MIN_CHARS, 12)
        self.assertEqual(config.NUDGE_QUESTION_GRACE_HOURS, 72.0)

    def test_unanswered_question_is_registered(self):
        kinds = [k for k, _ in loose_ends.DETECTORS]
        self.assertIn("unanswered_question", kinds)

    def test_lexicon_build_question_test_is_untouched(self):
        # The DoD: N3 must NOT change lexicon_build's bare `"?" in text` question
        # test (it feeds person_period_lexicon.question_count / the LSM tools).
        import lexicon_build
        src = Path(lexicon_build.__file__).read_text()
        self.assertIn('if "?" in text:', src)


# ---------------------------------------------------------------------------
# dropped_plan detector (RES-57 / N4) -- flag- + precision-gated
# ---------------------------------------------------------------------------

def _build_dropped_plan_fixture(db_path):
    """One direct dyad (Alice) whose most-recent plan-cue message never gets
    followed through, plus a group chat carrying a cue message (direct-chat
    filter must drop it). Alice has an outbound message so Stage-0 never mutes
    her for zero outbound, and no negative llm valence."""
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (?, ?, ?)",
            [(1, "u1", "Alice"), (9, "u9", "Grace")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+10", 1), (90, "+90", 9)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, ?, 'iMessage')",
            [(100, "c100", 0), (500, "c500", 1)],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (500, 90)],
        )
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1000, "g1000", 100, 0, 1, "hey", "2026-05-01T10:00:00"),
                (1001, "g1001", 100, 10, 0,
                 "let's grab dinner next week", "2026-05-02T10:00:00"),   # anchor
                (1002, "g1002", 100, 0, 1, "sounds good", "2026-05-02T10:05:00"),
                (1003, "g1003", 100, 10, 0, "ok", "2026-05-03T10:00:00"),
                # Group chat cue message -- must be filtered by the direct-chat gate.
                (5000, "g5000", 500, 90, 0, "we should reschedule", "2026-05-04T10:00:00"),
            ],
        )
        con.commit()
    finally:
        con.close()


class _DroppedPlanFixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_dropped_plan_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def _cfg(self, con, *, enabled=True):
        cfg = loose_ends._build_cfg(con)
        cfg["dropped_plan_enabled"] = enabled
        return cfg

    def _pass_gate(self):
        con = sqlite3.connect(self.db_path)
        try:
            db.set_state(con, "eval_gate:dropped_plan",
                         json.dumps({"passed": ["dropped_plan"], "failed": []}))
            con.commit()
        finally:
            con.close()

    def _fail_gate(self):
        con = sqlite3.connect(self.db_path)
        try:
            db.set_state(con, "eval_gate:dropped_plan",
                         json.dumps({"passed": [], "failed": ["dropped_plan"]}))
            con.commit()
        finally:
            con.close()

    def _dropped_rows(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(
                "SELECT chat_id, evidence_json FROM loose_threads "
                "WHERE kind = 'dropped_plan' ORDER BY chat_id").fetchall()
        finally:
            con.close()


class DroppedPlanDetectorTest(_DroppedPlanFixtureTest):
    def test_disabled_flag_yields_nothing_and_calls_no_model(self):
        con = self._open()
        cfg = self._cfg(con, enabled=False)
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            self.assertEqual(list(loose_ends.detect_dropped_plan(con, cfg)), [])
        m.assert_not_called()

    def test_enabled_but_gate_absent_writes_nothing_and_skips_llm(self):
        # Flag ON, but no eval_gate:dropped_plan key and no seed -> suppressed
        # BEFORE any verification call.
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            self.assertEqual(list(loose_ends.detect_dropped_plan(con, cfg)), [])
        m.assert_not_called()

    def test_enabled_but_gate_failing_writes_nothing_and_skips_llm(self):
        self._fail_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            self.assertEqual(list(loose_ends.detect_dropped_plan(con, cfg)), [])
        m.assert_not_called()

    def test_unknown_verdict_suppresses(self):
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "UNKNOWN"}) as m:
            cands = list(loose_ends.detect_dropped_plan(con, cfg))
        self.assertEqual(cands, [])
        m.assert_called_once()   # the gate passed, so the verifier DID run

    def test_none_reply_suppresses(self):
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json", return_value=None):
            self.assertEqual(list(loose_ends.detect_dropped_plan(con, cfg)), [])

    def test_fulfilled_verdict_suppresses(self):
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "fulfilled"}):
            self.assertEqual(list(loose_ends.detect_dropped_plan(con, cfg)), [])

    def test_unfulfilled_fires_on_the_direct_dyad_only(self):
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "unfulfilled"}) as m:
            cands = {c.chat_id: c for c in loose_ends.detect_dropped_plan(con, cfg)}
        # Only the direct chat 100 fires; the group chat 500 is filtered out, so
        # the verifier is asked exactly once (one candidate chat).
        self.assertEqual(set(cands), {100})
        m.assert_called_once()
        alice = cands[100]
        self.assertEqual(alice.kind, "dropped_plan")
        self.assertEqual(alice.handle_id, 10)
        self.assertEqual(alice.anchor_message_id, 1001)
        self.assertEqual(alice.evidence["verdict"], "unfulfilled")
        self.assertTrue(alice.evidence["anchor_text"].startswith("let's grab"))

    def test_think_passed_through_per_tristate(self):
        # The verifier forwards think=cfg['nudges_think'] (ollama's tri-state).
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        cfg["nudges_think"] = False
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "unfulfilled"}) as m:
            list(loose_ends.detect_dropped_plan(con, cfg))
        self.assertEqual(m.call_args.kwargs.get("think"), False)


class DroppedPlanDriverTest(_DroppedPlanFixtureTest):
    def test_zero_rows_when_flag_off_through_the_driver(self):
        # Default flag (OFF) via _build_cfg: detect_all writes no dropped_plan
        # rows and never calls the verifier.
        con = self._open()
        cfg = loose_ends._build_cfg(con)   # dropped_plan_enabled defaults False
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        m.assert_not_called()
        self.assertEqual(self._dropped_rows(), [])

    def test_zero_rows_when_gate_absent_through_the_driver(self):
        con = self._open()
        cfg = self._cfg(con, enabled=True)   # flag ON, gate absent
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        m.assert_not_called()
        self.assertEqual(self._dropped_rows(), [])

    def test_row_written_when_enabled_gated_and_unfulfilled(self):
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "unfulfilled"}):
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        rows = self._dropped_rows()
        self.assertEqual([r["chat_id"] for r in rows], [100])
        ev = json.loads(rows[0]["evidence_json"])
        self.assertEqual(ev["verdict"], "unfulfilled")

    def test_dry_run_skips_the_verifier_model(self):
        # Even fully enabled + gated (where a normal run WOULD call the LLM and
        # write a row), --dry-run must call no model: dropped_plan is the only
        # model-backed detector, and the help text promises "without ... calling
        # the model". No dropped_plan rows are produced under dry-run.
        self._pass_gate()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json") as m:
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00", dry_run=True)
        m.assert_not_called()
        self.assertEqual(self._dropped_rows(), [])

    def _status_map(self):
        con = sqlite3.connect(self.db_path)
        try:
            return {(k, c): s for k, c, s in con.execute(
                "SELECT kind, chat_id, status FROM loose_threads")}
        finally:
            con.close()

    def test_degraded_verify_does_not_auto_resolve_dropped_plan(self):
        # Regression: a degraded verify (model unreachable -> None) must NOT let
        # reconciliation silently close a still-open dropped_plan thread. A stale
        # deterministic thread in the same run still resolves normally.
        self._pass_gate()
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
                    "VALUES ('dropped_plan', 100, 'open', '2020-01-01')")
        con.execute("INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
                    "VALUES ('faded_initiator', 555, 'open', '2020-01-01')")
        con.commit()
        con.close()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json", return_value=None):
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        status = self._status_map()
        # dropped_plan verify ran degraded -> its kind is exempted from auto-resolve.
        self.assertEqual(status[("dropped_plan", 100)], "open")
        # A deterministic detector's stale thread still resolves as before.
        self.assertEqual(status[("faded_initiator", 555)], "resolved")

    def test_clean_run_still_auto_resolves_dropped_plan(self):
        # The exemption fires ONLY on a degraded verify: a clean 'fulfilled'
        # verdict (no degraded flag) leaves dropped_plan reconcilable, so a stale
        # thread the run does not re-detect is resolved normally.
        self._pass_gate()
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
                    "VALUES ('dropped_plan', 100, 'open', '2020-01-01')")
        con.commit()
        con.close()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json",
                               return_value={"verdict": "fulfilled"}):
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        self.assertEqual(self._status_map()[("dropped_plan", 100)], "resolved")

    def _assert_unreachable_verify_degrades(self, exc):
        # finding #1 regression: an EXCEPTION from the verify backend (Ollama
        # down -> ConnectionRefusedError/OSError; model not pulled -> RuntimeError)
        # must degrade the pass exactly like a None verdict, NOT crash the whole
        # run. Before the fix the exception escaped detect_dropped_plan ->
        # detect_all's unguarded loop and aborted the run before its single
        # commit, discarding every deterministic detector's writes.
        self._pass_gate()
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
                    "VALUES ('dropped_plan', 100, 'open', '2020-01-01')")
        con.execute("INSERT INTO loose_threads(kind, chat_id, status, detected_at) "
                    "VALUES ('faded_initiator', 555, 'open', '2020-01-01')")
        con.commit()
        con.close()
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        with mock.patch.object(loose_ends.ollama, "chat_once_json", side_effect=exc):
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")  # must not raise
        self.assertTrue(cfg.get("dropped_plan_degraded"))
        status = self._status_map()
        # Degraded -> dropped_plan exempted from absence-based auto-resolve.
        self.assertEqual(status[("dropped_plan", 100)], "open")
        # A deterministic detector's stale thread still resolves normally.
        self.assertEqual(status[("faded_initiator", 555)], "resolved")

    def test_unreachable_verify_connection_error_degrades(self):
        # Ollama down: chat_once raises ConnectionRefusedError (an OSError).
        self._assert_unreachable_verify_degrades(
            ConnectionRefusedError("connection refused"))

    def test_unreachable_verify_runtime_error_degrades(self):
        # Model not pulled: ollama.chat_once raises RuntimeError("Ollama error: ...")
        # and chat_once_json re-raises it (not a think-rejection).
        self._assert_unreachable_verify_degrades(
            RuntimeError("Ollama error: model 'x' not found"))


class DroppedPlanConfigTest(unittest.TestCase):
    def test_pinned_config_defaults(self):
        # Guards the *shipped* defaults in config.py, not the developer's
        # effective config: these opt-in detectors may legitimately be enabled
        # in a local .env (e.g. NUDGE_GOODNEWS_ENABLED=true), so read the module
        # attrs live would make this test .env-dependent (non-hermetic). Reload
        # config with the relevant env vars cleared and .env loading suppressed
        # to observe the pinned defaults, then restore the real config.
        import importlib

        import dotenv
        keys = ("NUDGE_DROPPED_PLAN_ENABLED", "NUDGE_GOODNEWS_ENABLED",
                "NUDGE_PLAN_CONTEXT_MSGS")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            # Patch dotenv.load_dotenv at the source: reload re-runs config's
            # `from dotenv import load_dotenv`, so patching config's own attr
            # would be clobbered. The no-op keeps .env from re-injecting values.
            with mock.patch.object(dotenv, "load_dotenv", lambda *a, **k: False):
                importlib.reload(config)
            self.assertFalse(config.NUDGE_DROPPED_PLAN_ENABLED)
            self.assertFalse(config.NUDGE_GOODNEWS_ENABLED)
            self.assertEqual(config.NUDGE_PLAN_CONTEXT_MSGS, 30)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
            importlib.reload(config)  # re-reads .env, restores real values

    def test_both_n4_detectors_registered(self):
        kinds = [k for k, _ in loose_ends.DETECTORS]
        self.assertIn("dropped_plan", kinds)
        self.assertIn("unreciprocated_goodnews", kinds)

    def test_cue_phrases_are_a_module_frozenset(self):
        self.assertIsInstance(loose_ends._PLAN_CUE_PHRASES, frozenset)
        self.assertIn("we should", loose_ends._PLAN_CUE_PHRASES)


# ---------------------------------------------------------------------------
# unreciprocated_goodnews detector (RES-57 / N4) -- opt-in, llm-source, coverage
# ---------------------------------------------------------------------------

def _goodnews_events(response_class):
    return json.dumps([{"kind": "good_news", "excerpt": "I got the job!",
                        "response_class": response_class}])


def _build_goodnews_fixture(db_path):
    """Direct dyads exercising the source='llm' gate + coverage:
      Alice (100)  llm chunk, good news, passive_destructive  -> fires (coverage 0.5)
      Bob   (200)  DISTILLED chunk with (impossible) events    -> excluded by source
      Carol (300)  llm chunk, good news, active_constructive   -> reciprocated, no fire
    """
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (?, ?, ?)",
            [(1, "u1", "Alice"), (2, "u2", "Bob"), (3, "u3", "Carol")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+10", 1), (20, "+20", 2), (30, "+30", 3)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, 0, 'iMessage')",
            [(100, "c100"), (200, "c200"), (300, "c300")],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (200, 20), (300, 30)],
        )
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1000, "g1000", 100, 0, 1, "congrats", "2026-05-01T10:00:00"),
                (1001, "g1001", 100, 10, 0, "I got the job!", "2026-05-01T09:00:00"),
                (2000, "g2000", 200, 0, 1, "ok", "2026-05-01T10:00:00"),
                (2001, "g2001", 200, 20, 0, "news", "2026-05-02T10:00:00"),
                (3000, "g3000", 300, 0, 1, "amazing!!", "2026-05-01T10:00:00"),
                (3001, "g3001", 300, 30, 0, "I got engaged!", "2026-05-01T09:00:00"),
            ],
        )
        # Alice: one llm chunk (fires) + one distilled chunk (coverage denominator).
        con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, 2, 'x')",
            [(900, 100, 1000, 1001, "2026-05-01T08:00:00", "2026-05-01T10:00:00"),
             (901, 100, 1000, 1001, "2026-04-01T08:00:00", "2026-04-01T10:00:00"),
             (902, 200, 2000, 2001, "2026-05-01T08:00:00", "2026-05-02T10:00:00"),
             (903, 300, 3000, 3001, "2026-05-01T08:00:00", "2026-05-01T10:00:00")],
        )
        con.executemany(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (?, ?, '[]', 2, ?, ?)",
            [
                (900, 0.5, _goodnews_events("passive_destructive"), "llm"),  # Alice: fires
                (901, None, "[]", "distilled"),                              # coverage denom
                (902, 0.5, _goodnews_events("passive_destructive"), "distilled"),  # excluded
                (903, 0.5, _goodnews_events("active_constructive"), "llm"),  # reciprocated
            ],
        )
        con.commit()
    finally:
        con.close()


class _GoodnewsFixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_goodnews_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def _cfg(self, con, *, enabled=True):
        cfg = loose_ends._build_cfg(con)
        cfg["goodnews_enabled"] = enabled
        return cfg


class UnreciprocatedGoodnewsDetectorTest(_GoodnewsFixtureTest):
    def test_disabled_flag_yields_nothing(self):
        con = self._open()
        self.assertEqual(
            list(loose_ends.detect_unreciprocated_goodnews(
                con, self._cfg(con, enabled=False))), [])

    def test_fires_on_llm_source_only_with_coverage(self):
        con = self._open()
        cands = {c.chat_id: c for c in loose_ends.detect_unreciprocated_goodnews(
            con, self._cfg(con, enabled=True))}
        # Alice fires; Bob is distilled (excluded despite the event); Carol's
        # good news was reciprocated (active_constructive) so it does not fire.
        self.assertEqual(set(cands), {100})
        alice = cands[100]
        self.assertEqual(alice.kind, "unreciprocated_goodnews")
        self.assertEqual(alice.handle_id, 10)
        self.assertEqual(alice.anchor_message_id, 1001)
        self.assertEqual(alice.evidence["response_class"], "passive_destructive")
        self.assertTrue(alice.evidence["good_news_excerpt"].startswith("I got"))
        cov = alice.evidence["llm_annotation_coverage"]
        self.assertEqual(cov["total_chunks"], 2)
        self.assertEqual(cov["llm_annotated_chunks"], 1)
        self.assertEqual(cov["fraction"], 0.5)
        self.assertIn("precision_note", alice.evidence)

    def test_driver_off_by_default_writes_nothing(self):
        con = self._open()
        # Pin the opt-in flags OFF so the test asserts the *default* behavior
        # regardless of the developer's .env (which may set GOODNEWS=true).
        # _build_cfg reads these names from loose_ends' own namespace (imported
        # via `from config import ...`), so patch them there.
        with mock.patch.object(loose_ends, "NUDGE_GOODNEWS_ENABLED", False), \
                mock.patch.object(loose_ends, "NUDGE_DROPPED_PLAN_ENABLED", False):
            cfg = loose_ends._build_cfg(con)   # goodnews_enabled defaults False
            loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        n = sqlite3.connect(self.db_path).execute(
            "SELECT COUNT(*) FROM loose_threads WHERE kind = 'unreciprocated_goodnews'"
        ).fetchone()[0]
        self.assertEqual(n, 0)

    def test_driver_writes_row_when_enabled(self):
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        rows = sqlite3.connect(self.db_path).execute(
            "SELECT chat_id FROM loose_threads WHERE kind = 'unreciprocated_goodnews' "
            "AND status = 'open' ORDER BY chat_id").fetchall()
        self.assertEqual([r[0] for r in rows], [100])

    def test_null_response_class_does_not_fire(self):
        # An unclassified good-news event (response_class null) is 'unknown', not
        # 'under-received' -- it must NOT fire (this was a false-positive source
        # producing near-zero-urgency noise rows).
        self.assertEqual(loose_ends._unreciprocated_good_news(
            _goodnews_events(None)), [])

    def test_under_received_classes_fire(self):
        for cls in ("passive_constructive", "passive_destructive", "active_destructive"):
            self.assertEqual(
                len(loose_ends._unreciprocated_good_news(_goodnews_events(cls))), 1, cls)

    def test_reciprocated_class_does_not_fire(self):
        self.assertEqual(loose_ends._unreciprocated_good_news(
            _goodnews_events("active_constructive")), [])

    def test_recency_floor_suppresses_stale_goodnews(self):
        # Alice's good-news chunk is ~1 day before corpus-now; a 0-day floor
        # makes even that "stale", so the dyad is skipped entirely.
        con = self._open()
        cfg = self._cfg(con, enabled=True)
        cfg["goodnews_max_age_days"] = 0
        self.assertEqual(list(loose_ends.detect_unreciprocated_goodnews(con, cfg)), [])


# ---------------------------------------------------------------------------
# inferred_occasion detector (RES-58 / N5)
# ---------------------------------------------------------------------------

def _build_occasion_fixture(db_path):
    """Direct dyads for the inferred_occasion detector; corpus-now is Alice's
    recent 'hey' (the global max date). Every congratulatory message is
    outbound so no dyad trips the Stage-0 zero-outbound gate:
      Alice (100)  'Happy Birthday' on 03-15 across 2024/25/26 -> fires (3 yrs)
      Mia   (200)  'happy birthday' on 07-04 across 2024/25 but MUTED -> Stage-0
      Wes   (300)  'Happy anniversary' on 09-20 across 2024/25 -> fires (2 yrs)
      Bob   (400)  'happy birthday' once (2025-11-11) -> single year, no fire
      group (600)  recurring 01-05 but is_group = 1 -> direct-chat filter drops
    """
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (?, ?, ?)",
            [(1, "u1", "Alice"), (2, "u2", "Mia"), (3, "u3", "Wes"),
             (4, "u4", "Bob"), (5, "u5", "Grace")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(10, "+10", 1), (20, "+20", 2), (30, "+30", 3),
             (40, "+40", 4), (50, "+50", 5)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, ?, 'iMessage')",
            [(100, "c100", 0), (200, "c200", 0), (300, "c300", 0),
             (400, "c400", 0), (600, "c600", 1)],
        )
        con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(100, 10), (200, 20), (300, 30), (400, 40), (600, 50)],
        )
        # (chat_id, handle_id, is_from_me, text, date) rows; ids assigned below.
        spec = []
        for y in ("2024", "2025", "2026"):
            spec.append((100, 0, 1, "Happy Birthday!!", f"{y}-03-15T09:00:00"))
        spec.append((100, 0, 1, "hey", "2026-06-01T08:00:00"))
        for y in ("2024", "2025"):
            spec.append((200, 0, 1, "happy birthday", f"{y}-07-04T09:00:00"))
        for y in ("2024", "2025"):
            spec.append((300, 0, 1, "Happy anniversary love", f"{y}-09-20T09:00:00"))
        spec.append((400, 0, 1, "happy birthday", "2025-11-11T09:00:00"))
        for y in ("2024", "2025"):
            spec.append((600, 50, 0, "happy birthday", f"{y}-01-05T09:00:00"))
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(1000 + i, f"g{1000 + i}", c, h, me, t, d)
             for i, (c, h, me, t, d) in enumerate(spec)],
        )
        # Mia is muted before any run -> Stage-0 must suppress her occasion.
        con.execute(
            "INSERT INTO nudge_exclusions(handle_id, contact_id, reason, created_at) "
            "VALUES (20, 2, 'muted', '2026-01-01T00:00:00')"
        )
        con.commit()
    finally:
        con.close()


class _OccasionFixtureTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_occasion_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con


class InferredOccasionDetectorTest(_OccasionFixtureTest):
    def test_fires_on_multiyear_recurrence_only(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cands = {c.chat_id: c for c in loose_ends.detect_inferred_occasion(con, cfg)}
        # Recurring direct dyads fire (Stage-0 is applied by the driver, not the
        # detector, so muted Mia is still a candidate HERE); single-year Bob and
        # the group chat never do.
        self.assertEqual(set(cands), {100, 200, 300})
        self.assertNotIn(400, cands)   # single year
        self.assertNotIn(600, cands)   # group chat
        alice = cands[100]
        self.assertEqual(alice.kind, "inferred_occasion")
        self.assertEqual(alice.handle_id, 10)
        self.assertEqual(alice.evidence["month_day"], "03-15")
        self.assertEqual(alice.evidence["observed_years"], ["2024", "2025", "2026"])
        self.assertEqual(alice.evidence["n_years"], 3)

    def test_sets_due_by_to_next_upcoming_occurrence(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cands = {c.chat_id: c for c in loose_ends.detect_inferred_occasion(con, cfg)}
        # corpus-now is 2026-06-01: Alice's 03-15 has passed this year (-> 2027),
        # Wes's 09-20 is still ahead (-> 2026).
        self.assertEqual(cands[100].due_by, "2027-03-15")
        self.assertEqual(cands[300].due_by, "2026-09-20")

    def test_no_sql_date_functions_in_where(self):
        import inspect
        src = inspect.getsource(loose_ends.detect_inferred_occasion).lower()
        # The load-bearing rule: no date function ever wraps the date column.
        self.assertNotIn("strftime", src)
        self.assertNotIn("julianday", src)
        self.assertNotIn("date(m.date", src)
        # The month-day/year are sliced off the ISO string in Python instead.
        self.assertIn("d[5:10]", inspect.getsource(loose_ends.detect_inferred_occasion))

    def test_driver_persists_due_by_and_stage0_suppresses_muted(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        loose_ends.detect_all(con, cfg, "2026-07-01T00:00:00+00:00")
        rows = con.execute(
            "SELECT chat_id, due_by FROM loose_threads "
            "WHERE kind = 'inferred_occasion' AND status = 'open' ORDER BY chat_id"
        ).fetchall()
        # Muted Mia (200) is gone; Alice (100) and Wes (300) survive with due_by.
        self.assertEqual([r[0] for r in rows], [100, 300])
        self.assertEqual({r[0]: r[1] for r in rows},
                         {100: "2027-03-15", 300: "2026-09-20"})


class NextOccurrenceTest(unittest.TestCase):
    def test_rolls_forward_and_handles_leap_day(self):
        now = datetime.datetime(2026, 6, 1, 12, 0, 0)
        # A past date rolls to next year; a future date stays this year.
        self.assertEqual(loose_ends._next_occurrence(now, "03-15"), "2027-03-15")
        self.assertEqual(loose_ends._next_occurrence(now, "09-20"), "2026-09-20")
        # Feb 29 in the non-leap target year 2027 falls back to Feb 28.
        self.assertEqual(loose_ends._next_occurrence(now, "02-29"), "2027-02-28")


def _build_two_occasion_fixture(db_path):
    """A single direct dyad (chat 700, Zoe) that congratulates on TWO recurring
    dates -- a birthday (03-15) and an anniversary (09-20), each across two
    years -- so the detector must collapse them to one candidate."""
    con = sqlite3.connect(db_path)
    try:
        schema.create_schema(con)
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (6, 'u6', 'Zoe')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+70', 6, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (700, 'c700', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (700, 70)")
        spec = []
        for y in ("2024", "2025"):
            spec.append((700, 0, 1, "Happy Birthday!!", f"{y}-03-15T09:00:00"))
            spec.append((700, 0, 1, "Happy anniversary love", f"{y}-09-20T09:00:00"))
        spec.append((700, 0, 1, "hey", "2026-06-01T08:00:00"))  # corpus-now anchor
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(2000 + i, f"g{2000 + i}", c, 0 if me else 70, me, t, d)
             for i, (c, h, me, t, d) in enumerate(spec)],
        )
        con.commit()
    finally:
        con.close()


class InferredOccasionMultiOccasionTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        _build_two_occasion_fixture(self.db_path)
        for module in (loose_ends, query_tools):
            patcher = mock.patch.object(module, "DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def test_collapses_to_single_soonest_candidate(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        cfg["today"] = datetime.datetime(2026, 8, 1)  # 09-20 nearer than next 03-15
        cands = [c for c in loose_ends.detect_inferred_occasion(con, cfg)
                 if c.chat_id == 700]
        # Exactly one candidate despite two recurring occasions.
        self.assertEqual(len(cands), 1)
        cand = cands[0]
        # The soonest upcoming occasion (anniversary) is primary; the birthday
        # rides along in evidence rather than being silently overwritten.
        self.assertEqual(cand.evidence["month_day"], "09-20")
        self.assertEqual(cand.due_by, "2026-09-20")
        others = {o["month_day"]: o["next_occurrence"]
                  for o in cand.evidence["other_occasions"]}
        self.assertEqual(others, {"03-15": "2027-03-15"})

    def test_driver_persists_exactly_one_row(self):
        con = self._open()
        cfg = loose_ends._build_cfg(con)
        loose_ends.detect_all(con, cfg, "2026-08-01T00:00:00+00:00")
        rows = con.execute(
            "SELECT chat_id, due_by FROM loose_threads "
            "WHERE kind = 'inferred_occasion' AND status = 'open'"
        ).fetchall()
        self.assertEqual([(r[0], r[1]) for r in rows], [(700, "2026-09-20")])


class InferredOccasionUrgencyTest(unittest.TestCase):
    def _cfg(self, today):
        return {"kind_weights": config.NUDGE_KIND_WEIGHTS,
                "halflife_days": max(config.NUDGE_URGENCY_HALFLIFE_DAYS, 1e-9),
                "today": today}

    def test_proximity_beats_stale_anchor_and_far_dates(self):
        today = datetime.datetime(2026, 8, 1)
        cfg = self._cfg(today)
        anchor = "2025-09-20T09:00:00"  # ~10 months old -> would score ~0.0002 on age
        near = loose_ends._score_urgency("inferred_occasion", anchor, today, cfg,
                                         due_by="2026-08-08")   # 7 days out
        far = loose_ends._score_urgency("inferred_occasion", anchor, today, cfg,
                                        due_by="2027-05-01")    # ~9 months out
        # A due-soon occasion scores near full weight (0.8), far above the old
        # anchor-age floor, and ranks above a far-future one.
        self.assertGreater(near, 0.5)
        self.assertGreater(near, far)

    def test_non_occasion_kinds_keep_age_formula(self):
        # A kind without a due_by is unaffected: score decays on anchor age.
        today = datetime.datetime(2026, 8, 1)
        cfg = self._cfg(today)
        fresh = loose_ends._score_urgency("faded_initiator", "2026-07-31T09:00:00",
                                          today, cfg)
        stale = loose_ends._score_urgency("faded_initiator", "2026-01-01T09:00:00",
                                          today, cfg)
        self.assertGreater(fresh, stale)


class InferredOccasionConfigTest(unittest.TestCase):
    def test_knobs_and_weight(self):
        self.assertEqual(config.NUDGE_OCCASION_MIN_YEARS, 2)
        self.assertIn("inferred_occasion", config.NUDGE_KIND_WEIGHTS)

    def test_inferred_occasion_is_registered(self):
        kinds = [k for k, _ in loose_ends.DETECTORS]
        self.assertIn("inferred_occasion", kinds)


if __name__ == "__main__":
    unittest.main()
