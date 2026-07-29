"""Hermetic tests for the conflict core (RES-46 / C1): the deterministic
finder, the eval-gated label write path, the read-tool result caps, and the
schema migration + model guard. No Ollama (the label pass is a stub
classifier), no real chat.db -- everything runs against a temporary SQLite DB.

Run from the repo root:  python3 -m unittest tests.test_conflict
"""

import contextlib
import importlib
import io
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

import sqlite_vec

import conflict_detect
import lexicon_build
import ollama
import query_tools
import schema
import sessions
from db import get_state, iso_to_day, load_vec_extension, set_state
from eval.evalkit import gates, labeling
from query_tools import conflict as conflict_mod

# A stub label classifier: signature classify_fn(text) -> validated turn dict,
# the same shape conflict_detect._classify_turn returns. Confident 'criticism'
# so it clears the floor and (with the seed) the gate.
def _stub_criticism(text):
    return {"class": "criticism", "top2": ["criticism", "contempt"], "confidence": 0.9}


def _build_fixture(db_path, *, classify_fn, owner_verdict=None, distilled=False,
                   model="test-model", dv=False, build_lexicon=False):
    """Build a one-dyad corpus with an injected conflict episode (latency
    withdrawal + one-sided session + dislike-tapback run), then run the
    detector. Returns nothing -- the DB file is left populated and committed.

    dv=True swaps Session A's texts for DV power-and-control markers (a threat
    token+phrase, an isolation PHRASE-ONLY line, a fear token+phrase) WITHOUT
    changing ids/dates/structure, so detection is identical and only the C2 DV
    screen sees the difference. build_lexicon=True also runs the F1 lexicon
    build so the scorecard's lexicon/LSM sections have data."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    schema.create_schema(con)
    con.execute("INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
                "VALUES (2, 'uid-2', 'Alex', 'Rivers')")
    con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (50, '+15550001111', 2, 'iMessage')")
    con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (1, 'chat1', NULL, 0, 'iMessage')")
    con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (1, 50)")
    # Session Z (fires latency_withdrawal via the 25h inter-session gap that
    # answers my '...sure') then Session A a day later (one-sided burst from
    # them: fires one_sided_session; the dislike run fires dislike_run).
    a_texts = (
        ("if you ever leave i'll kill you",      # threat: token 'kill' + phrase
         "you can't see your friends anymore",   # isolation: PHRASE ONLY (no token)
         "i'm scared of you")                     # fear: token 'scared' + phrase
        if dv else
        ("you never listen to me",
         "seriously this is a pattern",
         "hello?? are you even there")
    )
    msgs = [
        (10, "g-10", 1, 50, 0, "hey can we talk", "2026-03-01T09:00:00"),
        (11, "g-11", 1, 0, 1, "sure", "2026-03-01T09:01:00"),
        (12, "g-12", 1, 50, 0, a_texts[0], "2026-03-02T10:00:00"),
        (13, "g-13", 1, 50, 0, a_texts[1], "2026-03-02T10:02:00"),
        (14, "g-14", 1, 50, 0, a_texts[2], "2026-03-02T10:04:00"),
    ]
    con.executemany(
        "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date, "
        "item_type, is_emote, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0)",
        msgs,
    )
    # Two DISLIKE tapbacks (2002 -- NOT 2003=laugh) inside session A's window.
    con.executemany(
        "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
        "is_from_me, associated_message_type, date) VALUES (?, ?, 1, 50, 1, 2002, ?)",
        [(200, "g-12", "2026-03-02T10:01:00"), (201, "g-13", "2026-03-02T10:03:00")],
    )
    if distilled:
        # Chunk overlapping the window with a source='distilled' annotation:
        # it must NOT count toward valence corroboration (llm-only).
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(1, 1, 12, 14, '2026-03-02T10:00:00', '2026-03-02T10:04:00', 3, 'x')")
        con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (1, -0.9, '[]', 3, '[]', 'distilled')")
    if owner_verdict is not None:
        from db import set_state
        set_state(con, gates.state_key("conflict_labels"), json.dumps(owner_verdict))
    con.commit()

    sessions.build_sessions(con)
    conflict_detect.detect_conflicts(con, model, classify_fn=classify_fn)
    if build_lexicon:
        lexicon_build.build_person_period_lexicon(con)
    con.commit()
    con.close()


class _TempDBTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        self._saved = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved)

    def _con(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con


# --- schema migration + model guard -----------------------------------------


class SchemaTest(unittest.TestCase):
    def test_migration_idempotent(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        schema.create_schema(con)
        schema.create_schema(con)  # re-run must be a no-op, not an error
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("conflicts", tables)
        self.assertIn("conflict_signals", tables)
        cols = {r[1] for r in con.execute("PRAGMA table_info(conflicts)")}
        for c in ("chat_id", "first_date", "last_date", "start_session_id",
                  "end_session_id", "topic", "peak_signal_json", "corroboration",
                  "off_text_flag", "source"):
            self.assertIn(c, cols)
        # No status/role/aggressor/who-won column leaked in from C2's scope.
        for forbidden in ("status", "my_role", "aggressor", "who_won", "winner"):
            self.assertNotIn(forbidden, cols)
        sig_cols = {r[1] for r in con.execute("PRAGMA table_info(conflict_signals)")}
        for c in ("conflict_id", "message_id", "actor_is_me", "label_class",
                  "top2_labels_json", "band", "excerpt"):
            self.assertIn(c, sig_cols)

    def test_model_guard_refuses_mismatch(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        schema.create_schema(con)
        con.execute("INSERT INTO conflicts(chat_id, first_date, last_date) "
                    "VALUES (1, '2026-01-01', '2026-01-02')")
        con.commit()
        conflict_detect._check_model_guard(con, "modelA")  # records modelA
        conflict_detect._check_model_guard(con, "modelA")  # same model: fine
        with self.assertRaises(SystemExit):
            conflict_detect._check_model_guard(con, "modelB")


# --- deterministic detection -------------------------------------------------


class WindowQualifiesTest(unittest.TestCase):
    """The independence gate: a window needs an active-tension signal, not just
    two correlated low-activity signals (one_sided_session + volume_collapse)."""

    def test_low_activity_pair_alone_is_rejected(self):
        # The exact pair that dominated the raw corpus (~40% of windows): a
        # quiet, one-directional lull double-counted. Must NOT qualify.
        self.assertFalse(
            conflict_detect._window_qualifies({"one_sided_session", "volume_collapse"}))

    def test_active_tension_plus_low_activity_qualifies(self):
        # Pin the min-signals floor so the test is hermetic w.r.t. .env.
        with mock.patch.object(conflict_detect, "CONFLICT_MIN_SIGNALS", 2):
            self.assertTrue(
                conflict_detect._window_qualifies({"latency_withdrawal", "one_sided_session"}))
            self.assertTrue(
                conflict_detect._window_qualifies({"dislike_run", "volume_collapse"}))

    def test_active_tension_signal_alone_is_below_min_signals(self):
        # One active-tension signal is real tension but still fewer than
        # CONFLICT_MIN_SIGNALS distinct signals -- both conditions must hold.
        with mock.patch.object(conflict_detect, "CONFLICT_MIN_SIGNALS", 2):
            self.assertFalse(conflict_detect._window_qualifies({"latency_withdrawal"}))

    def test_single_low_activity_signal_is_rejected(self):
        self.assertFalse(conflict_detect._window_qualifies({"one_sided_session"}))


class DetectionTest(_TempDBTest):
    def test_finds_window_with_injected_signals(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        con = self._con()
        rows = con.execute("SELECT * FROM conflicts").fetchall()
        self.assertEqual(len(rows), 1)
        c = rows[0]
        peak = json.loads(c["peak_signal_json"])
        # All three per-session deterministic signals fired in the merged window.
        self.assertIn("latency_withdrawal", peak["signals"])
        self.assertIn("one_sided_session", peak["signals"])
        self.assertIn("dislike_run", peak["signals"])
        self.assertGreaterEqual(peak["signal_count"], 2)
        # 25h inter-session gap > CONFLICT_OFF_TEXT_GAP_HOURS -> off_text_flag set.
        self.assertEqual(c["off_text_flag"], 1)
        # Spans the two sessions of the episode.
        self.assertIsNotNone(c["start_session_id"])
        self.assertIsNotNone(c["end_session_id"])
        # Dislike count is a real receipt.
        self.assertEqual(peak["dislike_count"], 2)

    def test_all_distilled_writes_corroboration_none_not_refuse(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism, distilled=True)
        con = self._con()
        rows = con.execute("SELECT corroboration FROM conflicts").fetchall()
        # Detection did NOT refuse on an all-distilled corpus...
        self.assertEqual(len(rows), 1)
        # ...and valence corroboration is 'none' (distilled rows don't count).
        self.assertEqual(rows[0]["corroboration"], "none")

    def test_reruns_are_idempotent(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        # Re-run the detector; the chat's conflicts are rebuilt, not duplicated.
        con = self._con()
        conflict_detect.detect_conflicts(con, "test-model", classify_fn=_stub_criticism)
        con.commit()
        self.assertEqual(con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0], 1)


# --- eval-gated label write path ---------------------------------------------


class LabelGateTest(_TempDBTest):
    def test_cleared_classes_write_with_bands(self):
        # No owner verdict -> the committed F4 seed clears the horsemen.
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        con = self._con()
        sig = con.execute(
            "SELECT label_class, band, excerpt FROM conflict_signals").fetchall()
        self.assertTrue(sig)
        for r in sig:
            self.assertEqual(r["label_class"], "criticism")  # a seed-cleared class
            self.assertIn(r["band"], ("tentative", "likely"))  # a band, never a float
            self.assertLessEqual(len(r["excerpt"]), 200)

    def test_suppressed_class_writes_zero_rows(self):
        # Owner verdict clearing NOTHING overrides the seed -> every predicted
        # class is suppressed (fixture-driven; no owner personal labels).
        owner = {"passed": [], "failed": list(gates.TASKS["conflict_labels"]["classes"]),
                 "metric": "kappa", "values": {}, "n_labels": 60,
                 "labeled_at": "T", "source": "owner"}
        _build_fixture(self.db_path, classify_fn=_stub_criticism, owner_verdict=owner)
        con = self._con()
        # The conflict itself is still found (deterministic)...
        self.assertEqual(con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0], 1)
        # ...but not one escalation label row was written.
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM conflict_signals").fetchone()[0], 0)

    def test_below_confidence_floor_is_dropped(self):
        low = lambda text: {"class": "contempt", "top2": ["contempt"], "confidence": 0.2}  # noqa: E731
        _build_fixture(self.db_path, classify_fn=low)
        con = self._con()
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM conflict_signals").fetchone()[0], 0)


# --- read-tool result caps ---------------------------------------------------


class ReadToolCapTest(_TempDBTest):
    def _seed_one_conflict(self, n_signals):
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.execute("INSERT INTO chats(id, is_group) VALUES (1, 0)")
        con.execute(
            "INSERT INTO conflicts(id, chat_id, first_date, last_date, "
            "peak_signal_json, corroboration, off_text_flag, source) VALUES "
            "(7, 1, '2026-03-01', '2026-03-02', ?, 'none', 0, 'llm')",
            (json.dumps({"signals": ["one_sided_session", "dislike_run"],
                         "signal_count": 2}),))
        con.executemany(
            "INSERT INTO conflict_signals(conflict_id, message_id, actor_is_me, "
            "label_class, top2_labels_json, band, excerpt) VALUES "
            "(7, ?, 0, 'criticism', '[\"criticism\"]', 'likely', ?)",
            [(1000 + i, "x" * 500) for i in range(n_signals)])
        con.commit()
        con.close()

    def test_get_conflict_caps_signal_rows_and_excerpts(self):
        self._seed_one_conflict(20)
        out = query_tools.get_conflict(7)
        self.assertLessEqual(len(out["signals"]), 15)         # <= 15 rows/page
        self.assertEqual(out["signals_total"], 20)
        self.assertTrue(out["truncated"])
        for s in out["signals"]:
            self.assertLessEqual(len(s["excerpt"]), 200)      # excerpt <= 200
        # Second page returns the rest.
        page2 = query_tools.get_conflict(7, offset=15)
        self.assertEqual(len(page2["signals"]), 5)
        self.assertFalse(page2["truncated"])

    def test_find_conflicts_returns_envelope(self):
        self._seed_one_conflict(3)
        env = query_tools.find_conflicts(chat_id=1)
        self.assertEqual(env["total_matching"], 1)
        self.assertEqual(env["rows"][0]["conflict_id"], 7)
        self.assertIn("signals", env["rows"][0])
        # labeled_signal_turns is folded from the conflict_signals count (all 3
        # seeded rows are non-abstain 'criticism').
        self.assertEqual(env["rows"][0]["labeled_signal_turns"], 3)

    def test_labeled_signal_turns_excludes_abstain(self):
        # The folded count (SUM CASE label_class != 'abstain') must exclude
        # abstain rows, and a conflict with zero signals must report 0 (LEFT JOIN).
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.execute("INSERT INTO chats(id, is_group) VALUES (1, 0)")
        con.execute(
            "INSERT INTO conflicts(id, chat_id, first_date, last_date, "
            "peak_signal_json, corroboration, off_text_flag, source) VALUES "
            "(7, 1, '2026-03-01', '2026-03-02', ?, 'none', 0, 'llm'), "
            "(8, 1, '2026-02-01', '2026-02-02', ?, 'none', 0, 'llm')",
            (json.dumps({"signals": ["dislike_run"], "signal_count": 1}),
             json.dumps({"signals": [], "signal_count": 0})))
        con.executemany(
            "INSERT INTO conflict_signals(conflict_id, message_id, actor_is_me, "
            "label_class, top2_labels_json, band, excerpt) VALUES "
            "(7, ?, 0, ?, '[]', 'likely', 'x')",
            [(1001, "criticism"), (1002, "abstain"), (1003, "contempt"),
             (1004, "abstain")])
        con.commit()
        con.close()
        env = query_tools.find_conflicts(chat_id=1)
        by_id = {r["conflict_id"]: r for r in env["rows"]}
        self.assertEqual(by_id[7]["labeled_signal_turns"], 2)  # criticism+contempt
        self.assertEqual(by_id[8]["labeled_signal_turns"], 0)  # no signals at all

    def test_read_tools_empty_when_no_tables(self):
        # A DB without the conflict tables must return empty/err, not raise.
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE chats(id INTEGER PRIMARY KEY, is_group INTEGER)")
        con.commit()
        con.close()
        self.assertEqual(query_tools.find_conflicts()["rows"], [])
        self.assertIn("error", query_tools.get_conflict(1))


class FindConflictsRankingTest(_TempDBTest):
    """find_conflicts surfaces the best-substantiated episode first:
    corroboration outranks raw signal_count."""

    def _insert(self, cid, corroboration, signal_count, last_date,
                topic=None, valence_boost=False):
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.execute("INSERT OR IGNORE INTO chats(id, is_group) VALUES (1, 0)")
        con.execute(
            "INSERT INTO conflicts(id, chat_id, first_date, last_date, topic, "
            "peak_signal_json, corroboration, off_text_flag, source) VALUES "
            "(?, 1, ?, ?, ?, ?, ?, 0, 'llm')",
            (cid, last_date, last_date, topic,
             json.dumps({"signals": ["s"] * signal_count, "signal_count": signal_count,
                         "valence_boost": valence_boost}),
             corroboration))
        con.commit()
        con.close()

    def test_corroborated_lower_signal_outranks_uncorroborated_higher(self):
        # id 1: more signals but no tone corroboration; id 2: fewer signals but
        # llm-corroborated. The corroborated one must rank first.
        self._insert(1, "none", 4, "2026-03-02")
        self._insert(2, "llm", 2, "2026-03-01")
        rows = query_tools.find_conflicts(chat_id=1)["rows"]
        self.assertEqual([r["conflict_id"] for r in rows], [2, 1])

    def test_within_same_corroboration_signal_count_then_recency(self):
        self._insert(1, "none", 2, "2026-03-01")
        self._insert(2, "none", 3, "2026-03-01")  # more signals -> first
        self._insert(3, "none", 2, "2026-03-05")  # same signals as 1, newer -> before 1
        rows = query_tools.find_conflicts(chat_id=1)["rows"]
        self.assertEqual([r["conflict_id"] for r in rows], [2, 3, 1])

    def test_reconciliation_deprioritized_within_tier(self):
        # F3: a repair/make-up window with MORE signals must NOT outrank a genuine
        # tense window at the same corroboration tier -- an apology is not the
        # "tensest" stretch. Both llm-corroborated; id 1 is a repair with 4
        # signals, id 2 a non-repair with 2. The non-reconciliation ranks first.
        self._insert(1, "llm", 4, "2026-03-02", topic="repair")
        self._insert(2, "llm", 2, "2026-03-01")
        rows = query_tools.find_conflicts(chat_id=1)["rows"]
        self.assertEqual([r["conflict_id"] for r in rows], [2, 1])
        self.assertTrue(rows[1]["reconciliation"])
        self.assertFalse(rows[0]["reconciliation"])

    def test_valence_boost_is_surfaced_but_not_reconciliation(self):
        # valence_boost flags a corroborated valence DROP (tension), NOT a
        # make-up -- it is surfaced for context but must not mark reconciliation.
        self._insert(1, "llm", 3, "2026-03-01", valence_boost=True)
        (row,) = query_tools.find_conflicts(chat_id=1)["rows"]
        self.assertTrue(row["valence_boost"])
        self.assertFalse(row["reconciliation"])

    def test_n_corroborated_counts_weak_and_llm(self):
        # F2: total_matching counts every episode; n_corroborated only weak+llm.
        self._insert(1, "none", 2, "2026-03-01")
        self._insert(2, "weak", 2, "2026-03-02")
        self._insert(3, "llm", 2, "2026-03-03")
        env = query_tools.find_conflicts(chat_id=1)
        self.assertEqual(env["total_matching"], 3)
        self.assertEqual(env["n_corroborated"], 2)

    def test_disclaimer_epilogue_present_only_when_nonempty(self):
        # F1: the humility disclaimer rides the generic _epilogue channel on a
        # non-empty result, and is absent when nothing was found.
        self.assertNotIn("_epilogue", query_tools.find_conflicts(chat_id=1))
        self._insert(1, "none", 2, "2026-03-01")
        env = query_tools.find_conflicts(chat_id=1)
        self.assertIn("_epilogue", env)
        self.assertIn("does not determine who was at fault", env["_epilogue"])


# --- F4 gate wiring ----------------------------------------------------------


class GateWiringTest(unittest.TestCase):
    def test_classifier_registered_as_lazy_dotted_path(self):
        ref = gates.TASKS["conflict_labels"]["classifier"]
        # A STRING, so importing gates never pulls in Ollama/conflict_detect.
        self.assertEqual(ref, "conflict_detect.gate_classifier")
        self.assertIsInstance(ref, str)
        # ...and it resolves to the real callable on the owner-run gate path.
        resolved = labeling._resolve_classifier(gates.TASKS["conflict_labels"])
        self.assertTrue(callable(resolved))
        self.assertIs(resolved, conflict_detect.gate_classifier)

    def test_label_classes_match_gate_task(self):
        self.assertEqual(set(conflict_detect.GATE_CLASSES),
                         set(gates.TASKS["conflict_labels"]["classes"]))


# --- C2: diagnose_conflict scorecard + softened DV-suppression floor ---------


def _all_keys(obj):
    """Every dict key anywhere in a nested result (for verdict-field checks)."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for it in obj:
            keys |= _all_keys(it)
    return keys


_VERDICT_KEYS = {"aggressor", "who_won", "whowon", "winner", "verdict",
                 "at_fault", "fault", "blame", "guilty"}


class DiagnoseScorecardTest(_TempDBTest):
    def _conflict_id(self):
        return self._con().execute("SELECT id FROM conflicts").fetchone()[0]

    def test_clean_fixture_full_scorecard_no_footer(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism, build_lexicon=True)
        out = query_tools.diagnose_conflict(self._conflict_id())
        self.assertEqual(out["dv_screen"], "clean")
        self.assertNotIn("resource_footer", out)  # clean fixture -> no footer
        # Deterministic sections all present.
        for k in ("lexicon", "lsm", "session_roles", "latency", "volume"):
            if k == "lsm":
                self.assertIn("lsm", out["lexicon"])
            else:
                self.assertIn(k, out)
        # Role read is present (not withheld) and is the mechanical pattern.
        self.assertNotIn("withheld", out["interaction_pattern"])
        self.assertIn("pursuer", out["interaction_pattern"])
        # No aggressor / who-won verdict field anywhere.
        self.assertEqual(_all_keys(out) & _VERDICT_KEYS, set())

    def test_lexicon_and_lsm_sections_compute(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism, build_lexicon=True)
        out = query_tools.diagnose_conflict(self._conflict_id())
        lex = out["lexicon"]
        self.assertTrue(lex["available"])
        for side in ("me", "them"):
            self.assertIsInstance(lex[side]["i_rate"], float)
            self.assertIsInstance(lex[side]["you_rate"], float)
            self.assertIsInstance(lex[side]["we_per_100_msgs"], float)
        self.assertIsInstance(lex["lsm"], float)  # style matching computed

    def test_result_capped_well_under_limit(self):
        from config import TOOL_RESULT_MAX_CHARS
        _build_fixture(self.db_path, classify_fn=_stub_criticism, build_lexicon=True)
        out = query_tools.diagnose_conflict(self._conflict_id())
        self.assertLess(len(json.dumps(out)), TOOL_RESULT_MAX_CHARS // 4)

    def test_dv_marker_role_suppress(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True,
                       build_lexicon=True)
        out = query_tools.diagnose_conflict(self._conflict_id())
        self.assertEqual(out["dv_screen"], "role-suppress")
        # Role/aggressor read withheld...
        self.assertTrue(out["interaction_pattern"].get("withheld"))
        self.assertNotIn("pursuer", out["interaction_pattern"])
        # ...but the mechanical scorecard STILL returns...
        for k in ("lexicon", "session_roles", "latency", "volume"):
            self.assertIn(k, out)
        self.assertTrue(out["lexicon"]["available"])
        # ...and the fixed resource footer is appended.
        self.assertIn("resource_footer", out)
        self.assertIn("1-800-799-7233", out["resource_footer"])
        self.assertIn("thehotline.org", out["resource_footer"])
        # Still no verdict field.
        self.assertEqual(_all_keys(out) & _VERDICT_KEYS, set())

    def test_dv_screen_helper_returns_status_with_all_categories(self):
        # incl. the isolation PHRASE-ONLY case ("you can't see ...", no token).
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True)
        cid = self._conflict_id()
        screen = conflict_mod._dv_screen(self._con(), cid)
        self.assertEqual(screen["status"], "role-suppress")  # not a bare bool
        self.assertTrue(screen["suppress_role"])
        self.assertEqual(set(screen["markers"]), {"threat", "isolation", "fear"})
        self.assertIsNotNone(screen["footer"])

    def test_dv_screen_full_text_not_excerpt(self):
        # The screen must read full message text, not the stored 200-char
        # excerpts: a marker only present past 200 chars must still fire.
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        cid = self._conflict_id()
        con = self._con()
        con.execute(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, item_type, is_emote, has_attachment) VALUES "
            "(15, 'g-15', 1, 50, 0, ?, '2026-03-02T10:03:30', 0, 0, 0)",
            (("padding " * 40) + "i'll kill you",))
        con.commit()
        screen = conflict_mod._dv_screen(self._con(), cid)
        self.assertIn("threat", screen["markers"])

    def test_clean_fixture_screen_clean_no_suppress(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        screen = conflict_mod._dv_screen(self._con(), self._conflict_id())
        self.assertEqual(screen["status"], "clean")
        self.assertFalse(screen["suppress_role"])
        self.assertIsNone(screen["footer"])
        self.assertEqual(screen["markers"], [])

    def test_config_off_bypass_restores_role_but_keeps_footer(self):
        import config
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True,
                       build_lexicon=True)
        cid = self._conflict_id()
        saved = config.CONFLICT_DV_SCREEN
        config.CONFLICT_DV_SCREEN = False
        self.addCleanup(setattr, config, "CONFLICT_DV_SCREEN", saved)
        out = query_tools.diagnose_conflict(cid)
        self.assertEqual(out["dv_screen"], "bypassed-with-footer")
        # Role analysis restored...
        self.assertNotIn("withheld", out["interaction_pattern"])
        self.assertIn("pursuer", out["interaction_pattern"])
        # ...but the footer is STILL appended on a marker hit.
        self.assertIn("resource_footer", out)

    def test_transient_flag_overrides_config_on(self):
        # Config default is 'on'; the transient per-run bypass must still win.
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True)
        cid = self._conflict_id()
        conflict_mod.set_dv_screen_bypass(True)
        self.addCleanup(conflict_mod.set_dv_screen_bypass, False)
        out = query_tools.diagnose_conflict(cid)
        self.assertEqual(out["dv_screen"], "bypassed-with-footer")
        self.assertNotIn("withheld", out["interaction_pattern"])
        self.assertIn("resource_footer", out)

    def test_diagnose_errors_gracefully(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        self.assertIn("error", query_tools.diagnose_conflict(999999))


class DiagnoseRegistrationTest(unittest.TestCase):
    def test_three_place_registration_in_sync(self):
        import chat
        import tool_schemas
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertIn("diagnose_conflict", names)          # schema
        self.assertIn("diagnose_conflict", chat.TOOL_FUNCTIONS)  # dispatch
        self.assertTrue(callable(query_tools.diagnose_conflict))  # re-export
        self.assertIn("diagnose_conflict", chat.SYSTEM_PROMPT)    # prompt line


# --- C3 (RES-52): gated cross-chat conflict trace ----------------------------


def _build_trace_fixture(db_path):
    """A corpus for trace_conflict_across_chats: origin chat 1 (partner Jordan,
    handle 50) holds conflict 7; chat 2 (another chat) holds two same-day chunks
    -- one that NAMES Jordan, one that names 'my boss' (no name) -- both with a
    vector identical to the query (cosine sim 1.0). Chat 3 holds a Jordan-naming
    chunk with an ORTHOGONAL vector (sim 0.0, below threshold). The origin chat
    itself holds a Jordan-naming, high-sim chunk that must be EXCLUDED (the trace
    is cross-chat). All vectors are hand-planted; embed_texts is stubbed to the
    query vector [1,0,0]."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    load_vec_extension(con)
    schema.create_schema(con)
    con.execute("INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
                "VALUES (2, 'uid-2', 'Jordan', 'Lee')")
    con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (50, '+15550001111', 2, 'iMessage')")
    for cid, ident in ((1, 'origin'), (2, 'other-B'), (3, 'other-C')):
        con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                    "VALUES (?, ?, 0, 'iMessage')", (cid, ident))
    con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (1, 50)")
    # The episode's own message (becomes the trace query text).
    con.execute(
        "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date, "
        "item_type, is_emote, has_attachment) VALUES "
        "(10, 'g-10', 1, 50, 0, 'you never listen to me this weekend', "
        "'2026-03-02T10:00:00', 0, 0, 0)")
    con.execute("INSERT INTO conflicts(id, chat_id, first_date, last_date) VALUES "
                "(7, 1, '2026-03-02T10:00:00', '2026-03-02T10:04:00')")
    # (chunk_id, chat_id, text, vector) -- day '2026-03-02', inside the ±2d window.
    rows = [
        (100, 2, 'ranting about the weekend with Jordan again', [1.0, 0.0, 0.0]),
        (101, 2, 'the weekend with my boss was rough too',      [1.0, 0.0, 0.0]),
        (102, 3, 'Jordan came up but this is unrelated',        [0.0, 1.0, 0.0]),
        (103, 1, 'still upset at Jordan about the weekend',     [1.0, 0.0, 0.0]),
    ]
    for cid, chat_id, text, _vec in rows:
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(?, ?, ?, ?, '2026-03-02T09:00:00', '2026-03-02T09:05:00', 1, ?)",
            (cid, chat_id, cid, cid, text))
    con.execute(
        "CREATE VIRTUAL TABLE chunk_vectors USING vec0("
        "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[3] distance_metric=cosine, "
        "chat_id INTEGER, first_day INTEGER, last_day INTEGER)")
    day = iso_to_day('2026-03-02')
    for cid, chat_id, _text, vec in rows:
        con.execute(
            "INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, "
            "last_day) VALUES (?, ?, ?, ?, ?)",
            (cid, sqlite_vec.serialize_float32(vec), chat_id, day, day))
    con.commit()
    con.close()


class TraceAcrossChatsTest(_TempDBTest):
    def setUp(self):
        super().setUp()
        _build_trace_fixture(self.db_path)
        patcher = mock.patch.object(
            query_tools, "embed_texts",
            lambda texts, kind="document": [[1.0, 0.0, 0.0]] * len(texts))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_consent_required_returns_ask_no_data(self):
        # Without consent the tool returns a fixed ask-the-user payload -- no
        # embedding, no query, no links (consent is the UX layer).
        out = query_tools.trace_conflict_across_chats(7)
        self.assertEqual(out["status"], "consent_required")
        self.assertEqual(out["links"], [])
        self.assertIn("ask", out)
        self.assertNotIn("origin_chat_id", out)  # no data was read

    def test_name_reference_gate_filters_unnamed(self):
        # The acceptance case: two topically-identical, equally-similar chunks in
        # chat 2 -- only the one that literally NAMES Jordan links; "weekend with
        # my boss" is dropped despite sim 1.0.
        out = query_tools.trace_conflict_across_chats(7, consent=True)
        linked = [ln["chunk_id"] for ln in out["links"]]
        self.assertIn(100, linked)          # names Jordan
        self.assertNotIn(101, linked)       # "my boss" -- no name reference
        by_id = {ln["chunk_id"]: ln for ln in out["links"]}
        self.assertEqual(by_id[100]["matched_reference"]["value"], "jordan")
        self.assertEqual(by_id[100]["matched_reference"]["kind"], "name")

    def test_excludes_origin_chat(self):
        # Chunk 103 is in the origin chat, names Jordan, and is sim 1.0 -- but the
        # trace is cross-chat by construction, so it must never link.
        out = query_tools.trace_conflict_across_chats(7, consent=True)
        self.assertEqual(out["origin_chat_id"], 1)
        self.assertNotIn(103, [ln["chunk_id"] for ln in out["links"]])
        for ln in out["links"]:
            self.assertNotEqual(ln["chat_id"], 1)

    def test_similarity_threshold_filters_low_sim(self):
        # Chunk 102 names Jordan but is orthogonal (sim 0.0 < 0.65) -- filtered.
        out = query_tools.trace_conflict_across_chats(7, consent=True)
        self.assertNotIn(102, [ln["chunk_id"] for ln in out["links"]])
        for ln in out["links"]:
            self.assertGreater(ln["similarity"], out["sim_threshold"])

    def _vec_con(self):
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        con.row_factory = sqlite3.Row
        load_vec_extension(con)   # chunk_vectors is a vec0 virtual table
        return con

    def test_nothing_persisted(self):
        counts = ("conflicts", "chunks", "chunk_vectors")
        con = self._vec_con()
        before = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in counts}
        query_tools.trace_conflict_across_chats(7, consent=True)
        con2 = self._vec_con()
        tables = {r[0] for r in con2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("conflict_links", tables)   # no such table exists
        after = {t: con2.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in counts}
        self.assertEqual(before, after)              # trace wrote nothing

    def test_missing_conflict_errors_gracefully(self):
        out = query_tools.trace_conflict_across_chats(999999, consent=True)
        self.assertIn("error", out)
        self.assertEqual(out["links"], [])

    def test_chunk_knn_exclude_chat_ids_drops_whole_chat(self):
        # Direct _chunk_knn check: exclude_chat_ids removes a chat in-scan, and
        # leaving it unset is a no-op vs the existing callers.
        from query_tools.chunks import _chunk_knn
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        con.row_factory = sqlite3.Row
        load_vec_extension(con)
        qvec = sqlite_vec.serialize_float32([1.0, 0.0, 0.0])
        base = _chunk_knn(con, qvec, 10, None, None, None)
        base_ids = {r["chunk_id"] for r in base}
        self.assertIn(103, base_ids)   # origin chunk present without exclusion
        excl = _chunk_knn(con, qvec, 10, None, None, None, exclude_chat_ids=[1])
        excl_ids = {r["chunk_id"] for r in excl}
        self.assertNotIn(103, excl_ids)                 # chat 1 dropped in-scan
        self.assertEqual(excl_ids, base_ids - {103})    # only chat 1 removed
        for r in excl:
            self.assertNotEqual(r["chat_id"], 1)


class TraceRegistrationGateTest(unittest.TestCase):
    """The trace schema is registered ONLY when CONFLICT_TRACE_ENABLED; that
    registration gate -- not the per-call consent param -- is the security
    boundary. Reloads tool_schemas + chat with the flag toggled (mirrors
    test_drift_tools.RegistrationGateTest so it is robust under full discover:
    the live config module is used, and the flag is restored in cleanup)."""

    NAME = "trace_conflict_across_chats"

    def setUp(self):
        self._saved = importlib.import_module("config").CONFLICT_TRACE_ENABLED
        self.addCleanup(self._reload, self._saved)

    @staticmethod
    def _reload(enabled):
        import chat
        import tool_schemas
        importlib.import_module("config").CONFLICT_TRACE_ENABLED = enabled
        importlib.reload(tool_schemas)
        importlib.reload(chat)

    def test_disabled_is_default_and_registers_nothing(self):
        # Off out of the box: the function still imports, but no schema/dispatch/
        # prompt line exists, so the model can never call it.
        self._reload(False)
        import chat
        import tool_schemas
        self.assertFalse(importlib.import_module("config").CONFLICT_TRACE_ENABLED)
        self.assertTrue(callable(query_tools.trace_conflict_across_chats))  # importable
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertNotIn(self.NAME, names)
        self.assertNotIn(self.NAME, chat.TOOL_FUNCTIONS)
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertNotIn(self.NAME, ollama_names)
        self.assertNotIn(self.NAME, chat.SYSTEM_PROMPT)

    def test_enabled_registers_schema_dispatch_and_prompt(self):
        self._reload(True)
        import chat
        import tool_schemas
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertIn(self.NAME, names)
        self.assertIn(self.NAME, chat.TOOL_FUNCTIONS)
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertIn(self.NAME, ollama_names)
        self.assertIn(self.NAME, chat.SYSTEM_PROMPT)


# --- C4 (RES-53): draft_apology scaffold (Lewicki-6, never ghostwrites) -------


class DraftApologyTest(_TempDBTest):
    def _conflict_id(self):
        return self._con().execute("SELECT id FROM conflicts").fetchone()[0]

    def test_clean_returns_checklist_excerpts_no_footer(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertEqual(out["dv_screen"], "clean")
        self.assertNotIn("resource_footer", out)          # clean -> no footer
        self.assertNotIn("withheld", out)                 # scaffold NOT withheld
        # Emits no finished apology prose.
        self.assertFalse(out["emits_prose"])
        # Ranked Lewicki-6: acknowledge highest, offer-repair second, forgiveness
        # lowest and flagged skippable.
        ranks = {c["component"]: c["rank"] for c in out["checklist"]}
        self.assertEqual(ranks["acknowledge_responsibility"], 1)
        self.assertEqual(ranks["offer_of_repair"], 2)
        forg = next(c for c in out["checklist"]
                    if c["component"] == "request_for_forgiveness")
        self.assertEqual(forg["rank"], 6)
        self.assertTrue(forg["skippable"])
        # REAL grievance excerpts (verbatim, capped) are surfaced.
        self.assertTrue(out["grievances"])
        for g in out["grievances"]:
            self.assertLessEqual(len(g["excerpt"]), 200)
            self.assertIn(g["actor"], ("me", "them"))

    def test_repair_component_present_when_cleared(self):
        # Seed clears `repair` out of the box -> the component is present.
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertEqual(out["repair_attempts"]["status"], "available")
        self.assertIn("attempts", out["repair_attempts"])

    def test_repair_attempts_surface_when_present(self):
        # A repair-labeled corpus (them's repair bids) with `repair` cleared:
        # the ignored repair attempts surface verbatim, attributed to `them`.
        repair = lambda t: {"class": "repair", "top2": ["repair"], "confidence": 0.9}  # noqa: E731
        _build_fixture(self.db_path, classify_fn=repair)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertEqual(out["repair_attempts"]["status"], "available")
        attempts = out["repair_attempts"]["attempts"]
        self.assertTrue(attempts)
        for a in attempts:
            self.assertEqual(a["actor"], "them")
            self.assertLessEqual(len(a["excerpt"]), 200)

    def test_repair_unavailable_when_not_eval_cleared(self):
        # Owner verdict clearing `criticism` but NOT `repair` -> the repair
        # component is OMITTED and marked unavailable; checklist + excerpts stay.
        owner = {"passed": ["criticism"],
                 "failed": ["contempt", "defensiveness", "stonewalling", "repair"],
                 "metric": "kappa", "values": {}, "n_labels": 60,
                 "labeled_at": "T", "source": "owner"}
        _build_fixture(self.db_path, classify_fn=_stub_criticism, owner_verdict=owner)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertEqual(out["repair_attempts"]["status"], "unavailable")
        self.assertEqual(out["repair_attempts"]["reason"], "label class not eval-cleared")
        self.assertNotIn("attempts", out["repair_attempts"])  # never guessed
        # The checklist + real excerpts still return.
        self.assertIn("checklist", out)
        self.assertTrue(out["grievances"])

    def test_dv_role_suppress_withholds_scaffold_entirely(self):
        # High-risk DV fixture, screening ON -> _dv_screen role-suppress. UNLIKE
        # diagnose_conflict there is NO partial scaffold: the whole thing is
        # withheld and ONLY the resource pointer returns.
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True,
                       build_lexicon=True)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertEqual(out["dv_screen"], "role-suppress")
        self.assertTrue(out["withheld"])
        for k in ("checklist", "grievances", "repair_attempts"):
            self.assertNotIn(k, out)                  # scaffold withheld ENTIRELY
        self.assertIn("resource_footer", out)         # pointer ONLY
        self.assertIn("1-800-799-7233", out["resource_footer"])
        self.assertIn("thehotline.org", out["resource_footer"])

    def test_dv_bypass_returns_scaffold_with_footer(self):
        # Screening bypassed -> bypassed-with-footer: full checklist + excerpts
        # return WITH the resource footer appended (pointer never fully removed).
        import config
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True)
        cid = self._conflict_id()
        saved = config.CONFLICT_DV_SCREEN
        config.CONFLICT_DV_SCREEN = False
        self.addCleanup(setattr, config, "CONFLICT_DV_SCREEN", saved)
        out = query_tools.draft_apology(cid)
        self.assertEqual(out["dv_screen"], "bypassed-with-footer")
        self.assertNotIn("withheld", out)
        self.assertIn("checklist", out)               # scaffold returns
        self.assertIn("grievances", out)
        self.assertIn("resource_footer", out)         # footer STILL appended
        self.assertIn("1-800-799-7233", out["resource_footer"])

    def test_transient_bypass_overrides_config_on(self):
        # Config default 'on'; the transient per-run bypass still wins.
        _build_fixture(self.db_path, classify_fn=_stub_criticism, dv=True)
        cid = self._conflict_id()
        conflict_mod.set_dv_screen_bypass(True)
        self.addCleanup(conflict_mod.set_dv_screen_bypass, False)
        out = query_tools.draft_apology(cid)
        self.assertEqual(out["dv_screen"], "bypassed-with-footer")
        self.assertIn("checklist", out)
        self.assertIn("resource_footer", out)

    def test_never_emits_finished_prose(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertIs(out["emits_prose"], False)
        # No verdict/blame field leaks in either.
        self.assertEqual(_all_keys(out) & _VERDICT_KEYS, set())

    def test_result_capped_well_under_limit(self):
        from config import TOOL_RESULT_MAX_CHARS
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        out = query_tools.draft_apology(self._conflict_id())
        self.assertLess(len(json.dumps(out)), TOOL_RESULT_MAX_CHARS // 4)

    def test_errors_gracefully_on_missing_conflict(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        self.assertIn("error", query_tools.draft_apology(999999))

    def test_empty_when_no_conflict_tables(self):
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE chats(id INTEGER PRIMARY KEY, is_group INTEGER)")
        con.commit()
        con.close()
        self.assertIn("error", query_tools.draft_apology(1))


class DraftApologyRegistrationTest(unittest.TestCase):
    def test_three_place_registration_in_sync(self):
        import chat
        import tool_schemas
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertIn("draft_apology", names)                     # schema
        self.assertIn("draft_apology", chat.TOOL_FUNCTIONS)       # dispatch
        self.assertTrue(callable(query_tools.draft_apology))      # re-export
        self.assertIn("draft_apology", chat.SYSTEM_PROMPT)        # prompt line
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertIn("draft_apology", ollama_names)


# --- the wall-clock budgeted labeling tier -----------------------------------
#
# Everything below covers the budgeted second pass (conflict_detect's
# label_top_conflicts + the pieces it rides on). The clock is faked the way the
# tier actually spends it -- FakeClock only moves when a (fake) LLM request
# charges it -- so the deadline tests assert the dispatch RULE instead of racing
# a real wall clock. label_top_conflicts takes an injected `monotonic` (and an
# optional `rate`), so nothing here has to patch a module's time.


class _FakeClock:
    """time.monotonic() that only advances when something charges it."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _stub_neutral(text):
    """Every turn comes back non-escalating: a window the labeler DID process
    and found nothing in (which must still count as processed)."""
    return {"class": "neutral", "top2": ["neutral"], "confidence": 0.9}


class _Classifier:
    """A recording classify_fn stub. Charges the fake clock what a real request
    would have cost; `seconds` may be a list (per call, last value repeating) so
    a test can make one call overrun the banked rate -- which is what makes the
    MID-window deadline check fire rather than the between-windows one."""

    def __init__(self, clock=None, seconds=0.0, verdict=_stub_criticism):
        self.clock = clock
        self.seconds = list(seconds) if isinstance(seconds, (list, tuple)) else [seconds]
        self.verdict = verdict
        self.texts = []

    def __call__(self, text):
        charge = self.seconds[min(len(self.texts), len(self.seconds) - 1)]
        self.texts.append(text)
        if self.clock is not None and charge:
            self.clock.advance(charge)
        return self.verdict(text)

    @property
    def ids(self):
        """Message ids behind the recorded turns (a seeded turn's text ends in
        `m<message id>`), so a test can say WHICH windows were dispatched."""
        return [int(t.rsplit("m", 1)[1]) for t in self.texts]


def _warm_rate(prior):
    """A _TurnRate that has already paid the model's cold load, so the deadline
    arithmetic in a test is a flat prior * _TURN_RATE_SAFETY per call."""
    rate = conflict_detect._TurnRate(prior=prior)
    rate.record(0.0)          # the cold-load call: charged, never counted
    return rate


def _said(printed):
    """Everything a mock.patch('builtins.print') saw, as one string."""
    return " ".join(str(a) for call in printed.call_args_list for a in call.args)


# --- think resolution --------------------------------------------------------


class ResolveThinkTest(unittest.TestCase):
    """An UNSET CONFLICT_THINK is None at the config seam = "omit the flag",
    which on a thinking model buys a full reasoning pass per turn (197s vs
    2.2s measured). The pipeline default is resolved away from that; the
    tri-state itself is untouched, so an explicit value still passes through."""

    def test_unset_resolves_to_false_on_a_model_without_effort_levels(self):
        with mock.patch.object(conflict_detect, "CONFLICT_THINK", None):
            self.assertIs(conflict_detect._resolve_think("qwen3.5:4b"), False)
            self.assertIs(conflict_detect._resolve_think("gemma4:12b"), False)

    def test_unset_resolves_to_an_effort_level_on_the_gpt_oss_family(self):
        self.assertTrue(conflict_detect.has_effort_levels("gpt-oss:20b"))
        with mock.patch.object(conflict_detect, "CONFLICT_THINK", None):
            self.assertEqual(conflict_detect._resolve_think("gpt-oss:20b"), "low")

    def test_an_explicit_value_passes_through_untouched(self):
        for explicit in (True, False, "high"):
            with mock.patch.object(conflict_detect, "CONFLICT_THINK", explicit):
                self.assertEqual(conflict_detect._resolve_think("qwen3.5:4b"), explicit)
                # ...on an effort model too: the env always wins over the resolver.
                self.assertEqual(conflict_detect._resolve_think("gpt-oss:20b"), explicit)


class ThinkReachesTheRequestTest(_TempDBTest):
    """Resolving is only half of it -- both write paths have to SEND the
    resolved value. A request carrying think=None is the 90x footgun."""

    def _recorder(self):
        calls = []

        def fake_chat_once_json(messages, model=None, think=None, timeout=None,
                                retry_nudge=None):
            calls.append({"model": model, "think": think})
            return {"class": "criticism", "top2": ["criticism"], "confidence": 0.9}

        return calls, fake_chat_once_json

    def _unlabeled_fixture(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        con = self._con()
        con.execute("UPDATE conflicts SET labeled_at = NULL")
        con.commit()
        return con

    def test_detect_conflicts_sends_think_false_not_omitted(self):
        con = self._unlabeled_fixture()
        calls, fake = self._recorder()
        with mock.patch.object(conflict_detect, "CONFLICT_THINK", None), \
                mock.patch.object(ollama, "chat_once_json", fake), \
                mock.patch("builtins.print"):
            conflict_detect.detect_conflicts(con, "test-model")
        self.assertTrue(calls)
        for call in calls:
            self.assertIs(call["think"], False)

    def test_label_top_conflicts_sends_think_false_not_omitted(self):
        con = self._unlabeled_fixture()
        calls, fake = self._recorder()
        with mock.patch.object(conflict_detect, "CONFLICT_THINK", None), \
                mock.patch.object(ollama, "chat_once_json", fake), \
                mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model")
        self.assertTrue(calls)
        for call in calls:
            self.assertIs(call["think"], False)


# --- turn selection ----------------------------------------------------------


_LONG_TURN = "you keep bringing this up every single time and i am done"


def _turn(mid, text=_LONG_TURN):
    """One row of the shape _select_turns reads (a dict indexes like the
    sqlite3.Row the real caller passes)."""
    return {"id": mid, "is_from_me": mid % 2, "text": text}


class SelectTurnsTest(unittest.TestCase):
    def setUp(self):
        # Pin the prescreen threshold so the test is hermetic w.r.t. .env.
        patcher = mock.patch.object(conflict_detect, "CONFLICT_MIN_TURN_CHARS", 15)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _ids(out):
        return [t[0] for t in out]

    def test_span_brackets_the_peak_not_the_window_head(self):
        # 20 candidates, cap 6: the head slice would return 1..6 and provably
        # miss the escalation moment the finder keyed on.
        seg = [_turn(i) for i in range(1, 21)]
        out = conflict_detect._select_turns(seg, 6, peak_id=15)
        self.assertEqual(self._ids(out), [12, 13, 14, 15, 16, 17])

    def test_output_stays_chronological(self):
        seg = [_turn(i) for i in range(1, 21)]
        ids = self._ids(conflict_detect._select_turns(seg, 6, peak_id=15))
        self.assertEqual(ids, sorted(ids))          # never emitted peak-outward

    def test_a_peak_at_either_edge_clamps_the_span_inside_the_window(self):
        seg = [_turn(i) for i in range(1, 21)]
        self.assertEqual(
            self._ids(conflict_detect._select_turns(seg, 6, peak_id=20)),
            [15, 16, 17, 18, 19, 20])
        self.assertEqual(
            self._ids(conflict_detect._select_turns(seg, 6, peak_id=1)),
            [1, 2, 3, 4, 5, 6])

    def test_prescreen_drops_short_and_symbol_only_turns(self):
        seg = [_turn(1, "ok"),                      # under the char floor
               _turn(2),
               _turn(3, "\U0001f643" * 20),         # long enough, no alnum
               _turn(4, "!" * 20),                  # punctuation only
               _turn(5, "   "),                     # whitespace only
               _turn(6)]
        self.assertEqual(self._ids(conflict_detect._select_turns(seg, 6)), [2, 6])

    def test_no_peak_id_falls_back_to_the_chronological_head(self):
        seg = [_turn(i) for i in range(1, 21)]
        self.assertEqual(self._ids(conflict_detect._select_turns(seg, 6)),
                         [1, 2, 3, 4, 5, 6])

    def test_a_peak_id_absent_from_the_segment_falls_back_to_the_head(self):
        seg = [_turn(i) for i in range(1, 21)]
        self.assertEqual(self._ids(conflict_detect._select_turns(seg, 6, peak_id=999)),
                         [1, 2, 3, 4, 5, 6])

    def test_a_prescreened_out_peak_still_anchors_on_its_position(self):
        # A long silence is often answered by a one-word line, so the peak turn
        # itself is frequently the one the prescreen drops. The span must still
        # land WHERE it was rather than reverting to the window head.
        seg = [_turn(i) for i in range(1, 21)]
        seg[14] = _turn(15, "ok")                   # id 15, dropped by the prescreen
        out = self._ids(conflict_detect._select_turns(seg, 6, peak_id=15))
        self.assertEqual(out, [12, 13, 14, 16, 17, 18])
        self.assertNotIn(15, out)

    def test_cap_is_respected(self):
        seg = [_turn(i) for i in range(1, 51)]
        for cap in (1, 3, 6, 12):
            self.assertEqual(len(conflict_detect._select_turns(seg, cap, peak_id=25)),
                             cap)

    def test_non_positive_cap_selects_nothing(self):
        seg = [_turn(i) for i in range(1, 21)]
        self.assertEqual(conflict_detect._select_turns(seg, 0, peak_id=15), [])
        self.assertEqual(conflict_detect._select_turns(seg, -3, peak_id=15), [])

    def test_fewer_candidates_than_the_cap_returns_them_all(self):
        seg = [_turn(1), _turn(2), _turn(3)]
        self.assertEqual(self._ids(conflict_detect._select_turns(seg, 6, peak_id=2)),
                         [1, 2, 3])
        # A None cap is "no cap", not "no turns".
        self.assertEqual(self._ids(conflict_detect._select_turns(seg, None)), [1, 2, 3])


# --- the budgeted labeling loop ----------------------------------------------


# Every seeded turn is comfortably over CONFLICT_MIN_TURN_CHARS, so the
# prescreen never silently changes a budget test's call count.
_LABELABLE = "you keep bringing this up every single time"


class _ConflictWindowsTest(_TempDBTest):
    """A DB of already-DETECTED windows (no detection run): each window owns
    n_turns labelable messages inside its own day, so the labeling loop's work
    per window -- and therefore the deadline arithmetic -- is exact."""

    def _seed(self, windows, model="test-model"):
        """windows: [(conflict_id, n_turns, signal_count)]. signal_count drives
        _conflict_strength, i.e. the order the loop labels them in. Message ids
        are conflict_id * 100 + k, and each text ends in `m<message id>`."""
        con = self._con()
        schema.create_schema(con)
        con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                    "VALUES (1, 'chat1', 0, 'iMessage')")
        for cid, n_turns, signal_count in windows:
            day = f"2026-03-{cid:02d}"
            for k in range(n_turns):
                mid = cid * 100 + k
                con.execute(
                    "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
                    "text, date, item_type, is_emote, has_attachment) "
                    "VALUES (?, ?, 1, 50, ?, ?, ?, 0, 0, 0)",
                    (mid, f"g-{mid}", k % 2, f"{_LABELABLE} m{mid}",
                     f"{day}T09:{k:02d}:00"))
            con.execute(
                "INSERT INTO conflicts(id, chat_id, first_date, last_date, "
                "peak_signal_json, corroboration, off_text_flag, source) "
                "VALUES (?, 1, ?, ?, ?, 'none', 0, 'llm')",
                (cid, f"{day}T09:00:00", f"{day}T10:00:00",
                 json.dumps({"signals": ["s"] * signal_count,
                             "signal_count": signal_count})))
        set_state(con, "conflict_model", model)
        con.commit()
        return con

    @staticmethod
    def _stamps(con):
        return dict(con.execute("SELECT id, labeled_at FROM conflicts").fetchall())

    @staticmethod
    def _n_signals(con, conflict_id):
        return con.execute("SELECT COUNT(*) FROM conflict_signals WHERE conflict_id = ?",
                           (conflict_id,)).fetchone()[0]


class DeadlineGateTest(_ConflictWindowsTest):
    def test_labels_the_strongest_window_first(self):
        con = self._seed([(1, 1, 2), (2, 1, 4), (3, 1, 3)])
        cls = _Classifier()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=cls)
        self.assertEqual(cls.ids, [200, 300, 100])   # by signal_count, descending

    def test_stops_dispatching_when_the_next_window_will_not_fit(self):
        # Rate: 10s/call banked * 1.5 safety = 15s planned, so a 2-turn window
        # is planned at 30s. Each call really costs 10s, so windows 1 and 2 fit
        # inside the 55s budget and window 3's estimate (1040 + 30 > 1055) does
        # not -- the loop declines it BETWEEN windows.
        con = self._seed([(1, 2, 4), (2, 2, 3), (3, 2, 2)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=10.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now + 55, monotonic=clock.monotonic,
                rate=_warm_rate(10.0))
        self.assertEqual(n, 2)
        self.assertEqual(cls.ids, [100, 101, 200, 201])   # window 3 never dispatched
        stamps = self._stamps(con)
        self.assertIsNotNone(stamps[1])
        self.assertIsNotNone(stamps[2])
        self.assertIsNone(stamps[3])
        # What ran is committed; what didn't left no trace.
        self.assertEqual(self._n_signals(con, 1), 2)
        self.assertEqual(self._n_signals(con, 2), 2)
        self.assertEqual(self._n_signals(con, 3), 0)
        said = _said(printed)
        self.assertIn("stopped at the deadline", said)
        self.assertIn("1 of 3 still unlabeled", said)     # names what was left
        self.assertIn("rerun to resume", said)

    def test_a_window_that_runs_out_mid_way_is_rolled_back(self):
        # The second call costs 40s against a 1.5s plan, which re-prices the
        # rate mid-window: window 2's per-call check then fails after one turn.
        # A half-labeled window must not be committed or stamped.
        con = self._seed([(1, 1, 4), (2, 3, 3)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=[1.0, 40.0])
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now + 60, monotonic=clock.monotonic,
                rate=_warm_rate(1.0))
        self.assertEqual(n, 1)
        self.assertEqual(len(cls.texts), 2)               # one turn of window 2 ran
        stamps = self._stamps(con)
        self.assertIsNotNone(stamps[1])
        self.assertIsNone(stamps[2])                      # still unlabeled...
        self.assertEqual(self._n_signals(con, 2), 0)      # ...and rolled back
        self.assertEqual(self._n_signals(con, 1), 1)      # the whole window survives
        self.assertIn("1 of 2 still unlabeled", _said(printed))

    def test_a_budget_too_small_for_one_window_says_it_dispatched_nothing(self):
        # A cold rate reserves the model load (~60s) on top of the per-call
        # estimate, so a 30s budget cannot buy even one window. Declining is the
        # right answer, but silence would read as "nothing to do".
        con = self._seed([(1, 3, 2)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=10.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now + 30, monotonic=clock.monotonic,
                rate=conflict_detect._TurnRate())
        self.assertEqual(n, 0)
        self.assertEqual(cls.texts, [])
        self.assertIsNone(self._stamps(con)[1])
        said = _said(printed)
        self.assertIn("dispatched nothing", said)
        self.assertIn("raise CONFLICT_BUDGET_MINUTES", said)

    def test_an_expired_clock_stops_before_a_window_is_even_loaded(self):
        # The top-of-loop check, not just the per-window/per-call ones: window 1
        # ranks first and has NO labelable turns, so it costs no LLM call and
        # the request gates never see it -- but it still loads its chat's
        # messages. An expired budget must stop before that segment build.
        con = self._seed([(1, 2, 3), (2, 2, 2)])
        con.execute("UPDATE messages SET text = 'ok' WHERE id IN (100, 101)")
        con.commit()
        clock = _FakeClock()
        cls = _Classifier()
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now, monotonic=clock.monotonic,
                rate=_warm_rate(1.0))
        self.assertEqual(n, 0)
        self.assertEqual(cls.texts, [])
        self.assertTrue(all(v is None for v in self._stamps(con).values()))
        self.assertIn("dispatched nothing", _said(printed))

    def test_a_window_with_no_labelable_turns_is_stamped_without_a_request(self):
        # The prescreen buys such a window for free -- it is processed, so it
        # is stamped, and a rerun must not keep re-considering it.
        con = self._seed([(1, 2, 2)])
        con.execute("UPDATE messages SET text = 'ok'")
        con.commit()
        cls = _Classifier()
        with mock.patch("builtins.print"):
            n = conflict_detect.label_top_conflicts(con, "test-model",
                                                    max_label_turns=6,
                                                    classify_fn=cls)
        self.assertEqual(n, 1)
        self.assertEqual(cls.texts, [])
        self.assertIsNotNone(self._stamps(con)[1])

    def test_no_deadline_labels_every_window(self):
        # deadline=None is unbudgeted: a call price that would blow any budget
        # is irrelevant because nothing is ever consulted.
        con = self._seed([(1, 2, 3), (2, 2, 2), (3, 2, 1)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=10_000.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=None, monotonic=clock.monotonic)
        self.assertEqual(n, 3)
        self.assertTrue(all(v is not None for v in self._stamps(con).values()))
        self.assertNotIn("stopped at the deadline", _said(printed))


class ResumeTest(_ConflictWindowsTest):
    def test_labeled_at_is_stamped_after_a_window_commits(self):
        con = self._seed([(1, 2, 2)])
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=_Classifier())
        row = con.execute("SELECT labeled_at, topic FROM conflicts").fetchone()
        self.assertIsNotNone(row["labeled_at"])
        self.assertEqual(row["topic"], "criticism")

    def test_stamped_even_when_no_label_survived(self):
        # A window of all-neutral turns writes zero conflict_signals rows. The
        # rows are NOT the resume marker, or every rerun would re-buy it.
        con = self._seed([(1, 2, 2)])
        cls = _Classifier(verdict=_stub_neutral)
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=cls)
        self.assertEqual(len(cls.texts), 2)               # it really did the work
        self.assertEqual(self._n_signals(con, 1), 0)
        row = con.execute("SELECT labeled_at, topic FROM conflicts").fetchone()
        self.assertIsNotNone(row["labeled_at"])
        self.assertIsNone(row["topic"])                   # no survivors -> no topic

    def test_stamped_even_when_the_gate_suppresses_every_class(self):
        con = self._seed([(1, 2, 2)])
        owner = {"passed": [], "failed": list(gates.TASKS["conflict_labels"]["classes"]),
                 "metric": "kappa", "values": {}, "n_labels": 60,
                 "labeled_at": "T", "source": "owner"}
        set_state(con, gates.state_key("conflict_labels"), json.dumps(owner))
        con.commit()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=_Classifier())
        self.assertEqual(self._n_signals(con, 1), 0)      # every class suppressed
        self.assertIsNotNone(self._stamps(con)[1])        # ...but processed

    def test_a_stamped_window_is_skipped_on_the_next_run(self):
        con = self._seed([(1, 1, 2), (2, 1, 4)])
        first = _Classifier()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", n=1,
                                                max_label_turns=6, classify_fn=first)
        self.assertEqual(first.ids, [200])                # the strongest window
        second = _Classifier()
        with mock.patch("builtins.print"):
            n = conflict_detect.label_top_conflicts(con, "test-model",
                                                    max_label_turns=6,
                                                    classify_fn=second)
        self.assertEqual(n, 1)
        self.assertEqual(second.ids, [100])               # window 2 not re-dispatched

    def test_a_fully_labeled_corpus_dispatches_nothing_and_says_so(self):
        con = self._seed([(1, 1, 2)])
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=_Classifier())
        again = _Classifier()
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(con, "test-model",
                                                    max_label_turns=6,
                                                    classify_fn=again)
        self.assertEqual(n, 0)
        self.assertEqual(again.texts, [])
        self.assertIn("already labeled", _said(printed))

    def test_a_window_labeled_by_the_old_code_has_its_signals_replaced(self):
        # Pre-budget rows carry signals but no labeled_at, so this pass re-labels
        # them -- it must REPLACE the old rows, not stack a second set on top.
        con = self._seed([(1, 2, 2)])
        con.executemany(
            "INSERT INTO conflict_signals(conflict_id, message_id, actor_is_me, "
            "label_class, top2_labels_json, band, excerpt) "
            "VALUES (1, ?, 0, 'contempt', '[]', 'likely', 'stale')",
            [(9001,), (9002,), (9003,)])
        con.commit()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(con, "test-model", max_label_turns=6,
                                                classify_fn=_Classifier())
        rows = con.execute(
            "SELECT excerpt FROM conflict_signals WHERE conflict_id = 1").fetchall()
        self.assertEqual(len(rows), 2)                    # the 2 fresh turns, not 5
        self.assertNotIn("stale", [r[0] for r in rows])


class DetectChatReconcileTest(_TempDBTest):
    """Re-detection reconciles in place. A delete-then-rewrite would throw away
    the labels the last budget bought, and no rerun could ever extend coverage."""

    def test_an_unchanged_window_keeps_its_id_signals_topic_and_stamp(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        con = self._con()
        before = con.execute(
            "SELECT id, topic, labeled_at FROM conflicts").fetchone()
        n_signals = con.execute("SELECT COUNT(*) FROM conflict_signals").fetchone()[0]
        self.assertTrue(n_signals)
        self.assertIsNotNone(before["labeled_at"])        # the single-phase path stamps
        # Scramble the deterministic columns so their refresh is observable.
        con.execute("UPDATE conflicts SET corroboration = 'bogus', "
                    "peak_signal_json = '{}', off_text_flag = 0, "
                    "start_session_id = NULL")
        con.commit()
        with mock.patch("builtins.print"):
            conflict_detect.detect_conflicts(con, "test-model", label=False)
        con.commit()
        after = con.execute(
            "SELECT id, topic, labeled_at, corroboration, off_text_flag, "
            "peak_signal_json, start_session_id FROM conflicts").fetchone()
        self.assertEqual(after["id"], before["id"])       # same row, not a rewrite
        self.assertEqual(after["topic"], before["topic"])
        self.assertEqual(after["labeled_at"], before["labeled_at"])
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM conflict_signals").fetchone()[0],
            n_signals)
        # ...while the deterministic evidence IS refreshed.
        self.assertEqual(after["corroboration"], "none")
        self.assertEqual(after["off_text_flag"], 1)
        self.assertIsNotNone(after["start_session_id"])
        self.assertIn("latency_withdrawal",
                      json.loads(after["peak_signal_json"])["signals"])

    def test_a_window_that_no_longer_exists_is_dropped_with_its_signals(self):
        _build_fixture(self.db_path, classify_fn=_stub_criticism)
        con = self._con()
        con.execute(
            "INSERT INTO conflicts(id, chat_id, first_date, last_date, "
            "peak_signal_json, corroboration, off_text_flag, source, labeled_at) "
            "VALUES (900, 1, '2025-01-01T00:00:00', '2025-01-02T00:00:00', "
            "'{}', 'none', 0, 'llm', '2025-01-03T00:00:00')")
        con.execute("INSERT INTO conflict_signals(conflict_id, message_id, "
                    "actor_is_me, label_class, top2_labels_json, band, excerpt) "
                    "VALUES (900, 1, 0, 'criticism', '[]', 'likely', 'x')")
        con.commit()
        with mock.patch("builtins.print"):
            conflict_detect.detect_conflicts(con, "test-model", label=False)
        con.commit()
        self.assertIsNone(
            con.execute("SELECT id FROM conflicts WHERE id = 900").fetchone())
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM conflict_signals "
                        "WHERE conflict_id = 900").fetchone()[0], 0)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0], 1)


# --- rate banking ------------------------------------------------------------


class TurnRateTest(unittest.TestCase):
    def test_the_first_call_of_a_run_is_never_counted(self):
        # It carries the model's cold load (~54s measured); folding that into the
        # mean would tell the next run that every call costs a cold load.
        rate = conflict_detect._TurnRate()
        rate.record(54.0)
        self.assertIsNone(rate.measured())
        rate.record(2.0)
        self.assertEqual(rate.measured(), 2.0)

    def test_a_cold_rate_reserves_the_model_load_exactly_once(self):
        rate = conflict_detect._TurnRate(prior=2.0)
        warm_call = 2.0 * conflict_detect._TURN_RATE_SAFETY
        self.assertEqual(rate.next_call(),
                         warm_call + conflict_detect._COLD_LOAD_SECONDS)
        rate.record(2.0)
        self.assertEqual(rate.next_call(), warm_call)

    def test_window_cost_is_one_dispatch_plus_the_remaining_calls(self):
        rate = conflict_detect._TurnRate(prior=2.0)
        rate.record(2.0)                                  # warm
        self.assertEqual(rate.window(4), 4 * 2.0 * conflict_detect._TURN_RATE_SAFETY)


class TurnRateBankingTest(_ConflictWindowsTest):
    def _bare_db(self):
        con = self._con()
        schema.create_schema(con)
        con.commit()
        return con

    def test_a_measured_rate_is_banked_and_primes_the_next_run(self):
        con = self._bare_db()
        rate = conflict_detect._TurnRate()
        for seconds in (54.0, 2.0, 4.0):                  # first = the cold call
            rate.record(seconds)
        conflict_detect._save_turn_rate(con, rate, "test-model")
        self.assertEqual(get_state(con, "conflict_turn_seconds:test-model"), repr(3.0))
        self.assertEqual(conflict_detect._load_turn_rate(con, "test-model").prior, 3.0)

    def test_an_unmeasured_rate_is_never_stored(self):
        # A run whose ONLY call took 54s measured nothing (that call is the cold
        # load); writing the prior back would freeze a guess where nothing could
        # ever correct it.
        con = self._bare_db()
        rate = conflict_detect._TurnRate()
        rate.record(54.0)
        conflict_detect._save_turn_rate(con, rate, "test-model")
        self.assertIsNone(get_state(con, "conflict_turn_seconds:test-model"))

    def test_a_corrupt_or_nonsensical_banked_value_falls_back_to_the_cold_prior(self):
        con = self._bare_db()
        for stored in ("not-a-number", "", "0", "-3"):
            set_state(con, "conflict_turn_seconds:test-model", stored)
            con.commit()
            self.assertEqual(conflict_detect._load_turn_rate(con, "test-model").prior,
                             conflict_detect._COLD_TURN_SECONDS)

    def test_a_run_banks_what_it_measured(self):
        con = self._seed([(1, 3, 2)])
        clock = _FakeClock()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6,
                classify_fn=_Classifier(clock=clock, seconds=4.0),
                monotonic=clock.monotonic)
        self.assertEqual(get_state(con, "conflict_turn_seconds:test-model"), repr(4.0))

    def test_a_single_call_run_banks_nothing(self):
        con = self._seed([(1, 1, 2)])
        clock = _FakeClock()
        with mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6,
                classify_fn=_Classifier(clock=clock, seconds=54.0),
                monotonic=clock.monotonic)
        self.assertIsNone(get_state(con, "conflict_turn_seconds:test-model"))


# --- budget resolution -------------------------------------------------------


class ResolveBudgetMinutesTest(unittest.TestCase):
    def test_caller_beats_env(self):
        with mock.patch.object(conflict_detect, "CONFLICT_BUDGET_MINUTES", 13):
            self.assertEqual(conflict_detect._resolve_budget_minutes(7),
                             (7, "--budget-minutes"))

    def test_env_beats_the_default(self):
        with mock.patch.object(conflict_detect, "CONFLICT_BUDGET_MINUTES", 13):
            self.assertEqual(conflict_detect._resolve_budget_minutes(None),
                             (13, "CONFLICT_BUDGET_MINUTES"))

    def test_zero_means_unbudgeted_from_either_source(self):
        # 0 must restore the unbounded path, NOT fall back to the default the way
        # annotate's hybrid tier reads 0 as "unset".
        with mock.patch.object(conflict_detect, "CONFLICT_BUDGET_MINUTES", 25):
            self.assertEqual(conflict_detect._resolve_budget_minutes(0),
                             (0, "--budget-minutes"))
        with mock.patch.object(conflict_detect, "CONFLICT_BUDGET_MINUTES", 0):
            self.assertEqual(conflict_detect._resolve_budget_minutes(None)[0], 0)

    def test_a_negative_budget_clamps_to_zero(self):
        self.assertEqual(conflict_detect._resolve_budget_minutes(-5)[0], 0)
        with mock.patch.object(conflict_detect, "CONFLICT_BUDGET_MINUTES", -9):
            self.assertEqual(conflict_detect._resolve_budget_minutes(None)[0], 0)


# --- labeled_at migration + self-heal ----------------------------------------


# A `conflicts` table as it stood BEFORE the budgeted tier: no labeled_at.
_PRE_BUDGET_CONFLICTS_DDL = """
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
    );
"""


def _conflict_columns(con):
    return {row[1] for row in con.execute("PRAGMA table_info(conflicts)")}


class LabeledAtMigrationTest(unittest.TestCase):
    def _pre_budget_db(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        con.executescript(_PRE_BUDGET_CONFLICTS_DDL)
        con.execute("INSERT INTO conflicts(chat_id, first_date, last_date) "
                    "VALUES (1, '2026-01-01', '2026-01-02')")
        con.commit()
        return con

    def test_adds_labeled_at_to_a_table_that_predates_it(self):
        con = self._pre_budget_db()
        self.assertNotIn("labeled_at", _conflict_columns(con))
        schema.create_schema(con)
        self.assertIn("labeled_at", _conflict_columns(con))
        # The pre-existing window reads NULL = "label it": nothing recorded WHEN
        # the old path labeled it, and a made-up stamp would skip unverified work.
        self.assertIsNone(con.execute("SELECT labeled_at FROM conflicts").fetchone()[0])

    def test_the_migration_is_idempotent(self):
        con = self._pre_budget_db()
        schema.create_schema(con)
        schema.create_schema(con)                         # must not raise
        self.assertIn("labeled_at", _conflict_columns(con))
        self.assertEqual(con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0], 1)


class ConflictsReadyTest(unittest.TestCase):
    def _con(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def test_false_when_the_table_does_not_exist(self):
        self.assertFalse(conflict_detect._conflicts_ready(self._con()))

    def test_false_when_the_table_predates_labeled_at(self):
        con = self._con()
        con.executescript(_PRE_BUDGET_CONFLICTS_DDL)
        self.assertFalse(conflict_detect._conflicts_ready(con))

    def test_true_after_create_schema(self):
        con = self._con()
        schema.create_schema(con)
        self.assertTrue(conflict_detect._conflicts_ready(con))


# --- CLI wiring --------------------------------------------------------------


class _Recorder:
    """A stand-in for detect_conflicts / label_top_conflicts: records how main
    called it, so the phase selection is asserted without reaching an LLM path."""

    def __init__(self, result=0):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.result

    @property
    def only(self):
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


class CliWiringTest(_TempDBTest):
    def setUp(self):
        super().setUp()
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.commit()
        con.close()
        self.detect = _Recorder()
        self.label = _Recorder()
        for name, value in (("detect_conflicts", self.detect),
                            ("label_top_conflicts", self.label),
                            ("DB_PATH", self.db_path),
                            ("CONFLICT_THINK", None),
                            ("CONFLICT_BUDGET_MINUTES", 10)):
            patcher = mock.patch.object(conflict_detect, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        printer = mock.patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)

    def test_a_bare_run_is_two_phase_with_a_real_deadline(self):
        conflict_detect.main([])
        self.assertIs(self.detect.only["label"], False)   # phase 1: no LLM work
        labeling_call = self.label.only
        self.assertIsNone(labeling_call["n"])             # no fixed N, just the clock
        self.assertIsNotNone(labeling_call["deadline"])
        self.assertGreater(labeling_call["deadline"], time.monotonic())
        self.assertIs(labeling_call["think"], False)      # resolved, never omitted

    def test_budget_zero_without_label_top_is_the_legacy_single_phase_path(self):
        conflict_detect.main(["--budget-minutes", "0"])
        self.assertIs(self.detect.only["label"], True)    # labels as it detects
        self.assertIs(self.detect.only["think"], False)
        self.assertEqual(self.label.calls, [])

    def test_budget_zero_with_label_top_is_two_phase_without_a_deadline(self):
        conflict_detect.main(["--budget-minutes", "0", "--label-top", "5"])
        self.assertIs(self.detect.only["label"], False)
        self.assertEqual(self.label.only["n"], 5)
        self.assertIsNone(self.label.only["deadline"])

    def test_no_llm_with_a_budget_is_a_parser_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                conflict_detect.main(["--no-llm", "--budget-minutes", "5"])
        self.assertEqual(self.detect.calls, [])
        self.assertEqual(self.label.calls, [])

    def test_no_llm_alone_still_detects_without_labeling(self):
        conflict_detect.main(["--no-llm"])
        self.assertIs(self.detect.only["label"], False)
        self.assertEqual(self.label.calls, [])

    def test_dry_run_makes_no_labeling_call(self):
        conflict_detect.main(["--dry-run"])
        self.assertTrue(self.detect.only["dry_run"])
        self.assertEqual(self.label.calls, [])


class MainSelfHealTest(_TempDBTest):
    def test_main_migrates_a_db_that_predates_labeled_at(self):
        # conflict_detect is reachable without the ingest.py run that normally
        # applies migrations, so main has to self-heal or every query on the
        # resume column would blow up.
        con = sqlite3.connect(self.db_path)
        con.executescript(_PRE_BUDGET_CONFLICTS_DDL)
        con.commit()
        con.close()
        with mock.patch.object(conflict_detect, "DB_PATH", self.db_path), \
                mock.patch.object(conflict_detect, "detect_conflicts", _Recorder()), \
                mock.patch.object(conflict_detect, "label_top_conflicts", _Recorder()), \
                mock.patch("builtins.print"):
            conflict_detect.main([])
        self.assertIn("labeled_at", _conflict_columns(self._con()))


# --- a failed call must never poison the banked rate -------------------------


class _RaisingClassifier(_Classifier):
    """A _Classifier whose `fail_on`-th call (1-based) RAISES instead of
    returning, after charging the clock what the failure really cost: an ollama
    timeout burns CONFLICT_TIMEOUT (300s by default) before it raises, and that
    duration is exactly what must never reach the estimator."""

    def __init__(self, *args, fail_on=1, cost=300.0, error=TimeoutError, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on
        self.cost = cost
        self.error = error

    def __call__(self, text):
        if len(self.texts) + 1 != self.fail_on:
            return super().__call__(text)
        self.texts.append(text)
        if self.clock is not None:
            self.clock.advance(self.cost)
        raise self.error("the model never answered")


class TimedRateRecordingTest(unittest.TestCase):
    """_timed feeds the estimator only on SUCCESS. A banked failure latches: a
    300s prior makes the per-window gate decline every window, a run that
    dispatches nothing measures nothing, and a run that measures nothing never
    overwrites the value that stopped it."""

    def test_a_raised_call_never_reaches_the_measured_mean(self):
        clock = _FakeClock()
        rate = _warm_rate(1.0)
        cls = _RaisingClassifier(clock=clock, seconds=2.0, fail_on=3)
        timed = conflict_detect._timed(cls, rate, clock.monotonic)
        timed(f"{_LABELABLE} m1")
        timed(f"{_LABELABLE} m2")
        self.assertEqual(rate.measured(), 2.0)
        with self.assertRaises(TimeoutError):
            timed(f"{_LABELABLE} m3")
        self.assertEqual(rate.measured(), 2.0)         # the 300s failure is not evidence
        self.assertEqual(rate.per_call(), 2.0 * conflict_detect._TURN_RATE_SAFETY)

    def test_a_malformed_reply_is_a_successful_call_and_is_counted(self):
        # A garbage REPLY is a different thing from a failed request:
        # _classify_turn coerces it to an abstain rather than raising, so the
        # request really did cost what it cost.
        clock = _FakeClock()
        rate = _warm_rate(1.0)

        def fake_chat_once_json(messages, model=None, think=None, timeout=None,
                                retry_nudge=None):
            clock.advance(3.0)
            return "not a JSON object at all"

        timed = conflict_detect._timed(
            lambda text: conflict_detect._classify_turn(text, "test-model", False),
            rate, clock.monotonic)
        with mock.patch.object(ollama, "chat_once_json", fake_chat_once_json):
            out = timed(_LABELABLE)
        self.assertEqual(out["class"], "abstain")
        self.assertEqual(rate.measured(), 3.0)


class FailedCallBankingTest(_ConflictWindowsTest):
    """The same rule end to end: what a run banks to ingest_state is what its
    SUCCESSFUL calls measured."""

    def test_a_failing_run_banks_only_what_its_successful_calls_measured(self):
        # Window 1 (2 turns) and window 2's first two turns succeed at 4s each;
        # its third call times out after 300s. The banked mean must be 4.0, not
        # (4 + 4 + 4 + 4 + 300) / 5.
        con = self._seed([(1, 2, 4), (2, 3, 3)])
        clock = _FakeClock()
        cls = _RaisingClassifier(clock=clock, seconds=4.0, fail_on=5)
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                monotonic=clock.monotonic, rate=_warm_rate(4.0))
        self.assertEqual(get_state(con, "conflict_turn_seconds:test-model"), repr(4.0))

    def test_a_run_whose_only_non_cold_call_raises_banks_nothing(self):
        # A cold rate: call 1 pays the model load and is never counted, so a
        # recorded failure would be the run's ONLY measurement -- 300s frozen
        # into ingest_state where no later run could ever correct it.
        con = self._seed([(1, 1, 4), (2, 1, 3)])
        clock = _FakeClock()
        cls = _RaisingClassifier(clock=clock, seconds=54.0, fail_on=2)
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                monotonic=clock.monotonic)
        self.assertIsNone(get_state(con, "conflict_turn_seconds:test-model"))


# --- a raised call is the deadline contract: all of a window, or none of it ---


class MidWindowFailureTest(_ConflictWindowsTest):
    """An exception mid-window must not commit a half-labeled window. Without
    the rollback, _save_turn_rate's commit on the way out persists that window's
    partial signal rows with labeled_at still NULL -- rows a reader would read
    as the window's complete labeling."""

    def _run_until_failure(self, error=TimeoutError):
        """Window 1 (2 turns) completes; window 2 raises on its THIRD turn."""
        con = self._seed([(1, 2, 4), (2, 4, 3)])
        clock = _FakeClock()
        cls = _RaisingClassifier(clock=clock, seconds=1.0, fail_on=5, error=error)
        return con, clock, cls

    def _label(self, con, clock, cls):
        return conflict_detect.label_top_conflicts(
            con, "test-model", max_label_turns=6, classify_fn=cls,
            monotonic=clock.monotonic, rate=_warm_rate(1.0))

    def test_the_failure_propagates_to_the_caller(self):
        con, clock, cls = self._run_until_failure()
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            self._label(con, clock, cls)
        self.assertEqual(len(cls.texts), 5)             # it really got that far

    def test_the_half_labeled_window_is_rolled_back_and_left_unstamped(self):
        con, clock, cls = self._run_until_failure()
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            self._label(con, clock, cls)
        self.assertEqual(self._n_signals(con, 2), 0)    # its 2 written rows are gone
        self.assertIsNone(self._stamps(con)[2])         # ...and it is still unlabeled

    def test_windows_completed_earlier_in_the_run_keep_their_rows_and_stamps(self):
        con, clock, cls = self._run_until_failure()
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            self._label(con, clock, cls)
        self.assertEqual(self._n_signals(con, 1), 2)    # committed on its own
        self.assertIsNotNone(self._stamps(con)[1])

    def test_the_rate_is_still_banked_on_the_failing_path(self):
        # The rollback is in an `except`, so the `finally` still runs: the four
        # calls that DID complete are worth banking for the next run.
        con, clock, cls = self._run_until_failure()
        with self.assertRaises(TimeoutError), mock.patch("builtins.print"):
            self._label(con, clock, cls)
        self.assertEqual(get_state(con, "conflict_turn_seconds:test-model"), repr(1.0))

    def test_a_ctrl_c_mid_window_still_returns_instead_of_propagating(self):
        # KeyboardInterrupt is a BaseException too, so its handler has to stay
        # AHEAD of the generic one -- otherwise Ctrl-C starts raising out of a
        # run that used to exit cleanly.
        con, clock, cls = self._run_until_failure(error=KeyboardInterrupt)
        with mock.patch("builtins.print") as printed:
            n = self._label(con, clock, cls)
        self.assertEqual(n, 1)
        self.assertEqual(self._n_signals(con, 2), 0)
        self.assertIsNone(self._stamps(con)[2])
        self.assertIn("Interrupted", _said(printed))


# --- --limit has to scope the labeling phase too -----------------------------


class LabelScopeTest(_ConflictWindowsTest):
    """A --limit run scanned only some chats; without chat_ids the budgeted
    second phase queries `conflicts` corpus-wide and spends the whole budget
    labeling windows in chats the run was told not to touch."""

    def _seed_chats(self, windows):
        """windows: [(conflict_id, chat_id, n_turns, signal_count)] -- _seed's
        multi-chat form (that one only ever builds chat 1). Same id/text
        conventions, so _Classifier.ids still names the dispatched windows."""
        con = self._con()
        schema.create_schema(con)
        for chat_id in sorted({w[1] for w in windows}):
            con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                        "VALUES (?, ?, 0, 'iMessage')", (chat_id, f"chat{chat_id}"))
        for cid, chat_id, n_turns, signal_count in windows:
            day = f"2026-03-{cid:02d}"
            for k in range(n_turns):
                mid = cid * 100 + k
                con.execute(
                    "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
                    "text, date, item_type, is_emote, has_attachment) "
                    "VALUES (?, ?, ?, 50, ?, ?, ?, 0, 0, 0)",
                    (mid, f"g-{mid}", chat_id, k % 2, f"{_LABELABLE} m{mid}",
                     f"{day}T09:{k:02d}:00"))
            con.execute(
                "INSERT INTO conflicts(id, chat_id, first_date, last_date, "
                "peak_signal_json, corroboration, off_text_flag, source) "
                "VALUES (?, ?, ?, ?, ?, 'none', 0, 'llm')",
                (cid, chat_id, f"{day}T09:00:00", f"{day}T10:00:00",
                 json.dumps({"signals": ["s"] * signal_count,
                             "signal_count": signal_count})))
        set_state(con, "conflict_model", "test-model")
        con.commit()
        return con

    def test_only_in_scope_windows_are_labeled_when_one_out_of_scope_ranks_higher(self):
        # Chat 2's window is the STRONGEST in the table -- exactly the one an
        # unscoped ranking would spend the budget on first.
        con = self._seed_chats([(1, 1, 1, 2), (2, 2, 1, 5)])
        cls = _Classifier()
        with mock.patch("builtins.print"):
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls, chat_ids=[1])
        self.assertEqual(n, 1)
        self.assertEqual(cls.ids, [100])
        stamps = self._stamps(con)
        self.assertIsNotNone(stamps[1])
        self.assertIsNone(stamps[2])                    # out of scope: untouched
        self.assertEqual(self._n_signals(con, 2), 0)

    def test_chat_ids_none_is_corpus_wide(self):
        con = self._seed_chats([(1, 1, 1, 2), (2, 2, 1, 5)])
        cls = _Classifier()
        with mock.patch("builtins.print"):
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls, chat_ids=None)
        self.assertEqual(n, 2)
        self.assertEqual(cls.ids, [200, 100])           # strongest first, across chats
        self.assertTrue(all(v is not None for v in self._stamps(con).values()))

    def test_a_scope_holding_no_windows_dispatches_nothing(self):
        con = self._seed_chats([(1, 1, 1, 2)])
        cls = _Classifier()
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls, chat_ids=[2])
        self.assertEqual(n, 0)
        self.assertEqual(cls.texts, [])
        self.assertIsNone(self._stamps(con)[1])
        self.assertIn("No conflicts to label", _said(printed))

    def test_the_completed_summary_counts_the_scoped_set(self):
        con = self._seed_chats([(1, 1, 1, 2), (2, 2, 1, 5), (3, 2, 1, 4)])
        with mock.patch("builtins.print") as printed:
            conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=_Classifier(),
                chat_ids=[1])
        said = _said(printed)
        self.assertIn("Labeled 1 of 1 conflict windows", said)   # not "1 of 3"
        self.assertIn("(0 left unlabeled)", said)

    def test_the_deadline_summary_counts_the_scoped_set(self):
        # 10s/call banked * 1.5 = 15s planned, so a 2-turn window is planned at
        # 30s: window 1 fits inside the 35s budget and window 2 does not. The
        # backlog sentence must name the SCOPED two, not the table's four.
        con = self._seed_chats([(1, 1, 2, 4), (2, 1, 2, 3), (3, 2, 2, 5), (4, 2, 2, 2)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=10.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls, chat_ids=[1],
                deadline=clock.now + 35, monotonic=clock.monotonic,
                rate=_warm_rate(10.0))
        self.assertEqual(n, 1)
        self.assertEqual(cls.ids, [100, 101])           # chat 1 only
        said = _said(printed)
        self.assertIn("1 of 2 still unlabeled", said)
        self.assertNotIn("3 of 4 still unlabeled", said)


# --- the "dispatched nothing" line is keyed on requests actually issued ------


class DispatchedNothingLineTest(_ConflictWindowsTest):
    def test_not_printed_when_a_window_was_rolled_back_after_real_requests(self):
        # The window is admitted (3 turns planned at 4.5s inside a 10s budget),
        # its first call costs 40s and re-prices the rate, and the per-call gate
        # then rolls it back: `labeled` is 0 but a real request was spent, so
        # this run did NOT decline everything.
        con = self._seed([(1, 3, 2)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=40.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now + 10, monotonic=clock.monotonic,
                rate=_warm_rate(1.0))
        self.assertEqual(n, 0)
        self.assertEqual(len(cls.texts), 1)             # one request was issued
        self.assertEqual(self._n_signals(con, 1), 0)    # ...and rolled back
        self.assertIsNone(self._stamps(con)[1])
        said = _said(printed)
        self.assertNotIn("dispatched nothing", said)
        self.assertIn("stopped at the deadline", said)

    def test_printed_when_a_warm_rate_declines_every_window(self):
        # A caller can hand in a rate that already paid its cold load, so
        # rate.warm says nothing about whether THIS call dispatched anything:
        # 15s/call planned makes a 3-turn window 45s, over the 20s budget.
        con = self._seed([(1, 3, 2)])
        clock = _FakeClock()
        cls = _Classifier(clock=clock, seconds=10.0)
        rate = _warm_rate(10.0)
        with mock.patch("builtins.print") as printed:
            n = conflict_detect.label_top_conflicts(
                con, "test-model", max_label_turns=6, classify_fn=cls,
                deadline=clock.now + 20, monotonic=clock.monotonic, rate=rate)
        self.assertEqual(n, 0)
        self.assertEqual(cls.texts, [])
        self.assertTrue(rate.warm)                      # warm, yet nothing dispatched
        self.assertIn("dispatched nothing", _said(printed))


# --- Ctrl-C during detection must stop the run, not start the LLM phase ------


class DetectStatsTest(_TempDBTest):
    """detect_conflicts fills a caller-owned `stats` dict with the two facts a
    two-phase caller needs and the return value cannot carry: which chats were
    actually scanned, and whether a Ctrl-C stopped the pass."""

    def _chats(self, n_direct, n_group=0):
        con = self._con()
        schema.create_schema(con)
        for cid in range(1, n_direct + 1):
            con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                        "VALUES (?, ?, 0, 'iMessage')", (cid, f"chat{cid}"))
        for k in range(n_group):
            con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                        "VALUES (?, ?, 1, 'iMessage')", (100 + k, f"group{k}"))
        con.commit()
        return con

    def _detect(self, con, **kwargs):
        stats = {}
        with mock.patch("builtins.print"):
            n = conflict_detect.detect_conflicts(con, "test-model", label=False,
                                                 stats=stats, **kwargs)
        return n, stats

    def test_a_completed_scan_records_the_chats_and_no_interrupt(self):
        _n, stats = self._detect(self._chats(3, n_group=2))
        self.assertFalse(stats["interrupted"])
        self.assertEqual(stats["chat_ids"], [1, 2, 3])  # group chats are never scanned

    def test_chat_ids_respects_limit(self):
        _n, stats = self._detect(self._chats(4), limit=2)
        self.assertEqual(stats["chat_ids"], [1, 2])

    def test_chat_ids_is_filled_under_dry_run(self):
        n, stats = self._detect(self._chats(3), dry_run=True)
        self.assertEqual(n, 0)
        self.assertEqual(stats["chat_ids"], [1, 2, 3])
        self.assertFalse(stats["interrupted"])

    def test_an_interrupted_scan_is_recorded(self):
        con = self._chats(3)
        with mock.patch.object(conflict_detect, "detect_chat",
                               side_effect=KeyboardInterrupt):
            n, stats = self._detect(con)
        self.assertEqual(n, 0)                          # Ctrl-C on the first chat
        self.assertTrue(stats["interrupted"])
        self.assertEqual(stats["chat_ids"], [1, 2, 3])

    def test_stats_is_optional(self):
        con = self._chats(2)
        with mock.patch("builtins.print"):
            conflict_detect.detect_conflicts(con, "test-model", label=False)


class _DetectDouble(_Recorder):
    """detect_conflicts stand-in that also plays its out-of-band `stats`
    contract; a plain _Recorder leaves the dict at main's pre-seeded defaults,
    which is a different case (the degrade-gracefully one)."""

    def __init__(self, chat_ids=(), interrupted=False):
        super().__init__()
        self.chat_ids = list(chat_ids)
        self.interrupted = interrupted

    def __call__(self, *args, **kwargs):
        stats = kwargs.get("stats")
        if stats is not None:
            stats["chat_ids"] = list(self.chat_ids)
            stats["interrupted"] = self.interrupted
        return super().__call__(*args, **kwargs)


class CliStatsWiringTest(_TempDBTest):
    """What main does with those facts: --limit scopes the labeling phase to the
    chats detection scanned, and an interrupted detection never starts it. Same
    doubles as CliWiringTest, with detect_conflicts filling `stats`."""

    def setUp(self):
        super().setUp()
        con = sqlite3.connect(self.db_path)
        schema.create_schema(con)
        con.commit()
        con.close()
        self.label = _Recorder()
        for name, value in (("label_top_conflicts", self.label),
                            ("DB_PATH", self.db_path),
                            ("CONFLICT_THINK", None),
                            ("CONFLICT_BUDGET_MINUTES", 10)):
            patcher = mock.patch.object(conflict_detect, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        printer = mock.patch("builtins.print")
        self.printed = printer.start()
        self.addCleanup(printer.stop)

    def _detect(self, **kwargs):
        double = _DetectDouble(**kwargs)
        patcher = mock.patch.object(conflict_detect, "detect_conflicts", double)
        patcher.start()
        self.addCleanup(patcher.stop)
        return double

    def test_limit_scopes_the_labeling_phase_to_the_chats_that_were_scanned(self):
        detect = self._detect(chat_ids=[4, 9, 11])
        conflict_detect.main(["--limit", "3"])
        self.assertEqual(detect.only["limit"], 3)
        self.assertEqual(self.label.only["chat_ids"], [4, 9, 11])

    def test_a_bare_run_labels_corpus_wide(self):
        self._detect(chat_ids=[4, 9, 11])
        conflict_detect.main([])
        self.assertIsNone(self.label.only["chat_ids"])   # every chat was scanned

    def test_limit_zero_scopes_to_nothing_rather_than_to_everything(self):
        # The boundary the truthiness test misses: --limit 0 scans no chats, so
        # an unscoped labeling phase would spend the budget on the whole table.
        self._detect(chat_ids=[])
        conflict_detect.main(["--limit", "0"])
        self.assertEqual(self.label.only["chat_ids"], [])

    def test_an_interrupted_detection_never_starts_the_labeling_phase(self):
        self._detect(chat_ids=[4], interrupted=True)
        conflict_detect.main([])
        self.assertEqual(self.label.calls, [])           # zero Ollama traffic
        self.assertIn("Stopped before labeling", _said(self.printed))

    def test_a_completed_detection_still_labels(self):
        self._detect(chat_ids=[4])
        conflict_detect.main([])
        self.assertEqual(len(self.label.calls), 1)
        self.assertNotIn("Stopped before labeling", _said(self.printed))


if __name__ == "__main__":
    unittest.main()
