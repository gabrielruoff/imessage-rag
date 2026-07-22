"""Hermetic tests for the conflict core (RES-46 / C1): the deterministic
finder, the eval-gated label write path, the read-tool result caps, and the
schema migration + model guard. No Ollama (the label pass is a stub
classifier), no real chat.db -- everything runs against a temporary SQLite DB.

Run from the repo root:  python3 -m unittest tests.test_conflict
"""

import importlib
import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import sqlite_vec

import conflict_detect
import lexicon_build
import query_tools
import schema
import sessions
from db import iso_to_day, load_vec_extension
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


if __name__ == "__main__":
    unittest.main()
