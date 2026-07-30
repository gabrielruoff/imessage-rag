"""Hermetic tests for the V1 voiceprint core (RES-50).

Covers voiceprint.py's person_voiceprint build + guard, the query_tools
get_voiceprint read tool, and reset_db clearing the table. A hand-scripted
temp SQLite DB with known messages -- no Ollama, no real chat.db. Feature
values that matter (burst_size, reply pairs, key semantics, the per-scope std
map) are hand-computed and pinned.
"""

import contextlib
import importlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import chat
import config
import lexicon_build
import ollama
import query_tools
import reset_db
import schema
import sessions
import tool_schemas
import voiceprint
from query_tools import persona


def _msg(mid, chat, handle, from_me, text, date):
    return (mid, f"guid-{mid}", chat, handle, from_me, text, date, 0)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class _DBCase(unittest.TestCase):
    """Base: a fresh schema'd temp DB, with query_tools pointed at it."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        con = sqlite3.connect(self.db_path)
        try:
            schema.create_schema(con)
            self.populate(con)
            con.commit()
        finally:
            con.close()
        saved = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", saved)
        # These hand-scripted fixtures are intentionally tiny (well under the
        # production VOICEPRINT_MIN_RETURN_MESSAGES floor); drop the floor to 0
        # so build + read behave as they did before the floor existed. Tests
        # that exercise the floor set it explicitly.
        saved_floor = config.VOICEPRINT_MIN_RETURN_MESSAGES
        config.VOICEPRINT_MIN_RETURN_MESSAGES = 0
        self.addCleanup(setattr, config, "VOICEPRINT_MIN_RETURN_MESSAGES", saved_floor)

    def populate(self, con):  # overridden
        pass

    def _con(self):
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        return con

    def build(self, **kwargs):
        con = sqlite3.connect(self.db_path)
        try:
            return _quiet(voiceprint.build_voiceprints, con, **kwargs)
        finally:
            con.close()

    def rows(self, sql, params=()):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()


class CoreBuildTest(_DBCase):
    """Two 1:1 dyads (Mom, Al) + a group chat. Deliberately no '!' anywhere,
    so exclaim_rate is 0 for every row (a constant feature -> std 0)."""

    def populate(self, con):
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
            "VALUES (?, ?, ?, ?)",
            [(12, "u12", "Mom", None), (13, "u13", "Al", "B")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(70, "+1700", 12), (71, "+1710", 13)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, 'iMessage')",
            [(100, "c100", None, 0), (101, "c101", None, 0), (200, "g200", "Group", 1)],
        )
        con.executemany("INSERT INTO chat_participants VALUES (?, ?)",
                        [(100, 70), (101, 71), (200, 70), (200, 71)])
        import datetime
        base = datetime.datetime(2026, 1, 1, 10, 0, 0)
        rows = []
        mid = 1
        # dyad chat 100 (me + Mom): alternating, spread over 5 months
        for d in range(5):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(6):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 100, 0 if from_me else 70, from_me,
                                 "hey mom hows it going" if from_me else "not much here",
                                 t.isoformat()))
                mid += 1
        # dyad chat 101 (me + Al): shorter messages
        for d in range(4):
            day = base + datetime.timedelta(days=d * 25)
            for i in range(4):
                t = day + datetime.timedelta(seconds=i * 20)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 101, 0 if from_me else 71, from_me,
                                 "yo" if from_me else "ok", t.isoformat()))
                mid += 1
        # group 200: me, Mom, Al round-robin
        for i in range(6):
            t = base + datetime.timedelta(days=1, seconds=i * 40)
            who = i % 3
            from_me = 1 if who == 0 else 0
            handle = 0 if who == 0 else (70 if who == 1 else 71)
            rows.append(_msg(mid, 200, handle, from_me, "group msg", t.isoformat()))
            mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        # build the derived tables voiceprint reads (fw rates, openers/closers)
        _quiet(lexicon_build.build_person_period_lexicon, con)
        _quiet(sessions.build_sessions, con)

    def test_both_dyad_directions_present(self):
        self.build()
        keys = {(r["member_key"], r["counterpart_key"])
                for r in self.rows(
                    "SELECT member_key, counterpart_key FROM person_voiceprint "
                    "WHERE scope = 'dyad'")}
        # me-with-Mom AND Mom-with-me both exist, as distinct rows.
        self.assertIn(("me", "contact:12"), keys)
        self.assertIn(("contact:12", "me"), keys)

    def test_no_unknown_or_group_counterpart_in_dyad(self):
        self.build()
        # scope='all' uses '*'; scope='group' uses 'chat:<id>'. Dyad rows never do.
        for r in self.rows("SELECT * FROM person_voiceprint WHERE scope = 'dyad'"):
            self.assertNotIn(r["member_key"], ("unknown",))
            self.assertNotEqual(r["counterpart_key"], "*")

    def test_feature_std_per_scope_zero_omitted_and_readable(self):
        self.build()
        import json
        dyad_rows = self.rows(
            "SELECT * FROM person_voiceprint WHERE scope = 'dyad'")
        maps = [json.loads(r["feature_std_json"]) for r in dyad_rows]
        # Replicated: every dyad row carries the identical std map.
        self.assertTrue(all(m == maps[0] for m in maps))
        std = maps[0]
        # avg_msg_len varies across dyad rows -> present. exclaim_rate is 0 for
        # every row (constant) -> std 0 -> omitted.
        self.assertIn("avg_msg_len", std)
        self.assertNotIn("exclaim_rate", std)
        # Readable off the row without rescanning: values are plain floats.
        self.assertTrue(all(isinstance(v, float) for v in std.values()))
        # Per-scope: the 'all' rows get their own (independently computed) map.
        all_map = json.loads(self.rows(
            "SELECT feature_std_json FROM person_voiceprint WHERE scope='all' "
            "LIMIT 1")[0]["feature_std_json"])
        self.assertIsInstance(all_map, dict)

    def test_guard_skips_when_unchanged(self):
        self.assertEqual(self.build(), 10)
        # Nothing ingested since -> a second run is a no-op.
        self.assertEqual(self.build(), 0)
        # --force rebuilds regardless.
        self.assertEqual(self.build(force=True), 10)

    def test_get_voiceprint_dyad_is_default_for_named_person(self):
        self.build()
        card = query_tools.get_voiceprint(contact_id=12)
        self.assertEqual(card["scope"], "dyad")
        self.assertEqual(card["member_key"], "me")
        self.assertEqual(card["counterpart_key"], "contact:12")
        self.assertEqual(card["counterpart"], "Mom")
        self.assertIn("features", card)
        self.assertIn("avg_msg_len", card["features"])
        self.assertIn("feature_std", card)

    def test_get_voiceprint_subject_them_flips_direction(self):
        self.build()
        card = query_tools.get_voiceprint(contact_id=12, subject="them")
        self.assertEqual((card["member_key"], card["counterpart_key"]),
                         ("contact:12", "me"))

    def test_get_voiceprint_all_when_no_person(self):
        self.build()
        card = query_tools.get_voiceprint()
        self.assertEqual(card["scope"], "all")
        self.assertEqual(card["member_key"], "me")
        self.assertEqual(card["counterpart_key"], "*")

    def test_group_excluded_unless_asked(self):
        self.build()
        # A named person never yields a group row...
        self.assertEqual(query_tools.get_voiceprint(contact_id=12)["scope"], "dyad")
        # ...but scope='group' + a group chat_id does.
        card = query_tools.get_voiceprint(scope="group", chat_id=200)
        self.assertEqual(card["scope"], "group")
        self.assertEqual(card["counterpart_key"], "chat:200")
        # group without a chat_id is an error, not a silent dyad.
        self.assertIn("error", query_tools.get_voiceprint(scope="group"))

    def test_confidence_gate_high_and_low(self):
        self.build()
        # Default thresholds (500/8/3) -> this small fixture is 'low'.
        self.assertEqual(query_tools.get_voiceprint(contact_id=12)["confidence"], "low")
        # Drop the thresholds under the fixture's numbers -> 'high'.
        for var, val in (("VOICEPRINT_MIN_MESSAGES", 5),
                         ("VOICEPRINT_MIN_REPLY_PAIRS", 2),
                         ("VOICEPRINT_MIN_MONTHS", 1)):
            saved = getattr(config, var)
            setattr(config, var, val)
            self.addCleanup(setattr, config, var, saved)
        self.assertEqual(query_tools.get_voiceprint(contact_id=12)["confidence"], "high")

    def test_tone_omitted_when_no_llm_annotations(self):
        self.build()
        # No chunk_annotations in this fixture at all -> no tone block.
        self.assertNotIn("tone", query_tools.get_voiceprint(contact_id=12))

    def test_ambiguous_name_disambiguates(self):
        self.build()
        # Both "Mom" and "Al B" ... a substring matching several returns matches.
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (14, 'u14', 'Momo')")
        con.commit()
        con.close()
        res = query_tools.get_voiceprint(name="Mo")
        self.assertIn("error", res)
        self.assertIn("matches", res)

    def test_missing_row_returns_note(self):
        self.build()
        res = query_tools.get_voiceprint(contact_id=99999)
        self.assertFalse(res.get("found", False))
        self.assertIn("note", res)


class MinReturnFloorTest(CoreBuildTest):
    """The hard VOICEPRINT_MIN_RETURN_MESSAGES floor: a sub-floor profile (the
    fixture's Mom dyad is 15 of my messages) is neither built nor returned."""

    def test_read_declines_below_floor(self):
        self.build()  # floor 0 during build (base setUp) -> the row exists
        config.VOICEPRINT_MIN_RETURN_MESSAGES = 50  # base setUp restores it
        card = query_tools.get_voiceprint(contact_id=12)
        self.assertFalse(card.get("found", False))
        self.assertIn("Too little interaction", card["note"])
        self.assertEqual(card["n_messages"], 15)
        self.assertNotIn("features", card)

    def test_read_returns_at_or_above_floor(self):
        self.build()
        config.VOICEPRINT_MIN_RETURN_MESSAGES = 10  # 15 >= 10
        card = query_tools.get_voiceprint(contact_id=12)
        self.assertEqual(card["scope"], "dyad")
        self.assertIn("features", card)

    def test_build_skips_below_floor(self):
        config.VOICEPRINT_MIN_RETURN_MESSAGES = 50  # above every fixture profile
        self.assertEqual(self.build(force=True), 0)
        self.assertEqual(
            self.rows("SELECT COUNT(*) AS c FROM person_voiceprint")[0]["c"], 0)


class BurstAndReplyPairTest(_DBCase):
    """A hand-built stream so burst_size and reply pairs have one right answer.

    chat 300 (me + handle 90, no contact). Stream (date,id) order:
      me a       (t0)
      me b       (t0+30s)   gap 30 < 120  -> same burst as a
      them x     (t0+230s)               -> breaks the run
      me c       (t0+240s)               -> new burst (prev was 'them')
      me d       (t0+250s)  gap 10 < 120  -> same burst as c
      me e       (t0+560s)  gap 310 >=120 -> new burst
    me runs: [a,b], [c,d], [e] -> n_messages=5, n_runs=3, burst_size=5/3.
    reply pairs (them -> me adjacency): only c follows x -> 1.
    """

    def populate(self, con):
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (90, '+1900', NULL, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (300, 'c300', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (300, 90)")
        import datetime
        t0 = datetime.datetime(2026, 3, 1, 9, 0, 0)

        def at(sec):
            return (t0 + datetime.timedelta(seconds=sec)).isoformat()

        rows = [
            _msg(1, 300, 0, 1, "a", at(0)),
            _msg(2, 300, 0, 1, "b", at(30)),
            _msg(3, 300, 90, 0, "x", at(230)),
            _msg(4, 300, 0, 1, "c", at(240)),
            _msg(5, 300, 0, 1, "d", at(250)),
            _msg(6, 300, 0, 1, "e", at(560)),
        ]
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def test_burst_size_and_reply_pairs(self):
        self.build()
        row = self.rows(
            "SELECT * FROM person_voiceprint WHERE member_key='me' "
            "AND counterpart_key='handle:90' AND scope='dyad'")[0]
        self.assertEqual(row["n_messages"], 5)
        self.assertAlmostEqual(row["burst_size"], 5 / 3, places=6)
        self.assertEqual(row["n_reply_pairs"], 1)


class KeyFlipSelfHealTest(_DBCase):
    """A handle that later gains a contact_id: a full rebuild re-keys it
    handle:N -> contact:M and drops the stale key."""

    def populate(self, con):
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (90, '+1900', NULL, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (300, 'c300', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (300, 90)")
        import datetime
        base = datetime.datetime(2026, 1, 1, 9, 0, 0)
        rows = []
        mid = 1
        for d in range(3):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(4):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 300, 0 if from_me else 90, from_me,
                                 "hello there friend", t.isoformat()))
                mid += 1
        self._next_mid = mid
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def test_full_rebuild_self_heals_key_flip(self):
        self.build()
        keys = {r["member_key"] for r in self.rows(
            "SELECT member_key FROM person_voiceprint")}
        self.assertIn("handle:90", keys)
        self.assertNotIn("contact:55", keys)

        # The handle gains a contact, and a new message arrives (bumping the
        # guard's max message id) -> the next run rebuilds, not skips.
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (55, 'u55', 'Newly')")
        con.execute("UPDATE handles SET contact_id = 55 WHERE id = 90")
        con.execute(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, 300, 90, 0, 'one more', "
            "'2026-06-01T09:00:00', 0)", (self._next_mid, f"guid-{self._next_mid}"))
        con.commit()
        con.close()

        self.assertGreater(self.build(), 0)  # rebuilt, not skipped
        keys = {r["member_key"] for r in self.rows(
            "SELECT member_key FROM person_voiceprint")}
        self.assertIn("contact:55", keys)
        self.assertNotIn("handle:90", keys)  # stale key gone


class ToneTest(_DBCase):
    """tone comes from source='llm' chunk_annotations only; distilled rows
    never contribute."""

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")
        import datetime
        base = datetime.datetime(2026, 1, 1, 10, 0, 0)
        rows = []
        mid = 1
        for d in range(3):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(4):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 100, 0 if from_me else 70, from_me,
                                 "chatting away here", t.isoformat()))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        # One chunk in chat 100 with an LLM annotation and a distilled one.
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(1, 100, 1, 2, '2026-01-01', '2026-01-01', 2, 'x')")
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(2, 100, 3, 4, '2026-01-01', '2026-01-01', 2, 'y')")
        con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (1, 0.8, '[\"joy\"]', 3, '[]', 'llm')")
        con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (2, -0.9, '[\"anger\"]', 3, '[]', 'distilled')")

    def test_tone_uses_llm_only(self):
        self.build()
        card = query_tools.get_voiceprint(contact_id=12)
        self.assertIn("tone", card)
        self.assertTrue(card["tone"]["low_trust"])
        # Only the llm row (valence 0.8, joy) -- the distilled -0.9/anger ignored.
        self.assertAlmostEqual(card["tone"]["valence"], 0.8, places=4)
        self.assertIn("joy", card["tone"]["emotions"])
        self.assertNotIn("anger", card["tone"]["emotions"])


class AnnotationGuardTest(_DBCase):
    """voiceprint_annotations_hash: an in-place re-annotation adds NO messages,
    so the max-message-id guard alone skipped it and served stale tone_* -- the
    hole `annotate.py --quality hybrid` walks into on every run, since its
    phase-3 upgrades are source='llm' rows written with no new messages.

    Its own class rather than a method on CoreBuildTest: that fixture
    deliberately has zero chunk_annotations (test_tone_omitted_when_no_llm_
    annotations asserts it), and MinReturnFloorTest subclasses it, so both
    would have to change to host these.
    """

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (71, '+1701', NULL, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (200, 'c200', 'Crew', 1, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")
        con.execute("INSERT INTO chat_participants VALUES (200, 70)")
        con.execute("INSERT INTO chat_participants VALUES (200, 71)")
        import datetime
        base = datetime.datetime(2026, 1, 1, 10, 0, 0)
        rows = []
        mid = 1
        for chat_id, other in ((100, 70), (200, 71)):
            for d in range(3):
                day = base + datetime.timedelta(days=d * 20)
                for i in range(4):
                    t = day + datetime.timedelta(seconds=i * 30)
                    from_me = 1 if i % 2 == 0 else 0
                    rows.append(_msg(mid, chat_id, 0 if from_me else other, from_me,
                                     "chatting away here", t.isoformat()))
                    mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        # chunks 1-2 in the 1:1 chat (one llm, one distilled), chunk 3 in the
        # GROUP chat -- which scope='group'/'all' voiceprints read, and which
        # drift's is_group=0 fingerprint would be blind to.
        for cid, chat_id in ((1, 100), (2, 100), (3, 200)):
            con.execute(
                "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
                "first_date, last_date, n_messages, text) "
                "VALUES (?, ?, 1, 2, '2026-01-01', '2026-01-01', 2, 'x')",
                (cid, chat_id))
        con.executemany(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (?, ?, ?, 3, '[]', ?)",
            [(1, 0.8, '["joy"]', "llm"),
             (2, -0.9, '["anger"]', "distilled"),
             (3, 0.4, '["joy"]', "llm")])

    def _write(self, sql, params=()):
        """Mutate through a separate connection: _DBCase.build() opens its own
        per call, so an uncommitted change on another one would be invisible."""
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def _state(self, key):
        rows = self.rows("SELECT value FROM ingest_state WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def test_upgrade_to_llm_forces_rebuild(self):
        # The exact shape of an `--quality hybrid` phase-3 / --upgrade-distilled
        # write: same chunk_id (row COUNT unchanged), no new messages.
        self.assertGreater(self.build(), 0)
        self.assertEqual(self.build(), 0)          # unchanged -> skip
        self._write("UPDATE chunk_annotations SET source = 'llm', valence = -0.9 "
                    "WHERE chunk_id = 2")
        self.assertGreater(
            self.build(), 0,
            "an upgraded annotation with no new messages must force a rebuild")
        # ...and the stored tone actually moved: mean(0.8, -0.9) = -0.05.
        tone = self.rows("SELECT tone_valence FROM person_voiceprint "
                         "WHERE scope = 'dyad' AND member_key = 'me'")[0]
        self.assertAlmostEqual(tone["tone_valence"], -0.05, places=4)

    def test_emotions_only_rewrite_forces_rebuild(self):
        # What a lossy COUNT/SUM aggregate (drift's shape) cannot see: valence,
        # depth, source and the row count all hold still.
        self.build()
        self.assertEqual(self.build(), 0)
        self._write("UPDATE chunk_annotations SET emotions_json = '[\"sadness\"]' "
                    "WHERE chunk_id = 1")
        self.assertGreater(self.build(), 0)
        tone = self.rows("SELECT tone_emotions_json FROM person_voiceprint "
                         "WHERE scope = 'dyad' AND member_key = 'me'")[0]
        self.assertIn("sadness", tone["tone_emotions_json"])

    def test_valence_swap_between_two_chunks_forces_rebuild(self):
        # The other thing a lossy aggregate cannot see: chunks 1 and 3 trade
        # valences, so COUNT, SUM(valence), SUM(depth) and the source mix are
        # all bit-identical before and after. Only a per-row digest moves.
        self.build()
        self.assertEqual(self.build(), 0)
        before = self.rows(
            "SELECT COALESCE(SUM(valence), 0) AS s, COUNT(*) AS n "
            "FROM chunk_annotations WHERE source = 'llm'")[0]
        self._write("UPDATE chunk_annotations SET valence = "
                    "CASE chunk_id WHEN 1 THEN 0.4 ELSE 0.8 END "
                    "WHERE chunk_id IN (1, 3)")
        after = self.rows(
            "SELECT COALESCE(SUM(valence), 0) AS s, COUNT(*) AS n "
            "FROM chunk_annotations WHERE source = 'llm'")[0]
        # Precondition: the aggregate really is blind to this edit.
        self.assertAlmostEqual(before["s"], after["s"], places=9)
        self.assertEqual(before["n"], after["n"])
        self.assertGreater(
            self.build(), 0,
            "a valence swap leaves every SUM identical -- only an exact "
            "per-row digest can force the rebuild")

    def test_group_chat_annotation_change_forces_rebuild(self):
        # scope='group' and scope='all' rows read group-chat annotations, so
        # the fingerprint deliberately omits drift's `ch.is_group = 0`.
        self.build()
        self.assertEqual(self.build(), 0)
        self._write("UPDATE chunk_annotations SET valence = -0.7 WHERE chunk_id = 3")
        self.assertGreater(
            self.build(), 0,
            "a group-chat annotation change must force a rebuild (scope "
            "group/all read it)")

    def test_distilled_only_change_does_not_force_rebuild(self):
        # voiceprint cannot observe distilled rows (_tone is source='llm'
        # only), so a plain `--quality distilled` pass must NOT rebuild.
        self.build()
        self.assertEqual(self.build(), 0)
        self._write("UPDATE chunk_annotations SET valence = 0.123 WHERE chunk_id = 2")
        self.assertEqual(
            self.build(), 0,
            "a distilled-only change is invisible to tone and must not rebuild")

    def test_deleting_the_chunk_forces_rebuild(self):
        # The JOIN's real job: a tail re-chunk deletes chunks rows, and
        # chunk_annotations has no FK cascade, so the orphan must leave the
        # digest the same way it leaves _tone's own join.
        self.build()
        self.assertEqual(self.build(), 0)
        self._write("DELETE FROM chunks WHERE id = 1")
        self.assertGreater(self.build(), 0)

    def test_legacy_db_without_the_hash_rebuilds_once(self):
        self.build()
        self.assertEqual(self.build(), 0)
        # A DB from before the key existed: the message-id guard is set, the
        # annotations one is not.
        self._write("DELETE FROM ingest_state WHERE key = "
                    "'voiceprint_annotations_hash'")
        self.assertGreater(self.build(), 0, "a missing hash must rebuild once")
        self.assertIsNotNone(self._state("voiceprint_annotations_hash"))
        self.assertEqual(self.build(), 0, "and then guard on both keys")

    def test_dry_run_writes_no_hash(self):
        # --dry-run promises no writes; poisoning the guard would be the worst
        # kind, since the next real run would then skip.
        self.build(dry_run=True)
        self.assertIsNone(self._state("voiceprint_annotations_hash"))
        self.assertIsNone(self._state("voiceprint_max_message_id"))
        self.assertGreater(self.build(), 0)

    def test_hash_is_deterministic_and_llm_scoped(self):
        con = self._con()
        first = voiceprint._annotations_hash(con)
        self.assertEqual(first, voiceprint._annotations_hash(con))
        self._write("UPDATE chunk_annotations SET source = 'llm' WHERE chunk_id = 2")
        self.assertNotEqual(first, voiceprint._annotations_hash(self._con()))

    def test_hash_ignores_the_connection_row_factory(self):
        # Pins the load-bearing tuple() in repr(tuple(row)). build_voiceprints
        # takes a caller-supplied connection, and db.connect defaults to
        # rows=True; under sqlite3.Row a bare repr(row) is
        # "<sqlite3.Row object at 0x...>", so the digest would be
        # address-dependent -- different every process, rebuilding every run,
        # and no other test would notice.
        plain = sqlite3.connect(self.db_path)
        self.addCleanup(plain.close)
        as_rows = sqlite3.connect(self.db_path)
        as_rows.row_factory = sqlite3.Row
        self.addCleanup(as_rows.close)
        self.assertEqual(voiceprint._annotations_hash(plain),
                         voiceprint._annotations_hash(as_rows))


class DateScopedSignatureAndToneTest(_DBCase):
    """openers/closers (_phrase_counts) and tone (_tone) must honor the date
    window -- they previously leaked the whole corpus into a windowed voiceprint.
    Fixtures build sessions + chunk_annotations both inside and outside the
    window (unlike DateScopedTest, which builds neither, so it can't see this)."""

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")
        import datetime
        rows, mid = [], 1
        # 3 in-window (Jan) + 3 out-of-window (Jun) sessions, days apart (> the
        # 60-min gap) so each day is its own session. Each session: me opens, Mom
        # replies, me closes -> me is both initiator and ender. 3 repeats clears
        # the _SIGNATURE_MIN_COUNT=3 floor for each opener/closer phrase.
        for month, opener, closer in ((1, "alpha start here", "omega end alpha"),
                                      (6, "zulu start here", "omega end zulu")):
            for day in (3, 6, 9):
                base = datetime.datetime(2026, month, day, 10, 0, 0)
                rows.append(_msg(mid, 100, 0, 1, opener, base.isoformat()))
                mid += 1
                rows.append(_msg(mid, 100, 70, 0, "ok thanks mom",
                                 (base + datetime.timedelta(seconds=30)).isoformat()))
                mid += 1
                rows.append(_msg(mid, 100, 0, 1, closer,
                                 (base + datetime.timedelta(seconds=60)).isoformat()))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        _quiet(sessions.build_sessions, con)
        # Chunks + llm annotations: A in-window, B out-of-window, C straddles the
        # LOWER bound (starts BEFORE, ends INSIDE) -> excluded by the first_date rule.
        con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES (?, 100, ?, ?, ?, ?, 2, 'x')",
            [(1, 1, 3, "2026-01-05", "2026-01-05"),
             (2, 10, 12, "2026-06-05", "2026-06-05"),
             (3, 1, 2, "2025-12-20", "2026-01-10")])
        con.executemany(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (?, ?, ?, 3, '[]', 'llm')",
            [(1, 0.8, '["joy"]'), (2, -0.9, '["anger"]'), (3, 0.2, '["fear"]')])

    def _vp(self, **kw):
        return voiceprint.compute_voiceprint(self._con(), "me", [100], **kw)

    def test_openers_closers_windowed_to_in_range_only(self):
        r = self._vp(date_from="2026-01-01", date_to="2026-02-01")
        self.assertEqual({o["phrase"] for o in r["openers"]}, {"alpha start here"})
        self.assertEqual({c["phrase"] for c in r["closers"]}, {"omega end alpha"})

    def test_full_corpus_openers_closers_unchanged(self):
        r = self._vp()  # no window -> both windows present (byte-for-byte old behavior)
        self.assertEqual({o["phrase"] for o in r["openers"]},
                         {"alpha start here", "zulu start here"})
        self.assertEqual({c["phrase"] for c in r["closers"]},
                         {"omega end alpha", "omega end zulu"})

    def test_tone_windowed_and_straddle_excluded_by_first_date(self):
        r = self._vp(date_from="2026-01-01", date_to="2026-02-01")
        self.assertAlmostEqual(r["tone_valence"], 0.8, places=4)  # only chunk A
        self.assertEqual(r["tone_emotions"], ["joy"])
        self.assertNotIn("anger", r["tone_emotions"])  # chunk B out of window
        self.assertNotIn("fear", r["tone_emotions"])   # chunk C starts before df


class DateScopedTest(_DBCase):
    """Date-scoped get_voiceprint recomputes over a capped raw scan."""

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")
        import datetime
        base = datetime.datetime(2026, 1, 1, 10, 0, 0)
        rows = []
        mid = 1
        for d in range(6):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(6):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 100, 0 if from_me else 70, from_me,
                                 "message content here", t.isoformat()))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def test_scan_cap_returns_narrow_note(self):
        self.build()
        saved = config.VOICEPRINT_SCAN_MAX_MESSAGES
        config.VOICEPRINT_SCAN_MAX_MESSAGES = 3
        self.addCleanup(setattr, config, "VOICEPRINT_SCAN_MAX_MESSAGES", saved)
        res = query_tools.get_voiceprint(
            contact_id=12, date_from="2026-01-01", date_to="2027-01-01")
        self.assertFalse(res.get("recomputed", True))
        self.assertIn("note", res)
        self.assertGreater(res["messages_in_range"], 3)

    def test_date_scoped_recompute_returns_card(self):
        self.build()
        res = query_tools.get_voiceprint(
            contact_id=12, date_from="2026-01-01", date_to="2026-02-01")
        self.assertTrue(res["recomputed"])
        self.assertGreater(res["n_messages"], 0)
        self.assertIn("features", res)


class ResetDbTest(_DBCase):
    """reset_db clears person_voiceprint (and the guard in ingest_state)."""

    def populate(self, con):
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', NULL, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")
        con.execute(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (1, 'g1', 100, 70, 0, 'hi', "
            "'2026-01-01T10:00:00', 0)")

    def test_reset_db_clears_person_voiceprint(self):
        self.build()
        self.assertGreater(
            self.rows("SELECT COUNT(*) AS n FROM person_voiceprint")[0]["n"], 0)
        # reset_db binds DB_PATH at import; point it (and argv) at our temp DB.
        from pathlib import Path
        saved_path, saved_argv = reset_db.DB_PATH, sys.argv
        reset_db.DB_PATH = Path(self.db_path)
        sys.argv = ["reset_db", "-f"]
        self.addCleanup(setattr, reset_db, "DB_PATH", saved_path)
        self.addCleanup(setattr, sys, "argv", saved_argv)
        _quiet(reset_db.main)

        self.assertEqual(
            self.rows("SELECT COUNT(*) AS n FROM person_voiceprint")[0]["n"], 0)
        # The guard is wiped too (ingest_state cleared) -> clean rebuild next run.
        self.assertEqual(
            self.rows("SELECT COUNT(*) AS n FROM ingest_state "
                      "WHERE key = 'voiceprint_max_message_id'")[0]["n"], 0)


class DraftReplyTest(_DBCase):
    """V2 draft_reply: reply-in-context pairing, the structural decline below
    VOICEPRINT_MIN_CONTEXT_PAIRS (zero generation calls), and the in-tool
    dyad_terms post-check. Ollama is always mocked -- no network."""

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")

        def msg(mid, chunk, from_me, text, day, is_emote=0):
            date = f"2026-0{day}-01T10:00:{mid % 60:02d}"
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
                "text, date, has_attachment, is_emote, item_type, chunk_id) "
                "VALUES (?, ?, 100, ?, ?, ?, ?, 0, ?, 0, ?)",
                (mid, f"g{mid}", 0 if from_me else 70, from_me, text, date,
                 is_emote, chunk),
            )

        # chunk 1 (month 03, most recent): the "dinner" retrieval fixture --
        # 4 real (their message -> my reply) pairs, all matchable by FTS on
        # the incoming's "dinner".
        c1 = [
            (1, 0, "are we still on for dinner"),
            (2, 1, "yes dinner sounds great"),
            (3, 0, "what time dinner"),
            (4, 1, "seven works for dinner"),
            (5, 0, "where should we dinner"),
            (6, 1, "the usual dinner spot"),
            (7, 0, "cool see you at dinner"),
            (8, 1, "cant wait for dinner"),
        ]
        for mid, from_me, text in c1:
            msg(mid, 1, from_me, text, day=3)

        # chunk 2 (month 02): a hand-built stream for the pairing rigor test.
        # Real order: them aaa, me bbb, them ccc, them ddd, me eee, me fff,
        # them ggg. Only aaa->bbb and ddd->eee are true adjacency pairs; a
        # naive inbound x self co-occurrence would wrongly yield 4x3=12.
        c2 = [
            (11, 0, "aaa"), (12, 1, "bbb"), (13, 0, "ccc"), (14, 0, "ddd"),
            (15, 1, "eee"), (16, 1, "fff"), (17, 0, "ggg"),
        ]
        for mid, from_me, text in c2:
            msg(mid, 2, from_me, text, day=2)

        # chunk 3 (month 01): emote transparency. Real stream after dropping the
        # emote rows: them hello, me hi_back, me yo. Emotes (21 from me, 23 from
        # them) can neither serve as a reply nor anchor one -> exactly one pair
        # (hello -> hi_back); yo follows a self message so it pairs with nothing.
        msg(31, 3, 0, "hello", day=1)
        msg(21, 3, 1, "Loved a message", day=1, is_emote=1)
        msg(32, 3, 1, "hi back", day=1)
        msg(23, 3, 0, "Liked a message", day=1, is_emote=1)
        msg(33, 3, 1, "yo", day=1)

        con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES (?, 100, ?, ?, ?, ?, ?, 'x')",
            [(1, 1, 8, "2026-03-01", "2026-03-01", 8),
             (2, 11, 17, "2026-02-01", "2026-02-01", 7),
             (3, 31, 33, "2026-01-01", "2026-01-01", 5)],
        )
        # dyad_terms: "wombat" belongs to THIS dyad (grounded, allowed);
        # "floopy" belongs to another chat (an inside-joke universe member that
        # is NOT grounded in this dyad -> must be stripped from a draft).
        con.executemany(
            "INSERT INTO dyad_terms(chat_id, term) VALUES (?, ?)",
            [(100, "wombat"), (999, "floopy")],
        )

    # --- reply-in-context pairing (vs naive co-occurrence) ------------------
    def test_reply_pairs_only_true_adjacency(self):
        con = self._con()
        pairs = persona.reply_pairs_in_chunk(con, 2)
        # Only the two adjacency pairs -- never the 12 naive inbound x self combos.
        self.assertEqual([(p["incoming"], p["reply"]) for p in pairs],
                         [("aaa", "bbb"), ("ddd", "eee")])

    def test_emote_rows_neither_anchor_nor_serve(self):
        con = self._con()
        pairs = persona.reply_pairs_in_chunk(con, 3)
        # The emote between hello and hi_back is transparent (the pair still
        # forms); the emote before yo can't anchor, so yo pairs with nothing.
        self.assertEqual([(p["incoming"], p["reply"]) for p in pairs],
                         [("hello", "hi back")])

    # --- structural decline below the per-call floor ------------------------
    def test_below_floor_declines_with_zero_generation_calls(self):
        # 4 similar pairs available, floor raised to 5 -> low, no LLM call.
        saved = config.VOICEPRINT_MIN_CONTEXT_PAIRS
        config.VOICEPRINT_MIN_CONTEXT_PAIRS = 5
        self.addCleanup(setattr, config, "VOICEPRINT_MIN_CONTEXT_PAIRS", saved)

        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.draft_reply(
                "hey want to grab dinner tonight", contact_id=12)
        chat_once.assert_not_called()  # STRUCTURAL: no generation attempt at all
        self.assertEqual(res["confidence"], "low")
        self.assertIsNone(res["draft"])
        self.assertEqual(res["n_context_pairs"], 4)
        # The labeled "recent replies" pack is returned instead.
        self.assertTrue(res["exemplars"])
        self.assertTrue(all("reply" in ex and "incoming" in ex
                            for ex in res["exemplars"]))

    # --- generation path + dyad_terms post-check ----------------------------
    def test_high_confidence_generates_and_bundles_exemplars(self):
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "sure sounds good"}) as chat_once:
            res = query_tools.draft_reply(
                "hey want to grab dinner tonight", contact_id=12)
        chat_once.assert_called_once()
        self.assertEqual(res["confidence"], "high")
        self.assertEqual(res["draft"], "sure sounds good")
        self.assertEqual(res["n_context_pairs"], 4)
        self.assertLessEqual(len(res["exemplars"]), 8)
        self.assertTrue(res["exemplars"])

    def test_exemplars_truncated_to_text_preview_chars(self):
        from query_tools._shared import TEXT_PREVIEW_CHARS
        # A long inbound in chunk 1 must be clipped in the returned exemplar.
        con = sqlite3.connect(self.db_path)
        con.execute("UPDATE messages SET text = ? WHERE id = 1",
                    ("dinner " + "x" * (TEXT_PREVIEW_CHARS + 50),))
        con.commit()
        con.close()
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "ok"}):
            res = query_tools.draft_reply("dinner plans", contact_id=12)
        for ex in res["exemplars"]:
            self.assertLessEqual(len(ex["incoming"]), TEXT_PREVIEW_CHARS + 1)
            self.assertLessEqual(len(ex["reply"]), TEXT_PREVIEW_CHARS + 1)

    def test_postcheck_strips_ungrounded_inside_joke(self):
        # The model emits a grounded term (wombat, this dyad's) and an
        # ungrounded one (floopy, another chat's). Only floopy is stripped.
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "sounds good wombat floopy"}):
            res = query_tools.draft_reply(
                "hey want to grab dinner tonight", contact_id=12)
        self.assertIn("wombat", res["draft"])
        self.assertNotIn("floopy", res["draft"])
        self.assertTrue(res["notes"])
        self.assertIn("floopy", " ".join(res["notes"]))

    def test_declines_when_strip_leaves_a_fragment(self):
        # F7: when ungrounded-term stripping removes so much that only a
        # degenerate stub remains, draft_reply DECLINES rather than sending it.
        # "floopy" (another chat's term) is fully stripped -> nothing survives.
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "floopy"}):
            res = query_tools.draft_reply(
                "hey want to grab dinner tonight", contact_id=12)
        self.assertIsNone(res["draft"])
        self.assertEqual(res["confidence"], "low")
        self.assertTrue(res["notes"])

    # --- input / scope guards -----------------------------------------------
    def test_requires_incoming_text(self):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.draft_reply("   ", contact_id=12)
        chat_once.assert_not_called()
        self.assertIn("error", res)

    def test_requires_a_person(self):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.draft_reply("hello there")
        chat_once.assert_not_called()
        self.assertIn("error", res)


class CheckStyleMatchTest(_DBCase):
    """V3 check_style_match: deterministic LSM/feature distance of a user-written
    draft to a stored person_voiceprint row, with rule-based top-3 divergences.
    No model in the loop -- nothing is mocked because nothing calls Ollama.

    Fixture: I text Mom (contact 12) and Al (contact 13) in ALL-lowercase, no
    '!' anywhere; Mom texts back with Capitalized words. So across dyad rows
    lowercase_ratio VARIES (std > 0 -> comparable) while exclaim_rate is constant
    0 (std 0 -> omitted from feature_std_json, and skipped by the matcher)."""

    def populate(self, con):
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
            "VALUES (?, ?, ?, ?)",
            [(12, "u12", "Mom", None), (13, "u13", "Al", "B")],
        )
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(70, "+1700", 12), (71, "+1710", 13)],
        )
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, 'iMessage')",
            [(100, "c100", None, 0), (101, "c101", None, 0)],
        )
        con.executemany("INSERT INTO chat_participants VALUES (?, ?)",
                        [(100, 70), (101, 71)])
        import datetime
        base = datetime.datetime(2026, 1, 1, 10, 0, 0)
        rows = []
        mid = 1
        # me + Mom: me lowercase, Mom Capitalized -> lowercase_ratio varies.
        for d in range(5):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(6):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 100, 0 if from_me else 70, from_me,
                                 "hey how are you doing today" if from_me
                                 else "I Am Doing Well Thank You", t.isoformat()))
                mid += 1
        # me + Al: shorter messages -> avg_msg_len varies across dyad rows.
        for d in range(4):
            day = base + datetime.timedelta(days=d * 25)
            for i in range(4):
                t = day + datetime.timedelta(seconds=i * 20)
                from_me = 1 if i % 2 == 0 else 0
                rows.append(_msg(mid, 101, 0 if from_me else 71, from_me,
                                 "yo" if from_me else "ok", t.isoformat()))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        _quiet(lexicon_build.build_person_period_lexicon, con)
        _quiet(sessions.build_sessions, con)

    def _std_map(self):
        import json
        return json.loads(self.rows(
            "SELECT feature_std_json FROM person_voiceprint WHERE member_key='me' "
            "AND counterpart_key='contact:12' AND scope='dyad'")[0]["feature_std_json"])

    def test_deterministic_same_inputs_same_output(self):
        self.build()
        draft = "WHY ARE YOU DOING THIS"
        a = query_tools.check_style_match(draft, contact_id=12)
        b = query_tools.check_style_match(draft, contact_id=12)
        self.assertEqual(a, b)  # byte-identical: no model, no randomness

    def test_distance_and_top3_are_rule_based(self):
        self.build()
        res = query_tools.check_style_match("WHY ARE YOU DOING THIS", contact_id=12)
        self.assertIsInstance(res["distance"], float)
        self.assertGreater(res["n_features_compared"], 0)
        self.assertLessEqual(len(res["top_divergences"]), 3)
        # Every divergence is rule-based: a feature name, a z gap, and a
        # human explanation string (no LLM critique).
        for d in res["top_divergences"]:
            self.assertIn("feature", d)
            self.assertIsInstance(d["z"], float)
            self.assertIsInstance(d["explanation"], str)
            self.assertTrue(d["explanation"])
        # Sorted by z descending.
        zs = [d["z"] for d in res["divergences"]]
        self.assertEqual(zs, sorted(zs, reverse=True))

    def test_std_zero_feature_skipped_no_divide_by_zero(self):
        self.build()
        std = self._std_map()
        # exclaim_rate is constant 0 across the corpus -> omitted from the std map.
        self.assertNotIn("exclaim_rate", std)
        # A draft loaded with '!' still never surfaces exclaim_rate (skipped,
        # and no ZeroDivisionError is raised).
        res = query_tools.check_style_match("hey!!! what!!! now!!!", contact_id=12)
        feats = {d["feature"] for d in res["divergences"]}
        self.assertNotIn("exclaim_rate", feats)
        # lowercase_ratio DOES vary in this fixture -> it is comparable.
        self.assertIn("lowercase_ratio", std)

    def test_all_caps_draft_diverges_on_capitalization(self):
        self.build()
        # I text Mom in all-lowercase; an all-caps draft must flag capitalization.
        res = query_tools.check_style_match(
            "WHY WOULD YOU EVER SAY THAT TO ME", contact_id=12)
        lc = next((d for d in res["divergences"]
                   if d["feature"] == "lowercase_ratio"), None)
        self.assertIsNotNone(lc)
        self.assertIn("capitalization", lc["explanation"])
        # The draft's own lowercase_ratio (all caps -> 0) is far below my typical.
        self.assertLess(res["draft_features"]["lowercase_ratio"], 0.1)

    def test_uses_stored_std_never_rescans_corpus(self):
        self.build()
        # The read path must NOT recompute a voiceprint: make the corpus-scan
        # entry points explode. _msg_style (a pure helper) is left intact.
        with mock.patch.object(voiceprint, "compute_voiceprint",
                               side_effect=AssertionError("rescanned the corpus")), \
             mock.patch.object(voiceprint, "member_chats",
                               side_effect=AssertionError("rescanned the corpus")), \
             mock.patch.object(voiceprint, "scan_size",
                               side_effect=AssertionError("rescanned the corpus")):
            res = query_tools.check_style_match("hey how are you", contact_id=12)
        self.assertIsInstance(res["distance"], float)
        self.assertGreater(res["n_features_compared"], 0)

    def test_subject_them_targets_their_voiceprint(self):
        self.build()
        res = query_tools.check_style_match(
            "hey how are you", contact_id=12, subject="them")
        self.assertEqual(res["member_key"], "contact:12")
        self.assertEqual(res["counterpart_key"], "me")

    def test_missing_draft_and_missing_person_error(self):
        self.build()
        self.assertIn("error", query_tools.check_style_match("   ", contact_id=12))
        self.assertIn("error", query_tools.check_style_match("hi there", scope="dyad"))

    def test_missing_voiceprint_row_returns_note(self):
        self.build()
        res = query_tools.check_style_match("hi there", contact_id=99999)
        self.assertFalse(res.get("found", True))
        self.assertIn("note", res)


# --- V5 (RES-69): simulate_contact + past-self ------------------------------


class SimulateContactTest(_DBCase):
    """V5 simulate_contact for a named contact: the code-applied per-line estimate
    label, the block-list hard-refuse, the acknowledged UX gate + soft dormant
    warning, the structural decline below the floor, and the reaction base-rate
    tapback prediction. Ollama is always mocked -- no network.

    Fixture: chat 100 (me + Mom, contact 12). chunk 1 holds 4 (my message -> her
    reply) adjacency pairs, all matchable by FTS on the incoming's 'dinner'. Mom
    sent tapbacks on my messages (3 love, 1 laugh) over her 4 real messages."""

    def populate(self, con):
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                    "VALUES (12, 'u12', 'Mom')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (70, '+1700', 12, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (100, 'c100', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (100, 70)")

        def msg(mid, chunk, from_me, text, date):
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
                "text, date, has_attachment, is_emote, item_type, chunk_id) "
                "VALUES (?, ?, 100, ?, ?, ?, ?, 0, 0, 0, ?)",
                (mid, f"g{mid}", 0 if from_me else 70, from_me, text, date, chunk))

        # chunk 1: 4 (me -> Mom) adjacency pairs, matchable by FTS 'dinner'.
        c1 = [
            (1, 1, "are we still on for dinner"),
            (2, 0, "yes dinner sounds great"),
            (3, 1, "what time dinner"),
            (4, 0, "seven works for dinner"),
            (5, 1, "where should we dinner"),
            (6, 0, "the usual dinner spot"),
            (7, 1, "cool see you at dinner"),
            (8, 0, "cant wait for dinner"),
        ]
        for mid, from_me, text in c1:
            msg(mid, 1, from_me, text, f"2026-03-01T10:00:{mid:02d}")
        con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(1, 100, 1, 8, '2026-03-01', '2026-03-01', 8, 'x')")
        # Mom's tapbacks on my messages: 3 love (2000), 1 laugh (2003).
        con.executemany(
            "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
            "is_from_me, associated_message_type, date) "
            "VALUES (?, ?, 100, 70, 0, ?, '2026-03-01T10:05:00')",
            [(901, "g1", 2000), (902, "g3", 2000), (903, "g5", 2000),
             (904, "g7", 2003)])

    def test_every_simulation_line_carries_the_estimate_label(self):
        # Multi-line generation: EVERY non-empty returned line must be labeled in
        # code (post-generation), so the outer model can only relay a pre-labeled
        # artifact -- there is no unlabeled version to strip.
        with mock.patch.object(
                ollama, "chat_once",
                return_value={"content": "sure sounds good\nsee you at seven\n"}):
            res = query_tools.simulate_contact(
                "want to grab dinner tonight", contact_id=12, acknowledged=True)
        self.assertTrue(res["generated"])
        self.assertEqual(res["confidence"], "high")
        sim_lines = [ln for ln in res["simulation"].split("\n") if ln.strip()]
        self.assertTrue(sim_lines)
        for ln in sim_lines:
            self.assertIn("ESTIMATE", ln)
        self.assertIn("disclaimer", res)

    def test_empty_generation_declines_instead_of_blank_estimate(self):
        # An empty model reply must degrade to a low-confidence decline, never a
        # generated:true / confidence:high payload carrying a blank simulation.
        with mock.patch.object(ollama, "chat_once", return_value={"content": "   "}):
            res = query_tools.simulate_contact(
                "want to grab dinner tonight", contact_id=12, acknowledged=True)
        self.assertFalse(res["generated"])
        self.assertEqual(res["confidence"], "low")
        self.assertIsNone(res["simulation"])
        self.assertTrue(res["notes"])

    def test_blocked_contact_hard_refuses_without_generating(self):
        saved = config.PERSONA_SIM_BLOCK_IDS
        config.PERSONA_SIM_BLOCK_IDS = "12"
        self.addCleanup(setattr, config, "PERSONA_SIM_BLOCK_IDS", saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact(
                "hey", contact_id=12, acknowledged=True)
        chat_once.assert_not_called()  # never generates for a blocked contact
        self.assertTrue(res["refused"])
        self.assertEqual(res["reason"], "blocked_contact")
        self.assertFalse(res["generated"])
        self.assertIn("turned off", res["message"])

    def test_blocked_bare_handle_hard_refuses_without_generating(self):
        # A person that exists ONLY as an unlinked handle (no contacts row) must
        # still be denylistable, via a handle:N key -- the gap this closes.
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("INSERT INTO handles(id, handle_identifier, contact_id, "
                        "service) VALUES (71, '+1701', NULL, 'iMessage')")
            con.commit()
        finally:
            con.close()
        saved = config.PERSONA_SIM_BLOCK_IDS
        config.PERSONA_SIM_BLOCK_IDS = "handle:71"
        self.addCleanup(setattr, config, "PERSONA_SIM_BLOCK_IDS", saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact(
                "hey", handle_id=71, acknowledged=True)
        chat_once.assert_not_called()  # never generates for a blocked bare handle
        self.assertTrue(res["refused"])
        self.assertEqual(res["reason"], "blocked_contact")
        self.assertFalse(res["generated"])
        self.assertIn("turned off", res["message"])

    def test_living_contact_requires_acknowledged(self):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact("want dinner", contact_id=12)
        chat_once.assert_not_called()  # no generation before acknowledgment
        self.assertTrue(res["acknowledgment_required"])
        self.assertFalse(res["generated"])
        # A fresh (non-dormant) dyad surfaces no deceased warning.
        self.assertNotIn("deceased_warning", res)

    def test_below_floor_declines_structurally(self):
        saved = config.VOICEPRINT_MIN_CONTEXT_PAIRS
        config.VOICEPRINT_MIN_CONTEXT_PAIRS = 5  # 4 dinner pairs available
        self.addCleanup(setattr, config, "VOICEPRINT_MIN_CONTEXT_PAIRS", saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact(
                "want to grab dinner tonight", contact_id=12, acknowledged=True)
        chat_once.assert_not_called()  # STRUCTURAL: no generation attempt
        self.assertEqual(res["confidence"], "low")
        self.assertIsNone(res["simulation"])
        self.assertEqual(res["n_context_pairs"], 4)
        self.assertTrue(res["exemplars"])

    def test_reaction_base_rate_tapback_prediction(self):
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "ok"}):
            res = query_tools.simulate_contact(
                "want dinner", contact_id=12, acknowledged=True)
        tb = res["likely_tapback"]
        # Full-message denominator = Mom's 4 real messages (NOT the reacted ones).
        self.assertEqual(tb["denominator_messages"], 4)
        self.assertEqual(tb["most_likely"], "love")
        self.assertAlmostEqual(tb["base_rates"]["love"], 0.75, places=4)
        self.assertAlmostEqual(tb["base_rates"]["laugh"], 0.25, places=4)

    def test_soft_deceased_warning_in_acknowledgment_payload(self):
        # Mom goes quiet after 2026-03-01 and I keep texting into silence.
        con = self._con()
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment, is_emote, item_type) VALUES "
            "(?, ?, 100, 0, 1, ?, ?, 0, 0, 0)",
            [(50, "g50", "you there?", "2026-06-01T10:00:00"),
             (51, "g51", "miss you", "2026-06-15T10:00:00"),
             (52, "g52", "thinking of you", "2026-07-01T10:00:00")])
        con.commit()
        for var, val in (("PERSONA_SIM_DORMANT_DAYS", 30),
                         ("PERSONA_SIM_ONESIDED_MIN_MSGS", 3)):
            saved = getattr(config, var)
            setattr(config, var, val)
            self.addCleanup(setattr, config, var, saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact("hey", contact_id=12)
        chat_once.assert_not_called()
        self.assertTrue(res["acknowledgment_required"])
        warn = res["deceased_warning"]
        self.assertEqual(warn["your_messages_since_with_no_reply"], 3)
        self.assertGreaterEqual(warn["days_since_their_last_message"], 30)

    def test_requires_message_and_person(self):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            self.assertIn("error", query_tools.simulate_contact("  ", contact_id=12))
            self.assertIn("error", query_tools.simulate_contact(
                "hi", acknowledged=True))  # no person
        chat_once.assert_not_called()


class SimulatePastSelfTest(_DBCase):
    """V5 past-self: subject='me' + as_of_date via a bounded date-scoped raw
    rescan, gated at VOICEPRINT_MIN_RANGE_MESSAGES per range and capped at
    VOICEPRINT_SCAN_MAX_MESSAGES. Ollama mocked."""

    def populate(self, con):
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                    "VALUES (90, '+1900', NULL, 'iMessage')")
        con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                    "service) VALUES (300, 'c300', NULL, 0, 'iMessage')")
        con.execute("INSERT INTO chat_participants VALUES (300, 90)")
        import datetime
        base = datetime.datetime(2026, 1, 1, 9, 0, 0)
        rows = []
        mid = 1
        # them, me, them, me per month across 3 months -> (them -> me) pairs.
        for d in range(3):
            day = base + datetime.timedelta(days=d * 20)
            for i in range(4):
                t = day + datetime.timedelta(seconds=i * 30)
                from_me = 1 if i % 2 == 1 else 0
                rows.append(_msg(mid, 300, 0 if from_me else 90, from_me,
                                 "how are you doing today" if from_me
                                 else "not too bad you", t.isoformat()))
                mid += 1
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def _low_floors(self):
        for var, val in (("VOICEPRINT_MIN_RANGE_MESSAGES", 4),
                         ("VOICEPRINT_MIN_CONTEXT_PAIRS", 2)):
            saved = getattr(config, var)
            setattr(config, var, val)
            self.addCleanup(setattr, config, var, saved)

    def test_past_self_generates_labeled_estimate(self):
        self._low_floors()
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": "hanging in there"}) as chat_once:
            res = query_tools.simulate_contact(
                "how are you?", subject="me", as_of_date="2026-06-01")
        chat_once.assert_called_once()
        self.assertTrue(res["generated"])
        self.assertEqual(res["subject"], "me")
        for ln in [x for x in res["simulation"].split("\n") if x.strip()]:
            self.assertIn("ESTIMATE", ln)

    def test_past_self_needs_as_of_date(self):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact("hi", subject="me")
        chat_once.assert_not_called()
        self.assertIn("error", res)

    def test_past_self_scan_cap_refuses(self):
        saved = config.VOICEPRINT_SCAN_MAX_MESSAGES
        config.VOICEPRINT_SCAN_MAX_MESSAGES = 2
        self.addCleanup(setattr, config, "VOICEPRINT_SCAN_MAX_MESSAGES", saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact(
                "hi", subject="me", as_of_date="2026-06-01")
        chat_once.assert_not_called()
        self.assertFalse(res["generated"])
        self.assertIn("scan cap", res["note"])
        self.assertGreater(res["messages_in_range"], 2)

    def test_past_self_below_min_range_declines(self):
        saved = config.VOICEPRINT_MIN_RANGE_MESSAGES
        config.VOICEPRINT_MIN_RANGE_MESSAGES = 1000
        self.addCleanup(setattr, config, "VOICEPRINT_MIN_RANGE_MESSAGES", saved)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = query_tools.simulate_contact(
                "hi", subject="me", as_of_date="2026-06-01")
        chat_once.assert_not_called()
        self.assertFalse(res["generated"])
        self.assertIn("too little", res["note"])


class SimulateRegistrationGateTest(unittest.TestCase):
    """simulate_contact's schema is registered ONLY when PERSONA_SIM_ENABLED; that
    registration gate -- not the per-call acknowledged param -- is the security
    boundary. Reloads tool_schemas + chat with the flag toggled (mirrors
    test_conflict.TraceRegistrationGateTest)."""

    NAME = "simulate_contact"

    def setUp(self):
        self._saved = importlib.import_module("config").PERSONA_SIM_ENABLED
        self.addCleanup(self._reload, self._saved)

    @staticmethod
    def _reload(enabled):
        importlib.import_module("config").PERSONA_SIM_ENABLED = enabled
        importlib.reload(tool_schemas)
        importlib.reload(chat)

    def test_disabled_is_default_and_registers_nothing(self):
        self._reload(False)
        self.assertFalse(importlib.import_module("config").PERSONA_SIM_ENABLED)
        self.assertTrue(callable(query_tools.simulate_contact))  # still importable
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertNotIn(self.NAME, names)
        self.assertNotIn(self.NAME, chat.TOOL_FUNCTIONS)
        self.assertNotIn(self.NAME, chat.SYSTEM_PROMPT)

    def test_enabled_registers_schema_dispatch_and_prompt(self):
        self._reload(True)
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertIn(self.NAME, names)
        self.assertIn(self.NAME, chat.TOOL_FUNCTIONS)
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertIn(self.NAME, ollama_names)
        self.assertIn(self.NAME, chat.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
