"""Hermetic tests for the V1 voiceprint core (RES-50).

Covers voiceprint.py's person_voiceprint build + guard, the query_tools
get_voiceprint read tool, and reset_db clearing the table. A hand-scripted
temp SQLite DB with known messages -- no Ollama, no real chat.db. Feature
values that matter (burst_size, reply pairs, key semantics, the per-scope std
map) are hand-computed and pinned.
"""

import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import config
import lexicon_build
import ollama
import query_tools
import reset_db
import schema
import sessions
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


if __name__ == "__main__":
    unittest.main()
