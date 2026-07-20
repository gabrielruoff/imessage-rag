"""sessions.py + query_tools/analytics_time.py: the sessions pipeline
(splitting, incremental per-chat rebuild, gap-change forced rebuild) and the
two time-pattern tools, all against a hand-scripted fixture DB with known
dates so every aggregate has one right answer. The local-timezone resolver is
pinned to UTC so hour/weekday expectations are deterministic on any machine.
Hermetic: temp sqlite file, no Ollama, no real chat.db."""

import contextlib
import io
import os
import sqlite3
import statistics
import tempfile
import unittest
import zoneinfo
from unittest import mock

import query_tools
import sessions as sessions_pipeline
from db import get_state, load_vec_extension
from query_tools import analytics_time
from schema import create_schema

UTC = zoneinfo.ZoneInfo("UTC")


def seed_base(con):
    """Directory rows shared by every fixture: Bob (contact 5, duplicate
    handles 60+61), Carol (handle 62, no contact row); chats 10/12 = 1:1 with
    Bob (split-thread shape), 11 = group, 13 = 1:1 with Carol."""
    create_schema(con)
    con.execute(
        "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
        "nickname, organization) VALUES (5, 'uid-5', 'Bob', 'Jones', NULL, NULL)"
    )
    con.executemany(
        "INSERT INTO handles(id, handle_identifier, contact_id, service) "
        "VALUES (?, ?, ?, 'iMessage')",
        [(60, "+15550001111", 5), (61, "bob@example.com", 5),
         (62, "+15550002222", None)],
    )
    con.executemany(
        "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
        "VALUES (?, ?, ?, ?, 'iMessage')",
        [(10, "chat10", None, 0), (11, "chat11", "The Group", 1),
         (12, "chat12", None, 0), (13, "chat13", None, 0)],
    )
    con.executemany(
        "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
        [(10, 60), (11, 61), (11, 62), (12, 60), (13, 62)],
    )


def add_msg(con, mid, chat, handle, from_me, text, date, date_read=None,
            has_attachment=0, is_emote=0, item_type=0):
    con.execute(
        "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
        "date, date_read, has_attachment, is_emote, item_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, f"guid-{mid}", chat, handle, from_me, text, date, date_read,
         has_attachment, is_emote, item_type),
    )


def build_sessions_quiet(con, gap_minutes=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return sessions_pipeline.build_sessions(con, gap_minutes)


class FixtureBase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        self.con = sqlite3.connect(self.db_path)
        load_vec_extension(self.con)
        self.addCleanup(self.con.close)
        seed_base(self.con)
        self.seed()
        self.con.commit()

        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)

        tz_patch = mock.patch.object(analytics_time, "_local_tz", lambda: UTC)
        tz_patch.start()
        self.addCleanup(tz_patch.stop)

    def seed(self):
        pass


class BuildSessionsTest(FixtureBase):
    """chat 10: two sessions (60-min gap default) with non-renderable rows
    interleaved; chat 12: one single-message session."""

    def seed(self):
        c = self.con
        add_msg(c, 1, 10, 60, 0, "hey", "2026-03-01T10:00:00")
        # attachment-only (text NULL): still renderable
        add_msg(c, 2, 10, 0, 1, None, "2026-03-01T10:20:00", has_attachment=1)
        add_msg(c, 3, 10, 0, 1, "back now", "2026-03-01T13:00:00")  # 160 min gap
        add_msg(c, 4, 10, 60, 0, "cool", "2026-03-01T13:30:00")
        add_msg(c, 5, 12, 60, 0, "sms fallback", "2026-03-05T09:00:00")
        # Non-renderable rows inside session 1's window: must not count,
        # split, or bridge anything.
        add_msg(c, 6, 10, 60, 0, "Loved a message", "2026-03-01T10:05:00", is_emote=1)
        add_msg(c, 7, 10, 60, 0, "named the group", "2026-03-01T10:06:00", item_type=2)
        add_msg(c, 8, 10, 60, 0, None, "2026-03-01T10:07:00")  # no text, no attachment
        add_msg(c, 9, 10, 60, 0, "dateless", None)

    def sessions_for(self, chat_id):
        return self.con.execute(
            "SELECT * FROM sessions WHERE chat_id = ? ORDER BY first_date",
            (chat_id,),
        ).fetchall()

    def test_split_and_attribution(self):
        n = build_sessions_quiet(self.con)
        self.assertEqual(n, 3)
        self.con.row_factory = sqlite3.Row
        s1, s2 = self.sessions_for(10)
        self.assertEqual((s1["first_message_id"], s1["last_message_id"]), (1, 2))
        self.assertEqual((s1["n_messages"], s1["n_from_me"]), (2, 1))
        self.assertEqual((s1["initiator_handle_id"], s1["initiator_is_me"]), (60, 0))
        self.assertEqual((s1["ender_handle_id"], s1["ender_is_me"]), (None, 1))
        self.assertEqual(s1["first_date"], "2026-03-01T10:00:00")
        self.assertEqual(s1["last_date"], "2026-03-01T10:20:00")
        self.assertEqual((s2["first_message_id"], s2["last_message_id"]), (3, 4))
        self.assertEqual((s2["initiator_handle_id"], s2["initiator_is_me"]), (None, 1))
        self.assertEqual((s2["ender_handle_id"], s2["ender_is_me"]), (60, 0))
        (s12,) = self.sessions_for(12)
        self.assertEqual((s12["n_messages"], s12["n_from_me"]), (1, 0))

    def test_state_keys(self):
        build_sessions_quiet(self.con)
        # max renderable id is 5: rows 6-9 are emote/system/blank/dateless
        self.assertEqual(int(get_state(self.con, "sessions_max_message_id")), 5)
        self.assertEqual(float(get_state(self.con, "sessions_gap_minutes")), 60.0)

    def test_noop_when_up_to_date(self):
        build_sessions_quiet(self.con)
        self.assertEqual(build_sessions_quiet(self.con), 0)
        self.assertEqual(
            self.con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 3)

    def test_incremental_rebuilds_only_affected_chat(self):
        build_sessions_quiet(self.con)
        chat12_id = self.con.execute(
            "SELECT id FROM sessions WHERE chat_id = 12").fetchone()[0]
        add_msg(self.con, 10, 10, 60, 0, "one more", "2026-03-01T13:45:00")
        self.con.commit()
        n = build_sessions_quiet(self.con)
        self.assertEqual(n, 2)  # chat 10's two sessions rebuilt, chat 12 untouched
        self.assertEqual(
            self.con.execute("SELECT id FROM sessions WHERE chat_id = 12").fetchone()[0],
            chat12_id)
        tail = self.con.execute(
            "SELECT n_messages, last_message_id FROM sessions "
            "WHERE chat_id = 10 ORDER BY first_date DESC LIMIT 1").fetchone()
        self.assertEqual(tuple(tail), (3, 10))
        self.assertEqual(int(get_state(self.con, "sessions_max_message_id")), 10)

    def test_gap_change_forces_full_rebuild(self):
        build_sessions_quiet(self.con)
        # gap 10: chat 10 splits at 20/160/30-minute gaps -> 4 sessions
        self.assertEqual(build_sessions_quiet(self.con, gap_minutes=10), 5)
        self.assertEqual(float(get_state(self.con, "sessions_gap_minutes")), 10.0)
        # back to the default: forced again even with no new messages
        self.assertEqual(build_sessions_quiet(self.con), 3)


class ActivityPatternsTest(FixtureBase):
    """12 real messages across four chats (UTC-pinned buckets); an emote, a
    system row, and a dateless row that must count nowhere.
    2026-03-01 is a Sunday."""

    def seed(self):
        c = self.con
        add_msg(c, 1, 10, 60, 0, "hey", "2026-03-01T10:00:00")
        add_msg(c, 2, 10, 0, 1, "hi", "2026-03-01T10:30:00")
        add_msg(c, 3, 10, 60, 0, "up?", "2026-03-02T04:30:00")     # late night
        add_msg(c, 4, 10, 0, 1, "barely", "2026-03-02T05:00:00")   # boundary: not late
        add_msg(c, 5, 10, 60, 0, "night", "2026-03-03T23:59:59")   # not late (0..5)
        add_msg(c, 6, 10, 0, 1, "midnight", "2026-03-04T00:00:00")  # boundary: late
        add_msg(c, 20, 11, 61, 0, "group hello", "2026-03-05T12:00:00")
        add_msg(c, 21, 11, 0, 1, "hey all", "2026-03-05T12:01:00")
        add_msg(c, 30, 12, 60, 0, "sms side", "2026-03-10T08:00:00")
        add_msg(c, 31, 12, 0, 1, "got it", "2026-03-10T08:01:00")
        add_msg(c, 40, 13, 62, 0, "carol here", "2026-03-06T09:00:00")
        add_msg(c, 41, 13, 0, 1, "hi carol", "2026-03-06T09:01:00")
        add_msg(c, 50, 10, 60, 0, "Loved a message", "2026-03-01T10:05:00", is_emote=1)
        add_msg(c, 51, 10, 60, 0, "renamed", "2026-03-01T10:06:00", item_type=2)
        add_msg(c, 52, 10, 60, 0, "dateless", None)

    def test_heatmap(self):
        res = analytics_time.get_activity_patterns(chat_id=10)
        self.assertEqual(res["timezone"], "UTC")
        self.assertEqual(res["message_count"], 6)
        heat = res["heatmap"]["counts"]
        self.assertEqual(heat[6][10], 2)   # Sunday 10h: msgs 1+2
        self.assertEqual(heat[0][4], 1)    # Monday 04h
        self.assertEqual(heat[0][5], 1)
        self.assertEqual(heat[1][23], 1)   # Tuesday 23h
        self.assertEqual(heat[2][0], 1)    # Wednesday 00h
        self.assertEqual(sum(map(sum, heat)), 6)
        self.assertEqual(res["heatmap"]["weekdays"][0], "Monday")

    def test_volume_series_month_unscoped(self):
        res = analytics_time.get_activity_patterns(granularity="month")
        self.assertEqual(res["message_count"], 12)
        self.assertEqual(res["volume_series"]["rows"],
                         [{"period": "2026-03", "sent": 6, "received": 6}])

    def test_volume_series_day(self):
        res = analytics_time.get_activity_patterns(chat_id=10, granularity="day")
        self.assertEqual(res["volume_series"]["granularity"], "day")
        self.assertEqual(res["volume_series"]["rows"], [
            {"period": "2026-03-01", "sent": 1, "received": 1},
            {"period": "2026-03-02", "sent": 1, "received": 1},
            {"period": "2026-03-03", "sent": 0, "received": 1},
            {"period": "2026-03-04", "sent": 1, "received": 0},
        ])

    def test_day_series_auto_coarsens(self):
        with mock.patch.object(analytics_time, "MAX_SERIES_BUCKETS", 3):
            res = analytics_time.get_activity_patterns(granularity="day")
        vs = res["volume_series"]
        self.assertEqual(vs["granularity"], "month")
        self.assertEqual(len(vs["rows"]), 1)
        self.assertIn("coarsened to month", vs["note"])

    def test_late_night_share_and_boundaries(self):
        res = analytics_time.get_activity_patterns(chat_id=10)
        ln = res["late_night"]
        # In: 04:30 and 00:00. Out: 05:00 (end is exclusive) and 23:59.
        self.assertEqual((ln["n"], ln["total"]), (2, 6))
        self.assertEqual(ln["share"], round(2 / 6, 4))
        self.assertEqual((ln["start_hour"], ln["end_hour"]), (0, 5))
        self.assertNotIn("top_contacts", ln)  # scoped call: no leaderboard

    def test_late_night_unscoped_leaderboard(self):
        with mock.patch.object(analytics_time, "MIN_LEADERBOARD_MESSAGES", 1):
            res = analytics_time.get_activity_patterns(include=["late_night"])
        board = res["late_night"]["top_contacts"]
        # Bob: 1:1 chats 10+12 (8 msgs, 2 late) -- group traffic excluded;
        # Carol: chat 13 (2 msgs, 0 late).
        self.assertEqual(board[0], {"contact": "Bob Jones", "messages": 8,
                                    "late_night": 2, "share": 0.25})
        self.assertEqual(board[1], {"contact": "+15550002222", "messages": 2,
                                    "late_night": 0, "share": 0.0})

    def test_busiest_days_and_streaks(self):
        res = analytics_time.get_activity_patterns()
        self.assertEqual(res["busiest_days"][0], {"day": "2026-03-01", "count": 2})
        self.assertEqual(len(res["busiest_days"]), 7)
        streaks = res["streaks"]
        self.assertEqual(streaks["longest_streak"],
                         {"days": 6, "start": "2026-03-01", "end": "2026-03-06"})
        # Mar 6 09:01 -> Mar 10 08:00 = 3d 22h59m ~= 4.0 days
        self.assertEqual(streaks["longest_drought"]["days"], 4.0)
        self.assertTrue(streaks["longest_drought"]["from"].startswith("2026-03-06T09:01"))
        self.assertTrue(streaks["longest_drought"]["to"].startswith("2026-03-10T08:00"))

    def test_person_scope_covers_their_chats(self):
        res = analytics_time.get_activity_patterns(contact_id=5)
        self.assertEqual(res["message_count"], 10)  # chats 10+11+12, both sides

    def test_empty_scope(self):
        res = analytics_time.get_activity_patterns(contact_id=999)
        self.assertEqual(res["message_count"], 0)
        self.assertEqual(res["volume_series"]["rows"], [])
        self.assertIsNone(res["late_night"]["share"])
        self.assertIsNone(res["streaks"]["longest_streak"])

    def test_date_bounds_half_open(self):
        res = analytics_time.get_activity_patterns(
            chat_id=10, date_from="2026-03-02", date_to="2026-03-02")
        self.assertEqual(res["message_count"], 2)  # whole of Mar 2 included
        res = analytics_time.get_activity_patterns(
            chat_id=10, date_from="2026-03-03", date_to="2026-03-03")
        self.assertEqual(res["message_count"], 1)  # 23:59:59 still inside

    def test_include_subset_and_validation(self):
        res = analytics_time.get_activity_patterns(include=["heatmap"])
        self.assertIn("heatmap", res)
        self.assertNotIn("volume_series", res)
        self.assertNotIn("late_night", res)
        with self.assertRaises(ValueError):
            analytics_time.get_activity_patterns(include=["bogus"])
        with self.assertRaises(ValueError):
            analytics_time.get_activity_patterns(granularity="week")


class ResponseDynamicsTest(FixtureBase):
    """chat 10: a scripted exchange with known reply/read/double-text gaps;
    chat 11: a group with two sessions started by Bob and Carol."""

    def seed(self):
        c = self.con
        add_msg(c, 1, 10, 60, 0, "hey", "2026-03-01T10:00:00",
                date_read="2026-03-01T10:02:00")
        add_msg(c, 2, 10, 0, 1, "hi bob", "2026-03-01T10:05:00",
                date_read="2026-03-01T10:06:00")
        add_msg(c, 3, 10, 60, 0, "quick q", "2026-03-01T10:06:00")
        add_msg(c, 4, 10, 60, 0, "hello??", "2026-03-01T10:30:00")   # double-text them
        add_msg(c, 5, 10, 0, 1, "sorry!", "2026-03-02T09:00:00")     # 22.5h: initiation me
        add_msg(c, 6, 10, 0, 1, "you there?", "2026-03-02T09:20:00")  # double-text me
        add_msg(c, 20, 11, 61, 0, "group hello", "2026-04-01T12:00:00")
        add_msg(c, 21, 11, 0, 1, "hey all", "2026-04-01T12:01:00")
        add_msg(c, 22, 11, 62, 0, "carol here", "2026-04-01T12:02:00")
        add_msg(c, 23, 11, 62, 0, "next day", "2026-04-02T12:00:00")
        add_msg(c, 50, 10, 60, 0, "Loved a message", "2026-03-01T10:07:00", is_emote=1)
        self.con.commit()
        build_sessions_quiet(self.con)

    def test_requires_scope(self):
        self.assertIn("error", analytics_time.get_response_dynamics())
        # an explicit empty list means "omitted", so still no scope
        self.assertIn("error", analytics_time.get_response_dynamics(handle_id=[]))

    def test_counts_initiation_latency_double_text(self):
        res = analytics_time.get_response_dynamics(chat_id=10)
        self.assertEqual(res["message_count"], 6)
        self.assertEqual((res["from_me"], res["from_them"]), (3, 3))
        self.assertEqual(res["initiation"], {"me": 1, "them": 1})
        self.assertEqual(res["reply_latency"]["me"]["median_seconds"], 300.0)
        self.assertEqual(res["reply_latency"]["me"]["count"], 1)
        self.assertEqual(res["reply_latency"]["them"]["median_seconds"], 60.0)
        self.assertEqual(res["double_text"], {"me": 1, "them": 1})

    def test_read_to_reply(self):
        res = analytics_time.get_response_dynamics(chat_id=10)
        rr = res["read_to_reply"]
        self.assertEqual(rr["pairs"], 1)
        self.assertEqual(rr["median_unseen_seconds"], 120.0)        # 10:00 -> 10:02
        self.assertEqual(rr["median_deliberation_seconds"], 180.0)  # 10:02 -> 10:05
        self.assertEqual(rr["coverage"], {
            "inbound_with_date_read": 1, "inbound_total": 3,
            "outbound_with_date_read": 1, "outbound_total": 3,
        })

    def test_sessions_ghosted_burstiness(self):
        res = analytics_time.get_response_dynamics(chat_id=10)
        self.assertEqual(res["sessions"]["count"], 2)
        self.assertEqual(res["sessions"]["median_messages"], 3)
        self.assertEqual(res["sessions"]["median_duration_seconds"], 1500.0)
        self.assertEqual(res["sessions"]["ended_by_me_share"], 0.5)
        self.assertEqual(res["ghosted"], {"only_me": 1, "only_them": 0})
        gaps = [300, 60, 1440, 81000, 1200]
        mu, sigma = statistics.fmean(gaps), statistics.pstdev(gaps)
        self.assertAlmostEqual(res["burstiness"], (sigma - mu) / (sigma + mu), places=4)
        self.assertNotIn("initiations_by_member", res)  # not a group chat

    def test_group_chat_adds_member_initiations(self):
        res = analytics_time.get_response_dynamics(chat_id=11)
        rows = {r["member"]: r["sessions_initiated"]
                for r in res["initiations_by_member"]}
        self.assertEqual(rows, {"Bob Jones": 1, "+15550002222": 1})

    def test_person_scope_restricted_to_direct_chats(self):
        res = analytics_time.get_response_dynamics(contact_id=5)
        self.assertEqual(res["chat_ids"], [10, 12])  # group chat 11 dropped
        self.assertEqual(res["message_count"], 6)

    def test_empty_person_scope_returns_zeroes(self):
        res = analytics_time.get_response_dynamics(contact_id=999)
        self.assertNotIn("error", res)
        self.assertEqual(res["message_count"], 0)
        self.assertIsNone(res["burstiness"])
        self.assertEqual(res["sessions"]["count"], 0)

    def test_date_bounds_half_open(self):
        res = analytics_time.get_response_dynamics(chat_id=10, date_to="2026-03-01")
        self.assertEqual(res["message_count"], 4)  # all of Mar 1
        self.assertEqual(res["initiation"], {"me": 0, "them": 1})
        self.assertEqual(res["sessions"]["count"], 1)  # Mar 2 session starts later


class ExcludeChatIdTest(FixtureBase):
    """The Python-only exclude_chat_id carve-out (wrapped.py's --exclude):
    excluded chats leave the aggregate, everything else -- including
    messages with NO chat_id, which a bare NOT IN silently drops -- stays."""

    def seed(self):
        c = self.con
        add_msg(c, 1, 10, 60, 0, "bob a", "2026-03-01T10:00:00")
        add_msg(c, 2, 10, 0, 1, "bob b", "2026-03-01T10:01:00")
        add_msg(c, 3, 10, 60, 0, "bob c", "2026-03-01T10:02:00")
        add_msg(c, 4, 13, 62, 0, "carol a", "2026-03-02T10:00:00")
        add_msg(c, 5, 13, 0, 1, "carol b", "2026-03-02T10:01:00")
        # A real, dated message attached to no chat: the exclusion clause's
        # `chat_id IS NULL OR` arm exists for exactly this row.
        add_msg(c, 6, None, 62, 0, "orphaned", "2026-03-03T10:00:00")

    def test_exclusion_scalar_and_null_chat_survival(self):
        self.assertEqual(analytics_time.get_activity_patterns()["message_count"], 6)
        res = analytics_time.get_activity_patterns(exclude_chat_id=10)
        self.assertEqual(res["message_count"], 3)  # chat 13 + the NULL-chat row
        self.assertEqual(res["volume_series"]["rows"],
                         [{"period": "2026-03", "sent": 1, "received": 2}])

    def test_exclusion_list(self):
        res = analytics_time.get_activity_patterns(exclude_chat_id=[10, 13])
        self.assertEqual(res["message_count"], 1)  # only the NULL-chat row

    def test_exclusion_composes_with_scope(self):
        # Excluding a chat must not turn an unscoped call scoped, and must
        # also carve out of an explicit scope.
        res = analytics_time.get_activity_patterns(chat_id=[10, 13],
                                                   exclude_chat_id=13)
        self.assertEqual(res["message_count"], 3)

    def test_as_exclusion_list_never_truncates(self):
        from query_tools._shared import MAX_ID_LIST, _as_exclusion_list
        self.assertIsNone(_as_exclusion_list(None))
        self.assertIsNone(_as_exclusion_list([]))
        self.assertEqual(_as_exclusion_list(7), [7])
        big = list(range(MAX_ID_LIST * 3))
        # An exclusion is a privacy guard: truncating at MAX_ID_LIST would
        # silently re-admit the excluded.
        self.assertEqual(_as_exclusion_list(big), big)


if __name__ == "__main__":
    unittest.main()
