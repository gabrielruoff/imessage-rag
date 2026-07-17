"""query_tools/analytics_social.py: reaction stats (given/received/rates,
reciprocity, time-to-react, funniest leaderboard, hall of fame) and
milestones (firsts, nth/landmarks, anniversaries, longest silence, earliest
phrase) -- all against a hand-scripted fixture DB with known dates, so every
aggregate has one right answer. Hermetic: temp sqlite file, no Ollama, no
real chat.db."""

import datetime
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from db import load_vec_extension
from schema import create_schema

import query_tools
from query_tools import analytics_social


def _msg(mid, chat, handle, from_me, text, date, is_emote=0, item_type=0):
    return (mid, f"guid-{mid}", chat, handle, from_me, text, date, 0, is_emote, item_type)


class SocialFixture(unittest.TestCase):
    """Chat 10: a 1:1 with Bob (contact 5, duplicate handles 60+61) with a
    ~19-month silence in the middle. Chat 11: a group with Bob and Carol
    (handle 62, no contact row). Chat 12: a second 1:1 with Bob. Chat 13: a
    1:1 with Dana (contact 7) whose first message is in early January, for
    the anniversary year-wrap case."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        load_vec_extension(con)
        try:
            create_schema(con)
            con.executemany(
                "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
                "nickname, organization) VALUES (?, ?, ?, ?, NULL, NULL)",
                [(5, "uid-5", "Bob", "Jones"), (7, "uid-7", "Dana", "Smith")],
            )
            con.executemany(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (?, ?, ?, 'iMessage')",
                [
                    (60, "+15550001111", 5),     # Bob, phone
                    (61, "bob@example.com", 5),  # Bob, email (duplicate identity)
                    (62, "+15550002222", None),  # Carol, no contact row
                    (63, "+15550003333", 7),     # Dana
                ],
            )
            con.executemany(
                "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (?, ?, ?, ?, 'iMessage')",
                [
                    (10, "chat10", None, 0),
                    (11, "chat11", "The Group", 1),
                    (12, "chat12", None, 0),
                    (13, "chat13", None, 0),
                ],
            )
            con.executemany(
                "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
                [(10, 60), (11, 61), (11, 62), (12, 60), (13, 63)],
            )
            messages = [
                # Chat 10: three-message exchange in July 2024, then a long
                # silence broken BY ME in March 2026, then the phrase rows.
                _msg(1, 10, 60, 0, "hey", "2024-07-20T10:00:00"),
                _msg(2, 10, 0, 1, "hi bob", "2024-07-20T10:01:00"),
                _msg(3, 10, 60, 0, "you around later?", "2024-07-20T10:02:00"),
                _msg(4, 10, 0, 1, "morning bob", "2026-03-03T09:30:00"),
                _msg(5, 10, 60, 0, "morning", "2026-03-03T09:31:00"),
                _msg(6, 10, 0, 1, "the eagle has landed", "2026-03-04T08:00:00"),
                _msg(7, 10, 60, 0, "the eagle has landed indeed", "2026-03-05T08:00:00"),
                # Emote + system rows INSIDE the chat-10 silence: excluded by
                # _REAL_MESSAGE_CLAUSES, so they must not shorten the gap.
                _msg(40, 10, 60, 0, "Loved a message", "2025-01-01T00:00:00", is_emote=1),
                _msg(41, 10, 60, 0, "named the group", "2025-06-01T00:00:00", item_type=2),
                # Group chat 11: me, Bob (via email handle 61), Carol.
                _msg(20, 11, 61, 0, "group hello", "2026-04-01T12:00:00"),
                _msg(21, 11, 0, 1, "hey all", "2026-04-01T12:01:00"),
                _msg(22, 11, 62, 0, "carol here", "2026-04-01T12:02:00"),
                _msg(23, 11, 62, 0, "big announcement", "2026-04-01T12:03:00"),
                _msg(24, 11, 0, 1, "nice", "2026-04-01T12:04:00"),
                # Chat 12 (second 1:1 with Bob).
                _msg(30, 12, 60, 0, "sms fallback", "2026-05-01T08:00:00"),
                _msg(31, 12, 0, 1, "got it", "2026-05-01T08:05:00"),
                # Chat 13 (Dana): early-January first contact.
                _msg(50, 13, 63, 0, "happy new year", "2025-01-10T00:30:00"),
                _msg(51, 13, 0, 1, "you too dana", "2025-01-10T00:31:00"),
            ]
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment, is_emote, item_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                messages,
            )
            # Reactions. is_from_me=1 rows carry the counterparty's handle_id
            # (matching real ingested data); codes: 2000 love, 2003 laugh,
            # 3003 laugh-removal (excluded), 1000 stray (excluded).
            con.executemany(
                "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
                "is_from_me, associated_message_type, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    # Carol's "big announcement" (guid-23): 2 laughs + 1 love
                    (100, "guid-23", 11, 61, 0, 2003, "2026-04-01T12:04:00"),  # Bob, 60s
                    (101, "guid-23", 11, 62, 1, 2003, "2026-04-01T12:05:00"),  # Me, 120s
                    (102, "guid-23", 11, 61, 0, 2000, "2026-04-01T12:06:00"),  # Bob, 180s
                    # My "hey all" (guid-21): 1 love from Carol, 360s
                    (103, "guid-21", 11, 62, 0, 2000, "2026-04-01T12:07:00"),
                    # My chat-10 "hi bob" (guid-2): 1 laugh from Bob, 120s
                    (104, "guid-2", 10, 60, 0, 2003, "2024-07-20T10:03:00"),
                    # Removal + stray codes on guid-23: must not count anywhere
                    (105, "guid-23", 11, 61, 0, 3003, "2026-04-01T12:08:00"),
                    (106, "guid-23", 11, 61, 0, 1000, "2026-04-01T12:09:00"),
                    # Me loving Bob's "group hello" (guid-20 at 12:00), 540s
                    (107, "guid-20", 11, 61, 1, 2000, "2026-04-01T12:09:00"),
                ],
            )
            con.commit()
        finally:
            con.close()

        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)


class ReactionStatsTest(SocialFixture):
    # Corpus denominators: 7 real sent messages, 9 real received messages.

    def test_given_totals_types_and_rate(self):
        s = analytics_social.get_reaction_stats(include=["given"])
        self.assertEqual(set(s), {"chat_ids", "given"})
        self.assertEqual(s["given"]["total"], 2)  # 101 + 107; removals/strays out
        self.assertEqual(s["given"]["by_type"], {"laugh": 1, "love": 1})
        self.assertEqual(s["given"]["per_100_received_messages"], round(200 / 9, 2))

    def test_received_totals_types_and_rate(self):
        s = analytics_social.get_reaction_stats(include=["received"])
        self.assertEqual(s["received"]["total"], 2)  # 103 + 104
        self.assertEqual(s["received"]["by_type"], {"love": 1, "laugh": 1})
        self.assertEqual(s["received"]["per_100_sent_messages"], round(200 / 7, 2))

    def test_reciprocity_ratio(self):
        s = analytics_social.get_reaction_stats(include=["reciprocity"])
        self.assertEqual(s["reciprocity"], {"given": 2, "received": 2, "ratio": 1.0})

    def test_time_to_react_medians_per_side(self):
        s = analytics_social.get_reaction_stats(include=["time_to_react"])
        # me: 120s (101) and 540s (107) -> 330; them: 60/180/360/120 -> 150
        self.assertEqual(s["time_to_react"]["me_reacting"],
                         {"median_seconds": 330.0, "count": 2})
        self.assertEqual(s["time_to_react"]["them_reacting"],
                         {"median_seconds": 150.0, "count": 4})

    def test_funniest_min_reactions_gate(self):
        s = analytics_social.get_reaction_stats(include=["funniest"])
        self.assertEqual(s["funniest"], [])  # nobody has 20 reactions on my msgs
        with mock.patch.object(analytics_social, "_FUNNIEST_MIN_REACTIONS", 1):
            s = analytics_social.get_reaction_stats(include=["funniest"])
        self.assertEqual([r["participant"] for r in s["funniest"]],
                         ["Bob Jones", "+15550002222"])
        bob = s["funniest"][0]
        # Bob's chats span BOTH his handles (10, 11, 12): 6 of my messages.
        self.assertEqual(bob["handle_ids"], [60, 61])
        self.assertEqual(bob["laughs"], 1)
        self.assertEqual(bob["my_messages"], 6)
        self.assertEqual(bob["laughs_per_100_my_messages"], round(100 / 6, 2))
        carol = s["funniest"][1]
        self.assertEqual(carol["laughs_per_100_my_messages"], 0.0)
        self.assertEqual(carol["my_messages"], 2)

    def test_hall_of_fame_ordering_and_breakdown(self):
        s = analytics_social.get_reaction_stats(chat_id=11, include=["hall_of_fame"])
        rows = s["hall_of_fame"]
        self.assertEqual([r["guid"] for r in rows], ["guid-23", "guid-21", "guid-20"])
        self.assertEqual(rows[0]["reaction_count"], 3)  # not 5: 3003/1000 excluded
        self.assertEqual(rows[0]["reactions_by_type"], {"laugh": 2, "love": 1})
        capped = analytics_social.get_reaction_stats(
            chat_id=11, include=["hall_of_fame"], limit=1
        )
        self.assertEqual([r["guid"] for r in capped["hall_of_fame"]], ["guid-23"])

    def test_reaction_type_by_name_and_code(self):
        by_name = analytics_social.get_reaction_stats(
            include=["given", "received", "hall_of_fame"], reaction_type="love"
        )
        self.assertEqual(by_name["given"], {"total": 1, "by_type": {"love": 1},
                                            "per_100_received_messages": round(100 / 9, 2)})
        self.assertEqual(by_name["received"]["by_type"], {"love": 1})
        self.assertEqual({r["guid"] for r in by_name["hall_of_fame"]},
                         {"guid-23", "guid-21", "guid-20"})
        by_code = analytics_social.get_reaction_stats(
            include=["hall_of_fame"], reaction_type=2000
        )
        self.assertEqual({r["guid"] for r in by_name["hall_of_fame"]},
                         {r["guid"] for r in by_code["hall_of_fame"]})
        with self.assertRaises(ValueError):
            analytics_social.get_reaction_stats(reaction_type="applause")

    def test_date_bounds_are_half_open_on_reaction_date(self):
        # date_to with seconds granularity advances by 1s: 12:05:00 is IN.
        s = analytics_social.get_reaction_stats(
            include=["given"], date_to="2026-04-01T12:05:00"
        )
        self.assertEqual(s["given"]["total"], 1)  # 101 in, 107 (12:09) out
        s = analytics_social.get_reaction_stats(
            include=["given"], date_to="2026-04-01T12:04:59"
        )
        self.assertEqual(s["given"]["total"], 0)
        s = analytics_social.get_reaction_stats(
            include=["received"], date_from="2026-04-02"
        )
        self.assertEqual(s["received"]["total"], 0)  # both received are earlier

    def test_person_scope_is_their_chats(self):
        s = analytics_social.get_reaction_stats(contact_id=5, include=["given"])
        self.assertEqual(sorted(s["chat_ids"]), [10, 11, 12])
        self.assertEqual(s["given"]["total"], 2)
        # Denominator drops Dana's chat: 8 received messages, not 9.
        self.assertEqual(s["given"]["per_100_received_messages"], 25.0)

    def test_unknown_person_returns_zeroed_sections(self):
        s = analytics_social.get_reaction_stats(contact_id=999999)
        self.assertEqual(s["given"]["total"], 0)
        self.assertEqual(s["reciprocity"]["ratio"], None)
        self.assertEqual(s["funniest"], [])
        self.assertEqual(s["hall_of_fame"], [])

    def test_unknown_include_section_raises(self):
        with self.assertRaises(ValueError):
            analytics_social.get_reaction_stats(include=["given", "bogus"])


class MilestonesTest(SocialFixture):
    def test_firsts_per_side(self):
        m = analytics_social.get_milestones(chat_id=10)
        firsts = m["firsts"]
        self.assertEqual(firsts["first"]["id"], 1)
        self.assertEqual(firsts["first"]["sender"], "Bob Jones")
        self.assertEqual(firsts["last"]["id"], 7)
        self.assertEqual(firsts["first_from_me"]["id"], 2)
        self.assertEqual(firsts["first_from_them"]["id"], 1)

    def test_nth_message_and_out_of_range(self):
        m = analytics_social.get_milestones(chat_id=10, n=3)
        # Order by date,id over REAL rows only (40/41 excluded): 1,2,3,4,...
        self.assertEqual(m["nth"], {"total_messages": 7, "n": 3,
                                    "message": m["nth"]["message"]})
        self.assertEqual(m["nth"]["message"]["id"], 3)
        m = analytics_social.get_milestones(chat_id=10, n=100)
        self.assertIsNone(m["nth"]["message"])
        with self.assertRaises(ValueError):
            analytics_social.get_milestones(chat_id=10, n=0)

    def test_landmarks_skip_beyond_count(self):
        m = analytics_social.get_milestones(chat_id=10)
        self.assertEqual(m["nth"], {"total_messages": 7, "landmarks": []})
        with mock.patch.object(analytics_social, "_LANDMARK_ORDINALS", (3, 5, 100)):
            m = analytics_social.get_milestones(chat_id=10)
        marks = m["nth"]["landmarks"]
        self.assertEqual([lm["ordinal"] for lm in marks], [3, 5])
        self.assertEqual([lm["message"]["id"] for lm in marks], [3, 5])

    def test_scoped_anniversary(self):
        m = analytics_social.get_milestones(chat_id=10, reference_date="2026-07-01")
        self.assertEqual(m["anniversaries"], {
            "first_contact": "2024-07-20T10:00:00",
            "next_anniversary": "2026-07-20",
            "years": 2,
            "days_until": 19,
        })

    def test_unscoped_anniversaries_window_and_min_gate(self):
        # Default gate: nobody has 100 lifetime messages.
        m = analytics_social.get_milestones(reference_date="2027-03-25")
        self.assertEqual(m["anniversaries"]["upcoming"], [])
        with mock.patch.object(analytics_social, "_ANNIVERSARY_MIN_MESSAGES", 1):
            m = analytics_social.get_milestones(reference_date="2027-03-25")
        rows = m["anniversaries"]["upcoming"]
        # Only Carol (Apr 1) is within 30 days; Bob (Jul 20) and Dana (Jan 10) not.
        self.assertEqual([r["participant"] for r in rows], ["+15550002222"])
        self.assertEqual(rows[0]["next_anniversary"], "2027-04-01")
        self.assertEqual(rows[0]["years"], 1)
        self.assertEqual(rows[0]["days_until"], 7)

    def test_unscoped_anniversaries_year_wrap(self):
        with mock.patch.object(analytics_social, "_ANNIVERSARY_MIN_MESSAGES", 1):
            m = analytics_social.get_milestones(reference_date="2026-12-25")
        rows = m["anniversaries"]["upcoming"]
        # Dana's Jan 10 anniversary wraps into the next year: 16 days away.
        self.assertEqual([r["participant"] for r in rows], ["Dana Smith"])
        self.assertEqual(rows[0]["next_anniversary"], "2027-01-10")
        self.assertEqual(rows[0]["years"], 2)
        self.assertEqual(rows[0]["days_until"], 16)

    def test_longest_silence_and_reconnection(self):
        m = analytics_social.get_milestones(chat_id=10)
        silence = m["longest_silence"]
        self.assertTrue(silence["available"])
        # Emote (2025-01-01) and system (2025-06-01) rows sit inside the gap
        # but are not real messages, so the silence spans them untouched.
        self.assertEqual(silence["from"], "2024-07-20T10:02:00")
        self.assertEqual(silence["to"], "2026-03-03T09:30:00")
        expected_days = (datetime.datetime(2026, 3, 3, 9, 30)
                         - datetime.datetime(2024, 7, 20, 10, 2)).total_seconds() / 86400
        self.assertEqual(silence["days"], round(expected_days, 1))
        self.assertEqual(silence["broken_by"], "Me")
        self.assertEqual(silence["reconnection_message"]["id"], 4)

    def test_longest_silence_requires_scope(self):
        m = analytics_social.get_milestones()
        self.assertEqual(m["longest_silence"]["available"], False)

    def test_earliest_phrase_via_fts(self):
        m = analytics_social.get_milestones(chat_id=10, phrase="eagle has landed")
        self.assertEqual(m["earliest_phrase"]["message"]["id"], 6)
        # Unscoped hit is the same row; embedded quotes must not break FTS.
        m = analytics_social.get_milestones(phrase='"eagle" has landed')
        self.assertEqual(m["earliest_phrase"]["message"]["id"], 6)
        m = analytics_social.get_milestones(chat_id=10, phrase="jabberwocky")
        self.assertIsNone(m["earliest_phrase"]["message"])
        self.assertNotIn("earliest_phrase", analytics_social.get_milestones(chat_id=10))

    def test_unknown_person_returns_empty_sections(self):
        m = analytics_social.get_milestones(contact_id=999999, phrase="hey", n=1)
        self.assertIsNone(m["firsts"]["first"])
        self.assertEqual(m["nth"]["total_messages"], 0)
        self.assertEqual(m["longest_silence"]["available"], False)
        self.assertIsNone(m["earliest_phrase"]["message"])

    def test_person_scope_spans_their_chats(self):
        m = analytics_social.get_milestones(contact_id=5)
        self.assertEqual(sorted(m["chat_ids"]), [10, 11, 12])
        self.assertEqual(m["firsts"]["first"]["id"], 1)
        self.assertEqual(m["firsts"]["last"]["id"], 31)


if __name__ == "__main__":
    unittest.main()
