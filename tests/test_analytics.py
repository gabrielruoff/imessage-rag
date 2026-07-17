"""query_tools/analytics.py: conversation dynamics, activity histograms,
reaction leaderboards, participant stats, and word frequency -- all against a
hand-scripted fixture DB with known dates, so every aggregate has one right
answer. Hermetic: temp sqlite file, no Ollama, no real chat.db."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import ingest
import query_tools
from query_tools import analytics


def _msg(mid, chat, handle, from_me, text, date):
    return (mid, f"guid-{mid}", chat, handle, from_me, text, date, 0)


class AnalyticsFixture(unittest.TestCase):
    """Chat 10: a 1:1 with Bob (contact 5, duplicate handles 60+61).
    Chat 11: a group with Bob and Carol (handle 62, no contact row).
    Chat 12: a second 1:1 with Bob (the iMessage/SMS-twin shape)."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            ingest.create_schema(con)
            con.execute(
                "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
                "nickname, organization) VALUES (5, 'uid-5', 'Bob', 'Jones', NULL, NULL)"
            )
            con.executemany(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (?, ?, ?, 'iMessage')",
                [
                    (60, "+15550001111", 5),   # Bob, phone
                    (61, "bob@example.com", 5),  # Bob, email (duplicate identity)
                    (62, "+15550002222", None),  # Carol, no contact row
                ],
            )
            con.executemany(
                "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (?, ?, ?, ?, 'iMessage')",
                [
                    (10, "chat10", None, 0),
                    (11, "chat11", "The Group", 1),
                    (12, "chat12", None, 0),
                ],
            )
            con.executemany(
                "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
                [(10, 60), (11, 61), (11, 62), (12, 60)],
            )
            # Chat 10 timeline (all 2026-03, gap threshold tests assume
            # gap_minutes=180 / DOUBLE_TEXT_MIN_MINUTES=10):
            #   d1 10:00 them  -- initiation #1 (them, first message)
            #   d1 10:01 me    -- response (60s, sender change)
            #   d1 10:02 them  -- response (60s)
            #   d1 10:05 them  -- same sender, 3 min < 10 -> multi-part, not double-text
            #   d1 10:20 them  -- same sender, 15 min >= 10 -> double-text (them)
            #   d1 14:00 me    -- 220 min > 180 -> initiation #2 (me)
            #   d3 09:00 them  -- ~43h  -> initiation #3 (them); longest silence,
            #                    broken by them
            #   d3 09:30 me    -- 30 min response (me)
            self.chat10 = [
                _msg(1, 10, 60, 0, "hey", "2026-03-01T10:00:00"),
                _msg(2, 10, 0, 1, "hi bob", "2026-03-01T10:01:00"),
                _msg(3, 10, 60, 0, "you around later?", "2026-03-01T10:02:00"),
                _msg(4, 10, 60, 0, "for the game I mean", "2026-03-01T10:05:00"),
                _msg(5, 10, 60, 0, "hello??", "2026-03-01T10:20:00"),
                _msg(6, 10, 0, 1, "sorry yes", "2026-03-01T14:00:00"),
                _msg(7, 10, 60, 0, "morning", "2026-03-03T09:00:00"),
                _msg(8, 10, 0, 1, "morning bob", "2026-03-03T09:30:00"),
            ]
            # Group chat 11: me, Bob (via email handle 61), Carol.
            self.chat11 = [
                _msg(20, 11, 61, 0, "group hello", "2026-04-01T12:00:00"),
                _msg(21, 11, 0, 1, "hey all", "2026-04-01T12:01:00"),
                _msg(22, 11, 62, 0, "carol here", "2026-04-01T12:02:00"),
                _msg(23, 11, 62, 0, "big announcement", "2026-04-01T12:03:00"),
            ]
            # Chat 12 (second 1:1 with Bob): one exchange, later than chat 10.
            self.chat12 = [
                _msg(30, 12, 60, 0, "sms fallback", "2026-05-01T08:00:00"),
                _msg(31, 12, 0, 1, "got it", "2026-05-01T08:05:00"),
            ]
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                self.chat10 + self.chat11 + self.chat12,
            )
            # A tapback-ish emote row and a system row -- excluded everywhere.
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment, is_emote, item_type) "
                "VALUES (40, 'guid-40', 10, 60, 0, 'Loved a message', "
                "'2026-03-01T10:06:00', 0, 1, 0)"
            )
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment, is_emote, item_type) "
                "VALUES (41, 'guid-41', 10, 60, 0, 'named the group', "
                "'2026-03-01T10:07:00', 0, 0, 2)"
            )
            # Reactions. is_from_me=1 rows carry the counterparty's handle_id
            # (matching real ingested data); codes: 2000 love, 2003 laugh,
            # 3003 laugh-removal (excluded), 1000 stray (excluded).
            con.executemany(
                "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
                "is_from_me, associated_message_type, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    # Carol's "big announcement" (guid-23): 2 laughs + 1 love
                    (100, "guid-23", 11, 61, 0, 2003, "2026-04-01T12:04:00"),  # Bob laughs
                    (101, "guid-23", 11, 62, 1, 2003, "2026-04-01T12:05:00"),  # Me (laugh)
                    (102, "guid-23", 11, 61, 0, 2000, "2026-04-01T12:06:00"),  # Bob loves
                    # My "hey all" (guid-21): 1 love from Carol
                    (103, "guid-21", 11, 62, 0, 2000, "2026-04-01T12:07:00"),
                    # My chat-10 message (guid-2): 1 laugh from Bob
                    (104, "guid-2", 10, 60, 0, 2003, "2026-03-01T10:03:00"),
                    # Removal + stray codes on guid-23: must not count anywhere
                    (105, "guid-23", 11, 61, 0, 3003, "2026-04-01T12:08:00"),
                    (106, "guid-23", 11, 61, 0, 1000, "2026-04-01T12:09:00"),
                ],
            )
            con.commit()
        finally:
            con.close()

        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)


class ConversationDynamicsTest(AnalyticsFixture):
    def test_scripted_exchange_counts(self):
        d = query_tools.get_conversation_dynamics(chat_id=10)
        self.assertEqual(d["message_count"], 8 + 2)  # incl. emote+system rows
        self.assertEqual(d["initiations"], {"me": 1, "them": 2})
        self.assertEqual(d["double_texts"]["them"], 1)
        self.assertEqual(d["double_texts"]["me"], 0)

    def test_response_medians_per_side(self):
        d = query_tools.get_conversation_dynamics(chat_id=10)
        # them: one 60s response; me: 60s and 1800s -> median 930
        self.assertEqual(d["response_seconds"]["them"]["count"], 1)
        self.assertEqual(d["response_seconds"]["them"]["median_seconds"], 60)
        self.assertEqual(d["response_seconds"]["me"]["median_seconds"], 930)

    def test_longest_silence_and_who_broke_it(self):
        d = query_tools.get_conversation_dynamics(chat_id=10, top_silences=1)
        top = d["longest_silences"][0]
        self.assertEqual(top["broken_by"], "them")
        self.assertEqual(top["to"], "2026-03-03T09:00:00")
        self.assertAlmostEqual(top["days"], 1.8, places=1)

    def test_gap_minutes_override_flips_initiation_to_response(self):
        # With a 24h gap, the 14:00 message becomes a response, not an initiation.
        d = query_tools.get_conversation_dynamics(chat_id=10, gap_minutes=24 * 60)
        self.assertEqual(d["initiations"], {"me": 0, "them": 2})

    def test_person_scope_covers_only_direct_chats(self):
        d = query_tools.get_conversation_dynamics(contact_id=5)
        # Bob's 1:1 chats are 10 and 12; the group chat 11 must be excluded.
        self.assertEqual(sorted(d["chat_ids"]), [10, 12])

    def test_explicit_chat_id_list_merges_twin_chats(self):
        d = query_tools.get_conversation_dynamics(chat_id=[10, 12])
        self.assertEqual(d["message_count"], 12)

    def test_no_scope_raises(self):
        with self.assertRaises(ValueError):
            query_tools.get_conversation_dynamics()

    def test_unknown_person_returns_zeroed_result(self):
        d = query_tools.get_conversation_dynamics(contact_id=999999)
        self.assertEqual(d["message_count"], 0)
        self.assertEqual(d["longest_silences"], [])


class ActivityHistogramTest(AnalyticsFixture):
    def test_month_buckets_split_by_sender(self):
        h = query_tools.get_activity_histogram(contact_id=5, bucket="month")
        by_bucket = {r["bucket"]: r for r in h["rows"]}
        # Bob's chats = 10, 11, 12 (histogram includes groups)
        self.assertEqual(by_bucket["2026-03"]["total"], 10)
        self.assertEqual(by_bucket["2026-03"]["from_me"], 3)
        self.assertEqual(by_bucket["2026-03"]["from_them"], 7)
        self.assertEqual(by_bucket["2026-04"]["total"], 4)
        self.assertEqual(by_bucket["2026-05"]["total"], 2)
        self.assertFalse(h["truncated"])

    def test_hour_and_dow_buckets(self):
        h = query_tools.get_activity_histogram(chat_id=11, bucket="hour")
        self.assertEqual(h["rows"], [{"bucket": "12", "total": 4,
                                      "from_me": 1, "from_them": 3}])
        d = query_tools.get_activity_histogram(chat_id=11, bucket="dow")
        # 2026-04-01 is a Wednesday
        self.assertEqual(d["rows"][0]["bucket"], "Wednesday")

    def test_unknown_bucket_raises(self):
        with self.assertRaises(ValueError):
            query_tools.get_activity_histogram(bucket="year")

    def test_truncation_flag(self):
        with mock.patch.object(analytics, "MAX_HISTOGRAM_ROWS", 2):
            h = query_tools.get_activity_histogram(bucket="day")
        self.assertTrue(h["truncated"])
        self.assertEqual(len(h["rows"]), 2)
        self.assertIn("note", h)

    def test_date_range_filters(self):
        h = query_tools.get_activity_histogram(
            bucket="month", date_from="2026-04-01", date_to="2026-04-30"
        )
        self.assertEqual([r["bucket"] for r in h["rows"]], ["2026-04"])


class TopReactedMessagesTest(AnalyticsFixture):
    def test_ranking_and_breakdown_exclude_removals_and_strays(self):
        rows = query_tools.get_top_reacted_messages(chat_id=11)
        self.assertEqual(rows[0]["guid"], "guid-23")
        self.assertEqual(rows[0]["reaction_count"], 3)  # not 5: 3003/1000 excluded
        self.assertEqual(rows[0]["reactions_by_type"], {"laugh": 2, "love": 1})

    def test_reaction_type_by_name_and_code(self):
        by_name = query_tools.get_top_reacted_messages(chat_id=11, reaction_type="love")
        self.assertEqual({r["guid"] for r in by_name}, {"guid-23", "guid-21"})
        by_code = query_tools.get_top_reacted_messages(chat_id=11, reaction_type=2000)
        self.assertEqual({r["guid"] for r in by_name},
                         {r["guid"] for r in by_code})
        with self.assertRaises(ValueError):
            query_tools.get_top_reacted_messages(reaction_type="applause")

    def test_is_from_me_selects_whose_messages(self):
        mine = query_tools.get_top_reacted_messages(is_from_me=True)
        self.assertEqual({r["guid"] for r in mine}, {"guid-21", "guid-2"})

    def test_person_scope_and_empty_scope(self):
        rows = query_tools.get_top_reacted_messages(contact_id=5)
        self.assertTrue(rows)  # Bob's chats include the group
        self.assertEqual(query_tools.get_top_reacted_messages(contact_id=999999), [])


class ChatParticipantStatsTest(AnalyticsFixture):
    def test_group_rows_merge_identities_and_count_reactions(self):
        rows = query_tools.get_chat_participant_stats(chat_id=11)
        by_name = {r["participant"]: r for r in rows}
        self.assertIn("Me", by_name)
        # Carol: 2 sent, 3 reactions received on guid-23, 1 given (on guid-21)
        carol = by_name["+15550002222"]
        self.assertEqual(carol["messages_sent"], 2)
        self.assertEqual(carol["reactions_received"], 3)
        self.assertEqual(carol["reactions_received_by_type"], {"laugh": 2, "love": 1})
        self.assertEqual(carol["reactions_given"], 1)
        # Me: reaction 101 carries Carol's handle_id but is_from_me=1 -> mine
        self.assertEqual(by_name["Me"]["reactions_given"], 1)
        self.assertEqual(by_name["Me"]["reactions_received"], 1)
        # Bob reacted twice (100, 102), received none
        bob = by_name["Bob Jones"]
        self.assertEqual(bob["reactions_given"], 2)
        self.assertEqual(bob["reactions_received"], 0)

    def test_duplicate_handles_merge_into_one_row(self):
        rows = query_tools.get_chat_participant_stats(chat_id=[10, 11, 12])
        bobs = [r for r in rows if r["participant"] == "Bob Jones"]
        self.assertEqual(len(bobs), 1)
        self.assertEqual(sorted(bobs[0]["handle_ids"]), [60, 61])
        # 7 chat-10 rows (5 scripted + emote + system) + 1 group + 1 chat-12
        self.assertEqual(bobs[0]["messages_sent"], 9)

    def test_silent_member_still_listed(self):
        rows = query_tools.get_chat_participant_stats(chat_id=10)
        # Only Bob and Me have messages, but the row set seeds from
        # chat_participants, so exactly those two appear -- and a member with
        # zero messages in the window still shows up.
        windowed = query_tools.get_chat_participant_stats(
            chat_id=10, date_from="2027-01-01"
        )
        names = {r["participant"] for r in windowed}
        self.assertIn("Bob Jones", names)
        self.assertTrue(all(r["messages_sent"] == 0 for r in windowed))
        self.assertEqual({r["participant"] for r in rows} - {"Me"}, {"Bob Jones"})

    def test_chat_id_required(self):
        with self.assertRaises(ValueError):
            query_tools.get_chat_participant_stats(chat_id=None)


class WordFrequencyTest(AnalyticsFixture):
    def test_counts_and_stopword_exclusion(self):
        w = query_tools.get_word_frequency(chat_id=10, min_count=1)
        counts = {r["ngram"]: r["count"] for r in w["rows"]}
        self.assertEqual(counts.get("morning"), 2)
        self.assertNotIn("i", counts)  # stopword
        self.assertNotIn("loved", counts)   # emote row excluded
        self.assertNotIn("named", counts)   # system row excluded
        self.assertEqual(w["messages_scanned"], 8)

    def test_tokenizer_keeps_contractions_whole_across_apostrophes(self):
        # iOS types U+2019, not ASCII "'": both must tokenize identically and
        # never shed "don"/"t" fragments.
        self.assertEqual(analytics._tokenize("Don’t worry"), ["don't", "worry"])
        self.assertEqual(analytics._tokenize("don't worry"), ["don't", "worry"])
        self.assertIn("don't", analytics._STOPWORDS)

    def test_include_stopwords(self):
        w = query_tools.get_word_frequency(chat_id=10, min_count=1,
                                           include_stopwords=True)
        counts = {r["ngram"]: r["count"] for r in w["rows"]}
        self.assertIn("you", counts)

    def test_bigrams(self):
        w = query_tools.get_word_frequency(chat_id=10, ngram=2, min_count=1)
        counts = {r["ngram"]: r["count"] for r in w["rows"]}
        self.assertEqual(counts.get("morning bob"), 1)
        # all-stopword bigrams are dropped ("for the"); mixed ones survive
        self.assertNotIn("for the", counts)
        self.assertIn("you around", counts)

    def test_sender_scope_is_messages_they_sent(self):
        w = query_tools.get_word_frequency(contact_id=5, min_count=1)
        counts = {r["ngram"]: r["count"] for r in w["rows"]}
        self.assertIn("hey", counts)          # Bob sent this
        self.assertNotIn("bob", counts)       # only I say "bob"

    def test_is_from_me_scope_and_min_count(self):
        w = query_tools.get_word_frequency(is_from_me=True, min_count=2)
        counts = {r["ngram"]: r["count"] for r in w["rows"]}
        # Across all my messages only "bob" repeats; "morning" appears once
        # from me (Bob's "morning" is excluded by is_from_me=True) and "hey"
        # once, so min_count=2 drops both.
        self.assertEqual(counts, {"bob": 2})


if __name__ == "__main__":
    unittest.main()
