"""lexicon_build.py + query_tools/analytics_language.py: the person-period
lexicon and dyad-terms pipelines, and the get_word_trends /
get_group_dynamics tools -- all against a hand-scripted fixture DB with
known dates and token counts, so every aggregate has one right answer.
Hermetic: temp sqlite file, no Ollama, no real chat.db. sessions and
chunk_annotations rows are hand-planted (their pipelines live in other
modules and are not under test here)."""

import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import lexicon_build
import query_tools
import schema
from db import load_vec_extension
from query_tools import analytics_language


def _msg(mid, chat, handle, from_me, text, date):
    return (mid, f"guid-{mid}", chat, handle, from_me, text, date, 0)


class LanguageFixture(unittest.TestCase):
    """Chat 10: a 1:1 with Bob (contact 5, handle 60) -- lexicon/dyad terms.
    Chat 11: a group with Bob, Carol (62, no contact), silent Dave (63).
    Chat 12: a second 1:1 with Bob whose two sides send identical text
    (LSM must score 1.0)."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            load_vec_extension(con)
            schema.create_schema(con)
            con.execute(
                "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
                "nickname, organization) VALUES (5, 'uid-5', 'Bob', 'Jones', NULL, NULL)"
            )
            con.executemany(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (?, ?, ?, 'iMessage')",
                [
                    (60, "+15550001111", 5),     # Bob
                    (62, "+15550002222", None),  # Carol, no contact row
                    (63, "+15550003333", None),  # Dave, never speaks
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
                [(10, 60), (11, 60), (11, 62), (11, 63), (12, 60)],
            )
            # Chat 10 -- known token counts (26 tokens total):
            #   2026-03 me:   m2 (5 tokens, we+?, no laugh), m3 (2 tokens, lol)
            #   2026-03 them: m1 (4 tokens, we + 2x laughing emoji)
            #   2026-04: "blorptastic" x3 (coined by Bob at m5),
            #            "zonk fest" x2 (coined by me at m8)
            chat10 = [
                _msg(1, 10, 60, 0, "we should get luaus \U0001F602\U0001F602",
                     "2026-03-01T10:00:00"),
                _msg(2, 10, 0, 1, "are we doing luau tuesday?", "2026-03-01T10:01:00"),
                _msg(3, 10, 0, 1, "lol yes", "2026-03-02T09:00:00"),
                _msg(4, 10, 60, 0, "luau tuesday again?", "2026-04-05T12:00:00"),
                _msg(5, 10, 60, 0, "blorptastic", "2026-04-05T12:01:00"),
                _msg(6, 10, 60, 0, "so blorptastic", "2026-04-06T12:00:00"),
                _msg(7, 10, 0, 1, "blorptastic indeed", "2026-04-06T12:05:00"),
                _msg(8, 10, 0, 1, "zonk fest tonight", "2026-04-07T20:00:00"),
                _msg(9, 10, 0, 1, "zonk fest was great", "2026-04-08T09:00:00"),
            ]
            # Chat 11 -- equal shares (2 msgs each), adjacency-friendly gaps
            # except the 16-minute one before m23 (ADJACENCY_MINUTES=5).
            chat11 = [
                _msg(20, 11, 0, 1, "kickoff", "2026-05-01T12:00:00"),
                _msg(21, 11, 60, 0, "sounds good", "2026-05-01T12:02:00"),
                _msg(22, 11, 62, 0, "im in", "2026-05-01T12:04:00"),
                _msg(23, 11, 0, 1, "great", "2026-05-01T12:20:00"),
                _msg(24, 11, 60, 0, "pizza too", "2026-05-01T12:21:00"),
                _msg(25, 11, 62, 0, "pizza yes", "2026-05-01T12:22:00"),
            ]
            # Chat 12 -- both sides identical text.
            chat12 = [
                _msg(30, 12, 60, 0, "we are totally here now", "2026-06-01T10:00:00"),
                _msg(31, 12, 0, 1, "we are totally here now", "2026-06-01T10:01:00"),
            ]
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                chat10 + chat11 + chat12,
            )
            # Rows the pipelines must skip: emote, system, and dateless.
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment, is_emote, item_type) "
                "VALUES (40, 'guid-40', 10, 60, 0, 'emoteword emoteword emoteword?', "
                "'2026-03-01T10:06:00', 0, 1, 0)"
            )
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment, is_emote, item_type) "
                "VALUES (41, 'guid-41', 10, 60, 0, 'sysword named the group', "
                "'2026-03-01T10:07:00', 0, 0, 2)"
            )
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) "
                "VALUES (42, 'guid-42', 10, 60, 0, 'dateless blorptastic', NULL, 0)"
            )
            # Hand-planted sessions for chat 11: Carol starts 2, me 1.
            con.executemany(
                "INSERT INTO sessions(chat_id, first_message_id, last_message_id, "
                "first_date, last_date, n_messages, n_from_me, initiator_handle_id, "
                "initiator_is_me, ender_handle_id, ender_is_me) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (11, 20, 25, "2026-05-01T12:00:00", "2026-05-01T12:22:00",
                     6, 2, None, 1, 62, 0),
                    (11, 20, 25, "2026-05-02T09:00:00", "2026-05-02T09:30:00",
                     4, 1, 62, 0, None, 1),
                    (11, 20, 25, "2026-05-03T09:00:00", "2026-05-03T09:30:00",
                     3, 1, 62, 0, None, 1),
                ],
            )
            # Hand-planted chunks + annotations for chat 10 only.
            con.executemany(
                "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
                "first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (500, 10, 1, 3, "2026-03-01T10:00:00", "2026-03-02T09:00:00",
                     3, "chunk a"),
                    (501, 10, 4, 9, "2026-04-05T12:00:00", "2026-04-08T09:00:00",
                     6, "chunk b"),
                ],
            )
            con.executemany(
                "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, "
                "depth, events_json) VALUES (?, ?, ?, ?, ?)",
                [
                    (500, 0.5, '["joy", "amusement"]', 2, "[]"),
                    (501, -0.5, '["sadness", "joy"]', 3, "[]"),
                ],
            )
            con.commit()
        finally:
            con.close()

        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)

    # -- helpers -----------------------------------------------------------

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _build_lexicon(self):
        con = self._connect()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return lexicon_build.build_person_period_lexicon(con)
        finally:
            con.close()

    def _build_dyads(self, **kwargs):
        """Build dyad_terms with fixture-friendly thresholds. _CORPUS_TOP_N is
        patched to 0: in a corpus this small every token is 'top 200'."""
        kwargs.setdefault("min_uses", 2)
        kwargs.setdefault("min_lift", 1.0)
        kwargs.setdefault("top_k", 10)
        con = self._connect()
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(lexicon_build, "_CORPUS_TOP_N", 0):
                return lexicon_build.build_dyad_terms(con, **kwargs)
        finally:
            con.close()

    def _rows(self, sql, params=()):
        con = self._connect()
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def _execute(self, sql, params=()):
        con = self._connect()
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()


class PersonPeriodLexiconTest(LanguageFixture):
    def test_me_side_keyed_null_handle_with_counts_and_rates(self):
        self._build_lexicon()
        (row,) = self._rows(
            "SELECT * FROM person_period_lexicon "
            "WHERE chat_id = 10 AND handle_id IS NULL AND month = '2026-03'"
        )
        # m2 + m3: 7 tokens, avg chars (26 + 7) / 2.
        self.assertEqual(row["n_messages"], 2)
        self.assertEqual(row["n_words"], 7)
        self.assertAlmostEqual(row["avg_chars"], 16.5)
        self.assertEqual(row["we_count"], 1)        # "we" in m2
        self.assertEqual(row["question_count"], 1)  # "?" in m2
        self.assertEqual(row["laugh_count"], 1)     # "lol" in m3
        self.assertAlmostEqual(row["fw_pronoun_rate"], 1 / 7)  # we
        self.assertAlmostEqual(row["fw_aux_rate"], 2 / 7)      # are, doing

    def test_them_side_emoji_json_and_emoji_laugh(self):
        self._build_lexicon()
        (row,) = self._rows(
            "SELECT * FROM person_period_lexicon "
            "WHERE chat_id = 10 AND handle_id = 60 AND month = '2026-03'"
        )
        # Only m1: the emote (40), system (41), and dateless (42) rows are out.
        self.assertEqual(row["n_messages"], 1)
        self.assertEqual(row["n_words"], 4)
        self.assertEqual(row["we_count"], 1)
        self.assertEqual(row["question_count"], 0)
        self.assertEqual(row["laugh_count"], 1)  # laughing emoji counts
        self.assertEqual(row["emoji_json"], '{"\U0001F602": 2}')

    def test_full_rebuild_is_idempotent_and_excludes_junk_rows(self):
        n_first = self._build_lexicon()
        n_second = self._build_lexicon()
        self.assertEqual(n_first, n_second)
        total = self._rows("SELECT COUNT(*) AS n FROM person_period_lexicon")[0]["n"]
        self.assertEqual(total, n_second)
        # No emote/system vocabulary anywhere (zero rows would carry them).
        rates = self._rows(
            "SELECT SUM(n_messages) AS n FROM person_period_lexicon WHERE chat_id = 10"
        )
        self.assertEqual(rates[0]["n"], 9)  # the 9 real chat-10 messages


class DyadTermsTest(LanguageFixture):
    def test_unigram_lift_and_them_provenance(self):
        self._build_dyads()
        (row,) = self._rows(
            "SELECT * FROM dyad_terms WHERE chat_id = 10 AND term = 'blorptastic'")
        self.assertEqual(row["n_uses"], 3)
        # 26 tokens in chat 10, 20 elsewhere, 0 uses elsewhere (+1 smoothing).
        self.assertAlmostEqual(row["lift"], (3 / 26) / (1 / 20))
        self.assertEqual(row["first_message_id"], 5)
        self.assertEqual(row["coined_by_handle_id"], 60)
        self.assertEqual(row["coined_is_me"], 0)
        self.assertIsNone(row["llm_verdict"])

    def test_bigram_counting_stopword_gate_and_me_provenance(self):
        self._build_dyads()
        (row,) = self._rows(
            "SELECT * FROM dyad_terms WHERE chat_id = 10 AND term = 'zonk fest'")
        self.assertEqual(row["n_uses"], 2)  # adjacent pairs in m8 and m9
        self.assertEqual(row["first_message_id"], 8)
        self.assertIsNone(row["coined_by_handle_id"])  # NULL = me
        self.assertEqual(row["coined_is_me"], 1)
        # Bigrams with a stopword token never become candidates.
        terms = {r["term"] for r in self._rows("SELECT term FROM dyad_terms")}
        self.assertNotIn("was great", terms)      # "was" is aux
        self.assertNotIn("so blorptastic", terms)  # "so" is conj

    def test_min_uses_and_min_lift_filtering(self):
        self._build_dyads(min_uses=3)
        terms = {r["term"] for r in self._rows(
            "SELECT term FROM dyad_terms WHERE chat_id = 10")}
        self.assertEqual(terms, {"blorptastic"})  # everything else has 2 uses
        self._build_dyads(min_lift=1e9)
        self.assertEqual(self._rows("SELECT COUNT(*) AS n FROM dyad_terms")[0]["n"], 0)

    def test_top_k_keeps_highest_lift_per_chat(self):
        self._build_dyads(top_k=1)
        rows = self._rows("SELECT chat_id, term FROM dyad_terms WHERE chat_id = 10")
        self.assertEqual([(r["chat_id"], r["term"]) for r in rows],
                         [(10, "blorptastic")])

    def test_verdict_preserved_across_rebuild(self):
        self._build_dyads()
        self._execute(
            "UPDATE dyad_terms SET llm_verdict = 'inside_joke' "
            "WHERE chat_id = 10 AND term = 'blorptastic'")
        self._build_dyads()  # full rebuild
        (row,) = self._rows(
            "SELECT llm_verdict FROM dyad_terms WHERE chat_id = 10 AND term = 'blorptastic'")
        self.assertEqual(row["llm_verdict"], "inside_joke")
        others = self._rows(
            "SELECT COUNT(*) AS n FROM dyad_terms WHERE llm_verdict IS NOT NULL")
        self.assertEqual(others[0]["n"], 1)


class WordTrendsTest(LanguageFixture):
    def test_phrase_trend_split_by_period_and_side(self):
        r = analytics_language.get_word_trends(
            phrase="luau tuesday", chat_id=10, granularity="month",
            include=["phrase_trend"])
        self.assertEqual(r["phrase_trend"]["series"], [
            {"period": "2026-03", "me": 1, "them": 0},
            {"period": "2026-04", "me": 0, "them": 1},
        ])
        self.assertEqual(r["phrase_trend"]["total_messages"], 2)
        self.assertNotIn("rates", r)  # include selected one section

    def test_emoji_top(self):
        self._build_lexicon()
        r = analytics_language.get_word_trends(chat_id=10, include=["emoji"])
        self.assertEqual(r["emoji"]["top"][0], {"emoji": "\U0001F602", "count": 2})
        self.assertEqual(r["emoji"]["total_emoji"], 2)

    def test_rates_per_100_me_vs_them(self):
        self._build_lexicon()
        r = analytics_language.get_word_trends(
            chat_id=10, granularity="month", include=["rates"])
        by_period = {row["period"]: row for row in r["rates"]["series"]}
        march = by_period["2026-03"]
        self.assertEqual(march["me"]["n_messages"], 2)
        self.assertEqual(march["me"]["we_per_100"], 50.0)
        self.assertEqual(march["me"]["question_per_100"], 50.0)
        self.assertEqual(march["me"]["laugh_per_100"], 50.0)
        self.assertEqual(march["them"]["n_messages"], 1)
        self.assertEqual(march["them"]["we_per_100"], 100.0)
        self.assertEqual(march["them"]["question_per_100"], 0.0)
        self.assertEqual(march["them"]["laugh_per_100"], 100.0)

    def test_distinctive_terms_curated_first_with_birth_certificate(self):
        self._build_dyads()
        self._execute("UPDATE dyad_terms SET llm_verdict = 'inside_joke' "
                      "WHERE chat_id = 10 AND term = 'blorptastic'")
        self._execute("UPDATE dyad_terms SET llm_verdict = 'ordinary' "
                      "WHERE chat_id = 10 AND term = 'luau'")
        r = analytics_language.get_word_trends(chat_id=10, include=["distinctive_terms"])
        terms = r["distinctive_terms"]["terms"]
        self.assertEqual(terms[0]["term"], "blorptastic")
        self.assertTrue(terms[0]["curated"])
        self.assertEqual(terms[0]["verdict"], "inside_joke")
        # Birth certificate: the coining message.
        self.assertEqual(terms[0]["first_message_id"], 5)
        self.assertEqual(terms[0]["date"], "2026-04-05T12:01:00")
        self.assertEqual(terms[0]["text"], "blorptastic")
        self.assertEqual(terms[0]["coined_by"], "Bob Jones")
        names = [t["term"] for t in terms]
        self.assertNotIn("luau", names)  # LLM-rejected verdicts are dropped
        self.assertTrue(all(not t["curated"] for t in terms[1:]))

    def test_distinctive_terms_requires_chat_scope(self):
        self._build_dyads()
        r = analytics_language.get_word_trends(include=["distinctive_terms"])
        self.assertFalse(r["distinctive_terms"]["available"])

    def test_lsm_identical_sides_scores_one(self):
        self._build_lexicon()
        r = analytics_language.get_word_trends(chat_id=12, include=["lsm"])
        self.assertAlmostEqual(r["lsm"]["score"], 1.0, places=6)
        for value in r["lsm"]["per_category"].values():
            self.assertAlmostEqual(value, 1.0, places=6)
        self.assertEqual(r["lsm"]["trajectory"],
                         [{"period": "2026-06", "score": 1.0}])

    def test_lsm_unavailable_without_lexicon(self):
        r = analytics_language.get_word_trends(chat_id=12, include=["lsm"])
        self.assertFalse(r["lsm"]["available"])

    def test_emotions_and_valence_sections(self):
        r = analytics_language.get_word_trends(
            chat_id=10, granularity="month", include=["emotions", "valence"])
        self.assertEqual(r["emotions"]["top"][0], {"emotion": "joy", "count": 2})
        self.assertEqual(r["valence"]["series"], [
            {"period": "2026-03", "mean_valence": 0.5, "n_chunks": 1},
            {"period": "2026-04", "mean_valence": -0.5, "n_chunks": 1},
        ])

    def test_emotions_and_valence_unavailable_when_unannotated(self):
        r = analytics_language.get_word_trends(
            chat_id=11, include=["emotions", "valence"])
        self.assertFalse(r["emotions"]["available"])
        self.assertFalse(r["valence"]["available"])

    def test_person_scope_resolves_to_their_chats(self):
        self._build_lexicon()
        r = analytics_language.get_word_trends(contact_id=5, include=["emoji"])
        self.assertEqual(sorted(r["chat_ids"]), [10, 11, 12])
        r = analytics_language.get_word_trends(contact_id=999999, include=["emoji"])
        self.assertEqual(r["note"], "scope matched nothing")

    def test_unknown_include_section_raises(self):
        # A typo'd section must never come back as a plausible partial result.
        with self.assertRaises(ValueError):
            analytics_language.get_word_trends(chat_id=10, include=["emoji", "bogus"])


class GroupDynamicsTest(LanguageFixture):
    def test_error_on_direct_chat_and_missing_chat_id(self):
        r = analytics_language.get_group_dynamics(chat_id=10)
        self.assertIn("error", r)
        with self.assertRaises(ValueError):
            analytics_language.get_group_dynamics(chat_id=None)

    def test_every_chat_in_the_list_is_validated_not_just_the_first(self):
        # Chat 10 is 1:1: smuggled in behind a group id it would silently
        # skew every member share, so the whole call must be refused.
        r = analytics_language.get_group_dynamics(chat_id=[11, 10])
        self.assertIn("error", r)
        self.assertIn("1:1", r["error"])
        r = analytics_language.get_group_dynamics(chat_id=[11, 999999])
        self.assertIn("error", r)
        self.assertIn("not found", r["error"])

    def test_unknown_include_section_raises(self):
        with self.assertRaises(ValueError):
            analytics_language.get_group_dynamics(chat_id=11, include=["members", "bogus"])

    def test_members_share_and_word_counts(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["members"])
        by_name = {m["member"]: m for m in r["members"]}
        self.assertEqual(set(by_name), {"Me", "Bob Jones", "+15550002222"})
        for member in by_name.values():
            self.assertEqual(member["messages"], 2)
            self.assertAlmostEqual(member["share"], 0.333, places=3)
        self.assertEqual(by_name["Me"]["words"], 2)          # kickoff + great
        self.assertEqual(by_name["Bob Jones"]["words"], 4)
        self.assertEqual(by_name["+15550002222"]["words"], 4)
        self.assertTrue(by_name["Me"]["is_me"])

    def test_gini_zero_for_equal_shares_with_drift(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["gini"])
        self.assertEqual(r["gini"]["gini"], 0.0)
        self.assertEqual(r["gini"]["n_members"], 3)
        self.assertEqual(r["gini"]["drift"],
                         [{"period": "2026", "gini": 0.0, "n_members": 3}])

    def test_lurkers_lists_silent_participant(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["lurkers"])
        self.assertEqual(r["lurkers"], [{"member": "+15550003333", "messages": 0}])

    def test_igniters_from_sessions_and_unavailable_when_empty(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["igniters"])
        self.assertEqual(r["igniters"]["n_sessions"], 3)
        self.assertEqual(r["igniters"]["leaderboard"][0],
                         {"member": "+15550002222", "sessions_started": 2,
                          "share": 0.667})
        self.assertEqual(r["igniters"]["leaderboard"][1]["member"], "Me")
        self._execute("DELETE FROM sessions")
        r = analytics_language.get_group_dynamics(chat_id=11, include=["igniters"])
        self.assertFalse(r["igniters"]["available"])

    def test_adjacency_pairs_respect_gap_gate(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["adjacency"])
        pairs = {(p["replier"], p["replied_to"]): p["count"]
                 for p in r["adjacency"]["pairs"]}
        # m23 follows m22 by 16 min > ADJACENCY_MINUTES: no Me -> Carol pair.
        self.assertEqual(pairs, {
            ("Bob Jones", "Me"): 2,
            ("+15550002222", "Bob Jones"): 2,
        })

    def test_group_lsm_per_member_and_unavailable_without_lexicon(self):
        r = analytics_language.get_group_dynamics(chat_id=11, include=["group_lsm"])
        self.assertFalse(r["group_lsm"]["available"])
        self._build_lexicon()
        r = analytics_language.get_group_dynamics(chat_id=11, include=["group_lsm"])
        members = r["group_lsm"]["members"]
        self.assertEqual({m["member"] for m in members},
                         {"Me", "Bob Jones", "+15550002222"})
        for m in members:
            self.assertGreaterEqual(m["lsm_score"], 0.0)
            self.assertLessEqual(m["lsm_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
