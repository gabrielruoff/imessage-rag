"""query_tools/analytics_wellbeing.py (get_language_trajectory, RES-61 / W2):
the SQL + consent + grounding layer over the deterministic wellbeing core.

Hermetic: a temp sqlite file, hand-planted person_period_lexicon / messages /
chunks rows so every aggregate has one right answer; no Ollama, no real
chat.db. The change-point core (wellbeing.analyze_trajectory) has its own
hermetic suite (test_wellbeing.py); here it is stubbed to a fixed FLAGGED
result for the grounding/framing tests so the tool's own wiring -- count-based
aggregation, the consent gate, median-scoring target-lines-only excerpts, and
the envelope shape -- is what is under test.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import config
import query_tools
import schema
import wellbeing
from db import load_vec_extension
from query_tools import analytics_wellbeing

# FUNCTION_WORDS declaration order -> fw_*_rate column order; negation is 8th.
_NEG_IDX = 7


def _flagged_stub(*_args, **_kwargs):
    """Stand in for wellbeing.analyze_trajectory: a fixed concordant firing over
    2023-05..2023-07 with the absolutist_negation lexical family departed, so
    grounding keys off that family's vocabulary."""
    return {
        "flagged": True,
        "window": ["2023-05", "2023-07"],
        "peak_z": 4.25,
        "families": {
            "self_focus": {"departed": False, "direction": "stable", "markers": {}},
            "absolutist_negation": {"departed": True, "direction": "concerning",
                                    "markers": {}},
            "behavioral": {"departed": True, "direction": "concerning", "markers": {}},
        },
        "baseline_months": 12,
        "dropped_markers": [],
    }


class WellbeingFixture(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            load_vec_extension(con)
            schema.create_schema(con)
            con.commit()
        finally:
            con.close()

        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)

        # Tiny-fixture-friendly floors: 1 word, 1 marker hit. Individual tests
        # override consent / example count / floor as needed.
        for name, val in (("MIN_WELLBEING_WORDS", 1),
                          ("WELLBEING_MIN_MARKER_HITS", 1),
                          ("WELLBEING_N_EXAMPLES", 3),
                          ("WELLBEING_DENY_IDS", ""),
                          ("LOCAL_TZ", "")):
            p = mock.patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)

    # -- planting helpers --------------------------------------------------

    def _exec(self, sql, params=()):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def _lex(self, chat_id, handle_id, month, n_words, *, neg_rate=0.0,
             laugh=0, i=0, absol=0, emoji=None, n_messages=1):
        fw = [0.0] * 9
        fw[_NEG_IDX] = neg_rate
        self._exec(
            "INSERT INTO person_period_lexicon("
            "chat_id, handle_id, month, n_messages, n_words, avg_chars, "
            "fw_pronoun_rate, fw_ipron_rate, fw_article_rate, fw_conj_rate, "
            "fw_prep_rate, fw_aux_rate, fw_adverb_rate, fw_negation_rate, "
            "fw_quant_rate, we_count, question_count, laugh_count, emoji_json, "
            "i_count, you_count, absolutist_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (chat_id, handle_id, month, n_messages, n_words, 10.0, *fw,
             0, 0, laugh, emoji, i, 0, absol),
        )

    def _contact(self, cid, first, last):
        self._exec("INSERT INTO contacts(id, source_uniqueid, first_name, "
                   "last_name, nickname, organization) VALUES (?,?,?,?,NULL,NULL)",
                   (cid, f"uid-{cid}", first, last))

    def _handle(self, hid, contact_id, identifier="+1000"):
        self._exec("INSERT INTO handles(id, handle_identifier, contact_id, "
                   "service) VALUES (?,?,?,'iMessage')", (hid, identifier, contact_id))

    def _participant(self, chat_id, handle_id):
        self._exec("INSERT INTO chat_participants(chat_id, handle_id) VALUES (?,?)",
                   (chat_id, handle_id))

    def _msg(self, mid, chat_id, handle_id, from_me, text, date, chunk_id):
        self._exec(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, has_attachment, chunk_id) VALUES (?,?,?,?,?,?,?,0,?)",
            (mid, f"g{mid}", chat_id, handle_id, from_me, text, date, chunk_id))

    def _chunk(self, cid, chat_id, first_date, last_date):
        self._exec(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES (?,?,0,0,?,?,1,'x')",
            (cid, chat_id, first_date, last_date))

    def _call(self, **kwargs):
        return analytics_wellbeing.get_language_trajectory(**kwargs)


# --- count-based aggregation ------------------------------------------------

class AggregationTest(WellbeingFixture):
    def test_count_based_no_rate_of_rates(self):
        # Two SELF (handle NULL) rows, same month, different chats and word
        # counts. The month rate must be summed-counts / summed-words, NOT the
        # mean of the two per-row rates.
        self._lex(10, None, "2023-05", 100, neg_rate=0.10, i=5, absol=2,
                  emoji='{"\U0001F602": 3, "\U0001F642": 1}')
        self._lex(11, None, "2023-05", 300, neg_rate=0.02, i=15, absol=4, emoji=None)
        res = self._call(include=["series"])
        series = res["series"]
        # negation: (round(.10*100)+round(.02*300)) / 400 = 16/400 = 0.04
        self.assertEqual(series["fw_negation_rate"], [["2023-05", 0.04]])
        self.assertNotEqual(series["fw_negation_rate"][0][1],
                            round((0.10 + 0.02) / 2, 5))  # would be 0.06
        self.assertEqual(series["i_count"], [["2023-05", 0.05]])          # 20/400
        self.assertEqual(series["absolutist_count"], [["2023-05", 0.015]])  # 6/400

    def test_emoji_rate_count_then_divide_null_and_empty(self):
        # emoji count = summed emoji_json values / summed n_words; a NULL and an
        # empty-dict row contribute 0 numerator but their words still count.
        self._lex(10, None, "2023-06", 100, emoji='{"\U0001F602": 3, "\U0001F642": 1}')
        self._lex(10, None, "2023-06", 100, emoji=None)
        self._lex(11, None, "2023-06", 100, emoji="{}")
        res = self._call(include=["series"])
        self.assertEqual(res["series"]["emoji_rate"],
                         [["2023-06", round(4 / 300, 5)]])

    def test_below_min_words_returns_insufficient_envelope(self):
        self._lex(10, None, "2023-05", 200, neg_rate=0.10)
        with mock.patch.object(wellbeing, "analyze_trajectory") as spy:
            res = self._call(min_words=5000)
            spy.assert_not_called()  # no analysis below the floor
        self.assertEqual(res["scope"], "self")
        self.assertEqual(res["markers_available"], [])
        self.assertIsNone(res["series"])
        self.assertIsNone(res["windows"])
        self.assertIsNone(res["examples"])
        self.assertIsNone(res["support_block"])
        self.assertIn("insufficient", res["note"])
        self.assertEqual(res["baseline"], {"total_words": 200, "min_words": 5000})


# --- consent gate -----------------------------------------------------------

class ConsentTest(WellbeingFixture):
    def setUp(self):
        super().setUp()
        self._contact(5, "Bob", "Jones")
        self._handle(60, 5, "+15550001111")
        self._participant(10, 60)
        self._lex(10, 60, "2023-05", 100, neg_rate=0.10)  # target words exist

    def test_denylisted_contact_refused_in_code(self):
        # Denylist includes 5 -> refused, and analyze_trajectory is never reached.
        with mock.patch.object(config, "WELLBEING_DENY_IDS", "5 8"), \
                mock.patch.object(wellbeing, "analyze_trajectory") as spy:
            res = self._call(contact=5)
            spy.assert_not_called()
        self.assertTrue(res.get("refused"))
        self.assertEqual(res["scope"], "contact:5")
        self.assertEqual(res["note"], analytics_wellbeing._REFUSAL_MESSAGE)
        self.assertIsNone(res["series"])

    def test_contact_analyzed_by_default(self):
        # Default-open: an empty denylist analyzes any named contact.
        with mock.patch.object(config, "WELLBEING_DENY_IDS", ""), \
                mock.patch.object(wellbeing, "analyze_trajectory",
                                  side_effect=_flagged_stub) as spy:
            res = self._call(contact=5)
            spy.assert_called_once()
        self.assertEqual(res["scope"], "contact:5")

    def test_non_denylisted_contact_is_analyzed(self):
        # A denylist that does not include 5 still analyzes 5.
        with mock.patch.object(config, "WELLBEING_DENY_IDS", "7 8"), \
                mock.patch.object(wellbeing, "analyze_trajectory",
                                  side_effect=_flagged_stub) as spy:
            res = self._call(contact=5)
            spy.assert_called_once()
        self.assertEqual(res["scope"], "contact:5")


# --- grounding: median-scoring, target-lines-only ---------------------------

class GroundingSelfTest(WellbeingFixture):
    def setUp(self):
        super().setUp()
        # Enough words to clear the floor; analyze_trajectory is stubbed flagged.
        self._lex(10, None, "2023-05", 100, neg_rate=0.10)
        # Four SELF chunks in the window with differing negation density.
        # composites: 0.0, 2/3, 1.0 (max), 0.5 ; median of the four = 0.5835,
        # so the two nearest are 2/3 and 0.5 -- the max (1.0) and min (0.0) are
        # excluded.
        self._chunk(700, 10, "2023-05-10T10:00:00", "2023-05-10T10:05:00")
        self._chunk(701, 10, "2023-05-20T10:00:00", "2023-05-20T10:05:00")
        self._chunk(702, 10, "2023-06-05T10:00:00", "2023-06-05T10:05:00")
        self._chunk(703, 10, "2023-06-20T10:00:00", "2023-06-20T10:05:00")
        self._msg(1, 10, 0, 1, "okay sounds good", "2023-05-10T10:00:00", 700)
        self._msg(2, 10, 0, 1, "not really no", "2023-05-20T10:00:00", 701)
        # A them-line inside 701: must be stripped from the composite AND excerpt.
        self._msg(3, 10, 60, 0, "never nothing always", "2023-05-20T10:01:00", 701)
        self._msg(4, 10, 0, 1, "no not never", "2023-06-05T10:00:00", 702)
        self._msg(5, 10, 0, 1, "not sure", "2023-06-20T10:00:00", 703)

    def test_examples_are_median_not_max_and_target_lines_only(self):
        with mock.patch.object(wellbeing, "analyze_trajectory",
                               side_effect=_flagged_stub), \
                mock.patch.object(config, "WELLBEING_N_EXAMPLES", 2):
            res = self._call()
        chunk_ids = {e["chunk_id"] for e in res["examples"]}
        self.assertEqual(chunk_ids, {701, 703})   # the two nearest the median
        self.assertNotIn(702, chunk_ids)          # never the max-negative chunk
        self.assertNotIn(700, chunk_ids)          # nor the min
        by_id = {e["chunk_id"]: e for e in res["examples"]}
        # 701: only the target's own line; the them-line is gone; composite 2/3.
        self.assertEqual(by_id[701]["excerpt_lines"], ["not really no"])
        self.assertEqual(by_id[701]["n_lines"], 1)
        self.assertEqual(by_id[701]["composite"], round(2 / 3, 5))
        joined = " ".join(ln for e in res["examples"] for ln in e["excerpt_lines"])
        self.assertNotIn("never nothing always", joined)

    def test_self_framing_scope_and_support_block(self):
        with mock.patch.object(wellbeing, "analyze_trajectory",
                               side_effect=_flagged_stub):
            res = self._call()
        self.assertEqual(res["scope"], "self")
        self.assertEqual(len(res["windows"]), 1)
        self.assertIn("your language", res["windows"][0]["framing"])
        self.assertEqual(res["windows"][0]["start"], "2023-05")
        self.assertEqual(res["windows"][0]["end"], "2023-07")
        self.assertEqual(res["support_block"], wellbeing.support_block())


class GroundingContactTest(WellbeingFixture):
    def setUp(self):
        super().setUp()
        self._contact(5, "Bob", "Jones")
        self._handle(60, 5, "+15550001111")
        self._participant(10, 60)
        self._lex(10, 60, "2023-05", 100, neg_rate=0.10)  # the contact's words
        self._chunk(710, 10, "2023-06-01T10:00:00", "2023-06-01T10:05:00")
        # The contact's own line (is_from_me=0, their handle) and MY line, same
        # chunk: only the contact's line grounds and appears in the excerpt.
        self._msg(10, 10, 60, 0, "not really no", "2023-06-01T10:00:00", 710)
        self._msg(11, 10, 0, 1, "no not never", "2023-06-01T10:01:00", 710)

    def test_contact_framing_and_own_lines_only(self):
        # Default-open denylist (empty, from setUp) -> contact 5 is analyzed.
        with mock.patch.object(wellbeing, "analyze_trajectory",
                               side_effect=_flagged_stub):
            res = self._call(contact=5)
        self.assertEqual(res["scope"], "contact:5")
        self.assertIn("Bob Jones's language", res["windows"][0]["framing"])
        (example,) = res["examples"]
        self.assertEqual(example["chunk_id"], 710)
        self.assertEqual(example["excerpt_lines"], ["not really no"])  # not mine
        self.assertEqual(example["composite"], round(2 / 3, 5))


# --- envelope shape / include -----------------------------------------------

class EnvelopeTest(WellbeingFixture):
    PINNED_KEYS = {"scope", "markers_available", "dropped_markers", "baseline",
                   "series", "windows", "examples", "note", "support_block"}

    def setUp(self):
        super().setUp()
        self._lex(10, None, "2023-05", 100, neg_rate=0.10, i=5, absol=2)

    def test_all_pinned_keys_present_when_not_flagged(self):
        res = self._call()
        self.assertTrue(self.PINNED_KEYS <= set(res))
        self.assertEqual(res["scope"], "self")
        self.assertIsNone(res["support_block"])   # nothing fired
        self.assertEqual(res["windows"], [])
        self.assertEqual(res["examples"], [])
        self.assertIn("reply-latency", res["note"])  # documented v1 omission

    def test_include_selects_sections(self):
        res = self._call(include=["windows"])
        self.assertIsNotNone(res["windows"])
        self.assertIsNone(res["series"])
        self.assertIsNone(res["examples"])
        self.assertIsNone(res["baseline"])

    def test_unknown_include_section_raises(self):
        with self.assertRaises(ValueError):
            self._call(include=["windows", "bogus"])


# --- 3-place registration ---------------------------------------------------

class RegistrationTest(unittest.TestCase):
    def test_tool_is_registered_in_all_three_places(self):
        import chat
        from tool_schemas import TOOL_DEFINITIONS
        names = {t["name"] for t in TOOL_DEFINITIONS}
        self.assertIn("get_language_trajectory", names)
        self.assertTrue(hasattr(query_tools, "get_language_trajectory"))
        self.assertIn("get_language_trajectory", chat.TOOL_FUNCTIONS)
        self.assertIn("get_language_trajectory", chat.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
