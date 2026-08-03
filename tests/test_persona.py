"""Hermetic tests for the INTERNAL persona helpers.

Covers query_tools/persona.py (resolve_voice_scope, _clip, _tone_block,
directed_reply_pairs/collect_pairs), query_tools/persona_draft.py
(_strip_ungrounded_terms) and query_tools/persona_sim.py (_window_reply_pairs,
_predict_tapback, _dormant_signal, the denylist refusal). The four PUBLIC tools
are already end-to-end tested in tests/test_voiceprint.py; this module tests
the helpers underneath them. Hand-scripted temp SQLite DBs -- no Ollama, no
real chat.db, no chunk_vectors (retrieval falls back to FTS when absent).

Every refusal/decline here also asserts ollama.chat_once was never called: a
decline that still dispatched to the model would be the actual defect.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import config
import ollama
import query_tools
import schema
from query_tools import persona, persona_draft, persona_sim


def _contact(con, cid, first):
    con.execute("INSERT INTO contacts(id, source_uniqueid, first_name) "
                "VALUES (?, ?, ?)", (cid, f"u{cid}", first))


def _dyad(con, chat_id, handle_id, contact_id, ident):
    con.execute("INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (?, ?, ?, 'iMessage')", (handle_id, ident, contact_id))
    con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                "service) VALUES (?, ?, NULL, 0, 'iMessage')",
                (chat_id, f"c{chat_id}"))
    con.execute("INSERT INTO chat_participants VALUES (?, ?)", (chat_id, handle_id))


def _msg(con, mid, chat_id, handle_id, from_me, text, date, chunk=None, is_emote=0):
    con.execute(
        "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
        "date, has_attachment, is_emote, item_type, chunk_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, ?)",
        (mid, f"g{mid}", chat_id, handle_id, from_me, text, date, is_emote, chunk))


def _chunk(con, cid, chat_id, first_mid, last_mid, day, n):
    con.execute(
        "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
        "first_date, last_date, n_messages, text) VALUES (?, ?, ?, ?, ?, ?, ?, 'x')",
        (cid, chat_id, first_mid, last_mid, day, day, n))


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
        # These fixtures are intentionally tiny (well under the production
        # VOICEPRINT_MIN_RETURN_MESSAGES floor); drop it to 0 so card assembly
        # behaves as it does on a real corpus.
        saved_floor = config.VOICEPRINT_MIN_RETURN_MESSAGES
        config.VOICEPRINT_MIN_RETURN_MESSAGES = 0
        self.addCleanup(setattr, config, "VOICEPRINT_MIN_RETURN_MESSAGES", saved_floor)

    def populate(self, con):  # overridden
        pass

    def _con(self):
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        return con

    def _set_config(self, **kwargs):
        for var, val in kwargs.items():
            saved = getattr(config, var)
            setattr(config, var, val)
            self.addCleanup(setattr, config, var, saved)


class ResolveVoiceScopeTest(_DBCase):
    """persona.resolve_voice_scope: the single home for the five (member_key,
    counterpart_key, scope) refusals shared by get_voiceprint and
    check_style_match, plus the scope default. It never runs a model."""

    def _refusal(self, scope, subject, person_key, chat_id):
        with mock.patch.object(ollama, "chat_once") as chat_once:
            member, counterpart, out_scope, err = persona.resolve_voice_scope(
                scope, subject, person_key, chat_id)
        chat_once.assert_not_called()  # a refusal never reaches the model
        self.assertIsNotNone(err)
        self.assertIn("error", err)
        # A refusal returns the error ALONE -- never a half-resolved triple.
        self.assertEqual((member, counterpart, out_scope), (None, None, None))
        return err

    def test_an_invalid_scope_is_refused(self):
        err = self._refusal("sideways", "me", "contact:12", None)
        self.assertIn("dyad, group, or all", err["error"])

    def test_dyad_scope_without_a_person_is_refused(self):
        err = self._refusal("dyad", "me", None, None)
        self.assertIn("dyad", err["error"])

    def test_all_scope_for_them_without_a_person_is_refused(self):
        err = self._refusal("all", "them", None, None)
        self.assertIn("needs a person", err["error"])
        # subject='me' at the same scope needs nobody -- the refusal is specific.
        self.assertEqual(
            persona.resolve_voice_scope("all", "me", None, None),
            ("me", "*", "all", None))

    def test_group_scope_without_a_chat_id_is_refused(self):
        err = self._refusal("group", "me", "contact:12", None)
        self.assertIn("group chat_id", err["error"])
        # A named person does NOT substitute for the chat_id, and a group with a
        # chat_id but subject='them' and no person is its own refusal.
        self._refusal("group", "them", None, 900)

    def test_scope_defaults_to_dyad_when_a_person_is_named(self):
        self.assertEqual(
            persona.resolve_voice_scope(None, "me", "contact:12", None),
            ("me", "contact:12", "dyad", None))
        # subject='them' flips the pair, never the scope.
        self.assertEqual(
            persona.resolve_voice_scope(None, "them", "contact:12", None),
            ("contact:12", "me", "dyad", None))
        # No person named -> 'all' (the whole-corpus voice), not a dyad refusal.
        self.assertEqual(
            persona.resolve_voice_scope(None, "me", None, None),
            ("me", "*", "all", None))


class ReplyPairTest(_DBCase):
    """The adjacency rule underneath every persona exemplar: each real message
    paired with the message from the requested speaker that IMMEDIATELY follows
    it in (date, id) order. Fixture: chat 100 with chunk 1 (four them->me pairs
    in January) and chunk 2 (two them->me pairs in May)."""

    def populate(self, con):
        _contact(con, 12, "Mom")
        _dyad(con, 100, 70, 12, "+1700")
        jan = [(1, 0, "aaa"), (2, 1, "bbb"), (3, 0, "ccc"), (4, 1, "ddd"),
               (5, 0, "eee"), (6, 1, "fff"), (7, 0, "ggg"), (8, 1, "hhh")]
        for mid, from_me, text in jan:
            _msg(con, mid, 100, 0 if from_me else 70, from_me, text,
                 f"2026-01-01T10:00:{mid:02d}", chunk=1)
        _chunk(con, 1, 100, 1, 8, "2026-01-01", 8)
        may = [(11, 0, "mmm"), (12, 1, "nnn"), (13, 0, "ooo"), (14, 1, "ppp")]
        for mid, from_me, text in may:
            _msg(con, mid, 100, 0 if from_me else 70, from_me, text,
                 f"2026-05-01T10:00:{mid:02d}", chunk=2)
        _chunk(con, 2, 100, 11, 14, "2026-05-01", 4)

    def test_collect_pairs_dedupes_on_the_incoming_and_reply_ids(self):
        con = self._con()
        # The same chunk reached twice (fusion can rank a chunk from both arms)
        # must contribute its pairs ONCE -- the key is (incoming_id, reply_id).
        once = persona.collect_pairs(con, [1], cap=50)
        twice = persona.collect_pairs(con, [1, 1], cap=50)
        self.assertEqual(len(once), 4)
        self.assertEqual([(p["incoming_id"], p["reply_id"]) for p in twice],
                         [(1, 2), (3, 4), (5, 6), (7, 8)])

    def test_collect_pairs_stops_at_the_cap_mid_chunk(self):
        con = self._con()
        pairs = persona.collect_pairs(con, [1, 2], cap=2)
        # The cap bites inside chunk 1 (four pairs) -- it returns the first two
        # and never walks on into chunk 2.
        self.assertEqual([(p["incoming"], p["reply"]) for p in pairs],
                         [("aaa", "bbb"), ("ccc", "ddd")])

    def test_the_window_returns_the_most_recent_pairs_not_the_oldest(self):
        con = self._con()
        pairs = persona_sim._window_reply_pairs(
            con, [100], None, None, reply_from_me=1, cap=2)
        # Sorted date DESC then sliced: the May pairs, newest first. Chronological
        # order here would silently ground past-self on the OLDEST exchanges.
        self.assertEqual([p["reply"] for p in pairs], ["ppp", "nnn"])
        self.assertNotIn("bbb", [p["reply"] for p in pairs])


class ClipTest(unittest.TestCase):
    """persona._clip shares its name with chat_apple._clip and behaves the
    opposite way on the two points that matter -- a real merge hazard."""

    def test_clip_keeps_the_body_length_and_passes_none_through(self):
        # None passes THROUGH (chat_apple._clip maps None to '').
        self.assertIsNone(persona._clip(None, 5))
        # Under the limit is untouched; at the limit no marker is added.
        self.assertEqual(persona._clip("abcde", 5), "abcde")
        # `limit` chars of BODY plus the marker (chat_apple._clip caps the TOTAL).
        self.assertEqual(persona._clip("abcdefgh", 5), "abcde…")


class ToneBlockTest(_DBCase):
    """persona._tone_block: tone is LOW-TRUST and llm-only, so a scope with no
    such annotations must be silent about tone rather than neutral about it."""

    def populate(self, con):
        _contact(con, 12, "Mom")
        _dyad(con, 100, 70, 12, "+1700")

    def test_a_null_tone_is_omitted_from_the_card_rather_than_reported_neutral(self):
        self.assertIsNone(persona._tone_block(None, None))
        self.assertIsNone(persona._tone_block(None, ""))
        card = persona._base_card(
            self._con(), "me", "contact:12", "dyad", 10, 4, 3,
            {"avg_msg_len": 12.0}, [], [], {}, persona._tone_block(None, None))
        # The KEY is absent -- no valence 0, no empty emotions list to read as calm.
        self.assertNotIn("tone", card)
        self.assertEqual(card["scope"], "dyad")

    def test_malformed_emotions_json_degrades_to_an_empty_list(self):
        block = persona._tone_block(-0.4, "{not json at all")
        self.assertEqual(block, {"valence": -0.4, "emotions": [], "low_trust": True})
        # A valence with NO emotions column still yields a block (llm rows exist).
        self.assertEqual(persona._tone_block(0.2, None),
                         {"valence": 0.2, "emotions": [], "low_trust": True})


class DormantSignalTest(_DBCase):
    """persona_sim._dormant_signal is a FAIL-OPEN net: it WARNS and never
    blocks, and both conditions are required to warn. Fixture: Mom (chat 100)
    goes quiet on 2026-01-01 and I send three unanswered messages after, the
    last on 2026-04-01 (the corpus's 'now'), so days_since is exactly 90.
    Chat 101 is a dateless dyad."""

    def populate(self, con):
        _contact(con, 12, "Mom")
        _dyad(con, 100, 70, 12, "+1700")
        _msg(con, 1, 100, 70, 0, "talk soon", "2026-01-01T10:00:00")
        for i, date in enumerate(("2026-02-01T10:00:00", "2026-03-01T10:00:00",
                                  "2026-04-01T10:00:00"), start=2):
            _msg(con, i, 100, 0, 1, "you there?", date)
        _contact(con, 13, "Al")
        _dyad(con, 101, 71, 13, "+1710")
        _msg(con, 20, 101, 71, 0, "no date on this one", None)

    def test_both_conditions_are_required_to_warn(self):
        con = self._con()
        self._set_config(PERSONA_SIM_DORMANT_DAYS=90, PERSONA_SIM_ONESIDED_MIN_MSGS=3)
        fired = persona_sim._dormant_signal(con, [100], [70])
        self.assertEqual(fired["days_since_their_last_message"], 90)
        self.assertEqual(fired["your_messages_since_with_no_reply"], 3)
        self.assertEqual(fired["last_message_from_them"], "2026-01-01")
        self.assertIn("passed away", fired["warning"])
        # One day short on age alone -> silent, even with the one-sidedness.
        config.PERSONA_SIM_DORMANT_DAYS = 91
        self.assertIsNone(persona_sim._dormant_signal(con, [100], [70]))
        # Old enough again, but one message short of one-sided -> still silent.
        config.PERSONA_SIM_DORMANT_DAYS = 90
        config.PERSONA_SIM_ONESIDED_MIN_MSGS = 4
        self.assertIsNone(persona_sim._dormant_signal(con, [100], [70]))

    def test_no_signal_when_the_dyad_has_no_chats(self):
        con = self._con()
        self._set_config(PERSONA_SIM_DORMANT_DAYS=1, PERSONA_SIM_ONESIDED_MIN_MSGS=1)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            self.assertIsNone(persona_sim._dormant_signal(con, [], [70]))
            self.assertIsNone(persona_sim._dormant_signal(con, [100], []))
        chat_once.assert_not_called()

    def test_no_signal_when_a_date_is_missing(self):
        con = self._con()
        self._set_config(PERSONA_SIM_DORMANT_DAYS=1, PERSONA_SIM_ONESIDED_MIN_MSGS=0)
        with mock.patch.object(ollama, "chat_once") as chat_once:
            # Al's only message carries no date -> no anchor to age from, so the
            # net stays quiet instead of dating the silence from a NULL.
            self.assertIsNone(persona_sim._dormant_signal(con, [101], [71]))
        chat_once.assert_not_called()


class StripUngroundedTermsTest(unittest.TestCase):
    """persona_draft._strip_ungrounded_terms: the dyad_terms post-check run in
    tool code on a generated draft. In-memory DB -- it only ever reads
    dyad_terms."""

    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.addCleanup(self.con.close)
        schema.create_schema(self.con)
        self.con.executemany(
            "INSERT INTO dyad_terms(chat_id, term) VALUES (?, ?)",
            # chat 100 is THIS dyad (grounded); chat 999 is another dyad, whose
            # terms form the suspect universe. "s/o" tokenizes to the 1-char
            # fragments 's' and 'o'. "don" is the 3-char fragment that used to
            # be stripped OUT OF the contraction "don’t".
            [(100, "wombat"), (999, "floopy"), (999, "s/o"), (999, "don")])
        self.con.commit()

    def _strip(self, draft):
        """Run the post-check and assert it never reached the model -- it is
        pure code, and that assertion IS the safety test."""
        with mock.patch.object(ollama, "chat_once") as chat_once:
            result = persona_draft._strip_ungrounded_terms(
                self.con, draft, [100], [], "sure thing")
        chat_once.assert_not_called()
        return result

    def test_short_tokens_are_never_stripped(self):
        # "s/o" (another dyad's term) tokenizes to the 1-char fragments 's' and
        # 'o'. Without the len >= 3 rule those would be removed wherever they
        # appear as standalone words, which is noise, not an inside joke.
        clean, stripped = self._strip("It’s okay wombat floopy s o")
        self.assertEqual(stripped, ["floopy"])   # only the len >= 3 ungrounded one
        self.assertEqual(clean, "It’s okay wombat s o")
        self.assertIn("It’s", clean)             # apostrophe intact, word whole

    def test_a_curly_apostrophe_contraction_is_not_split(self):
        # THE regression this fixture's "don" row exists for. function_words.
        # TOKEN_RE accepts both apostrophes, so "don’t" is ONE token and the
        # ungrounded 3-char fragment "don" is never matched inside it. With the
        # straight-only pattern this produced "I ’t know wombat" and reported
        # "don" as stripped -- a mangled draft handed to the user.
        curly = self._strip("I don’t know wombat floopy")
        self.assertEqual(curly, ("I don’t know wombat", ["floopy"]))
        # The two spellings must agree: the corpus carries both (iOS types the
        # curly one), and a post-check that depends on which is a coin flip.
        straight = self._strip("I don't know wombat floopy")
        self.assertEqual(straight, ("I don't know wombat", ["floopy"]))

    def test_a_term_stored_with_an_apostrophe_is_stripped_whole(self):
        # The other direction: when the ungrounded TERM itself contains an
        # apostrophe it is stripped entirely, from either spelling, because
        # both sides normalize curly -> straight (the set via tokenize(), the
        # draft via _repl).
        self.con.execute("INSERT INTO dyad_terms(chat_id, term) VALUES (999, 'y’all')")
        self.con.commit()
        self.assertEqual(self._strip("y’all coming wombat"),
                         ("coming wombat", ["y'all"]))
        self.assertEqual(self._strip("y'all coming wombat"),
                         ("coming wombat", ["y'all"]))

    def test_a_possessive_form_of_an_ungrounded_term_survives(self):
        # DELIBERATE consequence of matching whole tokens: "floopy’s" is not
        # the token "floopy", so it is left alone where the bare word is
        # stripped. Accepted rather than fixed -- the post-check is a
        # belt-and-braces net that already missed every other inflection, and
        # leaving one possessive standing is strictly safer than the stranding
        # that chasing it produced.
        self.assertEqual(self._strip("hey floopy, ok"), ("hey, ok", ["floopy"]))
        self.assertEqual(self._strip("floopy’s here"), ("floopy’s here", []))


class TapbackBaseRateTest(_DBCase):
    """persona_sim._predict_tapback: the base-rate denominator is EVERY real
    message received from the contact. Fixture: Mom sends 10 real messages (one
    emote row besides) and reacts twice -- 2 love, once laugh is not sent."""

    def populate(self, con):
        _contact(con, 12, "Mom")
        _dyad(con, 100, 70, 12, "+1700")
        for mid in range(1, 11):
            _msg(con, mid, 100, 70, 0, f"hers {mid}", f"2026-03-01T10:00:{mid:02d}")
        _msg(con, 11, 100, 70, 0, "Liked a message", "2026-03-01T10:00:11", is_emote=1)
        _msg(con, 12, 100, 0, 1, "mine", "2026-03-01T10:00:12")
        con.executemany(
            "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
            "is_from_me, associated_message_type, date) "
            "VALUES (?, 'g12', 100, 70, 0, ?, '2026-03-01T10:05:00')",
            [(901, 2000), (902, 2000)])

    def test_the_denominator_is_every_message_received_not_only_reacted_ones(self):
        tb = persona_sim._predict_tapback(self._con(), [100], [70])
        # 10 real messages from Mom -- the emote row is not one, and the 2
        # reacted-to messages are NOT the denominator (that would read 1.0).
        self.assertEqual(tb["denominator_messages"], 10)
        self.assertEqual(tb["most_likely"], "love")
        self.assertAlmostEqual(tb["base_rates"]["love"], 0.2, places=4)
        self.assertIn("not reacted-only", tb["note"])


class GenerationRefusalTest(_DBCase):
    """The deceased/grief denylist fires in code before ANY lookup or gate."""

    def populate(self, con):
        _contact(con, 12, "Mom")

    def test_a_blocked_contact_returns_the_fixed_message_without_calling_the_model(self):
        # Contact 12 has NO handle and NO chat: the block must land before the
        # chat lookup (which would 'error' instead) and before `acknowledged`
        # (which would ask for confirmation instead).
        self._set_config(PERSONA_SIM_BLOCK_IDS="contact:12")
        with mock.patch.object(ollama, "chat_once") as chat_once:
            res = persona_sim._simulate_them(
                self._con(), "hey", 12, None, None, acknowledged=False)
        chat_once.assert_not_called()
        self.assertTrue(res["refused"])
        self.assertEqual(res["reason"], "blocked_contact")
        self.assertFalse(res["generated"])
        # The FIXED constant verbatim -- never LLM-authored, never persisted.
        self.assertEqual(res["message"], persona_sim._SIM_BLOCK_MESSAGE)
        self.assertNotIn("error", res)
        self.assertNotIn("acknowledgment_required", res)


if __name__ == "__main__":
    unittest.main()
