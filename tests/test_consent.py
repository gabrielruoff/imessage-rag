"""Hermetic tests for consent.py (the shared opt-in id-list helper). Pure
stdlib -- no DB, no Ollama, no config. Fixtures are inline strings, exactly the
shapes a raw env-var value takes, so the parser's tolerance is pinned here."""

import sys
import unittest

import consent


class TestParseConsentIds(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(consent.parse_consent_ids(None), frozenset())

    def test_empty_string_is_empty(self):
        self.assertEqual(consent.parse_consent_ids(""), frozenset())

    def test_whitespace_only_is_empty(self):
        self.assertEqual(consent.parse_consent_ids("   \t \n "), frozenset())

    def test_mixed_comma_and_space(self):
        # The pinned acceptance fixture: "12, 44 88" -> {12, 44, 88}.
        self.assertEqual(consent.parse_consent_ids("12, 44 88"),
                         frozenset({12, 44, 88}))

    def test_comma_separated(self):
        self.assertEqual(consent.parse_consent_ids("12,44,88"),
                         frozenset({12, 44, 88}))

    def test_space_separated(self):
        self.assertEqual(consent.parse_consent_ids("12 44 88"),
                         frozenset({12, 44, 88}))

    def test_malformed_tokens_skipped_no_raise(self):
        # "12,abc, ,44" -> {12, 44}; the non-numeric and empty tokens are
        # dropped, and nothing is raised (a bad .env must never crash a build).
        self.assertEqual(consent.parse_consent_ids("12,abc, ,44"),
                         frozenset({12, 44}))

    def test_duplicates_collapse(self):
        self.assertEqual(consent.parse_consent_ids("12,12, 12 44"),
                         frozenset({12, 44}))

    def test_negative_and_zero_skipped(self):
        self.assertEqual(consent.parse_consent_ids("0, -5, 12, -0"),
                         frozenset({12}))

    def test_surrounding_whitespace_and_trailing_comma(self):
        self.assertEqual(consent.parse_consent_ids("  12 , 44 ,  "),
                         frozenset({12, 44}))

    def test_returns_frozenset(self):
        self.assertIsInstance(consent.parse_consent_ids("12"), frozenset)

    def test_all_malformed_is_empty(self):
        self.assertEqual(consent.parse_consent_ids("abc, ., --"), frozenset())


class TestIsConsented(unittest.TestCase):
    def test_member_is_consented(self):
        self.assertTrue(consent.is_consented(44, "12, 44 88"))

    def test_non_member_is_not_consented(self):
        self.assertFalse(consent.is_consented(99, "12, 44 88"))

    def test_none_contact_id_is_false(self):
        self.assertFalse(consent.is_consented(None, "12, 44 88"))

    def test_none_id_false_even_when_list_empty(self):
        self.assertFalse(consent.is_consented(None, None))

    def test_empty_list_consents_to_nothing(self):
        self.assertFalse(consent.is_consented(12, ""))
        self.assertFalse(consent.is_consented(12, None))

    def test_malformed_list_still_matches_valid_ids(self):
        self.assertTrue(consent.is_consented(44, "12,abc, ,44"))
        self.assertFalse(consent.is_consented(0, "0, 12"))


class TestIsDenied(unittest.TestCase):
    def test_member_is_denied(self):
        self.assertTrue(consent.is_denied(44, "12, 44 88"))

    def test_non_member_is_not_denied(self):
        # Default-open: absence from the deny list means allowed.
        self.assertFalse(consent.is_denied(99, "12, 44 88"))

    def test_empty_list_denies_nothing(self):
        self.assertFalse(consent.is_denied(12, ""))
        self.assertFalse(consent.is_denied(12, None))

    def test_none_contact_id_is_never_denied(self):
        self.assertFalse(consent.is_denied(None, "12, 44 88"))
        self.assertFalse(consent.is_denied(None, None))

    def test_malformed_list_still_matches_valid_ids(self):
        self.assertTrue(consent.is_denied(44, "12,abc, ,44"))
        self.assertFalse(consent.is_denied(99, "12,abc, ,44"))


class TestParseMemberBlockIds(unittest.TestCase):
    def test_none_and_empty_are_empty(self):
        self.assertEqual(consent.parse_member_block_ids(None), frozenset())
        self.assertEqual(consent.parse_member_block_ids("  "), frozenset())

    def test_bare_int_maps_to_contact(self):
        # Back-compat: the historical PERSONA_SIM_BLOCK_IDS format was contact ids.
        self.assertEqual(consent.parse_member_block_ids("12, 44"),
                         frozenset({"contact:12", "contact:44"}))

    def test_explicit_contact_and_handle_keys(self):
        self.assertEqual(
            consent.parse_member_block_ids("contact:12, handle:70"),
            frozenset({"contact:12", "handle:70"}))

    def test_kind_is_case_insensitive(self):
        self.assertEqual(consent.parse_member_block_ids("Contact:12 HANDLE:70"),
                         frozenset({"contact:12", "handle:70"}))

    def test_mixed_and_junk_skipped(self):
        # Unknown kinds, non-numeric ids, and non-positive ids are dropped.
        self.assertEqual(
            consent.parse_member_block_ids("12, handle:70, chat:5, handle:abc, contact:0"),
            frozenset({"contact:12", "handle:70"}))

    def test_returns_frozenset_of_str(self):
        out = consent.parse_member_block_ids("12")
        self.assertIsInstance(out, frozenset)
        self.assertEqual(out, frozenset({"contact:12"}))


class TestIsBlockedMember(unittest.TestCase):
    def test_bare_int_blocks_contact_key(self):
        self.assertTrue(consent.is_blocked_member("contact:12", "12"))
        self.assertFalse(consent.is_blocked_member("handle:12", "12"))

    def test_handle_key_blockable(self):
        # The gap this closes: a person that is only a bare handle can be blocked.
        self.assertTrue(consent.is_blocked_member("handle:70", "handle:70"))
        self.assertFalse(consent.is_blocked_member("handle:70", "12"))

    def test_none_or_empty_person_key_is_false(self):
        self.assertFalse(consent.is_blocked_member(None, "12"))
        self.assertFalse(consent.is_blocked_member("", "12"))

    def test_empty_list_blocks_nothing(self):
        self.assertFalse(consent.is_blocked_member("contact:12", ""))
        self.assertFalse(consent.is_blocked_member("contact:12", None))


class TestConfigFree(unittest.TestCase):
    def test_importing_consent_does_not_pull_in_config(self):
        # F5 is a pure net-new helper: importing it must not drag in config
        # (nor sqlite/Ollama). Assert config stays absent from sys.modules.
        # Popping config leaks across the discover run (a later import gets a
        # FRESH config with defaults) unless we put the original object back --
        # restore it after asserting so no sibling test inherits the hole.
        saved_config = sys.modules.get("config")
        self.addCleanup(
            lambda: sys.modules.__setitem__("config", saved_config)
            if saved_config is not None else None)
        sys.modules.pop("config", None)
        import importlib
        importlib.reload(consent)
        self.assertNotIn("config", sys.modules)


if __name__ == "__main__":
    unittest.main()
