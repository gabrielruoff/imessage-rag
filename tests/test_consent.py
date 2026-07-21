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


class TestConfigFree(unittest.TestCase):
    def test_importing_consent_does_not_pull_in_config(self):
        # F5 is a pure net-new helper: importing it must not drag in config
        # (nor sqlite/Ollama). Assert config stays absent from sys.modules.
        sys.modules.pop("config", None)
        import importlib
        importlib.reload(consent)
        self.assertNotIn("config", sys.modules)


if __name__ == "__main__":
    unittest.main()
