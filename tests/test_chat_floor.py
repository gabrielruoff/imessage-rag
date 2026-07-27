"""Unit tests for chat_floor.TurnFloor -- the single-sourced wellbeing floor.

Hermetic: chat_floor imports only wellbeing (stdlib-only), so nothing here opens
a socket or a database.

Run from the repo root:  python3 -m unittest tests.test_chat_floor
"""

import unittest

import chat_floor
import wellbeing


class TurnFloorEpilogueTest(unittest.TestCase):
    def test_note_tool_collects_and_strips_epilogue_and_support_block(self):
        floor = chat_floor.TurnFloor()
        result = floor.note_tool(
            "find_conflicts",
            {"rows": [1, 2], "_epilogue": "call a hotline", "support_block": "dup crisis"},
        )
        # The model must never see the code-delivery directives.
        self.assertNotIn("_epilogue", result)
        self.assertNotIn("support_block", result)
        self.assertEqual(result["rows"], [1, 2])
        self.assertEqual(floor.epilogues, ["call a hotline"])

    def test_note_tool_passes_through_result_without_epilogue(self):
        floor = chat_floor.TurnFloor()
        payload = {"rows": [1]}
        self.assertIs(floor.note_tool("search_messages_sql", payload), payload)
        self.assertEqual(floor.epilogues, [])

    def test_blank_epilogue_is_ignored(self):
        floor = chat_floor.TurnFloor()
        floor.note_tool("find_conflicts", {"rows": [], "_epilogue": "   "})
        self.assertEqual(floor.epilogues, [])

    def test_epilogue_block_dedupes_preserving_order(self):
        floor = chat_floor.TurnFloor()
        for _ in range(3):
            floor.note_tool("diagnose_conflict", {"x": 1, "_epilogue": "same footer"})
        floor.note_tool("draft_apology", {"x": 1, "_epilogue": "second footer"})
        self.assertEqual(floor.epilogue_block(), "same footer\n\nsecond footer")

    def test_epilogue_block_is_none_when_empty(self):
        self.assertIsNone(chat_floor.TurnFloor().epilogue_block())

    def test_fold_epilogues_appends_after_blank_line(self):
        floor = chat_floor.TurnFloor()
        floor.note_tool("diagnose_conflict", {"x": 1, "_epilogue": "footer"})
        self.assertEqual(floor.fold_epilogues("the reply"), "the reply\n\nfooter")

    def test_fold_epilogues_returns_block_alone_when_reply_empty(self):
        floor = chat_floor.TurnFloor()
        floor.note_tool("diagnose_conflict", {"x": 1, "_epilogue": "footer"})
        self.assertEqual(floor.fold_epilogues(""), "footer")

    def test_fold_epilogues_is_noop_without_epilogues(self):
        # A non-wellbeing turn must be byte-for-byte unchanged.
        self.assertEqual(chat_floor.TurnFloor().fold_epilogues("hi"), "hi")


class TurnFloorSanitizeArmingTest(unittest.TestCase):
    def test_trajectory_tool_arms_sanitize(self):
        floor = chat_floor.TurnFloor()
        self.assertFalse(floor.sanitize)
        floor.note_tool("get_language_trajectory", {"rows": []})
        self.assertTrue(floor.sanitize)

    def test_arms_even_on_duplicate_or_error_result(self):
        # The old inline logic armed by NAME regardless of result shape.
        floor = chat_floor.TurnFloor()
        floor.note_tool("get_language_trajectory", {"duplicate_call": True})
        self.assertTrue(floor.sanitize)

    def test_other_tools_do_not_arm(self):
        floor = chat_floor.TurnFloor()
        floor.note_tool("search_messages_sql", {"rows": []})
        self.assertFalse(floor.sanitize)


class SanitizeReplyTest(unittest.TestCase):
    def test_sanitize_reply_matches_wellbeing(self):
        text = "You seem fine. This looks like depression to me."
        clean, redacted = chat_floor.sanitize_reply(text)
        self.assertTrue(redacted)
        self.assertNotIn("depression", clean)
        self.assertEqual((clean, redacted), wellbeing.sanitize_condition_names(text))


if __name__ == "__main__":
    unittest.main()
