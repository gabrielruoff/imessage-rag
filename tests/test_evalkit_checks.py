"""Hermetic unit tests for evalkit's deterministic layers: case validation +
placeholder resolution (cases.py), transcript checking (checks.py), and
scorecard aggregation (report.py). No DB, no Ollama, no app imports -- the
fixture transcript (eval/cases/fixtures/canned_transcript.json) uses invented
names and stands in for real runner output.

Run from the repo root:  python3 -m unittest tests.test_evalkit_checks
"""

import copy
import json
import unittest

from eval.evalkit import CASES_DIR, cases, checks, report

with open(CASES_DIR / "fixtures" / "canned_transcript.json") as f:
    FIXTURE = json.load(f)

ROSTER = {
    "ALICE": {
        "display_name": "Alice Quimby",
        "contact_id": 12,
        "handle_ids": [3, 4],
        "chat_ids": [7, 8],
    },
    "BOB": {"display_name": "Bob Tester", "contact_id": 20, "handle_ids": [9], "chat_ids": [11]},
    "CLUB": {"display_name": "Fixture Club", "chat_ids": [13, 14], "n_participants": 6},
    "YEAR": 2025,
    "WORD": "banana",
}
ROSTER_DEFS = {name: {"kind": "test", "definition": "test"} for name in ROSTER}

FIXTURE_CASE = {
    "id": "FX.1",
    "prompt": "How many texts did <ALICE> and I exchange each month of <YEAR>?",
    "expect": {
        "tools": [
            {
                "name": "get_activity_patterns",
                "args_contains": {"_scope": "<ALICE>", "granularity": "month",
                                  "_date_year": "<YEAR>"},
            }
        ]
    },
    "gt": {"recipe": "volume_series", "params": {"scope": "<ALICE>", "year": "<YEAR>"}},
    "probe": None,
    "grader_notes": "fixture",
    "quality_watch": None,
}

MANIFEST = {
    "corpus_fingerprint": FIXTURE["corpus_fingerprint"],
    "roster": ROSTER,
    "gt": {
        "FX.1": {
            "status": "ok",
            "values": {
                "2025-03_total": {"value": 431, "tol": "exact"},
                "share": {"value": 0.61, "tol": "pct1"},
                "total": {"value": 1234, "tol": "exact"},
                "first_date": {"value": "2019-03-02", "tol": "exact"},
            },
        }
    },
}


class ArgsMatchingTest(unittest.TestCase):
    def test_scope_matches_on_any_id_kind(self):
        entry = ROSTER["ALICE"]
        self.assertEqual(checks.args_diffs({"contact_id": 12}, {"_scope": "<ALICE>"}, ROSTER), [])
        self.assertEqual(
            checks.args_diffs({"handle_id": [4, 99]}, {"_scope": "<ALICE>"}, ROSTER), []
        )
        self.assertEqual(checks.args_diffs({"chat_id": 8}, {"_scope": "<ALICE>"}, ROSTER), [])
        diffs = checks.args_diffs({"chat_id": [11]}, {"_scope": "<ALICE>"}, ROSTER)
        self.assertEqual(len(diffs), 1)
        self.assertTrue(checks._scope_matches({"contact_id": [12]}, entry))
        self.assertFalse(checks._scope_matches({}, entry))

    def test_date_year_allows_one_day_slack(self):
        args = {"date_from": "2024-12-31", "date_to": "2026-01-01"}
        self.assertEqual(checks.args_diffs(args, {"_date_year": "<YEAR>"}, ROSTER), [])
        args = {"date_from": "2024-12-30", "date_to": "2025-12-31"}
        self.assertNotEqual(checks.args_diffs(args, {"_date_year": "<YEAR>"}, ROSTER), [])
        # both bounds are required
        self.assertNotEqual(
            checks.args_diffs({"date_from": "2025-01-01"}, {"_date_year": "<YEAR>"}, ROSTER), []
        )

    def test_literal_is_case_insensitive_and_list_aware(self):
        self.assertEqual(checks.args_diffs({"granularity": "Month"},
                                           {"granularity": "month"}, ROSTER), [])
        self.assertEqual(checks.args_diffs({"reaction_type": ["love", "like"]},
                                           {"reaction_type": "love"}, ROSTER), [])
        self.assertNotEqual(checks.args_diffs({"granularity": "year"},
                                              {"granularity": "month"}, ROSTER), [])

    def test_any_chat_id_intersects_roster_chat_ids(self):
        self.assertEqual(
            checks.args_diffs({"chat_id": [14]}, {"chat_id": "<CLUB>.any_chat_id"}, ROSTER), []
        )
        self.assertNotEqual(
            checks.args_diffs({"chat_id": [7]}, {"chat_id": "<CLUB>.any_chat_id"}, ROSTER), []
        )


class MatchExpectTest(unittest.TestCase):
    CALLS = [
        {"name": "get_chat_summary_stats", "arguments": {"contact_id": 12}, "turn": 1,
         "result": None, "msg_index": 2},
        {"name": "get_response_dynamics", "arguments": {"contact_id": 20}, "turn": 2,
         "result": None, "msg_index": 5},
    ]

    def test_alternate_name_matches(self):
        expect = {"name": "get_activity_patterns", "alternates": ["get_chat_summary_stats"],
                  "args_contains": {"_scope": "<ALICE>"}}
        match = checks.match_expect(expect, self.CALLS, ROSTER)
        self.assertTrue(match["args_ok"])
        self.assertEqual(match["matched_call"]["name"], "get_chat_summary_stats")

    def test_turn_scoping(self):
        expect = {"name": "get_response_dynamics", "args_contains": {"_scope": "<BOB>"},
                  "turn": 1}
        self.assertFalse(checks.match_expect(expect, self.CALLS, ROSTER)["args_ok"])
        expect["turn"] = 2
        self.assertTrue(checks.match_expect(expect, self.CALLS, ROSTER)["args_ok"])

    def test_no_call_reports_missing(self):
        match = checks.match_expect({"name": "get_milestones"}, self.CALLS, ROSTER)
        self.assertIsNone(match["matched_call"])
        self.assertIn("no call to get_milestones", match["arg_diffs"][0])


class CheckCaseFixtureTest(unittest.TestCase):
    def test_happy_path_routing_and_numeric(self):
        result = checks.check_case(FIXTURE_CASE, FIXTURE, MANIFEST)
        self.assertTrue(result["fingerprint_ok"])
        self.assertEqual(result["routing"]["auto"], "pass")
        values = result["numeric"]["values"]
        # 431 = 200 + 231 appears in the reply but in no tool result
        self.assertTrue(values["2025-03_total"]["found_in_reply"])
        self.assertFalse(values["2025-03_total"]["found_in_tool_results"])
        # share stored as 0.61, reply says 61% (pct1 tries x100)
        self.assertTrue(values["share"]["found_in_reply"])
        # 1,234 in the reply vs 1234 in the tool result
        self.assertTrue(values["total"]["found_in_reply"])
        self.assertIn("first_date", result["numeric"]["missing_from_reply"])
        confab = result["unsupported_numbers"]["numbers_in_reply_not_in_tool_results"]
        self.assertIn("431", confab)   # computed, not sourced -- graders decide
        self.assertIn("99.9", confab)  # invented outright
        self.assertNotIn("1,234", confab)
        self.assertNotIn("61%", confab)

    def test_banned_tool_fails_routing(self):
        case = copy.deepcopy(FIXTURE_CASE)
        case["expect"]["banned_tools"] = ["get_activity_patterns"]
        result = checks.check_case(case, FIXTURE, MANIFEST)
        self.assertEqual(result["routing"]["auto"], "fail")
        self.assertEqual(result["routing"]["banned_called"], ["get_activity_patterns"])

    def test_banned_anywhere_scans_the_whole_transcript(self):
        case = copy.deepcopy(FIXTURE_CASE)
        case["expect"]["banned_anywhere"] = ["get_activity_histogram"]
        transcript = copy.deepcopy(FIXTURE)
        transcript["messages"][6]["content"] += " (via get_activity_histogram)"
        result = checks.check_case(case, transcript, MANIFEST)
        self.assertEqual(result["routing"]["auto"], "fail")
        self.assertEqual(result["routing"]["banned_anywhere_hits"], ["get_activity_histogram"])

    def test_removed_tools_scanned_without_case_config(self):
        # REMOVED_TOOLS are banned in EVERY transcript, not just cases that
        # declare banned_anywhere.
        transcript = copy.deepcopy(FIXTURE)
        transcript["messages"][2]["tool_calls"][0]["function"]["name"] = (
            "get_conversation_dynamics"
        )
        result = checks.check_case(FIXTURE_CASE, transcript, MANIFEST)
        self.assertEqual(result["routing"]["auto"], "fail")
        self.assertEqual(
            result["routing"]["banned_anywhere_hits"], ["get_conversation_dynamics"]
        )

    def test_wrong_scope_fails_args(self):
        transcript = copy.deepcopy(FIXTURE)
        transcript["messages"][4]["tool_calls"][0]["function"]["arguments"]["contact_id"] = 99
        result = checks.check_case(FIXTURE_CASE, transcript, MANIFEST)
        self.assertEqual(result["routing"]["auto"], "fail")
        expected = result["routing"]["expected"][0]
        self.assertFalse(expected["args_ok"])
        self.assertTrue(any("_scope" in d for d in expected["arg_diffs"]))


def _probe_transcript(messages):
    transcript = copy.deepcopy(FIXTURE)
    transcript["messages"] = messages
    transcript["turn_boundaries"] = [
        {"turn": 1, "user_index": 1, "final_reply_index": len(messages) - 1}
    ]
    transcript["final_reply"] = messages[-1].get("content", "")
    return transcript


def _call(name, arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class ProbeTest(unittest.TestCase):
    def test_scope_requirement_fails_on_verbatim_error_relay(self):
        error = "get_response_dynamics requires a chat or person scope"
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "habits?"},
            _call("get_response_dynamics", {}),
            {"role": "tool", "content": json.dumps({"error": error})},
            {"role": "assistant", "content": f"Sorry: {error}."},
        ]
        case = {"id": "P.1", "prompt": "x", "probe": "scope_requirement",
                "expect": {}, "gt": None, "grader_notes": "n"}
        result = checks.check_case(case, _probe_transcript(messages), MANIFEST)
        self.assertEqual(result["probe"]["auto"], "fail")

    def test_scope_requirement_recovery_is_manual_and_no_error_is_pass(self):
        error = "scope required"
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "habits?"},
            _call("get_response_dynamics", {}),
            {"role": "tool", "content": json.dumps({"error": error})},
            {"role": "assistant", "content": "I looked at your busiest chat instead."},
        ]
        case = {"id": "P.2", "prompt": "x", "probe": "scope_requirement",
                "expect": {}, "gt": None, "grader_notes": "n"}
        result = checks.check_case(case, _probe_transcript(messages), MANIFEST)
        self.assertEqual(result["probe"]["auto"], "manual")
        result = checks.check_case(case, FIXTURE, MANIFEST)
        self.assertEqual(result["probe"]["auto"], "pass")

    def test_negative_control(self):
        base = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hour for nobody?"},
            _call("search_contacts_by_name", {"name": "Thad"}),
            {"role": "tool", "content": "[]"},
        ]
        case = {"id": "P.3", "prompt": "x", "probe": "negative_control",
                "expect": {}, "gt": None, "grader_notes": "n"}
        honest = base + [{"role": "assistant", "content": "I can't find them."}]
        result = checks.check_case(case, _probe_transcript(honest), MANIFEST)
        self.assertEqual(result["probe"]["auto"], "pass")
        invented = base + [
            _call("get_activity_patterns", {"contact_id": 12}),
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "They text at 9pm."},
        ]
        result = checks.check_case(case, _probe_transcript(invented), MANIFEST)
        self.assertEqual(result["probe"]["auto"], "fail")

    def test_gated_section_is_always_manual_with_detail(self):
        case = {"id": "P.4", "prompt": "x", "probe": "gated_section",
                "expect": {}, "gt": None, "grader_notes": "n"}
        manifest = copy.deepcopy(MANIFEST)
        manifest["gt"]["P.4"] = {"status": "ok",
                                 "values": {"gated": {"value": True, "tol": "exact"}}}
        result = checks.check_case(case, FIXTURE, manifest)
        self.assertEqual(result["probe"]["auto"], "manual")
        self.assertIn("gated=True", result["probe"]["detail"])


class NumberMatchingTest(unittest.TestCase):
    def test_extraction_normalizes_commas_and_skips_iso_date_fragments(self):
        numbers = checks.extract_numbers("sent 1,234 texts (12.5%) on 2026-07-14")
        self.assertIn((1234.0, False), numbers)
        self.assertIn((12.5, True), numbers)
        self.assertNotIn((-7.0, False), numbers)

    def test_tolerances(self):
        self.assertTrue(checks.value_found(431, "exact", "431 messages"))
        self.assertFalse(checks.value_found(431, "exact", "432 messages"))
        self.assertTrue(checks.value_found(72.41, "round", "about 72.4 seconds"))
        self.assertTrue(checks.value_found(72.41, "round", "about 72 seconds"))
        self.assertTrue(checks.value_found(0.174, "pct1", "17% of sessions"))
        self.assertFalse(checks.value_found(0.174, "pct1", "19% of sessions"))

    def test_dates_match_iso_and_long_form(self):
        self.assertTrue(checks.value_found("2026-07-14T03:00:00", "exact",
                                           "first texted on 2026-07-14"))
        self.assertTrue(checks.value_found("2026-07-14", "exact",
                                           "first texted on July 14, 2026"))
        self.assertFalse(checks.value_found("2026-07-14", "exact",
                                            "first texted on July 15, 2026"))

    def test_strings_match_case_insensitively(self):
        self.assertTrue(checks.value_found("Alice Quimby", "exact", "that's alice quimby"))
        self.assertFalse(checks.value_found("Alice Quimby", "exact", "that's Bob"))


class CaseValidationTest(unittest.TestCase):
    def _valid(self):
        return copy.deepcopy(FIXTURE_CASE)

    def test_duplicate_ids_rejected(self):
        suite = {"suite": "FX", "title": "t",
                 "targets": {a: 1 for a in cases.AXES},
                 "cases": [self._valid(), self._valid()]}
        with self.assertRaisesRegex(ValueError, "duplicate case id"):
            cases.validate_suite(suite, ROSTER_DEFS, set())

    def test_prompt_and_turns_are_exclusive(self):
        case = self._valid()
        case["turns"] = ["a", "b"]
        with self.assertRaisesRegex(ValueError, "prompt/turns"):
            cases.validate_case(case, ROSTER_DEFS)

    def test_unknown_probe_rejected(self):
        case = self._valid()
        case["probe"] = "vibes"
        with self.assertRaisesRegex(ValueError, "unknown probe"):
            cases.validate_case(case, ROSTER_DEFS)

    def test_unknown_placeholder_rejected(self):
        case = self._valid()
        case["prompt"] = "texts with <ZELDA>?"
        with self.assertRaisesRegex(ValueError, "ZELDA"):
            cases.validate_case(case, ROSTER_DEFS)

    def test_unknown_tool_rejected(self):
        case = self._valid()
        case["expect"]["tools"][0]["name"] = "get_activity_histogram"
        with self.assertRaisesRegex(ValueError, "not a known tool"):
            cases.validate_case(case, ROSTER_DEFS)


class ResolveRefTest(unittest.TestCase):
    def test_forms(self):
        self.assertIs(cases.resolve_ref("<ALICE>", ROSTER), ROSTER["ALICE"])
        self.assertEqual(cases.resolve_ref("<ALICE>.chat_ids", ROSTER), [7, 8])
        self.assertEqual(cases.resolve_ref("<YEAR>", ROSTER), 2025)
        self.assertEqual(cases.resolve_ref("in <YEAR> only", ROSTER), "in 2025 only")
        self.assertEqual(cases.resolve_ref("plain", ROSTER), "plain")
        self.assertEqual(cases.resolve_ref(42, ROSTER), 42)
        with self.assertRaises(KeyError):
            cases.resolve_ref("<ZELDA>", ROSTER)

    def test_substitute_text_uses_display_names(self):
        text = cases.substitute_text("Did <ALICE> text <BOB> in <YEAR>?", ROSTER)
        self.assertEqual(text, "Did Alice Quimby text Bob Tester in 2025?")


def _grade(cid, routing=1, accuracy=1, faithfulness=1, communication=1,
           levers=(), blocked=False):
    return {
        "id": cid, "run_id": "r1", "suite": cid.split(".")[0], "grader": "agent:test",
        "axes": {"routing": routing, "accuracy": accuracy,
                 "faithfulness": faithfulness, "communication": communication},
        "evidence": [], "levers": list(levers), "blocked": blocked,
        "blocked_reason": None, "notes": "",
    }


class ReportTest(unittest.TestCase):
    SUITES = [
        {"suite": "S1", "title": "t",
         "targets": {"routing": 0.9, "accuracy": 0.85, "faithfulness": 1.0,
                     "communication": 0.85},
         "cases": []},
    ]
    RUN_DOC = {"run_id": "r2", "model": "m", "config": {}, "flags": [],
               "corpus_fingerprint": {"message_count": 1, "max_message_date": "x"}}

    def test_aggregation_and_compare_delta(self):
        prev_grades = [
            _grade("S1.1"),
            _grade("S1.2", accuracy=0, levers=["shape"]),
        ]
        prev = report.build_scorecard(dict(self.RUN_DOC, run_id="r1"), self.SUITES, prev_grades)
        self.assertEqual(prev["suites"]["S1"]["axes"]["accuracy"]["mean"], 0.5)
        self.assertFalse(prev["overall_pass"])
        self.assertEqual(prev["lever_histogram"], {"shape": 1})

        cur_grades = [
            _grade("S1.1"),
            _grade("S1.2"),
            _grade("S1.3", blocked=True),  # excluded from means
        ]
        cur = report.build_scorecard(
            self.RUN_DOC, self.SUITES, cur_grades, prev=prev,
            transcript_ids=["S1.1", "S1.2", "S1.3", "S1.4"],
        )
        block = cur["suites"]["S1"]
        self.assertEqual(block["graded"], 2)
        self.assertEqual(block["blocked"], 1)
        self.assertEqual(block["axes"]["accuracy"]["mean"], 1.0)
        self.assertEqual(block["axes"]["accuracy"]["delta_vs_prev"], 0.5)
        self.assertFalse(cur["overall_pass"])  # S1.4 is ungraded
        self.assertEqual(cur["compared_to"], "r1")
        self.assertEqual(cur["newly_passing"], ["S1.2"])
        self.assertEqual(cur["newly_failing"], [])
        self.assertEqual(cur["ungraded"], ["S1.4"])
        self.assertEqual(cur["blocked_cases"], ["S1.3"])
        markdown = report.render_markdown(cur, cur_grades)
        self.assertIn("| S1 | 2/3 | 1 |", markdown)

        fully_graded = report.build_scorecard(
            self.RUN_DOC, self.SUITES, cur_grades, prev=prev,
            transcript_ids=["S1.1", "S1.2", "S1.3"],
        )
        self.assertTrue(fully_graded["overall_pass"])

    def test_run_flags_block_overall_pass(self):
        run_doc = dict(self.RUN_DOC, flags=["banned_anywhere:get_activity_histogram@S7.3"])
        card = report.build_scorecard(run_doc, self.SUITES, [_grade("S1.1")])
        self.assertFalse(card["overall_pass"])

    def test_ungraded_run_never_passes(self):
        # Zero grades (grading skipped/crashed) must not read as a pass, even
        # with no target missed and no flags.
        card = report.build_scorecard(self.RUN_DOC, self.SUITES, [])
        self.assertFalse(card["overall_pass"])
        card = report.build_scorecard(
            self.RUN_DOC, self.SUITES, [], transcript_ids=["S1.1"]
        )
        self.assertFalse(card["overall_pass"])
        self.assertEqual(card["ungraded"], ["S1.1"])

    def test_malformed_grades_block_overall_pass(self):
        card = report.build_scorecard(
            self.RUN_DOC, self.SUITES, [_grade("S1.1")],
            transcript_ids=["S1.1"], malformed_grades=["S1.2.grade.json"],
        )
        self.assertFalse(card["overall_pass"])
        self.assertEqual(card["malformed_grades"], ["S1.2.grade.json"])
        self.assertIn("S1.2.grade.json", report.render_markdown(card, [_grade("S1.1")]))

    def test_grade_error(self):
        self.assertIsNone(report.grade_error(_grade("S1.1")))
        self.assertIsNotNone(report.grade_error("not a dict"))
        self.assertIsNotNone(report.grade_error({"axes": {}}))  # no id
        bad_axes = _grade("S1.1")
        bad_axes["axes"]["accuracy"] = 0.5  # axes are 0/1, not fractional
        self.assertIsNotNone(report.grade_error(bad_axes))
        missing_axis = _grade("S1.1")
        del missing_axis["axes"]["faithfulness"]
        self.assertIsNotNone(report.grade_error(missing_axis))


if __name__ == "__main__":
    unittest.main()
