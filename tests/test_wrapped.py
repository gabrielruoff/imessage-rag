"""Hermetic tests for wrapped.py: temp-file DB, mocked query tools (patched
onto the query_tools package with create=True so they work before the six new
tools are integrated), and mocked ollama.chat_once -- no Ollama, no real DB."""

import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import config
import db
import ollama
import query_tools
import schema
import wrapped

# The six new tools plus the existing ones wrapped.py composes; every test
# that drives main() patches all of these so no real query path runs.
_TOOL_NAMES = (
    "get_activity_patterns",
    "get_response_dynamics",
    "get_reaction_stats",
    "get_word_trends",
    "get_milestones",
    "get_group_dynamics",
    "get_word_frequency",
    "search_contacts_by_name",
)

_HEATMAP = [[0] * 24 for _ in range(7)]
_HEATMAP[2][22] = 9  # Wednesday 22:00 hotspot
_HEATMAP[5][1] = 4

_PATTERNS = {
    "heatmap": _HEATMAP,
    "volume_series": [
        {"period": "2025-01", "sent": 5, "received": 7},
        {"period": "2025-02", "sent": 3, "received": 2},
    ],
    "late_night": {"share": 0.1, "n": 3},
    "busiest_days": [{"day": "2025-01-05", "n": 12}],
    "streaks": {"longest_streak_days": 9},
}

_DEFAULT_RESULTS = {
    "get_activity_patterns": _PATTERNS,
    "get_response_dynamics": {
        "reply_latency": {"me_median_s": 60},
        "initiation": {"me": 4, "them": 6},
        "sessions": {"count": 10},
    },
    "get_reaction_stats": {"given": {"laugh": 3}, "hall_of_fame": [{"id": 1, "text": "ha"}]},
    "get_word_trends": {"distinctive_terms": [{"term": "blorbo"}], "emoji": [], "rates": {}},
    "get_milestones": {"firsts": {"first": {"id": 1}}},
    "get_group_dynamics": {"members": [{"name": "A", "share": 0.5}]},
    "get_word_frequency": [{"word": "hello", "count": 4}],
    "search_contacts_by_name": [],
}


def seed_contact(con, cid, first, last=""):
    con.execute(
        "INSERT INTO contacts (id, first_name, last_name) VALUES (?, ?, ?)", (cid, first, last)
    )


def seed_handle(con, hid, identifier, contact_id=None):
    con.execute(
        "INSERT INTO handles (id, handle_identifier, contact_id) VALUES (?, ?, ?)",
        (hid, identifier, contact_id),
    )


def seed_chat(con, chat_id, identifier, is_group=0, display_name=None):
    con.execute(
        "INSERT INTO chats (id, chat_identifier, display_name, is_group) VALUES (?, ?, ?, ?)",
        (chat_id, identifier, display_name, is_group),
    )


_MSG_SEQ = iter(range(1, 10**6))


def seed_msg(con, chat_id, handle_id, date, is_from_me=0, text="hi"):
    mid = next(_MSG_SEQ)
    con.execute(
        "INSERT INTO messages (id, guid, chat_id, handle_id, is_from_me, text, date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, f"guid-{mid}", chat_id, handle_id, is_from_me, text, date),
    )
    return mid


def seed_msgs(con, chat_id, handle_id, dates, **kw):
    for d in dates:
        seed_msg(con, chat_id, handle_id, d, **kw)


class WrappedTestCase(unittest.TestCase):
    """Temp DB + full schema; DB_PATH patched onto wrapped for the duration."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, self.db_path)
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        db.load_vec_extension(self.con)
        schema.create_schema(self.con)
        self.con.row_factory = sqlite3.Row
        patcher = mock.patch.object(wrapped, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch_tools(self, **overrides):
        """Patch every composed tool onto query_tools (create=True keeps this
        working whether or not integration has landed). overrides replace the
        canned return value; a value of None *removes* the attribute for the
        test, so wrapped._tool hits its AttributeError not-integrated branch
        even now that the real tools exist."""
        mocks = {}
        for name in _TOOL_NAMES:
            value = overrides.get(name, _DEFAULT_RESULTS[name])
            if value is None:
                if hasattr(query_tools, name):
                    real = getattr(query_tools, name)
                    delattr(query_tools, name)
                    self.addCleanup(setattr, query_tools, name, real)
                continue
            m = value if isinstance(value, mock.Mock) else mock.Mock(return_value=value)
            patcher = mock.patch.object(query_tools, name, m, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
            mocks[name] = m
        return mocks

    def patch_llm(self, **kwargs):
        m = mock.Mock(**(kwargs or {"return_value": {"content": "A fine narrative."}}))
        patcher = mock.patch.object(ollama, "chat_once", m)
        patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def run_main(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = wrapped.main(argv)
        return code, out.getvalue()

    def out_path(self, name="report.md"):
        return os.path.join(tempfile.gettempdir(), f"wrapped-test-{os.getpid()}-{name}")


class TestBuildRoster(WrappedTestCase):
    def _seed_three(self):
        seed_contact(self.con, 1, "Alice")
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_handle(self.con, 2, "+15550002", 2)
        seed_handle(self.con, 3, "+15550003")  # unsaved number
        seed_chat(self.con, 1, "c1")
        seed_msgs(self.con, 1, 1, [f"2025-03-{d:02d}T12:00:00" for d in range(1, 11)])
        seed_msgs(self.con, 1, 2, [f"2025-03-{d:02d}T12:00:00" for d in range(1, 6)])
        seed_msgs(self.con, 1, 3, [f"2025-03-{d:02d}T12:00:00" for d in range(1, 4)])
        self.con.commit()

    def test_ranking_and_cap(self):
        self._seed_three()
        with mock.patch.object(config, "WRAPPED_TOP_CONTACTS", 2):
            roster = wrapped.build_roster(self.con)
        self.assertEqual([e["name"] for e in roster], ["Alice", "Bob"])
        self.assertEqual(roster[0]["n_messages"], 10)
        self.assertEqual(roster[0]["contact_id"], 1)

    def test_year_scope_and_unsaved_handle(self):
        self._seed_three()
        seed_msgs(self.con, 1, 2, [f"2024-06-{d:02d}T12:00:00" for d in range(1, 21)])
        self.con.commit()
        roster = wrapped.build_roster(self.con, year=2025, limit=10)
        by_name = {e["name"]: e["n_messages"] for e in roster}
        self.assertEqual(by_name["Bob"], 5)  # 2024 traffic not counted
        self.assertEqual(by_name["+15550003"], 3)
        self.assertIsNone([e for e in roster if e["name"] == "+15550003"][0]["contact_id"])
        all_time = wrapped.build_roster(self.con, limit=10)
        self.assertEqual([e["name"] for e in all_time][0], "Bob")  # 25 total

    def test_exclusions(self):
        self._seed_three()
        seed_chat(self.con, 2, "c2")
        seed_msgs(self.con, 2, 2, [f"2025-04-{d:02d}T12:00:00" for d in range(1, 11)])
        self.con.commit()
        roster = wrapped.build_roster(self.con, exclude_chat_ids=[2], limit=10)
        self.assertEqual({e["name"]: e["n_messages"] for e in roster}["Bob"], 5)
        roster = wrapped.build_roster(
            self.con, exclude_ranges=[("2025-03-01", "2025-03-31")], limit=10
        )
        self.assertEqual({e["name"] for e in roster}, {"Bob"})


class TestSilentExclusions(WrappedTestCase):
    def _roster_for(self, *entries):
        return [
            {"contact_id": cid, "handle_id": hid, "name": name, "n_messages": 1}
            for cid, hid, name in entries
        ]

    def test_rule_branches(self):
        for cid, name in ((1, "Alice"), (2, "Bob"), (3, "Carol"), (4, "Dave")):
            seed_contact(self.con, cid, name)
            seed_handle(self.con, cid, f"+1555000{cid}", cid)
        seed_chat(self.con, 1, "c1")
        # Alice: active through corpus end (defines MAX(date) = 2025-12).
        seed_msgs(self.con, 1, 1, [f"2025-{m:02d}-01T12:00:00" for m in range(1, 13)])
        # Bob: 12 lifetime, active 4 of the 6 months before last msg 2025-01 -> excluded.
        seed_msgs(self.con, 1, 2, ["2024-10-01T12:00:00", "2024-10-02T12:00:00",
                                   "2024-11-01T12:00:00", "2024-11-02T12:00:00",
                                   "2024-12-01T12:00:00", "2024-12-02T12:00:00",
                                   "2025-01-01T12:00:00", "2025-01-02T12:00:00",
                                   "2025-01-03T12:00:00", "2025-01-04T12:00:00",
                                   "2025-01-05T12:00:00", "2025-01-06T12:00:00"])
        # Carol: quiet since 2025-01 but only 2 active months in the window -> kept.
        seed_msgs(self.con, 1, 3, [f"2024-06-{d:02d}T12:00:00" for d in range(1, 11)]
                  + ["2025-01-01T12:00:00", "2025-01-02T12:00:00"])
        # Dave: quiet + bursty but under the lifetime minimum -> kept.
        seed_msgs(self.con, 1, 4, ["2024-11-01T12:00:00", "2024-12-01T12:00:00",
                                   "2025-01-01T12:00:00"])
        self.con.commit()
        roster = self._roster_for((1, 1, "Alice"), (2, 2, "Bob"), (3, 3, "Carol"), (4, 4, "Dave"))
        kept, excluded = wrapped.silent_exclusions(
            self.con, roster, min_messages=10, silent_months=6
        )
        self.assertEqual([e["name"] for e in excluded], ["Bob"])
        self.assertEqual(excluded[0]["last_month"], "2025-01")
        self.assertEqual([e["name"] for e in kept], ["Alice", "Carol", "Dave"])


class TestParseExcludeSpec(unittest.TestCase):
    def test_branches(self):
        self.assertEqual(wrapped.parse_exclude_spec("42"), ("chat", 42))
        self.assertEqual(
            wrapped.parse_exclude_spec("2025-01-01..2025-02-01"),
            ("range", ("2025-01-01", "2025-02-01")),
        )
        self.assertEqual(wrapped.parse_exclude_spec("Bob Smith"), ("name", "Bob Smith"))
        # A malformed range degrades to a name, not a crash.
        self.assertEqual(
            wrapped.parse_exclude_spec("2025-01-01..soon"), ("name", "2025-01-01..soon")
        )


class TestHeatmap(unittest.TestCase):
    def test_render(self):
        art = wrapped.render_heatmap_ascii(_HEATMAP)
        lines = art.splitlines()
        self.assertEqual(len(lines), 8)  # header + 7 weekdays
        self.assertIn("@", lines[3])  # Wednesday peak at max density
        self.assertNotIn("@", lines[6])  # Saturday mid-density
        self.assertTrue(lines[3].startswith("Wed"))
        self.assertIsNone(wrapped.render_heatmap_ascii(None))
        self.assertIsNone(wrapped.render_heatmap_ascii({"share": 0.1}))
        self.assertEqual(art, wrapped.render_heatmap_ascii({"grid": _HEATMAP}))


class TestMainFlow(WrappedTestCase):
    def _seed_basic(self):
        seed_contact(self.con, 1, "Alice")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_chat(self.con, 1, "c1")
        seed_chat(self.con, 2, "group1", is_group=1, display_name="The Gang")
        seed_msgs(self.con, 1, 1, [f"2025-05-{d:02d}T12:00:00" for d in range(1, 6)])
        seed_msgs(self.con, 2, 1, ["2025-05-10T12:00:00", "2025-05-11T12:00:00"])
        self.con.commit()

    def test_report_composition(self):
        self._seed_basic()
        self.patch_tools(get_word_trends={"available": False}, get_milestones=None)
        self.patch_llm()
        out = self.out_path("compose.md")
        code, stdout = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        self.assertIn("Featured roster:", stdout)
        self.assertIn(out, stdout)
        with open(out, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("# Wrapped — 2025", md)
        self.assertIn("## Year in Numbers", md)
        self.assertIn('"messages_sent": 8', md)
        self.assertIn("## Your People", md)
        self.assertIn("## Tempo", md)
        self.assertIn("## Reactions", md)
        self.assertIn("## Late Nights, Streaks & the Heatmap", md)
        self.assertIn("Wed", md)  # ASCII heatmap rendered
        self.assertIn("@", md)
        # get_milestones attribute missing entirely -> skipped with a note.
        self.assertIn("## Milestones & Anniversaries", md)
        self.assertIn("get_milestones is not available yet", md)
        # get_word_trends {"available": false} -> word_frequency data still used.
        self.assertIn("## Words & Inside Jokes", md)
        self.assertIn('"word": "hello"', md)
        self.assertIn("## Groups", md)
        self.assertIn("The Gang", md)
        # chunk_annotations/topics empty -> Eras omitted silently.
        self.assertNotIn("Eras & Emotional Arc", md)

    def test_words_section_skipped_when_both_unavailable(self):
        self._seed_basic()
        self.patch_tools(
            get_word_trends={"available": False},
            get_word_frequency=mock.Mock(return_value={"error": "nope"}),
        )
        self.patch_llm()
        out = self.out_path("words.md")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("## Words & Inside Jokes", md)
        self.assertIn("_Skipped:", md)

    def test_eras_section_when_annotations_exist(self):
        self._seed_basic()
        self.con.execute(
            "INSERT INTO chunk_annotations (chunk_id, valence) VALUES (1, 0.5)"
        )
        self.con.commit()
        trends = mock.Mock(return_value={"emotions": {"joy": 3}, "valence": {"mean": 0.2}})
        self.patch_tools(get_word_trends=trends)
        self.patch_llm()
        out = self.out_path("eras.md")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("## Eras & Emotional Arc", md)
        self.assertIn('"joy": 3', md)
        # The arc call asks for the emotions/valence sections explicitly.
        includes = [c.kwargs.get("include") for c in trends.call_args_list]
        self.assertIn(["emotions", "valence"], includes)

    def test_silent_exclusion_preview_and_include_override(self):
        seed_contact(self.con, 1, "Alice")
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_handle(self.con, 2, "+15550002", 2)
        seed_chat(self.con, 1, "c1")
        seed_msgs(self.con, 1, 1, [f"2025-{m:02d}-01T12:00:00" for m in range(1, 13)])
        seed_msgs(self.con, 1, 2, ["2024-10-01T12:00:00", "2024-11-01T12:00:00",
                                   "2024-12-01T12:00:00", "2025-01-01T12:00:00",
                                   "2025-01-02T12:00:00"])
        self.con.commit()
        self.patch_tools()
        self.patch_llm()
        with mock.patch.object(config, "WRAPPED_SILENT_MIN_MESSAGES", 5), \
                mock.patch.object(config, "WRAPPED_SILENT_MONTHS", 6):
            out = self.out_path("silent.md")
            code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
            self.assertEqual(code, 0)
            self.assertIn("excluded (went quiet 2025-01): Bob", stdout)
            self.assertIn('pass --include "Bob" to feature them', stdout)
            with open(out, encoding="utf-8") as f:
                self.assertNotIn('"name": "Bob"', f.read())

            out2 = self.out_path("silent-include.md")
            code, stdout = self.run_main(
                ["--yes", "--no-llm", "--include", "Bob", "--out", out2]
            )
            self.assertEqual(code, 0)
            self.assertNotIn("excluded (went quiet", stdout)
            self.assertIn("2. Bob", stdout)

    def test_prompt_abort_and_yes_skips_prompt(self):
        self._seed_basic()
        self.patch_tools()
        self.patch_llm()
        out = self.out_path("abort.md")
        with mock.patch("builtins.input", return_value="n") as fake_input:
            code, stdout = self.run_main(["--no-llm", "--out", out])
        self.assertEqual(code, 1)
        self.assertIn("Aborted.", stdout)
        fake_input.assert_called_once()
        self.assertFalse(os.path.exists(out))
        with mock.patch("builtins.input") as fake_input:
            code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        fake_input.assert_not_called()
        os.unlink(out)

    def test_no_llm_makes_zero_chat_once_calls(self):
        self._seed_basic()
        self.patch_tools()
        llm = self.patch_llm()
        out = self.out_path("nollm.md")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        llm.assert_not_called()
        os.unlink(out)

    def test_narratives_inserted_and_failure_falls_back(self):
        self._seed_basic()
        self.patch_tools()
        llm = self.patch_llm()
        out = self.out_path("llm.md")
        code, _ = self.run_main(["--yes", "--out", out])
        self.assertEqual(code, 0)
        self.assertGreater(llm.call_count, 0)
        self.assertEqual(llm.call_args.kwargs.get("model"), config.WRAPPED_MODEL)
        with open(out, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("A fine narrative.", md)
        os.unlink(out)

        self.patch_llm(side_effect=RuntimeError("ollama down"))
        out2 = self.out_path("llm-fail.md")
        code, _ = self.run_main(["--yes", "--out", out2])
        self.assertEqual(code, 0)  # stats-only fallback, never a crash
        with open(out2, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("_Narrative unavailable", md)
        self.assertIn('"messages_sent"', md)
        os.unlink(out2)

    def test_contact_mode_scopes_calls_and_ambiguity_exits(self):
        self._seed_basic()
        alice = {"id": 1, "first_name": "Alice", "last_name": "", "nickname": None,
                 "handles": [{"id": 1}]}
        mocks = self.patch_tools(
            search_contacts_by_name=mock.Mock(return_value=[alice])
        )
        self.patch_llm()
        out = self.out_path("contact.md")
        code, stdout = self.run_main(
            ["--contact", "Alice", "--year", "2025", "--yes", "--no-llm", "--out", out]
        )
        self.assertEqual(code, 0)
        self.assertIn("Featured: Alice", stdout)
        dyn = mocks["get_response_dynamics"]
        self.assertEqual(dyn.call_args.kwargs.get("contact_id"), 1)
        self.assertEqual(dyn.call_args.kwargs.get("date_from"), "2025-01-01")
        with open(out, encoding="utf-8") as f:
            md = f.read()
        self.assertNotIn("## Your People", md)  # deep-dive drops corpus sections
        self.assertNotIn("## Groups", md)
        os.unlink(out)

        self.patch_tools(search_contacts_by_name=mock.Mock(return_value=[alice, dict(alice)]))
        code, stdout = self.run_main(["--contact", "Alice", "--yes", "--no-llm"])
        self.assertEqual(code, 1)
        self.assertIn("ambiguous contact", stdout)

    def test_exclude_specs_flow_through_main(self):
        self._seed_basic()
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 2, "+15550002", 2)
        seed_msgs(self.con, 1, 2, [f"2025-06-{d:02d}T12:00:00" for d in range(1, 4)])
        self.con.commit()
        bob = {"id": 2, "first_name": "Bob", "last_name": "", "nickname": None, "handles": []}
        self.patch_tools(search_contacts_by_name=mock.Mock(return_value=[bob]))
        self.patch_llm()
        out = self.out_path("exclude.md")
        code, stdout = self.run_main(
            ["--yes", "--no-llm", "--exclude", "Bob", "--out", out]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("Bob", stdout.split("excluded")[0])
        os.unlink(out)


if __name__ == "__main__":
    unittest.main()
