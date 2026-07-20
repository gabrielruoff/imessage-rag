"""Hermetic tests for wrapped.py: temp-file DB, mocked query tools (patched
onto the query_tools package with create=True so they work before the six new
tools are integrated), and mocked ollama.chat_once -- no Ollama, no real DB."""

import io
import os
import re
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
import wrapped_html

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
    "get_word_frequency": [{"word": "zamboni", "count": 4}],
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

    def out_path(self, name="report.html"):
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
        art = wrapped_html.render_heatmap(_HEATMAP)
        self.assertEqual(art.count('class="hm-row"'), 7)  # one per weekday
        cells = re.findall(r'class="hm-cell q(\d)"', art)
        self.assertEqual(len(cells), 7 * 24 + 8)  # grid + the legend's scale
        self.assertIn("Wed", art)
        # Exact counts survive in the hover tooltip; the busiest cell takes the
        # top ramp step, an empty one the surface step.
        self.assertIn('title="Wed 10 PM · 9 messages"', art)
        self.assertIn('<span class="hm-cell q7" title="Wed 10 PM · 9 messages">', art)
        self.assertIn('<span class="hm-cell q0" title="Mon 12 AM · 0 messages">', art)
        self.assertIsNone(wrapped_html.render_heatmap(None))
        self.assertIsNone(wrapped_html.render_heatmap({"share": 0.1}))
        self.assertEqual(art, wrapped_html.render_heatmap({"grid": _HEATMAP}))

    def test_quantile_shading_spreads_a_skewed_grid(self):
        # One outlier must not flatten everything else onto one mid step: with
        # a linear ramp 1..8 all land on the same band.
        grid = [[0] * 24 for _ in range(7)]
        for hour in range(24):
            grid[0][hour] = hour + 1
        grid[1][0] = 5000  # outlier sets the peak
        art = wrapped_html.render_heatmap(grid)
        steps = {int(m) for m in re.findall(r"hm-cell q(\d)", art)}
        self.assertTrue({1, 7} <= steps, f"ramp under-used: {steps}")


class TestRenderValue(unittest.TestCase):
    def test_shapes_render_without_json(self):
        html = wrapped_html.render_value(
            {"messages_sent": 1234, "share": 0.25, "note_text": None,
             "people": [{"name": "Alice", "guid": "x", "messages": 3}]}
        )
        self.assertIn("1,234", html)  # thousands separator
        self.assertIn("25.0%", html)  # share keys render as percentages
        self.assertIn("—", html)  # None -> em dash
        self.assertIn("Messages sent", html)  # humanized tile label
        self.assertIn("<td>Alice</td>", html)
        self.assertNotIn("guid", html)  # db plumbing dropped
        self.assertNotIn("{", html)

    def test_unavailable_and_empty(self):
        self.assertIn(
            "no lexicon",
            wrapped_html.render_value({"available": False, "note": "no lexicon"}),
        )
        self.assertIn("None.", wrapped_html.render_value({"rows": []}))

    def test_uniform_dict_becomes_one_table(self):
        html = wrapped_html.render_value(
            {"me": {"median_s": 1}, "them": {"median_s": 2}}, depth=1
        )
        self.assertEqual(html.count("<table"), 1)
        self.assertIn('<th scope="row">Me</th>', html)
        # A ragged pair keeps the generic nested-block path.
        ragged = wrapped_html.render_value(
            {"me": {"median_s": 1}, "them": {"other": 2}}, depth=1
        )
        self.assertIn("block-title", ragged)

    def test_escaping(self):
        html = wrapped_html.render_value({"text": "<script>&x"})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;&amp;x", html)


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
        self.patch_tools(get_word_trends={"available": False})
        self.patch_llm()
        out = self.out_path("compose.html")
        code, stdout = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        self.assertIn("Featured roster:", stdout)
        self.assertIn(out, stdout)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<title>Wrapped — 2025</title>", page)
        self.assertIn("<h1>Wrapped — 2025</h1>", page)
        self.assertIn("<h2>Year in Numbers</h2>", page)
        self.assertIn(">8</div>", page)  # messages_sent tile
        self.assertIn("Messages sent", page)
        self.assertIn("<h2>Your People</h2>", page)
        self.assertIn('class="hbars"', page)  # people render as a bar list
        self.assertIn("<h2>Tempo</h2>", page)
        self.assertIn("<h2>Reactions</h2>", page)
        self.assertIn('class="bubbles"', page)  # hall of fame as chat bubbles
        self.assertIn("<h2>Late Nights, Streaks &amp; the Heatmap</h2>", page)
        self.assertIn('class="chart heatmap"', page)  # heatmap rendered
        self.assertIn('title="Wed 10 PM · 9 messages"', page)
        # Milestones was removed from the report by design.
        self.assertNotIn("Milestones", page)
        # get_word_trends {"available": false} -> word_frequency data still
        # feeds the word cloud.
        self.assertIn("<h2>Words &amp; Inside Jokes</h2>", page)
        self.assertIn('class="cloud"', page)
        self.assertIn("zamboni", page)
        self.assertIn("<h2>Groups</h2>", page)
        self.assertIn("The Gang", page)
        # chunk_annotations/topics empty -> Eras omitted silently.
        self.assertNotIn("Eras &amp; Emotional Arc", page)
        # The whole point of the HTML report: no JSON reaches the page.
        self.assertNotIn("```", page)
        self.assertNotIn('"messages_sent"', page)
        # Print restore: the beforeprint reveal hook and CSS-var bar heights
        # (an inline height: could never be restored over the hidden state).
        self.assertIn("beforeprint", page)
        self.assertIn("--h:", page)
        self.assertNotIn('style="height:', page)

    def test_words_section_skipped_when_both_unavailable(self):
        self._seed_basic()
        self.patch_tools(
            get_word_trends={"available": False},
            get_word_frequency=mock.Mock(return_value={"error": "nope"}),
        )
        self.patch_llm()
        out = self.out_path("words.html")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("<h2>Words &amp; Inside Jokes</h2>", page)
        self.assertIn("Skipped:", page)

    def test_eras_section_when_annotations_exist(self):
        self._seed_basic()
        self.con.execute(
            "INSERT INTO chunk_annotations (chunk_id, valence) VALUES (1, 0.5)"
        )
        self.con.commit()
        trends = mock.Mock(return_value={
            "emotions": {"top": [{"emotion": "joy", "count": 3}], "chunks_annotated": 3},
            "valence": {"series": [{"period": "2025", "mean_valence": 0.2, "n_chunks": 3}]},
        })
        self.patch_tools(get_word_trends=trends)
        self.patch_llm()
        out = self.out_path("eras.html")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("<h2>Eras &amp; Emotional Arc</h2>", page)
        self.assertIn("Joy", page)
        self.assertIn('<span class="hb-val">3</span>', page)  # emotions as bars
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
            out = self.out_path("silent.html")
            code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
            self.assertEqual(code, 0)
            self.assertIn("excluded (went quiet 2025-01): Bob", stdout)
            self.assertIn('pass --include "Bob" to feature them', stdout)
            with open(out, encoding="utf-8") as f:
                self.assertNotIn("Bob", f.read())

            out2 = self.out_path("silent-include.html")
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
        out = self.out_path("abort.html")
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
        out = self.out_path("nollm.html")
        code, _ = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        llm.assert_not_called()
        os.unlink(out)

    def test_narratives_inserted_and_failure_falls_back(self):
        self._seed_basic()
        self.patch_tools()
        llm = self.patch_llm()
        out = self.out_path("llm.html")
        code, _ = self.run_main(["--yes", "--out", out])
        self.assertEqual(code, 0)
        self.assertGreater(llm.call_count, 0)
        self.assertEqual(llm.call_args.kwargs.get("model"), config.WRAPPED_MODEL)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn('<p class="narrative">A fine narrative.</p>', page)
        os.unlink(out)

        self.patch_llm(side_effect=RuntimeError("ollama down"))
        out2 = self.out_path("llm-fail.html")
        code, _ = self.run_main(["--yes", "--out", out2])
        self.assertEqual(code, 0)  # stats-only fallback, never a crash
        with open(out2, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("Narrative unavailable", page)
        self.assertIn("Messages sent", page)  # stats-only fallback still renders
        os.unlink(out2)

    def test_contact_mode_scopes_calls_and_ambiguity_exits(self):
        self._seed_basic()
        alice = {"id": 1, "first_name": "Alice", "last_name": "", "nickname": None,
                 "handles": [{"id": 1}]}
        mocks = self.patch_tools(
            search_contacts_by_name=mock.Mock(return_value=[alice])
        )
        self.patch_llm()
        out = self.out_path("contact.html")
        code, stdout = self.run_main(
            ["--contact", "Alice", "--year", "2025", "--yes", "--no-llm", "--out", out]
        )
        self.assertEqual(code, 0)
        self.assertIn("Featured: Alice", stdout)
        dyn = mocks["get_response_dynamics"]
        self.assertEqual(dyn.call_args.kwargs.get("contact_id"), 1)
        self.assertEqual(dyn.call_args.kwargs.get("date_from"), "2025-01-01")
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertNotIn("<h2>Your People</h2>", page)  # deep-dive drops corpus sections
        self.assertNotIn("<h2>Groups</h2>", page)
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
        out = self.out_path("exclude.html")
        code, stdout = self.run_main(
            ["--yes", "--no-llm", "--exclude", "Bob", "--out", out]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("Bob", stdout.split("excluded")[0])
        os.unlink(out)

    def test_excluded_person_expands_to_their_direct_chats(self):
        """--exclude NAME must carve the person's 1:1 chats (never their
        group chats) out of every corpus-wide tool call, and their name out
        of the page -- the leak PR #33 fixes."""
        self._seed_basic()
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 2, "+15550002", 2)
        seed_chat(self.con, 3, "c3")  # Bob's 1:1 -- must be excluded
        self.con.executemany(
            "INSERT INTO chat_participants (chat_id, handle_id) VALUES (?, ?)",
            [(3, 2), (2, 1), (2, 2)],  # Bob 1:1; Alice + Bob in group 2
        )
        seed_msgs(self.con, 3, 2, [f"2025-06-{d:02d}T12:00:00" for d in range(1, 4)])
        self.con.commit()
        bob = {"id": 2, "first_name": "Bob", "last_name": "", "nickname": None, "handles": []}
        mocks = self.patch_tools(search_contacts_by_name=mock.Mock(return_value=[bob]))
        self.patch_llm()
        out = self.out_path("exclude-expand.html")
        code, stdout = self.run_main(
            ["--yes", "--no-llm", "--exclude", "Bob", "--out", out]
        )
        self.assertEqual(code, 0)
        self.assertIn("also excluding 1 direct chat(s)", stdout)
        # Every corpus-wide tool call gets the expanded exclusion list: Bob's
        # 1:1 chat 3 is in it, the shared group chat 2 is not.
        for name in ("get_activity_patterns", "get_reaction_stats",
                     "get_word_frequency", "get_word_trends"):
            for call in mocks[name].call_args_list:
                excluded = call.kwargs.get("exclude_chat_id")
                self.assertIn(3, excluded, f"{name} missing the 1:1 exclusion")
                self.assertNotIn(2, excluded, f"{name} must keep the group chat")
        with open(out, encoding="utf-8") as f:
            self.assertNotIn("Bob", f.read())  # _excluded_name_set path
        os.unlink(out)

    def test_business_chats_excluded_from_every_section(self):
        """urn:biz handles must not just be kept off the roster -- their chats
        must be carved out of every corpus-wide tool call, or the raw
        identifier surfaces as a 'contact' in sections like Late Nights."""
        self._seed_basic()
        seed_handle(self.con, 9, "urn:biz:b15ed773-9eed-11e7-baa2-test")
        seed_chat(self.con, 4, "biz1")
        self.con.execute(
            "INSERT INTO chat_participants (chat_id, handle_id) VALUES (4, 9)"
        )
        seed_msgs(self.con, 4, 9, [f"2025-06-{d:02d}T01:00:00" for d in range(1, 4)])
        self.con.commit()
        mocks = self.patch_tools()
        self.patch_llm()
        out = self.out_path("exclude-biz.html")
        code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        self.assertIn("also excluding 1 business chat(s)", stdout)
        self.assertNotIn("urn:biz", stdout.split("business chat(s)")[0])
        for name in ("get_activity_patterns", "get_reaction_stats",
                     "get_word_frequency", "get_word_trends"):
            for call in mocks[name].call_args_list:
                excluded = call.kwargs.get("exclude_chat_id")
                self.assertIn(4, excluded, f"{name} missing the business-chat exclusion")
        with open(out, encoding="utf-8") as f:
            self.assertNotIn("urn:biz", f.read())
        os.unlink(out)


if __name__ == "__main__":
    unittest.main()
