"""Hermetic tests for wrapped.py: temp-file DB, mocked query tools (patched
onto the query_tools package with create=True so they work before the six new
tools are integrated), and mocked ollama.chat_once -- no Ollama, no real DB."""

import io
import os
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import config
import db
import loose_ends
import wellbeing
import ollama
import query_tools
import schema
import wrapped
import wrapped_html
import wrapped_sections

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
    "get_voiceprint",
    "get_language_trajectory",
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

# Un-flagged trajectory envelope (the default): no window fired, so the opt-in
# wellbeing section returns None and is omitted even when enabled.
_TRAJECTORY_UNFLAGGED = {
    "scope": "self",
    "markers_available": ["i_count"],
    "dropped_markers": {},
    "baseline": {"total_words": 40000, "n_history_months": 24, "baseline_months": 24},
    "series": {"i_count": [["2023-01", 0.02], ["2023-02", 0.021]]},
    "windows": [],
    "examples": [],
    "note": "no sustained cross-family departure detected",
    "support_block": None,
}

# A flagged trajectory envelope: one sustained cross-family window. The section
# shades 2023-03..2023-04 and renders the code-emitted support footer.
_TRAJECTORY_FLAGGED = {
    "scope": "self",
    "markers_available": ["i_count", "late_night_share"],
    "dropped_markers": {},
    "baseline": {"total_words": 60000, "n_history_months": 30, "baseline_months": 27},
    "series": {
        "i_count": [["2023-01", 0.02], ["2023-02", 0.022], ["2023-03", 0.05],
                    ["2023-04", 0.06], ["2023-05", 0.03]],
        "late_night_share": [["2023-01", 0.10], ["2023-02", 0.12], ["2023-03", 0.40],
                             ["2023-04", 0.45], ["2023-05", 0.20]],
    },
    "windows": [{
        "start": "2023-03", "end": "2023-04", "peak_z": 4.1,
        "baseline_months": 27,
        "departed_families": ["self_focus", "behavioral"],
        "framing": "A sustained shift in your language from 2023-03 to 2023-04.",
    }],
    "examples": [],
    "note": "reply-latency is intentionally not a v1 marker",
    "support_block": wellbeing.support_block(),
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
    # Default self voiceprint card ("me","*","all"): every main()-driven test
    # renders a self-only Your Voiceprint section (empty consent -> no game).
    "get_voiceprint": {
        "member": "Me", "counterpart": "everyone", "scope": "all",
        "confidence": "high",
        "features": {"avg_msg_len": 42.0, "lowercase_ratio": 0.8, "burst_size": 2.0},
        "openers": [{"phrase": "yo", "count": 30}],
        "closers": [{"phrase": "night", "count": 10}],
    },
    "get_language_trajectory": _TRAJECTORY_UNFLAGGED,
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



def seed_conflict(con, cid, chat_id, first_date, last_date, peak=None, off_text=0):
    con.execute(
        "INSERT INTO conflicts (id, chat_id, first_date, last_date, "
        "peak_signal_json, off_text_flag) VALUES (?, ?, ?, ?, ?, ?)",
        (cid, chat_id, first_date, last_date, json.dumps(peak or {}), off_text),
    )


def seed_signal(con, conflict_id, label_class, actor_is_me=0):
    con.execute(
        "INSERT INTO conflict_signals (conflict_id, message_id, actor_is_me, "
        "label_class, band, excerpt) VALUES (?, ?, ?, ?, ?, ?)",
        (conflict_id, None, actor_is_me, label_class, "likely", ""),
    )


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
        # The Open Loops section (N5) composes from the real get_loose_threads,
        # which resolves query_tools.DB_PATH at call time -- point it at the temp
        # DB so the section reads the (empty, unless seeded) loose_threads table
        # here instead of the developer's real analysis DB.
        qt_patcher = mock.patch.object(query_tools, "DB_PATH", self.db_path)
        qt_patcher.start()
        self.addCleanup(qt_patcher.stop)

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
        seed_chat(self.con, 2, "group1", is_group=1, display_name="The Committee")
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
        self.assertIn("The Committee", page)
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


# --- feature: drift ---
class TestWindowChips(unittest.TestCase):
    """The reconnection-window chip builder (RES-66 / D4) is self-contained
    HTML over the shared .chip class -- no network, no new CSS."""

    def test_render_and_empty(self):
        html = wrapped_html.render_window_chips([
            {"contact": "Zoe", "days_until_dormant": 50, "status": "open"},
            {"contact": "Ada", "days_until_dormant": 1, "status": "closing"},
        ])
        self.assertIn('class="chips"', html)
        self.assertIn("Zoe · 50 days left", html)
        self.assertIn("Ada · 1 day left", html)  # singular unit
        self.assertIn('title="open reconnection window"', html)
        # Nothing to show -> empty string so the caller can drop the block.
        self.assertEqual(wrapped_html.render_window_chips([]), "")


class TestDriftSection(WrappedTestCase):
    """The opt-in "Who's Drifting" Wrapped section (RES-66 / D4): gated on
    DRIFT_ENABLED and rendering only corroborated fading/dormant ties."""

    _TITLE = "Who&#x27;s Drifting"  # esc()'d section heading on the page

    def _seed_basic(self):
        seed_contact(self.con, 1, "Alice")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_chat(self.con, 1, "c1")
        seed_msgs(self.con, 1, 1, [f"2025-05-{d:02d}T12:00:00" for d in range(1, 6)])
        self.con.commit()

    def _seed_drift(self, contact_key, label, phase, corroborated,
                    recip=0.5, latency=0.3, gap_days=40, score=0.7):
        self.con.execute(
            "INSERT INTO dyad_drift (contact_key, label, phase, "
            "phase_corroborated, mandatory_complete, drift_score, "
            "reciprocity_value, latency_value, coverage_json, last_month, "
            "confidence_tier, n_messages) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, '2025-05', 'high', 300)",
            (contact_key, label, phase, corroborated, score, recip, latency,
             f'{{"last_message_days": {gap_days}}}'),
        )
        self.con.commit()

    def _run(self, enabled):
        out = self.out_path(f"drift-{int(enabled)}.html")
        with mock.patch.object(config, "DRIFT_ENABLED", enabled):
            code, _ = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        os.unlink(out)
        return page

    def test_gated_off_by_default(self):
        """DRIFT_ENABLED off -> section absent even with a corroborated tie."""
        self._seed_basic()
        self._seed_drift("contact:2", "Zoe", "fading", 1)
        self.patch_tools()
        self.patch_llm()
        self.assertNotIn(self._TITLE, self._run(enabled=False))

    def test_renders_only_corroborated_cooldowns_when_enabled(self):
        """DRIFT_ENABLED on -> section renders; only phase_corroborated
        fading/dormant ties appear (a non-corroborated tie, and a corroborated
        non-declining phase, are both excluded)."""
        self._seed_basic()
        self._seed_drift("contact:2", "Zoe", "fading", 1, gap_days=40)   # shown
        self._seed_drift("contact:3", "Yara", "fading", 0)               # not corroborated
        self._seed_drift("contact:4", "Xena", "steady", 1)              # not a cooldown
        self.patch_tools()
        self.patch_llm()
        page = self._run(enabled=True)
        self.assertIn(self._TITLE, page)
        self.assertIn("Zoe", page)
        self.assertIn('class="sbars"', page)          # corroborated magnitude split bars
        self.assertIn("Reciprocity erosion", page)
        self.assertIn("Zoe · 50 days left", page)     # open reconnection-window chip
        self.assertNotIn("Yara", page)                # non-corroborated tie excluded
        self.assertNotIn("Xena", page)                # non-declining phase excluded

    def test_excluded_name_kept_out(self):
        """A person the report drops (excluded_names) must not resurface here."""
        self._seed_basic()
        self._seed_drift("contact:2", "Zoe", "fading", 1)
        with mock.patch.object(config, "DRIFT_ENABLED", True):
            shown = wrapped_sections._sec_drifting(self.con, excluded_names=frozenset())
            excluded = wrapped_sections._sec_drifting(self.con, excluded_names={"zoe"})
        self.assertIsNotNone(shown)                 # the tie renders when kept
        self.assertIsNone(excluded)                 # ...and is omitted when excluded


def _vp_card(member, opener, closer="bye"):
    """A minimal but valid voiceprint card (has 'features', so _sec_voiceprint
    accepts it) with a distinctive opener/closer phrase for grep assertions."""
    return {
        "member": member, "counterpart": "everyone", "scope": "all",
        "confidence": "high",
        "features": {"avg_msg_len": 30.0, "lowercase_ratio": 0.6, "burst_size": 2.0},
        "openers": [{"phrase": opener, "count": 7}],
        "closers": [{"phrase": closer, "count": 3}],
    }


def _vp_side_effect(self_card, by_contact_id):
    """A get_voiceprint mock: no contact_id -> the self card; a named contact ->
    that contact's card (or a not-found note), matching persona.get_voiceprint's
    subject='them'/scope='all' call shape from _sec_voiceprint."""
    def vp(contact_id=None, handle_id=None, name=None, subject="me",
           scope=None, chat_id=None, date_from=None, date_to=None):
        if contact_id is None:
            return self_card
        return by_contact_id.get(contact_id, {"found": False, "note": "none"})
    return vp


class TestVoiceprintSection(WrappedTestCase):
    """RES-64 / V4: self voiceprint card + consent-gated guess-who game."""

    _SELF = _vp_card("Me", "selfopenZZZ", "selfcloseZZZ")

    def _seed_two_featured(self):
        """Alice (contact 1) and Bob (contact 2), both featured on 2025 volume."""
        seed_contact(self.con, 1, "Alice")
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_handle(self.con, 2, "+15550002", 2)
        seed_chat(self.con, 1, "c1")
        seed_chat(self.con, 2, "c2")
        seed_msgs(self.con, 1, 1, [f"2025-05-{d:02d}T12:00:00" for d in range(1, 11)])
        seed_msgs(self.con, 2, 2, [f"2025-05-{d:02d}T12:00:00" for d in range(1, 8)])
        self.con.commit()

    def test_self_only_when_consent_empty(self):
        """Default empty consent: the self card renders, but NO third-party
        verbatim phrase or game card is anywhere in the emitted HTML."""
        self._seed_two_featured()
        vp = mock.Mock(side_effect=_vp_side_effect(
            self._SELF, {1: _vp_card("Alice", "wubbaZZZ"),
                         2: _vp_card("Bob", "bobbleZZZ")}))
        self.patch_tools(get_voiceprint=vp)
        self.patch_llm()
        out = self.out_path("vp-self.html")
        with mock.patch.object(config, "WRAPPED_VOICEPRINT_CONSENT_IDS", ""):
            code, _ = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("<h2>Your Voiceprint</h2>", page)
        self.assertIn("This is you", page)
        self.assertIn("selfopenZZZ", page)  # the reader's own phrase renders
        # Default redaction: no third-party card, phrase, or game anywhere.
        self.assertNotIn("Guess Who Texts Like This?", page)
        self.assertNotIn("wubbaZZZ", page)
        self.assertNotIn("bobbleZZZ", page)
        self.assertNotIn("It’s Alice.", page)

    def test_consent_enables_exactly_that_contact(self):
        """Adding one id opts in exactly that contact: their card and reveal
        appear; a non-consented featured contact's card stays out of the file."""
        self._seed_two_featured()
        vp = mock.Mock(side_effect=_vp_side_effect(
            self._SELF, {1: _vp_card("Alice", "wubbaZZZ"),
                         2: _vp_card("Bob", "bobbleZZZ")}))
        self.patch_tools(get_voiceprint=vp)
        self.patch_llm()
        out = self.out_path("vp-consent.html")
        with mock.patch.object(config, "WRAPPED_VOICEPRINT_CONSENT_IDS", "1"):
            code, _ = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("Guess Who Texts Like This?", page)
        self.assertIn("wubbaZZZ", page)      # consented contact opted in
        self.assertIn("It’s Alice.", page)   # reveal carries the name
        self.assertNotIn("bobbleZZZ", page)  # Bob not consented -> redacted

    def test_consent_never_overrides_exclusion(self):
        """A silence-excluded contact who is ALSO consented must not appear:
        consent adds, it never resurfaces an excluded contact."""
        seed_contact(self.con, 1, "Alice")
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_handle(self.con, 2, "+15550002", 2)
        seed_chat(self.con, 1, "c1")
        # Alice active through corpus end (defines MAX(date)); Bob goes quiet
        # 2025-01 after clearing the lifetime floor -> silence-excluded.
        seed_msgs(self.con, 1, 1, [f"2025-{m:02d}-01T12:00:00" for m in range(1, 13)])
        seed_msgs(self.con, 2, 2, ["2024-10-01T12:00:00", "2024-11-01T12:00:00",
                                   "2024-12-01T12:00:00", "2025-01-01T12:00:00",
                                   "2025-01-02T12:00:00"])
        self.con.commit()
        vp = mock.Mock(side_effect=_vp_side_effect(
            self._SELF, {1: _vp_card("Alice", "wubbaZZZ"),
                         2: _vp_card("Bob", "bobbleZZZ")}))
        self.patch_tools(get_voiceprint=vp)
        self.patch_llm()
        out = self.out_path("vp-excluded.html")
        # Both consented, but Bob is excluded by the silent-contact guard.
        with mock.patch.object(config, "WRAPPED_VOICEPRINT_CONSENT_IDS", "1,2"), \
                mock.patch.object(config, "WRAPPED_SILENT_MIN_MESSAGES", 5), \
                mock.patch.object(config, "WRAPPED_SILENT_MONTHS", 6):
            code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        self.assertIn("excluded (went quiet 2025-01): Bob", stdout)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("wubbaZZZ", page)       # Alice: featured + consented
        self.assertNotIn("bobbleZZZ", page)   # Bob: excluded despite consent
        self.assertNotIn("It’s Bob.", page)


class TestRepairReport(WrappedTestCase):
    """C5 / RES-54: the Wrapped Repair Report section. Deterministic aggregates
    over conflicts/conflict_signals, per-class eval-gated. The committed seed
    (eval/seed_gates/conflict_labels.json) pre-clears every class, so the gated
    stats render by default; the suppressed path is driven by patching the
    single gate seam."""

    def _seed_corpus(self):
        """Alice (chat 1) + Bob (chat 2), each a 1:1, with conflict windows:
          - conflict 1 (chat 1, 2025-03, 4h span, off_text=0, volume collapse):
            my stonewalling x2, my criticism x1, one repair
          - conflict 2 (chat 1, 2025-06, 2h span, off_text=0): my stonewalling
            x1, one repair  -> chat 1 has 2 repairs total
          - conflict 3 (chat 2, 2025-04, off_text=1, volume collapse): one
            repair -> EXCLUDED from recovery time (off-text)."""
        seed_contact(self.con, 1, "Alice")
        seed_contact(self.con, 2, "Bob")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_handle(self.con, 2, "+15550002", 2)
        seed_chat(self.con, 1, "c1")
        seed_chat(self.con, 2, "c2")
        self.con.executemany(
            "INSERT INTO chat_participants (chat_id, handle_id) VALUES (?, ?)",
            [(1, 1), (2, 2)],
        )
        seed_conflict(self.con, 1, 1, "2025-03-01T10:00:00", "2025-03-01T14:00:00",
                      peak={"peak_gap_hours": 6,
                            "signals": ["volume_collapse", "latency_withdrawal"]})
        seed_signal(self.con, 1, "stonewalling", actor_is_me=1)
        seed_signal(self.con, 1, "stonewalling", actor_is_me=1)
        seed_signal(self.con, 1, "criticism", actor_is_me=1)
        seed_signal(self.con, 1, "repair", actor_is_me=0)
        seed_conflict(self.con, 2, 1, "2025-06-01T10:00:00", "2025-06-01T12:00:00",
                      peak={"peak_gap_hours": 3, "signals": []})
        seed_signal(self.con, 2, "stonewalling", actor_is_me=1)
        seed_signal(self.con, 2, "repair", actor_is_me=1)
        seed_conflict(self.con, 3, 2, "2025-04-01T10:00:00", "2025-04-05T10:00:00",
                      peak={"peak_gap_hours": 50, "signals": ["volume_collapse"]},
                      off_text=1)
        seed_signal(self.con, 3, "repair", actor_is_me=0)
        self.con.commit()

    def _section(self, contact_entry=None, year=2025, exclude_chat_ids=()):
        return wrapped_sections._sec_repair_report(
            self.con, contact_entry, year, exclude_chat_ids)

    def test_skipped_when_no_conflicts(self):
        # Schema present (conflict tables exist) but no rows -> section skipped,
        # never empty scaffolding.
        self.assertIsNone(self._section())

    def test_renders_all_stats_with_default_gate(self):
        self._seed_corpus()
        sec = self._section()
        self.assertIsNotNone(sec)
        self.assertEqual(sec["title"], "Repair Report")
        # Deterministic base (never gated).
        self.assertEqual(sec["stats"]["conflict_windows"], 3)
        self.assertEqual(sec["stats"]["typical_cool_off_seconds"], 21600)  # median 6h
        self.assertEqual(sec["narrate_stats"]["windows_with_volume_collapse"], 2)
        # Gated horseman: stonewalling (3 of my turns) beats criticism (1).
        self.assertEqual(sec["stats"]["your_top_horseman"], "Stonewalling")
        self.assertEqual(
            sec["narrate_stats"]["your_most_common_horseman"]["your_turns"], 3)
        # Gated repair stats: chat 1 (Alice) has the most repairs.
        self.assertEqual(sec["stats"]["most_repaired_relationship"], "Alice")
        self.assertEqual(
            sec["narrate_stats"]["most_repaired_relationship"]["repair_turns"], 2)

    def test_recovery_trend_excludes_off_text_windows(self):
        self._seed_corpus()
        trend = self._section()["narrate_stats"]["recovery_time_trend"]
        periods = [r["period"] for r in trend]
        # 2025-04 is the off_text=1 window -> excluded; the two off_text=0
        # repaired windows survive with their exact spans.
        self.assertEqual(periods, ["2025-03", "2025-06"])
        self.assertNotIn("2025-04", periods)
        by_period = {r["period"]: r["recovery_seconds"] for r in trend}
        self.assertEqual(by_period["2025-03"], 14400)  # 4h
        self.assertEqual(by_period["2025-06"], 7200)   # 2h

    def test_all_gated_stats_suppressed_when_no_class_clears(self):
        self._seed_corpus()
        with mock.patch.object(wrapped_sections, "_conflict_class_cleared",
                               return_value=False):
            sec = self._section()
        # The deterministic fallback still renders.
        self.assertIsNotNone(sec)
        self.assertEqual(sec["stats"]["conflict_windows"], 3)
        # No label-derived stat survives, in stats OR narration.
        self.assertNotIn("your_top_horseman", sec["stats"])
        self.assertNotIn("most_repaired_relationship", sec["stats"])
        narr = sec["narrate_stats"]
        for key in ("your_most_common_horseman", "most_repaired_relationship",
                    "recovery_time_trend", "typical_recovery"):
            self.assertNotIn(key, narr)

    def test_repair_gate_suppresses_both_repair_stats_together(self):
        self._seed_corpus()

        def gate(con, cls):
            return cls != "repair"  # horsemen clear, repair does not

        with mock.patch.object(wrapped_sections, "_conflict_class_cleared",
                               side_effect=gate):
            sec = self._section()
        # Horseman still renders (its classes cleared)...
        self.assertEqual(sec["stats"]["your_top_horseman"], "Stonewalling")
        # ...but BOTH repair-gated stats vanish together.
        self.assertNotIn("most_repaired_relationship", sec["stats"])
        self.assertNotIn("most_repaired_relationship", sec["narrate_stats"])
        self.assertNotIn("recovery_time_trend", sec["narrate_stats"])
        self.assertNotIn("typical_recovery", sec["narrate_stats"])

    def test_only_gate_cleared_horseman_surfaces(self):
        self._seed_corpus()

        def gate(con, cls):
            # Stonewalling (the true most-common) is NOT cleared; criticism is.
            return cls != "stonewalling"

        with mock.patch.object(wrapped_sections, "_conflict_class_cleared",
                               side_effect=gate):
            sec = self._section()
        # The suppressed top class must not surface; the cleared runner-up does.
        self.assertEqual(sec["stats"]["your_top_horseman"], "Criticism")
        horsemen = [r["horseman"]
                    for r in sec["narrate_stats"]["your_most_common_horseman"]["breakdown"]]
        self.assertNotIn("Stonewalling", horsemen)

    def test_section_appears_self_contained_in_full_report(self):
        # Alice active across 2025 (kept off the silent-exclusion list), with
        # conflict windows on her 1:1 chat.
        seed_contact(self.con, 1, "Alice")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_chat(self.con, 1, "c1")
        self.con.execute(
            "INSERT INTO chat_participants (chat_id, handle_id) VALUES (1, 1)")
        seed_msgs(self.con, 1, 1, [f"2025-{m:02d}-15T12:00:00" for m in range(1, 13)])
        seed_conflict(self.con, 1, 1, "2025-03-01T10:00:00", "2025-03-01T14:00:00",
                      peak={"peak_gap_hours": 6, "signals": ["volume_collapse"]})
        seed_signal(self.con, 1, "stonewalling", actor_is_me=1)
        seed_signal(self.con, 1, "repair", actor_is_me=1)
        self.con.commit()
        self.patch_tools()
        self.patch_llm()
        out = self.out_path("repair.html")
        code, _ = self.run_main(["--year", "2025", "--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<h2>Repair Report</h2>", page)
        self.assertIn("Conflict windows", page)  # base tile label
        self.assertIn("Stonewalling", page)      # gated horseman rendered
        # No JSON reaches the page.
        self.assertNotIn("peak_signal_json", page)
        self.assertNotIn('"conflict_windows"', page)
        os.unlink(out)


class TestWellbeingSection(WrappedTestCase):
    """The opt-in W4 wellbeing trajectory section: off by default, enabled by
    --wellbeing or WRAPPED_WELLBEING, code-emitted support footer, and sanitized
    narration."""

    def _seed_basic(self):
        seed_contact(self.con, 1, "Alice")
        seed_handle(self.con, 1, "+15550001", 1)
        seed_chat(self.con, 1, "c1")
        seed_msgs(self.con, 1, 1, [f"2025-05-{d:02d}T12:00:00" for d in range(1, 6)])
        self.con.commit()

    def test_off_by_default_not_invoked(self):
        self._seed_basic()
        mocks = self.patch_tools()
        self.patch_llm()
        out = self.out_path("wb-off.html")
        with mock.patch.object(config, "WRAPPED_WELLBEING", False):
            code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        # Section builder never runs, so the tool is never called.
        mocks["get_language_trajectory"].assert_not_called()
        self.assertIn("available but off", stdout)  # discoverable in the preview
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertNotIn("A Gentle Check-In", page)
        os.unlink(out)

    def test_flag_enables_and_renders_support_footer(self):
        self._seed_basic()
        mocks = self.patch_tools(get_language_trajectory=_TRAJECTORY_FLAGGED)
        self.patch_llm()
        out = self.out_path("wb-flag.html")
        with mock.patch.object(config, "WRAPPED_WELLBEING", False):
            code, stdout = self.run_main(["--yes", "--no-llm", "--wellbeing", "--out", out])
        self.assertEqual(code, 0)
        mocks["get_language_trajectory"].assert_called()
        self.assertNotIn("available but off", stdout)  # flag flips discoverability off
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("<h2>A Gentle Check-In</h2>", page)
        self.assertIn('class="wb-traj"', page)          # trajectory chart rendered
        self.assertIn('class="wb-col in"', page)          # the flagged window shaded
        self.assertIn('class="wb-support"', page)         # code-emitted support footer
        self.assertIn("988", page)                        # its crisis-line content
        self.assertIn("741741", page)
        self.assertIn("findahelpline.com", page)
        os.unlink(out)

    def test_config_true_enables_without_flag(self):
        self._seed_basic()
        self.patch_tools(get_language_trajectory=_TRAJECTORY_FLAGGED)
        self.patch_llm()
        out = self.out_path("wb-config.html")
        with mock.patch.object(config, "WRAPPED_WELLBEING", True):
            code, stdout = self.run_main(["--yes", "--no-llm", "--out", out])
        self.assertEqual(code, 0)
        self.assertNotIn("available but off", stdout)
        with open(out, encoding="utf-8") as f:
            self.assertIn("<h2>A Gentle Check-In</h2>", f.read())
        os.unlink(out)

    def test_unflagged_window_omits_section(self):
        self._seed_basic()
        self.patch_tools(get_language_trajectory=_TRAJECTORY_UNFLAGGED)
        self.patch_llm()
        out = self.out_path("wb-unflagged.html")
        code, _ = self.run_main(["--yes", "--no-llm", "--wellbeing", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertNotIn("A Gentle Check-In", page)   # nothing to shade -> omitted
        self.assertNotIn('class="wb-support"', page)  # no footer without a window
        os.unlink(out)

    def test_narration_is_sanitized(self):
        self._seed_basic()
        self.patch_tools(get_language_trajectory=_TRAJECTORY_FLAGGED)

        # Only the wellbeing section's blurb names a condition; every other
        # section narrates cleanly, so a residual "depression" anywhere on the
        # page can only mean the sanitizer was skipped.
        def narrate(messages, **kwargs):
            user = messages[1]["content"] if len(messages) > 1 else ""
            if "A Gentle Check-In" in user:
                return {"content": "Honestly this stretch looks like depression."}
            return {"content": "A fine narrative."}

        self.patch_llm(side_effect=narrate)
        out = self.out_path("wb-sanitize.html")
        with mock.patch.object(config, "WRAPPED_WELLBEING", True):
            code, _ = self.run_main(["--yes", "--out", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            page = f.read()
        self.assertNotIn("depression", page)                     # scrubbed
        self.assertIn(wellbeing._CONDITION_FALLBACK, page)       # replaced sentence
        self.assertIn('class="wb-support"', page)                # footer still present
        os.unlink(out)


class TestOpenLoopsSection(WrappedTestCase):
    """The Open Loops board (N5): composes from get_loose_threads and drops
    both muted contacts (Stage-0, upstream) and Wrapped-excluded contacts."""

    def test_filters_name_exclusion_and_batch_hide_notice(self):
        # Mocked get_loose_threads: an occasion (kept), a faded thread for a
        # name-excluded contact (dropped), and the dormant-tie batch-hide notice
        # marker (never a board item).
        threads = [
            {"kind": "inferred_occasion", "chat_id": 100, "handle_id": 10,
             "contact": "Alice", "anchor_date": "2025-03-15T00:00:00",
             "due_by": "2027-03-15", "nudge_text": "Alice's birthday is coming up",
             "urgency": 0.4, "status": "open", "evidence": {}},
            {"kind": "faded_initiator", "chat_id": 200, "handle_id": 20,
             "contact": "Nate", "anchor_date": "2025-01-01T00:00:00",
             "due_by": None, "nudge_text": "Reach out to Nate", "urgency": 0.3,
             "status": "open", "evidence": {}},
            {"notice": "1 dormant-tie reconnect nudge(s) hidden.",
             "kind": "dormant_tie", "hidden": 1},
        ]
        with mock.patch.object(query_tools, "get_loose_threads",
                               mock.Mock(return_value=threads)):
            sec = wrapped_sections._sec_open_loops(excluded_names={"nate"})
        html = sec["extra_html"]
        self.assertEqual(sec["stats"]["open_loops"], 1)
        self.assertIn("Alice", html)
        self.assertNotIn("Nate", html)                 # name-excluded
        self.assertNotIn("dormant-tie reconnect", html)  # notice is not an item
        self.assertIn("2027", html)                    # occasion due date shown

    def test_muted_and_wrapped_excluded_absent_from_rendered_board(self):
        # End-to-end through the REAL detector + get_loose_threads. Alice is
        # kept; Mia is MUTED (Stage-0 writes no row); Wes has a row but is
        # WRAPPED-excluded (his 1:1 chat 300 is in exclude_chat_ids, as
        # wrapped.main would expand it). Both must be absent from the board.
        for cid, name in ((1, "Alice"), (2, "Mia"), (3, "Wes")):
            seed_contact(self.con, cid, name)
            seed_handle(self.con, 10 * cid, f"+{cid}", cid)
            seed_chat(self.con, 100 * cid, f"c{cid}")
            self.con.execute(
                "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
                (100 * cid, 10 * cid))
        for cid in (1, 2, 3):
            for y in ("2024", "2025"):
                seed_msg(self.con, 100 * cid, 0, f"{y}-03-15T09:00:00",
                         is_from_me=1, text="Happy Birthday!!")
        # Mute Mia before the run so Stage-0 suppresses her occasion entirely.
        self.con.execute(
            "INSERT INTO nudge_exclusions(handle_id, contact_id, reason) "
            "VALUES (20, 2, 'muted')")
        self.con.commit()

        with mock.patch.object(loose_ends, "DB_PATH", self.db_path):
            con2 = sqlite3.connect(self.db_path)
            con2.row_factory = sqlite3.Row
            try:
                cfg = loose_ends._build_cfg(con2)
                loose_ends.detect_all(con2, cfg, "2026-07-01T00:00:00+00:00")
            finally:
                con2.close()

        # Wes's 1:1 chat (300) is Wrapped-excluded, exactly as wrapped.main
        # expands an excluded/silent contact into their direct chats.
        sec = wrapped_sections._sec_open_loops(exclude_chat_ids={300})
        html = sec.get("extra_html") or ""
        self.assertIn("Alice", html)
        self.assertNotIn("Wes", html)   # wrapped-excluded (exclude_chat_ids)
        self.assertNotIn("Mia", html)   # muted (Stage-0: no row ever written)


if __name__ == "__main__":
    unittest.main()
