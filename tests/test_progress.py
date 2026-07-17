"""Hermetic tests for progress.py (the shared CLI progress helper). Pure
stdout capture -- no DB, no Ollama, no terminal. Every test runs with stdout
redirected to a StringIO, so 'auto' resolves to plain mode exactly the way it
does under the rest of the suite."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import progress


def _run(n, style="plain", **kwargs):
    """Drive one tracked loop of n advances under `style`; return stdout lines."""
    out = io.StringIO()
    with mock.patch.object(progress, "PROGRESS_STYLE", style):
        with redirect_stdout(out):
            with progress.track("Working", **kwargs) as p:
                for _ in range(n):
                    p.advance()
    return out.getvalue().splitlines()


class TestModeSelection(unittest.TestCase):
    def test_off_prints_nothing(self):
        self.assertEqual(_run(500, style="off", total=500), [])

    def test_disabled_prints_nothing(self):
        self.assertEqual(_run(500, total=500, enabled=False), [])

    def test_auto_without_tty_degrades_to_plain(self):
        lines = _run(500, style="auto", total=500, log_every=100)
        self.assertEqual(lines[0], "Working: 500")
        self.assertNotIn("\x1b", "\n".join(lines))  # no escape codes when piped

    def test_zero_total_is_silent_even_in_bar_mode(self):
        # An empty loop reports nothing in either mode -- no header, no
        # flash of an empty bar.
        self.assertEqual(_run(0, total=0), [])
        with mock.patch.object(progress, "_use_bar", return_value=True):
            self.assertEqual(_run(0, style="auto", total=0), [])

    def test_bar_only_is_silent_in_plain_mode(self):
        self.assertEqual(_run(500, total=500, bar_only=True), [])


class TestPlainOutput(unittest.TestCase):
    def test_header_and_stride(self):
        lines = _run(10, total=10, log_every=3)
        self.assertEqual(lines[0], "Working: 10")
        self.assertEqual([li.split()[0] for li in lines[1:]],
                         ["...3/10", "...6/10", "...9/10"])
        self.assertIn("(30%)", lines[1])

    def test_header_false_suppresses_opening_line(self):
        lines = _run(10, total=10, log_every=5, header=False)
        self.assertTrue(all(li.lstrip().startswith("...") for li in lines))

    def test_line_cap_stretches_stride_on_big_loops(self):
        # PROGRESS_LOG_EVERY is a floor on the stride; _MAX_PLAIN_LINES wins
        # when it's coarser. 100k items at log_every=200 would be 500 lines --
        # the cap stretches the stride to 2000 instead.
        lines = _run(100_000, total=100_000, log_every=200)
        self.assertLessEqual(len(lines) - 1, progress._MAX_PLAIN_LINES)
        self.assertEqual(lines[1].split()[0], "...2000/100000")

    def test_unknown_total_reports_count_and_elapsed(self):
        lines = _run(7, total=None, log_every=3)
        self.assertEqual(lines[0], "Working...")
        self.assertRegex(lines[1], r"^  \.\.\.3  \(\d+s\)$")

    def test_eta_only_after_rate_has_substance(self):
        out = io.StringIO()
        with mock.patch.object(progress, "PROGRESS_STYLE", "plain"):
            with redirect_stdout(out):
                with progress.track("Working", total=100, log_every=10,
                                    header=False) as p:
                    p.advance(10)          # elapsed ~0s: no ETA yet
                    p.started -= 20        # backdate: rate now has substance
                    p.advance(10)
        first, second = out.getvalue().splitlines()
        self.assertNotIn("eta", first)
        self.assertIn("eta", second)

    def test_batched_advance_counts_all_items(self):
        out = io.StringIO()
        with mock.patch.object(progress, "PROGRESS_STYLE", "plain"):
            with redirect_stdout(out):
                with progress.track("Working", total=64, log_every=16,
                                    header=False) as p:
                    for _ in range(4):
                        p.advance(16)
        self.assertEqual(out.getvalue().splitlines()[-1].split()[0], "...64/64")


class TestNestedTracks(unittest.TestCase):
    def test_parent_joins_the_same_live_display_in_bar_mode(self):
        # rich allows one live display at a time, so a nested track must add
        # a task to the parent's Progress rather than start its own.
        out = io.StringIO()
        with mock.patch.object(progress, "PROGRESS_STYLE", "auto"), \
                mock.patch.object(progress, "_use_bar", return_value=True), \
                redirect_stdout(out):
            with progress.track("Outer", total=2, transient=True) as outer:
                self.assertIsInstance(outer, progress._Rich)
                with progress.track("Inner", total=3, parent=outer,
                                    transient=True, bar_only=True) as inner:
                    self.assertIs(inner._progress, outer._progress)
                    self.assertNotEqual(inner._task_id, outer._task_id)
                    inner.advance(3)
                # transient inner task is removed once its loop ends.
                self.assertEqual(outer._progress.task_ids, [outer._task_id])
                outer.advance(2)

    def test_parent_is_inert_outside_bar_mode(self):
        # A plain-mode parent handle isn't a _Rich, so the nested track runs
        # the normal mode selection -- here bar_only, so it stays silent.
        out = io.StringIO()
        with mock.patch.object(progress, "PROGRESS_STYLE", "plain"):
            with redirect_stdout(out):
                with progress.track("Outer", total=2, bar_only=True) as outer:
                    with progress.track("Inner", total=3, parent=outer,
                                        bar_only=True) as inner:
                        inner.advance(3)
                    outer.advance(2)
        self.assertEqual(out.getvalue(), "")


class TestFmtDuration(unittest.TestCase):
    def test_units(self):
        self.assertEqual(progress._fmt_duration(42), "42s")
        self.assertEqual(progress._fmt_duration(90), "1m30s")
        self.assertEqual(progress._fmt_duration(3660), "1h01m")


if __name__ == "__main__":
    unittest.main()
