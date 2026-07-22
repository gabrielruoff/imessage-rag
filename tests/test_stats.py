"""Hermetic tests for stats.py (shared robust-statistics primitives). Pure
stdlib math on fixtures -- no DB, no Ollama, no app imports. Also asserts, in a
clean subprocess, that importing `stats` pulls in NO app modules (no config /
Ollama), matching the module's dependency-free contract."""

import os
import subprocess
import sys
import unittest

import stats


class TestQuantile(unittest.TestCase):
    def test_linear_interpolation(self):
        # numpy 'linear' (type 7): interior interpolation between order stats.
        self.assertAlmostEqual(stats._quantile([0, 10], 0.5), 5.0)
        self.assertAlmostEqual(stats._quantile([0, 2.302, 4.605], 0.05), 0.2302)
        self.assertAlmostEqual(stats._quantile([0, 2.302, 4.605], 0.95), 4.3747)

    def test_edges_and_empty(self):
        self.assertEqual(stats._quantile([7], 0.5), 7.0)
        self.assertEqual(stats._quantile([1, 2, 3], 0.0), 1.0)
        self.assertEqual(stats._quantile([1, 2, 3], 1.0), 3.0)
        self.assertIsNone(stats._quantile([], 0.5))


class TestTheilSen(unittest.TestCase):
    def test_perfect_line(self):
        pts = [(0, 1), (1, 3), (2, 5), (3, 7)]  # y = 2x + 1
        slope, intercept = stats.theil_sen(pts)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)

    def test_robust_to_outlier(self):
        # A single wild point does not move the median-of-slopes off y = x.
        pts = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 100)]
        slope, intercept = stats.theil_sen(pts)
        self.assertAlmostEqual(slope, 1.0)
        self.assertAlmostEqual(intercept, 0.0)

    def test_degenerate_returns_none(self):
        self.assertIsNone(stats.theil_sen([]))
        self.assertIsNone(stats.theil_sen([(1, 2)]))            # single point
        self.assertIsNone(stats.theil_sen([(1, 2), (1, 5)]))    # no distinct x


class TestRobustSlope(unittest.TestCase):
    def test_fewer_than_four_points_is_none(self):
        self.assertIsNone(stats.robust_slope([(0, 0), (1, 1), (2, 2)]))

    def test_clear_positive_slope_is_significant(self):
        pts = [(i, 2 * i) for i in range(8)]  # y = 2x, no noise
        out = stats.robust_slope(pts, n_boot=500, seed=0)
        self.assertAlmostEqual(out["slope"], 2.0)
        self.assertAlmostEqual(out["ci_low"], 2.0)
        self.assertAlmostEqual(out["ci_high"], 2.0)
        self.assertTrue(out["significant"])
        self.assertGreater(out["ci_low"], 0.0)

    def test_flat_noisy_slope_ci_includes_zero(self):
        pts = [(i, 1 if i % 2 == 0 else -1) for i in range(8)]  # oscillating, ~0 trend
        out = stats.robust_slope(pts, n_boot=800, seed=0)
        self.assertFalse(out["significant"])
        self.assertLessEqual(out["ci_low"], 0.0)
        self.assertGreaterEqual(out["ci_high"], 0.0)

    def test_bootstrap_is_deterministic_under_seed(self):
        pts = [(i, 1 if i % 2 == 0 else -1) for i in range(8)]
        a = stats.robust_slope(pts, n_boot=800, seed=0)
        b = stats.robust_slope(pts, n_boot=800, seed=0)
        self.assertEqual(a, b)
        # A different seed is allowed to differ; just confirm it still returns.
        self.assertIsNotNone(stats.robust_slope(pts, n_boot=800, seed=1))


class TestKaplanMeierMedian(unittest.TestCase):
    def test_uncensored_matches_step_below_half(self):
        # S drops 1.0 -> .8 -> .6 -> .4 at t=3, the first time S <= 0.5.
        durations = [1, 2, 3, 4, 5]
        censored = [False] * 5
        self.assertAlmostEqual(stats.kaplan_meier_median(durations, censored), 3.0)

    def test_single_observed_event(self):
        self.assertAlmostEqual(stats.kaplan_meier_median([5], [False]), 5.0)

    def test_all_censored_is_none(self):
        self.assertIsNone(stats.kaplan_meier_median([1, 2, 3], [True, True, True]))
        self.assertIsNone(stats.kaplan_meier_median([5], [True]))

    def test_median_never_reached_is_none(self):
        # One early event then heavy censoring: S floors at 0.8, never hits 0.5.
        durations = [1, 2, 3, 4, 5]
        censored = [False, True, True, True, True]
        self.assertIsNone(stats.kaplan_meier_median(durations, censored))

    def test_invalid_input_is_none(self):
        self.assertIsNone(stats.kaplan_meier_median([], []))
        self.assertIsNone(stats.kaplan_meier_median([1, 2], [False]))  # length mismatch


class TestFracOverThreshold(unittest.TestCase):
    def test_survival_at_thresholds(self):
        durations = [1, 2, 3, 4, 5]
        censored = [False] * 5
        self.assertAlmostEqual(stats.frac_over_threshold(durations, censored, 0.5), 1.0)
        self.assertAlmostEqual(stats.frac_over_threshold(durations, censored, 2.5), 0.6)
        self.assertAlmostEqual(stats.frac_over_threshold(durations, censored, 10), 0.0)

    def test_heavy_tail_with_censoring(self):
        # Long silences kept as censored observations rather than dropped.
        durations = [1, 2, 3, 100, 200]
        censored = [False, False, False, True, True]
        # Events only at 1,2,3 (n=5): S = .8 * .75 * (2/3) ... at t=3, n_at_risk=3.
        # t=1: .8 ; t=2: .6 ; t=3: .6*(2/3)=.4. Beyond t=3, curve is flat.
        self.assertAlmostEqual(stats.frac_over_threshold(durations, censored, 3), 0.4)
        self.assertAlmostEqual(stats.frac_over_threshold(durations, censored, 150), 0.4)

    def test_all_censored_is_none(self):
        self.assertIsNone(stats.frac_over_threshold([1, 2], [True, True], 1.0))

    def test_invalid_input_is_none(self):
        self.assertIsNone(stats.frac_over_threshold([], [], 1.0))
        self.assertIsNone(stats.frac_over_threshold([1], [False, True], 1.0))


class TestEcdfPercentile(unittest.TestCase):
    def test_basic_placement(self):
        sample = [1, 2, 3, 4]
        self.assertEqual(stats.ecdf_percentile(2, sample), 0.5)
        self.assertEqual(stats.ecdf_percentile(0, sample), 0.0)
        self.assertEqual(stats.ecdf_percentile(5, sample), 1.0)
        self.assertEqual(stats.ecdf_percentile(4, sample), 1.0)

    def test_empty_sample_is_none(self):
        self.assertIsNone(stats.ecdf_percentile(1.0, []))


class TestWinsorize(unittest.TestCase):
    def test_clamps_to_quantile_band(self):
        series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100]
        w = stats.winsorize(series, (0.1, 0.9))
        lo = stats._quantile(series, 0.1)
        hi = stats._quantile(series, 0.9)
        self.assertTrue(all(lo <= x <= hi for x in w))
        self.assertEqual(max(w), hi)     # the 100 got pulled down to the 90th pct
        self.assertEqual(len(w), len(series))

    def test_empty_and_singleton(self):
        self.assertEqual(stats.winsorize([]), [])
        self.assertEqual(stats.winsorize([5.0]), [5.0])


class TestWinsorizedLogGapEcdf(unittest.TestCase):
    def test_log_space_placement(self):
        sample = [1, 10, 100]  # logs: 0, ln10, ln100
        self.assertAlmostEqual(stats.winsorized_log_gap_ecdf(1, sample), 0.0)
        self.assertAlmostEqual(stats.winsorized_log_gap_ecdf(10, sample), 2 / 3)
        self.assertAlmostEqual(stats.winsorized_log_gap_ecdf(100, sample), 1.0)

    def test_heavy_tail_is_bounded_and_monotone(self):
        sample = [1, 2, 4, 8, 16, 32, 64, 128, 1024]
        small = stats.winsorized_log_gap_ecdf(2, sample)
        big = stats.winsorized_log_gap_ecdf(512, sample)
        self.assertTrue(0.0 <= small <= 1.0)
        self.assertTrue(0.0 <= big <= 1.0)
        self.assertLess(small, big)

    def test_no_positive_values_is_none(self):
        self.assertIsNone(stats.winsorized_log_gap_ecdf(1.0, []))
        self.assertIsNone(stats.winsorized_log_gap_ecdf(1.0, [0, -5, -1]))


class TestModifiedZ(unittest.TestCase):
    def test_flags_outlier(self):
        series = [1, 2, 3, 4, 100]  # median 3, MAD 1
        z = stats.modified_z(series)
        self.assertAlmostEqual(z[2], 0.0)             # the median maps to 0
        self.assertAlmostEqual(z[0], -2 * 0.6745)
        self.assertGreater(z[4], 3.5)                 # outlier above the usual cutoff

    def test_mad_zero_is_none(self):
        self.assertIsNone(stats.modified_z([5, 5, 5, 5]))
        self.assertIsNone(stats.modified_z([5, 5, 5]))  # odd length, still MAD 0

    def test_empty_is_none(self):
        self.assertIsNone(stats.modified_z([]))


class TestLeaveWindowOutBaseline(unittest.TestCase):
    def test_baseline_excludes_window(self):
        series = [1, 2, 3, 4, 5, 6, 7, 8]
        # window [2,4) drops indices 2,3 -> kept values [1,2,5,6,7,8].
        med, mad = stats.leave_window_out_baseline(series, (2, 4))
        self.assertAlmostEqual(med, 5.5)
        self.assertAlmostEqual(mad, 2.0)

    def test_too_few_remaining_is_none(self):
        # window [0,3) leaves only indices 3,4 -> 2 points (< 4).
        self.assertIsNone(stats.leave_window_out_baseline([1, 2, 3, 4, 5], (0, 3)))

    def test_empty_is_none(self):
        self.assertIsNone(stats.leave_window_out_baseline([], (0, 1)))


class TestCusumChangepoints(unittest.TestCase):
    def test_detects_level_shift(self):
        # Ten lows then ten highs; median 5, MAD 5 -> z = -1 then +1.
        series = [0] * 10 + [10] * 10
        pts = stats.cusum_changepoints(series, k=0.5, h=3)
        self.assertEqual(pts, [6, 16])

    def test_flat_series_is_empty(self):
        self.assertEqual(stats.cusum_changepoints([3, 3, 3, 3], k=0.5, h=3), [])

    def test_empty_series_is_empty(self):
        self.assertEqual(stats.cusum_changepoints([], k=0.5, h=3), [])


class TestDependencyContract(unittest.TestCase):
    def test_import_pulls_in_no_app_modules(self):
        # A clean interpreter that imports only `stats` must not have dragged in
        # any app module -- config, Ollama, sqlite, or a query_tools submodule.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        banned = ("config", "ollama", "sqlite3", "sqlite_vec",
                  "query_tools", "chat", "db")
        code = (
            "import sys, stats\n"
            "bad = [m for m in {banned!r} if m in sys.modules]\n"
            "assert not bad, bad\n"
        ).format(banned=banned)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
