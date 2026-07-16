"""Tests for query_tools' date-bound normalization: _normalize_date_bound and
_date_bound_step. These back the date_from/date_to params of
search_messages_sql / get_chat_summary_stats -- stored dates are unlabeled-UTC
ISO strings compared as plain strings, and an upper bound must come back
half-open (advanced by one unit of the granularity the caller wrote).
"""

import datetime
import unittest

from query_tools import _date_bound_step, _normalize_date_bound


class TestNormalizeDateBound(unittest.TestCase):
    def test_none_passes_through(self):
        self.assertIsNone(_normalize_date_bound(None))

    def test_lower_bound_bare_date_is_midnight(self):
        self.assertEqual(_normalize_date_bound("2026-01-02"), "2026-01-02T00:00:00")

    def test_upper_bound_bare_date_covers_the_whole_day(self):
        self.assertEqual(_normalize_date_bound("2026-01-02", upper=True), "2026-01-03T00:00:00")

    def test_upper_bound_minute_granularity_advances_one_minute(self):
        self.assertEqual(
            _normalize_date_bound("2026-01-02T13:46", upper=True), "2026-01-02T13:47:00"
        )

    def test_upper_bound_second_granularity_advances_one_second(self):
        self.assertEqual(
            _normalize_date_bound("2026-01-02T13:46:50", upper=True), "2026-01-02T13:46:51"
        )

    def test_upper_bound_fractional_seconds_advance_one_microsecond(self):
        self.assertEqual(
            _normalize_date_bound("2026-01-02T13:46:50.500", upper=True),
            "2026-01-02T13:46:50.500001",
        )

    def test_tz_aware_bound_is_converted_to_naive_utc(self):
        self.assertEqual(
            _normalize_date_bound("2026-01-01T13:00:00-05:00"), "2026-01-01T18:00:00"
        )

    def test_z_suffix_is_treated_as_utc(self):
        self.assertEqual(_normalize_date_bound("2026-01-01T13:00:00Z"), "2026-01-01T13:00:00")

    def test_unparseable_bound_raises_value_error(self):
        with self.assertRaises(ValueError):
            _normalize_date_bound("not-a-date")


class TestDateBoundStep(unittest.TestCase):
    def test_bare_date_steps_one_day(self):
        self.assertEqual(_date_bound_step("2026-01-02"), datetime.timedelta(days=1))

    def test_hour_only_steps_one_hour(self):
        self.assertEqual(_date_bound_step("2026-01-02T13"), datetime.timedelta(hours=1))

    def test_minutes_step_one_minute(self):
        self.assertEqual(_date_bound_step("2026-01-02T13:46"), datetime.timedelta(minutes=1))

    def test_seconds_step_one_second(self):
        self.assertEqual(_date_bound_step("2026-01-02T13:46:50"), datetime.timedelta(seconds=1))

    def test_offset_suffix_does_not_change_the_granularity(self):
        self.assertEqual(
            _date_bound_step("2026-01-02T13:46:50-05:00"), datetime.timedelta(seconds=1)
        )


if __name__ == "__main__":
    unittest.main()
