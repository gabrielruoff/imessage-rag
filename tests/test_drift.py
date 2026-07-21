"""Hermetic tests for the drift pipeline (drift.py + dyad_drift).

No Ollama and no real chat.db -- pure-function tests over fixtures plus a few
in-memory / temp-file SQLite builds. Covers the D1 acceptance criteria:
identity aggregation, the survival-based latency signal, the CI-excludes-zero
gate, the silence-never-fades corroboration floor, the skip-if-unchanged guard,
and the DRIFT_ENABLED=false no-op. (D5 will extend this same file later.)
"""

import datetime
import json
import math
import os
import sqlite3
import tempfile
import unittest

import config
import drift
import schema
import stats


def _months(n, year=2023):
    return [f"{year}-{m:02d}" for m in range(1, n + 1)]


def _base_cfg(**overrides):
    cfg = {
        "min_messages": 8,
        "min_months": 4,
        "min_valid_months": 4,
        "min_sessions": 2,
        "alpha": 0.05,
        "weights": dict(config.DRIFT_WEIGHTS),
        "thresholds": dict(config.DRIFT_PHASE_THRESHOLDS),
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

class TestSignalTrend(unittest.TestCase):
    def test_ci_excluding_zero_contributes(self):
        rising = {m: i for i, m in enumerate(_months(6))}
        out = drift._signal_trend(rising, lambda v: v, min_months=4, alpha=0.05)
        self.assertTrue(out["present"])
        self.assertTrue(out["significant"])
        self.assertGreater(out["drift_val"], 0.0)

    def test_ci_including_zero_contributes_zero(self):
        # An oscillating series has a slope whose bootstrap CI still spans 0.
        osc = {m: (1 if i % 2 == 0 else -1) for i, m in enumerate(_months(6))}
        out = drift._signal_trend(osc, lambda v: v, min_months=4, alpha=0.05)
        self.assertTrue(out["present"])
        self.assertFalse(out["significant"])
        self.assertEqual(out["drift_val"], 0.0)

    def test_flat_series_contributes_zero(self):
        const = {m: 5 for m in _months(6)}
        out = drift._signal_trend(const, lambda v: v, min_months=4, alpha=0.05)
        self.assertEqual(out["drift_val"], 0.0)
        self.assertFalse(out["significant"])

    def test_too_few_months_is_absent(self):
        out = drift._signal_trend({"2023-01": 1, "2023-02": 2, "2023-03": 3},
                                  lambda v: v, min_months=4, alpha=0.05)
        self.assertFalse(out["present"])
        self.assertEqual(out["drift_val"], 0.0)

    def test_anti_drift_trend_contributes_zero(self):
        # A significant trend in the GROWTH direction (oriented slope < 0) is
        # real but not drift -- it must contribute 0 to the composite.
        rising = {m: i for i, m in enumerate(_months(6))}
        out = drift._signal_trend(rising, lambda v: -v, min_months=4, alpha=0.05)
        self.assertTrue(out["significant"])
        self.assertEqual(out["drift_val"], 0.0)


class TestSurvivalLatency(unittest.TestCase):
    def test_trailing_message_is_right_censored(self):
        obs_end = datetime.datetime(2023, 6, 1, 0, 0, 0)
        stream = [
            (False, datetime.datetime(2023, 1, 1, 12, 0, 0)),   # them
            (True, datetime.datetime(2023, 1, 1, 12, 1, 0)),    # my reply (60s)
            (False, datetime.datetime(2023, 1, 1, 12, 5, 0)),   # them again (240s)
        ]
        obs = drift.reply_latency_observations(stream, obs_end)
        # Two observed replies (uncensored) + one censored trailing observation.
        censored = [o for o in obs if o[2]]
        observed = [o for o in obs if not o[2]]
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(censored), 1)
        self.assertAlmostEqual(observed[0][1], 60.0)
        # The trailing censored duration reaches all the way to observation end.
        self.assertGreater(censored[0][1], 60 * 60 * 24 * 100)

    def test_censored_slow_replies_do_not_trend_downward(self):
        # Early month: mix of fast + one genuinely slow (observed) reply.
        # Late month: the slow replies never came -> right-censored. A naive
        # estimator drops the censored, so its P(slow) FALLS (spurious
        # "improvement"); the survival estimator keeps them, so it does not.
        early = [("2023-01", 30.0, False)] * 3 + [("2023-01", 7200.0, False)]
        late = [("2023-04", 30.0, False)] + [("2023-04", 99999.0, True)] * 3
        frac, km = drift.monthly_latency_signal(early + late)

        self.assertGreaterEqual(frac["2023-04"], frac["2023-01"])   # not downward

        def naive_p_slow(obs, month):
            durs = [d for m, d, c in obs if m == month and not c]
            return sum(1 for d in durs if d > drift._LATENCY_THRESHOLD_SECONDS) / len(durs)

        naive_early = naive_p_slow(early + late, "2023-01")
        naive_late = naive_p_slow(early + late, "2023-04")
        self.assertLess(naive_late, naive_early)   # naive falsely improves
        # The Kaplan-Meier median is kept for the evidence trail (None when the
        # month's replies are mostly censored -- inestimable, not fabricated).
        self.assertIn("2023-01", km)


class TestPhaseFloor(unittest.TestCase):
    def setUp(self):
        self.th = {"thriving": 0.1, "steady": 0.35, "cooling": 0.6, "dormant_days": 90}

    def test_high_drift_without_corroboration_caps_at_cooling(self):
        self.assertEqual(drift.phase(0.9, False, 0, self.th), "cooling")

    def test_high_drift_with_corroboration_is_fading(self):
        self.assertEqual(drift.phase(0.9, True, 0, self.th), "fading")

    def test_dormant_requires_corroboration(self):
        self.assertEqual(drift.phase(0.9, False, 200, self.th), "cooling")
        self.assertEqual(drift.phase(0.9, True, 200, self.th), "dormant")

    def test_low_drift_is_steady_or_thriving(self):
        self.assertEqual(drift.phase(0.0, False, 0, self.th), "thriving")
        self.assertEqual(drift.phase(0.2, False, 0, self.th), "steady")


class TestScoreDyadSilenceOnly(unittest.TestCase):
    """Silence-only fixture: only the silence signal drifts. Even with silence
    weighted to dominate the composite, the corroboration floor keeps the phase
    out of the decline band (fading/dormant)."""

    def _behav(self, months, obs_end):
        max_gap = {m: 60 * (100 ** i) for i, m in enumerate(months)}  # strongly rising
        return {
            "volume": {m: 10 for m in months},               # flat
            "reciprocity": {m: 1.0 for m in months},         # flat, balanced
            "max_gap": max_gap,
            "latency_obs": [(m, 30.0, False) for m in months for _ in range(4)],  # flat/fast
            "all_gaps": list(max_gap.values()),
            "n_messages": 60,
            "last_dt": obs_end,                              # recent -> not dormant
            "both_active": True,
        }

    def test_silence_only_never_fades(self):
        months = _months(6)
        obs_end = datetime.datetime(2023, 6, 15, 12, 0, 0)
        behav = self._behav(months, obs_end)
        winsor = stats.winsorize([math.log(g) for g in behav["max_gap"].values()])
        # Weight silence to dominate: without the floor this WOULD be fading.
        cfg = _base_cfg(weights={s: (1.0 if s == "silence" else 0.0) for s in drift.SIGNALS})

        row = drift.score_dyad("contact:9", "Test", {}, [], [],
                               observation_end=obs_end, winsor_logs=winsor,
                               cfg=cfg, behav=behav)
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["drift_score"], 0.6)     # would be a fading candidate
        self.assertNotIn(row["phase"], ("fading", "dormant"))
        self.assertEqual(row["phase"], "cooling")            # floored
        self.assertEqual(row["phase_corroborated"], 0)
        # silence gate on; the flat mandatory signals present but not gated.
        self.assertEqual(row["silence_gate"], 1)
        self.assertEqual(row["reciprocity_gate"], 0)

    def test_score_dyad_populates_mandatory_coverage_and_series(self):
        months = _months(6)
        obs_end = datetime.datetime(2023, 6, 15, 12, 0, 0)
        behav = self._behav(months, obs_end)
        winsor = stats.winsorize([math.log(g) for g in behav["max_gap"].values()])
        row = drift.score_dyad("contact:9", "Test", {}, [], [],
                               observation_end=obs_end, winsor_logs=winsor,
                               cfg=_base_cfg(), behav=behav)
        # mandatory_complete: all three message-derived signals present.
        self.assertEqual(row["mandatory_complete"], 1)
        coverage = json.loads(row["coverage_json"])
        self.assertEqual(coverage["n_months"], 6)
        self.assertEqual(coverage["first_month"], "2023-01")
        self.assertIn("silence", coverage["signals_present"])
        series = json.loads(row["monthly_series_json"])
        self.assertEqual(len(series["silence"]), 6)
        self.assertTrue(all(len(pt) == 2 for pt in series["silence"]))


class TestScoreDyadCorroboratedDecline(unittest.TestCase):
    def test_corroborated_decline_is_fading(self):
        # Eroding reciprocity + declining volume, both drifting -> corroborated.
        months = _months(6)
        obs_end = datetime.datetime(2023, 6, 15, 12, 0, 0)
        behav = {
            "volume": {m: 20 - 3 * i for i, m in enumerate(months)},
            "reciprocity": {m: max(0.05, 1.0 - 0.18 * i) for i, m in enumerate(months)},
            "max_gap": {m: 3600 for m in months},
            "latency_obs": [(m, 30.0, False) for m in months for _ in range(4)],
            "all_gaps": [3600.0] * 6,
            "n_messages": 90,
            "last_dt": obs_end,
            "both_active": True,
        }
        winsor = stats.winsorize([math.log(3600.0)] * 6)
        cfg = _base_cfg(thresholds={"thriving": 0.1, "steady": 0.2,
                                    "cooling": 0.3, "dormant_days": 90})
        row = drift.score_dyad("contact:5", "Test", {}, [], [],
                               observation_end=obs_end, winsor_logs=winsor,
                               cfg=cfg, behav=behav)
        self.assertEqual(row["phase_corroborated"], 1)
        self.assertEqual(row["phase"], "fading")
        self.assertEqual(row["reciprocity_gate"], 1)


# ---------------------------------------------------------------------------
# DB-backed tests
# ---------------------------------------------------------------------------

def _seed_direct_chat(con, chat_id, handle_id, id_base,
                      n_months=6, per_month=6, year=2023):
    """Insert a 1:1 chat with alternating-sender messages across n_months."""
    con.execute(
        "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
        "VALUES (?, ?, NULL, 0, 'iMessage')", (chat_id, f"chat{chat_id}"))
    con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
                (chat_id, handle_id))
    mid = id_base
    for mo in range(1, n_months + 1):
        for k in range(per_month):
            is_from_me = k % 2
            hid = 0 if is_from_me else handle_id
            date = f"{year}-{mo:02d}-15T12:{k:02d}:00"
            con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
                "text, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mid, f"g{mid}", chat_id, hid, is_from_me, f"msg {mid}", date))
            mid += 1
    return mid


class DriftBuildTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        schema.create_schema(self.con)

        # Contact 1 owns two handles (10, 11) in two distinct 1:1 chats -- must
        # fold into ONE 'contact:1' relationship. Handle 12 is contact-less.
        self.con.execute(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name) "
            "VALUES (1, 'uid-1', 'Ada', 'Lovelace')")
        for hid, cid in ((10, 1), (11, 1), (12, None)):
            self.con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (?, ?, ?, 'iMessage')", (hid, f"+1555000{hid}", cid))
        nxt = _seed_direct_chat(self.con, 100, 10, 1000)
        nxt = _seed_direct_chat(self.con, 101, 11, nxt)
        nxt = _seed_direct_chat(self.con, 102, 12, nxt)
        # A group chat must NEVER produce a drift row.
        self.con.execute(
            "INSERT INTO chats(id, chat_identifier, is_group, service) "
            "VALUES (200, 'grp', 1, 'iMessage')")
        self.con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (200, 10)")
        self.con.execute(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date) "
            "VALUES (?, 'gg', 200, 10, 0, 'group hi', '2023-03-15T09:00:00')", (nxt,))
        self.con.commit()

        self._patch_config(min_messages=8, min_months=4, min_valid_months=4, min_sessions=2)

    def _patch_config(self, **vals):
        names = {"min_messages": "DRIFT_MIN_MESSAGES", "min_months": "DRIFT_MIN_MONTHS",
                 "min_valid_months": "DRIFT_MIN_VALID_MONTHS",
                 "min_sessions": "DRIFT_MIN_SESSIONS"}
        for key, value in vals.items():
            name = names[key]
            saved = getattr(config, name)
            setattr(config, name, value)
            self.addCleanup(setattr, config, name, saved)

    def _rows(self):
        return {r[0]: r for r in self.con.execute(
            "SELECT contact_key, n_messages, phase, drift_score, mandatory_complete "
            "FROM dyad_drift")}

    def test_identity_folds_handles_and_excludes_me_and_unknown(self):
        drift.build_drift(self.con, force=True)
        rows = self._rows()
        keys = set(rows)
        # One folded contact row + one contact-less handle row; no me/unknown/group.
        self.assertIn("contact:1", keys)
        self.assertIn("handle:12", keys)
        self.assertFalse(any(k == "me" or k.startswith("unknown") for k in keys))
        self.assertEqual(keys, {"contact:1", "handle:12"})
        # contact:1 folds BOTH of its chats' messages (36 + 36).
        self.assertEqual(rows["contact:1"][1], 72)
        self.assertEqual(rows["handle:12"][1], 36)

    def test_skip_if_unchanged_and_params_hash_guard(self):
        n1 = drift.build_drift(self.con)
        self.assertGreater(n1, 0)
        # Nothing changed -> skip (0 written), rows preserved.
        before = self._rows()
        self.assertEqual(drift.build_drift(self.con), 0)
        self.assertEqual(self._rows(), before)
        # Changing the weights changes the params hash -> rebuild despite no new
        # messages.
        saved = config.DRIFT_WEIGHTS
        config.DRIFT_WEIGHTS = {**saved, "volume": 0.99}
        self.addCleanup(setattr, config, "DRIFT_WEIGHTS", saved)
        self.assertGreater(drift.build_drift(self.con), 0)

    def test_force_rebuild_is_idempotent(self):
        drift.build_drift(self.con, force=True)
        first = self._rows()
        drift.build_drift(self.con, force=True)
        self.assertEqual(self._rows(), first)   # byte-identical rebuild

    def test_scored_row_has_json_evidence(self):
        drift.build_drift(self.con, force=True)
        gate, coverage, series = self.con.execute(
            "SELECT per_signal_gate_json, coverage_json, monthly_series_json "
            "FROM dyad_drift WHERE contact_key = 'contact:1'").fetchone()
        gate = json.loads(gate)
        self.assertEqual(set(gate), set(drift.SIGNALS))
        self.assertIn("km_median_seconds_by_month", gate["latency"])
        cov = json.loads(coverage)
        self.assertEqual(cov["n_messages"], 72)
        self.assertIn("volume", json.loads(series))


class DriftEnabledFlagTest(unittest.TestCase):
    def test_disabled_leaves_table_untouched(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(db_path) and os.remove(db_path))
        con = sqlite3.connect(db_path)
        schema.create_schema(con)
        # Sentinel row a disabled run must not disturb.
        con.execute("INSERT INTO dyad_drift(contact_key, drift_score) "
                    "VALUES ('contact:sentinel', 0.42)")
        con.commit()
        con.close()

        saved_enabled = config.DRIFT_ENABLED
        saved_path = drift.DB_PATH
        config.DRIFT_ENABLED = False
        drift.DB_PATH = db_path
        self.addCleanup(setattr, config, "DRIFT_ENABLED", saved_enabled)
        self.addCleanup(setattr, drift, "DB_PATH", saved_path)

        self.assertEqual(drift.main([]), 0)

        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT contact_key, drift_score FROM dyad_drift").fetchall()
        finally:
            con.close()
        self.assertEqual(rows, [("contact:sentinel", 0.42)])


if __name__ == "__main__":
    unittest.main()
