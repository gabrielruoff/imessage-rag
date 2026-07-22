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

    def test_params_hash_covers_alpha_and_min_gates(self):
        # Regression: the params hash must cover EVERY scoring input in
        # _current_config(), not just weights/thresholds -- retuning alpha
        # (DRIFT_SLOPE_CI) or a min_* gate previously left the hash unchanged, so
        # build_drift skipped and silently served stale scores.
        base = drift._params_hash()
        for name, delta in (("DRIFT_SLOPE_CI", 0.01), ("DRIFT_MIN_MESSAGES", 1),
                            ("DRIFT_MIN_MONTHS", 1), ("DRIFT_MIN_VALID_MONTHS", 1),
                            ("DRIFT_MIN_SESSIONS", 1)):
            saved = getattr(config, name)
            setattr(config, name, saved + delta)
            try:
                self.assertNotEqual(
                    drift._params_hash(), base,
                    f"changing {name} must change the params hash (else stale skip)")
            finally:
                setattr(config, name, saved)

    def test_annotation_upgrade_forces_rebuild(self):
        # Regression: an in-place re-annotation (--upgrade-distilled flips
        # source distilled->llm and rewrites valence) adds NO new messages, so
        # the message-id + params guard can't see it. The annotation fingerprint
        # must force a rebuild anyway.
        self.con.execute(
            "INSERT INTO chunks(id, chat_id, first_date) VALUES (1, 100, '2023-03-15')")
        self.con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, depth, source) "
            "VALUES (1, 0.0, 1, 'distilled')")
        self.con.commit()
        self.assertGreater(drift.build_drift(self.con), 0)   # writes the annotation hash
        self.assertEqual(drift.build_drift(self.con), 0)     # unchanged -> skip
        # In-place upgrade: same chunk_id (row COUNT unchanged), no new messages.
        self.con.execute(
            "UPDATE chunk_annotations SET source='llm', valence=0.5 WHERE chunk_id=1")
        self.con.commit()
        self.assertGreater(
            drift.build_drift(self.con), 0,
            "re-annotation with no new messages must still force a rebuild")

    def test_sessions_change_forces_rebuild(self):
        # Regression: a SESSION_GAP_MINUTES retune rebuilds session boundaries
        # (which feed the initiation signal) with NO new messages and NO config
        # change _params_hash sees. The sessions fingerprint must force a rebuild.
        self.con.execute(
            "INSERT INTO sessions(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, n_from_me, initiator_is_me, "
            "ender_is_me) VALUES "
            "(9001, 100, 1000, 1005, '2023-03-15', '2023-03-15', 6, 3, 1, 0)")
        self.con.commit()
        self.assertGreater(drift.build_drift(self.con), 0)   # writes the sessions hash
        self.assertEqual(drift.build_drift(self.con), 0)     # unchanged -> skip
        # In-place re-split: row COUNT unchanged, but a boundary moves.
        self.con.execute("UPDATE sessions SET first_message_id = 1001 WHERE id = 9001")
        self.con.commit()
        self.assertGreater(
            drift.build_drift(self.con), 0,
            "a sessions re-split with no new messages must still force a rebuild")

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


# ---------------------------------------------------------------------------
# D5 calibration suite -- the safety net for drift COMPOSITION. These exercise
# how the root drift.py assembles the seven signals into a phase/score, NOT the
# raw robust-statistics primitives (those live in tests/test_stats.py).
# ---------------------------------------------------------------------------

def _month_span(n, *, start_year=2020, start_month=1):
    """`n` consecutive 'YYYY-MM' months from a start, crossing year boundaries
    (unlike `_months`, which is single-year). Returned in chronological order."""
    out, year, mon = [], start_year, start_month
    for _ in range(n):
        out.append(f"{year:04d}-{mon:02d}")
        mon += 1
        if mon > 12:
            mon, year = 1, year + 1
    return out


class TestSurvivalLatencyComposition(unittest.TestCase):
    """The latency signal end-to-end: raw reply observations -> monthly
    P(reply > 1h) survival estimate -> a gated drift contribution."""

    def test_rising_slow_reply_fraction_drives_latency_drift(self):
        # Six months where the fraction of slow (>1h) replies climbs steadily;
        # each month stays estimable (some observed events every month).
        obs = []
        for i, m in enumerate(_months(6)):
            n_slow = i + 1            # 1,2,3,4,5,6 slow replies ...
            n_fast = 6 - i           # ... 5,4,3,2,1,0 fast; 7 observed events/month
            obs += [(m, 30.0, False)] * n_fast
            obs += [(m, 7200.0, False)] * n_slow
        frac, km = drift.monthly_latency_signal(obs)

        vals = [frac[m] for m in _months(6)]
        self.assertEqual(vals, sorted(vals))          # monotonically rising
        self.assertLess(vals[0], vals[-1])
        # Composed through the latency orientation, this is a present, gated,
        # positive drift contribution.
        out = drift._signal_trend(frac, drift._ORIENT["latency"],
                                  min_months=4, alpha=0.05)
        self.assertTrue(out["present"])
        self.assertTrue(out["significant"])
        self.assertGreater(out["drift_val"], 0.0)

    def test_all_censored_month_is_dropped_not_fabricated(self):
        # A month whose only observations are right-censored is inestimable ->
        # omitted from the survival series rather than reported as a spurious 1.0.
        obs = ([("2023-01", 30.0, False)] * 4
               + [("2023-02", 99999.0, True)] * 4)   # all censored
        frac, km = drift.monthly_latency_signal(obs)
        self.assertIn("2023-01", frac)
        self.assertNotIn("2023-02", frac)


class TestSilenceMonthly(unittest.TestCase):
    """ECDF placement of each month's longest gap on the corpus winsorized
    log-gap sample (the silence signal's raw monthly value)."""

    def test_higher_gap_places_at_higher_percentile(self):
        sample = [60, 120, 300, 600, 3600, 7200, 86400]
        winsor = stats.winsorize([math.log(g) for g in sample])
        max_gap = {"2023-01": 60, "2023-02": 3600, "2023-03": 86400}
        out = drift._silence_monthly(max_gap, winsor)
        self.assertLessEqual(out["2023-01"], out["2023-02"])
        self.assertLessEqual(out["2023-02"], out["2023-03"])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in out.values()))

    def test_empty_sample_yields_no_silence(self):
        self.assertEqual(drift._silence_monthly({"2023-01": 3600}, []), {})

    def test_nonpositive_gap_skipped(self):
        winsor = stats.winsorize([math.log(g) for g in (60, 3600, 86400)])
        out = drift._silence_monthly({"2023-01": 0.0, "2023-02": 3600}, winsor)
        self.assertNotIn("2023-01", out)   # log(0) has no placement
        self.assertIn("2023-02", out)


class TestComposite(unittest.TestCase):
    """Weight renormalization over the PRESENT signals: an absent signal leaves
    both numerator and denominator; a present-but-flat (CI-includes-zero) signal
    stays in the denominator but contributes 0 to the numerator."""

    def _sig(self, present, drift_val, *, significant=True, slope=1.0):
        return {"present": present, "significant": significant,
                "drift_val": drift_val, "n_months": 6, "slope": slope,
                "ci_low": None, "ci_high": None}

    def test_renormalizes_over_present_signals(self):
        # Three present signals, equal weights; only volume drifts. The two flat
        # mandatory signals stay in the denominator -> score = 1/3.
        signals = {s: self._sig(False, 0.0) for s in drift.SIGNALS}
        signals["volume"] = self._sig(True, 1.0)
        signals["reciprocity"] = self._sig(True, 0.0, significant=False, slope=0.0)
        signals["latency"] = self._sig(True, 0.0, significant=False, slope=0.0)
        score, mand, present = drift.composite(signals, {s: 1.0 for s in drift.SIGNALS})
        self.assertEqual(score, 0.3333)
        self.assertEqual(set(present), {"volume", "reciprocity", "latency"})
        self.assertTrue(mand)

    def test_absent_signal_drops_from_denominator(self):
        signals = {s: self._sig(False, 0.0) for s in drift.SIGNALS}
        signals["volume"] = self._sig(True, 1.0)
        score, mand, present = drift.composite(signals, {s: 1.0 for s in drift.SIGNALS})
        self.assertEqual(score, 1.0)       # sole present signal drives it fully
        self.assertEqual(present, ["volume"])
        self.assertFalse(mand)             # reciprocity/latency absent

    def test_no_present_signals_scores_zero(self):
        signals = {s: self._sig(False, 0.0) for s in drift.SIGNALS}
        score, mand, present = drift.composite(signals, {s: 1.0 for s in drift.SIGNALS})
        self.assertEqual(score, 0.0)
        self.assertEqual(present, [])
        self.assertFalse(mand)

    def test_weights_are_respected(self):
        # volume weight 3x reciprocity; only volume drifts -> 3/4.
        signals = {s: self._sig(False, 0.0) for s in drift.SIGNALS}
        signals["volume"] = self._sig(True, 1.0)
        signals["reciprocity"] = self._sig(True, 0.0, significant=False, slope=0.0)
        score, _, _ = drift.composite(signals, {"volume": 3.0, "reciprocity": 1.0})
        self.assertEqual(score, 0.75)


class TestMandatoryAndConfidence(unittest.TestCase):
    """The mandatory-set gate (all of volume/reciprocity/latency present) and
    the coverage-driven confidence tier."""

    def _present(self):
        return {"present": True, "significant": False, "drift_val": 0.0,
                "n_months": 6, "slope": 0.0, "ci_low": None, "ci_high": None}

    def _absent(self):
        return {"present": False, "significant": False, "drift_val": 0.0,
                "n_months": 0, "slope": None, "ci_low": None, "ci_high": None}

    def test_mandatory_complete_needs_all_three(self):
        signals = {s: (self._present() if s in drift.MANDATORY else self._absent())
                   for s in drift.SIGNALS}
        _, mand, _ = drift.composite(signals, {s: 1.0 for s in drift.SIGNALS})
        self.assertTrue(mand)
        signals["latency"] = self._absent()
        _, mand2, _ = drift.composite(signals, {s: 1.0 for s in drift.SIGNALS})
        self.assertFalse(mand2)

    def test_confidence_tier_low_without_mandatory(self):
        # No mandatory coverage -> low, no matter how many signals or months.
        self.assertEqual(drift.confidence_tier(False, 7, 20, 1.0), "low")

    def test_confidence_tier_ladder(self):
        self.assertEqual(drift.confidence_tier(True, 5, 8, 1.0), "high")
        self.assertEqual(drift.confidence_tier(True, 5, 8, None), "high")   # no valence rows
        self.assertEqual(drift.confidence_tier(True, 5, 8, 0.2), "medium")  # distilled valence
        self.assertEqual(drift.confidence_tier(True, 5, 6, 1.0), "medium")  # too few months
        self.assertEqual(drift.confidence_tier(True, 3, 8, 1.0), "medium")
        self.assertEqual(drift.confidence_tier(True, 2, 8, 1.0), "low")


class TestCorroborationRule(unittest.TestCase):
    """A decline is only corroborated by reciprocity erosion, or by latency
    inflation while both sides still actively send. Silence and the annotation
    signals never corroborate on their own."""

    def _drift(self):
        return {"present": True, "significant": True, "slope": 1.0,
                "drift_val": 0.5, "n_months": 6, "ci_low": None, "ci_high": None}

    def _flat(self):
        return {"present": True, "significant": False, "slope": 0.0,
                "drift_val": 0.0, "n_months": 6, "ci_low": None, "ci_high": None}

    def test_reciprocity_erosion_corroborates_even_without_both_active(self):
        signals = {s: self._flat() for s in drift.SIGNALS}
        signals["reciprocity"] = self._drift()
        self.assertTrue(drift.corroborated(signals, both_active=False))

    def test_latency_corroborates_only_when_both_active(self):
        signals = {s: self._flat() for s in drift.SIGNALS}
        signals["latency"] = self._drift()
        self.assertFalse(drift.corroborated(signals, both_active=False))
        self.assertTrue(drift.corroborated(signals, both_active=True))

    def test_silence_and_annotations_never_corroborate(self):
        signals = {s: self._flat() for s in drift.SIGNALS}
        for s in ("silence", "valence", "depth", "initiation"):
            signals[s] = self._drift()
        self.assertFalse(drift.corroborated(signals, both_active=True))


class TestBothActiveDefinition(unittest.TestCase):
    """The pinned 'both sides still actively send' definition that gates the
    latency corroborator: >= 3 messages from EACH side in the trailing 60 days
    before the observation end."""

    def test_pinned_constants(self):
        self.assertEqual(drift._RECENT_ACTIVE_DAYS, 60)
        self.assertEqual(drift._RECENT_ACTIVE_MIN, 3)

    def test_three_each_in_window_is_both_active(self):
        obs_end = datetime.datetime(2023, 6, 30, 12, 0, 0)
        recent = datetime.datetime(2023, 6, 15, 12, 0, 0)     # inside trailing 60d
        msgs = [(True, recent)] * 3 + [(False, recent)] * 3
        self.assertTrue(drift._behavioral_monthly({1: msgs}, obs_end)["both_active"])

    def test_one_side_below_three_is_not_both_active(self):
        obs_end = datetime.datetime(2023, 6, 30, 12, 0, 0)
        recent = datetime.datetime(2023, 6, 15, 12, 0, 0)
        msgs = [(True, recent)] * 5 + [(False, recent)] * 2   # them only 2
        self.assertFalse(drift._behavioral_monthly({1: msgs}, obs_end)["both_active"])

    def test_activity_before_the_window_does_not_count(self):
        obs_end = datetime.datetime(2023, 6, 30, 12, 0, 0)
        old = datetime.datetime(2023, 1, 15, 12, 0, 0)        # > 60d before obs_end
        msgs = [(True, old)] * 5 + [(False, old)] * 5
        self.assertFalse(drift._behavioral_monthly({1: msgs}, obs_end)["both_active"])


class TestChatToKeyIdentity(unittest.TestCase):
    """Identity aggregation at the resolution boundary: a contact's several
    handles fold to ONE 'contact:<id>' key, a contact-less handle falls back to
    'handle:<id>', and group / multi-counterparty chats are excluded."""

    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.addCleanup(self.con.close)
        schema.create_schema(self.con)

    def _chat(self, cid, is_group=0):
        self.con.execute(
            "INSERT INTO chats(id, chat_identifier, is_group, service) "
            "VALUES (?, ?, ?, 'iMessage')", (cid, f"c{cid}", is_group))

    def _handle(self, hid, contact_id=None):
        self.con.execute(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')", (hid, f"+1555{hid:04d}", contact_id))

    def _participant(self, cid, hid):
        self.con.execute(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)", (cid, hid))

    def test_fold_fallback_and_exclusions(self):
        self.con.execute(
            "INSERT INTO contacts(id, source_uniqueid, first_name) VALUES (1, 'u1', 'Ada')")
        for hid, cid in ((10, 1), (11, 1), (12, None), (13, None)):
            self._handle(hid, cid)
        self._chat(100)
        self._participant(100, 10)          # contact:1 via handle 10
        self._chat(101)
        self._participant(101, 11)          # contact:1 via handle 11 (folds)
        self._chat(102)
        self._participant(102, 12)          # handle:12 (no contact)
        self._chat(200, is_group=1)
        self._participant(200, 10)          # group -> skip
        self._chat(300)
        self._participant(300, 12)          # multi-counterparty direct ...
        self._participant(300, 13)          # ... -> skip
        self.con.commit()

        mapping = drift._chat_to_key(self.con)
        self.assertEqual(mapping[100][0], "contact:1")
        self.assertEqual(mapping[101][0], "contact:1")   # same key -> folds downstream
        self.assertEqual(mapping[102][0], "handle:12")
        self.assertEqual(mapping[100][1], "Ada")         # label from the contact
        self.assertNotIn(200, mapping)                   # group excluded
        self.assertNotIn(300, mapping)                   # multi-counterparty excluded


class TestMonthlySeriesShape(unittest.TestCase):
    """monthly_series_json is {signal: [[month, value], ...]}, chronological,
    and capped to the last 36 months (the same window the trend is fit over)."""

    def test_shape_and_36_month_cap(self):
        months = _month_span(40)                         # 40 consecutive months
        obs_end = datetime.datetime(2023, 6, 1, 0, 0, 0)
        behav = {
            "volume": {m: 10 for m in months},
            "reciprocity": {m: 1.0 for m in months},
            "max_gap": {},
            "latency_obs": [(m, 30.0, False) for m in months for _ in range(2)],
            "all_gaps": [],
            "n_messages": 400,
            "last_dt": obs_end,
            "both_active": True,
        }
        row = drift.score_dyad("contact:1", "T", {}, [], [],
                               observation_end=obs_end, winsor_logs=[],
                               cfg=_base_cfg(), behav=behav)
        series = json.loads(row["monthly_series_json"])
        self.assertEqual(len(series["volume"]), 36)      # capped from 40
        self.assertTrue(all(len(pt) == 2 for pt in series["volume"]))
        kept = [pt[0] for pt in series["volume"]]
        self.assertEqual(kept, sorted(months)[-36:])     # the most-recent 36
        self.assertEqual(kept, sorted(kept))             # chronological


class DriftPurgeTest(unittest.TestCase):
    """--purge wipes dyad_drift + its three ingest_state guard keys, leaves other
    keys alone, never errors on a fresh DB, and runs regardless of the flag."""

    def _fresh_db(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(db_path) and os.remove(db_path))
        con = sqlite3.connect(db_path)
        schema.create_schema(con)
        return db_path, con

    def _seed(self, con):
        con.execute("INSERT INTO dyad_drift(contact_key, drift_score) "
                    "VALUES ('contact:1', 0.7), ('handle:9', 0.2)")
        drift.set_state(con, "drift_max_message_id", 1234)
        drift.set_state(con, "drift_params_hash", "deadbeef")
        drift.set_state(con, "drift_annotations_hash", "cafef00d")
        drift.set_state(con, "drift_sessions_hash", "5e5510a5")
        con.commit()

    def test_purge_drift_wipes_rows_and_guard_keys(self):
        _db, con = self._fresh_db()
        self.addCleanup(con.close)
        self._seed(con)
        self.assertEqual(drift.purge_drift(con), 2)      # returns deleted-row count
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM dyad_drift").fetchone()[0], 0)
        self.assertIsNone(drift.get_state(con, "drift_max_message_id"))
        self.assertIsNone(drift.get_state(con, "drift_params_hash"))
        self.assertIsNone(drift.get_state(con, "drift_annotations_hash"))
        self.assertIsNone(drift.get_state(con, "drift_sessions_hash"))

    def test_purge_preserves_unrelated_ingest_state(self):
        _db, con = self._fresh_db()
        self.addCleanup(con.close)
        self._seed(con)
        drift.set_state(con, "last_message_rowid", 999)  # an unrelated key
        con.commit()
        drift.purge_drift(con)
        self.assertEqual(drift.get_state(con, "last_message_rowid"), "999")

    def test_purge_on_fresh_db_does_not_error(self):
        _db, con = self._fresh_db()
        self.addCleanup(con.close)
        self.assertEqual(drift.purge_drift(con), 0)

    def test_main_purge_runs_regardless_of_drift_enabled(self):
        # The realistic teardown: DRIFT_ENABLED is still true when you decide to
        # turn the feature off, so --purge must NOT be gated behind the flag.
        db_path, con = self._fresh_db()
        self._seed(con)
        con.close()

        saved_enabled = config.DRIFT_ENABLED
        saved_path = drift.DB_PATH
        config.DRIFT_ENABLED = True
        drift.DB_PATH = db_path
        self.addCleanup(setattr, config, "DRIFT_ENABLED", saved_enabled)
        self.addCleanup(setattr, drift, "DB_PATH", saved_path)

        self.assertEqual(drift.main(["--purge"]), 0)     # exercises the CLI/script path

        con = sqlite3.connect(db_path)
        try:
            n_rows = con.execute("SELECT COUNT(*) FROM dyad_drift").fetchone()[0]
            keys = [r[0] for r in con.execute(
                "SELECT key FROM ingest_state WHERE key IN "
                "('drift_max_message_id', 'drift_params_hash', "
                "'drift_annotations_hash', 'drift_sessions_hash')")]
        finally:
            con.close()
        self.assertEqual(n_rows, 0)
        self.assertEqual(keys, [])


if __name__ == "__main__":
    unittest.main()
