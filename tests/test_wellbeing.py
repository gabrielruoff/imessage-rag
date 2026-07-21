"""Hermetic tests for wellbeing.py (deterministic trajectory / change-point
core, RES-49 / W1). Pure stdlib math on hand-built monthly fixtures -- no DB,
no Ollama, no `config`. Every pinned gate (z_threshold, min_run,
min_baseline_months, family_cap, winsor_limits, cusum_k/h, the cross-family
concordance rule, the MAD==0 marker drop, the leave-window-out baseline, and
the denylist sanitizer) is pinned by a fixture below. A clean `import
wellbeing` is asserted, in a subprocess, to pull in no app modules.

False-positive-rate target: on the adversarial sarcasm/venting suite
(`TestAdversarialInputs`) the detector must fire 0 / N times. Those inputs spike
the lexical markers (absolutist/negation/"I") exactly the way a heavy-sarcasm or
venting stretch does, while behavior stays engaged (laughing, initiating,
daytime, emoji-rich) -- so the behavioral family never departs and the
concordance gate holds the line.
"""

import os
import subprocess
import sys
import unittest

from stats import modified_z

import wellbeing


# --- Fixture builders ------------------------------------------------------
# A 24-month axis with quiet, decorrelated baseline noise per marker (distinct
# phases so family composites average toward a tight baseline) and a strong,
# concurrent departure over the final DEP months. Calibrated so a genuine
# episode clears every gate while single-signal / benign-direction inputs do
# not.

N_MONTHS = 24
DEP_MONTHS = 4

# Distinct small (~0.005) baseline noise cycles, one per marker.
_NOISE = {
    "i_count":          [0.0, 0.006, -0.005, 0.004, -0.006, 0.005],
    "absolutist_count": [0.005, -0.004, 0.006, -0.005, 0.004, -0.006],
    "fw_negation_rate": [-0.005, 0.006, -0.004, 0.005, -0.006, 0.004],
    "late_night_share": [0.004, -0.006, 0.005, -0.004, 0.006, -0.005],
    "initiation_share": [-0.006, 0.005, -0.004, 0.006, -0.005, 0.004],
    "laugh_rate":       [0.006, -0.005, 0.004, -0.006, 0.005, -0.004],
    "emoji_rate":       [-0.004, 0.005, -0.006, 0.004, -0.005, 0.006],
}
_BASE = {
    "i_count": 0.05, "absolutist_count": 0.02, "fw_negation_rate": 0.03,
    "late_night_share": 0.10, "initiation_share": 0.50,
    "laugh_rate": 0.40, "emoji_rate": 0.30,
}
# The concerning-direction departure delta for each marker (sign = the way the
# raw value moves during a real episode: self/absolutist/negation/late-night up,
# initiation/laugh/emoji down).
_DELTA = {
    "i_count": +0.05, "absolutist_count": +0.03, "fw_negation_rate": +0.04,
    "late_night_share": +0.15, "initiation_share": -0.30,
    "laugh_rate": -0.25, "emoji_rate": -0.20,
}
ALL_MARKERS = tuple(_NOISE)


def _months(n, year=2023, month=1):
    out = []
    for _ in range(n):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def build_series(markers=ALL_MARKERS, *, deltas=None, n=N_MONTHS, dep=DEP_MONTHS):
    """Build {marker: [(month, value), ...]} for `markers`. `deltas` overrides
    the per-marker departure applied to the final `dep` months (default: the
    concerning `_DELTA`; pass {} for a flat, no-departure series)."""
    deltas = _DELTA if deltas is None else deltas
    mo = _months(n)
    series = {}
    for name in markers:
        cyc = _NOISE[name]
        vals = []
        for i, m in enumerate(mo):
            v = _BASE[name] + cyc[i % len(cyc)]
            if i >= n - dep:
                v += deltas.get(name, 0.0)
            vals.append((m, v))
        series[name] = vals
    return series


# --- support_block ---------------------------------------------------------

class TestSupportBlock(unittest.TestCase):
    def test_is_constant_nonempty_string(self):
        a = wellbeing.support_block()
        b = wellbeing.support_block()
        self.assertIsInstance(a, str)
        self.assertTrue(a.strip())
        self.assertEqual(a, b)  # code-emitted constant

    def test_contains_a_hotline(self):
        self.assertIn("988", wellbeing.support_block())

    def test_is_direction_independent_and_names_no_condition(self):
        # The block must be safe to show for ANY departure -- it names no
        # clinical condition, so sanitizing it is a no-op.
        clean, redacted = wellbeing.sanitize_condition_names(wellbeing.support_block())
        self.assertFalse(redacted)
        self.assertEqual(clean, wellbeing.support_block())


# --- sanitize_condition_names ----------------------------------------------

class TestSanitizeConditionNames(unittest.TestCase):
    FALLBACK = "a sustained shift in your language patterns"

    def test_clean_text_unchanged(self):
        text = "You have been messaging less at night. That is worth noticing."
        clean, redacted = wellbeing.sanitize_condition_names(text)
        self.assertEqual(clean, text)
        self.assertFalse(redacted)

    def test_redacts_whole_sentence_not_just_word(self):
        text = "This really looks like depression to me."
        clean, redacted = wellbeing.sanitize_condition_names(text)
        self.assertTrue(redacted)
        self.assertNotIn("depression", clean)
        self.assertNotIn("looks like", clean)  # the whole clause is gone
        self.assertIn(self.FALLBACK, clean)

    def test_only_offending_sentence_replaced(self):
        text = "You seem quieter lately. This pattern reads like anxiety. Take care of yourself."
        clean, redacted = wellbeing.sanitize_condition_names(text)
        self.assertTrue(redacted)
        self.assertIn("You seem quieter lately.", clean)
        self.assertIn("Take care of yourself.", clean)
        self.assertNotIn("anxiety", clean)
        self.assertIn(self.FALLBACK, clean)

    def test_case_insensitive_and_acronyms(self):
        for word in ("ANXIETY", "Depression", "OCD", "ADHD", "PTSD", "Bipolar"):
            clean, redacted = wellbeing.sanitize_condition_names(f"He clearly has {word}.")
            self.assertTrue(redacted, word)
            self.assertEqual(clean, self.FALLBACK + ".")

    def test_word_boundary_avoids_false_hits(self):
        # "diagnostics" is not "diagnosis"; "mechanic" is not "manic".
        for text in ("The diagnostics all came back fine.",
                     "The mechanic fixed the car."):
            clean, redacted = wellbeing.sanitize_condition_names(text)
            self.assertFalse(redacted, text)
            self.assertEqual(clean, text)

    def test_every_denylist_word_triggers(self):
        for word in wellbeing._CONDITION_DENYLIST:
            _, redacted = wellbeing.sanitize_condition_names(f"Maybe it is {word} then.")
            self.assertTrue(redacted, word)

    def test_multiple_offending_sentences(self):
        text = "This is trauma. Also a possible disorder. But you are strong."
        clean, redacted = wellbeing.sanitize_condition_names(text)
        self.assertTrue(redacted)
        self.assertNotIn("trauma", clean)
        self.assertNotIn("disorder", clean)
        self.assertIn("But you are strong.", clean)
        self.assertEqual(clean.count(self.FALLBACK), 2)

    def test_preserves_newline_structure(self):
        text = "First line is fine\nThis line mentions bipolar\nLast line ok"
        clean, redacted = wellbeing.sanitize_condition_names(text)
        self.assertTrue(redacted)
        self.assertIn("First line is fine", clean)
        self.assertIn("Last line ok", clean)
        self.assertNotIn("bipolar", clean)

    def test_empty_and_none(self):
        self.assertEqual(wellbeing.sanitize_condition_names(""), ("", False))
        self.assertEqual(wellbeing.sanitize_condition_names(None), (None, False))


# --- analyze_trajectory: true positive -------------------------------------

class TestTruePositive(unittest.TestCase):
    def test_concordant_episode_fires(self):
        res = wellbeing.analyze_trajectory(build_series())
        self.assertTrue(res["flagged"])
        # Window is the final DEP months of the axis.
        mo = _months(N_MONTHS)
        self.assertEqual(res["window"], [mo[N_MONTHS - DEP_MONTHS], mo[-1]])
        self.assertGreaterEqual(res["peak_z"], 3.5)
        # A lexical family AND the behavioral family departed (concordance).
        self.assertTrue(res["families"]["absolutist_negation"]["departed"])
        self.assertTrue(res["families"]["behavioral"]["departed"])
        self.assertEqual(res["families"]["behavioral"]["direction"], "concerning")
        self.assertGreaterEqual(res["baseline_months"], 12)
        self.assertEqual(res["dropped_markers"], [])

    def test_single_marker_family_does_not_carry(self):
        # self_focus has ONE marker (i_count): capped at family_cap it cannot
        # reach z_threshold alone, by design (documented conservative property).
        res = wellbeing.analyze_trajectory(build_series())
        self.assertFalse(res["families"]["self_focus"]["departed"])


# --- analyze_trajectory: concordance gate ----------------------------------

class TestConcordanceGate(unittest.TestCase):
    def test_lexical_only_does_not_fire(self):
        # Only the absolutist/negation lexical family departs; behavior flat.
        series = build_series(
            markers=["absolutist_count", "fw_negation_rate", "late_night_share",
                     "initiation_share", "laugh_rate", "emoji_rate"],
            deltas={"absolutist_count": +0.03, "fw_negation_rate": +0.04},
        )
        res = wellbeing.analyze_trajectory(series)
        self.assertFalse(res["flagged"])
        self.assertTrue(res["families"]["absolutist_negation"]["departed"])
        self.assertFalse(res["families"]["behavioral"]["departed"])

    def test_behavioral_only_does_not_fire(self):
        series = build_series(
            markers=["absolutist_count", "fw_negation_rate", "late_night_share",
                     "initiation_share", "laugh_rate", "emoji_rate"],
            deltas={"late_night_share": +0.15, "initiation_share": -0.30,
                    "laugh_rate": -0.25, "emoji_rate": -0.20},
        )
        res = wellbeing.analyze_trajectory(series)
        self.assertFalse(res["flagged"])
        self.assertTrue(res["families"]["behavioral"]["departed"])
        self.assertFalse(res["families"]["absolutist_negation"]["departed"])


# --- analyze_trajectory: benign-direction excursion ------------------------

class TestBenignDirection(unittest.TestCase):
    def test_benign_behavioral_swing_does_not_fire(self):
        # Behavior moves the WRONG way for concern (more laughing, more emoji,
        # more initiation, less late-night) while the lexical markers rise.
        benign = {
            "absolutist_count": +0.03, "fw_negation_rate": +0.04,
            "late_night_share": -0.05, "initiation_share": +0.30,
            "laugh_rate": +0.25, "emoji_rate": +0.20,
        }
        res = wellbeing.analyze_trajectory(build_series(deltas=benign))
        self.assertFalse(res["flagged"])
        self.assertFalse(res["families"]["behavioral"]["departed"])
        self.assertEqual(res["families"]["behavioral"]["direction"], "benign")


# --- analyze_trajectory: MAD==0 marker drop --------------------------------

class TestMadZeroMarkerDrop(unittest.TestCase):
    def test_constant_marker_dropped_family_survives(self):
        # A perfectly constant marker has MAD==0 -> dropped (never
        # epsilon-divided). The 4-member behavioral family loses emoji_rate but
        # still fires on its three survivors.
        series = build_series()
        mo = _months(N_MONTHS)
        series["emoji_rate"] = [(m, 0.30) for m in mo]  # constant
        res = wellbeing.analyze_trajectory(series)
        self.assertEqual(res["dropped_markers"], ["emoji_rate"])
        self.assertNotIn("emoji_rate", res["families"]["behavioral"]["markers"])
        self.assertTrue(res["families"]["behavioral"]["departed"])
        self.assertTrue(res["flagged"])

    def test_all_dropped_markers_reported_sorted(self):
        series = build_series()
        mo = _months(N_MONTHS)
        series["emoji_rate"] = [(m, 0.30) for m in mo]
        series["laugh_rate"] = [(m, 0.40) for m in mo]
        res = wellbeing.analyze_trajectory(series)
        self.assertEqual(res["dropped_markers"], ["emoji_rate", "laugh_rate"])


# --- analyze_trajectory: baseline sufficiency ------------------------------

class TestBaselineRequirement(unittest.TestCase):
    def test_too_few_baseline_months_does_not_fire(self):
        # 14 months total with a 3-month departure leaves only 11 baseline
        # months (< min_baseline_months=12), so nothing can fire.
        res = wellbeing.analyze_trajectory(build_series(n=14, dep=3))
        self.assertFalse(res["flagged"])

    def test_exactly_enough_baseline_fires(self):
        # 15 months, 3-month departure -> 12 baseline months, the boundary.
        res = wellbeing.analyze_trajectory(build_series(n=15, dep=3))
        self.assertTrue(res["flagged"])
        self.assertGreaterEqual(res["baseline_months"], 12)

    def test_custom_min_baseline_enforced(self):
        # Raising the requirement above what the fixture can supply blocks it.
        series = build_series(n=15, dep=3)
        self.assertFalse(
            wellbeing.analyze_trajectory(series, min_baseline_months=13)["flagged"])


# --- analyze_trajectory: leave-window-out ----------------------------------

class TestLeaveWindowOut(unittest.TestCase):
    def _behavioral_composite(self, series, family_cap=2.0):
        """Reconstruct the behavioral capped composite from public pieces, so we
        can contrast a full-series modified-z against the module's peak_z."""
        members = ["late_night_share", "initiation_share", "laugh_rate", "emoji_rate"]
        zmaps = {}
        for name in members:
            vals = [v for _, v in series[name]]
            zmaps[name] = wellbeing._member_z(vals, wellbeing.MARKERS[name][1], (0.05, 0.95))
        n = len(series[members[0]])
        comp = []
        for i in range(n):
            clamped = [max(-family_cap, min(family_cap, zmaps[nm][i])) for nm in members]
            comp.append(sum(clamped) / len(clamped))
        return comp

    def test_window_cannot_inflate_its_own_denominator(self):
        series = build_series()
        res = wellbeing.analyze_trajectory(series)
        self.assertTrue(res["flagged"])
        # A NAIVE full-series modified-z lets the departure inflate the MAD it is
        # scored against; the leave-window-out peak_z excludes the window and is
        # therefore strictly larger.
        full = modified_z(self._behavioral_composite(series))
        self.assertLess(max(full), res["peak_z"])


# --- analyze_trajectory: adversarial (false-positive target) ---------------

class TestAdversarialInputs(unittest.TestCase):
    """Sarcasm / venting stress inputs. FP target: 0 fires across the suite."""

    def _cases(self):
        # Sarcasm: absolutist + negation + "I" all spike, but the person is
        # laughing MORE, using MORE emoji, initiating MORE, at normal hours.
        yield "sarcasm", {
            "i_count": +0.06, "absolutist_count": +0.06, "fw_negation_rate": +0.05,
            "late_night_share": -0.03, "initiation_share": +0.20,
            "laugh_rate": +0.22, "emoji_rate": +0.16,
        }
        # Venting: "I" and negation spike, but engagement stays high.
        yield "venting", {
            "i_count": +0.07, "fw_negation_rate": +0.05, "absolutist_count": +0.02,
            "late_night_share": +0.02, "initiation_share": +0.15,
            "laugh_rate": +0.10, "emoji_rate": +0.05,
        }
        # A single loud month (spike, not a run) across every marker.
        yield "one_loud_month", None  # handled specially below

    def test_no_adversarial_input_fires(self):
        fires = 0
        total = 0
        for name, deltas in self._cases():
            total += 1
            if name == "one_loud_month":
                series = build_series(dep=1, deltas=_DELTA)
            else:
                series = build_series(deltas=deltas)
            res = wellbeing.analyze_trajectory(series)
            if res["flagged"]:
                fires += 1
        self.assertEqual(fires, 0, f"{fires}/{total} adversarial inputs fired")


# --- output contract / determinism / registry ------------------------------

class TestOutputContract(unittest.TestCase):
    def test_output_shape_and_types(self):
        res = wellbeing.analyze_trajectory(build_series())
        self.assertEqual(
            set(res),
            {"flagged", "window", "peak_z", "families", "baseline_months",
             "dropped_markers"},
        )
        self.assertIsInstance(res["flagged"], bool)
        self.assertIsInstance(res["window"], list)
        self.assertEqual(len(res["window"]), 2)
        self.assertIsInstance(res["peak_z"], float)
        self.assertIsInstance(res["baseline_months"], int)
        self.assertIsInstance(res["dropped_markers"], list)
        for fam, block in res["families"].items():
            self.assertEqual(set(block), {"departed", "direction", "markers"})
            self.assertIsInstance(block["departed"], bool)
            self.assertIsInstance(block["direction"], str)
            self.assertIsInstance(block["markers"], dict)

    def test_empty_input_is_unflagged_not_raised(self):
        res = wellbeing.analyze_trajectory({})
        self.assertFalse(res["flagged"])
        self.assertIsNone(res["window"])
        self.assertIsNone(res["peak_z"])
        self.assertEqual(res["baseline_months"], 0)
        self.assertEqual(res["dropped_markers"], [])
        # None also tolerated.
        self.assertFalse(wellbeing.analyze_trajectory(None)["flagged"])

    def test_unknown_markers_ignored(self):
        res = wellbeing.analyze_trajectory({"question_count": [("2023-01", 1.0)],
                                            "we_count": [("2023-02", 2.0)]})
        self.assertFalse(res["flagged"])
        self.assertEqual(res["dropped_markers"], [])  # unknown != dropped

    def test_deterministic(self):
        a = wellbeing.analyze_trajectory(build_series())
        b = wellbeing.analyze_trajectory(build_series())
        self.assertEqual(a, b)


class TestMarkerRegistry(unittest.TestCase):
    def test_pinned_markers_families_directions(self):
        self.assertEqual(wellbeing.MARKERS, {
            "i_count": ("self_focus", "up"),
            "absolutist_count": ("absolutist_negation", "up"),
            "fw_negation_rate": ("absolutist_negation", "up"),
            "late_night_share": ("behavioral", "up"),
            "initiation_share": ("behavioral", "down"),
            "laugh_rate": ("behavioral", "down"),
            "emoji_rate": ("behavioral", "down"),
        })
        self.assertEqual(wellbeing.FAMILY_TYPE, {
            "self_focus": "lexical",
            "absolutist_negation": "lexical",
            "behavioral": "behavioral",
        })

    def test_question_and_we_are_not_markers(self):
        self.assertNotIn("question_count", wellbeing.MARKERS)
        self.assertNotIn("we_count", wellbeing.MARKERS)

    def test_hit_floor_constant_pinned(self):
        self.assertEqual(wellbeing.WELLBEING_MIN_MARKER_HITS, 30)


class TestDependencyContract(unittest.TestCase):
    def test_import_pulls_in_no_app_modules(self):
        # A clean interpreter importing only `wellbeing` must not drag in any
        # app module -- config, Ollama, sqlite, or a query_tools submodule.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        banned = ("config", "ollama", "sqlite3", "sqlite_vec",
                  "query_tools", "chat", "db")
        code = (
            "import sys, wellbeing\n"
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
