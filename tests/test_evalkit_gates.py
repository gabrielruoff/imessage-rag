"""Hermetic tests for the RES-70 / F4 eval-gate harness: agreement math
(agreement.py), the label/gate fixture flow (labeling.py), the verdict shape +
3-step ``eval_gate_passed`` precedence (gates.py), and the committed conflict
seed's validity/consistency. No Ollama; the gate's model side is mocked; the
only DB is an in-memory SQLite stub.

Run from the repo root:  python3 -m unittest tests.test_evalkit_gates
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from eval.evalkit import EVAL_DIR, agreement, gates, labeling


# --- agreement.py math -------------------------------------------------------


class KappaMathTest(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        gold = ["a", "b", "a", "b"]
        self.assertEqual(agreement.cohens_kappa(gold, list(gold)), 1.0)

    def test_chance_agreement_is_zero(self):
        # p_o == p_e by construction -> kappa 0 (see the hand calc in the PR).
        gold = ["a", "a", "b", "b"]
        pred = ["a", "b", "a", "b"]
        self.assertAlmostEqual(agreement.cohens_kappa(gold, pred), 0.0)

    def test_empty_and_degenerate_are_none(self):
        self.assertIsNone(agreement.cohens_kappa([], []))
        # single label used everywhere on both sides: p_e == 1 -> undefined
        self.assertIsNone(agreement.cohens_kappa(["x", "x"], ["x", "x"]))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            agreement.cohens_kappa(["a"], ["a", "b"])

    def test_per_class_one_vs_rest(self):
        gold = ["a", "a", "b", "b"]
        pred = ["a", "a", "b", "b"]
        per = agreement.kappa_per_class(gold, pred)
        self.assertEqual(per["a"], 1.0)
        self.assertEqual(per["b"], 1.0)

    def test_small_sample_refusal(self):
        # 'rare' has support 1 (< min_labels 3) -> None; 'common' scores.
        gold = ["common", "common", "common", "rare"]
        pred = ["common", "common", "common", "rare"]
        per = agreement.kappa_per_class(gold, pred, min_labels=3)
        self.assertIsNone(per["rare"])
        self.assertIsNotNone(per["common"])


class PrecisionMathTest(unittest.TestCase):
    def test_precision_recall_hand_fixture(self):
        gold = ["pos", "pos", "neg", "pos"]
        pred = ["pos", "neg", "pos", "pos"]
        out = agreement.precision_recall(gold, pred, positive="pos")
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fp"], 1)
        self.assertEqual(out["fn"], 1)
        self.assertAlmostEqual(out["precision"], 2 / 3)
        self.assertAlmostEqual(out["recall"], 2 / 3)
        self.assertEqual(out["support"], 3)  # gold positives

    def test_precision_undefined_when_no_predicted_positive(self):
        out = agreement.precision_recall(["pos", "neg"], ["neg", "neg"], positive="pos")
        self.assertIsNone(out["precision"])
        self.assertEqual(out["support"], 1)


# --- compute_verdict shape ---------------------------------------------------


class ComputeVerdictTest(unittest.TestCase):
    def test_kappa_verdict_passes_and_shape(self):
        cfg = {"metric": "kappa", "floor": 0.6, "classes": ("a", "b")}
        gold = ["a", "a", "b", "b"]
        pred = ["a", "a", "b", "b"]
        v = gates.compute_verdict(cfg, gold, pred, min_labels=2, min_positives=2,
                                  labeled_at="T", source="owner")
        self.assertEqual(set(v), {"passed", "failed", "metric", "values",
                                  "n_labels", "labeled_at", "source"})
        self.assertEqual(v["passed"], ["a", "b"])
        self.assertEqual(v["failed"], [])
        self.assertEqual(v["metric"], "kappa")
        self.assertEqual(v["n_labels"], 4)
        self.assertEqual(v["source"], "owner")

    def test_below_sample_floor_never_passes(self):
        cfg = {"metric": "kappa", "floor": 0.6, "classes": ("a", "b")}
        gold = ["a", "a", "b", "b"]
        pred = ["a", "a", "b", "b"]
        v = gates.compute_verdict(cfg, gold, pred, min_labels=99, min_positives=2,
                                  labeled_at="T", source="owner")
        self.assertEqual(v["passed"], [])
        self.assertEqual(sorted(v["failed"]), ["a", "b"])
        self.assertIsNone(v["values"]["a"])

    # The shipped dropped_plan shape: a two-class label space gated on "positive".
    _PRECISION_CFG = {"metric": "precision", "floor": 0.85,
                      "classes": ("dropped_plan", "not_dropped_plan"),
                      "positive": "dropped_plan"}

    def test_precision_verdict_fails_on_false_positive(self):
        # A false positive (gold not_dropped_plan, predicted dropped_plan) drives
        # precision below the floor -> the gate FAILS. This path is UNREACHABLE
        # with a single-class label space, where every gold is positive, fp is
        # structurally 0, and precision is a spurious 1.0 (finding #2).
        gold = ["dropped_plan"] * 3 + ["not_dropped_plan"]
        pred = ["dropped_plan"] * 4                      # over-fires on the negative
        v = gates.compute_verdict(self._PRECISION_CFG, gold, pred, min_labels=2,
                                  min_positives=3, labeled_at="T", source="owner")
        self.assertEqual(v["failed"], ["dropped_plan"])
        self.assertEqual(v["passed"], [])
        self.assertEqual(v["values"]["dropped_plan"], 0.75)

    def test_precision_verdict_passes_with_real_negative(self):
        # Classifier correct on the negative -> precision 1.0 with a false
        # positive POSSIBLE (gold holds a negative) -> the gate clears.
        gold = ["dropped_plan"] * 3 + ["not_dropped_plan"]
        pred = ["dropped_plan"] * 3 + ["not_dropped_plan"]
        v = gates.compute_verdict(self._PRECISION_CFG, gold, pred, min_labels=2,
                                  min_positives=3, labeled_at="T", source="owner")
        self.assertEqual(v["passed"], ["dropped_plan"])
        self.assertEqual(v["values"]["dropped_plan"], 1.0)

    def test_precision_single_class_gold_is_unmeasurable(self):
        # finding #2 regression: an all-positive gold (the only thing a
        # single-class label space could produce) is UNMEASURABLE -- fp is
        # structurally 0, so precision is a spurious 1.0. The degenerate guard
        # must FAIL it closed, never pass it (mirrors the kappa p_e>=1.0 guard).
        gold = ["dropped_plan"] * 4
        pred = ["dropped_plan"] * 4
        v = gates.compute_verdict(self._PRECISION_CFG, gold, pred, min_labels=2,
                                  min_positives=3, labeled_at="T", source="owner")
        self.assertEqual(v["passed"], [])
        self.assertIn("dropped_plan", v["failed"])


# --- label / gate fixture flow -----------------------------------------------


def _fixture_con():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, text TEXT, n_messages INT)")
    con.execute("CREATE TABLE ingest_state(key TEXT PRIMARY KEY, value TEXT)")
    con.executemany(
        "INSERT INTO chunks(id, text, n_messages) VALUES (?, ?, 5)",
        [(f"ch{i}", f"synthetic chunk {i}") for i in range(8)],
    )
    con.commit()
    return con


FIXTURE_TASK = {
    "metric": "kappa",
    "floor": 0.6,
    "classes": ("a", "b"),
    "sample_sql": "SELECT id AS id, text AS text FROM chunks ORDER BY id LIMIT ?",
    "seed": False,
    "classifier": None,
}

PRECISION_FIXTURE_TASK = {
    "metric": "precision",
    "floor": 0.85,
    "classes": ("pos", "neg"),
    "positive": "pos",
    "sample_sql": "SELECT id AS id, text AS text FROM chunks ORDER BY id LIMIT ?",
    "seed": False,
    "classifier": None,
}


class LabelFlowTest(unittest.TestCase):
    def setUp(self):
        self.con = _fixture_con()
        self.addCleanup(self.con.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.labels_path = Path(self.tmp.name) / "fixture_task.jsonl"

    def test_label_appends_and_is_resumable(self):
        ask = lambda item, classes: "a"  # noqa: E731
        added = labeling.run_label(self.con, "fixture_task", FIXTURE_TASK,
                                   self.labels_path, n=2, ask=ask)
        self.assertEqual(added, 2)
        first = labeling.load_labels(self.labels_path)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(r["gold"] == "a" for r in first))
        # Resume: previously labeled ids are skipped, new ones appended.
        added2 = labeling.run_label(self.con, "fixture_task", FIXTURE_TASK,
                                    self.labels_path, n=2, ask=ask)
        self.assertEqual(added2, 2)
        all_recs = labeling.load_labels(self.labels_path)
        self.assertEqual(len(all_recs), 4)
        ids = [r["id"] for r in all_recs]
        self.assertEqual(len(ids), len(set(ids)))  # no duplicates
        self.assertEqual(ids[:2], [r["id"] for r in first])  # earlier labels intact

    def test_label_skip_and_quit(self):
        seq = iter(["a", None, "__quit__", "a"])
        ask = lambda item, classes: next(seq)  # noqa: E731
        added = labeling.run_label(self.con, "fixture_task", FIXTURE_TASK,
                                   self.labels_path, n=5, ask=ask)
        self.assertEqual(added, 1)  # one label, then a skip, then quit


class GateFlowTest(unittest.TestCase):
    def setUp(self):
        self.con = _fixture_con()
        self.addCleanup(self.con.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.labels_path = Path(self.tmp.name) / "fixture_task.jsonl"
        # Four owner labels: gold [a,a,b,b].
        golds = ["a", "a", "b", "b"]
        with open(self.labels_path, "w") as f:
            for i, g in enumerate(golds):
                f.write(json.dumps({"id": f"ch{i}", "text": f"t{i}", "gold": g}) + "\n")

    def test_gate_runs_classifier_and_persists(self):
        # Mock the model side: perfect predictions keyed by id.
        pred_map = {"ch0": "a", "ch1": "a", "ch2": "b", "ch3": "b"}

        def classifier(con, items):
            self.assertEqual([it["id"] for it in items], ["ch0", "ch1", "ch2", "ch3"])
            return pred_map

        verdict, lines = labeling.run_gate(
            self.con, "fixture_task", FIXTURE_TASK, classifier, self.labels_path,
            min_labels=2, min_positives=2, labeled_at="T",
        )
        self.assertEqual(verdict["passed"], ["a", "b"])
        self.assertEqual(verdict["source"], "owner")
        self.assertTrue(any("cleared: a, b" in ln for ln in lines))
        # Persisted to ingest_state under eval_gate:<task>.
        from db import get_state

        stored = json.loads(get_state(self.con, gates.state_key("fixture_task")))
        self.assertEqual(stored, verdict)
        # And the consumer helper now clears those classes (ingest_state wins).
        self.assertTrue(gates.eval_gate_passed(self.con, "fixture_task", "a",
                                               seed_dir=Path(self.tmp.name)))

    def test_gate_refuses_when_no_labels(self):
        empty = Path(self.tmp.name) / "empty.jsonl"
        with self.assertRaises(ValueError):
            labeling.run_gate(self.con, "fixture_task", FIXTURE_TASK,
                              lambda con, items: {}, empty,
                              min_labels=2, min_positives=2)


class PrecisionGateFlowTest(unittest.TestCase):
    """End-to-end proof the precision gate can now FAIL closed (finding #2). The
    old single-class label space made precision a structural 1.0 that the floor
    could never fail; the two-class space + degenerate guard make a false-positive
    classifier fail and stay suppressed."""

    def setUp(self):
        self.con = _fixture_con()
        self.addCleanup(self.con.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.labels_path = Path(self.tmp.name) / "precision_task.jsonl"
        # Owner labels over the TWO-class space so a false positive is recordable.
        golds = ["pos", "pos", "pos", "neg"]
        with open(self.labels_path, "w") as f:
            for i, g in enumerate(golds):
                f.write(json.dumps({"id": f"ch{i}", "text": f"t{i}", "gold": g}) + "\n")

    def test_false_positive_fails_gate_and_stays_suppressed(self):
        # Classifier over-fires: predicts pos on the neg-labeled ch3 -> a real
        # false positive -> precision 3/4 = 0.75 < 0.85 -> gate FAILS, consumer
        # stays suppressed.
        pred_map = {"ch0": "pos", "ch1": "pos", "ch2": "pos", "ch3": "pos"}
        verdict, _ = labeling.run_gate(
            self.con, "precision_task", PRECISION_FIXTURE_TASK,
            lambda con, items: pred_map, self.labels_path,
            min_labels=2, min_positives=3, labeled_at="T",
        )
        self.assertEqual(verdict["failed"], ["pos"])
        self.assertEqual(verdict["passed"], [])
        self.assertEqual(verdict["values"]["pos"], 0.75)
        self.assertFalse(gates.eval_gate_passed(self.con, "precision_task", "pos",
                                                seed_dir=Path(self.tmp.name)))

    def test_clean_classifier_passes_when_negatives_present(self):
        # Classifier correct on the negative -> precision 1.0 with a false
        # positive POSSIBLE (gold holds a neg) -> gate clears and persists.
        pred_map = {"ch0": "pos", "ch1": "pos", "ch2": "pos", "ch3": "neg"}
        verdict, _ = labeling.run_gate(
            self.con, "precision_task", PRECISION_FIXTURE_TASK,
            lambda con, items: pred_map, self.labels_path,
            min_labels=2, min_positives=3, labeled_at="T",
        )
        self.assertEqual(verdict["passed"], ["pos"])
        self.assertEqual(verdict["values"]["pos"], 1.0)
        self.assertTrue(gates.eval_gate_passed(self.con, "precision_task", "pos",
                                               seed_dir=Path(self.tmp.name)))


# --- eval_gate_passed 3-step precedence --------------------------------------


class EvalGatePassedPrecedenceTest(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.addCleanup(self.con.close)
        self.con.execute("CREATE TABLE ingest_state(key TEXT PRIMARY KEY, value TEXT)")
        self.con.commit()
        self.empty_seed = tempfile.TemporaryDirectory()
        self.addCleanup(self.empty_seed.cleanup)

    def test_a_seed_only_present_clears(self):
        # No ingest_state key -> the committed conflict seed clears its classes.
        self.assertTrue(gates.eval_gate_passed(self.con, "conflict_labels", "repair"))
        self.assertTrue(gates.eval_gate_passed(self.con, "conflict_labels", "criticism"))
        # A class not in the seed's passed set is not cleared.
        self.assertFalse(gates.eval_gate_passed(self.con, "conflict_labels", "not_a_class"))

    def test_b_ingest_state_overrides_seed(self):
        # Owner relabel: repair NOT passed -> overrides the seed that passed it.
        owner = {"passed": ["criticism"], "failed": ["repair"], "metric": "kappa",
                 "values": {}, "n_labels": 60, "labeled_at": "T", "source": "owner"}
        gates.persist_verdict(self.con, "conflict_labels", owner)
        self.assertFalse(gates.eval_gate_passed(self.con, "conflict_labels", "repair"))
        self.assertTrue(gates.eval_gate_passed(self.con, "conflict_labels", "criticism"))

    def test_c_neither_key_nor_seed_suppresses(self):
        # dropped_plan ships no seed file and has no ingest_state key -> absent.
        self.assertFalse(gates.seed_gate_path("dropped_plan").exists())
        self.assertFalse(gates.eval_gate_passed(self.con, "dropped_plan", "dropped_plan"))
        # An entirely unknown task with an empty seed dir is likewise suppressed.
        self.assertFalse(gates.eval_gate_passed(self.con, "made_up_task", "x",
                                                seed_dir=Path(self.empty_seed.name)))

    def test_missing_ingest_state_table_falls_back_to_seed(self):
        # A fresh/partial DB without ingest_state must not crash; it falls
        # through to the committed seed (survives a reset_db wipe).
        bare = sqlite3.connect(":memory:")
        self.addCleanup(bare.close)
        self.assertTrue(gates.eval_gate_passed(bare, "conflict_labels", "repair"))

    def test_con_none_uses_seed(self):
        self.assertTrue(gates.eval_gate_passed(None, "conflict_labels", "repair"))


# --- committed seed validity + consistency -----------------------------------


class SeedArtifactsTest(unittest.TestCase):
    def test_seed_gate_json_valid_and_matches_labels(self):
        labels_path = EVAL_DIR / "seed_labels" / "conflict_labels.jsonl"
        gate_path = EVAL_DIR / "seed_gates" / "conflict_labels.json"
        self.assertTrue(labels_path.exists())
        self.assertTrue(gate_path.exists())
        committed = json.loads(gate_path.read_text())
        # Recompute from the committed synthetic labels; must match byte-shape.
        recs = labeling.load_labels(labels_path)
        gold = [r["gold"] for r in recs]
        pred = [r["predicted"] for r in recs]
        recomputed = gates.compute_verdict(
            gates.TASKS["conflict_labels"], gold, pred,
            min_labels=50, min_positives=40,
            labeled_at=committed["labeled_at"], source="seed",
        )
        self.assertEqual(committed, recomputed)

    def test_seed_meets_kappa_floor_and_min_labels(self):
        committed = json.loads((EVAL_DIR / "seed_gates" / "conflict_labels.json").read_text())
        self.assertEqual(committed["metric"], "kappa")
        self.assertEqual(committed["source"], "seed")
        self.assertEqual(sorted(committed["passed"]),
                         sorted(gates.TASKS["conflict_labels"]["classes"]))
        for cls, val in committed["values"].items():
            self.assertGreaterEqual(val, 0.6, f"{cls} below kappa floor")
        # >= 50 gold labels per class (the EVAL_GATE_MIN_LABELS floor).
        recs = labeling.load_labels(EVAL_DIR / "seed_labels" / "conflict_labels.jsonl")
        from collections import Counter

        support = Counter(r["gold"] for r in recs)
        for cls in gates.TASKS["conflict_labels"]["classes"]:
            self.assertGreaterEqual(support[cls], 50, f"{cls} has < 50 seed labels")


if __name__ == "__main__":
    unittest.main()
