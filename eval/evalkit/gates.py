"""Eval-gate registry, verdict shape, and the consumer-side clearance helper
(RES-70 / F4).

Three measurement gates share this machinery:

  * ``conflict_labels``   -- Cohen's kappa over conflict-escalation labels
                             (Gottman-style types + ``repair``); floor 0.6.
                             Ships a committed seed verdict, so it is
                             pre-cleared out of the box.
  * ``dropped_plan``      -- binary detector precision; floor 0.85. No seed:
                             suppressed by default until the owner labels it.
  * ``wellbeing_narration`` -- a deterministic transcript check (W3), not a
                             hand-label task; registered only so the scorecard
                             can list it. No labels, no seed.

A verdict (the JSON stored under ``ingest_state`` key ``eval_gate:<task>`` and,
for the seed, in ``eval/seed_gates/<task>.json``) has the exact shape::

    {"passed": [...classes...], "failed": [...], "metric": "kappa|precision",
     "values": {class: value|null}, "n_labels": N, "labeled_at": iso,
     "source": "owner|seed"}

``eval_gate_passed(con, task, cls)`` is the single consumer-side helper. It is
importable without Ollama (app modules are pulled in lazily) and is the only
thing pipeline write-time code should call to decide whether a class is
cleared to ship.
"""

import json
import sqlite3

from . import EVAL_DIR, agreement

# Personal hand labels (gitignored, owner-only, agent-untouchable) vs the
# committed, non-personal seed artifacts.
LABELS_DIR = EVAL_DIR / "labels"
SEED_LABELS_DIR = EVAL_DIR / "seed_labels"
SEED_GATES_DIR = EVAL_DIR / "seed_gates"

# The task registry: metric / floor / classes / sampling source per task.
# ``classifier`` is the production classifier path the consumer pipeline wires
# when it lands (None = not yet); ``sample_sql`` is the ``label`` command's
# candidate-sampling query (one bound ``?`` = the LIMIT). ``seed`` marks the
# one task that ships a committed pre-clearing verdict.
TASKS = {
    "conflict_labels": {
        "metric": "kappa",
        "floor": 0.6,
        "classes": ("criticism", "contempt", "defensiveness", "stonewalling", "repair"),
        "sample_sql": (
            "SELECT c.id AS id, c.text AS text FROM chunks c "
            "WHERE c.n_messages >= 4 ORDER BY RANDOM() LIMIT ?"
        ),
        "seed": True,
        # RES-46 / C1: the conflict consumer pipeline landed. A dotted-path
        # STRING (not a callable) so importing gates stays Ollama-free --
        # labeling._resolve_classifier imports it lazily only on `gate`.
        "classifier": "conflict_detect.gate_classifier",
        "description": "conflict-escalation labeling (Gottman-style types + repair)",
    },
    "dropped_plan": {
        "metric": "precision",
        "floor": 0.85,
        "classes": ("dropped_plan",),
        "sample_sql": (
            "SELECT c.id AS id, c.text AS text FROM chunks c "
            "WHERE c.n_messages >= 3 ORDER BY RANDOM() LIMIT ?"
        ),
        "seed": False,
        "classifier": None,
        "description": "dropped-plan detector precision (binary)",
    },
    "wellbeing_narration": {
        "metric": "transcript_check",
        "floor": None,
        "classes": ("wellbeing_narration",),
        "sample_sql": None,
        "seed": False,
        "classifier": None,
        "description": "W3 wellbeing-narration compliance (deterministic transcript check)",
    },
}

# Metrics that have a hand-label gate (the others, e.g. transcript_check, are
# scored elsewhere and only listed here for the scorecard).
LABEL_METRICS = ("kappa", "precision")


def state_key(task):
    return f"eval_gate:{task}"


def seed_gate_path(task, seed_dir=None):
    return (seed_dir or SEED_GATES_DIR) / f"{task}.json"


def compute_verdict(task_cfg, gold, predicted, *, min_labels, min_positives,
                    labeled_at, source):
    """Build a verdict dict from paired gold/predicted labels per the task's
    metric. A class is ``passed`` only when its metric is defined, its sample
    is at/above the floor count, and its value meets the task floor -- an
    under-sampled or below-floor class lands in ``failed`` (its ``values``
    entry is the rounded metric, or null when undefined). ``values`` are
    rounded to 4 places so the stored/committed JSON is byte-stable.
    """
    metric = task_cfg["metric"]
    if metric not in LABEL_METRICS:
        raise ValueError(f"task metric {metric!r} is not hand-label gated")
    classes = list(task_cfg["classes"])
    floor = task_cfg["floor"]
    values, passed, failed = {}, [], []

    if metric == "kappa":
        per = agreement.kappa_per_class(
            gold, predicted, classes=classes, min_labels=min_labels)
        for c in classes:
            k = per.get(c)
            if k is None:  # undefined or below the per-class sample floor
                values[c] = None
                failed.append(c)
            else:
                values[c] = round(k, 4)
                (passed if k >= floor else failed).append(c)
    else:  # precision
        for c in classes:
            pr = agreement.precision_recall(gold, predicted, positive=c)
            prec, support = pr["precision"], pr["support"]
            if prec is None or support < min_positives:
                values[c] = None if prec is None else round(prec, 4)
                failed.append(c)
            else:
                values[c] = round(prec, 4)
                (passed if prec >= floor else failed).append(c)

    return {
        "passed": sorted(passed),
        "failed": sorted(failed),
        "metric": metric,
        "values": values,
        "n_labels": len(gold),
        "labeled_at": labeled_at,
        "source": source,
    }


def persist_verdict(con, task, verdict):
    """Write a verdict to ``ingest_state`` under ``eval_gate:<task>`` (source
    "owner" -- an owner relabel always wins over the seed at read time)."""
    from db import set_state

    set_state(con, state_key(task), json.dumps(verdict, sort_keys=True))
    con.commit()


def load_verdict(con, task, *, seed_dir=None):
    """The 3-step precedence read (returns the winning verdict dict, or None):

      1. ``ingest_state`` key ``eval_gate:<task>`` -- an owner relabel; wins.
      2. else a committed ``eval/seed_gates/<task>.json`` -- the shipped seed,
         so conflict stays pre-cleared across a reset_db / fresh-DB wipe of
         ingest_state (the seed lives in the repo, not the per-user DB).
      3. else None -- nothing cleared.
    """
    raw = None
    if con is not None:
        try:
            from db import get_state

            raw = get_state(con, state_key(task))
        except sqlite3.OperationalError:
            raw = None  # no ingest_state table yet -> treat as absent
    if raw:
        return json.loads(raw)
    seed = seed_gate_path(task, seed_dir)
    if seed.exists():
        return json.loads(seed.read_text())
    return None


def eval_gate_passed(con, task, cls, *, seed_dir=None):
    """True iff class ``cls`` of ``task`` is cleared to ship. The single
    consumer-side helper: pipeline write-time code calls this and SUPPRESSES
    any class it returns False for. Absent verdict (neither ingest_state key
    nor seed file) => suppressed (Master Spec A2)."""
    verdict = load_verdict(con, task, seed_dir=seed_dir)
    if not verdict:
        return False
    return cls in (verdict.get("passed") or [])
