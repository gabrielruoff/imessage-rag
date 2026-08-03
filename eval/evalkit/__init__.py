"""evalkit -- the analytics-eval framework (rationale: docs/DECISIONS.md, the
2026-07-17 evalkit entry).

Loads test prompts with expected-output success criteria (eval/cases/),
runs them through chat.run_turn, captures full structured transcripts
(eval/results/<run-id>/), checks them deterministically against computed
ground truth (eval/analytics_eval_manifest.json), and aggregates agent-written
grades into a scorecard. Stdlib-only; app modules are imported lazily inside
functions so env vars (TARGET_DB, ...) can be set before config.py loads.
"""

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
CASES_DIR = EVAL_DIR / "cases"
RESULTS_DIR = EVAL_DIR / "results"
MANIFEST_PATH = EVAL_DIR / "analytics_eval_manifest.json"
LOCK_PATH = RESULTS_DIR / ".ollama.lock"
