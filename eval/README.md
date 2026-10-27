# evalkit — usage and agent orchestration guide

A regression harness for the analytics tools: it sweeps 71 prompt cases
across 9 suites (S1 activity patterns, S2 response dynamics, S3 reactions,
S4 words/language, S5 milestones, S6 group dynamics, S7 routing/negative
controls, S8 multi-turn robustness, S9 retrieval/directory plumbing) through
`chat.py`'s tool-calling loop, compares replies against SQL-computed ground
truth, and aggregates grades into a scorecard. Mechanizes [`docs/ANALYTICS_TEST_PLAN.md`](../docs/ANALYTICS_TEST_PLAN.md)
(design rationale: [`docs/DECISIONS.md`](../docs/DECISIONS.md), the 2026-07-17
evalkit entry).
Everything below is copy-pasteable. `fast_eval.sh` / `prompts-smoke.txt` are a
separate, older path — leave them alone.

## Paths (worktrees have no venv and only a 0-byte DB stub)

Always run from the repo/worktree **root**. Worktrees share the main
checkout's venv and real DB, so the commands below spell out both paths:

- Python: `.venv/bin/python3`
- Real DB: `imessage_analysis.db`

Commands that need the DB refuse (exit 2) when it is missing or 0 bytes —
always pass `--db` with the real DB path.

## Progress

The slow commands report progress through the same `progress.track` helper the
pipeline scripts use, so `PROGRESS_STYLE` governs them identically: a live bar
with an ETA on a terminal, `  ...N/Total (X%)` lines as items complete when
stdout is piped or redirected (no escape codes in a captured sweep log), and
nothing at all under `PROGRESS_STYLE=off`. Tracked loops: `run` (per case),
`manifest` (per ground-truth recipe), and the two `p3 --judge` loops (per
sample). Per-case status lines still print above the bar.

Pass `--log-file` to `run` to also stream the full per-case detail — each
resolved prompt, every `[tool call] …` line, and the model's reply (chat.py's
own console output) — to a log file while the progress bar keeps the terminal.
Bare `--log-file` writes `eval/results/<run-id>/run.log`; `--log-file PATH`
picks the location. Append mode, so `--resume` extends it and `tail -f` follows
the sweep live. Omit the flag and nothing is logged (the default) — the
per-case `<id>.json`/`.txt` transcripts remain the authoritative artifacts.

These loops are tens of items at minutes apiece, so each passes an explicit
`log_every=1` — the repo-wide `PROGRESS_LOG_EVERY` default (200) is tuned for
the pipeline's ~350k-row passes and would emit no plain-mode line at all for a
loop this small. `progress.py`'s `_MAX_PLAIN_LINES` ceiling still caps any loop
at ~50 lines, so a 71-case sweep logs every second case rather than every one;
for the same reason the last line can sit one item short of the total (there is
no final flush on exit) — treat the command's own exit as the completion signal,
not a `100%` line.

`import progress` is deliberately **function-local** in every evalkit module: it
pulls in `config`, which reads env at import, and `__main__` imports every
submodule eagerly — a top-level import would freeze config before `run` applies
`--max-iterations` / `--tool-result-max-chars` and before `--db` sets
`TARGET_DB`.

## The subcommands

| Command | Phase | Ollama? | What it does |
|---|---|---|---|
| `selfcheck` | — | no | validate suite files, recipe registry, tool-list sync with `tool_schemas.py` |
| `manifest` | P0 | no | resolve the roster + compute ground truth into the (gitignored) manifest, stamped with a corpus fingerprint |
| `run` | P1 | **yes** | the sweep: drives `chat.run_turn` in-process, one case at a time, writing JSON transcripts to `eval/results/<run-id>/` |
| `check` | P2a | no | deterministic checkers per transcript: routing verdict + advisory numeric/faithfulness evidence |
| `p3` | P3 | `wrapped`/`--judge` only | generator evals: `wrapped`, `check-wrapped`, `annotate-sample`, `topics-sample` |
| `report` | P4 | no | aggregate grades + checks into `scorecard.json`/`.md`; `--compare` for regression diffs, `--strict` gating |

Hosted-model harnesses (they produced the evidence in `docs/DECISIONS.md`,
"hosted embeddings (Voyage) and hosted annotation (OpenRouter)"). The ones that
upload message text say so and need `--confirm-upload` or a `--budget`:

| Command | Ollama? | What it does |
|---|---|---|
| `recall-labels` | **yes** | the local model writes one synthetic search query per sampled chunk (the gold set) |
| `recall-embed` | no | **uploads** the gold chunks + `--pool` distractors to Voyage (`--dim` is sent as `output_dimension`); `--confirm-upload` |
| `recall` | no | recall@1/@10 and MRR, `EMBED_MODEL` vs the Voyage vectors, over the gold queries |
| `distill-ab` | no | k-fold kNN-label agreement, `EMBED_MODEL` vs Voyage vectors (**uploads**; `--confirm-upload`) |
| `annotate-pilot` | no | **uploads** sampled chunks to OpenRouter and compares `--model` with `--reference` and the local labels; `--budget` is a hard USD cap (reserved per attempt on the full request body; an unpriced model stops the run) |

Voyage vectors are cached in `eval/results/recall/<model>-<db stem>.db`, one per
source DB, and a cache built at another dimension is refused before any upload.
An older untagged `<model>.db` cache is ignored; rename it to keep it.

## Quick verification (no Ollama, no real DB needed for the first two)

```
# hermetic unit tests for the checkers (fast)
.venv/bin/python3 -m unittest tests.test_evalkit_checks

# static validation: suites, recipes, roster placeholders, tool-list drift
.venv/bin/python3 -m eval.evalkit selfcheck

# dry run: print a suite's resolved prompts (real roster names), no Ollama
.venv/bin/python3 -m eval.evalkit run --dry-run --suite S1 --run-id demo --db imessage_analysis.db
```

## Smallest end-to-end demo (one case; Ollama must be running)

```
.venv/bin/python3 -m eval.evalkit manifest --db imessage_analysis.db
.venv/bin/python3 -m eval.evalkit run --run-id demo-single --case S1.1 --db imessage_analysis.db
.venv/bin/python3 -m eval.evalkit check --run demo-single
.venv/bin/python3 -m eval.evalkit report --run demo-single
```

Then inspect `eval/results/demo-single/S1.1.json` (the full transcript),
`S1.1.checks.json`, and `scorecard.md`. The report will flag the case as
ungraded until a grade file exists (see P2b below) — that's expected.

## The Ollama lane (one at a time, no exceptions)

Only `run`, `p3 wrapped`, and `p3 ... --judge` touch Ollama. They serialize
themselves via `eval/results/.ollama.lock` and exit 2 if another holder is
alive. **Never run two Ollama-touching commands concurrently** — Ollama
serializes anyway and parallel prompts thrash the KV prefix cache. Everything
else (`manifest`, `check`, `report`, `p3 check-wrapped`, the samplers without
`--judge`, grading) is parallel-safe file I/O.

## P0 → P4 orchestration (the full eval)

```
# P0 — roster + ground truth (no Ollama; review the printed roster)
.venv/bin/python3 -m eval.evalkit manifest --db imessage_analysis.db
.venv/bin/python3 -m eval.evalkit selfcheck

# P1 — the sweep (ONE agent, Ollama lane)
.venv/bin/python3 -m eval.evalkit run --run-id $(date -u +%Y%m%dT%H%M%S)-baseline --all --db imessage_analysis.db
#   useful: --resume  --case S1.1  --suite S1  --model M  --dry-run
#           --max-iterations N  --tool-result-max-chars N  --case-timeout S
#   manifest also takes --roster-overrides my.json

# P2a — deterministic checks (parallel-safe)
.venv/bin/python3 -m eval.evalkit check --run <run-id>

# P2b — grading: spawn ONE grader agent per suite (S1..S9), file I/O only
# P3 — generator evals (Ollama lane, after/never during the sweep)
.venv/bin/python3 -m eval.evalkit p3 wrapped --run <run-id> --db imessage_analysis.db
.venv/bin/python3 -m eval.evalkit p3 check-wrapped --run <run-id>
.venv/bin/python3 -m eval.evalkit p3 annotate-sample --run <run-id> --db imessage_analysis.db   # exits 2 if no annotate run
.venv/bin/python3 -m eval.evalkit p3 topics-sample --run <run-id> --db imessage_analysis.db     # exits 2 if no topics run

# P4 — scorecard (+ synthesis agent reads scorecard.md)
.venv/bin/python3 -m eval.evalkit report --run <run-id> [--compare <prev-run-id>] [--strict]
```

## Grader agents (P2b): what to read, what to write

Read, per case id in your suite: `eval/results/<run>/<id>.json` (full
transcript), `<id>.checks.json` (deterministic evidence — routing is the only
auto verdict; numeric/faithfulness blocks are advisory), and the manifest's
`gt.<id>` entry. Write `eval/results/<run>/<id>.grade.json`:

```json
{
  "id": "S1.3", "run_id": "…", "suite": "S1", "grader": "agent:S1",
  "axes": {"routing": 1, "accuracy": 0, "faithfulness": 1, "communication": 1},
  "evidence": [{"axis": "accuracy", "note": "reply says March=385, GT=400", "ref": "messages[7]"}],
  "levers": ["shape"],
  "blocked": false, "blocked_reason": null, "notes": "…"
}
```

Rules:
- Axes are 0/1 (see the test plan's four definitions). **Every 0 needs ≥ 1
  evidence entry and ≥ 1 lever** from:
  `desc steer args shape size confab pipeline judge-batch`.
- Faithfulness is the confabulation guard — grade it strictly; the
  `unsupported_numbers` block lists candidates but has false positives
  (derived sums, rounded figures): verify against the transcript.
- A failure caused by a code bug (not model behavior) is `"blocked": true`
  with `blocked_reason` — blocked cases are excluded from suite means.
- A `banned_anywhere` flag in `run.json` means routing 0 for the whole run
  (the test plan's rule) — note it in every affected grade.
- `report` validates every grade file (string `id`, all four axes 0/1);
  malformed ones are skipped with a warning, and any malformed or missing
  grade blocks `overall_pass` — grade every transcript in your suite.

## Ground rules

- **Fingerprint**: GT and transcripts must come from the same corpus state;
  `check` refuses on mismatch (a new ingest invalidates the sweep — rebuild
  the manifest and re-run).
- **Privacy**: transcripts, the manifest, and scorecards carry real
  names/messages. `eval/results/` and `analytics_eval_manifest*.json` are
  gitignored — never commit them, never copy their contents into commits.
- **Read-only DB**: nothing here runs ingest; graders and GT never call
  Ollama.
- **Re-run after tuning**: change one lever, then
  `run --suite Sx --run-id <new> --db <real DB>` → `check` → grade →
  `report --run <new> --compare <prev>`. Keep every run dir — this is a
  regression harness, not a one-shot.

## Gates (measurement gates — hand labels + κ/precision)

`label` and `gate` are the shared eval-gate harness (RES-70 / F4): the κ/
precision machinery the project's three measurement gates consume. A gate
decides, per class, whether that class is *cleared to ship*; consumer pipeline
code calls `eval_gate_passed(con, task, cls)` at write time and **suppresses**
any class it returns `False` for (absent verdict ⇒ suppressed).

**Tasks** (`eval/evalkit/gates.py` `TASKS`): `conflict_labels` (κ over
Gottman-style escalation types + `repair`, floor 0.6, ships a seed),
`dropped_plan` (binary detector precision, floor 0.85, no seed → suppressed
until labeled), and `wellbeing_narration` (a deterministic transcript check
listed only for the scorecard — no hand labels, no seed).

**Seed labels vs personal labels — the load-bearing distinction:**

- `eval/labels/<task>.jsonl` — the **owner's personal hand labels**. May carry
  real corpus text, so the directory is **gitignored** (`eval/labels/.gitignore`
  ignores everything but itself). Written only by the owner running `label`;
  **agents never write here**.
- `eval/seed_labels/<task>.jsonl` + `eval/seed_gates/<task>.json` — **committed,
  non-personal** synthetic/illustrative seed labels and the verdict computed
  from them. Committed because they contain no owner corpus data. Only
  `conflict_labels` ships a seed, which pre-clears its classes out of the box.

**Precedence** (`eval_gate_passed`, in `eval/evalkit/gates.py`): (1) an
`ingest_state` key `eval_gate:<task>` — an owner relabel, always wins; else
(2) a committed `eval/seed_gates/<task>.json` — so conflict stays pre-cleared
across a `reset_db` / fresh-DB wipe of `ingest_state` (the seed lives in the
repo, not the per-user DB); else (3) nothing cleared → suppressed.

```
# owner-run, interactive, Ollama-free — writes eval/labels/<task>.jsonl
.venv/bin/python3 -m eval.evalkit label conflict_labels -n 60 --db imessage_analysis.db
# score labels vs the model side + persist the verdict to ingest_state
.venv/bin/python3 -m eval.evalkit gate conflict_labels --db imessage_analysis.db
```

Sample-size floors are pinned in `config.py`: `EVAL_GATE_MIN_LABELS` (50 gold
labels per class for κ) and `EVAL_GATE_MIN_PRECISION_POSITIVES` (40 gold
positives for precision). Below the floor a class can never be marked passed.
`gate`'s model side reuses the task's registered production classifier; those
land with their consumer pipelines (registry `classifier` is `None` until
then).
