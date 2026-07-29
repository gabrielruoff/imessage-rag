# PLAN: Conflict labeling under a wall-clock budget (hybrid tier for conflict_detect.py)

Status: IMPLEMENTED (commit `8c3f548`). Kept as design history — but read
`docs/DECISIONS.md` (2026-07-28 conflict-budget entry) for what actually
shipped, because implementation reversed or invalidated six of this document's
instructions:

- **§2c "`detect_chat`'s delete-then-rewrite naturally clears it [`labeled_at`]
  when a chat re-detects"** (lines ~175–176) — the code does the OPPOSITE, and
  deliberately. `detect_chat` RECONCILES a window in place, keyed on
  `chat_id` + `first_date` + `last_date`: an unchanged window keeps its row id,
  its `conflict_signals`, its topic and its `labeled_at` stamp, and only its
  deterministic columns are refreshed; windows that no longer exist are still
  dropped with their signals. This is the single most load-bearing deviation,
  because every budgeted run re-detects all 787 chats *before* it labels
  anything — under a delete-then-rewrite each run would throw away the labels
  the previous run's budget bought, and no rerun could ever extend coverage.
- **Phase 3, test 4: "cleared by `detect_chat`'s rewrite"** (lines ~211–212) —
  the same reversal. The shipped test asserts the opposite: re-detection
  PRESERVES an unchanged window's `labeled_at`.
- **Guardrails: "`--budget-minutes 0` must reproduce today's unbounded labeling
  path"** (lines ~296–297) — it does not, and should not. Labeling is resumable
  now, so `--budget-minutes 0` labels every *still-unlabeled* window; a second
  unbudgeted run labels only what the first left behind rather than re-labeling
  the corpus. `0` means "no deadline", not "no memory".
- **`tests/test_conflict_detect.py`** (lines ~201, ~215–216) — no such file.
  The conflict tests live in `tests/test_conflict.py`.
- **Phase 1's "add `CONFLICT_THINK=false` to the worktree `.env`"** (lines
  ~75–88) — redundant, and now actively counterproductive.
  `conflict_detect._resolve_think` resolves an UNSET `CONFLICT_THINK` to
  `False` for this pipeline; putting a value in `.env` converts the unset case
  the code handles into an explicit one that passes through untouched, so the
  resolver stops applying to it.
- **The scratchpad DB path** (lines ~62–63) — an absolute path into a different
  session's scratchpad directory. Stale and environment-specific: make your own
  copy rather than expecting that one.

Instructions for the implementing agent. Work happens in THIS worktree
(`.claude/worktrees/annotate-sweet-spot`), which already ships the annotate
hybrid tier this plan mirrors. Read `docs/PLAN_ANNOTATE_HYBRID.md` and
`annotate_hybrid.py` before writing code — the deadline-gating philosophy,
rate-banking, and "budget in TIME not items" contract are the same; only the
unit of work differs (one single-call turn classification instead of a batched
chunk group).

## Mission

Two deliverables, in order:

1. **Unblock the feature today (zero code)**: run the existing budgeted path
   with the think footgun disabled so `conflicts`/`conflict_signals` get real
   data in ~22 minutes.
2. **Make the 30-minute promise structural**: implement a wall-clock-budgeted
   labeling tier in `conflict_detect.py` (strongest-first, deadline-gated,
   resumable), folding in a durable fix for the think footgun and smarter
   turn selection.

**Hard success criterion**: a bare `make conflict` (equivalently
`python3 conflict_detect.py` with no flags) completes end-to-end in **under 30
minutes** on qwen3.5:4b on this machine, with enough windows labeled to make
the conflict read tools useful. The promise must hold **by construction**
(deadline gate), not by tuned constants that break on a bigger corpus.

## Measured facts — do NOT re-benchmark these

All measured 2026-07-28 on this machine, this corpus (350,216 messages, 787
direct chats, 20,298 sessions), qwen3.5:4b via Ollama, using the real
`_classify_turn` path against a scratchpad copy of the DB:

| Fact | Value |
|---|---|
| Deterministic detection pass (all 787 chats, `label=False`) | **24.1 s**, writes **1,305 windows** |
| Candidate turns at the current `max_label_turns=12` cap | **6,670** LLM calls |
| Candidate turns uncapped | 47,457 (min 1 / median **4** / mean 36.4 / max **1,940** per window) |
| Signal-count distribution | 2 signals: 920 windows, 3: 384, 4: 1 |
| Corroboration distribution | none: 1,186, weak: 95, **llm: 24** |
| Per-call latency, `think` flag OMITTED (current behavior) | **197 s** (one sample; user-observed average ~36 s ⇒ the 66-hour run) |
| Per-call latency, `think=False` | **2.2 s mean / 2.0 s median** (n=24, real turns, median 30 chars) |
| Model cold-load (first request) | ~54 s |
| Batched 12-turn requests (full schema) | **2.25 s/turn — no speedup** (output-token-bound), 4/8 responses unparseable (missing entries, invented classes). **Batching is a measured dead end; do not retry it.** |
| Parallel workers | Ollama serializes; ~20% gain (annotate hybrid measurement). Not worth it here. |
| Budget math | 30-min wall clock ≈ **~680 calls** at 2.2 s after detect + load overhead |

Root cause of the 66 hours: `CONFLICT_THINK` inherits `CHAT_THINK`, which
defaults to `None` (= omit the flag) — so qwen3.5:4b, a thinking model, runs a
full reasoning pass on all 6,670 calls. `think=False` alone is a ~16–90x
speedup but still 4.1 h total; the call count must ALSO be bounded.

## Environment

- Worktree: `/Users/gabrielruoff/LocalDocuments/GitHub/imessage-rag/.claude/worktrees/annotate-sweet-spot`
- Python: `.venv/bin/python3` **in the worktree** (system/brew python lacks deps).
- The worktree has its own `.env` (sets `CHAT_MODEL=qwen3.5:4b`) and its own
  `imessage_analysis.db` — config paths are pinned to `config.py`'s directory,
  so nothing here touches the parent checkout's DB.
- Ollama is up; `qwen3.5:4b` is pulled.
- A 238 MB scratchpad DB copy already exists for experiments:
  `/private/tmp/claude-501/-Users-gabrielruoff-LocalDocuments-GitHub-imessage-rag/4a3b4c2d-5bc3-4b91-87a6-4005d31479ea/scratchpad/analysis_copy.db`
  (it already contains the 1,305 deterministic windows). Test destructive or
  slow things against a copy like this by setting
  `conflict_detect.DB_PATH = <copy>` — never against the worktree DB while
  another run holds its write lock.

## Phase 1 — the zero-code ops run (start this FIRST, in the background)

This uses only code that already exists. Kick it off immediately; it runs
~22 minutes while you implement Phase 2 (it holds the worktree DB's write lock
and the serialized Ollama — schedule around that, see the parallelization map).

1. Add to the **worktree** `.env`:
   ```
   CONFLICT_THINK=false
   ```
2. Run in the background (from the worktree root):
   ```
   .venv/bin/python3 conflict_detect.py --label-top 90 --max-label-turns 6
   ```
   Expected: 24 s detect (1,305 windows) + 540 calls × ~2.2 s ≈ 20 min
   labeling + ~1 min model load ⇒ **~22 min end-to-end**. It records
   `conflict_model=qwen3.5:4b` in `ingest_state` (the guard — every later run
   must use the same model or clear the table).
3. Don't wait on it — proceed to Phase 2. Check it completed before Phase 5
   (it is Ctrl-C safe and resumable if anything goes wrong).

Phase 5's verification run will re-detect and re-label idempotently on the new
code path, so Phase 1's output being superseded is expected and fine.

## Phase 2 — implementation

All edits in `conflict_detect.py`, `config.py`, `.env.example`, `schema.py`,
`Makefile`, plus tests and docs (Phases 3–4). Keep the repo's conventions: all
tunables in `config.py`/`.env.example`, never hardcoded; comment style matches
the file.

### 2a. Durable think fix (strategy #3)

- In `conflict_detect.main` (or a small helper beside it), resolve the
  effective think value the way `annotate._resolve_quality` does: if
  `CONFLICT_THINK` is `None` (i.e. genuinely unset — no explicit env) **and**
  the model has no effort levels (`config.has_effort_levels(model)` is False),
  use `False`; on an effort-level model (gpt-oss family) use `'low'`. An
  explicit `CONFLICT_THINK` env always wins and passes through untouched.
- Respect the tri-state gotcha: the point is that the *pipeline default*
  becomes `False`, not that `None` stops meaning "omit the flag" at the
  `ollama.py` seam. Do not change `config._think` semantics or `CHAT_THINK`.
- Thread the resolved value through `detect_conflicts`, `label_top_conflicts`,
  and `gate_classifier` — today each closes over `CONFLICT_THINK` directly;
  they must all use the resolved value or the gate path stays 90x slow.

### 2b. Smarter turn selection (strategy #4)

Replace the current "first `max_label_turns` chronological turns" slice
(`detect_chat` and `label_top_conflicts` both do `[:max_label_turns]`) with a
single shared helper, e.g. `_select_turns(seg, max_turns, peak_id)`:

- **Peak-anchored**: if the window's `peak_gap_message_id` is present in the
  segment, center the selection on it (the message left hanging is the
  escalation moment). Take the turns in a contiguous span around it; fall back
  to the chronological head when there is no peak id. In a 1,940-turn window,
  first-12 provably misses the evidence.
- **Lexical prescreen**: before capping, drop turns that cannot carry an
  escalation signal — text under ~15 chars after stripping whitespace, or
  emoji/punctuation-only. Measured: ~70% of labels come back `neutral`, each
  costing 2.2 s to say nothing; the median window has only 4 candidate turns,
  so the prescreen mostly trims the big windows where it matters. Make the
  threshold a `config.py` knob (`CONFLICT_MIN_TURN_CHARS`, default 15, in
  `.env.example` with the other `CONFLICT_*` knobs).
- Keep selected turns in chronological order in the output (the labels read as
  a sequence; do not emit them peak-outward).
- Lower the `CONFLICT_MAX_LABEL_TURNS` default 12 → **6** in `config.py` and
  `.env.example`. With peak anchoring, 6 turns lose almost nothing.

### 2c. The wall-clock budget tier (strategy #5)

New default behavior for a bare run: **two-phase** — deterministic detection
over ALL chats (24 s, unchanged), then label windows **strongest-first** under
a wall-clock deadline. Mechanics:

- New knob `CONFLICT_BUDGET_MINUTES` in `config.py` (+`.env.example`),
  default **25** (leaves headroom under the 30-min promise for detect, model
  cold-load ~54 s, and the in-flight call that straddles the deadline).
  `0` = unbudgeted legacy behavior.
- New CLI flag `--budget-minutes N` overriding the env (same precedence
  convention as `annotate_hybrid._resolve_budget_minutes`: caller > env >
  default). It composes with `--label-top N` (both caps apply); it is
  meaningless with `--no-llm` (reject or ignore explicitly — pick one and
  document it).
- Reuse `label_top_conflicts`'s existing ranking (`_conflict_strength`) — the
  budget tier is essentially `label_top_conflicts` with the deadline gate and
  no fixed N. Refactor rather than duplicate: one labeling loop serving both
  the fixed-N and budgeted paths.
- **Deadline gate**: start the clock at the top of `main`. Before each LLM
  call, check `remaining > estimated_call_seconds`; when it isn't, stop
  dispatching, commit, and print how many windows were labeled vs left (mirror
  `annotate_hybrid`'s "candidates left; rerun to resume" summary line — say
  what was NOT covered; no silent truncation).
- **Rate estimator**: bank the observed per-call seconds in `ingest_state`
  under a key like `conflict_turn_seconds:<model>` (annotate hybrid precedent:
  `annotate_group_seconds:...`). Prime the first run's estimate from the
  banked value; with nothing banked, treat the first call as the probe (it
  includes the ~54 s cold load — either special-case it or use a robust
  estimate like max-seen, as annotate does). Do not import annotate's
  `_DispatchDeadline` — it is built for pooled batch dispatch; a sequential
  single-call loop needs only a few lines.
- **Resumability**: add a nullable `labeled_at` column to `conflicts` via an
  idempotent migration in `schema.create_schema` (the repo already has this
  pattern; `annotate._ensure_column` is the reference). The labeling loop
  skips windows with `labeled_at` set and stamps it after each window's
  labels commit (per-window commit, Ctrl-C safe — keep the existing
  KeyboardInterrupt contract). `detect_chat`'s delete-then-rewrite naturally
  clears it when a chat re-detects. "Zero survivors" still stamps
  `labeled_at` — the window was processed; presence of `conflict_signals`
  rows is NOT the marker.
- A rerun therefore continues down the strength ranking from wherever the
  last budget ran out — each 25-minute run extends coverage.
- `make conflict` needs no Makefile change beyond help text: the default IS
  the budgeted path now. Update the target's comment/echo if it describes the
  old behavior.

### 2d. Schema + docs bookkeeping

- `schema.py`: the `labeled_at` migration (idempotent, safe on a fresh DB and
  on the scratchpad copy).
- `CLAUDE.md` (worktree): update the `make conflict` command bullet (budgeted
  default, the new flags/knobs), the `conflicts` schema block comment
  (`labeled_at`), and the `ingest_state` inventory
  (`conflict_turn_seconds:<model>`). Mention the resolved-think behavior in
  the think gotcha bullet the way the annotate carve-out is mentioned.
- `.env.example`: `CONFLICT_BUDGET_MINUTES`, `CONFLICT_MIN_TURN_CHARS`, the
  new `CONFLICT_MAX_LABEL_TURNS` default, and a one-line warning that an
  explicit `CONFLICT_THINK=true` on qwen-class models costs ~90x.

## Phase 3 — tests

Hermetic (no Ollama, stub `classify_fn` — the injection point already exists).
Extend `tests/test_conflict_detect.py` (touching `tests/` is in scope for this
task). Cover at minimum:

1. Think resolution: unset → `False` on a non-effort model, `'low'` on
   gpt-oss, explicit env passes through (patch `config` attributes).
2. `_select_turns`: peak-anchored span selection, chronological output order,
   prescreen drops short/emoji turns, fallback with no peak id, cap respected.
3. Deadline gate with a fake clock: stops dispatching when the estimate
   exceeds the remainder, commits what ran, prints the left-behind count;
   budget 0 = unbounded.
4. Resume: `labeled_at` stamped (including zero-survivor windows), stamped
   windows skipped on rerun, cleared by `detect_chat`'s rewrite.
5. Rate banking: value persisted to `ingest_state`, primed on next run.

Run only the touched module (`.venv/bin/python3 -m unittest
tests.test_conflict_detect`), never the full discovery sweep.

## Phase 4 — verification (the success criterion, measured honestly)

1. Confirm the Phase 1 background run finished cleanly.
2. `make lint` in the worktree.
3. Unit tests for the touched module (Phase 3).
4. **The real run**: from the worktree, time a bare
   `.venv/bin/python3 conflict_detect.py` (no flags) against the worktree DB,
   in the background (`run_in_background` — it will take ~25–28 min and must
   not block your loop; poll it). Pass criteria:
   - wall clock **< 30 min**,
   - the summary reports the deadline honored and states how many windows
     were labeled vs remaining,
   - `SELECT COUNT(*) FROM conflicts WHERE labeled_at IS NOT NULL` is
     comfortably three digits (~150–250 expected with 6-turn peak-anchored
     selection at 2.2 s/call),
   - a second short run (e.g. `--budget-minutes 3`) labels only NEW windows
     (resume works, no re-labeling).
5. Report the measured wall clock and coverage numbers in your summary — no
   hedging; if it misses 30 min, say so and fix the budget default, don't
   reframe.

Commit to the worktree branch when green (conventional message, the repo's
co-author trailer). Do not push or open a PR unless the owner asks.

## Parallelization map — where to fan out

The core code change is one file (`conflict_detect.py`) plus small satellites,
so **don't parallelize the core editing** — two agents in one file buys merge
conflicts, not speed. The wall-clock wins are elsewhere:

1. **T+0, background Bash**: launch the Phase 1 ops run immediately. It's
   ~22 min of pure waiting that overlaps all of Phase 2. (It holds the
   worktree DB write lock and the serialized Ollama — so nothing else may
   write that DB or lean on Ollama until it exits; hermetic tests and code
   editing are unaffected.)
2. **T+0, parallel with Phase 2 start — one Explore agent**: inventory
   `tests/test_conflict_detect.py` (current fixtures, how `classify_fn` is
   stubbed, what `detect_chat` cases exist) and how `annotate_hybrid`'s tests
   exercise its deadline gate, returning a short map of reusable fixtures.
   Read-only, so it can't collide with anything.
3. **After 2a–2c interfaces are frozen (signatures + knob names decided) —
   two parallel agents**, disjoint files, safe concurrently:
   - a test-writer agent for Phase 3 (touches only `tests/`),
   - a docs agent for Phase 2d's `CLAUDE.md` / `.env.example` text (touches
     only docs/config examples — hand it the exact knob names and defaults).
   Meanwhile you finish the `conflict_detect.py`/`schema.py` work yourself.
4. **After implementation, before the verification run — a review workflow**
   (the user has sanctioned workflow use for this task): a find→verify shape
   over the diff, e.g. 3 finder dimensions (correctness of the deadline/resume
   logic, schema-migration safety on old DBs, repo-convention drift:
   hardcoded tunables / think tri-state / CLAUDE.md sync) each feeding an
   adversarial verifier that must reproduce the claim against the actual diff
   before it's reported. Apply confirmed findings, then run Phase 4.
5. **Phase 4's timed run is background**: while it ticks, fold in review
   findings' doc fixes and prepare the commit; only the final pass/fail
   assertion waits on it.

Strictly serial spine: Phase 1 launch → core implementation (2a→2c share the
labeling loop; do 2b before 2c so the budget tier is built on the final turn
selector) → interface freeze → fan-out (tests ∥ docs) → review workflow →
timed verification → commit.

## Guardrails

- **Never** re-attempt request batching or thinking-on labeling "to check" —
  both are measured above; a single thinking call can burn 3+ minutes.
- No foreground command over 10 minutes; anything longer runs in background.
- Don't run two writers against one SQLite DB (Phase 1 run vs experiments:
  use the scratchpad copy; Phase 4 run vs anything else: sequence it).
- The `conflict_model` guard: every labeling run here uses qwen3.5:4b; if you
  see the mixed-model refusal, you pointed a run at the wrong DB — stop and
  check, don't clear the table.
- Turn *content* is sensitive personal data: don't print message text or
  excerpts into logs/summaries beyond what the existing code already stores;
  labels and counts are fine.
- Don't touch the parent checkout (`imessage-rag/` root) — all work lands in
  this worktree.
- Keep `--no-llm` and `--label-top` working exactly as documented; the budget
  tier is additive, and `--budget-minutes 0` must reproduce today's unbounded
  labeling path (post think-fix speed).
