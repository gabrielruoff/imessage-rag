# Implementation plan: `--quality hybrid` — a wall-clock-budgeted annotate tier

Status: IMPLEMENTED. Kept as design history — but read `docs/DECISIONS.md`
(2026-07-27 hybrid entry) for what actually shipped, because measurement during
implementation refuted four of this document's premises:

- **§3 "Phase 1 and 2 … are cheap"** — false. The seed pass measured 641.9 s
  (10.7 min) for 350 seeds, one chunk per request; an 8-min budget produced an
  11.0-min run with zero LLM upgrades. Seeding is now batched AND capped at a
  share of the budget, and every LLM phase is deadline-gated.
- **§3 "bounded overshoot ≈ one group ≈ tens of seconds"** — false.
  `annotate._pooled` holds 2x `ANNOTATE_WORKERS` groups; measured +1.5 to
  +3.7 min. The gate now reserves a measured drain instead.
- **§2 `_APPLE_ANNOTATE_CHAR_BUDGET` 12000 → 8000** — did not fix the overflow
  it was chosen for (100% batched-call failure at 8,000). It is 2,000, and the
  dominant apple failure turned out to be malformed JSON, not overflow.
- **§1 "Stratified … → higher kNN fidelity corpus-wide"** — not supported.
  Full-corpus probing measured coverage as a wash (1.4% *worse* at k=10). The
  real wins are ~4x fewer near-duplicate seeds and deterministic proportional
  chat representation.

Two estimates were also off: the event prefilter yields ~20% of in-scope
chunks, not §3's 5–10%, and a 25-min run buys ~350–800 upgrades, not §2's
800–1,200.
Audience: implementation agents working in parallel. Read this whole document
before touching code; the **Contracts** section is binding — it is what makes
parallel work safe.

## 1. Goal

A new `annotate.py --quality hybrid` tier sitting between `distilled` (~90 s,
kNN everywhere, no events) and `fast` (~16 h, LLM everywhere). Success criteria
(hard):

- Completes in **< 30 minutes on `qwen3.5:4b`** (Ollama) and **< 1 hour on
  Apple FM** (`apple` sentinel), on this machine, warm model.
- Meets the criterion **by construction**: the tier runs against a wall-clock
  deadline (`ANNOTATE_BUDGET_MINUTES`), not a fixed chunk count, so it holds on
  any hardware/corpus size and degrades to *more coverage* on faster machines,
  never a blown deadline.

What the budget buys, in priority order:

1. **Stratified, embedding-diverse seed pool** (better than `distilled`'s
   random 450 seeds) → higher kNN fidelity corpus-wide.
2. **kNN classification of the whole corpus** (same as distilled, ~90 s), now
   also emitting a per-row **confidence score**.
3. **Targeted LLM upgrades** with the remaining time, alternating:
   - *event-candidate chunks* (lexicon-prefiltered; events are the thing kNN
     structurally cannot produce — `source='distilled'` rows always have
     `events_json='[]'`), then
   - *lowest-kNN-confidence chunks* (spend LLM time where kNN is most likely
     wrong).

   Upgraded rows are written `source='llm'`, so every valence-trusting reader
   (conflict corroboration, nudge Stage-0 exclusion, voiceprint tone) lights up
   for exactly the chunks that matter.

## 2. Measured facts (2026-07-27, this machine — size all defaults from these)

Corpus: 25,029 chunks, avg 693 chars (median 293); top-20 dyads = 19,483 chunks.
Existing distilled runs exist in all three DBs (~450 LLM seeds each).

| Config | s/chunk (warm) |
|---|---|
| qwen3.5:4b, batch 8, `think=False` | 1.1–1.4 |
| qwen3.5:4b, batch 16/24, `think=False` | 1.9–2.6 |
| qwen3.5:4b, batch 8, `think='low'` | **TIMES OUT** — `'low'` is truthy to qwen3.5 → full reasoning pass |
| qwen3.5:4b, 4 workers × batch 8 | 2.4 — Ollama serializes; workers ≈ no gain |
| Apple FM, char-budget batch (8–10 chunks) | 2.6–5.6; the current 12,000-char budget **overflowed the 4,096-token window** on 3/18 chunks, forcing single-chunk fallback retries |

Planning numbers: sustained ~1.5–2.5 s/chunk on qwen3.5:4b, ~3–5 s/chunk on
Apple FM. Budget of 25 min (qwen) / 50 min (apple) ⇒ roughly **800–1,200 LLM
chunks per run** on either backend. Cold model load can eat minutes — the
deadline covers it automatically (fewer upgrades that run).

Consequences baked into this plan:

- The hybrid preset uses **`think=False`**, not `'low'` (unlike
  `fast`/`balanced`). `ANNOTATE_THINK` still overrides, as for every tier.
- Batch size stays **8** (larger batches measured *worse* per-chunk).
- `_APPLE_ANNOTATE_CHAR_BUDGET` drops **12000 → 8000** (measured overflow at
  ~10k chars of chunk text; overflow costs a silent drop + per-chunk retries).
  This constant is shared by all tiers — the change is global and strictly
  reduces overflow fallbacks.
- Workers: keep the existing `ANNOTATE_WORKERS` plumbing (a parallel Ollama
  benefits), but nothing in this plan depends on parallelism.

## 3. Design

### New module `annotate_hybrid.py` (mirrors `annotate_distill.py`'s shape)

`annotate.py` owns the CLI and dispatches `quality == "hybrid"` to
`annotate_hybrid.hybrid_chunks(...)`, exactly as it dispatches `distilled` to
`annotate_distill.distill_chunks(...)`. Shared request/validation/write helpers
stay in `annotate` and are resolved late (module-attribute access), same as
`annotate_distill` does today.

Phases inside `hybrid_chunks` (one deadline covers all of them):

```
deadline = monotonic() + budget_minutes*60
Phase 1  seed top-up      -> annotate_distill._seed_chunks(..., sampler=stratified)
                             target = ANNOTATE_HYBRID_SEED_TARGET (default 800)
Phase 2  kNN classify all -> annotate_distill._classify_chunks (~90 s)
                             (now also writes knn_confidence)
Phase 3a event upgrades   -> LLM re-annotate event-candidate chunks in scope
                             (source='distilled' or unannotated), until done
                             or 60% of the remaining post-Phase-2 time is spent
Phase 3b confidence upgrades -> LLM re-annotate lowest-knn_confidence
                             source='distilled' chunks in scope, until deadline
```

Deadline semantics: check before *dispatching each group*; in-flight groups
finish and commit (bounded overshoot ≈ one group ≈ tens of seconds). Phase 1
and 2 are never skipped mid-phase for the deadline (they're the foundation and
are cheap); only Phase 3 dispatch is deadline-gated. On deadline or Ctrl-C:
commit, print a summary of what each phase completed, exit 0 — a rerun resumes
(seeds count toward target; `'llm'` rows are no longer pending; confidence
ranking re-reads persisted `knn_confidence`).

Scope: Phases 1–2 are corpus-wide by design (same as distilled — the kNN vote
must generalize). Phase 3 honors `--all` / `--top-dyads N` (default
`ANNOTATE_TOP_DYADS`), like the LLM tiers. `--limit` caps Phase 3 upgrades.
`--upgrade-distilled` remains an error with `hybrid` (hybrid *is* a
distilled-upgrader; extend the existing `parser.error` branch and message).

### Confidence score (Phase 2, consumed by Phase 3b)

`_knn_predict` gains a 4th return value `confidence` ∈ [0, 1], persisted to a
new nullable `chunk_annotations.knn_confidence` column (NULL for `'llm'` rows
and pre-existing rows). Definition (deterministic, no LLM):

```
w_i   = max(1e-6, 1 - d_i)          # existing neighbour weights
conf  = mean(w_i) * (1 - clamp(weighted_std(valence_i), 0, 1))
```

i.e. high when neighbours are close AND agree on valence. The exact formula
may be refined by the owning agent **only if** it stays deterministic, stays in
[0, 1], and stays monotone in neighbour agreement/proximity; the column name,
type, and NULL semantics are frozen.

### Event-candidate prefilter (Phase 3a)

A deterministic lexicon prefilter selects chunks likely to contain
`good_news`/`gratitude` events. Patterns live in `function_words.py` (the
existing home for lexicons/regexes) as `EVENT_CANDIDATE_PATTERNS` — two tuples
of lowercase substrings, e.g.:

- gratitude: `thank`, `thx`, `ty sm`, `appreciate`, `grateful`, `you're the best`, `owe you`
- good news: `congrat`, `got the job`, `got the offer`, `i got in`, `passed`,
  `promoted`, `engaged`, `pregnant`, `we got`, `accepted`, `great news`, `good news`, `guess what`

Selection SQL matches `messages.text` (LIKE, lowercased) within the scope's
chats and maps hits to distinct `chunk_id`s that are currently
un-upgraded (`source='distilled'` or unannotated). Do **not** wrap
`messages.date` in SQL functions; the match is over `text` only, restricted by
`chat_id IN (top-dyad subquery)` — same subquery shape as
`annotate._pending_chunks`. FTS5 (`messages_fts`) MAY be used instead of LIKE
for the single-word patterns if the owning agent verifies phrase queries work;
LIKE over top-dyad rows is acceptable and simpler. Expected yield: 5–10% of
in-scope chunks; Phase 3a's time cap (not the yield) bounds the work.

### Stratified seed sampler (Phase 1)

Replaces `random.sample` for hybrid only (`distilled` behavior unchanged).
Two-level:

1. Allocate the seed top-up across chats proportionally to their chunk counts
   (largest-remainder rounding; every chat with ≥ 1 pending chunk gets a
   chance).
2. Within a chat, prefer embedding diversity: farthest-point selection over
   `chunk_vectors` embeddings (numpy, greedy, seeded `random.Random(0)` for the
   start point — deterministic). If embeddings are missing for a chunk, fall
   back to random within the chat.

This also mitigates the documented caveat in `annotate_distill._seed_chunks`
(inherited pools biased toward top dyads).

## 4. Contracts (frozen — all agents code against these signatures)

```python
# annotate_sampling.py (NEW FILE — owned by W2)
def stratified_sample(con, pending, need):
    """pending: list[(chunk_id, text)] as returned by annotate._pending_chunks.
    Returns a list of `need` rows drawn from pending (fewer if pending is
    smaller), stratified per §3. Deterministic given (pending, need, DB).
    Pure read-only SQL on chunk_vectors/chunks; no writes, no LLM."""

# annotate_events.py (NEW FILE — owned by W3)
def event_candidate_chunk_ids(con, all_chats, top_dyads):
    """Returns an ordered list[int] of chunk ids in scope whose messages match
    EVENT_CANDIDATE_PATTERNS and whose annotation is absent or
    source='distilled'. Read-only. Order: chunk id ascending."""

# function_words.py (addition — owned by W3)
EVENT_CANDIDATE_PATTERNS = {"gratitude": (...), "good_news": (...)}

# annotate_distill.py (changes — owned by W1)
def _knn_predict(con, labels, emb, k):
    """-> (valence, emotions, depth, confidence)  # was 3-tuple"""
def _seed_chunks(con, model, think, workers, target, dry_run=False, sampler=None):
    """sampler: Optional[Callable[[con, pending, need], list]] — None keeps
    today's random.sample behavior exactly."""
# _classify_chunks writes knn_confidence on every distilled row it inserts.

# schema.py (changes — owned by W1)
# chunk_annotations gains: knn_confidence REAL   (nullable)
# DDL updated + idempotent ALTER migration in create_schema, following the
# dyad_terms.curate_failures pattern (PRAGMA table_info gate).

# annotate_hybrid.py (NEW FILE — owned by W4)
def hybrid_chunks(con, model, think=False, workers=1, all_chats=False,
                  top_dyads=None, limit=None, dry_run=False,
                  budget_minutes=None, seed_target=None):
    """The tier orchestrator per §3. budget_minutes/seed_target None ->
    resolve from config (ANNOTATE_BUDGET_MINUTES / ANNOTATE_HYBRID_SEED_TARGET,
    with the apple-vs-ollama default split). Returns count of rows written.
    Calls annotate._check_model_guard before any write (Phase 1 does this via
    _seed_chunks already; Phase 3 must not assume Phase 1 ran)."""

# annotate.py (changes — owned by W4)
QUALITY_PRESETS["hybrid"] = (False, 8)     # think=False — NOT 'low'; see §2
_APPLE_ANNOTATE_CHAR_BUDGET = 8000         # was 12000; see §2
# main(): quality == "hybrid" -> annotate_hybrid.hybrid_chunks(...)
# parser.error branches: --upgrade-distilled also rejected for hybrid;
# --quality help text gains the hybrid sentence.

# config.py + .env.example (owned by W5)
ANNOTATE_BUDGET_MINUTES      # int; empty/0 -> default 25 (ollama) / 50 (apple),
                             # resolved at run time via config.is_apple_model(model)
                             # — the resolution helper lives in annotate_hybrid,
                             # config just parses the env int (0 = unset).
ANNOTATE_HYBRID_SEED_TARGET  # int, default 800
```

Rules that bind every agent (from CLAUDE.md — violations will be rejected):

- All tunables go in `config.py` + `.env.example`; never hardcode elsewhere.
- `think` is tri-state — never collapse `None` and `False`. The hybrid preset
  is `False`; `ANNOTATE_THINK` override must keep working via
  `_resolve_quality` (do not bypass it).
- Model-guard: every LLM write path goes through the existing
  `annotate._check_model_guard` / `annotate._write_annotation`; `source` is
  `'llm'` for anything an LLM produced, `'distilled'` for kNN rows.
- Never wrap `messages.date` in SQL date functions inside WHERE.
- DB paths resolve via `config.py`; tests patch `annotate.DB_PATH` /
  module attributes, never real DBs.
- Don't read or modify `tests/` except W6.
- Lint: `make lint` (ruff, line-length 100) must pass on every touched file.

## 5. Workstreams (parallelizable)

File-ownership matrix — **no two workstreams edit the same file**, so all of
W1–W5 can run concurrently; W6/W7 follow. If a workstream needs a change in a
file it does not own, it asks the owner (or leaves a TODO for W6 integration).

| WS | Owns (files) | Depends on | Size |
|---|---|---|---|
| W1 confidence + schema | `annotate_distill.py`, `schema.py` | — | S/M |
| W2 stratified sampler | `annotate_sampling.py` (new) | — | M |
| W3 event prefilter | `annotate_events.py` (new), `function_words.py` | — | S/M |
| W4 orchestrator + CLI | `annotate_hybrid.py` (new), `annotate.py` | contracts only (stubs fine) | M/L |
| W5 config + env + make | `config.py`, `.env.example`, `Makefile` | — | S |
| W6 tests + docs | `tests/test_annotate_hybrid.py` (new), existing annotate tests if broken, `CLAUDE.md`, module docstrings | W1–W5 merged | M |
| W7 validation harness + timing run | scratch scripts only (no repo files) | W1–W5 merged | S/M |

### W1 — kNN confidence + schema migration

- `schema.py`: add `knn_confidence REAL` to the `chunk_annotations` DDL and an
  idempotent `PRAGMA table_info`-gated `ALTER TABLE` in `create_schema`
  (copy the `dyad_terms.curate_failures` pattern).
- `annotate_distill.py`: `_knn_predict` returns the 4-tuple per §3/§4;
  `_classify_chunks` writes the confidence into its INSERT. The `INSERT OR
  REPLACE` column list in `_classify_chunks` grows by one; the seed path and
  `annotate._write_annotation` are untouched (llm rows leave it NULL).
- `_seed_chunks` gains the optional `sampler=` parameter (default None =
  today's `random.sample` — distilled behavior must be bit-identical when
  sampler is None).
- Check: existing distilled callers (`distill_chunks`) still work; the empty
  `nn` early-return returns confidence `0.0`.

### W2 — stratified sampler (`annotate_sampling.py`)

- Implement `stratified_sample` per §3/§4. numpy is already a dependency
  (verify in `requirements.txt` before importing; if absent, implement
  farthest-point in pure python — 800 seeds over ≤ 25k vectors is fine).
- Embeddings come from `chunk_vectors` (`SELECT chunk_id, embedding FROM
  chunk_vectors WHERE chunk_id IN (...)`); embedding blobs are float32 arrays
  (`embed_dim` in `ingest_state`). Chunks without embeddings: random fallback
  within their chat.
- Deterministic: `random.Random(0)` for anything stochastic.
- Self-test hook: module runnable standalone against a copy of a real DB is a
  plus, not required (W7 covers it).

### W3 — event-candidate prefilter (`annotate_events.py` + `function_words.py`)

- `EVENT_CANDIDATE_PATTERNS` in `function_words.py` (start from §3's lists;
  keep them lowercase substrings; add obvious variants, keep < ~40 total).
- `event_candidate_chunk_ids` per §4: LIKE-match with `LOWER(text)` (or FTS if
  verified), scope via the same top-dyad subquery as
  `annotate._pending_chunks`, join to `messages.chunk_id IS NOT NULL`, filter
  out already-`'llm'` chunks via `LEFT JOIN chunk_annotations`.
- Escape LIKE wildcards if any pattern ever contains `%`/`_` (reuse the
  `annotate._like_escape` idea locally — do not import annotate's private
  helper; duplicate the 3-line escape or keep patterns wildcard-free).

### W4 — orchestrator (`annotate_hybrid.py`) + CLI wiring (`annotate.py`)

- Build against §4 contracts; until W1–W3 land, develop with local stubs, then
  drop the stubs at integration (the signatures are frozen, so this is a
  one-line import swap).
- `hybrid_chunks` per §3: monotonic-clock deadline, phase prints (mirror
  distilled's style: one header line per phase with counts + settings),
  `progress.track` for each phase, `ANNOTATE_COMMIT_EVERY` cadence, Ctrl-C
  commits + prints resume hint (copy the existing pattern; phase 3 groups run
  through `annotate._pooled` + `annotate._annotate_group` exactly like
  `annotate_chunks` does).
- Phase 3 pending queries: event candidates from W3's function; confidence
  candidates via `SELECT chunk_id FROM chunk_annotations WHERE
  source='distilled' AND knn_confidence IS NOT NULL AND chunk_id IN (scope)
  ORDER BY knn_confidence ASC` joined back to `chunks` for text. Rows without
  confidence (pre-hybrid distilled rows) sort last (`IS NULL` bucket) rather
  than never — use `ORDER BY knn_confidence IS NULL, knn_confidence` and
  include them.
- Dedup: a chunk upgraded in 3a must not be re-sent in 3b (3b's query already
  excludes it once it's `'llm'` — but also dedup in-memory within the run).
- Apple: groups built with `annotate._char_budget_groups` when
  `config.is_apple_model(model)`, else batch 8 — identical branch to
  `annotate_chunks`; factor if convenient but do not change `annotate_chunks`
  behavior.
- `annotate.py`: preset entry, apple char-budget constant 12000→8000, dispatch
  branch, `--upgrade-distilled` rejection for hybrid, help-text sentence,
  module-docstring paragraph. Keep `--terms` completely untouched.
- `--dry-run`: print the resolved budget, seed shortfall, kNN-pending count,
  event-candidate count, and low-confidence count in scope; no LLM calls, no
  writes (thread `dry_run` through every phase like distilled does).

### W5 — config, env, Makefile

- `config.py`: `ANNOTATE_BUDGET_MINUTES` (int, `0`/empty = unset →
  per-backend default applied in `annotate_hybrid`), `ANNOTATE_HYBRID_SEED_TARGET`
  (default 800). Follow the surrounding style/comments (explain the measured
  basis: §2 numbers).
- `.env.example`: both knobs, commented defaults.
- `Makefile`: `annotate-hybrid` target (`.venv/bin/python3 annotate.py
  --quality hybrid`), placed next to the other annotate targets, `.PHONY`
  updated, and a one-line mention in `make help` if a help target enumerates.

### W6 — tests + docs (after W1–W5 merge)

- `tests/test_annotate_hybrid.py`, hermetic (no Ollama, no real chat.db):
  patch `annotate.DB_PATH`/connection, fake `ollama.chat_once_json`. Cover:
  deadline stops Phase 3 dispatch (fake clock or tiny budget + slow fake LLM);
  Phase 3a→3b ordering and dedup; resume (second run skips `'llm'` rows);
  confidence written by `_classify_chunks` and NULL on llm rows; stratified
  sampler determinism + proportional allocation; event prefilter matches and
  scope; migration adds `knn_confidence` idempotently; CLI rejections
  (`--terms --quality hybrid`, `--upgrade-distilled --quality hybrid`).
- Run only targeted modules (`python3 -m unittest tests.test_annotate_hybrid`
  and any annotate/distill test module that exists) — never full discover.
- Docs: CLAUDE.md command bullet for `hybrid` (one sentence, match existing
  density); `annotate.py` docstring already updated by W4 — verify coherence.
  `docs/DECISIONS.md`: short entry recording the §2 measurements and the
  12000→8000 apple budget change rationale.

### W7 — validation harness + acceptance timing (after W1–W5 merge)

- **Never touch the real DBs in the main checkout.** Copy
  `imessage_analysis_qwen3.5:4b.db` (guard: `annotate_model` there is
  `qwen3.5:4b`) into the scratchpad; run against the copy with `DB_PATH`
  pointed at it (env/`--` mechanisms: `TARGET_DB` is not an env knob — patch
  `annotate.DB_PATH` via a small driver script, or set the path through
  `config` if a knob exists; check `config.py` first).
- Runs (each ≤ 10 min wall): `--dry-run` sanity; a `--limit`-capped real run
  on qwen3.5:4b measuring s/chunk for Phase 3; a short apple run via
  `APPLE_SHIM_PATH=<main checkout>/bin/applefm-shim` if the shim responds.
  Extrapolate to the full budget; a full-budget qwen run (25 min) is allowed
  as the final acceptance check (it exceeds no single-test cap of 10 min? —
  it does; split it: run with `ANNOTATE_BUDGET_MINUTES=8` and verify the
  deadline machinery + throughput, then extrapolate; the maintainer runs the
  full 25-min pass).
- Report: measured s/chunk per phase, projected chunks upgraded in 25/50 min,
  confirmation the run self-terminated at the deadline with a clean summary
  and that a rerun resumed.

## 6. Integration & sequencing

```
      ┌ W1 ┐
      ├ W2 ┤
start ├ W3 ┼──> merge (integration agent or W4) ──> W6 ──┐
      ├ W4 ┤                                     └─> W7 ─┴─> done
      └ W5 ┘
```

- W1–W5 in parallel, each on its own files.
- Integration = W4's agent swaps stubs for real imports, runs `make lint`,
  runs the new test module, fixes seams. W6 and W7 then run in parallel.
- Merge order if serialized: W5 → W1 → W2/W3 → W4 → W6/W7 (config first so
  imports resolve; but any order works since files are disjoint).

## 7. Environment notes for agents

- Work in this worktree: `/Users/gabrielruoff/LocalDocuments/GitHub/imessage-rag/.claude/worktrees/annotate-sweet-spot`. Do not `cd` to the main checkout.
- The worktree has **no `.venv` and no `.env`**. Run python via the main
  checkout's interpreter: `/Users/gabrielruoff/LocalDocuments/GitHub/imessage-rag/.venv/bin/python3`
  (it has all pinned deps). `config.load_dotenv` is pinned to the worktree dir,
  so no `.env` is inherited — defaults apply.
- Real DBs live in the **main checkout** (`imessage_analysis.db`,
  `imessage_analysis_qwen3.5:4b.db`, `imessage_analysis_applefm.db`). They are
  the user's data: read-only access for stats is fine (`file:...?mode=ro`),
  writes only ever to copies in the scratchpad (W7 only).
- Ollama is up (`localhost:11434`), `qwen3.5:4b` pulled and warm. Apple shim
  binary: `<main checkout>/bin/applefm-shim` (set `APPLE_SHIM_PATH` to use it
  from the worktree).
- No single test/benchmark command over 10 minutes.
- `make lint` before declaring any workstream done. Do not run
  `unittest discover`.

## 8. Acceptance criteria (whole feature)

1. `python3 annotate.py --quality hybrid --dry-run` prints resolved budget,
   seed shortfall, kNN-pending, event-candidate and low-confidence counts; no
   writes, no LLM calls.
2. With `ANNOTATE_BUDGET_MINUTES=8` on qwen3.5:4b: run self-terminates within
   ~9 min, commits, prints a per-phase summary; measured Phase-3 throughput
   ≤ 2.5 s/chunk (batch 8, think=False). Extrapolation confirms a 25-min
   default budget lands < 30 min total; apple 50-min default lands < 1 h.
3. Rerun after an interrupted/deadline-terminated run resumes: no chunk is
   LLM-annotated twice, seed target counts existing `'llm'` rows.
4. `distilled` tier behavior unchanged (random sampling path bit-identical;
   only addition: `knn_confidence` now populated on new distilled rows).
5. `--terms` completely untouched. Guards (`annotate_model`, embed guards)
   behave exactly as before; mixing models still refuses.
6. New/changed files pass `make lint`; `tests.test_annotate_hybrid` passes
   hermetically.
7. Downstream smoke: after a hybrid run on a DB copy, `conflict_detect.py` /
   `loose_ends.py` readers see the upgraded `'llm'` rows (no schema breakage
   from the new column — verify a couple of `chunk_annotations` readers via
   grep for `SELECT` column lists that would break; `INSERT OR REPLACE` sites
   are only in `annotate.py`/`annotate_distill.py`).
