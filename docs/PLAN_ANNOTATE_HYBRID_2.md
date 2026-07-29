# Follow-up plan: hybrid tier throughput on qwen3.5:4b (round 2)

Status: IMPLEMENTED (§3a–§3d, 2026-07-28). Kept as design history — see
`docs/DECISIONS.md` (the round-2 annotation-throughput entry) for what actually
shipped and what the post-implementation measurements said. §5's workstream
matrix and §7's acceptance criteria describe how the work was carried out, not
outstanding work; the 25-min acceptance pass in §1 remains the maintainer's.
Audience: implementation agents working in parallel. Companion to
`docs/PLAN_ANNOTATE_HYBRID.md` (the shipped tier, PR #42) — read that
document's status header first: it records which of round 1's premises were
refuted, and this plan builds on the *corrected* numbers. The **Contracts**
section here is binding; the file-ownership matrix is what makes parallel work
safe.

## 1. Goal

The hybrid tier works but realizes ~2.2–3.3 s/chunk against a raw model speed
of ~1.1–4.6 s/chunk, and a 25-min default budget buys only ~250–350 upgrades.
This plan removes the three measured taxes — degenerate batch fallbacks, an
O(rows × seeds) kNN pass that grows every run, and unranked event candidates —
without changing the tier's contract (wall-clock budget, resumable, Ctrl-C
safe, `source` semantics, guards).

Success criteria (hard):

- A 25-min `--quality hybrid` run on `qwen3.5:4b` completes **≥ 400 Phase-3
  upgrades** on the reference corpus (vs ~250–350 today), still self-terminating
  inside the budget.
- Phase 2 (kNN classify + backfill) costs **≤ 25 s** even at a 4,416-seed pool
  (vs 82 s measured).
- The worst Phase-3 group latency, single-stream, is **≤ ~70 s** (vs 226 s
  measured) — this is what un-poisons the deadline gate's reserve.
- Apple FM is **out of scope for performance**: the only apple requirement is
  no regression — `--quality hybrid --dry-run` and one tiny real batch still
  work through the shim. Do not tune for it, do not benchmark it beyond that
  smoke check.

## 2. Measured facts (2026-07-28, qwen3.5:4b warm, main Ollama, this machine)

Reference corpus: 25,029 chunks; top-20 dyads = 19,483; event prefilter yields
2,567 candidates in that scope (~20%, avg ~1,900 chars); low-confidence
candidates are short (median ~360 chars).

| Workload (batch 8 unless noted) | free-JSON (current) | schema-constrained |
|---|---|---|
| Longest event chunks (~2,150 avg chars) | 13.1 / **28.3** s/chunk | 5.4 / 5.9 s/chunk |
| Typical event chunks (~1,550 avg chars) | **4.6** s/chunk | 7.0 s/chunk |
| Short distilled chunks (~210 chars) | 3.5 | 2.7 (batch 16), 3.2 (batch 24) |
| `OLLAMA_NUM_PARALLEL=4` (2nd server) | — | 3.9 vs 4.8 s/chunk sequential (~1.24×) |

Readings:

- **Free-JSON is bimodal.** With full id coverage it is the fastest option
  (~4.6 s/chunk). When the model drops ids — correlated with long, event-dense
  chunks — `_annotate_group` falls back to one single-chunk request per miss
  and the group lands at 13–28 s/chunk (the banked 106–226 s/group rates).
- **Schema-constrained decoding is not universally faster** (constant grammar
  overhead; slower than free-JSON's happy path on typical chunks). Its value is
  that coverage is guaranteed *by construction* — the degenerate tail cannot
  happen. So: free-JSON stays the fast path; the schema goes on the *retry*.
- **`_save_group_rate` stores the worst group** (deliberately), so today's
  fallback blowups make `_DispatchDeadline` over-reserve and decline
  dispatches. Capping the worst case is worth budget twice: faster groups AND
  a smaller reserve. Stale banked rates self-heal — one run at the new worst
  overwrites them — so no migration is needed.
- **kNN classify is O(rows × seeds)**: 14.6 s at a 450-seed index vs 82.0 s at
  4,416 — and the tier widens the pool every run, so untreated this cost only
  grows.
- **Parallelism is a dead end on this hardware**: 4 server slots bought ~1.24×.
  Decision: leave `OLLAMA_NUM_PARALLEL` alone; do not spend further effort
  here. (Recorded so nobody re-litigates the "~2× with a parallel Ollama"
  hope in the older docstrings — update those comments where touched.)
- numpy is **not** a project dependency (the stratified sampler is pure
  python). The kNN fix below is a seed-index cap, not a matmul; adding numpy is
  explicitly out of scope.

## 3. Design (four independent changes)

### 3a. Schema-constrained retry for missed batch ids (highest ROI)

`_annotate_group` today: batched free-JSON request → any id missing from the
reply gets its own single-chunk `chat_once_json` request (each up to full
price, each with its own nudge retry). Change the middle step:

1. Batched free-JSON request (unchanged — the fast path stays the fast path).
2. Ids the reply missed/mangled → **one** request with Ollama structured
   outputs: `format` = a JSON Schema object whose `properties`/`required` list
   exactly the missed chunk ids, each mapping to the annotation schema
   (valence number; emotions array with the `EMOTION_LABELS` enum, maxItems 3;
   depth integer 1–5; events array of {kind enum, excerpt string,
   response_class enum-or-null}). `ollama.chat_once` already forwards a dict
   `format` verbatim into the request body — no ollama.py change needed.
3. Anything *still* unusable (schema request errored/timed out) → the existing
   single-chunk fallback, unchanged. Belt and braces; expected ~never.

The retry schema is small (only the missed ids), so its grammar overhead is
small; worst case becomes roughly one extra ~40–55 s request instead of up to
8 sequential singles.

**Apple carve-out:** `config.is_apple_model(model)` → skip step 2 entirely and
keep today's per-chunk fallback. Apple's `format` seam is not a guided schema
(see CLAUDE.md); passing a schema dict through `applefm.chat_once` is
undefined. One `if`, placed where the retry is dispatched.

### 3b. Rank event candidates by hit density (Phase 3a)

`event_candidate_chunk_ids` returns chunk-id-ascending today; a 25-min run
reaches ~10–15% of the 2,567 candidates, selected by *age*. Change the
ordering to pattern-hit density: count matching messages per chunk in the same
scan that finds candidates (`COUNT(*)` per `chunk_id` over the pattern-matched
messages), order by that count descending, chunk id ascending as the
tiebreaker (determinism). Resume semantics survive unchanged: upgraded rows
become `source='llm'` and drop out of the candidate query, so reruns advance
down the ranking exactly as they advanced down the id order.

**Order preservation bug-in-waiting:** `annotate_hybrid._chunk_rows` currently
fetches with `ORDER BY id` per slice and documents "ids arrive ascending" —
ranked input would be silently re-sorted. It must preserve the caller's order
(fetch the slice, reorder in python to the input sequence). Same workstream,
since the two changes are one feature.

### 3c. Cap the kNN seed index (Phase 2)

`annotate_distill._build_seed_index` loads every `source='llm'` row into the
temp vec0 table; query cost scales with it. Cap the *index* (never the pool):
when the pool exceeds `ANNOTATE_KNN_INDEX_CAP` (new knob, default 800, 0 =
uncapped), take a deterministic sample of that size — `random.Random(0)` over
the seed ids is sufficient (stratification of the index is not required; the
pool itself is already stratified by the tier's sampler). Applies to both
`_classify_chunks` and `backfill_confidence` (they share the index). Print the
cap when it engages (`using 800 of 4,416 seeds for the kNN index`). At the
default this pins Phase 2 near its 450-seed cost (~15–20 s) forever.

Accuracy note for the reviewer, not a gate: k=10 votes drawn from an
800-sample of a stratified pool is the regime the tier already operated in at
smaller pools; round 1's full-corpus probing found seed-pool composition
effects of ~1% at k=10. No held-out eval required for this change.

### 3d. Char-budget packing for Ollama batches

Fixed batch-8 lets group cost swing ~4.6→28 s/chunk with chunk length, which
feeds the worst-group estimator garbage and inflates the gate's reserve.
`_annotation_groups` gains an Ollama-side char budget: pack up to `batch_size`
rows per group, starting a new group early when the next row would push the
group's total chunk text past `_OLLAMA_ANNOTATE_CHAR_BUDGET = 10_000` (module
constant beside `_APPLE_ANNOTATE_CHAR_BUDGET`, same rationale: a property of
the backend, not a user knob; one oversized chunk still gets its own singleton
group — reuse `_char_budget_groups` with a `max_rows` parameter rather than
duplicating its loop). Short chunks keep batching at 8; long event chunks land
in 4–6-row groups whose latency clusters instead of tailing. Applies to every
LLM tier (they share `_annotation_groups`); apple's packing is untouched.

Interaction with rates: `_group_rate_key` embeds `batch_size`, which is now a
*maximum* — that is fine (the key identifies the packing config, not the exact
group size), but the comment on the key must say so.

## 4. Contracts (frozen)

```python
# annotate.py (owned by W3)
_OLLAMA_ANNOTATE_CHAR_BUDGET = 10_000          # module constant, §3d
def _char_budget_groups(pending, budget, max_rows=None):  # gains max_rows
def _annotation_groups(pending, model, batch_size):
    # apple: unchanged; ollama: _char_budget_groups(pending,
    #        _OLLAMA_ANNOTATE_CHAR_BUDGET, max_rows=batch_size)
def _annotation_schema(chunk_ids):
    """JSON Schema (dict) requiring exactly str(id) keys -> annotation object
    schema per §3a. Pure; no I/O."""
def _annotate_group(group, model, think):
    # same signature/return; internally: batch -> schema retry for missed ids
    # (ollama only, one request) -> per-chunk fallback for any remainder.

# annotate_events.py (owned by W1)
def event_candidate_chunk_ids(con, all_chats, top_dyads):
    """Same signature. Order changes: pattern-hit count DESC, chunk id ASC.
    Still excludes source='llm' rows; still read-only."""

# annotate_hybrid.py (owned by W1)
def _chunk_rows(con, chunk_ids):
    """Same signature. Returns rows in the CALLER'S id order (no re-sort)."""

# annotate_distill.py (owned by W2)
# _build_seed_index (and its callers _classify_chunks/backfill_confidence)
# honor config.ANNOTATE_KNN_INDEX_CAP per §3c. No signature changes.

# config.py + .env.example (owned by W2)
ANNOTATE_KNN_INDEX_CAP           # int, default 800; 0 = uncapped
```

Binding rules (unchanged from round 1, still enforced): tunables in
`config.py`/`.env.example` only; `think` tri-state preserved; every write
through `_check_model_guard`/`_write_annotation`; `source='llm'` only for LLM
output; no `messages.date` wrapping in WHERE; tests patch module attributes,
never real DBs; `make lint` clean; don't touch `tests/` outside W4; never run
`unittest discover`.

## 5. Workstreams

Disjoint files per workstream — W1–W3 run concurrently; W4–W5 follow the merge.

| WS | Owns (files) | Change | Size |
|---|---|---|---|
| W1 event ranking | `annotate_events.py`, `annotate_hybrid.py` | §3b + `_chunk_rows` order preservation | S |
| W2 kNN index cap | `annotate_distill.py`, `config.py`, `.env.example` | §3c + knob | S |
| W3 batch shape | `annotate.py` | §3d packing + §3a schema retry (same file, one owner) | M |
| W4 tests + docs | `tests/` (targeted modules), `CLAUDE.md`, `docs/DECISIONS.md` | after W1–W3 | M |
| W5 validation | scratch scripts only | after W1–W3 | S/M |

### W1 — event candidate ranking

- Rewrite the candidate query to aggregate hits per chunk (`GROUP BY
  m.chunk_id`, order `COUNT(*) DESC, m.chunk_id ASC`). Keep the scope subquery
  and the `'llm'` exclusion exactly as they are; keep it read-only.
- Fix `_chunk_rows` to restore caller order after the sliced fetch (dict lookup
  by id; drop the "arrive ascending" comment, say what the new contract is).
- Update the docstring sentence CLAUDE.md quotes ("chunk-id ascending…") — flag
  it for W4's CLAUDE.md edit rather than editing CLAUDE.md yourself.

### W2 — kNN index cap

- Implement §3c in `_build_seed_index`; both callers inherit it. Deterministic
  (`random.Random(0)`); cap message printed only when it engages.
- Knob in `config.py` (comment: the 14.6 s/82 s measurement and why 800) and
  `.env.example`.

### W3 — batch packing + schema retry

- §3d first (it's mechanical), then §3a on top. `_annotation_schema` builds
  from the existing constants (`EMOTION_LABELS`, `EVENT_KINDS`,
  `RESPONSE_CLASSES`) — no string literals that can drift from
  `_validate_annotation`.
- The schema retry parses with the same tolerance as the batch path (missing →
  fall through to singles); `_validate_annotation` remains the single
  validation gate, so a schema-shaped reply gets no special trust.
- Apple carve-out per §3a. While in the file, correct the stale "~2x with a
  parallel Ollama" claim in the module docstring to the measured ~1.24×.

### W4 — tests + docs (after merge)

- Targeted additions to the existing hybrid/annotate test modules (find them
  with a grep for `hybrid` under `tests/`; run only those modules): ranked
  order + order preservation through `_chunk_rows`; index cap determinism and
  the 0=uncapped path; `_annotation_schema` shape (ids as keys, required, enums
  match the module constants); `_annotate_group` retry ladder (batch miss →
  one schema call → singles) with a fake `ollama`; apple path never receives a
  dict `format`.
- Docs: `docs/DECISIONS.md` entry with §2's table and the NUM_PARALLEL
  negative result; CLAUDE.md — update the hybrid bullet's "chunk-id ascending"
  clause to the ranking, one sentence on the schema retry, nothing else.

### W5 — validation (after merge)

- Copies only, never the real DBs (`file:...?mode=ro` for reads; writable runs
  against a scratchpad copy of `imessage_analysis_qwen3.5:4b.db`). No single
  command over 10 minutes.
- Runs: `--dry-run` (event count now printed in ranked order — spot-check the
  first ids differ from ascending); `ANNOTATE_BUDGET_MINUTES=8` real run —
  verify Phase 2 ≤ 25 s at the full pool, no group over ~70 s in the log,
  upgrades/minute vs the pre-change banked rates; extrapolate to the 25-min
  criterion (§1). The full 25-min acceptance pass is the maintainer's.
- Apple smoke only (`APPLE_SHIM_PATH` from the main checkout): `--dry-run`
  plus one tiny `--limit` batch; pass/fail is "did not regress", nothing about
  speed.

## 6. Environment notes

Same as round 1 (`docs/PLAN_ANNOTATE_HYBRID.md` §7): work in this worktree;
python is the main checkout's `.venv/bin/python3`; no `.env` is inherited;
Ollama on `localhost:11434` with `qwen3.5:4b` pulled; second Ollama servers
for experiments go on another port and get killed afterwards. One addition:
banked group rates live in `ingest_state` under
`annotate_group_seconds:<phase>:<model>:b<batch>:w<workers>` — W5 should read
them before and after runs for the throughput comparison instead of re-timing
by hand.

## 7. Acceptance criteria

1. §1's four hard criteria, measured per W5 (8-min run + extrapolation; the
   25-min pass is the maintainer's).
2. Ranked events: first ids dispatched are the densest candidates, reruns
   advance, `--limit` still caps totals across 3a+3b.
3. `ANNOTATE_KNN_INDEX_CAP=0` reproduces today's behavior bit-for-bit;
   default caps only the index, never the stored pool.
4. Free-JSON happy path unchanged (full-coverage batch → zero extra requests);
   a forced miss triggers exactly one schema request before any singles;
   apple never sees a dict `format`.
5. Guards, resume, Ctrl-C, `--terms`, and the other quality tiers behave
   exactly as before; `make lint` clean; targeted test modules pass
   hermetically.
