# Repo Review: textmessagething

_Generated 2026-07-17. Scope: report-only, moderate refactors (restructure + dedupe). No code was changed._

## 1. Executive summary

This repo is in unusually good shape for one built through rapid AI-assisted iteration: zero dead code (vulture, ≥60% confidence), zero lint findings, zero real dependency issues, no checked-in noise, a strong and mostly-current `CLAUDE.md`, hermetic tests wired into CI, and a verified three-way sync between the 21 tool schemas, `query_tools` exports, and the chat system prompt. The dominant problem is the classic AI-merge one, in miniature: a handful of helpers copy-pasted across modules instead of imported — one pair has already begun to drift — plus one 987-line module and four functions complex enough (radon D–F) that agents will make blind edits inside them. The single biggest lever is restoring single-source-of-truth for the four duplicated helpers; everything else is incremental context-fit work. Target end state: every shared helper defined once, no source module over ~700 lines, no F-rated functions, and `CLAUDE.md` covering the evalkit subsystem.

**Top 3 actions (do these first):**
1. Consolidate the four identical `_date_clauses` copies into `query_tools/_shared.py` — removes the highest-risk drift surface in the query layer / trivial effort — `analytics.py:35`, `analytics_time.py:57`, `analytics_social.py:58`, `analytics_language.py:54`.
2. Merge the two already-diverging `_ask_json` JSON-retry helpers into one `ollama.py` function — stops an in-progress divergent copy / small effort — `annotate.py:78`, `topics.py:50` (caught by pylint duplicate-code).
3. Add evalkit to `CLAUDE.md`'s module map and commands — the newest, largest subsystem (2,896 LOC) is invisible to agent orientation / trivial effort — `grep evalkit CLAUDE.md` returns nothing.

**Severity tally:** Critical 0 · High 2 · Medium 3 · Low 5 · Informational 6

## 2. Metrics snapshot

| Metric | Current | Notes |
|---|---|---|
| Python files / total LOC | 49 files / 14,612 LOC | 10,473 source + 4,139 tests |
| Largest files (LOC) | `eval/evalkit/recipes.py` 987 · `query_tools/analytics_language.py` 639 · `wrapped.py` 602 · `query_tools/analytics_social.py` 598 · `tool_schemas.py` 536 | split candidates (tool_schemas is pure data — leave) |
| Unused symbols (vulture, ≥60% conf) | **0** | exceptionally clean |
| Unused imports/locals (ruff F401/F811/F841) | **0** | full `ruff check .` also clean |
| Duplicate blocks (pylint, ≥6 lines) | 1 clone pair (`_ask_json`, annotate↔topics) | manual grep found a second: `_date_clauses` ×4 |
| Unused deps (deptry) | 0 real | 2 reported hits are one false positive (`python-dotenv` ships module `dotenv`) |
| Undeclared deps (deptry) | 0 real | same false positive |
| Avg complexity (radon) | **A (3.53)** over 691 blocks | 13 functions rated D/E/F (3 F, 1 E, 9 D) |
| Maintainability index | all A except `eval/evalkit/checks.py` B (13.3), `eval/evalkit/recipes.py` B (15.6) | |
| Files >500 LOC (source) | 5 | see largest-files row |
| Checked-in noise | none | 87 tracked files; `.gitignore` covers DB/venv/results |
| Config sync | `.env.example` ↔ `config.py`: exact match both directions | verified by key diff |
| Tool registration sync | 21 schemas = 21 exports = 21 system-prompt entries | verified by name diff |

_Re-run the commands in the appendix after acting to measure progress._

## 3. Bloat findings

### 3.1 Duplication

**[High]** `_ask_json` (JSON-mode `chat_once` with one corrective retry) duplicated and **already drifting** — `annotate.py:78` (with `_RETRY_NUDGE` at `annotate.py:59`) vs `topics.py:50` (nudge at `topics.py:38`) — the copies differ in signature (`model` required vs `model=None`), docstring, and nudge wording; a fix to the retry logic in one will silently miss the other. This is the exact clone pylint duplicate-code reports. — Fix: add `chat_once_json(messages, model=None, retry_nudge=<generic default>)` to `ollama.py` (both files already import `ollama`); each caller passes its own nudge text so wording — and therefore behavior — is preserved. Delete both local copies.

**[High]** `_date_clauses` copy-pasted **verbatim ×4** across the analytics modules — `query_tools/analytics.py:35`, `query_tools/analytics_time.py:57`, `query_tools/analytics_social.py:58`, `query_tools/analytics_language.py:54` — identical 9-line bodies (diff-verified for time/social), each carrying the load-bearing "never wrap the column in SQL date functions" index constraint. Its siblings (`_append_id_filters`, `_normalize_date_bound`) already live in `_shared.py`; this one was left behind by successive passes. Four copies of an index-preserving invariant is the highest-value drift risk in the query layer. — Fix: move one copy to `query_tools/_shared.py` next to `_append_id_filters` (`_shared.py:150`), import it in the four modules, delete the copies.

**[Medium]** `_REAL` in `eval/evalkit/recipes.py:61` restates `query_tools/_shared.py:69`'s `_REAL_MESSAGE_CLAUSES` (plus a date-not-null clause) as a new string literal — the comment even names the constant it's copying. If the app's definition of a "real message" changes, evalkit's SQL ground truth silently diverges from tool answers and the eval reports false failures. `sessions.py` and `lexicon_build.py` import the constant; only recipes restates it. — Fix: build `_REAL` from the imported constant, e.g. `_REAL = "m.date IS NOT NULL AND " + " AND ".join(_REAL_MESSAGE_CLAUSES)`, resolved lazily (a tiny `_real_clause()` beside `_qt()` at `recipes.py:23`) to preserve the module's deliberate import-light property (`eval/evalkit/__main__.py:33`).

**[Low]** Five near-duplicate `_connect` helpers — `query_tools/_shared.py:14`, `wrapped.py:62`, `annotate.py:65`, `topics.py:44`, `eval/evalkit/recipes.py:29` — each 3–5 lines; they differ *deliberately* (vec extension yes/no, `row_factory` yes/no, which module attribute is the test patch seam), so this is drift-shaped but not yet drift. — Fix: one `db.connect(path, vec=True, rows=True)` helper in `db.py`; each caller keeps its patchable module-level `DB_PATH` and calls `db.connect(DB_PATH, ...)`, preserving every existing test seam.

### 3.2 Dead & speculative code

No findings — vulture reports **zero** unused symbols even at 60% confidence, ruff is clean, and no commented-out blocks, `if False:` branches, or single-implementation abstractions turned up. This is a genuine strength; nothing to do.

### 3.3 Over-engineering

**[Informational]** Lazy-import cycle `chunking.py:280` (`import ingest` inside `process()`) ↔ `ingest.py:32` (imports chunking at top). It exists to keep `ingest.embed_texts` the single patchable test seam, is documented in-line as cycle-safe, and matches `CLAUDE.md`'s described seam design. Leave it; changing the seam churns tests for no behavioral gain. The evalkit function-body imports (`recipes.py`, `manifest.py:45`, `runner.py`) are likewise deliberate import-lightness, documented in `eval/README.md` — not smells.

**[Informational]** `eval/evalkit/p3.py` — the name is opaque outside the plan doc's phase terminology (P0/P1/P3 in `eval/README.md`'s table). Rename to e.g. `grading_llm.py` only if evalkit is touched anyway; not worth a standalone change.

### 3.4 Dependency bloat

No findings. Three runtime deps (`python-dotenv`, `sqlite-vec`, `rich`), all imported; one dev dep (`ruff`), used by `make lint` and CI. All pinned per the repo convention. **[Informational]** If deptry ever joins the toolchain, add `[tool.deptry.package_module_name_map] python-dotenv = "dotenv"` to `pyproject.toml` to silence its one false-positive pair.

### 3.5 Checked-in noise

No findings. 87 tracked files, no caches/artifacts/secrets; `.gitignore` + `eval/.gitignore` cover the DB, venv, `.env`, and eval results.

## 4. Structural assessment

The repo is a flat script layer (pipeline CLIs at root) over two packages (`query_tools/`, `eval/evalkit/`) — names predict contents well, and `CLAUDE.md` documents the layout. Against the target structure the gaps are context-fit ones (a few oversized modules and functions), not organizational ones. A src-layout migration would rewrite every documented command for zero behavioral gain — explicitly not recommended (**[Informational]**).

### 4.1 Proposed structure

Unchanged except three moderate moves (marked):

```
textmessagething/
├── CLAUDE.md                     # + 3–4 lines covering evalkit (§5)
├── db.py                         # + connect(path, vec=, rows=)      [Low]
├── ollama.py                     # + chat_once_json(...)             [High]
├── annotate.py / topics.py       # − local _ask_json/_connect copies
├── wrapped.py                    # build_sections split into per-section helpers [Low]
├── query_tools/
│   ├── _shared.py                # + _date_clauses (single home)     [High]
│   └── analytics*.py             # − 4 local copies; F-rated fns get helper extraction
└── eval/evalkit/
    ├── roster.py                 # NEW: roster resolution out of recipes.py [Medium]
    └── recipes.py                # ~720 lines of homogeneous @recipe functions
```

### 4.2 Files to split (with the specific split)

**[Medium]** `eval/evalkit/recipes.py` (987 LOC, MI B-15.6, `resolve_roster` D-22) — two unrelated concerns share the file: roster resolution and the ground-truth recipe registry. Move `_contact_entry` (`recipes.py:67`), `_CONTACT_COUNTS_SQL` (`:100`), `_pick_phrase_hi`/`_pick_phrase_lo` (`:116`/`:140`), and `resolve_roster` (`:165–272`) into a new `eval/evalkit/roster.py` (~260 LOC); `manifest.py` and `__main__.py` update their imports. Recipes drops to ~720 LOC of uniform 10–15-line `@recipe` functions — still >500, but homogeneous registry entries an agent scans by name, which is the acceptable shape. Payoff: roster logic (the subtle part — exclusion heuristics, phrase picking) becomes one readable file.

**[Low]** `wrapped.py:350` `build_sections` (radon **F-42**, ~130 lines) — a linear composer with clear internal comment boundaries (`wrapped.py:364,389,427,435,467`). Extract per-section builders (`_sec_year_in_numbers`, `_sec_reactions`, `_sec_late_nights`, `_sec_milestones`, `_sec_words`, `_sec_eras`) that each return the section dict; `build_sections` becomes the ordered composition. Behavior-preserving; `tests/test_wrapped.py` (472 LOC) covers it.

**[Low]** `query_tools/analytics_time.py:321` `get_response_dynamics` (radon **F-46**, the repo's most complex function) — it computes seven independently-named result sections (`reply_latency`, `read_to_reply`, `initiation`, `sessions`, `ghosted`, `double_text`, `burstiness`). Extract one private helper per section, same file; the public function keeps its signature and assembles the dict. Covered by `tests/test_analytics_time.py`.

**[Low]** `query_tools/analytics_language.py:444` `get_group_dynamics` (radon **F-43**) — same treatment along its `include`-section seams. Covered by `tests/test_analytics_language.py`.

**[Informational]** `tool_schemas.py` (536 LOC) is a flat list of dict literals — data, not logic; splitting it would only break the "one file = whole tool surface" property. Leave. `query_tools/analytics_social.py` (598) and `analytics_language.py` (639) are cohesive one-domain modules of independent tool functions; no module split recommended beyond the function extractions above.

### 4.3 Modules to consolidate / rename / relocate

No relocations needed beyond §4.2. The `main`/`add_parser`/`build` name repetitions across CLIs are per-script entry-point convention, not parallel implementations — verb-family grep found no `get_/fetch_/load_` twins.

## 5. Claude Code fitness

### 5.1 Checklist results

| Item | Result |
|---|---|
| `CLAUDE.md` exists, current | **Gap (Medium)** — excellent for the core app (commands verified, 21-tool count verified, gotchas load-bearing), but `evalkit` — 2,896 LOC, the newest subsystem — appears nowhere in it (`grep evalkit CLAUDE.md` → 0 hits); only the older `fast_eval.sh` path is documented. An agent asked to run or extend the analytics evals gets no orientation. |
| `README.md` purpose + quickstart | Pass (135 lines; `eval/README.md` covers evalkit usage well) |
| Commands discoverable and work | Pass — `make` targets + CLAUDE.md invocations match the Makefile |
| Names predict contents | Pass (one blemish: `eval/evalkit/p3.py`, §3.3) |
| No grab-bag utils modules | Pass — `_shared.py` and `db.py` are curated, documented, all-members-used |
| Public API visible from `__init__.py` | Pass — `query_tools/__init__.py` re-exports all 21 tools, thin |
| Few files >500 lines | **Gap (Medium/Low)** — 5 source files; see §4.2 |
| Functions mostly <50 lines | **Gap (Low)** — 13 functions rated D–F; the three F-rated ones are the fix targets (§4.2) |
| One concern per module | Pass, except `recipes.py` (§4.2) |
| No duplicated logic | **Gap (High)** — §3.1: `_ask_json` ×2 (drifting), `_date_clauses` ×4, `_REAL` restated |
| Config/constants defined once | Pass — everything env-driven lives in `config.py`; `.env.example` in exact sync |
| Fixtures/helpers defined once | Pass — tests share `schema.create_schema`; per CLAUDE.md, `tests/` is out of scope for restructuring |
| Enforced formatter + linter | Pass — ruff (E+F, line 100) in `make lint` and CI |
| Consistent naming/errors/logging | Pass — consistent stdlib style, `progress.py` for CLI output |
| Type hints on public interfaces | **Gap (Informational)** — essentially none; repo-consistent, docstrings carry the contracts; adding them wholesale is churn, not lean-ness |
| Tests for critical modules, one command | Pass — 13 hermetic modules, `python -m unittest discover tests`, run in CI |
| CI / documented local check | Pass — `.github/workflows/ci.yml`: compileall + ruff + unittest |

**[Low]** One more orientation gap: `docs/EVALKIT_IMPLEMENTATION_PLAN.md` (797 lines) still reads as a forward-looking build spec ("an implementing agent should be able to build the framework from it") for a framework that now exists, and `docs/ANALYTICS_TEST_PLAN.md:3` still opens with "Not yet run." An agent landing on either doc can't tell plan from reality. — Fix: one status banner line at the top of each ("Status: implemented in `eval/evalkit/` — see `eval/README.md`; the code is authoritative" / current run status), no content rewrite.

### 5.2 Proposed CLAUDE.md

The existing `CLAUDE.md` is strong and should stay — a full redraft would be regression, not improvement. The one needed change, ready to drop in:

````markdown
<!-- In "## Commands", after the "Eval sweep" bullet: -->
- Analytics eval (evalkit): `.venv/bin/python3 -m eval.evalkit <selfcheck|manifest|run|check|report>`
  — regression harness sweeping 61 prompt cases (suites S1–S8, `eval/cases/`)
  through `chat.run_turn` against SQL-computed ground truth. `selfcheck` and
  `manifest` need no Ollama; `run` does (sequential, one case at a time —
  never parallel). Usage + agent-orchestration guide: `eval/README.md`.
  `fast_eval.sh` is the separate, older smoke path — leave it alone.

<!-- In "## Architecture", module map, after the wrapped.py sentence: -->
`eval/evalkit/` is the analytics eval harness (stdlib-only, imports the app
in-process): `cases.py` suite loading/validation, `recipes.py` ground-truth
recipes + roster resolution, `manifest.py` corpus-fingerprinted ground truth,
`runner.py` the transcript-capturing sweep, `checks.py` deterministic
grading, `p3.py` LLM-graded checks, `report.py` the scorecard.
````

## 6. Prioritized roadmap

Each step is a discrete, safe change; the suite (`python -m unittest discover tests`) is hermetic and CI mirrors it. Steps 1–5 are near-zero-risk; 6–8 are mechanical extractions guarded by existing tests.

| # | Step | Severity | Why | Risk | Effort | Verify with |
|---|---|---|---|---|---|---|
| 1 | Move `_date_clauses` into `query_tools/_shared.py`; import in the 4 analytics modules | High | 4 verbatim copies of an index-preserving invariant | Very low (identical bodies) | ~30 min | `python -m unittest tests.test_analytics tests.test_analytics_time tests.test_analytics_social tests.test_analytics_language` + `make lint` |
| 2 | Add `ollama.chat_once_json(messages, model=None, retry_nudge=…)`; delete both `_ask_json` copies | High | Copies already drifting; retry-logic fixes won't propagate | Low (keep per-caller nudge text verbatim) | ~45 min | `python -m unittest tests.test_annotate tests.test_topics` |
| 3 | Derive `recipes._REAL` from `_shared._REAL_MESSAGE_CLAUSES` (lazy, import-light) | Medium | Eval ground truth silently diverges if the app's clause changes | Low | ~20 min | `python -m unittest tests.test_evalkit_checks` + `python3 -m eval.evalkit selfcheck` |
| 4 | Add evalkit to `CLAUDE.md` (block in §5.2) | Medium | Largest new subsystem invisible to agent orientation | None (docs) | ~10 min | proofread; `grep evalkit CLAUDE.md` |
| 5 | Status banners on `docs/EVALKIT_IMPLEMENTATION_PLAN.md` + `docs/ANALYTICS_TEST_PLAN.md` | Low | Plan docs read as current build specs | None (docs) | ~10 min | proofread |
| 6 | Add `db.connect(path, vec=, rows=)`; five call sites delegate, patch seams untouched | Low | Drift-shaped near-duplicates | Low | ~45 min | full suite (touches five modules) |
| 7 | Split roster resolution out of `recipes.py` into `eval/evalkit/roster.py` | Medium | 987-line two-concern module, MI B | Low-medium (import updates in `manifest.py`, `__main__.py`) | ~1 h | `python3 -m eval.evalkit selfcheck` + `python -m unittest tests.test_evalkit_checks` |
| 8 | Extract per-section helpers from `wrapped.build_sections`, `get_response_dynamics`, `get_group_dynamics` | Low | Three F-rated functions (42–46) are blind-edit territory | Medium (intricate logic; strictly mechanical moves only) | ~2–3 h | `python -m unittest tests.test_wrapped tests.test_analytics_time tests.test_analytics_language` |

## Appendix — re-measurement commands

```bash
# LOC + largest files
find . -name '*.py' -not -path './.claude/*' -not -path './.venv/*' | xargs wc -l | sort -rn | head -10
# dead code / lint
vulture . --min-confidence 60 --exclude '.claude,.venv'
ruff check .
# duplication
pylint --disable=all --enable=duplicate-code --min-similarity-lines=6 query_tools eval/evalkit *.py
grep -rn 'def _date_clauses\|def _ask_json' --include='*.py' . --exclude-dir=.claude
# complexity
radon cc . -s -a --exclude '.claude/*,.venv/*' -n D
# deps
deptry . --exclude '.claude|.venv|tests'
```
