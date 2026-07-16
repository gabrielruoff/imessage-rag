# Repo Review: textmessagething

_Generated 2026-07-16. Scope: report-only, moderate refactors (restructure + dedupe). No code was changed._

## 1. Executive summary

This is an unusually healthy small repo: 1,895 source LOC across 6 modules, zero lint findings, zero dead code at vulture ≥80 confidence, zero duplicate blocks at pylint's 6-line threshold, and every file rated maintainability-A by radon. The dominant problems are not in the code — they are (a) three parallel hand-rolled Ollama HTTP clients spread across three modules, (b) a `CLAUDE.md` that has grown to 409 lines (~27KB) and now contains a visible merge scar with self-contradicting content, and (c) two files (`query_tools.py` at 806 lines, `ingest.py` at 704) past the ~500-line point where an agent can hold them comfortably in context. The target end-state: one shared Ollama client module, both big files split along their existing section boundaries, and a CLAUDE.md back under two pages with the history moved to a decisions doc.

**Top 3 actions (do these first):**
1. **Repair and slim `CLAUDE.md`** — high impact (it loads into every agent session) / low effort — merge scar at `CLAUDE.md:237–247` ("all eight" vs "all nine" `query_tools` functions in the same paragraph, the phrase "The model name, host, timeouts," appears twice mid-sentence).
2. **Consolidate the three Ollama HTTP clients into one `ollama.py`** — high impact (classic edit-one-copy-miss-the-others hazard) / medium effort — `query_tools.py:266–276`, `ingest.py:539–563`, `chat.py:103–156`.
3. **Split `query_tools.py` (806 LOC) and `ingest.py` (704 LOC)** along boundaries already marked by section comments — medium impact / low risk — `TOOL_DEFINITIONS` → `tool_schemas.py`; embedding section → `embed.py`.

## 2. Metrics snapshot

| Metric | Current | Notes |
|---|---|---|
| Python files / total LOC | 8 files / 2,422 LOC (1,895 source + 527 tests) | small, flat layout |
| Largest files (LOC) | `query_tools.py` 806, `ingest.py` 704 | both split candidates |
| Unused symbols (vulture ≥80) | 0 | |
| Duplicate blocks (pylint, ≥6 lines) | 0 (rated 10.00/10) | duplication that exists is below this threshold (see 3.1) |
| Unused deps (deptry) | 0 real (2 false flags: `python-dotenv` name-mapping, `ruff` is a dev tool) | |
| Undeclared deps (deptry) | 0 real (1 false flag: `dotenv` ↔ `python-dotenv`) | |
| Complexity (radon) | all files MI grade A; worst function `chat.py::ollama_chat_stream` C(11) | |
| Files >500 LOC | 2 | |
| CLAUDE.md | 409 lines / ~27KB | loaded every session; contains a merge scar |

_Re-run the commands in the appendix after acting to measure progress._

## 3. Bloat findings

### 3.1 Duplication

**Three parallel Ollama HTTP clients, three different styles.** This is the repo's one real single-source-of-truth violation, and it's the exact pattern where a future agent edits one copy and misses the others:

| Location | Endpoint | HTTP library | Timeout |
|---|---|---|---|
| `query_tools.py:266` `_ollama_embed` | `/api/embeddings` (single-item) | `urllib.request` | **hardcoded `timeout=30`** (`query_tools.py:272`) |
| `ingest.py:539–563` `_ollama_connection` + `ollama_embed_batch` | `/api/embed` (batch) | `http.client`, persistent per-thread | `OLLAMA_TIMEOUT` from config |
| `chat.py:103` `ollama_chat_stream` | `/api/chat` (streaming) | `http.client`, per-call | `CHAT_TIMEOUT` from config |

Why it matters: three request/serialize/error-handle implementations for the same server; the hardcoded 30s in `query_tools.py` already drifted from the configurable `OLLAMA_TIMEOUT=60` the others honor. Moderate fix: a single `ollama.py` exposing `embed_one()`, `embed_batch()`, and `chat_stream()` (keep `chat_stream`'s rendering callback out of it — see 4.2), each keeping its current endpoint and semantics so behavior is preserved. Optional follow-up (mark: slight behavior change, out of scope for the pure-dedupe pass): `/api/embed` accepts a single-item input, so `embed_one` could delegate to `embed_batch` and the `OLLAMA_EMBEDDINGS_PATH` config knob could retire.

**Duplicated tool documentation in `chat.py`'s `SYSTEM_PROMPT`** (`chat.py:33–53`) restates one-line summaries of all nine tools that `TOOL_DEFINITIONS` (`query_tools.py:625–806`) already describes. This is deliberate prompt engineering, not accidental drift — leave it, but note that the two lists must be kept in sync when tools change (worth a one-line comment or a CLAUDE.md convention entry).

### 3.2 Dead & speculative code

None found. vulture (≥80 confidence): 0 symbols. ruff F401/F811/F841: clean. No commented-out blocks, no `if False:`, no single-implementation abstractions. The one wrapper-that-wraps-one-thing is `reset-db.sh` (3 lines, only invoked by the Makefile's `reset-db` target) — the Makefile can call `.venv/bin/python3 reset_db.py -f` directly and the script can be deleted.

### 3.3 Over-engineering

Essentially none — the helper layer in `query_tools.py` (`_as_id_list`, `_resolve_scope`, `_stats_filters`, etc.) is all used by 2+ callers. One complexity hotspot: `chat.py::ollama_chat_stream` (C 11) mixes HTTP transport, NDJSON stream parsing, and `rich.Live` markdown rendering in one function. Moderate fix, best done as part of the `ollama.py` extraction: transport/parsing yields content chunks; the caller in `chat.py` owns the `Live` rendering.

### 3.4 Dependency bloat

Minimal and healthy: 4 pinned deps, all used. Two notes:
- `ruff==0.15.21` is a dev tool pinned in the runtime `requirements.txt` (installed by all three CI jobs and every `make install`). Moderate fix: split a `requirements-dev.txt` (or move deps into `pyproject.toml` with a `dev` optional group) so runtime installs stay minimal.
- deptry's 3 findings are all false positives (`dotenv` ↔ `python-dotenv` module/package name mapping, plus the `ruff` point above); a `[tool.deptry.package_module_name_map]` entry in `pyproject.toml` would silence them if deptry joins CI.

### 3.5 Checked-in noise

None. `.gitignore` correctly excludes `source_data/`, `data/`, `imessage_analysis.db`, `.venv/`, `.env`, `__pycache__/`. `git ls-files` shows only source, docs, config, and the skill files.

**One drift bug in a committed template:** `.env.example` documents `# EMBED_MODEL=gemma4:26b-a4b-it-qat`, but the real default in `config.py:47` is `gemma4:26b-a4b-it-qat` (the chat model, not the embed model, is gemma4). Anyone uncommenting that line as-is would point embeddings at a 26B chat model. Fix the comment to `# EMBED_MODEL=gemma4:26b-a4b-it-qat`.

**One duplicated CI definition:** `.github/workflows/test.yml` (on `pull_request`) is a byte-for-byte duplicate of the `test` job already in `.github/workflows/ci.yml` (on `push`) — same steps, same Python version. CLAUDE.md documents only `ci.yml`. Moderate fix: add `pull_request` to `ci.yml`'s `on:` block and delete `test.yml`, so the job definition exists once.

## 4. Structural assessment

### 4.1 Proposed structure

Flat layout is right for a 6-module tool; src-layout would be ceremony without payoff here. The moves are two splits and one extraction:

```
Current                          Target
├── chat.py          (237)       ├── chat.py          (~190: CLI + tool loop, thin)
├── config.py         (55)       ├── config.py
├── db.py             (10)       ├── db.py
├── ingest.py        (704)       ├── ollama.py        (NEW ~90: embed_one/embed_batch/chat_stream)
├── query_tools.py   (806)       ├── ingest.py        (~570: snapshot/decode/schema/ingest)
├── reset_db.py       (83)       ├── embed.py         (NEW ~100: build_vec_table + embed_messages)
├── reset-db.sh                  ├── query_tools.py   (~590: the 9 retrieval functions + helpers)
└── tests/                       ├── tool_schemas.py  (NEW ~200: TOOL_DEFINITIONS + schema fragments)
                                 ├── reset_db.py
                                 └── tests/            (unchanged)
```

### 4.2 Files to split (with the specific split)

- **`query_tools.py` (806 → ~590 + ~200).** Lines 605–806 are already a self-contained section ("Tool schemas, for wiring into Claude's tool_use API"): `_ID_OR_LIST_SCHEMA`, `_DATE_PARAM_DESC`, `TOOL_DEFINITIONS`. Move them to `tool_schemas.py`; `chat.py:27` changes its import from `query_tools` to `tool_schemas`. Zero logic moves; docstring-only churn. Payoff: an agent editing retrieval SQL no longer pages through 200 lines of JSON schema, and an agent editing tool descriptions never touches SQL.
- **`ingest.py` (704 → ~570 + ~100).** Lines 532–632 are already fenced by the comment `# Embeddings (sqlite-vec + Ollama)`: `_thread_local`, `_ollama_connection`, `ollama_embed_batch`, `build_vec_table`, `embed_messages`. Move to `embed.py` (with the HTTP part landing in `ollama.py`, below); `ingest.main` imports `build_vec_table`/`embed_messages` from it. `reset_db.py` and tests are unaffected.
- **`chat.py::ollama_chat_stream` (chat.py:103–156, C 11).** Split transport from rendering when it moves into `ollama.py`: `chat_stream()` yields `(content_piece, tool_calls, done)` events; `chat.py` keeps the `rich.Live` handling. Behavior-identical, and the one C-graded function in the repo disappears.

### 4.3 Modules to consolidate / rename / relocate

- **Create `ollama.py`** consolidating the three HTTP clients (see 3.1). It should read all connectivity constants from `config.py`, replacing the hardcoded `timeout=30` at `query_tools.py:272` with `OLLAMA_TIMEOUT` — flag this in the PR as the one deliberate behavior tweak (30s → 60s default) or preserve 30 via a new constant if strictness matters.
- **Delete `reset-db.sh`**; point the Makefile `reset-db` target at `.venv/bin/python3 reset_db.py -f`.
- **Merge `.github/workflows/test.yml` into `ci.yml`** (add `pull_request:` trigger) and delete it.

## 5. Claude Code fitness

### 5.1 Checklist results

| Item | Result |
|---|---|
| `CLAUDE.md` exists and is current | **Gap** — exists and is rich, but 409 lines (~27KB, vs. the ~1–2 page guidance), and contains a merge scar with contradictory content (`CLAUDE.md:237–247`: "all eight" then "all nine" `query_tools` functions; a half-sentence duplicated) |
| `README.md` purpose + quickstart | Pass |
| Build/test/run commands discoverable and work | Pass (`make install/lint/reset-db`, `python -m unittest discover tests`, documented in both README and CLAUDE.md) |
| Directory/module names predict contents | Pass (`ingest`, `query_tools`, `chat`, `config`, `reset_db`) |
| No grab-bag utils modules | Pass (`db.py` is 10 lines with one clearly-scoped function) |
| Public API visible without spelunking | Pass for a flat script layout |
| Few/no files >500 lines | **Gap** — `query_tools.py` 806, `ingest.py` 704 (see 4.2) |
| Functions mostly <50 lines | Mostly pass — `get_chat_summary_stats` (~84), `embed_messages` (~57), `ollama_chat_stream` (~54) are the outliers; only the last needs action |
| One concern per module | **Gap** — `ingest.py` bundles ETL + embedding; `query_tools.py` bundles retrieval + tool schemas |
| No duplicated logic | **Gap** — three Ollama clients (3.1) |
| Config/constants defined once | **Near-pass** — `config.py` is the single source, but `query_tools.py:272` hardcodes a timeout and `.env.example` misdocuments `EMBED_MODEL` |
| Fixtures/helpers defined once | Pass |
| Enforced linter | Pass (ruff in CI); **no formatter** — consider `ruff format` in CI so future AI edits don't drift style |
| Consistent naming/error handling | Pass |
| Type hints on public interfaces | **Gap (minor)** — sporadic (`search_messages_fts(query: str, ...)` but untyped ids/returns); full typing is beyond moderate scope, but new/moved modules should be typed |
| Tests exist, one command | Pass (`python -m unittest discover tests`, hermetic) |
| CI invocable check | Pass — though `_normalize_date_bound`/`_date_bound_step` (the subtlest logic in the repo, per CLAUDE.md's own warnings) have no direct test references, and `search_messages_vec`/`search_contact_by_handle`/`get_full_messages` appear in no test file (grep) |

### 5.2 Proposed CLAUDE.md

The current file's depth is genuinely valuable — the fix is relocation, not deletion. Move the narrative (decode research, timestamp deep-dive, duplicate-contact forensics, decision rationale) to `docs/DECISIONS.md`, keep a lean always-loaded core:

```markdown
# textmessagething

Makes iMessage history queryable by a local LLM: ingest macOS `chat.db` into a
normalized SQLite store (`imessage_analysis.db`) with FTS5 keyword search and
sqlite-vec semantic search, exposed to an Ollama-hosted chat model as tool calls.
Retrieval, not context-stuffing — the corpus never enters the prompt.

Full design history, schema rationale, and data forensics: `docs/DECISIONS.md`.

## Commands

- Install: `make install` (needs Homebrew Python — python.org builds lack sqlite
  loadable-extension support required by sqlite-vec)
- Ingest: `.venv/bin/python3 ingest.py [--no-embed] [--since-last] [--senders=...]`
- Chat CLI: `.venv/bin/python3 chat.py`
- Test: `.venv/bin/python3 -m unittest discover tests` (hermetic, no Ollama/chat.db)
- Lint: `make lint` (ruff, line-length 100, E+F)
- Reset analysis DB (keeps schema): `make reset-db`

## Architecture

`ingest.py` snapshots the live `chat.db`, decodes `attributedBody` blobs, and
upserts into `imessage_analysis.db` (append-only; re-runs never delete).
`query_tools.py` holds the nine retrieval functions the LLM calls;
`tool_schemas.py` their JSON schemas; `chat.py` the CLI tool-calling loop over
Ollama; `ollama.py` all Ollama HTTP; `config.py` all env-driven settings
(defaults work with no `.env`); `db.py` sqlite-vec loading.

## Gotchas (load-bearing)

- `message.date` strings are unlabeled UTC; compare as strings, never wrap the
  column in SQL date functions (kills `idx_messages_chat_date`). Date bounds go
  through `_normalize_date_bound` (half-open upper bound).
- gemma4's Ollama renderer honors only the FIRST system message; fold
  mid-conversation steering into it (see `run_turn`). Withhold tool schemas to
  actually stop tool calls — prompting alone doesn't.
- `chat.db` is live (WAL sidecars): only ever read the snapshot.
- Ingestion is append-only by design; `--senders` scopes reads, never deletes.
- DB paths resolve relative to `config.py`'s directory, not CWD.
- Don't modify or read `tests/` unless the task is about tests.

## Conventions

- Tool functions: id params accept scalar or list (`_as_id_list`); results
  truncated to `TEXT_PREVIEW_CHARS`, recoverable via `get_full_messages`.
- New tool = function in `query_tools.py` + schema in `tool_schemas.py` +
  entry in `chat.py`'s `TOOL_FUNCTIONS` and `SYSTEM_PROMPT` list (keep in sync).
- All tunables live in `config.py` / `.env.example`; never hardcode timeouts or
  model names elsewhere.
```

## 6. Prioritized roadmap

Each step is a discrete PR that leaves the repo green.

| # | Step | Why | Risk | Effort | Verify with |
|---|---|---|---|---|---|
| 1 | Fix `CLAUDE.md:237–247` merge scar; fix `.env.example` `EMBED_MODEL` comment | Corrects actively-wrong docs agents read every session | None (docs only) | 15 min | proofread; `make lint` |
| 2 | Merge `test.yml` into `ci.yml` (`on: push, pull_request`), delete `test.yml`; delete `reset-db.sh` and update Makefile | One definition per CI job / target | Low | 15 min | open a test PR, confirm jobs run; `make reset-db` |
| 3 | Extract `TOOL_DEFINITIONS` + schema fragments (`query_tools.py:605–806`) → `tool_schemas.py`; update `chat.py` import | 806→~590 lines; schemas and SQL become independently editable | Low (move-only) | 30 min | `python -m unittest discover tests`; run `chat.py` and make one tool call |
| 4 | Create `ollama.py` (embed_one/embed_batch/chat_stream); rewire `query_tools.py`, `ingest.py`, `chat.py`; replace hardcoded `timeout=30` with `OLLAMA_TIMEOUT` (call out in PR) | Kills the triple-client duplication; fixes config drift | Medium (touches live HTTP paths) | 2–3 h | tests + a real `chat.py` session incl. `search_messages_vec`; `ingest.py --since-last` embed run |
| 5 | Move embedding section (`ingest.py:532–632`) → `embed.py` | 704→~570 lines; ETL vs. embedding concerns separated | Low (move-only) | 30 min | tests; `ingest.py --no-embed` then full run |
| 6 | Slim `CLAUDE.md` per §5.2; move narrative to `docs/DECISIONS.md` | ~27KB → ~3KB of every-session context | None | 1 h | proofread |
| 7 | Split `requirements-dev.txt` (ruff); optionally add `ruff format --check` to CI | Lean runtime installs; style stability for future AI edits | Low | 30 min | CI green |
| 8 | (Optional) add direct tests for `_normalize_date_bound`/`_date_bound_step` | Subtlest logic in repo, currently untested | None | 1 h | new tests pass |

## 7. Appendix: commands used

```bash
find . -type f -name '*.py' -not -path './.venv/*' -not -path './.claude/*' | xargs wc -l | sort -n
python3 -m venv /tmp/leaner-venv && /tmp/leaner-venv/bin/pip install ruff vulture radon deptry pylint
ruff check . --statistics
ruff check . --select F401,F811,F841
vulture . --min-confidence 80 --exclude .claude
radon cc . -s -n C --exclude ".claude/*"
radon mi . -s --exclude ".claude/*"
radon raw chat.py config.py db.py ingest.py query_tools.py reset_db.py -s
deptry . --exclude '.claude|leaner-venv'
pylint --disable=all --enable=duplicate-code --min-similarity-lines=6 chat.py config.py db.py ingest.py query_tools.py reset_db.py
git ls-files
grep -n "eight\|nine" CLAUDE.md
grep -oE 'search_messages_vec|search_contact_by_handle|get_full_messages|_normalize_date_bound|_date_bound_step|get_message_reactions' tests/*.py | sort | uniq -c
```
