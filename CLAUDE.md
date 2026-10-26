# imessage-rag

Makes iMessage history queryable by a local LLM: ingest macOS `chat.db` into a
normalized SQLite store (`imessage_analysis.db`) with FTS5 keyword search and
sqlite-vec semantic search over conversation chunks, exposed to an Ollama chat
model (or opt-in Apple Foundation Models) as tool calls. Retrieval, not
context-stuffing — the corpus never enters the prompt.

Design history and rationale: [`docs/DECISIONS.md`](docs/DECISIONS.md). Feature
runbook: `docs/DEMO.md`. Every tunable: `config.py` (the single source;
`.env.example` is a bare inventory of the same names).

## Commands

Run `make` for the full self-documented target list. The ones you need:

| Task | Command |
|---|---|
| Fresh setup | `make bootstrap` (install → ollama → source-data → ingest → doctor; idempotent, so a failed run resumes) |
| Diagnose the environment | `make doctor` |
| Ingest | `make ingest` / `python3 ingest.py --help` |
| Annotate chunks | `make annotate` · `make annotate-hybrid` (wall-clock budget) · `python3 annotate.py --help` |
| Curate dyad terms | `python3 annotate.py --terms` (a separate pipeline; `--quality` is an error there) |
| Analytical suite | `make suite` (nudges → drift → voiceprint → conflict) |
| Everything, in order | `make pipeline` (ingest → annotate → suite → topics) |
| Chat | `make chat ARGS='-p "..."'` · `python3 chat.py` |
| Year-in-review | `make wrapped ARGS='--year 2025'` |
| MCP server (optional) | `make mcp-install` once, then `make mcp` (stdio; see `docs/MCP.md`) |
| Lint | `make lint` (ruff, E+F, line-length 100) |
| Test | `make test` — **slow; run only when you touched `tests/` or were asked.** Prefer `python3 -m unittest tests.test_foo` |
| Everything CI enforces | `make check` (lint + test + compileall) |
| Analytics regression | `python3 -m eval.evalkit …` / `make eval` — see `eval/README.md` |
| Reset the analysis DB | `make reset-db` |

Only `make ingest` (`INGEST_ARGS=`), `make chat` and `make wrapped` (`ARGS=`)
forward flags. For any other flag, invoke the CLI directly. `make install` needs
a Homebrew Python — python.org builds lack the sqlite loadable-extension support
sqlite-vec requires, and install refuses an interpreter without it.

## Architecture

`ingest.py` snapshots the live `chat.db`, decodes `attributedBody`, upserts into
`imessage_analysis.db` (append-only), builds FTS5, then runs the derived stages
(`chunking.py`, `sessions.py`, `lexicon_build.py`). `chunking.py` groups each
chat's messages into conversation chunks — split on a silence or a size cap —
and embeds them into the sqlite-vec `chunk_vectors` table; messages are embedded
as chunks, not individually. `chat.py` runs a tool-calling loop over the ~32
retrieval tools in `query_tools/`.

Each LLM pipeline picks its own generation model (`CHAT_MODEL`, `ANNOTATE_MODEL`,
`TOPIC_MODEL`, `WRAPPED_MODEL`, `NUDGES_MODEL`, the `CONFLICT_*` pair); empty
inherits `CHAT_MODEL`. `EMBED_MODEL` is deliberately not per-pipeline — see the
gotcha.

| Area | Files |
|---|---|
| Entry CLIs | `chat.py` (+`chat_apple*.py` for the Apple backend), `ingest.py`, `annotate.py` (+`annotate_{distill,hybrid,sampling,events,quota,terms,rate}.py`), `topics.py`, `wrapped.py` (+`wrapped_{sections,sections_suite,html,render,visuals}.py`), `reset_db.py` |
| Ingest / derived builders | `chunking.py`, `sessions.py`, `lexicon_build.py`, `schema.py`, `function_words.py` |
| Suite pipelines (after annotate) | `loose_ends.py` (+`nudge_detectors.py`, `nudge_shared.py`), `drift.py`, `voiceprint.py`, `conflict_detect.py` (+`conflict_signals.py`, `conflict_labeling.py`) |
| Shared libs | `config.py`, `db.py`, `ollama.py`/`applefm.py`, `embeddings.py`, `progress.py`, `stats.py`, `consent.py`, `wellbeing.py`, `chat_floor.py` |
| Retrieval tools | `query_tools/` (37 schemas, 5 flag-gated); `_shared.py` owns connection/caps/date-bounds/scope/identity |
| Optional web layer | `api/` (FastAPI SSE over `chat.run_turn_events`) + `frontend/` (Vite/React) — transport only |
| Optional MCP layer | `mcp_server/` (stdio, 21 read-only tools; transport only) |
| Eval | `eval/evalkit/` (S1–S17 regression + F4 gates) |

Schema: read `schema.py` — it is the DDL and the migration in one place.
App DB (web UI only, `config.APP_DB`): `api/store.py`.

## Conventions

- **New tool** = function in the right `query_tools/` module + re-export from
  `query_tools/__init__.py` + schema in `tool_schemas.py` (same `name`) + one
  line in `chat.py`'s `SYSTEM_PROMPT` tool list. Keep all four in sync. MCP
  exposure is a separate explicit allow-list (`mcp_server.EXPOSED_TOOLS`), never
  automatic, and never for a tool that writes or relies on a turn floor. A
  contract change to an exposed tool (id shapes, envelope keys, date semantics)
  also updates `mcp_server.INSTRUCTIONS`, its hand-written usage guide.
- **All tunables live in `config.py`**; never hardcode one elsewhere. Runtime
  deps → `requirements.txt`, dev → `requirements-dev.txt`, web →
  `requirements-api.txt`, MCP → `requirements-mcp.txt`. All pinned.
- **Comments state contracts; `docs/DECISIONS.md` states history.** If you are
  writing a paragraph about why a previous approach was wrong, it goes in
  `DECISIONS.md` with a one-line pointer from the code.
- **Participant identity** — both directions have one home in
  `query_tools/_shared.py`: `member_key` (row → stable key) and `label_for_key`
  (key → display label; `contact_label` for the contact case). Never
  re-implement either; modules keep thin named wrappers, not copies.
- **Shared helpers before new copies.** `db.py` owns the model guards
  (`check_model_guard`), budget resolution (`resolve_budget_minutes`),
  float-state banking (`get_float_state`/`set_float_state`) and `like_escape`;
  `stats.py` owns the robust-stats and calendar primitives; `annotate_quota.py`
  owns apportionment. Where two same-named helpers legitimately differ, each
  says so in its docstring — read that before merging them.
- Tool functions: id params accept a scalar or a list (`_as_id_list`); message
  `text` truncates to `TEXT_PREVIEW_CHARS` (chunk previews to
  `CHUNK_PREVIEW_CHARS`), recoverable via `get_full_messages`/`get_conversation`.
  Search tools return an envelope
  (`rows`/`total_matching`/`returned`/`order`/`truncated`), not a bare list.
- `REPO_REVIEW.md` / `REPO_FIX_PLAN.md` are gitignored working files — never commit.

## Gotchas (load-bearing)

- **Dates**: `messages.date` strings are unlabeled UTC ISO. Compare as plain
  strings; never wrap the column in a SQL date function inside `WHERE` (kills
  `idx_messages_chat_date`). Bucketing in `SELECT`/`GROUP BY` over
  already-filtered rows is fine. Bounds go through
  `query_tools._normalize_date_bound` (half-open upper bound, tz-aware → UTC);
  chunk day columns through `db.iso_to_day`.
- **Never mix embedding spaces**: `chunk_vectors` is dimensioned for and guarded
  by `embed_model`/`embed_dim` in `ingest_state`. Switching `EMBED_MODEL` is
  always `reset_db.py` + re-ingest — query- and document-side vectors must come
  from one model, which is why it is not a per-pipeline knob. Dropping nomic's
  task prefixes degrades retrieval; clear them only for a prefix-less model.
- **Context window**: Ollama keeps the first system message + the longest fitting
  suffix, so one oversized tool result can leave the model with *only* the system
  prompt and it will confabulate. `chat.run_turn` caps every result at
  `TOOL_RESULT_MAX_CHARS`; per-row tools cap their row counts.
- **`think` is tri-state**: `None` = omit the flag, which is NOT `False` = send
  `think: false`. Never default the param to `None`. An effort string (`'low'`)
  is an effort level only to models in `config._EFFORT_LEVEL_MODELS`; to
  everything else it is merely truthy and buys a full reasoning pass (measured
  ~90x on the per-turn conflict labeler). `annotate` and `conflict_detect` carve
  themselves out of the `CHAT_THINK` default for that reason; an explicit env
  value always wins, and inherits — `CHAT_THINK=true` reaches the carve-outs.
  Never route one pipeline's `think` into another.
- **Model guards**: `chunk_annotations`, `conflicts` and `loose_threads`' LLM
  columns each refuse a model change while non-empty (`db.check_model_guard`).
  Switching means clearing the artifact first. A guard refuses BEFORE anything
  destructive or expensive runs, and records only from the pass that actually
  writes. `--dry-run` deliberately does not guard — the guard writes.
- **`source='distilled'` rows are low-trust**: they always have `events_json=[]`,
  and valence-dependent reads (conflict corroboration, the nudge Stage-0
  exclusion, voiceprint tone) consume `source='llm'` only and degrade honestly.
  `annotate.py --quality balanced --upgrade-distilled` (or `--quality hybrid`)
  lights them up.
- **Conflict reads have THREE states, not two**: never looked at (`labeled`
  false), looked at with nothing shippable (`labeled` true, zero signal turns —
  labeling reads message TEXT only, so tapbacks/tone/off-channel yield nothing),
  and labeled with bands. Neither of the first two may be narrated as "analyzed
  and calm"; both carry a code-printed `_epilogue` saying so.
- **Safety floors are code, never prompt.** The DV screen
  (`query_tools/conflict_dv._dv_screen`), the wellbeing floor
  (`chat_floor.TurnFloor`, constructed by all three turn-runners —
  `chat.run_turn`, `chat.run_turn_events`, `chat_apple.run_turn`), the F4 eval
  gate (`eval_gate_passed`, default-closed) and the wellbeing deny-list /
  Wrapped consent allow-list (`consent.py`) are enforced in code. An enable-flag
  gates a feature's tool *registration* or pipeline *build*, never the prompt.
  Removing the capability is the control — do not try to prompt around it. The
  MCP server runs no turn floor, so floor-dependent tools are not exposed there.
- **MCP stdio: stdout is the wire.** Nothing in `mcp_server/` may print to stdout;
  client launches use `make -s --no-print-directory -C … mcp`.
- **Never read the live `chat.db`** (WAL sidecars) — only the snapshot from
  `ingest.snapshot_source`. Ingestion is append-only; `--senders`/`--lookback`
  scope what a run reads, never what it deletes.
- **Paths resolve relative to `config.py`'s directory**, not the CWD, and
  `load_dotenv` is pinned there — a worktree never inherits a parent `.env`.
- **`make pipeline`/`suite` must stay serial** (never `-j`): they hold a SQLite
  write lock across many LLM calls and share one serialized Ollama.
- **Apple FM** has a fixed 4,096-token window, so `chat_apple.py` stages
  route→fill→exec→digest→answer sessions instead of native tool-calling, and
  Apple `format="json"` is not a guided schema (use `applefm.respond(schema=…)`
  for constrained decoding). Batch annotation there packs by chars, not
  batch size.
- **Don't read or modify `tests/`** unless the task is about tests.
