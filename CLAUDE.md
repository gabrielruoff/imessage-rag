# textmessagething

Makes iMessage history queryable by a local LLM: ingest macOS `chat.db` into a
normalized SQLite store (`imessage_analysis.db`) with FTS5 keyword search and
sqlite-vec semantic search, exposed to an Ollama-hosted chat model as tool
calls. Retrieval, not context-stuffing — the corpus never enters the prompt.

Full design history, schema, rationale, and data forensics:
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Commands

- Install: `make install` (needs Homebrew Python — python.org builds lack the
  sqlite loadable-extension support sqlite-vec requires)
- Ingest: `.venv/bin/python3 ingest.py [--no-embed] [--since-last] [--senders=...]`
- Chat CLI: `.venv/bin/python3 chat.py`
- Test: `.venv/bin/python3 -m unittest discover tests` (hermetic — no Ollama or real `chat.db`)
- Lint: `make lint` (ruff, line-length 100, E+F)
- Reset analysis DB (keeps schema): `make reset-db`

## Architecture

`ingest.py` snapshots the live `chat.db`, decodes `attributedBody` blobs, and
upserts into `imessage_analysis.db` (append-only; re-runs never delete), then
calls `embed.py` (sqlite-vec table + batch embeddings). `query_tools.py` holds
the nine retrieval functions the LLM calls; `tool_schemas.py` their JSON
schemas; `chat.py` the CLI tool-calling loop over Ollama (chat and embeddings
both default to `gemma4:26b-a4b-it-qat`); `ollama.py` all Ollama HTTP
(embed_one / embed_batch / chat_stream); `config.py` all env-driven settings
(defaults work with no `.env`; see `.env.example`); `db.py` sqlite-vec
extension loading.

## Gotchas (load-bearing)

- `messages.date` strings are unlabeled UTC ISO; compare as plain strings and
  never wrap the column in SQL date functions (kills `idx_messages_chat_date`).
  Date bounds go through `query_tools._normalize_date_bound` (half-open upper
  bound, tz-aware input converted to UTC).
- gemma4's Ollama renderer honors only the FIRST system message; fold
  mid-conversation steering into it (see `chat.run_turn`). Withholding the
  tool schemas is the only reliable way to stop a tool call — prompting alone
  doesn't.
- `chat.db` is live (WAL sidecars): only ever read the snapshot
  (`ingest.snapshot_source`), never the original.
- Ingestion is append-only by design; `--senders` scopes what a run reads, it
  never deletes previously ingested rows.
- DB paths resolve relative to `config.py`'s directory, not the CWD.
- Don't modify or read files in `tests/` unless the task is about tests.

## Conventions

- Tool functions: id params accept a scalar or a list (`_as_id_list`); message
  `text` is truncated to `TEXT_PREVIEW_CHARS`, recoverable via
  `get_full_messages`.
- New tool = function in `query_tools.py` + schema in `tool_schemas.py` +
  entries in `chat.py`'s `TOOL_FUNCTIONS` and `SYSTEM_PROMPT` tool list (keep
  all four in sync).
- All tunables (models, hosts, timeouts, batch sizes) live in `config.py` /
  `.env.example`; never hardcode them elsewhere.
- Runtime deps go in `requirements.txt`, dev-only tools in
  `requirements-dev.txt` (both pinned).
