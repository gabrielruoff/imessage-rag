# textmessagething

Makes iMessage history queryable by a local LLM: ingest macOS `chat.db` into a
normalized SQLite store (`imessage_analysis.db`) with FTS5 keyword search and
sqlite-vec semantic search over conversation chunks, exposed to an
Ollama-hosted chat model as tool calls. Retrieval, not context-stuffing — the
corpus never enters the prompt.

Full design history, schema, rationale, and data forensics:
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Commands

- Install: `make install` (needs Homebrew Python — python.org builds lack the
  sqlite loadable-extension support sqlite-vec requires)
- Ingest: `.venv/bin/python3 ingest.py [--no-embed] [--since-last] [--senders=...]`
- Chat CLI: `.venv/bin/python3 chat.py` (interactive REPL), or
  `chat.py -p "..."` for a one-shot answer (prints only the reply; exit 1 if
  Ollama is unreachable)
- Test: `.venv/bin/python3 -m unittest discover tests` (hermetic — no Ollama or real `chat.db`).
  NEVER run tests unless your changes touched files in `tests/` or the user
  directly asks — the suite is slow and hangs the terminal. When you do run
  them, prefer a single module (`-m unittest tests.test_foo`).
- Lint: `make lint` (ruff, line-length 100, E+F)
- Reset analysis DB (keeps schema): `make reset-db`

## Architecture

`ingest.py` runs the source pipeline: snapshot the live `chat.db`, decode
`attributedBody` blobs, upsert into `imessage_analysis.db` (append-only —
re-runs never delete source-derived rows), build FTS5. `chunking.py` builds
the derived artifacts: group each chat's messages into conversation chunks
(split on a > `CHUNK_GAP_MINUTES` silence or at `CHUNK_MAX_CHARS`, with
`CHUNK_OVERLAP_MESSAGES` carried across size splits; rendered with a
participants/date header and speaker labels), then embed the chunks into the
sqlite-vec `chunk_vectors` table (resumable — only chunks without a vector
are embedded); `ingest.py` runs and re-exports both stages. Messages are
embedded as chunks, not individually: most messages ("ok", "lol") carry no
retrievable meaning alone, and chunking cuts vector count ~5–10x. Each vector
carries `chat_id`/`first_day`/`last_day` as vec0 metadata columns, so scoped
semantic search filters *inside* the KNN scan; distance is cosine.

Models (both local, via Ollama): chat = `gemma4:26b-a4b-it-qat`; embeddings =
`nomic-embed-text` — Ollama 0.32.0 refuses `/api/embed` for generation
models, so the chat model cannot double as the embedder. Nomic's task
prefixes (`search_document: ` / `search_query: `) are applied by default and
the embedding dimension is auto-detected at ingest time, so switching
embedders is a `.env` change + `reset_db.py` + re-ingest.

Module map: `query_tools/` holds the thirteen retrieval functions the LLM
calls — `messages.py` (message-level SQL/FTS search, context expansion, full
text, reactions), `chunks.py` (chunk-level semantic search, hybrid
FTS+semantic fused with reciprocal-rank fusion, more-like-this, chunk → its
messages), `directory.py` (summary stats; name/handle/chat resolution),
`_shared.py` (connection, result caps, date-bound and scope resolution), all
re-exported from `__init__.py` so `import query_tools` is the only import
callers need (`DB_PATH`/`embed_texts` are patchable package attributes,
resolved through the package at call time); `tool_schemas.py` their JSON
schemas; `chat.py` the CLI tool-calling loop over Ollama; `ollama.py` the
Ollama chat-streaming HTTP; `embeddings.py` the shared embedding client
(doc/query prefixes, `EMBED_DIM` override, `detect_embed_dim`) used by both
ingest (documents) and query_tools (queries) so both sides embed identically;
`config.py` all env-driven settings (defaults work with no `.env`; see
`.env.example`); `db.py` shared sqlite helpers (`load_vec_extension`,
`CONTACT_LABEL_SQL`, `iso_to_day`, `get_state`/`set_state`).

## Schema (imessage_analysis.db)

```
chats(id, chat_identifier, display_name, is_group, service)
contacts(id, source_uniqueid, first_name, last_name, nickname, organization)
handles(id, handle_identifier, contact_id, service)
                                  -- contact_id -> contacts.id (nullable);
                                  -- names live on contacts, never here
chat_participants(chat_id, handle_id)
messages(id, guid, chat_id, handle_id, is_from_me, text, decode_failed,
         date, date_read, date_delivered, item_type, is_emote, has_attachment,
         chunk_id)                -- chunk_id -> chunks.id (NULL = not chunked:
                                  -- dateless/system rows, or chunking pending)
reactions(id, target_message_guid, chat_id, handle_id, is_from_me,
          associated_message_type, date)
attachments(id, message_id, filename, mime_type, transfer_name)
chunks(id, chat_id, first_message_id, last_message_id, first_date, last_date,
       n_messages, text)          -- conversation windows; id AUTOINCREMENT so a
                                  -- rebuilt tail chunk never reuses an id
ingest_state(key, value)          -- last_message_rowid for --since-last, plus
                                  -- embed_model/embed_dim guarding chunk_vectors
messages_fts                      -- FTS5 external-content table over messages.text
chunk_vectors                     -- sqlite-vec vec0 table: chunk_id ->
                                  -- FLOAT[detected dim] distance_metric=cosine,
                                  -- + chat_id/first_day/last_day METADATA columns
```

## Gotchas (load-bearing)

- `messages.date` strings are unlabeled UTC ISO; compare as plain strings and
  never wrap the column in SQL date functions (kills `idx_messages_chat_date`).
  Date bounds go through `query_tools._normalize_date_bound` (half-open upper
  bound, tz-aware input converted to UTC). Chunk date filters and the
  `first_day`/`last_day` metadata writer must both use `db.iso_to_day`.
- Never mix embedding spaces: `chunk_vectors` is dimensioned for — and guarded
  by — the `embed_model`/`embed_dim` recorded in `ingest_state`; `ingest.py`
  refuses to embed against a mismatch. Switching `EMBED_MODEL` is always
  `reset_db.py` + re-ingest. Skipping nomic's task prefixes measurably
  degrades retrieval — set `EMBED_DOC_PREFIX`/`EMBED_QUERY_PREFIX` to `''`
  only for prefix-less models.
- Ollama fits a request by keeping the first system message + the longest
  suffix of whole messages that fits the context window: one tool result
  bigger than the window silently leaves the model with ONLY the system
  prompt (no user question), and it confabulates. So per-row tools cap their
  row count and `chat.run_turn` caps every tool result at
  `TOOL_RESULT_MAX_CHARS` (see docs/DECISIONS.md).
- gemma4's Ollama renderer honors only the FIRST system message; fold
  mid-conversation steering into it (see `chat.run_turn`). Withholding the
  tool schemas is the only reliable way to stop a tool call — prompting alone
  doesn't.
- `chat.db` is live (WAL sidecars): only ever read the snapshot
  (`ingest.snapshot_source`), never the original.
- Ingestion is append-only by design; `--senders` scopes what a run reads, it
  never deletes previously ingested rows. One carve-out: `chunks`/
  `chunk_vectors` are derived artifacts, and `--since-last` rebuilds a chat's
  tail chunk when new messages extend it — that never touches
  `messages`/`chats`/`handles`.
- DB paths resolve relative to `config.py`'s directory, not the CWD.
- Don't modify or read files in `tests/` unless the task is about tests.

## Conventions

- Tool functions: id params accept a scalar or a list (`_as_id_list`); message
  `text` is truncated to `TEXT_PREVIEW_CHARS` (chunk previews to
  `CHUNK_PREVIEW_CHARS`), recoverable via `get_full_messages` /
  `get_conversation`.
- New tool = function in the right `query_tools/` module (re-export it from
  `__init__.py`) + schema in `tool_schemas.py` (same `name` as the function —
  `chat.py` derives its dispatch table from that) + a one-line entry in
  `chat.py`'s `SYSTEM_PROMPT` tool list (keep all three in sync).
- All tunables (models, hosts, timeouts, batch sizes, chunking) live in
  `config.py` / `.env.example`; never hardcode them elsewhere.
- Runtime deps go in `requirements.txt`, dev-only tools in
  `requirements-dev.txt` (both pinned).
