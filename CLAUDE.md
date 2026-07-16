# textmessagething

## Goal

Make iMessage history queryable and analyzable by an LLM — both casual chat
("what did we say about the trip?") and analysis (sentiment/trend/topic
questions over time), for conversations that may scale to hundreds of MB or
GB. The data cannot be pasted into a chat window or read wholesale by an LLM
at query time — it needs to be parsed into a structured store that supports
targeted retrieval.

## Data sources

- `chat.db` — the real macOS Messages database (found already present in
  this project folder, ~500MB, ~382k messages across ~1.1k chats / 1,942
  handles). This is the primary/preferred source — richer and more reliable
  than parsed `.txt` exports.
- `address_book.abcddb` (optional, project-local path) — a copy of macOS's
  Contacts store (normally at `~/Library/Application
  Support/AddressBook/Sources/*/AddressBook-v22.abcddb`). `chat.db`'s
  `handle` table only has raw phone numbers/emails, no display names —
  `ingest.py::build_contact_lookup` reads this file (if present at
  `./address_book.abcddb`) to attach `first_name`/`last_name`/`nickname`/
  `organization` to each row in `handles`, matching by normalized phone
  (last-10-digits) or lowercased email. Best-effort: if the file isn't
  present, these columns are just left NULL.
- `data/<sender>.txt` — a sample export from the `imessage-exporter` CLI tool
  (https://github.com/ReagentX/imessage-exporter), representing the
  alternative/portable input format for cases where someone only has a `.txt`
  export rather than direct access to `chat.db`. Format: blank-line delimited
  blocks of `timestamp (+ optional read-receipt)` / `sender ("Me" or a
  phone/short-code)` / `body (can be multiline, or an attachment file path)`.
  **The `.txt` parser was not built** — we pivoted to ingesting `chat.db`
  directly once we found it in the project. If a `.txt`-only path is ever
  needed again, this format is the one to target.

## Architecture decisions (with rationale)

- **Language: Python 3.** Ingestion is a one-time/periodic batch ETL job, not
  a latency-sensitive service — Python's `sqlite3` + ecosystem support
  outweighed any need for Rust-level speed.
- **Storage: SQLite**, a new file (`imessage_analysis.db`) kept separate from
  the live `chat.db`. One file, three capabilities layered on top of each
  other: normalized tables, an FTS5 virtual table for keyword search, and a
  `sqlite-vec` virtual table for embeddings/semantic search.
- **Append-only / upsert store — ingestion never deletes.** Every run only
  adds to `imessage_analysis.db`: all writes are `INSERT OR REPLACE` /
  `INSERT OR IGNORE` idempotent upserts, so re-running (with the same or a
  different `--senders` filter) accumulates the union of everything ingested
  and never removes prior rows. Foreign keys are enforced at runtime
  (`PRAGMA foreign_keys = ON` in `ingest.py::main`, backing the
  `handles.contact_id -> contacts(id)` reference); a handle with no matching
  ingested contact gets a NULL `contact_id` rather than a dangling one.
- **Retrieval, not context-stuffing.** The LLM is meant to query this DB via
  tool calls (SQL filter / FTS5 / vector search) rather than have the whole
  corpus loaded into context. This is what lets the approach scale to
  arbitrarily large exports.
- **Vector search: `sqlite-vec`**, not a dedicated vector store (Pinecone,
  Weaviate, etc.). Rationale: this is a single person's message history, not
  a multi-tenant/distributed workload — `sqlite-vec` keeps everything in one
  file with no extra service to run, and its brute-force/ANN search is fast
  enough at this scale (low millions of vectors). A dedicated vector store
  would only make sense if the scope grew to many users' histories or
  required massive concurrent ANN search.
- **LLM stack is fully local**: generation via Ollama running
  `gemma4:26b-a4b-it-qat` (confirmed real model: Gemma 4, 25.2B params / 4B
  active MoE, QAT-quantized, ~15GB RAM, 64K–256K context, agentic-tuned).
  Embeddings via Ollama running `nomic-embed-text` (768-dim, 8192-token
  context, small footprint, most proven `sqlite-vec` integration). Chose
  local embeddings specifically because generation is already local and
  there's no reason to send personal message text to an external API.
  (Alternative considered: `embeddinggemma`, same vendor family as the
  generator — reasonable if stack consistency matters more than proven
  integration track record.) None of this is hardcoded: the Ollama host/port,
  endpoint paths, model names, timeouts, and embedding/tool tunables are all
  read from the environment via `config.py` (see the `config.py` file entry
  below), with the values above as defaults.

## Key technical findings

- **`message.text` is NULL for ~99% of rows** (~379.5k of ~382k in this
  data). The actual text lives in the `attributedBody` BLOB as an
  `NSAttributedString`/`NSKeyedArchiver` typedstream — there's no native
  Python decoder for this.
- **Decoder used: the `streamtyped` byte-pattern approach** from
  `imessage-exporter`'s Rust source
  (`imessage-database/src/util/streamtyped.rs`), ported to Python in
  `ingest.py::decode_streamtyped`. Technique: find the start marker
  `0x01 0x2B`, find the end marker `0x86 0x84`, decode the UTF-8 bytes
  between them, then drop a fixed prefix (1 char if clean UTF-8 decode, 3 if
  it had to lossy-decode). This only recovers plain text, not styling or
  inline attachment position — that would require the fuller typedstream
  deserializer (`imessage-exporter`'s `typedstream.rs` + their `crabstep`
  crate), which we deliberately did not port since we only need body text.
  **Result on real data: 0 decode failures across all ~350k non-reaction
  messages.**
- **Timestamps are nanoseconds since 2001-01-01** ("Mac absolute time"), not
  Unix epoch — converted via `apple_ns_to_iso()` in `ingest.py`.
- **No single table has a full message record** — had to join `message` +
  `handle` (sender) + `chat_message_join` (message→chat, not a direct FK on
  `message`) + `chat_handle_join` (chat participants, for group threads) +
  `message_attachment_join` + `attachment`.
- **Reactions/tapbacks are stored as rows in the `message` table**
  (`associated_message_type != 0`, ~32k of them here) pointing at a target
  message via `associated_message_guid` — these are not real messages and
  are split into their own `reactions` table during ingestion rather than
  mixed into the message stream.
- **`chat.db` is live** (has `-wal`/`-shm` sidecars since Messages.app may
  still be writing to it) — `ingest.py::snapshot_source` copies it (plus
  sidecars) to `chat_snapshot.db` and checkpoints the WAL there before
  anything reads from it, so the original is never touched.
- **`address_book.abcddb` contains genuine duplicate contact cards** — e.g.
  three separate `ZABCDRECORD` rows (distinct `Z_PK`) all own the exact same
  phone number, one pair with identical first/last name and no last name at
  all on another duplicate pair. These are unmerged contact cards in the
  real data (a sync artifact), not a join/query artifact — confirmed by
  checking `ZLINKID` (Apple's own merged-contact-link column), which is
  `NULL` on all of them, meaning Apple's own dedup never linked them either.
  There is no reliable ID to auto-merge on, so no merging is attempted:
  `ingest.py::build_contact_lookup` keeps one `contacts` row per
  `ZABCDRECORD`, and each `handles` row's `contact_id` resolves to whichever
  contact record happens to own that specific phone number/email (first
  match wins if more than one contact owns the same number, same as before
  this table existed).

## Files

- **`config.py`** — single source of runtime configuration, imported by
  `ingest.py`, `query_tools.py`, and `chat.py`. Reads every setting from the
  environment (calling `python-dotenv`'s `load_dotenv()` first, so a local
  `.env` is picked up automatically) and exposes them as module constants, each
  with a working default so nothing *has* to be set. Groups: DB paths
  (`SOURCE_DB`, `SNAPSHOT_DB`, `TARGET_DB`, `ADDRESS_BOOK_PATH`), Ollama
  connectivity (`OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_EMBED_PATH`,
  `OLLAMA_EMBEDDINGS_PATH`, `OLLAMA_CHAT_PATH`, `OLLAMA_TIMEOUT`,
  `CHAT_TIMEOUT`), models (`EMBED_MODEL` = `nomic-embed-text`, `CHAT_MODEL` =
  `gemma4:26b-a4b-it-qat`), and tunables (`EMBED_DIM` = 768, `EMBED_BATCH_SIZE`,
  `EMBED_MAX_WORKERS`, `MAX_TOOL_ITERATIONS`).
- **`.env.example` / `.env`** — `.env.example` is the committed template: every
  variable from `config.py` listed and commented out with its default, so it
  doubles as documentation of what's overridable. Copy it to `.env` and
  uncomment what you want to change; `.env` itself is git-ignored (may hold
  machine-specific paths/hosts). `.env` is optional — the defaults in
  `config.py` work without it.
- **`ingest.py`** — full pipeline: snapshot → decode → normalize into
  `imessage_analysis.db` → build FTS5 → build `sqlite-vec` table → embed via
  local Ollama. Flags: `--no-embed` (skip embedding, e.g. if Ollama isn't
  running), `--since-last` (incremental — only ingest messages with
  `ROWID` greater than the last recorded run, tracked in the `ingest_state`
  table), `--senders`/`-s` (comma-separated list of phone numbers/emails —
  matched against `handle.id` via `chat_handle_join`; only ingest chats,
  direct or group, that include at least one of them. `--senders` only
  scopes what *this* run reads from the source; it never deletes anything
  already in `imessage_analysis.db`. Ingestion is purely additive (see
  "append-only store" below), so running with different filters accumulates
  the union of all matched chats across runs rather than replacing the DB
  with just the latest filter's rows.
- **`query_tools.py`** — retrieval functions meant to be exposed as LLM tool
  calls: `search_messages_sql` (filter by chat/sender/date),
  `search_messages_fts` (FTS5 keyword search), `search_messages_vec`
  (embed query via Ollama + `sqlite-vec` nearest-neighbor search),
  `get_chat_summary_stats` (cheap aggregate stats, no LLM needed),
  `search_contacts_by_name` (name/org → contact + its handle_ids — the
  name→id bridge the ID-based tools need), `list_chats` (name/participant →
  chat_id, and resolves a chat_id to its participant names),
  `search_contact_by_handle` (a phone/email/handle_id → contact + the
  contact's other handles), and `get_message_reactions` (tapbacks on a
  message by its `guid`). The three `search_messages_*` tools resolve the
  sender to a human `sender` name (via join to `contacts`) and also return
  `guid` and `has_attachment` on each row. Includes `TOOL_DEFINITIONS` —
  tool-schema definitions (used by `chat.py`, and directly usable as Claude
  `tool_use` schemas).
- **`chat.py`** — interactive CLI that chats over the iMessage history via a
  local Ollama-hosted `gemma4:26b-a4b-it-qat`, wiring
  `query_tools.TOOL_DEFINITIONS` into a tool-calling loop (`run_turn` drives
  up to `MAX_TOOL_ITERATIONS` tool round-trips per user turn — default 8 —
  calling all eight `query_tools` functions). The model name, host, timeouts,
  and iteration cap all come from `config.py` (so the model default above can
  be changed without editing code), and the system prompt advertises the full
  tool set, telling the model to resolve a person/group name to a
  handle_id/chat_id (via `search_contacts_by_name` / `list_chats`) before
  message search. This is the realized version of the tool-calling loop listed
  under "Next steps" below.
- **`reset_db.py`** — clears all data rows from `imessage_analysis.db` while
  preserving the schema (tables, FTS5 triggers, `sqlite-vec` table); does NOT
  touch `chat.db`/`chat_snapshot.db`. Deletes in FK-safe order and guards
  each `DELETE` on the table actually existing (`sqlite_master` check), so a
  partial-schema DB won't error. `-f/--force` skips the confirm prompt. Run
  `ingest.py` afterward to repopulate.
- **`reset-db.sh`** — thin wrapper that runs `reset_db.py` via
  `.venv/bin/python3` (passes through args).
- **`imessage_analysis.db`** — output of a full ingestion run against the
  real `chat.db` in this repo (135MB), with a **partial** embedding run
  present: `message_vectors` holds ~3.5k vectors so far out of ~350k
  messages (full embedding still pending). Verified: ~350k messages +
  ~32k reactions = ~382k (matches `chat.db`'s `message` count exactly),
  ~1.1k chats, ~1.9k handles, ~14.9k attachments, ~260 contacts. FTS5 and
  `sqlite-vec` mechanics both confirmed working with real test queries.
- **`data/<sender>.txt`, `data/orphaned.txt`** — sample `.txt` exports, not
  currently used by the pipeline (see Data sources above).
- **`tests/`** — stdlib `unittest` smoke tests (no Ollama / real `chat.db`
  needed; all against temp SQLite). `test_migration_and_ingest.py` covers the
  in-place `handles` migration in `create_schema`, FK-safe idempotent re-ingest
  of contacts + handles, and the `query_tools` `search_contacts_by_name` /
  `list_chats` payload+filter behavior. Run: `.venv/bin/python3 -m unittest
  discover tests`.

## Schema (imessage_analysis.db)

```
chats(id, chat_identifier, display_name, is_group, service)
contacts(id, source_uniqueid, first_name, last_name, nickname, organization)
                                    -- id = ZABCDRECORD.Z_PK from address_book.abcddb, one row per contact card
handles(id, handle_identifier, contact_id, service)
                                    -- handle_identifier = raw handle.id (phone/email) from chat.db
                                    -- contact_id -> contacts.id (nullable, best-effort match)
                                    -- name/org NOT duplicated here: resolve via contact_id -> contacts
chat_participants(chat_id, handle_id)
messages(id, guid, chat_id, handle_id, is_from_me, text, decode_failed,
         date, date_read, date_delivered, item_type, is_emote, has_attachment)
reactions(id, target_message_guid, chat_id, handle_id, is_from_me,
          associated_message_type, date)
attachments(id, message_id, filename, mime_type, transfer_name)
ingest_state(key, value)          -- tracks last_message_rowid for --since-last
messages_fts                       -- FTS5 external-content table over messages.text
message_vectors                    -- sqlite-vec vec0 table, message_id -> FLOAT[768]
```

Supporting indexes (built in `ingest.py`): `idx_messages_chat_date`,
`idx_messages_handle`, `idx_reactions_target`, plus `idx_handles_identifier`,
`idx_handles_contact_id`, and `idx_chat_participants_handle` — the last three
back the name→id / handle→contact resolution joins used by the
`search_contact*` / `list_chats` query tools.

## Known environment quirks

While building this, ingestion had to be run from `/tmp` inside the dev
sandbox rather than directly against the mounted project folder — the
sandbox's FUSE mount of the host folder doesn't support the
`PRAGMA wal_checkpoint` operation SQLite needs during the snapshot step
(disk I/O error). This should **not** occur on a real Mac since `chat.db`
lives on local disk there. If it ever recurs, the workaround is: copy
`chat.db` to local disk, run `ingest.py` there, copy the resulting
`imessage_analysis.db` back.

**DB paths are anchored to the repo, not the CWD.** `config.py` resolves the
four DB-path constants (`SOURCE_DB`, `SNAPSHOT_DB`, `TARGET_DB`,
`ADDRESS_BOOK_PATH`) relative to `config.py`'s own directory when the value is
relative (the default), and uses absolute env-var values as-is. This keeps
`ingest.py` / `query_tools.py` / `reset_db.py` all pointed at the same
repo-root DB no matter where they're launched from. Consequence for the `/tmp`
workaround above: copy the scripts (`config.py` + siblings) alongside the data
so `__file__` points into `/tmp`, or set an absolute `SOURCE_DB` / `TARGET_DB`
env override.

**Minimum SQLite version: 3.35 (2021).** `ingest.py::create_schema` migrates an
old-layout `handles` table in place using `ALTER TABLE ... RENAME COLUMN`
(needs SQLite ≥ 3.25) and `ALTER TABLE ... DROP COLUMN` (needs ≥ 3.35). Both are
satisfied by the Homebrew Python mandated below (ships a current SQLite); the
python.org framework build also ships a recent SQLite, but is excluded for the
loadable-extension reason described next.

**`.venv` must be built from a Python whose `sqlite3` module supports
loadable extensions**, since `sqlite_vec.load(conn)` (used in
`ingest.py::build_vec_table` and `query_tools.py::_connect`) requires
`conn.enable_load_extension()` to exist. The **python.org macOS installer
build** (framework build, e.g. `/usr/local/bin/python3.14`) is compiled
without this support and raises
`AttributeError: 'sqlite3.Connection' object has no attribute
'enable_load_extension'`. **Homebrew's Python** (e.g.
`/opt/homebrew/bin/python3`) does support it and is what `.venv` should be
built from: `rm -rf .venv && /opt/homebrew/bin/python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt`. `requirements.txt` pins just two
third-party deps — `python-dotenv==1.2.2` (used by `config.py`) and
`sqlite-vec==0.1.9` (the vector extension); everything else is stdlib.

## Next steps / not yet built

1. **Finish embeddings**: a partial run exists (~3.5k vectors in
   `message_vectors`), so this is started but not complete. To fill in the
   rest: `ollama pull nomic-embed-text`, then `python3 ingest.py --since-last`
   on a machine where Ollama is actually reachable, until all ~350k
   messages are embedded.
2. **Wire `TOOL_DEFINITIONS` into a tool-calling loop** — DONE: `chat.py`
   implements this against `gemma4:26b-a4b-it-qat` via Ollama. Remaining
   option: a Claude-backed reasoning/analysis layer over the same tools
   (keeping embeddings local) if preferred for heavier analysis.
3. **Enrichment/analysis layer** (sentiment, emotion, topic tagging) —
   discussed as a future layer (batch-tag messages, store as extra columns
   on `messages`, aggregate via SQL for trend questions) but intentionally
   deferred — focus so far has been the storage/retrieval foundation, not
   the analysis types themselves.
4. **`.txt` parser** — not built; only needed if `chat.db` access isn't
   available for some future conversation/export.
