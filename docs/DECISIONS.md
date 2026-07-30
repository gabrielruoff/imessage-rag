# imessage-rag — design decisions & full reference

Full design history, rationale, data forensics, and per-file reference.
The always-loaded agent orientation lives in the repo-root `CLAUDE.md`;
this document is where the depth moved.

## Goal

Make iMessage history queryable and analyzable by an LLM — both casual chat
("what did we say about the trip?") and analysis (sentiment/trend/topic
questions over time), for conversations that may scale to hundreds of MB or
GB. The data cannot be pasted into a chat window or read wholesale by an LLM
at query time — it needs to be parsed into a structured store that supports
targeted retrieval.

## Data sources

- `source_data/chat.db` — the live macOS Messages database (project-local
  path, on the order of 500MB / ~380k messages across ~1.1k chats and ~1.9k
  handles on the dev corpus). This is the primary/preferred source — richer
  and more reliable than parsed `.txt` exports.
- `source_data/address_book.abcddb` (optional, project-local path) — a copy
  of macOS's Contacts store (normally at `~/Library/Application
  Support/AddressBook/Sources/*/AddressBook-v22.abcddb`). `chat.db`'s
  `handle` table only has raw phone numbers/emails, no display names —
  `ingest.py::build_contact_lookup` reads this file (if present at
  `source_data/address_book.abcddb`) to attach `first_name`/`last_name`/`nickname`/
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
  Embeddings via Ollama running `gemma4:26b-a4b-it-qat` (768-dim, 8192-token
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

- **`message.text` is NULL for ~99% of rows** (all but ~2.5k rows on the dev
  corpus). The actual text lives in the `attributedBody` BLOB as an
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
  Unix epoch — converted via `apple_ns_to_iso()` in `ingest.py`. **The resulting ISO
  strings are UTC with no offset suffix** — `apple_ns_to_iso` does
  `APPLE_EPOCH + timedelta(...)` on a naive `datetime` (no `tzinfo`,
  no `.astimezone()`), and since Apple's Mac-absolute-time reference date
  (2001-01-01 00:00:00) is itself defined in UTC, the naive arithmetic result is
  numerically UTC — it's just missing the `Z`/`+00:00` marker that would say so.
  Because they're compared as plain strings, `date_from`/`date_to` bounds must be
  normalized before they reach SQL — `query_tools.py::_normalize_date_bound` does
  this for both `search_messages_sql` and `get_chat_summary_stats`: it parses the
  bound, converts a tz-aware one (`...-05:00`, `...Z`) to naive UTC, and returns an
  upper bound **half-open**, advanced by one unit of whatever granularity the caller
  wrote (`_date_bound_step`), so `date_to='2026-01-02'` becomes `date <
  '2026-01-03T00:00:00'` and covers the whole day. A bare `date <= '2026-01-02'`
  would instead match only exactly midnight and silently drop the day — the bug this
  guards against. An unparseable bound raises `ValueError` (surfaced to the model as
  a tool error) rather than quietly matching nothing. **Normalize the parameter, never
  the column**: wrapping `date` in a SQL function would discard
  `idx_messages_chat_date`, and SQLite's `datetime()` emits a space-separated,
  second-truncated string that mis-compares against these `T`-separated stored
  values anyway. `chat.py`'s system prompt (`_build_system_prompt()`) still tells the
  model the current UTC time each turn — it can't know "now" otherwise — but UTC
  conversion is now a hint, not what correctness rests on.
- **No single table has a full message record** — had to join `message` +
  `handle` (sender) + `chat_message_join` (message→chat, not a direct FK on
  `message`) + `chat_handle_join` (chat participants, for group threads) +
  `message_attachment_join` + `attachment`.
- **Reactions/tapbacks are stored as rows in the `message` table**
  (`associated_message_type != 0`, ~32k of them on the dev corpus) pointing at a target
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

- **`README.md`** — short human quickstart (purpose, `make install` /
  `ingest.py` / `chat.py`); this file is the full reference.
- **`pyproject.toml`** — `ruff` configuration (`line-length = 100`,
  `select = ["E", "F"]`). See "Linting" below.
- **`db.py`** — single shared helper, `load_vec_extension(con)`, wrapping the
  three-line `enable_load_extension` / `sqlite_vec.load` / disable sequence
  needed on any raw `sqlite3.Connection` before it can touch a `vec0` virtual
  table. Used by `query_tools.py::_connect`, `embed.py::build_vec_table`, and
  `reset_db.py::main` so the extension-loading incantation exists in one
  place.
- **`ollama.py`** — all Ollama HTTP calls in one module: `embed_one` (single
  text, `OLLAMA_EMBEDDINGS_PATH`, used at query time by
  `query_tools.search_messages_vec`), `embed_batch` (list of texts,
  `OLLAMA_EMBED_PATH`, persistent per-thread keep-alive connection, used by
  `embed.py`), and `chat_stream` (streaming `OLLAMA_CHAT_PATH` generator
  yielding each chunk's `message` dict, used by `chat.py`, which owns the
  rendering). Every knob comes from `config.py`; nothing is hardcoded.
- **`config.py`** — single source of runtime configuration, imported by
  `ingest.py`, `embed.py`, `ollama.py`, `query_tools.py`, and `chat.py`.
  Reads every setting from the
  environment (calling `python-dotenv`'s `load_dotenv()` first, so a local
  `.env` is picked up automatically) and exposes them as module constants, each
  with a working default so nothing *has* to be set. Groups: DB paths
  (`SOURCE_DB`, `SNAPSHOT_DB`, `TARGET_DB`, `ADDRESS_BOOK_PATH`), Ollama
  connectivity (`OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_EMBED_PATH`,
  `OLLAMA_EMBEDDINGS_PATH`, `OLLAMA_CHAT_PATH`, `OLLAMA_TIMEOUT`,
  `CHAT_TIMEOUT`), models (`EMBED_MODEL` = `gemma4:26b-a4b-it-qat`, `CHAT_MODEL` =
  `gemma4:26b-a4b-it-qat`), and tunables (`EMBED_DIM` = 768, `EMBED_BATCH_SIZE`,
  `EMBED_MAX_WORKERS`, `MAX_TOOL_ITERATIONS`).
- **`.env.example` / `.env`** — `.env.example` is the committed template: every
  variable from `config.py` listed and commented out with its default, so it
  doubles as documentation of what's overridable. Copy it to `.env` and
  uncomment what you want to change; `.env` itself is git-ignored (may hold
  machine-specific paths/hosts). `.env` is optional — the defaults in
  `config.py` work without it.
- **`ingest.py`** — pipeline driver: snapshot → decode → normalize into
  `imessage_analysis.db` → build FTS5, then hand off to `embed.py` for the
  `sqlite-vec` table + embeddings. Flags: `--no-embed` (skip embedding, e.g. if Ollama isn't
  running), `--since-last` (incremental — only ingest messages with
  `ROWID` greater than the last recorded run, tracked in the `ingest_state`
  table), `--senders`/`-s` (comma-separated list of phone numbers/emails —
  matched against `handle.id` via `chat_handle_join`; only ingest chats,
  direct or group, that include at least one of them. `--senders` only
  scopes what *this* run reads from the source; it never deletes anything
  already in `imessage_analysis.db`. Ingestion is purely additive (see
  "append-only store" below), so running with different filters accumulates
  the union of all matched chats across runs rather than replacing the DB
  with just the latest filter's rows. `--lookback`/`-l` (a period like `7d`,
  `3mo`, `2y` — units d=days/w=weeks/mo=months/y=years, with `mo`/`y`
  calendar-correct and short-month/leap edges clamped) adds a
  `message.date >= now - PERIOD` lower bound, computed once against wall-clock
  UTC and applied inside the source read (messages, their reactions, and their
  attachments). It's orthogonal to the other flags — ANDs with `--since-last`'s
  ROWID bound and the sender/exclude chat allowlist — and, like them, is purely
  additive: a later unfiltered run still backfills the older messages. Dateless
  system rows (`message.date` NULL/0) fall outside any window and are skipped
  while `--lookback` is active; `last_message_rowid` advances to the newest
  *in-window* message, so a subsequent `--since-last` at worst re-scans a few
  already-ingested rows (every write is idempotent `INSERT OR REPLACE`) and
  never skips. One quirk to know: a fresh `--lookback` run can ingest a reaction
  that falls inside the window while its target message (always older, since a
  reaction post-dates what it reacts to) falls outside it, leaving
  `reactions.target_message_guid` pointing at a message not yet in `messages`.
  There's no FK — the target is a text GUID joined at query time — so this is
  benign (the reaction just can't be resolved) and self-heals on a later full
  backfill.
- **`embed.py`** — the embedding step, called from `ingest.main` after the
  relational ingest: `build_vec_table` (creates the `message_vectors` vec0
  table) and `embed_messages` (thread-pooled batch embedding via
  `ollama.embed_batch`; only messages without a vector are embedded, so
  re-runs resume where the last one stopped; Ctrl+C commits whatever
  finished before re-raising).
- **`query_tools.py`** — retrieval functions meant to be exposed as LLM tool
  calls: `search_messages_sql` (filter by chat/sender/date),
  `search_messages_fts` (FTS5 keyword search), `search_messages_vec`
  (embed query via Ollama + `sqlite-vec` nearest-neighbor search),
  `get_chat_summary_stats` (cheap aggregate stats, no LLM needed),
  `search_contacts_by_name` (name/org → contact + its handle_ids — the
  name→id bridge the ID-based tools need), `list_chats` (name/participant →
  chat_id, and resolves a chat_id to its participant names),
  `search_contact_by_handle` (a phone/email/handle_id → contact + the
  contact's other handles), `get_message_reactions` (tapbacks on one or more
  messages by `guid`), and `get_full_messages` (untruncated text for one or
  more message ids). The three `search_messages_*` tools resolve the sender
  to a human `sender` name (via join to `contacts`) and also return `guid`
  and `has_attachment` on each row; long `text` is truncated to
  `TEXT_PREVIEW_CHARS` (300 chars) and recoverable via `get_full_messages`.
  **`chat_id`/`handle_id` params accept a single id or a list of ids** across
  every message/chat/stats tool, and a **`contact_id` param** (single id or
  list, resolved through `handles.contact_id` via `_resolve_handle_filter`)
  scopes to a person across every handle (phone/email) they own — added so a
  person-centric question ("when did I last talk to X") resolves in ~2 tool
  calls instead of enumerating each of a contact's handles/chats one at a
  time. `list_chats` additionally accepts `handle_id`/`contact_id` for an
  exact chat-participation join, as a precise alternative to its fuzzy
  `query` substring match. `get_message_reactions` batches over a list of
  guids, tagging each returned row with its `target_message_guid`.
  `get_chat_summary_stats` requires at least one of `chat_id`/`handle_id`/
  `contact_id` (raises `ValueError` otherwise), and returns `attachment_count`/
  `reaction_count` alongside the message counts/date range; takes an optional
  `is_from_me` filter (narrows to only sent or only received messages), an
  optional `date_from`/`date_to` range, and an optional `group_by_chat` flag
  (returns one row per `chat_id` instead of a single pooled total, for
  comparing activity across chats). `attachment_count` counts rows in
  `attachments` joined through `messages` — individual attachments, so a message
  carrying three photos counts three, not one. **Every filter scopes all three
  counts alike**: `messages`, `reactions`, and `attachments`-through-`messages`
  all expose the same `chat_id`/`handle_id`/`is_from_me`/`date` columns, so each
  table's WHERE is built from the one prefix-parameterized `_stats_filters()`
  helper — that's what keeps `is_from_me=True` meaning "messages, attachments,
  and reactions I sent" rather than narrowing only the message count. Both
  count subqueries sit in the SELECT list ahead of the WHERE, so their params
  bind first (`att_params + reaction_params + params`); under `group_by_chat`
  they correlate on the outer `messages.chat_id` and therefore re-apply only the
  handle/is_from_me/date filters, not `chat_ids`.
- **`tool_schemas.py`** — `TOOL_DEFINITIONS`, the tool-schema definitions for
  the `query_tools.py` functions (used by `chat.py`, and directly usable as
  Claude `tool_use` schemas); id params use JSON Schema's `"type": ["integer",
  "array"]` form so both scalar and list calls validate. Keep in sync with
  `query_tools.py` and the tool list in `chat.py`'s `SYSTEM_PROMPT`.
- **`chat.py`** — interactive CLI that chats over the iMessage history via a
  local Ollama-hosted `gemma4:26b-a4b-it-qat`, wiring
  `tool_schemas.TOOL_DEFINITIONS` into a tool-calling loop (`run_turn` drives
  up to `MAX_TOOL_ITERATIONS` tool round-trips per user turn — default 20 —
  calling all nine `query_tools` functions). If that budget is exhausted,
  `run_turn` makes one extra answer-only request with the tool schemas
  withheld and `FINAL_TURN_NOTE` folded into the system prompt, so a turn
  always ends in a real answer rather than a give-up string (withholding the
  tools is load-bearing: the note alone does not stop the model from calling
  one — see the gemma4 renderer quirk under "Known environment quirks").
  The model name, host, timeouts,
  and iteration cap all come from `config.py` (so the model default above can
  be changed without editing code), and the system prompt advertises the full
  tool set, telling the model to resolve a person/group name to a
  handle_id/chat_id (via `search_contacts_by_name` / `list_chats`) before
  message search. This is the realized version of the tool-calling loop listed
  under "Next steps" below. `_build_system_prompt()` appends the current UTC
  date/time (computed fresh each call, not baked into the static
  `SYSTEM_PROMPT` constant) plus an explicit UTC-conversion instruction, and is
  called at both places the message list is (re)built (session start and
  `/clear`) — needed because the model has no other way to know today's date,
  which it must have to resolve "yesterday"/"this morning" into a bound at all.
  The UTC-conversion half of that instruction is now belt-and-braces rather than
  load-bearing: `_normalize_date_bound` (see "Key technical findings" above)
  accepts an offset-carrying bound and converts it itself, so a model that
  forgets to convert still gets a correct answer.
- **`reset_db.py`** — clears all data rows from `imessage_analysis.db` while
  preserving the schema (tables, FTS5 triggers, `sqlite-vec` table); does NOT
  touch `chat.db`/`chat_snapshot.db`. Deletes in FK-safe order and guards
  each `DELETE` on the table actually existing (`sqlite_master` check), so a
  partial-schema DB won't error. `-f/--force` skips the confirm prompt. Run
  `ingest.py` afterward to repopulate. Invoked as `make reset-db`
  (`.venv/bin/python3 reset_db.py -f`).
- **`imessage_analysis.db`** — output of a full ingestion run against the
  local `chat.db` (~135MB), with a **partial** embedding run present:
  `message_vectors` held ~3.5k vectors out of ~350k messages (full embedding
  still pending). Verified on the dev corpus: non-reaction messages +
  reactions summed exactly to `chat.db`'s `message` count, and chat, handle,
  attachment, and contact totals all reconciled. FTS5 and `sqlite-vec`
  mechanics both confirmed working with real test queries.
- **`data/*.txt`** — sample `.txt` exports, not
  currently used by the pipeline (see Data sources above).
- **`tests/`** — stdlib `unittest` smoke tests (no Ollama / real `chat.db`
  needed; all against temp SQLite). `test_migration_and_ingest.py` covers the
  in-place `handles` migration in `create_schema`, FK-safe idempotent re-ingest
  of contacts + handles, and the `query_tools` `search_contacts_by_name` /
  `list_chats` payload+filter behavior, including the `chat_id`/`handle_id`
  list-param and `contact_id`-scoping behavior of `search_messages_sql`,
  `search_messages_fts`, `list_chats`, `get_chat_summary_stats`, and the
  batch-guid behavior of `get_message_reactions` (fixture: a contact with two
  handles across a direct chat and a group chat). `test_chat.py` covers
  `chat.run_turn`'s budget-exhausted fallback (note folding, tool
  withholding). `test_date_bounds.py` covers `_normalize_date_bound` /
  `_date_bound_step`. Run: `.venv/bin/python3 -m
  unittest discover tests`. **Do not modify the tests in this directory unless
  the user specifically asks for test changes, and do not read the test files
  unless they're directly relevant to the current task** (e.g. the task is
  about testing itself, or understanding existing coverage is necessary to
  avoid duplicating or breaking it).

## Schema — the original ingest layout (2026-07, superseded)

Kept as the design-history snapshot of the pre-chunking schema. The LIVE schema
— every table, column and `ingest_state` key — is CLAUDE.md's "Schema
(imessage_analysis.db)" block, and `schema.py` is the DDL of record. This block
predates `chunks`/`messages.chunk_id`, every derived analytics table
(`sessions`, `person_period_lexicon`, `dyad_terms`, `chunk_annotations`,
`topics`/`chunk_topics`, `loose_threads`, `nudge_exclusions`, `dyad_drift`,
`person_voiceprint`, `conflicts`/`conflict_signals`), the 19 `ingest_state` keys
the code now writes, and the move from per-message `message_vectors` to
per-chunk `chunk_vectors` — `message_vectors` no longer has a writer at all and
survives only in `reset_db.DROP_TABLES`, so an old DB gets cleaned up.

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

Supporting indexes: the set has grown well past the six this snapshot listed
(`idx_messages_chunk`, `idx_chunks_chat`, `idx_sessions_chat`,
`idx_chunk_topics_cluster`, `idx_loose_threads_*`, `idx_conflicts_chat`,
`idx_conflict_signals_conflict`, …). `schema.py`'s `CREATE INDEX` statements are
the list of record — a second copy here would only have to be re-synced. The
original six still carry the load they were built for: `idx_messages_chat_date`,
`idx_messages_handle`, `idx_reactions_target`, plus `idx_handles_identifier`,
`idx_handles_contact_id`, and `idx_chat_participants_handle`, the last three
backing the name→id / handle→contact resolution joins used by the
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

**gemma4's Ollama renderer honors only the FIRST system message.** Ollama
(0.32.0) renders `gemma4` with a builtin Go renderer (`renderers.Gemma4Renderer`),
not a Modelfile template — `ollama show --template gemma4:26b-a4b-it-qat` returns
only the stub `{{ .Prompt }}`, so this behavior is invisible to inspection and has
to be probed. Any system message after the first is **silently dropped**: a
mid-conversation or trailing `{"role": "system", ...}` reminder never reaches the
model (verified — the model's `thinking` field showed it had never seen the text).
To steer the model mid-turn, fold the text into the first system message instead,
as `chat.py::run_turn` does with `FINAL_TURN_NOTE`. This is model-specific:
`qwen3:30b-a3b` (one alternative `CHAT_MODEL`) *does* render a
trailing system message — but folding into the first system message is correct for
both, so `run_turn` does that unconditionally rather than branching on model.
Related, and true of both models: prompting alone does not reliably stop a tool call
— the `tools` schemas must actually be withheld from the request. Also note both are
*thinking* models returning a `thinking` field alongside `content`; `chat.py`
deliberately renders only `content`.

**Ollama silently drops the conversation when one message exceeds the context
window.** Ollama (0.32.0) fits a request by keeping the first system message
plus the longest *suffix of whole messages* that fits in the model's context
window. A single tool result bigger than the window (observed: an unlimited
`get_chat_summary_stats(group_by_chat=True)` dump, ~550 rows / ~114KB ≈ 35k
tokens vs the 32k window) means no suffix fits, so the model receives *only
the system prompt* — no user question, no data — and confabulates a question
from the system prompt's example text (verified by probing: a marker planted
in a 20k-char tool message is retrievable, `prompt_eval_count` ≈ 10.7k; at
60k chars `prompt_eval_count` collapses to 19 and the marker is gone). There
is no error and `truncated` stays 0 in the runner log. Defenses: every tool
that can return one row per chat/message caps its row count
(`_shared.MAX_STATS_ROWS` etc.), and `chat.run_turn` hard-caps any tool
result at `TOOL_RESULT_MAX_CHARS` before appending it to the conversation.

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
loadable extensions**, since `sqlite_vec.load(conn)` (wrapped by
`db.py::load_vec_extension`, used from `embed.py::build_vec_table`,
`query_tools.py::_connect`, and `reset_db.py::main`) requires
`conn.enable_load_extension()` to exist. The **python.org macOS installer
build** (framework build, e.g. `/usr/local/bin/python3.14`) is compiled
without this support and raises
`AttributeError: 'sqlite3.Connection' object has no attribute
'enable_load_extension'`. **Homebrew's Python** (e.g.
`/opt/homebrew/bin/python3`) does support it and is what `.venv` should be
built from: `rm -rf .venv && /opt/homebrew/bin/python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt` (or
`make install`). `requirements.txt` pins three runtime deps —
`python-dotenv==1.2.2` (used by `config.py`), `sqlite-vec==0.1.9` (the
vector extension), and `rich==13.9.4` (used by `chat.py` to stream/render
the model's markdown replies in the terminal); `requirements-dev.txt` pins
`ruff==0.15.21` (linting, see below); everything else is stdlib.

## 2026-07-16: aggregate tools + loop fixes (eval-driven)

A 20-prompt eval sweep ([`eval_report_2026-07-16.md`](eval_report_2026-07-16.md))
showed narrative/retrieval prompts working and quantitative/superlative
prompts failing (empty replies, confident wrong numbers from 50-row samples,
a whole 20-iteration budget spent approximating a histogram). Changes, all
motivated by specific transcript failures:

- **`query_tools/analytics.py`** — five single-call aggregate tools:
  `get_conversation_dynamics` (initiations, response-time median/p90,
  longest silences + who broke them, double-texts; person scope = their 1:1
  chats), `get_activity_histogram` (month/day/dow/hour buckets, UTC,
  from_me/from_them split), `get_top_reacted_messages` (tapbacks *received*,
  filterable by type), `get_chat_participant_stats` (per-member rows incl.
  'Me' and silent members; identities merged in Python because from-me rows
  carry `handle_id=0` and contacts own duplicate handles),
  `get_word_frequency` (n-gram counts with stopword filtering; sender
  scope, unlike the other four which scope to a person's chats).
- **Reaction guid normalization** — `reactions.target_message_guid` was
  stored raw from chat.db's `associated_message_guid`, which carries a
  message-part prefix (`p:N/`, `bp:`) that `messages.guid` never has; well
  under 1% of reactions joined to their target. Ingest now strips the
  prefix and `create_schema` runs an idempotent in-place `UPDATE` for
  existing DBs. This bug silently zeroed `get_message_reactions` lookups.
- **Result envelopes** — `search_messages_sql`/`search_messages_fts` return
  `{rows, total_matching, returned, order, truncated}` so a capped page can
  never be mistaken for the population (the eval's longest-silence failure,
  where a confident gap was computed from just the newest 50 rows of a
  multi-thousand-row thread).
- **`reactions_sent_count` rename** — `get_chat_summary_stats.reaction_count`
  counted reactions *sent by* the scope; the model read it as "received" and
  inverted a group-chat ranking. The name now says what it counts.
- **chat.py loop robustness** — a reply with neither content nor tool calls
  (4/20 eval runs) now falls through to the existing tools-withheld
  final-turn request instead of returning an empty answer; exact-duplicate
  tool calls are short-circuited with a "reuse the earlier result" note
  (one eval run repeated an identical call ten times).
- **`eval/fast_eval.sh`** — single-stream prompt-sweep runner (Ollama
  serializes requests; parallel eval batches only add queueing and thrash
  the KV prefix cache) with reduced-context env defaults for fast smoke
  sweeps.

## 2026-07: the analytics layer (six section tools + derived tables)

Replaces the three narrow aggregate tools above (`get_conversation_dynamics`,
`get_activity_histogram`, `get_top_reacted_messages`) with six that each
answer a *family* of questions in one call, and adds the derived tables and
batch-LLM CLIs they read. Rationale for the decisions that aren't obvious
from the code:

- **Section tools, not more tools.** The eval sweep's failure mode was the
  model spending its whole iteration budget stitching together narrow calls.
  Each new tool returns one dict of small pre-aggregated sections
  (`include=[...]` selects a subset), so "when do we talk / how fast do we
  reply / who starts it" is one round-trip, not six. Every section is capped
  (`MAX_LEADERBOARD_ROWS`, `MAX_SERIES_BUCKETS`) with auto-coarsening
  day→month→year, so a result can't blow the context window — the
  `TOOL_RESULT_MAX_CHARS` failure mode is designed out rather than truncated.
- **`sessions` is a table, not a `messages.session_id` column.** Sessions are
  a *derived* artifact with a tunable gap; a column would make re-tuning
  `SESSION_GAP_MINUTES` a migration and would put a rebuildable value on the
  append-only source-derived table. As a separate table it falls under the
  same carve-out as `chunks` (reset_db deletes it; ingest rebuilds it), and
  the `sessions_gap_minutes` state key forces a full rebuild if the rule
  changes, so the table can never mix splitting rules. Rebuild is per-chat
  and whole: the largest chat takes milliseconds, and it's immune to the
  tail-surgery subtleties that make incremental chunk rebuilds delicate.
- **Sessions are one gate stricter than the analytics aggregates.** A session
  requires text or an attachment; the tools keep every dated real message.
  ~32 rows of 350k differ. A row with no renderable content can't start or
  end a conversation, so it must not open a session — but it *is* a message,
  so it still counts in a message count. The two populations are near-identical
  by design, not identical; don't assert equality between them.
- **Local-timezone bucketing happens in Python.** `messages.date` is
  unlabeled UTC. "Late night" bucketed in UTC is off by hours — the single
  most wrong-looking number a user could see. SQL can't do the tz conversion
  without wrapping the column in a function and killing
  `idx_messages_chat_date`, so rows are filtered in SQL (plain-string date
  compares) and bucketed in Python. `LOCAL_TZ` is read via the `config`
  module at call time so tests can pin a zone.
- **LSM needs a function-word lexicon, so `person_period_lexicon` is
  precomputed.** Language style matching compares *function-word* rates
  (pronouns, articles, conjunctions...), which means tokenizing every
  message. Per-person-per-month rows make that a one-time ingest cost;
  computing it per query would tokenize the corpus on every call. It's a full
  rebuild (a few seconds at 350k) — incremental upkeep would need
  read-modify-write on every affected month row for no measurable win.
- **`dyad_terms` uses add-one-smoothed lift** — `(uses_in_chat /
  tokens_in_chat) / ((uses_elsewhere + 1) / tokens_elsewhere)` — so a term
  used *only* in one chat (the inside jokes we're hunting) gets a finite lift
  instead of dividing by zero. Candidates exclude the 9 function-word
  categories plus the corpus's 200 most frequent tokens, computed in the same
  pass: without that, every chat's "distinctive" list is just "yeah" and the
  names everyone uses. Measured cost of the accumulation pass on the dev
  corpus: 13s, 210MB peak, 942k provenance entries — cheap enough that
  pruning candidates before the lift computation isn't worth the complexity.
- **`dyad_terms` rebuilds preserve `llm_verdict`.** Stats are recomputed in
  seconds; a verdict is an overnight LLM pass. Rebuild snapshots the verdicts
  per `(chat_id, term)` and reattaches them. This is why `annotate.py
  --terms` gets the same `annotate_model` anti-mixing guard as
  `chunk_annotations`: both hold expensive, model-specific, rebuild-surviving
  output.
- **`wrapped.py`'s silent-contact guard.** A
  year-in-review that cheerfully resurfaces an ex, or someone who died, is
  the classic nostalgia-product failure. Contacts who were active (>= 3 of
  the 6 months before their last message) and then stopped abruptly (silent
  >= `WRAPPED_SILENT_MONTHS` before corpus end, >=
  `WRAPPED_SILENT_MIN_MESSAGES` lifetime) are auto-excluded from the featured
  roster, the roster is *printed for confirmation* before anything is
  generated, and `--include "Name"` re-admits. The threshold is measured on
  the same `_REAL_MSG` population the roster ranks on, or a contact could
  clear it on system rows alone.
- **`wrapped.py` composes tools deterministically; the LLM only narrates.**
  Every number is computed by a query tool and passed to the model as JSON
  with instructions to use only the numbers present. The model never queries
  and never derives. Sections degrade to a skip note rather than crashing
  (`_tool` late-binds every tool and swallows missing/error/unavailable), so
  a report is never lost to one bad section. Note that `get_word_frequency`
  scopes by *sender* while every other tool scopes to a person's *chats* —
  under `--contact` its key is named `their_top_words` so the narrator can't
  mistake one side's vocabulary for shared vocabulary.
- **The Wrapped report renders HTML structurally, not from a schema.** It was
  markdown with a `json` fence per section: correct, unreadable, and a keepsake
  nobody would open twice. `wrapped_html.render_value` dispatches on the
  *shape* of a stats value (scalar → tile, list-of-dicts → table, period series
  → bar chart, dict-of-dicts → nested blocks), so a new analytics tool renders
  without touching the renderer — the alternative, a per-section template, puts
  every tool one schema change away from a broken report. Two rules earn their
  keep: a lone *list* under one key is a container and is unwrapped (its key
  would repeat the parent heading), while a lone *dict* is not — there the key
  is data (a group chat's name), and unwrapping silently deletes it. And the
  heatmap shades by **quantile band, not linearly against the peak**: hourly
  message counts are heavily skewed, so one 3am outlier sets the peak and flattens
  every other cell onto the same mid step, hiding the daily rhythm the heatmap
  exists to show. Exact counts stay in each cell's hover title, and the caption
  says the shading is relative — a non-linear scale that doesn't announce itself
  is a lie.
- **Wrapped stats are reshaped for readers before rendering or narration.**
  Raw tool results leak plumbing a keepsake must not show: ISO timestamps,
  `start_hour: 0` for midnight, a raw `chat<19-digit-id>` GUID as a group's name,
  error strings ("longest_silence needs a chat or person scope") rendered as
  content, and nested dicts crammed into table cells. Each `_sec_*` builder
  now reshapes its tool's result — drops unavailable sub-results, flattens
  nested rows (Tempo is one comparable row per contact, not a wall of nested
  blocks), names group chats by their members, and renames jargon keys — and
  `wrapped_html` formats values by key (`*_seconds` → "27 min", ISO dates →
  "Feb 3, 2025", `share` floats → percent, hour grids → "12 AM").
  Crucially, the **narrator is fed `wrapped_html.humanized(stats)`** — the
  same display-formatted copy the reader sees, hidden keys stripped — so the
  model can only quote values in the reader's format; narrating the raw JSON
  is how "between 0 and 5", "chat with ID <n>", and re-derived percentages
  got into prose. Sections also carry a `caption` (a one-line methodology
  note rendered under the narrative, e.g. that milestones span the full
  history) and a `hint` (unrendered context for the narrator naming the one
  or two facts worth telling — and clarifying units the model got wrong in
  testing, like 'started' counting conversations, not messages).
- **The Wrapped report is visual-first; tables are the fallback, not the
  format.** A keepsake that reads as a series of tables gets opened once.
  Each section renders through purpose-built visuals in `wrapped_html`
  (horizontal bar lists, opposed me/them split bars — square-root scaled so
  12-second and 39-minute reply medians share one picture, with exact values
  in the labels — a 100% stacked who-talks share bar, a word cloud, joke
  cards, and the hall of fame as iMessage-style bubbles with tapback pills);
  every chart keeps a `<details>` table view as the accessibility fallback,
  and the categorical slots 3–4 added for the share bar were validated with
  the data-viz palette checker in both modes. Because visuals live in
  `extra_html`, sections carry the full shaped data in `narrate_stats` so
  the LLM narrator still sees what the reader sees. The page gained inline
  JS (scroll-reveal, bars growing in, count-up tiles) — a deliberate trade
  approved against the old "no scripts" claim: the promise that matters is
  **no network**, and the JS degrades to a fully static page under
  `prefers-reduced-motion`, JS-off, and print. Two content calls from the
  same pass: "top words" filters through `function_words.COMMON_WORDS`
  (~500 everyday-English words) because a top-words list led by "ok/yeah"
  describes texting, not the person; and Milestones & Anniversaries plus the
  per-100 style table were cut as the sections readers skipped. Inside jokes
  are real now: the all-contacts report pulls the LLM-curated
  `inside_joke`/`nickname` verdicts straight from `dyad_terms` (first-use
  message as the hover), labeling 1:1 chats by the partner's contact name —
  **restricted to the featured roster's chats** (plus groups): dyad_terms is
  corpus-wide, so an unrestricted scan resurfaces terms from chats the
  silent-contact guard or `--exclude` deliberately dropped, and from
  bank/alert threads. The report's design system is the subject's own: iOS
  grouped-list surfaces and separators, `ui-rounded` display type, the real
  send-blue on bubbles only (chart series stay on the validated palette),
  a hero that renders the year's total as one sent message with its
  reactions as a tapback ("Delivered"), and a received-bubble sign-off.
  Three sections come from previously unused data: Ghosts & Double Texts
  (the `ghosted`/`double_text` fields the tempo call already fetched), The
  Marathon (top `sessions` rows by `n_messages`), and the Eras arc — a
  per-month tone strip (diverging blue↔gray↔red, buckets tuned to where
  corpus valence actually lives, ±0.02/±0.08) labeled with each month's
  leading emotion from `chunk_annotations.emotions_json`.
- **Topic k-means pushes assignment into sqlite-vec.** The centroids go in a
  scratch `vec0` table and each chunk's nearest centroid is a `k = 1` KNN;
  only the centroid *update* is Python (`array('f')`, no numpy). The
  `topic_model_k` guard is warn-only, unlike the embedding/annotation guards:
  a rebuild legitimately *replaces* a differently-parameterized clustering —
  the fix for a mismatch is the rebuild itself.
- **Unknown `include` sections raise, never silently drop.** A typo'd section
  name would otherwise return a plausible-looking partial result the model
  can't distinguish from the real thing. One shared `_validate_include` in
  `_shared.py` so all six tools agree.

Hermetic code-level tests for all of the above are in `tests/`.
End-to-end *response-quality* evaluation (does the model pick the right tool
and read the result correctly?) is specified but not yet run — see
[`ANALYTICS_TEST_PLAN.md`](ANALYTICS_TEST_PLAN.md).

## 2026-07-17: annotate.py quality presets (reasoning effort + batching)

A full-corpus annotate run against `gpt-oss:20b` projected ~158 hours.
Profiling one request showed the time goes to the model's *hidden reasoning
pass*, not the answer: at default effort it emitted ~2,000 chars of thinking
before an ~80-char JSON annotation (10.8s wall; prompt eval was 20ms).
`think: "low"` cut that to ~260 chars and 2.7s with an equivalent answer.
Batching several chunks into one request amortizes what remains: 6–12 chunks
per request ran at ~1.0s/chunk (plateaus past ~6; batching *without* low
effort only reached 4.4s/chunk — effort is the primary lever, batching the
secondary one).

Quality was checked by re-annotating 24 max-quality-annotated chunks and
measuring agreement, calibrated against the noise floor of re-running the
max path on itself (valence MAE 0.12, depth exact 50%/±1 100%, emotion
Jaccard 0.52 — the task is inherently noisy):

- low effort, single (balanced): valence MAE 0.23, depth ±1 100%, emotion
  Jaccard 0.59 — within the model's own run-to-run noise, but it surfaces
  fewer good_news/gratitude events (3 vs 9 on the sample).
- low effort, batch 8 (fast): depth ±1 88%, emotion Jaccard 0.34 — slightly
  below the noise floor; acceptable for corpus-scale aggregates.

Hence `--quality {fast,balanced,max}` (env `ANNOTATE_QUALITY`, default
**balanced**), with `ANNOTATE_THINK`/`ANNOTATE_BATCH_SIZE` as fine-grained
overrides. `max` is the original behavior. Implementation notes:

- Ollama's `think` accepts effort strings (`low`/`medium`/`high`) on gpt-oss
  and booleans on toggle models (qwen3). gpt-oss **silently returns empty
  content for `think: false`** — every annotation would be "skipped", so the
  presets use `"low"`, never `false`. gemma4 *accepts* `"low"` without
  erroring — but accepting is not honoring, and the term-curation entry below
  is what that cost: gemma4 has no effort levels, so `"low"` is merely truthy
  there and buys a full reasoning pass. Models that reject `think` outright
  drop it for the rest of the run (warned once).
- Batched replies are keyed by chunk id (`{"<id>": {...}}`); any id the
  reply misses or mangles falls back to a single-chunk request, so batching
  can't lose rows — worst case it costs one extra request.
- Estimated wall time for the ~58k-chunk default scope: max ~140h,
  balanced ~48h, fast ~16–35h.

## 2026-07-17: distilled annotations (kNN over chunk_vectors) + LLM concurrency

Even the fast preset left full-corpus annotation at ~24h+ (and re-measured on
*random* corpus chunks the LLM tiers run ~1.7x slower than the top-dyad
benchmarks above — balanced is really ~5s/chunk, ~100h corpus-wide). The
target was ≤15 min, and no per-chunk-LLM design reaches it locally, so the
default tier now stops calling the LLM per chunk:

**`--quality distilled` (new default)**: LLM-label a random corpus-wide seed
sample (`ANNOTATE_SEED_TARGET`, 450), then classify every remaining chunk by
a distance-weighted vote of its `ANNOTATE_KNN_K` (10) nearest seeds in the
*existing* `chunk_vectors` embedding space — sqlite-vec itself is the KNN
engine (a temp vec0 seed table), so there's no numpy/sklearn dependency.
Measured ~0.3ms/chunk: the full 74k corpus classifies in well under a
minute; a cold start (no seeds) is ~20–40 min of seeding.

Held-out agreement (317 train / 120 test random seeds, vs balanced LLM
labels), calibrated against the LLM's own run-to-run noise (valence MAE
0.12, depth ±1 100%, emotion Jaccard 0.52 when re-run against itself):

| | valence MAE | depth ±1 | emotion Jaccard |
|---|---|---|---|
| fast preset (vs max) | 0.23 | 88% | 0.34 |
| distilled kNN | 0.25 | 95% | 0.27 |

i.e. valence/depth at LLM fidelity, emotions slightly below the fast preset.
Hyperparameters (k 5–40, weight sharpening, vote thresholds) barely move any
metric; the emotion vote keeps top-≤3 labels with ≥0.15 of the weighted vote.

**Events are deliberately NOT distilled.** Excerpt extraction is generative,
and the "cheap filter → LLM confirm" idea died on measurement: ~13% of
random chunks contain a good_news/gratitude event, and the embedding filter
must flag ~70% of chunks to catch ~90% of them. So distilled rows carry
`events_json='[]'` plus a `source='distilled'` provenance column (DEFAULT
'llm' backfills pre-existing rows via ALTER in create_schema), and the
`--upgrade-distilled` flag lets an LLM tier re-annotate distilled rows in
scope — the intended path for filling events where they matter (e.g.
overnight `--quality fast --upgrade-distilled` over the top dyads). Only
`source='llm'` rows train the kNN vote. The `annotate_model` guard covers
distilled rows too: they inherit the seed model's bias.

Seed-pool caveats: any pre-existing `source='llm'` row counts toward
`ANNOTATE_SEED_TARGET`, so a pool inherited from an earlier scoped LLM run
(e.g. top dyads) is biased toward those chats rather than random — the
agreement table above was measured on a genuinely random pool. The seed
sample draws from distilled rows as well as unannotated ones (relabeling
them `'llm'`), so raising `ANNOTATE_SEED_TARGET` and rerunning grows the
pool even after the corpus is fully classified.

**Concurrency (`ANNOTATE_WORKERS`, 4)**: every LLM pass (chunk tiers, seed
pass, --terms) fans requests over a thread pool, mirroring
`chunking.embed_chunks` — workers only do HTTP; validation, SQLite writes,
commits and progress stay on the main thread. Pays off only when the server
runs parallel slots, so `make ollama-start` now sets `OLLAMA_NUM_PARALLEL=4`.
Measured on gpt-oss:20b at 4/4: ~2x (5.16 → 2.65 s/chunk sequential-vs-8-in-
flight; batch-8 prompts × 4 workers reach ~1.4s/chunk). It does not scale to
4x: Apple-Silicon decode is memory-bandwidth-bound and MoE experts diverge
across concurrent sequences, so weight reads only partially amortize. RAM
cost ~0.6 GB; Ollama scales total context so each slot keeps the full 32k
window. Against a serial server the extra workers just queue (CHAT_TIMEOUT
covers the wait), which is why 4 is a safe default.

## 2026-07-17: term curation was ~11x slow on a borrowed `think`

`annotate.py --terms` was projected at **8.6 days** for 2,635 pending terms.
Profiling found none of it was ours: the per-term `LIKE '%term%'` prompt scan
— which the code's own comment called "the expensive bit" — measures **2ms**
at 350k messages (it rides `idx_messages_chat_date`). All of it was one
request, and the request was slow for a reason that had nothing to do with
terms.

`main()` resolved `--quality` *before* dispatching on `--terms`, so term
curation inherited the chunk presets' `think`. Default quality is `distilled`
→ `think="low"`. gemma4 has no effort levels, so `"low"` is just truthy to it
and it runs a full reasoning pass — to answer a four-way classification whose
output is one word. One term, one prompt, one model, **identical verdict**
every time:

| `think` | wall |
|---|---|
| `"low"` (what shipped) | 282.6s |
| omitted | 52.0s |
| `false` | 26.2s |

The 282 seconds of hidden reasoning open by re-reading the three-line prompt
back to the model. So term curation is now its own pipeline, not a `--quality`
tier: it owns `ANNOTATE_TERMS_THINK` (default `false`), `--quality` with
`--terms` is a hard error, and `ANNOTATE_THINK` deliberately does not reach it
— one pipeline's `think` silently steering another is the whole bug.

Batching then amortizes the rest, and terms batch far better than chunks do:
the prompt is ~150 chars and the answer is one enum, so
`ANNOTATE_TERMS_BATCH_SIZE=25` fits easily. A batch never spans chats (the
system prompt says "in one text-message chat"; mixing chats would make the
examples contradict it), and replies are keyed by position — positions can't
collide or need escaping the way term text could. As with chunk batching, any
entry the reply misses or mangles falls back to a single-term request, so
batching cannot lose rows.

Measured end-to-end on the real corpus: **25 terms in 29.5s, one request, 0
skipped** (verdicts spread across all four classes). The same 25 terms cost
~2 hours before. Full corpus: ~8.6 days → **under an hour**.

Two smaller things the profile exposed:

- Failures were unbounded and free to repeat. `chat_once_json` retries once,
  so a term that can't be parsed burns *two* full requests — and since a skip
  left `llm_verdict` NULL, which is also the "never attempted" state, every
  future run paid again, forever. `dyad_terms.curate_failures` now counts
  failures, parks a term at `ANNOTATE_TERMS_MAX_FAILURES`, and survives
  lexicon rebuilds (a rebuild resetting it to 0 would restore the loop).
  `--retry-failed` unparks.

  Parking is sticky state, so it is deliberately hard to spend (all of this
  came out of the PR review). Only a term's *own* dedicated request can
  charge it a failure: a batch miss always falls back to one before counting,
  and a group whose fallbacks keep failing is abandoned unattempted rather
  than fanned out into ~50 doomed requests behind a single worker slot. A run
  in which nothing at all succeeds aborts at 20 failures and rolls them back
  — that is a misconfiguration, not 2,635 independent bad terms. And
  `--retry-failed --dry-run` previews the unpark instead of performing it: a
  preview that spends the state it previews is worse than a stray write,
  since the next bare run then re-attempts those terms at full price. The
  misconfiguration all of this guards against is near at hand rather than
  hypothetical: gpt-oss answers `think=false` with empty content and
  `--terms` defaults to `false`, so a gpt-oss `CHAT_MODEL` would otherwise
  park the whole corpus in three runs.
  `ollama.chat_once_json` therefore also detects that empty reply directly
  and degrades the run to no reasoning control — the same move it already
  made for models that reject `think` outright, driven off the reply instead
  of an error, because gpt-oss doesn't error.
- `make ollama-start` was a no-op against an already-running server (`brew
  services start`), so the `launchctl setenv OLLAMA_NUM_PARALLEL=4` above it
  never reached the process — leaving the worst case, an `ANNOTATE_WORKERS=4`
  pool queueing against a server with no parallel slots, looking configured
  while buying nothing. It now `restart`s. Ollama exposes no way to read
  `NUM_PARALLEL` back, so there is nothing to assert afterwards; making the
  restart unconditional is the only fix.

## 2026-07-20: the analytical suite (Wave 0 + Wave 1)

A wave of self-insight features from the Linear project *Close Read —
imessage-rag Analytical Suite*, specified in its canonical
**00 — Master Spec & Parallel-Work Contract** (and the per-feature design docs
below). Every feature turns a person's own complete iMessage history into
private insight — 100% offline, local-LLM only, the corpus never leaving the
machine or entering a prompt wholesale — the same retrieval-not-context-stuffing
posture as the rest of the app. The *parallel-work contract* is why each
feature's schema lands in its own `_create_*_tables` function called from
`create_schema` (never an edit to the shared `SCHEMA` blob), its config behind a
`# --- feature: X ---` fence, its tools behind matching fences in
`tool_schemas.py` / `query_tools/__init__.py` / `chat.py`, and its pipeline step
as a recipe line in `make pipeline`: the issues were built in parallel worktrees
and had to merge additively. Pipeline order is now
`ingest → annotate → nudges → drift → voiceprint → conflict → topics`; the four
new stages each read `chunk_annotations`, so they slot in after `annotate`, and
each is also a standalone `make` target.

**Foundations (Wave 0).** Shared machinery the features build on:

- `stats.py` (F2) — dependency-free robust statistics: Theil-Sen trend +
  bootstrap CI, Kaplan-Meier survival / ECDF placement, modified-z (MAD) outlier
  scoring, CUSUM change-point. Pure stdlib, no `config`/Ollama/sqlite, and every
  function is total (returns `None`/`[]` on degenerate input, never raises), so
  drift, conflict, and wellbeing all import one tested implementation. A
  load-bearing modeling note lives in the module: do NOT reuse
  `get_response_dynamics`'s censored-latency path — it drops gaps
  `>= INITIATION_GAP_HOURS`, biasing a dying tie's latency *downward* (the
  longest silences, exactly the fading signal, are thrown away); model absent
  replies as right-censored observations instead.
- Eval gates (F4, RES-70) — `eval/evalkit/{gates,agreement,labeling}.py` plus the
  `label` / `gate` subcommands. A measurement gate decides whether a modest local
  model is good enough to *ship* a class of labels: the owner hand-labels a
  sample (`eval/labels/<task>.jsonl`, gitignored, owner-only, agents never write
  there), `gate` runs the production classifier and scores Cohen's kappa
  (`conflict_labels`, floor 0.6) or detector precision (`dropped_plan`, floor
  0.85), and persists a pass/fail verdict. Clearance is default-closed: an absent
  verdict suppresses the class (Master Spec A2). `conflict_labels` ships a
  committed seed verdict (`eval/seed_gates/`), so conflict labels are pre-cleared
  out of the box and survive `reset_db` (the seed is in the repo, not the
  per-user DB).
- `consent.py` (F5) — default-closed, tolerant, config-free opt-in id parsing;
  landed as a shared primitive ahead of the features that will gate on explicit
  per-contact consent.
- Lexicon split (F1) — `person_period_lexicon` gained
  `i_count` / `you_count` / `absolutist_count` (raw PRONOUN_I / PRONOUN_YOU /
  ABSOLUTIST token counts), the speaker-attributable markers the wellbeing
  baseline and the voiceprint read; nullable until the next full rebuild.

**Clearing the Air — conflict (01, RES-46 / C1).** `conflict_detect.py` finds
"tense stretches" in 1:1 chats from **deterministic signals only** — reply-latency
withdrawal (survival over raw right-censored gaps vs a trailing-90-day baseline),
one-sided sessions, dislike-tapback runs, message-volume collapse — writing a
`conflicts` window when `>= CONFLICT_MIN_SIGNALS` fire, plus `conflict_signals`
for per-turn escalation labels. The LLM never *originates* a conflict; it only
attaches Gottman-style escalation / `repair` labels, and only for classes the F4
gate has cleared — surfaced as `tentative`/`likely` bands, never raw scores.
Valence corroborates *ranking* (source='llm' chunks only) but never detection,
so the finder is never dead on a default all-distilled corpus (a window with no
llm chunks is written `corroboration='none'`). A large intra-window gap sets
`off_text_flag` (the fight likely moved off-channel), which forbids any
resolution verdict downstream. Deliberately carries no role/aggressor/status
column — that posture is a later issue (C2). Read via `find_conflicts` /
`get_conflict`; guarded in `ingest_state` by `conflict_model`.

**Loose Ends — nudges (02, N1).** `loose_ends.py` mines the history for open
interpersonal loops and writes `loose_threads` to nudge about later. Shape:
Stage 0 is an exclusion gate that always runs first (durable `nudge_exclusions`
mutes + auto-mute of dyads whose most-recent llm-annotated exchange is negative,
`NUDGE_VALENCE_EXCLUDE`) — no candidate is ever generated for an excluded
contact; then pure-SQL detectors (N1 ships `faded_initiator` over the derived
`sessions` table); then Python urgency scoring
(`kind_weight * exp(-age/halflife)`, age from the corpus's latest message).
The one LLM touch is a one-line `nudge_text` narration, guarded by
`loose_ends_model`, which never blocks or deletes the SQL-detected rows.
`nudge_exclusions` is durable user intent — PRESERVED across `reset_db` — while
`loose_threads` is derived. Dismissals set `dismissed_at` and never delete (a
reused id could re-open a dismissed thread, hence AUTOINCREMENT). Read/act via
`get_loose_threads` / `resolve_loose_thread` / `mute_contact_from_nudges`.

**Fadeout Radar — drift (03, D1).** `drift.py` scores how each 1:1 relationship
is trending over time into `dyad_drift`, aggregated by **resolved counterparty**
(`member_key`, so a contact's several handles/chats fold into one relationship
before scoring; a full rebuild self-heals a `handle:N → contact:M` key flip).
Seven monthly signals (volume, reciprocity, latency, silence, valence, depth,
initiation), each turned into a 0..1 contribution by a Theil-Sen trend with a
bootstrap CI — a CI still spanning 0 contributes 0 — then combined into a
weight-renormalized composite `drift_score` and a `phase` label. Deterministic
and LLM-free; it consumes messages, the `sessions` table, and
`chunk_annotations`. **Opt-in: `DRIFT_ENABLED` defaults false**, so the pipeline
step is a no-op and `dyad_drift` never builds out of the box. This is enforced
inside `drift.py`, not by withholding the step — an enable-flag gates the BUILD,
not a prompt. Guarded by `drift_max_message_id` + `drift_params_hash` (a
skip-if-unchanged full rebuild). The read tools are a *separate* future issue's
`query_tools/drift.py` — a deliberate name-twin that shares nothing with this
compute pipeline; not yet landed, so nothing exposes drift to the LLM yet.

**Baseline — wellbeing (04, RES-49 / W1).** `wellbeing.py` is the deterministic,
hermetically testable core that decides whether a single conversational
identity's recent language + behavior shows a *sustained, cross-family*
departure from its own history. False positives are the enemy: gates stack
multiplicatively (per-marker winsorizing, family capping below the z-threshold
so no lone marker carries a family, and a concordance requirement across a
lexical and a behavioral family against a leave-window-out baseline). W1 is the
core only — no `config` wiring, no SQL, no delivery (those are W2/W3), and no
tool — so nothing surfaces wellbeing to the LLM yet; it is documented here as a
landed foundation, not a live feature.

**Voiceprint — persona (05, V1).** `voiceprint.py` builds a deterministic,
zero-LLM per-person voice profile into `person_voiceprint`: per-message style
(avg length, lowercase ratio, exclaim/ellipsis rates, burst size), the nine
function-word rates (n_words-weighted from `person_period_lexicon`), signature
openers/closers, and a **LOW-TRUST** emotional tone read from source='llm'
chunks only (NULL when there are none). Keyed by `member_key` × `counterpart_key`
× scope (`dyad`/`group`/`all`), so "my voice with Mom" and "Mom's voice with me"
are distinct rows; a confidence flag reads `high` only when a row clears all
three volume/engagement/breadth floors. Full rebuild, guarded by
`voiceprint_max_message_id` (and, from 2026-07-30, `voiceprint_annotations_hash`
— see the entry at the end of this file); read via `get_voiceprint` (which can
also recompute over a date window, capped). Later persona issues (draft-reply, style-match,
simulate) will extend `query_tools/persona.py`, which stays a thin read layer.

## 2026-07-21: the analytical suite (Waves 2–4 — surfaces, read tools, generative tools)

The Wave 0+1 foundations above are deterministic cores and their first read
tools; Waves 2–4 (same *Close Read* project, same Master Spec) put the last
mile on each feature — the LLM-facing read tools the compute pipelines were
built for, the generative tools that speak in the user's own material, and the
Wrapped surfaces — each still additive behind its `# --- feature: X ---`
fences. Default LLM tool count rose from 27 to **32**
(`len(tool_schemas.TOOL_DEFINITIONS)`); five more register only behind flags
(three drift tools + D3 behind `DRIFT_ENABLED`, `trace_conflict_across_chats`
behind `CONFLICT_TRACE_ENABLED`, and `simulate_contact` behind
`PERSONA_SIM_ENABLED` — the last added in V5 below). No new tables landed —
every Wave 2–4 tool and detector reads the six derived tables Wave 0+1 already
built.

**Clearing the Air — conflict (C2–C5).** `diagnose_conflict` (RES-51) is a
DETERMINISTIC scorecard over one episode — per-side I/you and we-talk ratios,
LSM, who opened / had the last word, latency + volume read off C1's stored
evidence, and a non-blame demand-withdraw read — carrying no aggressor/who-won
verdict. It introduced the **softened, code-enforced DV floor** (Master Spec
A10): `_dv_screen` scans a window's full text for threat/isolation/fear markers
and returns `clean` / `role-suppress` / `bypassed-with-footer`; a role-suppress
hit withholds ONLY the demand-withdraw read while the mechanical scorecard
still returns, and the hotline `resource_footer` is never removed (a bypass
restores the role read but still appends it). The scan is read-only and never
persisted. `draft_apology` (RES-53) SCAFFOLDS a repair message — the Lewicki-6
checklist ranked by measured effectiveness, the user's REAL grievance excerpts
pulled verbatim from labeled turns, and (only when the `repair` label class has
cleared its F4 kappa gate) the other person's ignored repair bids — and emits
no finished prose (`emits_prose` always False). Because drafting an apology is
itself the risky output, its DV floor is stricter than the scorecard's: a
role-suppress hit WITHHOLDS the entire scaffold and returns only the resource
pointer. `trace_conflict_across_chats` (RES-52) is an opt-in cross-chat
spillover trace, registration-gated behind `CONFLICT_TRACE_ENABLED` (its
per-call `consent` is a UX device, not the boundary), name-referenced, and
never persisted. The Wrapped **Repair Report** (RES-54) aggregates the conflict
tables into per-class stats, every one eval-gated (a class shows only when
`conflict_labels` cleared it) and off-text episodes excluded from recovery-time.

**Loose Ends — nudges (N2–N5).** Four detectors joined `faded_initiator` in the
`DETECTORS` registry, each pure detection behind the shared Stage-0 exclusion
gate: `dormant_tie` (RES-55, a once-close 1:1 tie gone quiet past its own p90
cadence; written but hidden until the durable `nudges_reconnect_ack` opt-in),
`unanswered_question` (RES-56, a session ending on the other person's genuine,
length/stoplist-gated question, with a mandatory cross-session answered-check),
`dropped_plan` (RES-57, an FTS cue-phrase pre-filter + a Searle-style LLM
verifier — HARD-GATED off by default AND, even on, writing nothing until the F4
`dropped_plan` precision gate clears at ≥ 0.85, which ships no seed so it stays
suppressed), and `unreciprocated_goodnews` (RES-57, opt-in, reading
`response_class` out of `chunk_annotations.events_json` for source='llm' rows
only and reporting llm-coverage so "no good news" ≠ "not annotated").
`inferred_occasion` (RES-58) infers a recurring birthday/anniversary from
congratulatory messages on the same calendar date across years and is the one
detector that fills `loose_threads.due_by` (the next upcoming occurrence). The
Wrapped **Open Loops** board (RES-58) surfaces the digest, honoring the same
exclusions.

**Fadeout Radar — drift (D2–D4).** The read layer the Wave-1 doc marked "not yet
landed" now exists: `query_tools/drift.py` (RES-59) reads `dyad_drift` verbatim
(no re-scoring, no Ollama) as `get_relationship_drift` (one resolved
counterparty) and `get_fadeout_watchlist` (a leaderboard partitioned by
coverage — the rankable tier never ranked against the insufficient-signal
bucket), both code-emitting the "still talk elsewhere?" channel caveat on any
cooling/fading/dormant phase and narrating strictly at the pipeline's
`confidence_tier`. `get_reconnection_candidates` (RES-65 / D3) surfaces curated
shared references (dyad_terms + high-valence llm shared moments) for a grounded
opener or refuses rather than fabricating; it writes nothing. All are the
name-twin of the compute `drift.py` and import nothing from it, and all are
schema-gated behind `DRIFT_ENABLED`. Calibration + the `--purge` /
`make drift-purge` teardown and heuristic defaults landed alongside (RES-60).
The Wrapped **Who's Drifting** section (RES-66) is opt-in (DRIFT_ENABLED-gated),
corroborated-only, and excludes the same silent contacts as the roster.

**Baseline — wellbeing (W2–W4).** `get_language_trajectory` (RES-61) wires the
W1 change-point core to the DB: count-based markers pooled across a scope, self
by default and per-contact by default too (default-open), refused in code only
for an id on the `WELLBEING_DENY_IDS` deny list (`consent.is_denied`). (Revised
from the original default-closed `WELLBEING_CONSENT_IDS` allowlist to a deny
list: analyze anyone unless explicitly suppressed.) Framing is
enforced by code, not prompt (RES-67 / W3): the tool emits a code-built
`support_block` via the generic `_epilogue` hook that `chat.run_turn` strips
from the model's view (along with the duplicate `support_block` field) and prints
verbatim, and the turn holds back streaming to run
`wellbeing.sanitize_condition_names` over the WHOLE reply before emitting — a
later revision (PR #32 review) widened the original final-sentence-only buffer,
since a condition named in an earlier sentence would otherwise slip through
prompt-only; the envelope carries a model-safe `flagged` boolean as the
transcript-reader's departure signal now that `support_block` is hidden. An S10
evalkit case guards the behavior. The opt-in Wrapped wellbeing section (RES-68 /
W4) reuses the same support footer + narration scrub and stays off unless
`--wellbeing` / `WRAPPED_WELLBEING`.

**Voiceprint — persona (V2–V4).** `draft_reply` (RES-62) drafts a reply in the
user's own voice grounded ONLY in real retrieved reply pairs from that 1:1 chat,
declining structurally (no draft) below the context-pairs floor.
`check_style_match` (RES-63) scores a user-written draft against a stored
voiceprint DETERMINISTICALLY (feature-distance + rule-based top-3 divergences —
no model judges it). The Wrapped **Your Voiceprint** card (RES-64) renders the
reader's own ("me", "\*", "all") card, plus a consent-gated "guess who texts
like this" game that is default-redacted: with an empty
`WRAPPED_VOICEPRINT_CONSENT_IDS` no third-party card appears, and consent only
adds a contact — it never overrides a roster exclusion.

## 2026-07-21: persona V5 — `simulate_contact` + past-self (RES-69)

`simulate_contact` extends `query_tools/persona.py` (still a thin read layer plus
this one in-tool generator) with an ESTIMATE of how a named contact might reply in
their voice, or a "talk to your past self" mode as of a date. It follows the
persona-V2 `draft_reply` architecture (in-tool `ollama.chat_once`, structural
decline below `VOICEPRINT_MIN_CONTEXT_PAIRS`) and the established registration
gate (`DRIFT_ENABLED` / `CONFLICT_TRACE_ENABLED`): the schema is withheld from
`TOOL_DEFINITIONS` + `SYSTEM_PROMPT` unless `PERSONA_SIM_ENABLED` (default false),
while the function stays unconditionally importable/re-exported — so the default
tool count stays **32** and rises to 33 only when the flag is on. This is the
fifth flag-gated tool (see the Waves 2–4 count above).

Guardrails, all in code (prompting is not a control here): (1) the returned
`simulation` is wrapped POST-generation — every non-empty line prefixed with an
`[ESTIMATE …]` label by `_wrap_simulation` — so the outer chat loop only ever
relays a pre-labeled artifact, never an unlabeled draft it could strip;
(2) a hard deceased/grief denylist `PERSONA_SIM_BLOCK_IDS` (parsed via the shared
`consent.is_consented`, membership = blocked) hard-refuses in code with no
generation — only a resolved contact id can be listed, since a bare handle can't
be marked; (3) a soft, fail-open dormant/deceased warning
(`PERSONA_SIM_DORMANT_DAYS` + `PERSONA_SIM_ONESIDED_MIN_MSGS`) rides in the
acknowledgment payload so it is seen before acknowledging — it warns, never
blocks; (4) a living-contact `acknowledged=true` UX flow (documented as a flow,
NOT the boundary — the registration gate is); (5) past-self is bounded by
`VOICEPRINT_MIN_RANGE_MESSAGES` / `VOICEPRINT_SCAN_MAX_MESSAGES` over a
date-scoped raw rescan; (6) the payload also carries a tapback base-rate
prediction computed over a FULL-MESSAGE denominator (every real message received
from the contact, NOT only messages that drew a reaction — that denominator
conditions on a reaction and overstates every rate). An empty generation is
treated as a low-confidence decline, never emitted as `generated: true` with a
blank `simulation`. Nothing sends, nothing is persisted.

## 2026-07-21: Apple Foundation Models backend (chat + embeddings, no-Ollama capable)

Goal: make Apple's on-device Foundation Model (Apple Intelligence, Apple
Silicon, macOS 26+) a selectable backend through the *existing* model env vars,
so the app can ship and run with **no Ollama models installed**. Opt in per
pipeline: `CHAT_MODEL=apple`, `EMBED_MODEL=apple`, `ANNOTATE_MODEL=apple`,
`TOPIC_MODEL=apple`, `WRAPPED_MODEL=apple`. Defaults are unchanged (gemma4 for
chat, nomic-embed-text for embeddings), so apple is never on unless asked for.

### Transport: a compiled Swift shim, not a pip or a server

- **`apple-fm-sdk` (the official pip)** — alpha, and Python ≤3.12 only. This
  repo runs 3.13/3.14 (`requires-python`), and the pip needs a full Xcode.
  Rejected.
- **`fm serve` (an OpenAI-compatible local server)** — macOS 27 only; the
  target is macOS 26. Rejected.
- **A local Swift shim, compiled with Command Line Tools** — chosen.
  `FoundationModels.framework` and `NaturalLanguage.framework` ship in the CLT
  SDK (`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`), so
  `swiftc -O -parse-as-library` builds `bin/applefm-shim` with **no Xcode**
  (verified: macOS 26.5.2 / CLT Swift 6.3.3 / arm64). `make apple-shim`
  compiles it and runs an availability self-check; kept out of `make install`
  (same philosophy as `install-ollama` — a slow, platform-specific step).

The shim is one persistent subprocess speaking JSON-lines over stdio
(`applefm.py` ↔ `bin/applefm-shim`), `id`-correlated, one request in flight
behind a lock, respawn-and-retry-once on a crash. Three ops: `availability`,
`respond` (fresh `LanguageModelSession` per call; guided generation from a
`DynamicGenerationSchema` built out of a small JSON-Schema subset), and `embed`
(`NLContextualEmbedding`). Selection is the `apple` sentinel
(`config.is_apple_model`); `ollama.py` and `embeddings.py` dispatch on it so
every other caller stays backend-agnostic.

### The 4,096-token window drives the whole chat design

The FM context window is a **fixed 4,096 tokens** (Apple TN3193; unchanged as
of the 26.4 SDK). The Ollama chat setup does not remotely fit: system prompt
~1,023 tok + the tool schemas ~7,272 tok (measured at the then-21 tools; the
toolset has since grown to 32, so the overhead is larger still) = **well over
8K tok, 2×+ the window, before the user types anything**. Native tool-calling
is impossible.

So the apple chat path (`chat_apple.py`) stages many small, fresh,
single-purpose sessions per turn and carries state in Python: **ROUTE** (pick
the next tool or `answer` from a compact first-sentence catalog, guided against
an enum) → **FILL** (generate that tool's args, guided against its simplified
`input_schema`) → **EXEC** (run the `query_tools` fn locally, reusing
`chat.py`'s registry + dedup guard) → **DIGEST** (compress the result to
question-relevant facts, slicing at row boundaries so a result bigger than the
window is still read whole — the raw result never re-enters a prompt) →
**ANSWER** (from the accumulated facts alone, streamed).

Two findings surfaced only against the real windowed model (the hermetic fake
shim can't reproduce them):

1. **`format="json"` cannot be a guided schema.** Ollama's `format="json"` is
   grammar-constrained-to-*some*-JSON; `DynamicGenerationSchema` has no "any
   object" — a schema with empty `properties` pins the reply to literally
   `{}`, which silently returned an empty object for every batch caller
   (annotate/topics). The fix mirrors what Ollama's mode actually leans on:
   JSON mode folds a directive into the system message and generates **free
   text**, and `chat_once_json` parses + nudge-retries. The batch callers spell
   out their exact JSON shape in their own system prompts, so it works —
   measured live: topics returns `{"label": "Pizza Friday"}`, annotate returns
   `{"valence": 1.0, "depth": 4}`.
2. **The scratchpad, not just each tool result, must be bounded.** Each result
   is capped (`APPLE_TOOL_RESULT_MAX_CHARS`, 5000), but several verbatim
   results *stack up* past the window, and a later ROUTE prompt (which also
   carries the ~2.8K-char catalog) then overflowed and crashed the turn. Fix:
   after every tool the scratchpad is **compacted** back under a char budget by
   one digest pass, and every stage catches the shim's `ContextOverflowError`
   and degrades (route → answer, fill → defaults, answer → compact-and-retry,
   finally the question alone) rather than raising out of the turn.

Verified end-to-end on the target machine: a total-count analytics question and
a find-and-read question both complete and answer without a crash.

### Embeddings: NLContextualEmbedding, 512-dim, retrieval quality unproven

`EMBED_MODEL=apple` embeds via `NLContextualEmbedding` (latin-script model) in
the shim: per-token vectors mean-pooled + L2-normalized, long texts segmented
and segment-pooled. Measured **512-dim, ‖v‖=1.0** through the full Python path
(`embed_texts` / `detect_embed_dim`), including a multi-segment ~8K-char text.
Task prefixes are NOT applied (nomic's `search_document:` / `search_query:` are
model-specific). Switching `EMBED_MODEL` is always `reset_db.py` + re-ingest —
the `ingest_state` embed guard already enforces it; nothing new was added.

Caveat, stated so it can be measured rather than assumed: `NLContextualEmbedding`
is a general contextual encoder, **not contrastively trained for retrieval** the
way nomic-embed-text is. Whether 512-dim NL embeddings retrieve as well as nomic
on this corpus is unproven; ship it measurable, never as the default.

### Known limits

- **Routing across the full toolset (32 tools) is the quality ceiling.** The
  on-device model is small; live runs show it inventing narrow date filters and
  occasionally a wrong scope. The staging is correct and never crashes, but multi-hop
  analytics accuracy is bounded by 3B-class routing. `eval/evalkit` records the
  apple path through a parallel recorder hook (`chat_apple._recorder`) so the
  sweep works under `CHAT_MODEL=apple`; quantifying the scorecard vs the gemma4
  baseline is a follow-up.
- **`think` is ignored** on apple (Foundation Models has no reasoning-effort
  flag): accepted and dropped at the seam, never forwarded.
- **Batch `--terms`** is left unclamped: a default `ANNOTATE_TERMS_BATCH_SIZE=25`
  batch can exceed the window on apple (it degrades to skipped-and-retried
  rows, not a crash). Set a small `ANNOTATE_TERMS_BATCH_SIZE` for apple runs.
- **The `chunk_annotations` model guard applies to apple too**: switching
  `ANNOTATE_MODEL` to/from apple against a non-empty table is refused (mixing
  annotation models skews every downstream aggregate).

## Linting

`ruff check .` (config in `pyproject.toml`: `line-length = 100`,
`select = ["E", "F"]`), or `make lint`. Runs in CI as the `lint` job in
`.github/workflows/ci.yml`, alongside the `compile-check` and `test` jobs
(the latter runs `python -m unittest discover tests` — hermetic, no Ollama
or real `chat.db` needed).

## Next steps / not yet built

1. **Finish embeddings**: a partial run exists (~3.5k vectors in
   `message_vectors`), so this is started but not complete. To fill in the
   rest: `ollama pull gemma4:26b-a4b-it-qat`, then `python3 ingest.py --since-last`
   on a machine where Ollama is actually reachable, until every message is
   embedded.
2. **Wire `TOOL_DEFINITIONS` into a tool-calling loop** — DONE: `chat.py`
   implements this against `gemma4:26b-a4b-it-qat` via Ollama. Remaining
   option: a Claude-backed reasoning/analysis layer over the same tools
   (keeping embeddings local) if preferred for heavier analysis.
3. **Enrichment/analysis layer** (sentiment, emotion, topic tagging) — DONE,
   see "the analytics layer" above. Built one level up from the original
   sketch: annotations are keyed to *chunks*, not messages, and live in their
   own `chunk_annotations` / `topics` tables rather than as extra columns on
   `messages` — same reasoning as embedding chunks instead of messages ("ok"
   has no sentiment worth storing), and it keeps the append-only source table
   free of derived, rebuildable values. `annotate.py` and `topics.py` are
   standalone overnight-scale CLIs, not part of `ingest.py`: they need a
   reachable Ollama and hours, while ingest must stay runnable offline.
4. **`.txt` parser** — not built; only needed if `chat.db` access isn't
   available for some future conversation/export.

## 2026-07-21: review follow-ups — drift rebuild guard + nudge reconciliation

Two correctness fixes from the PR #32 ("Close Read") review.

**Drift rebuild guard widened (`drift.py`).** `build_drift` skips when neither
the corpus high-water mark nor a params hash changed. Two scoring inputs weren't
covered, so a change to either silently served stale scores until `--force`:

- `_params_hash()` hashed only `DRIFT_WEIGHTS` + `DRIFT_PHASE_THRESHOLDS`, but
  scoring also consumes `alpha` (`DRIFT_SLOPE_CI`) and the four `DRIFT_MIN_*`
  gates — `alpha` drives every signal's CI/significance and thus `drift_score`,
  the per-signal gates, `phase`, and `confidence_tier`. It now hashes the full
  `_current_config()`, so retuning any scoring param forces a rebuild.
- The guard was blind to `chunk_annotations`. `annotate.py --upgrade-distilled`
  rewrites existing rows in place (`INSERT OR REPLACE`, same `chunk_id` PK):
  `source` flips `distilled → llm` and `valence`/`depth` change with **no new
  messages**, so both the message-id and params guards stayed identical while the
  valence signal + `confidence_tier` should have moved. A new
  `drift_annotations_hash` guard key fingerprints exactly the rows
  `_load_annotations` consumes — `COUNT` plus `SUM(source='llm')`,
  `SUM(valence)`, `SUM(depth)` (a plain `COUNT` can't detect an in-place
  upgrade). `purge_drift`/`make drift-purge` clears all three keys.

**Nudge reconciliation exempts degraded nondeterministic detectors
(`loose_ends.py`).** `_resolve_stale_threads` auto-resolves any open thread a run
did not re-detect — sound for the deterministic SQL detectors, wrong for
`dropped_plan`, the one detector that re-verifies via Ollama each run. When
`_verify_plan` returns `None` (unparseable/empty JSON after retry) or `""` (dict
missing `verdict`), the candidate is suppressed and the thread would flip to
`resolved` — a transient model hiccup silently closing a real loose end. Fix: a
module-level `NONDETERMINISTIC_KINDS = {"dropped_plan"}`; `detect_dropped_plan`
sets `cfg["dropped_plan_degraded"]` on a no-usable-verdict verify; `detect_all`
builds the degraded-kind set and `_resolve_stale_threads` skips those kinds
(`AND kind NOT IN (...)`). A clean `fulfilled`/`unknown` verdict is a legitimate
resolve and does NOT set the flag, so genuine closes still reconcile — the
exemption only defers to the next clean run, never resolves wrongly. (A hard
socket timeout instead raises and crashes the run before the reconcile/commit
step, so it can't silently over-resolve; left as-is.)

## 2026-07-27: `--quality hybrid` — annotation on a wall-clock budget

Between `distilled` (~90 s once seeded, kNN everywhere, **no events**) and
`fast` (~16 h corpus-wide on this machine) there was nothing. `hybrid` fills
the gap, and its defining choice is that it is specified in **wall clock, not
chunks**: `ANNOTATE_BUDGET_MINUTES` (empty/0 → 25 on Ollama, 50 on Apple FM,
resolved per run from `config.is_apple_model`). Four phases under one deadline
(`annotate_hybrid.hybrid_chunks`):

1. **seed top-up** — widen the LLM seed pool to `ANNOTATE_HYBRID_SEED_TARGET`
   (800) with `annotate_sampling.stratified_sample` instead of distilled's
   `random.sample`, batched like any other LLM phase and capped at
   `_SEED_PHASE_SHARE` (25%) of the budget.
2. **kNN classify + confidence backfill** — the distilled pass, corpus-wide,
   now also writing `chunk_annotations.knn_confidence`.
3. **3a event upgrades** — LLM-re-annotate the chunks
   `annotate_events.event_candidate_chunk_ids` flags, up to 60% of the time
   left after phase 2.
4. **3b confidence upgrades** — LLM-re-annotate the lowest-`knn_confidence`
   distilled rows with the rest.

Phases 1–2 are corpus-wide by design (the kNN vote has to generalize, so the
scope flags deliberately don't reach them). **Every phase is capped**, phase 2
included — see "Phase 1 is not preamble" and "Phase 2 is not free" below.
Phase 3 writes `source='llm'`, which is what
wakes the valence-trusting readers (conflict corroboration, the nudge Stage-0
exclusion, voiceprint tone) for exactly the chunks the budget reached.
`--upgrade-distilled` is rejected with `hybrid`: hybrid *is* a
distilled-upgrader.

**The runtime promise is structural, and the first cut of it did not hold.**
The intent — stop dispatching at a monotonic deadline, so the target (< 30 min
on qwen3.5:4b, < 1 h on Apple FM) is met by construction on any hardware and
any corpus size — is right, but two exemptions in the first implementation
falsified it in measurement, and both are now closed:

- Phase 1 was exempt from the deadline on the theory that it was cheap. It is
  not; unbatched it spent 10.7 min of an 8-min budget, and phase 3 never ran.
- "Dispatch stops at the deadline" ignored the pool's own backlog, which is
  worth minutes, not the "one group" the comment claimed.

Review of PR #42 found two more exemptions of exactly the same shape, both
reproduced and both now closed — phase 2 was ungated (below), and the gate's
"there is nothing to estimate from before the first completion" fallback let
the first wave of every phase go out blind, which on a 2-min budget meant
4.4 min of actual work. Together they are the difference between the promise
being *argued* and being *measured*.

### Phase 1 is not preamble: batch it, then cap it

Measured 2026-07-27, `ANNOTATE_BUDGET_MINUTES=8`, qwen3.5:4b, 25,029-chunk
corpus with 450 existing seeds: the 350-seed top-up dispatched **one chunk per
request** (`_seed_chunks` built `[[row] for row in sample]`) and cost
**641.9 s = 10.7 min at 1.83 s/chunk**. Phase 2 was genuinely cheap
(`_classify_chunks` 0.01 s with nothing pending, `backfill_confidence` 17.9 s
for 24,229 rows **against the ~800-seed pool of the day** — see "Phase 2 is not
free" for what that qualifier hides), 3a/3b never dispatched a single call, and
the run ended at 11.0 min against an 8-min budget. On a fresh corpus (800 seeds) that
extrapolates to ~24 min — the whole 25-min default — leaving the tier
indistinguishable from `distilled` with a bigger seed pool.

Two changes, both in `annotate_distill._seed_chunks`, both no-ops at their
defaults so `distilled` is untouched:

- **`batch_size`** (default 1). Hybrid passes its preset's 8. `batch_size == 1`
  deliberately keeps building `[[row] for row in sample]` literally rather than
  calling `annotate._annotation_groups`, because that helper's apple branch
  ignores `batch_size` and packs by char budget — routing the seed pass through
  it unconditionally would have silently re-batched the *distilled* tier on
  Apple FM.
- **`deadline`** (default None = today's ungated behavior). Hybrid passes
  `_SEED_PHASE_SHARE = 0.25` of the budget. A pass cut short says so in its own
  "Seeded N annotations" line (with the pool level against the target), and the
  next run tops up.

The 25% split is sized from the measured numbers: 0.25 × 25 min = 6.2 min, and
a fresh 800-seed pool at the measured batched seed rate costs well under that,
so the cap normally never binds; when it does bind (a slow machine, Apple FM),
it binds on the phase the tier does *not* exist for. Consequence handled in
`hybrid_chunks`: a seed pass cut short on a fresh DB can leave the pool under
`annotate_distill._KNN_MIN_SEEDS` (100), where `_build_seed_index` rightly
refuses. The floor is not weakened — the *outcome* is: `hybrid_chunks` catches
that `SystemExit`, prints the per-phase summary and a resume hint, and returns.
Anything else the kNN phases raise (no `chunk_vectors`, no `embed_dim`) still
propagates, since no rerun fixes those.

### Phase 2 is not free either: O(rows x seeds)

The first cut left the kNN phases ungated because they are deterministic and
`backfill_confidence` had measured 17.9 s. That number was taken against an
~800-seed pool, and the pass is **O(n_distilled x n_seeds)** — one vec0
`MATCH ... k=?` per row against a `temp.seed_vectors` holding *every*
`source='llm'` row. Re-measured 2026-07-28 on the same 25,029-chunk corpus and
machine, varying only the pool:

| seed pool | rows | wall | rows/s |
|---|---|---|---|
| 450 | 24,579 | **14.6 s** | 1,688 |
| 4,416 | 20,613 | **82.0 s** | 251 |

A 6.7x per-row slowdown for a 9.8x wider pool — and the pool widens with every
run of this tier, because phase 3 writes `source='llm'`. End to end that made a
1-min budget take 1.3 min and buy nothing.

Both passes now take the run deadline and stop at it (checked on a 256-row
stride, which is under a second of work at any measured rate). They were always
resumable — `_classify_chunks` only looks at chunks with no annotation,
`backfill_confidence` only at rows whose confidence is NULL — so stopping short
just defers.

The trap, and the reason this is not a one-line change: `hybrid_chunks` used to
detect a Ctrl-C inside phase 2 by *counting leftover rows*
(`_knn_work_pending`) and skipping phase 3 when it found any. Gate phase 2
without touching that and a deadline-stopped phase 2 disables every LLM upgrade
forever — the exact failure the tier exists to avoid. Both passes now report
`stopped` and `interrupted` through a caller-owned dict instead, which is
exact, cheaper (it deletes two full aggregate scans per run) and cannot drift
from the passes' own `todo` queries. Measured after: a 1-min budget lands at
**1.0 min** with the backfill stopping at 15,616 of 20,613 rows, and the next
run finishes the remaining 4,997 **and reaches phase 3a**.

### The deadline has to reserve the pool's drain

`annotate._pooled` keeps `2 × workers` groups in flight (one when sequential),
and every one of them still finishes, writes and commits after the generator
stops yielding. So "check the clock before each dispatch" overshoots by a whole
drain. Measured at the default `ANNOTATE_WORKERS=4`, batch 8: **+1.5 min on a
5-min budget, +2.0 min on 2 min, +3.7 min on a 1-min apple budget** — and phase
3a blew its own 60% sub-deadline the same way (72 s allotted, 239.7 s spent),
starving 3b to zero upgrades in one run.

`annotate._DispatchDeadline` (shared by phase 1 and phase 3) stops dispatching
a drain early instead. It wraps the per-group request to measure completed
groups, and reserves `(in_flight / workers) × mean_group_seconds` — the pool's
backlog retiring `workers` at a time. Measuring *latency under the run's own
concurrency* is what makes it self-correcting on a serializing Ollama, where
four concurrent requests each measure ~4× a lone one and dividing by `workers`
puts the throughput back. No config knob — it is an accounting rule, not a
preference.

**"The first dispatch of a phase can only ever be made blind" was wrong, and it
was the whole residual overshoot.** With nothing measured, `reserve()` returned
`0.0`, so the gate waved through groups until `_pooled` was holding its
`2 × workers` ceiling — 64 chunks at batch 8 — and only then got an estimate,
by which point the work was already committed. Reproduced on the reference
corpus with phases 1–2 both no-ops, so the clock was entirely phase 3's:

| budget | before | after (cold) | after (rerun) |
|---|---|---|---|
| 2 min | **4.4 min**, 72 event + 0 confidence | 0.0 min, 0 + 0 | — |
| 6 min | 5.8 min, 80 event + **0** confidence | 3.3 min, 40 + 0 | **4.8 min, 32 + 32** |

The fix is memory, not a guess: `_load_group_rate`/`_save_group_rate` carry the
measured seconds-per-request across runs in `ingest_state`, keyed
`annotate_group_seconds:<phase>:<model>:b<batch>:w<workers>`. All four parts of
the key matter — the reserve is denominated in *latency*, which batch and
workers both move, and the phases are genuinely different populations (a group
of 8 averaged **106.4 s** on the event candidates against **70.6 s** on the
confidence upgrades, whose chunks are a fifth the length). One shared estimator
would let whichever phase ran first anchor the others wrong in both directions,
which is also why the earlier "one `_GroupRate` for the whole run" is gone.

Two details that measurement forced:

- **What crosses runs is the slowest group, not the mean.** Carrying the mean
  forward was tried and measured *worse*: a run whose event phase averaged
  106.4 s met 148 s on the next run, under-reserved, and landed at 6.8 min on a
  6-min budget. The failure here is one-sided — under-buying is recovered by a
  rerun, an overrun breaks the only promise the tier makes — so the stored
  number is `max`, while the live mean still wins inside a run.
- **A cold DB still needs a prior.** `_COLD_GROUP_SECONDS_PER_CHUNK = 5.3`
  (× batch × workers ⇒ 169.6 s at the defaults) is sized to the slowest phase
  measured, not the average: slowest group 160.3 s seeding, 152.9–198.1 s on
  event upgrades, 77.7–107.3 s on confidence upgrades. It governs only until
  the first group completes.

**The consequence to state plainly, because the docs did not:** one LLM request
is the smallest unit that can be dispatched, so a phase whose share of the
clock is under one request's latency buys *nothing*. That is correct — starting
work that cannot finish is what "wall-clock budget" is supposed to rule out —
but it means budgets under roughly five minutes buy no upgrades at all, and
`_run_upgrades` now prints why and names the knob. The 25-minute default is
clear of the floor by 5×.

### Measured basis (2026-07-27, this machine, warm model)

Corpus: 25,029 chunks, avg 693 chars (median 293); top-20 dyads = 19,483 chunks.

| Config | s/chunk (warm) |
|---|---|
| qwen3.5:4b, batch 8, `think=False` | **1.1–1.4** |
| qwen3.5:4b, batch 16/24, `think=False` | 1.9–2.6 |
| qwen3.5:4b, batch 8, `think='low'` | **times out** |
| qwen3.5:4b, 4 workers × batch 8 | 2.4 |
| Apple FM, char-budgeted batch (8–10 chunks) | 2.6–5.6 |

Two preset consequences, both against the grain of the older tiers:

- **`QUALITY_PRESETS["hybrid"] = (False, 8)` — `think=False`, not `'low'`.**
  `fast`/`balanced` use `'low'` because they were tuned on gpt-oss, which has
  real effort levels. qwen3.5:4b is a *toggle* model: `'low'` is merely truthy
  there, so it buys a full reasoning pass and the request times out. This is
  the `think` tri-state gotcha (an effort level is not portable) showing up as
  a preset, not a bug.
- **Batch stays 8.** 16 and 24 measured *worse* per chunk (1.9–2.6 s vs
  1.1–1.4 s): the longer prompt costs more than the saved round-trips. Workers
  are kept plumbed but bought nothing here (2.4 s/chunk at 4 × 8) because this
  Ollama serializes.

Those planning numbers were optimistic. Re-measured on the production path
after the deadline fixes, per *request* rather than per chunk, because a
request is the unit the gate can actually decline (25,029-chunk corpus,
`ANNOTATE_WORKERS=4`, batch 8, 2026-07-28):

| phase | mean latency / request of 8 | slowest seen |
|---|---|---|
| 1 seed top-up | — (160 seeds in the 25% share of a 24-min budget) | 160.3 s |
| 3a event upgrades (~1,900-char chunks) | 106.4 s | 198.1 s |
| 3b confidence upgrades (median ~360 chars) | 70.6 s | 107.3 s |

3a is the expensive one because a batch of 8 long chunks often comes back
covering only some ids, and every miss costs its own single-chunk request.

**Two different 6-minute runs, on two different DB states** — quoted together
because the pair is the point, and stated as a pair here because an earlier
draft of this entry read as one run contradicting itself:

| 6-min run | starting state | wall used | upgrades |
|---|---|---|---|
| rerun of the reference DB (the "after (rerun)" cell above) | seeds at target, `knn_confidence` already backfilled, so phases 1–2 were no-ops | 4.8 min | **32 event + 32 confidence** |
| fresh copy of the reference DB | 450 seeds, no `knn_confidence` column — schema self-healed, seed pass declined (its 90 s share is under one request and it says so), 24,579 rows confidence-backfilled | 3.4 min | **32 event + 40 confidence** |

The fresh copy buys *more* upgrades in *less* wall clock because its seed pool
is 450 against the reference DB's wider one, and phase 2 is O(rows × seeds). At
`ANNOTATE_BUDGET_MINUTES=24` on the same fresh copy the seed pass does clear the
floor and drew **160 seeds** inside its 25% share. A cold model load needs no
allowance: the deadline absorbs it and simply buys fewer upgrades.

**Superseded 2026-07-28 — the 25-minute extrapolation.** This entry originally
projected "roughly 250–350 upgrades per run" from those 6-minute runs, on the
pre-round-2 code. Round 2 measured **284 upgrades in a 7.7-min run**, projecting
**~424–563** at 25 minutes; that is the number to plan against (see the
validation table in "2026-07-28: annotation throughput, round 2", where it is
explicitly labelled extrapolated). Neither figure is an end-to-end 25-minute
measurement, and `config.py`/`.env.example` now quote only the round-2 one.

### `_APPLE_ANNOTATE_CHAR_BUDGET`: 12,000 → 8,000 → 2,000 (global, every tier)

Measuring the apple path turned up a pre-existing defect rather than a hybrid
one — and the first fix for it (12,000 → 8,000, chosen from 3 overflows in 18
chunks) **did not work**. At 8,000, measured on the real applefm corpus,
14 of 18 batched calls (78%) came back unusable, 7+ raised
`context_overflow`, and 6 of 66 chunks (~10%) were lost when the single-chunk
fallback failed too. Re-measured on the production path
(`_annotate_group`: batch, then a single-chunk retry for every id the batch
missed) over spread samples of both phase-3 populations:

| budget | batched-call failure | `context_overflow` | chunks dropped | s/chunk |
|---|---|---|---|---|
| 8,000 | 9/9 = **100%** | 4 / 48 chunks | 1 (2%) | 4.3–5.2 |
| 4,000 | 6/7 = 86% | 1 / 24 chunks | 1 (4%) | 5.5 |
| 2,000 | 10/14 = 71% | 3 / 144 chunks | 5 (3%) | 1.6–4.1 |

Two things the earlier note got wrong. The window covers **prompt *and*
response**, so ~8,000 chars of chunk text plus `_ANNOTATE_SYSTEM` +
`_BATCH_SYSTEM_SUFFIX` plus JSON for 3–5 chunks does not fit. And overflow is
not even the main failure: dumping the raw replies shows the batched failures
are **malformed JSON** — `{"a": {...}, {"b": ...}` (an extra brace before the
second key), ```` ```json ```` fences, `"chunk_<id>"` keys instead of `"<id>"`.
An apple batched annotate request usually fails *whatever its size*, and each
failure costs its own wall time **plus** a single-chunk retry for every chunk in
it. Per-chunk time therefore falls as the budget falls, which is the opposite of
the usual batching tradeoff.

2,000 is one `CHUNK_MAX_CHARS`: never two full-size chunks in one request, while
genuinely short chunks still pair up. Residual and **not** fixable by any
budget: `_char_budget_groups` always emits at least one row, so a chunk larger
than the budget is still sent alone — 1 chunk in a 32-chunk sample (3%)
reproducibly overflows on its own, because ~2 kB of emoji/non-Latin text
tokenizes far worse than the ~4 chars/token English rule of thumb. That row is
lost for the run and retried on the next one.

This constant is shared by **all** tiers (`annotate._annotation_groups`), so the
change is global: `fast`/`balanced`/`max` on apple all pack smaller and measured
1.7–3.5× faster per chunk. It stays a module constant, not a config knob — it is
a property of the FM window, not something a user tunes.

**Open, not fixed here:** apple batched annotation is ~90% wasted work at every
budget measured. Making it pay would mean constrained decoding
(`applefm.respond(..., schema=...)`) or a per-chunk apple request path, both of
which change behavior for every tier and belong in their own change.

### `chunk_annotations.knn_confidence` (new nullable REAL)

`_knn_predict` returns a 4th value: `mean(w) * (1 - clamp(weighted std of the
neighbours' valence, 0, 1))`, where `w = max(1e-6, 1 - d)` are the existing
distance weights. High only when the neighbours were both **close** and **in
agreement**, deterministic, and bounded to [0, 1] by construction (cosine
distance is 0..2, so every `w` ≤ 1) rather than by clipping.

NULL semantics are load-bearing and the reason the column has no DEFAULT: NULL
on `source='llm'` rows (an LLM label needs no self-assessment) and on distilled
rows written before the column existed. Any numeric default would invent a
ranking signal. Phase 3b sorts `knn_confidence IS NULL, knn_confidence`, so
unscored rows stay eligible but sort last. An LLM upgrade goes through
`_write_annotation`'s `INSERT OR REPLACE`, which does not name the column — so
upgrading a distilled row correctly clears its confidence back to NULL.
`annotate_distill.backfill_confidence` scores pre-existing distilled rows
(UPDATE of that column *only* — relabeling committed valence/emotions/depth
would be a different, unrequested change), because `_classify_chunks` touches
only chunks with no annotation at all, which on any machine that has already
run `distilled` is none of them.

### The stratified seed sampler: what it actually buys

The plan predicted "higher kNN fidelity corpus-wide". **It does not, and that
claim should not be repeated.** Measured at need=800 against the full corpus
(768-dim nomic embeddings, 532 chats), probing with *every* chunk rather than a
sample:

| metric | `random.sample` | stratified |
|---|---|---|
| mean distance to nearest seed | 0.1503 | 0.1496 (−0.4%) |
| mean distance to the 10 nearest seeds (k = `ANNOTATE_KNN_K`) | 0.1830 | 0.1857 (**1.4% worse**) |

At 768 dimensions, 800 seeds over 25k chunks are already spread near-uniformly
by chance and no selection rule moves the average. An earlier 3,000-probe run
showed an ~11.5% covering-radius win; it did not survive full-corpus probing and
was sampling noise. Two real, smaller wins remain, and they are the whole
justification for the module:

- **~4× fewer redundant seeds.** Seeds whose nearest *other* seed is within
  0.02 cosine — a near-duplicate whose LLM call teaches the vote nothing:
  22/800 for `random.sample`, 5/800 here. In a tier whose premise is a
  wall-clock budget, recovering a few percent of the seed calls from duplicated
  work is worth ~4.5 s of preamble.
- **Deterministic, exactly proportional chat representation.** Over 25
  `random.sample` draws the number of chats represented ranged 92–117, summed
  deviation from proportional was 208.9 ± 12.0 rows, and the worst single-chat
  miss was 31.6 rows on the dominant dyad. The largest-remainder allocation here
  is deterministic and lands every chat within 1 row of proportional.

Two implementation notes worth keeping. The plan specified greedy
farthest-point selection; measured, FPS walks the convex hull and fills the pool
with outliers (nearest-seed 0.1455 → 0.1624, k=10 mean 0.1830 → 0.2102), so the
module uses **k-means++ D-squared sampling** instead — still reaches empty
regions, stays density-aware. And there is no numpy (it is not a dependency):
embeddings are projected once through a deterministic sparse
Johnson–Lindenstrauss matrix to 32 dims, which keeps the whole draw at ~4.5 s in
pure CPython against a budget measured in tens of minutes.

### The seed target counts sampler draws, not `source='llm'` rows

`_seed_chunks` sized its top-up from `SELECT COUNT(*) ... WHERE source = 'llm'`.
Every phase-3 upgrade is a `source='llm'` row, so **the tier's own output
satisfied its own seed target.** Reproduced on the reference corpus (4,416
`llm` rows): phase 1 printed `seed pool: 4416 LLM annotations (target 800,
labeling 0 more)` on every run, and `ANNOTATE_HYBRID_SEED_TARGET` plus the
whole of `annotate_sampling.py` never executed again.

Worse than a dead knob: phase-3 rows are drawn from the top-20 dyads, filtered
by a gratitude/good-news lexicon and by lowest kNN confidence, so the pool
`_build_seed_index` votes with drifts toward exactly the non-representative
shape `_seed_chunks`' own docstring warns about — and the module added to blunt
that bias is the thing that stops running.

Fixed with a `hybrid_seeds` counter in `ingest_state`, via a `counter_key`
argument that defaults to `None` (the distilled tier is unchanged). **Not a new
`source` value**, which was the other option: every valence-trusting reader in
the tree keys on `source = 'llm'` — `query_tools/`, `conflict_detect.py`,
`loose_ends.py`, `nudge_detectors.py`, `voiceprint.py`, `drift.py`,
`wrapped_sections.py`, `eval/`, plus `schema.py`/`reset_db.py` — and a third
value would mean touching all of them just to keep seeds visible. A counter
touches none.

Two properties that make the counter safe: it is **clamped** to the real
`COUNT(source='llm')`, so a hand-emptied `chunk_annotations` cannot leave it
vouching for seeds that no longer exist; and `reset_db.py` already clears
`ingest_state` wholesale, so a reset zeroes it along with the annotations it
described.

The consequence, stated because `config.py` and `.env.example` now have to say
it: a corpus already annotated by another tier **will** pay for
`ANNOTATE_HYBRID_SEED_TARGET` sampler seeds, spread over several runs at 25% of
each budget. That is the intent — the target is about the *shape* of the pool,
not its size — and it is bounded: once the counter reaches the target the
sampler is done for good. Measured before/after on the same DB:
`labeling 0 more` → `labeling 800 more`.

### The event prefilter: 1.72× more events per LLM call

Events are the one label kNN structurally cannot produce (`source='distilled'`
⇒ `events_json='[]'`), so phase 3a exists to buy them — but the budget covers
only a few hundred chunks of 25k, and asking the LLM which chunks to look at
would spend the exact resource being rationed. `EVENT_CANDIDATE_PATTERNS`
(`function_words.py`, ~38 lowercase, wildcard-free substrings) is the
deterministic shortlist; `annotate_events` matches them in one OR'd LIKE pass
(0.71 s scoped / 0.76 s corpus-wide; 38 separate per-pattern scans measured
4.9 s).

Scored against the 450 chunks this corpus had already LLM-annotated, whose
event base rate is **48.4%** (the annotator is liberal): **83.3% precision,
32.1% recall** — i.e. 1.72× as many events per LLM call as spending the budget
at random. Recall is capped by the annotator counting bare affection ("Love
you", "Nice") as an event; no lexicon reaches those.

Correcting the plan's estimate: it guessed a **5–10%** chunk-level yield, and
the real figure is **~20%** (20.5% inside the top-20 dyads, 19.0% corpus-wide).
The unit is the chunk, not the message — a chunk spans ~14 messages, so a
pattern firing on 1.3% of messages lights up ~14% of chunks. `"thank"` alone
accounts for most of that; it is kept broad because measured alone it is the
single most precise pattern in the set (86.0%), and narrowing it to
`"thank you"`/`"thanks for"`/… drops recall 32.1% → 25.2% for no precision gain
(83.3% → 82.1%). Dropping it entirely does land the yield at the plan's 7.7% —
at 14.7% recall. The pool size is not the run's bound; phase 3a's time cap is,
so the filter's job is to raise event density, not to shrink the pool.

### Self-heal: `annotate._ensure_column`

`knn_confidence` postdates DBs in the wild, and `annotate.py` is reachable
without the `ingest.py` run that normally applies migrations — so both kNN
tiers crashed on `no such column` against an existing DB during integration.
The `--terms`-only `_ensure_curate_failures` helper generalized into
`_ensure_column(con, table, column)`, which runs the canonical (idempotent,
`PRAGMA table_info`-gated) `schema.create_schema` migration rather than keeping
a second, drifting copy of the DDL. `hybrid_chunks` and `distill_chunks` both
call it before their first write. `tests/test_annotate_hybrid.py`
regression-tests both tiers against a DB with the column dropped.

**Correction (PR #42 review): it must not run under `--dry-run`.** The first
cut called it before the dry-run branches, reasoning that a preview which
crashes on a missing column reports nothing and that the migration touches
schema, never data. The second half is false: `create_schema` also rescans
`reactions` for the old part-guid prefix and rewrites the rows it finds
(`schema.py`), so a preview became the one write-free path that wrote —
reproduced on a copy of the real DB, where `--quality hybrid --dry-run`
committed the new column while printing "no LLM calls, no writes". It now takes
a `dry_run` flag and *returns False* instead of migrating, and the dry-run
readers degrade: the backfill counts every distilled row as pending (it is),
and the phase-3b preview orders by `chunk_id` instead of the column that does
not exist yet.

`--terms` takes the same flag — the reasoning is not tier-specific, and leaving
one caller migrating would have kept exactly one preview that writes. Its
degraded reads say what a pre-column DB actually knows: nothing is parked,
nothing can be unparked, and every unverdicted term is pending, so the scope the
preview prints is the whole uncurated set. Only `--dry-run` ever sees
`has_failures=False`; the real path migrates first, so every read past the
dry-run return can still rely on the column. (This entry and the README's
`--terms` asterisk both said the opposite for a while — that a `--terms` dry run
still migrated. That was true when it was written; the flag reached this caller
a commit later and the prose did not follow. Corrected in the PR #42 round-3
pass, which is why the claim is recorded here rather than quietly deleted: a
doc that once described the code is the kind that gets trusted after it stops.)

### Two smaller review fixes worth recording

**Phase 3b no longer materializes a corpus it cannot dispatch.** With no
`--limit`, `_low_confidence_rows` fetched every in-scope distilled row *with
its full chunk text* — 14,995 rows / 9.5 MB / ~26 MB resident on the reference
corpus — and then the gate could decline all of it. The fetch is now skipped
outright when the deadline has already passed, and otherwise capped at
`_LOW_CONF_FETCH_CAP = 5000`: at the fastest rate measured for this phase even
an hour of budget buys ~1,600 rows, and they arrive lowest-confidence-first, so
the cap can only ever drop rows the run was never going to reach. The reported
candidate count comes from the textless `COUNT` so the cap never understates
the scope.

**One Ctrl-C, one exit code.** The tier had three: phase 1 raised
`SystemExit(1)`, phases 2 and 3 caught the interrupt and returned 0, and
`db.run_cli`'s conventional 130 was never reached from any of them. `_seed_chunks`
now takes `interrupt_exit=False` (plus a `stats` dict, since there is no return
value on that path) so the budgeted caller re-raises into its own handler:
commit, print the per-phase summary, exit 0 — the same shape as
`annotate_chunks` and as phase 3. The distilled tier keeps `SystemExit(1)`
exactly as before.

**Phase 3a's ordering is a documented property, not an accident.**
`event_candidate_chunk_ids` returns chunk ids ascending and the deadline stops
the phase mid-list, so one run's event upgrades are a *prefix* — the oldest
conversations in scope. Reruns advance through the remainder (upgraded rows
leave the candidate pool), so this is coverage order, not a correctness bug,
but a user who runs `make annotate-hybrid` once gets events for their oldest
history and the docs had not said so. Left as-is rather than interleaved:
every "spread" rule needs a constant with no measured basis, and ascending
order is what makes a rerun's progress obvious. README and CLAUDE.md now state
it.

**Superseded 2026-07-28 — the order is now pattern-hit count descending, chunk
id ascending as the tiebreak.** The decision above rests on one objection: that
any alternative ordering needs a tuning constant with no measured basis. Hit
count is not such a constant. It is evidence the prefilter has *already
computed* — the same `LIKE` predicate that decides membership, counted instead
of existence-tested — so ranking by it introduces no new number to justify, and
the id tiebreak keeps a rerun's progress exactly as obvious as ascending order
made it. See "2026-07-28: annotation throughput, round 2" below.

## 2026-07-28: default `CHAT_MODEL` → `qwen3.5:4b`, and effort levels get translated

Two changes that only make sense together: the default chat model dropped from
a 26B MoE to a 4B, and `annotate.py` learned that a preset's reasoning-*effort*
level is not portable to the model it now defaults to.

### The default model: `gemma4:26b-a4b-it-qat` → `qwen3.5:4b`

Every timing default this repo ships was measured on `qwen3.5:4b` — the hybrid
tier's wall-clock budget and its per-request rates, the batch-8 preset, the
"~106s for 8 event-candidate chunks" figure the budget gate reasons from. The
default model was a different one, so the numbers in the comments described a
machine nobody was running. Naming the measured model as the default closes
that gap: the runtimes in `config.py`, `README.md` and `.env.example` now
describe the model you actually get.

The second reason is reach. ~4 GB resident against ~20 GB is the difference
between "the defaults work on this laptop" and "the defaults are aspirational."
A bigger `CHAT_MODEL` still routes more precisely across the ~32 retrieval
tools and remains a one-line `.env` override — the change is which end of that
tradeoff is the *default*, not which end is available.

**`EMBED_MODEL` deliberately did NOT change.** It stays `nomic-embed-text`.
Unlike the generation models it is not a preference: `chunk_vectors` is
dimensioned for it and `ingest_state` guards it, because query-side and
document-side vectors have to come from one model or retrieval silently
degrades. Switching it is `reset_db.py` + a full re-ingest, never a restart —
so a chat-model default change must not drag it along. `ANNOTATE_MODEL` /
`TOPIC_MODEL` / `WRAPPED_MODEL` are `os.getenv(...) or CHAT_MODEL` and inherit
the new default automatically, which is the whole point of that rule.

**`CHAT_THINK` also did not change, and that is a tradeoff, not an oversight.**
It still defaults to `None` — omit the flag, take the model's own default —
which on a thinking-capable model means thinking is ON for every `chat.py` tool
round-trip (85s vs 3.2s per call on `qwen3:4b`). Turning it off would be the
single largest speed win available, and it is deliberately not taken: a 4B
model choosing among ~32 tools and chaining two or three of them is the case
where a reasoning pass plausibly earns its cost, and unlike a one-word label
there is no measurement saying the output is identical either way. So it is
documented, with the numbers and the one-line `CHAT_THINK=false` remedy, in
`README.md` and both copies of `.env.example`, and left to the user.

### `config.has_effort_levels` + `annotate._resolve_quality(quality, model)`

The `--quality` presets were written against `gpt-oss:20b` (see the 2026-07-17
entry): `think="low"` there asks for a *cheap* reasoning pass, and that is the
entire reason `balanced` beats `max` on wall clock. To a model that merely
toggles thinking, the same string is only truthy — it does not ask for less
reasoning, it buys a **full** pass. This repo has measured that twice, and both
times it was expensive: 282.6s vs 26.2s per dyad term on gemma4 for an
identical verdict (the term-curation entry above), and a timed-out request on
qwen3.5:4b at batch 8. With gemma4 as the default this was a latent trap; with
a 4B toggle model as the default it would have made **`--quality fast` the
slowest tier in the ladder** — the exact inversion of what the tier names
promise.

So `config.has_effort_levels(name)` answers "does this model read
low/medium/high as an effort level", and `_resolve_quality` now takes the model
and translates the preset through it. Resolved `(think, batch_size)`:

| preset | on `qwen3.5:4b` (new default) | on `gpt-oss:20b` |
|---|---|---|
| `max` | `(None, 1)` | `(None, 1)` |
| `balanced` | `(False, 1)` | `('low', 1)` |
| `fast` | `(False, 8)` | `('low', 8)` |
| `distilled` | `(False, 1)` | `('low', 1)` |
| `hybrid` | `(False, 8)` | `(False, 8)` |

Only the *preset* is translated, and only when its value is an effort STRING.
`None` (omit the flag) and an explicit `True`/`False` pass through untouched,
so the tri-state survives to the request body — which matters, because `None`
and `False` are not the same request and `max` depends on the difference. An
explicit `ANNOTATE_THINK` still wins over everything: a user who names a think
has said something the translation has no business second-guessing.

**This is a real behavior change for existing gemma4 users, not only for the
new default.** On gemma4 — and on any other toggle model — `--quality
distilled`, `fast` and `balanced` previously sent `think="low"` and therefore
ran a full reasoning pass on every request. They now send `think=false` and run
none. Those tiers get substantially faster, and they lose whatever that pass
was buying; `max` is unaffected. Anyone who wants the old behavior back sets
`ANNOTATE_THINK=true` (or `low`) explicitly.

### Why the list names the effort-level family, not the toggle ones

`_EFFORT_LEVEL_MODELS = ("gpt-oss",)` is a substring match over model families
that understand effort levels. The inverse list was the obvious alternative and
is the wrong shape: the effort-level set is small, named, and changes rarely,
while "every model that merely toggles thinking" is unbounded and grows with
every release — it would be stale the week after it was written, and staleness
there fails *closed on the wrong side*. Under the list as written, an unknown
model is assumed to toggle, so an effort string resolves to `False`: the
failure mode is a cheaper answer from a model that could have reasoned. Under
the inverse, an unknown model would be assumed to have effort levels, and the
failure mode is the 282.6s-per-item hang this change exists to prevent.

## 2026-07-28: `make conflict` on a wall-clock budget (66 h → 25 min)

`make conflict` was documented as a pipeline stage and measured as a **66-hour**
one. None of it was the detector: the deterministic pass over all 787 direct
chats costs **24.1 s** and writes **1,305** windows. All of it was the per-turn
escalation labeling, and it was slow twice over — each call cost far too much,
and there were far too many calls.

### The borrowed `think`, for the third time: 197 s per call

`CONFLICT_THINK` is a plain `*_THINK` knob, so it inherits `CHAT_THINK`, which
defaults to `None` — omit the flag, take the model's own default. On
qwen3.5:4b, a thinking model, that buys a full reasoning pass on every single
labeling call: **197 s per call against 2.2 s mean / 2.0 s median** (n=24, real
turns, median 30 chars) with `think=False`. To choose one enum for one text
message.

This repo has now paid for a borrowed `think` three times (282.6 s per dyad
term, `--quality fast` inverted into the slowest tier, and this), so the fix has
the shape the second one established: `conflict_detect._resolve_think(model)`
reads a genuinely UNSET `CONFLICT_THINK` as `False` *for this pipeline*
(`'low'` on the `config.has_effort_levels` family), while an explicit value —
`CONFLICT_THINK` or the `CHAT_THINK` it inherits — still passes through
untouched. The tri-state itself is not touched: `None` still means "omit the
flag" at the `ollama.py` seam, and `config._think`/`CHAT_THINK` keep their
meanings. What changed is one pipeline's default. The resolved value is threaded
through `detect_conflicts`, `label_top_conflicts` **and** `gate_classifier`,
because all three used to close over `CONFLICT_THINK` directly and a single
missed seam leaves that path 90x slow.

That is a ~90x speedup and it is not enough. At the old
`CONFLICT_MAX_LABEL_TURNS=12` this corpus wants **6,670** calls: **4.1 h**, on
the same "stage of `make pipeline`" that is supposed to finish while you watch.

### The budget is denominated in TIME, not windows

A window count is not a runtime promise. Per-call latency moves with turn
length, model and machine, and the windows themselves are wildly uneven —
47,457 candidate turns across 1,305 windows, **min 1 / median 4 / mean 36.4 /
max 1,940**. "Label the top 90" therefore says nothing about wall clock, which
is exactly the failure `--label-top` shipped with. So `conflict_detect.py`
borrows `--quality hybrid`'s contract instead: `CONFLICT_BUDGET_MINUTES`
(default **25**, `--budget-minutes N` overriding it, `0` = no deadline), clocked
from the top of `main()` so the detection pass spends it too. A bare run is
two-phase — detect every chat, then label the **strongest** windows first
(`_conflict_strength`: signal count → corroboration → dislike volume → peak
reply gap) until the deadline. It stops dispatching rather than overrunning,
commits per window, and prints how many windows were labeled versus left. It
composes with `--label-top N` (both caps apply) and is an error with `--no-llm`,
which has no LLM work to bound.

Two things deliberately not done, both measured first: request **batching**
(12-turn batched requests measured **2.25 s/turn — no speedup at all**, being
output-token-bound, with 4 of 8 replies unparseable) and **parallel workers**
(this Ollama serializes). A sequential loop that re-checks the clock before
every single call is not a compromise here; it is the whole reason this gate is
a few lines against `annotate._DispatchDeadline`'s pool arithmetic.

### The cold load is reserved once and never banked

The model's first request of a run pays its load — **~54 s** measured for
qwen3.5:4b. `_TurnRate.next_call()` adds `_COLD_LOAD_SECONDS` (60.0) to the
estimate until the run has completed one call, so a budget too small to cover a
window *declines* it rather than overrunning by a minute. But the first
completed call is charged and never *counted*: `record()` flips `warm` and
returns without touching the mean. The distinction matters because the mean is
banked for the next run, and folding a one-off startup cost into it would tell
every future run that every call costs a cold load.

### `conflict_turn_seconds:<model>`, and why a failed call is not recorded

Per-call seconds bank in `ingest_state` under `conflict_turn_seconds:<model>`
(the `annotate_group_seconds:...` precedent), so a run is never planning blind:
the banked mean primes the estimator, the live mean takes over once this run has
measured anything, and both are multiplied by `_TURN_RATE_SAFETY` (1.5) because
turn latency scales with turn length and the measured mean comes from turns far
shorter than the longest ones a window holds. Only a *measured* value is written
back — banking the prior would freeze a guess where nothing could correct it.

What crosses runs here is the **mean**, not annotate's max. Annotate needs the
worst case because its pool commits a blind wave before any timing exists; a
sequential loop that gates before every request has no such wave to protect
against.

A call that **raises** is deliberately never recorded, and that exclusion is
load-bearing rather than tidy. Its duration is not evidence about what a working
call costs — an Ollama timeout contributes `CONFLICT_TIMEOUT`, 300 s by
default — and because the number is banked for the *next* run, it latches: a
prior that large makes the gate decline every window, a run that dispatches
nothing measures nothing, and a run that measures nothing never overwrites the
value that stopped it. A malformed *reply* is a different thing and still
counts: `_classify_turn` returns an abstain for it, so the request really did
cost what it cost.

### `labeled_at` is stamped even when no label survived

`conflicts.labeled_at` (nullable, idempotent migration in `schema.create_schema`)
is the resume marker, and the marker is the **stamp**, never the presence of
`conflict_signals` rows. A window's labels pass the F4 eval gate (only cleared
classes are written) and `CONFLICT_LABEL_CONF_FLOOR`, so a window that was
processed exactly as intended can legitimately end with zero rows. Keying resume
on rows would make every one of those windows a candidate again on every rerun,
forever — real requests re-spent on work already done, ahead of windows nobody
has looked at yet. So "processed" is what is recorded. The converse case
is handled too: a window labeled before the column existed carries signals and
no stamp, so the labeling loop deletes its signal rows before re-labeling rather
than stacking a second set on the first.

**One carve-out, added 2026-07-30, on the READ path only.** On a DB whose
`conflicts` table predates the column there is no stamp to read, and the old
fallback was a constant `'pre-budget'` literal that marked *every* window
labeled — including windows from an old `--no-llm` run nothing ever looked at.
The read tools now infer `labeled` there from whether the window has stored
`conflict_signals` rows, with a note saying the DB predates labeling stamps and
a rerun will stamp properly. The inference is one-way-safe and deliberately
conservative: signals cannot exist without a labeling pass, but their absence
stays ambiguous, so a pre-migration window that *was* labeled and legitimately
produced zero rows now reads "not analyzed" rather than "analyzed and clean".
That is the safe direction. Wherever the column exists, the invariant above is
unchanged: the marker is the stamp, never the rows.

### Reconcile the window, don't rewrite it

`detect_chat` used to delete a chat's conflicts and write them fresh. Under a
budget that is fatal, and not subtly: **every** budgeted run re-detects all 787
chats before it labels anything, so a delete-then-rewrite would discard exactly
the labels the previous run's 25 minutes bought, and no rerun could ever extend
coverage past one budget's worth. (The plan document said the rewrite would
"naturally clear" `labeled_at` and treated that as a feature — see the status
header on `docs/PLAN_CONFLICT_HYBRID.md`.)

`detect_chat` now reconciles, keyed on `chat_id` + `first_date` + `last_date`: a
window whose boundaries are unchanged keeps its row id, its `conflict_signals`,
its topic and its `labeled_at` stamp while its deterministic columns are
refreshed in place; windows that no longer exist are dropped with their signals.
Two things that buys beyond resumability — stable `conflict_id`s across runs
(they are what a user types at `get_conflict` and what `conflict_signals` hangs
off), and an already-labeled window is not re-labeled by a detection pass.
Changing the window is what makes it a candidate again; emptying the table is
what forces a wholesale re-label.

### `CONFLICT_MAX_LABEL_TURNS` 12 → 6, because selection is peak-anchored

Halving the per-window ceiling is what lets a 25-minute budget reach further
down the ranking, and it is only safe because the six turns are now the *right*
six — the cap binds on the big windows, which is where the evidence hides. The
old selection took the first N turns chronologically; against a 1,940-turn
window that provably misses the evidence. Selection is now anchored on the
window's `peak_gap_message_id` — the message left hanging is the escalation
moment — taking a contiguous span around it, emitted in chronological order
(the labels read as a sequence), falling back — when a window has no peak id —
to a chronological slice ranked by one length preference (see the 2026-07-30
entry below). Ahead of the cap, ONE lexical prescreen drops turns with nothing
to read at all: **no alphanumeric character anywhere** (a tapback, an emoji,
bare punctuation). **Superseded 2026-07-30:** as first shipped that prescreen
also dropped turns under `CONFLICT_MIN_TURN_CHARS` (15) stripped chars; the
length half was removed because it deleted the stonewalling and short-contempt
class the classifier is explicitly asked to find — see "The length floor was
deleting stonewalling, so it stopped being a filter". About 70% of labels come
back `neutral`, each costing a full request to say nothing, and the median
window has only 4 candidate turns — so the per-window turn cap, not a length
floor, is what bounds the spend.

### Measured (2026-07-28, reference corpus, qwen3.5:4b)

350,216 messages / 787 direct chats / 1,305 detected windows. A bare run at the
25-minute default finished in **24m53s** and labeled **211** windows;
`--budget-minutes 3` finished in **2m58s**; a rerun preserved every previously
labeled window and advanced down the strength ranking.

211 of 1,305 in one run is the honest shape of the promise, and it is worth
stating plainly: `make conflict` is now a stage you can run inside a demo, full
corpus coverage is a handful of reruns, and neither of those sentences was true
of a 66-hour run that had to be completed or abandoned whole.

**Re-measure after 2026-07-30.** Removing the length prescreen means a window
whose turns are all terse is now dispatched instead of stamped for free, so
211 of 1,305 is an upper bound on what the same budget buys today. The number
has not been re-taken (the fix landed against hermetic tests, no Ollama).

## 2026-07-28: annotation throughput, round 2

Four independent changes and one negative result, all against the same
constraint — the hybrid tier's budget is a fixed wall clock, so every second
saved is coverage bought. All
measurements below are 2026-07-28, `qwen3.5:4b` warm, one machine, batch 8
unless noted, on the reference corpus: **25,029 chunks**, 19,483 of them inside
the top-20 dyads, of which the event prefilter flags **2,567 candidates**
(~20%, averaging ~1,900 chars); the phase-3b low-confidence candidates are much
shorter, median ~360 chars.

### What a budget actually reaches, and why phase 3a's order is load-bearing

A 25-minute default budget reaches roughly **10–15% of those 2,567
candidates**. Phase 3a is therefore always a *prefix* of the candidate list,
and what the ordering decides is not "which chunks eventually" but "which
chunks ever" for anyone who runs the tier once.

`event_candidate_chunk_ids` now returns **pattern-hit count descending, chunk
id ascending as the tiebreak** (`GROUP BY m.chunk_id ORDER BY COUNT(*) DESC,
m.chunk_id`, where the old query was a plain `SELECT DISTINCT ... ORDER BY
chunk_id`). Hit count is the only evidence this prefilter has that a chunk is
event-bearing — the lexicon is per-message substrings — so counting the matches
it was already testing for existence spends the budget on the densest
candidates instead of the oldest history in scope. That reverses "Phase 3a's
ordering is a documented property, not an accident" above; the argument for the
reversal is recorded there.

Two properties survive the change and are worth stating because they are what
make it safe:

- **Resume is unchanged.** An upgraded chunk is written `source='llm'` and
  leaves the candidate set entirely, so reruns advance down the ranking exactly
  as they advanced down the ids.
- **The ranking survives the fetch.** `annotate_hybrid._chunk_rows` used to
  `ORDER BY id`, which would have re-sorted the ranking away at the last
  moment; it now returns rows in the *caller's* order. Its two id oddities are
  answered deliberately: an id with no `chunks` row is omitted rather than
  faked (a None-text row reaches the LLM as an empty prompt and burns a
  request), and a repeated id yields one row at its first position (the tier's
  contract is a wall clock, so paying twice for one chunk is never the right
  reading of a duplicate).

The candidate pool's composition is unchanged by the ranking — 63%
gratitude-only, 22% good-news-only, 15% both — but a density prefix
structurally favors the both-kinds chunks, since they hit more patterns.

### Free-JSON batching is bimodal, so the schema goes on the retry

| Workload | free-JSON | schema-constrained |
|---|---|---|
| Longest event chunks (~2,150 avg chars) | 13.1 / 28.3 s/chunk | 5.4 / 5.9 s/chunk |
| Typical event chunks (~1,550 avg chars) | 4.6 s/chunk | 7.0 s/chunk |
| Short distilled chunks (~210 chars) | 3.5 s/chunk | 2.7 (batch 16), 3.2 (batch 24) |

(Two figures in a cell are two runs of the same workload.)

The shape of that table is the finding. Free-JSON is **fastest of anything
available when the batched reply covers every id it was asked for**, and
13–28 s/chunk when it does not — because each dropped id used to buy its own
single-chunk request, each with `chat_once_json`'s own nudge retry. Dropping
ids correlates with long, event-dense groups, which is exactly the workload
phase 3a is made of. Schema-constrained decoding has the opposite profile: its
coverage is guaranteed by construction, but the grammar costs a constant — 7.0
against free-JSON's 4.6 s/chunk on typical chunks — so it is **not** universally
faster and does not belong on the fast path.

Hence a three-rung ladder in `annotate._annotate_group`, ordered by what each
rung costs when it works: batched free-JSON → (only if that reply missed ids)
**one** schema-constrained request naming exactly the missed ids → per-chunk
singles for anything still missing. A fully covered batch issues exactly one
request and nothing else; the worst case becomes roughly one extra ~40–55 s
request instead of up to `batch_size` sequential singles.

Three details that are decisions rather than mechanics:

- **The schema is built from the validator's own constants.**
  `_annotation_schema(chunk_ids)` composes `EMOTION_LABELS`, `EVENT_KINDS` and
  `RESPONSE_CLASSES` into one `properties`/`required` pair per id, so a grammar
  and `_validate_annotation` cannot drift into disagreeing about what a valid
  label is. It stays an optimization for *coverage*: `_validate_annotation`
  still clamps everything, and Ollama's grammar converter enforces an integer
  range but not a float's, so the numeric bounds are as much documentation as
  constraint.
- **`response_class` is nullable via `anyOf`, not a union type or a
  mixed-type enum.** All three express "one of the four classes, or null"; only
  `anyOf` is old and portable in the JSON-schema-to-grammar converter Ollama's
  structured outputs run on, where union types and nulls inside enums are
  version-dependent.
- **Failure is total and silent.** A transport error, a timeout, a server that
  rejects the schema, an unparseable or non-object reply all degrade to
  "covered nothing" and fall through to the singles. The reason is *where* a
  raise would land, not that it would kill the worker (it would not): `_pooled`
  hands a worker exception to the future and it surfaces on the **caller's**
  thread at `done.result()`, aborting the whole phase, group and all, with
  `_run_upgrades`' `finally` able to commit only what already landed. An Ollama
  build that will not take a dict `format` has to cost this group its retry,
  never the run. `_annotate_batch` above it is unguarded by contrast, and
  correctly: `chat_once_json` returns `None` on a malformed reply rather than
  raising, so its failures are values. `KeyboardInterrupt` is not an
  `Exception` and still propagates, which is what keeps the tier Ctrl-C safe.
- **It resolves `think` through `ollama.effective_think`.** This is the one
  annotation request that reaches `chat_once` directly — `chat_once_json`
  hardcodes `format="json"` and cannot carry a dict schema — so it is also the
  one that can miss the process-wide `think` degradation that seam owns. On a
  model that answers `think=false` with empty content (gpt-oss) or rejects the
  option, re-sending the rejected value would fail **every** schema retry for
  the rest of the run, silently, turning a rung that removes a request per miss
  into one that adds one. `effective_think` is the degradation's single home;
  the retry is always downstream of a batched `chat_once_json`, so by the time
  it runs the flag is already resolved.

**Apple FM never reaches the middle rung.** Its `format` seam is not a guided
schema (an empty-`properties` `DynamicGenerationSchema` pins the reply to
`{}`), so handing `applefm` a dict format is undefined; apple keeps batch →
singles, and the ladder branches on `config.is_apple_model` rather than hoping.

### The Ollama batch is packed by chars now, too

`_char_budget_groups` grew a `max_rows` argument and `_annotation_groups` now
packs **both** backends with it: apple by chars alone
(`_APPLE_ANNOTATE_CHAR_BUDGET`, 2,000, unchanged and no row cap), every other
backend by `_OLLAMA_ANNOTATE_CHAR_BUDGET = 10,000` **and** `batch_size`.

Nothing on Ollama has to fit a fixed window — `batch_size` already bounds the
request — so the reason is the table above: a fixed row count lets a group's
cost swing by roughly 3–6x with chunk length, and the long end of that is where
the batch drops ids. The swing also poisons the budget, not just the clock:
`_save_group_rate` banks the **worst** group of a run, so one 28 s/chunk group
inflates `_DispatchDeadline.reserve()` for the whole *next* run and the gate
declines dispatches it could have afforded. Capping the worst case is worth
budget twice. 10,000 chars is ~4–6 of the long chunks while short chunks still
fill a whole batch of 8, so the fast case is untouched — and `batch_size` in
the banked-rate key is now explicitly a *maximum*, identifying the packing
config rather than any group's size.

### Negative result: `OLLAMA_NUM_PARALLEL` does not buy ~2x

`annotate.py`'s docstring claimed concurrent workers were worth "~2x with a
parallel Ollama". Measured against a second server with `OLLAMA_NUM_PARALLEL=4`:
**3.9 s/chunk against 4.8 s/chunk sequential, ~1.24x.** This hardware still
effectively serializes generation, so the concurrency mostly overlaps request
overhead. `OLLAMA_NUM_PARALLEL` is deliberately left alone and the docstring
now records the measured number, so the hope is not re-litigated; the levers
that did pay are the packing and the retry above.

### `ANNOTATE_KNN_INDEX_CAP`: cap the index, never the pool

Both kNN passes (`_classify_chunks`, `backfill_confidence`) cost O(rows ×
seeds) — one vec0 `MATCH` per row against every indexed seed. Measured on the
same corpus and machine: **14.6 s at a 450-seed index vs 82.0 s at 4,416**, a
6.7x per-row slowdown for a 9.8x wider pool. That is not a one-off to wait out:
the hybrid tier writes `source='llm'` rows on every run, so the pool widens
forever while the budget it is spent from stays fixed.

`ANNOTATE_KNN_INDEX_CAP` (default 800; `0` — or any value below it — =
uncapped) bounds the **index**,
never the pool. Every LLM annotation is still written, still counted toward the
seed targets, still read by every other consumer; a capped run simply votes off
a deterministic sample of them. 800 pins the pass near its 450-seed cost
permanently.

Why any `<= 0` is uncapped here while `ANNOTATE_BUDGET_MINUTES` and `--limit`
*refuse* a negative: there, a sign error changes what the run **does** — an
unbounded `--limit` run, or a deadline already in the past that declines every
LLM phase. Here it changes only how long the deterministic, LLM-free kNN passes
take — both of them, since classify and the confidence backfill each rebuild the
index; the labels are the ones the tier gave before the cap existed. So the sign lands
on documented behavior and `_sample_seed_ids` stays a pure function with no
refusal path. `<= 0` and not `== 0` because the cap is raised to
`_KNN_MIN_SEEDS` before it is applied: under a `== 0` guard, `-1` would have
meant "an index of exactly 100", which is neither uncapped nor asked for.

**The determinism contract is the load-bearing part**, because both kNN passes
rebuild the index inside one run and must vote off the same one.
`_sample_seed_ids` is pure, and buys that two ways: the ids
are **sorted** before the draw, so the caller's row order cannot move it, and
the draw comes from a **fresh `random.Random(0)`**, not the module-global RNG —
`_seed_chunks` samples from that one, so consuming or reseeding it here would
change the seed draw as a side effect and make the index depend on how much
randomness the process had already spent. `_build_seed_index`'s own `SELECT`
gained an `ORDER BY a.chunk_id` for the same reason (an unordered `SELECT` has
no stable row order, and the vec0 insert order should be reproducible too), and
the labels dict is filtered **with** the index from the same list — an id in
one and not the other is a `KeyError` or a silently dropped vote.

Two boundaries: the `_KNN_MIN_SEEDS` (100) floor check still runs **before** the
cap and against the **uncapped** pool, so a speed knob can neither invent the
"not enough seeds" error on a healthy pool nor hide it on a short one; and a cap
below that floor is silently raised to it, so every vote is drawn from ≥ 100
seeds whatever the knob says.

**On accuracy — and a claim that did not survive measurement.** The cap does
not change what a vote is, only which already-stratified seeds are in the urn.
The plan justified shipping it without a held-out eval by citing "seed-pool
composition effects of ~1% at k=10". **That number is wrong and is not why
this shipped.** Measured post-implementation (2,000 held-out chunks
re-classified at a 4,416 pool, capped vs uncapped): valence r=0.90 (mean
|Δ| 0.076), depth agreement 85.1%, top-1 emotion agreement **59.1%**, full
emotion-set agreement 19.2%. Nothing like 1%.

What rescues the decision is the **control**: re-running with a *different*
800-seed sample disagrees with the first 800-sample by the same margins
(r=0.90, top-1 58.0%, emotions 17.4%). So the churn is the kNN vote's intrinsic
sensitivity to seed composition — it was always there, at every pool size — and
capping perturbs labels **no more than resampling the pool does**. The cap is
safe in the only sense that matters here: it does not degrade the vote relative
to the alternative sample it replaces.

The standing mitigation is unchanged and load-bearing: emotions are the label
the distilled tier already documents as its roughest, and every high-stakes
read (conflict corroboration, the nudge valence exclusion, voiceprint tone) is
required to consume `source='llm'` rows only, never distilled ones. Stratifying
the index sample is deliberately not attempted — the same control says it would
be chasing noise.

### Post-implementation validation, and two of this round's own premises

Measured after the change landed, same corpus and machine, against copies of
the reference DB. Three of the plan's four hard criteria were checked; the
25-min acceptance pass remains the maintainer's.

| Criterion | Result |
|---|---|
| Phase 2 ≤ 25 s at a 4,416-seed pool | **met, measured** — 18.0 s capped vs 96.2 s uncapped (5.3×) |
| Worst Phase-3 group ≤ ~70 s single-stream | **met, measured** — 14.1 s (3a) / 16.6 s (3b) |
| Apple FM no regression | **met, measured** — dry-run + one real batch clean; schema-retry calls on apple = 0 |
| ≥ 400 Phase-3 upgrades in 25 min | **extrapolated, not measured** — 284 upgrades in a 7.7-min run projects to ~424–563 |

The §3a mechanism was confirmed directly: across the measured runs the
free-JSON batch dropped **19 ids in 4 groups, and the schema retry recovered
19/19 in 4 requests — zero fell through to per-chunk singles.** §3b was
confirmed by prefix quality rather than by eyeballing ids: the new top-40
carries **4.9× the pattern hits** of the old chunk-id prefix (3.7× at top-100),
and the two orderings' top-100 overlap by **one chunk**.

**Premise that did not hold: the 226 s worst group.** This plan's §1 framed the
worst-group win as 226 s → ~70 s. Re-running the *pre-change* code on the same
chunks, the worst group measured **37.8 s**, not 226 s — the 226 s figure came
from a `_save_group_rate` bank written under conditions nobody reproduced. The
improvement that is actually attributable is **37.8 s → 14.1 s** worst group,
and 3.00 → 2.34 s/chunk on an identical 24-chunk set (1.28×). Real, and in the
right direction, but a third of what the headline implied. Recorded because the
next round should not plan against 226 s either.

**Regression this round introduced, and fixed on review: the cold prior was
~3.6× too high.** `_COLD_GROUP_SECONDS_PER_CHUNK` (5.3 × batch × workers =
169.6 s) was calibrated when a group was 8 rows and a dropped id cost 8
sequential singles. §3d's packing and §3a's retry both cut the real cost: the
banked event-phase worst group came out at **46.7 s**. Since
`_DispatchDeadline.reserve()` plans off the prior until a phase's first group
completes — and `_pooled` submits `2 × workers` groups before the first result
arrives, so at `ANNOTATE_WORKERS=4` every dispatch decision of a phase can be
made before anything has been measured — phase 3a stopped dispatching with
**3.4 of its 4.6 minutes unused** on a cold DB. This round had made the ranking
better while leaving less time to spend on it.

The fix is a **denominator**, not a retune. The prior's arithmetic is
seconds-per-chunk × rows-per-group × workers, and `_load_group_rate` was
supplying `batch_size` for the middle term — but `batch_size` is a *maximum*:
`_annotation_groups` packs by char budget too, so a real event group is ~5 rows
and an Apple FM group is 1–2. The dispatch loops already hold the packing they
built (`_run_upgrades`, `_seed_chunks`), so they now call
`_GroupRate.denominate` with its mean rows-per-group. Free, no new constant,
and it corrects the same over-estimate on both backends — the Apple FM one was
documented and outstanding before this round.

Three properties make that the safe version of this fix rather than an
optimistic one:

- **It only ever touches a cold prior.** `denominate` is a no-op on a value an
  earlier run banked, and the live mean beats the prior regardless, so nothing
  measured is overwritten by arithmetic.
- **The margin stays one-sided.** At ~5 rows the prior lands at ~106 s against
  the measured 46.7 s worst group — still ~2.3× conservative, where it had been
  ~3.6×. Under-buying self-heals after one run; overrunning is the one promise
  the tier makes.
- **5.3 itself is unchanged.** Lowering the *constant* would want its own
  measurement across phases and both backends. Nothing here is a new estimate;
  it is the existing estimate applied to the group that actually gets sent.

The review's alternative — making `_DispatchDeadline.dispatchable` `continue`
past a decline instead of `return`ing, so a phase that has since measured a
real rate resumes dispatching — was **not** taken. That `return` is what
guarantees a phase stops at all, and un-latching it puts new dispatches
arbitrarily close to the deadline on the strength of an estimate that has
already proved too low once. Fixing the denominator gets the same 3.4 minutes
back without spending the guarantee.

**Also observed, pre-existing:** on any budget under ~11.3 min the seed phase
dispatches nothing at all (its share of the budget is smaller than the cold
prior's reserve), which is why phase 1 is absent from the table above.

## 2026-07-30: PR #42 review fixes

Seven fixes off the PR #42 review. Five of them reverse or narrow a decision
recorded above, so each is entered here rather than left in a code comment.

### The length floor was deleting stonewalling, so it stopped being a filter

`_TURN_SYSTEM` defines stonewalling as "withdrawing, shutting down,
terse/absent non-answers", and contempt covers "k.", "fine.", "whatever" —
every one of them under a 15-character floor. The prescreen therefore deleted
the exact class the classifier is asked to find, and the failure compounded: a
window whose escalation was *entirely* terse withdrawal selected zero turns,
was stamped `labeled_at` as a free window anyway, and the read tools then
presented it as analyzed. The stamp is a one-way latch, so that false negative
survived every rerun until an explicit `--relabel`.

Fixed as the review's option (a). `_is_labelable` drops to the alphanumeric
test alone — a tapback, an emoji or bare punctuation still gives the model
nothing to read. `CONFLICT_MIN_TURN_CHARS` survives as a **ranking preference**
in `_select_turns`' no-anchor fallback: floor-clearing turns take slots first,
shorter ones fill the rest, and the chosen set is re-sorted chronologically
before it is returned, so the sequence contract holds on every path. It demotes;
it never deletes.

Kept as a *threshold partition* rather than a pure length sort on purpose. A
length-descending sort would already imply the floor, which would leave the knob
meaning nothing; the partition keeps it a knob, and breaking length ties by
position means a window of uniformly long turns still yields its head — exactly
the behavior this branch had when the floor was a filter.

Cost, unmeasured and worth expecting: windows that were stamped for free now
dispatch up to `CONFLICT_MAX_LABEL_TURNS` requests each, so windows-labeled-per-
run under a fixed budget may fall. That is the honest direction — the free ones
were never analyzed — and the wall clock absorbs it by construction, since a
rerun continues down the ranking.

The residual the fix does not remove — a window whose turns really are all
tapbacks still yields zero turns and still gets stamped — is answered on the
**reporting** side, not with a schema change or a new column.
`query_tools/conflict.py` gained `_LABELED_EMPTY_NOTE` (one episode) and
`_LABELED_EMPTY_SCOPE_NOTE` (a `find_conflicts` page), both saying the pass
reads message *text* only, so a stretch carried by tapbacks, by tone, or moved
off-channel can be looked at and still yield nothing. Zero turns is never a
positive finding of calm.

### `--relabel` guards before it clears, and is not the model-switch door

`--relabel --model <other>` deleted every `conflict_signals` row in the table
and *then* raised the `conflict_model` guard, because the guard only fired later
inside `label_top_conflicts` — every label previous budgets had bought
destroyed, nothing rebuilt. `_check_model_guard` now runs immediately before
`_clear_labels`, so the mismatch is refused with the labels still in place.

The alternative — clearing `conflict_signals` and the `conflict_model` state key
together, making `--relabel` the supported way to switch models — was
deliberately **not** taken. `--relabel` is the "what labeling PRODUCES changed"
door (a retuned `CONFLICT_MAX_LABEL_TURNS`/`CONFLICT_MIN_TURN_CHARS`, a newly
cleared eval gate); the guard's own message already names clearing `conflicts`
or `reset_db.py` as the model-switch door, and having two doors for it is how a
destructive one gets taken by accident. Under `--dry-run` neither the guard nor
the clear runs.

(Superseded in part by the adversarial round below: the `--no-llm` carve-out
this fix shipped was wrong — that path skips the labeling, not the deletion —
and the guard now runs before the detection sweep too, in a refuse-only mode
that leaves `conflict_model` for the pass that actually labels.)

### Negative budgets are refused, not clamped

`conflict_detect._resolve_budget_minutes` clamped with `max(0, ...)`, and 0
means UNBUDGETED there — so a typo'd `CONFLICT_BUDGET_MINUTES=-25` silently
bought the unbounded run the CLI already refuses `--budget-minutes -5` for.
`annotate_hybrid._resolve_budget_minutes` had the mirror-image hole: it passed
a negative straight through into an already-passed deadline, which declines
every phase and is indistinguishable from a machine too slow for one request.

Both now raise `SystemExit` naming the knob the value came from, and both
validate the caller argument as well as the env — a caller-supplied `-5` means
the same thing even though the parser already refuses it at the CLI. `0` keeps
its per-pipeline meaning: "unbudgeted" for conflict (the budget is additive
there, so switching it off must restore the unbounded path), "unset" for hybrid
(the budget *is* the tier, so 0 falls through to the per-backend default). A
clamp would have been wrong in hybrid too, for the opposite reason: 0 means
unset, so a typo'd `-25` would have quietly bought the 25-minute default.

### The pre-migration `labeled` value is derived, not asserted

`query_tools.conflict._labeled_at_select` fell back to a constant `'pre-budget'`
literal on a `conflicts` table with no `labeled_at` column, marking *every*
window labeled — including windows an old `--no-llm` run wrote that nothing ever
looked at. `get_conflict` then reported "we looked and found nothing" about a
window nothing looked at, the exact misreading the labeled/unlabeled split
exists to prevent.

The fallback is now `CASE WHEN EXISTS (SELECT 1 FROM conflict_signals ...)`,
still a constant fragment with no interpolation of user input. One alias detail
is load-bearing: an unqualified `id` inside that subquery binds to
`conflict_signals.id`, so the helper takes a **required** table alias and
`get_conflict` was re-written to `FROM conflicts c WHERE c.id = ?` with fully
qualified columns. `_PRE_MIGRATION_NOTE` states the remaining ambiguity out
loud rather than papering over it, and a rerun stamps properly. While in the
file, the three empty-scope early returns were given the `n_labeled` /
`n_corroborated` keys the docstring and the tool schema already promised.

### Hybrid: `--limit` counts what was written, not what was sent

Phase 3b derived its `remaining` from `len(seen)` — chunks *dispatched*,
including ones whose reply came back unparseable — so `--limit N` under-
delivered against its own docstring. The code was made to match the docstring
(the review's preference): `remaining = max(0, limit - events["done"])`, with
`seen` left purely as the dedup set. `--limit` is the **deliverable** cap; the
wall clock is the **spend** cap, and they are different promises.

One residual is documented in code rather than fixed: a phase-3b reply that
itself fails to parse still leaves the run one short, because making it up would
mean dispatching past N — spending budget the flag did not authorize. Exactness
there would need a success-target inside `_run_upgrades`, which cannot stop
mid-group cleanly under `_pooled`'s worker backlog.

### Hybrid: three cheaper guarantees

- **The `annotate_model` guard moved to the top of `hybrid_chunks`**, before the
  budget clock starts and before the `knn_confidence` migration. It used to fire
  only at phase 3a, and phase 1 only guards when it actually labels — so with
  the seed pool already at target, a switched `ANNOTATE_MODEL` paid for both
  corpus-wide kNN phases (up to half the budget) before being refused. The
  phase-3a call is kept as a local invariant for a future caller that reaches
  phase 3 directly. Skipped under `--dry-run`: `_check_model_guard` `set_state`s
  `annotate_model` and commits, and a preview that writes is not a preview.
- **Phase 3a's candidate fetch is capped at `_EVENT_FETCH_CAP = 2000`** before
  `_chunk_rows` materializes full chunk text — 3b had been capped for exactly
  this memory reason since it shipped, and the prefilter's ~19–20% hit rate
  means a corpus 10× the reference yields ~50k candidates, ~95 MB of text held
  to dispatch a few hundred rows. The arithmetic says the cap can only bind
  where the deadline already had: an event group is 4–6 chunks banking 46.7 s at
  4 workers ≈ 0.5 chunks/s, so an hour spent *entirely* in 3a is ~1,850 upgrades
  and no run spends an hour there. It is a prefix of a densest-first ranking, so
  it is lossless in practice, and it prints both the true candidate count and
  how many it left — no silent caps.
- **Phase 3a re-reads the clock** after the corpus-wide event prefilter and
  after the row fetch. `event_candidate_chunk_ids` is the run's one unavoidable
  full `messages` scan, and deriving `left` from a reading taken before it
  charged that scan to the phase's own 60% share.

### The seed pass keeps paid-for work

`annotate_distill._seed_chunks` committed and banked the `hybrid_seeds` counter
on the success and Ctrl-C paths but not on any other exception, where
`_run_upgrades` had had a `finally` for exactly this reason since it shipped. An
Ollama timeout propagating out of `_pooled` mid-pass therefore rolled back up to
`ANNOTATE_COMMIT_EVERY − 1` paid-for seed rows *and* skipped the counter update
for the rows that had already committed — and a counter behind the real row
count makes the next run over-buy seeds past `ANNOTATE_HYBRID_SEED_TARGET`, out
of the upgrade phases' slice.

Both now happen in a `finally` that runs exactly once on every exit path, so the
success path no longer records twice and the KeyboardInterrupt handler keeps its
message, its `interrupt_exit` split and its `stats['done']` unchanged. The clamp
to the real llm-row count is untouched — it lives at the top of the function,
and the `finally` reads it.

## 2026-07-30: PR #42 review fixes, adversarial-review round

Four reviewers were pointed at the fixes above and asked to refute them. What
survived verification, and what did not.

### The model guard moved again: before *detection*, and it records nothing

Fix 5 above closed the `--relabel` door only. Two holes were left open, both
demonstrated:

- **The default invocation.** `detect_chat` calls
  `_drop_conflicts(con, chat_id, keep=kept_ids)` and commits per chat, deleting
  a superseded window together with its `conflict_signals` rows — and the guard
  fired only afterwards, inside `label_top_conflicts`. A plain
  `conflict_detect.py --model <other>` therefore destroyed labels first. Worse:
  if the drop emptied `conflicts`, the guard's `COUNT(*)` read 0 and it
  *accepted* the new model, rewriting `conflict_model` over the wreckage. Any
  chat whose stored window shifted dates because new messages arrived hits this,
  which makes it the routine case, not a contrived one.
- **`--relabel --no-llm --model <other>`.** The `--no-llm` carve-out skipped the
  guard on a path that still runs `_clear_labels`, while `--relabel`'s help text
  — added by Fix 5 — promised the guard runs before the clear without
  qualification. `--no-llm` skips the labeling, not the deletion.

`_check_model_guard` grew a `record=False` mode and `main` calls it that way,
once, before either destructive step, for every run that labels or relabels.
Splitting refusal from recording also fixed the third finding of the round: the
Fix 5 call site did `set_state('conflict_model', model) + commit` before any
labeling, so a `--relabel` run interrupted during detection (which
`detect_conflicts` catches and returns from) left the key naming a model that
had never written a label — and refused the *next* run in its name, offering
"clear conflicts" as the recovery. Recording stays with the pass that labels;
the key keeps meaning "the model that labeled these rows".

`--no-llm` alone (no `--relabel`) is still unguarded: it neither labels nor
deletes globally, so the model flag really is inert there and refusing would
block a deterministic re-detect over a flag it does not use.

### Two more knobs validated where they are resolved

Fix 6's rule turned out to have two more instances in the same CLI:

- `CONFLICT_MAX_LABEL_TURNS`. `int(getenv(...) or "6")` reads `"0"` as 0, and
  `args.max_label_turns or CONFLICT_MAX_LABEL_TURNS` passed it through. The flag
  spelling has always been refused, with the reasoning spelled out in `main` —
  a non-positive cap selects no turns while every window is still stamped
  processed — but the env spelling reproduced exactly that at corpus scale, and
  the printed summary then blamed the prescreen ("had no labelable turn and cost
  no request"). Now a `SystemExit` naming the knob.
- `CONFLICT_BUDGET_MINUTES` under `--no-llm`. Fix 6 resolved (and so validated)
  the budget before the `--no-llm` distinction was made, so a negative env value
  refused a run the knob cannot affect — on the one path whose parser explicitly
  says the budget "has nothing to bound". The budget is no longer resolved at
  all under `--no-llm`.

### `labeled_signal_turns` means one thing in both read tools

`find_conflicts` counted non-abstain rows; `get_conflict` picked its note off
`COUNT(*)`. An abstain-only window was therefore listed as "labeled but produced
zero signal turns" and then, on the drill-in the envelope *tells* the caller to
make, answered with the eval-gated-bands caveat — the warning disappearing at
precisely the expansion step it had to survive. `get_conflict` now computes both
counts: `signals_total` still paginates every stored row, and a
`labeled_signal_turns` field carrying the same definition and the same name as
the listing decides the note.

Both "nothing to show" states also emit a code-printed `_epilogue` now
(`_LABELED_EMPTY_EPILOGUE`). The unlabeled sibling has had one since it landed,
with the file's own comment explaining why — it is the copy the user sees
regardless of model compliance — and the labeled-but-empty case licenses the
same false summary ("I checked those stretches and nothing came up"). In the
scope where every window is labeled and empty, `total == n_labeled`, so the
unlabeled epilogue never fires to cover it. A framing floor is code, never
prompt.

### Rejected: ranking the peak-anchored span by length

The reviewer's major finding was that `CONFLICT_MIN_TURN_CHARS` survives only in
the no-anchor fallback, which is the rare branch, so in the peak-anchored span
terse filler can crowd out a longer, more substantive turn — with a worked
example where a criticism two turns past the peak loses its slot to "k." and
"ok".

Not taken, and the reason is the point of the fix: inside a span centered on the
escalation moment, the terse turns ARE the stonewalling evidence — that example's
displacing run is `k / ok / lol / yeah / sure / fine`, a textbook stonewall.
Preferring length there would re-delete the class this change exists to catch,
one branch further in. The floor stays where a window has no escalation moment
to bracket and the slice would otherwise be an arbitrary head. The consequence
the reviewer identified is real, though, and is now named in `config.py`,
`.env.example` and CLAUDE.md rather than left to be rediscovered: a long turn
further from the peak can lose its slot to a terse neighbour.

### Hybrid: reporting that matches what was done

- **Drop accounting is measured against the true candidate count.** `dropped`
  was computed after `--limit`'s prefix, so it reported only the fetch cap's
  share of an already-shortened list: 50,000 candidates under `--limit 5000`
  with a 2,000 cap printed "3,000 left for a later run" where 48,000 were. The
  candidate count is now taken before either prefix, and the line names which
  prefix left what.
- **A dry run says "would fetch".** The fetch is short-circuited on that path
  (`event_rows = [] if dry_run else _chunk_rows(...)`), but the notice printed
  "fetching …" anyway — a preview describing an action it provably does not
  perform. The id preview also came off the capped list, so a cap under 10
  clipped the ten ids it advertises; it comes off the pre-cap list now.
- **A negative `--limit` is refused.** It inverted both prefixes it feeds:
  `event_ids[:-1]` drops the LEAST dense candidate instead of taking a
  densest-first prefix, and `max(0, limit - done)` then switched phase 3b off, so
  a stray minus bought a run nobody asked for. (The LLM tiers' `--limit` reaches
  SQLite as `LIMIT -1` = no limit; that was left as found here and **refused at
  the parser in round 3** — see below.)
- **The closing summary stops claiming it finished.** "finished every candidate
  in scope with budget to spare" printed even when `--limit` or a fetch cap had
  left thousands of candidates untouched with minutes unspent — which is exactly
  where `--limit`'s one-sided residual (a skipped dispatch that the other
  sub-phase has no candidates to make up) hides. It now reports how many were
  left and that the clock was not the reason.
- `_EVENT_FETCH_CAP`'s "no budget can dispatch that many" is a statement about
  the DEFAULT budget, not a law: `ANNOTATE_BUDGET_MINUTES` has no ceiling, and
  past ~111 budgeted minutes (2,000 / 0.5 chunks/s / 0.60) the cap rather than
  the clock is what stops 3a. Its comment says so now.

### Rejected: making `--limit N` exact

`--limit` remains "at most N". Delivering exactly N when a dispatch comes back
unparseable and the *other* sub-phase has no candidates to make it up from would
mean dispatching past N — spend the flag did not authorize — or a success target
inside `_run_upgrades`, which cannot stop mid-group cleanly under `_pooled`'s
worker backlog. The residual is documented in both directions instead, and the
summary line above is what makes it visible.

### A test that could not fail

`test_the_counter_is_not_applied_twice_on_the_success_path` asserted the
resulting counter value, and `_record_seeds` writes an ABSOLUTE value
(`n_seeded + n_done`, with `n_seeded` captured before the loop) — so it is
idempotent, and the test passed against a deliberate double application as
readily as against the fix. It counts `_record_seeds` invocations now. Worth
generalizing: for an idempotent write, only the call count can prove
"exactly once".

## 2026-07-30: PR #42 review fixes, round 3

Four findings the previous round's fixes did not cover. Two are code, two are
recorded where the thing they describe already lives: the
`ANNOTATE_KNN_INDEX_CAP` section above now says any value `<= 0` is uncapped
(and why that knob does *not* refuse a negative the way the budgets do), and the
`_ensure_column` section corrects the claim that a `--terms` dry run still
migrates — it stopped when the flag reached that caller, and the prose did not
follow.

### A negative row cap is refused once, at the parser

`--limit` reached three different consumers with three different meanings for a
negative: a SQL `LIMIT ?` on the per-chunk tiers and on `--terms`, where `-1`
means **no limit**, and a list prefix in hybrid's phase 3a, where
`event_ids[:-5]` is not a prefix of the density ranking at all — it drops the
five *least* dense candidates and dispatches everything else. `hybrid_chunks`
already refused its own; the other two silently bought an unbounded run out of
the one flag whose job is to bound one.

The check belongs at `annotate.main`'s parser, where every tier passes through,
rather than at each consumer. `hybrid_chunks` keeps its own `SystemExit` for
callers reaching it directly — the same defense-in-depth argument as the model
guard, which is now called both at the top of the run and at phase 3a.

`--top-dyads` is refused for the same reason (negative reaches the scope
subquery's `LIMIT` and widens the run to every chat, which is what `--all` is
for), and that pulled in two things the flag check alone would have left
half-done:

- **`ANNOTATE_TOP_DYADS` is validated too**, where it is *resolved* — the rule
  `CONFLICT_MAX_LABEL_TURNS` already follows. Refusing the flag while the env
  twin still reached the same `LIMIT` would have left the unbounded run
  available through the quieter of the two doors. All three resolution sites do
  it: `annotate.main`, `hybrid_chunks` (which resolves `config.ANNOTATE_TOP_DYADS`
  itself for direct callers, exactly as it keeps its own `--limit` refusal) and
  `annotate_chunks` (whose default argument freezes the value at import).
- **…but only where the run BINDS it**, which is the other half of that same
  precedent — `conflict_detect` deliberately does not resolve
  `CONFLICT_BUDGET_MINUTES` under `--no-llm` so a bad value cannot refuse a run
  it could not have affected. `--terms` has no scope, `--all` omits the scope
  clause, and `distilled` hardcodes `top_dyads=0` because it is corpus-wide by
  construction. The first cut refused all three, which took out bare
  `make annotate` over a knob those runs never read — and left the message
  prescribing `--all` as the remedy while refusing `--all` with the same message.
- **`--top-dyads 0` now means zero chats**, not the 20-chat default.
  `main` resolved it with `args.top_dyads or ANNOTATE_TOP_DYADS`, so an explicit
  `0` fell through — while `hybrid_chunks` has always read its own `0` as `0`
  (`if top_dyads is None`). The two halves of one flag disagreed, and `main` was
  the one that had to move: `--quality max --top-dyads 0` asked for nothing and
  would have bought a full default-scope LLM run.

`--limit 0` stays legal and unchanged — it is the empty *cap*. The refusal
message says only that, deliberately: what a tier does with an empty cap is the
tier's business (`distilled` ignores `--limit` outright, and hybrid's phase 1
can still spend LLM calls on seeds), so promising "nothing this run" would have
been a fourth doc claim that isn't true.

### The suite had a second Ollama seam, and nothing stood on it

`FakeLLM` stubbed `ollama.chat_once_json`, but the batch ladder's middle rung
(`_annotate_schema_retry`, added in round 2) dispatches through
`ollama.chat_once` — `chat_once_json` hardcodes `format='json'` and cannot carry
a dict schema. Any test with `batch_size > 1` and a dropped id would therefore
have made a **live HTTP request to localhost** from a suite whose header
promises it makes none. Nothing in *this* module had reached that rung, so
nothing had noticed the gap. (`tests/test_annotate.py` does cover the rung, at
the unit level: `_LadderLLM` stubs both seams and `TestAnnotateGroupLadder`
drives `_annotate_group` directly. That is where the ladder's own contract is
pinned; it could not have caught this, because it never runs through
`HybridFixture`.)

`FakeLLM.schema_retry` now stands in for it, `DBFixture` patches the seam by
default, and the default raises `ReachedTheNetwork` — deliberately **not** an
`Exception`, because `_annotate_schema_retry` catches `Exception` by design (its
contract is "degrade to the per-chunk rung, never propagate"), so an ordinary
error would have been swallowed and the test would have passed anyway. Three
tests now cover the rung *as the tier reaches it*, which is what the unit-level
ladder tests cannot say: a dropped id costs one schema retry naming exactly that
id and no single-chunk request, and all three chunks land `source='llm'`; a
retry that raises costs the group its retry and not the run; and an id no rung
can answer stays `distilled` rather than being written.

### A fixture pinned to a knob a developer's `.env` can move

`_SEEDS` was `max(_KNN_MIN_SEEDS, config.ANNOTATE_SEED_TARGET) + 10`, and the
kNN index-cap tests build at a cap of 200. `ANNOTATE_SEED_TARGET=150` in a
developer's `.env` left a 160-seed pool, *under* that cap: measured against
HEAD's file, two of those tests then failed outright (`160 != 200`, and the
"using 200 of 160 seeds" line the cap never printed) and a third —
`test_both_passes_of_a_run_vote_off_the_same_index` — silently weakened to
asserting determinism over an *uncapped* index. Failing on a colleague's `.env`
and quietly testing less on it are both defects; the cap is a named constant now
and the pool clears it by construction.

The env value stays in the `max()` on purpose: `distill_chunks` binds
`ANNOTATE_SEED_TARGET` as a **default argument**, frozen at import, so a test
reaching it through `annotate.main` cannot patch the target and the fixture must
simply clear whatever was baked in. The knobs that reach the module by other
routes are pinned at their call sites instead (`ANNOTATE_THINK`,
`ANNOTATE_BATCH_SIZE`, `ANNOTATE_BUDGET_MINUTES`, `ANNOTATE_COMMIT_EVERY`),
because "the suite is hermetic" has to be true of the whole module, not of the
one test a review happened to name.

## 2026-07-30: voiceprint's rebuild guard could not see an annotate run

`--quality hybrid` exists to write `source='llm'` annotations, and
`annotate_hybrid.py`'s module docstring sold exactly that: *"Upgraded rows are
written source='llm', so every valence-trusting reader (conflict corroboration,
the nudge Stage-0 exclusion, voiceprint tone) lights up."* Two of those three
were true. `voiceprint.build_voiceprints` skipped its rebuild whenever
`MAX(messages.id)` matched `ingest_state.voiceprint_max_message_id` — and an
annotate pass adds **no messages**. Reproduced on a copy of the real DB (350,549
messages): flip 500 rows `distilled → llm`, run `build_voiceprints`, get
`Voiceprints up to date (max message id 387248, 0 built)` and the old
`tone_valence` back.

The blast radius is narrower than the first draft of this entry claimed ("broken
on every single run"), and the correction is worth recording because it is the
difference between a permanent corruption and a resumability bug. `make
pipeline` runs annotate *before* voiceprint, so a run whose ingest added even
one message moved the message-id key and rebuilt anyway — verified on the real
corpus: the same 500-row flip left the tone stale when nothing was ingested and
refreshed it (0.2368 → −0.0531, matching `AVG(valence)` over the llm rows) when
one message was. What actually stayed stale is any run over an
otherwise-unchanged corpus — precisely the resumable `--quality hybrid` re-run
the tier is designed around, and a bare `make annotate && make voiceprint` —
until the next run that ingested something. No historical DB needs a `--force`
backfill on account of this; a live pipeline healed itself.

This is the hole `drift.py` closed for itself on 2026-07-21
(`drift_annotations_hash`, above). Voiceprint is now guarded on
`voiceprint_annotations_hash` as well.

**The fingerprint deliberately is NOT drift's.** Copying it would have been
wrong three ways, and each divergence is load-bearing:

- **No `ch.is_group = 0`.** drift is 1:1-only. `_enumerate_tasks` builds
  `scope='group'` tasks and a `scope='all'` task with no `chats` join at all, so
  group-chat annotations really do reach `_tone`; drift's filter would have
  reproduced the same miss for those scopes.
- **`source='llm'` only**, matching `_tone`'s own unconditional clause. A plain
  `--quality distilled` pass must *not* trigger a rebuild — voiceprint cannot
  observe those rows — where drift's `COUNT(*)` over every source would.
- **An exact per-row digest, not `COUNT`/`SUM`.** Not exactness bought at a
  price: measured, it is the **cheaper** of the two shapes on every corpus
  tried, because drift's plan scans the `chunks` table while this one scans the
  much smaller `chunk_annotations`. Each pair named with its corpus rather than
  averaged into one number — an earlier draft quoted "6.6 ms vs 23.9 ms on the
  same DB", and those two figures came from *different* DBs: on the
  25,029-chunk reference corpus (chunks 27 MB, chunk_annotations 1.7 MB) it is
  5–6 ms against drift's 7–8 ms, a **1.3×** gap; on a 74,233-chunk one (chunks
  83 MB, chunk_annotations 6.1 MB) 5–7 ms against drift's 21–24 ms, **4×**. The
  gap widens with the `chunks` table, which is drift's scan and not ours —
  that is the mechanism, and it is why one ratio was the wrong thing to record.
  Exactness is what buys the coverage: `emotions_json` is a TEXT list no `SUM`
  can fingerprint, so a lossy aggregate misses an emotions-only rewrite, and
  `SUM(valence)` misses a valence swap between two chunks. Both are regression
  tests, and both were verified to fail when the digest is swapped for drift's
  aggregate shape — a test that cannot fail would not have pinned either.

**Cost, measured rather than asserted** (25,029-chunk reference corpus): 5–6 ms
at 4,416 llm rows, 28–31 ms if every row were upgraded, against a **4.0 s** full
rebuild — 0.15%, ~0.7% at full hybrid coverage. The scan is over *all*
`chunk_annotations` rows (`schema.py` creates no index on that table), so the
curve grows with total annotations plus per-llm-row Python work, not with the
llm population alone — worth stating, because the fix's whole purpose is to grow
that population. `ORDER BY ca.chunk_id` is free: `chunk_id` is `INTEGER PRIMARY
KEY`, i.e. a rowid alias, so there is no temp b-tree — and it is what makes the
digest deterministic.

**Rejected: `group_concat`.** Deterministic ordering needs `ORDER BY` inside the
aggregate (SQLite ≥ 3.44, which nothing in the repo pins) and it materializes
the whole string in memory, to save 1–3 ms.

Two details that read like style and are not. The `JOIN chunks` is not there for
`chat_id` (immutable per `chunk_id` — nothing in the tree `UPDATE`s `chunks`,
and ids are `AUTOINCREMENT` so they are never reused): it **drops orphaned
annotations**, exactly as `_tone`'s own join does, and a tail re-chunk can
orphan rows with `MAX(messages.id)` unchanged. And the `tuple()` in
`repr(tuple(row))` is load-bearing: `build_voiceprints` takes a caller-supplied
connection, and under `db.connect`'s default `rows=True` a bare `repr(row)` is
`<sqlite3.Row object at 0x...>` — address-dependent, so the digest would differ
every process and force a rebuild on every run.

**What the pair still does not cover**, now said out loud in the module
docstring rather than left for the next reader to discover the way this one was:
a `SESSION_GAP_MINUTES` retune (re-splits every session, moving every
opener/closer), a `person_period_lexicon` rebuild (moves the 9 `fw_*_rate`
columns), a `VOICEPRINT_*` knob change (`BURST_GAP_SECONDS` moves every
`burst_size`; `MIN_RETURN_MESSAGES` moves which rows exist *and* the per-scope
`feature_std` population), and an AddressBook re-ingest that flips
`handle:N → contact:M` — which a rebuild self-heals but never triggers on its
own. All four are pre-existing and all four need `--force`. Closing one hole
while leaving four unnamed is how this bug survived; a half-guard that reads as
complete is worse than a documented partial one.

The skip line now names both inputs (`max message id N, annotations unchanged`).
Reporting only the input that had *not* moved is what made this invisible: the
line read as "nothing to do" immediately after the run that had changed the one
thing the guard was not checking.

A DB predating the key rebuilds exactly once (`None` equals no digest) and is
guarded on both thereafter — that rebuild *is* the repair, since such a DB is
already stale for every upgrade pass it has ever run. `reset_db` clears
`ingest_state` wholesale, so the new key needs no teardown of its own.
