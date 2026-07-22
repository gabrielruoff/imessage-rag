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
  with just the latest filter's rows.
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
`voiceprint_max_message_id`; read via `get_voiceprint` (which can also recompute
over a date window, capped). Later persona issues (draft-reply, style-match,
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
