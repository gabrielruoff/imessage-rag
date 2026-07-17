# textmessagething — design decisions & full reference

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

- `source_data/chat.db` — the real macOS Messages database (found already
  present in this project folder, ~500MB, ~382k messages across ~1.1k chats /
  ~1.9k handles). This is the primary/preferred source — richer and more
  reliable than parsed `.txt` exports.
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
`qwen3:30b-a3b` (what a local `.env` currently sets `CHAT_MODEL` to) *does* render a
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
`get_chat_summary_stats(group_by_chat=True)` dump, 555 rows / ~114KB ≈ 35k
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
  message-part prefix (`p:N/`, `bp:`) that `messages.guid` never has; only
  well under 1% of reactions joined to their target. Ingest now strips the
  prefix and `create_schema` runs an idempotent in-place `UPDATE` for
  existing DBs. This bug silently zeroed `get_message_reactions` lookups.
- **Result envelopes** — `search_messages_sql`/`search_messages_fts` return
  `{rows, total_matching, returned, order, truncated}` so a capped page can
  never be mistaken for the population (the eval's "longest silence = 39
  days" failure, computed from the newest 50 of 4,000 rows).
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
