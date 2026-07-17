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
- Ingest: `make ingest`, or `.venv/bin/python3 ingest.py [--no-embed]
  [--since-last] [--senders=...]`
- Chat CLI: `.venv/bin/python3 chat.py` (interactive REPL), or
  `chat.py -p "..."` for a one-shot answer (prints only the reply; exit 1 if
  Ollama is unreachable)
- Annotate chunks: `make annotate` (chunks + `--terms`) / `make annotate-all`
  (`--all` + `--terms`), or `.venv/bin/python3 annotate.py [--all |
  --top-dyads N] [--terms] [--limit N] [--quality distilled|fast|balanced|max]
  [--upgrade-distilled] [--dry-run]` — resumable, commits as it goes, Ctrl-C
  safe. Default quality `distilled`: LLM-labels a random ~`ANNOTATE_SEED_TARGET`
  seed sample, then kNN-classifies the whole corpus from `chunk_vectors`
  (~1 min warm; corpus-wide, scope flags don't apply; events left empty).
  LLM tiers (`fast`/`balanced`/`max`, slow → slower, higher fidelity + events)
  scope to top `ANNOTATE_TOP_DYADS` chats bare, honor `--all`, and with
  `--upgrade-distilled` re-annotate kNN rows in scope (how events get filled
  for chats that matter). LLM passes fan out over `ANNOTATE_WORKERS`; `--terms`
  is a mode switch, not an addition. See docs/DECISIONS.md for measurements.
- Topic clusters: `make topics`, or `.venv/bin/python3 topics.py [--k K]
  [--relabel] [--model M]` — k-means over chunk_vectors (assignment in
  sqlite-vec), then an LLM label pass
- Everything, in order: `make pipeline` (ingest → annotate → topics). These
  MUST run sequentially and the target is written to stay serial under `make
  -j`: they hold a SQLite write lock across many LLM calls (concurrent runs
  die with `database is locked`) and contend for one serialized Ollama
  instance, so parallelism costs correctness and buys nothing.
- Wrapped report: `.venv/bin/python3 wrapped.py [--year YYYY] [--contact NAME]
  [--no-llm] [--yes]` — markdown; previews the featured roster (with
  silent-contact auto-exclusions) before generating
- Eval sweep: `eval/fast_eval.sh eval/prompts-smoke.txt [-m model]` — runs a
  prompts file (`id|prompt` lines) through `chat.py -p` sequentially with
  reduced-context env defaults; transcripts land in `eval/results/`. Never
  run eval prompts in parallel — Ollama serializes and it thrashes the KV
  prefix cache.
- Analytics eval (evalkit): `.venv/bin/python3 -m eval.evalkit
  <selfcheck|manifest|run|check|report>` — regression harness sweeping the
  S1–S9 prompt suites (`eval/cases/`) through `chat.run_turn` in-process and
  grading replies against SQL-computed ground truth. `selfcheck`/`manifest`
  need no Ollama; `run` does (sequential, one case at a time — the no-parallel
  rule above applies). Usage + agent-orchestration guide: `eval/README.md`.
  `fast_eval.sh` is the separate, older smoke path — leave it as-is.
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

Each LLM pipeline picks its own generation model: `CHAT_MODEL` (chat.py, plus
`--model`), `ANNOTATE_MODEL` (annotate.py, `--model`), `TOPIC_MODEL`
(topics.py, `--model`), `WRAPPED_MODEL` (wrapped.py). Empty = inherit
`CHAT_MODEL`, so the defaults are a single knob until you need otherwise; the
same inherit-from-chat rule covers `*_THINK` and `*_TIMEOUT`. `EMBED_MODEL` is
deliberately NOT per-pipeline — query-side and document-side embeddings must
come from one model or retrieval silently degrades (see the gotcha below), and
ingest runs no generation model at all.

Module map: `query_tools/` holds the twenty-one retrieval functions the LLM
calls — `messages.py` (message-level SQL/FTS search, context expansion, full
text, reactions), `chunks.py` (chunk-level semantic search, hybrid
FTS+semantic fused with reciprocal-rank fusion, more-like-this, chunk → its
messages), `directory.py` (summary stats; name/handle/chat resolution),
`analytics.py` (per-participant chat stats; word frequency, which scopes by
sender), `analytics_time.py` (activity patterns with local-tz Python
bucketing; response dynamics over messages + the sessions table),
`analytics_social.py` (reaction stats; milestones), `analytics_language.py`
(word trends/LSM/distinctive terms from the lexicon tables; group dynamics)
— person scope means *that person's chats* for all analytics tools except
`get_word_frequency`,
`_shared.py` (connection, result caps, date-bound and scope resolution,
`_REAL_MESSAGE_CLAUSES` excluding emote/system rows from every aggregate), all
re-exported from `__init__.py` so `import query_tools` is the only import
callers need (`DB_PATH`/`embed_texts` are patchable package attributes,
resolved through the package at call time); `tool_schemas.py` their JSON
schemas; `chat.py` the CLI tool-calling loop over Ollama; `ollama.py` the
Ollama chat-streaming HTTP; `embeddings.py` the shared embedding client
(doc/query prefixes, `EMBED_DIM` override, `detect_embed_dim`) used by both
ingest (documents) and query_tools (queries) so both sides embed identically;
`config.py` all env-driven settings (defaults work with no `.env`; see
`.env.example`); `db.py` shared sqlite helpers (`load_vec_extension`,
`CONTACT_LABEL_SQL`, `iso_to_day`, `get_state`/`set_state`); `progress.py` the
shared CLI progress helper (`track(desc, total=...)` — a rich bar on a TTY,
periodic `  ...N/Total` lines when piped, silent under `PROGRESS_STYLE=off`;
progress never drives commit cadence); `schema.py` the
analysis-DB DDL + in-place migration (`create_schema`, run and re-exported
by `ingest.py`). Derived-analytics pipelines (all run by `ingest.py` after
chunking, and re-exported from it): `sessions.py` (gap-split session rows
with initiator/ender), `lexicon_build.py` (`person_period_lexicon` +
`dyad_terms`, full rebuilds — dyad_terms preserves `llm_verdict` curation);
`function_words.py` the bundled open LSM lexicon + tokenize/emoji/laugh
regexes + GoEmotions labels. Standalone batch CLIs: `annotate.py` (LLM chunk
annotations + dyad-term curation via `ollama.chat_once_json`),
`topics.py` (k-means topic clusters + LLM labels), `wrapped.py` (markdown
report composing the query tools deterministically; LLM only narrates
computed numbers). `eval/evalkit/` is the analytics eval harness (stdlib-only;
imports the app in-process, lazily — env overrides must be set before app
modules import): `cases.py` suite loading/validation, `roster.py` roster
resolution, `recipes.py` SQL ground-truth recipes, `manifest.py`
corpus-fingerprinted ground truth, `runner.py` the transcript-capturing sweep,
`checks.py` deterministic grading, `p3.py` LLM-graded checks, `report.py` the
scorecard.

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
                                  -- target_message_guid is normalized: the
                                  -- 'p:N/' / 'bp:' part prefix chat.db puts on
                                  -- associated_message_guid is stripped at
                                  -- ingest (and by an idempotent migration in
                                  -- create_schema) so it joins messages.guid
attachments(id, message_id, filename, mime_type, transfer_name)
chunks(id, chat_id, first_message_id, last_message_id, first_date, last_date,
       n_messages, text)          -- conversation windows; id AUTOINCREMENT so a
                                  -- rebuilt tail chunk never reuses an id
ingest_state(key, value)          -- last_message_rowid for --since-last, plus
                                  -- embed_model/embed_dim guarding chunk_vectors,
                                  -- sessions_max_message_id/sessions_gap_minutes,
                                  -- annotate_model, topic_model_k
sessions(id, chat_id, first_message_id, last_message_id, first_date,
         last_date, n_messages, n_from_me, initiator_handle_id,
         initiator_is_me, ender_handle_id, ender_is_me)
                                  -- gap-split conversation sessions (derived,
                                  -- rebuilt per chat; handle NULL = me)
person_period_lexicon(chat_id, handle_id, month, ...9 fw_*_rate cols...,
         we_count, question_count, laugh_count, emoji_json)
                                  -- per-person-per-month style stats
                                  -- (derived, full rebuild; handle NULL = me)
dyad_terms(chat_id, term, lift, n_uses, first_message_id,
         coined_by_handle_id, coined_is_me, llm_verdict)
                                  -- chat-distinctive vocabulary; llm_verdict
                                  -- (annotate.py --terms) survives rebuilds
chunk_annotations(chunk_id, valence, emotions_json, depth, events_json,
         source)                  -- batch annotations (annotate.py); source
                                  -- 'llm' or 'distilled' (kNN, empty events)
topics(cluster_id, label, n_chunks); chunk_topics(chunk_id, cluster_id)
                                  -- k-means topic clusters (topics.py)
messages_fts                      -- FTS5 external-content table over messages.text
chunk_vectors                     -- sqlite-vec vec0 table: chunk_id ->
                                  -- FLOAT[detected dim] distance_metric=cosine,
                                  -- + chat_id/first_day/last_day METADATA columns
```

## Gotchas (load-bearing)

- `messages.date` strings are unlabeled UTC ISO; compare as plain strings and
  never wrap the column in SQL date functions inside WHERE (kills
  `idx_messages_chat_date`); `substr`/`strftime` bucket expressions are fine
  in SELECT/GROUP BY over already-filtered rows (see `analytics.py`).
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
- `chunk_annotations` is guarded like `chunk_vectors`: `annotate.py` records
  `annotate_model` in `ingest_state` and refuses to annotate with a different
  model while the table is non-empty — switching ANNOTATE_MODEL means
  emptying `chunk_annotations` first (distilled rows inherit the seed model's
  bias, so the guard covers them too). Per-pipeline model vars make this
  easier to trip, not less: the guard is unchanged. `topics.py`'s
  `topic_model_k` guard is warn-only (a rebuild simply replaces the
  clustering).
- `chunk_annotations.source` is `'distilled'` for kNN-classified rows: their
  `events_json` is always `[]` (events can't be distilled — LLM-only), so
  event-based aggregates silently undercount until an LLM tier with
  `--upgrade-distilled` has covered the chats in question.
- `think` is per-call, not global: `ollama.chat_once`/`chat_stream` take
  `think=` with a `_UNSET` default. The tri-state is load-bearing — `None`
  means *omit* the flag (some models reject a `think` option) and is NOT the
  same as `False` (send `think: false`), so never default the param to `None`.
  On a thinking model the flag dominates everything else: qwen3:4b labels a
  topic in 3.2s with `think=false` vs 85s with it on — slower than a 30B.
- `TOPIC_LABEL_SAMPLES` defaults to 30 for a non-obvious reason: label samples
  are ranked by similarity to the cluster centroid, and a big cluster's most
  *central* chunks are its shortest ("ok", "yeah"). At 5 samples the whole
  prompt can be ~250 chars and the model accurately names it "Short
  confirmation responses". Raising it past ~30 risks the context window
  (~1200 chars/sample), where an oversized message is dropped, not rejected.
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
  `get_conversation`. `search_messages_sql`/`search_messages_fts` return an
  envelope (`rows`/`total_matching`/`returned`/`order`/`truncated`), not a
  bare list, so the model can tell a capped sample from the population.
- New tool = function in the right `query_tools/` module (re-export it from
  `__init__.py`) + schema in `tool_schemas.py` (same `name` as the function —
  `chat.py` derives its dispatch table from that) + a one-line entry in
  `chat.py`'s `SYSTEM_PROMPT` tool list (keep all three in sync).
- All tunables (models, hosts, timeouts, batch sizes, chunking) live in
  `config.py` / `.env.example`; never hardcode them elsewhere.
- Runtime deps go in `requirements.txt`, dev-only tools in
  `requirements-dev.txt` (both pinned).
