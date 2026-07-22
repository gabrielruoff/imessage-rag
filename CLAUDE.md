# imessage-rag

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
- Install Ollama + models: `make install-ollama` (brew install — NOT
  `curl ollama.com/install.sh`, which drops a `/Applications/Ollama.app` that
  `make ollama-start`'s `brew services` cannot manage — then `ollama-start`,
  then pulls every model `config.py` names, so it honors `.env`). Kept out of
  `make install` on purpose: that target is fast and offline, this one pulls
  ~15GB and installs a background service.
- Ingest: `make ingest`, or `.venv/bin/python3 ingest.py [--no-embed]
  [--since-last] [--senders=...] [--exclude=...]`
- Chat CLI: `.venv/bin/python3 chat.py` (interactive REPL), or
  `chat.py -p "..."` for a one-shot answer (prints only the reply; exit 1 if
  Ollama is unreachable)
- Annotate chunks: `make annotate` (chunks + `--terms`) / `make annotate-all`
  (`--all` + `--terms`), or `.venv/bin/python3 annotate.py [--all |
  --top-dyads N] [--limit N] [--quality distilled|fast|balanced|max]
  [--upgrade-distilled] [--dry-run]` — resumable, commits as it goes, Ctrl-C
  safe. Default quality `distilled`: LLM-labels a random ~`ANNOTATE_SEED_TARGET`
  seed sample, then kNN-classifies the whole corpus from `chunk_vectors`
  (~1 min warm; corpus-wide, scope flags don't apply; events left empty).
  LLM tiers (`fast`/`balanced`/`max`, slow → slower, higher fidelity + events)
  scope to top `ANNOTATE_TOP_DYADS` chats bare, honor `--all`, and with
  `--upgrade-distilled` re-annotate kNN rows in scope (how events get filled
  for chats that matter). LLM passes fan out over `ANNOTATE_WORKERS`; `--terms`
  is a mode switch, not an addition — it's the separate pipeline below, and
  rejects `--quality`/`--upgrade-distilled`. See docs/DECISIONS.md for
  measurements.
- Curate dyad terms: `.venv/bin/python3 annotate.py --terms [--limit N]
  [--retry-failed] [--dry-run]` — a separate pipeline, NOT a `--quality` tier
  (passing `--quality` with `--terms` is an error, and `ANNOTATE_THINK` does
  not reach it). It batches `ANNOTATE_TERMS_BATCH_SIZE` terms per request
  (never spanning chats) and runs `ANNOTATE_TERMS_THINK`, default false:
  reasoning about a one-word verdict measured ~11x latency for an identical
  answer (gemma4: 282.6s vs 26.2s per term). A term that fails
  `ANNOTATE_TERMS_MAX_FAILURES` times is parked in `dyad_terms.curate_failures`
  rather than retried by every later run; `--retry-failed` unparks it.
- Topic clusters: `make topics`, or `.venv/bin/python3 topics.py [--k K]
  [--relabel] [--model M]` — k-means over chunk_vectors (assignment in
  sqlite-vec), then an LLM label pass
- Analytical-suite pipelines (Wave 0+1), each a standalone CLI + `make` target,
  all reading `chunk_annotations` so they run after `annotate`:
  `make nudges` (`loose_ends.py` — Loose Ends detectors → `loose_threads`;
  Ollama for the one-line nudge narration only), `make drift` (`drift.py` —
  relationship-drift scoring → `dyad_drift`; LLM-free and opt-in: `DRIFT_ENABLED`
  defaults false, so the step is a no-op until you turn it on;
  `make drift-purge` runs `drift.py --purge` to wipe `dyad_drift` + its guard
  keys regardless of the flag, the teardown you run while turning the feature
  off), `make voiceprint`
  (`voiceprint.py` — per-person voiceprints → `person_voiceprint`; LLM-free),
  `make conflict` (`conflict_detect.py` — deterministic conflict finder +
  eval-gated labels → `conflicts`/`conflict_signals`; Ollama for the label pass).
- Everything, in order: `make pipeline`
  (ingest → annotate → nudges → drift → voiceprint → conflict → topics). These
  MUST run sequentially and the target is written to stay serial under `make
  -j`: they hold a SQLite write lock across many LLM calls (concurrent runs
  die with `database is locked`) and contend for one serialized Ollama
  instance, so parallelism costs correctness and buys nothing.
- Wrapped report: `.venv/bin/python3 wrapped.py [--year YYYY] [--contact NAME]
  [--no-llm] [--yes] [--wellbeing]` — a self-contained HTML page (inline CSS +
  JS, no network, light/dark, scroll-reveal animations); previews the featured
  roster (with silent-contact auto-exclusions) before generating. `--wellbeing`
  (or `WRAPPED_WELLBEING=true`) opts the wellbeing trajectory section in — off
  by default even when a window fires
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
- Eval-gate labeling (F4): `.venv/bin/python3 -m eval.evalkit label <task>`
  interactively hand-labels sampled items into the gitignored
  `eval/labels/<task>.jsonl` (owner-only — agents never write there);
  `... gate <task>` runs the task's production classifier over those labels,
  scores kappa/precision vs gold, and persists the pass/fail verdict to
  `ingest_state`. Tasks: `conflict_labels` (kappa), `dropped_plan` (precision),
  `wellbeing_narration`. See the eval-gate gotcha below for clearance precedence.
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
same inherit-from-chat rule covers `*_THINK` and `*_TIMEOUT` — except the
annotate `think` vars, which carve themselves out: empty `ANNOTATE_THINK`
means the `--quality` preset's value, and empty `ANNOTATE_TERMS_THINK` means
`false` (a term verdict is one word; reasoning about it is ~11x latency for
the same answer). Because empty is spoken for in both, they alone take
`omit`/`none` to reach the flagless third state that the other `*_THINK` vars
get from being unset. `EMBED_MODEL` is
deliberately NOT per-pipeline — query-side and document-side embeddings must
come from one model or retrieval silently degrades (see the gotcha below), and
ingest runs no generation model at all.

Module map: `query_tools/` holds the thirty-two retrieval functions the LLM
calls by default (`len(tool_schemas.TOOL_DEFINITIONS)`); five more register
only behind feature flags — the three drift read tools behind `DRIFT_ENABLED`,
`trace_conflict_across_chats` behind `CONFLICT_TRACE_ENABLED`, and
`simulate_contact` behind `PERSONA_SIM_ENABLED` (all five functions are always
importable from the package; only their schemas are conditional). The modules — `messages.py` (message-level SQL/FTS search, context expansion, full
text, reactions), `chunks.py` (chunk-level semantic search, hybrid
FTS+semantic fused with reciprocal-rank fusion, more-like-this, chunk → its
messages), `directory.py` (summary stats; name/handle/chat resolution),
`analytics.py` (per-participant chat stats; word frequency, which scopes by
sender), `analytics_time.py` (activity patterns with local-tz Python
bucketing; response dynamics over messages + the sessions table),
`analytics_social.py` (reaction stats; milestones), `analytics_language.py`
(word trends/LSM/distinctive terms from the lexicon tables),
`analytics_group.py` (group dynamics; imports the lexicon/LSM primitives from
`analytics_language.py`), `nudges.py` (the Loose Ends read tools —
`get_loose_threads`/`resolve_loose_thread`/`mute_contact_from_nudges` — over
`loose_threads`/`nudge_exclusions`), `conflict.py`
(`find_conflicts`/`get_conflict` over the deterministic conflict finder's
output tables, plus the generative surfaces `diagnose_conflict` — the
deterministic scorecard — and `draft_apology` — the Lewicki-6 repair scaffold,
both under the code-enforced DV floor — and, behind `CONFLICT_TRACE_ENABLED`,
`trace_conflict_across_chats`), `persona.py` (`get_voiceprint` reading
`person_voiceprint`, plus `draft_reply`, the deterministic `check_style_match`,
and — behind `PERSONA_SIM_ENABLED` — `simulate_contact`, the in-tool ESTIMATE of
a contact's voice or your past self: it generates via `ollama.chat_once` and
returns text it has already wrapped with a per-line "estimate, not <person>"
label in code, under a hard deceased/grief denylist (`PERSONA_SIM_BLOCK_IDS`,
parsed by `consent`), a soft last-message-age + one-sided dormant warning, and an
`acknowledged` UX gate for living contacts), `analytics_wellbeing.py`
(`get_language_trajectory` — the
deterministic cross-family change-point read, self by default and consent-gated
per contact via `consent.is_consented`), `drift.py` (the D2/D3 read tools
`get_relationship_drift`/`get_fadeout_watchlist`/`get_reconnection_candidates`
over `dyad_drift`, all schema-gated behind `DRIFT_ENABLED`)
— person scope means *that person's chats* for all analytics tools except
`get_word_frequency`,
`_shared.py` (connection, result caps, date-bound and scope resolution,
`_REAL_MESSAGE_CLAUSES` excluding emote/system rows from every aggregate,
`_restrict_to_direct_chats` — the shared 1:1-only scope narrowing the dyad
features (nudges/conflict) reuse — and `member_key`, the single source of
participant-identity merging), all
re-exported from `__init__.py` so `import query_tools` is the only import
callers need (`DB_PATH`/`embed_texts` are patchable package attributes,
resolved through the package at call time); `tool_schemas.py` their JSON
schemas; `chat.py` the CLI tool-calling loop over Ollama; `ollama.py` the
Ollama chat-streaming HTTP; `embeddings.py` the shared embedding client
(doc/query prefixes, `EMBED_DIM` override, `detect_embed_dim`) used by both
ingest (documents) and query_tools (queries) so both sides embed identically;
`config.py` all env-driven settings (defaults work with no `.env`; see
`.env.example`); `db.py` shared sqlite helpers (`load_vec_extension`,
`CONTACT_LABEL_SQL`, `iso_to_day`, `get_state`/`set_state`, plus the batch-CLI
`connect_default` and `run_cli` boilerplate); `progress.py` the
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
annotations + dyad-term curation via `ollama.chat_once_json`; the distilled
kNN tier lives in `annotate_distill.py`, which annotate.py calls),
`topics.py` (k-means topic clusters + LLM labels), `wrapped.py` (HTML report
composing the query tools deterministically; LLM only narrates computed
numbers — the `_sec_*` section builders live in `wrapped_sections.py`, and
`wrapped_html.py` holds the page shell (incl. the inline scroll-reveal JS),
the shape-dispatching stats renderer (tiles/tables/charts), and the visual
builders — hbars, split bars, iMessage bubbles, word cloud, share bars, joke
cards; sections put visuals in `extra_html` and the full shaped data in
`narrate_stats` so the narrator still sees everything). The analytical-suite
pipelines (Wave 0+1) are standalone CLIs too — each its own `make` target and a
serial step in `make pipeline`, none run by `ingest.py`: `loose_ends.py` (Loose
Ends / nudge detectors → `loose_threads`; Ollama only for the one-line
`nudge_text`, guarded by `loose_ends_model`), `drift.py` (deterministic
relationship-drift scoring → `dyad_drift`; opt-in behind `DRIFT_ENABLED`,
LLM-free), `voiceprint.py` (deterministic per-person voiceprints →
`person_voiceprint`; LLM-free), `conflict_detect.py` (deterministic conflict
finder + eval-gated escalation labels → `conflicts`/`conflict_signals`; Ollama
only for the label pass, guarded by `conflict_model`). Shared foundations they
build on: `stats.py` (dependency-free robust-stats primitives — Theil-Sen trend
+ bootstrap CI, Kaplan-Meier survival, MAD outlier / CUSUM change-point —
imported by drift/conflict/wellbeing), `consent.py` (default-closed opt-in id
parsing via `is_consented`, now consumed by the per-contact wellbeing gate and
the Wrapped voiceprint card), `wellbeing.py` (deterministic within-person
change-point core, now wired by `query_tools/analytics_wellbeing.py`'s
`get_language_trajectory` and the opt-in Wrapped section; also owns the
code-emitted `support_block` and the `sanitize_condition_names` scrub the chat
+ Wrapped surfaces run over their narration).
`eval/evalkit/` is the analytics eval harness
(stdlib-only; imports the app in-process, lazily — env overrides must be set
before app modules import): `cases.py` suite loading/validation, `roster.py`
roster resolution, `recipes/` SQL ground-truth recipes (a package: `_base.py`
holds the `@recipe` registry + shared helpers, one module per suite family
registers its recipes on import), `manifest.py`
corpus-fingerprinted ground truth, `runner.py` the transcript-capturing sweep,
`checks.py` deterministic grading, `p3.py` LLM-graded checks, `report.py` the
scorecard; plus the F4 eval-gate harness — `gates.py` (the measurement-gate
registry + the `eval_gate_passed` clearance helper), `agreement.py` (Cohen's
kappa / detector-precision math), `labeling.py` (the owner-only `label` + `gate`
subcommands). Its slow loops (`run`, `manifest`'s ground truth, `p3 --judge`)
reuse `progress.track` — imported function-locally, like every other app
import here, because `progress` pulls in `config` and `__main__` imports all
submodules eagerly.

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
                                  -- annotate_model, topic_model_k, the
                                  -- suite guards: loose_ends_model,
                                  -- conflict_model, drift_max_message_id +
                                  -- drift_params_hash + drift_annotations_hash +
                                  -- drift_sessions_hash,
                                  -- voiceprint_max_message_id,
                                  -- eval_gate:<task> (F4 gate verdicts), and
                                  -- nudges_reconnect_ack (durable dormant-tie
                                  -- reconnect opt-in; unset hides those rows)
sessions(id, chat_id, first_message_id, last_message_id, first_date,
         last_date, n_messages, n_from_me, initiator_handle_id,
         initiator_is_me, ender_handle_id, ender_is_me)
                                  -- gap-split conversation sessions (derived,
                                  -- rebuilt per chat; handle NULL = me)
person_period_lexicon(chat_id, handle_id, month, ...9 fw_*_rate cols...,
         we_count, question_count, laugh_count, emoji_json,
         i_count, you_count, absolutist_count)
                                  -- per-person-per-month style stats
                                  -- (derived, full rebuild; handle NULL = me).
                                  -- i/you/absolutist are raw PRONOUN_I /
                                  -- PRONOUN_YOU / ABSOLUTIST token counts (F1),
                                  -- nullable until the next full rebuild fills them
dyad_terms(chat_id, term, lift, n_uses, first_message_id,
         coined_by_handle_id, coined_is_me, llm_verdict, curate_failures)
                                  -- chat-distinctive vocabulary; llm_verdict
                                  -- (annotate.py --terms) survives rebuilds,
                                  -- as does curate_failures: at
                                  -- ANNOTATE_TERMS_MAX_FAILURES the term is
                                  -- parked (never retried until
                                  -- --retry-failed), so a rebuild must not
                                  -- reset the count to 0
chunk_annotations(chunk_id, valence, emotions_json, depth, events_json,
         source)                  -- batch annotations (annotate.py); source
                                  -- 'llm' or 'distilled' (kNN, empty events)
topics(cluster_id, label, n_chunks); chunk_topics(chunk_id, cluster_id)
                                  -- k-means topic clusters (topics.py)
loose_threads(id, kind, chat_id, handle_id, anchor_message_id, anchor_date,
         due_by, urgency, status, evidence_json, llm_verdict, llm_reason,
         nudge_text, source, detected_at, dismissed_at)
                                  -- Loose Ends detector output (loose_ends.py);
                                  -- status 'open'|'dismissed' (dismissed rows are
                                  -- kept, never deleted); the LLM columns are
                                  -- guarded by loose_ends_model. due_by is filled
                                  -- by the inferred_occasion detector (next
                                  -- upcoming date) and NULL for other kinds.
                                  -- reset_db clears it
nudge_exclusions(handle_id, contact_id, reason, created_at)
                                  -- contacts/handles muted from nudges; durable
                                  -- user intent, PRESERVED across reset_db
dyad_drift(contact_key, label, n_messages, n_months, first_month, last_month,
         drift_score, phase, phase_corroborated, mandatory_complete,
         confidence_tier, ...7x <signal>_value/<signal>_gate...,
         per_signal_gate_json, coverage_json, monthly_series_json)
                                  -- per-counterparty relationship drift (drift.py;
                                  -- one row per member_key, opt-in DRIFT_ENABLED);
                                  -- full rebuild, guarded by drift_max_message_id +
                                  -- drift_params_hash + drift_annotations_hash +
                                  -- drift_sessions_hash (annotations re-fingerprints
                                  -- chunk_annotations so an --upgrade-distilled
                                  -- re-annotation rebuilds; sessions re-fingerprints
                                  -- the sessions table so a SESSION_GAP_MINUTES
                                  -- retune rebuilds)
person_voiceprint(member_key, counterpart_key, scope, avg_msg_len,
         lowercase_ratio, exclaim_rate, ellipsis_rate, burst_size,
         ...9 fw_*_rate cols..., openers_json, closers_json, tone_valence,
         tone_emotions_json, n_messages, n_reply_pairs, n_months,
         feature_std_json)        -- deterministic per-person voice profile
                                  -- (voiceprint.py); scope 'dyad'|'group'|'all';
                                  -- tone_* are LOW-TRUST (source='llm' chunks
                                  -- only); full rebuild, guarded by
                                  -- voiceprint_max_message_id
conflicts(id, chat_id, first_date, last_date, start_session_id, end_session_id,
         topic, peak_signal_json, corroboration, off_text_flag, source)
                                  -- deterministic conflict episodes
                                  -- (conflict_detect.py; 1:1 only); corroboration
                                  -- 'none'|'weak'|'llm' (valence, source='llm'
                                  -- only); source guarded by conflict_model. NO
                                  -- role/aggressor/status column by design (C2)
conflict_signals(id, conflict_id, message_id, actor_is_me, label_class,
         top2_labels_json, band, excerpt)
                                  -- per-turn escalation labels for a conflict;
                                  -- band 'tentative'|'likely' (never a raw score),
                                  -- only for eval-gate-cleared classes; excerpt
                                  -- verbatim <= 200 chars
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
- Valence-dependent reads trust ONLY `chunk_annotations` rows with
  `source='llm'`; distilled (kNN) rows are never consumed for a high-stakes
  read. Conflict corroboration, the nudge Stage-0 valence exclusion, and the
  voiceprint tone field all read llm rows only and degrade honestly when there
  are none: conflict writes `corroboration='none'` and still detects, a dyad
  with no llm chunks is simply not valence-excluded, and `tone_*` reads NULL.
  Run `annotate.py --quality balanced --upgrade-distilled` over the chats you
  care about to light them.
- The F4 eval gate decides whether a pipeline's high-stakes labels ship.
  `eval_gate_passed(con, task, cls)` resolves clearance by precedence: an owner
  verdict in `ingest_state` under `eval_gate:<task>` (written by
  `eval.evalkit gate`) wins; else a committed `eval/seed_gates/<task>.json`
  seed; else nothing. An ABSENT verdict means the gated classes are SUPPRESSED
  (default-closed — Master Spec A2), not shown. Only `conflict_labels` ships a
  seed (pre-cleared out of the box, and because the seed lives in the repo it
  survives `reset_db` / a fresh DB); `dropped_plan` and `wellbeing_narration`
  stay suppressed until the owner hand-labels and gates them.
- An enable-flag gates a feature's tool REGISTRATION or pipeline BUILD, never
  the prompt. `DRIFT_ENABLED=false` (the default) makes `drift.py` exit and
  `dyad_drift` never build, AND withholds the three drift read tools + D3 from
  `TOOL_DEFINITIONS`; `CONFLICT_TRACE_ENABLED=false` (default) withholds
  `trace_conflict_across_chats` (its per-call `consent` param is a UX prompt
  device, NOT the boundary — the schema gate is); `PERSONA_SIM_ENABLED=false`
  (default) withholds `simulate_contact` (its per-call `acknowledged` param is a
  UX flow, NOT the boundary — the schema gate is; the deceased/grief denylist and
  the soft dormant warning are enforced in the tool's own code, not by the
  prompt). Same principle as withholding
  tool schemas to stop a tool call: prompting a model not to do something is not
  a control; removing the capability is. The N4 nudge detectors gate at BUILD
  the same way — `NUDGE_DROPPED_PLAN_ENABLED`/`NUDGE_GOODNEWS_ENABLED` default
  off and their off-path runs no LLM and never consults the eval gate; even ON,
  `dropped_plan` writes nothing until the F4 precision gate clears it
  (`eval_gate_passed(con, 'dropped_plan', 'dropped_plan')`, and it ships no
  seed, so an absent verdict suppresses). Wrapped's third-party content gates at
  BUILD too: `WRAPPED_WELLBEING` (default off) decides whether the wellbeing
  section is composed, and `WRAPPED_VOICEPRINT_CONSENT_IDS` (default empty) is
  default-redacted — with no consented ids NO third-party verbatim/stylometric
  card is emitted at all.
- The DV floor (Master Spec A10) is SOFTENED and enforced in code, never by
  prose. `query_tools.conflict._dv_screen` scans the full text of a conflict
  window and returns a status: `clean` (no marker), `role-suppress`
  (marker + `CONFLICT_DV_SCREEN` on — the default), or `bypassed-with-footer`
  (marker + screen off, via `CONFLICT_DV_SCREEN=off` or `chat.py
  --no-dv-screen` → `set_dv_screen_bypass`). A `role-suppress` hit makes
  `diagnose_conflict` withhold ONLY the demand-withdraw/role read while STILL
  returning its mechanical scorecard, whereas `draft_apology` withholds the
  ENTIRE scaffold (drafting an apology is itself the risky output — it can
  pressure a victim into apologizing to an abuser). The `resource_footer`
  (hotline text) is NEVER removed: a `bypassed-with-footer` hit restores the
  role read but still appends it. The scan is read-only; the hit is never
  persisted.
- Per-contact wellbeing is a code-enforced DENYLIST, never a prompt.
  `get_language_trajectory` analyzes any `contact` by default and refuses only a
  deny-listed one (`WELLBEING_DENY_IDS`, default empty → analyze anyone) via
  `consent.is_denied`, before any analysis; self is never affected. Distinct
  from the Wrapped voiceprint card's third-party game, which stays a
  default-closed ALLOWLIST (`WRAPPED_VOICEPRINT_CONSENT_IDS`, `consent.is_consented`)
  and only ADDS a contact — never overriding a roster/silent-contact exclusion
  (those filter first). `consent.py` ships both membership helpers; the caller
  picks allow vs deny.
- Wellbeing narration is framed by two code mechanisms in `chat.run_turn`, not
  by the prompt. When `get_language_trajectory` runs, the turn (a) arms a
  whole-reply sanitize: streaming is held back for that turn and the ENTIRE
  assembled reply is run through `wellbeing.sanitize_condition_names` before it
  is printed or returned, so a condition name in any sentence is redacted (not
  just the last one); and (b) collects any tool result's reserved `_epilogue`
  string (the code-emitted `support_block`), strips both `_epilogue` AND the
  duplicate `support_block` field from the data the model sees, and prints the
  epilogue verbatim after the reply. `_epilogue` is a generic hook any tool may
  set; the `get_language_trajectory` envelope also carries a model-safe
  `flagged` boolean (the transcript-reader's departure signal now that
  `support_block` is stripped). wrapped.py runs the same
  `sanitize_condition_names` scrub over its wellbeing blurb.
- `think` is per-call, not global: `ollama.chat_once`/`chat_stream` take
  `think=` with a `_UNSET` default. The tri-state is load-bearing — `None`
  means *omit* the flag (some models reject a `think` option) and is NOT the
  same as `False` (send `think: false`), so never default the param to `None`.
  On a thinking model the flag dominates everything else: qwen3:4b labels a
  topic in 3.2s with `think=false` vs 85s with it on — slower than a 30B.
  Corollary: an effort level is not a portable dial. `think='low'` is an
  effort level only to models that have them (gpt-oss); to gemma4 it is just
  truthy, so it buys a *full* reasoning pass — 282.6s vs 26.2s per dyad term,
  same verdict. Never route one pipeline's `think` into another's: `--terms`
  inherited the `--quality` presets' and paid ~11x for nothing, which is why
  term curation now owns `ANNOTATE_TERMS_THINK` and rejects `--quality`.
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
  `config.py` / `.env.example`; never hardcode them elsewhere. The Wave 0–4
  suite adds the `CONFLICT_*` (incl. `CONFLICT_DV_SCREEN`,
  `CONFLICT_TRACE_ENABLED`), `NUDGE_*` (dormant/question/dropped_plan/goodnews/
  occasion knobs + the enable flags), `DRIFT_*`, `VOICEPRINT_*` (+ the V5
  `PERSONA_SIM_*` family: `PERSONA_SIM_ENABLED`, `PERSONA_SIM_BLOCK_IDS`, and the
  soft-dormant `PERSONA_SIM_DORMANT_DAYS`/`PERSONA_SIM_ONESIDED_MIN_MSGS`),
  `WELLBEING_*` (+ the `WELLBEING_DENY_IDS` per-contact deny list), and `WRAPPED_*` (`WRAPPED_WELLBEING`,
  `WRAPPED_VOICEPRINT_CONSENT_IDS`) families — all env-driven with working
  defaults, same as the rest.
- Participant-identity merging (me / contact / handle / unknown) has one home:
  `query_tools/_shared.member_key`. Modules keep thin named wrappers for their
  exact contracts (`_person_key` maps no-identity rows to `None`, the others to
  an `"unknown"` bucket) — never re-implement the key logic.
- Runtime deps go in `requirements.txt`, dev-only tools in
  `requirements-dev.txt` (both pinned).
- Audit artifacts (`REPO_REVIEW.md`, `REPO_FIX_PLAN.md`) are local working
  files, gitignored — never commit them.
