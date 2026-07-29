# imessage-rag

Makes iMessage history queryable by a local LLM: ingest macOS `chat.db` into a
normalized SQLite store (`imessage_analysis.db`) with FTS5 keyword search and
sqlite-vec semantic search over conversation chunks, exposed to an Ollama chat
model (or opt-in Apple Foundation Models) as tool calls. Retrieval, not
context-stuffing — the corpus never enters the prompt.

Full design history, schema rationale, and data forensics:
[`docs/DECISIONS.md`](docs/DECISIONS.md). Feature runbook: `docs/DEMO.md`.

## Commands

- Install: `make install` (needs Homebrew Python — python.org builds lack the
  sqlite loadable-extension support sqlite-vec requires). `make bootstrap` =
  install + Ollama + source data + ingest.
- Ollama + models: `make install-ollama` (brew — NOT the curl installer, which
  drops an `/Applications/Ollama.app` `brew services` can't manage — then pulls
  every model `config.py` names). Apple Foundation Models backend (no Ollama):
  `make apple-shim`, then point any `*_MODEL` var at the `apple` sentinel
  (`CHAT_MODEL`/`EMBED_MODEL`/`ANNOTATE_MODEL`/…). Needs macOS 26+, Apple
  Silicon, Apple Intelligence. Both kept out of `make install` (slow / network).
- Ingest: `make ingest` / `python3 ingest.py [--no-embed] [--since-last]
  [--senders=...] [--exclude=...] [--lookback=PERIOD]` (`--lookback`/`-l` takes a
  period like `7d`/`3mo`/`2y` — units d/w/mo/y, calendar-correct — adding a
  `message.date >= now-PERIOD` lower bound; combines with the other flags, drops
  dateless/system rows for that run)
- Chat: `python3 chat.py` (REPL) or `chat.py -p "..."` (one-shot; prints only
  the reply, exit 1 if Ollama is unreachable).
- Annotate chunks: `make annotate` / `make annotate-all` / `make annotate-hybrid`,
  or `python3 annotate.py [--all | --top-dyads N] [--limit N]
  [--quality distilled|hybrid|fast|balanced|max] [--upgrade-distilled]
  [--dry-run]` — resumable, Ctrl-C safe. Default `distilled` = LLM-labels a
  ~`ANNOTATE_SEED_TARGET` seed then kNN-classifies the whole corpus from
  `chunk_vectors` (fast; events left empty). `hybrid` = a stratified
  `ANNOTATE_HYBRID_SEED_TARGET` seed draw + that same kNN pass (both corpus-wide),
  then LLM upgrades — event-candidate chunks first (chunk-id ascending, so one
  run covers the oldest history and reruns advance), lowest-`knn_confidence`
  next — until the `ANNOTATE_BUDGET_MINUTES` wall-clock deadline; budgeted in
  TIME not chunks, so it self-terminates and a rerun resumes, the scope flags
  reach only the upgrade phase, and `--upgrade-distilled` is an error with it
  (hybrid IS one). EVERY phase is deadline-gated, including the deterministic
  kNN pass (O(rows × seeds): 14.6 s at a 450-seed pool vs 82.0 s at 4,416) and
  seeding (also capped at 25% of the budget — unbatched and uncapped it
  measured 10.7 min of an 8-min run). Each phase starts from the per-request
  time an earlier run banked in `ingest_state`, so nothing is dispatched blind;
  the floor is that ONE request is the smallest dispatchable unit (~106 s for 8
  event chunks), so a budget under ~5 min buys nothing and says so. Its seed
  target counts a `hybrid_seeds` counter, NOT `source='llm'` rows — the tier's
  own upgrades are `llm` rows and counting them switched its sampler off
  permanently. LLM tiers (`fast`/`balanced`/`max`) scope to top
  `ANNOTATE_TOP_DYADS`, honor `--all`, and with `--upgrade-distilled`
  re-annotate kNN rows (how events get filled). `--terms` is a SEPARATE
  pipeline (below), not a `--quality` tier.
- Curate dyad terms: `python3 annotate.py --terms [--limit N] [--retry-failed]
  [--dry-run]` — its own pipeline; passing `--quality` is an error. Batches
  `ANNOTATE_TERMS_BATCH_SIZE` terms/request (never spanning chats), runs
  `ANNOTATE_TERMS_THINK` (default false: a one-word verdict measured ~11x latency
  with reasoning on). A term failing `ANNOTATE_TERMS_MAX_FAILURES` times is parked
  in `dyad_terms.curate_failures`; `--retry-failed` unparks it.
- Topics: `make topics` / `python3 topics.py [--k K] [--relabel] [--model M]` —
  k-means over chunk_vectors (assignment in sqlite-vec) + an LLM label pass.
- Analytical suite (each a standalone CLI + `make` target, all read
  `chunk_annotations` so they run after `annotate`): `make nudges`
  (`loose_ends.py` → `loose_threads`; Ollama only for the nudge narration),
  `make drift` (`drift.py` → `dyad_drift`; LLM-free, opt-in `DRIFT_ENABLED`,
  default off — `make drift-purge` tears it down), `make voiceprint`
  (`voiceprint.py` → `person_voiceprint`; LLM-free), `make conflict`
  (`conflict_detect.py [--budget-minutes N] [--label-top N]
  [--max-label-turns N] [--no-llm] [--limit N] [--dry-run]` →
  `conflicts`/`conflict_signals`; Ollama only for the per-turn labels). A bare
  `make conflict` is two-phase and budgeted in TIME, like `--quality hybrid`:
  deterministic detection over ALL 1:1 chats (~24 s on the reference corpus,
  unchanged), then per-turn labeling of the STRONGEST windows first
  (`_conflict_strength`: signal count → corroboration → dislikes → peak gap)
  until the `CONFLICT_BUDGET_MINUTES` (default 25) wall-clock deadline, clocked
  from the top of `main` so detection spends it too. It stops dispatching when
  the remainder can't cover the next window, commits, and prints how many
  windows were labeled vs left; a rerun continues down the ranking, skipping
  windows already stamped `conflicts.labeled_at` (stamped even when every label
  was suppressed, and preserved when an unchanged window re-detects).
  `--budget-minutes N` overrides the env (flag > env > default), `0` = the
  unbudgeted legacy path (label every detected window), it composes with
  `--label-top N` (both caps apply), and it is an ERROR with `--no-llm` (no LLM
  work to budget). Turn selection is peak-anchored on `peak_gap_message_id`
  (not the first N chronological turns) and prescreened by
  `CONFLICT_MIN_TURN_CHARS` (default 15; a shorter or alphanumeric-free turn
  can't carry an escalation signal and is never dispatched), so
  `CONFLICT_MAX_LABEL_TURNS` now defaults to 6, not 12. Per-call seconds bank in
  `ingest_state` under `conflict_turn_seconds:<model>` so the deadline
  arithmetic is never blind, and an unset `CONFLICT_THINK` resolves to `False`
  for this pipeline (197 s → 2.2 s per call — see the `think` gotcha).
- Everything, in order: `make pipeline`
  (ingest → annotate → nudges → drift → voiceprint → conflict → topics). MUST
  stay serial (written to under `make -j` too): they hold a SQLite write lock
  across many LLM calls (`database is locked`) and share one serialized Ollama.
- Wrapped report: `python3 wrapped.py [--year YYYY] [--contact NAME] [--no-llm]
  [--yes] [--wellbeing]` — a self-contained HTML page (inline CSS+JS, no network,
  light/dark, scroll-reveal); previews the roster (with silent-contact auto-
  exclusions) first. `--wellbeing`/`WRAPPED_WELLBEING` opts the trajectory
  section in (off by default even when a window fires).
- Eval: `eval/fast_eval.sh eval/prompts-smoke.txt [-m model]` (older smoke path,
  `id|prompt` lines through `chat.py -p`, sequential — never parallel). Analytics
  regression (evalkit): `python3 -m eval.evalkit <selfcheck|manifest|run|check|
  report>` sweeping the S1–S17 suites (`eval/cases/`) through `chat.run_turn`
  in-process against SQL ground truth (`run` needs Ollama, sequential). F4
  eval-gate labeling (owner-only): `python3 -m eval.evalkit label <task>` /
  `... gate <task>` — hand-labels into the gitignored `eval/labels/<task>.jsonl`,
  runs the classifier, persists the pass/fail to `ingest_state`. Tasks:
  `conflict_labels` (kappa), `dropped_plan` (precision), `wellbeing_narration`.
  See `eval/README.md`.
- Test: `python3 -m unittest discover tests` (hermetic — no Ollama/real
  `chat.db`). NEVER run unless you touched `tests/` or were asked — slow, hangs
  the terminal; prefer one module (`-m unittest tests.test_foo`). Lint: `make
  lint` (ruff, line-length 100, E+F). CI-equivalent: `make check`. Reset analysis
  DB (keeps schema): `make reset-db`.

## Architecture

`ingest.py` runs the source pipeline: snapshot the live `chat.db`, decode
`attributedBody`, upsert into `imessage_analysis.db` (append-only), build FTS5,
then run the derived stages. `chunking.py` groups each chat's messages into
conversation chunks (split on a > `CHUNK_GAP_MINUTES` silence or at
`CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_MESSAGES` carried across size splits) and embeds
them into the sqlite-vec `chunk_vectors` table (resumable; each vector carries
`chat_id`/`first_day`/`last_day` vec0 metadata so scoped semantic search filters
inside the KNN scan, cosine distance). Messages are embedded as chunks, not
individually (most carry no retrievable meaning alone; ~5–10x fewer vectors).

Models (local, via Ollama): chat `qwen3.5:4b` (~4 GB resident, and the model
every timing default in the repo was measured on — a bigger `CHAT_MODEL` routes
more precisely across the ~32 tools, at ~20 GB for a 26B MoE), embeddings
`nomic-embed-text` (Ollama refuses `/api/embed` for generation models, so chat
can't double as embedder; nomic task prefixes on by default, dim auto-detected).
Optional Apple Foundation Models backend (`apple` sentinel, `config.is_apple_model`):
on-device model via a compiled Swift shim (`applefm.py` ↔ `bin/applefm-shim`);
`ollama.py`/`embeddings.py` dispatch on the sentinel so callers stay
backend-agnostic; the chat path is a separate staged orchestrator
(`chat_apple.py`) because the FM window is a fixed 4,096 tokens.

Each LLM pipeline picks its own generation model — `CHAT_MODEL` (chat.py,
`--model`), `ANNOTATE_MODEL`, `TOPIC_MODEL`, `WRAPPED_MODEL`; empty = inherit
`CHAT_MODEL` (same for `*_THINK`/`*_TIMEOUT`, except the annotate think vars,
which carve themselves out — see the `think` gotcha). `EMBED_MODEL` is
deliberately NOT per-pipeline: query- and document-side embeddings must come from
one model or retrieval silently degrades (gotcha below).

## Module map

`query_tools/` holds 32 default retrieval tools (`len(tool_schemas.TOOL_DEFINITIONS)`,
flags off) + 5 that register only behind feature flags (the 3 drift read tools
behind `DRIFT_ENABLED`, `trace_conflict_across_chats` behind
`CONFLICT_TRACE_ENABLED`, `simulate_contact` behind `PERSONA_SIM_ENABLED`) — all
5 functions are always importable, only their schemas are conditional. Everything
re-exports from `query_tools/__init__.py`, so `import query_tools` is the only
import callers need (`DB_PATH`/`embed_texts` are patchable package attributes).

| Area | Files |
|---|---|
| Entry CLIs | `chat.py` (tool-calling loop; dispatches `chat_apple.py` on apple), `ingest.py`, `annotate.py` (+`annotate_distill.py` the distilled kNN tier, `annotate_hybrid.py` the wall-clock-budgeted hybrid tier — with `annotate_sampling.py` its stratified seed draw and `annotate_events.py` its event prefilter), `topics.py`, `wrapped.py` (+`wrapped_sections.py`/`wrapped_sections_suite.py` builders, `wrapped_html.py` shell+generic renderer/`wrapped_visuals.py` feature cards), `reset_db.py` |
| Ingest / derived builders | `chunking.py`, `sessions.py`, `lexicon_build.py` (`person_period_lexicon`+`dyad_terms`), `schema.py` (DDL + migration), `function_words.py` (LSM lexicon + regexes) — all run by `ingest.py` |
| Suite pipelines (standalone CLI + `make` target, after annotate) | `loose_ends.py` (+`nudge_detectors.py`, the six detectors) → `loose_threads`; `drift.py` → `dyad_drift` (opt-in); `voiceprint.py` → `person_voiceprint`; `conflict_detect.py` → `conflicts`/`conflict_signals` |
| Shared libs | `config.py` (all env settings), `db.py` (sqlite helpers, `load_vec_extension`, `iso_to_day`, `get_state`/`set_state`, `connect_default`/`run_cli`), `ollama.py` (+`applefm.py`), `embeddings.py` (doc/query prefixes, dim detect), `progress.py` (`track`), `stats.py` (robust-stats primitives), `consent.py` (`is_consented`/`is_denied`), `wellbeing.py` (change-point core + `sanitize_condition_names`/`support_block`) |
| `query_tools/` modules | `messages`, `chunks`, `directory`, `analytics`, `analytics_time`, `analytics_social`, `analytics_language`, `analytics_group`, `analytics_wellbeing` (`get_language_trajectory`), `nudges` (loose-ends read tools), `conflict` (read tools + `diagnose_conflict`) + `conflict_dv` (the single-source `_dv_screen` DV floor) + `conflict_apology` (`draft_apology`), `persona` (`get_voiceprint`) + `persona_draft` (`draft_reply`/`check_style_match`) + `persona_sim` (`simulate_contact`), `drift`; `_shared.py` owns connection/caps/date-bounds/scope, `_REAL_MESSAGE_CLAUSES`, `_restrict_to_direct_chats`, and `member_key` (the single participant-identity merge — never re-implemented); `tool_schemas.py` the JSON schemas |
| Eval | `eval/evalkit/` (analytics regression + F4 eval gates: `cases`, `roster`, `recipes/`, `manifest`, `runner`, `checks`, `p3`, `report`, `gates`, `agreement`, `labeling`), `eval/cases/` (S1–S17) |

New tool = function in the right `query_tools/` module (re-export from
`__init__.py`) + schema in `tool_schemas.py` (same `name`) + one line in
`chat.py`'s `SYSTEM_PROMPT` tool list — keep all three in sync.

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
                                  -- eval_gate:<task> (F4 gate verdicts),
                                  -- nudges_reconnect_ack (durable dormant-tie
                                  -- reconnect opt-in; unset hides those rows),
                                  -- hybrid_seeds (seeds --quality hybrid's own
                                  -- sampler has drawn -- its phase-3 upgrades
                                  -- are source='llm' too, so COUNT(llm) would
                                  -- let the tier satisfy its own seed target),
                                  -- annotate_group_seconds:<phase>:<model>:
                                  -- b<batch>:w<workers> (slowest LLM request
                                  -- seen, so the next budgeted run's first
                                  -- dispatch is not blind), and
                                  -- conflict_turn_seconds:<model> (the same
                                  -- banking for conflict_detect's budgeted
                                  -- labeling: observed seconds per per-turn
                                  -- call, so its deadline arithmetic starts
                                  -- primed instead of guessing)
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
         source, knn_confidence)  -- batch annotations (annotate.py); source
                                  -- 'llm' or 'distilled' (kNN, empty events).
                                  -- knn_confidence 0..1 = how close AND how
                                  -- unanimous that row's kNN neighbours were;
                                  -- NULL on 'llm' rows (an LLM label needs no
                                  -- self-assessment) and on distilled rows
                                  -- written before the column existed; hybrid
                                  -- LLM-upgrades lowest-confidence first
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
         topic, peak_signal_json, corroboration, off_text_flag, source,
         labeled_at)              -- deterministic conflict episodes
                                  -- (conflict_detect.py; 1:1 only); corroboration
                                  -- 'none'|'weak'|'llm' (valence, source='llm'
                                  -- only); source guarded by conflict_model. NO
                                  -- role/aggressor/status column by design (C2).
                                  -- labeled_at (nullable) is the budgeted
                                  -- labeling pass's resume marker: stamped once
                                  -- a window's labels commit -- INCLUDING a
                                  -- window where every label was suppressed, so
                                  -- the presence of conflict_signals rows is NOT
                                  -- the marker -- and preserved (with the labels)
                                  -- when an unchanged window re-detects, so each
                                  -- budgeted run extends coverage
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

- `messages.date` strings are unlabeled UTC ISO: compare as plain strings and
  never wrap the column in SQL date functions inside WHERE (kills
  `idx_messages_chat_date`); `substr`/`strftime` bucket expressions are fine in
  SELECT/GROUP BY over already-filtered rows. Date bounds go through
  `query_tools._normalize_date_bound` (half-open upper bound, tz-aware → UTC);
  chunk date filters and the `first_day`/`last_day` writer both use `db.iso_to_day`.
- Never mix embedding spaces: `chunk_vectors` is dimensioned for — and guarded by
  — the `embed_model`/`embed_dim` in `ingest_state`; `ingest.py` refuses a
  mismatch. Switching `EMBED_MODEL` is always `reset_db.py` + re-ingest. Skipping
  nomic's task prefixes degrades retrieval — set
  `EMBED_DOC_PREFIX`/`EMBED_QUERY_PREFIX` to `''` only for prefix-less models.
- Ollama fits a request by keeping the first system message + the longest suffix
  of whole messages that fits: one tool result bigger than the window silently
  leaves the model with ONLY the system prompt (no user question) and it
  confabulates. Per-row tools cap their row count; `chat.run_turn` caps every
  tool result at `TOOL_RESULT_MAX_CHARS`.
- gemma4's Ollama renderer honors only the FIRST system message; fold
  mid-conversation steering into it. (gemma4 is no longer the default
  `CHAT_MODEL` — this is a gemma4 fact, still load-bearing whenever someone
  points `CHAT_MODEL` at it.) Withholding the tool schemas is the only
  reliable way to stop a tool call — prompting alone doesn't.
- `chunk_annotations` is guarded like `chunk_vectors`: `annotate.py` records
  `annotate_model` and refuses a different model while the table is non-empty
  (distilled rows inherit the seed model's bias, so the guard covers them too) —
  switching `ANNOTATE_MODEL` means emptying the table first. `topics.py`'s
  `topic_model_k` guard is warn-only.
- `chunk_annotations.source='distilled'` (kNN) rows always have `events_json=[]`
  (events can't be distilled — LLM-only), so event-based aggregates undercount
  until an LLM tier with `--upgrade-distilled` has covered the chats in question
  — or a `--quality hybrid` run, whose phase 3a spends its budget on exactly the
  lexicon-flagged event candidates (measured 83.3% precision / 32.1% recall, so
  it raises event coverage, it does not complete it).
- Valence-dependent reads trust ONLY `source='llm'` rows; distilled (kNN) rows
  are never consumed for a high-stakes read. Conflict corroboration, the nudge
  Stage-0 valence exclusion, and the voiceprint tone field all read llm-only and
  degrade honestly (conflict writes `corroboration='none'` and still detects; a
  dyad with no llm chunks is simply not valence-excluded; `tone_*` reads NULL).
  Run `annotate.py --quality balanced --upgrade-distilled` to light them, or
  `--quality hybrid` to light as many as a wall-clock budget buys (it writes
  `source='llm'` too, so the same readers wake up for the chunks it reached).
- The F4 eval gate decides whether a pipeline's high-stakes labels ship.
  `eval_gate_passed(con, task, cls)` resolves clearance by precedence: an owner
  verdict in `ingest_state` under `eval_gate:<task>` (from `eval.evalkit gate`) >
  a committed `eval/seed_gates/<task>.json` seed > nothing. An ABSENT verdict
  means the gated classes are SUPPRESSED (default-closed, Master Spec A2). Only
  `conflict_labels` ships a seed (survives `reset_db` / a fresh DB);
  `dropped_plan` and `wellbeing_narration` stay suppressed until owner-gated.
- An enable-flag gates a feature's tool REGISTRATION or pipeline BUILD, never the
  prompt. `DRIFT_ENABLED=false` (default) makes `drift.py` exit, `dyad_drift`
  never build, and withholds the 3 drift read tools + D3;
  `CONFLICT_TRACE_ENABLED=false` withholds `trace_conflict_across_chats` (its
  per-call `consent` param is a UX device, NOT the boundary);
  `PERSONA_SIM_ENABLED=false` withholds `simulate_contact` (its `acknowledged`
  param is UX; the deceased/grief denylist and the soft dormant warning are
  enforced in the tool's own code). The N4 nudge detectors gate at BUILD too
  (`NUDGE_DROPPED_PLAN_ENABLED`/`NUDGE_GOODNEWS_ENABLED` default off; even ON,
  `dropped_plan` writes nothing until the F4 precision gate clears, and it ships
  no seed). Wrapped third-party content gates at BUILD: `WRAPPED_WELLBEING`
  (default off) decides the wellbeing section; `WRAPPED_VOICEPRINT_CONSENT_IDS`
  (default empty) is default-redacted — no consented ids ⇒ NO third-party
  verbatim/stylometric card. Removing the capability is the control, not prompting.
- The DV floor (Master Spec A10) is SOFTENED and enforced in code
  (`query_tools.conflict_dv._dv_screen`, its single home — imported by both
  `diagnose_conflict` and `draft_apology`), never by prose. `_dv_screen` returns
  a status: `clean` (no marker), `role-suppress` (marker + `CONFLICT_DV_SCREEN`
  on, the default), or `bypassed-with-footer` (marker + screen off via
  `CONFLICT_DV_SCREEN=off` or `chat.py --no-dv-screen`). `role-suppress`:
  `diagnose_conflict` withholds ONLY the demand-withdraw/role read while STILL
  returning its mechanical scorecard; `draft_apology` withholds the ENTIRE
  scaffold (drafting the apology is itself the risky output). The `resource_footer`
  (hotline) is NEVER removed. The scan is read-only; the hit is never persisted.
- Per-contact wellbeing is a code-enforced DENYLIST, never a prompt.
  `get_language_trajectory` analyzes any `contact` by default and refuses only a
  deny-listed one (`WELLBEING_DENY_IDS` via `consent.is_denied`); self is never
  affected. Distinct from the Wrapped voiceprint card's default-closed ALLOWLIST
  (`WRAPPED_VOICEPRINT_CONSENT_IDS` via `consent.is_consented`), which only ADDS a
  contact and never overrides a roster/silent-contact exclusion. `consent.py`
  ships both helpers; the caller picks allow vs deny.
- Wellbeing narration is framed by two code mechanisms in `chat.run_turn`, not
  the prompt: (a) a whole-reply sanitize — streaming is held and the ENTIRE
  assembled reply runs through `wellbeing.sanitize_condition_names` before it is
  printed/returned; and (b) it collects a tool result's reserved `_epilogue` (the
  code-emitted `support_block`), strips both `_epilogue` AND `support_block` from
  the data the model sees, and prints the epilogue verbatim after the reply (the
  envelope also carries a model-safe `flagged` bool). `wrapped.py` runs the same
  scrub. The apple chat path bypasses `run_turn`, so `chat_apple.py` re-enforces
  this SAME floor itself — it is not inherited.
- `think` is per-call tri-state: `None` means OMIT the flag (some models reject a
  `think` option) and is NOT the same as `False` (send `think: false`), so never
  default the param to `None`. The flag dominates a thinking model (qwen3:4b: 3.2s
  with `think=false` vs 85s on). An effort level is not portable: `think='low'` is
  an effort level only to models that have them (gpt-oss); to gemma4 it is just
  truthy, buying a full reasoning pass (282.6s vs 26.2s per dyad term, same
  verdict). `config.has_effort_levels` (family list `_EFFORT_LEVEL_MODELS`,
  currently just `gpt-oss`; the effort family is listed because it is small and
  named, while "every model that only toggles" is unbounded) is the single home
  for that, and `annotate._resolve_quality(quality, model)` applies it:
  a preset's effort STRING resolves to `False` on a non-effort model, so
  `distilled`/`fast`/`balanced` send `think=False` on qwen3.5:4b/gemma4 and
  `'low'` on gpt-oss — without it `--quality fast` would be the SLOWEST tier on
  the default model. `None` and explicit `True`/`False` pass through untouched
  (the tri-state survives) and an explicit `ANNOTATE_THINK` always wins.
  `conflict_detect` carves itself out the same way: an unset `CONFLICT_THINK`
  resolves to `False` (`'low'` on gpt-oss) for that pipeline instead of
  inheriting `CHAT_THINK`'s `None`, because omitting the flag made every
  per-turn labeling call a full reasoning pass (197 s vs 2.2 s measured on
  qwen3.5:4b); an explicit `CONFLICT_THINK` still wins and passes through
  untouched. `CHAT_THINK` deliberately still defaults to `None`, so on the
  thinking-capable default chat model thinking is ON for `chat.py` — a
  speed/quality call for the user (routing across ~32 tools may genuinely want
  it), remedied with `CHAT_THINK=false`. Never route one pipeline's `think` into
  another — `--terms` owns `ANNOTATE_TERMS_THINK` and rejects `--quality`.
- `TOPIC_LABEL_SAMPLES` defaults to 30: label samples are ranked by similarity to
  the cluster centroid, and a big cluster's most central chunks are its shortest
  ("ok", "yeah"); raising it past ~30 risks the context window (an oversized
  sample is dropped, not rejected).
- `chat.db` is live (WAL sidecars): only ever read the snapshot
  (`ingest.snapshot_source`), never the original. Ingestion is append-only;
  `--senders` scopes what a run reads, never deletes — the one carve-out is
  `chunks`/`chunk_vectors` (derived: `--since-last` rebuilds a chat's tail chunk
  without touching `messages`/`chats`/`handles`). DB paths resolve relative to
  `config.py`'s directory, not the CWD (and `load_dotenv` is pinned to that
  directory's `.env`, so a worktree never inherits a parent checkout's `.env`).
- Apple FM (the `apple` sentinel) has a fixed 4,096-token window, so the chat
  path can't do native tool-calling; `chat_apple.py` stages small
  route→fill→exec→digest→answer sessions and carries a Python scratchpad
  compacted under a char budget after every tool, and every stage catches
  `applefm.ContextOverflowError` and degrades. Apple `format="json"` is NOT a
  guided schema (an empty-`properties` `DynamicGenerationSchema` pins the reply to
  `{}`), so `applefm.chat_once`'s JSON mode folds a directive in and generates
  free text (`chat_once_json` parses + nudge-retries); a caller needing
  constrained decoding calls `applefm.respond(..., schema=...)`. Apple `think` is
  ignored (accepted and dropped at the seam); the one speed lever FM exposes is
  `maximumResponseTokens` — `applefm.respond` takes `max_tokens`, and
  `chat_apple.py` caps its route/fill/digest stages via
  `APPLE_ROUTE_MAX_TOKENS`/`APPLE_FILL_MAX_TOKENS`/`APPLE_DIGEST_MAX_TOKENS` while
  leaving ANSWER uncapped (`APPLE_ANSWER_MAX_TOKENS=0`) so the reply is never
  clipped. The persistent shim `prewarm()`s the model at startup (gated on
  availability) so the first `respond` skips cold-load, amortized across the
  session. The `annotate_model` and `embed_model`/`embed_dim` guards apply to
  apple exactly as to Ollama. Batch `--terms` is unclamped on apple — set a small
  `ANNOTATE_TERMS_BATCH_SIZE`. Batch ANNOTATION is clamped instead: on apple
  `annotate._annotation_groups` ignores `batch_size` and packs by
  `_APPLE_ANNOTATE_CHAR_BUDGET` chars, now 2,000 (one `CHUNK_MAX_CHARS`, down
  from 12,000) — a GLOBAL change, so `fast`/`balanced`/`max` on apple pack ~6x
  smaller and measured 1.7–3.5x faster per chunk, because a batched apple
  request usually fails whatever its size and each failure costs a single-chunk
  retry per id.
- Don't modify or read files in `tests/` unless the task is about tests.

## Conventions

- Tool functions: id params accept a scalar or a list (`_as_id_list`); message
  `text` is truncated to `TEXT_PREVIEW_CHARS` (chunk previews to
  `CHUNK_PREVIEW_CHARS`), recoverable via `get_full_messages`/`get_conversation`.
  `search_messages_sql`/`search_messages_fts` return an envelope
  (`rows`/`total_matching`/`returned`/`order`/`truncated`), not a bare list.
- Participant-identity merging (me / contact / handle / unknown) has one home:
  `query_tools/_shared.member_key`. Modules keep thin named wrappers for their
  contracts (`_person_key` → `None` for no-identity rows, others → an `"unknown"`
  bucket) — never re-implement the key logic.
- All tunables (models, hosts, timeouts, batch sizes, chunking, and the Wave 0–4
  `CONFLICT_*`/`NUDGE_*`/`DRIFT_*`/`VOICEPRINT_*`/`PERSONA_SIM_*`/`WELLBEING_*`/
  `WRAPPED_*` families) live in `config.py` / `.env.example`; never hardcode them
  elsewhere. Runtime deps → `requirements.txt`, dev-only → `requirements-dev.txt`
  (both pinned).
- Audit artifacts (`REPO_REVIEW.md`, `REPO_FIX_PLAN.md`) are gitignored working
  files — never commit them.
