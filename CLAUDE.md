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
  sqlite loadable-extension support sqlite-vec requires; `PYTHON=/path` overrides
  the default `/opt/homebrew/bin/python3`, and install probes the interpreter for
  loadable-extension support and refuses one built without it). `make bootstrap`
  = install → install-ollama → source-data → ingest → doctor; every step is
  idempotent, so a failed bootstrap resumes rather than redoes. Bare `make` =
  `make help`, self-documented from each target's `## ` line so it cannot drift.
  `make doctor` is the diagnostic (venv/deps/brew/.env, Full Disk Access, Ollama
  + every model `config.py` names, source and analysis row counts — vectors
  counted THROUGH the venv, since the system `sqlite3` can't open a vec0 table
  and would report 0). `make source-data` copies `chat.db` (+ WAL sidecars) and
  the AddressBook into `source_data/`, refusing to overwrite; `make env`
  scaffolds `.env` from `.env.example`.
- Ollama + models: `make install-ollama` (brew — NOT the curl installer, which
  drops an `/Applications/Ollama.app` `brew services` can't manage — then pulls
  every model `config.py` names). `make ollama-start` RESTARTS the service with
  `OLLAMA_NUM_PARALLEL=4` (the slots `ANNOTATE_WORKERS` feeds; restart, not
  start, because `brew services start` no-ops against a running service and
  would silently leave zero parallel slots), `make ollama-stop` stops it. Apple
  Foundation Models backend (no Ollama):
  `make apple-shim`, then point any `*_MODEL` var at the `apple` sentinel
  (`CHAT_MODEL`/`EMBED_MODEL`/`ANNOTATE_MODEL`/…). Needs macOS 26+, Apple
  Silicon, Apple Intelligence. Both kept out of `make install` (slow / network).
- Ingest: `make ingest` / `python3 ingest.py [--no-embed] [--since-last]
  [--senders/-s=...] [--exclude/-x=...] [--lookback=PERIOD]` (`--lookback`/`-l`
  takes a
  period like `7d`/`3mo`/`2y` — units d/w/mo/y, calendar-correct — adding a
  `message.date >= now-PERIOD` lower bound; combines with the other flags, drops
  dateless/system rows for that run). Through make: `make ingest
  INGEST_ARGS="--lookback 3mo"`.
- Chat: `python3 chat.py [-m MODEL] [--no-dv-screen] [-p "..."]` — bare is the
  REPL (`/clear` resets the conversation), `-p` is one-shot (prints only the
  reply, exit 1 if Ollama is unreachable). `-m`/`--model` overrides `CHAT_MODEL`
  for the run; `--no-dv-screen` transiently overrides `CONFLICT_DV_SCREEN` (never
  persisted; the resource footer still appends — see the DV gotcha). `make chat
  ARGS='-p "..."'` passes through and preflights Ollama first.
- Annotate chunks: `make annotate` / `make annotate-all` / `make annotate-hybrid`,
  or `python3 annotate.py [--all | --top-dyads N] [--limit N] [--model NAME]
  [--quality distilled|hybrid|fast|balanced|max] [--upgrade-distilled]
  [--dry-run]` — resumable, Ctrl-C safe. `make annotate`/`make annotate-all`
  each run TWO commands (the chunk pass, then `annotate.py --terms`); `make
  annotate-hybrid` deliberately does NOT chain `--terms`, whose cost is
  unbudgeted and would break the wall-clock promise the target exists for.
  Default `distilled` = LLM-labels a
  ~`ANNOTATE_SEED_TARGET` seed then kNN-classifies the whole corpus from
  `chunk_vectors` (fast; events left empty); it is corpus-wide by construction,
  so `--all`/`--top-dyads N`/`--limit N` are accepted and SILENTLY IGNORED there
  (not an error — though a NEGATIVE N is still refused by the parser, which runs
  before any tier does). `hybrid` = a stratified
  `ANNOTATE_HYBRID_SEED_TARGET` seed draw + that same kNN pass (both corpus-wide),
  then LLM upgrades — event-candidate chunks first (ranked by pattern-hit count
  DESC, chunk id ASC, so the prefix a budget affords is the DENSEST candidates,
  not the oldest; reruns still advance because an upgraded row is written
  `source='llm'` and leaves the candidate set), lowest-`knn_confidence`
  next — until the `ANNOTATE_BUDGET_MINUTES` wall-clock deadline; budgeted in
  TIME not chunks, so it self-terminates and a rerun resumes, the scope flags
  (`--all`/`--top-dyads N`) and `--limit N` reach only the upgrade phase, and
  `--upgrade-distilled` is an error with it
  (hybrid IS one). Under hybrid `--limit N` caps the upgrades WRITTEN across
  phases 3a+3b — successes, not dispatches: a reply the model came back
  unparseable on spent budget but produced no annotation and no longer consumes
  the cap. The wall clock is the SPEND cap, `--limit` the DELIVERABLE cap — an
  "at most N", with one deliberate residual: an unparseable reply is made up out
  of the OTHER sub-phase's candidates, so when those run out (3b's pool empty, or
  3a's ranked list exhausted) the run lands short of N. Making it up regardless
  means dispatching past N, which is spend the flag did not authorize; the
  closing summary says how many candidates were left rather than claiming it
  "finished every candidate in scope". A NEGATIVE `--limit` (or `--top-dyads`)
  is refused by `annotate.py`'s PARSER, for every tier and `--terms` alike:
  negative never means a smaller run anywhere — on the per-chunk tiers and
  `--terms` it reaches a SQL `LIMIT ?` where `-1` means NO limit, and in hybrid
  it inverted both prefixes (`[:-1]` drops the LEAST dense candidate and
  `max(0, limit - done)` then switched 3b off). `hybrid_chunks` keeps its own
  `SystemExit` for callers reaching it directly, and `ANNOTATE_TOP_DYADS` gets
  the same check at all three places it is RESOLVED (`main`, `hybrid_chunks`,
  `annotate_chunks`) — a knob is validated where it is resolved, not only where
  a flag supplies it — but ONLY on runs that bind it: `--terms`, `--all` and
  `distilled` never reach that `LIMIT`, and refusing them would break bare `make
  annotate` over a value they don't read (the `CONFLICT_MAX_LABEL_TURNS` /
  `--no-llm` rule, both halves). `--top-dyads 0` means ZERO chats, not the
  default: `main` resolved it with `or` and so read an explicit 0 as "20", where
  `hybrid_chunks` had always read 0 as 0. EVERY
  phase is deadline-gated, including the deterministic
  kNN pass (O(rows × seeds): 14.6 s at a 450-seed index vs 82.0 s at 4,416,
  which `ANNOTATE_KNN_INDEX_CAP` now bounds — the index, never the pool; and
  WHICH seeds it keeps is drawn proportionally to each chat's share of the
  CORPUS, never uniformly over the pool, because the pool is only whatever
  earlier runs happened to label: measured, all 4,416 llm rows on the
  reference corpus sat in ONE chat, so a uniform draw gave the other 61.8% of
  the corpus a vote from a dyad it has nothing to do with. It cannot fix a
  genuinely single-chat pool — widening the pool is the seed sampler's job;
  this stops the cap re-narrowing what the sampler widened) and
  seeding (also capped at 25% of the budget — unbatched and uncapped it
  measured 10.7 min of an 8-min run). Phase 3a truncates its ranked candidate
  list to `annotate_hybrid._EVENT_FETCH_CAP` (2,000 — a module constant, not a
  knob) BEFORE materializing chunk text, `_LOW_CONF_FETCH_CAP`'s memory argument
  applied to a densest-first ranking so the prefix is what the deadline would
  have bought anyway at the DEFAULT budget (past ~111 budgeted minutes the cap,
  not the clock, is what stops 3a — hence the printed line). It prints the TRUE
  candidate count (measured before either prefix), how many it left, and which
  prefix left them; a `--dry-run` says "would fetch", since that path fetches
  nothing.
  3a also re-reads the clock AFTER the corpus-wide event prefilter and the row
  fetch, so the run's one unavoidable full `messages` scan is not charged to the
  phase's own share. Each phase starts from the per-request
  time an earlier run banked in `ingest_state` (banked in a `finally`, so an
  Ollama timeout mid-seed-pass keeps the committed rows AND the `hybrid_seeds`
  counter in sync instead of letting the next run over-buy seeds at the upgrade
  phases' expense), so nothing is dispatched blind;
  the floor is that ONE request is the smallest dispatchable unit (a 10k-char
  group holds 4-6 event chunks and banked at 46.7 s), so a budget too small for
  one buys NO UPGRADES and says so, naming the knob — which is not the same as
  doing nothing: the deterministic phases are deadline-gated too but still run
  and COMMIT what they reach, and the next run resumes. A NEGATIVE
  `ANNOTATE_BUDGET_MINUTES` (or `budget_minutes` argument) is a `SystemExit`
  naming the knob, not an already-passed deadline that silently declines every
  phase; `0`/empty still means "unset" → the per-backend default (25 min Ollama,
  50 apple). The `annotate_model` guard runs at the TOP of a hybrid run, before
  the budget clock starts — it used to fire only at phase 3a, so a pool already
  at seed target let a switched `ANNOTATE_MODEL` burn both corpus-wide kNN
  phases before being refused; `--dry-run` is the one path that does NOT guard
  (the guard RECORDS the model, and a preview that writes is not a preview). Its
  seed
  target counts a `hybrid_seeds` counter, NOT `source='llm'` rows — the tier's
  own upgrades are `llm` rows and counting them switched its sampler off
  permanently. LLM tiers (`fast`/`balanced`/`max`) scope to top
  `ANNOTATE_TOP_DYADS`, honor `--all`, and with `--upgrade-distilled`
  re-annotate kNN rows (how events get filled). A batched request that comes
  back missing ids buys ONE schema-constrained retry (`format=` a JSON Schema
  over exactly the missed ids) before per-chunk singles — every non-apple tier;
  apple skips that rung, its `format` seam is not a guided schema. `--terms` is
  a SEPARATE pipeline (below), not a `--quality` tier.
- Curate dyad terms: `python3 annotate.py --terms [--limit N] [--model NAME]
  [--retry-failed] [--dry-run]` — its own pipeline; `--quality` is an error, and
  `--limit N` is a plain SQL row cap here. Batches
  `ANNOTATE_TERMS_BATCH_SIZE` terms/request (never spanning chats), runs
  `ANNOTATE_TERMS_THINK` (default false: a one-word verdict measured ~11x latency
  with reasoning on). A term failing `ANNOTATE_TERMS_MAX_FAILURES` times is parked
  in `dyad_terms.curate_failures`; `--retry-failed` unparks it.
- Topics: `make topics` / `python3 topics.py [--k K] [--relabel] [--model M]` —
  k-means over chunk_vectors (assignment in sqlite-vec) + an LLM label pass.
- Analytical suite (each a standalone CLI + `make` target, all read
  `chunk_annotations` so they run after `annotate`; `make suite` = all four in
  order, and is exactly what `make pipeline` runs between annotate and topics):
  `make nudges` (`loose_ends.py [--model M] [--limit N] [--no-llm] [--dry-run]
  [--acknowledge-reconnect]` → `loose_threads`; Ollama only for the one-line
  `nudge_text` narration, so `--no-llm` is SQL detection only and `--limit N`
  caps narrated threads, not detected ones. `--acknowledge-reconnect` is a pure
  toggle: it sets the durable `nudges_reconnect_ack` key so `get_loose_threads`
  stops hiding `dormant_tie` rows, then exits WITHOUT detecting. `--dry-run`
  prints the scope and writes nothing, skipping the LLM-backed `dropped_plan`
  detector), `make drift` (`drift.py [--force] [--dry-run] [--purge]` →
  `dyad_drift`; LLM-free, opt-in `DRIFT_ENABLED`, default off. Full rebuild,
  skip-if-unchanged, so `--force` rebuilds anyway and `--dry-run` reports how
  many dyads WOULD be scored; `make drift-purge` = `--purge`, which tears the
  table + its four guard keys down REGARDLESS of `DRIFT_ENABLED` — it is what
  you run WHILE turning the feature off, so gating it would make it unreachable
  in exactly the state it exists for), `make voiceprint`
  (`voiceprint.py [--force] [--dry-run]` → `person_voiceprint`; LLM-free, full
  rebuild guarded by `voiceprint_max_message_id` AND
  `voiceprint_annotations_hash` — the second because an annotate pass writes
  `source='llm'` rows with NO new messages, so on the message-id guard alone a
  run over an OTHERWISE-UNCHANGED corpus — the resumable `--quality hybrid`
  re-run, or a bare `make annotate && make voiceprint` — reported "up to date"
  and kept the stale `tone_*`. A `make pipeline` whose ingest DID add a message
  rebuilt anyway (voiceprint runs after annotate), which is why the bug reads as
  intermittent. A SESSION_GAP_MINUTES retune, a lexicon rebuild, a
  `VOICEPRINT_*` change and an AddressBook re-key are still NOT covered — those
  need `--force`),
  `make conflict`
  (`conflict_detect.py [--budget-minutes N] [--label-top N | --no-llm]
  [--max-label-turns N] [--relabel] [--limit N] [--model M] [--dry-run]` →
  `conflicts`/`conflict_signals`; Ollama only for the per-turn labels;
  `--label-top`/`--no-llm` are mutually exclusive, and `make conflict-fast` =
  `--no-llm`). A bare
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
  work to budget). A NEGATIVE value — from the flag, from
  `CONFLICT_BUDGET_MINUTES`, or from a direct caller — is a `SystemExit` naming
  the knob, never a clamp: clamping sent it to 0 = unbudgeted, the exact
  unbounded run the CLI already refuses `--budget-minutes -5` for.
  `ANNOTATE_BUDGET_MINUTES` refuses negatives the same way; only the meaning of
  `0` differs ("unbudgeted" here, "unset" there). Under `--no-llm` the budget is
  not resolved AT ALL (that path makes zero LLM calls, so a bad
  `CONFLICT_BUDGET_MINUTES` must not refuse a run it cannot affect).
  `CONFLICT_MAX_LABEL_TURNS` gets the same env-side check for the same reason
  the flag has one: `int(getenv(...) or "6")` reads `"0"` as 0, and a
  non-positive cap selects no turns while every window is still stamped
  processed — one .env line latching the whole corpus. `--relabel` clears every
  window's labels (`conflict_signals` + `topic` + the `labeled_at` stamp) so the
  pass starts over — needed when what labeling PRODUCES changed (a retuned
  `CONFLICT_MAX_LABEL_TURNS`/`CONFLICT_MIN_TURN_CHARS`, a newly cleared eval
  gate), since a stamped window is otherwise never revisited. The
  `conflict_model` guard runs BEFORE anything that can destroy a label — the
  `--relabel` deletion AND the detection sweep, whose `_drop_conflicts` deletes a
  superseded window's `conflict_signals` with the window — so both
  `--relabel --model <other>` and a plain `--model <other>` are REFUSED with every
  label intact, where they used to destroy them and only then refuse to rebuild
  (worse on the plain path: if the drop emptied `conflicts`, the guard's
  `COUNT(*)` read 0 and it silently ACCEPTED the new model). `--relabel` is NOT
  the model-switch door (that is clearing `conflicts` or `reset_db.py`, exactly
  what the guard's message says), and `--no-llm` does not exempt it: that flag
  skips the labeling, not the deletion. The early call REFUSES only —
  `conflict_model` is recorded by the labeling pass, so a run that labels nothing
  (interrupted detection, `--no-llm`) leaves the key untouched instead of
  refusing the next run in the name of a model that never wrote a label.
  `--dry-run` runs neither the guard nor the clear. Turn selection is
  peak-anchored on
  `peak_gap_message_id` (not the first N chronological turns) and the ONE hard
  prescreen is code, not a knob: a turn with no alphanumeric character at all
  (tapback/emoji/punctuation) gives the model nothing to read.
  `CONFLICT_MIN_TURN_CHARS` (default 15) is a RANKING preference, used only in
  the no-anchor fallback — floor-clearing turns take slots first, shorter ones
  fill the rest, and the chosen set is re-sorted CHRONOLOGICALLY (the labels
  read as a sequence on every path). It demotes, it never deletes: as a filter
  it deleted exactly the class `_TURN_SYSTEM` asks for — stonewalling IS
  "terse/absent non-answers", short contempt is "k.", "fine.", "whatever" — so a
  window whose escalation was entirely terse withdrawal dispatched nothing, was
  stamped `labeled_at` anyway, and read downstream as analyzed and clean. Those
  windows now cost requests, so windows-labeled-per-run can fall (honestly). The
  floor is deliberately NOT applied inside the peak-anchored span: a terse turn
  next to the escalation moment IS the stonewalling evidence, so preferring
  length there would re-delete the class the change exists to catch — the price
  is that a long turn further from the peak can now lose its slot to a terse
  neighbour. `CONFLICT_MAX_LABEL_TURNS` still defaults to 6, not 12, for the
  peak-anchoring reason. Per-call seconds bank in
  `ingest_state` under `conflict_turn_seconds:<model>` so the deadline
  arithmetic is never blind, and an unset `CONFLICT_THINK` resolves to `False`
  for this pipeline (197 s → 2.2 s per call — see the `think` gotcha).
- Everything, in order: `make pipeline`
  (ingest → annotate → `make suite` [nudges → drift → voiceprint → conflict] →
  topics). MUST
  stay serial (written to under `make -j` too): they hold a SQLite write lock
  across many LLM calls (`database is locked`) and share one serialized Ollama —
  which is why `pipeline`/`suite`/`bootstrap`/`eval` are `$(MAKE)` recipe lines,
  not prerequisites.
- Flags through `make`: only `make ingest` (`INGEST_ARGS=...`), `make chat` and
  `make wrapped` (`ARGS='...'`) forward arguments. `make annotate`,
  `annotate-all`, `annotate-hybrid`, `topics`, `nudges`, `drift`, `voiceprint`
  and `conflict` take NO passthrough — invoke the CLI directly for a flag. The
  Ollama preflight gates every target that makes LLM calls (`ingest`,
  `annotate*`, `topics`, `nudges`, `conflict`, `chat`, `wrapped`) and
  deliberately not the LLM-free ones (`drift`, `drift-purge`, `voiceprint`,
  `conflict-fast`).
- Wrapped report: `python3 wrapped.py [--year YYYY] [--contact NAME]
  [--exclude SPEC]... [--include NAME]... [--out PATH] [--no-llm]
  [--yes] [--wellbeing]` — a self-contained HTML page (inline CSS+JS, no network,
  light/dark, scroll-reveal); previews the roster (with silent-contact auto-
  exclusions) first. `--wellbeing`/`WRAPPED_WELLBEING` opts the trajectory
  section in (off by default even when a window fires). `--exclude` is repeatable
  and takes a `chat_id`, an ISO range `A..B`, or a contact name; `--include NAME`
  re-admits a contact the silent-contact filter auto-excluded; `--yes` skips the
  roster confirmation; `--out` overrides the default filename.
- Optional web layer (`api/` FastAPI + `frontend/` Vite/React over the same
  `imessage_analysis.db`): `make api-install` (installs `requirements-api.txt`
  into `.venv` — deliberately off the `make install` path), `make api-dev`
  (uvicorn `api.main:app --reload`; `API_HOST`/`API_PORT` override, loopback by
  default, `API_CORS_ORIGINS` allows the Vite dev origin), `make web-install` /
  `web-dev` / `web-build` (`npm --prefix frontend`). The API is TRANSPORT only:
  it persists the UI's conversations in `app.db` and streams
  `chat.run_turn_events` over SSE — retrieval and the wellbeing floor are never
  re-implemented there. Endpoints + SSE frames: `api/README.md`.
- Eval: `eval/fast_eval.sh eval/prompts-smoke.txt [-m model]` (older smoke path,
  `id|prompt` lines through `chat.py -p`, sequential — never parallel). Analytics
  regression (evalkit): `python3 -m eval.evalkit <selfcheck|manifest|run|check|
  report|p3>` sweeping the S1–S17 suites (`eval/cases/`) through `chat.run_turn`
  in-process against SQL ground truth (`run` needs Ollama, sequential). `p3` is
  the generator-eval lane (`p3 wrapped` / `check-wrapped` / `annotate-sample` /
  `topics-sample`; only `p3 wrapped` and `--judge` touch Ollama). `make eval`
  chains selfcheck → manifest → `run --run-id $(EVAL_RUN_ID) --all` → check →
  report into `eval/results/<EVAL_RUN_ID>` (default a UTC timestamp +
  `-baseline`, overridable to re-check/re-report an existing run); `make
  eval-dry` is the Ollama-free subset (selfcheck, manifest, `run --dry-run
  --all`). F4
  eval-gate labeling (owner-only): `python3 -m eval.evalkit label <task>` /
  `... gate <task>` — hand-labels into the gitignored `eval/labels/<task>.jsonl`,
  runs the classifier, persists the pass/fail to `ingest_state`. Tasks:
  `conflict_labels` (kappa), `dropped_plan` (precision), `wellbeing_narration`.
  See `eval/README.md`.
- Test: `python3 -m unittest discover tests` (hermetic — no Ollama/real
  `chat.db`). NEVER run unless you touched `tests/` or were asked — slow, hangs
  the terminal; prefer one module (`-m unittest tests.test_foo`). `make test` IS
  that discover run and `make check` = lint + test + `compileall`, so BOTH
  inherit the warning — run them only when asked or before a push. Lint: `make
  lint` (ruff, line-length 100, E+F). Reset analysis
  DB (keeps schema): `make reset-db`, i.e. `reset_db.py -f` — the bare CLI blocks
  on a `[y/N]`, `-f`/`--force` is the scripted path. `nudge_exclusions` and the
  committed `eval/seed_gates/conflict_labels.json` survive it by design.

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
| Shared libs | `config.py` (all env settings), `db.py` (sqlite helpers, `load_vec_extension`, `iso_to_day`, `get_state`/`set_state`, `connect_default`/`run_cli`), `ollama.py` (+`applefm.py`), `embeddings.py` (doc/query prefixes, dim detect), `progress.py` (`track`), `stats.py` (robust-stats primitives), `consent.py` (`is_consented`/`is_denied`), `wellbeing.py` (change-point core + `sanitize_condition_names`/`support_block`), `chat_floor.py` (the single-sourced wellbeing floor — arm/strip/dedupe/fold, used by all three turn-runners), `nudge_shared.py` (shared dyad primitives + the detector registry; exists to break the `loose_ends` ↔ `nudge_detectors` import cycle) |
| Optional web layer | `api/` (`main` app assembly + CORS + static mount, `chat_stream` SSE, `conversations` CRUD, `store` app.db persistence, `models`) over `chat.run_turn_events`; `frontend/` (Vite + React, gitignored build) |
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
                                  -- voiceprint_max_message_id +
                                  -- voiceprint_annotations_hash (the second
                                  -- fingerprints the source='llm' rows tone_*
                                  -- is built from, so an annotate pass -- which
                                  -- adds NO messages -- still rebuilds; without
                                  -- it a --quality hybrid run over an
                                  -- otherwise-unchanged corpus left the stored
                                  -- tone stale behind an "up to date" line (a
                                  -- pipeline run that DID ingest rebuilt
                                  -- anyway, so the bug read as intermittent).
                                  -- Deliberately NOT drift's shape: no
                                  -- is_group filter, since scope group/all read
                                  -- group chats, and an exact per-row digest,
                                  -- since emotions_json is TEXT no SUM can
                                  -- fingerprint),
                                  -- eval_gate:<task> (F4 gate verdicts),
                                  -- nudges_reconnect_ack (durable dormant-tie
                                  -- reconnect opt-in; unset hides those rows),
                                  -- hybrid_seeds (seeds --quality hybrid's own
                                  -- sampler has drawn -- its phase-3 upgrades
                                  -- are source='llm' too, so COUNT(llm) would
                                  -- let the tier satisfy its own seed target;
                                  -- written with the committed rows in a
                                  -- `finally`, so an ollama timeout mid-seed-
                                  -- pass cannot leave paid-for rows uncounted
                                  -- and make the next run over-buy past
                                  -- ANNOTATE_HYBRID_SEED_TARGET at the upgrade
                                  -- phases' expense),
                                  -- annotate_group_seconds:<phase>:<model>:
                                  -- b<batch>:w<workers> -- the slowest GROUP
                                  -- seen (one batched request plus whatever
                                  -- schema-retry / single-chunk rungs it
                                  -- needed), so the next budgeted run's first
                                  -- dispatch is not blind. Only a MEASURED
                                  -- value is ever written; <phase> is one of
                                  -- the hybrid tier's three (seed | event |
                                  -- lowconf), estimated separately because the
                                  -- populations differ (~1,900-char event
                                  -- candidates pack 4-6 to a group, ~360-char
                                  -- confidence rows fill a batch of 8), and
                                  -- conflict_turn_seconds:<model> (the same
                                  -- idea for conflict_detect's budgeted
                                  -- labeling, but the running MEAN of per-turn
                                  -- call seconds, not the worst: the first
                                  -- completed call is excluded -- it carries
                                  -- the model cold load -- and a call that
                                  -- RAISED is never recorded, or one ollama
                                  -- timeout would bank ~CONFLICT_TIMEOUT and
                                  -- latch, since a run that then declines
                                  -- every window measures nothing that could
                                  -- overwrite it)
sessions(id, chat_id, first_message_id, last_message_id, first_date,
         last_date, n_messages, n_from_me, initiator_handle_id,
         initiator_is_me, ender_handle_id, ender_is_me)
                                  -- gap-split conversation sessions (derived,
                                  -- rebuilt per chat; handle NULL = me)
person_period_lexicon(chat_id, handle_id, month, n_messages, n_words,
         avg_chars, ...9 fw_*_rate cols...,
         we_count, question_count, laugh_count, emoji_json,
         i_count, you_count, absolutist_count)
                                  -- per-person-per-month style stats
                                  -- (derived, full rebuild; handle NULL = me).
                                  -- i/you/absolutist are raw PRONOUN_I /
                                  -- PRONOUN_YOU / ABSOLUTIST token counts (F1),
                                  -- nullable until the next full rebuild fills
                                  -- them; downstream sums the raw counts and
                                  -- forms rates against summed n_words, never
                                  -- stored pre-divided
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
                                  -- status 'open' | 'dismissed' (user mute) |
                                  -- 'resolved' (auto-closed by the detector's
                                  -- reconciliation pass when the thread is no
                                  -- longer found -- hidden from both digests,
                                  -- re-openable); 'snoozed' is RESERVED with no
                                  -- writer (N5 owns it). No status ever deletes
                                  -- a row; the LLM columns are
                                  -- guarded by loose_ends_model. due_by is filled
                                  -- by the inferred_occasion detector (next
                                  -- upcoming date) and NULL for other kinds.
                                  -- reset_db clears it
nudge_exclusions(handle_id, contact_id, reason, created_at)
                                  -- contacts/handles muted from nudges; durable
                                  -- user intent, PRESERVED across reset_db.
                                  -- UNIQUE(handle_id, contact_id, reason) keeps
                                  -- duplicate mutes from stacking, but NULLs
                                  -- compare DISTINCT in a UNIQUE constraint, so
                                  -- every writer ALSO does the NULL-safe
                                  -- existence check (query_tools/nudges.
                                  -- _add_exclusion) before inserting
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
                                  -- voiceprint_max_message_id +
                                  -- voiceprint_annotations_hash (an annotate
                                  -- pass adds no messages, so the message-id
                                  -- key alone could never see the tone_* inputs
                                  -- change)
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
                                  -- budgeted run extends coverage. Read side: a
                                  -- STAMPED window with ZERO conflict_signals
                                  -- rows gets its own wording (query_tools.
                                  -- conflict._LABELED_EMPTY_NOTE /
                                  -- _LABELED_EMPTY_SCOPE_NOTE), not the
                                  -- eval-gated-bands caveat -- labeling reads
                                  -- message TEXT only, so a stretch carried by
                                  -- tapbacks, by tone, or moved off-channel can
                                  -- be looked at and yield nothing, and it is
                                  -- never narrated as "analyzed and calm". On a
                                  -- DB whose conflicts table PREDATES the
                                  -- column, `labeled` is INFERRED per row from
                                  -- whether signal rows exist, so a signal-less
                                  -- window from an old --no-llm run reads
                                  -- labeled=false where the old flat
                                  -- 'pre-budget' literal read true; the
                                  -- inference is one-way-safe (signals cannot
                                  -- exist without a pass, their absence stays
                                  -- ambiguous) and _PRE_MIGRATION_NOTE says so,
                                  -- with a rerun stamping properly
conflict_signals(id, conflict_id, message_id, actor_is_me, label_class,
         top2_labels_json, band, excerpt)
                                  -- per-turn escalation labels for a conflict;
                                  -- band 'tentative'|'likely' (never a raw score),
                                  -- only for eval-gate-cleared classes; excerpt
                                  -- verbatim <= 200 chars
messages_fts                      -- FTS5 external-content table over messages.text
chunk_vectors                     -- sqlite-vec vec0 table: chunk_id (PK) ->
                                  -- embedding FLOAT[detected dim]
                                  -- distance_metric=cosine, + chat_id/
                                  -- first_day/last_day METADATA columns
```

Transient/legacy, never read by a tool: `_topic_centroids` (topics.py, a vec0
scratch table dropped and recreated each k-means iteration), `temp.seed_vectors`
(`annotate_distill`'s kNN seed index, temp schema), and `message_vectors` (the
pre-chunking per-message table — no writer remains; `reset_db.DROP_TABLES` still
lists it so an old DB gets cleaned up).

## Schema (app.db — web UI only, `config.APP_DB`, gitignored)

```
conversations(id, title, created_at, updated_at)
turns(id, conversation_id, role, content, created_at)
                                  -- conversation_id -> conversations.id
                                  -- ON DELETE CASCADE; api/store._connect sets
                                  -- PRAGMA foreign_keys = ON per connection
                                  -- (off by default in sqlite3).
                                  -- idx_turns_conv ON turns(conversation_id)
```

Plain sqlite3, no sqlite-vec, its own connection helper — deliberately apart
from the analysis DB (`db.py`). Holds only the UI's chat history; `reset_db.py`
does not touch it. Timestamps are ISO-8601 UTC strings.


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
  switching `ANNOTATE_MODEL` means emptying the table first. The hybrid tier
  runs that guard at the TOP of its run, before the budget clock, never mid-run,
  so a switched model costs zero budget instead of being refused after two
  corpus-wide kNN phases (the phase-3a call is kept as a local invariant for a
  future caller reaching phase 3 directly); `--dry-run` is the one path that
  does NOT guard, because `_check_model_guard` is not a pure check — it
  `set_state`s `annotate_model` and commits, and a preview that writes is not a
  preview. `topics.py`'s `topic_model_k` guard is warn-only.
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
- `find_conflicts`/`get_conflict` separate THREE states, not two, and only the
  third is a positive finding: NEVER LOOKED AT (`labeled` false — the budgeted
  pass has not reached this window), LOOKED AT WITH NOTHING SHIPPABLE (`labeled`
  true, `labeled_signal_turns` 0 — labeling reads message TEXT only, so a
  stretch carried by tapbacks, by tone, or moved off-channel yields nothing, and
  a turn can also fall under `CONFLICT_LABEL_CONF_FLOOR` or be F4-suppressed),
  and LABELED WITH BANDS. The `labeling_note` key now appears whenever EITHER of
  the first two is in scope, not only for unlabeled rows, and neither may be
  narrated as "analyzed and calm" — the deterministic evidence (signals,
  signal_count, corroboration, off_text_flag) is what is complete for them. On a
  pre-migration DB a sentence saying `labeled` is INFERRED is appended. BOTH of
  the first two also carry a code-printed `_epilogue` (never a payload-only
  note): the framing floor is code, never prompt, and "looked at, nothing
  shippable" licenses the same false "I checked and it was calm" summary the
  unlabeled epilogue was added to prevent. Both tools count the same quantity
  under the same name — `labeled_signal_turns`, NON-abstain rows — so drilling
  from `find_conflicts` into `get_conflict` cannot flip the framing; that tool's
  `signals_total` is the larger, paginating count (abstains are stored and
  returned, they are just not findings).
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
- Wellbeing narration is framed by two code mechanisms single-sourced in
  `chat_floor.TurnFloor`, not the prompt: (a) a whole-reply sanitize — streaming is held and the ENTIRE
  assembled reply runs through `wellbeing.sanitize_condition_names` before it is
  printed/returned; and (b) it collects a tool result's reserved `_epilogue` (the
  code-emitted `support_block`), strips both `_epilogue` AND `support_block` from
  the data the model sees, and prints the epilogue verbatim after the reply (the
  envelope also carries a model-safe `flagged` bool). `wrapped.py` runs the same
  scrub directly. THREE turn-runners must enforce the floor — `chat.run_turn`,
  `chat.run_turn_events` (the SSE generator the optional web API streams) and
  `chat_apple.run_turn`, which bypasses `run_turn` entirely — and all three get
  it by constructing the same `chat_floor.TurnFloor`, which is why the floor
  lives in its own module rather than inside `run_turn`.
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
  untouched. That carve-out covers a genuinely UNSET value only: `CONFLICT_THINK`
  is `_think("CONFLICT_THINK", CHAT_THINK)`, so an explicit `CHAT_THINK=true` is
  INHERITED as `True` and passes straight through, buying the 197 s pass the
  carve-out exists to avoid — set `CONFLICT_THINK=false` to keep thinking for
  `chat.py` only. `CHAT_THINK` deliberately still defaults to `None`, so on the
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
