# evalkit — implementation plan

> **Status: implemented.** The framework this plan specifies now exists in
> `eval/evalkit/` — the code is authoritative and this document is kept as
> design rationale. For how to run it, see [`eval/README.md`](../eval/README.md).

Companion to [`ANALYTICS_TEST_PLAN.md`](ANALYTICS_TEST_PLAN.md) (the eval
spec this framework mechanizes). This document is self-contained: an
implementing agent should be able to build the framework from it plus the
referenced source files, without re-deriving the design.

**Goal:** a lightweight test framework, separate from the main app, that
loads test prompts with expected-output success criteria, runs them through
the chat app, captures full structured conversation transcripts, checks them
deterministically against computed ground truth, and exposes everything as
per-case JSON files so coding agents can fan out in parallel on grading and
synthesis — and draw actionable insights (which lever to tune in the chat
app or ingest pipeline).

**Decisions already made with the user:** framework lives in `eval/evalkit/`;
one chat model per run (`--model` flag; comparing models = two runs +
`report --compare`); `eval/fast_eval.sh` stays as-is; **JSON** case files;
stdlib-only (no new dependencies).

---

## 1. Verified integration facts (do not re-derive; all confirmed by reading code)

- **`chat.run_turn(messages, show_model_header=False, model=...)` mutates
  `messages` in place** (`chat.py:234`). After it returns, `messages` is the
  complete structured transcript: system, user, assistant messages with
  `tool_calls`, `{"role": "tool", "content": <capped-JSON string>}` results,
  final reply. Multi-turn = append the next user message and call `run_turn`
  again on the same list. This is the transcript-capture mechanism — never
  scrape stdout.
- **`chat.console` (chat.py:27) and `chat.ollama_chat_stream` (chat.py:134)
  are module globals resolved at call time** inside `run_turn`. Assign
  `chat.console = rich.console.Console(quiet=True)` to silence all output;
  wrap `chat.ollama_chat_stream` to time round trips and capture per-round
  tool calls. `tests/test_chat.py` already uses this pattern. No app edits.
- **`config.py` reads env at import; `load_dotenv()` does NOT override
  pre-set env vars.** Set `TARGET_DB` / `MAX_TOOL_ITERATIONS` /
  `TOOL_RESULT_MAX_CHARS` in `os.environ` **before importing**
  `config`/`chat`/`query_tools`. Therefore every evalkit module imports app
  modules **lazily inside functions**, never at module top level.
- **Real DB** (~222 MB) exists only in the main checkout
  `<repo-root>/imessage_analysis.db`.
  The worktree copy is a **0-byte stub** — the runner must refuse to start
  (exit 2, with an explicit fix hint) when the resolved DB is missing or
  0 bytes. Worktrees have **no `.venv`**; python is the main checkout's
  `.venv/bin/python3`.
- **Ollama serializes requests.** Sweeps (`run`) and LLM-judge steps
  (`p3 wrapped`, `p3 ... --judge`) run in ONE sequential lane, enforced by a
  lockfile. Everything else (manifest/GT, check, report, agent grading) is
  parallel-safe file I/O.
- **Ground truth** is computed by direct `query_tools.<fn>(...)` calls — no
  Ollama. `query_tools.DB_PATH` is a documented patchable package attribute,
  but setting `TARGET_DB` env pre-import is the primary mechanism.
- `run_turn` already retries `OllamaResponseError` per round trip, dedups
  identical tool calls, and makes a tools-withheld final request on budget
  exhaustion or an empty stall. Transport errors (`OSError`,
  `http.client.HTTPException`) **escape** `run_turn` — that is the runner's
  retry surface (retry the whole case once, then record `transport_error`).
- `ollama.chat_once(messages, model=None, format=None, timeout=None)`
  (ollama.py:24) is the non-streaming JSON-mode helper for `p3 --judge`.
- `eval/.gitignore` already ignores `results/`. It does **not** yet ignore
  the manifest — that must be added (real names/messages).
- Config knobs relevant to a run (snapshot them into every transcript):
  `CHAT_MODEL`, `MAX_TOOL_ITERATIONS`, `TOOL_RESULT_MAX_CHARS`,
  `CHAT_TIMEOUT`, `OLLAMA_HOST/PORT`, `LOCAL_TZ`, `SESSION_GAP_MINUTES`,
  `INITIATION_GAP_HOURS`, `DOUBLE_TEXT_MIN_MINUTES`, `ADJACENCY_MINUTES`,
  `LATE_NIGHT_START_HOUR/END_HOUR`, `DYAD_TERM_MIN_USES/MIN_LIFT`.

### 1.1 The 21 tools (exact names, for expectations and banned lists)

`search_messages_hybrid, search_conversations, search_messages_fts,
search_messages_sql, find_similar_conversations, get_conversation,
get_message_context, get_chat_summary_stats, get_activity_patterns,
get_response_dynamics, get_reaction_stats, get_milestones, get_word_trends,
get_group_dynamics, get_chat_participant_stats, get_word_frequency,
search_contacts_by_name, list_chats, search_contact_by_handle,
get_message_reactions, get_full_messages`

`ANALYTICS_TOOLS` (for S7.1's "no analytics tool" ban) =
`get_chat_summary_stats, get_activity_patterns, get_response_dynamics,
get_reaction_stats, get_milestones, get_word_trends, get_group_dynamics,
get_chat_participant_stats, get_word_frequency`.

**Removed-tool names** (S7.3 `banned_anywhere`, run-level flag):
`get_activity_histogram`, `get_conversation_dynamics`,
`get_top_reacted_messages`. Note: the test plan's mention of
`get_db_overview` is not a real tool — the real alternate for S5.7 is
`get_chat_summary_stats`.

### 1.2 Analytics tool return shapes (verified; what GT recipes extract from)

- **`get_activity_patterns(chat_id/handle_id/contact_id, date_from, date_to,
  granularity='month', include)`** → `{timezone, message_count,
  heatmap: {weekdays, counts: 7x24 (Monday-first rows, local hours 0-23)},
  volume_series: {granularity, series: [{period, sent, received}], note?},
  late_night: {start_hour, end_hour, n, total, share, top_contacts?:
  [{contact, messages, late_night, share}]}, busiest_days: [{day, count}],
  streaks: {longest_streak: {days, start, end}, longest_drought: {days,
  from, to}}}`. Buckets in LOCAL_TZ. Unscoped late_night adds the
  leaderboard.
- **`get_response_dynamics(scope required, date_from, date_to)`** →
  `{chat_ids, initiation_gap_hours, message_count, from_me, from_them,
  reply_latency: {me: {median_seconds, p90_seconds, count}, them: {...}},
  read_to_reply: {pairs, median_unseen_seconds,
  median_deliberation_seconds, coverage: {inbound_with_date_read,
  inbound_total, outbound_with_date_read, outbound_total}},
  initiation: {me, them}, sessions: {count, median_messages,
  median_duration_seconds, ended_by_me_share}, ghosted: {only_me,
  only_them}, double_text: {me, them}, burstiness}` (+
  `initiations_by_member: [{member, sessions_initiated}]` for an explicit
  group chat_id). No scope → `{"error": "..."}` (S2.9 probe relies on this).
  Ghosting semantics: `only_me` = sessions where every message is mine
  (I started, no reply).
- **`get_reaction_stats(scope?, date_from, date_to, include, reaction_type,
  limit)`** → `{chat_ids, given: {total, by_type: {name: n},
  per_100_received_messages}, received: {total, by_type,
  per_100_sent_messages}, reciprocity: {given, received, ratio},
  time_to_react: {me_reacting: {median_seconds, count}, them_reacting:
  {...}}, funniest: [{participant, contact_id, handle_ids, laughs,
  reactions_on_my_messages, my_messages, laughs_per_100_my_messages}],
  hall_of_fame: [{...message receipt cols..., guid, reaction_count,
  reactions_by_type}]}`. `reaction_type` accepts names ('laugh','love') or
  codes. Date bounds hit the reaction date except hall_of_fame (message
  date).
- **`get_milestones(scope?, phrase, n, reference_date, limit)`** →
  `{chat_ids, firsts: {first, last, first_from_me, first_from_them}` (each a
  receipt dict with date/sender/text), `nth: {total_messages, n, message}`
  or `{total_messages, landmarks: [{ordinal, message}]}` (landmark ordinals
  1000/5000/10000/25000/50000), `anniversaries` (scoped: {first_contact,
  next_anniversary, years, days_until}; unscoped: {reference_date,
  window_days, upcoming: [{participant, contact_id, first_date,
  lifetime_messages, next_anniversary, years, days_until}]} — min 100
  lifetime messages, 30-day window), `longest_silence: {available, from, to,
  days, broken_by, reconnection_message}` (scope required),
  `earliest_phrase: {phrase, message}}`.
- **`get_word_trends(phrase?, scope?, date_from, date_to,
  granularity='year', include)`** → `{granularity, chat_ids,
  phrase_trend: {phrase, total_messages, series: [{period, me, them}]},
  emoji: {top: [{emoji, count}], total_emoji},
  rates: {series: [{period, me: {n_messages, we_per_100, question_per_100,
  laugh_per_100}, them: {...}}]},
  distinctive_terms: {terms: [{term, chat_id, lift, n_uses, verdict,
  curated, first_message_id, date, text, coined_by}]} or {available: false,
  note}, lsm: {score, per_category, trajectory: [{period, score}]} or
  {available: false}, emotions: {top: [{emotion, count}], chunks_annotated}
  or {available: false, note}, valence: {series: [{period, mean_valence,
  n_chunks}]} or {available: false, note}}`. S4.8's gate: empty
  `chunk_annotations` ⇒ `{available: false}`.
- **`get_group_dynamics(chat_id required, date_from, date_to, include)`** →
  `{chat_ids, members: [{member, is_me, messages, share, words}],
  gini: {gini, n_members, drift}, lurkers: [{member, messages}],
  igniters: {n_sessions, leaderboard: [{member, sessions_started, share}]},
  adjacency: {gap_minutes, pairs: [{replier, replied_to, count}]},
  group_lsm: {members: [{member, lsm_score}]}}`. 1:1 chat →
  `{"error": "...is a 1:1 chat..."}` (S6.7 probe relies on this).
- **`get_word_frequency(scope?, is_from_me, date_from, date_to, ngram=1,
  top=25, min_count=2, include_stopwords=False)`** → `{rows: [{ngram,
  count}], ngram, messages_scanned, stopwords_excluded}`. Scopes by
  **sender** (unlike the other analytics tools).
- **`get_chat_summary_stats(scope?, is_from_me, date_from, date_to,
  group_by_chat, limit)`** → pooled `{message_count, attachment_count,
  reactions_sent_count, first_date, last_date, ...}` or per-chat rows.

---

## 2. Directory layout (files to create marked +)

```
eval/
+ __init__.py                   # empty — makes `python -m eval.evalkit` importable
+ evalkit/
+   __init__.py                 # docstring + shared paths: EVAL_DIR, CASES_DIR,
+                               #   RESULTS_DIR, MANIFEST_PATH, LOCK_PATH
+   __main__.py                 # argparse CLI: manifest | run | check | report | p3 | selfcheck
+   cases.py                    # load/validate case JSON, placeholder substitution/refs
+   recipes.py                  # P0: resolve_roster() + @recipe registry of GT functions
+   manifest.py                 # P0: recipes -> eval/analytics_eval_manifest.json + fingerprint
+   runner.py                   # P1: in-process sequential sweep, lockfile, transcripts
+   checks.py                   # P2a: deterministic checkers -> <id>.checks.json
+   report.py                   # P4: grades -> scorecard.json/.md, --compare
+   p3.py                       # P3: wrapped number check, annotate/topics samplers
+ cases/                        # committed — placeholders only, NO real names
+   roster.json                 # placeholder definitions (see §4.1)
+   S1.json … S8.json           # transcribed suites (see §5)
+   fixtures/canned_transcript.json   # synthetic-name fixture for checker tests
  results/                      # gitignored (already) — <run-id>/ per sweep
  analytics_eval_manifest.json  # generated by P0; add to eval/.gitignore
  fast_eval.sh, prompts-smoke.txt    # existing; keep untouched
+ README.md                     # agent orchestration guide (see §12)
tests/
+ test_evalkit_checks.py        # hermetic unit tests (unittest style, like existing tests)
docs/
  EVALKIT_IMPLEMENTATION_PLAN.md  # this file
```

Invocation, always from the repo/worktree root:

```
<repo-root>/.venv/bin/python3 \
    -m eval.evalkit <cmd> ...
```

Under `-m`, `sys.path[0]` = cwd, so the local `chat.py`/`query_tools` are
imported. The framework never modifies app modules on disk; runtime patches
(`chat.console`, timing wrapper) are in-process only.

Append to `eval/.gitignore`:

```
analytics_eval_manifest.json
analytics_eval_manifest.*.json
prompts-analytics.txt
```

Style: ruff, line-length 100, E+F (`make lint` must pass). Match the repo's
comment conventions (comments state constraints, not narration).

---

## 3. Schemas (the inter-agent contract — freeze these before parallel work)

### 3.1 Suite file (`eval/cases/Sx.json`)

```json
{
  "suite": "S1",
  "title": "Activity patterns (get_activity_patterns)",
  "targets": {"routing": 0.9, "accuracy": 0.85, "faithfulness": 1.0, "communication": 0.85},
  "cases": [ <case>, ... ]
}
```

### 3.2 Case

| field | type | meaning |
|---|---|---|
| `id` | str | `"S1.3"` |
| `prompt` | str | template with `<TOP1>`-style placeholders (single-turn); XOR `turns` |
| `turns` | [str] | multi-turn prompt list (S8 only) |
| `expect.tools` | [ToolExpect] | order-agnostic; each `{"name": str, "alternates": [str], "args_contains": {…}, "turn": int?}` |
| `expect.banned_tools` | [str] | tools that must not be **called** in this transcript |
| `expect.banned_anywhere` | [str] | strings that must not appear anywhere in the serialized transcript (also raises a run-level flag) |
| `gt` | `{"recipe": str, "params": {…}}` \| null | reference into the recipe registry |
| `probe` | str \| null | `scope_requirement \| error_path \| gated_section \| negative_control \| thin_data \| open_ended` |
| `grader_notes` | str | verbatim accuracy-check language from the test-plan row |
| `quality_watch` | str \| null | communication-axis caveats |

`args_contains` value forms (checker semantics in §9):
- literal (str/int/bool): arg equals it (case-insensitive for str), or arg-list contains it
- `"<TOP1>.any_chat_id"`: arg (scalar or list) intersects that roster entry's `chat_ids`
- pseudo-key `"_scope": "<TOP1>"`: passes if ANY of the call's `chat_id`/`handle_id`/`contact_id` args intersects the entry's `chat_ids`/`handle_ids`/`{contact_id}`
- pseudo-key `"_date_year": "<YEAR>"`: passes if `date_from`/`date_to` args fall within `[YEAR-01-01 − 1 day, YEAR-12-31 + 1 day]`

Do **not** put `include` in `args_contains` (models legitimately omit it —
default is all sections).

### 3.3 Roster file (`eval/cases/roster.json`, committed — definitions only)

```json
{"placeholders": {
  "TOP1":  {"kind": "contact", "definition": "#1 1:1 contact by lifetime message count"},
  "TOP2":  {"kind": "contact", "definition": "#2 1:1 contact by lifetime message count"},
  "SPARSE": {"kind": "contact", "definition": "a real contact with 30-100 lifetime messages"},
  "GROUP": {"kind": "group_chat", "definition": "busiest group chat with >= 4 participants"},
  "PHRASE_HI": {"kind": "string", "definition": ">= 200 lifetime FTS hits, usage visibly changed across years"},
  "PHRASE_LO": {"kind": "string", "definition": "a phrase with 1-5 lifetime hits"},
  "TERM":  {"kind": "string", "definition": "top-lift dyad_terms row for TOP1's chat"},
  "NOBODY": {"kind": "string", "definition": "plausible full name matching no contact"},
  "YEAR":  {"kind": "int", "definition": "most recent full calendar year in the corpus"}
}}
```

### 3.4 Manifest (`eval/analytics_eval_manifest.json`, gitignored — real names)

```json
{
  "created_utc": "2026-07-17T01:00:00Z",
  "corpus_fingerprint": {"message_count": 350123, "max_message_date": "2026-07-14T03:12:09",
                          "db_path": "/abs/path/imessage_analysis.db"},
  "roster": {
    "TOP1": {"display_name": "…", "contact_id": 12, "handle_ids": [3, 4], "chat_ids": [7, 8]},
    "TOP2": {…}, "SPARSE": {…},
    "GROUP": {"display_name": "…", "chat_ids": [<int>…], "n_participants": <int>},
    "PHRASE_HI": "…", "PHRASE_LO": "…", "TERM": "…", "NOBODY": "…", "YEAR": 2025
  },
  "gt": {"S1.1": {"recipe": "activity_argmax", "params_resolved": {…},
                   "values": {"top_hour": {"value": 22, "tol": "exact"}},
                   "raw": {…}, "status": "ok|error", "error": null}, …}
}
```

`values` tolerance vocabulary (`tol`): `exact` (counts, dates, names, hours),
`round` (floats — match at rounding to 0 or 1 decimals), `pct1` (shares/rates
— ±1 percentage point; stored value may be a 0–1 share, the checker also
tries ×100). `raw` keeps the full tool return(s) for graders.
Fingerprint = `SELECT COUNT(*), MAX(date) FROM messages`.

### 3.5 Transcript (`eval/results/<run-id>/<id>.json`)

```json
{
  "id": "S1.3", "run_id": "…", "status": "ok|timeout|transport_error|empty_reply",
  "prompt_resolved": "…", "turns_resolved": ["…"], "model": "…",
  "config": {"MAX_TOOL_ITERATIONS": 8, "TOOL_RESULT_MAX_CHARS": 12000, "…": "see §1 knob list"},
  "corpus_fingerprint": {…}, "started_utc": "…", "duration_s": 208.4,
  "round_trips": [{"turn": 1, "duration_s": 31.2,
                    "tool_calls": [{"name": "…", "arguments": {…}}]}],
  "turn_boundaries": [{"turn": 1, "user_index": 1, "final_reply_index": 6}],
  "messages": [ …the full mutated messages list, verbatim… ],
  "final_reply": "…"
}
```

Plus a human-skimmable `<id>.txt` sibling (prompt, tool-call lines, final
reply). `run.json` in the run dir:
`{"run_id", "label", "created_utc", "model", "config", "corpus_fingerprint",
"cases": {"S1.3": {"status": "ok", "duration_s": 208.4}}, "flags":
["banned_anywhere:get_activity_histogram@S7.3"]}`.

### 3.6 Checks (`<id>.checks.json`)

```json
{
  "id": "S1.3", "run_id": "…", "fingerprint_ok": true,
  "routing": {"auto": "pass|fail",
    "expected": [{"name": "get_activity_patterns", "matched_call": {…}|null,
                   "args_ok": true, "arg_diffs": []}],
    "called": ["search_contacts_by_name", "get_activity_patterns"],
    "banned_called": [], "banned_anywhere_hits": []},
  "numeric": {"advisory": true,
    "values": {"march_total": {"expected": 431, "tol": "exact",
                "found_in_reply": true, "found_in_tool_results": true}},
    "missing_from_reply": ["sent_received_split"]},
  "probe": {"type": null, "auto": "pass|fail|manual", "detail": "…"},
  "unsupported_numbers": {"advisory": true,
    "numbers_in_reply_not_in_tool_results": ["17%"]}
}
```

Routing is the only fully deterministic verdict. Numeric / faithfulness
blocks are **advisory evidence** for graders (prose number-matching is
fuzzy) — they never auto-fail a case.

### 3.7 Grade (`<id>.grade.json`, written by grading agents — not by evalkit)

```json
{
  "id": "S1.3", "run_id": "…", "suite": "S1", "grader": "agent:S1",
  "axes": {"routing": 1, "accuracy": 0, "faithfulness": 1, "communication": 1},
  "evidence": [{"axis": "accuracy", "note": "reply says March=412, GT=431", "ref": "messages[7]"}],
  "levers": ["shape"],
  "blocked": false, "blocked_reason": null, "notes": "…"
}
```

Rules: every 0 axis requires ≥1 `evidence` entry and ≥1 lever from
`desc steer args shape size confab pipeline judge-batch`. `blocked: true`
(the test plan's BLOCKED(bug) rule) excludes the case from means.

### 3.8 Scorecard (`scorecard.json` + `scorecard.md`)

```json
{
  "run_id": "…", "compared_to": null, "model": "…", "config": {…}, "corpus_fingerprint": {…},
  "suites": {"S1": {"n": 8, "graded": 8, "blocked": 0,
      "axes": {"routing": {"mean": 0.88, "target": 0.9, "pass": false, "delta_vs_prev": null}},
      "failing_cases": [{"id": "S1.3", "axes": {…}, "levers": ["shape"]}]}},
  "lever_histogram": {"shape": 4, "confab": 1},
  "run_flags": [], "p3": {…}, "overall_pass": false,
  "ungraded": [], "blocked_cases": []
}
```

---

## 4. Module specs

### 4.1 `cases.py`

- `PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*)(?:\.([a-z_]+))?>")`
- `load_suites(suites=None, case_ids=None) -> [suite dicts]` from
  `CASES_DIR/S*.json`; validates on load.
- `validate_suite/validate_case`: unique ids; `prompt` XOR `turns`; probe in
  vocabulary; `expect` keys well-formed; every placeholder used appears in
  `roster.json`; `gt.recipe` is a string (existence in the registry is
  `selfcheck`'s job, so cases.py stays import-light).
- `substitute_text(text, roster) -> str`: `<NAME>` → display form
  (contact/group dicts → `display_name`; scalars → `str(value)`).
- `resolve_ref(value, roster)`: for `gt.params` / `args_contains` values —
  a string that IS a placeholder ref returns the resolved object
  (`"<TOP1>"` → the roster entry dict; `"<TOP1>.chat_ids"` → the list;
  `"<YEAR>"` → int); a string merely containing placeholders returns it with
  inline substitution; everything else passes through.
- Constants: `PROBE_TYPES`, `LEVERS`, `AXES`, `ANALYTICS_TOOLS`,
  `ALL_TOOLS` (§1.1).

### 4.2 `recipes.py` (P0)

- `RECIPES: dict[str, Callable]`; `@recipe("name")` decorator. Signature:
  `fn(roster, **params) -> {"values": {name: {"value": v, "tol": t}}, "raw": {…}}`.
  App imports (`query_tools`, `db`, `sqlite3` connections) **inside**
  functions.
- Scope helper: `_scope_kwargs(entry, prefer="contact")` → contact-scope
  tools get `{"contact_id": entry["contact_id"]}`; chat-scope tools
  (`get_group_dynamics`, `distinctive_terms`, participant stats) get
  `{"chat_id": entry["chat_ids"]}`.
- Date helper: `_year_bounds(year)` → `("YYYY-01-01", "YYYY-12-31")`.
- `resolve_roster(db_path, overrides=None) -> roster dict`:
  - `TOP1`/`TOP2`: top 1:1 contacts by lifetime real-message count — SQL over
    `chats(is_group=0) ⋈ chat_participants ⋈ handles ⋈ contacts`, grouped by
    contact, counting both sides' dated real messages
    (reuse `db.CONTACT_LABEL_SQL` for names; exclude emote/system rows the
    same way `query_tools._shared._REAL_MESSAGE_CLAUSES` does).
    Collect the contact's `handle_ids` and all their 1:1 `chat_ids`
    (iMessage + SMS twins).
  - `SPARSE`: a named contact with 30–100 lifetime messages (pick closest to
    60 with a non-null first_name).
  - `GROUP`: busiest `is_group=1` chat with ≥4 participants; include every
    chat row sharing its `display_name` in `chat_ids` (split-thread merge).
  - `YEAR`: `max_year` if `MAX(date) >= f"{max_year}-12-31"` else
    `max_year - 1`.
  - `PHRASE_HI`: candidates from `get_word_frequency(is_from_me=1, top≈60)`;
    for each, per-year FTS counts (`messages_fts MATCH '"word"'` joined to
    messages, `substr(date,1,4)` in SELECT over already-matched rows); pick
    the ≥200-total word with the highest max/min year ratio.
  - `PHRASE_LO`: seeded-random walk over longer messages; take a 3-gram,
    keep the first with 1–5 lifetime FTS hits.
  - `TERM`: top `dyad_terms` row for TOP1's chat_ids, preferring curated
    (`llm_verdict IN ('inside_joke','nickname')`), else highest lift.
  - `NOBODY`: a hardcoded implausible name (e.g. "Thaddeus Vellingworth"),
    verified against `search_contacts_by_name` to match nothing.
  - Print every auto-pick with its evidence; merge `overrides` last.
- ~30 recipe functions (see the per-case inventory in §5; parametrized
  recipes are shared across cases).

### 4.3 `manifest.py` (P0)

- `build(db_path, suites=None, roster_overrides=None)`: set `TARGET_DB` env
  pre-import; resolve roster (or load it from an existing manifest when
  `--suite` scopes a partial rebuild); run each selected case's recipe;
  write `MANIFEST_PATH` atomically (`.tmp` + `os.replace`). Recipe
  exceptions land as `status: "error"` entries, never abort the build.
- `--suite Sx` writes a sidecar `analytics_eval_manifest.S x.json`;
  `--merge` folds sidecars into the main manifest (parallel P0 agents each
  own suites without write contention). Default (no `--suite`) computes
  everything in one pass — it is pure SQL and fast.
- `fingerprint(db_path)` helper shared with runner/checks.

### 4.4 `runner.py` (P1)

CLI: `run --run-id ID (--all | --suite Sx … | --case id …) [--db PATH]
[--model M] [--resume] [--dry-run] [--force] [--max-iterations N]
[--tool-result-max-chars N] [--case-timeout S(default 1200)] [--label L]`.

Startup order (load-bearing):
1. Set env from flags (`TARGET_DB`, `MAX_TOOL_ITERATIONS`,
   `TOOL_RESULT_MAX_CHARS`) **before importing** config/chat/query_tools.
   Refuse (exit 2 + fix hint) if the resolved DB is missing or 0 bytes.
2. `import chat`; `chat.console = Console(quiet=True)`; wrap
   `chat.ollama_chat_stream` with a closure that times each round trip,
   records `msg.get("tool_calls")`, and raises `CaseTimeout` when the
   per-case wall budget is exceeded (checked between round trips).
3. Acquire `LOCK_PATH` via `os.open(O_CREAT|O_EXCL|O_WRONLY)`; write
   `{"pid", "run_id", "started_utc"}`. On conflict: read the lock; if
   `os.kill(pid, 0)` raises (dead), remove and retry once; else exit 2.
   Release in `finally`.
4. Load manifest; recompute the DB fingerprint; mismatch → exit 2 unless
   `--force` (recorded in `run.json`).
5. Create the run dir; write/merge `run.json`.

Per case, strictly sequential:
- Resolve prompt/turns via `cases.substitute_text`.
- `messages = [{"role": "system", "content": chat._build_system_prompt()},
  {"role": "user", "content": turn1}]`; `chat.run_turn(messages,
  show_model_header=False, model=model)`; for S8, append each next user turn
  and call `run_turn` again, recording `turn_boundaries`.
- On transport error escaping `run_turn` (`OSError`, `HTTPException`,
  `OllamaResponseError`): retry the whole case once from scratch; second
  failure → status `transport_error`, keep partial messages.
- `CaseTimeout` → status `timeout`, keep partial messages. Empty final reply
  → status `empty_reply`.
- Write `<id>.json` + `<id>.txt` atomically; update `run.json` after every
  case (crash-resumable).
- `--resume` skips ids whose transcript exists with status `ok`; an explicit
  `--case` always re-runs.
- `--dry-run`: print id → resolved prompt → expected tools; no Ollama, no
  lock, no writes.

Exit codes: 0 all ok / 1 some cases failed / 2 setup refusal.

### 4.5 `checks.py` (P2a) — see §9 for matching semantics

`check --run ID [--suite Sx] [--case id] [--force]` — reads case + transcript
+ manifest, writes `<id>.checks.json` per §3.6, prints a summary table, and
appends `banned_anywhere` hits to `run.json` `flags`. Also emits
`grading_todo` (transcripts without grades) into the summary. Pure file I/O;
parallel-safe; refuses on fingerprint mismatch unless `--force`.

### 4.6 `report.py` (P4)

`report --run ID [--compare PREV] [--strict]` — aggregates grades + checks
into `scorecard.json`/`scorecard.md` per §3.8. Markdown: per-suite axis
table vs targets, lever histogram, every 0 with its evidence line, run
flags, P3 block, and with `--compare` a delta column + newly-failing/
newly-passing lists (reads the previous run's `scorecard.json`).
`--strict` exit 1 unless every target met and faithfulness == 1.0.

### 4.7 `p3.py` (P3)

- `p3 wrapped --run ID [--db PATH]` (Ollama lane, takes the lock):
  `subprocess.run([sys.executable, "wrapped.py", "--year", <YEAR>, "--yes",
  "--out", results/<run>/P3.1.wrapped_llm.md], env=env)` and the `--no-llm`
  twin to `P3.1.wrapped_stats.md`.
- `p3 check-wrapped --run ID` (parallel-safe): extract every number token
  (ints with commas, decimals, percentages, ISO dates) from the LLM
  version's narrative lines; each must appear in the stats version
  (rounding-tolerant). `P3.1.checks.json`:
  `{"numbers_checked": N, "invented": [...], "manual": ["tone", "verbatim
  excerpts", "silent-exclusions honored", "markdown sanity"]}`;
  `invented != []` is an auto-fail.
- `p3 annotate-sample --run ID [--judge]`: stratified 25 chunks (most-neg 5 /
  most-pos 5 / middle 5 / highest-depth 5 / event-bearing 5) with chunk text
  + stored annotation + blank judgment slots → `P3.2.samples.json`; also
  report the skip rate. `--judge` fills slots via
  `ollama.chat_once(format="json")` under the lock.
- `p3 topics-sample --run ID [--judge]`: 10 random clusters ×
  `TOPIC_LABEL_SAMPLES` nearest chunks + label + degenerate-cluster stat
  (>40% share) + `topic_model_k` state → `P3.3.samples.json`.
- Both samplers exit 2 with "prereq missing" when their tables are empty.

### 4.8 `__main__.py`

Thin argparse dispatcher. Each submodule exposes
`add_parser(subparsers) -> None` (registers its flags, sets
`parser.set_defaults(func=…)`); `__main__` just parses and calls
`args.func(args)`. This makes the CLI wiring writable up front and keeps
workstreams independent. `selfcheck` lives here (or in cases.py): validate
all committed suite files, every `gt.recipe` exists in `RECIPES`, every
placeholder is defined in roster.json.

---

## 5. Case transcription (S1–S8) — per-case recipe/expectation inventory

Transcribe every row of `docs/ANALYTICS_TEST_PLAN.md`'s suite tables into
`eval/cases/S{1..8}.json` using the §3.2 schema. Prompts come verbatim from
the plan (placeholders kept). The table below fixes the recipe name/params
and the machine-checkable expectations per case — `grader_notes` should
carry the plan's accuracy-check text verbatim.

Notation: `AP = get_activity_patterns`, `RD = get_response_dynamics`,
`RS = get_reaction_stats`, `MS = get_milestones`, `WT = get_word_trends`,
`GD = get_group_dynamics`, `WF = get_word_frequency`,
`PS = get_chat_participant_stats`, `SS = get_chat_summary_stats`.

| id | expect.tools (name / args_contains) | gt recipe(params) → values (tol) |
|---|---|---|
| S1.1 | AP (unscoped) | `activity_argmax(axis=hour)` → top_hour (exact) |
| S1.2 | AP `_scope:<TOP1>` | `activity_argmax(axis=weekday, scope=<TOP1>)` → top_weekday (exact) |
| S1.3 | AP `_scope:<TOP1>, granularity:"month", _date_year:<YEAR>` | `volume_series(scope=<TOP1>, year=<YEAR>)` → per-month sent/received/total (exact) |
| S1.4 | AP granularity:"year" | `year_compare()` → this_year_total, last_year_total, direction (exact) |
| S1.5 | AP (unscoped; late_night) | `late_night()` → share (pct1), top_contact (exact) |
| S1.6 | AP `_scope:<TOP1>` (streaks) | `streaks(scope=<TOP1>)` → days (exact), start, end (exact) |
| S1.7 | AP (busiest_days) | `busiest_day()` → day, count (exact) |
| S1.8 | AP or SS `_scope:<SPARSE>, _date_year:<YEAR>` | `message_count(scope=<SPARSE>, year=<YEAR>)` → total (exact) |
| S2.1 | RD `_scope:<TOP1>` | `reply_latency(scope=<TOP1>)` → me_median_s, them_median_s (round), faster_side (exact) |
| S2.2 | RD `_scope:<TOP2>` | `reply_latency(scope=<TOP2>)` → them_median_s, them_p90_s (round) |
| S2.3 | RD `_scope:<TOP1>` | `initiation(scope=<TOP1>)` → me, them (exact), more_side (exact) |
| S2.4 | RD `_scope:<TOP1>` | `double_text(scope=<TOP1>)` → me (exact) |
| S2.5 | RD `_scope:<TOP1>` | `ghosted(scope=<TOP1>)` → only_me (exact) |
| S2.6 | RD `_scope:<TOP1>` | `read_to_reply(scope=<TOP1>)` → pairs, inbound_with_date_read, inbound_total (exact) — grader enforces the coverage caveat |
| S2.7 | RD `_scope:<TOP1>` | `sessions_burstiness(scope=<TOP1>)` → session_count, median_messages (exact), burstiness (round) |
| S2.8 | RD `chat_id:<GROUP>.any_chat_id` | `group_response(group=<GROUP>)` → top_initiator (exact); raw carries the full result |
| S2.9 | (none) — probe `scope_requirement`, auto_probe `{"fail_if_reply_contains_verbatim_tool_error": true}` | none |
| S3.1 | RS (given) | `reactions_top_type(direction=given)` → top_type name+count (exact) |
| S3.2 | RS (hall_of_fame) | `hall_of_fame_top()` → text, date, reaction_count (exact) |
| S3.3 | RS `_scope:<TOP1>` | `reciprocity(scope=<TOP1>)` → given, received (exact), ratio (round) |
| S3.4 | RS (funniest) | `funniest()` → top participant, laughs_per_100 (round) |
| S3.5 | RS (time_to_react) | `time_to_react()` → them_median_s (round) |
| S3.6 | RS `_scope:<TOP1>` | `reaction_types_pair(scope=<TOP1>)` → given/received by_type (raw; top types exact) |
| S3.7 | RS `_date_year:<YEAR>`, reaction_type love | `reactions_received_year(type=love, year=<YEAR>)` → count (exact) |
| S4.1 | WT `phrase:<PHRASE_HI>` | `phrase_trend(phrase=<PHRASE_HI>)` → per-year me/them counts (exact), direction (exact) |
| S4.2 | WT or MS (alternates) `_scope:<TOP1>` | `term_first_use(scope=<TOP1>, term=<TERM>)` → date, coined_by, text (exact) |
| S4.3 | WT `_scope:<TOP1>` (distinctive_terms) | `distinctive_terms(scope=<TOP1>)` → top terms + curated flags (exact) |
| S4.4 | WT `_scope:<TOP1>` (emoji) | `emoji_top(scope=<TOP1>)` → top-3 emoji in order (exact) |
| S4.5 | WT `_scope:<TOP1>` (lsm) | `lsm(scope=<TOP1>)` → score (round, in [0,1]) |
| S4.6 | WT `_scope:<TOP1>` (rates) | `laugh_rates(scope=<TOP1>)` → recent me/them laugh_per_100 (pct1) |
| S4.7 | WF `_scope:<TOP2>` | `word_frequency_sender(scope=<TOP2>)` → top words (exact, order-tolerant) |
| S4.8 | WT `_scope:<TOP1>` — probe `gated_section` | `valence_gated(scope=<TOP1>)` → gated flag; if populated, per-period mean_valence (round) |
| S4.9 | WT phrase or FTS (alternates) | `phrase_hits_me(phrase=<PHRASE_LO>)` → hit count (exact) |
| S5.1 | MS `_scope:<TOP1>` | `milestone_firsts(scope=<TOP1>)` → first date, direction, text (exact) |
| S5.2 | MS `_scope:<TOP1>` | `nth_landmark(scope=<TOP1>)` → largest landmark ordinal + message date/text (exact) |
| S5.3 | MS `_scope:<TOP1>` | `longest_silence(scope=<TOP1>)` → days (round), from, to, broken_by (exact) |
| S5.4 | MS (unscoped anniversaries) | `anniversaries_upcoming()` → upcoming names+dates (exact) |
| S5.5 | MS `_scope:<TOP1>, phrase:<PHRASE_HI>` | `earliest_phrase(scope=<TOP1>, phrase=<PHRASE_HI>)` → date, text (exact) |
| S5.6 | MS `_scope:<SPARSE>` | `milestone_firsts(scope=<SPARSE>)` → last date/text (exact) |
| S5.7 | MS or SS (alternates, unscoped) | `milestone_firsts()` → earliest corpus date (exact) |
| S6.1 | GD `chat_id:<GROUP>.any_chat_id` | `group_dynamics(section=members)` → top member + share (pct1) |
| S6.2 | GD | `group_dynamics(section=lurkers)` → lurker names (exact) |
| S6.3 | GD | `group_dynamics(section=gini)` → gini (round) |
| S6.4 | GD | `group_dynamics(section=igniters)` → top igniter (exact) |
| S6.5 | GD | `group_dynamics(section=adjacency)` → top pair (exact) |
| S6.6 | GD | `group_dynamics(section=group_lsm)` → min-LSM member (exact) |
| S6.7 | (any) — probe `error_path`; acceptable recovery: PS or RD | none — auto_probe checks GD errored + a recovery tool was called |
| S7.1 | any of hybrid/conversations/fts; `banned_tools`: ANALYTICS_TOOLS | none |
| S7.2 | hybrid or conversations (alternates) | none — grader checks grounded quote |
| S7.3 | SS or MS `_scope:<TOP1>`; `banned_anywhere`: the 3 removed tool names | `total_messages_pair(scope=<TOP1>)` → total (exact) |
| S7.4 | AP AND RD (both expected) | `activity_and_latency(scope=<TOP1>)` → top_hour (exact), them_median_s (round) |
| S7.5 | ≥2 scoped calls (two ToolExpects: SS/RD ×`_scope:<TOP1>`, ×`_scope:<TOP2>`) | `compare_pair(a=<TOP1>, b=<TOP2>)` → totals + medians per contact (exact/round) |
| S7.6 | search_contacts_by_name — probe `negative_control` | none — auto_probe: resolution empty + no analytics call with a real id after |
| S7.7 | any scoped analytics — probe `thin_data` | `sparse_history(scope=<SPARSE>)` → lifetime total, per-year counts (exact) |
| S7.8 | (any) — probe `open_ended`, graded on faithfulness only | none |
| S7.9 | AP (late_night) | `late_night()` (reuse S1.5 GT) |
| S7.10 | AP then get_conversation / get_message_context / search_messages_sql (turn-agnostic) | `streaks(scope=<TOP1>)` (reuse S1.6) |
| S8.1 | turns×3; turn 2: AP `_scope:<TOP1>`; turn 3: RD `_scope:<TOP1>` | `top_contact_profile()` → top_contact, top_hour, initiation sides (exact) |
| S8.2 | turns×2; RD `_scope:<TOP1>` turn 1, RD `_scope:<TOP2>` turn 2 | `reply_latency_pair(a=<TOP1>, b=<TOP2>)` → both them-medians (round) |
| S8.3 | turns×2 (turn 2 after clarification): WT `_scope:<TOP1>` | `distinctive_terms(scope=<TOP1>)` (reuse S4.3) |
| S8.4 | turns×2; MS `_scope:<TOP1>` turn 1; retrieval tool turn 2 | `anniversary_first(scope=<TOP1>)` → first_contact date (exact) |

S1.1's `quality_watch` (and the suite-level note) must carry the local-time
warning: a UTC-shaped argmax differs by 4–5h — grade accuracy 0 if the reply
matches the UTC argmax instead of the LOCAL_TZ one (store both in GT raw).

---

## 6. Recipe inventory (deduplicated)

One function per row; parametrized reuse across cases as noted in §5:
`activity_argmax(axis, scope?)`, `volume_series(scope, year)`,
`year_compare()`, `late_night()`, `streaks(scope)`, `busiest_day()`,
`message_count(scope, year?)`, `reply_latency(scope)`,
`reply_latency_pair(a, b)`, `initiation(scope)`, `double_text(scope)`,
`ghosted(scope)`, `read_to_reply(scope)`, `sessions_burstiness(scope)`,
`group_response(group)`, `reactions_top_type(direction)`,
`hall_of_fame_top()`, `reciprocity(scope)`, `funniest()`,
`time_to_react()`, `reaction_types_pair(scope)`,
`reactions_received_year(type, year)`, `phrase_trend(phrase, scope?)`,
`term_first_use(scope, term)`, `distinctive_terms(scope)`,
`emoji_top(scope)`, `lsm(scope)`, `laugh_rates(scope)`,
`valence_gated(scope)`, `phrase_hits_me(phrase)`,
`milestone_firsts(scope?)`, `nth_landmark(scope)`, `longest_silence(scope)`,
`anniversaries_upcoming()`, `earliest_phrase(scope, phrase)`,
`word_frequency_sender(scope)`, `group_dynamics(group, section)`,
`total_messages_pair(scope)`, `activity_and_latency(scope)`,
`compare_pair(a, b)`, `sparse_history(scope)`, `top_contact_profile()`,
`anniversary_first(scope)`.

Each is 5–15 lines: call the query tool with `_scope_kwargs`/`_year_bounds`,
extract `values` with tolerance tags, keep the full return in `raw`.

---

## 7. Checker matching semantics (checks.py)

- **Tool-call extraction**: walk `transcript["messages"]`; assistant
  messages carry `tool_calls: [{"function": {"name", "arguments"}}]`
  (arguments already a dict). Map each assistant message to its turn via
  `turn_boundaries` for turn-scoped ToolExpects.
- **ToolExpect matches** if any call to `name` (or an entry in `alternates`)
  satisfies every `args_contains` entry (§3.2 value forms). Scope
  pseudo-param: intersect the call's `chat_id`/`handle_id`/`contact_id`
  (scalar or list) with the roster entry's ids. Date pseudo-param: parse
  `date_from`/`date_to` prefixes, allow ±1 day around the year bounds.
- **banned_tools**: fail if called at all. **banned_anywhere**: substring
  scan of `json.dumps(transcript["messages"])`; on hit, fail the case's
  routing AND append a flag to `run.json` (the test plan makes it
  routing 0 for the whole run — the flag is how graders learn that).
- **Number extraction** (advisory): regex
  `[-+]?\d[\d,]*(?:\.\d+)?%?` over reply text; normalize commas. Matching
  per `tol`: `exact` — numeric equality after comma-stripping; `round` —
  equal when either side is rounded to 0 or 1 decimals; `pct1` — within 1.0
  after trying both the raw value and ×100 (shares stored 0–1, replies say
  percent). Dates: ISO prefixes and "July 14, 2026"-style long forms both
  count (match on year + month name/number + day when present).
- **Probe auto-checks**: `gated_section` — some tool result contains
  `"available": false` (and manifest says gated) vs reply invents a mood
  arc (manual); `scope_requirement` — a tool result contained `"error"` and
  the reply relays that error text verbatim (fail) vs recovers (manual);
  `error_path` — GD errored + a recovery tool (PS/RD) was called (pass) /
  no recovery (manual); `negative_control` — contact resolution returned
  empty and no analytics tool was subsequently called with a real id.
- Every advisory block sets `"advisory": true` so graders know nothing
  auto-failed from it.

---

## 8. Tests (`tests/test_evalkit_checks.py`)

unittest style (match existing `tests/` conventions — hermetic, no DB, no
Ollama, no app imports beyond what's needed). Fixture
`eval/cases/fixtures/canned_transcript.json` uses invented names and
hand-built messages exercising: expected-tool match with args, alternate
tool match, `_scope` intersection, `_date_year` bounds, banned_tools hit,
banned_anywhere hit, gated/scope_requirement/negative_control probes,
number extraction with each tolerance, and confabulation-candidate listing.
Also cover: `cases.validate_case` failures (dup id, prompt+turns both set,
unknown probe), `resolve_ref` forms, and `report` aggregation + `--compare`
delta math over two hand-written grade files. Run with
`.venv/bin/python3 -m unittest tests.test_evalkit_checks` (never the full
discover — repo rule).

---

## 9. README for agents (`eval/README.md`, one page)

Must contain, as literal copy-pasteable commands:
- Paths: python = main checkout `.venv/bin/python3`; DB flag =
  `--db <repo-root>/imessage_analysis.db`;
  always run from the repo/worktree root (worktree DB is a 0-byte stub; no
  worktree venv).
- The Ollama-lane rule: only `run`, `p3 wrapped`, `p3 … --judge` touch
  Ollama; they self-enforce via `eval/results/.ollama.lock`; everything else
  is parallel-safe. Never run two Ollama-touching commands concurrently.
- P0→P4 orchestration:
  `manifest --db …` (review roster; optionally `--roster-overrides f.json`)
  → `run --run-id <utc-ts>-<label> --all --db …` (ONE agent)
  → `check --run <id>` (parallel-safe)
  → spawn one grader agent per suite (what to read, what to write — the
  §3.7 grade schema and lever vocabulary reproduced in the README)
  → `p3 wrapped` + `p3 check-wrapped` (+ samplers if prereqs exist)
  → `report --run <id> --compare <prev>` → synthesis agent reads
  `scorecard.md`.
- Rules: BLOCKED(bug); fingerprint mismatch invalidates grading; privacy
  (never commit `results/` or the manifest); re-run-after-tuning loop
  (`run --suite Sx --run-id <new>` → `check` → grade → `report --compare`).

---

## 10. Verification (framework smoke, not the full eval)

1. `.venv/bin/python3 -m unittest tests.test_evalkit_checks` (the one new
   module only).
2. `python -m eval.evalkit selfcheck` — all suite files valid, recipes
   exist, placeholders known.
3. `python -m eval.evalkit manifest --db <real-db>` — roster prints sanely;
   GT statuses all `ok` (eyeball 2–3 values against a direct query_tools
   call).
4. `python -m eval.evalkit run --dry-run --all` — resolved prompts + plan.
5. If Ollama is up: tiny live sweep
   `run --run-id smoke --case S1.1 --case S5.7 --db <real-db>
   --max-iterations 8 --tool-result-max-chars 12000` → `check --run smoke`
   → confirm full `messages` in the transcript, silent console, sensible
   checks.json. Launch a second `run` mid-sweep → must exit 2 (lock).
6. `make lint`.

---

## 11. Parallel implementation analysis

The §3 schemas are the inter-agent contract; every workstream codes against
them, not against each other's files. File ownership is disjoint, so
parallel agents cannot conflict.

**Serial foundation (do first, single agent, small):**
- **W0**: `eval/__init__.py`, `eval/evalkit/__init__.py` (paths),
  `eval/evalkit/__main__.py` (dispatcher calling each module's
  `add_parser`), `cases.py`, `eval/cases/roster.json`, `.gitignore`
  additions, `selfcheck`. Everything downstream imports `cases.py` and
  registers into `__main__.py`'s pattern — freeze these first.

**Parallel wave (after W0 lands; 5 independent agents, disjoint files):**
- **W1 — Case transcription**: `eval/cases/S1.json … S8.json` from §5 +
  `docs/ANALYTICS_TEST_PLAN.md`. Pure content; validated by `selfcheck`.
  No code dependencies beyond the §3.2 schema.
- **W2 — Ground truth**: `recipes.py` + `manifest.py` (§4.2–4.3, §6).
  Depends only on the manifest schema + roster shape. Can verify against the
  real DB independently (read-only; no Ollama).
- **W3 — Runner**: `runner.py` (§4.4). Depends only on the transcript/run
  schemas and the chat.py patch points. Testable with `--dry-run` before W1
  finishes by pointing at one hand-written case file.
- **W4 — Checks + tests**: `checks.py`, `fixtures/canned_transcript.json`,
  `tests/test_evalkit_checks.py` (§4.5, §7, §8). Depends only on the
  case/transcript/manifest/checks schemas — the fixture substitutes for real
  runner output.
- **W5 — Report + P3**: `report.py` + `p3.py` (§4.6–4.7). Depends only on
  the grade/scorecard schemas; test with hand-written grade files.

  (W4 and W5 could be one agent if fewer lanes are available; they are the
  least coupled pair. W1 is the most mechanical and benefits most from a
  dedicated agent double-checking against the test plan tables.)

**Serial integration (single agent, after the wave):**
- **W6**: `eval/README.md` (needs final CLI flags as built), `selfcheck`
  run over W1's files, `make lint`, then the §10 verification ladder
  (manifest against the real DB, dry-run, tiny live sweep if Ollama is up —
  the live sweep is the only Ollama-lane step and must not run concurrently
  with any other Ollama user).

Cross-cutting rules for every implementing agent:
- Only W0 touches `__init__.py`/`__main__.py`; later modules self-register
  via `add_parser` — if the dispatcher needs a tweak, W6 makes it.
- App imports stay function-local everywhere (env-before-import invariant).
- No new dependencies; ruff line-length 100 (E+F) must pass per workstream.
- Never commit real names: W1/W4 fixtures use placeholders or invented
  names; W2's outputs are gitignored (verify before finishing).
- The real DB is read-only for every workstream; nothing here runs ingest.
