# Task brief: build an end-to-end demo runbook for the analytical suite

**You are a fresh agent picking this up in a new session.** Your job is to produce a
single copy-pasteable Markdown file, `DEMO.md`, that walks the repo owner through
running and **demoing every feature added in the "Close Read — imessage-rag
Analytical Suite" project, end to end.** The owner has their real iMessage history
already ingested, so the runbook must be concrete and runnable — real `make`
targets, real CLI flags, real env vars, and natural-language chat prompts that
actually route to each new tool.

This brief has two parts:
1. **Run a gather workflow** (a multi-agent fan-out that reads the code and returns
   verified demo steps per feature family). The ready-to-paste script is in
   §4 below.
2. **Synthesize the results into `DEMO.md`** using the structure in §5.

Work on the `analytical-suite` branch (or whatever branch has the features — verify
with `git branch --show-current`; it should be `analytical-suite`, which is 31
commits ahead of `main`). **Do not touch `main`.** The deliverable is a new doc
file; do not commit it unless the owner asks.

---

## 0. Ground truth to load first (do this before anything else)

Read these, in order — they are the authoritative map and will keep every command
in your runbook accurate:

- **`CLAUDE.md`** (repo root) — the module map, schema block, the full Commands
  list, and the load-bearing gotchas. This is your single best source.
- **`docs/DECISIONS.md`** — deeper rationale (measurements, why gates exist).
- **`Makefile`** — the exact target names and what each one runs (self-documenting;
  `make help` lists them).
- **`config.py`** / **`.env.example`** — env var names + defaults (the feature gates).
- **`tool_schemas.py`** — the exact chat-tool names (the JSON `name` must match the
  `query_tools` function). Verify every tool name you cite against this file.

### Feature inventory (what "the new features" means)

The suite is Waves 0–4. **V5 (`simulate_contact`) shipped via RES-69** (PR #33,
`res-69-simulate-contact` → `analytical-suite`) — it is the Wave-4 experimental
persona leg, so **include it**, but it is OFF by default and registration-gated
behind `PERSONA_SIM_ENABLED` (see the env-gate table below). If you are on
`analytical-suite` before PR #33 has merged, `grep simulate_contact tool_schemas.py`
returns nothing — check out `res-69-simulate-contact` (or wait for the merge) so the
tool actually exists before documenting it. Five user-facing families plus two demo
surfaces:

| Family | What it does | Build target | New chat tools | Wrapped section |
|---|---|---|---|---|
| **Foundation** (F1–F5) | Plumbing: lexicon split, `stats.py`, query helpers, eval-gate harness, `consent.py` | (built by ingest / used by others) | — | — |
| **Clearing the Air** (Conflict, C1–C5) | Deterministic conflict episodes + diagnose scorecard + apology scaffold + cross-chat trace | `make conflict` | `find_conflicts`, `get_conflict`, `diagnose_conflict`, `draft_apology`, `trace_conflict_across_chats` | Repair Report |
| **Loose Ends** (Nudges, N1–N5) | Detects unresolved threads (faded initiator, dormant tie, unanswered question, dropped plan, un-reciprocated good news, inferred occasion) | `make nudges` | `get_loose_threads`, `resolve_loose_thread`, `mute_contact_from_nudges` | Open Loops |
| **Fadeout Radar** (Drift, D1–D5) | Relationship-drift scoring + fadeout watchlist + reconnection candidates/opener | `make drift` | `get_relationship_drift`, `get_fadeout_watchlist`, `get_reconnection_candidates` | Drifting |
| **Baseline** (Wellbeing, W1–W4) | Within-person language-trajectory change points (self by default; consent-gated per contact) | (deterministic read; lexicon from ingest) | `get_language_trajectory` | Wellbeing trajectory |
| **Voiceprint** (Persona, V1–V5) | Per-person voice profile + style-matched draft + deterministic style check + (V5) ESTIMATE a contact's voice / talk to past-self | `make voiceprint` | `get_voiceprint`, `draft_reply`, `check_style_match`, `simulate_contact` (V5, gated behind `PERSONA_SIM_ENABLED`) | Voiceprint card |
| **Wrapped** | Self-contained year-in-review HTML (composes all of the above) | `.venv/bin/python3 wrapped.py` | — | (the report itself) |
| **Chat CLI** | Tool-calling REPL over the whole corpus | `.venv/bin/python3 chat.py` | 32 tools by default; +5 registered only behind feature flags (3 drift, `trace_conflict_across_chats`, `simulate_contact`) | — |

Full-suite build in one shot: **`make pipeline`** runs, strictly serially,
`ingest → annotate → nudges → drift → voiceprint → conflict → topics`. It must stay
serial (shared SQLite write lock + one serialized Ollama), so never parallelize it.

### Env-gate cheat sheet (verified in `config.py`, 2026-07-21)

Several features are **off by default** and will silently do nothing until enabled.
The runbook MUST call these out at the top of the relevant section:

| Env var | Default | Effect when set |
|---|---|---|
| `DRIFT_ENABLED` | **off** | The entire Drift feature. Off = `make drift` no-ops, the dyad_drift table never builds, AND the 3 drift read tools + reconnection opener are withheld from the chat tool list. Must be `true` to demo any drift surface (including Wrapped Drifting). |
| `NUDGE_DROPPED_PLAN_ENABLED` | **off** | The `dropped_plan` detector. Even ON it writes nothing until the F4 `dropped_plan` precision eval-gate is cleared — and it ships **no seed**, so it stays suppressed until the owner hand-labels + gates it. |
| `NUDGE_GOODNEWS_ENABLED` | **off** | The `unreciprocated_goodnews` detector. |
| `CONFLICT_TRACE_ENABLED` | **off** | Registers `trace_conflict_across_chats` in the chat tool list. Off = the tool doesn't exist to the model. |
| `CONFLICT_DV_SCREEN` | **on** | The softened DV floor. On (default) = a conflict window containing a DV marker gets its role/demand-withdraw read suppressed (mechanical scorecard still returned) and `draft_apology` withholds its whole scaffold. `chat.py --no-dv-screen` or `CONFLICT_DV_SCREEN=off` bypasses the role read but STILL appends the resource footer (the hotline text is never removed). Screen is read-only, never persisted. |
| `WELLBEING_CONSENT_IDS` | empty | Allowlist of contact ids `get_language_trajectory` will analyze. `get_language_trajectory` is self by default; a per-contact request is **refused in code** unless that id is listed. Consent only ADDS a contact; it never overrides a roster/silent exclusion. |
| `WRAPPED_WELLBEING` | **off** | Whether Wrapped composes the wellbeing trajectory section. Off even when a change-point window fires. Equivalent to `wrapped.py --wellbeing`. |
| `WRAPPED_VOICEPRINT_CONSENT_IDS` | empty | Default-redacted. With no ids, Wrapped emits NO third-party verbatim/stylometric voiceprint card at all. |
| `VOICEPRINT_MIN_CONTEXT_PAIRS` | `4` | `draft_reply` structurally DECLINES (zero generation) below this many reply pairs — an honest-degrade, not a failure. (`simulate_contact` reuses this floor too.) |
| `PERSONA_SIM_ENABLED` | **off** | The V5 `simulate_contact` tool (ESTIMATE a contact's voice / past-self). Off = the schema is withheld from the chat tool list, so the tool doesn't exist to the model — this registration gate is the real security boundary, NOT the per-call `acknowledged` param. Must be `true` to demo it. |
| `PERSONA_SIM_BLOCK_IDS` | empty | Hard deceased/grief denylist of `contacts.id` for `simulate_contact`. A listed contact is hard-refused in code (fixed compassionate message, no generation). Default-allow. Parsed via `consent.parse_consent_ids`. |
| `PERSONA_SIM_DORMANT_DAYS` / `PERSONA_SIM_ONESIDED_MIN_MSGS` | `365` / `3` | Soft deceased/dormant WARNING (never a block): fires in the acknowledgment payload when the contact's last message to you is older than N days AND you've since sent ≥M messages with no reply. A fail-open net over the manual denylist. |

Key honest-degrade behaviors to document (don't present as bugs):
- Valence-dependent reads trust ONLY `chunk_annotations` rows with `source='llm'`.
  A distilled-only corpus makes conflict corroboration `'none'`, skips the nudge
  Stage-0 valence exclusion, and leaves voiceprint `tone_*` NULL. To "light" these,
  run `annotate.py --quality balanced --upgrade-distilled` over the chats you care
  about first.
- Conflict has NO aggressor/role/status column by design.
- The eval-gate `conflict_labels` ships pre-cleared via a committed seed
  (`eval/seed_gates/conflict_labels.json`), so conflict escalation labels work out
  of the box; `dropped_plan` and `wellbeing_narration` stay suppressed until gated.

---

## 1. The deliverable

- **File:** `DEMO.md` at the repo root (create it; overwrite if present).
- **Audience:** the repo owner, who has real data ingested and Ollama running.
- **Voice:** operator runbook. Every step is a copy-pasteable command or an exact
  chat prompt, with a one-line "what you'll see." No hand-waving.
- **Length:** thorough but skimmable — headers, tables, fenced commands.

---

## 2. Prerequisites the runbook should state (Phase 0 of the demo)

Before any feature demo, the owner needs a built corpus and a running model. In
`DEMO.md`, open with a "Phase 0 — one-time setup" section:

```bash
make bootstrap          # fresh clone → venv, Ollama + models (~15GB), source data, ingest
# or, if already installed:
make ollama-start       # ensure the model server is up
make doctor             # verify venv, Ollama, models, source data, DB — read the [ok]/[warn]/[fail] lines
```

Then build the derived analytics the features read:

```bash
make annotate           # LLM chunk annotations (distilled tier by default); needed for llm-valence reads
make pipeline           # ingest → annotate → nudges → drift → voiceprint → conflict → topics (serial)
```

Note in the runbook: `make pipeline` will **no-op the drift step** unless
`DRIFT_ENABLED=true` is set first (see §0 gate table), and the two opt-in nudge
detectors stay dark unless their flags are on. Tell the owner to set the gates they
want to demo in `.env` (`make env` scaffolds it) before running the pipeline.

---

## 3. How to run the gather workflow

Ultracode is on for this environment, so the **Workflow tool is the right vehicle**:
fan out one reader per feature family, each reads the actual source and returns
structured, verified demo steps, then you synthesize. Paste the script in §4 into
the `Workflow` tool. It returns an array of per-family objects (overview, build
steps, env flags, chat demos, wrapped demo, gotchas, and the files each reader
actually opened).

**Fallback if the Workflow tool is unavailable** (or ultracode is off): spawn the
same 7 readers with the `Agent` tool (subagent_type `Explore` or `general-purpose`)
in a single message so they run concurrently, one per feature family, using the
prompts embedded in the script. Or, if you must, read the files yourself per §0 —
but the fan-out is faster and keeps each family's detail straight.

After the workflow returns, **read every result** and cross-check the tool names
against `tool_schemas.py` and the commands against the `Makefile` before writing.

---

## 4. The gather workflow script (paste into the Workflow tool)

```javascript
export const meta = {
  name: 'demo-runbook-gather',
  description: 'Read each analytical-suite feature family and return verified end-to-end demo steps',
  phases: [
    { title: 'Gather', detail: 'one reader per feature family returns demo steps' },
  ],
}

phase('Gather')

const DEMO_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    feature: { type: 'string', description: 'Feature family name' },
    overview: { type: 'string', description: '2-4 sentence plain-English description of what the user will see' },
    prerequisites: { type: 'array', items: { type: 'string' }, description: 'What must already be done/populated before this demo works' },
    build_steps: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          cmd: { type: 'string', description: 'Exact shell command, copy-pasteable' },
          description: { type: 'string', description: 'What it does / what to look for' },
        },
        required: ['cmd', 'description'],
      },
    },
    env_flags: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          var: { type: 'string' },
          default: { type: 'string' },
          effect: { type: 'string', description: 'What turning it on does; note if a feature is OFF by default and MUST be enabled to demo' },
        },
        required: ['var', 'default', 'effect'],
      },
    },
    chat_demos: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          prompt: { type: 'string', description: 'Natural-language prompt to type into chat.py that triggers this feature' },
          tool: { type: 'string', description: 'The query_tools function the model should call' },
          expected: { type: 'string', description: 'What the answer should contain / demonstrate' },
        },
        required: ['prompt', 'expected'],
      },
    },
    wrapped_demo: { type: 'string', description: 'Exact wrapped.py command + which section to look for, or "n/a"' },
    gotchas: { type: 'array', items: { type: 'string' }, description: 'Honest-degrade behavior, gates, DV/consent floors, things that look empty if a prereq is missing' },
    verified_against: { type: 'array', items: { type: 'string' }, description: 'Files actually read to confirm these commands/flags/tool names' },
  },
  required: ['feature', 'overview', 'build_steps', 'chat_demos', 'gotchas', 'verified_against'],
}

const COMMON = `You are documenting a DEMO RUNBOOK for the OWNER of the imessage-rag repo, who has their real iMessage history already ingested into imessage_analysis.db. Your job: return copy-pasteable, VERIFIED commands and realistic chat prompts that let them SEE this feature work end-to-end.

Rules:
- READ the actual files before answering. Do not paraphrase from memory. Confirm exact make target names, CLI flag spellings, env var names + defaults, and query_tools function names (the tool "name" in tool_schemas.py must match the function).
- The venv python is ".venv/bin/python3". Make targets exist in the Makefile.
- For chat demos: chat.py is an interactive REPL; a one-shot is ".venv/bin/python3 chat.py -p \\"...\\"". Give prompts that will actually route to this feature's read tools.
- Note anything OFF BY DEFAULT that must be enabled (env flag) to demo, and anything gated by consent/eval-gate/DV floor that will honestly return empty/suppressed if a prereq is missing.
- Keep commands minimal and real. If a feature needs a pipeline step run first to populate a table, say which make target.
Return the structured object. List every file you read in verified_against.`

const AREAS = [
  {
    label: 'setup-pipeline',
    feature: 'Setup & the full pipeline',
    prompt: `${COMMON}

FOCUS: the setup + data-build prerequisites shared by every feature. Read: Makefile (esp. "pipeline", "ingest", "annotate", "nudges", "drift", "voiceprint", "conflict", "topics" and their ordering), ingest.py (top-level flow only), config.py (model defaults), CLAUDE.md Commands section, .env.example (skim).
Produce the "phase 0" steps a demoer runs ONCE before any feature demo: install (make install / make install-ollama), ingest (make ingest), annotate (make annotate — what tier, what it lights up), and the full "make pipeline" (state the exact serial order ingest -> annotate -> nudges -> drift -> voiceprint -> conflict -> topics and WHY it must be serial). In build_steps give the exact commands in order. In gotchas: which downstream features need which pipeline step; that drift is opt-in so "make pipeline" no-ops it unless DRIFT_ENABLED; that valence-dependent reads need source='llm' annotations (annotate --quality balanced --upgrade-distilled). chat_demos: ONE simple sanity prompt proving chat.py + Ollama work. wrapped_demo: n/a.`,
  },
  {
    label: 'conflict',
    feature: 'Clearing the Air (Conflict)',
    prompt: `${COMMON}

FOCUS: Conflict family (C1-C5). Read: conflict_detect.py, query_tools/conflict.py (find_conflicts, get_conflict, diagnose_conflict, draft_apology, _dv_screen, trace_conflict_across_chats), tool_schemas.py (conflict schemas + which are flag-gated), the conflict SYSTEM_PROMPT lines in chat.py, config.py CONFLICT_* vars, wrapped_sections.py (_sec_repair_report / C5), .env.example.
build_steps: "make conflict" to populate conflicts/conflict_signals (needs annotate for source='llm' corroboration). env_flags: CONFLICT_TRACE_ENABLED (default + effect on trace_conflict_across_chats registration), CONFLICT_DV_SCREEN (default on), the chat.py --no-dv-screen flag. chat_demos: prompts that hit find_conflicts, get_conflict, diagnose_conflict (deterministic scorecard), draft_apology (Lewicki-6 scaffold). gotchas: the SOFTENED DV floor (role-suppress vs bypassed-with-footer, resource footer never removed, draft_apology withholds ENTIRELY on role-suppress, never persisted); NO aggressor/role column by design; corroboration='none' when no llm chunks; conflict_labels ships pre-cleared via committed seed. wrapped_demo: wrapped.py command + the Repair Report section.`,
  },
  {
    label: 'nudges',
    feature: 'Loose Ends (Nudges)',
    prompt: `${COMMON}

FOCUS: Nudges family (N1-N5). Read: loose_ends.py (the DETECTORS registry — list all detector kinds), query_tools/nudges.py (get_loose_threads, resolve_loose_thread, mute_contact_from_nudges), tool_schemas.py (nudge schemas), chat.py SYSTEM_PROMPT nudge lines, config.py NUDGE_* vars (incl. the enable flags), wrapped_sections.py (_sec_open_loops / N5).
build_steps: "make nudges" to populate loose_threads (Ollama only for the one-line nudge_text). env_flags: NUDGE_DROPPED_PLAN_ENABLED and NUDGE_GOODNEWS_ENABLED (both default OFF — must enable to demo those two detectors; dropped_plan additionally needs the F4 dropped_plan eval gate which ships NO seed, so it stays suppressed until owner labels+gates it). chat_demos: prompts hitting get_loose_threads, resolve_loose_thread, mute_contact_from_nudges. gotchas: dismissed rows kept never deleted; nudge_exclusions preserved across reset_db; inferred_occasion fills due_by; dropped_plan suppression path. wrapped_demo: wrapped.py + Open Loops section.`,
  },
  {
    label: 'drift',
    feature: 'Fadeout Radar (Drift)',
    prompt: `${COMMON}

FOCUS: Drift family (D1-D5). Read: drift.py (incl. --purge), query_tools/drift.py (get_relationship_drift, get_fadeout_watchlist, get_reconnection_candidates — all DRIFT_ENABLED-gated), tool_schemas.py (drift schemas are conditional), chat.py SYSTEM_PROMPT drift lines, config.py DRIFT_* vars, Makefile (drift, drift-purge), wrapped_sections.py (_sec_drifting / D4).
CRITICAL: DRIFT_ENABLED defaults FALSE — the ENTIRE feature (make drift step, the three read tools' schema registration, and the D3 reconnection tool) is off until true. build_steps must LEAD with enabling it (export DRIFT_ENABLED=true in .env) THEN "make drift". env_flags: DRIFT_ENABLED (the master gate), key DRIFT_* calibration knobs, the drift_params_hash rebuild trigger. chat_demos: prompts hitting each of the 3 read tools (which relationships are fading / reconnection candidates / fadeout watchlist). gotchas: schema-gated not prompt-gated; make drift-purge wipes dyad_drift regardless of flag (teardown when turning feature off); the channel_caveat; source='llm'-only curated references; drift_params_hash guard. wrapped_demo: wrapped.py + Drifting section (also needs DRIFT_ENABLED).`,
  },
  {
    label: 'wellbeing',
    feature: 'Baseline (Wellbeing)',
    prompt: `${COMMON}

FOCUS: Wellbeing family (W1-W4). Read: wellbeing.py (change-point core, support_block, sanitize_condition_names), query_tools/analytics_wellbeing.py (get_language_trajectory — self by default, consent-gated per contact), consent.py (is_consented), chat.py run_turn (the final-sentence sanitize arming + _epilogue support_block handling), config.py WELLBEING_* (incl. WELLBEING_CONSENT_IDS) + WRAPPED_WELLBEING, wrapped_sections.py (_sec_wellbeing / W4), wrapped.py (--wellbeing flag).
build_steps: get_language_trajectory is a deterministic read (lexicon from ingest) — confirm what it reads. chat_demos: a SELF trajectory prompt (how has my language/mood shifted over time) hitting get_language_trajectory (self by default); then a per-contact prompt and explain it is REFUSED unless that contact id is in WELLBEING_CONSENT_IDS. gotchas: per-contact consent enforced in code (refuses non-allowlisted before analysis); the two run_turn framing mechanisms (final-sentence condition-name sanitize + code-printed support_block epilogue) — the reply's last sentence is scrubbed and a support block appended verbatim; consent only ADDS, never overrides exclusions. wrapped_demo: WRAPPED_WELLBEING=true OR wrapped.py --wellbeing (OFF by default even when a window fires) + the trajectory section.`,
  },
  {
    label: 'voiceprint',
    feature: 'Voiceprint (Persona)',
    prompt: `${COMMON}

FOCUS: Voiceprint/Persona family (V1-V5). Read: voiceprint.py (deterministic per-person profiles -> person_voiceprint), query_tools/persona.py (get_voiceprint, draft_reply, check_style_match, AND the V5 simulate_contact block + its helpers _simulate_them/_simulate_past_self/_wrap_simulation/_predict_tapback/_dormant_signal), tool_schemas.py (persona schemas — note simulate_contact is registered ONLY when PERSONA_SIM_ENABLED, mirroring the drift/trace gate), chat.py SYSTEM_PROMPT persona lines (incl. the PERSONA_SIM_ENABLED-gated block), config.py VOICEPRINT_* (incl. VOICEPRINT_MIN_CONTEXT_PAIRS) + the PERSONA_SIM_* family (PERSONA_SIM_ENABLED, PERSONA_SIM_BLOCK_IDS, PERSONA_SIM_DORMANT_DAYS, PERSONA_SIM_ONESIDED_MIN_MSGS) + WRAPPED_VOICEPRINT_CONSENT_IDS, consent.py (is_consented — parses PERSONA_SIM_BLOCK_IDS), wrapped_sections.py (_sec_voiceprint / V4), Makefile (voiceprint).
build_steps: "make voiceprint" to populate person_voiceprint (LLM-free). env_flags: MUST include PERSONA_SIM_ENABLED (default OFF; the schema is withheld until true — enable it in .env to demo simulate_contact) and PERSONA_SIM_BLOCK_IDS. chat_demos: get_voiceprint (what's my texting style / X's style), draft_reply (draft a reply to X in their/my voice — note it DECLINES below VOICEPRINT_MIN_CONTEXT_PAIRS with zero generation), check_style_match (deterministic, does this draft sound like me — reads feature_std_json, no rescan), AND simulate_contact: (a) subject='them' — "estimate how X would reply to '...'": it FIRST returns acknowledgment_required (the model relays any deceased_warning, confirms it is an ESTIMATE, then re-calls with acknowledged=true), the returned `simulation` is code-labeled an estimate on EVERY line, and it carries a reaction base-rate tapback prediction; (b) subject='me' + as_of_date — "talk to my past self as of 2020": a bounded date-scoped rescan (min-range/scan-cap gated). gotchas: tone_* fields LOW-TRUST (source='llm' chunks only, NULL if none); scope dyad/group/all; V4 Wrapped voiceprint card is DEFAULT-REDACTED — renders third-party cards ONLY for ids in WRAPPED_VOICEPRINT_CONSENT_IDS (empty -> no third-party verbatim card at all). V5 simulate_contact gotchas: registration gate (PERSONA_SIM_ENABLED) is the real boundary, `acknowledged` is only a UX flow; the per-line "[ESTIMATE …]" label is applied IN CODE post-generation so the relay is never presentable as the person's real words; PERSONA_SIM_BLOCK_IDS hard-refuses in code (deceased/grief floor, default-allow); the soft dormant WARNING never blocks; past-self is capped by VOICEPRINT_MIN_RANGE_MESSAGES / VOICEPRINT_SCAN_MAX_MESSAGES; nothing sends, nothing is persisted; 1:1 only. wrapped_demo: wrapped.py + how to light the voiceprint card via WRAPPED_VOICEPRINT_CONSENT_IDS (simulate_contact is chat-only — no Wrapped section).`,
  },
  {
    label: 'wrapped-chat',
    feature: 'Wrapped report & Chat CLI (the two demo surfaces)',
    prompt: `${COMMON}

FOCUS: the two front-doors that show everything off. Read: wrapped.py (all flags: --year, --contact, --no-llm, --yes, --wellbeing; the featured-roster preview + silent-contact auto-exclusions; the self-contained HTML output path/name), chat.py (the REPL, -p one-shot, --model, --no-dv-screen, the SYSTEM_PROMPT tool list — how many tools), config.py WRAPPED_* and CHAT_* vars.
build_steps: the canonical "generate my Wrapped" command and where the HTML lands / how to open it; and starting the chat REPL. env_flags: WRAPPED_WELLBEING, WRAPPED_VOICEPRINT_CONSENT_IDS, WRAPPED_MODEL, CHAT_MODEL. chat_demos: 2-3 showcase prompts that exercise multiple new tools in one session (e.g. a conflict + apology flow; a "who am I drifting from and what should I say" flow). wrapped_demo: the FULL wrapped command with --wellbeing to include every section; note --no-llm for a fast deterministic render. gotchas: Wrapped previews the roster before generating (interactive unless --yes); single self-contained HTML file (no network, light/dark); wellbeing + voiceprint third-party content are default-off/redacted.`,
  },
]

const results = await parallel(AREAS.map(a => () =>
  agent(a.prompt, { label: `read:${a.label}`, phase: 'Gather', schema: DEMO_SCHEMA })
    .then(r => ({ ...r, _label: a.label, _feature: a.feature }))
))

return results.filter(Boolean)
```

---

## 5. Synthesize `DEMO.md` (target structure)

Turn the workflow's array of results into one runbook with this shape. Order matters
— setup first, then families in the same order the pipeline builds them, then the two
demo surfaces, then teardown.

1. **Title + 2-sentence intro** — what the suite is (deterministic compute + local
   gemma narration only; retrieval, never context-stuffing) and what this doc does.
2. **Quickstart: "demo everything in ~10 minutes"** — a short numbered list that
   enables the gated features in `.env`, runs `make pipeline`, then fires 3–4
   marquee chat prompts and opens Wrapped. A busy reader should be able to stop here.
3. **Phase 0 — one-time setup** (from §2): bootstrap/doctor, annotate, and the
   `.env` gate flips needed for a full demo, presented as a small table the reader
   fills in based on what they want to show.
4. **One section per family**, in pipeline order — Conflict, Nudges, Drift,
   Wellbeing, Voiceprint. (The Voiceprint section must cover V5 `simulate_contact`
   + past-self as its gated, most-guarded sub-feature: lead with the
   `PERSONA_SIM_ENABLED` enable step, show the acknowledgment→`acknowledged=true`
   flow, and treat the deceased denylist / soft dormant warning / per-line estimate
   label as intended code-enforced safety, not caveats.) Each section:
   - **What you'll see** (the `overview`).
   - **Enable it** (any `env_flags` that are off-by-default — bold the ones that
     silently no-op).
   - **Build the data** (`build_steps`, e.g. `make conflict`).
   - **Demo in chat** — a table of `prompt → what you'll see`, using the exact
     `chat_demos`. Show both the REPL and a `chat.py -p "..."` one-shot form.
   - **See it in Wrapped** (`wrapped_demo`), if any.
   - **Good to know** (`gotchas` — the honest-degrade + safety-floor behavior).
5. **The two demo surfaces**:
   - **Chat CLI** — how to start the REPL (`make chat` / `.venv/bin/python3 chat.py`),
     the one-shot form, `--model`, `--no-dv-screen`, and a couple of multi-tool
     showcase prompts.
   - **Wrapped** — the full generate command (with `--wellbeing` to include every
     section, `WRAPPED_VOICEPRINT_CONSENT_IDS` to light the voiceprint card),
     `--no-llm` for a fast deterministic render, `--year` / `--contact`, and where the
     HTML file lands + that it's fully self-contained (open in any browser, offline).
6. **Safety & privacy notes** — a short subsection consolidating the DV floor,
   per-contact wellbeing consent, and default-redacted third-party Wrapped content, so
   a demoer understands why some reads come back suppressed/empty and that this is
   intended, code-enforced behavior.
7. **Teardown / reset** — `make drift-purge` (when turning Drift off), `make reset-db`
   (wipes data, keeps schema; note `nudge_exclusions` and the committed conflict seed
   survive), and how to flip the gate flags back off.

Keep every command in a fenced block. Prefer the `make` target where one exists;
show the underlying `.venv/bin/python3 ...` form when it takes flags the make target
doesn't pass through.

---

## 6. Accuracy checklist (do before declaring done)

- [ ] Every chat-tool name in `DEMO.md` exists in `tool_schemas.py` (grep it).
- [ ] Every `make` target cited exists in the `Makefile`.
- [ ] Every env var + default matches `config.py` (don't trust memory —
      `DRIFT_ENABLED`, the two `NUDGE_*_ENABLED`, `CONFLICT_TRACE_ENABLED`,
      `PERSONA_SIM_ENABLED` are OFF; `CONFLICT_DV_SCREEN` is ON; `WRAPPED_WELLBEING`
      OFF; all consent/block lists (`WELLBEING_CONSENT_IDS`,
      `WRAPPED_VOICEPRINT_CONSENT_IDS`, `PERSONA_SIM_BLOCK_IDS`) empty).
- [ ] The gated features (Drift; the two opt-in nudge detectors; cross-chat trace;
      V5 `simulate_contact`; Wrapped wellbeing/voiceprint) each have an explicit
      "enable this first" note.
- [ ] V5 `simulate_contact` is documented with its guardrails as INTENDED behavior:
      the registration gate (`PERSONA_SIM_ENABLED`) is the boundary (not
      `acknowledged`); the per-line "[ESTIMATE …]" label is code-applied; the
      `PERSONA_SIM_BLOCK_IDS` denylist hard-refuses; the soft dormant warning never
      blocks; past-self is scan-capped; nothing sends or persists.
- [ ] The DV floor, wellbeing consent, and Wrapped redaction are described as
      intended code-enforced behavior, not caveats to apologize for.
- [ ] `make pipeline`'s serial order and the "drift no-ops unless enabled" fact are
      stated.
- **Do NOT run the test suite** (it's slow and hangs the terminal, per CLAUDE.md).
  You may run read-only greps and `make help`. `wrapped.py --no-llm` is a safe
  optional smoke check if you want to confirm the report renders, but not required.

## 7. Guardrails

- Work on `analytical-suite`. **Never touch `main`** (there is an open PR #32,
  `analytical-suite → main`, left unmerged for the owner to approve — leave it).
- `DEMO.md` is a new doc; don't commit it unless the owner asks.
- If a workflow reader returns something that contradicts the source, trust the
  source file and fix it — the runbook's whole value is that its commands actually run.
