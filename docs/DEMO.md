# DEMO.md — Close Read Analytical Suite, end-to-end runbook

This is an operator runbook for demoing every feature in the **Close Read** analytical
suite (Waves 0–4) against your **real, already-ingested** iMessage history. Everything
here is deterministic local compute narrated by a local model (`CHAT_MODEL`, by default
`qwen3.5:4b` on Ollama) — the corpus is queried by tool calls, never stuffed into a
prompt, and nothing leaves the machine.

Every step below is a copy-pasteable command or an exact chat prompt, with a one-line
"what you'll see." Where a `make` target exists it's shown first; the underlying
`.venv/bin/python3 …` form is shown when it takes flags the target doesn't pass through.

> **The single most important thing to know:** several features are **off by default**
> and will *silently do nothing* until you enable them. They fail closed on purpose
> (privacy + false-positive safety). The gate table in [Phase 0](#phase-0--one-time-setup)
> is the map; each feature section repeats its own "enable this first" note.

---

## Quickstart — demo everything in ~10 minutes

A busy reader can stop here. This enables the gated features, builds the data, fires a
few marquee prompts, and opens Wrapped.

```bash
# 1. Turn on the gated features you want to demo (writes them into .env).
make env                                  # scaffold .env if you don't have one (all values commented)
cat >> .env <<'EOF'
DRIFT_ENABLED=true
CONFLICT_TRACE_ENABLED=true
NUDGE_GOODNEWS_ENABLED=true
PERSONA_SIM_ENABLED=true
WRAPPED_WELLBEING=true
EOF

# 2. Confirm the environment, then build the analytical layer (serial — do not -j it).
make doctor                               # read the [ok]/[warn]/[fail] lines
make annotate                             # chunk annotations (distilled tier)
.venv/bin/python3 annotate.py --quality balanced --upgrade-distilled   # light up llm-valence reads
make pipeline                             # ingest→annotate→nudges→drift→voiceprint→conflict→topics

# 3. Fire the marquee chat prompts (one-shot form; drop -p "..." for the REPL).
.venv/bin/python3 chat.py -p "Who should I reach back out to? Any loose ends?"
.venv/bin/python3 chat.py -p "Have I had any tense stretches with <NAME>? Show me the conflicts."
.venv/bin/python3 chat.py -p "Who am I drifting away from? Give me a fadeout watchlist."
.venv/bin/python3 chat.py -p "What's my texting style? How do I come across?"

# 4. Build the year-in-review and open it.
.venv/bin/python3 wrapped.py --year 2025 --wellbeing --yes
open wrapped_2025_all.html
```

Notes:
- `make pipeline` will **no-op the drift step** unless `DRIFT_ENABLED=true` is set first
  (step 1 does that). The two opt-in nudge detectors stay dark unless their flags are on.
- `annotate.py --quality balanced --upgrade-distilled` is what makes conflict
  corroboration, the nudge valence filter, and voiceprint *tone* show real signal instead
  of honestly degrading to empty. Run it before demoing those.
- Replace `<NAME>` with a real contact. If a prompt returns "nothing found," it usually
  means the relevant build step or gate hasn't run — the per-feature **Good to know**
  sections explain each empty result.

---

## Phase 0 — one-time setup

You already have history ingested, so the practical path is: **confirm with `doctor`,
then annotate and build the pipeline.** The full install steps are here for completeness.

### 0a. Install / verify the environment

```bash
make install            # create .venv + pinned deps (needs Homebrew Python for sqlite-vec)
make install-ollama     # brew-install Ollama + pull every model config.py names (a few GB at
                        #   the defaults, more if you repoint CHAT_MODEL); one-time
make ollama-start       # ensure the model server is up (brew services)
make doctor             # venv, deps, Full Disk Access, Ollama + models, source chat.db, analysis DB
```

`make doctor` prints `[ok]/[warn]/[fail]` for venv, Ollama, each model, the source
`chat.db`, and the analysis DB (message / chunk / embedded-vector counts). **Run it
first** — if it reports messages/chunks/vectors present, you can skip `make source-data`
and `make ingest` entirely.

Only if `doctor` says data is missing:

```bash
make source-data        # copy the live ~/Library/Messages/chat.db snapshot (needs Full Disk Access)
make ingest             # snapshot → normalize → chunk → embed (the long pole; resumable)
```

### 0b. Build the derived analytics every feature reads

```bash
make annotate           # LLM chunk annotations (distilled/kNN tier by default) → chunk_annotations
make annotate-hybrid    # LLM upgrades on a wall-clock budget (ANNOTATE_BUDGET_MINUTES); does NOT chain --terms
.venv/bin/python3 annotate.py --quality balanced --upgrade-distilled   # upgrade top chats to source='llm'
make pipeline           # ingest → annotate → suite → topics (serial)
make suite              # just the four suite features (nudges → drift → voiceprint → conflict), no re-ingest
```

**Why `make pipeline` is strictly serial:** every step holds the SQLite write lock across
many LLM calls (concurrent runs die with `database is locked`) and contends for one
serialized Ollama instance. It is written to stay serial even under `make -j` — never
parallelize it; parallelism costs correctness and buys nothing.

**Why the `--upgrade-distilled` line matters:** the default `make annotate` writes
`source='distilled'` rows (fast kNN classification, empty `events_json`). Valence-dependent
reads trust **only** `source='llm'` rows. Until you upgrade the chats you care about,
conflict corroboration reads `'none'`, the nudge Stage-0 valence exclusion doesn't fire,
and voiceprint `tone_*` is NULL. This is honest-degrade, not a bug — see each feature's
**Good to know**.

Every `--quality balanced --upgrade-distilled` line below is the *exhaustive* way to do
that, and it runs for hours. `make annotate-hybrid` (= `annotate.py --quality hybrid`) is
the bounded-time
alternative: it upgrades distilled rows until `ANNOTATE_BUDGET_MINUTES` is spent — event
candidates first, then the least-confident rows — so the same reads light up for the
chunks that matter most rather than for everything. (Unlike `make annotate` it does *not*
chain the `--terms` pass; that pipeline is unbudgeted and would break the wall-clock
promise.) Substitute it anywhere below when you
want a demo-able run instead of an overnight one. `--limit N` there caps the upgrades
actually **written**, not the requests sent: a reply the model came back unparseable on
spent budget but produced no annotation and doesn't consume the cap.

Leave the budget at its 25-minute
default for a demo: one LLM request is the smallest thing the tier can buy (a request
holds 4–6 event-candidate chunks and banked at 46.7s on the reference machine), so on a
1- or 2-minute budget the LLM **upgrade** phases decline to start work they cannot finish
and print why, naming the knob. That is **not** a no-op run: the seed top-up and the
corpus-wide kNN classify + confidence backfill are deadline-gated, not skipped — they run
and **commit whatever they reach**, and the next run resumes from there (measured: a
1-minute budget landed at exactly 1.0 min with the confidence backfill stopping at 15,616
of 20,613 rows; the next run finished the remaining 4,997 **and reached phase 3a**). A
tiny budget therefore buys
distilled coverage, just no LLM upgrades.

**Changed 2026-07-28, and it affects how long these runs take:** the `distilled` / `fast`
/ `balanced` presets carry a reasoning-*effort* level (`low`), which only `gpt-oss`-family
models understand as "think less" — to every other model it is merely truthy and buys a
*full* reasoning pass. `annotate.py` now translates it (`config.has_effort_levels`), so on
the default `qwen3.5:4b` — and on `gemma4`, where those tiers previously did do a full
pass — the three tiers send `think=false`. Substantially faster, one fewer reasoning pass.
`max` is unchanged (it omits the flag on every model), and an explicit `ANNOTATE_THINK`
still overrides the translation.

### 0c. The gate cheat-sheet — flip these in `.env` *before* building/demoing

`make env` scaffolds `.env` from `.env.example` (all values commented, every default
works). Set the gates for whatever you intend to show. Verified against `config.py`:

| Env var | Default | Set it when you want to… |
|---|---|---|
| **`DRIFT_ENABLED`** | **off** | Demo *any* Drift surface. Off = `make drift` no-ops, `dyad_drift` never builds, the 3 drift read tools + Wrapped "Who's Drifting" are all withheld. |
| **`CONFLICT_TRACE_ENABLED`** | **off** | Register `trace_conflict_across_chats` in the chat tool list. Off = the tool doesn't exist to the model. |
| **`NUDGE_GOODNEWS_ENABLED`** | **off** | Enable the `unreciprocated_goodnews` detector. |
| **`NUDGE_DROPPED_PLAN_ENABLED`** | **off** | Enable the `dropped_plan` detector. **Even on, it writes nothing** until you hand-label + clear its F4 precision gate (ships no seed). |
| **`PERSONA_SIM_ENABLED`** | **off** | Demo `simulate_contact` (V5). Off = schema withheld; the model can't call it. This registration gate *is* the security boundary. |
| **`WRAPPED_WELLBEING`** | **off** | Compose the Wrapped "Gentle Check-In" section (same as `wrapped.py --wellbeing`). |
| `CONFLICT_DV_SCREEN` | **on** | Leave on. It's the softened DV floor (details in Conflict section). |
| `WELLBEING_DENY_IDS` | empty | Deny list of `contacts.id` for *per-contact* wellbeing reads. Default-open: empty analyzes anyone; list an id to suppress it (self is never affected). |
| `WRAPPED_VOICEPRINT_CONSENT_IDS` | empty | Light the third-party "guess who texts like this" Wrapped card for listed ids. |
| `PERSONA_SIM_BLOCK_IDS` | empty | Hard-block specific `contacts.id` from `simulate_contact` (deceased/grief denylist). |

Off-by-default features silently no-op — the **bold** rows above are the ones a demoer
most often forgets.

---

## Feature 1 — Clearing the Air (Conflict)

**What you'll see.** A deterministic finder detects tense stretches in your **1:1** chats
from signals-with-receipts (reply-latency withdrawal, one-sided sessions, dislike-tapback
runs, volume collapse) and writes them to `conflicts` / `conflict_signals`. In chat you can
list conflicts, open one, get a deterministic **scorecard** with a non-blame
demand-withdraw read, and get a **Lewicki-6 apology scaffold** grounded in your real
grievance excerpts. A code-enforced DV floor watches the window text.

**Enable it.** No enable flag for the core feature. Two optional flags:
- **`CONFLICT_TRACE_ENABLED=true`** — required only for the cross-chat trace tool.
- `CONFLICT_DV_SCREEN` is **on** by default (leave it); `chat.py --no-dv-screen` bypasses
  the role read for one run.

**Build the data.** `make conflict` is two-phase and budgeted in **wall clock**, like
`--quality hybrid`: it detects windows across every 1:1 chat (~24 s on the reference
corpus), then labels the *strongest* windows first until `CONFLICT_BUDGET_MINUTES`
(default **25**) is spent, and prints how many it labeled versus left. One run does not
label the whole corpus — the reference run labeled **211 of 1,305** detected windows in
24m53s (measured 2026-07-28, *before* the length prescreen was demoted to a ranking
preference: windows whose turns were all under 15 characters used to be stamped for free
and now cost real requests, so expect somewhat fewer windows per run — honestly labeled
rather than silently skipped. Re-measure before quoting the number again) — but a rerun
resumes down the strength ranking, so coverage grows a run at a time.

```bash
make conflict                                  # detect everywhere, then label the strongest windows for ~25 min
.venv/bin/python3 conflict_detect.py --budget-minutes 3  # shorter run; resumes where the last one stopped
make conflict-fast                             # deterministic windows only (--no-llm): zero Ollama calls, unchanged
.venv/bin/python3 conflict_detect.py --dry-run # optional: how many 1:1 chats will be scanned, no writes
.venv/bin/python3 conflict_detect.py --limit 5 # optional faster demo: first 5 direct chats
.venv/bin/python3 conflict_detect.py --relabel # start labeling over: clears signals + topic + the labeled_at stamp
```

`--budget-minutes 0` drops the deadline entirely (label every still-unlabeled window,
however long it takes); `--budget-minutes` with `--no-llm` is an error, since that path
makes no LLM calls to bound; a *negative* value is refused outright, naming the knob it
came from, rather than clamped to 0 (which would mean unbudgeted). A very small budget can
legitimately dispatch *nothing* —
one window is the smallest unit of work — and it says so, naming the knob, rather than
looking like "nothing to do".

A window is labeled once and then skipped by every later run (`conflicts.labeled_at` is
the resume marker), so if you retune `CONFLICT_MAX_LABEL_TURNS` / `CONFLICT_MIN_TURN_CHARS`
or clear a new eval-gate class, `--relabel` is what makes the pass revisit them. It is
**not** the model-switch door: `--relabel --model <other>` is refused up front, with every
label the previous budgets bought still intact (the model guard runs *before* the
deletion — and before the detection sweep, which drops superseded windows together with
their signal rows, so a plain `--model <other>` is refused up front too). `--no-llm` does
not exempt the clear from that check; it skips the labeling, not the deletion. Switching
label models still means emptying `conflicts` or `make reset-db`.

Run `annotate.py --quality balanced --upgrade-distilled` first if you want
`corroboration='weak'/'llm'` instead of `'none'`. Detection itself is never blocked by
missing annotations.

**Demo in chat** (REPL prompts on the left; one-shot form is
`.venv/bin/python3 chat.py -p "…"`):

| Prompt | Tool | What you'll see |
|---|---|---|
| `Have I had any tense stretches with <NAME>? Show me the conflicts.` | `find_conflicts` | Ranked conflict episodes for that person's 1:1 chats — each with a `conflict_id`, dates, which signals fired, corroboration, `off_text_flag`. Empty if `make conflict` hasn't run or it's a group chat. |
| `Open conflict 3 and show me what happened.` | `get_conflict` | One episode's evidence + its labeled turns rendered as **tentative/likely bands** (never raw scores), each with actor me/them and a verbatim excerpt ≤200 chars. At most `CONFLICT_MAX_LABEL_TURNS` (**6**) turns, peak-anchored on the window's longest unanswered gap rather than its first few messages — and **zero** in two cases the tool words differently: a window no run's budget has reached (`labeled` false = NOT ANALYZED), and one that *was* labeled but produced no shippable turn (`labeled` true, 0 turns — labeling reads message text only). Neither is a finding of calm. |
| `Break down conflict 3 — who withdrew, who kept reaching out, and the tone on each side.` | `diagnose_conflict` | A deterministic scorecard: per-side I-vs-you / we-talk ratios, LSM match, opener/last-word roles, peak latency + volume-collapse read, and a **non-blame** demand-withdraw (pursuer vs withdrawer) read. Carries **no** aggressor/who-won verdict. |
| `I want to apologize to <NAME> about conflict 3. Help me write it.` | `draft_apology` | A **scaffold**, not a finished apology: the ranked Lewicki-6 checklist + your real grievance excerpts pulled verbatim. It won't ghostwrite prose unless you ask again. |
| `Did my fight with <NAME> (conflict 3) spill into other conversations?` | `trace_conflict_across_chats` | **Only exists if `CONFLICT_TRACE_ENABLED=true`.** The model asks first, then calls with `consent=true`; each link both clears a similarity threshold and literally names the partner. Nothing persisted. |

One-shot example:

```bash
.venv/bin/python3 chat.py -p "Have I had any tense stretches with <NAME>? Show me the conflicts, then diagnose the most recent one."
```

**See it in Wrapped.**

```bash
.venv/bin/python3 wrapped.py --no-llm --yes    # open the HTML, scroll to "Repair Report"
```

The deterministic finder facts always render (window count, typical cool-off time,
windows-with-volume-collapse, a windows-over-time chart) — detection is corpus-wide in a
single run, so those are complete after one `make conflict`. Label-derived stats (top
horseman, most-repaired relationship, recovery-time trend) need **two** things now: an
eval-gate-cleared class *and* a window some run's labeling budget actually reached. One
run covers the strongest couple hundred windows, so these thicken with each rerun instead
of arriving complete. Scope one relationship with `--contact <NAME>`; drop `--no-llm` for
narration.
The section is **skipped entirely** (no empty scaffolding) if there's no conflict data in
scope — so run `make conflict` first.

**Good to know** (intended, code-enforced behavior — not caveats to apologize for):
- **Conflicts are 1:1 only** by construction. A group-chat prompt yields nothing.
- **No aggressor/role/who-won column exists, by design.** The demand-withdraw read is a
  mechanical pattern, not a fault verdict — present it as such.
- **The softened DV floor** (code-enforced, never by prompt): on a high-risk marker with
  the screen on (default), `diagnose_conflict` withholds **only** the role read (the
  mechanical scorecard still returns), and `draft_apology` withholds the **entire** scaffold
  (drafting an apology can pressure a victim into apologizing to an abuser). With the screen
  off (`CONFLICT_DV_SCREEN=off` or `--no-dv-screen`) the role read/scaffold is restored, but
  the hotline **resource footer is still appended** — it is never removed. The scan is
  read-only; the hit is never persisted.
- **A detected window with no labeled turns is not a bug — in two different ways.**
  Detection is corpus-wide; labeling is bought with a wall-clock budget, strongest windows
  first. `find_conflicts` will happily list a window whose `get_conflict` shows zero turns
  and says NOT ANALYZED — rerun `make conflict` to extend coverage down the ranking
  (already-labeled windows are never re-bought). Separately, a window that **was** labeled
  can still produce zero signal turns, and the tools say so in their own words: labeling
  reads message **text** only, so a stretch carried by tapbacks, by tone, or moved
  off-channel yields nothing to label, and a turn can also fall under
  `CONFLICT_LABEL_CONF_FLOOR` or be suppressed by the eval gate. Three states, not two:
  never looked at, looked at with nothing shippable, and labeled with bands. Read the
  first two off their deterministic evidence, which is complete either way.
- `corroboration='none'` on an all-distilled corpus is correct — light it with the
  `--upgrade-distilled` annotate pass.
- Escalation labels are eval-gated bands. `conflict_labels` ships a **committed
  pre-cleared seed**, so the horsemen + repair classes work out of the box and survive
  `reset_db`.

---

## Feature 2 — Loose Ends (Nudges)

**What you'll see.** A scan of your 1:1 history for relationships you let drift, written
to `loose_threads` as urgency-ranked nudges. Six detectors: `faded_initiator` (they used
to start conversations and stopped), `dormant_tie` (a once-close tie gone quiet — **hidden
until you opt in**), `unanswered_question`, `inferred_occasion` (a recurring
birthday/anniversary — the only detector that fills `due_by` so it reminds you *before* the
date), plus two off-by-default detectors: `dropped_plan` and `unreciprocated_goodnews`.

**Enable it.** The four always-on detectors need no flag. For the two opt-in detectors:
- **`NUDGE_GOODNEWS_ENABLED=true`** — enables `unreciprocated_goodnews` (reads
  `source='llm'` events; needs the `--upgrade-distilled` annotate pass or it finds nothing).
- **`NUDGE_DROPPED_PLAN_ENABLED=true`** — enables `dropped_plan`, but it is **double-gated**:
  it also needs the F4 `dropped_plan` precision eval-gate cleared, and it **ships no seed**,
  so it writes nothing until you hand-label + gate it:
  ```bash
  .venv/bin/python3 -m eval.evalkit label dropped_plan
  .venv/bin/python3 -m eval.evalkit gate  dropped_plan
  ```

**Build the data.**

```bash
make nudges                                       # populate loose_threads (Ollama only for one-line nudge_text)
.venv/bin/python3 loose_ends.py --dry-run         # optional: candidates/suppressed/written, no writes/Ollama
.venv/bin/python3 loose_ends.py --acknowledge-reconnect   # ONE-TIME opt-in to reveal dormant_tie nudges
.venv/bin/python3 loose_ends.py --no-llm          # detect only: SQL detectors, no nudge_text narration
.venv/bin/python3 loose_ends.py --limit 20        # cap NARRATED threads (detection is unaffected); --model M overrides NUDGES_MODEL
```

`make nudges` must run **after** `make annotate` (the Stage-0 valence exclusion reads
`chunk_annotations`).

**Demo in chat:**

| Prompt | Tool | What you'll see |
|---|---|---|
| `Who should I reach back out to? Any loose ends I've been neglecting?` | `get_loose_threads` | Up to 5 urgency-ranked nudges: contact, detector kind, anchor date, one-line `nudge_text`. `dormant_tie` rows are hidden unless you ran `--acknowledge-reconnect`. |
| `Show me only the people who used to start our conversations and went quiet.` | `get_loose_threads` | Narrowed to `kind='faded_initiator'`. |
| `Whose birthday or anniversary is coming up that I usually message?` | `get_loose_threads` | `kind='inferred_occasion'` rows, each carrying a `due_by` so the reminder lands before the date. |
| `I've handled the one with <NAME> — dismiss it so it stops resurfacing.` | `resolve_loose_thread` | Sets `status='dismissed'`, stamps `dismissed_at`, never deletes; the detector won't re-open it. |
| `Stop nudging me about <NAME> entirely.` | `mute_contact_from_nudges` | Writes a durable `nudge_exclusions` row and dismisses their open threads; no detector generates candidates for them again. |
| `Let me review the nudges I already dismissed.` | `get_loose_threads` | `status='dismissed'` — proves dismissed rows are retained as a record. |

One-shot example:

```bash
.venv/bin/python3 chat.py -p "Who should I reach back out to? Any loose ends I've been neglecting?"
```

**See it in Wrapped.**

```bash
.venv/bin/python3 wrapped.py --yes    # open the HTML, scroll to "Open Loops"
```

Composes from `get_loose_threads(status='open')`, so it's already post-Stage-0 (muted
contacts never appear) and additionally drops silent/excluded names. Shows "no open loops
in scope" if empty. Requires `make nudges` first.

**Good to know:**
- **`dormant_tie` is written but hidden by default** — that's why reconnect nudges seem
  missing until you run `loose_ends.py --acknowledge-reconnect` once.
- **Dismissed rows are kept, never deleted.** A dismissed `(kind, chat_id)` is what stops
  the detector re-opening it — re-running `make nudges` won't resurrect it.
- **`nudge_exclusions` (mutes) survive `reset_db`** by design — muting a contact outlives a
  DB rebuild. `loose_threads` itself *is* cleared by `reset_db`.
- `unreciprocated_goodnews` reads `source='llm'` events only — on a distilled-only corpus
  it finds nothing (light it with `--upgrade-distilled`).
- `get_loose_threads` is hard-capped at `NUDGE_MAX_RESULTS` (5) — asking for "all" still
  returns at most 5.

---

## Feature 3 — Fadeout Radar (Drift)

**What you'll see.** Drift scores every 1:1 relationship's trend over time
(thriving/steady/cooling/fading/dormant) from seven monthly signals into `dyad_drift`.
Three chat tools expose it: `get_relationship_drift` (one tie's trajectory),
`get_fadeout_watchlist` (a coverage-partitioned leaderboard of fading ties), and
`get_reconnection_candidates` (curated shared references before drafting a reconnection
opener). Wrapped adds a "Who's Drifting" section.

**Enable it — the entire feature is off by default.**

> **`DRIFT_ENABLED=true` is the master gate.** Off (the default) means: `make drift`
> no-ops, `dyad_drift` never builds, **and** all three read tools are withheld from the chat
> model's schemas, **and** the Wrapped section is skipped. Nothing in this family works until
> it's on. Because the tool schemas are gated at registration, set the flag for the
> `chat.py` and `wrapped.py` processes too (put it in `.env` and it applies everywhere).

```bash
# STEP 1 — flip the master gate on (idempotent .env edit):
grep -q '^DRIFT_ENABLED=' .env && sed -i '' 's/^DRIFT_ENABLED=.*/DRIFT_ENABLED=true/' .env || echo 'DRIFT_ENABLED=true' >> .env

# STEP 2 — build dyad_drift (deterministic, no Ollama):
make drift
.venv/bin/python3 drift.py --dry-run   # optional: how many dyads WOULD be scored, no writes
.venv/bin/python3 drift.py --force     # force a rebuild even if corpus/params unchanged
```

`make drift` prints e.g. `Built dyad_drift: N dyads scored`. A dyad is scored only if it
clears `DRIFT_MIN_MESSAGES` (100) and `DRIFT_MIN_MONTHS` (4) — if you get 0 dyads on a
small corpus, lower those in `.env`.

**Demo in chat** (all require `DRIFT_ENABLED=true` for the process running `chat.py`):

| Prompt | Tool | What you'll see |
|---|---|---|
| `Are <NAME> and I drifting apart? How is my relationship with them trending?` | `get_relationship_drift` | Phase (thriving…dormant), a drift score, the ranked contributing signals with monthly evidence, current gap vs typical, and a confidence tier. On a cooling/fading/dormant tie it **first relays the channel caveat** ("this read is iMessage-only — do you still talk elsewhere?"). |
| `Who am I drifting away from? Show me a watchlist of relationships fading out.` | `get_fadeout_watchlist` | A leaderboard ranked **within the mandatory-complete coverage tier only**, plus a separate `insufficient_signal` bucket that must not be compared against it. |
| `Which of my fading friendships are dormant?` | `get_fadeout_watchlist` | Same, with a `phase_filter` narrowing to `dormant`. |
| `Help me reach back out to <NAME> — give me something real to reference.` | `get_reconnection_candidates` | Curated shared references (human-verdict inside jokes/nicknames + high-valence `source='llm'` memories); asks you to **pick one** before any opener is drafted. If none exist it **refuses to invent one** — that refusal is correct behavior. |

**See it in Wrapped** (also needs the flag):

```bash
DRIFT_ENABLED=true .venv/bin/python3 wrapped.py --no-llm --yes   # scroll to "Who's Drifting"
```

Every number is read verbatim from `dyad_drift` (no re-scoring, no Ollama). Missing from
the report ⇒ the flag was off for the `wrapped.py` process.

**Good to know:**
- **Schema-gated, not prompt-gated.** With the flag off, the tools are absent from the
  model entirely. Verify with `.venv/bin/python3 -c "import config; print(config.DRIFT_ENABLED)"`.
- **The channel caveat is code-emitted**, not a prompt hope — every cooling/fading/dormant
  payload carries it; iMessage is one channel.
- **The watchlist never ranks across coverage** — a low-coverage dyad's high score can't
  outrank a well-covered one.
- `get_reconnection_candidates` memories read `source='llm'` only; curated terms need
  `annotate.py --terms`. A distilled-only/unlabeled dyad yields no candidates and the tool
  refuses rather than inventing one.
- **Teardown:** `make drift-purge` wipes `dyad_drift` + its guard keys **regardless of the
  flag** — it's the teardown you run *while* turning the feature off. It also reminds you to
  set `DRIFT_ENABLED=false` yourself (a process can't unset your env var).

---

## Feature 4 — Baseline (Wellbeing)

**What you'll see.** A deterministic, LLM-free "language trajectory" read:
`get_language_trajectory` scores **your own** texting across your whole history for a
sustained, cross-family departure (rising first-person/absolutist/negation language, falling
laughs/emoji, more late-night messaging, dropping outbound initiation) — all from the
lexicon/sessions/messages tables built at ingest. **Self is the default.** When a departure
fires, `chat.py` scrubs the reply's final sentence of any clinical-condition name and prints
a fixed crisis-support block verbatim after the reply.

**Enable it.**
- Self read needs **no** flag and **no** extra build step — it reads tables `ingest.py`
  already built.
- A **per-contact** read runs by default and is **refused in code** only when that
  `contacts.id` is on the **`WELLBEING_DENY_IDS`** deny list (empty by default).
- The Wrapped section needs **`WRAPPED_WELLBEING=true`** (or `wrapped.py --wellbeing`).

**Check whether your history fires a window** before demoing (firing is deliberately hard —
many healthy corpora never flag, which is correct):

```bash
.venv/bin/python3 -c "import query_tools, json; print(json.dumps(query_tools.get_language_trajectory(include=['windows','baseline']), indent=2)[:1500])"
```

An empty `windows: []` means no sustained departure — honest-degrade, not a failure.

**Demo in chat:**

| Prompt | Tool | What you'll see |
|---|---|---|
| `How has my own texting language and mood shifted over time? Any sustained rough stretches?` | `get_language_trajectory` | **Self mode** (the default). Narrates only the computed markers/window — never a diagnosis. If a window fired: the reply's final sentence is scrubbed of any clinical term, and a fixed support block (988 / text HOME to 741741 / findahelpline.com) is printed verbatim **after** the reply, by code. If nothing fired, it says so honestly. |
| `Run a language trajectory check on my messages with <NAME>.` | `get_language_trajectory` | Runs by default (default-open) against `<NAME>`'s own lines, with non-clinical narration + the code-emitted support block if a shift fires. |
| *(after setting `WELLBEING_DENY_IDS=42`)* `Check contact 42's language trajectory.` | `get_language_trajectory` | **Refused in code** *before any analysis* — contact 42 is on the deny list — and never hints who else is on it. Self is never affected. |

One-shot example:

```bash
.venv/bin/python3 chat.py -p "How has my own texting language and mood shifted over time?"
```

**See it in Wrapped.**

```bash
.venv/bin/python3 wrapped.py --wellbeing --year 2025 --yes    # or set WRAPPED_WELLBEING=true
```

The "A Gentle Check-In" section appears **only if your self trajectory fires a window**;
otherwise it's silently omitted and `wrapped.py` prints an "available but off" note. Its
LLM blurb is scrubbed by the same `sanitize_condition_names` pass.

**Good to know:**
- **Per-contact consent is enforced in code**, not by prompt — refusal happens before any
  analysis and reveals nothing about the allowlist.
- **Two framing mechanisms fire only on a flagged result:** (a) the reply's final sentence
  is buffered and run through `sanitize_condition_names` (any sentence naming
  depression/anxiety/bipolar/etc. is replaced wholesale) before it's emitted; (b) the
  code-emitted support block is stripped from the data the model sees and printed verbatim
  after the reply. Both are no-ops when nothing fired.
- **Firing is intentionally hard** (false positives are the enemy): it needs a sustained
  (≥3-month), change-point-gated, cross-family departure vs your own leave-one-out baseline,
  with ≥12 baseline months and ≥5000 words in scope. Returning "no departure" is the common,
  correct outcome.
- If the F1 markers read NULL on a stale DB, re-ingest — those lexicon columns fill on a
  full rebuild.

---

## Feature 5 — Voiceprint (Persona)

**What you'll see.** A deterministic per-person "voiceprint" (message length, casing,
exclaim/ellipsis rates, burst size, 9 function-word rates, signature openers/closers, plus a
low-trust LLM tone block) in `person_voiceprint`. In chat you can ask for your or someone's
style (`get_voiceprint`), draft a reply in your own voice grounded in real exemplars
(`draft_reply`), deterministically score whether a draft "sounds like" you
(`check_style_match`), and — off by default — get a clearly-labeled statistical **estimate**
of how a contact might reply or how your past self texted (`simulate_contact`, V5).

**Enable it.**
- `get_voiceprint` / `draft_reply` / `check_style_match` need only `make voiceprint`.
- **`PERSONA_SIM_ENABLED=true`** is required for `simulate_contact` (V5). Off = the schema is
  withheld and the model **cannot** call it — this registration gate is the real security
  boundary, not the per-call `acknowledged` flag.
- `PERSONA_SIM_BLOCK_IDS` (empty, default-allow) hard-blocks specific `contacts.id`.

**Build the data.**

```bash
.venv/bin/python3 annotate.py --quality hybrid      # optional, FIRST: light the low-trust tone_* fields
                                                    # (--quality balanced --upgrade-distilled also works)
make voiceprint                                     # populate person_voiceprint (deterministic, no Ollama)
.venv/bin/python3 voiceprint.py --dry-run           # optional: how many rows would be built, no writes
.venv/bin/python3 voiceprint.py --force             # force a rebuild the guard would skip (see below)
```

Annotate FIRST: the rebuild guard watches new messages *and* the `source='llm'`
annotations `tone_*` is built from, so an annotate pass on its own is enough to
make the next `make voiceprint` rebuild. Running it the other way round leaves
the tone fields empty until you build again. `--force` is for the inputs the
guard still does not watch — a `SESSION_GAP_MINUTES` retune (which moves every
opener/closer), a lexicon rebuild, a `VOICEPRINT_*` knob change, or an
AddressBook re-ingest that re-keys a handle to a contact.

**Demo in chat:**

| Prompt | Tool | What you'll see |
|---|---|---|
| `What's my texting style? How do I come across?` | `get_voiceprint` | Your overall card — avg length, lowercase ratio, exclaim/ellipsis rates, burst size, function-word rates, signature openers/closers, a high/low confidence flag. The `tone` block appears only if you have `source='llm'` annotations. |
| `How does <NAME> text? Describe their style with me.` | `get_voiceprint` | Their card (or your voice *with* them). Ambiguous names return candidate ids to disambiguate; unbuilt table returns a "run make voiceprint" note. |
| `Draft a reply to <NAME> in my voice — they just texted "you still coming friday?"` | `draft_reply` | Retrieves real (their message → your reply) exemplars, generates **one** reply in your voice, strips ungrounded inside-jokes. **Declines structurally** below `VOICEPRINT_MIN_CONTEXT_PAIRS` (4) exemplars: `draft=null`, `confidence='low'`, returns your recent replies instead — zero generation. |
| `Does this draft sound like me to <NAME>? "hey!! omg yes absolutely, cannot wait :)"` | `check_style_match` | Deterministic, no model: z-normalizes the draft's features against the stored corpus std (`feature_std_json`, no rescan), returns a distance (0 = identical voice) + top divergences with rule-based explanations. |

**V5 `simulate_contact` — the gated, most-guarded sub-feature.** Lead with the enable step;
its guardrails are **intended, code-enforced safety**, not caveats:

```bash
# In .env, then restart chat.py:
PERSONA_SIM_ENABLED=true
```

| Prompt | What you'll see |
|---|---|
| `Estimate how <NAME> would reply to "want to grab dinner this weekend?"` | Only works with `PERSONA_SIM_ENABLED=true`. The **first** call returns `acknowledgment_required=true` (plus a `deceased_warning` if the contact looks dormant); the model relays it, confirms this is an **estimate**, then re-calls with `acknowledged=true`. The returned `simulation` is code-labeled `[ESTIMATE of <NAME>'s voice — not their real words]` on **every line**, and carries a base-rate `likely_tapback` prediction. Declines below 4 context pairs. |
| `Talk to my past self as of 2020-01-01 — what would I have said to "how's the new job going?"` | `subject='me'`, `as_of_date='2020-01-01'`. A bounded date-scoped rescan of how *you* actually replied up to that date; gated at `VOICEPRINT_MIN_RANGE_MESSAGES` (200) and capped at `VOICEPRINT_SCAN_MAX_MESSAGES` (20000). Each line labeled `[ESTIMATE of your past self as of … — not a real message]`. |

**See it in Wrapped.**

```bash
.venv/bin/python3 wrapped.py --yes    # your own voiceprint card renders in iMessage bubbles

# To light the consent-gated "guess who texts like this" third-party game, list contact ids:
WRAPPED_VOICEPRINT_CONSENT_IDS=<id[,id2]> .venv/bin/python3 wrapped.py --yes
```

`simulate_contact` is **chat-only** — it has no Wrapped section.

**Good to know:**
- **`tone_*` fields are low-trust** (read `source='llm'` chunks only). Distilled-only ⇒ NULL
  tone, and the card omits the tone block rather than reporting a spurious neutral.
- **`draft_reply` / `simulate_contact` decline structurally** below 4 similar exemplars —
  enforced in code, expected on thin dyads. Both are **1:1 only**; nothing auto-sends (there
  is no send path) and nothing is persisted.
- **The registration gate (`PERSONA_SIM_ENABLED`) is the boundary**, not `acknowledged`
  (a UX flow). The per-line `[ESTIMATE …]` label is applied **in code** post-generation —
  there is no unlabeled version for the model to strip.
- **`PERSONA_SIM_BLOCK_IDS` hard-refuses** a listed contact in code (fixed compassionate
  message, no generation). The soft dormant heuristic only **warns**, never blocks —
  fail-open, with the manual denylist as the floor.
- **The Wrapped third-party card is default-redacted** — empty `WRAPPED_VOICEPRINT_CONSENT_IDS`
  ⇒ your own card renders but no third-party verbatim content at all.

---

## The two demo surfaces

### Chat CLI

```bash
make chat                              # interactive REPL (echoes each [tool call])
.venv/bin/python3 chat.py              # same, directly
.venv/bin/python3 chat.py -p "…"       # one-shot: prints only the reply, exit 1 if Ollama is down
make chat ARGS='-p "…"'                # one-shot via make (pre-checks Ollama)
.venv/bin/python3 chat.py -m gemma4:26b-a4b-it-qat    # override the chat model (-m / --model)
.venv/bin/python3 chat.py --no-dv-screen -p "…"       # restore the conflict role read for this run only
```

The REPL registers **32 tools by default**; **5 more** register only behind flags (3 drift
read tools + `get_reconnection_candidates`… i.e. the 3 drift tools behind `DRIFT_ENABLED`,
`trace_conflict_across_chats` behind `CONFLICT_TRACE_ENABLED`, and `simulate_contact` behind
`PERSONA_SIM_ENABLED`). A disabled tool routes nowhere — prompting won't summon it.

**Multi-tool showcase prompts** (one session chains several new tools):

```bash
# Conflict → diagnose → apology, in one turn:
.venv/bin/python3 chat.py -p "We had a fight in my chat with <NAME> last month — find it, then walk me through how to make it right."

# Voiceprint → drafted reply:
.venv/bin/python3 chat.py -p "How does <NAME> text — length, casing, openers/closers — and draft a reply from me saying I can't make dinner Friday."

# Drift → reconnection (needs DRIFT_ENABLED=true):
.venv/bin/python3 chat.py -p "Who am I drifting away from, and help me write a message to reconnect with the top one."
```

### Wrapped

```bash
.venv/bin/python3 wrapped.py --year 2025                        # canonical; previews roster, asks to proceed
.venv/bin/python3 wrapped.py --year 2025 --wellbeing --yes      # include EVERY section, skip the prompt
.venv/bin/python3 wrapped.py --year 2025 --yes --no-llm         # fast, deterministic, no Ollama
.venv/bin/python3 wrapped.py --contact "<NAME>" --yes           # single-relationship deep-dive
make wrapped ARGS='--year 2025 --yes'                           # via make (pre-checks Ollama)
open wrapped_2025_all.html                                      # self-contained; double-click also works
```

- **Interactive by default:** it prints the featured roster + any silent-contact
  auto-exclusions, then blocks on `Proceed? [y/N]`. Pass `--yes` to skip; anything but `y`
  aborts and writes nothing.
- **Silent-contact auto-exclusion** is a safety guard (drops abruptly-gone-quiet contacts
  from the roster *and* carves their chats out of every section). Re-admit one with
  `--include "<NAME>"`.
- Output is **one self-contained HTML file** — inline CSS/JS, no network, light/dark,
  scroll-reveal. Filename: `wrapped_<year|all>_<scope>.html` in the CWD; override with
  `--out PATH`.
- `--wellbeing` opts the wellbeing section in; `WRAPPED_VOICEPRINT_CONSENT_IDS` lights the
  third-party voiceprint card. Both are default-off/redacted.

---

## Safety & privacy notes (why some reads come back empty — on purpose)

These are **intended, code-enforced** behaviors. When a demo surface returns
suppressed/empty output, this is usually why:

- **DV floor (Conflict).** A high-risk marker in a conflict window suppresses the
  demand-withdraw/role read (`diagnose_conflict` still returns the mechanical scorecard) and
  withholds the entire `draft_apology` scaffold. The hotline resource footer is **never**
  removed, even when the screen is bypassed. Read-only; never persisted.
- **Per-contact wellbeing deny list.** `get_language_trajectory` analyzes any contact by
  default and refuses a contact on `WELLBEING_DENY_IDS`, *before* any analysis, revealing
  nothing about the list. Self is never affected.
- **Default-redacted third-party Wrapped content.** The wellbeing section
  (`WRAPPED_WELLBEING`) and the third-party voiceprint card
  (`WRAPPED_VOICEPRINT_CONSENT_IDS`) emit nothing unless explicitly enabled/allowlisted.
- **`simulate_contact` guardrails.** The registration gate (`PERSONA_SIM_ENABLED`) is the
  boundary; every simulated line is code-labeled an estimate; `PERSONA_SIM_BLOCK_IDS`
  hard-refuses listed contacts; nothing sends or persists.
- **Honest degrade on distilled-only annotations.** Valence-dependent reads trust only
  `source='llm'` rows and degrade truthfully (conflict `corroboration='none'`, no nudge
  valence exclusion, NULL voiceprint tone) rather than fabricating. Light them with
  `annotate.py --quality balanced --upgrade-distilled` (or `--quality hybrid` for as
  many as a wall-clock budget buys), then rerun `make voiceprint` — its rebuild guard
  watches those annotations, so the run picks them up without `--force`.

---

## Teardown / reset

```bash
make drift-purge        # wipe dyad_drift + its guard keys (run WHILE turning Drift off)
make reset-db           # wipe derived data, KEEP schema
```

After `make drift-purge`, set `DRIFT_ENABLED=false` in `.env` yourself (a process can't unset
your env var), and flip any other gate flags (`CONFLICT_TRACE_ENABLED`,
`NUDGE_GOODNEWS_ENABLED`, `PERSONA_SIM_ENABLED`, `WRAPPED_WELLBEING`) back off if you enabled
them for the demo.

`make reset-db` keeps the schema but clears derived data. Two things **survive** it by
design: `nudge_exclusions` (your contact mutes) and the committed
`eval/seed_gates/conflict_labels.json` seed (so conflict escalation labels stay pre-cleared
on a fresh DB).

---

*Every command above was verified against the source on the `analytical-suite` integration
branch, which now includes the V5 `simulate_contact` work (RES-69, PR #33, merged
2026-07-21): tool names against `tool_schemas.py`, targets against the `Makefile`, env
defaults against `config.py`.*
