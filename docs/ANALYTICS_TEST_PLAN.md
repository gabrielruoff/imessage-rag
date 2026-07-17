# Analytics features — LLM response-quality eval plan

> **Status: mechanized.** This spec is implemented as the evalkit harness
> (`eval/evalkit/`, suites S1–S8 in `eval/cases/`) — run evals there, per
> [`eval/README.md`](../eval/README.md); this document remains the spec and
> grading rationale.

Design rationale for the features under test lives in
[`DECISIONS.md`](DECISIONS.md) ("the analytics layer"); this document is the
forward plan for evaluating them.

**Goal:** measure the accuracy and quality of the LLM's end-to-end responses
for every analytics feature, so the chat app (tool descriptions, system
prompt, result formats) and the ingest pipeline (chunking, sessions, lexicon,
annotations) can be iteratively optimized until all features function
exceptionally. Code-level correctness is assumed — it's covered by the
hermetic test modules in `tests/`; a suite here failing on a code bug should
be reported as `BLOCKED(bug)`, not scored.

The eval runs against the **real corpus** in the main checkout
(`<repo-root>`,
`.venv/bin/python3`), with Ollama up. Everything that touches Ollama runs
in **one sequential lane** (Ollama serializes; parallel prompts thrash the
KV prefix cache). Everything else — ground-truth computation and transcript
grading — fans out to **parallel agents**.

---

## Execution phases (what to spawn, when)

| Phase | Agents | Parallel? | Needs Ollama? |
|---|---|---|---|
| **P0** Manifest: resolve placeholders + compute ground truth | one agent per suite S1–S8 (or one combined) | **yes** | no |
| **P1** Sweep: run all chat prompts | exactly **one** agent | **no — strictly sequential** | yes |
| **P2** Grade: score transcripts vs ground truth | one agent per suite S1–S8 | **yes** | no |
| **P3** Generator evals: wrapped/annotate/topics output quality | one agent, sequential (generation + LLM-judge share the Ollama lane) | **no** | yes |
| **P4** Synthesis: scorecard + lever recommendations | one agent | — | no |

Re-runs after an optimization change: repeat P1 for the affected suites
only, then P2/P4. Keep every sweep's transcripts (they land in
`eval/results/`, timestamped) so scores are comparable across tuning
iterations — this is a regression harness, not a one-shot.

### P0 — Manifest agent prompt (template)

> Working dir `<repo-root>`,
> python `.venv/bin/python3`. Read-only against `imessage_analysis.db` — you
> are computing ground truth, never modifying data. Using direct
> `query_tools.<fn>(...)` calls (import the package; no Ollama, no chat.py):
> 1. Resolve the placeholder roster below and write it plus every
>    ground-truth value to `eval/analytics_eval_manifest.json` (create
>    `eval/` entry; JSON keyed by prompt id).
> 2. For each prompt id in your assigned suite, run the GT recipe from the
>    suite table and store the exact values a correct answer must contain.
> Record the DB row count and max message date in the manifest — a graded
> sweep is only comparable to GT from the same corpus state.

**Placeholder roster** (resolved once, shared by all suites):

| Placeholder | Definition |
|---|---|
| `<TOP1>`, `<TOP2>` | #1 and #2 1:1 contacts by lifetime message count (use display names the model can resolve) |
| `<SPARSE>` | a real contact with 30–100 lifetime messages |
| `<GROUP>` | busiest group chat (by messages) with ≥ 4 participants; use its display name |
| `<PHRASE_HI>` | a word with ≥ 200 lifetime FTS hits whose usage visibly changed across years |
| `<PHRASE_LO>` | a phrase with 1–5 lifetime hits |
| `<TERM>` | a top-lift `dyad_terms` row for `<TOP1>`'s chat |
| `<NOBODY>` | a plausible full name that matches no contact |
| `<YEAR>` | most recent full calendar year in the corpus |

### P1 — Sweep agent prompt (template)

> Working dir = main checkout. Read
> `eval/analytics_eval_manifest.json`; substitute placeholders into the
> prompt tables of suites S1–S8 below; write `eval/prompts-analytics.txt`
> as `id|prompt` lines. Run
> `eval/fast_eval.sh eval/prompts-analytics.txt` — it executes
> sequentially by design; **never** run prompts in parallel or in a second
> concurrent Ollama session. Verify a transcript exists in `eval/results/`
> for every id; re-run individually any that died on transport errors.
> Do not grade — just report the results directory.

### P2 — Grader agent prompt (template, one per suite)

> You are grading suite <Sx> transcripts in `eval/results/<run>/` against
> `eval/analytics_eval_manifest.json`. No Ollama; read files only. For each
> prompt id, score four axes, each 0 or 1:
> - **routing** — called the expected tool with correct scope/date/phrase
>   args (transcript shows the call); acceptable alternates noted per row.
> - **accuracy** — the numbers/names/dates in the reply match ground truth
>   (counts exact; medians/rates within rounding; "top X" must be GT's X).
> - **faithfulness** — every quantitative or factual claim in the reply is
>   traceable to the tool result in the transcript; any unsupported number,
>   invented message quote, or stat presented despite an
>   `{"available": false}` / error section = 0. This axis is the
>   confabulation guard — grade it strictly.
> - **communication** — actually answers the question asked, states real
>   caveats (sparse data, low coverage) without hedging on solid data, and
>   doesn't dump raw JSON or recite section names at the user.
> Output a per-prompt table (id, 4 axis scores, one-line evidence for every
> 0), the suite mean per axis, and a **failure taxonomy** line per 0 score
> using the lever vocabulary in the "Optimization levers" section.

**Suite pass targets:** routing ≥ 0.9, accuracy ≥ 0.85, faithfulness = 1.0,
communication ≥ 0.85. Faithfulness is non-negotiable: a single confabulated
statistic is a release blocker for that feature.

---

## Suite S1 — Activity patterns (`get_activity_patterns`)

| id | prompt | expected call | ground truth (P0 recipe) & accuracy check |
|---|---|---|---|
| S1.1 | What hour of the day do I text the most? | unscoped, heatmap | argmax hour of heatmap (LOCAL_TZ); reply names that hour |
| S1.2 | What's the busiest day of the week for me and `<TOP1>`? | scoped to `<TOP1>` | argmax weekday of scoped heatmap |
| S1.3 | How many texts did `<TOP1>` and I exchange each month of `<YEAR>`? | scoped, date-bounded, granularity month | volume_series for `<YEAR>`; spot-check 3 months exactly; sent/received split preserved |
| S1.4 | Am I texting more or less this year than last year? | unscoped, yearly series | compare the two year buckets; reply gets the direction right and cites both totals |
| S1.5 | How much do I text after midnight, and who's on the other end? | unscoped, late_night | late_night share + top_contacts leaderboard; named contact = GT #1 |
| S1.6 | What's the longest streak of consecutive days `<TOP1>` and I have talked? | scoped, streaks | streak length + dates match |
| S1.7 | What was my single busiest texting day ever? | unscoped, busiest_days | GT top day (date + count) |
| S1.8 | Did `<SPARSE>` and I text much in `<YEAR>`? | scoped + date-bounded | small GT count; reply gives the number and honestly frames it as light traffic |

**Quality watch:** heatmap answers must be in local time (S1.1 vs a UTC
argmax differs by 4–5 h — a UTC-shaped answer means LOCAL_TZ isn't reaching
the reply). Series answers must not recite all 12 buckets when 2 sentences
answer the question (communication axis).

## Suite S2 — Response dynamics (`get_response_dynamics`)

| id | prompt | expected call | ground truth & accuracy check |
|---|---|---|---|
| S2.1 | Who replies faster, me or `<TOP1>`? | scoped to `<TOP1>` | per-direction median latencies; correct comparison + both numbers |
| S2.2 | How long does `<TOP2>` usually take to reply to me? | scoped | their-direction median (and p90 if cited) |
| S2.3 | Who starts our conversations more, me or `<TOP1>`? | scoped, initiation | initiation shares; correct side named |
| S2.4 | How often do I double-text `<TOP1>`? | scoped, double_text | GT count/rate |
| S2.5 | Has `<TOP1>` ever just not replied to a conversation I started? | scoped, ghosted | ghosted count (initiator-only sessions) |
| S2.6 | Does `<TOP1>` leave me on read? | scoped, read_to_reply | reply must surface the coverage caveat (their-read-receipts coverage is a few %) — an unqualified answer scores 0 on communication |
| S2.7 | Are my chats with `<TOP1>` long conversations or quick bursts? | scoped, sessions/burstiness | session count + median length; burstiness sign correct |
| S2.8 | Who's quickest to respond in `<GROUP>`? | group-scoped | per-member initiation/latency data; named member consistent with GT |
| S2.9 | What are my texting habits like overall? | any reasonable | **scope-requirement probe**: tool errors without scope — model must either pick a sensible scope, ask, or use `get_activity_patterns`; relaying `{"error": ...}` verbatim or fabricating stats = routing 0 |

## Suite S3 — Reactions (`get_reaction_stats`)

| id | prompt | expected call | ground truth & accuracy check |
|---|---|---|---|
| S3.1 | Which tapback do I use the most? | unscoped, given | GT top type by count, named (e.g. "loved" not "2000") |
| S3.2 | Which of my messages got the most reactions ever? | unscoped, hall_of_fame | GT #1 message quoted (text + who reacted); quote must match the receipt, not be paraphrased into fiction |
| S3.3 | Does `<TOP1>` react to my texts more than I react to theirs? | scoped, reciprocity | per-direction totals/rates; correct comparison |
| S3.4 | Who finds me funniest? | unscoped, funniest | GT leaderboard #1 (laughs-per-100, min-20 floor respected) |
| S3.5 | How quickly do people usually react to my messages? | time_to_react | GT median |
| S3.6 | What reactions do `<TOP1>` and I use with each other, and are we different? | scoped, given+received | per-type breakdown both directions; a real contrast drawn from the data |
| S3.7 | Did anyone love my messages in `<YEAR>`? | date-bounded, received | GT loved-count in range |

## Suite S4 — Words & language (`get_word_trends`, `get_word_frequency`)

| id | prompt | expected call | ground truth & accuracy check |
|---|---|---|---|
| S4.1 | How has my use of the word "`<PHRASE_HI>`" changed over the years? | phrase_trend, unscoped | per-year counts; trend direction right; me/them split acknowledged |
| S4.2 | When did `<TOP1>` and I start saying "`<TERM>`"? | distinctive_terms or phrase_trend + milestones acceptable | GT first use (date, who coined it, first_text) |
| S4.3 | What inside jokes or words are unique to me and `<TOP1>`? | distinctive_terms, chat-scoped | top GT terms present; uncurated terms not presented as certainties (curated flag respected) |
| S4.4 | What emoji do I use the most with `<TOP1>`? | emoji section | GT top-3 emoji in order |
| S4.5 | How similar is my texting style to `<TOP1>`'s? | lsm | GT LSM score, in [0,1], with a plain-language reading; inventing a psychological diagnosis = communication 0 |
| S4.6 | Do `<TOP1>` and I laugh a lot in our chat? | rates (laugh) | GT laughs-per-period; comparative framing supported by data |
| S4.7 | What are the most common words `<TOP2>` sends me? | `get_word_frequency` (sender-scoped, surviving tool) | GT top words |
| S4.8 | Has the mood of my chats with `<TOP1>` changed over time? | emotions/valence | **gated section probe**: if `chunk_annotations` is empty, GT is `{"available": false}` — the reply must say the analysis isn't available, not invent a mood arc; if populated, valence trajectory must match GT |
| S4.9 | Have I ever said "`<PHRASE_LO>`"? | phrase_trend or FTS search (both fine) | GT hits (1–5); exact count or the actual messages |

## Suite S5 — Milestones (`get_milestones`)

| id | prompt | expected call | ground truth & accuracy check |
|---|---|---|---|
| S5.1 | When did `<TOP1>` and I first text, and who texted first? | scoped, firsts | GT date, direction, first-message text quoted verbatim |
| S5.2 | What was the 10,000th message between me and `<TOP1>`? | scoped, nth | GT landmark row (skip to 1,000th if the chat is smaller — P0 picks the largest landmark that exists) |
| S5.3 | What's the longest `<TOP1>` and I have gone without talking? | scoped, longest_silence | GT gap duration + dates + reconnection message |
| S5.4 | Do I have any texting anniversaries coming up? | unscoped, reference_date ≈ today | GT contacts in the 30-day window; min-100-msg floor respected |
| S5.5 | When did I first mention "`<PHRASE_HI>`" to `<TOP1>`? | scoped, earliest_phrase | GT first FTS hit (date + text) |
| S5.6 | What's the last thing `<SPARSE>` sent me? | scoped, firsts (last) | GT last message |
| S5.7 | How far back does my message history go? | firsts unscoped or `get_db_overview` (both fine) | GT earliest corpus date |

## Suite S6 — Group dynamics (`get_group_dynamics`)

| id | prompt | expected call | ground truth & accuracy check |
|---|---|---|---|
| S6.1 | Who carries `<GROUP>`? | group-scoped, members | GT top member by share, with share ≈ GT |
| S6.2 | Who lurks in `<GROUP>`? | lurkers | GT lurker list |
| S6.3 | Is `<GROUP>` dominated by a few people or balanced? | gini | reading consistent with GT gini (cites the members data, not vibes) |
| S6.4 | Who starts the conversations in `<GROUP>`? | igniters | GT top igniter |
| S6.5 | Who talks to whom the most inside `<GROUP>`? | adjacency | GT top pair |
| S6.6 | Whose texting style stands out most in `<GROUP>`? | group_lsm | GT min-LSM member |
| S6.7 | What are the group dynamics of my chat with `<TOP1>`? | — | **error-path probe**: tool returns an error for non-group chats; model must recover (explain, or pivot to `get_chat_participant_stats` / `get_response_dynamics`), never fabricate group stats for a dyad |

## Suite S7 — Routing, negative controls, graceful degradation

The steering sentence and 21-tool selection are what this suite stresses;
it is the direct check on the plan's top risk.

| id | prompt | expected behavior |
|---|---|---|
| S7.1 | Find messages where `<TOP1>` and I talked about the beach | FTS/semantic **search**, no analytics tool in transcript |
| S7.2 | What did `<TOP1>` say about their job interview? | retrieval (semantic/hybrid), then grounded quote |
| S7.3 | How many messages have `<TOP1>` and I exchanged in total? | `get_chat_participant_stats` or `get_milestones` — a count matching GT; **must not** try removed tools (`get_activity_histogram`, `get_conversation_dynamics`, `get_top_reacted_messages` anywhere in a transcript = routing 0 for the run) |
| S7.4 | When am I most active, and does `<TOP1>` reply faster at night? | multi-tool: activity + dynamics; both grounded |
| S7.5 | Compare my friendships with `<TOP1>` and `<TOP2>` — who do I talk to more, who replies faster? | ≥ 2 scoped calls; per-contact numbers not blended |
| S7.6 | What hour does `<NOBODY>` usually text me? | contact resolution fails → says can't find them; no stats invented |
| S7.7 | How has my texting with `<SPARSE>` evolved over the years? | tools return thin data → honest "not much history" framing, small numbers cited correctly |
| S7.8 | What's my texting personality? | open-ended: any grounded combination is fine; graded on faithfulness only — every claim must trace to a tool result |
| S7.9 | lol who do i even text at 3am | casual phrasing still routes to `get_activity_patterns` late_night |
| S7.10 | Show me the exact messages from my longest streak with `<TOP1>` | analytics → then message retrieval (get_conversation / context); receipts real |

## Suite S8 — Conversational robustness (multi-turn)

Run these as short interactive sessions (`chat.py`, scripted stdin), one at
a time, in the P1 lane. GT from the manifest.

| id | turns | expected behavior |
|---|---|---|
| S8.1 | "Who do I text the most?" → "What hour of the day do we usually talk?" → "And who texts first, usually?" | pronoun carry-over: turns 2–3 stay scoped to the turn-1 contact; numbers match that contact's GT |
| S8.2 | "How fast does `<TOP1>` reply?" → "Is that faster than `<TOP2>`?" | second call re-scoped; comparison uses both GT medians |
| S8.3 | "What are our inside jokes?" (no prior contact) → user clarifies "`<TOP1>`" | model asks or picks visibly; after clarification, correct scoped call |
| S8.4 | "When's my anniversary with `<TOP1>`?" → "What did we say that day?" | milestones → retrieval pivot with the GT date carried over |

---

## P3 — Generator output quality (wrapped / annotate / topics)

One sequential agent (generation and LLM-judge calls share the Ollama
lane). These evaluate the *batch* LLM features' output quality.

### P3.1 wrapped.py narrative faithfulness
Run `wrapped.py --year <YEAR> --yes --out <scratchpad>/wrapped_llm.md` and
`--no-llm --out <scratchpad>/wrapped_stats.md`.
**Grade (script, no judge needed):** extract every number (counts, ranks,
percentages, dates) from the LLM version's narrative sentences; each must
appear in the stats version / computed JSON. Zero tolerance — any invented
number fails the feature.
**Grade (read):** narratives are descriptive, never evaluative (no
"you should text them more", no armchair psychology); excerpts quoted are
verbatim real messages; silent-contact auto-exclusions honored (nobody from
the excluded list is featured); markdown renders sanely (headers, no raw
JSON, ASCII heatmap intact).

### P3.2 annotate.py annotation accuracy (sampled)
Prereq: an annotate run exists (`--limit 200` is enough; user launches the
full overnight run separately). Sample 25 annotated chunks stratified by
valence (most-negative 5, most-positive 5, middle 5, highest-depth 5,
event-bearing 5). For each, show the judge (chat model via `chat_once`, or
grade manually) the chunk text and stored annotation; ask for
agree/disagree per field.
**Success:** valence *sign* agreement ≥ 80%; every stored emotion in
`EMOTION_LABELS`; events' `message_id`s point at messages actually
containing the excerpt; depth roughly monotone with judge's rating
(spot-check, not exact). Also report the run's skip rate (chunks that
failed JSON twice) — > 5% means the output contract/prompt needs work.

### P3.3 topics.py label quality (sampled)
Prereq: a topics run. For 10 random clusters, pull `TOPIC_LABEL_SAMPLES`
nearest chunks + the label.
**Success:** ≥ 8/10 labels are 2–5 words and a reader (or judge prompt)
agrees they describe the sampled chunks; no cluster > 40% of all chunks
(degenerate k-means); `topic_model_k` state matches the run.

---

## Optimization levers (P4 vocabulary)

Every 0 score gets tagged with one lever so the synthesis agent can rank
fixes by frequency. These map to the implementation plan's risk section.

| Tag | Failure looks like | What to tune |
|---|---|---|
| `desc` | wrong tool, or right tool ignored | tool description wording in `tool_schemas.py` ("aggregate analytics — how-often/who-most/when") |
| `steer` | analytics question answered by counting search results, or search question triggering analytics | SYSTEM_PROMPT steering sentence; last-resort: withhold analytics schemas for non-statistical turns |
| `args` | right tool, wrong scope/dates/phrase | parameter descriptions; scope-requirement notes on `get_response_dynamics`/`get_group_dynamics` |
| `shape` | reply misreads the result (wrong section, blended directions, misquoted number) | section naming/nesting in the tool's return dict; flatter leaderboards; per-direction labels ("me"/"them") |
| `size` | reply recites everything or truncation artifacts appear | `MAX_LEADERBOARD_ROWS`, `MAX_SERIES_BUCKETS`, auto-coarsen note wording |
| `confab` | numbers/quotes with no source in the transcript | SYSTEM_PROMPT grounding language; error/`available:false` phrasing that the model relays instead of papering over |
| `pipeline` | GT itself looks wrong (sessions split oddly, dyad terms are noise, lexicon rates implausible) | `SESSION_GAP_MINUTES`, `DYAD_TERM_MIN_LIFT`/`MIN_USES`, stopword list, chunking params — file as an ingest-tuning finding with the evidence |
| `judge-batch` | P3 failures | annotate/wrapped/topics prompt templates, `format="json"` contract, retry nudge |

### P4 — Synthesis agent prompt (template)

> Read all suite grade tables + P3 results. Produce: (1) the scorecard —
> per-suite axis means vs targets (routing ≥ .9, accuracy ≥ .85,
> faithfulness 1.0, communication ≥ .85); (2) lever histogram — count of
> failures per tag, with the three highest-impact concrete changes spelled
> out (exact description/prompt wording proposals where the tag is
> `desc`/`steer`/`args`); (3) the re-run set — which suites to sweep again
> after those changes land. Compare against the previous iteration's
> scorecard if one exists in `eval/results/`.

---

## Ground rules (all agents)

- **Ollama lane is exclusive**: P1 and P3 never overlap each other or any
  other Ollama user; prompts within them run one at a time.
- Graders and GT agents are read-only on the DB and never call Ollama.
- Placeholders are resolved once (P0) and frozen in the manifest; a GT/DB
  mismatch mid-eval (new ingest ran) invalidates the sweep — check the
  corpus fingerprint in the manifest before grading.
- Real names/messages appear in transcripts and the manifest: keep all
  eval outputs inside the repo's `eval/results/` and the scratchpad, never
  in commits (check `eval/results/` is gitignored; the manifest with
  resolved names should be too).
- Report failures with the transcript line as evidence; never edit source
  to make a test pass — levers are proposals for the user/orchestrator.

## Coverage map (feature → suites)

| Feature | Evaluated by |
|---|---|
| `get_activity_patterns` (heatmap, late-night, streaks, series) | S1, S7.4, S7.9 |
| sessions pipeline (via dynamics/igniters) | S2, S6.4, `pipeline` lever |
| `get_response_dynamics` | S2, S7.4–S7.5, S8.2 |
| `get_reaction_stats` | S3 |
| lexicon + dyad_terms pipelines | S4.2–S4.6, `pipeline` lever |
| `get_word_trends` / `get_word_frequency` | S4, S8.3 |
| `get_milestones` | S5, S8.4 |
| `get_group_dynamics` | S6 |
| tool routing at 21 tools / supersession | S7 (all), every suite's routing axis |
| multi-turn scope handling | S8 |
| `wrapped.py` | P3.1 |
| `annotate.py` (+ emotions/valence surfacing) | P3.2, S4.8 |
| `topics.py` | P3.3 |
