# Eval report: analysis-prompt sweep, 2026-07-16

20 prompts from [`analysis_prompts.md`](analysis_prompts.md) (2–3 per category,
placeholders filled in) run through `chat.py -p` against
gemma4:26b-a4b-it-qat with a corpus of roughly 380k messages. Raw transcripts
(prompt, tool-call trace, streamed reply, duration) were captured per run;
findings are summarized inline below.

**This report describes tool and model behavior only.** The sweep ran against
a private corpus; contact names, chat names, message content, and row ids
have been removed rather than pseudonymized, and per-conversation statistics
are omitted. Call counts, durations, and verdicts are the real measurements.

## Executive summary

**The qualitative/narrative half of the prompt library works today; the
quantitative/superlative half does not.** Every prompt answered by *finding a
conversation and reading it* (fight forensics, coaching, nostalgia, drafting)
produced a specific, evidence-backed, often genuinely impressive answer.
Every prompt requiring *aggregation the toolset doesn't provide* (rankings by
reactions, time-bucketed volume, response-time gaps, phrase frequency) either
returned an **empty reply** (4 of 20 runs), a **confidently wrong number**
(longest-silence), or a **sample presented as the population** (style,
who-carries-the-conversation).

The single highest-leverage direction is exactly what
`analysis_prompts.md` predicted: add a small set of **single-call aggregate
tools** (conversation dynamics, activity histograms, reaction leaderboards,
per-participant group-chat stats). Secondary: a loop-level fix for the
empty-reply failure mode, and truncation disclosure on capped tool results.

### Scoreboard

| Verdict | Count | Prompts |
|---|---|---|
| Pass — specific, grounded, useful | 8 | p01 style, p07 conflict, p09 first-vs-recent, p13 laughed-about, p14 forgotten msg, p18 drafting, p19 hard-conversation, p10 drop-off |
| Partial — answered but with a real defect | 6 | p02 overused words, p04 carries-conversation, p05 arc, p08 apologizes-first, p11 main-character, p15 kindest |
| Fail — wrong answer stated confidently | 2 | p06 longest-silence, p03 unhinged-time |
| Fail — empty reply, no answer at all | 4 | p00 longest-silence (calibration), p12 comedic-hit-rate, p16 Wrapped, p17 most-loved |

The split maps almost perfectly onto prompt type: all four empty replies and
both wrong answers are quantitative/superlative questions; all eight passes
are retrieval-and-read questions.

## What works (keep and lean into this)

The **hybrid-search → `get_conversation` → synthesize** pattern is the
success story. Representative runs:

- **p07 conflict-forensics** (6 calls, 514s): hybrid search with emotion
  keywords scoped to the contact → expanded two chunks → reconstructed a
  multi-turn exchange in correct order, with accurate timestamps and correct
  speaker attribution throughout. This is the "killer feature" category, and
  it works.
- **p19 hard-conversation coaching** (2 calls, 311s): one
  `search_conversations` call → a specific, grounded answer that cited
  retrieved messages rather than generic advice, and generalized across
  several retrieved threads instead of one. Best answer of the sweep.
- **p18 drafting-with-callbacks** (4 calls): retrieval surfaced
  chat-distinctive vocabulary, and the drafts used it correctly.
- **p13 laughed-about-2024** (1 call, 47s): date-scoped hybrid search,
  correct and charming answer, and — once the GPU was uncontended and the KV
  cache warm — *fast*. The happy path is 1–3 tool calls and under a minute.

The system prompt's resolve-contact-first guidance is followed reliably
(`search_contacts_by_name` → scoped search in nearly every run), and in p11
the model correctly passed **both** chat_ids for a group thread that exists
as two rows (iMessage + SMS continuation) — it merged them without being
told.

## Failure modes

### 1. Empty replies — 20% of runs (loop bug, fix in `chat.py`)

Four runs ended with `run_turn` returning `''`: the model stopped calling
tools and emitted a message with no content, well before the 20-iteration
budget (7, 8, and 15 calls used). All four were prompts where the toolset
gave it no path to the number it needed (reactions-per-person, Wrapped
superlatives), so it appears to stall out rather than say "I can't compute
this." p16 (Wrapped) is the starkest: 8 sensible scoping calls, then nothing.

`chat.py` treats empty-content-no-tool-calls as a final answer and returns
it. A cheap fix: when content is empty, make one retry with the
`FINAL_TURN_NOTE` folded in (the same mechanism already used at budget
exhaustion), which forces "answer with what you have / say what's missing."
That converts 4 hard failures into honest partial answers.

### 2. Confidently wrong numbers from capped samples

`search_messages_sql` returns the most recent `limit≤50` rows
(`ORDER BY m.date DESC`) **with no indication that the result is a slice of
a larger population**. The model repeatedly treated the visible window as
the whole history:

- **p06 longest-silence**: answered with a specific gap in days derived from
  the last 50 messages of a multi-thousand-message, multi-year thread.
  Plausible-sounding, almost certainly wrong, stated with full confidence.
  This is worse than the empty replies — nothing signals the user to
  distrust it.
- **p01 texting-style / p04 carries-conversation**: both generalized from the
  most recent 50 rows. p04 extrapolated a long-run characterization from a
  single recent exchange.
- **p09 first-vs-recent-month**: date-bounded queries still return the
  *last* 50 rows of each window, so "first month" is actually "the tail of
  the first month."

Two complementary fixes: (a) per-row tools should return a
`{"rows": [...], "total_matching": N, "order": "newest-first"}` envelope so
the model *knows* it's sampling; (b) a system-prompt line: "row-returning
tools cap at 50 newest rows — never claim a superlative (longest, first,
most) from a capped result; use stats tools or say the window you checked."

### 3. No aggregation path for the questions users most want to ask

The doc's feasibility section predicted this; the transcripts confirm every
instance of it:

- **p05 relationship-arc** burned the *entire 20-iteration budget*, including
  **ten identical `search_messages_sql(contact_id=[...], limit=1)` calls** in
  a row, then approximated "volume by month" with 6 coarse year/quarter
  stats calls (1756s total). It also misread the stats it did get (claimed
  every message in a period came from one side — a misreading of the
  `from_me`/`from_them` fields) and its ratio math is wrong (it described a
  roughly one-third/two-thirds split as "roughly one-third").
- **p12 comedic-hit-rate / p17 most-loved / p16 Wrapped**: there is no tool
  that ranks messages by reactions received, and `get_chat_summary_stats`'s
  `reaction_count` counts reactions *sent by* the scoped person. p17 probed
  semantically-found "love"-topic messages with `get_message_reactions` and
  got zero rows every time — post-eval investigation showed those zeros were
  not "no tapbacks" but a data bug: `reactions.target_message_guid` kept
  chat.db's `p:N/`/`bp:` part prefix, so well under 1% of reactions joined
  `messages.guid` at all (fixed alongside the recommendations below). All
  three prompts ended empty or inverted.
- **p11 main-character**: presented reactions-*sent* as "gets reacted to
  most" — semantically inverted answer delivered confidently. It also only
  checked 3 participants (the ones it could name) rather than all members.
- **p03 unhinged-time**: with no hour-of-day histogram available, it
  keyword-searched the literal word "unhinged" and reported the timestamps
  of messages *containing that word*. Metric error masquerading as an
  answer. Same confirmation-bias shape in p02 (seeded FTS with "literally",
  then reported "literally" as the top overused word — no frequency evidence
  involved anywhere).

### 4. Minor but recurring model quirks

- **Stray HTML fragments in tool args**: hybrid queries ending in
  `hate-it`, `apology</div>`, `laugh</h5>`. Harmless to sqlite/FTS but a
  sign of degraded argument generation under long contexts.
- **Duplicate identical calls**: p05's ten `limit=1` calls, p12 and p16 each
  repeated a call verbatim. A dispatch-level dedup guard (return the cached
  result with a note "identical call already made") would reclaim iterations.
- **Wasted probe calls**: p04 guessed chunk_ids without searching, and ran
  FTS with query `'?'` — three near-empty results.
- **Silent disambiguation**: a `search_contacts_by_name` call on a common
  first name returned 3 matches; the model picked one without noting the
  ambiguity (p14).

## Issues discovered outside the model's behavior

- **Split chats across services**: a group thread in the corpus is two chat
  rows — an iMessage chat that stops at a service switch, and an SMS chat
  that continues from it. The model merged them this time, but any
  single-chat_id scoping silently drops the recent half. Worth either
  merging at ingest, exposing a "sibling chats" hint in `list_chats`, or a
  system-prompt note.
- **`handle_id=0` sentinel**: from-me messages carry `handle_id=0` (and a
  handful of received rows do too), which joins to no `handles` row.
  Query-side code handles it via `is_from_me`, but any future per-handle
  join needs to know. NULL would be cleaner.
- **Duplicate handle rows**: one contact's phone number appeared as 4
  separate `handles` rows all linked to the same `contacts` row.
  Contact-scoped queries are fine; handle-scoped stats could silently split.
- **`reaction_count` naming**: as noted above, it means "reactions sent by
  the scope," which the model (reasonably) misreads. Rename or document in
  the schema description (e.g. `reactions_sent_count`).
- **Reaction target guids never joined** (found post-report while
  implementing the fixes): ingest stored `associated_message_guid` raw, so
  99.6% of reactions carried a `p:N/` or `bp:` part prefix that
  `messages.guid` lacked — `get_message_reactions` silently returned nothing
  for nearly every message. Fixed at ingest + idempotent migration.

## Performance findings

- One prompt ≠ one generation: each is 1–20 sequential Ollama round-trips
  with 10–25k-token contexts. Duration tracks round-trips almost linearly
  (~35s/round-trip solo, ~90–120s under the 4-way contention this eval ran
  with).
- **Ollama serializes requests** (measured), so running eval batches in
  parallel added queueing *and* thrashed the KV prefix cache — every
  round-trip re-prefilled the whole conversation. p05 (20 iterations,
  contended): 1756s. p13 (1 call, solo, warm cache): 47s.
- Eval-rig implications: run prompt sweeps **single-stream**; for smoke
  tests set `TOOL_RESULT_MAX_CHARS≈12000` and `MAX_TOOL_ITERATIONS=8` (both
  env-only), and iterate on a small tool-calling model via the existing `-m`
  flag, reserving gemma4 for final quality passes.
- The latency is a product problem, not just an eval problem: a real user
  waits 5–30 minutes for the quantitative prompts. The aggregate tools below
  collapse those 6–20 round-trips into 1–2, so they are simultaneously the
  accuracy fix and the speed fix.

## Recommended direction, ranked

1. **`get_conversation_dynamics(chat_id|contact_id, date range)`** —
   initiation counts, median/percentile response times per side, longest
   gaps (with who broke them), double-text counts, per-month message
   balance. Directly fixes p00/p04/p06 (two empty/wrong today) and unlocks
   the attachment-style, drifting, and "was I too eager" prompts — the
   doc's own "highest-leverage next addition," now confirmed empirically.
2. **`get_activity_histogram(scope, bucket=month|dow|hour, split_by_sender)`**
   — one call replaces p05's 20-iteration thrash and powers arcs, busiest
   day, time-of-day, and every Wrapped volume stat.
3. **Reaction analytics**: `get_top_reacted_messages(scope, date range,
   reaction_type?, received_only=true)` plus per-participant group-chat
   stats (messages sent, reactions received/sent per member). Fixes p11's
   inversion, and turns p12/p16/p17 from empty replies into one-call
   answers. Rename or re-document `reaction_count` while at it.
4. **Loop robustness in `chat.py`**: retry-once-with-`FINAL_TURN_NOTE` on
   empty content; dedup guard for identical repeated tool calls.
5. **Truncation disclosure**: `total_matching` + order metadata on capped
   row results, plus a system-prompt rule against claiming superlatives
   from capped samples.
6. **Nice-to-have**: `get_word_frequency(scope, is_from_me, ngram=1..3)`
   for the style/overuse prompts (currently answered by vibes, not counts);
   a "sibling chats" hint for service-split threads; contact-disambiguation
   nudge in the system prompt.

Items 1–3 are also the honest scope boundary: corpus-wide *emotional* trends
("am I warmer than in 2020") still want the deferred enrichment layer, but
none of the failures in this sweep needed it — every one traces to a missing
aggregate, not missing sentiment.

## Appendix: per-prompt results

| id | prompt (abridged) | calls | secs | verdict |
|---|---|---|---|---|
| p00 | longest silence, 1:1 (calibration, solo) | 8 | 304 | empty reply |
| p01 | texting style | 1 | 135 | pass (recent-50 sample, undisclosed) |
| p02 | overused words | 4 | 437 | partial — no frequency evidence, seeded query |
| p03 | unhinged time of day | 2 | 268 | fail — literal keyword as metric |
| p04 | who carries conversation, 1:1 | 9 | 873 | partial — one exchange generalized; 3 wasted probes |
| p05 | arc by month, 1:1 | 20 | 1756 | partial — 10 dup calls, yearly not monthly, misread stats |
| p06 | longest silence, 1:1 | 3 | 388 | fail — confident wrong number from last-50 window |
| p07 | conflict beat-by-beat | 6 | 514 | pass |
| p08 | who apologizes first | 2 | 254 | partial — 10 chunks as "historically", hedged OK |
| p09 | first vs recent month, 1:1 | 6 | 283 | pass (50-row slices per window) |
| p10 | engagement drop-off over time | 7 | 650 | pass — good buckets; didn't attribute which side |
| p11 | group main character | 8 | 843 | partial — reactions semantics inverted; merged split chat correctly |
| p12 | comedic hit rate | 7 | 700 | empty reply |
| p13 | laughed about in 2024 | 1 | 47 | pass |
| p14 | forgotten message from a quiet contact | 3 | 228 | pass (silent contact disambiguation) |
| p15 | kindest things 2025 | 1 | 210 | partial — 2 examples, single source |
| p16 | 2025 Wrapped | 8 | 778 | empty reply |
| p17 | most-loved message | 15 | 691 | empty reply |
| p18 | drafting with callbacks | 4 | 316 | pass |
| p19 | hard-conversation coaching | 2 | 311 | pass — best of sweep |
