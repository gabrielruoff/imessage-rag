# Analysis prompt ideas

Emotionally resonant prompts to run against the iMessage analysis model
(`chat.py`), organized by use case and written the way a user would type them.
Feasibility notes against the current toolset are at the bottom.

## 1. "Texting mirror" — self-insight

- "What's my texting style? Analyze how I write — message length, emoji use,
  punctuation, how quickly I get to the point — and describe me like a
  stranger reading my texts would."
- "Am I a dry texter? Compare my average message length and reaction usage
  against the people I text most."
- "Do I double-text? Who do I double-text the most, and does it work?"
- "What words or phrases do I badly overuse? Rank them."
- "How has my texting voice changed from 2020 to now — am I warmer or more
  distant?"
- "What time of day do I send my most unhinged messages?"

## 2. Relationship dynamics & attachment

- "Analyze my attachment style with [name] based on our texts. Do I pursue,
  withdraw, or mirror them?"
- "Between me and [name], who carries the conversation? Who asks more
  questions, who texts first, who lets threads die?"
- "Show the arc of my relationship with [name]: message volume by month since
  we met. When was the peak? Are we drifting?"
- "What's the longest silence between me and [name], and who broke it?"
- "Am I a different person with [name] than with [other name]? Compare how I
  talk to each."
- "Who in my life do I only ever text back, never text first?"

## 3. Argument forensics (the killer feature)

- "Find the fight [name] and I had around [month]. Walk me through it beat by
  beat — who escalated first, and at which exact message?"
- "Who was actually right in the argument [name] and I had about [topic]? Be
  honest, don't take my side."
- "Historically, who apologizes first — me or [name]? Pull the receipts."
- "Reread that argument and tell me what they were actually trying to say
  that I missed."
- "Rewrite my worst message from that fight the way I should have sent it."
- "Do my arguments with [name] follow a pattern? Find our last three
  conflicts and compare how they start and end."

## 4. Romance & dating post-mortems

- "Read my first month of texts with [partner] and our most recent month.
  What changed, in specifics?"
- "Find the exact message where things turned romantic between me and
  [name]."
- "Was I too eager early on with [name]? Look at who initiated,
  message-length ratio, response gaps — rate my game."
- "What red flags were in [ex]'s texts that I ignored? Quote them."
- "When did [ex] start pulling away? Show me the data — I want to know if I
  imagined it."

## 5. Group chat awards show

- "Who's the main character of [group chat]? Who talks most, who gets reacted
  to most, who gets left on read?"
- "Rank everyone in [group chat] by comedic hit rate — laugh reactions per
  message sent."
- "Who starts conversations in [group chat] and who kills them?"
- "What are our inside jokes, and when did each one originate? Find the first
  use."
- "Give [group chat] superlatives: most likely to disappear for a week, most
  supportive, biggest lurker."

## 6. Nostalgia & memory-keeping

- "What did [name] and I laugh about the most in [year]?"
- "Show me the kindest things people said to me this year."
- "Take me back to [summer 2021 / the trip / the wedding] — reconstruct what
  we were talking about that week."
- "What was the last normal conversation I had with [person] before
  [event]?" (grief use case — extremely powerful, handle gently)
- "Find a message from [name] I probably forgot about that would make me
  smile."

## 7. Wrapped / year-in-review

- "Give me my 2025 texting Wrapped: top 5 people, busiest day, most-reacted
  message, my most-used phrase, and the friendship that grew the most."
- "Who was my person this year, by the numbers? Who was it last year?"
- "What's my most-loved message of all time — the one that got the strongest
  reactions?"

## 8. Coaching before the next conversation

- "I need to have a hard conversation with [name] tonight. Based on how our
  past tense conversations went, how do they respond best — and what do I do
  that shuts them down?"
- "How do I come across when I'm stressed? Find examples of me texting during
  a bad week and tell me the truth."
- "Draft a birthday message for [name] that uses our actual inside jokes and
  references our year."
- "Before I text [ex] back: read our history and tell me if this is a good
  idea."

---

## Feasibility notes against the current toolset

- **Works today (~2–6 tool calls):** everything in nostalgia, Wrapped,
  group-chat reactions, argument retrieval + qualitative judgment, style
  description. FTS/vector search finds the argument; `get_full_messages`
  recovers full text; `get_chat_summary_stats` with `group_by_chat` /
  `is_from_me` / date ranges covers volume arcs and comparisons.
- **Strains the current tools:** anything built on *response-time gaps,
  initiations, and silences* ("who texts first", "longest silence", "who
  breaks it", "double-texting"). These are computable but would force the
  model to page through raw messages within the 20-iteration budget. A single
  new tool — something like `get_conversation_dynamics(chat_id | contact_id,
  date range)` returning initiation counts, median response times per side,
  longest gaps, and streaks — would unlock categories 2, 4, and half of 1
  cheaply. That's arguably the highest-leverage next addition, since it
  powers the most emotionally sticky prompts (attachment style, drifting,
  "was I too eager").
- **Needs the deferred enrichment layer:** trend-shaped emotional questions
  ("am I warmer than in 2020", "when did they start pulling away" *at
  scale*) really want per-message sentiment stored as columns. Small-sample
  versions work today by having the model read excerpts, but corpus-wide
  trends need the batch sentiment/topic tagging already penciled in as
  next-step #3 in `CLAUDE.md`.

**Design caution:** prompts like "who was right in the fight" and the grief
prompts are the most magnetic and the most sensitive. The system prompt
should steer the model toward honesty-with-care (don't reflexively side with
the user, don't editorialize about deceased contacts) — a prompt-engineering
task, not a tooling one.
