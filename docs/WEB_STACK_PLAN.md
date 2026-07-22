# Web Stack Plan — Backend API + Frontend for imessage-rag

**Status:** Recommendation / handoff. No code written yet.
**Goal:** Put a web UI on the existing local chat engine. Phase 1 = a basic chat
app exposing the chat feature. Endgame = a fully-fledged analytical + generative
application.

---

## TL;DR recommendation

- **Backend:** FastAPI + Uvicorn — a thin async layer wrapping the existing
  Python modules (`chat.run_turn`, `query_tools`, analytics pipelines,
  `wrapped`). No engine rewrite.
- **Frontend:** React + TypeScript + Vite + Tailwind + shadcn/ui, with
  **assistant-ui** or **Vercel AI SDK UI** for the streaming chat component and
  Recharts/visx/Observable Plot for the analytics phase.
- **Seam:** SSE (`text/event-stream`) for chat token streaming; reuse the tools'
  existing typed envelopes as the API contract.
- **Persistence:** keep SQLite. Add a small separate app-state DB (`app.db`) for
  conversations/turns. No Postgres/Redis/queue — single-user local.
- **Assumption (confirm):** local-first, single-user on macOS. Multi-user hosting
  changes DB, auth, and the privacy guarantee.

---

## Grounding: how the codebase actually works (verified)

These are the facts the recommendation is built on — verify still-current before
large changes.

- **Everything is Python.** All logic (chat loop, tool-calling, retrieval,
  analytics) is in top-level Python modules. No existing web/API/server layer —
  grep for `fastapi|flask|uvicorn|starlette` returns nothing.
- **Chat entry point:** `chat.run_turn(messages, show_model_header=True,
  model=None)` at `chat.py:444`. It:
  - takes a message list, **mutates it in place**, runs the tool-call loop
    (`MAX_TOOL_ITERATIONS`) against Ollama, returns the final reply **string**.
  - **prints to a `rich` console as it streams** (via `ollama_chat_stream`
    wrapper) — i.e. output is currently side-effect printing, not yielded.
  - is **synchronous** and stateless across turns (state = the `messages` list
    the caller owns).
  - dispatches to `chat_apple.run_turn` when the model is the `apple` sentinel
    (`chat.py:453`).
  - The **evalkit already calls `chat.run_turn` in-process** — proves the
    function is a clean programmatic entry point.
- **Streaming primitive exists:** `ollama.chat_stream` at `ollama.py:177`
  **yields** each chunk's `message` dict as it arrives. So token streaming is
  available at the transport layer; only `run_turn` swallows it into prints +
  a final string.
- **Wellbeing floor lives *inside* `run_turn`** (`chat.py:452`–`585`): whole-reply
  sanitize (`_SANITIZE_REPLY` / `wellbeing.sanitize_condition_names`), the
  `_epilogue` verbatim-append channel, and stripping `_epilogue`/`support_block`
  from what the model sees. `../chat_apple.py` re-implements this floor because it
  bypasses `run_turn`. **Any new streaming/web entry path must re-enforce the
  same floor — it is not inherited.**
- **Data layer:** SQLite (`imessage_analysis.db`) with sqlite-vec (`chunk_vectors`,
  cosine) + FTS5 (`messages_fts`). Single-file, local.
- **Models:** local only — Ollama (chat `gemma4:26b-a4b-it-qat`, embed
  `nomic-embed-text`) or opt-in Apple Foundation Models. **Ollama serializes**
  requests; pipelines hold a **SQLite write lock across LLM calls** (concurrent
  runs die with `database is locked`; `make pipeline` is deliberately serial even
  under `make -j`). => the API must treat generation as serialized/queued, not
  fan-out. Fine for single-user.
- **Deps are minimal:** `../requirements.txt` = `python-dotenv`, `sqlite-vec`,
  `rich`. Runs on a Homebrew-Python `.venv` (`../.venv/bin/python3`).
- **Tool surface:** 32 retrieval functions in `../query_tools` (+5 behind feature
  flags), each with a typed **envelope** return
  (`rows`/`total_matching`/`returned`/`order`/`truncated`). `../wrapped.py` already
  emits a self-contained HTML analytics report (inline CSS/JS, charts) — reusable
  as an embedded report early on.
- **Privacy invariant:** "retrieval, not context-stuffing — the corpus never
  enters the prompt"; local LLM. This is the load-bearing design constraint and
  should stay true through the web layer.

---

## Backend: FastAPI + Uvicorn — rationale

- Thin adapter over existing modules; every route = `import query_tools` /
  `import chat` / `import wrapped`. No rewrite.
- Native SSE + WebSocket streaming (chat token stream is the core UX).
- Pydantic response models map ~1:1 onto the existing tool envelopes.
- Auto OpenAPI docs as the tool/analytics surface grows.

### The single highest-leverage refactor (do first)

Turn `run_turn` into a **generator/event stream** instead of a console printer.
Yield structured events: `token`, `tool_call` (name/args/row-count), `epilogue`,
`done`. `ollama.chat_stream` already yields deltas, so this is refactoring
`run_turn`'s consumption, not new infra.

**Must preserve while doing this:** the wellbeing floor (whole-reply sanitize,
`_epilogue` emit, `support_block` strip). Consider extracting the floor into a
shared helper both `run_turn`, the streaming variant, and `chat_apple` can call,
to stop the triplication.

### Persistence

Keep SQLite. Add a **separate** `app.db` for `conversations` / `turns` (don't
mix app state into the derived `imessage_analysis.db`). No Postgres/Redis/queue.

### Concurrency

Serialize generation (one in-flight chat/annotate; queue the rest). Do **not**
design for high concurrency — Ollama + the SQLite write lock forbid it, and
single-user doesn't need it.

---

## Frontend: React + TypeScript + Vite — rationale

- Chosen **because** the analytical + generative roadmap is where React's
  ecosystem (AI-chat UI + charting) is deepest.
- **Vite** dev/build; SSE proxy to FastAPI in dev.
- **Tailwind + shadcn/ui** (Radix) for fast, low-lock-in UI.
- **Chat component: do not hand-roll.** Use **assistant-ui** or **Vercel AI SDK
  UI** — streaming messages, tool-call rendering, stop/regenerate, SSE built in.
- **TanStack Query** for server state.
- **Charts (analytics phase):** Recharts for standard dashboards; visx /
  Observable Plot for bespoke. Can embed `../wrapped.py` HTML early.

---

## The seam

- **SSE** (POST + `text/event-stream`) for chat streaming — simplest correct
  choice for local single-user. Reserve WebSockets for later live/bidirectional
  dashboards.
- **Tool envelopes = API contract.** Frontend renders `rows`/`truncated`
  directly; tool-call notices become UI affordances ("searched 1,240 messages…").
- Keep code-enforced floors (DV screen, consent gates, wellbeing sanitize)
  **server-side**.

---

## Suggested repo layout

```
imessage-rag/
  (existing Python modules stay put)
  api/            # FastAPI app: routers over chat, query_tools, analytics
  frontend/       # React + Vite + TS frontend
```

One Uvicorn process serves API + built `frontend/dist` in prod; Vite dev server
with proxy in dev.

---

## Growth path

1. **Phase 1 (now):** FastAPI `/chat` SSE endpoint + React chat pane.
2. **Analytical phase:** expose analytics tools + `wrapped`-style sections as
   endpoints → dashboard routes with charts.
3. **Generative phase:** persona/draft/conflict tools are already generative —
   surface as UI actions, floors stay server-side.
4. **Eventual native app:** **Tauri** (Rust webview) bundling the React frontend
   with the Python backend as a sidecar — stays local, matches privacy story.
   Premature now.

---

## Open decision (needs owner input)

**Local-first single-user (assumed) vs hosted multi-user.**
If hosting for multiple users is ever intended, three things change:
- Postgres instead of SQLite,
- real auth + per-user model/data isolation,
- the "corpus never leaves the machine" privacy guarantee needs rethinking.

Everything above assumes **local-first, single-user on macOS**, which is what the
entire codebase signals.

---

## Suggested first PR for the next agent

1. Add `api/` FastAPI app with an SSE `/chat` endpoint.
2. Refactor `run_turn` into an event-yielding generator (preserve the wellbeing
   floor; ideally extract the floor into a shared helper).
3. Add `app.db` + `conversations`/`turns` tables for chat persistence.
4. Scaffold `frontend/` (Vite + React + TS + Tailwind + shadcn/ui) with a minimal
   streaming chat pane wired to the SSE endpoint via assistant-ui / Vercel AI SDK.
5. Add FastAPI/uvicorn deps to `../requirements.txt`; add `make` targets for
   `api-dev` and `web-dev`.

Before scaffolding, re-confirm the local-first assumption with the owner.
