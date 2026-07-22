# imessage-rag frontend

A minimal, streaming chat UI over the local iMessage-RAG engine — React +
TypeScript + Vite + Tailwind v4. It talks to the FastAPI backend (`api/`) over
Server-Sent Events; the corpus never leaves your machine.

## Develop

Two terminals, from the repo root:

```bash
make api-dev    # FastAPI backend on http://127.0.0.1:8000 (needs Ollama running)
make web-dev    # Vite dev server on http://localhost:5173
```

Vite proxies `/api` (including the SSE stream) to `127.0.0.1:8000`, so open the
Vite URL and chat. Override the backend target in `vite.config.ts` if you moved
the API's host/port.

## Commands

```bash
npm install        # or: make web-install
npm run dev        # dev server (make web-dev)
npm run build      # typecheck + production build to dist/ (make web-build)
npm run lint       # oxlint
npm run preview    # preview the production build
```

## How it fits together

- `lib/types.ts` — the wire contract: the `ChatEvent` discriminated union and the
  conversation/turn/health shapes, mirroring the FastAPI models.
- `lib/sse.ts` — a hand-rolled `text/event-stream` reader over `fetch` +
  `getReader()` (EventSource can't POST); yields typed `ChatEvent`s.
- `hooks/useChatStream.ts` — owns the displayed message list, POSTs a turn, and
  folds streamed `token`/`tool_call`/`epilogue`/`done`/`error` frames into the
  optimistic assistant bubble. `done.reply` is authoritative (a wellbeing turn
  buffers into one chunk server-side).
- `components/` — `ConversationSidebar`, `ChatPane` (`MessageList` +
  `MessageBubble` + `Composer`), and `ToolCallNotice` (the retrieval "receipts").
  The code-emitted support block arrives as a distinct `epilogue` event and is
  rendered as a separate, non-model callout — never as assistant text.
- TanStack Query drives the conversation sidebar + health poll.

## Production

`make web-build` emits `frontend/dist`; the FastAPI app serves it at `/` when the
directory exists (one process serves API + UI).
