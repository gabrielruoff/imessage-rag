# api/ — web backend

FastAPI transport over the existing chat engine for a **local, single-user** web
UI. It never re-implements retrieval or the wellbeing floor — it persists the
UI's conversation history in `app.db` and streams `chat.run_turn_events` to the
browser over SSE. The engine stays stateless; every request rebuilds the
messages list from the stored turns.

## Run (dev)

```sh
.venv/bin/python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Then run the Vite frontend on :5173 (allowed by CORS via `API_CORS_ORIGINS`).
In production, build the frontend to `frontend/dist` and it is served at `/`.

## Endpoints

- `POST /api/chat` — SSE stream (see frames below). Body:
  `{conversation_id?: int, message: str, model?: str}`.
- `GET /api/conversations` — list, newest first, with `turn_count`.
- `POST /api/conversations` — create empty (`{title?}`).
- `GET /api/conversations/{id}` — conversation + `turns` (404 if missing).
- `DELETE /api/conversations/{id}` — 204.
- `GET /api/health` — `{ok, backend, model, reachable}`.

## SSE frame format (`POST /api/chat`)

Each frame is a named SSE event; `data` is a JSON object = the engine event with
its `type` key removed.

- `event: meta` — `{"conversation_id": <int>, "model": <str>}` (always first).
- `event: token` — `{"text": <str>}` a content delta (one buffered token on a
  wellbeing turn).
- `event: tool_call` — `{"name", "arguments", "count", "tokens_est"}`.
- `event: retry` — `{"error": <str>}`.
- `event: epilogue` — `{"text": <str>}` verbatim support block; render distinctly,
  NOT model-attributed.
- `event: done` — `{"reply": <str>}` terminal; full sanitized reply.
- `event: error` — `{"message": <str>}` a transport failure; terminal.

The assistant turn is persisted server-side from `done.reply` after the stream
ends, so the client does not need to POST it back.

## Notes

- Only one generation runs at a time (module lock in `chat_stream.py`): Ollama
  serializes and both DBs take write locks, so concurrency would only deadlock.
- `app.db` is plain sqlite3 (no sqlite-vec), separate from the analysis DB.
  Path/host settings live in `config.py` (`APP_DB`, `API_HOST`, `API_PORT`,
  `API_CORS_ORIGINS`).
