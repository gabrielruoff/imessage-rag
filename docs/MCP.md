# MCP server

An optional [Model Context Protocol](https://modelcontextprotocol.io) server that
lets an agent client (Claude Code, Claude Desktop, MCP Inspector, any stdio MCP
client) search and analyze your ingested iMessage history. It is transport only:
it serves 21 of the read-only retrieval tools from `query_tools/`, with the same
schemas `chat.py` gives the local model (`tool_schemas.TOOL_DEFINITIONS`,
verbatim). The code is the `mcp_server/` package. Design history is in
[`DECISIONS.md`](DECISIONS.md), in the entry "MCP server over the read-only
retrieval tools" (the "MCP server entry" the code points at).

Read [Privacy & security model](#privacy--security-model) before you connect an
agent. It changes where your messages go.

## Capabilities

The tools are listed in this order (`mcp_server.EXPOSED_TOOLS`). Every tool is
annotated read-only, non-destructive, idempotent and closed-world.

### Search & retrieval

| Tool | Title | What it does |
|---|---|---|
| `search_messages_sql` | Filter messages | Filter messages by chat, sender or date range. |
| `search_messages_fts` | Keyword search | FTS5 keyword/phrase search across all message text. |
| `search_conversations` | Semantic search | Semantic search over conversation chunks. **Needs Ollama** (query embedding). |
| `search_messages_hybrid` | Hybrid search | The default search: keyword + semantic, merged by reciprocal-rank fusion. **Needs Ollama.** |
| `find_similar_conversations` | Similar conversations | More-like-this from a `chunk_id`, using its stored vector (no Ollama). |
| `get_conversation` | Expand a conversation | Expand a chunk into its full messages. |
| `get_message_context` | Messages around a message | The messages just before and after one message. |
| `get_full_messages` | Full message text | Untruncated text of messages by id. |
| `get_message_reactions` | Reactions on messages | Tapbacks on messages, and who reacted. |

### Contacts & chats

| Tool | Title | What it does |
|---|---|---|
| `search_contacts_by_name` | Find a contact | Name or organization to contact, handle and chat ids. Call it first. |
| `search_contact_by_handle` | Look up a phone or email | Contact info for a phone number, email or `handle_id`. |
| `list_chats` | List chats | Chats with resolved participant names; find a `chat_id`. |

### Analytics

| Tool | Title | What it does |
|---|---|---|
| `get_chat_summary_stats` | Message counts | Message, attachment and reactions-sent counts, plus date range. |
| `get_activity_patterns` | Activity patterns | When you text: heatmap, volume series, late-night share. |
| `get_response_dynamics` | Response dynamics | Reply latency, who initiates, sessions, ghosting, for one person or chat. |
| `get_reaction_stats` | Reaction stats | Reactions given and received, reciprocity, time to react. |
| `get_word_trends` | Word trends | Language over time: phrase trends, emoji, rates. Emotion/valence sections need `make annotate`. |
| `get_word_frequency` | Word frequency | Most frequent words or phrases, with real counts, scoped by sender. |
| `get_milestones` | Milestones | Firsts, nth messages, anniversaries, longest silences. |
| `get_group_dynamics` | Group chat dynamics | Who dominates, lurks and replies to whom in a group chat. |
| `get_chat_participant_stats` | Per-member stats | One row per participant: messages, dates, reactions. |

### Not exposed, and why

- **`get_language_trajectory`.** Its wellbeing floor (`chat_floor.TurnFloor`)
  scrubs the whole reply and appends a support epilogue. That only works inside
  this app's own turn-runners. An MCP server never sees the agent's reply, so the
  floor can't be enforced, and the repo's rule is that removing the capability is
  the control.
- **Every Voice, Conflict, Nudge and Drift tool.** They are outside this server's
  scope. Several write to the database (`resolve_loose_thread`,
  `mute_contact_from_nudges`), and others depend on the DV screen or on code-printed
  epilogues that only a turn-runner can deliver.

Adding a tool means an explicit edit to `EXPOSED_TOOLS` plus a `DECISIONS.md`
note. A tool that writes, or that relies on a turn-level floor, is never added.

## Prerequisites

- `make ingest` has run, so `imessage_analysis.db` exists. The server refuses to
  start without it.
- `make mcp-install` has installed `requirements-mcp.txt` into `.venv`.
- Ollama is running for the two semantic tools (`search_conversations`,
  `search_messages_hybrid`). The other 19 work without it.
- `make annotate` has run, if you want the emotion/valence sections of
  `get_word_trends`.

No Full Disk Access is needed. The server reads only `imessage_analysis.db`,
never `chat.db`.

## Run

```sh
make mcp-install   # once
make mcp           # serves on stdio
```

`make mcp` sits silently waiting on stdin. That is normal: an MCP client is
supposed to launch it and talk JSON-RPC over stdin/stdout. A one-line banner
(`imessage-rag MCP server: 21 read-only tools over <db> (stdio; Ctrl-D to stop)`)
goes to stderr. Ctrl-D (closing stdin) stops it; Ctrl-C does not, because the
SDK's stdin reader thread can't be interrupted. `.venv/bin/python3 -m mcp_server --help`
prints usage and exits.

## Client setup

Replace `/ABS/PATH/imessage-rag` with the absolute path to your checkout.

**Every make-based launch command carries `-s --no-print-directory`. Don't drop
them.** stdout is the protocol wire, and `make -C` can print `make: Entering
directory …` to stdout (GNU make 4.x always, macOS make 3.81 when `MAKEFLAGS=w`
is set), which corrupts the stream before the server starts.

### Claude Code

```sh
claude mcp add imessage --scope user -- make -s --no-print-directory -C /ABS/PATH/imessage-rag mcp
```

Use `--scope user` or `--scope local`. Don't commit a `.mcp.json`: the path is
specific to your machine, and the server exposes your private messages to any
session that loads it. Tools then appear as `mcp__imessage__search_messages_fts`
and so on.

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "imessage": {
      "command": "make",
      "args": ["-s", "--no-print-directory", "-C", "/ABS/PATH/imessage-rag", "mcp"]
    }
  }
}
```

If Desktop can't find `make`, use its absolute path (`/usr/bin/make`).

### Any other stdio client

Launch `.venv/bin/python3 -m mcp_server` with the working directory at the repo
root. This is a MUST, not a convenience: `python -m` puts the cwd first on
`sys.path`, so launched from another directory the server would import that
directory's `config.py`, `db.py`, `json.py` and so on into a process holding your
message DB. Don't use `PYTHONPATH` instead; it leaves the cwd ahead of it. The
make launch guarantees the cwd. A direct launch skips `make mcp`'s
missing-dependency check.

### Testing with MCP Inspector

```sh
npx @modelcontextprotocol/inspector make -s --no-print-directory -C /ABS/PATH/imessage-rag mcp
```

## Protocol notes

- **Spec revision 2026-07-28**, via the `mcp==2.3.0` Python SDK. Clients that
  still use the older `initialize` handshake are served too, with no configuration.
- **stdio only.** There is no HTTP listener.
- **`tools/list`** returns the 21 tools in a fixed order, with titles and
  read-only annotations. Annotations are hints for client UIs, not controls.
- **Errors.** An unknown tool name is a protocol error (`-32602`). Arguments that
  fail the tool's JSON Schema, a tool that returns an `{"error": ...}` dict, and
  any exception inside a tool all come back as a tool result with `isError: true`
  and a JSON `{"error": "..."}` body, so the model can correct itself.
- **Results** are JSON as text content. There is no `outputSchema` or
  `structuredContent`.
- **Result cap.** One result's JSON is capped at `MCP_RESULT_MAX_CHARS` (default
  40000). That sits under Claude Code's default 25k-token MCP output limit even at
  ~2 chars/token for escaped emoji. An over-cap list result is cut at whole-row
  boundaries and carries a note saying so; anything else comes back as a prefix
  labeled as a fragment. Either way it stays valid JSON, rather than being cut
  mid-token by the client. This is the same helper `chat.py` uses.

## Privacy & security model

- **Retrieved message text goes to whatever model the agent uses.** With a hosted
  model (Claude Code, Claude Desktop, any cloud LLM), every message a tool returns
  leaves your machine and goes to that provider. This is unlike `make chat`, which
  talks only to your local Ollama.
- **Message text can carry prompt injection.** Other people wrote much of it. A
  message that reads like an instruction is still just data, and the server's
  instructions tell the agent so, but that is guidance, not a guarantee. Be
  careful giving the same agent shell, email or browser tools alongside this server.
- **stdio only, no listener.** Only the client process that launched the server
  can talk to it.
- **Read-only.** None of the 21 tools writes to the database.
- **No per-contact deny-list or consent list applies to these tools.**
  `WELLBEING_DENY_IDS` gates only `get_language_trajectory`, which is not exposed.
  `WRAPPED_VOICEPRINT_CONSENT_IDS`, `PERSONA_SIM_BLOCK_IDS` and nudge mutes gate
  features this server doesn't expose either. Every contact's messages can be
  retrieved, exactly as they can from `make chat`.
- **Floor-dependent tools are excluded** (see [Not exposed](#not-exposed-and-why)).

## Configuration

All of these live in `config.py` and can be set in `.env` or the client's `env`.
`.env.example` lists `MCP_RESULT_MAX_CHARS` commented out under `# --- MCP server ---`,
like every line in that file; uncomment it in your `.env` to change it.

| Variable | Effect |
|---|---|
| `TARGET_DB` | The analysis DB to serve (default `imessage_analysis.db`, relative to the repo). |
| `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_*` | Where the semantic tools reach Ollama for query embeddings. |
| `EMBED_MODEL` | Must match the model the DB was ingested with, or semantic results are silently poor. |
| `MCP_RESULT_MAX_CHARS` | Hard cap on one tool result's JSON (default 40000). Separate from `TOOL_RESULT_MAX_CHARS`, which is sized for the local chat model. |

## Troubleshooting

- **`mcp_server: … not found -- run 'make ingest' first`.** No analysis DB at
  `TARGET_DB`. Run `make ingest`, or fix the path.
- **`MCP deps missing -- run: make mcp-install`.** Run `make mcp-install`.
- **Semantic tools return an error.** Ollama isn't reachable. Run
  `make ollama-start`. Meanwhile the agent can fall back to `search_messages_fts`.
- **Semantic results look irrelevant.** `EMBED_MODEL` differs from the model the
  DB was built with. Unset it, or re-ingest (`make reset-db` + `make ingest`).
- **"database is locked", in both directions.** The DB doesn't use WAL. A tool
  call can fail while `make pipeline`, `annotate` or `ingest` holds the write
  lock. A long MCP query can also make one of those runs fail its commit after
  SQLite's 5-second busy timeout. Don't run heavy agent sessions during a pipeline
  run.
- **A result ends with a truncation note.** It hit `MCP_RESULT_MAX_CHARS`. Ask
  with narrower filters or a smaller `limit`, or raise the cap.
- **The client reports invalid JSON or the connection drops at startup.**
  Something wrote to stdout, which is the wire. Check that the launch command
  carries `-s --no-print-directory`, and that no local change adds a `print()` in
  the server path.

## Non-goals

- Streamable HTTP or any network listener, and therefore no auth.
- Write tools, MCP resources and prompts, `outputSchema`/`structuredContent`,
  progress notifications, and Tasks.
- Roots, Sampling and Logging, all deprecated in 2026-07-28 (logs go to stderr).
- Non-default cache TTLs.
- An MCP Registry `server.json`, an MCPB/DXT bundle, or a committed `.mcp.json`.
- Any tool beyond the 21 above.
