# textmessagething

Makes iMessage history queryable and analyzable by an LLM. Ingests the
macOS Messages `chat.db` into a normalized SQLite store
(`imessage_analysis.db`) with keyword search (FTS5) and semantic search
(`sqlite-vec` + local Ollama embeddings), then exposes that store as tool
calls for a local Ollama-hosted chat model to query — instead of pasting
message history into a prompt.

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for full architecture, rationale,
schema, and known environment quirks ([`CLAUDE.md`](CLAUDE.md) is the short
agent-facing orientation).

## Quickstart

```sh
make install          # creates .venv (Homebrew Python required — see CLAUDE.md), installs deps
python3 ingest.py --no-embed   # snapshot chat.db -> imessage_analysis.db (skip embeddings if Ollama isn't running)
python3 chat.py                 # interactive CLI chat over your iMessage history
```

Requires a local `chat.db` (already present under `source_data/` for this
project) and, for embeddings/chat, Ollama running `gemma4:26b-a4b-it-qat` and
`gemma4:26b-a4b-it-qat` (`make ollama-start`).

## Other commands

```sh
make lint             # ruff check .
python -m unittest discover tests   # test suite, no Ollama/chat.db needed
make reset-db          # wipe imessage_analysis.db's data, keep schema
```
