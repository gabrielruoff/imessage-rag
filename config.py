"""Shared config for the pipeline, loaded from the environment (and a local
.env if present) so every default here can be overridden without editing
code. Copy .env.example to .env and uncomment what you want to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Anchor DB paths so they're independent of the current working directory:
# relative paths (defaults or relative env overrides) resolve against this
# file's directory (the repo root), while absolute env values are used as-is.
# This keeps reset_db.py / ingest.py / query_tools.py all pointed at the same
# DB regardless of where they're launched from. (Trade-off: the /tmp
# dev-sandbox workaround must copy the scripts alongside the data, or set an
# absolute SOURCE_DB/TARGET_DB env override.)
_ROOT = Path(__file__).resolve().parent


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _ROOT / p


# --- DB paths ---------------------------------------------------------------
SOURCE_DB = _resolve(os.getenv("SOURCE_DB", "source_data/chat.db"))
SNAPSHOT_DB = _resolve(os.getenv("SNAPSHOT_DB", "source_data/chat_snapshot.db"))
TARGET_DB = _resolve(os.getenv("TARGET_DB", "imessage_analysis.db"))
ADDRESS_BOOK_PATH = _resolve(os.getenv("ADDRESS_BOOK_PATH", "source_data/address_book.abcddb"))

# --- Ollama connectivity -----------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
# batch: {"input": [...]} -> {"embeddings": [...]}
OLLAMA_EMBED_PATH = os.getenv("OLLAMA_EMBED_PATH", "/api/embed")
OLLAMA_CHAT_PATH = os.getenv("OLLAMA_CHAT_PATH", "/api/chat")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "300"))

# --- Models -------------------------------------------------------------------
# nomic-embed-text, not the chat model: Ollama 0.32.0 refuses /api/embed for
# generation models (see "Known environment quirks" in CLAUDE.md).
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:26b-a4b-it-qat")
# Chain-of-thought control for thinking-capable chat models (e.g. qwen3):
# 'false' disables the hidden thinking pass (much faster per tool round-trip),
# 'true' forces it on. Unset (the default) omits the flag so Ollama uses the
# model's own default -- required for models that reject a "think" option.
_chat_think = os.getenv("CHAT_THINK", "").strip().lower()
CHAT_THINK = None if _chat_think == "" else _chat_think in ("1", "true", "yes")

# --- Tunables -----------------------------------------------------------------
# Embedding dimension override. Unset/0 (the default) means auto-detect the
# model's native output size with a probe request at ingest time. Setting it
# truncates + renormalizes vectors to the first EMBED_DIM components
# (Matryoshka reduction -- only meaningful for MRL-trained embedding models).
EMBED_DIM = int(os.getenv("EMBED_DIM") or "0")
# Task prefixes prepended to embedded text. The defaults are what
# nomic-embed-text (the default EMBED_MODEL) was trained with -- skipping
# them measurably degrades its retrieval quality. Set both to '' for models
# without task prefixes.
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # texts per Ollama request
# concurrent requests, matches Ollama's default OLLAMA_NUM_PARALLEL
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "4"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "20"))  # tool round-trips/turn
# Hard cap on one tool result's JSON (chars) before it goes into the chat
# context. Ollama sizes truncation per whole *message*: any single message
# larger than the model's context window is silently dropped along with all
# messages before it, leaving the model only the system prompt -- so an
# oversized tool result must be cut client-side. ~60k chars of dense JSON is
# ~20k tokens, comfortably inside the 32k window gemma4 runs with.
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "60000"))

# --- Conversation chunking ------------------------------------------------------
# Messages are embedded as conversation chunks, not individually: contiguous
# messages in one chat, split when the gap between two messages exceeds
# CHUNK_GAP_MINUTES (a session boundary) or the chunk text would exceed
# CHUNK_MAX_CHARS (carrying the last CHUNK_OVERLAP_MESSAGES messages into the
# next chunk for continuity). Chunks shorter than CHUNK_MIN_CHARS are stored
# but not embedded.
CHUNK_GAP_MINUTES = int(os.getenv("CHUNK_GAP_MINUTES", "60"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "10"))
CHUNK_OVERLAP_MESSAGES = int(os.getenv("CHUNK_OVERLAP_MESSAGES", "2"))
