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
OLLAMA_EMBEDDINGS_PATH = os.getenv(  # single-item embed endpoint
    "OLLAMA_EMBEDDINGS_PATH", "/api/embeddings"
)
OLLAMA_CHAT_PATH = os.getenv("OLLAMA_CHAT_PATH", "/api/chat")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "300"))

# --- Models -------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemma4:26b-a4b-it-qat")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:26b-a4b-it-qat")

# --- Tunables -----------------------------------------------------------------
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # gemma4:26b-a4b-it-qat output dimension
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # texts per Ollama request
# concurrent requests, matches Ollama's default OLLAMA_NUM_PARALLEL
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "4"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "20"))  # tool round-trips/turn
