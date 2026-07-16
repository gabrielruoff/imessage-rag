"""Shared config for the pipeline, loaded from the environment (and a local
.env if present) so every default here can be overridden without editing
code. Copy .env.example to .env and uncomment what you want to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- DB paths ---------------------------------------------------------------
SOURCE_DB = Path(os.getenv("SOURCE_DB", "chat.db"))
SNAPSHOT_DB = Path(os.getenv("SNAPSHOT_DB", "chat_snapshot.db"))
TARGET_DB = Path(os.getenv("TARGET_DB", "imessage_analysis.db"))
ADDRESS_BOOK_PATH = Path(os.getenv("ADDRESS_BOOK_PATH", "address_book.abcddb"))

# --- Ollama connectivity -----------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_EMBED_PATH = os.getenv("OLLAMA_EMBED_PATH", "/api/embed")  # batch: {"input": [...]} -> {"embeddings": [...]}
OLLAMA_EMBEDDINGS_PATH = os.getenv("OLLAMA_EMBEDDINGS_PATH", "/api/embeddings")  # single-item embed endpoint
OLLAMA_CHAT_PATH = os.getenv("OLLAMA_CHAT_PATH", "/api/chat")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "300"))

# --- Models -------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:26b-a4b-it-qat")

# --- Tunables -----------------------------------------------------------------
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # nomic-embed-text output dimension
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # texts per Ollama request
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "4"))  # concurrent requests, matches Ollama's default OLLAMA_NUM_PARALLEL
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))  # max tool round-trips per chat.py turn
