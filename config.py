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

# --- CLI progress -------------------------------------------------------------
# How the batch CLIs report progress. 'auto' (the default) draws a live rich
# progress bar when stdout is a terminal and falls back to periodic plain
# lines when piped or redirected; 'plain' forces the lines; 'off' silences
# progress entirely (start/summary lines still print).
PROGRESS_STYLE = os.getenv("PROGRESS_STYLE", "auto").strip().lower()
# Items between progress lines in plain (non-terminal) mode. Purely cosmetic
# -- it never affects how often a job commits.
PROGRESS_LOG_EVERY = int(os.getenv("PROGRESS_LOG_EVERY", "200"))

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
# Embedded chunks between commits during embed_chunks -- the resumability
# granularity if the run is interrupted (annotate.py's ANNOTATE_COMMIT_EVERY
# is the analogous knob there).
EMBED_COMMIT_EVERY = int(os.getenv("EMBED_COMMIT_EVERY", "200"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "20"))  # tool round-trips/turn
# Minimum gap (minutes) between two consecutive same-sender messages for the
# second to count as a double-text rather than a multi-part message.
DOUBLE_TEXT_MIN_MINUTES = int(os.getenv("DOUBLE_TEXT_MIN_MINUTES", "10"))
# Hard cap on one tool result's JSON (chars) before it goes into the chat
# context. Ollama sizes truncation per whole *message*: any single message
# larger than the model's context window is silently dropped along with all
# messages before it, leaving the model only the system prompt -- so an
# oversized tool result must be cut client-side. ~60k chars of dense JSON is
# ~20k tokens, comfortably inside the 32k window gemma4 runs with.
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "60000"))

# --- Analytics: sessions & response dynamics ----------------------------------
# Silence gap (minutes) that closes a session row in the `sessions` table.
# Distinct from CHUNK_GAP_MINUTES (embedding windows) and deliberately
# shorter than INITIATION_GAP_HOURS (who-texts-first attribution).
SESSION_GAP_MINUTES = int(os.getenv("SESSION_GAP_MINUTES", "60"))
# A session starting after at least this much silence counts as an initiation.
INITIATION_GAP_HOURS = int(os.getenv("INITIATION_GAP_HOURS", "12"))
# Max gap (minutes) between a reply and the message before it for the pair to
# count as an adjacency (who-replies-to-whom) in get_group_dynamics.
ADJACENCY_MINUTES = int(os.getenv("ADJACENCY_MINUTES", "5"))
# Local-clock hour window [start, end) counted as "late night" by
# get_activity_patterns.
LATE_NIGHT_START_HOUR = int(os.getenv("LATE_NIGHT_START_HOUR", "0"))
LATE_NIGHT_END_HOUR = int(os.getenv("LATE_NIGHT_END_HOUR", "5"))
# IANA timezone name for hour/weekday bucketing ('America/New_York'). Empty
# (the default) uses the system local timezone -- messages.date is stored as
# unlabeled UTC, and UTC-bucketed "late night" would be off by hours.
LOCAL_TZ = os.getenv("LOCAL_TZ", "")

# --- Analytics: lexicon & dyad terms --------------------------------------------
# A term must appear at least this often in a chat, and be at least this many
# times more frequent there than in the rest of the corpus, to be a candidate
# "your term" (dyad_terms); top K kept per chat.
DYAD_TERM_MIN_USES = int(os.getenv("DYAD_TERM_MIN_USES", "5"))
DYAD_TERM_MIN_LIFT = float(os.getenv("DYAD_TERM_MIN_LIFT", "5.0"))
DYAD_TERM_TOP_K = int(os.getenv("DYAD_TERM_TOP_K", "50"))

# --- Batch annotation (annotate.py) ----------------------------------------------
# Empty = use CHAT_MODEL. Recorded in ingest_state; mixing models in one
# chunk_annotations table is refused (same guard as embeddings).
ANNOTATE_MODEL = os.getenv("ANNOTATE_MODEL", "") or CHAT_MODEL
ANNOTATE_TOP_DYADS = int(os.getenv("ANNOTATE_TOP_DYADS", "20"))
ANNOTATE_COMMIT_EVERY = int(os.getenv("ANNOTATE_COMMIT_EVERY", "50"))

# --- Topic clustering (topics.py) -------------------------------------------------
TOPIC_K = int(os.getenv("TOPIC_K", "50"))
TOPIC_KMEANS_ITERS = int(os.getenv("TOPIC_KMEANS_ITERS", "15"))
TOPIC_LABEL_SAMPLES = int(os.getenv("TOPIC_LABEL_SAMPLES", "5"))

# --- wrapped.py --------------------------------------------------------------------
WRAPPED_MODEL = os.getenv("WRAPPED_MODEL", "") or CHAT_MODEL
WRAPPED_TOP_CONTACTS = int(os.getenv("WRAPPED_TOP_CONTACTS", "8"))
# Auto-exclude contacts who went abruptly silent (>= WRAPPED_SILENT_MONTHS
# before corpus end, >= WRAPPED_SILENT_MIN_MESSAGES lifetime) from the
# featured roster -- resurfacing an ex or the estranged uninvited is the
# classic nostalgia-product failure. --include overrides per name.
WRAPPED_SILENT_MONTHS = int(os.getenv("WRAPPED_SILENT_MONTHS", "6"))
WRAPPED_SILENT_MIN_MESSAGES = int(os.getenv("WRAPPED_SILENT_MIN_MESSAGES", "200"))

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
