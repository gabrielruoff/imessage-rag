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
#
# The tri-state matters: None means "omit the flag entirely", which is NOT the
# same as False ("send think: false"). Some models reject the option outright,
# so the distinction has to survive all the way to the request body.


def _think(name, default=None):
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes")


CHAT_THINK = _think("CHAT_THINK")

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
# Speed/quality preset (annotate.py --quality overrides). 'distilled' (the
# default) LLM-labels a random seed sample, then kNN-classifies every other
# chunk from its chunk_vectors embedding (~20s corpus-wide; valence/depth at
# LLM fidelity, emotions rougher, events left empty). The LLM-per-chunk
# tiers -- where on a reasoning model the hidden thinking pass dominates
# per-chunk wall time -- are 'max' (one chunk per request, model-default
# reasoning effort), 'balanced' (low effort, ~3x faster, agreement with
# 'max' within the model's own run-to-run noise but fewer events found)
# and 'fast' (low effort + batched requests, ~4-8x faster, noisier labels).
ANNOTATE_QUALITY = (os.getenv("ANNOTATE_QUALITY", "") or "distilled").strip().lower()
# Fine-grained preset overrides (empty/0 = use the preset's value).
# ANNOTATE_THINK becomes Ollama's `think`: 'low'/'medium'/'high' for
# reasoning-effort models (gpt-oss), 'true'/'false' for toggle models
# (qwen3), or 'omit'/'none' to send no flag at all -- the third state, which
# is distinct from false and which empty cannot express here (empty means
# "the preset's value"). gpt-oss silently returns EMPTY output for
# think=false -- use the effort levels there; ollama.chat_once_json notices
# that empty reply once and drops reasoning control for the rest of the run
# rather than failing every row.
ANNOTATE_THINK = os.getenv("ANNOTATE_THINK", "").strip().lower()
ANNOTATE_BATCH_SIZE = int(os.getenv("ANNOTATE_BATCH_SIZE") or "0")
# Unset = inherit CHAT_TIMEOUT. Worth setting independently: an annotation is
# one short JSON answer, but a thinking model can spend minutes on it, and a
# timeout raises out of the pass rather than skipping the row.
_annotate_timeout = os.getenv("ANNOTATE_TIMEOUT", "").strip()
ANNOTATE_TIMEOUT = int(_annotate_timeout) if _annotate_timeout else CHAT_TIMEOUT
# Concurrent LLM requests for the annotate/terms/seed passes. >1 only pays
# off when the Ollama server itself runs parallel slots (OLLAMA_NUM_PARALLEL
# -- `make ollama-start` sets 4); against a serial server extra workers just
# queue harmlessly. Measured ~2x at 4/4 on gpt-oss:20b (MoE decode is
# memory-bandwidth-bound, so it does not scale to 4x).
ANNOTATE_WORKERS = int(os.getenv("ANNOTATE_WORKERS", "4"))
# The distilled preset: LLM-labeled seed annotations to accumulate before
# the kNN pass (agreement plateaus around 300-450 seeds), and the number of
# nearest seed neighbours that vote for each classified chunk.
ANNOTATE_SEED_TARGET = int(os.getenv("ANNOTATE_SEED_TARGET", "450"))
ANNOTATE_KNN_K = int(os.getenv("ANNOTATE_KNN_K", "10"))

# --- Dyad-term curation (annotate.py --terms) -------------------------------
# Terms are their own pipeline, NOT a --quality tier: classifying one term is
# a four-way choice answered in one word, so a reasoning pass is pure latency.
# Measured 2026-07-17, gemma4:26b-a4b-it-qat, same term, same verdict every
# time: think='low' 282.6s, think unset 52.0s, think=false 26.2s. --quality's
# think used to leak in here and cost ~11x for an identical answer, so --terms
# now has its own default (false, below) instead of inheriting a chunk preset.
# Empty = false; 'omit'/'none' reaches the unset row of that table (52.0s).
# Set to 'low' on gpt-oss, which returns EMPTY output for think=false (see
# ANNOTATE_THINK above -- that empty reply is detected once and the run
# continues without reasoning control, so the mismatch costs a warning, not a
# failure counter on every term); ANNOTATE_THINK does NOT apply here.
ANNOTATE_TERMS_THINK = os.getenv("ANNOTATE_TERMS_THINK", "").strip().lower()
# Terms classified per request. Each term's prompt is ~150 chars and its
# answer is one enum, so batching costs almost nothing and cuts the request
# count ~25x; a batch never spans chats (the prompt is chat-scoped).
ANNOTATE_TERMS_BATCH_SIZE = int(os.getenv("ANNOTATE_TERMS_BATCH_SIZE") or "25")
# A term whose curation keeps failing is parked after this many failed
# attempts (dyad_terms.curate_failures) rather than being retried -- at full
# price, the retry nudge doubling it -- by every future run forever. Only a
# term's own dedicated request can charge it a failure, and a run in which
# nothing at all succeeds aborts without recording any: parking is for bad
# terms, and a bad config must not be able to spend it on the whole corpus.
# `annotate.py --terms --retry-failed` unparks them.
ANNOTATE_TERMS_MAX_FAILURES = int(os.getenv("ANNOTATE_TERMS_MAX_FAILURES") or "3")

# --- Topic clustering (topics.py) -------------------------------------------------
# Empty = use CHAT_MODEL (same convention as ANNOTATE_MODEL/WRAPPED_MODEL).
# The topic_model_k guard in ingest_state stays warn-only -- a rebuild just
# replaces the clustering, so mixing models across runs costs nothing.
TOPIC_MODEL = os.getenv("TOPIC_MODEL", "") or CHAT_MODEL
# Unset = inherit CHAT_THINK. Worth setting to false independently: the label
# pass is one short JSON object per cluster, where a thinking pass is pure
# latency (measured 85s -> 3.2s per call on qwen3:4b) and buys no quality.
TOPIC_THINK = _think("TOPIC_THINK", CHAT_THINK)
_topic_timeout = os.getenv("TOPIC_TIMEOUT", "").strip()
TOPIC_TIMEOUT = int(_topic_timeout) if _topic_timeout else CHAT_TIMEOUT
TOPIC_K = int(os.getenv("TOPIC_K", "50"))
TOPIC_KMEANS_ITERS = int(os.getenv("TOPIC_KMEANS_ITERS", "15"))
# Chunks shown to the LLM per cluster, ranked by cosine similarity to the
# cluster centroid. 30, not 5: the most-central chunks of a big cluster are
# also its *shortest* ("ok", "yeah") -- the centroid of 2700 chunks looks like
# filler -- so 5 samples can be a ~250-char prompt, and the model correctly but
# uselessly names it "Short confirmation responses". 30 reaches the actual
# topic. Raising this further needs a prompt-size check: at ~1200 chars per
# sample, 30 tops out around 36k chars (~12k tokens), and a message past the
# context window is silently dropped rather than erroring (see the Ollama
# truncation gotcha in CLAUDE.md).
TOPIC_LABEL_SAMPLES = int(os.getenv("TOPIC_LABEL_SAMPLES", "30"))

# --- wrapped.py --------------------------------------------------------------------
WRAPPED_MODEL = os.getenv("WRAPPED_MODEL", "") or CHAT_MODEL
# Unset = inherit CHAT_THINK / CHAT_TIMEOUT.
WRAPPED_THINK = _think("WRAPPED_THINK", CHAT_THINK)
_wrapped_timeout = os.getenv("WRAPPED_TIMEOUT", "").strip()
WRAPPED_TIMEOUT = int(_wrapped_timeout) if _wrapped_timeout else CHAT_TIMEOUT
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
