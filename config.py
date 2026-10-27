"""Shared config for the pipeline, loaded from the environment (and a local
.env if present) so every default here can be overridden without editing code.
Copy .env.example to .env and uncomment what you want to change.

This module is the single source of truth for what every knob MEANS. The
measurements and dead ends behind the defaults live in docs/DECISIONS.md;
comments here state the contract, not the history.
"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv

# DB paths resolve against THIS FILE's directory, not the CWD, so every CLI
# points at the same DB whatever it is launched from. Absolute env values win.
_ROOT = Path(__file__).resolve().parent

# .env is loaded from _ROOT ONLY. A bare load_dotenv() walks UP the tree, so a
# checkout under .claude/worktrees/<name>/ would silently inherit the parent
# checkout's flags, models and DB paths.
load_dotenv(_ROOT / ".env")


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

# --- Web API server (api/, single-user local UI) -----------------------------
# app.db holds the web UI's own conversation history (plain sqlite3, no
# sqlite-vec), kept separate from the analysis DB so reset_db.py can empty the
# latter without destroying chat history. Anchored like the other DBs.
APP_DB = _resolve(os.getenv("APP_DB", "app.db"))
# uvicorn bind + the dev frontend's CORS origins (Vite defaults to 5173).
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

# --- Models -------------------------------------------------------------------
# nomic-embed-text, not the chat model: Ollama 0.32.0 refuses /api/embed for
# generation models (docs/DECISIONS.md, "Known environment quirks").
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# The model every timing default in this repo was measured on, and ~4 GB
# resident against a 26B MoE's ~20 GB. A bigger CHAT_MODEL routes more precisely
# across the ~32 retrieval tools. The *_MODEL vars below inherit it when empty;
# EMBED_MODEL deliberately does NOT (it is dimensioned into chunk_vectors, so
# changing it is a reset_db + re-ingest).
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")


# `think` is TRI-STATE and the distinction has to survive to the request body:
# None omits the flag entirely (some models reject the option), False sends
# `think: false`, True forces a reasoning pass. None is NOT False.
def _think(name, default=None):
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes")


CHAT_THINK = _think("CHAT_THINK")


def _positive_int_env(name, default):
    """An int knob that must be > 0; an empty value means unset."""
    raw = os.getenv(name) or str(default)
    if int(raw) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


# Model families that read 'low'/'medium'/'high' as a reasoning-EFFORT LEVEL.
# To every other model an effort string is merely TRUTHY and buys a FULL pass,
# which is why an effort level is not portable. The EFFORT family is listed
# rather than the toggle ones because that set is small and named; an unlisted
# model is assumed to toggle, and the failure mode is a cheaper answer.
_EFFORT_LEVEL_MODELS = ("gpt-oss",)


def has_effort_levels(name) -> bool:
    """True when `name` understands 'low'/'medium'/'high' as an effort level
    rather than as a plain truthy "think harder". See _EFFORT_LEVEL_MODELS."""
    return isinstance(name, str) and any(
        family in name.lower() for family in _EFFORT_LEVEL_MODELS)

# --- Apple Foundation Models backend (opt-in) --------------------------------
# On-device backend via Apple Intelligence (Apple Silicon, macOS 26+), through a
# local Swift shim (`make apple-shim`). Opt in by setting any *_MODEL var to the
# "apple" sentinel; the defaults above stay Ollama's. The shim's context window
# is a fixed 4,096 tokens, so the apple chat path stages many small sessions
# (chat_apple.py) instead of context-stuffing.


def is_apple_model(name) -> bool:
    """True when a *_MODEL value selects the Apple Foundation Models backend:
    the bare "apple" sentinel, or an "apple:<variant>" form reserved for future
    model variants. Every dispatch seam (ollama.py, embeddings.py, chat.py)
    gates on this so all other callers stay backend-agnostic."""
    return isinstance(name, str) and (name == "apple" or name.startswith("apple:"))


# The compiled Swift shim, built by `make apple-shim`; anchored like the DBs.
APPLE_SHIM_PATH = _resolve(os.getenv("APPLE_SHIM_PATH", "bin/applefm-shim"))
# Per-tool-result char cap for the apple chat orchestrator, far below
# TOOL_RESULT_MAX_CHARS: a result over this is DIGESTED down to
# question-relevant facts, never truncated blind.
APPLE_TOOL_RESULT_MAX_CHARS = int(os.getenv("APPLE_TOOL_RESULT_MAX_CHARS", "5000"))
# Per-stage output-token caps, forwarded as GenerationOptions.
# maximumResponseTokens -- the main per-call speed lever FM exposes (it has no
# reasoning-effort flag). 0 = uncapped, which ANSWER stays by default: it is the
# user-facing reply and must not be clipped mid-sentence. FILL is generous so a
# valid args object is never truncated into a {} fallback.
# APPLE_DIGEST_MAX_TOKENS caps TOKENS, not characters, so it can bind below
# chat_apple._SCRATCHPAD_BUDGET -- raise it if digest output looks truncated.
APPLE_ROUTE_MAX_TOKENS = int(os.getenv("APPLE_ROUTE_MAX_TOKENS", "128"))
APPLE_FILL_MAX_TOKENS = int(os.getenv("APPLE_FILL_MAX_TOKENS", "256"))
APPLE_DIGEST_MAX_TOKENS = int(os.getenv("APPLE_DIGEST_MAX_TOKENS", "512"))
APPLE_ANSWER_MAX_TOKENS = int(os.getenv("APPLE_ANSWER_MAX_TOKENS", "0"))
# Apple chat loop controls, SEPARATE from MAX_TOOL_ITERATIONS so tuning the
# staged on-device loop never shortens the Ollama one. APPLE_NO_PROGRESS_LIMIT
# is the number of consecutive no-signal iterations (duplicate call, error, or
# zero rows) after which the loop answers from what it has.
APPLE_MAX_TOOL_ITERATIONS = int(os.getenv("APPLE_MAX_TOOL_ITERATIONS", "8"))
APPLE_NO_PROGRESS_LIMIT = int(os.getenv("APPLE_NO_PROGRESS_LIMIT", "2"))

# --- Hosted backends (opt-in, paid, uploads text) ------------------------------
# Selected by a model-name PREFIX, like the apple sentinel: EMBED_MODEL=
# "voyage/<id>" (Voyage AI) and ANNOTATE_MODEL="openrouter/<id>" (OpenRouter,
# annotate.py per-chunk tiers only). Choosing the model is the upload consent;
# every hosted run prints what it uploads and where (hosted.upload_notice).


def is_openrouter_model(name) -> bool:
    """True when a *_MODEL value selects OpenRouter. annotate.py dispatches its
    per-chunk tiers on it; ollama.py's seam (chat_once, chat_once_json,
    chat_stream) refuses it, so no other pipeline reaches the cloud."""
    return isinstance(name, str) and name.startswith("openrouter/")


def is_voyage_model(name) -> bool:
    """True when EMBED_MODEL selects Voyage AI. embeddings.py (query: sync;
    document: batch-only), chunking/ingest (the batch path) and the EMBED_DIM
    rule below gate on it."""
    return isinstance(name, str) and name.startswith("voyage/")


def hosted_model_id(name) -> str:
    """The provider's model id: the "openrouter/" or "voyage/" prefix stripped.
    The full prefixed value is what guards and ingest_state record."""
    for prefix in ("openrouter/", "voyage/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# API keys, read here only; never printed or logged.
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
# Spend cap (USD) shared by both clients. No default on purpose: a hosted
# ingest/annotate run refuses to start without this or its --budget flag.
HOSTED_BUDGET_USD = os.getenv("HOSTED_BUDGET_USD", "").strip() or None
if HOSTED_BUDGET_USD is not None:
    HOSTED_BUDGET_USD = float(HOSTED_BUDGET_USD)
# nan/inf would disable the cap, so only a finite positive value is accepted.
if HOSTED_BUDGET_USD is not None and not (
        math.isfinite(HOSTED_BUDGET_USD) and HOSTED_BUDGET_USD > 0):
    raise ValueError(
        f"HOSTED_BUDGET_USD must be finite and > 0, got {HOSTED_BUDGET_USD!r}")
# Max hosted requests in flight per client per run (--max-parallel). 1 = strictly
# sequential. Separate from EMBED_MAX_WORKERS/ANNOTATE_WORKERS (Ollama only).
HOSTED_MAX_PARALLEL = _positive_int_env("HOSTED_MAX_PARALLEL", 1)
# hosted.Limiter halves its effective parallelism on a 429 and climbs back by 1
# after this many consecutive successes (never above --max-parallel).
HOSTED_LIMIT_RECOVER_AFTER = _positive_int_env("HOSTED_LIMIT_RECOVER_AFTER", 10)
# Output-token cap per OpenRouter annotation request; also the worst-case
# completion size the budget reserves.
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS") or "1024")
# Read timeout (s) for one hosted completion / sync embedding request.
HOSTED_TIMEOUT = int(os.getenv("HOSTED_TIMEOUT") or "180")
# Timeout (s) for a Voyage batch file upload or download.
VOYAGE_TIMEOUT = int(os.getenv("VOYAGE_TIMEOUT") or "600")
# Seconds between Voyage batch status polls.
VOYAGE_POLL_SECONDS = int(os.getenv("VOYAGE_POLL_SECONDS") or "30")
# Cap (s) on the exponential retry backoff when no Retry-After is given.
HOSTED_BACKOFF_MAX_SECONDS = int(os.getenv("HOSTED_BACKOFF_MAX_SECONDS") or "60")
# Voyage batch price, USD per million tokens; the budget reserves UTF-8 bytes x
# this before each upload and settles at the reported usage.
VOYAGE_BATCH_PRICE_PER_M = float(os.getenv("VOYAGE_BATCH_PRICE_PER_M") or "0.0804")
# Voyage batch limit: inputs per batch.
VOYAGE_BATCH_MAX_INPUTS = int(os.getenv("VOYAGE_BATCH_MAX_INPUTS") or "100000")

# --- CLI progress -------------------------------------------------------------
# 'auto' draws a live bar on a terminal and falls back to periodic plain lines
# when piped; 'plain' forces the lines; 'off' silences progress (start/summary
# lines still print).
PROGRESS_STYLE = os.getenv("PROGRESS_STYLE", "auto").strip().lower()
# Items between progress lines in plain mode. Cosmetic -- never commit cadence.
PROGRESS_LOG_EVERY = int(os.getenv("PROGRESS_LOG_EVERY", "200"))

# --- Tunables -----------------------------------------------------------------
# Embedding dimension override. Unset/0 = auto-detect with a probe request at
# ingest time. Setting it truncates + renormalizes to the first EMBED_DIM
# components (Matryoshka reduction -- only meaningful for MRL-trained models).
EMBED_DIM = int(os.getenv("EMBED_DIM") or "0")
# On a voyage/ EMBED_MODEL, EMBED_DIM is instead Voyage's output_dimension: no
# probe, no truncation; unset means 1024, and only these values are accepted.
VOYAGE_OUTPUT_DIMS = (256, 512, 1024, 2048)
if is_voyage_model(EMBED_MODEL):
    EMBED_DIM = EMBED_DIM or 1024
    if EMBED_DIM not in VOYAGE_OUTPUT_DIMS:
        raise ValueError(f"EMBED_DIM must be one of {VOYAGE_OUTPUT_DIMS} for "
                         f"{EMBED_MODEL}, got {EMBED_DIM}")
# Task prefixes prepended to embedded text; these are what nomic-embed-text was
# trained with, and skipping them measurably degrades retrieval. Set both to ''
# for models without task prefixes.
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # texts per Ollama request
# concurrent requests, matches Ollama's default OLLAMA_NUM_PARALLEL
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "4"))
# Embedded chunks between commits: the resumability granularity of embed_chunks
# (ANNOTATE_COMMIT_EVERY is the analogous knob for annotate.py).
EMBED_COMMIT_EVERY = int(os.getenv("EMBED_COMMIT_EVERY", "200"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "20"))  # tool round-trips/turn
# Minimum gap (minutes) between two consecutive same-sender messages for the
# second to count as a double-text rather than a multi-part message.
DOUBLE_TEXT_MIN_MINUTES = int(os.getenv("DOUBLE_TEXT_MIN_MINUTES", "10"))

# Hard cap on one tool result's JSON (chars) before it enters the chat context.
# Ollama truncates per whole MESSAGE: a single message larger than the window is
# dropped along with everything before it, leaving only the system prompt, so an
# oversized result must be cut client-side. ~60k chars is ~20k tokens.
TOOL_RESULT_MAX_CHARS = _positive_int_env("TOOL_RESULT_MAX_CHARS", 60000)
# Hard cap on one MCP tool result's JSON (chars; mcp_server). Sized under Claude
# Code's default 25k-token MCP output limit even at the worst case of ~2
# chars/token for \u-escaped emoji. Over the cap, the result is cut at row
# boundaries with a note rather than mangled by the client. Separate from
# TOOL_RESULT_MAX_CHARS, which is sized to Ollama's whole-message window drop.
MCP_RESULT_MAX_CHARS = _positive_int_env("MCP_RESULT_MAX_CHARS", 40000)

# --- Analytics: sessions & response dynamics ----------------------------------
# Silence gap (minutes) that closes a `sessions` row. Distinct from
# CHUNK_GAP_MINUTES (embedding windows) and deliberately shorter than
# INITIATION_GAP_HOURS (who-texts-first attribution).
SESSION_GAP_MINUTES = int(os.getenv("SESSION_GAP_MINUTES", "60"))
# A session starting after at least this much silence counts as an initiation.
INITIATION_GAP_HOURS = int(os.getenv("INITIATION_GAP_HOURS", "12"))
# Max gap (minutes) between a reply and the message before it for the pair to
# count as an adjacency (who-replies-to-whom) in get_group_dynamics.
ADJACENCY_MINUTES = int(os.getenv("ADJACENCY_MINUTES", "5"))
# Local-clock hour window [start, end) counted as "late night".
LATE_NIGHT_START_HOUR = int(os.getenv("LATE_NIGHT_START_HOUR", "0"))
LATE_NIGHT_END_HOUR = int(os.getenv("LATE_NIGHT_END_HOUR", "5"))
# IANA timezone for hour/weekday bucketing. Empty = the system local zone.
# messages.date is unlabeled UTC, so UTC-bucketed "late night" is off by hours.
LOCAL_TZ = os.getenv("LOCAL_TZ", "")

# --- Analytics: lexicon & dyad terms --------------------------------------------
# A term must appear at least MIN_USES times in a chat and be at least MIN_LIFT
# times more frequent there than in the rest of the corpus to be a candidate
# "your term" (dyad_terms); top K kept per chat.
DYAD_TERM_MIN_USES = int(os.getenv("DYAD_TERM_MIN_USES", "5"))
DYAD_TERM_MIN_LIFT = float(os.getenv("DYAD_TERM_MIN_LIFT", "5.0"))
DYAD_TERM_TOP_K = int(os.getenv("DYAD_TERM_TOP_K", "50"))

# --- Batch annotation (annotate.py) ----------------------------------------------
# Empty = use CHAT_MODEL. Recorded in ingest_state; mixing models in one
# chunk_annotations table is refused (db.check_model_guard). The refusal comes at
# the TOP of a run, before any budget clock starts, so a switched model costs
# nothing. --dry-run does NOT run the check, because the check RECORDS
# annotate_model and a preview that writes is not a preview.
ANNOTATE_MODEL = os.getenv("ANNOTATE_MODEL", "") or CHAT_MODEL
# Chats the LLM tiers scope to (top N by message count); --top-dyads N
# overrides, --all removes the scope. hybrid applies it to the UPGRADE phase
# only (its seed and kNN phases are corpus-wide) and the distilled tier ignores
# it entirely. 0 = no chats. A NEGATIVE value is REFUSED wherever it is resolved
# (annotate.main, hybrid_chunks, annotate_chunks): it reaches the scope
# subquery's `LIMIT ?`, where SQLite reads -1 as NO limit, so it would silently
# widen a scoped run to --all. Only on runs that BIND it -- --terms, --all and
# the distilled tier never do, and a bad knob must not refuse a run it cannot
# affect.
ANNOTATE_TOP_DYADS = int(os.getenv("ANNOTATE_TOP_DYADS", "20"))
# Annotated chunks between commits: the resumability granularity on Ctrl-C.
ANNOTATE_COMMIT_EVERY = int(os.getenv("ANNOTATE_COMMIT_EVERY", "50"))
# Speed/quality preset (--quality overrides):
#   distilled (default) LLM-labels a random seed sample then kNN-classifies the
#     rest from chunk_vectors -- fast, corpus-wide, events left EMPTY
#   hybrid      stratified seeds + that same kNN pass, then LLM upgrades of the
#     event-candidate and lowest-confidence chunks until ANNOTATE_BUDGET_MINUTES
#   max/balanced/fast  one LLM label per chunk, at decreasing reasoning effort
#     and increasing batching
ANNOTATE_QUALITY = (os.getenv("ANNOTATE_QUALITY", "") or "distilled").strip().lower()
# Fine-grained preset overrides (empty/0 = use the preset's value).
# ANNOTATE_THINK becomes Ollama's `think`: 'low'/'medium'/'high' for
# effort-level models, 'true'/'false' for toggle models, or 'omit'/'none' to
# send no flag at all -- the third state, which empty cannot express here.
# gpt-oss returns EMPTY output for think=false, so use its effort levels;
# ollama.chat_once_json notices that empty reply once and drops reasoning
# control for the rest of the run rather than failing every row.
ANNOTATE_THINK = os.getenv("ANNOTATE_THINK", "").strip().lower()
ANNOTATE_BATCH_SIZE = int(os.getenv("ANNOTATE_BATCH_SIZE") or "0")
# Unset = 900s, NOT CHAT_TIMEOUT -- the one *_TIMEOUT that does not inherit.
# CHAT_TIMEOUT budgets one interactive turn; an annotate request budgets a
# BATCH (ANNOTATE_BATCH_SIZE / the --quality preset, 8 under 'hybrid') of
# chunks, issued ANNOTATE_WORKERS-at-a-time against a server with a fixed
# number of parallel slots, so the wall clock a batched request sees is
# batch x per-chunk x contention. Inheriting 300 timed out the seed pass on a
# 9b model at batch 8 (2026-08-03) -- and a thinking model can spend minutes
# on a single row besides. A too-generous ceiling costs only latency on the
# rare runaway, since a timed-out batch now drops to the cheaper rungs
# (annotate._annotate_batch) rather than aborting the pass.
_annotate_timeout = os.getenv("ANNOTATE_TIMEOUT", "").strip()
ANNOTATE_TIMEOUT = int(_annotate_timeout) if _annotate_timeout else 900
# Concurrent LLM requests for the annotate/terms/seed passes. >1 only pays off
# when the Ollama server runs parallel slots (`make ollama-start` sets 4);
# against a serial server extra workers queue harmlessly. The payoff is
# MODEL-dependent and modest on the default model -- ~1.24x, not a multiplier.
ANNOTATE_WORKERS = int(os.getenv("ANNOTATE_WORKERS", "4"))
# The distilled preset: LLM-labeled seeds to accumulate before the kNN pass
# (agreement plateaus around 300-450), and how many nearest seed neighbours vote
# for each classified chunk.
ANNOTATE_SEED_TARGET = int(os.getenv("ANNOTATE_SEED_TARGET", "450"))
ANNOTATE_KNN_K = int(os.getenv("ANNOTATE_KNN_K", "10"))
# Ceiling on LLM seeds loaded into the temporary kNN index the classify and
# confidence-backfill passes vote against; `<= 0` = uncapped (a stray sign lands
# on the documented pre-cap behavior rather than on "an index of exactly
# _KNN_MIN_SEEDS"). Deliberately NOT refused the way a negative --limit or
# budget is: those change what a run DOES, while an uncapped index only changes
# how long two deterministic, LLM-free passes take.
#
# It caps the INDEX, never the stored pool: every source='llm' row is still
# written, counted toward the seed targets and read by every other consumer. A
# capped run votes off a deterministic sample (annotate_distill._sample_seed_ids)
# drawn proportionally to each chat's share of the CORPUS -- not uniformly over
# the pool, which is only whatever earlier runs happened to label. It cannot fix
# a genuinely single-chat pool (widening the POOL is the seed sampler's job);
# it stops the cap re-narrowing what the sampler widened. Never applied below
# annotate_distill._KNN_MIN_SEEDS. Measurements, and what capping costs against
# the control of simply resampling the pool: docs/DECISIONS.md, "the kNN index
# inherited one dyad".
ANNOTATE_KNN_INDEX_CAP = int(os.getenv("ANNOTATE_KNN_INDEX_CAP") or "800")
# The hybrid preset's wall-clock budget, in minutes. Empty/0 = UNSET, and
# annotate_hybrid resolves a per-backend default (25 Ollama / 50 Apple FM)
# through is_apple_model(model) -- that default depends on the model chosen at
# run time, so config only parses the env int. A NEGATIVE value is REFUSED where
# it is resolved, naming the knob, rather than yielding an already-passed
# deadline that declines every phase for no stated reason
# (CONFLICT_BUDGET_MINUTES validates the same way; only the meaning of 0
# differs -- there it is "unbudgeted").
#
# Budgeting wall clock rather than a chunk count is what makes the tier meet its
# target BY CONSTRUCTION: it holds on any hardware and corpus size, and a faster
# machine degrades to MORE coverage. EVERY phase is gated, not just the LLM
# ones. The floor worth knowing: nothing smaller than one request can be
# dispatched, so a budget whose phase share is under one request's latency buys
# NO UPGRADES and says so -- which is not "the run did nothing", since the
# deterministic phases still run and commit what they reach. Below ~5 minutes
# declining the upgrade phases is the common outcome. Sizing measurements and
# what a run actually buys: docs/DECISIONS.md, "annotation throughput, round 2".
ANNOTATE_BUDGET_MINUTES = int(os.getenv("ANNOTATE_BUDGET_MINUTES") or "0")
# Seeds the hybrid tier's stratified sampler aims to have drawn before its kNN
# pass -- a separate knob from ANNOTATE_SEED_TARGET so the distilled tier's ~90s
# profile is untouched. The sampler buys ~4x fewer near-duplicate seeds and
# deterministic proportional chat representation, NOT better corpus-wide kNN
# coverage (measured a wash); the extra seeds are what widen the vote.
#
# Counted against a `hybrid_seeds` counter in ingest_state, NOT against
# COUNT(source='llm'): the tier's own upgrades are 'llm' rows, so counting those
# let its output satisfy its own target and switched the sampler off for good.
# So a corpus already annotated by another tier still pays for this many sampler
# seeds -- the target is about the SHAPE of the pool, not its size. It is a
# TARGET, not a per-run quota: seeding gets a bounded slice of the budget
# (annotate_hybrid._SEED_PHASE_SHARE), so the pool fills over several runs.
ANNOTATE_HYBRID_SEED_TARGET = int(os.getenv("ANNOTATE_HYBRID_SEED_TARGET", "800"))

# --- Dyad-term curation (annotate.py --terms) -------------------------------
# Terms are their own pipeline, NOT a --quality tier: classifying one term is a
# four-way choice answered in one word, so a reasoning pass is pure latency
# (measured ~11x for an identical verdict). --quality's think used to leak in
# here, so --terms has its own default. Empty = false; 'omit'/'none' sends no
# flag. Set 'low' on gpt-oss, which returns EMPTY output for think=false.
# ANNOTATE_THINK does NOT apply here.
ANNOTATE_TERMS_THINK = os.getenv("ANNOTATE_TERMS_THINK", "").strip().lower()
# Terms per request. Each prompt is ~150 chars and each answer is one enum, so
# batching cuts the request count ~25x; a batch never spans chats (the prompt is
# chat-scoped).
ANNOTATE_TERMS_BATCH_SIZE = int(os.getenv("ANNOTATE_TERMS_BATCH_SIZE") or "25")
# A term whose curation keeps failing is PARKED after this many failed attempts
# (dyad_terms.curate_failures) rather than retried at full price by every future
# run forever. Only a term's own dedicated request can charge it a failure, and
# a run in which nothing at all succeeds records none: parking is for bad terms,
# and a bad config must not spend it on the whole corpus. --retry-failed unparks.
ANNOTATE_TERMS_MAX_FAILURES = int(os.getenv("ANNOTATE_TERMS_MAX_FAILURES") or "3")

# --- Topic clustering (topics.py) -------------------------------------------------
# Empty = use CHAT_MODEL. The topic_model_k guard in ingest_state is warn-only:
# a rebuild just replaces the clustering, so mixing models costs nothing.
TOPIC_MODEL = os.getenv("TOPIC_MODEL", "") or CHAT_MODEL
# Unset = inherit CHAT_THINK. Worth setting false independently: the label pass
# is one short JSON object per cluster, where thinking is pure latency.
TOPIC_THINK = _think("TOPIC_THINK", CHAT_THINK)
_topic_timeout = os.getenv("TOPIC_TIMEOUT", "").strip()
TOPIC_TIMEOUT = int(_topic_timeout) if _topic_timeout else CHAT_TIMEOUT
TOPIC_K = int(os.getenv("TOPIC_K", "50"))
TOPIC_KMEANS_ITERS = int(os.getenv("TOPIC_KMEANS_ITERS", "15"))
# Chunks shown to the LLM per cluster, ranked by similarity to the centroid.
# 30, not 5: a big cluster's most-central chunks are also its SHORTEST ("ok",
# "yeah"), so a small sample gets correctly but uselessly labeled "Short
# confirmation responses". Raising it past ~30 risks the context window, where
# an oversized message is silently DROPPED rather than erroring.
TOPIC_LABEL_SAMPLES = int(os.getenv("TOPIC_LABEL_SAMPLES", "30"))

# --- wrapped.py --------------------------------------------------------------------
WRAPPED_MODEL = os.getenv("WRAPPED_MODEL", "") or CHAT_MODEL
# Unset = inherit CHAT_THINK / CHAT_TIMEOUT.
WRAPPED_THINK = _think("WRAPPED_THINK", CHAT_THINK)
_wrapped_timeout = os.getenv("WRAPPED_TIMEOUT", "").strip()
WRAPPED_TIMEOUT = int(_wrapped_timeout) if _wrapped_timeout else CHAT_TIMEOUT
WRAPPED_TOP_CONTACTS = int(os.getenv("WRAPPED_TOP_CONTACTS", "8"))
# Auto-exclude contacts who went abruptly silent (>= WRAPPED_SILENT_MONTHS
# before corpus end, >= WRAPPED_SILENT_MIN_MESSAGES lifetime) from the featured
# roster -- resurfacing an ex or the estranged uninvited is the classic
# nostalgia-product failure. --include overrides per name.
WRAPPED_SILENT_MONTHS = int(os.getenv("WRAPPED_SILENT_MONTHS", "6"))
WRAPPED_SILENT_MIN_MESSAGES = int(os.getenv("WRAPPED_SILENT_MIN_MESSAGES", "200"))
# --- feature: persona ---
# Third-party "Your Voiceprint" game consent (RES-64 / V4): comma/space-
# separated `contacts.id` values the user marked OK to include. Empty (the
# default) = self-only: NO third-party verbatim text or stylometric card appears
# in the generated file. Consent ADDS a contact; it never overrides a roster
# exclusion (silent-contact guard / --contact / --exclude).
WRAPPED_VOICEPRINT_CONSENT_IDS = os.getenv("WRAPPED_VOICEPRINT_CONSENT_IDS", "")
# --- end feature: persona ---
# --- feature: wellbeing ---
# Opt-in Wrapped wellbeing trajectory section (RES-68 / W4). OFF by default; the
# section is built only when this is truthy OR wrapped.py --wellbeing is passed
# (the flag overrides for that run). A rendered window carries the same
# code-emitted support footer as the chat surface (wellbeing.support_block), and
# all of its narration runs through wellbeing.sanitize_condition_names.
WRAPPED_WELLBEING = os.getenv("WRAPPED_WELLBEING", "").strip().lower() in (
    "1", "true", "yes", "on")
# --- end feature: wellbeing ---

# --- Conversation chunking ------------------------------------------------------
# Messages are embedded as conversation chunks, not individually: contiguous
# messages in one chat, split when the gap exceeds CHUNK_GAP_MINUTES or the text
# would exceed CHUNK_MAX_CHARS (carrying CHUNK_OVERLAP_MESSAGES into the next
# chunk). Chunks under CHUNK_MIN_CHARS are stored but not embedded.
CHUNK_GAP_MINUTES = int(os.getenv("CHUNK_GAP_MINUTES", "60"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "10"))
CHUNK_OVERLAP_MESSAGES = int(os.getenv("CHUNK_OVERLAP_MESSAGES", "2"))

# --- Eval gates (eval/evalkit label/gate; RES-70 / F4) ----------------------
# Per-class support floors: a class with fewer than this many gold labels scores
# None rather than a statistic from too few examples, and can never clear the
# gate. Precision stabilizes with fewer examples than a full kappa needs, hence
# the two numbers.
EVAL_GATE_MIN_LABELS = int(os.getenv("EVAL_GATE_MIN_LABELS") or "50")
EVAL_GATE_MIN_PRECISION_POSITIVES = int(os.getenv("EVAL_GATE_MIN_PRECISION_POSITIVES") or "40")
# --- feature: nudges ---
# --- Loose Ends / nudges (loose_ends.py) --------------------------------------
# NUDGES_MODEL/THINK/TIMEOUT follow the usual inherit-from-CHAT rule and are used
# for the one-line nudge_text narration and (when enabled) dropped_plan's
# verification; the other detectors are model-free SQL.
NUDGES_MODEL = os.getenv("NUDGES_MODEL", "") or CHAT_MODEL
NUDGES_THINK = _think("NUDGES_THINK", CHAT_THINK)
_nudges_timeout = os.getenv("NUDGES_TIMEOUT", "").strip()
NUDGES_TIMEOUT = int(_nudges_timeout) if _nudges_timeout else CHAT_TIMEOUT
# Stage-0 auto-exclusion: a dyad whose most-recent llm-annotated exchange has
# mean valence <= this is dropped (a negative sign-off is not a nudge). Only
# source='llm' rows count; no llm rows = no valence exclusion on that dyad.
NUDGE_VALENCE_EXCLUDE = float(os.getenv("NUDGE_VALENCE_EXCLUDE") or "-0.5")
# Stage-0 identity exclusions. The schema stores no owner identity, so a
# self-chat (your own second number, saved under your own name) would otherwise
# surface as a nudge: list your own contact_id / handle(s) here once.
# NUDGE_SHORTCODE_MAX_DIGITS is the length ceiling for treating an all-digit
# handle as an automated SMS shortcode; real 10-15 digit numbers are unaffected.
NUDGE_OWNER_CONTACT_IDS = os.getenv("NUDGE_OWNER_CONTACT_IDS", "")
NUDGE_OWNER_HANDLES = os.getenv("NUDGE_OWNER_HANDLES", "")
NUDGE_SHORTCODE_MAX_DIGITS = int(os.getenv("NUDGE_SHORTCODE_MAX_DIGITS") or "6")
# faded_initiator fires when the counterparty's trailing-window session
# initiation share falls to <= this fraction of their all-time share...
NUDGE_FADED_INIT_DROP = float(os.getenv("NUDGE_FADED_INIT_DROP") or "0.5")
# ...and only for dyads with at least this many lifetime sessions...
NUDGE_FADED_INIT_MIN_SESSIONS = int(os.getenv("NUDGE_FADED_INIT_MIN_SESSIONS") or "10")
# ...measuring "trailing" over this many days back from the corpus's latest
# message (the data's "now").
NUDGE_FADED_INIT_WINDOW_DAYS = int(os.getenv("NUDGE_FADED_INIT_WINDOW_DAYS") or "90")
# Urgency = kind_weight * exp(-age_days / halflife * ln 2); age is measured from
# the corpus's latest message so the freshest loose threads score highest.
NUDGE_URGENCY_HALFLIFE_DAYS = float(os.getenv("NUDGE_URGENCY_HALFLIFE_DAYS") or "30")
# Hard cap on get_loose_threads rows (a nudge digest is a handful, not a feed).
NUDGE_MAX_RESULTS = int(os.getenv("NUDGE_MAX_RESULTS") or "5")


def _nudge_kind_weights():
    """Per-kind urgency weights, with a JSON env override merged over the
    defaults (a partial override keeps the other defaults). Malformed JSON or a
    non-object falls back to the defaults rather than raising at import time."""
    import json
    default = {
        "unanswered_question": 1.0,
        "faded_initiator": 0.7,
        "dormant_tie": 0.6,
        "dropped_plan": 0.9,
        "unreciprocated_goodnews": 0.5,
        "inferred_occasion": 0.8,
    }
    raw = os.getenv("NUDGE_KIND_WEIGHTS", "").strip()
    if not raw:
        return default
    try:
        override = json.loads(raw)
    except (ValueError, TypeError):
        return default
    if not isinstance(override, dict):
        return default
    merged = dict(default)
    for key, value in override.items():
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue
    return merged


NUDGE_KIND_WEIGHTS = _nudge_kind_weights()
# --- end feature: nudges ---
# --- feature: drift ---
# Relationship-drift pipeline (drift.py). DRIFT_ENABLED gates it end-to-end:
# false (the default) makes the `make pipeline` drift step a no-op and
# dyad_drift is never built. DRIFT_WEIGHTS / DRIFT_PHASE_THRESHOLDS are
# JSON-valued and parsed once here, with a clear ValueError on malformed input
# (a bad .env must fail loudly at import, not silently later).
import json as _json  # noqa: E402  (local to the drift block; keeps the edit additive)

DRIFT_ENABLED = os.getenv("DRIFT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# A dyad needs at least this many total messages, distinct active months, and
# (per signal) valid monthly data points to be scored; robust_slope itself needs
# >= 4 points, so DRIFT_MIN_VALID_MONTHS below 4 makes every trend absent.
DRIFT_MIN_MESSAGES = int(os.getenv("DRIFT_MIN_MESSAGES") or "100")
DRIFT_MIN_MONTHS = int(os.getenv("DRIFT_MIN_MONTHS") or "4")
DRIFT_MIN_VALID_MONTHS = int(os.getenv("DRIFT_MIN_VALID_MONTHS") or "4")
# A dyad needs this many gap-split sessions before the initiation-asymmetry
# signal is computed at all (too few sessions makes the share meaningless).
DRIFT_MIN_SESSIONS = int(os.getenv("DRIFT_MIN_SESSIONS") or "5")
# Confidence level for every trend's bootstrap CI (robust_slope alpha): a CI
# that still spans 0 at this level contributes 0 to the composite.
DRIFT_SLOPE_CI = float(os.getenv("DRIFT_SLOPE_CI") or "0.05")

# HEURISTIC, NOT VALIDATED. These weights and phase cutoffs are a hand-tuned
# starting point, not a calibrated model: no study establishes that this
# composite tracks real relationship "drift", and copy built around drift output
# must never imply one does. The bias is deliberate -- the mandatory
# message-derived trio (volume, reciprocity, latency) carries the most weight
# since two of its members are the only corroborators that can lift a decline
# past `cooling`; silence is weighted lowest because it can never decide a
# decline on its own; valence/depth may come from distilled annotations and sit
# below the message trio. Phase thresholds stay conservative: for a
# nostalgia-adjacent tool a false "this friendship is dying" is far costlier
# than a missed one. Re-tune freely -- any change re-hashes drift_params and
# forces a rebuild.
_DRIFT_WEIGHTS_DEFAULT = (
    '{"volume": 0.22, "reciprocity": 0.22, "latency": 0.16, '
    '"silence": 0.06, "valence": 0.14, "depth": 0.10, "initiation": 0.10}'
)
_DRIFT_PHASE_THRESHOLDS_DEFAULT = (
    '{"thriving": 0.10, "steady": 0.35, "cooling": 0.60, "dormant_days": 90}'
)


def _parse_json_dict_env(name, default):
    """Parse a JSON-object-valued env var (drift's weights / phase thresholds).
    A malformed value raises a clear ValueError at import rather than silently
    poisoning a downstream aggregate."""
    raw = os.getenv(name, "").strip() or default
    try:
        value = _json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must be valid JSON, got {raw!r}: {e}") from None
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be a JSON object (got {type(value).__name__})"
        )
    return value


DRIFT_WEIGHTS = _parse_json_dict_env("DRIFT_WEIGHTS", _DRIFT_WEIGHTS_DEFAULT)
DRIFT_PHASE_THRESHOLDS = _parse_json_dict_env(
    "DRIFT_PHASE_THRESHOLDS", _DRIFT_PHASE_THRESHOLDS_DEFAULT)
# --- end feature: drift ---
# --- feature: persona ---
# --- Voiceprint (voiceprint.py) ------------------------------------------------
# The card confidence gate: a person_voiceprint row reads 'high' only when it
# clears ALL THREE, else 'low'. Volume, two-sided engagement, and
# breadth-over-time -- a chatty one-week fling shouldn't outrank a three-year tie.
VOICEPRINT_MIN_MESSAGES = int(os.getenv("VOICEPRINT_MIN_MESSAGES") or "500")
VOICEPRINT_MIN_REPLY_PAIRS = int(os.getenv("VOICEPRINT_MIN_REPLY_PAIRS") or "8")
VOICEPRINT_MIN_MONTHS = int(os.getenv("VOICEPRINT_MIN_MONTHS") or "3")
# HARD floor, distinct from the high/low gate above: below this many of the
# member's own messages the stylometry is meaningless, so the pipeline neither
# builds nor returns the row (get_voiceprint declines with a coverage note).
VOICEPRINT_MIN_RETURN_MESSAGES = int(os.getenv("VOICEPRINT_MIN_RETURN_MESSAGES") or "30")
# A DIFFERENT floor again: draft_reply (V2) degrades its per-call style
# retrieval below this many example reply pairs. Separate knob, separate job.
VOICEPRINT_MIN_CONTEXT_PAIRS = int(os.getenv("VOICEPRINT_MIN_CONTEXT_PAIRS") or "4")
# draft_reply post-strips ungrounded inside-joke/nickname tokens. If less than
# this fraction of the draft survives, the remainder is a degenerate stub and
# draft_reply declines instead of returning it.
DRAFT_MIN_SURVIVING_FRACTION = float(os.getenv("DRAFT_MIN_SURVIVING_FRACTION") or "0.5")
# Date-scoped get_voiceprint recomputes over a raw scan: below MIN_RANGE_MESSAGES
# the window is flagged as too thin, above SCAN_MAX_MESSAGES the tool asks the
# caller to narrow the range rather than run an unbounded scan.
VOICEPRINT_MIN_RANGE_MESSAGES = int(os.getenv("VOICEPRINT_MIN_RANGE_MESSAGES") or "200")
VOICEPRINT_SCAN_MAX_MESSAGES = int(os.getenv("VOICEPRINT_SCAN_MAX_MESSAGES") or "20000")
# Max seconds between two consecutive same-sender messages for the second to
# extend the same burst (double-text run) rather than start a new one.
VOICEPRINT_BURST_GAP_SECONDS = int(os.getenv("VOICEPRINT_BURST_GAP_SECONDS") or "120")
# --- end feature: persona ---

# --- feature: conflict ---
# --- Conflict core (conflict_detect.py; RES-46 / C1) ------------------------
# Deterministic tense-stretch finder + eval-gated escalation labels. Empty
# CONFLICT_MODEL inherits CHAT_MODEL; it is recorded as `conflict_model` and
# mixing models in one conflicts table is refused (db.check_model_guard), BEFORE
# --relabel's clear and before detection, so a refused run destroys nothing.
# --relabel is NOT the model-switch door -- clearing `conflicts` is.
#
# CONFLICT_THINK is a plain *_THINK via _think, but conflict_detect._resolve_think
# then reads an UNSET one as False for its own dispatches, since omitting the
# flag costs ~90x per turn. "Unset" means unset at BOTH levels: an explicit
# CHAT_THINK=true is INHERITED here as True and passes straight through, so set
# CONFLICT_THINK=false explicitly if you want thinking for chat.py only.
CONFLICT_MODEL = os.getenv("CONFLICT_MODEL", "") or CHAT_MODEL
CONFLICT_THINK = _think("CONFLICT_THINK", CHAT_THINK)
_conflict_timeout = os.getenv("CONFLICT_TIMEOUT", "").strip()
CONFLICT_TIMEOUT = int(_conflict_timeout) if _conflict_timeout else CHAT_TIMEOUT
# Latency-withdrawal gate (F2 survival estimator over RAW reply gaps, with
# right-censoring): a window qualifies on latency when P(reply gap > Q hours)
# in-window >= FRAC AND that fraction is >= 3x the dyad's trailing-90-day
# baseline. A deterministic ORIGINATING signal.
CONFLICT_LATENCY_Q_HOURS = float(os.getenv("CONFLICT_LATENCY_Q_HOURS") or "9")
CONFLICT_LATENCY_FRAC = float(os.getenv("CONFLICT_LATENCY_FRAC") or "0.5")
# Message-volume collapse: window daily rate <= (1 - X) * the dyad's trailing-
# 8-week median daily rate.
CONFLICT_VOLUME_COLLAPSE = float(os.getenv("CONFLICT_VOLUME_COLLAPSE") or "0.6")
# A gap this large (hours) inside a window sets off_text_flag (the conversation
# likely moved off-channel), which forbids any resolution verdict downstream.
CONFLICT_OFF_TEXT_GAP_HOURS = float(os.getenv("CONFLICT_OFF_TEXT_GAP_HOURS") or "6")
# Minimum drop of window mean valence below the chat all-time mean (both from
# source='llm' rows ONLY) before valence corroboration boosts a window's RANK.
# Gates ranking, never detection -- an all-distilled corpus still detects.
CONFLICT_MIN_VALENCE_DROP = float(os.getenv("CONFLICT_MIN_VALENCE_DROP") or "0.4")
# Floor on the model's self-reported per-turn label confidence (documented
# UNCALIBRATED -- an ordering heuristic, not a probability): below it the label
# row is dropped at write time.
CONFLICT_LABEL_CONF_FLOOR = float(os.getenv("CONFLICT_LABEL_CONF_FLOOR") or "0.7")
# Cosine-similarity threshold consumed by C3 (conflict tracing); declared here
# so the knob lives with its siblings.
CONFLICT_TRACE_SIM_THRESHOLD = float(os.getenv("CONFLICT_TRACE_SIM_THRESHOLD") or "0.65")
# CONFLICT_MIN_SIGNALS: distinct deterministic signals a window must trip to be
# written. CONFLICT_MAX_LABEL_TURNS: per-conflict cap on LLM-labeled turns -- 6,
# not the original 12, because selection is peak-anchored (centered on the
# message left hanging) rather than the first N chronological turns, and this
# corpus contains 1,940-turn windows where a head slice misses the evidence
# entirely. A NON-POSITIVE cap is REFUSED by conflict_detect.main, not honored:
# it selects no turns while every window is still stamped processed, so one .env
# line would latch the whole corpus as labeled having dispatched nothing.
CONFLICT_MIN_SIGNALS = int(os.getenv("CONFLICT_MIN_SIGNALS") or "2")
CONFLICT_MAX_LABEL_TURNS = int(os.getenv("CONFLICT_MAX_LABEL_TURNS") or "6")
# A RANKING preference among label candidates, NOT a filter: with more candidate
# turns than the cap and no peak to bracket, turns at or over this many
# characters are selected first and shorter ones fill what is left. It demotes,
# it never deletes -- as a filter it deleted the class the classifier is asked to
# find (stonewalling IS terse non-answers), and such windows were stamped
# labeled_at having dispatched nothing. Deliberately NOT applied inside the
# peak-anchored span: a terse turn next to the escalation moment IS the evidence.
# The price is that a long turn further from the peak can lose its slot to a
# terse neighbour. The one true prescreen left is code, not a knob: a turn with
# no alphanumeric character at all has nothing for the model to read.
CONFLICT_MIN_TURN_CHARS = int(os.getenv("CONFLICT_MIN_TURN_CHARS") or "15")
# Wall-clock minutes ONE conflict_detect.py run may spend, clocked from the top
# of main() so detection counts against it too. Labeling stops dispatching when
# the remainder cannot cover the next window, commits, and says what it left;
# a rerun continues down the strength ranking (conflicts.labeled_at is the
# resume marker). 25 leaves headroom under the 30-minute target. 0 = NO budget
# (label every detected window). A NEGATIVE value is REFUSED where it is
# resolved, not clamped -- clamping sent it to 0 = unbudgeted, i.e. an env typo
# silently bought the unbounded run the CLI refuses `--budget-minutes -5` for.
CONFLICT_BUDGET_MINUTES = int(os.getenv("CONFLICT_BUDGET_MINUTES") or "25")
# DV-suppression floor for diagnose_conflict (Master Spec A10, SOFTENED,
# code-enforced). 'on' (the default) screens the full text of every message in a
# conflict window for power-and-control markers and, on a hit, suppresses ONLY
# the aggressor/role output (the mechanical scorecard still returns) and appends
# a fixed hotline footer. 'off' restores the role analysis but STILL appends the
# footer. chat.py --no-dv-screen overrides transiently. The hit is never
# persisted. Accepts on/off/true/false/1/0/yes/no.
CONFLICT_DV_SCREEN = os.getenv("CONFLICT_DV_SCREEN", "on").strip().lower() not in (
    "0", "off", "false", "no")
# --- end feature: conflict ---

# --- feature: nudges ---
# --- Dormant-tie nudges (loose_ends.py dormant_tie detector; RES-55 / N2) -----
# A once-close 1:1 tie now quiet for anomalously long. "Once-close" = the top
# NUDGE_DORMANT_TOP_FRAC of eligible dyads by a z-scored closeness composite over
# the trailing 12 months -- z(total messages) + z(mean chunk depth) + z(LSM),
# each z-scored across all eligible dyads (a missing component contributes 0).
# Depth reads chunk_annotations of ANY source (it distills fine). A dyad is
# eligible when it is a 1:1 chat with at least NUDGE_DORMANT_MIN_MESSAGES real
# messages in that window.
NUDGE_DORMANT_TOP_FRAC = float(os.getenv("NUDGE_DORMANT_TOP_FRAC") or "0.2")
NUDGE_DORMANT_MIN_MESSAGES = int(os.getenv("NUDGE_DORMANT_MIN_MESSAGES") or "200")
# "Overdue" = the current silence exceeds BOTH the dyad's own p90 inter-message
# gap (the FULL gap distribution via loose_ends._dyad_gap_p90 -- deliberately
# NOT reply-latency median, which excludes the long silences and is circular)
# AND this absolute floor in days, so a short-cadence pair does not fire on a
# two-week lull.
NUDGE_DORMANT_MIN_GAP_DAYS = int(os.getenv("NUDGE_DORMANT_MIN_GAP_DAYS") or "21")
# --- end feature: nudges ---
# --- feature: wellbeing ---
# --- Wellbeing trajectory (query_tools/analytics_wellbeing.py; RES-61 / W2) --
# The pinned gates for the deterministic change-point core (wellbeing.py), which
# takes each as a keyword argument -- the math module hard-codes none of them.
# Conservative by design: false positives are the enemy, so raising any
# threshold only makes the tool quieter. WELLBEING_MIN_MARKER_HITS is the
# lifetime raw-hit floor applied to a marker BEFORE the core is called.
WELLBEING_Z_THRESHOLD = float(os.getenv("WELLBEING_Z_THRESHOLD") or "3.5")
WELLBEING_MIN_RUN_MONTHS = int(os.getenv("WELLBEING_MIN_RUN_MONTHS") or "3")
WELLBEING_MIN_BASELINE_MONTHS = int(os.getenv("WELLBEING_MIN_BASELINE_MONTHS") or "12")
WELLBEING_FAMILY_CAP = float(os.getenv("WELLBEING_FAMILY_CAP") or "2.0")
WELLBEING_CUSUM_K = float(os.getenv("WELLBEING_CUSUM_K") or "0.5")
WELLBEING_CUSUM_H = float(os.getenv("WELLBEING_CUSUM_H") or "4.0")
WELLBEING_MIN_MARKER_HITS = int(os.getenv("WELLBEING_MIN_MARKER_HITS") or "30")
# Grounding excerpts returned per firing window.
WELLBEING_N_EXAMPLES = int(os.getenv("WELLBEING_N_EXAMPLES") or "3")
# Corpus floor: below this many words in scope the tool returns an
# insufficient-data envelope and runs no analysis.
MIN_WELLBEING_WORDS = int(os.getenv("MIN_WELLBEING_WORDS") or "5000")
# Per-contact DENYlist of `contacts.id` values to SUPPRESS from per-contact
# wellbeing analysis. Default-OPEN: an empty list denies nothing, so
# get_language_trajectory will analyze any named contact. Self is never
# affected. (The Wrapped voiceprint card's consent list is the opposite shape --
# default-closed ALLOWlist. consent.py ships both helpers.)
WELLBEING_DENY_IDS = os.getenv("WELLBEING_DENY_IDS", "")
# --- end feature: wellbeing ---

# --- feature: conflict ---
# --- C3 cross-chat conflict trace (RES-52; query_tools/conflict.py) ----------
# trace_conflict_across_chats is OFF BY DEFAULT and gated at REGISTRATION: its
# schema is withheld from chat.py's tool list unless this is truthy (withholding
# the schema is the repo's one reliable stop). Flipping it on is the user's
# explicit, PERSISTENT opt-in; the tool's per-call `consent` param is a UX
# prompt-gate ONLY, never the security boundary. Written to NO table.
CONFLICT_TRACE_ENABLED = os.getenv("CONFLICT_TRACE_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on")
# Day-buffer padded onto each side of a conflict's window when scanning OTHER
# chats for spillover (converted to vec0 day numbers via db.iso_to_day).
CONFLICT_TRACE_BUFFER_DAYS = int(os.getenv("CONFLICT_TRACE_BUFFER_DAYS") or "2")
# --- end feature: conflict ---
# --- feature: nudges ---
# --- Unanswered-question nudges (loose_ends.py unanswered_question detector; RES-56 / N3) ---
# A 1:1 session that ended on the OTHER person's genuine, unanswered question.
# "Genuine" = ends with '?', trimmed text at least NUDGE_QUESTION_MIN_CHARS long,
# and its final sentence misses the pinned rhetorical/pleasantry stoplist. That
# test is deliberately UNLIKE lexicon_build's bare `"?" in text` (which powers
# person_period_lexicon.question_count and must not change).
NUDGE_QUESTION_MIN_CHARS = int(os.getenv("NUDGE_QUESTION_MIN_CHARS") or "12")
# The MANDATORY cross-session answered-check grace: the deadline is
# anchor + max(the dyad's p90 inter-message gap, this floor in hours), and a
# self-reply inside it suppresses the candidate. The floor keeps a fast-cadence
# pair from firing on a single afternoon's silence.
NUDGE_QUESTION_GRACE_HOURS = float(os.getenv("NUDGE_QUESTION_GRACE_HOURS") or "72")
# A question ANSWERED after the deadline (just late) is resolved, not open --
# calling it "unanswered" tells the reader they never replied when they did.
# Default: drop it. True keeps the old signal, riding evidence.answered_late.
NUDGE_QUESTION_INCLUDE_ANSWERED_LATE = os.getenv(
    "NUDGE_QUESTION_INCLUDE_ANSWERED_LATE", "").strip().lower() in ("1", "true", "yes", "on")
# A question unanswered longer than this is too stale to be actionable.
NUDGE_QUESTION_MAX_AGE_DAYS = int(os.getenv("NUDGE_QUESTION_MAX_AGE_DAYS") or "365")
# --- end feature: nudges ---
# --- feature: nudges ---
# --- N4 nudges: dropped_plan + unreciprocated_goodnews (loose_ends.py; RES-57) ---
# Both detectors ship HARD-GATED (default OFF). The flag-off path never runs an
# LLM and never consults the eval gate.
#
# dropped_plan: an FTS cue-phrase pre-filter finds a message proposing a plan or
# making a commitment, then ollama.chat_once_json asks Searle-style questions
# over the anchor + capped following context and returns
# fulfilled | unfulfilled | UNKNOWN. UNKNOWN => SUPPRESS. Even when ON, a row is
# written only if the F4 precision gate clears the dropped_plan class, checked at
# write time -- and an absent verdict suppresses (it ships no seed).
NUDGE_DROPPED_PLAN_ENABLED = os.getenv(
    "NUDGE_DROPPED_PLAN_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# How many messages following the anchor are shown to the verifier as context.
NUDGE_PLAN_CONTEXT_MSGS = int(os.getenv("NUDGE_PLAN_CONTEXT_MSGS") or "30")
# unreciprocated_goodnews: reads response_class out of chunk_annotations.
# events_json (a JSON field, NOT a column) for source='llm' rows ONLY, and fires
# when a good-news moment drew anything other than an active-constructive
# response. Coverage is reported in evidence_json so "no good news" is never
# confused with "not annotated". Its precision is inherited from the annotations,
# not an SQL-exact signal, and the tool payload says so.
NUDGE_GOODNEWS_ENABLED = os.getenv(
    "NUDGE_GOODNEWS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# Recency floor: a good-news moment older than this is not actionable, and its
# urgency decays to ~0 anyway, so it is never even written.
NUDGE_GOODNEWS_MAX_AGE_DAYS = int(os.getenv("NUDGE_GOODNEWS_MAX_AGE_DAYS") or "90")
# --- end feature: nudges ---


# --- Inferred-occasion nudges (loose_ends.py inferred_occasion detector; RES-58 / N5) ---
# A recurring annual occasion a 1:1 dyad marks -- a birthday or anniversary --
# inferred from CONGRATULATORY messages landing on the SAME CALENDAR DATE across
# years. The (month, day) must recur in at least this many DISTINCT years, which
# a one-off "congrats" can never satisfy, so the phrase list stays permissive.
# This is the one detector that fills loose_threads.due_by (the next upcoming
# occurrence), so the digest can surface the reminder BEFORE the date.
NUDGE_OCCASION_MIN_YEARS = int(os.getenv("NUDGE_OCCASION_MIN_YEARS") or "2")

# --- feature: persona (V5 simulate_contact; RES-69) --------------------------
# The highest-risk persona leg: ESTIMATE how a contact might reply in their
# voice, or talk to your PAST SELF as of a date. OFF BY DEFAULT and gated at
# REGISTRATION -- the schema is withheld from chat.py's tool list unless truthy.
# The tool's per-call `acknowledged` param is a UX flow device, NOT the security
# boundary. Nothing auto-sends and nothing is persisted.
PERSONA_SIM_ENABLED = os.getenv("PERSONA_SIM_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on")
# Hard deceased/grief denylist the tool refuses in CODE, with a fixed
# compassionate message and no generation. DEFAULT-ALLOW (owner considered
# default-deny and declined for v1); there is NO reliable automatic deceased
# detection from chat.db, so this manual list is the FLOOR. Entries are
# member_key-style: a bare int is a `contacts.id` (back-compat), and explicit
# `contact:N` / `handle:N` keys are accepted too, so a person who exists only as
# an unlinked handle can be denylisted.
PERSONA_SIM_BLOCK_IDS = os.getenv("PERSONA_SIM_BLOCK_IDS", "")
# Soft deceased/dormant heuristic: a FAIL-OPEN net OVER the manual list, not a
# substitute. Before simulating a living contact the tool checks last-message age
# AND a strongly one-sided (texting-into-silence) pattern; when BOTH fire it
# WARNS and asks for confirmation. It never blocks -- the heuristic cannot tell
# deceased from long-lost. This is the age half.
PERSONA_SIM_DORMANT_DAYS = int(os.getenv("PERSONA_SIM_DORMANT_DAYS") or "365")
# ...and the user must have sent at least this many messages AFTER that last
# inbound with no reply since for the one-sided half to fire.
PERSONA_SIM_ONESIDED_MIN_MSGS = int(os.getenv("PERSONA_SIM_ONESIDED_MIN_MSGS") or "3")
# --- end feature: persona (V5 simulate_contact) ---
