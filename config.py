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

# --- Apple Foundation Models backend (opt-in) --------------------------------
# On-device backend via Apple Intelligence (Apple Silicon, macOS 26+), routed
# through a local Swift shim (`make apple-shim`). Opt in by setting any of the
# *_MODEL vars -- CHAT_MODEL, ANNOTATE_MODEL, TOPIC_MODEL, WRAPPED_MODEL,
# EMBED_MODEL -- to the "apple" sentinel; the defaults above stay gemma4/nomic,
# so apple is never on unless asked for. The shim's context window is a fixed
# 4,096 tokens, so the apple chat path stages many small sessions and digests
# oversized tool results rather than context-stuffing (see chat_apple.py and
# docs/DECISIONS.md).


def is_apple_model(name) -> bool:
    """True when a *_MODEL value selects the Apple Foundation Models backend:
    the bare "apple" sentinel, or an "apple:<variant>" form reserved for future
    model variants. Every dispatch seam (ollama.py, embeddings.py, chat.py)
    gates on this so all other callers stay backend-agnostic."""
    return isinstance(name, str) and (name == "apple" or name.startswith("apple:"))


# The compiled Swift shim, built by `make apple-shim`; resolved relative to
# this file's directory like the DB paths, so it is independent of the CWD.
APPLE_SHIM_PATH = _resolve(os.getenv("APPLE_SHIM_PATH", "bin/applefm-shim"))
# Per-tool-result char cap for the apple chat orchestrator. Far below
# TOOL_RESULT_MAX_CHARS (sized for gemma4's 32k window): the FM window is a
# fixed 4,096 tokens, so a result over this is digested down to
# question-relevant facts, never truncated blind.
APPLE_TOOL_RESULT_MAX_CHARS = int(os.getenv("APPLE_TOOL_RESULT_MAX_CHARS", "5000"))

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
# --- feature: persona ---
# Third-party "Your Voiceprint" game consent (RES-64 / V4): a comma/space-
# separated list of `contacts.id` values the user marked "OK to include" in the
# Wrapped voiceprint section's guess-who game. Empty (the default) = self-only:
# NO third-party verbatim text or stylometric card appears in the generated file
# (default redaction). Parsed via consent.parse_consent_ids (tolerant; non-
# positive/non-integer tokens skipped). Consent ADDS a contact; it never
# overrides a roster exclusion (silent-contact guard / --contact / --exclude).
WRAPPED_VOICEPRINT_CONSENT_IDS = os.getenv("WRAPPED_VOICEPRINT_CONSENT_IDS", "")
# --- end feature: persona ---
# --- feature: wellbeing ---
# Opt-in Wrapped wellbeing trajectory section (RES-68 / W4). OFF by default:
# the section is built only when this is truthy OR wrapped.py's --wellbeing flag
# is passed (the flag overrides this config for that run). When a flagged window
# renders, the section carries the same code-emitted, direction-independent
# support footer as the chat surface (wellbeing.support_block), and every piece
# of its narration is run through wellbeing.sanitize_condition_names first.
WRAPPED_WELLBEING = os.getenv("WRAPPED_WELLBEING", "").strip().lower() in (
    "1", "true", "yes", "on")
# --- end feature: wellbeing ---

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

# --- Eval gates (eval/evalkit label/gate; RES-70 / F4) ----------------------
# Minimum hand labels a measurement gate needs before it may mark a class
# passed. Per-class support floor for kappa tasks (conflict_labels): a class
# with fewer than this many gold labels scores None rather than a kappa from
# too few examples, and can never be cleared. Pinned at 50.
EVAL_GATE_MIN_LABELS = int(os.getenv("EVAL_GATE_MIN_LABELS") or "50")
# The analogous floor for precision (detector) tasks (dropped_plan): a positive
# class needs at least this many gold positives in the labeled sample before
# its precision may clear the gate. Pinned at 40 (precision stabilizes with
# fewer examples than a full kappa needs).
EVAL_GATE_MIN_PRECISION_POSITIVES = int(os.getenv("EVAL_GATE_MIN_PRECISION_POSITIVES") or "40")
# --- feature: nudges ---
# --- Loose Ends / nudges (loose_ends.py) --------------------------------------
# Standalone loose-thread detector. NUDGES_MODEL/THINK/TIMEOUT follow the usual
# inherit-from-CHAT rule and are used ONLY for the one-line nudge_text
# narration -- the detectors themselves are model-free SQL. Empty NUDGES_MODEL =
# CHAT_MODEL; NUDGES_THINK inherits CHAT_THINK (tri-state: unset omits the flag,
# which some models require); NUDGES_TIMEOUT inherits CHAT_TIMEOUT.
NUDGES_MODEL = os.getenv("NUDGES_MODEL", "") or CHAT_MODEL
NUDGES_THINK = _think("NUDGES_THINK", CHAT_THINK)
_nudges_timeout = os.getenv("NUDGES_TIMEOUT", "").strip()
NUDGES_TIMEOUT = int(_nudges_timeout) if _nudges_timeout else CHAT_TIMEOUT
# Stage-0 auto-exclusion: a dyad whose most-recent llm-annotated exchange has
# mean valence <= this is dropped (a negative sign-off is not a nudge). Only
# chunk_annotations rows with source='llm' count; no llm rows = no valence
# exclusion on that dyad.
NUDGE_VALENCE_EXCLUDE = float(os.getenv("NUDGE_VALENCE_EXCLUDE") or "-0.5")
# Stage-0 identity exclusions. NUDGE_OWNER_CONTACT_IDS / NUDGE_OWNER_HANDLES are
# the account owner's OWN contact-card ids and handle identifiers (comma/space
# separated) -- the schema stores no owner identity, so a self-chat (your own
# second number, saved under your own name) would otherwise surface as a nudge.
# List your own contact_id / number(s) here once to suppress it. Both default
# empty (no owner suppression). NUDGE_SHORTCODE_MAX_DIGITS is the length ceiling
# for treating an all-digit handle as an automated SMS shortcode (e.g. 22022);
# real 10-15 digit phone numbers are unaffected.
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
# Relationship-drift pipeline (drift.py). DRIFT_ENABLED gates the whole
# pipeline end-to-end: false (the default) makes the `make pipeline` drift step
# a no-op -- drift.py exits immediately and the dyad_drift table is never built
# out of the box. DRIFT_WEIGHTS and DRIFT_PHASE_THRESHOLDS are this repo's first
# JSON-valued env vars: each is parsed once, here, with a clear ValueError on
# malformed input (a bad .env must fail loudly at import, not silently later).
import json as _json  # noqa: E402  (local to the drift block; keeps the edit additive)

DRIFT_ENABLED = os.getenv("DRIFT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# A dyad needs at least this many total messages, distinct active months, and
# (per signal) valid monthly data points to be scored; robust_slope itself
# needs >= 4 points, so DRIFT_MIN_VALID_MONTHS below 4 makes every trend absent.
DRIFT_MIN_MESSAGES = int(os.getenv("DRIFT_MIN_MESSAGES") or "100")
DRIFT_MIN_MONTHS = int(os.getenv("DRIFT_MIN_MONTHS") or "4")
DRIFT_MIN_VALID_MONTHS = int(os.getenv("DRIFT_MIN_VALID_MONTHS") or "4")
# A dyad needs this many gap-split sessions before the initiation-asymmetry
# signal is computed at all (too few sessions makes the share meaningless).
DRIFT_MIN_SESSIONS = int(os.getenv("DRIFT_MIN_SESSIONS") or "5")
# Confidence level for every trend's bootstrap CI (robust_slope alpha): a CI
# that still spans 0 at this level contributes 0 to the composite.
DRIFT_SLOPE_CI = float(os.getenv("DRIFT_SLOPE_CI") or "0.05")

# The two structured vars. .env.example shows the full default JSON inline.
#
# HEURISTIC, NOT VALIDATED. These weights and phase cutoffs are a hand-tuned
# starting point, not an empirically calibrated model: no study establishes
# that this particular composite tracks real relationship "drift", and copy
# built around drift output must never imply one does. They are deliberately
# biased toward the signals we trust most:
#   * the mandatory message-derived trio (volume, reciprocity, latency) carries
#     the most weight -- it is always available, and two of its members
#     (reciprocity, latency) are the ONLY corroborators that can lift a decline
#     label past `cooling`, so under-weighting them would starve the phase floor;
#   * silence is weighted lowest -- it is heavy-tailed and, by design, can never
#     decide a decline on its own (the corroboration floor caps silence-only at
#     cooling), so a large weight would only inflate scores it cannot act on;
#   * valence/depth come from chunk_annotations that may be distilled (biased,
#     and already down-weighted upstream), so they sit below the message trio.
# Phase thresholds stay intentionally conservative (a high bar to call a tie
# "fading"): for a nostalgia-adjacent tool a false "this friendship is dying"
# is far costlier than a missed one. Re-tune freely -- any change here re-hashes
# drift_params and forces a rebuild on the next run.
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
# clears ALL THREE of these, else 'low'. Volume, two-sided engagement, and
# breadth-over-time -- a chatty one-week fling shouldn't outrank a three-year
# tie.
VOICEPRINT_MIN_MESSAGES = int(os.getenv("VOICEPRINT_MIN_MESSAGES") or "500")
VOICEPRINT_MIN_REPLY_PAIRS = int(os.getenv("VOICEPRINT_MIN_REPLY_PAIRS") or "8")
VOICEPRINT_MIN_MONTHS = int(os.getenv("VOICEPRINT_MIN_MONTHS") or "3")
# HARD floor (distinct from the high/low gate above): below this many of the
# member's own messages a profile is statistically meaningless, so the pipeline
# neither builds nor returns it. get_voiceprint declines with a coverage note
# (like draft_reply's structural decline), and the build skips writing the row.
# 30 keeps the huge self/dyad profiles while dropping the long tail of 1-4
# message "voiceprints" that are pure noise.
VOICEPRINT_MIN_RETURN_MESSAGES = int(os.getenv("VOICEPRINT_MIN_RETURN_MESSAGES") or "30")
# A DIFFERENT floor from the gate above: V2 (draft_reply) degrades its
# per-call style retrieval below this many example reply pairs. Separate knob,
# separate job -- kept distinct on purpose.
VOICEPRINT_MIN_CONTEXT_PAIRS = int(os.getenv("VOICEPRINT_MIN_CONTEXT_PAIRS") or "4")
# draft_reply post-strips ungrounded inside-joke/nickname tokens. If less than
# this fraction of the generated draft survives the strip, the remainder is a
# degenerate stub (e.g. "It'") -- draft_reply declines instead of sending it.
DRAFT_MIN_SURVIVING_FRACTION = float(os.getenv("DRAFT_MIN_SURVIVING_FRACTION") or "0.5")
# Date-scoped get_voiceprint recomputes over a raw message scan. Below
# MIN_RANGE_MESSAGES the window is too thin to trust (flagged); above
# SCAN_MAX_MESSAGES the tool asks the caller to narrow the range rather than
# run an unbounded scan.
VOICEPRINT_MIN_RANGE_MESSAGES = int(os.getenv("VOICEPRINT_MIN_RANGE_MESSAGES") or "200")
VOICEPRINT_SCAN_MAX_MESSAGES = int(os.getenv("VOICEPRINT_SCAN_MAX_MESSAGES") or "20000")
# Max seconds between two consecutive same-sender messages for the second to
# extend the same burst (double-text run) rather than start a new one.
VOICEPRINT_BURST_GAP_SECONDS = int(os.getenv("VOICEPRINT_BURST_GAP_SECONDS") or "120")
# --- end feature: persona ---

# --- feature: conflict ---
# --- Conflict core (conflict_detect.py; RES-46 / C1) ------------------------
# Deterministic tense-stretch finder + eval-gated escalation labels. Per-
# pipeline model knobs follow the STANDARD inherit-from-CHAT rule (CONFLICT_
# THINK is a plain *_THINK via _think, NOT annotate's empty-means-preset
# carve-out). Empty CONFLICT_MODEL inherits CHAT_MODEL; recorded in
# ingest_state as `conflict_model` and mixing models in one conflicts table is
# refused (same guard shape as chunk_annotations' annotate_model).
CONFLICT_MODEL = os.getenv("CONFLICT_MODEL", "") or CHAT_MODEL
CONFLICT_THINK = _think("CONFLICT_THINK", CHAT_THINK)
_conflict_timeout = os.getenv("CONFLICT_TIMEOUT", "").strip()
CONFLICT_TIMEOUT = int(_conflict_timeout) if _conflict_timeout else CHAT_TIMEOUT
# Latency-withdrawal gate (F2 survival estimator over RAW reply gaps, with
# right-censoring): a window qualifies on latency when P(reply gap > Q hours)
# in-window >= FRAC AND that fraction is >= 3x the dyad's trailing-90-day
# baseline fraction. This is a deterministic ORIGINATING signal.
CONFLICT_LATENCY_Q_HOURS = float(os.getenv("CONFLICT_LATENCY_Q_HOURS") or "9")
CONFLICT_LATENCY_FRAC = float(os.getenv("CONFLICT_LATENCY_FRAC") or "0.5")
# Message-volume collapse: window daily rate <= (1 - X) * the dyad's trailing-
# 8-week median daily rate.
CONFLICT_VOLUME_COLLAPSE = float(os.getenv("CONFLICT_VOLUME_COLLAPSE") or "0.6")
# A gap this large (hours) inside a conflict window sets off_text_flag (the
# conversation likely moved off-channel -- a call, in person): it forbids any
# resolution verdict downstream (C4).
CONFLICT_OFF_TEXT_GAP_HOURS = float(os.getenv("CONFLICT_OFF_TEXT_GAP_HOURS") or "6")
# Minimum drop of window mean valence below the chat all-time mean (both from
# chunk_annotations.source='llm' ONLY) before valence corroboration boosts a
# window's RANK. Gates ranking, never detection -- an all-distilled corpus
# still detects (corroboration='none'), it just gets no valence boost.
CONFLICT_MIN_VALENCE_DROP = float(os.getenv("CONFLICT_MIN_VALENCE_DROP") or "0.4")
# Floor applied to the model's self-reported per-turn label confidence
# (documented UNCALIBRATED -- an ordering heuristic, not a probability): below
# it the label row is dropped at write time.
CONFLICT_LABEL_CONF_FLOOR = float(os.getenv("CONFLICT_LABEL_CONF_FLOOR") or "0.7")
# Cosine-similarity threshold consumed later by C3 (conflict tracing) -- just
# declared here so the knob lives with its siblings.
CONFLICT_TRACE_SIM_THRESHOLD = float(os.getenv("CONFLICT_TRACE_SIM_THRESHOLD") or "0.65")
# Internal detection thresholds (beyond the pinned starter set above).
# CONFLICT_MIN_SIGNALS: distinct deterministic signals a window must trip to
# be written (latency_withdrawal / one_sided_session / dislike_run /
# volume_collapse). CONFLICT_MAX_LABEL_TURNS: per-conflict cap on LLM-labeled
# turns (cost + prompt-size safety).
CONFLICT_MIN_SIGNALS = int(os.getenv("CONFLICT_MIN_SIGNALS") or "2")
CONFLICT_MAX_LABEL_TURNS = int(os.getenv("CONFLICT_MAX_LABEL_TURNS") or "12")
# DV-suppression floor for diagnose_conflict (C2 / Master Spec A10, SOFTENED,
# code-enforced). 'on' (the default) screens the FULL text of every message in
# a conflict window for high-risk power-and-control markers and, on a hit,
# suppresses ONLY the aggressor/role-assignment output (the mechanical
# scorecard still returns) and appends a fixed hotline resource footer. 'off'
# restores the role analysis but STILL appends the footer on a hit. A per-run
# chat.py --no-dv-screen flag overrides this to 'off' transiently. The hit is
# never persisted anywhere. Accepts on/off/true/false/1/0/yes/no.
CONFLICT_DV_SCREEN = os.getenv("CONFLICT_DV_SCREEN", "on").strip().lower() not in (
    "0", "off", "false", "no")
# --- end feature: conflict ---

# --- feature: nudges ---
# --- Dormant-tie nudges (loose_ends.py dormant_tie detector; RES-55 / N2) -----
# A once-close 1:1 tie that has now gone quiet for anomalously long. "Once-
# close" = the top NUDGE_DORMANT_TOP_FRAC of eligible dyads by a z-scored
# closeness composite over the trailing 12 months:
#   closeness = z(total messages) + z(mean chunk depth) + z(LSM),
# each term z-scored across ALL eligible dyads (a dyad missing a component --
# no annotated chunk, or one-sided lexicon -- contributes 0 for that term).
# Depth is read from chunk_annotations of ANY source (it distills fine); LSM
# comes from person_period_lexicon via query_tools' public lsm_score. A dyad is
# eligible when it is a direct (1:1) chat with at least NUDGE_DORMANT_MIN_MESSAGES
# real messages in that trailing window.
NUDGE_DORMANT_TOP_FRAC = float(os.getenv("NUDGE_DORMANT_TOP_FRAC") or "0.2")
NUDGE_DORMANT_MIN_MESSAGES = int(os.getenv("NUDGE_DORMANT_MIN_MESSAGES") or "200")
# "Overdue" = the current silence (corpus now minus the dyad's last message)
# exceeds BOTH the dyad's own p90 inter-message gap (the FULL gap distribution
# via loose_ends._dyad_gap_p90 -- deliberately NOT reply-latency median, which
# excludes the long silences and is circular) AND this absolute floor in days
# (so a naturally short-cadence pair does not fire on a two-week lull even when
# its p90 is tiny).
NUDGE_DORMANT_MIN_GAP_DAYS = int(os.getenv("NUDGE_DORMANT_MIN_GAP_DAYS") or "21")
# --- end feature: nudges ---
# --- feature: wellbeing ---
# --- Wellbeing trajectory (query_tools/analytics_wellbeing.py; RES-61 / W2) --
# The pinned gates for the deterministic change-point core (wellbeing.py, W1),
# which takes each as a keyword argument. W2 owns these knobs and wires them
# through; the math module hard-codes none of them. Conservative by design --
# false positives are the enemy, so raising any threshold only makes the tool
# quieter. WELLBEING_MIN_MARKER_HITS mirrors wellbeing.WELLBEING_MIN_MARKER_HITS
# (the lifetime raw-hit floor W2 applies to a marker BEFORE calling W1).
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
# Per-contact DENYlist: a comma/space-separated list of `contacts.id` values to
# SUPPRESS from per-contact wellbeing analysis. Default-open -- an empty list
# (the default) denies nothing, so get_language_trajectory will analyze any
# named contact; add an id here to refuse that specific contact in code. Self is
# never affected. Parsed via consent.parse_consent_ids (tolerant; non-positive/
# non-integer tokens skipped).
WELLBEING_DENY_IDS = os.getenv("WELLBEING_DENY_IDS", "")
# --- end feature: wellbeing ---

# --- feature: conflict ---
# --- C3 cross-chat conflict trace (RES-52; query_tools/conflict.py) ----------
# trace_conflict_across_chats is OFF BY DEFAULT and gated at REGISTRATION: its
# schema is withheld from chat.py's tool list unless CONFLICT_TRACE_ENABLED is
# truthy (withholding the schema is the repo's one reliable stop -- see the
# gotchas). Flipping this on is the user's explicit, PERSISTENT opt-in; the
# tool's per-call `consent` param is a UX prompt-gate ONLY, never the security
# boundary. The trace is computed on demand and written to NO table.
# Separate appended fence (kept out of C1's block above) so C3 stays additive.
CONFLICT_TRACE_ENABLED = os.getenv("CONFLICT_TRACE_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on")
# Day-buffer padded onto each side of a conflict's [first_date, last_date]
# window when scanning OTHER chats for spillover (converted to vec0 day numbers
# via db.iso_to_day -- the same converter the metadata writer uses).
CONFLICT_TRACE_BUFFER_DAYS = int(os.getenv("CONFLICT_TRACE_BUFFER_DAYS") or "2")
# --- end feature: conflict ---
# --- feature: nudges ---
# --- Unanswered-question nudges (loose_ends.py unanswered_question detector; RES-56 / N3) ---
# A 1:1 session that ended on the OTHER person's genuine, unanswered question.
# "Genuine" = the last message ends with '?', its trimmed text is at least
# NUDGE_QUESTION_MIN_CHARS long, and its final sentence misses the pinned
# rhetorical/pleasantry stoplist (loose_ends._RHETORICAL_STOPLIST): a short
# "u up?" or a bare "right?" is a pleasantry, not a loose end. This length/
# stoplist test is deliberately UNLIKE lexicon_build's bare `"?" in text`
# (which powers person_period_lexicon.question_count and must not change).
NUDGE_QUESTION_MIN_CHARS = int(os.getenv("NUDGE_QUESTION_MIN_CHARS") or "12")
# The MANDATORY cross-session answered-check grace: the question is a loose end
# only if it went unanswered for longer than the dyad's own cadence. The
# deadline is anchor + max(the dyad's p90 inter-message gap via
# loose_ends._dyad_gap_p90, this floor in hours); a self-reply inside the
# deadline suppresses the candidate. The 72h floor keeps a naturally
# fast-cadence pair from firing on a single afternoon's silence.
NUDGE_QUESTION_GRACE_HOURS = float(os.getenv("NUDGE_QUESTION_GRACE_HOURS") or "72")
# A question ANSWERED after the deadline (just late) is resolved, not an open
# loose end -- surfacing it as "unanswered" misleads (the reader is told they
# never replied when they did). Default: drop it. Set true to keep the old
# "you replied slowly" signal, which then rides evidence.answered_late.
NUDGE_QUESTION_INCLUDE_ANSWERED_LATE = os.getenv(
    "NUDGE_QUESTION_INCLUDE_ANSWERED_LATE", "").strip().lower() in ("1", "true", "yes", "on")
# A question that has gone unanswered longer than this is too stale to be an
# actionable loose end (mirrors NUDGE_GOODNEWS_MAX_AGE_DAYS). Default 365 days.
NUDGE_QUESTION_MAX_AGE_DAYS = int(os.getenv("NUDGE_QUESTION_MAX_AGE_DAYS") or "365")
# --- end feature: nudges ---
# --- feature: nudges ---
# --- N4 nudges: dropped_plan + unreciprocated_goodnews (loose_ends.py; RES-57) ---
# Both detectors ship HARD-GATED (default OFF). Their flag-off path never runs
# an LLM and never consults the eval gate -- turning them off is a pure config
# read, independent of whether the owner has run the F4 measurement gate.
#
# dropped_plan (a.k.a. unmet_commitment): an FTS cue-phrase pre-filter finds a
# message that proposes a plan or makes a commitment, then ollama.chat_once_json
# asks Searle-style questions over the anchor + capped following context and
# returns fulfilled | unfulfilled | UNKNOWN. UNKNOWN => SUPPRESS. Off by default;
# even when ON, a row is written ONLY if the F4 precision gate clears the
# dropped_plan class (>= 0.85) -- eval_gate_passed(con, 'dropped_plan',
# 'dropped_plan') is checked at write time, and an absent verdict suppresses
# (dropped_plan ships no seed). The verification call passes think=NUDGES_THINK
# via ollama's tri-state; with gemma the effective value must be false -- a
# one-word verdict does not warrant a reasoning pass (~11x latency for the same
# answer), and NUDGES_THINK is only ever True/False/None (config._think), never
# an effort-level string, so no such string is ever inherited into gemma.
NUDGE_DROPPED_PLAN_ENABLED = os.getenv(
    "NUDGE_DROPPED_PLAN_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# How many messages following the anchor are shown to the verifier as context.
NUDGE_PLAN_CONTEXT_MSGS = int(os.getenv("NUDGE_PLAN_CONTEXT_MSGS") or "30")
# unreciprocated_goodnews: reads response_class out of chunk_annotations.
# events_json (a JSON field, NOT a column) for source='llm' rows ONLY, and fires
# when a good-news moment drew anything other than an active-constructive
# response. Coverage (the fraction of the dyad's chunks that carry an llm-source
# annotation) is reported in evidence_json so "no good news" is never confused
# with "not annotated". Opt-in (default OFF); its precision is inherited from
# the gemma annotations, not an SQL-exact signal, and the tool payload says so.
NUDGE_GOODNEWS_ENABLED = os.getenv(
    "NUDGE_GOODNEWS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
# Recency floor: a good-news moment older than this many days is not actionable
# (nudging someone to belatedly celebrate months-old news is noise), and its
# urgency decays to ~0 anyway, so it is never even written. Keeps the digest
# table clean rather than filled with unreachable stale rows.
NUDGE_GOODNEWS_MAX_AGE_DAYS = int(os.getenv("NUDGE_GOODNEWS_MAX_AGE_DAYS") or "90")
# --- end feature: nudges ---


# --- Inferred-occasion nudges (loose_ends.py inferred_occasion detector; RES-58 / N5) ---
# A recurring annual occasion a 1:1 dyad marks -- a birthday or anniversary --
# inferred from CONGRATULATORY messages that land on the SAME CALENDAR DATE
# across years. The (month, day) must recur in at least this many DISTINCT years
# before it counts as an occasion (a one-off "congrats" can never satisfy this,
# so the phrase list stays permissive). This is the one detector that fills the
# loose_threads.due_by column: the next upcoming occurrence, so the digest can
# surface the reminder BEFORE the date. The month-day/year are sliced off the
# ISO date string in Python -- no SQL date function ever wraps the date column
# in a WHERE clause (the load-bearing date-handling rule).
NUDGE_OCCASION_MIN_YEARS = int(os.getenv("NUDGE_OCCASION_MIN_YEARS") or "2")

# --- feature: persona (V5 simulate_contact; RES-69) --------------------------
# simulate_contact is the highest-risk persona leg: ESTIMATE how a contact might
# reply in their voice, or talk to your PAST SELF as of a date. It is OFF BY
# DEFAULT and gated at REGISTRATION -- PERSONA_SIM_ENABLED withholds the schema
# from chat.py's tool list unless truthy (withholding the schema is the repo's
# one reliable stop -- see the gotchas). Flipping it on is the user's explicit,
# PERSISTENT opt-in; the tool's per-call `acknowledged` param is a UX flow
# device, NOT the security boundary. Nothing here auto-sends and nothing is
# persisted. Separate appended fence (kept out of the V1-V3 persona block above)
# so V5 stays a clean additive edit.
PERSONA_SIM_ENABLED = os.getenv("PERSONA_SIM_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on")
# Hard deceased/grief denylist: a comma/space-separated list the user marks
# do-not-simulate. The tool hard-refuses these in CODE with a fixed
# compassionate message and never generates. DEFAULT-ALLOW (a person absent from
# the list is simulable -- owner considered default-deny and declined for v1).
# There is NO reliable automatic deceased detection from chat.db; this manual
# list is the FLOOR. Entries are `member_key`-style: a bare int is a
# `contacts.id` (back-compat), and explicit `contact:N` / `handle:N` keys are
# also accepted -- so a person who exists only as an unlinked handle (no
# contacts row) can be denylisted too. Parsed via consent.parse_member_block_ids
# (tolerant; junk/non-positive tokens skipped); membership here means "blocked".
PERSONA_SIM_BLOCK_IDS = os.getenv("PERSONA_SIM_BLOCK_IDS", "")
# Soft deceased/dormant heuristic (owner revision 2026-07-20): a FAIL-OPEN safety
# net OVER the manual block list, not a substitute for it. Before simulating a
# living contact the tool checks last-message age + a strongly one-sided
# (texting-into-silence) pattern; when BOTH fire it WARNS and asks for
# confirmation. It never blocks -- the heuristic cannot tell deceased from
# long-lost. A contact's most-recent message TO the user older than this many
# days is the age half of the signal.
PERSONA_SIM_DORMANT_DAYS = int(os.getenv("PERSONA_SIM_DORMANT_DAYS") or "365")
# ...and the user must have sent at least this many messages AFTER that last
# inbound message with no reply since (texting into silence) for the one-sided
# half to fire. Both halves are required; the result is a warning, not a block.
PERSONA_SIM_ONESIDED_MIN_MSGS = int(os.getenv("PERSONA_SIM_ONESIDED_MIN_MSGS") or "3")
# --- end feature: persona (V5 simulate_contact) ---
