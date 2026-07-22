"""
Query layer over imessage_analysis.db, meant to be exposed as LLM tool calls
(Claude tool_use / function-calling) rather than pre-loading data into context.

Retrieval modes:
  - search_messages_sql     structured filters (chat, sender, date range)
  - search_messages_fts     keyword / phrase search via FTS5
  - search_conversations    semantic search over conversation chunks
                            (sqlite-vec + Ollama embeddings)
  - search_messages_hybrid  keyword + semantic together, fused with
                            reciprocal-rank fusion

plus expansion helpers: get_conversation (a chunk's messages),
get_message_context (messages around any hit), find_similar_conversations
(more-like-this from a chunk's stored vector), get_full_messages.

Each returns a small list of dicts, meant to be fed back to the LLM as tool
results -- never the whole table.

Layout: messages.py (message-level tools), chunks.py (chunk/semantic/hybrid
tools), directory.py (contact/chat resolution + stats), analytics.py
(participant stats, word frequency), analytics_time.py (activity patterns,
response dynamics), analytics_social.py (reaction stats, milestones),
analytics_language.py (word trends), analytics_group.py (group dynamics),
_shared.py (connection, caps, scope/date helpers). Every
tool is re-exported here, so `import query_tools` stays the only import
callers need -- add new tools to the right submodule and to __all__ below.
DB_PATH and embed_texts are module attributes by contract: tests patch them
on this package, and the internals resolve both through it at call time.
"""

from embeddings import embed_texts

from ._shared import (
    DB_PATH,
    _date_bound_step,
    _normalize_date_bound,
    _restrict_to_direct_chats,
)
from .analytics import (
    get_chat_participant_stats,
    get_word_frequency,
)
from .analytics_group import get_group_dynamics
from .analytics_language import get_word_trends
from .analytics_social import (
    get_milestones,
    get_reaction_stats,
)
from .analytics_time import (
    get_activity_patterns,
    get_response_dynamics,
)
from .chunks import (
    find_similar_conversations,
    get_conversation,
    search_conversations,
    search_messages_hybrid,
)
# --- feature: conflict ---
from .conflict import (
    diagnose_conflict,
    find_conflicts,
    get_conflict,
    set_dv_screen_bypass,
)
# --- end feature: conflict ---
from .directory import (
    get_chat_summary_stats,
    list_chats,
    search_contact_by_handle,
    search_contacts_by_name,
)
from .messages import (
    get_full_messages,
    get_message_context,
    get_message_reactions,
    search_messages_fts,
    search_messages_sql,
)
# --- feature: persona ---
from .persona import get_voiceprint
from .persona_draft import (
    check_style_match,
    draft_reply,
)
# --- end feature: persona ---

# --- feature: nudges ---
from .nudges import (
    get_loose_threads,
    mute_contact_from_nudges,
    resolve_loose_thread,
)
# --- end feature: nudges ---

# --- feature: wellbeing ---
from .analytics_wellbeing import get_language_trajectory
# --- end feature: wellbeing ---
# --- feature: drift ---
# Re-exported UNCONDITIONALLY (the functions always exist); DRIFT_ENABLED gates
# only whether their SCHEMAS are registered in tool_schemas/chat, which is the
# repo's one reliable way to keep the model from calling a tool.
from .drift import (
    get_fadeout_watchlist,
    get_relationship_drift,
)
# --- end feature: drift ---
# --- feature: drift ---
# D3 (RES-65): re-exported UNCONDITIONALLY like the D2 tools -- DRIFT_ENABLED
# gates only its schema registration, not its importability.
from .drift import get_reconnection_candidates
# --- end feature: drift ---

# --- feature: conflict (C3 trace) ---
# Re-exported UNCONDITIONALLY (the function always exists); CONFLICT_TRACE_ENABLED
# gates only whether its SCHEMA is registered (tool_schemas/chat), the repo's
# one reliable way to keep the model from calling it. Separate appended fence so
# C3 stays a clean additive edit alongside C1/C2's conflict re-exports above.
from .conflict import trace_conflict_across_chats
# --- end feature: conflict (C3 trace) ---

# --- feature: conflict (C4 apology) ---
# draft_apology is unconditional (no config gate; the DV floor is enforced in
# the function). Separate appended fence so C4 stays a clean additive edit
# alongside the concurrent conflict worktrees.
from .conflict_apology import draft_apology
# --- end feature: conflict (C4 apology) ---

# --- feature: persona (V5 simulate) ---
# Re-exported UNCONDITIONALLY (the function always exists); PERSONA_SIM_ENABLED
# gates only whether its SCHEMA is registered (tool_schemas/chat), the repo's one
# reliable way to keep the model from calling it. Separate appended fence so V5
# stays a clean additive edit alongside the V1-V3 persona re-exports above.
from .persona_sim import simulate_contact
# --- end feature: persona (V5 simulate) ---

__all__ = [
    "DB_PATH",
    "embed_texts",
    "find_similar_conversations",
    # --- feature: conflict ---
    "diagnose_conflict",
    "find_conflicts",
    "get_conflict",
    "set_dv_screen_bypass",
    # --- end feature: conflict ---
    "get_activity_patterns",
    "get_chat_participant_stats",
    "get_chat_summary_stats",
    "get_conversation",
    "get_full_messages",
    "get_group_dynamics",
    "get_message_context",
    "get_message_reactions",
    "get_milestones",
    "get_reaction_stats",
    "get_response_dynamics",
    # --- feature: persona ---
    "check_style_match",
    "draft_reply",
    "get_voiceprint",
    # --- end feature: persona ---
    "get_word_frequency",
    "get_word_trends",
    "list_chats",
    "search_contact_by_handle",
    "search_contacts_by_name",
    "search_conversations",
    "search_messages_fts",
    "search_messages_hybrid",
    "search_messages_sql",
    # private date helpers, re-exported because tests import them by name
    "_date_bound_step",
    "_normalize_date_bound",
    # shared query helper for the dyad-scoped features (1:1-only restriction)
    "_restrict_to_direct_chats",
    # --- feature: nudges ---
    "get_loose_threads",
    "mute_contact_from_nudges",
    "resolve_loose_thread",
    # --- end feature: nudges ---
    # --- feature: wellbeing ---
    "get_language_trajectory",
    # --- end feature: wellbeing ---
    # --- feature: drift ---
    "get_fadeout_watchlist",
    "get_relationship_drift",
    # --- end feature: drift ---
    # --- feature: drift ---
    "get_reconnection_candidates",
    # --- end feature: drift ---
    # --- feature: conflict (C3 trace) ---
    "trace_conflict_across_chats",
    # --- end feature: conflict (C3 trace) ---
    # --- feature: conflict (C4 apology) ---
    "draft_apology",
    # --- end feature: conflict (C4 apology) ---
    # --- feature: persona (V5 simulate) ---
    "simulate_contact",
    # --- end feature: persona (V5 simulate) ---
]
