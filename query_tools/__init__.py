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
# --- end feature: persona ---

# --- feature: nudges ---
from .nudges import (
    get_loose_threads,
    mute_contact_from_nudges,
    resolve_loose_thread,
)
# --- end feature: nudges ---

__all__ = [
    "DB_PATH",
    "embed_texts",
    "find_similar_conversations",
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
]
