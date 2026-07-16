"""Tool schemas for query_tools.py's retrieval functions, for wiring into an
LLM tool-calling loop (used by chat.py; directly usable as Claude tool_use
schemas). Keep in sync with the functions in query_tools.py and the tool list
in chat.py's SYSTEM_PROMPT.
"""

# Shared schema fragment for params that accept either a single id or a list
# of ids (chat_id, handle_id, contact_id, target_message_guid).
_ID_OR_LIST_SCHEMA = {
    "type": ["integer", "array"],
    "items": {"type": "integer"},
    "description": "a single id, or a list of ids",
}

# Shared description for date_from/date_to params. Bounds are normalized by
# query_tools._normalize_date_bound, so this advertises what that accepts.
_DATE_PARAM_DESC = (
    "ISO date ('2026-01-01') or ISO datetime ('2026-01-01T13:00:00'). date_to "
    "covers the whole unit named: '2026-01-01' means through the end of Jan 1, "
    "'2026-01-01T13:00' through the end of that minute. Stored dates are UTC; "
    "pass UTC, or include an offset ('2026-01-01T13:00:00-05:00', '...Z') and "
    "it will be converted for you."
)

TOOL_DEFINITIONS = [
    {
        "name": "search_messages_sql",
        "description": "Filter messages by chat, sender, or date range. Use for "
                        "structured questions like 'messages from March' or "
                        "'messages I sent in chat X'. Prefer contact_id to scope "
                        "to one person across every handle/phone/email they use "
                        "(get it from search_contacts_by_name). chat_id and "
                        "handle_id each accept a single id or a list of ids. Long "
                        "message text is truncated -- use get_full_messages(ids) "
                        "for the complete content of specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "is_from_me": {"type": "boolean"},
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "search_messages_fts",
        "description": "Keyword/phrase search across all message text using FTS5 "
                        "syntax. Optionally scope to a chat/handle/contact (single "
                        "id or list each; contact_id covers every handle that "
                        "person owns). Long message text is truncated -- use "
                        "get_full_messages(ids) for the complete content of "
                        "specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_conversations",
        "description": "Semantic search over conversation chunks (windows of "
                        "consecutive messages in one chat, with speakers "
                        "labeled): finds conversations similar in meaning to a "
                        "natural-language query, even without keyword overlap. "
                        "Best for topic/theme/analysis questions. Optionally "
                        "scope by chat_id, by handle_id/contact_id (that "
                        "person's chats), and/or a date range. Each result has "
                        "a text preview -- expand one with "
                        "get_conversation(chunk_id).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_messages_hybrid",
        "description": "The best default search: runs keyword search and "
                        "semantic conversation search together and merges the "
                        "rankings (reciprocal-rank fusion). Returns ranked "
                        "conversation chunks, each tagged matched_by "
                        "('keyword' and/or 'semantic'). Same scoping as "
                        "search_conversations. Expand a result with "
                        "get_conversation(chunk_id).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_similar_conversations",
        "description": "More-like-this: given a chunk_id from a previous "
                        "conversation search, find other conversations about "
                        "similar things (e.g. the same topic in another chat or "
                        "time period). Uses the stored vector -- cheap. Accepts "
                        "the same chat/handle/contact/date filters as "
                        "search_conversations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer"},
                "k": {"type": "integer", "default": 10},
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "get_conversation",
        "description": "Expand a conversation chunk (by chunk_id from "
                        "search_conversations / search_messages_hybrid / "
                        "find_similar_conversations) into its full messages, "
                        "with senders, dates, and guids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer"},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "get_message_context",
        "description": "Get the messages immediately before and after one "
                        "message in its chat -- turns a single search hit into "
                        "a readable stretch of conversation. Pass a message "
                        "`id` from any message search result; the target row "
                        "is flagged is_target.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "integer"},
                "before": {"type": "integer", "default": 10},
                "after": {"type": "integer", "default": 10},
            },
            "required": ["message_id"],
        },
    },
    {
        "name": "get_chat_summary_stats",
        "description": "Get message counts, date range, attachment_count "
                        "(individual attachments -- a message with three photos "
                        "counts three), and reaction_count. Scope with chat_id "
                        "(single id or list, aggregates across all of them), or "
                        "with handle_id/contact_id to get totals for one person "
                        "across every chat/handle they use (e.g. 'how many "
                        "messages have I exchanged with X'). Pass is_from_me to "
                        "narrow to only sent or only received messages. Pass "
                        "date_from/date_to to narrow to a date or datetime "
                        "range. Every filter applies to all three counts alike "
                        "-- is_from_me=true counts the messages, attachments, "
                        "and reactions you sent. Pass group_by_chat=true "
                        "to get one row per chat instead of a single pooled "
                        "total, for comparing activity across chats. At least "
                        "one of chat_id, handle_id, contact_id is required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "is_from_me": {"type": "boolean"},
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "group_by_chat": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "search_contact_by_handle",
        "description": "Look up contact info (name, organization) for a handle "
                        "(phone number/email or internal handle_id), including "
                        "every other handle/phone-number/email owned by the same "
                        "contact.",
        "input_schema": {
            "type": "object",
            "properties": {
                "handle_id": {"type": "integer"},
                "handle_identifier": {"type": "string", "description": "raw phone number or email"},
            },
        },
    },
    {
        "name": "search_contacts_by_name",
        "description": "Find contacts by name or organization. Use this FIRST to "
                        "turn a person's name into the handle_id / chat_id values "
                        "the other tools need. Returns matched contacts and all "
                        "phone numbers/emails (handles) each one owns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_chats",
        "description": "List conversations with their resolved participant names. "
                        "Use to find a chat_id from a person/group name, or to "
                        "learn who a chat_id refers to. `query` fuzzy-matches by "
                        "participant name or chat identifier substring. Prefer "
                        "handle_id/contact_id (single id or list, from "
                        "search_contacts_by_name) for an exact match on every "
                        "chat that person actually participates in.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "substring to match against participant names / chat identifier",
                },
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "get_message_reactions",
        "description": "List the reactions/tapbacks on one or more messages and "
                        "who reacted. Pass the `guid` (or a list of guids) from "
                        "message search results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_message_guid": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "a single guid, or a list of guids",
                },
            },
            "required": ["target_message_guid"],
        },
    },
    {
        "name": "get_full_messages",
        "description": "Get the full, untruncated text of one or more messages "
                        "by id. Search results truncate long message text to "
                        "keep responses fast -- call this when you need the "
                        "complete content of specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "one or more message ids from a prior search result",
                },
            },
            "required": ["ids"],
        },
    },
]
