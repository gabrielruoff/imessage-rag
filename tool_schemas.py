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
                        "for the complete content of specific messages. Returns "
                        "{rows, total_matching, returned, order, truncated}: rows "
                        "holds only the newest `limit` matches -- when truncated "
                        "is true it is a SAMPLE, never the whole population.",
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
                        "specific messages. Returns {rows, total_matching, "
                        "returned, order, truncated}: rows holds only the `limit` "
                        "best-ranked hits -- when truncated is true it is a "
                        "SAMPLE of the matches, not all of them.",
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
                        "counts three), and reactions_sent_count (reactions SENT "
                        "by whoever matches the filters -- NOT received; for "
                        "reactions received use get_reaction_stats or "
                        "get_chat_participant_stats). Scope with chat_id "
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
                        "total, for comparing activity across chats -- rows "
                        "are the top `limit` chats by message count (e.g. "
                        "'who do I text the most'). With no "
                        "scope params at all, aggregates the whole database "
                        "(e.g. 'how many messages total').",
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
                "limit": {"type": "integer", "default": 25,
                          "description": "max rows when group_by_chat=true"},
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
                        "learn who a chat_id refers to (pass chat_id -- single "
                        "id or list -- to label chat_ids you already have, e.g. "
                        "from get_chat_summary_stats). `query` fuzzy-matches by "
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
                "chat_id": _ID_OR_LIST_SCHEMA,
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
                        "message search results. Each row carries a readable "
                        "`reaction` name (love/like/laugh/...) beside the raw "
                        "type code.",
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
    {
        "name": "get_activity_patterns",
        "description": "Aggregate WHEN-do-we-text analytics in one call -- use "
                        "for how-often/busiest/when questions, never by counting "
                        "search results. Sections: heatmap (7x24 weekday-by-hour, "
                        "local time), volume_series (per day/month/year, me vs "
                        "them), late_night (share of messages in the small hours; "
                        "unscoped adds a top-contacts leaderboard), busiest_days, "
                        "streaks (longest daily streak and drought). "
                        "handle_id/contact_id scope to that person's chats (both "
                        "sides counted); no scope = the whole corpus. Pass "
                        "include=['heatmap',...] to fetch only some sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "granularity": {"type": "string", "enum": ["day", "month", "year"],
                                "default": "month",
                                "description": "volume_series bucket size"},
                "include": {
                    "type": "array", "items": {"type": "string"},
                    "description": "sections to return: heatmap, volume_series, "
                                   "late_night, busiest_days, streaks (default all)",
                },
            },
        },
    },
    {
        "name": "get_response_dynamics",
        "description": "Relationship tempo analytics for ONE conversation or "
                        "person -- who texts first, reply speeds, ghosting. "
                        "Sections: reply_latency (median/p90 per side), "
                        "read_to_reply (unseen vs deliberation time, from read "
                        "receipts), initiation (who breaks long silences), "
                        "sessions (count, length, who ends them), ghosted "
                        "(one-sided sessions), double_text, burstiness. REQUIRES "
                        "a scope: chat_id (single or list -- pass every row of a "
                        "split iMessage/SMS thread), or handle_id/contact_id "
                        "(that person's 1:1 chats; group chats need an explicit "
                        "chat_id, which adds initiations_by_member).",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
        },
    },
    {
        "name": "get_reaction_stats",
        "description": "Tapback/reaction analytics in one call -- use for "
                        "who-reacts-most / most-loved-message / reaction-habit "
                        "questions. Sections: given and received (totals, "
                        "per-type, per-100-message rates), reciprocity, "
                        "time_to_react (median seconds, me vs them), funniest "
                        "(contacts ranked by laughs per 100 of my messages), "
                        "hall_of_fame (top most-reacted messages with text). "
                        "reaction_type narrows to one tapback (name like "
                        "'laugh'/'love' or raw code). handle_id/contact_id scope "
                        "to that person's chats; no scope = whole corpus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "reaction_type": {
                    "type": ["string", "integer"],
                    "description": "tapback name (e.g. 'laugh', 'love') or raw "
                                   "Apple type code",
                },
                "include": {
                    "type": "array", "items": {"type": "string"},
                    "description": "sections to return: given, received, "
                                   "reciprocity, time_to_react, funniest, "
                                   "hall_of_fame (default all)",
                },
                "limit": {"type": "integer", "default": 10,
                          "description": "hall_of_fame row cap"},
            },
        },
    },
    {
        "name": "get_word_trends",
        "description": "Language analytics over time -- use for how-our-words-"
                        "changed / inside-joke / emoji / style questions, not "
                        "for plain top-word counts (that's get_word_frequency). "
                        "Sections: phrase_trend (a phrase's usage per period, me "
                        "vs them -- requires `phrase`), emoji (top emoji), rates "
                        "(we/question/laugh per 100 messages), distinctive_terms "
                        "(this chat's own vocabulary with who coined each term "
                        "and its first message -- needs a chat/person scope), "
                        "lsm (language style matching score + trajectory), "
                        "emotions/valence (from LLM chunk annotations, when "
                        "available).",
        "input_schema": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string",
                           "description": "exact word/phrase for phrase_trend"},
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "granularity": {"type": "string", "enum": ["month", "year"],
                                "default": "year"},
                "include": {
                    "type": "array", "items": {"type": "string"},
                    "description": "sections to return: phrase_trend, emoji, "
                                   "rates, distinctive_terms, lsm, emotions, "
                                   "valence (default all applicable)",
                },
            },
        },
    },
    {
        "name": "get_milestones",
        "description": "Milestone lookups in one call -- use for first-message / "
                        "nth-message / anniversary / longest-silence questions. "
                        "Sections: firsts (first/last message and each side's "
                        "first, with text), nth (the nth message when `n` given, "
                        "else the 1000th/5000th/10000th/... landmarks), "
                        "anniversaries (scoped: first-contact date and next "
                        "anniversary; unscoped: contacts with an anniversary in "
                        "the next 30 days), longest_silence (largest gap + the "
                        "message that broke it -- needs a scope), "
                        "earliest_phrase (first use of `phrase`). "
                        "handle_id/contact_id scope to that person's chats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "phrase": {"type": "string",
                           "description": "word/phrase for earliest_phrase"},
                "n": {"type": "integer",
                      "description": "return the nth message in scope (1-based)"},
                "reference_date": {"type": "string",
                                   "description": "ISO date anchoring the "
                                                  "unscoped anniversary window "
                                                  "(default today)"},
                "limit": {"type": "integer", "default": 10,
                          "description": "anniversaries row cap"},
            },
        },
    },
    {
        "name": "get_group_dynamics",
        "description": "Structural analytics for a GROUP chat -- who dominates, "
                        "who lurks, who replies to whom. Sections: members "
                        "(message/word share), gini (participation inequality + "
                        "per-year drift), lurkers, igniters (who starts "
                        "conversations), adjacency (top replier->replied pairs), "
                        "group_lsm (whose style matches the group). chat_id is "
                        "REQUIRED and must be a group chat (pass a list to merge "
                        "a split thread); for 1:1 questions use "
                        "get_response_dynamics instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "include": {
                    "type": "array", "items": {"type": "string"},
                    "description": "sections to return: members, gini, lurkers, "
                                   "igniters, adjacency, group_lsm (default all)",
                },
            },
            "required": ["chat_id"],
        },
    },
    {
        "name": "get_chat_participant_stats",
        "description": "Per-member stats for a chat: one row per participant "
                        "(including 'Me' and silent members) with messages_sent, "
                        "first/last message dates, reactions_given, and "
                        "reactions_received broken down by tapback type -- "
                        "group-chat rankings like 'who talks most' or 'who gets "
                        "the most laughs per message' in one call. chat_id is "
                        "required; pass a list to merge a split iMessage/SMS "
                        "thread into one table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
            "required": ["chat_id"],
        },
    },
    {
        "name": "get_word_frequency",
        "description": "Most frequent words (ngram=1) or phrases (ngram=2-3) "
                        "with real counts -- 'what words do I overuse', catch "
                        "phrases, inside-joke frequency. Unlike the other "
                        "analytics tools this scopes by SENDER: "
                        "handle_id/contact_id means messages that person sent; "
                        "is_from_me=true means messages I sent (combine with "
                        "nothing else for 'my' corpus-wide habits). Common "
                        "English stopwords are excluded unless "
                        "include_stopwords=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "is_from_me": {"type": "boolean"},
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "ngram": {"type": "integer", "default": 1,
                          "description": "1 = single words, 2-3 = phrases"},
                "top": {"type": "integer", "default": 25},
                "min_count": {"type": "integer", "default": 2},
                "include_stopwords": {"type": "boolean", "default": False},
            },
        },
    },
    # --- feature: nudges ---
    {
        "name": "get_loose_threads",
        "description": "The user's current 'loose ends' digest: relationships "
                        "that have drifted (e.g. someone who used to reach out "
                        "and has gone quiet), detected offline and ranked by "
                        "urgency. Use for 'who should I reach back out to', "
                        "'any loose ends', 'who have I been neglecting'. Returns "
                        "at most 5 rows, each with a resolved contact name, the "
                        "detector `kind`, `urgency`, the anchor message/date, the "
                        "`evidence`, and a one-line `nudge_text`. status defaults "
                        "to 'open'; kind optionally narrows to one detector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "dismissed"],
                           "default": "open"},
                "kind": {"type": "string",
                         "description": "narrow to one detector, e.g. 'faded_initiator'"},
                "limit": {"type": "integer", "default": 5,
                          "description": "hard-capped at 5"},
            },
        },
    },
    {
        "name": "resolve_loose_thread",
        "description": "Dismiss one loose thread (nudge) by its id -- the user "
                        "has handled it or wants it gone. Sets it to dismissed "
                        "(never deletes), so it stops resurfacing. Pass the `id` "
                        "from a get_loose_threads row.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "integer",
                              "description": "the loose thread id to dismiss"},
            },
            "required": ["thread_id"],
        },
    },
    {
        "name": "mute_contact_from_nudges",
        "description": "Permanently exclude a contact from future nudges AND "
                        "dismiss their currently-open loose threads -- use when "
                        "the user says to stop nudging about someone. Prefer "
                        "contact_id (covers every handle they own; get it from "
                        "search_contacts_by_name); handle_id also accepted. "
                        "Muting twice never stacks duplicate rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "integer"},
                "handle_id": {"type": "integer"},
                "reason": {"type": "string", "default": "muted"},
            },
        },
    },
    # --- end feature: nudges ---
    # --- feature: persona ---
    {
        "name": "get_voiceprint",
        "description": "How a person texts -- their deterministic 'voiceprint': "
                        "message length, lowercase habit, exclamation/ellipsis "
                        "rates, double-texting burst size, function-word rates, "
                        "signature openers/closers, and a LOW-TRUST emotional "
                        "tone. Defaults to YOUR voice: name a person "
                        "(contact_id/handle_id/name) for your voice WITH them (a "
                        "dyad), or omit for your overall voice. subject='them' "
                        "returns the named person's voice instead. Group chats "
                        "only with scope='group' + a group chat_id. Passing "
                        "date_from/date_to recomputes over that window (capped). "
                        "Returns a card with a confidence flag (high/low), "
                        "coverage counts, and the per-feature corpus std "
                        "(feature_std) for normalization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "integer",
                               "description": "the person (from search_contacts_by_name)"},
                "handle_id": {"type": "integer"},
                "name": {"type": "string",
                         "description": "person name to resolve (contact_id is exact)"},
                "subject": {"type": "string", "enum": ["me", "them"],
                            "default": "me",
                            "description": "whose voice: 'me' (default) or 'them' "
                                           "(the named person)"},
                "scope": {"type": "string", "enum": ["dyad", "group", "all"],
                          "description": "dyad (default when a person is named), "
                                         "all (default when none), group (needs a "
                                         "group chat_id)"},
                "chat_id": {"type": "integer",
                            "description": "group chat id, required for scope='group'"},
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
            },
        },
    },
    # --- end feature: persona ---
    # --- feature: conflict ---
    {
        "name": "find_conflicts",
        "description": "Detected conflict/argument episodes ('tense stretches') "
                        "in a 1:1 chat or with a person, found from DETERMINISTIC "
                        "signals with receipts -- reply-latency withdrawal, "
                        "one-sided sessions, dislike-tapback runs, message-volume "
                        "collapse -- ranked strongest first. Use for 'did we fight "
                        "in June', 'our worst arguments', 'tense periods with X'. "
                        "1:1 only (group chats excluded). Prefer contact_id to "
                        "cover a person across every handle. Each row names which "
                        "`signals` fired and a `signal_count`; escalation labels "
                        "are eval-gated and shown as bands, not certainties -- "
                        "expand one with get_conflict(conflict_id). Returns "
                        "{rows, total_matching, returned, order, truncated}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": _ID_OR_LIST_SCHEMA,
                "handle_id": _ID_OR_LIST_SCHEMA,
                "contact_id": _ID_OR_LIST_SCHEMA,
                "date_from": {"type": "string", "description": _DATE_PARAM_DESC},
                "date_to": {"type": "string", "description": _DATE_PARAM_DESC},
                "limit": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "get_conflict",
        "description": "One conflict episode (by conflict_id from find_conflicts): "
                        "its deterministic evidence (peak_signal), tone "
                        "corroboration, off_text_flag, and up to 15 labeled "
                        "signal turns (paginated; excerpts <= 200 chars). Labels "
                        "are eval-gated BANDS (band 'tentative'/'likely'), never "
                        "raw scores -- describe them as tentative, and never read "
                        "a resolution or apology into an episode whose "
                        "off_text_flag is true (the exchange moved off-channel).",
        "input_schema": {
            "type": "object",
            "properties": {
                "conflict_id": {"type": "integer"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 15},
            },
            "required": ["conflict_id"],
        },
    },
    {
        "name": "diagnose_conflict",
        "description": "A DETERMINISTIC scorecard for one conflict episode (by "
                        "conflict_id from find_conflicts): per-side I-vs-you and "
                        "we-talk language ratios, style-matching (LSM), who opened "
                        "the episode and who had the last word, reply-latency and "
                        "message-volume-collapse read off the stored evidence, and "
                        "a non-blame demand-withdraw read (who withdrew vs kept "
                        "reaching out). It carries NO aggressor/who-won verdict -- "
                        "present it as mechanical description, not a ruling on who "
                        "was right. A built-in safety screen may withhold the "
                        "withdraw/pursue read and append a support-resource footer; "
                        "if a resource_footer is present, include it verbatim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "conflict_id": {"type": "integer"},
            },
            "required": ["conflict_id"],
        },
    },
    # --- end feature: conflict ---
]
