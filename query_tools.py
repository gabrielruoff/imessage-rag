"""
Query layer over imessage_analysis.db, meant to be exposed as LLM tool calls
(Claude tool_use / function-calling) rather than pre-loading data into context.

Three retrieval modes:
  - search_messages_sql   structured filters (chat, sender, date range)
  - search_messages_fts   keyword / phrase search via FTS5
  - search_messages_vec   semantic search via sqlite-vec + Ollama embeddings

Each returns a small list of dicts, meant to be fed back to the LLM as tool
results -- never the whole table.
"""

import contextlib
import json
import sqlite3
import urllib.error
import urllib.request

import sqlite_vec

from config import EMBED_MODEL, OLLAMA_EMBEDDINGS_PATH, OLLAMA_HOST, OLLAMA_PORT, TARGET_DB

DB_PATH = str(TARGET_DB)
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_EMBEDDINGS_PATH}"


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.row_factory = sqlite3.Row
    return con


# Human-readable label for a contact/handle: nickname > "First Last" > raw
# handle_identifier. Reused wherever a message's sender needs a name. Assumes
# the query LEFT JOINs handles AS h and contacts AS c (on h.contact_id).
_CONTACT_LABEL = (
    "COALESCE("
    "NULLIF(c.nickname, ''), "
    "NULLIF(TRIM(COALESCE(c.first_name, '') || ' ' || COALESCE(c.last_name, '')), ''), "
    "h.handle_identifier)"
)

# Resolve a message's sender: "Me" when is_from_me, else the contact label.
# Requires the join below.
_SENDER_SELECT = f"CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE {_CONTACT_LABEL} END AS sender"
_MSG_JOIN = (
    "LEFT JOIN handles h ON h.id = m.handle_id "
    "LEFT JOIN contacts c ON c.id = h.contact_id"
)

# Ceilings on model-requested result sizes -- defaults above are just what's
# used when the model omits the arg; these are enforced regardless of what
# the model asks for, so a tool call can't blow up prompt size.
MAX_MESSAGE_LIMIT = 50   # search_messages_sql / search_messages_fts
MAX_VEC_K = 20           # search_messages_vec
MAX_CONTACTS_LIMIT = 20  # search_contacts_by_name
MAX_CHATS_LIMIT = 50     # list_chats

TEXT_PREVIEW_CHARS = 300


def _truncate_text_field(rows):
    """Cap each row's `text` field for prompt size; full text is available
    via get_full_messages(ids)."""
    for r in rows:
        text = r.get("text")
        if text and len(text) > TEXT_PREVIEW_CHARS:
            r["text"] = text[:TEXT_PREVIEW_CHARS] + "…"
    return rows


def search_messages_sql(chat_id=None, handle_id=None, is_from_me=None,
                         date_from=None, date_to=None, limit=50):
    """Structured filter search -- no text matching, just facets."""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))
    clauses, params = [], []
    if chat_id is not None:
        clauses.append("m.chat_id = ?"); params.append(chat_id)
    if handle_id is not None:
        clauses.append("m.handle_id = ?"); params.append(handle_id)
    if is_from_me is not None:
        clauses.append("m.is_from_me = ?"); params.append(int(is_from_me))
    if date_from is not None:
        clauses.append("m.date >= ?"); params.append(date_from)
    if date_to is not None:
        clauses.append("m.date <= ?"); params.append(date_to)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
               m.text, m.date, m.has_attachment
        FROM messages m
        {_MSG_JOIN}
        {where}
        ORDER BY m.date DESC
        LIMIT ?
    """
    params.append(limit)

    with contextlib.closing(_connect()) as con:
        rows = con.execute(sql, params).fetchall()
    return _truncate_text_field([dict(r) for r in rows])


def search_messages_fts(query: str, limit=50):
    """Keyword/phrase search via FTS5 (supports FTS5 query syntax:
    'foo AND bar', '"exact phrase"', 'foo NOT bar', etc.)"""
    limit = max(1, min(limit, MAX_MESSAGE_LIMIT))
    with contextlib.closing(_connect()) as con:
        rows = con.execute(f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            {_MSG_JOIN}
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
    return _truncate_text_field([dict(r) for r in rows])


def _ollama_embed(text: str):
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return body["embedding"]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ollama embedding request failed: {e}") from e


def search_messages_vec(query: str, k=10, chat_id=None):
    """Semantic search: embed the query and find nearest message vectors."""
    k = max(1, min(k, MAX_VEC_K))
    query_vec = sqlite_vec.serialize_float32(_ollama_embed(query))

    sql = f"""
        SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
               m.text, m.date, m.has_attachment, v.distance
        FROM message_vectors v
        JOIN messages m ON m.id = v.message_id
        {_MSG_JOIN}
        WHERE v.embedding MATCH ? AND k = ?
    """
    params = [query_vec, k]
    if chat_id is not None:
        sql += " AND m.chat_id = ?"
        params.append(chat_id)
    sql += " ORDER BY v.distance"

    with contextlib.closing(_connect()) as con:
        rows = con.execute(sql, params).fetchall()
    return _truncate_text_field([dict(r) for r in rows])


def search_contact_by_handle(handle_id=None, handle_identifier=None):
    """Look up contact info for a handle, plus every other handle the same contact owns."""
    if handle_id is None and handle_identifier is None:
        raise ValueError("must provide handle_id or handle_identifier")

    handle_cols = "id, handle_identifier, contact_id, service"
    with contextlib.closing(_connect()) as con:
        if handle_id is not None:
            handle = con.execute(
                f"SELECT {handle_cols} FROM handles WHERE id = ?", (handle_id,)
            ).fetchone()
        else:
            handle = con.execute(
                f"SELECT {handle_cols} FROM handles WHERE handle_identifier = ?",
                (handle_identifier,),
            ).fetchone()

        if handle is None:
            return None

        handle = dict(handle)
        contact = None
        other_handles = []
        if handle["contact_id"] is not None:
            contact = con.execute(
                "SELECT id, source_uniqueid, first_name, last_name, nickname, organization "
                "FROM contacts WHERE id = ?", (handle["contact_id"],)
            ).fetchone()
            contact = dict(contact) if contact else None

            other_handles = con.execute(
                "SELECT id, handle_identifier, service FROM handles "
                "WHERE contact_id = ? AND id != ?",
                (handle["contact_id"], handle["id"]),
            ).fetchall()
            other_handles = [dict(r) for r in other_handles]

    return {
        "handle": handle,
        "contact": contact,
        "other_handles": other_handles,
    }


def search_contacts_by_name(name: str, limit=20):
    """Find contacts by name or organization -- the bridge from a human name to
    the handle_id / chat_id values the other tools need. Returns each matched
    contact plus all handles (phone numbers/emails) it owns."""
    limit = max(1, min(limit, MAX_CONTACTS_LIMIT))
    like = f"%{name}%"
    with contextlib.closing(_connect()) as con:
        contacts = con.execute(
            """
            SELECT id, source_uniqueid, first_name, last_name, nickname, organization
            FROM contacts
            WHERE first_name LIKE ? OR last_name LIKE ? OR nickname LIKE ?
               OR organization LIKE ?
               OR TRIM(COALESCE(first_name,'') || ' ' || COALESCE(last_name,'')) LIKE ?
            ORDER BY last_name, first_name
            LIMIT ?
            """,
            (like, like, like, like, like, limit),
        ).fetchall()

        results = []
        for c in contacts:
            c = dict(c)
            handles = con.execute(
                "SELECT id, handle_identifier, service FROM handles WHERE contact_id = ?",
                (c["id"],),
            ).fetchall()
            c["handles"] = [dict(h) for h in handles]
            results.append(c)
    return results


def list_chats(query=None, limit=50):
    """List chats (conversations) with resolved participant names -- the bridge
    from a person/chat name to a chat_id, and the way to label a chat_id. Pass
    `query` to filter by participant name or chat_identifier (substring match)."""
    limit = max(1, min(limit, MAX_CHATS_LIMIT))
    like = f"%{query}%" if query else None
    with contextlib.closing(_connect()) as con:
        # Per-chat participant label list, resolved through handles/contacts.
        # The substring filter and limit are applied in SQL: the HAVING clause
        # is a no-op when `like` is NULL, else it matches the chat_identifier,
        # display_name, or the GROUP_CONCAT'd participant labels. LIKE is
        # case-insensitive for ASCII, matching the prior Python behavior.
        rows = con.execute(
            f"""
            SELECT ch.id, ch.chat_identifier, ch.display_name, ch.is_group, ch.service,
                   GROUP_CONCAT({_CONTACT_LABEL}, ', ') AS participants
            FROM chats ch
            LEFT JOIN chat_participants cp ON cp.chat_id = ch.id
            LEFT JOIN handles h ON h.id = cp.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            GROUP BY ch.id
            HAVING :like IS NULL
                OR ch.chat_identifier LIKE :like
                OR ch.display_name LIKE :like
                OR participants LIKE :like
            ORDER BY ch.id
            LIMIT :limit
            """,
            {"like": like, "limit": limit},
        ).fetchall()

    return [dict(r) for r in rows]


def get_message_reactions(target_message_guid: str):
    """List the reactions/tapbacks on a message (by its guid), with who reacted.
    Message search results include each message's `guid`."""
    with contextlib.closing(_connect()) as con:
        rows = con.execute(
            f"""
            SELECT r.id, r.is_from_me,
                   CASE WHEN r.is_from_me = 1 THEN 'Me' ELSE {_CONTACT_LABEL} END AS reactor,
                   r.associated_message_type, r.date
            FROM reactions r
            LEFT JOIN handles h ON h.id = r.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE r.target_message_guid = ?
            ORDER BY r.date
            """,
            (target_message_guid,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_full_messages(ids):
    """Return the full, untruncated text for one or more message ids.
    Use after a search result's `text` was truncated and the full content
    is needed."""
    if isinstance(ids, int):
        ids = [ids]
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with contextlib.closing(_connect()) as con:
        rows = con.execute(f"""
            SELECT m.id, m.guid, m.chat_id, m.handle_id, m.is_from_me, {_SENDER_SELECT},
                   m.text, m.date, m.has_attachment
            FROM messages m
            {_MSG_JOIN}
            WHERE m.id IN ({placeholders})
        """, ids).fetchall()
    return [dict(r) for r in rows]


def get_chat_summary_stats(chat_id: int):
    """Cheap aggregate stats for a chat -- no LLM call needed."""
    with contextlib.closing(_connect()) as con:
        row = con.execute("""
            SELECT
                COUNT(*) AS message_count,
                MIN(date) AS first_message,
                MAX(date) AS last_message,
                SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) AS from_me,
                SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) AS from_them
            FROM messages WHERE chat_id = ?
        """, (chat_id,)).fetchone()
    return dict(row)


# --- Tool schemas, for wiring into Claude's tool_use API -------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_messages_sql",
        "description": "Filter messages by chat, sender, or date range. Use for "
                        "structured questions like 'messages from March' or "
                        "'messages I sent in chat X'. Long message text is "
                        "truncated -- use get_full_messages(ids) for the "
                        "complete content of specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer"},
                "handle_id": {"type": "integer"},
                "is_from_me": {"type": "boolean"},
                "date_from": {"type": "string", "description": "ISO date"},
                "date_to": {"type": "string", "description": "ISO date"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "search_messages_fts",
        "description": "Keyword/phrase search across all message text using FTS5 "
                        "syntax. Long message text is truncated -- use "
                        "get_full_messages(ids) for the complete content of "
                        "specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_messages_vec",
        "description": "Semantic search: find messages similar in meaning to a "
                        "natural-language query, even without exact keyword "
                        "overlap. Long message text is truncated -- use "
                        "get_full_messages(ids) for the complete content of "
                        "specific messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "chat_id": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_chat_summary_stats",
        "description": "Get message counts and date range for a given chat.",
        "input_schema": {
            "type": "object",
            "properties": {"chat_id": {"type": "integer"}},
            "required": ["chat_id"],
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
                        "learn who a chat_id refers to. Optional `query` filters "
                        "by participant name or chat identifier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "substring to match against participant names / chat identifier"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "get_message_reactions",
        "description": "List the reactions/tapbacks on a specific message and who "
                        "reacted. Pass the `guid` from a message search result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_message_guid": {"type": "string"},
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
