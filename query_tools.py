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


def search_messages_sql(chat_id=None, handle_id=None, is_from_me=None,
                         date_from=None, date_to=None, limit=50):
    """Structured filter search -- no text matching, just facets."""
    clauses, params = [], []
    if chat_id is not None:
        clauses.append("chat_id = ?"); params.append(chat_id)
    if handle_id is not None:
        clauses.append("handle_id = ?"); params.append(handle_id)
    if is_from_me is not None:
        clauses.append("is_from_me = ?"); params.append(int(is_from_me))
    if date_from is not None:
        clauses.append("date >= ?"); params.append(date_from)
    if date_to is not None:
        clauses.append("date <= ?"); params.append(date_to)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT id, chat_id, handle_id, is_from_me, text, date
        FROM messages
        {where}
        ORDER BY date DESC
        LIMIT ?
    """
    params.append(limit)

    with contextlib.closing(_connect()) as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_messages_fts(query: str, limit=50):
    """Keyword/phrase search via FTS5 (supports FTS5 query syntax:
    'foo AND bar', '"exact phrase"', 'foo NOT bar', etc.)"""
    with contextlib.closing(_connect()) as con:
        rows = con.execute("""
            SELECT m.id, m.chat_id, m.handle_id, m.is_from_me, m.text, m.date
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
    return [dict(r) for r in rows]


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
    query_vec = sqlite_vec.serialize_float32(_ollama_embed(query))

    sql = """
        SELECT m.id, m.chat_id, m.handle_id, m.is_from_me, m.text, m.date, v.distance
        FROM message_vectors v
        JOIN messages m ON m.id = v.message_id
        WHERE v.embedding MATCH ? AND k = ?
    """
    params = [query_vec, k]
    if chat_id is not None:
        sql += " AND m.chat_id = ?"
        params.append(chat_id)
    sql += " ORDER BY v.distance"

    with contextlib.closing(_connect()) as con:
        rows = con.execute(sql, params).fetchall()
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
                        "'messages I sent in chat X'.",
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
        "description": "Keyword/phrase search across all message text using FTS5 syntax.",
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
                        "natural-language query, even without exact keyword overlap.",
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
]
