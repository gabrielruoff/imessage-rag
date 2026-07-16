"""
Interactive CLI chat over iMessage history via a local Ollama-hosted Gemma
model, using tool calling into query_tools.py's retrieval functions.

Usage:
    .venv/bin/python3 chat.py
"""

import http.client
import json

import query_tools
from query_tools import TOOL_DEFINITIONS

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
CHAT_PATH = "/api/chat"
CHAT_MODEL = "gemma4:26b-a4b-it-qat"
CHAT_TIMEOUT = 300
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = (
    "You are an assistant with access to the user's iMessage history through "
    "tool calls: search_messages_sql (filter by chat/sender/date), "
    "search_messages_fts (keyword search), search_messages_vec (semantic/topic "
    "search), and get_chat_summary_stats (message counts for a chat). Always "
    "call a tool to look up message content before answering -- never guess "
    "or invent messages. Cite dates when relevant."
)

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOL_DEFINITIONS
]

TOOL_FUNCTIONS = {
    "search_messages_sql": query_tools.search_messages_sql,
    "search_messages_fts": query_tools.search_messages_fts,
    "search_messages_vec": query_tools.search_messages_vec,
    "get_chat_summary_stats": query_tools.get_chat_summary_stats,
}


def ollama_chat(messages):
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=CHAT_TIMEOUT)
    payload = json.dumps({
        "model": CHAT_MODEL,
        "messages": messages,
        "tools": OLLAMA_TOOLS,
        "stream": False,
    }).encode()
    conn.request("POST", CHAT_PATH, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    if "error" in body:
        raise RuntimeError(f"Ollama error: {body['error']}")
    return body["message"]


def run_turn(messages):
    """Run one user turn to completion, including any tool-call round trips."""
    for _ in range(MAX_TOOL_ITERATIONS):
        message = ollama_chat(messages)
        messages.append(message)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            count = len(result) if isinstance(result, list) else "ok"
            print(f"  [tool call] {name}({args}) -> {count}")
            messages.append({"role": "tool", "content": json.dumps(result)})

    return "(gave up after too many tool calls)"


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Chatting with your iMessage history (gemma via Ollama). Ctrl+C to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            reply = run_turn(messages)
        except (http.client.HTTPException, OSError) as e:
            print(f"[error contacting Ollama: {e}]")
            messages.pop()
            continue
        print(f"gemma> {reply}\n")


if __name__ == "__main__":
    main()
