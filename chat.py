"""
Interactive CLI chat over iMessage history via a local Ollama-hosted Gemma
model, using tool calling into query_tools.py's retrieval functions.

Usage:
    .venv/bin/python3 chat.py                      # interactive REPL
    .venv/bin/python3 chat.py -p "your question"   # one-shot: answer and exit
"""

import argparse
import datetime
import http.client
import json
import readline  # noqa: F401 — enables arrow-key/history line editing in input()
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import ollama
import query_tools
from config import CHAT_MODEL, MAX_TOOL_ITERATIONS
from tool_schemas import TOOL_DEFINITIONS

console = Console()

# Keep the tool list below in sync with tool_schemas.TOOL_DEFINITIONS and
# TOOL_FUNCTIONS -- it deliberately restates a one-line summary of each tool.
SYSTEM_PROMPT = (
    "You are an assistant with access to the user's iMessage history through "
    "tool calls:\n"
    "- search_messages_sql (filter by chat/sender/date)\n"
    "- search_messages_fts (keyword search)\n"
    "- search_messages_vec (semantic/topic search)\n"
    "- get_chat_summary_stats (message counts for a chat)\n"
    "- search_contacts_by_name (name/org -> contact + their handle_ids)\n"
    "- list_chats (name/participant -> chat_id, or label a chat_id)\n"
    "- search_contact_by_handle (a phone/email/handle_id -> contact name)\n"
    "- get_message_reactions (tapbacks on a message, by its guid)\n"
    "- get_full_messages (full untruncated text for one or more message ids, "
    "when a search result was truncated)\n"
    "When a question names a person or group, first use search_contacts_by_name "
    "or list_chats to resolve it to the handle_id / chat_id the message-search "
    "tools need. Message results already include a `sender` name, a `guid`, and "
    "has_attachment. Message search results truncate long text -- use "
    "get_full_messages with the message `id` if you need the complete content. "
    "Always call a tool to look up message content before answering -- never "
    "guess or invent messages. Cite dates when relevant."
)

def _build_system_prompt():
    """SYSTEM_PROMPT plus the current UTC date/time, computed fresh each time --
    message dates in the DB are stored as unlabeled-UTC ISO strings (see
    query_tools.search_messages_sql's docstring), so the model needs both
    'what time is it now' and 'the DB is UTC, not local' to correctly convert a
    timezone-relative request (e.g. '1-3pm EST yesterday') before calling
    date_from/date_to on search_messages_sql or get_chat_summary_stats."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"The current date/time is {now} (UTC). All message dates in the "
        "database, and the date_from/date_to arguments those tools take, are "
        "UTC with no offset suffix -- if the user's question is relative to a "
        "timezone or to 'now' (e.g. 'yesterday', '3pm EST', 'this morning'), "
        "convert it to UTC before calling the tool."
    )

FINAL_TURN_NOTE = (
    "This is your final turn -- no tool calls remain. Answer the user now using only "
    "the information already gathered; do not call any more tools. If the information "
    "gathered is incomplete, say so plainly rather than guessing."
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
    "search_contact_by_handle": query_tools.search_contact_by_handle,
    "search_contacts_by_name": query_tools.search_contacts_by_name,
    "list_chats": query_tools.list_chats,
    "get_message_reactions": query_tools.get_message_reactions,
    "get_full_messages": query_tools.get_full_messages,
}


def ollama_chat_stream(messages, tools=OLLAMA_TOOLS, show_model_header=True):
    """Stream one chat request via ollama.chat_stream, live-rendering markdown
    content as it arrives and returning the assembled final message dict (same
    shape as a non-streaming response: role/content/tool_calls).

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one. show_model_header=False
    skips the "model>" line above the reply (one-shot mode, where nothing
    else shares the terminal)."""
    role = "assistant"
    content = ""
    tool_calls = None
    live = None
    try:
        for msg in ollama.chat_stream(messages, tools):
            role = msg.get("role", role)
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
            piece = msg.get("content", "")
            if piece:
                if live is None:
                    if show_model_header:
                        console.print(f"[bold]{CHAT_MODEL}>[/bold]")
                    live = Live(Markdown(""), console=console, refresh_per_second=12)
                    live.start()
                content += piece
                live.update(Markdown(content))
    finally:
        if live is not None:
            live.stop()

    result = {"role": role, "content": content}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def run_turn(messages, show_model_header=True):
    """Run one user turn to completion, including any tool-call round trips.

    All MAX_TOOL_ITERATIONS iterations may call tools. If that budget is
    exhausted, one final request is made with the tools withheld and a
    final-turn note folded into the system prompt, forcing an answer out of
    whatever was already gathered."""
    for _ in range(MAX_TOOL_ITERATIONS):
        message = ollama_chat_stream(messages, show_model_header=show_model_header)
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
            console.print(f"  [tool call] {name}({args}) -> {count}", style="dim cyan")
            messages.append({"role": "tool", "content": json.dumps(result)})

    # Budget exhausted. gemma4's Ollama renderer keeps only the *first* system
    # message and drops later ones, so the note has to be folded into the system
    # prompt rather than appended as its own turn. Extend messages[0] as found
    # rather than rebuilding from SYSTEM_PROMPT -- callers may seed a system
    # prompt with extra context that must survive into this request.
    if messages and messages[0]["role"] == "system":
        head, rest = messages[0]["content"], messages[1:]
    else:
        head, rest = SYSTEM_PROMPT, messages
    final_messages = [{"role": "system", "content": head + "\n\n" + FINAL_TURN_NOTE}] + rest
    message = ollama_chat_stream(final_messages, tools=None, show_model_header=show_model_header)
    messages.append(message)
    return message.get("content", "")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with your iMessage history via a local Ollama model."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help='run a single prompt non-interactively and exit (quote it: -p "...")',
    )
    args = parser.parse_args()

    if args.prompt is not None:
        prompt = args.prompt.strip()
        if not prompt:
            parser.error("prompt is empty")
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = run_turn(messages, show_model_header=False)
        except (http.client.HTTPException, OSError) as e:
            print(f"[error contacting Ollama: {e}]")
            sys.exit(1)
        # A non-empty reply was already streamed live by ollama_chat_stream.
        if not reply:
            print(f"empty content. Return value: {reply!r}")
        return

    messages = [{"role": "system", "content": _build_system_prompt()}]
    print(f"Chatting with your iMessage history ({CHAT_MODEL} via Ollama). Ctrl+C to quit.")
    print("Commands: /clear (reset conversation)\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/clear":
            messages = [{"role": "system", "content": _build_system_prompt()}]
            console.print("[dim]conversation cleared[/dim]\n")
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            reply = run_turn(messages)
        except (http.client.HTTPException, OSError) as e:
            print(f"[error contacting Ollama: {e}]")
            messages.pop()
            continue
        except KeyboardInterrupt:
            print()
            break
        # A non-empty reply was already streamed live by ollama_chat_stream;
        # only surface the return value when there was nothing to render.
        if not reply:
            print(f"empty content. Return value: {reply!r}")
        print()


if __name__ == "__main__":
    main()
