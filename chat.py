"""
Interactive CLI chat over iMessage history via a local Ollama-hosted Gemma
model, using tool calling into query_tools.py's retrieval functions.

Usage:
    .venv/bin/python3 chat.py                      # interactive REPL
    .venv/bin/python3 chat.py -p "your question"   # one-shot: answer and exit
    .venv/bin/python3 chat.py -m qwen3:8b          # override the chat model
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
from config import CHAT_MODEL, MAX_TOOL_ITERATIONS, TOOL_RESULT_MAX_CHARS
from tool_schemas import TOOL_DEFINITIONS

console = Console()

# Keep the tool list below in sync with tool_schemas.TOOL_DEFINITIONS -- it
# deliberately restates a one-line summary of each tool. (TOOL_FUNCTIONS is
# derived from TOOL_DEFINITIONS, so it never needs a manual edit.)
SYSTEM_PROMPT = (
    "You are an assistant with access to the user's iMessage history through "
    "tool calls:\n"
    "- search_messages_hybrid (keyword + semantic search over conversations -- "
    "the best default search)\n"
    "- search_conversations (semantic/topic search over conversation chunks)\n"
    "- search_messages_fts (exact keyword/phrase search over messages)\n"
    "- search_messages_sql (filter messages by chat/sender/date)\n"
    "- find_similar_conversations (more conversations like a chunk_id)\n"
    "- get_conversation (a chunk_id -> its full messages)\n"
    "- get_message_context (messages before/after one message id)\n"
    "- get_chat_summary_stats (message counts for a chat, a person, or the whole "
    "database; group_by_chat ranks the top chats by activity)\n"
    "- search_contacts_by_name (name/org -> contact + their handle_ids)\n"
    "- list_chats (name/participant -> chat_id, or label a chat_id)\n"
    "- search_contact_by_handle (a phone/email/handle_id -> contact name)\n"
    "- get_message_reactions (tapbacks on a message, by its guid)\n"
    "- get_full_messages (full untruncated text for one or more message ids, "
    "when a search result was truncated)\n"
    "When a question names a person or group, first use search_contacts_by_name "
    "or list_chats to resolve it to the handle_id / chat_id / contact_id the "
    "search tools need. For 'what did we say about X' or topic/analysis "
    "questions, prefer search_messages_hybrid or search_conversations; their "
    "results are conversation chunks with a truncated text preview -- expand a "
    "promising chunk_id with get_conversation. For a single message hit, "
    "get_message_context shows the surrounding exchange. Message results "
    "already include a `sender` name, a `guid`, and has_attachment; use "
    "get_full_messages with message `id`s if you need complete long texts. "
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

# Every schema name must be a query_tools function of the same name; a
# missing one fails here at import time, not at the model's first call.
TOOL_FUNCTIONS = {t["name"]: getattr(query_tools, t["name"]) for t in TOOL_DEFINITIONS}


def ollama_chat_stream(messages, tools=OLLAMA_TOOLS, show_model_header=True, model=None):
    """Stream one chat request via ollama.chat_stream, live-rendering markdown
    content as it arrives and returning the assembled final message dict (same
    shape as a non-streaming response: role/content/tool_calls).

    Pass tools=None to withhold the tool schemas entirely, which is the only
    reliable way to stop the model from calling one. show_model_header=False
    skips the "model>" line above the reply (one-shot mode, where nothing
    else shares the terminal). model=None uses CHAT_MODEL from config."""
    model = model or CHAT_MODEL
    role = "assistant"
    content = ""
    tool_calls = None
    live = None
    try:
        for msg in ollama.chat_stream(messages, tools, model=model):
            role = msg.get("role", role)
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
            piece = msg.get("content", "")
            if piece:
                if live is None:
                    if show_model_header:
                        console.print(f"[bold]{model}>[/bold]")
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


def _estimate_tokens(text):
    """Rough token count (~4 chars/token). Ollama exposes no tokenizer
    endpoint, and the real cost of a tool result only shows up inside the
    *next* request's aggregate prompt_eval_count, so an upfront per-call
    estimate is the best available."""
    return max(1, len(text) // 4)


def _cap_tool_result(result, result_json):
    """Cap one tool result's JSON at TOOL_RESULT_MAX_CHARS while keeping it
    valid JSON. An oversized tool message makes Ollama silently drop it AND
    every message before it (only the system prompt survives), so the model
    would answer with no user question in context. List results (the only
    shape that gets big in practice) are cut at whole-row boundaries so every
    kept row stays parseable; anything else keeps a prefix explicitly labeled
    as a fragment."""
    if len(result_json) <= TOOL_RESULT_MAX_CHARS:
        return result_json
    note = (
        f"result was {len(result_json)} chars, over the {TOOL_RESULT_MAX_CHARS}-char "
        "cap; re-call with narrower filters or a smaller limit"
    )
    if isinstance(result, list) and result:
        # Proportional first guess, then halve until the wrapper fits.
        keep = max(1, len(result) * TOOL_RESULT_MAX_CHARS // len(result_json))
        while True:
            capped = json.dumps({
                "truncated": True, "rows_total": len(result), "rows_returned": keep,
                "note": note, "rows": result[:keep],
            })
            if len(capped) <= TOOL_RESULT_MAX_CHARS:
                return capped
            if keep == 1:
                break  # a single row alone busts the cap -- fragment it below
            keep = max(1, keep // 2)
    # JSON-escaping the fragment can inflate it, so leave envelope headroom
    # and halve on the (rare) overshoot.
    cut = TOOL_RESULT_MAX_CHARS - 200
    while True:
        capped = json.dumps({"truncated": True, "note": note,
                             "partial_json": result_json[:cut]})
        if len(capped) <= TOOL_RESULT_MAX_CHARS or cut == 0:
            return capped
        cut //= 2


def run_turn(messages, show_model_header=True, model=None):
    """Run one user turn to completion, including any tool-call round trips.

    All MAX_TOOL_ITERATIONS iterations may call tools. If that budget is
    exhausted, one final request is made with the tools withheld and a
    final-turn note folded into the system prompt, forcing an answer out of
    whatever was already gathered."""
    for _ in range(MAX_TOOL_ITERATIONS):
        message = ollama_chat_stream(messages, show_model_header=show_model_header, model=model)
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
            result_json = _cap_tool_result(result, json.dumps(result))
            console.print(
                f"  [tool call] {name}({args}) -> {count}"
                f" (~{_estimate_tokens(result_json)} tokens)",
                style="dim cyan",
            )
            messages.append({"role": "tool", "content": result_json})

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
    message = ollama_chat_stream(
        final_messages, tools=None, show_model_header=show_model_header, model=model
    )
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
    parser.add_argument(
        "-m",
        "--model",
        default=CHAT_MODEL,
        help="override the chat model for this run (default: CHAT_MODEL from .env, "
        "currently %(default)s)",
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
            reply = run_turn(messages, show_model_header=False, model=args.model)
        except (http.client.HTTPException, OSError) as e:
            print(f"[error contacting Ollama: {e}]")
            sys.exit(1)
        # A non-empty reply was already streamed live by ollama_chat_stream.
        if not reply:
            print(f"empty content. Return value: {reply!r}")
        return

    messages = [{"role": "system", "content": _build_system_prompt()}]
    print(f"Chatting with your iMessage history ({args.model} via Ollama). Ctrl+C to quit.")
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
            reply = run_turn(messages, model=args.model)
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
