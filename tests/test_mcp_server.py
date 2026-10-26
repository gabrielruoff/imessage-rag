"""mcp_server: the exposed tool set, its listing metadata, the call-handler
error contract, the result cap, and the stdio entrypoint. Hermetic: temp
fixture DB with placeholder people, query_tools.embed_texts stubbed, no Ollama,
no real chat.db or analysis DB.

The optional-dep gate follows test_api.py: find_spec over the third-party
deps only, so an ImportError raised from inside mcp_server is a real failure,
not a silent skip."""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import sqlite_vec

import config
import ingest
import query_tools
import tool_schemas
from db import iso_to_day, load_vec_extension

_HAS_MCP_DEPS = all(importlib.util.find_spec(n) for n in ("mcp", "jsonschema"))

if _HAS_MCP_DEPS:
    from mcp import Client, MCPError, StdioServerParameters

    import mcp_server
else:                                                     # pragma: no cover
    Client = MCPError = StdioServerParameters = mcp_server = None

_NEEDS_MCP_DEPS = unittest.skipUnless(
    _HAS_MCP_DEPS, "MCP deps not installed (pip install -r requirements-mcp.txt)")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MCP_PLAN §3.1, in tools/list order.
EXPECTED_TOOLS = [
    "search_messages_sql", "search_messages_fts", "search_conversations",
    "search_messages_hybrid", "find_similar_conversations", "get_conversation",
    "get_message_context", "get_full_messages", "get_message_reactions",
    "search_contacts_by_name", "search_contact_by_handle", "list_chats",
    "get_chat_summary_stats", "get_activity_patterns", "get_response_dynamics",
    "get_reaction_stats", "get_word_trends", "get_word_frequency", "get_milestones",
    "get_group_dynamics", "get_chat_participant_stats",
]

# Out of scope: D1 plus the voice/conflict/nudge tools (several write, or rely
# on the DV screen / epilogue floors).
EXCLUDED_TOOLS = [
    "get_language_trajectory", "get_loose_threads", "resolve_loose_thread",
    "mute_contact_from_nudges", "get_voiceprint", "find_conflicts", "get_conflict",
    "diagnose_conflict",
]


def _run(coro):
    return asyncio.run(coro)


def _make_fixture_db(test):
    """Temp DB: contact 1 'Alice Example' (handle 1), contact 2 'Bob Example'
    (handle 2). Chat 1 is a 1:1 with Alice; chat 2 the group 'Test Group'.
    Chunks are built and each gets a 3-dim vector; returns (path, chunk_ids)."""
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    path = os.path.join(tmp.name, "fixture.db")
    con = sqlite3.connect(path)
    try:
        load_vec_extension(con)
        ingest.create_schema(con)
        con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, nickname, "
            "organization) VALUES (?, ?, ?, ?, NULL, NULL)",
            [(1, "u1", "Alice", "Example"), (2, "u2", "Bob", "Example")])
        con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, 'iMessage')",
            [(1, "+15550000001", 1), (2, "+15550000002", 2)])
        con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, 'iMessage')",
            [(1, "chat1", None, 0), (2, "chat2", "Test Group", 1)])
        con.executemany("INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
                        [(1, 1), (2, 1), (2, 2)])
        con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date, "
            "has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            [(1, "g1", 1, 1, 0, "planning the picnic", "2026-01-01T10:00:00"),
             (2, "g2", 1, None, 1, "I will bring lemonade", "2026-01-01T10:01:00"),
             (3, "g3", 2, 2, 0, "the park is reserved", "2026-02-01T09:00:00"),
             (4, "g4", 2, 1, 0, "great news", "2026-02-01T09:02:00")])
        with contextlib.redirect_stdout(io.StringIO()):  # builder progress line
            ingest.build_chunks(con)
        rows = con.execute(
            "SELECT id, chat_id, first_date, last_date FROM chunks ORDER BY id").fetchall()
        ingest.build_vec_table(con, 3)
        for i, (cid, chat_id, first, last) in enumerate(rows):
            vec = [1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.0]
            con.execute(
                "INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, last_day) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, sqlite_vec.serialize_float32(vec), chat_id,
                 iso_to_day(first), iso_to_day(last)))
        con.commit()
    finally:
        con.close()
    return path, [r[0] for r in rows]


def _payload(result):
    return json.loads(result.content[0].text)


@_NEEDS_MCP_DEPS
class ExposedSetTest(unittest.TestCase):
    def test_exposed_tools_are_exactly_the_plan_list_in_order(self):
        self.assertEqual(list(mcp_server.EXPOSED_TOOLS), EXPECTED_TOOLS)
        self.assertEqual([t.name for t in mcp_server.TOOLS], EXPECTED_TOOLS)

    def test_excluded_and_gated_tools_are_absent(self):
        exposed = set(mcp_server.EXPOSED_TOOLS)
        for name in EXCLUDED_TOOLS:
            self.assertNotIn(name, exposed)
        # GATED_TOOLS maps flag -> tuple of names; iterating the dict would
        # yield flag names and pass vacuously, so flatten the values.
        gated = [n for names in tool_schemas.GATED_TOOLS.values() for n in names]
        self.assertTrue(gated)
        self.assertFalse(exposed & set(gated))

    def test_tool_metadata_matches_schemas(self):
        defs = {d["name"]: d for d in tool_schemas.TOOL_DEFINITIONS}
        for tool in mcp_server.TOOLS:
            with self.subTest(tool=tool.name):
                self.assertEqual(tool.input_schema, defs[tool.name]["input_schema"])
                self.assertEqual(tool.description, defs[tool.name]["description"])
                self.assertTrue(tool.title)
                self.assertEqual(tool.title, mcp_server.EXPOSED_TOOLS[tool.name])
                a = tool.annotations
                self.assertIs(a.read_only_hint, True)
                self.assertIs(a.destructive_hint, False)
                self.assertIs(a.idempotent_hint, True)
                self.assertIs(a.open_world_hint, False)
        self.assertTrue(mcp_server.INSTRUCTIONS.strip())


@_NEEDS_MCP_DEPS
class InMemoryServerTest(unittest.TestCase):
    def setUp(self):
        self.db_path, self.chunk_ids = _make_fixture_db(self)
        for target, attr, value in (
                (query_tools, "DB_PATH", self.db_path),
                # The fixture vectors are 3-dim; the query embeds onto chunk 1's.
                (query_tools, "embed_texts",
                 lambda texts, kind="document": [[1.0, 0.0, 0.0]] * len(texts))):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, name, arguments=None):
        async def go():
            async with Client(mcp_server.server) as client:
                return await client.call_tool(name, arguments)
        return _run(go())

    def test_list_tools_returns_all_in_order(self):
        async def go():
            async with Client(mcp_server.server) as client:
                return await client.list_tools()
        self.assertEqual([t.name for t in _run(go()).tools], EXPECTED_TOOLS)

    def test_success_call_returns_parseable_json(self):
        result = self.call("search_messages_sql", {"chat_id": 1})
        self.assertFalse(result.is_error)
        body = _payload(result)
        self.assertEqual({r["id"] for r in body["rows"]}, {1, 2})

        result = self.call("list_chats", {})
        self.assertFalse(result.is_error)
        self.assertEqual(len(_payload(result)), 2)

    def test_missing_arguments_default_to_empty(self):
        result = self.call("list_chats")
        self.assertFalse(result.is_error)

    def test_bad_arguments_are_a_tool_error(self):
        result = self.call("search_messages_sql", {"limit": "lots"})
        self.assertTrue(result.is_error)
        self.assertIn("invalid arguments", _payload(result)["error"])

    def test_tool_exception_is_a_tool_error(self):
        def boom(**kwargs):
            raise RuntimeError("fixture failure")
        with mock.patch.dict(mcp_server._FUNCS, {"list_chats": boom}), \
                self.assertLogs(level="ERROR"):
            result = self.call("list_chats", {})
        self.assertTrue(result.is_error)
        self.assertIn("RuntimeError: fixture failure", _payload(result)["error"])

    def test_tool_error_dict_is_a_tool_error(self):
        result = self.call("get_group_dynamics", {"chat_id": 1})  # a 1:1 chat
        self.assertTrue(result.is_error)
        self.assertTrue(_payload(result)["error"])

    def test_unknown_tool_is_a_protocol_error(self):
        # Caught inside the client block: once it escapes, the client's task
        # group re-raises it wrapped in an ExceptionGroup.
        async def go():
            async with Client(mcp_server.server) as client:
                try:
                    await client.call_tool("resolve_loose_thread", {})
                except MCPError as e:
                    return e
        err = _run(go())
        self.assertIsInstance(err, MCPError)
        self.assertIn("Unknown tool", str(err))

    def test_semantic_tool_with_stubbed_embeddings(self):
        result = self.call("search_conversations", {"query": "picnic", "k": 2})
        self.assertFalse(result.is_error)
        rows = _payload(result)
        rows = rows["rows"] if isinstance(rows, dict) else rows
        self.assertEqual(rows[0]["chunk_id"], self.chunk_ids[0])

    def test_oversized_result_is_capped_valid_json(self):
        with mock.patch.object(config, "MCP_RESULT_MAX_CHARS", 300):
            result = self.call("search_messages_sql", {})
        self.assertFalse(result.is_error)
        self.assertLessEqual(len(result.content[0].text), 300)
        body = _payload(result)
        self.assertIs(body["truncated"], True)
        self.assertIn("note", body)

    def test_tiny_cap_returns_promptly_as_valid_json(self):
        # A cap under the fragment headroom once looped forever (-1 // 2 == -1),
        # freezing the event loop; run in a thread so a regression fails, not hangs.
        cases = [{"blob": "x" * 500}, [{"text": "y" * 500}]]  # non-list; one huge row
        out = []

        def go():
            for case in cases:
                out.append(mcp_server._result(case).content[0].text)
        with mock.patch.object(config, "MCP_RESULT_MAX_CHARS", 50):
            t = threading.Thread(target=go, daemon=True)
            t.start()
            t.join(timeout=10)
        self.assertFalse(t.is_alive(), "cap_tool_result_json hung on a tiny cap")
        self.assertEqual(len(out), len(cases))
        for text in out:
            self.assertIs(json.loads(text)["truncated"], True)

    def test_no_exposed_tool_emits_floor_output(self):
        # The turn-level floors (epilogue, wellbeing support block) can't be
        # enforced over MCP; no exposed tool may produce them.
        by_name = {"chunk_id": self.chunk_ids[0], "chat_id": 2}
        by_type = {"string": "picnic", "integer": 1, "array": [1]}
        for tool in mcp_server.TOOLS:
            schema = tool.input_schema
            args = {}
            for p in schema.get("required", []):
                typ = schema["properties"][p]["type"]
                args[p] = by_name.get(p, by_type[typ[0] if isinstance(typ, list) else typ])
            # Thin-fixture errors are fine here; keep their tracebacks out of the log.
            with self.subTest(tool=tool.name), \
                    mock.patch.object(mcp_server.log, "disabled", True):
                body = _payload(self.call(tool.name, args))
                rows = body.get("rows", []) if isinstance(body, dict) else body
                for d in [body, *rows]:
                    if isinstance(d, dict):
                        self.assertNotIn("_epilogue", d)
                        self.assertNotIn("support_block", d)


@_NEEDS_MCP_DEPS
class StdioEntrypointTest(unittest.TestCase):
    def _run_module(self, *args, env=None):
        return subprocess.run(
            [sys.executable, "-m", "mcp_server", *args], cwd=REPO_ROOT,
            env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=60)

    def test_help_exits_zero(self):
        proc = self._run_module("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_db_exits_nonzero_without_creating_it(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        missing = os.path.join(tmp.name, "nope.db")
        proc = self._run_module(env={"TARGET_DB": missing})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertFalse(os.path.exists(missing))

    def test_stdio_lists_and_calls(self):
        db_path, _ = _make_fixture_db(self)
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_server"],
            env={"TARGET_DB": db_path}, cwd=REPO_ROOT)

        async def go():
            async with Client(params) as client:
                tools = await client.list_tools()
                result = await client.call_tool("search_contacts_by_name", {"name": "Alice"})
                return tools, result
        tools, result = _run(go())
        self.assertEqual([t.name for t in tools.tools], EXPECTED_TOOLS)
        self.assertFalse(result.is_error)
        self.assertTrue(_payload(result))

    def test_raw_stdio_stdout_stays_json_rpc_through_exit(self):
        # A tool that print()s must not reach the wire, including the stdout
        # buffer flushed at exit; also pins the legacy (pre-session) handshake.
        db_path, _ = _make_fixture_db(self)
        code = (
            "import sys, mcp_server\n"
            "def noisy(**kw):\n"
            "    print('stray line from a tool')\n"
            "    return [{'ok': 1}]\n"
            "mcp_server._FUNCS['list_chats'] = noisy\n"
            "sys.exit(mcp_server.main([]))\n")
        proc = subprocess.Popen(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            env={**os.environ, "TARGET_DB": db_path}, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.kill)
        lines = []

        def reader():
            for line in proc.stdout:
                lines.append(line)
        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def send(msg):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        def wait_for(msg_id, timeout=30):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                for line in list(lines):
                    with contextlib.suppress(ValueError):
                        msg = json.loads(line)
                        if isinstance(msg, dict) and msg.get("id") == msg_id:
                            return msg
                time.sleep(0.05)
            self.fail(f"no response to id {msg_id}; stdout={lines!r}")

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "raw-test", "version": "0"}}})
        init = wait_for(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "list_chats", "arguments": {}}})
        call = wait_for(2)  # stdin stays open until here, else -32000 Connection closed
        proc.stdin.close()
        proc.wait(timeout=30)
        t.join(timeout=10)
        proc.stdout.close()
        proc.stderr.close()

        self.assertEqual(init["result"]["protocolVersion"], "2025-11-25")
        self.assertIn("result", call)
        self.assertFalse(call["result"].get("isError"))
        for line in lines:
            if line.strip():
                json.loads(line)  # anything else on stdout corrupts the wire


if __name__ == "__main__":
    unittest.main()
