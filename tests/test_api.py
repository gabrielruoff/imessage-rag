"""Hermetic smoke tests for the web API (api/).

No Ollama, no real chat.db: chat.run_turn_events is swapped for a scripted sync
generator and app.db points at a temp file. Covers the SSE frame sequence and
turn persistence, /api/health, and conversation CRUD.

Run from the repo root:  python3 -m unittest tests.test_api
"""

import os
import tempfile
import unittest

from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

import chat
from api import store
from api.main import app


def _fake_run_turn_events(messages, model=None):
    """A scripted stand-in for chat.run_turn_events: one tool call, two content
    deltas, then the terminal done. Mutates nothing (the engine's in-place
    mutation is irrelevant to the transport layer under test)."""
    yield {"type": "tool_call", "name": "search_messages_fts",
           "arguments": {"query": "hi"}, "count": 3, "tokens_est": 12}
    yield {"type": "token", "text": "Hello "}
    yield {"type": "token", "text": "world"}
    yield {"type": "done", "reply": "Hello world"}


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        # Point app.db at a throwaway temp file and (re)create the schema.
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._saved_db = store.DB_PATH
        store.DB_PATH = self._db_path
        store.init_db()
        self.addCleanup(self._teardown_db)
        # sse-starlette caches a module-global exit Event on first use, binding
        # it to that event loop; each TestClient spins a fresh loop, so reset it
        # or the second streaming test hits "bound to a different event loop".
        AppStatus.should_exit_event = None
        self.client = TestClient(app)

    def _teardown_db(self):
        store.DB_PATH = self._saved_db
        try:
            os.unlink(self._db_path)
        except OSError:
            pass


class ChatStreamTest(ApiTestBase):
    def setUp(self):
        super().setUp()
        self._saved_events = chat.run_turn_events
        chat.run_turn_events = _fake_run_turn_events
        self.addCleanup(setattr, chat, "run_turn_events", self._saved_events)

    def test_chat_streams_frames_and_persists(self):
        resp = self.client.post("/api/chat", json={"conversation_id": None,
                                                    "message": "hi there"})
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        # The named SSE frames the frontend keys off of.
        self.assertIn("event: meta", body)
        self.assertIn("event: tool_call", body)
        self.assertIn("event: token", body)
        self.assertIn("event: done", body)
        # meta carries the new conversation id.
        self.assertIn("conversation_id", body)

        # A conversation was created and both turns persisted.
        convs = store.list_conversations()
        self.assertEqual(len(convs), 1)
        conv_id = convs[0]["id"]
        turns = store.get_turns(conv_id)
        roles = [t["role"] for t in turns]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(turns[0]["content"], "hi there")
        self.assertEqual(turns[1]["content"], "Hello world")

    def test_chat_continues_existing_conversation(self):
        conv = store.create_conversation("seed")
        resp = self.client.post("/api/chat", json={"conversation_id": conv["id"],
                                                    "message": "follow up"})
        self.assertEqual(resp.status_code, 200)
        turns = store.get_turns(conv["id"])
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])


class HealthTest(ApiTestBase):
    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(set(data), {"ok", "backend", "model", "reachable"})
        self.assertTrue(data["ok"])
        self.assertIn(data["backend"], ("apple", "ollama"))
        self.assertIsInstance(data["reachable"], bool)


class ConversationCrudTest(ApiTestBase):
    def test_create_list_get_delete(self):
        # Create.
        created = self.client.post("/api/conversations", json={"title": "my chat"})
        self.assertEqual(created.status_code, 200)
        conv_id = created.json()["id"]
        self.assertEqual(created.json()["title"], "my chat")
        self.assertEqual(created.json()["turn_count"], 0)

        # Seed a turn so turn_count is exercised.
        store.add_turn(conv_id, "user", "hello")

        # List.
        listed = self.client.get("/api/conversations")
        self.assertEqual(listed.status_code, 200)
        rows = listed.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], conv_id)
        self.assertEqual(rows[0]["turn_count"], 1)

        # Get (with turns).
        got = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(len(got.json()["turns"]), 1)
        self.assertEqual(got.json()["turns"][0]["content"], "hello")

        # Get missing -> 404.
        self.assertEqual(self.client.get("/api/conversations/9999").status_code, 404)

        # Delete -> 204, then gone.
        deleted = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/api/conversations/{conv_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
