"""Hermetic tests for annotate.py (batch chunk annotation + --terms
curation). Temp SQLite only; ollama.chat_once is mocked with canned JSON
message dicts -- no Ollama, no real DB."""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import annotate
import ollama
import schema
from config import ANNOTATE_MODEL
from db import get_state, load_vec_extension, set_state


def _reply(payload):
    """A canned chat_once return: Ollama message dict whose content is JSON."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {"role": "assistant", "content": content}


class AnnotateFixture(unittest.TestCase):
    """Temp DB with one chat, three messages, one chunk covering them.

    Message texts: 1 "I got the job at the lab today" (unique excerpt
    target), 2 "ok", 3 "ok cool" -- so the excerpt "ok" is ambiguous
    (substring of both 2 and 3)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "test.db")
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        load_vec_extension(self.con)
        schema.create_schema(self.con)

        self.con.execute(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (1, 'chat1', NULL, 0, 'iMessage')")
        msgs = [
            (1, "I got the job at the lab today", "2026-01-01T10:00:00"),
            (2, "ok", "2026-01-01T10:01:00"),
            (3, "ok cool", "2026-01-01T10:02:00"),
        ]
        self.con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, chunk_id) VALUES (?, ?, 1, 1, 0, ?, ?, 1)",
            [(mid, f"guid-{mid}", text, date) for mid, text, date in msgs],
        )
        self.con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (1, 1, 1, 3, '2026-01-01T10:00:00', '2026-01-01T10:02:00', 3, "
            "'Conversation with Alice on 2026-01-01:\n"
            "Alice: I got the job at the lab today\nMe: ok\nAlice: ok cool')")
        self.con.commit()

        patcher = mock.patch.object(annotate, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def annotation(self, chunk_id=1):
        return self.con.execute(
            "SELECT valence, emotions_json, depth, events_json "
            "FROM chunk_annotations WHERE chunk_id = ?", (chunk_id,)).fetchone()

    def n_annotations(self):
        return self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]


class TestAnnotateChunks(AnnotateFixture):
    def test_happy_path_clamps_and_persists(self):
        reply = _reply({
            "valence": 3,                                     # clamped -> 1.0
            "emotions": ["joy", "zeal", "love", "anger"],     # bogus dropped, max 3
            "depth": 9,                                       # clamped -> 5
            "events": [],
        })
        with mock.patch.object(ollama, "chat_once", return_value=reply) as fake:
            annotate.main(["--all", "--quality", "max"])
        self.assertEqual(fake.call_count, 1)
        valence, emotions_json, depth, events_json = self.annotation()
        self.assertEqual(valence, 1.0)
        self.assertEqual(json.loads(emotions_json), ["joy", "love", "anger"])
        self.assertEqual(depth, 5)
        self.assertEqual(json.loads(events_json), [])
        # model recorded for the mixing guard
        self.assertEqual(get_state(self.con, "annotate_model"), ANNOTATE_MODEL)

    def test_event_excerpt_resolves_message_id(self):
        reply = _reply({
            "valence": 0.5, "emotions": ["joy"], "depth": 3,
            "events": [
                {"kind": "good_news", "excerpt": "I got the job",
                 "response_class": "active_constructive"},       # unique -> msg 1
                {"kind": "gratitude", "excerpt": "ok",
                 "response_class": "enthusiastic"},               # ambiguous + bad class
                {"kind": "bad_kind", "excerpt": "whatever",
                 "response_class": None},                         # invalid kind: dropped
            ],
        })
        with mock.patch.object(ollama, "chat_once", return_value=reply):
            annotate.main(["--all", "--quality", "max"])
        events = json.loads(self.annotation()[3])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["kind"], "good_news")
        self.assertEqual(events[0]["message_id"], 1)
        self.assertEqual(events[0]["response_class"], "active_constructive")
        self.assertIsNone(events[1]["message_id"])          # matched msgs 2 and 3
        self.assertIsNone(events[1]["response_class"])      # invalid class -> null

    def test_malformed_json_retries_once_then_skips(self):
        bad = _reply("this is not json {")
        with mock.patch.object(ollama, "chat_once", return_value=bad) as fake:
            annotate.main(["--all", "--quality", "max"])
        self.assertEqual(fake.call_count, 2)                # one retry, no more
        self.assertEqual(self.n_annotations(), 0)           # skipped: retried next run
        # the retry carried a corrective nudge
        retry_messages = fake.call_args_list[1][0][0]
        self.assertIn("only the JSON object", retry_messages[-1]["content"])

    def test_resumable_second_run_skips_annotated(self):
        reply = _reply({"valence": 0, "emotions": [], "depth": 1, "events": []})
        with mock.patch.object(ollama, "chat_once", return_value=reply):
            annotate.main(["--all", "--quality", "max"])
        self.assertEqual(self.n_annotations(), 1)
        with mock.patch.object(ollama, "chat_once") as fake:
            annotate.main(["--all", "--quality", "max"])
        fake.assert_not_called()

    def test_model_guard_exits_on_mismatch_with_rows(self):
        set_state(self.con, "annotate_model", "some-other-model")
        self.con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json) VALUES (99, 0, '[]', 1, '[]')")
        self.con.commit()
        with mock.patch.object(ollama, "chat_once") as fake:
            with self.assertRaises(SystemExit):
                annotate.main(["--all", "--quality", "max"])
        fake.assert_not_called()

    def test_model_guard_allows_mismatch_when_table_empty(self):
        set_state(self.con, "annotate_model", "some-other-model")
        self.con.commit()
        reply = _reply({"valence": 0, "emotions": [], "depth": 1, "events": []})
        with mock.patch.object(ollama, "chat_once", return_value=reply):
            annotate.main(["--all", "--quality", "max"])
        self.assertEqual(self.n_annotations(), 1)
        self.assertEqual(get_state(self.con, "annotate_model"), ANNOTATE_MODEL)

    def test_dry_run_writes_nothing(self):
        with mock.patch.object(ollama, "chat_once") as fake:
            annotate.main(["--all", "--quality", "max", "--dry-run"])
        fake.assert_not_called()
        self.assertEqual(self.n_annotations(), 0)
        self.assertIsNone(get_state(self.con, "annotate_model"))


class TestCurateTerms(AnnotateFixture):
    def setUp(self):
        super().setUp()
        self.con.executemany(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, first_message_id, "
            "coined_by_handle_id, coined_is_me, llm_verdict) "
            "VALUES (1, ?, 10.0, 5, 1, NULL, 1, NULL)",
            [("blorbo",), ("zippy",)],
        )
        self.con.commit()

    def verdict(self, term):
        return self.con.execute(
            "SELECT llm_verdict FROM dyad_terms WHERE chat_id = 1 AND term = ?",
            (term,)).fetchone()[0]

    def test_verdicts_validated_and_persisted(self):
        # terms processed in ORDER BY chat_id, term: blorbo then zippy
        replies = [_reply({"verdict": "inside_joke"}),
                   _reply({"verdict": "hilarious"})]     # not in the enum
        with mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            annotate.main(["--terms"])
        self.assertEqual(fake.call_count, 2)
        self.assertEqual(self.verdict("blorbo"), "inside_joke")
        self.assertIsNone(self.verdict("zippy"))         # invalid verdict skipped

    def test_terms_resumable_skips_curated(self):
        self.con.execute(
            "UPDATE dyad_terms SET llm_verdict = 'ordinary' WHERE term = 'blorbo'")
        self.con.commit()
        with mock.patch.object(
                ollama, "chat_once",
                return_value=_reply({"verdict": "noise"})) as fake:
            annotate.main(["--terms"])
        self.assertEqual(fake.call_count, 1)             # only zippy pending
        self.assertEqual(self.verdict("blorbo"), "ordinary")
        self.assertEqual(self.verdict("zippy"), "noise")

    def test_terms_dry_run_writes_nothing(self):
        with mock.patch.object(ollama, "chat_once") as fake:
            annotate.main(["--terms", "--dry-run"])
        fake.assert_not_called()
        self.assertIsNone(self.verdict("blorbo"))
        self.assertIsNone(self.verdict("zippy"))

    def test_terms_model_guard_exits_on_mismatch_with_existing_verdicts(self):
        # A verdict is expensive LLM output that survives dyad_terms rebuilds,
        # so it gets the same anti-mixing guard as chunk_annotations.
        set_state(self.con, "annotate_model", "some-other-model")
        self.con.execute(
            "UPDATE dyad_terms SET llm_verdict = 'ordinary' WHERE term = 'blorbo'")
        self.con.commit()
        with mock.patch.object(ollama, "chat_once") as fake:
            with self.assertRaises(SystemExit):
                annotate.main(["--terms"])
        fake.assert_not_called()

    def test_terms_model_guard_allows_mismatch_when_no_verdicts_yet(self):
        # Uncurated dyad_terms rows carry no model output, so they must not
        # trip the guard -- only llm_verdict rows count.
        set_state(self.con, "annotate_model", "some-other-model")
        self.con.commit()
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"verdict": "nickname"})):
            annotate.main(["--terms"])
        self.assertEqual(self.verdict("blorbo"), "nickname")


if __name__ == "__main__":
    unittest.main()
