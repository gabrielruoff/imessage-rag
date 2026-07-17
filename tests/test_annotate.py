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
from config import ANNOTATE_MODEL, ANNOTATE_TERMS_MAX_FAILURES
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
        # Both terms share a chat, so one batched request covers them, keyed
        # by position in ORDER BY chat_id, term: blorbo=0, zippy=1. zippy's
        # verdict is not in the enum, so it falls back to a single-term
        # request -- which returns another bogus verdict and is skipped.
        replies = [_reply({"0": "inside_joke", "1": "hilarious"}),
                   _reply({"verdict": "hilarious"})]
        with mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            annotate.main(["--terms"])
        self.assertEqual(fake.call_count, 2)
        self.assertEqual(self.verdict("blorbo"), "inside_joke")
        self.assertIsNone(self.verdict("zippy"))         # invalid verdict skipped

    def test_terms_batched_into_one_request(self):
        # The point of batching: two pending terms, one LLM request.
        with mock.patch.object(
                ollama, "chat_once",
                return_value=_reply({"0": "inside_joke", "1": "noise"})) as fake:
            annotate.main(["--terms"])
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(self.verdict("blorbo"), "inside_joke")
        self.assertEqual(self.verdict("zippy"), "noise")

    def test_terms_batch_never_spans_chats(self):
        # A batch shares one system prompt saying "in one text-message chat",
        # so terms from different chats must not ride in the same request.
        self.con.execute(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, "
            "first_message_id, coined_by_handle_id, coined_is_me, llm_verdict) "
            "VALUES (2, 'wug', 10.0, 5, 1, NULL, 1, NULL)")
        self.con.commit()
        with mock.patch.object(
                ollama, "chat_once",
                return_value=_reply({"0": "noise", "1": "noise"})) as fake:
            annotate.main(["--terms"])
        # chat 1 (blorbo, zippy) batched; chat 2 (wug) is its own singleton.
        self.assertEqual(fake.call_count, 2)
        # ...and the split is by chat, not just two requests of any shape:
        # each call's user message carries only its own chat's terms.
        prompts = [call[0][0][-1]["content"] for call in fake.call_args_list]
        self.assertIn("blorbo", prompts[0])
        self.assertIn("zippy", prompts[0])
        self.assertNotIn("wug", prompts[0])
        self.assertIn("wug", prompts[1])
        self.assertNotIn("blorbo", prompts[1])

    def test_retry_failed_dry_run_leaves_parking_intact(self):
        # --dry-run must not unpark: the preview would spend the very state
        # it is previewing, and the next bare run would re-attempt the terms
        # at full price -- the loop curate_failures exists to close.
        self.con.execute(
            "UPDATE dyad_terms SET curate_failures = ? WHERE term = 'blorbo'",
            (ANNOTATE_TERMS_MAX_FAILURES,))
        self.con.commit()
        with mock.patch.object(ollama, "chat_once") as fake:
            annotate.main(["--terms", "--retry-failed", "--dry-run"])
        fake.assert_not_called()
        (failures,) = self.con.execute(
            "SELECT curate_failures FROM dyad_terms WHERE term = 'blorbo'"
        ).fetchone()
        self.assertEqual(failures, ANNOTATE_TERMS_MAX_FAILURES)

    def test_retry_failed_dry_run_previews_the_unparked_scope(self):
        # The preview must show what the real run would curate -- which
        # includes the parked terms it would have unparked.
        self.con.execute(
            "UPDATE dyad_terms SET curate_failures = ? WHERE term = 'blorbo'",
            (ANNOTATE_TERMS_MAX_FAILURES,))
        self.con.commit()
        with mock.patch.object(ollama, "chat_once"):
            with mock.patch("builtins.print") as out:
                annotate.main(["--terms", "--retry-failed", "--dry-run"])
        printed = "\n".join(str(c[0][0]) for c in out.call_args_list if c[0])
        self.assertIn("1:blorbo", printed)
        self.assertIn("2 dyad terms pending", printed)

    def test_run_that_never_succeeds_aborts_without_parking(self):
        # A model answering nothing is a config failure, not N bad terms:
        # without this, one gpt-oss/think=false mismatch parks the corpus.
        terms = [(f"term{i}",) for i in range(annotate._TERMS_ABORT_AFTER + 5)]
        self.con.executemany(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, "
            "first_message_id, coined_by_handle_id, coined_is_me, llm_verdict) "
            "VALUES (1, ?, 10.0, 5, 1, NULL, 1, NULL)", terms)
        self.con.commit()
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"verdict": "nonsense"})):
            with self.assertRaises(SystemExit):
                annotate.main(["--terms"])
        # Rolled back: nothing parked, so a fixed config just reruns.
        (charged,) = self.con.execute(
            "SELECT COUNT(*) FROM dyad_terms WHERE curate_failures > 0").fetchone()
        self.assertEqual(charged, 0)

    def test_abandoned_terms_are_not_charged_a_failure(self):
        # A batch that comes back unusable falls back to one request per term;
        # once _GROUP_GIVEUP of those have failed in a row the rest of the
        # group is left alone -- never asked, so never charged.
        terms = [(f"term{i}",) for i in range(annotate._GROUP_GIVEUP + 5)]
        self.con.executemany(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, "
            "first_message_id, coined_by_handle_id, coined_is_me, llm_verdict) "
            "VALUES (1, ?, 10.0, 5, 1, NULL, 1, NULL)", terms)
        self.con.commit()
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"verdict": "nonsense"})):
            annotate.main(["--terms"])
        # Exactly _GROUP_GIVEUP terms got their own request and failed; the
        # rest were abandoned at 0 and stay pending for the next run.
        (charged,) = self.con.execute(
            "SELECT COUNT(*) FROM dyad_terms WHERE curate_failures > 0").fetchone()
        self.assertEqual(charged, annotate._GROUP_GIVEUP)

    def test_terms_think_defaults_off_and_ignores_quality(self):
        # Reasoning about a one-word verdict cost ~11x for an identical
        # answer, so --terms sends think=False rather than a --quality tier's.
        with mock.patch.object(
                ollama, "chat_once",
                return_value=_reply({"0": "noise", "1": "noise"})) as fake:
            annotate.main(["--terms"])
        self.assertIs(fake.call_args.kwargs["think"], False)

    def test_terms_think_omit_reaches_the_flagless_state(self):
        # None ("omit the flag") is a distinct state from False ("send
        # think: false") and is the right setting for some models, so the env
        # var needs a spelling for it -- otherwise 'none' would forward the
        # literal string "none" to Ollama.
        with mock.patch.object(annotate, "ANNOTATE_TERMS_THINK", "omit"):
            with mock.patch.object(
                    ollama, "chat_once",
                    return_value=_reply({"0": "noise", "1": "noise"})) as fake:
                annotate.main(["--terms"])
        self.assertIsNone(fake.call_args.kwargs["think"])

    def test_quality_rejected_with_terms(self):
        # --quality is a chunk tier; silently accepting it here is what let
        # its `think` leak into term curation and cost ~11x per term.
        with mock.patch.object(ollama, "chat_once") as fake:
            with self.assertRaises(SystemExit):
                annotate.main(["--terms", "--quality", "max"])
        fake.assert_not_called()

    def test_failed_terms_are_parked_after_max_failures(self):
        # An unparseable reply burns two requests (chat_once_json retries
        # once); without a failure counter the same term would do that on
        # every future run forever.
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply("not json at all")):
            for _ in range(ANNOTATE_TERMS_MAX_FAILURES):
                annotate.main(["--terms"])
        (failures,) = self.con.execute(
            "SELECT curate_failures FROM dyad_terms WHERE term = 'blorbo'"
        ).fetchone()
        self.assertEqual(failures, ANNOTATE_TERMS_MAX_FAILURES)
        # Parked now: a further run must not call the model at all.
        with mock.patch.object(ollama, "chat_once") as fake:
            annotate.main(["--terms"])
        fake.assert_not_called()

    def test_retry_failed_unparks_terms(self):
        self.con.execute(
            "UPDATE dyad_terms SET curate_failures = ? WHERE term = 'blorbo'",
            (ANNOTATE_TERMS_MAX_FAILURES,))
        self.con.execute("UPDATE dyad_terms SET llm_verdict = 'noise' "
                         "WHERE term = 'zippy'")
        self.con.commit()
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"verdict": "nickname"})):
            annotate.main(["--terms", "--retry-failed"])
        self.assertEqual(self.verdict("blorbo"), "nickname")
        (failures,) = self.con.execute(
            "SELECT curate_failures FROM dyad_terms WHERE term = 'blorbo'"
        ).fetchone()
        self.assertEqual(failures, 0)

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
