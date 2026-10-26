"""Hermetic tests for annotate.py (batch chunk annotation + --terms
curation). Temp SQLite only; ollama.chat_once is mocked with canned JSON
message dicts -- no Ollama, no real DB."""

import json
import os
import re
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


# ---------------------------------------------------------------------------
# The schema-constrained retry rung
# ---------------------------------------------------------------------------

_ANNOTATION = {"valence": 0.5, "emotions": ["joy"], "depth": 2, "events": []}
_CHUNK_ID_RE = re.compile(r"chunk (\d+)")


def _chunk_ids_in(content):
    """Chunk ids a request prompt names, in order, deduplicated. The group
    texts below name their own id, so a singleton request -- which carries the
    bare chunk text with no '### chunk <id>' header -- is readable too."""
    out = []
    for match in _CHUNK_ID_RE.findall(content):
        chunk_id = int(match)
        if chunk_id not in out:
            out.append(chunk_id)
    return out


class TestAnnotationSchema(unittest.TestCase):
    """annotate._annotation_schema: the grammar the retry spends its one
    request on. Coverage of the named ids is guaranteed by CONSTRUCTION here,
    which is the entire reason the rung exists -- and the enums come from the
    validator's own constants, so a schema and _validate_annotation cannot
    drift into disagreeing about what a valid label is."""

    IDS = [7, 12, 3]

    def setUp(self):
        self.schema = annotate._annotation_schema(self.IDS)

    def test_every_id_is_a_property_and_required(self):
        keys = [str(chunk_id) for chunk_id in self.IDS]
        self.assertEqual(self.schema["type"], "object")
        self.assertEqual(list(self.schema["properties"]), keys)
        self.assertEqual(self.schema["required"], keys)

    def test_the_enums_are_the_module_s_own_constants(self):
        obj = self.schema["properties"]["7"]
        self.assertEqual(obj["properties"]["emotions"]["items"]["enum"],
                         list(annotate.EMOTION_LABELS))
        event = obj["properties"]["events"]["items"]
        self.assertEqual(event["properties"]["kind"]["enum"],
                         list(annotate.EVENT_KINDS))
        response_class = event["properties"]["response_class"]["anyOf"]
        self.assertEqual(response_class[0]["enum"], list(annotate.RESPONSE_CLASSES))
        # ...and null stays reachable: "there was no response" is a real answer,
        # not a missing one.
        self.assertIn({"type": "null"}, response_class)

    def test_the_bounds_are_the_ones_the_validator_clamps_to(self):
        obj = self.schema["properties"]["7"]
        self.assertEqual(obj["properties"]["emotions"]["maxItems"], 3)
        # events is the one array whose items are objects with a free-form
        # string in them, i.e. the only field a grammar can run away on -- and
        # a runaway costs ANNOTATE_TIMEOUT, which drops the group to singles.
        # Its ceiling has no measured basis and nothing downstream reads it;
        # what matters is that there IS one.
        self.assertIn("maxItems", obj["properties"]["events"])
        depth = obj["properties"]["depth"]
        self.assertEqual((depth["type"], depth["minimum"], depth["maximum"]),
                         ("integer", 1, 5))
        self.assertEqual(sorted(obj["required"]),
                         ["depth", "emotions", "events", "valence"])

    def test_it_is_json_serializable(self):
        # It goes into an Ollama request body, so anything json.dumps cannot
        # take (a tuple enum, say) would fail at request time -- on a worker
        # thread, inside the retry's blanket except, where it reads as "the
        # model covered nothing" and quietly costs a request every group.
        self.assertEqual(json.loads(json.dumps(self.schema)), self.schema)

    def test_each_id_gets_its_own_object(self):
        first, second = (self.schema["properties"][key] for key in ("7", "12"))
        self.assertEqual(first, second)
        self.assertIsNot(first, second)         # no aliased mutable dict


_COVER_ASKED = object()


class _LadderLLM:
    """Fake for both ollama seams _annotate_group uses, with ONE ordered log so
    a test can assert the ladder's SHAPE rather than a pair of call counts.

    chat_once_json serves the free-JSON rungs (the batched request and the
    per-chunk singles); chat_once serves the schema-constrained retry and is
    the only place a dict `format` may ever appear. `schema` is what that retry
    gets back: _COVER_ASKED (a well-formed reply covering every id it asked
    for), an Exception to raise, or a canned message dict. `batch` is an
    Exception the batched request raises (None: answer per `covered`), and
    `singles` lists per-single outcomes in call order -- an Exception to raise
    or a value to return; once exhausted, singles return an annotation."""

    def __init__(self, covered=(), schema=_COVER_ASKED, batch=None, singles=()):
        self.covered = set(covered)
        self.schema = schema
        self.batch = batch
        self.singles = list(singles)
        self.log = []                           # (rung, ids) in call order
        self.formats = []                       # every `format` chat_once saw
        self.thinks = []                        # every `think` chat_once saw

    def chat_once_json(self, messages, model=None, think=None, timeout=None,
                       **kwargs):
        ids = _chunk_ids_in(messages[-1]["content"])
        if len(ids) > 1:
            self.log.append(("batch", ids))
            if self.batch is not None:
                raise self.batch
            return {str(chunk_id): dict(_ANNOTATION)
                    for chunk_id in ids if chunk_id in self.covered}
        self.log.append(("single", ids))
        if self.singles:
            outcome = self.singles.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return dict(_ANNOTATION)

    def chat_once(self, messages, model=None, format=None, timeout=None,
                  think=None):
        ids = _chunk_ids_in(messages[-1]["content"])
        self.log.append(("schema", ids))
        self.formats.append(format)
        self.thinks.append(think)
        if isinstance(self.schema, BaseException):
            raise self.schema
        if self.schema is _COVER_ASKED:
            return _reply({str(chunk_id): dict(_ANNOTATION) for chunk_id in ids})
        return self.schema


class TestAnnotateGroupLadder(unittest.TestCase):
    """annotate._annotate_group's batch -> schema retry -> singles ladder,
    ordered by what each rung costs when it works."""

    GROUP = [(1, "chunk 1: alpha"), (2, "chunk 2: beta"), (3, "chunk 3: gamma")]

    def run_group(self, llm, group=None, model="test-annotate-model",
                  fail_fast=False):
        with mock.patch.object(ollama, "chat_once_json", llm.chat_once_json):
            with mock.patch.object(ollama, "chat_once", llm.chat_once):
                return annotate._annotate_group(
                    self.GROUP if group is None else group, model, False,
                    fail_fast=fail_fast)

    def assertCovered(self, out):
        self.assertEqual(sorted(out), [1, 2, 3])
        self.assertTrue(all(isinstance(data, dict) for data in out.values()))

    def test_a_fully_covered_batch_is_exactly_one_request(self):
        # The fast path stays the fast path: free-JSON with full coverage is
        # the cheapest thing available, so nothing else may go out behind it.
        llm = _LadderLLM(covered=(1, 2, 3))
        out = self.run_group(llm)
        self.assertEqual(llm.log, [("batch", [1, 2, 3])])
        self.assertCovered(out)

    def test_a_miss_buys_one_schema_request_and_no_singles(self):
        # The whole point: two missed ids used to be two more requests.
        llm = _LadderLLM(covered=(1,))
        out = self.run_group(llm)
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("schema", [2, 3])])
        self.assertCovered(out)
        # ...constrained on exactly the missed ids, so the grammar stays small.
        self.assertEqual(llm.formats, [annotate._annotation_schema([2, 3])])

    def test_a_failed_schema_request_falls_through_to_singles(self):
        # Expected ~never, but the group must still complete: the per-chunk
        # rung stays underneath as belt and braces.
        for label, schema_reply in (
                ("server error body", RuntimeError("boom")),
                ("unparseable content", _reply("not json {")),
                ("empty reply", {}),
                ("not a JSON object", _reply("[1, 2]")),
                ("wrong keys", _reply({"chunk_2": _ANNOTATION}))):
            with self.subTest(schema=label):
                llm = _LadderLLM(covered=(1,), schema=schema_reply)
                out = self.run_group(llm)
                self.assertEqual(llm.log, [("batch", [1, 2, 3]),
                                           ("schema", [2, 3]),
                                           ("single", [2]), ("single", [3])])
                self.assertCovered(out)
                # An unusable reply is not a transport failure.
                self.assertFalse(out.failed)

    def test_a_batch_timeout_skips_the_schema_rung_and_buys_singles(self):
        llm = _LadderLLM(batch=TimeoutError())
        out = self.run_group(llm)
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("single", [1]),
                                   ("single", [2]), ("single", [3])])
        self.assertCovered(out)
        self.assertTrue(out.failed)

    def test_a_second_transport_failure_ends_the_ladder(self):
        llm = _LadderLLM(batch=TimeoutError(), singles=[TimeoutError()])
        out = self.run_group(llm)
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("single", [1])])
        self.assertEqual(out, {1: None, 2: None, 3: None})
        self.assertTrue(out.failed)

    def test_fail_fast_stops_at_the_first_transport_failure(self):
        llm = _LadderLLM(batch=TimeoutError())
        out = self.run_group(llm, fail_fast=True)
        self.assertEqual(llm.log, [("batch", [1, 2, 3])])
        self.assertEqual(out, {1: None, 2: None, 3: None})
        self.assertTrue(out.failed)

    def test_fail_fast_keeps_rows_an_earlier_rung_covered(self):
        llm = _LadderLLM(covered=(1,), schema=TimeoutError())
        out = self.run_group(llm, fail_fast=True)
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("schema", [2, 3])])
        self.assertIsInstance(out[1], dict)
        self.assertEqual((out[2], out[3]), (None, None))
        self.assertTrue(out.failed)

    def test_a_refused_connection_propagates(self):
        # No server running is a config error: fail loudly, never skip rows.
        for fail_fast in (False, True):
            with self.subTest(fail_fast=fail_fast):
                llm = _LadderLLM(batch=ConnectionRefusedError())
                with self.assertRaises(ConnectionRefusedError):
                    self.run_group(llm, fail_fast=fail_fast)
                self.assertEqual(llm.log, [("batch", [1, 2, 3])])

    def test_a_schema_timeout_flags_the_group_and_singles_still_run(self):
        llm = _LadderLLM(covered=(1,), schema=TimeoutError())
        out = self.run_group(llm)
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("schema", [2, 3]),
                                   ("single", [2]), ("single", [3])])
        self.assertCovered(out)
        self.assertTrue(out.failed)

    def test_an_error_body_on_a_single_still_propagates(self):
        # A config error (e.g. a model that is not pulled) must stay loud.
        llm = _LadderLLM(covered=(1,), schema=_reply("not json {"),
                         singles=[RuntimeError("model not found")])
        with self.assertRaises(RuntimeError):
            self.run_group(llm)

    def test_only_a_transport_failure_makes_a_group_unbankable(self):
        malformed = self.run_group(_LadderLLM(schema=_reply("not json {"),
                                              singles=[None, None, None]))
        self.assertEqual(malformed, {1: None, 2: None, 3: None})
        self.assertTrue(annotate._group_bankable(malformed))
        failed = self.run_group(_LadderLLM(batch=TimeoutError()), fail_fast=True)
        self.assertFalse(annotate._group_bankable(failed))

    def test_ctrl_c_in_the_schema_request_still_propagates(self):
        # The blanket except is `Exception`, deliberately: swallowing a
        # KeyboardInterrupt here would cost the tier its Ctrl-C safety.
        llm = _LadderLLM(covered=(1,), schema=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.run_group(llm)

    def test_apple_never_receives_a_dict_format(self):
        # applefm's `format` seam is not a guided schema (an empty-properties
        # DynamicGenerationSchema pins the reply to {}), so a dict format is
        # undefined there. Apple keeps batch -> singles.
        llm = _LadderLLM(covered=(1,))
        out = self.run_group(llm, model="apple")
        self.assertEqual(llm.log, [("batch", [1, 2, 3]),
                                   ("single", [2]), ("single", [3])])
        self.assertEqual(llm.formats, [])
        self.assertCovered(out)

    def test_the_schema_retry_honours_the_think_degradation(self):
        # chat_once_json owns the process-wide degradation (a model that
        # rejects `think`, or that answers think=false with EMPTY content --
        # gpt-oss's signature), and the retry is the one annotation request
        # that goes to chat_once directly. Re-sending the rejected value would
        # fail EVERY schema retry for the rest of the run, silently, so the
        # rung that exists to remove a request per miss would add one instead.
        llm = _LadderLLM(covered=(1,))
        with mock.patch.object(ollama, "_think_unsupported", True):
            out = self.run_group(llm)
        self.assertEqual(llm.thinks, [None])            # flag omitted...
        # ...and it is still a schema request, not a degraded free-JSON one.
        self.assertEqual(llm.formats, [annotate._annotation_schema([2, 3])])
        self.assertEqual(llm.log, [("batch", [1, 2, 3]), ("schema", [2, 3])])
        self.assertCovered(out)

    def test_an_undegraded_run_forwards_think_verbatim(self):
        # The tri-state is not collapsed on the way through: False means "send
        # think: false", which is NOT the same request as omitting the flag,
        # and hybrid's preset is exactly False.
        llm = _LadderLLM(covered=(1,))
        with mock.patch.object(ollama, "_think_unsupported", False):
            self.run_group(llm)
        self.assertEqual(llm.thinks, [False])

    def test_a_singleton_group_never_reaches_the_schema_rung(self):
        # It has had no batched request at all, so its "miss" is just its first
        # request and belongs on the single-chunk path.
        llm = _LadderLLM()
        out = self.run_group(llm, group=[(1, "chunk 1: alpha")])
        self.assertEqual(llm.log, [("single", [1])])
        self.assertEqual(sorted(out), [1])


# ---------------------------------------------------------------------------
# Request packing
# ---------------------------------------------------------------------------

class TestGroupPacking(unittest.TestCase):
    """annotate._char_budget_groups / _annotation_groups. Both backends pack by
    a char budget; the row cap and the reason are what differ."""

    def rows(self, *sizes):
        return [(i, "x" * n) for i, n in enumerate(sizes, start=1)]

    def ids(self, groups):
        return [[chunk_id for chunk_id, _text in group] for group in groups]

    def test_the_row_cap_and_the_char_budget_both_bind(self):
        rows = self.rows(*[10] * 6)
        self.assertEqual(
            self.ids(annotate._char_budget_groups(rows, 1000, max_rows=4)),
            [[1, 2, 3, 4], [5, 6]])                     # rows bind
        self.assertEqual(
            self.ids(annotate._char_budget_groups(rows, 25, max_rows=4)),
            [[1, 2], [3, 4], [5, 6]])                   # chars bind first

    def test_a_group_always_has_a_row_and_an_oversized_chunk_still_ships(self):
        rows = self.rows(10, 50, 10)
        groups = annotate._char_budget_groups(rows, 25, max_rows=4)
        self.assertEqual(self.ids(groups), [[1], [2], [3]])
        self.assertTrue(all(groups))                    # never an empty group

    def test_max_rows_none_is_the_char_only_packing_apple_has_always_had(self):
        rows = self.rows(*[10] * 6)
        self.assertEqual(self.ids(annotate._char_budget_groups(rows, 1000)),
                         [[1, 2, 3, 4, 5, 6]])          # no row cap at all
        self.assertEqual(
            annotate._annotation_groups(rows, "apple", 1),
            annotate._char_budget_groups(rows, annotate._APPLE_ANNOTATE_CHAR_BUDGET))

    def test_ollama_short_chunks_are_still_exact_batch_size_slices(self):
        # The fast case is untouched: at these lengths the char budget never
        # bites and the packing is the plain slicing it has always been.
        rows = self.rows(*[50] * 20)
        self.assertEqual(annotate._annotation_groups(rows, "test-annotate-model", 8),
                         [rows[i:i + 8] for i in range(0, 20, 8)])

    def test_ollama_long_chunks_close_a_group_before_batch_size(self):
        rows = self.rows(*[3000] * 8)
        groups = annotate._annotation_groups(rows, "test-annotate-model", 8)
        self.assertEqual(self.ids(groups), [[1, 2, 3], [4, 5, 6], [7, 8]])
        self.assertTrue(all(
            sum(len(text) for _chunk_id, text in group)
            <= annotate._OLLAMA_ANNOTATE_CHAR_BUDGET for group in groups))

    def test_a_batch_size_below_one_is_still_one_row_per_group(self):
        rows = self.rows(*[10] * 3)
        self.assertEqual(
            self.ids(annotate._annotation_groups(rows, "test-annotate-model", 0)),
            [[1], [2], [3]])


if __name__ == "__main__":
    unittest.main()
