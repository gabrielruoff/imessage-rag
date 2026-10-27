"""Hermetic tests for hosted annotation through annotate.main: the D1 quality
rules, the hosted-only flags, the guards (annotate_model / terms_model), the
completeness contract and concurrency. Temp SQLite + httpx.MockTransport; the
key is a placeholder and retry sleeps are patched to 0."""

import asyncio
import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import httpx

import annotate
import annotate_hosted
import config
import hosted
import ollama
import openrouter
import schema
from db import get_state, load_vec_extension, set_state

HOSTED = "openrouter/test/model"
LOCAL = "local-model"
GOOD = {"valence": 0.4, "emotions": ["joy"], "depth": 2, "events": []}
LISTING = {"data": [{"id": "test/model",
                     "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}}]}


def completion(content=GOOD):
    return httpx.Response(200, json={
        "model": "test/model-20261007",  # a dated echo: must never reach the guard
        "choices": [{"message": {"content": json.dumps(content)
                                 if not isinstance(content, str) else content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.00001,
                  "prompt_tokens_details": {"cached_tokens": 50}}})


class HostedFixture(unittest.TestCase):
    N_CHUNKS = 5

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "test.db")
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        load_vec_extension(self.con)
        schema.create_schema(self.con)
        self.con.execute("INSERT INTO chats(id, chat_identifier, display_name, is_group, "
                         "service) VALUES (1, 'chat1', NULL, 0, 'iMessage')")
        for i in range(1, self.N_CHUNKS + 1):
            self.con.execute(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, chunk_id) VALUES (?, ?, 1, 1, 0, 'placeholder text', ?, ?)",
                (i, f"guid-{i}", f"2026-01-0{i}T10:00:00", i))
            self.con.execute(
                "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
                "first_date, last_date, n_messages, text) VALUES "
                "(?, 1, ?, ?, '2026-01-01', '2026-01-01', 1, ?)",
                (i, i, i, f"Conversation with Alice:\nAlice: placeholder text {i}"))
        self.con.execute(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, first_message_id) "
            "VALUES (1, 'blorbo', 5.0, 9, 1)")
        self.con.commit()

        self.requests = []
        self.handler = lambda request: completion()
        patches = [
            mock.patch.object(annotate, "DB_PATH", self.db_path),
            mock.patch.object(annotate, "ANNOTATE_MODEL", HOSTED),
            mock.patch.object(annotate, "ANNOTATE_QUALITY", "balanced"),
            # env-sourced knobs pinned to their unset state (a .env must not leak in)
            mock.patch.object(annotate, "ANNOTATE_THINK", ""),
            mock.patch.object(annotate, "ANNOTATE_BATCH_SIZE", 0),
            mock.patch.object(config, "HOSTED_MAX_PARALLEL", 1),
            mock.patch.object(config, "OPEN_ROUTER_API_KEY", "placeholder-key"),
            mock.patch.object(config, "HOSTED_BUDGET_USD", None),
            mock.patch.object(config, "CHAT_MODEL", LOCAL),
            mock.patch.object(hosted, "retry_delay", return_value=0),
            mock.patch.object(openrouter, "client", self.client),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.out = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.out))

    def client(self):
        async def route(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json=LISTING)
            self.requests.append(json.loads(request.content))
            reply = self.handler(request)
            return await reply if asyncio.iscoroutine(reply) else reply
        return httpx.AsyncClient(transport=httpx.MockTransport(route),
                                 base_url=openrouter.BASE_URL)

    def main(self, *argv):
        annotate.main(["--all", "--budget", "1", *argv])

    def n_rows(self, source="llm"):
        return self.con.execute("SELECT COUNT(*) FROM chunk_annotations WHERE source = ?",
                                (source,)).fetchone()[0]


class TestHostedRun(HostedFixture):
    def test_happy_path_writes_llm_rows_and_records_configured_model(self):
        self.main()
        self.assertEqual(self.n_rows(), self.N_CHUNKS)
        self.assertEqual(len(self.requests), self.N_CHUNKS)
        self.assertEqual(get_state(self.con, "annotate_model"), HOSTED)
        self.assertEqual(self.requests[0]["model"], "test/model")
        self.assertEqual(self.requests[0]["reasoning"], {"enabled": False})  # balanced
        self.assertIn("uploading chunk text (test/model) to OpenRouter", self.out.getvalue())
        self.assertIn("pending: 0", self.out.getvalue())
        self.assertIn("spent: $", self.out.getvalue())

    def test_quality_max_omits_reasoning(self):
        self.main("--quality", "max")
        self.assertNotIn("reasoning", self.requests[0])

    def test_401_stops_with_pending_count_and_writes_nothing(self):
        self.handler = lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})
        with self.assertRaises(SystemExit) as cm:
            self.main()
        self.assertIn("401", str(cm.exception.code))
        self.assertIn(f"{self.N_CHUNKS} chunks still pending", str(cm.exception.code))
        self.assertEqual(self.n_rows(), 0)
        self.assertEqual(len(self.requests), 1)

    def test_unparseable_listed_not_written(self):
        def handler(request):
            bad = "Alice: placeholder text 3" in request.content.decode()
            return completion("not json" if bad else GOOD)
        self.handler = handler
        with self.assertRaises(SystemExit) as cm:
            self.main()
        self.assertIn("1 chunks still pending", cm.exception.code)
        self.assertIn("unparseable after 3 retries: 3", cm.exception.code)
        self.assertEqual(self.n_rows(), self.N_CHUNKS - 1)
        self.assertIsNone(self.con.execute(
            "SELECT 1 FROM chunk_annotations WHERE chunk_id = 3").fetchone())

    def test_refused_after_every_provider_listed(self):
        def handler(request):
            body = json.loads(request.content)
            if "placeholder text 2" not in request.content.decode():
                return completion()
            if body["provider"].get("ignore"):
                return httpx.Response(404, json={"error": {"message": "No endpoints found"}})
            return httpx.Response(403, json={"error": {
                "message": "requires moderation; input flagged",
                "metadata": {"provider_name": "ProvA"}}})
        self.handler = handler
        with self.assertRaises(SystemExit) as cm:
            self.main()
        self.assertIn("refused by every provider: 2", cm.exception.code)
        self.assertEqual(self.n_rows(), self.N_CHUNKS - 1)

    def test_commits_every_n_and_resume_skips_written_rows(self):
        reader = sqlite3.connect(self.db_path)
        self.addCleanup(reader.close)
        seen = []

        def handler(request):
            seen.append(reader.execute("SELECT COUNT(*) FROM chunk_annotations").fetchone()[0])
            return completion()
        self.handler = handler
        with mock.patch.object(annotate_hosted, "ANNOTATE_COMMIT_EVERY", 2):
            self.main("--limit", "3")
        self.assertEqual(seen, [0, 0, 2])  # committed after the 2nd write
        self.assertEqual(self.n_rows(), 3)
        self.requests.clear()
        self.main()
        self.assertEqual(len(self.requests), self.N_CHUNKS - 3)
        self.assertEqual(self.n_rows(), self.N_CHUNKS)

    def test_end_of_run_recount_comes_from_the_db(self):
        real_write = annotate._write_annotation

        def lossy(con, chunk_id, data, source="llm"):
            if chunk_id != 4:  # a write that silently never lands
                real_write(con, chunk_id, data, source=source)
        with mock.patch.object(annotate, "_write_annotation", lossy):
            with self.assertRaises(SystemExit) as cm:
                self.main()
        self.assertIn("1 chunks still pending", cm.exception.code)
        self.assertIn("pending: 1", self.out.getvalue())

    def test_max_parallel_never_exceeded(self):
        state = {"now": 0, "peak": 0}

        async def handler(request):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return completion()
        self.handler = handler
        self.main("--max-parallel", "2")
        self.assertEqual(state["peak"], 2)
        self.assertEqual(self.n_rows(), self.N_CHUNKS)

    def test_budget_required_before_guard_or_request(self):
        with self.assertRaises(SystemExit) as cm:
            annotate.main(["--all"])
        self.assertIn("HOSTED_BUDGET_USD", str(cm.exception.code))
        self.assertIsNone(get_state(self.con, "annotate_model"))
        self.assertEqual(self.requests, [])

    def test_dry_run_sends_nothing_and_does_not_guard(self):
        annotate.main(["--all", "--dry-run"])
        self.assertEqual(self.requests, [])
        self.assertIsNone(get_state(self.con, "annotate_model"))

    def test_guard_refuses_hosted_onto_local_llm_rows(self):
        set_state(self.con, "annotate_model", LOCAL)
        self.con.execute("INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, "
                         "depth, events_json, source) VALUES (1, 0, '[]', 1, '[]', 'llm')")
        self.con.commit()
        with self.assertRaises(SystemExit):
            self.main()
        self.assertEqual(self.requests, [])
        self.assertEqual(get_state(self.con, "annotate_model"), LOCAL)


    def test_s4_decoding_error_mid_run_is_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 2:
                raise httpx.DecodingError("placeholder bad encoding")
            return completion()
        self.handler = handler
        self.main()
        self.assertEqual(self.n_rows(), self.N_CHUNKS)
        self.assertEqual(len(calls), self.N_CHUNKS + 1)

    def test_s4_other_exception_still_commits_written_rows(self):
        real_write = annotate._write_annotation
        writes = []

        def failing(con, chunk_id, data, source="llm"):
            writes.append(chunk_id)
            if len(writes) == 3:
                raise sqlite3.OperationalError("placeholder disk I/O error")
            real_write(con, chunk_id, data, source=source)
        with mock.patch.object(annotate, "_write_annotation", failing), \
                self.assertRaises(sqlite3.OperationalError):
            self.main()
        self.assertEqual(self.n_rows(), 2)  # read on a separate connection: committed

    def test_interrupt_commits_written_rows_and_propagates(self):
        real_write = annotate._write_annotation
        writes = []

        def interrupting(con, chunk_id, data, source="llm"):
            writes.append(chunk_id)
            if len(writes) == 3:
                raise KeyboardInterrupt
            real_write(con, chunk_id, data, source=source)
        with mock.patch.object(annotate, "_write_annotation", interrupting), \
                self.assertRaises(KeyboardInterrupt):  # never a 0 exit with chunks pending
            self.main()
        self.assertEqual(self.n_rows(), 2)
        self.assertIn("Interrupted -- committed 2 annotations", self.out.getvalue())

    def test_s4_whitespace_only_key_counts_as_unset(self):
        with mock.patch.object(config, "OPEN_ROUTER_API_KEY", " \n"), \
                self.assertRaisesRegex(SystemExit, "OPEN_ROUTER_API_KEY is not set"):
            self.main()
        self.assertEqual(self.requests, [])


class TestQualityRules(HostedFixture):
    def test_distilled_or_hybrid_flag_refused(self):
        for quality in ("distilled", "hybrid"):
            with self.assertRaises(SystemExit) as cm, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.main("--quality", quality)
            self.assertEqual(cm.exception.code, 2)
        self.assertEqual(self.requests, [])

    def test_env_hybrid_refused(self):
        with mock.patch.object(annotate, "ANNOTATE_QUALITY", "hybrid"), \
                self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.main()
        self.assertEqual(self.requests, [])

    def test_env_distilled_runs_balanced_with_note(self):
        with mock.patch.object(annotate, "ANNOTATE_QUALITY", "distilled"):
            self.main()
        self.assertIn("running balanced", self.out.getvalue())
        self.assertEqual(self.requests[0]["reasoning"], {"enabled": False})
        self.assertEqual(self.n_rows(), self.N_CHUNKS)

    def test_fast_is_one_chunk_per_request_with_note(self):
        self.main("--quality", "fast")
        self.assertIn("one chunk per request", self.out.getvalue())
        self.assertEqual(len(self.requests), self.N_CHUNKS)

    def test_hosted_flags_refused_for_a_local_model(self):
        for flags in (["--budget", "1"], ["--max-parallel", "2"]):
            with self.assertRaises(SystemExit) as cm, \
                    contextlib.redirect_stderr(io.StringIO()):
                annotate.main(["--all", "--model", LOCAL, *flags])
            self.assertEqual(cm.exception.code, 2)

    def test_max_parallel_below_one_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self.main("--max-parallel", "0")
        self.assertIn("--max-parallel", str(cm.exception.code))


class TestTermsUnderHosted(HostedFixture):
    def verdict(self):
        return self.con.execute(
            "SELECT llm_verdict FROM dyad_terms WHERE term = 'blorbo'").fetchone()[0]

    def test_terms_use_chat_model(self):
        with mock.patch.object(annotate, "curate_terms") as fake:
            annotate.main(["--terms"])
        self.assertEqual(fake.call_args.args[1], LOCAL)
        self.assertIn("runs locally on CHAT_MODEL", self.out.getvalue())

    def test_terms_with_hosted_model_flag_refused(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            annotate.main(["--terms", "--model", HOSTED])
        self.assertEqual(cm.exception.code, 2)

    def test_terms_guard_migrates_then_local_terms_pass_after_hosted_chunks(self):
        set_state(self.con, "annotate_model", LOCAL)
        self.con.execute("UPDATE dyad_terms SET llm_verdict = 'ordinary'")
        self.con.execute("INSERT INTO dyad_terms(chat_id, term, lift, n_uses, "
                         "first_message_id) VALUES (1, 'zippy', 4.0, 7, 1)")
        self.con.commit()
        self.main()  # hosted chunks: annotate_model -> HOSTED
        self.assertEqual(get_state(self.con, "terms_model"), LOCAL)
        self.assertEqual(get_state(self.con, "annotate_model"), HOSTED)
        with mock.patch.object(ollama, "chat_once",
                               return_value={"role": "assistant",
                                             "content": '{"verdict": "nickname"}'}):
            annotate.main(["--terms"])
        self.assertEqual(self.con.execute(
            "SELECT llm_verdict FROM dyad_terms WHERE term = 'zippy'").fetchone()[0],
            "nickname")
        annotate._migrate_terms_guard(self.con)  # idempotent
        self.assertEqual(get_state(self.con, "terms_model"), LOCAL)

    def test_no_migration_without_curated_terms(self):
        set_state(self.con, "annotate_model", LOCAL)
        self.con.commit()
        annotate._migrate_terms_guard(self.con)
        self.assertIsNone(get_state(self.con, "terms_model"))


if __name__ == "__main__":
    unittest.main()
