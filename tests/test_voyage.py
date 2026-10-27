"""Hermetic tests for the Voyage backend: voyage.py (sync query call, packing,
the async Batch API client) and its ingest integration (embeddings.py,
chunking.embed_chunks_voyage, chunking.build_vec_table).

No network and no real key: every request goes to an in-process fake Voyage
behind httpx.MockTransport; temp SQLite only; the poll interval and the retry
backoff are patched to ~0."""

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

import chunking
import config
import embeddings
import hosted
import ingest
import voyage
from db import get_state, load_vec_extension

DIM = 4
MODEL = "voyage/voyage-4-large"
REAL_RETRY_DELAY = hosted.retry_delay  # the fixture patches it to 0


class FakeVoyage:
    """Just enough of the Files + Batches API. A batch reports in_progress on
    its first poll and completed on the next; its output is written in reverse
    input order. `fail_once` chunk ids land in the error file the first time
    they are submitted; `stale_once` ids get a wrong-hash custom_id once;
    `script` maps (method, path-prefix) to a queue of canned responses served
    before the normal handler. `redirect_content` makes file downloads answer
    307 to a signed storage URL, as the live API does."""

    def __init__(self):
        self.files, self.batches, self.deleted = {}, {}, []
        self.requests = []
        self.fail_once, self.stale_once, self.script = set(), set(), {}
        self.batch_status = "completed"
        self.redirect_content, self.storage_auth = False, []

    def transport(self):
        return httpx.MockTransport(self.handle)

    def _vec(self, chunk_id):
        return [float(chunk_id), 0.0, 0.0, 1.0]

    def handle(self, request):
        method, path = request.method, request.url.path
        if request.url.host == "storage.example.com":
            self.storage_auth.append(request.headers.get("authorization"))
            file_id = request.url.params["f"]
            return httpx.Response(200, content=b"".join(ln + b"\n" for ln in self.files[file_id]))
        self.requests.append((method, path))
        for (m, prefix), queue in self.script.items():
            if m == method and path.startswith(prefix) and queue:
                item = queue.pop(0)
                return item(request) if callable(item) else item
        if method == "GET" and path == "/v1/batches":
            # newest first, two per page, to exercise pagination
            ids = list(reversed(self.batches))
            start = ids.index(request.url.params["after"]) + 1 if "after" in request.url.params else 0
            page = ids[start:start + 2]
            return httpx.Response(200, json={
                "data": [self.batches[i] for i in page],
                "has_more": start + 2 < len(ids), "last_id": page[-1] if page else None})
        if method == "GET" and path.startswith("/v1/batches/") \
                and path.rsplit("/", 1)[1] not in self.batches:
            return httpx.Response(404, json={"detail": "not found"})
        if method == "POST" and path == "/v1/embeddings":
            body = json.loads(request.content)
            self.last_sync_body = body
            data = [{"index": i, "embedding": [float(i)] * DIM}
                    for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": list(reversed(data))})
        if method == "POST" and path == "/v1/files":
            lines = [ln for ln in request.content.split(b"\n")
                     if ln.startswith(b'{"custom_id"')]
            file_id = f"file-{len(self.files)}"
            self.files[file_id] = lines
            return httpx.Response(200, json={"id": file_id})
        if method == "POST" and path == "/v1/batches":
            return self.create(request)
        if method == "GET" and path.startswith("/v1/batches/"):
            batch = self.batches[path.rsplit("/", 1)[1]]
            batch["polls"] += 1
            if batch["polls"] == 1:
                batch["status"] = "in_progress"
            elif batch["status"] == "in_progress":
                self._complete(batch)
            return httpx.Response(200, json=batch)
        if method == "GET" and path.endswith("/content"):
            file_id = path.split("/")[3]
            if self.redirect_content:
                return httpx.Response(307, headers={
                    "Location": f"https://storage.example.com/signed?x=1&f={file_id}"})
            return httpx.Response(200, content=b"".join(ln + b"\n" for ln in self.files[file_id]))
        if method == "DELETE" and path.startswith("/v1/files/"):
            self.deleted.append(path.rsplit("/", 1)[1])
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404)

    def create(self, request):
        body = json.loads(request.content)
        self.last_batch_body = body
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = {"id": batch_id, "status": "validating", "model": "voyage-4-large",
                                  "input_file_id": body["input_file_id"],
                                  "request_counts": {"total": 0}, "polls": 0}
        return httpx.Response(200, json=self.batches[batch_id])

    def _complete(self, batch):
        out, err = [], []
        for raw in self.files[batch["input_file_id"]]:
            line = json.loads(raw)
            cid, digest = voyage.parse_custom_id(line["custom_id"])
            if cid in self.fail_once:
                self.fail_once.discard(cid)
                err.append({"custom_id": line["custom_id"], "response": None,
                            "error": {"code": "batch_expired", "message": "expired"}})
                continue
            if cid in self.stale_once:
                self.stale_once.discard(cid)
                line["custom_id"] = f"{cid}:000000000000"
            out.append({"custom_id": line["custom_id"], "error": None, "response": {
                "status_code": 200,
                "body": {"data": [{"index": 0, "embedding": self._vec(cid)}],
                         "usage": {"total_tokens": 10}}}})
        n = len(self.files)
        self.files[f"file-{n}"] = [json.dumps(r).encode() for r in reversed(out)]
        batch["output_file_id"] = f"file-{n}"
        if err:
            self.files[f"file-{n + 1}"] = [json.dumps(r).encode() for r in err]
            batch["error_file_id"] = f"file-{n + 1}"
        batch["status"] = self.batch_status
        batch["request_counts"] = {"total": len(out) + len(err),
                                   "completed": len(out), "failed": len(err)}

    def n_created(self):
        return sum(1 for m, p in self.requests if (m, p) == ("POST", "/v1/batches"))


class VoyageFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.con = sqlite3.connect(os.path.join(tmp.name, "test.db"))
        self.addCleanup(self.con.close)
        load_vec_extension(self.con)
        ingest.create_schema(self.con)
        self.fake = FakeVoyage()
        real_client = voyage.BatchClient
        for patcher in (
            mock.patch.object(config, "EMBED_MODEL", MODEL),
            mock.patch.object(config, "EMBED_DIM", DIM),
            mock.patch.object(config, "VOYAGE_API_KEY", "test-placeholder"),
            mock.patch.object(config, "VOYAGE_POLL_SECONDS", 0),
            mock.patch.object(chunking, "EMBED_MODEL", MODEL),
            mock.patch.object(embeddings, "EMBED_MODEL", MODEL),
            mock.patch.object(embeddings, "EMBED_DIM", DIM),
            mock.patch.object(hosted, "retry_delay", lambda attempt, response=None: 0.0),
            mock.patch.object(voyage, "BatchClient", lambda limiter: real_client(
                limiter, transport=self.fake.transport())),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        chunking.build_vec_table(self.con, DIM)

    def add_chunks(self, texts):
        for i, text in enumerate(texts, start=1):
            self.con.execute(
                "INSERT INTO chunks(id, chat_id, first_date, last_date, n_messages, text) "
                "VALUES (?, 1, '2026-01-01T10:00:00', '2026-01-02T10:00:00', 1, ?)",
                (i, text))
        self.con.commit()

    def vectors(self):
        return dict(self.con.execute(
            "SELECT chunk_id, vec_to_json(embedding) FROM chunk_vectors").fetchall())

    def run_voyage(self, budget=1.0):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            chunking.embed_chunks_voyage(self.con, hosted.Budget(budget), 2)
        return out.getvalue()


class TestPacking(unittest.TestCase):
    def test_one_input_per_line_and_batch_cap(self):
        rows = [(i, f"placeholder text {i}") for i in range(1, 8)]
        with mock.patch.object(config, "VOYAGE_BATCH_MAX_INPUTS", 3):
            batches = voyage.pack_lines(rows)
        self.assertEqual([len(lines) for _, lines in batches], [3, 3, 1])
        self.assertEqual([r for batch_rows, _ in batches for r in batch_rows], rows)
        for batch_rows, lines in batches:
            for (cid, text), raw in zip(batch_rows, lines):
                self.assertTrue(raw.endswith(b"\n"))
                line = json.loads(raw)
                self.assertEqual(line["body"]["input"], [text])
                self.assertEqual(voyage.parse_custom_id(line["custom_id"])[0], cid)

    def test_emoji_text_round_trips_and_reserves_by_utf8_bytes(self):
        text = "\U0001F600" * 50 + " placeholder"
        ((_, (raw,)),) = voyage.pack_lines([(9, text)])
        line = json.loads(raw)
        self.assertEqual(line["body"]["input"], [text])
        # 50 four-byte emoji + 12 ASCII bytes: bytes, not characters, bound tokens.
        with mock.patch.object(config, "VOYAGE_BATCH_PRICE_PER_M", 1e6):
            self.assertEqual(voyage.worst_case_usd([text]), 212)

    def test_custom_id_round_trip(self):
        cid = voyage.make_custom_id(42, "placeholder chunk")
        self.assertEqual(voyage.parse_custom_id(cid),
                         (42, voyage.text_hash("placeholder chunk")))
        self.assertEqual(len(cid.split(":")[1]), 12)


class TestSyncAndEmbeddings(VoyageFixture):
    def test_query_goes_sync_in_input_order_with_no_prefix_and_no_output(self):
        real = voyage.embed_sync
        with mock.patch.object(voyage, "embed_sync", lambda texts, input_type: real(
                texts, input_type, transport=self.fake.transport())), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            vecs = embeddings.embed_texts(["a", "b"], kind="query")
        self.assertEqual(vecs, [[0.0] * DIM, [1.0] * DIM])
        body = self.fake.last_sync_body
        self.assertEqual(body["input"], ["a", "b"])  # no nomic prefix
        self.assertEqual((body["model"], body["input_type"], body["output_dimension"]),
                         ("voyage-4-large", "query", DIM))
        self.assertEqual(out.getvalue(), "")  # MCP stdout is the wire

    def test_sync_is_bounded_then_raises(self):
        self.fake.script[("POST", "/v1/embeddings")] = [httpx.Response(503)] * 5
        with self.assertRaises(RuntimeError):
            voyage.embed_sync(["q"], "query", transport=self.fake.transport())
        self.assertEqual(len(self.fake.requests), voyage.SYNC_ATTEMPTS)

    def test_document_embedding_raises(self):
        with self.assertRaisesRegex(RuntimeError, "batch-only"):
            embeddings.embed_texts(["x"], kind="document")

    def test_detect_embed_dim_makes_no_request(self):
        with mock.patch.object(httpx.Client, "send") as send:
            self.assertEqual(embeddings.detect_embed_dim(), DIM)
        send.assert_not_called()


class TestVecTableGuard(VoyageFixture):
    def test_records_voyage_model_and_dim(self):
        self.assertEqual(get_state(self.con, "embed_model"), MODEL)
        self.assertEqual(get_state(self.con, "embed_dim"), str(DIM))

    def test_refuses_a_nomic_db_with_vectors(self):
        self.con.execute("DROP TABLE chunk_vectors")
        with mock.patch.object(chunking, "EMBED_MODEL", "nomic-embed-text"):
            chunking.build_vec_table(self.con, DIM)
        self.con.execute("INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, "
                         "last_day) VALUES (1, '[1,0,0,0]', 1, 1, 1)")
        with self.assertRaisesRegex(SystemExit, "Mixing embedding"):
            chunking.build_vec_table(self.con, DIM)


class TestBatchRun(VoyageFixture):
    def test_full_run_stores_vectors_out_of_order_and_cleans_up(self):
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        out = self.run_voyage()
        vecs = self.vectors()
        self.assertEqual(sorted(vecs), [1, 2, 3])
        for cid, vec in vecs.items():
            self.assertEqual(json.loads(vec), self.fake._vec(cid))
        self.assertEqual(json.loads(get_state(self.con, "voyage_batch_ids")), [])
        self.assertEqual(self.fake.n_created(), 1)
        body = self.fake.last_batch_body
        self.assertEqual(body["completion_window"], "12h")
        self.assertEqual(body["request_params"]["input_type"], "document")
        self.assertNotIn("requests_params", body)
        # input, output deleted after commit
        self.assertEqual(sorted(self.fake.deleted), ["file-0", "file-1"])
        self.assertIn("0 chunks without a vector", out)
        self.assertIn("spent: $", out)

    def test_short_chunks_are_not_submitted(self):
        self.add_chunks(["placeholder long enough", "tiny"])
        self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1])

    def test_stale_hash_is_discarded_and_resubmitted(self):
        self.add_chunks(["placeholder one", "placeholder two"])
        self.fake.stale_once = {2}
        out = self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1, 2])
        self.assertEqual(self.fake.n_created(), 2)
        self.assertIn("1 stale discarded", out)

    def test_error_file_lines_resubmitted_until_empty(self):
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        self.fake.fail_once = {1, 3}
        out = self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1, 2, 3])
        self.assertEqual(self.fake.n_created(), 2)
        self.assertIn("2 errored", out)
        self.assertIn("0 errored", out)
        # second batch resubmitted only the errored chunks
        resubmitted = [voyage.parse_custom_id(json.loads(ln)["custom_id"])[0]
                       for ln in self.fake.files[self.fake.batches["batch-1"]["input_file_id"]]]
        self.assertEqual(sorted(resubmitted), [1, 3])

    def test_output_file_307_to_storage_is_followed_without_the_key(self):
        self.fake.redirect_content = True
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        self.run_voyage()
        vecs = self.vectors()
        self.assertEqual(sorted(vecs), [1, 2, 3])
        for cid, vec in vecs.items():
            self.assertEqual(json.loads(vec), self.fake._vec(cid))
        self.assertTrue(self.fake.storage_auth)
        self.assertEqual(self.fake.storage_auth, [None] * len(self.fake.storage_auth))

    def test_error_file_307_to_storage_is_followed(self):
        self.fake.redirect_content = True
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        self.fake.fail_once = {1, 3}
        out = self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1, 2, 3])
        self.assertIn("2 errored", out)
        # batch-0's output and error files, batch-1's output file
        self.assertEqual(len(self.fake.storage_auth), 3)
        self.assertEqual(self.fake.storage_auth, [None] * 3)

    def test_429_with_retry_after_then_success(self):
        self.add_chunks(["placeholder one"])
        self.fake.script[("POST", "/v1/files")] = [
            httpx.Response(429, headers={"Retry-After": "0"})]
        delays = []
        with mock.patch.object(hosted, "retry_delay",
                               lambda attempt, response=None: delays.append(
                                   response.headers.get("retry-after")) or 0.0):
            self.run_voyage()
        self.assertEqual(delays, ["0"])
        self.assertEqual(sorted(self.vectors()), [1])

    def test_401_stops_with_pending_count(self):
        self.add_chunks(["placeholder one", "placeholder two"])
        self.fake.script[("POST", "/v1/files")] = [httpx.Response(401, json={"detail": "bad"})]
        with self.assertRaises(hosted.HostedStop) as ctx:
            self.run_voyage()
        self.assertIn("401", ctx.exception.cause)
        self.assertEqual(len(chunking.unembedded_chunks(self.con)), 2)

    def test_ingest_turns_stop_into_exit_naming_pending(self):
        self.add_chunks(["placeholder one", "placeholder two"])
        self.fake.script[("POST", "/v1/files")] = [httpx.Response(401, json={"detail": "bad"})]
        with mock.patch.object(ingest, "detect_embed_dim", lambda: DIM), \
                mock.patch.object(ingest, "EMBED_MODEL", MODEL), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(SystemExit) as ctx:
            ingest._embed_voyage(self.con, hosted.Budget(1.0), 1)
        self.assertIn("2 chunks still have no vector", str(ctx.exception.code))
        self.assertIn("uploading chunk text to Voyage AI (voyage-4-large)", out.getvalue())

    def test_resume_from_saved_batch_creates_no_new_batch(self):
        self.add_chunks(["placeholder one", "placeholder two"])
        rows = [(1, "placeholder one"), (2, "placeholder two")]
        ((_, lines),) = voyage.pack_lines(rows)
        self.fake.files["file-0"] = [ln.rstrip(b"\n") for ln in lines]
        self.fake.batches["batch-0"] = {"id": "batch-0", "status": "in_progress",
                                        "input_file_id": "file-0",
                                        "request_counts": {"total": 2}, "polls": 1}
        self.con.execute("INSERT INTO ingest_state(key, value) "
                         "VALUES ('voyage_batch_ids', '[\"batch-0\"]')")
        self.con.commit()
        self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1, 2])
        self.assertEqual(self.fake.n_created(), 0)
        self.assertEqual(json.loads(get_state(self.con, "voyage_batch_ids")), [])

    def test_budget_refused_before_upload(self):
        self.add_chunks(["placeholder one"])
        with self.assertRaisesRegex(hosted.HostedStop, "budget"):
            self.run_voyage(budget=1e-12)
        self.assertEqual(self.fake.requests, [])

    def test_failed_batches_stall_then_stop(self):
        self.add_chunks(["placeholder one"])
        self.fake.batch_status = "failed"
        self.fake.fail_once = set()
        # a failed batch delivers nothing: make its output empty
        orig = self.fake._complete

        def complete_empty(batch):
            orig(batch)
            self.fake.files[batch["output_file_id"]] = []
        self.fake._complete = complete_empty
        with self.assertRaisesRegex(hosted.HostedStop, "embedded nothing"):
            self.run_voyage()
        self.assertEqual(self.fake.n_created(), 3)

    def test_end_of_run_nonzero_when_still_missing(self):
        self.add_chunks(["placeholder one"])
        with mock.patch.object(chunking, "_embed_voyage", mock.AsyncMock()), \
                self.assertRaisesRegex(SystemExit, "1 chunks still have no vector"):
            self.run_voyage()


class TestReviewRegressions(VoyageFixture):
    def save_batch(self, batch_id, rows, **fields):
        """Plant a batch a previous run submitted, its id saved in ingest_state."""
        ((_, lines),) = voyage.pack_lines(rows)
        file_id = f"file-{len(self.fake.files)}"
        self.fake.files[file_id] = [ln.rstrip(b"\n") for ln in lines]
        self.fake.batches[batch_id] = {"id": batch_id, "status": "in_progress",
                                       "model": "voyage-4-large", "input_file_id": file_id,
                                       "request_counts": {"total": len(rows)}, "polls": 1,
                                       **fields}
        self.con.execute("INSERT OR REPLACE INTO ingest_state(key, value) "
                         "VALUES ('voyage_batch_ids', ?)", (json.dumps([batch_id]),))
        self.con.commit()

    def test_m1_lost_create_reply_adopts_the_batch_instead_of_duplicating(self):
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        def create_then_drop(request):
            self.fake.create(request)
            for i in range(3):  # newer, unrelated batches: reconciliation must page
                self.fake.batches[f"other-{i}"] = {"id": f"other-{i}", "status": "completed",
                                                   "input_file_id": f"file-other-{i}"}
            raise httpx.RemoteProtocolError("connection dropped", request=request)
        self.fake.script[("POST", "/v1/batches")] = [create_then_drop]
        self.run_voyage()
        created = [b for b in self.fake.batches if b.startswith("batch-")]
        self.assertEqual(len(created), 1)
        self.assertEqual(self.fake.requests.count(("GET", "/v1/batches")), 2)  # paged
        self.assertEqual(sorted(self.vectors()), [1, 2, 3])
        self.assertEqual(get_state(self.con, "voyage_pending_input_file"), "")

    def test_m1_resume_adopts_a_batch_from_a_saved_pending_input_file(self):
        self.add_chunks(["placeholder one", "placeholder two"])
        self.save_batch("batch-orphan", [(1, "placeholder one"), (2, "placeholder two")])
        self.con.execute("UPDATE ingest_state SET value = '[]' WHERE key = 'voyage_batch_ids'")
        self.con.execute("INSERT INTO ingest_state(key, value) VALUES "
                         "('voyage_pending_input_file', ?)",
                         (self.fake.batches["batch-orphan"]["input_file_id"],))
        self.con.commit()
        self.run_voyage()
        self.assertEqual(self.fake.n_created(), 0)
        self.assertEqual(sorted(self.vectors()), [1, 2])

    def test_m1_resume_deletes_a_pending_input_file_with_no_batch(self):
        self.add_chunks(["placeholder one"])
        self.fake.files["file-orphan"] = []
        self.con.execute("INSERT INTO ingest_state(key, value) VALUES "
                         "('voyage_pending_input_file', 'file-orphan')")
        self.con.commit()
        self.run_voyage()
        self.assertIn("file-orphan", self.fake.deleted)
        self.assertEqual(self.fake.n_created(), 1)

    def test_m2_a_saved_batch_voyage_no_longer_serves_is_dropped(self):
        self.add_chunks(["placeholder one"])
        self.con.execute("INSERT INTO ingest_state(key, value) "
                         "VALUES ('voyage_batch_ids', '[\"batch-gone\"]')")
        self.con.commit()
        out = self.run_voyage()
        self.assertIn("dropping Voyage batch batch-gone", out)
        self.assertEqual(self.fake.n_created(), 1)
        self.assertEqual(sorted(self.vectors()), [1])
        self.assertEqual(json.loads(get_state(self.con, "voyage_batch_ids")), [])

    def test_m2_a_404_outside_resume_still_stops(self):
        self.add_chunks(["placeholder one"])
        self.fake.script[("GET", "/v1/batches/batch-0")] = [httpx.Response(404)]
        with self.assertRaises(voyage.NotFound):
            self.run_voyage()

    def test_m3_resumed_batch_from_another_model_is_discarded(self):
        self.add_chunks(["placeholder one"])
        self.save_batch("batch-old", [(1, "placeholder one")], model="voyage-3")
        out = self.run_voyage()
        self.assertIn("dropping Voyage batch batch-old", out)
        # its vector never landed; the chunk was resubmitted under the current model
        self.assertEqual(self.fake.n_created(), 1)
        self.assertEqual(json.loads(self.vectors()[1]), self.fake._vec(1))

    def test_m3_resumed_batch_from_another_space_key_is_discarded(self):
        self.add_chunks(["placeholder one"])
        self.save_batch("batch-old", [(1, "placeholder one")])
        self.con.execute("INSERT INTO ingest_state(key, value) "
                         "VALUES ('voyage_batch_space', 'voyage/voyage-4-large|2048')")
        self.con.commit()
        out = self.run_voyage()
        self.assertIn("dropping Voyage batch batch-old", out)
        self.assertEqual(self.fake.n_created(), 1)

    def test_m3_wrong_length_vectors_are_never_inserted(self):
        self.add_chunks(["placeholder one"])
        self.save_batch("batch-old", [(1, "placeholder one")])
        self.fake._vec = lambda cid: [1.0, 0.0]  # 2-dim from the resumed batch
        with self.assertRaisesRegex(hosted.HostedStop, "embedded nothing"):
            self.run_voyage()
        self.assertEqual(self.vectors(), {})

    def test_l1_key_never_reaches_error_messages(self):
        key = "sk-placeholder-secret\n"

        def leak(request):
            raise httpx.LocalProtocolError(f"Illegal header value {key!r}")
        with mock.patch.object(config, "VOYAGE_API_KEY", key):
            fake = httpx.MockTransport(leak)
            with self.assertRaises(RuntimeError) as ctx:
                voyage.embed_sync(["q"], "query", transport=fake)
            self.assertNotIn("sk-placeholder", str(ctx.exception))
            self.assertEqual(voyage._headers()["Authorization"],
                             "Bearer sk-placeholder-secret")
            self.add_chunks(["placeholder one"])
            self.fake.script[("POST", "/v1/files")] = [leak]
            with self.assertRaises(hosted.HostedStop) as ctx:
                self.run_voyage()
            self.assertNotIn("sk-placeholder", ctx.exception.cause)
            self.assertIn("LocalProtocolError", ctx.exception.cause)

    def test_l3_interrupt_while_polling_keeps_the_batch_id(self):
        self.add_chunks(["placeholder one"])

        def interrupt(request):
            raise KeyboardInterrupt
        self.fake.script[("GET", "/v1/batches/batch-0")] = [interrupt]
        with self.assertRaises(KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            chunking.embed_chunks_voyage(self.con, hosted.Budget(1.0), 1)
        self.assertEqual(json.loads(get_state(self.con, "voyage_batch_ids")), ["batch-0"])

    def test_l3_spent_equals_reported_tokens_times_price(self):
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        budget = hosted.Budget(1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            chunking.embed_chunks_voyage(self.con, budget, 1)
        self.assertAlmostEqual(budget.spent, 3 * 10 * config.VOYAGE_BATCH_PRICE_PER_M / 1e6)
        self.assertAlmostEqual(budget.reserved, 0.0)

    def test_failed_batch_summary_shows_error_codes_and_failed_lines(self):
        self.add_chunks(["placeholder one"])
        self.fake.batch_status = "failed"
        orig = self.fake._complete

        def complete_failed(batch):
            orig(batch)
            batch["errors"] = {"data": [{"code": "invalid_line", "line": 1,
                                         "message": "quoted input text"}]}
            self.fake.files[batch["output_file_id"]] = [json.dumps(
                {"custom_id": "1:x", "response": {"status_code": 500}}).encode()]
        self.fake._complete = complete_failed
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(hosted.HostedStop):
            chunking.embed_chunks_voyage(self.con, hosted.Budget(1.0), 1)
        self.assertIn("1 failed", out.getvalue())
        self.assertIn("('invalid_line', 1)", out.getvalue())
        self.assertNotIn("quoted input text", out.getvalue())

    def test_s4_resume_after_partial_collect_embeds_each_chunk_once(self):
        self.add_chunks(["placeholder one", "placeholder two", "placeholder three"])
        real_row, calls = chunking._voyage_row, []

        def dies_on_second_line(tgt, custom_id, vec):
            calls.append(custom_id)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real_row(tgt, custom_id, vec)
        with mock.patch.object(chunking, "_voyage_row", dies_on_second_line), \
                self.assertRaises(KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            chunking.embed_chunks_voyage(self.con, hosted.Budget(1.0), 1, commit_every=1)
        self.assertEqual(len(self.vectors()), 1)  # one committed before the crash
        self.assertEqual(json.loads(get_state(self.con, "voyage_batch_ids")), ["batch-0"])
        self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1, 2, 3])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0], 3)
        self.assertEqual(self.fake.n_created(), 1)

    def test_s4_pending_input_from_another_space_is_not_adopted(self):
        self.add_chunks(["placeholder one"])
        self.save_batch("batch-orphan", [(1, "placeholder one")])
        file_id = self.fake.batches["batch-orphan"]["input_file_id"]
        self.con.execute("UPDATE ingest_state SET value = '[]' WHERE key = 'voyage_batch_ids'")
        self.con.execute("INSERT INTO ingest_state(key, value) VALUES "
                         "('voyage_pending_input_file', ?), "
                         "('voyage_pending_input_space', 'voyage/voyage-3|4')", (file_id,))
        self.con.commit()
        out = self.run_voyage()
        self.assertIn("not adopting Voyage batch batch-orphan", out)
        self.assertNotIn(("GET", "/v1/batches/batch-orphan"), self.fake.requests)
        self.assertIn(file_id, self.fake.deleted)
        self.assertEqual(self.fake.n_created(), 1)  # resubmitted under the current space
        self.assertEqual(sorted(self.vectors()), [1])
        self.assertEqual(get_state(self.con, "voyage_pending_input_file"), "")

    def test_s4_sync_gives_up_when_retry_after_exceeds_the_cap(self):
        self.fake.script[("POST", "/v1/embeddings")] = [
            httpx.Response(429, headers={"Retry-After": "3600"})]
        with mock.patch.object(hosted, "retry_delay", REAL_RETRY_DELAY), \
                mock.patch.object(config, "HOSTED_BACKOFF_MAX_SECONDS", 60), \
                mock.patch.object(voyage.time, "sleep") as sleep, \
                self.assertRaisesRegex(RuntimeError, "Retry-After 3600s"):
            voyage.embed_sync(["q"], "query", transport=self.fake.transport())
        sleep.assert_not_called()
        self.assertEqual(len(self.fake.requests), 1)

    def test_s4_sync_and_batch_retry_a_decoding_error(self):
        def bad(request):
            raise httpx.DecodingError("placeholder bad encoding")
        self.fake.script[("POST", "/v1/embeddings")] = [bad]
        vecs = voyage.embed_sync(["q"], "query", transport=self.fake.transport())
        self.assertEqual(vecs, [[0.0] * DIM])
        self.add_chunks(["placeholder one"])
        self.fake.script[("POST", "/v1/files")] = [bad]
        self.run_voyage()
        self.assertEqual(sorted(self.vectors()), [1])

    def test_s4_stop_never_quotes_the_response_body(self):
        self.add_chunks(["placeholder one"])
        self.fake.script[("POST", "/v1/files")] = [
            httpx.Response(400, text="echo: placeholder private text")]
        with self.assertRaises(hosted.HostedStop) as ctx:
            self.run_voyage()
        self.assertEqual(ctx.exception.cause, "Voyage: HTTP 400")
        self.fake.script[("POST", "/v1/files")] = [
            httpx.Response(400, json={"detail": "placeholder bad request"})]
        with self.assertRaises(hosted.HostedStop) as ctx:
            self.run_voyage()
        self.assertEqual(ctx.exception.cause, "Voyage: HTTP 400 placeholder bad request")


    def test_stream_file_frees_its_slot_when_the_consumer_raises(self):
        # limit=1: a slot kept by the suspended generator would hang delete_file.
        self.fake.files["file-x"] = [b'{"n": 1}', b'{"n": 2}']

        async def run():
            limiter = hosted.Limiter(1)
            async with voyage.BatchClient(limiter) as client:
                try:
                    async for _ in client.stream_file("file-x"):
                        raise ValueError("consumer fails mid-stream")
                except ValueError:  # the live traceback keeps the generator alive
                    self.assertEqual(limiter.in_flight, 0)
                    await asyncio.wait_for(client.delete_file("file-x"), 5)

        asyncio.run(run())
        self.assertEqual(self.fake.deleted, ["file-x"])

class TestIngestFlags(unittest.TestCase):
    def test_voyage_refuses_without_key_before_snapshot(self):
        with mock.patch.object(ingest, "EMBED_MODEL", MODEL), \
                mock.patch.object(ingest, "VOYAGE_API_KEY", " \n"), \
                mock.patch.object(ingest, "snapshot_source") as snap, \
                mock.patch("sys.argv", ["ingest.py", "--budget", "1"]), \
                self.assertRaisesRegex(SystemExit, "VOYAGE_API_KEY"):
            ingest.main()
        snap.assert_not_called()

    def test_flags_with_no_embed_are_refused(self):
        with mock.patch.object(ingest, "EMBED_MODEL", MODEL), \
                mock.patch("sys.argv", ["ingest.py", "--no-embed", "--budget", "1"]), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            ingest.main()
        self.assertIn("no effect with --no-embed", err.getvalue())

    def test_budget_and_max_parallel_refused_for_local_model(self):
        for argv in (["--budget", "1"], ["--max-parallel", "2"]):
            with self.subTest(argv=argv), \
                    mock.patch.object(ingest, "EMBED_MODEL", "nomic-embed-text"), \
                    mock.patch("sys.argv", ["ingest.py", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit):
                ingest.main()
            self.assertIn("only apply to hosted models", err.getvalue())


if __name__ == "__main__":
    unittest.main()
