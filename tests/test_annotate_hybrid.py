"""Hermetic tests for the hybrid annotate tier (annotate.py --quality hybrid)
and the pieces it is built from: annotate_hybrid's budgeted phase machine,
annotate_distill's knn_confidence + backfill, annotate_sampling's stratified
draw, annotate_events' prefilter, and the chunk_annotations.knn_confidence
migration. Temp SQLite only; BOTH Ollama seams the annotate ladder dispatches
through are faked (chat_once_json, and chat_once for the schema-retry rung) --
no Ollama, no real DB.

The clock is faked the way the tier actually spends it: FakeClock only moves
when a (fake) LLM request charges it, so a deadline test asserts the dispatch
rule rather than racing a real wall clock.
"""

import contextlib
import json
import os
import random
import re
import sqlite3
import tempfile
import unittest
from unittest import mock

import annotate
import annotate_distill
import annotate_events
import annotate_hybrid
import annotate_sampling
import config
import ollama
import schema
from db import get_state, load_vec_extension, set_state

try:
    import sqlite_vec
except ImportError:                                       # pragma: no cover
    sqlite_vec = None


def _vec_available():
    """chunk_vectors is a sqlite-vec vec0 virtual table, so every kNN test
    needs a Python build with loadable-extension support (python.org builds
    lack it -- see CLAUDE.md's install note). Skip cleanly rather than fail."""
    if sqlite_vec is None:
        return False
    con = sqlite3.connect(":memory:")
    try:
        load_vec_extension(con)
        return True
    except (AttributeError, sqlite3.Error):
        return False
    finally:
        con.close()


_HAS_VEC = _vec_available()
_NEEDS_VEC = unittest.skipUnless(_HAS_VEC, "sqlite-vec extension unavailable")

_DIM = 8
_MODEL = "test-annotate-model"

# The kNN index cap the TestSeedIndexCap tests ask for. Pinned here because
# the fixture pool has to stay comfortably ABOVE it: at or under it the cap
# stops biting and those tests silently change meaning from "the cap bounded
# the index" to "the pool already fit".
_INDEX_CAP = 200

# Enough LLM rows to clear every floor that gates the kNN passes:
# annotate_distill._KNN_MIN_SEEDS (the vote refuses fewer), _INDEX_CAP above,
# and ANNOTATE_SEED_TARGET (so the distilled tier's own seed pass is a no-op
# and a test can drive _classify_chunks straight away).
#
# The env value has to stay in the max() -- distill_chunks binds
# ANNOTATE_SEED_TARGET as a DEFAULT ARGUMENT, frozen at import, so a test that
# reaches it through annotate.main cannot patch the target and the pool must
# simply clear whatever a developer's .env baked in. What the other two floors
# add is that a LOW env value can no longer drag the pool under the cap the
# index tests use: unpinned, `ANNOTATE_SEED_TARGET=150` left a 160-seed pool,
# which broke two of those tests outright ("160 != 200") and silently weakened a
# third to asserting determinism over an UNCAPPED index. (A high value still
# makes the fixture bigger, which costs setUp time and nothing else.)
_SEEDS = max(annotate_distill._KNN_MIN_SEEDS,
             _INDEX_CAP + 100,
             config.ANNOTATE_SEED_TARGET) + 10
# chat 1 keeps the pending chunks the LLM phases work on; chat 2 exists so the
# top-dyad scope has something to exclude.
_PENDING_MAIN = 30
_PENDING_OTHER = 10
_FIRST_PENDING = _SEEDS + 1
_OTHER_START = _FIRST_PENDING + _PENDING_MAIN
_LAST = _OTHER_START + _PENDING_OTHER - 1

# Event-candidate chunks (their messages carry EVENT_CANDIDATE_PATTERNS text).
_EVENT_IDS = [_FIRST_PENDING, _FIRST_PENDING + 1, _FIRST_PENDING + 2]
_EVENT_IN_OTHER_CHAT = _OTHER_START + 4

_ANNOTATION = {"valence": 0.5, "emotions": ["joy"], "depth": 2, "events": []}

# Body text for the chunks that must NOT match the event prefilter.
_PLAIN = "logistics about the meeting time"


def _unit_vector(rng):
    v = [rng.gauss(0.0, 1.0) for _ in range(_DIM)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def _chunk_text(chunk_id, body):
    """Chunk text that names its own id, so the fake LLM can tell which chunks
    a request covered from the prompt alone (a singleton request carries the
    bare chunk text, with no '### chunk <id>' header to read)."""
    return f"chunk {chunk_id}: {body}"


_ID_RE = re.compile(r"chunk (\d+)")


def _ids_in(content):
    """Chunk ids named by a request prompt, in order, deduplicated."""
    out = []
    for match in _ID_RE.findall(content):
        chunk_id = int(match)
        if chunk_id not in out:
            out.append(chunk_id)
    return out


class FakeClock:
    """time.monotonic() that only advances when something charges it. Patched
    over annotate_hybrid.time so a deadline test is exact instead of timing-
    dependent: the fake LLM charges the seconds a real request would cost."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LLMError(RuntimeError):
    """What an ollama timeout looks like from inside a dispatch loop: an
    ordinary exception propagating out of _pooled, which is neither a
    KeyboardInterrupt nor a SystemExit and so hits no handler on the way out."""


class ReachedTheNetwork(BaseException):
    """Deliberately NOT an Exception. It stands in for the ollama.chat_once
    seam in every fixture here, and that seam's own caller
    (annotate._annotate_schema_retry) catches `Exception` on purpose -- its
    contract is "degrade to the per-chunk rung, never propagate". An ordinary
    error would therefore be swallowed, the test would pass, and a real run
    would have opened an HTTP connection to localhost. This one propagates."""


def _would_have_called_ollama(*_args, **_kwargs):
    raise ReachedTheNetwork(
        "a test reached ollama.chat_once -- the annotate ladder's schema-retry "
        "rung dispatches there, so pass a FakeLLM (its .schema_retry stands in) "
        "or patch the seam; this suite makes no network calls")


class FakeLLM:
    """Stand-in for ollama.chat_once_json AND -- via `schema_retry` -- for
    ollama.chat_once, the seam annotate._annotate_schema_retry uses for the
    middle rung of the batch ladder (chat_once_json hardcodes format='json' and
    cannot carry a dict schema). Stubbing only the first left that rung aimed at
    a live localhost Ollama, reachable from any batch_size > 1 request that came
    back missing an id.

    Records the chunk ids of every request, optionally charges a clock, and can
    fail chosen chunks (which is how a dispatched-but-unwritten chunk gets
    tested for re-dispatch).

    The three failure modes are distinct, because the ladder treats them
    differently:
      fail_ids    -- this model never answers these ids, on ANY rung: the batch
                     omits them, the schema retry omits them, the single-chunk
                     request returns None. Handled, counted, skipped.
      batch_drops -- ids only the free-JSON batch omits (the real, measured
                     failure: coverage is bimodal on long event-dense chunks).
                     The schema retry, whose coverage is guaranteed by the
                     grammar, answers them.
      raise_after -- the REQUEST itself blows up mid-pass, and `retry_raises`
                     is the same for the schema rung, which must cost the group
                     its retry rather than the run."""

    def __init__(self, clock=None, seconds=0.0, fail_ids=(), interrupt_after=None,
                 raise_after=None, batch_drops=(), retry_raises=False):
        self.clock = clock
        self.seconds = seconds
        self.fail_ids = set(fail_ids)
        self.interrupt_after = interrupt_after
        self.raise_after = raise_after
        self.batch_drops = set(batch_drops)
        self.retry_raises = retry_raises
        self.requests = []          # ids per request
        self.order = []             # ids in dispatch order, flattened
        self.retry_requests = []    # ids per SCHEMA-RETRY request
        self.retry_schemas = []     # the `format` each retry was given

    def __call__(self, messages, model=None, think=None, timeout=None, **kwargs):
        if self.interrupt_after is not None and len(self.requests) >= self.interrupt_after:
            raise KeyboardInterrupt
        if self.raise_after is not None and len(self.requests) >= self.raise_after:
            raise LLMError("request timed out")
        ids = _ids_in(messages[-1]["content"])
        self.requests.append(ids)
        self.order.extend(ids)
        if self.clock is not None and self.seconds:
            self.clock.advance(self.seconds)
        if len(ids) > 1:
            return {str(cid): dict(_ANNOTATION) for cid in ids
                    if cid not in self.fail_ids and cid not in self.batch_drops}
        if not ids or ids[0] in self.fail_ids:
            return None
        return dict(_ANNOTATION)

    def schema_retry(self, messages, model=None, format=None, timeout=None,
                     think=None, **kwargs):
        """ollama.chat_once as _annotate_schema_retry calls it: returns the
        response's `message` dict, whose content is JSON. Kept out of
        `requests`/`order` -- those are the ids DISPATCHED, and a retry re-asks
        ids the batch already recorded."""
        ids = _ids_in(messages[-1]["content"])
        self.retry_requests.append(ids)
        self.retry_schemas.append(format)
        if self.clock is not None and self.seconds:
            self.clock.advance(self.seconds)
        if self.retry_raises:
            raise LLMError("this build will not take a dict format")
        return {"content": json.dumps({str(cid): dict(_ANNOTATION) for cid in ids
                                       if cid not in self.fail_ids})}


class DBFixture(unittest.TestCase):
    """Bare temp DB with the full schema and annotate.DB_PATH patched at it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "test.db")
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        if _HAS_VEC:
            load_vec_extension(self.con)
        schema.create_schema(self.con)
        patcher = mock.patch.object(annotate, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The seam no FakeLLM installed itself over: a test that reaches it has
        # found a path to a real Ollama, and says so loudly instead of quietly
        # making the request. Individual tests override this (see
        # HybridFixture.hybrid).
        patcher = mock.patch.object(ollama, "chat_once", _would_have_called_ollama)
        patcher.start()
        self.addCleanup(patcher.stop)
        # "Hermetic" has to include the developer's .env. config reads these at
        # import and the tier modules bind their own copies, and each was
        # measured to flip a test in THIS module: ANNOTATE_THINK=true or
        # ANNOTATE_BATCH_SIZE=32 breaks the preset test, ANNOTATE_BUDGET_MINUTES
        # =2 breaks the apple packing test, ANNOTATE_COMMIT_EVERY=1 breaks the
        # seed-pass rollback test. ANNOTATE_TOP_DYADS is the exception and is
        # pinned for the other reason: no LEGAL value flips anything, but a
        # negative one now aborts main() before it dispatches, so an invalid
        # .env would fail these tests for a reason that is not theirs.
        # Pinned to the documented UNSET values so a test asserts
        # the code's behavior and not the machine's; a test wanting another
        # value patches over this (mock.patch stacks).
        for module, knob, value in ((annotate, "ANNOTATE_THINK", ""),
                                    (annotate, "ANNOTATE_BATCH_SIZE", 0),
                                    (annotate, "ANNOTATE_TOP_DYADS", 20),
                                    (annotate, "ANNOTATE_COMMIT_EVERY", 50),
                                    (annotate_distill, "ANNOTATE_COMMIT_EVERY", 50),
                                    (config, "ANNOTATE_COMMIT_EVERY", 50),
                                    (config, "ANNOTATE_BUDGET_MINUTES", 0)):
            patcher = mock.patch.object(module, knob, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def build_vec_table(self, dim=_DIM):
        self.con.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}] distance_metric=cosine,
                chat_id INTEGER,
                first_day INTEGER,
                last_day INTEGER
            )
        """)
        set_state(self.con, "embed_dim", dim)

    def add_vectors(self, rows, seed=17):
        """rows: [(chunk_id, chat_id)]."""
        rng = random.Random(seed)
        self.con.executemany(
            "INSERT OR REPLACE INTO chunk_vectors"
            "(chunk_id, embedding, chat_id, first_day, last_day) "
            "VALUES (?, ?, ?, 1, 1)",
            [(chunk_id, sqlite_vec.serialize_float32(_unit_vector(rng)), chat_id)
             for chunk_id, chat_id in rows],
        )

    def sources(self):
        return dict(self.con.execute(
            "SELECT chunk_id, source FROM chunk_annotations").fetchall())

    def confidence(self, chunk_id):
        return self.con.execute(
            "SELECT knn_confidence FROM chunk_annotations WHERE chunk_id = ?",
            (chunk_id,)).fetchone()[0]

    def n_source(self, source):
        return self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations WHERE source = ?",
            (source,)).fetchone()[0]

    def committed(self):
        """A SECOND connection onto the same file, which can see only what the
        code under test actually COMMITTED. self.con would show its own
        uncommitted writes too, so it cannot answer "did this survive?"."""
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        if _HAS_VEC:
            load_vec_extension(con)
        return con


@_NEEDS_VEC
class HybridFixture(DBFixture):
    """Two chats, `_SEEDS` LLM-annotated chunks plus 40 pending ones, every
    chunk embedded. Chat 1 dominates by message count, so `--top-dyads 1`
    means "chat 1" and chat 2 is the out-of-scope control."""

    def setUp(self):
        super().setUp()
        self.build_vec_table()
        self.con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, 0, 'iMessage')",
            [(1, "chat1"), (2, "chat2")],
        )

        chunks, messages, vectors, annotations = [], [], [], []
        msg_id = 0
        for chunk_id in range(1, _LAST + 1):
            chat_id = 1 if chunk_id < _OTHER_START else 2
            if chunk_id in _EVENT_IDS or chunk_id == _EVENT_IN_OTHER_CHAT:
                bodies = ["thank you so much for everything"]
                if chunk_id == _EVENT_IDS[0]:
                    # Two matching messages in one chunk: the prefilter must
                    # still name the chunk exactly once.
                    bodies.append("congrats on the new job")
            elif chunk_id == 1:
                # An already-'llm' chunk that matches the patterns: the
                # prefilter must not offer to buy its events a second time.
                bodies = ["thanks again, that means a lot"]
            else:
                bodies = [_PLAIN]
            for body in bodies:
                msg_id += 1
                messages.append((msg_id, f"guid-{msg_id}", chat_id, body,
                                 f"2026-01-01T00:{msg_id % 60:02d}:00", chunk_id))
            chunks.append((chunk_id, chat_id, _chunk_text(chunk_id, bodies[0])))
            vectors.append((chunk_id, chat_id))
            if chunk_id <= _SEEDS:
                annotations.append((chunk_id, 0.25, '["joy"]', 2, "[]", "llm"))

        self.con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (?, ?, NULL, NULL, '2026-01-01T00:00:00', "
            "'2026-01-01T00:05:00', 1, ?)", chunks)
        self.con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, chunk_id) VALUES (?, ?, ?, 1, 0, ?, ?, ?)", messages)
        self.add_vectors(vectors)
        self.con.executemany(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (?, ?, ?, ?, ?, ?)", annotations)
        # The pre-existing 'llm' rows stand in for seeds this tier's sampler
        # already drew: hybrid's seed target counts that counter, not every
        # source='llm' row (phase 3 writes those too).
        set_state(self.con, annotate_hybrid._SEED_COUNTER_KEY, str(_SEEDS))
        self.con.commit()

    @contextlib.contextmanager
    def fake_clock(self, clock):
        """Both tier modules read their own module-level `time`, and one run's
        deadlines cross between them -- hybrid_chunks computes the seed
        deadline, annotate_distill's gate reads it -- so a fake clock has to
        replace both or the two halves disagree about what time it is."""
        with mock.patch.object(annotate_hybrid, "time", clock):
            with mock.patch.object(annotate_distill, "time", clock):
                yield

    def hybrid(self, llm, model=_MODEL, **kwargs):
        """Run the tier with batch 1 and one worker by default, so dispatch
        order is the tier's own and not the thread pool's."""
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("seed_target", _SEEDS)
        kwargs.setdefault("top_dyads", 1)
        retry = getattr(llm, "schema_retry", _would_have_called_ollama)
        with mock.patch.object(ollama, "chat_once_json", llm):
            with mock.patch.object(ollama, "chat_once", retry):
                return annotate_hybrid.hybrid_chunks(self.con, model, **kwargs)


# ---------------------------------------------------------------------------
# The wall-clock deadline
# ---------------------------------------------------------------------------

class TestDeadline(HybridFixture):
    def test_deadline_stops_dispatch_early_enough_to_land_inside_the_budget(self):
        # 60s of budget, 25s per request, one worker. Phase 3a owns the first
        # 60% (36s) and dispatches once at t=0 -- the first dispatch of a phase
        # is always ungated, since nothing has been measured yet. At t=25 the
        # gate has an estimate (25s per group, one group in flight) and refuses:
        # 25 + 25 >= 36. 3b then dispatches once at t=25 and refuses at t=50
        # (50 + 25 >= 60). The point of the reserve is the last assertion: the
        # run FINISHES inside the budget instead of overshooting it by a drain.
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=25.0)
        with self.fake_clock(clock):
            written = self.hybrid(llm, budget_minutes=1)

        self.assertEqual(len(llm.order), 2)
        self.assertEqual(llm.order[0], _EVENT_IDS[0])
        self.assertEqual(self.n_source("llm"), _SEEDS + 2)
        # seeds(0) + kNN(40) + 1 event upgrade + 1 confidence upgrade
        self.assertEqual(written, _PENDING_MAIN + _PENDING_OTHER + 2)
        # The candidates the deadline refused are untouched and still upgradable.
        self.assertEqual(self.sources()[_EVENT_IDS[1]], "distilled")
        self.assertLessEqual(clock.now - 1000.0, 60.0)

    def test_deadline_spent_in_the_knn_phases_skips_the_llm_upgrades(self):
        # A budget the kNN phases exhaust leaves nothing for phase 3 -- which
        # must degrade to "no upgrades, rerun", not to an overrun.
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=25.0)
        real_classify = annotate_distill._classify_chunks

        def slow_classify(con, k, dry_run=False, deadline=None, stats=None):
            n = real_classify(con, k, dry_run=dry_run, deadline=deadline,
                              stats=stats)
            clock.advance(120.0)
            return n

        with self.fake_clock(clock):
            with mock.patch.object(annotate_distill, "_classify_chunks", slow_classify):
                written = self.hybrid(llm, budget_minutes=1)

        self.assertEqual(llm.order, [])
        self.assertEqual(written, _PENDING_MAIN + _PENDING_OTHER)
        self.assertEqual(self.n_source("llm"), _SEEDS)

    def test_the_knn_passes_stop_at_the_budget_instead_of_running_on(self):
        # Phase 2 used to be exempt from the deadline on the theory that kNN is
        # cheap. It is O(rows x seeds): 17.9 s for 24,229 rows against an ~800-
        # seed pool but 69.5 s for 20,613 against a 4,416-seed one, and a 1-min
        # budget spent 1.1 min inside it and bought no upgrades at all. Both
        # passes are resumable, so the gate only defers work to the next run.
        clock = FakeClock()
        with self.fake_clock(clock):
            stats = {"stopped": False, "interrupted": False}
            n = annotate_distill._classify_chunks(
                self.con, config.ANNOTATE_KNN_K, deadline=clock.now, stats=stats)
        self.assertEqual(n, 0)
        self.assertTrue(stats["stopped"])
        self.assertEqual(self.n_source("distilled"), 0)

        # ...and with clock left, it classifies as before.
        stats = {"stopped": False, "interrupted": False}
        n = annotate_distill._classify_chunks(
            self.con, config.ANNOTATE_KNN_K, stats=stats)
        self.assertEqual(n, _PENDING_MAIN + _PENDING_OTHER)
        self.assertFalse(stats["stopped"])

    def test_the_confidence_backfill_stops_at_the_budget_too(self):
        annotate_distill._classify_chunks(self.con, config.ANNOTATE_KNN_K)
        self.con.execute("UPDATE chunk_annotations SET knn_confidence = NULL")
        self.con.commit()
        clock = FakeClock()
        with self.fake_clock(clock):
            stats = {"stopped": False, "interrupted": False}
            n = annotate_distill.backfill_confidence(
                self.con, config.ANNOTATE_KNN_K, deadline=clock.now, stats=stats)
        self.assertEqual(n, 0)
        self.assertTrue(stats["stopped"])
        self.assertIsNone(self.confidence(_EVENT_IDS[0]))

    def test_a_knn_phase_out_of_budget_skips_the_upgrades_not_the_reverse(self):
        # A stopped (not interrupted) phase 2 must land on the "budget spent"
        # report -- the bug to avoid is gating phase 2 and having its leftovers
        # read as an interrupt, which would disable every LLM upgrade forever.
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=25.0)
        real_classify = annotate_distill._classify_chunks

        def stopped_classify(con, k, dry_run=False, deadline=None, stats=None):
            n = real_classify(con, k, dry_run=dry_run, stats=stats)
            clock.advance(120.0)
            if stats is not None:
                stats["stopped"] = True
            return n

        with self.fake_clock(clock):
            with mock.patch.object(annotate_distill, "_classify_chunks",
                                   stopped_classify):
                with mock.patch("builtins.print") as printed:
                    self.hybrid(llm, budget_minutes=1)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertIn("the budget was spent on phases 1-2", said)
        self.assertNotIn("were interrupted", said)
        self.assertEqual(llm.order, [])

    def test_the_seed_pass_gets_a_bounded_share_of_the_budget(self):
        # Phase 1 is LLM work, not preamble: measured unbounded it spent 10.7
        # min of an 8-min budget and phase 3 never ran. It now stops at
        # _SEED_PHASE_SHARE of the clock (15s of this 60s budget) -- one
        # ungated dispatch at t=0, refused at t=25 -- and says so.
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=25.0)
        with self.fake_clock(clock):
            self.hybrid(llm, seed_target=_SEEDS + 5, budget_minutes=1, limit=0)
        self.assertEqual(len(llm.order), 1)
        self.assertEqual(self.n_source("llm"), _SEEDS + 1)

    def test_a_seed_pass_cut_short_below_the_knn_floor_degrades_cleanly(self):
        # A budget too small to reach annotate_distill._KNN_MIN_SEEDS leaves the
        # kNN vote refusing (rightly -- it would be noise). For a tier whose
        # contract is "spend N minutes and commit what you got" that has to be a
        # report, not a SystemExit.
        self.con.execute("DELETE FROM chunk_annotations")
        self.con.commit()
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=25.0)
        with self.fake_clock(clock):
            written = self.hybrid(llm, seed_target=200, budget_minutes=1)
        self.assertEqual(written, 1)                     # the one seed it bought
        self.assertEqual(self.n_source("llm"), 1)
        self.assertEqual(self.n_source("distilled"), 0)  # the floor still holds

    def test_an_expensive_event_prefilter_does_not_eat_phase_3a_s_share(self):
        # event_candidate_chunk_ids is the run's one unavoidable corpus-wide
        # `messages` scan, and 3a's share used to be derived from a clock read
        # BEFORE it: the scan was charged to the phase, which then got a
        # fraction of the 60% its own header line claimed. Here the scan costs
        # half the 60s budget, so the two readings differ by 12s -- enough for
        # two more 5s requests, i.e. every candidate instead of one.
        self.hybrid(FakeLLM(), limit=0)          # seeds + kNN, no upgrades
        clock = FakeClock()
        llm = FakeLLM(clock=clock, seconds=5.0)
        real_prefilter = annotate_events.event_candidate_chunk_ids

        def slow_prefilter(con, all_chats, top_dyads):
            ids = real_prefilter(con, all_chats, top_dyads)
            clock.advance(30.0)
            return ids

        with self.fake_clock(clock):
            with mock.patch.object(annotate_events, "event_candidate_chunk_ids",
                                   slow_prefilter):
                with mock.patch("builtins.print") as printed:
                    self.hybrid(llm, budget_minutes=1)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        # Stale `now`: deadline 1036, dispatches at t=1030 only. Re-read after
        # the scan: deadline 1048, dispatches at t=1030/1035/1040.
        self.assertIn("event upgrades 3 (0 skipped)", said)
        self.assertEqual(llm.order[:3], _EVENT_IDS)
        # ...and the budget as a whole is still respected.
        self.assertLessEqual(clock.now - 1000.0, 60.0)

    def test_a_missing_chunk_vectors_table_still_exits(self):
        # The softened exit is only for the seed floor; a DB with no embeddings
        # at all is a misconfiguration no rerun fixes and must still fail loudly.
        self.con.execute("DROP TABLE chunk_vectors")
        self.con.commit()
        with self.assertRaises(SystemExit):
            self.hybrid(FakeLLM(), limit=0)


# ---------------------------------------------------------------------------
# Phase 3a -> 3b ordering, dedup, scope
# ---------------------------------------------------------------------------

class TestPhaseThree(HybridFixture):
    def seed_and_classify(self, model=_MODEL):
        """A first pass with limit=0: seeds + kNN only, no LLM upgrades."""
        llm = FakeLLM()
        self.hybrid(llm, model=model, limit=0)
        self.assertEqual(llm.order, [])
        return llm

    def test_events_are_upgraded_before_low_confidence_rows(self):
        self.seed_and_classify()
        events = annotate_events.event_candidate_chunk_ids(self.con, False, 1)
        low_conf = [row[0] for row in
                    annotate_hybrid._low_confidence_rows(self.con, False, 1)]
        self.assertEqual(events, _EVENT_IDS)

        llm = FakeLLM()
        self.hybrid(llm)
        expected = events + [cid for cid in low_conf if cid not in events]
        self.assertEqual(llm.order, expected)
        # ...and phase 3b really is confidence-ordered, not id-ordered.
        self.assertNotEqual(llm.order, sorted(llm.order))

    def test_no_chunk_is_dispatched_twice_in_one_run(self):
        self.seed_and_classify()
        llm = FakeLLM()
        self.hybrid(llm)
        self.assertEqual(len(llm.order), len(set(llm.order)))
        self.assertEqual(len(llm.order), _PENDING_MAIN)

    def test_event_chunk_the_llm_failed_is_not_resent_in_phase_3b(self):
        # A skipped chunk stays source='distilled' with its confidence intact,
        # so phase 3b's query still offers it -- only the in-run `seen` set
        # keeps the budget from buying the same failing request twice.
        self.seed_and_classify()
        llm = FakeLLM(fail_ids=[_EVENT_IDS[0]])
        self.hybrid(llm)
        self.assertEqual(llm.order.count(_EVENT_IDS[0]), 1)
        self.assertEqual(len(llm.order), len(set(llm.order)))
        self.assertEqual(self.sources()[_EVENT_IDS[0]], "distilled")

    def test_a_batch_that_drops_an_id_buys_one_schema_retry_not_singles(self):
        # The middle rung of the ladder, and the ONE annotate request that
        # reaches ollama.chat_once rather than chat_once_json. Free-JSON
        # coverage is bimodal on long event-dense chunks, and every miss used to
        # cost its own single-chunk request; the schema retry caps that at one
        # extra request whose coverage the grammar guarantees. The rung's own
        # contract is pinned at the unit level in tests/test_annotate.py
        # (TestAnnotateGroupLadder); what is new here is reaching it THROUGH the
        # tier, which is what noticed that this module's fixture left the seam
        # unstubbed and pointed at a live Ollama.
        self.seed_and_classify()
        llm = FakeLLM(batch_drops=[_EVENT_IDS[1]])
        self.hybrid(llm, batch_size=8, limit=3)
        # One batched request for the whole group, then one retry naming
        # exactly the id it missed -- and NO single-chunk request after it.
        self.assertEqual(llm.requests, [_EVENT_IDS])
        self.assertEqual(llm.retry_requests, [[_EVENT_IDS[1]]])
        self.assertEqual(set(llm.retry_schemas[0]["properties"]),
                         {str(_EVENT_IDS[1])})
        for chunk_id in _EVENT_IDS:
            self.assertEqual(self.sources()[chunk_id], "llm")

    def test_a_schema_retry_that_fails_costs_the_group_its_retry_not_the_run(self):
        # An Ollama build that will not take a dict `format`, or a grammar that
        # runs past the timeout: _annotate_schema_retry swallows all of it and
        # the per-chunk rung underneath still completes the group.
        self.seed_and_classify()
        llm = FakeLLM(batch_drops=[_EVENT_IDS[1]], retry_raises=True)
        self.hybrid(llm, batch_size=8, limit=3)
        self.assertEqual(len(llm.retry_requests), 1)
        self.assertEqual(llm.requests, [_EVENT_IDS, [_EVENT_IDS[1]]])
        for chunk_id in _EVENT_IDS:
            self.assertEqual(self.sources()[chunk_id], "llm")

    def test_an_id_no_rung_can_answer_is_skipped_not_written(self):
        # batch_size > 1 with a chunk the model never answers: the batch omits
        # it, the schema retry omits it, the single-chunk request comes back
        # unparseable. It stays distilled and costs no write -- the case that
        # would have made a live request from a suite that promises none.
        self.seed_and_classify()
        llm = FakeLLM(fail_ids=[_EVENT_IDS[1]])
        self.hybrid(llm, batch_size=8, limit=3)
        self.assertEqual(llm.retry_requests, [[_EVENT_IDS[1]]])
        self.assertEqual(llm.requests[:2], [_EVENT_IDS, [_EVENT_IDS[1]]])
        self.assertEqual(self.sources()[_EVENT_IDS[1]], "distilled")
        self.assertEqual(self.sources()[_EVENT_IDS[0]], "llm")

    def test_limit_caps_the_total_across_both_sub_phases(self):
        self.seed_and_classify()
        llm = FakeLLM()
        self.hybrid(llm, limit=2)
        self.assertEqual(llm.order, _EVENT_IDS[:2])

    def test_limit_counts_upgrades_written_not_requests_dispatched(self):
        # --limit is a work-product cap, not a spend cap (the wall clock is the
        # spend cap), so a dispatch that came back unparseable -- budget spent,
        # nothing written -- must not consume it. Counting `seen`, which holds
        # every DISPATCHED chunk, made --limit 2 deliver one upgrade.
        self.seed_and_classify()
        llm = FakeLLM(fail_ids=[_EVENT_IDS[0]])
        written = self.hybrid(llm, limit=2)
        # Three requests: two event candidates (the first unusable) and one
        # low-confidence row that phase 3b buys to make the count up to 2.
        self.assertEqual(llm.order[:2], _EVENT_IDS[:2])
        self.assertEqual(len(llm.order), 3)
        self.assertEqual(written, 2)
        self.assertEqual(self.n_source("llm"), _SEEDS + 2)
        # ...and the skipped one is untouched, still distilled, still upgradable.
        self.assertEqual(self.sources()[_EVENT_IDS[0]], "distilled")

    def test_the_event_fetch_is_capped_and_says_what_it_left(self):
        # _chunk_rows materializes every candidate's full TEXT, and the
        # prefilter's ~19-20% hit rate makes that ~50k rows on a corpus 10x the
        # reference -- ~95 MB held to dispatch a few hundred. The cap is a
        # prefix of a density ranking, so it drops only rows no budget could
        # have reached; the repo takes no silent caps, so it prints them.
        self.seed_and_classify()
        fetched = []
        real_fetch = annotate_hybrid._chunk_rows

        def spy(con, chunk_ids):
            fetched.append(list(chunk_ids))
            return real_fetch(con, chunk_ids)

        with mock.patch.object(annotate_hybrid, "_EVENT_FETCH_CAP", 2):
            with mock.patch.object(annotate_hybrid, "_chunk_rows", spy):
                with mock.patch("builtins.print") as printed:
                    self.hybrid(FakeLLM())
        self.assertEqual(fetched, [_EVENT_IDS[:2]])
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        # The header still reports the true candidate count, not the cap.
        self.assertIn("phase 3a: 3 event-candidate chunks", said)
        self.assertIn("fetching the densest 2 of them; 1 left for a later run",
                      said)
        self.assertIn("fetch cap", said)                 # named by cause

    def test_what_the_cap_left_is_counted_against_the_true_candidate_total(self):
        # The drop count used to be derived AFTER --limit's prefix, so it
        # reported only what the CAP dropped out of an already-shortened list
        # and understated the backlog by however much --limit had cut -- by an
        # unbounded amount (50,000 candidates, --limit 5,000, cap 2,000 printed
        # "3,000 left" where 48,000 were). Both prefixes shorten the same
        # densest-first ranking, so they are interchangeable for WHAT runs; the
        # reporting is what has to name each one's share.
        self.seed_and_classify()
        with mock.patch.object(annotate_hybrid, "_EVENT_FETCH_CAP", 1):
            with mock.patch("builtins.print") as printed:
                self.hybrid(FakeLLM(), limit=2)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertIn("phase 3a: 3 event-candidate chunks", said)
        # 1 fetched of 3 candidates: --limit cut one, the cap cut one more.
        self.assertIn("fetching the densest 1 of them; 2 left for a later run",
                      said)
        self.assertIn("--limit 2", said)
        self.assertIn("fetch cap", said)
        # ...and the run does not sign off as having finished the scope.
        self.assertNotIn("finished every candidate in scope", said)
        self.assertIn("candidates left over", said)

    def test_a_dry_run_does_not_claim_a_fetch_it_never_makes(self):
        # The fetch is short-circuited on a dry run (event_rows = []), but the
        # cap notice printed "fetching ..." regardless -- the one line on a
        # preview describing an action the preview provably does not perform.
        # The id preview also came off the CAPPED list, so a cap under 10 clipped
        # the ten ids it advertises.
        self.seed_and_classify()
        fetched = []
        with mock.patch.object(annotate_hybrid, "_EVENT_FETCH_CAP", 1):
            with mock.patch.object(annotate_hybrid, "_chunk_rows",
                                   lambda con, ids: fetched.append(list(ids))):
                with mock.patch("builtins.print") as printed:
                    self.hybrid(FakeLLM(), dry_run=True)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertEqual(fetched, [])
        self.assertIn("would fetch the densest 1 of them", said)
        self.assertNotIn("fetching the densest", said)
        # All three candidates previewed, not the one the (unpaid) cap allows.
        self.assertIn(f"--dry-run: first chunk ids: "
                      f"{', '.join(str(c) for c in _EVENT_IDS)}", said)

    def test_a_negative_limit_is_refused_rather_than_inverted(self):
        # `event_ids[:-1]` is not a prefix of the density ranking -- it drops the
        # LEAST dense candidate and keeps the rest -- and `max(0, -1 - done)`
        # then switches phase 3b off, so a stray minus bought a run nobody asked
        # for. Same rule as the negative budget: refuse, naming the flag.
        self.seed_and_classify()
        llm = FakeLLM()
        with self.assertRaises(SystemExit) as cm:
            self.hybrid(llm, limit=-1)
        self.assertIn("--limit", str(cm.exception))
        self.assertEqual(llm.order, [])
        # 0 is still a legitimate (empty) run, not an error.
        self.assertEqual(self.hybrid(FakeLLM(), limit=0), 0)

    def test_ctrl_c_commits_what_landed_and_returns_normally(self):
        # Ctrl-C is the other way phase 3 ends early, and it has to behave like
        # the deadline: commit, report, exit 0 -- never lose the upgrades the
        # budget already paid for.
        self.seed_and_classify()
        llm = FakeLLM(interrupt_after=2)
        written = self.hybrid(llm)
        self.assertEqual(llm.order, _EVENT_IDS[:2])
        self.assertEqual(written, 2)
        self.assertEqual(self.n_source("llm"), _SEEDS + 2)

    def test_scope_keeps_phase_3_inside_the_top_dyads(self):
        self.seed_and_classify()
        llm = FakeLLM()
        self.hybrid(llm)
        self.assertTrue(all(cid < _OTHER_START for cid in llm.order))
        # The other chat's chunks are still kNN-classified (phases 1-2 are
        # corpus-wide by design); they just never cost an LLM call.
        self.assertEqual(self.sources()[_EVENT_IN_OTHER_CHAT], "distilled")

    def test_model_guard_still_refuses_to_mix_annotation_models(self):
        # Phase 1 guards its own writes but returns early once the seed pool is
        # full, so phase 3 has to run the guard itself -- otherwise hybrid
        # would be the one tier that could quietly mix models into the table.
        self.seed_and_classify()
        self.assertEqual(get_state(self.con, "annotate_model"), _MODEL)
        llm = FakeLLM()
        with self.assertRaises(SystemExit):
            self.hybrid(llm, model="some-other-model")
        self.assertEqual(llm.order, [])

    def test_the_model_guard_refuses_before_any_phase_runs(self):
        # The guard used to sit at phase 3a. Phase 1 returns before its own
        # guard call once the seed pool is at target, so a switched
        # ANNOTATE_MODEL was refused only after the two corpus-wide kNN phases
        # had spent up to half the budget on a run that was about to abort.
        # Nothing here has been classified yet: the fixture's rows are all
        # 'llm' seeds, so a phase 2 that ran would be visible as 40 distilled
        # rows.
        set_state(self.con, "annotate_model", _MODEL)
        self.con.commit()
        llm = FakeLLM()
        with mock.patch("builtins.print") as printed:
            with self.assertRaises(SystemExit):
                self.hybrid(llm, model="some-other-model")
        self.assertEqual(llm.order, [])
        self.assertEqual(self.n_source("distilled"), 0)
        self.assertEqual(self.n_source("llm"), _SEEDS)
        # The refusal is the FIRST thing said -- no budget header, no phase
        # lines ahead of it -- and it did not overwrite the recorded model.
        said = [str(a) for call in printed.call_args_list for a in call.args]
        self.assertIn("the model is now some-other-model", said[0])
        self.assertEqual(get_state(self.con, "annotate_model"), _MODEL)

    def test_apple_groups_phase_3_by_char_budget_not_row_count(self):
        # The apple backend's fixed 4,096-token window makes a row count the
        # wrong batching unit, and hybrid has to take the same branch every
        # other tier does (annotate._annotation_groups) rather than its
        # batch_size.
        self.seed_and_classify(model="apple")
        llm = FakeLLM()
        self.hybrid(llm, model="apple", batch_size=1)
        # These chunk texts are ~50 chars, so each sub-phase packs into one
        # request instead of the 30 that batch_size=1 would have sent.
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(llm.requests[0], _EVENT_IDS)
        self.assertEqual(len(llm.order), _PENDING_MAIN)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume(HybridFixture):
    def test_second_run_llm_annotates_nothing_twice(self):
        first = FakeLLM()
        self.hybrid(first, all_chats=True)
        self.assertEqual(len(first.order), _PENDING_MAIN + _PENDING_OTHER)

        second = FakeLLM()
        self.hybrid(second, all_chats=True)
        self.assertEqual(second.order, [])
        self.assertEqual(self.n_source("llm"),
                         _SEEDS + _PENDING_MAIN + _PENDING_OTHER)
        self.assertEqual(self.n_source("distilled"), 0)

    def test_seed_target_counts_the_seeds_this_tier_drew(self):
        llm = FakeLLM()
        self.hybrid(llm, seed_target=_SEEDS + 5, limit=0)
        self.assertEqual(len(llm.order), 5)                 # the shortfall only
        self.assertEqual(self.n_source("llm"), _SEEDS + 5)

        again = FakeLLM()
        self.hybrid(again, seed_target=_SEEDS + 5, limit=0)
        self.assertEqual(again.order, [])

    def test_phase_three_upgrades_do_not_count_toward_the_seed_target(self):
        # Phase 3 writes source='llm' too, so counting every such row made the
        # tier's own output satisfy its own seed target: on the reference
        # corpus (4,416 'llm' rows against a target of 800) phase 1 printed
        # "labeling 0 more" on every run and the stratified sampler -- the
        # whole point of the knob -- never executed again.
        llm = FakeLLM()
        self.hybrid(llm, seed_target=_SEEDS, limit=5)
        self.assertEqual(len(llm.order), 5)                 # upgrades, not seeds
        self.assertEqual(self.n_source("llm"), _SEEDS + 5)
        self.assertEqual(int(get_state(self.con, annotate_hybrid._SEED_COUNTER_KEY)),
                         _SEEDS)

        again = FakeLLM()
        self.hybrid(again, seed_target=_SEEDS + 3, limit=0)
        self.assertEqual(len(again.order), 3)               # the sampler still runs

    def test_the_seed_counter_never_claims_more_than_the_rows_that_exist(self):
        # An emptied chunk_annotations (reset_db also clears ingest_state, but
        # a hand-cleared table would not) must not leave the counter vouching
        # for seeds that are gone.
        self.con.execute("DELETE FROM chunk_annotations")
        self.con.commit()
        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            n = annotate_distill._seed_chunks(
                self.con, _MODEL, False, 1, 3,
                sampler=lambda con, pending, need: pending[:need],
                counter_key=annotate_hybrid._SEED_COUNTER_KEY)
        self.assertEqual(n, 3)
        self.assertEqual(int(get_state(self.con, annotate_hybrid._SEED_COUNTER_KEY)), 3)

    def test_ctrl_c_in_the_seed_pass_reports_and_returns_like_every_phase(self):
        # Phase 1 used to raise SystemExit(1) while phases 2 and 3 returned 0,
        # so one Ctrl-C had three meanings depending on where it landed.
        llm = FakeLLM(interrupt_after=2)
        written = self.hybrid(llm, seed_target=_SEEDS + 5, limit=0)
        self.assertEqual(written, 2)
        self.assertEqual(self.n_source("llm"), _SEEDS + 2)
        self.assertEqual(self.n_source("distilled"), 0)     # phase 2 never ran


# ---------------------------------------------------------------------------
# knn_confidence
# ---------------------------------------------------------------------------

class TestKnnConfidence(HybridFixture):
    def test_classify_writes_confidence_and_llm_rows_leave_it_null(self):
        annotate_distill._classify_chunks(self.con, config.ANNOTATE_KNN_K)
        rows = self.con.execute(
            "SELECT knn_confidence FROM chunk_annotations WHERE source = 'distilled'"
        ).fetchall()
        self.assertEqual(len(rows), _PENDING_MAIN + _PENDING_OTHER)
        for (conf,) in rows:
            self.assertIsNotNone(conf)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
        (n_llm_scored,) = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations "
            "WHERE source = 'llm' AND knn_confidence IS NOT NULL").fetchone()
        self.assertEqual(n_llm_scored, 0)

    def test_llm_upgrade_clears_the_distilled_row_confidence(self):
        llm = FakeLLM()
        self.hybrid(llm, limit=0)
        self.assertIsNotNone(self.confidence(_EVENT_IDS[0]))
        self.hybrid(FakeLLM(), limit=1)
        self.assertEqual(self.sources()[_EVENT_IDS[0]], "llm")
        self.assertIsNone(self.confidence(_EVENT_IDS[0]))

    def test_knn_predict_returns_a_bounded_confidence_without_neighbours(self):
        # The empty-neighbour early return still has to satisfy the column's
        # 0..1 contract (the hybrid tier sorts on it).
        self.con.execute("CREATE VIRTUAL TABLE temp.seed_vectors USING vec0("
                         f"chunk_id integer PRIMARY KEY, embedding FLOAT[{_DIM}] "
                         "distance_metric=cosine)")
        emb = sqlite_vec.serialize_float32([0.0] * (_DIM - 1) + [1.0])
        valence, emotions, depth, conf = annotate_distill._knn_predict(
            self.con, {}, emb, 5)
        self.assertEqual((valence, emotions, depth), (0.0, [], 1))
        self.assertEqual(conf, 0.0)


class TestBackfillConfidence(HybridFixture):
    def classify_then_blank(self):
        """A corpus in the state the backfill exists for: distilled rows
        written before knn_confidence existed."""
        annotate_distill._classify_chunks(self.con, config.ANNOTATE_KNN_K)
        self.con.execute("UPDATE chunk_annotations SET knn_confidence = NULL")
        self.con.commit()

    def labels(self):
        return self.con.execute(
            "SELECT chunk_id, valence, emotions_json, depth, events_json, source "
            "FROM chunk_annotations ORDER BY chunk_id").fetchall()

    def test_backfill_fills_only_confidence(self):
        self.classify_then_blank()
        before = self.labels()
        n = annotate_distill.backfill_confidence(self.con, config.ANNOTATE_KNN_K)
        self.assertEqual(n, _PENDING_MAIN + _PENDING_OTHER)
        self.assertEqual(self.labels(), before)
        (n_null,) = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations "
            "WHERE source = 'distilled' AND knn_confidence IS NULL").fetchone()
        self.assertEqual(n_null, 0)
        # 'llm' rows are not distilled rows: the backfill must not score them.
        (n_llm_scored,) = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations "
            "WHERE source = 'llm' AND knn_confidence IS NOT NULL").fetchone()
        self.assertEqual(n_llm_scored, 0)

    def test_backfill_skips_rows_without_an_embedding(self):
        self.classify_then_blank()
        orphan = _LAST + 1
        self.con.execute(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (?, 1, NULL, NULL, '2026-01-01T00:00:00', "
            "'2026-01-01T00:05:00', 1, 'no vector here')", (orphan,))
        self.con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, "
            "depth, events_json, source) VALUES (?, 0.0, '[]', 1, '[]', "
            "'distilled')", (orphan,))
        self.con.commit()
        n = annotate_distill.backfill_confidence(self.con, config.ANNOTATE_KNN_K)
        self.assertEqual(n, _PENDING_MAIN + _PENDING_OTHER)   # not the orphan
        self.assertIsNone(self.confidence(orphan))

    def test_backfill_is_a_no_op_on_a_second_run(self):
        self.classify_then_blank()
        annotate_distill.backfill_confidence(self.con, config.ANNOTATE_KNN_K)
        scored = dict(self.con.execute(
            "SELECT chunk_id, knn_confidence FROM chunk_annotations").fetchall())
        n = annotate_distill.backfill_confidence(self.con, config.ANNOTATE_KNN_K)
        self.assertEqual(n, 0)
        self.assertEqual(dict(self.con.execute(
            "SELECT chunk_id, knn_confidence FROM chunk_annotations").fetchall()),
            scored)

    def test_backfill_dry_run_writes_nothing(self):
        self.classify_then_blank()
        n = annotate_distill.backfill_confidence(
            self.con, config.ANNOTATE_KNN_K, dry_run=True)
        self.assertEqual(n, _PENDING_MAIN + _PENDING_OTHER)
        (n_null,) = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations "
            "WHERE knn_confidence IS NULL").fetchone()
        self.assertEqual(n_null, _SEEDS + _PENDING_MAIN + _PENDING_OTHER)


# ---------------------------------------------------------------------------
# ANNOTATE_KNN_INDEX_CAP -- the ceiling on the kNN index (not the seed pool)
# ---------------------------------------------------------------------------

class TestSeedIndexSample(unittest.TestCase):
    """annotate_distill._sample_seed_ids, the pure half of the cap. Both kNN
    passes cost O(rows x seeds) and the hybrid tier widens the seed POOL on
    every run, so the INDEX gets a ceiling instead."""

    POOL = list(range(1, 501))

    def test_a_pool_that_already_fits_is_kept_whole(self):
        self.assertEqual(annotate_distill._sample_seed_ids(self.POOL, 800),
                         set(self.POOL))
        self.assertEqual(annotate_distill._sample_seed_ids(self.POOL, len(self.POOL)),
                         set(self.POOL))

    def test_a_cap_of_zero_or_less_is_uncapped(self):
        for cap in (0, -1):
            with self.subTest(cap=cap):
                self.assertEqual(annotate_distill._sample_seed_ids(self.POOL, cap),
                                 set(self.POOL))

    def test_the_draw_is_deterministic_and_independent_of_row_order(self):
        # Both passes of one run rebuild the index and must vote off the same
        # one, so the draw is a pure function of the ids -- not of the order
        # the caller's SELECT happened to return them in.
        first = annotate_distill._sample_seed_ids(self.POOL, 200)
        second = annotate_distill._sample_seed_ids(list(reversed(self.POOL)), 200)
        self.assertEqual(len(first), 200)
        self.assertEqual(first, second)
        self.assertTrue(first.issubset(self.POOL))

    def test_the_draw_does_not_move_with_the_module_rng(self):
        # random.Random(0), not the global RNG _seed_chunks samples from:
        # consuming or reseeding that one here would change the SEED draw as a
        # side effect, and would make this sample depend on how much
        # randomness the process had already spent.
        random.seed(11)
        first = annotate_distill._sample_seed_ids(self.POOL, 200)
        random.seed(22)
        random.random()
        second = annotate_distill._sample_seed_ids(self.POOL, 200)
        self.assertEqual(first, second)

    def test_a_cap_below_the_floor_is_raised_to_it(self):
        # A speed knob must not be able to take the vote below the minimum the
        # pool check itself enforces.
        kept = annotate_distill._sample_seed_ids(self.POOL, 5)
        self.assertEqual(len(kept), annotate_distill._KNN_MIN_SEEDS)

    def test_the_floor_never_invents_seeds_a_short_pool_lacks(self):
        short = list(range(1, 21))
        self.assertEqual(annotate_distill._sample_seed_ids(short, 5), set(short))


class TestCorpusProportionalIndex(unittest.TestCase):
    """The cap picks WHICH rows by each chat's share of the CORPUS, not
    uniformly over the pool. The pool is whatever earlier runs happened to
    label -- on both real corpora overwhelmingly one dyad -- so a uniform draw
    handed most of the corpus a vote from a chat it has nothing to do with."""

    # 1,000 pool rows: 700 in the dominant chat, 30 in each of 10 others --
    # the shape both real corpora have (one chat at 71-100% of the pool).
    def setUp(self):
        self.pool = list(range(1, 1001))
        self.chat_of = {cid: (1 if cid <= 700 else 2 + (cid - 701) // 30)
                        for cid in self.pool}
        # ...but that dominant chat is only 30% of the CORPUS.
        self.corpus = {1: 3000, **{c: 700 for c in range(2, 12)}}

    def _by_chat(self, kept):
        out = {}
        for cid in kept:
            out[self.chat_of[cid]] = out.get(self.chat_of[cid], 0) + 1
        return out

    def test_the_draw_follows_corpus_share_not_pool_share(self):
        uniform = self._by_chat(
            annotate_distill._sample_seed_ids(self.pool, 200))
        kept = annotate_distill._sample_seed_ids(
            self.pool, 200, chat_of=self.chat_of, corpus_sizes=self.corpus)
        self.assertEqual(len(kept), 200)
        by_chat = self._by_chat(kept)
        # The uniform draw tracks the POOL, so the dominant chat takes ~70%.
        self.assertGreater(uniform[1], 120)
        # Corpus share is 3000/10000, so it takes 30% -- and every other chat
        # gets its own 700/10000 = 14 slots rather than whatever chance left.
        self.assertEqual(by_chat[1], 60)
        self.assertEqual(sorted(by_chat)[1:], list(range(2, 12)))
        self.assertTrue(all(by_chat[c] == 14 for c in range(2, 12)), by_chat)

    def test_every_chat_in_the_pool_gets_a_seat(self):
        # The uniform draw leaves small chats to chance; proportional
        # allocation gives each one its share, which is the whole point.
        kept = annotate_distill._sample_seed_ids(
            self.pool, 200, chat_of=self.chat_of, corpus_sizes=self.corpus)
        self.assertEqual(set(self._by_chat(kept)), set(range(1, 12)))

    def test_a_starved_chat_hands_its_slots_back(self):
        # Chat 1 is 99% of the corpus but the pool holds only 10 of its rows:
        # the quota must not silently shrink the index below the cap.
        pool = list(range(1, 111))
        chat_of = {cid: (1 if cid <= 10 else 2) for cid in pool}
        kept = annotate_distill._sample_seed_ids(
            pool, 100, chat_of=chat_of, corpus_sizes={1: 99000, 2: 1000})
        self.assertEqual(len(kept), 100)

    def test_it_is_deterministic_and_order_independent(self):
        # Both passes of one run rebuild the index and must vote off the same
        # one -- now across two extra mappings whose order must not matter.
        first = annotate_distill._sample_seed_ids(
            self.pool, 200, chat_of=self.chat_of, corpus_sizes=self.corpus)
        random.seed(31)
        random.random()
        second = annotate_distill._sample_seed_ids(
            list(reversed(self.pool)),
            200,
            chat_of=dict(reversed(list(self.chat_of.items()))),
            corpus_sizes=dict(reversed(list(self.corpus.items()))))
        self.assertEqual(first, second)

    def test_the_floor_still_holds_with_chat_information(self):
        kept = annotate_distill._sample_seed_ids(
            self.pool, 5, chat_of=self.chat_of, corpus_sizes=self.corpus)
        self.assertEqual(len(kept), annotate_distill._KNN_MIN_SEEDS)

    def test_without_chat_information_the_uniform_draw_is_unchanged(self):
        # Every existing caller and test still exercises this path; it must be
        # byte-identical to what shipped before.
        for cap in (0, -1, 5, 200, 800, 5000):
            with self.subTest(cap=cap):
                self.assertEqual(
                    annotate_distill._sample_seed_ids(self.pool, cap),
                    annotate_distill._sample_seed_ids(
                        self.pool, cap, chat_of=None, corpus_sizes=None))

    def test_a_null_chat_id_does_not_break_the_ordering(self):
        # chunks.chat_id is nullable, and None cannot be compared to an int.
        chat_of = dict(self.chat_of)
        for cid in range(1, 51):
            chat_of[cid] = None
        kept = annotate_distill._sample_seed_ids(
            self.pool, 200, chat_of=chat_of,
            corpus_sizes={None: 500, **self.corpus})
        self.assertEqual(len(kept), 200)


@_NEEDS_VEC
class TestSeedIndexCap(HybridFixture):
    """_build_seed_index applying that cap: to the temp vec0 index AND to the
    label dict, from the same list -- _knn_predict looks its neighbours up in
    that dict, so an id in one and not the other is a KeyError or a lost
    vote."""

    def build(self, cap):
        """(labels, indexed ids, what it printed) for one build at `cap`.

        _last_index_report is reset per build and restored after: it is process
        state whose only job is to suppress the SECOND identical report inside
        one run, and every test here but one is asserting what a FIRST build
        says. The one that cares about the suppression drives it directly."""
        with mock.patch.object(annotate_distill, "ANNOTATE_KNN_INDEX_CAP", cap):
            with mock.patch.object(annotate_distill, "_last_index_report", None):
                with mock.patch("builtins.print") as printed:
                    labels = annotate_distill._build_seed_index(self.con)
        indexed = {row[0] for row in
                   self.con.execute("SELECT chunk_id FROM temp.seed_vectors")}
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        return labels, indexed, said

    def test_the_cap_bounds_the_index_and_says_so(self):
        # _SEEDS is pinned above _INDEX_CAP, so this asserts the cap biting --
        # never a pool that happened to fit under it.
        self.assertGreater(_SEEDS, _INDEX_CAP)
        labels, indexed, said = self.build(_INDEX_CAP)
        self.assertEqual(len(indexed), _INDEX_CAP)
        self.assertEqual(said, f"using {_INDEX_CAP} of {_SEEDS:,} seeds "
                               f"for the kNN index")
        # ...the POOL is untouched: every seed is still a stored annotation.
        self.assertEqual(self.n_source("llm"), _SEEDS)
        self.assertEqual(set(labels), indexed)

    def test_both_passes_of_a_run_vote_off_the_same_index(self):
        first_labels, first_indexed, _ = self.build(_INDEX_CAP)
        second_labels, second_indexed, _ = self.build(_INDEX_CAP)
        self.assertEqual(first_indexed, second_indexed)
        self.assertEqual(first_labels, second_labels)

    def test_the_second_pass_of_a_run_does_not_repeat_the_index_line(self):
        # _classify_chunks and backfill_confidence each rebuild the index inside
        # one hybrid run, so the unguarded print put the same sentence on screen
        # twice, wedged between two different progress bars. It is news once.
        with mock.patch.object(annotate_distill, "ANNOTATE_KNN_INDEX_CAP", _INDEX_CAP):
            with mock.patch.object(annotate_distill, "_last_index_report", None):
                with mock.patch("builtins.print") as printed:
                    annotate_distill._build_seed_index(self.con)
                    annotate_distill._build_seed_index(self.con)
        said = [str(a) for call in printed.call_args_list for a in call.args]
        self.assertEqual([line for line in said if "kNN index" in line],
                         [f"using {_INDEX_CAP} of {_SEEDS:,} seeds "
                          f"for the kNN index"])

    def test_a_cap_of_zero_or_less_reproduces_the_uncapped_index(self):
        # A negative env value is uncapped too, and deliberately not refused the
        # way a negative --limit or ANNOTATE_BUDGET_MINUTES is: a sign error
        # THERE changes what the run does (an unbounded --limit run; a deadline
        # already in the past that declines every LLM phase), while here it only
        # makes the kNN passes slower and gives the labels the tier gave before
        # the cap existed. So the sign has to land on that documented pre-cap
        # behavior, end to end and not just inside _sample_seed_ids.
        for cap in (0, -1):
            with self.subTest(cap=cap):
                labels, indexed, said = self.build(cap)
                self.assertEqual(len(indexed), _SEEDS)
                self.assertEqual(set(labels), indexed)
                self.assertEqual(self.build(_SEEDS + 100)[1], indexed)
                self.assertNotIn("kNN index", said)

    def test_a_pool_under_the_cap_says_nothing(self):
        # The line is news only when the cap actually bit.
        self.assertNotIn("kNN index", self.build(_SEEDS + 50)[2])

    def test_a_cap_below_the_floor_is_raised_to_the_floor(self):
        _labels, indexed, _said = self.build(5)
        self.assertEqual(len(indexed), annotate_distill._KNN_MIN_SEEDS)

    def test_the_short_pool_error_is_about_the_pool_not_the_index(self):
        # The floor check runs BEFORE the cap and against the uncapped count,
        # so capping can neither invent this error nor hide it.
        self.con.execute("DELETE FROM chunk_annotations WHERE chunk_id > 50")
        self.con.commit()
        for cap in (0, 800, 5):
            with self.subTest(cap=cap):
                with self.assertRaises(SystemExit):
                    self.build(cap)


# ---------------------------------------------------------------------------
# The stratified sampler
# ---------------------------------------------------------------------------

@_NEEDS_VEC
class TestStratifiedSample(DBFixture):
    """Two chats sized 60/40 so an exact proportional split is checkable."""

    BIG, SMALL = 60, 40

    def setUp(self):
        super().setUp()
        self.build_vec_table()
        rows = ([(i, 1) for i in range(1, self.BIG + 1)]
                + [(i, 2) for i in range(self.BIG + 1, self.BIG + self.SMALL + 1)])
        self.con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (?, ?, NULL, NULL, '2026-01-01T00:00:00', "
            "'2026-01-01T00:05:00', 1, ?)",
            [(chunk_id, chat_id, _chunk_text(chunk_id, _PLAIN))
             for chunk_id, chat_id in rows])
        self.add_vectors(rows)
        self.con.commit()
        self.pending = self.con.execute(
            "SELECT id, text FROM chunks ORDER BY id").fetchall()
        self.chat_of = dict(self.con.execute(
            "SELECT id, chat_id FROM chunks").fetchall())

    def split(self, sample):
        counts = {}
        for row in sample:
            chat = self.chat_of[row[0]]
            counts[chat] = counts.get(chat, 0) + 1
        return counts

    def test_deterministic_and_returns_the_caller_s_own_rows(self):
        first = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        second = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        self.assertEqual(first, second)
        for a, b in zip(first, second):
            self.assertIs(a, b)
        by_id = {id(row) for row in self.pending}
        self.assertTrue(all(id(row) in by_id for row in first))

    def test_allocation_is_exactly_proportional(self):
        sample = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        self.assertEqual(len(sample), 10)
        self.assertEqual(self.split(sample), {1: 6, 2: 4})

    def test_sample_size_is_min_need_and_pending(self):
        everything = annotate_sampling.stratified_sample(
            self.con, self.pending, len(self.pending) + 50)
        self.assertEqual(len(everything), len(self.pending))
        self.assertEqual(annotate_sampling.stratified_sample(self.con, self.pending, 0),
                         [])
        self.assertEqual(annotate_sampling.stratified_sample(self.con, [], 10), [])

    def test_degrades_to_stratified_random_without_chunk_vectors(self):
        self.con.execute("DROP TABLE chunk_vectors")
        self.con.commit()
        sample = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        self.assertEqual(len(sample), 10)
        self.assertEqual(self.split(sample), {1: 6, 2: 4})

    def test_degrades_without_embed_dim(self):
        self.con.execute("DELETE FROM ingest_state WHERE key = 'embed_dim'")
        self.con.commit()
        sample = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        self.assertEqual(len(sample), 10)
        self.assertEqual(self.split(sample), {1: 6, 2: 4})

    def test_degrades_when_some_chunks_have_no_vector(self):
        self.con.execute("DELETE FROM chunk_vectors WHERE chunk_id % 2 = 0")
        self.con.commit()
        sample = annotate_sampling.stratified_sample(self.con, self.pending, 10)
        self.assertEqual(len(sample), 10)
        self.assertEqual(self.split(sample), {1: 6, 2: 4})


# ---------------------------------------------------------------------------
# The event-candidate prefilter
# ---------------------------------------------------------------------------

class TestEventCandidates(DBFixture):
    """chat 1 dominates by message count; chat 2 is the out-of-scope control."""

    def setUp(self):
        super().setUp()
        self.con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, NULL, 0, 'iMessage')", [(1, "chat1"), (2, "chat2")])
        # (chunk_id, chat_id, [message bodies])
        self.layout = [
            (1, 1, ["thanks so much for the ride"]),        # gratitude
            (2, 1, [_PLAIN]),                               # no pattern
            (3, 1, ["congrats!!", "so proud of you"]),      # two hits, one id
            (4, 1, ["i got the job at the lab"]),           # already 'llm'
            (5, 2, ["thank you for everything"]),           # out of scope
        ]
        chunks, messages = [], []
        msg_id = 0
        for chunk_id, chat_id, bodies in self.layout:
            for body in bodies:
                msg_id += 1
                messages.append((msg_id, f"guid-{msg_id}", chat_id, body,
                                 f"2026-01-01T00:{msg_id:02d}:00", chunk_id))
            chunks.append((chunk_id, chat_id, _chunk_text(chunk_id, bodies[0])))
        # Pad chat 1 so it wins the top-dyad ordering by message count.
        for extra in range(20):
            msg_id += 1
            messages.append((msg_id, f"guid-{msg_id}", 1, _PLAIN,
                             f"2026-01-02T00:{extra:02d}:00", 2))
        self.con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (?, ?, NULL, NULL, '2026-01-01T00:00:00', "
            "'2026-01-01T00:05:00', 1, ?)", chunks)
        self.con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
            "date, chunk_id) VALUES (?, ?, ?, 1, 0, ?, ?, ?)", messages)
        self.con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source) VALUES (4, 0.9, '[\"joy\"]', 3, '[]', 'llm')")
        self.con.execute(
            "INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, depth, "
            "events_json, source, knn_confidence) "
            "VALUES (1, 0.1, '[]', 1, '[]', 'distilled', 0.4)")
        self.con.commit()

    def test_matches_patterns_excludes_llm_rows_and_ranks_by_hit_count(self):
        ids = annotate_events.event_candidate_chunk_ids(self.con, True, None)
        # 1 (distilled), 3 (unannotated) and 5 match; 2 matches nothing; 4
        # matches but is already 'llm'. 3 matches TWICE, so it outranks the
        # one-hit chunks despite its higher id -- the ranking IS the feature:
        # the caller's deadline stops partway down this list, so the prefix it
        # can afford has to be the densest candidates rather than the oldest.
        # 3 still appears exactly once, however many of its messages hit; 1 and
        # 5 tie at one hit each and the id tiebreak orders them.
        self.assertEqual(ids, [3, 1, 5])

    def test_the_ranking_survives_the_row_fetch(self):
        # The SQL ranking is only half the feature. annotate_hybrid._chunk_rows
        # turns these ids into the (id, text) rows the LLM path wants, and a
        # re-sort there would hand the budget straight back to chunk-id order
        # without changing a single assertion above.
        ids = annotate_events.event_candidate_chunk_ids(self.con, True, None)
        rows = annotate_hybrid._chunk_rows(self.con, ids)
        self.assertEqual([chunk_id for chunk_id, _text in rows], [3, 1, 5])

    def test_scope_restricted_to_top_dyads(self):
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, False, 1),
                         [3, 1])
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, False, 2),
                         [3, 1, 5])

    def test_llm_annotated_chunk_is_never_a_candidate(self):
        self.assertNotIn(4, annotate_events.event_candidate_chunk_ids(self.con, True, None))
        # ...and upgrading a candidate takes it out of the pool.
        self.con.execute("UPDATE chunk_annotations SET source = 'llm' WHERE chunk_id = 1")
        self.con.commit()
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, True, None),
                         [3, 5])


class TestChunkRowFetch(DBFixture):
    """annotate_hybrid._chunk_rows: the id -> (id, text) fetch behind every
    phase-3 dispatch. Its ORDER is part of the contract now that its callers
    hand it a ranking, and its two id oddities are answered deliberately."""

    def setUp(self):
        super().setUp()
        self.con.execute(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, "
            "service) VALUES (1, 'chat1', NULL, 0, 'iMessage')")
        self.con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) "
            "VALUES (?, 1, NULL, NULL, '2026-01-01T00:00:00', "
            "'2026-01-01T00:05:00', 1, ?)",
            [(chunk_id, _chunk_text(chunk_id, _PLAIN)) for chunk_id in range(1, 6)])
        self.con.commit()

    def ids(self, chunk_ids):
        return [chunk_id for chunk_id, _text in
                annotate_hybrid._chunk_rows(self.con, chunk_ids)]

    def test_rows_come_back_in_the_caller_s_order_not_the_table_s(self):
        self.assertEqual(self.ids([4, 1, 5, 2]), [4, 1, 5, 2])

    def test_the_order_holds_across_the_parameter_safe_slices(self):
        # The fetch goes out in _ID_SLICE blocks (SQLite's 999-variable
        # ceiling), so SQL row order is per-block and meaningless -- exactly
        # the seam where a ranking gets quietly lost.
        with mock.patch.object(annotate_hybrid, "_ID_SLICE", 2):
            self.assertEqual(self.ids([5, 3, 1, 4, 2]), [5, 3, 1, 4, 2])

    def test_an_id_with_no_chunk_row_is_omitted_not_faked(self):
        # A None-text row would reach the LLM as an empty prompt and burn a
        # request to come back unparseable.
        self.assertEqual(self.ids([3, 99, 1]), [3, 1])

    def test_a_repeated_id_yields_one_row_at_its_first_position(self):
        # The tier's whole contract is a wall-clock budget, so paying twice
        # for the same chunk is never the right reading of a duplicate.
        self.assertEqual(self.ids([3, 1, 3, 2]), [3, 1, 2])


# ---------------------------------------------------------------------------
# Schema migration + the _ensure_column self-heal
# ---------------------------------------------------------------------------

def _columns(con, table):
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


class TestKnnConfidenceMigration(unittest.TestCase):
    def test_added_to_a_table_created_without_it_and_idempotent(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        # The pre-hybrid chunk_annotations DDL, verbatim in shape.
        con.execute("""
            CREATE TABLE chunk_annotations (
                chunk_id INTEGER PRIMARY KEY,
                valence REAL,
                emotions_json TEXT,
                depth INTEGER,
                events_json TEXT,
                source TEXT DEFAULT 'llm'
            )
        """)
        con.execute("INSERT INTO chunk_annotations(chunk_id, valence, emotions_json, "
                    "depth, events_json, source) "
                    "VALUES (1, 0.5, '[]', 2, '[]', 'distilled')")
        self.assertNotIn("knn_confidence", _columns(con, "chunk_annotations"))

        schema.create_schema(con)
        self.assertIn("knn_confidence", _columns(con, "chunk_annotations"))
        # Nullable with no default: a row written before the column has no
        # recorded confidence, and NULL has to say so.
        self.assertIsNone(con.execute(
            "SELECT knn_confidence FROM chunk_annotations WHERE chunk_id = 1"
        ).fetchone()[0])

        schema.create_schema(con)
        self.assertEqual(_columns(con, "chunk_annotations").count("knn_confidence"), 1)

    def test_fresh_schema_has_the_column(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        schema.create_schema(con)
        self.assertIn("knn_confidence", _columns(con, "chunk_annotations"))


class TestEnsureColumnSelfHeal(HybridFixture):
    """A DB whose chunk_annotations predates knn_confidence is reachable
    without an ingest.py run, and both kNN tiers write that column -- so both
    must self-heal instead of dying on 'no such column'. This was a real crash
    during integration."""

    def setUp(self):
        super().setUp()
        self.con.execute("ALTER TABLE chunk_annotations DROP COLUMN knn_confidence")
        self.con.commit()
        self.assertNotIn("knn_confidence", _columns(self.con, "chunk_annotations"))
        for target, value in (("ANNOTATE_HYBRID_SEED_TARGET", _SEEDS),
                              ("ANNOTATE_SEED_TARGET", _SEEDS)):
            patcher = mock.patch.object(config, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(annotate, "ANNOTATE_WORKERS", 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_healed(self):
        self.assertIn("knn_confidence", _columns(self.con, "chunk_annotations"))
        (n_scored,) = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations "
            "WHERE source = 'distilled' AND knn_confidence IS NOT NULL").fetchone()
        self.assertEqual(n_scored, _PENDING_MAIN + _PENDING_OTHER)

    def test_hybrid_runs_against_a_pre_column_db(self):
        with mock.patch.object(ollama, "chat_once_json", FakeLLM()):
            annotate.main(["--quality", "hybrid", "--all", "--limit", "0"])
        self.assert_healed()

    def test_distilled_runs_against_a_pre_column_db(self):
        with mock.patch.object(ollama, "chat_once_json", FakeLLM()) as llm:
            annotate.main(["--quality", "distilled"])
        self.assertEqual(llm.order, [])              # seed pool already full
        self.assert_healed()

    def test_dry_runs_preview_a_pre_column_db_without_migrating_it(self):
        # The self-heal runs schema.create_schema, which is not schema-only
        # (it rescans reactions for the old guid prefix and rewrites them), so
        # calling it under --dry-run made a preview the one write-free path
        # that wrote. Both tiers now defer it and degrade their reads.
        for quality in ("hybrid", "distilled"):
            with self.subTest(quality=quality):
                with mock.patch.object(ollama, "chat_once_json") as llm:
                    annotate.main(["--quality", quality, "--dry-run"])
                llm.assert_not_called()
                self.assertNotIn("knn_confidence",
                                 _columns(self.con, "chunk_annotations"))


class TestTermsEnsureColumnSelfHeal(DBFixture):
    """The same _ensure_column contract for its third caller, --terms.

    Deliberately NOT under HybridFixture: that fixture is @_NEEDS_VEC, and this
    path reads no vectors at all. Inheriting the skip would have left the only
    pin on a doc claim that ALREADY drifted once silently green on exactly the
    interpreters CLAUDE.md's install note calls out (python.org builds, no
    sqlite-vec)."""

    def test_a_terms_dry_run_does_not_migrate_its_column_either(self):
        # --terms migrated under --dry-run in the first cut; the flag reached it
        # later and nothing pinned the change, so the README went on promising a
        # migration the code had stopped making until the PR #42 round-3 pass.
        self.con.execute("ALTER TABLE dyad_terms DROP COLUMN curate_failures")
        self.con.executemany(
            "INSERT INTO dyad_terms(chat_id, term, lift, n_uses, "
            "first_message_id, coined_by_handle_id, coined_is_me, llm_verdict) "
            "VALUES (1, ?, 10.0, 5, NULL, NULL, 1, NULL)",
            [("blorbo",), ("zonk",)])
        self.con.commit()
        with mock.patch.object(ollama, "chat_once") as llm:
            with mock.patch("builtins.print") as printed:
                annotate.main(["--terms", "--dry-run"])
        llm.assert_not_called()
        self.assertNotIn("curate_failures", _columns(self.con, "dyad_terms"))
        # And the degraded read is the honest one rather than a crash: a DB
        # without the column has recorded no failures, so nothing is parked and
        # every unverdicted term is in scope.
        said = "\n".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertIn("2 dyad terms pending", said)
        self.assertNotIn("parked", said)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

class TestHybridCli(DBFixture):
    def test_terms_with_hybrid_quality_is_rejected(self):
        with mock.patch.object(ollama, "chat_once_json") as llm:
            with self.assertRaises(SystemExit):
                annotate.main(["--terms", "--quality", "hybrid"])
        llm.assert_not_called()

    def test_upgrade_distilled_with_hybrid_is_rejected(self):
        with mock.patch.object(ollama, "chat_once_json") as llm:
            with self.assertRaises(SystemExit):
                annotate.main(["--quality", "hybrid", "--upgrade-distilled"])
        llm.assert_not_called()

    def test_a_negative_row_cap_is_refused_before_a_tier_is_chosen(self):
        # hybrid_chunks refuses a negative --limit itself (there it is a list
        # slice: event_ids[:-1] keeps everything but the LEAST dense candidate),
        # but that check cannot see the other tiers. On the per-chunk tiers and
        # on --terms the same value reaches a SQL `LIMIT ?`, where -1 means NO
        # limit -- the flag that exists to bound a run buying an unbounded one.
        # --top-dyads has the same shape against the scope subquery.
        for argv in (["--quality", "fast", "--limit", "-1"],
                     ["--quality", "hybrid", "--limit", "-5"],
                     ["--terms", "--limit", "-1"],
                     ["--quality", "fast", "--top-dyads", "-1"]):
            with self.subTest(argv=argv):
                with mock.patch.object(ollama, "chat_once_json") as llm:
                    with self.assertRaises(SystemExit):
                        annotate.main(argv)
                llm.assert_not_called()

    def test_a_row_cap_of_zero_is_still_a_legitimate_run(self):
        # 0 is the empty cap, which someone can legitimately ask for; only the
        # sign is refused.
        with mock.patch.object(ollama, "chat_once_json") as llm:
            annotate.main(["--quality", "fast", "--limit", "0"])
        llm.assert_not_called()

    def _one_pending_chunk_per_chat(self):
        """Two chats with one un-annotated chunk each, so a scope assertion can
        be about the CHUNKS a run would touch and not about the label it
        printed."""
        self.con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, "
            "service) VALUES (?, ?, NULL, 0, 'iMessage')", [(1, "c1"), (2, "c2")])
        self.con.executemany(
            "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
            "first_date, last_date, n_messages, text) VALUES "
            "(?, ?, NULL, NULL, '2026-01-01T00:00:00', '2026-01-01T00:05:00', "
            "1, 'text')", [(1, 1), (2, 2)])
        self.con.executemany(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, "
            "text, date, chunk_id) VALUES (?, ?, ?, 1, 0, 'hi', "
            "'2026-01-01T00:00:00', ?)",
            [(1, "g1", 1, 1), (2, "g2", 2, 2)])
        self.con.commit()

    def _dry_run_scope(self, argv):
        """The chunk ids a --dry-run of `argv` says it would annotate. Parsed
        from the preview line's own comma-separated list -- NOT with _ids_in,
        which reads '### chunk N' prompt headers and would quietly return []
        for every scope, making the assertions below unfalsifiable."""
        with mock.patch.object(ollama, "chat_once_json") as llm:
            with mock.patch("builtins.print") as printed:
                annotate.main(argv + ["--dry-run"])
        llm.assert_not_called()
        said = "\n".join(str(a) for call in printed.call_args_list for a in call.args)
        listed = re.search(r"--dry-run: first chunk ids: (.*)", said).group(1)
        return said, [] if listed == "(none)" else [int(c) for c in listed.split(", ")]

    def test_top_dyads_zero_means_zero_chats_not_the_default(self):
        # main resolved this with `args.top_dyads or ANNOTATE_TOP_DYADS`, so an
        # explicit 0 fell through to the 20-chat default -- while hybrid_chunks
        # has always read its own 0 as 0. `--quality max --top-dyads 0` asked
        # for nothing and would have bought a full default-scope LLM run.
        #
        # Asserted on the chunks the run would TOUCH: the printed "top 0 chats"
        # label only proves what main resolved, and would go on passing if the
        # scope subquery then treated 0 as "every chat".
        self._one_pending_chunk_per_chat()
        said, scoped = self._dry_run_scope(["--quality", "fast", "--top-dyads", "0"])
        self.assertIn("top 0 chats", said)
        self.assertEqual(scoped, [])
        self.assertEqual(self._dry_run_scope(["--quality", "fast"])[1], [1, 2])
        self.assertEqual(
            self._dry_run_scope(["--quality", "fast", "--top-dyads", "1"])[1], [1])

    def test_a_negative_scope_is_refused_from_the_env_too(self):
        # The flag is refused at the parser; the env twin reaches the SAME scope
        # subquery LIMIT, where SQLite reads -1 as no limit and a scoped run
        # silently becomes an --all one. A knob is validated where it is
        # RESOLVED, not only where a flag supplies it.
        with mock.patch.object(annotate, "ANNOTATE_TOP_DYADS", -1):
            with mock.patch.object(ollama, "chat_once_json") as llm:
                with self.assertRaises(SystemExit) as cm:
                    annotate.main(["--quality", "fast"])
        self.assertIn("ANNOTATE_TOP_DYADS", str(cm.exception))
        llm.assert_not_called()

    def test_the_env_refusal_spares_the_runs_that_never_read_the_knob(self):
        # The other half of the same precedent: a bad knob must not refuse a run
        # it could not have affected. `--terms` has no scope, `--all` omits the
        # scope clause, and distilled hardcodes top_dyads=0 -- so refusing them
        # would take out bare `make annotate` over a value those runs never
        # bind, and would leave the message naming a remedy (--all) it had just
        # refused.
        with mock.patch.object(annotate, "ANNOTATE_TOP_DYADS", -1):
            with mock.patch.object(ollama, "chat_once_json") as llm:
                for argv in (["--terms"],
                             ["--quality", "fast", "--all"],
                             ["--quality", "distilled", "--dry-run"]):
                    with self.subTest(argv=argv):
                        try:
                            annotate.main(argv)
                        except SystemExit as exc:
                            # These paths can still exit for their OWN reasons
                            # on an empty fixture DB (distilled wants
                            # chunk_vectors). What matters is that it is never
                            # the scope knob refusing a run it cannot affect.
                            self.assertNotIn("ANNOTATE_TOP_DYADS", str(exc))
            llm.assert_not_called()

    def test_upgrade_distilled_still_allowed_on_the_llm_tiers(self):
        # The rejection must be about the two kNN tiers, not the flag itself.
        with mock.patch.object(ollama, "chat_once_json", FakeLLM()):
            annotate.main(["--quality", "fast", "--all", "--upgrade-distilled"])

    def test_hybrid_preset_is_think_false_and_batch_eight(self):
        # An effort level is not portable: 'low' is merely truthy to a toggle
        # model and buys a full reasoning pass. False is a distinct state from
        # None (which omits the flag).
        self.assertEqual(annotate.QUALITY_PRESETS["hybrid"], (False, 8))
        think, batch_size = annotate._resolve_quality("hybrid", "qwen3.5:4b")
        self.assertIs(think, False)
        self.assertEqual(batch_size, 8)

    def test_an_effort_level_preset_is_translated_for_toggle_models_only(self):
        # The presets were written against gpt-oss, where 'low' asks for a
        # CHEAP reasoning pass. To a toggle model the same string is only
        # truthy and buys a FULL one, which would make the "fast" tier the
        # slowest -- so it resolves to False there and is left alone on an
        # effort-level model.
        #
        # ANNOTATE_THINK is pinned empty because an explicit one WINS over the
        # translation by design (the test below asserts exactly that), so a
        # developer .env carrying ANNOTATE_THINK=false made this test assert
        # its own env instead of the translation. The suite is hermetic.
        with mock.patch.object(annotate, "ANNOTATE_THINK", ""):
            for quality in ("fast", "balanced"):
                self.assertEqual(annotate.QUALITY_PRESETS[quality][0], "low")
                self.assertIs(
                    annotate._resolve_quality(quality, "qwen3.5:4b")[0], False)
                self.assertEqual(
                    annotate._resolve_quality(quality, "gpt-oss:20b")[0], "low")
            # None means "omit the flag" and is NOT an effort level: it survives
            # on every backend, or 'max' would stop meaning the model's own
            # default.
            self.assertIsNone(annotate._resolve_quality("max", "qwen3.5:4b")[0])
            self.assertIsNone(annotate._resolve_quality("max", "gpt-oss:20b")[0])

    def test_an_explicit_annotate_think_still_wins_over_the_translation(self):
        # Naming a think is user intent the translation must not second-guess.
        with mock.patch.object(annotate, "ANNOTATE_THINK", "low"):
            self.assertEqual(annotate._resolve_quality("fast", "qwen3.5:4b")[0], "low")

    def test_apple_char_budget_is_one_chunk_s_worth_of_text(self):
        # 8k did not stop the 4,096-token FM window overflowing (7+ overflows,
        # 6 of 66 chunks lost when measured); 2k -- one CHUNK_MAX_CHARS, so
        # never two full-size chunks in one request -- did, and measured
        # 1.7-3.5x faster per chunk because a failed apple batch costs its own
        # wall time plus a single-chunk retry for every id in it. Shared by
        # every tier, so this pin is a global one.
        self.assertEqual(annotate._APPLE_ANNOTATE_CHAR_BUDGET, 2000)
        self.assertLessEqual(annotate._APPLE_ANNOTATE_CHAR_BUDGET,
                             config.CHUNK_MAX_CHARS)


class TestHybridDryRun(HybridFixture):
    def test_dry_run_makes_no_llm_call_and_writes_no_annotations(self):
        before = self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations").fetchone()[0]
        with mock.patch.object(annotate, "ANNOTATE_WORKERS", 1):
            with mock.patch.object(config, "ANNOTATE_HYBRID_SEED_TARGET", _SEEDS + 5):
                with mock.patch.object(ollama, "chat_once_json") as llm:
                    annotate.main(["--quality", "hybrid", "--all", "--dry-run"])
        llm.assert_not_called()
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM chunk_annotations").fetchone()[0], before)
        self.assertEqual(self.n_source("distilled"), 0)
        # The model guard records nothing either: a preview is not a run.
        self.assertIsNone(get_state(self.con, "annotate_model"))


# ---------------------------------------------------------------------------
# distilled is unchanged
# ---------------------------------------------------------------------------

class TestDistilledSamplerHook(HybridFixture):
    def test_sampler_none_is_the_historical_random_sample_draw(self):
        # The seed draw with sampler=None must stay bit-identical to
        # random.sample(pending, need) -- the distilled tier's behavior.
        pending = annotate._pending_chunks(self.con, all_chats=True, top_dyads=0,
                                           limit=None, upgrade_distilled=True)
        random.seed(4242)
        expected = [row[0] for row in random.sample(pending, 5)]

        random.seed(4242)
        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            n = annotate_distill._seed_chunks(self.con, _MODEL, False, 1, _SEEDS + 5)
        self.assertEqual(n, 5)
        self.assertEqual(llm.order, expected)

    def test_sampler_hook_replaces_the_draw(self):
        calls = []

        def sampler(con, pending, need):
            calls.append((len(pending), need))
            return pending[:need]

        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            n = annotate_distill._seed_chunks(self.con, _MODEL, False, 1, _SEEDS + 3,
                                              sampler=sampler)
        self.assertEqual(n, 3)
        self.assertEqual(calls, [(_PENDING_MAIN + _PENDING_OTHER, 3)])
        self.assertEqual(llm.order, [_FIRST_PENDING, _FIRST_PENDING + 1,
                                     _FIRST_PENDING + 2])


class TestDispatchDeadline(unittest.TestCase):
    """annotate._DispatchDeadline is the piece that keeps a budgeted phase from
    overrunning by its own backlog: _pooled holds 2x workers groups, and every
    one of them still finishes after the generator stops."""

    def test_nothing_is_reserved_until_a_group_has_completed(self):
        clock = FakeClock(0.0)
        gate = annotate._DispatchDeadline(10.0, 4, clock.monotonic)
        self.assertEqual(gate.reserve(), 0.0)
        # An un-measured phase dispatches on the plain check alone -- there is
        # no honest estimate to make before the first completion.
        self.assertEqual(list(gate.dispatchable(["a", "b"])), ["a", "b"])
        self.assertFalse(gate.stopped)

    def test_reserve_grows_with_the_backlog_this_gate_actually_holds(self):
        # An empty pool reserves one group's latency, not _pooled's 2x-workers
        # ceiling -- otherwise a phase starting fresh would refuse its own first
        # dispatch and buy nothing. The reserve then tracks what is in flight.
        clock = FakeClock(0.0)
        rate = annotate._GroupRate()
        rate.record(20.0)
        gate = annotate._DispatchDeadline(1000.0, 4, clock.monotonic, rate)
        self.assertEqual(gate.reserve(), 20.0)              # 1 group / 4 workers

        groups = gate.dispatchable(iter(range(10)))
        for _ in range(4):
            next(groups)
        self.assertEqual(gate.reserve(), 25.0)              # 5 / 4 waves x 20s

        clock.now = 960.0
        self.assertEqual(next(groups), 4)                   # 960 + 30 < 1000
        clock.now = 970.0
        with self.assertRaises(StopIteration):              # 970 + 35 >= 1000
            next(groups)
        self.assertTrue(gate.stopped)

    def test_a_completed_group_leaves_the_backlog(self):
        clock = FakeClock(0.0)
        rate = annotate._GroupRate()
        rate.record(20.0)
        gate = annotate._DispatchDeadline(1000.0, 4, clock.monotonic, rate)
        groups = gate.dispatchable(iter(range(10)))
        for _ in range(8):
            next(groups)
        self.assertEqual(gate.reserve(), 45.0)              # 9 / 4 waves
        for _ in range(4):
            gate.timed(lambda group: clock.advance(20.0))("group")
        self.assertEqual(gate.reserve(), 25.0)              # 5 / 4 waves

    def test_a_sequential_pool_reserves_one_group_per_outstanding_group(self):
        clock = FakeClock(0.0)
        gate = annotate._DispatchDeadline(1000.0, 1, clock.monotonic)
        gate.timed(lambda group: clock.advance(30.0))("group")
        self.assertEqual(gate.reserve(), 30.0)

    def test_the_estimate_is_the_mean_of_completed_groups(self):
        clock = FakeClock(0.0)
        gate = annotate._DispatchDeadline(1000.0, 1, clock.monotonic)
        charge = [10.0, 30.0]
        run = gate.timed(lambda group: clock.advance(charge.pop(0)))
        run("a")
        self.assertEqual(gate.reserve(), 10.0)
        run("b")
        self.assertEqual(gate.reserve(), 20.0)

    def test_the_rate_is_shared_so_a_later_phase_is_not_blind(self):
        # Only the first LLM phase of a run should have to dispatch on nothing.
        clock = FakeClock(0.0)
        rate = annotate._GroupRate()
        first = annotate._DispatchDeadline(1000.0, 1, clock.monotonic, rate)
        first.timed(lambda group: clock.advance(40.0))("group")
        second = annotate._DispatchDeadline(1000.0, 1, clock.monotonic, rate)
        self.assertEqual(second.reserve(), 40.0)

    def test_a_prior_makes_the_very_first_dispatch_honest(self):
        # The blind wave is the residual overshoot: _pooled commits 2x workers
        # groups before any timing exists, measured as a 2-min budget landing
        # at 2.6 min with phase 3b starved to zero. A prior from an earlier run
        # closes it -- the gate can decline before ever measuring anything.
        clock = FakeClock(0.0)
        gate = annotate._DispatchDeadline(30.0, 4, clock.monotonic,
                                          annotate._GroupRate(prior=40.0))
        self.assertEqual(gate.reserve(), 40.0)
        self.assertEqual(list(gate.dispatchable(["a", "b"])), [])
        self.assertTrue(gate.stopped)

    def test_a_measurement_always_beats_the_prior(self):
        clock = FakeClock(0.0)
        rate = annotate._GroupRate(prior=40.0)
        gate = annotate._DispatchDeadline(1000.0, 1, clock.monotonic, rate)
        gate.timed(lambda group: clock.advance(10.0))("group")
        self.assertEqual(rate.mean(), 10.0)         # mean stays measurement-only
        self.assertEqual(gate.reserve(), 10.0)


class TestGroupRateMemory(DBFixture):
    """The prior is carried across runs in ingest_state, so only a DB that has
    never run this model/batch/workers combination dispatches on a guess."""

    def test_a_measured_rate_primes_the_next_run(self):
        rate = annotate._GroupRate()
        rate.record(40.0)
        annotate._save_group_rate(self.con, rate, "upgrade", _MODEL, 8, 4)
        primed = annotate._load_group_rate(self.con, "upgrade", _MODEL, 8, 4)
        self.assertIsNone(primed.mean())            # nothing measured yet...
        self.assertEqual(primed.estimate(), 40.0)   # ...but not blind either

    def test_the_slowest_group_is_what_crosses_runs_not_the_mean(self):
        # Group latency is noisy enough between runs (a 106.4 s mean measured
        # 148 s the next run) that carrying the mean forward under-reserves and
        # overruns. The failure is one-sided: under-buying is recovered by a
        # rerun, an overrun breaks the tier's only promise.
        rate = annotate._GroupRate()
        for seconds in (10.0, 90.0, 20.0):
            rate.record(seconds)
        self.assertEqual(rate.mean(), 40.0)
        annotate._save_group_rate(self.con, rate, "upgrade", _MODEL, 8, 4)
        primed = annotate._load_group_rate(self.con, "upgrade", _MODEL, 8, 4)
        self.assertEqual(primed.estimate(), 90.0)
        # ...and within a run the live mean still wins over it.
        primed.record(30.0)
        self.assertEqual(primed.estimate(), 30.0)

    def test_an_unmeasured_rate_is_never_stored(self):
        # Storing the prior back would freeze a cold guess into the DB.
        annotate._save_group_rate(self.con, annotate._GroupRate(prior=99.0),
                                  "upgrade", _MODEL, 8, 4)
        self.assertIsNone(get_state(
            self.con, annotate._group_rate_key("upgrade", _MODEL, 8, 4)))

    def test_the_key_separates_phase_model_batch_and_workers(self):
        # Group LATENCY is what the reserve is denominated in, and all four
        # move it -- a seed group is not an upgrade group, and 4 concurrent
        # requests against a serializing Ollama each measure ~4x a lone one.
        rate = annotate._GroupRate()
        rate.record(40.0)
        annotate._save_group_rate(self.con, rate, "upgrade", _MODEL, 8, 4)
        cold = annotate._COLD_GROUP_SECONDS_PER_CHUNK
        for phase, model, batch, workers in (("seed", _MODEL, 8, 4),
                                             ("upgrade", "other", 8, 4),
                                             ("upgrade", _MODEL, 1, 4),
                                             ("upgrade", _MODEL, 8, 1)):
            with self.subTest(phase=phase, batch=batch, workers=workers):
                primed = annotate._load_group_rate(self.con, phase, model,
                                                   batch, workers)
                self.assertEqual(primed.estimate(), cold * batch * workers)

    def test_a_corrupt_stored_rate_falls_back_to_the_cold_prior(self):
        set_state(self.con, annotate._group_rate_key("upgrade", _MODEL, 8, 4),
                  "not-a-number")
        primed = annotate._load_group_rate(self.con, "upgrade", _MODEL, 8, 4)
        self.assertEqual(primed.estimate(),
                         annotate._COLD_GROUP_SECONDS_PER_CHUNK * 8 * 4)


class TestColdPriorDenomination(HybridFixture):
    """The cold prior is arithmetic -- seconds/chunk x rows/group x workers --
    and _load_group_rate has only batch_size for the middle term, which is a
    CEILING: _annotation_groups packs by char budget too. Charging it for 8 rows
    a group does not hold measured a ~3.6x over-estimate on the event phase, and
    because reserve() plans off the prior until a phase's first group completes,
    that is time the phase never gets back inside the run."""

    COLD = annotate._COLD_GROUP_SECONDS_PER_CHUNK

    def test_the_mean_row_count_of_a_packing(self):
        self.assertEqual(annotate._mean_group_rows([]), 0.0)
        self.assertEqual(annotate._mean_group_rows([[1, 2, 3], [4]]), 2.0)

    def test_a_cold_prior_is_rescaled_to_the_group_that_is_really_sent(self):
        rate = annotate._load_group_rate(self.con, "event", _MODEL, 8, 4)
        self.assertEqual(rate.estimate(), self.COLD * 8 * 4)
        rate.denominate(5.0)
        self.assertEqual(rate.estimate(), self.COLD * 5.0 * 4)

    def test_a_banked_measurement_is_never_rescaled(self):
        # Only the arithmetic is re-done. A number an earlier run actually
        # measured is the ground truth the prior exists to stand in for.
        measured = annotate._GroupRate()
        measured.record(40.0)
        annotate._save_group_rate(self.con, measured, "event", _MODEL, 8, 4)
        primed = annotate._load_group_rate(self.con, "event", _MODEL, 8, 4)
        primed.denominate(1.0)
        self.assertEqual(primed.estimate(), 40.0)

    def test_an_empty_packing_and_a_rate_with_no_prior_say_nothing(self):
        rate = annotate._load_group_rate(self.con, "event", _MODEL, 8, 4)
        rate.denominate(0.0)
        self.assertEqual(rate.estimate(), self.COLD * 8 * 4)
        blind = annotate._GroupRate()
        blind.denominate(4.0)
        self.assertIsNone(blind.estimate())

    def test_a_live_measurement_still_beats_a_denominated_prior(self):
        rate = annotate._load_group_rate(self.con, "event", _MODEL, 8, 4)
        rate.denominate(2.0)
        rate.record(9.0)
        self.assertEqual(rate.estimate(), 9.0)

    def test_a_phase_denominates_from_the_packing_it_built(self):
        # End to end, and BEFORE the first dispatch decision: these rows are
        # long enough that the 10k-char budget closes a group at 2 of them,
        # nothing like the batch size of 8 the prior was charged for.
        rows = [(chunk_id, "x" * 4000) for chunk_id in range(1, 9)]
        rate = annotate._load_group_rate(self.con, "event", _MODEL, 8, 4)
        self.assertEqual(rate.estimate(), self.COLD * 8 * 4)
        stats = {"done": 0, "skipped": 0, "stopped": False}
        clock = FakeClock()
        with self.fake_clock(clock):
            with mock.patch("builtins.print"):
                # A deadline already past, so the gate declines everything and
                # no request goes out -- the denomination is the only thing
                # under test, and it has to have happened by then.
                annotate_hybrid._run_upgrades(
                    self.con, rows, _MODEL, False, 4, 8, clock.now - 1.0,
                    "Upgrading", set(), stats, rate, phase="event")
        self.assertEqual(rate.estimate(), self.COLD * 2.0 * 4)
        self.assertTrue(stats["stopped"])
        self.assertEqual(stats["done"], 0)


class TestSeedBatching(HybridFixture):
    """The seed pass is LLM work like any other phase, so it batches and it
    stops at a deadline -- but only when a caller asks for both."""

    def test_batch_size_one_is_still_one_chunk_per_request(self):
        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            annotate_distill._seed_chunks(self.con, _MODEL, False, 1, _SEEDS + 4,
                                          sampler=lambda con, p, n: p[:n])
        self.assertEqual([len(ids) for ids in llm.requests], [1, 1, 1, 1])

    def test_batch_size_one_on_apple_does_not_pick_up_char_budget_packing(self):
        # annotate._annotation_groups' apple branch ignores batch_size and packs
        # by chars; routing the seed pass through it unconditionally would have
        # silently re-batched the DISTILLED tier on Apple FM.
        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            annotate_distill._seed_chunks(self.con, "apple", False, 1, _SEEDS + 4,
                                          sampler=lambda con, p, n: p[:n])
        self.assertEqual([len(ids) for ids in llm.requests], [1, 1, 1, 1])

    def test_batch_size_above_one_batches(self):
        llm = FakeLLM()
        with mock.patch.object(ollama, "chat_once_json", llm):
            n = annotate_distill._seed_chunks(self.con, _MODEL, False, 1, _SEEDS + 10,
                                              sampler=lambda con, p, n: p[:n],
                                              batch_size=4)
        self.assertEqual(n, 10)
        self.assertEqual([len(ids) for ids in llm.requests], [4, 4, 2])

    def test_a_deadline_stops_the_seed_dispatch_and_the_line_says_so(self):
        clock = FakeClock(0.0)
        llm = FakeLLM(clock=clock, seconds=25.0)
        with mock.patch.object(annotate_distill, "time", clock):
            with mock.patch.object(ollama, "chat_once_json", llm):
                with mock.patch("builtins.print") as printed:
                    n = annotate_distill._seed_chunks(
                        self.con, _MODEL, False, 1, _SEEDS + 5,
                        sampler=lambda con, p, need: p[:need], deadline=60.0)
        # Dispatched at t=0 (ungated) and t=25 (25 + a 25s reserve < 60);
        # refused at t=50, where 50 + 25 >= 60.
        self.assertEqual(n, 2)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertIn("stopped at the seed phase's share of the budget", said)
        self.assertIn(f"{_SEEDS + 2} of {_SEEDS + 5} seeds", said)

    def test_no_deadline_never_stops_dispatch(self):
        clock = FakeClock(0.0)
        llm = FakeLLM(clock=clock, seconds=1e6)
        with mock.patch.object(annotate_distill, "time", clock):
            with mock.patch.object(ollama, "chat_once_json", llm):
                n = annotate_distill._seed_chunks(
                    self.con, _MODEL, False, 1, _SEEDS + 5,
                    sampler=lambda con, p, need: p[:need])
        self.assertEqual(n, 5)


class TestSeedPassKeepsPaidForWork(HybridFixture):
    """The seed pass commits and banks its counter in a `finally`, like
    annotate_hybrid._run_upgrades. An ollama timeout is an ordinary exception
    propagating out of _pooled: it hits no handler here, so without the finally
    it rolled back up to ANNOTATE_COMMIT_EVERY - 1 seed rows the budget had
    already paid for AND left the hybrid_seeds counter behind the rows that DID
    commit -- so the next run re-bought seeds it already owned, overshooting
    ANNOTATE_HYBRID_SEED_TARGET at the upgrade phases' expense."""

    def seed_until_it_blows_up(self, **kwargs):
        llm = FakeLLM(raise_after=3)
        with mock.patch.object(ollama, "chat_once_json", llm):
            with self.assertRaises(LLMError):
                annotate_distill._seed_chunks(
                    self.con, _MODEL, False, 1, _SEEDS + 10,
                    sampler=lambda con, pending, need: pending[:need], **kwargs)
        return llm

    def test_an_llm_error_mid_pass_keeps_the_rows_and_the_counter(self):
        self.assertGreater(config.ANNOTATE_COMMIT_EVERY, 3)   # else nothing to lose
        self.seed_until_it_blows_up(
            counter_key=annotate_hybrid._SEED_COUNTER_KEY)
        # A second connection: only what was COMMITTED is visible.
        con = self.committed()
        self.assertEqual(con.execute(
            "SELECT COUNT(*) FROM chunk_annotations WHERE source = 'llm'"
        ).fetchone()[0], _SEEDS + 3)
        self.assertEqual(
            int(get_state(con, annotate_hybrid._SEED_COUNTER_KEY)), _SEEDS + 3)

    def test_the_distilled_tier_keeps_its_rows_too_and_moves_no_counter(self):
        # counter_key=None is the distilled tier's call: same commit, and no
        # counter touched -- that tier's target is every source='llm' row, so a
        # hybrid_seeds value it moved would be a number nobody asked it for.
        self.seed_until_it_blows_up()
        con = self.committed()
        self.assertEqual(con.execute(
            "SELECT COUNT(*) FROM chunk_annotations WHERE source = 'llm'"
        ).fetchone()[0], _SEEDS + 3)
        self.assertEqual(
            int(get_state(con, annotate_hybrid._SEED_COUNTER_KEY)), _SEEDS)

    def test_the_counter_is_not_applied_twice_on_the_success_path(self):
        # The finally is the ONE place the counter moves; the pre-fix trailing
        # call after the try/except had to go with it.
        #
        # Asserted by COUNTING the calls, not by reading the counter back:
        # _record_seeds writes an ABSOLUTE value (n_seeded captured before the
        # loop, plus n_done), so it is idempotent and a resulting-value
        # assertion passes just as happily against a double application -- it
        # would assert the implementation back to itself.
        calls = []
        real = annotate_distill._record_seeds

        def spy(con, counter_key, n_seeded, n_done, stats=None):
            calls.append((counter_key, n_seeded, n_done))
            return real(con, counter_key, n_seeded, n_done, stats)

        llm = FakeLLM()
        with mock.patch.object(annotate_distill, "_record_seeds", spy):
            with mock.patch.object(ollama, "chat_once_json", llm):
                n = annotate_distill._seed_chunks(
                    self.con, _MODEL, False, 1, _SEEDS + 4,
                    sampler=lambda con, pending, need: pending[:need],
                    counter_key=annotate_hybrid._SEED_COUNTER_KEY)
        self.assertEqual(n, 4)
        self.assertEqual(
            calls, [(annotate_hybrid._SEED_COUNTER_KEY, _SEEDS, 4)])
        self.assertEqual(
            int(get_state(self.con, annotate_hybrid._SEED_COUNTER_KEY)), _SEEDS + 4)

    def test_ctrl_c_still_reports_and_carries_the_count_out(self):
        # The interrupt path's own semantics are unchanged by the finally: its
        # message, interrupt_exit=False re-raising rather than SystemExit(1),
        # and stats["done"] carrying the committed count out to the caller.
        stats = {"done": 0}
        llm = FakeLLM(interrupt_after=2)
        with mock.patch.object(ollama, "chat_once_json", llm):
            with mock.patch("builtins.print") as printed:
                with self.assertRaises(KeyboardInterrupt):
                    annotate_distill._seed_chunks(
                        self.con, _MODEL, False, 1, _SEEDS + 10,
                        sampler=lambda con, pending, need: pending[:need],
                        counter_key=annotate_hybrid._SEED_COUNTER_KEY,
                        interrupt_exit=False, stats=stats)
        self.assertEqual(stats["done"], 2)
        said = " ".join(str(a) for call in printed.call_args_list for a in call.args)
        self.assertIn("Interrupted -- committed 2 seed annotations", said)
        con = self.committed()
        self.assertEqual(
            int(get_state(con, annotate_hybrid._SEED_COUNTER_KEY)), _SEEDS + 2)

    def test_an_interrupt_with_the_default_becomes_a_system_exit(self):
        llm = FakeLLM(interrupt_after=1)
        with mock.patch.object(ollama, "chat_once_json", llm):
            with mock.patch("builtins.print"):
                with self.assertRaises(SystemExit):
                    annotate_distill._seed_chunks(
                        self.con, _MODEL, False, 1, _SEEDS + 10,
                        sampler=lambda con, pending, need: pending[:need])


class TestBudgetResolution(unittest.TestCase):
    def test_precedence_caller_then_env_then_backend_default(self):
        resolve = annotate_hybrid._resolve_budget_minutes
        with mock.patch.object(config, "ANNOTATE_BUDGET_MINUTES", 7):
            self.assertEqual(resolve("qwen3.5:4b", 3)[0], 3)
            self.assertEqual(resolve("qwen3.5:4b")[0], 7)
            self.assertEqual(resolve("apple")[0], 7)
        with mock.patch.object(config, "ANNOTATE_BUDGET_MINUTES", 0):
            self.assertEqual(resolve("qwen3.5:4b")[0],
                             annotate_hybrid._DEFAULT_BUDGET_MINUTES_OLLAMA)
            self.assertEqual(resolve("apple")[0],
                             annotate_hybrid._DEFAULT_BUDGET_MINUTES_APPLE)
            # 0 means "unset" in the argument too -- there is no zero-minute run.
            self.assertEqual(resolve("qwen3.5:4b", 0)[0],
                             annotate_hybrid._DEFAULT_BUDGET_MINUTES_OLLAMA)

    def test_a_negative_env_budget_is_refused_by_name(self):
        # Passed through, a negative budget is a deadline already in the past:
        # every phase declines and the run reports a spent budget, which reads
        # exactly like a machine too slow to afford one request. Clamping to 0
        # would be worse -- 0 means "unset" here, so a typo'd -25 would quietly
        # buy the 25-minute default.
        resolve = annotate_hybrid._resolve_budget_minutes
        with mock.patch.object(config, "ANNOTATE_BUDGET_MINUTES", -25):
            with self.assertRaises(SystemExit) as caught:
                resolve("qwen3.5:4b")
            self.assertIn("ANNOTATE_BUDGET_MINUTES", str(caught.exception))
            # An explicit budget still overrides the env, per the precedence.
            self.assertEqual(resolve("qwen3.5:4b", 3)[0], 3)

    def test_a_negative_caller_budget_is_refused_too(self):
        resolve = annotate_hybrid._resolve_budget_minutes
        with mock.patch.object(config, "ANNOTATE_BUDGET_MINUTES", 0):
            with self.assertRaises(SystemExit) as caught:
                resolve("qwen3.5:4b", -1)
            self.assertIn("budget_minutes", str(caught.exception))
            # ...and the zero/unset paths are untouched by the check.
            self.assertEqual(resolve("qwen3.5:4b", 0)[0],
                             annotate_hybrid._DEFAULT_BUDGET_MINUTES_OLLAMA)
            self.assertEqual(resolve("qwen3.5:4b")[0],
                             annotate_hybrid._DEFAULT_BUDGET_MINUTES_OLLAMA)


if __name__ == "__main__":
    unittest.main()
