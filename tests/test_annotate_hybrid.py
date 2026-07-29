"""Hermetic tests for the hybrid annotate tier (annotate.py --quality hybrid)
and the pieces it is built from: annotate_hybrid's budgeted phase machine,
annotate_distill's knn_confidence + backfill, annotate_sampling's stratified
draw, annotate_events' prefilter, and the chunk_annotations.knn_confidence
migration. Temp SQLite only; ollama.chat_once_json is faked -- no Ollama, no
real DB.

The clock is faked the way the tier actually spends it: FakeClock only moves
when a (fake) LLM request charges it, so a deadline test asserts the dispatch
rule rather than racing a real wall clock.
"""

import contextlib
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

# Enough LLM rows to clear both floors that gate the kNN passes:
# annotate_distill._KNN_MIN_SEEDS (the vote refuses fewer) and
# ANNOTATE_SEED_TARGET (so the distilled tier's own seed pass is a no-op and
# a test can drive _classify_chunks straight away).
_SEEDS = max(annotate_distill._KNN_MIN_SEEDS, config.ANNOTATE_SEED_TARGET) + 10
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


class FakeLLM:
    """Stand-in for ollama.chat_once_json. Records the chunk ids of every
    request, optionally charges a clock, and can fail chosen chunks (which is
    how a dispatched-but-unwritten chunk gets tested for re-dispatch)."""

    def __init__(self, clock=None, seconds=0.0, fail_ids=(), interrupt_after=None):
        self.clock = clock
        self.seconds = seconds
        self.fail_ids = set(fail_ids)
        self.interrupt_after = interrupt_after
        self.requests = []          # ids per request
        self.order = []             # ids in dispatch order, flattened

    def __call__(self, messages, model=None, think=None, timeout=None, **kwargs):
        if self.interrupt_after is not None and len(self.requests) >= self.interrupt_after:
            raise KeyboardInterrupt
        ids = _ids_in(messages[-1]["content"])
        self.requests.append(ids)
        self.order.extend(ids)
        if self.clock is not None and self.seconds:
            self.clock.advance(self.seconds)
        if len(ids) > 1:
            return {str(cid): dict(_ANNOTATION)
                    for cid in ids if cid not in self.fail_ids}
        if not ids or ids[0] in self.fail_ids:
            return None
        return dict(_ANNOTATION)


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
        with mock.patch.object(ollama, "chat_once_json", llm):
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

    def test_limit_caps_the_total_across_both_sub_phases(self):
        self.seed_and_classify()
        llm = FakeLLM()
        self.hybrid(llm, limit=2)
        self.assertEqual(llm.order, _EVENT_IDS[:2])

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

    def test_matches_patterns_excludes_llm_rows_and_returns_unique_ascending(self):
        ids = annotate_events.event_candidate_chunk_ids(self.con, True, None)
        # 1 (distilled) and 3 (unannotated) match; 3 matches twice but appears
        # once; 2 matches nothing; 4 matches but is already 'llm'.
        self.assertEqual(ids, [1, 3, 5])

    def test_scope_restricted_to_top_dyads(self):
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, False, 1),
                         [1, 3])
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, False, 2),
                         [1, 3, 5])

    def test_llm_annotated_chunk_is_never_a_candidate(self):
        self.assertNotIn(4, annotate_events.event_candidate_chunk_ids(self.con, True, None))
        # ...and upgrading a candidate takes it out of the pool.
        self.con.execute("UPDATE chunk_annotations SET source = 'llm' WHERE chunk_id = 1")
        self.con.commit()
        self.assertEqual(annotate_events.event_candidate_chunk_ids(self.con, True, None),
                         [3, 5])


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
        for quality in ("fast", "balanced"):
            self.assertEqual(annotate.QUALITY_PRESETS[quality][0], "low")
            self.assertIs(annotate._resolve_quality(quality, "qwen3.5:4b")[0], False)
            self.assertEqual(annotate._resolve_quality(quality, "gpt-oss:20b")[0], "low")
        # None means "omit the flag" and is NOT an effort level: it survives on
        # every backend, or 'max' would stop meaning the model's own default.
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


if __name__ == "__main__":
    unittest.main()
