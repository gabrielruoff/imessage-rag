"""Hermetic tests for topics.py (sqlite-vec k-means + LLM label pass).
Temp SQLite with a REAL vec0 chunk_vectors table and hand-planted 4-dim
vectors forming two obvious clusters; ollama.chat_once is mocked."""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import sqlite_vec

import config
import ollama
import schema
import topics
from db import get_state, load_vec_extension, set_state

# random.Random(0).sample over the id-ordered chunk ids [1..6] picks seeds
# 4 and 6, so the clusters are laid out as {1, 2, 4} vs {3, 5, 6} to make
# the two initial centroids span both clusters (deterministic convergence).
CLUSTER_A_IDS = (1, 2, 4)
CLUSTER_B_IDS = (3, 5, 6)
VECTORS = {
    1: [1.0, 0.0, 0.0, 0.0],
    2: [0.98, 0.02, 0.0, 0.0],
    3: [0.0, 1.0, 0.0, 0.0],
    4: [0.99, 0.01, 0.0, 0.0],
    5: [0.02, 0.98, 0.0, 0.0],
    6: [0.01, 0.99, 0.0, 0.0],
}


def _reply(payload):
    return {"role": "assistant", "content": json.dumps(payload)}


class TopicsFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "test.db")
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        load_vec_extension(self.con)
        schema.create_schema(self.con)
        # Same DDL shape as chunking.build_vec_table, at test dimension 4.
        self.con.execute("""
            CREATE VIRTUAL TABLE chunk_vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[4] distance_metric=cosine,
                chat_id INTEGER,
                first_day INTEGER,
                last_day INTEGER
            )
        """)
        set_state(self.con, "embed_model", "nomic-embed-text")
        set_state(self.con, "embed_dim", 4)
        self.con.commit()

        patcher = mock.patch.object(topics, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def seed_vectors(self):
        for chunk_id, vec in VECTORS.items():
            topic = "potluck" if chunk_id in CLUSTER_A_IDS else "trivia night"
            self.con.execute(
                "INSERT INTO chunks(id, chat_id, first_message_id, last_message_id, "
                "first_date, last_date, n_messages, text) "
                "VALUES (?, 1, ?, ?, '2026-01-01T10:00:00', '2026-01-01T10:05:00', "
                "2, ?)",
                (chunk_id, chunk_id * 10, chunk_id * 10 + 1,
                 f"Conversation about the {topic} (chunk {chunk_id})"),
            )
            self.con.execute(
                "INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, "
                "last_day) VALUES (?, ?, 1, 1, 1)",
                (chunk_id, sqlite_vec.serialize_float32(vec)),
            )
        self.con.commit()

    def cluster_of(self):
        return dict(self.con.execute(
            "SELECT chunk_id, cluster_id FROM chunk_topics"))

    def topic_rows(self):
        return self.con.execute(
            "SELECT cluster_id, label, n_chunks FROM topics ORDER BY cluster_id"
        ).fetchall()


class TestClustering(TopicsFixture):
    def test_kmeans_groups_persist_and_state(self):
        self.seed_vectors()
        replies = [_reply({"label": "Potluck Planning"}),
                   _reply({"label": "Trivia Plans"})]
        with mock.patch.object(ollama, "chat_once", side_effect=replies):
            topics.main(["--k", "2"])

        assign = self.cluster_of()
        self.assertEqual(set(assign), set(VECTORS))
        group_a = {assign[i] for i in CLUSTER_A_IDS}
        group_b = {assign[i] for i in CLUSTER_B_IDS}
        self.assertEqual(len(group_a), 1)
        self.assertEqual(len(group_b), 1)
        self.assertNotEqual(group_a, group_b)

        rows = self.topic_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r[2] for r in rows}, {3})          # n_chunks = 3 each
        self.assertEqual({r[1] for r in rows},
                         {"Potluck Planning", "Trivia Plans"})
        self.assertEqual(get_state(self.con, "topic_model_k"),
                         "nomic-embed-text:4:2")
        # scratch centroid table cleaned up
        self.assertIsNone(self.con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = '_topic_centroids'"
        ).fetchone())

    def test_k_clamped_to_chunk_count(self):
        self.seed_vectors()
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"label": "Something"})):
            topics.main(["--k", "50"])
        self.assertEqual(get_state(self.con, "topic_model_k"),
                         "nomic-embed-text:4:6")
        self.assertLessEqual(len(self.topic_rows()), 6)

    def test_empty_vectors_early_exit(self):
        with mock.patch.object(ollama, "chat_once") as fake:
            topics.main([])
        fake.assert_not_called()
        self.assertEqual(self.topic_rows(), [])
        self.assertEqual(self.cluster_of(), {})
        self.assertIsNone(get_state(self.con, "topic_model_k"))


class TestLabelPass(TopicsFixture):
    def run_clustered(self, replies):
        self.seed_vectors()
        with mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            topics.main(["--k", "2"])
        return fake

    def test_labels_persisted_and_truncated(self):
        long_label = "An Extremely Verbose Topic Label That Runs On And On Well Past Sixty"
        self.run_clustered([_reply({"label": long_label}),
                            _reply({"label": "  Trivia Plans  "})])
        labels = {r[1] for r in self.topic_rows()}
        self.assertIn("Trivia Plans", labels)              # stripped
        self.assertTrue(any(len(label) == 60 for label in labels))

    def test_malformed_label_retries_then_skips_and_resumes(self):
        # One cluster's label response is garbage twice -> retried once,
        # then skipped (stays NULL); the other labels fine.
        bad = {"role": "assistant", "content": "not json {"}
        fake = self.run_clustered([bad, bad, _reply({"label": "Trivia Plans"})])
        self.assertEqual(fake.call_count, 3)
        labels = [r[1] for r in self.topic_rows()]
        self.assertIn(None, labels)
        self.assertIn("Trivia Plans", labels)
        # the default label pass is resumable: only the NULL cluster is sent
        con = sqlite3.connect(self.db_path)
        self.addCleanup(con.close)
        load_vec_extension(con)
        with mock.patch.object(ollama, "chat_once",
                               return_value=_reply({"label": "Potluck"})) as fake2:
            topics.label_clusters(con)
        self.assertEqual(fake2.call_count, 1)
        self.assertEqual(sorted(r[1] for r in self.topic_rows()),
                         ["Potluck", "Trivia Plans"])

    def test_relabel_renames_all_clusters_without_reclustering(self):
        self.run_clustered([_reply({"label": "Old A"}), _reply({"label": "Old B"})])
        assign_before = self.cluster_of()
        replies = [_reply({"label": "New A"}), _reply({"label": "New B"})]
        with mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            topics.main(["--relabel"])
        self.assertEqual(fake.call_count, 2)
        self.assertEqual({r[1] for r in self.topic_rows()}, {"New A", "New B"})
        self.assertEqual(self.cluster_of(), assign_before)   # assignments untouched


class TestLabelModelPlumbing(TopicsFixture):
    """The label pass used to accept a model= it never passed, silently
    falling through to CHAT_MODEL inside ollama.chat_once."""

    def test_label_pass_uses_topic_config(self):
        # Non-default sentinels: with the real defaults TOPIC_THINK is None and
        # TOPIC_TIMEOUT is CHAT_TIMEOUT, so asserting against them would pass
        # even if _ask_json hardcoded the values or read the CHAT_* vars.
        # assertIs on think, not assertEqual -- False vs None is the whole point
        # of the tri-state, and False == 0 would not catch a collapsed sentinel.
        self.seed_vectors()
        replies = [_reply({"label": "A"}), _reply({"label": "B"})]
        with mock.patch.object(config, "TOPIC_MODEL", "topic-sentinel"), \
             mock.patch.object(config, "TOPIC_THINK", False), \
             mock.patch.object(config, "TOPIC_TIMEOUT", 999), \
             mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            topics.main(["--k", "2"])
        self.assertEqual(fake.call_count, 2)
        for call in fake.call_args_list:
            self.assertEqual(call.kwargs["model"], "topic-sentinel")
            self.assertEqual(call.kwargs["timeout"], 999)
            self.assertIs(call.kwargs["think"], False)
            self.assertEqual(call.kwargs["format"], "json")

    def test_model_flag_overrides_topic_model(self):
        self.seed_vectors()
        replies = [_reply({"label": "A"}), _reply({"label": "B"})]
        with mock.patch.object(ollama, "chat_once", side_effect=replies) as fake:
            topics.main(["--k", "2", "--model", "some-other-model"])
        for call in fake.call_args_list:
            self.assertEqual(call.kwargs["model"], "some-other-model")

    def test_relabel_honors_model_flag(self):
        self.seed_vectors()
        with mock.patch.object(ollama, "chat_once",
                               side_effect=[_reply({"label": "A"}),
                                            _reply({"label": "B"})]):
            topics.main(["--k", "2"])
        with mock.patch.object(ollama, "chat_once",
                               side_effect=[_reply({"label": "C"}),
                                            _reply({"label": "D"})]) as fake:
            topics.main(["--relabel", "--model", "relabel-model"])
        for call in fake.call_args_list:
            self.assertEqual(call.kwargs["model"], "relabel-model")


if __name__ == "__main__":
    unittest.main()
