"""Smoke tests for conversation chunking and the chunk-based query tools.

Hermetic: temp SQLite only -- no Ollama (embed_texts is stubbed) and no real
chat.db. Covers ingest.build_chunks boundary behavior (session gap, size
split + overlap carry-over, header/speaker labels, attachment rendering,
incremental tail rebuild), ingest.embed_chunks metadata, and the
query_tools chunk tools (search_conversations, find_similar_conversations,
get_conversation, get_message_context, search_messages_hybrid).
"""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import sqlite_vec

import ingest
import query_tools
from db import iso_to_day, load_vec_extension


class ChunkFixture(unittest.TestCase):
    """Temp DB seeded with contacts/handles/chats; each test adds messages.

    Fixture people: contact 1 "Alice Smith" (handle 1), contact 2 nickname
    "Bobby" (handle 2). Chat 1 is a direct chat with Alice; chat 2 is a group
    ("Potluck") with Alice and Bob.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "test.db")
        self.con = sqlite3.connect(self.db_path)
        self.addCleanup(self.con.close)
        load_vec_extension(self.con)
        ingest.create_schema(self.con)

        self.con.executemany(
            "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, nickname, "
            "organization) VALUES (?, ?, ?, ?, ?, ?)",
            [(1, "u1", "Alice", "Smith", None, None),
             (2, "u2", "Robert", "Jones", "Bobby", None)],
        )
        self.con.executemany(
            "INSERT INTO handles(id, handle_identifier, contact_id, service) "
            "VALUES (?, ?, ?, ?)",
            [(1, "+15550001", 1, "iMessage"), (2, "+15550002", 2, "iMessage")],
        )
        self.con.executemany(
            "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
            "VALUES (?, ?, ?, ?, ?)",
            [(1, "chat1", None, 0, "iMessage"), (2, "chat2", "Potluck", 1, "iMessage")],
        )
        self.con.executemany(
            "INSERT INTO chat_participants(chat_id, handle_id) VALUES (?, ?)",
            [(1, 1), (2, 1), (2, 2)],
        )
        self.con.commit()

        patcher = mock.patch.object(query_tools, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_msg(self, msg_id, chat_id, handle_id, is_from_me, text, date, has_attachment=0):
        self.con.execute(
            "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, date, "
            "has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, f"guid-{msg_id}", chat_id, handle_id, is_from_me, text, date,
             has_attachment),
        )

    def chunks(self, chat_id):
        return self.con.execute(
            "SELECT id, first_message_id, last_message_id, n_messages, text "
            "FROM chunks WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()

    def chunk_of(self, msg_id):
        return self.con.execute(
            "SELECT chunk_id FROM messages WHERE id = ?", (msg_id,)).fetchone()[0]


class TestBuildChunks(ChunkFixture):
    def test_gap_split_header_and_labels(self):
        self.add_msg(1, 1, 1, 0, "Are we still planning the potluck?", "2026-01-01T10:00:00")
        self.add_msg(2, 1, None, 1, "Yes, Presidents Day weekend", "2026-01-01T10:01:00")
        self.add_msg(3, 1, 1, 0, "Perfect, I'll look at cabins", "2026-01-01T10:03:00")
        # > 60 min gap: session boundary
        self.add_msg(4, 1, 1, 0, "Found a cabin near the lift", "2026-01-01T13:30:00")
        ingest.build_chunks(self.con)

        chunks = self.chunks(1)
        self.assertEqual(len(chunks), 2)
        first, second = chunks
        self.assertEqual(first[3], 3)   # n_messages
        self.assertEqual(second[3], 1)
        self.assertEqual((first[1], first[2]), (1, 3))
        self.assertTrue(first[4].startswith("Conversation with Alice Smith on 2026-01-01:"))
        self.assertIn("Alice Smith: Are we still planning the potluck?", first[4])
        self.assertIn("Me: Yes, Presidents Day weekend", first[4])
        self.assertEqual({self.chunk_of(m) for m in (1, 2, 3)}, {first[0]})
        self.assertEqual(self.chunk_of(4), second[0])

    def test_group_chat_header_uses_display_name(self):
        self.add_msg(10, 2, 2, 0, "Who's bringing snacks", "2026-02-01T09:00:00")
        ingest.build_chunks(self.con)
        (chunk,) = self.chunks(2)
        self.assertTrue(chunk[4].startswith("Conversation with Potluck on 2026-02-01:"))
        self.assertIn("Bobby: Who's bringing snacks", chunk[4])  # nickname wins

    def test_size_split_carries_overlap(self):
        # "Me: " + 26-char text = 30-char line (31 with newline); max_chars=100
        # fits three lines per chunk, then splits carrying the last message.
        for i in range(5):
            self.add_msg(20 + i, 1, None, 1, f"note {i}".ljust(26, "x"),
                         f"2026-01-01T10:0{i}:00")
        ingest.build_chunks(self.con, max_chars=100, overlap_messages=1)

        chunks = self.chunks(1)
        self.assertEqual(len(chunks), 2)
        first, second = chunks
        self.assertEqual((first[1], first[2]), (20, 22))
        self.assertEqual((second[1], second[2]), (22, 24))  # msg 22 carried over
        self.assertIn("note 2", second[4])
        # a carried message keeps the chunk_id of the first chunk it landed in
        self.assertEqual(self.chunk_of(22), first[0])

    def test_oversized_overlap_not_carried(self):
        # 60-char texts -> 64-char lines; max_chars=100 fits only one line per
        # chunk, so any carried overlap would overflow immediately. The
        # overlap must be dropped instead of producing chunks that duplicate
        # already-chunked messages.
        for i in range(3):
            self.add_msg(70 + i, 1, None, 1, f"long {i}".ljust(60, "x"),
                         f"2026-01-01T10:0{i}:00")
        ingest.build_chunks(self.con, max_chars=100, overlap_messages=2)

        chunks = self.chunks(1)
        self.assertEqual(len(chunks), 3)
        for chunk, msg_id in zip(chunks, (70, 71, 72)):
            self.assertEqual((chunk[1], chunk[2]), (msg_id, msg_id))
            self.assertEqual(chunk[3], 1)
            self.assertEqual(self.chunk_of(msg_id), chunk[0])

    def test_tail_rebuild_preserves_overlap_prefix(self):
        # Same layout as test_size_split_carries_overlap: tail chunk (22-24)
        # begins with carried message 22 (whose chunk_id points at the first
        # chunk). The rebuild must re-include it, not start at 23.
        for i in range(5):
            self.add_msg(20 + i, 1, None, 1, f"note {i}".ljust(26, "x"),
                         f"2026-01-01T10:0{i}:00")
        ingest.build_chunks(self.con, max_chars=100, overlap_messages=1)
        first, tail = self.chunks(1)
        self.assertEqual((tail[1], tail[2]), (22, 24))

        self.add_msg(25, 1, None, 1, "ok", "2026-01-01T10:05:00")
        ingest.build_chunks(self.con, max_chars=100, overlap_messages=1)

        chunks = self.chunks(1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], first[0])
        new_tail = chunks[1]
        self.assertEqual((new_tail[1], new_tail[2]), (22, 25))  # overlap prefix kept
        self.assertIn("note 2", new_tail[4])
        self.assertIn("Me: ok", new_tail[4])
        # the carried message still belongs to its first chunk
        self.assertEqual(self.chunk_of(22), first[0])

    def test_attachment_only_message_renders_marker(self):
        self.add_msg(30, 1, 1, 0, None, "2026-01-01T10:00:00", has_attachment=1)
        self.con.execute(
            "INSERT INTO attachments(id, message_id, filename, mime_type, transfer_name) "
            "VALUES (1, 30, '/tmp/img.heic', 'image/heic', 'img.heic')")
        self.add_msg(31, 1, None, 1, "nice photo", "2026-01-01T10:01:00")
        ingest.build_chunks(self.con)
        (chunk,) = self.chunks(1)
        self.assertIn("Alice Smith: [attachment: image/heic]", chunk[4])
        self.assertEqual(chunk[3], 2)

    def test_non_renderable_messages_stay_unchunked(self):
        self.add_msg(40, 1, 1, 0, None, "2026-01-01T10:00:00")  # no text, no attachment
        built = ingest.build_chunks(self.con)
        self.assertEqual(built, 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
        self.assertIsNone(self.chunk_of(40))

    def test_incremental_tail_rebuild_extends_conversation(self):
        self.add_msg(50, 1, 1, 0, "morning chat", "2026-01-01T10:00:00")
        self.add_msg(51, 1, None, 1, "indeed", "2026-01-01T10:01:00")
        self.add_msg(52, 1, 1, 0, "afternoon topic", "2026-01-01T14:00:00")
        ingest.build_chunks(self.con)
        first, tail = self.chunks(1)

        # a vector for the tail chunk, to verify it's cleaned up on rebuild
        ingest.build_vec_table(self.con, 4)
        self.con.execute(
            "INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, last_day) "
            "VALUES (?, ?, 1, 1, 1)",
            (tail[0], sqlite_vec.serialize_float32([1.0, 0.0, 0.0, 0.0])),
        )

        # new message 10 min after the tail -- should extend it, not split
        self.add_msg(53, 1, 1, 0, "more on that", "2026-01-01T14:10:00")
        ingest.build_chunks(self.con)

        chunks = self.chunks(1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], first[0])  # untouched non-tail chunk
        new_tail = chunks[1]
        self.assertNotEqual(new_tail[0], tail[0])  # AUTOINCREMENT: id not reused
        self.assertEqual((new_tail[1], new_tail[2]), (52, 53))
        self.assertIn("more on that", new_tail[4])
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (tail[0],)
        ).fetchone()[0], 0)

    def test_embed_chunks_writes_vectors_with_metadata(self):
        self.add_msg(60, 1, 1, 0, "let's plan the potluck", "2026-01-01T10:00:00")
        self.add_msg(61, 2, 2, 0, "snacks are handled", "2026-02-01T09:00:00")
        ingest.build_chunks(self.con)
        ingest.build_vec_table(self.con, 4)

        fake = mock.Mock(side_effect=lambda texts, kind="document":
                         [[1.0, 0.0, 0.0, 0.0]] * len(texts))
        with mock.patch.object(ingest, "embed_texts", fake):
            ingest.embed_chunks(self.con, min_chars=10, batch_size=1, max_workers=1)

        self.assertEqual(fake.call_count, 2)
        rows = self.con.execute(
            "SELECT v.chunk_id, v.chat_id, v.first_day, v.last_day, c.first_date, c.last_date "
            "FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id ORDER BY v.chat_id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        for _, chat_id, first_day, last_day, first_date, last_date in rows:
            self.assertIn(chat_id, (1, 2))
            self.assertEqual(first_day, iso_to_day(first_date))
            self.assertEqual(last_day, iso_to_day(last_date))


class TestChunkQueryTools(ChunkFixture):
    """Three chunks with hand-planted vectors: A (chat 1, potluck, Jan), B (chat 1,
    dinner, Jan), C (chat 2 group, potluck, Feb). Query vector equals A's."""

    def setUp(self):
        super().setUp()
        self.add_msg(1, 1, 1, 0, "thinking about the potluck", "2026-01-01T10:00:00")
        self.add_msg(2, 1, None, 1, "same, I want to bring dessert", "2026-01-01T10:01:00")
        self.add_msg(3, 1, 1, 0, "dinner tonight at the new place?", "2026-01-01T19:00:00")
        self.add_msg(4, 1, None, 1, "yes see you at 7", "2026-01-01T19:02:00")
        self.add_msg(5, 2, 2, 0, "the venue is booked", "2026-02-01T09:00:00")
        ingest.build_chunks(self.con)

        rows = self.con.execute(
            "SELECT id, chat_id, first_date, last_date FROM chunks ORDER BY chat_id, id"
        ).fetchall()
        (self.chunk_a, self.chunk_b, self.chunk_c) = [r[0] for r in rows]
        ingest.build_vec_table(self.con, 3)
        vectors = {self.chunk_a: [1.0, 0.0, 0.0],
                   self.chunk_b: [0.0, 1.0, 0.0],
                   self.chunk_c: [0.9, 0.1, 0.0]}
        for (cid, chat_id, first_date, last_date) in rows:
            self.con.execute(
                "INSERT INTO chunk_vectors(chunk_id, embedding, chat_id, first_day, last_day) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, sqlite_vec.serialize_float32(vectors[cid]), chat_id,
                 iso_to_day(first_date), iso_to_day(last_date)),
            )
        self.con.commit()

        patcher = mock.patch.object(
            query_tools, "embed_texts",
            lambda texts, kind="document": [[1.0, 0.0, 0.0]] * len(texts))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_search_conversations_ranks_and_labels(self):
        rows = query_tools.search_conversations("potluck plans", k=3)
        self.assertEqual([r["chunk_id"] for r in rows],
                         [self.chunk_a, self.chunk_c, self.chunk_b])
        self.assertEqual(rows[0]["participants"], "Alice Smith")
        self.assertEqual(rows[1]["participants"], "Potluck")  # display_name wins
        self.assertIn("first_message_id", rows[0])
        self.assertTrue(rows[0]["text"].startswith("Conversation with"))

    def test_search_conversations_scope_and_date_filters(self):
        # Bob (contact 2) only participates in chat 2
        rows = query_tools.search_conversations("potluck", k=3, contact_id=2)
        self.assertEqual([r["chunk_id"] for r in rows], [self.chunk_c])
        # only chat 1 has January chunks
        rows = query_tools.search_conversations("potluck", k=3, date_from="2026-01-01",
                                                date_to="2026-01-31")
        self.assertEqual({r["chunk_id"] for r in rows}, {self.chunk_a, self.chunk_b})
        # scope that matches nothing returns [] rather than over-matching
        self.assertEqual(query_tools.search_conversations("potluck", contact_id=999), [])

    def test_search_conversations_empty_scope_lists_mean_no_filter(self):
        # LLMs pass [] for an omitted param -- it must behave as unscoped,
        # not as a scope that matched nothing.
        rows = query_tools.search_conversations("potluck plans", k=3, chat_id=[], handle_id=[])
        self.assertEqual([r["chunk_id"] for r in rows],
                         [self.chunk_a, self.chunk_c, self.chunk_b])

    def test_find_similar_conversations_excludes_source(self):
        rows = query_tools.find_similar_conversations(self.chunk_a, k=2)
        self.assertEqual([r["chunk_id"] for r in rows], [self.chunk_c, self.chunk_b])
        with self.assertRaises(ValueError):
            query_tools.find_similar_conversations(99999)

    def test_get_conversation_expands_chunk(self):
        convo = query_tools.get_conversation(self.chunk_a)
        self.assertEqual(convo["chat_id"], 1)
        self.assertEqual([m["id"] for m in convo["messages"]], [1, 2])
        self.assertEqual(convo["messages"][0]["sender"], "Alice Smith")
        self.assertEqual(convo["messages"][1]["sender"], "Me")
        self.assertIsNone(query_tools.get_conversation(99999))

    def test_get_message_context_windows_and_flags(self):
        rows = query_tools.get_message_context(3, before=1, after=1)
        self.assertEqual([r["id"] for r in rows], [2, 3, 4])
        self.assertEqual([r["is_target"] for r in rows], [False, True, False])
        self.assertEqual(query_tools.get_message_context(99999), [])

    def test_hybrid_fuses_keyword_and_semantic(self):
        rows = query_tools.search_messages_hybrid("potluck", k=3)
        self.assertEqual(rows[0]["chunk_id"], self.chunk_a)  # top of both arms
        self.assertEqual(rows[0]["matched_by"], ["semantic", "keyword"])
        by_id = {r["chunk_id"]: r for r in rows}
        self.assertEqual(by_id[self.chunk_b]["matched_by"], ["semantic"])  # no 'potluck' text
        self.assertIn("rrf_score", rows[0])
        self.assertEqual(
            [r["rrf_score"] for r in rows],
            sorted((r["rrf_score"] for r in rows), reverse=True),
        )


if __name__ == "__main__":
    unittest.main()
