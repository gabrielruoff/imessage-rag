"""Lightweight smoke tests for the contacts refactor.

No Ollama and no real chat.db required -- everything runs against temporary /
in-memory SQLite databases. Covers the delicate bits of PR #1:
  - the in-place handles-table migration in ingest.create_schema
  - FK-safe, idempotent re-ingest of contacts + handles
  - the two query_tools payload/behavior fixes (source_uniqueid, SQL-side
    list_chats filter+limit)

Run from the repo root:  python3 -m unittest discover tests
"""

import datetime
import os
import sqlite3
import tempfile
import unittest

import ingest
import query_tools


class MigrationTest(unittest.TestCase):
    def test_old_handles_layout_migrates_in_place(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        # Old schema: contact_id held the raw phone/email; name/org denormalized.
        con.execute(
            """CREATE TABLE handles (
                id INTEGER PRIMARY KEY, contact_id TEXT, service TEXT,
                first_name TEXT, last_name TEXT, nickname TEXT, organization TEXT)"""
        )
        con.execute(
            "INSERT INTO handles VALUES (5, '+12015551234', 'iMessage', "
            "'Jane', 'Doe', NULL, 'Acme')"
        )
        con.commit()

        ingest.create_schema(con)

        cols = {row[1]: row[2] for row in con.execute("PRAGMA table_info(handles)")}
        # Raw handle string preserved under its new column name.
        self.assertIn("handle_identifier", cols)
        # contact_id is now an integer FK, no longer TEXT.
        self.assertEqual(cols.get("contact_id"), "INTEGER")
        # Denormalized name/org columns dropped.
        for gone in ("first_name", "last_name", "nickname", "organization"):
            self.assertNotIn(gone, cols)

        row = con.execute(
            "SELECT handle_identifier, contact_id FROM handles WHERE id = 5"
        ).fetchone()
        self.assertEqual(row[0], "+12015551234")
        self.assertIsNone(row[1])  # reset to NULL, to be repopulated by ingest_handles


class ReingestTest(unittest.TestCase):
    def _fake_src(self):
        src = sqlite3.connect(":memory:")
        self.addCleanup(src.close)
        src.execute("CREATE TABLE handle (id TEXT, service TEXT)")  # ROWID implicit
        src.executemany(
            "INSERT INTO handle(id, service) VALUES (?, ?)",
            [("+12015551234", "iMessage"), ("friend@example.com", "iMessage")],
        )
        src.commit()
        return src

    def test_reingest_is_fk_safe_and_idempotent(self):
        tgt = sqlite3.connect(":memory:")
        self.addCleanup(tgt.close)
        tgt.execute("PRAGMA foreign_keys = ON")
        ingest.create_schema(tgt)

        contacts = [(1, "uid-1", "Jane", "Doe", None, "Acme")]
        # Only the phone maps to a contact; the email handle stays contact_id NULL.
        handle_to_contact = {ingest.normalize_handle_id("+12015551234"): 1}

        def run_once():
            src = self._fake_src()
            ingest.ingest_contacts(tgt, contacts)
            ingest.ingest_handles(tgt=tgt, src=src, contacts=contacts,
                                  handle_to_contact=handle_to_contact)
            tgt.commit()

        run_once()
        counts_1 = (
            tgt.execute("SELECT COUNT(*) FROM contacts").fetchone()[0],
            tgt.execute("SELECT COUNT(*) FROM handles").fetchone()[0],
        )
        # Second run must not raise (INSERT OR REPLACE on the integer-PK parent
        # is an in-place replace, so the handles.contact_id FK never dangles).
        run_once()
        counts_2 = (
            tgt.execute("SELECT COUNT(*) FROM contacts").fetchone()[0],
            tgt.execute("SELECT COUNT(*) FROM handles").fetchone()[0],
        )
        self.assertEqual(counts_1, (1, 2))
        self.assertEqual(counts_1, counts_2)  # stable, purely additive

        linked = tgt.execute(
            "SELECT contact_id FROM handles WHERE handle_identifier = '+12015551234'"
        ).fetchone()[0]
        unlinked = tgt.execute(
            "SELECT contact_id FROM handles WHERE handle_identifier = 'friend@example.com'"
        ).fetchone()[0]
        self.assertEqual(linked, 1)
        self.assertIsNone(unlinked)


def _sql_rows(**kwargs):
    """search_messages_sql returns an envelope; most tests only need rows."""
    return query_tools.search_messages_sql(**kwargs)["rows"]


def _fts_rows(query, **kwargs):
    return query_tools.search_messages_fts(query, **kwargs)["rows"]


class QueryToolsTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))

        con = sqlite3.connect(self.db_path)
        try:
            ingest.create_schema(con)
            con.execute(
                "INSERT INTO contacts(id, source_uniqueid, first_name, last_name, "
                "nickname, organization) VALUES (2, 'uid-2', 'Alice', 'Smith', NULL, 'Globex')"
            )
            # Alice owns two handles (phone + email) and appears in both a
            # direct chat (100, via her phone) and a group chat (101, via her
            # email) -- this is the "mom has multiple handles/chats" shape
            # that contact_id-scoped filtering needs to collapse into one call.
            con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (50, '+19995551234', 2, 'iMessage')"
            )
            con.execute(
                "INSERT INTO handles(id, handle_identifier, contact_id, service) "
                "VALUES (51, 'alice@example.com', 2, 'iMessage')"
            )
            con.executemany(
                "INSERT INTO chats(id, chat_identifier, display_name, is_group, service) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (100, "chat100", None, 0, "iMessage"),
                    (101, "chat101", None, 1, "iMessage"),
                    (102, "special-id", None, 0, "iMessage"),
                ],
            )
            con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (100, 50)")
            con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (101, 51)")
            # 1001 sits mid-day with sub-second precision, matching how
            # apple_ns_to_iso really stores dates -- an all-midnight fixture
            # can't tell an inclusive date_to from one that drops the day.
            con.executemany(
                "INSERT INTO messages(id, guid, chat_id, handle_id, is_from_me, text, "
                "date, has_attachment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (1000, "guid-1000", 100, 50, 0, "hi from phone",
                     "2026-01-01T00:00:00", 0),
                    (1001, "guid-1001", 100, 50, 1, "hi back",
                     "2026-01-02T13:46:50.648660", 1),
                    (1002, "guid-1002", 101, 51, 0, "hi from email in group",
                     "2026-01-03T00:00:00", 1),
                ],
            )
            # Two attachments on one message, so attachment_count can't be
            # mistaken for a count of messages-carrying-attachments.
            con.executemany(
                "INSERT INTO attachments(id, message_id, filename, mime_type, transfer_name) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (3000, 1001, "/tmp/a.jpg", "image/jpeg", "a.jpg"),
                    (3001, 1001, "/tmp/b.jpg", "image/jpeg", "b.jpg"),
                    (3002, 1002, "/tmp/c.pdf", "application/pdf", "c.pdf"),
                ],
            )
            # Reaction 2002 is_from_me=1 carries the counterparty's handle_id,
            # matching real ingested data -- that's what makes handle/contact
            # scoping work for messages and reactions I sent.
            con.executemany(
                "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
                "is_from_me, associated_message_type, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (2000, "guid-1000", 100, 50, 0, 2000, "2026-01-01T00:05:00"),
                    (2001, "guid-1002", 101, 51, 0, 2000, "2026-01-03T00:05:00"),
                    (2002, "guid-1000", 100, 50, 1, 2000, "2026-01-01T00:06:00"),
                ],
            )
            con.commit()
        finally:
            con.close()

        # Point the query layer at this temp DB (_connect reads the module global).
        self._saved_db_path = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved_db_path)

    def test_search_contacts_by_name_includes_source_uniqueid(self):
        results = query_tools.search_contacts_by_name("Alice")
        self.assertEqual(len(results), 1)
        self.assertIn("source_uniqueid", results[0])
        self.assertEqual(results[0]["source_uniqueid"], "uid-2")

    def test_list_chats_respects_limit(self):
        self.assertLessEqual(len(query_tools.list_chats(limit=2)), 2)

    def test_list_chats_filters_by_participant_name(self):
        # Alice now owns two handles (see setUp) and appears in both chat 100
        # (direct) and chat 101 (group) via her phone and email respectively.
        chats = query_tools.list_chats(query="Alice")
        self.assertEqual({c["id"] for c in chats}, {100, 101})

    def test_list_chats_filters_by_chat_identifier(self):
        chats = query_tools.list_chats(query="special")
        self.assertEqual([c["id"] for c in chats], [102])

    def test_search_messages_sql_handle_id_list_unions_handles(self):
        rows = _sql_rows(handle_id=[50, 51])
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_contact_id_resolves_all_handles(self):
        rows = _sql_rows(contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_chat_id_list(self):
        rows = _sql_rows(chat_id=[100, 101])
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

    def test_search_messages_sql_single_chat_id_still_works(self):
        rows = _sql_rows(chat_id=100)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001})

    def test_list_chats_filters_by_contact_id(self):
        chats = query_tools.list_chats(contact_id=2)
        self.assertEqual({c["id"] for c in chats}, {100, 101})

    def test_list_chats_filters_by_handle_id(self):
        chats = query_tools.list_chats(handle_id=50)
        self.assertEqual({c["id"] for c in chats}, {100})

    def test_list_chats_empty_handle_id_list_means_no_filter(self):
        self.assertEqual(
            {c["id"] for c in query_tools.list_chats(handle_id=[])},
            {c["id"] for c in query_tools.list_chats()},
        )

    def test_get_chat_summary_stats_with_chat_id_list(self):
        stats = query_tools.get_chat_summary_stats(chat_id=[100, 101])
        self.assertEqual(stats["message_count"], 3)

    def test_get_chat_summary_stats_with_contact_id(self):
        stats = query_tools.get_chat_summary_stats(contact_id=2)
        self.assertEqual(stats["message_count"], 3)
        self.assertEqual(stats["from_me"], 1)
        self.assertEqual(stats["from_them"], 2)

    def test_get_chat_summary_stats_unscoped_aggregates_whole_db(self):
        stats = query_tools.get_chat_summary_stats()
        self.assertEqual(stats["message_count"], 3)

    def test_get_chat_summary_stats_empty_scope_list_means_no_filter(self):
        stats = query_tools.get_chat_summary_stats(handle_id=[])
        self.assertEqual(stats["message_count"], 3)

    def test_get_chat_summary_stats_unknown_contact_returns_zeros(self):
        stats = query_tools.get_chat_summary_stats(contact_id=999999)
        self.assertEqual(stats["message_count"], 0)

    def test_get_message_reactions_single_guid_still_works(self):
        # guid-1000 carries two reactions (theirs and mine) -- see setUp.
        rows = query_tools.get_message_reactions("guid-1000")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["target_message_guid"] for r in rows}, {"guid-1000"})

    def test_get_message_reactions_batch_guids(self):
        rows = query_tools.get_message_reactions(["guid-1000", "guid-1002"])
        self.assertEqual(
            {(r["target_message_guid"]) for r in rows}, {"guid-1000", "guid-1002"}
        )

    def test_search_messages_fts_scopes_by_contact_id(self):
        rows = _fts_rows("hi", contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1000, 1001, 1002})

        rows = _fts_rows("email", contact_id=2)
        self.assertEqual({r["id"] for r in rows}, {1002})

    def test_search_messages_sql_envelope_reports_truncation(self):
        result = query_tools.search_messages_sql(chat_id=[100, 101], limit=2)
        self.assertEqual(result["total_matching"], 3)
        self.assertEqual(result["returned"], 2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["order"], "newest-first")
        self.assertTrue(result["truncated"])
        # newest-first: the two most recent of the three messages
        self.assertEqual([r["id"] for r in result["rows"]], [1002, 1001])

    def test_search_messages_sql_envelope_untruncated(self):
        result = query_tools.search_messages_sql(chat_id=100)
        self.assertEqual(result["total_matching"], 2)
        self.assertFalse(result["truncated"])

    def test_search_messages_sql_empty_scope_returns_empty_envelope(self):
        result = query_tools.search_messages_sql(contact_id=999999)
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["total_matching"], 0)
        self.assertFalse(result["truncated"])

    def test_search_messages_fts_envelope_shape(self):
        result = query_tools.search_messages_fts("hi", limit=1)
        self.assertEqual(result["total_matching"], 3)
        self.assertEqual(result["returned"], 1)
        self.assertEqual(result["order"], "best-match-first")
        self.assertTrue(result["truncated"])

    def test_get_message_reactions_labels_tapback_name(self):
        rows = query_tools.get_message_reactions("guid-1000")
        self.assertEqual({r["reaction"] for r in rows}, {"love"})
        self.assertEqual({r["associated_message_type"] for r in rows}, {2000})

    def test_get_chat_summary_stats_counts_individual_attachments(self):
        stats = query_tools.get_chat_summary_stats(chat_id=100)
        self.assertEqual(stats["message_count"], 2)
        # Chat 100 has one message carrying attachments, but two attachments on
        # it -- attachment_count counts the attachments, not the messages.
        self.assertEqual(stats["attachment_count"], 2)
        # chat 100 has two reactions, both on guid-1000 (see setUp).
        self.assertEqual(stats["reactions_sent_count"], 2)

    def test_get_chat_summary_stats_is_from_me_filters(self):
        stats = query_tools.get_chat_summary_stats(chat_id=100, is_from_me=True)
        self.assertEqual(stats["message_count"], 1)
        self.assertEqual(stats["from_me"], 1)
        self.assertEqual(stats["from_them"], 0)

    def test_get_chat_summary_stats_is_from_me_scopes_every_count(self):
        # is_from_me narrows attachments and reactions too, not just messages:
        # of chat 100's two reactions only 2002 is mine, and both attachments
        # hang off 1001, which I sent.
        mine = query_tools.get_chat_summary_stats(chat_id=100, is_from_me=True)
        self.assertEqual(mine["reactions_sent_count"], 1)
        self.assertEqual(mine["attachment_count"], 2)

        theirs = query_tools.get_chat_summary_stats(chat_id=100, is_from_me=False)
        self.assertEqual(theirs["reactions_sent_count"], 1)
        self.assertEqual(theirs["attachment_count"], 0)

    def test_get_chat_summary_stats_group_by_chat(self):
        rows = query_tools.get_chat_summary_stats(chat_id=[100, 101], group_by_chat=True)
        by_chat = {r["chat_id"]: r for r in rows}
        self.assertEqual(set(by_chat), {100, 101})
        self.assertEqual(by_chat[100]["message_count"], 2)
        self.assertEqual(by_chat[100]["reactions_sent_count"], 2)
        self.assertEqual(by_chat[100]["attachment_count"], 2)
        self.assertEqual(by_chat[101]["message_count"], 1)
        self.assertEqual(by_chat[101]["reactions_sent_count"], 1)
        self.assertEqual(by_chat[101]["attachment_count"], 1)

    def test_get_chat_summary_stats_group_by_chat_ranks_and_limits(self):
        # Unscoped: chat 100 (2 msgs) outranks chat 101 (1 msg); chat 102 has
        # no messages so it never produces a row.
        rows = query_tools.get_chat_summary_stats(group_by_chat=True)
        self.assertEqual([r["chat_id"] for r in rows], [100, 101])

        top = query_tools.get_chat_summary_stats(group_by_chat=True, limit=1)
        self.assertEqual([r["chat_id"] for r in top], [100])
        # The per-chat subqueries still correlate correctly through the
        # ranking CTE.
        self.assertEqual(top[0]["message_count"], 2)
        self.assertEqual(top[0]["reactions_sent_count"], 2)
        self.assertEqual(top[0]["attachment_count"], 2)

    def test_get_chat_summary_stats_date_range_filters(self):
        # chat 100: msg 1000 @ 2026-01-01T00:00:00 (reactions @ 00:05, 00:06),
        # msg 1001 @ 2026-01-02T13:46:50.648660 (see setUp). date_from excludes
        # 1000 and both its reactions.
        stats = query_tools.get_chat_summary_stats(chat_id=100, date_from="2026-01-02")
        self.assertEqual(stats["message_count"], 1)
        self.assertEqual(stats["reactions_sent_count"], 0)

        # A bare date_to covers the whole day named -- no T23:59:59 needed.
        stats = query_tools.get_chat_summary_stats(chat_id=100, date_to="2026-01-01")
        self.assertEqual(stats["message_count"], 1)
        self.assertEqual(stats["reactions_sent_count"], 2)

    def test_get_chat_summary_stats_group_by_chat_respects_date_range(self):
        rows = query_tools.get_chat_summary_stats(
            chat_id=[100, 101], date_from="2026-01-02", group_by_chat=True
        )
        by_chat = {r["chat_id"]: r for r in rows}
        self.assertEqual(by_chat[100]["message_count"], 1)
        self.assertEqual(by_chat[100]["reactions_sent_count"], 0)
        self.assertEqual(by_chat[101]["message_count"], 1)
        self.assertEqual(by_chat[101]["reactions_sent_count"], 1)

    def test_search_messages_sql_date_from_supports_full_datetime(self):
        rows = _sql_rows(chat_id=100, date_from="2026-01-01T12:00:00")
        self.assertEqual({r["id"] for r in rows}, {1001})

    def test_search_messages_sql_bare_date_to_covers_whole_day(self):
        # The regression: 1001 sits at 13:46 on Jan 2, so a date_to of
        # '2026-01-02' compared as a plain string would drop it.
        rows = _sql_rows(chat_id=100, date_to="2026-01-02")
        self.assertEqual({r["id"] for r in rows}, {1000, 1001})

    def test_search_messages_sql_single_day_range(self):
        # date_from == date_to is how "what did we say on Jan 2" arrives.
        rows = _sql_rows(
            chat_id=100, date_from="2026-01-02", date_to="2026-01-02"
        )
        self.assertEqual({r["id"] for r in rows}, {1001})

    def test_date_to_is_inclusive_of_the_unit_named(self):
        # 1001 is at 13:46:50.648660. A second-granularity bound includes its
        # own second; the second before it does not.
        rows = _sql_rows(chat_id=100, date_to="2026-01-02T13:46:50")
        self.assertIn(1001, {r["id"] for r in rows})

        rows = _sql_rows(chat_id=100, date_to="2026-01-02T13:46:49")
        self.assertNotIn(1001, {r["id"] for r in rows})

    def test_date_bounds_accept_utc_offset(self):
        # 08:46:50-05:00 is the same instant as the stored 13:46:50 UTC, so an
        # offset bound must select exactly what its UTC equivalent selects --
        # the tool converts rather than trusting the caller to have done it.
        offset = _sql_rows(
            chat_id=100, date_from="2026-01-02T08:46:50-05:00"
        )
        utc = _sql_rows(chat_id=100, date_from="2026-01-02T13:46:50")
        self.assertEqual({r["id"] for r in offset}, {r["id"] for r in utc})
        self.assertEqual({r["id"] for r in offset}, {1001})

    def test_date_bounds_accept_z_suffix(self):
        rows = _sql_rows(chat_id=100, date_from="2026-01-02T13:46:50Z")
        self.assertEqual({r["id"] for r in rows}, {1001})

    def test_date_to_with_space_separator_keeps_second_granularity(self):
        # fromisoformat accepts a space in place of 'T'. The bound must not be
        # mistaken for a bare date and advanced by a whole day -- '...13:46:49'
        # names a second, so 1001 at 13:46:50.648660 sits outside it.
        rows = _sql_rows(chat_id=100, date_to="2026-01-02 13:46:49")
        self.assertNotIn(1001, {r["id"] for r in rows})

        rows = _sql_rows(chat_id=100, date_to="2026-01-02 13:46:50")
        self.assertIn(1001, {r["id"] for r in rows})

    def test_date_to_with_comma_decimal_mark_keeps_microsecond_granularity(self):
        # ISO 8601 allows ',' as the decimal mark and fromisoformat accepts it.
        # A fractional bound advances by 1 microsecond, so ',648659' ends just
        # before 1001's .648660 -- a seconds-granularity misread would let a
        # whole extra second in.
        rows = _sql_rows(
            chat_id=100, date_to="2026-01-02T13:46:50,648659"
        )
        self.assertNotIn(1001, {r["id"] for r in rows})

    def test_date_to_compact_form_keeps_second_granularity(self):
        # Compact ISO ('20260102T134649') has no colons to count -- it must
        # still read as second granularity, not hour.
        rows = _sql_rows(chat_id=100, date_to="20260102T134649")
        self.assertNotIn(1001, {r["id"] for r in rows})

        rows = _sql_rows(chat_id=100, date_to="20260102T134650")
        self.assertIn(1001, {r["id"] for r in rows})

    def test_malformed_date_bound_raises_rather_than_matching_nothing(self):
        # A silent empty result would read as "no messages that day" to the
        # model; the ValueError reaches it as a tool error instead.
        with self.assertRaises(ValueError):
            _sql_rows(chat_id=100, date_from="last tuesday")
        with self.assertRaises(ValueError):
            query_tools.get_chat_summary_stats(chat_id=100, date_to="01/02/2026")

    def test_get_chat_summary_stats_group_by_chat_with_empty_scope(self):
        self.assertEqual(
            query_tools.get_chat_summary_stats(contact_id=999, group_by_chat=True), []
        )


class ReactionGuidNormalizationTest(unittest.TestCase):
    """chat.db's associated_message_guid carries a part prefix ('p:0/<guid>',
    'bp:<guid>') that messages.guid never has -- reactions must be stored
    with the bare guid or they join to nothing."""

    def test_strip_reaction_guid_prefix(self):
        self.assertEqual(ingest.strip_reaction_guid_prefix("p:0/GUID-A"), "GUID-A")
        self.assertEqual(ingest.strip_reaction_guid_prefix("p:13/GUID-A"), "GUID-A")
        self.assertEqual(ingest.strip_reaction_guid_prefix("bp:GUID-B"), "GUID-B")
        self.assertEqual(ingest.strip_reaction_guid_prefix("GUID-C"), "GUID-C")
        self.assertIsNone(ingest.strip_reaction_guid_prefix(None))

    def test_create_schema_migrates_prefixed_guids_in_place(self):
        con = sqlite3.connect(":memory:")
        ingest.create_schema(con)
        con.executemany(
            "INSERT INTO reactions(id, target_message_guid, chat_id, handle_id, "
            "is_from_me, associated_message_type, date) VALUES (?, ?, 1, 0, 1, 2000, "
            "'2026-01-01T00:00:00')",
            [(1, "p:0/GUID-A"), (2, "bp:GUID-B"), (3, "GUID-C"), (4, "p:2/GUID-D")],
        )
        ingest.create_schema(con)  # re-running applies the idempotent migration
        got = {r[0] for r in con.execute("SELECT target_message_guid FROM reactions")}
        self.assertEqual(got, {"GUID-A", "GUID-B", "GUID-C", "GUID-D"})


class LookbackParseTest(unittest.TestCase):
    """The --lookback period grammar and calendar arithmetic (pure helpers)."""

    def test_parse_accepts_all_units_and_is_lenient(self):
        self.assertEqual(ingest._parse_lookback("1d"), (1, "d"))
        self.assertEqual(ingest._parse_lookback("10w"), (10, "w"))
        self.assertEqual(ingest._parse_lookback("3mo"), (3, "mo"))
        self.assertEqual(ingest._parse_lookback("2y"), (2, "y"))
        # Surrounding whitespace and case are tolerated.
        self.assertEqual(ingest._parse_lookback(" 7D "), (7, "d"))
        self.assertEqual(ingest._parse_lookback("4MO"), (4, "mo"))
        # None is the "flag not given" sentinel -- passes straight through.
        self.assertIsNone(ingest._parse_lookback(None))

    def test_parse_rejects_malformed_periods(self):
        # Bare 'm' is rejected on purpose (minutes/months ambiguity); junk,
        # trailing text, and non-positive counts all raise a clean SystemExit.
        for bad in ("abc", "3m", "5mos", "mo", "3", "", "0d", "-1d", "1.5y"):
            with self.assertRaises(SystemExit):
                ingest._parse_lookback(bad)

    def test_subtract_period_days_and_weeks(self):
        when = datetime.datetime(2026, 7, 22, 15, 30, 45)
        self.assertEqual(ingest._subtract_period(when, 10, "d").date(),
                         datetime.date(2026, 7, 12))
        self.assertEqual(ingest._subtract_period(when, 1, "w").date(),
                         datetime.date(2026, 7, 15))

    def test_subtract_period_months_and_years_are_calendar_correct(self):
        when = datetime.datetime(2026, 7, 22, 15, 30, 45)
        self.assertEqual(ingest._subtract_period(when, 3, "mo").date(),
                         datetime.date(2026, 4, 22))
        self.assertEqual(ingest._subtract_period(when, 2, "y").date(),
                         datetime.date(2024, 7, 22))
        # Crossing a year boundary via months.
        self.assertEqual(
            ingest._subtract_period(datetime.datetime(2026, 1, 15), 3, "mo").date(),
            datetime.date(2025, 10, 15))

    def test_subtract_period_clamps_short_month_and_leap_day(self):
        # Mar 31 - 1mo has no Feb 31; clamp to the last valid day of the target
        # month (28 in a common year, 29 in a leap year).
        self.assertEqual(
            ingest._subtract_period(datetime.datetime(2026, 3, 31), 1, "mo").date(),
            datetime.date(2026, 2, 28))
        self.assertEqual(
            ingest._subtract_period(datetime.datetime(2024, 3, 31), 1, "mo").date(),
            datetime.date(2024, 2, 29))

    def test_cutoff_ns_round_trips_and_preserves_time_of_day(self):
        now = datetime.datetime(2026, 7, 22, 12, 0, 0)
        ns = ingest._lookback_cutoff_ns((3, "mo"), now=now)
        # The ns lower bound decodes back through the real ingest path to the
        # exact cutoff instant (not just the date), so mid-day messages on the
        # boundary day sort correctly against it.
        self.assertTrue(ingest.apple_ns_to_iso(ns).startswith("2026-04-22T12:00:00"))
        # No period -> no filter.
        self.assertIsNone(ingest._lookback_cutoff_ns(None))


class LookbackIngestTest(unittest.TestCase):
    """--lookback threaded through the real source read (messages, reactions,
    attachments), and its composition with the chat allowlist."""

    def _apple_ns(self, iso):
        """Naive-UTC ISO -> Apple-epoch ns, mirroring ingest's own math."""
        delta = datetime.datetime.fromisoformat(iso) - ingest.APPLE_EPOCH
        return (delta.days * 86400 + delta.seconds) * 1_000_000_000 \
            + delta.microseconds * 1000

    def _fake_src(self):
        """A minimal chat.db-shaped source: message + chat_message_join, plus
        the attachment join tables the attachment reader touches."""
        src = sqlite3.connect(":memory:")
        self.addCleanup(src.close)
        src.executescript("""
            CREATE TABLE message(
                ROWID INTEGER PRIMARY KEY, guid TEXT, handle_id INT, is_from_me INT,
                text TEXT, attributedBody BLOB, date INT, date_read INT,
                date_delivered INT, item_type INT, is_emote INT,
                cache_has_attachments INT, associated_message_type INT,
                associated_message_guid TEXT);
            CREATE TABLE chat_message_join(message_id INT, chat_id INT);
            CREATE TABLE attachment(
                ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT,
                transfer_name TEXT);
            CREATE TABLE message_attachment_join(attachment_id INT, message_id INT);
        """)
        # (rowid, chat, date_iso|None, associated_message_type, target_guid)
        rows = [
            (1, 10, "2025-01-01T00:00:00", 0, "g1"),      # old message   -> drop
            (2, 10, "2026-06-01T00:00:00", 0, "g2"),      # recent message-> keep
            (3, 20, "2026-06-02T00:00:00", 0, "g3"),      # recent, chat 20-> keep
            (4, 10, "2026-06-03T00:00:00", 2000, "g2"),   # recent reaction-> keep
            (5, 10, "2024-01-01T00:00:00", 2000, "g1"),   # old reaction  -> drop
            (6, 10, None, 0, "g6"),                       # dateless/system-> drop
        ]
        for rowid, chat, diso, atype, aguid in rows:
            src.execute(
                "INSERT INTO message(ROWID, guid, handle_id, is_from_me, text, "
                "attributedBody, date, date_read, date_delivered, item_type, "
                "is_emote, cache_has_attachments, associated_message_type, "
                "associated_message_guid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rowid, f"guid{rowid}", 1, 0, f"msg{rowid}", None,
                 self._apple_ns(diso) if diso else None, 0, 0, 0, 0, 0, atype, aguid),
            )
            src.execute("INSERT INTO chat_message_join(message_id, chat_id) "
                        "VALUES (?, ?)", (rowid, chat))
        # One attachment on the old message (should drop) and one on a recent
        # message (should survive) under a lookback window.
        src.executemany(
            "INSERT INTO attachment(ROWID, filename, mime_type, transfer_name) "
            "VALUES (?, ?, ?, ?)",
            [(1, "a1", "x", "a1"), (2, "a2", "x", "a2")],
        )
        src.executemany(
            "INSERT INTO message_attachment_join(attachment_id, message_id) "
            "VALUES (?, ?)", [(1, 1), (2, 2)],
        )
        src.commit()
        return src

    def _fresh_tgt(self):
        tgt = sqlite3.connect(":memory:")
        self.addCleanup(tgt.close)
        ingest.create_schema(tgt)
        return tgt

    # Fixed "now" so the window boundary is deterministic regardless of run date.
    _NOW = datetime.datetime(2026, 7, 22)

    def test_lookback_drops_old_rows_reactions_and_attachments(self):
        src, tgt = self._fake_src(), self._fresh_tgt()
        cutoff = ingest._lookback_cutoff_ns((3, "mo"), now=self._NOW)
        ingest.ingest_messages_and_reactions(src, tgt, 0, None, cutoff)
        ingest.ingest_attachments(src, tgt, 0, None, cutoff)
        tgt.commit()
        self.assertEqual({r[0] for r in tgt.execute("SELECT id FROM messages")},
                         {2, 3})   # old msg 1 and dateless msg 6 excluded
        self.assertEqual({r[0] for r in tgt.execute("SELECT id FROM reactions")},
                         {4})      # old reaction 5 excluded
        self.assertEqual({r[0] for r in tgt.execute("SELECT id FROM attachments")},
                         {2})      # attachment on old msg 1 excluded

    def test_lookback_composes_with_chat_allowlist(self):
        src, tgt = self._fake_src(), self._fresh_tgt()
        cutoff = ingest._lookback_cutoff_ns((3, "mo"), now=self._NOW)
        # Allow only chat 10: recent-but-chat-20 msg 3 must fall away too, which
        # also proves the SQL placeholder order (rowid, *chat_ids, date) is right.
        ingest.ingest_messages_and_reactions(src, tgt, 0, {10}, cutoff)
        tgt.commit()
        self.assertEqual({r[0] for r in tgt.execute("SELECT id FROM messages")}, {2})

    def test_no_lookback_ingests_everything_including_dateless(self):
        src, tgt = self._fake_src(), self._fresh_tgt()
        ingest.ingest_messages_and_reactions(src, tgt, 0, None, None)
        tgt.commit()
        # Unfiltered behaviour is unchanged: every non-reaction row lands,
        # dateless system rows included.
        self.assertEqual({r[0] for r in tgt.execute("SELECT id FROM messages")},
                         {1, 2, 3, 6})


if __name__ == "__main__":
    unittest.main()
