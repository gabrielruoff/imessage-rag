#!/usr/bin/env python3
"""Clears all data from imessage_analysis.db while preserving its schema
(tables, FTS5 triggers). The sqlite-vec chunk_vectors table is the one
exception: it's dropped, not emptied, because its vector dimension is bound
to the embedding model that built it -- ingest.py recreates it. Does NOT
touch chat.db / chat_snapshot.db. Run ingest.py afterward to repopulate.
"""

import argparse
import sqlite3
import sys

from config import TARGET_DB as DB_PATH
from db import load_vec_extension

TABLES = [
    # messages_fts is external-content FTS5 over messages; it's kept in
    # sync by triggers (messages_ai/ad/au) that fire when messages rows
    # change, so it must not be deleted separately/first here.
    "chunk_topics",
    "topics",
    "chunk_annotations",
    "dyad_terms",
    "person_period_lexicon",
    "sessions",
    "chunks",
    "attachments",
    "reactions",
    "messages",
    "chat_participants",
    "handles",
    "contacts",
    "chats",
    "ingest_state",
]

# Dropped outright rather than emptied: their declared vector dimension is
# bound to the embedding model that built them, and ingest.py recreates them
# (build_vec_table) sized to whatever EMBED_MODEL is current. message_vectors
# is the legacy per-message table, superseded by chunk_vectors.
DROP_TABLES = ["chunk_vectors", "message_vectors"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH} — nothing to reset.")
        return

    if not args.force:
        confirm = input(
            f"This will delete ALL data in {DB_PATH.name} (schema is kept). Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    conn = sqlite3.connect(DB_PATH)
    try:
        load_vec_extension(conn)

        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

        conn.execute("PRAGMA foreign_keys = OFF")
        for table in DROP_TABLES:
            if table in existing:
                conn.execute(f"DROP TABLE {table}")
        for table in TABLES:
            if table in existing:
                conn.execute(f"DELETE FROM {table}")

        if "messages_fts" in existing:
            # Per-row delete triggers only tombstone entries in FTS5's
            # shadow tables (messages_fts_data/_idx) -- they don't reclaim
            # the underlying b-tree segments. 'rebuild' resyncs the index
            # from the now-empty messages table, actually emptying them.
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"Done. {DB_PATH.name} has been reset to an empty state.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted, exiting.")
        sys.exit(130)
