#!/usr/bin/env python3
"""Clears all data from imessage_analysis.db while preserving its schema
(tables, FTS5, sqlite-vec virtual table). Does NOT touch chat.db /
chat_snapshot.db. Run ingest.py afterward to repopulate.
"""

import argparse
import sqlite3

import sqlite_vec

from config import TARGET_DB as DB_PATH

TABLES = [
    # messages_fts is external-content FTS5 over messages; it's kept in
    # sync by triggers (messages_ai/ad/au) that fire when messages rows
    # change, so it must not be deleted separately/first here.
    "message_vectors",
    "attachments",
    "reactions",
    "messages",
    "chat_participants",
    "handles",
    "contacts",
    "chats",
    "ingest_state",
]


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
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

        conn.execute("PRAGMA foreign_keys = OFF")
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
    main()
