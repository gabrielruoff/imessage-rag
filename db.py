"""Shared sqlite-vec extension loading, used wherever a raw sqlite3
connection needs vec0 table support."""

import sqlite_vec


def load_vec_extension(con):
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
