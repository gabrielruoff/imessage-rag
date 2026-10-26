"""Packaging smoke test: proves the sqlite-vec loadable extension works from
inside a frozen (PyInstaller) app -- the single riskiest unknown for shipping
imessage-rag as a bundled .app. Mirrors what db.load_vec_extension does, then
actually builds a vec0 table and runs a KNN query.
"""

import struct
import sqlite3
import sys

import sqlite_vec


def serialize(vec):
    return struct.pack("%df" % len(vec), *vec)


def main():
    print("frozen:", getattr(sys, "frozen", False))
    print("sqlite3.sqlite_version:", sqlite3.sqlite_version)
    print("sqlite_vec at:", sqlite_vec.loadable_path())

    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    ver = con.execute("select vec_version()").fetchone()[0]
    print("vec_version:", ver)

    # Exercise a real vec0 table (what chunk_vectors is) + KNN scan.
    con.execute("create virtual table v using vec0(id integer primary key, emb float[4])")
    rows = [(1, [0.1, 0.2, 0.3, 0.4]), (2, [0.9, 0.8, 0.7, 0.6]), (3, [0.0, 0.0, 0.1, 0.0])]
    for rid, emb in rows:
        con.execute("insert into v(id, emb) values (?, ?)", (rid, serialize(emb)))
    q = serialize([0.1, 0.2, 0.3, 0.4])
    hits = con.execute(
        "select id, distance from v where emb match ? order by distance limit 2", (q,)
    ).fetchall()
    print("knn hits:", hits)

    assert hits[0][0] == 1, "nearest neighbour should be row 1"
    print("\nOK: sqlite-vec loads and queries from a frozen bundle.")


if __name__ == "__main__":
    main()
