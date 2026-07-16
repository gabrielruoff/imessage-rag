"""Embedding step of the pipeline: build the sqlite-vec table and embed
message text via local Ollama (see ollama.embed_batch). Called by ingest.main
after the relational ingest completes; safe to re-run -- only messages without
a vector are embedded.
"""

import http.client
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import sqlite_vec

from config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MAX_WORKERS, EMBED_MODEL
from db import load_vec_extension
from ollama import embed_batch


def build_vec_table(tgt: sqlite3.Connection):
    load_vec_extension(tgt)
    tgt.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS message_vectors USING vec0(
            message_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBED_DIM}]
        )
    """)


def embed_messages(
    tgt: sqlite3.Connection,
    min_chars=3,
    batch_size=EMBED_BATCH_SIZE,
    max_workers=EMBED_MAX_WORKERS,
    batch_log_every=200,
):
    rows = tgt.execute("""
        SELECT m.id, m.text FROM messages m
        LEFT JOIN message_vectors v ON v.message_id = m.id
        WHERE v.message_id IS NULL
          AND m.text IS NOT NULL
          AND length(m.text) >= ?
    """, (min_chars,)).fetchall()

    print(f"Embedding {len(rows)} messages via Ollama ({EMBED_MODEL})...")
    chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def process(chunk):
        texts = [text for _, text in chunk]
        vecs = embed_batch(texts)
        return chunk, vecs

    n_done, n_err, n_processed, n_logged = 0, 0, 0, 0
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_chunk = {pool.submit(process, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk, vecs = future.result()
            except (http.client.HTTPException, OSError, KeyError, json.JSONDecodeError):
                n_err += len(chunk)
                continue
            tgt.executemany(
                "INSERT OR REPLACE INTO message_vectors(message_id, embedding) VALUES (?, ?)",
                [(msg_id, sqlite_vec.serialize_float32(vec))
                 for (msg_id, _), vec in zip(chunk, vecs)],
            )
            n_done += len(chunk)
            n_processed += len(chunk)
            if n_processed - n_logged >= batch_log_every:
                tgt.commit()
                print(f"  ...{n_processed}/{len(rows)}")
                n_logged = n_processed
    except KeyboardInterrupt:
        # Don't wait on in-flight worker threads (each blocked on an Ollama
        # HTTP call) -- drop anything not yet started and commit whatever
        # finished so it isn't lost.
        print("\nInterrupted -- cancelling remaining embedding batches...")
        pool.shutdown(wait=False, cancel_futures=True)
        tgt.commit()
        print(f"Embedded {n_done} messages before interrupt ({n_err} errors)")
        raise
    pool.shutdown(wait=True)
    tgt.commit()
    print(f"Embedded {n_done} messages ({n_err} errors)")
