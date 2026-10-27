"""Shared Ollama embedding client, used by ingest.py (chunk/document
embeddings) and query_tools.py (query embeddings). Every request goes through
the batch /api/embed endpoint on a persistent keep-alive connection per
thread, so ingest's ThreadPoolExecutor workers and one-off query embeds share
the same code path.
"""

import http.client
import json
import math
import threading

from config import (
    EMBED_DIM,
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    OLLAMA_EMBED_PATH,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OLLAMA_TIMEOUT,
    is_apple_model,
    is_voyage_model,
)

_thread_local = threading.local()


def _connection() -> http.client.HTTPConnection:
    """One persistent keep-alive connection per thread."""
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_TIMEOUT)
        _thread_local.conn = conn
    return conn


def _fit_dim(vec):
    """Apply the EMBED_DIM override, if set: truncate to the first EMBED_DIM
    components and L2-renormalize (Matryoshka-style reduction -- only
    meaningful for MRL-trained models like nomic-embed-text v1.5). With
    EMBED_DIM unset (0), vectors pass through at the model's native size."""
    if not EMBED_DIM or len(vec) == EMBED_DIM:
        return vec
    if len(vec) < EMBED_DIM:
        raise RuntimeError(
            f"EMBED_DIM={EMBED_DIM} exceeds {EMBED_MODEL}'s native embedding "
            f"size ({len(vec)}) -- unset EMBED_DIM or lower it"
        )
    vec = vec[:EMBED_DIM]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_texts(texts: list, kind: str = "document") -> list:
    """Embed a batch of texts in a single Ollama request; returns one vector
    per input. `kind` selects the task prefix prepended to each text:
    'document' -> EMBED_DOC_PREFIX, 'query' -> EMBED_QUERY_PREFIX. Defaults
    are nomic-embed-text's trained prefixes ('search_document: ' /
    'search_query: '); set both to '' in .env for models without task
    prefixes."""
    if is_apple_model(EMBED_MODEL):
        # Apple's NLContextualEmbedding takes raw text -- nomic's task prefixes
        # are model-specific, so we skip them entirely (see applefm.embed_texts_apple)
        # and hand `kind` straight through. Import lazily so the Ollama-only path
        # never touches the shim client. _fit_dim/EMBED_DIM still apply, identically.
        import applefm
        return [_fit_dim(v) for v in applefm.embed_texts_apple(texts, kind)]
    if is_voyage_model(EMBED_MODEL):
        # Voyage carries the task in input_type, so nomic's prefixes are
        # skipped. Documents go through ingest's Batch API path only.
        # Lazy import, like apple: the Ollama-only path never loads the client.
        if kind != "query":
            raise RuntimeError("voyage document embeddings are batch-only -- run ingest")
        import voyage
        return voyage.embed_sync(texts, "query")
    prefix = EMBED_QUERY_PREFIX if kind == "query" else EMBED_DOC_PREFIX
    if prefix:
        texts = [prefix + t for t in texts]
    conn = _connection()
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    try:
        conn.request(
            "POST", OLLAMA_EMBED_PATH, body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read())
    except (http.client.HTTPException, OSError):
        # connection may be dead; drop it so the next call on this thread reopens one
        _thread_local.conn = None
        raise
    if "embeddings" not in body:
        raise RuntimeError(f"Ollama embed request failed: {body.get('error', body)}")
    return [_fit_dim(v) for v in body["embeddings"]]


def detect_embed_dim() -> int:
    """The effective embedding dimension: EMBED_MODEL's native output size,
    reduced by the EMBED_DIM override if one is set. Determined empirically
    from a probe request -- generation models served by Ollama (like gemma4)
    embed at their hidden size, which isn't documented anywhere queryable. The
    apple backend auto-detects the same way (embed_texts routes the probe
    through embed_texts_apple): NLContextualEmbedding is expected to be 512-dim,
    but the probe measures the true dim rather than assuming it."""
    if is_voyage_model(EMBED_MODEL):
        return EMBED_DIM  # Voyage's output_dimension, validated in config: no paid probe
    return len(embed_texts(["dimension probe"], kind="document")[0])
