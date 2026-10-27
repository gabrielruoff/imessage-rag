"""Voyage AI embedding client (EMBED_MODEL="voyage/<id>"): a sync query call for
chat turns and an async Batch API client for ingest's document embeddings.
Retry timing, classification, the concurrency gate and the stop come from
hosted.py; every tunable from config.py.

Contracts:
- embed_sync is the chat-time query path: bounded retries, then RuntimeError,
  which propagates to the semantic tool and comes back as a tool error. It
  prints nothing -- under the MCP server stdout is the wire.
- BatchClient retries transient errors without limit and raises HostedStop on
  anything else (NotFound for a 404). Input JSONL is built in memory and
  uploaded from bytes; it is never written to disk (it holds message text).
  Output is streamed by line.
- No error message carries a transport exception's text: its repr can quote
  request headers, i.e. the API key. Only the exception's type name is used.
"""

import asyncio
import hashlib
import json
import time

import httpx

import config
import hosted

API = "https://api.voyageai.com/v1"
# A batch in one of these states will not change again.
TERMINAL = {"completed", "failed", "cancelled", "expired"}
# The query path's attempt bound (a chat turn must not hang on a dead API).
SYNC_ATTEMPTS = 3
# GET /v1/batches page size (Voyage's maximum).
_LIST_PAGE = 100


class NotFound(hosted.HostedStop):
    """A 404: the batch or file is gone on Voyage's side."""


def _headers():
    return {"Authorization": f"Bearer {config.VOYAGE_API_KEY.strip()}"}


def model_id():
    return config.hosted_model_id(config.EMBED_MODEL)


def embed_sync(texts, input_type, transport=None):
    """POST /v1/embeddings; one vector per text, in input order. A retry never
    waits longer than HOSTED_BACKOFF_MAX_SECONDS: when Retry-After asks for
    more, it gives up at once (RuntimeError) rather than hang a chat turn."""
    body = {"input": list(texts), "model": model_id(), "input_type": input_type,
            "output_dimension": config.EMBED_DIM, "output_dtype": "float"}
    with httpx.Client(timeout=config.HOSTED_TIMEOUT, transport=transport) as client:
        for attempt in range(SYNC_ATTEMPTS):
            response = None
            try:
                response = client.post(f"{API}/embeddings", json=body, headers=_headers())
                verdict = hosted.classify(response)
            except hosted.REQUEST_ERRORS as e:
                verdict, cause = hosted.classify(e), type(e).__name__
            else:
                cause = f"HTTP {response.status_code}"
            if verdict == "ok":
                data = sorted(response.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if verdict != "retry":
                break
            if attempt + 1 < SYNC_ATTEMPTS:
                delay = hosted.retry_delay(attempt, response)
                if delay > config.HOSTED_BACKOFF_MAX_SECONDS:
                    cause += f", Retry-After {delay:.0f}s"
                    break
                time.sleep(delay)
    raise RuntimeError(f"Voyage query embedding failed ({cause})")


# ---------------------------------------------------------------------------
# Batch packing
# ---------------------------------------------------------------------------

def text_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def make_custom_id(chunk_id, text):
    """"<chunk_id>:<sha1(text)[:12]>" -- the hash lets collect discard a vector
    for text that changed (a tail-chunk rebuild) between submit and collect."""
    return f"{chunk_id}:{text_hash(text)}"


def parse_custom_id(custom_id):
    chunk_id, digest = custom_id.split(":", 1)
    return int(chunk_id), digest


def worst_case_usd(texts):
    """Budget reservation: UTF-8 bytes bound tokens from above."""
    return sum(len(t.encode()) for t in texts) * config.VOYAGE_BATCH_PRICE_PER_M / 1e6


def pack_lines(rows):
    """(chunk_id, text) rows -> [(batch_rows, jsonl_lines)], each batch <=
    VOYAGE_BATCH_MAX_INPUTS inputs; a line is newline-ended UTF-8 bytes.

    One input per line: its custom_id names the chunk on its own, so a batch
    persisted by id can be collected by a later process with no in-memory id
    list, and output order never matters. Per-line overhead is unbilled. A
    one-input line is always within Voyage's per-line input and token limits
    (an input is truncated at 32K tokens, the line limit is 120K)."""
    lines = [
        (json.dumps({"custom_id": make_custom_id(cid, text), "body": {"input": [text]}},
                    ensure_ascii=False) + "\n").encode()
        for cid, text in rows
    ]
    step = config.VOYAGE_BATCH_MAX_INPUTS
    return [(rows[i:i + step], lines[i:i + step]) for i in range(0, len(lines), step)]


# ---------------------------------------------------------------------------
# Async Batch API client
# ---------------------------------------------------------------------------

class BatchClient:
    """One httpx.AsyncClient per run; every request passes through `limiter`
    (a hosted.Limiter created in the running loop). Use as `async with`."""

    def __init__(self, limiter, transport=None):
        self.limiter = limiter
        self.http = httpx.AsyncClient(
            base_url=API, headers=_headers(), transport=transport,
            timeout=httpx.Timeout(config.HOSTED_TIMEOUT, read=config.VOYAGE_TIMEOUT,
                                  write=config.VOYAGE_TIMEOUT))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.http.aclose()

    async def _attempt(self, method, path, **kwargs):
        """One request -> (response, None) or (None, (verdict, exc type name))."""
        async with self.limiter:
            try:
                return await self.http.request(method, path, **kwargs), None
            except hosted.REQUEST_ERRORS as e:
                return None, (hosted.classify(e), type(e).__name__)

    def _verdict(self, response, cause):
        """Shared bookkeeping after one attempt; raises HostedStop on a stop."""
        verdict = hosted.classify(response) if response is not None else cause[0]
        if verdict == "ok":
            self.limiter.succeeded()
        elif verdict == "retry":
            if response is not None and response.status_code == 429:
                self.limiter.throttled()
        elif response is not None:
            stop = NotFound if response.status_code == 404 else hosted.HostedStop
            try:  # the provider's short message only: the raw body can quote input
                msg = hosted.error_message(response.json())
            except (ValueError, httpx.ResponseNotRead):
                msg = ""
            raise stop(f"Voyage: HTTP {response.status_code} {msg}".rstrip())
        else:
            raise hosted.HostedStop(f"Voyage: {cause[1]}")
        return verdict

    async def _request(self, method, path, **kwargs):
        attempt = 0
        while True:
            response, cause = await self._attempt(method, path, **kwargs)
            if self._verdict(response, cause) == "ok":
                return response
            await asyncio.sleep(hosted.retry_delay(attempt, response))
            attempt += 1

    async def upload(self, jsonl_bytes):
        """POST /v1/files (purpose=batch) from memory; returns the file id."""
        response = await self._request(
            "POST", "/files", data={"purpose": "batch"},
            files={"file": ("input.jsonl", jsonl_bytes, "application/jsonl")})
        return response.json()["id"]

    async def find_batch(self, input_file_id):
        """The batch created from `input_file_id`, or None (pages through
        GET /v1/batches)."""
        params = {"limit": _LIST_PAGE}
        while True:
            page = (await self._request("GET", "/batches", params=params)).json()
            for batch in page.get("data") or []:
                if batch.get("input_file_id") == input_file_id:
                    return batch
            if not page.get("has_more") or not page.get("last_id"):
                return None
            params = {"limit": _LIST_PAGE, "after": page["last_id"]}

    async def create_batch(self, input_file_id):
        """POST /v1/batches. A failed attempt may still have created the batch
        (the reply was lost), so every non-ok attempt reconciles through
        find_batch before re-POSTing: never two batches for one file."""
        body = {
            "endpoint": "/v1/embeddings",
            "completion_window": "12h",
            "input_file_id": input_file_id,
            "request_params": {"model": model_id(), "input_type": "document",
                               "output_dimension": config.EMBED_DIM,
                               "output_dtype": "float"},
        }
        attempt = 0
        while True:
            response, cause = await self._attempt("POST", "/batches", json=body)
            ok = response is not None and hosted.classify(response) == "ok"
            if not ok and (found := await self.find_batch(input_file_id)):
                return found
            if self._verdict(response, cause) == "ok":
                return response.json()
            await asyncio.sleep(hosted.retry_delay(attempt, response))
            attempt += 1

    async def get_batch(self, batch_id):
        return (await self._request("GET", f"/batches/{batch_id}")).json()

    async def stream_file(self, file_id):
        """Yield the file's non-empty lines. A transient failure mid-stream
        re-requests the file and skips the lines already yielded."""
        yielded, attempt = 0, 0
        while True:
            response, cause = None, None
            async with self.limiter:
                try:
                    # The live API 307s to a signed storage URL; httpx drops the
                    # Authorization header on that cross-origin hop.
                    async with self.http.stream("GET", f"/files/{file_id}/content",
                                                follow_redirects=True) as r:
                        if hosted.classify(r) == "ok":
                            seen = 0
                            async for line in r.aiter_lines():
                                if not line.strip():
                                    continue
                                seen += 1
                                if seen > yielded:
                                    yielded = seen
                                    yield line
                        else:
                            await r.aread()
                        response = r
                except hosted.REQUEST_ERRORS as e:
                    response, cause = None, (hosted.classify(e), type(e).__name__)
            if self._verdict(response, cause) == "ok":
                return
            await asyncio.sleep(hosted.retry_delay(attempt, response))
            attempt += 1

    async def delete_file(self, file_id):
        """DELETE /v1/files/{id}, best-effort: one attempt, warn on failure."""
        response, _ = await self._attempt("DELETE", f"/files/{file_id}")
        if response is None or hosted.classify(response) != "ok":
            print(f"warning: could not delete Voyage file {file_id}; delete it from "
                  f"the Voyage dashboard (it holds message text or derived vectors)")
