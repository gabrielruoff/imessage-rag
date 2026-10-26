"""Shared machinery for the hosted clients (voyage.py, openrouter.py): the spend
cap, retry timing and classification, the concurrency gate, the non-transient
stop, and the upload notice. Every tunable comes from config.py.

Completeness contract both clients follow: a transient failure ("retry") is
retried until it succeeds and is never dropped; a "stop" ends the run through
HostedStop, leaving the item pending for the next run.
"""

import asyncio
import email.utils
import random
import threading
import time

import httpx

import config


class Budget:
    """Hard USD cap. reserve() a worst case before each request (refused when
    spent + every in-flight reservation would exceed the limit), then settle()
    that reservation at the actual cost. Thread-safe."""

    def __init__(self, limit):
        self.limit, self.spent, self.reserved = limit, 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual

    def report(self):
        return f"spent: ${self.spent:.4f} of ${self.limit:.2f}"


def resolve_budget_usd(flag):
    """The run's spend cap: --budget, else HOSTED_BUDGET_USD. No default -- a
    hosted run with neither refuses to start."""
    usd = flag if flag is not None else config.HOSTED_BUDGET_USD
    if usd is None:
        raise SystemExit("hosted runs need a spend cap: pass --budget USD or set "
                         "HOSTED_BUDGET_USD")
    if usd <= 0:
        raise SystemExit(f"--budget / HOSTED_BUDGET_USD must be > 0, got {usd}")
    return usd


def check_max_parallel(n, flag="--max-parallel"):
    if n < 1:
        raise SystemExit(f"{flag} must be >= 1, got {n}")
    return n


def retry_delay(attempt, response=None):
    """Seconds to wait before retry `attempt` (0-based): the response's
    Retry-After (delta-seconds or HTTP-date) when present, else exponential
    backoff from 2s, capped at HOSTED_BACKOFF_MAX_SECONDS, with jitter."""
    raw = response.headers.get("retry-after") if response is not None else None
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                return max(0.0, email.utils.parsedate_to_datetime(raw).timestamp()
                           - time.time())
            except (TypeError, ValueError):
                pass  # unparseable header: fall back to backoff
    cap = config.HOSTED_BACKOFF_MAX_SECONDS
    return min(cap, 2.0 * 2 ** attempt) * random.uniform(0.5, 1.0)


def classify(response_or_exc):
    """One retry table for both clients: "ok" (2xx), "retry" (429, 5xx,
    connect/read errors, timeouts), "moderation" (an OpenRouter 403 whose body
    says moderation/flagged: reroute away from that provider), else "stop"
    (401/402/400 and every other 4xx)."""
    if isinstance(response_or_exc, httpx.HTTPStatusError):
        response_or_exc = response_or_exc.response
    if isinstance(response_or_exc, httpx.TransportError):  # incl. timeouts
        return "retry"
    if not isinstance(response_or_exc, httpx.Response):
        return "stop"
    status = response_or_exc.status_code
    if 200 <= status < 300:
        return "ok"
    if status == 429 or status >= 500:
        return "retry"
    if status == 403:
        try:
            body = response_or_exc.text.lower()
        except httpx.ResponseNotRead:
            body = ""
        if "moderation" in body or "flagged" in body:
            return "moderation"
    return "stop"


class HostedStop(RuntimeError):
    """A non-transient stop (auth, credits, invalid request, budget). The CLI
    turns it into a non-zero exit naming `cause` and the count still pending."""

    def __init__(self, cause):
        super().__init__(cause)
        self.cause = cause


class Limiter:
    """Async concurrency gate: `async with limiter:` around each request. The
    effective limit starts at max_parallel, halves (min 1) on throttled() and
    climbs back by 1 after HOSTED_LIMIT_RECOVER_AFTER consecutive succeeded()
    calls, never above max_parallel. max_parallel=1 is strictly sequential."""

    def __init__(self, max_parallel, recover_after=None):
        self.max_parallel = self.limit = check_max_parallel(max_parallel)
        self.recover_after = recover_after or config.HOSTED_LIMIT_RECOVER_AFTER
        self.in_flight = 0
        self._streak = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            await self._cond.wait_for(lambda: self.in_flight < self.limit)
            self.in_flight += 1
        return self

    async def __aexit__(self, *exc):
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify_all()

    def throttled(self):
        """A 429: halve the effective limit."""
        self.limit = max(1, self.limit // 2)
        self._streak = 0

    def succeeded(self):
        # A raised limit admits waiters on the next __aexit__'s notify; one is
        # always due, since a waiter implies in_flight >= the old limit >= 1.
        self._streak += 1
        if self._streak >= self.recover_after and self.limit < self.max_parallel:
            self.limit += 1
            self._streak = 0


def upload_notice(service, what):
    """D3: choosing the hosted model is the consent; say what goes where.
    CLI runs only -- never on the chat-time query path (MCP stdout is the wire)."""
    print(f"uploading {what} to {service}")
