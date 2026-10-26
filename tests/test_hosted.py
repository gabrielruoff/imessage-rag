"""Hermetic tests for hosted.py (the machinery shared by the hosted clients) and
the ollama.py seam that keeps openrouter/ models out of every non-annotate
pipeline. No network: responses are built in memory, sockets are stubbed."""

import asyncio
import email.utils
import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx

import config
import hosted
import ollama


def _resp(status, text="", headers=None):
    return httpx.Response(status, text=text, headers=headers or {})


class BudgetTest(unittest.TestCase):
    def test_reserve_refuses_what_would_exceed_spent_plus_reserved(self):
        b = hosted.Budget(1.0)
        self.assertTrue(b.reserve(0.6))
        self.assertFalse(b.reserve(0.5))  # 0.6 reserved + 0.5 > 1.0
        b.settle(0.6, 0.2)
        self.assertAlmostEqual(b.spent, 0.2)
        self.assertAlmostEqual(b.reserved, 0.0)
        self.assertTrue(b.reserve(0.8))
        self.assertFalse(b.reserve(0.01))
        self.assertIn("spent: $0.2000", b.report())

    def test_concurrent_reservations_never_overcommit(self):
        b = hosted.Budget(4.0)
        granted = []
        start = threading.Barrier(100)

        def worker():
            start.wait()
            granted.append(b.reserve(0.125))  # exact in binary

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(granted), 32)
        self.assertEqual(b.reserved, 4.0)


class ResolveTest(unittest.TestCase):
    def test_flag_wins_over_env(self):
        with mock.patch.object(config, "HOSTED_BUDGET_USD", 9.0):
            self.assertEqual(hosted.resolve_budget_usd(0.5), 0.5)
            self.assertEqual(hosted.resolve_budget_usd(None), 9.0)

    def test_no_budget_anywhere_refuses_naming_both(self):
        with mock.patch.object(config, "HOSTED_BUDGET_USD", None), \
                self.assertRaises(SystemExit) as ctx:
            hosted.resolve_budget_usd(None)
        self.assertIn("--budget", str(ctx.exception))
        self.assertIn("HOSTED_BUDGET_USD", str(ctx.exception))

    def test_non_positive_budget_is_refused(self):
        for bad in (0, -1.0):
            with self.subTest(value=bad), self.assertRaises(SystemExit):
                hosted.resolve_budget_usd(bad)

    def test_max_parallel_below_one_is_refused_naming_the_flag(self):
        self.assertEqual(hosted.check_max_parallel(3), 3)
        with self.assertRaises(SystemExit) as ctx:
            hosted.check_max_parallel(0, flag="--max-parallel")
        self.assertIn("--max-parallel", str(ctx.exception))


class RetryDelayTest(unittest.TestCase):
    def test_retry_after_in_seconds(self):
        self.assertEqual(hosted.retry_delay(0, _resp(429, headers={"Retry-After": "7"})), 7.0)

    def test_retry_after_as_http_date(self):
        when = email.utils.formatdate(time.time() + 30, usegmt=True)
        delay = hosted.retry_delay(0, _resp(429, headers={"Retry-After": when}))
        self.assertTrue(25 <= delay <= 31, delay)
        past = email.utils.formatdate(time.time() - 30, usegmt=True)
        self.assertEqual(hosted.retry_delay(0, _resp(429, headers={"Retry-After": past})), 0.0)

    def test_backoff_doubles_from_two_seconds_with_jitter_and_is_capped(self):
        with mock.patch.object(config, "HOSTED_BACKOFF_MAX_SECONDS", 60):
            for _ in range(50):
                self.assertTrue(1.0 <= hosted.retry_delay(0) <= 2.0)
                self.assertTrue(4.0 <= hosted.retry_delay(2) <= 8.0)
                self.assertTrue(30.0 <= hosted.retry_delay(30) <= 60.0)
            # an unparseable header falls back to backoff rather than raising
            self.assertLessEqual(
                hosted.retry_delay(9, _resp(503, headers={"Retry-After": "soon"})), 60.0)


class ClassifyTest(unittest.TestCase):
    def test_the_table(self):
        req = httpx.Request("POST", "https://example.invalid/x")
        cases = [
            (_resp(200), "ok"),
            (_resp(429), "retry"),
            (_resp(500), "retry"),
            (_resp(503), "retry"),
            (httpx.ConnectError("refused", request=req), "retry"),
            (httpx.ReadError("reset", request=req), "retry"),
            (httpx.ReadTimeout("slow", request=req), "retry"),
            (_resp(403, '{"error": {"message": "requires moderation on Placeholder. '
                        'Your input was flagged for x"}}'), "moderation"),
            (_resp(403, '{"error": {"message": "forbidden"}}'), "stop"),
            (_resp(400), "stop"),
            (_resp(401), "stop"),
            (_resp(402), "stop"),
            (_resp(404), "stop"),
            (ValueError("not an http failure"), "stop"),
        ]
        for value, want in cases:
            with self.subTest(value=value):
                self.assertEqual(hosted.classify(value), want)

    def test_status_error_is_classified_by_its_response(self):
        req = httpx.Request("GET", "https://example.invalid/x")
        err = httpx.HTTPStatusError("x", request=req, response=_resp(429))
        self.assertEqual(hosted.classify(err), "retry")


class LimiterTest(unittest.TestCase):
    def _peak(self, limiter, n_tasks):
        async def run():
            peak = 0

            async def task():
                nonlocal peak
                async with limiter:
                    peak = max(peak, limiter.in_flight)
                    await asyncio.sleep(0.001)

            await asyncio.gather(*(task() for _ in range(n_tasks)))
            return peak

        return asyncio.run(run())

    def test_never_more_than_max_parallel_in_flight(self):
        self.assertEqual(self._peak(hosted.Limiter(3), 20), 3)
        self.assertEqual(self._peak(hosted.Limiter(1), 10), 1)  # strictly sequential

    def test_halves_on_429_and_recovers_one_step_per_streak(self):
        lim = hosted.Limiter(8, recover_after=3)
        lim.throttled()
        self.assertEqual(lim.limit, 4)
        lim.throttled()
        lim.throttled()
        lim.throttled()
        self.assertEqual(lim.limit, 1)
        self.assertEqual(self._peak(lim, 6), 1)
        for _ in range(3):
            lim.succeeded()
        self.assertEqual(lim.limit, 2)
        for _ in range(100):
            lim.succeeded()
        self.assertEqual(lim.limit, 8)  # never above max_parallel

    def test_a_429_resets_the_success_streak(self):
        lim = hosted.Limiter(4, recover_after=3)
        lim.throttled()
        lim.succeeded()
        lim.succeeded()
        lim.throttled()
        lim.succeeded()
        self.assertEqual(lim.limit, 1)

    def test_rejects_max_parallel_below_one(self):
        with self.assertRaises(SystemExit):
            hosted.Limiter(0)


class UploadNoticeTest(unittest.TestCase):
    def test_says_what_goes_where(self):
        out = io.StringIO()
        with redirect_stdout(out):
            hosted.upload_notice("Voyage AI (placeholder-model)", "chunk text")
        self.assertEqual(out.getvalue(), "uploading chunk text to Voyage AI (placeholder-model)\n")


class OllamaSeamTest(unittest.TestCase):
    """D2: every non-annotate pipeline reaches a model through these three
    functions, so an openrouter/ name must be refused before any connection."""

    def setUp(self):
        conn = mock.patch.object(ollama.http.client, "HTTPConnection",
                                 side_effect=AssertionError("no connection expected"))
        conn.start()
        self.addCleanup(conn.stop)

    def _calls(self, model):
        msgs = [{"role": "user", "content": "hi"}]
        return {
            "chat_once": lambda: ollama.chat_once(msgs, model=model),
            "chat_once_json": lambda: ollama.chat_once_json(msgs, model=model),
            "chat_stream": lambda: next(ollama.chat_stream(msgs, model=model)),
        }

    def test_an_openrouter_model_argument_is_refused(self):
        for name, call in self._calls("openrouter/placeholder/model").items():
            with self.subTest(fn=name), self.assertRaises(RuntimeError) as ctx:
                call()
            self.assertIn("annotate-only", str(ctx.exception))

    def test_an_openrouter_chat_model_default_is_refused(self):
        with mock.patch.object(ollama, "CHAT_MODEL", "openrouter/placeholder/model"):
            for name, call in self._calls(None).items():
                with self.subTest(fn=name), self.assertRaises(RuntimeError):
                    call()


if __name__ == "__main__":
    unittest.main()
