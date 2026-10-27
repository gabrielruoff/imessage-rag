"""Hermetic tests for openrouter.py: request shape, the think -> reasoning
mapping, retries, moderation rerouting, budget settling and the price lookup.
httpx.MockTransport only -- no network, no real key."""

import asyncio
import contextlib
import io
import json
import unittest
from unittest import mock

import httpx

import config
import hosted
import openrouter

MODEL = "test/model"
SYSTEM = "placeholder system prompt"
SCHEMA = {"type": "object", "properties": {"valence": {"type": "number"}}}
IN_PRICE, OUT_PRICE = 1e-7, 5e-7
GOOD = {"valence": 0.5, "emotions": [], "depth": 2, "events": []}


def ok(content=GOOD, cached=0, usage=True):
    body = {"model": "test/model-20261007",
            "choices": [{"message": {"content": content if isinstance(content, str)
                                     else json.dumps(content)}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.00001,
                         "prompt_tokens_details": {"cached_tokens": cached}}
    return httpx.Response(200, json=body)


def err(status, message, **extra):
    error = {"code": status, "message": message, **extra}
    return httpx.Response(status, json={"error": error})


class Harness:
    """Runs Annotator.annotate_one calls against scripted responses and
    records every request body."""

    def __init__(self, responses, think=False, budget=10.0):
        self.responses = list(responses)
        self.bodies = []
        self.think, self.budget = think, hosted.Budget(budget)

    def handler(self, request):
        self.bodies.append(json.loads(request.content))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def run(self, texts, max_parallel=1):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.handler),
                                         base_url=openrouter.BASE_URL) as http:
                self.ann = openrouter.Annotator(
                    http, MODEL, system=SYSTEM, schema=SCHEMA, think=self.think,
                    session_id="sess-1", prices=(IN_PRICE, OUT_PRICE), budget=self.budget,
                    limiter=hosted.Limiter(max_parallel), parse=json.loads)
                return [await self.ann.annotate_one(t) for t in texts]
        return asyncio.run(go())


class OpenRouterTestCase(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(hosted, "retry_delay", return_value=0)
        self.retry_delay = p.start()
        self.addCleanup(p.stop)
        self.out = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.out))


class TestRequestBody(OpenRouterTestCase):
    def test_recommended_shape_and_cached_prefix_byte_identical(self):
        h = Harness([ok(), ok()])
        h.run(["chunk one", "chunk two"])
        a, b = h.bodies
        self.assertEqual(a["model"], MODEL)
        self.assertEqual(a["max_tokens"], config.OPENROUTER_MAX_TOKENS)
        self.assertEqual(a["messages"][0], {"role": "system", "content": [
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]})
        self.assertEqual(a["messages"][1], {"role": "user", "content": "chunk one"})
        self.assertEqual(a["response_format"], {"type": "json_schema", "json_schema": {
            "name": "annotation", "strict": True, "schema": SCHEMA}})
        self.assertEqual(a["reasoning"], {"enabled": False})
        self.assertEqual(a["provider"], {"require_parameters": True})
        self.assertEqual(a["usage"], {"include": True})
        self.assertEqual(a["session_id"], b["session_id"])
        for key in ("messages", "response_format"):
            first = a[key][0] if key == "messages" else a[key]
            second = b[key][0] if key == "messages" else b[key]
            self.assertEqual(json.dumps(first), json.dumps(second))

    def test_never_sets_provider_order(self):
        body = openrouter.request_body(MODEL, SYSTEM, SCHEMA, "t", False, "s", ignore=["X"])
        self.assertEqual(body["provider"], {"require_parameters": True, "ignore": ["X"]})

    def test_reasoning_tri_state(self):
        self.assertEqual(openrouter.reasoning_for(False), {"enabled": False})
        self.assertEqual(openrouter.reasoning_for(True), {"enabled": True})
        self.assertEqual(openrouter.reasoning_for("low"), {"effort": "low"})
        body = openrouter.request_body(MODEL, SYSTEM, SCHEMA, "t", None, "s")
        self.assertNotIn("reasoning", body)


class TestAnnotateOne(OpenRouterTestCase):
    def test_mandatory_reasoning_falls_back_to_low_once(self):
        h = Harness([err(400, "Reasoning is mandatory for this endpoint"), ok(), ok()])
        results = h.run(["a", "b"])
        self.assertEqual([r[0] for r in results], ["ok", "ok"])
        self.assertEqual([b["reasoning"] for b in h.bodies],
                         [{"enabled": False}, {"effort": "low"}, {"effort": "low"}])
        self.assertEqual(self.out.getvalue().count("requires reasoning"), 1)

    def test_429_retry_after_then_success_never_dropped(self):
        h = Harness([httpx.Response(429, headers={"Retry-After": "7"}),
                     httpx.Response(503), httpx.ConnectError("down"), ok()])
        self.assertEqual(h.run(["a"]), [("ok", GOOD)])
        self.assertEqual(len(h.bodies), 4)
        self.assertEqual(self.retry_delay.call_args_list[0].args[1].headers["retry-after"], "7")

    def test_moderation_reroutes_then_refused_after_all_providers(self):
        mod = "requires moderation on {}. Your input was flagged"
        h = Harness([
            err(403, mod.format("ProvA"), metadata={"provider_name": "ProvA"}),
            err(403, mod.format("ProvB"), metadata={"provider_name": "ProvB"}),
            err(404, "No endpoints found that can handle the requested parameters"),
        ])
        self.assertEqual(h.run(["a"]), [("refused", None)])
        self.assertNotIn("ignore", h.bodies[0]["provider"])
        self.assertEqual(h.bodies[1]["provider"]["ignore"], ["ProvA"])
        self.assertEqual(h.bodies[2]["provider"]["ignore"], ["ProvA", "ProvB"])
        self.assertEqual({b["model"] for b in h.bodies}, {MODEL})

    def test_401_is_a_stop(self):
        h = Harness([err(401, "No auth credentials found")])
        with self.assertRaises(hosted.HostedStop) as cm:
            h.run(["a"])
        self.assertIn("401", cm.exception.cause)

    def test_unparseable_retried_three_times_then_reported(self):
        h = Harness([ok("not json")] * 4)
        self.assertEqual(h.run(["a"]), [("unparseable", None)])
        self.assertEqual(len(h.bodies), 1 + openrouter.UNPARSEABLE_RETRIES)

    def test_budget_settles_4xx_at_zero(self):
        h = Harness([err(400, "bad request")])
        with self.assertRaises(hosted.HostedStop):
            h.run(["a"])
        self.assertEqual(h.budget.spent, 0.0)
        self.assertEqual(h.budget.reserved, 0.0)

    def test_budget_settles_5xx_without_usage_at_worst_case(self):
        h = Harness([httpx.Response(502), ok()])
        h.run(["a"])
        worst = (len(json.dumps(h.bodies[0]).encode()) * IN_PRICE
                 + config.OPENROUTER_MAX_TOKENS * OUT_PRICE)
        actual = max(0.00001, 100 * IN_PRICE + 10 * OUT_PRICE)
        self.assertAlmostEqual(h.budget.spent, worst + actual)
        self.assertAlmostEqual(h.budget.reserved, 0.0)

    def test_budget_exhausted_is_a_stop_before_sending(self):
        h = Harness([ok()], budget=1e-9)
        with self.assertRaises(hosted.HostedStop) as cm:
            h.run(["a"])
        self.assertIn("budget", cm.exception.cause)
        self.assertEqual(h.bodies, [])

    def test_cache_warning_once_when_never_hit(self):
        h = Harness([ok()] * 6)
        h.run(["a"] * 6)
        self.assertEqual(self.out.getvalue().count("prompt cache not hit"), 1)
        h = Harness([ok(), ok(cached=900), ok(), ok(), ok()])
        self.out.truncate(0)
        h.run(["a"] * 5)
        self.assertNotIn("prompt cache not hit", self.out.getvalue())

    def test_s4_decoding_error_is_retried(self):
        h = Harness([httpx.DecodingError("placeholder bad encoding"), ok()])
        self.assertEqual(h.run(["a"]), [("ok", GOOD)])
        self.assertEqual(len(h.bodies), 2)

    def test_s4_key_never_reaches_the_stop(self):
        key = "sk-placeholder-secret \n"
        with mock.patch.object(config, "OPEN_ROUTER_API_KEY", key):
            async def headers():
                async with openrouter.client() as http:
                    return http.headers["Authorization"]
            self.assertEqual(asyncio.run(headers()), "Bearer sk-placeholder-secret")
        h = Harness([httpx.LocalProtocolError(f"Illegal header value {key!r}")])
        with self.assertRaises(hosted.HostedStop) as cm:
            h.run(["a"])
        self.assertNotIn("sk-placeholder", str(cm.exception))
        self.assertIn("LocalProtocolError", cm.exception.cause)
        self.assertNotIn("sk-placeholder", self.out.getvalue())

    def test_s4_stop_never_quotes_input_text(self):
        h = Harness([err(400, "Input rejected", metadata={
            "flagged_input": "placeholder private text"})])
        with self.assertRaises(hosted.HostedStop) as cm:
            h.run(["a"])
        self.assertIn("400: Input rejected", cm.exception.cause)
        self.assertNotIn("placeholder private text", cm.exception.cause)
        h = Harness([httpx.Response(400, text="echo: placeholder private text")])
        with self.assertRaises(hosted.HostedStop) as cm:
            h.run(["a"])
        self.assertIn("400", cm.exception.cause)
        self.assertNotIn("placeholder private text", cm.exception.cause)


class TestFetchPrices(OpenRouterTestCase):
    def fetch(self, handler, ids):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                         base_url=openrouter.BASE_URL) as http:
                return await openrouter.fetch_prices(http, ids)
        return asyncio.run(go())

    def test_prices_and_unlisted_model(self):
        listing = {"data": [{"id": MODEL, "pricing": {"prompt": "0.0000001",
                                                      "completion": "0.0000005"}}]}
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=listing)

        self.assertEqual(self.fetch(handler, [MODEL]), {MODEL: (1e-7, 5e-7)})
        self.assertEqual(calls, ["/api/v1/models"] * 2)
        with self.assertRaises(hosted.HostedStop):
            self.fetch(lambda r: httpx.Response(200, json=listing), ["test/other"])


    def test_s4_non_positive_price_is_a_stop(self):
        for price in ("-1", "0"):
            listing = {"data": [{"id": MODEL, "pricing": {"prompt": price,
                                                          "completion": "0.0000005"}}]}
            with self.subTest(price=price), \
                    self.assertRaisesRegex(hosted.HostedStop, MODEL):
                self.fetch(lambda r: httpx.Response(200, json=listing), [MODEL])

if __name__ == "__main__":
    unittest.main()
