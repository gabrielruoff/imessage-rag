"""Async OpenRouter client for hosted annotation (annotate.py's per-chunk tiers
only; every other pipeline is refused at the ollama.py seam). One chunk per
request in the cached shape: the system block carries cache_control and the
strict schema sits inside that cached prefix, so the system text and schema
must be byte-identical across a run's calls -- the caller builds them once and
passes them in.

Retries, the spend cap and the concurrency gate are hosted.py's: a transient
failure is retried until it succeeds, a moderation 403 is rerouted away from
the flagging provider, anything else is a HostedStop.
"""

import asyncio
import json

import httpx

import config
import hosted

BASE_URL = "https://openrouter.ai/api/v1"
# Parse failures on one chunk retried before it is reported unparseable.
UNPARSEABLE_RETRIES = 3
# Successful calls after the first that may all read 0 cached tokens before the
# one-time "prompt cache is not hitting" warning.
CACHE_CHECK_CALLS = 3


def client():
    """One AsyncClient per run (connection reuse). Never logs the key."""
    return httpx.AsyncClient(
        base_url=BASE_URL, timeout=httpx.Timeout(config.HOSTED_TIMEOUT),
        headers={"Authorization": f"Bearer {config.OPEN_ROUTER_API_KEY.strip()}"})


def reasoning_for(think):
    """The repo's tri-state think -> OpenRouter `reasoning`: False/True toggle
    it, an effort string is an effort level, None omits the key (returns None)."""
    if think is None:
        return None
    if isinstance(think, bool):
        return {"enabled": think}
    return {"effort": think}


def request_body(model_id, system, schema, text, think, session_id, ignore=()):
    """The recommended one-chunk request. Never sets provider.order: sticky
    routing on session_id keeps the warm provider's cache."""
    body = {
        "model": model_id,
        "max_tokens": config.OPENROUTER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system,
                                            "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "annotation", "strict": True, "schema": schema}},
    }
    reasoning = reasoning_for(think)
    if reasoning is not None:
        body["reasoning"] = reasoning
    provider = {"require_parameters": True}  # never silently drop the schema
    if ignore:
        provider["ignore"] = list(ignore)
    body["provider"] = provider
    body["usage"] = {"include": True}
    body["session_id"] = session_id
    return body


def _json(resp):
    try:
        return resp.json()
    except (ValueError, httpx.ResponseNotRead):
        return None


def _error(resp, data):
    """(status, message) for a failed request: the HTTP status and the
    provider's short error.message, never the raw body or metadata (either can
    quote request text). A transport exception gives only its type name: its
    text can quote request headers, i.e. the API key."""
    if not isinstance(resp, httpx.Response):
        return "transport error", type(resp).__name__
    return resp.status_code, hosted.error_message(data)


async def fetch_prices(http, model_ids):
    """{model_id: (usd per input token, usd per output token)} from the live
    model list. A model not listed, or priced <= 0 (OpenRouter's "-1" marks a
    variable-priced route; a non-positive price would let every budget
    reservation through), is a HostedStop."""
    attempt = 0
    while True:
        try:
            resp = await http.get("/models")
        except hosted.REQUEST_ERRORS as exc:
            resp = exc
        kind = hosted.classify(resp)
        if kind == "ok":
            break
        if kind != "retry":
            status, msg = _error(resp, _json(resp) if isinstance(resp, httpx.Response) else None)
            raise hosted.HostedStop(f"OpenRouter /models {status}: {msg}")
        await asyncio.sleep(hosted.retry_delay(
            attempt, resp if isinstance(resp, httpx.Response) else None))
        attempt += 1
    listing = {m["id"]: m.get("pricing") or {} for m in resp.json()["data"]}
    missing = [m for m in model_ids if m not in listing]
    if missing:
        raise hosted.HostedStop(f"not listed on OpenRouter: {', '.join(missing)}")
    prices = {m: (float(listing[m]["prompt"]), float(listing[m]["completion"]))
              for m in model_ids}
    unpriced = [m for m, pair in prices.items() if min(pair) <= 0]
    if unpriced:
        raise hosted.HostedStop(f"no positive per-token price on OpenRouter: "
                                f"{', '.join(unpriced)}")
    return prices


class Annotator:
    """One run's annotation requests. `parse(content)` returns the annotation
    or raises (ValueError/TypeError/AttributeError) on an unusable reply.
    `think` is the run's; the mandatory-reasoning fallback changes it once."""

    def __init__(self, http, model_id, *, system, schema, think, session_id,
                 prices, budget, limiter, parse):
        self.http, self.model_id = http, model_id
        self.system, self.schema = system, schema
        self.think, self.session_id = think, session_id
        self.in_price, self.out_price = prices
        self.budget, self.limiter, self.parse = budget, limiter, parse
        self._ok_calls = 0
        self._cache_hit = self._echo_printed = False

    async def _post(self, body):
        """One request under the budget and the limiter -> (response or
        transport exception, parsed JSON or None). Reserves the worst case
        (UTF-8 bytes of the body, schema included, x input price + max_tokens
        x output price) and settles at max(usage.cost, tokens x price); a 4xx
        without usage at 0, anything else without usage at the worst case."""
        worst = (len(json.dumps(body).encode()) * self.in_price
                 + config.OPENROUTER_MAX_TOKENS * self.out_price)
        if not self.budget.reserve(worst):
            raise hosted.HostedStop(f"budget cap reached ({self.budget.report()})")
        actual = worst
        try:
            async with self.limiter:
                resp = await self.http.post("/chat/completions", json=body)
            data = _json(resp)
            usage = data.get("usage") if isinstance(data, dict) else None
            if usage:
                actual = max(float(usage.get("cost") or 0.0),
                             (usage.get("prompt_tokens") or 0) * self.in_price
                             + (usage.get("completion_tokens") or 0) * self.out_price)
            elif 400 <= resp.status_code < 500:
                actual = 0.0
            return resp, data
        except hosted.REQUEST_ERRORS as exc:
            return exc, None
        finally:
            self.budget.settle(worst, actual)

    def _observe(self, data):
        """Log the served (possibly dated) model id once, and warn once when
        the cached prefix never reads back."""
        if not isinstance(data, dict):
            return
        if not self._echo_printed and data.get("model"):
            self._echo_printed = True
            print(f"OpenRouter served {data['model']}")
        self._ok_calls += 1
        details = (data.get("usage") or {}).get("prompt_tokens_details") or {}
        if details.get("cached_tokens"):
            self._cache_hit = True
        elif self._ok_calls == 1 + CACHE_CHECK_CALLS and not self._cache_hit:
            print(f"warning: prompt cache not hit in {self._ok_calls} calls "
                  f"(cached_tokens stayed 0); each request pays the full prefix")

    async def annotate_one(self, text):
        """-> ("ok", annotation) | ("refused", None) | ("unparseable", None).
        Transient failures retry without limit; a moderation 403 retries with
        the flagging provider in provider.ignore ("refused" once none is
        left); anything else raises HostedStop."""
        ignore, attempt, bad = [], 0, 0
        while True:
            body = request_body(self.model_id, self.system, self.schema, text,
                                self.think, self.session_id, ignore)
            resp, data = await self._post(body)
            kind = hosted.classify(resp)
            if kind == "ok":
                self.limiter.succeeded()
                self._observe(data)
                try:
                    return "ok", self.parse(data["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    bad += 1
                    if bad > UNPARSEABLE_RETRIES:
                        return "unparseable", None
                    continue
            if kind == "retry":
                response = resp if isinstance(resp, httpx.Response) else None
                if response is not None and response.status_code == 429:
                    self.limiter.throttled()
                await asyncio.sleep(hosted.retry_delay(attempt, response))
                attempt += 1
                continue
            if kind == "moderation":
                err = data.get("error") if isinstance(data, dict) else None
                meta = (err.get("metadata") if isinstance(err, dict) else None) or {}
                provider = meta.get("provider_name")
                if not provider or provider in ignore:
                    return "refused", None
                ignore.append(provider)
                continue
            status, msg = _error(resp, data)
            lower = msg.lower()
            if "reasoning is mandatory" in lower and body.get("reasoning") != {"effort": "low"}:
                if self.think != "low":  # sticky for the run, printed once
                    self.think = "low"
                    print(f"note: {self.model_id} requires reasoning; using effort 'low'")
                continue
            if ignore and ("no endpoints" in lower or "no allowed providers" in lower):
                return "refused", None
            raise hosted.HostedStop(f"OpenRouter {status}: {msg}")
