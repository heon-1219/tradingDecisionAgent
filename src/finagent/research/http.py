"""Shared async HTTP client: per-host rate limits, GET retries, timeouts and secret redaction.

Every research/broker connector (EDGAR, Tavily, Jina Reader, Alpaca) goes through this client so
the project's global constraints hold in one place instead of being re-implemented per connector:
secrets never reach logs, and slow or rate-limited upstream hosts get a per-host minimum-interval
throttle.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default req/s per host; hosts not listed here are unlimited. A host also matches if the request
# host is a subdomain of a listed one (e.g. "data.sec.gov" matches "sec.gov").
DEFAULT_RATES: dict[str, float] = {
    "sec.gov": 5.0,
    "api.tavily.com": 1.5,
    "r.jina.ai": 0.3,
    "paper-api.alpaca.markets": 3.0,
}

_MAX_GET_ATTEMPTS = 3  # 1 initial try + 2 retries
_RETRY_BACKOFF_SEC = (0.5, 1.5)

# Never log a header value whose key looks like a credential, and never log a query string that
# carries a key/token param (the whole query is dropped rather than trying to redact just the
# offending param, since connectors may name it differently).
_SECRET_HEADER_RE = re.compile(r"key|secret|token|authorization", re.I)
_SECRET_PARAM_RE = re.compile(r"key|token", re.I)


class HttpError(Exception):
    """Raised for any 4xx/5xx response left after retries."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def _body_excerpt(text: str, limit: int = 200) -> str:
    """Collapse whitespace and truncate a response body for safe inclusion in an error message."""
    return " ".join((text or "").split())[:limit]


def _redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {k: ("***" if _SECRET_HEADER_RE.search(k) else v) for k, v in headers.items()}


def _redact_url(url: httpx.URL) -> str:
    if any(_SECRET_PARAM_RE.search(name) for name in url.params.keys()):
        return str(url.copy_with(query=None))
    return str(url)


class _HostRateLimiter:
    """Per-host minimum-interval limiter: at most `rate` requests/second to any listed host."""

    def __init__(self, rates: dict[str, float], sleep: Callable[[float], Awaitable[None]]):
        self._intervals = {host: 1.0 / rate for host, rate in rates.items() if rate > 0}
        self._sleep = sleep
        self._last_call: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _interval_for(self, host: str | None) -> float | None:
        if not host:
            return None
        if host in self._intervals:
            return self._intervals[host]
        for known, interval in self._intervals.items():
            if host.endswith("." + known):
                return interval
        return None

    async def wait(self, host: str | None) -> None:
        interval = self._interval_for(host)
        if interval is None:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last_call.get(host)
            if last is not None:
                remaining = interval - (now - last)
                if remaining > 0:
                    await self._sleep(remaining)
            self._last_call[host] = time.monotonic()


class HttpClient:
    """Async JSON/text HTTP client with per-host throttling, GET retries and log redaction."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        rates: dict[str, float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._sleep = sleep or asyncio.sleep
        self._limiter = _HostRateLimiter(DEFAULT_RATES if rates is None else rates, self._sleep)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _log(self, method: str, url: httpx.URL, headers: dict[str, str] | None) -> None:
        logger.debug("%s %s headers=%s", method, _redact_url(url), _redact_headers(headers))

    async def _get(
        self, url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float = 20.0
    ) -> httpx.Response:
        request_url = httpx.URL(url, params=params or {})
        host = request_url.host
        resp: httpx.Response | None = None
        for attempt in range(_MAX_GET_ATTEMPTS):
            await self._limiter.wait(host)
            self._log("GET", request_url, headers)
            resp = await self._client.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < _MAX_GET_ATTEMPTS - 1:
                    await self._sleep(_RETRY_BACKOFF_SEC[attempt])
                    continue
            break
        assert resp is not None
        if resp.status_code >= 400:
            raise HttpError(resp.status_code, resp.text)
        return resp

    async def get_json(
        self, url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float = 20.0
    ) -> Any:
        resp = await self._get(url, params=params, headers=headers, timeout=timeout)
        try:
            return resp.json()
        except ValueError:
            # A 200 with a non-JSON body (e.g. an HTML interstitial/error page from an upstream
            # that doesn't bother setting a 4xx/5xx status) would otherwise raise a raw
            # json.JSONDecodeError past every connector's HttpError-only handling. Surface it as
            # an HttpError instead so callers only ever need to guard against one exception type.
            raise HttpError(resp.status_code, f"{_body_excerpt(resp.text)} (invalid JSON)") from None

    async def get_text(
        self, url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float = 20.0
    ) -> str:
        resp = await self._get(url, params=params, headers=headers, timeout=timeout)
        return resp.text

    async def post_json(
        self, url: str, *, json: Any = None, headers: dict | None = None, timeout: float = 20.0
    ) -> Any:
        # POSTs are never retried (the project forbids blind order retries; callers that need
        # retry-with-reconciliation semantics implement it themselves).
        request_url = httpx.URL(url)
        host = request_url.host
        await self._limiter.wait(host)
        self._log("POST", request_url, headers)
        resp = await self._client.post(url, json=json, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise HttpError(resp.status_code, resp.text)
        return resp.json()
