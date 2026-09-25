import logging

import httpx
import pytest

from finagent.research.http import HttpClient, HttpError


def _client(handler, **kwargs) -> HttpClient:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return HttpClient(client=async_client, **kwargs)


async def test_get_retries_429_then_succeeds():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    client = _client(handler, sleep=fake_sleep)
    result = await client.get_json("https://example.com/data")

    assert result == {"ok": True}
    assert len(calls) == 2
    assert len(sleeps) == 1


async def test_get_gives_up_after_two_retries():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, text="down")

    async def fake_sleep(sec: float) -> None:
        pass

    client = _client(handler, sleep=fake_sleep)
    with pytest.raises(HttpError) as exc_info:
        await client.get_text("https://example.com/data")

    assert exc_info.value.status == 503
    assert len(calls) == 3  # 1 initial + 2 retries


async def test_post_500_is_not_retried():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="boom")

    client = _client(handler)
    with pytest.raises(HttpError) as exc_info:
        await client.post_json("https://example.com/submit", json={"a": 1})

    assert exc_info.value.status == 500
    assert len(calls) == 1


async def test_rate_limiter_spaces_calls_to_the_same_host():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    waits: list[float] = []

    async def fake_sleep(sec: float) -> None:
        waits.append(sec)

    client = _client(handler, rates={"example.com": 2.0}, sleep=fake_sleep)  # interval 0.5s
    await client.get_json("https://example.com/a")
    await client.get_json("https://example.com/b")

    assert len(waits) == 1
    assert waits[0] == pytest.approx(0.5, abs=0.05)


async def test_rate_limiter_leaves_unlisted_hosts_unlimited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    waits: list[float] = []

    async def fake_sleep(sec: float) -> None:
        waits.append(sec)

    client = _client(handler, rates={"example.com": 2.0}, sleep=fake_sleep)
    await client.get_json("https://other.example/a")
    await client.get_json("https://other.example/b")

    assert waits == []


async def test_secret_header_value_never_logged(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    with caplog.at_level(logging.DEBUG, logger="finagent.research.http"):
        await client.get_json(
            "https://example.com/orders",
            headers={"APCA-API-SECRET-KEY": "super-secret-value", "Accept": "application/json"},
        )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "super-secret-value" not in text


async def test_url_with_key_param_logged_without_query_string(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    with caplog.at_level(logging.DEBUG, logger="finagent.research.http"):
        await client.get_json("https://example.com/search", params={"apikey": "shh", "q": "test"})

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "shh" not in text
    assert "apikey" not in text
