from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx

from finagent.research.evidence import EvidenceStore
from finagent.research.http import HttpClient
from finagent.store.appdb import AppDB
from finagent.trading.marketdata import DATA_BASE, AlpacaData

FAKE_SECRET = "s3cr3t-DISTINCTIVE-VALUE-do-not-leak-9f8e7d"


async def _noop_sleep(sec: float) -> None:
    pass


def _marketdata(handler, now=None, key_id="test-key-id", secret=FAKE_SECRET):
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    async_client = httpx.AsyncClient(transport=transport)
    http = HttpClient(client=async_client, sleep=_noop_sleep)
    db = AppDB(":memory:")
    evidence = EvidenceStore(db)
    md = AlpacaData(key_id, secret, http, now=now)
    return md, calls, evidence


# --- constructor guards ------------------------------------------------------


def test_constructor_refuses_empty_key_id():
    db = AppDB(":memory:")
    http = HttpClient(client=httpx.AsyncClient())
    import pytest

    with pytest.raises(ValueError):
        AlpacaData("", "s", http)


def test_constructor_refuses_empty_secret():
    http = HttpClient(client=httpx.AsyncClient())
    import pytest

    with pytest.raises(ValueError):
        AlpacaData("k", "", http)


# --- snapshots -----------------------------------------------------------------


SNAPSHOTS_PAYLOAD = {
    "snapshots": {
        "AAPL": {
            "latestQuote": {"bp": 150.10, "ap": 150.15, "t": "2026-09-27T14:59:00Z"},
            "latestTrade": {"p": 150.12, "t": "2026-09-27T14:59:30Z"},
        }
    }
}


async def test_snapshots_maps_latest_quote_and_trade():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/stocks/snapshots"
        assert request.url.params["symbols"] == "AAPL"
        assert request.url.params["feed"] == "iex"
        return httpx.Response(200, json=SNAPSHOTS_PAYLOAD)

    md, calls, _ = _marketdata(handler)
    quotes = await md.snapshots(["AAPL"])

    assert calls
    q = quotes["AAPL"]
    assert q.symbol == "AAPL"
    assert q.bid == Decimal("150.10")
    assert q.ask == Decimal("150.15")
    assert q.last == Decimal("150.12")
    assert q.as_of == "2026-09-27T14:59:30Z"
    assert q.feed == "iex"


async def test_snapshots_joins_multiple_symbols():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbols"] == "AAPL,MSFT"
        return httpx.Response(200, json={"snapshots": {}})

    md, calls, _ = _marketdata(handler)
    await md.snapshots(["AAPL", "MSFT"])
    assert calls


# --- bars ------------------------------------------------------------------------


def _bar(t: str, price: str) -> dict:
    return {"t": t, "o": price, "h": price, "l": price, "c": price, "v": 1000}


async def test_bars_requests_feed_iex_when_end_is_recent():
    now = datetime(2026, 9, 27, 15, 0, 0, tzinfo=timezone.utc)
    recent_end = "2026-09-27T14:55:00Z"  # 5 minutes before "now"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["feed"] == "iex"
        return httpx.Response(200, json={"bars": {"AAPL": [_bar("2026-09-27T14:00:00Z", "150.00")]}}, )

    md, calls, _ = _marketdata(handler, now=lambda: now)
    bars = await md.bars("AAPL", end=recent_end)
    assert calls
    assert len(bars) == 1
    assert bars[0].c == Decimal("150.00")


async def test_bars_requests_feed_sip_when_end_is_old():
    now = datetime(2026, 9, 27, 15, 0, 0, tzinfo=timezone.utc)
    old_end = "2026-09-27T14:00:00Z"  # 1 hour before "now" -> >= 15 min old

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["feed"] == "sip"
        return httpx.Response(200, json={"bars": {"AAPL": []}})

    md, calls, _ = _marketdata(handler, now=lambda: now)
    await md.bars("AAPL", end=old_end)
    assert calls


async def test_bars_feed_iex_when_no_end_given():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["feed"] == "iex"
        return httpx.Response(200, json={"bars": {"AAPL": []}})

    md, calls, _ = _marketdata(handler)
    await md.bars("AAPL")
    assert calls


async def test_bars_sends_adjustment_all_and_timeframe():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["adjustment"] == "all"
        assert request.url.params["timeframe"] == "1Day"
        assert request.url.params["symbols"] == "AAPL"
        return httpx.Response(200, json={"bars": {"AAPL": []}})

    md, calls, _ = _marketdata(handler)
    await md.bars("AAPL")
    assert calls


async def test_bars_parses_ohlcv_as_decimal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"bars": {"AAPL": [_bar("2026-09-27T14:00:00Z", "150.25")]}}
        )

    md, _, _ = _marketdata(handler)
    bars = await md.bars("AAPL")
    assert len(bars) == 1
    bar = bars[0]
    assert bar.o == Decimal("150.25")
    assert bar.h == Decimal("150.25")
    assert bar.l == Decimal("150.25")
    assert bar.c == Decimal("150.25")
    assert bar.v == Decimal("1000")
    assert bar.t == "2026-09-27T14:00:00Z"


async def test_bars_follows_next_page_token_until_limit_reached():
    pages = [
        {
            "bars": {"AAPL": [_bar(f"2026-09-27T1{i}:00:00Z", "100.00") for i in range(3)]},
            "next_page_token": "page-2",
        },
        {
            "bars": {"AAPL": [_bar(f"2026-09-27T2{i}:00:00Z", "101.00") for i in range(3)]},
            "next_page_token": None,
        },
    ]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if call_count == 0:
            assert "page_token" not in request.url.params
        else:
            assert request.url.params["page_token"] == "page-2"
        resp = httpx.Response(200, json=pages[call_count])
        call_count += 1
        return resp

    md, calls, _ = _marketdata(handler)
    bars = await md.bars("AAPL", limit=5)

    assert len(calls) == 2
    assert len(bars) == 5  # stops once the limit (5) is reached, not all 6 available


async def test_bars_stops_without_a_second_call_when_no_next_page_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": {"AAPL": [_bar("2026-09-27T14:00:00Z", "100.00")]}})

    md, calls, _ = _marketdata(handler)
    bars = await md.bars("AAPL", limit=500)
    assert len(calls) == 1
    assert len(bars) == 1


# --- news --------------------------------------------------------------------------


NEWS_PAYLOAD = {
    "news": [
        {
            "headline": "Apple beats earnings expectations",
            "summary": "Apple reported strong Q4 results.",
            "source": "benzinga",
            "created_at": "2026-09-27T12:00:00Z",
            "url": "https://example.com/apple-news",
        }
    ]
}


async def test_news_records_tier2_untrusted_with_created_at_as_published_at():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta1/news"
        assert request.url.params["include_content"] == "false"
        return httpx.Response(200, json=NEWS_PAYLOAD)

    md, calls, evidence = _marketdata(handler)
    result = await md.news("session-1", evidence, symbols=["AAPL"], limit=5)

    assert calls
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.tier == 2
    assert source.untrusted is True
    assert source.published_at == "2026-09-27T12:00:00Z"
    assert source.connector == "news"


async def test_news_wraps_every_external_field_inside_untrusted_boundary():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=NEWS_PAYLOAD)

    md, _, evidence = _marketdata(handler)
    result = await md.news("session-1", evidence, symbols=["AAPL"])

    source_id = result.sources[0].id
    assert f'<untrusted source="{source_id}">' in result.text
    assert "</untrusted>" in result.text
    # headline, summary, source and created_at must all be inside the wrap, not appended after it
    open_tag = f'<untrusted source="{source_id}">'
    start = result.text.index(open_tag) + len(open_tag)
    end = result.text.index("</untrusted>", start)
    wrapped_body = result.text[start:end]
    assert "Apple beats earnings expectations" in wrapped_body
    assert "Apple reported strong Q4 results." in wrapped_body
    assert "benzinga" in wrapped_body
    assert "2026-09-27T12:00:00Z" in wrapped_body
    assert result.text[end + len("</untrusted>"):].strip() == "" or "benzinga" not in result.text[end + len("</untrusted>"):]


async def test_news_limit_is_capped_at_50():
    seen_params = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json={"news": []})

    md, _, evidence = _marketdata(handler)
    await md.news("session-1", evidence, limit=500)
    assert seen_params[0]["limit"] == "50"


async def test_news_no_results_is_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"news": []})

    md, _, evidence = _marketdata(handler)
    result = await md.news("session-1", evidence)
    assert result.is_error is False
    assert result.sources == []


# --- movers / most_actives ---------------------------------------------------------


async def test_movers_returns_raw_dict():
    payload = {"gainers": [{"symbol": "AAPL", "percent_change": 5.0}], "losers": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta1/screener/stocks/movers"
        assert request.url.params["top"] == "10"
        return httpx.Response(200, json=payload)

    md, calls, _ = _marketdata(handler)
    result = await md.movers()
    assert calls
    assert result == payload


async def test_most_actives_returns_list_by_volume():
    payload = {"most_actives": [{"symbol": "AAPL", "volume": 1000000}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta1/screener/stocks/most-actives"
        assert request.url.params["by"] == "volume"
        assert request.url.params["top"] == "10"
        return httpx.Response(200, json=payload)

    md, calls, _ = _marketdata(handler)
    result = await md.most_actives()
    assert calls
    assert result == payload["most_actives"]


# --- secrets never leak into logs -------------------------------------------------


async def test_secret_never_appears_in_logs(caplog):
    import logging

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SNAPSHOTS_PAYLOAD)

    md, _, _ = _marketdata(handler)
    with caplog.at_level(logging.DEBUG):
        await md.snapshots(["AAPL"])

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_SECRET not in log_text
