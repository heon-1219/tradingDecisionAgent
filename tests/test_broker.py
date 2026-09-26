import logging
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from finagent.trading.broker import PAPER_BASE, AlpacaPaper, BrokerError, OrderUncertain
from finagent.trading.models import OrderLeg

FAKE_SECRET = "s3cr3t-DISTINCTIVE-VALUE-do-not-leak-9f8e7d"


def _broker(handler, key_id: str = "test-key-id", secret: str = FAKE_SECRET) -> tuple[AlpacaPaper, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    client = httpx.AsyncClient(base_url=PAPER_BASE, transport=transport)
    return AlpacaPaper(key_id, secret, client=client), calls


# --- paper-guard test (verbatim from the brief) ----------------------------------


def test_trading_host_is_paper_only():
    b = AlpacaPaper("k", "s")
    assert str(b._client.base_url).rstrip("/") == "https://paper-api.alpaca.markets"
    with pytest.raises(ValueError):
        AlpacaPaper("k", "s", client=httpx.AsyncClient(base_url="https://api.alpaca.markets"))


def test_constructor_refuses_empty_key_id():
    with pytest.raises(ValueError):
        AlpacaPaper("", "s")


def test_constructor_refuses_empty_secret():
    with pytest.raises(ValueError):
        AlpacaPaper("k", "")


def test_constructor_accepts_client_with_correct_host_and_trailing_slash_variants():
    client = httpx.AsyncClient(base_url="https://paper-api.alpaca.markets/")
    b = AlpacaPaper("k", "s", client=client)
    assert b._client is client


# --- account -----------------------------------------------------------------


ACCOUNT_PAYLOAD = {
    "status": "ACTIVE",
    "equity": "100000.50",
    "cash": "50000.25",
    "buying_power": "200000.00",
    "non_marginable_buying_power": "50000.25",
    "last_equity": "99000.10",
    "trading_blocked": False,
    "account_blocked": False,
    "trade_suspended_by_user": False,
    "shorting_enabled": True,
    # deliberately NOT including the removed PDT fields (pattern_day_trader, daytrade_count,
    # daytrading_buying_power, ...) - the model must not require them.
}


async def test_account_parses_decimal_strings():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/account"
        return httpx.Response(200, json=ACCOUNT_PAYLOAD)

    b, calls = _broker(handler)
    account = await b.account()

    assert isinstance(account.equity, Decimal)
    assert account.equity == Decimal("100000.50")
    assert isinstance(account.cash, Decimal)
    assert account.status == "ACTIVE"
    assert account.shorting_enabled is True
    assert calls


async def test_account_ignores_unknown_fields_including_removed_pdt_fields():
    payload = dict(ACCOUNT_PAYLOAD)
    payload["pattern_day_trader"] = False
    payload["daytrade_count"] = 0
    payload["some_future_field_we_dont_know_about"] = {"nested": True}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    b, _ = _broker(handler)
    account = await b.account()
    assert account.status == "ACTIVE"


# --- positions -----------------------------------------------------------------


async def test_positions_returns_dict_keyed_by_symbol():
    payload = [
        {
            "symbol": "AAPL",
            "qty": "10",
            "qty_available": "10",
            "avg_entry_price": "150.00",
            "current_price": "155.25",
            "market_value": "1552.50",
            "unrealized_pl": "52.50",
            "unrealized_plpc": "0.035",
            "side": "long",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/positions"
        return httpx.Response(200, json=payload)

    b, _ = _broker(handler)
    positions = await b.positions()

    assert set(positions.keys()) == {"AAPL"}
    assert positions["AAPL"].avg_entry_price == Decimal("150.00")
    assert isinstance(positions["AAPL"].qty, Decimal)


# --- orders (list) ---------------------------------------------------------------


async def test_orders_sends_status_limit_and_comma_joined_symbols():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/orders"
        params = request.url.params
        assert params["status"] == "open"
        assert params["limit"] == "50"
        assert params["symbols"] == "AAPL,MSFT"
        assert params["nested"] == "false"
        return httpx.Response(200, json=[])

    b, calls = _broker(handler)
    result = await b.orders(symbols=["AAPL", "MSFT"])
    assert result == []
    assert calls


async def test_orders_parses_list_of_order_snapshots():
    payload = [
        {
            "id": "order-1",
            "client_order_id": "coid-1",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "notional": None,
            "type": "market",
            "time_in_force": "day",
            "limit_price": None,
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "150.10",
            "submitted_at": "2026-09-27T10:00:00Z",
            "updated_at": "2026-09-27T10:00:05Z",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    b, _ = _broker(handler)
    orders = await b.orders()
    assert len(orders) == 1
    assert orders[0].qty == Decimal("10")
    assert orders[0].notional is None


# --- asset / clock ---------------------------------------------------------------


async def test_asset_maps_class_field_from_alpacas_class_key():
    payload = {
        "symbol": "AAPL",
        "class": "us_equity",
        "status": "active",
        "tradable": True,
        "fractionable": True,
        "shortable": True,
        "marginable": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/assets/AAPL"
        return httpx.Response(200, json=payload)

    b, _ = _broker(handler)
    asset = await b.asset("AAPL")
    assert asset.cls == "us_equity"
    assert asset.symbol == "AAPL"


async def test_clock_parses():
    payload = {
        "is_open": True,
        "next_open": "2026-09-28T13:30:00Z",
        "next_close": "2026-09-27T20:00:00Z",
        "timestamp": "2026-09-27T15:00:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/clock"
        return httpx.Response(200, json=payload)

    b, _ = _broker(handler)
    clock = await b.clock()
    assert clock.is_open is True


# --- submit_order ----------------------------------------------------------------


ORDER_ID = "5f2c9a1e-2b3c-4d4e-9f5a-6b7c8d9e0f1a"

ORDER_RESPONSE = {
    "id": ORDER_ID,
    "client_order_id": "coid-abc",
    "symbol": "AAPL",
    "side": "buy",
    "qty": "3",
    "notional": None,
    "type": "limit",
    "time_in_force": "day",
    "limit_price": "150.00",
    "status": "accepted",
    "filled_qty": "0",
    "filled_avg_price": None,
    "submitted_at": "2026-09-27T10:00:00Z",
    "updated_at": "2026-09-27T10:00:00Z",
}


async def test_submit_order_sends_client_order_id_and_decimal_strings_no_exponent():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        assert request.url.path == "/v2/orders"
        bodies.append(_json.loads(request.content))
        return httpx.Response(200, json=ORDER_RESPONSE)

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="aapl", side="buy", qty=Decimal("3"), type="limit", limit_price=Decimal("150.00"))
    order = await b.submit_order(leg, client_order_id="coid-abc")

    assert len(bodies) == 1
    body = bodies[0]
    assert body["client_order_id"] == "coid-abc"
    assert body["symbol"] == "AAPL"
    assert body["side"] == "buy"
    assert body["type"] == "limit"
    assert body["time_in_force"] == "day"
    assert body["qty"] == "3"
    assert "notional" not in body
    assert body["limit_price"] == "150.00"
    assert "e" not in body["qty"].lower()
    assert "extended_hours" not in body
    assert "order_class" not in body
    assert order.id == ORDER_ID


async def test_submit_order_with_notional_sends_notional_not_qty():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        bodies.append(_json.loads(request.content))
        return httpx.Response(200, json=ORDER_RESPONSE)

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", notional=Decimal("500.50"), type="market")
    await b.submit_order(leg, client_order_id="coid-xyz")

    body = bodies[0]
    # format(d, "f") preserves the Decimal's own precision (no exponent notation), rather than
    # normalizing trailing zeros away.
    assert body["notional"] == "500.50"
    assert "qty" not in body
    assert "limit_price" not in body


async def test_submit_order_timeout_raises_order_uncertain():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain) as exc_info:
        await b.submit_order(leg, client_order_id="coid-timeout")
    assert exc_info.value.status == 0


async def test_submit_order_transport_error_raises_order_uncertain():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain):
        await b.submit_order(leg, client_order_id="coid-connect-error")


async def test_submit_order_5xx_raises_order_uncertain():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "internal error"})

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain) as exc_info:
        await b.submit_order(leg, client_order_id="coid-500")
    assert exc_info.value.status == 500


async def test_submit_order_422_duplicate_client_order_id_raises_brokererror_with_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"code": 40010001, "message": "client_order_id must be unique"}
        )

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(BrokerError) as exc_info:
        await b.submit_order(leg, client_order_id="coid-dup")
    assert exc_info.value.code == 40010001
    assert exc_info.value.status == 422
    assert not isinstance(exc_info.value, OrderUncertain)


async def test_submit_order_403_insufficient_buying_power_maps_to_brokererror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"code": 40310000, "message": "insufficient buying power"}
        )

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(BrokerError) as exc_info:
        await b.submit_order(leg, client_order_id="coid-403")
    assert "insufficient buying power" in exc_info.value.message
    assert exc_info.value.status == 403


# --- get_order / get_order_by_client_id / cancel_order / set_suspend_trade -------


async def test_get_order_by_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v2/orders/{ORDER_ID}"
        return httpx.Response(200, json=ORDER_RESPONSE)

    b, _ = _broker(handler)
    order = await b.get_order(ORDER_ID)
    assert order.id == ORDER_ID


async def test_get_order_by_client_id_found():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "orders:by_client_order_id" in request.url.path
        assert request.url.params["client_order_id"] == "coid-abc"
        return httpx.Response(200, json=ORDER_RESPONSE)

    b, _ = _broker(handler)
    order = await b.get_order_by_client_id("coid-abc")
    assert order is not None
    assert order.client_order_id == "coid-abc"


async def test_get_order_by_client_id_returns_none_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "order not found"})

    b, _ = _broker(handler)
    order = await b.get_order_by_client_id("does-not-exist")
    assert order is None


async def test_cancel_order_issues_delete():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/v2/orders/{ORDER_ID}"
        return httpx.Response(204)

    b, calls = _broker(handler)
    result = await b.cancel_order(ORDER_ID)
    assert result is None
    assert calls


async def test_set_suspend_trade_issues_patch_with_body():
    import json as _json

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/v2/account/configurations"
        bodies.append(_json.loads(request.content))
        return httpx.Response(200, json={"suspend_trade": True})

    b, _ = _broker(handler)
    await b.set_suspend_trade(True)
    assert bodies == [{"suspend_trade": True}]


# --- secrets never leak into logs or exception text ------------------------------


async def test_secret_never_appears_in_logs_or_broker_error_text(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"code": 40010001, "message": "client_order_id must be unique"})

    b, _ = _broker(handler)
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")

    with caplog.at_level(logging.DEBUG):
        try:
            await b.submit_order(leg, client_order_id="coid-secret-check")
        except BrokerError as exc:
            exc_text = str(exc) + repr(exc)
        else:
            exc_text = ""

    assert FAKE_SECRET not in exc_text
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_SECRET not in log_text


async def test_secret_never_appears_in_logs_for_reads(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ACCOUNT_PAYLOAD)

    b, _ = _broker(handler)
    with caplog.at_level(logging.DEBUG):
        await b.account()

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_SECRET not in log_text


# --- CRITICAL 1: path injection via order_id / symbol / client_order_id ----------


VALID_ORDER_ID = "904837e3-3b76-47ec-b432-046db621571b"


@pytest.mark.parametrize(
    "bad_order_id", ["../positions", "", "a/b", "%2e%2e", "not-a-uuid", "x" * 300]
)
async def test_cancel_order_rejects_invalid_order_id(bad_order_id):
    b, calls = _broker(lambda r: httpx.Response(204))
    with pytest.raises(ValueError):
        await b.cancel_order(bad_order_id)
    assert calls == []


@pytest.mark.parametrize("bad_order_id", ["../positions", "", "a/b", "%2e%2e", "not-a-uuid"])
async def test_get_order_rejects_invalid_order_id(bad_order_id):
    b, calls = _broker(lambda r: httpx.Response(200, json=ORDER_RESPONSE))
    with pytest.raises(ValueError):
        await b.get_order(bad_order_id)
    assert calls == []


async def test_cancel_order_accepts_valid_uuid_and_percent_encodes_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v2/orders/{VALID_ORDER_ID}"
        return httpx.Response(204)

    b, calls = _broker(handler)
    await b.cancel_order(VALID_ORDER_ID)
    assert calls


@pytest.mark.parametrize("bad_symbol", ["a/b", "../etc", "%2e%2e", "", "x" * 30])
async def test_asset_rejects_invalid_symbol(bad_symbol):
    b, calls = _broker(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        await b.asset(bad_symbol)
    assert calls == []


@pytest.mark.parametrize("bad_client_order_id", ["", "x" * 200, "has space", "semi;colon"])
async def test_get_order_by_client_id_rejects_invalid_client_order_id(bad_client_order_id):
    b, calls = _broker(lambda r: httpx.Response(200, json=ORDER_RESPONSE))
    with pytest.raises(ValueError):
        await b.get_order_by_client_id(bad_client_order_id)
    assert calls == []


@pytest.mark.parametrize("bad_client_order_id", ["", "x" * 200, "has space", "semi;colon"])
async def test_submit_order_rejects_invalid_client_order_id(bad_client_order_id):
    b, calls = _broker(lambda r: httpx.Response(200, json=ORDER_RESPONSE))
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(ValueError):
        await b.submit_order(leg, bad_client_order_id)
    assert calls == []


# --- IMPORTANT 2: paper-guard hardening (origin, not just host; every request) ---


async def test_constructor_refuses_http_scheme():
    with pytest.raises(ValueError):
        AlpacaPaper("k", "s", client=httpx.AsyncClient(base_url="http://paper-api.alpaca.markets"))


async def test_constructor_refuses_non_standard_port():
    with pytest.raises(ValueError):
        AlpacaPaper(
            "k", "s", client=httpx.AsyncClient(base_url="https://paper-api.alpaca.markets:8080")
        )


async def test_constructor_accepts_explicit_default_https_port():
    client = httpx.AsyncClient(base_url="https://paper-api.alpaca.markets:443")
    b = AlpacaPaper("k", "s", client=client)
    assert b._client is client


async def test_base_url_mutation_after_construction_is_refused_before_any_request():
    b, calls = _broker(lambda r: httpx.Response(204))
    b._client.base_url = "https://api.alpaca.markets"
    with pytest.raises(ValueError):
        await b.cancel_order(VALID_ORDER_ID)
    assert calls == []


async def test_redirect_is_never_followed_and_secret_never_reaches_other_host():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "paper-api.alpaca.markets":
            return httpx.Response(302, headers={"Location": "https://evil.example/steal"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=PAPER_BASE, transport=transport, follow_redirects=True)
    b = AlpacaPaper("k", FAKE_SECRET, client=client)

    with pytest.raises(BrokerError) as exc_info:
        await b.clock()

    assert exc_info.value.status == 302
    assert len(calls) == 1  # the redirect was never followed
    assert calls[0].url.host == "paper-api.alpaca.markets"
    assert FAKE_SECRET not in str(exc_info.value)


# --- IMPORTANT 3: 3xx / invalid-JSON / validation-failure never escape raw -------


async def test_read_3xx_raises_brokererror_not_a_raw_exception():
    b, _ = _broker(lambda r: httpx.Response(302, headers={"Location": "https://evil.example/"}))
    with pytest.raises(BrokerError) as exc_info:
        await b.clock()
    assert exc_info.value.status == 302
    assert not isinstance(exc_info.value, OrderUncertain)


async def test_submit_order_3xx_raises_order_uncertain():
    b, _ = _broker(lambda r: httpx.Response(302, headers={"Location": "https://evil.example/"}))
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain) as exc_info:
        await b.submit_order(leg, client_order_id="coid-302")
    assert exc_info.value.status == 302


async def test_submit_order_200_non_json_raises_order_uncertain():
    b, _ = _broker(lambda r: httpx.Response(200, text="<html>ok</html>"))
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain):
        await b.submit_order(leg, client_order_id="coid-badjson")


async def test_submit_order_200_missing_fields_raises_order_uncertain():
    b, _ = _broker(lambda r: httpx.Response(200, json={"id": "x"}))
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    with pytest.raises(OrderUncertain):
        await b.submit_order(leg, client_order_id="coid-badshape")


async def test_account_200_non_json_raises_brokererror_not_order_uncertain():
    b, _ = _broker(lambda r: httpx.Response(200, text="<html>ok</html>"))
    with pytest.raises(BrokerError) as exc_info:
        await b.account()
    assert not isinstance(exc_info.value, OrderUncertain)


async def test_account_200_missing_fields_raises_brokererror():
    b, _ = _broker(lambda r: httpx.Response(200, json={"status": "ACTIVE"}))
    with pytest.raises(BrokerError) as exc_info:
        await b.account()
    assert not isinstance(exc_info.value, OrderUncertain)


async def test_get_order_200_non_json_raises_brokererror():
    b, _ = _broker(lambda r: httpx.Response(200, text="oops"))
    with pytest.raises(BrokerError) as exc_info:
        await b.get_order(VALID_ORDER_ID)
    assert not isinstance(exc_info.value, OrderUncertain)


# --- Minor 4: OrderLeg validation hardening --------------------------------------


def test_orderleg_rejects_float_qty():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="AAPL", side="buy", qty=0.1, type="market")


def test_orderleg_rejects_float_notional():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="AAPL", side="buy", notional=100.0, type="market")


def test_orderleg_rejects_float_limit_price():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="limit", limit_price=10.5)


def test_orderleg_accepts_int_and_str_quantities():
    assert OrderLeg(symbol="AAPL", side="buy", qty=1, type="market").qty == Decimal("1")
    assert OrderLeg(symbol="AAPL", side="buy", qty="1.5", type="market").qty == Decimal("1.5")


def test_orderleg_requires_limit_price_for_limit_order():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="limit")


def test_orderleg_forbids_limit_price_for_market_order():
    with pytest.raises(ValidationError):
        OrderLeg(
            symbol="AAPL", side="buy", qty=Decimal("1"), type="market", limit_price=Decimal("10")
        )


def test_orderleg_rejects_non_positive_limit_price():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="limit", limit_price=Decimal("0"))


def test_orderleg_rejects_notional_with_limit_type():
    with pytest.raises(ValidationError):
        OrderLeg(
            symbol="AAPL",
            side="buy",
            notional=Decimal("100"),
            type="limit",
            limit_price=Decimal("10"),
        )


def test_orderleg_rejects_notional_with_gtc_time_in_force():
    with pytest.raises(ValidationError):
        OrderLeg(
            symbol="AAPL",
            side="buy",
            notional=Decimal("100"),
            type="market",
            time_in_force="gtc",
        )
