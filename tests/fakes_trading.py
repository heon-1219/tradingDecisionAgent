"""Scriptable in-memory stand-ins for `AlpacaPaper` (broker) and `AlpacaData` (market data), used by
`tests/test_execution.py`. No network: every call is recorded, and every response or failure is
set up by the test.

Default world: an ACTIVE account with $100,000 equity and $50,000 cash buying power; a 10-share
MSFT position; tradable, fractionable AAPL, MSFT and KEY (KeyCorp - a real ticker that also
happens to be the word the ledger's dict-key redaction looks for); the market open; quotes
AAPL $200, MSFT $400, KEY $15.

`FakeBroker.submit_actions` scripts `submit_order`, one entry consumed per call (a missing entry
means "ok"):
- "ok": the order is placed and returned (status `initial_status`).
- "placed_then_timeout": the order reaches the broker, then the call raises `OrderUncertain`
  (the response was lost).
- "timeout": the call raises `OrderUncertain` and nothing is placed.
- an exception instance: raised as-is, nothing placed (e.g. a 4xx `BrokerError`).
- a callable `(broker, leg, client_order_id)`: runs, and whatever it returns is returned (it may
  call `broker.place(...)` and/or raise).
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from finagent.trading.broker import BrokerError, OrderUncertain
from finagent.trading.models import (
    AccountSnapshot,
    AssetInfo,
    ClockInfo,
    OrderLeg,
    OrderSnapshot,
    PositionSnapshot,
    Quote,
)

D = Decimal


def make_account(**over) -> AccountSnapshot:
    fields = dict(
        status="ACTIVE",
        equity=D("100000"),
        cash=D("50000"),
        buying_power=D("50000"),
        non_marginable_buying_power=D("50000"),
        last_equity=D("100000"),
        trading_blocked=False,
        account_blocked=False,
        trade_suspended_by_user=False,
        shorting_enabled=False,
    )
    fields.update(over)
    return AccountSnapshot(**fields)


def make_asset(symbol: str, **over) -> AssetInfo:
    fields = dict(
        symbol=symbol,
        cls="us_equity",
        status="active",
        tradable=True,
        fractionable=True,
        shortable=False,
        marginable=True,
    )
    fields.update(over)
    return AssetInfo(**fields)


def make_position(symbol: str, qty: str, price: str) -> PositionSnapshot:
    q, p = D(qty), D(price)
    return PositionSnapshot(
        symbol=symbol,
        qty=q,
        qty_available=q,
        avg_entry_price=p,
        current_price=p,
        market_value=q * p,
        unrealized_pl=D("0"),
        unrealized_plpc=D("0"),
        side="long",
    )


def make_clock(is_open: bool = True) -> ClockInfo:
    return ClockInfo(
        is_open=is_open,
        next_open="2026-09-29T13:30:00Z",
        next_close="2026-09-28T20:00:00Z",
        timestamp="2026-09-28T14:00:00Z",
    )


def make_quote(symbol: str, last: str, spread: str = "0.01") -> Quote:
    l, s = D(last), D(spread)
    return Quote(symbol=symbol, bid=l - s, ask=l + s, last=l, as_of="2026-09-28T14:00:00Z",
                 feed="iex")


def make_order(leg: OrderLeg, client_order_id: str, status: str = "accepted",
               order_id: str | None = None) -> OrderSnapshot:
    return OrderSnapshot(
        id=order_id or str(uuid.uuid4()),
        client_order_id=client_order_id,
        symbol=leg.symbol,
        side=leg.side,
        qty=leg.qty,
        notional=leg.notional,
        type=leg.type,
        time_in_force=leg.time_in_force,
        limit_price=leg.limit_price,
        status=status,
        filled_qty=D("0"),
        submitted_at="2026-09-28T14:00:01Z",
        updated_at="2026-09-28T14:00:01Z",
    )


def make_open_order(symbol: str, side: str = "buy", qty: str = "2", limit: str | None = None,
                    status: str = "new") -> OrderSnapshot:
    return OrderSnapshot(
        id=str(uuid.uuid4()),
        client_order_id=f"manual-{symbol}-{uuid.uuid4().hex[:6]}",
        symbol=symbol,
        side=side,
        qty=D(qty),
        type="limit" if limit else "market",
        time_in_force="day",
        limit_price=D(limit) if limit else None,
        status=status,
        filled_qty=D("0"),
    )


class FakeBroker:
    """Duck-typed `AlpacaPaper`. See the module docstring for `submit_actions`."""

    def __init__(self, *, account: AccountSnapshot | None = None,
                 positions: dict[str, PositionSnapshot] | None = None,
                 open_orders: list[OrderSnapshot] | None = None,
                 assets: dict[str, AssetInfo] | None = None,
                 clock: ClockInfo | None = None):
        self.account_snapshot = account or make_account()
        self.positions_map = (
            positions if positions is not None else {"MSFT": make_position("MSFT", "10", "400")}
        )
        self.open_orders = list(open_orders or [])
        self.assets = (
            assets if assets is not None
            else {s: make_asset(s) for s in ("AAPL", "MSFT", "KEY")}
        )
        self.asset_errors: dict[str, Exception] = {}
        self.clock_info = clock or make_clock()
        self.account_error: Exception | None = None

        self.initial_status = "accepted"
        self.submit_actions: list[Any] = []
        self.submits = 0
        self.submitted: list[tuple[OrderLeg, str]] = []
        self.orders_by_id: dict[str, OrderSnapshot] = {}
        self.orders_by_cid: dict[str, OrderSnapshot] = {}

        self.reconcile_calls: list[str] = []
        self.reconcile_error: Exception | None = None

        self.status_script: dict[str, list[str]] = {}
        self.get_order_calls: list[str] = []
        self.get_order_error: Exception | None = None

        self.cancelled: list[str] = []
        self.cancel_errors: dict[str, Exception] = {}
        self.suspend_calls: list[bool] = []
        self.suspend_error: Exception | None = None
        self.orders_calls: list[dict] = []
        self.calls: list[str] = []

    # --- helpers for tests ------------------------------------------------------------------

    def place(self, leg: OrderLeg, client_order_id: str, status: str | None = None) -> OrderSnapshot:
        order = make_order(leg, client_order_id, status or self.initial_status)
        self.orders_by_id[order.id] = order
        self.orders_by_cid[client_order_id] = order
        return order

    # --- AlpacaPaper surface ------------------------------------------------------------------

    async def account(self) -> AccountSnapshot:
        self.calls.append("account")
        if self.account_error is not None:
            raise self.account_error
        return self.account_snapshot

    async def positions(self) -> dict[str, PositionSnapshot]:
        self.calls.append("positions")
        return dict(self.positions_map)

    async def orders(self, status: str = "open", limit: int = 50,
                     symbols: list[str] | None = None) -> list[OrderSnapshot]:
        self.calls.append("orders")
        self.orders_calls.append({"status": status, "limit": limit, "symbols": symbols})
        return list(self.open_orders)

    async def asset(self, symbol: str) -> AssetInfo:
        self.calls.append(f"asset:{symbol}")
        if symbol in self.asset_errors:
            raise self.asset_errors[symbol]
        if symbol not in self.assets:
            raise BrokerError(404, 40410000, "asset not found")
        return self.assets[symbol]

    async def clock(self) -> ClockInfo:
        self.calls.append("clock")
        return self.clock_info

    async def submit_order(self, leg: OrderLeg, client_order_id: str) -> OrderSnapshot:
        self.calls.append("submit_order")
        self.submits += 1
        self.submitted.append((leg, client_order_id))
        action = self.submit_actions.pop(0) if self.submit_actions else "ok"
        if action == "ok":
            return self.place(leg, client_order_id)
        if action == "placed_then_timeout":
            self.place(leg, client_order_id)
            raise OrderUncertain(0, None, "ReadTimeout: timed out")
        if action == "timeout":
            raise OrderUncertain(0, None, "ReadTimeout: timed out")
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action(self, leg, client_order_id)
        raise AssertionError(f"unknown submit action {action!r}")

    async def get_order(self, order_id: str) -> OrderSnapshot:
        self.calls.append("get_order")
        self.get_order_calls.append(order_id)
        if self.get_order_error is not None:
            raise self.get_order_error
        order = self.orders_by_id[order_id]
        script = self.status_script.get(order_id)
        if script:
            order = order.model_copy(update={"status": script.pop(0)})
            self.orders_by_id[order_id] = order
            self.orders_by_cid[order.client_order_id] = order
        return order

    async def get_order_by_client_id(self, client_order_id: str) -> OrderSnapshot | None:
        self.calls.append("get_order_by_client_id")
        self.reconcile_calls.append(client_order_id)
        if self.reconcile_error is not None:
            raise self.reconcile_error
        return self.orders_by_cid.get(client_order_id)

    async def cancel_order(self, order_id: str) -> None:
        self.calls.append("cancel_order")
        if order_id in self.cancel_errors:
            raise self.cancel_errors[order_id]
        self.cancelled.append(order_id)

    async def set_suspend_trade(self, suspended: bool) -> None:
        self.calls.append("set_suspend_trade")
        self.suspend_calls.append(suspended)
        if self.suspend_error is not None:
            raise self.suspend_error


class FakeData:
    """Duck-typed `AlpacaData`: `snapshots` returns the configured quotes for the requested symbols
    (unknown symbols are simply absent, like Alpaca) and records every request."""

    def __init__(self, quotes: dict[str, Quote] | None = None):
        self.quotes = dict(quotes) if quotes is not None else {
            "AAPL": make_quote("AAPL", "200"),
            "MSFT": make_quote("MSFT", "400"),
            "KEY": make_quote("KEY", "15"),
        }
        self.calls: list[list[str]] = []
        self.error: Exception | None = None

    def set_price(self, symbol: str, last: str, spread: str = "0.01") -> None:
        self.quotes[symbol] = make_quote(symbol, last, spread)

    async def snapshots(self, symbols: list[str]) -> dict[str, Quote]:
        self.calls.append(list(symbols))
        if self.error is not None:
            raise self.error
        return {s: self.quotes[s] for s in symbols if s in self.quotes}
