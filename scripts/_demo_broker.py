"""`FillingBroker`: the fake world's `FakeBroker` with orders that fill, for the demo server.

`FakeBroker` leaves a placed order at its initial status for ever, which makes a demo's order
timeline and portfolio look dead. This subclass moves every placed order through
`accepted -> partially_filled -> filled` as time passes (`PARTIAL_AFTER_SEC`, `FILLED_AFTER_SEC`
after it was placed; an order for a single share or a dollar amount skips the partial step), and
when an order fills it updates the account's cash and buying power and the position. Fills are at
the limit price, or at `price_of(symbol)` for a market order. `orders()` lists the placed orders
that are still working, so the Orders panel has something to show.

Like its base class it is in-memory only and never talks to a network: it is a toy for showing the
UI, not a market simulator (no partial fills at different prices, no rejections, no short sales).
"""
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal

_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE.parent / "src", _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _fakeworld import FakeBroker  # noqa: E402 - the very class `build_fake_world` checks against
from finagent.trading.models import OrderLeg, OrderSnapshot, PositionSnapshot  # noqa: E402

PARTIAL_AFTER_SEC = 2.0
FILLED_AFTER_SEC = 6.0
_CENT = Decimal("0.01")
_THOUSANDTH = Decimal("0.001")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FillingBroker(FakeBroker):
    """See the module docstring. It is built with `FakeBroker`'s own arguments (that is how
    `build_fake_world(broker_cls=...)` builds it); set `price_of` (symbol -> price, for market
    orders) and, in tests, `monotonic` (a clock in seconds) afterwards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.price_of: Callable[[str], Decimal | None] = lambda symbol: None
        self.monotonic: Callable[[], float] = time.monotonic
        self._placed_at: dict[str, float] = {}
        self._settled: set[str] = set()

    def place(self, leg: OrderLeg, client_order_id: str, status: str | None = None) -> OrderSnapshot:
        order = super().place(leg, client_order_id, status)
        order = order.model_copy(update={"submitted_at": _stamp(), "updated_at": _stamp()})
        self._store(order)
        self._placed_at[order.id] = self.monotonic()
        return order

    def _store(self, order: OrderSnapshot) -> None:
        self.orders_by_id[order.id] = order
        self.orders_by_cid[order.client_order_id] = order

    def _fill_price(self, order: OrderSnapshot) -> Decimal | None:
        if order.limit_price is not None:
            return order.limit_price
        price = self.price_of(order.symbol)
        return Decimal(price) if price is not None else None

    def _advance(self, order: OrderSnapshot) -> OrderSnapshot:
        """The order as it stands now; a fill is applied to the account exactly once."""
        placed = self._placed_at.get(order.id)
        if placed is None or order.id in self._settled or order.id in self.cancelled:
            return order
        price = self._fill_price(order)
        if price is None or price <= 0:
            return order  # nothing to price it with: it stays accepted
        qty = order.qty if order.qty is not None else (order.notional / price).quantize(_THOUSANDTH, ROUND_DOWN)
        elapsed = self.monotonic() - placed
        if elapsed >= FILLED_AFTER_SEC:
            update = {"status": "filled", "filled_qty": qty, "filled_avg_price": price}
            self._settled.add(order.id)
            self._apply_fill(order, qty, price)
        elif elapsed >= PARTIAL_AFTER_SEC and order.qty is not None and qty >= 2:
            half = (qty / 2).to_integral_value(rounding=ROUND_DOWN)
            update = {"status": "partially_filled", "filled_qty": half, "filled_avg_price": price}
        else:
            return order
        if order.status != update["status"]:
            order = order.model_copy(update={**update, "updated_at": _stamp()})
            self._store(order)
        return order

    def _apply_fill(self, order: OrderSnapshot, qty: Decimal, price: Decimal) -> None:
        value = (qty * price).quantize(_CENT)
        sign = Decimal(1) if order.side == "buy" else Decimal(-1)
        account = self.account_snapshot
        self.account_snapshot = account.model_copy(update={
            "cash": account.cash - sign * value,
            "buying_power": account.buying_power - sign * value,
            "non_marginable_buying_power": account.non_marginable_buying_power - sign * value,
        })
        held = self.positions_map.get(order.symbol)
        old_qty = held.qty if held else Decimal(0)
        new_qty = old_qty + sign * qty
        if new_qty <= 0:
            self.positions_map.pop(order.symbol, None)
            return
        if order.side == "buy":
            avg = ((held.avg_entry_price * old_qty if held else Decimal(0)) + price * qty) / new_qty
        else:
            avg = held.avg_entry_price
        current = held.current_price if held else price
        self.positions_map[order.symbol] = PositionSnapshot(
            symbol=order.symbol, qty=new_qty, qty_available=new_qty, avg_entry_price=avg.quantize(Decimal("0.0001")),
            current_price=current, market_value=(new_qty * current).quantize(_CENT),
            unrealized_pl=(new_qty * (current - avg)).quantize(_CENT),
            unrealized_plpc=((current - avg) / avg).quantize(Decimal("0.0001")) if avg else Decimal(0), side="long",
        )

    def _advance_all(self) -> None:
        for order in list(self.orders_by_id.values()):
            self._advance(order)

    # --- the AlpacaPaper surface ----------------------------------------------------------------

    async def account(self):
        self._advance_all()
        return await super().account()

    async def positions(self):
        self._advance_all()
        return await super().positions()

    async def orders(self, status: str = "open", limit: int = 50, symbols: list[str] | None = None):
        seeded = await super().orders(status=status, limit=limit, symbols=symbols)
        self._advance_all()
        working = [o for o in self.orders_by_id.values()
                   if o.status not in ("filled", "canceled") and o.id not in self.cancelled]
        return (seeded + working)[:limit]

    async def get_order(self, order_id: str) -> OrderSnapshot:
        order = await super().get_order(order_id)
        return self._advance(order)

    async def cancel_order(self, order_id: str) -> None:
        await super().cancel_order(order_id)
        order = self.orders_by_id.get(order_id)
        if order is not None and order.id not in self._settled:
            self._store(order.model_copy(update={"status": "canceled", "updated_at": _stamp()}))
