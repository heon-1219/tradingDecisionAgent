"""Snapshot models for the Alpaca paper-trading broker and market-data connectors.

These are read/parse-only snapshots of Alpaca API responses (account, positions, orders, assets,
clock, quotes, bars) plus `OrderLeg`, the one write-shaped model needed by `broker.submit_order`.
Money fields are `Decimal` (Alpaca sends numbers as strings on the trading API; market-data
endpoints send JSON floats, which pydantic converts via `Decimal(str(x))` rather than the binary
float value, so no precision is lost either way) - per the project's "money is never float"
constraint.

Every snapshot model uses `extra="ignore"` so a field Alpaca adds later never breaks parsing, and
none of them declare the pattern-day-trader fields Alpaca has since removed from the account
endpoint - they are simply absent from `AccountSnapshot`, so a payload that omits them (or, for
forward compatibility, one that still includes them) parses the same way either way.

Task 2.1 extends this file with `Mandate`, `RiskReason`/`RiskDecision`, `Ticket` and the trading
enums - keep additions here tidy so that later diff stays reviewable.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _AlpacaSnapshot(BaseModel):
    """Base for models parsed from an Alpaca response: unknown fields are dropped rather than
    raising, and a field with an alias (e.g. `cls` <- Alpaca's `class`) can still be constructed
    by its Python name."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class AccountSnapshot(_AlpacaSnapshot):
    status: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    non_marginable_buying_power: Decimal
    last_equity: Decimal
    trading_blocked: bool
    account_blocked: bool
    trade_suspended_by_user: bool
    shorting_enabled: bool


class PositionSnapshot(_AlpacaSnapshot):
    symbol: str
    qty: Decimal
    qty_available: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: Decimal
    side: str


class OrderSnapshot(_AlpacaSnapshot):
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: Decimal | None = None
    notional: Decimal | None = None
    type: str
    time_in_force: str
    limit_price: Decimal | None = None
    status: str
    filled_qty: Decimal | None = None
    filled_avg_price: Decimal | None = None
    submitted_at: str | None = None
    updated_at: str | None = None


class AssetInfo(_AlpacaSnapshot):
    symbol: str
    cls: str = Field(alias="class")
    status: str
    tradable: bool
    fractionable: bool
    shortable: bool
    marginable: bool
    borrow_status: str | None = None


class ClockInfo(_AlpacaSnapshot):
    is_open: bool
    next_open: str
    next_close: str
    timestamp: str


class Quote(BaseModel):
    """A best bid/ask/last snapshot from AlpacaData.snapshots."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    as_of: str | None = None
    feed: str


class Bar(BaseModel):
    """One OHLCV bar from AlpacaData.bars."""

    model_config = ConfigDict(extra="ignore")

    t: str
    o: Decimal
    h: Decimal
    l: Decimal  # noqa: E741 - matches Alpaca's own bar field name
    c: Decimal
    v: Decimal


class OrderLeg(BaseModel):
    """One order to submit via `AlpacaPaper.submit_order`. Exactly one of `qty`/`notional` must be
    given, and it must be positive - this is the shape the broker sends to Alpaca, not a response
    it parses, so unknown fields are rejected rather than ignored."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: Literal["buy", "sell"]
    qty: Decimal | None = None
    notional: Decimal | None = None
    type: Literal["market", "limit"] = "limit"
    time_in_force: Literal["day", "gtc"] = "day"
    limit_price: Decimal | None = None

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _exactly_one_positive_quantity(self) -> "OrderLeg":
        if (self.qty is None) == (self.notional is None):
            raise ValueError("Exactly one of qty or notional must be set.")
        if self.qty is not None and self.qty <= 0:
            raise ValueError("qty must be positive.")
        if self.notional is not None and self.notional <= 0:
            raise ValueError("notional must be positive.")
        return self
