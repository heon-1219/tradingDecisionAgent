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

import re
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# US-equity ticker: an alphanumeric first character, then up to 14 more of letters/digits/dot/
# hyphen (1-15 chars total), after upper-casing. Two things keep this safe as a URL path segment
# (see broker.py's `asset()`), neither alone sufficient:
#  - no "/" (also keeps crypto pairs like "BTC/USD" out of scope for v1);
#  - a mandatory leading alphanumeric, which is what actually rules out "." and "..": disallowing
#    "/" does NOT stop a bare "." or ".." from being path-injected, since those resolve via
#    ordinary URL dot-segment normalization (`/v2/assets/.` -> `/v2/assets`, `/v2/assets/..` ->
#    `/v2`) without needing any "/" of their own inside the symbol string.
# Matched with fullmatch (see normalize_symbol), not match/search: a plain re.match(pattern + "$")
# would let a trailing "\n" through, since (outside MULTILINE) "$" is satisfied either at the
# absolute end of the string or just before a trailing newline.
_SYMBOL_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,14}")


def normalize_symbol(symbol: str) -> str:
    """Upper-case and strictly validate a ticker symbol. Raises `ValueError` for anything that
    doesn't fullmatch `_SYMBOL_RE` - including path-traversal attempts ("../positions", "a/b",
    bare "." / ".."), percent-encoded tricks ("%2e%2e"), and a trailing newline - since callers
    may use the result to build a URL path segment (`broker.asset`, `marketdata.bars`) as well as
    an order body field."""
    upper = (symbol or "").upper()
    if not _SYMBOL_RE.fullmatch(upper):
        raise ValueError(f"invalid symbol: {symbol!r}")
    return upper


def _reject_float(v: Any) -> Any:
    """Before-validator for money fields: `float` is never accepted (money is `Decimal`), but
    `Decimal`, `int` and `str` all pass through unchanged for pydantic's own Decimal coercion."""
    if isinstance(v, float):
        raise ValueError("float is not allowed for a money field; use Decimal, int, or str.")
    return v


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
    it parses, so unknown fields are rejected rather than ignored.

    Money fields never accept a raw `float` (`_reject_float`): only `Decimal`, `int` or `str`, so
    a caller can't smuggle in binary floating-point imprecision. `limit_price` must be present
    (and positive) exactly when `type == "limit"`, absent for `type == "market"`; a `notional`
    (fractional/dollar-based) order can only be a `market` `day` order, matching Alpaca's own
    rule that notional orders don't support `limit` or `gtc`.
    """

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
        return normalize_symbol(v)

    @field_validator("qty", "notional", "limit_price", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)

    @model_validator(mode="after")
    def _validate_quantity_and_price(self) -> "OrderLeg":
        if (self.qty is None) == (self.notional is None):
            raise ValueError("Exactly one of qty or notional must be set.")
        if self.qty is not None and self.qty <= 0:
            raise ValueError("qty must be positive.")
        if self.notional is not None:
            if self.notional <= 0:
                raise ValueError("notional must be positive.")
            if self.type == "limit":
                raise ValueError("a notional order cannot be a limit order.")
            if self.time_in_force == "gtc":
                raise ValueError("a notional order cannot use time_in_force=gtc.")
        if self.type == "limit":
            if self.limit_price is None:
                raise ValueError("limit_price is required for a limit order.")
            if self.limit_price <= 0:
                raise ValueError("limit_price must be positive.")
        elif self.limit_price is not None:
            raise ValueError("limit_price is only allowed for a limit order.")
        return self
