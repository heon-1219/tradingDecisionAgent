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
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

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


class Mandate(BaseModel):
    """The user-approved trading constraints in effect for a session - everything downstream
    (risk rules, ticket proposals) is bounded by these fields, never by a value the model invents.

    `allow_symbols`/`deny_symbols` are normalized the same way `OrderLeg.symbol` is (upper-cased,
    validated by `normalize_symbol`, deduplicated while preserving first-seen order), since they
    gate exactly those symbols. Percent fields are bounded to `(0, 100]`, except `cash_buffer_pct`
    which may also be `0`. `budget_usd`/`max_order_usd` must be positive when set (`None` means
    "no cap" and is left alone). `order_types` must be non-empty.

    Decimal fields already serialize as JSON strings by pydantic v2's own default (verified
    against the installed version) and round-trip back to the exact same `Decimal` via
    `model_validate_json` - the `field_serializer` below just pins that behavior explicitly rather
    than relying on it silently, per the project's "money is Decimal, never float" constraint.
    """

    model_config = ConfigDict(extra="forbid")

    direction_text: str = ""
    budget_usd: Decimal | None = Field(default=None, gt=0)
    universe: Literal["us_equity"] = "us_equity"
    allow_symbols: list[str] = []
    deny_symbols: list[str] = []
    max_position_pct: Decimal = Field(default=Decimal("20"), gt=0, le=100)
    max_positions: int = Field(default=8, ge=1)
    cash_buffer_pct: Decimal = Field(default=Decimal("5"), ge=0, le=100)
    horizon: str = "weeks to months"
    order_types: list[Literal["market", "limit"]] = ["limit", "market"]
    max_orders_per_day: int = Field(default=10, ge=1)
    max_order_usd: Decimal | None = Field(default=None, gt=0)
    price_band_pct: Decimal = Field(default=Decimal("3"), gt=0, le=100)
    allow_fractional: bool = True
    version: int = Field(default=1, ge=1)

    @field_validator(
        "budget_usd",
        "max_position_pct",
        "cash_buffer_pct",
        "max_order_usd",
        "price_band_pct",
        mode="before",
    )
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)

    @field_validator("allow_symbols", "deny_symbols", mode="before")
    @classmethod
    def _normalize_symbol_list(cls, v: Any) -> Any:
        if v is None:
            return v
        out: list[str] = []
        for s in v:
            norm = normalize_symbol(s)
            if norm not in out:
                out.append(norm)
        return out

    @field_validator("order_types")
    @classmethod
    def _non_empty_order_types(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("order_types must not be empty.")
        return v

    @field_serializer(
        "budget_usd",
        "max_position_pct",
        "cash_buffer_pct",
        "max_order_usd",
        "price_band_pct",
        when_used="json",
    )
    def _serialize_decimal(self, v: Decimal | None) -> str | None:
        return None if v is None else str(v)


class RiskReason(BaseModel):
    """One reason attached to a `RiskDecision` - `severity` is purely informational (`"block"`
    doesn't imply the verdict was `"reject"`; the verdict is decided independently)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: Literal["info", "warn", "block"]


class RiskDecision(BaseModel):
    """The outcome of applying the risk rules to one proposed `OrderLeg`. `original` is the leg as
    proposed (by the LLM); `leg` is the same leg after risk has had a chance to clamp it (e.g. cut
    `qty` down to fit `max_position_pct`) - for `verdict in ("accept", "reject")` the two are
    identical, and `leg != original` is exactly the signal that a clamp happened."""

    model_config = ConfigDict(extra="forbid")

    index: int
    verdict: Literal["accept", "clamp", "reject"]
    leg: OrderLeg
    original: OrderLeg
    reasons: list[RiskReason] = []
    est_price: Decimal | None = None
    est_cost_usd: Decimal | None = None

    @field_validator("est_price", "est_cost_usd", mode="before")
    @classmethod
    def _no_float(cls, v: Any) -> Any:
        return _reject_float(v)


class MarketContext(BaseModel):
    """The market/account snapshot bundle risk rules and the LLM are given for one turn. `symbol`
    keys of `positions`, `assets` and `quotes` are normalized (`normalize_symbol`) so a lookup by a
    freshly-normalized `OrderLeg.symbol` always hits, regardless of how the caller happened to key
    the dict it passed in."""

    model_config = ConfigDict(extra="forbid")

    account: AccountSnapshot
    positions: dict[str, PositionSnapshot]
    open_orders: list[OrderSnapshot]
    assets: dict[str, AssetInfo]
    quotes: dict[str, Quote]
    clock: ClockInfo
    orders_today: int

    @field_validator("positions", "assets", "quotes", mode="before")
    @classmethod
    def _normalize_keys(cls, v: Any) -> Any:
        if v is None:
            return v
        return {normalize_symbol(k): val for k, val in v.items()}


class TicketStatus(str, Enum):
    """Lifecycle states for a `Ticket`, from proposal through terminal outcome."""

    proposed = "proposed"
    submitting = "submitting"
    submitted = "submitted"
    rejected_by_user = "rejected_by_user"
    expired = "expired"
    superseded = "superseded"
    failed = "failed"


class Ticket(BaseModel):
    """A proposed multi-leg order, pending user confirmation via `POST /api/tickets/{id}/confirm`
    (the only path that reaches `broker.submit_order` - see `global-constraints.md`).

    All three datetimes must be timezone-aware (a naive `datetime` is rejected outright, since
    comparing it against an aware clock elsewhere would raise or silently misbehave). `hash` is a
    64-char lowercase hex digest (e.g. sha256) used to detect a resubmission of the same proposal.
    `client_order_ids` must have exactly one id per leg (it is generated at proposal time, before
    any broker call, so the count is always known); `broker_order_ids` starts empty (nothing
    submitted yet) and, once populated, must also have exactly one entry per leg - `None` for a leg
    whose submission hasn't completed (or reconciled) yet is a valid entry.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    status: TicketStatus
    legs: list[OrderLeg]
    decisions: list[RiskDecision]
    thesis: str
    risks: str
    invalidation: str
    evidence_ids: list[str]
    mandate_version: int
    quote_as_of: datetime
    created_at: datetime
    expires_at: datetime
    hash: str
    client_order_ids: list[str]
    broker_order_ids: list[str | None] = []

    @field_validator("quote_as_of", "created_at", "expires_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("datetime must be timezone-aware.")
        return v

    @field_validator("hash")
    @classmethod
    def _valid_hash(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("hash must be a 64-character lowercase hex string.")
        return v

    @model_validator(mode="after")
    def _validate_id_lengths(self) -> "Ticket":
        if len(self.client_order_ids) != len(self.legs):
            raise ValueError("client_order_ids must have exactly one entry per leg.")
        if self.broker_order_ids and len(self.broker_order_ids) != len(self.legs):
            raise ValueError("broker_order_ids, when non-empty, must have exactly one entry per leg.")
        return self
