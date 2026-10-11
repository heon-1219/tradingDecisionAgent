"""Deterministic pre-trade risk rules: the LLM proposes order legs, this code decides.

`evaluate` checks each proposed `OrderLeg` against the account/market snapshot (`MarketContext`)
and the user's `Mandate`, and returns one `RiskDecision` per leg: `accept`, `clamp` (the leg was
shrunk or re-priced to fit a limit - `leg` holds the changed copy) or `reject`. `overall` rolls
those up into the single status a ticket shows.

Pure and deterministic: no I/O, no clock reads, no randomness, and the inputs are never mutated
or aliased (every returned leg is a copy). All arithmetic runs in a private decimal context, so the
caller's ambient `decimal` settings cannot change a result. Money is `Decimal`, never `float`.

Per leg, the rules run in `_RULES` order (the plan's reason-code table, mirrored by
`REASON_CODES`) and stop at the first blocking reason; warnings and clamps found before that stay
on the decision. Legs are evaluated in list order, and every accepted or clamped leg carries its
effects into the legs after it (`_Book`), always in the conservative direction:
- buys consume buying power and budget and add to the symbol's position value, but never make
  shares sellable (an unfilled buy is not a position);
- sells use up qty available to sell, but never free buying power, budget or position value;
- a later leg on the opposite side of a symbol already traded in the ticket is a `wash_trade`;
- each leg counts towards `max_orders_per_day`, and a buy into a symbol not yet held (or on
  order) opens a position for `max_positions`.
Rejected legs have no effect. Pending open BUY orders in `ctx.open_orders` count as position value,
as invested budget, as cash already committed and as held symbols (remaining qty x limit price,
else x the reference price; or the unfilled notional).

Prices and estimates:
- reference price: the last trade, else the bid/ask midpoint (both sides needed); a missing or
  non-positive value counts as absent, and no reference at all is `no_quote`.
- `est_price`: the limit price if set; else, for a buy, max(ask, reference) x
  (1 + MARKET_SLIPPAGE_PCT/100), so a stale low ask can't understate the cost; for a sell, the
  bid, falling back to the reference price.
- `est_cost_usd`: est_price x qty, or the notional. Every leg must come to at least $1.00.
- A notional sell counts as notional / min(bid, reference) shares, never fewer.
- Buying power is cash only, never margin. Usable = min(buying_power, non_marginable_buying_power,
  cash - pending open buys) - cash_buffer_pct% of equity - buys accepted earlier in the ticket.
  Alpaca's `cash` is not reduced by open orders, so the pending buys come off it here; and on a
  margin account holding positions `non_marginable_buying_power` can exceed the cash, so it is
  never trusted alone. The buffer is a share of equity, as the mandate's label says ("% of equity
  always kept in cash"). Nothing left (<= 0) rejects every buy; a pending open buy that can't be
  priced rejects every buy too (fail closed: the cash it commits is unknown).
- Rejected legs report `leg == original` and the estimates of the original leg.

Clamps: a qty clamp floors to whole shares unless fractional trading is allowed for the leg
(asset fractionable, `mandate.allow_fractional`, time_in_force "day"), in which case it floors to
9 decimals; a notional clamp floors to cents. A clamp (or tick rounding) that leaves nothing, or
less than $1.00, rejects the leg with that rule's own code. The $1.00 floor never blocks a sell of
the whole working position, so a sub-$1 residual can always be closed. Accepted and clamped legs
come back with plain-form Decimals (qty without exponent or trailing zeros, notional in cents,
limit price at its tick precision); a leg with more than MAX_LEG_DIGITS digits is rejected
(`precision`) up front, so no value is ever rounded silently.

Reason messages reach the model. A string they quote from the broker (an account or asset status,
the asset class, an open order's id, the next market open) is first reduced to one safe token
(`_token`: letters, digits and `_ . : + -`, at most 32 characters), so it cannot carry an
instruction or markup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Callable, Literal

from pydantic import ValidationError

from finagent.trading.models import (
    AssetInfo,
    Mandate,
    MarketContext,
    OrderLeg,
    OrderSnapshot,
    Quote,
    RiskDecision,
    RiskReason,
)

MARKET_SLIPPAGE_PCT = Decimal("1.0")  # buffer on ask for market-buy cost estimates
LARGE_ORDER_PCT = Decimal("10")  # warn when one leg's estimated cost exceeds this % of equity
MIN_NOTIONAL_USD = Decimal("1.00")  # a notional clamp may not go below Alpaca's $1 minimum
FRACTIONAL_DECIMALS = 9  # Alpaca accepts fractional qty to 9 decimal places
# Most digits (in plain positional form) a leg's qty / notional / limit_price may have. Half the
# 40-digit working precision: a leg value times a price (quotes carry <= ~17 digits) stays exact,
# and tick/cent quantizing never overflows. 20 digits is far beyond any real order.
MAX_LEG_DIGITS = 20

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_CENT = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")
_QTY_STEP = Decimal(1).scaleb(-FRACTIONAL_DECIMALS)

# Fixed, private arithmetic context: independent of whatever the caller's thread has set, and loud
# (trapping) rather than silently producing NaN/Infinity.
_CTX = Context(
    prec=40, rounding=ROUND_HALF_EVEN, traps=[InvalidOperation, DivisionByZero, Overflow]
)


# --- price helpers ------------------------------------------------------------------------------


def _positive(value: Decimal | None) -> Decimal | None:
    return value if value is not None and value > 0 else None


def _reference(quote: Quote | None) -> Decimal | None:
    """Last trade, else the bid/ask midpoint; None when neither is available."""
    if quote is None:
        return None
    last = _positive(quote.last)
    if last is not None:
        return last
    bid, ask = _positive(quote.bid), _positive(quote.ask)
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return None


def _market_buy_price(quote: Quote | None, ref: Decimal | None) -> Decimal | None:
    """Per-share estimate for a buy with no limit price: max(ask, reference) x (1 + slippage), so a
    stale low ask can't understate it. Used for new legs and for pending open buys alike."""
    ask = _positive(quote.ask) if quote is not None else None
    known = [p for p in (ask, ref) if p is not None]
    return max(known) * (1 + MARKET_SLIPPAGE_PCT / _HUNDRED) if known else None


def _est_price(leg: OrderLeg, quote: Quote | None, ref: Decimal | None) -> Decimal | None:
    if leg.limit_price is not None:
        return leg.limit_price
    if leg.side == "buy":
        return _market_buy_price(quote, ref)
    bid = _positive(quote.bid) if quote is not None else None
    return bid if bid is not None else ref


def _shares_sold(leg: OrderLeg, quote: Quote | None, ref: Decimal | None) -> Decimal | None:
    """Shares a sell leg takes out of the position. A notional sell converts at the LOWER of the
    bid and the reference price (whichever exists), so a stale high bid can't understate it.
    None only for a notional with no price at all (which `_no_quote` rejects)."""
    if leg.qty is not None:
        return leg.qty
    bid = _positive(quote.bid) if quote is not None else None
    known = [p for p in (bid, ref) if p is not None]
    return leg.notional / min(known) if known else None


def _est_cost(leg: OrderLeg, price: Decimal | None) -> Decimal | None:
    if leg.notional is not None:
        return leg.notional
    if price is None or leg.qty is None:
        return None
    return price * leg.qty


def _tick_for(price: Decimal) -> Decimal:
    return _CENT if price >= 1 else _SUB_PENNY


def _round_to_tick(price: Decimal, side: str) -> Decimal:
    """Buys round down, sells round up - never a worse price than the one proposed."""
    rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
    rounded = price.quantize(_tick_for(price), rounding=rounding)
    # A sub-dollar sell can round up to exactly 1.0000; put it on the $0.01 grid (1.00). Exact:
    # anything >= 1 here is already a whole number of cents.
    return rounded.quantize(_CENT) if rounded >= 1 else rounded


def _plain(value: Decimal) -> Decimal:
    """Strip trailing zeros without ever producing exponent notation (1E+3 -> 1000)."""
    normalized = value.normalize()
    return normalized.quantize(_ONE) if normalized.as_tuple().exponent > 0 else normalized


def _fit_qty(budget: Decimal, price: Decimal, whole: bool) -> Decimal:
    """The largest qty with price x qty <= budget, floored to whole shares or 9 decimals. Uses
    exact integer division (`//`), so the result never overshoots the budget through rounding."""
    if budget <= 0 or price <= 0:
        return _ZERO
    if whole:
        return _plain(budget // price)
    return _plain((budget.scaleb(FRACTIONAL_DECIMALS) // price).scaleb(-FRACTIONAL_DECIMALS))


def _is_fractional(qty: Decimal) -> bool:
    return qty != qty.to_integral_value()


def _plain_digits(value: Decimal) -> int:
    """Digits needed to write `value` in plain positional form, trailing zeros ignored
    (123.45 -> 5, 0.001 -> 3, 1E+3 -> 4). Read off the exact tuple, so nothing is rounded."""
    _, digits, exponent = value.as_tuple()
    significant = list(digits)
    while len(significant) > 1 and significant[-1] == 0:
        significant.pop()
        exponent += 1
    if exponent >= 0:
        return len(significant) + exponent
    return max(len(significant), -exponent)


def _open_buy_value(order: OrderSnapshot, ctx: MarketContext) -> Decimal | None:
    """What a pending open buy order still commits: the unfilled notional, or the unfilled qty x
    its limit price. With no limit price (a market order, or a stop order - stop prices aren't
    modeled) the unfilled qty is valued like a new market buy, `_market_buy_price`, with the
    position's current price standing in for a missing reference. None when it can't be priced."""
    filled = order.filled_qty or _ZERO
    if order.notional is not None:
        return max(order.notional - filled * (order.filled_avg_price or _ZERO), _ZERO)
    remaining = max((order.qty or _ZERO) - filled, _ZERO)
    limit = _positive(order.limit_price)
    if limit is not None:
        return remaining * limit
    symbol = order.symbol.upper()
    quote = ctx.quotes.get(symbol)
    ref = _reference(quote)
    position = ctx.positions.get(symbol)
    if ref is None and position is not None:
        ref = _positive(position.current_price)
    price = _market_buy_price(quote, ref)
    if price is None:
        return _ZERO if remaining == 0 else None
    return remaining * price


def _normalized(leg: OrderLeg) -> OrderLeg:
    """The same leg with plain-form Decimals, so a ticket serializes cleanly (5E+2 -> 500.00).
    Values never change - by now qty has at most 9 decimals and MAX_LEG_DIGITS digits, the
    notional is in cents and the limit price sits on its tick - and `Inexact` is trapped here so
    a slip could only ever raise, never round. (`Rounded` is not trapped: dropping trailing zeros,
    100.000 -> 100.00, signals it without changing the value.)"""
    update: dict[str, Decimal] = {}
    with localcontext() as exact:
        exact.traps[Inexact] = True
        if leg.qty is not None:
            update["qty"] = _plain(leg.qty)
        if leg.notional is not None:
            update["notional"] = leg.notional.quantize(_CENT)
        if leg.limit_price is not None:
            update["limit_price"] = leg.limit_price.quantize(_tick_for(leg.limit_price))
    return leg.model_copy(update=update)


# --- message formatting (Decimal only, never float) ---------------------------------------------


def _usd(value: Decimal) -> str:
    q = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    return f"-${-q:,.2f}" if q < 0 else f"${q:,.2f}"


def _price(value: Decimal) -> str:
    """Like `_usd`, but a price's sub-penny digits are kept ($0.4999, $500.009)."""
    plain = _plain(value)
    if plain.as_tuple().exponent >= -2:
        return _usd(value)
    return f"${plain:,f}"


def _qty(value: Decimal) -> str:
    return f"{_plain(value.quantize(_QTY_STEP, rounding=ROUND_HALF_UP)):,f}"


def _pct(value: Decimal) -> str:
    return f"{_plain(value):f}%"


_TOKEN_STRIP_RE = re.compile(r"[^A-Za-z0-9_.:+\-]")
_TOKEN_MAX_CHARS = 32


def _token(value: object) -> str:
    """A provider string (account or asset status, asset class, order id, clock time) reduced to
    one safe token before it goes into a reason message the model reads: only letters, digits and
    `_ . : + -`, at most 32 characters, "unknown" when nothing is left. Alpaca's own values
    ("ACTIVE", "us_equity", a UUID, an RFC 3339 time) pass unchanged."""
    return _TOKEN_STRIP_RE.sub("", str(value if value is not None else ""))[:_TOKEN_MAX_CHARS] or "unknown"


# --- working state -------------------------------------------------------------------------------


@dataclass
class _Book:
    """The working account state: the snapshot plus pending open buy orders, updated by each leg
    accepted so far in this `evaluate` call. Every update is in the conservative direction."""

    bought: Decimal  # est. cost of buys accepted earlier in this ticket (buying power and budget)
    orders: int  # legs accepted earlier in this ticket
    held: set[str]  # symbols held, on order (pending open buy), or bought earlier in the ticket
    qty_available: dict[str, Decimal]  # shares available to sell; only sells change it
    value: dict[str, Decimal]  # position + pending open buys + earlier buys, per symbol
    invested: Decimal  # all positions' value + all pending open buys (budget), before this ticket
    pending_buys: Decimal  # what the priced pending open buys commit (comes out of cash)
    unpriced: tuple[str, ...]  # symbols with a pending open buy that can't be valued
    sides: dict[str, set[str]]  # sides already traded in this ticket, per symbol

    @classmethod
    def from_ctx(cls, ctx: MarketContext) -> "_Book":
        held = {s for s, p in ctx.positions.items() if p.qty != 0}
        value = {s: abs(p.market_value) for s, p in ctx.positions.items()}
        invested = sum(value.values(), _ZERO)
        pending_buys = _ZERO
        unpriced: list[str] = []
        for order in ctx.open_orders:
            if order.side.lower() != "buy":
                continue
            symbol = order.symbol.upper()
            held.add(symbol)
            pending = _open_buy_value(order, ctx)
            if pending is None:
                unpriced.append(symbol)
                continue
            value[symbol] = value.get(symbol, _ZERO) + pending
            invested += pending
            pending_buys += pending
        return cls(
            bought=_ZERO,
            orders=0,
            held=held,
            qty_available={s: p.qty_available for s, p in ctx.positions.items()},
            value=value,
            invested=invested,
            pending_buys=pending_buys,
            unpriced=tuple(unpriced),
            sides={},
        )

    def commit(self, leg: OrderLeg, cost: Decimal, sold: Decimal | None) -> None:
        """Carry an accepted leg forward; `sold` is `_shares_sold` for a sell, None for a buy."""
        symbol = leg.symbol
        self.orders += 1
        self.sides.setdefault(symbol, set()).add(leg.side)
        if leg.side == "buy":
            self.bought += cost
            self.held.add(symbol)
            self.value[symbol] = self.value.get(symbol, _ZERO) + cost
        else:
            self.qty_available[symbol] = self.qty_available.get(symbol, _ZERO) - sold


@dataclass
class _Check:
    """One leg under evaluation. `leg` is the working copy: clamps replace it, never mutate it."""

    ctx: MarketContext
    mandate: Mandate
    book: _Book
    leg: OrderLeg
    asset: AssetInfo | None
    quote: Quote | None
    ref: Decimal | None
    reasons: list[RiskReason] = field(default_factory=list)
    blocked: bool = False

    @property
    def whole_shares(self) -> bool:
        asset = self.asset
        fractional_ok = (
            asset is not None
            and asset.fractionable
            and self.mandate.allow_fractional
            and self.leg.time_in_force == "day"
        )
        return not fractional_ok

    @property
    def est_price(self) -> Decimal | None:
        return _est_price(self.leg, self.quote, self.ref)

    @property
    def est_cost(self) -> Decimal | None:
        return _est_cost(self.leg, self.est_price)

    @property
    def closes_position(self) -> bool:
        """A sell of everything still available to sell in the symbol (more than that is a
        `_short_sell`). Exempt from the $1.00 floor, so a sub-$1 residual - which fractional clamps
        can leave behind - can always be sold."""
        leg = self.leg
        if leg.side != "sell":
            return False
        have = self.book.qty_available.get(leg.symbol, _ZERO)
        sold = _shares_sold(leg, self.quote, self.ref)
        return have > 0 and sold is not None and sold >= have

    def block(self, code: str, message: str) -> None:
        self.reasons.append(RiskReason(code=code, message=message, severity="block"))
        self.blocked = True

    def warn(self, code: str, message: str) -> None:
        self.reasons.append(RiskReason(code=code, message=message, severity="warn"))


def _shrink(c: _Check, code: str, budget: Decimal, why: str) -> None:
    """Shrink the working leg so its estimated cost fits `budget` (a clamp, severity warn), or
    block with `code` when nothing fits. `why` is the lower-case explanation of the overshoot."""
    leg = c.leg
    because = why[0].upper() + why[1:]  # the same explanation, as a sentence of its own
    if leg.notional is not None:
        fits = max(min(budget, leg.notional), _ZERO)
        new_notional = fits.quantize(_CENT, rounding=ROUND_FLOOR)
        if new_notional < MIN_NOTIONAL_USD:
            c.block(code, f"{because}, and the clamped notional {_usd(new_notional)} would be "
                          f"below the {_usd(MIN_NOTIONAL_USD)} minimum")
            return
        c.leg = leg.model_copy(update={"notional": new_notional})
        c.warn(code, f"Clamped notional {_usd(leg.notional)} → {_usd(new_notional)}: {why}")
        return
    whole, price = c.whole_shares, c.est_price
    new_qty = min(_fit_qty(budget, price, whole), leg.qty)
    if new_qty <= 0:
        kind = "whole-share quantity" if whole else "quantity"
        c.block(code, f"{because}, and no {kind} fits")
        return
    if new_qty * price < MIN_NOTIONAL_USD:
        c.block(code, f"{because}, and the clamped qty {_qty(new_qty)} would cost "
                      f"{_price(new_qty * price)}, below the {_usd(MIN_NOTIONAL_USD)} minimum")
        return
    c.leg = leg.model_copy(update={"qty": new_qty})
    c.warn(code, f"Clamped qty {_qty(leg.qty)} → {_qty(new_qty)}: {why}")


# --- rules, in table order ----------------------------------------------------------------------


def _account_blocked(c: _Check) -> None:
    account = c.ctx.account
    flags = [
        name
        for name in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
        if getattr(account, name)
    ]
    if account.status.upper() != "ACTIVE":
        flags.insert(0, f"status {_token(account.status)}")
    if flags:
        c.block("account_blocked", f"The account cannot trade right now ({', '.join(flags)})")


def _asset_not_tradable(c: _Check) -> None:
    symbol, asset = c.leg.symbol, c.asset
    if asset is None:
        c.block("asset_not_tradable", f"No asset information for {symbol}, so it cannot be traded")
        return
    problems = []
    if not asset.tradable:
        problems.append("not tradable")
    if asset.status.lower() != "active":
        problems.append(f"status {_token(asset.status)}")
    if asset.cls.lower() != "us_equity":
        problems.append(f"asset class {_token(asset.cls)}, not us_equity")
    if problems:
        c.block("asset_not_tradable", f"{symbol} cannot be traded: {', '.join(problems)}")


def _denied_symbol(c: _Check) -> None:
    symbol, mandate = c.leg.symbol, c.mandate
    if symbol in mandate.deny_symbols:
        c.block("denied_symbol", f"{symbol} is on the mandate's deny list")
    elif mandate.allow_symbols and symbol not in mandate.allow_symbols:
        c.block(
            "denied_symbol",
            f"{symbol} is not on the mandate's allow list ({', '.join(mandate.allow_symbols)})",
        )


def _order_type_not_allowed(c: _Check) -> None:
    order_type = c.leg.type
    if order_type not in c.mandate.order_types:
        allowed = ", ".join(c.mandate.order_types)
        c.block(
            "order_type_not_allowed",
            f"{order_type.capitalize()} orders are not allowed by the mandate (allowed: {allowed})",
        )


def _precision(c: _Check) -> None:
    """Runs before any rule does arithmetic on the leg: a value with more than MAX_LEG_DIGITS
    digits would otherwise be rounded silently by the working context (e.g. 0.999...9 with 41
    nines normalizes UP to 1)."""
    leg = c.leg
    too_long = [
        f"{name} {value:f}"
        for name, value in (("qty", leg.qty), ("notional", leg.notional),
                            ("limit_price", leg.limit_price))
        if value is not None and _plain_digits(value) > MAX_LEG_DIGITS
    ]
    if too_long:
        c.block(
            "precision",
            f"{'; '.join(too_long)}: more than {MAX_LEG_DIGITS} digits, too precise to check "
            "exactly",
        )


def _notional_needs_market_day(c: _Check) -> None:
    # OrderLeg's own validation already forbids this; kept as defense in depth for legs that
    # bypassed it (model_construct / unvalidated assignment).
    leg = c.leg
    if leg.notional is not None and (leg.type != "market" or leg.time_in_force != "day"):
        c.block(
            "notional_needs_market_day",
            f"A notional (dollar) order must be a market day order, not {leg.type} "
            f"{leg.time_in_force}",
        )


def _notional_invalid(c: _Check) -> None:
    """Order size: a notional must be at least $1.00 in whole cents, and so must any leg's
    estimated cost (Alpaca rejects sub-$1 orders). A qty leg with no estimate yet has no quote,
    which `_no_quote` blocks. Later price/qty changes keep the $1.00 floor themselves: `_bad_tick`
    and `_shrink` reject under their own codes. A sell of the whole working position is exempt
    from the $1.00 floor (not from whole cents)."""
    leg = c.leg
    notional = leg.notional
    closing = c.closes_position
    if notional is not None:
        if notional < MIN_NOTIONAL_USD and not closing:
            c.block(
                "notional_invalid",
                f"Notional {_price(notional)} is below the {_usd(MIN_NOTIONAL_USD)} minimum",
            )
        elif notional.normalize().as_tuple().exponent < -2:
            c.block(
                "notional_invalid", f"Notional {_price(notional)} has more than 2 decimal places"
            )
        return
    cost = c.est_cost
    if cost is not None and cost < MIN_NOTIONAL_USD and not closing:
        c.block(
            "notional_invalid",
            f"Estimated cost {_price(cost)} of {_qty(leg.qty)} {leg.symbol} is below the "
            f"{_usd(MIN_NOTIONAL_USD)} minimum order size",
        )


def _fractional_not_allowed(c: _Check) -> None:
    leg, asset, mandate = c.leg, c.asset, c.mandate
    if leg.notional is not None:
        what = "Notional (dollar) orders trade fractional shares and are"
    elif leg.qty is not None and _is_fractional(leg.qty):
        what = f"Fractional qty {_qty(leg.qty)} is"
    else:
        return
    why = []
    if asset is not None and not asset.fractionable:
        why.append(f"{leg.symbol} is not fractionable")
    if not mandate.allow_fractional:
        why.append("the mandate does not allow fractional shares")
    if leg.time_in_force != "day":
        why.append(f"time_in_force is {leg.time_in_force}, and fractional orders must be day")
    if why:
        c.block("fractional_not_allowed", f"{what} not allowed: {'; '.join(why)}")
    elif leg.qty is not None and leg.qty.normalize().as_tuple().exponent < -FRACTIONAL_DECIMALS:
        c.block(
            "fractional_not_allowed",
            f"Qty {leg.qty:f} has more than {FRACTIONAL_DECIMALS} decimal places",
        )


def _limit_required(c: _Check) -> None:
    # OrderLeg's own validation already forbids this; defense in depth, as above.
    if c.leg.type == "limit" and c.leg.limit_price is None:
        c.block("limit_required", "A limit order needs a limit_price")


def _bad_tick(c: _Check) -> None:
    leg = c.leg
    price = leg.limit_price
    if price is None:
        return
    tick = _tick_for(price)
    rounded = _round_to_tick(price, leg.side)
    if rounded <= 0:
        c.block(
            "bad_tick",
            f"Limit price {_price(price)} rounds to {_price(rounded)} at the {_price(tick)} tick; "
            "a limit price must be positive",
        )
        return
    if rounded != price:
        direction = "down" if leg.side == "buy" else "up"
        cost = rounded * leg.qty if leg.qty is not None else None
        if cost is not None and cost < MIN_NOTIONAL_USD and not c.closes_position:
            c.block(
                "bad_tick",
                f"Rounding the limit price {_price(price)} {direction} to {_price(rounded)} "
                f"would make the estimated cost {_price(cost)}, below the "
                f"{_usd(MIN_NOTIONAL_USD)} minimum order size",
            )
            return
        c.leg = leg.model_copy(update={"limit_price": rounded})
        c.warn(
            "bad_tick",
            f"Rounded limit price {_price(price)} → {_price(rounded)} "
            f"({leg.side}: {direction} to the {_price(tick)} tick)",
        )


def _price_band(c: _Check) -> None:
    limit, ref = c.leg.limit_price, c.ref
    if limit is None or ref is None:
        return  # no reference: `_no_quote` blocks next
    deviation = abs(limit - ref) / ref * _HUNDRED
    band = c.mandate.price_band_pct
    if deviation > band:
        shown = deviation.quantize(_CENT, rounding=ROUND_UP)  # never shows as inside the band
        c.block(
            "price_band",
            f"Limit {_price(limit)} is {shown}% from the reference price {_price(ref)} "
            f"(band {_pct(band)})",
        )


def _no_quote(c: _Check) -> None:
    if c.ref is None:
        c.block(
            "no_quote",
            f"No usable quote for {c.leg.symbol} (no last trade and no bid/ask), so the order "
            "cannot be priced",
        )


def _short_sell(c: _Check) -> None:
    leg = c.leg
    if leg.side != "sell":
        return
    symbol = leg.symbol
    have = c.book.qty_available.get(symbol, _ZERO)
    want = _shares_sold(leg, c.quote, c.ref)  # `_no_quote` has passed, so a price exists
    if want <= have:
        return
    selling = f"{_qty(want)} {symbol}" if leg.qty is not None else (
        f"{_usd(leg.notional)} of {symbol} (about {_qty(want)} shares)"
    )
    if symbol in c.book.qty_available:
        c.block(
            "short_sell",
            f"Selling {selling} exceeds the {_qty(have)} {symbol} available to sell; short "
            "selling is not allowed",
        )
    else:
        c.block(
            "short_sell",
            f"Selling {selling} with no {symbol} position would be a short sale, which is not "
            "allowed",
        )


def _wash_trade(c: _Check) -> None:
    leg = c.leg
    opposite = "sell" if leg.side == "buy" else "buy"
    for order in c.ctx.open_orders:
        if order.symbol.upper() == leg.symbol and order.side.lower() == opposite:
            c.block(
                "wash_trade",
                f"An open {opposite} order for {leg.symbol} (id {_token(order.id)}) is on the opposite "
                "side; this order could trade against it (potential wash trade)",
            )
            return
    if opposite in c.book.sides.get(leg.symbol, ()):
        c.block(
            "wash_trade",
            f"An earlier {opposite} leg for {leg.symbol} in this ticket is on the opposite side; "
            "this order could trade against it (potential wash trade)",
        )


def _max_order_usd(c: _Check) -> None:
    cap = c.mandate.max_order_usd
    cost = c.est_cost
    if cap is not None and cost > cap:
        _shrink(c, "max_order_usd", cap,
                f"estimated cost {_usd(cost)} exceeds the mandate's max order {_usd(cap)}")


def _max_position_pct(c: _Check) -> None:
    leg, mandate, equity = c.leg, c.mandate, c.ctx.account.equity
    if leg.side != "buy":
        return
    budget = mandate.budget_usd
    if budget is not None and budget <= equity:
        base, basis = budget, "budget"
    else:
        base, basis = equity, "equity"
    cap = base * mandate.max_position_pct / _HUNDRED
    have = c.book.value.get(leg.symbol, _ZERO)
    cost = c.est_cost
    if have + cost > cap:
        _shrink(
            c,
            "max_position_pct",
            cap - have,
            f"{leg.symbol} position {_usd(have)} + estimated cost {_usd(cost)} exceeds "
            f"{_pct(mandate.max_position_pct)} of {basis} {_usd(base)} ({_usd(cap)})",
        )


def _max_positions(c: _Check) -> None:
    leg = c.leg
    if leg.side != "buy" or leg.symbol in c.book.held:
        return
    count, cap = len(c.book.held), c.mandate.max_positions
    if count >= cap:
        held = "1 position is" if count == 1 else f"{count} positions are"
        c.block(
            "max_positions",
            f"Buying {leg.symbol} would open a new position, but {held} already held or on "
            f"order (mandate max {cap})",
        )


def _buying_power(c: _Check) -> None:
    """Cash only, never margin: usable = min(buying_power, non_marginable_buying_power, cash -
    pending open buys) - cash_buffer_pct% of equity - earlier buys in this ticket (see the module
    docstring)."""
    if c.leg.side != "buy":
        return
    book = c.book
    if book.unpriced:
        symbols = ", ".join(book.unpriced)
        c.block(
            "buying_power",
            f"Cannot check buying power: a pending open buy order for {symbols} has no price and "
            "no quote",
        )
        return
    account = c.ctx.account
    available = min(account.buying_power, account.cash - book.pending_buys)
    if account.non_marginable_buying_power is not None:
        available = min(available, account.non_marginable_buying_power)
    pct = c.mandate.cash_buffer_pct
    buffer = max(account.equity, _ZERO) * pct / _HUNDRED  # kept in cash, a share of equity
    used = book.bought
    usable = available - buffer - used
    cost = c.est_cost
    if cost <= usable:
        return
    notes = []
    if book.pending_buys > 0:
        notes.append(f"{_usd(book.pending_buys)} in open buy orders")
    if used > 0:
        notes.append(f"{_usd(used)} committed by earlier legs")
    after = f" (after {' and '.join(notes)})" if notes else ""
    if usable <= 0:
        kept = f" once {_pct(pct)} of equity ({_usd(buffer)}) is kept in cash" if buffer > 0 else ""
        c.block(
            "buying_power",
            f"Estimated cost {_usd(cost)} exceeds usable buying power {_usd(_ZERO)}: nothing is "
            f"left{kept}{after}",
        )
        return
    _shrink(c, "buying_power", usable,
            f"estimated cost {_usd(cost)} exceeds usable buying power {_usd(usable)}{after}")


def _budget(c: _Check) -> None:
    budget = c.mandate.budget_usd
    if budget is None or c.leg.side != "buy":
        return
    if c.book.unpriced:
        symbols = ", ".join(c.book.unpriced)
        c.block(
            "budget",
            f"Cannot check the {_usd(budget)} budget: a pending open buy order for {symbols} has "
            "no price and no quote",
        )
        return
    invested = c.book.invested + c.book.bought
    cost = c.est_cost
    if invested + cost > budget:
        _shrink(
            c,
            "budget",
            budget - invested,
            f"invested {_usd(invested)} (positions, open buy orders and earlier legs) + estimated "
            f"cost {_usd(cost)} exceeds the mandate's budget {_usd(budget)}",
        )


def _orders_per_day(c: _Check) -> None:
    today, earlier, cap = c.ctx.orders_today, c.book.orders, c.mandate.max_orders_per_day
    if today + earlier >= cap:
        c.block(
            "orders_per_day",
            f"Daily order limit reached: {today} orders today + {earlier} earlier in this ticket "
            f"(mandate max {cap} per day)",
        )


def _market_closed(c: _Check) -> None:
    if not c.ctx.clock.is_open:
        c.warn(
            "market_closed",
            f"The market is closed; this order queues until the next open "
            f"({_token(c.ctx.clock.next_open)})",
        )


def _large_order(c: _Check) -> None:
    equity = c.ctx.account.equity
    cost = c.est_cost
    if cost > equity * LARGE_ORDER_PCT / _HUNDRED:
        c.warn(
            "large_order",
            f"Large order: estimated cost {_usd(cost)} is more than {_pct(LARGE_ORDER_PCT)} of "
            f"equity {_usd(equity)}",
        )


# Evaluation order = the plan's reason-code table, plus `precision` (fix round 2), which must run
# before the first rule that does arithmetic on the leg. Each rule is named after the code it emits.
_RULES: tuple[Callable[[_Check], None], ...] = (
    _account_blocked,
    _asset_not_tradable,
    _denied_symbol,
    _order_type_not_allowed,
    _precision,
    _notional_needs_market_day,
    _notional_invalid,
    _fractional_not_allowed,
    _limit_required,
    _bad_tick,
    _price_band,
    _no_quote,
    _short_sell,
    _wash_trade,
    _max_order_usd,
    _max_position_pct,
    _max_positions,
    _buying_power,
    _budget,
    _orders_per_day,
    _market_closed,
    _large_order,
)

# Every reason code `evaluate` can emit, in evaluation order.
REASON_CODES: tuple[str, ...] = tuple(rule.__name__.lstrip("_") for rule in _RULES)


# --- public API ---------------------------------------------------------------------------------


def evaluate(legs: list[OrderLeg], ctx: MarketContext, mandate: Mandate) -> list[RiskDecision]:
    """One `RiskDecision` per leg, in order; see the module docstring for the rules.

    Legs are expected to be validated `OrderLeg`s. One that bypassed validation is either rejected
    by a rule or makes this raise `pydantic.ValidationError` - it is never accepted or clamped."""
    with localcontext(_CTX):
        book = _Book.from_ctx(ctx)
        decisions: list[RiskDecision] = []
        for index, proposed in enumerate(legs):
            original = proposed.model_copy()
            quote = ctx.quotes.get(original.symbol)
            ref = _reference(quote)
            check = _Check(
                ctx=ctx,
                mandate=mandate,
                book=book,
                leg=original.model_copy(),
                asset=ctx.assets.get(original.symbol),
                quote=quote,
                ref=ref,
            )
            for rule in _RULES:
                rule(check)
                if check.blocked:
                    break

            if check.blocked:
                verdict, final = "reject", original.model_copy()
            else:
                final = _normalized(check.leg)
                verdict = "clamp" if final != original else "accept"  # compares values
            price = _est_price(final, quote, ref)
            cost = _est_cost(final, price)
            if verdict != "reject":
                sold = _shares_sold(final, quote, ref) if final.side == "sell" else None
                book.commit(final, cost, sold)
            decisions.append(_decision(index, verdict, final, original, check.reasons, price, cost))
        return decisions


def _decision(
    index: int,
    verdict: str,
    leg: OrderLeg,
    original: OrderLeg,
    reasons: list[RiskReason],
    price: Decimal | None,
    cost: Decimal | None,
) -> RiskDecision:
    """Build the decision, failing closed on a leg that bypassed `OrderLeg` validation.

    `OrderLeg`'s model validator re-runs on an instance embedded in a `RiskDecision`, so a leg
    built with `model_construct` (or mutated after validation) into a shape OrderLeg forbids - a
    limit order without a price, a notional limit/gtc order, a non-positive qty - cannot sit in a
    validated decision. The `model_construct` fallback below is taken only for such a leg that a
    rule has already REJECTED: the caller still gets the reject verdict and its reason, and the
    leg can never reach a ticket as tradable. An accepted or clamped invalid leg is never papered
    over this way - its ValidationError propagates, so `evaluate` fails closed."""
    fields = dict(index=index, verdict=verdict, leg=leg, original=original, reasons=reasons,
                  est_price=price, est_cost_usd=cost)
    try:
        return RiskDecision(**fields)
    except ValidationError:
        if verdict != "reject":
            raise
        return RiskDecision.model_construct(**fields)


def overall(decisions: list[RiskDecision]) -> Literal["ok", "warnings", "blocked"]:
    """The ticket-level status: `blocked` when nothing can go ahead (no decisions, or all
    rejected); `warnings` when any leg was rejected or clamped or carries a warning; otherwise `ok`
    (info reasons don't count)."""
    if not decisions or all(d.verdict == "reject" for d in decisions):
        return "blocked"
    for d in decisions:
        if d.verdict != "accept" or any(r.severity != "info" for r in d.reasons):
            return "warnings"
    return "ok"
