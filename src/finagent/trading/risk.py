"""Deterministic pre-trade risk rules: the LLM proposes order legs, this code decides.

`evaluate` checks each proposed `OrderLeg` against the account/market snapshot (`MarketContext`)
and the user's `Mandate`, and returns one `RiskDecision` per leg: `accept`, `clamp` (the leg was
shrunk or re-priced to fit a limit - `leg` holds the changed copy) or `reject`. `overall` rolls
those up into the single status a ticket shows.

Pure and deterministic: no I/O, no clock reads, no randomness, and the inputs are never mutated
or aliased (every returned leg is a copy). All arithmetic runs in a private decimal context, so the
caller's ambient `decimal` settings cannot change a result. Money is `Decimal`, never `float`.

Per leg, the rules run in `_RULES` order (the task brief's reason-code table, mirrored by
`REASON_CODES`) and stop at the first blocking reason; warnings and clamps found before that stay
on the decision. Legs are evaluated in list order, and every accepted or clamped leg carries its
effects into the legs after it (`_Book`): buys consume usable buying power and add to the working
position (value, and qty available to sell), sells take shares and value away, each counts
towards `max_orders_per_day`, and a buy into a symbol not yet held opens a position for
`max_positions`. Rejected legs have no effect.

Prices and estimates:
- reference price: the last trade, else the bid/ask midpoint (both sides needed); a missing or
  non-positive value counts as absent, and no reference at all is `no_quote`.
- `est_price`: the limit price if set; else, for a buy, ask x (1 + MARKET_SLIPPAGE_PCT/100) and,
  for a sell, the bid - each falling back to the reference price when that side is missing.
- `est_cost_usd`: est_price x qty, or the notional.
- Rejected legs report `leg == original` and the estimates of the original leg.

Clamps: a qty clamp floors to whole shares unless fractional trading is allowed for the leg
(asset fractionable, `mandate.allow_fractional`, time_in_force "day"), in which case it floors to
9 decimals; a notional clamp floors to cents and may not fall below $1.00. A clamp that leaves
nothing rejects the leg with the clamping rule's own code.
"""
from __future__ import annotations

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
    Quote,
    RiskDecision,
    RiskReason,
)

MARKET_SLIPPAGE_PCT = Decimal("1.0")  # buffer on ask for market-buy cost estimates
LARGE_ORDER_PCT = Decimal("10")  # warn when one leg's estimated cost exceeds this % of equity
MIN_NOTIONAL_USD = Decimal("1.00")  # a notional clamp may not go below Alpaca's $1 minimum
FRACTIONAL_DECIMALS = 9  # Alpaca accepts fractional qty to 9 decimal places

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


def _est_price(leg: OrderLeg, quote: Quote | None, ref: Decimal | None) -> Decimal | None:
    if leg.limit_price is not None:
        return leg.limit_price
    if leg.side == "buy":
        ask = _positive(quote.ask) if quote is not None else None
        if ask is not None:
            return ask * (1 + MARKET_SLIPPAGE_PCT / _HUNDRED)
    else:
        bid = _positive(quote.bid) if quote is not None else None
        if bid is not None:
            return bid
    return ref


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


# --- working state -------------------------------------------------------------------------------


@dataclass
class _Book:
    """What the legs accepted so far in this `evaluate` call have committed."""

    bp_used: Decimal
    orders: int
    held: set[str]  # symbols with an open position, including ones opened earlier in the ticket
    qty_available: dict[str, Decimal]
    value: dict[str, Decimal]  # position market value for max_position_pct

    @classmethod
    def from_ctx(cls, ctx: MarketContext) -> "_Book":
        return cls(
            bp_used=_ZERO,
            orders=0,
            held={s for s, p in ctx.positions.items() if p.qty != 0},
            qty_available={s: p.qty_available for s, p in ctx.positions.items()},
            value={s: abs(p.market_value) for s, p in ctx.positions.items()},
        )

    def commit(self, leg: OrderLeg, price: Decimal, cost: Decimal) -> None:
        symbol = leg.symbol
        shares = leg.qty if leg.qty is not None else leg.notional / price
        self.orders += 1
        if leg.side == "buy":
            self.bp_used += cost
            self.held.add(symbol)
            self.qty_available[symbol] = self.qty_available.get(symbol, _ZERO) + shares
            self.value[symbol] = self.value.get(symbol, _ZERO) + cost
        else:
            self.qty_available[symbol] = self.qty_available.get(symbol, _ZERO) - shares
            self.value[symbol] = max(_ZERO, self.value.get(symbol, _ZERO) - cost)


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
    whole = c.whole_shares
    new_qty = min(_fit_qty(budget, c.est_price, whole), leg.qty)
    if new_qty <= 0:
        kind = "whole-share quantity" if whole else "quantity"
        c.block(code, f"{because}, and no {kind} fits")
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
        problems.append(f"status {asset.status}")
    if asset.cls.lower() != "us_equity":
        problems.append(f"asset class {asset.cls}, not us_equity")
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
    want = leg.qty if leg.qty is not None else leg.notional / c.est_price
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
                f"An open {opposite} order for {leg.symbol} (id {order.id}) is on the opposite "
                "side; this order could trade against it (potential wash trade)",
            )
            return


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
        open_ = "1 position is" if count == 1 else f"{count} positions are"
        c.block(
            "max_positions",
            f"Buying {leg.symbol} would open a new position, but {open_} already open "
            f"(mandate max {cap})",
        )


def _buying_power(c: _Check) -> None:
    if c.leg.side != "buy":
        return
    buying_power = c.ctx.account.buying_power
    used = c.book.bp_used
    usable = buying_power * (_HUNDRED - c.mandate.cash_buffer_pct) / _HUNDRED - used
    cost = c.est_cost
    if cost > usable:
        after = f" (after {_usd(used)} committed by earlier legs)" if used > 0 else ""
        _shrink(c, "buying_power", usable,
                f"estimated cost {_usd(cost)} exceeds usable buying power {_usd(usable)}{after}")


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
            f"({c.ctx.clock.next_open})",
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


# Evaluation order = the brief's reason-code table. Each rule is named after the one code it emits.
_RULES: tuple[Callable[[_Check], None], ...] = (
    _account_blocked,
    _asset_not_tradable,
    _denied_symbol,
    _order_type_not_allowed,
    _notional_needs_market_day,
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
                verdict = "clamp" if check.leg != original else "accept"
                final = check.leg
            price = _est_price(final, quote, ref)
            cost = _est_cost(final, price)
            if verdict != "reject":
                book.commit(final, price, cost)
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
    fields = dict(index=index, verdict=verdict, leg=leg, original=original, reasons=reasons,
                  est_price=price, est_cost_usd=cost)
    try:
        return RiskDecision(**fields)
    except ValidationError:
        if verdict != "reject":
            raise  # never hand back an accepted/clamped leg that fails OrderLeg's own validation
        # OrderLeg's after-validator re-runs on an embedded instance, so a leg that bypassed it
        # (model_construct) cannot sit in a validated RiskDecision. It is rejected anyway; build the
        # decision unvalidated so the caller still gets the verdict and reason.
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
