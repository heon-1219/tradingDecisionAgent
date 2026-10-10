"""Tests for `finagent.trading.risk` - the deterministic pre-trade risk rules.

Fixture world (see the factories below): an account with $20,000 equity and $1,950 of cash,
buying power and non-marginable buying power, the default `Mandate()` (20% max position, a 5% cash
buffer = 5% of the $20,000 equity, so $1,950 - $1,000 = $950 usable; 3% price band, 10 orders a
day), the market open, and four symbols:

- VOO  last $500, bid $499.90, ask $500.10 - a *whole-share* asset in this fixture
       (`fractionable=False`), so a clamp lands on an integer the way the brief's cases expect;
- AAPL last $200, bid $199.90, ask $200.10 - fractionable;
- PNY  last $0.50, bid $0.4990, ask $0.5010 - a sub-dollar, whole-share asset (sub-penny ticks);
- NOQ  a tradable asset with no quote at all.
"""
from __future__ import annotations

from decimal import ROUND_UP, Decimal, localcontext

import pytest

from finagent.trading.models import (
    AccountSnapshot,
    AssetInfo,
    ClockInfo,
    Mandate,
    MarketContext,
    OrderLeg,
    OrderSnapshot,
    PositionSnapshot,
    Quote,
    RiskDecision,
    RiskReason,
)
from finagent.trading.risk import MARKET_SLIPPAGE_PCT, REASON_CODES, evaluate, overall

D = Decimal

# --- factories -----------------------------------------------------------------------------


def make_account(**over) -> AccountSnapshot:
    fields = dict(
        status="ACTIVE",
        equity=D("20000"),
        cash=D("1950"),
        buying_power=D("1950"),
        non_marginable_buying_power=D("1950"),
        last_equity=D("20000"),
        trading_blocked=False,
        account_blocked=False,
        trade_suspended_by_user=False,
        shorting_enabled=False,
    )
    fields.update(over)
    return AccountSnapshot(**fields)


def make_asset(symbol: str, *, fractionable: bool = True, **over) -> AssetInfo:
    fields = dict(
        symbol=symbol,
        cls="us_equity",
        status="active",
        tradable=True,
        fractionable=fractionable,
        shortable=False,
        marginable=True,
    )
    fields.update(over)
    return AssetInfo(**fields)


def make_quote(symbol: str, *, last=None, bid=None, ask=None) -> Quote:
    return Quote(symbol=symbol, last=last, bid=bid, ask=ask, as_of="2026-09-25T15:59:00Z", feed="iex")


def make_position(symbol: str, qty, price, qty_available=None) -> PositionSnapshot:
    qty, price = D(str(qty)), D(str(price))
    return PositionSnapshot(
        symbol=symbol,
        qty=qty,
        qty_available=qty if qty_available is None else D(str(qty_available)),
        avg_entry_price=price,
        current_price=price,
        market_value=qty * price,
        unrealized_pl=D("0"),
        unrealized_plpc=D("0"),
        side="long",
    )


def make_order(symbol: str, side: str, **over) -> OrderSnapshot:
    fields = dict(
        id="open-order-1",
        client_order_id="cid-1",
        symbol=symbol,
        side=side,
        qty=D("1"),
        type="limit",
        time_in_force="day",
        limit_price=D("500"),
        status="new",
    )
    fields.update(over)
    return OrderSnapshot(**fields)


def make_clock(is_open: bool = True) -> ClockInfo:
    return ClockInfo(
        is_open=is_open,
        next_open="2026-09-28T09:30:00-04:00",
        next_close="2026-09-25T16:00:00-04:00",
        timestamp="2026-09-25T12:00:00-04:00",
    )


ASSETS = {
    "VOO": make_asset("VOO", fractionable=False),
    "AAPL": make_asset("AAPL"),
    "PNY": make_asset("PNY", fractionable=False),
    "NOQ": make_asset("NOQ"),
}
QUOTES = {
    "VOO": make_quote("VOO", last="500", bid="499.90", ask="500.10"),
    "AAPL": make_quote("AAPL", last="200", bid="199.90", ask="200.10"),
    "PNY": make_quote("PNY", last="0.5000", bid="0.4990", ask="0.5010"),
}


def make_ctx(
    *,
    positions=(),
    open_orders=(),
    assets=None,
    quotes=None,
    is_open: bool = True,
    orders_today: int = 0,
    **account,
) -> MarketContext:
    """The fixture context; keyword arguments not named here override account fields."""
    return MarketContext(
        account=make_account(**account),
        positions={p.symbol: p for p in positions},
        open_orders=list(open_orders),
        assets=dict(ASSETS if assets is None else assets),
        quotes=dict(QUOTES if quotes is None else quotes),
        clock=make_clock(is_open),
        orders_today=orders_today,
    )


def ctx_with_position(qty, symbol: str = "VOO", price="500", **ctx_over) -> MarketContext:
    return make_ctx(positions=[make_position(symbol, qty, price)], **ctx_over)


def _leg(side, symbol, qty, limit, notional, tif) -> OrderLeg:
    return OrderLeg(
        symbol=symbol,
        side=side,
        qty=None if qty is None else D(str(qty)),
        notional=None if notional is None else D(str(notional)),
        type="market" if limit is None else "limit",
        time_in_force=tif,
        limit_price=None if limit is None else D(str(limit)),
    )


def buy(symbol="VOO", qty=None, *, limit=None, notional=None, tif="day") -> OrderLeg:
    return _leg("buy", symbol, qty, limit, notional, tif)


def sell(symbol="VOO", qty=None, *, limit=None, notional=None, tif="day") -> OrderLeg:
    return _leg("sell", symbol, qty, limit, notional, tif)


def codes(decision: RiskDecision) -> list[str]:
    return [r.code for r in decision.reasons]


def reason(decision: RiskDecision, code: str) -> RiskReason:
    return next(r for r in decision.reasons if r.code == code)


@pytest.fixture
def ctx() -> MarketContext:
    return make_ctx()


@pytest.fixture
def mandate() -> Mandate:
    return Mandate()


# --- the brief's exact cases ---------------------------------------------------------------


def test_buy_clamped_to_buying_power(ctx, mandate):   # cash 1,950 less 5% of 20,000 equity -> 950 usable
    d = evaluate([OrderLeg(symbol="VOO", side="buy", qty=Decimal("5"), type="limit", limit_price=Decimal("500"))], ctx, mandate)[0]
    assert d.verdict == "clamp" and d.leg.qty == Decimal("1") and "buying_power" in {r.code for r in d.reasons}


def test_sell_more_than_held_is_short(ctx, mandate):
    d = evaluate([OrderLeg(symbol="VOO", side="sell", qty=Decimal("3"), type="market")], ctx_with_position(qty=2), mandate)[0]
    assert d.verdict == "reject" and d.reasons[0].code == "short_sell"


def test_two_legs_share_buying_power(ctx, mandate):
    # Leg 0 uses $500 of the $950 usable; leg 1 ($1,000 of fractionable AAPL) gets the $450 left.
    first, second = evaluate([buy("VOO", 1, limit="500"), buy("AAPL", 5, limit="200")], ctx, mandate)
    assert first.verdict == "accept" and first.reasons == []
    assert second.verdict == "clamp"
    assert second.leg.qty == D("2.25")
    assert second.est_cost_usd == D("450")
    assert codes(second) == ["buying_power"]
    assert reason(second, "buying_power").message == (
        "Clamped qty 5 → 2.25: estimated cost $1,000.00 exceeds usable buying power $450.00 "
        "(after $500.00 committed by earlier legs)"
    )


# --- one case (or more) per reason code ------------------------------------------------------

_LIMIT_WITHOUT_PRICE = OrderLeg.model_construct(
    symbol="VOO", side="buy", qty=D("1"), notional=None, type="limit", time_in_force="day",
    limit_price=None,
)
_NOTIONAL_LIMIT = OrderLeg.model_construct(
    symbol="AAPL", side="buy", qty=None, notional=D("100"), type="limit", time_in_force="day",
    limit_price=D("200"),
)
_NOTIONAL_GTC = OrderLeg.model_construct(
    symbol="AAPL", side="buy", qty=None, notional=D("100"), type="market", time_in_force="gtc",
    limit_price=None,
)
# Enough cash buying power that it never binds (risk uses the lowest of buying_power,
# non_marginable_buying_power and cash).
_RICH = dict(buying_power=D("100000"), non_marginable_buying_power=D("100000"), cash=D("100000"))

CASES = [
    # account_blocked
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(trading_blocked=True), Mandate(),
                 "reject", "account_blocked", id="account_blocked-trading"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(account_blocked=True), Mandate(),
                 "reject", "account_blocked", id="account_blocked-account"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(trade_suspended_by_user=True), Mandate(),
                 "reject", "account_blocked", id="account_blocked-suspended"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(status="ACCOUNT_CLOSED"), Mandate(),
                 "reject", "account_blocked", id="account_blocked-status-not-active"),
    # asset_not_tradable
    pytest.param(buy("VOO", 1, limit="500"),
                 make_ctx(assets={**ASSETS, "VOO": make_asset("VOO", tradable=False)}), Mandate(),
                 "reject", "asset_not_tradable", id="asset-not-tradable"),
    pytest.param(buy("VOO", 1, limit="500"),
                 make_ctx(assets={**ASSETS, "VOO": make_asset("VOO", status="inactive")}), Mandate(),
                 "reject", "asset_not_tradable", id="asset-inactive"),
    pytest.param(buy("VOO", 1, limit="500"),
                 make_ctx(assets={**ASSETS, "VOO": make_asset("VOO", cls="crypto")}), Mandate(),
                 "reject", "asset_not_tradable", id="asset-not-us-equity"),
    pytest.param(buy("ZZZ", 1, limit="10"), make_ctx(), Mandate(),
                 "reject", "asset_not_tradable", id="asset-unknown"),
    # denied_symbol
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(), Mandate(deny_symbols=["voo"]),
                 "reject", "denied_symbol", id="denied-deny-list"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(), Mandate(allow_symbols=["AAPL"]),
                 "reject", "denied_symbol", id="denied-not-in-allow-list"),
    # order_type_not_allowed
    pytest.param(buy("VOO", 1), make_ctx(), Mandate(order_types=["limit"]),
                 "reject", "order_type_not_allowed", id="order-type-market-not-allowed"),
    # precision
    pytest.param(buy("AAPL", "1." + "0" * 39 + "1", limit="200"), make_ctx(), Mandate(),
                 "reject", "precision", id="precision-41-digit-qty"),
    # notional_needs_market_day (OrderLeg itself forbids these; the rule is defense in depth)
    pytest.param(_NOTIONAL_LIMIT, make_ctx(), Mandate(),
                 "reject", "notional_needs_market_day", id="notional-limit"),
    pytest.param(_NOTIONAL_GTC, make_ctx(), Mandate(),
                 "reject", "notional_needs_market_day", id="notional-gtc"),
    # notional_invalid
    pytest.param(buy("AAPL", notional="0.5"), make_ctx(), Mandate(),
                 "reject", "notional_invalid", id="notional-below-one-dollar"),
    pytest.param(buy("AAPL", notional="10.555"), make_ctx(), Mandate(),
                 "reject", "notional_invalid", id="notional-sub-cent"),
    pytest.param(buy("AAPL", "0.000000001", limit="200"), make_ctx(), Mandate(),
                 "reject", "notional_invalid", id="tiny-fractional-qty-cost-below-one-dollar"),
    pytest.param(buy("PNY", 1), make_ctx(), Mandate(),
                 "reject", "notional_invalid", id="whole-share-market-cost-below-one-dollar"),
    # fractional_not_allowed
    pytest.param(buy("VOO", "0.5", limit="500"), make_ctx(), Mandate(),
                 "reject", "fractional_not_allowed", id="fractional-asset-not-fractionable"),
    pytest.param(buy("AAPL", "0.5", limit="200"), make_ctx(), Mandate(allow_fractional=False),
                 "reject", "fractional_not_allowed", id="fractional-mandate-disallows"),
    pytest.param(buy("AAPL", "0.5", limit="200", tif="gtc"), make_ctx(), Mandate(),
                 "reject", "fractional_not_allowed", id="fractional-gtc"),
    pytest.param(buy("VOO", notional="100"), make_ctx(), Mandate(),
                 "reject", "fractional_not_allowed", id="fractional-notional-not-fractionable"),
    pytest.param(buy("AAPL", notional="100"), make_ctx(), Mandate(allow_fractional=False),
                 "reject", "fractional_not_allowed", id="fractional-notional-mandate-disallows"),
    pytest.param(buy("AAPL", "0.0100000001", limit="200"), make_ctx(), Mandate(),
                 "reject", "fractional_not_allowed", id="fractional-more-than-9-decimals"),
    # limit_required (OrderLeg itself forbids this; the rule is defense in depth)
    pytest.param(_LIMIT_WITHOUT_PRICE, make_ctx(), Mandate(),
                 "reject", "limit_required", id="limit-without-price"),
    # bad_tick
    pytest.param(buy("VOO", 1, limit="500.005"), make_ctx(), Mandate(),
                 "clamp", "bad_tick", id="bad-tick-rounded"),
    pytest.param(buy("PNY", 100000, limit="0.00004"), make_ctx(), Mandate(),
                 "reject", "bad_tick", id="bad-tick-rounds-to-zero"),
    pytest.param(buy("PNY", 3, limit="0.33334"), make_ctx(), Mandate(),
                 "reject", "bad_tick", id="bad-tick-rounds-cost-below-one-dollar"),
    # price_band
    pytest.param(buy("VOO", 1, limit="520"), make_ctx(), Mandate(),
                 "reject", "price_band", id="price-band-buy-above"),
    pytest.param(sell("VOO", 1, limit="480"), ctx_with_position(5), Mandate(),
                 "reject", "price_band", id="price-band-sell-below"),
    # no_quote
    pytest.param(buy("NOQ", 1), make_ctx(), Mandate(),
                 "reject", "no_quote", id="no-quote-market"),
    pytest.param(buy("NOQ", 1, limit="10"), make_ctx(), Mandate(),
                 "reject", "no_quote", id="no-quote-limit"),
    pytest.param(buy("VOO", 1),
                 make_ctx(quotes={**QUOTES, "VOO": make_quote("VOO", bid="0", ask="0")}), Mandate(),
                 "reject", "no_quote", id="no-quote-zero-bid-ask"),
    # short_sell
    pytest.param(sell("VOO", 1), make_ctx(), Mandate(),
                 "reject", "short_sell", id="short-sell-no-position"),
    # wash_trade
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(open_orders=[make_order("VOO", "sell")]),
                 Mandate(), "reject", "wash_trade", id="wash-buy-vs-open-sell"),
    pytest.param(sell("VOO", 1, limit="500"),
                 ctx_with_position(5, open_orders=[make_order("VOO", "buy")]), Mandate(),
                 "reject", "wash_trade", id="wash-sell-vs-open-buy"),
    # max_order_usd
    pytest.param(buy("VOO", 5, limit="500"), make_ctx(**_RICH), Mandate(max_order_usd=D("1000")),
                 "clamp", "max_order_usd", id="max-order-clamp"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(**_RICH), Mandate(max_order_usd=D("100")),
                 "reject", "max_order_usd", id="max-order-nothing-fits"),
    # max_position_pct
    pytest.param(buy("AAPL", 25, limit="200"), make_ctx(**_RICH), Mandate(),
                 "clamp", "max_position_pct", id="max-position-clamp"),
    pytest.param(buy("AAPL", 1, limit="200"), ctx_with_position(20, "AAPL", "200", **_RICH),
                 Mandate(), "reject", "max_position_pct", id="max-position-already-full"),
    # max_positions
    pytest.param(buy("VOO", 1, limit="500"), ctx_with_position(1, "AAPL", "200"),
                 Mandate(max_positions=1), "reject", "max_positions", id="max-positions"),
    # buying_power
    pytest.param(buy("VOO", 5, limit="500"), make_ctx(), Mandate(),
                 "clamp", "buying_power", id="buying-power-clamp"),
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(buying_power=D("1100")), Mandate(),
                 "reject", "buying_power", id="buying-power-nothing-fits"),
    # budget
    pytest.param(buy("VOO", 2, limit="500"), ctx_with_position(20, "AAPL", "200", **_RICH),
                 Mandate(budget_usd=D("4500"), max_position_pct=D("100")),
                 "clamp", "budget", id="budget-clamp"),
    pytest.param(buy("VOO", 1, limit="500"), ctx_with_position(20, "AAPL", "200", **_RICH),
                 Mandate(budget_usd=D("4000"), max_position_pct=D("100")),
                 "reject", "budget", id="budget-exhausted"),
    # orders_per_day
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(orders_today=10), Mandate(),
                 "reject", "orders_per_day", id="orders-per-day"),
    # market_closed
    pytest.param(buy("VOO", 1, limit="500"), make_ctx(is_open=False), Mandate(),
                 "accept", "market_closed", id="market-closed"),
    # large_order
    pytest.param(buy("VOO", 5, limit="500"), make_ctx(**_RICH), Mandate(),
                 "accept", "large_order", id="large-order"),
]


@pytest.mark.parametrize("leg, context, mandate_, verdict, code", CASES)
def test_rule_table(leg, context, mandate_, verdict, code):
    [d] = evaluate([leg], context, mandate_)
    assert d.index == 0
    assert d.verdict == verdict
    assert code in codes(d)
    blocks = [r for r in d.reasons if r.severity == "block"]
    if verdict == "reject":
        # Evaluation stops at the first blocking reason, so it is always the last one.
        assert [r.code for r in blocks] == [code]
        assert d.reasons[-1].code == code
        assert d.leg == d.original == leg
    else:
        assert blocks == []
        assert reason(d, code).severity == "warn"
        assert (d.leg != d.original) == (verdict == "clamp")
    assert all(r.message for r in d.reasons)


def test_reason_codes_are_the_plans_table_in_order():
    # The plan's table, plus precision (fix round 2) before the first rule that does arithmetic.
    assert REASON_CODES == (
        "account_blocked", "asset_not_tradable", "denied_symbol", "order_type_not_allowed",
        "precision", "notional_needs_market_day", "notional_invalid", "fractional_not_allowed",
        "limit_required", "bad_tick", "price_band", "no_quote", "short_sell", "wash_trade",
        "max_order_usd", "max_position_pct", "max_positions", "buying_power", "budget",
        "orders_per_day", "market_closed", "large_order",
    )


def test_rule_table_covers_every_reason_code():
    assert {p.values[4] for p in CASES} == set(REASON_CODES)


def test_market_slippage_constant():
    assert MARKET_SLIPPAGE_PCT == Decimal("1.0")


# --- evaluation order ------------------------------------------------------------------------


def test_stops_at_first_block_in_table_order():
    [d] = evaluate([buy("VOO", "0.5", limit="520")], make_ctx(trading_blocked=True),
                   Mandate(deny_symbols=["VOO"], order_types=["market"]))
    assert codes(d) == ["account_blocked"]


def test_clamped_then_rejected_leg_keeps_original_and_both_reasons():
    [d] = evaluate([buy("VOO", 5, limit="500")], make_ctx(orders_today=10, **_RICH),
                   Mandate(max_order_usd=D("1000")))
    assert d.verdict == "reject"
    assert codes(d) == ["max_order_usd", "orders_per_day"]
    assert [r.severity for r in d.reasons] == ["warn", "block"]
    assert d.leg == d.original and d.leg.qty == D("5")
    assert d.est_price == D("500") and d.est_cost_usd == D("2500")


def test_rejected_leg_still_gets_estimates_when_priceable():
    band, no_quote_market, no_quote_limit = evaluate(
        [buy("VOO", 2, limit="520"), buy("NOQ", 1), buy("NOQ", 3, limit="10")], make_ctx(), Mandate()
    )
    assert band.verdict == "reject" and band.est_price == D("520") and band.est_cost_usd == D("1040")
    assert no_quote_market.est_price is None and no_quote_market.est_cost_usd is None
    assert no_quote_limit.est_price == D("10") and no_quote_limit.est_cost_usd == D("30")


def test_invalid_leg_that_is_accepted_would_raise_rather_than_pass():
    # A leg that bypassed OrderLeg validation is never returned as accepted: a rule rejects it, or
    # building the decision fails closed. (A market order carrying a limit price passes every rule.)
    bad = OrderLeg.model_construct(symbol="VOO", side="buy", qty=D("1"), notional=None,
                                   type="market", time_in_force="day", limit_price=D("500"))
    with pytest.raises(ValueError):
        evaluate([bad], make_ctx(), Mandate())


# --- reference price and estimates -----------------------------------------------------------


def test_market_buy_estimate_adds_slippage_to_ask():
    [d] = evaluate([buy("VOO", 1)], make_ctx(), Mandate())
    assert d.verdict == "accept"
    assert d.est_price == D("500.10") * (1 + MARKET_SLIPPAGE_PCT / 100) == D("505.101")
    assert d.est_cost_usd == D("505.101")


def test_market_buy_without_ask_uses_reference_plus_slippage():
    quotes = {**QUOTES, "VOO": make_quote("VOO", last="500", bid="499.90")}
    [d] = evaluate([buy("VOO", 1)], make_ctx(quotes=quotes), Mandate())
    assert d.est_price == D("505")


def test_market_buy_estimate_uses_reference_when_ask_is_stale_low():
    # Review probe 4: a stale $1.00 ask must not make 900 VOO (last $500) look like a $909 order.
    quotes = {**QUOTES, "VOO": make_quote("VOO", last="500", bid="0.50", ask="1.00")}
    [one] = evaluate([buy("VOO", 1)], make_ctx(quotes=quotes), Mandate())
    assert one.est_price == D("505")  # max(1.00, 500) x 1.01
    [d] = evaluate([buy("VOO", 900)], make_ctx(quotes=quotes), Mandate(max_order_usd=D("1000")))
    assert d.verdict == "clamp" and d.leg.qty == D("1")
    assert reason(d, "max_order_usd").message == (
        "Clamped qty 900 → 1: estimated cost $454,500.00 exceeds the mandate's max order $1,000.00"
    )


def test_market_sell_estimate_is_bid_else_reference():
    [with_bid] = evaluate([sell("VOO", 1)], ctx_with_position(2), Mandate())
    assert with_bid.est_price == D("499.90")
    quotes = {**QUOTES, "VOO": make_quote("VOO", last="500", ask="500.10")}
    [no_bid] = evaluate([sell("VOO", 1)], ctx_with_position(2, quotes=quotes), Mandate())
    assert no_bid.est_price == D("500")


def test_reference_falls_back_to_mid_for_price_band():
    quotes = {**QUOTES, "VOO": make_quote("VOO", bid="499", ask="501")}  # no last -> mid 500
    inside, outside = evaluate(
        [buy("VOO", 1, limit="514"), buy("VOO", 1, limit="516")], make_ctx(quotes=quotes, **_RICH),
        Mandate(),
    )
    assert inside.verdict == "accept"
    assert outside.verdict == "reject" and codes(outside) == ["price_band"]
    assert reason(outside, "price_band").message == (
        "Limit $516.00 is 3.20% from the reference price $500.00 (band 3%)"
    )


def test_lone_bid_is_not_a_reference():
    quotes = {**QUOTES, "VOO": make_quote("VOO", bid="499.90")}
    [d] = evaluate([sell("VOO", 1)], ctx_with_position(2, quotes=quotes), Mandate())
    assert codes(d) == ["no_quote"]


def test_price_band_boundary_is_inclusive():
    [d] = evaluate([buy("VOO", 1, limit="515")], make_ctx(), Mandate())
    assert d.verdict == "accept" and d.reasons == []


# --- tick rounding ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leg, context, expected",
    [
        pytest.param(buy("VOO", 1, limit="500.009"), make_ctx(), D("500.00"), id="buy>=1-floors"),
        pytest.param(sell("VOO", 1, limit="500.001"), ctx_with_position(5), D("500.01"),
                     id="sell>=1-ceils"),
        pytest.param(buy("PNY", 100, limit="0.49995"), make_ctx(), D("0.4999"),
                     id="buy<1-floors"),
        pytest.param(sell("PNY", 100, limit="0.49915"), ctx_with_position(100, "PNY", "0.5"),
                     D("0.4992"), id="sell<1-ceils"),
    ],
)
def test_tick_rounding(leg, context, expected):
    [d] = evaluate([leg], context, Mandate())
    assert d.verdict == "clamp"
    assert d.leg.limit_price == expected
    assert d.original.limit_price == leg.limit_price
    assert codes(d) == ["bad_tick"]
    assert reason(d, "bad_tick").severity == "warn"


def test_tick_rounding_message():
    [d] = evaluate([buy("VOO", 1, limit="500.009")], make_ctx(), Mandate())
    assert reason(d, "bad_tick").message == (
        "Rounded limit price $500.009 → $500.00 (buy: down to the $0.01 tick)"
    )


def test_sub_dollar_sell_rounding_up_to_a_dollar_uses_cent_ticks():
    quotes = {**QUOTES, "PNY": make_quote("PNY", last="1.00", bid="0.9999", ask="1.0001")}
    [d] = evaluate([sell("PNY", 10, limit="0.99995")],
                   ctx_with_position(10, "PNY", "1", quotes=quotes), Mandate())
    assert d.leg.limit_price == D("1.00")
    assert d.leg.limit_price.as_tuple().exponent == -2


@pytest.mark.parametrize("leg", [buy("VOO", 1, limit="500.000"), buy("PNY", 100, limit="0.4999")])
def test_on_tick_price_is_untouched(leg):
    [d] = evaluate([leg], make_ctx(), Mandate())
    assert d.verdict == "accept" and d.reasons == []
    assert d.leg == leg


# --- fractional vs whole-share clamping ------------------------------------------------------


def test_fractional_clamp_floors_to_nine_decimals(ctx):
    [d] = evaluate([buy("AAPL", 10, limit="201")], ctx, Mandate())
    assert d.verdict == "clamp"
    assert d.leg.qty == D("4.726368159")  # 950 / 201 = 4.72636815920... floored
    assert d.est_cost_usd == D("949.999999959") <= D("950")


@pytest.mark.parametrize(
    "leg, mandate_",
    [
        pytest.param(buy("AAPL", 10, limit="201"), Mandate(allow_fractional=False),
                     id="mandate-disallows-fractional"),
        pytest.param(buy("AAPL", 10, limit="201", tif="gtc"), Mandate(), id="gtc"),
    ],
)
def test_whole_share_clamp_floors_to_integer(leg, mandate_):
    [d] = evaluate([leg], make_ctx(), mandate_)
    assert d.verdict == "clamp" and d.leg.qty == D("4")  # floor(950 / 201)


def test_non_fractionable_asset_clamps_to_integer():
    assets = {**ASSETS, "AAPL": make_asset("AAPL", fractionable=False)}
    [d] = evaluate([buy("AAPL", 10, limit="201")], make_ctx(assets=assets), Mandate())
    assert d.verdict == "clamp" and d.leg.qty == D("4")
    assert reason(d, "buying_power").message == (
        "Clamped qty 10 → 4: estimated cost $2,010.00 exceeds usable buying power $950.00"
    )


def test_whole_share_clamp_that_reaches_zero_rejects_with_the_rules_code():
    # $1,095 of buying power less the $1,000 buffer (5% of $20,000 equity) leaves $95
    [d] = evaluate([buy("VOO", 1, limit="500")], make_ctx(buying_power=D("1095")), Mandate())
    assert d.verdict == "reject" and codes(d) == ["buying_power"]
    assert reason(d, "buying_power").message == (
        "Estimated cost $500.00 exceeds usable buying power $95.00, and no whole-share quantity fits"
    )


# --- notional clamps and the $1.00 floor -----------------------------------------------------


def test_notional_clamped_to_usable_buying_power(ctx):
    [d] = evaluate([buy("AAPL", notional="2000")], ctx, Mandate())
    assert d.verdict == "clamp" and d.leg.notional == D("950.00") and d.leg.qty is None
    assert d.est_cost_usd == D("950.00")
    assert d.est_price == D("200.10") * D("1.01")
    assert reason(d, "buying_power").message == (
        "Clamped notional $2,000.00 → $950.00: estimated cost $2,000.00 exceeds usable buying power "
        "$950.00"
    )


def test_notional_clamp_floors_to_cents(ctx):
    [d] = evaluate([buy("AAPL", notional="50")], ctx, Mandate(max_order_usd=D("10.555")))
    assert d.verdict == "clamp" and d.leg.notional == D("10.55")
    assert codes(d) == ["max_order_usd"]


def test_notional_clamp_below_one_dollar_rejects():
    [d] = evaluate([buy("AAPL", notional="5")], make_ctx(buying_power=D("1000.99")), Mandate())
    assert d.verdict == "reject" and codes(d) == ["buying_power"]  # usable $0.99 < $1.00
    [d] = evaluate([buy("AAPL", notional="5")], make_ctx(), Mandate(max_order_usd=D("0.50")))
    assert d.verdict == "reject" and codes(d) == ["max_order_usd"]


def test_notional_clamp_to_exactly_one_dollar_is_allowed():
    [d] = evaluate([buy("AAPL", notional="5")], make_ctx(buying_power=D("1.00")),
                   Mandate(cash_buffer_pct=D("0")))
    assert d.verdict == "clamp" and d.leg.notional == D("1.00")


def test_notional_sell_is_checked_against_available_shares():
    context = ctx_with_position(1, "AAPL", "200")
    too_much, fine = evaluate([sell("AAPL", notional="300")], context, Mandate()), evaluate(
        [sell("AAPL", notional="150")], context, Mandate()
    )
    assert too_much[0].verdict == "reject" and codes(too_much[0]) == ["short_sell"]  # 300/199.90 > 1
    assert fine[0].verdict == "accept"


_STALE_HIGH_BID = {**QUOTES, "AAPL": make_quote("AAPL", last="200", bid="400", ask="401")}


def test_notional_sell_uses_the_lower_of_bid_and_reference():
    # Fix round 1 open item B: at a stale $400 bid, $300 looks like 0.75 shares; at the $200
    # reference it is 1.5 shares - more than the 1 held.
    context = ctx_with_position(1, "AAPL", "200", quotes=_STALE_HIGH_BID)
    [d] = evaluate([sell("AAPL", notional="300")], context, Mandate())
    assert d.verdict == "reject" and codes(d) == ["short_sell"]
    assert reason(d, "short_sell").message == (
        "Selling $300.00 of AAPL (about 1.5 shares) exceeds the 1 AAPL available to sell; short "
        "selling is not allowed"
    )


def test_notional_sells_use_the_conservative_share_count_across_legs():
    # 2 held: $300 at the $200 reference uses 1.5, so $150 (0.75 more) no longer fits.
    context = ctx_with_position(2, "AAPL", "200", quotes=_STALE_HIGH_BID)
    ds = evaluate([sell("AAPL", notional="300"), sell("AAPL", notional="150")], context, Mandate())
    assert [d.verdict for d in ds] == ["accept", "reject"]
    assert codes(ds[1]) == ["short_sell"]


# --- sequential effects across legs ----------------------------------------------------------


def test_rejected_leg_has_no_effect_on_later_legs():
    rejected, accepted = evaluate([buy("VOO", 1, limit="520"), buy("VOO", 1, limit="500")],
                                  make_ctx(), Mandate(max_orders_per_day=1))
    assert rejected.verdict == "reject"
    assert accepted.verdict == "accept"  # neither buying power nor the order count was consumed


def test_orders_per_day_counts_earlier_accepted_legs():
    legs = [buy("AAPL", 1, limit="200"), buy("AAPL", 1, limit="200"), buy("AAPL", 1, limit="200")]
    ds = evaluate(legs, make_ctx(orders_today=8, **_RICH), Mandate())
    assert [d.verdict for d in ds] == ["accept", "accept", "reject"]
    assert codes(ds[2]) == ["orders_per_day"]
    [boundary] = evaluate([buy("AAPL", 1, limit="200")], make_ctx(orders_today=9), Mandate())
    assert boundary.verdict == "accept"


def test_buy_earlier_in_ticket_never_makes_shares_sellable():
    # Review probe 1: no position; an unfilled buy is not a position, so the sell is a short sale.
    ds = evaluate([buy("VOO", 1, limit="486"), sell("VOO", 1, limit="514")], make_ctx(), Mandate())
    assert [d.verdict for d in ds] == ["accept", "reject"]
    assert codes(ds[1]) == ["short_sell"]


def test_opposite_side_leg_in_same_ticket_is_a_wash_trade():
    # With shares to sell, the sell passes short_sell but trades against the earlier buy leg.
    ds = evaluate([buy("VOO", 1, limit="500"), sell("VOO", 1, limit="500")], ctx_with_position(5),
                  Mandate())
    assert [d.verdict for d in ds] == ["accept", "reject"]
    assert codes(ds[1]) == ["wash_trade"]
    assert reason(ds[1], "wash_trade").message == (
        "An earlier buy leg for VOO in this ticket is on the opposite side; this order could "
        "trade against it (potential wash trade)"
    )


def test_rejected_leg_does_not_count_for_intra_ticket_wash_trade():
    ds = evaluate([buy("VOO", 1, limit="520"), sell("VOO", 1, limit="500")], ctx_with_position(5),
                  Mandate())
    assert [d.verdict for d in ds] == ["reject", "accept"]  # the buy was rejected (price band)


def test_sells_consume_available_qty():
    ds = evaluate([sell("VOO", 2), sell("VOO", 1)], ctx_with_position(2), Mandate())
    assert [d.verdict for d in ds] == ["accept", "reject"]
    assert codes(ds[1]) == ["short_sell"]
    assert reason(ds[1], "short_sell").message == (
        "Selling 1 VOO exceeds the 0 VOO available to sell; short selling is not allowed"
    )


def test_sells_do_not_free_buying_power():
    ds = evaluate([sell("VOO", 5, limit="500"), buy("AAPL", 5, limit="200")], ctx_with_position(5),
                  Mandate())
    assert ds[0].verdict == "accept"
    assert ds[1].verdict == "clamp" and ds[1].leg.qty == D("4.75")  # still only $950 usable


def test_sells_do_not_free_budget():
    context = ctx_with_position(20, "AAPL", "200", **_RICH)  # $4,000 invested
    ds = evaluate([sell("AAPL", 10, limit="200"), buy("VOO", 3, limit="500")], context,
                  Mandate(budget_usd=D("5000"), max_position_pct=D("100")))
    assert ds[0].verdict == "accept"
    assert ds[1].verdict == "clamp" and ds[1].leg.qty == D("2")  # $1,000 of room, not $3,000
    assert codes(ds[1]) == ["budget"]


def test_max_positions_counts_positions_opened_earlier_in_ticket():
    legs = [buy("VOO", 1, limit="500"), buy("AAPL", 1, limit="200"), buy("VOO", 1, limit="500")]
    ds = evaluate(legs, make_ctx(**_RICH), Mandate(max_positions=1))
    assert [d.verdict for d in ds] == ["accept", "reject", "accept"]
    assert reason(ds[1], "max_positions").message == (
        "Buying AAPL would open a new position, but 1 position is already held or on order "
        "(mandate max 1)"
    )


def test_position_pct_accumulates_across_legs():
    # 20% of $20,000 = $4,000 per position: $3,000 first, so only $1,000 (5 shares) fits after.
    first, second = evaluate([buy("AAPL", 15, limit="200"), buy("AAPL", 10, limit="200")],
                             make_ctx(**_RICH), Mandate())
    assert first.verdict == "accept"
    assert second.verdict == "clamp" and second.leg.qty == D("5")
    assert reason(second, "max_position_pct").message == (
        "Clamped qty 10 → 5: AAPL position $3,000.00 + estimated cost $2,000.00 exceeds 20% of "
        "equity $20,000.00 ($4,000.00)"
    )


def test_sell_earlier_in_ticket_does_not_free_position_room():
    # Review probe 2: at the $4,000 cap, selling first must not make room for a buy-back.
    context = ctx_with_position(20, "AAPL", "200", **_RICH)
    [alone] = evaluate([buy("AAPL", 5, limit="200")], context, Mandate())
    assert alone.verdict == "reject" and codes(alone) == ["max_position_pct"]
    ds = evaluate([sell("AAPL", 20, limit="206"), buy("AAPL", 20, limit="194")], context, Mandate())
    assert [d.verdict for d in ds] == ["accept", "reject"]
    assert codes(ds[1]) == ["wash_trade"]


def test_position_pct_base_is_capped_by_budget():
    [d] = evaluate([buy("AAPL", 10, limit="200")], make_ctx(**_RICH), Mandate(budget_usd=D("5000")))
    assert d.verdict == "clamp" and d.leg.qty == D("5")  # 20% of min($20,000, $5,000) = $1,000
    assert "20% of budget $5,000.00 ($1,000.00)" in reason(d, "max_position_pct").message


def test_sells_never_trip_position_pct():
    [d] = evaluate([sell("AAPL", 10, limit="200")], ctx_with_position(40, "AAPL", "200"), Mandate())
    assert d.verdict == "accept"


# --- pending open buy orders ------------------------------------------------------------------


def test_pending_open_buy_counts_toward_position_pct():
    # Review probe 12: 20 AAPL @ $200 already on order fills the $4,000 cap.
    context = make_ctx(open_orders=[make_order("AAPL", "buy", qty=D("20"), limit_price=D("200"))],
                       **_RICH)
    [d] = evaluate([buy("AAPL", 20, limit="200")], context, Mandate())
    assert d.verdict == "reject" and codes(d) == ["max_position_pct"]
    assert reason(d, "max_position_pct").message.startswith("AAPL position $4,000.00 + ")


def test_pending_open_buy_counts_only_its_unfilled_part():
    # 10 AAPL held ($2,000) + an order for 15 with 10 filled (5 left, $1,000) = $3,000, so a
    # $1,000 buy still fits under the $4,000 cap (the filled 10 are not counted twice).
    order = make_order("AAPL", "buy", qty=D("15"), filled_qty=D("10"), limit_price=D("200"),
                       status="partially_filled")
    context = ctx_with_position(10, "AAPL", "200", open_orders=[order], **_RICH)
    [d] = evaluate([buy("AAPL", 5, limit="200")], context, Mandate())
    assert d.verdict == "accept"


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(make_order("AAPL", "buy", qty=D("20"), type="market", limit_price=None),
                     id="market-order-at-reference"),
        pytest.param(make_order("AAPL", "buy", qty=None, notional=D("4000"), type="market",
                                limit_price=None), id="notional-order"),
    ],
)
def test_pending_market_and_notional_buys_are_valued(order):
    [d] = evaluate([buy("AAPL", 1, limit="200")], make_ctx(open_orders=[order], **_RICH), Mandate())
    assert d.verdict == "reject" and codes(d) == ["max_position_pct"]


def test_pending_open_sell_orders_do_not_count_as_value():
    order = make_order("AAPL", "sell", qty=D("20"), limit_price=D("200"))
    context = ctx_with_position(20, "AAPL", "200", open_orders=[order], **_RICH)
    [d] = evaluate([buy("VOO", 1, limit="500")], context, Mandate())
    assert d.verdict == "accept"


def test_pending_open_buy_counts_as_a_held_position():
    # Review probe 12b: VOO held + AAPL on order = 2 positions; a third symbol is over the limit.
    context = make_ctx(positions=[make_position("VOO", 1, "500")],
                       open_orders=[make_order("AAPL", "buy", qty=D("1"), limit_price=D("200"))],
                       **_RICH)
    [third] = evaluate([buy("PNY", 10, limit="0.5")], context, Mandate(max_positions=2))
    assert third.verdict == "reject" and codes(third) == ["max_positions"]
    [more_aapl] = evaluate([buy("AAPL", 1, limit="200")], context, Mandate(max_positions=2))
    assert more_aapl.verdict == "accept"


# Review round 2 probe: the last trade ($100) is stale; the ask is $200.10.
_STALE_LAST = {**QUOTES, "AAPL": make_quote("AAPL", last="100", bid="199.90", ask="200.10")}


@pytest.mark.parametrize("order_type", ["market", "stop"])
def test_pending_market_or_stop_buy_is_valued_like_a_new_market_buy(order_type):
    # 20 x max($200.10, $100) x 1.01 = $4,042.02, not 20 x $100 = $2,000: the cap is already used.
    order = make_order("AAPL", "buy", qty=D("20"), type=order_type, limit_price=None)
    context = make_ctx(open_orders=[order], quotes=_STALE_LAST, **_RICH)
    [d] = evaluate([buy("AAPL", 10)], context, Mandate())
    assert d.verdict == "reject" and codes(d) == ["max_position_pct"]
    assert reason(d, "max_position_pct").message.startswith("AAPL position $4,042.02 + ")


def test_pending_market_buy_counts_conservatively_toward_budget():
    # $4,042.02 already committed is over a $2,600 budget, so a $500 VOO buy doesn't fit.
    order = make_order("AAPL", "buy", qty=D("20"), type="market", limit_price=None)
    context = make_ctx(open_orders=[order], quotes=_STALE_LAST, **_RICH)
    [d] = evaluate([buy("VOO", 1, limit="500")], context, Mandate(budget_usd=D("2600")))
    assert d.verdict == "reject" and codes(d) == ["budget"]


# --- cash-only buying power ------------------------------------------------------------------


def test_buying_power_is_cash_only_on_a_margin_account():
    # Reg T margin: buying_power is 2x the cash (non-marginable) figure; only the cash counts.
    context = make_ctx(buying_power=D("4000"), non_marginable_buying_power=D("1950"))
    [d] = evaluate([buy("VOO", 5, limit="500")], context, Mandate())
    assert d.verdict == "clamp" and d.leg.qty == D("1")
    assert reason(d, "buying_power").message == (
        "Clamped qty 5 → 1: estimated cost $2,500.00 exceeds usable buying power $950.00"
    )


def test_buying_power_uses_the_lower_of_the_two_figures():
    context = make_ctx(buying_power=D("1950"), non_marginable_buying_power=D("5000"), cash=D("5000"))
    [d] = evaluate([buy("VOO", 5, limit="500")], context, Mandate())
    assert d.leg.qty == D("1")  # min(1950, 5000, 5000) - 5% x $20,000 = $950


# Final review A-I1 / A-m8: usable = min(buying_power, non_marginable_buying_power,
# cash - pending open buys) - cash_buffer_pct x equity - earlier legs.


def test_cash_caps_a_buy_even_when_non_marginable_buying_power_is_higher():
    # The reviewer's probe: a margin paper account holding $60k of MSFT with $40k cash reports
    # non_marginable_buying_power $70k. Only the cash may be spent: 40,000 - 5% x 100,000 = 35,000.
    context = make_ctx(
        equity=D("100000"), cash=D("40000"), buying_power=D("140000"),
        non_marginable_buying_power=D("70000"), positions=[make_position("MSFT", 150, 400)],
    )
    [d] = evaluate([buy("AAPL", 330, limit="200")], context,
                   Mandate(max_position_pct=D("100"), max_positions=20))
    assert d.verdict == "clamp" and d.leg.qty == D("175") and d.est_cost_usd == D("35000")
    assert reason(d, "buying_power").message == (
        "Clamped qty 330 → 175: estimated cost $66,000.00 exceeds usable buying power $35,000.00"
    )
    assert D("40000") - d.est_cost_usd >= D("5000")  # cash never goes below the buffer


@pytest.mark.parametrize("bp,nmbp,cash", [
    ("4000", "1950", "1950"),  # Reg T margin: buying_power is double the cash figures
    ("1950", "5000", "5000"),  # buying_power is the lowest
    ("5000", "1950", "5000"),  # non_marginable_buying_power is the lowest
    ("5000", "5000", "1950"),  # cash is the lowest (the probe's case)
])
def test_usable_cash_is_the_lowest_of_the_three_figures_less_the_buffer(bp, nmbp, cash):
    # equity $20,000, 5% buffer = $1,000 kept in cash: 1,950 - 1,000 = 950 usable either way
    context = make_ctx(buying_power=D(bp), non_marginable_buying_power=D(nmbp), cash=D(cash))
    [d] = evaluate([buy("AAPL", 50, limit="200")], context, Mandate(max_position_pct=D("100")))
    assert d.verdict == "clamp" and d.est_cost_usd == D("950") and d.leg.qty == D("4.75")
    assert codes(d) == ["buying_power"]


def test_pending_open_buys_come_out_of_cash():
    # Alpaca's cash is not reduced by open orders: a $500 open VOO buy leaves 1,950 - 500 = 1,450,
    # less the $1,000 buffer = $450 for this ticket.
    context = make_ctx(buying_power=D("5000"), non_marginable_buying_power=D("5000"), cash=D("1950"),
                       open_orders=[make_order("VOO", "buy")])
    [d] = evaluate([buy("AAPL", 5, limit="200")], context, Mandate(max_position_pct=D("100")))
    assert d.verdict == "clamp" and d.leg.qty == D("2.25") and d.est_cost_usd == D("450")
    assert reason(d, "buying_power").message == (
        "Clamped qty 5 → 2.25: estimated cost $1,000.00 exceeds usable buying power $450.00 "
        "(after $500.00 in open buy orders)"
    )


def test_pending_open_buys_and_earlier_legs_both_count():
    context = make_ctx(buying_power=D("5000"), non_marginable_buying_power=D("5000"), cash=D("1950"),
                       open_orders=[make_order("VOO", "buy", limit_price=D("100"))])
    first, second = evaluate([buy("AAPL", 2, limit="200"), buy("AAPL", 5, limit="200")], context,
                             Mandate(max_position_pct=D("100")))
    assert first.verdict == "accept"
    assert second.verdict == "clamp" and second.est_cost_usd == D("450")  # 1,950 - 100 - 1,000 - 400
    assert reason(second, "buying_power").message.endswith(
        "(after $100.00 in open buy orders and $400.00 committed by earlier legs)"
    )


def test_the_cash_buffer_is_a_share_of_equity_not_of_cash():
    # The label says "% of equity kept in cash": equity 100k, cash 30k, buffer 20% -> keep 20k,
    # spend 10k (20% of the cash figure would have allowed 24k).
    context = make_ctx(equity=D("100000"), cash=D("30000"), buying_power=D("30000"),
                       non_marginable_buying_power=D("30000"), positions=[make_position("MSFT", 175, 400)])
    [d] = evaluate([buy("AAPL", 150, limit="200")], context,
                   Mandate(max_position_pct=D("100"), max_positions=20, cash_buffer_pct=D("20")))
    assert d.verdict == "clamp" and d.est_cost_usd == D("10000")


def test_a_buffer_that_takes_all_the_cash_rejects_buys_but_not_sells():
    context = ctx_with_position(2, buying_power=D("1000"), non_marginable_buying_power=D("1000"),
                                cash=D("1000"))  # 5% of $20,000 equity is the whole $1,000
    bought, sold = evaluate([buy("AAPL", 1, limit="200")], context, Mandate()), evaluate(
        [sell("VOO", 1, limit="500")], context, Mandate())
    assert bought[0].verdict == "reject" and codes(bought[0]) == ["buying_power"]
    assert reason(bought[0], "buying_power").message == (
        "Estimated cost $200.00 exceeds usable buying power $0.00: nothing is left once 5% of "
        "equity ($1,000.00) is kept in cash"
    )
    assert sold[0].verdict == "accept"


def test_negative_cash_rejects_every_buy():
    context = make_ctx(buying_power=D("5000"), non_marginable_buying_power=D("5000"), cash=D("-100"))
    [d] = evaluate([buy("AAPL", 1, limit="200")], context, Mandate(cash_buffer_pct=D("0")))
    assert d.verdict == "reject" and codes(d) == ["buying_power"]


def test_an_unpriced_pending_open_buy_blocks_buys_fail_closed():
    # NOQ has no quote and no position: what that open market buy commits is unknown, so the cash
    # left can't be known either.
    order = make_order("NOQ", "buy", type="market", limit_price=None)
    context = make_ctx(open_orders=[order], **_RICH)
    [d] = evaluate([buy("AAPL", 1, limit="200")], context, Mandate())
    assert d.verdict == "reject" and codes(d) == ["buying_power"]
    assert reason(d, "buying_power").message == (
        "Cannot check buying power: a pending open buy order for NOQ has no price and no quote"
    )


# --- budget -----------------------------------------------------------------------------------

_S_ASSETS = {**ASSETS, **{f"S{i}": make_asset(f"S{i}") for i in range(8)}}
_S_QUOTES = {**QUOTES, **{f"S{i}": make_quote(f"S{i}", last="100", bid="99.9", ask="100.1")
                          for i in range(8)}}


def test_budget_caps_total_across_legs():
    # Review probe 19: eight $1,000 buys against a $5,000 budget.
    legs = [buy(f"S{i}", 10, limit="100") for i in range(8)]
    ds = evaluate(legs, make_ctx(assets=_S_ASSETS, quotes=_S_QUOTES, **_RICH),
                  Mandate(budget_usd=D("5000")))
    assert [d.verdict for d in ds] == ["accept"] * 5 + ["reject"] * 3
    assert all(codes(d) == ["budget"] for d in ds[5:])
    assert sum(d.est_cost_usd for d in ds if d.verdict != "reject") <= D("5000")


def test_budget_counts_positions_and_pending_open_buys():
    # $4,000 of AAPL held + $500 of AAPL on order leaves $500 of a $5,000 budget.
    order = make_order("AAPL", "buy", qty=D("2.5"), limit_price=D("200"))
    context = ctx_with_position(20, "AAPL", "200", open_orders=[order], **_RICH)
    [d] = evaluate([buy("VOO", 2, limit="500")], context,
                   Mandate(budget_usd=D("5000"), max_position_pct=D("100")))
    assert d.verdict == "clamp" and d.leg.qty == D("1")
    assert reason(d, "budget").message == (
        "Clamped qty 2 → 1: invested $4,500.00 (positions, open buy orders and earlier legs) + "
        "estimated cost $1,000.00 exceeds the mandate's budget $5,000.00"
    )


def test_an_unpriceable_pending_buy_fails_closed_with_or_without_a_budget():
    # The cash it commits is unknown, so the buying-power rule (which runs before `budget`) blocks
    # every buy, budget or not (final review A-I1; it used to pass without a budget).
    order = make_order("NOQ", "buy", qty=D("5"), type="market", limit_price=None)
    context = make_ctx(open_orders=[order], **_RICH)
    [d] = evaluate([buy("VOO", 1, limit="500")], context, Mandate(budget_usd=D("5000")))
    assert d.verdict == "reject" and codes(d) == ["buying_power"]
    [no_budget] = evaluate([buy("VOO", 1, limit="500")], context, Mandate())
    assert no_budget.verdict == "reject" and codes(no_budget) == ["buying_power"]


# --- minimum sizes and plain-form output ------------------------------------------------------


def test_notional_invalid_messages():
    small, sub_cent = evaluate([buy("AAPL", notional="0.5"), buy("AAPL", notional="10.555")],
                               make_ctx(), Mandate())
    assert reason(small, "notional_invalid").message == "Notional $0.50 is below the $1.00 minimum"
    assert reason(sub_cent, "notional_invalid").message == (
        "Notional $10.555 has more than 2 decimal places"
    )


def test_fractional_clamp_below_one_dollar_rejects():
    # Review probe 11: $0.50 of cash would clamp 10 AAPL to 0.0025 shares ($0.50).
    [d] = evaluate([buy("AAPL", 10, limit="200")], make_ctx(buying_power=D("0.50")),
                   Mandate(cash_buffer_pct=D("0")))
    assert d.verdict == "reject" and codes(d) == ["buying_power"]
    assert reason(d, "buying_power").message == (
        "Estimated cost $2,000.00 exceeds usable buying power $0.50, and the clamped qty 0.0025 "
        "would cost $0.50, below the $1.00 minimum"
    )


def test_whole_share_clamp_below_one_dollar_rejects_with_the_clamping_code():
    # $0.90 cap on a $0.50 stock fits exactly 1 share = $0.50: under the $1.00 floor.
    [d] = evaluate([buy("PNY", 100, limit="0.5")], make_ctx(), Mandate(max_order_usd=D("0.90")))
    assert d.verdict == "reject" and codes(d) == ["max_order_usd"]


def test_tiny_original_qty_is_rejected_as_below_minimum_size():
    # Fix round 1 open item A: 1e-9 AAPL at $200 is $0.0000002.
    [d] = evaluate([buy("AAPL", "0.000000001", limit="200")], make_ctx(), Mandate())
    assert d.verdict == "reject" and codes(d) == ["notional_invalid"]
    assert reason(d, "notional_invalid").message == (
        "Estimated cost $0.0000002 of 0.000000001 AAPL is below the $1.00 minimum order size"
    )


def test_tick_rounding_that_pushes_cost_below_one_dollar_rejects():
    # 3 x $0.33334 = $1.00002 as proposed, but the buy tick floors it to 3 x $0.3333 = $0.9999.
    [d] = evaluate([buy("PNY", 3, limit="0.33334")], make_ctx(), Mandate())
    assert d.verdict == "reject" and codes(d) == ["bad_tick"]
    assert reason(d, "bad_tick").message == (
        "Rounding the limit price $0.33334 down to $0.3333 would make the estimated cost "
        "$0.9999, below the $1.00 minimum order size"
    )


def test_no_accepted_or_clamped_leg_costs_less_than_one_dollar():
    # Sweep sizes, prices and caps around the $1.00 floor; whatever survives must be >= $1.00.
    # (No leg here sells a whole position - that is the one exemption, tested separately below.)
    context = ctx_with_position(1000, "PNY", "0.5")
    legs = [
        make(symbol, qty, limit=limit)
        for make in (buy, sell)
        for symbol, limits in (("PNY", (None, "0.4999", "0.50005", "0.33334")),
                               ("AAPL", (None, "199", "200.001")))
        for qty in ("0.001", "0.004", "0.005", "0.006", "1", "2", "3", "7")
        for limit in limits
        if not (make is sell and symbol == "AAPL")  # no AAPL position to sell
    ]
    survivors = 0
    for mandate_ in (Mandate(), Mandate(max_order_usd=D("1.20")), Mandate(max_order_usd=D("0.90"))):
        for context_ in (context, ctx_with_position(1000, "PNY", "0.5", buying_power=D("1001.10"))):
            for leg in legs:
                [d] = evaluate([leg], context_, mandate_)
                if d.verdict != "reject":
                    survivors += 1
                    assert d.est_cost_usd >= D("1.00"), (leg, mandate_, d)
    assert survivors > 50  # the sweep really exercised accepted and clamped legs


_FRACTIONAL_PNY = {**ASSETS, "PNY": make_asset("PNY")}


@pytest.mark.parametrize(
    "leg, context",
    [
        pytest.param(sell("PNY", 1), ctx_with_position(1, "PNY", "0.5"), id="1-PNY-market"),
        pytest.param(sell("PNY", 1, limit="0.5"), ctx_with_position(1, "PNY", "0.5"),
                     id="1-PNY-limit"),
        pytest.param(sell("AAPL", "0.003"), ctx_with_position("0.003", "AAPL", "200"),
                     id="0.003-AAPL-worth-60c"),
    ],
)
def test_selling_the_whole_position_is_exempt_from_the_one_dollar_floor(leg, context):
    [d] = evaluate([leg], context, Mandate())
    assert d.verdict == "accept" and d.reasons == []
    assert d.est_cost_usd < D("1.00")


def test_selling_part_of_a_sub_dollar_position_is_still_rejected():
    [d] = evaluate([sell("PNY", "0.5")], ctx_with_position(1, "PNY", "0.5", assets=_FRACTIONAL_PNY),
                   Mandate())
    assert d.verdict == "reject" and codes(d) == ["notional_invalid"]
    [whole_share] = evaluate([sell("PNY", "0.5")], ctx_with_position(1, "PNY", "0.5"), Mandate())
    assert whole_share.verdict == "reject"  # PNY is whole-share in the default fixture


def test_notional_sell_of_the_whole_position_is_exempt():
    quotes = {**QUOTES, "PNY": make_quote("PNY", last="0.50", bid="0.50", ask="0.51")}
    context = ctx_with_position(1, "PNY", "0.5", assets=_FRACTIONAL_PNY, quotes=quotes)
    [whole] = evaluate([sell("PNY", notional="0.50")], context, Mandate())
    assert whole.verdict == "accept"  # $0.50 / min($0.50, $0.50) = exactly the 1 share held
    [part] = evaluate([sell("PNY", notional="0.40")], context, Mandate())
    assert part.verdict == "reject" and codes(part) == ["notional_invalid"]


def test_closing_sell_uses_the_working_position():
    # 1.003 AAPL: after selling 1 in the same ticket, the remaining 0.003 ($0.60) can be closed.
    ds = evaluate([sell("AAPL", 1), sell("AAPL", "0.003")], ctx_with_position("1.003", "AAPL", "200"),
                  Mandate())
    assert [d.verdict for d in ds] == ["accept", "accept"]
    [alone] = evaluate([sell("AAPL", "0.003")], ctx_with_position("1.003", "AAPL", "200"), Mandate())
    assert alone.verdict == "reject" and codes(alone) == ["notional_invalid"]


def test_closing_sell_is_tick_rounded_not_rejected():
    [d] = evaluate([sell("PNY", 1, limit="0.49915")], ctx_with_position(1, "PNY", "0.5"), Mandate())
    assert d.verdict == "clamp" and d.leg.limit_price == D("0.4992")
    assert codes(d) == ["bad_tick"]


# --- precision --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leg",
    [
        pytest.param(buy("AAPL", "1." + "0" * 39 + "1", limit="200"), id="qty-41-digits"),
        pytest.param(buy("AAPL", "0." + "9" * 41, limit="200"), id="qty-41-nines-would-round-up"),
        pytest.param(buy("AAPL", notional="5." + "0" * 39 + "1"), id="notional-41-digits"),
        pytest.param(buy("VOO", 1, limit="500." + "0" * 39 + "1"), id="limit-41-digits"),
    ],
)
def test_overlong_values_are_rejected_never_rounded(leg):
    # Review round 2 probe: at 40 working digits these used to normalize silently (0.99..9 -> 1).
    [d] = evaluate([leg], make_ctx(**_RICH), Mandate())
    assert d.verdict == "reject" and codes(d) == ["precision"]
    assert d.leg == d.original == leg


def test_precision_boundary_is_twenty_digits():
    [ok] = evaluate([buy("VOO", "1" + "0" * 19, limit="500")], make_ctx(), Mandate())  # 20 digits
    assert ok.verdict == "clamp" and ok.leg.qty == D("1")  # clamped, not rejected
    assert codes(ok) == ["max_position_pct", "buying_power"]
    [too_long] = evaluate([buy("VOO", "1" + "0" * 20, limit="500")], make_ctx(), Mandate())
    assert too_long.verdict == "reject" and codes(too_long) == ["precision"]
    assert reason(too_long, "precision").message == (
        "qty 100000000000000000000: more than 20 digits, too precise to check exactly"
    )


def test_fractional_clamp_of_exactly_one_dollar_is_allowed():
    [d] = evaluate([buy("AAPL", 10, limit="200")], make_ctx(buying_power=D("1.00")),
                   Mandate(cash_buffer_pct=D("0")))
    assert d.verdict == "clamp" and d.leg.qty == D("0.005")


def test_accepted_legs_come_back_in_plain_form():
    # Review probe 5: exponent-form Decimals are value-equal but serialize badly.
    leg = OrderLeg(symbol="VOO", side="buy", qty=D("1E+1"), type="limit", limit_price=D("5E+2"))
    [d] = evaluate([leg], make_ctx(**_RICH), Mandate(max_position_pct=D("100")))
    assert d.verdict == "accept" and d.leg == d.original
    assert d.leg.model_dump(mode="json")["qty"] == "10"
    assert d.leg.model_dump(mode="json")["limit_price"] == "500.00"
    [n] = evaluate([buy("AAPL", notional="1E+2")], make_ctx(), Mandate())
    assert n.verdict == "accept" and n.leg.model_dump(mode="json")["notional"] == "100.00"
    [sub] = evaluate([buy("PNY", 10, limit="0.5")], make_ctx(), Mandate())
    assert sub.leg.model_dump(mode="json")["limit_price"] == "0.5000"


def test_integer_qty_with_trailing_zeros_is_whole():
    # Review probe 20: 1.0000000000 (10 zero decimals) is 1 share, not a fractional/precision error.
    leg = OrderLeg(symbol="VOO", side="buy", qty=D("1.0000000000"), type="limit",
                   limit_price=D("500"))
    [d] = evaluate([leg], make_ctx(), Mandate())
    assert d.verdict == "accept" and d.leg.model_dump(mode="json")["qty"] == "1"


def test_whole_share_asset_rejects_any_nonzero_fraction():
    leg = OrderLeg(symbol="VOO", side="buy", qty=D("1.00000000001"), type="limit",
                   limit_price=D("500"))
    [d] = evaluate([leg], make_ctx(), Mandate())
    assert d.verdict == "reject" and codes(d) == ["fractional_not_allowed"]


# --- warnings ---------------------------------------------------------------------------------


def test_market_closed_message():
    [d] = evaluate([buy("VOO", 1, limit="500")], make_ctx(is_open=False), Mandate())
    assert reason(d, "market_closed").message == (
        "The market is closed; this order queues until the next open (2026-09-28T09:30:00-04:00)"
    )


def test_large_order_is_strictly_above_ten_percent_of_equity():
    [at] = evaluate([buy("VOO", 4, limit="500")], make_ctx(**_RICH), Mandate())
    assert at.verdict == "accept" and at.reasons == []  # $2,000 is exactly 10%
    [above] = evaluate([buy("VOO", 5, limit="500")], make_ctx(**_RICH), Mandate())
    assert above.verdict == "accept" and codes(above) == ["large_order"]
    assert reason(above, "large_order").message == (
        "Large order: estimated cost $2,500.00 is more than 10% of equity $20,000.00"
    )


# --- overall ---------------------------------------------------------------------------------


def _decision(verdict: str, *reasons: RiskReason) -> RiskDecision:
    leg = buy("VOO", 1, limit="500")
    return RiskDecision(index=0, verdict=verdict, leg=leg, original=leg, reasons=list(reasons))


def test_overall_outcomes():
    warn = RiskReason(code="market_closed", message="closed", severity="warn")
    info = RiskReason(code="note", message="fyi", severity="info")
    block = RiskReason(code="short_sell", message="short", severity="block")
    assert overall([]) == "blocked"
    assert overall([_decision("reject", block), _decision("reject", block)]) == "blocked"
    assert overall([_decision("accept"), _decision("reject", block)]) == "warnings"
    assert overall([_decision("clamp", warn)]) == "warnings"
    assert overall([_decision("accept", warn)]) == "warnings"
    assert overall([_decision("accept", info)]) == "ok"
    assert overall([_decision("accept"), _decision("accept")]) == "ok"


def test_overall_on_real_decisions(ctx, mandate):
    assert overall(evaluate([buy("VOO", 1, limit="500")], ctx, mandate)) == "ok"
    assert overall(evaluate([buy("VOO", 5, limit="500")], ctx, mandate)) == "warnings"
    assert overall(evaluate([sell("VOO", 1)], ctx, mandate)) == "blocked"


# --- purity -----------------------------------------------------------------------------------


def _scenario():
    legs = [
        buy("PNY", 100, limit="0.49995"),   # sub-penny tick clamp
        buy("AAPL", 10, limit="201"),       # fractional clamp to the $400 max order
        sell("VOO", 3),                     # short sell: only 2 held
        buy("AAPL", notional="600"),        # notional clamp to the $400 max order
        buy("NOQ", 1),                      # no quote
    ]
    return legs, ctx_with_position(2, "VOO", "500", is_open=False), Mandate(max_order_usd=D("400"))


def test_evaluate_is_deterministic():
    legs, context, mandate_ = _scenario()
    first = [d.model_dump(mode="json") for d in evaluate(legs, context, mandate_)]
    second = [d.model_dump(mode="json") for d in evaluate(legs, context, mandate_)]
    with localcontext() as lc:  # an odd ambient decimal context must not change anything
        lc.prec = 3
        lc.rounding = ROUND_UP
        third = [d.model_dump(mode="json") for d in evaluate(legs, context, mandate_)]
    assert first == second == third
    assert [d["verdict"] for d in first] == ["clamp", "clamp", "reject", "clamp", "reject"]


def test_evaluate_never_mutates_or_aliases_its_inputs():
    legs, context, mandate_ = _scenario()
    before = (context.model_dump(mode="json"), mandate_.model_dump(mode="json"),
              [leg.model_dump(mode="json") for leg in legs])
    decisions = evaluate(legs, context, mandate_)
    after = (context.model_dump(mode="json"), mandate_.model_dump(mode="json"),
             [leg.model_dump(mode="json") for leg in legs])
    assert before == after
    for d, leg in zip(decisions, legs):
        assert d.original == leg
        assert d.original is not leg and d.leg is not leg


def test_money_outputs_are_decimal():
    legs, context, mandate_ = _scenario()
    for d in evaluate(legs, context, mandate_):
        for value in (d.est_price, d.est_cost_usd, d.leg.qty, d.leg.notional, d.leg.limit_price):
            assert value is None or isinstance(value, Decimal)
