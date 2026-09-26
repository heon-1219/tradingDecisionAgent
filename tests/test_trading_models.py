import pathlib
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finagent.trading.models import (  # noqa: E402
    Mandate,
    MarketContext,
    OrderLeg,
    RiskDecision,
    RiskReason,
    Ticket,
    TicketStatus,
)


# --- OrderLeg (Task 1.5, exercised here per the brief) ---------------------


def test_order_leg_rejects_both_qty_and_notional():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="aapl", side="buy", qty=Decimal("1"), notional=Decimal("100"), type="market")


def test_order_leg_rejects_neither_qty_nor_notional():
    with pytest.raises(ValidationError):
        OrderLeg(symbol="aapl", side="buy", type="market")


def test_order_leg_uppercases_symbol():
    leg = OrderLeg(symbol="aapl", side="buy", qty=Decimal("1"), type="market")
    assert leg.symbol == "AAPL"


# --- Mandate ----------------------------------------------------------------


def test_mandate_defaults():
    m = Mandate()
    assert m.direction_text == ""
    assert m.budget_usd is None
    assert m.universe == "us_equity"
    assert m.allow_symbols == []
    assert m.deny_symbols == []
    assert m.max_position_pct == Decimal("20")
    assert m.max_positions == 8
    assert m.cash_buffer_pct == Decimal("5")
    assert m.horizon == "weeks to months"
    assert m.order_types == ["limit", "market"]
    assert m.max_orders_per_day == 10
    assert m.max_order_usd is None
    assert m.price_band_pct == Decimal("3")
    assert m.allow_fractional is True
    assert m.version == 1


def test_mandate_normalizes_and_dedupes_allow_deny_symbols_preserving_order():
    m = Mandate(allow_symbols=["aapl", "MSFT", "aapl", "goog"], deny_symbols=["tsla", "tsla"])
    assert m.allow_symbols == ["AAPL", "MSFT", "GOOG"]
    assert m.deny_symbols == ["TSLA"]


def test_mandate_rejects_invalid_symbol_in_allow_list():
    with pytest.raises(ValidationError):
        Mandate(allow_symbols=["../etc"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_position_pct", Decimal("0")),
        ("max_position_pct", Decimal("100.01")),
        ("cash_buffer_pct", Decimal("-1")),
        ("cash_buffer_pct", Decimal("100.01")),
        ("price_band_pct", Decimal("0")),
        ("price_band_pct", Decimal("100.01")),
    ],
)
def test_mandate_percent_fields_out_of_bounds_rejected(field, value):
    with pytest.raises(ValidationError):
        Mandate(**{field: value})


def test_mandate_cash_buffer_pct_may_be_zero():
    m = Mandate(cash_buffer_pct=Decimal("0"))
    assert m.cash_buffer_pct == Decimal("0")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_positions", 0),
        ("max_orders_per_day", 0),
        ("version", 0),
    ],
)
def test_mandate_integer_lower_bounds_rejected(field, value):
    with pytest.raises(ValidationError):
        Mandate(**{field: value})


@pytest.mark.parametrize("field", ["budget_usd", "max_order_usd"])
def test_mandate_optional_usd_fields_must_be_positive_when_set(field):
    with pytest.raises(ValidationError):
        Mandate(**{field: Decimal("0")})
    with pytest.raises(ValidationError):
        Mandate(**{field: Decimal("-10")})
    # positive is fine
    Mandate(**{field: Decimal("10")})


def test_mandate_order_types_must_be_non_empty():
    with pytest.raises(ValidationError):
        Mandate(order_types=[])


def test_mandate_rejects_float_money_fields():
    with pytest.raises(ValidationError):
        Mandate(budget_usd=1000.0)
    with pytest.raises(ValidationError):
        Mandate(max_position_pct=20.0)


def test_mandate_json_round_trip_preserves_decimal_exactly():
    m = Mandate(
        budget_usd=Decimal("12345.6789"),
        max_position_pct=Decimal("17.5"),
        cash_buffer_pct=Decimal("2.25"),
        max_order_usd=Decimal("999.999"),
        price_band_pct=Decimal("3.001"),
        allow_symbols=["aapl"],
    )
    payload = m.model_dump_json()
    restored = Mandate.model_validate_json(payload)
    assert restored == m
    assert restored.budget_usd == m.budget_usd
    assert str(restored.max_position_pct) == str(m.max_position_pct)
    assert str(restored.cash_buffer_pct) == str(m.cash_buffer_pct)
    assert str(restored.max_order_usd) == str(m.max_order_usd)
    assert str(restored.price_band_pct) == str(m.price_band_pct)


# --- RiskReason / RiskDecision ----------------------------------------------


def _leg(symbol="AAPL", qty="1"):
    return OrderLeg(symbol=symbol, side="buy", qty=Decimal(qty), type="market")


def test_risk_reason_severity_literal():
    r = RiskReason(code="max_position_pct", message="clamped qty", severity="warn")
    assert r.severity == "warn"
    with pytest.raises(ValidationError):
        RiskReason(code="x", message="y", severity="critical")


def test_risk_decision_holds_original_and_clamped_leg():
    original = _leg(qty="10")
    clamped = _leg(qty="5")
    decision = RiskDecision(
        index=0,
        verdict="clamp",
        leg=clamped,
        original=original,
        reasons=[RiskReason(code="max_position_pct", message="clamped", severity="warn")],
        est_price=Decimal("150.00"),
        est_cost_usd=Decimal("750.00"),
    )
    assert decision.leg.qty == Decimal("5")
    assert decision.original.qty == Decimal("10")
    assert decision.verdict == "clamp"


# --- MarketContext -----------------------------------------------------------


def test_market_context_normalizes_dict_keys(monkeypatch=None):
    from finagent.trading.models import AccountSnapshot, ClockInfo

    account = AccountSnapshot(
        status="ACTIVE",
        equity=Decimal("1000"),
        cash=Decimal("1000"),
        buying_power=Decimal("1000"),
        non_marginable_buying_power=Decimal("1000"),
        last_equity=Decimal("1000"),
        trading_blocked=False,
        account_blocked=False,
        trade_suspended_by_user=False,
        shorting_enabled=False,
    )
    clock = ClockInfo(
        is_open=True,
        next_open="2026-01-01T00:00:00Z",
        next_close="2026-01-01T00:00:00Z",
        timestamp="2026-01-01T00:00:00Z",
    )
    ctx = MarketContext(
        account=account,
        positions={"aapl": _position()},
        open_orders=[],
        assets={"aapl": _asset()},
        quotes={"aapl": _quote()},
        clock=clock,
        orders_today=0,
    )
    assert list(ctx.positions.keys()) == ["AAPL"]
    assert list(ctx.assets.keys()) == ["AAPL"]
    assert list(ctx.quotes.keys()) == ["AAPL"]


def _position():
    from finagent.trading.models import PositionSnapshot

    return PositionSnapshot(
        symbol="AAPL",
        qty=Decimal("1"),
        qty_available=Decimal("1"),
        avg_entry_price=Decimal("150"),
        current_price=Decimal("150"),
        market_value=Decimal("150"),
        unrealized_pl=Decimal("0"),
        unrealized_plpc=Decimal("0"),
        side="long",
    )


def _asset():
    from finagent.trading.models import AssetInfo

    return AssetInfo(
        symbol="AAPL",
        **{"class": "us_equity"},
        status="active",
        tradable=True,
        fractionable=True,
        shortable=True,
        marginable=True,
    )


def _quote():
    from finagent.trading.models import Quote

    return Quote(symbol="AAPL", bid=Decimal("149.9"), ask=Decimal("150.1"), last=Decimal("150"), feed="iex")


# --- TicketStatus / Ticket ----------------------------------------------------


def test_ticket_status_members():
    expected = {
        "proposed",
        "submitting",
        "submitted",
        "rejected_by_user",
        "expired",
        "superseded",
        "failed",
    }
    assert {s.value for s in TicketStatus} == expected


def _ticket_kwargs(**overrides):
    now = datetime.now(timezone.utc)
    leg = _leg()
    decision = RiskDecision(
        index=0,
        verdict="accept",
        leg=leg,
        original=leg,
        reasons=[],
        est_price=Decimal("150"),
        est_cost_usd=Decimal("150"),
    )
    kwargs = dict(
        id="tic_1",
        session_id="sess_1",
        status=TicketStatus.proposed,
        legs=[leg],
        decisions=[decision],
        thesis="thesis",
        risks="risks",
        invalidation="invalidation",
        evidence_ids=["E1"],
        mandate_version=1,
        quote_as_of=now,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        hash="a" * 64,
        client_order_ids=["c1"],
    )
    kwargs.update(overrides)
    return kwargs


def test_ticket_valid_construction():
    t = Ticket(**_ticket_kwargs())
    assert t.status == TicketStatus.proposed
    assert t.broker_order_ids == []


def test_ticket_rejects_naive_datetime():
    naive = datetime.now()
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(quote_as_of=naive))
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(created_at=naive))
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(expires_at=naive))


def test_ticket_hash_must_be_64_char_lowercase_hex():
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(hash="A" * 64))  # uppercase not allowed
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(hash="a" * 63))  # too short
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(hash="z" * 64))  # not hex


def test_ticket_client_order_ids_length_must_match_legs():
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(client_order_ids=["c1", "c2"]))
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(client_order_ids=[]))


def test_ticket_broker_order_ids_length_must_match_legs_when_non_empty():
    # empty is fine (default / explicit)
    Ticket(**_ticket_kwargs(broker_order_ids=[]))
    # non-empty but wrong length is rejected
    with pytest.raises(ValidationError):
        Ticket(**_ticket_kwargs(broker_order_ids=["b1", "b2"]))
    # non-empty matching length is fine, including None entries
    t = Ticket(**_ticket_kwargs(broker_order_ids=[None]))
    assert t.broker_order_ids == [None]


def test_ticket_json_round_trip():
    t = Ticket(**_ticket_kwargs())
    payload = t.model_dump_json()
    restored = Ticket.model_validate_json(payload)
    assert restored == t
    assert restored.legs[0].qty == t.legs[0].qty
    assert restored.decisions[0].est_price == t.decisions[0].est_price
    assert restored.quote_as_of == t.quote_as_of
    assert restored.created_at == t.created_at
    assert restored.expires_at == t.expires_at
