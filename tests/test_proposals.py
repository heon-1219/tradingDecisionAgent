"""Tests for `finagent.trading.proposals` - `canonical_hash` and `TicketStore`.

Fixture world: a single in-memory `AppDB`, a controllable `clock` (a mutable holder so tests can
advance "now" deterministically), and small `OrderLeg`/`RiskDecision` factories mirroring
`test_trading_models.py`'s style.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finagent.store.appdb import AppDB
from finagent.trading.models import OrderLeg, RiskDecision, RiskReason, Ticket, TicketStatus
from finagent.trading.proposals import TicketStateError, TicketStore, canonical_hash

D = Decimal
FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    """A controllable, injectable clock: advances only when a test tells it to."""

    def __init__(self, now: datetime = FIXED_NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_leg(symbol="AAPL", side="buy", qty="1", **over) -> OrderLeg:
    fields = dict(symbol=symbol, side=side, qty=D(qty), type="market")
    fields.update(over)
    return OrderLeg(**fields)


def make_decision(leg: OrderLeg | None = None, *, index=0, verdict="accept", reasons=None,
                   est_price="150", est_cost_usd="150", original=None) -> RiskDecision:
    leg = leg or make_leg()
    return RiskDecision(
        index=index,
        verdict=verdict,
        leg=leg,
        original=original or leg,
        reasons=reasons or [],
        est_price=None if est_price is None else D(est_price),
        est_cost_usd=None if est_cost_usd is None else D(est_cost_usd),
    )


def make_store(clock: Clock | None = None) -> tuple[TicketStore, AppDB, Clock]:
    db = AppDB(":memory:")
    clock = clock or Clock()
    return TicketStore(db, clock=clock), db, clock


def _create(store: TicketStore, session_id="sess-1", legs=None, decisions=None, **over) -> Ticket:
    legs = legs if legs is not None else [make_leg()]
    decisions = decisions if decisions is not None else [make_decision(legs[0])]
    fields = dict(
        thesis="thesis",
        risks="risks",
        invalidation="invalidation",
        evidence_ids=["E1"],
        mandate_version=1,
        quote_as_of=FIXED_NOW,
        ttl_sec=300,
    )
    fields.update(over)
    return store.create(session_id, legs, decisions, **fields)


# --- canonical_hash -----------------------------------------------------------------------


def test_canonical_hash_is_64_char_hex():
    leg = make_leg()
    decision = make_decision(leg)
    digest = canonical_hash([leg], [decision], "T0000000000")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_canonical_hash_deterministic_for_same_inputs():
    leg = make_leg()
    decision = make_decision(leg)
    d1 = canonical_hash([leg], [decision], "T0000000000")
    d2 = canonical_hash([leg], [decision], "T0000000000")
    assert d1 == d2


def test_canonical_hash_stable_across_process_restart_equivalent_objects():
    # Rebuild fresh, equal-but-not-identical objects (as a new process would) and confirm the
    # hash comes out the same - i.e. it depends only on values, never object identity or ordering
    # accidents.
    leg_a = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    leg_b = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="market")
    decision_a = RiskDecision(index=0, verdict="accept", leg=leg_a, original=leg_a, reasons=[],
                               est_price=Decimal("150"), est_cost_usd=Decimal("150"))
    decision_b = RiskDecision(index=0, verdict="accept", leg=leg_b, original=leg_b, reasons=[],
                               est_price=Decimal("150"), est_cost_usd=Decimal("150"))
    assert canonical_hash([leg_a], [decision_a], "TABC") == canonical_hash([leg_b], [decision_b], "TABC")


def test_canonical_hash_changes_with_ticket_id():
    leg = make_leg()
    decision = make_decision(leg)
    assert canonical_hash([leg], [decision], "TAAA") != canonical_hash([leg], [decision], "TBBB")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda leg: leg.model_copy(update={"qty": D("2")}),
        lambda leg: leg.model_copy(update={"symbol": "MSFT"}),
        lambda leg: leg.model_copy(update={"side": "sell"}),
        lambda leg: leg.model_copy(update={"type": "limit", "limit_price": D("10")}),
    ],
)
def test_canonical_hash_changes_when_leg_field_changes(mutate):
    leg = make_leg()
    decision = make_decision(leg)
    base = canonical_hash([leg], [decision], "T0000000000")
    mutated_leg = mutate(leg)
    mutated_decision = make_decision(mutated_leg)
    assert canonical_hash([mutated_leg], [mutated_decision], "T0000000000") != base


def test_canonical_hash_changes_when_verdict_changes():
    leg = make_leg()
    base = canonical_hash([leg], [make_decision(leg, verdict="accept")], "T0000000000")
    clamped = canonical_hash([leg], [make_decision(leg, verdict="clamp")], "T0000000000")
    assert base != clamped


def test_canonical_hash_changes_when_est_price_or_cost_changes():
    leg = make_leg()
    base = canonical_hash([leg], [make_decision(leg, est_price="150", est_cost_usd="150")], "T1")
    other_price = canonical_hash([leg], [make_decision(leg, est_price="151", est_cost_usd="150")], "T1")
    other_cost = canonical_hash([leg], [make_decision(leg, est_price="150", est_cost_usd="151")], "T1")
    assert base != other_price
    assert base != other_cost


def test_canonical_hash_unaffected_by_reason_message_wording():
    leg = make_leg()
    reasons_a = [RiskReason(code="large_order", message="Original wording", severity="warn")]
    reasons_b = [RiskReason(code="large_order", message="Totally different wording!", severity="warn")]
    a = canonical_hash([leg], [make_decision(leg, verdict="clamp", reasons=reasons_a)], "T1")
    b = canonical_hash([leg], [make_decision(leg, verdict="clamp", reasons=reasons_b)], "T1")
    assert a == b


def test_canonical_hash_changes_when_reason_code_or_severity_changes():
    leg = make_leg()
    base_reasons = [RiskReason(code="large_order", message="m", severity="warn")]
    other_code = [RiskReason(code="max_positions", message="m", severity="warn")]
    other_sev = [RiskReason(code="large_order", message="m", severity="block")]
    base = canonical_hash([leg], [make_decision(leg, verdict="clamp", reasons=base_reasons)], "T1")
    assert base != canonical_hash([leg], [make_decision(leg, verdict="clamp", reasons=other_code)], "T1")
    assert base != canonical_hash([leg], [make_decision(leg, verdict="clamp", reasons=other_sev)], "T1")


# --- TicketStore.create --------------------------------------------------------------------


def test_create_returns_valid_proposed_ticket():
    store, _db, _clock = make_store()
    ticket = _create(store)
    assert ticket.status == TicketStatus.proposed
    assert ticket.id.startswith("T")
    assert len(ticket.id) == 11  # "T" + 10 base32 chars
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in ticket.id[1:])
    assert ticket.client_order_ids == [f"fa-{ticket.id}-0"]
    assert ticket.created_at == FIXED_NOW
    assert ticket.expires_at == FIXED_NOW + timedelta(seconds=300)
    assert len(ticket.hash) == 64


def test_create_generates_one_client_order_id_per_leg():
    store, _db, _clock = make_store()
    legs = [make_leg(symbol="AAPL"), make_leg(symbol="MSFT")]
    decisions = [make_decision(legs[0], index=0), make_decision(legs[1], index=1)]
    ticket = _create(store, legs=legs, decisions=decisions)
    assert ticket.client_order_ids == [f"fa-{ticket.id}-0", f"fa-{ticket.id}-1"]


def test_create_supersedes_previous_active_ticket():
    store, db, _clock = make_store()
    first = _create(store, session_id="sess-1")
    second = _create(store, session_id="sess-1")

    superseded = store.get(first.id)
    assert superseded.status == TicketStatus.superseded

    row = db.query("SELECT status FROM tickets WHERE id=?", (first.id,))[0]
    assert row["status"] == TicketStatus.superseded.value

    current = store.active("sess-1")
    assert current.id == second.id


def test_create_does_not_supersede_a_different_sessions_ticket():
    store, _db, _clock = make_store()
    other = _create(store, session_id="sess-other")
    _create(store, session_id="sess-1")

    still_active = store.get(other.id)
    assert still_active.status == TicketStatus.proposed


def test_create_does_not_supersede_a_non_proposed_ticket():
    store, _db, _clock = make_store()
    first = _create(store, session_id="sess-1")
    store.transition(first.id, {TicketStatus.proposed}, TicketStatus.submitting)
    _create(store, session_id="sess-1")  # should not touch the submitting ticket

    unaffected = store.get(first.id)
    assert unaffected.status == TicketStatus.submitting


# --- TicketStore.get / active --------------------------------------------------------------


def test_get_returns_none_for_unknown_id():
    store, _db, _clock = make_store()
    assert store.get("Tdoesnotexist") is None


def test_get_json_round_trip_equals_created_ticket():
    store, _db, _clock = make_store()
    created = _create(store)
    fetched = store.get(created.id)
    assert fetched == created


def test_active_returns_none_when_no_ticket():
    store, _db, _clock = make_store()
    assert store.active("sess-none") is None


def test_active_ticket_is_not_expired():
    clock = Clock()
    store, _db, _ = make_store(clock)
    ticket = _create(store, session_id="sess-1", ttl_sec=300)
    clock.now = FIXED_NOW + timedelta(seconds=299)
    assert store.active("sess-1").id == ticket.id


def test_expired_ticket_is_not_active():
    clock = Clock()
    store, _db, _ = make_store(clock)
    _create(store, session_id="sess-1", ttl_sec=300)
    clock.now = FIXED_NOW + timedelta(seconds=301)
    assert store.active("sess-1") is None


def test_active_ignores_non_proposed_tickets():
    store, _db, clock = make_store()
    ticket = _create(store, session_id="sess-1")
    store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.rejected_by_user)
    assert store.active("sess-1") is None


# --- TicketStore.transition -----------------------------------------------------------------


def test_transition_success_updates_status_and_returns_new_ticket():
    store, db, _clock = make_store()
    ticket = _create(store)
    updated = store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
    assert updated.status == TicketStatus.submitting
    assert updated.id == ticket.id

    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.submitting.value
    reloaded = store.get(ticket.id)
    assert reloaded.status == TicketStatus.submitting


def test_transition_can_set_broker_order_ids():
    store, _db, _clock = make_store()
    ticket = _create(store)
    updated = store.transition(
        ticket.id, {TicketStatus.proposed}, TicketStatus.submitting,
        broker_order_ids=["b1"],
    )
    assert updated.broker_order_ids == ["b1"]
    assert store.get(ticket.id).broker_order_ids == ["b1"]


def test_transition_rejects_unknown_fields():
    store, _db, _clock = make_store()
    ticket = _create(store)
    with pytest.raises(ValueError):
        store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting, status="submitted")


def test_transition_wrong_expect_raises_and_leaves_state_unchanged():
    store, db, _clock = make_store()
    ticket = _create(store)
    with pytest.raises(TicketStateError):
        store.transition(ticket.id, {TicketStatus.submitted}, TicketStatus.submitting)

    row = db.query("SELECT status, body FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.proposed.value
    unchanged = store.get(ticket.id)
    assert unchanged == ticket


def test_transition_unknown_id_raises():
    store, _db, _clock = make_store()
    with pytest.raises(TicketStateError):
        store.transition("Tdoesnotexist", {TicketStatus.proposed}, TicketStatus.submitting)


def test_transition_after_success_a_second_call_with_the_same_expect_fails():
    store, _db, _clock = make_store()
    ticket = _create(store)
    store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
    with pytest.raises(TicketStateError):
        store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)


async def test_concurrent_transition_race_exactly_one_succeeds():
    store, _db, _clock = make_store()
    ticket = _create(store)

    results = []

    def attempt():
        try:
            store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
            return "ok"
        except TicketStateError:
            return "fail"

    outcomes = await asyncio.gather(
        *(asyncio.to_thread(attempt) for _ in range(8))
    )
    assert outcomes.count("ok") == 1
    assert outcomes.count("fail") == 7

    final = store.get(ticket.id)
    assert final.status == TicketStatus.submitting


# --- TicketStore.expire_stale -----------------------------------------------------------------


def test_expire_stale_flips_expired_proposed_tickets_and_returns_count():
    clock = Clock()
    store, db, _ = make_store(clock)
    ticket = _create(store, session_id="sess-1", ttl_sec=300)
    clock.now = FIXED_NOW + timedelta(seconds=301)

    count = store.expire_stale()
    assert count == 1

    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.expired.value
    reloaded = store.get(ticket.id)
    assert reloaded.status == TicketStatus.expired


def test_expire_stale_leaves_unexpired_tickets_alone():
    clock = Clock()
    store, _db, _ = make_store(clock)
    ticket = _create(store, session_id="sess-1", ttl_sec=300)
    clock.now = FIXED_NOW + timedelta(seconds=10)

    count = store.expire_stale()
    assert count == 0
    assert store.get(ticket.id).status == TicketStatus.proposed


def test_expire_stale_ignores_non_proposed_tickets():
    clock = Clock()
    store, _db, _ = make_store(clock)
    ticket = _create(store, session_id="sess-1", ttl_sec=300)
    store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitted)
    clock.now = FIXED_NOW + timedelta(seconds=301)

    count = store.expire_stale()
    assert count == 0
    assert store.get(ticket.id).status == TicketStatus.submitted


def test_expire_stale_returns_count_across_multiple_sessions():
    clock = Clock()
    store, _db, _ = make_store(clock)
    _create(store, session_id="sess-1", ttl_sec=300)
    _create(store, session_id="sess-2", ttl_sec=300)
    clock.now = FIXED_NOW + timedelta(seconds=301)

    assert store.expire_stale() == 2
