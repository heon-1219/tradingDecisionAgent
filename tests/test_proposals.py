"""Tests for `finagent.trading.proposals` - `canonical_hash` and `TicketStore`.

Fixture world: a single in-memory `AppDB`, a controllable `clock` (a mutable holder so tests can
advance "now" deterministically), and small `OrderLeg`/`RiskDecision` factories mirroring
`test_trading_models.py`'s style.

Policy under test (fix round 1): a ticket's `legs` are exactly the *executable* legs -
`[d.leg for d in decisions if d.verdict != "reject"]`, in order; `create()` derives `legs` from
`decisions` itself and only cross-checks a caller-supplied `legs` against that derivation.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finagent.store.appdb import AppDB
from finagent.trading.models import OrderLeg, RiskDecision, RiskReason, Ticket, TicketStatus
from finagent.trading.proposals import TicketStateError, TicketStore, canonical_hash

D = Decimal
FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

_TERMINAL_STATUSES = [
    TicketStatus.expired,
    TicketStatus.superseded,
    TicketStatus.rejected_by_user,
    TicketStatus.failed,
]


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


def _create(store: TicketStore, session_id="sess-1", decisions=None, legs=None, **over) -> Ticket:
    decisions = decisions if decisions is not None else [make_decision(make_leg())]
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
    return store.create(session_id, decisions, legs, **fields)


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
        lambda leg: leg.model_copy(update={"time_in_force": "gtc"}),
        lambda leg: leg.model_copy(update={"limit_price": D("11")}),
    ],
)
def test_canonical_hash_changes_when_leg_field_changes(mutate):
    # A qty-based limit order (not notional), so both a bare time_in_force flip and a bare
    # limit_price change are independently valid mutations.
    leg = OrderLeg(symbol="AAPL", side="buy", qty=D("1"), type="limit", limit_price=D("10"))
    decision = make_decision(leg)
    base = canonical_hash([leg], [decision], "T0000000000")
    mutated_leg = mutate(leg)
    mutated_decision = make_decision(mutated_leg)
    assert canonical_hash([mutated_leg], [mutated_decision], "T0000000000") != base


def test_canonical_hash_changes_when_only_notional_changes():
    leg_a = OrderLeg(symbol="MSFT", side="buy", notional=D("100.00"), type="market")
    leg_b = leg_a.model_copy(update={"notional": D("200.00")})
    a = canonical_hash([leg_a], [make_decision(leg_a)], "T1")
    b = canonical_hash([leg_b], [make_decision(leg_b)], "T1")
    assert a != b


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


def test_canonical_hash_recomputed_from_stored_ticket_matches_hash():
    store, _db, _clock = make_store()
    created = _create(store)
    fetched = store.get(created.id)
    assert canonical_hash(fetched.legs, fetched.decisions, fetched.id) == created.hash


# --- TicketStore.create: legs/decisions policy ---------------------------------------------


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


def test_create_derives_legs_from_decisions_without_an_explicit_legs_argument():
    store, _db, _clock = make_store()
    leg = make_leg()
    ticket = _create(store, decisions=[make_decision(leg)])
    assert ticket.legs == [leg]


def test_create_excludes_rejected_decisions_from_legs():
    store, _db, _clock = make_store()
    kept = make_leg(symbol="AAPL")
    rejected_leg = make_leg(symbol="TSLA", side="sell")
    decisions = [
        make_decision(kept, index=0, verdict="accept"),
        make_decision(rejected_leg, index=1, verdict="reject"),
    ]
    ticket = _create(store, decisions=decisions)
    assert ticket.legs == [kept]
    assert len(ticket.decisions) == 2  # rejects are kept in decisions for the user to see
    assert ticket.client_order_ids == [f"fa-{ticket.id}-0"]  # one id per *executable* leg


def test_create_keeps_clamped_leg_not_original_in_legs():
    store, _db, _clock = make_store()
    original = make_leg(qty="1000")
    clamped = make_leg(qty="10")
    decision = RiskDecision(index=0, verdict="clamp", leg=clamped, original=original, reasons=[])
    ticket = _create(store, decisions=[decision])
    assert ticket.legs == [clamped]


def test_create_multi_leg_mixed_verdicts_preserves_order():
    store, _db, _clock = make_store()
    a = make_leg(symbol="AAPL")
    b = make_leg(symbol="MSFT")
    c = make_leg(symbol="TSLA")
    decisions = [
        make_decision(a, index=0, verdict="accept"),
        make_decision(b, index=1, verdict="reject"),
        make_decision(c, index=2, verdict="clamp"),
    ]
    ticket = _create(store, decisions=decisions)
    assert ticket.legs == [a, c]
    assert ticket.client_order_ids == [f"fa-{ticket.id}-0", f"fa-{ticket.id}-1"]


def test_create_rejects_when_every_decision_is_rejected():
    store, _db, _clock = make_store()
    leg = make_leg()
    decisions = [make_decision(leg, verdict="reject")]
    with pytest.raises(ValueError, match="no executable legs"):
        _create(store, decisions=decisions)
    assert store.active("sess-1") is None


def test_create_accepts_explicit_legs_matching_the_derived_list():
    store, _db, _clock = make_store()
    leg = make_leg()
    decisions = [make_decision(leg)]
    ticket = _create(store, decisions=decisions, legs=[leg])
    assert ticket.legs == [leg]


def test_create_rejects_explicit_legs_that_disagree_with_a_clamp():
    store, _db, _clock = make_store()
    original = make_leg(qty="1000")
    clamped = make_leg(qty="10")
    decision = RiskDecision(index=0, verdict="clamp", leg=clamped, original=original, reasons=[])
    with pytest.raises(ValueError, match="must equal the executable legs"):
        _create(store, decisions=[decision], legs=[original])


def test_create_rejects_explicit_legs_unrelated_to_an_all_rejected_decision_list():
    store, _db, _clock = make_store()
    rejected_source = make_leg(symbol="AAPL")
    unrelated = make_leg(symbol="TSLA", side="sell")
    decisions = [make_decision(rejected_source, verdict="reject")]
    with pytest.raises(ValueError):
        _create(store, decisions=decisions, legs=[unrelated])


def test_create_rejects_explicit_legs_with_wrong_count():
    store, _db, _clock = make_store()
    a = make_leg(symbol="AAPL")
    b = make_leg(symbol="MSFT")
    with pytest.raises(ValueError, match="must equal the executable legs"):
        _create(store, decisions=[make_decision(a)], legs=[a, b])


def test_create_rejects_decisions_index_mismatch():
    store, _db, _clock = make_store()
    leg = make_leg()
    decisions = [make_decision(leg, index=1)]  # only element must have index 0
    with pytest.raises(ValueError, match="index"):
        _create(store, decisions=decisions)
    assert store.active("sess-1") is None


def test_create_rejects_decisions_out_of_index_order():
    store, _db, _clock = make_store()
    a = make_leg(symbol="AAPL")
    b = make_leg(symbol="MSFT")
    decisions = [make_decision(a, index=1), make_decision(b, index=0)]
    with pytest.raises(ValueError, match="index"):
        _create(store, decisions=decisions)


def test_create_generates_one_client_order_id_per_executable_leg():
    store, _db, _clock = make_store()
    legs = [make_leg(symbol="AAPL"), make_leg(symbol="MSFT")]
    decisions = [make_decision(legs[0], index=0), make_decision(legs[1], index=1)]
    ticket = _create(store, decisions=decisions)
    assert ticket.client_order_ids == [f"fa-{ticket.id}-0", f"fa-{ticket.id}-1"]


# --- create(): round-trip validation (Important 5) --------------------------------------------


def _model_construct_bad_leg() -> OrderLeg:
    # A limit order with no limit_price is invalid; `model_construct` bypasses OrderLeg's own
    # validators, mirroring risk.py's documented fallback for a *rejected* leg (see risk.py's
    # `_decision`).
    return OrderLeg.model_construct(
        symbol="AAPL", side="buy", qty=D("1"), notional=None, type="limit",
        time_in_force="day", limit_price=None,
    )


def test_create_refuses_a_ticket_that_cannot_round_trip():
    store, db, _clock = make_store()
    good_leg = make_leg()
    bad_leg = _model_construct_bad_leg()
    poisoned = RiskDecision.model_construct(
        index=1, verdict="reject", leg=bad_leg, original=bad_leg,
        reasons=[RiskReason(code="limit_required", message="m", severity="block")],
        est_price=None, est_cost_usd=None,
    )
    decisions = [make_decision(good_leg, index=0), poisoned]
    with pytest.raises(ValueError, match="round-trip"):
        _create(store, decisions=decisions)
    # Nothing was written: no ticket row exists for this attempt.
    assert db.query("SELECT COUNT(*) AS n FROM tickets")[0]["n"] == 0
    assert store.active("sess-1") is None


def test_create_supersede_step_survives_a_pre_existing_corrupt_row():
    store, db, _clock = make_store()
    # Simulate a legacy/corrupt row bypassing TicketStore entirely (e.g. written before this
    # guard existed).
    db.execute(
        "INSERT INTO tickets (id, session_id, status, body, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        ("Tcorrupt00", "sess-1", "proposed", "{not valid json", FIXED_NOW.isoformat(),
         (FIXED_NOW + timedelta(seconds=300)).isoformat()),
    )
    # create() must not crash while trying to supersede the corrupt row.
    ticket = _create(store, session_id="sess-1")
    assert store.active("sess-1").id == ticket.id
    row = db.query("SELECT status FROM tickets WHERE id=?", ("Tcorrupt00",))[0]
    assert row["status"] == TicketStatus.superseded.value


def test_active_skips_a_corrupt_row_and_returns_the_good_ticket():
    store, db, _clock = make_store()
    good = _create(store, session_id="sess-1")
    db.execute(
        "INSERT INTO tickets (id, session_id, status, body, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        ("Tcorrupt01", "sess-2", "proposed", "{not valid json", FIXED_NOW.isoformat(),
         (FIXED_NOW + timedelta(seconds=300)).isoformat()),
    )
    assert store.active("sess-1").id == good.id
    assert store.active("sess-2") is None  # the only "proposed" row for sess-2 is unparseable


def test_expire_stale_skips_a_corrupt_row_and_still_expires_the_good_one():
    clock = Clock()
    store, db, _ = make_store(clock)
    good = _create(store, session_id="sess-1", ttl_sec=300)
    db.execute(
        "INSERT INTO tickets (id, session_id, status, body, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        ("Tcorrupt02", "sess-2", "proposed", "{not valid json", FIXED_NOW.isoformat(),
         FIXED_NOW.isoformat()),
    )
    clock.now = FIXED_NOW + timedelta(seconds=301)
    count = store.expire_stale()
    assert count == 1  # only the good, parseable ticket is counted
    assert store.get(good.id).status == TicketStatus.expired
    corrupt_row = db.query("SELECT status FROM tickets WHERE id=?", ("Tcorrupt02",))[0]
    assert corrupt_row["status"] == "proposed"  # left alone, not crashed on


# --- TicketStore.create: supersede ----------------------------------------------------------


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


def test_create_interleaved_with_a_concurrent_transition_does_not_clobber_it():
    """Regression for Important 2: a `transition()` that lands *between* create()'s supersede
    SELECT and its per-row UPDATE must not be overwritten back to `superseded`, and the ticket it
    moved to `submitting` must still be transitionable afterwards."""

    class HookDB(AppDB):
        hook = None

        def query(self, sql, params=()):
            rows = super().query(sql, params)
            if self.hook and sql.startswith("SELECT id, body FROM tickets WHERE session_id"):
                hook, self.hook = self.hook, None
                hook()
            return rows

    db = HookDB(":memory:")
    store = TicketStore(db, clock=lambda: FIXED_NOW)
    first = _create(store, session_id="sess-1")

    won = []
    db.hook = lambda: won.append(
        store.transition(first.id, {TicketStatus.proposed}, TicketStatus.submitting).status
    )
    _create(store, session_id="sess-1")  # its supersede SELECT triggers the hook above

    assert won == [TicketStatus.submitting]
    row = db.query("SELECT status FROM tickets WHERE id=?", (first.id,))[0]
    assert row["status"] == TicketStatus.submitting.value  # not clobbered to "superseded"
    assert store.get(first.id).status == TicketStatus.submitting

    # The executor must still be able to finish this ticket.
    finished = store.transition(first.id, {TicketStatus.submitting}, TicketStatus.submitted)
    assert finished.status == TicketStatus.submitted


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
    store, db, _clock = make_store()
    ticket = _create(store)
    with pytest.raises(ValueError):
        store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting, status="submitted")
    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.proposed.value  # nothing touched


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


def test_transition_empty_expect_always_raises():
    store, _db, _clock = make_store()
    ticket = _create(store)
    with pytest.raises(TicketStateError):
        store.transition(ticket.id, set(), TicketStatus.submitting)


def test_transition_never_allows_to_proposed():
    store, db, _clock = make_store()
    ticket = _create(store)
    store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.expired)
    with pytest.raises(ValueError):
        store.transition(ticket.id, {TicketStatus.expired}, TicketStatus.proposed)
    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.expired.value  # unchanged by the refused attempt


@pytest.mark.parametrize("terminal", _TERMINAL_STATUSES)
def test_transition_refuses_any_move_out_of_a_terminal_state(terminal):
    store, db, _clock = make_store()
    ticket = _create(store)
    # Force the ticket into the terminal state under test.
    if terminal == TicketStatus.superseded:
        _create(store, session_id="sess-1")  # supersedes `ticket`
    else:
        store.transition(ticket.id, {TicketStatus.proposed}, terminal)
    assert store.get(ticket.id).status == terminal

    with pytest.raises(TicketStateError):
        store.transition(ticket.id, {terminal}, TicketStatus.submitting)

    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == terminal.value  # still unchanged


def test_transition_validation_failure_leaves_row_completely_unchanged():
    """Regression for Important 1: a `Ticket` that fails validation (wrong-length
    broker_order_ids) must never partially commit - neither the status column nor the body."""
    store, db, _clock = make_store()
    ticket = _create(store)  # single leg
    with pytest.raises(Exception):
        store.transition(
            ticket.id, {TicketStatus.proposed}, TicketStatus.submitting,
            broker_order_ids=["a", "b"],  # 2 ids for 1 leg: invalid
        )
    row = db.query("SELECT status FROM tickets WHERE id=?", (ticket.id,))[0]
    assert row["status"] == TicketStatus.proposed.value
    assert store.get(ticket.id).status == TicketStatus.proposed
    assert store.active("sess-1").id == ticket.id  # not hidden/bricked


async def _run_concurrent_transition_race(store, ticket_id):
    def attempt():
        try:
            store.transition(ticket_id, {TicketStatus.proposed}, TicketStatus.submitting)
            return "ok"
        except TicketStateError:
            return "fail"

    return await asyncio.gather(*(asyncio.to_thread(attempt) for _ in range(8)))


async def test_concurrent_transition_race_exactly_one_succeeds_async():
    store, _db, _clock = make_store()
    ticket = _create(store)
    outcomes = await _run_concurrent_transition_race(store, ticket.id)
    assert outcomes.count("ok") == 1
    assert outcomes.count("fail") == 7

    final = store.get(ticket.id)
    assert final.status == TicketStatus.submitting


def test_concurrent_transition_race_with_real_threads_exactly_one_succeeds():
    """A genuine multi-thread variant (not asyncio.to_thread), exercising the shared `RLock`
    across real OS threads rather than the event loop's thread pool."""
    store, _db, _clock = make_store()
    ticket = _create(store)
    results: list[str] = []
    results_lock = threading.Lock()

    def attempt():
        try:
            store.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
            outcome = "ok"
        except TicketStateError:
            outcome = "fail"
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1
    assert results.count("fail") == 7
    assert store.get(ticket.id).status == TicketStatus.submitting


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
