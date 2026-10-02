"""Tests for `finagent.trading.execution.Executor` - the only code path that submits orders.

Fixture world: `fakes_trading.FakeBroker`/`FakeData` (see that module for the default account,
positions, assets and quotes), a real in-memory `AppDB` + `TicketStore` + `Ledger` (configured with
fake Alpaca credentials as known secrets), a controllable clock, and a recording `sleep` so polling
never actually waits. Tickets are proposed the same way the chat layer will: `build_context` ->
`risk.evaluate` -> `TicketStore.create`.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from fakes_trading import (
    FakeBroker,
    FakeData,
    make_account,
    make_asset,
    make_open_order,
    make_order,
    make_position,
)
from finagent.store.appdb import AppDB
from finagent.trading.broker import BrokerError, OrderUncertain
from finagent.trading.execution import (
    REQUOTE_DRIFT_PCT,
    ConfirmResult,
    Executor,
)
from finagent.trading.ledger import Ledger
from finagent.trading.models import Mandate, OrderLeg, OrderSnapshot, Ticket, TicketStatus
from finagent.trading.proposals import TicketStateError, TicketStore
from finagent.trading.risk import evaluate

D = Decimal
FIXED_NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)
SESSION = "sess-1"
FAKE_KEY_ID = "PKFAKEKEYID0123456789"
FAKE_SECRET = "fakeSecretValue-9f8e7d6c5b4a3210zyxw"


class Clock:
    def __init__(self, now: datetime = FIXED_NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class World:
    """Everything one test needs, wired together."""

    def __init__(self, *, broker: FakeBroker | None = None, data: FakeData | None = None):
        self.db = AppDB(":memory:")
        self.clock = Clock()
        self.tickets = TicketStore(self.db, clock=self.clock)
        self.ledger = Ledger(self.db, known_secrets=[FAKE_KEY_ID, FAKE_SECRET])
        self.broker = broker or FakeBroker()
        self.data = data or FakeData()
        self.settings = SimpleNamespace(trading_enabled=True, ticket_ttl_sec=300)
        self.sleeps: list[float] = []
        self.mandates: dict[str, Mandate] = {}
        self.mandate_calls: list[str] = []
        self.executor = Executor(
            self.broker,
            self.data,
            self.tickets,
            self.ledger,
            self.settings,
            mandate_for=self.mandate_for,
            sleep=self.sleep,
        )

    def mandate_for(self, session_id: str) -> Mandate:
        self.mandate_calls.append(session_id)
        return self.mandates.get(session_id, Mandate())

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    async def propose(self, legs: list[OrderLeg], session_id: str = SESSION) -> Ticket:
        ctx = await self.executor.build_context([leg.symbol for leg in legs])
        mandate = self.mandates.get(session_id, Mandate())
        decisions = evaluate(legs, ctx, mandate)
        return self.tickets.create(
            session_id,
            decisions=decisions,
            thesis="thesis",
            risks="risks",
            invalidation="invalidation",
            evidence_ids=["E1"],
            mandate_version=mandate.version,
            quote_as_of=self.clock(),
            ttl_sec=300,
        )

    def ledger_entries(self, session_id: str = SESSION) -> list[dict]:
        return [json.loads(line) for line in self.ledger.export_jsonl(session_id)]

    def ledger_kinds(self, session_id: str = SESSION) -> list[str]:
        return [entry["kind"] for entry in self.ledger_entries(session_id)]

    def order_rows(self) -> list[dict]:
        return self.db.query("SELECT * FROM orders ORDER BY id")


def aapl(qty: str = "10", limit: str = "200.00") -> OrderLeg:
    return OrderLeg(symbol="AAPL", side="buy", qty=D(qty), type="limit", limit_price=D(limit))


def key_market(qty: str = "100") -> OrderLeg:
    return OrderLeg(symbol="KEY", side="buy", qty=D(qty), type="market")


def msft_sell(qty: str = "5", limit: str = "400.00") -> OrderLeg:
    return OrderLeg(symbol="MSFT", side="sell", qty=D(qty), type="limit", limit_price=D(limit))


@pytest.fixture
def world() -> World:
    return World()


def test_requote_drift_threshold_is_one_percent():
    assert REQUOTE_DRIFT_PCT == D("1.0")


# --- happy path and polling -------------------------------------------------------------------


async def test_confirm_happy_path_submits_each_leg_once(world):
    ticket = await world.propose([aapl(), key_market()])
    assert [d.verdict for d in ticket.decisions] == ["accept", "accept"]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert isinstance(result, ConfirmResult)
    assert result.status == "submitted"
    assert result.new_ticket is None
    assert world.broker.submits == 2
    assert [cid for _leg, cid in world.broker.submitted] == ticket.client_order_ids
    assert [leg for leg, _cid in world.broker.submitted] == ticket.legs

    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted
    assert result.ticket == stored
    assert stored.broker_order_ids == [o.id for o in result.orders]
    assert [o.client_order_id for o in result.orders] == ticket.client_order_ids
    assert all(o.status == "accepted" for o in result.orders)
    assert world.broker.reconcile_calls == []

    rows = world.order_rows()
    assert [r["client_order_id"] for r in rows] == ticket.client_order_ids
    assert [r["broker_order_id"] for r in rows] == stored.broker_order_ids
    assert all(r["ticket_id"] == ticket.id for r in rows)
    assert OrderSnapshot.model_validate_json(rows[0]["body"]).symbol == "AAPL"

    assert "AAPL" in result.message and "KEY" in result.message
    kinds = world.ledger_kinds()
    for kind in ("confirm_request", "claim", "submit", "broker_response", "result"):
        assert kind in kinds
    assert kinds[0] == "confirm_request" and kinds[-1] == "result"
    assert kinds.count("submit") == 2 and kinds.count("broker_response") == 2


async def test_confirm_polls_until_a_stop_status_two_seconds_apart(world):
    world.broker.initial_status = "pending_new"
    ticket = await world.propose([aapl()])

    def script_first_order(broker, leg, cid):
        order = broker.place(leg, cid)
        broker.status_script[order.id] = ["pending_new", "pending_new", "filled"]
        return order

    world.broker.submit_actions = [script_first_order]
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert len(world.broker.get_order_calls) == 3
    assert world.sleeps == [2.0, 2.0, 2.0]
    assert result.orders[0].status == "filled"
    assert OrderSnapshot.model_validate_json(world.order_rows()[0]["body"]).status == "filled"
    assert world.ledger_kinds().count("order_status") == 3


async def test_the_ticket_is_persisted_as_submitted_before_polling_starts(world):
    world.broker.initial_status = "pending_new"
    ticket = await world.propose([aapl()])
    seen: list[tuple[TicketStatus, list]] = []

    async def inspecting_sleep(seconds):
        stored = world.tickets.get(ticket.id)
        seen.append((stored.status, list(stored.broker_order_ids)))

    world.executor._sleep = inspecting_sleep
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert seen, "expected at least one poll"
    assert seen[0] == (TicketStatus.submitted, [result.orders[0].id])


async def test_confirm_polls_at_most_five_times(world):
    world.broker.initial_status = "pending_new"
    ticket = await world.propose([aapl()])

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert len(world.broker.get_order_calls) == 5
    assert world.sleeps == [2.0] * 5
    assert result.orders[0].status == "pending_new"
    assert world.broker.submits == 1


@pytest.mark.parametrize("status", ["accepted", "new", "filled", "partially_filled", "rejected"])
async def test_confirm_does_not_poll_an_order_already_in_a_stop_status(world, status):
    world.broker.initial_status = status
    ticket = await world.propose([aapl()])

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.get_order_calls == []
    assert world.sleeps == []


async def test_a_polling_failure_is_recorded_not_raised(world):
    world.broker.initial_status = "pending_new"
    world.broker.get_order_error = BrokerError(500, None, "internal error")
    ticket = await world.propose([aapl()])

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.tickets.get(ticket.id).status == TicketStatus.submitted
    assert result.orders[0].status == "pending_new"  # the last snapshot we actually got
    assert world.broker.submits == 1


# --- refusals before anything is claimed --------------------------------------------------------


async def test_confirm_disabled_changes_nothing(world):
    ticket = await world.propose([aapl()])
    world.settings.trading_enabled = False

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "disabled"
    assert world.broker.submits == 0
    assert world.tickets.get(ticket.id).status == TicketStatus.proposed
    assert world.ledger_kinds() == ["confirm_request", "result"]


async def test_confirm_expired_ticket_is_marked_expired(world):
    ticket = await world.propose([aapl()])
    world.clock.now = ticket.expires_at + timedelta(seconds=1)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "expired"
    assert result.ticket.status == TicketStatus.expired
    assert world.tickets.get(ticket.id).status == TicketStatus.expired
    assert world.broker.submits == 0


async def test_confirm_at_exactly_expires_at_is_still_allowed(world):
    ticket = await world.propose([aapl()])
    world.clock.now = ticket.expires_at

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"


@pytest.mark.parametrize("bad_hash", ["0" * 64, "", "not-a-hash", "é" * 64])
async def test_confirm_wrong_hash_is_rejected_as_ticket_changed(world, bad_hash):
    ticket = await world.propose([aapl()])

    result = await world.executor.confirm(ticket.id, SESSION, bad_hash)

    assert result.status == "rejected"
    assert "ticket changed" in result.message
    assert world.tickets.get(ticket.id).status == TicketStatus.proposed
    assert world.broker.submits == 0


async def test_confirm_unknown_ticket_is_not_found(world):
    result = await world.executor.confirm("TNOSUCHTICKT", SESSION, "0" * 64)
    assert result.status == "not_found"
    assert result.ticket is None


async def test_confirm_from_another_session_is_not_found_and_reveals_nothing(world):
    ticket = await world.propose([aapl()])

    result = await world.executor.confirm(ticket.id, "sess-other", ticket.hash)

    assert result.status == "not_found"
    assert result.ticket is None
    assert world.tickets.get(ticket.id).status == TicketStatus.proposed
    assert world.broker.submits == 0


async def test_confirm_of_an_unparseable_ticket_row_is_an_error(world):
    ticket = await world.propose([aapl()])
    world.db.execute("UPDATE tickets SET body=? WHERE id=?", ("{not json", ticket.id))

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert world.broker.submits == 0


async def test_confirm_replay_after_submission_is_rejected_without_a_second_submit(world):
    ticket = await world.propose([aapl()])
    first = await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    assert first.status == "submitted"

    replay = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert replay.status == "rejected"
    assert "submitted" in replay.message
    assert world.broker.submits == 1


# --- claim first ------------------------------------------------------------------------------


async def test_two_concurrent_confirms_submit_exactly_once(world):
    ticket = await world.propose([aapl(), key_market()])

    results = await asyncio.gather(
        world.executor.confirm(ticket.id, SESSION, ticket.hash),
        world.executor.confirm(ticket.id, SESSION, ticket.hash),
    )

    assert sorted(r.status for r in results) == ["rejected", "submitted"]
    assert world.broker.submits == 2  # one per leg, once


def test_two_confirms_racing_on_separate_threads_submit_exactly_once(world):
    """Both threads read the ticket as `proposed` (the barrier holds each one right after its read)
    before either claims it, so the loser really does reach the claim CAS - and loses it."""
    ticket = asyncio.run(world.propose([aapl(), key_market()]))
    barrier = threading.Barrier(2, timeout=10)
    original_get = world.tickets.get
    local = threading.local()

    def get_then_wait(ticket_id):
        found = original_get(ticket_id)
        if not getattr(local, "waited", False):
            local.waited = True
            barrier.wait()
        return found

    world.tickets.get = get_then_wait
    results: list[ConfirmResult] = []

    def run():
        results.append(asyncio.run(world.executor.confirm(ticket.id, SESSION, ticket.hash)))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sorted(r.status for r in results) == ["rejected", "submitted"]
    loser = next(r for r in results if r.status == "rejected")
    assert "already being confirmed" in loser.message
    assert world.broker.submits == 2


async def test_a_lost_claim_never_submits_even_when_the_read_said_proposed(world):
    ticket = await world.propose([aapl()])
    stale = world.tickets.get(ticket.id)
    real_get = world.tickets.get
    world.tickets.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
    world.tickets.get = lambda _ticket_id: stale  # another confirm claimed it after our read

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "rejected"
    assert "already being confirmed" in result.message
    assert world.broker.submits == 0
    assert real_get(ticket.id).status == TicketStatus.submitting  # the winner's, untouched


# --- re-check and requote ---------------------------------------------------------------------


async def test_requote_on_price_drift(world):
    ticket = await world.propose([key_market()])
    world.data.set_price("KEY", "15.30")  # est. price moves ~2%

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "requote"
    assert world.broker.submits == 0
    assert world.tickets.get(ticket.id).status == TicketStatus.superseded
    new = result.new_ticket
    assert new is not None and new.id != ticket.id
    assert new.status == TicketStatus.proposed
    assert world.tickets.get(new.id) == new
    assert new.legs == ticket.legs
    assert new.decisions[0].est_price > ticket.decisions[0].est_price
    assert new.thesis == ticket.thesis and new.evidence_ids == ticket.evidence_ids
    assert new.quote_as_of == world.clock()
    assert new.expires_at == world.clock() + timedelta(seconds=world.settings.ticket_ttl_sec)
    assert "requote" in world.ledger_kinds()


async def test_small_drift_within_the_threshold_still_submits(world):
    ticket = await world.propose([key_market()])
    world.data.set_price("KEY", "15.05")  # ~0.3%

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.submits == 1


async def test_requote_on_verdict_change_drops_the_rejected_leg_from_the_new_ticket(world):
    ticket = await world.propose([aapl(), key_market()])
    world.broker.assets["KEY"] = make_asset("KEY", tradable=False)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "requote"
    assert world.broker.submits == 0
    new = result.new_ticket
    assert [d.verdict for d in new.decisions] == ["accept", "reject"]
    assert new.legs == [ticket.legs[0]]


async def test_requote_on_clamp(world):
    ticket = await world.propose([aapl()])
    world.broker.account_snapshot = make_account(
        buying_power=D("1500"), non_marginable_buying_power=D("1500"), cash=D("1500")
    )

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "requote"
    assert world.broker.submits == 0
    assert result.new_ticket.decisions[0].verdict == "clamp"
    assert result.new_ticket.legs[0].qty < ticket.legs[0].qty


async def test_requote_uses_the_sessions_current_mandate(world):
    ticket = await world.propose([aapl(), key_market()])
    world.mandates[SESSION] = Mandate(deny_symbols=["KEY"], version=2)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "requote"
    assert SESSION in world.mandate_calls
    assert result.new_ticket.mandate_version == 2
    assert result.new_ticket.decisions[1].reasons[0].code == "denied_symbol"


async def test_all_legs_rejected_on_recheck_is_rejected_with_reasons(world):
    ticket = await world.propose([aapl()])
    world.broker.account_snapshot = make_account(trading_blocked=True)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "rejected"
    assert result.new_ticket is None
    assert "cannot trade" in result.message
    assert world.tickets.get(ticket.id).status == TicketStatus.superseded
    assert world.tickets.active(SESSION) is None
    assert world.broker.submits == 0


async def test_a_recheck_failure_fails_the_ticket_without_submitting(world):
    ticket = await world.propose([aapl()])
    world.broker.account_error = BrokerError(503, None, "service unavailable")

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert world.tickets.get(ticket.id).status == TicketStatus.failed
    assert world.broker.submits == 0
    assert "nothing was submitted" in result.message.lower()


# --- submission failures and reconcile --------------------------------------------------------


async def test_timeout_then_reconcile_finds_the_order_without_a_second_submit(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["placed_then_timeout"]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.submits == 1
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]
    stored = world.tickets.get(ticket.id)
    assert stored.broker_order_ids == [result.orders[0].id]
    assert world.order_rows()[0]["broker_order_id"] == result.orders[0].id
    assert "reconcile" in world.ledger_kinds()


async def test_timeout_and_reconcile_finds_nothing_marks_the_leg_not_placed(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["timeout"]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert "not placed" in result.message
    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.failed
    assert stored.broker_order_ids == [None]
    assert world.broker.submits == 1
    assert world.order_rows() == []


async def test_duplicate_client_order_id_reconciles_to_the_existing_order(world):
    ticket = await world.propose([aapl()])
    existing = world.broker.place(ticket.legs[0], ticket.client_order_ids[0])
    world.broker.submit_actions = [
        BrokerError(422, 40010001, "client_order_id must be unique")
    ]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.submits == 1
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]
    assert result.orders[0].id == existing.id
    assert world.tickets.get(ticket.id).broker_order_ids == [existing.id]


async def test_httpx_level_failure_reconciles(world):
    ticket = await world.propose([aapl()])

    def placed_then_transport_error(broker, leg, cid):
        broker.place(leg, cid)
        raise httpx.ReadTimeout("timed out")

    world.broker.submit_actions = [placed_then_transport_error]
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.submits == 1
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]


async def test_a_failed_reconcile_marks_the_leg_uncertain_and_never_resubmits(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["timeout"]
    world.broker.reconcile_error = BrokerError(503, None, "service unavailable")

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert world.broker.submits == 1
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]  # exactly one reconcile
    assert result.status == "submitted"  # an uncertain leg may have been placed
    assert "uncertain" in result.message
    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted
    assert stored.broker_order_ids == [None]
    assert result.orders == []
    assert world.broker.get_order_calls == []
    rows = world.order_rows()  # recorded, so it still counts towards orders_today
    assert [r["client_order_id"] for r in rows] == ticket.client_order_ids
    assert rows[0]["broker_order_id"] is None


async def test_partial_multi_leg_failure_is_submitted_with_a_per_leg_error(world):
    ticket = await world.propose([aapl(), key_market(), msft_sell()])
    world.broker.submit_actions = [
        "ok",
        BrokerError(403, 40310000, "insufficient buying power"),
        "ok",
    ]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.submits == 3
    assert world.broker.reconcile_calls == []  # a clean 4xx is not ambiguous
    stored = world.tickets.get(ticket.id)
    assert stored.broker_order_ids[1] is None
    assert stored.broker_order_ids[0] and stored.broker_order_ids[2]
    assert [o.symbol for o in result.orders] == ["AAPL", "MSFT"]
    assert "insufficient buying power" in result.message
    assert "KEY" in result.message
    assert "2 of 3" in result.message


async def test_every_leg_failing_fails_the_ticket(world):
    ticket = await world.propose([aapl(), key_market()])
    world.broker.submit_actions = [
        BrokerError(403, 40310000, "insufficient buying power"),
        BrokerError(422, 42210000, "qty must be > 0"),
    ]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert world.tickets.get(ticket.id).status == TicketStatus.failed
    assert world.broker.submits == 2
    assert result.orders == []


async def test_a_value_error_before_the_request_is_a_leg_error(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = [ValueError("invalid client_order_id: 'x'")]

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert world.broker.reconcile_calls == []
    assert "invalid client_order_id" in result.message


async def test_an_unexpected_submit_exception_is_treated_as_uncertain_and_reconciled(world):
    ticket = await world.propose([aapl()])

    def placed_then_bug(broker, leg, cid):
        broker.place(leg, cid)
        raise RuntimeError("something odd")

    world.broker.submit_actions = [placed_then_bug]
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]
    assert world.broker.submits == 1


async def test_cancellation_mid_submit_persists_the_ticket_as_submitted(world):
    ticket = await world.propose([aapl(), key_market()])

    def cancelled(broker, leg, cid):
        raise asyncio.CancelledError()

    world.broker.submit_actions = ["ok", cancelled]
    with pytest.raises(asyncio.CancelledError):
        await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted  # never stranded in `submitting`
    assert stored.broker_order_ids[0] is not None
    assert stored.broker_order_ids[1] is None
    assert world.broker.submits == 2


# --- reject / refresh -------------------------------------------------------------------------


async def test_reject_marks_the_ticket_rejected_by_user(world):
    ticket = await world.propose([aapl()])

    result = await world.executor.reject(ticket.id, SESSION)

    assert result.status == "rejected"
    assert result.message == "Rejected by user"
    assert result.ticket.status == TicketStatus.rejected_by_user
    assert world.tickets.get(ticket.id).status == TicketStatus.rejected_by_user
    assert "reject" in world.ledger_kinds()

    confirm = await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    assert confirm.status == "rejected"
    assert world.broker.submits == 0


async def test_reject_of_a_non_proposed_ticket_reports_its_status(world):
    ticket = await world.propose([aapl()])
    await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    result = await world.executor.reject(ticket.id, SESSION)

    assert result.status == "rejected"
    assert "ticket is submitted" in result.message
    assert world.tickets.get(ticket.id).status == TicketStatus.submitted


async def test_reject_from_another_session_is_not_found(world):
    ticket = await world.propose([aapl()])
    result = await world.executor.reject(ticket.id, "sess-other")
    assert result.status == "not_found"
    assert world.tickets.get(ticket.id).status == TicketStatus.proposed


async def test_refresh_requotes_into_a_new_ticket(world):
    ticket = await world.propose([key_market()])
    world.data.set_price("KEY", "15.50")
    world.clock.now = FIXED_NOW + timedelta(seconds=30)

    result = await world.executor.refresh(ticket.id, SESSION)

    assert result.status == "requote"
    new = result.new_ticket
    assert new.status == TicketStatus.proposed
    assert new.quote_as_of == world.clock()
    assert new.decisions[0].est_price > ticket.decisions[0].est_price
    assert world.tickets.get(ticket.id).status == TicketStatus.superseded
    assert world.broker.submits == 0


async def test_refresh_of_a_non_proposed_ticket_is_rejected(world):
    ticket = await world.propose([aapl()])
    await world.executor.reject(ticket.id, SESSION)

    result = await world.executor.refresh(ticket.id, SESSION)

    assert result.status == "rejected"
    assert "ticket is rejected_by_user" in result.message


async def test_refresh_with_every_leg_rejected_is_rejected_and_keeps_the_old_ticket(world):
    ticket = await world.propose([aapl()])
    world.broker.account_snapshot = make_account(trading_blocked=True)

    result = await world.executor.refresh(ticket.id, SESSION)

    assert result.status == "rejected"
    assert result.new_ticket is None
    assert world.tickets.get(ticket.id).status == TicketStatus.proposed


async def test_refresh_from_another_session_is_not_found(world):
    ticket = await world.propose([aapl()])
    result = await world.executor.refresh(ticket.id, "sess-other")
    assert result.status == "not_found"


# --- build_context ----------------------------------------------------------------------------


async def test_build_context_quotes_open_order_and_position_symbols_too():
    broker = FakeBroker(
        open_orders=[make_open_order("NVDA", qty="2")],
        positions={"MSFT": make_position("MSFT", "10", "400")},
    )
    data = FakeData()
    data.set_price("NVDA", "120")
    world = World(broker=broker, data=data)

    ctx = await world.executor.build_context(["AAPL"])

    assert data.calls[-1] == ["AAPL", "MSFT", "NVDA"]
    assert set(ctx.quotes) == {"AAPL", "MSFT", "NVDA"}
    assert set(ctx.assets) == {"AAPL"}  # assets only for the requested symbols
    assert [o.symbol for o in ctx.open_orders] == ["NVDA"]
    assert broker.orders_calls[-1]["status"] == "open"
    assert broker.orders_calls[-1]["limit"] == 500
    assert ctx.clock.is_open is True
    assert ctx.orders_today == 0


async def test_build_context_leaves_out_an_asset_that_fails_so_risk_rejects_it(world):
    world.broker.asset_errors["KEY"] = BrokerError(404, 40410000, "asset not found")

    ctx = await world.executor.build_context(["AAPL", "KEY"])

    assert set(ctx.assets) == {"AAPL"}
    decisions = evaluate([aapl(), key_market()], ctx, Mandate())
    assert decisions[1].verdict == "reject"
    assert decisions[1].reasons[0].code == "asset_not_tradable"


async def test_build_context_counts_todays_orders_by_utc_day(world):
    ticket = await world.propose([aapl(), key_market()])
    await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    ctx = await world.executor.build_context(["AAPL"])
    assert ctx.orders_today == 2

    world.clock.now = FIXED_NOW.replace(hour=23, minute=59)
    assert (await world.executor.build_context(["AAPL"])).orders_today == 2
    world.clock.now = FIXED_NOW + timedelta(days=1)
    assert (await world.executor.build_context(["AAPL"])).orders_today == 0


def test_executor_adds_created_at_to_a_legacy_orders_table():
    db = AppDB(":memory:")
    db.execute("DROP TABLE orders")
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL, "
        "client_order_id TEXT UNIQUE, broker_order_id TEXT, body TEXT, updated_at TEXT)"
    )
    tickets = TicketStore(db, clock=Clock())
    settings = SimpleNamespace(trading_enabled=True, ticket_ttl_sec=300)
    for _ in range(2):  # idempotent
        Executor(FakeBroker(), FakeData(), tickets, Ledger(db), settings,
                 mandate_for=lambda _session_id: Mandate())
    columns = {row["name"] for row in db.query("PRAGMA table_info(orders)")}
    assert "created_at" in columns


# --- kill switch ------------------------------------------------------------------------------


async def test_kill_switch_without_broker_suspension(world):
    ticket = await world.propose([aapl()])

    await world.executor.kill_switch()

    assert world.settings.trading_enabled is False
    assert world.broker.suspend_calls == []
    assert world.broker.cancelled == []
    assert "kill_switch" in world.ledger_kinds("system")
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    assert result.status == "disabled"
    assert world.broker.submits == 0


async def test_kill_switch_with_broker_suspension_cancels_every_open_order():
    orders = [make_open_order("AAPL"), make_open_order("MSFT", side="sell"),
              make_open_order("KEY")]
    world = World(broker=FakeBroker(open_orders=orders))
    world.broker.cancel_errors[orders[1].id] = BrokerError(422, None, "order is not cancelable")

    await world.executor.kill_switch(suspend_broker=True)  # a failed cancel is not raised

    assert world.settings.trading_enabled is False
    assert world.broker.suspend_calls == [True]
    assert world.broker.calls.index("set_suspend_trade") < world.broker.calls.index("cancel_order")
    assert world.broker.cancelled == [orders[0].id, orders[2].id]
    entry = [e for e in world.ledger_entries("system") if e["kind"] == "kill_switch"][-1]
    assert entry["payload"]["suspend_broker"] is True


async def test_kill_switch_thrown_mid_confirm_stops_the_remaining_legs(world):
    ticket = await world.propose([aapl(), key_market()])

    def place_then_kill(broker, leg, cid):
        world.settings.trading_enabled = False  # someone hits the kill switch right now
        return broker.place(leg, cid)

    world.broker.submit_actions = [place_then_kill]
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert world.broker.submits == 1
    assert result.status == "submitted"
    assert "kill switch" in result.message
    assert world.tickets.get(ticket.id).broker_order_ids[1] is None


async def test_kill_switch_still_cancels_when_suspension_fails():
    orders = [make_open_order("AAPL")]
    world = World(broker=FakeBroker(open_orders=orders))
    world.broker.suspend_error = BrokerError(500, None, "internal error")

    await world.executor.kill_switch(suspend_broker=True)

    assert world.settings.trading_enabled is False
    assert world.broker.cancelled == [orders[0].id]


# --- secrets ----------------------------------------------------------------------------------


async def test_the_ledger_and_results_never_contain_the_fake_credentials(world):
    leak = f"APCA-API-KEY-ID={FAKE_KEY_ID} apiKey={FAKE_SECRET} https://u:{FAKE_SECRET}@x"

    # Happy path (including a symbol spelled "KEY").
    happy = await world.propose([aapl(), key_market()])
    ok = await world.executor.confirm(happy.id, SESSION, happy.hash)
    assert ok.status == "submitted"

    # Failure path: a 4xx, a timeout whose reconcile fails, both carrying the credentials.
    world.clock.now = FIXED_NOW + timedelta(seconds=10)
    failing = await world.propose([aapl("1"), key_market("1"), msft_sell("1")])
    world.broker.submit_actions = [
        BrokerError(403, 40310000, f"forbidden {leak}"),
        OrderUncertain(0, None, f"ReadTimeout {leak}"),
        "ok",
    ]

    def reconcile_error_with_leak():
        world.broker.reconcile_error = BrokerError(503, None, f"unavailable {leak}")

    reconcile_error_with_leak()
    bad = await world.executor.confirm(failing.id, SESSION, failing.hash)
    assert bad.status == "submitted"

    # And the kill switch's own failures.
    world.broker.suspend_error = BrokerError(500, None, f"boom {leak}")
    await world.executor.kill_switch(suspend_broker=True)

    exported = "\n".join(
        list(world.ledger.export_jsonl(SESSION)) + list(world.ledger.export_jsonl("system"))
    )
    assert FAKE_SECRET not in exported
    assert FAKE_KEY_ID not in exported
    for result in (ok, bad):
        assert FAKE_SECRET not in result.message
        assert FAKE_KEY_ID not in result.message
    assert '"symbol": "KEY"' in exported  # symbols are values, never redacted dict keys


# --- fix round 1: refresh races -------------------------------------------------------------------


def _yield_during_account(world: World) -> None:
    """Make `build_context` yield to the event loop, so a concurrent coroutine can act on the ticket
    between another one's read and its write."""
    original = world.broker.account

    async def yielding_account():
        await asyncio.sleep(0)
        return await original()

    world.broker.account = yielding_account


async def test_refresh_racing_confirm_never_leaves_a_live_copy(world):
    ticket = await world.propose([key_market()])
    _yield_during_account(world)

    refreshed, confirmed = await asyncio.gather(
        world.executor.refresh(ticket.id, SESSION),
        world.executor.confirm(ticket.id, SESSION, ticket.hash),
    )

    assert confirmed.status == "submitted"
    assert refreshed.status == "rejected"
    assert refreshed.new_ticket is None
    assert "ticket is submitting" in refreshed.message
    assert world.tickets.active(SESSION) is None  # nothing left to confirm a second time
    assert world.broker.submits == 1


async def test_refresh_racing_reject_never_leaves_a_live_copy(world):
    ticket = await world.propose([key_market()])
    _yield_during_account(world)

    refreshed, rejected = await asyncio.gather(
        world.executor.refresh(ticket.id, SESSION),
        world.executor.reject(ticket.id, SESSION),
    )

    assert rejected.message == "Rejected by user"
    assert refreshed.status == "rejected"
    assert refreshed.new_ticket is None
    assert world.tickets.get(ticket.id).status == TicketStatus.rejected_by_user
    assert world.tickets.active(SESSION) is None


async def test_refresh_supersedes_the_old_ticket_before_creating_the_new_one(world):
    ticket = await world.propose([key_market()])
    original_create = world.tickets.create
    seen: list[TicketStatus] = []

    def create_after_checking(*args, **kwargs):
        seen.append(world.tickets.get(ticket.id).status)
        return original_create(*args, **kwargs)

    world.tickets.create = create_after_checking
    result = await world.executor.refresh(ticket.id, SESSION)

    assert result.status == "requote"
    assert seen == [TicketStatus.superseded]


def test_confirm_refresh_and_reject_racing_on_threads_leave_at_most_one_live_outcome():
    for _ in range(5):
        world = World()
        ticket = asyncio.run(world.propose([key_market()]))
        barrier = threading.Barrier(3, timeout=10)
        original_get = world.tickets.get
        local = threading.local()

        def get_then_wait(ticket_id, _get=original_get, _barrier=barrier, _local=local):
            found = _get(ticket_id)
            if not getattr(_local, "waited", False):
                _local.waited = True
                _barrier.wait()
            return found

        world.tickets.get = get_then_wait
        results: dict[str, ConfirmResult] = {}
        calls = {
            "confirm": lambda w=world, t=ticket: w.executor.confirm(t.id, SESSION, t.hash),
            "refresh": lambda w=world, t=ticket: w.executor.refresh(t.id, SESSION),
            "reject": lambda w=world, t=ticket: w.executor.reject(t.id, SESSION),
        }

        def run(name, results=results, calls=calls):
            results[name] = asyncio.run(calls[name]())

        threads = [threading.Thread(target=run, args=(name,)) for name in calls]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        world.tickets.get = original_get

        winners = [
            results["confirm"].status == "submitted",
            results["refresh"].status == "requote",
            results["reject"].message == "Rejected by user",
        ]
        assert sum(winners) == 1, {k: (v.status, v.message) for k, v in results.items()}
        active = world.tickets.active(SESSION)
        if results["refresh"].status == "requote":
            assert active is not None and world.broker.submits == 0
        else:
            assert active is None
        assert world.broker.submits in (0, 1)


# --- fix round 1: legs interrupted mid-call are recorded ------------------------------------------


async def test_cancellation_after_the_broker_placed_the_order_records_it_as_uncertain(world):
    ticket = await world.propose([aapl()])

    def place_then_cancel(broker, leg, cid):
        broker.place(leg, cid)
        raise asyncio.CancelledError()

    world.broker.submit_actions = [place_then_cancel]
    with pytest.raises(asyncio.CancelledError):
        await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    rows = world.order_rows()
    assert [r["client_order_id"] for r in rows] == ticket.client_order_ids
    assert rows[0]["broker_order_id"] is None
    assert world.tickets.get(ticket.id).status == TicketStatus.submitted
    assert (await world.executor.build_context(["AAPL"])).orders_today == 1
    interrupted = [e for e in world.ledger_entries() if e["kind"] == "result"][-1]["payload"]
    assert interrupted["status"] == "interrupted"
    assert [(leg["leg"], leg["outcome"]) for leg in interrupted["legs"]] == [(0, "uncertain")]


async def test_cancellation_during_reconcile_records_the_leg_as_uncertain(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["placed_then_timeout"]

    async def cancelled_lookup(client_order_id):
        raise asyncio.CancelledError()

    world.broker.get_order_by_client_id = cancelled_lookup
    with pytest.raises(asyncio.CancelledError):
        await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    rows = world.order_rows()
    assert [r["client_order_id"] for r in rows] == ticket.client_order_ids
    assert rows[0]["broker_order_id"] is None
    assert (await world.executor.build_context(["AAPL"])).orders_today == 1
    assert world.broker.submits == 1


async def test_a_non_cancellation_base_exception_mid_leg_still_records_the_leg(world):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["placed_then_timeout"]

    async def interrupted_lookup(client_order_id):
        raise KeyboardInterrupt

    world.broker.get_order_by_client_id = interrupted_lookup
    with pytest.raises(KeyboardInterrupt):
        await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert [r["client_order_id"] for r in world.order_rows()] == ticket.client_order_ids
    assert world.tickets.get(ticket.id).status == TicketStatus.submitted


# --- fix round 1: persisting the outcome ----------------------------------------------------------


def _flaky_outcome_transition(world: World, failures: int) -> None:
    """Make the outcome CAS (submitting -> submitted/failed) raise a DB error `failures` times."""
    real = world.tickets.transition
    state = {"left": failures}

    def flaky(ticket_id, expect, to, **fields):
        if to in (TicketStatus.submitted, TicketStatus.failed) and state["left"] > 0:
            state["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real(ticket_id, expect, to, **fields)

    world.tickets.transition = flaky


async def test_a_transient_db_error_on_the_outcome_cas_is_retried(world):
    ticket = await world.propose([aapl()])
    _flaky_outcome_transition(world, failures=1)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted
    assert result.status == "submitted"
    assert stored.broker_order_ids == [result.orders[0].id]
    assert world.broker.submits == 1


async def test_a_persistent_db_error_on_the_outcome_cas_never_claims_success(world):
    ticket = await world.propose([aapl()])
    _flaky_outcome_transition(world, failures=10)

    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert "may have been placed" in result.message
    assert "AAPL" in result.message
    assert world.broker.submits == 1
    assert [r["client_order_id"] for r in world.order_rows()] == ticket.client_order_ids


async def test_a_ticket_state_error_on_the_outcome_cas_never_claims_success(world):
    ticket = await world.propose([aapl()])
    real = world.tickets.transition

    def refuse_outcome(ticket_id, expect, to, **fields):
        if to in (TicketStatus.submitted, TicketStatus.failed):
            raise TicketStateError("changed underneath us")
        return real(ticket_id, expect, to, **fields)

    world.tickets.transition = refuse_outcome
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "error"
    assert "may have been placed" in result.message
    assert world.broker.submits == 1


# --- fix round 1: reconcile must match the leg ----------------------------------------------------


@pytest.mark.parametrize(
    "mismatch",
    [
        {"client_order_id": "manual-xyz"},
        {"symbol": "MSFT"},
        {"side": "sell"},
    ],
)
async def test_reconcile_never_adopts_an_order_that_does_not_match_the_leg(world, mismatch):
    ticket = await world.propose([aapl()])
    world.broker.submit_actions = ["timeout"]
    cid = ticket.client_order_ids[0]
    foreign = make_order(ticket.legs[0], cid).model_copy(update=mismatch)

    async def wrong_lookup(client_order_id):
        world.broker.reconcile_calls.append(client_order_id)
        return foreign

    world.broker.get_order_by_client_id = wrong_lookup
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    stored = world.tickets.get(ticket.id)
    assert stored.broker_order_ids == [None]
    assert stored.status == TicketStatus.submitted  # uncertain: ours may still exist
    assert result.orders == []
    assert "uncertain" in result.message
    assert world.broker.reconcile_calls == [cid]
    assert world.broker.submits == 1
    assert [(r["client_order_id"], r["broker_order_id"]) for r in world.order_rows()] == [(cid, None)]


# --- fix round 1: ValueError subclasses may come after placement ----------------------------------


def _pydantic_validation_error() -> ValidationError:
    try:
        OrderSnapshot.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: json.JSONDecodeError("Expecting value", "", 0),
        _pydantic_validation_error,
        lambda: UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
    ids=["json", "pydantic", "unicode"],
)
async def test_a_value_error_subclass_after_placement_is_reconciled(world, make_error):
    ticket = await world.propose([aapl()])

    def place_then_fail(broker, leg, cid):
        broker.place(leg, cid)
        raise make_error()

    world.broker.submit_actions = [place_then_fail]
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)

    assert result.status == "submitted"
    assert world.broker.reconcile_calls == [ticket.client_order_ids[0]]
    assert world.broker.submits == 1
    assert world.tickets.get(ticket.id).broker_order_ids == [result.orders[0].id]


# --- fix round 1: construction --------------------------------------------------------------------


def test_mandate_for_is_a_required_keyword_argument(world):
    with pytest.raises(TypeError):
        Executor(world.broker, world.data, world.tickets, world.ledger, world.settings)


# --- Task 3.5: startup recovery of tickets stuck in `submitting` ---------------------------------


def _stick(world: World, ticket: Ticket) -> Ticket:
    """Simulate a crash mid-confirm: the claim CAS ran, nothing after it did."""
    return world.tickets.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)


async def test_recover_stuck_found_order_moves_the_ticket_to_submitted(world):
    ticket = _stick(world, await world.propose([aapl()]))
    placed = world.broker.place(ticket.legs[0], ticket.client_order_ids[0])

    moved = await world.executor.recover_stuck()

    assert moved == 1
    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted
    assert stored.broker_order_ids == [placed.id]
    assert world.broker.submits == 0
    assert world.broker.reconcile_calls == ticket.client_order_ids
    rows = world.order_rows()
    assert [(r["client_order_id"], r["broker_order_id"]) for r in rows] == [
        (ticket.client_order_ids[0], placed.id)
    ]
    entries = world.ledger_entries()
    recon = [e for e in entries if e["kind"] == "reconcile"]
    assert recon and recon[-1]["payload"]["outcome"] == "found"
    assert recon[-1]["payload"]["source"] == "startup"
    recover = [e for e in entries if e["kind"] == "recover"]
    assert recover and recover[-1]["payload"]["status"] == "submitted"


async def test_recover_stuck_missing_order_fails_the_ticket_with_a_ledger_entry(world):
    ticket = _stick(world, await world.propose([aapl()]))

    moved = await world.executor.recover_stuck()

    assert moved == 1
    assert world.tickets.get(ticket.id).status == TicketStatus.failed
    assert world.broker.submits == 0
    assert world.order_rows() == []
    kinds = world.ledger_kinds()
    assert "reconcile" in kinds and kinds[-1] == "recover"
    assert world.ledger_entries()[-1]["payload"]["status"] == "failed"


async def test_recover_stuck_broker_error_leaves_the_ticket_as_is(world, caplog):
    ticket = _stick(world, await world.propose([aapl()]))
    world.broker.reconcile_error = BrokerError(503, None, f"down {FAKE_SECRET}")

    moved = await world.executor.recover_stuck()

    assert moved == 0
    assert world.tickets.get(ticket.id).status == TicketStatus.submitting
    assert world.order_rows() == []
    assert FAKE_SECRET not in caplog.text
    assert FAKE_SECRET not in json.dumps(world.ledger_entries())


async def test_recover_stuck_mixed_legs_keep_the_found_one_and_fail_the_other(world):
    ticket = _stick(world, await world.propose([aapl(), msft_sell()]))
    placed = world.broker.place(ticket.legs[1], ticket.client_order_ids[1])

    assert await world.executor.recover_stuck() == 1

    stored = world.tickets.get(ticket.id)
    assert stored.status == TicketStatus.submitted
    assert stored.broker_order_ids == [None, placed.id]


async def test_recover_stuck_never_adopts_a_mismatched_order(world):
    ticket = _stick(world, await world.propose([aapl()]))
    wrong_leg = OrderLeg(symbol="MSFT", side="buy", qty=D("1"), type="market")
    world.broker.place(wrong_leg, ticket.client_order_ids[0])

    assert await world.executor.recover_stuck() == 1

    stored = world.tickets.get(ticket.id)
    # Uncertain: never adopted, but the ticket is not `failed` (an order may exist).
    assert stored.status == TicketStatus.submitted
    assert stored.broker_order_ids == [None]
    rows = world.order_rows()
    assert len(rows) == 1 and rows[0]["broker_order_id"] is None


async def test_recover_stuck_ignores_tickets_in_other_states(world):
    proposed = await world.propose([aapl()])
    moved = await world.executor.recover_stuck()
    assert moved == 0
    assert world.tickets.get(proposed.id).status == TicketStatus.proposed
    assert world.broker.reconcile_calls == []


# --- Task 3.5: order polling for GET /api/tickets/{id} -------------------------------------------


async def test_ticket_orders_refreshes_a_working_order_from_the_broker(world):
    ticket = await world.propose([aapl()])
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    order_id = result.orders[0].id
    world.broker.status_script[order_id] = ["filled"]
    submits = world.broker.submits

    tracked = await world.executor.ticket_orders(ticket.id, SESSION)

    assert tracked is not None
    stored, rows = tracked
    assert stored.id == ticket.id and stored.status == TicketStatus.submitted
    assert [r.client_order_id for r in rows] == ticket.client_order_ids
    assert rows[0].broker_order_id == order_id
    assert rows[0].order.status == "filled"
    assert world.broker.submits == submits
    body = OrderSnapshot.model_validate_json(world.order_rows()[0]["body"])
    assert body.status == "filled"
    polls = [e for e in world.ledger_entries() if e["kind"] == "order_status"]
    assert polls[-1]["payload"]["status"] == "filled"
    assert polls[-1]["payload"]["source"] == "poll"


async def test_ticket_orders_does_not_refetch_a_final_order(world):
    ticket = await world.propose([aapl()])
    result = await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    world.broker.status_script[result.orders[0].id] = ["filled"]
    await world.executor.ticket_orders(ticket.id, SESSION)
    calls = len(world.broker.get_order_calls)

    _, rows = await world.executor.ticket_orders(ticket.id, SESSION)

    assert rows[0].order.status == "filled"
    assert len(world.broker.get_order_calls) == calls


async def test_ticket_orders_keeps_the_stored_snapshot_on_a_broker_error(world, caplog):
    ticket = await world.propose([aapl()])
    await world.executor.confirm(ticket.id, SESSION, ticket.hash)
    world.broker.get_order_error = BrokerError(500, None, f"boom {FAKE_SECRET}")

    _, rows = await world.executor.ticket_orders(ticket.id, SESSION)

    assert rows[0].order.status == "accepted"
    assert FAKE_SECRET not in caplog.text


async def test_ticket_orders_of_an_unknown_or_foreign_ticket_is_none(world):
    ticket = await world.propose([aapl()])
    assert await world.executor.ticket_orders("TNOPE", SESSION) is None
    assert await world.executor.ticket_orders(ticket.id, "another-session") is None


async def test_ticket_orders_before_confirm_has_no_rows(world):
    ticket = await world.propose([aapl()])
    stored, rows = await world.executor.ticket_orders(ticket.id, SESSION)
    assert stored.status == TicketStatus.proposed and rows == []
    assert world.broker.get_order_calls == []


# --- Task 3.5 fix round 1: reject needs no broker -------------------------------------------------


async def test_reject_ticket_works_without_an_executor(world):
    from finagent.trading.execution import reject_ticket

    ticket = await world.propose([aapl()])
    result = reject_ticket(world.tickets, world.ledger, ticket.id, SESSION)
    assert result.status == "rejected" and result.message == "Rejected by user"
    assert world.tickets.get(ticket.id).status == TicketStatus.rejected_by_user
    assert world.ledger_kinds()[-1] == "reject"
    again = reject_ticket(world.tickets, world.ledger, ticket.id, SESSION)
    assert again.status == "rejected" and again.message == "ticket is rejected_by_user"
    foreign = reject_ticket(world.tickets, world.ledger, ticket.id, "another-session")
    assert foreign.status == "not_found" and foreign.ticket is None
    assert world.broker.calls.count("submit_order") == 0


async def test_executor_reject_and_reject_ticket_agree(world):
    from finagent.trading.execution import reject_ticket

    first = await world.propose([aapl()])
    via_executor = await world.executor.reject(first.id, SESSION)
    second = await world.propose([aapl()], session_id="sess-2")
    via_function = reject_ticket(world.tickets, world.ledger, second.id, "sess-2")
    assert (via_executor.status, via_executor.message) == (via_function.status, via_function.message)
    assert via_executor.ticket.status == via_function.ticket.status == TicketStatus.rejected_by_user
    assert world.ledger_kinds()[-1] == world.ledger_kinds("sess-2")[-1] == "reject"
