"""Executor: the only code path that submits orders.

`confirm()` is reached solely from the user's Confirm click (`POST /api/tickets/{id}/confirm`); the
LLM never gets an order-placing tool. Paper only: the broker is `AlpacaPaper`, whose host is a
hard-coded constant.

**confirm(ticket_id, session_id, ticket_hash)**

1. A `confirm_request` ledger entry is always written. If `settings.trading_enabled` is false (the
   kill switch), the result is `disabled` and nothing changes.
2. The ticket is loaded. An unknown id, or a ticket from another session, is `not_found` (so a
   caller can't probe other sessions' ids); an unparseable row is `error`.
3. Anything but `proposed` is `rejected` ("ticket is <status>"). This is what makes a replayed
   confirm after submission harmless: it never reaches the broker again.
4. Past `expires_at` the ticket is moved to `expired` and the result is `expired`.
5. The client's `ticket_hash` must equal the stored hash (`hmac.compare_digest`), or the result is
   `rejected` ("ticket changed").
6. **Claim first.** The ticket moves `proposed -> submitting` through `TicketStore.transition`,
   an exact compare-and-set. If two confirms race, exactly one wins; the loser gets `rejected`
   ("already being confirmed"). From here on every decision comes from the executor's own CAS
   results, never from re-reading the ticket's status.
7. **Re-check.** The market context is rebuilt and `risk.evaluate` re-runs on the ticket's legs
   against the session's *current* mandate. `ticket.legs[i]` pairs with the i-th non-rejected
   decision stored on the ticket. If any leg's new verdict is not `accept`, its leg changed, or its
   estimated price moved more than `REQUOTE_DRIFT_PCT` percent, the ticket is superseded and a new
   one is created from the new decisions (`requote`). If every leg is now rejected, no new ticket
   can exist and the result is `rejected` with the reasons. The requote path never submits. If the
   re-check itself fails (the broker or market data is unreachable), the claimed ticket is marked
   `failed` and nothing is submitted: a claimed ticket can't go back to `proposed`.
8. **Submit** each leg once, sequentially, with its pre-assigned `client_order_id` (a leg reached
   after the kill switch was thrown mid-confirm is not submitted). There are no blind retries. An `OrderUncertain` (timeout, 5xx, 3xx, unparseable 2xx), a 422 duplicate
   `client_order_id` (code 40010001), an HTTP-level failure, or any other unexpected exception may
   mean the order reached Alpaca, so it is reconciled with ONE `get_order_by_client_id` call:
   found -> the leg is placed; not found -> the leg failed ("not placed"); the lookup itself failed
   -> the leg is `uncertain` and is never resubmitted. A clean 4xx `BrokerError` or a `ValueError`
   (raised before any request) is a plain leg error. Every placed or reconciled order is written to
   the local `orders` table; so is an uncertain leg (with no broker id), so it still counts towards
   the daily order limit.
9. **Persist before polling.** `broker_order_ids` (aligned with the legs, `None` where no order is
   known) and the new status are written in one CAS: `submitted` if at least one leg is placed or
   uncertain, else `failed`.
10. **Poll** each placed order up to `POLL_ATTEMPTS` times, `POLL_INTERVAL_SEC` apart (the injected
    `sleep`), stopping once its status is in `STOP_POLL_STATUSES` - final, or acknowledged
    (`accepted`, `new`, `partially_filled`). A polling failure is recorded, never raised. The
    local `orders` row is updated on every poll.
11. The result carries the latest ticket, the latest snapshot of every placed order, and a short
    English summary with each leg's outcome. Status: `submitted` when the ticket was marked
    submitted, `error` when every leg failed.

If anything unexpected escapes after the claim, including cancellation of the calling task, the
ticket is still persisted (`submitted` if any leg was placed, uncertain or mid-flight, else
`failed`), so it is never stranded in `submitting`. Cancellation is then re-raised; any other
exception becomes an `error` result.

**Ledger.** Kinds: `confirm_request`, `claim`, `requote`, `submit`, `broker_response`,
`reconcile`, `order_status`, `result` (confirm); `reject`; `refresh`; `kill_switch` (under the
session id `SYSTEM_SESSION`). Payloads carry symbols as *values*, never as dict keys: the ledger
redacts the value of any dict key that looks secret-like, and "KEY" (KeyCorp) is a real ticker.
They carry client order ids, leg indexes, statuses and short sanitized messages - never headers,
raw HTTP bodies or credentials. Every broker message shown to a user or stored goes through
`Ledger.redact_text` first (the ledger's known-secret scrub plus its free-text passes).

**orders_today** counts local `orders` rows whose `created_at` falls on today's *UTC* date. This is
a simplification: the US trading day runs on Eastern time, so the count resets at 00:00 UTC (8 pm
ET in summer, 7 pm in winter) rather than at the ET midnight or the market open. It counts rows
across all sessions, since every session trades the same Alpaca account. The `created_at` column
is added to an existing `orders` table on first use (`_ensure_orders_created_at`).
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Any, Literal

from finagent.store.appdb import AppDB
from finagent.trading.broker import BrokerError, OrderUncertain
from finagent.trading.ledger import Ledger
from finagent.trading.models import (
    Mandate,
    MarketContext,
    OrderLeg,
    OrderSnapshot,
    RiskDecision,
    Ticket,
    TicketStatus,
    normalize_symbol,
)
from finagent.trading.proposals import TicketStateError, TicketStore, utcnow
from finagent.trading.risk import evaluate

logger = logging.getLogger(__name__)

REQUOTE_DRIFT_PCT = Decimal("1.0")
POLL_ATTEMPTS = 5
POLL_INTERVAL_SEC = 2.0
OPEN_ORDERS_LIMIT = 500
DUPLICATE_CLIENT_ORDER_ID = 40010001  # Alpaca: 422 "client_order_id must be unique"
SYSTEM_SESSION = "system"  # ledger session id for events that belong to no chat session

# Alpaca order statuses after which an order will not change again today.
FINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}
)
# Polling stops at a final status or once the order is acknowledged and working.
STOP_POLL_STATUSES = FINAL_ORDER_STATUSES | {"accepted", "new", "partially_filled"}

_MAX_MESSAGE_CHARS = 160
_DEFAULT_TTL_SEC = 300
_CENT = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")
_HUNDRED = Decimal("100")

ConfirmStatus = Literal[
    "submitted", "requote", "rejected", "disabled", "expired", "not_found", "error"
]


@dataclass
class ConfirmResult:
    status: ConfirmStatus
    ticket: Ticket | None
    orders: list[OrderSnapshot]
    message: str
    new_ticket: Ticket | None = None


@dataclass
class _LegOutcome:
    index: int
    leg: OrderLeg
    client_order_id: str
    state: Literal["placed", "uncertain", "error"]
    order: OrderSnapshot | None = None
    error: str | None = None
    reconciled: bool = False


@dataclass
class _Run:
    """The mutable state of one claimed confirm - enough to persist the ticket correctly whatever
    point it stopped at."""

    ticket: Ticket  # as returned by the claim CAS (status `submitting`)
    session_id: str
    attempted: int = 0  # legs whose submit_order call has started
    outcomes: list[_LegOutcome] = field(default_factory=list)
    persisted: bool = False
    maybe_placed: bool = False  # >= 1 leg placed or uncertain (decided by `_persist`)
    final_ticket: Ticket | None = None


# --- small helpers --------------------------------------------------------------------------------


def _hash_matches(given: Any, expected: str) -> bool:
    """Constant-time compare. Encoding first keeps `compare_digest` from raising `TypeError` on a
    non-ASCII string from the client."""
    if not isinstance(given, str):
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def _valid_symbols(symbols: Iterable[str]) -> set[str]:
    """Normalized symbols; anything `normalize_symbol` refuses (e.g. a crypto pair) is dropped."""
    out: set[str] = set()
    for symbol in symbols:
        try:
            out.add(normalize_symbol(symbol))
        except (ValueError, TypeError):
            continue
    return out


def _money(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    step = _CENT if abs(value) >= 1 else _SUB_PENNY
    return f"${value.quantize(step, rounding=ROUND_HALF_UP):,f}"


def _plain(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized.as_tuple().exponent > 0:
        normalized = normalized.quantize(Decimal(1))
    return f"{normalized:f}"


def _leg_label(index: int, leg: OrderLeg) -> str:
    size = f"{_plain(leg.qty)} {leg.symbol}" if leg.qty is not None else (
        f"{_money(leg.notional)} of {leg.symbol}"
    )
    price = f"limit {_money(leg.limit_price)}" if leg.type == "limit" else "market"
    return f"Leg {index + 1} {leg.side.upper()} {size} @ {price}"


def _leg_fields(leg: OrderLeg) -> dict[str, Any]:
    return {
        "symbol": leg.symbol,
        "side": leg.side,
        "type": leg.type,
        "time_in_force": leg.time_in_force,
        "qty": leg.qty,
        "notional": leg.notional,
        "limit_price": leg.limit_price,
    }


def _drift_pct(old: Decimal | None, new: Decimal | None) -> Decimal | None:
    """Percent move from `old` to `new`; None when it can't be computed (treated as a requote)."""
    if old is None or new is None or old <= 0:
        return None
    return abs(new - old) / old * _HUNDRED


def _block_reasons(decisions: list[RiskDecision]) -> str:
    parts = []
    for decision in decisions:
        blocks = [r.message for r in decision.reasons if r.severity == "block"]
        if blocks:
            parts.append(f"leg {decision.index + 1} {decision.leg.symbol}: {blocks[0]}")
    return "; ".join(parts)


def _ensure_orders_created_at(db: AppDB) -> None:
    """Add `orders.created_at` (needed for `orders_today`) to a table created before it existed."""
    if "created_at" in {row["name"] for row in db.query("PRAGMA table_info(orders)")}:
        return
    try:
        db.execute("ALTER TABLE orders ADD COLUMN created_at TEXT")
    except sqlite3.OperationalError:
        # Another Executor on the same DB may have added it in between; anything else is real.
        if "created_at" not in {row["name"] for row in db.query("PRAGMA table_info(orders)")}:
            raise


# --- the executor ---------------------------------------------------------------------------------


class Executor:
    """Confirm / reject / refresh tickets, and the kill switch. See the module docstring."""

    def __init__(
        self,
        broker,
        data,
        tickets: TicketStore,
        ledger: Ledger,
        settings,
        *,
        mandate_for: Callable[[str], Mandate] | None = None,
        db: AppDB | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        now: Callable[[], datetime] | None = None,
    ):
        self._broker = broker
        self._data = data
        self._tickets = tickets
        self._ledger = ledger
        self._settings = settings
        self._mandate_for = mandate_for or (lambda _session_id: Mandate())
        self._db = db if db is not None else tickets._db  # noqa: SLF001 - same DB by default
        self._sleep = sleep
        self._now = now or getattr(tickets, "_clock", None) or utcnow
        _ensure_orders_created_at(self._db)

    # --- market context -------------------------------------------------------------------------

    async def build_context(self, symbols: list[str]) -> MarketContext:
        """Account, positions, open orders, assets (for `symbols` only), quotes and clock.

        Quotes cover `symbols` plus every symbol with an open order or a position: the risk
        engine values pending open buys and positions from them. An asset lookup that fails for
        one symbol leaves it out of `assets`, so risk rejects that leg as not tradable."""
        wanted = sorted(_valid_symbols(symbols))
        account = await self._broker.account()
        positions = await self._broker.positions()
        open_orders = await self._broker.orders(status="open", limit=OPEN_ORDERS_LIMIT)
        assets = {}
        for symbol in wanted:
            try:
                assets[symbol] = await self._broker.asset(symbol)
            except (ValueError, BrokerError) as exc:
                logger.info("no asset info for %s: %s", symbol, self._short(exc))
        quote_symbols = sorted(
            set(wanted)
            | _valid_symbols(order.symbol for order in open_orders)
            | _valid_symbols(positions)
        )
        quotes = await self._data.snapshots(quote_symbols) if quote_symbols else {}
        quotes = {s: q for s, q in quotes.items() if _valid_symbols([s])}
        clock = await self._broker.clock()
        return MarketContext(
            account=account,
            positions=positions,
            open_orders=open_orders,
            assets=assets,
            quotes=quotes,
            clock=clock,
            orders_today=self._orders_today(),
        )

    def _orders_today(self) -> int:
        today = self._now().astimezone(timezone.utc).date().isoformat()
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM orders WHERE substr(created_at, 1, 10) = ?", (today,)
        )
        return int(rows[0]["n"])

    # --- confirm --------------------------------------------------------------------------------

    async def confirm(self, ticket_id: str, session_id: str, ticket_hash: str) -> ConfirmResult:
        self._log(session_id, "confirm_request", {"ticket_id": str(ticket_id)[:64]})
        if not self._settings.trading_enabled:
            return self._finish(session_id, ticket_id, ConfirmResult(
                "disabled", None, [], "Trading is disabled (kill switch); nothing was submitted."
            ))

        loaded = self._load(ticket_id, session_id)
        if isinstance(loaded, ConfirmResult):
            return self._finish(session_id, ticket_id, loaded)
        ticket = loaded

        if ticket.status != TicketStatus.proposed:
            return self._finish(session_id, ticket_id, ConfirmResult(
                "rejected", ticket, [],
                f"ticket is {ticket.status.value}; this request submitted nothing",
            ))

        if self._now() > ticket.expires_at:
            try:
                ticket = self._tickets.transition(
                    ticket.id, {TicketStatus.proposed}, TicketStatus.expired
                )
            except TicketStateError:
                ticket = self._latest(ticket.id, ticket)
            return self._finish(session_id, ticket_id, ConfirmResult(
                "expired", ticket, [],
                "This ticket has expired; nothing was submitted. Ask for a fresh quote.",
            ))

        if not _hash_matches(ticket_hash, ticket.hash):
            return self._finish(session_id, ticket_id, ConfirmResult(
                "rejected", ticket, [],
                "ticket changed; review the latest ticket before confirming. Nothing was "
                "submitted.",
            ))

        try:
            claimed = self._tickets.transition(
                ticket.id, {TicketStatus.proposed}, TicketStatus.submitting
            )
        except TicketStateError:
            return self._finish(session_id, ticket_id, ConfirmResult(
                "rejected", self._latest(ticket.id, None), [],
                "ticket is already being confirmed (or is no longer proposed); nothing was "
                "submitted by this request",
            ))
        self._log(session_id, "claim", {"ticket_id": claimed.id, "legs": len(claimed.legs)})

        run = _Run(ticket=claimed, session_id=session_id)
        try:
            result = await self._run_claimed(run)
        except Exception:  # noqa: BLE001 - never strand a claimed ticket in `submitting`
            logger.exception("confirm of ticket %s failed after it was claimed", claimed.id)
            self._persist(run)
            result = self._crash_result(run)
        except BaseException:  # cancellation: persist what we know, then let it propagate
            self._persist(run)
            self._log(session_id, "result", {
                "ticket_id": claimed.id,
                "status": "interrupted",
                "legs": self._outcome_payload(run),
            })
            raise
        return self._finish(session_id, ticket_id, result, run)

    async def _run_claimed(self, run: _Run) -> ConfirmResult:
        ticket = run.ticket
        try:
            ctx = await self.build_context([leg.symbol for leg in ticket.legs])
            mandate = self._mandate_for(run.session_id)
            decisions = evaluate(ticket.legs, ctx, mandate)
        except Exception as exc:  # noqa: BLE001 - no fresh check, no submission
            logger.warning("re-check of ticket %s failed: %s", ticket.id, self._short(exc))
            self._persist(run)  # nothing attempted -> failed
            return ConfirmResult(
                "error", self._latest(ticket.id, run.final_ticket), [],
                f"Could not re-check the ticket before submitting ({self._short(exc)}); nothing "
                "was submitted. Ask for a fresh quote.",
            )

        changes = self._requote_reasons(ticket, decisions)
        if changes:
            return self._requote_on_confirm(run, decisions, mandate, changes)

        for index, (leg, client_order_id) in enumerate(zip(ticket.legs, ticket.client_order_ids)):
            if not self._settings.trading_enabled:
                # The kill switch was thrown mid-confirm: submit nothing more.
                run.outcomes.append(_LegOutcome(
                    index, leg, client_order_id, "error",
                    error="not submitted: trading was disabled (kill switch)",
                ))
                continue
            self._log(run.session_id, "submit", {
                "ticket_id": ticket.id,
                "leg": index,
                "client_order_id": client_order_id,
                **_leg_fields(leg),
            })
            run.attempted = index + 1
            run.outcomes.append(await self._submit_one(run, index, leg, client_order_id))

        self._persist(run)
        await self._poll(run)
        return self._submission_result(run)

    def _requote_reasons(self, ticket: Ticket, decisions: list[RiskDecision]) -> list[str]:
        """Why this ticket can't be submitted as-is; empty when nothing material changed."""
        stored = [d for d in ticket.decisions if d.verdict != "reject"]
        if len(stored) != len(ticket.legs) or len(decisions) != len(ticket.legs):
            return ["the ticket's legs and decisions no longer line up"]
        reasons: list[str] = []
        for index, (leg, old, new) in enumerate(zip(ticket.legs, stored, decisions)):
            label = f"leg {index + 1} {leg.symbol}"
            if new.verdict != "accept":
                detail = "; ".join(
                    r.message for r in new.reasons if r.severity in ("block", "warn")
                )
                reasons.append(f"{label} is now {new.verdict}" + (f" ({detail})" if detail else ""))
            elif new.leg != leg:
                reasons.append(f"{label} changed")
            else:
                drift = _drift_pct(old.est_price, new.est_price)
                if drift is None:
                    reasons.append(f"{label} has no comparable price estimate")
                elif drift > REQUOTE_DRIFT_PCT:
                    shown = drift.quantize(_CENT, rounding=ROUND_UP)
                    reasons.append(
                        f"{label} estimated price moved {shown}% "
                        f"({_money(old.est_price)} -> {_money(new.est_price)})"
                    )
        return reasons

    def _requote_on_confirm(
        self, run: _Run, decisions: list[RiskDecision], mandate: Mandate, changes: list[str]
    ) -> ConfirmResult:
        ticket = run.ticket
        try:
            superseded = self._tickets.transition(
                ticket.id, {TicketStatus.submitting}, TicketStatus.superseded
            )
        except TicketStateError as exc:
            logger.error("could not supersede claimed ticket %s: %s", ticket.id, exc)
            run.persisted = True  # it is no longer ours to move
            return ConfirmResult(
                "error", self._latest(ticket.id, None), [],
                "The ticket changed state during re-check; nothing was submitted.",
            )
        run.persisted = True
        run.final_ticket = superseded
        summary = "; ".join(changes)
        try:
            new_ticket = self._new_ticket(ticket, decisions, mandate)
        except ValueError:
            reasons = _block_reasons(decisions) or summary
            self._log(run.session_id, "requote", {
                "ticket_id": ticket.id, "new_ticket_id": None, "source": "confirm",
                "reasons": changes,
            })
            return ConfirmResult(
                "rejected", superseded, [],
                f"No leg can be placed at current prices; nothing was submitted. {reasons}",
            )
        self._log(run.session_id, "requote", {
            "ticket_id": ticket.id, "new_ticket_id": new_ticket.id, "source": "confirm",
            "reasons": changes,
        })
        return ConfirmResult(
            "requote", superseded, [],
            f"Prices or risk checks changed since this ticket was proposed; nothing was "
            f"submitted. Review the new ticket: {summary}.",
            new_ticket=new_ticket,
        )

    def _new_ticket(self, ticket: Ticket, decisions: list[RiskDecision], mandate: Mandate) -> Ticket:
        return self._tickets.create(
            ticket.session_id,
            decisions=decisions,
            thesis=ticket.thesis,
            risks=ticket.risks,
            invalidation=ticket.invalidation,
            evidence_ids=list(ticket.evidence_ids),
            mandate_version=mandate.version,
            quote_as_of=self._now(),
            ttl_sec=int(getattr(self._settings, "ticket_ttl_sec", _DEFAULT_TTL_SEC)),
        )

    # --- submission -----------------------------------------------------------------------------

    async def _submit_one(
        self, run: _Run, index: int, leg: OrderLeg, client_order_id: str
    ) -> _LegOutcome:
        try:
            order = await self._broker.submit_order(leg, client_order_id)
        except OrderUncertain as exc:
            return await self._reconcile(run, index, leg, client_order_id, exc)
        except BrokerError as exc:
            ambiguous = (
                exc.code == DUPLICATE_CLIENT_ORDER_ID
                or exc.status == 0
                or exc.status >= 500
                or 300 <= exc.status < 400
            )
            if ambiguous:
                return await self._reconcile(run, index, leg, client_order_id, exc)
            return self._leg_error(run, index, leg, client_order_id, exc)
        except ValueError as exc:  # raised before any request was sent
            return self._leg_error(run, index, leg, client_order_id, exc)
        except Exception as exc:  # noqa: BLE001 - HTTP-level or unknown: the order may exist
            return await self._reconcile(run, index, leg, client_order_id, exc)

        self._log(run.session_id, "broker_response", {
            "ticket_id": run.ticket.id,
            "leg": index,
            "client_order_id": client_order_id,
            "broker_order_id": order.id,
            "symbol": order.symbol,
            "status": order.status,
        })
        self._record_order(run.ticket.id, client_order_id, order)
        return _LegOutcome(index, leg, client_order_id, "placed", order=order)

    def _leg_error(
        self, run: _Run, index: int, leg: OrderLeg, client_order_id: str, exc: Exception
    ) -> _LegOutcome:
        message = self._short(exc)
        self._log(run.session_id, "broker_response", {
            "ticket_id": run.ticket.id,
            "leg": index,
            "client_order_id": client_order_id,
            "symbol": leg.symbol,
            "status": "error",
            "http_status": getattr(exc, "status", None),
            "code": getattr(exc, "code", None),
            "message": message,
        })
        return _LegOutcome(index, leg, client_order_id, "error", error=message)

    async def _reconcile(
        self, run: _Run, index: int, leg: OrderLeg, client_order_id: str, cause: Exception
    ) -> _LegOutcome:
        """ONE lookup by client_order_id - never a resubmission."""
        base = {
            "ticket_id": run.ticket.id,
            "leg": index,
            "client_order_id": client_order_id,
            "symbol": leg.symbol,
            "cause": self._short(cause),
        }
        try:
            found = await self._broker.get_order_by_client_id(client_order_id)
        except Exception as exc:  # noqa: BLE001 - the leg's fate is unknown; never resubmit
            self._log(run.session_id, "reconcile", {
                **base, "outcome": "failed", "message": self._short(exc),
            })
            self._record_uncertain(run.ticket.id, client_order_id)
            return _LegOutcome(
                index, leg, client_order_id, "uncertain",
                error="the broker did not confirm this order and the follow-up lookup failed; "
                      "check open orders before trying again",
            )
        if found is None:
            self._log(run.session_id, "reconcile", {**base, "outcome": "not_found"})
            return _LegOutcome(
                index, leg, client_order_id, "error",
                error=f"not placed (the broker has no order with this id; {base['cause']})",
            )
        self._log(run.session_id, "reconcile", {
            **base, "outcome": "found", "broker_order_id": found.id, "status": found.status,
        })
        self._record_order(run.ticket.id, client_order_id, found)
        return _LegOutcome(index, leg, client_order_id, "placed", order=found, reconciled=True)

    def _persist(self, run: _Run) -> None:
        """Write `broker_order_ids` and the outcome status in one CAS (once). A leg whose submit
        started but has no outcome (interrupted mid-call) counts as uncertain."""
        if run.persisted:
            return
        run.persisted = True
        ticket = run.ticket
        by_index = {o.index: o for o in run.outcomes}
        broker_order_ids: list[str | None] = []
        maybe_placed = False
        for index in range(len(ticket.legs)):
            outcome = by_index.get(index)
            if outcome is not None and outcome.state == "placed":
                broker_order_ids.append(outcome.order.id)
                maybe_placed = True
            else:
                broker_order_ids.append(None)
                if index < run.attempted and (outcome is None or outcome.state == "uncertain"):
                    maybe_placed = True
        run.maybe_placed = maybe_placed
        to = TicketStatus.submitted if maybe_placed else TicketStatus.failed
        try:
            run.final_ticket = self._tickets.transition(
                ticket.id, {TicketStatus.submitting}, to, broker_order_ids=broker_order_ids
            )
        except (TicketStateError, ValueError) as exc:
            logger.error("could not persist ticket %s as %s: %s", ticket.id, to.value, exc)

    async def _poll(self, run: _Run) -> None:
        pending = [
            o for o in run.outcomes
            if o.state == "placed" and o.order.status.lower() not in STOP_POLL_STATUSES
        ]
        for attempt in range(1, POLL_ATTEMPTS + 1):
            if not pending:
                return
            await self._sleep(POLL_INTERVAL_SEC)
            still: list[_LegOutcome] = []
            for outcome in pending:
                base = {
                    "ticket_id": run.ticket.id,
                    "leg": outcome.index,
                    "client_order_id": outcome.client_order_id,
                    "broker_order_id": outcome.order.id,
                    "symbol": outcome.leg.symbol,
                    "poll": attempt,
                }
                try:
                    snapshot = await self._broker.get_order(outcome.order.id)
                except Exception as exc:  # noqa: BLE001 - polling is best effort
                    self._log(run.session_id, "order_status", {
                        **base, "status": "unknown", "message": self._short(exc),
                    })
                    still.append(outcome)
                    continue
                outcome.order = snapshot
                self._record_order(run.ticket.id, outcome.client_order_id, snapshot)
                self._log(run.session_id, "order_status", {
                    **base,
                    "status": snapshot.status,
                    "filled_qty": snapshot.filled_qty,
                    "filled_avg_price": snapshot.filled_avg_price,
                })
                if snapshot.status.lower() not in STOP_POLL_STATUSES:
                    still.append(outcome)
            pending = still

    # --- results --------------------------------------------------------------------------------

    def _submission_result(self, run: _Run) -> ConfirmResult:
        final = run.final_ticket
        placed = [o for o in run.outcomes if o.state == "placed"]
        uncertain = [o for o in run.outcomes if o.state == "uncertain"]
        failed = [o for o in run.outcomes if o.state == "error"]
        total = len(run.ticket.legs)
        head = f"{len(placed)} of {total} orders placed"
        if uncertain:
            head += f", {len(uncertain)} uncertain"
        if failed:
            head += f", {len(failed)} failed"
        head += "."
        if not placed and not uncertain:
            head += " No order was placed."
        lines = [head]
        for outcome in run.outcomes:
            label = _leg_label(outcome.index, outcome.leg)
            if outcome.state == "placed":
                via = ", found by reconcile" if outcome.reconciled else ""
                lines.append(f"{label}: placed (status {outcome.order.status}{via}).")
            elif outcome.state == "uncertain":
                lines.append(f"{label}: uncertain - {outcome.error}; do not resubmit.")
            else:
                lines.append(f"{label}: error - {outcome.error}.")
        # Same decision `_persist` wrote (from the outcomes, not a re-read), so even if that CAS
        # somehow failed, placed orders are still reported as submitted.
        return ConfirmResult(
            "submitted" if run.maybe_placed else "error",
            self._latest(run.ticket.id, final),
            [o.order for o in placed],
            " ".join(lines),
        )

    def _crash_result(self, run: _Run) -> ConfirmResult:
        if run.outcomes and run.persisted:
            return self._submission_result(run)
        nothing = "nothing was submitted" if run.attempted == 0 else (
            "some orders may have been submitted; check open orders before trying again"
        )
        return ConfirmResult(
            "error", self._latest(run.ticket.id, run.final_ticket), [],
            f"Unexpected error while confirming; {nothing}.",
        )

    def _outcome_payload(self, run: _Run) -> list[dict[str, Any]]:
        return [
            {
                "leg": o.index,
                "symbol": o.leg.symbol,
                "client_order_id": o.client_order_id,
                "outcome": o.state,
                "broker_order_id": o.order.id if o.order is not None else None,
                "order_status": o.order.status if o.order is not None else None,
                "error": o.error,
            }
            for o in run.outcomes
        ]

    def _finish(
        self, session_id: str, ticket_id: str, result: ConfirmResult, run: _Run | None = None
    ) -> ConfirmResult:
        payload: dict[str, Any] = {
            "ticket_id": str(ticket_id)[:64],
            "status": result.status,
            "message": result.message,
        }
        if result.new_ticket is not None:
            payload["new_ticket_id"] = result.new_ticket.id
        if run is not None:
            payload["legs"] = self._outcome_payload(run)
        self._log(session_id, "result", payload)
        return result

    # --- reject / refresh -----------------------------------------------------------------------

    async def reject(self, ticket_id: str, session_id: str) -> ConfirmResult:
        loaded = self._load(ticket_id, session_id)
        if isinstance(loaded, ConfirmResult):
            result = loaded
        else:
            try:
                ticket = self._tickets.transition(
                    loaded.id, {TicketStatus.proposed}, TicketStatus.rejected_by_user
                )
                result = ConfirmResult("rejected", ticket, [], "Rejected by user")
            except TicketStateError:
                current = self._latest(loaded.id, loaded)
                result = ConfirmResult("rejected", current, [], f"ticket is {current.status.value}")
        self._log(session_id, "reject", {
            "ticket_id": str(ticket_id)[:64], "status": result.status, "message": result.message,
        })
        return result

    async def refresh(self, ticket_id: str, session_id: str) -> ConfirmResult:
        """Re-quote a still-proposed ticket at current prices into a new ticket (which supersedes
        it). Never submits."""
        result = await self._refresh(ticket_id, session_id)
        payload: dict[str, Any] = {
            "ticket_id": str(ticket_id)[:64], "status": result.status, "message": result.message,
        }
        if result.new_ticket is not None:
            payload["new_ticket_id"] = result.new_ticket.id
        self._log(session_id, "refresh", payload)
        return result

    async def _refresh(self, ticket_id: str, session_id: str) -> ConfirmResult:
        loaded = self._load(ticket_id, session_id)
        if isinstance(loaded, ConfirmResult):
            return loaded
        if loaded.status != TicketStatus.proposed:
            return ConfirmResult("rejected", loaded, [], f"ticket is {loaded.status.value}")
        try:
            ctx = await self.build_context([leg.symbol for leg in loaded.legs])
            mandate = self._mandate_for(session_id)
            decisions = evaluate(loaded.legs, ctx, mandate)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the endpoint
            logger.warning("refresh of ticket %s failed: %s", loaded.id, self._short(exc))
            return ConfirmResult(
                "error", loaded, [], f"Could not refresh the quote ({self._short(exc)})."
            )
        try:
            new_ticket = self._new_ticket(loaded, decisions, mandate)
        except ValueError:
            reasons = _block_reasons(decisions) or "every leg was rejected"
            return ConfirmResult(
                "rejected", loaded, [], f"No leg can be placed at current prices: {reasons}"
            )
        return ConfirmResult(
            "requote", self._latest(loaded.id, loaded), [],
            "Re-quoted at current prices; review the new ticket.", new_ticket=new_ticket,
        )

    # --- kill switch ----------------------------------------------------------------------------

    async def kill_switch(self, suspend_broker: bool = False) -> None:
        """Turn trading off locally (`settings.trading_enabled = False`), and optionally suspend
        trading at the broker and cancel every open order (up to `OPEN_ORDERS_LIMIT`). Broker
        failures are logged and recorded in the ledger, never raised: the local switch is already
        off, and one failed step must not stop the others."""
        self._settings.trading_enabled = False
        payload: dict[str, Any] = {"trading_enabled": False, "suspend_broker": suspend_broker}
        if suspend_broker:
            try:
                await self._broker.set_suspend_trade(True)
                payload["broker_suspended"] = True
            except Exception as exc:  # noqa: BLE001 - keep going: cancel what we can
                logger.error("kill switch: broker suspension failed: %s", self._short(exc))
                payload["broker_suspended"] = False
                payload["suspend_error"] = self._short(exc)
            cancelled: list[dict[str, Any]] = []
            cancel_failures: list[dict[str, Any]] = []
            try:
                open_orders = await self._broker.orders(status="open", limit=OPEN_ORDERS_LIMIT)
            except Exception as exc:  # noqa: BLE001
                logger.error("kill switch: listing open orders failed: %s", self._short(exc))
                payload["list_error"] = self._short(exc)
                open_orders = []
            for order in open_orders:
                try:
                    await self._broker.cancel_order(order.id)
                    cancelled.append({"broker_order_id": order.id, "symbol": order.symbol})
                except Exception as exc:  # noqa: BLE001 - each cancel is independent
                    logger.warning(
                        "kill switch: cancel of order %s failed: %s", order.id, self._short(exc)
                    )
                    cancel_failures.append({
                        "broker_order_id": order.id, "symbol": order.symbol,
                        "message": self._short(exc),
                    })
            payload["cancelled"] = cancelled
            payload["cancel_failures"] = cancel_failures
        self._log(SYSTEM_SESSION, "kill_switch", payload)

    # --- plumbing -------------------------------------------------------------------------------

    def _load(self, ticket_id: str, session_id: str) -> Ticket | ConfirmResult:
        try:
            ticket = self._tickets.get(ticket_id)
        except TicketStateError:
            logger.warning("ticket %s could not be parsed", ticket_id)
            return ConfirmResult(
                "error", None, [], "This ticket could not be read; ask for a fresh quote."
            )
        if ticket is None or ticket.session_id != session_id:
            return ConfirmResult("not_found", None, [], "Ticket not found.")
        return ticket

    def _latest(self, ticket_id: str, fallback: Ticket | None) -> Ticket | None:
        """The stored ticket, for display only - never used to decide anything."""
        try:
            return self._tickets.get(ticket_id) or fallback
        except TicketStateError:
            return fallback

    def _short(self, exc: BaseException) -> str:
        """A short, single-line, secret-scrubbed description of `exc` for users and the ledger."""
        text = exc.message if isinstance(exc, BrokerError) else str(exc)
        text = text or type(exc).__name__
        text = self._ledger.redact_text(" ".join(text.split()))
        text = "".join(ch for ch in text if ch.isprintable())
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[: _MAX_MESSAGE_CHARS - 1] + "…"
        return text

    def _log(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._ledger.append(session_id, kind, payload)
        except Exception:  # noqa: BLE001 - an audit write must never abort an order flow
            logger.exception("ledger append failed for kind %s", kind)

    def _record_order(self, ticket_id: str, client_order_id: str, order: OrderSnapshot) -> None:
        now = self._now().astimezone(timezone.utc).isoformat()
        try:
            self._db.execute(
                "INSERT INTO orders (ticket_id, client_order_id, broker_order_id, body, "
                "updated_at, created_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET "
                "broker_order_id=excluded.broker_order_id, body=excluded.body, "
                "updated_at=excluded.updated_at",
                (ticket_id, client_order_id, order.id, order.model_dump_json(), now, now),
            )
        except Exception:  # noqa: BLE001 - the ticket still records the broker id
            logger.exception("could not record order %s locally", client_order_id)

    def _record_uncertain(self, ticket_id: str, client_order_id: str) -> None:
        now = self._now().astimezone(timezone.utc).isoformat()
        try:
            self._db.execute(
                "INSERT INTO orders (ticket_id, client_order_id, broker_order_id, body, "
                "updated_at, created_at) VALUES (?,?,NULL,NULL,?,?) "
                "ON CONFLICT(client_order_id) DO NOTHING",
                (ticket_id, client_order_id, now, now),
            )
        except Exception:  # noqa: BLE001
            logger.exception("could not record uncertain order %s locally", client_order_id)
