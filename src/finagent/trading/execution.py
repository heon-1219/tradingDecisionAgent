"""Executor: the only code path that submits orders.

`confirm()` is reached solely from the user's Confirm click (`POST /api/tickets/{id}/confirm`); the
LLM never gets an order-placing tool. Paper only: the broker is `AlpacaPaper`, whose host is a
hard-coded constant, and `build_context` - which every proposal, confirm re-check and refresh runs
first - refuses an account that says it is not a paper account (`AccountSnapshot.paper is False`):
it calls `on_not_paper` (the runtime then drops the broker) and raises `NotPaperAccount`. A
confirm then ends `rejected` with the ticket `failed` and nothing submitted; a refresh ends
`rejected` and creates nothing; `propose_trades` creates no ticket.

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
   one is created from the new decisions (`requote`). For a *limit* leg the estimated price is the
   limit price itself, so the drift check can never trigger for it - the limit is what protects
   it (and a quote that moves far from it trips the risk engine's price band instead, which is a
   verdict change). A new *warn*-level reason at confirm time (say, the market has closed since the
   proposal) is not a verdict change, so it does not requote and is not surfaced. If every leg is
   now rejected, no new ticket can exist and the result is `rejected` with the reasons. The requote
   path never submits. If the re-check itself fails (the broker or market data is unreachable), the
   claimed ticket is marked `failed` and nothing is submitted: a claimed ticket can't go back to
   `proposed`.
8. **Submit** each leg once, sequentially, with its pre-assigned `client_order_id`. Before every
   leg the run checks for a stop: a leg reached after the kill switch was thrown, or after this
   executor learned that its account is not a paper account (`mark_not_paper`: the runtime calls
   it when it drops the broker, whichever path noticed), is not submitted. If no leg was submitted
   at all, the result is the stop's own: `disabled` for the kill switch, `rejected` with the
   not-paper message otherwise, and the ticket is `failed`. There are no blind retries. An
   `OrderUncertain` (timeout, 5xx, 3xx, unparseable 2xx), a 422 duplicate `client_order_id` (code
   40010001), an HTTP-level failure, a `ValueError` *subclass* (a `JSONDecodeError`, a pydantic
   `ValidationError` or a `UnicodeDecodeError` can all surface while reading a response, i.e. after
   the order may have been placed), or any other unexpected exception may mean the order reached
   Alpaca, so it is reconciled by `get_order_by_client_id` lookups (never a resubmission):
   - an order whose `client_order_id`, symbol and side match the leg -> the leg is placed;
   - no order -> one more lookup `RECONCILE_RETRY_SEC` later (an order that just reached Alpaca
     may not be visible yet); still none -> the leg is `uncertain` ("not found at lookup time";
     after a 422 duplicate, "already used"), never "not placed";
   - an order that does *not* match the leg, or a lookup that itself failed -> the leg is
     `uncertain`: never adopted, never resubmitted.
   A clean 4xx `BrokerError`, or a plain `ValueError` (exactly that type: the broker raises it for
   an invalid id or symbol before any request is sent), is a plain leg error. Every placed or
   reconciled order is written to the local `orders` table; so is an uncertain leg (with no broker
   id), so it still counts towards the daily order limit.
9. **Persist before polling.** `broker_order_ids` (aligned with the legs, `None` where no order is
   known) and the new status are written in one CAS: `submitted` if at least one leg is placed or
   uncertain, else `failed`.
10. **Poll** each placed order up to `POLL_ATTEMPTS` times, `POLL_INTERVAL_SEC` apart (the injected
    `sleep`), stopping once its status is in `STOP_POLL_STATUSES` - final, or acknowledged
    (`accepted`, `new`, `partially_filled`). A polling failure is recorded, never raised. The
    local `orders` row is updated on every poll.
11. The result carries the latest ticket, the latest snapshot of every placed order, and a short
    English summary with each leg's outcome. Status: `submitted` only when the outcome CAS actually
    wrote `submitted`; `error` when every leg failed, or when the outcome could not be written
    (then the message says orders may have been placed - success is never claimed for a ticket
    whose state wasn't saved).

**Interruptions.** If anything escapes after the claim - a bug, or cancellation of the calling task
mid-call - every leg whose submit had started but has no recorded outcome is treated as
`uncertain`: it gets an `orders` row (no broker id) and appears as uncertain in the ledger. Then
the outcome is persisted (`submitted` if any leg was placed or uncertain, else `failed`), so the
ticket is never stranded in `submitting`. `run.persisted` is set only once that CAS has succeeded
(or raised `TicketStateError`, meaning the ticket is no longer ours to move); any other failure (a
locked database, say) is retried once by the handler. Cancellation is then re-raised; any other
exception becomes an `error` result (or the normal submission result, if everything was saved).

**refresh(ticket_id, session_id)** re-evaluates a `proposed` ticket at current prices, or an
`expired` one when it is the session's newest ticket (`TicketStore.latest`), so that the person who
let a quote lapse can ask for a fresh one. A refused refresh is `rejected` with a `reason`:
`not_newest` (an expired ticket that is not the newest, or one already re-quoted), `ticket_state`
(any other status), `not_paper`, or `no_executable_legs` (the risk check blocks every leg at
current prices under the current mandate; `block_codes` lists its codes). Like confirm, it acts on
the old ticket only through a CAS: after the re-evaluation it first checks there is something
executable (if not: `rejected`, and the old ticket is left untouched), then moves a proposed ticket
`proposed -> superseded` - if a confirm, reject or expiry got there first, that CAS fails and
refresh returns `rejected` without creating anything - and only then creates the new ticket. An
expired ticket is terminal and stays as it is. So a refresh racing a confirm can never leave a
second live copy of the same trade.

**One active ticket per session.** `TicketStore.create` supersedes every other `proposed` ticket in
the session, so a requote (from confirm or refresh) also supersedes any unrelated proposal made in
the same session meanwhile. This is the store's one-active-ticket rule, kept deliberately.

**reject_ticket(tickets, ledger, ticket_id, session_id)** is the whole of a rejection: a local CAS
`proposed -> rejected_by_user` and a `reject` ledger entry. It is a module-level function because
it needs no broker: `Executor.reject` delegates to it, and the API calls it directly when no broker
is configured (so there is no executor) to dismiss a ticket.

**recover_stuck()** runs once at startup, before any confirm can be in flight. A ticket still in
`submitting` means the process died mid-confirm. Each of its legs is looked up once with
`get_order_by_client_id` (never resubmitted): a matching order is recorded and adopted, a missing
one is "not placed", a mismatched one is uncertain (recorded without a broker id). If any lookup
fails, the ticket is left as it is and retried at the next startup. Otherwise the outcome is
written with the same CAS as a confirm: `submitted` if any leg is placed or uncertain, else
`failed`.

**ticket_orders(ticket_id, session_id)** serves order polling after confirm: the ticket and its
local `orders` rows, each non-final order re-read with `get_order` (a read; never a submit). A
changed status updates the local row and writes an `order_status` ledger entry.

**Ledger.** Kinds: `confirm_request`, `claim`, `requote`, `submit`, `broker_response`,
`reconcile`, `order_status`, `result` (confirm); `reject`; `refresh`; `recover` and `reconcile`
(startup recovery, `source: "startup"`); `order_status` (`source: "poll"`); `kill_switch` (under
the session id `SYSTEM_SESSION`); and `proposal` (`source: "requote"`, with `supersedes`) for the
new ticket a confirm-time re-quote or a refresh creates - the ticket as shown, like the entry the
chat toolbox writes for a ticket the model proposed. Payloads carry symbols as *values*, never as
dict keys: the ledger redacts the value of any dict key that looks secret-like, and "KEY"
(KeyCorp) is a real ticker.
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
from finagent.trading.proposals import TicketStateError, TicketStore, ticket_audit_payload
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
NOT_PAPER_MESSAGE = (
    "The connected Alpaca account is not a paper account, so nothing was submitted. This app trades paper "
    "accounts only: trading is disconnected until paper-trading keys are saved in Settings."
)
_STOP_ERRORS = {
    "kill_switch": "not submitted: trading was disabled (kill switch)",
    "not_paper": "not submitted: the account turned out not to be a paper account, so trading was "
                 "disconnected",
}
# After an uncertain submit, a lookup that finds nothing is repeated once after this pause: an order
# that just reached Alpaca may not be visible yet. A lookup, never a resubmit.
RECONCILE_RETRY_SEC = 2.0
_DEFAULT_TTL_SEC = 300
_NEWER_TICKET = "ticket is expired and a newer ticket exists; nothing was re-quoted"
_CENT = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")
_HUNDRED = Decimal("100")

RefreshRefusal = Literal["not_newest", "ticket_state", "not_paper", "no_executable_legs"]

ConfirmStatus = Literal[
    "submitted", "requote", "rejected", "disabled", "expired", "not_found", "error"
]


class NotPaperAccount(Exception):
    """The broker's account says it is not a paper account (`AccountSnapshot.paper is False`)."""

    def __init__(self) -> None:
        super().__init__("the Alpaca account is not a paper account")


@dataclass
class ConfirmResult:
    status: ConfirmStatus
    ticket: Ticket | None
    orders: list[OrderSnapshot]
    message: str
    new_ticket: Ticket | None = None
    # Why a refresh was refused (`RefreshRefusal`), None otherwise; for `no_executable_legs`, the
    # distinct block codes of the risk check, sorted.
    reason: str | None = None
    block_codes: list[str] = field(default_factory=list)


@dataclass
class TrackedOrder:
    """One local `orders` row of a ticket, for polling. `order` is None for an uncertain leg (no
    broker id is known) or a row whose stored body cannot be read."""

    client_order_id: str
    broker_order_id: str | None
    order: OrderSnapshot | None


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


def _reconcile_mismatch(found: OrderSnapshot, leg: OrderLeg, client_order_id: str) -> str:
    """Why a reconciled order can't be adopted as this leg's order; empty when it matches."""
    problems = []
    if found.client_order_id != client_order_id:
        problems.append("client_order_id")
    if (found.symbol or "").upper() != leg.symbol:
        problems.append("symbol")
    if (found.side or "").lower() != leg.side:
        problems.append("side")
    return f"{', '.join(problems)} differ" if problems else ""


def _block_reasons(decisions: list[RiskDecision]) -> str:
    parts = []
    for decision in decisions:
        blocks = [r.message for r in decision.reasons if r.severity == "block"]
        if blocks:
            parts.append(f"leg {decision.index + 1} {decision.leg.symbol}: {blocks[0]}")
    return "; ".join(parts)


def _block_codes(decisions: list[RiskDecision]) -> list[str]:
    return sorted({r.code for d in decisions for r in d.reasons if r.severity == "block"})


def stored_orders(db: AppDB, ticket_id: str) -> list[TrackedOrder]:
    """A ticket's local `orders` rows, oldest first, as stored (no broker call)."""
    rows = db.query(
        "SELECT client_order_id, broker_order_id, body FROM orders WHERE ticket_id=? ORDER BY id",
        (ticket_id,),
    )
    tracked: list[TrackedOrder] = []
    for row in rows:
        order: OrderSnapshot | None = None
        if row["body"]:
            try:
                order = OrderSnapshot.model_validate_json(row["body"])
            except Exception:  # noqa: BLE001 - an unreadable row is shown without a snapshot
                logger.warning("order row %s has an unreadable body", row["client_order_id"])
        tracked.append(TrackedOrder(row["client_order_id"], row["broker_order_id"], order))
    return tracked


def _load_ticket(tickets: TicketStore, ticket_id: str, session_id: str) -> Ticket | ConfirmResult:
    """The session's ticket, or the result to return instead: `error` for an unparseable row,
    `not_found` for an unknown id or another session's ticket."""
    try:
        ticket = tickets.get(ticket_id)
    except TicketStateError:
        logger.warning("ticket %s could not be parsed", ticket_id)
        return ConfirmResult(
            "error", None, [], "This ticket could not be read; ask for a fresh quote."
        )
    if ticket is None or ticket.session_id != session_id:
        return ConfirmResult("not_found", None, [], "Ticket not found.")
    return ticket


def _latest_ticket(tickets: TicketStore, ticket_id: str, fallback: Ticket | None) -> Ticket | None:
    try:
        return tickets.get(ticket_id) or fallback
    except TicketStateError:
        return fallback


def reject_ticket(
    tickets: TicketStore, ledger: Ledger, ticket_id: str, session_id: str
) -> ConfirmResult:
    """Reject a ticket: a local CAS `proposed -> rejected_by_user` plus a `reject` ledger entry. It
    needs no broker, so it is a function that `Executor.reject` and the API (when no broker is
    configured, so there is no executor) both call. A ticket that is no longer `proposed` is
    reported with its current status; nothing is changed."""
    loaded = _load_ticket(tickets, ticket_id, session_id)
    if isinstance(loaded, ConfirmResult):
        result = loaded
    else:
        try:
            ticket = tickets.transition(
                loaded.id, {TicketStatus.proposed}, TicketStatus.rejected_by_user
            )
            result = ConfirmResult("rejected", ticket, [], "Rejected by user")
        except TicketStateError:
            current = _latest_ticket(tickets, loaded.id, loaded)
            result = ConfirmResult("rejected", current, [], f"ticket is {current.status.value}")
    try:
        ledger.append(session_id, "reject", {
            "ticket_id": str(ticket_id)[:64], "status": result.status, "message": result.message,
        })
    except Exception as exc:  # noqa: BLE001 - an audit write must never abort the action
        logger.error("ledger append failed for kind reject (%s)", type(exc).__name__)
    return result


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
    """Confirm / reject / refresh tickets, and the kill switch. See the module docstring.

    `mandate_for(session_id)` must return the session's *current* mandate; it is a required
    keyword argument (no silent default), since re-checking against the wrong mandate would let a
    confirm through that the user's latest limits forbid. `db` and `now` default to the
    `TicketStore`'s own DB and clock."""

    def __init__(
        self,
        broker,
        data,
        tickets: TicketStore,
        ledger: Ledger,
        settings,
        *,
        mandate_for: Callable[[str], Mandate],
        db: AppDB | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        now: Callable[[], datetime] | None = None,
        on_not_paper: Callable[[], Any] | None = None,
    ):
        self._broker = broker
        self._on_not_paper = on_not_paper  # told (synchronously) when the account is not paper
        self._data = data
        self._tickets = tickets
        self._ledger = ledger
        self._settings = settings
        self._mandate_for = mandate_for
        self._db = db if db is not None else tickets.db
        self._sleep = sleep
        self._now = now or tickets.clock
        # Set once this executor's broker is known to read a non-paper account (`mark_not_paper`):
        # a confirm in flight submits no further leg, like a thrown kill switch.
        self._not_paper_seen = False
        _ensure_orders_created_at(self._db)

    def mark_not_paper(self) -> None:
        """This executor's broker reads a non-paper account: any confirm still in flight submits
        no further leg. Called by `build_context` when it sees such an account, and by the runtime
        when it drops this executor's broker for that reason (the chat's `get_account` can notice
        first, without the lock, while a confirm sits between legs)."""
        self._not_paper_seen = True

    # --- market context -------------------------------------------------------------------------

    async def build_context(self, symbols: list[str]) -> MarketContext:
        """Account, positions, open orders, assets (for `symbols` only), quotes and clock.

        Quotes cover `symbols` plus every symbol with an open order or a position: the risk
        engine values pending open buys and positions from them. An asset lookup that fails for
        one symbol leaves it out of `assets`, so risk rejects that leg as not tradable."""
        wanted = sorted(_valid_symbols(symbols))
        account = await self._broker.account()
        if getattr(account, "paper", None) is False:
            self._not_paper()
            raise NotPaperAccount()
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

    def _not_paper(self) -> None:
        logger.warning("the broker's account is not a paper account: refusing to trade")
        self.mark_not_paper()
        if self._on_not_paper is not None:
            try:
                self._on_not_paper()
            except Exception as exc:  # noqa: BLE001 - the refusal itself must still happen
                logger.error("the not-paper callback failed (%s)", type(exc).__name__)

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
            self._settle_interrupted(run)
            result = self._crash_result(run)
        except BaseException:  # cancellation: record and persist what we know, then propagate
            self._settle_interrupted(run)
            self._log(session_id, "result", {
                "ticket_id": claimed.id,
                "status": "interrupted",
                "legs": self._outcome_payload(run),
            })
            raise
        return self._finish(session_id, ticket_id, result, run)

    def _settle_interrupted(self, run: _Run) -> None:
        """After something escaped `_run_claimed`: every leg whose submit started but has no
        outcome may have reached the broker, so it becomes `uncertain` (with an `orders` row, so it
        counts towards the daily limit); then the outcome is persisted - this is the one retry if
        the normal path's persist failed. Synchronous only (it may run during cancellation) and
        never raises."""
        recorded = {o.index for o in run.outcomes}
        for index in range(min(run.attempted, len(run.ticket.legs))):
            if index in recorded:
                continue
            client_order_id = run.ticket.client_order_ids[index]
            self._record_uncertain(run.ticket.id, client_order_id)
            run.outcomes.append(_LegOutcome(
                index, run.ticket.legs[index], client_order_id, "uncertain",
                error="interrupted before the broker's answer was known; check open orders "
                      "before trying again",
            ))
        run.outcomes.sort(key=lambda o: o.index)
        try:
            self._persist(run)
        except Exception:  # noqa: BLE001 - logged; the result will not claim success
            logger.exception("could not persist the outcome of ticket %s", run.ticket.id)

    async def _run_claimed(self, run: _Run) -> ConfirmResult:
        ticket = run.ticket
        try:
            ctx = await self.build_context([leg.symbol for leg in ticket.legs])
            mandate = self._mandate_for(run.session_id)
            decisions = evaluate(ticket.legs, ctx, mandate)
        except NotPaperAccount:
            self._persist(run)  # nothing attempted -> failed
            return ConfirmResult(
                "rejected", self._latest(ticket.id, run.final_ticket), [], NOT_PAPER_MESSAGE,
            )
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

        stopped_by: str | None = None
        for index, (leg, client_order_id) in enumerate(zip(ticket.legs, ticket.client_order_ids)):
            stop = self._stop_reason()
            if stop is not None:
                # The kill switch was thrown, or the account turned out not to be a paper account,
                # mid-confirm: submit nothing more.
                stopped_by = stopped_by or stop
                run.outcomes.append(_LegOutcome(
                    index, leg, client_order_id, "error", error=_STOP_ERRORS[stop],
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
        if run.attempted == 0 and stopped_by is not None:
            # Stopped before the first leg (during the re-check): nothing can exist at the broker,
            # so this is the stop's own answer, not an error. The ticket is `failed`.
            return self._stopped_result(run, stopped_by)
        await self._poll(run)
        return self._submission_result(run)

    def _stop_reason(self) -> Literal["kill_switch", "not_paper"] | None:
        """Why no further leg may be submitted now, if anything: checked before every leg."""
        if not self._settings.trading_enabled:
            return "kill_switch"
        if self._not_paper_seen:
            return "not_paper"
        return None

    def _stopped_result(self, run: _Run, stop: str) -> ConfirmResult:
        ticket = self._latest(run.ticket.id, run.final_ticket)
        if stop == "not_paper":
            return ConfirmResult("rejected", ticket, [], NOT_PAPER_MESSAGE)
        return ConfirmResult(
            "disabled", ticket, [],
            "Trading was disabled (kill switch) while this ticket was being re-checked; nothing "
            "was submitted.",
        )

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

    def _is_newest(self, ticket: Ticket) -> bool:
        """Whether `ticket` is still its session's newest ticket. Only that one may be re-quoted
        after it expired: an older one has been replaced already (by a later proposal, or by an
        earlier refresh of this very ticket). An unreadable newer row counts as newer."""
        try:
            newest = self._tickets.latest(ticket.session_id)
        except TicketStateError:
            return False
        return newest is not None and newest.id == ticket.id

    def _new_ticket(self, ticket: Ticket, decisions: list[RiskDecision], mandate: Mandate) -> Ticket:
        """The re-quoted copy of `ticket`, recorded in the ledger as shown (a `proposal` entry,
        like the one the chat toolbox writes for a ticket the model proposed)."""
        new_ticket = self._tickets.create(
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
        try:
            shown = ticket_audit_payload(new_ticket)
        except Exception as exc:  # noqa: BLE001 - an audit entry must never abort a re-quote
            logger.warning(
                "proposal ledger entry for ticket %s not built (%s)",
                new_ticket.id, type(exc).__name__,
            )
        else:
            # `supersedes` names the ticket this one replaces; `replaced_status` says whether that
            # ticket was live (`proposed`: it is now superseded) or had expired (it stays expired).
            replaced = "expired" if ticket.status == TicketStatus.expired else "proposed"
            self._log(new_ticket.session_id, "proposal", {
                "source": "requote", "supersedes": ticket.id, "replaced_status": replaced, **shown,
            })
        return new_ticket

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
        except ValueError as exc:
            if type(exc) is ValueError:  # the broker's own check, before any request was sent
                return self._leg_error(run, index, leg, client_order_id, exc)
            # A subclass (JSONDecodeError, pydantic ValidationError, UnicodeDecodeError, ...) can
            # come from reading a response - i.e. after the order may have been placed.
            return await self._reconcile(run, index, leg, client_order_id, exc)
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
        """Look the order up by client_order_id - never a resubmission. A lookup that finds
        nothing is repeated once after `RECONCILE_RETRY_SEC` (an order that just reached Alpaca
        may not be visible yet); if it still finds nothing, the leg is uncertain, not "not
        placed": after a timeout, a 5xx or an unreadable answer a 404 proves nothing, and after a
        422 duplicate the id is in use somewhere."""
        base = {
            "ticket_id": run.ticket.id,
            "leg": index,
            "client_order_id": client_order_id,
            "symbol": leg.symbol,
            "cause": self._short(cause),
        }
        lookups = 0
        found: OrderSnapshot | None = None
        try:
            for attempt in range(2):
                if attempt:
                    await self._sleep(RECONCILE_RETRY_SEC)
                lookups += 1
                found = await self._broker.get_order_by_client_id(client_order_id)
                if found is not None:
                    break
        except Exception as exc:  # noqa: BLE001 - the leg's fate is unknown; never resubmit
            self._log(run.session_id, "reconcile", {
                **base, "outcome": "failed", "lookups": lookups, "message": self._short(exc),
            })
            self._record_uncertain(run.ticket.id, client_order_id)
            return _LegOutcome(
                index, leg, client_order_id, "uncertain",
                error="the broker did not confirm this order and the follow-up lookup failed; "
                      "check open orders before trying again",
            )
        base["lookups"] = lookups
        if found is None:
            self._log(run.session_id, "reconcile", {**base, "outcome": "not_found"})
            self._record_uncertain(run.ticket.id, client_order_id)
            duplicate = isinstance(cause, BrokerError) and cause.code == DUPLICATE_CLIENT_ORDER_ID
            if duplicate:
                why = ("the broker said this client_order_id was already used, but no order with "
                       "it was found at lookup time")
            else:
                why = f"not found at lookup time ({base['cause']})"
            return _LegOutcome(
                index, leg, client_order_id, "uncertain",
                error=f"{why}; check open orders before trying again",
            )
        mismatch = _reconcile_mismatch(found, leg, client_order_id)
        if mismatch:
            # Never adopt an order that isn't provably this leg's; ours may still exist.
            self._log(run.session_id, "reconcile", {
                **base, "outcome": "mismatch", "found_broker_order_id": found.id,
                "message": mismatch,
            })
            self._record_uncertain(run.ticket.id, client_order_id)
            return _LegOutcome(
                index, leg, client_order_id, "uncertain",
                error=f"the follow-up lookup returned an order that does not match this leg "
                      f"({mismatch}); check open orders before trying again",
            )
        self._log(run.session_id, "reconcile", {
            **base, "outcome": "found", "broker_order_id": found.id, "status": found.status,
        })
        self._record_order(run.ticket.id, client_order_id, found)
        return _LegOutcome(index, leg, client_order_id, "placed", order=found, reconciled=True)

    def _persist(self, run: _Run) -> None:
        """Write `broker_order_ids` and the outcome status in one CAS. A leg whose submit started
        but has no outcome (interrupted mid-call) counts as uncertain.

        `run.persisted` is set only once this is settled: the CAS succeeded, or it raised
        `TicketStateError` (the ticket is no longer ours to move, so retrying can't help;
        `final_ticket` stays None and the result will not claim success). Any other exception - a
        locked database, say - propagates with `run.persisted` still False, so the caller can
        retry."""
        if run.persisted:
            return
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
        to = TicketStatus.submitted if maybe_placed else TicketStatus.failed
        try:
            run.final_ticket = self._tickets.transition(
                ticket.id, {TicketStatus.submitting}, to, broker_order_ids=broker_order_ids
            )
        except TicketStateError as exc:
            logger.error("could not persist ticket %s as %s: %s", ticket.id, to.value, exc)
        run.persisted = True

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
        saved = final is not None and final.status == TicketStatus.submitted
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
        if final is None:
            lines.insert(0, "The ticket's outcome could not be saved; orders may have been placed "
                            "- check open orders before trying again.")
        for outcome in run.outcomes:
            label = _leg_label(outcome.index, outcome.leg)
            if outcome.state == "placed":
                via = ", found by reconcile" if outcome.reconciled else ""
                lines.append(f"{label}: placed (status {outcome.order.status}{via}).")
            elif outcome.state == "uncertain":
                lines.append(f"{label}: uncertain - {outcome.error}; do not resubmit.")
            else:
                lines.append(f"{label}: error - {outcome.error}.")
        # `submitted` only when our own outcome CAS wrote it - never claim success for a ticket
        # whose state wasn't saved.
        return ConfirmResult(
            "submitted" if saved else "error",
            self._latest(run.ticket.id, final),
            [o.order for o in placed],
            " ".join(lines),
        )

    def _crash_result(self, run: _Run) -> ConfirmResult:
        if run.outcomes:  # `_settle_interrupted` has turned every in-flight leg into an outcome
            return self._submission_result(run)
        nothing = "nothing was submitted" if run.attempted == 0 else (
            "orders may have been placed; check open orders before trying again"
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
        return reject_ticket(self._tickets, self._ledger, ticket_id, session_id)

    async def refresh(self, ticket_id: str, session_id: str) -> ConfirmResult:
        """Re-quote a proposed ticket, or the session's newest ticket after it expired, at current
        prices into a new ticket (a proposed one becomes superseded; an expired one stays expired).
        A refusal is `rejected` with a `reason` (see the module docstring). Never submits."""
        result = await self._refresh(ticket_id, session_id)
        payload: dict[str, Any] = {
            "ticket_id": str(ticket_id)[:64], "status": result.status, "message": result.message,
        }
        if result.reason is not None:
            payload["reason"] = result.reason
        if result.new_ticket is not None:
            payload["new_ticket_id"] = result.new_ticket.id
        self._log(session_id, "refresh", payload)
        return result

    async def _refresh(self, ticket_id: str, session_id: str) -> ConfirmResult:
        loaded = self._load(ticket_id, session_id)
        if isinstance(loaded, ConfirmResult):
            return loaded
        lapsed = loaded.status == TicketStatus.expired
        if lapsed and not self._is_newest(loaded):
            return ConfirmResult("rejected", loaded, [], _NEWER_TICKET, reason="not_newest")
        if not lapsed and loaded.status != TicketStatus.proposed:
            return ConfirmResult("rejected", loaded, [], f"ticket is {loaded.status.value}",
                                 reason="ticket_state")
        try:
            ctx = await self.build_context([leg.symbol for leg in loaded.legs])
            mandate = self._mandate_for(session_id)
            decisions = evaluate(loaded.legs, ctx, mandate)
        except NotPaperAccount:
            message = NOT_PAPER_MESSAGE.replace("submitted", "re-quoted")
            return ConfirmResult("rejected", loaded, [], message, reason="not_paper")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the endpoint
            logger.warning("refresh of ticket %s failed: %s", loaded.id, self._short(exc))
            return ConfirmResult(
                "error", loaded, [], f"Could not refresh the quote ({self._short(exc)})."
            )
        if all(d.verdict == "reject" for d in decisions):
            # Nothing executable: leave the old ticket exactly as it is.
            reasons = _block_reasons(decisions) or "every leg was rejected"
            return ConfirmResult(
                "rejected", loaded, [], f"No leg can be placed at current prices: {reasons}",
                reason="no_executable_legs", block_codes=_block_codes(decisions),
            )
        # Supersede a proposed ticket FIRST, through a CAS: if a confirm, reject or expiry moved it
        # while we were re-evaluating, we lose here and create nothing - so there is never a
        # second live copy of a trade that is already being (or has been) submitted. An expired
        # ticket is terminal: it stays expired, and nothing can confirm it.
        if lapsed:
            if not self._is_newest(loaded):  # a chat turn proposed something meanwhile
                return ConfirmResult("rejected", loaded, [], _NEWER_TICKET, reason="not_newest")
            superseded = loaded
        else:
            try:
                superseded = self._tickets.transition(
                    loaded.id, {TicketStatus.proposed}, TicketStatus.superseded
                )
            except TicketStateError:
                current = self._latest(loaded.id, loaded)
                return ConfirmResult(
                    "rejected", current, [],
                    f"ticket is {current.status.value}; nothing was re-quoted",
                    reason="ticket_state",
                )
        try:
            new_ticket = self._new_ticket(loaded, decisions, mandate)
        except ValueError as exc:
            logger.error("refresh of ticket %s could not create the new ticket: %s", loaded.id, exc)
            return ConfirmResult(
                "error", superseded, [],
                "Could not create the re-quoted ticket; ask for a fresh proposal.",
            )
        return ConfirmResult(
            "requote", superseded, [],
            "Re-quoted at current prices; review the new ticket.", new_ticket=new_ticket,
        )

    # --- startup recovery and order polling -------------------------------------------------------

    async def recover_stuck(self) -> int:
        """Settle every ticket left in `submitting` by a crash; returns how many were moved. Call it
        only at startup, before any confirm can be in flight (see the module docstring)."""
        rows = self._db.query(
            "SELECT id FROM tickets WHERE status=? ORDER BY created_at",
            (TicketStatus.submitting.value,),
        )
        moved = 0
        for row in rows:
            try:
                if await self._recover_one(row["id"]):
                    moved += 1
            except Exception as exc:  # noqa: BLE001 - one bad ticket must not stop the sweep
                logger.warning("recovery of ticket %s failed: %s", row["id"], type(exc).__name__)
        return moved

    async def _recover_one(self, ticket_id: str) -> bool:
        try:
            ticket = self._tickets.get(ticket_id)
        except TicketStateError:
            logger.warning("stuck ticket %s could not be parsed; left as is", ticket_id)
            return False
        if ticket is None or ticket.status != TicketStatus.submitting:
            return False
        found: list[OrderSnapshot | None] = []
        for client_order_id in ticket.client_order_ids:
            try:
                found.append(await self._broker.get_order_by_client_id(client_order_id))
            except Exception as exc:  # noqa: BLE001 - unknown fate: leave it for the next startup
                logger.warning(
                    "stuck ticket %s left as is: order lookup failed (%s)",
                    ticket.id, type(exc).__name__,
                )
                return False
        run = _Run(ticket=ticket, session_id=ticket.session_id, attempted=len(ticket.legs))
        for index, (leg, client_order_id, order) in enumerate(
            zip(ticket.legs, ticket.client_order_ids, found)
        ):
            base = {
                "ticket_id": ticket.id, "leg": index, "client_order_id": client_order_id,
                "symbol": leg.symbol, "source": "startup",
            }
            if order is None:
                self._log(run.session_id, "reconcile", {**base, "outcome": "not_found"})
                run.outcomes.append(_LegOutcome(
                    index, leg, client_order_id, "error",
                    error="not placed (the broker has no order with this id)",
                ))
                continue
            mismatch = _reconcile_mismatch(order, leg, client_order_id)
            if mismatch:
                self._log(run.session_id, "reconcile", {
                    **base, "outcome": "mismatch", "found_broker_order_id": order.id,
                    "message": mismatch,
                })
                self._record_uncertain(ticket.id, client_order_id)
                run.outcomes.append(_LegOutcome(
                    index, leg, client_order_id, "uncertain",
                    error=f"the lookup returned an order that does not match this leg ({mismatch})",
                ))
                continue
            self._log(run.session_id, "reconcile", {
                **base, "outcome": "found", "broker_order_id": order.id, "status": order.status,
            })
            self._record_order(ticket.id, client_order_id, order)
            run.outcomes.append(_LegOutcome(
                index, leg, client_order_id, "placed", order=order, reconciled=True,
            ))
        self._persist(run)
        final = run.final_ticket
        self._log(run.session_id, "recover", {
            "ticket_id": ticket.id,
            "status": final.status.value if final is not None else "unchanged",
            "legs": self._outcome_payload(run),
        })
        return final is not None

    async def ticket_orders(
        self, ticket_id: str, session_id: str
    ) -> tuple[Ticket, list[TrackedOrder]] | None:
        """The ticket and its local order rows, non-final orders refreshed from the broker (reads
        only). None for an unknown ticket, another session's, or an unreadable one."""
        try:
            ticket = self._tickets.get(ticket_id)
        except TicketStateError:
            return None
        if ticket is None or ticket.session_id != session_id:
            return None
        tracked: list[TrackedOrder] = []
        for row in stored_orders(self._db, ticket.id):
            order = row.order
            if row.broker_order_id and (
                order is None or order.status.lower() not in FINAL_ORDER_STATUSES
            ):
                order = await self._poll_once(ticket, row.client_order_id, row.broker_order_id, order)
            tracked.append(TrackedOrder(row.client_order_id, row.broker_order_id, order))
        return self._latest(ticket.id, ticket), tracked

    async def _poll_once(
        self, ticket: Ticket, client_order_id: str, broker_order_id: str,
        stored: OrderSnapshot | None,
    ) -> OrderSnapshot | None:
        try:
            fresh = await self._broker.get_order(broker_order_id)
        except Exception as exc:  # noqa: BLE001 - polling is best effort
            logger.info("order %s poll failed: %s", broker_order_id, type(exc).__name__)
            return stored
        changed = stored is None or (fresh.status, fresh.filled_qty, fresh.filled_avg_price) != (
            stored.status, stored.filled_qty, stored.filled_avg_price
        )
        if changed:
            self._record_order(ticket.id, client_order_id, fresh)
            self._log(ticket.session_id, "order_status", {
                "ticket_id": ticket.id,
                "client_order_id": client_order_id,
                "broker_order_id": fresh.id,
                "symbol": fresh.symbol,
                "status": fresh.status,
                "filled_qty": fresh.filled_qty,
                "filled_avg_price": fresh.filled_avg_price,
                "source": "poll",
            })
        return fresh

    # --- kill switch ----------------------------------------------------------------------------

    async def kill_switch(
        self, suspend_broker: bool = False, *, persisted: bool | None = None
    ) -> dict[str, Any]:
        """Turn trading off locally (`settings.trading_enabled = False`), and optionally suspend
        trading at the broker and cancel every open order (up to `OPEN_ORDERS_LIMIT`). Broker
        failures are logged and recorded in the ledger, never raised: the local switch is already
        off, and one failed step must not stop the others. `persisted` (whether the caller has
        already written `TRADING_ENABLED=false` to the env file) goes into the ledger entry when
        given. Returns that entry's payload: `broker_suspended` says whether this call suspended
        trading at the broker."""
        self._settings.trading_enabled = False
        payload: dict[str, Any] = {"trading_enabled": False, "suspend_broker": suspend_broker}
        if persisted is not None:
            payload["persisted"] = persisted
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
        return payload

    # --- plumbing -------------------------------------------------------------------------------

    def _load(self, ticket_id: str, session_id: str) -> Ticket | ConfirmResult:
        return _load_ticket(self._tickets, ticket_id, session_id)

    def _latest(self, ticket_id: str, fallback: Ticket | None) -> Ticket | None:
        """The stored ticket, for display only - never used to decide anything."""
        return _latest_ticket(self._tickets, ticket_id, fallback)

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
