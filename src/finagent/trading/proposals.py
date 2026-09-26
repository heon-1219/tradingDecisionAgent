"""Ticket store: proposal lifecycle, canonical hash, TTL expiry, and supersede.

A `Ticket` (see `trading/models.py`) is the server-side record of a proposed order set awaiting
user confirmation; `POST /api/tickets/{id}/confirm` is the *only* path that reaches
`broker.submit_order` (see `global-constraints.md`), and it must act on exactly what was proposed
- that is what `canonical_hash` binds. The hash covers every field that matters for what will
actually be submitted (the legs, and each decision's verdict/leg/estimates/reason codes) but
deliberately excludes reason *messages*, so a later wording tweak to a risk message never breaks a
previously-shown ticket's hash, while any numeric or verdict change always does.

`TicketStore` persists the full `Ticket` as JSON in `tickets.body` (round-tripped with
`model_dump_json`/`model_validate_json`, so nothing is lost - Decimals, tz-aware datetimes, nested
legs/decisions); `status`/`created_at`/`expires_at` are duplicated into their own columns purely so
`active()`/`expire_stale()` can filter/sort in SQL without deserializing every row. Both copies are
always written together, so a reader never sees the column and the body disagree.

Two lifecycle operations need to be atomic against a concurrent call:

- `create()` must mark the session's currently-active (`proposed`) ticket as `superseded` and
  insert the new one as a single unit - no other `create()` for the same session may run between
  the two steps. Like `research/evidence.py`'s `EvidenceStore.add()` (count-then-insert), this is
  done with the store's own `threading.Lock`, not by asking `AppDB` for a multi-statement
  transaction; `AppDB.execute()` already serializes each individual statement under its own lock,
  and wrapping both statements between one acquisition of `TicketStore`'s lock is enough to keep
  two concurrent `create()` calls from interleaving.
- `transition()` is a compare-and-set: `UPDATE tickets SET status=? WHERE id=? AND status IN (...)`
  is one SQL statement, so `AppDB`'s own per-statement lock is *itself* sufficient to guarantee that
  when two callers race to transition the same ticket out of the same `expect` set, exactly one
  `UPDATE` sees a matching row (`cur.rowcount == 1`) and the other sees zero - the loser's `WHERE`
  clause can no longer match once the winner's write has committed. Only after winning does this
  method re-serialize the ticket's JSON body (merging in `to` and any given `fields`); a second
  writer can't be doing that at the same time for the same ticket, because it never won the CAS.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from finagent.store.appdb import AppDB
from finagent.trading.models import OrderLeg, RiskDecision, Ticket, TicketStatus

# RFC 4648 base32 alphabet (A-Z, 2-7) - no padding, no ambiguous characters (0/1/8 excluded).
_BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_TICKET_ID_RANDOM_CHARS = 10

# The only extra fields `transition()` may write into the ticket body alongside the new status.
_ALLOWED_TRANSITION_FIELDS = frozenset({"broker_order_ids"})


class TicketStateError(Exception):
    """Raised by `TicketStore.transition` when `ticket_id` is unknown, or its current status is
    not one of `expect` - including when another caller won a concurrent race first. The ticket
    row is left completely unchanged."""


def utcnow() -> datetime:
    """Default `clock` for `TicketStore`: the current tz-aware UTC time."""
    return datetime.now(timezone.utc)


def _new_ticket_id() -> str:
    """"T" + 10 characters drawn from the RFC 4648 base32 alphabet via `secrets` (cryptographically
    random, not guessable/enumerable - this id also appears in `client_order_ids`, which Alpaca
    exposes back to the account)."""
    return "T" + "".join(secrets.choice(_BASE32_ALPHABET) for _ in range(_TICKET_ID_RANDOM_CHARS))


def canonical_hash(legs: list[OrderLeg], decisions: list[RiskDecision], ticket_id: str) -> str:
    """sha256 hex digest of the sorted-key, tightly-separated JSON of exactly what a ticket commits
    to: the (risk-normalized) legs, each decision's index/verdict/leg/reason codes+severities
    (never the reason *message* text) and its numeric estimates, and the ticket id itself (so two
    tickets built from identical legs/decisions never collide). Every Decimal becomes a JSON
    string via pydantic's own `mode="json"` dump (exact - never a float), matching the rest of the
    project's "money is Decimal" constraint.
    """
    obj = {
        "ticket_id": ticket_id,
        "legs": [leg.model_dump(mode="json") for leg in legs],
        "decisions": [
            {
                "index": decision.index,
                "verdict": decision.verdict,
                "leg": decision.leg.model_dump(mode="json"),
                "reasons": [
                    {"code": reason.code, "severity": reason.severity}
                    for reason in decision.reasons
                ],
                "est_price": None if decision.est_price is None else str(decision.est_price),
                "est_cost_usd": (
                    None if decision.est_cost_usd is None else str(decision.est_cost_usd)
                ),
            }
            for decision in decisions
        ],
    }
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TicketStore:
    """SQLite-backed store for `Ticket`s: proposal, session-scoped supersede, atomic lifecycle
    transitions, and TTL-based expiry. See the module docstring for the concurrency design."""

    def __init__(self, db: AppDB, clock: Callable[[], datetime] = utcnow):
        self._db = db
        self._clock = clock
        self._lock = threading.Lock()  # serializes supersede-then-insert in create()

    def create(
        self,
        session_id: str,
        legs: list[OrderLeg],
        decisions: list[RiskDecision],
        *,
        thesis: str,
        risks: str,
        invalidation: str,
        evidence_ids: list[str],
        mandate_version: int,
        quote_as_of: datetime,
        ttl_sec: int = 300,
    ) -> Ticket:
        """Build and store a new `proposed` ticket for `legs`/`decisions` (already risk-normalized
        by the caller - the same objects `canonical_hash` binds to), superseding whatever ticket
        was previously active for `session_id`."""
        ticket_id = _new_ticket_id()
        client_order_ids = [f"fa-{ticket_id}-{i}" for i in range(len(legs))]
        digest = canonical_hash(legs, decisions, ticket_id)
        created_at = self._clock()
        expires_at = created_at + timedelta(seconds=ttl_sec)
        ticket = Ticket(
            id=ticket_id,
            session_id=session_id,
            status=TicketStatus.proposed,
            legs=legs,
            decisions=decisions,
            thesis=thesis,
            risks=risks,
            invalidation=invalidation,
            evidence_ids=evidence_ids,
            mandate_version=mandate_version,
            quote_as_of=quote_as_of,
            created_at=created_at,
            expires_at=expires_at,
            hash=digest,
            client_order_ids=client_order_ids,
        )
        body = ticket.model_dump_json()

        with self._lock:
            for row in self._db.query(
                "SELECT id, body FROM tickets WHERE session_id=? AND status=?",
                (session_id, TicketStatus.proposed.value),
            ):
                superseded = Ticket.model_validate_json(row["body"]).model_copy(
                    update={"status": TicketStatus.superseded}
                )
                self._db.execute(
                    "UPDATE tickets SET status=?, body=? WHERE id=?",
                    (TicketStatus.superseded.value, superseded.model_dump_json(), row["id"]),
                )
            self._db.execute(
                "INSERT INTO tickets (id, session_id, status, body, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    ticket_id,
                    session_id,
                    TicketStatus.proposed.value,
                    body,
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        rows = self._db.query("SELECT body FROM tickets WHERE id=?", (ticket_id,))
        if not rows:
            return None
        return Ticket.model_validate_json(rows[0]["body"])

    def active(self, session_id: str) -> Ticket | None:
        """The newest `proposed` ticket for `session_id` that has not yet expired against
        `clock()`. An expired `proposed` ticket is simply not returned here - it stays `proposed`
        in storage until `expire_stale()` flips it."""
        rows = self._db.query(
            "SELECT body FROM tickets WHERE session_id=? AND status=?",
            (session_id, TicketStatus.proposed.value),
        )
        now = self._clock()
        candidates = sorted(
            (Ticket.model_validate_json(row["body"]) for row in rows),
            key=lambda t: t.created_at,
            reverse=True,
        )
        for ticket in candidates:
            if ticket.expires_at > now:
                return ticket
        return None

    def transition(
        self,
        ticket_id: str,
        expect: set[TicketStatus],
        to: TicketStatus,
        **fields,
    ) -> Ticket:
        """Atomic compare-and-set: move `ticket_id` to status `to` only if its current status is
        in `expect`. Raises `TicketStateError` (leaving the row untouched) when the id is unknown
        or the current status isn't in `expect` - including the losing side of a race, see the
        module docstring. `fields` may only set `broker_order_ids`; any other key raises
        `ValueError` without touching the database."""
        bad_fields = set(fields) - _ALLOWED_TRANSITION_FIELDS
        if bad_fields:
            raise ValueError(
                f"transition() does not accept field(s): {', '.join(sorted(bad_fields))}"
            )

        expect_values = [status.value for status in expect]
        placeholders = ",".join("?" for _ in expect_values)
        cur = self._db.execute(
            f"UPDATE tickets SET status=? WHERE id=? AND status IN ({placeholders})",
            (to.value, ticket_id, *expect_values),
        )
        if cur.rowcount != 1:
            raise TicketStateError(
                f"ticket {ticket_id!r} does not exist or is not in one of the expected "
                f"states {sorted(expect_values)}"
            )

        rows = self._db.query("SELECT body FROM tickets WHERE id=?", (ticket_id,))
        data = json.loads(rows[0]["body"])
        data["status"] = to.value
        data.update(fields)
        new_ticket = Ticket.model_validate(data)
        self._db.execute(
            "UPDATE tickets SET body=?, status=? WHERE id=?",
            (new_ticket.model_dump_json(), to.value, ticket_id),
        )
        return new_ticket

    def expire_stale(self) -> int:
        """Flip every `proposed` ticket whose `expires_at` is at or before `clock()` to `expired`.
        Returns how many were flipped. Each row's own `UPDATE ... WHERE id=? AND status=?` is a
        compare-and-set guarding against a concurrent `transition()` moving the same ticket out of
        `proposed` first (in which case that row is simply skipped, not counted)."""
        now = self._clock()
        rows = self._db.query(
            "SELECT id, body FROM tickets WHERE status=?", (TicketStatus.proposed.value,)
        )
        count = 0
        for row in rows:
            ticket = Ticket.model_validate_json(row["body"])
            if ticket.expires_at > now:
                continue
            expired = ticket.model_copy(update={"status": TicketStatus.expired})
            cur = self._db.execute(
                "UPDATE tickets SET status=?, body=? WHERE id=? AND status=?",
                (
                    TicketStatus.expired.value,
                    expired.model_dump_json(),
                    row["id"],
                    TicketStatus.proposed.value,
                ),
            )
            if cur.rowcount:
                count += 1
        return count
