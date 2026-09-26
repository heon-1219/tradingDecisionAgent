"""Ticket store: proposal lifecycle, canonical hash, TTL expiry, and supersede.

A `Ticket` (see `trading/models.py`) is the server-side record of a proposed order set awaiting
user confirmation; `POST /api/tickets/{id}/confirm` is the *only* path that reaches
`broker.submit_order` (see `global-constraints.md`), and it must act on exactly what was proposed
- that is what `canonical_hash` binds. The hash covers every field that matters for what will
actually be submitted (the legs, and each decision's verdict/leg/estimates/reason codes) but
deliberately excludes reason *messages*, so a later wording tweak to a risk message never breaks a
previously-shown ticket's hash, while any numeric or verdict change always does.

**Legs vs. decisions.** A ticket's `legs` are exactly the *executable* legs -
`[d.leg for d in decisions if d.verdict != "reject"]`, in decision order; `decisions` keeps every
decision, including rejects, which are shown to the user as information only (a rejected leg is
never submitted and never appears in `legs` or `client_order_ids`). `create()` derives `legs` from
`decisions` itself: a caller-supplied `legs` is only ever cross-checked against that derived list
(and rejected with `ValueError` if it disagrees) - it is never a second source of truth. A
`decisions` list where every entry is a reject derives no legs at all, and `create()` refuses to
produce a ticket with nothing to execute.

`TicketStore` persists the full `Ticket` as JSON in `tickets.body` (round-tripped with
`model_dump_json`/`model_validate_json`, so nothing is lost - Decimals, tz-aware datetimes, nested
legs/decisions); `status`/`created_at`/`expires_at` are duplicated into their own columns purely so
`active()`/`expire_stale()` can filter/sort in SQL without deserializing every row. Both copies are
always written together, in the same `UPDATE`/`INSERT`, so a reader never sees the column and the
body disagree partway through a write.

**Concurrency.** `create()`, `transition()` and `expire_stale()` all run under the *same*
`threading.RLock` (an `RLock`, not a plain `Lock`, so a call that re-enters this store from within
another - e.g. a test harness or a future orchestrator that confirms a ticket while `create()` is
mid-flight on the same thread - does not deadlock itself; a *different* thread still blocks on it
exactly like a plain lock). This single lock is what makes each of the following true:

- `transition()` is now genuinely one indivisible operation: while holding the lock, it reads the
  ticket's current body, checks it against `expect` (and against the fixed set of terminal
  states, see below), builds and *fully validates* the new `Ticket` (merging in `to` and any given
  `fields`), and only then issues **one** `UPDATE tickets SET status=?, body=? WHERE id=? AND
  status IN (...)`. A `rowcount != 1` (an unknown id, or the status changed out from under it
  between the read and the write) raises `TicketStateError` with the row completely untouched -
  there is no longer a first `UPDATE` that can land before a validation failure, so a bad
  `broker_order_ids` (say) can never strand the row with the status column and the JSON body
  disagreeing. The final `WHERE ... IN (...)` guard is a belt-and-suspenders check against a
  writer outside this lock (there shouldn't be one); the lock itself is what actually prevents two
  callers from interleaving their read and their write.
- `create()`'s supersede-then-insert holds the same lock across marking every previously-`proposed`
  ticket for the session `superseded` and inserting the new one, so a `transition()` racing to move
  one of those tickets out of `proposed` (e.g. a confirm reaching `submitting`) cannot land *inside*
  that window - it either completes first (in which case `create()`'s own `UPDATE ... WHERE
  status='proposed'` guard simply finds no matching row and leaves that ticket alone) or it waits
  for `create()` to finish. Either way, a ticket a confirm has already moved out of `proposed` is
  never clobbered back to `superseded`.
- `expire_stale()` never fights a concurrent `transition()` over the same row for the same reason:
  they share the same lock, and its own per-row `UPDATE ... WHERE status='proposed'` guard is
  redundant-but-harmless defense in depth.

**Terminal states.** `expired`, `superseded`, `rejected_by_user` and `failed` are absorbing: once a
ticket reaches one of them, `transition()` refuses to move it anywhere else, regardless of what
`expect` claims - a caller cannot "resurrect" a superseded or expired ticket by passing a permissive
`expect`. `transition()` also always refuses `to == proposed`; only `create()` may produce a
`proposed` ticket.

**Never trust `model_construct`.** A `RiskDecision`/`OrderLeg` built via `model_construct` (risk.py's
own documented fallback for a *rejected* leg that fails `OrderLeg`'s validators) can slip past a
plain `Ticket(...)` constructor call unvalidated, because pydantic's default `revalidate_instances`
setting trusts an object that is already an instance of the target class. `create()` closes this by
round-tripping the ticket through `Ticket.model_validate_json` (a *fresh* parse, which always fully
validates) before ever writing it, and refusing with `ValueError` if that fails - a ticket that
can't be read back is never written in the first place. `active()`/`expire_stale()` are still
defensive against an already-corrupt legacy row (from before this guard existed, or written some
other way): each skips an unparseable row (logging a warning) rather than raising and bricking every
other ticket in the sweep; `create()`'s supersede step does the same for an old row it can't parse,
marking its status column `superseded` without touching a body it can't safely rewrite.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from finagent.store.appdb import AppDB
from finagent.trading.models import OrderLeg, RiskDecision, Ticket, TicketStatus

logger = logging.getLogger(__name__)

# RFC 4648 base32 alphabet (A-Z, 2-7) - no padding, no ambiguous characters (0/1/8 excluded).
_BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_TICKET_ID_RANDOM_CHARS = 10

# The only extra fields `transition()` may write into the ticket body alongside the new status.
_ALLOWED_TRANSITION_FIELDS = frozenset({"broker_order_ids"})

# Absorbing states: `transition()` refuses to move a ticket out of any of these, no matter what
# `expect` says.
_TERMINAL_STATUSES = frozenset({
    TicketStatus.expired,
    TicketStatus.superseded,
    TicketStatus.rejected_by_user,
    TicketStatus.failed,
})


class TicketStateError(Exception):
    """Raised by `TicketStore.transition` when `ticket_id` is unknown, its current status is not
    one of `expect`, its current status is a terminal one, or its body cannot be parsed - including
    when another caller won a concurrent race first. The ticket row is left completely unchanged."""


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
    to: `legs` (the executable legs), each decision's index/verdict/leg/reason codes+severities
    (never the reason *message* text) and its numeric estimates, and the ticket id itself (so two
    tickets built from identical legs/decisions never collide). Every Decimal becomes a JSON
    string via pydantic's own `mode="json"` dump (exact - never a float), matching the rest of the
    project's "money is Decimal" constraint.

    Because the hash is over the *exact string form* of each Decimal, two value-equal Decimals with
    different textual representations (`Decimal("1.5")` vs `Decimal("1.50")`) hash differently.
    This is intentional (it's the same "exact, not numeric" rule the rest of the ticket hashing
    relies on) but means a hash is never something to recompute-and-compare from freshly
    re-evaluated legs/decisions and expect to match an earlier one bit-for-bit unless every
    Decimal was formatted identically both times. Callers verifying a ticket has not changed must
    always compare against the `hash` stored on the ticket itself (e.g.
    `canonical_hash(ticket.legs, ticket.decisions, ticket.id) == ticket.hash`), never against a hash
    of some other, separately re-evaluated set of legs/decisions.
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


def _executable_legs(decisions: list[RiskDecision]) -> list[OrderLeg]:
    return [decision.leg for decision in decisions if decision.verdict != "reject"]


class TicketStore:
    """SQLite-backed store for `Ticket`s: proposal, session-scoped supersede, atomic lifecycle
    transitions, and TTL-based expiry. See the module docstring for the concurrency design."""

    def __init__(self, db: AppDB, clock: Callable[[], datetime] = utcnow):
        self._db = db
        self._clock = clock
        self._lock = threading.RLock()  # shared by create()/transition()/expire_stale()

    def create(
        self,
        session_id: str,
        decisions: list[RiskDecision],
        legs: list[OrderLeg] | None = None,
        *,
        thesis: str,
        risks: str,
        invalidation: str,
        evidence_ids: list[str],
        mandate_version: int,
        quote_as_of: datetime,
        ttl_sec: int = 300,
    ) -> Ticket:
        """Build and store a new `proposed` ticket, superseding whatever ticket was previously
        active for `session_id`.

        `legs` is derived from `decisions` - `[d.leg for d in decisions if d.verdict != "reject"]`,
        in order - and is never taken from the caller as an independent value: if `legs` is given,
        it must equal that derived list exactly, or `ValueError` is raised (nothing is written).
        Also raises `ValueError` when every decision was rejected (there would be nothing to
        execute), or when `decisions[i].index != i` for any `i` (decisions must be in the exact
        order their own `index` field claims)."""
        for i, decision in enumerate(decisions):
            if decision.index != i:
                raise ValueError(
                    f"decisions[{i}].index must be {i} (decisions must be in index order), "
                    f"got {decision.index}"
                )
        executable_legs = _executable_legs(decisions)
        if legs is not None and legs != executable_legs:
            raise ValueError(
                "legs must equal the executable legs derived from decisions "
                "([d.leg for d in decisions if d.verdict != 'reject'], in order); "
                "do not pass legs independently of decisions"
            )
        if not executable_legs:
            raise ValueError("no executable legs: every decision was rejected")

        ticket_id = _new_ticket_id()
        client_order_ids = [f"fa-{ticket_id}-{i}" for i in range(len(executable_legs))]
        digest = canonical_hash(executable_legs, decisions, ticket_id)
        created_at = self._clock()
        expires_at = created_at + timedelta(seconds=ttl_sec)
        ticket = Ticket(
            id=ticket_id,
            session_id=session_id,
            status=TicketStatus.proposed,
            legs=executable_legs,
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
        # A decision/leg built via `model_construct` (risk.py's fallback for a rejected leg that
        # fails OrderLeg's own validators) can pass straight through the `Ticket(...)` call above
        # unvalidated - pydantic trusts an already-constructed instance of the target class by
        # default. A fresh parse of the JSON we're about to write always fully validates, so this
        # is the last chance to refuse a ticket that could never be read back once stored.
        try:
            Ticket.model_validate_json(body)
        except Exception as exc:  # noqa: BLE001 - any parse/validation failure must block the write
            raise ValueError(f"ticket failed round-trip validation: {exc}") from exc

        with self._lock:
            for row in self._db.query(
                "SELECT id, body FROM tickets WHERE session_id=? AND status=?",
                (session_id, TicketStatus.proposed.value),
            ):
                try:
                    superseded = Ticket.model_validate_json(row["body"]).model_copy(
                        update={"status": TicketStatus.superseded}
                    )
                except Exception:  # noqa: BLE001 - a corrupt legacy row must not abort create()
                    logger.warning(
                        "ticket %s has an unparseable body; marking it superseded without "
                        "rewriting its body",
                        row["id"],
                    )
                    self._db.execute(
                        "UPDATE tickets SET status=? WHERE id=? AND status=?",
                        (TicketStatus.superseded.value, row["id"], TicketStatus.proposed.value),
                    )
                    continue
                self._db.execute(
                    "UPDATE tickets SET status=?, body=? WHERE id=? AND status=?",
                    (
                        TicketStatus.superseded.value,
                        superseded.model_dump_json(),
                        row["id"],
                        TicketStatus.proposed.value,
                    ),
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
        in storage until `expire_stale()` flips it. A row whose body cannot be parsed is skipped
        (logged), not raised - one corrupt legacy ticket must not hide every other ticket in the
        session."""
        rows = self._db.query(
            "SELECT body FROM tickets WHERE session_id=? AND status=?",
            (session_id, TicketStatus.proposed.value),
        )
        now = self._clock()
        candidates: list[Ticket] = []
        for row in rows:
            try:
                candidates.append(Ticket.model_validate_json(row["body"]))
            except Exception:  # noqa: BLE001 - see docstring
                logger.warning(
                    "skipping an unparseable proposed ticket for session %s", session_id
                )
        candidates.sort(key=lambda t: t.created_at, reverse=True)
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
        """Atomic compare-and-set: move `ticket_id` to status `to` only if its current status is in
        `expect` (and is not a terminal state - see the module docstring - and `to` is not
        `proposed`, which only `create()` may produce). Raises `TicketStateError` (leaving the row
        untouched) for an unknown id, a status not in `expect`, a terminal current status, or an
        unparseable body; raises `ValueError` (also leaving the row untouched, and without even
        reading the database) for an unknown field in `fields` or `to == proposed`. `fields` may
        only set `broker_order_ids`.

        The whole check-then-write sequence runs under `self._lock` (shared with `create()` and
        `expire_stale()`), and the new `Ticket` is fully validated *before* the single `UPDATE`
        that commits it - so a validation failure (e.g. a `broker_order_ids` of the wrong length)
        never partially writes anything, and two racing callers can never interleave their reads
        and writes of the same ticket."""
        bad_fields = set(fields) - _ALLOWED_TRANSITION_FIELDS
        if bad_fields:
            raise ValueError(
                f"transition() does not accept field(s): {', '.join(sorted(bad_fields))}"
            )
        if to == TicketStatus.proposed:
            raise ValueError("transition() cannot move a ticket to 'proposed'; only create() does.")

        with self._lock:
            rows = self._db.query("SELECT body FROM tickets WHERE id=?", (ticket_id,))
            if not rows:
                raise TicketStateError(f"ticket {ticket_id!r} does not exist")
            try:
                current = Ticket.model_validate_json(rows[0]["body"])
            except Exception as exc:  # noqa: BLE001 - see docstring
                raise TicketStateError(
                    f"ticket {ticket_id!r} has an unparseable body and cannot be transitioned"
                ) from exc
            if current.status in _TERMINAL_STATUSES:
                raise TicketStateError(
                    f"ticket {ticket_id!r} is in terminal state {current.status.value!r}; no "
                    "further transition is allowed"
                )
            if current.status not in expect:
                raise TicketStateError(
                    f"ticket {ticket_id!r} is {current.status.value!r}, not one of the expected "
                    f"states {sorted(status.value for status in expect)}"
                )

            merged = json.loads(rows[0]["body"])
            merged["status"] = to.value
            merged.update(fields)
            new_ticket = Ticket.model_validate(merged)  # raises before any write, per the docstring

            expect_values = [status.value for status in expect]
            placeholders = ",".join("?" for _ in expect_values)
            cur = self._db.execute(
                f"UPDATE tickets SET status=?, body=? WHERE id=? AND status IN ({placeholders})",
                (to.value, new_ticket.model_dump_json(), ticket_id, *expect_values),
            )
            if cur.rowcount != 1:
                raise TicketStateError(
                    f"ticket {ticket_id!r} changed state before this transition could commit"
                )
            return new_ticket

    def expire_stale(self) -> int:
        """Flip every `proposed` ticket whose `expires_at` is at or before `clock()` to `expired`.
        Returns how many were flipped. Runs under the same lock as `create()`/`transition()`. A row
        whose body cannot be parsed is skipped (logged), not raised, and not counted. Each row's own
        `UPDATE ... WHERE id=? AND status=?` is a compare-and-set guarding against a concurrent
        `transition()` having already moved the same ticket out of `proposed` (in which case that
        row is simply skipped, not counted) - redundant with the shared lock, but cheap defense in
        depth."""
        now = self._clock()
        with self._lock:
            rows = self._db.query(
                "SELECT id, body FROM tickets WHERE status=?", (TicketStatus.proposed.value,)
            )
            count = 0
            for row in rows:
                try:
                    ticket = Ticket.model_validate_json(row["body"])
                except Exception:  # noqa: BLE001 - see docstring
                    logger.warning(
                        "skipping ticket %s during expire_stale(): unparseable body", row["id"]
                    )
                    continue
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
