"""Append-only audit ledger: every trading-relevant event (proposal, confirmation, submission,
reconciliation, rejection, ...) gets one row here, for later reconstruction of what happened and
why. Per the project's global constraint that secrets never reach the ledger, `append()` redacts
every payload *before* it is ever written to disk - not just at read/export time - so a stray
`APCA-API-SECRET-KEY` (or a mis-plumbed mandate field with "key"/"token" in its name) can never
even transiently land in `AppDB`'s `ledger` table.

Redaction is deliberately broad rather than narrowly matching only the header names this project
happens to use today: any dict key matching `/secret|key|token|password|authorization/i`, at any
nesting depth (inside nested dicts and lists), has its value replaced outright - never partially
masked, since a partial mask can leak enough of a secret to be useful. As defense in depth against
a secret pasted into a *value* rather than carried by a well-named key (e.g. a raw header line
logged into a free-text message), any string value containing the literal Alpaca header names
`APCA-API-SECRET-KEY` or `APCA-API-KEY-ID` is redacted too, whole-string, regardless of its key.

Decimals never reach `json.dumps` directly (which cannot serialize them) - the same recursive walk
that redacts also turns every `Decimal` into its exact `str()`, matching the project's "money is
Decimal, never float" rule end to end: nothing here ever round-trips through `float`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from finagent.store.appdb import AppDB

_REDACTED = "[REDACTED]"

# Case-insensitive: matches a dict key anywhere in a payload whose value must never be logged.
_SECRET_KEY_RE = re.compile(r"secret|key|token|password|authorization", re.I)

# A string *value* that looks like a raw Alpaca auth header line, wherever it appears (e.g. pasted
# into a free-text log message under an innocuous key like "message").
_SECRET_HEADER_VALUE_RE = re.compile(r"APCA-API-SECRET-KEY|APCA-API-KEY-ID", re.I)


def _redact(value: Any) -> Any:
    """Recursively redact `value` for ledger storage: dict keys matching `_SECRET_KEY_RE` have
    their value replaced wholesale (the redaction doesn't recurse into an already-redacted value);
    every other dict/list is walked; a `Decimal` becomes its exact string; anything else is
    returned unchanged, except a string that itself looks like a raw Alpaca auth header."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, sub_value in value.items():
            if isinstance(key, str) and _SECRET_KEY_RE.search(key):
                redacted[key] = _REDACTED
            else:
                redacted[key] = _redact(sub_value)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, str) and _SECRET_HEADER_VALUE_RE.search(value):
        return _REDACTED
    return value


class Ledger:
    """Thin, append-only wrapper around `AppDB`'s `ledger` table."""

    def __init__(self, db: AppDB):
        self._db = db

    def append(self, session_id: str, kind: str, payload: dict) -> int:
        """Redact and store one ledger entry; returns its row id."""
        redacted = _redact(payload)
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._db.execute(
            "INSERT INTO ledger (session_id, kind, payload, created_at) VALUES (?,?,?,?)",
            (session_id, kind, json.dumps(redacted, sort_keys=True), created_at),
        )
        return cur.lastrowid

    def export_jsonl(self, session_id: str) -> Iterator[str]:
        """Yield one JSON-encoded line per ledger entry for `session_id`, oldest first (insertion
        order, i.e. row id order - `ledger.id` is an autoincrementing primary key)."""
        rows = self._db.query(
            "SELECT id, session_id, kind, payload, created_at FROM ledger "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        for row in rows:
            yield json.dumps(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "kind": row["kind"],
                    "created_at": row["created_at"],
                    "payload": json.loads(row["payload"]),
                }
            )
