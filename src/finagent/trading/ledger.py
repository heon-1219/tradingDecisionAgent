"""Append-only audit ledger: every trading-relevant event (proposal, confirmation, submission,
reconciliation, rejection, ...) gets one row here, for later reconstruction of what happened and
why. Per the project's global constraint that secrets never reach the ledger, `append()` redacts
every payload *before* it is ever written to disk - not just at read/export time - so a stray
`APCA-API-SECRET-KEY` (or a mis-plumbed mandate field with "key"/"token" in its name) can never
even transiently land in `AppDB`'s `ledger` table.

Redaction is deliberately broad rather than narrowly matching only the header names this project
happens to use today:

- Any dict/mapping key matching `/secret|key|token|password|authorization|credential|cookie/i`, at
  any nesting depth (inside nested dicts, lists, tuples and sets), has its value replaced outright
  - never partially masked, since a partial mask can leak enough of a secret to be useful. A
  non-`str` key (a `Decimal`, an `int`, ...) is stringified before this check and before being used
  as the output key, both so the check applies uniformly and so a payload with mixed key types
  never trips `json.dumps(..., sort_keys=True)`'s inability to compare, say, `int` and `str` keys.
- A 2-item list/tuple whose first item is a string matching that same key pattern is treated as a
  `[name, value]` pair (e.g. an HTTP header represented as `["APCA-API-SECRET-KEY", "..."]` rather
  than a dict) - the name is kept (so the ledger still shows *which* header was redacted) and the
  value is replaced outright.
- As defense in depth against a secret pasted into a *value* rather than carried by a well-named
  key or pair (e.g. a raw header line logged into a free-text message, or an env-style
  `NAME=value` string), any string value containing one of the literal Alpaca credential header/env
  names - `APCA-API-SECRET-KEY`, `APCA-API-KEY-ID`, `APCA_API_SECRET_KEY`, `APCA_API_KEY_ID`,
  `ALPACA_API_SECRET_KEY`, `ALPACA_API_KEY_ID` (case-insensitive, dash or underscore) - is redacted
  wholesale, regardless of its key.

A `pydantic.BaseModel` is coerced via `model_dump(mode="json")` before redaction (so a payload can
carry, say, an `OrderLeg` directly, and any of its fields still gets the same key-based scan);
`bytes` becomes `"<bytes len=N>"` (never logged raw); a `Decimal` becomes its exact `str()`,
matching the project's "money is Decimal, never float" rule end to end - nothing here ever
round-trips through `float`.

`append()` never raises for a serialization problem: whatever the payload's shape, redaction and
JSON encoding are wrapped so a caller that already placed real broker orders can't be crashed by a
logging call. If serialization fails for any reason (an exotic type this module doesn't know how to
redact-and-encode), the original payload is discarded entirely and a placeholder that carries no
part of it is stored instead - "fail safe, never leak" rather than "fail loud."
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from finagent.store.appdb import AppDB

logger = logging.getLogger(__name__)

_REDACTED = "[REDACTED]"

# Case-insensitive: matches a (stringified) dict/mapping key anywhere in a payload whose value must
# never be logged.
_SECRET_KEY_RE = re.compile(r"secret|key|token|password|authorization|credential|cookie", re.I)

# A string *value* that looks like a raw Alpaca auth header or env-var line, wherever it appears
# (e.g. pasted into a free-text log message under an innocuous key like "message"). Both the
# dash-separated header form and the underscore/env form, for both Alpaca's official "APCA" prefix
# and the common "ALPACA" spelling, case-insensitive.
_SECRET_HEADER_VALUE_RE = re.compile(
    r"APCA[-_]API[-_]SECRET[-_]KEY|APCA[-_]API[-_]KEY[-_]ID|"
    r"ALPACA[-_]API[-_]SECRET[-_]KEY|ALPACA[-_]API[-_]KEY[-_]ID",
    re.I,
)


def _is_kv_pair(items: list) -> bool:
    """A 2-item ordered sequence (list/tuple, never a set - sets have no defined "first" item)
    whose first item is a string matching the secret-key pattern: treated as `[name, value]`, e.g.
    an HTTP header pair, so the value (not the name) is what gets redacted."""
    return len(items) == 2 and isinstance(items[0], str) and bool(_SECRET_KEY_RE.search(items[0]))


def _redact(value: Any) -> Any:
    """Recursively redact `value` for ledger storage. See the module docstring for the exact
    rules; this never raises for a type it doesn't specifically handle (falls through to `value`
    unchanged) - `append()`'s own broad try/except is the last-resort safety net for anything
    `json.dumps` still can't encode afterwards."""
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, sub_value in value.items():
            str_key = key if isinstance(key, str) else str(key)
            if _SECRET_KEY_RE.search(str_key):
                redacted[str_key] = _REDACTED
            else:
                redacted[str_key] = _redact(sub_value)
        return redacted
    if isinstance(value, (list, tuple)):
        items = list(value)
        if _is_kv_pair(items):
            return [items[0], _REDACTED]
        return [_redact(item) for item in items]
    if isinstance(value, (set, frozenset)):
        return [_redact(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str) and _SECRET_HEADER_VALUE_RE.search(value):
        return _REDACTED
    return value


class Ledger:
    """Thin, append-only wrapper around `AppDB`'s `ledger` table."""

    def __init__(self, db: AppDB):
        self._db = db

    def append(self, session_id: str, kind: str, payload: dict) -> int:
        """Redact and store one ledger entry; returns its row id. Never raises for a serialization
        problem - see the module docstring."""
        try:
            serialized = json.dumps(_redact(payload), sort_keys=True)
        except Exception:  # noqa: BLE001 - fail safe: never crash a caller that already acted
            logger.exception(
                "ledger payload for session %s kind %s could not be serialized; storing a "
                "placeholder instead",
                session_id,
                kind,
            )
            serialized = json.dumps({"_ledger_error": "payload could not be serialized"})
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._db.execute(
            "INSERT INTO ledger (session_id, kind, payload, created_at) VALUES (?,?,?,?)",
            (session_id, kind, serialized, created_at),
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
