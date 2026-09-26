"""Append-only audit ledger: every trading-relevant event (proposal, confirmation, submission,
reconciliation, rejection, ...) gets one row here, for later reconstruction of what happened and
why. Per the project's global constraint that secrets never reach the ledger, `append()` redacts
every payload *before* it is ever written to disk - not just at read/export time - so a stray
`APCA-API-SECRET-KEY` (or a mis-plumbed mandate field with "key"/"token" in its name) can never
even transiently land in `AppDB`'s `ledger` table.

Redaction is deliberately broad rather than narrowly matching only the header names this project
happens to use today, and works at several levels:

1. **Dict/mapping keys.** Any key matching `/secret|key|token|password|authorization|credential|
   cookie/i`, at any nesting depth (inside nested dicts, lists, tuples and sets), has its value
   replaced outright - never partially masked, since a partial mask can leak enough of a secret to
   be useful. This is deliberately the broadest of the three levels (it also catches, say, a dict
   *keyed by* a stock ticker literally spelled `"KEY"`, over-redacting an unrelated value - see
   `test_append_documented_over_redaction_of_a_dict_keyed_by_the_ticker_key` - accepted for now;
   Task 2.4 keeps trading symbols out of dict keys). A non-`str` key (a `Decimal`, `bytes`, an
   `int`, ...) is stringified (bytes via `"<bytes len=N>"`, never `repr()`, which would leak the
   raw content back into the key text) before this check and before being used as the output key,
   both so the check applies uniformly and so a payload with mixed key types never trips
   `json.dumps(..., sort_keys=True)`'s inability to compare, say, an `int` key against a `str` key.
2. **Header-shaped values**, independent of dict keys, using a *narrower* "looks like a credential
   header name" pattern (`_looks_like_header_name`, see the constant's own docstring) chosen to
   avoid flagging an ordinary word or ticker that merely contains "key" (a bare `"KEY"` - a
   plausible ticker symbol - does not match it, but `"APCA-API-KEY-ID"`, `"api_key"` and
   `"X-Auth-Token"` do):
   - A HAR/Postman-style header entry - a mapping with a `name`/`key`/`header`/`field` field (any
     case) whose *value* looks like a header name, plus a sibling `value`/`val` field - keeps the
     name field as-is (so the ledger still shows *which* header this was) and redacts the
     `value`/`val` field wholesale.
   - Within any list/tuple, a run of `[name, value]` (or `[name, value, ...]`, or a longer flat
     sequence of alternating names/values, e.g. `["APCA-API-KEY-ID", "id", "APCA-API-SECRET-KEY",
     "secret"]`) redacts exactly the item immediately after a matching name, keeping the name and
     any unrelated trailing items untouched.
3. **Free-text string scanning**, independent of both of the above, for a secret pasted into an
   otherwise ordinary string rather than carried by a key/pair:
   - A query-string parameter named `key`, `apikey`, `api_key`/`api-key`, `token`, `access_token`,
     `secret`, `password`, `signature` or `sig` has its value redacted, keeping the parameter name
     (`?apikey=[REDACTED]`).
   - `"Authorization: <scheme> <credential>"` and a bare `"Bearer <token>"` both redact just the
     credential/token, keeping `"Authorization:"`/`"Bearer"` visible.
   - A more general `"<name>=<value>"` / `"<name>: <value>"` anywhere in text redacts the value
     when `<name>` contains one of `secret`, `token`, `password`, `credential`, `access_token`, or
     `api[-_]?key` (signature too) - narrower than (1)'s dict-key pattern for the same
     false-positive reason as (2).

A `pydantic.BaseModel` is coerced via `model_dump(mode="json")` before redaction (so a payload can
carry, say, an `OrderLeg` directly, and any of its fields still gets the same key-based scan, and a
`pydantic.SecretStr` field - already masked by pydantic itself - gets an extra layer via its field
name); `bytes` becomes `"<bytes len=N>"` (never logged raw, whether as a key or a value); a
`Decimal` becomes its exact `str()`, matching the project's "money is Decimal, never float" rule
end to end - nothing here ever round-trips through `float`.

`append()` never raises for a serialization problem: whatever the payload's shape (an unbounded
recursive structure, a cyclic reference, a type this module has no specific handling for - a raw
`dataclass` or exception instance, say), redaction and JSON encoding are wrapped so a caller that
already placed real broker orders can't be crashed by a logging call. If serialization fails for any
reason, the original payload is discarded entirely and a placeholder that carries no part of it is
stored instead - "fail safe, never leak" rather than "fail loud."
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
# never be logged. Deliberately the broadest of this module's patterns - see the module docstring.
_SECRET_KEY_RE = re.compile(r"secret|key|token|password|authorization|credential|cookie", re.I)

# A narrower "this string names a credential-bearing header/parameter" pattern, used only to decide
# whether a *value* (never a dict key - that's `_SECRET_KEY_RE`, deliberately broader) found in a
# [name, value] pair or a HAR/Postman-style {"name": ..., "value": ...} entry should trigger
# redacting its paired value. Chosen so a bare "KEY"/"KEYS" (a plausible ticker symbol) or an
# unrelated word that happens to contain "key" (e.g. "monkey") does NOT match, while real header
# names do: contains "api", "secret", "token", "password", "credential", "cookie" or
# "authorization"; or starts with "apca-"/"x-" (the project's own header prefix, and the common
# custom-header convention); or matches "key[-_]id" or "api[-_]?key" (covers "api_key",
# "apikey", "APCA-API-KEY-ID").
_HEADER_NAME_RE = re.compile(
    r"api|secret|token|password|credential|cookie|authorization"
    r"|^apca-|^x-|key[-_]id|api[-_]?key",
    re.I,
)


def _looks_like_header_name(text: str) -> bool:
    return bool(_HEADER_NAME_RE.search(text))


# --- free-text patterns (see module docstring, level 3) ------------------------------------------

_QUERY_PARAM_RE = re.compile(
    r"([?&])(key|apikey|api[-_]key|token|access_token|secret|password|signature|sig)="
    r"[^&\s'\"]*",
    re.I,
)
_AUTH_HEADER_RE = re.compile(r"(authorization\s*:\s*)\S.*", re.I)
_BEARER_RE = re.compile(r"\b(Bearer)\s+\S+", re.I)
_FREE_TEXT_KV_RE = re.compile(
    r"\b([\w.\-]*(?:secret|token|password|credential|access[-_]?token|api[-_]?key|signature)"
    r"[\w.\-]*)(\s*[:=]\s*)([^\s&,;'\"]+)",
    re.I,
)

# HAR/Postman-style header-entry field names (case-insensitive).
_NAME_FIELD_KEYS = frozenset({"name", "key", "header", "field"})
_VALUE_FIELD_KEYS = frozenset({"value", "val"})


def _redact_free_text(text: str) -> str:
    text = _QUERY_PARAM_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}={_REDACTED}", text)
    text = _AUTH_HEADER_RE.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)} {_REDACTED}", text)
    text = _FREE_TEXT_KV_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", text)
    return text


def _stringify_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return f"<bytes len={len(key)}>"
    return str(key)


def _har_style_fields(mapping: Mapping) -> tuple[str, str] | None:
    """If `mapping` looks like a HAR/Postman header entry - a `name`/`key`/`header`/`field` field
    (any case) whose value looks like a header name (`_looks_like_header_name`), plus a sibling
    `value`/`val` field - return `(name_field_key, value_field_key)` so the caller can keep the
    name field as-is (it's a label, not a secret - true whether it's spelled `name` or `key`) and
    redact only the value field. `None` when this mapping doesn't have that shape."""
    name_field: str | None = None
    value_field: str | None = None
    for key, sub_value in mapping.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if lowered in _NAME_FIELD_KEYS and isinstance(sub_value, str) and _looks_like_header_name(sub_value):
            name_field = key
        if lowered in _VALUE_FIELD_KEYS:
            value_field = key
    if name_field is not None and value_field is not None:
        return name_field, value_field
    return None


def _redact_sequence(items: list) -> list:
    """Redact a list/tuple: a run of `name, value` where `name` is a string matching
    `_looks_like_header_name` redacts exactly the item immediately following it (whatever type it
    is), keeping the name and any other item untouched - handles a plain 2-item pair, a longer flat
    header list, and a 3+-tuple where only one element follows the name."""
    result = []
    skip_next = False
    for item in items:
        if skip_next:
            result.append(_REDACTED)
            skip_next = False
            continue
        if isinstance(item, str) and _looks_like_header_name(item):
            # Still run it through the free-text scan: a "name" item can itself be a compound
            # string that embeds its own value (e.g. "APCA-API-SECRET-KEY: shh" as a single list
            # element rather than split across two elements), which the pair-name check alone
            # would otherwise leave untouched.
            result.append(_redact_free_text(item))
            skip_next = True
        else:
            result.append(_redact(item))
    return result


def _redact(value: Any) -> Any:
    """Recursively redact `value` for ledger storage. See the module docstring for the exact
    rules; this never raises for a type it doesn't specifically handle (falls through to `value`
    unchanged) - `append()`'s own broad try/except is the last-resort safety net for anything
    `json.dumps` still can't encode afterwards (or that causes this function itself to fail, e.g.
    a cyclic structure recursing past Python's stack limit)."""
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        har_fields = _har_style_fields(value)
        redacted: dict[str, Any] = {}
        for key, sub_value in value.items():
            str_key = _stringify_key(key)
            if har_fields is not None and str_key == har_fields[0]:
                # The header's *name* field (e.g. HAR's "name" or Postman's "key") - a label, not
                # a secret, kept visible even though its own key (like literal "key") may also
                # match the broader `_SECRET_KEY_RE` dict-key pattern below.
                redacted[str_key] = _redact_free_text(sub_value) if isinstance(sub_value, str) else sub_value
            elif har_fields is not None and str_key == har_fields[1]:
                redacted[str_key] = _REDACTED
            elif _SECRET_KEY_RE.search(str_key):
                redacted[str_key] = _REDACTED
            else:
                redacted[str_key] = _redact(sub_value)
        return redacted
    if isinstance(value, (list, tuple)):
        return _redact_sequence(list(value))
    if isinstance(value, (set, frozenset)):
        return [_redact(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str):
        return _redact_free_text(value)
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
