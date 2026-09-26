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
     when `<name>` is *identifier-shaped* (contains `_` or `-`, or is entirely lower-case, or
     entirely upper-case - so `API_KEY`, `api-key`, `access_token`, `X-Api-Key` and
     `APCA-API-SECRET-KEY` all qualify, but an ordinary Titlecase word starting a sentence, like
     `"Secret: Apple's new product..."`, does not) *and* contains one of `secret`, `token`,
     `password`, `credential`, `access_token`, or `api[-_]?key` (`signature` too) - narrower than
     (1)'s dict-key pattern for the same false-positive reason as (2), and narrower still than an
     earlier version of this rule that flagged any Titlecase word merely because it happened to
     read like an ordinary label. `<name>` is the *full* run of name characters (alphanumerics,
     `_`, `.`, `-`) before the separator, however long, and either the whole run or its last
     dotted component may qualify (`_is_secret_like_free_text_name`). So a long dotted path like
     `...AlpacaClient._headers.APCA_API_SECRET_KEY=...` is caught too. A quoted name or value
     (`{'APCA-API-SECRET-KEY': '...'}`, `{"api_key": "..."}`, `API_KEY="..."`, including
     backslash-escaped JSON quotes) has just the quoted body redacted, and the quotes stay.

   This scan (`_pair_value_spans`) is driven by separators and walks backward over the
   name, with no name-matching regex and no cap on the name's length. Both matter. Round 2's single
   regex nested a name-charset star, then the keyword alternation, then another star. That shape
   is catastrophically slow (quadratic-or-worse backtracking) on an adversarial string like `"a."`
   repeated thousands of times, or a long dotted traceback path with no match at the end, which
   is exactly what a broker error message or exception traceback can produce. Round 3's
   replacement capped the name at 64 characters, which made it linear but left a blind spot. A
   name run longer than 64 characters before the separator was never matched at all, so its value
   was stored verbatim. The current scan has neither problem - see its docstring for the linear-time
   argument. All free-text passes (query param, Authorization, Bearer, name/value) compute their
   spans against the same original string, and the union is replaced once, so no pass can destroy
   context another pass needs. Every string leaf is also capped at 100,000 characters
   (`_MAX_LEAF_LEN`) before any regex runs against it at all, as defense in depth independent of
   any single pattern's own complexity.

4. **Known secret values** (`Ledger(db, known_secrets=...)`, added in Task 2.4). Levels 1-3 key off
   *names*, so a secret embedded in a shape none of them recognizes - Basic auth inside JSON text,
   a camelCase `apiKey=`, URL userinfo (`user:pass@host`), an odd operator (`api_key => v`),
   fullwidth separators, a value glued to other text - would otherwise be stored verbatim. When the
   ledger is given the actual secret values (the Alpaca key id and secret, say), every verbatim
   occurrence of each one is replaced with `[REDACTED]` in every string leaf *and* every dict key,
   as a pre-pass that runs before levels 1-3 ever see the payload. Each value is stripped of
   surrounding whitespace first, and anything shorter than `_MIN_KNOWN_SECRET_LEN` characters (or
   blank, or not a string - an unset setting is `None`) is ignored, so a short word can never be
   blanked out everywhere. Values are applied longest first, so a secret that contains another is
   replaced as one unit. This is `str.replace` per secret per leaf: linear in the payload size
   times the (small) number of secrets. As a final fail-safe, if a known secret still appears in
   the serialized JSON (which could only happen through JSON escaping itself), the whole payload is
   replaced with the same placeholder used for a serialization failure. `redact_text()` exposes the
   same known-secret scrub plus the free-text passes for a single string, so a caller can clean a
   broker message before showing it to a user.

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
from collections.abc import Iterable, Iterator, Mapping
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
    # `_HEADER_NAME_RE` itself has no unbounded-quantifier-then-alternation shape (see the
    # free-text patterns section below for why that combination is dangerous), so it's already
    # linear regardless of input length - `_truncate_leaf` here is defense in depth, consistent
    # with "no regex in this module runs against an untruncated string leaf."
    return bool(_HEADER_NAME_RE.search(_truncate_leaf(text)))


# --- free-text patterns (see module docstring, level 3) ------------------------------------------
#
# No pattern below has the "unbounded quantifier followed by a late alternation" shape that made
# round 2's free-text pattern catastrophically slow (the engine backtracked a leading name-charset
# star one character at a time, retrying the keyword alternation at every split point, for every
# starting position - quadratic-or-worse on a long dotted run with no match at the end). The
# general `<name><sep><value>` scan (`_pair_value_spans`) uses no name-matching regex at all: it
# finds separators, then walks *backward* from each over the name - see that function for the
# linear-time argument. Every regex it does use is a single character class, or a repetition whose
# next character is always unambiguous, and can never backtrack.
#
# Every pass reports the spans it wants hidden in the *original* text, and `_redact_free_text`
# replaces the union once. No pass ever runs on another pass's output. Earlier versions applied
# the passes in sequence, so one pass's replacement could destroy context the next one needed. For
# example, the query-param pass turned `?api_key='v'` into `?api_key=[REDACTED]'v'`, so the pair
# scan never saw the quoted value. And `&secret=APCA_API_SECRET_KEY: "v"` had its nested name and
# separator swallowed.

_QUERY_PARAM_RE = re.compile(
    r"([?&])(key|apikey|api[-_]key|token|access_token|secret|password|signature|sig)="
    r"([^&\s'\"]*)",
    re.I,
)
_AUTH_HEADER_RE = re.compile(r"(authorization\s*:\s*)\S.*", re.I)
_BEARER_RE = re.compile(r"\b(Bearer)\s+(\S+)", re.I)

# Building blocks of `_pair_value_spans`.
_SEPARATOR_RE = re.compile(r"[:=]")
_WHITESPACE_RE = re.compile(r"\s*")
_BACKSLASHES_RE = re.compile(r"\\*")
# An unquoted value: everything up to the next whitespace, `&`, `,`, `;` or quote.
_UNQUOTED_VALUE_RE = re.compile(r"[^\s&,;'\"]*")
# A quoted value's body, up to (not including) its closing quote, skipping backslash escapes. This
# is the standard "unrolled loop" shape: at every position exactly one branch can consume the next
# character, so a match never backtracks.
_QUOTED_BODY_RES = {
    '"': re.compile(r'[^"\\]*(?:\\.[^"\\]*)*', re.S),
    "'": re.compile(r"[^'\\]*(?:\\.[^'\\]*)*", re.S),
}
_QUOTE_CHARS = "\"'"
# Besides alphanumerics (`str.isalnum()`, the same set as `\w` minus "_"): the characters a
# free-text name run may contain - `\w`, `.` and `-`, as before.
_NAME_PUNCTUATION = "_.-"

# Keywords that make a free-text name "secret-like" (plus the identifier-shape check in
# `_is_secret_like_name_part`). A fixed alternation of short literals, run once per name - and
# names are disjoint slices of the text - so its total cost is linear in the length of the text.
_FREE_TEXT_SECRET_WORD_RE = re.compile(
    r"secret|token|password|credential|access[-_]?token|api[-_]?key|signature", re.I
)

# HAR/Postman-style header-entry field names (case-insensitive).
_NAME_FIELD_KEYS = frozenset({"name", "key", "header", "field"})
_VALUE_FIELD_KEYS = frozenset({"value", "val"})

# Defense in depth, independent of any single pattern's own complexity: no regex in this module
# ever runs against a string leaf longer than this - see the module docstring.
_MAX_LEAF_LEN = 100_000


def _truncate_leaf(text: str) -> str:
    if len(text) <= _MAX_LEAF_LEN:
        return text
    cut = len(text) - _MAX_LEAF_LEN
    return f"{text[:_MAX_LEAF_LEN]}…[truncated {cut} chars]"


def _is_secret_like_name_part(name: str) -> bool:
    """`name` is *identifier-shaped* - contains `_` or `-`, or is entirely lower-case, or entirely
    upper-case - and contains one of the secret-ish keywords. Dots, digits and caseless
    characters don't affect the lower/upper-case test, so a dotted path is identifier-shaped
    exactly when it contains `_`/`-` or has no mixed case anywhere. This is what keeps ordinary
    Titlecase prose like `"Secret: Apple's new product..."` untouched (a single capitalized word,
    no underscore/hyphen, is neither all-lower nor all-upper) while still catching `API_KEY`,
    `api-key`, `password=`, `TOKEN:` and `X-Api-Key:`."""
    identifier_shaped = "_" in name or "-" in name or name == name.lower() or name == name.upper()
    return identifier_shaped and bool(_FREE_TEXT_SECRET_WORD_RE.search(name))


def _is_secret_like_free_text_name(name: str) -> bool:
    """A free-text name run (however long) is secret-like when either the whole run or its last
    dotted component passes `_is_secret_like_name_part`. The whole run catches
    `...AlpacaClient._headers.APCA_API_SECRET_KEY` (it has underscores and "secret"); the last
    component also catches a mixed-case path ending in a lower-case field, like
    `MyApp.Config.password`, whose full run is not identifier-shaped. A Titlecase last component
    (`note.Secret`) still passes neither check, keeping the Titlecase-prose rule."""
    if _is_secret_like_name_part(name):
        return True
    last = name.rpartition(".")[2]
    return last != name and _is_secret_like_name_part(last)


def _name_before_separator(text: str, sep: int, floor: int) -> str:
    """The name run that ends at the separator `text[sep]`: walk backward over any whitespace, then
    at most one closing quote (optionally escaped by backslashes, as in `\\"api_key\\":`), then
    the whole contiguous run of name characters (alphanumerics, `_`, `.`, `-`) - however long it
    is. Never walks below `floor`."""
    i = sep
    while i > floor and text[i - 1].isspace():
        i -= 1
    if i > floor and text[i - 1] in _QUOTE_CHARS:
        i -= 1
        while i > floor and text[i - 1] == "\\":
            i -= 1
    end = i
    while i > floor and (text[i - 1].isalnum() or text[i - 1] in _NAME_PUNCTUATION):
        i -= 1
    return text[i:end]


def _secret_value_span(text: str, sep: int, covered_end: int) -> tuple[int, int] | None:
    """The `[start, end)` span of the value after the separator `text[sep]` that should be
    redacted, or `None` if there's nothing new to redact. Scanning never starts before
    `covered_end` (the end of the last span already redacted).

    - After optional whitespace, a quote opens a quoted value. The span is the body up to the
      matching unescaped closing quote. If the quote is itself escaped (`\\"...\\"`, JSON inside a
      string), the span runs to the next backslash-quote. The quotes stay visible. An unclosed
      quote redacts through the end of the text, since failing safe beats leaking the rest.
    - Otherwise the value is unquoted, running up to the next whitespace, `&`, `,`, `;` or
      quote, as before."""
    n = len(text)
    value_start = _WHITESPACE_RE.match(text, sep + 1).end()
    quote_at = _BACKSLASHES_RE.match(text, value_start).end()
    if quote_at < n and text[quote_at] in _QUOTE_CHARS:
        quote = text[quote_at]
        start = max(quote_at + 1, covered_end)
        if quote_at == value_start:
            end = _QUOTED_BODY_RES[quote].match(text, start).end()
            if end >= n or text[end] != quote:
                end = n
        else:
            end = text.find("\\" + quote, start)
            if end == -1:
                end = n
    else:
        start = max(value_start, covered_end)
        end = _UNQUOTED_VALUE_RE.match(text, start).end()
    return (start, end) if end > start else None


def _pair_value_spans(text: str) -> list[tuple[int, int]]:
    """The spans of every `<name><sep><value>` value in `text` (`sep` is `:` or `=`, optionally
    spaced) whose name is secret-like (`_is_secret_like_free_text_name`). Names and separators are
    never part of a span, so they stay visible. It is driven by separators, not by a name-matching
    regex. Each separator is found with a single-character class. The name is then the *full* run
    walked backward from it, so a name has no length cap and no blind spot: 64, 65 or 5,000
    characters all behave the same.

    Linear in `len(text)` by construction:
    - Each backward walk stops at `floor`, just past the previous separator, at the latest. The
      walked ranges are disjoint, so the walks and the name checks on them total O(N).
    - The whitespace and backslash skips after a separator cover the run starting right after
      it. A separator ends any such run, so these runs are disjoint too.
    - Every value scan starts at or after `covered_end`. When it finds something, it moves
      `covered_end` to the end of what it scanned. So no character is value-scanned twice,
      including when a value itself contains more separators (`API_KEY:API_KEY:...`).
    Each value is decided on its own separator, so a secret pair nested in another value is still
    caught. Examples are `config=APCA_API_SECRET_KEY=shh` and the YAML-ish
    `credentials:\\n  api_key: shh`."""
    spans: list[tuple[int, int]] = []
    covered_end = 0
    floor = 0
    for match in _SEPARATOR_RE.finditer(text):
        sep = match.start()
        name = _name_before_separator(text, sep, floor)
        floor = sep + 1
        if name and _is_secret_like_free_text_name(name):
            span = _secret_value_span(text, sep, covered_end)
            if span is not None:
                spans.append(span)
                covered_end = span[1]
    return spans


def _free_text_secret_spans(text: str) -> list[tuple[int, int]]:
    """Every span any free-text pass wants hidden, all computed against the same `text` - see the
    free-text patterns section for why no pass runs on another pass's output."""
    spans = [m.span(3) for m in _QUERY_PARAM_RE.finditer(text)]
    spans += [(m.end(1), m.end()) for m in _AUTH_HEADER_RE.finditer(text)]
    spans += [m.span(2) for m in _BEARER_RE.finditer(text)]
    spans += _pair_value_spans(text)
    return spans


def _redact_free_text(text: str) -> str:
    """Truncate `text` to `_MAX_LEAF_LEN`, then replace each maximal run of overlapping or
    adjacent secret spans with a single `[REDACTED]`."""
    text = _truncate_leaf(text)
    merged: list[list[int]] = []
    for start, end in sorted(_free_text_secret_spans(text)):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        return text
    pieces: list[str] = []
    pos = 0
    for start, end in merged:
        pieces.append(text[pos:start])
        pieces.append(_REDACTED)
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces)


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


# --- known secret values (see module docstring, level 4) -----------------------------------------

# A configured secret shorter than this is ignored: blanking a short common word out of every
# ledger string would destroy the audit trail while protecting nothing real (Alpaca key ids and
# secrets are far longer).
_MIN_KNOWN_SECRET_LEN = 8

_PLACEHOLDER = {"_ledger_error": "payload could not be serialized"}


def _normalize_known_secrets(values: Iterable[Any]) -> tuple[str, ...]:
    """Strip, drop non-strings/blank/short values, deduplicate, and order longest first."""
    kept = {
        value.strip()
        for value in values
        if isinstance(value, str) and len(value.strip()) >= _MIN_KNOWN_SECRET_LEN
    }
    return tuple(sorted(kept, key=lambda s: (-len(s), s)))


def _scrub_str(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, _REDACTED)
    return text


def _scrub_known(value: Any, secrets: tuple[str, ...]) -> Any:
    """Replace every verbatim occurrence of a known secret in every string leaf and dict key. Runs
    before `_redact`, and produces only shapes `_redact` already handles (dicts with string keys,
    lists, and the original leaves)."""
    if isinstance(value, BaseModel):
        return _scrub_known(value.model_dump(mode="json"), secrets)
    if isinstance(value, str):
        return _scrub_str(value, secrets)
    if isinstance(value, Mapping):
        return {
            _scrub_str(_stringify_key(key), secrets): _scrub_known(sub_value, secrets)
            for key, sub_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_scrub_known(item, secrets) for item in value]
    return value


class Ledger:
    """Thin, append-only wrapper around `AppDB`'s `ledger` table. `known_secrets` are exact secret
    values (e.g. the Alpaca key id and secret) scrubbed verbatim from every payload - see the module
    docstring, level 4."""

    def __init__(self, db: AppDB, known_secrets: Iterable[Any] = ()):
        self._db = db
        self._known_secrets = _normalize_known_secrets(known_secrets)

    def redact_text(self, text: str) -> str:
        """The same redaction a ledger string leaf gets (known secret values, then the free-text
        passes), for a single string a caller wants to show elsewhere - e.g. a broker error
        message in an API response."""
        return _redact_free_text(_scrub_str(text, self._known_secrets))

    def _serialize(self, payload: Any) -> str:
        scrubbed = _scrub_known(payload, self._known_secrets) if self._known_secrets else payload
        serialized = json.dumps(_redact(scrubbed), sort_keys=True)
        if any(secret in serialized for secret in self._known_secrets):
            # Only JSON escaping itself could reintroduce a scrubbed value; fail safe.
            logger.error("ledger payload still contained a known secret after scrubbing")
            return json.dumps(_PLACEHOLDER)
        return serialized

    def append(self, session_id: str, kind: str, payload: dict) -> int:
        """Redact and store one ledger entry; returns its row id. Never raises for a serialization
        problem - see the module docstring."""
        try:
            serialized = self._serialize(payload)
        except Exception:  # noqa: BLE001 - fail safe: never crash a caller that already acted
            logger.exception(
                "ledger payload for session %s kind %s could not be serialized; storing a "
                "placeholder instead",
                session_id,
                kind,
            )
            serialized = json.dumps(_PLACEHOLDER)
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
