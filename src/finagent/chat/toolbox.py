"""The chat agent's complete tool surface: specs + dispatch, returning `ToolOutput`.

`ChatToolbox` is the only bridge from the LLM to the research connectors, market data, the paper
account, the product knowledge base, mandate drafting and trade *proposals*. It deliberately has no
order-placing, order-cancelling or mandate-saving tool: `propose_trades` only creates a ticket
(risk-checked, hashed, with a TTL) and the user confirms it in the UI - `POST
/api/tickets/{id}/confirm` is the only path to `broker.submit_order` (see global-constraints.md).
`FORBIDDEN_TOOL_NAME_RE` is checked against every tool name at import time.

Rules every tool follows:
- `execute()` never raises: an unknown tool, bad arguments or any failure becomes a `ToolOutput`
  with `is_error=True` and a short message (never exception internals, response bodies or keys).
- Arguments are validated here (type, range, enum, pattern) before any connector is called; the
  JSON-schema in `specs()` is advisory for the model, not trusted.
- Money from the model (`qty`, `notional`, `limit_price`, mandate amounts) goes through
  `parse_money`: `str` -> `Decimal(str)`, `int` -> `Decimal(int)`, `float` -> `Decimal(repr(f))`
  (the shortest decimal the model wrote), anything else (including `bool` and `None`) is refused.
- External text (news, web pages, filing text, grounded answers, screener strings, broker error
  messages) reaches the model only sanitized and inside `<untrusted source="...">` blocks. Numbers
  from our own data (quotes, bar metrics, account, product DB) are not wrapped.
- Research tools register `SourceRecord`s in the session's `EvidenceStore` and cite them as `[E#]`;
  `propose_trades` only accepts evidence ids that exist in that session.
- With a `ledger`, every call appends a `tool_call` audit entry (hashes only), every created ticket
  a `proposal` entry and every `propose_trades` call that ends without a ticket, whatever the cause,
  a `proposal_rejected` entry (see `chat/audit.py`). An audit write that fails is logged by type and
  never breaks the call.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from ..agent.llm import ToolSpec
from ..agent.tools import VIEW_ID_NAME
from ..research.evidence import (
    ResearchResult,
    SourceRecord,
    sanitize_untrusted,
    wrap_untrusted,
)
from ..research.http import HttpError
from ..trading.broker import BrokerError
from ..trading.execution import NotPaperAccount
from ..trading.ledger import MIN_KNOWN_SECRET_LEN
from ..trading.metrics import summarize
from ..trading.models import Mandate, OrderLeg, RiskDecision, normalize_symbol
from ..trading.risk import evaluate, overall
from .audit import ProposalOutcome, ToolAudit

log = logging.getLogger("finagent.chat.toolbox")

# No tool may ever be named like an order-placing action (checked at import and in tests).
FORBIDDEN_TOOL_NAME_RE = re.compile(r"submit|place|execute|order_now|buy_now|sell_now|cancel", re.I)

TEXT_LIMIT = 4000  # knowledge / product-QA outputs are trimmed to about this many characters
_RESEARCH_MAX_CHARS = 12000  # cap when the toolbox itself has to sanitize+wrap a connector's text
MAX_QUOTE_SYMBOLS = 10
MAX_LEGS = 10
MAX_EVIDENCE_IDS = 30
_PROPOSAL_TURNS_KEPT = 2048  # (session_id, turn_id) proposal slots remembered (oldest dropped)

# Calendar days of daily bars fetched per lookback: enough trading days for the matching window
# (1y needs 253 closes for a 252-bar return).
LOOKBACK_DAYS = {"1m": 35, "3m": 100, "6m": 190, "1y": 380, "2y": 745}
# Minimum bars each `summarize()` metric needs to cover its full window; with fewer, the tool shows
# "n/a" instead of a shorter-window number presented under a longer window's name.
_METRIC_MIN_BARS = {
    "return_1m": 22,
    "return_3m": 64,
    "return_1y": 253,
    "vol_1y": 253,
    "max_dd_1y": 253,
    "sma_50": 50,
    "sma_200": 200,
}
_METRIC_ORDER = ("last", "return_1m", "return_3m", "return_1y", "vol_1y", "max_dd_1y", "sma_50", "sma_200")

# v_etf_gl.wu_inv_ast_type values, plus friendly aliases the model may use.
ETF_ASSET_TYPES = ("Equity", "Bond", "Commodity", "Alternatives", "Mixed Assets", "Money Market")
_ETF_ASSET_ALIASES = {
    "equities": "Equity", "stock": "Equity", "stocks": "Equity",
    "bonds": "Bond", "fixed income": "Bond", "fixed-income": "Bond",
    "commodities": "Commodity",
    "alternative": "Alternatives",
    "mixed": "Mixed Assets", "mixed asset": "Mixed Assets", "multi-asset": "Mixed Assets",
    "cash": "Money Market", "money-market": "Money Market",
}
# ORDER BY fragments chosen from this fixed map - never built from model text.
_ETF_SORTS = {
    "aum": "(du_last_aum IS NULL), du_last_aum DESC, ticker",
    "expense_ratio": "(expense_ratio_pct IS NULL), expense_ratio_pct ASC, (du_last_aum IS NULL), du_last_aum DESC, ticker",
}
_ETF_DEFAULT_BASE_DATE = "2026-08-22"
_LIKE_ESCAPE = "\\"
_ETF_KEYWORD_SQL = (
    f"(pd_nm LIKE ? ESCAPE '{_LIKE_ESCAPE}' OR cu_base_index LIKE ? ESCAPE '{_LIKE_ESCAPE}' "
    f"OR cu_strtegy LIKE ? ESCAPE '{_LIKE_ESCAPE}')"
)

_TICKER_OR_CIK_RE = re.compile(r"(?:CIK)?\d{1,10}|[A-Za-z0-9][A-Za-z0-9.\-]{0,14}", re.I)
_CIK_RE = re.compile(r"(?:CIK)?\d{1,10}", re.I)
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
_FORM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 /\-]{0,14}")
_CONCEPT_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{0,150}")
_DOMAIN_RE = re.compile(r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}")
_EVIDENCE_ID_RE = re.compile(r"E\d{1,7}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:\d{2})?)?")
_TOKEN_STRIP_RE = re.compile(r"[^A-Za-z0-9._:/+\-]")
# A plain ASCII decimal, optionally in exponent notation ("12.5", "-3", ".5", "1e3").
_DECIMAL_STR_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
MONEY_ABS_LIMIT = Decimal("1e15")
MONEY_MAX_DECIMALS = 12
# Provider/DB numbers outside 1e-30..1e30 are treated as missing, and formatting falls back to
# scientific notation beyond this, so an absurd exponent can never become a megabyte of digits.
_DISPLAY_MAX_ADJUSTED = 30
# Evidence from market-data tools; a buy must also cite at least one research source.
MARKET_DATA_CONNECTORS = frozenset({"alpaca_quotes", "alpaca_bars", "alpaca_screener"})
# `<untrusted source="E1,E2">` openers, whole wrapped blocks, and any leftover wrapper tag.
_UNTRUSTED_OPENER_RE = re.compile(r'<untrusted source="([^"]*)">')
_UNTRUSTED_BLOCK_RE = re.compile(r'<untrusted source="[^"]*">.*?</untrusted>', re.S)
_UNTRUSTED_TAG_RE = re.compile(r"<\s*/?\s*untrusted", re.I)
_QUOTE_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?")

_CENT = Decimal("0.01")
_HUNDRED = Decimal("100")
_CLOSING_LINE = "The user must confirm this ticket in the UI; you cannot place orders."


@dataclass
class ToolOutput:
    text: str
    sources: list[SourceRecord] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    is_error: bool = False


class _ArgError(ValueError):
    """Invalid tool arguments; the message is shown to the model."""


# --- money --------------------------------------------------------------------------------------


def parse_money(value: Any, field_name: str) -> Decimal:
    """Convert a money/amount value from the model into a finite `Decimal`.

    `str` -> `Decimal(str)`, `int` -> `Decimal(int)`, `float` -> `Decimal(repr(float))` (the
    shortest decimal the model wrote, never the binary expansion), `Decimal` kept as is. `bool`,
    `None` and every other type raise `ValueError`, and so do:
    - strings that are not a plain ASCII decimal (`Decimal()` itself would also accept "1_000",
      non-ASCII digits, "NaN" and "Infinity");
    - non-finite values, magnitudes of `MONEY_ABS_LIMIT` (1e15) or more, and values needing more
      than `MONEY_MAX_DECIMALS` decimal places (pathological exponents such as 1e-999999, which
      would blow up formatting and quantize)."""
    hint = f'{field_name} must be a decimal string such as "10" or "187.25"'
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{hint} (got {type(value).__name__})")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(repr(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text or len(text) > 40 or not _DECIMAL_STR_RE.fullmatch(text):
            raise ValueError(f"{hint} (plain ASCII digits: no $ sign, spaces, underscores or thousands separators)")
        try:
            number = Decimal(text)
        except InvalidOperation:  # exponent beyond the context's range, e.g. "1e99999999999999999999"
            raise ValueError(f"{field_name} is out of range") from None
    else:
        raise ValueError(f"{hint} (got {type(value).__name__})")
    if not number.is_finite():
        raise ValueError(f"{field_name} must be a finite number")
    if abs(number) >= MONEY_ABS_LIMIT:
        raise ValueError(f"{field_name} is too large (its magnitude must be below 1e15)")
    if number != 0 and number.normalize().as_tuple().exponent < -MONEY_MAX_DECIMALS:
        raise ValueError(f"{field_name} has too many decimal places (at most {MONEY_MAX_DECIMALS})")
    return number


# --- argument helpers ---------------------------------------------------------------------------


def _str_arg(args: dict, key: str, *, required: bool = False, max_len: int = 500) -> str | None:
    value = args.get(key)
    if value is None:
        if required:
            raise _ArgError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise _ArgError(f"'{key}' must be a string")
    value = value.strip()
    if not value:
        if required:
            raise _ArgError(f"'{key}' must not be empty")
        return None
    if len(value) > max_len:
        raise _ArgError(f"'{key}' is too long (max {max_len} characters)")
    return value


def _int_arg(args: dict, key: str, *, default: int, lo: int, hi: int) -> int:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\s*\d{1,9}\s*", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ArgError(f"'{key}' must be an integer")
    if not lo <= value <= hi:
        raise _ArgError(f"'{key}' must be between {lo} and {hi}")
    return value


def _enum_arg(args: dict, key: str, choices: tuple[str, ...], *, default: str | None = None) -> str | None:
    value = args.get(key)
    if value is None:
        return default
    lookup = {c.lower(): c for c in choices}
    if not isinstance(value, str) or value.strip().lower() not in lookup:
        raise _ArgError(f"'{key}' must be one of: {', '.join(choices)}")
    return lookup[value.strip().lower()]


def _str_list_arg(
    args: dict,
    key: str,
    *,
    required: bool = False,
    max_items: int,
    max_len: int = 100,
    pattern: re.Pattern | None = None,
) -> list[str] | None:
    value = args.get(key)
    if value is None:
        if required:
            raise _ArgError(f"'{key}' is required")
        return None
    if not isinstance(value, list):
        raise _ArgError(f"'{key}' must be an array of strings")
    if required and not value:
        raise _ArgError(f"'{key}' must not be empty")
    if len(value) > max_items:
        raise _ArgError(f"'{key}' accepts at most {max_items} items")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _ArgError(f"'{key}' must contain only non-empty strings")
        item = item.strip()
        if len(item) > max_len:
            raise _ArgError(f"an item of '{key}' is too long (max {max_len} characters)")
        if pattern is not None and not pattern.fullmatch(item):
            raise _ArgError(f"invalid value in '{key}': {item!r}")
        out.append(item)
    return out


def _symbol(value: Any) -> str:
    try:
        return normalize_symbol(value if isinstance(value, str) else "")
    except ValueError:
        raise _ArgError(f"invalid symbol {_token(value, 20)!r} (use a US equity ticker such as AAPL)") from None


def _symbol_arg(args: dict, key: str = "symbol") -> str:
    raw = _str_arg(args, key, required=True, max_len=20)
    return _symbol(raw)


def _symbols_arg(args: dict, key: str, *, max_items: int) -> list[str] | None:
    raw = _str_list_arg(args, key, max_items=max_items, max_len=20)
    if raw is None:
        return None
    out: list[str] = []
    for s in raw:
        sym = _symbol(s)
        if sym not in out:
            out.append(sym)
    return out or None


def _date_arg(args: dict, key: str) -> str | None:
    value = _str_arg(args, key, max_len=10)
    if value is None:
        return None
    try:
        if not _DATE_RE.fullmatch(value):
            raise ValueError
        date.fromisoformat(value)
    except ValueError:
        raise _ArgError(f"'{key}' must be a date like 2026-09-01") from None
    return value


def _timestamp_arg(args: dict, key: str) -> str | None:
    value = _str_arg(args, key, max_len=40)
    if value is None:
        return None
    try:
        if not _TIMESTAMP_RE.fullmatch(value):
            raise ValueError
        datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise _ArgError(f"'{key}' must be a date (2026-09-01) or an RFC 3339 timestamp") from None
    return value


def _money_arg(args: dict, key: str) -> Decimal | None:
    if key not in args:
        return None
    try:
        return parse_money(args[key], key)
    except ValueError as exc:
        raise _ArgError(str(exc)) from None


def _validation_message(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value"))
        for prefix in ("Value error, ", "Assertion failed, "):
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid value"


# --- formatting ---------------------------------------------------------------------------------


def _token(value: Any, max_len: int = 40) -> str:
    """A provider string reduced to a safe single token (ticker/enum/timestamp characters only)."""
    return _TOKEN_STRIP_RE.sub("", str(value if value is not None else ""))[:max_len]


def _sym(value: Any) -> str:
    try:
        return normalize_symbol(value)
    except (ValueError, TypeError):
        return _token(value, 20) or "?"


def _displayable(value: Decimal) -> bool:
    """False for values whose plain (non-scientific) form would be absurdly long."""
    return value == 0 or (value.is_finite() and abs(value.adjusted()) <= _DISPLAY_MAX_ADJUSTED)


def _sci(value: Decimal) -> str:
    return format(value, ".3e") if value.is_finite() else str(value)


def _plain(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if not _displayable(value):
        return _sci(value)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _commas(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if not _displayable(value):
        return _sci(value)
    text = format(value, ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _usd(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    if not _displayable(value):
        return f"${_sci(value)}"
    try:
        q = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    except ArithmeticError:
        return f"${_sci(value)}"
    return f"-${-q:,.2f}" if q < 0 else f"${q:,.2f}"


def _usd_signed(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    text = _usd(value)
    return text if text.startswith("-") or value == 0 else "+" + text


def _price(value: Decimal | None) -> str:
    """Like `_usd`, but sub-penny digits (up to 4 places) are kept."""
    if value is None:
        return "n/a"
    if not _displayable(value):
        return f"${_sci(value)}"
    try:
        if value == value.quantize(_CENT):
            return _usd(value)
        q = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except ArithmeticError:
        return f"${_sci(value)}"
    return f"${q:,.4f}"


def _pct_from_fraction(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    if not _displayable(value):
        return f"{_sci(value)} (fraction)"
    try:
        q = (value * _HUNDRED).quantize(_CENT, rounding=ROUND_HALF_UP)
    except ArithmeticError:
        return f"{_sci(value)} (fraction)"
    return f"+{q}%" if q > 0 else f"{q}%"


def _signed_plain(value: Decimal | None) -> str:
    if value is None:
        return "n/a"
    text = _plain(value)
    return text if text.startswith("-") or value == 0 else "+" + text


def _compact_usd(value: Decimal | None) -> str:
    """$997.35B-style amount. Never raises: anything unrepresentable falls back to $1.000e+40."""
    if value is None or not value.is_finite():
        return "n/a"
    if not _displayable(value):
        return f"${_sci(value)}"
    try:
        for threshold, suffix in ((Decimal("1e12"), "T"), (Decimal("1e9"), "B"), (Decimal("1e6"), "M"), (Decimal("1e3"), "K")):
            if abs(value) >= threshold:
                return f"${(value / threshold).quantize(_CENT, rounding=ROUND_HALF_UP)}{suffix}"
        return _usd(value)
    except ArithmeticError:
        return f"${_sci(value)}"


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def _dec(value: Any) -> Decimal | None:
    """A provider/DB number as a finite Decimal of sane magnitude, or None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, float):
            number = Decimal(repr(value))
        elif isinstance(value, (int, Decimal)):
            number = Decimal(value)
        elif isinstance(value, str):
            text = value.strip()
            if len(text) > 60 or not _DECIMAL_STR_RE.fullmatch(text):
                return None
            number = Decimal(text)
        else:
            return None
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and _displayable(number) else None


def _clean(value: Any, max_chars: int) -> str:
    return sanitize_untrusted(str(value), max_chars=max_chars) if value is not None else ""


def _wrapped_per_source(text: str, source_ids: list[str]) -> bool:
    """True when every id in `source_ids` has its own `<untrusted source="…">` opener (the
    attribute may list several ids) and no wrapper tag remains outside the well-formed blocks - a
    literal "<untrusted" in attacker text, or a block opened for some other id, does not count."""
    opened: set[str] = set()
    for match in _UNTRUSTED_OPENER_RE.finditer(text):
        opened.update(part.strip() for part in match.group(1).split(","))
    if any(source_id not in opened for source_id in source_ids):
        return False
    return not _UNTRUSTED_TAG_RE.search(_UNTRUSTED_BLOCK_RE.sub("", text))


def _parse_quote_time(value: Any) -> datetime | None:
    """An Alpaca quote/trade timestamp (RFC 3339, possibly with nanoseconds) as aware UTC, or None."""
    if not isinstance(value, str):
        return None
    match = _QUOTE_TIME_RE.fullmatch(value.strip())
    if match is None:
        return None
    base, fraction, zone = match.groups()
    zone = "+00:00" if zone in (None, "Z") else zone
    try:
        parsed = datetime.fromisoformat(base + (fraction or "")[:7] + zone)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _oldest_quote_time(ctx: Any, symbols: list[str]) -> datetime | None:
    times = []
    for symbol in symbols:
        quote = ctx.quotes.get(symbol)
        parsed = _parse_quote_time(quote.as_of) if quote is not None else None
        if parsed is not None:
            times.append(parsed)
    return min(times) if times else None


def _one_line(value: Any, max_chars: int) -> str:
    """Our own DB text on one line (a stray newline must not fake a result row), capped."""
    return " ".join(str(value if value is not None else "").split())[:max_chars]


def _trim(text: str, limit: int = TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... (truncated)"


def _is_http_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc) and not re.search(r"\s", url)


def _leg_text(leg: OrderLeg) -> str:
    side = _token(getattr(leg, "side", ""), 10).upper()
    sym = _sym(getattr(leg, "symbol", ""))
    qty, notional = getattr(leg, "qty", None), getattr(leg, "notional", None)
    size = f"{_plain(qty)} {sym}" if qty is not None else f"{_usd(notional)} of {sym}"
    kind = _token(getattr(leg, "type", ""), 10)
    limit = getattr(leg, "limit_price", None)
    if limit is not None:
        kind = f"{kind} {_price(limit)}"
    return f"{side} {size} {kind} {_token(getattr(leg, 'time_in_force', ''), 5)}".strip()


def _risk_report(decisions: list[RiskDecision]) -> list[str]:
    lines: list[str] = []
    for d in decisions:
        head = f"{d.index + 1}. {_leg_text(d.original)} -> {d.verdict.upper()}"
        if d.verdict == "clamp":
            head += f" (clamped to {_leg_text(d.leg)})"
        estimates = []
        if d.est_price is not None:
            estimates.append(f"est. price {_price(d.est_price)}")
        if d.est_cost_usd is not None:
            label = "value" if d.leg.side == "sell" else "cost"
            estimates.append(f"est. {label} {_usd(d.est_cost_usd)}")
        if estimates:
            head += "; " + "; ".join(estimates)
        lines.append(head)
        for reason in d.reasons:
            lines.append(f"   - [{reason.severity}] {reason.code}: {reason.message}")
    return lines


def _mandate_lines(m: Mandate) -> list[str]:
    return [
        f"- direction: {m.direction_text or '(none)'}",
        f"- budget: {_usd(m.budget_usd) if m.budget_usd is not None else 'no cap'}; "
        f"max per order: {_usd(m.max_order_usd) if m.max_order_usd is not None else 'no cap'}",
        f"- allowed symbols: {', '.join(m.allow_symbols) or 'any'}; denied symbols: {', '.join(m.deny_symbols) or 'none'}",
        f"- max position: {_plain(m.max_position_pct)}% of equity; max positions: {m.max_positions}; "
        f"cash buffer: {_plain(m.cash_buffer_pct)}%",
        f"- order types: {', '.join(m.order_types)}; max orders per day: {m.max_orders_per_day}; "
        f"price band: {_plain(m.price_band_pct)}%; fractional shares: {'allowed' if m.allow_fractional else 'not allowed'}",
        f"- horizon: {m.horizon}; universe: {m.universe}",
    ]


# --- tool definitions ---------------------------------------------------------------------------

_MONEY_DESC = 'Decimal string, e.g. "10" or "187.25" (a plain JSON number is accepted, but send a decimal string).'


def _money_schema(description: str) -> dict:
    return {"type": ["string", "number"], "description": f"{description} {_MONEY_DESC}"}


def _obj(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}


@dataclass(frozen=True)
class _ToolDef:
    name: str
    needs: tuple[str, ...]
    description: str
    schema: dict


_LEG_SCHEMA = _obj(
    {
        "symbol": {"type": "string", "description": "US equity ticker, e.g. AAPL"},
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "qty": _money_schema("Number of shares. Give exactly one of qty or notional."),
        "notional": _money_schema("Dollar amount (market day orders only). Give exactly one of qty or notional."),
        "type": {"type": "string", "enum": ["limit", "market"], "default": "limit"},
        "time_in_force": {"type": "string", "enum": ["day", "gtc"], "default": "day"},
        "limit_price": _money_schema("Limit price in USD; required for (and only for) limit orders."),
    },
    ("symbol", "side"),
)

_TOOL_DEFS: tuple[_ToolDef, ...] = (
    # --- account (read-only) ---
    _ToolDef(
        "get_account", ("broker", "data"),
        "Paper-trading account summary (simulated money): status, equity, cash, buying power and trading flags. Read-only.",
        _obj({}),
    ),
    _ToolDef(
        "get_positions", ("broker", "data"),
        "Open positions in the paper account: quantity, average entry, current price, market value and unrealized P/L. Read-only.",
        _obj({}),
    ),
    _ToolDef(
        "list_orders", ("broker", "data"),
        "Lists paper-account orders (read-only). Orders are only ever placed by the user confirming a ticket in the UI.",
        _obj({
            "status": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            "symbols": {**_STR_LIST, "description": "Only these tickers (optional, at most 10)"},
        }),
    ),
    # --- market ---
    _ToolDef(
        "get_market_status", ("broker", "data"),
        "US market clock: whether the market is open now, and the next open and close times.",
        _obj({}),
    ),
    _ToolDef(
        "lookup_asset", ("broker", "data"),
        "Broker tradability of one US equity: status, tradable, fractionable, shortable, marginable.",
        _obj({"symbol": {"type": "string", "description": "Ticker, e.g. AAPL"}}, ("symbol",)),
    ),
    _ToolDef(
        "get_quotes", ("broker", "data"),
        "Latest last-trade, bid and ask for up to 10 US equities (Alpaca IEX feed). Use these numbers; never quote prices from memory.",
        _obj({"symbols": {**_STR_LIST, "description": "1 to 10 tickers"}}, ("symbols",)),
    ),
    _ToolDef(
        "price_metrics", ("broker", "data"),
        "Deterministic daily-price statistics computed server-side from historical bars: last close, 1m/3m/1y total return, "
        "1y annualized volatility, 1y max drawdown, SMA50 and SMA200. Returns, volatility and drawdown are fractions "
        "(0.0500 = 5%). Use these numbers instead of computing your own.",
        _obj({
            "symbol": {"type": "string"},
            "lookback": {"type": "string", "enum": list(LOOKBACK_DAYS), "default": "1y",
                         "description": "History to fetch; windows longer than the history show n/a"},
        }, ("symbol",)),
    ),
    _ToolDef(
        "get_movers", ("broker", "data"),
        "Today's top US stock gainers and losers (kind=movers) or most active stocks by volume (kind=most_actives), from Alpaca's screener.",
        _obj({
            "kind": {"type": "string", "enum": ["movers", "most_actives"], "default": "movers"},
            "top": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        }),
    ),
    # --- research ---
    _ToolDef(
        "search_news", ("data",),
        "Recent market news headlines and summaries (Alpaca news), optionally for specific tickers. Returns untrusted text with [E#] ids.",
        _obj({
            "symbols": {**_STR_LIST, "description": "Tickers (optional, at most 10)"},
            "start": {"type": "string", "description": "Only news since this date (2026-09-01) or RFC 3339 time (optional)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        }),
    ),
    _ToolDef(
        "web_search", ("web", "tavily"),
        "Web search (Tavily). Returns untrusted snippets with [E#] ids; open a promising result with fetch_source.",
        _obj({
            "query": {"type": "string"},
            "topic": {"type": "string", "enum": ["general", "news", "finance"], "default": "general"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD (optional)"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD (optional)"},
            "include_domains": {**_STR_LIST, "description": "Restrict to these domains, e.g. sec.gov (optional, at most 10)"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        }, ("query",)),
    ),
    _ToolDef(
        "fetch_source", ("web",),
        "Fetches the readable text of one web page (http/https URL). Returns untrusted text with an [E#] id.",
        _obj({
            "url": {"type": "string"},
            "focus_query": {"type": "string", "description": "Keep only the paragraphs most relevant to this (optional)"},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 8000, "default": 6000},
        }, ("url",)),
    ),
    _ToolDef(
        "sec_filings", ("edgar",),
        "Lists a company's recent SEC EDGAR filings (default forms 10-K, 10-Q, 8-K) with accession numbers and [E#] ids.",
        _obj({
            "ticker_or_cik": {"type": "string"},
            "forms": {**_STR_LIST, "description": "Form types, e.g. [\"10-K\"] (optional)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        }, ("ticker_or_cik",)),
    ),
    _ToolDef(
        "sec_financial_facts", ("edgar",),
        "Reported XBRL financial facts (us-gaap concepts such as Revenues, NetIncomeLoss, EarningsPerShareDiluted) from SEC EDGAR, latest periods first.",
        _obj({
            "ticker_or_cik": {"type": "string"},
            "concepts": {**_STR_LIST, "description": "1 to 10 us-gaap concept names"},
            "periods": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
        }, ("ticker_or_cik", "concepts")),
    ),
    _ToolDef(
        "sec_filing_passages", ("edgar",),
        "The passages of one SEC filing most relevant to a query (e.g. risk factors, segment results). Returns untrusted text with [E#] ids.",
        _obj({
            "accession_no": {"type": "string", "description": "e.g. 0000320193-25-000079 (from sec_filings)"},
            "cik": {"type": "string"},
            "query": {"type": "string"},
            "sections": {**_STR_LIST, "description": "Prefer these section headings, e.g. [\"Item 1A\"] (optional)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        }, ("accession_no", "cik", "query")),
    ),
    _ToolDef(
        "grounded_web_answer", ("grounded",),
        "Answers a question with a Google-Search-grounded model call. Returns an untrusted answer plus its web sources as [E#] ids; "
        "use for fresh facts other tools lack, and check key numbers against primary sources.",
        _obj({"question": {"type": "string"}}, ("question",)),
    ),
    _ToolDef(
        "screen_us_etfs", ("rdb",),
        "Screens the US-listed ETF master (6,000+ ETFs on sale, base date 2026-08-22) by asset type, keyword (name, index or strategy), "
        "maximum expense ratio and minimum AUM, sorted by AUM or expense ratio.",
        _obj({
            "asset_type": {"type": "string", "enum": list(ETF_ASSET_TYPES)},
            "keyword": {"type": "string", "description": "Matched against the fund name, underlying index and strategy text"},
            "max_expense_ratio": _money_schema("Maximum expense ratio in percent (0.2 means 0.20%)."),
            "min_aum_usd": _money_schema("Minimum assets under management in USD."),
            "sort_by": {"type": "string", "enum": list(_ETF_SORTS), "default": "aum"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
        }),
    ),
    _ToolDef(
        "ask_product_database", ("orchestrator",),
        "Asks the product-database Q&A agent (Korean bonds, Korean ETFs/ETNs, foreign ETFs, funds, holdings, knowledge graph) a "
        "multi-step question; returns its answer and the retrieved context. Slow - use for questions the other tools cannot answer directly.",
        _obj({"question": {"type": "string"}}, ("question",)),
    ),
    # --- knowledge ---
    _ToolDef(
        "search_products", ("registry",),
        "Text search over product names, managers, underlying indices and strategies in the product database (hybrid lexical + semantic).",
        _obj({
            "query": {"type": "string"},
            "table": {"type": "string", "enum": list(VIEW_ID_NAME), "description": "Restrict to one table (optional)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
            "mode": {"type": "string", "enum": ["hybrid", "lexical", "vector"], "default": "hybrid"},
        }, ("query",)),
    ),
    _ToolDef(
        "graph_neighbors", ("registry",),
        "Knowledge-graph relations of an entity (product, company, theme, manager): e.g. constituentOf, hasConstituent, managedBy, hasTheme_inverse.",
        _obj({
            "entity": {"type": "string"},
            "relation": {"type": "string", "description": "Only this relation (optional)"},
        }, ("entity",)),
    ),
    # --- mandate ---
    _ToolDef(
        "draft_mandate", (),
        "Drafts changes to the session's trading mandate (budget, allow/deny lists, sizing limits, order types). The draft is shown to "
        "the user, who reviews and saves it in the UI; it is not in effect until saved. Omitted fields keep their current values.",
        _obj({
            "direction_text": {"type": "string", "description": "The user's investment direction in one or two sentences"},
            "budget_usd": _money_schema("Total budget in USD (null clears the cap)."),
            "allow_symbols": {**_STR_LIST, "description": "Only these tickers may be traded (empty = any)"},
            "deny_symbols": {**_STR_LIST, "description": "These tickers may never be traded"},
            "max_position_pct": _money_schema("Max % of equity in one position (0-100]."),
            "max_positions": {"type": "integer", "minimum": 1},
            "cash_buffer_pct": _money_schema("% of equity always kept in cash [0-100]."),
            "horizon": {"type": "string"},
            "order_types": {"type": "array", "items": {"type": "string", "enum": ["limit", "market"]}},
            "max_orders_per_day": {"type": "integer", "minimum": 1},
            "max_order_usd": _money_schema("Max USD per order (null clears the cap)."),
            "price_band_pct": _money_schema("Max % a limit price may deviate from the quote (0-100]."),
            "allow_fractional": {"type": "boolean"},
        }),
    ),
    # --- trading (proposals only) ---
    _ToolDef(
        "propose_trades", ("broker", "data", "executor", "tickets"),
        "Proposes a ticket of order legs for the USER to confirm in the UI; it never places orders. Each leg is risk-checked against "
        "the account, live quotes and the mandate (accepted, clamped or rejected). Cite the evidence ids behind the thesis "
        "(at least one for any buy). At most one proposal per turn.",
        _obj({
            "legs": {"type": "array", "items": _LEG_SCHEMA, "description": f"1 to {MAX_LEGS} order legs"},
            "thesis": {"type": "string", "description": "Why, citing [E#] ids"},
            "risks": {"type": "string"},
            "invalidation": {"type": "string", "description": "What would prove the thesis wrong"},
            "evidence_ids": {**_STR_LIST, "description": "Evidence ids (E#) from this session's research"},
        }, ("legs", "thesis", "risks", "invalidation", "evidence_ids")),
    ),
)

_TOOLS_BY_NAME = {d.name: d for d in _TOOL_DEFS}
for _def in _TOOL_DEFS:  # a hard guard, not an assert (asserts vanish under -O)
    if FORBIDDEN_TOOL_NAME_RE.search(_def.name):
        raise RuntimeError(f"tool name {_def.name!r} looks like an order-placing action")

_MANDATE_MONEY_FIELDS = ("budget_usd", "max_position_pct", "cash_buffer_pct", "max_order_usd", "price_band_pct")
_MANDATE_NULLABLE = ("budget_usd", "max_order_usd")
_LEG_FIELDS = frozenset(_LEG_SCHEMA["properties"])
_LEG_MONEY_FIELDS = ("qty", "notional", "limit_price")


class ChatToolbox:
    """Tool specs + dispatch for the chat agent. See the module docstring for the rules."""

    def __init__(
        self,
        *,
        registry,
        orchestrator,
        broker,
        data,
        edgar,
        web,
        grounded,
        evidence,
        sessions,
        tickets,
        executor,
        settings,
        now: Callable[[], datetime] | None = None,
        turn_slots: OrderedDict[tuple[str, str], str] | None = None,
        ledger=None,
        known_secrets: Iterable[str] = (),
    ):
        self._registry = registry
        self._orchestrator = orchestrator
        self._broker = broker
        self._data = data
        self._edgar = edgar
        self._web = web
        self._grounded = grounded
        self._evidence = evidence
        self._sessions = sessions
        self._tickets = tickets
        self._executor = executor
        self._settings = settings
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._rdb = getattr(registry, "rdb", None) if registry is not None else None
        # (session_id, turn_id) -> ticket id ("" while a proposal is in flight). The owner of several
        # toolboxes in turn (ChatRuntime rebuilds) passes one shared map, so a rebuild mid-turn
        # cannot reset the one-proposal-per-turn slot.
        self._proposal_turns: OrderedDict[tuple[str, str], str] = (
            turn_slots if turn_slots is not None else OrderedDict()
        )
        self._audit = ToolAudit(ledger, _TOOLS_BY_NAME)  # ledger None: nothing is recorded
        # Secret values to blank out of broker messages besides the current settings' keys: the
        # owner's accumulated list, so a key that was replaced is still scrubbed.
        self._known_secrets = tuple(known_secrets)

    # --- specs ---------------------------------------------------------------------------------

    def _available(self, tool: _ToolDef) -> bool:
        deps = {
            "broker": self._broker,
            "data": self._data,
            "executor": self._executor,
            "tickets": self._tickets,
            "web": self._web,
            "edgar": self._edgar,
            "grounded": self._grounded,
            "registry": self._registry,
            "rdb": self._rdb,
            "orchestrator": self._orchestrator,
        }
        for need in tool.needs:
            if need == "tavily":
                if not getattr(self._settings, "tavily_api_key", None):
                    return False
            elif deps[need] is None:
                return False
        return True

    def specs(self) -> list[ToolSpec]:
        """Every tool whose dependencies are configured (the rest are omitted, not stubbed)."""
        return [
            ToolSpec(t.name, t.description, copy.deepcopy(t.schema))
            for t in _TOOL_DEFS
            if self._available(t)
        ]

    # --- dispatch ------------------------------------------------------------------------------

    async def execute(self, session_id: str, name: str, args: dict | None, *, turn_id: str) -> ToolOutput:
        """Run one tool call (`_dispatch`), then append its `tool_call` audit entry. The latency
        recorded is the tool's own: it is read as soon as `_dispatch` returns or raises, before the
        entry is built."""
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return max(0, int((time.perf_counter() - started) * 1000))

        try:
            out = await self._dispatch(session_id, name, args, turn_id)
        except (asyncio.CancelledError, Exception) as exc:
            # A cancellation, or an exception `_dispatch` did not turn into a ToolOutput: record
            # it and let it propagate - never swallow it. Any other BaseException
            # (KeyboardInterrupt, GeneratorExit, SystemExit) propagates without a ledger write.
            self._audit.tool_call(session_id, turn_id, name, args, elapsed_ms(), raised=exc)
            raise
        self._audit.tool_call(session_id, turn_id, name, args, elapsed_ms(), out)
        return out

    def _refused(self, session_id: str, turn_id: str, name: Any, text: str) -> ToolOutput:
        """A call refused before its handler ran. For `propose_trades` that is a proposal that
        ended without a ticket, so it is audited as one."""
        if name == "propose_trades":
            self._audit.proposal_rejected(session_id, turn_id, "dispatch_error")
        return ToolOutput(text, is_error=True)

    async def _dispatch(self, session_id: str, name: str, args: dict | None, turn_id: str) -> ToolOutput:
        tool = _TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
        if tool is None or not self._available(tool):
            available = ", ".join(s.name for s in self.specs())
            return self._refused(session_id, turn_id, name,
                                 f"Unknown tool: {_token(name, 60)}. Available tools: {available}.")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return self._refused(session_id, turn_id, name, "Invalid arguments: expected a JSON object.")
        props = tool.schema.get("properties", {})
        unknown = [k for k in args if k not in props]
        if unknown:
            allowed = ", ".join(props) or "none"
            return self._refused(
                session_id, turn_id, name,
                f"Invalid arguments: unknown argument(s) {', '.join(_token(k, 40) for k in unknown)}. Allowed: {allowed}.",
            )
        missing = [k for k in tool.schema.get("required", []) if args.get(k) is None]
        if missing:
            return self._refused(session_id, turn_id, name,
                                 f"Invalid arguments: missing required argument(s) {', '.join(missing)}.")
        handler = getattr(self, f"_t_{tool.name}")
        try:
            return await handler(session_id, args, turn_id)
        except _ArgError as exc:
            return ToolOutput(f"Invalid arguments: {exc}", is_error=True)
        except BrokerError as exc:  # includes OrderUncertain
            return ToolOutput(self._broker_error_text(exc), is_error=True)
        except HttpError as exc:
            log.warning("tool %s: upstream HTTP %s", tool.name, exc.status)
            return ToolOutput(f"Upstream data request failed (HTTP {exc.status}).", is_error=True)
        except Exception as exc:  # noqa: BLE001 - tool failures go back to the model, never raise
            # Frames only: the exception message and any chained causes can carry provider text
            # or secrets, so they are never logged (no exc_info).
            frames = "".join(traceback.format_tb(exc.__traceback__))
            log.warning("tool %s failed: %s (message omitted)\n%s", tool.name, type(exc).__name__, frames)
            return ToolOutput(f"Tool error ({type(exc).__name__})", is_error=True)

    def _scrub(self, text: str) -> str:
        """`text` without any key this app uses or has used (the settings' current keys and the
        owner's accumulated `known_secrets`), longest first; values under the 8-character floor
        the ledger and the log scrubber use are ignored."""
        current = (getattr(self._settings, attr, None) for attr in (
            "alpaca_api_key_id", "alpaca_api_secret_key", "tavily_api_key", "gemini_api_key",
            "anthropic_api_key", "openai_api_key"))
        secrets = {value.strip() for value in (*current, *self._known_secrets)
                   if isinstance(value, str) and len(value.strip()) >= MIN_KNOWN_SECRET_LEN}
        for secret in sorted(secrets, key=lambda s: (-len(s), s)):
            text = text.replace(secret, "[redacted]")
        return text

    def _broker_error_text(self, exc: BrokerError) -> str:
        message = self._scrub(str(getattr(exc, "message", "") or type(exc).__name__))
        clean = sanitize_untrusted(message, max_chars=300)
        return f"Broker request failed (HTTP {exc.status}): {wrap_untrusted('broker', clean)}"

    def _research(self, result: ResearchResult) -> ToolOutput:
        """A connector result as a ToolOutput. Defense in depth: unless every untrusted source has
        its own `<untrusted source="…id…">` block and no stray wrapper tag is left outside the
        blocks, the whole text is sanitized and wrapped here."""
        text = result.text or ""
        if not result.is_error:
            untrusted_ids = list(dict.fromkeys(s.id for s in result.sources if s.untrusted))
            if untrusted_ids and not _wrapped_per_source(text, untrusted_ids):
                text = wrap_untrusted(",".join(untrusted_ids), sanitize_untrusted(text, max_chars=_RESEARCH_MAX_CHARS))
        return ToolOutput(text=text, sources=list(result.sources), is_error=bool(result.is_error))

    # --- account -------------------------------------------------------------------------------

    async def _t_get_account(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        a = await self._broker.account()
        lines = [
            "Alpaca PAPER trading account (simulated money):",
            f"- status: {_token(a.status)}",
            f"- equity: {_usd(a.equity)} (previous close {_usd(a.last_equity)}, change today {_usd_signed(a.equity - a.last_equity)})",
            f"- cash: {_usd(a.cash)}",
            f"- buying power: {_usd(a.buying_power)} (non-marginable {_usd(a.non_marginable_buying_power)}; "
            "proposals are sized against cash, never margin)",
            f"- trading blocked: {_yn(a.trading_blocked)}; account blocked: {_yn(a.account_blocked)}; "
            f"trading suspended by user: {_yn(a.trade_suspended_by_user)}; shorting enabled: {_yn(a.shorting_enabled)}",
        ]
        return ToolOutput("\n".join(lines))

    async def _t_get_positions(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        positions = list((await self._broker.positions()).values())
        if not positions:
            return ToolOutput("No open positions in the paper account.")
        positions.sort(key=lambda p: p.market_value, reverse=True)
        total = sum((p.market_value for p in positions), Decimal(0))
        lines = [f"{len(positions)} open position(s) in the paper account, total market value {_usd(total)}:"]
        for p in positions:
            available = f" ({_plain(p.qty_available)} available to sell)" if p.qty_available != p.qty else ""
            lines.append(
                f"- {_sym(p.symbol)}: {_plain(p.qty)} shares {_token(p.side, 10)}{available}, avg entry {_price(p.avg_entry_price)}, "
                f"current {_price(p.current_price)}, market value {_usd(p.market_value)}, "
                f"unrealized P/L {_usd_signed(p.unrealized_pl)} ({_pct_from_fraction(p.unrealized_plpc)})"
            )
        return ToolOutput("\n".join(lines))

    async def _t_list_orders(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        status = _enum_arg(args, "status", ("open", "closed", "all"), default="open")
        limit = _int_arg(args, "limit", default=20, lo=1, hi=50)
        symbols = _symbols_arg(args, "symbols", max_items=10)
        orders = await self._broker.orders(status=status, limit=limit, symbols=symbols)
        if not orders:
            return ToolOutput(f"No {status} orders in the paper account.")
        lines = [f"{len(orders)} {status} order(s) in the paper account (most recent first, up to {limit}):"]
        for o in orders:
            size = f"{_plain(o.qty)} shares" if o.qty is not None else f"{_usd(o.notional)} notional"
            price = f" {_price(o.limit_price)}" if o.limit_price is not None else ""
            filled = ""
            if o.filled_qty:
                filled = f", filled {_plain(o.filled_qty)}"
                if o.filled_avg_price is not None:
                    filled += f" @ {_price(o.filled_avg_price)}"
            submitted = f", submitted {_token(o.submitted_at)}" if o.submitted_at else ""
            lines.append(
                f"- {_sym(o.symbol)} {_token(o.side, 10)} {size} {_token(o.type, 20)}{price} ({_token(o.time_in_force, 10)}), "
                f"status {_token(o.status, 30)}{filled}{submitted}"
            )
        return ToolOutput("\n".join(lines))

    # --- market --------------------------------------------------------------------------------

    async def _t_get_market_status(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        c = await self._broker.clock()
        state = "OPEN" if c.is_open else "CLOSED"
        return ToolOutput(
            f"US market is {state} (as of {_token(c.timestamp)}). Next open: {_token(c.next_open)}; "
            f"next close: {_token(c.next_close)}."
        )

    async def _t_lookup_asset(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        symbol = _symbol_arg(args)
        a = await self._broker.asset(symbol)
        text = (
            f"{_sym(a.symbol)}: class {_token(a.cls)}, status {_token(a.status)}, tradable {_yn(a.tradable)}, "
            f"fractionable {_yn(a.fractionable)}, shortable {_yn(a.shortable)}, marginable {_yn(a.marginable)}"
        )
        if a.borrow_status:
            text += f", borrow status {_token(a.borrow_status)}"
        return ToolOutput(text)

    async def _t_get_quotes(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        raw = _str_list_arg(args, "symbols", required=True, max_items=MAX_QUOTE_SYMBOLS, max_len=20)
        valid: list[str] = []
        invalid: list[str] = []
        for s in raw:
            try:
                sym = normalize_symbol(s)
            except ValueError:
                invalid.append(s)
                continue
            if sym not in valid:
                valid.append(sym)
        quotes = await self._data.snapshots(valid) if valid else {}
        lines: list[str] = []
        found: list[str] = []
        latest = ""
        for sym in valid:
            q = quotes.get(sym)
            if q is None:
                lines.append(f"- {sym}: no quote available")
                continue
            found.append(sym)
            as_of = _token(q.as_of) if q.as_of else "n/a"
            latest = max(latest, as_of) if as_of != "n/a" else latest
            lines.append(f"- {sym}: last {_price(q.last)}, bid {_price(q.bid)}, ask {_price(q.ask)}, as of {as_of}")
        for s in invalid:
            lines.append(f"- {_token(s, 20) or '?'}: invalid symbol")
        body = "\n".join(lines)
        header = ("Latest quotes (Alpaca IEX feed: prices from a single exchange, which can differ slightly "
                  "from the consolidated tape):")
        if not found:
            return ToolOutput(f"{header}\n{body}")
        src = self._evidence.add(
            session_id, connector="alpaca_quotes", title=f"Alpaca IEX quotes: {', '.join(found)}", url=None,
            publisher="Alpaca Markets (IEX feed)", tier=1, published_at=latest or None, snippet=body,
            untrusted=False,
        )
        return ToolOutput(f"[{src.id}] {header}\n{body}", sources=[src])

    async def _t_price_metrics(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        symbol = _symbol_arg(args)
        lookback = _enum_arg(args, "lookback", tuple(LOOKBACK_DAYS), default="1y")
        now = self._now()
        start = (now - timedelta(days=LOOKBACK_DAYS[lookback])).date().isoformat()
        # An end at least 16 minutes old lets the connector use the consolidated (SIP) feed.
        end = (now - timedelta(minutes=16)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bars = await self._data.bars(symbol, timeframe="1Day", start=start, end=end, limit=1000)
        if not bars:
            return ToolOutput(f"No daily bars were returned for {symbol} since {start}.")
        metrics = summarize(bars)
        n = len(bars)
        lines = []
        for key in _METRIC_ORDER:
            need = _METRIC_MIN_BARS.get(key)
            if need is not None and n < need:
                lines.append(f"- {key}: n/a (needs {need} daily bars; only {n} available)")
            else:
                value = metrics.get(key)
                lines.append(f"- {key}: {value if value is not None else 'n/a'}")
        first = _token(min(b.t for b in bars)[:10])
        body = "\n".join(lines)
        header = (
            f"{symbol} daily price metrics (lookback {lookback}: {n} daily bars from {first} to "
            f"{_token(metrics.get('as_of'))}; returns, volatility and drawdown are fractions, e.g. 0.0500 = 5%; "
            "prices in USD):"
        )
        src = self._evidence.add(
            session_id, connector="alpaca_bars", title=f"{symbol} daily bars ({lookback}) and metrics", url=None,
            publisher="Alpaca Markets", tier=1, published_at=_token(metrics.get("as_of")) or None,
            snippet=f"{header}\n{body}", untrusted=False,
        )
        return ToolOutput(f"[{src.id}] {header}\n{body}", sources=[src])

    async def _t_get_movers(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        kind = _enum_arg(args, "kind", ("movers", "most_actives"), default="movers")
        top = _int_arg(args, "top", default=10, lo=1, hi=20)
        lines: list[str] = []
        if kind == "movers":
            payload = await self._data.movers(top=top)
            payload = payload if isinstance(payload, dict) else {}
            market = _clean(payload.get("market_type"), 20) or "n/a"
            updated = _clean(payload.get("last_updated"), 40) or "n/a"
            lines.append(f"Top US stock movers (Alpaca screener; market type {market}; last updated {updated}):")
            for label, key in (("Gainers", "gainers"), ("Losers", "losers")):
                items = payload.get(key) if isinstance(payload.get(key), list) else []
                lines.append(f"{label}:")
                shown = 0
                for item in items[:top]:
                    if not isinstance(item, dict):
                        continue
                    sym = _clean(item.get("symbol"), 20) or "?"
                    lines.append(
                        f"- {sym}: {_signed_plain(_dec(item.get('percent_change')))}% "
                        f"(change {_signed_plain(_dec(item.get('change')))}, price {_plain(_dec(item.get('price')))})"
                    )
                    shown += 1
                if not shown:
                    lines.append("- none")
        else:
            items = await self._data.most_actives(top=top)
            items = items if isinstance(items, list) else []
            lines.append("Most active US stocks by volume (Alpaca screener):")
            for item in items[:top]:
                if not isinstance(item, dict):
                    continue
                sym = _clean(item.get("symbol"), 20) or "?"
                lines.append(
                    f"- {sym}: volume {_commas(_dec(item.get('volume')))}, trades {_commas(_dec(item.get('trade_count')))}"
                )
            if len(lines) == 1:
                lines.append("- none")
        body = "\n".join(lines)
        src = self._evidence.add(
            session_id, connector="alpaca_screener", title=f"Alpaca screener: {kind} (top {top})", url=None,
            publisher="Alpaca Markets", tier=2, published_at=None, snippet=body, untrusted=True,
        )
        return ToolOutput(f"[{src.id}] {wrap_untrusted(src.id, body)}", sources=[src])

    # --- research ------------------------------------------------------------------------------

    async def _t_search_news(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        symbols = _symbols_arg(args, "symbols", max_items=10)
        start = _timestamp_arg(args, "start")
        limit = _int_arg(args, "limit", default=10, lo=1, hi=20)
        result = await self._data.news(session_id, self._evidence, symbols=symbols, start=start, limit=limit)
        return self._research(result)

    async def _t_web_search(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        query = _str_arg(args, "query", required=True, max_len=400)
        topic = _enum_arg(args, "topic", ("general", "news", "finance"), default="general")
        start_date = _date_arg(args, "start_date")
        end_date = _date_arg(args, "end_date")
        domains = _str_list_arg(args, "include_domains", max_items=10, max_len=100, pattern=_DOMAIN_RE)
        max_results = _int_arg(args, "max_results", default=5, lo=1, hi=10)
        result = await self._web.search(
            session_id, query, topic=topic, start_date=start_date, end_date=end_date,
            include_domains=domains, max_results=max_results,
        )
        return self._research(result)

    async def _t_fetch_source(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        url = _str_arg(args, "url", required=True, max_len=2000)
        if not _is_http_url(url):
            raise _ArgError("'url' must be an http(s) URL")
        focus = _str_arg(args, "focus_query", max_len=300)
        max_chars = _int_arg(args, "max_chars", default=6000, lo=500, hi=8000)
        result = await self._web.fetch(session_id, url, focus_query=focus, max_chars=max_chars)
        return self._research(result)

    async def _t_sec_filings(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        who = self._ticker_or_cik(args)
        forms = _str_list_arg(args, "forms", max_items=10, max_len=15, pattern=_FORM_RE)
        limit = _int_arg(args, "limit", default=10, lo=1, hi=20)
        kwargs: dict[str, Any] = {"limit": limit}
        if forms:
            kwargs["forms"] = forms
        return self._research(await self._edgar.filings(session_id, who, **kwargs))

    async def _t_sec_financial_facts(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        who = self._ticker_or_cik(args)
        concepts = _str_list_arg(args, "concepts", required=True, max_items=10, max_len=150, pattern=_CONCEPT_RE)
        periods = _int_arg(args, "periods", default=8, lo=1, hi=20)
        return self._research(await self._edgar.financial_facts(session_id, who, concepts, periods=periods))

    async def _t_sec_filing_passages(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        accession = _str_arg(args, "accession_no", required=True, max_len=20)
        if not _ACCESSION_RE.fullmatch(accession):
            raise _ArgError("'accession_no' must look like 0000320193-25-000079")
        cik = _str_arg(args, "cik", required=True, max_len=13)
        if not _CIK_RE.fullmatch(cik):
            raise _ArgError("'cik' must be a numeric SEC CIK")
        query = _str_arg(args, "query", required=True, max_len=300)
        sections = _str_list_arg(args, "sections", max_items=10, max_len=100)
        k = _int_arg(args, "k", default=5, lo=1, hi=8)
        result = await self._edgar.filing_passages(session_id, accession, cik, query, sections=sections, k=k)
        return self._research(result)

    @staticmethod
    def _ticker_or_cik(args: dict) -> str:
        value = _str_arg(args, "ticker_or_cik", required=True, max_len=20)
        if not _TICKER_OR_CIK_RE.fullmatch(value):
            raise _ArgError("'ticker_or_cik' must be a ticker (AAPL) or a numeric CIK")
        return value

    async def _t_grounded_web_answer(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        question = _str_arg(args, "question", required=True, max_len=1000)
        answer = await self._grounded.answer(question)  # never cached (see research/grounding.py)
        if answer.error:
            return ToolOutput(f"Grounded web answer unavailable: {sanitize_untrusted(answer.error, max_chars=300)}",
                              is_error=True)
        if not (answer.text or "").strip():
            return ToolOutput("The search returned no answer (possibly filtered).", is_error=True)
        sources: list[SourceRecord] = []
        source_lines: list[str] = []
        for item in answer.sources or []:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            uri = uri if isinstance(uri, str) and len(uri) <= 2000 and _is_http_url(uri) else None
            # A missing title falls back to the URI - through the same sanitizer as a title.
            title = _clean(item.get("title") or "", 200) or (_clean(uri, 200) if uri else "") or "Untitled"
            # Only the title goes into the evidence record, never the grounded answer text.
            record = self._evidence.add(
                session_id, connector="grounded_web", title=title, url=uri, publisher=title, tier=3,
                published_at=None, snippet=title, untrusted=True,
            )
            if any(s.id == record.id for s in sources):
                continue
            sources.append(record)
            source_lines.append(f"[{record.id}] {wrap_untrusted(record.id, title)}")
        body = sanitize_untrusted(answer.text, max_chars=TEXT_LIMIT)
        queries = [q for q in (_clean(q, 200) for q in (answer.queries or [])[:5]) if q]
        if queries:
            body += f" (search queries: {'; '.join(queries)})"
        answer_ids = ",".join(s.id for s in sources) or "grounded_web"
        text = (
            "Grounded web answer (Gemini with Google Search; untrusted, check key numbers against primary sources):\n"
            f"{wrap_untrusted(answer_ids, body)}\nSources:\n" + ("\n".join(source_lines) or "(none reported)")
        )
        events = []
        if answer.search_suggestions_html:
            # Google requires this attribution HTML be shown unmodified; the UI renders it as is.
            events.append({"type": "source-grounding", "search_suggestions_html": answer.search_suggestions_html})
        return ToolOutput(text, sources=sources, events=events)

    async def _t_screen_us_etfs(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        asset_type = self._etf_asset_type(args.get("asset_type"))
        keyword = _str_arg(args, "keyword", max_len=100)
        max_er = _money_arg(args, "max_expense_ratio")
        if max_er is not None and not Decimal(0) <= max_er <= _HUNDRED:
            raise _ArgError("'max_expense_ratio' is a percentage between 0 and 100")
        min_aum = _money_arg(args, "min_aum_usd")
        if min_aum is not None and min_aum < 0:
            raise _ArgError("'min_aum_usd' must not be negative")
        sort_by = _enum_arg(args, "sort_by", tuple(_ETF_SORTS), default="aum")
        limit = _int_arg(args, "limit", default=10, lo=1, hi=25)

        # Fully parameterized: every model value is a bound parameter; the SQL text is assembled
        # only from the fixed fragments below and the fixed ORDER BY map.
        where = ["is_on_sale = 1"]
        params: list[Any] = []
        filters: list[str] = []
        if asset_type:
            where.append("wu_inv_ast_type = ?")
            params.append(asset_type)
            filters.append(f"asset type {asset_type}")
        if keyword:
            escaped = (keyword.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
                       .replace("%", _LIKE_ESCAPE + "%").replace("_", _LIKE_ESCAPE + "_"))
            where.append(_ETF_KEYWORD_SQL)
            params.extend([f"%{escaped}%"] * 3)
            filters.append(f"keyword {keyword!r}")
        if max_er is not None:
            where.append("expense_ratio_pct <= ?")
            params.append(float(max_er))
            filters.append(f"expense ratio <= {_plain(max_er)}%")
        if min_aum is not None:
            where.append("du_last_aum >= ?")
            params.append(float(min_aum))
            filters.append(f"AUM >= {_compact_usd(min_aum)}")
        where_sql = " AND ".join(where)
        total = self._rdb.fetch_all(f"SELECT COUNT(*) FROM v_etf_gl WHERE {where_sql}", tuple(params))[0][0]
        rows = self._rdb.fetch_all(
            "SELECT ticker, pd_nm, expense_ratio_pct, du_last_aum, wu_inv_ast_type, cu_base_index "
            f"FROM v_etf_gl WHERE {where_sql} ORDER BY {_ETF_SORTS[sort_by]} LIMIT ?",
            tuple(params) + (limit,),
        )
        base_date = ((getattr(self._registry, "meta", None) or {}).get("v_etf_gl") or {}).get("base_date") \
            or _ETF_DEFAULT_BASE_DATE
        filter_text = "; ".join(filters) or "none"
        if not rows:
            return ToolOutput(
                f"No US-listed ETFs on sale match these filters ({filter_text}) in the ETF master (base date {base_date})."
            )
        order_text = "largest AUM first" if sort_by == "aum" else "lowest expense ratio first"
        lines = []
        for i, (ticker, name, er, aum, kind, index) in enumerate(rows, 1):
            er_dec = _dec(er)
            lines.append(
                f"{i}. {_token(ticker, 20)} — {_one_line(name, 100)} | expense ratio "
                f"{_plain(er_dec) + '%' if er_dec is not None else 'n/a'} | AUM {_compact_usd(_dec(aum))} | "
                f"{_one_line(kind, 30) or 'n/a'} | index: {_one_line(index, 100) or 'n/a'}"
            )
        header = (f"US ETF screen (US-listed ETFs on sale, ETF master base date {base_date}): {total} match(es); "
                  f"showing {len(rows)}, {order_text}. Filters: {filter_text}.")
        body = "\n".join(lines)
        src = self._evidence.add(
            session_id, connector="product_db", title=f"US ETF master (base date {base_date})", url=None,
            publisher="Product database (v_etf_gl)", tier=1, published_at=base_date,
            snippet=f"{header}\n{body}", untrusted=False,
        )
        return ToolOutput(f"[{src.id}] {header}\n{body}", sources=[src])

    @staticmethod
    def _etf_asset_type(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            key = value.strip().lower()
            for canonical in ETF_ASSET_TYPES:
                if key == canonical.lower():
                    return canonical
            if key in _ETF_ASSET_ALIASES:
                return _ETF_ASSET_ALIASES[key]
        raise _ArgError(f"'asset_type' must be one of: {', '.join(ETF_ASSET_TYPES)}")

    async def _t_ask_product_database(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        question = _str_arg(args, "question", required=True, max_len=1000)
        response = await self._orchestrator.answer("chat", question)
        answer = str(getattr(response, "answer", "") or "").strip()
        context = str(getattr(response, "retrieved_context", "") or "").strip()
        src = self._evidence.add(
            session_id, connector="product_qa", title=f"Product database Q&A: {question[:120]}", url=None,
            publisher="Product database Q&A agent", tier=1, published_at=None,
            snippet=answer[:2000] or question, untrusted=False,
        )
        text = f"[{src.id}] Product database answer:\n{answer or '(no answer)'}\n\nRetrieved context:\n{context or '(none)'}"
        return ToolOutput(_trim(text), sources=[src])

    # --- knowledge -----------------------------------------------------------------------------

    async def _t_search_products(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        query = _str_arg(args, "query", required=True, max_len=200)
        forward: dict[str, Any] = {"query": query}
        table = _enum_arg(args, "table", tuple(VIEW_ID_NAME))
        if table:
            forward["table"] = table
        if args.get("k") is not None:
            forward["k"] = _int_arg(args, "k", default=10, lo=1, hi=30)
        mode = _enum_arg(args, "mode", ("hybrid", "lexical", "vector"))
        if mode:
            forward["mode"] = mode
        result = await self._registry.execute("search_products", forward)
        return self._knowledge(session_id, f"search_products: {query}", result)

    async def _t_graph_neighbors(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        entity = _str_arg(args, "entity", required=True, max_len=200)
        forward: dict[str, Any] = {"entity": entity}
        relation = _str_arg(args, "relation", max_len=60)
        if relation:
            forward["relation"] = relation
        result = await self._registry.execute("graph_neighbors", forward)
        return self._knowledge(session_id, f"graph_neighbors: {entity}" + (f" ({relation})" if relation else ""), result)

    def _knowledge(self, session_id: str, title: str, result) -> ToolOutput:
        content = str(getattr(result, "content", "") or "")
        if getattr(result, "is_error", False):
            return ToolOutput(_trim(content) or "Knowledge lookup failed.", is_error=True)
        src = self._evidence.add(
            session_id, connector="product_kb", title=title[:200], url=None, publisher="Product knowledge base",
            tier=1, published_at=None, snippet=content[:2000] or title, untrusted=False,
        )
        return ToolOutput(f"[{src.id}] " + _trim(content), sources=[src])

    # --- mandate -------------------------------------------------------------------------------

    async def _t_draft_mandate(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        current = self._sessions.get_mandate(session_id) or Mandate()
        data = current.model_dump()
        changed: list[str] = []
        for key, value in args.items():
            if value is None and key not in _MANDATE_MONEY_FIELDS:
                continue  # null = omitted: keep the current value (only the money caps clear on null)
            if key in _MANDATE_MONEY_FIELDS:
                if value is None and key in _MANDATE_NULLABLE:
                    data[key] = None
                else:
                    try:
                        data[key] = parse_money(value, key)
                    except ValueError as exc:
                        raise _ArgError(str(exc)) from None
            elif key in ("max_positions", "max_orders_per_day"):
                data[key] = _int_arg(args, key, default=data[key], lo=1, hi=10_000)
            elif key == "allow_fractional":
                if not isinstance(value, bool):
                    raise _ArgError("'allow_fractional' must be true or false")
                data[key] = value
            elif key in ("allow_symbols", "deny_symbols"):
                data[key] = _str_list_arg(args, key, max_items=50, max_len=20) or []
            elif key == "order_types":
                data[key] = _str_list_arg(args, key, max_items=2, max_len=10) or []
            elif key == "direction_text":
                if not isinstance(value, str):
                    raise _ArgError("'direction_text' must be a string")
                if len(value) > 1000:
                    raise _ArgError("'direction_text' is too long (max 1000 characters)")
                data[key] = value.strip()
            elif key == "horizon":
                data[key] = _str_arg(args, key, required=True, max_len=100)
            changed.append(key)
        try:
            draft = Mandate.model_validate(data)
        except ValidationError as exc:
            raise _ArgError(_validation_message(exc)) from None
        lines = [
            "Draft mandate (NOT saved and NOT in effect):",
            *_mandate_lines(draft),
            f"Changed fields: {', '.join(changed) or 'none'}.",
            "The user must review and save this draft in the UI; until then the current mandate stays in effect.",
        ]
        event = {"type": "data-mandate", "mandate": draft.model_dump(mode="json"), "draft": True}
        return ToolOutput("\n".join(lines), events=[event])

    # --- trade proposals -----------------------------------------------------------------------

    async def _t_propose_trades(self, session_id: str, args: dict, turn_id: str) -> ToolOutput:
        """`_propose`, plus the audit catch-all: a call that ends without a ticket for any reason
        leaves one `proposal_rejected` entry. A cancellation is not caught here (it leaves only
        its cancelled `tool_call`). Every exception is re-raised for `_dispatch` to report."""
        outcome = self._audit.proposal_outcome(session_id, turn_id)
        try:
            out = await self._propose(session_id, args, turn_id, outcome)
        except _ArgError:
            outcome.rejected("invalid_arguments")
            raise
        except NotPaperAccount:
            outcome.rejected("not_paper_account")
            return ToolOutput(
                "No ticket was created: the connected Alpaca account is not a paper account. This app trades "
                "paper accounts only, so trading is now disconnected. Tell the user to save paper-trading keys "
                "in Settings. You cannot place orders.",
                is_error=True,
            )
        except BrokerError:
            outcome.rejected("broker_error")
            raise
        except HttpError:
            outcome.rejected("upstream_error")
            raise
        except Exception:
            outcome.rejected("unexpected_error")
            raise
        outcome.rejected("unexpected_error")  # a no-op unless `_propose` returned without recording
        return out

    async def _propose(self, session_id: str, args: dict, turn_id: str, outcome: ProposalOutcome) -> ToolOutput:
        slot = (session_id, turn_id)
        if slot in self._proposal_turns:
            existing = self._proposal_turns[slot]
            already = f" (ticket {existing})" if existing else ""
            outcome.rejected("one_proposal_per_turn")
            return ToolOutput(
                "Only one trade proposal per turn is allowed, and this turn already produced one"
                f"{already}. Ask the user to review it in the UI; you cannot place orders.",
                is_error=True,
            )
        legs = self._parse_legs(args.get("legs"))
        thesis = _str_arg(args, "thesis", required=True, max_len=2000)
        risks = _str_arg(args, "risks", required=True, max_len=2000)
        invalidation = _str_arg(args, "invalidation", required=True, max_len=2000)
        cited = self._evidence_records(session_id, args.get("evidence_ids"))
        evidence_ids = [record.id for record in cited]
        if any(leg.side == "buy" for leg in legs) and all(r.connector in MARKET_DATA_CONNECTORS for r in cited):
            listed = ", ".join(f"{r.id} ({r.connector})" for r in cited) or "none"
            raise _ArgError(
                "a proposal with a buy leg needs at least one research source among its evidence ids (news, web "
                "search or page, SEC filings, grounded web answer, or the product database); quotes, price metrics "
                f"and movers alone are not enough. Cited: {listed}."
            )

        # Reserve the turn's single proposal slot before the first await, so concurrent calls in
        # the same turn cannot both create a ticket; it is released unless a ticket is created.
        self._proposal_turns[slot] = ""
        while len(self._proposal_turns) > _PROPOSAL_TURNS_KEPT:
            self._proposal_turns.popitem(last=False)
        created = False
        try:
            symbols = list(dict.fromkeys(leg.symbol for leg in legs))
            ctx = await self._executor.build_context(symbols)
            mandate = self._sessions.get_mandate(session_id) or Mandate()
            decisions = evaluate(legs, ctx, mandate)
            report = _risk_report(decisions)
            refused = dict(decisions=decisions, mandate_version=mandate.version, evidence_ids=evidence_ids)
            if all(d.verdict == "reject" for d in decisions):
                outcome.rejected("all_legs_rejected", **refused)
                return self._all_rejected(report)
            # The ticket is only as fresh as its stalest leg quote; the store clock is the fallback
            # when no leg quote carries a timestamp.
            quote_as_of = _oldest_quote_time(ctx, symbols)
            if quote_as_of is None:
                clock = getattr(self._tickets, "clock", None)
                quote_as_of = clock() if callable(clock) else self._now()
            try:
                ticket = self._tickets.create(
                    session_id,
                    decisions=decisions,
                    thesis=thesis,
                    risks=risks,
                    invalidation=invalidation,
                    evidence_ids=evidence_ids,
                    mandate_version=mandate.version,
                    quote_as_of=quote_as_of,
                    ttl_sec=int(getattr(self._settings, "ticket_ttl_sec", 300)),
                )
            except ValueError as exc:
                if str(exc).startswith("no executable legs"):
                    outcome.rejected("all_legs_rejected", **refused)
                    return self._all_rejected(report)
                outcome.rejected("ticket_validation_failed", **refused)
                log.warning("ticket creation refused: %s", type(exc).__name__)
                return ToolOutput("The ticket could not be created (a leg failed validation); adjust the legs and try again.",
                                  is_error=True)
            created = True
            self._proposal_turns[slot] = ticket.id
        finally:
            if not created:
                self._proposal_turns.pop(slot, None)
        outcome.created(ticket)

        executable = [d for d in decisions if d.verdict != "reject"]
        buy_cost = sum((d.est_cost_usd for d in executable if d.leg.side == "buy" and d.est_cost_usd is not None), Decimal(0))
        sell_value = sum((d.est_cost_usd for d in executable if d.leg.side == "sell" and d.est_cost_usd is not None), Decimal(0))
        lines = [
            f"Ticket {ticket.id} proposed for the user's review: overall {overall(decisions)}; "
            f"{len(executable)} of {len(decisions)} leg(s) executable; expires {ticket.expires_at.isoformat()}; "
            f"mandate v{mandate.version}.",
            "Risk report:",
            *report,
            f"Estimated buy cost {_usd(buy_cost)}; estimated sell value {_usd(sell_value)}.",
            f"Evidence cited: {', '.join(evidence_ids) or 'none'}.",
        ]
        if getattr(self._settings, "trading_enabled", True) is False:
            lines.append("Note: trading is currently disabled (kill switch), so this ticket cannot be confirmed until the user re-enables it.")
        lines.append(_CLOSING_LINE)
        event = {"type": "data-trade-proposal", "ticket": ticket.model_dump(mode="json")}
        return ToolOutput("\n".join(lines), events=[event])

    @staticmethod
    def _all_rejected(report: list[str]) -> ToolOutput:
        lines = [
            "No ticket was created: every leg was rejected by the risk checks (overall: blocked).",
            "Risk report:",
            *report,
            "Adjust the legs (or suggest a mandate change) before proposing again. You cannot place orders.",
        ]
        return ToolOutput("\n".join(lines))

    @staticmethod
    def _parse_legs(raw: Any) -> list[OrderLeg]:
        if not isinstance(raw, list) or not raw:
            raise _ArgError(f"'legs' must be a non-empty array of order legs (at most {MAX_LEGS})")
        if len(raw) > MAX_LEGS:
            raise _ArgError(f"'legs' accepts at most {MAX_LEGS} legs")
        legs: list[OrderLeg] = []
        errors: list[str] = []
        for i, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                errors.append(f"leg {i}: must be an object")
                continue
            label = f"leg {i} ({_token(item.get('symbol'), 20) or '?'})"
            unknown = [k for k in item if k not in _LEG_FIELDS]
            if unknown:
                errors.append(f"{label}: unknown field(s) {', '.join(_token(k, 30) for k in unknown)}")
                continue
            fields: dict[str, Any] = {}
            try:
                for key, value in item.items():
                    if key in _LEG_MONEY_FIELDS:
                        fields[key] = parse_money(value, key)
                    elif key in ("side", "type", "time_in_force") and isinstance(value, str):
                        fields[key] = value.strip().lower()
                    else:
                        fields[key] = value
                legs.append(OrderLeg(**fields))
            except ValidationError as exc:
                errors.append(f"{label}: {_validation_message(exc)}")
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
        if errors:
            raise _ArgError("invalid order legs - " + "; ".join(errors))
        return legs

    def _evidence_records(self, session_id: str, raw: Any) -> list[SourceRecord]:
        """The cited evidence records (deduplicated, in order); every id must exist in this session."""
        if not isinstance(raw, list):
            raise _ArgError("'evidence_ids' must be an array of evidence ids such as [\"E1\", \"E4\"]")
        if len(raw) > MAX_EVIDENCE_IDS:
            raise _ArgError(f"'evidence_ids' accepts at most {MAX_EVIDENCE_IDS} ids")
        records: list[SourceRecord] = []
        unknown: list[str] = []
        for item in raw:
            eid = item.strip() if isinstance(item, str) else ""
            record = self._evidence.get(session_id, eid) if _EVIDENCE_ID_RE.fullmatch(eid) else None
            if record is None:
                label = eid if _EVIDENCE_ID_RE.fullmatch(eid) else repr(_token(item, 20))
                if label not in unknown:
                    unknown.append(label)
                continue
            if all(r.id != record.id for r in records):
                records.append(record)
        if unknown:
            raise _ArgError(
                f"unknown evidence id(s) for this session: {', '.join(unknown)}. Cite only [E#] ids returned by "
                "research tools in this session."
            )
        return records
