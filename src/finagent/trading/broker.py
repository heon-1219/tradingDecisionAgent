"""Alpaca paper-trading broker client.

Hard guard, per the project's paper-only constraint: the trading host is always
`paper-api.alpaca.markets` over plain `https` on the default port. No setting, environment
variable or argument can change it. This is enforced twice, not once:

- `__init__` checks the *full origin* (scheme, host, port - not just `.host`) of any injected
  `httpx.AsyncClient`, so `http://paper-api.alpaca.markets` (secret sent over plaintext) or
  `https://paper-api.alpaca.markets:8080` is refused just as a wrong host would be.
- `_request` re-checks the origin of every outgoing request, built explicitly via
  `self._client.build_request(...)` and inspected *before* `send()`. This also catches a client
  whose `base_url` was mutated after construction (`b._client.base_url = "https://api.alpaca..."`)
  - a one-time constructor check alone cannot, since `base_url` is a plain mutable attribute.
  `_request` also always sends with `follow_redirects=False`, overriding whatever the injected
  client's own default is: an ordinary 3xx is surfaced as an error response (see below) rather
  than auto-followed.
- `__init__` additionally refuses an injected client that has a custom `auth` handler or any
  `request`/`response` event hook configured. Both run *during* `send()`, after the origin check
  above has already passed on the request as originally built, and either can rewrite the
  request's URL to a different host before it's transmitted; a custom `auth` can go further and
  re-send an entirely new request in response to the very redirect this class is trying to treat
  as an error (httpx's `Auth.auth_flow` is a generator that may `yield` a second request after
  inspecting the first response - independent of `follow_redirects`). What this class does *not*
  and cannot defend against is an injected client using a custom low-level `Transport`: the
  transport is the actual network boundary being trusted to send exactly the request it's given,
  the same way any HTTP client trusts its transport - callers must not inject an untrusted one.

No retries anywhere here, including reads: unlike `research.http.HttpClient` (which retries GETs
against research connectors), this client is one attempt per call, errors mapped and raised. That
keeps every method's behavior uniform and easy to reason about - and it is required for
`submit_order`, since the project forbids blind order retries (a resubmitted order could double-
fill).

Response handling never lets an unparseable "success" (or a request that failed in a way that
isn't a clean HTTP error status) escape as a raw exception (`httpx.RequestError` - which includes
`TimeoutException`, `TransportError`, `TooManyRedirects` *and* `DecodingError`, e.g. a mislabeled
`Content-Encoding` - `json.JSONDecodeError`, `RecursionError` on pathologically deep JSON, or a
pydantic `ValidationError`) - every path is mapped to `BrokerError`/`OrderUncertain`, so the
confirmation flow can always reconcile instead of crashing:

- A request error (timeout/transport/decoding failure), a 3xx (redirect), or a 5xx on
  `submit_order` raises `OrderUncertain`: the order may have reached Alpaca even though this call
  didn't get a confirmed, parseable response, so callers must reconcile via
  `get_order_by_client_id` rather than resubmit. So does a 2xx whose parsed `client_order_id`
  doesn't match the one just submitted - Alpaca returning *some* order snapshot that isn't
  provably the one this call submitted is exactly the "reconcile, don't trust it blindly" case.
- The same conditions on every other method raise `BrokerError` (nothing was submitted, so there
  is nothing ambiguous to reconcile).
- Every 4xx (including 422 "client_order_id must be unique" on `submit_order`) raises
  `BrokerError`, never `OrderUncertain` - Alpaca rejected the request outright.
- A 2xx response whose body isn't valid JSON, is JSON but fails to validate against the expected
  snapshot model, or (for the list endpoints `positions`/`orders`) is validly-shaped JSON that
  just isn't a list, is treated exactly like a 5xx: `OrderUncertain` from `submit_order`,
  `BrokerError` everywhere else.

Every request carries `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers built once in `__init__` and
passed explicitly on each call (never relying on a client's own default headers); logging here
never includes headers or request/response bodies containing the secret, only method+path. Every
value that becomes part of a URL path segment (`order_id`, `symbol`) or a query/body value
(`client_order_id`) is checked with `re.fullmatch` (not `re.match`, which would let a trailing
`"\n"` slip through a `$`-anchored pattern) against a strict allowlist *before* any request is
attempted, and a path segment is additionally percent-encoded - defense in depth against path
injection. The symbol pattern also requires a leading alphanumeric character: disallowing `"/"` is
not by itself enough to stop a `"."`/`".."` segment from being path-injected, since those resolve
via ordinary URL dot-segment normalization (`/v2/assets/.` -> `/v2/assets`, `/v2/assets/..` ->
`/v2`) with no `/` of their own needed in the *symbol* string.
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from .models import (
    AccountSnapshot,
    AssetInfo,
    ClockInfo,
    OrderLeg,
    OrderSnapshot,
    PositionSnapshot,
    normalize_symbol,
)

logger = logging.getLogger(__name__)

PAPER_BASE = "https://paper-api.alpaca.markets"
_PAPER_HOST = "paper-api.alpaca.markets"
_PAPER_SCHEME = "https"
_PAPER_PORTS = (None, 443)  # None = default https port (not written out in the URL)

# Alpaca order ids are UUIDs (canonical, dashed, lower/upper hex); client_order_id is our own
# caller-chosen token. Both are validated (via re.fullmatch - see _validate_order_id/
# _validate_client_order_id) before ever building a URL/query with them. NOTE: these patterns are
# matched with fullmatch, not match/search, so the trailing "$" cannot be satisfied by stopping
# just before a trailing "\n" the way plain re.match(pattern + "$") would allow.
_ORDER_ID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CLIENT_ORDER_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class BrokerError(Exception):
    """Raised for any 4xx/5xx/3xx Alpaca response, transport failure, or unparseable "success"
    body, outside `submit_order`'s uncertain-outcome cases."""

    def __init__(self, status: int, code: int | None, message: str):
        super().__init__(f"Alpaca broker error {status}: {message}")
        self.status = status
        self.code = code
        self.message = message


class OrderUncertain(BrokerError):
    """Raised only by `submit_order`, for anything that isn't a clean, validated 2xx: a timeout/
    transport error, a 3xx, a 5xx, an unparseable body, or a body that fails snapshot validation.
    In every one of those cases the order MAY have reached Alpaca despite this call not getting a
    usable response. Callers must reconcile with `get_order_by_client_id`, never resubmit blindly.
    `status` is 0 for a transport error (no response was ever received to have a status)."""


def _decimal_str(d: Decimal) -> str:
    """Serialize a Decimal the way Alpaca expects: plain digits, never exponent notation
    (`str(Decimal("1E+2"))` would be "1E+2", which Alpaca's API rejects)."""
    return format(d, "f")


def _body_excerpt(text: str, limit: int = 200) -> str:
    return " ".join((text or "").split())[:limit]


def _error_details(resp: httpx.Response) -> tuple[int | None, str]:
    """Pull (code, message) out of an Alpaca error body; body/text only, never headers."""
    try:
        data = resp.json()
    except ValueError:
        return None, _body_excerpt(resp.text)
    if isinstance(data, dict):
        code = data.get("code")
        message = data.get("message")
        if message:
            return code, str(message)
    return None, _body_excerpt(resp.text)


def _validate_order_id(order_id: str) -> str:
    if not isinstance(order_id, str) or not _ORDER_ID_RE.fullmatch(order_id):
        raise ValueError(f"invalid order_id (must be a UUID): {order_id!r}")
    return order_id


def _validate_client_order_id(client_order_id: str) -> str:
    if not isinstance(client_order_id, str) or not _CLIENT_ORDER_ID_RE.fullmatch(client_order_id):
        raise ValueError(f"invalid client_order_id: {client_order_id!r}")
    return client_order_id


def _is_paper_origin(url: httpx.URL) -> bool:
    return url.scheme == _PAPER_SCHEME and url.host == _PAPER_HOST and url.port in _PAPER_PORTS


class AlpacaPaper:
    def __init__(self, key_id: str, secret: str, client: httpx.AsyncClient | None = None):
        if not key_id or not secret:
            raise ValueError("Alpaca key_id and secret are both required.")
        if client is not None:
            if not _is_paper_origin(client.base_url):
                raise ValueError(
                    f"AlpacaPaper only talks to {PAPER_BASE!r} (paper trading, https, default "
                    f"port); got a client whose base_url is {str(client.base_url)!r}."
                )
            if client.auth is not None:
                raise ValueError(
                    "AlpacaPaper refuses a client with a custom auth handler: it runs during "
                    "send(), after this class's origin check has already passed, and could "
                    "rewrite the request's URL or re-send it to a different host."
                )
            hooks = client.event_hooks or {}
            if hooks.get("request") or hooks.get("response"):
                raise ValueError(
                    "AlpacaPaper refuses a client with request/response event hooks: they run "
                    "during send(), after this class's origin check has already passed, and "
                    "could rewrite the request's URL."
                )
            self._client = client
        else:
            self._client = httpx.AsyncClient(base_url=PAPER_BASE, timeout=httpx.Timeout(15.0))
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
        uncertain_on_failure: bool = False,
    ) -> httpx.Response:
        logger.debug("%s %s", method, path)
        exc_cls = OrderUncertain if uncertain_on_failure else BrokerError

        request = self._client.build_request(method, path, params=params, json=json, headers=self._headers)
        if not _is_paper_origin(request.url):
            # Catches a base_url mutated after construction (or any other way a built request
            # could end up pointed somewhere else): re-checked on every call, not just at
            # __init__, since base_url is a plain mutable attribute on the client.
            raise ValueError(
                f"Refusing to send a trading request to a non-paper origin: {request.url!r}"
            )
        try:
            # follow_redirects=False overrides the client's own default: a 3xx must never be
            # auto-followed (which could send the secret headers to whatever host Location
            # points at) - it is surfaced below as an ordinary error response instead.
            resp = await self._client.send(request, follow_redirects=False)
        except httpx.RequestError as exc:
            # httpx.RequestError is the common base of TimeoutException, TransportError,
            # TooManyRedirects AND DecodingError (a mislabeled/corrupt Content-Encoding raises
            # DecodingError while send() reads the body - it is a RequestError, not a
            # TransportError, so catching only the latter two would miss it).
            message = f"{type(exc).__name__}: {exc}"
            raise exc_cls(0, None, message) from None

        if resp.status_code >= 500 or 300 <= resp.status_code < 400:
            code, message = _error_details(resp)
            raise exc_cls(resp.status_code, code, message)
        if resp.status_code >= 400:
            # Always BrokerError, even from submit_order: Alpaca rejected the request outright
            # (e.g. 422 duplicate client_order_id, 403 insufficient buying power), so this is not
            # an ambiguous "may have reached Alpaca" case.
            code, message = _error_details(resp)
            raise BrokerError(resp.status_code, code, message)
        return resp

    def _parse_json(self, resp: httpx.Response, *, uncertain_on_failure: bool) -> Any:
        try:
            return resp.json()
        except (ValueError, RecursionError):
            # ValueError: json.JSONDecodeError (malformed JSON). RecursionError: pathologically
            # deep nesting can blow the interpreter's recursion limit while decoding - still an
            # "unusable body", not a crash we want escaping this method.
            exc_cls = OrderUncertain if uncertain_on_failure else BrokerError
            message = f"{_body_excerpt(resp.text)} (invalid JSON)"
            raise exc_cls(resp.status_code, None, message) from None

    def _validate_model(
        self, model: type[BaseModel], data: Any, resp: httpx.Response, *, uncertain_on_failure: bool
    ) -> Any:
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            exc_cls = OrderUncertain if uncertain_on_failure else BrokerError
            message = f"{_body_excerpt(str(exc), 300)} (response failed validation)"
            raise exc_cls(resp.status_code, None, message) from None

    def _expect_list(self, data: Any, resp: httpx.Response, *, uncertain_on_failure: bool) -> list:
        """positions()/orders() require a JSON array; a `None` body (Alpaca sending literal
        `null`) is treated as empty, but any other non-list shape (a scalar, a string, or a dict -
        all of which are silently iterable in Python and would otherwise produce a raw TypeError
        or iterate over the wrong thing, e.g. a dict's keys) is an error."""
        if data is None:
            return []
        if not isinstance(data, list):
            exc_cls = OrderUncertain if uncertain_on_failure else BrokerError
            message = f"expected a JSON array, got {type(data).__name__}"
            raise exc_cls(resp.status_code, None, message)
        return data

    # --- reads --------------------------------------------------------------------

    async def account(self) -> AccountSnapshot:
        resp = await self._request("GET", "/v2/account")
        data = self._parse_json(resp, uncertain_on_failure=False)
        return self._validate_model(AccountSnapshot, data, resp, uncertain_on_failure=False)

    async def positions(self) -> dict[str, PositionSnapshot]:
        resp = await self._request("GET", "/v2/positions")
        data = self._parse_json(resp, uncertain_on_failure=False)
        items = self._expect_list(data, resp, uncertain_on_failure=False)
        snapshots = [
            self._validate_model(PositionSnapshot, item, resp, uncertain_on_failure=False)
            for item in items
        ]
        return {s.symbol: s for s in snapshots}

    async def orders(
        self, status: str = "open", limit: int = 50, symbols: list[str] | None = None
    ) -> list[OrderSnapshot]:
        params: dict[str, Any] = {"status": status, "limit": limit, "nested": "false"}
        if symbols:
            params["symbols"] = ",".join(symbols)
        resp = await self._request("GET", "/v2/orders", params=params)
        data = self._parse_json(resp, uncertain_on_failure=False)
        items = self._expect_list(data, resp, uncertain_on_failure=False)
        return [
            self._validate_model(OrderSnapshot, o, resp, uncertain_on_failure=False)
            for o in items
        ]

    async def asset(self, symbol: str) -> AssetInfo:
        symbol = normalize_symbol(symbol)
        resp = await self._request("GET", f"/v2/assets/{quote(symbol, safe='')}")
        data = self._parse_json(resp, uncertain_on_failure=False)
        return self._validate_model(AssetInfo, data, resp, uncertain_on_failure=False)

    async def clock(self) -> ClockInfo:
        resp = await self._request("GET", "/v2/clock")
        data = self._parse_json(resp, uncertain_on_failure=False)
        return self._validate_model(ClockInfo, data, resp, uncertain_on_failure=False)

    async def get_order(self, order_id: str) -> OrderSnapshot:
        order_id = _validate_order_id(order_id)
        resp = await self._request("GET", f"/v2/orders/{quote(order_id, safe='')}")
        data = self._parse_json(resp, uncertain_on_failure=False)
        return self._validate_model(OrderSnapshot, data, resp, uncertain_on_failure=False)

    async def get_order_by_client_id(self, client_order_id: str) -> OrderSnapshot | None:
        client_order_id = _validate_client_order_id(client_order_id)
        try:
            resp = await self._request(
                "GET",
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
        except BrokerError as exc:
            if exc.status == 404:
                return None
            raise
        data = self._parse_json(resp, uncertain_on_failure=False)
        return self._validate_model(OrderSnapshot, data, resp, uncertain_on_failure=False)

    # --- writes -------------------------------------------------------------------

    async def submit_order(self, leg: OrderLeg, client_order_id: str) -> OrderSnapshot:
        """Exactly one POST /v2/orders. See the module docstring for the error-mapping contract:
        a timeout/transport error/3xx/5xx/unparseable-or-invalid 2xx body raises OrderUncertain;
        every 4xx (including 422 duplicate client_order_id, 403 insufficient buying power, ...)
        raises BrokerError."""
        client_order_id = _validate_client_order_id(client_order_id)
        body: dict[str, Any] = {
            "symbol": leg.symbol,
            "side": leg.side,
            "type": leg.type,
            "time_in_force": leg.time_in_force,
            "client_order_id": client_order_id,
        }
        if leg.qty is not None:
            body["qty"] = _decimal_str(leg.qty)
        else:
            assert leg.notional is not None  # OrderLeg guarantees exactly one is set
            body["notional"] = _decimal_str(leg.notional)
        if leg.limit_price is not None:  # OrderLeg guarantees this iff type == "limit"
            body["limit_price"] = _decimal_str(leg.limit_price)

        resp = await self._request("POST", "/v2/orders", json=body, uncertain_on_failure=True)
        data = self._parse_json(resp, uncertain_on_failure=True)
        order = self._validate_model(OrderSnapshot, data, resp, uncertain_on_failure=True)
        if order.client_order_id != client_order_id:
            # A 2xx body that doesn't echo back the client_order_id we just sent isn't provably
            # the order this call submitted - treat it exactly like any other uncertain outcome
            # rather than trusting it: reconcile via get_order_by_client_id.
            raise OrderUncertain(
                resp.status_code,
                None,
                f"Alpaca returned client_order_id {order.client_order_id!r}, expected "
                f"{client_order_id!r}; reconcile via get_order_by_client_id.",
            )
        return order

    async def cancel_order(self, order_id: str) -> None:
        order_id = _validate_order_id(order_id)
        await self._request("DELETE", f"/v2/orders/{quote(order_id, safe='')}")

    async def set_suspend_trade(self, suspended: bool) -> None:
        await self._request(
            "PATCH", "/v2/account/configurations", json={"suspend_trade": suspended}
        )
