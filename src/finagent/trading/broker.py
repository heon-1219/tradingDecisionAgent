"""Alpaca paper-trading broker client.

Hard guard, per the project's paper-only constraint: the trading host is always
`paper-api.alpaca.markets`. No setting, environment variable or argument can change it - `__init__`
asserts the host of any injected `httpx.AsyncClient` and refuses (`ValueError`) anything else, so
even a test or a future caller cannot point this class at a live account by accident.

No retries anywhere here, including reads: unlike `research.http.HttpClient` (which retries GETs
against research connectors), this client is one attempt per call, errors mapped and raised. That
keeps every method's behavior uniform and easy to reason about - and it is required for
`submit_order`, since the project forbids blind order retries (a resubmitted order could double-
fill). A transport error, timeout, or 5xx on `submit_order` raises `OrderUncertain`: the order may
have reached Alpaca even though this call didn't get a confirmed response, so callers must
reconcile via `get_order_by_client_id` rather than resubmit. Every other 4xx/5xx (including 422
"client_order_id must be unique", which submit_order also uses this same 4xx path for) raises the
plain `BrokerError`.

Every request carries `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers built once in `__init__` and
passed explicitly on each call (never relying on a client's own default headers); logging here
never includes headers or request/response bodies containing the secret, only method+path.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import httpx

from .models import (
    AccountSnapshot,
    AssetInfo,
    ClockInfo,
    OrderLeg,
    OrderSnapshot,
    PositionSnapshot,
)

logger = logging.getLogger(__name__)

PAPER_BASE = "https://paper-api.alpaca.markets"
_PAPER_HOST = "paper-api.alpaca.markets"


class BrokerError(Exception):
    """Raised for any 4xx/5xx Alpaca response outside `submit_order`'s uncertain-outcome cases."""

    def __init__(self, status: int, code: int | None, message: str):
        super().__init__(f"Alpaca broker error {status}: {message}")
        self.status = status
        self.code = code
        self.message = message


class OrderUncertain(BrokerError):
    """Raised only by `submit_order`, only for a timeout/transport error or a 5xx: the order MAY
    have reached Alpaca despite this call not getting a confirmed response. Callers must reconcile
    with `get_order_by_client_id`, never resubmit blindly. `status` is 0 for a transport error
    (no response was ever received to have a status)."""


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


class AlpacaPaper:
    def __init__(self, key_id: str, secret: str, client: httpx.AsyncClient | None = None):
        if not key_id or not secret:
            raise ValueError("Alpaca key_id and secret are both required.")
        if client is not None:
            host = client.base_url.host
            if host != _PAPER_HOST:
                raise ValueError(
                    f"AlpacaPaper only talks to {_PAPER_HOST!r} (paper trading); got a client "
                    f"whose base_url host is {host!r}."
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
        try:
            resp = await self._client.request(
                method, path, params=params, json=json, headers=self._headers
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if uncertain_on_failure:
                raise OrderUncertain(0, None, message) from None
            raise BrokerError(0, None, message) from None

        if resp.status_code >= 500:
            code, message = _error_details(resp)
            if uncertain_on_failure:
                raise OrderUncertain(resp.status_code, code, message)
            raise BrokerError(resp.status_code, code, message)
        if resp.status_code >= 400:
            code, message = _error_details(resp)
            raise BrokerError(resp.status_code, code, message)
        return resp

    # --- reads --------------------------------------------------------------------

    async def account(self) -> AccountSnapshot:
        resp = await self._request("GET", "/v2/account")
        return AccountSnapshot.model_validate(resp.json())

    async def positions(self) -> dict[str, PositionSnapshot]:
        resp = await self._request("GET", "/v2/positions")
        items = resp.json() or []
        snapshots = [PositionSnapshot.model_validate(item) for item in items]
        return {s.symbol: s for s in snapshots}

    async def orders(
        self, status: str = "open", limit: int = 50, symbols: list[str] | None = None
    ) -> list[OrderSnapshot]:
        params: dict[str, Any] = {"status": status, "limit": limit, "nested": "false"}
        if symbols:
            params["symbols"] = ",".join(symbols)
        resp = await self._request("GET", "/v2/orders", params=params)
        return [OrderSnapshot.model_validate(o) for o in (resp.json() or [])]

    async def asset(self, symbol: str) -> AssetInfo:
        resp = await self._request("GET", f"/v2/assets/{symbol}")
        return AssetInfo.model_validate(resp.json())

    async def clock(self) -> ClockInfo:
        resp = await self._request("GET", "/v2/clock")
        return ClockInfo.model_validate(resp.json())

    async def get_order(self, order_id: str) -> OrderSnapshot:
        resp = await self._request("GET", f"/v2/orders/{order_id}")
        return OrderSnapshot.model_validate(resp.json())

    async def get_order_by_client_id(self, client_order_id: str) -> OrderSnapshot | None:
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
        return OrderSnapshot.model_validate(resp.json())

    # --- writes -------------------------------------------------------------------

    async def submit_order(self, leg: OrderLeg, client_order_id: str) -> OrderSnapshot:
        """Exactly one POST /v2/orders. See the module docstring for the error-mapping contract:
        a timeout/transport error/5xx raises OrderUncertain, everything else (including 422
        duplicate client_order_id, 403 insufficient buying power, ...) raises BrokerError."""
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
        if leg.type == "limit":
            if leg.limit_price is None:
                raise ValueError("limit_price is required for a limit order.")
            body["limit_price"] = _decimal_str(leg.limit_price)

        resp = await self._request("POST", "/v2/orders", json=body, uncertain_on_failure=True)
        return OrderSnapshot.model_validate(resp.json())

    async def cancel_order(self, order_id: str) -> None:
        await self._request("DELETE", f"/v2/orders/{order_id}")

    async def set_suspend_trade(self, suspended: bool) -> None:
        await self._request(
            "PATCH", "/v2/account/configurations", json={"suspend_trade": suspended}
        )
