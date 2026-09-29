"""Local-only guard for the trading API: an ASGI middleware in front of `/api/*`.

`/api/*` can place paper orders and write keys into `.env`, so it must be unreachable from other
machines and from other websites open in the user's browser. For every HTTP request under `/api/`:

1. **Loopback only.** The client address must be loopback (`127.0.0.0/8`, `::1`, an IPv4-mapped
   loopback, or Starlette's `testclient`), unless `allow_remote()` is true (the `api_allow_remote`
   setting, unsafe without your own auth in front). Otherwise 403 `local_only`.
2. **Host allowlist** (DNS-rebinding defence). The `Host` header's hostname must be `127.0.0.1`,
   `localhost`, `[::1]` or `testserver`, whatever `allow_remote()` says. Otherwise 400 `bad_host`.
3. **Anti-CSRF** for POST, PUT, PATCH and DELETE: the header `X-FinAgent-Client: 1` is required (a
   cross-site page cannot send a custom header without a CORS preflight, and no CORS middleware is
   installed), else 403 `csrf_header_missing`. An `Origin` header, when present, must be
   `http://<Host>` or `https://<Host>`, else 403 `origin_mismatch`.

Responses under `/api/`, `/` and `/app/` also get anti-framing and no-sniff headers, so a hostile
page cannot frame the app and trick the user into pressing Confirm. `/answer`, `/answer/stream`,
`/health` and `/ui` pass through untouched. This is a pure ASGI middleware (not `BaseHTTPMiddleware`)
so streaming responses and cancellation behave exactly as without it.
"""
from __future__ import annotations

import ipaddress
import json
from collections.abc import Callable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

API_PREFIX = "/api"
CLIENT_HEADER = "x-finagent-client"
ALLOWED_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "[::1]", "testserver"})
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_TEST_CLIENT_HOST = "testclient"  # Starlette's TestClient; real servers always report an IP
_SECURITY_HEADERS = [
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"x-content-type-options", b"nosniff"),
]


def is_api_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


def _route_path(scope: Scope) -> str:
    """The path the router matches on: `path` minus a `root_path` prefix (as Starlette does)."""
    path: str = scope.get("path", "")
    root = scope.get("root_path", "")
    if root and path.startswith(root) and len(path) > len(root) and path[len(root)] == "/":
        return path[len(root):]
    return path


def _is_app_path(path: str) -> bool:
    return path == "/" or path == "/app" or path.startswith("/app/")


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == _TEST_CLIENT_HOST:
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


def host_name(host_header: str) -> str:
    """The hostname part of a `Host` header, lower-cased (`[::1]:8000` -> `[::1]`)."""
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":", 1)[0]


def _origin_matches(origin: str, host_header: str) -> bool:
    origin = origin.strip().lower().rstrip("/")
    host = host_header.strip().lower()
    return origin in (f"http://{host}", f"https://{host}")


def _refusal(status: int, code: str, message: str):
    body = json.dumps({"error": code, "message": message}).encode("utf-8")

    async def respond(send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *_SECURITY_HEADERS,
            ],
        })
        await send({"type": "http.response.body", "body": body})

    return respond


class LocalApiGuard:
    """See the module docstring. `allow_remote` is read on every request, so a settings change
    applies without a restart."""

    def __init__(self, app: ASGIApp, *, allow_remote: Callable[[], bool] = lambda: False):
        self.app = app
        self.allow_remote = allow_remote

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw, path = scope.get("path", ""), _route_path(scope)
        api = is_api_path(path) or is_api_path(raw)  # guard both spellings; never under-match
        if not api and not _is_app_path(path):
            await self.app(scope, receive, send)
            return
        if api:
            refusal = self._check(scope)
            if refusal is not None:
                await refusal(send)
                return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {name.lower() for name, _ in message.get("headers", [])}
                extra = [(n, v) for n, v in _SECURITY_HEADERS if n not in present]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _check(self, scope: Scope):
        client = scope.get("client")
        client_host = client[0] if client else None
        if not is_loopback(client_host) and not self._remote_allowed():
            return _refusal(403, "local_only", "This API only answers requests from this computer.")
        headers = Headers(scope=scope)
        host_header = headers.get("host", "")
        if host_name(host_header) not in ALLOWED_HOSTNAMES:
            return _refusal(400, "bad_host", "Open the app at http://127.0.0.1 or http://localhost.")
        if scope.get("method", "GET").upper() in STATE_CHANGING:
            if headers.get(CLIENT_HEADER) != "1":
                return _refusal(403, "csrf_header_missing",
                                "This request must come from the FinAgent app itself.")
            origin = headers.get("origin")
            if origin is not None and not _origin_matches(origin, host_header):
                return _refusal(403, "origin_mismatch",
                                "This request came from another website and was refused.")
        return None

    def _remote_allowed(self) -> bool:
        try:
            return bool(self.allow_remote())
        except Exception:  # noqa: BLE001 - fail closed
            return False
