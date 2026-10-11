"""Local-only guard for the trading API: an ASGI middleware in front of `/api/*`.

`/api/*` can place paper orders and write keys into `.env`, so it must be unreachable from other
machines and from other websites open in the user's browser. For every HTTP request under `/api/`:

1. **Loopback only.** The client address must be loopback (`127.0.0.0/8`, `::1` or an IPv4-mapped
   loopback), unless `allow_remote()` is true (the `api_allow_remote` setting, unsafe without your
   own auth in front). A request with no client address is refused too. Otherwise 403
   `local_only`. The address is the socket peer: `main()` and the Docker image run uvicorn without
   proxy headers (`proxy_headers=False`, `--no-proxy-headers`), so an `X-Forwarded-For` header can
   never make a remote client look local.
2. **No proxied requests** (defence in depth for the rule above). A request carrying
   `X-Forwarded-For`, `Forwarded`, `X-Real-IP` or `X-Forwarded-Host` came through a proxy, or is
   forged: a local browser never sends them. 403 `forwarded_request`, unless `allow_remote()` is
   true (the loopback rule is off then anyway, and your own authenticating proxy adds them).
3. **Host allowlist** (DNS-rebinding defence). The `Host` header must parse strictly
   (`parse_host`: `name[:port]` or `[ipv6][:port]`, digits-only port) to `127.0.0.1`, `localhost`
   or `[::1]`, whatever `allow_remote()` says. Otherwise 400 `bad_host`.
4. **No cross-site browser requests**, for every method. A `Sec-Fetch-Site` header (browsers send
   it on every request) other than `same-origin` or `none` - so `cross-site` or `same-site` - is
   403 `cross_site`. No header at all is allowed: non-browser clients do not send it.
5. **Anti-CSRF** for POST, PUT, PATCH and DELETE: the header `X-FinAgent-Client: 1` is required (a
   cross-site page cannot send a custom header without a CORS preflight, and no CORS middleware is
   installed), else 403 `csrf_header_missing`. An `Origin` header, when present, must be
   `http://<Host>` or `https://<Host>`, else 403 `origin_mismatch`.
6. **Body cap.** A request body over `MAX_BODY_BYTES` (1 MiB) is 413 `too_large`: by its
   `Content-Length` before anything is read, or, without one (chunked), counted while it is read.
   The guard reads such a body itself, up to the cap, before the route sees it, and stops at the
   first chunk past the cap.

`/answer` and `/answer/stream` get rule 4 only (403 `cross_site`): another site's page must not be
able to spend the model budget through a visitor's browser. Graders' HTTP clients send no
`Sec-Fetch-Site`, so the competition contract is unchanged for them.

There are no test-only allowances: tests use a real loopback address and Host. WebSocket
connections under `/api` are refused outright (close code 1008): there are no WebSocket routes, and
one added later would otherwise run without the checks above.

Responses under `/api/`, `/` and `/app/` also get anti-framing and no-sniff headers, so a hostile
page cannot frame the app and trick the user into pressing Confirm. The web app itself (`/` and
`/app/`) is always sent with `Cache-Control: no-cache`: the browser may keep a copy but must
revalidate it, so it never runs a stale script after an update. A link from another site still
opens it (rule 4 is for `/api/*` and `/answer` only). `/health` and `/ui` pass through untouched,
and so do `/answer` and `/answer/stream` apart from rule 4. This is a pure ASGI middleware (not
`BaseHTTPMiddleware`) so streaming responses and cancellation behave exactly as without it.
"""
from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Callable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

API_PREFIX = "/api"
CLIENT_HEADER = "x-finagent-client"
ALLOWED_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Headers only a proxy adds: a local browser never sends them (rule 2).
FORWARDED_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "x-forwarded-host")
# `Sec-Fetch-Site` values let through (rule 4); a missing header is let through too.
SAME_SITE_FETCH = frozenset({"same-origin", "none"})
ANSWER_PATHS = frozenset({"/answer", "/answer/stream"})
MAX_BODY_BYTES = 1024 * 1024  # rule 6
_HOST_RE = re.compile(
    r"(?P<host>\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?)(?::(?P<port>[0-9]{1,5}))?"
)
_SECURITY_HEADERS = [
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"x-content-type-options", b"nosniff"),
]
_APP_CACHE_CONTROL = (b"cache-control", b"no-cache")
_POLICY_VIOLATION = 1008  # WebSocket close code


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
    try:
        address = ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


def parse_host(host_header: str) -> str | None:
    """The lower-cased hostname of a well-formed `Host` header (`[::1]:8000` -> `[::1]`), or None.
    Accepts exactly `name[:port]` or `[ipv6][:port]` with a numeric port of at most 65535."""
    match = _HOST_RE.fullmatch(host_header or "")
    if match is None:
        return None
    port = match.group("port")
    if port is not None and int(port) > 65535:
        return None
    host = match.group("host").lower()
    if host.startswith("["):
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ValueError:
            return None
    return host


def _is_api(scope: Scope) -> bool:
    return is_api_path(_route_path(scope)) or is_api_path(scope.get("path", ""))  # never under-match


def _is_answer(scope: Scope) -> bool:
    return _route_path(scope) in ANSWER_PATHS or scope.get("path", "") in ANSWER_PATHS


def _cross_site(headers: Headers) -> bool:
    site = headers.get("sec-fetch-site")
    return site is not None and site.strip().lower() not in SAME_SITE_FETCH


def _declared_length(headers: Headers) -> int | None:
    """The `Content-Length` as a number, or None when it is absent, not a plain number, or sent
    beside a `Transfer-Encoding` (the body is then counted as it is read). uvicorn's h11 parser
    frames such a body by its Transfer-Encoding, so a small declared length proves nothing."""
    if "transfer-encoding" in headers:
        return None
    value = headers.get("content-length")
    if value is None or not value.strip().isdigit():
        return None
    return int(value.strip())


async def _read_capped(receive: Receive, cap: int) -> tuple[list[Message], bool]:
    """Read a body of unknown length up to `cap` bytes: the messages read, and whether the body
    went past the cap (reading stops at the first chunk that does)."""
    messages: list[Message] = []
    total = 0
    while True:
        message = await receive()
        messages.append(message)
        if message.get("type") != "http.request":
            return messages, False  # a disconnect: the route will see it
        total += len(message.get("body", b""))
        if total > cap:
            return messages, True
        if not message.get("more_body", False):
            return messages, False


def _replay(messages: list[Message], receive: Receive) -> Receive:
    """`receive` for the route: the messages the guard has read already, then the real one (for
    a disconnect)."""
    pending = list(messages)

    async def replayed() -> Message:
        if pending:
            return pending.pop(0)
        return await receive()

    return replayed


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


_CROSS_SITE = _refusal(403, "cross_site", "This request came from another website and was refused.")
_TOO_LARGE = _refusal(413, "too_large", "The request body is too large.")


class LocalApiGuard:
    """See the module docstring. `allow_remote` is read on every request, so a settings change
    applies without a restart."""

    def __init__(self, app: ASGIApp, *, allow_remote: Callable[[], bool] = lambda: False):
        self.app = app
        self.allow_remote = allow_remote

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket" and _is_api(scope):
            await send({"type": "websocket.close", "code": _POLICY_VIOLATION})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        api = _is_api(scope)
        if not api and _is_answer(scope):
            if _cross_site(Headers(scope=scope)):
                await _CROSS_SITE(send)
                return
            await self.app(scope, receive, send)
            return
        if not api and not _is_app_path(_route_path(scope)):
            await self.app(scope, receive, send)
            return
        if api:
            refusal = self._check(scope)
            if refusal is not None:
                await refusal(send)
                return
            if _declared_length(Headers(scope=scope)) is None:
                messages, too_large = await _read_capped(receive, MAX_BODY_BYTES)
                if too_large:
                    await _TOO_LARGE(send)
                    return
                receive = _replay(messages, receive)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not api:  # the web app: every response, whatever it set itself
                    headers = [(n, v) for n, v in headers if n.lower() != _APP_CACHE_CONTROL[0]]
                    headers.append(_APP_CACHE_CONTROL)
                present = {name.lower() for name, _ in headers}
                headers.extend((n, v) for n, v in _SECURITY_HEADERS if n not in present)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _check(self, scope: Scope):
        client = scope.get("client")
        client_host = client[0] if client else None
        remote_allowed = self._remote_allowed()
        if not is_loopback(client_host) and not remote_allowed:
            return _refusal(403, "local_only", "This API only answers requests from this computer.")
        headers = Headers(scope=scope)
        if not remote_allowed and any(name in headers for name in FORWARDED_HEADERS):
            return _refusal(403, "forwarded_request",
                            "This API does not answer requests that came through a proxy.")
        host_header = headers.get("host", "")
        if parse_host(host_header) not in ALLOWED_HOSTNAMES:
            return _refusal(400, "bad_host", "Open the app at http://127.0.0.1 or http://localhost.")
        if _cross_site(headers):
            return _CROSS_SITE
        if scope.get("method", "GET").upper() in STATE_CHANGING:
            if headers.get(CLIENT_HEADER) != "1":
                return _refusal(403, "csrf_header_missing",
                                "This request must come from the FinAgent app itself.")
            origin = headers.get("origin")
            if origin is not None and not _origin_matches(origin, host_header):
                return _refusal(403, "origin_mismatch",
                                "This request came from another website and was refused.")
        declared = _declared_length(headers)
        if declared is not None and declared > MAX_BODY_BYTES:
            return _TOO_LARGE
        return None

    def _remote_allowed(self) -> bool:
        try:
            return bool(self.allow_remote())
        except Exception:  # noqa: BLE001 - fail closed
            return False
