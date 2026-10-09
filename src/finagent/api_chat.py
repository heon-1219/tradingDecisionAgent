"""`/api/*`: the chat and paper-trading API behind the web app.

Sessions, the chat stream (SSE), portfolio, mandate, tickets (confirm / reject / refresh / poll),
settings, the kill switch and the ledger export. `api_security.LocalApiGuard` keeps all of it
local-only (loopback clients, a Host allowlist, and `X-FinAgent-Client: 1` on every POST, PUT,
PATCH and DELETE). The model provider settings and `POST /api/llm/models` live in `api_llm`, whose
router is included here; the shared pieces (JSON, the error envelope, the guarded route class) are
in `api_common`.

**Only `POST /api/tickets/{id}/confirm` can place an order**: it is the one route that calls
`Executor.confirm`, and nothing else here reaches `broker.submit_order`. Confirm, reject, refresh and
the kill switch run under `ChatRuntime.lock` (serialised against a settings rebuild) inside a
shielded task, so a client that disconnects mid-confirm cannot interrupt a submission. Every ticket
action first calls `TicketStore.expire_stale()`.

**Ticket results** are `{status, message, ticket, orders, new_ticket}` with this HTTP mapping:
submitted 200; requote 409 on confirm (the new ticket is in `new_ticket`) but 200 on refresh (that
is what refresh is for); disabled 423; expired 410; not_found 404; rejected 200 for a ticket the
user has rejected (reject is idempotent), 410 for an expired ticket on confirm and reject
(expire_stale runs first, so the executor reports it as rejected), else 409 (hash mismatch, wrong
state, nothing executable). Refresh re-quotes the chat's newest ticket even after it expired (200
requote), so a refused refresh of an expired ticket is 409: a newer ticket exists;
error 502. No broker: confirm and refresh answer 503 `broker_not_connected`; reject still works (it
is a local state change, `execution.reject_ticket`), so a card can be dismissed after the keys are
cleared.

**Errors** are `{"error": "<code>", "message": "<friendly English>"}`, never exception text (see
`api_common`). Without a chat runtime every route answers 503 `chat_unavailable`.

**Paper only.** A broker whose account is not a paper account is never kept (`ChatRuntime`):
`GET /api/settings` then says `alpaca: "not_paper"`, such keys are refused by `PUT /api/settings`
with 400 `alpaca_not_paper`, and `GET /api/portfolio` reports `paper` from the account itself
(`true`, or `null` when Alpaca's response did not say).

**Chat stream.** `POST /api/chat` answers 404 for an unknown session before streaming. The turn
runs as a task; its events arrive through `emit` -> `queue.put_nowait` and go out as
`data: <json>\\n\\n`, with a `: keep-alive` comment after 10 s of silence. The stream ends after
`finish`. If the client goes away, the generator is cancelled or closed and cancels the turn.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, StrictBool, ValidationError

from . import api_llm
from .api_common import ApiError, Runtime, dumps, respond, runtime  # noqa: F401 - re-exported
from .api_common import Body as _Body
from .api_common import GuardedRoute as _GuardedRoute
from .api_common import error_response as _error
from .api_common import field_errors as _field_errors
from .config import env_fallbacks
from .envfile import EnvFileError, check_env_value
from .llm_admin import LLMCheckFailed
from .trading.execution import ConfirmResult, TrackedOrder, reject_ticket, stored_orders
from .trading.models import Mandate, TicketStatus
from .trading.proposals import TicketStateError

log = logging.getLogger("finagent.api_chat")

KEEPALIVE_SEC = 10.0
ID_MAX = 64
MAX_CHAT_CHARS = 16_000  # the agent refuses more than 8,000 with a friendly, localised message
MANDATE_MAX_BYTES = 32_768
_STREAM_FAILED = "Sorry, something went wrong while writing this reply. Please try again in a moment."
_KEEPALIVE = ": keep-alive\n\n"


SessionId = Annotated[str, PathParam(min_length=1, max_length=ID_MAX)]
TicketId = Annotated[str, PathParam(min_length=1, max_length=ID_MAX)]

router = APIRouter(prefix="/api", route_class=_GuardedRoute, dependencies=[Depends(runtime)])
router.include_router(api_llm.router)  # POST /api/llm/models


# --- request models -------------------------------------------------------------------------------


class SessionCreate(_Body):
    title: str | None = Field(default=None, max_length=200)
    lang: Literal["en", "ko"] | None = None


class SessionRename(_Body):
    title: str = Field(max_length=200)


class ChatRequest(_Body):
    session_id: str = Field(min_length=1, max_length=ID_MAX)
    message: str = Field(max_length=MAX_CHAT_CHARS)


class TicketAction(_Body):
    session_id: str = Field(min_length=1, max_length=ID_MAX)


class ConfirmRequest(TicketAction):
    hash: str = Field(min_length=1, max_length=128)


class KillSwitchRequest(_Body):
    suspend_broker: StrictBool = False


class SettingsUpdate(_Body):
    alpaca_key_id: str | None = Field(default=None, max_length=256)
    alpaca_secret: str | None = Field(default=None, max_length=256)
    tavily_key: str | None = Field(default=None, max_length=256)
    sec_user_agent: str | None = Field(default=None, max_length=256)
    trading_enabled: StrictBool | None = None
    # the model provider (rules and checks: api_llm)
    llm_provider: api_llm.Provider | None = None
    llm_model: str | None = Field(default=None, max_length=256)
    llm_effort: api_llm.Effort | None = None
    gemini_key: str | None = Field(default=None, max_length=256)
    openai_key: str | None = Field(default=None, max_length=256)
    anthropic_key: str | None = Field(default=None, max_length=256)
    gemini_paid: StrictBool | None = None


# Request field -> (settings field, label for messages).
_KEY_FIELDS = {
    "alpaca_key_id": ("alpaca_api_key_id", "Alpaca key ID"),
    "alpaca_secret": ("alpaca_api_secret_key", "Alpaca secret key"),
    "tavily_key": ("tavily_api_key", "Tavily API key"),
    "sec_user_agent": ("sec_user_agent", "SEC user agent"),
    **api_llm.LLM_KEY_FIELDS,
}
# Request fields copied to the settings field of the same name, as they are.
_PLAIN_FIELDS = frozenset({"trading_enabled"}) | api_llm.LLM_PLAIN_FIELDS


# --- helpers -----------------------------------------------------------------------------------------


def _session_or_404(rt, session_id: str) -> dict:
    session = rt.sessions.get(session_id)
    if session is None:
        raise ApiError(404, "session_not_found", "This chat session was not found.")
    return session


def _dump(model: BaseModel | None) -> dict | None:
    return model.model_dump(mode="json") if model is not None else None


def _order_json(row: TrackedOrder) -> dict:
    return {"client_order_id": row.client_order_id, "broker_order_id": row.broker_order_id,
            "order": _dump(row.order)}


def _local_orders(rt, ticket_id: str, session_id: str):
    """`Executor.ticket_orders` without a broker: the stored rows, not refreshed."""
    try:
        ticket = rt.tickets.get(ticket_id)
    except TicketStateError:
        return None
    if ticket is None or ticket.session_id != session_id:
        return None
    return ticket, stored_orders(rt.db, ticket.id)


def status_for(action: str, result: ConfirmResult) -> int:
    """HTTP status for an executor result; see the module docstring."""
    ticket_status = result.ticket.status if result.ticket is not None else None
    status = result.status
    if status == "submitted":
        return 200
    if status == "requote":
        return 200 if action == "refresh" else 409
    if status == "disabled":
        return 423
    if status == "expired":
        return 410
    if status == "not_found":
        return 404
    if status == "rejected":
        if action == "reject" and ticket_status == TicketStatus.rejected_by_user:
            return 200
        if ticket_status == TicketStatus.expired and action != "refresh":
            return 410
        return 409
    return 502


def _consume(task: asyncio.Future) -> None:
    """Done-callback: retrieve a background task's outcome so nothing is logged with its text."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("background task failed: %s", type(exc).__name__)


async def _shielded(coro: Awaitable[Any]) -> Any:
    """Run `coro` to completion even if the request is cancelled (client gone)."""
    task = asyncio.ensure_future(coro)
    task.add_done_callback(_consume)
    return await asyncio.shield(task)


# --- the chat stream ------------------------------------------------------------------------------


def _sse(event: Any) -> str:
    try:
        body = dumps(event)
    except (TypeError, ValueError, RecursionError) as exc:  # e.g. a circular structure
        log.warning("chat event not serializable (%s)", type(exc).__name__)
        kind = event.get("type") if isinstance(event, dict) else None
        body = dumps({"type": kind if isinstance(kind, str) else "error"})
    return "data: " + body + "\n\n"


def _is_finish(event: Any) -> bool:
    return isinstance(event, dict) and event.get("type") == "finish"


async def event_stream(
    run: Callable[[Callable[[dict], None]], Awaitable[Any]],
    *,
    keepalive_sec: float = KEEPALIVE_SEC,
) -> AsyncIterator[str]:
    """Run `run(emit)` as a task and yield its events as SSE until `finish`. Cancelling or closing
    this generator cancels the task. A turn that ends without `finish` (it should not) is closed
    with a generic `error` and `finish`."""
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.ensure_future(run(queue.put_nowait))
    task.add_done_callback(_consume)
    getter: asyncio.Future | None = None
    try:
        while True:
            if getter is None:
                getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({getter, task}, timeout=keepalive_sec,
                                         return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                event, getter = getter.result(), None
                yield _sse(event)
                if _is_finish(event):
                    return
                continue
            if task in done:
                while not queue.empty():
                    event = queue.get_nowait()
                    yield _sse(event)
                    if _is_finish(event):
                        return
                yield _sse({"type": "error", "message": _STREAM_FAILED})
                yield _sse({"type": "finish", "usage": {}, "elapsed_ms": 0})
                return
            yield _KEEPALIVE
    finally:
        if getter is not None:
            getter.cancel()
        if not task.done():
            task.cancel()


# --- sessions and chat --------------------------------------------------------------------------------


@router.post("/sessions")
async def create_session(rt: Runtime, body: SessionCreate | None = None):
    body = body or SessionCreate()
    session = rt.sessions.create(title=body.title or "New chat", lang=body.lang or "en")
    return respond(session, 201)


@router.get("/sessions")
async def list_sessions(rt: Runtime):
    return respond({"sessions": rt.sessions.list()})


@router.get("/sessions/{session_id}")
async def get_session(session_id: SessionId, rt: Runtime):
    session = _session_or_404(rt, session_id)
    rt.tickets.expire_stale()
    mandate = rt.sessions.get_mandate(session_id)
    return respond({
        "session": session,
        "messages": rt.sessions.messages(session_id, limit=50),
        "active_ticket": _dump(rt.tickets.active(session_id)),
        # every ticket of the chat, oldest first: a reload shows requotes no turn mentions
        "tickets": rt.tickets.for_session(session_id),
        "mandate": (mandate or Mandate()).model_dump(mode="json"),
        "mandate_saved": mandate is not None,
    })


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: SessionId, body: SessionRename, rt: Runtime):
    try:
        rt.sessions.rename(session_id, body.title)
    except KeyError:
        raise ApiError(404, "session_not_found", "This chat session was not found.") from None
    return respond(_session_or_404(rt, session_id))


@router.post("/chat")
async def chat(body: ChatRequest, rt: Runtime):
    _session_or_404(rt, body.session_id)
    agent = rt.agent

    async def run(emit: Callable[[dict], None]) -> Any:
        return await agent.run_turn(body.session_id, body.message, emit)

    return StreamingResponse(
        event_stream(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- portfolio and mandate ------------------------------------------------------------------------------


@router.get("/portfolio")
async def portfolio(rt: Runtime):
    broker = rt.broker
    if broker is None:
        return respond({"connected": False})
    try:
        account = await broker.account()
        positions = await broker.positions()
        open_orders = await broker.orders(status="open", limit=100)
    except Exception as exc:  # noqa: BLE001 - logged by type only
        log.warning("portfolio: broker request failed (%s)", type(exc).__name__)
        return respond({"connected": True, "error": "broker_unavailable",
                        "message": "The broker could not be reached. Try again shortly."}, 502)
    if account.paper is False:  # not a paper account: the runtime drops this broker for good
        await _shielded(rt.drop_non_paper_broker(broker))
        return respond({"connected": False})
    return respond({
        "connected": True,
        "paper": account.paper,  # from the account itself: true, or null when Alpaca did not say
        "account": account.model_dump(mode="json"),
        "positions": [p.model_dump(mode="json") for p in positions.values()],
        "open_orders": [o.model_dump(mode="json") for o in open_orders],
    })


@router.get("/mandate/{session_id}")
async def get_mandate(session_id: SessionId, rt: Runtime):
    _session_or_404(rt, session_id)
    saved = rt.sessions.get_mandate(session_id)
    return respond({"mandate": (saved or Mandate()).model_dump(mode="json"), "saved": saved is not None})


@router.put("/mandate/{session_id}")
async def put_mandate(session_id: SessionId, request: Request, rt: Runtime):
    _session_or_404(rt, session_id)
    raw = await request.body()
    if len(raw) > MANDATE_MAX_BYTES:
        raise ApiError(413, "too_large", "The mandate is too large.")
    try:
        data = json.loads(raw, parse_float=Decimal)  # JSON numbers become exact Decimals
    except (ValueError, RecursionError):
        data = None
    if not isinstance(data, dict):
        raise ApiError(422, "invalid_request", "The mandate must be a JSON object.",
                       fields=[{"loc": "body", "msg": "expected a JSON object"}])
    try:
        mandate = Mandate.model_validate(data)
    except ValidationError as exc:
        raise ApiError(422, "invalid_request", "Some mandate fields are invalid.",
                       fields=_field_errors(exc.errors())) from None
    try:
        saved = rt.sessions.save_mandate(session_id, mandate)
    except KeyError:
        raise ApiError(404, "session_not_found", "This chat session was not found.") from None
    try:
        rt.ledger.append(session_id, "mandate_saved",
                         {"version": saved.version, "mandate": saved.model_dump(mode="json")})
    except Exception as exc:  # noqa: BLE001 - the mandate is saved; the audit write is best effort
        log.warning("mandate_saved ledger entry failed (%s)", type(exc).__name__)
    return respond({"mandate": saved.model_dump(mode="json"), "saved": True})


# --- tickets ------------------------------------------------------------------------------------------------


@router.get("/tickets/{ticket_id}")
async def get_ticket(
    ticket_id: TicketId, rt: Runtime,
    session_id: Annotated[str, Query(min_length=1, max_length=ID_MAX)],
):
    rt.tickets.expire_stale()
    executor = rt.executor
    if executor is not None:
        tracked = await executor.ticket_orders(ticket_id, session_id)
    else:
        tracked = _local_orders(rt, ticket_id, session_id)
    if tracked is None:
        raise ApiError(404, "ticket_not_found", "Ticket not found.")
    ticket, rows = tracked
    return respond({"ticket": _dump(ticket), "orders": [_order_json(row) for row in rows]})


async def _ticket_action(rt, action: str, ticket_id: str, session_id: str, ticket_hash: str = "") -> Response:
    async def locked() -> ConfirmResult | None:
        async with rt.lock:
            executor = rt.executor
            if action == "reject":  # a local state change: works without a broker
                rt.tickets.expire_stale()
                return reject_ticket(rt.tickets, rt.ledger, ticket_id, session_id)
            if executor is None:
                return None
            rt.tickets.expire_stale()
            if action == "confirm":  # the only call in this module that can submit an order
                return await executor.confirm(ticket_id, session_id, ticket_hash)
            return await executor.refresh(ticket_id, session_id)

    try:
        result = await _shielded(locked())
    except Exception as exc:  # noqa: BLE001 - logged by type only, never echoed
        log.warning("ticket %s failed: %s", action, type(exc).__name__)
        if action == "refresh":
            if isinstance(exc, ValueError):
                return _error(400, "invalid_request", "This ticket cannot be re-quoted.")
            return _error(502, "refresh_failed", "The ticket could not be re-quoted. Try again shortly.")
        if action == "confirm":
            return _error(500, "confirm_failed", "The confirmation could not be completed. Check the "
                          "ticket and your open orders before trying again.")
        return _error(500, "reject_failed", "The ticket could not be rejected. Try again.")
    if result is None:
        return _error(503, "broker_not_connected", "Connect your Alpaca paper account in Settings first.")
    return respond({
        "status": result.status,
        "message": rt.ledger.redact_text(result.message),
        "ticket": _dump(result.ticket),
        "orders": [o.model_dump(mode="json") for o in result.orders],
        "new_ticket": _dump(result.new_ticket),
    }, status_for(action, result))


@router.post("/tickets/{ticket_id}/confirm")
async def confirm_ticket_route(ticket_id: TicketId, body: ConfirmRequest, rt: Runtime):
    return await _ticket_action(rt, "confirm", ticket_id, body.session_id, body.hash)


@router.post("/tickets/{ticket_id}/reject")
async def reject_ticket_route(ticket_id: TicketId, body: TicketAction, rt: Runtime):
    return await _ticket_action(rt, "reject", ticket_id, body.session_id)


@router.post("/tickets/{ticket_id}/refresh")
async def refresh_ticket_route(ticket_id: TicketId, body: TicketAction, rt: Runtime):
    return await _ticket_action(rt, "refresh", ticket_id, body.session_id)


# --- settings and kill switch ---------------------------------------------------------------------------


_set = api_llm.is_set


def _settings_body(rt) -> dict:
    s = rt.settings
    if rt.alpaca_status == "not_paper":  # keys of a non-paper account: no broker is kept for them
        alpaca = "not_paper"
    else:
        alpaca = "missing" if rt.broker is None else (rt.alpaca_status or "unreachable")
    return {
        "alpaca": alpaca,
        "tavily": _set(s.tavily_api_key),
        "sec_user_agent": _set(s.sec_user_agent),
        "gemini": _set(s.gemini_api_key),
        "trading_enabled": bool(s.trading_enabled),
        "api_allow_remote": bool(getattr(s, "api_allow_remote", False)),
        # Names of writable settings that only the process environment provides (see env_fallbacks).
        "env_fallbacks": env_fallbacks(rt.env_path),
        **api_llm.llm_settings_body(rt),  # provider, model, effort, mode, limits, gemini_paid, providers
    }


@router.get("/settings")
async def get_settings_status(rt: Runtime):
    # Unchecked, or unsettled (unreachable, invalid, ...): check again, at most once a minute. The
    # check may drop the broker of a non-paper account, so it runs to the end even if we are gone.
    await _shielded(rt.refresh_alpaca_status_if_due())
    return respond(_settings_body(rt))


@router.put("/settings")
async def put_settings(body: SettingsUpdate, rt: Runtime):
    changes: dict[str, Any] = {}
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is None:
            continue  # an explicit null means "leave as is"; "" removes a key
        if field in _PLAIN_FIELDS:
            changes[field] = value
            continue
        target, label = _KEY_FIELDS[field]
        value = value.strip(" \t\r\n")  # ASCII whitespace only: a Unicode separator is refused below
        if value:
            try:
                check_env_value(value)
            except EnvFileError:
                raise ApiError(400, "invalid_value", f"The {label} contains characters that cannot be "
                               "saved: use plain ASCII only, without line breaks, double quotes, "
                               "backslashes or '${'.", field=field) from None
        changes[target] = value or None

    has_id, has_secret = "alpaca_api_key_id" in changes, "alpaca_api_secret_key" in changes
    if has_id != has_secret or (has_id and (changes["alpaca_api_key_id"] is None)
                                != (changes["alpaca_api_secret_key"] is None)):
        raise ApiError(400, "alpaca_pair_required",
                       "Enter the Alpaca key ID and secret key together (or clear both).")
    alpaca_status = None
    if has_id and changes["alpaca_api_key_id"] is not None:
        alpaca_status = await rt.check_alpaca_keys(changes["alpaca_api_key_id"],
                                                   changes["alpaca_api_secret_key"])
        if alpaca_status == "invalid":
            raise ApiError(400, "alpaca_invalid", "Alpaca rejected these keys. Check that they are "
                           "paper-trading keys and try again. Nothing was saved.")
        if alpaca_status == "not_paper":
            raise ApiError(400, "alpaca_not_paper", "These keys belong to a live Alpaca account. This app "
                           "only works with paper-trading keys. Nothing was saved.")
        if alpaca_status != "connected":
            raise ApiError(502, "alpaca_unreachable", "Alpaca could not be reached to check the keys. "
                           "Nothing was saved; try again shortly.")
    # may make one live model call; raises ApiError. `llm_context`: what the decision was made for
    warning, llm_context = await api_llm.check_llm_changes(rt, changes)
    if changes:
        try:
            recheck = await _shielded(rt.apply_settings(changes, alpaca_status=alpaca_status,
                                                        llm_context=llm_context))
            if recheck is not None:  # a concurrent write changed the configuration: checked again
                warning = api_llm.raise_for_llm_status(recheck)
        except LLMCheckFailed as exc:
            api_llm.raise_for_llm_status(exc.status, rt.llm_context(changes)[0])
        except Exception as exc:  # noqa: BLE001 - logged by type only
            log.warning("saving settings failed: %s", type(exc).__name__)
            raise ApiError(500, "settings_not_saved", "The settings could not be saved.") from None
    return respond({**_settings_body(rt), **({"warning": warning} if warning else {})})


@router.post("/kill-switch")
async def kill_switch(rt: Runtime, body: KillSwitchRequest | None = None):
    body = body or KillSwitchRequest()
    return respond(await _shielded(rt.kill_switch(body.suspend_broker)))


# --- ledger export ------------------------------------------------------------------------------------------


@router.get("/ledger/{session_id}.jsonl")
async def export_ledger(session_id: SessionId, rt: Runtime):
    _session_or_404(rt, session_id)
    ledger = rt.ledger

    async def lines() -> AsyncIterator[str]:
        for line in ledger.export_jsonl(session_id):  # already redacted by the Ledger
            yield line + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="ledger-{session_id}.jsonl"'},
    )
