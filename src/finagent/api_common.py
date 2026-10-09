"""What every `/api/*` router module shares: the JSON body builder, the error envelope, the guarded
route class and the runtime dependency. `api_chat` (the main router) and `api_llm` (the model
provider settings and `POST /api/llm/models`) are built from these; `api_chat` re-exports them
under their earlier names.

**Errors** are `{"error": "<code>", "message": "<friendly English>"}`, never exception text:
unexpected failures are logged by exception type only, and request-validation errors list field
locations and pydantic's own messages (`fields: [{loc, msg}]`) without echoing any input, so a
rejected secret is never sent back. Every JSON body is built here with Decimals as strings.
Without a chat runtime (`app.state.fin.chat is None`) every route answers 503 `chat_unavailable`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("finagent.api_common")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return f"<{type(obj).__name__}>"  # never repr(): it could carry anything


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


def respond(payload: Any, status: int = 200) -> Response:
    return Response(dumps(payload).encode("utf-8"), status_code=status, media_type="application/json")


def error_response(status: int, code: str, message: str, **extra: Any) -> Response:
    return respond({"error": code, "message": message, **extra}, status)


class ApiError(Exception):
    """Raised inside a route to answer `{"error": code, "message": message, **extra}`."""

    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(code)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def field_errors(errors: list[dict]) -> list[dict]:
    """Field locations and messages only - never `input`/`ctx`, which echo what was sent."""
    out = []
    for err in errors[:20]:
        loc = [str(part) for part in err.get("loc", ()) if part not in ("body", "query", "path")]
        out.append({"loc": ".".join(loc) or "body", "msg": str(err.get("msg", "invalid"))[:200]})
    return out


class GuardedRoute(APIRoute):
    """Turns `ApiError`, validation errors and unexpected exceptions into safe JSON errors."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()
        path = self.path

        async def guarded(request: Request) -> Response:
            try:
                return await handler(request)
            except ApiError as exc:
                return error_response(exc.status, exc.code, exc.message, **exc.extra)
            except RequestValidationError as exc:
                return error_response(422, "invalid_request", "Some fields are missing or invalid.",
                                      fields=field_errors(list(exc.errors())))
            except StarletteHTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - logged by type only, never echoed
                log.warning("%s %s failed: %s", request.method, path, type(exc).__name__)
                return error_response(500, "internal_error", "Something went wrong on the server. Please try again.")

        return guarded


def runtime(request: Request):
    chat = getattr(getattr(request.app.state, "fin", None), "chat", None)
    if chat is None:
        raise ApiError(503, "chat_unavailable",
                       "The chat service is not available. Check the server log and restart the app.")
    return chat


Runtime = Annotated[Any, Depends(runtime)]


class Body(BaseModel):
    """A request body: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")
