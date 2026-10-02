"""Wrappers the scenario evaluation puts around the model client and the broker (see
`eval_scenarios.py`): a step counter, a counter of the client's own retries, and a read-only broker.
None of them changes what the agent does; they only observe, or refuse writes.
"""
from __future__ import annotations

import asyncio
import logging
import re

_ERROR_KINDS = {
    "RateLimitError": "rate_limited_429", "InternalServerError": "server_error_5xx", "APITimeoutError": "timeout",
    "APIConnectionError": "connection", "AuthenticationError": "auth", "PermissionDeniedError": "auth",
    "NotFoundError": "model_not_found", "BadRequestError": "bad_request", "GEMINI_API_KEY": "missing_key",
}


def error_kind(text: str) -> str:
    """A short label for a failed model step. Only the exception class is read from the step's
    error text; the provider's message (which can quote request details) is never kept."""
    match = re.match(r"LLM call error: (\w+)", text or "")
    return _ERROR_KINDS.get(match[1] if match else "", "error_turn")


class _CountingConversation:
    def __init__(self, owner: "CountingLLM", inner):
        self._owner, self._inner = owner, inner

    async def step(self):
        self._owner.calls += 1
        try:
            turn = await self._inner.step()
        except asyncio.CancelledError:
            self._owner.errors.append("cancelled_at_time_budget")
            raise
        except Exception as exc:  # noqa: BLE001 - recorded by type, then left to the agent
            self._owner.errors.append(type(exc).__name__)
            raise
        if turn.stop_reason == "error":
            self._owner.errors.append(error_kind(turn.text))
        return turn

    def add_tool_results(self, results) -> None:
        self._inner.add_tool_results(results)

    def add_user(self, text: str) -> None:
        self._inner.add_user(text)

    def add_assistant(self, text: str) -> None:
        self._inner.add_assistant(text)

    def transcript(self):
        return self._inner.transcript()


class CountingLLM:
    """A thin `LLMClient` wrapper: `calls` counts every `conv.step`, `errors` lists the failed ones."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "llm")
        self.model = getattr(inner, "model", None)
        self.calls = 0
        self.errors: list[str] = []

    def start(self, system, tools, user_text):
        return _CountingConversation(self, self.inner.start(system, tools, user_text))


class ProviderRetryLog(logging.Handler):
    """Counts the model client's retry warnings (logger `finagent.llm`): a 429 or a 5xx that the
    client retried inside one step never reaches the event stream, but it is why a turn ran out of
    its budget. Only the kind is counted; the messages are not kept."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()  # also right when a record factory has pre-formatted it
        except Exception:  # noqa: BLE001 - a malformed record is not a retry
            return
        if "retrying" not in message:
            return
        if message.startswith("429"):
            kind = "rate_limited_429"
        elif message.startswith("InternalServerError"):
            kind = "server_error_5xx"
        else:
            kind = "timeout_or_connection"
        self.counts[kind] = self.counts.get(kind, 0) + 1


class ReadOnlyBroker:
    """Wraps a broker so that only its reads go through. Order writes raise `PermissionError`, and
    any other attribute does not exist: the evaluation proposes, it never trades."""

    READS = frozenset({"account", "positions", "orders", "asset", "clock", "get_order", "get_order_by_client_id",
                       "aclose"})
    WRITES = frozenset({"submit_order", "cancel_order", "set_suspend_trade"})

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name: str):
        if name in ReadOnlyBroker.READS:
            return getattr(self._inner, name)
        if name in ReadOnlyBroker.WRITES:
            async def blocked(*args, **kwargs):
                raise PermissionError(f"eval_scenarios is read-only: {name} is blocked (proposals only, never orders)")
            return blocked
        raise AttributeError(name)
