"""Audit-ledger entries written by the chat toolbox.

`ToolAudit` appends, through the `Ledger` it is given (none: nothing is recorded):

- **`tool_call`** - one per `ChatToolbox.execute` call, after the call finishes:
  `{"turn_id", "tool", "args_sha256", "result_sha256", "is_error", "latency_ms", "evidence_ids"}`.
  `tool` is the tool's name when it is a registered tool, else `"unknown_tool"`. `args_sha256` is
  the SHA-256 of the canonical JSON of the arguments (`args_sha256`); `result_sha256` is the
  SHA-256 of the output text's UTF-8 bytes; `evidence_ids` are the ids of the sources the call
  returned. A call that did not finish has no result hash: a cancelled one is recorded with
  `"is_error": true, "cancelled": true`, any other escape with `"raised": "<exception type>"`.
- **`proposal`** - one per ticket `propose_trades` creates: the ticket as shown
  (`trading.proposals.ticket_audit_payload`) plus `turn_id` and `"source": "chat"`.
- **`proposal_rejected`** - one per proposal that `propose_trades` itself refused, with a `reason`
  code and never a ticket id: `all_legs_rejected` and `ticket_validation_failed` (both with the
  mandate version, the cited evidence ids and each decision's verdict and reason codes),
  `invalid_arguments` and `one_proposal_per_turn`.

**Hashes, ids, codes and numbers only.** Tool arguments and results can hold untrusted web text
and personal data, and a thesis is model-authored free text, so none of them is stored: a hash
lets an auditor match an entry against a transcript without the ledger becoming a second copy of
it. (A hash of a short, guessable argument can be confirmed by guessing it; that is accepted.)

**Never in the way.** Building and appending an entry is wrapped: a failure is logged by exception
type and the tool call carries on. Nothing here awaits, so a cancellation is never swallowed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Container
from typing import Any

from ..trading.proposals import decision_audit_rows, ticket_audit_payload

log = logging.getLogger("finagent.chat.audit")

UNKNOWN_TOOL = "unknown_tool"


def sha256_text(text: str) -> str:
    """SHA-256 (hex) of `text` as UTF-8."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def args_sha256(args: Any) -> str:
    """SHA-256 (hex) of the canonical JSON of a tool call's arguments: sorted keys, compact
    separators, ASCII output, non-JSON values through `str`. Anything but a dict (None, a string,
    a list) is hashed as `null`."""
    canonical = json.dumps(
        args if isinstance(args, dict) else None, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolAudit:
    """Appends the toolbox's audit entries; see the module docstring. `known_tools` is the set of
    registered tool names."""

    def __init__(self, ledger: Any, known_tools: Container[str]):
        self._ledger = ledger
        self._known_tools = known_tools

    def _append(self, session_id: Any, kind: str, build) -> None:
        if self._ledger is None:
            return
        try:
            self._ledger.append(str(session_id), kind, build())
        except Exception as exc:  # noqa: BLE001 - an audit write must never break a tool call
            log.warning("ledger entry %s could not be written (%s)", kind, type(exc).__name__)

    def tool_call(self, session_id: Any, turn_id: Any, name: Any, args: Any, started: float,
                  output: Any = None, raised: BaseException | None = None) -> None:
        """One finished call (`output`), or one that escaped `execute` (`raised`). `started` is
        the `time.perf_counter()` reading taken before the call."""

        def build() -> dict:
            known = isinstance(name, str) and name in self._known_tools
            try:
                digest = args_sha256(args)
            except Exception:  # noqa: BLE001 - e.g. mixed key types or a cycle: recorded as null
                digest = None
            payload = {
                "turn_id": str(turn_id),
                "tool": name if known else UNKNOWN_TOOL,
                "args_sha256": digest,
            }
            if output is not None:
                payload["result_sha256"] = sha256_text(output.text)
                payload["is_error"] = bool(output.is_error)
            else:
                payload["is_error"] = True
                if raised is None or isinstance(raised, asyncio.CancelledError):
                    payload["cancelled"] = True
                else:
                    payload["raised"] = type(raised).__name__
            payload["latency_ms"] = max(0, int((time.perf_counter() - started) * 1000))
            payload["evidence_ids"] = [source.id for source in output.sources] if output is not None else []
            return payload

        self._append(session_id, "tool_call", build)

    def proposal(self, session_id: Any, turn_id: Any, ticket: Any) -> None:
        self._append(session_id, "proposal",
                     lambda: {"turn_id": str(turn_id), "source": "chat", **ticket_audit_payload(ticket)})

    def proposal_rejected(self, session_id: Any, turn_id: Any, reason: str, *, decisions=None,
                          mandate_version: int | None = None, evidence_ids=None) -> None:
        def build() -> dict:
            payload: dict[str, Any] = {"turn_id": str(turn_id), "reason": reason}
            if mandate_version is not None:
                payload["mandate_version"] = mandate_version
            if evidence_ids is not None:
                payload["evidence_ids"] = list(evidence_ids)
            if decisions is not None:
                payload["decisions"] = decision_audit_rows(decisions)
            return payload

        self._append(session_id, "proposal_rejected", build)
