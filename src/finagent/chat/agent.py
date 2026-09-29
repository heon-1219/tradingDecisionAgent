"""The chat agent's turn loop: one user message in, streamed events out, one answer back.

`ChatAgent.run_turn(session_id, message, emit)` drives the model through `ChatToolbox` for one
turn and reports progress through `emit(event)` (a sync callable, e.g. an SSE queue). The LLM
proposes, code decides and the human approves: the toolbox has no order-placing tool, and
`propose_trades` only creates a ticket that the user confirms in the UI.

Events (Vercel AI SDK stream-protocol names, see the design spec §7), in order:
    start {messageId}
    reasoning-delta {delta}                       interim narration of a tool-calling round
    tool-input-available {toolCallId, toolName, input}
    tool-output-available {toolCallId, output, isError, sources: [{id, title, url, publisher, tier}]}
    source-url {sourceId, url, title}             once per source per turn
    data-mandate / data-trade-proposal / ...      the toolbox's own events, passed through as is
    data-grounding-flags {numbers}                numbers in the answer that no tool output supports
    text-delta {delta}                            the final text in ~200-character chunks
    error {message}                               generic and language-appropriate, never provider text
    finish {usage, elapsed_ms}                    always last, also after an error
A request refused before the turn starts (unknown session, empty or over-long message, a reply
already in progress for the session) emits only `error` and `finish`.

Budgets: at most `settings.chat_max_rounds` model rounds and `settings.chat_time_budget_sec`
seconds from the start of `run_turn`. A tool still running when time runs out is cancelled (the
toolbox releases its one-ticket-per-turn slot on `CancelledError`), and every call of that round
without a result gets an error result, so the provider history stays valid. Then the model is
nudged once to answer without tools (a short grace step allowed past the budget); if it still gives
no text, a fallback lists the tools that ran.

Conversations are cached per session in memory, keyed by the tool-name set. A missing entry (a new
`ChatAgent`, e.g. after a restart), a changed tool set, or an entry dropped after an error, a
cancellation, a nudge or an empty reply is rebuilt from the stored text history, which is always
consistent (tool calls never reach it).

`run_turn` never raises except `asyncio.CancelledError` (client disconnect), which it re-raises
after releasing the session lock and dropping the cached conversation. Logs carry ids, counts,
types and timings only: never message or answer text, and never exception strings (provider errors
can carry response bodies or keys).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..agent.llm import Conversation, LLMClient, ToolCall, ToolSpec
from ..i18n import Lang, detect_lang, language_name, pick
from .grounding_check import unsupported_numbers
from .prompts import PROMPT_VERSION, build_chat_system_prompt
from .toolbox import ToolOutput

log = logging.getLogger("finagent.chat.agent")

MAX_MESSAGE_CHARS = 8000
HISTORY_LIMIT = 24  # stored user/assistant rows replayed when a conversation is rebuilt
CHUNK_CHARS = 200
MAX_CACHED_CONVERSATIONS = 64  # sessions whose provider conversation is kept in memory (LRU)
NUDGE_TEXT = "Budget reached: answer now with what you have. Do not call tools."
TOOL_CANCELLED_TEXT = "Tool cancelled: the time budget for this reply ran out."
TOOL_REFUSED_TEXT = "Tool not run: the budget for this reply is spent. Answer with what you have."
SUMMARY_CONTENT = "turn summary"

_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_]{1,64}")
# Events left out of the persisted turn summary (the text itself is the assistant row).
_UNSUMMARIZED = frozenset({"start", "finish", "text-delta", "reasoning-delta", "source-url"})

_MESSAGES: dict[str, tuple[str, str]] = {
    "unknown_session": (
        "This chat session was not found. Start a new chat and try again.",
        "대화 세션을 찾을 수 없습니다. 새 대화를 시작한 뒤 다시 시도해 주세요.",
    ),
    "empty": ("Please type a message first.", "메시지를 입력해 주세요."),
    "too_long": (
        f"That message is too long (at most {MAX_MESSAGE_CHARS:,} characters). Please shorten it and send it again.",
        f"메시지가 너무 깁니다(최대 {MAX_MESSAGE_CHARS:,}자). 줄여서 다시 보내 주세요.",
    ),
    "busy": (
        "A reply is still in progress in this chat. Wait for it to finish, then send your message again.",
        "이 대화의 답변이 아직 진행 중입니다. 답변이 끝난 뒤 다시 보내 주세요.",
    ),
    "failed": (
        "Sorry, something went wrong while writing this reply. Please try again in a moment.",
        "죄송합니다. 답변을 작성하는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.",
    ),
    "empty_reply": (
        "The model returned an empty reply. Please try again, or rephrase your question.",
        "모델이 빈 답변을 반환했습니다. 다시 시도하거나 질문을 바꿔 주세요.",
    ),
}


def utcnow() -> datetime:
    """The current tz-aware UTC time (the default `now` of `ChatAgent`)."""
    return datetime.now(timezone.utc)


def _msg(key: str, lang: Lang) -> str:
    en, ko = _MESSAGES[key]
    return pick(lang, en, ko)


def fallback_text(lang: Lang, tools: list[str]) -> str:
    """The answer when the budget ran out and the nudged model still gave no text."""
    names = ", ".join(dict.fromkeys(tools))
    if names:
        return pick(
            lang,
            f"I ran out of time before finishing. Here is what I found so far: I ran {names}, but had no time "
            "left to write up the answer. Please ask again, ideally with a narrower question.",
            f"시간이 부족해 답변을 마치지 못했습니다. 지금까지 진행한 내용: {names} 도구를 실행했지만 답변을 "
            "정리할 시간이 없었습니다. 범위를 좁혀 다시 질문해 주세요.",
        )
    return pick(
        lang,
        "I ran out of time before finishing. Here is what I found so far: nothing yet, as no tool finished "
        "in time. Please ask again, ideally with a narrower question.",
        "시간이 부족해 답변을 마치지 못했습니다. 지금까지 진행한 내용: 제시간에 끝난 도구가 없어 아직 "
        "없습니다. 범위를 좁혀 다시 질문해 주세요.",
    )


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """`text` in chunks of at most `size` characters that concatenate back to exactly `text`.
    A chunk ends just after whitespace when there is some in the second half of its window."""
    chunks: list[str] = []
    start, n = 0, len(text)
    while n - start > size:
        cut = start + size
        for j in range(start + size, start + size // 2, -1):
            if text[j - 1].isspace():
                cut = j
                break
        chunks.append(text[start:cut])
        start = cut
    if start < n:
        chunks.append(text[start:])
    return chunks


def _safe_name(name: Any) -> str:
    """A model-supplied tool name, fit for a log line."""
    return name if isinstance(name, str) and _SAFE_NAME_RE.fullmatch(name) else "?"


def _safe_emit(emit: Callable[[dict], None], event: dict, where: str) -> bool:
    try:
        emit(event)
        return True
    except Exception as exc:  # noqa: BLE001 - a broken consumer must never kill the turn
        log.warning("%s: emit failed (%s)", where, type(exc).__name__)
        return False


class _StepFailed(Exception):
    """The model step failed (an error turn or an exception); carries no provider text."""


class _TurnState:
    """Everything one turn accumulates: usage, tools run, grounding corpus and the event summary."""

    def __init__(self, emit: Callable[[dict], None], session_id: str, turn_id: str, lang: Lang):
        self._emit = emit
        self.session_id, self.turn_id, self.lang = session_id, turn_id, lang
        self.usage: dict[str, int] = {}
        self.tools: list[str] = []  # names of the tool calls that ran (not cancelled), in order
        self.corpus: list[str] = []  # tool output texts, then the message, mandate and date
        self.summary: list[dict] = []
        self.call_names: dict[str, str] = {}
        self.sources_seen: set[str] = set()
        self.rounds = 0
        self._emit_warned = False

    def emit(self, event: dict) -> None:
        compact = self._compact(event)
        if compact is not None:
            self.summary.append(compact)
        try:
            self._emit(event)
        except Exception as exc:  # noqa: BLE001 - a broken consumer must never kill the turn
            if not self._emit_warned:  # once per turn: a dead consumer fails every event
                self._emit_warned = True
                log.warning("chat turn %s: emit failed (%s); continuing", self.turn_id, type(exc).__name__)

    def _compact(self, event: dict) -> dict | None:
        kind = event.get("type")
        if not isinstance(kind, str) or kind in _UNSUMMARIZED:
            return None
        out: dict[str, Any] = {"type": kind}
        if kind == "tool-input-available":
            out["toolName"] = event.get("toolName")
            out["toolCallId"] = event.get("toolCallId")
        elif kind == "tool-output-available":
            cid = event.get("toolCallId")
            out["toolName"] = self.call_names.get(cid)
            out["toolCallId"] = cid
            out["isError"] = bool(event.get("isError"))
        elif kind == "data-trade-proposal":
            ticket = event.get("ticket")
            if isinstance(ticket, dict) and ticket.get("id"):
                out["ticket_id"] = ticket["id"]
        return out

    def add_usage(self, usage: dict | None) -> None:
        for key, value in (usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                self.usage[key] = self.usage.get(key, 0) + value


class ChatAgent:
    """Runs chat turns: see the module docstring."""

    def __init__(
        self,
        llm: LLMClient,
        toolbox,
        sessions,
        settings,
        now: Callable[[], datetime] = utcnow,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._llm = llm
        self._toolbox = toolbox
        self._sessions = sessions
        self._settings = settings
        self._now = now
        self._clock = clock
        self._locks: dict[str, asyncio.Lock] = {}
        # session id -> (tool names the conversation was built with, the provider conversation)
        self._convs: OrderedDict[str, tuple[frozenset[str], Conversation]] = OrderedDict()

    # --- entry point ---------------------------------------------------------------------------

    async def run_turn(self, session_id: str, message: str, emit: Callable[[dict], None]) -> str:
        """Run one turn and return the final text (also streamed as `text-delta`). Never raises
        except `asyncio.CancelledError`."""
        t0 = self._clock()
        text = message if isinstance(message, str) else ""
        lang = detect_lang(text[: MAX_MESSAGE_CHARS + 1])  # an over-long message is refused anyway
        problem = self._validate(session_id, text)
        if problem is not None:
            return self._refuse(emit, lang, problem, t0)
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        if lock.locked():  # one turn per session at a time; never queued
            log.info("chat session %s: turn refused, a reply is still in progress", session_id)
            return self._refuse(emit, lang, "busy", t0)
        await lock.acquire()  # free, so this does not suspend: no other turn can slip in
        try:
            return await self._run_locked(session_id, text, lang, emit, t0)
        finally:
            lock.release()
            if self._locks.get(session_id) is lock:
                del self._locks[session_id]

    def _validate(self, session_id: str, text: str) -> str | None:
        try:
            known = isinstance(session_id, str) and self._sessions.get(session_id) is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("chat: session lookup failed (%s)", type(exc).__name__)
            return "failed"
        if not known:
            return "unknown_session"
        if not text.strip():
            return "empty"
        if len(text) > MAX_MESSAGE_CHARS:
            return "too_long"
        return None

    def _refuse(self, emit: Callable[[dict], None], lang: Lang, key: str, t0: float) -> str:
        message = _msg(key, lang)
        _safe_emit(emit, {"type": "error", "message": message}, "chat refusal")
        _safe_emit(emit, {"type": "finish", "usage": {}, "elapsed_ms": self._elapsed_ms(t0)}, "chat refusal")
        return message

    # --- one turn, under the session lock --------------------------------------------------------

    async def _run_locked(self, session_id: str, message: str, lang: Lang, emit, t0: float) -> str:
        turn_id = uuid.uuid4().hex
        st = _TurnState(emit, session_id, turn_id, lang)
        st.emit({"type": "start", "messageId": turn_id})
        try:
            self._sessions.append(session_id, "user", message, {"turn_id": turn_id, "lang": lang})
        except Exception as exc:  # noqa: BLE001 - e.g. the session was deleted meanwhile (KeyError)
            log.warning("chat turn %s: user message not saved (%s)", turn_id, type(exc).__name__)
            error = _msg("unknown_session" if isinstance(exc, KeyError) else "failed", lang)
            st.emit({"type": "error", "message": error})
            self._finish(st, t0)
            return error

        try:
            final, cacheable = await self._drive(st, message, t0)
            if not cacheable:
                self._convs.pop(session_id, None)
            flags = self._grounding_flags(st, final)
            if flags:
                st.emit({"type": "data-grounding-flags", "numbers": flags})
            for chunk in chunk_text(final):
                st.emit({"type": "text-delta", "delta": chunk})
        except asyncio.CancelledError:
            self._convs.pop(session_id, None)
            log.info("chat turn %s cancelled after %d ms (%d rounds)", turn_id, self._elapsed_ms(t0), st.rounds)
            raise
        except Exception as exc:  # noqa: BLE001 - run_turn never raises
            self._convs.pop(session_id, None)
            if isinstance(exc, _StepFailed):
                log.warning("chat turn %s: model step failed (%s)", turn_id, exc.args[0] if exc.args else "?")
            else:
                log.warning("chat turn %s failed: %s", turn_id, type(exc).__name__)
            error = _msg("failed", lang)
            st.emit({"type": "error", "message": error})
            self._persist(st, error, [], error=True)
            self._finish(st, t0)
            return error

        self._persist(st, final, flags)
        elapsed = self._finish(st, t0)
        log.info(
            "chat turn %s session %s: %d rounds, %d tool calls, %d grounding flags, %d chars, %d ms",
            turn_id, session_id, st.rounds, len(st.tools), len(flags), len(final), elapsed,
        )
        return final

    async def _drive(self, st: _TurnState, message: str, t0: float) -> tuple[str, bool]:
        """The model/tool loop. Returns (final text, whether the cached conversation is reusable)."""
        specs = self._toolbox.specs()
        mandate_json = self._mandate_json(st.session_id)
        today = self._today()
        turn_text = "\n".join([
            message,
            "",
            f"Today: {today} (UTC), market clock unknown until checked",
            f"Answer language: {language_name(st.lang)}",
            f"Active mandate: {mandate_json or 'none'}",
        ])
        conv = self._conversation(st.session_id, message, specs, turn_text)
        extra_corpus = [message, *([mandate_json] if mandate_json else []), f"Today: {today}"]

        deadline = t0 + self._budget_sec()
        max_rounds = self._max_rounds()
        final: str | None = None
        while st.rounds < max_rounds:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            st.rounds += 1
            try:
                turn = await asyncio.wait_for(conv.step(), timeout=remaining)
            except TimeoutError:
                log.info("chat turn %s: model step hit the time budget (round %d)", st.turn_id, st.rounds)
                break
            except Exception as exc:  # noqa: BLE001 - never the exception text (provider bodies, keys)
                raise _StepFailed(type(exc).__name__) from None
            st.add_usage(turn.usage)
            if turn.stop_reason == "error":
                raise _StepFailed("error turn")  # turn.text holds provider error text: never shown
            text = turn.text or ""
            if not turn.tool_calls:
                final = text
                break
            if text.strip():
                st.emit({"type": "reasoning-delta", "delta": text})
            if not await self._run_tools(st, conv, turn.tool_calls, deadline):
                break

        cacheable = True
        if final is None:  # rounds or time ran out
            final = await self._nudge(st, conv)
            cacheable = False  # the provider history may end oddly; the stored history does not
        if not final.strip():
            final = _msg("empty_reply", st.lang)
            cacheable = False
        st.corpus.extend(extra_corpus)
        return final, cacheable

    async def _run_tools(self, st: _TurnState, conv: Conversation, calls: list[ToolCall], deadline: float) -> bool:
        """Run one round's tool calls in order and feed every result back in ONE message.
        Returns False when the time budget ran out during the round."""
        results: list[tuple[str, str, bool]] = []
        in_time = True
        for c in calls:
            st.call_names[c.id] = c.name
            st.emit({"type": "tool-input-available", "toolCallId": c.id, "toolName": c.name, "input": c.input})
            out = None
            if in_time:
                remaining = deadline - self._clock()
                if remaining > 0:
                    try:
                        out = await asyncio.wait_for(
                            self._toolbox.execute(st.session_id, c.name, c.input, turn_id=st.turn_id),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        log.info("chat turn %s: tool %s cancelled at the time budget", st.turn_id, _safe_name(c.name))
                    except Exception as exc:  # noqa: BLE001 - execute() should never raise; be safe anyway
                        log.warning("chat turn %s: tool %s raised %s", st.turn_id, _safe_name(c.name),
                                    type(exc).__name__)
                        out = ToolOutput(f"Tool error ({type(exc).__name__})", is_error=True)
                if out is None:
                    in_time = False
            if out is None:
                results.append((c.id, TOOL_CANCELLED_TEXT, True))
                st.emit({"type": "tool-output-available", "toolCallId": c.id, "output": TOOL_CANCELLED_TEXT,
                         "isError": True, "sources": []})
                continue
            st.tools.append(c.name)
            st.corpus.append(out.text)
            results.append((c.id, out.text, bool(out.is_error)))
            st.emit({
                "type": "tool-output-available",
                "toolCallId": c.id,
                "output": out.text,
                "isError": bool(out.is_error),
                "sources": [
                    {"id": s.id, "title": s.title, "url": s.url, "publisher": s.publisher, "tier": s.tier}
                    for s in out.sources
                ],
            })
            for s in out.sources:
                if s.url and s.id not in st.sources_seen:
                    st.sources_seen.add(s.id)
                    st.emit({"type": "source-url", "sourceId": s.id, "url": s.url, "title": s.title})
            for event in out.events:
                st.emit(event)
        conv.add_tool_results(results)
        return in_time

    async def _nudge(self, st: _TurnState, conv: Conversation) -> str:
        """One last tool-free step past the budget; the fallback text if it gives nothing."""
        conv.add_user(NUDGE_TEXT)
        fallback = fallback_text(st.lang, st.tools)
        try:
            turn = await asyncio.wait_for(conv.step(), timeout=self._grace_sec())
        except TimeoutError:
            log.info("chat turn %s: the nudge step timed out", st.turn_id)
            return fallback
        except Exception as exc:  # noqa: BLE001
            log.warning("chat turn %s: the nudge step failed (%s)", st.turn_id, type(exc).__name__)
            return fallback
        st.add_usage(turn.usage)
        if turn.stop_reason == "error":
            log.warning("chat turn %s: the nudge step returned an error turn", st.turn_id)
            return fallback
        if turn.tool_calls:  # answered, never run, so every call has a result
            log.info("chat turn %s: the nudged model still called %d tool(s)", st.turn_id, len(turn.tool_calls))
            conv.add_tool_results([(c.id, TOOL_REFUSED_TEXT, True) for c in turn.tool_calls])
        text = turn.text or ""
        return text if text.strip() else fallback

    # --- conversation cache --------------------------------------------------------------------

    def _conversation(self, session_id: str, message: str, specs: list[ToolSpec], turn_text: str) -> Conversation:
        names = frozenset(s.name for s in specs)
        cached = self._convs.get(session_id)
        if cached is not None and cached[0] == names:
            conv = cached[1]
            self._convs.move_to_end(session_id)
            conv.add_user(turn_text)
            return conv
        rows = self._prior_history(session_id, message)
        if rows and rows[-1]["role"] == "user":  # a dangling user row (e.g. a cancelled turn)
            rows[-1]["content"] += "\n\n" + turn_text
        else:
            rows.append({"role": "user", "content": turn_text})
        system = build_chat_system_prompt(self._settings, [s.name for s in specs])
        conv = self._llm.start(system, specs, rows[0]["content"])
        for row in rows[1:]:
            if row["role"] == "assistant":
                conv.add_assistant(row["content"])
            else:
                conv.add_user(row["content"])
        self._convs[session_id] = (names, conv)
        self._convs.move_to_end(session_id)
        while len(self._convs) > MAX_CACHED_CONVERSATIONS:
            self._convs.popitem(last=False)
        log.info("chat session %s: conversation built from %d stored row(s)", session_id, len(rows) - 1)
        return conv

    def _prior_history(self, session_id: str, message: str) -> list[dict]:
        """Stored user/assistant rows before this turn's message: leading assistant rows dropped,
        consecutive same-role rows merged, so roles alternate starting with the user."""
        rows = self._sessions.history(session_id, limit=HISTORY_LIMIT + 1)
        if rows and rows[-1]["role"] == "user" and rows[-1]["content"] == message:
            rows = rows[:-1]  # the message persisted at the start of this turn
        rows = rows[-HISTORY_LIMIT:]
        merged: list[dict] = []
        for row in rows:
            role, content = row.get("role"), row.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
                continue
            if not merged and role == "assistant":
                continue
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n\n" + content
            else:
                merged.append({"role": role, "content": content})
        return merged

    # --- helpers -------------------------------------------------------------------------------

    def _mandate_json(self, session_id: str) -> str | None:
        try:
            mandate = self._sessions.get_mandate(session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("chat session %s: mandate not readable (%s)", session_id, type(exc).__name__)
            return None
        if mandate is None:
            return None
        return json.dumps(mandate.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False)

    def _today(self) -> str:
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc).date().isoformat()

    def _grounding_flags(self, st: _TurnState, final: str) -> list[str]:
        try:
            return unsupported_numbers(final, st.corpus)
        except Exception as exc:  # noqa: BLE001 - flags are advisory; never fail the turn over them
            log.warning("chat turn %s: grounding check failed (%s)", st.turn_id, type(exc).__name__)
            return []

    def _persist(self, st: _TurnState, final: str, flags: list[str], *, error: bool = False) -> None:
        meta: dict[str, Any] = {
            "turn_id": st.turn_id,
            "lang": st.lang,
            "usage": dict(st.usage),
            "tools": list(st.tools),
            "grounding_flags": list(flags),
            "prompt_version": PROMPT_VERSION,
        }
        if error:
            meta["error"] = True
        try:
            self._sessions.append(st.session_id, "assistant", final, meta)
        except Exception as exc:  # noqa: BLE001
            log.warning("chat turn %s: assistant message not saved (%s)", st.turn_id, type(exc).__name__)
        try:
            self._sessions.append(st.session_id, "event", SUMMARY_CONTENT,
                                  {"turn_id": st.turn_id, "events": st.summary})
        except Exception as exc:  # noqa: BLE001
            log.warning("chat turn %s: turn summary not saved (%s)", st.turn_id, type(exc).__name__)

    def _finish(self, st: _TurnState, t0: float) -> int:
        elapsed = self._elapsed_ms(t0)
        st.emit({"type": "finish", "usage": dict(st.usage), "elapsed_ms": elapsed})
        return elapsed

    def _elapsed_ms(self, t0: float) -> int:
        return max(0, int((self._clock() - t0) * 1000))

    def _budget_sec(self) -> float:
        try:
            return float(getattr(self._settings, "chat_time_budget_sec", 90.0))
        except (TypeError, ValueError):
            return 90.0

    def _max_rounds(self) -> int:
        try:
            return max(1, int(getattr(self._settings, "chat_max_rounds", 8)))
        except (TypeError, ValueError):
            return 8

    def _grace_sec(self) -> float:
        try:
            timeout = float(getattr(self._settings, "llm_timeout_sec", 20.0))
        except (TypeError, ValueError):
            timeout = 20.0
        return min(20.0, max(5.0, timeout))

