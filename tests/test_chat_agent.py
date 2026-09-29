"""Tests for `finagent.chat.agent.ChatAgent` - the chat turn loop.

Fixture world: a real `ChatToolbox` over an in-memory `AppDB` (`EvidenceStore`, `SessionStore`,
`TicketStore`, `Ledger`, `Executor`) with `fakes_trading.FakeBroker`/`FakeData` and a tiny EDGAR
fake as the one research connector. The LLM is `StubLLM` (scripted turns); `RecLLM` also keeps every
conversation it started so tests can read the provider-side transcript. No network anywhere.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fakes_trading import FakeBroker, FakeData
from finagent.agent.llm import StubConversation, StubLLM, ToolCall, Turn
from finagent.chat.agent import (
    CACHE_MAX_TOOL_CHARS,
    CACHE_MAX_TURNS,
    MAX_MESSAGE_CHARS,
    NUDGE_TEXT,
    TOOL_CANCELLED_TEXT,
    UNKNOWN_TOOL,
    ChatAgent,
    chunk_text,
)
from finagent.chat.prompts import PROMPT_VERSION
from finagent.chat.sessions import SessionStore
from finagent.chat.toolbox import ChatToolbox, ToolOutput
from finagent.research.evidence import EvidenceStore, ResearchResult
from finagent.store.appdb import AppDB
from finagent.trading.execution import Executor
from finagent.trading.ledger import Ledger
from finagent.trading.models import Mandate
from finagent.trading.proposals import TicketStore

FIXED_NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)
TODAY = "2026-09-28"
HANGUL = re.compile(r"[가-힣]")


# --- fakes ------------------------------------------------------------------------------------


class MiniEdgar:
    """Just `filings`: one SEC source with a URL (a research source, not market data)."""

    def __init__(self, evidence: EvidenceStore):
        self.evidence = evidence

    async def filings(self, session_id, ticker_or_cik, forms=("10-K", "10-Q", "8-K"), limit=10):
        src = self.evidence.add(
            session_id, connector="edgar", title="Apple Inc. 10-K filed 2025-10-31",
            url="https://www.sec.gov/apple-10k", publisher="SEC EDGAR", tier=1,
            published_at="2025-10-31", snippet="10-K filed 2025-10-31", untrusted=False,
        )
        return ResearchResult(text=f"[{src.id}] 10-K filed 2025-10-31 accession 0000320193-25-000079",
                              sources=[src])


class RecLLM(StubLLM):
    """`StubLLM` that also keeps every conversation it started."""

    def __init__(self, script=None, script_factory=None, conversation_cls=StubConversation, **conv_kwargs):
        super().__init__(script, script_factory)
        self.convs: list[StubConversation] = []
        self._cls, self._conv_kwargs = conversation_cls, conv_kwargs

    def start(self, system, tools, user_text):
        self.started.append((system, user_text))
        script = self._factory(user_text) if self._factory else self._script
        conv = self._cls(script, system, user_text, **self._conv_kwargs)
        self.convs.append(conv)
        return conv


class GatedConversation(StubConversation):
    """`step()` first waits for `gate` (so a turn can be held in flight): every step, or only the
    0-based step numbers in `gated_steps`."""

    def __init__(self, script, system, user_text, gate: asyncio.Event, entered: asyncio.Event,
                 gated_steps: set[int] | None = None):
        super().__init__(script, system, user_text)
        self.gate, self.entered, self.gated_steps = gate, entered, gated_steps
        self.steps = 0

    async def step(self):
        n, self.steps = self.steps, self.steps + 1
        if self.gated_steps is None or n in self.gated_steps:
            self.entered.set()
            await self.gate.wait()
        return await super().step()


class ClockedConversation(StubConversation):
    """Each `step()` advances the injected clock by the next entry of `advances` (a slow model)."""

    def __init__(self, script, system, user_text, clock: "FakeClock", advances: list[float]):
        super().__init__(script, system, user_text)
        self.clock, self.advances = clock, list(advances)

    async def step(self):
        if self.advances:
            self.clock.t += self.advances.pop(0)
        return await super().step()


class RaisingConversation(StubConversation):
    async def step(self):
        raise RuntimeError("provider body with sk-SECRET-456 inside")


class Events(list):
    """The `emit` callable: records every event."""

    def __call__(self, event: dict) -> None:
        self.append(event)

    def types(self) -> list[str]:
        return [e["type"] for e in self]

    def of(self, kind: str) -> list[dict]:
        return [e for e in self if e["type"] == kind]


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class World:
    def __init__(self):
        self.db = AppDB(":memory:")
        self.evidence = EvidenceStore(self.db)
        self.sessions = SessionStore(self.db)
        self.session_id = self.sessions.create()["id"]
        self.tickets = TicketStore(self.db, clock=lambda: FIXED_NOW)
        self.ledger = Ledger(self.db)
        self.broker = FakeBroker()
        self.data = FakeData()
        self.settings = SimpleNamespace(
            ticket_ttl_sec=300,
            trading_enabled=True,
            tavily_api_key=None,
            chat_time_budget_sec=90.0,
            chat_max_rounds=8,
            llm_timeout_sec=40.0,
        )
        self.executor = Executor(
            self.broker, self.data, self.tickets, self.ledger, self.settings,
            mandate_for=lambda s: self.sessions.get_mandate(s) or Mandate(),
        )
        self.toolbox = ChatToolbox(
            registry=None, orchestrator=None, broker=self.broker, data=self.data,
            edgar=MiniEdgar(self.evidence), web=None, grounded=None, evidence=self.evidence,
            sessions=self.sessions, tickets=self.tickets, executor=self.executor,
            settings=self.settings, now=lambda: FIXED_NOW,
        )

    def agent(self, llm, **kw) -> ChatAgent:
        return ChatAgent(llm, self.toolbox, self.sessions, self.settings, now=lambda: FIXED_NOW, **kw)

    def ticket_count(self) -> int:
        return self.db.query("SELECT COUNT(*) AS n FROM tickets")[0]["n"]

    def roles(self) -> list[str]:
        return [m["role"] for m in self.sessions.messages(self.session_id)]


@pytest.fixture
def world() -> World:
    return World()


def call(cid: str, name: str, **args) -> ToolCall:
    return ToolCall(cid, name, args)


def proposal_args(evidence_ids: list[str], qty: str = "10") -> dict:
    return {
        "legs": [{"symbol": "AAPL", "side": "buy", "qty": qty, "type": "limit", "limit_price": "200"}],
        "thesis": "Services growth supports the multiple.",
        "risks": "Regulatory pressure on the App Store.",
        "invalidation": "Services revenue growth below 5% next quarter.",
        "evidence_ids": evidence_ids,
    }


def tool_results(conv: StubConversation) -> dict[str, dict]:
    """tool_use_id -> {content, is_error} over the whole transcript."""
    out = {}
    for msg in conv.transcript():
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for r in msg["content"]:
                out[r["tool_use_id"]] = r
    return out


FINAL = "I proposed buying AAPL for your review in the ticket card [E1][E2]. This is paper trading, not advice."


# --- 1. the happy path --------------------------------------------------------------------------


async def test_research_propose_final_turn(world):
    llm = RecLLM([
        Turn("Checking the filings and the quote.",
             [call("c1", "sec_filings", ticker_or_cik="AAPL"), call("c2", "get_quotes", symbols=["AAPL"])],
             "tool_use", {"input": 100, "output": 10}),
        Turn("", [ToolCall("c3", "propose_trades", proposal_args(["E1", "E2"]))], "tool_use",
             {"input": 200, "output": 20, "cache_read": None}),
        Turn(FINAL, [], "end_turn", {"input": 300, "output": 30}),
    ])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "Should I buy Apple?", events)

    assert text == FINAL
    types = events.types()
    n_text = types.count("text-delta")
    assert n_text >= 1
    assert types == [
        "start",
        "reasoning-delta",
        "tool-input-available", "tool-output-available", "source-url",
        "tool-input-available", "tool-output-available",
        "tool-input-available", "tool-output-available", "data-trade-proposal",
        *(["text-delta"] * n_text),
        "finish",
    ]
    turn_id = events[0]["messageId"]
    assert re.fullmatch(r"[0-9a-f]{32}", turn_id)
    assert events[1] == {"type": "reasoning-delta", "delta": "Checking the filings and the quote."}
    assert events[2] == {"type": "tool-input-available", "toolCallId": "c1", "toolName": "sec_filings",
                         "input": {"ticker_or_cik": "AAPL"}}
    out1 = events[3]
    assert out1["toolCallId"] == "c1" and out1["isError"] is False and "[E1]" in out1["output"]
    assert out1["sources"] == [{"id": "E1", "title": "Apple Inc. 10-K filed 2025-10-31",
                                "url": "https://www.sec.gov/apple-10k", "publisher": "SEC EDGAR", "tier": 1}]
    assert events[4] == {"type": "source-url", "sourceId": "E1", "url": "https://www.sec.gov/apple-10k",
                         "title": "Apple Inc. 10-K filed 2025-10-31"}
    assert events[6]["sources"][0]["id"] == "E2" and events[6]["sources"][0]["url"] is None
    assert events[8]["isError"] is False, events[8]["output"]
    ticket = world.tickets.active(world.session_id)
    assert ticket is not None and world.ticket_count() == 1
    assert events[9] == {"type": "data-trade-proposal", "ticket": ticket.model_dump(mode="json")}
    assert "".join(e["delta"] for e in events.of("text-delta")) == FINAL
    assert events[-1]["type"] == "finish"
    assert events[-1]["usage"] == {"input": 600, "output": 60}
    assert isinstance(events[-1]["elapsed_ms"], int)
    assert not events.of("data-grounding-flags")
    assert world.broker.submits == 0

    msgs = world.sessions.messages(world.session_id)
    assert [m["role"] for m in msgs] == ["user", "assistant", "event"]
    assert msgs[0]["content"] == "Should I buy Apple?"
    assert msgs[0]["meta"] == {"turn_id": turn_id, "lang": "en"}
    assert msgs[1]["content"] == FINAL
    meta = msgs[1]["meta"]
    assert meta["turn_id"] == turn_id and meta["lang"] == "en"
    assert meta["usage"] == {"input": 600, "output": 60}
    assert meta["tools"] == ["sec_filings", "get_quotes", "propose_trades"]
    assert meta["grounding_flags"] == [] and meta["prompt_version"] == PROMPT_VERSION
    summary = msgs[2]["meta"]
    assert summary["turn_id"] == turn_id
    assert {"type": "data-trade-proposal", "ticket_id": ticket.id} in summary["events"]
    assert {"type": "tool-input-available", "toolName": "propose_trades", "toolCallId": "c3"} in summary["events"]
    # compact: no tool outputs or text in the summary row
    assert "10-K filed" not in str(summary) and FINAL not in str(summary)

    # the per-turn user text sent to the model (not what was persisted)
    system, user_text = llm.started[0]
    assert user_text.split("\n") == [
        "Should I buy Apple?",
        "",
        f"Today: {TODAY} (UTC), market clock unknown until checked",
        "Answer language: English",
        "Active mandate: none",
    ]
    assert "propose_trades" in system


async def test_active_mandate_is_compact_json_in_turn_text(world):
    world.sessions.save_mandate(world.session_id, Mandate(budget_usd="5000"))
    llm = RecLLM([Turn("ok")])
    await world.agent(llm).run_turn(world.session_id, "What is my mandate?", Events())
    line = [ln for ln in llm.started[0][1].split("\n") if ln.startswith("Active mandate: ")][0]
    body = line[len("Active mandate: "):]
    assert body.startswith("{") and ": " not in body and ", " not in body
    assert '"budget_usd":"5000"' in body and '"version":1' in body


# --- 2. one proposal per turn -------------------------------------------------------------------


async def test_second_proposal_in_same_turn_is_refused_to_the_model(world):
    llm = RecLLM([
        Turn("", [call("c1", "sec_filings", ticker_or_cik="AAPL"), call("c1b", "sec_filings", ticker_or_cik="AAPL")]),
        Turn("", [ToolCall("c2", "propose_trades", proposal_args(["E1"]))]),
        Turn("", [ToolCall("c3", "propose_trades", proposal_args(["E1"], qty="5"))]),
        Turn("Done."),
    ])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "Buy some Apple", events)
    assert text == "Done."
    assert world.ticket_count() == 1
    results = tool_results(llm.convs[0])
    assert results["c2"]["is_error"] is False
    assert results["c3"]["is_error"] is True
    assert results["c3"]["content"].startswith("Only one trade proposal per turn")
    out3 = [e for e in events.of("tool-output-available") if e["toolCallId"] == "c3"][0]
    assert out3["isError"] is True
    assert len(events.of("data-trade-proposal")) == 1
    # the same source twice in a turn gives one source-url event
    assert [e["sourceId"] for e in events.of("source-url")] == ["E1"]
    # every round's results go back in ONE message
    batches = [m for m in llm.convs[0].transcript() if m["role"] == "user" and isinstance(m["content"], list)]
    assert [len(b["content"]) for b in batches] == [2, 1, 1]


# --- 3. history rebuild -------------------------------------------------------------------------


async def test_history_rebuilt_in_new_agent_and_cached_within_one(world):
    llm1 = RecLLM([Turn("First answer."), Turn("Second answer.")])
    agent1 = world.agent(llm1)
    await agent1.run_turn(world.session_id, "First question", Events())
    await agent1.run_turn(world.session_id, "Second question", Events())
    assert len(llm1.started) == 1  # the cached conversation was reused for the second turn

    llm2 = RecLLM([Turn("Third answer.")])
    text = await world.agent(llm2).run_turn(world.session_id, "Third question", Events())
    assert text == "Third answer."
    assert len(llm2.started) == 1
    assert llm2.started[0][1] == "First question"  # raw stored text, not the old per-turn text
    t = llm2.convs[0].transcript()
    assert [(m["role"], m["content"]) for m in t[1:5]] == [
        ("user", "First question"),
        ("assistant", "First answer."),
        ("user", "Second question"),
        ("assistant", "Second answer."),
    ]
    assert t[5]["role"] == "user" and t[5]["content"].startswith("Third question\n\nToday: ")
    assert t[6] == {"role": "assistant", "content": "Third answer.", "tool_calls": []}
    assert "Third question" not in str(t[1:5])


async def test_rebuild_drops_leading_assistant_and_merges_same_role_rows(world):
    sid = world.session_id
    for role, content in [("assistant", "orphan greeting"), ("user", "u1"), ("user", "u2"),
                          ("assistant", "a1"), ("event", "ignored"), ("assistant", "a2"), ("user", "u3 dangling")]:
        world.sessions.append(sid, role, content)
    llm = RecLLM([Turn("ok")])
    await world.agent(llm).run_turn(sid, "now", Events())
    assert llm.started[0][1] == "u1\n\nu2"
    t = llm.convs[0].transcript()
    assert t[2] == {"role": "assistant", "content": "a1\n\na2"}
    # a dangling user row is merged with this turn's text, so roles keep alternating
    assert t[3]["role"] == "user" and t[3]["content"].startswith("u3 dangling\n\nnow\n\nToday: ")
    assert [m["role"] for m in t] == ["system", "user", "assistant", "user", "assistant"]
    assert "orphan greeting" not in str(t)


# --- 4. budgets ---------------------------------------------------------------------------------


async def test_injected_clock_stops_loop_and_nudges(world):
    clock = FakeClock()

    async def status_advancing_clock(session_id, args, turn_id):
        clock.t += 1000  # far past the 90 s budget
        return ToolOutput("US market is OPEN.")

    world.toolbox._t_get_market_status = status_advancing_clock
    llm = RecLLM([
        Turn("", [call("c1", "get_market_status")]),
        Turn("Answer after the nudge."),
        Turn("never reached"),
    ])
    events = Events()
    text = await world.agent(llm, clock=clock).run_turn(world.session_id, "Is the market open?", events)
    assert text == "Answer after the nudge."
    t = llm.convs[0].transcript()
    assert t[-2] == {"role": "user", "content": NUDGE_TEXT}
    assert t[-1]["content"] == "Answer after the nudge."
    assert tool_results(llm.convs[0])["c1"]["is_error"] is False
    assert events[-1]["type"] == "finish" and events[-1]["elapsed_ms"] >= 1_000_000


async def test_slow_tool_is_cancelled_within_budget(world):
    # Deterministic: the injected clock says the model step used all but 0.05 s of the 90 s budget,
    # so the hanging tool gets a 0.05 s timeout; it never finishes, so it is always cancelled.
    clock = FakeClock()
    started = asyncio.Event()

    async def hanging_status(session_id, args, turn_id):
        started.set()
        await asyncio.sleep(3600)
        return ToolOutput("never")

    world.toolbox._t_get_market_status = hanging_status
    llm = RecLLM(
        [Turn("Checking.", [call("c1", "get_market_status"), call("c2", "get_account")]), Turn("Partial answer.")],
        conversation_cls=ClockedConversation, clock=clock, advances=[89.95],
    )
    events = Events()
    agent = world.agent(llm, clock=clock)
    text = await asyncio.wait_for(agent.run_turn(world.session_id, "Is the market open?", events), 30)
    assert started.is_set()
    assert text == "Partial answer."
    outs = {e["toolCallId"]: e for e in events.of("tool-output-available")}
    for cid in ("c1", "c2"):
        assert outs[cid]["isError"] is True and outs[cid]["output"] == TOOL_CANCELLED_TEXT
    assert "account" not in world.broker.calls  # c2 never ran
    results = tool_results(llm.convs[0])
    assert results["c1"] == {"tool_use_id": "c1", "content": TOOL_CANCELLED_TEXT, "is_error": True}
    assert results["c2"] == {"tool_use_id": "c2", "content": TOOL_CANCELLED_TEXT, "is_error": True}
    t = llm.convs[0].transcript()
    assert t[-2] == {"role": "user", "content": NUDGE_TEXT}
    assert events[-1]["type"] == "finish"
    assert world.roles() == ["user", "assistant", "event"]


@pytest.mark.parametrize("nudge_text", ["", "Best effort answer."])
async def test_round_budget_nudges_then_falls_back(world, nudge_text):
    world.settings.chat_max_rounds = 2

    def always_tools(_user_text):
        turns = [Turn("", [call(f"c{i}", "get_market_status")]) for i in range(1, 3)]
        turns.append(Turn(nudge_text, [call("c3", "get_market_status")]))  # ignores the nudge
        return turns

    llm = RecLLM(script_factory=always_tools)
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "Is the market open?", events)
    assert len(events.of("tool-output-available")) == 2  # two rounds ran, the nudge's call did not
    if nudge_text:
        assert text == nudge_text
    else:
        assert text.startswith("I ran out of time before finishing. Here is what I found so far:")
        assert "get_market_status" in text
    results = tool_results(llm.convs[0])
    assert results["c3"]["is_error"] is True  # history stays valid: every call has a result
    t = llm.convs[0].transcript()
    nudge_at = t.index({"role": "user", "content": NUDGE_TEXT})
    assert t[nudge_at + 1]["tool_calls"][0]["id"] == "c3"
    assert "".join(e["delta"] for e in events.of("text-delta")) == text
    assert events[-1]["type"] == "finish"


# --- 5. grounding flags -------------------------------------------------------------------------


async def test_unsupported_number_is_flagged(world):
    llm = RecLLM([Turn("Revenue grew 37.5% last year.")])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "How did Apple do?", events)
    assert text == "Revenue grew 37.5% last year."  # never altered
    flags = events.of("data-grounding-flags")
    assert flags == [{"type": "data-grounding-flags", "numbers": ["37.5%"]}]
    types = events.types()
    assert types.index("data-grounding-flags") < types.index("text-delta")
    assert world.sessions.messages(world.session_id)[1]["meta"]["grounding_flags"] == ["37.5%"]


async def test_supported_numbers_are_not_flagged(world):
    llm = RecLLM([
        Turn("", [call("c1", "get_quotes", symbols=["AAPL"])]),
        Turn(f"AAPL last traded at $200.00; a P/E of 31.4 is above average. As of {TODAY}."),
    ])
    events = Events()
    await world.agent(llm).run_turn(world.session_id, "Is a P/E of 31.4 high for AAPL?", events)
    assert not events.of("data-grounding-flags")


# --- 6. single flight ---------------------------------------------------------------------------


async def test_concurrent_turn_on_same_session_is_refused(world):
    gate, entered = asyncio.Event(), asyncio.Event()
    llm = RecLLM([Turn("First done.")], conversation_cls=GatedConversation, gate=gate, entered=entered)
    agent = world.agent(llm)
    first_events = Events()
    first = asyncio.create_task(agent.run_turn(world.session_id, "first", first_events))
    await asyncio.wait_for(entered.wait(), 5)

    second_events = Events()
    text = await agent.run_turn(world.session_id, "second", second_events)
    assert second_events.types() == ["error", "finish"]
    assert "in progress" in second_events[0]["message"] and text == second_events[0]["message"]
    assert len(llm.started) == 1  # the second turn never reached the LLM
    assert [m["content"] for m in world.sessions.messages(world.session_id)] == ["first"]

    gate.set()
    assert await asyncio.wait_for(first, 5) == "First done."
    # the lock is released afterwards
    third_events = Events()
    third = await agent.run_turn(world.session_id, "third", third_events)
    assert third and third_events.types()[0] == "start"  # the script is spent: an empty-reply notice
    assert world.roles()[-3:] == ["user", "assistant", "event"]


async def test_other_sessions_are_not_blocked(world):
    gate, entered = asyncio.Event(), asyncio.Event()
    llm = RecLLM([Turn("done")], conversation_cls=GatedConversation, gate=gate, entered=entered)
    agent = world.agent(llm)
    first = asyncio.create_task(agent.run_turn(world.session_id, "first", Events()))
    await asyncio.wait_for(entered.wait(), 5)
    other = world.sessions.create()["id"]
    other_task = asyncio.create_task(agent.run_turn(other, "other", Events()))
    await asyncio.sleep(0)
    gate.set()
    assert await asyncio.wait_for(asyncio.gather(first, other_task), 5) == ["done", "done"]


# --- 7. step errors -----------------------------------------------------------------------------


async def test_step_error_is_generic_and_next_turn_rebuilds(world, caplog):
    caplog.set_level(logging.DEBUG)
    turns = iter([
        [Turn("LLM call error: AuthenticationError: key sk-SECRET-123 rejected", [], "error")],
        [Turn("Recovered.")],
    ])
    llm = RecLLM(script_factory=lambda u: next(turns))
    agent = world.agent(llm)
    events = Events()
    text = await agent.run_turn(world.session_id, "PRIVATE-QUESTION-TEXT", events)
    assert events.types() == ["start", "error", "finish"]
    msg = events[1]["message"]
    assert "sk-SECRET" not in msg and "AuthenticationError" not in msg and msg
    assert text == msg
    msgs = world.sessions.messages(world.session_id)
    assert [m["role"] for m in msgs] == ["user", "assistant", "event"]
    assert msgs[1]["content"] == msg

    events2 = Events()
    assert await agent.run_turn(world.session_id, "try again", events2) == "Recovered."
    assert len(llm.started) == 2  # rebuilt from stored history, not the broken cached conversation
    t = llm.convs[1].transcript()
    assert (t[1]["role"], t[1]["content"]) == ("user", "PRIVATE-QUESTION-TEXT")
    assert (t[2]["role"], t[2]["content"]) == ("assistant", msg)
    assert "sk-SECRET" not in caplog.text and "PRIVATE-QUESTION-TEXT" not in caplog.text


async def test_step_exception_is_generic(world, caplog):
    caplog.set_level(logging.DEBUG)
    llm = RecLLM([], conversation_cls=RaisingConversation)
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "hello", events)
    assert events.types() == ["start", "error", "finish"]
    assert "sk-SECRET" not in text and "provider body" not in text
    assert "sk-SECRET" not in caplog.text and "provider body" not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_step_error_after_tools_persists_summary_and_drops_cache(world):
    scripts = iter([
        [Turn("", [call("c1", "get_market_status")]), Turn("LLM call error: boom", [], "error")],
        [Turn("fine")],
    ])
    llm = RecLLM(script_factory=lambda u: next(scripts))
    agent = world.agent(llm)
    events = Events()
    await agent.run_turn(world.session_id, "one", events)
    assert events.types()[-2:] == ["error", "finish"]
    summary = world.sessions.messages(world.session_id)[-1]["meta"]["events"]
    assert {"type": "tool-input-available", "toolName": "get_market_status", "toolCallId": "c1"} in summary
    assert {"type": "error"} in summary
    await agent.run_turn(world.session_id, "two", Events())
    assert len(llm.started) == 2


# --- 8. emit failures ---------------------------------------------------------------------------


async def test_raising_emit_does_not_break_the_turn(world, caplog):
    caplog.set_level(logging.DEBUG)

    def bad_emit(event):
        raise RuntimeError("emit-broken-detail")

    llm = RecLLM([Turn("", [call("c1", "get_market_status")]), Turn("Final.")])
    text = await world.agent(llm).run_turn(world.session_id, "hello", bad_emit)
    assert text == "Final."
    assert world.sessions.messages(world.session_id)[1]["content"] == "Final."
    assert "RuntimeError" in caplog.text and "emit-broken-detail" not in caplog.text


# --- 9. text-delta chunking ---------------------------------------------------------------------

LONG = " ".join(f"word{i}" for i in range(300)) + "\n\nEnd of answer."


@pytest.mark.parametrize("final", [LONG, "x" * 450, "short", "tail " * 41 + "y" * 5])
async def test_text_delta_chunks_concatenate_exactly(world, final):
    events = Events()
    text = await world.agent(RecLLM([Turn(final)])).run_turn(world.session_id, "hello", events)
    deltas = [e["delta"] for e in events.of("text-delta")]
    assert "".join(deltas) == text == final
    assert all(0 < len(d) <= 200 for d in deltas)


def test_chunk_text_breaks_at_whitespace():
    chunks = chunk_text(LONG)
    assert "".join(chunks) == LONG and len(chunks) > 5
    assert all(c[-1].isspace() for c in chunks[:-1])
    assert chunk_text("x" * 450) == ["x" * 200, "x" * 200, "x" * 50]
    assert chunk_text("") == []


# --- 10. tool set changes -----------------------------------------------------------------------


async def test_changed_tool_set_rebuilds_conversation(world, monkeypatch):
    llm = RecLLM(script_factory=lambda u: [Turn("ok")])
    agent = world.agent(llm)
    await agent.run_turn(world.session_id, "one", Events())
    full = world.toolbox.specs()
    monkeypatch.setattr(world.toolbox, "specs", lambda: [s for s in full if s.name != "propose_trades"])
    await agent.run_turn(world.session_id, "two", Events())
    assert len(llm.started) == 2
    assert llm.started[0][0] != llm.started[1][0]
    await agent.run_turn(world.session_id, "three", Events())
    assert len(llm.started) == 2  # same set again: cached


# --- 11. Korean ---------------------------------------------------------------------------------

KO_MESSAGE = "애플 주식을 지금 사도 될까요?"


async def test_korean_turn_text_and_error(world):
    llm = RecLLM([Turn("LLM call error: x", [], "error")])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, KO_MESSAGE, events)
    assert "Answer language: Korean" in llm.started[0][1].split("\n")
    assert HANGUL.search(events.of("error")[0]["message"]) and HANGUL.search(text)
    assert world.sessions.messages(world.session_id)[0]["meta"]["lang"] == "ko"


async def test_korean_fallback(world):
    world.settings.chat_max_rounds = 1
    llm = RecLLM([Turn("", [call("c1", "get_market_status")]), Turn("", [call("c2", "get_market_status")])])
    text = await world.agent(llm).run_turn(world.session_id, KO_MESSAGE, Events())
    assert HANGUL.search(text) and "get_market_status" in text
    assert "ran out of time" not in text


# --- validation, cancellation, privacy ----------------------------------------------------------


@pytest.mark.parametrize("session_id, message", [
    ("S-UNKNOWN", "hello"),
    (None, "   \n "),
    (None, "x" * (MAX_MESSAGE_CHARS + 1)),
])
async def test_invalid_requests_emit_error_and_persist_nothing(world, session_id, message):
    llm = RecLLM([Turn("never")])
    events = Events()
    text = await world.agent(llm).run_turn(session_id or world.session_id, message, events)
    assert events.types() == ["error", "finish"]
    assert text == events[0]["message"] and text
    assert llm.started == []
    assert world.sessions.messages(world.session_id) == []


async def test_max_length_message_is_accepted(world):
    llm = RecLLM([Turn("ok")])
    assert await world.agent(llm).run_turn(world.session_id, "y" * MAX_MESSAGE_CHARS, Events()) == "ok"


async def test_cancellation_releases_lock_drops_cache_and_reraises(world):
    gate, entered = asyncio.Event(), asyncio.Event()
    llm = RecLLM(script_factory=lambda u: [Turn("after")], conversation_cls=GatedConversation,
                 gate=gate, entered=entered)
    agent = world.agent(llm)
    task = asyncio.create_task(agent.run_turn(world.session_id, "first", Events()))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.roles() == ["user"]  # nothing more persisted
    gate.set()
    events = Events()
    assert await agent.run_turn(world.session_id, "second", events) == "after"
    assert len(llm.started) == 2  # the cancelled turn's conversation was dropped
    assert events.types()[0] == "start"


async def test_logs_carry_no_message_or_answer_text(world, caplog):
    caplog.set_level(logging.DEBUG)
    llm = RecLLM([Turn("", [call("c1", "get_market_status")]), Turn("SECRET-ANSWER-TEXT 99.9%")])
    await world.agent(llm).run_turn(world.session_id, "SECRET-QUESTION-TEXT", Events())
    assert "SECRET-QUESTION-TEXT" not in caplog.text and "SECRET-ANSWER-TEXT" not in caplog.text
    assert "99.9" not in caplog.text


# --- hardening round 1 --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -5])
def test_chunk_text_nonpositive_size_does_not_hang(size):
    assert chunk_text("ab c", size) == ["a", "b", " ", "c"]


async def test_model_written_tool_names_are_never_stored_or_shown(world):
    bad = "</script><img src=x>"
    world.settings.chat_max_rounds = 1
    llm = RecLLM([
        Turn("", [ToolCall("c1", bad, {}), call("c2", "get_market_status")]),
        Turn("", [ToolCall("c3", bad, {})]),  # the nudged model calls it again, and writes nothing
    ])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "Is the market open?", events)
    inputs = {e["toolCallId"]: e for e in events.of("tool-input-available")}
    assert inputs["c1"]["toolName"] == UNKNOWN_TOOL
    assert inputs["c2"]["toolName"] == "get_market_status"
    assert [e["isError"] for e in events.of("tool-output-available")] == [True, False]
    # the fallback lists only real tools
    assert "get_market_status" in text and UNKNOWN_TOOL not in text
    msgs = world.sessions.messages(world.session_id)
    assert msgs[1]["meta"]["tools"] == [UNKNOWN_TOOL, "get_market_status"]
    names = [e["toolName"] for e in msgs[2]["meta"]["events"] if "toolName" in e]
    assert set(names) == {UNKNOWN_TOOL, "get_market_status"}
    stored = json.dumps(msgs, ensure_ascii=False)
    streamed = json.dumps([e for e in events if e["type"] != "tool-output-available"], ensure_ascii=False)
    for blob in (stored, streamed, text):
        assert "<img" not in blob and "script" not in blob
    # the raw name still reaches the toolbox, which refuses it
    assert tool_results(llm.convs[0])["c1"]["content"].startswith("Unknown tool")


async def test_cached_conversation_is_rebuilt_after_max_turns(world):
    llm = RecLLM(script_factory=lambda u: [Turn(f"answer {i}") for i in range(CACHE_MAX_TURNS + 2)])
    agent = world.agent(llm)
    for i in range(CACHE_MAX_TURNS):
        await agent.run_turn(world.session_id, f"question {i}", Events())
    assert len(llm.started) == 1
    await agent.run_turn(world.session_id, "one more", Events())
    assert len(llm.started) == 2
    t = llm.convs[1].transcript()
    assert t[1]["content"].startswith("question")  # rebuilt from the stored history


async def test_cached_conversation_is_rebuilt_after_large_tool_outputs(world):
    half = CACHE_MAX_TOOL_CHARS // 2 + 5_000

    async def big_status(session_id, args, turn_id):
        return ToolOutput("x" * half)

    world.toolbox._t_get_market_status = big_status
    llm = RecLLM(script_factory=lambda u: [Turn("", [call("c", "get_market_status")]), Turn("ok")] * 3)
    agent = world.agent(llm)
    await agent.run_turn(world.session_id, "one", Events())
    assert len(llm.started) == 1
    await agent.run_turn(world.session_id, "two", Events())  # crosses the limit: dropped afterwards
    assert len(llm.started) == 1
    await agent.run_turn(world.session_id, "three", Events())
    assert len(llm.started) == 2


async def test_cancel_during_tool_releases_slot_lock_and_cache(world):
    e1 = world.evidence.add(
        world.session_id, connector="edgar", title="Apple 10-K", url="https://www.sec.gov/a",
        publisher="SEC EDGAR", tier=1, published_at=None, snippet="10-K", untrusted=False,
    ).id
    entered = asyncio.Event()

    async def hanging_build_context(symbols):
        entered.set()
        await asyncio.sleep(3600)

    world.executor.build_context = hanging_build_context
    llm = RecLLM(script_factory=lambda u: [Turn("", [ToolCall("c1", "propose_trades", proposal_args([e1]))]),
                                           Turn("after")])
    agent = world.agent(llm)
    events = Events()
    task = asyncio.create_task(agent.run_turn(world.session_id, "buy apple", events))
    await asyncio.wait_for(entered.wait(), 5)
    turn_id = events[0]["messageId"]
    assert (world.session_id, turn_id) in world.toolbox._proposal_turns  # the slot is held in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (world.session_id, turn_id) not in world.toolbox._proposal_turns  # released
    assert world.roles() == ["user"]
    assert events[-1]["type"] == "tool-input-available"  # no error/finish after a cancellation
    del world.executor.build_context  # back to the real one: the next turn's proposal completes
    next_events = Events()
    # the lock was released: the next turn runs
    text =await asyncio.wait_for(agent.run_turn(world.session_id, "hello again", next_events), 30)
    assert next_events.types()[0] == "start" and text
    assert len(llm.started) == 2  # the cancelled turn's conversation was dropped


async def test_cancel_during_nudge_reraises_and_cleans_up(world):
    world.settings.chat_max_rounds = 1
    gate, entered = asyncio.Event(), asyncio.Event()
    llm = RecLLM(script_factory=lambda u: [Turn("", [call("c1", "get_market_status")]), Turn("late")],
                 conversation_cls=GatedConversation, gate=gate, entered=entered, gated_steps={1})
    agent = world.agent(llm)
    task = asyncio.create_task(agent.run_turn(world.session_id, "first", Events()))
    await asyncio.wait_for(entered.wait(), 5)
    assert llm.convs[0].transcript()[-1] == {"role": "user", "content": NUDGE_TEXT}  # parked in the nudge
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert world.roles() == ["user"]
    gate.set()
    events = Events()
    await asyncio.wait_for(agent.run_turn(world.session_id, "second", events), 30)
    assert events.types()[0] == "start"
    assert len(llm.started) == 2


async def test_grace_step_timeout_gives_fallback_then_finish(world, monkeypatch):
    world.settings.chat_max_rounds = 1
    gate, entered = asyncio.Event(), asyncio.Event()  # the gate never opens: the nudge step hangs
    llm = RecLLM([Turn("", [call("c1", "get_market_status")]), Turn("never")],
                 conversation_cls=GatedConversation, gate=gate, entered=entered, gated_steps={1})
    agent = world.agent(llm)
    monkeypatch.setattr(agent, "_grace_sec", lambda: 0.05)
    events = Events()
    text = await asyncio.wait_for(agent.run_turn(world.session_id, "Is the market open?", events), 30)
    assert entered.is_set()
    assert text.startswith("I ran out of time before finishing.") and "get_market_status" in text
    assert events.types()[-2:] == ["text-delta", "finish"]
    assert "".join(e["delta"] for e in events.of("text-delta")) == text
    assert world.roles() == ["user", "assistant", "event"]


@pytest.mark.parametrize("exc, expected", [
    (KeyError("gone"), "not found"),
    (RuntimeError("disk full /secret/path"), "went wrong"),
])
async def test_failing_user_append_emits_error_then_finish(world, monkeypatch, exc, expected):
    real_append = world.sessions.append

    def append(session_id, role, content, meta=None):
        if role == "user":
            raise exc
        return real_append(session_id, role, content, meta)

    monkeypatch.setattr(world.sessions, "append", append)
    llm = RecLLM([Turn("never")])
    events = Events()
    text = await world.agent(llm).run_turn(world.session_id, "hello", events)
    assert events.types() == ["start", "error", "finish"]
    assert expected in events[1]["message"] and text == events[1]["message"]
    assert "secret" not in text
    assert llm.started == []
    assert world.sessions.messages(world.session_id) == []
