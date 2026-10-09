"""Tests for the `/api/*` router (`finagent.api_chat`), its security guard (`finagent.api_security`)
and the `ChatRuntime` wiring (`finagent.app_state`).

World: a real `ChatRuntime` (AppDB on a tmp file, EvidenceStore, SessionStore, TicketStore, Ledger,
Executor, ChatToolbox, ChatAgent) over `fakes_trading.FakeBroker`/`FakeData`, a scripted `StubLLM`,
a fake QA orchestrator for `/answer`, an HTTP client that refuses every request (no network), and a
tmp env file. The real `.env` is never touched. The model provider checks (`PUT /api/settings`,
`POST /api/llm/models`) go through the world's fake validator, model lister and client factory;
their tests are in `test_api_llm.py`, which reuses this world.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from fakes_trading import (
    FakeBroker,
    FakeData,
    make_account,
    make_asset,
    make_clock,
    make_quote,
)
from finagent.agent.llm import LLMHandle, StubLLM, ToolCall, Turn
from finagent.api import create_app
from finagent.api_chat import event_stream
from finagent.api_security import LocalApiGuard, is_loopback, parse_host
from finagent.app_state import ENV_NAMES, ChatRuntime
from finagent.config import Settings
from finagent.models import AnswerResponse
from finagent.research.http import HttpClient
from finagent.trading.broker import BrokerError
from finagent.trading.models import Mandate, MarketContext, OrderLeg, TicketStatus
from finagent.trading.risk import evaluate

FIXED_NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)
HDR = {"X-FinAgent-Client": "1"}
LOCAL_URL = "http://127.0.0.1"
LOCAL_CLIENT = ("127.0.0.1", 50000)

ALPACA_ID = "PKTESTKEYID0123456789"
ALPACA_SECRET = "testSecretValue-abcdef0123456789"
NEW_ID = "PKNEWKEYID9876543210"
NEW_SECRET = "newSecretValue-0123456789abcdefgh"
BAD_ID = "PKBADKEYID5555555555"
BAD_SECRET = "badSecretValue-5555555555555555"
DOWN_ID = "PKDOWNKEYID444444444"
DOWN_SECRET = "downSecretValue-4444444444444444"
TAVILY = "tvly-test-key-0123456789abcdef"
SEC_UA_FROM_SHELL = "Shell User shell-user@example.com"
GEMINI = "gemini-test-key-0123456789abcdef"
LIVE_ID = "AKLIVEKEYID333333333"  # keys of a live (non-paper) account
LIVE_SECRET = "liveSecretValue-3333333333333333"
PAPER_ID = "PKPAPERKEYID22222222"  # keys of an account that reports itself as paper
PAPER_SECRET = "paperSecretValue-222222222222222"
# Model provider keys: the fake validator and model lister decide by the key value.
NEW_GEMINI = "gemini-new-key-9876543210fedcba"
OPENAI = "sk-openai-test-key-0123456789abcd"
ANTHROPIC = "sk-ant-test-key-0123456789abcdef"
BAD_LLM_KEY = "sk-bad-llm-key-55555555555555"       # the provider rejects it
LIMITED_LLM_KEY = "sk-limited-llm-key-4444444444"   # valid, rate limited right now
DOWN_LLM_KEY = "sk-down-llm-key-33333333333333"     # the provider cannot be reached
LEAKY_LLM_KEY = "sk-leaky-llm-key-22222222222222"   # the check raises with the key in its message
SECRETS = [ALPACA_ID, ALPACA_SECRET, NEW_ID, NEW_SECRET, BAD_ID, BAD_SECRET, DOWN_ID, DOWN_SECRET,
           TAVILY, GEMINI, LIVE_ID, LIVE_SECRET, PAPER_ID, PAPER_SECRET, NEW_GEMINI, OPENAI, ANTHROPIC,
           BAD_LLM_KEY, LIMITED_LLM_KEY, DOWN_LLM_KEY, LEAKY_LLM_KEY]
LLM_KEY_FIELDS = {"gemini": "gemini_api_key", "openai": "openai_api_key", "anthropic": "anthropic_api_key"}

ENV_TEXT = (
    "# local settings\n"
    "LLM_PROVIDER=stub\n"
    f"ALPACA_API_KEY_ID={ALPACA_ID}\n"
    f"ALPACA_API_SECRET_KEY={ALPACA_SECRET}\n"
    "# research\n"
    f"TAVILY_API_KEY={TAVILY}\n"
    "TRADING_ENABLED=true\n"
)

FINAL = "I proposed a trade for your review [E1]. This is paper trading, not investment advice."
LIMIT_LEG = {"symbol": "AAPL", "side": "buy", "qty": "10", "type": "limit", "limit_price": "200"}
MARKET_LEG = {"symbol": "AAPL", "side": "buy", "qty": "10", "type": "market"}


# --- the world --------------------------------------------------------------------------------------


class Clock:
    def __init__(self, now: datetime = FIXED_NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class ClosableBroker(FakeBroker):
    def __init__(self, key_id: str, **kw):
        super().__init__(**kw)
        self.key_id = key_id
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeOrchestrator:
    """Holds the model handle, like the real QA orchestrator; `think_trace` names the model in use."""

    def __init__(self, llm=None):
        self.llm = llm

    async def answer(self, question_id, question, on_event=None):
        model = f"{self.llm.name}/{self.llm.model}" if self.llm is not None else "none"
        return AnswerResponse(question_id=question_id, question=question, retrieved_context="ctx",
                              think_trace=f"trace by {model}", answer=f"answer to {question}")


class ProviderError(Exception):
    """What a provider SDK raises: an HTTP status and a message (here: with the key in it)."""

    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


class SwitchedLLM(StubLLM):
    """The client the world's `llm_factory` builds for a provider/model: it answers every turn with
    one line naming itself, and records `aclose()`."""

    def __init__(self, provider: str, model: str, effort: str = "medium"):
        super().__init__(script_factory=lambda user_text: [Turn(f"Answer from {provider}/{model}.", [], "end_turn")])
        self.name, self.model, self.effort = provider, model, effort
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


# What each provider's list-models endpoint returns in this world, unfiltered.
MODEL_LISTS = {
    "gemini": [{"id": "gemini-3.8-flash", "label": None, "created": 0.0},
               {"id": "gemini-3.8-pro", "label": None, "created": 0.0},
               {"id": "gemini-embedding-001", "label": None, "created": 0.0},
               {"id": "gemini-3.8-flash-preview-tts", "label": None, "created": 0.0}],
    "openai": [{"id": "gpt-test-large", "label": None, "created": 200.0},
               {"id": "gpt-test-small", "label": None, "created": 100.0},
               {"id": "text-embedding-test", "label": None, "created": 300.0},
               {"id": "omni-moderation-test", "label": None, "created": 300.0}],
    "anthropic": [{"id": "claude-test-sonnet", "label": "Claude Test Sonnet", "created": 300.0},
                  {"id": "claude-test-opus", "label": "Claude Test Opus", "created": 200.0},
                  {"id": "claude-test-opus-old", "label": "Claude Test Opus (old)", "created": 100.0}],
}


def _refuse_all(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no network in tests", request=request)


class ApiWorld:
    def __init__(self, tmp_path, *, alpaca: bool = True, chat: bool = True, **settings_over):
        self.tmp_path = tmp_path
        self.env_path = tmp_path / ".env"
        self.env_path.write_bytes(ENV_TEXT.encode("utf-8") if alpaca else b"# local settings\nTRADING_ENABLED=true\n")
        fields = dict(
            llm_provider="stub",
            alpaca_api_key_id=ALPACA_ID if alpaca else None,
            alpaca_api_secret_key=ALPACA_SECRET if alpaca else None,
            tavily_api_key=TAVILY,
            gemini_api_key=GEMINI,
            anthropic_api_key=None,
            openai_api_key=None,
            sec_user_agent=None,
            trading_enabled=True,
            app_db_path=tmp_path / "app.db",
            ticket_ttl_sec=300,
            api_allow_remote=False,
        )
        fields.update(settings_over)
        self.settings = Settings(_env_file=None, **fields)
        self.clock = Clock()
        self.data = FakeData()
        self.brokers: list[ClosableBroker] = []
        self.legs = [LIMIT_LEG]
        self.llm = StubLLM(script_factory=self._script)
        self.llm.model = self.settings.llm_model
        self.with_chat = chat
        self.runtime: ChatRuntime | None = None
        self.before_start = None  # optional hook(runtime) run inside the state factory
        self.close_grace_sec = 0.0
        self.llm_checks: list[tuple[str, str, str | None]] = []  # (provider, model, key) the validator saw
        self.llm_built: list[SwitchedLLM] = []                    # clients the factory made
        self.model_requests: list[tuple[str, str]] = []           # (provider, key) the model lister saw
        self.model_lists = {name: list(items) for name, items in MODEL_LISTS.items()}

    # scripted model: one proposal, then a final answer
    def _script(self, user_text):
        args = {
            "legs": [dict(leg) for leg in self.legs],
            "thesis": "Services growth supports the multiple.",
            "risks": "Regulatory pressure.",
            "invalidation": "Services growth below 5%.",
            "evidence_ids": ["E1"],
        }
        return [
            Turn("", [ToolCall("c1", "propose_trades", args)], "tool_use"),
            Turn(FINAL, [], "end_turn"),
        ]

    def broker_factory(self, key_id: str, secret: str) -> ClosableBroker:
        # `paper` as AlpacaPaper.account() derives it from the account number: unknown (None) for
        # the default keys, False for the live-account keys, True for the paper-account keys.
        paper = {LIVE_ID: False, PAPER_ID: True}.get(key_id)
        broker = ClosableBroker(key_id, account=make_account(paper=paper))
        if (key_id, secret) in {(DOWN_ID, DOWN_SECRET)}:
            broker.account_error = BrokerError(503, None, f"service unavailable {secret}")
        elif (key_id, secret) not in {(ALPACA_ID, ALPACA_SECRET), (NEW_ID, NEW_SECRET), (LIVE_ID, LIVE_SECRET),
                                      (PAPER_ID, PAPER_SECRET)}:
            broker.account_error = BrokerError(401, None, f"unauthorized {secret}")
        self.brokers.append(broker)
        return broker

    # the model provider fakes: no network, decided by the key value
    async def validate_llm(self, candidate) -> str:
        provider, model = candidate.llm_provider, candidate.llm_model
        key = getattr(candidate, LLM_KEY_FIELDS[provider], None)
        self.llm_checks.append((provider, model, key))
        if key == LEAKY_LLM_KEY:
            raise ProviderError(None, f"connection reset while sending {key}")
        if key == BAD_LLM_KEY:
            return "invalid_key"
        if key == DOWN_LLM_KEY:
            return "unreachable"
        if model == "no-such-model":
            return "model_unavailable"
        return "rate_limited" if key == LIMITED_LLM_KEY else "ok"

    async def list_models(self, provider: str, api_key: str, settings) -> list[dict]:
        self.model_requests.append((provider, api_key))
        if api_key == BAD_LLM_KEY:
            raise ProviderError(401, f"Incorrect API key provided: {api_key}")
        if api_key == LIMITED_LLM_KEY:
            raise ProviderError(429, f"rate limit reached for {api_key}")
        if api_key in (DOWN_LLM_KEY, LEAKY_LLM_KEY):
            raise ProviderError(None, f"connection reset while sending {api_key}")
        return list(self.model_lists[provider])

    def make_llm(self, settings) -> SwitchedLLM:
        llm = SwitchedLLM(settings.llm_provider, settings.llm_model, settings.llm_effort)
        self.llm_built.append(llm)
        return llm

    def data_factory(self, key_id: str, secret: str, http) -> FakeData:
        return self.data

    def state(self):
        handle = LLMHandle(self.llm)  # as AppState.load does: one handle for /answer and /api
        fin = SimpleNamespace(settings=self.settings, llm=handle, tools=None,
                              orchestrator=FakeOrchestrator(handle), chat=None)
        if self.with_chat:
            async def no_sleep(_seconds):
                return None

            fin.chat = ChatRuntime(
                self.settings,
                llm=handle,
                registry=None,
                orchestrator=fin.orchestrator,
                env_path=self.env_path,
                broker_factory=self.broker_factory,
                data_factory=self.data_factory,
                http_factory=lambda: HttpClient(client=httpx.AsyncClient(transport=httpx.MockTransport(_refuse_all))),
                clock=self.clock,
                sleep=no_sleep,
                close_grace_sec=self.close_grace_sec,
                llm_validator=self.validate_llm,
                llm_models=self.list_models,
                llm_factory=self.make_llm,
            )
            self.runtime = fin.chat
            if self.before_start is not None:
                self.before_start(fin.chat)
        return fin

    @property
    def broker(self) -> ClosableBroker | None:
        return self.runtime.broker if self.runtime is not None else None

    def client(self, **kw) -> TestClient:
        # A real loopback address and Host: the guard has no test-only allowances.
        kw.setdefault("headers", HDR)
        kw.setdefault("base_url", LOCAL_URL)
        kw.setdefault("client", LOCAL_CLIENT)
        return TestClient(create_app(self.state), **kw)

    def add_evidence(self, session_id: str) -> str:
        record = self.runtime.evidence.add(
            session_id, connector="edgar", title="Apple Inc. 10-K", url="https://www.sec.gov/apple-10k",
            publisher="SEC EDGAR", tier=1, published_at="2025-10-31", snippet="10-K filed 2025-10-31",
            untrusted=False,
        )
        return record.id

    def env(self) -> str:
        return self.env_path.read_bytes().decode("utf-8")

    def total_submits(self) -> int:
        return sum(b.submits for b in self.brokers)


@pytest.fixture
def world(tmp_path) -> ApiWorld:
    return ApiWorld(tmp_path)


@pytest.fixture(autouse=True)
def _clean_trading_env(monkeypatch):
    """A shell-exported trading key would show up in `env_fallbacks`; keep every test independent
    of the developer's environment."""
    for name in [*ENV_NAMES.values(), "API_ALLOW_REMOTE"]:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def sse_events(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block or block.startswith(":"):
            continue
        assert block.startswith("data: "), block
        events.append(json.loads(block[len("data: "):]))
    return events


def new_session(client, **body) -> str:
    r = client.post("/api/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def chat(client, session_id: str, message: str = "Buy some Apple for me") -> list[dict]:
    r = client.post("/api/chat", json={"session_id": session_id, "message": message})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return sse_events(r.text)


def propose(client, world: ApiWorld, session_id: str) -> dict:
    world.add_evidence(session_id)
    events = chat(client, session_id)
    proposals = [e for e in events if e["type"] == "data-trade-proposal"]
    assert len(proposals) == 1, [e["type"] for e in events]
    return proposals[0]["ticket"]


def confirm(client, ticket: dict, session_id: str, hash_: str | None = None):
    return client.post(f"/api/tickets/{ticket['id']}/confirm",
                       json={"session_id": session_id, "hash": hash_ or ticket["hash"]})


def assert_no_secrets(response) -> None:
    raw = response.content
    for secret in SECRETS:
        assert secret.encode() not in raw, f"secret leaked in {response.request.url}"


# --- 1. chat stream -----------------------------------------------------------------------------------


def test_chat_stream_parses_ends_with_finish_and_proposes_one_ticket(world):
    with world.client() as client:
        sid = new_session(client)
        world.add_evidence(sid)
        events = chat(client, sid)
        types = [e["type"] for e in events]
        assert types[0] == "start" and types[-1] == "finish"
        assert types.count("finish") == 1
        proposals = [e for e in events if e["type"] == "data-trade-proposal"]
        assert len(proposals) == 1
        ticket = proposals[0]["ticket"]
        assert world.runtime.tickets.active(sid).id == ticket["id"]
        # money is serialized as strings, never floats
        assert ticket["legs"][0]["qty"] == "10" and ticket["legs"][0]["limit_price"] == "200.00"
        assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == FINAL
        assert world.total_submits() == 0


def test_chat_unknown_session_is_404_before_the_stream(world):
    with world.client() as client:
        r = client.post("/api/chat", json={"session_id": "SNOPE", "message": "hi"})
        assert r.status_code == 404
        assert r.json()["error"] == "session_not_found"


def test_chat_request_is_validated(world):
    with world.client() as client:
        sid = new_session(client)
        assert client.post("/api/chat", json={"session_id": sid}).status_code == 422
        assert client.post("/api/chat", json={"session_id": sid, "message": "x", "extra": 1}).status_code == 422
        assert client.post("/api/chat", json={"session_id": sid, "message": "x" * 20001}).status_code == 422


# --- 2-6. ticket actions ------------------------------------------------------------------------------


def test_confirm_with_a_wrong_hash_is_409_and_submits_nothing(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        r = confirm(client, ticket, sid, "0" * 64)
        assert r.status_code == 409
        body = r.json()
        assert body["status"] == "rejected" and "changed" in body["message"]
        assert set(body) == {"status", "message", "ticket", "orders", "new_ticket"}
        assert world.total_submits() == 0


def test_confirm_with_trading_disabled_is_423(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.settings.trading_enabled = False
        r = confirm(client, ticket, sid)
        assert r.status_code == 423 and r.json()["status"] == "disabled"
        assert world.total_submits() == 0


def test_confirm_success_is_200_and_places_the_order_on_the_broker(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        r = confirm(client, ticket, sid)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "submitted"
        assert body["ticket"]["status"] == "submitted"
        assert [cid for _leg, cid in world.broker.submitted] == [f"fa-{ticket['id']}-0"]
        assert body["orders"][0]["client_order_id"] == f"fa-{ticket['id']}-0"
        assert body["orders"][0]["qty"] == "10"
        # a replay never reaches the broker again
        again = confirm(client, ticket, sid)
        assert again.status_code == 409
        assert world.broker.submits == 1
        # polling view
        r = client.get(f"/api/tickets/{ticket['id']}", params={"session_id": sid})
        assert r.status_code == 200
        polled = r.json()
        assert polled["ticket"]["status"] == "submitted"
        assert polled["orders"][0]["client_order_id"] == f"fa-{ticket['id']}-0"
        assert polled["orders"][0]["order"]["status"] == "accepted"
        assert_no_secrets(r)


def test_confirm_of_an_expired_ticket_is_410(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.clock.advance(301)
        r = confirm(client, ticket, sid)
        assert r.status_code == 410
        assert r.json()["ticket"]["status"] == "expired"
        assert world.total_submits() == 0


def test_confirm_from_another_session_is_404(world):
    with world.client() as client:
        sid = new_session(client)
        other = new_session(client)
        ticket = propose(client, world, sid)
        r = confirm(client, ticket, other)
        assert r.status_code == 404 and r.json()["status"] == "not_found"
        assert r.json()["ticket"] is None
        assert client.get(f"/api/tickets/{ticket['id']}", params={"session_id": other}).status_code == 404
        assert confirm(client, {"id": "TNOPE", "hash": ticket["hash"]}, sid).status_code == 404
        assert world.total_submits() == 0


def test_confirm_requote_is_409_with_the_new_ticket(world):
    world.legs = [MARKET_LEG]
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.data.set_price("AAPL", "204")  # a 2% move on a market order
        r = confirm(client, ticket, sid)
        assert r.status_code == 409
        body = r.json()
        assert body["status"] == "requote" and body["new_ticket"]["status"] == "proposed"
        assert body["ticket"]["status"] == "superseded"
        assert world.total_submits() == 0


def test_confirm_without_a_broker_is_503(tmp_path):
    world = ApiWorld(tmp_path, alpaca=False)
    with world.client() as client:
        sid = new_session(client)
        r = client.post("/api/tickets/TABC/confirm", json={"session_id": sid, "hash": "0" * 64})
        assert r.status_code == 503 and r.json()["error"] == "broker_not_connected"


def test_reject_and_refresh(world):
    world.legs = [MARKET_LEG]
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.data.set_price("AAPL", "201")
        r = client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid})
        assert r.status_code == 200, r.text
        fresh = r.json()["new_ticket"]
        assert r.json()["status"] == "requote" and fresh["status"] == "proposed"
        # the old one is superseded: refreshing it again is a state conflict
        assert client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid}).status_code == 409
        r = client.post(f"/api/tickets/{fresh['id']}/reject", json={"session_id": sid})
        assert r.status_code == 200 and r.json()["ticket"]["status"] == "rejected_by_user"
        # rejecting again is idempotent
        assert client.post(f"/api/tickets/{fresh['id']}/reject", json={"session_id": sid}).status_code == 200
        assert client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": sid}).status_code == 409
        assert client.post(f"/api/tickets/{fresh['id']}/reject", json={"session_id": "SOTHER"}).status_code == 404
        assert world.total_submits() == 0


def test_refresh_failure_is_502_without_exception_text(world, caplog):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)

        async def boom(ticket_id, session_id):
            raise RuntimeError(f"kaboom {ALPACA_SECRET}")

        world.runtime.executor.refresh = boom
        r = client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid})
        assert r.status_code == 502 and r.json()["error"] == "refresh_failed"
        assert b"kaboom" not in r.content
        assert_no_secrets(r)
        assert "kaboom" not in caplog.text and ALPACA_SECRET not in caplog.text


def test_ticket_actions_run_expire_stale_first(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.clock.advance(301)
        r = client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": sid})
        assert r.status_code == 410
        assert world.runtime.tickets.get(ticket["id"]).status == TicketStatus.expired


# --- sessions, mandate, portfolio, ledger ---------------------------------------------------------------


def test_sessions_crud_and_detail(world):
    with world.client() as client:
        r = client.post("/api/sessions", json={"title": "Dividends", "lang": "ko"})
        assert r.status_code == 201
        sid = r.json()["id"]
        assert r.json()["lang"] == "ko"
        assert client.post("/api/sessions", json={"lang": "fr"}).status_code == 422
        listed = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in listed] == [sid]
        r = client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"})
        assert r.status_code == 200 and r.json()["title"] == "Renamed"
        assert client.patch("/api/sessions/SNOPE", json={"title": "x"}).status_code == 404
        assert client.get("/api/sessions/SNOPE").status_code == 404

        ticket = propose(client, world, sid)
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["session"]["title"] == "Renamed"
        assert detail["active_ticket"]["id"] == ticket["id"]
        assert [m["role"] for m in detail["messages"]][:2] == ["user", "assistant"]
        assert detail["mandate"] == Mandate().model_dump(mode="json")
        assert detail["mandate_saved"] is False

        world.clock.advance(301)
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["active_ticket"] is None
        assert world.runtime.tickets.get(ticket["id"]).status == TicketStatus.expired


def test_post_sessions_accepts_an_empty_body(world):
    with world.client() as client:
        r = client.post("/api/sessions")
        assert r.status_code == 201 and r.json()["title"] == "New chat"


def test_mandate_get_put_versions_and_ledger(world):
    with world.client() as client:
        sid = new_session(client)
        r = client.get(f"/api/mandate/{sid}")
        assert r.status_code == 200
        assert r.json() == {"mandate": Mandate().model_dump(mode="json"), "saved": False}

        r = client.put(f"/api/mandate/{sid}", json={"budget_usd": "20000", "max_positions": 5})
        assert r.status_code == 200, r.text
        assert r.json()["saved"] is True
        assert r.json()["mandate"]["budget_usd"] == "20000" and r.json()["mandate"]["version"] == 1
        r = client.put(f"/api/mandate/{sid}", content=b'{"budget_usd": 20000.5}',
                       headers={**HDR, "content-type": "application/json"})
        assert r.status_code == 200 and r.json()["mandate"]["budget_usd"] == "20000.5"
        assert r.json()["mandate"]["version"] == 2
        assert client.get(f"/api/mandate/{sid}").json()["saved"] is True

        kinds = [json.loads(line)["kind"] for line in world.runtime.ledger.export_jsonl(sid)]
        assert kinds.count("mandate_saved") == 2

        bad = client.put(f"/api/mandate/{sid}", json={"max_position_pct": "150", "bogus": 1})
        assert bad.status_code == 422
        fields = {f["loc"] for f in bad.json()["fields"]}
        assert {"max_position_pct", "bogus"} <= fields
        assert client.put(f"/api/mandate/{sid}", content=b"not json",
                          headers={**HDR, "content-type": "application/json"}).status_code == 422
        assert client.put("/api/mandate/SNOPE", json={}).status_code == 404
        assert client.get("/api/mandate/SNOPE").status_code == 404


def test_portfolio_connected_disconnected_and_broker_error(tmp_path, world):
    with world.client() as client:
        r = client.get("/api/portfolio")
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        assert body["account"]["equity"] == "100000"
        assert [p["symbol"] for p in body["positions"]] == ["MSFT"]
        assert body["open_orders"] == []
        world.broker.account_error = BrokerError(500, None, f"exploded {ALPACA_SECRET}")
        r = client.get("/api/portfolio")
        assert r.status_code == 502
        assert r.json()["connected"] is True and r.json()["error"] == "broker_unavailable"
        assert set(r.json()) == {"connected", "error", "message"}
        assert b"exploded" not in r.content
        assert_no_secrets(r)

    (tmp_path / "x").mkdir()
    other = ApiWorld(tmp_path / "x", alpaca=False)
    with other.client() as client:
        assert client.get("/api/portfolio").json() == {"connected": False}


def test_ledger_export_streams_jsonl_for_the_session(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        assert confirm(client, ticket, sid).status_code == 200
        r = client.get(f"/api/ledger/{sid}.jsonl")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert "attachment" in r.headers["content-disposition"]
        lines = [json.loads(line) for line in r.text.splitlines() if line]
        kinds = [line["kind"] for line in lines]
        assert "confirm_request" in kinds and "result" in kinds
        assert all(line["session_id"] == sid for line in lines)
        assert_no_secrets(r)
        assert client.get("/api/ledger/SNOPE.jsonl").status_code == 404


# --- 7-9. settings and kill switch ------------------------------------------------------------------------


def test_settings_get_reports_status_and_never_secrets(world):
    with world.client() as client:
        r = client.get("/api/settings")
        assert r.status_code == 200
        body = r.json()
        assert body == {
            "alpaca": "connected",
            "tavily": "set",
            "sec_user_agent": "missing",
            "gemini": "set",
            "trading_enabled": True,
            "model": world.settings.llm_model,
            "effort": world.settings.llm_effort,
            "api_allow_remote": False,
            "env_fallbacks": [],
            # Task 3.8: the model provider and the research depth (the test world runs the stub provider)
            "provider": "stub",
            "mode": "deep",
            "max_rounds": 12,
            "time_budget_sec": 300,
            "gemini_paid": False,
            "providers": {"gemini": "set", "openai": "missing", "anthropic": "missing"},
        }
        assert_no_secrets(r)


def test_settings_alpaca_status_invalid_and_missing(tmp_path):
    bad = ApiWorld(tmp_path, alpaca_api_key_id=BAD_ID, alpaca_api_secret_key=BAD_SECRET)
    with bad.client() as client:
        r = client.get("/api/settings")
        assert r.json()["alpaca"] == "invalid"
        assert_no_secrets(r)
    (tmp_path / "m").mkdir()
    missing = ApiWorld(tmp_path / "m", alpaca=False)
    with missing.client() as client:
        assert client.get("/api/settings").json()["alpaca"] == "missing"


def test_put_settings_with_invalid_alpaca_keys_writes_nothing(world):
    with world.client() as client:
        before = world.env()
        r = client.put("/api/settings", json={"alpaca_key_id": BAD_ID, "alpaca_secret": BAD_SECRET})
        assert r.status_code == 400 and r.json()["error"] == "alpaca_invalid"
        assert_no_secrets(r)
        assert world.env() == before
        assert world.settings.alpaca_api_key_id == ALPACA_ID
        assert world.brokers[-1].key_id == BAD_ID and world.brokers[-1].closed

        r = client.put("/api/settings", json={"alpaca_key_id": DOWN_ID, "alpaca_secret": DOWN_SECRET})
        assert r.status_code == 502 and r.json()["error"] == "alpaca_unreachable"
        assert_no_secrets(r)
        assert world.env() == before
        assert world.brokers[-1].closed


def test_put_settings_alpaca_keys_must_come_together(world):
    with world.client() as client:
        before = world.env()
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID})
        assert r.status_code == 400 and r.json()["error"] == "alpaca_pair_required"
        r = client.put("/api/settings", json={"alpaca_key_id": "", "alpaca_secret": NEW_SECRET})
        assert r.status_code == 400
        assert world.env() == before
        assert_no_secrets(r)


def test_put_settings_valid_keys_write_in_place_and_rebuild(tmp_path):
    world = ApiWorld(tmp_path, alpaca=False)
    world.env_path.write_bytes(b"# keep me\nLLM_PROVIDER=stub\nTRADING_ENABLED=true\n")
    with world.client() as client:
        assert world.runtime.broker is None
        assert client.get("/api/portfolio").json() == {"connected": False}
        r = client.put("/api/settings", json={
            "alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET,
            "sec_user_agent": "Jane Doe jane@example.com",
        })
        assert r.status_code == 200, r.text
        assert r.json()["alpaca"] == "connected" and r.json()["sec_user_agent"] == "set"
        assert_no_secrets(r)
        assert world.env() == (
            "# keep me\nLLM_PROVIDER=stub\nTRADING_ENABLED=true\n"
            f"ALPACA_API_KEY_ID={NEW_ID}\nALPACA_API_SECRET_KEY={NEW_SECRET}\n"
            'SEC_USER_AGENT="Jane Doe jane@example.com"\n'
        )
        assert world.settings.alpaca_api_key_id == NEW_ID
        assert world.runtime.broker is not None and world.runtime.broker.key_id == NEW_ID
        assert world.runtime.executor is not None and world.runtime.edgar is not None
        assert client.get("/api/portfolio").json()["connected"] is True
        names = {spec.name for spec in world.runtime.toolbox.specs()}
        assert "propose_trades" in names and "sec_filings" in names


def test_put_settings_empty_string_removes_a_key(world):
    with world.client() as client:
        assert "web_search" in {s.name for s in world.runtime.toolbox.specs()}
        r = client.put("/api/settings", json={"tavily_key": ""})
        assert r.status_code == 200 and r.json()["tavily"] == "missing"
        assert "TAVILY_API_KEY" not in world.env()
        assert "# research\n" in world.env()
        assert world.settings.tavily_api_key is None
        assert "web_search" not in {s.name for s in world.runtime.toolbox.specs()}


def test_put_settings_old_clients_are_closed_after_the_grace_delay(world):
    with world.client() as client:
        old = world.runtime.broker
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 200
        assert world.runtime.broker is not old
        for _ in range(100):  # the zero-second grace task runs on the app's loop thread
            if old.closed:
                break
            time.sleep(0.01)
        assert old.closed


def test_put_settings_rejects_unsafe_values_without_echoing_them(world):
    with world.client() as client:
        before = world.env()
        r = client.put("/api/settings", json={"tavily_key": TAVILY + "\nEVIL=1"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_value"
        assert_no_secrets(r)
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET * 20})
        assert r.status_code == 422
        assert_no_secrets(r)
        r = client.put("/api/settings", content=f'{{"tavily_key": "{TAVILY}", "x": }}'.encode(),
                       headers={**HDR, "content-type": "application/json"})
        assert r.status_code == 422
        assert_no_secrets(r)
        r = client.put("/api/settings", json={"tavily_key": TAVILY, "unknown": NEW_SECRET})
        assert r.status_code == 422
        assert_no_secrets(r)
        r = client.put("/api/settings", json={"trading_enabled": "yes"})
        assert r.status_code == 422
        assert world.env() == before
        assert world.total_submits() == 0


def test_kill_switch_disables_persists_and_blocks_confirm(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        r = client.post("/api/kill-switch", json={"suspend_broker": True})
        assert r.status_code == 200
        assert r.json()["trading_enabled"] is False
        assert world.settings.trading_enabled is False
        assert "TRADING_ENABLED=false\n" in world.env()
        assert "TRADING_ENABLED=true" not in world.env()
        assert world.broker.suspend_calls == [True]
        assert client.get("/api/settings").json()["trading_enabled"] is False
        assert confirm(client, ticket, sid).status_code == 423
        kinds = [json.loads(line)["kind"] for line in world.runtime.ledger.export_jsonl("system")]
        assert "kill_switch" in kinds

        r = client.put("/api/settings", json={"trading_enabled": True})
        assert r.status_code == 200 and r.json()["trading_enabled"] is True
        assert "TRADING_ENABLED=true\n" in world.env()
        assert world.total_submits() == 0


def test_kill_switch_without_a_broker(tmp_path):
    world = ApiWorld(tmp_path, alpaca=False)
    with world.client() as client:
        r = client.post("/api/kill-switch", json={"suspend_broker": True})
        assert r.status_code == 200
        assert r.json()["trading_enabled"] is False and r.json()["broker_connected"] is False
        assert world.settings.trading_enabled is False
        assert "TRADING_ENABLED=false\n" in world.env()


def test_no_endpoint_but_confirm_reaches_submit_order(world):
    world.legs = [MARKET_LEG]
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        client.get(f"/api/sessions/{sid}")
        client.get(f"/api/tickets/{ticket['id']}", params={"session_id": sid})
        client.get("/api/portfolio")
        client.get(f"/api/mandate/{sid}")
        client.put(f"/api/mandate/{sid}", json={"max_positions": 3})
        world.data.set_price("AAPL", "201")
        r = client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid})
        fresh = r.json()["new_ticket"]
        client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        client.get(f"/api/ledger/{sid}.jsonl")
        client.post(f"/api/tickets/{fresh['id']}/reject", json={"session_id": sid})
        client.post("/api/kill-switch", json={"suspend_broker": True})
        assert world.total_submits() == 0


# --- 10. security ------------------------------------------------------------------------------------------


def test_non_loopback_client_is_refused_but_answer_is_unaffected(world):
    with world.client(client=("203.0.113.9", 40000)) as client:
        r = client.get("/api/sessions")
        assert r.status_code == 403 and r.json()["error"] == "local_only"
        assert client.post("/api/sessions", json={}).status_code == 403
        r = client.get("/answer", params={"question_id": "Q-001", "question": "hello"})
        assert r.status_code == 200
        assert set(r.json()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "127.0.0.2"])
def test_loopback_clients_are_allowed(world, host):
    with world.client(client=(host, 40000)) as client:
        assert client.get("/api/sessions").status_code == 200


def test_api_allow_remote_opens_the_loopback_rule(tmp_path):
    world = ApiWorld(tmp_path, api_allow_remote=True)
    with world.client(client=("203.0.113.9", 40000)) as client:
        assert client.get("/api/sessions").status_code == 200
        assert client.get("/api/sessions", headers={"host": "evil.example"}).status_code == 400


def test_host_allowlist(world):
    with world.client() as client:
        r = client.get("/api/sessions", headers={"host": "evil.example"})
        assert r.status_code == 400 and r.json()["error"] == "bad_host"
        assert client.get("/api/sessions", headers={"host": "evil.example:8000"}).status_code == 400
        for host in ("127.0.0.1:8000", "localhost:8000", "[::1]:8000", "LOCALHOST", "127.0.0.1"):
            assert client.get("/api/sessions", headers={"host": host}).status_code == 200, host
        assert client.get("/answer", params={"question": "x"}, headers={"host": "evil.example"}).status_code == 200


def test_state_changing_requests_need_the_client_header(world):
    with world.client(headers={}) as client:
        r = client.post("/api/sessions", json={})
        assert r.status_code == 403 and r.json()["error"] == "csrf_header_missing"
        assert client.post("/api/sessions", json={}, headers={"X-FinAgent-Client": "0"}).status_code == 403
        assert client.put("/api/settings", json={"trading_enabled": False}).status_code == 403
        assert world.settings.trading_enabled is True
        assert client.post("/api/kill-switch", json={"suspend_broker": False}).status_code == 403
        assert world.settings.trading_enabled is True
        assert client.get("/api/sessions").status_code == 200  # reads need no header
        assert client.post("/api/sessions", json={}, headers=HDR).status_code == 201


def test_a_mismatched_origin_is_refused(world):
    with world.client() as client:
        r = client.post("/api/sessions", json={}, headers={"origin": "http://evil.example"})
        assert r.status_code == 403 and r.json()["error"] == "origin_mismatch"
        assert client.post("/api/sessions", json={}, headers={"origin": "null"}).status_code == 403
        assert client.post("/api/sessions", json={},
                           headers={"origin": "http://127.0.0.1.evil.example"}).status_code == 403
        assert client.post("/api/sessions", json={}, headers={"origin": "http://127.0.0.1"}).status_code == 201
        assert client.post("/api/sessions", json={}, headers={"origin": "https://127.0.0.1"}).status_code == 201
        assert client.post("/api/sessions", json={}, headers={
            "origin": "http://127.0.0.1:8000", "host": "127.0.0.1:8000"}).status_code == 201


def test_api_responses_forbid_framing_and_sniffing(world):
    with world.client() as client:
        r = client.get("/api/sessions")
        assert r.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert r.headers["x-content-type-options"] == "nosniff"
        assert "access-control-allow-origin" not in r.headers


# --- 11. startup sweep ----------------------------------------------------------------------------------------


def _stuck_ticket(runtime: ChatRuntime, session_id: str) -> str:
    ctx = MarketContext(
        account=make_account(), positions={}, open_orders=[],
        assets={"AAPL": make_asset("AAPL")}, quotes={"AAPL": make_quote("AAPL", "200")},
        clock=make_clock(), orders_today=0,
    )
    leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("1"), type="limit", limit_price=Decimal("200"))
    ticket = runtime.tickets.create(
        session_id, decisions=evaluate([leg], ctx, Mandate()), thesis="t", risks="r",
        invalidation="i", evidence_ids=[], mandate_version=1, quote_as_of=FIXED_NOW,
    )
    runtime.tickets.transition(ticket.id, {TicketStatus.proposed}, TicketStatus.submitting)
    return ticket.id


def test_startup_sweep_reconciles_stuck_tickets_and_expires_stale_ones(world):
    ids = {}

    def seed(runtime: ChatRuntime):
        s1 = runtime.sessions.create()["id"]
        s2 = runtime.sessions.create()["id"]
        ids["found"] = _stuck_ticket(runtime, s1)
        ids["missing"] = _stuck_ticket(runtime, s2)
        found = runtime.tickets.get(ids["found"])
        runtime.broker.place(found.legs[0], found.client_order_ids[0])
        stale = runtime.tickets.create(
            s1, decisions=runtime.tickets.get(ids["found"]).decisions, thesis="t", risks="r",
            invalidation="i", evidence_ids=[], mandate_version=1, quote_as_of=FIXED_NOW, ttl_sec=-1,
        )
        ids["stale"] = stale.id

    world.before_start = seed
    with world.client():
        tickets = world.runtime.tickets
        assert tickets.get(ids["found"]).status == TicketStatus.submitted
        assert tickets.get(ids["missing"]).status == TicketStatus.failed
        assert tickets.get(ids["stale"]).status == TicketStatus.expired
        assert world.total_submits() == 0


def test_startup_without_a_broker_leaves_stuck_tickets(tmp_path, caplog):
    world = ApiWorld(tmp_path, alpaca=False)
    ids = {}

    def seed(runtime: ChatRuntime):
        ids["t"] = _stuck_ticket(runtime, runtime.sessions.create()["id"])

    world.before_start = seed
    with caplog.at_level("WARNING"), world.client():
        assert world.runtime.tickets.get(ids["t"]).status == TicketStatus.submitting
    assert "submitting" in caplog.text


# --- 12-14. contracts, disconnects, no chat runtime -----------------------------------------------------------


def test_answer_still_returns_the_five_fields(world):
    with world.client() as client:
        r = client.get("/answer", params={"question_id": "Q-001", "question": "hello", "foo": "bar"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
        assert body["answer"] == "answer to hello"
        assert body["think_trace"] == f"trace by stub/{world.settings.llm_model}"


def test_api_is_503_without_a_chat_runtime(tmp_path):
    world = ApiWorld(tmp_path, chat=False)
    with world.client() as client:
        for method, path in (("get", "/api/sessions"), ("get", "/api/settings"), ("post", "/api/sessions"),
                             ("get", "/api/portfolio"), ("post", "/api/kill-switch")):
            r = getattr(client, method)(path) if method == "get" else client.post(path, json={})
            assert r.status_code == 503 and r.json()["error"] == "chat_unavailable", path
        assert client.get("/answer", params={"question": "x"}).status_code == 200


def test_root_serves_the_web_app_when_present(tmp_path, world):
    web = tmp_path / "webapp"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>FinAgent</title>", encoding="utf-8")
    (web / "app.css").write_text("body{}", encoding="utf-8")
    with TestClient(create_app(world.state, web_app_dir=web)) as client:
        r = client.get("/")
        assert r.status_code == 200 and "FinAgent" in r.text
        assert r.headers["content-type"].startswith("text/html")
        assert r.headers["x-frame-options"] == "DENY"
        r = client.get("/app/app.css")
        assert r.status_code == 200 and r.text == "body{}"
        assert r.headers["x-frame-options"] == "DENY"
        assert client.get("/ui").status_code in (200, 404)


def test_root_keeps_the_plain_text_line_without_the_web_app(tmp_path, world):
    with TestClient(create_app(world.state, web_app_dir=tmp_path / "absent")) as client:
        r = client.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        assert "GET /answer" in r.text
        assert client.get("/app/app.css").status_code == 404


async def test_event_stream_disconnect_cancels_the_turn():
    cancelled = asyncio.Event()

    async def run(emit):
        emit({"type": "start", "messageId": "m1"})
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stream = event_stream(run, keepalive_sec=5)
    first = await stream.__anext__()
    assert first == 'data: {"type": "start", "messageId": "m1"}\n\n'
    await stream.aclose()  # what the server does when the client goes away
    await asyncio.wait_for(cancelled.wait(), 1)


async def test_event_stream_sends_keepalives_and_ends_after_finish():
    release = asyncio.Event()

    async def run(emit):
        await release.wait()
        emit({"type": "data-x", "amount": Decimal("1.50"), "at": FIXED_NOW, "text": "한국어"})
        emit({"type": "finish", "usage": {}, "elapsed_ms": 1})
        emit({"type": "late"})  # anything after finish is not sent

    stream = event_stream(run, keepalive_sec=0.01)
    assert await stream.__anext__() == ": keep-alive\n\n"
    release.set()
    rest = [chunk async for chunk in stream if not chunk.startswith(":")]
    events = [json.loads(chunk[len("data: "):]) for chunk in rest]
    assert events[0] == {"type": "data-x", "amount": "1.50", "at": FIXED_NOW.isoformat(), "text": "한국어"}
    assert "한국어" in rest[0]  # ensure_ascii=False
    assert [e["type"] for e in events] == ["data-x", "finish"]


async def test_event_stream_closes_a_turn_that_ends_without_finish_or_raises(caplog):
    async def quiet(emit):
        emit({"type": "start", "messageId": "m"})

    async def broken(emit):
        raise RuntimeError(f"provider said {ALPACA_SECRET}")

    for run in (quiet, broken):
        chunks = [chunk async for chunk in event_stream(run, keepalive_sec=5)]
        events = [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ")]
        assert events[-1]["type"] == "finish"
        assert events[-2]["type"] == "error"
        assert ALPACA_SECRET not in "".join(chunks)
    assert ALPACA_SECRET not in caplog.text


async def test_event_stream_survives_an_unserializable_event():
    loop_ = {"type": "data-x"}
    loop_["self"] = loop_  # circular: json.dumps raises ValueError

    async def run(emit):
        emit(loop_)
        emit({"type": "finish", "usage": {}, "elapsed_ms": 1})

    chunks = [chunk async for chunk in event_stream(run, keepalive_sec=5)]
    events = [json.loads(c[len("data: "):]) for c in chunks]
    assert events == [{"type": "data-x"}, {"type": "finish", "usage": {}, "elapsed_ms": 1}]


def test_clearing_the_alpaca_keys_drops_the_broker_but_keeps_ticket_history(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        assert confirm(client, ticket, sid).status_code == 200
        r = client.put("/api/settings", json={"alpaca_key_id": "", "alpaca_secret": ""})
        assert r.status_code == 200 and r.json()["alpaca"] == "missing"
        assert "ALPACA_API_KEY_ID" not in world.env() and "ALPACA_API_SECRET_KEY" not in world.env()
        assert world.runtime.broker is None and world.runtime.executor is None
        assert "propose_trades" not in {s.name for s in world.runtime.toolbox.specs()}
        r = client.get(f"/api/tickets/{ticket['id']}", params={"session_id": sid})
        assert r.status_code == 200
        assert r.json()["orders"][0]["order"]["status"] == "accepted"
        assert client.get(f"/api/tickets/{ticket['id']}", params={"session_id": "SOTHER"}).status_code == 404
        assert client.get("/api/portfolio").json() == {"connected": False}


def test_the_rebuilt_ledger_still_scrubs_the_replaced_keys(world):
    with world.client() as client:
        assert client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET}).status_code == 200
        sid = new_session(client)
        world.runtime.ledger.append(sid, "note", {"text": f"old {ALPACA_SECRET} new {NEW_SECRET}"})
        exported = "".join(world.runtime.ledger.export_jsonl(sid))
        assert ALPACA_SECRET not in exported and NEW_SECRET not in exported


# --- fix round 1 ----------------------------------------------------------------------------------------------

# I1: Unicode line separators and non-ASCII characters never reach the env file.


@pytest.mark.parametrize("ch", ["\x85", "\u2028", "\u2029", "\x0b", "\x0c", "é"])
def test_put_settings_refuses_non_ascii_and_line_separators(world, ch):
    with world.client() as client:
        before = world.env()
        brokers = len(world.brokers)
        for body, field in (
            ({"sec_user_agent": f"Jane jane@example.com{ch}API_ALLOW_REMOTE=true #"}, "sec_user_agent"),
            ({"tavily_key": f"tvly-{ch}x"}, "tavily_key"),
            ({"alpaca_key_id": f"PK{ch}X", "alpaca_secret": NEW_SECRET}, "alpaca_key_id"),
            ({"alpaca_key_id": NEW_ID, "alpaca_secret": f"{NEW_SECRET}{ch}"}, "alpaca_secret"),
        ):
            r = client.put("/api/settings", json=body)
            assert r.status_code == 400, (body, r.text)
            assert r.json()["error"] == "invalid_value" and r.json()["field"] == field
            assert_no_secrets(r)
        assert world.env() == before
        assert len(world.brokers) == brokers  # refused before any live key check


def test_put_settings_two_writes_never_create_a_new_key(world):
    value = "Jane jane@example.com API_ALLOW_REMOTE=true #"
    with world.client() as client:
        assert client.put("/api/settings", json={"sec_user_agent": value}).status_code == 200
        assert client.put("/api/settings", json={"trading_enabled": False}).status_code == 200
        reread = Settings(_env_file=world.env_path)
        assert reread.api_allow_remote is False
        assert reread.sec_user_agent == value
        assert reread.trading_enabled is False


def test_put_settings_strips_only_ascii_whitespace(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"tavily_key": f"  {TAVILY}\n"})
        assert r.status_code == 200 and f"TAVILY_API_KEY={TAVILY}\n" in world.env()
        r = client.put("/api/settings", json={"tavily_key": f"{TAVILY}\u2028"})
        assert r.status_code == 400 and r.json()["field"] == "tavily_key"


# M1: no test-only allowances in production.


def test_testclient_and_testserver_are_not_allowed(world):
    assert is_loopback("testclient") is False
    assert parse_host("testserver") == "testserver"
    with TestClient(create_app(world.state), headers=HDR) as client:  # client "testclient", Host "testserver"
        r = client.get("/api/sessions")
        assert r.status_code == 403 and r.json()["error"] == "local_only"
        assert client.get("/answer", params={"question": "x"}).status_code == 200
    with world.client(base_url="http://testserver") as client:
        r = client.get("/api/sessions")
        assert r.status_code == 400 and r.json()["error"] == "bad_host"


# M2: uvicorn never trusts X-Forwarded-For.


def test_main_binds_loopback_without_proxy_headers(monkeypatch):
    import uvicorn

    import finagent.api as api_module

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("FINAGENT_HOST", raising=False)
    api_module.main()
    assert calls[-1]["host"] == "127.0.0.1" and calls[-1]["proxy_headers"] is False
    monkeypatch.setenv("FINAGENT_HOST", "0.0.0.0")
    api_module.main()
    assert calls[-1]["host"] == "0.0.0.0" and calls[-1]["proxy_headers"] is False


# M3: strict Host parsing.


@pytest.mark.parametrize("header, expected", [
    ("127.0.0.1", "127.0.0.1"), ("127.0.0.1:8000", "127.0.0.1"), ("LOCALHOST:80", "localhost"),
    ("[::1]", "[::1]"), ("[::1]:8000", "[::1]"), ("example.com:443", "example.com"),
    ("[::1]evil", None), ("[::1]:80x", None), ("[::1]:", None), ("localhost:", None),
    ("localhost:8000:1", None), ("localhost:123456", None), ("user@localhost", None),
    ("localhost/x", None), ("local host", None), ("", None), ("[::1", None), ("[]", None),
    (":8000", None), ("localhost:-1", None), ("[::1]]:80", None), ("[x::1]", None),
])
def test_parse_host_is_strict(header, expected):
    assert parse_host(header) == expected


@pytest.mark.parametrize("host", ["[::1]evil", "[::1]:80x", "localhost:", "localhost:8000:1", "user@localhost"])
def test_malformed_host_headers_are_refused(world, host):
    with world.client() as client:
        r = client.get("/api/sessions", headers={"host": host})
        assert r.status_code == 400 and r.json()["error"] == "bad_host"


# M4: the one-proposal-per-turn slot survives a rebuild.


async def test_a_rebuild_mid_turn_keeps_the_one_proposal_per_turn_slot(tmp_path):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        sid = runtime.sessions.create()["id"]
        world.add_evidence(sid)
        args = world._script("x")[0].tool_calls[0].input
        first = await runtime.toolbox.execute(sid, "propose_trades", args, turn_id="turn-1")
        assert not first.is_error, first.text
        old_toolbox = runtime.toolbox
        runtime.rebuild()
        assert runtime.toolbox is not old_toolbox
        second = await runtime.toolbox.execute(sid, "propose_trades", args, turn_id="turn-1")
        assert second.is_error and "one trade proposal per turn" in second.text
        assert runtime.db.query("SELECT COUNT(*) AS n FROM tickets")[0]["n"] == 1
        third = await runtime.toolbox.execute(sid, "propose_trades", args, turn_id="turn-2")
        assert not third.is_error  # a new turn may propose again
    finally:
        await runtime.shutdown()


# M6: .env wins over the process environment; a setting that only the environment provides is
# reported by name (clearing it in the app would not survive a restart).


def test_environment_only_settings_are_warned_by_name_and_reported(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TRADING_ENABLED", "false")  # also in the env file: the file wins
    monkeypatch.setenv("SEC_USER_AGENT", SEC_UA_FROM_SHELL)  # only in the environment
    world = ApiWorld(tmp_path)
    with caplog.at_level("WARNING"), world.client() as client:
        r = client.get("/api/settings")
    assert r.json()["env_fallbacks"] == ["SEC_USER_AGENT"]
    assert "env_overrides" not in r.json()
    assert_no_secrets(r)
    warned = [rec.getMessage() for rec in caplog.records if "process environment" in rec.getMessage()]
    assert any("SEC_USER_AGENT" in m for m in warned)
    assert not any("TRADING_ENABLED" in m for m in warned)
    assert SEC_UA_FROM_SHELL not in caplog.text and SEC_UA_FROM_SHELL not in r.text


# M7: a failed rebuild or env write changes nothing.


def test_a_failed_rebuild_changes_nothing(world, caplog):
    def data_factory(key_id, secret, http):
        if key_id == NEW_ID:
            raise RuntimeError(f"boom {NEW_SECRET}")
        return world.data

    world.data_factory = data_factory
    with world.client() as client:
        before, old_broker, old_toolbox = world.env(), world.runtime.broker, world.runtime.toolbox
        r = client.put("/api/settings", json={
            "alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET, "tavily_key": "", "trading_enabled": False,
        })
        assert r.status_code == 500 and r.json()["error"] == "settings_not_saved"
        assert_no_secrets(r)
        assert world.env() == before
        assert world.settings.alpaca_api_key_id == ALPACA_ID
        assert world.settings.alpaca_api_secret_key == ALPACA_SECRET
        assert world.settings.tavily_api_key == TAVILY and world.settings.trading_enabled is True
        assert world.runtime.broker is old_broker and world.runtime.toolbox is old_toolbox
        assert client.get("/api/settings").json()["alpaca"] == "connected"
    assert "boom" not in caplog.text and NEW_SECRET not in caplog.text


def test_a_failed_env_write_changes_nothing_and_discards_the_new_clients(world, monkeypatch):
    import finagent.app_state as app_state

    def boom(path, updates):
        raise OSError("disk full")

    with world.client() as client:
        old_broker = world.runtime.broker
        monkeypatch.setattr(app_state, "update_env_file", boom)
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 500 and r.json()["error"] == "settings_not_saved"
        assert world.settings.alpaca_api_key_id == ALPACA_ID
        assert world.runtime.broker is old_broker
        built = [b for b in world.brokers if b.key_id == NEW_ID]
        assert len(built) == 2  # the live check, then the discarded build
        for _ in range(100):
            if all(b.closed for b in built):
                break
            time.sleep(0.01)
        assert all(b.closed for b in built)


# M11: reject is a local transition and works without a broker.


def test_reject_works_without_a_broker_but_confirm_and_refresh_do_not(world):
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        assert client.put("/api/settings", json={"alpaca_key_id": "", "alpaca_secret": ""}).status_code == 200
        assert world.runtime.executor is None
        r = confirm(client, ticket, sid)
        assert r.status_code == 503 and r.json()["error"] == "broker_not_connected"
        r = client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid})
        assert r.status_code == 503
        assert client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": "SOTHER"}).status_code == 404
        r = client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": sid})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "rejected" and r.json()["ticket"]["status"] == "rejected_by_user"
        assert client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": sid}).status_code == 200
        kinds = [json.loads(line)["kind"] for line in world.runtime.ledger.export_jsonl(sid)]
        assert kinds.count("reject") == 2  # the foreign-session attempt is logged under its own id
        assert world.total_submits() == 0


# M13: the guard at the ASGI level.


async def _guard(scope: dict, *, allow_remote: bool = False) -> tuple[list[str], list[dict]]:
    called: list[str] = []
    sent: list[dict] = []

    async def app(scope, receive, send):
        called.append(scope["type"])
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "websocket.connect"} if scope["type"] == "websocket" else {"type": "http.request"}

    async def send(message):
        sent.append(message)

    await LocalApiGuard(app, allow_remote=lambda: allow_remote)(scope, receive, send)
    return called, sent


def _scope(path: str, *, kind: str = "http", client=("127.0.0.1", 5000), host: bytes = b"127.0.0.1",
           root_path: str = "", method: str = "GET") -> dict:
    return {"type": kind, "path": path, "root_path": root_path, "client": client, "method": method,
            "headers": [(b"host", host)]}


async def test_guard_handles_root_path_in_both_spellings():
    remote = ("203.0.113.9", 5000)
    for scope in (_scope("/prefix/api/sessions", root_path="/prefix", client=remote),
                  _scope("/api/sessions", root_path="/prefix", client=remote)):
        called, sent = await _guard(scope)
        assert called == [] and sent[0]["status"] == 403
    called, sent = await _guard(_scope("/prefix/api/sessions", root_path="/prefix"))
    assert called == ["http"] and sent[0]["status"] == 200


async def test_guard_refuses_a_request_without_a_client_address():
    called, sent = await _guard(_scope("/api/sessions", client=None))
    assert called == [] and sent[0]["status"] == 403
    assert json.loads(sent[1]["body"])["error"] == "local_only"


async def test_guard_rejects_websockets_on_the_api_and_passes_others():
    called, sent = await _guard(_scope("/api/chat", kind="websocket"))
    assert called == [] and sent == [{"type": "websocket.close", "code": 1008}]
    called, sent = await _guard(_scope("/api", kind="websocket", client=None))
    assert called == [] and sent[0]["type"] == "websocket.close"
    called, _ = await _guard(_scope("/other", kind="websocket"))
    assert called == ["websocket"]
    called, _ = await _guard({"type": "lifespan"})
    assert called == ["lifespan"]


# --- Task 3.6: audit-ledger coverage -------------------------------------------------------------------------


def test_ledger_export_has_the_tool_call_and_the_proposal_and_no_secret(world):
    with world.client() as client:
        sid = new_session(client)
        world.add_evidence(sid)
        events = chat(client, sid)
        turn_id = events[0]["messageId"]
        ticket = next(e for e in events if e["type"] == "data-trade-proposal")["ticket"]
        r = client.get(f"/api/ledger/{sid}.jsonl")
        assert r.status_code == 200
        lines = [json.loads(line) for line in r.text.splitlines() if line]
        assert [line["kind"] for line in lines] == ["proposal", "tool_call"]
        proposal, tool_call = (line["payload"] for line in lines)
        assert proposal["ticket_id"] == ticket["id"] and proposal["hash"] == ticket["hash"]
        assert proposal["turn_id"] == tool_call["turn_id"] == turn_id
        assert proposal["legs"] == ticket["legs"] and proposal["evidence_ids"] == ["E1"]
        assert proposal["decisions"][0]["verdict"] == "accept"
        assert tool_call["tool"] == "propose_trades" and tool_call["is_error"] is False
        assert len(tool_call["args_sha256"]) == len(tool_call["result_sha256"]) == 64
        # neither the model's free text nor a key is in the export
        assert "Services growth" not in r.text and "Regulatory pressure" not in r.text
        assert_no_secrets(r)


def test_tool_calls_reach_the_new_ledger_after_a_settings_rebuild(world):
    with world.client() as client:
        sid = new_session(client)
        old_ledger, old_toolbox = world.runtime.ledger, world.runtime.toolbox
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 200
        new_ledger = world.runtime.ledger
        assert new_ledger is not old_ledger and world.runtime.toolbox is not old_toolbox
        seen: dict[str, list[str]] = {"old": [], "new": []}

        def record(label, ledger):
            real = ledger.append

            def recording(session_id, kind, payload):
                seen[label].append(kind)
                return real(session_id, kind, payload)

            ledger.append = recording

        record("old", old_ledger)
        record("new", new_ledger)
        ticket = propose(client, world, sid)
        assert seen == {"old": [], "new": ["proposal", "tool_call"]}
        r = client.get(f"/api/ledger/{sid}.jsonl")
        lines = [json.loads(line) for line in r.text.splitlines() if line]
        assert [line["kind"] for line in lines] == ["proposal", "tool_call"]
        assert lines[0]["payload"]["ticket_id"] == ticket["id"]
        assert_no_secrets(r)


# --- Task 3.6: log scrubbing ----------------------------------------------------------------------------------


async def test_the_runtime_installs_the_log_scrubber_with_its_known_secrets(tmp_path, caplog):
    world = ApiWorld(tmp_path)
    probe = logging.getLogger("finagent.probe")
    probe.warning("before %s", ALPACA_SECRET)
    assert f"before {ALPACA_SECRET}" in caplog.text  # the control: nothing is installed yet
    caplog.clear()
    runtime = world.state().chat
    try:
        probe.warning("after %s %s %s %s", ALPACA_ID, ALPACA_SECRET, TAVILY, GEMINI)
        assert "after [redacted] [redacted] [redacted] [redacted]" in caplog.text
        logging.getLogger("thirdparty").warning("not ours %s", ALPACA_ID)
        assert f"not ours {ALPACA_ID}" in caplog.text
    finally:
        await runtime.shutdown()


def test_replaced_keys_stay_scrubbed_in_logs_after_a_rebuild(world, caplog):
    with world.client() as client:
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 200
        logging.getLogger("finagent.probe").warning(
            "old %s %s new %s %s", ALPACA_ID, ALPACA_SECRET, NEW_ID, NEW_SECRET)
    assert "old [redacted] [redacted] new [redacted] [redacted]" in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text


def test_a_traceback_logged_during_confirm_never_shows_a_secret(world, caplog):
    with caplog.at_level("DEBUG"), world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        # The re-check fails with a broker error that carries the secret, and writing that failure
        # to the ticket fails too: the second error escapes into the executor's `logger.exception`
        # with the broker error chained to it.
        world.broker.account_error = BrokerError(503, None, f"service unavailable {ALPACA_SECRET}")
        real = world.runtime.tickets.transition

        def failing(ticket_id, expect, to, **fields):
            if to in (TicketStatus.submitted, TicketStatus.failed):
                raise sqlite3.OperationalError("database is locked")
            return real(ticket_id, expect, to, **fields)

        world.runtime.tickets.transition = failing
        r = confirm(client, ticket, sid)
        assert r.json()["status"] == "error"
        assert_no_secrets(r)
        assert world.total_submits() == 0
    assert "confirm of ticket" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "BrokerError: Alpaca broker error 503: service unavailable [redacted]" in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text
