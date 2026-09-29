"""Tests for the `/api/*` router (`finagent.api_chat`), its security guard (`finagent.api_security`)
and the `ChatRuntime` wiring (`finagent.app_state`).

World: a real `ChatRuntime` (AppDB on a tmp file, EvidenceStore, SessionStore, TicketStore, Ledger,
Executor, ChatToolbox, ChatAgent) over `fakes_trading.FakeBroker`/`FakeData`, a scripted `StubLLM`,
a fake QA orchestrator for `/answer`, an HTTP client that refuses every request (no network), and a
tmp env file. The real `.env` is never touched.
"""
from __future__ import annotations

import asyncio
import json
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
from finagent.agent.llm import StubLLM, ToolCall, Turn
from finagent.api import create_app
from finagent.api_chat import event_stream
from finagent.app_state import ChatRuntime
from finagent.config import Settings
from finagent.models import AnswerResponse
from finagent.research.http import HttpClient
from finagent.trading.broker import BrokerError
from finagent.trading.models import Mandate, MarketContext, OrderLeg, TicketStatus
from finagent.trading.risk import evaluate

FIXED_NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)
HDR = {"X-FinAgent-Client": "1"}

ALPACA_ID = "PKTESTKEYID0123456789"
ALPACA_SECRET = "testSecretValue-abcdef0123456789"
NEW_ID = "PKNEWKEYID9876543210"
NEW_SECRET = "newSecretValue-0123456789abcdefgh"
BAD_ID = "PKBADKEYID5555555555"
BAD_SECRET = "badSecretValue-5555555555555555"
DOWN_ID = "PKDOWNKEYID444444444"
DOWN_SECRET = "downSecretValue-4444444444444444"
TAVILY = "tvly-test-key-0123456789abcdef"
GEMINI = "gemini-test-key-0123456789abcdef"
SECRETS = [ALPACA_ID, ALPACA_SECRET, NEW_ID, NEW_SECRET, BAD_ID, BAD_SECRET, DOWN_ID, DOWN_SECRET,
           TAVILY, GEMINI]

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
    async def answer(self, question_id, question, on_event=None):
        return AnswerResponse(question_id=question_id, question=question, retrieved_context="ctx",
                              think_trace="trace", answer=f"answer to {question}")


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
        self.with_chat = chat
        self.runtime: ChatRuntime | None = None
        self.before_start = None  # optional hook(runtime) run inside the state factory

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
        broker = ClosableBroker(key_id)
        if (key_id, secret) in {(DOWN_ID, DOWN_SECRET)}:
            broker.account_error = BrokerError(503, None, f"service unavailable {secret}")
        elif (key_id, secret) not in {(ALPACA_ID, ALPACA_SECRET), (NEW_ID, NEW_SECRET)}:
            broker.account_error = BrokerError(401, None, f"unauthorized {secret}")
        self.brokers.append(broker)
        return broker

    def data_factory(self, key_id: str, secret: str, http) -> FakeData:
        return self.data

    def state(self):
        fin = SimpleNamespace(settings=self.settings, llm=self.llm, tools=None,
                              orchestrator=FakeOrchestrator(), chat=None)
        if self.with_chat:
            async def no_sleep(_seconds):
                return None

            fin.chat = ChatRuntime(
                self.settings,
                llm=self.llm,
                registry=None,
                orchestrator=fin.orchestrator,
                env_path=self.env_path,
                broker_factory=self.broker_factory,
                data_factory=self.data_factory,
                http_factory=lambda: HttpClient(client=httpx.AsyncClient(transport=httpx.MockTransport(_refuse_all))),
                clock=self.clock,
                sleep=no_sleep,
                close_grace_sec=0,
            )
            self.runtime = fin.chat
            if self.before_start is not None:
                self.before_start(fin.chat)
        return fin

    @property
    def broker(self) -> ClosableBroker | None:
        return self.runtime.broker if self.runtime is not None else None

    def client(self, **kw) -> TestClient:
        kw.setdefault("headers", HDR)
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
        for host in ("127.0.0.1:8000", "localhost:8000", "[::1]:8000", "LOCALHOST", "testserver"):
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
                           headers={"origin": "http://testserver.evil.example"}).status_code == 403
        assert client.post("/api/sessions", json={}, headers={"origin": "http://testserver"}).status_code == 201
        assert client.post("/api/sessions", json={}, headers={"origin": "https://testserver"}).status_code == 201
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
