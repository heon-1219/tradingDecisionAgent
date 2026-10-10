"""Tests for `scripts/demo_server.py` (the keyless demo server) and `scripts/_demo_broker.py` (the
fake broker whose orders fill).

The walk-through tests drive the real app, built by `demo_server.build_demo_app`, through the same
HTTP calls the web app makes: create a chat, run a turn, read the ticket, confirm it, watch the
orders, look at the portfolio. Everything is offline; state lives under pytest's tmp directory.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))  # at the end: scripts/trace.py must never shadow the stdlib

import _demo_broker  # noqa: E402
import _fakeworld as fw  # noqa: E402
import demo_server  # noqa: E402

from finagent.config import ENV_NAMES  # noqa: E402
from finagent.i18n import detect_lang  # noqa: E402
from finagent.trading.models import OrderLeg  # noqa: E402

WRITE = {"X-FinAgent-Client": "1"}
CORE = "Build a low-cost core ETF portfolio with $5,000."


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def no_settings_in_the_environment(monkeypatch):
    # Whatever the developer's shell exports must not change what these tests see.
    for name in {*ENV_NAMES.values(), "GOOGLE_API_KEY"}:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def demo(tmp_path):
    app, world = demo_server.build_demo_app(tmp_path)
    world.broker.monotonic = Clock()
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client, world
    fw.release_sqlite(world.runtime.db, getattr(world.registry, "rdb", None))


def new_session(client, lang="en") -> str:
    r = client.post("/api/sessions", json={"title": "demo", "lang": lang}, headers=WRITE)
    assert r.status_code == 201
    return r.json()["id"]


def chat(client, sid: str, message: str) -> list[dict]:
    r = client.post("/api/chat", json={"session_id": sid, "message": message}, headers=WRITE)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def of(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e["type"] == kind]


def text_of(events: list[dict]) -> str:
    return "".join(e["delta"] for e in of(events, "text-delta"))


# --- the app ---------------------------------------------------------------------------------------


def test_the_demo_serves_the_web_app_and_reports_a_connected_fake_account(demo, tmp_path):
    client, world = demo
    page = client.get("/")
    assert page.status_code == 200 and "/app/js/main.js" in page.text and "/app/components.css" in page.text
    assert client.get("/app/js/components/index.js").status_code == 200
    assert client.get("/health").json()["llm"] == "demo/demo-scripted"

    settings = client.get("/api/settings").json()
    assert settings["alpaca"] == "connected" and settings["trading_enabled"] is True
    assert settings["model"] == "demo-scripted" and settings["env_fallbacks"] == []
    assert isinstance(world.broker, _demo_broker.FillingBroker) and world.runtime.broker is world.broker
    # state is under the directory it was given, never under data/ or next to the real .env
    assert pathlib.Path(world.settings.app_db_path).parent == tmp_path
    assert world.runtime.env_path.parent == tmp_path and world.runtime.env_path != ROOT / ".env"

    portfolio = client.get("/api/portfolio").json()
    assert portfolio["connected"] is True and portfolio["account"]["cash"] == "100000"
    assert {p["symbol"] for p in portfolio["positions"]} == {"VTI", "AAPL", "BND"}
    assert portfolio["open_orders"] == []


def test_walk_through_chat_ticket_confirm_fill_and_portfolio(demo):
    client, world = demo
    sid = new_session(client)
    events = chat(client, sid, CORE)

    # the stream the UI renders: thinking text, tool cards with sources, one ticket, a cited answer
    assert events[0]["type"] == "start" and events[-1]["type"] == "finish" and not of(events, "error")
    assert of(events, "reasoning-delta")
    called = [e["toolName"] for e in of(events, "tool-input-available")]
    assert called[:2] == ["get_account", "get_positions"] and called[-1] == "propose_trades"
    assert all(isinstance(e["input"], dict) for e in of(events, "tool-input-available"))
    outputs = of(events, "tool-output-available")
    assert len(outputs) == len(called) and not any(e["isError"] for e in outputs)
    assert any(e["sources"] for e in outputs)
    proposals = of(events, "data-trade-proposal")
    assert len(proposals) == 1
    ticket = proposals[0]["ticket"]
    answer = text_of(events)
    assert re.search(r"\[E\d+\]", answer) and re.search(r"^\|\s*-{3,}", answer, re.M)
    assert world.broker.submits == 0  # proposing never submits

    # the ticket as the card reads it: money as strings, a hash, an expiry
    read = client.get(f"/api/tickets/{ticket['id']}", params={"session_id": sid}).json()
    assert read["ticket"] == ticket and read["orders"] == []
    assert ticket["status"] == "proposed" and re.fullmatch(r"[0-9a-f]{64}", ticket["hash"])
    assert [(leg["side"], leg["symbol"], leg["type"]) for leg in ticket["legs"]] == [
        ("buy", "VOO", "limit"), ("buy", "BND", "limit")]
    for decision in ticket["decisions"]:
        assert decision["verdict"] == "accept"
        assert isinstance(decision["est_cost_usd"], str) and isinstance(decision["leg"]["qty"], str)
    buy_cost = sum(Decimal(d["est_cost_usd"]) for d in ticket["decisions"])

    # confirm: the only call that submits
    confirm = client.post(f"/api/tickets/{ticket['id']}/confirm", json={"session_id": sid, "hash": ticket["hash"]},
                          headers=WRITE)
    assert confirm.status_code == 200
    body = confirm.json()
    assert body["status"] == "submitted" and body["ticket"]["status"] == "submitted"
    assert world.broker.submits == 2 and len(body["orders"]) == 2
    assert {o["status"] for o in body["orders"]} == {"accepted"}
    # a second confirm is refused and submits nothing
    again = client.post(f"/api/tickets/{ticket['id']}/confirm", json={"session_id": sid, "hash": ticket["hash"]},
                        headers=WRITE)
    assert again.status_code == 409 and world.broker.submits == 2

    # the order timeline, as the card polls it: accepted -> partially filled -> filled
    def statuses():
        polled = client.get(f"/api/tickets/{ticket['id']}", params={"session_id": sid}).json()
        return [row["order"]["status"] for row in polled["orders"]], polled

    assert statuses()[0] == ["accepted", "accepted"]
    assert len(client.get("/api/portfolio").json()["open_orders"]) == 2
    world.broker.monotonic.t += _demo_broker.PARTIAL_AFTER_SEC
    assert statuses()[0] == ["partially_filled", "partially_filled"]
    world.broker.monotonic.t += _demo_broker.FILLED_AFTER_SEC
    filled, polled = statuses()
    assert filled == ["filled", "filled"]
    for row, leg in zip(polled["orders"], ticket["legs"]):
        assert row["order"]["filled_qty"] == leg["qty"] and row["order"]["filled_avg_price"] == leg["limit_price"]

    # the portfolio moved: cash down by the fills, the new position is there, nothing is left open
    portfolio = client.get("/api/portfolio").json()
    assert Decimal(portfolio["account"]["cash"]) == Decimal("100000") - buy_cost
    assert Decimal(portfolio["account"]["buying_power"]) == Decimal("100000") - buy_cost
    positions = {p["symbol"]: p for p in portfolio["positions"]}
    assert positions["VOO"]["qty"] == ticket["legs"][0]["qty"]
    assert Decimal(positions["BND"]["qty"]) == Decimal("100") + Decimal(ticket["legs"][1]["qty"])
    assert portfolio["open_orders"] == []
    # polling again changes nothing (a fill is applied once)
    assert statuses()[0] == ["filled", "filled"]
    assert Decimal(client.get("/api/portfolio").json()["account"]["cash"]) == Decimal("100000") - buy_cost

    # what a reload needs: history with the turn summary, the sources, the activity log
    detail = client.get(f"/api/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "event"]
    stored = detail["messages"][2]["meta"]["events"]
    assert {"type": "data-trade-proposal", "ticket_id": ticket["id"]} in stored
    assert detail["active_ticket"] is None
    listed = {s["id"] for s in client.get(f"/api/sources/{sid}").json()["sources"]}
    assert set(re.findall(r"\[(E\d+)\]", answer)) <= listed and set(ticket["evidence_ids"]) <= listed
    ledger = client.get(f"/api/ledger/{sid}.jsonl")
    assert ledger.status_code == 200 and "attachment" in ledger.headers["content-disposition"]
    kinds = [json.loads(line)["kind"] for line in ledger.text.splitlines() if line.strip()]
    assert "confirm_request" in kinds and "submit" in kinds
    assert world.refused == []  # no request left the fake transport


def test_modify_supersedes_the_ticket_and_the_old_one_cannot_be_confirmed(demo):
    client, world = demo
    sid = new_session(client)
    old = of(chat(client, sid, CORE), "data-trade-proposal")[0]["ticket"]
    # the message the ticket card's Modify fields send
    events = chat(client, sid, "Modify: buy 2 VOO limit 610.00; sell 10 BND limit 73.50")
    new = of(events, "data-trade-proposal")[0]["ticket"]
    assert new["id"] != old["id"]
    assert [(leg["side"], leg["qty"], leg["symbol"], leg["limit_price"]) for leg in new["legs"]] == [
        ("buy", "2", "VOO", "610.00"), ("sell", "10", "BND", "73.50")]
    stale = client.post(f"/api/tickets/{old['id']}/confirm", json={"session_id": sid, "hash": old["hash"]}, headers=WRITE)
    assert stale.status_code == 409 and stale.json()["ticket"]["status"] == "superseded"
    assert world.broker.submits == 0
    assert client.get(f"/api/sessions/{sid}").json()["active_ticket"]["id"] == new["id"]
    # refresh re-quotes the live ticket into a new one (200), reject dismisses that one
    fresh = client.post(f"/api/tickets/{new['id']}/refresh", json={"session_id": sid}, headers=WRITE)
    assert fresh.status_code == 200 and fresh.json()["status"] == "requote"
    newest = fresh.json()["new_ticket"]
    rejected = client.post(f"/api/tickets/{newest['id']}/reject", json={"session_id": sid}, headers=WRITE)
    assert rejected.status_code == 200 and rejected.json()["ticket"]["status"] == "rejected_by_user"
    assert world.broker.submits == 0


def test_the_other_triggers_the_ui_needs(demo):
    client, world = demo
    sid = new_session(client)

    # a mandate draft, then saving it the way the card does (strings), then a 422 with field messages
    events = chat(client, sid, "Draft a mandate for me: a $10,000 budget, at most 15% per position.")
    draft = of(events, "data-mandate")[0]
    assert draft["draft"] is True and draft["mandate"]["budget_usd"] == "10000"
    payload = {k: v for k, v in draft["mandate"].items() if k != "version"}
    payload.update(max_positions="6", max_orders_per_day="4", allow_symbols=["VOO", "BND"])
    saved = client.put(f"/api/mandate/{sid}", json=payload, headers=WRITE)
    assert saved.status_code == 200 and saved.json()["saved"] is True
    assert saved.json()["mandate"]["version"] == 1 and saved.json()["mandate"]["max_positions"] == 6
    bad = client.put(f"/api/mandate/{sid}", json={**payload, "max_position_pct": "250", "allow_symbols": ["../x"]},
                     headers=WRITE)
    assert bad.status_code == 422
    assert {f["loc"].split(".")[0] for f in bad.json()["fields"]} == {"max_position_pct", "allow_symbols"}

    # grounded search: Google's search-suggestions HTML arrives unmodified
    events = chat(client, sid, "Give me a grounded answer about index funds.")
    html = fw.GROUNDED_SAMPLE["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["renderedContent"]
    assert [e["search_suggestions_html"] for e in of(events, "source-grounding")] == [html]
    assert re.search(r"\[E\d+\]", text_of(events))

    # an answer with a number the sources do not support
    events = chat(client, sid, "Explain dollar-cost averaging in plain words.")
    assert of(events, "data-grounding-flags") == [{"type": "data-grounding-flags", "numbers": ["7.25%"]}]
    assert not of(events, "tool-input-available")

    # a failed model step: an error event, then finish
    events = chat(client, sid, "Now simulate an error please.")
    assert [e["type"] for e in events][-2:] == ["error", "finish"] and not of(events, "text-delta")

    # the fetched page: its injected instruction is not obeyed
    events = chat(client, sid, f"Summarise {demo_server.DEMO_PAGE_URL}")
    assert [e["toolName"] for e in of(events, "tool-input-available")] == ["fetch_source"]
    fetched = of(events, "tool-output-available")[0]
    assert "<untrusted source=" in fetched["output"] and fetched["sources"][0]["url"] == demo_server.DEMO_PAGE_URL
    assert of(events, "source-url")[0]["url"] == demo_server.DEMO_PAGE_URL
    assert of(events, "data-trade-proposal") == [] and "XYZ" not in text_of(events)

    # Korean in, Korean out
    ko = new_session(client, "ko")
    events = chat(client, ko, "5,000달러로 저비용 미국 ETF 핵심 포트폴리오를 구성해 줘")
    assert detect_lang(text_of(events)) == "ko" and len(of(events, "data-trade-proposal")) == 1
    assert world.broker.submits == 0 and world.refused == []


def test_kill_switch_blocks_confirm_and_writes_only_the_scratch_env_file(demo, tmp_path):
    client, world = demo
    real_env = ROOT / ".env"
    before = real_env.stat().st_mtime_ns if real_env.exists() else None
    sid = new_session(client)
    ticket = of(chat(client, sid, CORE), "data-trade-proposal")[0]["ticket"]

    killed = client.post("/api/kill-switch", json={"suspend_broker": True}, headers=WRITE)
    assert killed.status_code == 200
    assert killed.json() == {"trading_enabled": False, "suspend_broker": True, "broker_connected": True, "persisted": True}
    assert world.broker.suspend_calls == [True]
    assert "TRADING_ENABLED=false" in (tmp_path / ".env").read_text(encoding="utf-8")
    blocked = client.post(f"/api/tickets/{ticket['id']}/confirm", json={"session_id": sid, "hash": ticket["hash"]},
                          headers=WRITE)
    assert blocked.status_code == 423 and world.broker.submits == 0

    on = client.put("/api/settings", json={"trading_enabled": True}, headers=WRITE)
    assert on.status_code == 200 and on.json()["trading_enabled"] is True
    ok = client.post(f"/api/tickets/{ticket['id']}/confirm", json={"session_id": sid, "hash": ticket["hash"]},
                     headers=WRITE)
    assert ok.status_code == 200 and world.broker.submits == 2
    after = real_env.stat().st_mtime_ns if real_env.exists() else None
    assert after == before  # the project's .env was not touched


def test_saving_keys_in_the_demo_never_builds_a_real_broker(tmp_path, monkeypatch):
    import finagent.app_state as app_state

    def forbidden(*args, **kwargs):
        raise AssertionError("AlpacaPaper must never be constructed by the demo server")

    monkeypatch.setattr(app_state, "AlpacaPaper", forbidden)
    app, world = demo_server.build_demo_app(tmp_path)
    try:
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
            r = client.put("/api/settings", json={"alpaca_key_id": "typed-key-id", "alpaca_secret": "typed-secret"},
                           headers=WRITE)
            assert r.status_code == 200 and r.json()["alpaca"] == "connected"
            assert "typed-secret" not in r.text and "typed-key-id" not in r.text
            assert world.runtime.broker is world.broker
            assert isinstance(world.runtime.broker, _demo_broker.FillingBroker)
            cleared = client.put("/api/settings", json={"alpaca_key_id": "", "alpaca_secret": ""}, headers=WRITE)
            assert cleared.status_code == 200 and cleared.json()["alpaca"] == "missing"
            assert client.get("/api/portfolio").json() == {"connected": False}
    finally:
        fw.release_sqlite(world.runtime.db, getattr(world.registry, "rdb", None))


def test_the_demo_says_so_in_its_settings(demo):
    # Final review B-M8 (backend): the web app shows a DEMO badge from this flag.
    client, world = demo
    assert world.runtime.demo is True
    assert client.get("/api/settings").json()["demo"] is True


def test_typed_keys_stay_in_memory_in_the_demo(demo, tmp_path):
    # Final review B-M8: nothing persistent gets a key a visitor typed into the demo.
    client, world = demo
    env_before = (tmp_path / ".env").read_text(encoding="utf-8")
    r = client.put("/api/settings", headers=WRITE, json={
        "alpaca_key_id": "typed-key-id-123", "alpaca_secret": "typed-secret-456",
        "tavily_key": "tvly-typed-key-789", "gemini_key": "demo-valid-key"})
    assert r.status_code == 200, r.text
    assert world.runtime.settings.tavily_api_key == "tvly-typed-key-789"  # in use, in memory
    assert world.runtime.settings.alpaca_api_key_id == "typed-key-id-123"
    env_after = (tmp_path / ".env").read_text(encoding="utf-8")
    assert env_after == env_before
    stored = " ".join(row["payload"] for row in world.runtime.db.query("SELECT payload FROM ledger"))
    for typed in ("typed-key-id-123", "typed-secret-456", "tvly-typed-key-789", "demo-valid-key"):
        assert typed not in env_after and typed not in stored
    # Non-secret settings still behave as in the real app (the kill switch is persisted).
    client.post("/api/kill-switch", json={}, headers=WRITE)
    assert "TRADING_ENABLED=false" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_the_real_app_has_no_demo_flag(tmp_path):
    from test_api_chat import ApiWorld

    world = ApiWorld(tmp_path)
    with world.client() as client:
        assert "demo" not in client.get("/api/settings").json()
    assert world.runtime.demo is False


def test_switching_the_model_provider_in_the_demo_never_reaches_a_provider(tmp_path, monkeypatch):
    import finagent.app_state as app_state
    import finagent.llm_admin as llm_admin

    def boom(*args, **kwargs):
        raise AssertionError("the demo must not use the real model-provider functions")

    async def aboom(*args, **kwargs):
        boom()

    monkeypatch.setattr(llm_admin, "fetch_models", aboom)
    monkeypatch.setattr(llm_admin, "validate_llm", aboom)
    monkeypatch.setattr(app_state, "make_llm", boom)
    app, world = demo_server.build_demo_app(tmp_path)
    try:
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
            listed = client.post("/api/llm/models", json={"provider": "openai", "api_key": demo_server.DEMO_VALID_KEY},
                                 headers=WRITE)
            assert listed.status_code == 200, listed.text
            assert [m["id"] for m in listed.json()["models"]] == ["gpt-6", "gpt-6-mini"]  # no embedding model
            bad = client.post("/api/llm/models", json={"provider": "openai", "api_key": demo_server.DEMO_INVALID_KEY},
                              headers=WRITE)
            assert bad.status_code == 400 and bad.json()["error"] == "llm_invalid_key"

            switched = client.put("/api/settings", headers=WRITE, json={
                "llm_provider": "openai", "llm_model": "gpt-6", "openai_key": demo_server.DEMO_VALID_KEY})
            assert switched.status_code == 200, switched.text
            body = switched.json()
            assert body["provider"] == "openai" and body["model"] == "gpt-6" and body["providers"]["openai"] == "set"
            assert world.llm_admin.built == [("openai", "gpt-6")]
            assert demo_server.DEMO_VALID_KEY not in switched.text
            refused = client.put("/api/settings", headers=WRITE, json={
                "llm_provider": "anthropic", "llm_model": "claude-opus-5-5", "anthropic_key": demo_server.DEMO_INVALID_KEY})
            assert refused.status_code == 400 and refused.json()["error"] == "llm_invalid_key"
            assert world.llm_admin.built == [("openai", "gpt-6")]  # nothing was built for the refused key

            # the scripted model still answers, whatever provider the settings name
            sid = new_session(client)
            events = chat(client, sid, CORE)
            assert len(of(events, "data-trade-proposal")) == 1 and not of(events, "error")
    finally:
        fw.release_sqlite(world.runtime.db, getattr(world.registry, "rdb", None))


def test_the_research_depth_flow_by_http(tmp_path):
    """Task 4.3's checks against the demo: list models, save the valid key (deep), the rejected key
    (400, nothing changes), the rate-limited key (saved with a warning), and the Gemini paid switch
    (shallow <-> deep, which the next reply's finish event reports)."""
    app, world = demo_server.build_demo_app(tmp_path)
    try:
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
            start = client.get("/api/settings").json()
            assert (start["provider"], start["gemini_paid"], start["mode"], start["max_rounds"]) == ("gemini", True, "deep", 12)

            shallow = client.put("/api/settings", headers=WRITE, json={"gemini_paid": False})
            assert shallow.status_code == 200, shallow.text
            assert (shallow.json()["mode"], shallow.json()["max_rounds"], shallow.json()["time_budget_sec"]) == ("shallow", 4, 150)
            assert world.llm_admin.checked == []  # the paid switch needs no key check
            sid = new_session(client)
            [done] = of(chat(client, sid, CORE), "finish")
            assert (done["mode"], done["max_rounds"]) == ("shallow", 4)

            listed = client.post("/api/llm/models", headers=WRITE, json={"provider": "anthropic", "api_key": demo_server.DEMO_VALID_KEY})
            assert listed.status_code == 200 and listed.json()["suggested"] == "claude-opus-5-5"
            limited = client.post("/api/llm/models", headers=WRITE, json={"provider": "anthropic", "api_key": demo_server.DEMO_LIMITED_KEY})
            assert limited.status_code == 429 and limited.json()["error"] == "llm_rate_limited"

            deep = client.put("/api/settings", headers=WRITE, json={
                "llm_provider": "anthropic", "llm_model": "claude-opus-5-5", "anthropic_key": demo_server.DEMO_VALID_KEY})
            assert deep.status_code == 200, deep.text
            assert (deep.json()["provider"], deep.json()["mode"], deep.json()["max_rounds"]) == ("anthropic", "deep", 12)
            [done] = of(chat(client, sid, CORE), "finish")
            assert (done["mode"], done["max_rounds"]) == ("deep", 12)

            before = client.get("/api/settings").json()
            refused = client.put("/api/settings", headers=WRITE, json={
                "llm_provider": "openai", "llm_model": "gpt-6", "openai_key": demo_server.DEMO_INVALID_KEY})
            assert refused.status_code == 400 and refused.json()["error"] == "llm_invalid_key"
            assert client.get("/api/settings").json() == before  # nothing changed

            warned = client.put("/api/settings", headers=WRITE, json={
                "llm_provider": "openai", "llm_model": "gpt-6", "openai_key": demo_server.DEMO_LIMITED_KEY})
            assert warned.status_code == 200 and warned.json()["warning"] == "llm_rate_limited"
            assert warned.json()["provider"] == "openai"
            for answer in (listed, limited, deep, refused, warned):
                assert demo_server.DEMO_VALID_KEY not in answer.text and demo_server.DEMO_LIMITED_KEY not in answer.text

            back = client.put("/api/settings", headers=WRITE, json={"llm_provider": "gemini", "llm_model": "gemini-3.8-flash"})
            assert back.status_code == 200, back.text
            assert (back.json()["mode"], back.json()["gemini_paid"]) == ("shallow", False)
    finally:
        fw.release_sqlite(world.runtime.db, getattr(world.registry, "rdb", None))


def test_the_help_names_the_test_keys_for_the_provider_settings(capsys):
    with pytest.raises(SystemExit):
        demo_server.main(["--help"])
    out = capsys.readouterr().out
    assert demo_server.DEMO_VALID_KEY in out and demo_server.DEMO_INVALID_KEY in out
    assert demo_server.DEMO_LIMITED_KEY in out and "shallow mode" in out


def test_the_help_lists_the_messages_to_try_as_the_readme_says(capsys):
    """The README sends the reader to `--help` for the messages to type; the module docstring shows
    the same list, so the three must not drift apart."""
    with pytest.raises(SystemExit):
        demo_server.main(["--help"])
    out = capsys.readouterr().out
    assert "messages to try" in out
    for message, effect in demo_server.MESSAGES_TO_TRY:
        assert message in out and effect in out
        if not message.startswith("("):  # the Korean line is worded differently in the docstring
            assert message in demo_server.__doc__ and effect in demo_server.__doc__
    # The README's own examples are in the list.
    for message in ("Build a low-cost core ETF portfolio with $5,000", "Modify: buy 3 VOO limit 610",
                    "Draft a mandate: $10,000 budget, 15% per position",
                    "Summarise https://research.example.com/market-note"):
        assert message in out


# --- main() ------------------------------------------------------------------------------------------


def test_main_binds_loopback_on_the_given_port_and_cleans_up(monkeypatch, capsys):
    import uvicorn

    seen = {}

    def fake_run(app, **kwargs):
        seen.update(kwargs)
        seen["paths"] = set(app.openapi()["paths"])  # the app is the real one
        seen["env"] = {name: os.environ.get(name) for name in ("GEMINI_API_KEY", "ALPACA_API_SECRET_KEY", "TRADING_ENABLED")}

    made = []
    real_mkdtemp = demo_server.tempfile.mkdtemp

    def mkdtemp(**kwargs):
        path = real_mkdtemp(**kwargs)
        made.append(pathlib.Path(path))
        return path

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(demo_server.tempfile, "mkdtemp", mkdtemp)
    # Keys exported in the shell are not used by the demo: it drops them from its own environment.
    monkeypatch.setenv("GEMINI_API_KEY", "shell-gemini-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "shell-alpaca-secret")
    monkeypatch.setenv("TRADING_ENABLED", "false")
    assert demo_server.main(["--port", "8771"]) == 0
    assert seen["env"] == {"GEMINI_API_KEY": None, "ALPACA_API_SECRET_KEY": None, "TRADING_ENABLED": None}
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8771
    assert seen["proxy_headers"] is False and seen["workers"] == 1
    assert {"/api/chat", "/api/tickets/{ticket_id}/confirm", "/api/sources/{session_id}"} <= seen["paths"]
    assert len(made) == 1 and made[0].name.startswith("finagent-demo-")
    assert ROOT not in made[0].parents  # scratch state, never inside the repository
    assert not made[0].exists()  # removed on exit
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8771/" in out
    assert "ALPACA_API_SECRET_KEY, GEMINI_API_KEY, TRADING_ENABLED" in out  # names only
    assert "shell-gemini-key" not in out and "shell-alpaca-secret" not in out


def test_main_has_no_host_option_and_checks_the_port(capsys):
    with pytest.raises(SystemExit):
        demo_server.main(["--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        demo_server.main(["--port", "70000"])
    assert demo_server.HOST == "127.0.0.1" and demo_server.DEFAULT_PORT == 8770
    capsys.readouterr()


# --- the filling broker ----------------------------------------------------------------------------------


def filling_broker():
    world_broker = fw.seeded_broker(fw._utcnow(), _demo_broker.FillingBroker)
    clock = Clock()
    world_broker.monotonic = clock
    world_broker.price_of = lambda symbol: Decimal("610.01") if symbol == "VOO" else None
    return world_broker, clock


async def test_orders_fill_over_time_and_a_fill_is_applied_once():
    broker, clock = filling_broker()
    buy = await broker.submit_order(OrderLeg(symbol="VTI", side="buy", qty="10", limit_price="331.00"), "c1")
    assert buy.status == "accepted" and (await broker.get_order(buy.id)).status == "accepted"
    assert [o.id for o in await broker.orders()] == [buy.id]
    clock.t += _demo_broker.PARTIAL_AFTER_SEC
    part = await broker.get_order(buy.id)
    assert (part.status, part.filled_qty, part.filled_avg_price) == ("partially_filled", Decimal("5"), Decimal("331.00"))
    assert (await broker.account()).cash == fw.SEED_CASH  # nothing is booked before the fill completes
    clock.t += _demo_broker.FILLED_AFTER_SEC
    done = await broker.get_order(buy.id)
    assert (done.status, done.filled_qty) == ("filled", Decimal("10"))
    account = await broker.account()
    assert account.cash == fw.SEED_CASH - Decimal("3310.00") == account.buying_power
    vti = (await broker.positions())["VTI"]
    assert vti.qty == Decimal("50") and vti.avg_entry_price == Decimal("306.2000")  # (40*300 + 10*331) / 50
    assert vti.market_value == Decimal("16500.00")
    for _ in range(3):  # reading again books nothing more
        await broker.get_order(buy.id)
        await broker.positions()
    assert (await broker.account()).cash == fw.SEED_CASH - Decimal("3310.00")
    assert await broker.orders() == []
    assert (await broker.get_order_by_client_id("c1")).status == "filled"


async def test_sells_market_orders_single_shares_and_cancels():
    broker, clock = filling_broker()
    sell = await broker.submit_order(OrderLeg(symbol="AAPL", side="sell", qty="30", limit_price="249.99"), "s1")
    one = await broker.submit_order(OrderLeg(symbol="BND", side="buy", qty="1", limit_price="74.01"), "s2")
    market = await broker.submit_order(OrderLeg(symbol="VOO", side="buy", notional="1000", type="market"), "s3")
    unpriced = await broker.submit_order(OrderLeg(symbol="QQQ", side="buy", notional="500", type="market"), "s4")
    clock.t += _demo_broker.PARTIAL_AFTER_SEC
    assert (await broker.get_order(one.id)).status == "accepted"  # one share has no partial step
    assert (await broker.get_order(market.id)).status == "accepted"  # nor has a dollar amount
    await broker.cancel_order(sell.id)
    clock.t += _demo_broker.FILLED_AFTER_SEC
    assert (await broker.get_order(sell.id)).status == "canceled"  # a canceled order never fills
    assert "AAPL" in await broker.positions()
    assert (await broker.get_order(one.id)).status == "filled"
    filled = await broker.get_order(market.id)
    assert filled.status == "filled" and filled.filled_avg_price == Decimal("610.01")
    assert filled.filled_qty == Decimal("1.639")  # 1000 / 610.01, rounded down to a thousandth
    assert (await broker.get_order(unpriced.id)).status == "accepted"  # no price: it stays working
    account = await broker.account()
    assert account.cash == fw.SEED_CASH - Decimal("74.01") - (Decimal("1.639") * Decimal("610.01")).quantize(Decimal("0.01"))
    # selling a whole position removes it
    all_out = await broker.submit_order(OrderLeg(symbol="BND", side="sell", qty="101", limit_price="73.99"), "s5")
    clock.t += _demo_broker.FILLED_AFTER_SEC
    assert (await broker.get_order(all_out.id)).status == "filled"
    assert "BND" not in await broker.positions()
    assert [o.id for o in await broker.orders()] == [unpriced.id]
