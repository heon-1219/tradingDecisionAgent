"""Final review fix round 1: the kill switch, re-enabling trading, the settings audit trail and the
non-paper drop, on a real `ChatRuntime` (the world of `test_api_chat`: fake broker and market data,
a scripted model, a tmp env file, no network).

- A-m1: a kill press wins over a re-enable that is already waiting on (or holding) the lock.
- A-m2: the kill switch writes `TRADING_ENABLED=false` before it waits for the lock or calls the
  broker.
- A-m3(b): re-enabling trading resumes the broker the kill switch suspended (`broker_resumed`).
- A-m4: settings writes and the non-paper drop are ledgered, by field name only.
- A-m6: a non-paper drop from the chat's `get_account` stops a confirm that sits between legs.
- B-M5: candidate Alpaca keys join the log scrub before any call is made with them.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from fakes_trading import make_account
from finagent.api_chat import _ticket_action
from finagent.app_state import ENV_NAMES, KillSwitchPressed
from finagent.trading.broker import BrokerError
from test_api_chat import (
    ALPACA_ID,
    ALPACA_SECRET,
    ANTHROPIC,
    LIMIT_LEG,
    NEW_ID,
    NEW_SECRET,
    SECRETS,
    TAVILY,
    ApiWorld,
    new_session,
    propose,
)

KEY_LEG = {"symbol": "KEY", "side": "buy", "qty": "10", "type": "limit", "limit_price": "15"}
NEW_TAVILY = "tvly-new-key-9876543210fedcba"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [*ENV_NAMES.values(), "API_ALLOW_REMOTE", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def world(tmp_path) -> ApiWorld:
    return ApiWorld(tmp_path)


class Gate:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def hold(self):
        self.entered.set()
        await self.release.wait()


async def _ticket(runtime, world: ApiWorld, legs, turn: str = "t1"):
    world.legs = legs
    sid = runtime.sessions.create()["id"]
    world.add_evidence(sid)
    args = world._script("x")[0].tool_calls[0].input
    out = await runtime.toolbox.execute(sid, "propose_trades", args, turn_id=turn)
    assert not out.is_error, out.text
    [ticket] = [runtime.tickets.get(e["ticket"]["id"]) for e in out.events
                if e.get("type") == "data-trade-proposal"]
    return sid, ticket


def _system_entries(runtime, kind: str | None = None) -> list[dict]:
    entries = [json.loads(line) for line in runtime.ledger.export_jsonl("system")]
    return [e for e in entries if kind is None or e["kind"] == kind]


# --- A-m1: the kill generation --------------------------------------------------------------------


async def test_a_kill_press_wins_over_a_re_enable_that_holds_the_lock(tmp_path):
    # The reviewer's race: a PUT re-enabling trading holds the lock (here: its LLM re-check), a
    # confirm queues behind it, and then the user presses the kill switch.
    world = ApiWorld(tmp_path, trading_enabled=False)
    runtime = world.state().chat
    try:
        runtime.settings.trading_enabled = True
        sid, ticket = await _ticket(runtime, world, [LIMIT_LEG])
        runtime.settings.trading_enabled = False
        gate = Gate()

        async def slow_recheck(changes):
            await gate.hold()
            return "ok"

        runtime._recheck_llm = slow_recheck
        put = asyncio.ensure_future(
            runtime.apply_settings({"trading_enabled": True}, llm_context=("x", "y", "z")))
        await gate.entered.wait()
        queued = asyncio.ensure_future(_ticket_action(runtime, "confirm", ticket.id, sid, ticket.hash))
        await asyncio.sleep(0)
        kill = asyncio.ensure_future(runtime.kill_switch(False))
        await asyncio.sleep(0)
        gate.release.set()
        with pytest.raises(KillSwitchPressed):
            await put
        response = await queued
        await kill
        assert response.status_code == 423 and json.loads(response.body)["status"] == "disabled"
        assert runtime.settings.trading_enabled is False
        assert "TRADING_ENABLED=false" in world.env() and "TRADING_ENABLED=true" not in world.env()
        assert world.total_submits() == 0
    finally:
        await runtime.shutdown()


async def test_a_re_enable_after_the_kill_press_still_works(tmp_path):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        await runtime.kill_switch(False)
        await runtime.apply_settings({"trading_enabled": True})  # captured after the press
        assert runtime.settings.trading_enabled is True
        assert "TRADING_ENABLED=true" in world.env()
    finally:
        await runtime.shutdown()


def test_put_settings_refuses_a_re_enable_when_the_kill_switch_was_pressed_meanwhile(world):
    with world.client() as client:
        assert client.post("/api/kill-switch", json={}).status_code == 200
        runtime = world.runtime
        check = runtime.check_alpaca_keys

        async def check_then_kill(key_id, secret):
            status = await check(key_id, secret)
            await runtime.kill_switch(False)  # pressed while this PUT checks the new keys
            return status

        runtime.check_alpaca_keys = check_then_kill
        before = world.env()
        r = client.put("/api/settings", json={
            "trading_enabled": True, "alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 409 and r.json()["error"] == "kill_switch_pressed"
        assert world.settings.trading_enabled is False
        assert world.env() == before and NEW_ID not in world.env()  # nothing was written
        assert world.settings.alpaca_api_key_id == ALPACA_ID


# --- A-m2: .env first -----------------------------------------------------------------------------


async def test_the_kill_switch_writes_the_env_file_before_the_lock_and_the_broker(tmp_path):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        broker = runtime.broker
        seen: list[tuple[str, bool]] = []
        suspend, orders = broker.set_suspend_trade, broker.orders

        async def recording_suspend(flag):
            seen.append(("suspend", "TRADING_ENABLED=false" in world.env()))
            return await suspend(flag)

        async def recording_orders(**kw):
            seen.append(("orders", "TRADING_ENABLED=false" in world.env()))
            return await orders(**kw)

        broker.set_suspend_trade, broker.orders = recording_suspend, recording_orders
        async with runtime.lock:  # a confirm in flight holds the lock (broker calls can take 15 s)
            kill = asyncio.ensure_future(runtime.kill_switch(True))
            await asyncio.sleep(0)
            assert runtime.settings.trading_enabled is False
            assert "TRADING_ENABLED=false\n" in world.env()  # persisted already
            assert seen == []  # the broker steps wait for the lock
        result = await kill
        assert result["persisted"] is True
        assert seen == [("suspend", True), ("orders", True)]
        [entry] = _system_entries(runtime, "kill_switch")
        assert entry["payload"]["persisted"] is True and entry["payload"]["broker_suspended"] is True
    finally:
        await runtime.shutdown()


# --- A-m3(b): resume the broker on re-enable --------------------------------------------------------


def test_re_enabling_resumes_the_broker_the_kill_switch_suspended(world):
    with world.client() as client:
        assert client.post("/api/kill-switch", json={"suspend_broker": True}).status_code == 200
        assert world.broker.suspend_calls == [True]
        r = client.put("/api/settings", json={"trading_enabled": True})
        assert r.status_code == 200 and r.json()["trading_enabled"] is True
        assert r.json()["broker_resumed"] is True
        assert world.broker.suspend_calls == [True, False]
        # Nothing left to resume: a kill without suspension, then a re-enable.
        client.post("/api/kill-switch", json={"suspend_broker": False})
        r = client.put("/api/settings", json={"trading_enabled": True})
        assert r.status_code == 200 and r.json()["broker_resumed"] is None
        assert world.broker.suspend_calls == [True, False]
        # A write that does not turn trading on says nothing about the broker.
        assert "broker_resumed" not in client.put("/api/settings", json={"tavily_key": NEW_TAVILY}).json()


def test_a_failed_resume_is_reported_and_tried_again_on_the_next_re_enable(world):
    with world.client() as client:
        client.post("/api/kill-switch", json={"suspend_broker": True})
        world.broker.suspend_error = BrokerError(503, None, "service unavailable")
        r = client.put("/api/settings", json={"trading_enabled": True})
        assert r.status_code == 200 and r.json()["trading_enabled"] is True
        assert r.json()["broker_resumed"] is False
        world.broker.suspend_error = None
        r = client.put("/api/settings", json={"trading_enabled": True})
        assert r.json()["broker_resumed"] is True
        assert world.broker.suspend_calls == [True, False, False]


def test_new_alpaca_keys_are_never_resumed_for_the_old_suspension(world):
    with world.client() as client:
        client.post("/api/kill-switch", json={"suspend_broker": True})
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET,
                                              "trading_enabled": True})
        assert r.status_code == 200 and r.json()["broker_resumed"] is None
        assert world.broker.suspend_calls == []  # the new account was never suspended by the app


# --- A-m4: the settings audit trail ------------------------------------------------------------------


def test_settings_writes_are_ledgered_by_field_name_only(world):
    with world.client() as client:
        client.post("/api/kill-switch", json={})
        assert client.put("/api/settings", json={"trading_enabled": True}).status_code == 200
        assert client.put("/api/settings", json={
            "alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET}).status_code == 200
        assert client.put("/api/settings", json={"tavily_key": NEW_TAVILY}).status_code == 200
        assert client.put("/api/settings", json={
            "llm_provider": "anthropic", "llm_model": "claude-test-opus",
            "anthropic_key": ANTHROPIC}).status_code == 200
        entries = _system_entries(world.runtime, "settings_changed")
        assert [e["payload"]["fields"] for e in entries] == [
            "trading_enabled",
            "alpaca_api_key_id, alpaca_api_secret_key",
            "tavily_api_key",
            "anthropic_api_key, llm_model, llm_provider",
        ]
        assert entries[0]["payload"]["trading_enabled"] is True
        for entry in entries:
            assert set(entry["payload"]) <= {"fields", "trading_enabled", "broker_resumed"}
        raw = json.dumps(entries)
        assert "[REDACTED]" not in raw  # no value was ever there to redact
        stored = " ".join(row["payload"] for row in world.runtime.db.query(
            "SELECT payload FROM ledger WHERE session_id='system'"))
        for value in [*SECRETS, NEW_TAVILY, TAVILY, "claude-test-opus"]:
            assert value not in stored


def test_the_non_paper_drop_is_ledgered(world):
    with world.client() as client:
        world.broker.account_snapshot = make_account(paper=False)
        assert client.get("/api/portfolio").json() == {"connected": False}
        [entry] = _system_entries(world.runtime, "broker_dropped")
        assert entry["payload"] == {"reason": "not_paper"}


# --- A-m6: the non-paper drop stops an in-flight confirm ----------------------------------------------


async def test_a_non_paper_drop_from_get_account_stops_the_remaining_legs(tmp_path):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        sid, ticket = await _ticket(runtime, world, [LIMIT_LEG, KEY_LEG])
        broker = runtime.broker
        gate = Gate()
        submit = broker.submit_order

        async def first_leg_waits(leg, cid):
            if broker.submits == 0:
                order = await submit(leg, cid)
                await gate.hold()
                return order
            return await submit(leg, cid)

        broker.submit_order = first_leg_waits
        confirming = asyncio.ensure_future(_ticket_action(runtime, "confirm", ticket.id, sid, ticket.hash))
        await gate.entered.wait()
        broker.account_snapshot = make_account(paper=False)
        out = await runtime.toolbox.execute(sid, "get_account", {}, turn_id="t2")  # no lock
        assert out.is_error and runtime.broker is None and runtime.alpaca_status == "not_paper"
        gate.release.set()
        body = json.loads((await confirming).body)
        assert broker.submits == 1
        assert body["status"] == "submitted"
        assert "not to be a paper account" in body["message"]
        assert body["ticket"]["broker_order_ids"][1] is None
    finally:
        await runtime.shutdown()


# --- B-M5: candidate keys are scrubbed before any call ------------------------------------------------


async def test_candidate_alpaca_keys_join_the_log_scrub_before_any_call(tmp_path, caplog):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        def leaky_factory(key_id, secret):
            logging.getLogger("finagent.probe").warning("building a broker for %s / %s", key_id, secret)
            raise ValueError("refused")

        runtime._broker_factory = leaky_factory
        assert await runtime.check_alpaca_keys(NEW_ID, NEW_SECRET) == "invalid"
        assert "building a broker for [redacted] / [redacted]" in caplog.text
        assert NEW_ID not in caplog.text and NEW_SECRET not in caplog.text
        assert ALPACA_SECRET not in caplog.text
    finally:
        await runtime.shutdown()
