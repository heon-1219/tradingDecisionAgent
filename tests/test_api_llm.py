"""Tests for Task 3.8 at the API: the model provider settings (`GET`/`PUT /api/settings`),
`POST /api/llm/models`, and the paper flag (`alpaca: not_paper`, `alpaca_not_paper`).

The world is `test_api_chat.ApiWorld`: a real `ChatRuntime` over fakes. Its model validator, model
lister and client factory are fakes that decide by the key value, so nothing here reaches a
provider. The real `.env` is never touched: every write goes to the world's tmp env file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from types import SimpleNamespace

from fakes_trading import make_account
from finagent.agent.llm import LLMHandle, StubConversation, StubLLM, Turn
from finagent.app_state import ENV_NAMES
from finagent.config import Settings
from finagent.trading.broker import BrokerError
from test_api_chat import (
    ALPACA_ID,
    ALPACA_SECRET,
    ANTHROPIC,
    BAD_LLM_KEY,
    DOWN_LLM_KEY,
    GEMINI,
    HDR,
    LEAKY_LLM_KEY,
    LIMITED_LLM_KEY,
    LIVE_ID,
    LIVE_SECRET,
    NEW_GEMINI,
    NEW_ID,
    NEW_SECRET,
    ODD_LLM_KEY,
    OPENAI,
    PAPER_ID,
    PAPER_SECRET,
    SECRETS,
    TAVILY,
    ApiWorld,
    assert_no_secrets,
    chat,
    confirm,
    new_session,
    propose,
)

GEMINI_MODEL = "gemini-3.8-flash"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """A shell-exported key or LLM setting would show up in `env_fallbacks`."""
    for name in [*ENV_NAMES.values(), "API_ALLOW_REMOTE", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def gemini_world(tmp_path, **over) -> ApiWorld:
    """The default product configuration: Gemini with a key that is not declared paid."""
    fields = dict(llm_provider="gemini", llm_model=GEMINI_MODEL, gemini_paid=False)
    fields.update(over)
    world = ApiWorld(tmp_path, **fields)
    world.env_path.write_bytes(world.env_path.read_bytes().replace(
        b"LLM_PROVIDER=stub\n", f"LLM_PROVIDER=gemini\nGEMINI_API_KEY={GEMINI}\n".encode()))
    return world


@pytest.fixture
def world(tmp_path) -> ApiWorld:
    return gemini_world(tmp_path)


def answer_text(events: list[dict]) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "text-delta")


def no_secret_in(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


# --- GET /api/settings ------------------------------------------------------------------------------


def test_get_settings_reports_the_provider_and_the_shallow_mode(world):
    with world.client() as client:
        r = client.get("/api/settings")
        body = r.json()
        assert body["provider"] == "gemini" and body["model"] == GEMINI_MODEL and body["effort"] == "medium"
        assert (body["mode"], body["max_rounds"], body["time_budget_sec"]) == ("shallow", 4, 150)
        assert body["gemini_paid"] is False
        assert body["providers"] == {"gemini": "set", "openai": "missing", "anthropic": "missing"}
        assert body["gemini"] == "set"  # the older field is kept
        assert_no_secrets(r)


def test_get_settings_limits_follow_the_settings(tmp_path):
    world = gemini_world(tmp_path, shallow_max_rounds=3, shallow_time_budget_sec=90.5, openai_api_key=OPENAI)
    with world.client() as client:
        body = client.get("/api/settings").json()
        assert (body["mode"], body["max_rounds"], body["time_budget_sec"]) == ("shallow", 3, 90.5)
        assert body["providers"]["openai"] == "set"


# --- PUT /api/settings: switching provider and model ----------------------------------------------------


def test_switching_from_gemini_to_anthropic_with_a_valid_key(world):
    with world.client() as client:
        sid = new_session(client)
        world.legs = []  # (unused: the first model answers without proposing)
        world.llm._factory = lambda user_text: [Turn("Answer from the first model.", [], "end_turn")]
        assert answer_text(chat(client, sid, "hello")) == "Answer from the first model."
        handle, agent, first = world.runtime.llm, world.runtime.agent, world.runtime.llm.current
        assert isinstance(handle, LLMHandle) and first is world.llm
        assert len(agent._convs) == 1  # the provider conversation of this session is cached

        r = client.put("/api/settings", json={
            "llm_provider": "anthropic", "llm_model": "claude-test-opus", "anthropic_key": ANTHROPIC})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider"] == "anthropic" and body["model"] == "claude-test-opus"
        assert (body["mode"], body["max_rounds"], body["time_budget_sec"]) == ("deep", 12, 300)
        assert body["providers"] == {"gemini": "set", "openai": "missing", "anthropic": "set"}
        assert "warning" not in body
        assert_no_secrets(r)

        # one validation call, with the candidate configuration
        assert world.llm_checks == [("anthropic", "claude-test-opus", ANTHROPIC)]
        # the handle and the one ChatAgent survive; the client behind the handle is the new one
        assert world.runtime.llm is handle and world.runtime.agent is agent
        [switched] = world.llm_built
        assert handle.current is switched and (handle.name, handle.model) == ("anthropic", "claude-test-opus")
        assert agent._convs == {}  # provider conversations were dropped
        # .env: the provider line is replaced in place, the rest appended; a restart reads the same
        env = world.env()
        assert "LLM_PROVIDER=anthropic\n" in env and "LLM_PROVIDER=gemini" not in env
        assert "LLM_MODEL=claude-test-opus\n" in env and f"ANTHROPIC_API_KEY={ANTHROPIC}\n" in env
        assert f"GEMINI_API_KEY={GEMINI}\n" in env and "# local settings\n" in env
        reread = Settings(_env_file=world.env_path)
        assert (reread.llm_provider, reread.llm_model, reread.anthropic_api_key) == (
            "anthropic", "claude-test-opus", ANTHROPIC)
        assert world.settings.llm_provider == "anthropic" and world.settings.anthropic_api_key == ANTHROPIC

        # the next turn of the same session runs on the new model, rebuilt from the stored history
        events = chat(client, sid, "and now?")
        assert answer_text(events) == "Answer from anthropic/claude-test-opus."
        assert events[-1]["mode"] == "deep" and events[-1]["max_rounds"] == 12
        system, first_user_text = switched.started[0]
        assert first_user_text == "hello" and "up to 12 tool rounds" in system
        assert len(world.llm.started) == 1  # the old client was not used again
        # ... and so does the QA path behind /answer: it holds the same handle
        qa = client.get("/answer", params={"question": "hello"}).json()
        assert qa["think_trace"] == "trace by anthropic/claude-test-opus"
        assert client.get("/api/settings").json()["provider"] == "anthropic"


def test_switching_model_only_validates_and_swaps(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"llm_model": "gemini-3.8-pro"})
        assert r.status_code == 200 and r.json()["model"] == "gemini-3.8-pro" and r.json()["mode"] == "shallow"
        assert world.llm_checks == [("gemini", "gemini-3.8-pro", GEMINI)]
        assert world.runtime.llm.model == "gemini-3.8-pro"
        assert "LLM_MODEL=gemini-3.8-pro\n" in world.env()


def test_replacing_the_key_of_the_provider_in_use_validates_and_swaps(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"gemini_key": f"  {NEW_GEMINI}\n"})
        assert r.status_code == 200, r.text
        assert world.llm_checks == [("gemini", GEMINI_MODEL, NEW_GEMINI)]
        assert world.settings.gemini_api_key == NEW_GEMINI and f"GEMINI_API_KEY={NEW_GEMINI}\n" in world.env()
        assert len(world.llm_built) == 1 and world.runtime.llm.current is world.llm_built[0]
        assert_no_secrets(r)


def test_nothing_is_validated_or_swapped_when_the_configuration_does_not_change(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"llm_provider": "gemini", "llm_model": GEMINI_MODEL,
                                              "gemini_key": GEMINI, "llm_effort": "medium", "gemini_paid": False})
        assert r.status_code == 200
        assert world.llm_checks == [] and world.llm_built == []
        assert world.runtime.llm.current is world.llm


@pytest.mark.parametrize("body, code, status", [
    ({"llm_provider": "anthropic", "llm_model": "claude-test-opus", "anthropic_key": BAD_LLM_KEY},
     "llm_invalid_key", 400),
    ({"gemini_key": BAD_LLM_KEY}, "llm_invalid_key", 400),
    ({"llm_model": "no-such-model"}, "llm_model_unavailable", 400),
    ({"llm_provider": "openai", "llm_model": "no-such-model", "openai_key": OPENAI}, "llm_model_unavailable", 400),
    ({"llm_provider": "openai", "llm_model": "gpt-test-large", "openai_key": DOWN_LLM_KEY}, "llm_unreachable", 502),
    ({"llm_provider": "openai", "llm_model": "gpt-test-large", "openai_key": LEAKY_LLM_KEY}, "llm_unreachable", 502),
])
def test_a_failed_check_saves_nothing(world, caplog, body, code, status):
    with caplog.at_level(logging.DEBUG), world.client() as client:
        before = world.env_path.read_bytes()
        r = client.put("/api/settings", json={**body, "tavily_key": "", "trading_enabled": False})
        assert r.status_code == status and r.json()["error"] == code, r.text
        assert set(r.json()) == {"error", "message"}
        assert_no_secrets(r)
        assert "connection reset" not in r.text  # never the provider's (or the exception's) text
        assert world.env_path.read_bytes() == before  # byte for byte
        s = world.settings
        assert (s.llm_provider, s.llm_model, s.gemini_api_key) == ("gemini", GEMINI_MODEL, GEMINI)
        assert s.openai_api_key is None and s.anthropic_api_key is None
        assert s.tavily_api_key == TAVILY and s.trading_enabled is True  # the other fields too: all or nothing
        assert world.runtime.llm.current is world.llm and world.llm_built == []
        assert len(world.llm_checks) == 1
        assert client.get("/api/settings").json()["provider"] == "gemini"
    no_secret_in(caplog.text)
    assert "connection reset" not in caplog.text


def test_a_rate_limited_key_is_saved_with_a_warning(world):
    with world.client() as client:
        r = client.put("/api/settings", json={
            "llm_provider": "openai", "llm_model": "gpt-test-large", "openai_key": LIMITED_LLM_KEY})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["warning"] == "llm_rate_limited"
        assert body["provider"] == "openai" and body["mode"] == "deep" and body["providers"]["openai"] == "set"
        assert_no_secrets(r)
        assert f"OPENAI_API_KEY={LIMITED_LLM_KEY}\n" in world.env() and "LLM_PROVIDER=openai\n" in world.env()
        assert world.runtime.llm.name == "openai"
        assert "warning" not in client.get("/api/settings").json()  # the warning belongs to that save only


@pytest.mark.parametrize("body", [
    {"llm_provider": "openai", "llm_model": "gpt-test-large"},   # no OpenAI key saved, none given
    {"llm_provider": "anthropic"},
    {"gemini_key": ""},                                          # removing the key of the provider in use
    {"llm_provider": "openai", "llm_model": "gpt-test-large", "openai_key": ""},
])
def test_the_chosen_provider_must_have_a_key(world, body):
    with world.client() as client:
        before = world.env_path.read_bytes()
        r = client.put("/api/settings", json=body)
        assert r.status_code == 400 and r.json()["error"] == "llm_key_required", r.text
        assert world.llm_checks == []  # refused before any live check
        assert world.env_path.read_bytes() == before
        assert world.settings.llm_provider == "gemini" and world.settings.gemini_api_key == GEMINI


def test_a_model_must_be_named(world):
    with world.client() as client:
        before = world.env_path.read_bytes()
        for body in ({"llm_model": ""}, {"llm_model": "   "},
                     {"llm_provider": "anthropic", "anthropic_key": ANTHROPIC, "llm_model": ""}):
            r = client.put("/api/settings", json=body)
            assert r.status_code == 400 and r.json()["error"] == "llm_model_required", r.text
        assert world.llm_checks == [] and world.env_path.read_bytes() == before
        assert world.settings.llm_model == GEMINI_MODEL


def test_gemini_paid_toggles_the_mode_without_a_validation_call(world):
    with world.client() as client:
        sid = new_session(client)
        world.llm._factory = lambda user_text: [Turn("ok", [], "end_turn")]
        chat(client, sid, "hello")
        assert "shallow mode" in world.llm.started[-1][0]

        r = client.put("/api/settings", json={"gemini_paid": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["gemini_paid"] is True
        assert (body["mode"], body["max_rounds"], body["time_budget_sec"]) == ("deep", 12, 300)
        assert world.llm_checks == [] and world.llm_built == []  # no live check, no new client
        assert world.runtime.llm.current is world.llm
        assert "GEMINI_PAID=true\n" in world.env()
        assert Settings(_env_file=world.env_path).gemini_paid is True

        events = chat(client, sid, "again")  # the conversation is rebuilt with the deep prompt
        assert events[-1]["mode"] == "deep" and events[-1]["max_rounds"] == 12
        assert len(world.llm.started) == 2 and "up to 12 tool rounds" in world.llm.started[-1][0]

        r = client.put("/api/settings", json={"gemini_paid": False})
        assert r.json()["mode"] == "shallow" and "GEMINI_PAID=false\n" in world.env()
        assert world.llm_checks == []


def test_effort_builds_a_new_client_without_a_validation_call(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"llm_effort": "high"})
        assert r.status_code == 200 and r.json()["effort"] == "high"
        assert world.llm_checks == []
        [built] = world.llm_built
        assert built.effort == "high" and world.runtime.llm.current is built
        assert "LLM_EFFORT=high\n" in world.env()


def test_a_key_for_another_provider_is_saved_without_a_check(world):
    with world.client() as client:
        r = client.put("/api/settings", json={"openai_key": OPENAI, "anthropic_key": ANTHROPIC})
        assert r.status_code == 200, r.text
        assert r.json()["providers"] == {"gemini": "set", "openai": "set", "anthropic": "set"}
        assert r.json()["provider"] == "gemini"
        assert world.llm_checks == [] and world.llm_built == []  # the provider in use did not change
        assert f"OPENAI_API_KEY={OPENAI}\n" in world.env() and f"ANTHROPIC_API_KEY={ANTHROPIC}\n" in world.env()
        assert_no_secrets(r)
        # switching to it later is checked with the saved key
        r = client.put("/api/settings", json={"llm_provider": "openai", "llm_model": "gpt-test-large"})
        assert r.status_code == 200 and world.llm_checks == [("openai", "gpt-test-large", OPENAI)]
        # and a key that is not in use can be removed
        r = client.put("/api/settings", json={"anthropic_key": ""})
        assert r.status_code == 200 and r.json()["providers"]["anthropic"] == "missing"
        assert "ANTHROPIC_API_KEY" not in world.env()


def test_llm_fields_follow_the_value_rules(world):
    with world.client() as client:
        before = world.env_path.read_bytes()
        for field in ("gemini_key", "openai_key", "anthropic_key", "llm_model"):
            for bad in (f"{NEW_GEMINI}\nEVIL=1", "키-값", 'a"b', "a\\b", "${HOME}"):
                r = client.put("/api/settings", json={field: bad})
                assert r.status_code == 400 and r.json()["error"] == "invalid_value", (field, bad, r.text)
                assert r.json()["field"] == field
                assert_no_secrets(r)
        for body in ({"llm_provider": "stub"}, {"llm_provider": "mistral"}, {"llm_effort": "extreme"},
                     {"gemini_paid": "yes"}, {"gemini_key": NEW_GEMINI * 20}, {"llm_provider": None, "x": 1}):
            r = client.put("/api/settings", json=body)
            assert r.status_code == 422, (body, r.text)
            assert_no_secrets(r)
        assert client.put("/api/settings", json={"llm_provider": None, "gemini_key": None}).status_code == 200
        assert world.env_path.read_bytes() == before and world.llm_checks == []


def test_a_failed_env_write_changes_nothing_and_retires_the_new_client(world, monkeypatch):
    import finagent.app_state as app_state

    def boom(path, updates):
        raise OSError("disk full")

    with world.client() as client:
        monkeypatch.setattr(app_state, "update_env_file", boom)
        r = client.put("/api/settings", json={
            "llm_provider": "anthropic", "llm_model": "claude-test-opus", "anthropic_key": ANTHROPIC})
        assert r.status_code == 500 and r.json()["error"] == "settings_not_saved"
        assert world.settings.llm_provider == "gemini" and world.settings.anthropic_api_key is None
        assert world.runtime.llm.current is world.llm
        [built] = world.llm_built
        for _ in range(100):
            if built.closed:
                break
            time.sleep(0.01)
        assert built.closed  # made for the switch, never put in service


# --- secrets and connectors after a switch ------------------------------------------------------------------


def test_known_secret_lists_hold_every_provider_key_old_and_new(world, caplog):
    with world.client() as client:
        assert client.put("/api/settings", json={"gemini_key": NEW_GEMINI}).status_code == 200
        assert client.put("/api/settings", json={
            "llm_provider": "anthropic", "llm_model": "claude-test-opus", "anthropic_key": ANTHROPIC,
            "openai_key": OPENAI}).status_code == 200
        client.put("/api/settings", json={"gemini_key": BAD_LLM_KEY, "llm_provider": "gemini"})  # refused
        sid = new_session(client)
        keys = [GEMINI, NEW_GEMINI, ANTHROPIC, OPENAI, BAD_LLM_KEY]
        text = " ".join(keys)
        # the ledger
        world.runtime.ledger.append(sid, "note", {"text": text})
        exported = "".join(world.runtime.ledger.export_jsonl(sid))
        # the log scrubber
        logging.getLogger("finagent.probe").warning("keys %s", text)
        # the toolbox scrub (what a broker error message goes through)
        scrubbed = world.runtime.toolbox._scrub(text)
        for key in keys[:4]:
            assert key not in exported and key not in scrubbed
        assert "keys [redacted] [redacted] [redacted] [redacted] [redacted]" in caplog.text
    no_secret_in(caplog.text)


def test_grounding_is_built_from_the_gemini_key_whatever_the_chat_provider(world):
    with world.client() as client:
        assert world.runtime.grounded is not None
        assert client.put("/api/settings", json={
            "llm_provider": "anthropic", "llm_model": "claude-test-opus", "anthropic_key": ANTHROPIC,
        }).status_code == 200
        assert world.runtime.grounded is not None  # Gemini search grounding, with Anthropic as the chat model
        assert "grounded_web_answer" in {s.name for s in world.runtime.toolbox.specs()}
        assert world.runtime.grounded._api_key == GEMINI
        # a new Gemini key reaches the connector; removing it removes the tool
        assert client.put("/api/settings", json={"gemini_key": NEW_GEMINI}).status_code == 200
        assert world.runtime.grounded._api_key == NEW_GEMINI and world.llm_checks[-1][0] == "anthropic"
        assert client.put("/api/settings", json={"gemini_key": ""}).status_code == 200
        assert world.runtime.grounded is None
        assert "grounded_web_answer" not in {s.name for s in world.runtime.toolbox.specs()}


def test_grounding_needs_only_the_key_at_startup(tmp_path):
    with ApiWorld(tmp_path, llm_provider="stub").client() as _client:
        pass
    world = ApiWorld(tmp_path, llm_provider="anthropic", llm_model="claude-test-opus", anthropic_api_key=ANTHROPIC)
    runtime = world.state().chat
    assert runtime.grounded is not None  # the world has a Gemini key
    (tmp_path / "n").mkdir()
    none = ApiWorld(tmp_path / "n", gemini_api_key=None).state().chat
    assert none.grounded is None


# --- a turn in flight during a switch -------------------------------------------------------------------------


class GatedConversation(StubConversation):
    def __init__(self, script, system, user_text, gate: asyncio.Event, entered: asyncio.Event):
        super().__init__(script, system, user_text)
        self.gate, self.entered = gate, entered

    async def step(self):
        self.entered.set()
        await self.gate.wait()
        return await super().step()


class GatedLLM(StubLLM):
    name, model = "gemini", GEMINI_MODEL

    def __init__(self):
        super().__init__()
        self.gate, self.entered = asyncio.Event(), asyncio.Event()
        self.closed = False

    def start(self, system, tools, user_text):
        self.started.append((system, user_text))
        return GatedConversation([Turn("Answer from the old model.", [], "end_turn")], system, user_text,
                                 self.gate, self.entered)

    async def aclose(self) -> None:
        self.closed = True


async def test_a_turn_in_flight_finishes_on_the_client_it_started_with(tmp_path):
    world = gemini_world(tmp_path)
    world.close_grace_sec = 60.0
    old = world.llm = GatedLLM()
    runtime = world.state().chat
    try:
        sid = runtime.sessions.create()["id"]
        events: list[dict] = []
        turn = asyncio.create_task(runtime.agent.run_turn(sid, "first", events.append))
        await asyncio.wait_for(old.entered.wait(), 5)

        await runtime.apply_settings({"llm_provider": "anthropic", "llm_model": "claude-test-opus",
                                      "anthropic_api_key": ANTHROPIC})
        assert runtime.llm.name == "anthropic" and not old.closed  # retired, not closed: a turn still uses it
        old.gate.set()
        assert await asyncio.wait_for(turn, 5) == "Answer from the old model."
        # the turn keeps the limits it started with (shallow), although the mode is deep now
        assert (events[-1]["mode"], events[-1]["max_rounds"]) == ("shallow", 4)

        more: list[dict] = []
        assert await runtime.agent.run_turn(sid, "second", more.append) == "Answer from anthropic/claude-test-opus."
        assert more[-1]["mode"] == "deep" and len(old.started) == 1
        [switched] = world.llm_built
        assert switched.started[0][1] == "first"  # rebuilt from the stored text history
    finally:
        await runtime.shutdown()
    assert old.closed  # shutdown closes what the grace delay had not yet


class Timeline:
    """Virtual time for the runtime's grace-close sleeps (`close_sleep`): `sleep(d)` returns once
    `advance` has moved the clock to its deadline. `requested` keeps every delay asked for."""

    def __init__(self):
        self.now = 0.0
        self.waiters: list[tuple[float, asyncio.Future]] = []
        self.requested: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.requested.append(seconds)
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + seconds, future))
        await future

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for due, future in list(self.waiters):
            if due <= self.now:
                self.waiters.remove((due, future))
                if not future.done():
                    future.set_result(None)
        for _ in range(20):  # let the woken close tasks run to the end
            await asyncio.sleep(0)


async def test_the_swapped_client_outlives_the_longest_turn(tmp_path):
    """The old client is closed only after the longest turn the settings allow, plus the nudge
    and one model call: a deep turn (300 s) that started on it must be able to finish."""
    world = gemini_world(tmp_path)
    timeline = Timeline()
    world.close_sleep = timeline.sleep
    old = world.llm = GatedLLM()
    runtime = world.state().chat
    try:
        sid = runtime.sessions.create()["id"]
        turn = asyncio.create_task(runtime.agent.run_turn(sid, "first", lambda event: None))
        await asyncio.wait_for(old.entered.wait(), 5)
        await runtime.apply_settings({"llm_provider": "anthropic", "llm_model": "claude-test-opus",
                                      "anthropic_api_key": ANTHROPIC})
        await timeline.advance(0)  # let the close tasks start their waits
        s = runtime.settings
        grace = max(timeline.requested)
        assert grace >= max(s.chat_time_budget_sec, s.shallow_time_budget_sec) + 20 + s.llm_timeout_sec
        await timeline.advance(130)  # past the old fixed 120 s grace
        assert not old.closed
        await timeline.advance(grace - 140)  # 10 s before the deadline
        assert not old.closed
        old.gate.set()
        assert await asyncio.wait_for(turn, 5) == "Answer from the old model."
        await timeline.advance(20)
        assert old.closed and not runtime.llm.current.closed
    finally:
        await runtime.shutdown()


async def test_http_clients_and_brokers_keep_their_own_grace(tmp_path):
    world = gemini_world(tmp_path)
    timeline = Timeline()
    world.close_sleep = timeline.sleep
    world.close_grace_sec = 7.0
    runtime = world.state().chat
    try:
        old_broker = runtime.broker
        await runtime.apply_settings({"alpaca_api_key_id": NEW_ID, "alpaca_api_secret_key": NEW_SECRET},
                                     alpaca_status="connected")
        await timeline.advance(0)
        assert timeline.requested == [7.0]
        await timeline.advance(7)
        assert old_broker.closed
    finally:
        await runtime.shutdown()


# --- POST /api/llm/models ---------------------------------------------------------------------------------------


def test_models_are_listed_filtered_and_suggested(world):
    with world.client() as client:
        r = client.post("/api/llm/models", json={"provider": "anthropic", "api_key": ANTHROPIC})
        assert r.status_code == 200, r.text
        assert r.json() == {
            "provider": "anthropic",
            "models": [{"id": "claude-test-sonnet", "label": "Claude Test Sonnet"},
                       {"id": "claude-test-opus", "label": "Claude Test Opus"},
                       {"id": "claude-test-opus-old", "label": "Claude Test Opus (old)"}],
            "suggested": "claude-test-opus",  # the newest Opus
        }
        assert_no_secrets(r)
        r = client.post("/api/llm/models", json={"provider": "openai", "api_key": f" {OPENAI}\n"})
        assert r.json() == {"provider": "openai", "suggested": "gpt-test-large",
                            "models": [{"id": "gpt-test-large", "label": "gpt-test-large"},
                                       {"id": "gpt-test-small", "label": "gpt-test-small"}]}
        assert world.model_requests == [("anthropic", ANTHROPIC), ("openai", OPENAI)]
        # nothing was stored
        assert world.settings.anthropic_api_key is None and world.settings.openai_api_key is None
        assert ANTHROPIC not in world.env() and OPENAI not in world.env()


def test_models_use_the_saved_key_and_suggest_the_model_in_use(world):
    with world.client() as client:
        r = client.post("/api/llm/models", json={"provider": "gemini"})
        assert r.status_code == 200, r.text
        assert r.json() == {"provider": "gemini", "suggested": GEMINI_MODEL,
                            "models": [{"id": "gemini-3.8-pro", "label": "gemini-3.8-pro"},
                                       {"id": "gemini-3.8-flash", "label": "gemini-3.8-flash"}]}
        assert world.model_requests == [("gemini", GEMINI)]
        assert_no_secrets(r)
        for body in ({"provider": "gemini", "api_key": None}, {"provider": "gemini", "api_key": "  "}):
            assert client.post("/api/llm/models", json=body).status_code == 200
        assert world.model_requests == [("gemini", GEMINI)] * 3
        # a provider without a saved key, and no key given
        r = client.post("/api/llm/models", json={"provider": "openai"})
        assert r.status_code == 400 and r.json()["error"] == "llm_key_required"
        assert len(world.model_requests) == 3


def test_the_model_list_is_capped_at_sixty(world):
    world.model_lists["openai"] = [{"id": f"gpt-test-{n}", "label": None, "created": float(n)} for n in range(150)]
    with world.client() as client:
        body = client.post("/api/llm/models", json={"provider": "openai", "api_key": OPENAI}).json()
        assert len(body["models"]) == 60 and body["models"][0]["id"] == "gpt-test-149"
        assert body["suggested"] == "gpt-test-149"


@pytest.mark.parametrize("key, status, code", [
    (BAD_LLM_KEY, 400, "llm_invalid_key"),
    (LIMITED_LLM_KEY, 429, "llm_rate_limited"),
    (DOWN_LLM_KEY, 502, "llm_unreachable"),
    (ODD_LLM_KEY, 502, "llm_list_failed"),  # a provider 400 that is not about the key
])
def test_model_list_errors_never_carry_the_key_or_the_provider_text(world, caplog, key, status, code):
    with caplog.at_level(logging.DEBUG), world.client() as client:
        before = world.env_path.read_bytes()
        r = client.post("/api/llm/models", json={"provider": "openai", "api_key": key})
        assert r.status_code == status and r.json()["error"] == code, r.text
        assert set(r.json()) == {"error", "message"}
        assert_no_secrets(r)
        for fragment in ("Incorrect API key", "rate limit reached", "connection reset", "unsupported listing"):
            assert fragment not in r.text
        assert world.env_path.read_bytes() == before and world.settings.openai_api_key is None
        logging.getLogger("finagent.probe").warning("later %s", key)  # even a later log line is scrubbed
    no_secret_in(caplog.text)
    for fragment in ("Incorrect API key", "rate limit reached", "connection reset"):
        assert fragment not in caplog.text


def test_model_list_requests_are_validated_and_guarded(world):
    with world.client() as client:
        for body in ({}, {"provider": "stub"}, {"provider": "mistral"}, {"provider": "openai", "extra": 1},
                     {"provider": "openai", "api_key": OPENAI * 20}, {"provider": "openai", "api_key": 7}):
            r = client.post("/api/llm/models", json=body)
            assert r.status_code == 422 and r.json()["error"] == "invalid_request", (body, r.text)
            assert_no_secrets(r)
        r = client.post("/api/llm/models", json={"provider": "openai", "api_key": "sk-키"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_value" and r.json()["field"] == "api_key"
        assert world.model_requests == []
        # the key belongs in the body: there is no GET, and a key in the query string is not read
        assert client.get("/api/llm/models", params={"provider": "openai", "api_key": OPENAI}).status_code == 405
    # the guard applies as to every /api POST
    with world.client(headers={}) as client:
        r = client.post("/api/llm/models", json={"provider": "gemini"})
        assert r.status_code == 403 and r.json()["error"] == "csrf_header_missing"
    with world.client(client=("203.0.113.9", 40000)) as client:
        r = client.post("/api/llm/models", json={"provider": "gemini"})
        assert r.status_code == 403 and r.json()["error"] == "local_only"
    with world.client() as client:
        r = client.post("/api/llm/models", json={"provider": "gemini"}, headers={**HDR, "origin": "http://evil.example"})
        assert r.status_code == 403 and r.json()["error"] == "origin_mismatch"
    assert world.model_requests == []


def test_model_list_is_503_without_a_chat_runtime(tmp_path):
    with ApiWorld(tmp_path, chat=False).client() as client:
        r = client.post("/api/llm/models", json={"provider": "gemini"})
        assert r.status_code == 503 and r.json()["error"] == "chat_unavailable"


# --- Part B: the runtime honours the paper flag ---------------------------------------------------------------------


def test_a_live_account_at_startup_is_not_paper_and_gets_no_broker(tmp_path):
    world = ApiWorld(tmp_path, alpaca_api_key_id=LIVE_ID, alpaca_api_secret_key=LIVE_SECRET)
    with world.client() as client:
        r = client.get("/api/settings")
        assert r.json()["alpaca"] == "not_paper"
        assert_no_secrets(r)
        rt = world.runtime
        assert rt.broker is None and rt.executor is None and rt.data is None
        names = {s.name for s in rt.toolbox.specs()}
        assert not names & {"propose_trades", "get_account", "get_positions", "list_orders", "get_quotes"}
        assert client.get("/api/portfolio").json() == {"connected": False}
        sid = new_session(client)
        r = client.post("/api/tickets/TABC/confirm", json={"session_id": sid, "hash": "0" * 64})
        assert r.status_code == 503 and r.json()["error"] == "broker_not_connected"
        r = client.post("/api/tickets/TABC/refresh", json={"session_id": sid})
        assert r.status_code == 503
        assert world.total_submits() == 0
        # an unrelated settings change rebuilds the connectors: still no broker for these keys
        assert client.put("/api/settings", json={"tavily_key": ""}).status_code == 200
        assert rt.broker is None and rt.executor is None
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"
        # valid paper keys bring trading back
        r = client.put("/api/settings", json={"alpaca_key_id": NEW_ID, "alpaca_secret": NEW_SECRET})
        assert r.status_code == 200 and r.json()["alpaca"] == "connected"
        assert rt.broker is not None and rt.executor is not None
        assert "propose_trades" in {s.name for s in rt.toolbox.specs()}


def test_put_settings_refuses_the_keys_of_a_live_account(world):
    with world.client() as client:
        before = world.env_path.read_bytes()
        old_broker = world.runtime.broker
        r = client.put("/api/settings", json={"alpaca_key_id": LIVE_ID, "alpaca_secret": LIVE_SECRET,
                                              "tavily_key": ""})
        assert r.status_code == 400 and r.json()["error"] == "alpaca_not_paper", r.text
        assert set(r.json()) == {"error", "message"}
        assert_no_secrets(r)
        assert world.env_path.read_bytes() == before
        assert world.settings.alpaca_api_key_id == ALPACA_ID and world.settings.alpaca_api_secret_key == ALPACA_SECRET
        assert world.settings.tavily_api_key == TAVILY
        assert world.runtime.broker is old_broker
        assert world.brokers[-1].key_id == LIVE_ID and world.brokers[-1].closed  # the probe was closed
        assert client.get("/api/settings").json()["alpaca"] == "connected"


def test_portfolio_paper_reflects_the_account(tmp_path, world):
    with world.client() as client:  # the default fake account does not say: unknown
        body = client.get("/api/portfolio").json()
        assert body["connected"] is True and body["paper"] is None and body["account"]["paper"] is None
        assert client.get("/api/settings").json()["alpaca"] == "connected"  # unknown keeps the old behaviour
    (tmp_path / "p").mkdir()
    paper = ApiWorld(tmp_path / "p", alpaca_api_key_id=PAPER_ID, alpaca_api_secret_key=PAPER_SECRET)
    with paper.client() as client:
        body = client.get("/api/portfolio").json()
        assert body["paper"] is True and body["account"]["paper"] is True
        assert client.get("/api/settings").json()["alpaca"] == "connected"


def test_an_account_that_turns_out_not_to_be_paper_is_dropped_at_the_portfolio(world):
    from fakes_trading import make_account

    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        broker = world.runtime.broker
        broker.account_snapshot = make_account(paper=False)  # e.g. the first check could not tell
        r = client.get("/api/portfolio")
        assert r.status_code == 200 and r.json() == {"connected": False}
        assert world.runtime.broker is None and world.runtime.executor is None
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"
        assert confirm(client, ticket, sid).status_code == 503
        assert world.total_submits() == 0


def test_the_ledger_export_json_stays_valid_after_a_switch(world):
    with world.client() as client:
        sid = new_session(client)
        assert client.put("/api/settings", json={"gemini_paid": True}).status_code == 200
        world.runtime.ledger.append(sid, "note", {"text": "ok"})
        lines = [json.loads(line) for line in client.get(f"/api/ledger/{sid}.jsonl").text.splitlines() if line]
        assert [line["kind"] for line in lines] == ["note"]


# --- fix round 1, I1: the paper flag where trades happen ---------------------------------------------------------


def _account_down_at_startup(runtime) -> None:
    runtime.broker.account_error = BrokerError(503, None, "service unavailable")


@pytest.mark.parametrize("startup", ["unreachable", "connected"])
def test_a_confirm_on_an_account_that_turns_out_live_submits_nothing(tmp_path, startup):
    """The reviewer's probe: the startup check could not tell (a 503, or no `paper` in the
    response), a ticket was proposed, and then the account reports `paper: false`."""
    world = ApiWorld(tmp_path)
    if startup == "unreachable":
        world.before_start = _account_down_at_startup
    with world.client() as client:
        assert world.runtime.alpaca_status == startup
        broker = world.runtime.broker
        broker.account_error = None
        sid = new_session(client)
        ticket = propose(client, world, sid)  # the account does not say: a ticket is created
        broker.account_snapshot = make_account(paper=False)
        r = confirm(client, ticket, sid)
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["status"] == "rejected" and "not a paper account" in body["message"]
        assert body["ticket"]["status"] == "failed" and body["orders"] == []
        assert world.total_submits() == 0
        # the runtime dropped the broker: later calls see no broker
        assert world.runtime.broker is None and world.runtime.executor is None
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"
        assert client.get("/api/portfolio").json() == {"connected": False}
        r = confirm(client, ticket, sid)
        assert r.status_code == 503 and r.json()["error"] == "broker_not_connected"
        kinds = [json.loads(line)["kind"] for line in world.runtime.ledger.export_jsonl(sid)]
        assert "submit" not in kinds and "result" in kinds
        assert world.total_submits() == 0


def test_proposing_on_a_live_account_creates_no_ticket(tmp_path):
    world = ApiWorld(tmp_path)
    with world.client() as client:
        world.runtime.broker.account_snapshot = make_account(paper=False)
        sid = new_session(client)
        world.add_evidence(sid)
        events = chat(client, sid)
        assert not [e for e in events if e["type"] == "data-trade-proposal"]
        [out] = [e for e in events if e["type"] == "tool-output-available"]
        assert out["isError"] is True and "not a paper account" in out["output"]
        assert world.runtime.db.query("SELECT COUNT(*) AS n FROM tickets")[0]["n"] == 0
        assert world.runtime.broker is None
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"
        rows = [json.loads(line) for line in world.runtime.ledger.export_jsonl(sid)]
        rejected = [row["payload"] for row in rows if row["kind"] == "proposal_rejected"]
        assert rejected == [{"turn_id": events[0]["messageId"], "reason": "not_paper_account"}]
        assert world.total_submits() == 0


def test_a_refresh_on_a_live_account_creates_nothing(tmp_path):
    world = ApiWorld(tmp_path)
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.runtime.broker.account_snapshot = make_account(paper=False)
        r = client.post(f"/api/tickets/{ticket['id']}/refresh", json={"session_id": sid})
        assert r.status_code == 409 and r.json()["status"] == "rejected"
        assert "not a paper account" in r.json()["message"] and r.json()["new_ticket"] is None
        assert world.runtime.broker is None


def test_get_settings_rechecks_an_unsettled_alpaca_status_at_most_once_a_minute(tmp_path):
    world = ApiWorld(tmp_path)
    now = [1000.0]
    world.monotonic = lambda: now[0]
    world.before_start = _account_down_at_startup
    with world.client() as client:
        broker = world.runtime.broker
        checks = broker.calls.count("account")
        broker.account_error = None  # Alpaca answers again
        now[0] += 30
        assert client.get("/api/settings").json()["alpaca"] == "unreachable"  # checked under 60 s ago
        assert broker.calls.count("account") == checks
        now[0] += 31
        assert client.get("/api/settings").json()["alpaca"] == "connected"
        assert broker.calls.count("account") == checks + 1
        now[0] += 600
        assert client.get("/api/settings").json()["alpaca"] == "connected"  # settled: not re-checked
        assert broker.calls.count("account") == checks + 1


def test_get_settings_recheck_drops_an_account_that_turned_out_live(tmp_path):
    world = ApiWorld(tmp_path)
    now = [1000.0]
    world.monotonic = lambda: now[0]
    world.before_start = _account_down_at_startup
    with world.client() as client:
        broker = world.runtime.broker
        broker.account_error = None
        broker.account_snapshot = make_account(paper=False)
        now[0] += 61
        r = client.get("/api/settings")
        assert r.json()["alpaca"] == "not_paper"
        assert world.runtime.broker is None and world.runtime.executor is None
        now[0] += 61
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"  # settled


# --- fix round 1, minor 2: dropping only the broker that read the account ---------------------------------------


async def test_a_broker_that_was_already_replaced_is_not_dropped(tmp_path):
    world = ApiWorld(tmp_path)
    runtime = world.state().chat
    try:
        stale = runtime.broker
        await runtime.apply_settings({"alpaca_api_key_id": NEW_ID, "alpaca_api_secret_key": NEW_SECRET},
                                     alpaca_status="connected")
        current = runtime.broker
        assert current is not stale
        await runtime.drop_non_paper_broker(stale)  # the old broker read a live account: too late to matter
        assert runtime.broker is current and runtime.alpaca_status == "connected"
        await runtime.drop_non_paper_broker(current)
        assert runtime.broker is None and runtime.alpaca_status == "not_paper"
    finally:
        await runtime.shutdown()


def test_keys_once_found_live_work_again_when_a_later_check_says_paper(tmp_path):
    world = ApiWorld(tmp_path, alpaca_api_key_id=LIVE_ID, alpaca_api_secret_key=LIVE_SECRET)
    with world.client() as client:
        assert client.get("/api/settings").json()["alpaca"] == "not_paper"
        world.live_paper = True  # e.g. the user re-entered the keys after Alpaca fixed the account
        r = client.put("/api/settings", json={"alpaca_key_id": LIVE_ID, "alpaca_secret": LIVE_SECRET})
        assert r.status_code == 200 and r.json()["alpaca"] == "connected", r.text
        assert world.runtime.broker is not None and world.runtime.executor is not None


# --- fix round 1, minor 3: two settings writes at once ------------------------------------------------------------


async def test_interleaved_puts_validate_the_configuration_that_is_applied(tmp_path):
    from finagent.api_chat import SettingsUpdate, put_settings
    from finagent.api_common import ApiError

    world = gemini_world(tmp_path)
    runtime = world.state().chat
    try:
        await runtime.lock.acquire()  # e.g. a confirm is running: both writes wait for the lock
        switch = asyncio.create_task(put_settings(
            SettingsUpdate(llm_provider="openai", llm_model="gpt-test-large", openai_key=OPENAI), runtime))
        key_only = asyncio.create_task(put_settings(SettingsUpdate(openai_key=BAD_LLM_KEY), runtime))
        for _ in range(20):
            await asyncio.sleep(0)
        # the switch was checked; the key-only write needed no check (OpenAI was not in use when it decided)
        assert world.llm_checks == [("openai", "gpt-test-large", OPENAI)]
        runtime.lock.release()
        assert (await switch).status_code == 200
        with pytest.raises(ApiError) as caught:
            await key_only
        # applied after the switch, the key-only write would change the key in use: checked again
        assert caught.value.code == "llm_invalid_key" and caught.value.status == 400
        assert world.llm_checks[-1] == ("openai", "gpt-test-large", BAD_LLM_KEY)
        assert runtime.settings.llm_provider == "openai" and runtime.settings.openai_api_key == OPENAI
        assert BAD_LLM_KEY not in world.env() and f"OPENAI_API_KEY={OPENAI}" in world.env()
    finally:
        if runtime.lock.locked():
            runtime.lock.release()
        await runtime.shutdown()


async def test_a_write_that_still_matches_what_was_checked_is_not_checked_twice(tmp_path):
    from finagent.api_chat import SettingsUpdate, put_settings

    world = gemini_world(tmp_path)
    runtime = world.state().chat
    try:
        await runtime.lock.acquire()
        first = asyncio.create_task(put_settings(SettingsUpdate(llm_model="gemini-3.8-pro"), runtime))
        second = asyncio.create_task(put_settings(SettingsUpdate(tavily_key=""), runtime))
        for _ in range(20):
            await asyncio.sleep(0)
        runtime.lock.release()
        assert (await first).status_code == 200 and (await second).status_code == 200
        assert world.llm_checks == [("gemini", "gemini-3.8-pro", GEMINI)]
    finally:
        if runtime.lock.locked():
            runtime.lock.release()
        await runtime.shutdown()


# --- fix round 1, minor 7: one handle for /answer and /api --------------------------------------------------------


def test_app_state_load_hands_the_same_handle_to_the_orchestrator_and_the_runtime(monkeypatch):
    import finagent.app_state as app_state

    seen: dict = {}

    def orchestrator(settings, llm, *rest):
        seen["orchestrator"] = llm
        return SimpleNamespace(llm=llm)

    def chat_runtime(settings, *, llm, **kwargs):
        seen["runtime"] = llm
        return SimpleNamespace(llm=llm)

    # `AppState.load` is heavy (it reads the product store): everything but the wiring is faked
    monkeypatch.setattr(app_state, "RDB", lambda *a, **k: SimpleNamespace(meta=lambda: {}))
    monkeypatch.setattr(app_state, "Lexicon", SimpleNamespace(load=lambda path: "lexicon"))
    monkeypatch.setattr(app_state, "Ontology", SimpleNamespace(load=lambda path: SimpleNamespace(graph=[])))
    monkeypatch.setattr(app_state, "SearchIndex", SimpleNamespace(from_rdb=lambda rdb: SimpleNamespace(docs=[])))
    monkeypatch.setattr(app_state, "EntityResolver", lambda *a: "resolver")
    monkeypatch.setattr(app_state, "DocStore", SimpleNamespace(load=lambda path: SimpleNamespace(titles=lambda: [])))
    monkeypatch.setattr(app_state, "ToolRegistry", lambda *a, **k: "tools")
    monkeypatch.setattr(app_state, "build_system_prompt", lambda *a, **k: "system")
    monkeypatch.setattr(app_state, "value_hints", lambda rdb: {})
    monkeypatch.setattr(app_state, "Orchestrator", orchestrator)
    monkeypatch.setattr(app_state, "ChatRuntime", chat_runtime)
    client = StubLLM([])
    state = app_state.AppState.load(Settings(_env_file=None, llm_provider="stub", vector_backend="off"),
                                    llm=client, build_graph=False)
    assert isinstance(state.llm, LLMHandle) and state.llm.current is client
    assert seen["orchestrator"] is seen["runtime"] is state.llm is state.chat.llm
