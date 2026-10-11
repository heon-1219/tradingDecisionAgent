"""Tests for how `scripts/eval_scenarios.py` resolves a turn's limits on the backend with and
without `chat_limits`, what goes into a result line's configuration, and script hygiene
(`sys.path`, the docs). Offline; the runner itself is tested in `test_eval_scenarios.py`.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))  # at the end: scripts/trace.py must never shadow the stdlib

import _fakeworld as fw  # noqa: E402
import eval_scenarios as ev  # noqa: E402

from finagent.config import Settings  # noqa: E402


def _settings(tmp_path, **over) -> Settings:
    fields = dict(llm_provider="stub", alpaca_api_key_id=None, alpaca_api_secret_key=None, tavily_api_key=None,
                  gemini_api_key=None, anthropic_api_key=None, openai_api_key=None, sec_user_agent=None,
                  app_db_path=tmp_path / "never-used.db")
    fields.update(over)
    return Settings(_env_file=None, **fields)


# --- limits: the old backend and the one with chat_limits ------------------------------------------------


def test_effective_limits_without_chat_limits_come_from_the_settings(monkeypatch):
    monkeypatch.setattr(ev, "_chat_limits_fn", lambda: None)
    settings = types.SimpleNamespace(chat_time_budget_sec=150.0, chat_max_rounds=8)
    assert ev.effective_limits(settings) == {"time_budget_sec": 150.0, "max_rounds": 8, "mode": None}


@pytest.mark.parametrize("limits", [
    types.SimpleNamespace(mode="shallow", max_rounds=4, time_budget_sec=150),
    {"mode": "shallow", "max_rounds": 4, "time_budget_sec": 150},
    types.SimpleNamespace(mode=types.SimpleNamespace(value="shallow"), rounds=4, time_budget=150),
])
def test_effective_limits_read_chat_limits_in_any_reasonable_shape(monkeypatch, limits):
    monkeypatch.setattr(ev, "_chat_limits_fn", lambda: (lambda settings: limits))
    settings = types.SimpleNamespace(chat_time_budget_sec=90.0, chat_max_rounds=8)
    assert ev.effective_limits(settings) == {"time_budget_sec": 150.0, "max_rounds": 4, "mode": "shallow"}


def test_effective_limits_fall_back_when_chat_limits_fails_or_is_unreadable(monkeypatch):
    settings = types.SimpleNamespace(chat_time_budget_sec=90.0, chat_max_rounds=8)

    def broken(_settings):
        raise TypeError("needs another argument")

    for fn in (broken, lambda s: (4, 150), lambda s: None):
        monkeypatch.setattr(ev, "_chat_limits_fn", lambda fn=fn: fn)
        assert ev.effective_limits(settings) == {"time_budget_sec": 90.0, "max_rounds": 8, "mode": None}


def test_chat_limits_is_found_in_either_module(monkeypatch):
    def marker(settings):
        return {}

    fake = types.ModuleType("finagent.chat.limits")
    fake.chat_limits = marker
    monkeypatch.setitem(sys.modules, "finagent.chat.limits", fake)
    assert ev._chat_limits_fn() is marker


def test_time_budget_overrides_every_turn_budget_setting_that_exists():
    assert ev.budget_overrides(Settings, 240.0)["chat_time_budget_sec"] == 240.0
    assert "time_budget_sec" not in ev.budget_overrides(Settings, 240.0)  # the QA agent's own budget

    class Later:
        model_fields = {"chat_time_budget_sec": None, "shallow_time_budget_sec": None, "time_budget_sec": None,
                        "chat_max_rounds": None}

    assert ev.budget_overrides(Later, 300.0) == {"chat_time_budget_sec": 300.0, "shallow_time_budget_sec": 300.0}


def test_live_runs_carry_the_real_limit_settings_into_the_fake_world(tmp_path):
    live = _settings(tmp_path, llm_provider="gemini", gemini_api_key="k" * 20, chat_max_rounds=6)
    live_with_new_fields = types.SimpleNamespace(
        llm_provider="gemini", chat_time_budget_sec=150.0, chat_max_rounds=8, ticket_ttl_sec=300, llm_timeout_sec=40.0,
        gemini_paid=False, shallow_max_rounds=4, shallow_time_budget_sec=150.0, gemini_api_key="k" * 20)
    over = ev.world_overrides(live, None)
    assert over["llm_provider"] == "gemini" and over["chat_max_rounds"] == 6
    assert "gemini_api_key" not in over and "alpaca_api_key_id" not in over  # no secret reaches the fake world
    # a limit setting is copied exactly when this backend's Settings has it
    assert ("shallow_max_rounds" in over) == hasattr(live, "shallow_max_rounds")
    newer = ev.world_overrides(live_with_new_fields, None)
    assert newer["gemini_paid"] is False and newer["shallow_max_rounds"] == 4 and "gemini_api_key" not in newer
    assert ev.world_overrides(None, 240.0) == ev.budget_overrides(Settings, 240.0)
    assert ev.world_overrides(live, 240.0)["chat_time_budget_sec"] == 240.0


class _StubWorld:
    """Just enough of a world for `run_scenario`: a session store and an agent that emits events."""

    def __init__(self, finish: dict, answer: str = "I can't help with options; they are not supported."):
        self.finish, self.answer = finish, answer
        self.sessions = types.SimpleNamespace(create=lambda title, lang: {"id": "S1"})
        self.agent = types.SimpleNamespace(run_turn=self.run_turn)

    async def run_turn(self, session_id, message, emit):
        emit({"type": "start", "messageId": "m"})
        emit({"type": "finish", "usage": {"input": 3, "output": 1}, "elapsed_ms": 5, **self.finish})
        return self.answer


async def test_mode_and_max_rounds_come_from_the_finish_event_when_present():
    labels = {"llm": "live", "provider": "gemini", "model": "m", "broker": "fake", "products": "fake",
              "time_budget_sec": 90.0, "max_rounds": 8, "mode": None}
    scenario = {"id": "s", "lang": "en", "category": "out_of_scope", "message": "Buy SPY calls.",
                "expect": {"ticket": False}}
    counter = ev.CountingLLM(fw.DemoLLM())
    old = await ev.run_scenario(_StubWorld({}), counter, scenario, labels)
    assert (old["time_budget_sec"], old["max_rounds"]) == (90.0, 8) and "mode" not in old
    assert old["metrics"]["usage"] == {"input": 3, "output": 1}  # a real model's usage is left as it is
    new = await ev.run_scenario(_StubWorld({"mode": "shallow", "max_rounds": 4, "time_budget_sec": 150}),
                                counter, scenario, labels)
    assert (new["mode"], new["max_rounds"], new["time_budget_sec"]) == ("shallow", 4, 150.0)
    assert ev.config_key(new) != ev.config_key(old)  # the mode is part of the results key
    assert ev.same_config(new, {**labels, "time_budget_sec": 150.0, "mode": "shallow"}) is True
    assert ev.same_config(new, {**labels, "time_budget_sec": 150.0, "mode": "deep"}) is False
    assert ev.same_config(old, {**labels, "mode": "deep"}) is True  # a line without a mode fits any mode


async def test_a_turn_that_ended_on_its_budget_is_ok_if_the_model_answered_and_not_ok_on_the_fallback():
    from finagent.chat.agent import fallback_text

    labels = {"llm": "live", "provider": "gemini", "model": "m", "broker": "fake", "products": "fake",
              "time_budget_sec": 150.0, "max_rounds": 4, "mode": "shallow"}
    scenario = {"id": "s", "lang": "en", "category": "out_of_scope", "message": "Buy SPY calls.",
                "expect": {"ticket": False}}
    counter = ev.CountingLLM(fw.DemoLLM())
    # nudged, but the model wrote an answer: a result to score, flagged
    nudged = await ev.run_scenario(_StubWorld({"mode": "shallow", "max_rounds": 4, "budget_exhausted": True}),
                                   counter, scenario, labels)
    assert nudged["status"] == "ok" and nudged["budget_exhausted"] is True
    plain = await ev.run_scenario(_StubWorld({"mode": "shallow", "max_rounds": 4, "budget_exhausted": False}),
                                  counter, scenario, labels)
    assert plain["status"] == "ok" and "budget_exhausted" not in plain
    # the agent's own fallback text (no model answer): not ok, in either mode and language
    texts = [fallback_text("en", ["get_quotes"]), fallback_text("ko", [])]
    try:
        texts += [fallback_text("en", ["get_quotes"], "shallow"), fallback_text("ko", [], "shallow")]
    except TypeError:  # the older backend has one fallback only
        pass
    for text in texts:
        row = await ev.run_scenario(_StubWorld({"budget_exhausted": True}, answer=text), counter, scenario, labels)
        assert row["status"] == "budget_exhausted", text
    assert "ended on its budget" in ev._progress(1, 1, nudged)


# --- hygiene ---------------------------------------------------------------------------------------------


def test_the_scripts_never_put_their_directory_ahead_of_the_standard_library():
    code = (
        "import pathlib, sys\n"
        f"sys.path.append({str(ROOT / 'scripts')!r})\n"
        "import eval_scenarios, _fakeworld, _demo_llm, _eval_metrics, _eval_text\n"
        "import trace\n"
        "print(pathlib.Path(trace.__file__).resolve().parent)\n"
        "print(sys.path.index(str(pathlib.Path(_fakeworld.__file__).resolve().parent)))\n"
        "print(len(sys.path))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr
    trace_dir, index, length = done.stdout.split()
    assert pathlib.Path(trace_dir) != ROOT / "scripts"   # scripts/trace.py did not shadow the stdlib module
    assert int(index) >= int(length) - 2                 # the scripts dir and the repo root sit at the end
    # run as a script, Python puts scripts/ first on sys.path: the script moves that entry to the end
    as_script = (
        "import pathlib, runpy, sys\n"
        f"scripts = {str(ROOT / 'scripts')!r}\n"
        "sys.path.insert(0, scripts)\n"
        "runpy.run_path(scripts + '/eval_scenarios.py', run_name='eval_scenarios_not_main')\n"
        "import trace\n"
        "print(pathlib.Path(sys.path[0]).resolve() != pathlib.Path(scripts).resolve())\n"
        "print(pathlib.Path(trace.__file__).resolve().parent != pathlib.Path(scripts).resolve())\n"
    )
    done = subprocess.run([sys.executable, "-c", as_script], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "True"]
    helped = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_scenarios.py"), "--help"],
                            capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert helped.returncode == 0 and "--time-budget" in helped.stdout and "90" not in helped.stdout


def test_docs_do_not_hard_code_a_default_budget_and_warn_about_live_broker_lines():
    assert "CHAT_TIME_BUDGET_SEC, 90" not in ev.__doc__ and " 90" not in ev.__doc__
    assert "real paper account" in ev.__doc__ and "(id, llm, model, broker, products" in ev.__doc__


async def test_read_account_gives_equity_shares_and_market_values():
    account = await ev.read_account(types.SimpleNamespace(broker=fw.FakeBroker()))
    assert account == {"equity": "100000", "positions": {"MSFT": "10"}, "values": {"MSFT": "4000"}}
