from types import SimpleNamespace

import pytest

from finagent.config import ENV_NAMES, ChatLimits, Settings, chat_limits, chat_mode, env_fallbacks
from finagent.models import AnswerResponse


def test_settings_defaults(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    # English edition: Google Gemini, through its OpenAI-compatible endpoint, is the default provider
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-3.8-flash"
    assert s.gemini_base_url.rstrip("/").endswith("/v1beta/openai")
    # Kept from the competition contract: anything over 300 s counts as no answer
    assert s.time_budget_sec == 120 and s.time_budget_sec < 300
    assert s.max_tool_rounds == 8
    assert s.store_path.name == "products.db"


def test_answer_response_fields():
    r = AnswerResponse(question_id="Q-1", question="q", retrieved_context="c", think_trace="t", answer="a")
    assert set(r.model_dump().keys()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}


def test_env_file_wins_over_the_process_environment(tmp_path, monkeypatch):
    # A value saved in .env (by hand or from the app) must not be undone by a stale shell export.
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\nTRADING_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    monkeypatch.setenv("TRADING_ENABLED", "true")
    s = Settings(_env_file=env)
    assert s.gemini_api_key == "from-file"
    assert s.trading_enabled is False


def test_process_environment_fills_what_the_env_file_leaves_out(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=\nLLM_MODEL=gemini-x\n", encoding="utf-8")  # an empty value is "not set"
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    monkeypatch.setenv("TAVILY_API_KEY", "tv-shell")
    s = Settings(_env_file=env)
    assert s.gemini_api_key == "from-shell"
    assert s.tavily_api_key == "tv-shell"
    assert s.llm_model == "gemini-x"


def test_an_empty_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv("TRADING_ENABLED", "")
    assert Settings(_env_file=None).trading_enabled is True


def test_explicit_arguments_still_win(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    assert Settings(_env_file=env, gemini_api_key="explicit").gemini_api_key == "explicit"


def test_env_fallbacks_lists_app_settings_that_only_the_environment_provides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TRADING_ENABLED=false\nSEC_USER_AGENT=\n", encoding="utf-8")
    for name in ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    assert env_fallbacks(env) == []
    monkeypatch.setenv("TRADING_ENABLED", "true")   # in the file: the file wins, so not a fallback
    monkeypatch.setenv("TAVILY_API_KEY", "tv-shell")  # not in the file
    monkeypatch.setenv("SEC_USER_AGENT", "Jane jane@example.com")  # empty in the file = not set
    assert env_fallbacks(env) == ["SEC_USER_AGENT", "TAVILY_API_KEY"]
    assert env_fallbacks(tmp_path / "missing.env") == ["SEC_USER_AGENT", "TAVILY_API_KEY", "TRADING_ENABLED"]


# --- Task 3.8: research depth (shallow / deep) and the app-writable LLM settings ----------------------


def _llm_settings(**over) -> Settings:
    return Settings(_env_file=None, **over)


def test_chat_limit_defaults():
    s = _llm_settings()
    assert (s.chat_max_rounds, s.chat_time_budget_sec) == (12, 300.0)
    assert (s.shallow_max_rounds, s.shallow_time_budget_sec) == (4, 150.0)
    assert s.gemini_paid is False


@pytest.mark.parametrize("provider, paid, mode", [
    ("gemini", False, "shallow"),   # a free-tier Gemini key: a few model calls a minute
    ("gemini", True, "deep"),       # the user says the Gemini key is on a paid plan
    ("openai", False, "deep"),      # OpenAI and Anthropic keys are always paid
    ("openai", True, "deep"),
    ("anthropic", False, "deep"),
    ("anthropic", True, "deep"),
    ("stub", False, "deep"),        # tests keep the deep limits unless they opt in
    ("stub", True, "deep"),
])
def test_chat_mode_truth_table(provider, paid, mode):
    s = _llm_settings(llm_provider=provider, gemini_paid=paid)
    assert chat_mode(s) == mode
    limits = chat_limits(s)
    assert limits == (ChatLimits("shallow", 4, 150.0) if mode == "shallow" else ChatLimits("deep", 12, 300.0))


def test_chat_limits_follow_the_four_settings():
    shallow = _llm_settings(llm_provider="gemini", shallow_max_rounds=3, shallow_time_budget_sec=77,
                            chat_max_rounds=20, chat_time_budget_sec=500)
    assert chat_limits(shallow) == ChatLimits("shallow", 3, 77.0)
    deep = shallow.model_copy(update={"gemini_paid": True})
    assert chat_limits(deep) == ChatLimits("deep", 20, 500.0)


def test_chat_limits_tolerate_a_bare_or_broken_settings_object():
    assert chat_limits(SimpleNamespace()) == ChatLimits("deep", 12, 300.0)
    assert chat_limits(SimpleNamespace(llm_provider="gemini")) == ChatLimits("shallow", 4, 150.0)
    odd = SimpleNamespace(chat_max_rounds="many", chat_time_budget_sec=None)
    assert chat_limits(odd) == ChatLimits("deep", 12, 300.0)
    assert chat_limits(SimpleNamespace(chat_max_rounds=0, chat_time_budget_sec=5)).max_rounds == 1


def test_the_app_writable_llm_settings_have_env_names():
    assert {ENV_NAMES[name] for name in (
        "llm_provider", "llm_model", "llm_effort", "gemini_api_key", "openai_api_key", "anthropic_api_key",
        "gemini_paid",
    )} == {"LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
           "GEMINI_PAID"}


def test_gemini_paid_is_read_from_the_env_file_and_reported_as_a_fallback(tmp_path, monkeypatch):
    for name in ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_PAID=true\nLLM_PROVIDER=gemini\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.gemini_paid is True and chat_mode(s) == "deep"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-shell")
    monkeypatch.setenv("GEMINI_PAID", "false")  # in the file: the file wins, so not a fallback
    assert env_fallbacks(env) == ["OPENAI_API_KEY"]
