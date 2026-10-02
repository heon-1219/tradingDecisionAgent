from finagent.config import Settings, env_fallbacks
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
    for name in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT", "TRADING_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    assert env_fallbacks(env) == []
    monkeypatch.setenv("TRADING_ENABLED", "true")   # in the file: the file wins, so not a fallback
    monkeypatch.setenv("TAVILY_API_KEY", "tv-shell")  # not in the file
    monkeypatch.setenv("SEC_USER_AGENT", "Jane jane@example.com")  # empty in the file = not set
    assert env_fallbacks(env) == ["SEC_USER_AGENT", "TAVILY_API_KEY"]
    assert env_fallbacks(tmp_path / "missing.env") == ["SEC_USER_AGENT", "TAVILY_API_KEY", "TRADING_ENABLED"]
