from finagent.config import Settings
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
