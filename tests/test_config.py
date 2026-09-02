from finagent.config import Settings
from finagent.models import AnswerResponse


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-opus-5"
    assert s.time_budget_sec == 50
    assert s.store_path.name == "products.db"


def test_answer_response_fields():
    r = AnswerResponse(question_id="Q-1", question="q", retrieved_context="c", think_trace="t", answer="a")
    assert set(r.model_dump().keys()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
