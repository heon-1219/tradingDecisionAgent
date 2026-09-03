from finagent.config import Settings
from finagent.models import AnswerResponse


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings(_env_file=None)
    # 대회 규정: 답변 파이프라인은 NCP HyperCLOVA X 필수 → 기본값이 clova 여야 한다
    assert s.llm_provider == "clova"
    assert s.llm_model == "HCX-005"
    assert s.clova_base_url.endswith("/v1/openai")
    # 주최측 공지: 300초 초과는 미응답 처리
    assert s.time_budget_sec == 120 and s.time_budget_sec < 300
    assert s.max_tool_rounds == 8
    assert s.store_path.name == "products.db"


def test_answer_response_fields():
    r = AnswerResponse(question_id="Q-1", question="q", retrieved_context="c", think_trace="t", answer="a")
    assert set(r.model_dump().keys()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
