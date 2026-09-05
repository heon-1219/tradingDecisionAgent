from finagent.agent.llm import StubLLM, ToolCall, ToolSpec, Turn, make_llm
from finagent.config import Settings


async def test_stub_conversation_replays_script():
    llm = StubLLM([Turn("", [ToolCall("c1", "sql_query", {"sql": "select 1"})], "tool_use"), Turn("done", [], "end_turn")])
    conv = llm.start("sys", [ToolSpec("sql_query", "d", {"type": "object"})], "q")
    t1 = await conv.step()
    assert t1.tool_calls[0].name == "sql_query" and t1.stop_reason == "tool_use"
    conv.add_tool_results([("c1", "ok", False)])
    t2 = await conv.step()
    assert t2.stop_reason == "end_turn" and t2.text == "done"
    assert conv.transcript()[-1]["role"] == "assistant"
    t3 = await conv.step()  # script exhausted -> empty end_turn
    assert t3.stop_reason == "end_turn"


def test_make_llm_by_provider():
    assert make_llm(Settings(_env_file=None, llm_provider="stub")).name == "stub"
    # 대회 규정상 기본값은 HyperCLOVA X 여야 한다
    default = Settings(_env_file=None)
    assert default.llm_provider == "clova" and default.llm_model == "HCX-005"
    c = make_llm(Settings(_env_file=None, llm_provider="clova", clova_api_key="k"))
    assert c.name == "clova" and "clovastudio" in str(c.client.base_url)
    # function calling 은 max_tokens >= 1024 에서만 동작하므로 낮은 값은 올려준다
    low = make_llm(Settings(_env_file=None, llm_provider="clova", clova_api_key="k", llm_max_tokens=256))
    assert low.max_tokens == 1024
    a = make_llm(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-test", llm_model="claude-opus-5"))
    assert a.name == "anthropic" and a.model == "claude-opus-5"
    o = make_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key="x", openai_base_url="http://localhost:1/v1"))
    assert o.name == "openai"


async def test_rate_limit_is_retried_not_surfaced_as_error():
    """429 는 모델 실력과 무관한 인프라 오류다. 즉시 포기하면 그 문항을 통째로 잃는다."""
    import httpx
    import openai

    from finagent.agent.llm import call_with_rate_limit_retry, retry_delay

    calls = {"n": 0}
    slept: list[float] = []

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai.RateLimitError(
                "rate exceeded",
                response=httpx.Response(429, request=httpx.Request("POST", "http://x"), headers={"retry-after": "1"}),
                body=None,
            )
        return "ok"

    async def fake_sleep(s):
        slept.append(s)

    assert await call_with_rate_limit_retry(flaky, retries=4, sleep=fake_sleep) == "ok"
    assert calls["n"] == 3 and slept == [1.0, 1.0]  # Retry-After 를 존중


async def test_rate_limit_gives_up_after_retries():
    import httpx
    import openai

    from finagent.agent.llm import call_with_rate_limit_retry

    async def always():
        raise openai.RateLimitError(
            "rate exceeded", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None
        )

    async def fake_sleep(s):
        pass

    try:
        await call_with_rate_limit_retry(always, retries=2, sleep=fake_sleep)
    except openai.RateLimitError:
        pass
    else:
        raise AssertionError("재시도 소진 후에는 예외가 올라와야 한다")


def test_retry_delay_backoff_and_cap():
    from finagent.agent.llm import retry_delay

    assert retry_delay(0) == 2.0 and retry_delay(1) == 4.0 and retry_delay(2) == 8.0
    assert retry_delay(10) == 30.0            # 상한
    assert retry_delay(0, retry_after="7") == 7.0   # 서버 지시 우선
    assert retry_delay(0, retry_after="bad") == 2.0  # 파싱 실패 시 백오프로
