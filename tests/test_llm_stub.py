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
