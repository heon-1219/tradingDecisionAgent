import json

import httpx
import openai
import pytest

from finagent.agent.llm import AnthropicConversation, StubLLM, ToolCall, ToolSpec, Turn, make_llm
from finagent.config import Settings

GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT")


@pytest.fixture(autouse=True)
def _no_ambient_llm_env(monkeypatch):
    # A developer shell may export a real (or stale) key; these tests must not depend on it.
    for var in GEMINI_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


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


def test_stub_add_assistant_appends_a_plain_assistant_message():
    """Used only when rebuilding a conversation from stored text history: no tool calls, so no
    thought signatures are needed."""
    llm = StubLLM([Turn("done", [], "end_turn")])
    conv = llm.start("sys", [], "q")
    conv.add_assistant("earlier reply")
    assert conv.transcript()[-1] == {"role": "assistant", "content": "earlier reply"}


def test_anthropic_add_assistant_appends_a_plain_assistant_message():
    conv = AnthropicConversation(None, "model", "low", 100, False, "sys", [], "q")
    conv.add_assistant("earlier reply")
    assert conv.transcript()[-1] == {"role": "assistant", "content": "earlier reply"}


def test_make_llm_by_provider():
    assert make_llm(Settings(_env_file=None, llm_provider="stub")).name == "stub"
    # The English edition runs on Google Gemini by default
    default = Settings(_env_file=None)
    assert default.llm_provider == "gemini" and default.llm_model == "gemini-3.8-flash"
    g = make_llm(Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k"))
    assert g.name == "gemini" and "generativelanguage.googleapis.com" in str(g.client.base_url)
    a = make_llm(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-test", llm_model="claude-opus-5"))
    assert a.name == "anthropic" and a.model == "claude-opus-5"
    o = make_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key="x", openai_base_url="http://localhost:1/v1"))
    assert o.name == "openai"


def test_gemini_key_is_read_from_either_google_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    assert Settings(_env_file=None).gemini_api_key == "from-google"
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")  # GEMINI_API_KEY wins when both are set
    assert Settings(_env_file=None).gemini_api_key == "from-gemini"


# --------------------------------------------------------------------------- Gemini over HTTP (mocked)
def _completion(message: dict, finish: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gemini-3.8-flash",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class FakeGemini:
    """Stands in for generativelanguage.googleapis.com: replays canned responses, records request bodies."""

    def __init__(self, *responses: tuple[int, dict]):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)


def _gemini_with(fake: FakeGemini, **overrides):
    llm = make_llm(Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k", **overrides))
    llm.client = openai.AsyncOpenAI(
        api_key="k", base_url=llm.client.base_url, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    return llm


TOOLS = [ToolSpec("sql_query", "Run SQL", {"type": "object", "properties": {"sql": {"type": "string"}}})]
SIGNED_CALL = {
    "id": "call_1", "type": "function",
    "function": {"name": "sql_query", "arguments": json.dumps({"sql": "select 1"})},
    "extra_content": {"google": {"thought_signature": "sig-1"}},
}


async def test_gemini_echoes_thought_signatures_back():
    """Gemini 3 rejects a follow-up request whose function calls lost their thought signature.

    Over the OpenAI-compatible endpoint the signature rides on each tool call as
    ``extra_content.google.thought_signature``; the assistant message we send back must keep it.
    """
    fake = FakeGemini((200, _completion({"content": None, "tool_calls": [SIGNED_CALL]}, "tool_calls")),
                      (200, _completion({"content": "done"})))
    conv = _gemini_with(fake).start("sys", TOOLS, "How many bonds?")
    t1 = await conv.step()
    assert t1.stop_reason == "tool_use" and t1.tool_calls[0].input == {"sql": "select 1"}
    conv.add_tool_results([("call_1", "cnt=3", False)])
    t2 = await conv.step()
    assert t2.text == "done" and t2.stop_reason == "end_turn"
    echoed = fake.bodies[1]["messages"][2]
    assert echoed["role"] == "assistant"
    assert echoed["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig-1"}}
    assert fake.bodies[1]["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "cnt=3"}


async def test_add_assistant_message_shape_is_sent_in_order_over_the_openai_protocol():
    """GeminiConversation subclasses OpenAIConversation and doesn't override add_assistant, so this
    one MockTransport check covers the OpenAI-compatible message shape used by both OpenAI and Gemini:
    a plain {"role": "assistant", "content": text} message, with no tool_calls key, sent in the
    position it was added (after the existing system/user messages, before the request is issued)."""
    fake = FakeGemini((200, _completion({"content": "ok"})))
    conv = _gemini_with(fake).start("sys", TOOLS, "q")
    conv.add_assistant("previously said this")
    await conv.step()
    sent = fake.bodies[0]["messages"]
    assert sent == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "previously said this"},
    ]


@pytest.mark.parametrize("effort,sent", [("low", "low"), ("medium", "medium"), ("high", "high"),
                                         ("xhigh", "high"), ("max", "high")])
async def test_gemini_maps_effort_to_reasoning_effort(effort, sent):
    fake = FakeGemini((200, _completion({"content": "ok"})))
    await _gemini_with(fake, llm_effort=effort).start("sys", TOOLS, "q").step()
    assert fake.bodies[0]["reasoning_effort"] == sent


async def test_gemini_drops_reasoning_effort_if_the_model_rejects_it():
    rejected = {"error": {"code": 400, "message": "reasoning_effort is not supported for this model",
                          "status": "INVALID_ARGUMENT"}}
    fake = FakeGemini((400, rejected), (200, _completion({"content": "ok"})), (200, _completion({"content": "again"})))
    conv = _gemini_with(fake).start("sys", TOOLS, "q")
    assert (await conv.step()).text == "ok"
    assert "reasoning_effort" in fake.bodies[0] and "reasoning_effort" not in fake.bodies[1]
    conv.add_user("next")
    assert (await conv.step()).text == "again"
    assert "reasoning_effort" not in fake.bodies[2]  # remembered for the rest of the conversation


async def test_gemini_without_a_key_fails_fast_with_a_clear_message():
    llm = make_llm(Settings(_env_file=None, llm_provider="gemini"))
    turn = await llm.start("sys", TOOLS, "q").step()
    assert turn.stop_reason == "error" and "GEMINI_API_KEY" in turn.text


# --------------------------------------------------------------------------- retries
async def test_rate_limit_is_retried_not_surfaced_as_error():
    """A 429 is an infrastructure error, not a model failure; giving up at once loses the whole question."""
    from finagent.agent.llm import call_with_rate_limit_retry

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
    assert calls["n"] == 3 and slept == [1.0, 1.0]  # honours Retry-After


async def test_rate_limit_gives_up_after_retries():
    from finagent.agent.llm import call_with_rate_limit_retry

    async def always():
        raise openai.RateLimitError(
            "rate exceeded", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None
        )

    async def fake_sleep(s):
        pass

    with pytest.raises(openai.RateLimitError):  # once retries are exhausted the error must surface
        await call_with_rate_limit_retry(always, retries=2, sleep=fake_sleep)


async def test_overloaded_server_is_retried():
    """Gemini answers 503 "model is overloaded" under load; that is transient, like a 429."""
    from finagent.agent.llm import call_with_rate_limit_retry

    calls = {"n": 0}
    slept: list[float] = []

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai.InternalServerError(
                "overloaded", response=httpx.Response(503, request=httpx.Request("POST", "http://x")), body=None
            )
        return "ok"

    async def fake_sleep(s):
        slept.append(s)

    assert await call_with_rate_limit_retry(flaky, retries=4, sleep=fake_sleep) == "ok"
    assert calls["n"] == 3 and slept == [2.0, 4.0]


async def test_overloaded_server_gives_up_after_server_error_retries():
    from finagent.agent.llm import call_with_rate_limit_retry

    calls = {"n": 0}

    async def always():
        calls["n"] += 1
        raise openai.InternalServerError(
            "overloaded", response=httpx.Response(503, request=httpx.Request("POST", "http://x")), body=None
        )

    async def fake_sleep(s):
        pass

    with pytest.raises(openai.InternalServerError):
        await call_with_rate_limit_retry(always, retries=4, sleep=fake_sleep, server_error_retries=2)
    assert calls["n"] == 3  # the first try plus two retries


def test_retry_delay_backoff_and_cap():
    from finagent.agent.llm import retry_delay

    assert retry_delay(0) == 2.0 and retry_delay(1) == 4.0 and retry_delay(2) == 8.0
    assert retry_delay(10) == 30.0            # capped
    assert retry_delay(0, retry_after="7") == 7.0   # the server's Retry-After wins
    assert retry_delay(0, retry_after="bad") == 2.0  # unparseable -> fall back to backoff
