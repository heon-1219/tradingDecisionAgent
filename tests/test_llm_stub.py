import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from finagent.agent.llm import (
    AnthropicConversation,
    LLMHandle,
    StubConversation,
    StubLLM,
    ToolCall,
    ToolSpec,
    Turn,
    make_llm,
)
from finagent.config import Settings

GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT",
                   "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "LLM_MAX_TOKENS")


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


# --------------------------------------------------------------------------- LLMHandle (Task 3.8)
def _named_stub(model: str, *turns: Turn) -> StubLLM:
    llm = StubLLM(list(turns))
    llm.model = model
    return llm


def test_llm_handle_forwards_name_model_and_start():
    inner = _named_stub("model-a", Turn("from a"))
    handle = LLMHandle(inner)
    assert handle.current is inner and handle.name == "stub" and handle.model == "model-a"
    conv = handle.start("sys", [], "q")
    assert inner.started == [("sys", "q")] and isinstance(conv, StubConversation)
    assert handle.started is inner.started  # anything else is forwarded too
    assert LLMHandle(StubLLM([])).model is None  # a client without a model name
    with pytest.raises(AttributeError):
        handle.no_such_attribute


async def test_llm_handle_swap_changes_the_client_for_new_conversations_only():
    a, b = _named_stub("model-a", Turn("from a")), _named_stub("model-b", Turn("from b"))
    handle = LLMHandle(a)
    before = handle.generation
    in_flight = handle.start("sys", [], "q1")  # a turn already running holds this conversation
    assert handle.swap(b) is a  # the old client is handed back, to be retired by the caller
    assert handle.current is b and handle.model == "model-b" and handle.generation != before
    assert (await in_flight.step()).text == "from a"  # it finishes on the client it started with
    assert (await handle.start("sys", [], "q2").step()).text == "from b"
    assert len(a.started) == 1 and len(b.started) == 1


def test_llm_handle_wraps_once_and_never_nests():
    inner = StubLLM([])
    handle = LLMHandle.wrap(inner)
    assert isinstance(handle, LLMHandle) and LLMHandle.wrap(handle) is handle
    with pytest.raises(TypeError):
        LLMHandle(handle)
    with pytest.raises(TypeError):
        handle.swap(LLMHandle(StubLLM([])))


async def test_every_holder_of_the_handle_sees_the_swap():
    """The QA orchestrator, the chat runtime and the chat agent hold the same handle object."""
    handle = LLMHandle(_named_stub("model-a", Turn("from a")))
    holders = [SimpleNamespace(llm=handle) for _ in range(3)]
    handle.swap(_named_stub("model-b", Turn("from b")))
    for holder in holders:
        assert holder.llm.model == "model-b"
        assert (await holder.llm.start("sys", [], "q").step()).text == "from b"


# --------------------------------------------------------------------------- closing and the output cap
async def test_provider_clients_close_their_sdk_client():
    for llm in (
        make_llm(Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k")),
        make_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key="k")),
        make_llm(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k", llm_model="claude-x")),
    ):
        assert not llm.client.is_closed()
        await llm.aclose()
        assert llm.client.is_closed()
    assert not hasattr(StubLLM([]), "aclose")  # nothing to close


def _openai_with(fake: FakeGemini, **overrides):
    llm = make_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key="k", **overrides))
    llm.client = openai.AsyncOpenAI(
        api_key="k", base_url=overrides.get("openai_base_url") or "https://api.openai.com/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    return llm


async def test_openai_sends_the_output_cap_as_max_completion_tokens():
    """OpenAI's reasoning models reject `max_tokens`; its API takes `max_completion_tokens` for every model."""
    fake = FakeGemini((200, _completion({"content": "ok"})))
    await _openai_with(fake, llm_max_tokens=321).start("sys", TOOLS, "q").step()
    assert fake.bodies[0]["max_completion_tokens"] == 321 and "max_tokens" not in fake.bodies[0]
    assert "reasoning_effort" not in fake.bodies[0]


async def test_gemini_and_custom_openai_servers_keep_max_tokens():
    fake = FakeGemini((200, _completion({"content": "ok"})))
    await _gemini_with(fake, llm_max_tokens=321).start("sys", TOOLS, "q").step()
    assert fake.bodies[0]["max_tokens"] == 321 and "max_completion_tokens" not in fake.bodies[0]
    fake = FakeGemini((200, _completion({"content": "ok"})))
    await _openai_with(fake, llm_max_tokens=321, openai_base_url="http://localhost:1/v1").start("sys", TOOLS, "q").step()
    assert fake.bodies[0]["max_tokens"] == 321 and "max_completion_tokens" not in fake.bodies[0]


@pytest.mark.parametrize("base_url, first, second, message", [
    (None, "max_completion_tokens", "max_tokens",
     "Unrecognized request argument supplied: max_completion_tokens"),
    ("http://localhost:1/v1", "max_tokens", "max_completion_tokens",
     "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."),
])
async def test_a_rejected_output_cap_parameter_is_swapped_once_for_the_whole_client(base_url, first, second, message):
    rejected = {"error": {"message": message, "type": "invalid_request_error", "param": first, "code": None}}
    fake = FakeGemini((400, rejected), (200, _completion({"content": "ok"})), (200, _completion({"content": "again"})))
    overrides = {"openai_base_url": base_url} if base_url else {}
    llm = _openai_with(fake, **overrides)
    conv = llm.start("sys", TOOLS, "q")
    assert (await conv.step()).text == "ok"
    assert first in fake.bodies[0] and second not in fake.bodies[0]
    assert second in fake.bodies[1] and first not in fake.bodies[1]
    assert (await llm.start("sys", TOOLS, "q2").step()).text == "again"  # remembered by the client
    assert second in fake.bodies[2] and first not in fake.bodies[2]


async def test_an_output_cap_that_is_only_too_large_is_not_swapped():
    too_large = {"error": {"message": "max_completion_tokens is too large: 9000. This model supports at most 8192.",
                           "type": "invalid_request_error"}}
    fake = FakeGemini((400, too_large))
    turn = await _openai_with(fake).start("sys", TOOLS, "q").step()
    assert turn.stop_reason == "error" and len(fake.bodies) == 1


# --- final review C-m5: provider errors are logged by type and status only ----------------------

PROVIDER_TEXT = "PROVIDER-BODY-MARKER account acct_12345 prompt excerpt"


def _ours(caplog) -> str:
    """What this app logged (the SDKs' own DEBUG records are theirs)."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("finagent"))


async def test_an_openai_compatible_error_is_logged_without_the_providers_text(caplog):
    body = {"error": {"code": 400, "message": f"bad request: {PROVIDER_TEXT}", "status": "INVALID_ARGUMENT"}}
    fake = FakeGemini((400, body))
    with caplog.at_level("DEBUG"):
        turn = await _gemini_with(fake).start("sys", TOOLS, "q").step()
    assert turn.stop_reason == "error"
    assert "BadRequestError" in _ours(caplog) and "400" in _ours(caplog)
    assert "PROVIDER-BODY-MARKER" not in _ours(caplog)


async def test_a_rejected_reasoning_effort_is_logged_without_the_providers_text(caplog):
    rejected = {"error": {"code": 400, "message": f"reasoning_effort is not supported: {PROVIDER_TEXT}",
                          "status": "INVALID_ARGUMENT"}}
    fake = FakeGemini((400, rejected), (200, _completion({"content": "ok"})))
    with caplog.at_level("DEBUG"):
        assert (await _gemini_with(fake).start("sys", TOOLS, "q").step()).text == "ok"
    assert "rejected reasoning_effort" in _ours(caplog)
    assert "PROVIDER-BODY-MARKER" not in _ours(caplog)


def _anthropic_conversation(*responses: tuple[int, dict], fallbacks: bool = False):
    import anthropic

    import httpx2  # the installed anthropic SDK is built on httpx2, not httpx

    queue = list(responses)

    def handler(request):
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body)

    client = anthropic.AsyncAnthropic(api_key="k", max_retries=0,
                                      http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return AnthropicConversation(client, "claude-test", "low", 100, fallbacks, "sys", [], "q")


_ANTHROPIC_OK = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
                 "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                 "usage": {"input_tokens": 1, "output_tokens": 1}}


async def test_an_anthropic_error_is_logged_without_the_providers_text(caplog):
    conv = _anthropic_conversation((500, {"type": "error", "error": {"type": "api_error", "message": PROVIDER_TEXT}}))
    with caplog.at_level("DEBUG"):
        turn = await conv.step()
    assert turn.stop_reason == "error"
    assert "InternalServerError" in _ours(caplog) and "500" in _ours(caplog)
    assert "PROVIDER-BODY-MARKER" not in _ours(caplog)


@pytest.mark.parametrize("fallbacks, message", [
    (True, f"fallbacks beta is not available: {PROVIDER_TEXT}"),
    (False, f"effort is not supported for this model: {PROVIDER_TEXT}"),
])
async def test_an_anthropic_retry_warning_is_logged_without_the_providers_text(caplog, fallbacks, message):
    conv = _anthropic_conversation(
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": message}}),
        (200, _ANTHROPIC_OK), fallbacks=fallbacks)
    with caplog.at_level("DEBUG"):
        assert (await conv.step()).text == "ok"
    assert "retrying" in _ours(caplog)
    assert "PROVIDER-BODY-MARKER" not in _ours(caplog)
