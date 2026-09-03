"""Provider-agnostic chat-with-tools adapters.

`LLMClient.start()` returns a `Conversation` that owns the provider-specific message history
(so Anthropic thinking blocks are echoed back untouched), exposes `step()` to get the next
assistant turn, and `add_tool_results()` to feed tool outputs back in ONE message.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("finagent.llm")


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # tool_use | end_turn | max_tokens | refusal | error
    usage: dict = field(default_factory=dict)


class Conversation(Protocol):
    async def step(self) -> Turn: ...
    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None: ...
    def add_user(self, text: str) -> None: ...
    def transcript(self) -> list[dict]: ...


class LLMClient(Protocol):
    name: str

    def start(self, system: str, tools: list[ToolSpec], user_text: str) -> Conversation: ...


# --------------------------------------------------------------------------- stub
class StubConversation:
    def __init__(self, script: list[Turn], system: str, user_text: str):
        self._script = list(script)
        self._messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]

    async def step(self) -> Turn:
        turn = self._script.pop(0) if self._script else Turn("", [], "end_turn")
        self._messages.append({"role": "assistant", "content": turn.text, "tool_calls": [c.__dict__ for c in turn.tool_calls]})
        return turn

    def add_tool_results(self, results):
        self._messages.append({"role": "user", "content": [{"tool_use_id": cid, "content": c, "is_error": e} for cid, c, e in results]})

    def add_user(self, text: str):
        self._messages.append({"role": "user", "content": text})

    def transcript(self):
        return self._messages


class StubLLM:
    """Deterministic scripted LLM for tests (`script` is consumed in order per conversation)."""

    name = "stub"

    def __init__(self, script: list[Turn] | None = None, script_factory=None):
        self._script, self._factory = script or [], script_factory
        self.started: list[tuple[str, str]] = []

    def start(self, system, tools, user_text):
        self.started.append((system, user_text))
        script = self._factory(user_text) if self._factory else self._script
        return StubConversation(script, system, user_text)


# --------------------------------------------------------------------------- anthropic
class AnthropicConversation:
    def __init__(self, client, model: str, effort: str, max_tokens: int, fallbacks: bool, system: str, tools: list[ToolSpec], user_text: str):
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.fallbacks = fallbacks
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        self.messages: list[dict] = [{"role": "user", "content": user_text}]

    async def step(self) -> Turn:
        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        try:
            try:
                if self.fallbacks:
                    resp = await self.client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                    )
                else:
                    resp = await self.client.messages.create(**kwargs)
            except anthropic.BadRequestError as e:
                msg = str(e).lower()
                if self.fallbacks and ("fallback" in msg or "beta" in msg):
                    log.warning("fallbacks unsupported here, retrying without: %s", e)
                    self.fallbacks = False
                    resp = await self.client.messages.create(**kwargs)
                elif "effort" in msg or "thinking" in msg or "output_config" in msg:
                    log.warning("effort/thinking unsupported for %s, retrying plain: %s", self.model, e)
                    kwargs.pop("thinking", None)
                    kwargs.pop("output_config", None)
                    resp = await self.client.messages.create(**kwargs)
                else:
                    raise
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            log.error("anthropic error: %s", e)
            return Turn(f"LLM 호출 오류: {type(e).__name__}: {e}", [], "error")

        self.messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            return Turn("", [], "refusal")
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in resp.content if getattr(b, "type", "") == "tool_use"]
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                     "cache_read": getattr(resp.usage, "cache_read_input_tokens", None)}
        return Turn(text, calls, "tool_use" if calls else (resp.stop_reason or "end_turn"), usage)

    def add_tool_results(self, results):
        self.messages.append(
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": cid, "content": content, "is_error": bool(err)} for cid, content, err in results
            ]}
        )

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def transcript(self):
        return self.messages


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, settings):
        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None, timeout=settings.llm_timeout_sec, max_retries=1
        )
        self.model, self.effort, self.max_tokens = settings.llm_model, settings.llm_effort, settings.llm_max_tokens
        self.fallbacks = settings.anthropic_fallbacks

    def start(self, system, tools, user_text):
        return AnthropicConversation(self.client, self.model, self.effort, self.max_tokens, self.fallbacks, system, tools, user_text)


# --------------------------------------------------------------------------- openai-compatible
class OpenAIConversation:
    def __init__(self, client, model: str, max_tokens: int, system: str, tools: list[ToolSpec], user_text: str):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]

    async def step(self) -> Turn:
        import openai

        try:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools or None, max_tokens=self.max_tokens
            )
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
            log.error("openai error: %s", e)
            return Turn(f"LLM 호출 오류: {type(e).__name__}: {e}", [], "error")
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        self.messages.append(
            {"role": "assistant", "content": msg.content or "",
             **({"tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]} if msg.tool_calls else {})}
        )
        fr = choice.finish_reason
        stop = "tool_use" if calls else ("max_tokens" if fr == "length" else "end_turn")
        usage = {"input": getattr(resp.usage, "prompt_tokens", None), "output": getattr(resp.usage, "completion_tokens", None)} if resp.usage else {}
        return Turn(msg.content or "", calls, stop, usage)

    def add_tool_results(self, results):
        for cid, content, err in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": ("[ERROR] " if err else "") + content})

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def transcript(self):
        return self.messages


class OpenAILLM:
    name = "openai"

    def __init__(self, settings):
        import openai

        self.client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key or None, base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_sec, max_retries=1,
        )
        self.model, self.max_tokens = settings.llm_model, settings.llm_max_tokens

    def start(self, system, tools, user_text):
        return OpenAIConversation(self.client, self.model, self.max_tokens, system, tools, user_text)


class ClovaLLM(OpenAILLM):
    """NCP HyperCLOVA X via CLOVA Studio's OpenAI-compatible endpoint.

    The competition mandates HyperCLOVA X for the answer pipeline, so this is the default provider.
    CLOVA Studio exposes `https://clovastudio.stream.ntruss.com/v1/openai`, which accepts the OpenAI
    SDK plus `tools` / `tool_choice` / `parallel_tool_calls`, so the OpenAI conversation shape is
    reused verbatim.  Two CLOVA-specific rules are enforced here:

    * function calling only works when `max_tokens` >= 1024 — a smaller value silently returns no
      tool calls, which would look like the model refusing to use its tools;
    * function calling cannot be combined with the thinking/structured-output modes, so nothing else
      is switched on.
    """

    name = "clova"
    MIN_TOOL_TOKENS = 1024

    def __init__(self, settings):
        import openai

        key = settings.clova_api_key or settings.openai_api_key
        self.client = openai.AsyncOpenAI(
            api_key=key or None,
            base_url=settings.clova_base_url,
            timeout=settings.llm_timeout_sec,
            max_retries=1,
        )
        self.model = settings.llm_model
        self.max_tokens = max(int(settings.llm_max_tokens), self.MIN_TOOL_TOKENS)
        if settings.llm_max_tokens < self.MIN_TOOL_TOKENS:
            log.warning(
                "CLOVA function calling needs max_tokens >= %d; raising %d -> %d",
                self.MIN_TOOL_TOKENS, settings.llm_max_tokens, self.max_tokens,
            )


def make_llm(settings) -> LLMClient:
    if settings.llm_provider == "clova":
        return ClovaLLM(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    if settings.llm_provider == "openai":
        return OpenAILLM(settings)
    return StubLLM([])
