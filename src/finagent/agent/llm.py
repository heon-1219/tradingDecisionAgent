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


# --------------------------------------------------------------------------- rate limits
def retry_delay(attempt: int, retry_after: str | float | None = None, base: float = 2.0, cap: float = 30.0) -> float:
    """429 재시도 대기 시간. 서버가 Retry-After 를 주면 그 값을 우선한다."""
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), cap))
        except (TypeError, ValueError):
            pass
    return min(base * (2 ** attempt), cap)


_last_call_at = {"t": 0.0}
# 마지막 응답의 x-ratelimit-* 헤더. CLOVA 는 QPM(요청)과 TPM(토큰)을 따로 재고 둘 중 하나만
# 소진돼도 429 를 낸다. 헤더를 읽어두면 한도에 닿기 전에 스스로 멈출 수 있다.
rate_state: dict[str, float | None] = {"requests": None, "tokens": None, "reset_s": None}


def _num(v) -> float | None:
    """'123', '1.5s', '2m' 같은 값을 초/개수로 관대하게 파싱한다."""
    if v is None:
        return None
    s = str(v).strip().lower()
    mult = 1.0
    if s.endswith("ms"):
        s, mult = s[:-2], 0.001
    elif s.endswith("s"):
        s, mult = s[:-1], 1.0
    elif s.endswith("m"):
        s, mult = s[:-1], 60.0
    try:
        return float(s) * mult
    except ValueError:
        return None


def note_rate_headers(headers) -> dict[str, float | None]:
    """응답 헤더에서 남은 한도를 기록한다."""
    if not headers:
        return rate_state
    get = headers.get if hasattr(headers, "get") else (lambda k: None)
    rate_state["requests"] = _num(get("x-ratelimit-remaining-requests"))
    rate_state["tokens"] = _num(get("x-ratelimit-remaining-tokens"))
    resets = [_num(get("x-ratelimit-reset-requests")), _num(get("x-ratelimit-reset-tokens"))]
    resets = [r for r in resets if r is not None]
    rate_state["reset_s"] = max(resets) if resets else None
    return rate_state


async def wait_if_nearly_exhausted(min_tokens: float = 8000, min_requests: float = 1, sleep=None) -> float:
    """남은 한도가 한 번의 호출도 못 버틸 만큼이면 리셋까지 미리 기다린다(429 예방)."""
    import asyncio

    sleep = sleep or asyncio.sleep
    reset = rate_state.get("reset_s")
    if reset is None or reset <= 0:
        return 0.0
    toks, reqs = rate_state.get("tokens"), rate_state.get("requests")
    short = (toks is not None and toks < min_tokens) or (reqs is not None and reqs < min_requests)
    if not short:
        return 0.0
    wait = min(reset, 60.0)
    log.warning("한도 소진 임박(잔여 토큰 %s, 요청 %s) — %.1f초 대기", toks, reqs, wait)
    await sleep(wait)
    rate_state["tokens"] = rate_state["requests"] = None
    return wait


async def throttle(min_interval: float) -> None:
    """호출 사이 최소 간격을 강제한다. CLOVA 는 QPM 뿐 아니라 TPM(분당 토큰)도 재므로,
    프롬프트가 큰 요청을 몰아 보내면 서비스앱에서도 429 가 난다."""
    import asyncio
    import time as _t

    if min_interval <= 0:
        return
    wait = _last_call_at["t"] + min_interval - _t.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call_at["t"] = _t.monotonic()


async def call_with_rate_limit_retry(make_call, *, retries: int = 4, sleep=None, min_interval: float = 0.0):
    """429(rate exceeded)만 지수 백오프로 재시도한다.

    CLOVA Studio 는 테스트앱 기준 요청 빈도 제한이 낮아, 평가처럼 질의를 연속으로 보내면 429 가 난다.
    이때 즉시 오류로 끝내면 그 문항은 '모델이 답하지 못한' 상태가 되므로 반드시 기다렸다 다시 부른다.
    """
    import asyncio

    import openai

    sleep = sleep or asyncio.sleep
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            await throttle(min_interval)
            await wait_if_nearly_exhausted()
            return await make_call()
        except openai.RateLimitError as e:  # 429 만 재시도 대상
            last = e
            if attempt >= retries:
                break
            after = None
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "headers", None):
                after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                note_rate_headers(resp.headers)
            wait = retry_delay(attempt, after)
            log.warning("429 rate limited; %.1f초 후 재시도 (%d/%d)", wait, attempt + 1, retries)
            await sleep(wait)
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- openai-compatible
class OpenAIConversation:
    def __init__(self, client, model: str, max_tokens: int, system: str, tools: list[ToolSpec], user_text: str,
                 rate_limit_retries: int = 4, min_interval: float = 0.0):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.rate_limit_retries = rate_limit_retries
        self.min_interval = min_interval
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]

    async def step(self) -> Turn:
        import openai

        try:
            async def _call():
                raw = await self.client.chat.completions.with_raw_response.create(
                    model=self.model, messages=self.messages, tools=self.tools or None, max_tokens=self.max_tokens
                )
                note_rate_headers(getattr(raw, "headers", None))
                return raw.parse()

            resp = await call_with_rate_limit_retry(
                _call, retries=self.rate_limit_retries, min_interval=self.min_interval
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
        self.rate_limit_retries = getattr(settings, "rate_limit_retries", 4)
        self.min_interval = getattr(settings, "request_min_interval_sec", 0.0)

    def start(self, system, tools, user_text):
        return OpenAIConversation(self.client, self.model, self.max_tokens, system, tools, user_text,
                                  rate_limit_retries=self.rate_limit_retries,
                                  min_interval=getattr(self, "min_interval", 0.0))


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
        self.rate_limit_retries = getattr(settings, "rate_limit_retries", 4)
        self.min_interval = getattr(settings, "request_min_interval_sec", 0.0)
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
