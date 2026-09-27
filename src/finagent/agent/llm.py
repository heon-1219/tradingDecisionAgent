"""Provider-agnostic chat-with-tools adapters.

`LLMClient.start()` returns a `Conversation` that owns the provider-specific message history
(so Anthropic thinking blocks and Gemini thought signatures are echoed back untouched), exposes `step()`
to get the next assistant turn, and `add_tool_results()` to feed tool outputs back in ONE message.
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
    def add_assistant(self, text: str) -> None: ...
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

    def add_assistant(self, text: str):
        """Append a plain assistant message with no tool calls - used only when rebuilding a
        conversation from stored text history, where no thought signature echoing is needed."""
        self._messages.append({"role": "assistant", "content": text})

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
            return Turn(f"LLM call error: {type(e).__name__}: {e}", [], "error")

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

    def add_assistant(self, text: str):
        """Append a plain assistant message with no tool calls - used only when rebuilding a
        conversation from stored text history, where no thought signature echoing is needed."""
        self.messages.append({"role": "assistant", "content": text})

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
    """Wait before retrying a 429 or an overloaded server. A server-sent Retry-After takes precedence."""
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), cap))
        except (TypeError, ValueError):
            pass
    return min(base * (2 ** attempt), cap)


_last_call_at = {"t": 0.0}
# x-ratelimit-* headers of the last response, for providers that send them. Some providers meter requests
# and tokens per minute separately and return 429 when either runs out; reading the headers lets us pause
# before hitting the limit.
rate_state: dict[str, float | None] = {"requests": None, "tokens": None, "reset_s": None}


def _num(v) -> float | None:
    """Leniently parse values such as '123', '1.5s' or '2m' into seconds or counts."""
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
    """Record the remaining quota from response headers."""
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
    """If the remaining quota could not survive even one more call, wait for the reset first (prevents a 429)."""
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
    log.warning("quota nearly exhausted (tokens left %s, requests left %s) - waiting %.1fs", toks, reqs, wait)
    await sleep(wait)
    rate_state["tokens"] = rate_state["requests"] = None
    return wait


async def throttle(min_interval: float) -> None:
    """Enforce a minimum interval between calls. Providers that meter tokens per minute as well as requests
    return 429 when large prompts are sent back to back, even on paid tiers."""
    import asyncio
    import time as _t

    if min_interval <= 0:
        return
    wait = _last_call_at["t"] + min_interval - _t.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call_at["t"] = _t.monotonic()


async def call_with_rate_limit_retry(
    make_call, *, retries: int = 4, sleep=None, min_interval: float = 0.0, timeout_retries: int = 2,
    server_error_retries: int = 2,
):
    """Retry 429s (rate exceeded) and 5xx (e.g. Gemini's 503 "model is overloaded") with exponential backoff,
    and timeouts/connection errors briefly.

    Low-tier keys have low request limits, so sending questions back to back (as an evaluation run does)
    triggers 429s. Failing at once would leave that question unanswered by the model, so we always wait and
    call again.

    The same goes for timeouts: abandoning a question after a single 40 s timeout throws away most of the
    300 s budget (in the competition, G-020 ended unanswered at 84.8 s). The orchestrator enforces the budget
    separately, so here we retry only twice, quickly.
    """
    import asyncio

    import openai

    sleep = sleep or asyncio.sleep
    last: Exception | None = None
    timeouts = server_errors = 0
    for attempt in range(retries + 1):
        try:
            await throttle(min_interval)
            await wait_if_nearly_exhausted()
            return await make_call()
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            last = e
            timeouts += 1
            if timeouts > timeout_retries:
                break
            log.warning("%s; retrying in 2s (%d/%d)", type(e).__name__, timeouts, timeout_retries)
            await sleep(2.0)
        except openai.InternalServerError as e:  # 5xx: overloaded or transient server fault
            last = e
            if server_errors >= server_error_retries or attempt >= retries:
                break
            wait = retry_delay(server_errors)
            server_errors += 1
            log.warning("%s (%s); retrying in %.1fs (%d/%d)", type(e).__name__, getattr(e, "status_code", "5xx"),
                        wait, server_errors, server_error_retries)
            await sleep(wait)
        except openai.RateLimitError as e:  # exponential backoff for 429 only
            last = e
            if attempt >= retries:
                break
            after = None
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "headers", None):
                after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                note_rate_headers(resp.headers)
            wait = retry_delay(attempt, after)
            log.warning("429 rate limited; retrying in %.1fs (%d/%d)", wait, attempt + 1, retries)
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

    # Hooks for providers that speak this protocol with extras (see GeminiConversation).
    def _extra_kwargs(self) -> dict:
        return {}

    @staticmethod
    def _tool_call_dict(tc) -> dict:
        return {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}

    async def _create(self):
        async def _call():
            raw = await self.client.chat.completions.with_raw_response.create(
                model=self.model, messages=self.messages, tools=self.tools or None, max_tokens=self.max_tokens,
                **self._extra_kwargs(),
            )
            note_rate_headers(getattr(raw, "headers", None))
            return raw.parse()

        return await call_with_rate_limit_retry(_call, retries=self.rate_limit_retries, min_interval=self.min_interval)

    async def step(self) -> Turn:
        import openai

        try:
            resp = await self._create()
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
            log.error("openai-compatible error: %s", e)
            return Turn(f"LLM call error: {type(e).__name__}: {e}", [], "error")
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
             **({"tool_calls": [self._tool_call_dict(tc) for tc in msg.tool_calls]} if msg.tool_calls else {})}
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

    def add_assistant(self, text: str):
        """Append a plain assistant message with no tool calls - used only when rebuilding a
        conversation from stored text history, where no thought signature echoing is needed.
        Inherited by `GeminiConversation` unchanged: a rebuilt-history message never carries a
        thought signature (there is no tool call to attach one to)."""
        self.messages.append({"role": "assistant", "content": text})

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


# --------------------------------------------------------------------------- gemini
MISSING_GEMINI_KEY = (
    "LLM call error: GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
    "and add GEMINI_API_KEY=... to .env (GOOGLE_API_KEY also works)."
)


class GeminiConversation(OpenAIConversation):
    """OpenAI-shaped conversation with the two things Gemini adds on top of the protocol.

    * Thought signatures: Gemini 3 models attach an encrypted signature to each function call
      (``extra_content.google.thought_signature``) and reject the next request if it is missing from the
      assistant message we send back, so tool calls are echoed with their ``extra_content`` intact.
    * reasoning_effort: sent when configured; if a model rejects it, it is dropped once for the whole client.
    """

    def __init__(self, owner: "GeminiLLM", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner

    def _extra_kwargs(self) -> dict:
        effort = self.owner.reasoning_effort
        return {"reasoning_effort": effort} if effort else {}

    @staticmethod
    def _tool_call_dict(tc) -> dict:
        d = OpenAIConversation._tool_call_dict(tc)
        extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
        if extra:
            d["extra_content"] = extra
        return d

    async def _create(self):
        import openai

        try:
            return await super()._create()
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if self.owner.reasoning_effort and ("reasoning" in msg or "thinking" in msg):
                log.warning("%s rejected reasoning_effort=%s; continuing without it: %s",
                            self.model, self.owner.reasoning_effort, e)
                self.owner.reasoning_effort = None
                return await super()._create()
            raise

    async def step(self) -> Turn:
        if self.owner.missing_key:
            return Turn(MISSING_GEMINI_KEY, [], "error")
        return await super().step()


class GeminiLLM(OpenAILLM):
    """Google Gemini through its OpenAI-compatible endpoint (``/v1beta/openai/``).

    The endpoint accepts the OpenAI SDK with ``tools`` and ``reasoning_effort``, so the OpenAI conversation
    shape is reused; `GeminiConversation` adds thought-signature echoing and the effort fallback.
    """

    name = "gemini"
    # Gemini's reasoning_effort levels are low / medium / high; the higher settings map to high.
    EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

    def __init__(self, settings):
        import openai

        key = settings.gemini_api_key
        self.missing_key = not key
        if self.missing_key:
            log.error("GEMINI_API_KEY is not set - every question will fail until it is added to .env")
        self.client = openai.AsyncOpenAI(
            api_key=key or "missing-GEMINI_API_KEY",  # the SDK refuses an empty key; step() short-circuits instead
            base_url=settings.gemini_base_url,
            timeout=settings.llm_timeout_sec,
            max_retries=1,
        )
        self.model, self.max_tokens = settings.llm_model, settings.llm_max_tokens
        self.rate_limit_retries = getattr(settings, "rate_limit_retries", 4)
        self.min_interval = getattr(settings, "request_min_interval_sec", 0.0)
        self.reasoning_effort: str | None = self.EFFORT.get(settings.llm_effort)

    def start(self, system, tools, user_text):
        return GeminiConversation(self, self.client, self.model, self.max_tokens, system, tools, user_text,
                                  rate_limit_retries=self.rate_limit_retries, min_interval=self.min_interval)


def make_llm(settings) -> LLMClient:
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    if settings.llm_provider == "openai":
        return OpenAILLM(settings)
    return StubLLM([])
