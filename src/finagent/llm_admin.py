"""Checking a provider key and model, and listing the models a key can use.

Behind `PUT /api/settings` (the live check before a provider, model or key change is saved) and
`POST /api/llm/models` (the model picker). `ChatRuntime` calls `validate_llm` and `fetch_models`
through injectable factories, so tests never touch the network.

**The check** (`validate_llm`) is ONE minimal model call with a client built from the candidate
settings: a one-word prompt, an output cap of `VALIDATION_MAX_TOKENS`, no tools, no retries. It
returns a status and never raises:

| status | when |
|---|---|
| `ok` | the call succeeded |
| `invalid_key` | 401 or 403; a 400 that names the API key (Gemini answers a bad key with 400), or the account's credit or billing |
| `model_unavailable` | 404; any other 400 (the provider refused the request for this model) |
| `rate_limited` | 429: the key works |
| `unreachable` | a network error, a timeout, a 5xx, anything else |

**The list** (`fetch_models`) calls the provider's list-models endpoint (the OpenAI SDK with the
right base URL for `openai` and `gemini`, the Anthropic SDK for `anthropic`) and raises the
provider's exception; the caller classifies it with `classify_llm_error`. The whole listing,
every page included, must finish within `LIST_DEADLINE_SEC` (a `TimeoutError` otherwise).
`filter_models` keeps the chat-capable text models (by id pattern: no embeddings, image (Gemini's
"nano-banana" too), audio, TTS, live, moderation and the like; `-instruct` ids are kept for a
custom OpenAI-compatible endpoint, where they are chat models), newest first, at most
`MAX_MODELS`. If that leaves nothing of a non-empty list, the unfiltered list is offered instead
(capped and sorted the same way), so an unusual endpoint still gets a picker. `suggest_model`
picks the one to preselect. No model catalogue lives here: the list always comes from the provider.

**Secrets.** Nothing here logs, returns or stores a key or a provider message: failures are logged
by exception type and reported as a status. An exception's text is read only to classify it.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Literal

from .agent.llm import TOKEN_PARAMS, other_token_param, rejects_token_param
from .config import Settings

log = logging.getLogger("finagent.llm_admin")

PROVIDERS = ("gemini", "openai", "anthropic")
# Provider -> the settings field that holds its API key.
KEY_FIELDS = {"gemini": "gemini_api_key", "openai": "openai_api_key", "anthropic": "anthropic_api_key"}
LLMStatus = Literal["ok", "invalid_key", "model_unavailable", "rate_limited", "unreachable"]
STATUSES = frozenset({"ok", "invalid_key", "model_unavailable", "rate_limited", "unreachable"})

VALIDATION_MAX_TOKENS = 16
VALIDATION_PROMPT = "ping"
ADMIN_TIMEOUT_SEC = 20.0
MAX_MODELS = 60
LIST_DEADLINE_SEC = 30.0  # for a whole model listing, all pages included
_MAX_FETCHED = 2000  # models read from a provider's list before filtering (a paginated list stops here)

_KEY_WORDS = ("api key", "api_key", "apikey", "x-api-key", "credit", "billing")


class LLMCheckFailed(Exception):
    """A settings write whose model configuration failed its check: `status` is a status from the
    table above, or `key_required` / `model_required`. Carries no key and no provider text."""

    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


def key_for(settings: Any, provider: str) -> str | None:
    """The saved API key of `provider`, or None."""
    field = KEY_FIELDS.get(provider)
    value = getattr(settings, field, None) if field else None
    return value if isinstance(value, str) and value else None


def classify_llm_error(exc: BaseException) -> LLMStatus:
    """A provider (or network) exception as one of the statuses in the module docstring. Works for
    both SDKs: their API errors carry `status_code`; anything without one is `unreachable`."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "invalid_key"
    if status == 429:
        return "rate_limited"
    if status == 404:
        return "model_unavailable"
    if status == 400:
        text = str(exc).lower()  # read to classify only: never logged or returned
        return "invalid_key" if any(word in text for word in _KEY_WORDS) else "model_unavailable"
    return "unreachable"


def _timeout(settings: Any) -> float:
    try:
        return min(ADMIN_TIMEOUT_SEC, float(getattr(settings, "llm_timeout_sec", ADMIN_TIMEOUT_SEC)))
    except (TypeError, ValueError):
        return ADMIN_TIMEOUT_SEC


def _openai_client(provider: str, api_key: str, settings: Any, http_client: Any):
    import openai

    base_url = settings.gemini_base_url if provider == "gemini" else (getattr(settings, "openai_base_url", None) or None)
    extra = {"http_client": http_client} if http_client is not None else {}
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_timeout(settings), max_retries=0, **extra)


def _anthropic_client(api_key: str, settings: Any, http_client: Any):
    import anthropic

    extra = {"http_client": http_client} if http_client is not None else {}
    return anthropic.AsyncAnthropic(api_key=api_key, timeout=_timeout(settings), max_retries=0, **extra)


async def _openai_ping(client: Any, model: str, token_param: str) -> None:
    import openai

    async def send(param: str) -> None:
        await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": VALIDATION_PROMPT}],
            **{param: VALIDATION_MAX_TOKENS},
        )

    try:
        await send(token_param)
    except openai.BadRequestError as exc:
        if not rejects_token_param(exc, token_param):
            raise
        await send(other_token_param(token_param))  # the chat client makes the same swap


async def validate_llm(settings: Any, *, http_client: Any = None) -> LLMStatus:
    """One minimal call to `settings.llm_provider` / `settings.llm_model` with that provider's key
    (see the module docstring). `settings` is a candidate copy: nothing is saved here.
    `http_client` is for tests."""
    provider, model = getattr(settings, "llm_provider", None), getattr(settings, "llm_model", None)
    if provider not in KEY_FIELDS:
        return "ok"  # the stub provider: nothing to check
    key = key_for(settings, provider)
    try:
        if provider == "anthropic":
            client = _anthropic_client(key or "", settings, http_client)
            try:
                await client.messages.create(
                    model=model, max_tokens=VALIDATION_MAX_TOKENS,
                    messages=[{"role": "user", "content": VALIDATION_PROMPT}],
                )
            finally:
                await client.close()
        else:
            client = _openai_client(provider, key or "", settings, http_client)
            custom = provider == "gemini" or bool(getattr(settings, "openai_base_url", None))
            try:
                await _openai_ping(client, model, TOKEN_PARAMS[0] if custom else TOKEN_PARAMS[1])
            finally:
                await client.close()
    except Exception as exc:  # noqa: BLE001 - classified; logged by type only (the text can hold the key)
        status = classify_llm_error(exc)
        log.info("llm validation for %s: %s (%s)", provider, status, type(exc).__name__)
        return status
    return "ok"


def _created(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _gemini_id(model_id: str) -> str:
    """Gemini lists `models/<id>` (Vertex: `publishers/google/models/<id>`); chat takes `<id>`."""
    return model_id.rsplit("models/", 1)[-1]


async def fetch_models(provider: str, api_key: str, settings: Any, *, http_client: Any = None,
                       deadline_sec: float = LIST_DEADLINE_SEC) -> list[dict]:
    """The provider's model list for `api_key`, unfiltered, as `{"id", "label", "created"}` (label
    None and created 0.0 when the provider gives none). Raises the provider's exception, or
    `TimeoutError` when the listing (every page) takes longer than `deadline_sec`."""
    if provider not in KEY_FIELDS:
        raise ValueError("unknown provider")
    out: list[dict] = []
    if provider == "anthropic":
        client = _anthropic_client(api_key, settings, http_client)
        pages = client.models.list(limit=100)
    else:
        client = _openai_client(provider, api_key, settings, http_client)
        pages = client.models.list()
    try:
        async with asyncio.timeout(deadline_sec):
            async for model in pages:
                model_id = getattr(model, "id", None)
                if isinstance(model_id, str) and provider == "gemini":
                    model_id = _gemini_id(model_id)
                created = getattr(model, "created_at", None) if provider == "anthropic" else getattr(model, "created", None)
                out.append({"id": model_id, "label": getattr(model, "display_name", None), "created": _created(created)})
                if len(out) >= _MAX_FETCHED:
                    break
    finally:
        await client.close()
    return out


# --- filtering -----------------------------------------------------------------------------------------

# An id is split at every non-alphanumeric character; a model with one of these words is not a
# chat-capable text model (or cannot run this app's tool loop).
_DROP_WORDS = frozenset({
    "embedding", "embeddings", "embed", "image", "images", "imagen", "audio", "tts", "speech", "whisper",
    "transcribe", "transcription", "realtime", "live", "moderation", "dall", "sora", "veo", "aqa", "rerank",
    "instruct", "davinci", "babbage", "search", "computer", "robotics", "guard",
})
_DROP_SUBSTRINGS = ("embed", "moderation", "transcri", "whisper")
# Gemini's image and media models do not always say "image" as a word ("nano-banana", "imagegen").
_GEMINI_DROP_SUBSTRINGS = ("image", "nano-banana")
# The ids a provider's own chat models start with (none for openai: a custom OPENAI_BASE_URL may
# serve anything).
_ID_PREFIX = {"gemini": "gemini-", "anthropic": "claude"}
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_DIGITS_RE = re.compile(r"(\d+)")


def _is_chat_model(provider: str, model_id: str, custom_endpoint: bool = False) -> bool:
    lowered = model_id.lower()
    if not lowered.startswith(_ID_PREFIX.get(provider, "")):
        return False
    substrings = _DROP_SUBSTRINGS + (_GEMINI_DROP_SUBSTRINGS if provider == "gemini" else ())
    if any(part in lowered for part in substrings):
        return False
    # On OpenAI itself "-instruct" is a legacy completions model; behind a custom endpoint
    # (OpenRouter and similar) "meta-llama/...-instruct" is a chat model.
    words = _DROP_WORDS - {"instruct"} if custom_endpoint else _DROP_WORDS
    return not (words & set(_WORD_SPLIT_RE.split(lowered)))


def _natural_key(model_id: str) -> list:
    # Text and number runs alternate, so the same position always holds the same type.
    return [int(part) if i % 2 else part for i, part in enumerate(_DIGITS_RE.split(model_id.lower()))]


def filter_models(provider: str, raw: Any, *, keep: str | None = None, custom_endpoint: bool = False) -> list[dict]:
    """`raw` (from `fetch_models`) as the model picker's list: chat-capable text models only, as
    `{"id", "label"}`, newest first (models without a date: higher versions first), at most
    `MAX_MODELS`. `keep` (the model in use) is never cut by the cap. `custom_endpoint`: an
    OpenAI-compatible server other than OpenAI (`OPENAI_BASE_URL`). If the filter leaves nothing of
    a non-empty list, the unfiltered list is used (see the module docstring)."""
    valid: dict[str, dict] = {}
    for item in raw if isinstance(raw, (list, tuple)) else ():
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if provider == "gemini":
            model_id = _gemini_id(model_id)
        if model_id in valid:
            continue
        label = item.get("label")
        valid[model_id] = {"id": model_id, "label": label if isinstance(label, str) and label.strip() else model_id,
                           "_created": _created(item.get("created"))}
    seen = {k: v for k, v in valid.items() if _is_chat_model(provider, k, custom_endpoint)} or valid
    ordered = sorted(seen.values(), key=lambda m: _natural_key(m["id"]), reverse=True)
    ordered.sort(key=lambda m: m["_created"], reverse=True)  # stable: the id order breaks ties
    if len(ordered) > MAX_MODELS:
        tail = [m for m in ordered[MAX_MODELS:] if m["id"] == keep]
        ordered = ordered[: MAX_MODELS - len(tail)] + tail
    return [{"id": m["id"], "label": m["label"]} for m in ordered]


def suggest_model(provider: str, models: list[dict], current: str | None) -> str | None:
    """The id to preselect: the model in use if it is listed; else the provider's preference
    (Gemini: the app's default model; Anthropic: the newest Opus, then the newest Sonnet; OpenAI:
    no fixed name); else the first of the list. None for an empty list."""
    ids = [m["id"] for m in models]
    if not ids:
        return None
    if current in ids:
        return current
    if provider == "gemini":
        default = Settings.model_fields["llm_model"].default
        if default in ids:
            return default
    if provider == "anthropic":
        for family in ("opus", "sonnet"):
            for model_id in ids:  # newest first
                if family in model_id.lower():
                    return model_id
    return ids[0]
