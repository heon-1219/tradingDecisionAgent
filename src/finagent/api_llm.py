"""The model provider part of `/api/*`: what `GET`/`PUT /api/settings` say and check about the LLM,
and `POST /api/llm/models`. `api_chat` owns the settings routes and calls in here; this router is
included in `api_chat.router`, so it has the same `/api` prefix, the 503 dependency and
`LocalApiGuard`.

**GET /api/settings** gains (`llm_settings_body`): `provider`, `model`, `effort`, the research
depth in force (`mode`, `max_rounds`, `time_budget_sec`, all from `config.chat_limits`),
`gemini_paid`, and `providers: {gemini, openai, anthropic: "set" | "missing"}`. Never a key.

**PUT /api/settings** gains `llm_provider`, `llm_model`, `llm_effort`, `gemini_paid` and the keys
`gemini_key`, `openai_key`, `anthropic_key` (the usual value rules: `""` removes, `null` leaves as
is). `check_llm_changes` runs before anything is saved, when the provider, the model or a provider
key is in the request:

1. the configuration as it would be must be usable: a model is named (400 `llm_model_required`)
   and the chosen provider has a key (400 `llm_key_required`);
2. if the provider, the model or *that* provider's key actually changes, one live validation call
   (`ChatRuntime.check_llm`, see `llm_admin.validate_llm`): a rejected key is 400
   `llm_invalid_key`, an unknown model 400 `llm_model_unavailable`, a network error, a 5xx or a
   provider response the app does not recognise 502 `llm_unreachable`; nothing is saved. A rate-limited key (429) works: the settings are saved and
   the response carries `"warning": "llm_rate_limited"`.

`gemini_paid` and `llm_effort` need no live call. A key for a provider that is not in use is saved
without one; it is checked when that provider is chosen. The decision is made outside the lock, so
`check_llm_changes` also returns the configuration it was made for; `ChatRuntime.apply_settings`
compares it under the lock and checks again if a concurrent write changed it (a failure there maps
to the same errors through `raise_for_llm_status`).

**POST /api/llm/models** `{provider, api_key?}` lists the chat models a key can use, for the model
picker: `{"provider", "models": [{"id", "label"}], "suggested"}`. Without `api_key` the saved key
of that provider is used; with neither, 400 `llm_key_required`. Errors: 400 `llm_invalid_key` (the
key was refused), 429 `llm_rate_limited`, 502 `llm_unreachable` (network error, 5xx, or an
unrecognised response), 502 `llm_list_failed` (the provider refused the listing for another
reason). It changes no state, and the key (in the body, never in a URL) is not saved, echoed or
logged; it is only added to the in-memory scrub list.

No error body here contains a key or a provider's own message.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import Field

from .api_common import ApiError, Body, GuardedRoute, Runtime, respond
from .config import chat_limits
from .envfile import EnvFileError, check_env_value
from .llm_admin import KEY_FIELDS, PROVIDERS, classify_llm_error, filter_models, key_for, suggest_model

log = logging.getLogger("finagent.api_llm")

Provider = Literal["gemini", "openai", "anthropic"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
# PUT /api/settings request field -> (settings field, label for messages), merged into api_chat's map.
LLM_KEY_FIELDS = {
    "gemini_key": ("gemini_api_key", "Gemini API key"),
    "openai_key": ("openai_api_key", "OpenAI API key"),
    "anthropic_key": ("anthropic_api_key", "Anthropic API key"),
    "llm_model": ("llm_model", "model name"),
}
# Request fields copied to the settings field of the same name, as they are.
LLM_PLAIN_FIELDS = frozenset({"llm_provider", "llm_effort", "gemini_paid"})

_NOTHING_SAVED = " Nothing was saved."
_MESSAGES = {
    "llm_invalid_key": "The provider rejected this API key. Check the key (and that the account has API access "
                       "and credit) and try again.",
    "llm_model_unavailable": "This model is not available for this key. Pick another model and try again.",
    "llm_unreachable": "The provider could not be reached, or answered in a way this app does not recognise. "
                       "Try again shortly.",
    "llm_rate_limited": "The provider is rate limiting this key right now. Try again shortly.",
    "llm_list_failed": "The provider did not list the models for this key. Check the key's access, or enter "
                       "a model id yourself.",
}
_STATUS_ERRORS = {"invalid_key": (400, "llm_invalid_key"), "model_unavailable": (400, "llm_model_unavailable"),
                  "unreachable": (502, "llm_unreachable")}

router = APIRouter(route_class=GuardedRoute)


def is_set(value: Any) -> str:
    return "set" if value else "missing"


def llm_settings_body(rt) -> dict:
    """The model provider fields of `GET /api/settings` (see the module docstring)."""
    s = rt.settings
    limits = chat_limits(s)
    seconds = limits.time_budget_sec
    return {
        "provider": s.llm_provider,
        "model": s.llm_model,
        "effort": s.llm_effort,
        "mode": limits.mode,
        "max_rounds": limits.max_rounds,
        "time_budget_sec": int(seconds) if float(seconds).is_integer() else seconds,
        "gemini_paid": bool(getattr(s, "gemini_paid", False)),
        "providers": {name: is_set(key_for(s, name)) for name in PROVIDERS},
    }


def raise_for_llm_status(status: str, provider: Any = None) -> str | None:
    """A check's outcome as `PUT /api/settings` reports it: None for ok, the warning code for a
    rate-limited key (it works: save, and say so), else raises the `ApiError` (nothing saved)."""
    if status == "ok":
        return None
    if status == "rate_limited":
        return "llm_rate_limited"
    if status == "model_required":
        raise ApiError(400, "llm_model_required", "Name a model for the provider." + _NOTHING_SAVED)
    if status == "key_required":
        raise ApiError(400, "llm_key_required", f"Enter an API key for {provider or 'the provider'} first."
                       + _NOTHING_SAVED)
    http_status, code = _STATUS_ERRORS.get(status, _STATUS_ERRORS["unreachable"])
    raise ApiError(http_status, code, _MESSAGES[code] + _NOTHING_SAVED)


async def check_llm_changes(rt, changes: dict[str, Any]) -> tuple[str | None, tuple | None]:
    """Check the LLM part of a `PUT /api/settings` before it is applied; `changes` is settings
    field -> new value (None removes). Raises `ApiError` (nothing is saved then). Returns (a
    warning code to add to the response or None, the configuration the decision was made for -
    `rt.llm_context(changes)` - or None when the request touches no LLM field). See the module
    docstring."""
    if "llm_model" in changes and not changes["llm_model"]:
        raise_for_llm_status("model_required")
    if not {"llm_provider", "llm_model", *KEY_FIELDS.values()} & set(changes):
        return None, None
    context = rt.llm_context(changes)
    provider, model, key = context
    if provider not in KEY_FIELDS:
        return None, context  # the stub provider (tests): nothing to check
    if not model:
        raise_for_llm_status("model_required")
    if not key:
        raise_for_llm_status("key_required", provider)
    if context == rt.llm_context({}):
        return None, context  # nothing the provider would see differently
    return raise_for_llm_status(await rt.check_llm(changes)), context


class ModelsRequest(Body):
    provider: Provider
    api_key: str | None = Field(default=None, max_length=256)


@router.post("/llm/models")
async def list_models(body: ModelsRequest, rt: Runtime):
    key = (body.api_key or "").strip(" \t\r\n")
    if key:
        try:
            check_env_value(key)  # the same value rules as a key that is saved
        except EnvFileError:
            raise ApiError(400, "invalid_value", "The API key contains characters that cannot be used: use plain "
                           "ASCII only, without line breaks, double quotes, backslashes or '${'.",
                           field="api_key") from None
    else:
        key = key_for(rt.settings, body.provider)
    if not key:
        raise ApiError(400, "llm_key_required", f"Enter an API key for {body.provider} first.")
    try:
        raw = await rt.list_llm_models(body.provider, key)
    except Exception as exc:  # noqa: BLE001 - classified; logged by type only (the text can hold the key)
        status = classify_llm_error(exc)
        log.info("listing %s models failed: %s (%s)", body.provider, status, type(exc).__name__)
        if status == "rate_limited":
            raise ApiError(429, "llm_rate_limited", _MESSAGES["llm_rate_limited"]) from None
        if status == "model_unavailable":  # the provider answered and refused, but not over the key
            raise ApiError(502, "llm_list_failed", _MESSAGES["llm_list_failed"]) from None
        http_status, code = _STATUS_ERRORS.get(status, _STATUS_ERRORS["unreachable"])
        raise ApiError(http_status, code, _MESSAGES[code]) from None
    current = rt.settings.llm_model
    custom = body.provider == "openai" and bool(getattr(rt.settings, "openai_base_url", None))
    models = filter_models(body.provider, raw, keep=current, custom_endpoint=custom)
    return respond({"provider": body.provider, "models": models,
                    "suggested": suggest_model(body.provider, models, current)})
