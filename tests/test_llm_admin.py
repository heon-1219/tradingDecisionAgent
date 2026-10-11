"""Tests for `finagent.llm_admin`: checking a provider key and model, and listing a key's models.

No network: the SDK clients run over a mock transport (`httpx.MockTransport` for the OpenAI SDK,
`httpx2.MockTransport` for the Anthropic SDK, which only accepts its own HTTP library), and the
pure helpers are tested directly.
"""
from __future__ import annotations

import json
import logging

import anthropic
import httpx
import httpx2
import openai
import pytest

from finagent import llm_admin
from finagent.config import Settings
from finagent.llm_admin import (
    MAX_MODELS,
    VALIDATION_MAX_TOKENS,
    classify_llm_error,
    fetch_models,
    filter_models,
    suggest_model,
    validate_llm,
)

GEMINI_KEY = "gm-admin-key-0123456789"
OPENAI_KEY = "sk-admin-openai-0123456789"
ANTHROPIC_KEY = "sk-ant-admin-0123456789"
KEYS = (GEMINI_KEY, OPENAI_KEY, ANTHROPIC_KEY)


def settings(provider: str, model: str, **over) -> Settings:
    fields = dict(llm_provider=provider, llm_model=model, gemini_api_key=GEMINI_KEY, openai_api_key=OPENAI_KEY,
                  anthropic_api_key=ANTHROPIC_KEY, openai_base_url=None)
    fields.update(over)
    return Settings(_env_file=None, **fields)


class Recorder:
    """A mock transport handler: replays `responses` (status, body) and records every request."""

    def __init__(self, *responses, lib=httpx):
        self.responses = list(responses)
        self.lib = lib
        self.requests: list = []

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0)
        if isinstance(status, Exception):
            raise status
        return self.lib.Response(status, json=body)

    def client(self):
        return self.lib.AsyncClient(transport=self.lib.MockTransport(self))

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


CHAT_OK = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": ""}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 16, "total_tokens": 19},
}
MESSAGE_OK = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "max_tokens",
    "stop_sequence": None, "content": [{"type": "text", "text": "pong"}],
    "usage": {"input_tokens": 3, "output_tokens": 16},
}


def openai_error(status: int, message: str = "nope") -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": None}}


def anthropic_error(kind: str, message: str = "nope") -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


# --- the validation call ----------------------------------------------------------------------------


async def test_validating_gemini_makes_one_minimal_call(caplog):
    rec = Recorder((200, CHAT_OK))
    with caplog.at_level(logging.DEBUG):
        assert await validate_llm(settings("gemini", "gemini-3.8-flash"), http_client=rec.client()) == "ok"
    [request] = rec.requests
    assert request.url.host == "generativelanguage.googleapis.com" and request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {GEMINI_KEY}"
    body = rec.body()
    assert body["model"] == "gemini-3.8-flash" and body["max_tokens"] == VALIDATION_MAX_TOKENS <= 32
    assert "tools" not in body and len(body["messages"]) == 1 and len(body["messages"][0]["content"]) < 20
    assert all(key not in caplog.text for key in KEYS)


async def test_validating_openai_uses_its_own_host_key_and_output_cap_field():
    rec = Recorder((200, CHAT_OK))
    assert await validate_llm(settings("openai", "gpt-x"), http_client=rec.client()) == "ok"
    [request] = rec.requests
    assert request.url.host == "api.openai.com" and request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = rec.body()
    assert body["max_completion_tokens"] == VALIDATION_MAX_TOKENS and "max_tokens" not in body and "tools" not in body
    # a custom OpenAI-compatible endpoint (OPENAI_BASE_URL) is honoured, with max_tokens
    rec = Recorder((200, CHAT_OK))
    custom = settings("openai", "some/model", openai_base_url="https://router.example.com/api/v1")
    assert await validate_llm(custom, http_client=rec.client()) == "ok"
    assert rec.requests[0].url.host == "router.example.com" and "max_tokens" in rec.body()


async def test_validating_openai_swaps_a_rejected_output_cap_field_once():
    rejected = openai_error(400, "Unrecognized request argument supplied: max_completion_tokens")
    rec = Recorder((400, rejected), (200, CHAT_OK))
    assert await validate_llm(settings("openai", "gpt-x"), http_client=rec.client()) == "ok"
    assert "max_completion_tokens" in rec.body(0) and "max_tokens" in rec.body(1)


async def test_validating_anthropic_makes_one_minimal_call():
    rec = Recorder((200, MESSAGE_OK), lib=httpx2)
    assert await validate_llm(settings("anthropic", "claude-opus-5-5"), http_client=rec.client()) == "ok"
    [request] = rec.requests
    assert request.url.host == "api.anthropic.com" and request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == ANTHROPIC_KEY
    body = rec.body()
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == VALIDATION_MAX_TOKENS
    assert "tools" not in body and "thinking" not in body and len(body["messages"]) == 1


@pytest.mark.parametrize("provider, model, lib, status, body, expected", [
    ("gemini", "gemini-x", httpx, 401, openai_error(401), "invalid_key"),
    ("gemini", "gemini-x", httpx, 403, openai_error(403), "invalid_key"),
    # Gemini answers a bad key with 400 API_KEY_INVALID, not 401
    ("gemini", "gemini-x", httpx, 400, openai_error(400, "API key not valid. Please pass a valid API key."),
     "invalid_key"),
    ("gemini", "gemini-x", httpx, 404, openai_error(404, "models/gemini-x is not found"), "model_unavailable"),
    ("gemini", "gemini-x", httpx, 400, openai_error(400, "gemini-x is not supported for generateContent"),
     "model_unavailable"),
    ("gemini", "gemini-x", httpx, 429, openai_error(429, "quota exceeded"), "rate_limited"),
    ("gemini", "gemini-x", httpx, 503, openai_error(503, "overloaded"), "unreachable"),
    ("openai", "gpt-x", httpx, 401, openai_error(401, f"Incorrect API key provided: {OPENAI_KEY}"), "invalid_key"),
    ("openai", "gpt-x", httpx, 404, openai_error(404, "The model gpt-x does not exist"), "model_unavailable"),
    ("openai", "gpt-x", httpx, 429, openai_error(429, "rate limit"), "rate_limited"),
    ("openai", "gpt-x", httpx, 500, openai_error(500, "server error"), "unreachable"),
    ("anthropic", "claude-x", httpx2, 401, anthropic_error("authentication_error", "invalid x-api-key"), "invalid_key"),
    ("anthropic", "claude-x", httpx2, 403, anthropic_error("permission_error"), "invalid_key"),
    ("anthropic", "claude-x", httpx2, 404, anthropic_error("not_found_error", "model: claude-x"), "model_unavailable"),
    ("anthropic", "claude-x", httpx2, 429, anthropic_error("rate_limit_error"), "rate_limited"),
    ("anthropic", "claude-x", httpx2, 529, anthropic_error("overloaded_error"), "unreachable"),
    ("anthropic", "claude-x", httpx2, 400,
     anthropic_error("invalid_request_error", "Your credit balance is too low to access the Anthropic API."),
     "invalid_key"),
])
async def test_validation_outcomes_by_provider_response(caplog, provider, model, lib, status, body, expected):
    rec = Recorder((status, body), lib=lib)
    with caplog.at_level(logging.DEBUG):
        assert await validate_llm(settings(provider, model), http_client=rec.client()) == expected
    assert len(rec.requests) == 1  # one call, never retried
    # neither a key nor the provider's message is logged: the exception type only
    assert all(key not in caplog.text for key in KEYS)
    assert body["error"]["message"] not in caplog.text


@pytest.mark.parametrize("provider, model, lib", [("gemini", "gemini-x", httpx), ("openai", "gpt-x", httpx),
                                                   ("anthropic", "claude-x", httpx2)])
async def test_a_network_failure_is_unreachable(provider, model, lib):
    rec = Recorder((lib.ConnectError("no route to host"), None), lib=lib)
    assert await validate_llm(settings(provider, model), http_client=rec.client()) == "unreachable"


async def test_the_stub_provider_needs_no_validation():
    assert await validate_llm(settings("stub", "anything")) == "ok"


def test_classify_handles_exceptions_that_are_not_provider_errors():
    assert classify_llm_error(RuntimeError("boom")) == "unreachable"
    assert classify_llm_error(TimeoutError()) == "unreachable"
    request = httpx.Request("POST", "http://x")
    assert classify_llm_error(openai.APITimeoutError(request=request)) == "unreachable"
    assert classify_llm_error(openai.APIConnectionError(request=request)) == "unreachable"
    odd = openai.BadRequestError("something else entirely", response=httpx.Response(400, request=request), body=None)
    assert classify_llm_error(odd) == "model_unavailable"  # the request reached the provider and was refused
    request2 = httpx2.Request("POST", "http://x")
    refused = anthropic.PermissionDeniedError("no", response=httpx2.Response(403, request=request2), body=None)
    assert classify_llm_error(refused) == "invalid_key"


# --- listing models -----------------------------------------------------------------------------------


def openai_models(*ids_created) -> dict:
    return {"object": "list", "data": [{"id": i, "object": "model", "created": c, "owned_by": "x"}
                                       for i, c in ids_created]}


async def test_fetch_models_from_openai_and_gemini():
    rec = Recorder((200, openai_models(("gpt-new", 200), ("gpt-old", 100))))
    raw = await fetch_models("openai", OPENAI_KEY, settings("gemini", "g"), http_client=rec.client())
    assert rec.requests[0].url.host == "api.openai.com" and rec.requests[0].url.path == "/v1/models"
    assert rec.requests[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert [(m["id"], m["created"]) for m in raw] == [("gpt-new", 200.0), ("gpt-old", 100.0)]

    rec = Recorder((200, openai_models(("models/gemini-3.8-flash", 0), ("models/embedding-001", 0))))
    raw = await fetch_models("gemini", GEMINI_KEY, settings("gemini", "g"), http_client=rec.client())
    assert rec.requests[0].url.host == "generativelanguage.googleapis.com"
    assert rec.requests[0].headers["authorization"] == f"Bearer {GEMINI_KEY}"
    assert [m["id"] for m in raw] == ["gemini-3.8-flash", "embedding-001"]  # the "models/" prefix is dropped


async def test_fetch_models_from_anthropic():
    page = {"data": [
        {"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5", "created_at": "2026-08-01T00:00:00Z"},
        {"id": "claude-haiku-4-5", "type": "model", "display_name": "Claude Haiku 4.5", "created_at": "2025-10-01T00:00:00Z"},
    ], "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-haiku-4-5"}
    rec = Recorder((200, page), lib=httpx2)
    raw = await fetch_models("anthropic", ANTHROPIC_KEY, settings("gemini", "g"), http_client=rec.client())
    assert rec.requests[0].url.path == "/v1/models" and rec.requests[0].headers["x-api-key"] == ANTHROPIC_KEY
    assert [(m["id"], m["label"]) for m in raw] == [("claude-opus-5-5", "Claude Opus 5.5"),
                                                   ("claude-haiku-4-5", "Claude Haiku 4.5")]
    assert raw[0]["created"] > raw[1]["created"] > 0


async def test_fetch_models_raises_the_provider_error_for_the_caller_to_classify():
    rec = Recorder((401, openai_error(401)))
    with pytest.raises(openai.AuthenticationError) as caught:
        await fetch_models("openai", OPENAI_KEY, settings("gemini", "g"), http_client=rec.client())
    assert classify_llm_error(caught.value) == "invalid_key"


def raw(*ids, created=None, label=None):
    return [{"id": i, "label": label, "created": (created or {}).get(i, 0.0)} for i in ids]


def test_filter_keeps_chat_models_and_drops_the_rest():
    gemini = raw("gemini-3.8-flash", "gemini-3.8-pro", "gemini-embedding-001", "text-embedding-004", "embedding-001",
                 "imagen-4.0-generate", "gemini-3.8-flash-image", "gemini-3.8-flash-preview-tts",
                 "gemini-3.8-flash-live-001", "gemini-3.8-flash-native-audio-dialog", "veo-3.0-generate", "aqa",
                 "gemma-3-27b-it", "gemini-robotics-er-1.5")
    assert [m["id"] for m in filter_models("gemini", gemini)] == ["gemini-3.8-pro", "gemini-3.8-flash"]

    openai_raw = raw("gpt-9", "gpt-9-mini", "o9", "text-embedding-3-small", "omni-moderation-latest", "whisper-1",
                     "tts-1-hd", "dall-e-3", "gpt-image-1", "gpt-9-audio-preview", "gpt-9-realtime-preview",
                     "gpt-9-transcribe", "gpt-9-mini-tts", "sora-2", "davinci-002", "babbage-002",
                     "gpt-3.5-turbo-instruct", "gpt-9-search-preview", "computer-use-preview")
    assert {m["id"] for m in filter_models("openai", openai_raw)} == {"gpt-9", "gpt-9-mini", "o9"}

    claude = raw("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "not-a-claude-model")
    assert {m["id"] for m in filter_models("anthropic", claude)} == {
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"}


def test_filter_output_shape_order_and_cap():
    items = raw("gpt-b", "gpt-a", "gpt-c", created={"gpt-a": 300.0, "gpt-b": 200.0, "gpt-c": 300.0})
    items[0]["label"] = "GPT B"
    out = filter_models("openai", items)
    # newest first; the same date in descending natural order of the id; a label falls back to the id
    assert out == [{"id": "gpt-c", "label": "gpt-c"}, {"id": "gpt-a", "label": "gpt-a"}, {"id": "gpt-b", "label": "GPT B"}]
    # without dates (Gemini): higher versions first
    versions = filter_models("gemini", raw("gemini-2.5-pro", "gemini-10.0-pro", "gemini-3.8-flash"))
    assert [m["id"] for m in versions] == ["gemini-10.0-pro", "gemini-3.8-flash", "gemini-2.5-pro"]
    many = filter_models("openai", raw(*[f"gpt-{n}" for n in range(200)]))
    assert len(many) == MAX_MODELS == 60 and many[0]["id"] == "gpt-199"
    # duplicates, blanks and junk are dropped
    assert filter_models("openai", [{"id": "gpt-a"}, {"id": "gpt-a"}, {"id": ""}, {"id": None}, "junk", {}]) == [
        {"id": "gpt-a", "label": "gpt-a"}]


def test_the_reviewers_probe_ids():
    gemini = filter_models("gemini", raw("publishers/google/models/gemini-2.5-flash", "gemini-nano-banana-2.1",
                                         "gemini-3.8-flash-image-preview", "gemini-imagegen-1", "gemini-3.8-pro"))
    assert [m["id"] for m in gemini] == ["gemini-3.8-pro", "gemini-2.5-flash"]  # Vertex-style path stripped
    llama = raw("meta-llama/llama-3.3-70b-instruct", "gpt-3.5-turbo-instruct")
    assert {m["id"] for m in filter_models("openai", llama, custom_endpoint=True)} == {
        "meta-llama/llama-3.3-70b-instruct", "gpt-3.5-turbo-instruct"}  # OpenRouter-style ids are chat models
    # on OpenAI itself "-instruct" is a legacy completions model
    assert [m["id"] for m in filter_models("openai", llama + raw("gpt-test"))] == ["gpt-test"]


def test_when_the_filter_leaves_nothing_the_raw_list_is_used(caplog):
    odd = raw("mystery-model-b", "mystery-model-a", created={"mystery-model-a": 2.0, "mystery-model-b": 1.0})
    assert [m["id"] for m in filter_models("gemini", odd)] == ["mystery-model-a", "mystery-model-b"]
    many = raw(*[f"text-embedding-{n}" for n in range(100)])
    out = filter_models("openai", many)
    assert len(out) == MAX_MODELS and out[0]["id"] == "text-embedding-99"
    assert filter_models("openai", []) == [] and filter_models("openai", [{"id": ""}]) == []


async def test_listing_has_an_overall_deadline_across_pages():
    import asyncio

    pages = {"n": 0}

    async def endless(request):
        pages["n"] += 1
        await asyncio.sleep(0.05)
        model_id = f"claude-page-{pages['n']}"
        return httpx2.Response(200, json={
            "data": [{"id": model_id, "type": "model", "display_name": model_id, "created_at": "2026-08-01T00:00:00Z"}],
            "has_more": True, "first_id": model_id, "last_id": model_id})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(endless))
    with pytest.raises(TimeoutError) as caught:
        await fetch_models("anthropic", ANTHROPIC_KEY, settings("gemini", "g"), http_client=client, deadline_sec=0.3)
    assert classify_llm_error(caught.value) == "unreachable"
    assert 2 <= pages["n"] <= 10


@pytest.mark.parametrize("provider, factory", [
    ("anthropic", "_anthropic_client"), ("openai", "_openai_client"), ("gemini", "_openai_client")])
async def test_a_listing_that_fails_before_the_first_page_still_closes_the_client(monkeypatch, provider, factory):
    """`models.list(...)` may raise when it is called (before any page is fetched): the client made
    for the listing is closed all the same."""
    class ListingFailed(Exception):
        pass

    closed: list[bool] = []

    class Models:
        def list(self, **kwargs):
            raise ListingFailed()

    class Client:
        models = Models()

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(llm_admin, factory, lambda *args, **kwargs: Client())
    with pytest.raises(ListingFailed):
        await fetch_models(provider, ANTHROPIC_KEY, settings("gemini", "g"))
    assert closed == [True]


def test_the_current_model_is_never_cut_by_the_cap():
    items = raw(*[f"gpt-{n}" for n in range(200)])
    out = filter_models("openai", items, keep="gpt-3")
    assert len(out) == MAX_MODELS and out[-1]["id"] == "gpt-3" and out[0]["id"] == "gpt-199"
    assert suggest_model("openai", out, "gpt-3") == "gpt-3"


def test_suggested_model():
    gemini = filter_models("gemini", raw("gemini-3.8-pro", "gemini-3.8-flash", "gemini-2.5-flash"))
    assert suggest_model("gemini", gemini, "gemini-2.5-flash") == "gemini-2.5-flash"  # the current model, if listed
    assert suggest_model("gemini", gemini, "claude-opus-5-5") == "gemini-3.8-flash"   # else the provider preference
    assert suggest_model("gemini", filter_models("gemini", raw("gemini-9-pro")), None) == "gemini-9-pro"

    dated = {"claude-opus-4-8": 100.0, "claude-opus-5-5": 300.0, "claude-sonnet-5-5": 400.0, "claude-haiku-4-5": 50.0}
    claude = filter_models("anthropic", raw(*dated, created=dated))
    assert suggest_model("anthropic", claude, "gemini-3.8-flash") == "claude-opus-5-5"  # the newest Opus
    no_opus = [m for m in claude if "opus" not in m["id"]]
    assert suggest_model("anthropic", no_opus, None) == "claude-sonnet-5-5"             # then Sonnet
    assert suggest_model("anthropic", [m for m in no_opus if "sonnet" not in m["id"]], None) == "claude-haiku-4-5"

    gpt = filter_models("openai", raw("gpt-old", "gpt-new", created={"gpt-old": 1.0, "gpt-new": 2.0}))
    assert suggest_model("openai", gpt, "gemini-3.8-flash") == "gpt-new"  # no hard-coded name: the first listed
    assert suggest_model("openai", [], "gpt-new") is None


def test_no_model_catalogue_is_hard_coded():
    """The list always comes from the provider: the module names one Gemini default and no others."""
    import inspect
    import re

    source = inspect.getsource(llm_admin)
    assert not re.search(r"gpt-\d|claude-(opus|sonnet|haiku)-\d|o\d-mini", source)
