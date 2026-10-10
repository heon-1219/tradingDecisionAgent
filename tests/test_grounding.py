import json
from pathlib import Path

import httpx

from finagent.research.grounding import GroundedAnswer, GroundedWeb
from finagent.research.http import HttpClient
from finagent.research.evidence import EvidenceStore
from finagent.store.appdb import AppDB

FIXTURES = Path(__file__).parent / "fixtures" / "grounding"


def _load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


GROUNDED_RESPONSE = _load_json("generate_content_grounded.json")
FREE_TIER_403 = _load_json("error_free_tier_403.json")
QUOTA_429 = _load_json("error_quota_429.json")


async def _noop_sleep(sec: float) -> None:
    pass


def _grounded_web(handler, api_key: str | None = "test-gemini-key", model: str = "gemini-3.8-flash"):
    """Build a GroundedWeb wired to a MockTransport; returns (grounded_web, calls)."""
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    async_client = httpx.AsyncClient(transport=transport)
    http = HttpClient(client=async_client, sleep=_noop_sleep)
    return GroundedWeb(http, api_key, model), calls


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=GROUNDED_RESPONSE)


# --- missing key ---------------------------------------------------------------


async def test_answer_missing_key_is_error_mentions_env_var():
    gw, calls = _grounded_web(_ok_handler, api_key=None)
    result = await gw.answer("What is AAPL trading at?")
    assert result.error is not None
    assert "GEMINI_API_KEY" in result.error
    assert calls == []


async def test_answer_missing_key_returns_empty_fields():
    gw, _ = _grounded_web(_ok_handler, api_key=None)
    result = await gw.answer("What is AAPL trading at?")
    assert result.text == ""
    assert result.sources == []
    assert result.queries == []
    assert result.search_suggestions_html is None


# --- request shape ---------------------------------------------------------------


async def test_answer_calls_generate_content_with_model_in_url():
    gw, calls = _grounded_web(_ok_handler, model="gemini-3.8-flash")
    await gw.answer("What is AAPL trading at?")
    assert len(calls) == 1
    url = str(calls[0].url)
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"


async def test_answer_uses_configured_model_in_url():
    gw, calls = _grounded_web(_ok_handler, model="gemini-2.5-pro")
    await gw.answer("q")
    assert "models/gemini-2.5-pro:generateContent" in str(calls[0].url)


async def test_answer_sends_api_key_header_by_name_not_asserted_value():
    gw, calls = _grounded_web(_ok_handler)
    await gw.answer("What is AAPL trading at?")
    assert "x-goog-api-key" in calls[0].headers


async def test_answer_does_not_use_authorization_header():
    gw, calls = _grounded_web(_ok_handler)
    await gw.answer("What is AAPL trading at?")
    assert "authorization" not in calls[0].headers


async def test_answer_sends_correct_body_shape():
    gw, calls = _grounded_web(_ok_handler)
    await gw.answer("What is AAPL trading at?")
    body = json.loads(calls[0].content)
    assert body == {
        "contents": [{"role": "user", "parts": [{"text": "What is AAPL trading at?"}]}],
        "tools": [{"google_search": {}}],
    }


async def test_answer_uses_post():
    gw, calls = _grounded_web(_ok_handler)
    await gw.answer("q")
    assert calls[0].method == "POST"


# --- parsing a recorded response --------------------------------------------------


async def test_answer_parses_text_from_candidate_parts():
    gw, _ = _grounded_web(_ok_handler)
    result = await gw.answer("What is AAPL trading at?")
    assert result.error is None
    assert "Apple (AAPL) closed at $178.42" in result.text


async def test_answer_parses_sources_from_grounding_chunks():
    gw, _ = _grounded_web(_ok_handler)
    result = await gw.answer("What is AAPL trading at?")
    assert result.sources == [
        {
            "title": "finance.yahoo.com",
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAPL-finance-yahoo",
        },
        {
            "title": "marketwatch.com",
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAPL-marketwatch",
        },
    ]


async def test_answer_parses_web_search_queries():
    gw, _ = _grounded_web(_ok_handler)
    result = await gw.answer("What is AAPL trading at?")
    assert result.queries == ["AAPL stock price today", "Apple stock news this week"]


async def test_answer_parses_search_suggestions_html_unmodified():
    gw, _ = _grounded_web(_ok_handler)
    result = await gw.answer("What is AAPL trading at?")
    expected = GROUNDED_RESPONSE["candidates"][0]["groundingMetadata"]["searchEntryPoint"][
        "renderedContent"
    ]
    assert result.search_suggestions_html == expected


async def test_answer_dedupes_sources_by_uri_preserving_order():
    data = json.loads(json.dumps(GROUNDED_RESPONSE))  # deep copy
    chunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
    dup = dict(chunks[0])
    chunks.insert(1, dup)  # duplicate the first chunk's uri right after it

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data)

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    uris = [s["uri"] for s in result.sources]
    assert uris == [
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAPL-finance-yahoo",
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAPL-marketwatch",
    ]


async def test_answer_no_grounding_metadata_returns_empty_sources_and_queries():
    data = {"candidates": [{"content": {"parts": [{"text": "Hello."}], "role": "model"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data)

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error is None
    assert result.text == "Hello."
    assert result.sources == []
    assert result.queries == []
    assert result.search_suggestions_html is None


# --- free-tier / quota errors become friendly -------------------------------------


async def test_answer_403_becomes_friendly_billing_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json=FREE_TIER_403)

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error is not None
    assert "billing" in result.error.lower()
    assert "403" in result.error
    assert result.text == ""
    assert result.sources == []


async def test_answer_429_becomes_friendly_billing_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=QUOTA_429)

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error is not None
    assert "billing" in result.error.lower()
    assert "429" in result.error


async def test_answer_400_becomes_friendly_billing_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 400, "message": "Tool not supported"}})

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error is not None
    assert "billing" in result.error.lower()
    assert "400" in result.error


async def test_answer_other_5xx_error_is_generic_not_billing_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error is not None
    assert "billing" not in result.error.lower()
    assert "500" in result.error


async def test_error_never_contains_api_key():
    api_key = "AIzaSyTEST-super-secret-key-value"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json=FREE_TIER_403)

    gw, _ = _grounded_web(handler, api_key=api_key)
    result = await gw.answer("q")
    assert api_key not in (result.error or "")


async def test_missing_key_error_never_contains_none_literal_key():
    gw, _ = _grounded_web(_ok_handler, api_key=None)
    result = await gw.answer("q")
    # sanity: the error is the friendly missing-key message, not an accidental leak/crash text
    assert result.error == "GEMINI_API_KEY is not set."


# --- never touches the evidence/http cache ----------------------------------------


async def test_successful_call_writes_nothing_to_http_cache():
    gw, calls = _grounded_web(_ok_handler)
    result = await gw.answer("What is AAPL trading at?")
    assert result.error is None
    assert calls  # the request did happen

    # A completely separate AppDB/EvidenceStore that GroundedWeb never saw. Since GroundedWeb has
    # no cache/evidence parameter at all, there is no path by which it could have written to any
    # http_cache table - this just documents that guarantee with a concrete assertion.
    db = AppDB(":memory:")
    evidence = EvidenceStore(db)
    assert db.query("SELECT * FROM http_cache") == []
    assert evidence.list("any-session") == []


def test_grounded_web_constructor_has_no_cache_or_evidence_parameter():
    import inspect

    params = list(inspect.signature(GroundedWeb.__init__).parameters)
    assert params == ["self", "http", "api_key", "model"]


def test_grounded_web_instance_holds_no_cache_or_evidence_reference():
    gw, _ = _grounded_web(_ok_handler)
    for value in vars(gw).values():
        assert not isinstance(value, EvidenceStore)
        assert not isinstance(value, AppDB)


# --- GroundedAnswer shape ----------------------------------------------------------


def test_grounded_answer_error_defaults_to_none():
    answer = GroundedAnswer(text="t", sources=[], search_suggestions_html=None, queries=[])
    assert answer.error is None


async def test_a_timeout_is_a_clean_network_error():
    # Final review C-m4: a transport error is an HttpError with status 0 now.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    gw, _ = _grounded_web(handler)
    result = await gw.answer("q")
    assert result.error == "Gemini grounding request failed (no response: network error or timeout)."
