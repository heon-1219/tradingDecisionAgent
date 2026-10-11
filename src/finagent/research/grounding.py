"""Gemini native Google Search grounding connector (optional, never cached).

The rest of the app talks to Gemini through its OpenAI-compatible endpoint
(`config.gemini_base_url`), but Google Search grounding is only exposed on Gemini's *native*
`generateContent` API, so this connector goes there directly instead of through the OpenAI-style
client. It answers one question with a single grounded call and returns the model's text plus the
search metadata Google requires be surfaced: the sources it grounded on, the search queries it
ran, and `searchEntryPoint.renderedContent` ("Search Suggestions" HTML that Google's terms require
be shown unmodified, verbatim, wherever a grounded answer is shown).

Never cached: Gemini's terms forbid caching grounded output, so - unlike every other connector in
this package - `GroundedWeb` takes no `EvidenceStore`/cache dependency at all (see its
constructor). A grounded answer lives only in the chat turn that requested it; the toolbox is
responsible for keeping it out of the evidence/http_cache tables when it registers the sources.

Free-tier Gemini API keys cannot use Google Search grounding on Gemini 3.x models; Google reports
that as an ordinary HTTP error (commonly 400/403/429, with a message about quota, billing, or the
tool not being supported), which we can't tell apart from a genuine misconfiguration - so any of
those three statuses is mapped to one friendly, actionable message. Other statuses get a generic
message. Neither ever includes the API key or any request header.
"""
from __future__ import annotations

from dataclasses import dataclass

from .http import HttpClient, HttpError

_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_TIMEOUT_SEC = 30.0

_MISSING_KEY_ERROR = "GEMINI_API_KEY is not set."
_FREE_TIER_ERROR = (
    "Google Search grounding needs a Gemini API key with billing enabled "
    "(free-tier keys cannot use it)."
)
# Statuses Google is known to return for a free-tier/unsupported-tool key. We can't distinguish
# "no billing" from other causes at this layer, so all three get the same actionable message.
_FREE_TIER_STATUSES = (400, 403, 429)


@dataclass
class GroundedAnswer:
    text: str
    sources: list[dict]
    search_suggestions_html: str | None
    queries: list[str]
    error: str | None = None


class GroundedWeb:
    """Gemini native `generateContent` + `google_search` tool connector.

    Deliberately has no cache/evidence parameter - see the module docstring for why. `http` is the
    same shared `HttpClient` every other connector uses (for its timeout/logging/redaction
    behavior), but every call here is a plain POST with no retry and no cache lookup.
    """

    def __init__(self, http: HttpClient, api_key: str | None, model: str):
        self._http = http
        self._api_key = api_key
        self._model = model

    async def answer(self, question: str) -> GroundedAnswer:
        if not self._api_key:
            return GroundedAnswer(
                text="", sources=[], search_suggestions_html=None, queries=[], error=_MISSING_KEY_ERROR
            )

        url = _GENERATE_CONTENT_URL.format(model=self._model)
        body = {
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "tools": [{"google_search": {}}],
        }
        headers = {"x-goog-api-key": self._api_key}
        try:
            data = await self._http.post_json(url, json=body, headers=headers, timeout=_TIMEOUT_SEC)
        except HttpError as exc:
            return GroundedAnswer(
                text="",
                sources=[],
                search_suggestions_html=None,
                queries=[],
                error=self._friendly_error(exc),
            )

        return self._parse(data)

    @staticmethod
    def _friendly_error(exc: HttpError) -> str:
        # Only the status code is echoed back, never the response body: Google's error body is
        # usually safe (it never saw the key - that went in a header), but this connector never
        # takes the risk of a body accidentally containing something sensitive.
        if exc.status in _FREE_TIER_STATUSES:
            return f"{_FREE_TIER_ERROR} (HTTP {exc.status})"
        if exc.status == 0:  # no HTTP response at all (see research.http.HttpError)
            return "Gemini grounding request failed (no response: network error or timeout)."
        return f"Gemini grounding request failed (HTTP {exc.status})."

    @staticmethod
    def _parse(data: object) -> GroundedAnswer:
        candidates = (data or {}).get("candidates") or [] if isinstance(data, dict) else []
        if not candidates:
            return GroundedAnswer(text="", sources=[], search_suggestions_html=None, queries=[])
        candidate = candidates[0] or {}

        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)
        )

        grounding_metadata = candidate.get("groundingMetadata") or {}

        sources: list[dict] = []
        seen_uris: set[str] = set()
        for chunk in grounding_metadata.get("groundingChunks") or []:
            web = (chunk or {}).get("web") or {}
            uri = web.get("uri")
            if not uri or uri in seen_uris:
                continue
            seen_uris.add(uri)
            sources.append({"title": web.get("title"), "uri": uri})

        queries = list(grounding_metadata.get("webSearchQueries") or [])
        search_entry_point = grounding_metadata.get("searchEntryPoint") or {}
        search_suggestions_html = search_entry_point.get("renderedContent")

        return GroundedAnswer(
            text=text,
            sources=sources,
            search_suggestions_html=search_suggestions_html,
            queries=queries,
        )
