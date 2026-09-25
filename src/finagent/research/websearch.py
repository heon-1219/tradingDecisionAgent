"""Web search connector: Tavily search/extract, with a keyless Jina Reader fallback for fetch.

`search` gives the chat agent general web research (Tavily `/search`); `fetch` retrieves and
compacts one page (Tavily `/extract`, falling back to Jina Reader - `r.jina.ai`, which needs no
key - whenever Tavily has no key or its extract fails). Every result is free text pulled from the
open web, so per the project's "external text is untrusted" constraint every `SourceRecord` here
is `untrusted=True` and every returned line is wrapped with `wrap_untrusted`, after being run
through `sanitize_untrusted` (which strips hidden/zero-width text - a common prompt-injection
vector on fetched pages).

Each source is assigned a trust `tier` by (registrable) domain: SEC/investor-relations/exchanges
are tier 1, major wire/financial press is tier 2, other media defaults to tier 3, and blogs/forums/
reddit are tier 4. This is a coarse, regex/suffix-based classification (not a curated allowlist),
so it only recognizes the domains listed below plus the `ir.`/`investor(s).` hostname pattern.
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from rank_bm25 import BM25Okapi

from .evidence import EvidenceStore, ResearchResult, SourceRecord, sanitize_untrusted, wrap_untrusted
from .http import HttpClient, HttpError

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
_JINA_READER_PREFIX = "https://r.jina.ai/"

_TTL_SEARCH_NEWS = 15 * 60
_TTL_SEARCH_DEFAULT = 12 * 3600
_TTL_FETCH = 12 * 3600

_SNIPPET_MAX_CHARS = 1000  # per-result search snippet cap, independent of fetch's max_chars

_WORD_RE = re.compile(r"\w+")

# --- domain tiers ----------------------------------------------------------------

_TIER1_DOMAINS = {
    "sec.gov",
    # exchanges
    "nyse.com",
    "nasdaq.com",
    "cboe.com",
    "otcmarkets.com",
    "cme.com",
    "theice.com",
}
_TIER2_DOMAINS = {"reuters.com", "bloomberg.com", "wsj.com", "ft.com", "benzinga.com"}
# Explicitly named in the brief: reddit. The rest are common blog/forum platforms added for
# reasonable coverage; anything not listed here (or in tiers 1/2) defaults to tier 3 ("other
# media"), so this list intentionally errs small rather than guessing at unlisted sites.
_TIER4_DOMAINS = {
    "reddit.com",
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",
    "quora.com",
    "stackexchange.com",
}
_IR_HOST_PREFIXES = ("ir.", "investor.", "investors.")


def _host_matches(host: str, domain: str) -> bool:
    """True if `host` is exactly `domain` or a subdomain of it (registrable-domain suffix match)."""
    return host == domain or host.endswith("." + domain)


def _tier_for_host(host: str | None) -> int:
    if not host:
        return 3
    host = host.lower()
    if host.startswith(_IR_HOST_PREFIXES):
        return 1
    if any(_host_matches(host, d) for d in _TIER1_DOMAINS):
        return 1
    if any(_host_matches(host, d) for d in _TIER2_DOMAINS):
        return 2
    if any(_host_matches(host, d) for d in _TIER4_DOMAINS):
        return 4
    return 3


def _hostname(url: str | None) -> str | None:
    if not url:
        return None
    return urlsplit(url).hostname


# --- focus_query paragraph selection (BM25, like edgar.py's passage ranking) ------


def _focus_paragraphs(text: str, focus_query: str, max_chars: int) -> str:
    """Keep the most BM25-relevant blank-line-separated paragraphs, up to max_chars total."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return text[:max_chars]
    if len(paragraphs) == 1:
        return paragraphs[0][:max_chars]

    tokenized = [_WORD_RE.findall(p.lower()) for p in paragraphs]
    bm25 = BM25Okapi(tokenized)
    query_tokens = _WORD_RE.findall(focus_query.lower())
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
    selected: list[int] = []
    total = 0
    for i in ranked:
        extra = len(paragraphs[i]) + (2 if selected else 0)  # +2 for the "\n\n" join
        if selected and total + extra > max_chars:
            continue
        selected.append(i)
        total += extra
        if total >= max_chars:
            break

    selected.sort()  # restore original reading order
    return "\n\n".join(paragraphs[i] for i in selected)[:max_chars]


class WebSearch:
    """Tavily web search/extract connector, with a keyless Jina Reader fallback for `fetch`."""

    def __init__(self, http: HttpClient, evidence: EvidenceStore, tavily_key: str | None):
        self._http = http
        self._evidence = evidence
        self._tavily_key = tavily_key

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._tavily_key}"}

    @staticmethod
    def _missing_key_result() -> ResearchResult:
        return ResearchResult(
            text=(
                "TAVILY_API_KEY is required for web search. No request was made."
            ),
            sources=[],
            is_error=True,
        )

    # --- search ------------------------------------------------------------------

    async def search(
        self,
        session_id: str,
        query: str,
        topic: str = "general",
        start_date: str | None = None,
        end_date: str | None = None,
        include_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> ResearchResult:
        if not self._tavily_key:
            return self._missing_key_result()

        payload: dict[str, Any] = {
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if include_domains:
            payload["include_domains"] = include_domains

        cache_key = "websearch:search:" + json.dumps(payload, sort_keys=True)
        ttl = _TTL_SEARCH_NEWS if topic == "news" else _TTL_SEARCH_DEFAULT
        data = self._evidence.cache_get(cache_key, ttl)
        if data is None:
            try:
                data = await self._http.post_json(
                    _TAVILY_SEARCH_URL, json=payload, headers=self._auth_headers()
                )
            except HttpError:
                return ResearchResult(
                    text=f"Tavily search failed for '{query}'.", sources=[], is_error=True
                )
            self._evidence.cache_put(cache_key, data)

        results = (data or {}).get("results") or []
        sources: list[SourceRecord] = []
        lines: list[str] = []
        for r in results[:max_results]:
            url = r.get("url")
            host = _hostname(url)
            tier = _tier_for_host(host)
            published = r.get("published_date")
            clean = sanitize_untrusted(r.get("content") or "", max_chars=_SNIPPET_MAX_CHARS)
            source = self._evidence.add(
                session_id,
                connector="websearch",
                title=r.get("title") or url or "Untitled",
                url=url,
                publisher=host,
                tier=tier,
                published_at=published,
                snippet=clean,
                untrusted=True,
            )
            sources.append(source)
            suffix = f" ({published})" if published else ""
            lines.append(f"[{source.id}] {wrap_untrusted(source.id, clean)}{suffix}")

        if not lines:
            return ResearchResult(text=f"No web search results for '{query}'.", sources=[])
        return ResearchResult(text="\n".join(lines), sources=sources)

    # --- fetch -------------------------------------------------------------------

    async def fetch(
        self,
        session_id: str,
        url: str,
        focus_query: str | None = None,
        max_chars: int = 6000,
    ) -> ResearchResult:
        host = _hostname(url)
        tier = _tier_for_host(host)

        cache_key = f"websearch:fetch:{url}"
        raw_text = self._evidence.cache_get(cache_key, _TTL_FETCH)
        if raw_text is None:
            raw_text = await self._fetch_raw(url)
            if raw_text is None:
                return ResearchResult(text=f"Could not fetch {url}.", sources=[], is_error=True)
            self._evidence.cache_put(cache_key, raw_text)

        if focus_query:
            body = _focus_paragraphs(raw_text, focus_query, max_chars)
        else:
            body = raw_text[:max_chars]

        clean = sanitize_untrusted(body, max_chars=max_chars)
        source = self._evidence.add(
            session_id,
            connector="websearch",
            title=url,
            url=url,
            publisher=host,
            tier=tier,
            published_at=None,
            snippet=clean,
            untrusted=True,
        )
        return ResearchResult(text=f"[{source.id}] {wrap_untrusted(source.id, clean)}", sources=[source])

    async def _fetch_raw(self, url: str) -> str | None:
        if self._tavily_key:
            raw = await self._fetch_tavily_extract(url)
            if raw is not None:
                return raw
        return await self._fetch_jina(url)

    async def _fetch_tavily_extract(self, url: str) -> str | None:
        try:
            data = await self._http.post_json(
                _TAVILY_EXTRACT_URL,
                json={"urls": [url], "extract_depth": "basic"},
                headers=self._auth_headers(),
            )
        except HttpError:
            return None
        results = (data or {}).get("results") or []
        if not results:
            return None
        return results[0].get("raw_content") or None

    async def _fetch_jina(self, url: str) -> str | None:
        try:
            return await self._http.get_text(
                f"{_JINA_READER_PREFIX}{url}", headers={"X-Return-Format": "markdown"}
            )
        except HttpError:
            return None
