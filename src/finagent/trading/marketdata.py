"""Alpaca market-data connector: real-time quotes, historical bars, news and screeners.

Goes through the shared `research.http.HttpClient` (GET retries are fine here, unlike the paper
broker in `broker.py` - these are read-only market-data lookups, never order submissions). Same
`APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers as the broker, sent explicitly on each request,
against a different host (`data.alpaca.markets`).

`news` returns a `ResearchResult` like every other research connector (see `research/websearch.py`
and `research/edgar.py`): headlines/summaries are free text from an outside provider, so each
record is tier 2, `untrusted=True`, and every externally-sourced field - headline, summary,
source and created_at - is sanitized and placed *inside* the `<untrusted>` wrap, never appended
after it. This mirrors `websearch.py`'s handling of Tavily's `published_date`: an unsanitized
field placed outside the wrapper would let a malicious value inject text (or a literal
"</untrusted>") that the model would see as if it were trusted.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from ..research.evidence import EvidenceStore, ResearchResult, SourceRecord, sanitize_untrusted, wrap_untrusted
from ..research.http import HttpClient
from .models import Bar, Quote, normalize_symbol

DATA_BASE = "https://data.alpaca.markets"

# Alpaca's SIP feed requires data to be at least ~15 minutes old (free/paper accounts don't have
# a real-time SIP subscription); everything more recent must use the free IEX feed instead.
_SIP_MIN_DELAY = timedelta(minutes=15)

# Hard cap on /v2/stocks/bars pagination, independent of `limit`: a next_page_token that never
# stops (a misbehaving/malicious upstream, or a bug) would otherwise loop forever. 20 pages of the
# API's own max page size comfortably covers any `limit` this method is realistically called
# with; if it's ever hit, `bars` just returns what it collected instead of looping past it.
_MAX_BARS_PAGES = 20

_NEWS_MAX_LIMIT = 50
_NEWS_HEADLINE_MAX_CHARS = 200
_NEWS_SUMMARY_MAX_CHARS = 1000
_NEWS_META_MAX_CHARS = 60


def _parse_timestamp(value: str) -> datetime | None:
    """Parse an Alpaca RFC3339 timestamp (or a bare date) into an aware UTC datetime, or None if
    it can't be parsed - callers treat that as "can't tell, assume recent"."""
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class AlpacaData:
    def __init__(
        self,
        key_id: str,
        secret: str,
        http: HttpClient,
        now: Callable[[], datetime] | None = None,
    ):
        if not key_id or not secret:
            raise ValueError("Alpaca key_id and secret are both required.")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self._http = http
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _feed_for_end(self, end: str | None) -> str:
        if not end:
            return "iex"
        end_dt = _parse_timestamp(end)
        if end_dt is None:
            return "iex"
        return "sip" if self._now() - end_dt >= _SIP_MIN_DELAY else "iex"

    # --- quotes ----------------------------------------------------------------------

    async def snapshots(self, symbols: list[str]) -> dict[str, Quote]:
        data = await self._http.get_json(
            f"{DATA_BASE}/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": "iex"},
            headers=self._headers,
        )
        result: dict[str, Quote] = {}
        for symbol, snap in ((data or {}).get("snapshots") or {}).items():
            quote = (snap or {}).get("latestQuote") or {}
            trade = (snap or {}).get("latestTrade") or {}
            result[symbol] = Quote(
                symbol=symbol,
                bid=quote.get("bp"),
                ask=quote.get("ap"),
                last=trade.get("p"),
                as_of=trade.get("t") or quote.get("t"),
                feed="iex",
            )
        return result

    # --- bars ------------------------------------------------------------------------

    async def bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        symbol = normalize_symbol(symbol)  # also the key Alpaca's response groups bars under
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "all",
            "feed": self._feed_for_end(end),
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        bars: list[Bar] = []
        page_token: str | None = None
        for _ in range(_MAX_BARS_PAGES):
            request_params = dict(params)
            if page_token:
                request_params["page_token"] = page_token
            data = await self._http.get_json(
                f"{DATA_BASE}/v2/stocks/bars", params=request_params, headers=self._headers
            )
            for raw in ((data or {}).get("bars") or {}).get(symbol, []):
                if not isinstance(raw, dict) or "t" not in raw:
                    continue  # malformed bar (no timestamp) - skip rather than raise
                try:
                    bar = Bar(t=raw["t"], o=raw.get("o"), h=raw.get("h"), l=raw.get("l"), c=raw.get("c"), v=raw.get("v"))
                except ValidationError:
                    continue  # malformed bar (missing/invalid OHLCV) - skip rather than raise
                bars.append(bar)
                if len(bars) >= limit:
                    return bars
            page_token = (data or {}).get("next_page_token")
            if not page_token:
                return bars
        return bars

    # --- news --------------------------------------------------------------------

    async def news(
        self,
        session_id: str,
        evidence: EvidenceStore,
        symbols: list[str] | None = None,
        start: str | None = None,
        limit: int = 10,
    ) -> ResearchResult:
        params: dict[str, Any] = {
            "limit": min(limit, _NEWS_MAX_LIMIT),
            "include_content": "false",
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start:
            params["start"] = start

        data = await self._http.get_json(
            f"{DATA_BASE}/v1beta1/news", params=params, headers=self._headers
        )
        items = (data or {}).get("news") or []

        sources: list[SourceRecord] = []
        lines: list[str] = []
        for item in items:
            clean_headline = sanitize_untrusted(
                item.get("headline") or "", max_chars=_NEWS_HEADLINE_MAX_CHARS
            )
            clean_summary = sanitize_untrusted(
                item.get("summary") or "", max_chars=_NEWS_SUMMARY_MAX_CHARS
            )
            raw_source = item.get("source")
            raw_created = item.get("created_at")
            clean_source = (
                sanitize_untrusted(str(raw_source), max_chars=_NEWS_META_MAX_CHARS)
                if raw_source
                else None
            )
            clean_created = (
                sanitize_untrusted(str(raw_created), max_chars=_NEWS_META_MAX_CHARS)
                if raw_created
                else None
            )

            snippet = f"{clean_headline} {clean_summary}".strip()
            body = f"{clean_headline} — {clean_summary}" if clean_summary else clean_headline
            meta = ", ".join(x for x in (clean_source, clean_created) if x)
            if meta:
                body = f"{body} ({meta})"

            source = evidence.add(
                session_id,
                connector="news",
                title=clean_headline or "Untitled",
                url=item.get("url"),
                publisher=clean_source,
                tier=2,
                published_at=clean_created,
                snippet=snippet,
                untrusted=True,
            )
            sources.append(source)
            lines.append(f"[{source.id}] {wrap_untrusted(source.id, body)}")

        if not lines:
            return ResearchResult(text="No news found.", sources=[])
        return ResearchResult(text="\n".join(lines), sources=sources)

    # --- screeners -----------------------------------------------------------------

    async def movers(self, top: int = 10) -> dict:
        return await self._http.get_json(
            f"{DATA_BASE}/v1beta1/screener/stocks/movers",
            params={"top": top},
            headers=self._headers,
        )

    async def most_actives(self, top: int = 10) -> list[dict]:
        data = await self._http.get_json(
            f"{DATA_BASE}/v1beta1/screener/stocks/most-actives",
            params={"by": "volume", "top": top},
            headers=self._headers,
        )
        return (data or {}).get("most_actives") or []
