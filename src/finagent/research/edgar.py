"""SEC EDGAR connector: ticker/CIK lookup, filings list, XBRL company facts, filing-text
passage search (BM25) and full-text search.

Every request must carry a descriptive `User-Agent` (SEC blocks anonymous/blank agents with a
403); the value comes from `Settings.sec_user_agent` via the constructor. If it is missing, every
public method here returns an error `ResearchResult` (or `None` for `cik_for`, whose return type
has no room for one) instead of making a request.

XBRL facts (`financial_facts`) and filing/search metadata (`filings`, `fulltext`) come from SEC's
structured JSON APIs, so they are recorded as trusted (`untrusted=False`). Filing document text
(`filing_passages`) is prose pulled out of a company's own HTML filing, so it is sanitized, marked
`untrusted=True` and wrapped with `wrap_untrusted` before it reaches the model, per the project's
"external text is untrusted" constraint.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from rank_bm25 import BM25Okapi

from .evidence import EvidenceStore, ResearchResult, SourceRecord, sanitize_untrusted, wrap_untrusted
from .http import HttpClient, HttpError

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index"

_TTL_TICKERS = 24 * 3600
_TTL_SUBMISSIONS = 12 * 3600
_TTL_COMPANYFACTS = 12 * 3600
_TTL_DOCUMENT = 10 * 365 * 24 * 3600  # "permanent": filings are immutable once filed
_TTL_FULLTEXT = 15 * 60

_PREFERRED_FORMS = ("10-K", "10-Q")

_CIK_ONLY_RE = re.compile(r"^\d{1,10}$")
_ITEM_HEADING_RE = re.compile(r"^item\s+\d+[a-z]?\.", re.I | re.M)
_WORD_RE = re.compile(r"\w+")
_MAX_CHUNK_CHARS = 2000


# --- minimal HTML -> text -------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Stdlib-only HTML-to-text: keeps block boundaries as newlines and table rows as lines."""

    _SKIP_TAGS = {"script", "style"}
    _BLOCK_TAGS = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br"}
    _CELL_TAGS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in self._CELL_TAGS:
            self._parts.append(" | ")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", ln).strip(" \t") for ln in raw.splitlines()]
        # A table row's first cell leaves a leading "| " artifact (the " | " separator is
        # emitted before every <td>/<th>, including the first one in a row) - drop it.
        lines = [re.sub(r"^\|\s*", "", ln) for ln in lines]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.text()


def _split_into_chunks(text: str) -> list[tuple[str, str]]:
    """Split on Item headings; large sections split further on blank lines (~2000 char cap)."""
    matches = list(_ITEM_HEADING_RE.finditer(text))
    if not matches:
        segments = [("", text)]
    else:
        segments = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            nl = text.find("\n", m.start())
            heading = text[m.start(): nl if nl != -1 else m.end()].strip()
            segments.append((heading, text[start:end]))

    chunks: list[tuple[str, str]] = []
    for heading, seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= _MAX_CHUNK_CHARS:
            chunks.append((heading, seg))
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", seg):
            para = para.strip()
            if not para:
                continue
            if buf and len(buf) + len(para) + 2 > _MAX_CHUNK_CHARS:
                chunks.append((heading, buf))
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append((heading, buf))
    return chunks


class Edgar:
    """SEC EDGAR connector: tickers, filings, XBRL company facts and filing-text search."""

    def __init__(self, http: HttpClient, evidence: EvidenceStore, user_agent: str | None):
        self._http = http
        self._evidence = evidence
        self._user_agent = user_agent

    # --- shared helpers --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent}

    @staticmethod
    def _missing_user_agent_result() -> ResearchResult:
        return ResearchResult(
            text=(
                "SEC_USER_AGENT is required to query SEC EDGAR (a descriptive User-Agent with "
                "contact information, or SEC returns 403). No request was made."
            ),
            sources=[],
            is_error=True,
        )

    @staticmethod
    def _normalize_cik(cik: str) -> str:
        s = str(cik).strip()
        if s.upper().startswith("CIK"):
            s = s[3:]
        return s.zfill(10)

    async def _resolve_cik(self, ticker_or_cik: str) -> str | None:
        s = str(ticker_or_cik).strip()
        core = s[3:] if s.upper().startswith("CIK") else s
        if _CIK_ONLY_RE.match(core):
            return core.zfill(10)
        return await self.cik_for(s)

    @staticmethod
    def _document_url(cik10: str, accession_no: str, primary_document: str) -> str:
        cik_int = str(int(cik10))
        accn_nodash = accession_no.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{primary_document}"

    # --- cached fetches ----------------------------------------------------------

    async def _tickers_map(self) -> dict[str, str] | None:
        key = "edgar:company_tickers"
        data = self._evidence.cache_get(key, _TTL_TICKERS)
        if data is None:
            try:
                data = await self._http.get_json(_TICKERS_URL, headers=self._headers())
            except HttpError:
                return None
            self._evidence.cache_put(key, data)
        mapping: dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik_str = entry.get("cik_str")
            if ticker and cik_str is not None:
                mapping[ticker] = f"{int(cik_str):010d}"
        return mapping

    async def _submissions(self, cik10: str) -> dict | None:
        key = f"edgar:submissions:{cik10}"
        data = self._evidence.cache_get(key, _TTL_SUBMISSIONS)
        if data is not None:
            return data
        try:
            data = await self._http.get_json(
                _SUBMISSIONS_URL.format(cik10=cik10), headers=self._headers()
            )
        except HttpError:
            return None
        self._evidence.cache_put(key, data)
        return data

    async def _companyfacts(self, cik10: str) -> dict | None:
        key = f"edgar:companyfacts:{cik10}"
        data = self._evidence.cache_get(key, _TTL_COMPANYFACTS)
        if data is not None:
            return data
        try:
            data = await self._http.get_json(
                _COMPANYFACTS_URL.format(cik10=cik10), headers=self._headers()
            )
        except HttpError:
            return None
        self._evidence.cache_put(key, data)
        return data

    async def _document_text(self, cik10: str, accession_no: str, primary_document: str) -> str | None:
        accn_nodash = accession_no.replace("-", "")
        key = f"edgar:document:{cik10}:{accn_nodash}:{primary_document}"
        html = self._evidence.cache_get(key, _TTL_DOCUMENT)
        if html is not None:
            return html
        url = self._document_url(cik10, accession_no, primary_document)
        try:
            html = await self._http.get_text(url, headers=self._headers())
        except HttpError:
            return None
        self._evidence.cache_put(key, html)
        return html

    # --- public API ----------------------------------------------------------------

    async def cik_for(self, ticker: str) -> str | None:
        """Resolve a ticker to its 10-digit zero-padded CIK, or None if unknown/unavailable."""
        if not self._user_agent:
            return None
        mapping = await self._tickers_map()
        if mapping is None:
            return None
        return mapping.get(str(ticker).strip().upper())

    async def filings(
        self,
        session_id: str,
        ticker_or_cik: str,
        forms: tuple[str, ...] | list[str] | None = ("10-K", "10-Q", "8-K"),
        limit: int = 10,
    ) -> ResearchResult:
        if not self._user_agent:
            return self._missing_user_agent_result()

        cik10 = await self._resolve_cik(ticker_or_cik)
        if cik10 is None:
            return ResearchResult(
                text=f"Could not resolve '{ticker_or_cik}' to a CIK on SEC EDGAR.",
                sources=[],
                is_error=True,
            )

        data = await self._submissions(cik10)
        if data is None:
            return ResearchResult(
                text=f"Could not fetch SEC EDGAR submissions for CIK {cik10}.",
                sources=[],
                is_error=True,
            )

        recent = data.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        form_list = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])
        company_name = data.get("name", ticker_or_cik)

        wanted_forms = set(forms) if forms else None
        sources: list[SourceRecord] = []
        lines: list[str] = []
        for i, accession_no in enumerate(accession_numbers):
            form = form_list[i] if i < len(form_list) else ""
            if wanted_forms is not None and form not in wanted_forms:
                continue
            filing_date = filing_dates[i] if i < len(filing_dates) else None
            report_date = report_dates[i] if i < len(report_dates) else None
            primary_doc = primary_docs[i] if i < len(primary_docs) else None
            doc_url = (
                self._document_url(cik10, accession_no, primary_doc)
                if primary_doc
                else f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/"
                f"{accession_no.replace('-', '')}/{accession_no}-index.htm"
            )
            source = self._evidence.add(
                session_id,
                connector="edgar",
                title=f"{company_name} {form} filed {filing_date}",
                url=doc_url,
                publisher="SEC EDGAR",
                tier=1,
                published_at=filing_date,
                snippet=(
                    f"{form} filed {filing_date}, report period {report_date}, "
                    f"accession {accession_no}."
                ),
                untrusted=False,
            )
            sources.append(source)
            lines.append(
                f"[{source.id}] {form} filed {filing_date} "
                f"(report period {report_date or 'n/a'}) accession {accession_no}"
            )
            if len(lines) >= limit:
                break

        if not lines:
            wanted_desc = ", ".join(sorted(wanted_forms)) if wanted_forms else "any"
            return ResearchResult(
                text=f"No filings found for {ticker_or_cik} matching forms: {wanted_desc}.",
                sources=[],
            )
        return ResearchResult(text="\n".join(lines), sources=sources)

    async def financial_facts(
        self,
        session_id: str,
        ticker_or_cik: str,
        concepts: list[str],
        periods: int = 8,
    ) -> ResearchResult:
        if not self._user_agent:
            return self._missing_user_agent_result()

        cik10 = await self._resolve_cik(ticker_or_cik)
        if cik10 is None:
            return ResearchResult(
                text=f"Could not resolve '{ticker_or_cik}' to a CIK on SEC EDGAR.",
                sources=[],
                is_error=True,
            )

        data = await self._companyfacts(cik10)
        if data is None:
            return ResearchResult(
                text=f"Could not fetch SEC EDGAR XBRL company facts for CIK {cik10}.",
                sources=[],
                is_error=True,
            )

        facts = data.get("facts", {}).get("us-gaap", {})
        url = _COMPANYFACTS_URL.format(cik10=cik10)
        sources: list[SourceRecord] = []
        lines: list[str] = []

        for concept in concepts:
            concept_data = facts.get(concept) or {}
            units = concept_data.get("units") or {}
            if not units:
                lines.append(f"{concept}: no data found.")
                continue

            if "USD" in units:
                unit_name, entries = "USD", units["USD"]
            else:
                unit_name, entries = next(iter(units.items()))

            preferred = [e for e in entries if e.get("form") in _PREFERRED_FORMS]
            pool = preferred if preferred else entries

            # Sort by filed date first so "last one wins" below means "latest-filed wins",
            # regardless of what order the source JSON happens to list entries in.
            pool_by_filed = sorted(pool, key=lambda e: e.get("filed") or "")
            deduped: dict[tuple, dict] = {}
            for entry in pool_by_filed:
                deduped[(entry.get("end"), entry.get("fp"))] = entry
            ordered = sorted(deduped.values(), key=lambda e: e.get("end") or "", reverse=True)
            selected = ordered[:periods]

            if not selected:
                lines.append(f"{concept}: no data found.")
                continue

            snippet_lines = [
                f"FY{e.get('fy')}{e.get('fp')} end={e.get('end')} val={e.get('val')} {unit_name} "
                f"form={e.get('form')} filed={e.get('filed')} accn={e.get('accn')}"
                for e in selected
            ]
            source = self._evidence.add(
                session_id,
                connector="edgar",
                title=f"{concept} (XBRL us-gaap, {ticker_or_cik})",
                url=url,
                publisher="SEC EDGAR XBRL",
                tier=1,
                published_at=selected[0].get("end"),
                snippet=f"{concept}: " + "; ".join(snippet_lines),
                untrusted=False,
            )
            sources.append(source)
            for sline in snippet_lines:
                lines.append(f"[{source.id}] {concept} {sline}")

        return ResearchResult(text="\n".join(lines), sources=sources)

    async def filing_passages(
        self,
        session_id: str,
        accession_no: str,
        cik: str,
        query: str,
        sections: list[str] | None = None,
        k: int = 5,
    ) -> ResearchResult:
        if not self._user_agent:
            return self._missing_user_agent_result()

        cik10 = self._normalize_cik(cik)
        submissions = await self._submissions(cik10)
        if submissions is None:
            return ResearchResult(
                text=f"Could not fetch SEC EDGAR submissions for CIK {cik10}.",
                sources=[],
                is_error=True,
            )

        recent = submissions.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        forms = recent.get("form", [])

        target_nodash = accession_no.replace("-", "")
        primary_doc = None
        form = None
        for i, accn in enumerate(accession_numbers):
            if accn.replace("-", "") == target_nodash:
                primary_doc = primary_docs[i] if i < len(primary_docs) else None
                form = forms[i] if i < len(forms) else None
                break

        if not primary_doc:
            return ResearchResult(
                text=f"Accession {accession_no} not found in SEC EDGAR submissions for CIK {cik10}.",
                sources=[],
                is_error=True,
            )

        html = await self._document_text(cik10, accession_no, primary_doc)
        if html is None:
            return ResearchResult(
                text=f"Could not fetch filing document for accession {accession_no}.",
                sources=[],
                is_error=True,
            )

        text = _html_to_text(html)
        chunks = _split_into_chunks(text)
        if sections:
            wanted = [s.lower() for s in sections]
            filtered = [c for c in chunks if any(w in c[0].lower() for w in wanted)]
            if filtered:
                chunks = filtered

        if not chunks:
            return ResearchResult(
                text=f"No content extracted from filing {accession_no}.",
                sources=[],
                is_error=True,
            )

        tokenized_corpus = [_WORD_RE.findall(c[1].lower()) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = _WORD_RE.findall(query.lower())
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        top = ranked[:k]

        doc_url = self._document_url(cik10, accession_no, primary_doc)
        sources: list[SourceRecord] = []
        lines: list[str] = []
        for (heading, chunk_text), _score in top:
            clean = sanitize_untrusted(chunk_text, max_chars=_MAX_CHUNK_CHARS)
            source = self._evidence.add(
                session_id,
                connector="edgar",
                title=(f"{form} {heading}".strip() if (form or heading) else f"Filing {accession_no}"),
                url=doc_url,
                publisher="SEC EDGAR",
                tier=1,
                published_at=None,
                snippet=clean,
                untrusted=True,
            )
            sources.append(source)
            lines.append(f"[{source.id}] {wrap_untrusted(source.id, clean)}")

        return ResearchResult(text="\n".join(lines), sources=sources)

    async def fulltext(
        self,
        session_id: str,
        query: str,
        forms: list[str] | tuple[str, ...] | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
    ) -> ResearchResult:
        if not self._user_agent:
            return self._missing_user_agent_result()

        params: dict[str, str] = {"q": query}
        if forms:
            params["forms"] = ",".join(forms)
        if start or end:
            params["dateRange"] = "custom"
            if start:
                params["startdt"] = start
            if end:
                params["enddt"] = end

        cache_key = "edgar:fulltext:" + json.dumps(params, sort_keys=True)
        data = self._evidence.cache_get(cache_key, _TTL_FULLTEXT)
        if data is None:
            try:
                data = await self._http.get_json(_FULLTEXT_URL, params=params, headers=self._headers())
            except HttpError:
                return ResearchResult(
                    text=f"SEC EDGAR full-text search failed for '{query}'.",
                    sources=[],
                    is_error=True,
                )
            self._evidence.cache_put(cache_key, data)

        hits = ((data or {}).get("hits") or {}).get("hits") or []
        sources: list[SourceRecord] = []
        lines: list[str] = []
        for hit in hits[:limit]:
            src = hit.get("_source") if isinstance(hit, dict) else None
            if not isinstance(src, dict):
                continue
            adsh = src.get("adsh")
            ciks = src.get("ciks") or []
            if not adsh or not ciks:
                continue
            try:
                cik_int = int(str(ciks[0]).lstrip("0") or "0")
            except ValueError:
                continue
            form = src.get("form", "")
            file_date = src.get("file_date")
            display_names = src.get("display_names") or []
            name = display_names[0] if display_names else "Unknown"
            accn_nodash = adsh.replace("-", "")
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{adsh}-index.htm"
            )
            source = self._evidence.add(
                session_id,
                connector="edgar",
                title=f"{form} - {name}",
                url=index_url,
                publisher="SEC EDGAR full-text search",
                tier=1,
                published_at=file_date,
                snippet=f"{form} filed {file_date}, {name}, accession {adsh}.",
                untrusted=False,
            )
            sources.append(source)
            lines.append(f"[{source.id}] {form} filed {file_date} - {name} (accession {adsh})")

        if not lines:
            return ResearchResult(text=f"No SEC EDGAR full-text search results for '{query}'.", sources=[])
        return ResearchResult(text="\n".join(lines), sources=sources)
