import json
from pathlib import Path

import httpx
import pytest

from finagent.research.edgar import Edgar
from finagent.research.evidence import EvidenceStore
from finagent.research.http import HttpClient
from finagent.store.appdb import AppDB

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def _load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


TICKERS = _load_json("company_tickers.json")
SUBMISSIONS = _load_json("submissions_aapl.json")
COMPANYFACTS = _load_json("companyfacts_aapl_revenues.json")
FULLTEXT = _load_json("fulltext_search.json")
FILING_HTML = _load_text("filing_10k.html")


def _make_handler(calls: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=TICKERS)
        if "/submissions/" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        if "/xbrl/companyfacts/" in url:
            return httpx.Response(200, json=COMPANYFACTS)
        if "efts.sec.gov" in url:
            return httpx.Response(200, json=FULLTEXT)
        if url.endswith(".htm"):
            return httpx.Response(200, text=FILING_HTML)
        return httpx.Response(404, text="not found")

    return handler


def _edgar(calls: list[httpx.Request] | None = None, user_agent: str | None = "Test Agent test@example.com"):
    transport = httpx.MockTransport(_make_handler(calls))
    async_client = httpx.AsyncClient(transport=transport)
    http = HttpClient(client=async_client)
    db = AppDB(":memory:")
    evidence = EvidenceStore(db)
    return Edgar(http, evidence, user_agent), evidence


# --- cik_for -----------------------------------------------------------------


async def test_cik_for_zero_pads_cik():
    edgar, _ = _edgar()
    cik = await edgar.cik_for("AAPL")
    assert cik == "0000320193"
    assert len(cik) == 10


async def test_cik_for_is_case_insensitive():
    edgar, _ = _edgar()
    cik = await edgar.cik_for("aapl")
    assert cik == "0000320193"


async def test_cik_for_unknown_ticker_returns_none():
    edgar, _ = _edgar()
    assert await edgar.cik_for("NOPE") is None


async def test_cik_for_missing_user_agent_returns_none_without_request():
    calls: list[httpx.Request] = []
    edgar, _ = _edgar(calls, user_agent=None)
    result = await edgar.cik_for("AAPL")
    assert result is None
    assert calls == []


# --- header present on every request ------------------------------------------


async def test_user_agent_header_present_on_every_request():
    calls: list[httpx.Request] = []
    edgar, _ = _edgar(calls)
    await edgar.filings("s1", "AAPL")
    assert calls, "expected at least one HTTP request"
    for req in calls:
        assert req.headers["user-agent"] == "Test Agent test@example.com"


# --- missing user agent is an error, not a crash -------------------------------


async def test_filings_missing_user_agent_is_error_not_crash():
    calls: list[httpx.Request] = []
    edgar, _ = _edgar(calls, user_agent=None)
    result = await edgar.filings("s1", "AAPL")
    assert result.is_error is True
    assert "SEC_USER_AGENT" in result.text
    assert calls == []


async def test_financial_facts_missing_user_agent_is_error_not_crash():
    edgar, _ = _edgar(user_agent="")
    result = await edgar.financial_facts("s1", "AAPL", ["Revenues"])
    assert result.is_error is True
    assert "SEC_USER_AGENT" in result.text


async def test_filing_passages_missing_user_agent_is_error_not_crash():
    edgar, _ = _edgar(user_agent=None)
    result = await edgar.filing_passages("s1", "0000320193-24-000005", "0000320193", "risk")
    assert result.is_error is True
    assert "SEC_USER_AGENT" in result.text


async def test_fulltext_missing_user_agent_is_error_not_crash():
    edgar, _ = _edgar(user_agent=None)
    result = await edgar.fulltext("s1", "risk factors")
    assert result.is_error is True
    assert "SEC_USER_AGENT" in result.text


# --- malformed upstream responses -----------------------------------------------


async def test_filings_200_html_response_is_error_not_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        # A 200 with an HTML body (e.g. an interstitial/error page) instead of the expected JSON.
        return httpx.Response(200, text="<html><body>Service unavailable, try again later</body></html>")

    transport = httpx.MockTransport(handler)
    http = HttpClient(client=httpx.AsyncClient(transport=transport))
    evidence = EvidenceStore(AppDB(":memory:"))
    edgar = Edgar(http, evidence, "Test Agent test@example.com")

    # A raw CIK skips the ticker lookup, so this exercises the submissions fetch directly.
    result = await edgar.filings("s1", "0000320193")

    assert result.is_error is True
    assert result.sources == []


# --- filings -------------------------------------------------------------------


async def test_filings_filters_by_form_and_prefixes_lines_with_evidence_id():
    edgar, evidence = _edgar()
    result = await edgar.filings("s1", "AAPL", forms=("10-K",), limit=10)
    assert result.is_error is False
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.tier == 1
    assert source.untrusted is False
    assert f"[{source.id}]" in result.text
    assert "10-K" in result.text
    assert "8-K" not in result.text  # filtered out


async def test_filings_accepts_a_raw_cik():
    edgar, _ = _edgar()
    result = await edgar.filings("s1", "0000320193", forms=("10-K", "10-Q", "8-K"))
    assert result.is_error is False
    assert len(result.sources) == 3


async def test_filings_unknown_ticker_is_error():
    edgar, _ = _edgar()
    result = await edgar.filings("s1", "NOPE")
    assert result.is_error is True


# --- financial_facts ------------------------------------------------------------


async def test_financial_facts_returns_latest_8_periods_with_form_and_filed():
    edgar, evidence = _edgar()
    result = await edgar.financial_facts("s1", "AAPL", ["Revenues"], periods=8)
    assert result.is_error is False
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.tier == 1
    assert source.untrusted is False

    lines = [ln for ln in result.text.splitlines() if ln.strip()]
    assert len(lines) == 8
    # newest end date first
    assert "end=2023-12-30" in lines[0]
    assert "end=2022-03-31" in lines[-1]  # 2021-12-31 dropped, only 8 kept
    for ln in lines:
        assert f"[{source.id}]" in ln
        assert "form=" in ln
        assert "filed=" in ln
    # 8-K entries are not preferred once 10-K/10-Q exist for that period
    assert "form=8-K" not in result.text


async def test_financial_facts_dedupes_same_end_and_fp_keeping_latest_filed():
    edgar, _ = _edgar()
    result = await edgar.financial_facts("s1", "AAPL", ["Revenues"], periods=8)
    # (end=2023-06-30, fp=Q3) appears twice in the fixture (val 700 filed 2023-08-01,
    # val 705 filed 2023-08-10) - only one line for that period, the later filing.
    matching = [ln for ln in result.text.splitlines() if "end=2023-06-30" in ln]
    assert len(matching) == 1
    assert "val=705" in matching[0]
    assert "filed=2023-08-10" in matching[0]


async def test_financial_facts_unknown_concept_notes_no_data():
    edgar, _ = _edgar()
    result = await edgar.financial_facts("s1", "AAPL", ["NotAConcept"], periods=8)
    assert result.is_error is False
    assert "NotAConcept" in result.text
    assert result.sources == []


# --- filing_passages -------------------------------------------------------------


async def test_filing_passages_picks_item_1a_chunk_for_risk_query():
    edgar, evidence = _edgar()
    result = await edgar.filing_passages(
        "s1", "0000320193-24-000005", "0000320193", "risk", k=3
    )
    assert result.is_error is False
    assert result.sources, "expected at least one passage"
    top_source = result.sources[0]
    assert top_source.tier == 1
    assert top_source.untrusted is True
    top_snippet = top_source.snippet.lower()
    assert "risk factors" in top_snippet or "subject to a variety of risks" in top_snippet
    assert "<untrusted source=" in result.text
    assert f'source="{top_source.id}"' in result.text


async def test_filing_passages_unknown_accession_is_error():
    edgar, _ = _edgar()
    result = await edgar.filing_passages("s1", "0000000000-00-000000", "0000320193", "risk")
    assert result.is_error is True


# --- fulltext ----------------------------------------------------------------------


async def test_fulltext_returns_hits_as_evidence():
    edgar, _ = _edgar()
    result = await edgar.fulltext("s1", "risk factors", forms=["10-K"], limit=10)
    assert result.is_error is False
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.untrusted is False
    assert source.tier == 1
    assert f"[{source.id}]" in result.text
    assert "0000320193-24-000005" in result.text
