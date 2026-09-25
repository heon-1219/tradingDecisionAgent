# Agentic Trading Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the English-edition financial-product agent into a chat-based trading assistant. It researches across local and external sources, explains a strategy, and places Alpaca paper orders only after the user confirms a server-issued ticket.

**Architecture:**
- Three new packages sit beside the existing QA pipeline:
  - `research/`: federated connectors plus an evidence store.
  - `trading/`: a paper-only broker client, market data, metrics, a pure risk engine, tickets, and execution.
  - `chat/`: sessions, a toolbox, the agent loop and SSE.
- A vanilla-JS web app in the style of AI Studio talks to them over `/api/*`.
- The LLM can only read and propose. A server endpoint that the Confirm button calls is the only path to the broker.

**Tech stack:** Python 3.12+ (the dev venv is 3.14), FastAPI, httpx (async), pydantic v2, SQLite (stdlib), and the existing `openai` SDK to reach Gemini's OpenAI-compatible endpoint. Tests use pytest + pytest-asyncio + `httpx.MockTransport`. The UI is vanilla ES modules with vendored `streaming-markdown` and `DOMPurify`.

**Spec:** `docs/superpowers/specs/2026-09-26-trading-agent-design.md` (research: `…-research.md`).

**How to read the tasks.** Each task names its files, the interfaces it consumes and produces, and the tests that must exist, with real test code for safety-critical logic. Implementers work TDD-first: write the listed tests, watch them fail, implement until they pass, then run the full suite. Code bodies are left to the implementer except where a snippet pins down exact behaviour.

## Global Constraints
- Run everything from `fin-agent-en/`. Python: `../fin-agent/.venv/Scripts/python.exe`. Tests: `../fin-agent/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`. The full suite takes about 2.5 minutes; mark network or live tests with `@pytest.mark.live` and skip them by default.
- The shell exports a stale `GEMINI_API_KEY` that overrides `.env`. Prefix live commands with `env -u GEMINI_API_KEY -u GOOGLE_API_KEY`.
- **Paper only.** The trading host is the constant `https://paper-api.alpaca.markets`. No setting, environment variable or argument may change it. Code that builds a trading URL asserts the host.
- **The LLM never gets an order-placing tool.** Only `POST /api/tickets/{id}/confirm` reaches `broker.submit_order`.
- **No blind order retries.** After a timeout, a 5xx, or 422 "client_order_id must be unique", reconcile with `GET /v2/orders:by_client_order_id`.
- **Money is `Decimal`** (never `float`) in `trading/`. Alpaca sends numbers as strings.
- **Secrets never reach prompts, logs, the ledger or API responses.** Redact `APCA-*` headers and keys in every log and ledger record.
- **External text is untrusted.** It is sanitized and wrapped as `<untrusted source="E7">…</untrusted>` before the model sees it.
- **User-facing text is English first, then Korean.** The UI has an EN/한국어 toggle; answers follow the question's language (`finagent.i18n.detect_lang`).
- **Keep the existing `/answer`, `/answer/stream` and `/ui` contracts, and all existing tests, green.**
- **Commits only if the user has enabled them.** Otherwise leave changes staged-free and report them.
- New dependencies: none required. `httpx`, `pydantic` and `openai` are already present. Vendor front-end libraries as files under `web/app/vendor/` rather than loading them from a CDN.

## File structure (new)

```
src/finagent/store/appdb.py           SQLite app DB (sessions, messages, mandates, tickets, orders, ledger, evidence, cache)
src/finagent/research/http.py         shared async HTTP: per-host rate limits, GET retries, timeouts, redaction
src/finagent/research/evidence.py     SourceRecord, ResearchResult, EvidenceStore, sanitize/wrap untrusted text
src/finagent/research/edgar.py        SEC EDGAR connector
src/finagent/research/websearch.py    Tavily search/extract + Jina Reader fallback
src/finagent/research/grounding.py    Gemini native google_search grounding (optional, never cached)
src/finagent/trading/models.py        Mandate, OrderLeg, snapshots, RiskReason/RiskDecision, Ticket, enums
src/finagent/trading/broker.py        AlpacaPaper trading client (paper host hard-coded)
src/finagent/trading/marketdata.py    AlpacaData: snapshots, bars, news, movers
src/finagent/trading/metrics.py       deterministic price metrics
src/finagent/trading/risk.py          pure risk rules
src/finagent/trading/proposals.py     TicketStore (lifecycle, hash, TTL, supersede)
src/finagent/trading/ledger.py        append-only audit ledger
src/finagent/trading/execution.py     Executor: confirm/reject/refresh, reconcile, poll, kill switch
src/finagent/chat/sessions.py         SessionStore
src/finagent/chat/toolbox.py          ChatToolbox: tool specs + dispatch, ToolOutput
src/finagent/chat/prompts.py          chat system prompt
src/finagent/chat/grounding_check.py  numeric grounding validator
src/finagent/chat/agent.py            ChatAgent turn loop with event emission
src/finagent/api_chat.py              APIRouter for /api/*
web/app/index.html, app.css, js/*.js, vendor/*   the new UI
scripts/check_alpaca.py, scripts/check_research.py, scripts/eval_scenarios.py
tests/test_appdb.py, test_evidence.py, test_edgar.py, test_websearch.py, test_grounding.py, test_broker.py,
tests/test_marketdata.py, test_metrics.py, test_risk.py, test_proposals.py, test_execution.py, test_ledger.py,
tests/test_chat_toolbox.py, test_grounding_check.py, test_chat_agent.py, test_api_chat.py, test_federation.py
```

---

## Phase 0: Finish the English edition and verify the federated core

### Task 0.1: Integrate the translation work and align coupled strings
**Files:** modify `src/finagent/ontology/loader.py`, `src/finagent/agent/prompts.py` (the `값:` marker becomes `values:`); `scripts/eval_truth.py`, `scripts/eval_run.py`, `scripts/audit_vector.py` (match both the Korean and the new English answer/trace strings); `web/console.html` (step filters match both `라운드 N:` and the new English round lines); `docs/api_spec.md` (labels match the code); `scripts/fetch_constituents_naver.py` (`write_readme()` writes the bilingual README); `Wolves_of_Walnut_Street_Proposal.html` (render 장세헌 as "David Seheon Chang"), then regenerate the PDF with headless Edge.
- [ ] New English strings the code now emits (the Korean variants still exist for Korean questions, and older recorded results are Korean):
  - Round trace lines:
    - `Round {n}: {tool}({args}) → result {len} chars, {rows} rows`
    - `Round {n}: … → error: …`
    - `Round {n}: text reply without a tool call`
    - `Round {n}: final_answer(status=…)`
    - `Round 0 routing: graph=ran(…)/skipped(…) / vector=… / document=…`
    - The old Korean form was `라운드 N: …`.
  - Internal-error trace: `Internal error: {Type}: …` (was `내부 오류`). Validation trace: `Input validation: question parameter missing`.
  - Fallback answer (English): `This cannot be confirmed with the provided data. A system problem prevented an answer from being generated ({reason}). Please rephrase…`. The Korean one still contains `시스템 처리 중 문제가 발생`.
  - Evidence labels:
    - `[Evidence N] …`
    - `External documents` (was `외부 문서`)
    - `Knowledge graph`, `Product search`, `Entity resolution`, `Schema`
    - `· base date …`, `· N rows`
    - `Refs:`
    - `[Answer evidence summary]`
    - `No data referenced`
  - Decline and clarify wording:
    - `This cannot be confirmed with the provided data.`
    - `Could you tell me which condition you mean?`
    - `A product exactly matching '…' cannot be confirmed in the provided data.`
    - `Is one of these the product you are looking for?`
    - `Could you check the product name you are looking for?`
    - The English decline markers are `cannot be confirmed`, `could not be found`, `does not exist`, `is not available`, `not provided`.
- [ ] Update every matcher so it accepts both the Korean and the English string:
  - `scripts/eval_truth.py`: `DECLINE`, `DECLINE_RX`, `CLARIFY`, `ERROR_MARKERS`, and the `<요청 실패` marker.
  - `scripts/eval_run.py`: `classify`.
  - `scripts/audit_vector.py`: the `_ROUND`/`_NOTOOL` regexes and the `'문서'` evidence check.
  - `web/console.html`:
    - the step filter becomes `/^(?:라운드|Round) \d+: (sql_query|graph_|search_|resolve_|table_info)/`
    - "fix" words add `corrected|stated|warning`
    - "error" words add `error|timed out|exceeded|rejected|failed|refused`
  - Add a unit test in `tests/test_eval_truth_markers.py` feeding one Korean and one English sample of each kind.
- [ ] Fix English mention extraction in `src/finagent/kb/entity.py` (`extract_product_mentions`). Today "Which domestic ETF has the largest net assets…" yields `['Which domestic ETF']`, "Which ETF tracks the KOSPI 200 index?" yields `['Which ETF']`, and "Will KODEX 200 rise next year?" yields `['KODEX 200 rise next year']`. These become false "product does not exist" clarifications.
  - Rule for English questions (`detect_lang == "en"`): a mention never starts with a question or stop word (Which, What, How, Will, Does, Is, Are, Do, Show, List, Tell, Give, Find, Compare, the, a, an), and ends before the first lower-case English word that isn't part of a known product-name pattern.
  - Tests: the three examples above yield `[]`, `[]` and `['KODEX 200']`. `"What is the expense ratio of TIGER 미국S&P500?"` yields `['TIGER 미국S&P500']`. Every existing Korean mention test stays unchanged.
  - Then add an orchestrator test: an English question naming an existing product is not rewritten into a clarification.
- [ ] Change the `값:` marker to `values:` in `schema_card` and in the `prompts.py` check together. Assert in `tests/test_ontology.py` that a card line reads `(values:A/B)`.
- [ ] Run the full suite. Expected: all pass (≥ 240 tests).

### Task 0.2: English answer key
**Files:** create `data/eval/groundtruth_en.jsonl`; modify `scripts/eval_truth.py` so it recognizes English decline and clarify answers (`cannot be confirmed`, `could not find`, `not available in the provided data`, `did you mean`).
- [ ] Extract the English G-001…G-054 questions that `web/console.html` already carries into `groundtruth_en.jsonl`. Copy every other field from `groundtruth.jsonl` and replace only `question`.
- [ ] Test: both files have the same ids, `truth_sql` and `check` values (`tests/test_eval_files.py`).
- [ ] Test: `eval_truth.score()` gives an English decline the same verdict as the Korean one.

### Task 0.3: Federation tests (every source contributes)
**Files:** create `tests/test_federation.py`. It uses the real store (`@pytest.mark.slow`) and `StubLLM` scripts.
- [ ] For each source family, one test calls the tool through `ToolRegistry.execute` and asserts that non-empty evidence of that kind lands in the retrieved context:

| Family | Tool call |
|---|---|
| SQL | `sql_query` on `v_etf_gl` |
| Graph | `graph_neighbors('삼성전자','constituentOf')` |
| Lexical and vector | `search_products(mode='hybrid')` |
| Docs | `search_documents('ETF 위험요인')` |

- [ ] Add one cross-source test: `Orchestrator.answer` with a stub script that calls the graph and then SQL. Assert that `retrieved_context` cites both.
- [ ] Add one test per routing prefetch family (holdings → graph, descriptive → docs), with one English and one Korean question each.

### Task 0.4: Evaluate on Gemini, fix, and document
**Files:** `data/eval/truth_results_gemini_ko.jsonl` and `…_en.jsonl` (outputs); `README.md` and `docs/api_spec.md` (a results section).
- [ ] Start the API on port 8000, then run `env -u GEMINI_API_KEY -u GOOGLE_API_KEY … scripts/eval_truth.py --file data/eval/groundtruth.jsonl --out data/eval/truth_results_gemini_ko.jsonl --delay 4`, then the same with `_en`.
- [ ] Group failures by source family and by language. Fix the top causes in prompts, routing or tools, each with a regression test. Re-run only the failed ids with `--ids`.
- [ ] Update the README results table in both languages: the Gemini KO/EN accuracy and refusals next to the historical HyperCLOVA X figures (47/54, 19/19), with the date and model.

---

## Phase 1: Research connectors and market data

### Task 1.1: App DB, HTTP client and evidence store
**Files:** create `src/finagent/store/__init__.py`, `store/appdb.py`, `research/__init__.py`, `research/http.py`, `research/evidence.py`; add settings to `config.py`; tests `tests/test_appdb.py` and `tests/test_evidence.py`.

**New settings** (config.py):
- `alpaca_api_key_id: str | None`, `alpaca_api_secret_key: str | None`
- `tavily_api_key: str | None`
- `sec_user_agent: str | None`
- `grounding_model: str = "gemini-3.8-flash"`
- `trading_enabled: bool = True`
- `app_db_path: Path = DATA_DIR / "store" / "app.db"`
- `chat_time_budget_sec: float = 90.0`, `chat_max_rounds: int = 8`, `ticket_ttl_sec: int = 300`

**Interfaces produced:**
```python
# store/appdb.py
class AppDB:
    def __init__(self, path: Path | str): ...          # ":memory:" allowed; WAL; creates schema
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor: ...   # serialized by a lock
    def query(self, sql: str, params: tuple = ()) -> list[dict]: ...
# tables: sessions, messages, mandates, tickets, orders, ledger, evidence, http_cache

# research/http.py
class HttpClient:
    def __init__(self, client: httpx.AsyncClient | None = None, rates: dict[str, float] | None = None): ...
    async def get_json(self, url: str, *, params=None, headers=None, timeout: float = 20.0) -> Any: ...
    async def get_text(self, url: str, *, params=None, headers=None, timeout: float = 20.0) -> str: ...
    async def post_json(self, url: str, *, json=None, headers=None, timeout: float = 20.0) -> Any: ...
# GETs retry 429/5xx up to 2 times with backoff; POSTs are never retried.
# Per-host token bucket (rates in req/s; defaults: sec.gov 5, api.tavily.com 1.5, r.jina.ai 0.3, alpaca 3).
class HttpError(Exception): status: int; body: str

# research/evidence.py
@dataclass
class SourceRecord:
    id: str; session_id: str; connector: str; title: str; url: str | None; publisher: str | None
    tier: int; published_at: str | None; retrieved_at: str; snippet: str; content_sha256: str; untrusted: bool
@dataclass
class ResearchResult:
    text: str; sources: list[SourceRecord]; is_error: bool = False
def sanitize_untrusted(text: str, max_chars: int = 4000) -> str   # strip tags/hidden text/zero-width chars, collapse ws, cap
def wrap_untrusted(source_id: str, text: str) -> str               # '<untrusted source="E3">…</untrusted>'
class EvidenceStore:
    def __init__(self, db: AppDB): ...
    def add(self, session_id: str, *, connector, title, url, publisher, tier, published_at, snippet, untrusted) -> SourceRecord
        # ids E1, E2, … per session; the same url+sha256 within a session returns the existing record
    def get(self, session_id: str, source_id: str) -> SourceRecord | None
    def list(self, session_id: str) -> list[SourceRecord]
    def cache_get(self, key: str, ttl_sec: float) -> Any | None
    def cache_put(self, key: str, payload: Any) -> None
```

- [ ] **Tests (test_evidence.py):**
  - `sanitize_untrusted` removes `<script>`, `display:none` text, zero-width characters and HTML comments, and caps the length.
  - `wrap_untrusted` escapes a closing tag that appears inside the text.
  - `add` numbers ids per session and dedupes.
  - The cache expires by TTL (inject a clock).
- [ ] **Tests (test_appdb.py):** the schema is created twice without error, and concurrent `execute` calls through `asyncio.to_thread` don't corrupt data.
- [ ] **HttpClient tests** with `MockTransport`:
  - 429 then 200 on GET is retried.
  - POST 500 is not retried.
  - The rate limiter spaces calls (inject a sleep).
  - The `APCA-API-SECRET-KEY` header value never appears in log records (use caplog).

### Task 1.2: SEC EDGAR connector
**Files:** `research/edgar.py`; `tests/test_edgar.py` with fixtures under `tests/fixtures/edgar/`.

**Interface:**
```python
class Edgar:
    def __init__(self, http: HttpClient, evidence: EvidenceStore, user_agent: str): ...
    async def cik_for(self, ticker: str) -> str | None        # company_tickers.json, cached 24 h
    async def filings(self, session_id, ticker_or_cik, forms=("10-K","10-Q","8-K"), limit=10) -> ResearchResult
    async def financial_facts(self, session_id, ticker_or_cik, concepts: list[str], periods=8) -> ResearchResult
    async def filing_passages(self, session_id, accession_no, cik, query, sections=None, k=5) -> ResearchResult
    async def fulltext(self, session_id, query, forms=None, start=None, end=None, limit=10) -> ResearchResult
```
- Records are tier 1, `untrusted=False` for XBRL facts and `True` for filing text.
- Every request sends `User-Agent: <sec_user_agent>`. If it is missing, return `ResearchResult(is_error=True)` explaining that `SEC_USER_AGENT` is required.
- Passages are split into structural chunks (Item headings; tables kept whole) and ranked with BM25 from `rank_bm25`.

- [ ] **Tests:**
  - CIK zero-padding.
  - `financial_facts` returns the latest 8 periods of `Revenues` with the form and filed date.
  - The header is present on every request.
  - A missing user agent is an error, not a crash.
  - Passages from a small HTML 10-K fixture pick the Item 1A chunk for the query "risk".

### Task 1.3: Web search connector
**Files:** `research/websearch.py`; `tests/test_websearch.py`.
```python
class WebSearch:
    def __init__(self, http: HttpClient, evidence: EvidenceStore, tavily_key: str | None): ...
    async def search(self, session_id, query, topic="general", start_date=None, end_date=None,
                     include_domains=None, max_results=5) -> ResearchResult     # Tavily /search
    async def fetch(self, session_id, url, focus_query=None, max_chars=6000) -> ResearchResult  # Tavily /extract, else Jina r.jina.ai
```
- Tier by domain: sec.gov / investor-relations / exchanges = 1; reuters / bloomberg / wsj / ft / benzinga = 2; other media = 3; blogs, forums and reddit = 4. All results are `untrusted=True` and wrapped.
- [ ] **Tests:**
  - Without a key, `search` returns an error result mentioning `TAVILY_API_KEY`.
  - Tavily JSON is mapped to records with `published_date`.
  - When the Tavily key is missing, `fetch` falls back to Jina.
  - Hidden text is stripped from the fetched page.

### Task 1.4: Gemini grounding (optional)
**Files:** `research/grounding.py`; `tests/test_grounding.py`.
```python
class GroundedWeb:
    def __init__(self, http: HttpClient, api_key: str | None, model: str): ...
    async def answer(self, question: str) -> GroundedAnswer
@dataclass
class GroundedAnswer:
    text: str; sources: list[dict]; search_suggestions_html: str | None; queries: list[str]; error: str | None = None
```
- Calls `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` with header `x-goog-api-key`, body `{"contents":[{"role":"user","parts":[{"text": question}]}], "tools":[{"google_search":{}}]}`.
- Parses `candidates[0].groundingMetadata` fields `groundingChunks[].web{uri,title}`, `webSearchQueries` and `searchEntryPoint.renderedContent`.
- **Never** writes to the evidence cache. The toolbox registers the sources as non-cached records whose `connector='grounded_web'`. The UI must render `search_suggestions_html` unmodified.
- [ ] **Tests:**
  - Parsing a recorded response.
  - A 403/400 "not available on free tier" becomes a friendly error.
  - Nothing is written to `http_cache`.

### Task 1.5: Alpaca paper broker and market data
**Files:** `trading/__init__.py`, `trading/models.py` (the snapshot types only here; the rest come in 2.1), `trading/broker.py`, `trading/marketdata.py`; tests `tests/test_broker.py` and `tests/test_marketdata.py`.
```python
PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"
class BrokerError(Exception): status: int; code: int | None; message: str
class OrderUncertain(BrokerError): ...   # timeout/5xx on submit: the order MAY exist
class AlpacaPaper:
    def __init__(self, key_id: str, secret: str, client: httpx.AsyncClient | None = None): ...
    async def account(self) -> AccountSnapshot
    async def positions(self) -> dict[str, PositionSnapshot]
    async def orders(self, status="open", limit=50, symbols: list[str] | None = None) -> list[OrderSnapshot]
    async def asset(self, symbol: str) -> AssetInfo
    async def clock(self) -> ClockInfo
    async def submit_order(self, leg: OrderLeg, client_order_id: str) -> OrderSnapshot   # single attempt
    async def get_order(self, order_id: str) -> OrderSnapshot
    async def get_order_by_client_id(self, client_order_id: str) -> OrderSnapshot | None
    async def cancel_order(self, order_id: str) -> None
    async def set_suspend_trade(self, suspended: bool) -> None
class AlpacaData:
    def __init__(self, key_id: str, secret: str, http: HttpClient): ...
    async def snapshots(self, symbols: list[str]) -> dict[str, Quote]          # feed=iex; Quote has bid, ask, last, as_of, feed
    async def bars(self, symbol, timeframe="1Day", start=None, end=None, limit=500) -> list[Bar]  # adjustment=all; sip only if end >= 15 min old
    async def news(self, session_id, evidence, symbols=None, start=None, limit=10) -> ResearchResult  # tier 2, untrusted
    async def movers(self, top=10) -> dict; async def most_actives(self, top=10) -> list[dict]
```

**Snapshot models** (pydantic, Decimal):
- `AccountSnapshot` — status, equity, cash, buying_power, non_marginable_buying_power, last_equity, trading_blocked, account_blocked, trade_suspended_by_user, shorting_enabled.
- `PositionSnapshot` — symbol, qty, qty_available, avg_entry_price, current_price, market_value, unrealized_pl, unrealized_plpc, side.
- `OrderSnapshot` — id, client_order_id, symbol, side, qty, notional, type, time_in_force, limit_price, status, filled_qty, filled_avg_price, submitted_at, updated_at.
- `AssetInfo` — symbol, cls, status, tradable, fractionable, shortable, marginable, borrow_status.
- `ClockInfo` — is_open, next_open, next_close, timestamp.
- `Quote` and `Bar`.

- [ ] **Paper-guard test:**
  ```python
  def test_trading_host_is_paper_only():
      b = AlpacaPaper("k", "s")
      assert str(b._client.base_url).rstrip("/") == "https://paper-api.alpaca.markets"
      with pytest.raises(ValueError):
          AlpacaPaper("k", "s", client=httpx.AsyncClient(base_url="https://api.alpaca.markets"))
  ```
- [ ] **Submit tests:**
  - Submit sends `client_order_id` and serializes Decimals as strings.
  - A timeout raises `OrderUncertain`.
  - 422 "client_order_id must be unique" raises `BrokerError(code=40010001)`.
  - 403 insufficient buying power maps to `BrokerError` with a message.
- [ ] **Parsing tests:**
  - Account strings parse to Decimal.
  - The removed PDT fields are not required.
- [ ] **Market-data tests:**
  - Snapshots map `latestQuote.ap`/`bp` and `latestTrade.p`.
  - Bars request `feed=iex` when `end` is recent.
  - News records carry `created_at` as `published_at`.

### Task 1.6: Metrics
**Files:** `trading/metrics.py`; `tests/test_metrics.py`.
```python
def simple_returns(closes: list[Decimal]) -> list[Decimal]
def total_return(closes) -> Decimal
def annualized_volatility(closes, periods_per_year=252) -> Decimal
def max_drawdown(closes) -> Decimal            # negative fraction
def sma(closes, n: int) -> Decimal | None
def correlation(a: list[Decimal], b: list[Decimal]) -> Decimal | None
def summarize(bars: list[Bar]) -> dict         # {'last','return_1m','return_3m','return_1y','vol_1y','max_dd_1y','sma_50','sma_200'} as strings
```
- [ ] **Tests:** hand-computed values on a 6-point series; `max_drawdown([100,120,90,130]) == Decimal("-0.25")`; `sma` returns None when there are fewer than n points; correlation of identical series is 1.

---

## Phase 2: Trading core

### Task 2.1: Trading models
**Files:** `trading/models.py` (add to it); `tests/test_trading_models.py`.
```python
class Mandate(BaseModel):
    direction_text: str = ""
    budget_usd: Decimal | None = None
    universe: Literal["us_equity"] = "us_equity"
    allow_symbols: list[str] = []
    deny_symbols: list[str] = []
    max_position_pct: Decimal = Decimal("20")
    max_positions: int = 8
    cash_buffer_pct: Decimal = Decimal("5")
    horizon: str = "weeks to months"
    order_types: list[Literal["market", "limit"]] = ["limit", "market"]
    max_orders_per_day: int = 10
    max_order_usd: Decimal | None = None
    price_band_pct: Decimal = Decimal("3")
    allow_fractional: bool = True
    version: int = 1
class OrderLeg(BaseModel):
    symbol: str; side: Literal["buy", "sell"]; qty: Decimal | None = None; notional: Decimal | None = None
    type: Literal["market", "limit"] = "limit"; time_in_force: Literal["day", "gtc"] = "day"
    limit_price: Decimal | None = None
    # validator: uppercase symbol; exactly one of qty/notional; positive values
class RiskReason(BaseModel): code: str; message: str; severity: Literal["info", "warn", "block"]
class RiskDecision(BaseModel):
    index: int; verdict: Literal["accept", "clamp", "reject"]; leg: OrderLeg; original: OrderLeg
    reasons: list[RiskReason]; est_price: Decimal | None; est_cost_usd: Decimal | None
class MarketContext(BaseModel):
    account: AccountSnapshot; positions: dict[str, PositionSnapshot]; open_orders: list[OrderSnapshot]
    assets: dict[str, AssetInfo]; quotes: dict[str, Quote]; clock: ClockInfo; orders_today: int
class TicketStatus(str, Enum):
    proposed = "proposed"; submitting = "submitting"; submitted = "submitted"; rejected_by_user = "rejected_by_user"
    expired = "expired"; superseded = "superseded"; failed = "failed"
class Ticket(BaseModel):
    id: str; session_id: str; status: TicketStatus; legs: list[OrderLeg]; decisions: list[RiskDecision]
    thesis: str; risks: str; invalidation: str; evidence_ids: list[str]; mandate_version: int
    quote_as_of: datetime; created_at: datetime; expires_at: datetime; hash: str
    client_order_ids: list[str]; broker_order_ids: list[str | None] = []
```
- [ ] **Tests:** `OrderLeg` rejects both qty and notional, and neither; it uppercases the symbol; `Mandate` round-trips through JSON.

### Task 2.2: Risk engine
**Files:** `trading/risk.py`; `tests/test_risk.py`.
```python
def evaluate(legs: list[OrderLeg], ctx: MarketContext, mandate: Mandate) -> list[RiskDecision]
def overall(decisions: list[RiskDecision]) -> Literal["ok", "warnings", "blocked"]
MARKET_SLIPPAGE_PCT = Decimal("1.0")   # buffer on ask for market-buy cost estimates
```

**Reason codes and rules** (in evaluation order):

| Code | Verdict | Condition |
|---|---|---|
| `account_blocked` | reject | trading_blocked / account_blocked / trade_suspended_by_user |
| `asset_not_tradable` | reject | not tradable, not active, or class not us_equity |
| `denied_symbol` | reject | in `deny_symbols`, or `allow_symbols` is set and the symbol is absent |
| `order_type_not_allowed` | reject | type not in `mandate.order_types` |
| `notional_needs_market_day` | reject | notional with limit, or with gtc |
| `fractional_not_allowed` | reject | fractional qty and the asset is not fractionable, not day, or `mandate.allow_fractional` is False |
| `limit_required` | reject | limit order without `limit_price` |
| `bad_tick` | clamp | limit price rounded down (buy) or up (sell) to 0.01 at ≥ $1, else 0.0001 |
| `price_band` | reject | limit more than `price_band_pct` from the reference price (last trade, else mid) |
| `no_quote` | reject | no quote for the symbol |
| `short_sell` | reject | sell qty > qty_available (or no position) |
| `wash_trade` | reject | an open order on the opposite side for the same symbol |
| `max_order_usd` | clamp | shrink qty/notional to the cap (floor; for whole-share assets floor to an integer) |
| `max_position_pct` | clamp | post-trade position value > pct × equity → shrink |
| `max_positions` | reject | the buy would open a new position beyond the limit |
| `buying_power` | clamp | cost > buying_power × (1 − cash_buffer_pct/100) → shrink; reject if nothing is left |
| `orders_per_day` | reject | `ctx.orders_today + legs accepted so far ≥ max_orders_per_day` |
| `market_closed` | warn | clock.is_open False: "queues until next open" |
| `large_order` | warn | est_cost > 10% of equity |

- `est_price`: limit price if set; otherwise ask × (1 + MARKET_SLIPPAGE_PCT/100) for buys and bid for sells.
- `est_cost_usd`: est_price × qty, or notional.
- Legs are evaluated in order, and the account and position effects of earlier accepted legs carry into later ones.

- [ ] **Tests:** a table-driven test with a fixture context and at least one case per code, including these exact cases:
  ```python
  def test_buy_clamped_to_buying_power(ctx, mandate):   # buying_power 1000, cash buffer 5% -> 950 usable
      d = evaluate([OrderLeg(symbol="VOO", side="buy", qty=Decimal("5"), type="limit", limit_price=Decimal("500"))], ctx, mandate)[0]
      assert d.verdict == "clamp" and d.leg.qty == Decimal("1") and "buying_power" in {r.code for r in d.reasons}
  def test_sell_more_than_held_is_short(ctx, mandate):
      d = evaluate([OrderLeg(symbol="VOO", side="sell", qty=Decimal("3"), type="market")], ctx_with_position(qty=2), mandate)[0]
      assert d.verdict == "reject" and d.reasons[0].code == "short_sell"
  def test_two_legs_share_buying_power(ctx, mandate): ...   # second leg is clamped by what the first consumed
  ```

### Task 2.3: Ticket store and ledger
**Files:** `trading/proposals.py`, `trading/ledger.py`; `tests/test_proposals.py` and `tests/test_ledger.py`.
```python
def canonical_hash(legs: list[OrderLeg], decisions: list[RiskDecision], ticket_id: str) -> str  # sha256 of sorted-key JSON
class TicketStore:
    def __init__(self, db: AppDB, clock: Callable[[], datetime] = utcnow): ...
    def create(self, session_id, legs, decisions, *, thesis, risks, invalidation, evidence_ids,
               mandate_version, quote_as_of, ttl_sec=300) -> Ticket      # supersedes the session's active ticket
    def get(self, ticket_id: str) -> Ticket | None
    def active(self, session_id: str) -> Ticket | None                 # status proposed and not expired
    def transition(self, ticket_id, expect: set[TicketStatus], to: TicketStatus, **fields) -> Ticket  # atomic compare-and-set
    def expire_stale(self) -> int
class Ledger:
    def __init__(self, db: AppDB): ...
    def append(self, session_id: str, kind: str, payload: dict) -> int   # redacts keys matching /secret|key|token/i
    def export_jsonl(self, session_id: str) -> Iterator[str]
```
- Ticket ids look like `T` + 10 base32 characters. `client_order_ids = [f"fa-{ticket_id}-{i}"]`.
- [ ] **Tests:**
  - Creating supersedes the previous active ticket.
  - An expired ticket is not active.
  - `transition` with a wrong `expect` raises and leaves the state unchanged.
  - The hash changes when any leg field changes.
  - The ledger redacts `{"alpaca_api_secret_key": "x"}`.
  - Export yields valid JSON lines in order.

### Task 2.4: Executor
**Files:** `trading/execution.py`; `tests/test_execution.py` with a `FakeBroker` / `FakeData` in `tests/fakes_trading.py`.
```python
@dataclass
class ConfirmResult:
    status: Literal["submitted", "requote", "rejected", "disabled", "expired", "not_found", "error"]
    ticket: Ticket | None; orders: list[OrderSnapshot]; message: str; new_ticket: Ticket | None = None
class Executor:
    def __init__(self, broker, data, tickets: TicketStore, ledger: Ledger, settings,
                 sleep=asyncio.sleep, now=utcnow): ...
    async def build_context(self, symbols: list[str]) -> MarketContext
    async def confirm(self, ticket_id: str, session_id: str, ticket_hash: str) -> ConfirmResult
    async def reject(self, ticket_id: str, session_id: str) -> ConfirmResult
    async def refresh(self, ticket_id: str, session_id: str) -> ConfirmResult   # re-quote -> new ticket
    async def kill_switch(self, suspend_broker: bool = False) -> None           # trading_enabled False (+ broker suspend, cancel open)
REQUOTE_DRIFT_PCT = Decimal("1.0")
```

**`confirm` algorithm:**
1. If `settings.trading_enabled` is false, return `disabled`.
2. Load the ticket. If the session doesn't match, return `not_found`. If `now > expires_at`, transition to `expired` and return `expired`. If the hash doesn't match, return `rejected` ("ticket changed").
3. Rebuild the context and re-run `evaluate` on the ticket's legs. If any verdict or leg changes, or any price moves more than `REQUOTE_DRIFT_PCT` from the ticket's estimate, create a new ticket (superseding this one) and return `requote` with `new_ticket`.
4. Transition to `submitting`. For each leg, call `submit_order(leg, cid)`. On `OrderUncertain` or `BrokerError(code=40010001)`, call `get_order_by_client_id(cid)`: use the order if found, otherwise record an error for that leg. Other `BrokerError`s record an error for that leg.
5. Poll each order up to 5 times, 2 s apart (`sleep` injected), until it reaches a final state or `accepted`/`new`.
6. Transition to `submitted` if any leg was submitted, else `failed`. Append ledger entries (`confirm_request`, `submit`, `broker_response`, `order_status`) and return.

- [ ] **Tests:**
  - Happy path.
  - Expired.
  - Wrong hash.
  - Disabled.
  - Requote on price drift.
  - Timeout then reconcile finds the order (no second submit; assert `FakeBroker.submits == 1`).
  - Duplicate `client_order_id` reconciles.
  - A partial multi-leg failure yields `submitted` with a per-leg error.
  - A confirm replay after submission returns `rejected` without a second submit.

---

## Phase 3: Chat agent and API

### Task 3.1: Sessions and the `add_assistant` hook in the LLM
**Files:** `chat/__init__.py`, `chat/sessions.py`; modify `agent/llm.py` to add `add_assistant(text)` to `StubConversation`, `OpenAIConversation` and `AnthropicConversation`; tests `tests/test_sessions.py` plus additions to `tests/test_llm_stub.py`.
```python
class SessionStore:
    def __init__(self, db: AppDB): ...
    def create(self, title: str = "New chat", lang: str = "en") -> dict      # {'id','title','created_at','lang'}
    def list(self) -> list[dict]; def get(self, session_id) -> dict | None; def rename(self, session_id, title) -> None
    def append(self, session_id, role: Literal["user", "assistant", "event"], content: str, meta: dict | None = None) -> int
    def history(self, session_id, limit: int = 12) -> list[dict]             # last N user/assistant messages, oldest first
    def get_mandate(self, session_id) -> Mandate | None; def save_mandate(self, session_id, mandate: Mandate) -> Mandate  # bumps version
```
- [ ] **Tests:** round trips; `history` excludes `event` rows and keeps order; saving a mandate bumps its version; the `add_assistant` message shape is correct for OpenAI (`{"role":"assistant","content":text}`).

### Task 3.2: Toolbox
**Files:** `chat/toolbox.py`; `tests/test_chat_toolbox.py`.
```python
@dataclass
class ToolOutput:
    text: str; sources: list[SourceRecord] = field(default_factory=list)
    events: list[dict] = field(default_factory=list); is_error: bool = False
class ChatToolbox:
    def __init__(self, *, registry: ToolRegistry, orchestrator, broker: AlpacaPaper | None, data: AlpacaData | None,
                 edgar: Edgar | None, web: WebSearch | None, grounded: GroundedWeb | None,
                 evidence: EvidenceStore, sessions: SessionStore, tickets: TicketStore, executor: Executor | None,
                 settings): ...
    def specs(self) -> list[ToolSpec]          # tools whose dependency is missing are omitted
    async def execute(self, session_id: str, name: str, args: dict) -> ToolOutput
```

**Tools** (all JSON-schema'd, with English descriptions):

| Group | Tools |
|---|---|
| Account | `get_account`, `get_positions`, `list_orders` |
| Market | `get_market_status`, `lookup_asset`, `get_quotes` (≤10 symbols), `get_price_history` + `compute_metrics` (merged as `price_metrics(symbol, lookback='1y')`, which returns `summarize()`), `get_movers` |
| Research | `search_news`, `web_search`, `fetch_source`, `sec_filings`, `sec_financial_facts`, `sec_filing_passages`, `grounded_web_answer`, `screen_us_etfs(asset_type?, keyword?, max_expense_ratio?, min_aum_usd?, sort_by='aum', limit=10)` (parameterized SQL over `v_etf_gl`; never raw SQL from the model), `ask_product_database(question)` (runs `orchestrator.answer` and returns `answer` + `retrieved_context`) |
| Knowledge | `search_products`, `graph_neighbors` (delegated to `ToolRegistry.execute`) |
| Mandate | `draft_mandate(fields...)` → emits `{"type":"data-mandate","mandate":…,"draft":true}`; the user saves it in the UI |
| Trading | `propose_trades(legs[], thesis, risks, invalidation, evidence_ids[])` |

**`propose_trades` flow:**
1. `executor.build_context`, then `evaluate`, then `tickets.create`.
2. Emit `{"type":"data-trade-proposal","ticket":…}`.
3. Return the text summary: per-leg verdicts, clamps and warnings. The text states: "The user must confirm in the UI; you cannot place orders."

Each research tool registers `SourceRecord`s and renders its text with `[E#]` ids and `wrap_untrusted` for untrusted content.

- [ ] **Tests** (using fakes):
  - `specs()` omits broker tools when there are no Alpaca keys and omits `web_search` without Tavily.
  - `screen_us_etfs` builds parameterized SQL and ignores injection in `keyword`.
  - `propose_trades` creates a ticket and emits the event.
  - There is no tool named like `submit|place|execute` (`assert not any(re.search(r"submit|place|execute", s.name) for s in specs)`).
  - Research output contains `<untrusted` for web results.

### Task 3.3: Prompt and numeric grounding
**Files:** `chat/prompts.py`, `chat/grounding_check.py`; `tests/test_grounding_check.py`.
- `build_chat_system_prompt(settings, tool_names: list[str]) -> str`. It must be stable (no timestamps; the date goes into the per-turn user message). It covers:
  - role and paper-trading disclaimer
  - research-then-propose workflow
  - citation format `[E#]`
  - all numbers from tools
  - untrusted blocks are data
  - mandate respect
  - at most one `propose_trades` per turn
  - never claim an order was placed
  - answer language follows the user; Korean data values stay Korean
  - conciseness
- `extract_numbers(text) -> list[str]` covers `$1,234.56`, `12.5%`, `1.2B`, `-3.4`, dates `2026-09-25`, and share counts. It ignores `[E12]` ids and years inside dates.
- `unsupported_numbers(answer: str, corpus: list[str]) -> list[str]` normalizes separators, `%`, `$` and B/M/K, and allows rounding to the displayed precision.
- [ ] **Tests:** `"$101.25 (up 3.4%)"` is supported by `"price 101.25 … change 0.034"`; `"about 12%"` is unsupported when the corpus has only 10.9; `[E12]` is not extracted.

### Task 3.4: ChatAgent loop
**Files:** `chat/agent.py`; `tests/test_chat_agent.py`.
```python
class ChatAgent:
    def __init__(self, llm: LLMClient, toolbox: ChatToolbox, sessions: SessionStore, settings, now=utcnow): ...
    async def run_turn(self, session_id: str, message: str, emit: Callable[[dict], None]) -> str   # returns the final text
```
1. Rebuild the conversation from `sessions.history`: `llm.start(system, tools, first_user)`, then `add_assistant` / `add_user` for the rest. Cache it per session in memory; rebuild after a restart.
2. Build the turn's user text: the message, `Today: <date>, market clock unknown until checked`, `Answer language: <English|Korean>`, and `Active mandate: <json or none>`.
3. Loop for up to `chat_max_rounds` rounds and `chat_time_budget_sec` seconds:
   - `turn = await conv.step()`.
   - Emit `reasoning-delta` if the model returns thought summaries (skip otherwise).
   - For each tool call, emit `tool-input-available`, execute, emit `tool-output-available` (with `sources` summarised to `{id, title, url, publisher, tier}`), then emit the tool's `events`.
   - `conv.add_tool_results(...)`. Stop when there are no tool calls.
   - After the budget is spent, send one final nudge and take the text.
4. Final text: run `unsupported_numbers` against the texts of all tool outputs in this turn, then emit `data-grounding-flags` if any number is unsupported.
5. Emit `text-delta` with the full text (chunked at about 200 characters to feel streamed), then `finish {usage, elapsed_ms}`. Persist the user message, the assistant message and the event summaries.
6. Errors: `error {message}` and a friendly assistant message; never raise out of `run_turn`.

- [ ] **Tests** (`StubLLM` scripts):
  - A research → propose → final turn emits the events in order and creates one ticket.
  - A second `propose_trades` in the same turn is refused by the toolbox, with an error returned to the model.
  - History is rebuilt after a new `ChatAgent` instance is created.
  - The time budget stops the loop.
  - An unsupported number produces `data-grounding-flags`.

### Task 3.5: API router and app wiring
**Files:** `api_chat.py`; modify `api.py` (include the router, serve `web/app` at `/` and `/app/*`, keep `/ui`) and `app_state.py` (build `AppDB`, `HttpClient`, connectors, broker/data when keys exist, `TicketStore`, `Ledger`, `Executor`, `SessionStore`, `ChatToolbox`, `ChatAgent`); tests `tests/test_api_chat.py`.

**Endpoints** (Pydantic request models):
- `POST /api/sessions`, `GET /api/sessions`, `GET /api/sessions/{id}` (with the last 50 messages and the active ticket)
- `POST /api/chat {session_id, message}` → `text/event-stream`; each event is `data: <json>\n\n`; keep-alive comments every 10 s
- `GET /api/portfolio` → account, positions, open orders, or `{"connected": false}`
- `GET /api/mandate/{session_id}`, `PUT /api/mandate/{session_id}`
- `POST /api/tickets/{id}/confirm {session_id, hash}`, `POST /api/tickets/{id}/reject {session_id}`, `POST /api/tickets/{id}/refresh {session_id}`
- `GET /api/settings` → `{alpaca: connected|missing|invalid, tavily: set|missing, sec_user_agent: set|missing, gemini: set|missing, trading_enabled, model, effort}` (never secrets)
- `PUT /api/settings {alpaca_key_id?, alpaca_secret?, tavily_key?, sec_user_agent?, trading_enabled?}`: validate Alpaca keys live on the paper host before saving them into `.env`, then rebuild the dependent components
- `POST /api/kill-switch {suspend_broker: bool}`
- `GET /api/ledger/{session_id}.jsonl`
- Bind to 127.0.0.1 in the documented run command.

- [ ] **Tests** (TestClient with fakes injected through `create_app(state_factory=...)`):
  - The SSE stream parses and ends with `finish`.
  - Confirm with a wrong hash returns 409.
  - Confirm with trading disabled returns 423.
  - Settings responses never contain the secret.
  - The existing `/answer` still returns the five fields.

---

## Phase 4: Web app in the style of AI Studio

### Task 4.1: Shell, tokens, i18n, SSE client, markdown
**Files:** `web/app/index.html`, `web/app/app.css`, `web/app/js/{main,i18n,api,sse,markdown,state}.js`, `web/app/vendor/{streaming-markdown.min.js,purify.min.js}`. Download the vendored files once from npm or jsDelivr; ask the user before downloading and record their versions in `web/app/vendor/VERSIONS.md`.

- **Layout:**
  - left rail: 64px, expandable to 240px, holds sessions and New chat
  - top bar: title, PAPER badge, model chip, muted tokens and latency
  - centre column: at most 760px wide
  - right panel: 340px, with tabs Portfolio / Orders / Mandate / Settings
  - composer fixed at the bottom
- **Tokens:** the dark-first palette from the UI research (`--bg:#0e0f12` …, accent `#8ab4ff`); light theme via `[data-theme=light]`; theme follows the system with a toggle; `--font: "Google Sans Flex", system-ui, "Noto Sans KR", sans-serif` (use system fonts if the web font is not vendored); `tabular-nums` on numbers.
- **i18n:** a dictionary with `en` and `ko`, English by default, stored in `localStorage` under `finagent-app.lang` (inside try/catch).
- **SSE client:** fetch POST with a ReadableStream parser (reuse the console's parser logic); Stop aborts the fetch.
- **Markdown:** streaming-markdown, then DOMPurify, then the DOM; tables and code blocks get copy buttons.
- [ ] **Verify:** `node --check` on every JS file; a script confirms every i18n key exists in both languages; the page loads with the server running (Browser pane) and shows the empty state with suggested prompts.

### Task 4.2: Conversation components
**Files:** `web/app/js/components/{message,toolcard,sources,thinking,mandate,ticket,portfolio,orders,settings,toast}.js`.

- **ToolCard:** a family icon (market / research / knowledge / account / trading) and a status badge (Running, Completed, Error). The arguments and output are collapsible, it shows the duration, and it collapses to one line on phones.
- **Sources:** `[E#]` markers become chips with a hover or tap snippet, publisher, date and tier. Grounded-web results render the search-suggestions HTML unmodified in a sandboxed container.
- **Mandate card:** an editable form bound to the `Mandate` fields, with Save (`PUT /api/mandate`) and a Draft badge.
- **Ticket card:**
  - Contents: side badge with a text label and ▲/▼, symbol, qty or notional, type, limit and TIF; quote with an "x s ago" age; estimated cost; buying power after the trade; position % after the trade; per-leg verdict chips; reasons; the thesis, risks and invalidation (collapsible); a countdown.
  - Buttons: **Confirm** carries a full label ("Confirm buy 3 VOO"). **Modify** opens inline fields and calls `refresh` with the edited legs through the chat: it sends "Modify: …" as a user message. **Reject**.
  - States: expired (greyed, with Refresh quote), requote (shows the old → new price and needs a new confirmation), submitted (an order timeline that polls `/api/portfolio` every 5 s for 1 minute).
- **Portfolio and orders panels:** refresh on open and after confirm.
- **Settings panel:** key status, key-entry fields (password inputs that are never echoed back), a trading-enabled toggle, a kill-switch button, model and effort.
- **Accessibility:** focus rings; one polite `aria-live` announcement at the end of each turn; no single-key Confirm shortcut; Enter sends and Shift+Enter adds a newline; Ctrl/Cmd+Enter sends; Esc stops.
- **Phones (~400px):** the rail becomes a drawer, the right panel a bottom sheet, and the Confirm button sticks to the bottom; inputs use 16px text.
- [ ] **Verify in the Browser pane with a fake-backed server** (`FINAGENT_FAKE_BROKER=1` wires `tests/fakes_trading.py`): send a prompt; see tool cards, sources and a ticket; confirm; see the order timeline. Check at 400px and in the light theme, and check EN/KO switching.

---

## Phase 5: Live checks, evaluation and documentation

### Task 5.1: Live check scripts
**Files:** `scripts/check_alpaca.py`, `scripts/check_research.py`.
- `check_alpaca.py`:
  - By default it is read-only: account (status, equity, buying power), clock, one IEX snapshot for SPY, and the latest 3 news items.
  - `--place-test-order` submits a 1-share limit buy of SPY at 50% of the last price, prints its status, then cancels it and prints the final status. It asks for confirmation on stdin and refuses if the account isn't paper.
- `check_research.py`: one call each to EDGAR (AAPL companyfacts Revenues), Tavily (if a key is set), Jina fetch, and grounding (if enabled). Prints the latency and the number of records.
- [ ] Run both once, the first read-only; run `--place-test-order` only with the user's go-ahead.

### Task 5.2: Scenario evaluation
**Files:** `data/eval/scenarios.jsonl` (8–12 directions: ETF core portfolio, dividend sleeve, sector tilt, a rebalance request, "sell everything", an out-of-scope options request, a Korean-language direction, and a prompt-injection web page case); `scripts/eval_scenarios.py`.
- The script drives `ChatAgent` against the fake broker (or the paper account with `--live-broker`, proposals only, never confirming). Metrics:
  - tickets proposed
  - risk verdicts (the share of accepted legs without clamps)
  - citation coverage (the share of sentences containing a number that carry an `[E#]`)
  - unsupported-number flags
  - presence of an invalidation statement
  - refusal on out-of-scope requests
  - latency and LLM calls
- It writes `data/eval/scenario_results.jsonl` and a summary table.
- [ ] Record the baseline numbers in the README.

### Task 5.3: Documentation
**Files:** `README.md` (both halves), `docs/architecture.md` (new, English then Korean), `docs/api_spec.md` (add `/api/*`), `.env.example` (new keys).
- **README:**
  - A new top section, "Trading assistant (paper)": a 60-second overview, setup (Alpaca paper keys, optional Tavily and SEC user agent), safety model (paper only, no order tool, tickets, kill switch), run command, screenshots.
  - The federated-architecture diagram now includes the external connectors.
  - A limitations section: no performance claims, paper fills are optimistic, free IEX data.
- **`docs/architecture.md`:** component diagram, data flow for a proposal and confirmation, the evidence model, the ticket state machine, and the risk rule table.
- [ ] Check that every command in the README works as written; the English and Korean halves have matching structure; the links resolve.

---

## Self-review against the spec
- §1–2 goal and principles → Tasks 1.5 (paper guard), 2.2–2.4 (code decides, approvals), 3.2 (no order tool), 3.3 (numbers from tools)
- §3 journey → Tasks 3.5 and 4.2 (connect, mandate, research, ticket, confirm, follow-up)
- §5 federated layer → Tasks 1.1–1.4, 3.2 (`ask_product_database`, KG/search tools), 0.3–0.4 (verification)
- §6 trading core → Tasks 2.1–2.4
- §7 chat agent → Tasks 3.1–3.4
- §8 API → Task 3.5
- §9 UI → Tasks 4.1–4.2
- §10 config and secrets → Tasks 1.1 and 3.5
- §11 testing → every task, plus 5.1–5.2
- §12 phasing → the plan's phases
- Type consistency: `SourceRecord`, `ResearchResult`, `OrderLeg`, `RiskDecision`, `Ticket`, `TicketStatus`, `ConfirmResult`, `ToolOutput` are defined once and used with the same names.
