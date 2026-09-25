# Agentic trading assistant: design

Status: approved direction (2026-09-26). The user asked for this pivot and said to go ahead with planning and implementation. This document is the reference for the implementation plan in `docs/superpowers/plans/2026-09-26-trading-agent.md`.

## 1. Goal

The user supplies an Alpaca **paper-trading** API key and a direction in plain language, for example "build a dividend ETF sleeve with $20k, moderate risk" or "I think AI semiconductors outperform over the next quarter". The agent then:

1. researches across local knowledge and external sources, with citations;
2. explains a strategy: thesis, risks, and the conditions that would invalidate it;
3. proposes concrete orders;
4. places an order only after the user confirms that exact order in the UI.

The work happens in a clean, AI-Studio-style chat interface.

**Non-goals for v1:**
- live-money trading
- options, crypto, short selling, margin
- autonomous execution without per-order confirmation
- intraday or high-frequency strategies
- backtest-driven alpha claims

## 2. Principles

These come from the research summarized in §13.

1. **The LLM proposes, code decides, the human approves.** The model has no tool that places orders. Deterministic code sizes the order, checks it against limits and submits it. The only path to the broker is a server endpoint that the UI's Confirm button calls. In published systems, approval gates built into agent frameworks have had forgery and argument-rewrite bugs (ADK #6461, Pydantic AI #6968, LangChain #33787), so the gate lives in our own service.
2. **Paper only, enforced in code.** The trading host is a constant (`paper-api.alpaca.markets`); no configuration value can point it elsewhere. Startup refuses anything that mentions the live host.
3. **Numbers come from tools.** Prices, returns, volatility, sizes and costs are computed by deterministic tools. A validator flags any number in an answer that does not appear in a tool result.
4. **Every claim is traceable.** Each research result is an `Evidence` record with an id, source, URL, and published and retrieved times. Answers cite evidence ids, and each proposal stores the ids it relied on.
5. **Retrieved text is data, not instructions.** Web, news and filing text is wrapped and marked untrusted. A turn that read external text can still only *propose*.
6. **Low frequency and conservative by default.** Long-only, no leverage, limit orders preferred, small order sizes, and a cap on orders per day. Live contests (Alpha Arena) lost money mainly through overtrading and leverage.
7. **Honest about skill.** The UI and prompts present the output as research and suggestions, not predictions. No credible study shows LLM agents earn repeatable returns (FINSABER, StockBench, LiveTradeBench), so v1 makes no performance claims.

## 3. User journey

1. **Connect.** In Settings, paste the Alpaca paper key id and secret. The server checks them with `GET /v2/account` against the paper host and stores them server-side in `.env` or the app DB, never in the LLM context. The UI shows a PAPER badge, equity and buying power.
2. **Direction to mandate.** The user states a direction. The agent drafts a **Mandate**: universe or allow-list, budget, max % per position, max number of positions, cash buffer, horizon, style, allowed order types and max orders per day. It is shown as a card; the user edits and saves it. Every later proposal is checked against it.
3. **Research.** The agent calls tools, and the UI streams each call as a compact card with its sources.
4. **Explain.** The agent writes the strategy with citations, risks and invalidation conditions.
5. **Propose.** The agent calls `propose_trades`. The risk engine validates, clamps or rejects each leg, and the server creates a **ticket**: an order preview with estimated cost, buying power after the trade, the portfolio before and after, warnings, a 5-minute expiry and a hash.
6. **Confirm.** The user presses Confirm, Modify or Reject on the ticket. On Confirm the server re-quotes and re-runs the risk checks. If the price moved beyond a threshold it issues a new ticket instead of submitting. Otherwise it submits with an idempotent `client_order_id`, polls the status and posts the result into the chat.
7. **Follow up.** The user asks things like "How is it doing?" or "Trim NVDA". The agent reads positions and news, and exits go through the same propose, ticket, confirm path.

## 4. Architecture

```
web/app (vanilla ES modules, AI-Studio-style)
   │  POST /api/chat (SSE)          POST /api/tickets/{id}/confirm|reject
   ▼                                 ▼
chat/agent.py ── ChatAgent loop ──► tools (read + propose only)
   │                                   │
   │                ┌──────────────────┼──────────────────────────────┐
   │                ▼                  ▼                              ▼
   │        research/ (federated)   trading/marketdata.py      trading/proposals.py
   │        ├ local: product DB,    (Alpaca data: quotes,      (ticket store, TTL, hash)
   │        │  KG, lexical+vector,   bars, news, clock,               │
   │        │  docs (existing kb/)   movers)                          ▼
   │        ├ edgar.py (SEC)                                  trading/risk.py (pure rules)
   │        ├ websearch.py (Tavily)                                   │
   │        ├ grounding.py (Gemini                                    ▼
   │        │  Google Search, opt.)                    trading/execution.py ──► trading/broker.py
   │        └ evidence.py (ledger,                     (confirm → re-quote →      (httpx, paper host
   │           cache, untrusted wrap)                   re-risk → submit →         hard-coded)
   │                                                    reconcile → poll)
   ▼
store/app.db (SQLite): sessions, messages, mandates, tickets, orders, evidence, audit ledger
```

- **Kept unchanged:** the existing single-question QA pipeline (`agent/orchestrator.py`, `/answer`, `/answer/stream`, `/ui`). It remains the competition contract and a regression test for the federated core.
- **New packages:** `research/`, `trading/`, `chat/`. They depend on `kb/` and `agent/llm.py`; nothing in `agent/` depends on them.
- **Each unit has one purpose and a narrow interface**, so it can be tested in isolation with mocked HTTP.

## 5. Federated research layer

"Federated" means one question fans out to heterogeneous sources behind a common interface, and the results merge with their provenance kept.

**Connectors.** Each one returns `Evidence` items.

| Connector | Source | Key | Use |
|---|---|---|---|
| `local.product_db` | existing SQLite masters (incl. 6,037 US-listed ETFs in `v_etf_gl`: ticker, expense ratio, AUM, asset class, index) | none | ETF screening and attributes |
| `local.knowledge_graph` | existing rdflib KG (1.08M triples) | none | themes, constituents, issuer relations |
| `local.search` | existing BM25 + LSA vector index | none | fuzzy product lookup |
| `local.docs` | existing reference docs | none | term and risk explanations |
| `local.product_qa` | the existing QA orchestrator as a sub-agent (`ask_product_database`) | Gemini | multi-step product questions answered through the proven pipeline |
| `edgar` | SEC EDGAR: submissions, companyfacts (XBRL), filing documents, full-text search | none; User-Agent required; ≤5 req/s | fundamentals, filings, risk factors |
| `news` | Alpaca News (Benzinga) | Alpaca | company and ETF news with clean timestamps |
| `web` | Tavily search + extract (Jina Reader as a keyless fallback fetcher) | `TAVILY_API_KEY` (free 1,000 credits/mo) | general and financial web research |
| `grounded_web` (optional) | Gemini Google Search grounding through a native `generateContent` call | Gemini (grounding on 3.x needs a paid tier) | quick grounded answers with sources and search suggestions |
| `market` | Alpaca market data: snapshots, bars, clock/calendar, movers | Alpaca | prices and history (free IEX feed; SIP only if `end` is ≥15 min old) |

**`Evidence` record:** `id` (e.g. `E12`), `connector`, `title`, `url`, `publisher`, `tier` (1 = regulator/issuer, 2 = licensed newswire, 3 = media, 4 = blogs/social), `published_at`, `retrieved_at`, `snippet`, `content_sha256`, `untrusted` (true for web and news).

The evidence store also acts as a cache, with a TTL per connector:

| Content | TTL |
|---|---|
| EDGAR filings | permanent |
| companyfacts | 12 h |
| news | 15 min |
| web search | 15 min for news topics, 12 h otherwise |

Grounded-web results are **not** cached or indexed. They live only in the chat history, which the Gemini terms allow for a user's own history.

**Untrusted text:**
- strip hidden HTML and zero-width characters, and cap chunk length;
- wrap in `<untrusted source="E12">…</untrusted>`;
- the prompt says instructions inside such blocks are data;
- URLs found inside retrieved text are never followed automatically.

**Numeric grounding.** After each assistant message, extract its numbers (%, $, dates, share counts) and check that each appears in a cited evidence snippet or in a tool result from that turn. Unsupported numbers are flagged in the UI and logged, and are never silently shown as facts. v1 flags without blocking; the eval tracks the rate.

**Verification of the existing federated core (Phase 0, before new work):**
- **Gemini evaluation run:** run the 54-question answer key in Korean, plus the English versions the new console already carries, through `/answer` on Gemini. Report accuracy per source family (SQL, KG, lexical/vector, docs, cross-source) and per language.
- **Connector tests:** one test per source proving it contributes. SQL rows, KG neighbours, lexical and vector hits and doc chunks each appear in `retrieved_context` for a question that needs them.
- **Fix and document:** fix what breaks, then update README and docs with the measured Gemini results, next to the historical HyperCLOVA X numbers.

## 6. Trading core

**Mandate** (pydantic, persisted per session):

| Field | Default |
|---|---|
| `direction_text` | — |
| `budget_usd` | ≤ buying power |
| `universe` | `us_equity` + ETFs |
| `allow_symbols` / `deny_symbols` | empty |
| `max_position_pct` | 20% of budget |
| `max_positions` | 8 |
| `cash_buffer_pct` | 5% |
| `horizon` | weeks / months |
| `order_types` | limit, market |
| `max_orders_per_day` | 10 |
| `max_order_usd` | — |
| `price_band_pct` | 3% |

**Risk engine** (`trading/risk.py`, pure functions, fully unit-tested). The input is a proposed leg plus a context snapshot: account, positions, open orders, asset, quote, clock and mandate. The output is `accept`, `clamp` (with the adjusted fields) or `reject`, with machine-readable reasons. Rules:
- **Account:** tradable; not blocked or suspended.
- **Asset:** `us_equity`, active and tradable. Fractional orders need `fractionable` and `day`; notional orders are market + day only.
- **No shorts:** a sell above `qty_available` is rejected.
- **Wash-trade guard:** reject if there is an open order on the opposite side of the same symbol.
- **Price increments:** 2 decimals at ≥ $1, 4 decimals below.
- **Limit price:** within `price_band_pct` of the last quote.
- **Buying power:** the estimated cost, with a slippage buffer for market orders, is at most buying power minus the cash buffer.
- **Mandate limits:** position %, number of positions, order USD and orders per day.
- **Session:** a warning when the market is closed (the order will queue as `accepted`).
- **Order types:** v1 supports only market and limit orders (`day` or `gtc`). Bracket, stop and trailing orders are out of scope until the core loop is proven.

**Tickets** (`trading/proposals.py`):
- **Lifecycle:** `proposed` → `confirmed` → `submitting` → `submitted` (then mirrors the order status) | `rejected_by_user` | `expired` | `superseded` | `failed`.
- **Stored with each ticket:** the normalized legs after risk checks, the risk report, a quote snapshot with its time, `expires_at` (now + 5 min), `sha256` of the canonical order JSON, the evidence ids, and one `client_order_id` per leg (`fa-<ticket>-<leg>`).
- **One active ticket per session.** A new proposal supersedes the previous one.

**Execution** (`trading/execution.py`), called only by `POST /api/tickets/{id}/confirm {hash}`:
1. **Check the ticket:** it belongs to this session, is not expired, and its hash matches.
2. **Refresh:** fetch the account, clock and quotes again and re-run the risk checks. If anything drifted beyond the thresholds, create a new ticket and return it instead of submitting.
3. **Submit** each leg with `POST /v2/orders` and its stored `client_order_id`. Never blind-retry. On a timeout, 5xx or 422 "client_order_id must be unique", reconcile with `GET /v2/orders:by_client_order_id`.
4. **Poll** `GET /v2/orders/{id}` for up to about 10 s, then push status updates to the chat.

**Kill switch:** a Settings toggle "Trading enabled" (default on for paper). When off, confirm returns 423. An optional button sets Alpaca `suspend_trade=true` and cancels open orders.

**Audit ledger:** an append-only table recording every tool call (args hash, result hash, latency), evidence id, mandate version, ticket as shown, approval (who, when, hash), broker request and response, and order status change. It is exportable as JSONL.

## 7. Chat agent

**Sessions** (SQLite): messages, mandate, active ticket, and settings (model, effort, language).

**Loop:** reuse `agent/llm.py` (`GeminiLLM`, with thought-signature echo and retries). Each user turn runs a tool loop with a time budget (about 90 s) and at most 8 rounds, streaming events. The final assistant text is plain markdown with `[E12]`-style citations.

**Tools** (read and propose only):
- **Account:** `get_account`, `get_positions`, `list_orders`
- **Market:** `get_market_status`, `lookup_asset`, `get_quotes`, `get_price_history`, `compute_metrics` (returns, volatility, drawdown, SMA, correlation, deterministic), `get_movers`
- **Research:** `search_news`, `web_search`, `fetch_source`, `sec_filings`, `sec_financial_facts`, `sec_filing_passages`, `grounded_web_answer` (optional), `screen_us_etfs` (structured screen over `v_etf_gl`), `ask_product_database` (the existing federated QA), `search_products` / `graph_neighbors` (direct KG/lexical lookups)
- **Mandate:** `draft_mandate`, which proposes changes; the user saves them in the UI.
- **Trading:** `propose_trades(legs[], thesis, risks, invalidation, evidence_ids[])` returns the ticket id plus the risk report (accepted, clamped, rejected legs).

**Prompt rules:**
- Answer in the user's language; English is the default.
- Research before proposing, and cite evidence ids.
- All numbers come from tools; never state a price or size from memory.
- Respect the mandate.
- Propose at most one ticket per turn.
- Never claim an order was placed; only the confirmation flow places orders.
- State that this is paper trading and not investment advice.

**Streaming events** follow the Vercel AI SDK stream-protocol names, so a React client could consume them later: `start`, `reasoning-delta` (summaries only), `tool-input-available`, `tool-output-available` (with evidence), `source-url`, `text-delta`, `data-mandate`, `data-trade-proposal`, `data-order-status`, `data-grounding-flags`, `finish`, `error`.

## 8. API

| Method and path | Purpose |
|---|---|
| `GET /answer`, `GET /answer/stream`, `GET /ui` | unchanged (competition contract and dev console) |
| `GET /` | the new app (`web/app/index.html`) |
| `POST /api/sessions` / `GET /api/sessions` / `GET /api/sessions/{id}` | chat sessions |
| `POST /api/chat` | body `{session_id, message}`; SSE response (fetch stream) |
| `GET /api/portfolio` | account summary, positions, open orders |
| `GET/PUT /api/mandate/{session_id}` | read or save the mandate |
| `POST /api/tickets/{id}/confirm` | body `{hash}`; executes |
| `POST /api/tickets/{id}/reject` | rejects the ticket |
| `POST /api/tickets/{id}/refresh` | re-quotes the ticket |
| `GET/PUT /api/settings` | keys status (never the secrets), trading enabled, model, effort |
| `GET /api/ledger/export` | JSONL download for the session |

The app binds to localhost by default. It is a single-user personal tool; if it is ever exposed, it needs auth.

## 9. UI (web/app)

**Style:** after Google AI Studio.
- A 64px left rail listing sessions and a New chat button; it expands to 240px.
- A centre conversation column, at most about 760px wide.
- A 340px right panel with tabs: Portfolio, Orders, Mandate, Settings.
- A top bar with the session title, a PAPER badge, a model chip, and muted token/latency figures.
- Dark first with a light theme; a single accent colour.
- Fonts: Google Sans Flex, with system and Korean fallbacks.

**Components:**
- **Messages:** markdown messages (a vendored streaming-markdown renderer plus DOMPurify).
- **Activity:** a collapsible "Thinking" summary, and tool-call cards with a family icon, status badge, collapsible arguments and results, and duration.
- **Sources:** source chips with hover snippets.
- **Mandate card:** editable, with a Save button.
- **Trade ticket card:** BUY/SELL badge with text, symbol, qty or notional, type, limit price and TIF. It also shows:
  - the quote and its age, estimated cost, buying power after the trade, and position % after the trade;
  - risk notes and a rationale with citations;
  - a countdown; Confirm / Modify / Reject buttons; states and an order timeline.
- **Composer:** Enter sends; Shift+Enter adds a newline; Ctrl/Cmd+Enter also sends; Esc stops.
- **Other:** a disclaimer footer; an EN/한국어 toggle that reuses the console's dictionary pattern.

**Accessibility:**
- Contrast ≥ 4.5:1.
- Visible focus rings.
- No `aria-live` on the element receiving streamed tokens; a single polite announcement at the end.
- No single-key shortcut for Confirm; the Confirm button carries a full label ("Confirm buy 10 VOO").

**Responsive at about 400px:**
- the rail becomes a drawer and the right panel a bottom sheet;
- sticky Confirm button;
- inputs use at least 16px text.

## 10. Configuration and secrets

**New `.env` keys:**
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` (paper)
- `TAVILY_API_KEY` (optional)
- `SEC_USER_AGENT` ("Name email", required by the SEC)
- `GROUNDING_MODEL` (optional)
- `TRADING_ENABLED=true`, `APP_DB=data/store/app.db`

**Rules:**
- Missing keys disable the related tools, with a clear message.
- Secrets never enter prompts, logs or the ledger (they are redacted).
- Settings can store keys entered in the UI. They are saved to the local `.env` only after a live check and never echoed back.

## 11. Testing and evaluation

- **Unit:**
  - risk rules (table-driven)
  - ticket lifecycle and hash/expiry
  - execution reconciliation paths (timeout, 5xx, duplicate `client_order_id`)
  - connectors against recorded HTTP fixtures (`httpx.MockTransport`)
  - evidence wrapping and numeric grounding
  - chat loop with `StubLLM`
  - SSE event contract
- **Integration:** the FastAPI TestClient drives the full propose → confirm → submit flow against a fake broker. A paper-host guard test makes sure the live host can never be reached.
- **Live smoke (manual, opt-in):**
  - `scripts/check_alpaca.py`: read-only account, clock and quote checks. It places no orders unless run with `--place-test-order`, which submits a 1-share limit order far from market and cancels it.
  - `scripts/check_research.py`: one call per connector.
- **Federated core:** the Phase 0 evaluation, repeated after each phase to catch regressions.
- **Agent quality:** a small scenario set of directions, scored on:
  - mandate compliance
  - citation coverage and the rate of unsupported numbers
  - presence of an invalidation condition
  - proposal validity (the risk engine accepts it without clamps)
  - latency and cost

  No return claims. Forward paper trading is logged for later analysis only.

## 12. Phasing

The implementation plan spells out each phase in detail.

| Phase | Deliverable |
|---|---|
| 0 | English edition finished; federated core verified on Gemini (KO + EN eval), fixes, and docs updated with measured results |
| 1 | `research/` connectors + evidence store + `trading/broker.py` and `marketdata.py` (read-only) + metrics, all with mocked-HTTP tests |
| 2 | Mandate, risk engine, tickets, execution, kill switch, ledger, with tests |
| 3 | Chat agent (sessions, tools, prompt, loop, SSE, numeric grounding), plus the `/api/*` endpoints |
| 4 | AI-Studio-style web app |
| 5 | Scenario evaluation, live smoke on the paper account (with the user's keys), and README/docs (English first, Korean second) |

## 13. Research basis (summary)

**What published systems show:**
- Multi-agent systems (TradingAgents, FinCon) report gains that shrink or vanish under leakage control, longer windows and real costs (FINSABER, KTD-Fin, Alpha Illusion).
- Reasoning and leaderboard-top models don't trade better (StockBench, LiveTradeBench).
- Tools and structured data matter more than model choice: FinRetrieval measured 90.8% with structured data vs 19.8% with web search alone.
- Current systems move sizing and risk into deterministic code (ai-hedge-fund v2, FinRobot).

**Safety practice:**
- Brokers use ring-fenced accounts, confirmation screens and kill switches.
- Robinhood Cortex allows one order per conversation behind an explicit confirmation screen.
- tastytrade MCP requires a 60 s single-use token.
- FINRA's 2026 report stresses human-in-the-loop controls and logging of agent actions.

**Alpaca:**
- Use direct `httpx` rather than the synchronous alpaca-py SDK, whose client retries order POSTs.
- Account fields: the PDT fields were removed in July 2026; use `borrow_status`, not `easy_to_borrow`.
- Paper fills are optimistic, with no slippage, fees or dividends.

**External RAG:**
- Gemini Google Search grounding is not available on the OpenAI-compatible endpoint, so it needs a native call, and its terms forbid caching. The plan follows this.
- EDGAR is free and primary (User-Agent required).
- Tavily has date filters and an extract endpoint.
- Alpaca News has Benzinga timestamps.

Full source lists are kept in the research notes (`docs/superpowers/specs/2026-09-26-trading-agent-research.md`).

## 14. Risks and open points

- **Free-tier Gemini limits.** Keys hit 429 and 503 often, so evaluations and chats need pacing. Grounding on 3.x needs a paid tier.
- **IEX vs NBBO.** Free IEX quotes differ from the NBBO that paper fills use. Estimates carry a buffer.
- **Korean products.** They stay as research context only; Alpaca cannot trade them.
- **Stored secrets.** Keys in `.env` are fine for a personal local tool. Deploying beyond localhost needs auth and a secret store.
