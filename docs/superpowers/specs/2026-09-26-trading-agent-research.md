# Research notes for the trading-assistant design (2026-09-25/26)

Four parallel web-research passes back `2026-09-26-trading-agent-design.md`:
1. LLM trading agents
2. the Alpaca API
3. external RAG for finance
4. chat UI

This file keeps the findings that shaped decisions, with sources. Where a report flagged something as unverified, the point below is marked "(unverified)".

## 1. LLM trading agents: what worked, what failed

- **TradingAgents** (Tauric Research) is a multi-agent LangGraph system: analysts, a bull/bear debate, a trader, a risk team and a portfolio manager.
  - The paper reports AAPL +26.62% with Sharpe 8.21, over Jan–Mar 2024 only.
  - An independent reproduction on GOOGL for May–Jul 2025 gave 15.8% (GPT-4o), against 19.1% buy-and-hold.
  - Later releases fixed look-ahead leaks in data, memory and company profiles.
  - https://github.com/tauricresearch/tradingagents · https://arxiv.org/html/2412.20138 · https://github.com/TauricResearch/TradingAgents/releases
- **ai-hedge-fund v2.4.0** (2026-09-24) was rebuilt as a deterministic pipeline: "The LLM never touches the trade… deterministic code sizes and places orders; risk limits are hard gates." https://github.com/virattt/ai-hedge-fund
- **FinRobot:** "Numbers are code-calculated. Narratives are LLM-assisted." https://github.com/AI4Finance-Foundation/FinRobot
- **Critiques of reported results:**
  - FinMem's return drops about 72% after the model's pretraining cutoff.
  - The TradingAgents Sharpe falls from 0.43 to 0.22 once costs are included.
  - 35 of 40 friction components are unmodeled. https://arxiv.org/html/2605.16895v1
- **FINSABER (KDD 2026):** LLM advantages deteriorate over 20 years and 100+ symbols. https://arxiv.org/abs/2505.07078
- **StockBench:**
  - Best agents made about 2% against 0.4% for passive over 82 days.
  - All agents underperformed passive in a downturn.
  - Thinking variants were worse.
  - https://arxiv.org/html/2510.02209
- **LiveTradeBench:** arena score vs return ρ ≈ 0.05. https://www.emergentmind.com/papers/2511.03628
- **KTD-Fin:** under leakage control, returns are mostly explained by market and style exposure. https://arxiv.org/abs/2605.28359
- **Look-ahead recall:** models recall pre-cutoff data, and instructions don't prevent it. https://arxiv.org/abs/2504.14765 · https://arxiv.org/abs/2510.07920
- **Alpha Arena Season 1 (nof1, real money):**
  - Most models lost; GPT-5 −62.7%, Gemini 2.5 Pro −56.7%.
  - PnL was dominated by fees and overtrading.
  - https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/ · https://protos.com/llm-crypto-trading-contest-finds-llms-cant-trade-crypto/
- **Debate:**
  - Stanford (Aug 2026, 455 runs): debate portfolios beat 1/N only 33% of the time.
  - Reasoning quality did not predict Sharpe (r = 0.07).
  - https://arxiv.org/html/2609.29701
- **Tools vs web:** FinRetrieval measured 90.8% with structured APIs vs 19.8% with web search alone. https://arxiv.org/abs/2603.04403
- **Adversarial headlines:** up to −17.7 pp annual return. https://arxiv.org/abs/2601.13082
- **Approval gates in frameworks are not security boundaries:**
  - ADK #6461: forged confirmation. https://github.com/google/adk-python/issues/6461
  - Pydantic AI #6968: approved args ≠ executed args. https://github.com/pydantic/pydantic-ai/issues/6968
  - LangChain #33787: edit was ignored. https://github.com/langchain-ai/langchain/issues/33787
- **Industry and regulator practice:**
  - Robinhood agentic trading: a ring-fenced account and per-trade notifications. https://robinhood.com/us/en/newsroom/robinhood-is-now-open-to-agents/
  - FINRA 2026 report on AI agents: human-in-the-loop controls, logging and guardrails. https://www.finra.org/media-center/blog/observations-on-ai-agents
- **Evaluation practice:**
  - Point-in-time replay only after the model's knowledge cutoff.
  - Include costs.
  - Use baselines: SPY, equal-weight, a rule-based strategy.
  - Run 5–10 repeats.
  - Deflated Sharpe: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

## 2. Alpaca (paper)

**SDK and connection**
- alpaca-py 0.44.0 (2026-08-11) supports Python 3.10–3.14.
  - Its REST client is synchronous and uses requests with no timeout.
  - It retries on 429 and 504, including order POSTs, so an order can be submitted twice.
  - Streams call `asyncio.run()` internally, which conflicts with FastAPI (issue #618).
  - Decision: use `httpx.AsyncClient` directly. https://github.com/alpacahq/alpaca-py
- Hosts:
  - Paper trading: `https://paper-api.alpaca.markets`. Live: `https://api.alpaca.markets`.
  - Market data: `https://data.alpaca.markets`.
  - Paper keys fail on the live host and vice versa.
- Auth headers: `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`. https://docs.alpaca.markets/docs/authentication

**Account rules**
- The PDT rule was retired on 2026-06-04, and the PDT/DTBP account fields were removed by 2026-07-06.
  - New intraday margin framework: pre-trade rejection of orders that would create an intraday margin deficit.
  - https://docs.alpaca.markets/us/docs/understanding-finras-new-intraday-margin-rule-and-the-end-of-pdt
- Account numeric values are returned as strings. `portfolio_value` is deprecated; use `equity`.
- Configurations: `no_shorting`, `suspend_trade`, `max_margin_multiplier`. https://docs.alpaca.markets/us/reference/getaccountconfig-1

**Orders** — `POST /v2/orders`. https://docs.alpaca.markets/us/reference/postorder · https://docs.alpaca.markets/us/docs/orders-at-alpaca
- Fields: `symbol`, `qty|notional`, `side`, `type`, `time_in_force`, `limit_price`, `stop_price`, `trail_price|trail_percent`, `extended_hours`, `client_order_id` (≤128 characters), `order_class` (`simple|bracket|oco|oto`), `take_profit`, `stop_loss`.
- Order statuses:
  - Final: `filled`, `canceled`, `expired`, `rejected`, `replaced`.
  - `accepted` means queued, which is common when the market is closed.
- Fractional orders: `fractionable` asset and DAY only. Notional orders: market + DAY is the safe subset. https://docs.alpaca.markets/us/docs/fractional-trading
- Tick size: 2 decimals at ≥ $1, 4 below; otherwise the order is rejected with 42210000.
- Idempotency: a duplicate `client_order_id` returns 422, which proves the first order exists. Reconcile with `GET /v2/orders:by_client_order_id`.
- Wash-trade guard: 403 "potential wash trade detected" when an opposite-side order for the same symbol is open.

**Other endpoints**
- Assets: `GET /v2/assets/{symbol}` returns `tradable`, `fractionable`, `shortable`, `marginable`, `borrow_status`, `attributes[]`.
- Clock: `GET /v2/clock`. Calendar: `GET /v2/calendar`.
- Market data:
  - `/v2/stocks/snapshots`, `/v2/stocks/bars`.
  - The free plan is IEX real-time only; historical SIP data needs `end` at least 15 minutes old.
  - Free plan limit: 200 requests/min.
  - https://docs.alpaca.markets/us/docs/about-market-data-api
- News: `GET /v1beta1/news` (Benzinga, since 2015, limit ≤ 50). https://docs.alpaca.markets/us/reference/news-3
- Screener: `/v1beta1/screener/stocks/most-actives` and `/v1beta1/screener/stocks/movers`.
- Trading API rate limit: 200 requests/min. https://alpaca.markets/support/usage-limit-api-calls

**Paper trading and the official MCP server**
- Paper trading starts at $100k. Fills are simulated against NBBO with random partial fills (about 10%).
  - Not simulated: slippage, market impact, fees, dividends.
  - https://docs.alpaca.markets/us/docs/paper-trading
- The official MCP server (v2.3.2, 65 tools) has no confirmation step and a single environment flag switches it to live. We borrow its parameter shapes and its "order MAY have been placed" timeout handling, and do not expose it directly. https://github.com/alpacahq/alpaca-mcp-server

## 3. External RAG

**Gemini Google Search grounding**
- Works only through the native API (Interactions API or `generateContent` with the `google_search` tool).
  - It is not available on the OpenAI-compatible chat endpoint.
  - Combining it with function calling is a native, preview feature for Gemini 3 only.
  - https://ai.google.dev/gemini-api/docs/google-search · https://ai.google.dev/gemini-api/docs/openai · https://ai.google.dev/gemini-api/docs/tool-combination
- Pricing: on 3.x the free tier has no grounding; the paid tier includes 5,000 queries/month, then $14 per 1k. https://ai.google.dev/gemini-api/docs/pricing
- Terms of service: grounded results may not be cached, indexed or otherwise reused. Keeping them in a user's own chat history is allowed. Search Suggestions must be shown. https://ai.google.dev/gemini-api/terms

**Other external sources**
- SEC EDGAR: free, ≤ 10 requests/s, a declared User-Agent is required. https://www.sec.gov/search-filings/edgar-application-programming-interfaces
  - Endpoints: `data.sec.gov/submissions/CIK##########.json`, `/api/xbrl/companyfacts/…`, `/frames/…`.
  - Full-text search since 2001: https://www.sec.gov/edgar/search/efts-faq.html
- Tavily:
  - Free tier: 1,000 credits/month. Search costs 1 credit (basic) or 2 (advanced).
  - Extract: 1 credit per 5 URLs.
  - Filters: `topic` news/finance, `start_date` and `end_date`.
  - https://docs.tavily.com/documentation/api-reference/endpoint/search
- Jina Reader: fetches any URL as markdown; 20 RPM without a key. https://jina.ai/reader/
- FRED/ALFRED keep the values as known at each date (real-time vintages), so there is no look-ahead. https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- yfinance is for personal use only and gets rate-limited, so it is not used. https://github.com/ranaroussi/yfinance

**Benchmarks and practices**
- FinanceBench (GPT-4-Turbo):

  | Setup | Correct |
  |---|---|
  | Shared vector store | 19% |
  | Per-document store | 50% |
  | Long context | 79% |
  | Oracle | 85% |

  Lesson: narrow down to the right document first. https://arxiv.org/abs/2311.11944
- Practices:
  - Chunk by document structure: Items, whole tables, speaker turns. https://arxiv.org/html/2402.05131v3
  - Enforce numeric grounding.
  - Tier sources by reliability.
  - Spotlighting (marking untrusted text) cut prompt-injection success from over 50% to under 2%. https://arxiv.org/abs/2403.14720
  - Keep the evidence ledger with timestamps. https://arxiv.org/html/2602.14233v1

## 4. Chat UI

**Google AI Studio's layout**
- Left navigation, central chat, and a right "Run settings" panel (model, system instructions, thinking level, tools, safety).
- Ctrl/Cmd+Enter runs the prompt; thought summaries are collapsed.
- https://ai.google.dev/gemini-api/docs/ai-studio-quickstart · https://blog.google/innovation-and-ai/technology/developers-tools/ai-studio-updates-more-control/
- Google Sans Flex is open-source (OFL). https://fonts.google.com/specimen/Google+Sans+Flex
- Exact colours and spacing could not be verified (sign-in required).

**Patterns worth borrowing**
- Perplexity shows its plan and steps.
- Vercel AI Elements has tool-state badges and a Confirmation component. https://elements.ai-sdk.dev/components/tool
- Gradio uses nested steps.

**Confirmation patterns**
- Robinhood Cortex requires an explicit final confirmation screen and allows one order per conversation. https://cdn.robinhood.com/assets/robinhood/legal/cortex_assistant_disclosure.pdf
- tastytrade MCP uses single-use tokens that expire after 60 s and are bound to the exact order. https://github.com/tastytrade/tastytrade-mcp
- The AI SDK checks approvals on the server and can sign them. https://ai-sdk.dev/docs/agents/tool-approvals

**Implementation and accessibility**
- Keep vanilla ES modules with no build step.
- Vendor streaming-markdown and DOMPurify. https://developer.chrome.com/docs/ai/render-llm-responses
- Name SSE events after the AI SDK stream protocol. https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
- Accessibility:
  - WCAG contrast of 4.5:1 or more.
  - Never put `aria-live` on the element that receives streamed tokens.
  - No single-key shortcut for Confirm.
