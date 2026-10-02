English | [한국어](#한국어)

# Evaluation API Specification

## Endpoint

| Item | Value |
|---|---|
| Method / Path | `GET /answer` |
| Content-Type (response) | `application/json; charset=utf-8` |
| Authentication | None (no auth headers, no POST body) |
| Response time | Organizer's rule: **a response that takes more than 300 seconds counts as no response**; anything within that limit is still scored, even if slow (scores are graded). Internal server budget is 120 seconds; when it is exceeded, the server answers immediately from the evidence gathered so far |
| Answer-generation LLM | Google Gemini (`LLM_PROVIDER=gemini`, default `LLM_MODEL=gemini-3.8-flash`, called through Gemini's OpenAI-compatible endpoint); Anthropic and OpenAI remain available as alternate providers. The competition build used NCP HyperCLOVA X (`HCX-005`, CLOVA Studio OpenAI-compatible endpoint), as the competition rules required |
| Answer language | `answer` follows the language of the question: English by default, Korean when the question is written in Korean. Korean product or company names inside an English question do not make it a Korean question. Labels in `think_trace` and `retrieved_context` are always English |
| Auxiliary | `GET /health` (status, table base dates, index size, graph asserted/inferred triple counts, vector backend), `GET /` (info message) |
| Reference data | 5 SQLite views (`v_bond_kr` · `v_etf_kr` · `v_etf_gl` · `v_fund_pub` · `v_etf_constituent`) · 53,375 product rows + 22,599 constituent rows · knowledge graph of 1,082,874 triples |

## Request (Query String)

| Parameter | Type | Required | Description |
|---|---|---|---|
| `question_id` | string | Yes | Organizer's question ID (e.g. `Q-001`). Returned as-is |
| `question` | string | Yes | Original text of the evaluation question (URL-encoded) |

- Undefined parameters are ignored (handled without a 500).
- If `question` is empty, the server returns 200 OK with a guidance message in the same schema.

```bash
curl -G "https://{endpoint}/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘"
```

(The example question asks: "Show me KRW bonds currently available for sale rated AA- or higher.")

```python
import requests
requests.get("https://{endpoint}/answer", params={"question_id": "Q-001", "question": "평가질의"}).json()
```

(`평가질의` = "evaluation question".)

## Response (200 OK, JSON)

| Field | Type | Required | Description |
|---|---|---|---|
| `question_id` | string | Yes | Same as the request value |
| `question` | string | Yes | Same as the request value |
| `retrieved_context` | string | Yes | Evidence for the answer. Each `[Evidence n]` block lists the referenced table (Korean name · view name), data base date, **row count**, the SQL/SPARQL/search query that was executed, and the referenced product codes and names. When external documents are used: document title and source.<br>The count is the **total number of rows that match the filter**; when the displayed rows are cut at the 200-row cap, it is written separately as "N total (200 shown)" |
| `think_trace` | string | Yes | Numbered processing steps: pre-analysis (entities, themes, ontology validation) → each round's tool calls and result sizes → final status and elapsed time |
| `answer` | string | Yes | Final answer, in the language of the question (English by default; Korean when the question is in Korean). If the data cannot confirm the answer, it starts with the fixed cannot-confirm sentence ("This cannot be confirmed with the provided data." in English, "제공된 데이터로는 확인할 수 없습니다" in Korean); if the conditions are ambiguous, it asks a clarifying question (ending with a question mark) |

### JSON Schema

```json
{
  "type": "object",
  "required": ["question_id", "question", "retrieved_context", "think_trace", "answer"],
  "properties": {
    "question_id": {"type": "string"},
    "question": {"type": "string"},
    "retrieved_context": {"type": "string"},
    "think_trace": {"type": "string"},
    "answer": {"type": "string"}
  }
}
```

### Example response

The example question is in Korean, so `answer` is in Korean; the labels in `retrieved_context` and `think_trace` are English. Table names such as `국내채권마스터` (domestic bond master) and product names come from the Korean data and stay Korean.

```json
{
  "question_id": "Q-001",
  "question": "현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘",
  "retrieved_context": "[Evidence 1] 국내채권마스터(v_bond_kr) · base date 2026-08-21 · 15837 rows\n  sql: SELECT DISTINCT pd_no, pd_nm, crd_grd, buy_yield, after_tax_yield, mat_date FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-') ORDER BY buy_yield DESC LIMIT 200\n  Refs: KR350801G954 한국가스공사 394 | KR6056554B20 SK이노베이션10-4 | ...",
  "think_trace": "1. Pre-analysis: product mentions none\n2. Round 1: sql_query(...) → result 3120 chars, 15837 rows\n3. Round 2: final_answer(status=answered)\n4. Final: status=answered, 1 evidence, 14.2s",
  "answer": "현재 구매 가능한 원화 채권 중 신용등급 AA- 이상 종목은 총 15,837건입니다 (국내채권마스터, 기준일 2026-08-21). 매수수익률이 수록된 종목(has_buy_yield=1) 기준 상위: ... (근거: 국내채권마스터 · 2026-08-21)"
}
```

The same question asked in English returns the same `retrieved_context` and `think_trace`, with an English `answer`:

```json
"answer": "There are 15,837 KRW bonds currently available for purchase with a credit rating of AA- or higher (국내채권마스터, as of 2026-08-21). Top issues among those with a recorded purchase yield (has_buy_yield=1): ... (Source: 국내채권마스터 · 2026-08-21)"
```

### Example of an unconfirmable question (same schema, 200 OK)

The question asks: "Find bonds with a credit rating of AAAA."

```json
{
  "question_id": "Q-031",
  "question": "신용등급 AAAA인 채권 찾아줘",
  "retrieved_context": "No data referenced",
  "think_trace": "1. Pre-analysis: product mentions none; 1 ontology violation(s)\n2. Round 1: final_answer(status=cannot_confirm)\n3. Final: status=cannot_confirm, 0 evidence, 6.1s",
  "answer": "제공된 데이터로는 확인할 수 없습니다. 신용등급 'AAAA'는 존재하지 않는 값입니다. 유효한 신용등급: AAA, AA+, AA0, AA-, A+, ... D"
}
```

Asked in English, the answer reads:

```json
"answer": "This cannot be confirmed with the provided data. 'AAAA' is not a valid credit rating. Valid credit ratings: AAA, AA+, AA0, AA-, A+, ... D"
```

## Error handling

- Even when an internal error occurs, the server does not return HTTP 500; it responds with 200 and the same schema (a retry message in `answer`, the error type in `think_trace`).
- When an LLM call fails or times out, the evidence gathered so far is still returned in `retrieved_context`.

---

# Chat and trading API (`/api/*`)

The chat and trading API is the HTTP surface of the local paper-trading app (`web/app`). It is not part of the competition contract: `GET /answer`, `GET /answer/stream`, `GET /health` and `/ui` are unchanged, and the rules below do not apply to them. Examples assume the server runs on `http://127.0.0.1:8000`. How the pieces work is described in [architecture.md](architecture.md).

## Access rules

| Rule | Detail | Refusal |
|---|---|---|
| Loopback clients only | The socket peer address must be in `127.0.0.0/8`, be `::1`, or be an IPv4-mapped loopback address. A request with no client address is refused. `X-Forwarded-For` is ignored because the server runs with `proxy_headers=False`. The setting `api_allow_remote=true` lifts only this rule and is unsafe without your own authentication | 403 `local_only` |
| Host allowlist | The `Host` header must be exactly `127.0.0.1`, `localhost` or `[::1]`, with an optional numeric port. This holds even when remote access is allowed | 400 `bad_host` |
| Anti-CSRF header | `POST`, `PUT`, `PATCH` and `DELETE` must send `X-FinAgent-Client: 1`. `GET` does not need it | 403 `csrf_header_missing` |
| Origin | If an `Origin` header is present it must be `http://<Host>` or `https://<Host>`, using the value of the `Host` header. Clients that send no `Origin` (curl, scripts) are not affected | 403 `origin_mismatch` |
| No CORS, no WebSocket | No CORS headers are sent. WebSocket connections under `/api` are closed with code 1008 | |
| Anti-framing | Responses under `/api/*`, `/` and `/app/*` carry `X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'` and `X-Content-Type-Options: nosniff` | |
| Chat service available | If the chat runtime failed to start, every `/api/*` route answers 503. `/answer` keeps working | 503 `chat_unavailable` |

## Errors and conventions

- **Error envelope.** Errors are `{"error": "<code>", "message": "<text>"}`, with the message in English. Some add a field: `fields` (a list of `{loc, msg}`, on 422) or `field` (on 400 `invalid_value`). An error never contains exception text, the input that was sent, or a secret.
- **Unexpected failures** on any route answer 500 `internal_error` with a fixed message. They are logged by exception type only.
- **Where the envelope does not apply.**
    - Confirm, reject and refresh answer with their result body (`status`, `message`, `ticket`, `orders`, `new_ticket`) for every outcome, including 404, 409, 410, 423 and 502. See [Status mapping](#status-mapping).
    - A path or method that does not exist gets FastAPI's default `{"detail": "Not Found"}` or `{"detail": "Method Not Allowed"}`.
    - The 502 of `GET /api/portfolio` has `connected: true` besides `error` and `message`.
- **Money is a decimal string** (`"500.00"`, `"0.5"`), never a JSON float. Quantities, prices, costs and percentages follow the same rule. Mandate money may also be sent as a JSON number, which is parsed exactly, but strings are preferred.
- **Ids.** Session ids are `S` plus 12 characters, ticket ids `T` plus 10, evidence ids `E1`, `E2`, ... per session. Ids in a path may be at most 64 characters.
- **Request bodies** must be JSON with `Content-Type: application/json`. Unknown fields are rejected with 422 `invalid_request`.
- **Timestamps** are ISO 8601 and carry `Z` or a `+00:00` offset.

## Endpoint overview

| Method | Path | Purpose | Success |
|---|---|---|---|
| `POST` | `/api/sessions` | Create a chat session | 201 |
| `GET` | `/api/sessions` | List sessions | 200 |
| `GET` | `/api/sessions/{session_id}` | One session with its recent messages, active ticket and mandate | 200 |
| `PATCH` | `/api/sessions/{session_id}` | Rename a session | 200 |
| `POST` | `/api/chat` | Send a message; the reply streams as Server-Sent Events | 200 |
| `GET` | `/api/portfolio` | Paper account, positions and open orders | 200 |
| `GET` | `/api/mandate/{session_id}` | Read the session's mandate | 200 |
| `PUT` | `/api/mandate/{session_id}` | Save the session's mandate | 200 |
| `GET` | `/api/tickets/{ticket_id}?session_id=...` | A ticket with its orders | 200 |
| `POST` | `/api/tickets/{ticket_id}/confirm` | Confirm a ticket. **The only call that can place an order** | 200 |
| `POST` | `/api/tickets/{ticket_id}/reject` | Reject a ticket | 200 |
| `POST` | `/api/tickets/{ticket_id}/refresh` | Re-quote a ticket | 200 |
| `GET` | `/api/settings` | Connection status; never secrets | 200 |
| `PUT` | `/api/settings` | Save keys or the trading flag | 200 |
| `POST` | `/api/kill-switch` | Turn trading off | 200 |
| `GET` | `/api/ledger/{session_id}.jsonl` | Download a session's audit ledger | 200 |

## Sessions

**`POST /api/sessions`** creates a session. The body is optional.

| Field | Type | Default | Notes |
|---|---|---|---|
| `title` | string, at most 200 characters | `New chat` | Trimmed and cut to 80 characters; a blank title becomes `New chat` |
| `lang` | `en` or `ko` | `en` | Stored with the session |

```json
{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "created_at": "2026-10-02T10:07:53.497995+00:00", "lang": "en"}
```

Status codes: 201, 422.

**`GET /api/sessions`** returns the sessions, newest first. `updated_at` is the time of the latest message, or the creation time if there is none.

```json
{"sessions": [{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "lang": "en", "created_at": "2026-10-02T10:07:53+00:00", "updated_at": "2026-10-02T10:12:40+00:00"}]}
```

**`GET /api/sessions/{session_id}`** returns one session. Stale tickets are expired first.

| Field | Meaning |
|---|---|
| `session` | `{id, title, lang, created_at}` |
| `messages` | The last 50 messages of any role (`user`, `assistant`, `event`), oldest first: `{role, content, meta, created_at}`. `meta` is a JSON object or `null`. `event` rows are per-turn summaries without tool output |
| `active_ticket` | The newest unexpired `proposed` ticket, or `null` |
| `mandate` | The saved mandate, or the defaults if none was saved |
| `mandate_saved` | `true` if a mandate was saved for this session |

Status codes: 200, 404 `session_not_found`.

**`PATCH /api/sessions/{session_id}`** renames a session. Body: `{"title": "..."}` (at most 200 characters). Returns the session `{id, title, lang, created_at}`. Status codes: 200, 404 `session_not_found`, 422.

## Chat (SSE)

**`POST /api/chat`** runs one agent turn and streams it.

Request body:

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | An existing session. Unknown: HTTP 404 `session_not_found`, before any stream starts |
| `message` | string | At most 16,000 characters at this endpoint (422 above that). The agent itself refuses an empty message or one over 8,000 characters, and a message sent while another reply is still running in the same session. Those refusals are part of the stream (HTTP 200): one `error` event and one `finish` event |

Response: `200 text/event-stream` with `Cache-Control: no-cache` and `X-Accel-Buffering: no`. Each event is one line, `data: <JSON>`, followed by a blank line. A comment line `: keep-alive` is sent after 10 seconds of silence and carries no data. The stream ends after `finish`. If the client disconnects, the turn is cancelled. Use `fetch` with a streamed body or `curl -N`; `EventSource` cannot send a POST.

Status codes: 200 (stream), 404 `session_not_found`, 422 `invalid_request`, 503 `chat_unavailable`.

### Events

| `type` | Fields | When | Example |
|---|---|---|---|
| `start` | `messageId` | First event of a turn. The id identifies the turn | `data: {"type":"start","messageId":"6f1c0e9d2b7a4c58a3e1d94b7c2f8a10"}` |
| `reasoning-delta` | `delta` | The model wrote text in a round that also calls tools (interim narration) | `data: {"type":"reasoning-delta","delta":"Checking the fund's cost and holdings first."}` |
| `tool-input-available` | `toolCallId`, `toolName`, `input` | Before a tool runs. A tool name that is not in the current tool set appears as `unknown_tool` | `data: {"type":"tool-input-available","toolCallId":"call_1","toolName":"get_quotes","input":{"symbols":["VOO"]}}` |
| `tool-output-available` | `toolCallId`, `output`, `isError`, `sources` | After a tool ran. `sources` is a list of `{id, title, url, publisher, tier}` (`url` may be `null`). A tool cancelled at the time budget has `isError: true` and no sources | `data: {"type":"tool-output-available","toolCallId":"call_1","output":"[E1] Latest quotes ...","isError":false,"sources":[{"id":"E1","title":"Alpaca IEX quotes: VOO","url":null,"publisher":"Alpaca Markets (IEX feed)","tier":1}]}` |
| `source-url` | `sourceId`, `url`, `title` | Once per source with a URL, the first time it appears in a turn | `data: {"type":"source-url","sourceId":"E2","url":"https://www.sec.gov/ixviewer/...","title":"VOO annual report"}` |
| `source-grounding` | `search_suggestions_html` | After `grounded_web_answer` returned Google's "Search Suggestions" HTML. Show it unmodified | `data: {"type":"source-grounding","search_suggestions_html":"<div class=\"container\">...</div>"}` |
| `data-mandate` | `mandate`, `draft` | After `draft_mandate`. `draft` is always `true`: the draft is not in effect until the user saves it with `PUT /api/mandate/{session_id}` | `data: {"type":"data-mandate","mandate":{"budget_usd":"2000","max_position_pct":"20", ...},"draft":true}` |
| `data-trade-proposal` | `ticket` | After `propose_trades` created a ticket. The object is the [ticket](#ticket-object) | `data: {"type":"data-trade-proposal","ticket":{"id":"T4HQ7ZB2NAK","status":"proposed", ...}}` |
| `data-grounding-flags` | `numbers` | After the final text is composed, if some numbers in it are not supported by this turn's tool outputs. Advisory only | `data: {"type":"data-grounding-flags","numbers":["$1,250","3.4%"]}` |
| `text-delta` | `delta` | The final answer, in chunks of at most about 200 characters. The whole text is composed first, so this is chunked delivery, not token streaming | `data: {"type":"text-delta","delta":"VOO tracks the S&P 500 [E1]. ..."}` |
| `error` | `message` | The turn was refused or failed. The message is generic, in the language of the user's message, and never carries provider text | `data: {"type":"error","message":"Sorry, something went wrong while writing this reply. Please try again in a moment."}` |
| `finish` | `usage`, `elapsed_ms` | Always the last event, also after an error. `usage` holds token counts summed over the turn, as the provider reports them (for example `input` and `output`) | `data: {"type":"finish","usage":{"input":4210,"output":380},"elapsed_ms":18342}` |

Order within a turn: `start`; then for each round an optional `reasoning-delta` and, for every tool call in order, `tool-input-available`, `tool-output-available`, any `source-url` events and any events the tool produced (`data-mandate`, `data-trade-proposal`, `source-grounding`); then an optional `data-grounding-flags`, the `text-delta` chunks and `finish`. A refusal or failure is `error` followed by `finish`. Order status updates are not pushed on the stream: poll `GET /api/tickets/{ticket_id}`.

## Portfolio

**`GET /api/portfolio`** reads the paper account live from the broker. It has three outcomes.

| Case | Status | Body |
|---|---|---|
| No Alpaca keys saved | 200 | `{"connected": false}` |
| Connected | 200 | `{"connected": true, "paper": true, "account": {...}, "positions": [...], "open_orders": [...]}` |
| Broker unreachable | 502 | `{"connected": true, "error": "broker_unavailable", "message": "..."}` |

```json
{
  "connected": true,
  "paper": true,
  "account": {"status": "ACTIVE", "equity": "100000", "cash": "100000", "buying_power": "100000", "non_marginable_buying_power": "100000", "last_equity": "100000", "trading_blocked": false, "account_blocked": false, "trade_suspended_by_user": false, "shorting_enabled": false},
  "positions": [{"symbol": "VOO", "qty": "4", "qty_available": "4", "avg_entry_price": "500.00", "current_price": "501.20", "market_value": "2004.80", "unrealized_pl": "4.80", "unrealized_plpc": "0.0024", "side": "long"}],
  "open_orders": []
}
```

`open_orders` holds up to 100 open orders in the [order object](#order-object) shape. The 502 case uses the error code `broker_unavailable`.

## Mandate

The mandate is the set of limits every proposal and every confirmation is checked against.

**`GET /api/mandate/{session_id}`** returns `{"mandate": {...}, "saved": true|false}`. `saved` is `false` when the defaults are returned. Status codes: 200, 404 `session_not_found`.

**`PUT /api/mandate/{session_id}`** saves the mandate. The body is a JSON object of mandate fields (at most 32,768 bytes, else 413 `too_large`). The server assigns `version` (previous version plus 1; the first save is 1) and ignores any `version` in the body. Each save also appends a `mandate_saved` ledger entry. Returns `{"mandate": {...}, "saved": true}`. Status codes: 200, 404 `session_not_found`, 413 `too_large`, 422 `invalid_request` (with `fields`).

| Field | Type | Default | Rule |
|---|---|---|---|
| `direction_text` | string | `""` | The user's direction in plain words |
| `budget_usd` | decimal string or `null` | `null` (no cap) | Greater than 0 |
| `universe` | string | `us_equity` | Only `us_equity` |
| `allow_symbols` | array of tickers | `[]` | Upper-cased and de-duplicated. Empty means any |
| `deny_symbols` | array of tickers | `[]` | Upper-cased and de-duplicated |
| `max_position_pct` | decimal string | `20` | Greater than 0 and at most 100 |
| `max_positions` | integer | `8` | At least 1 |
| `cash_buffer_pct` | decimal string | `5` | 0 to 100 |
| `horizon` | string | `weeks to months` | Free text |
| `order_types` | array | `["limit","market"]` | Not empty; each `limit` or `market` |
| `max_orders_per_day` | integer | `10` | At least 1 |
| `max_order_usd` | decimal string or `null` | `null` (no cap) | Greater than 0 |
| `price_band_pct` | decimal string | `3` | Greater than 0 and at most 100 |
| `allow_fractional` | boolean | `true` | |
| `version` | integer | `1` | Read-only: set by the server |

```json
{"mandate": {"direction_text": "", "budget_usd": "20000", "universe": "us_equity", "allow_symbols": ["VOO", "SCHD"], "deny_symbols": [], "max_position_pct": "10", "max_positions": 8, "cash_buffer_pct": "5", "horizon": "weeks to months", "order_types": ["limit", "market"], "max_orders_per_day": 10, "max_order_usd": null, "price_band_pct": "3", "allow_fractional": true, "version": 1}, "saved": true}
```

## Tickets

A ticket is the server's record of a proposal. It is created only by the agent's `propose_trades` tool, and it is read, confirmed, rejected or refreshed through the endpoints below. Every ticket action expires stale tickets first.

### Ticket object

| Field | Meaning |
|---|---|
| `id` | `T` plus 10 characters |
| `session_id` | The session that owns it. Any other session gets `not_found` |
| `status` | `proposed`, `submitting`, `submitted`, `rejected_by_user`, `expired`, `superseded` or `failed` |
| `legs` | The executable legs only: `{symbol, side, qty, notional, type, time_in_force, limit_price}`. `side` is `buy` or `sell`, `type` is `market` or `limit`, `time_in_force` is `day` or `gtc`; `qty`, `notional` and `limit_price` are strings or `null` |
| `decisions` | One risk decision per proposed leg, rejected legs included (for display only): `{index, verdict, leg, original, reasons, est_price, est_cost_usd}`. `verdict` is `accept`, `clamp` or `reject`; `original` is the leg as proposed; each reason is `{code, message, severity}` with `severity` `warn` or `block` |
| `thesis`, `risks`, `invalidation` | The model's text |
| `evidence_ids` | The evidence ids it cited |
| `mandate_version` | Version of the mandate in force at proposal time |
| `quote_as_of` | Time of the oldest quote the estimates use |
| `created_at`, `expires_at` | Creation time, and creation plus 300 seconds |
| `hash` | 64 hexadecimal characters; send it back with `confirm` |
| `client_order_ids` | One per executable leg, `fa-<ticket id>-<n>` |
| `broker_order_ids` | Empty until the ticket is submitted; then one entry per leg, `null` where no order is known |

```json
{
  "id": "T4HQ7ZB2NAK",
  "session_id": "S3F7QKX2M4ABC",
  "status": "proposed",
  "legs": [{"symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"}],
  "decisions": [{
    "index": 0,
    "verdict": "clamp",
    "leg": {"symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"},
    "original": {"symbol": "VOO", "side": "buy", "qty": "10", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"},
    "reasons": [{"code": "max_position_pct", "message": "Clamped qty 10 → 4: VOO position $0.00 + estimated cost $5,000.00 exceeds 10% of budget $20,000.00 ($2,000.00)", "severity": "warn"}],
    "est_price": "500.00",
    "est_cost_usd": "2000.00"
  }],
  "thesis": "Core US equity exposure [E1][E2].",
  "risks": "Equity drawdowns.",
  "invalidation": "The thesis fails if ...",
  "evidence_ids": ["E1", "E2"],
  "mandate_version": 1,
  "quote_as_of": "2026-10-02T14:31:05Z",
  "created_at": "2026-10-02T14:31:20Z",
  "expires_at": "2026-10-02T14:36:20Z",
  "hash": "eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c",
  "client_order_ids": ["fa-T4HQ7ZB2NAK-0"],
  "broker_order_ids": []
}
```

### Order object

Orders are Alpaca order snapshots: `{id, client_order_id, symbol, side, qty, notional, type, time_in_force, limit_price, status, filled_qty, filled_avg_price, submitted_at, updated_at}`. Numbers are strings or `null`. An order status is final when it is `filled`, `canceled`, `expired`, `rejected`, `replaced` or `done_for_day`.

### Get a ticket

**`GET /api/tickets/{ticket_id}?session_id=...`** returns the ticket and its orders. This is the call to poll after a confirm: with a broker, every non-final order is re-read from Alpaca (reads only, never a submit) and the local copy is updated. Without a broker it returns the stored rows.

```json
{"ticket": {"id": "T4HQ7ZB2NAK", "status": "submitted", "...": "..."}, "orders": [{"client_order_id": "fa-T4HQ7ZB2NAK-0", "broker_order_id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "order": {"id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "client_order_id": "fa-T4HQ7ZB2NAK-0", "symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00", "status": "accepted", "filled_qty": "0", "filled_avg_price": null, "submitted_at": "2026-10-02T14:32:01Z", "updated_at": "2026-10-02T14:32:01Z"}}]}
```

`order` is `null` for an uncertain leg (no broker id is known): tell the user to check open orders. Status codes: 200, 404 `ticket_not_found` (unknown id, or a ticket of another session), 422 (`session_id` missing).

### Confirm, reject, refresh

| Call | Body | What it does |
|---|---|---|
| `POST /api/tickets/{ticket_id}/confirm` | `{"session_id": "...", "hash": "<ticket.hash>"}` | Re-checks the ticket and, if nothing material changed, submits its orders. **The only call that can place an order.** Needs a broker and `trading_enabled` |
| `POST /api/tickets/{ticket_id}/reject` | `{"session_id": "..."}` | Moves a `proposed` ticket to `rejected_by_user`. Works without a broker. Rejecting an already rejected ticket is answered with 200 |
| `POST /api/tickets/{ticket_id}/refresh` | `{"session_id": "..."}` | Re-quotes a `proposed` ticket at current prices and replaces it with a new ticket. Never submits. Needs a broker |

All three answer with the same result body:

| Field | Meaning |
|---|---|
| `status` | `submitted`, `requote`, `rejected`, `disabled`, `expired`, `not_found` or `error` |
| `message` | A short English summary, passed through secret redaction. For `submitted` it lists each leg's outcome |
| `ticket` | The ticket as it stands now, or `null` |
| `orders` | The orders that were placed (order objects), else `[]` |
| `new_ticket` | The replacement ticket for `requote`, else `null` |

```json
{"status": "submitted", "message": "1 of 1 orders placed. Leg 1 BUY 4 VOO @ limit $500.00: placed (status accepted).", "ticket": {"id": "T4HQ7ZB2NAK", "status": "submitted", "...": "..."}, "orders": [{"id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "status": "accepted", "...": "..."}], "new_ticket": null}
```

Before the executor runs, these can happen instead:

| Status | Code | When |
|---|---|---|
| 503 | `broker_not_connected` | Confirm or refresh without saved Alpaca keys. Reject still works |
| 422 | `invalid_request` | The body is missing a field, for example `hash` |
| 400 | `invalid_request` | Refresh of a ticket that cannot be re-quoted |
| 502 | `refresh_failed` | Refresh failed unexpectedly |
| 500 | `confirm_failed` | Confirm failed unexpectedly. The message tells the user to check open orders before trying again |
| 500 | `reject_failed` | Reject failed unexpectedly |

### Status mapping

The HTTP status of a ticket action follows the result `status`. "n/a" means that action never produces that result.

| Result `status` | Meaning | confirm | reject | refresh |
|---|---|---|---|---|
| `submitted` | Orders were sent and the outcome was saved | 200 | n/a | n/a |
| `requote` | Nothing was submitted. A new ticket (in `new_ticket`) replaced this one because the price, a verdict or a leg changed | 409 | n/a | 200 |
| `disabled` | The kill switch is on. Nothing changed | 423 | n/a | n/a |
| `expired` | The ticket was past its expiry when confirm ran | 410 | n/a | n/a |
| `not_found` | Unknown id, or a ticket of another session | 404 | 404 | 404 |
| `rejected`, ticket is `rejected_by_user` | The user had already rejected it | 409 | 200 | 409 |
| `rejected`, ticket is `expired` | Expired before the action | 410 | 410 | 410 |
| `rejected`, any other case | Hash mismatch, ticket not `proposed`, claimed by another confirm, or no leg executable at current prices | 409 | 409 | 409 |
| `error` | The re-check failed, every leg failed, the outcome could not be saved, or the ticket could not be read. If the outcome could not be saved, the message says orders may have been placed: check open orders before trying again | 502 | 502 | 502 |

## Settings

**`GET /api/settings`** returns connection status. It never returns a secret.

| Field | Values |
|---|---|
| `alpaca` | `connected`, `missing` (no keys), `invalid` (Alpaca rejected the keys) or `unreachable` |
| `tavily`, `sec_user_agent`, `gemini` | `set` or `missing` |
| `trading_enabled` | boolean; `false` after the kill switch |
| `model`, `effort` | The LLM model name and reasoning effort |
| `api_allow_remote` | boolean |
| `env_fallbacks` | Names of writable settings (for example `TAVILY_API_KEY`) that only the process environment provides. `.env` wins over the environment, so clearing one of these in the app does not survive a restart |

```json
{"alpaca": "connected", "tavily": "missing", "sec_user_agent": "set", "gemini": "set", "trading_enabled": true, "model": "gemini-3.8-flash", "effort": "medium", "api_allow_remote": false, "env_fallbacks": []}
```

**`PUT /api/settings`** saves keys to the local `.env` and the trading flag. Every field is optional:

| Field | Type | Notes |
|---|---|---|
| `alpaca_key_id`, `alpaca_secret` | string, at most 256 characters | Must be sent together. They are checked live against the paper host before anything is saved |
| `tavily_key` | string, at most 256 characters | |
| `sec_user_agent` | string, at most 256 characters | A name and contact address, as the SEC requires |
| `trading_enabled` | boolean | The only way to turn trading back on after the kill switch |

For the string fields, `""` removes the key and `null` (or leaving the field out) leaves it as it is. Values are trimmed of ASCII whitespace and must be printable ASCII without `"`, `\` or `${`. Saving rebuilds the connectors in place, so the agent's tool set changes from the next message on. The response is the same body as `GET /api/settings`.

| Status | Code | When |
|---|---|---|
| 200 | | Saved |
| 400 | `invalid_value` | A value has characters that cannot be saved; `field` names it |
| 400 | `alpaca_pair_required` | Only one of the Alpaca key id and secret was given (or one was cleared and not the other) |
| 400 | `alpaca_invalid` | Alpaca rejected the keys; nothing was saved |
| 502 | `alpaca_unreachable` | Alpaca could not be reached to check the keys; nothing was saved |
| 500 | `settings_not_saved` | Writing the settings failed |
| 422 | `invalid_request` | Unknown field or wrong type |

## Kill switch

**`POST /api/kill-switch`** turns trading off. Body (optional): `{"suspend_broker": false}`.

It sets `trading_enabled` to `false` at once, so an in-flight confirm submits no further legs, and writes `TRADING_ENABLED=false` to `.env`. With `suspend_broker: true` and a broker connected, it also asks Alpaca to suspend trading on the paper account and cancels the open orders. While trading is disabled, confirm answers 423 `disabled`; research and proposals still work. Turn it back on with `PUT /api/settings {"trading_enabled": true}`.

```json
{"trading_enabled": false, "suspend_broker": false, "broker_connected": true, "persisted": true}
```

`persisted` is `false` if `.env` could not be written (the switch is still off in memory). Status codes: 200, 422. The kill switch is recorded in the ledger under the session id `system`.

## Ledger export

**`GET /api/ledger/{session_id}.jsonl`** downloads the session's audit ledger as `application/x-ndjson` with a `Content-Disposition: attachment; filename="ledger-<session_id>.jsonl"` header. Each line is a JSON object:

```json
{"id": 12, "session_id": "S3F7QKX2M4ABC", "kind": "claim", "created_at": "2026-10-02T14:32:00.512345+00:00", "payload": {"ticket_id": "T4HQ7ZB2NAK", "legs": 1}}
```

`kind` is one of `mandate_saved`, `confirm_request`, `claim`, `requote`, `submit`, `broker_response`, `reconcile`, `order_status`, `result`, `reject`, `refresh` or `recover`. Payloads are redacted before they are stored, so secrets never appear. Status codes: 200, 404 `session_not_found`. Kill-switch entries belong to the session id `system`, which is not a session, so they cannot be exported here.

## Example: from a message to a confirmed ticket

This walk-through uses a POSIX shell (on Windows, Git Bash). The ids, the hash and the prices are placeholders: replace them with the values in your own responses. `curl -w` prints the HTTP status at the end. Never paste a real key into an example or a script you share.

**1. Create a session.**

```bash
curl -s -w '\nHTTP %{http_code}\n' -X POST http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"title": "Core ETF sleeve", "lang": "en"}'
```

```text
{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "created_at": "2026-10-02T14:30:02.114233+00:00", "lang": "en"}
HTTP 201
```

**2. Send a message and read the stream** (`-N` turns off curl's buffering so events appear as they arrive).

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"session_id": "S3F7QKX2M4ABC", "message": "Build a $2,000 core position in a broad US equity ETF, moderate risk."}'
```

```text
data: {"type":"start","messageId":"6f1c0e9d2b7a4c58a3e1d94b7c2f8a10"}

data: {"type":"tool-input-available","toolCallId":"call_1","toolName":"get_quotes","input":{"symbols":["VOO"]}}

data: {"type":"tool-output-available","toolCallId":"call_1","output":"[E1] Latest quotes ...","isError":false,"sources":[{"id":"E1","title":"Alpaca IEX quotes: VOO","url":null,"publisher":"Alpaca Markets (IEX feed)","tier":1}]}

data: {"type":"data-trade-proposal","ticket":{"id":"T4HQ7ZB2NAK","status":"proposed","hash":"eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c", ...}}

data: {"type":"text-delta","delta":"I propose buying 4 shares of VOO with a limit order ... [E1]. Review the ticket and confirm or reject it."}

data: {"type":"finish","usage":{"input":4210,"output":380},"elapsed_ms":18342}
```

The `data-trade-proposal` event holds the ticket. Keep its `id` and `hash`. (The proposal needs saved Alpaca paper keys; without them the agent can research but has no `propose_trades` tool.)

**3. Read the ticket.** A `GET` needs no `X-FinAgent-Client` header.

```bash
curl -s "http://127.0.0.1:8000/api/tickets/T4HQ7ZB2NAK?session_id=S3F7QKX2M4ABC"
```

**4. Confirm it.** The `hash` must be the one from the ticket you reviewed.

```bash
curl -s -w '\nHTTP %{http_code}\n' -X POST http://127.0.0.1:8000/api/tickets/T4HQ7ZB2NAK/confirm \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"session_id": "S3F7QKX2M4ABC", "hash": "eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c"}'
```

```text
{"status": "submitted", "message": "1 of 1 orders placed. Leg 1 BUY 4 VOO @ limit $500.00: placed (status accepted).", "ticket": {...}, "orders": [{...}], "new_ticket": null}
HTTP 200
```

If the price moved, you get `HTTP 409` with `"status": "requote"` and the replacement ticket in `new_ticket`: review it and confirm again with the new `id` and `hash`. After a `submitted` result, repeat step 3 every few seconds until no order is in a non-final status.

---

## 한국어

[English](#evaluation-api-specification) | 한국어

# 평가용 API 명세

## Endpoint

| 항목 | 값 |
|---|---|
| Method / Path | `GET /answer` |
| Content-Type (응답) | `application/json; charset=utf-8` |
| 인증 | 없음 (인증 헤더·POST 바디 미사용) |
| 응답 시간 | 주최측 기준: **300초 초과 시 미응답 처리**, 그 안이면 늦어도 채점(점수 차등). 서버 내부 예산 120초, 초과 시 확보된 근거로 즉시 답변 |
| 답변 생성 LLM | Google Gemini (`LLM_PROVIDER=gemini`, 기본 `LLM_MODEL=gemini-3.8-flash`, Gemini의 OpenAI 호환 엔드포인트로 호출). Anthropic·OpenAI는 선택 가능한 대체 제공자로 남아 있습니다. 대회 제출본은 대회 규정에 따라 NCP HyperCLOVA X (`HCX-005`, CLOVA Studio OpenAI 호환 엔드포인트)를 사용했습니다 |
| 답변 언어 | `answer`는 질문의 언어를 따릅니다: 기본은 영어, 질문이 한국어로 작성되면 한국어. 영어 질문 안에 한국어 상품명·기업명이 들어 있어도 영어 질문으로 봅니다. `think_trace`·`retrieved_context`의 라벨은 항상 영어입니다 |
| 보조 | `GET /health` (상태·테이블 기준일·색인 규모·그래프 명시/추론 트리플 수·벡터 백엔드), `GET /` (안내 문구) |
| 참조 데이터 | SQLite 뷰 5개(`v_bond_kr` · `v_etf_kr` · `v_etf_gl` · `v_fund_pub` · `v_etf_constituent`) · 상품 53,375행 + 편입종목 22,599행 · 지식그래프 1,082,874 트리플 |

## Request (Query String)

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question_id` | string | 필수 | 주최측 문항 ID (예: `Q-001`). 그대로 반환 |
| `question` | string | 필수 | 평가 질의 원문 (URL 인코딩) |

- 정의되지 않은 파라미터는 무시됩니다 (500 없이 처리).
- `question`이 비어 있으면 200 OK + 동일 스키마로 안내 메시지를 반환합니다.

```bash
curl -G "https://{endpoint}/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘"
```

```python
import requests
requests.get("https://{endpoint}/answer", params={"question_id": "Q-001", "question": "평가질의"}).json()
```

## Response (200 OK, JSON)

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question_id` | string | 필수 | 요청값 그대로 |
| `question` | string | 필수 | 요청값 그대로 |
| `retrieved_context` | string | 필수 | 답변 근거. `[근거 n]` 블록마다 참조 테이블(한글명·뷰명)·데이터 기준일·**건수**·실행한 SQL/SPARQL/검색어·참조 상품코드와 상품명. 외부 문서를 쓴 경우 문서 제목·출처.<br>건수는 **필터에 걸린 전체 건수**이며, 표시 행이 200행 상한에 잘린 경우 "전체 N건 (표시 200건)"으로 구분해 적습니다 |
| `think_trace` | string | 필수 | 번호가 매겨진 처리 과정: 사전분석(엔티티·테마·온톨로지 검증) → 각 라운드의 도구 호출과 결과 크기 → 최종 상태·소요 시간 |
| `answer` | string | 필수 | 최종 답변. 질문의 언어를 따릅니다(기본 영어, 한국어로 쓴 질문이면 한국어). 확인 불가 시 고정 문장으로 시작(한국어 "제공된 데이터로는 확인할 수 없습니다", 영어 "This cannot be confirmed with the provided data."), 조건 모호 시 역질문(물음표로 끝남) |

### JSON Schema

```json
{
  "type": "object",
  "required": ["question_id", "question", "retrieved_context", "think_trace", "answer"],
  "properties": {
    "question_id": {"type": "string"},
    "question": {"type": "string"},
    "retrieved_context": {"type": "string"},
    "think_trace": {"type": "string"},
    "answer": {"type": "string"}
  }
}
```

> 아래 예시는 대회 제출본(HyperCLOVA X)의 출력 형식입니다. 영어판에서는 `retrieved_context`·`think_trace`의 라벨이 영어로 나옵니다(예: `[근거 1]` → `[Evidence 1]`, `사전분석` → `Pre-analysis`, `참조한 데이터 없음` → `No data referenced`). 예시 질문이 한국어이므로 `answer`는 영어판에서도 한국어입니다. 영어로 질문한 경우의 예시는 위 English 섹션을 참고하세요.

### 응답 예시

```json
{
  "question_id": "Q-001",
  "question": "현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘",
  "retrieved_context": "[근거 1] 국내채권마스터(v_bond_kr) · 기준일 2026-08-21 · 전체 15,837건 (표시 200건)\n  sql: SELECT DISTINCT pd_no, pd_nm, crd_grd, buy_yield, after_tax_yield, mat_date FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-') ORDER BY buy_yield DESC LIMIT 200\n  참조: KR350801G954 한국가스공사 394 | KR6056554B20 SK이노베이션10-4 | ...",
  "think_trace": "1. 사전분석: 상품언급 없음; 신용등급 'AA-' 유효 → crd_grd IN ('AAA','AA+','AA0','AA-')\n2. 라운드 1: sql_query(...) → 결과 3120자, 전체 15,837건 중 200건 표시\n3. 라운드 2: final_answer(status=answered)\n4. 최종: status=answered, 근거 1건, 소요 14.2s",
  "answer": "현재 구매 가능한 원화 채권 중 신용등급 AA- 이상 종목은 총 15,837건입니다 (국내채권마스터, 기준일 2026-08-21). 매수수익률이 수록된 종목(has_buy_yield=1) 기준 상위: ... (근거: 국내채권마스터 · 2026-08-21)"
}
```

### 확인 불가 질의 예시 (동일 스키마, 200 OK)

```json
{
  "question_id": "Q-031",
  "question": "신용등급 AAAA인 채권 찾아줘",
  "retrieved_context": "참조한 데이터 없음",
  "think_trace": "1. 사전분석: 상품언급 없음; 온톨로지 위반 1건\n2. 라운드 1: final_answer(status=cannot_confirm)\n3. 최종: status=cannot_confirm, 근거 0건, 소요 6.1s",
  "answer": "제공된 데이터로는 확인할 수 없습니다. 신용등급 'AAAA'는 존재하지 않는 값입니다. 유효한 신용등급: AAA, AA+, AA0, AA-, A+, ... D"
}
```

## 오류 처리

- 내부 오류가 발생해도 HTTP 500을 반환하지 않고 200 + 동일 스키마(`answer`에 재시도 안내, `think_trace`에 오류 유형)로 응답합니다.
- LLM 호출 실패·시간 초과 시에도 확보된 근거를 `retrieved_context`에 담아 반환합니다.

---

# 채팅 및 트레이딩 API (`/api/*`)

채팅 및 트레이딩 API는 로컬 모의투자 앱(`web/app`)이 쓰는 HTTP 인터페이스입니다. 대회 제출 규약에는 포함되지 않습니다. `GET /answer`, `GET /answer/stream`, `GET /health`, `/ui`는 바뀌지 않았으며 아래 규칙이 적용되지 않습니다. 예시는 서버가 `http://127.0.0.1:8000`에서 실행 중이라고 가정합니다. 동작 원리는 [architecture.md](architecture.md)에 있습니다.

## 접근 규칙

| 규칙 | 내용 | 거부 응답 |
|---|---|---|
| 루프백 클라이언트만 허용 | 소켓 상대 주소가 `127.0.0.0/8`, `::1`, 또는 IPv4 매핑 루프백이어야 합니다. 클라이언트 주소가 없는 요청도 거부합니다. 서버를 `proxy_headers=False`로 실행하므로 `X-Forwarded-For`는 무시합니다. 설정 `api_allow_remote=true`는 이 규칙만 풀며, 별도 인증이 없으면 안전하지 않습니다 | 403 `local_only` |
| Host 허용 목록 | `Host` 헤더가 정확히 `127.0.0.1`, `localhost`, `[::1]` 중 하나여야 하며 숫자 포트는 붙을 수 있습니다. 원격 접근을 허용해도 이 규칙은 유지됩니다 | 400 `bad_host` |
| CSRF 방지 헤더 | `POST`, `PUT`, `PATCH`, `DELETE`는 `X-FinAgent-Client: 1`을 보내야 합니다. `GET`에는 필요하지 않습니다 | 403 `csrf_header_missing` |
| Origin | `Origin` 헤더가 있으면 `Host` 헤더 값을 사용한 `http://<Host>` 또는 `https://<Host>`여야 합니다. `Origin`을 보내지 않는 클라이언트(curl, 스크립트)는 영향을 받지 않습니다 | 403 `origin_mismatch` |
| CORS·WebSocket 없음 | CORS 헤더를 보내지 않습니다. `/api` 아래 WebSocket 연결은 코드 1008로 닫습니다 | |
| 프레이밍 방지 | `/api/*`, `/`, `/app/*` 응답에 `X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`가 붙습니다 | |
| 채팅 서비스 사용 가능 | 채팅 런타임이 시작에 실패하면 모든 `/api/*` 라우트가 503으로 답합니다. `/answer`는 계속 동작합니다 | 503 `chat_unavailable` |

## 오류와 공통 규칙

- **오류 형식.** 오류는 `{"error": "<code>", "message": "<text>"}`이며 메시지는 영어입니다. 일부는 필드를 더 담습니다. `fields`(422일 때 `{loc, msg}` 목록) 또는 `field`(400 `invalid_value`일 때)입니다. 오류에는 예외 문구, 보낸 입력값, 비밀값이 들어가지 않습니다.
- **예기치 못한 실패**는 어느 라우트에서든 고정된 메시지와 함께 500 `internal_error`로 답합니다. 로그에는 예외 유형만 남깁니다.
- **이 형식을 쓰지 않는 경우.**
    - confirm, reject, refresh는 404, 409, 410, 423, 502를 포함한 모든 결과에서 결과 본문(`status`, `message`, `ticket`, `orders`, `new_ticket`)으로 답합니다. [상태 코드 매핑](#상태-코드-매핑)을 참고하세요.
    - 존재하지 않는 경로나 메서드는 FastAPI 기본 응답 `{"detail": "Not Found"}` 또는 `{"detail": "Method Not Allowed"}`를 받습니다.
    - `GET /api/portfolio`의 502에는 `error`, `message`와 함께 `connected: true`가 있습니다.
- **금액은 십진수 문자열입니다**(`"500.00"`, `"0.5"`). JSON 부동소수점 수는 쓰지 않습니다. 수량, 가격, 비용, 비율도 같은 규칙입니다. 투자 원칙의 금액은 JSON 숫자로 보내도 정확하게 파싱되지만 문자열을 권장합니다.
- **ID.** 세션 ID는 `S` 뒤에 12자, 티켓 ID는 `T` 뒤에 10자, 근거 ID는 세션마다 `E1`, `E2`, ...입니다. 경로의 ID는 최대 64자입니다.
- **요청 본문**은 `Content-Type: application/json`인 JSON이어야 합니다. 알 수 없는 필드는 422 `invalid_request`로 거부합니다.
- **시각**은 ISO 8601이며 `Z` 또는 `+00:00` 오프셋이 붙습니다.

## 엔드포인트 개요

| 메서드 | 경로 | 용도 | 성공 |
|---|---|---|---|
| `POST` | `/api/sessions` | 채팅 세션 생성 | 201 |
| `GET` | `/api/sessions` | 세션 목록 | 200 |
| `GET` | `/api/sessions/{session_id}` | 세션 하나와 최근 메시지, 활성 티켓, 투자 원칙 | 200 |
| `PATCH` | `/api/sessions/{session_id}` | 세션 이름 변경 | 200 |
| `POST` | `/api/chat` | 메시지 전송. 답변은 Server-Sent Events로 스트리밍 | 200 |
| `GET` | `/api/portfolio` | 모의 계좌, 보유 종목, 미체결 주문 | 200 |
| `GET` | `/api/mandate/{session_id}` | 세션의 투자 원칙 조회 | 200 |
| `PUT` | `/api/mandate/{session_id}` | 세션의 투자 원칙 저장 | 200 |
| `GET` | `/api/tickets/{ticket_id}?session_id=...` | 티켓과 주문 조회 | 200 |
| `POST` | `/api/tickets/{ticket_id}/confirm` | 티켓 확인. **주문을 낼 수 있는 유일한 호출** | 200 |
| `POST` | `/api/tickets/{ticket_id}/reject` | 티켓 거절 | 200 |
| `POST` | `/api/tickets/{ticket_id}/refresh` | 티켓 재호가(re-quote) | 200 |
| `GET` | `/api/settings` | 연결 상태. 비밀값은 절대 반환하지 않음 | 200 |
| `PUT` | `/api/settings` | 키 또는 거래 플래그 저장 | 200 |
| `POST` | `/api/kill-switch` | 거래 끄기 | 200 |
| `GET` | `/api/ledger/{session_id}.jsonl` | 세션의 감사 원장 내려받기 | 200 |

## 세션

**`POST /api/sessions`**는 세션을 만듭니다. 본문은 생략할 수 있습니다.

| 필드 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `title` | 문자열, 최대 200자 | `New chat` | 공백을 다듬고 80자로 자릅니다. 비어 있으면 `New chat`이 됩니다 |
| `lang` | `en` 또는 `ko` | `en` | 세션과 함께 저장됩니다 |

```json
{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "created_at": "2026-10-02T10:07:53.497995+00:00", "lang": "en"}
```

상태 코드: 201, 422.

**`GET /api/sessions`**는 세션을 최신순으로 돌려줍니다. `updated_at`은 가장 최근 메시지의 시각이며, 메시지가 없으면 생성 시각입니다.

```json
{"sessions": [{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "lang": "en", "created_at": "2026-10-02T10:07:53+00:00", "updated_at": "2026-10-02T10:12:40+00:00"}]}
```

**`GET /api/sessions/{session_id}`**는 세션 하나를 돌려줍니다. 먼저 만료된 티켓을 정리합니다.

| 필드 | 의미 |
|---|---|
| `session` | `{id, title, lang, created_at}` |
| `messages` | 역할(`user`, `assistant`, `event`)과 관계없이 최근 50개를 오래된 순으로: `{role, content, meta, created_at}`. `meta`는 JSON 객체 또는 `null`입니다. `event` 행은 도구 출력이 없는 턴 요약입니다 |
| `active_ticket` | 만료되지 않은 가장 최근의 `proposed` 티켓, 없으면 `null` |
| `mandate` | 저장된 투자 원칙, 저장된 것이 없으면 기본값 |
| `mandate_saved` | 이 세션에 투자 원칙이 저장돼 있으면 `true` |

상태 코드: 200, 404 `session_not_found`.

**`PATCH /api/sessions/{session_id}`**는 세션 이름을 바꿉니다. 본문: `{"title": "..."}`(최대 200자). 세션 `{id, title, lang, created_at}`를 돌려줍니다. 상태 코드: 200, 404 `session_not_found`, 422.

## 채팅(SSE)

**`POST /api/chat`**는 에이전트 턴 하나를 실행하고 스트리밍합니다.

요청 본문:

| 필드 | 타입 | 비고 |
|---|---|---|
| `session_id` | 문자열 | 존재하는 세션. 없으면 스트림이 시작되기 전에 HTTP 404 `session_not_found` |
| `message` | 문자열 | 이 엔드포인트에서는 최대 16,000자(초과 시 422). 에이전트 자체는 빈 메시지, 8,000자를 넘는 메시지, 같은 세션에서 다른 답변이 진행 중일 때 보낸 메시지를 거절합니다. 이 거절은 스트림의 일부이며(HTTP 200) `error` 이벤트 하나와 `finish` 이벤트 하나로 나옵니다 |

응답: `Cache-Control: no-cache`, `X-Accel-Buffering: no`가 붙은 `200 text/event-stream`. 이벤트는 한 줄의 `data: <JSON>` 뒤에 빈 줄이 오는 형태입니다. 10초간 이벤트가 없으면 데이터가 없는 주석 줄 `: keep-alive`를 보냅니다. 스트림은 `finish` 뒤에 끝납니다. 클라이언트 연결이 끊기면 턴이 취소됩니다. 스트리밍 본문을 읽는 `fetch` 또는 `curl -N`을 사용하세요. `EventSource`는 POST를 보낼 수 없습니다.

상태 코드: 200(스트림), 404 `session_not_found`, 422 `invalid_request`, 503 `chat_unavailable`.

### 이벤트

| `type` | 필드 | 발생 시점 | 예시 |
|---|---|---|---|
| `start` | `messageId` | 턴의 첫 이벤트. 이 ID가 턴을 식별합니다 | `data: {"type":"start","messageId":"6f1c0e9d2b7a4c58a3e1d94b7c2f8a10"}` |
| `reasoning-delta` | `delta` | 도구를 호출하는 라운드에서 모델이 쓴 텍스트(중간 설명) | `data: {"type":"reasoning-delta","delta":"Checking the fund's cost and holdings first."}` |
| `tool-input-available` | `toolCallId`, `toolName`, `input` | 도구가 실행되기 전. 현재 도구 목록에 없는 도구 이름은 `unknown_tool`로 나옵니다 | `data: {"type":"tool-input-available","toolCallId":"call_1","toolName":"get_quotes","input":{"symbols":["VOO"]}}` |
| `tool-output-available` | `toolCallId`, `output`, `isError`, `sources` | 도구가 실행된 뒤. `sources`는 `{id, title, url, publisher, tier}` 목록(`url`은 `null`일 수 있음). 시간 예산 때문에 취소된 도구는 `isError: true`이고 출처가 없습니다 | `data: {"type":"tool-output-available","toolCallId":"call_1","output":"[E1] Latest quotes ...","isError":false,"sources":[{"id":"E1","title":"Alpaca IEX quotes: VOO","url":null,"publisher":"Alpaca Markets (IEX feed)","tier":1}]}` |
| `source-url` | `sourceId`, `url`, `title` | URL이 있는 출처가 한 턴에서 처음 나올 때 한 번 | `data: {"type":"source-url","sourceId":"E2","url":"https://www.sec.gov/ixviewer/...","title":"VOO annual report"}` |
| `source-grounding` | `search_suggestions_html` | `grounded_web_answer`가 Google "검색 제안" HTML을 돌려준 뒤. 수정하지 않고 그대로 표시해야 합니다 | `data: {"type":"source-grounding","search_suggestions_html":"<div class=\"container\">...</div>"}` |
| `data-mandate` | `mandate`, `draft` | `draft_mandate` 뒤. `draft`는 항상 `true`입니다. 사용자가 `PUT /api/mandate/{session_id}`로 저장하기 전에는 효력이 없습니다 | `data: {"type":"data-mandate","mandate":{"budget_usd":"2000","max_position_pct":"20", ...},"draft":true}` |
| `data-trade-proposal` | `ticket` | `propose_trades`가 티켓을 만든 뒤. 객체는 [티켓](#티켓-객체)입니다 | `data: {"type":"data-trade-proposal","ticket":{"id":"T4HQ7ZB2NAK","status":"proposed", ...}}` |
| `data-grounding-flags` | `numbers` | 최종 텍스트가 완성된 뒤, 그 안의 일부 숫자가 이 턴의 도구 출력으로 뒷받침되지 않을 때. 참고용입니다 | `data: {"type":"data-grounding-flags","numbers":["$1,250","3.4%"]}` |
| `text-delta` | `delta` | 최종 답변을 최대 200자 안팎의 조각으로. 전체 텍스트를 먼저 만든 뒤 나눠 보내므로 토큰 단위 스트리밍이 아니라 조각 전달입니다 | `data: {"type":"text-delta","delta":"VOO tracks the S&P 500 [E1]. ..."}` |
| `error` | `message` | 턴이 거절되었거나 실패했을 때. 메시지는 일반적인 문구이며, 사용자 메시지의 언어로 나오고, 제공자의 문구는 담지 않습니다 | `data: {"type":"error","message":"Sorry, something went wrong while writing this reply. Please try again in a moment."}` |
| `finish` | `usage`, `elapsed_ms` | 항상 마지막 이벤트이며 오류 뒤에도 옵니다. `usage`는 제공자가 보고한 토큰 수를 턴 전체로 합산한 값입니다(예: `input`, `output`) | `data: {"type":"finish","usage":{"input":4210,"output":380},"elapsed_ms":18342}` |

한 턴 안의 순서: `start`. 이어서 라운드마다 선택적 `reasoning-delta`와, 도구 호출을 순서대로 하나씩 `tool-input-available`, `tool-output-available`, 있다면 `source-url`, 그리고 도구가 만든 이벤트(`data-mandate`, `data-trade-proposal`, `source-grounding`). 그다음 선택적 `data-grounding-flags`, `text-delta` 조각들, `finish`. 거절이나 실패는 `error` 뒤에 `finish`가 옵니다. 주문 상태 변화는 스트림으로 푸시하지 않으므로 `GET /api/tickets/{ticket_id}`를 폴링하세요.

## 포트폴리오

**`GET /api/portfolio`**는 브로커에서 모의 계좌를 실시간으로 읽습니다. 결과는 세 가지입니다.

| 경우 | 상태 | 본문 |
|---|---|---|
| 저장된 Alpaca 키 없음 | 200 | `{"connected": false}` |
| 연결됨 | 200 | `{"connected": true, "paper": true, "account": {...}, "positions": [...], "open_orders": [...]}` |
| 브로커에 닿지 않음 | 502 | `{"connected": true, "error": "broker_unavailable", "message": "..."}` |

```json
{
  "connected": true,
  "paper": true,
  "account": {"status": "ACTIVE", "equity": "100000", "cash": "100000", "buying_power": "100000", "non_marginable_buying_power": "100000", "last_equity": "100000", "trading_blocked": false, "account_blocked": false, "trade_suspended_by_user": false, "shorting_enabled": false},
  "positions": [{"symbol": "VOO", "qty": "4", "qty_available": "4", "avg_entry_price": "500.00", "current_price": "501.20", "market_value": "2004.80", "unrealized_pl": "4.80", "unrealized_plpc": "0.0024", "side": "long"}],
  "open_orders": []
}
```

`open_orders`에는 [주문 객체](#주문-객체) 형태의 미체결 주문이 최대 100개 들어갑니다. 502의 오류 코드는 `broker_unavailable`입니다.

## 투자 원칙(Mandate)

투자 원칙은 모든 제안과 모든 확인 단계에서 대조하는 한도 모음입니다.

**`GET /api/mandate/{session_id}`**는 `{"mandate": {...}, "saved": true 또는 false}`를 돌려줍니다. 기본값을 돌려줄 때 `saved`는 `false`입니다. 상태 코드: 200, 404 `session_not_found`.

**`PUT /api/mandate/{session_id}`**는 투자 원칙을 저장합니다. 본문은 투자 원칙 필드로 이루어진 JSON 객체입니다(최대 32,768바이트, 초과 시 413 `too_large`). `version`은 서버가 정하며(이전 버전 + 1, 첫 저장은 1) 본문의 `version`은 무시합니다. 저장할 때마다 `mandate_saved` 원장 항목도 추가됩니다. `{"mandate": {...}, "saved": true}`를 돌려줍니다. 상태 코드: 200, 404 `session_not_found`, 413 `too_large`, 422 `invalid_request`(`fields` 포함).

| 필드 | 타입 | 기본값 | 규칙 |
|---|---|---|---|
| `direction_text` | 문자열 | `""` | 사용자의 투자 방향을 풀어 쓴 문장 |
| `budget_usd` | 십진수 문자열 또는 `null` | `null`(한도 없음) | 0보다 커야 함 |
| `universe` | 문자열 | `us_equity` | `us_equity`만 가능 |
| `allow_symbols` | 티커 배열 | `[]` | 대문자로 바꾸고 중복을 제거합니다. 비어 있으면 모든 종목 허용 |
| `deny_symbols` | 티커 배열 | `[]` | 대문자로 바꾸고 중복을 제거합니다 |
| `max_position_pct` | 십진수 문자열 | `20` | 0 초과 100 이하 |
| `max_positions` | 정수 | `8` | 1 이상 |
| `cash_buffer_pct` | 십진수 문자열 | `5` | 0 이상 100 이하 |
| `horizon` | 문자열 | `weeks to months` | 자유 텍스트 |
| `order_types` | 배열 | `["limit","market"]` | 비어 있으면 안 됨. 각 항목은 `limit` 또는 `market` |
| `max_orders_per_day` | 정수 | `10` | 1 이상 |
| `max_order_usd` | 십진수 문자열 또는 `null` | `null`(한도 없음) | 0보다 커야 함 |
| `price_band_pct` | 십진수 문자열 | `3` | 0 초과 100 이하 |
| `allow_fractional` | 불리언 | `true` | |
| `version` | 정수 | `1` | 읽기 전용: 서버가 정합니다 |

```json
{"mandate": {"direction_text": "", "budget_usd": "20000", "universe": "us_equity", "allow_symbols": ["VOO", "SCHD"], "deny_symbols": [], "max_position_pct": "10", "max_positions": 8, "cash_buffer_pct": "5", "horizon": "weeks to months", "order_types": ["limit", "market"], "max_orders_per_day": 10, "max_order_usd": null, "price_band_pct": "3", "allow_fractional": true, "version": 1}, "saved": true}
```

## 티켓

티켓은 제안에 대한 서버의 기록입니다. 에이전트의 `propose_trades` 도구만 만들 수 있으며, 아래 엔드포인트로 조회, 확인, 거절, 갱신합니다. 모든 티켓 동작은 먼저 만료된 티켓을 정리합니다.

### 티켓 객체

| 필드 | 의미 |
|---|---|
| `id` | `T` 뒤에 10자 |
| `session_id` | 티켓을 소유한 세션. 다른 세션은 `not_found`를 받습니다 |
| `status` | `proposed`, `submitting`, `submitted`, `rejected_by_user`, `expired`, `superseded`, `failed` 중 하나 |
| `legs` | 실행 가능한 레그만: `{symbol, side, qty, notional, type, time_in_force, limit_price}`. `side`는 `buy` 또는 `sell`, `type`은 `market` 또는 `limit`, `time_in_force`는 `day` 또는 `gtc`이며, `qty`, `notional`, `limit_price`는 문자열 또는 `null`입니다 |
| `decisions` | 제안된 레그마다 리스크 결정 하나(거부된 레그 포함, 표시용): `{index, verdict, leg, original, reasons, est_price, est_cost_usd}`. `verdict`는 `accept`, `clamp`, `reject`이고, `original`은 제안된 원래 레그이며, 각 사유는 `{code, message, severity}`이고 `severity`는 `warn` 또는 `block`입니다 |
| `thesis`, `risks`, `invalidation` | 모델이 쓴 텍스트 |
| `evidence_ids` | 인용한 근거 ID |
| `mandate_version` | 제안 시점에 적용 중이던 투자 원칙의 버전 |
| `quote_as_of` | 추정에 쓰인 가장 오래된 호가의 시각 |
| `created_at`, `expires_at` | 생성 시각, 그리고 생성 시각 + 300초 |
| `hash` | 16진수 64자. `confirm`을 호출할 때 그대로 보냅니다 |
| `client_order_ids` | 실행 가능한 레그마다 하나, `fa-<티켓 ID>-<n>` |
| `broker_order_ids` | 티켓이 제출되기 전에는 비어 있고, 이후에는 레그마다 한 항목(주문을 모르면 `null`) |

```json
{
  "id": "T4HQ7ZB2NAK",
  "session_id": "S3F7QKX2M4ABC",
  "status": "proposed",
  "legs": [{"symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"}],
  "decisions": [{
    "index": 0,
    "verdict": "clamp",
    "leg": {"symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"},
    "original": {"symbol": "VOO", "side": "buy", "qty": "10", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00"},
    "reasons": [{"code": "max_position_pct", "message": "Clamped qty 10 → 4: VOO position $0.00 + estimated cost $5,000.00 exceeds 10% of budget $20,000.00 ($2,000.00)", "severity": "warn"}],
    "est_price": "500.00",
    "est_cost_usd": "2000.00"
  }],
  "thesis": "Core US equity exposure [E1][E2].",
  "risks": "Equity drawdowns.",
  "invalidation": "The thesis fails if ...",
  "evidence_ids": ["E1", "E2"],
  "mandate_version": 1,
  "quote_as_of": "2026-10-02T14:31:05Z",
  "created_at": "2026-10-02T14:31:20Z",
  "expires_at": "2026-10-02T14:36:20Z",
  "hash": "eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c",
  "client_order_ids": ["fa-T4HQ7ZB2NAK-0"],
  "broker_order_ids": []
}
```

### 주문 객체

주문은 Alpaca 주문 스냅샷입니다: `{id, client_order_id, symbol, side, qty, notional, type, time_in_force, limit_price, status, filled_qty, filled_avg_price, submitted_at, updated_at}`. 숫자는 문자열 또는 `null`입니다. 주문 상태가 `filled`, `canceled`, `expired`, `rejected`, `replaced`, `done_for_day` 중 하나이면 최종 상태입니다.

### 티켓 조회

**`GET /api/tickets/{ticket_id}?session_id=...`**는 티켓과 그 주문을 돌려줍니다. confirm 뒤에 폴링할 호출이 이것입니다. 브로커가 있으면 최종 상태가 아닌 주문을 모두 Alpaca에서 다시 읽고(읽기만 하며 제출은 하지 않음) 로컬 사본을 갱신합니다. 브로커가 없으면 저장된 행을 그대로 돌려줍니다.

```json
{"ticket": {"id": "T4HQ7ZB2NAK", "status": "submitted", "...": "..."}, "orders": [{"client_order_id": "fa-T4HQ7ZB2NAK-0", "broker_order_id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "order": {"id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "client_order_id": "fa-T4HQ7ZB2NAK-0", "symbol": "VOO", "side": "buy", "qty": "4", "notional": null, "type": "limit", "time_in_force": "day", "limit_price": "500.00", "status": "accepted", "filled_qty": "0", "filled_avg_price": null, "submitted_at": "2026-10-02T14:32:01Z", "updated_at": "2026-10-02T14:32:01Z"}}]}
```

불확실한 레그(브로커 ID를 알 수 없는 경우)의 `order`는 `null`입니다. 사용자에게 미체결 주문을 직접 확인하라고 안내하세요. 상태 코드: 200, 404 `ticket_not_found`(없는 ID이거나 다른 세션의 티켓), 422(`session_id` 누락).

### 확인·거절·갱신

| 호출 | 본문 | 동작 |
|---|---|---|
| `POST /api/tickets/{ticket_id}/confirm` | `{"session_id": "...", "hash": "<ticket.hash>"}` | 티켓을 다시 검사하고, 의미 있는 변화가 없으면 주문을 제출합니다. **주문을 낼 수 있는 유일한 호출입니다.** 브로커와 `trading_enabled`가 필요합니다 |
| `POST /api/tickets/{ticket_id}/reject` | `{"session_id": "..."}` | `proposed` 티켓을 `rejected_by_user`로 바꿉니다. 브로커 없이도 동작합니다. 이미 거절한 티켓을 다시 거절하면 200으로 답합니다 |
| `POST /api/tickets/{ticket_id}/refresh` | `{"session_id": "..."}` | `proposed` 티켓을 현재 가격으로 다시 호가해 새 티켓으로 바꿉니다. 제출하지 않습니다. 브로커가 필요합니다 |

세 호출 모두 같은 결과 본문으로 답합니다.

| 필드 | 의미 |
|---|---|
| `status` | `submitted`, `requote`, `rejected`, `disabled`, `expired`, `not_found`, `error` 중 하나 |
| `message` | 비밀값 가림 처리를 거친 짧은 영어 요약. `submitted`에서는 레그별 결과를 나열합니다 |
| `ticket` | 현재 시점의 티켓, 없으면 `null` |
| `orders` | 접수된 주문(주문 객체), 없으면 `[]` |
| `new_ticket` | `requote`일 때 대체 티켓, 그 밖에는 `null` |

```json
{"status": "submitted", "message": "1 of 1 orders placed. Leg 1 BUY 4 VOO @ limit $500.00: placed (status accepted).", "ticket": {"id": "T4HQ7ZB2NAK", "status": "submitted", "...": "..."}, "orders": [{"id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "status": "accepted", "...": "..."}], "new_ticket": null}
```

실행기가 동작하기 전에 다음 응답이 나올 수 있습니다.

| 상태 | 코드 | 발생 조건 |
|---|---|---|
| 503 | `broker_not_connected` | 저장된 Alpaca 키 없이 confirm 또는 refresh. reject는 그대로 동작합니다 |
| 422 | `invalid_request` | 본문에 필드(예: `hash`)가 빠짐 |
| 400 | `invalid_request` | 다시 호가할 수 없는 티켓에 대한 refresh |
| 502 | `refresh_failed` | refresh가 예기치 않게 실패 |
| 500 | `confirm_failed` | confirm이 예기치 않게 실패. 메시지는 다시 시도하기 전에 미체결 주문을 확인하라고 안내합니다 |
| 500 | `reject_failed` | reject가 예기치 않게 실패 |

### 상태 코드 매핑

티켓 동작의 HTTP 상태는 결과의 `status`를 따릅니다. "해당 없음"은 그 동작이 그 결과를 내지 않는다는 뜻입니다.

| 결과 `status` | 의미 | confirm | reject | refresh |
|---|---|---|---|---|
| `submitted` | 주문을 보냈고 결과도 저장했습니다 | 200 | 해당 없음 | 해당 없음 |
| `requote` | 아무것도 제출하지 않았습니다. 가격, 판정 또는 레그가 바뀌어 새 티켓(`new_ticket`)이 이 티켓을 대체했습니다 | 409 | 해당 없음 | 200 |
| `disabled` | 킬 스위치가 켜져 있습니다. 아무것도 바뀌지 않았습니다 | 423 | 해당 없음 | 해당 없음 |
| `expired` | confirm이 실행될 때 티켓이 이미 만료돼 있었습니다 | 410 | 해당 없음 | 해당 없음 |
| `not_found` | 없는 ID이거나 다른 세션의 티켓입니다 | 404 | 404 | 404 |
| `rejected`, 티켓이 `rejected_by_user` | 사용자가 이미 거절했습니다 | 409 | 200 | 409 |
| `rejected`, 티켓이 `expired` | 동작 전에 이미 만료됐습니다 | 410 | 410 | 410 |
| `rejected`, 그 밖의 경우 | 해시 불일치, `proposed`가 아닌 티켓, 다른 confirm이 이미 선점함, 현재 가격에서 실행 가능한 레그가 없음 | 409 | 409 | 409 |
| `error` | 재검사 실패, 모든 레그 실패, 결과를 저장하지 못함, 또는 티켓을 읽을 수 없음. 결과를 저장하지 못한 경우 메시지가 주문이 들어갔을 수 있다고 알리므로 다시 시도하기 전에 미체결 주문을 확인하세요 | 502 | 502 | 502 |

## 설정

**`GET /api/settings`**는 연결 상태를 돌려줍니다. 비밀값은 절대 반환하지 않습니다.

| 필드 | 값 |
|---|---|
| `alpaca` | `connected`, `missing`(키 없음), `invalid`(Alpaca가 키를 거부함), `unreachable` |
| `tavily`, `sec_user_agent`, `gemini` | `set` 또는 `missing` |
| `trading_enabled` | 불리언. 킬 스위치 이후에는 `false` |
| `model`, `effort` | LLM 모델 이름과 추론 강도 |
| `api_allow_remote` | 불리언 |
| `env_fallbacks` | 프로세스 환경 변수만 제공하는, 쓰기 가능한 설정의 이름(예: `TAVILY_API_KEY`). `.env`가 환경 변수보다 우선하므로 앱에서 이 값을 지워도 재시작하면 되살아납니다 |

```json
{"alpaca": "connected", "tavily": "missing", "sec_user_agent": "set", "gemini": "set", "trading_enabled": true, "model": "gemini-3.8-flash", "effort": "medium", "api_allow_remote": false, "env_fallbacks": []}
```

**`PUT /api/settings`**는 키를 로컬 `.env`에, 거래 플래그와 함께 저장합니다. 모든 필드는 선택입니다.

| 필드 | 타입 | 비고 |
|---|---|---|
| `alpaca_key_id`, `alpaca_secret` | 문자열, 최대 256자 | 반드시 함께 보내야 합니다. 저장하기 전에 모의투자 호스트로 실제 확인합니다 |
| `tavily_key` | 문자열, 최대 256자 | |
| `sec_user_agent` | 문자열, 최대 256자 | SEC가 요구하는 이름과 연락처 |
| `trading_enabled` | 불리언 | 킬 스위치 이후 거래를 다시 켜는 유일한 방법 |

문자열 필드에서 `""`는 키를 삭제하고, `null`(또는 필드 생략)은 그대로 둡니다. 값은 ASCII 공백을 다듬은 뒤 `"`, `\`, `${`가 없는 출력 가능한 ASCII여야 합니다. 저장하면 커넥터를 그 자리에서 다시 만들므로 에이전트의 도구 구성은 다음 메시지부터 바뀝니다. 응답은 `GET /api/settings`와 같은 본문입니다.

| 상태 | 코드 | 발생 조건 |
|---|---|---|
| 200 | | 저장됨 |
| 400 | `invalid_value` | 저장할 수 없는 문자가 값에 있음. `field`가 해당 필드를 알려 줍니다 |
| 400 | `alpaca_pair_required` | Alpaca 키 ID와 시크릿 중 하나만 보냄(또는 하나만 삭제함) |
| 400 | `alpaca_invalid` | Alpaca가 키를 거부함. 아무것도 저장하지 않았습니다 |
| 502 | `alpaca_unreachable` | 키를 확인하려 했지만 Alpaca에 닿지 못함. 아무것도 저장하지 않았습니다 |
| 500 | `settings_not_saved` | 설정을 기록하지 못함 |
| 422 | `invalid_request` | 알 수 없는 필드이거나 타입이 틀림 |

## 킬 스위치

**`POST /api/kill-switch`**는 거래를 끕니다. 본문(선택): `{"suspend_broker": false}`.

`trading_enabled`를 즉시 `false`로 바꾸므로 진행 중인 confirm은 더 이상 레그를 제출하지 않으며, `.env`에 `TRADING_ENABLED=false`를 기록합니다. `suspend_broker: true`이고 브로커가 연결돼 있으면 Alpaca에 모의 계좌의 거래 정지를 요청하고 미체결 주문도 취소합니다. 거래가 꺼져 있는 동안 confirm은 423 `disabled`로 답하며 리서치와 제안은 계속 동작합니다. 다시 켜려면 `PUT /api/settings {"trading_enabled": true}`를 사용하세요.

```json
{"trading_enabled": false, "suspend_broker": false, "broker_connected": true, "persisted": true}
```

`.env`를 기록하지 못하면 `persisted`가 `false`입니다(스위치는 메모리에서 이미 꺼져 있음). 상태 코드: 200, 422. 킬 스위치는 세션 ID `system` 아래 원장에 기록됩니다.

## 원장 내보내기

**`GET /api/ledger/{session_id}.jsonl`**은 세션의 감사 원장을 `application/x-ndjson`으로, `Content-Disposition: attachment; filename="ledger-<session_id>.jsonl"` 헤더와 함께 내려받게 합니다. 각 줄은 JSON 객체입니다.

```json
{"id": 12, "session_id": "S3F7QKX2M4ABC", "kind": "claim", "created_at": "2026-10-02T14:32:00.512345+00:00", "payload": {"ticket_id": "T4HQ7ZB2NAK", "legs": 1}}
```

`kind`는 `mandate_saved`, `confirm_request`, `claim`, `requote`, `submit`, `broker_response`, `reconcile`, `order_status`, `result`, `reject`, `refresh`, `recover` 중 하나입니다. 페이로드는 저장되기 전에 가려지므로 비밀값이 나타나지 않습니다. 상태 코드: 200, 404 `session_not_found`. 킬 스위치 항목은 세션이 아닌 세션 ID `system`에 속하므로 여기서는 내보낼 수 없습니다.

## 예시: 메시지에서 티켓 확인까지

이 예시는 POSIX 셸(Windows에서는 Git Bash)을 가정합니다. ID, 해시, 가격은 자리표시자이므로 직접 받은 응답의 값으로 바꾸세요. `curl -w`는 마지막에 HTTP 상태를 출력합니다. 실제 키를 예시나 공유하는 스크립트에 붙여 넣지 마세요.

**1. 세션 만들기.**

```bash
curl -s -w '\nHTTP %{http_code}\n' -X POST http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"title": "Core ETF sleeve", "lang": "en"}'
```

```text
{"id": "S3F7QKX2M4ABC", "title": "Core ETF sleeve", "created_at": "2026-10-02T14:30:02.114233+00:00", "lang": "en"}
HTTP 201
```

**2. 메시지를 보내고 스트림 읽기**(`-N`은 curl의 버퍼링을 꺼서 이벤트가 도착하는 대로 보이게 합니다).

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"session_id": "S3F7QKX2M4ABC", "message": "Build a $2,000 core position in a broad US equity ETF, moderate risk."}'
```

```text
data: {"type":"start","messageId":"6f1c0e9d2b7a4c58a3e1d94b7c2f8a10"}

data: {"type":"tool-input-available","toolCallId":"call_1","toolName":"get_quotes","input":{"symbols":["VOO"]}}

data: {"type":"tool-output-available","toolCallId":"call_1","output":"[E1] Latest quotes ...","isError":false,"sources":[{"id":"E1","title":"Alpaca IEX quotes: VOO","url":null,"publisher":"Alpaca Markets (IEX feed)","tier":1}]}

data: {"type":"data-trade-proposal","ticket":{"id":"T4HQ7ZB2NAK","status":"proposed","hash":"eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c", ...}}

data: {"type":"text-delta","delta":"I propose buying 4 shares of VOO with a limit order ... [E1]. Review the ticket and confirm or reject it."}

data: {"type":"finish","usage":{"input":4210,"output":380},"elapsed_ms":18342}
```

티켓은 `data-trade-proposal` 이벤트에 들어 있습니다. 그 `id`와 `hash`를 보관하세요. (제안을 받으려면 저장된 Alpaca 모의투자 키가 필요합니다. 키가 없으면 에이전트는 리서치만 할 수 있고 `propose_trades` 도구가 없습니다.)

**3. 티켓 조회.** `GET`에는 `X-FinAgent-Client` 헤더가 필요하지 않습니다.

```bash
curl -s "http://127.0.0.1:8000/api/tickets/T4HQ7ZB2NAK?session_id=S3F7QKX2M4ABC"
```

**4. 확인.** `hash`는 검토한 티켓의 것이어야 합니다.

```bash
curl -s -w '\nHTTP %{http_code}\n' -X POST http://127.0.0.1:8000/api/tickets/T4HQ7ZB2NAK/confirm \
  -H 'Content-Type: application/json' -H 'X-FinAgent-Client: 1' \
  -d '{"session_id": "S3F7QKX2M4ABC", "hash": "eb310ac7504bbfd593b95c35a3030ecb663dde7ab6ed04b1ecfeb61608ea078c"}'
```

```text
{"status": "submitted", "message": "1 of 1 orders placed. Leg 1 BUY 4 VOO @ limit $500.00: placed (status accepted).", "ticket": {...}, "orders": [{...}], "new_ticket": null}
HTTP 200
```

가격이 움직였다면 `HTTP 409`와 함께 `"status": "requote"`가 오고, 대체 티켓은 `new_ticket`에 들어 있습니다. 그 티켓을 검토한 뒤 새 `id`와 `hash`로 다시 확인하세요. `submitted` 결과를 받은 뒤에는 최종 상태가 아닌 주문이 없어질 때까지 몇 초 간격으로 3단계를 반복하세요.
