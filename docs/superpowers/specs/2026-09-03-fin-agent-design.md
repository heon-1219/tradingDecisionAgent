# 금융상품 Agent (Financial Product Analyst) — Design Spec

Date: 2026-09-03
Target: 제10회 2026 미래에셋증권 AI Festival, 예선 제출 (마감 2026-09-06 23:59), 평가 API 운영 2026-09-07 ~ 09-20.

## 1. Goal

A single HTTP service exposing `GET /answer` that answers natural-language questions about four
provided product master tables (국내채권 / 국내ETF / 해외ETF / 공모펀드) using evidence from the data,
and explicitly declines (or asks a clarifying question) when the data cannot confirm an answer.

Scoring drivers this design targets:

| Weight | Item | What the design does about it |
|---|---|---|
| 40% | API answer accuracy, evidence, latency | Federated retrieval (SQL + graph + lexical/vector), ontology-validated filters, explicit evidence block, 50s time budget |
| 20% | Source code + `ontology/*.ttl` | Clean `src/` package, reproducible ETL, five Turtle files that the runtime actually loads |
| 40% | Technical proposal | Architecture is legible: 적재 → 온톨로지 → 그래프/색인 → 검색 → 생성. `docs/` carries diagrams and reproduction steps |

## 2. Assumptions (decisions made without user input)

1. **LLM provider**: Anthropic Claude by default (`LLM_PROVIDER=anthropic`), switchable to any
   OpenAI-compatible endpoint (`LLM_PROVIDER=openai`, covers OpenAI, Gemini's OpenAI-compatible API,
   NCP CLOVA Studio, vLLM). Keys come from `.env`. The organizers said LLM usage rules are "추후 공지";
   the abstraction lets the team switch in minutes.
2. **Runtime**: Python 3.11+ (developed on 3.14), FastAPI + uvicorn, deployed as one Docker
   container on the team's NCP VM. Optional Caddy sidecar for HTTPS.
3. **Repository root** is `fin-agent/` (this folder), matching the mandated layout
   (`src/`, `ontology/`, `requirements.txt`, `README.md`). The competition PDFs are NOT committed.
4. **Raw data** (`data/raw/*.xlsx`) is committed for reproducibility (18 MB, private repo).
5. **External data** is allowed by the rules and needed for the hard questions (ETF constituents,
   corporate relations). This design ships the ingestion path and a best-effort fetcher; the
   curated lexicons are hand-maintained YAML.

## 3. Data facts that drive the design

| Table | Rows | Key | Base date | Notes |
|---|---|---|---|---|
| PRBD01N001 국내채권 | 21,882 | (pd_no, pd_exg_mkt, info_seq) | 2026-08-21 | 634 rows are actually sellable (장외 · 온오프 겸용 · buy_yield present); 국공채 have no credit grade |
| PREF01N001 국내ETF | 1,780 | pd_itm_no | 2026-08-21 | 1,235 ETF + 545 ETN; 총보수(cu_charge_rt) only 12% populated; Korean labels |
| PREF02N001 해외ETF | 6,037 | pd_itm_no (RIC) | 2026-08-22 | All US-listed; English names + ~300-char strategy text; 총보수·AUM fully populated |
| PRFD01N001 공모펀드 | 23,676 | itm_no | 2026-08-21 | 공모 14,716 / 사모 8,960; 판매중 10,962; many share classes of one fund; fee columns have undocumented units |

Sample-question coverage:

- "판매가능 원화채권 AA- 이상" → SQL only (78 rows). Easy.
- "국민성장펀드 구조·투자전략 동향" → 4 share classes exist; 동향 needs external documents. Medium.
- "캠브리콘 편입 중국반도체 ETF" → needs constituents (not in data). External fetch + graph.
- "에코프로 자회사 편입 ETF 중 순자산 큰 상품의 위험요인" → constituents + corporate graph + SQL.
- "KODEX AI로봇 ETF" → does not exist (KODEX 로봇액티브, RISE AI&로봇 do) → clarify/decline.
- "신용등급 AAAA" → invalid per ontology enumeration → decline, state valid grades.
- "Kimi 관련 투자상품" → no entity match → decline.

## 4. Approaches considered

- **A. NL2SQL only**: one LLM call writes SQL, one composes the answer. Fast and simple, but blind to
  themes, relations, constituents, and prone to answering unanswerable questions.
- **B. Tool-using agent over federated stores (chosen)**: an LLM plans and calls typed tools
  (`sql_query`, `search_products`, `graph_query`, `resolve_entity`, `search_documents`) with a time
  budget; deterministic pre-checks (entity resolution, ontology validation) run before the first LLM
  call to cut round-trips and catch unanswerable cases early.
- **C. Fixed pipeline (classifier → templates)**: most predictable latency, but brittle on 상 difficulty
  questions and hard to extend in four days.

B is chosen; C's guardrails (time budget, validators, forced final answer) are folded in.

## 5. Architecture

```
GET /answer?question_id&question
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Orchestrator (async, 50s budget, trace collector)                    │
│                                                                      │
│  0. Pre-analysis (no LLM): normalize text, lexicon match             │
│     (themes / companies / managers / product names), ontology        │
│     validators (credit grade, risk grade, region, asset class)       │
│  1. LLM turn 1: intent + plan + first tool calls                     │
│  2..n. Tool rounds (parallel tool execution)                         │
│  n+1. Final answer JSON {status, answer, evidence_refs}              │
└───────┬───────────────┬──────────────────┬───────────────────────────┘
        │               │                  │
   RDB (SQLite)    Graph (rdflib)     Index (BM25 + char n-gram,
   4 tables +      ABox built from    optional dense embeddings)
   4 views         SQLite + lexicons  products + external docs
        ▲               ▲                  ▲
        └──── ETL ──────┴──── build ───────┘   ontology/*.ttl → schema cards + validators
```

### 5.1 Packages (`src/finagent/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | pydantic-settings: provider, model, keys, paths, budget | — |
| `models.py` | Pydantic response schema (`AnswerResponse`) | — |
| `etl/` | xlsx → SQLite (`data/store/products.db`): typed columns, yyyymmdd → ISO dates, derived views | pandas, sqlite3 |
| `ontology/` | Load `ontology/*.ttl`; derive schema cards (view column ↔ property label/comment/enum) and validators | rdflib |
| `kb/rdb.py` | Read-only SQL execution: SELECT-only, single statement, auto LIMIT, timeout, row → dict | sqlite3 |
| `kb/lexicon.py` | YAML lexicons: themes (ko/en synonyms), companies (aliases, subsidiaries, tickers), managers (aliases, groups) | pyyaml |
| `kb/graph.py` | Build in-memory KG (ABox) from SQLite + lexicons + constituents; SPARQL; export `data/store/kg.ttl` | rdflib, lexicon |
| `kb/search.py` | Lexical index (BM25 over tokens + char 2/3-gram) over product names/strategy/docs; optional dense vectors | numpy |
| `kb/entity.py` | Mention → candidates (product / company / manager / theme / index) with scores | search, lexicon |
| `kb/docs.py` | External markdown documents (`data/external/docs/*.md`) chunking + indexing | search |
| `agent/llm.py` | Provider-agnostic async chat with tool calling (anthropic / openai) | anthropic, openai |
| `agent/tools.py` | Tool schemas + executors | kb.* |
| `agent/prompts.py` | System prompt, schema cards injection, answer policy | ontology |
| `agent/orchestrator.py` | Pre-analysis, agent loop, budget, trace, evidence assembly | llm, tools, entity |
| `agent/answer.py` | Final answer policy: answer / cannot_confirm / clarify; evidence + trace formatting | — |
| `api.py` | FastAPI app: `GET /answer`, `GET /health`; never returns 500 | orchestrator |

### 5.2 Data flow

1. `python -m finagent.etl.build_store` reads `data/raw/*_data.xlsx` → `data/store/products.db`
   with tables `bond_kr`, `etf_kr`, `etf_gl`, `fund_pub` (original column names kept) and views
   `v_bond_kr`, `v_etf_kr`, `v_etf_gl`, `v_fund_pub` (Korean-documented, derived columns:
   `is_sellable`, ISO dates, `fund_family`, `manager_name`, `product_type`).
2. On server start: open SQLite (read-only), load ontology → schema cards, build lexical index,
   build KG from SQLite + `data/enrichment/*.yaml` + `data/enrichment/constituents/*.csv`
   (cached as pickle in `data/store/`).
3. Per request: orchestrator → tools → final JSON.

### 5.3 Tools exposed to the LLM

| Tool | Input | Output |
|---|---|---|
| `table_info(table)` | view name | schema card: columns, Korean meaning, enum values, base date, sellable semantics |
| `sql_query(sql, purpose)` | SELECT over views | rows (≤200), row count, executed SQL, base date |
| `search_products(query, table?, k)` | free text (ko/en) | top-k products with ids, names, key fields, score |
| `resolve_entity(mention)` | text | typed candidates: product / company / manager / theme / index; empty ⇒ unknown |
| `graph_query(sparql)` | SPARQL SELECT over the KG | bindings |
| `graph_neighbors(entity, relation?)` | entity label or IRI | related entities grouped by relation |
| `search_documents(query, k)` | text | chunks from external markdown docs with source + date |
| `final_answer(status, answer, evidence_refs)` | structured | ends the loop |

### 5.4 Ontology (`ontology/`)

- `common.ttl`: `fp:Product` root, `fp:ManagementCompany`, `fp:Issuer`, `fp:Company`, `fp:Theme`,
  `fp:Index`, `fp:RiskGrade` (1–6 individuals), `fp:AssetClass`, `fp:Region`, `fp:DistributionCycle`;
  properties `fp:managedBy`, `fp:issuedBy`, `fp:hasRiskGrade`, `fp:hasAssetClass`,
  `fp:investsInRegion`, `fp:hasTheme`, `fp:tracksIndex`, `fp:hasConstituent`, `fp:subsidiaryOf`,
  `fp:baseDate`, `fp:isOnSale`, `fp:productCode`, `fp:name`; annotation `fp:sourceColumn`,
  `fp:sourceTable`; SHACL shapes for grade ranges.
- `bond_kr.ttl` (`fp:DomesticBond`): `fp:CreditGrade` enumeration (AAA … D, with ordering
  `fp:gradeRank`), bond type/interest type/offering type individuals, coupon, maturity, duration,
  yields, sellable channel.
- `etf_kr.ttl` (`fp:DomesticETF`, `fp:DomesticETN`): strategy (실물복제/합성복제/액티브),
  leverage factor, returns by period, volatility, dividend fields, pension eligibility.
- `etf_gl.ttl` (`fp:ForeignETF`): expense ratio, AUM, exchange, replication method,
  inverse/leverage flags, strategy text.
- `fund_pub.ttl` (`fp:PublicFund`, `fp:ShareClass`, `fp:FundFamily`): 운용속성, 제로인 유형,
  benchmark, hedge flag, class fee type/channel, fee components (units flagged as unverified).

Every datatype property carries `fp:sourceColumn` + `fp:sourceTable` so `ontology/loader.py`
can generate the schema cards and validators from the TTL itself (ontology as runtime layer).

### 5.5 Knowledge graph (ABox)

Built at startup, not committed. Entities: products (all four tables), managers (normalized via
`managers.yaml`), issuers, indices/benchmarks, themes (matched from names/strategy via
`themes.yaml`), companies (from `companies.yaml` + constituents), fund families (share classes
grouped by name normalization). ~250k triples; pickled cache for fast restart.

### 5.6 Answer policy

Final answer status is one of:

- `answered`: evidence rows exist; answer cites product codes/names, numbers with units, base date.
- `cannot_confirm`: named entity unresolved, ontology-invalid constraint, or zero rows after
  reasonable retries. Answer states "제공된 데이터로는 확인할 수 없습니다" and what was checked.
- `clarify`: ambiguous mention (several near matches) or missing condition. Answer asks one
  precise question and lists candidates.

Hard rules in the system prompt: no return forecasts, no definitive recommendations, only numbers
from tool results, always mention 기준일.

### 5.7 Response contract (from the PDF)

`200 OK`, `application/json; charset=utf-8`:

```json
{"question_id": "...", "question": "...", "retrieved_context": "...", "think_trace": "...", "answer": "..."}
```

- `retrieved_context`: numbered evidence blocks: table + Korean name + base date + condition +
  row count + executed SQL/SPARQL + referenced product codes/names.
- `think_trace`: numbered steps (pre-analysis, each tool call with purpose and result size, decision).
- Unknown query params ignored; missing params → 200 with explanatory answer; any exception →
  200 with a fallback body (logged). `GET /health` for monitoring.

### 5.8 Latency

Budget 50s (configurable). Startup pre-loads everything. Tool calls in one LLM turn execute
concurrently. When the budget is nearly exhausted the orchestrator forces `final_answer` with the
evidence gathered so far. Target: 하 ≤ 15s, 중 ≤ 30s, 상 ≤ 50s.

### 5.9 Error handling

- SQL guard rejects non-SELECT, multiple statements, and unknown tables; errors are returned to the
  LLM as tool results so it can retry (max 2 retries per tool).
- LLM provider errors: one retry with backoff, then fallback answer.
- Startup fails loudly if the store is missing (README documents the build step).

### 5.10 Testing

- Unit: ETL normalization (dates, flags, fund family), SQL guard, validators from TTL,
  lexicon matching, entity resolution, answer formatting.
- Contract: FastAPI test client with a stub LLM: schema, extra params, missing params, exception
  path all return 200 with the five fields.
- Evaluation script: `scripts/eval_run.py` runs `data/eval/questions.jsonl` (PDF samples + team
  questions incl. unanswerables) against a running server and writes latency + answers.

## 6. Deliverables in this repo

`src/`, `ontology/*.ttl` (5 files), `requirements.txt`, `README.md` (환경 구성 · 실행 명령 · API 명세),
`Dockerfile`, `docker-compose.yml`, `.env.example`, `docs/api_spec.md`, `docs/proposal_outline.md`
(skeleton for the 기술제안서 with architecture/flow diagrams), `scripts/` (store build, constituent
fetcher, doc parser, eval runner), `tests/`.

## 7. Out of scope

Web UI, authentication, multi-tenant deployment, live market data, automatic document crawling
beyond the best-effort constituent fetcher.
