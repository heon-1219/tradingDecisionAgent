English | [한국어](#한국어)

*Note: This draft describes the competition build, for which the competition rules mandated NCP HyperCLOVA X. This English edition runs on Google Gemini (default model `gemini-3.8-flash`) instead.*

# Technical Proposal Draft (Following the Recommended Outline) — Financial Product Agent

> Skeleton for writing the PDF submission. The diagrams for each section are drawn in mermaid; export them as images when converting to PDF.

## 01. Overall Architecture and Specialized Features

**One-line summary**: a Federated Query Agent that uses the ontology as an "execution layer" rather than as "documentation". Structured masters are routed to SQL, relations (managers, themes, constituents, subsidiaries) to the graph, names and descriptive text to the lexical index, and background knowledge to external documents; value domains are validated before answering to reduce hallucinations and wrong answers on unanswerable questions.

Differentiators
1. **Pre-analysis → Round 0 retrieval execution plan (deterministic pre-analysis)** — before calling the LLM, the system extracts product-name spans, resolves entities, matches the theme/company lexicons and validates values against the ontology; based on the result it picks *only what that question needs* from graph, vector and documents, runs those lookups in advance and **injects the result rows into the first prompt as facts** (across the 54 questions, graph fires 6 times, vector 7 and documents 0, and nothing fires for 41 questions) → fewer rounds (lower latency) and early detection of unanswerable questions (AAAA, grade 7, non-existent products). It also passes along resolved identifiers (e.g. Cambricon → the store's spelling `한무기`) and the list of IDs to query, so SQL does not spin its wheels on names.
2. **Ontology → auto-generated schema cards/validators** — the runtime reads the TTL's `fp:sourceColumn`, enumeration individuals and SHACL to build the prompt and the validators (the "runtime identification +20pp" perspective from the technical session).
3. **Measurement-driven hybrid search — including the size of its gain** — when BM25 and vectors (TF-IDF+LSA, 384 dimensions) were combined as equals with RRF, we observed irrelevant documents that both retrievers ranked mid-list pushing out the top lexical match (the #1 result for "배터리 양극재" (battery cathode materials) was polluted). The final design gives the head to lexical results and the bottom 30% of slots to documents found only by semantic search. Measured on 1,313 queries, this structure **guarantees that the hybrid's recall@1 and @5 are identical to lexical-only down to the decimal** (vectors never change the top ranks), and the gain is limited to **recall@10 +3.5%p** on theme queries (87.0% → 90.5%). We do not hide this number — TF-IDF+LSA is based on lexical co-occurrence, not semantic embeddings, so this is the ceiling of the approach (05-2 ⑥).
4. **Materialized ontology inference** — the TBox is merged into the runtime graph and superclass inheritance, inverse relations and domain/range are applied. Queries such as `?p a fp:Product` and `?am fp:manages ?p` work without separate loading (a working pipeline implementation of technical session 01-4's "put in 2, get 5"). The current graph has **1,082,874 triples** (965,823 asserted + 117,051 inferred) **· 73 predicates**.
5. **Constituents in both the RDB and the graph** — the top holdings of 1,162 domestic ETFs, 22,599 rows, are collected and **loaded into the SQLite table `etf_constituent`**, and the knowledge graph's `hasConstituent` is generated from that table. Because both use the same source, SQL results and graph results cannot diverge (cross-checked: Samsung Electronics 229/229, EcoPro BM 42/42). Companies are identified only by the **ticker code** the source provides; name similarity is not used.
   - However, `subsidiaryOf` (parent–subsidiary) has **only 4 edges**. The organizer's four masters have no ownership column at all, and we kept to the rule of computing from the provided data, so we did not extend it. We do not claim "multi-level subsidiary traversal" as a general feature; the range that actually works is the EcoPro group and SK Innovation, as explicitly listed in the lexicon.
6. **Answer-policy state machine** — the three states `answered / cannot_confirm / clarify` are enforced through the tool schema, and the server corrects violations.
7. **Never-fail contract** — always 200 + 5 fields; 120-second time budget, at most 8 rounds; when the budget is nearly used up, the loop is force-terminated and the answer is built from the evidence gathered.

```mermaid
flowchart TB
    subgraph Ingest[Ingestion]
      X[4 Excel masters] --> E[ETL: normalization · derived columns · 5 views]
      C[Naver Finance constituents<br/>22,599 rows] --> E
      E --> S[(SQLite<br/>5 tables · 5 views · 215 columns)]
      Y[Lexicon YAML<br/>77 managers · 48 themes · 97 companies] --> K
      T[ontology/*.ttl, 6 files] --> L[Loader: schema cards · validators]
    end
    subgraph Index[Indexing]
      S --> B[BM25 index]
      S --> VEC[Vector index TF-IDF+LSA]
      S --> K[(Knowledge Graph)]
      M[External docs MD] --> D[Document chunk index]
    end
    subgraph Serve[Retrieval · Generation]
      Q[Question] --> P[Pre-analysis] --> A[LLM tool loop] --> F[final_answer] --> J[JSON response]
      A --- S & K & B & VEC & D
      L -.-> P & A
    end
```

## 02. Proposed Method

### What specialized data we collected and how we cleaned it
- The organizer's four masters: code-like numeric columns pinned as strings (leading zeros preserved), yyyymmdd→ISO, derived `is_sellable` (bonds not yet redeemed, **21,866 rows**), `has_buy_yield` (634 rows with a purchase yield), `is_on_sale`, `fund_family` (23,676 fund class rows = **17,795 families**) and `manager_name`. Base dates are recorded in the `meta` table and shown automatically in the evidence. **All 215 provided columns** of the 5 views are exposed (no selection — see 05-2 for why).
- Constituents: Naver Finance API (top holdings and weights); 1,162 of 1,235 ETFs succeeded, 22,599 rows. **Loaded as a SQLite table**, and the graph is generated from that table. The `itemCode` the source provides is preserved, so 79% of rows have a ticker code. Caching and re-fetch scripts are provided.
- Lexicons: 77 managers (brand prefixes TIGER/KODEX/RISE…), 48 themes (Korean/English synonyms), 97 companies (aliases, tickers, parent companies, unlisted status, Chinese names such as 한무기 = Cambricon).
- Unstructured: National Growth Fund (국민성장펀드) press releases and similar material converted PDF→Markdown (`parse_docs.py`), with source and base date in the front matter.

### How we identified and filled missing values

Premise: per the organizer's notice, **zeros and missing values in the provided data are intentional**. We therefore do not fill missing values in the original columns; they are excluded from calculations, and answers state "not provided". What we fill are *the derived columns we created*, and the basis for filling must always lie **within the provided data**.

**How we identify them** — we checked the NULL ratio of every column in every view, and for derived columns we also recorded their derivation basis (which original column they come from). One derived column stood out in this check.

**Worked example: recovering the fund manager (`manager_name`)**

| Step | Details |
|---|---|
| Symptom | Of 23,676 public-fund rows, **25.4% (6,016 rows)** had no manager |
| Cause | The manager was being **guessed from the prefix of the fund name**. Funds whose names do not reveal a brand could not be filled |
| Better basis | The organizer provides `or_co_xtn_itt_cd` (manager institution code) **100% populated**. The manager can be derived from the code instead of guessed |
| Filling | Collect the manager spellings observed for each institution code, take the majority spelling as the representative name, and fill the rows the name prefix could not with the code |
| Result | 17,660 rows (74.6%) → **21,592 rows (91.2%)**. 2,084 NULL rows remain |
| Traceability | The basis used for filling is kept in a new column **`manager_source`** — `name` 17,660 / `code` 3,932 / NULL 2,084 |

**What we decided not to fill**

- **Manager rename mapping**: sometimes both an old brand and a new brand are observed under one institution code. But the provided data **has no fund inception date**, so there is no basis for deciding which spelling is the current name. Pulling in an external table would step outside the rule ("compute from the provided data"), so we use the majority spelling as-is — even if it is the old brand.
- **Domestic ETF total expense ratio (only 18% populated) · fund returns (only about 30% populated) · the unit of the fund fee columns**: these are values the source does not provide. We do not estimate them; we answer "no data". Fund fees have an undisclosed unit, so **only relative comparison between classes** is allowed.
- **`subsidiaryOf` (parent–subsidiary)**: the organizer's four masters have no ownership column. The **4 edges** written by hand in the lexicon are all there is, and we do not claim this as a general feature.

**One more identification method** — this one catches *silent errors* rather than missing values. One filling step in the pipeline was **reporting success without filling any rows**, because `pandas`'s `Series.map` turns a returned `None` into `NaN`, and `NaN` is truthy. There was no exception, no log, and the tests were green. So this project made it a procedure, for every filling step, to **recount the number of filled rows before and after and compare them**. We look at counts, not at success flags.

### Data model (5 views)

```mermaid
erDiagram
    v_etf_kr ||--o{ v_etf_constituent : "pd_itm_no = etf_itm_no"

    v_etf_kr {
        text pd_itm_no PK "item number (1,780 rows)"
        text pd_ticker "ticker"
        text pd_nm "product name"
        text product_type "domestic ETF 1,235 / domestic ETN 545"
        text manager_name "manager"
        real pd_net_tamt "total net assets"
        int risk_grade "risk grade 1-6"
    }
    v_etf_constituent {
        text etf_itm_no FK "v_etf_kr.pd_itm_no (22,599 rows / 1,162 ETFs)"
        text etf_ticker "ticker"
        text company_code "source ticker code - 79% filled"
        text company_name "constituent name"
        real weight_pct "weight %"
        int is_cash "cash-equivalent flag"
        text as_of "2026-09-05"
    }
    v_etf_gl {
        text pd_itm_no PK "item number (6,037 rows)"
        text ticker "ticker"
        text product_type "foreign ETF 5,972 / foreign ETN 65"
        text manager_name "manager"
        real expense_ratio_pct "total expense ratio %"
        real du_last_aum "net assets (AUM)"
    }
    v_fund_pub {
        text itm_no PK "class item number (23,676 rows)"
        text fund_family "fund family key - 17,795 funds"
        text manager_name "manager - 91.2% filled"
        text manager_source "fill source name/code/NULL"
        text or_co_xtn_itt_cd "manager institution code - 100% provided"
        text product_type "public 14,716 / private 8,960"
        real fd_yr1_ern_r "1-year return"
    }
    v_bond_kr {
        text pd_no "item number - not a standalone PK (21,882 rows / 20,497 issues)"
        text info_seq "OTC LOT serial number"
        text pd_nm "product name"
        text pd_exg_mkt "exchange 17,746 / OTC 4,136"
        text crd_grd "credit grade AAA-D"
        int is_sellable "available for purchase: 21,866 rows"
        int has_buy_yield "purchase yield recorded: 634 rows"
    }
```

- **There is only one real foreign key.** `v_etf_constituent.etf_itm_no` → `v_etf_kr.pd_itm_no`. The other four views cover different product groups and share no common key, so cross-product queries are solved with `UNION ALL` rather than joins.
- 1 row of `v_fund_pub` = 1 fund **class**. "How many funds" is `COUNT(DISTINCT fund_family)`.
- `v_bond_kr` has several rows per `pd_no` (exchange/OTC; OTC rows are per LOT, `info_seq`). The number of issues is `COUNT(DISTINCT pd_no)`.
- There is no constituent data for foreign ETFs or public funds.
- The diagram above shows only the join and identifier columns; in reality the 5 views have **215 columns**, all exposed through the schema cards.

### What ontology engineering we applied
- TBox of **6 files** (`common` · `bond_kr` · `etf_kr` · `etf_gl` · `fund_pub` · `etf_constituent`): `fp:Product` superclass, subclasses per product group, enumeration individuals (20-step ordered credit grades, 6 risk grades, asset class, region, distribution cycle), relation properties (managed by, issued by, holds, subsidiary, theme, index) and SHACL constraints. Of 261 property definitions, **215 are linked to actual view columns** (`fp:sourceTable`/`fp:sourceColumn`).
- **Class domains matched to how the tables are actually composed**: `v_etf_kr` holds 1,235 domestic ETFs and 545 domestic ETNs in the same columns, and `v_etf_gl` holds 5,972 foreign ETFs and 65 foreign ETNs. If a property's `domain` pointed only at the ETF class, ETN rows would be misclassified, so we added two classes, `fp:DomesticListedProduct` / `fp:ForeignListedProduct`, so the domain covers every row type in the table.
- Runtime binding: a property's `fp:sourceTable/sourceColumn/unit` → schema card; enumerations → value validators (`AA- 이상` ("AA- or higher") = automatically generated list of grades with rank ≤ 4).
- Automatic ABox construction: SQLite → **1,082,874** triples (products, managers, indices, themes, companies, families, holdings, affiliates). 73 predicates — the holding weights, previously written as a separate predicate per company (`weight_<company name>`, 5,082 predicates), were folded into a column.

### Which retrieval methodology we chose and why
- Routing by query element: numeric filter/sort/aggregation → SQL; relation traversal → graph; name and description matching → BM25 (+ Korean bigrams, exact-name boost); background knowledge → documents.
- Given the limits of any single retriever (technical-session benchmark), the LLM combines several tools to merge sources, and the evidence is structured and returned as `retrieved_context`.
  - In practice, however (05-2 ③), the tools the model actually used were heavily skewed toward `sql_query`, and five tools were never called. So rather than leaving routing to "the model will choose", we moved the design toward **loading the needed information into the prompt up front** (exposing all 215 columns, injecting pre-analysis hints).
- Vectors default to TF-IDF+LSA, reproducible without an external embedding key; commercial embeddings are available through an environment-variable switch (`VECTOR_BACKEND=openai`).
- **We measured the vector leg's actual contribution (05-2 ⑥).** For name, code and typo queries, lexical dominates and vectors cannot win. The only thing vectors retrieve on their own is theme/descriptive queries (`space exploration satellite` → Procure Space ETF, absent from the lexical top 5), which is why the current structure that gives vectors only the bottom slots is the design the data supports. Anyone can reproduce this with the audit script (`scripts/audit_vector.py`, no LLM calls, fixed seed).

## 03. System Architecture Diagram (Ingestion → Indexing → Retrieval → Generation)
(Reuse the 01 diagram + a container deployment diagram: NCP VPC/Public Subnet · Ubuntu VM · Docker Compose · public IP, port 80)
- Configured to match the organizer's notice (evaluation call method): plain HTTP · standard port 80 · no auth header · sequential calls (no concurrent requests) · 300-second timeout per question. Store loading and vector-cache warming are finished at image build time to shorten container startup, and `restart: unless-stopped` recovers automatically from failures.

## 04. Main Feature Flow (Data Collection → Ontology Design → Graph Construction → Retrieval)

```mermaid
sequenceDiagram
    participant U as Evaluator
    participant API as /answer
    participant P as Pre-analysis
    participant L as LLM
    participant T as Tools (RDB/KG/index/docs)
    U->>API: question_id, question
    API->>P: product-name extraction · entity resolution · ontology validation
    P-->>L: question + pre-analysis hints
    loop up to 8 rounds / 120 s
      L->>T: sql_query · graph_neighbors · search_products …(parallel)
      T-->>L: results + evidence (table · base date · codes)
    end
    L->>API: final_answer(status, answer)
    API-->>U: retrieved_context · think_trace · answer
```

Three example scenarios (hard/medium/easy) and two unanswerable scenarios are attached as actual response JSON (`data/eval/results.jsonl` · `data/eval/truth_results.jsonl`). The answer key has 54 questions (10 easy · 16 medium · 9 hard · 19 unanswerable).

## 05. Expected Impact and Scalability
- In the business: built-in support for advisory assistance, product discovery and compliance (evidence citation, no forecasts). Base dates and evidence are always attached, enabling audit trails.
- Extension: a new product group = 1 view + 1 TTL + lexicon entries, and that's it (the loader generates schema cards and validators automatically). The graph grows by adding pipelines for constituents and disclosure documents. Switching the LLM provider takes one environment variable.

## 05-1. Reflecting the Organizer's Notice (2026-08-24)
- **LLM**: NCP HyperCLOVA X is mandatory for the answer-generation pipeline → function calling through the CLOVA Studio OpenAI-compatible endpoint (`HCX-005`). Thanks to the adapter structure, the cost of switching providers was close to zero.
- **Changed definition of "available for purchase"**: following the notice that `BUYABLE_QUANTITY` is invalid, redefined as "not yet redeemed = available for purchase" (bonds 634 → 21,866). Because this item changes evaluation answers substantially, the ETL, tests and prompts were revised together.
- **Handling missing values / zeros**: they are intentional, so they are not counted as 0; they are excluded or explicitly stated as "not provided".
- **Cross-product queries**: a product-group UNION pattern is built into the prompt. Foreign ETFs have no 1-year return, so the prompt instructs the model to exclude them from comparisons and say so in the answer.
- **Response time**: to fit the 300-second no-response rule, the internal budget was raised to 120 seconds and 8 rounds (accuracy first, with margin before 300 seconds). Measurements showed that what causes latency is not prompt size but **the number of round-trips** (05-2).

## 05-2. What We Confirmed by Measurement (2026-09-06)

Below we list only the design claims that are backed by measurement. Anything we did not measure, we say we did not measure.

**① Prompt size does not cause latency.** For prompts of 599 / 5,591 / 8,789 / 13,495 tokens, the median first-call latency was 2.90 / 3.65 / 2.94 / 3.82 seconds. Practically flat while the prompt grew 22-fold. The 35-second call observed earlier was not due to size but **waiting on the rate limit of the test-app tier**.
→ So putting all 215 columns into the schema cards costs nothing in response time.

**② What causes latency is the number of round-trips.** The marginal cost of one more round is 1→2 **+11.2 s**, 2→3 +1.8 s, 3→4 +3.7 s. The first round-trip is the most expensive. Yet **17 of 22 questions** spend one round, right before submitting, emitting only text without calling a tool — a round-trip that adds nothing to the answer. The correlation between answer length and elapsed time was **0.00** (writing longer is not what makes it slow).

**③ The model does not ask about the schema.** Tool calls over 22 questions: `sql_query` 24, `final_answer` 20, `graph_neighbors` 2, and `table_info` · `search_products` · `resolve_entity` · `graph_query` · `search_documents` **0 each**.
→ A column that is not in the prompt from the start effectively does not exist for the model. In fact, `has_buy_yield` was in the prompt's column list but had no ontology definition, so it never appeared in the schema card, and the model never used that column. This is why we decided **not to select among the provided columns but to expose all 215**.

**④ Provenance of the knowledge graph.** The current graph has **1,082,874 triples** (965,823 asserted + 117,051 inferred · 73 predicates). Provenance was measured at 1,074,824 triples: organizer masters **58.9%**, ontology/inference **27.0%**, our lexicons **8.1%**, external collection (domestic ETF constituents) **6.0%** — 85.9% is rooted in the provided data, and external collection covers only one item for which the provided data has no column at all. The affiliate relations added afterwards (608 edges) are 0.7% of the total and move each share by less than 1%p.

**⑤ SQL and the graph give the same answer.** After moving constituents into a SQLite table and generating the graph from that table, we cross-checked the two paths on five companies: Samsung Electronics 229/229 · Samsung Electronics preferred 19/19 · EcoPro BM 42/42 · LG 26/26 · KT 48/48.

**⑥ How much the vector index adds on top of lexical search.** Using **1,313 queries** generated automatically from the data (4 product groups · 7 types: official name/code/abbreviation/constituent name/theme word/typo/synonym), we measured the lexical, vector and combined modes separately.

| Type | Lexical recall@10 | Vector recall@10 | Combined recall@10 |
|---|---|---|---|
| Official name | 100.0% | 60.0% | 100.0% |
| Code · ticker | 49.4% | 1.9% | 49.4% |
| Typo | 93.1% | 45.0% | 93.1% |
| Theme word | 87.0% | 55.8% | **90.5%** |

The combined mode's recall@1 and @5 are **exactly identical to lexical-only for every type** — because, by construction, vectors do not touch the top slots, which also means the design goal "hybrid is never worse than lexical" holds. Of the 1,313 queries, vectors alone rescued 22 (1.7%) and the combination lost 6 (0.5%), **all in the tail slots, ranks 8–10**.
→ So we do not write that "semantic search absorbs typos and synonyms". **Typo tolerance comes from BM25's Korean bigrams** (lexical 93.1% vs. vector 45.0%). The only area where vectors win is queries where the user knows the product's character but not its name.

**⑦ Answer accuracy (2026-09-06 · `HCX-005`).** On our own 54-question answer key: **47/54 (87%)**. By difficulty: easy 8/10 · medium 14/16 · hard 6/9 · **unanswerable 19/19** — zero cases of answering a question that cannot be answered. Average response **20.6 s**, longest 87.4 s, **0 over 300 s**. Per-question verdicts, elapsed times, evidence and full traces are in `data/eval/truth_results.jsonl`.
Because we fixed errors in the key itself while expanding it from 20 to 54 questions (counting ETNs in a question about ETFs; pinning a single correct answer when 4 products tied at the lowest total expense ratio of 0.01%; allowing rounded integers so that 0.65 also passed as "1"; a grader that did not recognize "…상품은 없습니다" ("there is no such product") as a refusal), the figures above are **based on the corrected new set**, and earlier grading results were discarded.
We also measured separately on the 8 sample questions on p12 of the organizer's task description — **7/8** (4 of the 5 difficulty-rated questions correct, all 3 unanswerable questions correct). All **194 unit tests pass**.

**Not yet measured (we do not give numbers)**

- Container startup time (needs re-measurement after adding the constituent table and rebuilding the graph).
- Concurrent-request throughput — the organizer announced sequential calls, so we did not load-test.
- Change in retrieval quality when switching to commercial embeddings (`VECTOR_BACKEND=openai`).

**Our quality-control principle** — all 11 defects found in this review passed with no exception, no log and no red test. Relation counts were inflated twofold and still "correct" (because the answer key was computed with the same code), counts cut at a cap were reported under the name of totals, and a step that filled no rows reported success. So this project makes it a procedure to **recount results after every change** — counting a value obtained by SQL once more through the graph, and a value obtained through the graph once more by SQL, and comparing them.

## 06. Appendix
- Ontology schema summary table (classes/properties/enumerations: 6 files · 215 of 261 properties linked to view columns), API specification (`docs/api_spec.md`), evaluation reproduction procedure (README section 7), test list, data limitations, system anatomy (`docs/anatomy.html` — including charts and ERD).

---

## 한국어

[English](#technical-proposal-draft-following-the-recommended-outline--financial-product-agent) | 한국어

*참고: 이 초안은 대회 규정상 NCP HyperCLOVA X 사용이 필수였던 대회 제출본을 기준으로 합니다. 이 영어판은 대신 Google Gemini(기본 모델 `gemini-3.8-flash`)로 동작합니다.*

# 기술제안서 초안 (권장 목차 기준) — 금융상품 Agent

> 제출용 PDF 작성을 위한 뼈대. 각 절의 다이어그램은 mermaid 로 그려 두었으니 PDF 변환 시 이미지로 내보내면 됩니다.

## 01. 전체 아키텍처 및 특화 기능

**한 줄 요약**: 온톨로지를 "문서"가 아닌 "실행 계층"으로 쓰는 Federated Query Agent. 정형 마스터는 SQL, 관계(운용사·테마·편입종목·자회사)는 그래프, 이름·서술 텍스트는 어휘 색인, 배경 지식은 외부 문서로 각각 라우팅하고, 답변 전에 값 도메인을 검증해 환각과 답변불가 문항 오답을 줄인다.

차별화 포인트
1. **사전 분석 → 라운드 0 검색 실행 계획(Deterministic pre-analysis)** — LLM 호출 전에 상품명 스팬 추출·엔티티 해석·테마/기업 사전 매칭·온톨로지 값 검증을 수행하고, 그 결과로 그래프·벡터·문서 중 *그 질문에 필요한 것만* 골라 미리 실행해 결과 행을 첫 프롬프트에 **사실로 주입**한다(54문항 기준 그래프 6 · 벡터 7 · 문서 0이 발동하고 41문항은 아무것도 발동하지 않는다) → 라운드 수 감소(응답 시간↓), 답변불가 문항(AAAA, 7등급, 미존재 상품) 조기 판별. 해소한 식별자(예: 캠브리콘 → 스토어 표기 `한무기`)와 조회용 ID 목록도 함께 넘겨 SQL이 이름으로 헛도는 것을 막는다.
2. **온톨로지 → 스키마 카드/검증기 자동 생성** — TTL의 `fp:sourceColumn`·열거형 개체·SHACL 을 런타임이 읽어 프롬프트와 검증기를 만든다(기술세션의 "런타임 식별 +20pp" 관점).
3. **측정 기반 하이브리드 검색 — 그리고 그 이득의 크기까지 측정** — BM25와 벡터(TF-IDF+LSA 384차원)를 RRF로 대등 결합했더니 두 검색기가 함께 중간 순위로 올린 무관 문서가 최상위 어휘 매칭을 밀어내는 현상을 관측(“배터리 양극재” 1위 오염). 최종안은 어휘 결과가 head를, 의미 검색 전용 문서가 하위 30% 슬롯을 갖는 구조다. 질의 1,313개로 재보니 이 구조는 **하이브리드의 recall@1·@5가 어휘 단독과 소수점까지 동일**함을 보장하며(벡터가 상위 순위를 바꾸지 않음), 이득은 테마형 질의의 **recall@10 +3.5%p**에 국한된다(87.0% → 90.5%). 우리는 이 수치를 감추지 않는다 — TF-IDF+LSA는 의미 임베딩이 아니라 어휘 공기 기반이므로 이 정도가 이 방식의 상한이다(05-2 ⑥).
4. **온톨로지 추론 물질화** — TBox를 런타임 그래프에 병합하고 상위 클래스 상속·역관계·domain/range를 적용. `?p a fp:Product`, `?am fp:manages ?p` 같은 질의가 별도 적재 없이 성립(기술세션 01-4의 “2개를 넣으면 5개가 된다”를 실제 파이프라인으로 구현). 현재 그래프 **1,082,874 트리플**(주장 965,823 + 추론 117,051) **· 술어 73종**.
5. **편입종목을 RDB와 그래프에 동시에** — 국내 ETF 1,162종목의 상위 편입종목 22,599행을 수집해 **SQLite 테이블 `etf_constituent`로 적재**하고, 지식그래프의 `hasConstituent`를 그 테이블에서 생성. 같은 원천을 쓰므로 SQL 결과와 그래프 결과가 갈라지지 않는다(삼성전자 229/229, 에코프로비엠 42/42로 대조 확인). 기업 식별은 원천이 주는 **종목코드**로만 하고 이름 유사도는 쓰지 않는다.
   - 다만 `subsidiaryOf`(모회사·자회사)는 **엣지 4개**뿐이다. 주최측 4종 마스터에 소유 관계 컬럼이 아예 없어, 제공 데이터에서 계산한다는 규칙을 지켜 확장하지 않았다. "다단계 자회사 순회"를 일반 기능으로 주장하지 않으며, 실제로 동작하는 범위는 사전에 명시한 에코프로 계열·SK이노베이션이다.
6. **답변 정책 상태기계** — `answered / cannot_confirm / clarify` 3상태를 도구 스키마로 강제하고, 위반 시 서버가 보정.
7. **무중단 계약** — 항상 200 + 5필드, 시간 예산 120초·최대 8라운드, 예산 임박 시 강제 종료 후 확보 근거로 답변.

```mermaid
flowchart TB
    subgraph Ingest[적재]
      X[Excel 4종] --> E[ETL: 정규화·파생컬럼·뷰 5개]
      C[네이버금융 편입종목<br/>22,599행] --> E
      E --> S[(SQLite<br/>5 테이블 · 5 뷰 · 215 컬럼)]
      Y[사전 YAML<br/>운용사 77·테마 48·기업 97] --> K
      T[ontology/*.ttl 6종] --> L[Loader: 스키마카드·검증기]
    end
    subgraph Index[색인]
      S --> B[BM25 색인]
      S --> VEC[벡터 색인 TF-IDF+LSA]
      S --> K[(Knowledge Graph)]
      M[외부 문서 MD] --> D[문서 청크 색인]
    end
    subgraph Serve[검색·생성]
      Q[질문] --> P[사전 분석] --> A[LLM 도구 루프] --> F[final_answer] --> J[JSON 응답]
      A --- S & K & B & VEC & D
      L -.-> P & A
    end
```

## 02. 제안 방법

### 어떤 특화 데이터를 어떻게 수집·정제했는가
- 주최측 4종 마스터: 코드형 숫자 컬럼을 문자열로 고정(선행 0 보존), yyyymmdd→ISO, `is_sellable`(상환 미도래 채권 **21,866행**)·`has_buy_yield`(매수수익률 수록 634행)·`is_on_sale`·`fund_family`(펀드 클래스 23,676행 = **17,795 패밀리**)·`manager_name` 파생. 기준일을 `meta` 테이블에 기록해 근거에 자동 표기. 5개 뷰의 **제공 컬럼 215개를 전량 노출**(선별하지 않음 — 이유는 05-2 참조).
- 편입종목: 네이버금융 API(상위 종목·비중), 1,235 ETF 중 1,162 성공, 22,599행. **SQLite 테이블로 적재**하고 그래프를 그 테이블에서 생성. 원천이 주는 `itemCode`를 보존해 79% 행이 종목코드를 갖는다. 캐시·재수집 스크립트 제공.
- 사전: 운용사 77개(브랜드 접두어 TIGER/KODEX/RISE…), 테마 48개(한/영 동의어), 기업 97개(별칭·티커·모회사·비상장 여부, 한무기=캠브리콘 등 중국명 대응).
- 비정형: 국민성장펀드 보도자료 등을 PDF→Markdown(`parse_docs.py`)으로 변환, front-matter에 출처·기준일.

### 결측치를 어떻게 식별하고 보완했는가

전제: 주최측 공지에 따라 **제공 데이터의 0·결측은 의도된 값**이다. 그래서 원본 컬럼의 결측은 채우지 않고 계산에서 제외하며, 답변에는 "제공되지 않음"이라고 밝힌다. 보완 대상은 *우리가 만든 파생 컬럼*이며, 보완의 근거는 반드시 **제공 데이터 안**에 있어야 한다.

**식별 방법** — 뷰별로 컬럼 NULL 비율을 전수 점검하고, 파생 컬럼은 그 유도 근거(어느 원본 컬럼에서 왔는가)까지 함께 기록했다. 이 점검에서 파생 컬럼 하나가 눈에 띄었다.

**작업 예시: 펀드 운용사(`manager_name`) 복구**

| 단계 | 내용 |
|---|---|
| 증상 | 공모펀드 23,676행 중 **25.4%(6,016행)** 에 운용사가 비어 있음 |
| 원인 | 운용사를 **펀드 이름의 접두어에서 추측**하고 있었다. 이름에 브랜드가 드러나지 않는 펀드는 채울 방법이 없었다 |
| 더 나은 근거 | 주최측이 `or_co_xtn_itt_cd`(운용사 기관코드)를 **100% 채워서 제공**하고 있었다. 추측이 아니라 코드에서 유도할 수 있다 |
| 보완 | 기관코드별로 관측된 운용사 표기를 모아 다수 표기를 대표명으로 삼고, 이름 접두어로 못 채운 행을 코드로 채움 |
| 결과 | 17,660행(74.6%) → **21,592행(91.2%)**. 남은 NULL 2,084행 |
| 추적성 | 어느 근거로 채웠는지를 새 컬럼 **`manager_source`** 에 남김 — `name` 17,660 / `code` 3,932 / NULL 2,084 |

**보완하지 않기로 한 것**

- **운용사 개명 대응표**: 한 기관코드에 옛 브랜드와 새 브랜드가 함께 관측되는 경우가 있다. 그러나 제공 데이터에 **펀드 설정일이 없어** 두 표기 중 어느 쪽이 현재 이름인지 판정할 근거가 없다. 외부 표를 끌어오면 규칙("제공 데이터에서 계산") 밖으로 나가므로, 다수 표기를 그대로 쓴다 — 그것이 옛 브랜드여도.
- **국내ETF 총보수(18%만 수록) · 펀드 수익률(약 30%만 수록) · 펀드 보수 컬럼의 단위**: 원천이 주지 않은 값이다. 추정하지 않고 "데이터 없음"으로 답한다. 펀드 보수는 단위 미공개이므로 **클래스 간 상대 비교만** 허용한다.
- **`subsidiaryOf`(모회사·자회사)**: 주최측 4종 마스터에 소유 관계 컬럼이 없다. 사전에 손으로 적은 **엣지 4개**가 전부이며, 이를 일반 기능으로 주장하지 않는다.

**식별 방법 하나를 더 적어 둔다** — 결측이 아니라 *조용한 오류*를 잡는 방법이다. 파이프라인의 한 채움 단계는 `pandas`의 `Series.map`이 `None`을 돌려주면 `NaN`이 되고 `NaN`은 참이라는 이유로 **아무 행도 채우지 않은 채 성공을 보고**하고 있었다. 예외도 없고 로그도 없고 테스트도 초록이었다. 그래서 이 프로젝트는 각 채움 단계마다 **작업 전후의 채움 건수를 다시 세어 비교하는 것**을 절차로 두었다. 성공 여부가 아니라 개수를 본다.

### 데이터 모델 (뷰 5개)

```mermaid
erDiagram
    v_etf_kr ||--o{ v_etf_constituent : "pd_itm_no = etf_itm_no"

    v_etf_kr {
        text pd_itm_no PK "종목번호 (1,780행)"
        text pd_ticker "티커"
        text pd_nm "상품명"
        text product_type "국내ETF 1,235 / 국내ETN 545"
        text manager_name "운용사"
        real pd_net_tamt "순자산 총액"
        int risk_grade "위험등급 1~6"
    }
    v_etf_constituent {
        text etf_itm_no FK "v_etf_kr.pd_itm_no (22,599행 / 1,162 ETF)"
        text etf_ticker "티커"
        text company_code "원천 종목코드 - 79% 채움"
        text company_name "편입 종목명"
        real weight_pct "편입 비중 %"
        int is_cash "현금성 여부"
        text as_of "2026-09-05"
    }
    v_etf_gl {
        text pd_itm_no PK "종목번호 (6,037행)"
        text ticker "티커"
        text product_type "해외ETF 5,972 / 해외ETN 65"
        text manager_name "운용사"
        real expense_ratio_pct "총보수 %"
        real du_last_aum "순자산 (AUM)"
    }
    v_fund_pub {
        text itm_no PK "클래스 종목번호 (23,676행)"
        text fund_family "펀드 패밀리 키 - 17,795 펀드"
        text manager_name "운용사 - 91.2% 채움"
        text manager_source "채움 출처 name/code/NULL"
        text or_co_xtn_itt_cd "운용사 기관코드 - 100% 제공"
        text product_type "공모 14,716 / 사모 8,960"
        real fd_yr1_ern_r "1년 수익률"
    }
    v_bond_kr {
        text pd_no "종목번호 - 단독 PK 아님 (21,882행 / 20,497 종목)"
        text info_seq "장외 LOT 일련번호"
        text pd_nm "상품명"
        text pd_exg_mkt "장내 17,746 / 장외 4,136"
        text crd_grd "신용등급 AAA~D"
        int is_sellable "구매가능 21,866행"
        int has_buy_yield "매수수익률 수록 634행"
    }
```

- **실제 외래키는 하나뿐이다.** `v_etf_constituent.etf_itm_no` → `v_etf_kr.pd_itm_no`. 나머지 네 뷰는 상품군이 달라 공통 키가 없으므로 교차질의는 조인이 아니라 `UNION ALL`로 푼다.
- `v_fund_pub` 1행 = 펀드 **클래스** 1개. "펀드 몇 개"는 `COUNT(DISTINCT fund_family)`.
- `v_bond_kr`은 한 `pd_no`에 여러 행(장내/장외, 장외는 LOT 단위 `info_seq`). 종목 수는 `COUNT(DISTINCT pd_no)`.
- 해외ETF·공모펀드의 편입종목 데이터는 없다.
- 위 표는 조인·식별 컬럼만 뽑은 것이고, 실제로는 5개 뷰에 **215개 컬럼**이 있으며 전량이 스키마 카드로 노출된다.

### 어떤 온톨로지 엔지니어링을 적용했는가
- TBox **6파일**(`common` · `bond_kr` · `etf_kr` · `etf_gl` · `fund_pub` · `etf_constituent`): `fp:Product` 상위 클래스, 상품군별 하위 클래스, 열거형 개체(신용등급 20단계 서열, 위험등급 6단계, 자산군·지역·분배주기), 관계 속성(운용·발행·편입·자회사·테마·지수), SHACL 제약. 속성 정의 261개 중 **215개가 실제 뷰 컬럼에 연결**(`fp:sourceTable`/`fp:sourceColumn`).
- **클래스 도메인을 테이블 실제 구성에 맞춤**: `v_etf_kr`은 국내ETF 1,235건과 국내ETN 545건을, `v_etf_gl`은 해외ETF 5,972건과 해외ETN 65건을 같은 컬럼에 담는다. 속성 `domain`이 ETF 클래스만 가리키면 ETN 행이 잘못 분류되므로, `fp:DomesticListedProduct` / `fp:ForeignListedProduct` 두 클래스를 추가해 도메인이 테이블의 모든 행 유형을 덮게 했다.
- 런타임 결합: 속성의 `fp:sourceTable/sourceColumn/unit` → 스키마 카드; 열거형 → 값 검증기(`AA- 이상` = 서열 ≤ 4 등급 목록 자동 생성). 
- ABox 자동 구축: SQLite→트리플 **1,082,874** 개(상품·운용사·지수·테마·기업·패밀리·편입·계열사). 술어는 73종 — 편입 비중을 기업마다 다른 술어(`weight_<기업명>`, 5,082종)로 적던 것을 컬럼으로 접었다.

### 어떤 검색 방법론을 선택했고 왜 그렇게 설계했는가
- 질의 요소별 라우팅: 숫자 필터/정렬/집계 → SQL, 관계 순회 → 그래프, 이름·서술 매칭 → BM25(+한글 bigram, 정확명 부스트), 배경 지식 → 문서.
- 단일 검색기의 한계(기술세션 벤치마크)를 고려해 LLM이 여러 도구를 조합해 소스를 결합하고, 근거를 구조화해 `retrieved_context` 로 반환.
  - 다만 실측(05-2 ③)에서 모델이 실제로 쓴 도구는 `sql_query`에 크게 치우쳤고 다섯 개 도구는 한 번도 호출되지 않았다. 라우팅은 "모델이 알아서 고른다"에 맡기지 않고, **필요한 정보를 프롬프트에 미리 실어 주는 쪽**으로 설계를 옮겼다(컬럼 215개 전량 노출, 사전 분석 힌트 주입).
- 벡터는 외부 임베딩 키 없이 재현 가능한 TF-IDF+LSA를 기본으로 하고, 상용 임베딩은 환경변수 전환(`VECTOR_BACKEND=openai`)으로 열어 둠.
- **벡터 레그의 실제 기여를 측정했다(05-2 ⑥).** 이름·코드·오타 질의에서는 어휘가 압도하고 벡터는 이기지 못한다. 벡터가 단독으로 건지는 것은 테마·서술형 질의뿐이며(`space exploration satellite` → Procure Space ETF, 어휘 top-5 부재), 그래서 하위 슬롯만 배정하는 현재 구조가 데이터가 지지하는 설계다. 감사 스크립트(`scripts/audit_vector.py`, LLM 미호출·시드 고정)로 누구나 재현할 수 있다.

## 03. 시스템 구성도 (적재 → 색인 → 검색 → 생성)
(01 다이어그램 재사용 + 컨테이너 배포도: NCP VPC/Public Subnet · Ubuntu VM · Docker Compose · 공인 IP 80번 포트)
- 주최측 공지(평가 호출 방식)에 맞춘 구성: HTTP 기본·표준 포트 80·인증 헤더 없음·순차 호출(동시 요청 없음)·문항당 타임아웃 300초. 이미지 빌드 시점에 스토어 적재와 벡터 캐시 워밍을 끝내 컨테이너 기동을 단축하고, `restart: unless-stopped` 로 장애 시 자동 복구한다.

## 04. 주요 기능 흐름도 (데이터 수집 → 온톨로지 설계 → 그래프 구축 → 검색)

```mermaid
sequenceDiagram
    participant U as 평가자
    participant API as /answer
    participant P as 사전분석
    participant L as LLM
    participant T as 도구(RDB/KG/색인/문서)
    U->>API: question_id, question
    API->>P: 상품명 추출·엔티티 해석·온톨로지 검증
    P-->>L: 질문 + 사전분석 힌트
    loop 최대 8라운드 / 120초
      L->>T: sql_query · graph_neighbors · search_products …(병렬)
      T-->>L: 결과 + 근거(테이블·기준일·코드)
    end
    L->>API: final_answer(status, answer)
    API-->>U: retrieved_context · think_trace · answer
```

예시 시나리오 3개(상/중/하)와 답변불가 시나리오 2개를 실제 응답 JSON으로 첨부(`data/eval/results.jsonl` · `data/eval/truth_results.jsonl`). 정답 키는 54문항(하 10 · 중 16 · 상 9 · 답변불가 19).

## 05. 기대효과·확장성
- 현업: 상담 보조·상품 탐색·컴플라이언스(근거 표기, 전망 금지) 내재화. 기준일·근거가 항상 붙어 감사 추적 가능.
- 확장: 새 상품군 = 뷰 1개 + TTL 1개 + 사전 항목 추가로 끝(로더가 스키마 카드·검증기 자동 생성). 편입종목·공시 문서 파이프라인 추가로 그래프 확대. LLM 제공자 교체는 환경변수 1개.

## 05-1. 주최측 공지 반영 (2026-08-24)
- **LLM**: 답변 생성 파이프라인 NCP HyperCLOVA X 필수 → CLOVA Studio OpenAI 호환 엔드포인트(`HCX-005`)로 function calling 수행. 어댑터 구조라 제공자 교체 비용이 0에 가까웠음.
- **구매가능 정의 변경**: `BUYABLE_QUANTITY` 무효 공지에 따라 "상환 미도래 = 구매가능"으로 재정의(채권 634건 → 21,866건). 평가 답이 크게 달라지는 항목이라 ETL·테스트·프롬프트를 함께 수정.
- **결측/0 처리**: 의도된 결측이므로 0으로 계산하지 않고 제외 또는 "제공되지 않음" 명시.
- **교차질의**: 상품군 UNION 패턴을 프롬프트에 내장. 해외ETF는 1년 수익률 미수록이므로 비교 대상에서 제외하고 그 사실을 답변에 밝히도록 지시.
- **응답시간**: 300초 미응답 기준에 맞춰 내부 예산 120초·8라운드로 상향(정확도 우선, 300초 여유 확보). 실측 결과 지연을 만드는 것은 프롬프트 크기가 아니라 **왕복 횟수**였다(05-2).

## 05-2. 측정으로 확인한 것 (2026-09-06)

설계 주장 중 측정으로 뒷받침되는 것만 아래에 적는다. 측정하지 않은 것은 측정하지 않았다고 적는다.

**① 프롬프트 크기는 지연을 만들지 않는다.** 프롬프트 599 / 5,591 / 8,789 / 13,495 토큰에 대한 첫 호출 지연 중앙값은 2.90 / 3.65 / 2.94 / 3.82초였다. 22배 커지는 동안 사실상 평평하다. 이전에 관측된 35초짜리 호출은 크기 때문이 아니라 **테스트앱 등급의 rate limit 대기**였다.
→ 따라서 스키마 카드에 컬럼 215개를 전부 실어도 응답 시간에서 손해를 보지 않는다.

**② 지연을 만드는 것은 왕복 횟수다.** 라운드를 하나 더 도는 한계 비용은 1→2 **+11.2초**, 2→3 +1.8초, 3→4 +3.7초. 첫 왕복이 가장 비싸다. 그런데 22문항 중 **17문항**이 제출 직전에 도구를 부르지 않고 텍스트만 내보내는 라운드를 한 번 소비한다 — 답에 아무것도 더하지 않는 왕복이다. 답변 길이와 소요 시간의 상관계수는 **0.00**이었다(길게 쓰는 게 느린 것이 아니다).

**③ 모델은 스키마를 물어보지 않는다.** 22문항 동안의 도구 호출은 `sql_query` 24회, `final_answer` 20회, `graph_neighbors` 2회, 그리고 `table_info`·`search_products`·`resolve_entity`·`graph_query`·`search_documents`는 **각각 0회**였다.
→ 프롬프트에 처음부터 실려 있지 않은 컬럼은 모델에게 존재하지 않는 것과 같다. 실제로 `has_buy_yield`는 프롬프트의 컬럼 목록에는 있었으나 온톨로지 정의가 없어 스키마 카드에 나타나지 않았고, 모델은 그 컬럼을 쓰지 않았다. 이것이 **제공 컬럼을 선별하지 않고 215개 전량을 노출**하기로 한 이유다.

**④ 지식그래프의 출처 구성.** 현재 그래프는 **1,082,874 트리플**(주장 965,823 + 추론 117,051 · 술어 73종)이다. 출처 구성은 1,074,824 트리플 시점에 측정했고 주최측 마스터 유래 **58.9%**, 온톨로지·추론 **27.0%**, 우리 사전 **8.1%**, 외부 수집(국내 ETF 편입종목) **6.0%** 였다 — 제공 데이터에 뿌리를 둔 것이 85.9%이며, 외부 수집은 제공 데이터에 컬럼 자체가 없는 항목 하나뿐이다. 이후 추가된 계열사 관계(608엣지)는 전체의 0.7%라 각 항목의 비율을 1%p 미만 움직인다.

**⑤ SQL과 그래프가 같은 답을 낸다.** 편입종목이 SQLite 테이블로 내려오고 그래프가 그 테이블에서 생성되도록 바꾼 뒤, 다섯 기업으로 두 경로를 대조했다: 삼성전자 229/229 · 삼성전자우 19/19 · 에코프로비엠 42/42 · LG 26/26 · KT 48/48.

**⑥ 벡터 색인이 어휘 검색 위에 더하는 것의 크기.** 데이터에서 자동 생성한 질의 **1,313개**(4개 상품군 · 정식명칭/코드/약칭/편입종목명/테마어/오타/유의어 7유형)로 어휘·벡터·결합 세 모드를 각각 측정했다.

| 유형 | 어휘 recall@10 | 벡터 recall@10 | 결합 recall@10 |
|---|---|---|---|
| 정식명칭 | 100.0% | 60.0% | 100.0% |
| 코드·티커 | 49.4% | 1.9% | 49.4% |
| 오타 | 93.1% | 45.0% | 93.1% |
| 테마어 | 87.0% | 55.8% | **90.5%** |

결합의 recall@1·@5는 **전 유형에서 어휘 단독과 완전히 동일**하다 — 구조상 벡터가 상위 슬롯에 개입하지 않기 때문이며, 이는 "하이브리드가 어휘보다 나빠지지 않는다"는 설계 목표가 지켜졌다는 뜻이기도 하다. 1,313건 중 벡터가 단독으로 구제한 것은 22건(1.7%), 결합이 잃은 것은 6건(0.5%)이고 **전부 8~10위 꼬리 슬롯**에서 일어났다.
→ 그래서 우리는 "의미 검색이 오타·유의어를 흡수한다"고 쓰지 않는다. **오타 내성은 BM25의 한글 bigram이 제공한다**(어휘 93.1% 대 벡터 45.0%). 벡터가 이기는 구간은 이름을 모르고 성격만 아는 질의 하나뿐이다.

**⑦ 정답 정확도 (2026-09-06 · `HCX-005`).** 자체 정답 키 54문항 기준 **47/54 (87%)**. 난이도별로 하 8/10 · 중 14/16 · 상 6/9 · **답변불가 19/19** — 답할 수 없는 질문을 답해버리는 사례는 0건이다. 평균 응답 **20.6초**, 최장 87.4초로 **300초 초과 0건**. 문항별 판정·소요시간·근거·추적 전문은 `data/eval/truth_results.jsonl`에 있다.
정답 키를 20 → 54문항으로 확장하면서 키 자체의 오류도 함께 고쳤으므로(ETF를 묻는 문항에서 ETN을 센 것, 최저 총보수 0.01%에 4개가 동률인데 하나만 정답으로 고정한 것, 반올림 정수를 허용해 0.65가 "1"로도 통과한 것, "…상품은 없습니다"를 거절로 인식하지 못한 채점기), 위 수치는 **고친 새 세트 기준**이고 이전 채점 결과는 폐기했다.
주최측 과제설명 p12의 예시 8문항으로도 따로 재봤다 — **7/8**(난이도 5문항 중 4문항 정답, 답변불가 3문항 전부 정답). 단위 테스트는 **194개 전부 통과**한다.

**아직 측정하지 않은 것 (숫자를 쓰지 않는다)**

- 컨테이너 기동 시간(편입종목 테이블 추가·그래프 재구축 이후 재측정 필요).
- 동시 요청 처리량 — 주최측이 순차 호출을 공지해 부하 시험을 하지 않았다.
- 상용 임베딩(`VECTOR_BACKEND=openai`) 전환 시의 검색 품질 변화.

**품질 관리 원칙으로 삼은 것** — 이번 점검에서 발견한 결함 11건은 전부 예외도 로그도 빨간 테스트도 없이 통과했다. 관계 수가 두 배로 부풀려진 채 "정답"이었고(정답 키를 같은 코드로 계산했기 때문), 상한에 잘린 개수가 총계라는 이름으로 보고됐고, 아무 행도 채우지 않은 단계가 성공을 보고했다. 그래서 이 프로젝트는 **바꾼 뒤에 결과를 다시 세는 것**을 절차로 둔다 — SQL로 나온 값을 그래프로, 그래프로 나온 값을 SQL로 한 번 더 세어 대조한다.

## 06. 부록
- 온톨로지 스키마 요약표(클래스/속성/열거형: 6파일 · 속성 261개 중 215개가 뷰 컬럼에 연결), API 명세(`docs/api_spec.md`), 평가 재현 절차(README 7절), 테스트 목록, 데이터 제한 사항, 시스템 해부도(`docs/anatomy.html` — 차트·ERD 포함).
