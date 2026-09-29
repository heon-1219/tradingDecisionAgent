**English** | [한국어](#한국어)

# Financial Product Agent

**Financial Product Analyst API, English edition.** This is team **Wolves of Walnut Street**'s entry for the Financial Product Agent track (금융상품 Agent) of the 10th Mirae Asset Securities AI Festival 2026.

`GET /answer` takes a question about financial products and returns an answer grounded in data. An LLM agent explores and computes over four structured product masters supplied by the organizers (Korean bonds, Korean-listed ETFs/ETNs, US-listed ETFs/ETNs and Korean funds) plus the holdings of Korean ETFs. It attaches evidence that has been checked against a domain ontology.

> **Every measured result in this README comes from the competition build**, which ran NCP HyperCLOVA X (`HCX-005`) because the organizers required it. The English edition runs on Google Gemini and has not been re-benchmarked on Gemini.

## 1. What changed in the English edition

| Area | Competition build | English edition |
|---|---|---|
| LLM | NCP HyperCLOVA X `HCX-005` through CLOVA Studio's OpenAI-compatible endpoint (organizer rule) | Google Gemini (`gemini-3.8-flash` by default) through Gemini's OpenAI-compatible endpoint. Anthropic and OpenAI (or any OpenAI-compatible server) remain as alternates |
| Answer language | Korean | Follows the question: English by default, Korean when the question is written in Korean |
| `think_trace` / `retrieved_context` labels | Korean | English |
| Test console (`/ui`) | Korean | English by default, with an EN / 한국어 toggle |
| Documents | Korean | English first, Korean second: this README, the technical proposal, `docs/anatomy.html` |
| Data | Korean | Unchanged. Still Korean |

- **Gemini through the OpenAI-compatible endpoint.** The agent reaches Gemini through the `openai` package that the project already used, so Gemini needs no extra dependency. `LLM_EFFORT` becomes Gemini's `reasoning_effort`. Thought signatures are echoed back so that multi-turn tool calls work, and 429 and 5xx responses are retried. `python scripts/check_llm.py` checks the key, the model, function calling and latency with a single call. Details are in [section 5](#5-configuration-env).
- **The answer follows the question's language.** [`src/finagent/i18n.py`](src/finagent/i18n.py) decides the language. Words written only in Hangul vote Korean by syllable count, and words without Hangul vote English by lowercase-letter count. Mixed-script product names such as `미국S&P500` and all-caps names such as `TIGER` or `ETF` don't vote, and a tie goes to English. So a Korean product name inside an English question does not flip the answer to Korean: "Which ETFs hold 삼성전자?" gets an English answer, and "TIGER 미국S&P500 총보수는?" gets a Korean one.
- **The data stays Korean.** Korean bonds, Korean ETFs/ETNs, US-listed ETFs, Korean funds, ETF holdings and the reference documents are unchanged. English questions may therefore name Korean products and companies, in Korean or in English. The agent maps English company names to the form the data uses (for example, "Samsung Electronics" becomes 삼성전자). The company lexicon ([`data/enrichment/companies.yaml`](data/enrichment/companies.yaml)) holds each company's English name and aliases.
- **The test console** at `/ui` ([`web/console.html`](web/console.html)) is in English by default, and an EN / 한국어 toggle switches its language.
- **Documents.** The technical proposal has an English translation at [`Wolves_of_Walnut_Street_Proposal.html`](Wolves_of_Walnut_Street_Proposal.html) / [`.pdf`](Wolves_of_Walnut_Street_Proposal.pdf), and the Korean original is kept. [`docs/anatomy.html`](docs/anatomy.html) is in English, and [`docs/anatomy.ko.html`](docs/anatomy.ko.html) is the Korean original.
- **Removed:** the `clova` provider along with `CLOVA_API_KEY` / `CLOVA_BASE_URL`. [Section 14](#14-competition-build-notes-hyperclova-x) keeps the CLOVA-specific history.

## 2. Quick start

Requirements:

- Python 3.11+. The Docker image uses 3.12, and development was on 3.14.
- 4 GB+ RAM, for the knowledge-graph build.
- Internet access, for the LLM API.
- A Gemini API key from <https://aistudio.google.com/apikey>.

```bash
git clone <this repo> fin-agent-en && cd fin-agent-en
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then set GEMINI_API_KEY=... in .env
```

```bash
# 1) Build the SQLite store from data/raw: Excel masters + constituent CSV -> data/store/products.db (about 1-2 min)
python scripts/build_store.py

# 2) (optional) Precompute the vector cache (data/store/vector_cache.npz); saves about 80 s on every start
python scripts/warmup.py

# 3) Check the LLM: key, model access, function calling and latency (one call, a few seconds)
python scripts/check_llm.py

# 4) Start the API. Startup builds the search index, the vectors and the knowledge graph with inference:
#    about 1.5 min with the vector cache, 3-3.5 min without (see the note in section 6)
uvicorn finagent.api:app --app-dir src --host 127.0.0.1 --port 8000
#    or: PYTHONPATH=src python -m finagent.api   (same host and port; FINAGENT_HOST overrides the host)
#    Loopback only, since /api/* can place paper orders. To serve /answer to other machines use
#    --host 0.0.0.0 (or Docker); /api/* still refuses every non-loopback client.
```

`curl -s http://localhost:8000/health` answers once startup has finished. Open <http://localhost:8000/ui> for the test console, which shows every step a question takes. You can also call the API directly:

```bash
# English question -> English answer (the data's Korean names are kept as they are)
curl -G "http://localhost:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=Which three domestic ETFs holding Samsung Electronics have the largest net assets?"

# Korean question -> Korean answer
curl -G "http://localhost:8000/answer" \
  --data-urlencode "question_id=Q-002" \
  --data-urlencode "question=현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요?"
```

In Windows PowerShell, call `curl.exe` and put the whole command on one line. Without a Gemini key the server still starts, but no question can be answered. The LLM-free tools in [section 9](#9-evaluation-and-measured-results) (`scripts/explore.py` and others) work without a key.

## 3. Example questions

The Korean questions mostly come from the answer key in `data/eval/groundtruth.jsonl`. The last three should be declined, because the data cannot confirm them.

| English | Korean | What it exercises |
|---|---|---|
| Which three domestic ETFs holding Samsung Electronics (삼성전자) have the largest net assets? | 삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘 | Holdings: graph + SQL join |
| How many currently purchasable KRW bonds are rated AA- or higher? | 현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요? | SQL count over distinct bonds |
| Among US-listed equity ETFs with at least $10 billion in AUM, which three have the lowest expense ratios? | 운용규모(AUM)가 100억 달러 이상인 미국 상장 주식형 ETF 중 총보수가 가장 낮은 3개는? | SQL filter and sort |
| What is the expense ratio of the VanEck China Semiconductor ETF? | VanEck China Semiconductor ETF의 총보수는 몇 퍼센트인가요? | Lookup by name |
| How many domestic ETFs hold EcoPro BM (에코프로비엠)? | 에코프로비엠을 편입한 국내 ETF는 몇 개인가요? | Holdings count |
| How many share classes of the National Growth Fund (국민성장펀드) are on sale? | 국민성장펀드는 몇 개의 클래스로 판매되고 있나요? | Fund families vs. share classes |
| Across domestic ETFs and public funds, which three products have the highest one-year return? | 국내 ETF와 공모펀드를 합쳐서 1년 수익률이 가장 높은 상품 3개를 알려줘 | Cross-product query (`UNION ALL`) |
| Find bonds with a credit rating of AAAA. | 신용등급 AAAA인 채권을 찾아줘 | Declined: not a valid rating (ontology check) |
| What is next year's expected return of the TIGER 반도체 ETF? | TIGER 반도체 ETF의 내년 예상 수익률을 알려줘 | Declined: a forecast |
| Which public funds hold Samsung Electronics? | 삼성전자를 보유한 공모펀드를 알려줘 | Declined: the data has no fund holdings |

## 4. Architecture

```mermaid
flowchart LR
    Q[GET /answer<br/>question_id, question] --> P[Pre-analysis<br/>product names · entity resolution · ontology checks]
    P --> RT{Round 0<br/>retrieval plan}
    RT -->|asks about a relation| G
    RT -->|asks for a description| D
    RT -->|names it| V
    RT --> A
    A[LLM agent, Gemini by default<br/>tool-calling loop · 120 s budget · up to 8 rounds]
    A -->|sql_query / table_info| R[(RDB · SQLite<br/>5 tables + 5 views · 215 columns)]
    A -->|graph_neighbors / graph_query| G[(Knowledge graph · rdflib<br/>1,082,874 triples · 73 predicates)]
    A -->|search_products / resolve_entity| V[(Lexical index · BM25<br/>75,089 documents)]
    A -->|search_products mode=vector| E[(Vector index · TF-IDF+LSA<br/>75,089 documents · 384 dims)]
    A -->|search_documents| D[(External documents<br/>Markdown chunks)]
    O[ontology/*.ttl] -.schema cards · allowed-value checks.-> P
    O -.TBox merge and inference (superclasses, inverses, domain/range).-> G
    O -.-> A
    A --> F[final_answer<br/>answered / cannot_confirm / clarify]
    F --> J[JSON<br/>retrieved_context · think_trace · answer]
```

### Components

| Component | Role | Code |
|---|---|---|
| ETL | Excel masters + constituent CSV → SQLite (`data/store/products.db`). It converts dates to ISO, adds purchasable flags and fund-family keys, and normalizes managers (derived from institution codes). **5 views expose all 215 columns** | `src/finagent/etl/` |
| Ontology | Loads `ontology/*.ttl` (TBox) at runtime to build **schema cards** (column ↔ meaning, unit, allowed values) and **validators** (value domains of credit ratings and risk grades) | `src/finagent/ontology/loader.py` |
| Knowledge graph | Builds the ABox from SQLite (including `etf_constituent`) and the lexicons (managers, themes, companies). It then **merges the ontology TBox and materializes inferences** (superclasses, inverse relations, domain/range): **1,082,874 triples** (965,823 asserted + 117,051 inferred) **with 73 predicates**. It derives `affiliateOf` (group affiliates) from the holdings of group ETFs. Holdings come from the SQLite table, not from a separate graph source, so SQL results and graph results cannot drift apart. Supports SPARQL and neighbor lookups | `src/finagent/kb/graph.py` |
| Vector index | Dense-vector (semantic) search over product text. The default backend is TF-IDF (words + Hangul bigrams) → truncated SVD to 384 dims → cosine similarity, which **runs deterministically without an embedding API key**. `VECTOR_BACKEND=openai` switches to an embedding API | `src/finagent/kb/vector.py` |
| Lexical index | BM25 with Hangul syllable bigrams and an exact-product-name boost. The entity resolver does exact matches, similar candidates and company relations | `src/finagent/kb/search.py`, `entity.py`, `lexicon.py` |
| Hybrid retrieval | `search_products` combines lexical (BM25) and semantic (vector) results. `mode=hybrid` is the default, and `lexical` or `vector` can be chosen | `src/finagent/kb/vector.py` |
| Docs | Chunks and searches unstructured documents (PDF → Markdown), labelled with source and as-of date | `src/finagent/kb/docs.py`, `scripts/parse_docs.py` |
| Retrieval routing | **The round-0 retrieval plan.** From the pre-analysis, it picks *only the stores that question needs* among graph, vector and documents. It runs them up front and injects the rows into the first prompt as facts, so not every store is hit every time. On the 54-question answer key (competition build), graph fired 6 times, vector 7 times and documents 0 times, and **41 questions fired nothing** (pure aggregation or sorting). The median planning time was 0.8 ms, and 42 of 59 questions added 0 ms of run time. `think_trace` records which store was chosen and why | `src/finagent/agent/routing.py` |
| Agent | Tool-calling LLM loop (Gemini by default; Anthropic and OpenAI-compatible as alternates) with a time budget, evidence and trace collection, and the answer policy. **An answer submitted without a single lookup is rejected and sent back**, because without tools the model makes things up from pretraining. Measured on the competition build: of 5 runs of the same question, every run that called tools was right and every run that didn't was wrong | `src/finagent/agent/` |
| Identifier passing | Resolved identifiers travel all the way to SQL. For **32 companies**, the lexicon's Korean name differs from the source spelling (Cambricon 캠브리콘 → `한무기`, 네이버 → `NAVER`, TSMC → `TSMC ADR`). Using the question's name in `WHERE company_name LIKE` returns 0 rows for them, so the store's actual `company_code` is injected into the prompt. Graph results also carry a list of lookup IDs (`pd_itm_no IN ('…')`) so the model doesn't write SQL from names alone | `agent/orchestrator.py` (`_load_constituent_keys`, `_constituent_key_note`) · `agent/tools.py` |
| View semantics | Each view comes with its **deciding columns pinned down**: inverse `cu_inverse_short_yn='Y'`, monthly distribution `pd_dvid_cycl='M'`, bond ETFs `wu_inv_ast_type='채권'`, ETF vs. ETN `pd_grp_no='ETF'`, number of funds `COUNT(DISTINCT fund_family)`. It also lists the 8 columns the holdings view has and the join that brings in product attributes. Using a column that view lacks is a missing join, not missing data | `agent/tools.py` (`SEMANTICS`) |
| Answer format | Answers are **plain prose only** (markdown is stripped), and **a refusal stays a refusal**: no list of alternative products follows a "cannot confirm", because answering an unanswerable question costs points | `agent/answer.py` |
| Answer language | English by default, Korean when the question is written in Korean (see section 1) | `src/finagent/i18n.py` |
| API | FastAPI `GET /answer` and `GET /health`, plus the `/ui` test console and `/answer/stream` (SSE, console only). Always 200 with the 5 fields | `src/finagent/api.py`, `web/console.html` |

### Mapping to the required components

| Task requirement | Implementation |
|---|---|
| ① Analyze and clean structured and unstructured data, plus a product-domain ontology ([Parsing] pdf/ppt → Markdown) | `etl/` (4 structured masters + holdings) · `scripts/parse_docs.py` (PDF/PPTX → Markdown with source and as-of date in the front matter) · `ontology/*.ttl`, **6 files** |
| ② Financial-product knowledge base: **RDB + Vector + Graph** ([Extraction], [EntityResolution]) | RDB = SQLite, **5 views, 215 columns** · Vector = `kb/vector.py` (TF-IDF+LSA, 384 dims) · Graph = `kb/graph.py` (**1,082,874 triples** including inferences) · Entity resolution = `kb/entity.py` + `data/enrichment/*.yaml`. The lexical and vector indexes hold **75,089 documents**: 53,375 from the organizers' 4 masters + 21,714 Korean ETF holdings |
| ③ Intent analysis and retrieval engine ([NL2SQL], [Retrieval]) | The pre-analysis (intent, entities, ontology checks) becomes a **retrieval plan**. Round 0 deterministically runs only the stores it needs among graph, vector and documents. With those results in hand as facts, the LLM finishes aggregation and sorting with `sql_query` (NL2SQL). The point is that store selection is not left to the model's whims |
| ④ Answer generator (grounded, no hallucination) | `final_answer` forced into 3 states + the `retrieved_context` evidence block + the cannot-confirm and ask-back policy |

### Retrieval methodology — decided by measurement

How `search_products` combines its results was decided by measurement, not taste. All retrieval numbers here were measured on the competition build. These runs call no LLM.

- At first, BM25 and the vector index were fused as equals with **RRF (Reciprocal Rank Fusion)**. RRF lifts documents both retrievers agree on, but in this corpus it let irrelevant funds that each retriever ranked mid-list push out the top lexical match. For example, the Korean query "배터리 양극재 관련 상품" (battery cathode products) got an unrelated fund of funds at rank 1.
- A wrong top candidate contaminates the answer. So in the final structure **lexical results take the head of the list, and roughly the bottom 30% of slots go to documents that only semantic search found**. For queries where lexical search is strong, hybrid structurally cannot do worse than lexical, and semantic search acts as a recall supplement.
- The vector quality itself was also tuned by measurement. Korean product names are full of boilerplate such as `증권투자신탁`, `사모`, `파생형` and `상장지수` (securities investment trust, private, derivative, exchange-traded), and those tokens dominated the LSA space. Cutting high-document-frequency tokens with `max_df=0.005` raised the result from 1 of 7 probes to 6 of 7 (for example, `space exploration satellite` → Procure Space ETF, and `China semiconductor ETF` now also recalls Korean semiconductor ETFs).
- Cross-lingual matching (Korean query ↔ English-named foreign ETF) is handled by the **theme lexicon** (Korean/English synonyms) and graph tagging, not by the vectors.

**We also measured the size of that gain.** We generated **1,313 queries** automatically from the data, covering 4 product families and 7 types (official name, code/ticker, short name, holding name, theme word, typo, synonym), and measured each of the three modes:

| Type | Lexical recall@10 | Vector recall@10 | Hybrid recall@10 |
|---|---|---|---|
| Official name | 100.0% | 60.0% | 100.0% |
| Code / ticker | 49.4% | 1.9% | 49.4% |
| Typo | 93.1% | 45.0% | 93.1% |
| Theme word | 87.0% | 55.8% | **90.5%** |

Hybrid recall@1 and @5 are **identical to lexical alone for every type**. This shows that the "lexical takes the head" design holds. It also means that **the vector's net contribution is limited to +3.5 pp of recall@10 on theme queries**: 22 rescues and 6 losses out of 1,313, all in the tail slots at ranks 8-10.

So this document does not claim that "semantic search absorbs typos and synonyms". **Typo tolerance comes from BM25's Hangul bigrams** (lexical 93.1% vs. vector 45.0%). TF-IDF+LSA is built on lexical co-occurrence rather than meaning, so this is about as far as the method goes. The vector index does win on its own in real cases (`space exploration satellite` → Procure Space ETF, which is absent from the lexical top 5), but that range is narrow. To reproduce: `python scripts/audit_vector.py --sections BC` (no LLM calls, fixed seed).

### Data model (five views)

```mermaid
erDiagram
    v_etf_kr ||--o{ v_etf_constituent : "pd_itm_no = etf_itm_no"

    v_etf_kr {
        text pd_itm_no PK "item number (1,780 rows)"
        text pd_ticker "ticker"
        text pd_nm "product name"
        text product_type "domestic ETF 1,235 / domestic ETN 545"
        text manager_name "manager"
        real pd_net_tamt "net assets"
        int risk_grade "risk grade 1-6"
    }
    v_etf_constituent {
        text etf_itm_no FK "v_etf_kr.pd_itm_no (22,599 rows / 1,162 ETFs)"
        text etf_ticker "ticker"
        text company_code "source stock code - 79% filled"
        text company_name "holding name"
        real weight_pct "weight %"
        int is_cash "cash-like flag"
        text as_of "2026-09-05"
    }
    v_etf_gl {
        text pd_itm_no PK "item number (6,037 rows)"
        text ticker "ticker"
        text product_type "foreign ETF 5,972 / foreign ETN 65"
        text manager_name "manager"
        real expense_ratio_pct "expense ratio %"
        real du_last_aum "AUM"
    }
    v_fund_pub {
        text itm_no PK "share-class item number (23,676 rows)"
        text fund_family "fund family key - 17,795 funds"
        text manager_name "manager - 91.2% filled"
        text manager_source "fill source name/code/NULL"
        text or_co_xtn_itt_cd "manager institution code - 100% supplied"
        text product_type "public 14,716 / private 8,960"
        real fd_yr1_ern_r "1-year return"
    }
    v_bond_kr {
        text pd_no "item number - not a sole PK (21,882 rows / 20,497 bonds)"
        text info_seq "OTC lot sequence"
        text pd_nm "product name"
        text pd_exg_mkt "exchange 17,746 / OTC 4,136"
        text crd_grd "credit rating AAA to D"
        int is_sellable "purchasable 21,866 rows"
        int has_buy_yield "buy yield recorded 634 rows"
    }
```

How to read it:

- **There is exactly one real foreign key**: `v_etf_constituent.etf_itm_no` → `v_etf_kr.pd_itm_no`. The other four views are different product families with no common key, so cross-family questions are solved with `UNION ALL`, not joins.
- One row of `v_fund_pub` is **one fund share class**. "How many funds" must be counted with `COUNT(DISTINCT fund_family)` (23,676 rows = 17,795 funds).
- `v_bond_kr` can have several rows per `pd_no`: exchange vs. OTC, and OTC rows per lot (`info_seq`). To count bonds, use `COUNT(DISTINCT pd_no)`.
- **Foreign ETFs and funds have no holdings data.** Holdings exist only for Korean ETFs.
- The diagram shows only the columns used for joins and identification. The 5 views actually have **215 columns**, and all of them are exposed through schema cards.

### Missing values: finding and filling them

Missing values in the supplied data are **intentional** (an organizer notice), so they are never computed as 0. Missing values in *columns we derived ourselves*, however, are filled when the source has better evidence.

| Target | How it was found | How it was filled | Result |
|---|---|---|---|
| Fund `manager_name` | A per-column NULL-rate check found **25.4% missing**. The cause was guessing the manager from the fund-name prefix | Derived from `or_co_xtn_itt_cd` (manager institution code), which the organizers supply **100% filled**. The majority spelling per code is the representative name | 17,660 rows (74.6%) → **21,592 rows (91.2%)**. The fill source is recorded in `manager_source` (name 17,660 / code 3,932 / NULL 2,084) |
| Holding stock codes | The source (Naver Finance) provides `itemCode`, but the collector was discarding it | Kept by the collector | **79%** of rows have a stock code. Companies are identified only by this code, never by name similarity |
| Korean ETF expense ratio · fund returns | The source itself is partial (expense ratio 18%, fund returns about 30%) | **Not filled.** When a value is missing, the answer says "no data" | — |

We did not build a manager-rename table. Without fund inception dates there is no way to decide which of two spellings is the current name, so the majority spelling attached to the institution code is used as is (it may be an old brand name).

### Ontology inference (technical session 01-4)

At startup the TBox (`ontology/*.ttl`) is merged into the same graph as the ABox, and three kinds of rule are materialized: **superclass inheritance** (`x a fp:DomesticETF` → `x a fp:ETF`, `fp:Product`), **inverse relations** (`fp:managedBy` ↔ `fp:manages` and others, per `owl:inverseOf`) and **domain/range type inference**. Queries such as `?p a fp:Product` or `?am fp:manages ?p` therefore work without extra loading. The graph currently has **1,082,874 triples and 73 predicates**, of which **965,823 are asserted (ABox) and 117,051 inferred**. `/health` reports them as `kg_asserted` / `kg_inferred`.

> There used to be **5,082 predicates**. Holding weights were written as a separate predicate per company (`weight_<company name>`). Once holdings moved into a SQLite table, the weight became a column and the predicates dropped to 73.

### Organizer notices (2026-08-24)

| Notice | How the competition build handled it |
|---|---|
| The answer-generation pipeline **must use NCP HyperCLOVA X** | Default `LLM_PROVIDER=clova`, with function calling through CLOVA Studio's OpenAI-compatible endpoint. *The English edition replaces this with Gemini (section 14 keeps the history)* |
| `BUYABLE_QUANTITY` values are invalid. Everything except delisted or ended listings is purchasable | `is_sellable` redefined: for bonds, **21,866 not yet redeemed** (previously 634). Bonds with a recorded buy yield are split out as `has_buy_yield` |
| 0 and missing values are intentional. Exclude them, or answer that there is no value | Stated in prompt rule 6. NULLs are never computed or sorted as 0 |
| 35 questions in total (10 hard / 10 medium / 10 easy + 5 unanswerable) | The evaluation set was expanded to match: an answer key of **54 questions** (easy 10 · medium 16 · hard 9 · unanswerable 19) |
| Cross-family questions (comparing two or more product families) will appear | The prompt gained a `UNION ALL` pattern and guidance for "foreign ETFs have no 1-year return" |
| No code-value tables supplied, and no questions about them | No reliance on interpreting raw code values |
| Over 300 s counts as no answer. Within that, late answers are still scored (with a penalty) | Internal budget raised from 50 s to **120 s**, rounds from 6 to 8 |

### Answer policy (grading rules)

- Numbers and product names come only from tool results. The evidence (table name, as-of date, product code, unit) is shown in `retrieved_context`.
- Cannot confirm (a product, company or value that doesn't exist, or anything outside the data's scope) → the answer says plainly that the provided data cannot confirm it. Ambiguous → ask back.
- Violations of ontology allowed values (e.g. credit rating AAAA, risk grade 7) are caught in pre-analysis and turned into cannot-confirm.
- No return forecasts and no definitive recommendations.

## 5. Configuration (.env)

`cp .env.example .env`, then set at least `GEMINI_API_KEY`. The other values have working defaults.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` / `anthropic` / `openai` (OpenAI or any OpenAI-compatible server) / `stub` (tests only) |
| `LLM_MODEL` | `gemini-3.8-flash` | Model ID for the chosen provider |
| `GEMINI_API_KEY` | | Create one at <https://aistudio.google.com/apikey>. `GOOGLE_API_KEY` also works |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini's OpenAI-compatible endpoint. Change it only for a proxy |
| `LLM_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max`. Sent to Gemini as `reasoning_effort` (`xhigh` and `max` map to `high`), and to Anthropic as output effort |
| `LLM_MAX_TOKENS` | `8192` | Output-token ceiling per call, **including Gemini's thinking tokens** |
| `LLM_TIMEOUT_SEC` | `40` | Timeout per LLM call |
| `RATE_LIMIT_RETRIES` | `4` | Retries after a 429 (rate limit) |
| `REQUEST_MIN_INTERVAL_SEC` | `0` | Minimum seconds between calls. With 0, the client waits only when a provider reports a nearly exhausted quota in its response headers |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_FALLBACKS` | —, `true` | Alternate provider: Anthropic |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — | Alternate provider: OpenAI or an OpenAI-compatible server (vLLM, …) |
| `TIME_BUDGET_SEC` | `120` | Internal time budget per question. Kept from the competition, where over 300 s counted as no answer |
| `MAX_TOOL_ROUNDS` | `8` | Maximum tool-call rounds |
| `SQL_ROW_LIMIT`, `SQL_TIMEOUT_SEC` | `200`, `5` | Row cap and timeout of the read-only `sql_query` tool |
| `VECTOR_BACKEND` | `lsa` | `lsa` (offline TF-IDF+SVD, no key needed) / `openai` (embedding API) / `off` |
| `VECTOR_DIMS`, `VECTOR_MAX_DF` | `384`, `0.005` | Vector dimensions and the document-frequency cut-off for high-frequency tokens |
| `VECTOR_EMBED_MODEL` | `text-embedding-3-small` | Embedding model for `VECTOR_BACKEND=openai` (uses `OPENAI_API_KEY` / `OPENAI_BASE_URL`) |

### Gemini integration details

- **Endpoint and dependency.** Calls go to Gemini's OpenAI-compatible endpoint (`GEMINI_BASE_URL`) through the `openai` package already in `requirements.txt`, so there is no Gemini-specific SDK (`GeminiLLM` in [`src/finagent/agent/llm.py`](src/finagent/agent/llm.py)).
- **Key source.** Environment variables take precedence over `.env`, so a `GEMINI_API_KEY` exported in your shell overrides the one in the file. `check_llm.py` prints which source it used. Without a key the server still starts and logs an error, and every question fails with an LLM error until a key is added.
- **Reasoning effort.** `LLM_EFFORT` is sent as `reasoning_effort` (`low` / `medium` / `high`; `xhigh` and `max` map to `high`). If a model rejects the parameter, the client drops it once and carries on without it.
- **Thought signatures.** Gemini 3 models attach an encrypted signature to each function call (`extra_content.google.thought_signature`) and reject the next request if the signature is missing from the assistant message sent back. The adapter echoes every tool call with its `extra_content` intact, so multi-turn tool calling works.
- **Output ceiling.** Gemini's thinking tokens count against `LLM_MAX_TOKENS`, so the default of 8192 leaves room for thinking plus the answer. The visible answer itself is short: the 54 answers recorded in the competition have a median of 108 characters and a maximum of 610. The prompt's answer rules and the per-question time budget guard against run-on answers. Lower the ceiling if you see long, slow answers.
- **Retries.** A 429 is retried up to `RATE_LIMIT_RETRIES` times with exponential backoff (2 s doubling, capped at 30 s; a `Retry-After` header takes precedence). A 5xx such as 503 "model is overloaded" is retried up to 2 times, and timeouts and connection errors up to 2 times after 2 s. Free-tier keys allow only a few requests per minute. For back-to-back runs, raise `REQUEST_MIN_INTERVAL_SEC` or use `eval_truth.py --delay`. Gemini shows quotas in Google AI Studio, not in response headers.
- **Connection check.** `python scripts/check_llm.py` needs no server and no index. It prints the provider, model, endpoint, key source, effort and `max_tokens`. It then makes one tool-calling request and reports `OK` (function calling works), `WARN` (answered in text without calling the tool) or `FAIL`, with a hint for rejected keys, 403, unknown model names, 429 and 503. `--model <name>` tries another model, and `--repeat N` makes N back-to-back calls to probe rate limits. The exit code is 0 when healthy, 1 on warnings or failures and 2 when the key is missing.

### Switching to another LLM provider

- **Anthropic:** `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` and an Anthropic model ID in `LLM_MODEL`. The adapter uses adaptive thinking with `LLM_EFFORT` as output effort. `ANTHROPIC_FALLBACKS=true` requests server-side fallbacks and is dropped automatically where unsupported.
- **OpenAI or an OpenAI-compatible server:** `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `LLM_MODEL=...`, and `OPENAI_BASE_URL` for a non-OpenAI server.
- Set `LLM_MODEL` whenever you switch, because the default is a Gemini model ID. `check_llm.py` works with every provider, although its hints are written for Gemini.

## 6. Docker

```bash
cp .env.example .env               # set GEMINI_API_KEY
docker compose up -d --build       # the image build also builds the store and warms the vector cache
curl -s http://localhost/health
```

- `docker-compose.yml` maps container port 8000 to host port **80**, so the API is at `http://localhost/answer` and the console at `http://localhost/ui`.
- Without Compose: `docker build -t fin-agent-en .` and then `docker run -d --env-file .env -p 8000:8000 fin-agent-en` (API on port 8000).
- `restart: unless-stopped` brings the container back after a crash or a server reboot.
- Startup builds the knowledge graph (about 3 minutes on the competition server), so the Dockerfile's `HEALTHCHECK` has a `start-period` of 420 s. After a restart, wait until `/health` returns 200 before calling. The startup times in this README were measured **before** the holdings table was added and the graph rebuilt, and they have not been re-measured since. The 420 s `start-period` allows for that.
- `.env` is listed in `.dockerignore`, so it is never baked into the image (the API key stays out). Compose reads it at runtime through `env_file: .env`, so **the project directory on the host must contain a `.env` file**.
- Optional HTTPS: set `DOMAIN` in `.env` and run `docker compose --profile https up -d` for Caddy with Let's Encrypt on port 443. This needs a domain that points to the server.
- The competition server ran exactly this Compose setup on Naver Cloud Platform (VPC · Ubuntu 24.04 · 4 vCPU / 16 GB). The organizers specified the standard HTTP port, so there was no certificate or reverse proxy.

## 7. API

`GET /answer?question_id=Q-001&question=...` → `200 OK`

```json
{"question_id": "Q-001", "question": "...", "retrieved_context": "...", "think_trace": "...", "answer": "..."}
```

- The response is `application/json`, and all five fields are strings. There is no authentication header, and the path is fixed at `/answer`.
- Every response is 200 with the same schema, including unanswerable questions and internal errors. Unknown query parameters are ignored, and an empty `question` returns a usage message.
- `answer` is in the question's language (section 1). The labels in `think_trace` and `retrieved_context` are in English.
- Other routes:
  - `GET /health`: status, tables with as-of dates and row counts, index size, `kg_triples` / `kg_asserted` / `kg_inferred`, vector backend, document chunks and `provider/model`.
  - `GET /`: a plain-text notice.
  - `GET /ui`: the local test console.
  - `GET /answer/stream`: the same answer streamed step by step as Server-Sent Events. It is for the console only and is deliberately kept apart from the graded `/answer` contract.
- Full details (request/response schema, examples, error handling) are in [docs/api_spec.md](docs/api_spec.md). That spec is in Korean and was written for the competition build, so it names HyperCLOVA X as the LLM.

## 8. Ontology (`ontology/`)

| File | Contents |
|---|---|
| `common.ttl` | Common superclass `fp:Product`; listed-product split `fp:DomesticListedProduct` / `fp:ForeignListedProduct`; manager, issuer, company, theme, index and fund family; enumerations for risk grade (1-6), asset class, investment region and distribution cycle; common relations (`managedBy`, `hasConstituent`, `affiliateOf` (group affiliate, symmetric), `subsidiaryOf`, `hasTheme`, `tracksIndex` …); SHACL constraints |
| `bond_kr.ttl` | `fp:DomesticBond`: the credit-rating scale (AAA to D, ordered), bond class, interest-rate and coupon-payment enumerations, yield and duration properties |
| `etf_kr.ttl` | `fp:DomesticETF` / `fp:DomesticETN`: strategy (physical, synthetic, active), leverage multiple, period returns and volatility, distribution information |
| `etf_gl.ttl` | `fp:ForeignETF` / `fp:ForeignETN`: expense ratio, AUM, exchange, replication method, inverse/leverage |
| `fund_pub.ttl` | `fp:PublicFund` / `fp:PrivateFund` / `fp:ShareClass`: management attributes, Zeroin type, benchmark, class fees and channels, asset-allocation ratios, manager institution code and its fill source (`manager_source`) |
| `etf_constituent.ttl` | Korean ETF holdings: the holding relation, stock code, weight, cash-like flag and as-of date. The graph's `hasConstituent` is generated from this view |

Every data property carries `fp:sourceTable` / `fp:sourceColumn` / `fp:unit` annotations. The runtime (`ontology/loader.py`) reads them to build the LLM's schema cards and value validators, which makes the ontology an executable layer rather than documentation. Instances (ABox) are generated from SQLite at startup and can be exported as Turtle with `KG.export()`.

## 9. Evaluation and measured results

### Reproducing the evaluation

```bash
# (A) Grading against the answer key: compare with answers computed from the data by SQL/graph (recommended)
python scripts/eval_truth.py --compute                      # print the answer key only (no LLM needed)
python scripts/eval_truth.py --base http://127.0.0.1:8000   # grade the running server's answers

# (B) Status check: response status and latency of 33 questions (no comparison with answers)
python scripts/eval_run.py --base http://127.0.0.1:8000

# Look inside each store directly, without an LLM (no key needed)
python scripts/explore.py health
python scripts/explore.py search "차이나 반도체" --mode hybrid
python scripts/explore.py pre "신용등급 AAAA인 채권 찾아줘"

# Follow one question step by step, including tool arguments and raw tool output (calls the LLM)
python scripts/trace.py "Which three domestic ETFs holding Samsung Electronics have the largest net assets?"

# Audits (no LLM)
python scripts/audit_vector.py --sections BC   # retrieval quality: lexical vs. vector vs. hybrid (about 14 min)
python scripts/audit_graph.py                  # knowledge-graph provenance and consistency
```

Results are written to `data/eval/results.jsonl` / `data/eval/truth_results.jsonl`, with per-question `seconds`, grade, and the full evidence and trace. The `truth_results.jsonl` in the repository is the recorded competition run. `eval_truth.py --base` overwrites it unless you pass `--out`. The question sets are in Korean, so the agent answers them in Korean. With a free-tier Gemini key, add `--delay <seconds>` to `eval_truth.py` if you hit 429s.

> **About the answer key.** The organizers did not publish the evaluation questions or answers; the task description says the sample (reference) question set is not included.
> `data/eval/groundtruth.jsonl` therefore consists of **54 questions whose answers can be computed from the supplied data by SQL or graph queries** (easy 10 · medium 16 · hard 9 · unanswerable 19).
> The key is recomputed from the data at grading time, so if the data is updated, the answers update with it.

### Measured results (competition build, HyperCLOVA X)

**Measured accuracy (2026-09-06 · `LLM_PROVIDER=clova` · `HCX-005`).** These are competition-build numbers. The English edition has not been re-benchmarked on Gemini.

| Item | Result |
|---|---|
| Own answer key, 54 questions | **47/54 (87%)** |
| By difficulty | easy 8/10 · medium 14/16 · hard 6/9 · **unanswerable 19/19** |
| Response time | mean **20.6 s**, longest 87.4 s, **0 answers over 300 s** |
| The 8 sample questions on p.12 of the organizers' task description | **7/8** (4 of 5 difficulty questions right, all 3 unanswerable questions right) |

Per-question verdicts, times, evidence and full traces are kept as they were in `data/eval/truth_results.jsonl`. These numbers use the **new set**, after the answer key grew from 20 to 54 questions and **errors in the key itself** were fixed:

- counting ETNs in a question that asked about ETFs;
- fixing a single correct answer when four products tied at the lowest expense ratio of 0.01%;
- accepting rounded integers, which let 0.65 pass as "1";
- a grader that didn't recognize "…상품은 없습니다" ("there is no such product") as a refusal.

Results graded before those fixes were discarded.

### Tests

```bash
python -m pytest -q -m "not slow"   # unit and contract tests (stub LLM, small store; no key, no network)
python -m pytest -q -m slow         # full ETL check against data/raw (about 2 min)
```

Plain `python -m pytest -q` runs both. At competition submission **194 tests passed**. The English edition adds tests for the Gemini adapter and for language detection.

## 10. External data (rule: organizer data first)

- `data/enrichment/constituents/kr_etf_constituents.csv` holds the top holdings of **1,162 Korean ETFs (22,599 rows)** from Naver Finance, `as_of` 2026-09-05. `scripts/fetch_constituents_naver.py` re-collects them.
  `build_store.py` **loads this CSV into the SQLite table `etf_constituent` (view `v_etf_constituent`)**, and the knowledge graph's `hasConstituent` is generated from that table, so SQL results and graph results cannot drift apart.
  Companies are identified only by the **stock code** the source provides (`company_code`, 79% present), never by name similarity. Name matching once attributed 677 rows to the wrong company, because `"lg"` matched as a substring of `"shinhanfinancialgroup"`.
- `data/enrichment/*.yaml` are lexicons for 77 managers, 59 themes (Korean/English) and 97 companies (aliases, tickers, parent company). They are the source of entity resolution and graph relations, and English company and theme names are resolved through them.
  Of these, `subsidiaryOf` (parent/subsidiary) has **only 4 edges**. The organizers' 4 masters have no ownership columns at all, and we kept to the rule of computing from supplied data rather than extending it. "Multi-level subsidiary traversal" is not presented as a general feature.
- `data/external/docs/*.md` hold three Korean reference documents: an overview of the National Growth Fund (국민성장펀드; sourced from a Financial Services Commission press release), ETF terms and indicators, and general ETF risk factors. `scripts/parse_docs.py` converts more PDF/PPTX files to Markdown (`.pptx` needs the optional `python-pptx` package).

## 11. Limitations

- **The English edition has not been re-benchmarked on Gemini.** Its accuracy, refusal behavior and latency are unmeasured. Every number in this README comes from the competition build running HyperCLOVA X.
- **The data is Korean.** Product, manager and company names and the reference documents are in Korean, so English answers quote those names in Korean, as they appear in the data.
- The product masters are snapshots from 2026-08-21/22, and prices and returns are as of those dates.
- Korean ETF expense ratios are present for only 18% of products, fund fee columns have undisclosed units (relative comparison only), and fund returns are present for about 30%. Missing values are answered as "no data".
- Foreign ETFs and funds have no holdings data. Holdings exist only for Korean ETFs (collected for 1,162 of 1,235).
- The holdings' as-of date (`as_of` 2026-09-05) differs from the product masters' (2026-08-21/22). Answers that mix the two sources state the difference.
- Fund managers are 91.2% filled, and **2,084 rows are still NULL** (institution codes that don't converge on one spelling). There is no manager-rename table, because without fund inception dates it is impossible to tell which spelling is the current name.
- **Relations between companies are "affiliates", not "subsidiaries".** The holdings of group ETFs (`PLUS 한화그룹주`, `TIGER 삼성그룹`, …) are the member companies of that group, so `affiliateOf` was derived from them: 65 companies in 8 groups, 608 edges, 0 false positives. That establishes membership of the **same business group**, not **equity ownership**. Hanwha Systems and Hanwha Ocean are subsidiaries of Hanwha Aerospace, while Hanwha Life is a subsidiary of Hanwha Corp., and the data has no ownership stakes to tell those layers apart. Answers therefore say "affiliate". The hand-written `subsidiaryOf` stays at 4 edges (EcoPro group 3 · SK Innovation 1).
- **Affiliates drawn from group ETFs cover only some listed companies.** Unlisted affiliates are invisible, and the Hyundai Motor Group yields only 2 companies. `TIGER 현대차그룹플러스` mixes in HD Hyundai (a separate business group), so the token was kept narrow (`현대차`). Widening it raises recall but lets false positives in. SK Group isn't covered, because its ETF has no holdings rows.
- **The vector index's net contribution is +3.5 pp of recall@10 on theme queries** (1,313 queries measured; see section 4). The top ranks are decided by lexical search, and the vector index fills only the lower slots. It exists and works as a required component, but we don't hide that in this corpus the range where semantic search beats lexical search is narrow.
- **The external documents are only 3 files with 39 chunks**: ETF terms and indicators, general ETF risk factors and the National Growth Fund overview. They don't cover descriptive questions in general. Questions that need background knowledge not in the masters are answered as cannot-confirm.
- The vector search uses TF-IDF+LSA (384 dims) by default so that evaluation reproduces without an external embedding API key. A commercial embedding model might improve semantic search, and `VECTOR_BACKEND=openai` with `VECTOR_EMBED_MODEL` switches to one without code changes. **That switch has not been measured.**
- **Not measured**: Gemini accuracy and latency (see above), container startup time (needs re-measuring after the holdings table and graph rebuild), concurrent request throughput (the organizers announced sequential calls, so no load test was run) and the effect of commercial embeddings. This document gives no numbers for these.
- **The supplied data differs from what the task PDF describes.** The PDF mentions about 145,000 records in total across the 4 products with a base date of 2026-07-11. The file actually distributed (`ai-festival2026_..._DtataSet260824.zip`) has 53,375 records in total with a base date of 2026-08-21 (foreign ETFs 2026-08-22). This implementation follows **the file actually supplied**. If the evaluation data is updated, re-running `python scripts/build_store.py` picks it up (assuming the same schema).

## 12. Repository layout

```
fin-agent-en/
├─ src/finagent/           # implementation: etl / ontology / kb / agent / api.py, i18n.py (answer language)
├─ ontology/               # 6 TTL files: common, bond_kr, etf_kr, etf_gl, fund_pub, etf_constituent
├─ data/raw/               # the organizers' 8 Excel files (data + schema of the 4 product masters)
├─ data/enrichment/        # managers.yaml, themes.yaml, companies.yaml, constituents/kr_etf_constituents.csv
├─ data/external/docs/     # external reference documents (Korean Markdown; front matter with source and as-of date)
├─ data/eval/              # questions.jsonl (33 status checks), groundtruth.jsonl (54-question answer key),
│                          # truth_results.jsonl (recorded competition run)
├─ data/tuning/            # CLOVA Studio tuning datasets from the competition (not used at runtime)
├─ data/store/             # build output: products.db, vector_cache.npz (git-ignored)
├─ scripts/                # build_store, warmup, check_llm, explore, trace, eval_truth, eval_run,
│                          # audit_vector, audit_graph, parse_docs, fetch_constituents_naver, build_tuning_data
├─ tests/                  # pytest
├─ web/console.html        # local test console, served at /ui
├─ docs/                   # anatomy.html (English), anatomy.ko.html (Korean), api_spec.md, proposal drafts, superpowers/
├─ Wolves_of_Walnut_Street_Proposal.html / .pdf   # technical proposal, English translation
├─ Wolves_of_Walnut_Street_제안서.html / .pdf       # technical proposal, Korean original
└─ requirements.txt · pyproject.toml · Dockerfile · docker-compose.yml · Caddyfile · .env.example
```

## 13. Documents

| Document | Contents |
|---|---|
| [`Wolves_of_Walnut_Street_Proposal.html`](Wolves_of_Walnut_Street_Proposal.html) / [`.pdf`](Wolves_of_Walnut_Street_Proposal.pdf) | Technical proposal, English translation |
| [`Wolves_of_Walnut_Street_제안서.html`](Wolves_of_Walnut_Street_제안서.html) / [`.pdf`](Wolves_of_Walnut_Street_제안서.pdf) | Technical proposal, Korean original as submitted |
| [`docs/anatomy.html`](docs/anatomy.html) | Walkthrough of the whole system, in English |
| [`docs/anatomy.ko.html`](docs/anatomy.ko.html) | The same walkthrough, Korean original |
| [`docs/api_spec.md`](docs/api_spec.md) | API specification (Korean, written for the competition build) |
| [`docs/proposal_outline.md`](docs/proposal_outline.md), [`docs/proposal_skeleton.md`](docs/proposal_skeleton.md) | Proposal drafts (Korean) |
| [`docs/superpowers/`](docs/superpowers/) | Design spec and implementation plan from development |

## 14. Competition build notes (HyperCLOVA X)

- **Evaluation endpoint.** During the organizers' evaluation (2026-09-07 to 09-20), the competition build served `http://223.130.161.217/answer` on Naver Cloud Platform (VPC · Ubuntu 24.04 · Docker Compose · standard port 80, no authentication). That endpoint ran HyperCLOVA X and answered in Korean. It is not this edition.
- **LLM.** The organizer notice of 2026-08-24 made NCP HyperCLOVA X mandatory for the answer-generation pipeline. The competition build therefore defaulted to `LLM_PROVIDER=clova` with `HCX-005`, called through CLOVA Studio's OpenAI-compatible endpoint (`https://clovastudio.stream.ntruss.com/v1/openai`).
- **Why the competition build capped `LLM_MAX_TOKENS` at 1536.** With a 4096 ceiling, one 92.9 s call wrote a methodology essay instead of the answer, although the data had been retrieved two rounds earlier. 1536 covered the realistic worst case (a 10-row cross-product table with evidence, about 813 tokens) while ruling out essays thousands of tokens long. CLOVA function calling needs at least 1024. HyperCLOVA X has no thinking tokens, while Gemini's count against the ceiling, which is why this edition's default is 8192.
- **Rate limits.** CLOVA meters requests and tokens per minute separately and returns 429 when either runs out. The client's handling of `x-ratelimit-*` headers (pausing before the limit) comes from that period and still applies to providers that send those headers.
- **Tuning data.** CLOVA Studio cannot combine tuning with function calling, so `scripts/build_tuning_data.py` generated text-to-text datasets (`nl2sql`, `answer`, `intent`) in `data/tuning/`, keeping only pairs whose SQL actually ran. The agent does not use them at runtime.

---

## 한국어

[English (맨 위로)](#financial-product-agent) | **한국어**

**금융상품 Agent — Financial Product Analyst API 영문판.** 제10회 2026 미래에셋증권 AI Festival · 금융상품 Agent 과제 제출물입니다. 팀명 **Wolves of Walnut Street**.

`GET /answer`는 금융상품에 관한 질문을 받아 데이터에 근거한 답을 돌려주는 API입니다. 주최측이 제공한 정형 상품 마스터 4종(국내채권·국내 상장 ETF/ETN·미국 상장 ETF/ETN·국내 펀드)과 국내 ETF 편입종목을 LLM Agent가 스스로 탐색·연산하고, 온톨로지로 검증한 근거를 붙여 답합니다.

> **이 문서의 모든 측정 결과는 대회 제출본의 것입니다.** 대회 제출본은 주최측 규정에 따라 NCP HyperCLOVA X(`HCX-005`)로 동작했습니다. 영문판은 Google Gemini로 동작하며, Gemini 기준으로 다시 측정하지 않았습니다.

## 1. 영문판에서 바뀐 점

| 항목 | 대회 제출본 | 영문판 |
|---|---|---|
| LLM | NCP HyperCLOVA X `HCX-005`, CLOVA Studio OpenAI 호환 엔드포인트 (주최측 규정) | Google Gemini(기본 `gemini-3.8-flash`), Gemini OpenAI 호환 엔드포인트. Anthropic·OpenAI(및 OpenAI 호환 서버)는 대체 제공자로 유지 |
| 답변 언어 | 한국어 | 질문 언어를 따름: 기본은 영어, 질문이 한국어로 쓰였으면 한국어 |
| `think_trace` / `retrieved_context` 표기 | 한국어 | 영어 |
| 테스트 콘솔(`/ui`) | 한국어 | 기본 영어, EN / 한국어 전환 버튼 |
| 문서 | 한국어 | 영어 먼저, 한국어 다음: 이 README, 기술제안서, `docs/anatomy.html` |
| 데이터 | 한국어 | 변경 없음 — 여전히 한국어 |

- **OpenAI 호환 엔드포인트를 통한 Gemini.** 프로젝트가 이미 쓰던 `openai` 패키지로 Gemini를 호출하므로 추가 의존성이 없습니다. `LLM_EFFORT`는 Gemini의 `reasoning_effort`로 전달됩니다. 여러 턴에 걸친 도구 호출이 동작하도록 thought signature를 그대로 돌려보내며, 429·5xx 응답은 재시도합니다. `python scripts/check_llm.py` 한 번의 호출로 키·모델·function calling·지연시간을 점검합니다. 자세한 내용은 [5절](#5-환경-설정-env)에 있습니다.
- **답변은 질문 언어를 따릅니다.** 언어는 [`src/finagent/i18n.py`](src/finagent/i18n.py)가 정합니다. 한글로만 된 단어는 음절 수만큼 한국어에, 한글이 없는 단어는 소문자 수만큼 영어에 투표합니다. `미국S&P500` 같은 혼합 표기 상품명과 `TIGER`·`ETF` 같은 대문자 이름은 투표하지 않으며, 동률이면 영어입니다. 따라서 영어 질문 속 한글 상품명이 답변을 한국어로 뒤집지 않습니다. "Which ETFs hold 삼성전자?"에는 영어로, "TIGER 미국S&P500 총보수는?"에는 한국어로 답합니다.
- **데이터는 한국어 그대로입니다.** 국내채권·국내 ETF/ETN·미국 상장 ETF·국내 펀드·ETF 편입종목·참고 문서는 바뀌지 않았습니다. 따라서 영어 질문에도 한국 상품·기업 이름이 한글 또는 영문으로 나올 수 있습니다. Agent는 영문 기업명을 데이터의 표기로 바꿔 조회합니다(예: "Samsung Electronics" → 삼성전자). 기업 사전([`data/enrichment/companies.yaml`](data/enrichment/companies.yaml))에 기업별 영문명과 별칭이 있습니다.
- **테스트 콘솔** `/ui`([`web/console.html`](web/console.html))는 기본이 영어이고, EN / 한국어 버튼으로 화면 언어를 바꿉니다.
- **문서.** 기술제안서의 영어 번역본은 [`Wolves_of_Walnut_Street_Proposal.html`](Wolves_of_Walnut_Street_Proposal.html) / [`.pdf`](Wolves_of_Walnut_Street_Proposal.pdf)에 있고, 한국어 원본도 그대로 둡니다. [`docs/anatomy.html`](docs/anatomy.html)은 영어판이며, [`docs/anatomy.ko.html`](docs/anatomy.ko.html)이 한국어 원본입니다.
- **제거:** `clova` 제공자와 `CLOVA_API_KEY` / `CLOVA_BASE_URL`. CLOVA 관련 기록은 [14절](#14-대회-당시-기록-hyperclova-x)에 남겨 두었습니다.

## 2. 빠른 시작

요구사항:

- Python 3.11+. Docker 이미지는 3.12를 쓰고, 개발은 3.14에서 했습니다.
- 메모리 4GB 이상(지식그래프 구축).
- 인터넷(LLM API).
- <https://aistudio.google.com/apikey>에서 발급한 Gemini API 키.

```bash
git clone <this repo> fin-agent-en && cd fin-agent-en
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # .env 에 GEMINI_API_KEY=... 입력
```

```bash
# 1) data/raw 로 SQLite 스토어 구축: Excel 마스터 + 편입종목 CSV -> data/store/products.db (약 1~2분)
python scripts/build_store.py

# 2) (선택) 벡터 캐시 예열 (data/store/vector_cache.npz) — 이후 기동마다 약 80초 단축
python scripts/warmup.py

# 3) LLM 점검: 키·모델 접근·function calling·지연시간 (호출 1회, 수 초)
python scripts/check_llm.py

# 4) API 실행. 기동 시 검색 색인·벡터·지식그래프+추론을 구축합니다:
#    벡터 캐시가 있으면 약 1분 30초, 없으면 3~3.5분 (6절의 주의 참고)
uvicorn finagent.api:app --app-dir src --host 127.0.0.1 --port 8000
#    또는: PYTHONPATH=src python -m finagent.api   (호스트·포트 동일, FINAGENT_HOST로 호스트 변경)
#    /api/*는 모의 주문을 낼 수 있으므로 루프백에만 바인딩합니다. 다른 기기에 /answer를 제공하려면
#    --host 0.0.0.0(또는 Docker)을 쓰세요. 이때도 /api/*는 루프백이 아닌 클라이언트를 모두 거부합니다.
```

기동이 끝나면 `curl -s http://localhost:8000/health`가 응답합니다. <http://localhost:8000/ui>를 열면 질문이 답까지 가는 모든 단계를 보여주는 테스트 콘솔이 뜹니다. API를 직접 호출할 수도 있습니다:

```bash
# 영어 질문 -> 영어 답변 (데이터의 한글 이름은 그대로 표기)
curl -G "http://localhost:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=Which three domestic ETFs holding Samsung Electronics have the largest net assets?"

# 한국어 질문 -> 한국어 답변
curl -G "http://localhost:8000/answer" \
  --data-urlencode "question_id=Q-002" \
  --data-urlencode "question=현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요?"
```

Windows PowerShell에서는 `curl.exe`를 쓰고 명령을 한 줄로 입력하십시오. Gemini 키가 없어도 서버는 뜨지만 질문에는 답할 수 없습니다. [9절](#9-평가-재현과-측정-결과)의 LLM 없는 도구(`scripts/explore.py` 등)는 키 없이 동작합니다.

## 3. 질문 예시

한국어 질문은 대부분 `data/eval/groundtruth.jsonl`의 정답 키에서 가져왔습니다. 마지막 세 문항은 데이터로 확인할 수 없으므로 거절해야 맞습니다.

| English | 한국어 | 확인하는 기능 |
|---|---|---|
| Which three domestic ETFs holding Samsung Electronics (삼성전자) have the largest net assets? | 삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘 | 편입종목: 그래프 + SQL 조인 |
| How many currently purchasable KRW bonds are rated AA- or higher? | 현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요? | 종목 기준 SQL 집계 |
| Among US-listed equity ETFs with at least $10 billion in AUM, which three have the lowest expense ratios? | 운용규모(AUM)가 100억 달러 이상인 미국 상장 주식형 ETF 중 총보수가 가장 낮은 3개는? | SQL 필터·정렬 |
| What is the expense ratio of the VanEck China Semiconductor ETF? | VanEck China Semiconductor ETF의 총보수는 몇 퍼센트인가요? | 이름으로 조회 |
| How many domestic ETFs hold EcoPro BM (에코프로비엠)? | 에코프로비엠을 편입한 국내 ETF는 몇 개인가요? | 편입종목 집계 |
| How many share classes of the National Growth Fund (국민성장펀드) are on sale? | 국민성장펀드는 몇 개의 클래스로 판매되고 있나요? | 펀드 패밀리 대 클래스 |
| Across domestic ETFs and public funds, which three products have the highest one-year return? | 국내 ETF와 공모펀드를 합쳐서 1년 수익률이 가장 높은 상품 3개를 알려줘 | 교차질의(`UNION ALL`) |
| Find bonds with a credit rating of AAAA. | 신용등급 AAAA인 채권을 찾아줘 | 거절: 존재하지 않는 등급(온톨로지 검증) |
| What is next year's expected return of the TIGER 반도체 ETF? | TIGER 반도체 ETF의 내년 예상 수익률을 알려줘 | 거절: 미래 전망 |
| Which public funds hold Samsung Electronics? | 삼성전자를 보유한 공모펀드를 알려줘 | 거절: 펀드 편입종목 데이터 없음 |

## 4. 아키텍처

```mermaid
flowchart LR
    Q[GET /answer<br/>question_id, question] --> P[사전 분석<br/>상품명 추출·엔티티 해석·온톨로지 검증]
    P --> RT{라운드 0<br/>검색 실행 계획}
    RT -->|관계를 묻는가| G
    RT -->|서술형을 묻는가| D
    RT -->|이름으로 지목했는가| V
    RT --> A
    A[LLM Agent · 기본 Gemini<br/>도구 호출 루프 · 120초 예산 · 최대 8라운드]
    A -->|sql_query / table_info| R[(RDB · SQLite<br/>5 테이블 + 5 뷰 · 215 컬럼)]
    A -->|graph_neighbors / graph_query| G[(Knowledge Graph · rdflib<br/>1,082,874 트리플 · 술어 73종)]
    A -->|search_products / resolve_entity| V[(Lexical Index · BM25<br/>75,089 문서)]
    A -->|search_products mode=vector| E[(Vector Index · TF-IDF+LSA<br/>75,089 문서 · 384차원)]
    A -->|search_documents| D[(외부 문서<br/>Markdown chunks)]
    O[ontology/*.ttl] -.스키마 카드·허용값 검증.-> P
    O -.TBox 병합·추론(상위클래스·역관계·domain/range).-> G
    O -.-> A
    A --> F[final_answer<br/>answered / cannot_confirm / clarify]
    F --> J[JSON<br/>retrieved_context · think_trace · answer]
```

### 구성 요소

| 구성 요소 | 역할 | 코드 |
|---|---|---|
| ETL | Excel 마스터 + 편입종목 CSV → SQLite(`data/store/products.db`). 날짜 ISO 변환, 판매가능 플래그, 펀드 패밀리 키, 운용사 정규화(기관코드 유래), **뷰 5개 · 215 컬럼 전량 노출** | `src/finagent/etl/` |
| Ontology | `ontology/*.ttl`(TBox)을 런타임에 로드해 **스키마 카드**(컬럼 ↔ 의미·단위·허용값)와 **검증기**(신용등급·위험등급 값 도메인)를 생성 | `src/finagent/ontology/loader.py` |
| Knowledge Graph | SQLite(`etf_constituent` 포함) + 사전(운용사·테마·기업)으로 ABox를 구축하고, **온톨로지 TBox를 병합해 추론**(상위 클래스·역관계·domain/range)까지 물질화 — **1,082,874 트리플**(주장 965,823 + 추론 117,051) **· 술어 73종**. 그룹주 ETF 편입종목에서 `affiliateOf`(계열사) 관계도 도출한다. 편입종목은 그래프가 아니라 SQLite 테이블에서 생성되므로 SQL 결과와 그래프 결과가 갈라지지 않는다. SPARQL·이웃 조회 | `src/finagent/kb/graph.py` |
| Vector Index | 상품 텍스트의 밀집 벡터(의미) 검색. 기본 백엔드는 TF-IDF(단어+한글 bigram) → Truncated SVD 384차원 → 코사인 유사도로, **임베딩 API 키 없이 결정적으로 동작**. `VECTOR_BACKEND=openai`로 임베딩 API 전환 가능 | `src/finagent/kb/vector.py` |
| Lexical Index | BM25 + 한글 음절 bigram, 정확 상품명 부스트. 엔티티 해석기(정확 일치/유사 후보/기업 관계) | `src/finagent/kb/search.py`, `entity.py`, `lexicon.py` |
| Hybrid Retrieval | `search_products`가 어휘(BM25) + 의미(벡터)를 결합(`mode=hybrid`가 기본, `lexical`/`vector` 선택 가능) | `src/finagent/kb/vector.py` |
| Docs | 비정형 문서(PDF→Markdown) 청킹·검색. 출처·기준일 표기 | `src/finagent/kb/docs.py`, `scripts/parse_docs.py` |
| Retrieval Routing | **라운드 0의 검색 실행 계획.** 사전 분석 결과로 그래프·벡터·문서 중 *그 질문에 필요한 것만* 골라 미리 실행하고, 결과 행을 첫 프롬프트에 사실로 주입한다. 모든 저장소를 항상 두드리지 않는다 — 54문항 정답 키 기준(대회 제출본) 그래프 6건·벡터 7건·문서 0건이 켜지고 **41건은 아무것도 켜지 않는다**(순수 집계·정렬 질문). 계획 수립 중앙값 0.8ms, 59문항 중 42문항은 실행 시간 추가가 0ms. 어느 저장소를 왜 골랐는지는 `think_trace` 에 남는다 | `src/finagent/agent/routing.py` |
| Agent | 도구 호출 LLM 루프(기본 Gemini, 대체로 Anthropic·OpenAI 호환), 시간 예산, 근거·추적 수집, 답변 정책. **조회를 한 번도 하지 않은 답변은 제출을 거부하고 되돌려보낸다** — 모델이 도구 없이 답하면 사전학습 지식으로 지어내기 때문이다(대회 제출본 실측: 같은 질문 5회 중 도구를 부른 실행은 전부 정답, 안 부른 실행은 전부 오답) | `src/finagent/agent/` |
| 식별자 전달 | 해소한 식별자를 SQL까지 넘긴다. 사전 한글명과 원천 표기가 다른 기업 **32개**(캠브리콘→`한무기`, 네이버→`NAVER`, TSMC→`TSMC ADR`)는 질문의 이름으로 `WHERE company_name LIKE` 를 쓰면 0행이 나오므로, 스토어의 실제 `company_code` 를 프롬프트에 주입한다. 그래프 결과에도 조회용 ID 목록(`pd_itm_no IN ('…')`)을 동봉해 이름만으로 SQL을 짜지 않게 한다 | `agent/orchestrator.py`(`_load_constituent_keys`·`_constituent_key_note`) · `agent/tools.py` |
| 뷰 시맨틱스 | 뷰마다 **판별 컬럼을 못박아** 전달한다 — 인버스 `cu_inverse_short_yn='Y'`, 월배당 `pd_dvid_cycl='M'`, 채권형 ETF `wu_inv_ast_type='채권'`, ETF/ETN 구분 `pd_grp_no='ETF'`, 펀드 개수 `COUNT(DISTINCT fund_family)`, 그리고 편입종목 뷰가 가진 컬럼 8개와 상품 속성을 얻는 조인(그 뷰에 없는 컬럼을 쓰는 것은 데이터 부재가 아니라 조인 누락이다) | `agent/tools.py` (`SEMANTICS`) |
| Answer 형식 | 답변은 **줄글만**(마크다운 제거), **거절은 거절로 끝난다** — 확인 불가 뒤에 대체 상품 목록을 붙이지 않는다(답할 수 없는 질문에 답하면 감점 대상) | `agent/answer.py` |
| 답변 언어 | 기본 영어, 질문이 한국어로 쓰였으면 한국어(1절 참고) | `src/finagent/i18n.py` |
| API | FastAPI `GET /answer`, `GET /health`, 그리고 `/ui` 테스트 콘솔과 `/answer/stream`(SSE, 콘솔 전용). 항상 200 + 5개 필드 | `src/finagent/api.py`, `web/console.html` |

### 과제 필수 구성요소 대응

| 과제 요구 | 구현 |
|---|---|
| ① 정형·비정형 분석/정제 + 상품 도메인 특화 Ontology ([Parsing] pdf·ppt → Markdown) | `etl/`(정형 4종 + 편입종목) · `scripts/parse_docs.py`(PDF/PPTX → Markdown, front-matter에 출처·기준일) · `ontology/*.ttl` **6종** |
| ② 금융상품 KnowledgeBase **RDB + Vector + Graph** ([Extraction]·[EntityResolution]) | RDB=SQLite **5뷰 · 215컬럼** · Vector=`kb/vector.py`(TF-IDF+LSA 384차원) · Graph=`kb/graph.py`(**1,082,874 트리플**, 추론 포함) · Entity Resolution=`kb/entity.py` + `data/enrichment/*.yaml`. 어휘·벡터 색인은 **75,089 문서**(주최측 4종 마스터 53,375 + 국내ETF 편입종목 21,714) |
| ③ Intent Analysis & Retrieval Engine ([NL2SQL]·[Retrieval]) | 사전 분석(의도·엔티티·온톨로지 검증)이 **검색 실행 계획**이 되어 라운드 0에서 그래프/벡터/문서 중 필요한 것만 결정론적으로 실행 → 그 결과를 사실로 쥐고 LLM이 `sql_query`(NL2SQL)로 집계·정렬을 마무리. 저장소 선택을 모델의 변덕에 맡기지 않는 것이 핵심 |
| ④ Answer Generator (근거 기반·환각 방지) | `final_answer` 3상태 강제 + `retrieved_context` 근거 블록 + 확인불가/역질문 정책 |

### 검색 방법론 — 측정으로 결정한 부분

`search_products`의 하이브리드 결합 방식은 취향이 아니라 측정으로 정했습니다. 이 절의 검색 수치는 모두 대회 제출본에서 측정한 것이며, 측정에는 LLM 호출이 없습니다.

- 처음에는 BM25와 벡터를 **RRF(Reciprocal Rank Fusion)** 로 대등하게 결합했습니다. RRF는 두 검색기가 동의하는 문서를 상위로 올리는데, 이 코퍼스에서는 두 검색기가 각각 중간 순위로 올린 무관한 펀드가 최상위 어휘 매칭을 밀어내는 사례가 발생했습니다(예: "배터리 양극재 관련 상품" → 1위가 무관한 재간접 펀드).
- 최상위 후보가 틀리면 답변 생성이 오염되므로, 최종 구조는 **어휘 결과가 리스트의 머리(head)를 차지하고, 하위 약 30% 슬롯을 "의미 검색만 찾아낸" 문서에 할당**하는 방식입니다. 어휘 검색이 강한 질의에서 하이브리드는 어휘 검색보다 나빠질 수 없고(구조적 보장), 의미 검색은 recall 보강 역할을 합니다.
- 벡터 품질 자체도 측정으로 조정했습니다. 한글 상품명에는 `증권투자신탁`·`사모`·`파생형`·`상장지수` 같은 보일러플레이트가 많아 LSA 공간이 그 축에 지배되었고, 문서빈도 상위 토큰을 잘라내는 `max_df=0.005`로 바꾸자 7개 프로브 중 1개만 맞던 결과가 6개로 개선되었습니다(예: `space exploration satellite` → Procure Space ETF, `China semiconductor ETF` → 국내 반도체 ETF까지 recall).
- 크로스링구얼(한글 질의 ↔ 영문 해외ETF)은 벡터가 아니라 **테마 사전**(한/영 동의어)과 그래프 태깅으로 처리합니다.

**그리고 그 이득의 크기까지 측정했습니다.** 데이터에서 자동 생성한 질의 **1,313개**(4개 상품군 · 정식명칭/코드/약칭/편입종목명/테마어/오타/유의어 7유형)로 세 모드를 각각 재면:

| 유형 | 어휘 recall@10 | 벡터 recall@10 | 결합 recall@10 |
|---|---|---|---|
| 정식명칭 | 100.0% | 60.0% | 100.0% |
| 코드·티커 | 49.4% | 1.9% | 49.4% |
| 오타 | 93.1% | 45.0% | 93.1% |
| 테마어 | 87.0% | 55.8% | **90.5%** |

결합의 recall@1·@5는 **전 유형에서 어휘 단독과 완전히 동일**합니다. 위에 적은 "어휘가 head를 차지한다"는 설계가 지켜지고 있다는 뜻이고, 동시에 **벡터의 순기여가 테마형 recall@10 +3.5%p에 국한된다**는 뜻이기도 합니다(1,313건 중 구제 22건, 손실 6건, 전부 8~10위 꼬리 슬롯).

그래서 이 문서는 "의미 검색이 오타·유의어를 흡수한다"고 쓰지 않습니다. **오타 내성은 BM25의 한글 bigram이 제공합니다**(어휘 93.1% 대 벡터 45.0%). TF-IDF+LSA는 의미 임베딩이 아니라 어휘 공기(co-occurrence) 기반이므로 이 정도가 이 방식의 상한입니다. 벡터가 단독으로 이기는 사례는 실재하지만(`space exploration satellite` → Procure Space ETF, 어휘 top-5 부재) 그 구간은 좁습니다. 재현: `python scripts/audit_vector.py --sections BC` (LLM 미호출, 시드 고정).

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

읽는 법:

- **실제 외래키는 하나뿐입니다.** `v_etf_constituent.etf_itm_no` → `v_etf_kr.pd_itm_no`. 나머지 네 뷰는 상품군이 달라 공통 키가 없으므로, 교차질의는 조인이 아니라 `UNION ALL`로 풉니다.
- `v_fund_pub` 1행 = **펀드 클래스 1개**. "펀드 몇 개"를 물으면 `COUNT(DISTINCT fund_family)`로 세야 합니다(23,676행 = 17,795 펀드).
- `v_bond_kr`은 한 `pd_no`에 여러 행이 올 수 있습니다(장내/장외 구분, 장외는 LOT 단위 `info_seq`). 종목 수를 세려면 `COUNT(DISTINCT pd_no)`.
- 해외ETF·공모펀드의 **편입종목 데이터는 없습니다**. 편입종목은 국내 ETF에만 존재합니다.
- 위 표는 조인·식별에 쓰이는 컬럼만 뽑은 것이고, 실제로는 5개 뷰에 **215개 컬럼**이 있으며 전량이 스키마 카드로 노출됩니다.

### 결측치 식별과 보완

제공 데이터의 결측은 **의도된 것**(주최측 공지)이므로 0으로 계산하지 않습니다. 다만 *우리가 만든 파생 컬럼*의 결측은 원천에 더 나은 근거가 있으면 채웁니다.

| 대상 | 어떻게 식별했나 | 어떻게 채웠나 | 결과 |
|---|---|---|---|
| 펀드 `manager_name` | 컬럼별 NULL 비율 점검에서 **25.4% 결측** 발견. 원인은 펀드 이름 접두어로 운용사를 추측한 것 | 주최측이 **100% 채워서 주는** `or_co_xtn_itt_cd`(운용사 기관코드)에서 유도. 코드별 다수 표기를 대표명으로 사용 | 17,660행(74.6%) → **21,592행(91.2%)**. 채움 출처를 `manager_source`(name 17,660 / code 3,932 / NULL 2,084)에 기록 |
| 편입종목 종목코드 | 원천(네이버금융)이 `itemCode`를 주는데 수집기가 버리고 있었음 | 수집기에서 보존 | **79%** 행이 종목코드 보유. 기업 식별은 이름 유사도가 아니라 이 코드로만 |
| 국내ETF 총보수 · 펀드 수익률 | 원천 자체가 부분 수록(총보수 18%, 펀드 수익률 약 30%) | **채우지 않음.** 값이 없으면 "데이터 없음"으로 답함 | — |

운용사 개명은 별도 표를 만들지 않았습니다. 펀드 설정일이 없으면 두 표기 중 어느 쪽이 현재 이름인지 판정할 수 없기 때문에, 기관코드에 딸린 다수 표기를 그대로 씁니다(옛 브랜드명일 수 있음).

### 온톨로지 추론 (기술세션 01-4 대응)

기동 시 TBox(`ontology/*.ttl`)를 ABox와 같은 그래프에 병합하고 세 가지 규칙을 물질화합니다: **상위 클래스 상속**(`x a fp:DomesticETF` → `x a fp:ETF`, `fp:Product`), **역관계**(`owl:inverseOf` 선언에 따라 `fp:managedBy` ↔ `fp:manages` 등), **domain/range 타입 추론**. 그 결과 `?p a fp:Product`·`?am fp:manages ?p` 같은 질의가 별도 적재 없이 동작합니다. 현재 그래프는 **1,082,874 트리플 · 술어 73종**이며, 그중 **주장(ABox) 965,823 · 추론 117,051** 입니다 — `/health`가 `kg_asserted`/`kg_inferred`로 보고합니다.

> 이전에는 술어가 **5,082종**이었습니다. 편입 비중을 `weight_<기업명>`처럼 기업마다 다른 술어로 적었기 때문인데, 편입종목이 SQLite 테이블로 내려오면서 비중은 컬럼이 되었고 술어는 73종으로 정리됐습니다.

### 주최측 공지 반영 (2026-08-24)

| 공지 | 대회 제출본의 반영 |
|---|---|
| 답변 생성 파이프라인은 **NCP HyperCLOVA X 필수** | `LLM_PROVIDER=clova` 기본값, CLOVA Studio OpenAI 호환 엔드포인트로 function calling 사용. *영문판은 이를 Gemini로 교체(당시 기록은 14절)* |
| `BUYABLE_QUANTITY` 값 무효, 상장폐지·리스팅 종료 제외 종목은 모두 구매가능 | `is_sellable` 재정의 — 채권은 **상환 미도래 21,866건**(기존 634건). 매수수익률 수록 종목은 `has_buy_yield`로 분리 |
| 0·결측값은 의도된 것, 값을 빼거나 없다고 응답 | 프롬프트 규칙 6번에 명시. NULL을 0으로 계산·정렬하지 않음 |
| 총 35문항(상10/중10/하10 + 답변불가 5) | 평가 세트를 그 구성에 맞춰 확장. 정답 키 **54문항**(하 10 · 중 16 · 상 9 · 답변불가 19) |
| 교차질의(2개 이상 상품군 비교) 출제 | 프롬프트에 `UNION ALL` 패턴과 "해외ETF는 1년 수익률 미제공" 처리 지침 추가 |
| 코드값 테이블 미제공, 해당 질의 없음 | 원문 코드값 해석에 의존하지 않음 |
| 300초 초과는 미응답, 그 안이면 늦어도 채점(점수 차등) | 내부 예산 50초 → **120초**, 라운드 6 → 8 |

### 답변 정책 (평가 규칙 반영)

- 수치·상품명은 도구 결과에서만. 근거(테이블명·기준일·상품코드·단위)는 `retrieved_context`에 표시.
- 확인 불가(존재하지 않는 상품/기업/값, 데이터 범위 밖) → `제공된 데이터로는 확인할 수 없습니다` 명시(영어 질문에는 같은 뜻의 영어 문장). 모호 → 역질문.
- 온톨로지 허용값 위반(예: 신용등급 AAAA, 위험등급 7)은 사전 분석에서 잡아 확인 불가로 보정.
- 수익률 전망·단정적 추천 금지.

## 5. 환경 설정 (.env)

`cp .env.example .env` 후 최소한 `GEMINI_API_KEY`를 넣습니다. 나머지 값은 기본값으로 동작합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` / `anthropic` / `openai`(OpenAI 또는 OpenAI 호환 서버) / `stub`(테스트 전용) |
| `LLM_MODEL` | `gemini-3.8-flash` | 선택한 제공자의 모델 ID |
| `GEMINI_API_KEY` | | <https://aistudio.google.com/apikey>에서 발급. `GOOGLE_API_KEY`도 인식 |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini OpenAI 호환 엔드포인트. 프록시를 거칠 때만 변경 |
| `LLM_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max`. Gemini에는 `reasoning_effort`로(`xhigh`·`max`는 `high`), Anthropic에는 output effort로 전달 |
| `LLM_MAX_TOKENS` | `8192` | 호출당 출력 토큰 상한. **Gemini의 thinking 토큰 포함** |
| `LLM_TIMEOUT_SEC` | `40` | LLM 호출 1회의 제한시간 |
| `RATE_LIMIT_RETRIES` | `4` | 429(요청 한도) 재시도 횟수 |
| `REQUEST_MIN_INTERVAL_SEC` | `0` | 호출 사이 최소 간격(초). 0이면 제공자가 응답 헤더로 잔여 한도 소진 임박을 알릴 때만 대기 |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_FALLBACKS` | —, `true` | 대체 제공자: Anthropic |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — | 대체 제공자: OpenAI 또는 OpenAI 호환 서버(vLLM 등) |
| `TIME_BUDGET_SEC` | `120` | 문항당 내부 시간 예산. 대회 규정(300초 초과 시 미응답 처리)에서 유지 |
| `MAX_TOOL_ROUNDS` | `8` | 도구 호출 최대 라운드 |
| `SQL_ROW_LIMIT`, `SQL_TIMEOUT_SEC` | `200`, `5` | 읽기 전용 `sql_query` 도구의 행 상한과 제한시간 |
| `VECTOR_BACKEND` | `lsa` | `lsa`(오프라인 TF-IDF+SVD, 키 불필요) / `openai`(임베딩 API) / `off` |
| `VECTOR_DIMS`, `VECTOR_MAX_DF` | `384`, `0.005` | 벡터 차원 수와 고빈도 토큰 제거 임계값 |
| `VECTOR_EMBED_MODEL` | `text-embedding-3-small` | `VECTOR_BACKEND=openai`의 임베딩 모델(`OPENAI_API_KEY` / `OPENAI_BASE_URL` 사용) |

### Gemini 연동 세부

- **엔드포인트와 의존성.** `requirements.txt`에 이미 있는 `openai` 패키지로 Gemini의 OpenAI 호환 엔드포인트(`GEMINI_BASE_URL`)를 호출하므로, Gemini 전용 SDK가 없습니다([`src/finagent/agent/llm.py`](src/finagent/agent/llm.py)의 `GeminiLLM`).
- **키 출처.** 환경변수가 `.env`보다 우선하므로, 셸에 export된 `GEMINI_API_KEY`가 파일의 값을 덮어씁니다. `check_llm.py`가 어느 쪽 키를 썼는지 출력합니다. 키가 없어도 서버는 뜨고 오류를 로그에 남기며, 키를 넣을 때까지 모든 질문이 LLM 오류로 실패합니다.
- **추론 강도.** `LLM_EFFORT`는 `reasoning_effort`(`low` / `medium` / `high`; `xhigh`·`max`는 `high`)로 전달됩니다. 모델이 이 파라미터를 거부하면 클라이언트가 한 번 빼고 계속 진행합니다.
- **Thought signature.** Gemini 3 모델은 function call마다 암호화된 서명(`extra_content.google.thought_signature`)을 붙이고, 다음 요청의 assistant 메시지에 그 서명이 빠져 있으면 요청을 거부합니다. 어댑터가 모든 도구 호출을 `extra_content`째 그대로 돌려보내므로 여러 턴의 도구 호출이 동작합니다.
- **출력 상한.** Gemini의 thinking 토큰이 `LLM_MAX_TOKENS`에 포함되므로, 기본값 8192는 thinking과 답변이 모두 들어갈 여유를 둔 값입니다. 보이는 답변 자체는 짧습니다 — 대회 때 기록된 답변 54개의 길이는 중앙값 108자, 최대 610자입니다. 장황한 답변은 프롬프트의 답변 규칙과 문항당 시간 예산이 막습니다. 길고 느린 답변이 보이면 상한을 낮추십시오.
- **재시도.** 429는 `RATE_LIMIT_RETRIES`회까지 지수 백오프로 재시도합니다(2초에서 두 배씩, 최대 30초; `Retry-After` 헤더가 있으면 우선). 503 "model is overloaded" 같은 5xx는 최대 2회, 타임아웃·연결 오류는 2초 뒤 최대 2회 재시도합니다. 무료 등급 키는 분당 요청 수가 적습니다. 연속 호출에는 `REQUEST_MIN_INTERVAL_SEC`를 올리거나 `eval_truth.py --delay`를 쓰십시오. Gemini는 한도를 응답 헤더가 아니라 Google AI Studio에서 보여줍니다.
- **연결 점검.** `python scripts/check_llm.py`는 서버·색인 없이 동작합니다. 제공자·모델·엔드포인트·키 출처·effort·`max_tokens`를 출력합니다. 그다음 도구 호출 요청 1회를 보내 `OK`(function calling 정상), `WARN`(도구 없이 텍스트로 답함), `FAIL`을 보고하며, 거부된 키·403·모델명 오류·429·503에는 조치 힌트를 붙입니다. `--model <이름>`으로 다른 모델을, `--repeat N`으로 N회 연속 호출해 요청 한도를 시험합니다. 종료 코드는 정상 0, 경고·실패 1, 키 없음 2입니다.

### 다른 LLM 제공자로 전환

- **Anthropic:** `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`, `LLM_MODEL`에 Anthropic 모델 ID. 어댑터는 adaptive thinking을 쓰고 `LLM_EFFORT`를 output effort로 전달합니다. `ANTHROPIC_FALLBACKS=true`는 서버 측 fallback을 요청하며, 지원되지 않는 환경에서는 자동으로 빠집니다.
- **OpenAI 또는 OpenAI 호환 서버:** `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `LLM_MODEL=...`, OpenAI가 아닌 서버라면 `OPENAI_BASE_URL`.
- 기본값이 Gemini 모델 ID이므로, 제공자를 바꿀 때는 `LLM_MODEL`도 함께 지정하십시오. `check_llm.py`는 모든 제공자에서 동작하지만 조치 힌트는 Gemini 기준입니다.

## 6. Docker

```bash
cp .env.example .env               # GEMINI_API_KEY 입력
docker compose up -d --build       # 이미지 빌드 시점에 스토어 적재 + 벡터 캐시 워밍까지 수행
curl -s http://localhost/health
```

- `docker-compose.yml`이 컨테이너 8000번을 호스트 **80번**에 매핑하므로, API는 `http://localhost/answer`, 콘솔은 `http://localhost/ui`에 열립니다.
- Compose 없이: `docker build -t fin-agent-en .` 후 `docker run -d --env-file .env -p 8000:8000 fin-agent-en`(API는 8000번).
- `restart: unless-stopped`로 컨테이너 크래시·서버 재부팅 시 자동 복구됩니다.
- 기동 후 지식그래프 구축에 약 3분이 걸리므로(대회 서버 기준) Dockerfile `HEALTHCHECK`의 `start-period`를 420초로 두었습니다. 재기동했다면 `/health`가 200을 줄 때까지 기다린 뒤 호출하십시오. 이 문서의 기동 시간 수치는 편입종목 테이블 추가·그래프 재구축 **이전** 측정값이며 이후 재측정하지 않았습니다. `start-period` 420초는 그 여유를 감안한 값입니다.
- `.env`는 `.dockerignore`에 있어 이미지에 포함되지 않습니다(API 키 보호). 런타임에 `env_file: .env`로 읽으므로 **호스트의 프로젝트 디렉터리에 `.env` 파일이 있어야 합니다.**
- 선택 사항인 HTTPS: `.env`에 `DOMAIN`을 넣고 `docker compose --profile https up -d`를 실행하면 Caddy + Let's Encrypt가 443번에서 동작합니다. 서버를 가리키는 도메인이 필요합니다.
- 대회 서버는 이 Compose 구성 그대로 네이버 클라우드 플랫폼(VPC · Ubuntu 24.04 · 4 vCPU / 16GB)에서 돌았습니다. 주최측 공지가 HTTP 표준 포트를 명시하고 있어 인증서·리버스 프록시를 두지 않았습니다.

## 7. API 명세

`GET /answer?question_id=Q-001&question=...` → `200 OK`

```json
{"question_id": "Q-001", "question": "...", "retrieved_context": "...", "think_trace": "...", "answer": "..."}
```

- 응답은 `application/json`, 다섯 필드 모두 문자열입니다. 인증 헤더는 없고 경로는 `/answer` 고정입니다.
- 확인 불가 질의와 내부 오류를 포함해 모든 응답이 200 OK로 동일 스키마를 유지합니다. 정의되지 않은 파라미터는 무시하며, `question`이 비어 있으면 사용법 안내를 돌려줍니다.
- `answer`는 질문 언어로 작성됩니다(1절). `think_trace`와 `retrieved_context`의 표기는 영어입니다.
- 그 밖의 경로:
  - `GET /health`: 상태, 테이블별 기준일·행수, 색인 규모, `kg_triples` / `kg_asserted` / `kg_inferred`, 벡터 백엔드, 문서 청크 수, `제공자/모델`.
  - `GET /`: 안내 문구(평문).
  - `GET /ui`: 로컬 테스트 콘솔.
  - `GET /answer/stream`: 같은 답변을 Server-Sent Events로 단계별 스트리밍. 콘솔 전용이며, 채점 대상인 `/answer` 계약과 일부러 분리했습니다.
- 상세(요청/응답 JSON 스키마, 예시, 오류 처리)는 [docs/api_spec.md](docs/api_spec.md)에 있습니다. 이 명세는 대회 제출본 기준의 한국어 문서라 LLM을 HyperCLOVA X로 적고 있습니다.

## 8. 온톨로지 (`ontology/`)

| 파일 | 내용 |
|---|---|
| `common.ttl` | 공통 상위 클래스 `fp:Product`, 상장 상품 구분 `fp:DomesticListedProduct` / `fp:ForeignListedProduct`, 운용사·발행사·기업·테마·지수·펀드패밀리, 위험등급(1~6)·자산군·투자지역·분배주기 열거형, 공통 관계(`managedBy`, `hasConstituent`, `affiliateOf`(계열사·대칭), `subsidiaryOf`, `hasTheme`, `tracksIndex` …), SHACL 제약 |
| `bond_kr.ttl` | `fp:DomesticBond` — 신용등급 체계(AAA~D, 서열), 채권 분류·금리·이자지급 열거형, 수익률/듀레이션 등 속성 |
| `etf_kr.ttl` | `fp:DomesticETF` / `fp:DomesticETN` — 전략(실물/합성/액티브), 배수, 기간 수익률·변동성, 분배 정보 |
| `etf_gl.ttl` | `fp:ForeignETF` / `fp:ForeignETN` — 총보수, AUM, 거래소, 복제방법, 인버스/레버리지 |
| `fund_pub.ttl` | `fp:PublicFund` / `fp:PrivateFund` / `fp:ShareClass` — 운용속성, 제로인 유형, 벤치마크, 클래스 수수료·채널, 자산구성비율, 운용사 기관코드와 채움 출처(`manager_source`) |
| `etf_constituent.ttl` | 국내 ETF 편입종목 — 편입 관계, 종목코드·비중·현금성 여부·기준일. 그래프의 `hasConstituent`가 이 뷰에서 생성됩니다 |

모든 데이터 속성은 `fp:sourceTable` / `fp:sourceColumn` / `fp:unit` 주석을 가지며, 런타임(`ontology/loader.py`)이 이를 읽어 LLM 스키마 카드와 값 검증기를 만듭니다. 즉 온톨로지는 문서가 아니라 실행 계층입니다. 인스턴스(ABox)는 기동 시 SQLite에서 생성되며 `KG.export()`로 Turtle 내보내기가 가능합니다.

## 9. 평가 재현과 측정 결과

### 재현 방법

```bash
# (A) 정답 키 기반 채점 — 데이터에서 SQL/그래프로 계산한 정답과 대조 (권장)
python scripts/eval_truth.py --compute                      # 정답 키만 확인 (LLM 불필요)
python scripts/eval_truth.py --base http://127.0.0.1:8000   # 서버 답변 채점

# (B) 상태 기반 점검 — 33문항의 응답 상태·지연시간 확인 (정답 대조 없음)
python scripts/eval_run.py --base http://127.0.0.1:8000

# LLM 없이 각 저장소를 직접 들여다보기 (키 불필요)
python scripts/explore.py health
python scripts/explore.py search "차이나 반도체" --mode hybrid
python scripts/explore.py pre "신용등급 AAAA인 채권 찾아줘"

# 질문 하나를 도구 인자·도구 원문 출력까지 단계별로 따라가기 (LLM 호출)
python scripts/trace.py "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘"

# 감사 (LLM 불필요)
python scripts/audit_vector.py --sections BC   # 검색 품질: 어휘 대 벡터 대 결합 (약 14분)
python scripts/audit_graph.py                  # 지식그래프 출처·정합성 점검
```

결과는 `data/eval/results.jsonl` / `data/eval/truth_results.jsonl`에 저장됩니다(문항별 `seconds`, 채점 결과, 근거·추적 전문). 저장소의 `truth_results.jsonl`은 대회 당시 기록이며, `eval_truth.py --base`는 `--out`을 주지 않으면 이 파일을 덮어씁니다. 평가 질문 세트가 한국어이므로 Agent는 한국어로 답합니다. 무료 등급 Gemini 키로 429가 나면 `eval_truth.py`에 `--delay <초>`를 붙이십시오.

> **정답 기준에 대하여** — 주최측은 평가 질의와 정답을 공개하지 않았습니다(과제설명서: "예시질문(참고용 질의set)은 본 자료에 포함되지 않습니다").
> 따라서 `data/eval/groundtruth.jsonl`은 **제공 데이터에서 SQL·그래프로 정답이 계산되는 질문 54개**(하 10 · 중 16 · 상 9 · 답변불가 19)로 구성했습니다.
> 정답 키는 채점 시점에 데이터에서 다시 계산되므로, 데이터가 갱신되면 정답도 함께 갱신됩니다.

### 실측 결과 (대회 제출본 · HyperCLOVA X)

**실측 정답률 (2026-09-06 · `LLM_PROVIDER=clova` · `HCX-005`)** — 대회 제출본의 수치입니다. 영문판은 Gemini 기준으로 다시 측정하지 않았습니다.

| 구분 | 결과 |
|---|---|
| 자체 정답 키 54문항 | **47/54 (87%)** |
| 난이도별 | 하 8/10 · 중 14/16 · 상 6/9 · **답변불가 19/19** |
| 응답 시간 | 평균 **20.6초**, 최장 87.4초 — **300초 초과 0건** |
| 주최측 과제설명 p12 예시 8문항 | **7/8** (난이도 5문항 중 4문항 정답, 답변불가 3문항 전부 정답) |

문항별 판정·소요시간·근거·추적 전문은 `data/eval/truth_results.jsonl`에 그대로 있습니다. 이 수치는 정답 키를 20 → 54문항으로 늘리고 **키 자체의 오류**를 고친 **새 세트 기준**입니다:

- ETF를 묻는 문항에서 ETN을 센 것;
- 최저 총보수 0.01%에 4개가 동률인데 하나만 정답으로 고정한 것;
- 반올림 정수를 허용해 0.65가 "1"로도 통과하던 것;
- "…상품은 없습니다"를 거절로 인식하지 못하던 채점기.

그 이전 채점 결과는 폐기했습니다.

### 테스트

```bash
python -m pytest -q -m "not slow"   # 단위·계약 테스트 (stub LLM, 소형 스토어; 키·네트워크 불필요)
python -m pytest -q -m slow         # data/raw 기준 전체 ETL 검증 (약 2분)
```

`python -m pytest -q`만 쓰면 둘 다 실행됩니다. 대회 제출 시점에는 **194개가 통과**했습니다. 영문판은 Gemini 어댑터와 언어 판정 테스트를 추가했습니다.

## 10. 외부 데이터 (규칙: 주최측 데이터 우선)

- `data/enrichment/constituents/kr_etf_constituents.csv` — 국내 ETF **1,162종목 · 22,599행**의 상위 편입종목(네이버금융, `as_of` 2026-09-05). `scripts/fetch_constituents_naver.py`로 재수집.
  `build_store.py`가 이 CSV를 **SQLite 테이블 `etf_constituent`(뷰 `v_etf_constituent`)로 적재**하고, 지식그래프의 `hasConstituent`는 그 테이블에서 생성됩니다 — SQL 결과와 그래프 결과가 갈라지지 않습니다.
  기업 식별은 원천이 주는 **종목코드**(`company_code`, 79% 수록)로만 합니다. 이름 유사도 매칭은 쓰지 않습니다(과거 `"lg"`가 `"shinhanfinancialgroup"`의 부분 문자열로 잡혀 677행이 오귀속된 적이 있습니다).
- `data/enrichment/*.yaml` — 운용사 77 · 테마 59(한/영) · 기업 97(별칭·티커·모회사) 사전. 엔티티 해석과 그래프 관계의 원천이며, 영문 기업명·테마명도 이 사전으로 해석합니다.
  이 중 `subsidiaryOf`(모회사·자회사)는 **엣지 4개**뿐입니다 — 주최측 4종 마스터에 소유 관계 컬럼이 아예 없고, 제공 데이터에서 계산한다는 규칙을 지켜 확장하지 않았습니다. "다단계 자회사 순회"를 일반 기능으로 소개하지 않습니다.
- `data/external/docs/*.md` — 한국어 참고 문서 3종: 국민성장펀드 개요(금융위 보도자료 출처), ETF 용어·지표 해설, ETF 위험요인 일반. `scripts/parse_docs.py`로 PDF/PPTX를 Markdown으로 변환해 추가할 수 있습니다(`.pptx`는 선택 패키지 `python-pptx` 필요).

## 11. 제한 사항

- **영문판은 Gemini 기준으로 다시 측정하지 않았습니다.** 정답률·거절 동작·응답 시간이 측정되지 않았으며, 이 문서의 모든 수치는 HyperCLOVA X로 동작한 대회 제출본의 것입니다.
- **데이터는 한국어입니다.** 상품명·운용사명·기업명과 참고 문서가 한국어이므로, 영어 답변도 그 이름은 데이터에 있는 그대로 한글로 인용합니다.
- 상품 마스터는 2026-08-21/22 스냅샷이며 시세·수익률은 해당 기준일 값입니다.
- 국내ETF 총보수는 18%만 수록, 펀드 보수 컬럼은 단위 미공개(상대 비교만), 펀드 수익률은 약 30%만 수록 — 없는 값은 "데이터 없음"으로 답합니다.
- 해외ETF·펀드의 편입종목 데이터는 없습니다. 편입종목은 국내 ETF에만 있습니다(1,235개 중 1,162개 수집).
- 편입종목의 기준일(`as_of` 2026-09-05)이 상품 마스터 기준일(2026-08-21/22)과 다릅니다. 두 소스를 섞어 답할 때는 시점 차이를 함께 밝힙니다.
- 펀드 운용사는 91.2% 채워져 있고 **2,084행은 여전히 NULL**입니다(기관코드가 특정 표기로 수렴하지 않는 경우). 운용사 개명 대응표는 만들지 않았습니다 — 펀드 설정일이 없어 어느 표기가 현재 이름인지 판정할 수 없기 때문입니다.
- **회사 간 관계는 `계열사`이지 `자회사`가 아닙니다.** 그룹주 ETF(`PLUS 한화그룹주`, `TIGER 삼성그룹` 등)의 편입종목이 곧 그 그룹의 구성사이므로 거기서 `affiliateOf` 를 도출했습니다 — 8개 그룹 65개사 608엣지, 오탐 0건. 다만 이것은 **같은 기업집단**이라는 사실이지 **지분 소유**가 아닙니다. 한화시스템·한화오션은 한화에어로스페이스의 자회사지만 한화생명은 (주)한화의 자회사이며, 데이터에 지분율이 없어 그 층위를 구분할 수 없습니다. 그래서 답변에도 "계열사"로 표기합니다. 손으로 적은 `subsidiaryOf` 는 엣지 4개 그대로입니다(에코프로 계열 3 · SK이노베이션 1).
- **그룹주 ETF에서 뽑히는 계열사는 상장사 일부입니다.** 비상장 계열사는 보이지 않고, 현대차그룹은 2개사만 잡힙니다 — `TIGER 현대차그룹플러스` 가 HD현대(별개 기업집단)를 섞고 있어 토큰을 `현대차`로 좁게 잡았기 때문입니다. 넓히면 recall 이 오르지만 오탐이 들어옵니다. SK그룹은 해당 ETF에 편입종목 행이 없어 커버되지 않습니다.
- **벡터 색인의 순기여는 테마형 질의의 recall@10 +3.5%p 입니다**(질의 1,313개 실측, 4절 참조). 상위 순위는 어휘 검색이 결정하며, 벡터는 하위 슬롯만 채웁니다. 필수 구성요소로서 존재하고 동작하지만, 이 코퍼스에서 의미 검색이 어휘 검색을 이기는 구간은 좁다는 사실을 감추지 않습니다.
- **외부 문서는 3개 파일 39청크뿐입니다.** ETF 용어·지표 해설, ETF 위험요인 일반, 국민성장펀드 개요. 서술형 질의 전반을 덮지 못하며, 마스터에 없는 배경 지식이 필요한 질문은 확인 불가로 답합니다.
- 벡터 검색은 외부 임베딩 API 키 없이도 평가가 재현되도록 TF-IDF+LSA(384차원)를 기본값으로 씁니다. 상용 임베딩으로 바꾸면 의미 검색 품질이 더 오를 여지가 있으며, `VECTOR_BACKEND=openai`와 `VECTOR_EMBED_MODEL`로 전환할 수 있습니다(코드 변경 불필요). **다만 그 전환 효과는 측정하지 않았습니다.**
- **측정하지 않은 것** — Gemini 기준 정답률·응답 시간(위 참조), 컨테이너 기동 시간(편입종목 테이블 추가·그래프 재구축 이후 재측정 필요), 동시 요청 처리량(주최측이 순차 호출을 공지해 부하 시험을 하지 않았습니다), 상용 임베딩 전환 효과. 이것들에는 이 문서에 숫자를 쓰지 않습니다.
- **제공 데이터가 과제설명 PDF의 기재와 다릅니다.** PDF는 4종 합계 약 14.5만 건·기준일 2026-07-11로 안내하지만, 실제 배포된 파일(`ai-festival2026_..._DtataSet260824.zip`)은 합계 53,375건이고 기준일이 2026-08-21(해외ETF 2026-08-22)입니다. 본 구현은 **실제 제공 파일**을 기준으로 하며, 평가용 데이터가 갱신되면 `python scripts/build_store.py` 재실행만으로 반영됩니다(스키마는 동일 가정).

## 12. 디렉터리 구조

```
fin-agent-en/
├─ src/finagent/           # 구현체 (etl / ontology / kb / agent / api.py, i18n.py = 답변 언어 판정)
├─ ontology/               # TTL 6종: common, bond_kr, etf_kr, etf_gl, fund_pub, etf_constituent
├─ data/raw/               # 주최측 제공 Excel 8개 (상품 마스터 4종의 data + schema)
├─ data/enrichment/        # managers.yaml, themes.yaml, companies.yaml, constituents/kr_etf_constituents.csv
├─ data/external/docs/     # 외부 참고 문서 (한국어 Markdown, front-matter에 출처·기준일)
├─ data/eval/              # questions.jsonl(상태 점검 33문항), groundtruth.jsonl(정답 키 54문항),
│                          # truth_results.jsonl(대회 당시 채점 기록)
├─ data/tuning/            # 대회 당시 만든 CLOVA Studio 튜닝 데이터셋 (런타임 미사용)
├─ data/store/             # 빌드 산출물 products.db, vector_cache.npz (git 제외)
├─ scripts/                # build_store, warmup, check_llm, explore, trace, eval_truth, eval_run,
│                          # audit_vector, audit_graph, parse_docs, fetch_constituents_naver, build_tuning_data
├─ tests/                  # pytest
├─ web/console.html        # 로컬 테스트 콘솔 (/ui 에서 제공)
├─ docs/                   # anatomy.html(영어), anatomy.ko.html(한국어), api_spec.md, 제안서 초안, superpowers/
├─ Wolves_of_Walnut_Street_Proposal.html / .pdf   # 기술제안서 영어 번역본
├─ Wolves_of_Walnut_Street_제안서.html / .pdf       # 기술제안서 한국어 원본
└─ requirements.txt · pyproject.toml · Dockerfile · docker-compose.yml · Caddyfile · .env.example
```

## 13. 문서

| 문서 | 내용 |
|---|---|
| [`Wolves_of_Walnut_Street_Proposal.html`](Wolves_of_Walnut_Street_Proposal.html) / [`.pdf`](Wolves_of_Walnut_Street_Proposal.pdf) | 기술제안서 영어 번역본 |
| [`Wolves_of_Walnut_Street_제안서.html`](Wolves_of_Walnut_Street_제안서.html) / [`.pdf`](Wolves_of_Walnut_Street_제안서.pdf) | 기술제안서 한국어 원본(제출본) |
| [`docs/anatomy.html`](docs/anatomy.html) | 시스템 전체 해설, 영어 |
| [`docs/anatomy.ko.html`](docs/anatomy.ko.html) | 같은 해설의 한국어 원본 |
| [`docs/api_spec.md`](docs/api_spec.md) | API 명세(한국어, 대회 제출본 기준) |
| [`docs/proposal_outline.md`](docs/proposal_outline.md), [`docs/proposal_skeleton.md`](docs/proposal_skeleton.md) | 제안서 초안(한국어) |
| [`docs/superpowers/`](docs/superpowers/) | 개발 당시의 설계 명세와 구현 계획 |

## 14. 대회 당시 기록 (HyperCLOVA X)

- **평가용 엔드포인트.** 주최측 평가 기간(2026-09-07 ~ 09-20)에 대회 제출본이 `http://223.130.161.217/answer`로 서비스했습니다(네이버 클라우드 플랫폼 VPC · Ubuntu 24.04 · Docker Compose · 표준 포트 80, 인증 없음). 그 엔드포인트는 HyperCLOVA X로 동작하며 한국어로 답했고, 이 영문판이 아닙니다.
- **LLM.** 2026-08-24 주최측 공지로 답변 생성 파이프라인에 NCP HyperCLOVA X가 필수가 되었습니다. 그래서 대회 제출본은 `LLM_PROVIDER=clova` · `HCX-005`를 기본값으로, CLOVA Studio OpenAI 호환 엔드포인트(`https://clovastudio.stream.ntruss.com/v1/openai`)로 호출했습니다.
- **대회 제출본이 `LLM_MAX_TOKENS`를 1536으로 제한한 이유.** 4096 상한에서는 호출 한 번이 92.9초 동안 답 대신 방법론 서술을 썼습니다. 데이터는 이미 두 라운드 전에 확보돼 있었습니다. 1536은 현실적 최악(교차질의 10행 표 + 근거, 약 813토큰)을 담으면서 수천 토큰짜리 서술은 불가능하게 만드는 값이었고, CLOVA function calling은 1024 이상이어야 동작했습니다. HyperCLOVA X에는 thinking 토큰이 없지만 Gemini는 thinking 토큰이 상한에 포함되므로, 영문판 기본값은 8192입니다.
- **요청 한도.** CLOVA는 분당 요청 수와 분당 토큰 수를 따로 재며 둘 중 하나만 넘어도 429를 돌려줍니다. 클라이언트의 `x-ratelimit-*` 헤더 처리(한도 소진 전에 대기)는 그때 만든 것이며, 그런 헤더를 보내는 제공자에게는 지금도 적용됩니다.
- **튜닝 데이터.** CLOVA Studio는 튜닝과 function calling을 동시에 쓸 수 없어서, `scripts/build_tuning_data.py`가 도구가 필요 없는 text→text 구간의 학습셋(`nl2sql`·`answer`·`intent`)을 `data/tuning/`에 만들었습니다. SQL을 실제로 실행해 결과가 나오는 쌍만 남긴 것입니다. Agent는 런타임에 이 데이터를 쓰지 않습니다.
