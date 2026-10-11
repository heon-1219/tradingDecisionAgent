English | [한국어](#한국어)

*Note: This skeleton describes the competition build, for which the competition rules mandated NCP HyperCLOVA X. This English edition runs on Google Gemini (default model `gemini-3.8-flash`) instead.*

# Technical Proposal Skeleton — 10th Mirae Asset Securities AI Festival 2026 · Financial Product Agent

Submission deadline **2026-09-06 23:59** · free format (PDF recommended) · weight **40%**

This document is not the proposal itself but **a frame for filling in the proposal**. For each section it records
`claim / the measurement that backs the claim / figure to include / source file`.
**A section whose measurement cell is empty must not be written yet.** Anything not measured is written up as not measured.

---

## Scoring map — which section earns which part of the score

| Organizer's scoring item | Weight | Where the proposal answers it |
|---|---|---|
| ① Source code (implementation · environment definition · README · Ontology.ttl) | 20% | 06 Appendix + the repository itself |
| ② **Technical proposal** (problem definition · proposed method · architecture diagram · expected impact) | **40%** | **All of 01 – 06** |
| ③ API answers (accuracy · evidence display · response time) | 40% | Reflected as measured figures in 02-4 and 05-2 |

The organizer's recommended outline has 6 sections. **Use the section numbers and titles exactly as given** — graders must be able to find things easily.

---

## 0. Cover + Proposal Summary (1p)

- Team name / task name / submission date / repository URL / **API End-point URL**
- 5-line summary: what we built · what is different · what we measured
- **One-line claim**: "a federated retrieval agent that uses the ontology not as documentation but as a **runtime component**"

> The API End-point URL is a **mandatory** item per p8. Put it in large type on the cover.

---

## 01. Overall Architecture and Specialized Features (2p)

**Core claim** — one question splits into five paths — RDB · knowledge graph · lexical index · vector index · document store —
and merges back into one answer (federated retrieval). And because those five paths
reference **the same single ontology**, they can verify each other's answers.

**3 differentiators — say only these (do not add more)**

1. **The ontology is a runtime component.** 6 TTL files → schema cards + validators generated automatically at startup.
   Adding a new product group = 1 view + 1 TTL. (Basis: `ontology/loader.py`)
2. **SQL and the graph verify each other.** Constituents are moved into a single SQLite table and
   the graph is generated from that table → both paths return the same counts. (Basis: 05-2 ⑤)
3. **Recount after every change.** A procedure born from the experience of 11 defects passing with no exception, log or red test.
   (Basis: 05-2 quality-control principle)

| To include | Source |
|---|---|
| One-page architecture figure | Top diagram of `docs/anatomy.html` |
| Scale numbers (1.08M triples · 75,089 indexed documents · 215 columns) | `/health` response |

---

## 02. Proposed Method (4p) — structured as answers to the three questions the organizer specified

### 02-1. What specialized data did we collect, and how did we clean it?

- **Principle first**: organizer-provided data is the reference, and external collection is used only for **items for which the provided data has no column at all**.
  → Proven by the provenance of the knowledge graph: rooted in provided data **85.9%**, external collection **6.0%** (05-2 ④)
- **3 cleaning cases** (all of them came out of failures, which makes them convincing):
  - Redefining "available for purchase" — reflects the notice that `BUYABLE_QUANTITY` is invalid; bonds 634 → 21,866
  - Manager name — derived by majority vote over the `or_co_xtn_itt_cd` code rather than from the fund-name string; coverage 74.6% → 91.2%,
    source recorded in `manager_source`. Sentinel `99999999` excluded
  - Constituent identifiers — reusing the alias matcher for bulk loading misattributed 677 rows (LG→Shinhan Financial Group, KT→SKT).
    Fixed by restoring the source `itemCode` and removing the alias matcher
- **Handling missing values**: they are intentional, so they are not counted as 0; they are excluded or stated as "not provided"

| To include | Source |
|---|---|
| Table of row counts and base dates for the 4 masters | `/health` |
| Before/after cleaning comparison table | The 3 cases above |

### 02-2. What ontology engineering did we apply?

- **6-file layout**: `common.ttl` (fp:Product superclass) + `bond_kr` `etf_kr` `etf_gl` `fund_pub` + `etf_constituent`
  → meets the p9 mandatory submission criteria as-is
- **2 design decisions — this is the core of this section**
  - `rdfs:domain` is **an inference rule, not a constraint**. Where a constraint was needed, we used a SHACL NodeShape.
    (Learned from an incident where using domain as a constraint wrongly covered several row types in one table)
  - The TBox is merged into the ABox and inference is **materialized in advance** → zero inference cost at query time
- **215 of 261 properties are linked to actual view columns**. Unlinked properties do not appear in the schema cards,
  and a column that is not in the schema cards **effectively does not exist** for the model (the `has_buy_yield` case in 05-2 ③).

| To include | Source |
|---|---|
| Class hierarchy diagram | Ontology section of `docs/anatomy.html` |
| Mermaid ERD | `docs/anatomy.html` / `Mermaid ERD/` in the parent folder |

### 02-3. Which retrieval methodology did we choose, and why?

- **5 federated retrieval paths** and the question types each path answers
- **Why we did not merge them into one**: SQL does aggregation and the graph does relations.
  When the **same table is read in two ways**, as with constituents, errors are caught by comparing the two answers (05-2 ⑤)
- **Design law — derived from measurement**:
  > Facts supplied together at the point of use were followed (5/5); prompt rules given in advance were ignored (4/4).
  → So instead of adding rules, **change the schema cards and example SQL themselves.**
  (e.g. removing foreign ETFs from the UNION in the cross-product query example made the failures disappear — the warning text was ignored, but the example was copied)

### 02-4. Answer-generation principles (addressing the organizer's p6 rules — as a separate subsection)

| Organizer's requirement | Our implementation |
|---|---|
| Show evidence (referenced data) | Mandatory `retrieved_context` field + `(근거: 표 · 기준일)` ("(Evidence: table · base date)") at the end of the answer |
| State explicitly or ask back for unconfirmable questions | The ontology decides "does this product group have that data?" → if not, decline definitively |
| No unfounded return forecasts | Detect forecast-type questions and decline. The schema cards state that the masters are **base-date snapshots** |
| No definitive investment recommendations | Caution notice only on recommendation/ranking answers, not on simple lookups |

> ⚠ **Open issue**: whether to remove the evidence citation inside `answer`. "Evidence display" is explicitly part of the p7 scoring,
> and the p11 example puts `(근거: 해외ETF마스터)` ("(Evidence: foreign ETF master)") inside the answer. **Decide before submission.**

---

## 03. System Architecture Diagram (1p) — Ingestion → Indexing → Retrieval → Generation

A one-page pipeline figure. Stamp **measured numbers** on every stage.

```
Ingestion   4 masters + constituents → 5 SQLite views (53,974 rows)
Indexing    BM25 lexical index + TF-IDF/LSA vector index (75,089 documents · 384 dimensions)
Retrieval   ontology schema cards → tool selection → SQL / graph / documents (cross-verification)
Generation  HCX-005 function calling → validators → output with a separate evidence field
```

- Startup breakdown: ontology 2,964 triples → index → vectors (15.2 s, cached) → graph 1.08M triples
- **LLM**: NCP HyperCLOVA X `HCX-005`, OpenAI-compatible endpoint. Thanks to the adapter structure, the cost of switching providers ≈ 0

---

## 04. Main Feature Flow (1p) — Data Collection → Ontology Design → Graph Construction → Retrieval

**A single figure that follows one question all the way through** shows all four stages.
Pick the example question from the same family as the organizer's p12 [hard] example:

> "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘"
> ("Show me the 3 domestic ETFs holding Samsung Electronics with the largest net assets")

Draw it so that actual values appear at each stage (entity resolution → 229 graph neighbors → SQL sort → validation → answer).
Screenshots of the trace in `docs/anatomy.html` and of the `/ui` console serve directly as evidence.

---

## 05. Expected Impact · Scalability (1p)

- **Where it applies in the business**: advisory assistance · product discovery · compliance (evidence citation and the no-forecast rule are built into the structure)
- **Extension**: a new product group = 1 view + 1 TTL + lexicon entries. The loader generates schema cards and validators automatically
- **Audit trail**: the base date and evidence are always attached to the response

---

## 06. Appendix (2–3p)

- Ontology schema summary table (6 files · 215 of 261 properties linked)
- **API specification** — request/response JSON schema (`docs/api_spec.md`). Mandatory item per p8
- Evaluation reproduction procedure (`README` section 7) — graders must be able to run it as-is
- Test list (currently 80 pass)
- **Data limitations** — writing these honestly here is what earns points
  - Foreign ETFs have only one period return, `du_er_1d` (1 day) → excluded from 1-year comparisons
  - Public funds include no fee information
  - Public funds and foreign ETFs have no constituent data → such queries are declined
  - Returns such as `fd_yr1_ern_r` contain outliers originating in the source (1-year >200% in 25/7,022 rows).
    State explicitly that, as organizer-provided values, they are **used as-is without modification**
- System anatomy (`docs/anatomy.html`)

---

## Blanks to fill before submission (after re-measurement)

| Blank | Where it goes | Status |
|---|---|---|
| Accuracy __/54 (__%) | 05-2 | ⏳ measurement started 16:29 |
| Average response time __ s / max __ s | 05-2, 03 | ⏳ |
| Number of questions over 60 s | 05-2 | ⏳ |
| Accuracy on unanswerable questions (penalty item) | 02-4 | ⏳ |
| Container startup time | 03 | Not measured |
| API End-point URL | Cover, 06 | After deployment |

**Rule**: do not write an accuracy figure in 05-2 until the table above is complete.
Earlier grading results were discarded because the answer key itself had errors.

---

## Page allocation (for a 12–14 page PDF)

| Section | Pages |
|---|---|
| 0 Cover · summary | 1 |
| 01 Architecture | 2 |
| 02 Proposed method | 4 |
| 03 Architecture diagram | 1 |
| 04 Flow diagram | 1 |
| 05 Expected impact | 1 |
| 06 Appendix | 2–3 |

**At least 7 figures must go in.** Graders look rather than read.
They are all already in `docs/anatomy.html`, so they only need to be captured.

---

## 한국어

[English](#technical-proposal-skeleton--10th-mirae-asset-securities-ai-festival-2026--financial-product-agent) | 한국어

*참고: 이 뼈대는 대회 규정상 NCP HyperCLOVA X 사용이 필수였던 대회 제출본을 기준으로 합니다. 이 영어판은 대신 Google Gemini(기본 모델 `gemini-3.8-flash`)로 동작합니다.*

# 기술제안서 뼈대 — 제10회 2026 미래에셋증권 AI Festival · 금융상품 Agent

제출 마감 **2026-09-06 23:59** · 자유 양식(PDF 권장) · 배점 **40%**

이 문서는 제안서 자체가 아니라 **제안서를 채우기 위한 틀**이다. 각 절마다
`주장 / 그 주장을 받치는 측정치 / 넣을 그림 / 원본 파일`을 적어 두었다.
**측정치 칸이 빈 절은 아직 쓰면 안 되는 절이다.** 측정하지 않은 것은 측정하지 않았다고 쓴다.

---

## 채점 대응표 — 어느 절이 어느 배점을 받는가

| 주최측 배점 | 비중 | 제안서에서 답하는 곳 |
|---|---|---|
| ① 소스코드 (구현체·환경정의·README·Ontology.ttl) | 20% | 06 부록 + 리포지토리 자체 |
| ② **기술제안서** (문제정의·제안방법·구성도·기대효과) | **40%** | **01 ~ 06 전체** |
| ③ API 답변 (정확성·근거표시·응답시간) | 40% | 02-4, 05-2 에 실측치로 반영 |

주최측 권장 목차는 6개다. **목차 번호와 제목을 그대로 쓴다** — 채점자가 찾기 쉬워야 한다.

---

## 0. 표지 + 제안 요약 (1p)

- 팀명 / 과제명 / 제출일 / 리포지토리 URL / **API End-point URL**
- 요약 5줄: 무엇을 만들었는가 · 무엇이 다른가 · 무엇을 측정했는가
- **한 줄 주장**: "온톨로지를 문서가 아니라 **런타임 부품**으로 쓰는 연합 검색 에이전트"

> API End-point URL 은 p8 에서 **필수 명시** 항목이다. 표지에 크게 박는다.

---

## 01. 전체 아키텍처 및 특화기능 (2p)

**핵심 주장** — 하나의 질문이 RDB · 지식그래프 · 어휘색인 · 벡터색인 · 문서저장소
다섯 경로로 갈라졌다가 하나의 답으로 합쳐진다(연합 검색). 그리고 그 다섯 경로가
**같은 온톨로지 한 벌**을 참조하므로 서로의 답을 검증할 수 있다.

**차별화 포인트 3개 — 이것만 말한다(더 늘리지 않는다)**

1. **온톨로지가 런타임 부품이다.** TTL 6개 → 스키마 카드 + 검증기를 기동 시 자동 생성.
   새 상품군 추가 = 뷰 1개 + TTL 1개. (근거: `ontology/loader.py`)
2. **SQL과 그래프가 서로를 검증한다.** 편입종목을 SQLite 테이블 하나로 내리고
   그래프를 그 테이블에서 생성 → 두 경로가 같은 수를 낸다. (근거: 05-2 ⑤)
3. **바꾼 뒤에 다시 센다.** 결함 11건이 예외·로그·빨간 테스트 없이 통과했던 경험에서
   나온 절차. (근거: 05-2 품질관리 원칙)

| 넣을 것 | 원본 |
|---|---|
| 아키텍처 1장 그림 | `docs/anatomy.html` 상단 다이어그램 |
| 규모 숫자 (트리플 108만 · 색인 75,089문서 · 컬럼 215개) | `/health` 응답 |

---

## 02. 제안 방법 (4p) — 주최측이 지정한 세 질문에 답하는 형식

### 02-1. 어떤 특화 데이터를 어떻게 수집·정제했는가

- **원칙 먼저**: 주최측 제공 데이터가 기준이고, 외부 수집은 **제공 데이터에 컬럼 자체가 없는 항목**에만 쓴다.
  → 지식그래프 출처 구성으로 증명: 제공 데이터 뿌리 **85.9%**, 외부 수집 **6.0%** (05-2 ④)
- **정제 사례 3개**(전부 실패에서 나온 것이라 설득력이 있다):
  - 구매가능 재정의 — `BUYABLE_QUANTITY` 무효 공지 반영, 채권 634건 → 21,866건
  - 운용사명 — 펀드명 문자열이 아니라 `or_co_xtn_itt_cd` 코드 다수결로 도출, 커버리지 74.6% → 91.2%,
    `manager_source` 로 출처 표기. 센티널 `99999999` 제외
  - 편입종목 식별자 — 별칭 매칭기를 대량 적재에 재사용해 677행이 오귀속(LG→신한지주, KT→SKT).
    원천 `itemCode` 복원 + 별칭 매칭기 제거로 해결
- **결측 처리**: 의도된 결측이므로 0으로 계산하지 않고 제외하거나 "제공되지 않음" 명시

| 넣을 것 | 원본 |
|---|---|
| 4개 마스터 건수·기준일 표 | `/health` |
| 정제 전후 비교 표 | 위 3개 사례 |

### 02-2. 어떤 온톨로지 엔지니어링을 적용했는가

- **6파일 구성**: `common.ttl`(fp:Product 상위) + `bond_kr` `etf_kr` `etf_gl` `fund_pub` + `etf_constituent`
  → p9 필수 제출 기준을 그대로 충족
- **설계 판단 2개 — 여기가 이 절의 알맹이다**
  - `rdfs:domain` 은 **제약이 아니라 추론 규칙**이다. 제약이 필요한 자리에는 SHACL NodeShape 을 썼다.
    (domain 을 제약처럼 쓰다가 한 테이블 안의 여러 행 유형을 잘못 덮은 사고에서 배운 것)
  - TBox 를 ABox 에 병합하고 추론을 **미리 물질화**했다 → 질의 시점 추론 비용 0
- **속성 261개 중 215개가 실제 뷰 컬럼에 연결**. 연결되지 않은 속성은 스키마 카드에 안 나타나고,
  스키마 카드에 없는 컬럼은 모델에게 **존재하지 않는 것과 같다**(05-2 ③ 의 `has_buy_yield` 사례).

| 넣을 것 | 원본 |
|---|---|
| 클래스 계층도 | `docs/anatomy.html` 온톨로지 섹션 |
| Mermaid ERD | `docs/anatomy.html` / 상위폴더 `Mermaid ERD/` |

### 02-3. 어떤 검색 방법론을 선택했고 왜 그렇게 설계했는가

- **연합 검색 5경로**와 각 경로가 답하는 질문 유형
- **왜 하나로 합치지 않았는가**: SQL은 집계를 하고 그래프는 관계를 한다.
  편입종목처럼 **같은 테이블을 두 방식으로 읽는** 경우, 두 답을 대조해 오류를 잡는다(05-2 ⑤)
- **설계 법칙 — 측정에서 나온 것**:
  > 사용 시점에 함께 준 사실은 지켜졌고(5/5), 미리 준 프롬프트 규칙은 무시됐다(4/4).
  → 그래서 규칙을 늘리는 대신 **스키마 카드·예시 SQL 자체를 바꾼다.**
  (예: 교차질의 예시에서 해외ETF를 UNION 에서 빼자 실패가 사라졌다 — 경고문은 무시됐지만 예시는 복사됐다)

### 02-4. 답변 생성 원칙 (주최측 p6 규칙 대응 — 별도 소절로 뺀다)

| 주최측 요구 | 우리 구현 |
|---|---|
| 근거(참조 데이터) 표시 | `retrieved_context` 필수 필드 + answer 말미 `(근거: 표 · 기준일)` |
| 확인 불가 질의는 명시 또는 역질문 | 온톨로지가 "이 상품군에 그 데이터가 있는가"를 판정 → 없으면 단정적으로 거절 |
| 근거 없는 수익률 전망 금지 | 전망성 질문 판별 후 거절. 마스터는 **기준일 스냅샷**이라는 사실을 스키마 카드에 명시 |
| 단정적 투자 추천 금지 | 추천·순위 답변에만 유의 문구, 단순 조회에는 붙이지 않음 |

> ⚠ **미결 사항**: answer 안의 근거 표기를 뺄 것인가. p7 배점에 "근거 표시"가 명시되어 있고
> p11 예시는 answer 안에 `(근거: 해외ETF마스터)` 를 넣는다. **제출 전 확정할 것.**

---

## 03. 시스템 구성도 (1p) — 적재 → 색인 → 검색 → 생성

한 장짜리 파이프라인 그림. 각 단계에 **실측 숫자**를 박는다.

```
적재  4개 마스터 + 편입종목 → SQLite 뷰 5개 (53,974행)
색인  BM25 어휘색인 + TF-IDF/LSA 벡터색인 (75,089문서 · 384차원)
검색  온톨로지 스키마 카드 → 도구 선택 → SQL / 그래프 / 문서 (교차 검증)
생성  HCX-005 function calling → 검증기 → 근거 필드 분리 출력
```

- 기동 시간 내역: 온톨로지 2,964트리플 → 색인 → 벡터(15.2초, 캐시) → 그래프 108만 트리플
- **LLM**: NCP HyperCLOVA X `HCX-005`, OpenAI 호환 엔드포인트. 어댑터 구조라 제공자 교체 비용 ≈ 0

---

## 04. 주요 기능 흐름도 (1p) — 데이터 수집 → 온톨로지 설계 → 그래프 구축 → 검색

**한 질문을 끝까지 따라가는 그림 하나**가 네 단계를 다 보여준다.
예시 질문은 주최측 p12 난이도 [상] 예시와 같은 계열로 고른다:

> "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘"

단계마다 실제 값이 나오게 그린다(개체해소 → 그래프 이웃 229건 → SQL 정렬 → 검증 → 답변).
`docs/anatomy.html` 의 트레이스와 `/ui` 콘솔 화면을 캡처해 붙이면 그대로 근거가 된다.

---

## 05. 기대효과 · 확장성 (1p)

- **현업 적용 지점**: 상담 보조 · 상품 탐색 · 컴플라이언스(근거 표기·전망 금지가 구조에 내재)
- **확장**: 새 상품군 = 뷰 1개 + TTL 1개 + 사전 항목. 로더가 스키마 카드·검증기를 자동 생성
- **감사 추적**: 기준일과 근거가 항상 응답에 붙는다

---

## 06. 부록 (2~3p)

- 온톨로지 스키마 요약표 (6파일 · 속성 261개 중 215개 연결)
- **API 명세** — 요청/응답 JSON 스키마 (`docs/api_spec.md`). p8 필수 항목
- 평가 재현 절차 (`README` 7절) — 채점자가 그대로 돌릴 수 있어야 한다
- 테스트 목록 (현재 80 pass)
- **데이터 제한 사항** — 여기에 정직하게 적는 것이 점수가 된다
  - 해외ETF에는 기간 수익률이 `du_er_1d`(1일) 하나뿐 → 1년 기준 비교에서 제외
  - 공모펀드에는 보수 정보 미포함
  - 공모펀드·해외ETF에는 편입종목 데이터 없음 → 해당 질의는 거절
  - `fd_yr1_ern_r` 등 수익률에 원천 유래 이상치 존재(1년 >200% 가 25/7,022행).
    주최측 제공 값이므로 **가공하지 않고 그대로 사용**한다고 명시
- 시스템 해부도 (`docs/anatomy.html`)

---

## 제출 전 채워야 할 빈칸 (재측정 완료 후)

| 빈칸 | 들어갈 곳 | 상태 |
|---|---|---|
| 정답률 __/54 (__%) | 05-2 | ⏳ 16:29 측정 시작 |
| 평균 응답시간 __초 / 최대 __초 | 05-2, 03 | ⏳ |
| 60초 초과 문항 수 | 05-2 | ⏳ |
| 답변불가 문항 정확도 (감점 항목) | 02-4 | ⏳ |
| 컨테이너 기동 시간 | 03 | 미측정 |
| API End-point URL | 표지, 06 | 배포 후 |

**규칙**: 위 표가 다 차기 전에는 05-2 에 정답률 숫자를 쓰지 않는다.
이전 채점 결과는 정답 키 자체에 오류가 있어 폐기했다.

---

## 분량 배분 (PDF 12~14p 기준)

| 절 | 쪽 |
|---|---|
| 0 표지·요약 | 1 |
| 01 아키텍처 | 2 |
| 02 제안 방법 | 4 |
| 03 구성도 | 1 |
| 04 흐름도 | 1 |
| 05 기대효과 | 1 |
| 06 부록 | 2~3 |

**그림이 7장 이상 들어가야 한다.** 채점자는 읽지 않고 본다.
전부 `docs/anatomy.html` 에 이미 있으므로 캡처만 하면 된다.
