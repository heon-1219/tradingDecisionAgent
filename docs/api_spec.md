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
