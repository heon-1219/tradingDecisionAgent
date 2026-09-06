# 평가용 API 명세

## Endpoint

| 항목 | 값 |
|---|---|
| Method / Path | `GET /answer` |
| Content-Type (응답) | `application/json; charset=utf-8` |
| 인증 | 없음 (인증 헤더·POST 바디 미사용) |
| 응답 시간 | 주최측 기준: **300초 초과 시 미응답 처리**, 그 안이면 늦어도 채점(점수 차등). 서버 내부 예산 120초, 초과 시 확보된 근거로 즉시 답변 |
| 답변 생성 LLM | NCP HyperCLOVA X (`HCX-005`, CLOVA Studio OpenAI 호환 엔드포인트) — 대회 규정 |
| 보조 | `GET /health` (상태·테이블 기준일·색인 규모·그래프 명시/추론 트리플 수·벡터 백엔드), `GET /` (안내 문구) |
| 참조 데이터 | SQLite 뷰 5개(`v_bond_kr` · `v_etf_kr` · `v_etf_gl` · `v_fund_pub` · `v_etf_constituent`) · 상품 53,375행 + 편입종목 22,599행 · 지식그래프 약 107만 트리플 |

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
| `answer` | string | 필수 | 최종 답변(한국어). 확인 불가 시 "제공된 데이터로는 확인할 수 없습니다"로 시작, 조건 모호 시 역질문(물음표로 끝남) |

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
