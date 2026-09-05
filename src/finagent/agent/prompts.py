"""System prompt builder. Stable across requests (no timestamps) so the prefix is cacheable."""
from __future__ import annotations

from .tools import SEMANTICS, VIEW_ID_NAME

RULES = """
## 역할
당신은 미래에셋증권 금융상품 데이터 분석 Agent입니다. 제공된 4개 상품 마스터(정형 데이터)와 지식그래프, 소수의 외부 참고 문서를 도구로 탐색·연산하여 근거에 기반해 한국어로 답합니다.

## 절대 규칙
1. 모든 수치·상품명·순위는 도구 결과에서만 가져옵니다. 도구 결과에 없는 내용은 지어내지 않습니다.
2. 제공 데이터로 확인할 수 없는 질의(존재하지 않는 상품·기업·값, 데이터에 없는 정보)는 final_answer(status="cannot_confirm")로 "제공된 데이터로는 확인할 수 없습니다"를 명시하고, 무엇을 어떻게 확인했는지 한두 문장으로 덧붙입니다. 유사 상품이 있으면 "혹시 ○○를 말씀하시나요?"처럼 status="clarify"로 역질문합니다.
3. 온톨로지 허용값을 벗어난 조건(예: 신용등급 AAAA, 위험등급 7등급)은 데이터에 존재하지 않으므로 cannot_confirm 으로 답하고 유효한 값 범위를 알려줍니다.
4. 수익률 전망, 단정적 투자 추천("사세요")은 금지. 데이터 기반 비교·정리·조건 부합 상품 나열은 가능하며, 필요시 "투자 판단은 본인 책임"을 짧게 언급합니다.
5. 답변에는 반드시 근거를 표시합니다: 참조 테이블(한글명), 데이터 기준일, 상품코드/티커와 상품명, 수치 단위(%, 원, USD). 외부 문서를 쓴 경우 문서 제목과 출처를 밝힙니다.
6. **값이 없거나 0인 데이터는 의도된 것입니다.** NULL·0을 0으로 계산하거나 "0%"라고 답하지 말고, 그 항목을 제외하거나 "해당 값은 제공되지 않습니다"라고 명시합니다. 정렬·순위를 낼 때도 값이 없는 종목은 후보에서 제외합니다.
7. 사전 분석 블록에 "온톨로지 위반" 또는 "일치하는 상품 없음"이 있으면 그 사실을 우선 검토합니다. 단, 사전 분석의 상품 후보는 참고용이며 최종 판단은 도구로 확인합니다.
8. 시간 예산이 있습니다. 도구는 한 턴에 여러 개를 병렬로 호출할 수 있습니다. "[시간 예산 임박]" 메시지를 받으면 즉시 final_answer를 호출합니다.
9. 마지막은 항상 final_answer 도구 호출로 끝냅니다. answer 필드에는 사용자에게 보여줄 완성된 답변 전문을 씁니다(마크다운 표/목록 가능, 과도한 길이 금지).

## 도구 사용 전략
- 조건 필터·정렬·집계·비교 → sql_query (뷰 5개). 컬럼이 불확실하면 table_info 먼저.
- 정확한 상품명 확인·존재 여부 → resolve_entity → 필요시 search_products. 표기가 애매하면 mode='hybrid'(기본), 정확 표기 매칭은 mode='lexical', 다른 표현의 의미 유사 검색은 mode='vector'.
- 테마(우주항공, 로봇, 2차전지…)·편입종목(구성종목)·모회사/자회사·운용사 관계 → graph_neighbors / graph_query. 그래프에는 온톨로지 추론이 적용되어 상위 클래스 질의(?p a fp:Product / fp:ETF / fp:Fund)와 역관계(fp:manages, fp:constituentOf, fp:hasSubsidiary)도 가능합니다. 편입종목 데이터는 국내 ETF 대부분(1,162종목)에 대해 네이버금융 기준 상위 종목이 적재되어 있습니다(해외ETF·펀드는 편입종목 없음 → 그 범위는 "확인할 수 없음"으로 명시).
- 배경 설명·위험요인·용어·펀드 동향 → search_documents (출처와 기준일 표기). 상품 마스터와 충돌 시 마스터 우선.
- 해외ETF 검색은 영문(예: 'China semiconductor', 'space', 'robotics')으로. 국내ETF 이름은 한글 약어명(pd_abrv_nm).
- '최근 N개월' 같은 기간은 데이터 기준일(2026-08-21/22)을 기준으로 계산합니다(예: 최근 6개월 = 2026-02-21 이후).
- 펀드는 클래스별 행이 많으므로 fund_family 로 묶어 대표 1개(예: 순자산 최대 클래스)를 제시합니다.

## 값 표기 규칙 (여기서 틀리면 SQL 은 성공해도 답이 틀립니다)
- 금액 단위 환산: 1만=10000, 1억=100000000, 1조=1000000000000. "순자산 1조원 이상" → pd_net_tamt >= 1000000000000.
- 날짜는 'YYYY-MM-DD' 문자열입니다. **연도만 주어지면 LIKE 를 쓰세요**: "만기가 2027년" → mat_date LIKE '2027%'.
  mat_date='2027-12-31' 처럼 등호로 쓰면 그 하루만 걸러집니다.
- 위험등급을 숫자로 비교할 때는 risk_grade(1~6)를 쓰세요. pd_risk_nm 은 '매우높은위험(1등급)' 형태의 라벨이라
  pd_risk_nm='1등급' 은 항상 0건입니다.
- "상장·판매중·구매 가능·현재" 조건은 ETF·펀드는 is_on_sale=1, 채권은 is_sellable=1 입니다. 빠뜨리면 상장폐지 상품까지 세게 됩니다.
  "ETF 몇 개"는 ETN 을 빼기 위해 pd_grp_no='ETF' 도 함께 겁니다.
- **도구가 돌려준 숫자는 고쳐 쓰지 말고 그대로 옮겨 적으세요.** COUNT 결과가 240 이면 답도 240 입니다.

## 교차 질의 (2개 이상 상품군을 함께 비교)
"국내/해외ETF와 공모펀드를 1년 수익률 기준 TOP10" 처럼 여러 상품군을 한 번에 묻는 문항이 출제됩니다.
UNION ALL 로 상품군을 합치고 공통 컬럼 이름을 맞춘 뒤 정렬하세요. 수익률 단위는 세 테이블 모두 %입니다.

SELECT '국내ETF' AS 구분, pd_abrv_nm AS 상품명, pd_itm_no AS 코드, du_er_1y AS 수익률1년, pd_net_tamt AS 순자산
  FROM v_etf_kr WHERE is_on_sale=1 AND pd_grp_no='ETF' AND du_er_1y IS NOT NULL
UNION ALL
SELECT '해외ETF', pd_nm, ticker, NULL, du_last_aum FROM v_etf_gl WHERE is_on_sale=1
UNION ALL
SELECT '공모펀드', itm_nm, itm_no, fd_yr1_ern_r, fd_nast_suma
  FROM v_fund_pub WHERE product_type='공모펀드' AND is_on_sale=1 AND fd_yr1_ern_r IS NOT NULL
ORDER BY 수익률1년 DESC LIMIT 10

주의: 해외ETF에는 기간 수익률이 du_er_1d(1일)만 있고 1년 수익률이 없습니다. 이런 경우 해외ETF는 순위에서 제외하고
"해외ETF는 1년 수익률이 제공되지 않아 비교에서 제외했다"고 답변에 명시하세요. 없는 값을 0으로 채우면 안 됩니다.
특정 종목 보유 조건(예: "삼성전자를 보유한")은 v_etf_constituent 를 조인해 SQL 한 번으로 풉니다(아래 패턴).
편입종목은 국내 ETF에만 있으므로, 해외ETF·펀드까지 묻는 질의라면 그 범위는 확인 불가임을 밝힙니다.

## 자주 쓰는 SQL 패턴
- 구매가능 원화채권 AA- 이상: SELECT DISTINCT pd_no, pd_nm, crd_grd, srfc_irt, mat_date FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-') ORDER BY mat_date
  (매수수익률·세후수익률까지 필요하면 AND has_buy_yield=1 을 추가 — 그 값이 수록된 종목만 남습니다)
- 미국 상장 주식형 해외ETF 중 총보수 낮고 규모 큰 3개: SELECT ticker, pd_nm, expense_ratio_pct, du_last_aum, cu_base_index FROM v_etf_gl WHERE wu_inv_ast_type='Equity' AND is_on_sale=1 AND cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL AND expense_ratio_pct IS NOT NULL ORDER BY expense_ratio_pct ASC, du_last_aum DESC LIMIT 3
- 국내 ETF 순자산 상위: SELECT pd_ticker, pd_abrv_nm, manager_name, pd_net_tamt, pd_risk_nm FROM v_etf_kr WHERE pd_grp_no='ETF' AND is_on_sale=1 AND pd_net_tamt IS NOT NULL ORDER BY pd_net_tamt DESC LIMIT 10
- 펀드 패밀리 요약: SELECT fund_family, MIN(itm_nm) AS name, COUNT(*) AS classes, MAX(fd_nast_suma) AS max_nav, MAX(bmrk_nm), MAX(or_attr_desc) FROM v_fund_pub WHERE product_type='공모펀드' AND itm_nm LIKE '%국민성장%' GROUP BY fund_family
- 특정 종목을 편입한 ETF (순자산 상위 3개): SELECT e.pd_abrv_nm, e.pd_net_tamt FROM v_etf_kr e JOIN v_etf_constituent c ON c.etf_itm_no = e.pd_itm_no WHERE c.company_name='삼성전자' AND c.is_cash=0 AND e.is_on_sale=1 AND e.pd_net_tamt IS NOT NULL ORDER BY e.pd_net_tamt DESC LIMIT 3
  (종목코드를 알면 c.company_code='005930' 이 더 정확합니다. 개수를 물으면 COUNT(DISTINCT e.pd_itm_no).)
"""


# 시스템 프롬프트에 넣을 핵심 컬럼. 전체 컬럼(48~58개)을 매 요청에 실으면 입력이 1.3만 토큰이 되어
# CLOVA TPM 한도(테스트앱 3만/분)를 한 번의 호출로 소진한다. 나머지는 table_info 로 필요할 때만 준다.
CORE_COLUMNS = {
    "v_etf_constituent": [
        "etf_itm_no", "etf_ticker", "etf_name", "company_code", "company_name",
        "weight_pct", "is_cash", "as_of",
    ],
    "v_bond_kr": [
        "pd_no", "pd_nm", "product_type", "pd_exg_mkt", "is_sellable", "has_buy_yield", "issuer_name",
        "bd_knd", "std_pd_mcls_nm", "bd_ofr_tcd", "crd_grd", "pd_risk_nm", "curr_cd", "srfc_irt",
        "isu_date", "mat_date", "remaining_days", "remaining_years", "applied_yield", "buy_yield",
        "after_tax_yield", "dur", "pd_pen_tr_yn", "base_date",
    ],
    "v_etf_kr": [
        "pd_itm_no", "pd_ticker", "pd_abrv_nm", "pd_nm", "product_type", "pd_grp_no", "is_on_sale",
        "manager_name", "cu_strtegy", "cu_lev_fector", "cu_base_index", "cu_charge_rt", "wu_inv_ast_type",
        "wu_inv_rgn", "risk_grade", "pd_risk_nm", "pd_pen_tr_yn", "pd_net_tamt", "du_last_aum", "du_clpr",
        "du_er_1m", "du_er_3m", "du_er_6m", "du_er_ytd", "du_er_1y", "du_vlty_1y", "du_vol_1d",
        "pd_dvid_cycl", "pd_dvid_yield", "listing_date", "delisting_date", "base_date",
    ],
    "v_etf_gl": [
        "pd_itm_no", "ticker", "pd_nm", "product_type", "is_on_sale", "pd_exg_mkt_cd", "pd_mkt_id",
        "pd_curr_cd", "manager_name", "cu_base_index", "cu_strtegy", "expense_ratio_pct", "cu_lev_fector",
        "cu_inverse_short_yn", "wu_inv_ast_type", "wu_inv_rgn", "du_last_aum", "du_clpr", "du_er_1d",
        "du_vol_1d", "listing_date", "base_date",
    ],
    "v_fund_pub": [
        "itm_no", "itm_nm", "fund_family", "product_type", "prvo_pbff_desc", "is_on_sale", "manager_name",
        "or_attr_desc", "zrin_btyp_nm", "zrin_ptn_nm", "fd_ivst_rgn_desc", "ovrs_fd_desc", "bmrk_nm",
        "is_hedged", "risk_grade", "han_clas_fee_type", "han_clas_sales_channel", "fd_nast_suma",
        "fd_mm3_ern_r", "fd_mm6_ern_r", "fd_yr1_ern_r", "fd_yr3_ern_r", "base_date",
    ],
}


def value_hints(rdb, max_values: int = 12) -> dict[tuple[str, str], list[str]]:
    """Actual distinct values of the low-cardinality text columns, for the schema cards.

    The ontology only carries allowed values for columns modelled as enumerations, so text columns
    like ``ovrs_fd_desc`` ("국내/해외 구분") reached the model as a label with no vocabulary — it then
    omitted the filter entirely, or invented one (``pd_risk_nm='1등급'``, which never matches the real
    label '매우높은위험(1등급)').  Reading the values from the store costs a few tokens and removes the
    guessing.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for view, cols in CORE_COLUMNS.items():
        for col in cols:
            try:
                rows = rdb.query(
                    f"SELECT DISTINCT {col} FROM {view} WHERE {col} IS NOT NULL LIMIT {max_values + 1}"
                ).rows
            except Exception:  # noqa: BLE001 - a column we cannot sample simply gets no hint
                continue
            vals = [str(r[col]).strip() for r in rows if r.get(col) is not None]
            vals = [v for v in vals if v]
            if not (1 < len(vals) <= max_values):
                continue
            if any(len(v) > 24 or v.replace(".", "", 1).replace("-", "", 1).isdigit() for v in vals):
                continue  # long free text or plain numbers: listing them helps nobody
            out[(view, col)] = sorted(vals)
    return out


def build_system_prompt(ontology, rdb_meta: dict, doc_titles: list[str] | None = None, settings=None,
                        hints: dict[tuple[str, str], list[str]] | None = None) -> str:
    budget = getattr(settings, "time_budget_sec", 120)
    rounds = getattr(settings, "max_tool_rounds", 8)
    parts = [
        RULES.strip(),
        f"\n## 예산\n한 문항당 약 {budget:.0f}초, 도구 호출 최대 {rounds}라운드입니다. 300초를 넘기면 미응답 처리되므로 반드시 그 안에 final_answer 를 호출합니다.",
        "\n## 데이터 테이블(뷰) 개요",
    ]
    for view in VIEW_ID_NAME:
        m = rdb_meta.get(view) or {}
        parts.append(
            f"- {view}: {m.get('korean_name', '')} · 원천 {m.get('source_id', '')} · 기준일 {m.get('base_date', '')} · "
            f"{m.get('row_count', '?')}행 — {SEMANTICS[view]}"
        )
    if ontology is not None:
        parts.append(
            "\n## 스키마 요약(자주 쓰는 컬럼만 · 컬럼: 한글 의미 (단위; 허용값))\n"
            "여기 없는 컬럼이 필요하면 table_info 도구로 해당 뷰의 전체 정의를 받으세요."
        )
        for view in VIEW_ID_NAME:
            card = ontology.schema_card(view, columns=CORE_COLUMNS.get(view), compact=True)
            if hints:
                lines = []
                for ln in card.splitlines():
                    col = ln[2:].split(":", 1)[0] if ln.startswith("- ") else None
                    vals = hints.get((view, col)) if col else None
                    if vals and "값:" not in ln:
                        ln = f"{ln} (값:{'/'.join(vals)})" if "(" not in ln else f"{ln[:-1]}; 값:{'/'.join(vals)})"
                    lines.append(ln)
                card = "\n".join(lines)
            parts.append(card)
    if doc_titles:
        parts.append("\n## 외부 참고 문서\n- " + "\n- ".join(doc_titles))
    return "\n".join(parts)
