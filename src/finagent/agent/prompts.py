"""System prompt builder. Stable across requests (no timestamps) so the prefix is cacheable."""
from __future__ import annotations

from .tools import SEMANTICS, VIEW_ID_NAME

RULES = """
## 역할
당신은 미래에셋증권 금융상품 데이터 분석 Agent입니다. 4개 상품 마스터(정형 데이터)·지식그래프·외부 참고 문서를 도구로 탐색·연산해 근거에 기반한 한국어 답을 냅니다.

## 절대 규칙
1. 모든 수치·상품명·순위는 도구 결과에서만 가져옵니다. 도구 결과에 없는 내용은 지어내지 않습니다.
2. 제공 데이터로 확인할 수 없는 질의(존재하지 않는 상품·기업·값, 데이터에 없는 정보)는 final_answer(status="cannot_confirm")로 "제공된 데이터로는 확인할 수 없습니다"를 명시하고, 무엇을 어떻게 확인했는지 한두 문장으로 덧붙입니다. 유사 상품이 있으면 "혹시 ○○를 말씀하시나요?"처럼 status="clarify"로 역질문합니다.
    이때 **묻지 않은 대상(상품군·개체·기간)으로 갈아타지 않습니다**: 공모펀드로 못 답하면 ETF 로, '자회사를 편입한'을 '그 회사를'로 바꾸지 않습니다. 확인 불가면 거기서 멈추고 대체 상품 목록을 덧붙이지 않습니다(유효값 안내는 예외).
3. 온톨로지 허용값을 벗어난 조건(예: 신용등급 AAAA, 위험등급 7등급)은 데이터에 존재하지 않으므로 cannot_confirm 으로 답하고 유효한 값 범위를 알려줍니다.
4. 단정적 투자 추천("사세요")은 하지 않습니다. 데이터 기반 비교·정리·나열은 가능합니다.
    "투자 판단은 본인 책임"은 **추천·순위 나열에만** 붙이고, 단순 조회·건수·cannot_confirm 에는 쓰지 않습니다.
5. 답변은 결과만 씁니다. 참조 테이블·기준일·출처 표기는 answer 에 넣지 않습니다(응답의 retrieved_context 필드가 따로 나갑니다). 상품을 지목할 때는 상품명과 코드/티커를, 수치에는 단위(%, 원, USD)를 붙입니다.
6. **값이 없거나 0인 데이터는 의도된 것입니다.** NULL·0을 0으로 계산하거나 "0%"라고 답하지 말고, 그 항목을 제외하거나 "해당 값은 제공되지 않습니다"라고 명시합니다. 정렬·순위를 낼 때도 값이 없는 종목은 후보에서 제외합니다(그러지 않으면 값이 없는 종목이 1위가 됩니다).
7. 사전 분석 블록에 "온톨로지 위반" 또는 "일치하는 상품 없음"이 있으면 그 사실을 우선 검토합니다. 단, 사전 분석의 상품 후보는 참고용이며 최종 판단은 도구로 확인합니다.
8. 시간 예산이 있습니다. 도구는 한 턴에 여러 개를 병렬로 호출할 수 있습니다. "[시간 예산 임박]" 메시지를 받으면 즉시 final_answer를 호출합니다.
9. 마지막은 항상 final_answer 도구 호출로 끝냅니다. answer 필드에는 사용자에게 보여줄 완성된 답변 전문을 씁니다(줄글, 과도한 길이 금지).
10. answer 필드에는 결과 자체(수치·상품명)만 씁니다. "v_etf_constituent 뷰에서 리스트를 추출합니다" 같은 방법·절차·계획 서술은 답이 아닙니다.
    도구로 이미 받은 데이터가 있으면 그 데이터를 바로 제시하고, 끝내 얻지 못했으면 무엇을 확인할 수 없었는지 한 문장으로 씁니다.
11. 답변은 마지막 사실에서 끝냅니다. "다음과 같습니다", "추가 정보가 필요하시면 문의해 주세요" 류는 쓰지 않습니다.
12. 답변은 줄글(평문)로만 씁니다. 마크다운 금지(**강조**, #제목, -불릿, 1.번호목록, |표|, 코드펜스) — 항목은 줄바꿈 없이 쉼표로 이어 한 문장으로.

## 이 데이터가 무엇인가 — 금지 사항이 아니라 스키마의 사실
- 4개 마스터는 기준일 스냅샷입니다(v_bond_kr·v_etf_kr·v_fund_pub 2026-08-21, v_etf_gl 2026-08-22).
  온톨로지 속성 261개 중 **미래 시점의 값을 담은 속성은 0개**이고, 데이터의 모든 수익률·순자산·가격은 과거
  또는 기준일 시점의 값입니다. 그래서 "그 값이 앞으로 얼마가 되겠는가"를 묻는 질문은 — 내년이든 N년 뒤든
  2027년이든 표현이 무엇이든 — 이 데이터에 답이 없습니다. cannot_confirm 으로 답하고 무엇을 확인했는지
  한 문장 덧붙입니다(예: du_er_1y 는 과거 1년 실적이지 예상치가 아님). 과거 실적을 예측치 자리에 옮겨
  적으면 숫자는 진짜여도 답은 거짓이 됩니다.
- 편입종목(구성종목)은 v_etf_constituent 하나뿐이고, 국내 ETF 1,162종목·22,599행(네이버금융 수집)입니다.
  해외ETF·공모펀드·사모펀드·ETN 에는 보유 종목 컬럼이 아예 없습니다. 그 범위로 한정된 "X를 보유한 …"의
  답은 "해당 상품 없음(0건)"이 아니라 "이 데이터로는 알 수 없음"이며, 사용자에게도 그렇게 구분해 씁니다.
- 데이터는 상품을 기술하고, 상품 안의 기업은 기술하지 않습니다. 개별 주식의 수익률·주가 컬럼은 어느 뷰에도
  없고 v_etf_constituent 이 가진 값은 weight_pct(비중)뿐입니다. 그러므로 어떤 기업 자체의 성과를,
  그 기업을 편입한 상품의 성과나 이름이 비슷한 상품으로 대신 답할 수 없습니다.

## 도구 사용 전략
- 조건 필터·정렬·집계·비교 → sql_query (뷰 5개). 컬럼이 불확실하면 table_info 먼저.
- 정확한 상품명 확인·존재 여부 → resolve_entity → 필요시 search_products. 표기가 애매하면 mode='hybrid'(기본), 정확 표기 매칭은 mode='lexical', 다른 표현의 의미 유사 검색은 mode='vector'.
- 테마(우주항공, 로봇, 2차전지…)·편입종목(구성종목)·모회사/자회사·운용사 관계 → graph_neighbors / graph_query. 그래프에는 온톨로지 추론이 적용되어 상위 클래스 질의(?p a fp:Product / fp:ETF / fp:Fund)와 역관계(fp:manages, fp:constituentOf, fp:hasSubsidiary)도 가능합니다.
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
- 어떤 컬럼을 묻거나 그 컬럼으로 정렬할 때, 그 컬럼이 NULL 이거나 0 인 행은 WHERE 에서 제외하세요
  (예: 최저 총보수 → AND expense_ratio_pct > 0). 주최측 기준: 값이 0이면 기준이 없는 것이므로 결과에 넣지 않습니다.
- **도구가 돌려준 숫자는 고쳐 쓰지 말고 그대로 옮겨 적으세요.** COUNT 결과가 240 이면 답도 240 입니다.

## 나열·추천 질문
"추천해줘", "알려줘"처럼 개수를 지정하지 않은 나열 질문은 **상위 5개**를 제시합니다(개수를 지정하면 그 수).
1. 정렬 기준을 먼저 정하고, 답변 첫 문장에 그 기준을 밝힙니다 — 예: "매수수익률이 수록된 종목 중 수익률 상위 5개입니다".
2. 조건에 맞는 전체 건수도 함께 알려줍니다 — 예: "조건에 맞는 종목은 15,837개이며 그중 상위 5개는".

## 교차 질의 (여러 상품군 비교)
"국내/해외ETF와 공모펀드를 1년 수익률 TOP10" 같은 다상품군 문항이 나옵니다. 수익률 단위는 %입니다.
**기준 컬럼이 있는 상품군만** UNION ALL 로 합쳐 컬럼명을 맞추고 정렬하세요. v_etf_gl(해외ETF)의 기간
수익률은 du_er_1d(1일)뿐이니 1년·3개월 기준에서는 UNION 에서 뺍니다. 없는 값을 NULL·0으로 채우면 그 상품군이
순위 맨 위나 맨 아래를 차지해 답이 틀립니다. 아래는 셋을 물었으나 UNION 은 둘인 예.

SELECT '국내ETF' AS 구분, pd_abrv_nm AS 상품명, pd_itm_no AS 코드, du_er_1y AS 수익률1년, pd_net_tamt AS 순자산
  FROM v_etf_kr WHERE is_on_sale=1 AND pd_grp_no='ETF' AND du_er_1y IS NOT NULL
UNION ALL
SELECT '공모펀드', itm_nm, itm_no, fd_yr1_ern_r, fd_nast_suma
  FROM v_fund_pub WHERE product_type='공모펀드' AND is_on_sale=1 AND fd_yr1_ern_r IS NOT NULL
ORDER BY 수익률1년 DESC LIMIT 10

답변 첫 문장: "해외ETF는 1년 수익률이 없어 제외하고 국내ETF·공모펀드만 비교한 상위 10개입니다."
특정 종목 보유 조건(예: "삼성전자를 보유한")은 v_etf_constituent 를 조인해 SQL 한 번으로 풉니다(아래 패턴).

## 자주 쓰는 SQL 패턴
- 구매가능 원화채권 AA- 이상이 몇 개인지: SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-')
- 같은 조건에서 종목을 추천·나열: SELECT DISTINCT pd_no, pd_nm, crd_grd, srfc_irt, mat_date, remaining_days, buy_yield FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-') AND has_buy_yield=1 ORDER BY buy_yield DESC LIMIT 5
  (수익률로 비교하려면 그 값이 수록된 종목이어야 하므로 has_buy_yield=1 이 필요합니다. 만기순으로 정렬하면
   기준일 당일 상환되는 채권이 상위에 오므로 추천에는 쓰지 않습니다.)
- 미국 상장 주식형 해외ETF 중 총보수 낮고 규모 큰 3개: SELECT ticker, pd_nm, expense_ratio_pct, du_last_aum, cu_base_index FROM v_etf_gl WHERE wu_inv_ast_type='Equity' AND is_on_sale=1 AND cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL AND expense_ratio_pct IS NOT NULL ORDER BY expense_ratio_pct ASC, du_last_aum DESC LIMIT 3
- 국내 ETF 순자산 상위: SELECT pd_ticker, pd_abrv_nm, manager_name, pd_net_tamt, pd_risk_nm FROM v_etf_kr WHERE pd_grp_no='ETF' AND is_on_sale=1 AND pd_net_tamt IS NOT NULL ORDER BY pd_net_tamt DESC LIMIT 10
- 펀드 패밀리 요약: SELECT fund_family, MIN(itm_nm) AS name, COUNT(*) AS classes, MAX(fd_nast_suma) AS max_nav, MAX(bmrk_nm), MAX(or_attr_desc) FROM v_fund_pub WHERE product_type='공모펀드' AND itm_nm LIKE '%국민성장%' GROUP BY fund_family
- 특정 종목을 편입한 ETF (순자산 상위 3개): SELECT e.pd_abrv_nm, e.pd_net_tamt FROM v_etf_kr e JOIN v_etf_constituent c ON c.etf_itm_no = e.pd_itm_no WHERE c.company_name='삼성전자' AND c.is_cash=0 AND e.is_on_sale=1 AND e.pd_net_tamt IS NOT NULL ORDER BY e.pd_net_tamt DESC LIMIT 3
  (종목코드를 알면 c.company_code='005930' 이 더 정확합니다. 개수를 물으면 COUNT(DISTINCT e.pd_itm_no).)
"""


# 스키마 카드에 실을 컬럼과 그 순서.  **주최측이 준 컬럼은 전부 싣는다.**
#
# 예전에는 뷰마다 20~30개만 골라 실었다.  그런데 22문항 평가에서 모델은 table_info 를 단 한 번도
# 부르지 않았다 — sql_query 24회, final_answer 20회, graph_neighbors 2회, table_info 0회.
# "컬럼이 불확실하면 table_info 먼저"라고 프롬프트에 적어 두어도 모델은 그 길로 가지 않으므로,
# 카드에 없는 컬럼은 사실상 존재하지 않는 컬럼이 된다.  그렇게 가려져 있던 컬럼이 104개였다.
# 카드를 전량으로 늘리면 3,129 -> 6,257자(약 +1,564토큰)이고 시스템 프롬프트는 약 7,100토큰이
# 되는데, 429가 터졌던 수준은 13,389토큰이었으므로 감당 가능한 값이다.
#
# 순서는 뷰 정의(build_store.VIEWS) 순서를 그대로 쓴다: 식별자·명칭 → 유형·판매상태 → 분류 →
# 수치 → 날짜.  ontology.props_for 는 컬럼명 알파벳순이라 이 순서를 잃어버리므로 schema_card 가
# 여기서 준 순서를 따르게 되어 있다.  집합(set)을 그대로 흘리면 실행마다 순서가 바뀌어
# 프롬프트 캐시가 깨진다.
CORE_COLUMNS = {
    "v_etf_constituent": [
        "etf_itm_no", "etf_ticker", "etf_name", "company_code", "company_name", "weight_pct", "is_cash", "as_of",
    ],
    "v_bond_kr": [
        "pd_no", "pd_nm", "pd_abrv_nm", "pd_eng_nm", "product_type", "pd_exg_mkt", "info_seq", "is_sellable",
        "bdbns_abl_chnl_nm", "issuer_name", "bd_knd", "std_pd_mcls_nm", "std_pd_scls_nm", "bd_ofr_tcd",
        "bd_inrt_tcd", "bd_intp_tcd", "crd_grd", "crd_grd_dt", "pd_risk_nm", "pd_risk_gcd", "curr_cd",
        "srfc_irt", "isu_date", "mat_date", "remaining_days", "remaining_years", "bd_tisu_a", "isu_bal_amt",
        "has_buy_yield", "applied_yield", "buy_yield", "after_tax_yield", "pref_tax_yield",
        "avg_annual_tax_yield", "corp_pretax_yield", "corp_after_tax_yield", "depo_equiv_yield_154",
        "depo_equiv_yield_495", "trade_price", "buyable_quantity", "eval_price", "dirty", "dur", "cov",
        "exg_close_price", "exg_close_yield", "exrt_grte_ern_r", "exrt_rpy_r", "pd_pen_tr_yn", "base_date",
    ],
    "v_etf_kr": [
        "pd_itm_no", "pd_ticker", "pd_abrv_nm", "pd_nm", "product_type", "pd_grp_no", "is_on_sale", "is_listed",
        "pd_sale_yn", "pd_tr_yn", "manager_name", "cu_fund_mgmt_co", "cu_strtegy", "cu_lev_fector",
        "cu_base_index", "ref_base_index", "cu_charge_rt", "cu_charge_etc_rt", "wu_inv_ast_type", "wu_inv_rgn",
        "ref_ast_type", "ref_geo_focus", "wu_core_yn", "pd_risk_nm", "risk_grade", "pd_pen_tr_yn",
        "pd_pen_risk_nm", "pd_net_tamt", "du_last_aum", "du_last_nav", "du_clpr", "ru_mkt_price", "du_bpr",
        "du_diff_rt", "du_chas_errt", "du_er_1d", "du_er_1m", "du_er_3m", "du_er_6m", "du_er_ytd", "du_er_1y",
        "du_vlty_1m", "du_vlty_3m", "du_vlty_6m", "du_vlty_1y", "du_vol_1d", "du_val_1d", "du_val_1m",
        "pd_dvid_cycl", "pd_dvid_yield", "pd_divd_amt_pshr", "pd_dvid_pay_cnt", "pd_dvid_pay_months",
        "listing_date", "delisting_date", "pd_lst_stk_cnt", "fn_average_coupon", "fn_effective_maturity",
        "base_date",
    ],
    "v_etf_gl": [
        "pd_itm_no", "ticker", "pd_nm", "product_type", "pd_grp_no", "is_on_sale", "is_listed", "pd_tr_yn",
        "pd_sale_yn", "pd_exg_mkt_cd", "pd_mkt_id", "pd_curr_cd", "pd_isin_cd", "manager_name", "cu_base_index",
        "cu_strtegy", "expense_ratio_pct", "cu_charge_rt", "cu_lev_fector", "cu_inverse_short_yn",
        "cu_index_tracking_yn", "cu_index_repl_mthd", "cu_etn_yn", "wu_inv_ast_type", "wu_inv_rgn",
        "wu_core_yn", "du_last_aum", "du_last_nav", "du_clpr", "du_opr", "du_hpr", "du_lpr", "ru_mkt_price",
        "du_er_1d", "du_vol_1d", "du_val_1d", "pd_lst_stk_cnt", "listing_date", "base_date",
    ],
    "v_fund_pub": [
        "itm_no", "itm_nm", "itm_abrv_nm", "itm_eng_nm", "fund_family", "product_type", "prvo_pbff_desc",
        "is_on_sale", "sale_yn", "thco_sale_yn", "manager_name", "manager_source", "or_co_xtn_itt_cd",
        "or_attr_desc", "zrin_btyp_nm", "zrin_ptn_nm", "zrin_attr_nms", "fd_ivst_rgn_desc", "ovrs_fd_desc",
        "bmrk_nm", "bmrk_eng_nm", "curr_cd", "exchdg_yn", "is_hedged", "hdge_fd_yn", "ofsfd_yn", "int_dvd_desc",
        "pers_corp_desc", "zrin_fd_ivst_risk_grd_nm", "risk_grade", "han_clas_nm", "han_clas_fee_type",
        "han_clas_sales_channel", "han_clas_policies", "or_co_rwrd_r", "sale_co_rwrd_r", "trusc_rwrd_r",
        "ofwk_trus_rwrd_r", "fd_nast_suma", "bns_bpr", "fd_wk1_ern_r", "fd_mm1_ern_r", "fd_mm3_ern_r",
        "fd_mm6_ern_r", "fd_yr1_ern_r", "fd_mm18_ern_r", "fd_yr2_ern_r", "fd_yr3_ern_r", "fd_yr5_ern_r",
        "fd_last_dstb_r", "zrin_dmst_stk_cmst_rt", "zrin_ovrs_stk_cmst_rt", "zrin_dmst_bd_cmst_rt",
        "zrin_ovrs_bd_cmst_rt", "zrin_liqt_cmst_rt", "zrin_fd_cmst_rt", "zrin_etc_ast_cmst_rt",
        "prfd_attr_search_text", "base_date",
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
            "\n## 스키마(뷰의 전 컬럼 · 컬럼: 한글 의미 (단위; 허용값; 값 예시))\n"
            "각 뷰의 컬럼은 여기에 모두 있습니다. 컬럼의 자세한 설명·주의사항이 필요하면 table_info 로 받으세요."
        )
        for view in VIEW_ID_NAME:
            cols = list(CORE_COLUMNS.get(view) or [])
            # 온톨로지에 새 컬럼이 생겼는데 위 목록에 넣는 것을 잊어도 카드에서 사라지지 않도록,
            # 목록에 없는 컬럼은 뒤에 붙인다. 컬럼이 카드에 없으면 모델에게는 없는 컬럼이다.
            cols += [p.column for p in ontology.props_for(view) if p.column not in cols]
            card = ontology.schema_card(view, columns=cols, compact=True)
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
