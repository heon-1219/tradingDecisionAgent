"""Tool registry, orchestrator and API contract tests with a scripted stub LLM on a tiny store."""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finagent.agent.answer import CANNOT_PREFIX, Evidence, format_evidence, normalize_answer
from finagent.agent.llm import StubLLM, ToolCall, Turn
from finagent.agent.orchestrator import Orchestrator, _call_key, asks_forward_looking
from finagent.agent.prompts import build_system_prompt
from finagent.agent.tools import ToolRegistry
from finagent.api import create_app
from finagent.config import Settings
from finagent.kb.docs import DocStore
from finagent.kb.entity import EntityResolver
from finagent.kb.graph import KG
from finagent.kb.lexicon import Lexicon
from finagent.kb.rdb import RDB
from finagent.kb.search import SearchIndex

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_store(tmp_path_factory):
    p = tmp_path_factory.mktemp("s") / "t.db"
    c = sqlite3.connect(p)
    c.execute(
        "create table bond_kr(pd_no, pd_nm, pd_abrv_nm, pd_eng_nm, product_type, pd_exg_mkt, is_sellable, issuer_name, bd_knd, "
        "std_pd_mcls_nm, std_pd_scls_nm, bd_ofr_tcd, bd_inrt_tcd, bd_intp_tcd, crd_grd, pd_risk_gcd, curr_cd, buy_yield, "
        "after_tax_yield, srfc_irt, mat_date, base_date, pd_risk_nm)"
    )
    c.executemany(
        "insert into bond_kr values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("KR1", "한국전력공사채권1012", "한전1012", "KEPCO", "국내채권", "장외", 1, "한국전력공사", "일반특수법인채", "특수채", "공사채", "공모", "고정금리", "이표채", "AAA", "16", "KRW", 4.136, 3.95, 3.1, "2029-12-19", "2026-08-21", "매우낮은위험(6등급)"),
            ("KR2", "CJ대한통운98-3", "CJ98-3", "CJ", "국내채권", "장외", 1, "씨제이대한통운", "일반회사채", "회사채", "일반사채", "공모", "변동금리", "이표채", "AA-", "15", "KRW", 3.99, 3.74, 3.0, "2028-02-04", "2026-08-21", "낮은위험(5등급)"),
            ("KR3", "국고채권01125", "국고", "KTB", "국내채권", "장내", 0, "대한민국", "국고채권", "국공채", "국고채", "공모", "고정금리", "복리채", None, "16", "KRW", None, None, 1.125, "2039-09-10", "2026-08-21", "매우낮은위험(6등급)"),
        ],
    )
    c.execute("create view v_bond_kr as select * from bond_kr")
    c.execute(
        "create table etf_kr(pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, is_on_sale, base_date, manager_name, cu_base_index, "
        "ref_base_index, wu_inv_ast_type, wu_inv_rgn, risk_grade, cu_strtegy, cu_lev_fector, listing_date, pd_dvid_cycl, pd_net_tamt, pd_grp_no, cu_charge_rt, pd_risk_nm)"
    )
    c.executemany(
        "insert into etf_kr values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("KRE1", "305720", "KODEX 2차전지산업", "삼성 KODEX 2차전지산업", "국내ETF", 1, "2026-08-21", "삼성자산운용", "", "KRX 2차전지", "주식", "국내", 1, "실물복제", 1.0, "2018-09-11", "A", 1.0e12, "ETF", None, "매우높은위험(1등급)"),
            ("KRE2", "445290", "KODEX 로봇액티브", "삼성 KODEX 로봇액티브", "국내ETF", 1, "2026-08-21", "삼성자산운용", "", "", "주식", "국내", 1, "액티브", 1.0, "2022-11-15", "A", 5.0e10, "ETF", 0.5, "매우높은위험(1등급)"),
        ],
    )
    c.execute("create view v_etf_kr as select * from etf_kr")
    c.execute(
        "create table etf_constituent(etf_itm_no, etf_ticker, etf_name, company_code, company_name, "
        "company_id, weight_pct, is_cash, as_of)"
    )
    c.execute(
        "insert into etf_constituent values ('KRE1','305720','KODEX 2차전지산업','247540','에코프로비엠',"
        "'ecopro_bm',7.0,0,'2026-08-21')"
    )
    c.execute("create view v_etf_constituent as select * from etf_constituent")
    c.execute(
        "create view v_etf_gl as select null pd_itm_no, null ticker, null pd_nm, null product_type, null is_on_sale, null base_date, null manager_name, "
        "null cu_base_index, null wu_inv_ast_type, null wu_inv_rgn, null expense_ratio_pct, null du_last_aum, null cu_lev_fector, null cu_inverse_short_yn, "
        "null listing_date, null cu_strtegy, null pd_exg_mkt_cd, null pd_curr_cd where 0"
    )
    c.execute(
        "create view v_fund_pub as select null itm_no, null itm_nm, null itm_abrv_nm, null product_type, null is_on_sale, null base_date, null manager_name, "
        "null fund_family, null bmrk_nm, null or_attr_desc, null zrin_ptn_nm, null fd_ivst_rgn_desc, null risk_grade, null is_hedged, null fd_nast_suma, "
        "null zrin_btyp_nm, null zrin_fd_ivst_risk_grd_nm, null ovrs_fd_desc, null sale_yn, null int_dvd_desc, null curr_cd where 0"
    )
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.executemany("insert into meta values (?,?,?,?,?,?)", [("bond_kr", "PRBD01N001", "국내채권마스터", "2026-08-21", 3, "x"), ("etf_kr", "PREF01N001", "국내ETF마스터", "2026-08-21", 2, "x")])
    c.commit()
    c.close()
    return p


def _state(tiny_store, llm, tmp_path):
    settings = Settings(_env_file=None, llm_provider="stub", time_budget_sec=20, max_tool_rounds=4)
    rdb = RDB(tiny_store)
    lx = Lexicon.load(ROOT / "data/enrichment")
    idx = SearchIndex.from_rdb(rdb)
    resolver = EntityResolver(idx, lx)
    kg = KG.build(rdb, lx, None, log=lambda *a: None)
    docs = DocStore.load(ROOT / "data/external/docs")
    from finagent.ontology.loader import FallbackValidators, Ontology

    try:
        ontology = Ontology.load(ROOT / "ontology")
    except Exception:
        ontology = FallbackValidators()
    tools = ToolRegistry(rdb, idx, resolver, kg, docs, ontology, lx, settings)
    orch = Orchestrator(settings, llm, tools, resolver, ontology, lx, build_system_prompt(ontology, rdb.meta(), docs.titles(), settings))

    class S:  # minimal AppState stand-in
        pass

    s = S()
    s.rdb, s.index, s.kg, s.docs, s.llm, s.settings, s.orchestrator, s.tools = rdb, idx, kg, docs, llm, settings, orch, tools
    return s


async def test_tools_sql_and_guard(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("sql_query", {"sql": "select pd_no, pd_nm, crd_grd from v_bond_kr where is_sellable=1 and crd_grd in ('AAA','AA+','AA0','AA-')"})
    assert not r.is_error and r.evidence.table == "v_bond_kr" and r.evidence.base_date == "2026-08-21" and r.evidence.row_count == 2
    assert "KR1 한국전력공사채권1012" in r.evidence.refs
    bad = await s.tools.execute("sql_query", {"sql": "delete from bond_kr"})
    assert bad.is_error
    g = await s.tools.execute("graph_neighbors", {"entity": "에코프로비엠", "relation": "constituentOf"})
    assert "KODEX 2차전지산업" in g.content
    e = await s.tools.execute("resolve_entity", {"mention": "에코프로 자회사"})
    assert "에코프로비엠" in e.content
    d = await s.tools.execute("search_documents", {"query": "국민성장펀드 구조"})
    assert "국민성장펀드" in d.content
    unknown = await s.tools.execute("nope", {})
    assert unknown.is_error


async def test_orchestrator_answered_with_evidence(tiny_store, tmp_path):
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": "select pd_no, pd_nm, crd_grd, buy_yield from v_bond_kr where is_sellable=1 and crd_grd in ('AAA','AA+','AA0','AA-')", "purpose": "AA- 이상 매수가능"})], "tool_use"),
        Turn("", [ToolCall("c2", "final_answer", {"status": "answered", "answer": "AA- 이상 매수가능 채권은 2종입니다: 한국전력공사채권1012(AAA, 4.136%), CJ대한통운98-3(AA-, 3.99%). (근거: 국내채권마스터, 2026-08-21)", "evidence_refs": ["v_bond_kr KR1", "v_bond_kr KR2"]})], "tool_use"),
    ]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("Q-001", "현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘")
    assert resp.question_id == "Q-001" and "[근거 1]" in resp.retrieved_context and "v_bond_kr" in resp.retrieved_context
    assert "final_answer(status=answered)" in resp.think_trace and "2종" in resp.answer
    assert "'AA-' 유효" in s.llm.started[0][1]  # pre-analysis injected credit-grade hint


async def test_orchestrator_invalid_grade_forces_cannot_confirm(tiny_store, tmp_path):
    script = [Turn("", [ToolCall("c1", "final_answer", {"status": "answered", "answer": "AAAA 채권 3종이 있습니다."})], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("Q-X", "신용등급 AAAA인 채권 찾아줘")
    assert resp.answer.startswith("제공된 데이터로는 확인할 수 없습니다")
    assert "온톨로지 위반" in s.llm.started[0][1] or "AAAA" in s.llm.started[0][1]


async def test_orchestrator_llm_error_falls_back(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([Turn("boom", [], "error")]), tmp_path)
    resp = await s.orchestrator.answer("Q-E", "아무 질문")
    assert "확인할 수 없습니다" in resp.answer and "LLM 오류" in resp.think_trace


async def test_orchestrator_text_only_then_nudge(tiny_store, tmp_path):
    script = [Turn("그냥 텍스트", [], "end_turn"), Turn("", [ToolCall("c", "final_answer", {"status": "clarify", "answer": "어떤 채권 종류를 말씀하시나요"})], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("Q-C", "채권 알려줘")
    assert resp.answer.endswith("?") and "clarify" in resp.think_trace


def test_api_contract(tiny_store, tmp_path):
    llm = StubLLM(script_factory=lambda q: [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": f"답변: {q[:10]}"})], "tool_use")])
    app = create_app(lambda: _state(tiny_store, llm, tmp_path))
    with TestClient(app) as client:
        r = client.get("/answer", params={"question_id": "Q-001", "question": "현재 판매가능한 원화채권", "foo": "bar"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert set(body) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
        assert body["question_id"] == "Q-001" and body["answer"].startswith("답변")
        r2 = client.get("/answer", params={"question_id": "Q-002"})
        assert r2.status_code == 200 and "question 파라미터" in r2.json()["answer"]
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["kg_triples"] > 0


def test_api_never_500_on_orchestrator_crash(tiny_store, tmp_path):
    class Boom:
        async def answer(self, qid, q):
            raise RuntimeError("kaboom")

    def factory():
        s = _state(tiny_store, StubLLM([]), tmp_path)
        s.orchestrator = Boom()
        return s

    with TestClient(create_app(factory)) as client:
        r = client.get("/answer", params={"question_id": "Q", "question": "x"})
        assert r.status_code == 200 and "RuntimeError" in r.json()["think_trace"] and r.json()["answer"]


def test_answer_helpers():
    ev = Evidence("sql", "v_bond_kr", "국내채권마스터", "2026-08-21", "SELECT 1", 2, ["KR1 a", "KR2 b"])
    txt = format_evidence([ev], ["v_bond_kr"])
    assert txt.startswith("[근거 1] 국내채권마스터(v_bond_kr) · 기준일 2026-08-21 · 2건") and "SELECT 1" in txt
    assert normalize_answer("cannot_confirm", "해당 상품 없음").startswith("제공된 데이터로는 확인할 수 없습니다")
    assert normalize_answer("clarify", "어느 것").endswith("?")
    assert json.loads('{"a": 1}')["a"] == 1


async def test_nonexistent_product_is_not_answered_as_a_similar_one(tiny_store, tmp_path):
    """'KODEX AI로봇' 은 존재하지 않는다. 모델이 KODEX 로봇액티브로 바꿔 답하면 서버가 역질문으로 보정한다."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "KODEX AI로봇의 순자산은 500억원이며 삼성자산운용이 운용합니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    pre = s.orchestrator.pre_analyze("KODEX AI로봇 ETF 정보 알려줘")
    assert pre.unresolved and pre.unresolved[0][0] == "KODEX AI로봇"
    assert "존재하지 않습니다" in pre.text
    resp = await s.orchestrator.answer("U-003", "KODEX AI로봇 ETF 정보 알려줘")
    assert "확인할 수 없습니다" in resp.answer
    assert "KODEX 로봇액티브" in resp.answer  # 후보를 제시하며 되물음
    assert "역질문으로 보정" in resp.think_trace


async def test_real_product_is_not_falsely_flagged(tiny_store, tmp_path):
    """운용사 접두어나 '펀드' 접미사 차이는 미존재로 보면 안 된다(오탐 방지)."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    assert s.orchestrator.pre_analyze("KODEX 로봇액티브 순자산 알려줘").unresolved == []
    assert s.orchestrator.pre_analyze("삼성 KODEX 2차전지산업의 위험등급은?").unresolved == []


async def test_acknowledged_decline_is_left_alone(tiny_store, tmp_path):
    """모델이 이미 '없다'고 답했으면 서버가 덧붙이지 않는다."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "cannot_confirm",
        "answer": "KODEX AI로봇이라는 상품은 확인할 수 없습니다. 혹시 KODEX 로봇액티브를 말씀하시나요?",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("U-003", "KODEX AI로봇 ETF 정보 알려줘")
    assert resp.answer.count("확인할 수 없습니다") == 1
    assert "역질문으로 보정" not in resp.think_trace


def test_forward_looking_detector_covers_the_class_not_one_phrasing():
    """온톨로지에 미래 값 컬럼이 없다는 사실에 근거한 판정. 데이터에 있는 값은 건드리면 안 된다."""
    for q in ("TIGER 반도체 ETF의 내년 예상 수익률", "이 펀드 앞으로 오를까요?",
              "향후 주가 전망 알려줘", "내년에 수익률이 얼마나 될까"):
        assert asks_forward_looking(q), q
    # 만기보장수익률(exrt_grte_ern_r)은 실제로 데이터에 있는 값이므로 예측 질문이 아니다
    for q in ("만기보장수익률이 가장 높은 채권", "보장수익률 알려줘",
              "최근 1년 수익률 상위 3개", "작년 수익률", "만기가 2027년인 채권",
              # 운용사·상품 이름의 '미래'는 미래 시점이 아니다(상품명 9,816건이 '미래'를 포함한다)
              "미래에셋 ETF의 1년 수익률 알려줘", "BNK 미래전략기술액티브의 수익률은?",
              "작년 연말 수익률 알려줘"):
        assert not asks_forward_looking(q), q


async def test_past_return_is_not_served_as_a_forecast(tiny_store, tmp_path):
    """G-019 재발 방지: 과거 1년 수익률(du_er_1y)을 '내년 예상 수익률'로 제시하면 안 된다.

    숫자 자체는 도구가 낸 진짜 값이라 숫자 교차검증은 통과한다. 틀린 것은 라벨이므로
    서버가 상태를 확인불가로 내리고 과거 실적임을 명시한 채로 남긴다.
    """
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "KODEX 로봇액티브의 내년 예상 수익률은 260.41%입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("F-001", "KODEX 로봇액티브의 내년 예상 수익률을 알려줘")
    assert resp.answer.startswith(CANNOT_PREFIX)
    assert "미래에 실현될 값을 담은 항목이 없습니다" in resp.answer
    assert "보장하지 않으므로" in resp.answer
    assert "(참고)" in resp.answer and "260.41%" in resp.answer  # 조회 결과는 과거 실적으로 남긴다
    assert "확인불가로 보정" in resp.think_trace


async def test_guaranteed_yield_question_is_still_answered(tiny_store, tmp_path):
    """만기보장수익률은 데이터에 있는 컬럼이다 — 예측 가드가 이 질문을 막으면 안 된다."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "만기보장수익률이 가장 높은 채권은 한국전력공사채권1012입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("F-002", "만기보장수익률이 가장 높은 채권 알려줘")
    assert resp.answer.startswith("만기보장수익률이 가장 높은 채권은")
    assert CANNOT_PREFIX not in resp.answer
    assert "확인불가로 보정" not in resp.think_trace


async def test_count_star_on_bonds_reports_product_count(tiny_store, tmp_path):
    """'몇 개'를 COUNT(*) 로 물으면 행 수가 나온다. 실제 종목 수를 함께 알려줘야 한다."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("sql_query", {"sql": "SELECT COUNT(*) FROM v_bond_kr WHERE is_sellable=1"})
    assert not r.is_error
    assert "행 수" in r.content and "채권 종목" in r.content

    # 이미 DISTINCT 를 쓴 질의에는 경고하지 않는다
    ok = await s.tools.execute("sql_query", {"sql": "SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1"})
    assert "행 수" not in ok.content

    # 목록 조회에도 경고하지 않는다
    lst = await s.tools.execute("sql_query", {"sql": "SELECT pd_no, pd_nm FROM v_bond_kr LIMIT 2"})
    assert "행 수" not in lst.content


def test_request_prompt_stays_within_token_budget(tiny_store, tmp_path):
    """매 라운드 실제로 전송되는 크기 = 시스템 프롬프트 + 도구 정의.

    무엇이 진짜 제약인가:
      * 주최측은 토큰 상한을 두지 않았다. 명시된 제약은 문항당 300초 응답 시간뿐이다.
      * 실제 제약은 CLOVA 의 분당 토큰 수(TPM)다. 429 가 실제로 관측된 크기는 13,389토큰이었고,
        그 상태로는 분당 2회밖에 호출하지 못해 평가 22문항 중 21문항이 429 로 실패했다.
      * 프롬프트가 커지면 느려진다는 통념은 측정으로 반증됐다. 프롬프트 크기를
        599 / 5,591 / 8,789 / 13,495토큰으로 바꿔가며 첫 호출 지연시간 중앙값을 재니
        2.90 / 3.65 / 2.94 / 3.82초로, 차이가 잡음 범위 안이었다. 크기는 지연의 원인이 아니다.
        따라서 상한의 근거는 300초 예산이 아니라 오직 TPM 이다.

    무엇을 재야 하는가: 시스템 프롬프트만 재면 실제 전송량보다 작다. 매 라운드 도구 스키마
    8개(OpenAI 호환 wire 형식으로 직렬화 시 약 2,594토큰)가 함께 나가기 때문이다.
    tiny_store 기준 시스템 프롬프트 8,754 + 도구 2,594 = 11,348토큰이 한 번의 요청 크기다.

    상한 12,000 의 근거: 이 테스트는 tiny_store 로 재므로 실서비스보다 작다. 실제 스토어에
    value_hints 까지 실은 운영 프롬프트는 9,187토큰이라 요청 총량은 11,781토큰, 즉 여기서 재는
    값보다 약 433토큰 크다. 상한 12,000 이 걸리는 시점의 운영 크기는 약 12,433토큰으로,
    429 가 관측된 13,389 보다 950토큰 이상 아래다. 오늘 크기 대비 약 650토큰의 증가 여유가 있다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    wire_tools = json.dumps(
        [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
         for t in s.tools.specs()],
        ensure_ascii=False,
    )
    prompt_tokens = len(s.orchestrator.system_prompt) / 1.6
    tool_tokens = len(wire_tools) / 1.6
    approx_tokens = prompt_tokens + tool_tokens
    assert approx_tokens < 12000, (
        f"요청 1회 크기가 너무 큼: {approx_tokens:,.0f} 토큰 "
        f"(시스템 프롬프트 {prompt_tokens:,.0f} + 도구 정의 {tool_tokens:,.0f})"
    )
    # 요약본에도 판정에 필요한 핵심 컬럼은 남아 있어야 한다
    for col in ("is_sellable", "has_buy_yield", "crd_grd", "is_on_sale", "fund_family", "expense_ratio_pct"):
        assert col in s.orchestrator.system_prompt, col
    # 전체 정의는 table_info 로 여전히 받을 수 있다
    assert "table_info" in s.orchestrator.system_prompt


async def test_table_info_still_returns_full_card(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("table_info", {"table": "v_bond_kr"})
    assert not r.is_error and len(r.content) > len(s.orchestrator.system_prompt) / 8


# ------------------------------------------------- 막다른 길에서 빠져나오게 하는 두 장치
# (1) SQL 오류가 '무엇이 맞는지'까지 알려준다  (2) 같은 호출을 반복하면 그렇다고 말해 준다

async def test_wrong_column_error_names_the_right_column(tiny_store, tmp_path):
    """G-007 재발 방지: "no such column: pd_ticker" 는 무엇이 틀렸는지만 말한다.

    실제 컬럼명(v_etf_gl.ticker)을 짚어 줘야 모델이 추측 없이 한 번에 고칠 수 있다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("sql_query", {"sql": "SELECT pd_ticker, pd_nm FROM v_etf_gl LIMIT 3"})
    assert r.is_error
    assert "no such column: pd_ticker" in r.content      # 원래 메시지는 지우지 않는다
    assert "v_etf_gl.ticker" in r.content                # 접두어 혼동(pd_ 를 붙임)을 정확히 짚는다

    # 별칭이 붙은 JOIN 질의는 그 별칭이 가리키는 뷰만 본다(G-015 의 pd_no)
    j = await s.tools.execute("sql_query", {"sql": (
        "SELECT e.pd_no FROM v_etf_kr e JOIN v_etf_constituent c ON e.pd_itm_no=c.etf_itm_no LIMIT 3")})
    assert j.is_error and "pd_itm_no" in j.content and "v_etf_kr" in j.content

    # 없는 뷰를 물으면 쓸 수 있는 5개를 알려준다
    t = await s.tools.execute("sql_query", {"sql": "SELECT * FROM v_etf_korea LIMIT 1"})
    assert t.is_error and "v_etf_kr" in t.content
    for v in ("v_bond_kr", "v_etf_gl", "v_fund_pub", "v_etf_constituent"):
        assert v in t.content, v

    # 컬럼 목록을 SQL 로 캐내려는 시도는 가드가 막는다 — 막힌 이유 대신 쓸 도구를 알려준다
    p = await s.tools.execute("sql_query", {"sql": "PRAGMA table_info(v_etf_kr)"})
    assert p.is_error and "table_info" in p.content

    # 정상 질의는 손대지 않는다
    ok = await s.tools.execute("sql_query", {"sql": "SELECT pd_abrv_nm FROM v_etf_kr LIMIT 1"})
    assert not ok.is_error and "대신" not in ok.content

    # 고칠 방법이 없는 오류는 그대로 둔다
    syn = await s.tools.execute("sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE"})
    assert syn.is_error and "syntax error" in syn.content and "대신" not in syn.content


async def test_wide_view_error_stays_bounded(tiny_store, tmp_path):
    """넓은 뷰의 컬럼을 통째로 실으면 컨텍스트를 잡아먹는다. 근접 후보 + 총 개수만 준다."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    wide = [f"col_{i}" for i in range(60)] + ["ticker"]
    s.tools._colcache = {"v_etf_gl": wide}   # 실제 스토어의 v_etf_gl 은 39컬럼, v_etf_kr 은 59컬럼
    msg = s.tools._sql_error_hint("SELECT pd_ticker FROM v_etf_gl", "SQL 오류: no such column: pd_ticker")
    assert "v_etf_gl.ticker" in msg
    assert f"컬럼 {len(wide)}개 중" in msg and "table_info('v_etf_gl')" in msg
    assert "col_59" not in msg and len(msg) < 400   # 전체를 쏟아붓지 않는다


def test_call_key_defines_the_same_call():
    """'같은 호출'의 정의를 고정한다. 느슨해지면 멀쩡한 재시도까지 막는다."""
    a = ToolCall("1", "sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE manager_name='삼성자산운용'", "purpose": "1차"})
    b = ToolCall("2", "sql_query", {"sql": "select  pd_nm\nfrom v_etf_kr\nwhere manager_name = '삼성자산운용';", "purpose": "다시 시도"})
    c = ToolCall("3", "sql_query", {"sql": "select pd_nm from v_etf_kr where manager_name = '미래에셋자산운용'"})
    assert _call_key(a) == _call_key(b)   # 대소문자·개행·연산자 공백·세미콜론·purpose 만 다르면 같은 호출
    assert _call_key(a) != _call_key(c)   # 리터럴 값이 다르면 다른 호출(리터럴은 소문자화하지 않는다)
    assert _call_key(ToolCall("4", "search_products", {"query": "로봇"})) != \
        _call_key(ToolCall("5", "search_products", {"query": "로봇 ETF"}))
    assert _call_key(ToolCall("6", "final_answer", {"status": "answered"})) is None


class _Tap(StubLLM):
    """모델에게 실제로 전달된 도구 결과 문자열을 그대로 모아 두는 StubLLM."""

    def __init__(self, script):
        super().__init__(script)
        self.tool_contents: list[str] = []

    def start(self, system, tools, user_text):
        conv = super().start(system, tools, user_text)
        orig = conv.add_tool_results
        conv.add_tool_results = lambda rs: (self.tool_contents.extend(c for _, c, _ in rs), orig(rs))[1]
        return conv


async def test_identical_call_is_not_re_executed_and_says_so(tiny_store, tmp_path):
    """G-015 재발 방지: 같은 질의를 세 번 보내면 두 번째부터는 실행하지 않고 '막혔다'고 알린다."""
    sql = "SELECT pd_no, pd_abrv_nm FROM v_etf_kr WHERE is_on_sale=1"   # pd_no 는 없는 컬럼
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": sql, "purpose": "1차 시도"})], "tool_use"),
        Turn("", [ToolCall("c2", "sql_query", {"sql": sql, "purpose": "2차 시도"})], "tool_use"),
        Turn("", [ToolCall("c3", "sql_query", {"sql": sql.lower() + " ;", "purpose": "3차 시도"})], "tool_use"),
        Turn("", [ToolCall("c4", "final_answer", {"status": "cannot_confirm", "answer": "확인 불가"})], "tool_use"),
    ]
    llm = _Tap(script)
    s = _state(tiny_store, llm, tmp_path)
    executed = []
    real = s.rdb.query
    s.rdb.query = lambda q: (executed.append(q), real(q))[1]   # DB 를 실제로 몇 번 때렸는지 센다

    resp = await s.orchestrator.answer("R-001", "국내 ETF 목록 알려줘")

    assert sum(1 for q in executed if "pd_no" in q.lower()) == 1, executed   # 2·3번째는 미실행
    assert len(llm.tool_contents) == 3
    assert "번째 동일한 호출입니다" not in llm.tool_contents[0]
    assert "2번째 동일한 호출입니다" in llm.tool_contents[1]
    assert "3번째 동일한 호출입니다" in llm.tool_contents[2]
    assert "결과가 달라지지 않습니다" in llm.tool_contents[1]
    assert "table_info" in llm.tool_contents[1]              # 실패한 호출에는 컬럼 확인을 지시한다
    # 반복분도 첫 결과(오류 메시지)를 그대로 담고 있다
    assert "no such column: pd_no" in llm.tool_contents[1]
    # think_trace 에도 남는다
    assert "동일 호출 2번째" in resp.think_trace and "동일 호출 3번째" in resp.think_trace
    assert "동일 호출 3회 → 조기 마무리 안내" in resp.think_trace


async def test_zero_row_repeat_suggests_relaxing_conditions(tiny_store, tmp_path):
    """0건이 반복되면 '조건을 하나씩 빼라'고 구체적으로 말한다. 근거는 두 번 세지 않는다."""
    sql = "SELECT pd_abrv_nm FROM v_etf_kr WHERE manager_name='없는운용사'"
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": sql})], "tool_use"),
        Turn("", [ToolCall("c2", "sql_query", {"sql": sql})], "tool_use"),
        Turn("", [ToolCall("c3", "final_answer", {"status": "cannot_confirm", "answer": "확인 불가"})], "tool_use"),
    ]
    llm = _Tap(script)
    s = _state(tiny_store, llm, tmp_path)
    resp = await s.orchestrator.answer("R-003", "없는운용사 ETF 알려줘")
    assert "조건을 하나씩 빼면서" in llm.tool_contents[1]
    assert resp.retrieved_context.count("[근거") == 1   # 반복분에는 evidence 를 붙이지 않는다


async def test_two_different_calls_both_run_untouched(tiny_store, tmp_path):
    """정상 질문에는 아무 일도 일어나면 안 된다 — 서로 다른 두 호출은 둘 다 실행된다."""
    q1 = "SELECT pd_abrv_nm FROM v_etf_kr WHERE is_on_sale=1"
    q2 = "SELECT pd_no, pd_nm FROM v_bond_kr WHERE is_sellable=1"
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": q1})], "tool_use"),
        Turn("", [ToolCall("c2", "sql_query", {"sql": q2})], "tool_use"),
        Turn("", [ToolCall("c3", "final_answer", {"status": "answered", "answer": "ETF 2종, 채권 2종입니다."})], "tool_use"),
    ]
    llm = _Tap(script)
    s = _state(tiny_store, llm, tmp_path)
    executed = []
    real = s.rdb.query
    s.rdb.query = lambda q: (executed.append(q), real(q))[1]

    resp = await s.orchestrator.answer("R-004", "판매중인 ETF와 채권 알려줘")
    assert any("v_etf_kr" in q for q in executed) and any("v_bond_kr" in q for q in executed)
    assert "동일 호출" not in resp.think_trace
    assert not any("동일한 호출입니다" in c for c in llm.tool_contents)
    assert resp.retrieved_context.count("[근거") == 2
