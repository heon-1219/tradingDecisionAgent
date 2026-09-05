"""Tool registry, orchestrator and API contract tests with a scripted stub LLM on a tiny store."""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finagent.agent.answer import Evidence, format_evidence, normalize_answer
from finagent.agent.llm import StubLLM, ToolCall, Turn
from finagent.agent.orchestrator import Orchestrator
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
        "std_pd_mcls_nm, crd_grd, pd_risk_gcd, curr_cd, buy_yield, after_tax_yield, srfc_irt, mat_date, base_date, pd_risk_nm)"
    )
    c.executemany(
        "insert into bond_kr values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("KR1", "한국전력공사채권1012", "한전1012", "KEPCO", "국내채권", "장외", 1, "한국전력공사", "일반특수법인채", "특수채", "AAA", "16", "KRW", 4.136, 3.95, 3.1, "2029-12-19", "2026-08-21", "매우낮은위험(6등급)"),
            ("KR2", "CJ대한통운98-3", "CJ98-3", "CJ", "국내채권", "장외", 1, "씨제이대한통운", "일반회사채", "회사채", "AA-", "15", "KRW", 3.99, 3.74, 3.0, "2028-02-04", "2026-08-21", "낮은위험(5등급)"),
            ("KR3", "국고채권01125", "국고", "KTB", "국내채권", "장내", 0, "대한민국", "국고채권", "국공채", None, "16", "KRW", None, None, 1.125, "2039-09-10", "2026-08-21", "매우낮은위험(6등급)"),
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
        "create view v_etf_gl as select null pd_itm_no, null ticker, null pd_nm, null product_type, null is_on_sale, null base_date, null manager_name, "
        "null cu_base_index, null wu_inv_ast_type, null wu_inv_rgn, null expense_ratio_pct, null du_last_aum, null cu_lev_fector, null cu_inverse_short_yn, "
        "null listing_date, null cu_strtegy, null pd_exg_mkt_cd where 0"
    )
    c.execute(
        "create view v_fund_pub as select null itm_no, null itm_nm, null itm_abrv_nm, null product_type, null is_on_sale, null base_date, null manager_name, "
        "null fund_family, null bmrk_nm, null or_attr_desc, null zrin_ptn_nm, null fd_ivst_rgn_desc, null risk_grade, null is_hedged, null fd_nast_suma, "
        "null zrin_btyp_nm, null zrin_fd_ivst_risk_grd_nm where 0"
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
    cons = tmp_path / "cons"
    cons.mkdir(exist_ok=True)
    (cons / "c.csv").write_text(
        "etf_id,etf_table,ticker,etf_name,company_id,company_name,weight_pct,as_of,source\nKRE1,v_etf_kr,305720,KODEX 2차전지산업,ecopro_bm,에코프로비엠,7.0,2026-08-21,t\n",
        encoding="utf-8",
    )
    kg = KG.build(rdb, lx, None, cons, log=lambda *a: None)
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
