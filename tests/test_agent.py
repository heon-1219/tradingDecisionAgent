"""Tool registry, orchestrator and API contract tests with a scripted stub LLM on a tiny store."""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finagent.agent.answer import (
    CANNOT_PREFIX,
    Evidence,
    format_evidence,
    normalize_answer,
    strip_closing_offer,
    strip_tool_call_syntax,
)
from finagent.agent.llm import StubLLM, ToolCall, Turn
from finagent.agent.orchestrator import (
    Orchestrator,
    _call_key,
    asks_forward_looking,
    asks_holdings_without_data,
)
from finagent.agent.prompts import build_system_prompt
from finagent.agent.tools import GRAPH_LABELS_SHOWN, ToolRegistry
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


def test_leaked_tool_call_syntax_is_unwrapped_not_deleted():
    """모델이 final_answer 를 '호출'하지 않고 그 구문을 answer 안에 글자로 써 보낸 실제 사례들.

    기록된 54개 답변 중 6개가 이랬다. 껍데기만 벗기고 안의 본문은 반드시 살아남아야 한다 —
    통째로 지우면 답이 사라지고, 그냥 두면 사용자가 도구 구문을 읽게 된다.
    """
    # G-006: 호출 구문이 답변 전체인 경우
    assert strip_tool_call_syntax(
        'final_answer(status="completed", answer="국내 ETF 중 위험등급 1등급이면서 순자산 1조원 이상인 상품은 총 22개입니다.")'
    ) == "국내 ETF 중 위험등급 1등급이면서 순자산 1조원 이상인 상품은 총 22개입니다."

    # G-016: 진짜 문장 뒤에 호출 구문이 붙은 경우. 앞 문장도 안의 본문도 모두 남아야 한다.
    g016 = ('제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", '
            'answer="제공된 데이터에서는 신용등급이 \'AAAA\'인 채권을 찾을 수 없습니다. '
            '유효한 신용등급으로는 AAA, AA+, AA0, AA- 등이 있으니 이러한 등급을 가진 채권을 찾으시면 도움이 될 것 같습니다.")')
    out = strip_tool_call_syntax(g016)
    assert "final_answer" not in out and 'status=' not in out and 'answer="' not in out
    assert out.startswith("제공된 데이터로는 확인할 수 없습니다. 제공된 데이터에서는 신용등급이 'AAAA'인 채권을 찾을 수 없습니다.")
    assert out.endswith("도움이 될 것 같습니다.")

    # G-043: 인자 안의 \n 이스케이프는 실제 줄바꿈으로 풀린다(목록이 한 줄로 뭉치지 않게)
    g043 = ('final_answer(status="completed", answer="판매 중인 국내 ETF와 공모펀드를 통틀어 위험등급 1등급으로 '
            '분류된 상품은 총 1,097개입니다.\\n\\n- 국내 ETF: 223개\\n- 공모펀드: 874개")')
    out = strip_tool_call_syntax(g043)
    assert "\\n" not in out and "- 국내 ETF: 223개\n- 공모펀드: 874개" in out

    # G-054: 안내 문구 + 코드펜스로 감싼 경우. 껍데기 세 겹이 모두 벗겨져야 한다.
    g054 = ('final_answer 도구를 사용하여 답변을 제출하겠습니다:\n\n```plaintext\n'
            'final_answer(status="cannot_confirm", answer="제공된 데이터로는 총보수 정보를 찾을 수 없습니다.")\n```')
    assert strip_tool_call_syntax(g054) == "제공된 데이터로는 총보수 정보를 찾을 수 없습니다."

    # 출력 상한을 낮추면 구문이 닫히기 전에 잘릴 수 있다. 그 조각도 같은 방식으로 벗긴다.
    assert strip_tool_call_syntax(
        '제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", answer="제공된 데이터로는 특정 상품을'
    ) == "제공된 데이터로는 확인할 수 없습니다. 제공된 데이터로는 특정 상품을"

    # ---- 손대면 안 되는 것들: 괄호·따옴표가 든 정상 답변(실제 기록 G-003 / G-009 / G-026) ----
    for legit in (
        "매수수익률이 수록된 채권 중 세후 운용수익률이 가장 높은 종목은 '국고채권 01500-5003(20-2)'이며, 수익률은 5.70%입니다.",
        "해외 ETF 중 운용규모(AUM)가 가장 큰 상품은 Vanguard 500 Index Fund;ETF - 997,351,790,000원 (9,973억 5,179만원)입니다.",
        # 번호 목록 예시는 여기서 뺐다.  답변을 줄글로만 내보내기로 하면서 normalize_answer 가
        # 목록 마커를 걷어 한 문장으로 잇기 때문에 == legit 이 더는 성립하지 않는다.  마커만
        # 사라지고 항목은 하나도 잃지 않는다는 기대치는
        # test_flattened_list_supersedes_the_earlier_passthrough_expectation 이 대신 검증한다.
        # strip_tool_call_syntax 가 그 문자열을 건드리지 않는다는 원래 의도도 그쪽에 함께 있다.
        '위험등급은 "매우높은위험(1등급)" 형태의 라벨로 제공됩니다.',
    ):
        assert strip_tool_call_syntax(legit) == legit, legit
        assert normalize_answer("answered", legit) == legit, legit


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
              "작년 연말 수익률 알려줘",
              # '규모'는 값이 아니라 크기를 가리키는 말이다. 값 표현으로 넣었더니 이 정상 질문이
              # '미래 … 규모'로 걸렸다. G-045 는 '순자산'만으로도 잡히므로 '규모'는 필요 없다.
              "미래 성장 규모가 큰 테마", "미래 성장성이 높은 산업 관련 ETF"):
        assert not asks_forward_looking(q), q


def test_future_year_counts_as_a_future_time_reference():
    """G-045 재발 방지: '2027년 말'처럼 연도로만 적은 미래 시점도 미래 표현으로 센다.

    기준연도(데이터 base_date 의 연도)를 넘는 연도만 미래다. 함정은 G-002 —
    "만기가 2027년인 채권"은 미래 연도를 말하지만 mat_date 라는 **기록된 날짜 필드**를 묻는
    질문이라 답할 수 있어야 한다. 값 표현이 아니라 날짜 필드에 붙은 연도는 세지 않는다.
    """
    for q in ("KODEX 200의 2027년 말 예상 순자산 규모는 얼마인가요?",
              "2028년 순자산 규모를 알려줘", "2030년 수익률이 어떻게 될까요?"):
        assert asks_forward_looking(q, 2026), q
    # ★ 미래 연도 + 날짜 필드 = 예측이 아니라 조회다. 여기서 발동하면 답할 수 있는 문항이 죽는다.
    for q in ("구매 가능한 원화채권 중 신용등급 AAA이고 만기가 2027년인 종목 수를 알려줘",
              "만기 2029년 채권 목록", "2027년 상장 예정 ETF가 있나요?",
              "2026년 말 순자산 규모는?",   # 기준연도 이하 = 과거·현재 스냅샷
              "2025년 수익률 알려줘"):
        assert not asks_forward_looking(q, 2026), q
    # 기준연도는 스토어 meta 에서 온다 — 기준연도가 밀리면 판정도 함께 밀려야 한다
    assert asks_forward_looking("2027년 순자산 규모", 2026)
    assert not asks_forward_looking("2027년 순자산 규모", 2027)


def test_holdings_scope_guard_fires_only_where_there_is_no_holdings_data():
    """G-040·G-041 재발 방지: 편입 데이터는 국내 ETF 1,162 종목에만 있다.

    해외ETF·공모펀드·사모펀드·ETN 으로 범위를 좁힌 편입 질문은 데이터에 관계가 없어
    답할 수 없다. 반대로 국내 ETF 편입 질문은 정상 답변 대상이므로 절대 건드리면 안 된다.
    """
    for q in ("엔비디아를 편입한 해외 ETF를 알려줘",
              "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
              "삼성전자를 보유한 공모펀드를 알려줘",
              "테슬라를 포함한 미국 ETF가 있나요?",
              "카카오를 편입한 ETN을 알려줘"):
        assert asks_holdings_without_data(q), q
    # ★ 답할 수 있는 문항 — 여기서 발동하면 정상 답변을 확인불가로 뭉갠다
    for q in ("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘",
              "SK하이닉스를 편입하고 있는 국내 ETF는 몇 개인가요?",
              "두산에너빌리티를 보유한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘",
              "국내 ETF 가운데 삼성전자 비중이 가장 높은 상품은 무엇인가요?",
              "에코프로비엠을 편입한 ETF는 몇 개인가요?",          # 범위어가 없으면 발동하지 않는다
              "국내 ETF와 공모펀드를 합쳐서 1년 수익률이 가장 높은 상품 3개를 알려줘",  # 편입어 없음
              "판매중인 상품 기준으로 채권에 투자하는 국내 ETF와 채권형 공모펀드는 각각 몇 개인가요?",
              "해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘"):
        assert not asks_holdings_without_data(q), q


async def test_fund_holdings_answer_is_rewritten_to_cannot_confirm(tiny_store, tmp_path):
    """G-041 재발 방지: v_fund_pub 에는 보유 종목 컬럼이 없다.

    search_products 는 이름·의미 유사도만 돌려주므로 '그 종목을 담고 있다'를 성립시키지 못한다.
    모델이 그렇게 단정하면 서버가 확인불가로 내리고 근거가 없다는 사실을 밝힌다.
    """
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 KB 만기가 있는 채권 EMP입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("H-001", "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?")
    assert resp.answer.startswith(CANNOT_PREFIX)
    assert "국내 ETF 1,162 종목" in resp.answer
    assert "확인불가로 보정" in resp.think_trace


async def test_domestic_etf_holdings_answer_is_left_alone(tiny_store, tmp_path):
    """국내 ETF 편입 질문은 답할 수 있다 — 가드가 건드리면 안 된다(오탐 방지)."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "에코프로비엠을 편입한 국내 ETF는 KODEX 2차전지산업 1개입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("H-002", "에코프로비엠을 편입한 국내 ETF는 몇 개인가요?")
    assert not resp.answer.startswith(CANNOT_PREFIX)
    assert "KODEX 2차전지산업" in resp.answer
    assert "확인불가로 보정" not in resp.think_trace


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
    tiny_store 기준 시스템 프롬프트 9,152 + 도구 2,594 = 11,746토큰이 한 번의 요청 크기다.
    ("이 데이터가 무엇인가" 사실 절 추가로 8,754 -> 9,152. 중복된 편입종목 안내 2곳을 지워
     +398토큰에 그쳤다.)

    상한 12,000 의 근거: 이 테스트는 tiny_store 로 재므로 실서비스보다 작다. 실제 스토어에
    value_hints 까지 실은 운영 프롬프트는 9,584토큰이라 요청 총량은 12,178토큰, 즉 여기서 재는
    값보다 약 432토큰 크다. 운영 크기가 상한을 넘었지만 상한의 근거는 12,000 이라는 숫자가 아니라
    TPM 이다 — 429 가 실제로 관측된 13,389 보다 1,211토큰 아래이므로 아직 여유가 있다.
    이 테스트에 남은 증가 여유는 약 254토큰이다. 더 늘려야 하면 상한을 올리기 전에
    CORE_COLUMNS 의 저빈도 컬럼이나 value_hints 를 줄이는 쪽을 먼저 본다.
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
    # 상한 12,000 -> 14,000.  이 한도는 "프롬프트가 커지면 느려진다"는 가정에서 나왔는데, 측정은
    # 그 가정을 부정했다: 프롬프트 599 / 5,591 / 8,789 / 13,495 토큰에 대한 첫 호출 지연 중앙값이
    # 2.90 / 3.65 / 2.94 / 3.82초로 22배 커지는 동안 평평했다(지연을 만드는 것은 왕복 횟수다).
    # 그래서 한도는 이제 비용이 아니라 "프롬프트가 조용히 부풀지 않는지" 보는 경보다.  12,000 은
    # 오늘 여유가 10토큰까지 줄어 규칙 한 줄을 더할 때마다 깨졌고, 그때마다 문구를 줄이는 압박이
    # 정확도보다 예산을 앞세우게 만들었다.  14,000 은 같은 경보를 유지하되 그 압박을 없앤다.
    assert approx_tokens < 14000, (
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


async def test_missing_column_says_what_to_do_not_just_what_is_wrong(tiny_store, tmp_path):
    """119초 문항 재발 방지: du_er_1y FROM v_etf_gl 실패 직후 모델이 답 대신 '접근 방법'을 썼다.

    v_etf_gl 에는 1년 수익률 컬럼이 없고 du_er_1d(1일)뿐이다. 철자가 가까워 제안은 나오지만
    그 제안은 답이 될 수 없다 — 그래서 '무엇이 맞는지'만이 아니라 '그래서 무엇을 하라'가 필요하다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    s.tools._colcache = {"v_etf_gl": ["ticker", "pd_nm", "is_on_sale", "du_er_1d", "du_last_aum"]}
    msg = s.tools._sql_error_hint(
        "SELECT ticker, du_er_1y FROM v_etf_gl WHERE is_on_sale=1 ORDER BY du_er_1y DESC LIMIT 10",
        "SQL 오류: no such column: du_er_1y",
    )
    assert "du_er_1d" in msg                       # 기존 동작: 가까운 컬럼은 계속 짚어 준다
    assert "뜻" in msg and "같지는 않습니다" in msg  # 이름이 가까운 것 != 뜻이 같은 것
    assert "제외" in msg                            # 대체 금지 → 그 상품군을 제외하고
    assert "방법 설명 금지" in msg                   # 계획을 쓰지 말고 바로 답하라

    # 근접 후보가 하나도 없을 때도 같은 지시가 붙는다
    s.tools._colcache = {"v_etf_gl": ["ticker", "pd_nm"]}
    bare = s.tools._sql_error_hint("SELECT zzz_qqq FROM v_etf_gl", "SQL 오류: no such column: zzz_qqq")
    assert "제외" in bare and "방법 설명 금지" in bare

    # 컬럼 오류가 아닌 실패에는 붙지 않는다
    syn = await s.tools.execute("sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE"})
    assert syn.is_error and "방법 설명 금지" not in syn.content


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


async def test_identical_call_is_re_executed_and_says_so(tiny_store, tmp_path):
    """G-015 재발 방지: 같은 질의를 세 번 보내면 '막혔다'고 알린다.

    단, 호출을 **막지는 않는다**(G-040). 실행을 건너뛰어 아끼는 것은 도구 시간 몇 밀리초뿐이고,
    그 대가로 모델이 막다른 길에서 빠져나오려는 시도를 잘라 버린다. 실행은 그대로 하고
    결과 뒤에 '몇 번째 같은 호출인지'만 덧붙인다.
    """
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

    assert sum(1 for q in executed if "pd_no" in q.lower()) == 3, executed   # 세 번 모두 실행된다
    assert len(llm.tool_contents) == 3
    assert "번째 동일한 호출입니다" not in llm.tool_contents[0]
    assert "2번째 동일한 호출입니다" in llm.tool_contents[1]
    assert "3번째 동일한 호출입니다" in llm.tool_contents[2]
    assert "결과가 달라지지 않습니다" in llm.tool_contents[1]
    assert "실행하지 않고" not in llm.tool_contents[1]   # 안내 문구가 사실과 어긋나면 안 된다
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
    executed = []
    real = s.rdb.query
    s.rdb.query = lambda q: (executed.append(q), real(q))[1]
    resp = await s.orchestrator.answer("R-003", "없는운용사 ETF 알려줘")
    assert sum(1 for q in executed if "없는운용사" in q) == 2   # 재실행은 막지 않는다
    assert "조건을 하나씩 빼면서" in llm.tool_contents[1]
    # 실행은 두 번 했어도 근거는 한 번만 센다 — 같은 결과가 [근거] 로 두 줄 찍히면 안 된다
    assert resp.retrieved_context.count("[근거") == 1


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


def test_closing_offer_is_stripped_but_facts_and_caveats_survive():
    """기록된 54개 답변 중 8개가 "추가적인 정보가 필요하시면 …"으로 끝났다.

    그 문장은 정보가 0인데 생성 시간을 쓴다(측정: 고정 2.7초 + 63토큰/초). 실제로 지워지는 것은
    54개 중 6개고 나머지 48개는 바이트 단위로 그대로다. 지우면 안 되는 것이 더 중요하다 —
    수치가 든 문장, "투자 판단은 본인 책임" 같은 유의사항, "…없습니다" 같은 확인 불가 사실.
    """
    # 실제 기록에서 지워지는 문장들 (G-021 / G-022 / G-009 / G-043 / G-054)
    assert strip_closing_offer(
        "혹시 이 중에서 찾고 계신 상품이 있는지 확인 부탁드립니다. 추가적인 정보가 필요하시면 말씀해 주세요."
    ) == "혹시 이 중에서 찾고 계신 상품이 있는지 확인 부탁드립니다."
    assert strip_closing_offer(
        "그 중 어떤 상품도 질문하신 상품과 일치하지 않습니다. 추가적인 정보가 필요하시면 다시 문의해 주세요."
    ) == "그 중 어떤 상품도 질문하신 상품과 일치하지 않습니다."
    assert strip_closing_offer(
        "각 ETF의 운용규모는 미국 달러(USD) 기준입니다. 투자 결정 시 추가적인 정보를 확인하는 것을 권장드립니다."
    ) == "각 ETF의 운용규모는 미국 달러(USD) 기준입니다."
    assert strip_closing_offer(
        "위험등급 정보는 'risk_grade' 열을 통해 확인하였습니다. 투자 결정 시에는 추가적인 정보를 고려하시기 바랍니다."
    ) == "위험등급 정보는 'risk_grade' 열을 통해 확인하였습니다."
    assert strip_closing_offer(
        "총보수 정보는 도구에서 반환되지 않았습니다. "
        "추가적인 정보가 필요하다면 직접 금융기관의 웹사이트를 방문하거나 고객 서비스에 문의하시는 것을 권장드립니다."
    ) == "총보수 정보는 도구에서 반환되지 않았습니다."

    # ---- 손대면 안 되는 것들 ----
    keep = (
        # 유의사항이 같은 문장에 섞여 있으면 통째로 남긴다(G-020 / G-037 실제 기록).
        "투자 판단은 본인의 책임 하에 이루어져야 하며, 추가적인 정보가 필요하신 경우 개별 ETF의 세부 사항을 확인하시기 바랍니다.",
        "투자 판단은 본인의 책임이며, 추가적인 정보는 제공된 도구들을 통해 더욱 상세히 확인할 수 있습니다.",
        # 수치가 든 문장은 답의 본체다
        "조건에 맞는 종목은 15,837개이며 그중 상위 5개는 아래와 같습니다.",
        "추가적인 정보로 순자산은 4조 1,402억원입니다.",
        # 확인 불가·제외 사실
        "해외ETF는 1년 수익률이 제공되지 않아 비교에서 제외했습니다.",
        "제공된 데이터로는 확인할 수 없습니다.",
        # 그냥 안내가 아닌 마지막 문장
        "이 상품의 총보수는 0.05%입니다.",
    )
    for t in keep:
        assert strip_closing_offer(t) == t, t
        assert normalize_answer("answered", t) == t, t

    # 파이프라인에 실제로 물려 있고, 빈 답을 만들지 않는다
    assert normalize_answer("answered", "총 22개입니다. 추가적인 정보가 필요하시면 문의해 주세요.") == "총 22개입니다."
    assert normalize_answer("answered", "추가적인 정보가 필요하시면 문의해 주세요.") == CANNOT_PREFIX


async def test_graph_neighbors_shows_few_labels_but_the_true_total(tiny_store, tmp_path):
    """라벨은 8개만 찍고 총 건수는 그대로 보고한다.

    잘린 개수를 총계로 보고한 것이 예전의 실제 버그였다(에코프로비엠 42건이 60건 상한으로 보고됨).
    라벨 40개는 삼성전자 한 번 호출에 1,501자였고 그 텍스트는 이후 모든 라운드에 다시 실려 나간다.
    """
    class _FakeKG:
        def __init__(self, n):
            self.n = n

        def neighbors(self, entity, relation=None, limit=50, totals=None):
            if totals is not None:
                totals["constituentOf"] = self.n
            return {"constituentOf": [{"iri": f"d:etf/{i}", "label": f"ETF{i:03d}"} for i in range(min(self.n, limit))]}

    s = _state(tiny_store, StubLLM([]), tmp_path)
    s.tools.kg = _FakeKG(229)
    r = await s.tools.execute("graph_neighbors", {"entity": "삼성전자"})
    assert "229건" in r.content and r.evidence.row_count == 229      # 총계는 잘리지 않는다
    assert r.content.count("ETF") == GRAPH_LABELS_SHOWN == 8         # 라벨은 8개만
    assert "ETF007" in r.content and "ETF008" not in r.content
    assert "예시 8건" in r.content and r.content.rstrip().endswith("...")
    assert "상위" not in r.content                                    # 정렬돼 있지 않으므로 '상위'라 부르지 않는다
    assert len(r.content) < 200                                       # 40개 시절 삼성전자는 1,501자였다

    # 총계가 상한 이하면 아무것도 잘리지 않고 "예시" 표기도 붙지 않는다
    s.tools.kg = _FakeKG(3)
    r3 = await s.tools.execute("graph_neighbors", {"entity": "에코프로"})
    assert "3건" in r3.content and "예시" not in r3.content and "..." not in r3.content
    assert r3.content.count("ETF") == 3


# ---------------------------------------------------------------- 온톨로지가 판정을 소유하는가
# 아래 네 가지는 예전에 프롬프트의 문장으로만 있었고, 문장은 코드가 읽을 수 없었다. 이제 판정의
# 출처는 TTL 이므로 TTL 을 고치면 여기 결과가 따라 움직여야 한다.


def test_ontology_answers_which_product_classes_carry_holdings():
    """편입종목을 가진 상품군을 그래프에서 읽어 온다 — 하드코딩 목록이 아니라."""
    from finagent.ontology.loader import Ontology

    o = Ontology.load(ROOT / "ontology")
    assert [c.name for c in o.holding_classes()] == ["DomesticETF"]
    assert o.holding_view() == "v_etf_constituent"
    # 상위 클래스는 하위 중 하나라도 가지면 참이다: 'ETF' 로만 물은 질문을 막으면 안 된다
    assert o.class_has_holdings("ETF") and o.class_has_holdings("DomesticETF")
    # ETN·펀드·채권은 하위까지 내려가도 편입종목이 없다
    for c in ("ETN", "DomesticETN", "Fund", "PublicFund", "PrivateFund", "ForeignETF", "DomesticBond"):
        assert not o.class_has_holdings(c), c


def test_ontology_scope_detection_agrees_with_the_regex_guard(tiny_store, tmp_path):
    """온톨로지 주도 상품군 판정이 기존 정규식 가드와 같은 답을 낸다(둘이 갈라지면 사고다)."""
    from finagent.agent.orchestrator import product_groups

    from finagent.agent.orchestrator import _HOLDING_WORD

    s = _state(tiny_store, StubLLM([]), tmp_path)
    o = s.orchestrator.ontology
    for q in ("엔비디아를 편입한 해외 ETF를 알려줘",
              "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
              "삼성전자를 보유한 공모펀드를 알려줘",
              "테슬라를 포함한 미국 ETF가 있나요?",
              "카카오를 편입한 ETN을 알려줘",
              "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘",
              "에코프로비엠을 편입한 ETF는 몇 개인가요?",
              "해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘"):
        g = product_groups(q, o)
        blind = bool(g and _HOLDING_WORD.search(q)) and not any(o.class_has_holdings(c.name) for c in g)
        assert blind == asks_holdings_without_data(q), (q, [c.name for c in g])


async def test_fund_holdings_question_gets_the_fact_not_a_holdings_count(tiny_store, tmp_path):
    """G-041: 공모펀드로 한정된 편입 질문에 '편입 상품 100건' 을 주면 안 된다.

    그 100 은 국내ETF 의 건수였고(그나마 상한에 잘린 값이었다), 모델은 그것을 공모펀드의
    건수로 읽었다. 이제는 건수 대신 '이 상품군에는 편입 데이터가 없다'는 사실이 들어간다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?")
    assert "편입 상품" not in pre.text and "편입 국내ETF" not in pre.text   # 건수를 주지 않는다
    assert "편입종목 데이터 범위" in pre.text
    assert "공모펀드(v_fund_pub)" in pre.text and "v_etf_constituent" in pre.text
    assert "확인할 수 없음" in pre.text                                     # 0건이 아니라 확인불가
    assert "fp:hasConstituent 의 정의역 = DomesticETF" in pre.text          # 근거는 온톨로지다


async def test_answerable_etf_holdings_question_keeps_its_count(tiny_store, tmp_path):
    """국내 ETF 편입 질문은 답할 수 있다 — 주입이 이 문항을 건드리면 정답을 막는다."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("에코프로비엠을 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개")
    assert "편입종목 데이터 범위" not in pre.text
    assert "편입 국내ETF 1건(그래프)" in pre.text      # 작은 스토어의 실제 편입 건수
    assert "정렬 기준 '순자산'" in pre.text and "v_etf_kr.pd_net_tamt" in pre.text
    assert "요청 개수: 3개" in pre.text


async def test_period_specific_criterion_names_the_group_that_lacks_it(tiny_store, tmp_path):
    """'1년 수익률' 로 세 상품군을 한 번에 물으면, 해외ETF 에 그 컬럼이 없다는 사실이 들어간다.

    이것이 92초짜리 3-way UNION 을 만든 사고의 원인이었다: 프롬프트에 산문으로 적혀 있었지만
    바로 위 예시가 3-way UNION 이라 모델은 예시를 따라갔다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("국내ETF, 해외ETF, 공모펀드를 통틀어 1년 수익률 기준 TOP10을 알려줘")
    assert "v_etf_kr.du_er_1y" in pre.text and "v_fund_pub.fd_yr1_ern_r" in pre.text
    assert "해외ETF(v_etf_gl)에는 '1년 수익률' 컬럼이 없습니다" in pre.text
    assert "du_er_1d" in pre.text                     # 대신 무엇이 있는지도 밝힌다
    assert "요청 개수: 10개" in pre.text


async def test_bare_return_criterion_is_reported_as_ambiguous_not_guessed(tiny_store, tmp_path):
    """맨 '수익률' 은 기간이 없어 여러 컬럼에 대응한다. 하나를 몰래 고르면 안 된다."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("수익률이 가장 높은 상품 10개를 알려줘")
    assert "기간·대상이 지정되지 않아" in pre.text
    assert "du_er_1y" in pre.text and "du_er_1m" in pre.text   # 후보를 이름으로 준다
    assert "임의로 고르지 말고" in pre.text


async def test_label_miss_is_not_reported_as_missing_data(tiny_store, tmp_path):
    """'운용규모' 는 라벨 조회로 0건이지만 v_etf_gl.du_last_aum 이 바로 그것이다.

    라벨이 없다는 이유로 '데이터에 없다'고 쓰면 답할 수 있는 문항이 확인불가가 된다.
    주석·영문 라벨까지 뒤져 후보를 주되, 부재를 단정하지 않는다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘")
    assert "du_last_aum" in pre.text
    assert "라벨이 없다고 데이터에 없는 것이 아닙니다" in pre.text
    # 부재 단정 문구는 하나도 나오면 안 된다 ('정확히 일치하는 컬럼이 없습니다' 는 후보 안내이지 부재 단정이 아니다)
    for bad in ("이 기준에 해당하는 컬럼이 없습니다", "순위 비교에서 제외", "확인할 수 없"):
        assert bad not in pre.text, bad
    # 같은 이유로, 라벨이 다른 뷰에 대해서도 부재를 단정하지 않는다 (순자산 vs v_etf_gl.du_last_aum)
    pre2 = s.orchestrator.pre_analyze("해외 ETF 중 순자산이 가장 큰 상품 3개")
    assert "없다고 단정하지 말고" in pre2.text and "du_last_aum(AUM)" in pre2.text
    assert "이 기준에 해당하는 컬럼이 없습니다" not in pre2.text


async def test_questions_without_a_ranking_criterion_get_no_criterion_line(tiny_store, tmp_path):
    """대부분의 문항에는 정렬 기준이 없다. 그럴 때 주입은 완전히 침묵해야 한다."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    for q in ("현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요?",
              "판매중인 상품 기준으로 채권에 투자하는 국내 ETF와 채권형 공모펀드는 각각 몇 개인가요?",
              "구매 가능한 원화채권 가운데 신용등급이 A0인 채권은 서로 다른 종목(종목코드) 기준으로 몇 개인가요?",
              "KODEX 로봇액티브를 운용하는 자산운용사는 어디인가요?",
              "삼성 KODEX 2차전지산업의 위험등급은?"):
        assert "정렬 기준" not in s.orchestrator.pre_analyze(q).text, q


# ===================================================================== 줄글(평문) 답변
# 주최측 응답 예시의 answer 는 평문 한 줄이다. 프롬프트 규칙 12 로 한 번, normalize_answer 로 한 번
# 막는다. 여기서 검증하는 것은 코드 방어 쪽이다 — 프롬프트 규칙은 지켜지지 않은 전례가 있다.
def test_markdown_emphasis_is_stripped_without_touching_numbers():
    assert normalize_answer("answered", "총보수는 **0.65**%입니다.") == "총보수는 0.65%입니다."
    assert normalize_answer("answered", "펀드는 '__한국투자네비게이터__'입니다.") == "펀드는 '한국투자네비게이터'입니다."
    assert normalize_answer("answered", "*가장 낮은* 값입니다.") == "가장 낮은 값입니다."
    # 짝이 맞지 않는 강조(출력이 잘린 경우)도 남기지 않는다
    assert normalize_answer("answered", "총보수는 **0.65%입니다.") == "총보수는 0.65%입니다."
    # 숫자·컬럼명·문장 부호는 마크다운이 아니다 — 그대로 둔다
    for intact in (
        "총보수 0.03%인 상품은 1개이며 순자산은 1,436.24억원입니다.",
        "du_er_1y 는 과거 1년 실적이며 pd_net_tamt 는 순자산입니다.",
        "국고채권 01500-5003(20-2) 의 세후 운용수익률은 5.7027%입니다.",
        "Vanguard 500 Index Fund;ETF - 997,351,790,000원 (9,973억 5,179만원)입니다.",
    ):
        assert normalize_answer("answered", intact) == intact, intact


def test_markdown_list_becomes_one_sentence_keeping_every_item():
    out = normalize_answer("answered", (
        "국내 ETF 중 순자산이 가장 큰 상품 3개는 다음과 같습니다:\n\n"
        "1. **KODEX 200 (069500)**: 25조 4,748억원\n"
        "2. **TIGER 미국S&P500 (360750)**: 20조 4,183억원\n"
        "3. **TIGER 미국나스닥100 (133690)**: 11조 3,932억원\n"
    ))
    assert "\n" not in out and "*" not in out
    assert out == (
        "국내 ETF 중 순자산이 가장 큰 상품 3개는 다음과 같습니다. "
        "KODEX 200 (069500): 25조 4,748억원, TIGER 미국S&P500 (360750): 20조 4,183억원, "
        "TIGER 미국나스닥100 (133690): 11조 3,932억원."
    )
    for keep in ("069500", "360750", "133690", "25조 4,748억원", "20조 4,183억원", "11조 3,932억원"):
        assert keep in out, keep


def test_headers_rules_tables_and_code_fences_are_stripped():
    out = normalize_answer("answered", (
        "### 결과\n\n| 상품명 | 순자산 |\n|---|---:|\n| KODEX 200 | 25조 |\n| TIGER 200 | 10조 |\n\n"
        "```\n총 2건\n```"
    ))
    for gone in ("###", "|", "---", "```"):
        assert gone not in out, gone
    for keep in ("KODEX 200", "25조", "TIGER 200", "10조", "총 2건"):
        assert keep in out, keep


def test_bullet_list_of_a_real_recorded_answer_is_flattened():
    # G-021 의 실제 답변 모양. 마커만 사라지고 코드 4개는 전부 남아야 한다.
    out = normalize_answer("answered", (
        "이 ETF의 정보는 다음과 같습니다:\n\n"
        "- **상품명**: KODEX 미국휴머노이드로봇\n"
        "- **단축코드**: KR7433970001, KR7433980000, KR70082V0008, KR7434060000\n"
        "- **운용사**: 삼성자산운용\n"
    ))
    assert not out.startswith("-") and "\n" not in out and "**" not in out
    for keep in ("KR7433970001", "KR7433980000", "KR70082V0008", "KR7434060000", "삼성자산운용"):
        assert keep in out, keep


# ============================================ cannot_confirm: 거절했으면 거기서 끝난다
# 사고 사례: "한화 에어로스페이스의 자회사를 편입한 ETF 중 …" 에 "찾을 수 없습니다" 라고 해놓고
# "그러나 한화 에어로스페이스를 편입한 ETF 목록을 제공할 수 있습니다" 하며 40개를 나열했다.
def test_cannot_confirm_drops_the_substitute_product_listing():
    out = normalize_answer("cannot_confirm", (
        "제공된 데이터로는 한화 에어로스페이스의 자회사를 편입한 ETF를 확인할 수 없습니다. "
        "자회사와 상품을 잇는 편입 데이터가 없습니다.\n"
        "그러나 한화 에어로스페이스를 편입한 ETF 목록을 제공할 수 있습니다:\n\n"
        "1. **KODEX 200** (069500) - 25조 4,748억원\n"
        "2. **TIGER 200** (102110) - 10조 4,278억원\n"
        "3. **PLUS 고배당주** (161510) - 1조 2,000억원\n\n"
        "이 외에도 총 40개의 ETF가 한화 에어로스페이스를 편입하고 있습니다."
    ))
    assert out == (
        "제공된 데이터로는 한화 에어로스페이스의 자회사를 편입한 ETF를 확인할 수 없습니다. "
        "자회사와 상품을 잇는 편입 데이터가 없습니다."
    )
    for gone in ("KODEX 200", "069500", "TIGER 200", "40개", "그러나", "목록"):
        assert gone not in out, gone


def test_cannot_confirm_drops_a_marker_less_row_listing_too():
    # 변경 1 로 마커가 사라지므로, 목록 판정은 마커가 아니라 '행 단위 상품'이라는 내용으로 한다.
    out = normalize_answer("cannot_confirm", (
        "제공된 데이터로는 자회사를 편입한 ETF를 확인할 수 없습니다.\n"
        "대신 한화에어로스페이스를 편입한 ETF는 다음과 같습니다.\n"
        "KODEX 200 25조 4,748억원\nTIGER 200 10조 4,278억원\nPLUS 고배당주 1조 2,000억원\n"
    ))
    assert out == "제공된 데이터로는 자회사를 편입한 ETF를 확인할 수 없습니다."


def test_cannot_confirm_keeps_the_valid_values_of_the_asked_condition():
    """지우면 안 되는 쪽 — 조건의 유효값을 인라인으로 알려주는 답변(G-016·G-017·G-022·G-052 실제 답변)."""
    for keep in (
        "제공된 데이터로는 확인할 수 없습니다. 신용등급 'AAAA'는 존재하지 않는 값입니다. "
        "유효한 신용등급은 AAA, AA+, AA0, AA-, A+, A0, A-, BBB+, BBB0, BBB-, BB+, BB0, BB-, "
        "B+, B0, B-, CCC, CC, C0, D입니다.",
        "제공된 데이터로는 확인할 수 없습니다. 위험등급 8등급은 존재하지 않습니다. "
        "위험등급은 1~6등급만 존재하며, 1등급이 가장 높은 위험입니다.",
        "제공된 데이터로는 확인할 수 없습니다. 제공된 데이터에는 'TIGER 양자컴퓨팅커버드콜 ETF'라는 "
        "상품이 존재하지 않습니다. 대신, SOL 미국양자컴퓨팅TOP10, RISE 미국양자컴퓨팅, "
        "PLUS 미국양자컴퓨팅TOP10, KIWOOM 미국양자컴퓨팅 등의 상품이 있습니다.",
        "제공된 데이터로는 확인할 수 없습니다. 제공된 데이터로는 쿠팡을 편입한 국내 ETF 정보를 찾을 수 없습니다.",
    ):
        assert normalize_answer("cannot_confirm", keep) == keep, keep


def test_answered_status_never_loses_a_product_listing():
    """상품 목록을 지우는 것은 cannot_confirm 전용이다. 정상 답변의 목록은 문장으로 이어질 뿐이다."""
    out = normalize_answer("answered", (
        "순자산 상위 3개는 다음과 같습니다:\n"
        "1. KODEX 200 (069500) - 25조 4,748억원\n"
        "2. TIGER 200 (102110) - 10조 4,278억원\n"
        "3. TIGER 반도체TOP10 (396500) - 9조 7,582억원"
    ))
    for keep in ("KODEX 200", "069500", "25조 4,748억원", "TIGER 200", "TIGER 반도체TOP10", "9조 7,582억원"):
        assert keep in out, keep


def test_cannot_confirm_listing_is_kept_when_the_refusal_would_not_survive():
    """목록이 답의 전부이면 손대지 않는다 — 여기서 통째로 날리는 것이 더 큰 손해다."""
    only_list = "1. KODEX 200 (069500) - 25조원\n2. TIGER 200 (102110) - 10조원\n3. ACE 200 (105190) - 1조원"
    out = normalize_answer("cannot_confirm", only_list)
    for keep in ("KODEX 200", "TIGER 200", "ACE 200"):
        assert keep in out, keep


def test_recorded_answers_that_must_pass_through_unchanged():
    """54개 회귀셋 중 바뀌면 안 되는 실제 답변들(수치·확인불가 문구·스키마 사실)."""
    for status, text in (
        ("answered", "현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 총 15,837개입니다."),
        ("answered", "TIGER 미국나스닥100을 운용하는 자산운용사는 미래에셋자산운용입니다."),
        ("answered", "판매중인 국내 ETF는 1,163개, 판매중인 공모펀드는 4,542개(클래스 합산)입니다."),
        ("cannot_confirm", "제공된 데이터로는 TIGER 반도체 ETF의 내년 예상 수익률을 확인할 수 없습니다."),
        ("cannot_confirm", "제공된 데이터로는 '문샷AI(Kimi)' 관련 투자상품을 확인할 수 없습니다."),
        ("cannot_confirm", "제공된 데이터로는 확인할 수 없습니다. 편입 종목(구성종목) 데이터는 국내 ETF "
                           "1,162 종목(네이버금융 수집)에만 있습니다. 해외 ETF·공모펀드·사모펀드·ETN 의 보유 "
                           "종목은 전달받은 데이터에 없으므로, 특정 기업을 편입했는지 여부와 그 순위는 확인할 수 없습니다."),
    ):
        assert normalize_answer(status, text) == text, text


def test_flattened_list_supersedes_the_earlier_passthrough_expectation():
    """test_leaked_tool_call_syntax_is_unwrapped_not_deleted 의 마지막 기대(목록 그대로 통과)는

    변경 1(줄글 강제)로 의도적으로 바뀌었다. strip_tool_call_syntax 는 여전히 손대지 않고,
    normalize_answer 만 목록을 한 문장으로 잇는다 — 항목은 하나도 사라지지 않는다.
    """
    legit = ("1. KB RISE 글로벌양자컴퓨팅증권상장지수투자신탁 (RISE 시리즈)\n"
             "2. 삼성 KODEX 글로벌양자컴퓨팅증권상장지수투자신탁 (KODEX 시리즈)")
    assert strip_tool_call_syntax(legit) == legit
    out = normalize_answer("answered", legit)
    assert out == ("KB RISE 글로벌양자컴퓨팅증권상장지수투자신탁 (RISE 시리즈), "
                   "삼성 KODEX 글로벌양자컴퓨팅증권상장지수투자신탁 (KODEX 시리즈).")


def test_prompt_carries_the_plain_prose_and_no_substitution_rules():
    from finagent.agent.prompts import RULES
    assert "줄글(평문)로만" in RULES and "마크다운 금지" in RULES and "번호목록" in RULES
    assert "묻지 않은 대상(상품군·개체·기간)으로 갈아타지 않습니다" in RULES
    assert "확인 불가면 거기서 멈추고 대체 상품 목록을 덧붙이지 않습니다" in RULES
    assert "마크다운 표/목록 가능" not in RULES        # 규칙 9 에서 뺐다


# ===================================================================== 조회 없는 답변 차단
# 이 아래는 append-only 로 추가한 블록이다(위의 기존 테스트는 손대지 않았다).
#
# 배경: HCX-005 가 도구를 한 번도 부르지 않고 사전학습 지식으로 답해 버리는 경우가 있고, 실측에서
# 그런 답은 예외 없이 틀렸다(G-034 는 운용사를 약칭으로, G-010 은 DB 에 없는 상품과 수치를 지어냈다).
# 그래서 '조회 0회 상태의 final_answer' 는 받지 않고 한 라운드 더 돌린다.
from finagent.agent.orchestrator import FORCED_LOOKUP_MIN_REMAINING_SEC, LOOKUP_TOOLS


class _FullTap(StubLLM):
    """모델에게 되돌려 준 tool_result 를 (id, 내용, 오류여부) 그대로, user 메시지도 함께 모은다."""

    def __init__(self, script):
        super().__init__(script)
        self.tool_results: list[tuple[str, str, bool]] = []
        self.user_msgs: list[str] = []

    def start(self, system, tools, user_text):
        conv = super().start(system, tools, user_text)
        orig_tr, orig_user = conv.add_tool_results, conv.add_user
        conv.add_tool_results = lambda rs: (self.tool_results.extend(rs), orig_tr(rs))[1]
        conv.add_user = lambda t: (self.user_msgs.append(t), orig_user(t))[1]
        return conv

    @property
    def directives(self) -> str:
        return "\n".join(self.user_msgs)


def _rebudget(s, *, time_budget_sec=120, max_tool_rounds=4):
    """오케스트레이터의 시간·라운드 예산만 갈아끼운다.

    기존 ``_state`` 는 20초 예산이라 강제 재시도가 애초에 걸리지 않는다(그래서 기존 테스트는
    이 변경의 영향을 받지 않는다). 여기서는 실제 운영값(120초)에 가까운 예산을 준다.
    """
    s.orchestrator.settings = Settings(
        _env_file=None, llm_provider="stub",
        time_budget_sec=time_budget_sec, max_tool_rounds=max_tool_rounds,
    )
    return s


_ASK = "삼성 KODEX 2차전지산업의 운용사 이름을 알려줘"          # G-034 와 같은 모양의 질문
_SQL = "SELECT pd_nm, manager_name FROM v_etf_kr WHERE pd_nm LIKE '%2차전지%'"


def test_lookup_tools_matches_the_registry(tiny_store, tmp_path):
    """LOOKUP_TOOLS 가 실제 도구 목록과 어긋나면(도구 추가·개명) 여기서 걸린다.

    허용 목록으로 적은 이유는, 모델이 존재하지 않는 이름의 도구를 부른 것을 '조회했다'로
    세면 안 되기 때문이다. 대신 목록이 낡을 수 있으므로 레지스트리와 대조해 둔다.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    names = {t.name for t in s.tools.specs()}
    assert "final_answer" in names
    assert LOOKUP_TOOLS == names - {"final_answer"}
    assert FORCED_LOOKUP_MIN_REMAINING_SEC > 14   # 기존 '시간 예산 임박' 문턱보다는 커야 한다


async def test_final_answer_without_any_lookup_is_sent_back_once(tiny_store, tmp_path):
    """G-034 재발 방지: 조회 0회로 낸 답변은 받지 않고 한 라운드 더 돌린다."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-034", _ASK)

    assert "제출 거부" in resp.think_trace
    assert "삼성자산운용" in resp.answer                  # 조회 결과에서 가져온 값으로 답이 바뀌었다
    assert resp.retrieved_context.count("[근거") == 1     # 조회가 실제로 일어났다
    # 지시는 '사용 시점의 사실'로, user 메시지로 전달한다
    assert "답변 전에 최소 한 번은 데이터를 조회해야 합니다" in llm.directives
    assert "sql_query" in llm.directives
    assert "공식 명칭" in llm.directives


async def test_forced_retry_answers_every_pending_tool_call(tiny_store, tmp_path):
    """CLOVA(OpenAI 호환)는 assistant 의 tool_call 마다 결과를 요구한다 — 하나라도 비면 다음 요청이 400 이다.

    final_answer 와 같은 턴에 딸려 와서 **실행되지 않은** 호출까지 전부 결과를 돌려주는지 본다.
    실행되지 않은 호출은 조회로 세지 않는다(결과를 못 봤으면 근거가 없는 것과 같다) — 그래서
    이 경우에도 재시도가 걸린다.
    """
    script = [
        Turn("", [ToolCall("x1", "sql_query", {"sql": _SQL}),
                  ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-034b", _ASK)

    by_id = {i: c for i, c, _ in llm.tool_results}
    assert "f1" in by_id and "x1" in by_id            # 대기 중이던 두 호출 모두 결과를 받았다
    assert "제출이 거부되었습니다" in by_id["f1"]
    assert "실행되지 않았습니다" in by_id["x1"]
    assert "제출 거부" in resp.think_trace and "삼성자산운용" in resp.answer


async def test_forced_retry_happens_at_most_once(tiny_store, tmp_path):
    """두 번째에도 도구를 안 부르면 그대로 받는다.

    무한 루프로 응답시간(주최측 권장 60초)을 넘기는 쪽이 근거 없는 답 하나보다 나쁘다.
    """
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "그래도 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f3", "final_answer", {"status": "answered", "answer": "세번째답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-034c", _ASK)

    assert "그래도 삼성입니다" in resp.answer
    assert "세번째답변은오면안된다" not in resp.answer          # 라운드가 더 돌지 않았다
    assert resp.think_trace.count("제출 거부") == 1
    assert sum(1 for m in llm.user_msgs if "답변 전 데이터 조회 필수" in m) == 1


async def test_no_forced_retry_when_a_lookup_already_ran(tiny_store, tmp_path):
    """이미 조회한 뒤의 final_answer 에는 재시도가 붙으면 안 된다 — 정상 문항에 +11초를 물리지 않는다."""
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-010", _ASK)

    assert "제출 거부" not in resp.think_trace
    assert not any("답변 전 데이터 조회 필수" in m for m in llm.user_msgs)
    assert "삼성자산운용" in resp.answer and "이답변은오면안된다" not in resp.answer


async def test_table_info_alone_counts_as_a_lookup(tiny_store, tmp_path):
    """table_info 는 값이 아니라 스키마를 돌려주지만 조회로 친다.

    기준일·허용값처럼 table_info 만으로 답이 되는 문항이 있어서다. 이 판정을 바꾸려면
    orchestrator.LOOKUP_TOOLS 에서 한 줄만 빼면 된다.
    """
    script = [
        Turn("", [ToolCall("t1", "table_info", {"table": "v_etf_kr"})], "tool_use"),
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "국내ETF 기준일은 2026-08-21 입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-TI", "국내 ETF 데이터의 기준일은 언제인가요?")
    assert "제출 거부" not in resp.think_trace
    assert "2026-08-21" in resp.answer


async def test_forced_retry_applies_to_cannot_confirm_too(tiny_store, tmp_path):
    """확인불가도 '조회한 뒤의 거절'이라야 근거가 선다.

    동시에, 조회 결과가 0건이면 답을 지어내지 말고 cannot_confirm 을 유지하라는 지시가
    함께 나가야 한다 — 이 변경으로 확인불가 문항이 답을 짓기 시작하면 순손실이다.
    """
    sql = "SELECT pd_abrv_nm FROM v_etf_kr WHERE manager_name='없는운용사'"
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "cannot_confirm", "answer": "제공된 데이터로는 확인할 수 없습니다."})], "tool_use"),
        Turn("", [ToolCall("c1", "sql_query", {"sql": sql})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "cannot_confirm", "answer": "조회 결과가 0건이라 제공된 데이터로는 확인할 수 없습니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-NC", "없는운용사 ETF 알려줘")

    assert "제출 거부" in resp.think_trace
    assert "확인할 수 없습니다" in resp.answer                       # 거절이 유지된다
    assert "status=cannot_confirm 을 그대로 유지" in llm.directives  # 지어내지 말라는 지시가 나갔다
    assert "답을 지어내지 마세요" in llm.directives


async def test_forced_retry_skipped_when_time_budget_is_short(tiny_store, tmp_path):
    """남은 예산이 재시도 비용을 못 대면 되돌리지 않고 있는 답을 받는다."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path), time_budget_sec=20)
    resp = await s.orchestrator.answer("G-T1", _ASK)

    assert "강제 재시도 생략" in resp.think_trace
    assert "삼성입니다" in resp.answer and "이답변은오면안된다" not in resp.answer
    assert not llm.user_msgs


async def test_forced_retry_skipped_when_no_round_is_left_to_run_tools(tiny_store, tmp_path):
    """시간이 남아도 도구를 돌릴 라운드가 없으면 되돌려 봐야 근거가 생기지 않는다."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path), time_budget_sec=120, max_tool_rounds=1)
    resp = await s.orchestrator.answer("G-T2", _ASK)

    assert "강제 재시도 생략" in resp.think_trace
    assert "삼성입니다" in resp.answer and "이답변은오면안된다" not in resp.answer


async def test_rejected_answer_is_kept_if_the_retry_round_produces_nothing(tiny_store, tmp_path):
    """되돌린 뒤 모델이 오류·침묵으로 아무것도 못 내놓아도 답변을 잃지 않는다.

    되돌린 final_answer 를 그냥 버리면, 재시도 라운드가 실패했을 때 status·evidence_refs 까지
    통째로 날아가고 fallback 문구만 남는다. 근거가 없어도 원래 있던 답이 빈손보다 낫다.
    """
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("boom", [], "error"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-T3", _ASK)

    assert "제출 거부" in resp.think_trace
    assert "처음 제출된 답변을 사용" in resp.think_trace
    assert "삼성입니다" in resp.answer
