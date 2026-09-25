"""Tool registry, orchestrator and API contract tests with a scripted stub LLM on a tiny store."""
import json
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finagent.agent.answer import (
    CANNOT_PREFIX_EN,
    CANNOT_PREFIX_KO,
    Evidence,
    fallback_answer,
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
    assert resp.question_id == "Q-001" and "[Evidence 1]" in resp.retrieved_context and "v_bond_kr" in resp.retrieved_context
    assert "final_answer(status=answered)" in resp.think_trace and "2종" in resp.answer
    assert "'AA-' is valid" in s.llm.started[0][1]  # pre-analysis injected credit-grade hint


async def test_orchestrator_invalid_grade_forces_cannot_confirm(tiny_store, tmp_path):
    script = [Turn("", [ToolCall("c1", "final_answer", {"status": "answered", "answer": "AAAA 채권 3종이 있습니다."})], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("Q-X", "신용등급 AAAA인 채권 찾아줘")
    assert resp.answer.startswith("제공된 데이터로는 확인할 수 없습니다")
    assert "신용등급 'AAAA'는 존재하지 않는 값입니다. 유효한 신용등급: AAA, AA+" in resp.answer   # Korean question, Korean text
    assert "Ontology violation" in s.llm.started[0][1] and "AAAA" in s.llm.started[0][1]


async def test_orchestrator_invalid_grade_in_an_english_question(tiny_store, tmp_path):
    """The English credit-rating and risk-grade phrasings reach the same ontology check; the correction is English."""
    script = [Turn("", [ToolCall("c1", "final_answer", {"status": "answered", "answer": "There are 3 AAAA bonds."})], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("Q-XE", "Find bonds with a credit rating of AAAA")
    assert resp.answer.startswith(f"{CANNOT_PREFIX_EN} Credit rating 'AAAA' does not exist. Valid credit grades: AAA, AA+")
    assert "3 AAAA bonds" not in resp.answer and "corrected to cannot_confirm" in resp.think_trace
    o = s.orchestrator
    for q in ("Which bonds are rated AAAA?", "Show me bonds rated AAAA or higher", "List AAAA or higher bonds"):
        assert any("'AAAA' does not exist" in x for x in o.pre_analyze(q).invalid), q
    for q in ("Show me ETFs with risk grade 7", "Which funds have a risk level of 7?", "grade 7 ETFs"):
        assert any("Risk grade '7' does not exist" in x for x in o.pre_analyze(q).invalid), q
    # valid values pass, and a lower-case article or a product name is not a rating
    assert o.pre_analyze("How many KRW bonds are rated AA- or higher?").invalid == []
    assert "'AA-' is valid" in o.pre_analyze("How many KRW bonds are rated AA- or higher?").text
    assert o.pre_analyze("What is the fee of BondBloxx BB Rated USD High Yield Corp Bd ETF?").invalid == []
    assert o.pre_analyze("Which ETFs have risk grade 1?").invalid == []
    # Korean phrasing still behaves as before
    assert o.pre_analyze("위험등급 7등급 ETF 알려줘").invalid and o.pre_analyze("위험등급 1등급 ETF 알려줘").invalid == []


async def test_orchestrator_llm_error_falls_back(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([Turn("boom", [], "error")]), tmp_path)
    resp = await s.orchestrator.answer("Q-E", "아무 질문")
    assert "확인할 수 없습니다" in resp.answer and "LLM error" in resp.think_trace
    assert "(LLM 응답 없음)" in resp.answer                       # Korean question: Korean fallback


async def test_orchestrator_llm_error_falls_back_in_english(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([Turn("boom", [], "error")]), tmp_path)
    resp = await s.orchestrator.answer("Q-EE", "What is the total fee of KODEX 200?")
    assert resp.answer.startswith(CANNOT_PREFIX_EN) and "(no response from the LLM)" in resp.answer


async def test_answer_language_line_reaches_the_model(tiny_store, tmp_path):
    """The language is stated in the first user message; the (cached) system prompt stays identical."""
    llm = StubLLM(script_factory=lambda q: [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "x"})], "tool_use")])
    s = _state(tiny_store, llm, tmp_path)
    await s.orchestrator.answer("L-1", "Which domestic ETFs hold 에코프로비엠?")
    await s.orchestrator.answer("L-2", "에코프로비엠을 편입한 국내 ETF는 몇 개인가요?")
    (sys_en, user_en), (sys_ko, user_ko) = llm.started
    assert user_en.startswith("Question: Which domestic ETFs hold 에코프로비엠?\nAnswer language: English\n")
    assert user_ko.startswith("Question: 에코프로비엠을 편입한 국내 ETF는 몇 개인가요?\nAnswer language: Korean\n")
    assert sys_en == sys_ko and "Answer language" in sys_en            # the rule is in the system prompt, the value is not
    assert "answer language stated in the user message" in sys_en


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
        assert r2.status_code == 200 and set(r2.json()) == set(body)
        assert r2.json()["answer"] == "The question parameter is required. E.g. /answer?question_id=Q-001&question=your+question"
        assert r2.json()["think_trace"] == "Input validation: question parameter missing"
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
        assert r.json()["think_trace"].startswith("Internal error: RuntimeError")
        assert r.json()["answer"].startswith(CANNOT_PREFIX_EN) and "(internal_error)" in r.json()["answer"]
        # the fallback follows the question's language
        r = client.get("/answer", params={"question_id": "Q", "question": "판매중인 국내 ETF는 몇 개인가요?"})
        assert r.status_code == 200 and r.json()["answer"].startswith(CANNOT_PREFIX_KO)
        # the stream route reports a missing question in English too
        assert "The question parameter is required." in client.get("/answer/stream").text


def test_answer_helpers():
    ev = Evidence("sql", "v_bond_kr", "국내채권마스터", "2026-08-21", "SELECT 1", 2, ["KR1 a", "KR2 b"])
    txt = format_evidence([ev], ["v_bond_kr"])
    assert txt.startswith("[Evidence 1] 국내채권마스터(v_bond_kr) · base date 2026-08-21 · 2 rows") and "SELECT 1" in txt
    assert format_evidence([]) == "No data referenced"
    assert normalize_answer("cannot_confirm", "해당 상품 없음", "ko").startswith("제공된 데이터로는 확인할 수 없습니다")
    assert normalize_answer("clarify", "어느 것", "ko") == "어느 것 어떤 조건을 말씀하시는지 알려주시겠어요?"
    assert json.loads('{"a": 1}')["a"] == 1


def test_answer_helpers_in_english():
    """The code's own answer text follows the answer language; English is the default."""
    assert normalize_answer("cannot_confirm", "No such product.") == f"{CANNOT_PREFIX_EN} No such product."
    assert normalize_answer("clarify", "Which one") == "Which one Could you tell me which condition you mean?"
    assert normalize_answer("clarify", "Did you mean KODEX 200?") == "Did you mean KODEX 200?"
    # an English answer that already says it cannot be confirmed gets no second prefix
    for said in ("The 2027 return cannot be confirmed from a base-date snapshot.",
                 "I cannot confirm that any public fund holds 삼성전자."):
        assert normalize_answer("cannot_confirm", said) == said, said
    # and the Korean acknowledgement still counts for an English-language call
    assert normalize_answer("cannot_confirm", "제공된 데이터로는 확인할 수 없습니다.") == "제공된 데이터로는 확인할 수 없습니다."
    assert normalize_answer("answered", "") == CANNOT_PREFIX_EN
    assert normalize_answer("answered", "", "ko") == CANNOT_PREFIX_KO
    # a leaked call whose body repeats the English prefix is de-duplicated like the Korean one
    leak = f'{CANNOT_PREFIX_EN} final_answer(status="cannot_confirm", answer="{CANNOT_PREFIX_EN} No data.")'
    assert strip_tool_call_syntax(leak) == f"{CANNOT_PREFIX_EN} No data."


def test_fallback_answer_follows_the_language():
    en = fallback_answer("timeout")
    assert en.startswith(CANNOT_PREFIX_EN) and "(timeout)" in en and "try again" in en
    ko = fallback_answer("시간 초과", "ko")
    assert ko == (f"{CANNOT_PREFIX_KO} 시스템 처리 중 문제가 발생하여 답변을 생성하지 못했습니다(시간 초과). "
                  "질문을 조금 더 구체적으로 바꾸어 다시 시도해 주세요.")


def test_leaked_tool_call_syntax_is_unwrapped_not_deleted():
    """Real cases where the model, instead of *calling* final_answer, wrote the call out as text inside answer.

    6 of the 54 recorded answers did this.  Only the shell may be peeled; the body inside must survive —
    deleting it all loses the answer, leaving it makes the user read tool syntax.
    """
    # G-006: the call is the whole answer
    assert strip_tool_call_syntax(
        'final_answer(status="completed", answer="국내 ETF 중 위험등급 1등급이면서 순자산 1조원 이상인 상품은 총 22개입니다.")'
    ) == "국내 ETF 중 위험등급 1등급이면서 순자산 1조원 이상인 상품은 총 22개입니다."

    # G-016: a call appended after a real sentence.  Both the sentence before and the body inside must stay.
    g016 = ('제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", '
            'answer="제공된 데이터에서는 신용등급이 \'AAAA\'인 채권을 찾을 수 없습니다. '
            '유효한 신용등급으로는 AAA, AA+, AA0, AA- 등이 있으니 이러한 등급을 가진 채권을 찾으시면 도움이 될 것 같습니다.")')
    out = strip_tool_call_syntax(g016)
    assert "final_answer" not in out and 'status=' not in out and 'answer="' not in out
    assert out.startswith("제공된 데이터로는 확인할 수 없습니다. 제공된 데이터에서는 신용등급이 'AAAA'인 채권을 찾을 수 없습니다.")
    assert out.endswith("도움이 될 것 같습니다.")

    # G-043: \n escapes inside the argument become real line breaks (so a list does not collapse onto one line)
    g043 = ('final_answer(status="completed", answer="판매 중인 국내 ETF와 공모펀드를 통틀어 위험등급 1등급으로 '
            '분류된 상품은 총 1,097개입니다.\\n\\n- 국내 ETF: 223개\\n- 공모펀드: 874개")')
    out = strip_tool_call_syntax(g043)
    assert "\\n" not in out and "- 국내 ETF: 223개\n- 공모펀드: 874개" in out

    # G-054: an announcement line + a code fence.  All three layers of shell must come off.
    g054 = ('final_answer 도구를 사용하여 답변을 제출하겠습니다:\n\n```plaintext\n'
            'final_answer(status="cannot_confirm", answer="제공된 데이터로는 총보수 정보를 찾을 수 없습니다.")\n```')
    assert strip_tool_call_syntax(g054) == "제공된 데이터로는 총보수 정보를 찾을 수 없습니다."

    # With a lower output cap the call can be cut off before it closes.  That fragment is peeled the same way.
    assert strip_tool_call_syntax(
        '제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", answer="제공된 데이터로는 특정 상품을'
    ) == "제공된 데이터로는 확인할 수 없습니다. 제공된 데이터로는 특정 상품을"

    # ---- must not be touched: normal answers containing parens and quotes (real records G-003 / G-009 / G-026) ----
    for legit in (
        "매수수익률이 수록된 채권 중 세후 운용수익률이 가장 높은 종목은 '국고채권 01500-5003(20-2)'이며, 수익률은 5.70%입니다.",
        "해외 ETF 중 운용규모(AUM)가 가장 큰 상품은 Vanguard 500 Index Fund;ETF - 997,351,790,000원 (9,973억 5,179만원)입니다.",
        # The numbered-list example was taken out of here.  Since answers go out as prose only,
        # normalize_answer strips the list markers and joins the items into one sentence, so == legit no
        # longer holds.  The expectation that only the markers vanish and no item is lost is checked by
        # test_flattened_list_supersedes_the_earlier_passthrough_expectation instead, together with the
        # original intent that strip_tool_call_syntax leaves that string alone.
        '위험등급은 "매우높은위험(1등급)" 형태의 라벨로 제공됩니다.',
    ):
        assert strip_tool_call_syntax(legit) == legit, legit
        assert normalize_answer("answered", legit) == legit, legit


async def test_nonexistent_product_is_not_answered_as_a_similar_one(tiny_store, tmp_path):
    """'KODEX AI로봇' does not exist.  If the model answers with KODEX 로봇액티브 instead, the server corrects it to a clarifying question."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "KODEX AI로봇의 순자산은 500억원이며 삼성자산운용이 운용합니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    pre = s.orchestrator.pre_analyze("KODEX AI로봇 ETF 정보 알려줘")
    assert pre.unresolved and pre.unresolved[0][0] == "KODEX AI로봇"
    assert "does not exist in the data" in pre.text
    resp = await s.orchestrator.answer("U-003", "KODEX AI로봇 ETF 정보 알려줘")
    assert "확인할 수 없습니다" in resp.answer
    assert "KODEX 로봇액티브" in resp.answer and "이 중 찾으시는 상품이 있나요?" in resp.answer  # asks back, offering candidates
    assert "corrected to a clarifying question" in resp.think_trace


async def test_nonexistent_product_correction_in_english(tiny_store, tmp_path):
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "KODEX AI로봇 has net assets of 50 billion won and is managed by 삼성자산운용.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("U-003E", "Tell me about the KODEX AI로봇 ETF")
    assert resp.answer.startswith("A product exactly matching 'KODEX AI로봇' cannot be confirmed in the provided data.")
    assert "KODEX 로봇액티브" in resp.answer and "Is one of these the product you are looking for?" in resp.answer
    assert "(For reference, the following was looked up on the basis of a similar product)" in resp.answer
    assert "corrected to a clarifying question" in resp.think_trace


async def test_clarify_suffix_without_candidates_follows_the_language(tiny_store, tmp_path):
    """No similar product to offer: the code's own question back comes in the question's language."""
    from finagent.agent.orchestrator import PreAnalysis

    for q, want in (("KODEX 없는상품 순자산 알려줘", "찾으시는 상품명을 다시 확인해 주시겠어요?"),
                    ("What are the net assets of KODEX 없는상품?", "Could you check the product name you are looking for?")):
        script = [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "500억원"})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        s.orchestrator.pre_analyze = lambda _q: PreAnalysis("[x]", [], ["KODEX 없는상품"], [], [("KODEX 없는상품", [])])
        resp = await s.orchestrator.answer("U-NC", q)
        assert want in resp.answer, q


async def test_real_product_is_not_falsely_flagged(tiny_store, tmp_path):
    """A manager prefix or a '펀드' (fund) suffix is no reason to call a product nonexistent (false-positive guard)."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    assert s.orchestrator.pre_analyze("KODEX 로봇액티브 순자산 알려줘").unresolved == []
    assert s.orchestrator.pre_analyze("삼성 KODEX 2차전지산업의 위험등급은?").unresolved == []


async def test_acknowledged_decline_is_left_alone(tiny_store, tmp_path):
    """If the model already said 'there is none', the server adds nothing."""
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "cannot_confirm",
        "answer": "KODEX AI로봇이라는 상품은 확인할 수 없습니다. 혹시 KODEX 로봇액티브를 말씀하시나요?",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("U-003", "KODEX AI로봇 ETF 정보 알려줘")
    assert resp.answer.count("확인할 수 없습니다") == 1
    assert "corrected to a clarifying question" not in resp.think_trace


async def test_acknowledged_decline_in_english_is_left_alone(tiny_store, tmp_path):
    """The English acknowledgements ('could not be found', 'did you mean') count, even with status=answered."""
    for text in ("KODEX AI로봇 could not be found in the data. Did you mean KODEX 로봇액티브?",
                 "There is no such product; are you looking for KODEX 로봇액티브?"):
        script = [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": text})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        resp = await s.orchestrator.answer("U-003A", "Tell me about the KODEX AI로봇 ETF")
        assert resp.answer == text, text
        assert "corrected to a clarifying question" not in resp.think_trace


def test_forward_looking_detector_covers_the_class_not_one_phrasing():
    """A judgement grounded in the fact that the ontology has no future-value column.  Values the data holds must not be touched."""
    for q in ("TIGER 반도체 ETF의 내년 예상 수익률", "이 펀드 앞으로 오를까요?",
              "향후 주가 전망 알려줘", "내년에 수익률이 얼마나 될까"):
        assert asks_forward_looking(q), q
    # the guaranteed yield to maturity (exrt_grte_ern_r) is a value the data really holds, so it is no forecast question
    for q in ("만기보장수익률이 가장 높은 채권", "보장수익률 알려줘",
              "최근 1년 수익률 상위 3개", "작년 수익률", "만기가 2027년인 채권",
              # the '미래' (future) in manager and product names is not a future point (9,816 product names contain '미래')
              "미래에셋 ETF의 1년 수익률 알려줘", "BNK 미래전략기술액티브의 수익률은?",
              "작년 연말 수익률 알려줘",
              # '규모' (size) names a magnitude, not a value.  Added as a value word, it caught this ordinary
              # question as '미래 … 규모'.  G-045 is caught by '순자산' alone, so '규모' is not needed.
              "미래 성장 규모가 큰 테마", "미래 성장성이 높은 산업 관련 ETF",
              # English product names are not forecasts either
              "RISE 200의 1년 수익률은?", "Future Mobility ETF의 순자산은?"):
        assert not asks_forward_looking(q), q


def test_forward_looking_detector_in_english():
    """The same four families in English: forecast word, future-time word, future year, rise/fall."""
    for q in ("Next year, what is the expected return of TIGER 반도체?",
              "Will KODEX 200 rise next year?",
              "What is the price outlook for 삼성전자?",
              "How much will RISE 200 go up?",
              "Is 삼성전자 likely to fall?",
              "What will the 1-year return of KODEX 200 be in 3 years?",
              "What will the net assets of KODEX 200 be at the end of 2027?",
              "Forecast the dividend of TIGER 200 for 2028",
              "What are the projected returns of KODEX 200?"):
        assert asks_forward_looking(q, 2026), q
    for q in ("What is the 1-year return of KODEX 200?",
              "Which bond has the highest guaranteed yield to maturity?",
              "How many KRW bonds rated AAA mature in 2027?",
              "Which ETFs will be listed in 2027?",
              "What was the return of KODEX 200 in 2025?",
              "What is the dividend frequency of RISE 200?",
              "Could you tell me which ETF rose the most last month?",
              "Which futures ETFs have the lowest fees?",
              "Which Mirae Asset funds have the highest 1-year return?",
              # product names are capitalised: 'Future', 'Outlook', '2030' inside a name are not forecasts
              "What is the AUM of the Goldman Sachs Future Tech Leaders Equity ETF?",
              "What is the AUM of RH Tactical Outlook ETF?",
              "What is the AUM of the iShares iBonds Dec 2030 Term Corporate ETF?",
              "What is the net asset value of TIGER TDF 2045?"):
        assert not asks_forward_looking(q, 2026), q
    # the base year comes from the store meta, for English years too
    assert not asks_forward_looking("What will the net assets of KODEX 200 be in 2027?", 2027)


def test_future_year_counts_as_a_future_time_reference():
    """G-045 regression: a future point written only as a year, like '2027년 말', counts as a future-time expression.

    Only a year after the base year (the year of the data's base_date) is in the future.  The trap is
    G-002 — "만기가 2027년인 채권" (bonds maturing in 2027) names a future year but asks about a
    **recorded date field**, mat_date, and must be answerable.  A year attached to a date field rather
    than a value word does not count.
    """
    for q in ("KODEX 200의 2027년 말 예상 순자산 규모는 얼마인가요?",
              "2028년 순자산 규모를 알려줘", "2030년 수익률이 어떻게 될까요?"):
        assert asks_forward_looking(q, 2026), q
    # ★ future year + date field = a lookup, not a forecast.  Firing here kills answerable questions.
    for q in ("구매 가능한 원화채권 중 신용등급 AAA이고 만기가 2027년인 종목 수를 알려줘",
              "만기 2029년 채권 목록", "2027년 상장 예정 ETF가 있나요?",
              "2026년 말 순자산 규모는?",   # at or before the base year = past/current snapshot
              "2025년 수익률 알려줘"):
        assert not asks_forward_looking(q, 2026), q
    # the base year comes from the store meta — when it moves, the judgement must move with it
    assert asks_forward_looking("2027년 순자산 규모", 2026)
    assert not asks_forward_looking("2027년 순자산 규모", 2027)


def test_holdings_scope_guard_fires_only_where_there_is_no_holdings_data():
    """G-040/G-041 regression: holdings data exists only for 1,162 domestic ETFs.

    A holdings question narrowed to foreign ETFs, public funds, private funds or ETNs has no such
    relation in the data and cannot be answered.  Conversely a domestic-ETF holdings question is a
    normal answer and must never be touched.
    """
    for q in ("엔비디아를 편입한 해외 ETF를 알려줘",
              "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
              "삼성전자를 보유한 공모펀드를 알려줘",
              "테슬라를 포함한 미국 ETF가 있나요?",
              "카카오를 편입한 ETN을 알려줘",
              # English
              "Which foreign ETFs hold NVIDIA?",
              "Which public funds hold 삼성전자?",
              "Does any mutual fund contain SK하이닉스?",
              "Which US ETFs include Tesla?",
              "Which ETNs hold 카카오?",
              "What are the top holdings of the overseas ETF QQQ?"):
        assert asks_holdings_without_data(q), q
    # ★ answerable questions — firing here crushes a normal answer into cannot-confirm
    for q in ("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘",
              "SK하이닉스를 편입하고 있는 국내 ETF는 몇 개인가요?",
              "두산에너빌리티를 보유한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘",
              "국내 ETF 가운데 삼성전자 비중이 가장 높은 상품은 무엇인가요?",
              "에코프로비엠을 편입한 ETF는 몇 개인가요?",          # no scope word, no firing
              "국내 ETF와 공모펀드를 합쳐서 1년 수익률이 가장 높은 상품 3개를 알려줘",  # no holding word
              "판매중인 상품 기준으로 채권에 투자하는 국내 ETF와 채권형 공모펀드는 각각 몇 개인가요?",
              "해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘",
              # English
              "Which domestic ETFs hold 삼성전자?",
              "Which Korean ETFs and public funds hold 삼성전자?",
              "How many ETFs hold 에코프로비엠?",
              "Which public funds have the highest 1-year return?",
              "Show me foreign ETFs including leveraged ones",
              "Tell us which ETFs hold 삼성전자"):
        assert not asks_holdings_without_data(q), q


async def test_fund_holdings_answer_is_rewritten_to_cannot_confirm(tiny_store, tmp_path):
    """G-041 regression: v_fund_pub has no holdings column.

    search_products returns only name/meaning similarity, so it cannot establish 'holds that stock'.
    When the model asserts it anyway, the server lowers the answer to cannot-confirm and states that
    there is no evidence.
    """
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 KB 만기가 있는 채권 EMP입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("H-001", "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?")
    assert resp.answer.startswith(CANNOT_PREFIX_KO)
    assert "국내 ETF 1,162 종목" in resp.answer
    assert "corrected to cannot_confirm" in resp.think_trace


async def test_fund_holdings_answer_is_rewritten_in_english(tiny_store, tmp_path):
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "The public fund holding SK하이닉스 with the largest net assets is KB 만기가 있는 채권 EMP.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("H-001E", "Which public fund holding SK하이닉스 has the largest net assets?")
    assert resp.answer.startswith(CANNOT_PREFIX_EN)
    assert "exists only for 1,162 domestic ETFs" in resp.answer and "EMP" not in resp.answer
    assert "corrected to cannot_confirm" in resp.think_trace


async def test_domestic_etf_holdings_answer_is_left_alone(tiny_store, tmp_path):
    """A domestic-ETF holdings question is answerable — the guard must not touch it (false-positive guard)."""
    for q, text in (("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?", "에코프로비엠을 편입한 국내 ETF는 KODEX 2차전지산업 1개입니다."),
                    ("How many domestic ETFs hold 에코프로비엠?", "1 domestic ETF holds 에코프로비엠: KODEX 2차전지산업.")):
        script = [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": text})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        resp = await s.orchestrator.answer("H-002", q)
        assert resp.answer == text, q
        assert "corrected to cannot_confirm" not in resp.think_trace


async def test_past_return_is_not_served_as_a_forecast(tiny_store, tmp_path):
    """G-019 regression: the past 1-year return (du_er_1y) must not be presented as 'next year's expected return'.

    The number itself is a real value the tool produced, so the numeric cross-check passes.  What is
    wrong is the label, so the server lowers the status to cannot-confirm and keeps it, stated as past performance.
    """
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "KODEX 로봇액티브의 내년 예상 수익률은 260.41%입니다.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("F-001", "KODEX 로봇액티브의 내년 예상 수익률을 알려줘")
    assert resp.answer.startswith(CANNOT_PREFIX_KO)
    assert "미래에 실현될 값을 담은 항목이 없습니다" in resp.answer
    assert "보장하지 않으므로" in resp.answer
    assert "(참고)" in resp.answer and "260.41%" in resp.answer  # the lookup result stays, as past performance
    assert "corrected to cannot_confirm" in resp.think_trace


async def test_past_return_is_not_served_as_a_forecast_in_english(tiny_store, tmp_path):
    script = [Turn("", [ToolCall("c", "final_answer", {
        "status": "answered",
        "answer": "The expected return of KODEX 로봇액티브 next year is 260.41%.",
    })], "tool_use")]
    s = _state(tiny_store, StubLLM(script), tmp_path)
    resp = await s.orchestrator.answer("F-001E", "Next year, what is the expected return of KODEX 로봇액티브?")
    assert resp.answer.startswith(CANNOT_PREFIX_EN)
    assert "has no item holding a value to be realised in the future" in resp.answer
    assert "(For reference)" in resp.answer and "260.41%" in resp.answer
    assert "corrected to cannot_confirm" in resp.think_trace


async def test_guaranteed_yield_question_is_still_answered(tiny_store, tmp_path):
    """The guaranteed yield to maturity is a column in the data — the forecast guard must not block this question."""
    for q, text in (("만기보장수익률이 가장 높은 채권 알려줘", "만기보장수익률이 가장 높은 채권은 한국전력공사채권1012입니다."),
                    ("Which bond has the highest guaranteed yield to maturity?",
                     "The bond with the highest guaranteed yield to maturity is 한국전력공사채권1012.")):
        script = [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": text})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        resp = await s.orchestrator.answer("F-002", q)
        assert resp.answer == text, q
        assert "corrected to cannot_confirm" not in resp.think_trace


async def test_count_star_on_bonds_reports_product_count(tiny_store, tmp_path):
    """Asking 'how many' with COUNT(*) returns a row count.  The real bond count must come with it."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("sql_query", {"sql": "SELECT COUNT(*) FROM v_bond_kr WHERE is_sellable=1"})
    assert not r.is_error
    assert "row count" in r.content and "number of bonds" in r.content

    # no warning on a query that already uses DISTINCT
    ok = await s.tools.execute("sql_query", {"sql": "SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1"})
    assert "row count" not in ok.content

    # nor on a listing query
    lst = await s.tools.execute("sql_query", {"sql": "SELECT pd_no, pd_nm FROM v_bond_kr LIMIT 2"})
    assert "row count" not in lst.content


def test_request_prompt_stays_within_token_budget(tiny_store, tmp_path):
    """The size actually sent every round = system prompt + tool definitions.

    What the real constraint was (competition build, HyperCLOVA X):
      * The organisers set no token limit.  The only stated constraint is 300 seconds per question.
      * The real constraint was CLOVA's tokens per minute (TPM).  A 429 was actually observed at 13,389
        tokens; at that size only two calls per minute went through and 21 of the 22 evaluation
        questions failed with 429.
      * The common belief that a bigger prompt is slower was refuted by measurement.  Varying the prompt
        over 599 / 5,591 / 8,789 / 13,495 tokens, the median first-call latency was 2.90 / 3.65 / 2.94 /
        3.82 seconds, within noise.  Size does not cause latency, so the basis of the limit was never the
        300-second budget, only TPM.

    What to measure: the system prompt alone understates what is sent, because the 8 tool schemas
    (about 2,594 tokens serialized in the OpenAI-compatible wire format) go out every round too.  On
    tiny_store, system prompt 9,152 + tools 2,594 = 11,746 tokens was one request.
    (Adding the "what this data is" fact section took it from 8,754 -> 9,152; deleting two duplicate
     holdings notes kept that to +398 tokens.)

    Basis of the 12,000 limit: this test measures on tiny_store, so it is smaller than production.  The
    production prompt on the real store, value_hints included, was 9,584 tokens, a request of 12,178
    tokens — about 432 more than measured here.  Production exceeded the limit, but the basis of the
    limit was TPM, not the number 12,000: that was still 1,211 tokens below the 13,389 where 429 was
    observed.  The test had about 254 tokens of headroom left.  Before raising a limit, look first at
    trimming CORE_COLUMNS' rarely used columns or value_hints.

    English edition: every number above comes from the Korean build and was estimated as len/1.6, a ratio
    for HCX's tokenizer on Korean-heavy text.  That ratio overcounts English roughly 2.5-fold, so the
    estimate is now per script: about 4 characters per token for Latin text and about 1 token per Hangul
    syllable (Korean data values, labels and the schema cards).  Under this estimate the Korean build's
    request on tiny_store is about 9,600 tokens and the English one about 8,500.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    wire_tools = json.dumps(
        [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
         for t in s.tools.specs()],
        ensure_ascii=False,
    )

    def approx(text: str) -> float:
        hangul = sum(1 for ch in text if "가" <= ch <= "힣")
        return hangul + (len(text) - hangul) / 4

    prompt_tokens = approx(s.orchestrator.system_prompt)
    tool_tokens = approx(wire_tools)
    approx_tokens = prompt_tokens + tool_tokens
    # Limit 12,000 -> 14,000 (Korean build, len/1.6).  The limit came from the assumption that a bigger
    # prompt is slower, and measurement refuted it: over prompts of 599 / 5,591 / 8,789 / 13,495 tokens
    # the median first-call latency stayed flat at 2.90 / 3.65 / 2.94 / 3.82 seconds while the size grew
    # 22-fold (round trips are what cost time).  So the limit is an alarm that the prompt is not quietly
    # bloating, not a cost.  12,000 had shrunk to 10 tokens of headroom and broke with every added rule
    # line, and each time the pressure to shorten wording put the budget ahead of accuracy.  14,000 kept
    # the same alarm without that pressure.
    # English edition: 11,000 under the per-script estimate above (Korean build 9,611, English 8,526 at
    # the time of the switch) — the same alarm, with about 2,500 tokens of room.
    assert approx_tokens < 11000, (
        f"one request is too large: {approx_tokens:,.0f} tokens "
        f"(system prompt {prompt_tokens:,.0f} + tool definitions {tool_tokens:,.0f})"
    )
    # the key columns needed for judgements must still be on the cards
    for col in ("is_sellable", "has_buy_yield", "crd_grd", "is_on_sale", "fund_family", "expense_ratio_pct"):
        assert col in s.orchestrator.system_prompt, col
    # the full definitions are still available through table_info
    assert "table_info" in s.orchestrator.system_prompt


async def test_table_info_still_returns_full_card(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("table_info", {"table": "v_bond_kr"})
    assert not r.is_error and len(r.content) > len(s.orchestrator.system_prompt) / 8


# ------------------------------------------------- two devices that get the model out of a dead end
# (1) SQL errors say what is right  (2) repeating the same call gets said out loud

async def test_wrong_column_error_names_the_right_column(tiny_store, tmp_path):
    """G-007 regression: "no such column: pd_ticker" says only what is wrong.

    Pointing at the real column (v_etf_gl.ticker) lets the model fix it in one go, without guessing.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    r = await s.tools.execute("sql_query", {"sql": "SELECT pd_ticker, pd_nm FROM v_etf_gl LIMIT 3"})
    assert r.is_error
    assert "no such column: pd_ticker" in r.content      # the original message is kept
    assert "v_etf_gl.ticker" in r.content                # names the prefix confusion (pd_ added) exactly

    # an aliased JOIN query looks only at the view the alias points to (G-015's pd_no)
    j = await s.tools.execute("sql_query", {"sql": (
        "SELECT e.pd_no FROM v_etf_kr e JOIN v_etf_constituent c ON e.pd_itm_no=c.etf_itm_no LIMIT 3")})
    assert j.is_error and "pd_itm_no" in j.content and "v_etf_kr" in j.content

    # asking for a view that does not exist lists the 5 that do
    t = await s.tools.execute("sql_query", {"sql": "SELECT * FROM v_etf_korea LIMIT 1"})
    assert t.is_error and "v_etf_kr" in t.content
    for v in ("v_bond_kr", "v_etf_gl", "v_fund_pub", "v_etf_constituent"):
        assert v in t.content, v

    # fishing for the column list through SQL is blocked by the guard — name the tool to use instead
    p = await s.tools.execute("sql_query", {"sql": "PRAGMA table_info(v_etf_kr)"})
    assert p.is_error and "table_info" in p.content

    # a normal query is left alone
    ok = await s.tools.execute("sql_query", {"sql": "SELECT pd_abrv_nm FROM v_etf_kr LIMIT 1"})
    assert not ok.is_error and "instead of" not in ok.content

    # an error with no known fix is left as is
    syn = await s.tools.execute("sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE"})
    assert syn.is_error and "syntax error" in syn.content and "instead of" not in syn.content


async def test_wide_view_error_stays_bounded(tiny_store, tmp_path):
    """Listing every column of a wide view eats the context.  Only nearby candidates + the total count."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    wide = [f"col_{i}" for i in range(60)] + ["ticker"]
    s.tools._colcache = {"v_etf_gl": wide}   # the real store's v_etf_gl has 39 columns, v_etf_kr 59
    msg = s.tools._sql_error_hint("SELECT pd_ticker FROM v_etf_gl", "SQL error: no such column: pd_ticker")
    assert "v_etf_gl.ticker" in msg
    assert f"has {len(wide)} columns; names close to" in msg and "table_info('v_etf_gl')" in msg
    # does not dump the whole list (the English directive alone is ~300 characters; the 61 names would add ~430)
    assert "col_59" not in msg and len(msg) < 600


async def test_missing_column_says_what_to_do_not_just_what_is_wrong(tiny_store, tmp_path):
    """Regression for the 119-second question: right after du_er_1y FROM v_etf_gl failed, the model wrote an 'approach' instead of an answer.

    v_etf_gl has no 1-year return column, only du_er_1d (1 day).  The spelling is close, so a suggestion
    comes out, but it cannot be the answer — so besides 'what is right' the message needs 'so do this'.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    s.tools._colcache = {"v_etf_gl": ["ticker", "pd_nm", "is_on_sale", "du_er_1d", "du_last_aum"]}
    msg = s.tools._sql_error_hint(
        "SELECT ticker, du_er_1y FROM v_etf_gl WHERE is_on_sale=1 ORDER BY du_er_1y DESC LIMIT 10",
        "SQL error: no such column: du_er_1y",
    )
    assert "du_er_1d" in msg                          # existing behaviour: the close column is still named
    assert "does not mean the same meaning" in msg    # a close name != the same meaning
    assert "excluded" in msg                          # no substitution → exclude that product group
    assert "do not describe the method" in msg        # no plan; answer right away

    # the same directive is attached when there is no close candidate at all
    s.tools._colcache = {"v_etf_gl": ["ticker", "pd_nm"]}
    bare = s.tools._sql_error_hint("SELECT zzz_qqq FROM v_etf_gl", "SQL error: no such column: zzz_qqq")
    assert "excluded" in bare and "do not describe the method" in bare

    # not attached to failures that are not column errors
    syn = await s.tools.execute("sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE"})
    assert syn.is_error and "do not describe the method" not in syn.content


def test_call_key_defines_the_same_call():
    """Pin down the definition of 'the same call'.  Loosen it and legitimate retries get blocked too."""
    a = ToolCall("1", "sql_query", {"sql": "SELECT pd_nm FROM v_etf_kr WHERE manager_name='삼성자산운용'", "purpose": "1차"})
    b = ToolCall("2", "sql_query", {"sql": "select  pd_nm\nfrom v_etf_kr\nwhere manager_name = '삼성자산운용';", "purpose": "다시 시도"})
    c = ToolCall("3", "sql_query", {"sql": "select pd_nm from v_etf_kr where manager_name = '미래에셋자산운용'"})
    assert _call_key(a) == _call_key(b)   # differing only in case, newlines, operator spacing, semicolon, purpose = same call
    assert _call_key(a) != _call_key(c)   # a different literal value is a different call (literals are not lower-cased)
    assert _call_key(ToolCall("4", "search_products", {"query": "로봇"})) != \
        _call_key(ToolCall("5", "search_products", {"query": "로봇 ETF"}))
    assert _call_key(ToolCall("6", "final_answer", {"status": "answered"})) is None


class _Tap(StubLLM):
    """A StubLLM that collects the tool-result strings actually handed to the model, as is."""

    def __init__(self, script):
        super().__init__(script)
        self.tool_contents: list[str] = []

    def start(self, system, tools, user_text):
        conv = super().start(system, tools, user_text)
        orig = conv.add_tool_results
        conv.add_tool_results = lambda rs: (self.tool_contents.extend(c for _, c, _ in rs), orig(rs))[1]
        return conv


async def test_identical_call_is_re_executed_and_says_so(tiny_store, tmp_path):
    """G-015 regression: sending the same query three times gets told it is stuck.

    But the call is **not blocked** (G-040).  Skipping execution saves only a few milliseconds of tool
    time, and in exchange cuts off the model's attempt to get out of the dead end.  It runs as usual,
    and only 'which identical call this is' is appended to the result.
    """
    sql = "SELECT pd_no, pd_abrv_nm FROM v_etf_kr WHERE is_on_sale=1"   # pd_no is not a column
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
    s.rdb.query = lambda q: (executed.append(q), real(q))[1]   # counts how often the DB is really hit

    resp = await s.orchestrator.answer("R-001", "국내 ETF 목록 알려줘")

    assert sum(1 for q in executed if "pd_no" in q.lower()) == 3, executed   # all three run
    assert len(llm.tool_contents) == 3
    assert "identical call #" not in llm.tool_contents[0]
    assert "This is identical call #2 for this question" in llm.tool_contents[1]
    assert "This is identical call #3 for this question" in llm.tool_contents[2]
    assert "the same input gives the same result" in llm.tool_contents[1]
    assert "without running" not in llm.tool_contents[1]   # the note must not contradict what happened
    assert "table_info" in llm.tool_contents[1]             # a failed call is told to check the columns
    # the repeat still carries the first result (the error message) as is
    assert "no such column: pd_no" in llm.tool_contents[1]
    # and it stays in think_trace
    assert "identical call #2" in resp.think_trace and "identical call #3" in resp.think_trace
    assert "identical call 3 times → told to wrap up early" in resp.think_trace


async def test_zero_row_repeat_suggests_relaxing_conditions(tiny_store, tmp_path):
    """Repeated 0 rows get a concrete 'drop the conditions one at a time'.  The evidence is not counted twice."""
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
    assert sum(1 for q in executed if "없는운용사" in q) == 2   # re-running is not blocked
    assert "Drop the conditions one at a time" in llm.tool_contents[1]
    # run twice, but the evidence counts once — the same result must not print as two [Evidence] lines
    assert resp.retrieved_context.count("[Evidence") == 1


async def test_two_different_calls_both_run_untouched(tiny_store, tmp_path):
    """Nothing may happen to a normal question — two different calls both run."""
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
    assert "identical call" not in resp.think_trace
    assert not any("identical call" in c for c in llm.tool_contents)
    assert resp.retrieved_context.count("[Evidence") == 2


def test_closing_offer_is_stripped_but_facts_and_caveats_survive():
    """8 of the 54 recorded answers ended with "추가적인 정보가 필요하시면 …" (if you need more information …).

    That sentence carries no information and costs generation time (measured: a fixed 2.7 seconds + 63
    tokens/second).  Only 6 of the 54 are actually trimmed; the other 48 stay byte for byte.  What must
    not be deleted matters more — sentences with numbers, caveats like "투자 판단은 본인 책임" (investment
    decisions are your own responsibility), and cannot-confirm facts like "…없습니다" (there is none).
    """
    # sentences from the real records that are deleted (G-021 / G-022 / G-009 / G-043 / G-054)
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

    # ---- must not be touched ----
    keep = (
        # a caveat in the same sentence keeps the whole sentence (real records G-020 / G-037).
        "투자 판단은 본인의 책임 하에 이루어져야 하며, 추가적인 정보가 필요하신 경우 개별 ETF의 세부 사항을 확인하시기 바랍니다.",
        "투자 판단은 본인의 책임이며, 추가적인 정보는 제공된 도구들을 통해 더욱 상세히 확인할 수 있습니다.",
        # a sentence with numbers is the body of the answer
        "조건에 맞는 종목은 15,837개이며 그중 상위 5개는 아래와 같습니다.",
        "추가적인 정보로 순자산은 4조 1,402억원입니다.",
        # cannot-confirm and exclusion facts
        "해외ETF는 1년 수익률이 제공되지 않아 비교에서 제외했습니다.",
        "제공된 데이터로는 확인할 수 없습니다.",
        # a last sentence that is not mere guidance
        "이 상품의 총보수는 0.05%입니다.",
    )
    for t in keep:
        assert strip_closing_offer(t) == t, t
        assert normalize_answer("answered", t) == t, t

    # it is really wired into the pipeline, and it never produces an empty answer
    assert normalize_answer("answered", "총 22개입니다. 추가적인 정보가 필요하시면 문의해 주세요.") == "총 22개입니다."
    assert normalize_answer("answered", "추가적인 정보가 필요하시면 문의해 주세요.", "ko") == CANNOT_PREFIX_KO


def test_english_closing_offer_is_stripped_but_facts_and_caveats_survive():
    """The English counterpart: the same three keep-rules (numbers, caveats, cannot-confirm facts)."""
    for text, want in (
        ("The total fee of KODEX 200 is 0.15%. Let me know if you need anything else.",
         "The total fee of KODEX 200 is 0.15%."),
        ("15,837 bonds match. Feel free to ask if you have more questions!", "15,837 bonds match."),
        ("TIGER 200 is managed by 미래에셋자산운용. If you have more questions, please let me know.",
         "TIGER 200 is managed by 미래에셋자산운용."),
        ("The fund is 공모펀드. If you need further information, please do not hesitate to contact us.",
         "The fund is 공모펀드."),
    ):
        assert strip_closing_offer(text) == want, text
    for keep in (
        "Investment decisions are your own responsibility; feel free to ask for more details.",
        "If you need more details, the 1-year return is 12.3%.",
        "Foreign ETFs were excluded; let me know if you want them anyway.",
        "No further information is available.",
        "For more details, see the prospectus.",
    ):
        assert strip_closing_offer(keep) == keep, keep
    assert normalize_answer("answered", "Let me know if you need anything else.") == CANNOT_PREFIX_EN


async def test_graph_neighbors_shows_few_labels_but_the_true_total(tiny_store, tmp_path):
    """Print only 8 labels, but report the total as is.

    Reporting the truncated count as the total was a real bug before (에코프로비엠's 42 reported as the 60 cap).
    40 labels were 1,501 characters for one 삼성전자 call, and that text is re-sent in every later round.
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
    assert "229 total" in r.content and r.evidence.row_count == 229  # the total is never truncated
    assert r.content.count("ETF") == GRAPH_LABELS_SHOWN == 8         # only 8 labels
    assert "ETF007" in r.content and "ETF008" not in r.content
    # the label line is still cut with "...", but a line of SQL lookup IDs follows it.
    labels_line = r.content.splitlines()[0]
    assert "8 examples shown" in labels_line and labels_line.rstrip().endswith("...")
    assert "lookup IDs" in r.content and "pd_itm_no IN (" in r.content
    assert not re.search(r"\btop\b", r.content, re.I)                # not sorted, so never called 'top'
    # Limit 200 -> 700.  8 labels alone do not let the model write SQL, so a lookup-ID line was added,
    # which makes the output longer (550 characters for a 229-item relation).  The limit exists to stop
    # what 삼성전자 did in the 40-label days, dumping 1,501 characters, and 550 is a third of that.  Output
    # that was merely short, without IDs, really failed to produce answers — 100 우주항공 theme products
    # came back and, names alone being unusable, the question ended as cannot-confirm.
    assert len(r.content) < 700

    # at or below the threshold nothing is cut and no "examples" note is added
    s.tools.kg = _FakeKG(3)
    r3 = await s.tools.execute("graph_neighbors", {"entity": "에코프로"})
    assert "3 total" in r3.content and "examples" not in r3.content and "..." not in r3.content
    assert r3.content.count("ETF") == 3


# ---------------------------------------------------------------- does the ontology own the judgement?
# The four checks below used to exist only as sentences in the prompt, and code cannot read sentences.
# The judgement now comes from the TTL, so editing the TTL must move these results with it.


def test_ontology_answers_which_product_classes_carry_holdings():
    """The product groups that carry holdings are read from the graph — not from a hard-coded list."""
    from finagent.ontology.loader import Ontology

    o = Ontology.load(ROOT / "ontology")
    assert [c.name for c in o.holding_classes()] == ["DomesticETF"]
    assert o.holding_view() == "v_etf_constituent"
    # a superclass is true when any subclass is: a question that only says 'ETF' must not be blocked
    assert o.class_has_holdings("ETF") and o.class_has_holdings("DomesticETF")
    # ETNs, funds and bonds have no holdings, all the way down
    for c in ("ETN", "DomesticETN", "Fund", "PublicFund", "PrivateFund", "ForeignETF", "DomesticBond"):
        assert not o.class_has_holdings(c), c


def test_ontology_scope_detection_agrees_with_the_regex_guard(tiny_store, tmp_path):
    """The ontology-driven product-group judgement gives the same answer as the regex guard (a split is an incident)."""
    from finagent.agent.orchestrator import mentions_holdings, product_groups

    s = _state(tiny_store, StubLLM([]), tmp_path)
    o = s.orchestrator.ontology
    for q in ("엔비디아를 편입한 해외 ETF를 알려줘",
              "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
              "삼성전자를 보유한 공모펀드를 알려줘",
              "테슬라를 포함한 미국 ETF가 있나요?",
              "카카오를 편입한 ETN을 알려줘",
              "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘",
              "에코프로비엠을 편입한 ETF는 몇 개인가요?",
              "해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘",
              # English: the classes' English labels and English spelling variants
              "Which foreign ETFs hold NVIDIA?",
              "Which public funds hold 삼성전자?",
              "Does any mutual fund contain SK하이닉스?",
              "Which US ETFs include Tesla?",
              "Which ETNs hold 카카오?",
              "Which domestic ETFs hold 삼성전자?",
              "Which Korean ETFs and public funds hold 삼성전자?",
              "How many ETFs hold 에코프로비엠?",
              "Which PLUS ETF holds 삼성전자?"):
        g = product_groups(q, o)
        blind = bool(g and mentions_holdings(q)) and not any(o.class_has_holdings(c.name) for c in g)
        assert blind == asks_holdings_without_data(q), (q, [c.name for c in g])
    # English labels are matched only in English questions: a Korean question is grouped exactly as before
    assert [c.name for c in product_groups("Vanguard 500 Index Fund의 운용사는?", o)] == []


async def test_fund_holdings_question_gets_the_fact_not_a_holdings_count(tiny_store, tmp_path):
    """G-041: a holdings question restricted to public funds must not be given 'held by 100 products'.

    That 100 was the count for domestic ETFs (and a value cut at the cap at that), and the model read it
    as the public-fund count.  Now the fact 'this product group has no holdings data' goes in instead of a count.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    for q in ("SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
              "Which public fund holding SK하이닉스 has the largest net assets?"):
        pre = s.orchestrator.pre_analyze(q)
        assert "held by" not in pre.text                                          # no count is given
        assert "Holdings data scope" in pre.text
        assert "공모펀드(v_fund_pub)" in pre.text and "v_etf_constituent" in pre.text
        assert "cannot be confirmed with the provided data" in pre.text           # not 0, but cannot confirm
        assert "the domain of the ontology's fp:hasConstituent = DomesticETF" in pre.text   # the basis is the ontology


async def test_answerable_etf_holdings_question_keeps_its_count(tiny_store, tmp_path):
    """A domestic-ETF holdings question is answerable — if the injection touched it, it would block the right answer."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("에코프로비엠을 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개")
    assert "Holdings data scope" not in pre.text
    assert "held by 1 국내ETF (graph)" in pre.text      # the small store's real holding count
    assert "Sort criterion '순자산'" in pre.text and "v_etf_kr.pd_net_tamt" in pre.text
    assert "Requested count: 3" in pre.text


async def test_english_ranking_criterion_resolves_through_the_korean_labels(tiny_store, tmp_path):
    """English ranking phrasings reach the same ontology lookup; the phrase is shown as asked."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("Which 3 domestic ETFs holding 에코프로비엠 have the largest net assets?")
    assert "Holdings data scope" not in pre.text and "held by 1 국내ETF (graph)" in pre.text
    assert "Sort criterion 'net assets'" in pre.text and "v_etf_kr.pd_net_tamt" in pre.text
    assert "Requested count: 3" in pre.text
    pre = s.orchestrator.pre_analyze("Top 10 across domestic ETFs, foreign ETFs and public funds by 1-year return")
    assert "Sort criterion '1-year return'" in pre.text
    assert "v_etf_kr.du_er_1y" in pre.text and "v_fund_pub.fd_yr1_ern_r" in pre.text
    assert "해외ETF(v_etf_gl) has no '1년 수익률' column" in pre.text and "du_er_1d" in pre.text
    assert "Requested count: 10" in pre.text
    pre = s.orchestrator.pre_analyze("What are the three largest domestic ETFs by net assets?")
    assert "Sort criterion 'net assets'" in pre.text and "Requested count: 3" in pre.text
    for q in ("Which domestic ETFs are sorted by total fee?", "List bonds ranked by coupon rate",
              "Which fund has the highest expense ratio?"):
        assert "Sort criterion" in s.orchestrator.pre_analyze(q).text, q
    # no ranking, or a ranking word without a measure: silence, as in Korean
    for q in ("How many KRW bonds are rated AA- or higher?", "Which company manages KODEX 로봇액티브?",
              "Which are the largest ETFs?", "What is the risk grade of 삼성 KODEX 2차전지산업?"):
        assert "Sort criterion" not in s.orchestrator.pre_analyze(q).text, q


async def test_recent_months_window_in_english(tiny_store, tmp_path):
    s = _state(tiny_store, StubLLM([]), tmp_path)
    assert "'last 6 months' = 2026-02-21 ~ 2026-08-21" in s.orchestrator.pre_analyze("최근 6개월 상장 ETF").text
    assert "'last 6 months' = 2026-02-21 ~ 2026-08-21" in s.orchestrator.pre_analyze(
        "Which ETFs were listed in the last 6 months?").text


async def test_period_specific_criterion_names_the_group_that_lacks_it(tiny_store, tmp_path):
    """Asking for the '1년 수익률' (1-year return) of three product groups at once injects the fact that 해외ETF lacks that column.

    This caused the incident of the 92-second 3-way UNION: it was written in the prompt as prose, but the
    example right above it was a 3-way UNION and the model followed the example.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("국내ETF, 해외ETF, 공모펀드를 통틀어 1년 수익률 기준 TOP10을 알려줘")
    assert "v_etf_kr.du_er_1y" in pre.text and "v_fund_pub.fd_yr1_ern_r" in pre.text
    assert "해외ETF(v_etf_gl) has no '1년 수익률' column" in pre.text
    assert "du_er_1d" in pre.text                     # and says what there is instead
    assert "Requested count: 10" in pre.text


async def test_bare_return_criterion_is_reported_as_ambiguous_not_guessed(tiny_store, tmp_path):
    """A bare '수익률' (return) has no period and maps to several columns.  Quietly picking one is not allowed."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    for q in ("수익률이 가장 높은 상품 10개를 알려줘", "Which 10 products have the highest return?"):
        pre = s.orchestrator.pre_analyze(q)
        assert "names no period or target" in pre.text, q
        assert "du_er_1y" in pre.text and "du_er_1m" in pre.text   # the candidates are given by name
        assert "do not pick one arbitrarily" in pre.text


async def test_label_miss_is_not_reported_as_missing_data(tiny_store, tmp_path):
    """'운용규모' (AUM) gets 0 label matches, yet v_etf_gl.du_last_aum is exactly that.

    Writing 'not in the data' because there is no label turns an answerable question into cannot-confirm.
    Candidates are given, searching comments and English labels too, but absence is never asserted.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    pre = s.orchestrator.pre_analyze("해외 ETF 중 운용규모(AUM)가 가장 큰 상품 3개의 이름을 알려줘")
    assert "du_last_aum" in pre.text
    assert "No label does not mean no data" in pre.text
    # not a single absence assertion may appear ('has no column whose ontology label matches exactly' is
    # candidate guidance, not an assertion of absence)
    for bad in ("has no column for this criterion", "Exclude it from the ranking", "cannot be confirmed"):
        assert bad not in pre.text, bad
    # for the same reason no absence is asserted for a view whose label differs (순자산 vs v_etf_gl.du_last_aum)
    for q in ("해외 ETF 중 순자산이 가장 큰 상품 3개", "Which 3 foreign ETFs have the largest net assets?"):
        pre2 = s.orchestrator.pre_analyze(q)
        assert "Do not conclude it is missing" in pre2.text and "du_last_aum(AUM)" in pre2.text, q
        assert "has no column for this criterion" not in pre2.text


async def test_questions_without_a_ranking_criterion_get_no_criterion_line(tiny_store, tmp_path):
    """Most questions have no ranking criterion.  Then the injection must stay completely silent."""
    s = _state(tiny_store, StubLLM([]), tmp_path)
    for q in ("현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요?",
              "판매중인 상품 기준으로 채권에 투자하는 국내 ETF와 채권형 공모펀드는 각각 몇 개인가요?",
              "구매 가능한 원화채권 가운데 신용등급이 A0인 채권은 서로 다른 종목(종목코드) 기준으로 몇 개인가요?",
              "KODEX 로봇액티브를 운용하는 자산운용사는 어디인가요?",
              "삼성 KODEX 2차전지산업의 위험등급은?"):
        assert "Sort criterion" not in s.orchestrator.pre_analyze(q).text, q


# ===================================================================== plain-prose answers
# The answer in the organisers' sample response is one line of plain text.  It is blocked once by prompt
# rule 12 and once by normalize_answer.  What is checked here is the code-side defence — prompt rules
# have a record of being ignored.
def test_markdown_emphasis_is_stripped_without_touching_numbers():
    assert normalize_answer("answered", "총보수는 **0.65**%입니다.") == "총보수는 0.65%입니다."
    assert normalize_answer("answered", "펀드는 '__한국투자네비게이터__'입니다.") == "펀드는 '한국투자네비게이터'입니다."
    assert normalize_answer("answered", "*가장 낮은* 값입니다.") == "가장 낮은 값입니다."
    # unpaired emphasis (truncated output) is not left behind either
    assert normalize_answer("answered", "총보수는 **0.65%입니다.") == "총보수는 0.65%입니다."
    # numbers, column names and punctuation are not markdown — they stay
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
    # The shape of G-021's real answer.  Only the markers go; all 4 codes must stay.
    out = normalize_answer("answered", (
        "이 ETF의 정보는 다음과 같습니다:\n\n"
        "- **상품명**: KODEX 미국휴머노이드로봇\n"
        "- **단축코드**: KR7433970001, KR7433980000, KR70082V0008, KR7434060000\n"
        "- **운용사**: 삼성자산운용\n"
    ))
    assert not out.startswith("-") and "\n" not in out and "**" not in out
    for keep in ("KR7433970001", "KR7433980000", "KR70082V0008", "KR7434060000", "삼성자산운용"):
        assert keep in out, keep


# ============================================ cannot_confirm: once refused, the answer ends there
# Incident: to "한화 에어로스페이스의 자회사를 편입한 ETF 중 …" the model said "찾을 수 없습니다" (cannot be
# found), then went on "그러나 한화 에어로스페이스를 편입한 ETF 목록을 제공할 수 있습니다" and listed 40.
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


def test_english_cannot_confirm_drops_the_substitute_product_listing():
    """The same cut for an English answer: refusal kept, the pivot sentence and the product rows dropped."""
    out = normalize_answer("cannot_confirm", (
        "The ETFs holding subsidiaries of 한화에어로스페이스 cannot be confirmed with the provided data. "
        "There is no holdings data linking subsidiaries to products.\n"
        "However, here are ETFs that hold 한화에어로스페이스 itself:\n\n"
        "1. **KODEX 200** (069500) - 25.47 trillion won\n"
        "2. **TIGER 200** (102110) - 10.43 trillion won\n"
        "3. **PLUS 고배당주** (161510) - 1.2 trillion won\n"
    ))
    assert out == ("The ETFs holding subsidiaries of 한화에어로스페이스 cannot be confirmed with the provided data. "
                   "There is no holdings data linking subsidiaries to products.")
    # the valid values of the asked condition, inline, are kept
    keep = ("This cannot be confirmed with the provided data. Credit rating 'AAAA' does not exist. "
            "Valid credit ratings: AAA, AA+, AA0, AA-, A+, A0, A-.")
    assert normalize_answer("cannot_confirm", keep) == keep


def test_cannot_confirm_drops_a_marker_less_row_listing_too():
    # Change 1 removes the markers, so a list is judged by content ('one product per line'), not by markers.
    out = normalize_answer("cannot_confirm", (
        "제공된 데이터로는 자회사를 편입한 ETF를 확인할 수 없습니다.\n"
        "대신 한화에어로스페이스를 편입한 ETF는 다음과 같습니다.\n"
        "KODEX 200 25조 4,748억원\nTIGER 200 10조 4,278억원\nPLUS 고배당주 1조 2,000억원\n"
    ))
    assert out == "제공된 데이터로는 자회사를 편입한 ETF를 확인할 수 없습니다."


def test_cannot_confirm_keeps_the_valid_values_of_the_asked_condition():
    """The side that must not be deleted — answers giving the condition's valid values inline (real answers G-016, G-017, G-022, G-052)."""
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
    """Deleting a product list is for cannot_confirm only.  A normal answer's list is only joined into a sentence."""
    out = normalize_answer("answered", (
        "순자산 상위 3개는 다음과 같습니다:\n"
        "1. KODEX 200 (069500) - 25조 4,748억원\n"
        "2. TIGER 200 (102110) - 10조 4,278억원\n"
        "3. TIGER 반도체TOP10 (396500) - 9조 7,582억원"
    ))
    for keep in ("KODEX 200", "069500", "25조 4,748억원", "TIGER 200", "TIGER 반도체TOP10", "9조 7,582억원"):
        assert keep in out, keep


def test_cannot_confirm_listing_is_kept_when_the_refusal_would_not_survive():
    """If the list is the whole answer it is left alone — throwing it all away here is the bigger loss."""
    only_list = "1. KODEX 200 (069500) - 25조원\n2. TIGER 200 (102110) - 10조원\n3. ACE 200 (105190) - 1조원"
    out = normalize_answer("cannot_confirm", only_list)
    for keep in ("KODEX 200", "TIGER 200", "ACE 200"):
        assert keep in out, keep


def test_recorded_answers_that_must_pass_through_unchanged():
    """Real answers from the 54-question regression set that must not change (numbers, cannot-confirm wording, schema facts)."""
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
    """The last expectation of test_leaked_tool_call_syntax_is_unwrapped_not_deleted (a list passes through as is)

    was changed on purpose by change 1 (prose only).  strip_tool_call_syntax still leaves it alone, and
    only normalize_answer joins the list into one sentence — not a single item disappears.
    """
    legit = ("1. KB RISE 글로벌양자컴퓨팅증권상장지수투자신탁 (RISE 시리즈)\n"
             "2. 삼성 KODEX 글로벌양자컴퓨팅증권상장지수투자신탁 (KODEX 시리즈)")
    assert strip_tool_call_syntax(legit) == legit
    out = normalize_answer("answered", legit)
    assert out == ("KB RISE 글로벌양자컴퓨팅증권상장지수투자신탁 (RISE 시리즈), "
                   "삼성 KODEX 글로벌양자컴퓨팅증권상장지수투자신탁 (KODEX 시리즈).")


def test_prompt_carries_the_plain_prose_and_no_substitution_rules():
    from finagent.agent.prompts import RULES
    assert "plain prose only" in RULES and "No markdown" in RULES and "numbered lists" in RULES
    assert "do not switch to a target that was not asked about (product group, entity, period)" in RULES
    assert "If it cannot be confirmed, stop there and do not append a list of substitute products" in RULES
    assert "markdown tables/lists allowed" not in RULES    # dropped from rule 9


# ===================================================================== blocking answers given without a lookup
# The block below was added append-only (the existing tests above were not touched).
#
# Background: HCX-005 (the competition build's model) sometimes answered from pre-training knowledge
# without calling a single tool, and measured, such answers were wrong without exception (G-034 gave the
# manager's short form, G-010 invented a product and numbers not in the DB).  So a 'final_answer after 0
# lookups' is not accepted and one more round runs.
from finagent.agent.orchestrator import FORCED_LOOKUP_MIN_REMAINING_SEC, LOOKUP_TOOLS


class _FullTap(StubLLM):
    """Collects the tool_results handed back to the model as (id, content, is_error), and the user messages too."""

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
    """Swap only the orchestrator's time and round budget.

    The existing ``_state`` has a 20-second budget, so the forced retry never kicks in (which is why the
    existing tests are unaffected by this change).  Here a budget close to the production value (120
    seconds) is given.
    """
    s.orchestrator.settings = Settings(
        _env_file=None, llm_provider="stub",
        time_budget_sec=time_budget_sec, max_tool_rounds=max_tool_rounds,
    )
    return s


# This must be a question on which round 0's prefetch does not fire.  The gates below want "no lookup
# happened for this question", and naming a product makes round 0 run a vector search, which breaks that
# premise (the prefetch counts as a real lookup the server ran).
_ASK = "판매 가능한 원화채권은 몇 개인가요?"          # pure aggregation — graph, vector, document all skipped
_SQL = "SELECT pd_nm, manager_name FROM v_etf_kr WHERE pd_nm LIKE '%2차전지%'"


def test_lookup_tools_matches_the_registry(tiny_store, tmp_path):
    """If LOOKUP_TOOLS drifts from the real tool list (a tool added or renamed), this catches it.

    It is written as an allow-list because the model calling a tool whose name does not exist must not
    count as 'looked up'.  The list can go stale instead, so it is checked against the registry.
    """
    s = _state(tiny_store, StubLLM([]), tmp_path)
    names = {t.name for t in s.tools.specs()}
    assert "final_answer" in names
    assert LOOKUP_TOOLS == names - {"final_answer"}
    assert FORCED_LOOKUP_MIN_REMAINING_SEC > 14   # must be above the existing 'time budget nearly spent' threshold


async def test_final_answer_without_any_lookup_is_sent_back_once(tiny_store, tmp_path):
    """G-034 regression: an answer given after 0 lookups is not accepted; one more round runs."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-034", _ASK)

    assert "submission rejected" in resp.think_trace
    assert "삼성자산운용" in resp.answer                  # the answer changed to the value from the lookup
    assert resp.retrieved_context.count("[Evidence") == 1  # a lookup really happened
    # the instruction is given as 'a fact at the point of use', in a user message
    assert "You must look up data at least once before answering" in llm.directives
    assert "sql_query" in llm.directives
    assert "official product names" in llm.directives


async def test_forced_retry_answers_every_pending_tool_call(tiny_store, tmp_path):
    """CLOVA (OpenAI-compatible) requires a result for every assistant tool_call — one missing and the next request fails with 400.

    Checks that every call gets a result, including those that came along with final_answer in the same
    turn and were **not executed**.  Unexecuted calls do not count as lookups (a result not seen is no
    evidence) — so the retry kicks in here too.
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
    assert "f1" in by_id and "x1" in by_id            # both pending calls got a result
    assert "Submission rejected" in by_id["f1"]
    assert "was not executed" in by_id["x1"]
    assert "submission rejected" in resp.think_trace and "삼성자산운용" in resp.answer


async def test_forced_retry_happens_at_most_once(tiny_store, tmp_path):
    """If tools are still not called the second time, the answer is accepted.

    An infinite loop exceeding the response time (the organisers recommend 60 seconds) is worse than one
    answer without evidence.
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
    assert "세번째답변은오면안된다" not in resp.answer          # no further round ran
    assert resp.think_trace.count("submission rejected") == 1
    assert sum(1 for m in llm.user_msgs if "[Data lookup required before answering]" in m) == 1


async def test_no_forced_retry_when_a_lookup_already_ran(tiny_store, tmp_path):
    """A final_answer after a lookup must not get the retry — a normal question must not pay +11 seconds."""
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-010", _ASK)

    assert "submission rejected" not in resp.think_trace
    assert not any("[Data lookup required before answering]" in m for m in llm.user_msgs)
    assert "삼성자산운용" in resp.answer and "이답변은오면안된다" not in resp.answer


async def test_table_info_alone_counts_as_a_lookup(tiny_store, tmp_path):
    """table_info returns a schema, not values, but counts as a lookup.

    Some questions are answered by table_info alone (base dates, allowed values).  To change this
    judgement, remove one line from orchestrator.LOOKUP_TOOLS.
    """
    script = [
        Turn("", [ToolCall("t1", "table_info", {"table": "v_etf_kr"})], "tool_use"),
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "국내ETF 기준일은 2026-08-21 입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-TI", "국내 ETF 데이터의 기준일은 언제인가요?")
    assert "submission rejected" not in resp.think_trace
    assert "2026-08-21" in resp.answer


async def test_forced_retry_applies_to_cannot_confirm_too(tiny_store, tmp_path):
    """A cannot-confirm too only stands as 'a refusal after a lookup'.

    At the same time the instruction must say that with 0 rows the answer is not to be made up and
    cannot_confirm is kept — if this change made cannot-confirm questions start inventing answers, it
    would be a net loss.
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

    assert "submission rejected" in resp.think_trace
    assert "확인할 수 없습니다" in resp.answer                     # the refusal stands
    assert "keeping status=cannot_confirm is the right answer" in llm.directives   # told not to make things up
    assert "do not make up an answer" in llm.directives


async def test_forced_retry_skipped_when_time_budget_is_short(tiny_store, tmp_path):
    """If the remaining budget cannot pay for the retry, the answer is accepted instead of sent back."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path), time_budget_sec=20)
    resp = await s.orchestrator.answer("G-T1", _ASK)

    assert "forced retry skipped" in resp.think_trace
    assert "삼성입니다" in resp.answer and "이답변은오면안된다" not in resp.answer
    assert not llm.user_msgs


async def test_forced_retry_skipped_when_no_round_is_left_to_run_tools(tiny_store, tmp_path):
    """Even with time left, if no round remains to run tools, sending it back produces no evidence."""
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("", [ToolCall("f2", "final_answer", {"status": "answered", "answer": "이답변은오면안된다"})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path), time_budget_sec=120, max_tool_rounds=1)
    resp = await s.orchestrator.answer("G-T2", _ASK)

    assert "forced retry skipped" in resp.think_trace
    assert "삼성입니다" in resp.answer and "이답변은오면안된다" not in resp.answer


async def test_rejected_answer_is_kept_if_the_retry_round_produces_nothing(tiny_store, tmp_path):
    """Even if the model produces nothing after the send-back (error, silence), the answer is not lost.

    Simply discarding the sent-back final_answer would, when the retry round fails, lose status and
    evidence_refs too and leave only the fallback wording.  Without evidence, the original answer still
    beats empty hands.
    """
    script = [
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        Turn("boom", [], "error"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-T3", _ASK)

    assert "submission rejected" in resp.think_trace
    assert "using the answer first submitted" in resp.think_trace
    assert "삼성입니다" in resp.answer


# ======================================================================= the count cross-check
async def test_count_answer_is_checked_against_the_single_count_result(tiny_store, tmp_path):
    """A count question whose one COUNT result is missing from the answer gets the number stated, in the question's language."""
    sql = "SELECT COUNT(*) AS n FROM v_etf_kr WHERE is_on_sale=1"
    for q, text, note in (
        ("판매중인 국내 ETF는 몇 개인가요?", "판매중인 국내 ETF는 3개입니다.", "(조회 결과 확인: 해당 조건에 맞는 건수는 2건입니다.)"),
        ("How many domestic ETFs are on sale?", "There are 3 domestic ETFs on sale.",
         "(Checked against the query result: 2 items match the condition.)"),
        ("What is the number of domestic ETFs on sale?", "3 domestic ETFs are on sale.",
         "(Checked against the query result: 2 items match the condition.)"),
    ):
        script = [Turn("", [ToolCall("c1", "sql_query", {"sql": sql})], "tool_use"),
                  Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": text})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        resp = await s.orchestrator.answer("C-1", q)
        assert resp.answer == f"{text}\n\n{note}", q
        assert "query result stated" in resp.think_trace
    # a question that asks no count is left alone, in either language
    for q in ("판매중인 국내 ETF를 알려줘", "Which domestic ETFs are on sale?"):
        script = [Turn("", [ToolCall("c1", "sql_query", {"sql": sql})], "tool_use"),
                  Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "3"})], "tool_use")]
        s = _state(tiny_store, StubLLM(script), tmp_path)
        assert (await s.orchestrator.answer("C-2", q)).answer == "3", q


# ======================================================================= final_answer gate
# When the tool loop ends without final_answer, the last text let out becomes the answer as is.
# 7 of the 54 questions took this path and 4 of them were wrong (both non-count failures were here too).
from finagent.agent.orchestrator import unwrap_final_answer_json  # noqa: E402

# Three real incident strings.  The reproduction tests use them as is.
#
# G-007: the raw final_answer argument JSON went out in the answer field and was shown to the grader.
#        It was cut mid-sentence by max_tokens and json.loads cannot open it.
_G007_TEXT = (
    '{\n'
    '"answer": "미국 증시에 상장된 주식형 ETF 중 총보수가 가장 낮은 상품을...\\n'
    '1. JPMorgan Alerian MLP Index ETN (AMJB)\\n'
    '2. ARK '
)
# G-009: already holding 3 SQL results, it let out cannot-confirm.
_G009_TEXT = "제공된 데이터로는 확인할 수 없습니다."
# G-025: one answer claimed both cannot-confirm and an aggregate result.
_G025_TEXT = "제공된 데이터로는 확인할 수 없습니다. 조회 결과 총 5개의 클래스가 판매되고 있습니다."


def _text(t):
    return Turn(t, [], "end_turn")


# ------------------------------------------------------------------ A. sending back (the resubmission instruction)

async def test_missing_final_answer_is_sent_back_once(tiny_store, tmp_path):
    """About to end without final_answer, the loop is sent back once — and exactly once."""
    script = [_text("1차 텍스트"), _text("2차 텍스트"), _text("3차 텍스트"), _text("4차 텍스트")]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-009a", _ASK)

    assert sum(1 for m in llm.user_msgs if "[Answer not submitted]" in m) == 1
    assert "asked to resubmit" in resp.think_trace
    assert "No final_answer on resubmission either" in resp.think_trace
    # no round is added after the send-back (no infinite loop) — the 4th scripted turn is not consumed.
    assert "4차 텍스트" not in resp.answer


async def test_retry_final_answer_is_adopted(tiny_store, tmp_path):
    """If the sent-back round calls final_answer, that answer is used."""
    script = [
        _text("1차 텍스트"),
        _text(_G009_TEXT),
        Turn("", [ToolCall("f1", "final_answer",
                           {"status": "answered", "answer": "순자산 상위 3개는 A, B, C 입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-009b", _ASK)

    assert "Resubmission: final_answer(status=answered) → adopted" in resp.think_trace
    assert "순자산 상위 3개는 A, B, C 입니다." in resp.answer
    assert not resp.answer.startswith("제공된 데이터로는 확인할 수 없습니다")   # the let-out text was not used


async def test_retry_directive_states_the_lookups_already_made(tiny_store, tmp_path):
    """The G-009 shape: holding SQL results already, it let out cannot-confirm.

    The instruction is given as 'a fact at the point of use' — in this project rules given in advance
    were ignored 4/4 and instructions given as facts of the moment were followed 5/5.  The fact that the
    lookups already ran must be in the instruction.
    """
    script = [
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        _text(_G009_TEXT),
        _text(_G009_TEXT),
        Turn("", [ToolCall("f1", "final_answer",
                           {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
    ]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-009c", _ASK)

    directive = next(m for m in llm.user_msgs if "[Answer not submitted]" in m)
    assert "Data lookups run for this question: 1" in directive
    assert "Recorded as evidence" in directive
    assert "final_answer(status, answer)" in directive
    assert "No data lookup tool has been called" not in directive   # lookups did run, so that sentence would be false
    assert "complete sentences in Korean" in directive               # the question is Korean
    assert "삼성자산운용" in resp.answer


async def test_retry_directive_when_no_lookup_ran(tiny_store, tmp_path):
    """With 0 lookups the instruction states exactly that."""
    script = [_text("1차"), _text("2차"), _text("3차")]
    llm = _FullTap(script)
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    await s.orchestrator.answer("G-009d", _ASK)

    directive = next(m for m in llm.user_msgs if "[Answer not submitted]" in m)
    assert "No data lookup tool has been called" in directive
    assert "Data lookups run" not in directive.split("[Answer not submitted]")[1].split("What to do now")[0]
    assert "complete sentences in Korean" in directive


async def test_retry_directive_names_english_for_an_english_question(tiny_store, tmp_path):
    llm = _FullTap([_text("1st"), _text("2nd"), _text("3rd")])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    await s.orchestrator.answer("G-009d-en", "How many buyable KRW bonds are there?")
    directive = next(m for m in llm.user_msgs if "[Answer not submitted]" in m)
    assert "complete sentences in English" in directive and "No data lookup tool has been called" in directive


async def test_no_retry_when_the_time_budget_is_short(tiny_store, tmp_path):
    """With too little budget nothing is sent back — the extra round costs ≈11 seconds (measured).

    ``_state``'s default budget is 20 seconds, short of FORCED_LOOKUP_MIN_REMAINING_SEC (33 seconds).
    That is also why the existing 165 tests were unaffected by this change.
    """
    script = [_text("1차 텍스트"), _text("2차 텍스트"), _text("이턴은오면안된다")]
    llm = _FullTap(script)
    s = _state(tiny_store, llm, tmp_path)                       # 20-second budget, not re-budgeted
    resp = await s.orchestrator.answer("G-009e", _ASK)

    assert "resubmission request skipped" in resp.think_trace and "not enough budget" in resp.think_trace
    assert not any("[Answer not submitted]" in m for m in llm.user_msgs)
    assert "이턴은오면안된다" not in resp.answer


async def test_no_retry_after_an_llm_error(tiny_store, tmp_path):
    """With no working conversation (an LLM error), sending it back only repeats the same failure."""
    llm = _FullTap([Turn("boom", [], "error"), _text("이턴은오면안된다")])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-009f", _ASK)

    assert not any("[Answer not submitted]" in m for m in llm.user_msgs)
    assert "LLM error" in resp.think_trace and "이턴은오면안된다" not in resp.answer


async def test_no_retry_when_the_lookup_gate_already_used_its_round(tiny_store, tmp_path):
    """If the zero-lookup gate already used a round, this gate does not fire (at most +1 round in total)."""
    llm = _FullTap([
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성입니다."})], "tool_use"),
        _text("되돌린 뒤에도 도구를 안 불렀다"),
    ])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-009g", _ASK)

    assert "submission rejected" in resp.think_trace
    assert not any("[Answer not submitted]" in m for m in llm.user_msgs)
    assert "삼성입니다" in resp.answer                          # the sent-back answer is restored as is


async def test_retry_answers_every_pending_tool_call(tiny_store, tmp_path):
    """Calls left unexecuted because the maximum rounds were exceeded get results too, before the send-back.

    OpenAI/CLOVA require a result for every assistant tool_call — one missing and the next request fails with 400.
    """
    llm = _FullTap([
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        Turn("", [ToolCall("c2", "sql_query", {"sql": _SQL + " LIMIT 1"})], "tool_use"),   # not run: over the round limit
        Turn("", [ToolCall("f1", "final_answer", {"status": "answered", "answer": "운용사는 삼성자산운용입니다."})], "tool_use"),
    ])
    s = _rebudget(_state(tiny_store, llm, tmp_path), max_tool_rounds=1)
    resp = await s.orchestrator.answer("G-009h", _ASK)

    by_id = {i: c for i, c, _ in llm.tool_results}
    assert "c2" in by_id and "was not executed" in by_id["c2"]
    assert "삼성자산운용" in resp.answer


# ------------------------------------------------------------------ B. unwrapping JSON in the fallback text

def test_unwrap_plain_final_answer_json():
    assert unwrap_final_answer_json('{"status": "answered", "answer": "총 5개입니다."}') == ("answered", "총 5개입니다.")
    assert unwrap_final_answer_json('{"answer": "총 5개입니다.", "status": "cannot_confirm"}') == (
        "cannot_confirm", "총 5개입니다.")


def test_unwrap_code_fenced_and_prose_wrapped_json():
    fenced = '```json\n{"answer": "총 5개입니다.", "status": "answered"}\n```'
    assert unwrap_final_answer_json(fenced) == ("answered", "총 5개입니다.")
    prose = '아래와 같이 답변을 제출합니다.\n{"answer": "총 5개입니다.", "status": "answered"}\n이상입니다.'
    assert unwrap_final_answer_json(prose) == ("answered", "총 5개입니다.")


def test_unwrap_decodes_escaped_newlines_and_quotes():
    raw = '{"answer": "1행\\n2행\\t끝 \\"인용\\" \\uac00", "status": "answered"}'
    status, answer = unwrap_final_answer_json(raw)
    assert status == "answered"
    assert answer == '1행\n2행\t끝 "인용" 가'


def test_unwrap_survives_a_truncated_json_block():
    """G-007's real incident string — cut by max_tokens, json.loads cannot open it."""
    got = unwrap_final_answer_json(_G007_TEXT)
    assert got is not None
    status, answer = got
    assert status == ""                                        # cut before reaching the status key
    assert answer.startswith("미국 증시에 상장된 주식형 ETF")
    assert "JPMorgan Alerian MLP Index ETN (AMJB)" in answer
    assert "\\n" not in answer and "\n" in answer              # the escape became a real line break
    assert '"answer"' not in answer


def test_unwrap_returns_none_for_ordinary_text():
    """Ordinary answers are not touched — if it cannot be unwrapped, behaviour stays as it is."""
    assert unwrap_final_answer_json(_G009_TEXT) is None
    assert unwrap_final_answer_json(_G025_TEXT) is None
    assert unwrap_final_answer_json("") is None
    assert unwrap_final_answer_json("final_answer 도구는 \"answer\" 와 \"status\" 를 받습니다.") is None
    assert unwrap_final_answer_json('{"answer": null, "status": "answered"}') is None
    assert unwrap_final_answer_json('{"answer": "   ", "status": "answered"}') is None


async def test_fallback_unwraps_a_tool_call_json_answer(tiny_store, tmp_path):
    """Sent back and still no final_answer: fall back, but unwrap it if it is JSON (G-007)."""
    llm = _FullTap([
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        _text("작업을 마치겠습니다."),
        _text("작업을 마치겠습니다."),
        _text(_G007_TEXT),                                     # the sent-back round lets out raw JSON too
    ])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-007", "미국 증시 주식형 ETF 중 총보수가 가장 낮은 상품")

    assert "final_answer argument JSON" in resp.think_trace
    assert '"answer"' not in resp.answer and '{' not in resp.answer
    assert "JPMorgan Alerian MLP Index ETN (AMJB)" in resp.answer


async def test_fallback_json_status_overrides_the_evidence_heuristic(tiny_store, tmp_path):
    """If the status inside the JSON is a valid value, it is followed."""
    blob = '```json\n{"answer": "해당 조건의 상품을 찾지 못했습니다.", "status": "cannot_confirm"}\n```'
    llm = _FullTap([
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),   # evidence exists, so the default would be answered
        _text(blob), _text(blob), _text(blob),
    ])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-007b", _ASK)

    assert "read as answer/status=cannot_confirm" in resp.think_trace
    assert resp.answer.startswith("제공된 데이터로는 확인할 수 없습니다")


async def test_fallback_keeps_the_existing_status_when_json_status_is_invalid(tiny_store, tmp_path):
    """If the status is not a valid value, the existing judgement applies."""
    blob = '{"answer": "순자산 상위 3개는 A, B, C 입니다.", "status": "done"}'
    llm = _FullTap([
        Turn("", [ToolCall("c1", "sql_query", {"sql": _SQL})], "tool_use"),
        _text(blob), _text(blob), _text(blob),
    ])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-007c", _ASK)

    assert "only answer read (status from the existing judgement)" in resp.think_trace
    assert "순자산 상위 3개는 A, B, C 입니다." in resp.answer   # evidence exists, so answered as is


async def test_fallback_leaves_plain_text_untouched(tiny_store, tmp_path):
    """Non-JSON text is not changed at all — G-025's self-contradicting sentence raises nothing either."""
    llm = _FullTap([_text(_G025_TEXT), _text(_G025_TEXT), _text(_G025_TEXT)])
    s = _rebudget(_state(tiny_store, llm, tmp_path))
    resp = await s.orchestrator.answer("G-025", _ASK)

    assert "final_answer argument JSON" not in resp.think_trace
    assert "final_answer not called → using the last text as the answer" in resp.think_trace
    assert "확인할 수 없습니다" in resp.answer
