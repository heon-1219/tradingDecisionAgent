"""Round-0 retrieval routing: which store gets asked, and what round 0 injects."""
import asyncio

import pytest

from finagent.agent import routing
from finagent.agent.routing import MentionSig, Signals, ThemeSig


def sig(q, **kw):
    return Signals(question=q, **kw)


def sources(q, **kw):
    return set(routing.plan_retrieval(sig(q, **kw)).sources)


# ---------------------------------------------------------------- the default is silence
@pytest.mark.parametrize("q", [
    "현재 구매 가능한 원화채권 중 신용등급 AA- 이상인 종목은 몇 개인가요?",
    "국내 상장 ETF 중 순자산이 가장 큰 상품 3개의 이름을 알려줘",
    "판매중인 국내 ETF와 판매중인 공모펀드는 각각 몇 개인가요?",
    "해외 ETF 중 인버스나 숏 전략을 사용하는 상품은 몇 개인가요?",
])
def test_pure_sql_questions_route_nowhere(q):
    """집계·정렬만 하는 질문은 SQL 하나로 풀린다 — 여기서 저장소를 켜면 라우팅이 아니라 무차별 실행이다."""
    plan = routing.plan_retrieval(sig(q))
    assert plan.calls == []
    assert {s for s, _ in plan.skipped} == {"graph", "vector", "document"}


def test_summary_names_every_store_and_its_reason():
    s = routing.plan_retrieval(sig("국내 ETF 중 총보수가 가장 낮은 상품 3개를 알려줘")).summary()
    for ko in ("그래프", "벡터", "문서"):
        assert ko in s
    assert "실행" not in s  # 아무것도 안 켜졌으면 실행이라고 적으면 안 된다


# ---------------------------------------------------------------- graph: holdings
@pytest.mark.parametrize("q,entity", [
    ("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?", "에코프로비엠"),
    ("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘", "삼성전자"),
    ("SK하이닉스를 편입하고 있는 국내 ETF는 몇 개인가요?", "SK하이닉스"),
    ("두산에너빌리티를 보유한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "두산에너빌리티"),
    ("쿠팡을 편입한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "쿠팡"),
    ("야놀자를 보유한 국내 ETF가 있나요?", "야놀자"),
    ("캠브리콘이 편입된 중국 반도체 ETF를 알려줘", "캠브리콘"),
])
def test_holding_questions_route_to_graph(q, entity):
    plan = routing.plan_retrieval(sig(q))
    assert [c.source for c in plan.calls] == ["graph"]
    assert plan.calls[0].tool == "graph_neighbors"
    assert plan.calls[0].args == {"entity": entity, "relation": "constituentOf"}


def test_holder_extraction_survives_particles_and_ignores_scope_nouns():
    assert routing.holder_mentions("삼성전자를 보유한 ETF") == ["삼성전자"]
    assert routing.holder_mentions("캠브리콘이 편입된 ETF") == ["캠브리콘"]
    # '상품'·'종목' 같은 범위 명사는 기업이 아니다
    assert "상품" not in routing.holder_mentions("판매중인 상품에 포함된 ETF")
    assert "종목" not in routing.holder_mentions("여러 종목을 포함한 ETF")


def test_lexicon_canonical_name_wins_over_the_raw_token():
    """그래프 라벨은 사전의 표준명으로 붙어 있다 — 문장의 표기가 아니라 그것을 조회해야 한다."""
    plan = routing.plan_retrieval(sig("에코프로비엠을 편입한 국내 ETF는?", companies=["에코프로비엠"]))
    assert plan.calls[0].args["entity"] == "에코프로비엠"


@pytest.mark.parametrize("q", [
    "삼성전자를 보유한 공모펀드를 알려줘",
    "엔비디아를 편입한 해외 ETF를 알려줘",
    "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
])
def test_holding_blind_scope_does_not_touch_the_graph(q):
    """편입 데이터가 없는 상품군을 지목한 질문에 국내ETF 그래프를 켜면 다른 상품군의 답을 주게 된다."""
    plan = routing.plan_retrieval(sig(q, holding_blind=True))
    assert plan.calls == []
    assert ("graph", "지목된 상품군에 편입종목 데이터가 없음") in plan.skipped


def test_ranking_holdings_question_carries_the_unordered_caution():
    plan = routing.plan_retrieval(sig("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 3개"))
    assert "정렬" in plan.calls[0].caution


# ---------------------------------------------------------------- graph: themes
def test_theme_routing_needs_the_word_theme():
    themes = [ThemeSig("space_aero", "우주항공")]
    assert sources("최근 6개월 우주항공 테마 관련 ETF를 알려줘", themes=themes) == {"graph"}
    # 상품명에서 딸려온 테마 매칭은 그 질문이 테마 질문이라는 근거가 아니다
    assert sources("KODEX 우주항공레버리지 ETF의 순자산을 알려줘", themes=themes) == set()


def test_generic_geography_and_structure_themes_never_route():
    for tid, label in (("korea", "국내/한국"), ("usa", "미국"), ("leverage", "레버리지")):
        assert sources("이 테마 상품을 알려줘", themes=[ThemeSig(tid, label)]) == set()


def test_theme_call_uses_the_graph_theme_node_and_prefers_the_specific_theme():
    plan = routing.plan_retrieval(sig(
        "중국반도체 테마 ETF", themes=[ThemeSig("china", "중국"), ThemeSig("china_semi", "중국반도체")]))
    assert plan.calls[0].args == {"entity": "d:theme/china_semi", "relation": "hasTheme_inverse"}


# ---------------------------------------------------------------- documents
@pytest.mark.parametrize("q", [
    "국민성장펀드의 구조와 투자전략 동향을 설명해줘",
    "TIGER 미국나스닥100의 위험요인은 무엇인가요?",
    "ETF 투자 시 유의사항을 알려줘",
])
def test_narrative_questions_route_to_documents(q):
    assert "document" in sources(q)


@pytest.mark.parametrize("q", [
    "국민성장펀드는 몇 개의 클래스로 판매되고 있나요?",
    "신한글로벌탄소중립솔루션증권투자신탁의 벤치마크 지수는 무엇인가요?",
    "한국도로공사고속도로건설채권 634(변)의 신용등급은 무엇인가요?",
])
def test_attribute_lookups_do_not_route_to_documents(q):
    """'무엇인가요'는 서술형 신호가 아니다 — 이 셋은 모두 컬럼 하나로 답이 나온다."""
    assert "document" not in sources(q)


# ---------------------------------------------------------------- vector / product lookup
def test_named_product_that_exists_routes_to_the_product_search():
    plan = routing.plan_retrieval(sig(
        "Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘",
        mentions=[MentionSig("Invesco QQQ Trust", resolved=True, table="v_etf_gl")]))
    assert [c.source for c in plan.calls] == ["vector"]
    assert plan.calls[0].args == {"query": "Invesco QQQ Trust", "k": 3, "table": "v_etf_gl"}


def test_named_product_that_does_not_exist_routes_nowhere():
    """'그런 상품 없음'이 정답인 문항에 이름이 비슷한 행을 사실로 넣으면 오답 쪽으로 민다."""
    plan = routing.plan_retrieval(sig(
        "KODEX 우주항공레버리지 ETF의 순자산을 알려줘",
        mentions=[MentionSig("KODEX 우주항공레버리지 ETF", resolved=False, table="v_fund_pub")]))
    assert plan.calls == []
    assert any(s == "vector" and "오답" in w for s, w in plan.skipped)


def test_constituent_view_is_not_used_to_scope_a_product_search():
    """v_etf_constituent 는 편입종목 뷰다 — 여기로 좁히면 운용사·총보수 질문에 종목 행만 돌아온다."""
    plan = routing.plan_retrieval(sig(
        "TIGER 미국나스닥100을 운용하는 자산운용사는 어디인가요?",
        mentions=[MentionSig("TIGER 미국나스닥100", resolved=True, table="v_etf_constituent")]))
    assert "table" not in plan.calls[0].args


# ---------------------------------------------------------------- guards
def test_forward_looking_questions_prefetch_nothing():
    """스키마에 미래 시점 값이 없다. 현재 스냅샷을 미리 넣어 주면 그것이 예측치로 둔갑한다."""
    plan = routing.plan_retrieval(sig(
        "KODEX 200의 2027년 말 예상 순자산 규모는 얼마인가요?", forward_looking=True,
        mentions=[MentionSig("KODEX 200", resolved=True, table="v_etf_kr")]))
    assert plan.calls == []
    assert len(plan.skipped) == 3


def test_plan_never_exceeds_the_call_cap():
    plan = routing.plan_retrieval(sig(
        "삼성전자를 보유한 우주항공 테마 ETF의 위험요인과 투자전략 구조는?",
        themes=[ThemeSig("space_aero", "우주항공")],
        mentions=[MentionSig("우주항공 ETF", resolved=True, table="v_etf_kr")]))
    assert len(plan.calls) <= routing.MAX_CALLS


# ---------------------------------------------------------------- execution
class _Res:
    def __init__(self, content, is_error=False, evidence=None):
        self.content, self.is_error, self.evidence = content, is_error, evidence


class _Ev:
    def __init__(self, row_count):
        self.row_count = row_count


class _Tools:
    def __init__(self, results):
        self.results, self.seen = results, []

    async def execute(self, name, args):
        self.seen.append((name, args))
        r = self.results.get(name)
        if isinstance(r, Exception):
            raise r
        return r


def run(plan, tools, **kw):
    return asyncio.run(routing.run_plan(plan, tools, **kw))


def test_results_are_injected_as_facts_not_as_advice():
    plan = routing.plan_retrieval(sig("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res(
        "constituentOf (42건, 예시 8건): IBK 코스닥150, TIGER 코스닥150 레버리지 ...", evidence=_Ev(42))}))
    assert "IBK 코스닥150" in pf.block          # 조언이 아니라 행 자체가 들어간다
    assert "graph_neighbors" in pf.block
    assert "선택 근거" in pf.block
    assert "전체 42건" in pf.block
    assert pf.ran == 1


def test_a_graph_detour_is_reported_as_the_absence_it_actually_is():
    """graph_neighbors 는 constituentOf 가 비면 그 노드의 *다른* 관계를 대신 준다.

    '야놀자를 보유한 국내 ETF가 있나요?' 에 회사채 '야놀자1-1' 이 '전체 15건' 이라는 머리말과 함께
    돌아왔다 — 정답이 '편입 ETF 없음'인 문항에서.  없음은 없음으로 적어야 한다.
    """
    plan = routing.plan_retrieval(sig("야놀자를 보유한 국내 ETF가 있나요?"))
    detour = "\n".join([
        "('constituentOf' 관계는 없어 이 엔티티의 전체 관계를 대신 보여줍니다)",
        "productType (1건): 국내채권",
        "issuedBy (1건): (주)야놀자",
    ])
    pf = run(plan, _Tools({"graph_neighbors": _Res(detour, evidence=_Ev(15))}))
    assert "국내채권" not in pf.block
    assert "15건" not in pf.block
    assert "0건" in pf.block


def test_constituent_rows_are_dropped_from_a_product_attribute_lookup():
    """QQQ 의 운용사를 묻는 질문에서 편입종목 뷰 행들이 정답 행(v_etf_gl)을 밀어냈다."""
    plan = routing.plan_retrieval(sig(
        "Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘",
        mentions=[MentionSig("Invesco QQQ Trust", resolved=True, table="v_etf_gl")]))
    content = "\n".join([
        "- [v_etf_constituent] id=A | Invesco QQQ Trust | etf_name=KODEX ...",
        "- [v_etf_constituent] id=B | Invesco QQQ Trust | etf_name=KIWOOM ...",
        "- [v_etf_gl] id=QQQ | Invesco QQQ Trust, Series 1 | manager_name=Invesco",
    ])
    pf = run(plan, _Tools({"search_products": _Res(content, evidence=_Ev(3))}))
    assert "manager_name=Invesco" in pf.block
    assert "id=A" not in pf.block and "id=B" not in pf.block
    assert "3건 중 2건" in pf.block  # 제외한 건수를 총계와 함께 밝힌다


def test_the_routing_decision_is_traced_for_every_store():
    plan = routing.plan_retrieval(sig("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res("constituentOf (42건): ...", evidence=_Ev(42))}))
    head = pf.steps[0]
    assert head.startswith("라운드 0 라우팅")
    assert "그래프=실행" in head and "벡터=생략" in head and "문서=생략" in head


def test_a_question_with_no_plan_still_records_why_nothing_ran():
    pf = run(routing.plan_retrieval(sig("국내 ETF 중 순자산 상위 3개는?")), _Tools({}))
    assert pf.block == ""
    # 문구를 "사전 검색 대상 없음 → 이 질문은 RDB(sql_query)로 처리합니다" 로 바꿨다.
    # think_trace 는 채점 항목이라, 이 줄만 읽고 "아무 저장소도 안 썼다"로 보이면 안 된다.
    assert pf.steps and "사전 검색 대상 없음" in pf.steps[0] and "sql_query" in pf.steps[0]


def test_clipped_output_reports_the_true_total_not_the_cap():
    """상한을 총계로 보고하는 것은 이 프로젝트에서 두 번 난 버그다."""
    long = "constituentOf (238건, 예시 8건): " + "가나다라, " * 900
    plan = routing.plan_retrieval(sig("삼성전자를 보유한 국내 ETF는?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res(long, evidence=_Ev(238))}))
    assert "전체 238건" in pf.block
    assert len(pf.block) < 3000


def test_a_failing_store_is_skipped_and_the_answer_goes_on():
    plan = routing.plan_retrieval(sig(
        "국민성장펀드의 구조와 투자전략 동향은?",
        mentions=[MentionSig("국민성장펀드", resolved=True, table="v_fund_pub")]))
    assert {c.source for c in plan.calls} == {"document", "vector"}
    pf = run(plan, _Tools({"search_documents": RuntimeError("문서 스토어 없음"),
                           "search_products": _Res("- [v_fund_pub] id=1 | 국민성장", evidence=_Ev(1))}))
    assert pf.ran == 1
    assert "국민성장" in pf.block
    assert any("건너뜀" in s for s in pf.steps)


def test_a_similarly_named_other_product_is_never_injected_as_the_answer():
    """검색 결과는 순위지 동일성이 아니다.

    'Invesco QQQ Trust' 로 검색한 상품 뷰 최상위는 'Invesco NASDAQ Internet ETF' 였다(실측, k=10 까지
    QQQ 의 v_etf_gl 행은 나오지 않는다).  힌트라면 무해하지만 사실로 주입하면 다른 상품의 운용사를
    이 상품의 운용사로 만든다.  남길 행이 없으면 아무것도 싣지 않고 도구 루프에 맡긴다.
    """
    plan = routing.plan_retrieval(sig(
        "Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘",
        mentions=[MentionSig("Invesco QQQ Trust", resolved=True)]))
    other = "- [v_etf_gl] id=PNQI.O | Invesco NASDAQ Internet ETF | manager_name=Invesco Capital Management LLC"
    pf = run(plan, _Tools({"search_products": _Res(other, evidence=_Ev(5))}))
    assert pf.block == ""
    assert any("사용할 행 없음" in s for s in pf.steps)


def test_an_errored_tool_result_is_not_injected():
    plan = routing.plan_retrieval(sig("야놀자를 보유한 국내 ETF가 있나요?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res("지식그래프가 로드되지 않았습니다.", is_error=True)}))
    assert pf.block == ""
    assert pf.ran == 0


def test_the_total_budget_stops_the_remaining_calls():
    plan = routing.plan_retrieval(sig(
        "국민성장펀드의 구조와 투자전략 동향은?",
        mentions=[MentionSig("국민성장펀드", resolved=True, table="v_fund_pub")]))
    pf = run(plan, _Tools({"search_documents": _Res("문서", evidence=_Ev(1)),
                           "search_products": _Res("상품", evidence=_Ev(1))}), total_budget=0.0)
    assert pf.ran == 0
    assert any("예산" in s for s in pf.steps)


def test_precomputed_mentions_are_used_without_reresolving():
    class _Boom:
        def resolve(self, *a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("resolver was called even though mentions were supplied")

    s = routing.signals_for("KODEX 200의 순자산은?", resolver=_Boom(),
                            mentions=["KODEX 200"], unresolved=[])
    assert [(m.text, m.resolved) for m in s.mentions] == [("KODEX 200", True)]
    s2 = routing.signals_for("KODEX AI로봇 ETF 정보 알려줘", resolver=_Boom(),
                             mentions=["KODEX AI로봇"], unresolved=["KODEX AI로봇"])
    assert s2.mentions[0].resolved is False
