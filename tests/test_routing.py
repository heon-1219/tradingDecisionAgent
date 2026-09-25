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
    # the same questions in English
    "How many buyable KRW bonds are rated AA- or higher?",
    "What are the three largest Korea-listed ETFs by net assets?",
    "How many foreign ETFs use an inverse or short strategy?",
    "What is the risk grade of RISE 200?",
])
def test_pure_sql_questions_route_nowhere(q):
    """Questions that only aggregate or sort are solved by one SQL query — switching stores on here is indiscriminate execution, not routing."""
    plan = routing.plan_retrieval(sig(q))
    assert plan.calls == []
    assert {s for s, _ in plan.skipped} == {"graph", "vector", "document"}


def test_summary_names_every_store_and_its_reason():
    s = routing.plan_retrieval(sig("국내 ETF 중 총보수가 가장 낮은 상품 3개를 알려줘")).summary()
    for name in ("graph", "vector", "document"):
        assert name in s
    assert "=ran(" not in s  # nothing switched on, so nothing may be reported as ran


# ---------------------------------------------------------------- graph: holdings
@pytest.mark.parametrize("q,entity", [
    ("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?", "에코프로비엠"),
    ("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘", "삼성전자"),
    ("SK하이닉스를 편입하고 있는 국내 ETF는 몇 개인가요?", "SK하이닉스"),
    ("두산에너빌리티를 보유한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "두산에너빌리티"),
    ("쿠팡을 편입한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "쿠팡"),
    ("야놀자를 보유한 국내 ETF가 있나요?", "야놀자"),
    ("캠브리콘이 편입된 중국 반도체 ETF를 알려줘", "캠브리콘"),
    # English: the holder comes after the verb; a Korean name inside the English question is taken
    ("Which domestic ETFs hold 에코프로비엠?", "에코프로비엠"),
    ("How many Korean ETFs are holding 쿠팡?", "쿠팡"),
    ("Do any domestic ETFs contain 야놀자's shares?", "야놀자"),
    ("Which ETFs include 두산에너빌리티 in their portfolio?", "두산에너빌리티"),
])
def test_holding_questions_route_to_graph(q, entity):
    plan = routing.plan_retrieval(sig(q))
    assert [c.source for c in plan.calls] == ["graph"]
    assert plan.calls[0].tool == "graph_neighbors"
    assert plan.calls[0].args == {"entity": entity, "relation": "constituentOf"}


def test_holder_extraction_survives_particles_and_ignores_scope_nouns():
    assert routing.holder_mentions("삼성전자를 보유한 ETF") == ["삼성전자"]
    assert routing.holder_mentions("캠브리콘이 편입된 ETF") == ["캠브리콘"]
    # scope nouns such as '상품' (product) and '종목' (stock) are not companies
    assert "상품" not in routing.holder_mentions("판매중인 상품에 포함된 ETF")
    assert "종목" not in routing.holder_mentions("여러 종목을 포함한 ETF")


def test_english_holder_extraction_takes_only_hangul_names():
    """An English token the lexicon does not know would come back from the graph as a false '0' fact."""
    assert routing.holder_mentions("Which ETFs hold 삼성전자?") == ["삼성전자"]
    assert routing.holder_mentions("Which ETFs hold Coupang?") == []
    assert routing.holder_mentions("ETFs holding 에코프로's subsidiaries") == ["에코프로"]
    assert routing.holder_mentions("Which ETFs hold 자회사?") == []          # relation words stay excluded
    # an English name the lexicon resolved reaches the plan through Signals.companies
    plan = routing.plan_retrieval(sig("Which domestic ETFs hold Samsung Electronics?", companies=["삼성전자"]))
    assert plan.calls[0].args == {"entity": "삼성전자", "relation": "constituentOf"}
    # nothing to ask the graph with: skipped, and the reason says so
    plan = routing.plan_retrieval(sig("Which domestic ETFs hold Coupang?"))
    assert plan.calls == [] and any(s == "graph" and "could not be identified" in w for s, w in plan.skipped)


def test_english_holding_words_are_lower_case_only():
    """Product names are written in capitals — 'ASML Holding NV' is a name, not a holdings question."""
    assert not routing.is_holding_question("What is the total fee of ASML Holding NV ADRhedged?")
    assert routing.is_holding_question("Holding 삼성전자, which ETFs are the largest?")   # sentence start is folded
    assert not routing.is_holding_question("What is the fee, including taxes, of KODEX 200?")
    # English words never switch a Korean question on
    assert not routing.is_holding_question("KODEX 200 hold 전략의 총보수는?")


def test_lexicon_canonical_name_wins_over_the_raw_token():
    """Graph labels carry the lexicon's canonical name — query that, not the sentence's spelling."""
    plan = routing.plan_retrieval(sig("에코프로비엠을 편입한 국내 ETF는?", companies=["에코프로비엠"]))
    assert plan.calls[0].args["entity"] == "에코프로비엠"


@pytest.mark.parametrize("q", [
    "삼성전자를 보유한 공모펀드를 알려줘",
    "엔비디아를 편입한 해외 ETF를 알려줘",
    "SK하이닉스를 담고 있는 공모펀드 중 순자산이 가장 큰 상품은 무엇인가요?",
    "Which public funds hold 삼성전자?",
])
def test_holding_blind_scope_does_not_touch_the_graph(q):
    """Switching on the domestic-ETF graph for a group without holdings data would answer for another group."""
    plan = routing.plan_retrieval(sig(q, holding_blind=True))
    assert plan.calls == []
    assert ("graph", "the named product group has no holdings data") in plan.skipped


def test_ranking_holdings_question_carries_the_unordered_caution():
    plan = routing.plan_retrieval(sig("삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 3개"))
    assert "not sorted" in plan.calls[0].caution


# ---------------------------------------------------------------- graph: themes
def test_theme_routing_needs_the_word_theme():
    themes = [ThemeSig("space_aero", "우주항공")]
    assert sources("최근 6개월 우주항공 테마 관련 ETF를 알려줘", themes=themes) == {"graph"}
    # a theme match that came along with a product name is no evidence of a theme question
    assert sources("KODEX 우주항공레버리지 ETF의 순자산을 알려줘", themes=themes) == set()
    # English
    assert sources("Which ETFs belong to the aerospace theme?", themes=themes) == {"graph"}
    assert sources("What are the net assets of the aerospace ETFs?", themes=themes) == set()


def test_generic_geography_and_structure_themes_never_route():
    for tid, label in (("korea", "국내/한국"), ("usa", "미국"), ("leverage", "레버리지")):
        assert sources("이 테마 상품을 알려줘", themes=[ThemeSig(tid, label)]) == set()
        assert sources("Show me products in this theme", themes=[ThemeSig(tid, label)]) == set()


def test_theme_call_uses_the_graph_theme_node_and_prefers_the_specific_theme():
    plan = routing.plan_retrieval(sig(
        "중국반도체 테마 ETF", themes=[ThemeSig("china", "중국"), ThemeSig("china_semi", "중국반도체")]))
    assert plan.calls[0].args == {"entity": "d:theme/china_semi", "relation": "hasTheme_inverse"}


# ---------------------------------------------------------------- documents
@pytest.mark.parametrize("q", [
    "국민성장펀드의 구조와 투자전략 동향을 설명해줘",
    "TIGER 미국나스닥100의 위험요인은 무엇인가요?",
    "ETF 투자 시 유의사항을 알려줘",
    # English
    "Explain the structure and investment strategy of 국민성장펀드",
    "What are the risk factors of TIGER 미국나스닥100?",
    "What precautions should I take when investing in ETFs?",
    "What does tracking error mean?",
    "Why do leveraged ETFs lose value in sideways markets?",
    "What are the pros and cons of covered-call ETFs?",
    "How is 국민성장펀드 managed?",
    "Give me an overview of recent fund trends",
])
def test_narrative_questions_route_to_documents(q):
    assert "document" in sources(q)


@pytest.mark.parametrize("q", [
    "국민성장펀드는 몇 개의 클래스로 판매되고 있나요?",
    "신한글로벌탄소중립솔루션증권투자신탁의 벤치마크 지수는 무엇인가요?",
    "한국도로공사고속도로건설채권 634(변)의 신용등급은 무엇인가요?",
    # English: 'what is' / 'tell me' end attribute lookups, not narrative ones
    "How many classes is 국민성장펀드 sold in?",
    "What is the benchmark index of 신한글로벌탄소중립솔루션증권투자신탁?",
    "Tell me the credit rating of 한국도로공사고속도로건설채권 634(변)",
    "What is the risk grade of KODEX 200?",
    "What is the AUM of the Goldman Sachs Future Tech Leaders Equity ETF?",
])
def test_attribute_lookups_do_not_route_to_documents(q):
    """'무엇인가요' ("what is") is not a narrative signal — each of these is answered by one column."""
    assert "document" not in sources(q)


# ---------------------------------------------------------------- vector / product lookup
def test_named_product_that_exists_routes_to_the_product_search():
    plan = routing.plan_retrieval(sig(
        "Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘",
        mentions=[MentionSig("Invesco QQQ Trust", resolved=True, table="v_etf_gl")]))
    assert [c.source for c in plan.calls] == ["vector"]
    assert plan.calls[0].args == {"query": "Invesco QQQ Trust", "k": 3, "table": "v_etf_gl"}


def test_named_product_that_does_not_exist_routes_nowhere():
    """Where 'no such product' is the answer, injecting similarly named rows as facts pushes toward a wrong answer."""
    plan = routing.plan_retrieval(sig(
        "KODEX 우주항공레버리지 ETF의 순자산을 알려줘",
        mentions=[MentionSig("KODEX 우주항공레버리지 ETF", resolved=False, table="v_fund_pub")]))
    assert plan.calls == []
    assert any(s == "vector" and "wrong answer" in w for s, w in plan.skipped)


def test_constituent_view_is_not_used_to_scope_a_product_search():
    """v_etf_constituent is the holdings view — scoping to it returns only holding rows for a manager or fee question."""
    plan = routing.plan_retrieval(sig(
        "TIGER 미국나스닥100을 운용하는 자산운용사는 어디인가요?",
        mentions=[MentionSig("TIGER 미국나스닥100", resolved=True, table="v_etf_constituent")]))
    assert "table" not in plan.calls[0].args


# ---------------------------------------------------------------- guards
def test_forward_looking_questions_prefetch_nothing():
    """The schema holds no future values.  Prefetching the current snapshot turns it into a forecast."""
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
        "constituentOf (42 total, 8 examples shown): IBK 코스닥150, TIGER 코스닥150 레버리지 ...", evidence=_Ev(42))}))
    assert "IBK 코스닥150" in pf.block          # the rows themselves go in, not advice
    assert "graph_neighbors" in pf.block
    assert "why chosen" in pf.block
    assert "42 in total" in pf.block
    assert pf.ran == 1


def test_a_graph_detour_is_reported_as_the_absence_it_actually_is():
    """graph_neighbors gives the node's *other* relations when constituentOf is empty.

    '야놀자를 보유한 국내 ETF가 있나요?' came back with a corporate bond named '야놀자1-1' under a
    '15 in total' header — for a question whose answer is 'no ETF holds it'.  Absence must be written as absence.
    """
    plan = routing.plan_retrieval(sig("야놀자를 보유한 국내 ETF가 있나요?"))
    detour = "\n".join([
        "(No 'constituentOf' relation, so all relations of this entity are shown instead)",
        "productType (1 total): 국내채권",
        "issuedBy (1 total): (주)야놀자",
    ])
    pf = run(plan, _Tools({"graph_neighbors": _Res(detour, evidence=_Ev(15))}))
    assert "국내채권" not in pf.block
    assert "15 in total" not in pf.block
    assert "has 0 constituentOf" in pf.block and "0 in total" in pf.block


def test_constituent_rows_are_dropped_from_a_product_attribute_lookup():
    """For QQQ's manager, holdings-view rows pushed out the row with the answer (v_etf_gl)."""
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
    assert "2 of 3 search results excluded" in pf.block  # the dropped count is given against the total


def test_the_routing_decision_is_traced_for_every_store():
    plan = routing.plan_retrieval(sig("에코프로비엠을 편입한 국내 ETF는 몇 개인가요?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res("constituentOf (42 total): ...", evidence=_Ev(42))}))
    head = pf.steps[0]
    assert head.startswith("Round 0 routing")
    assert "graph=ran" in head and "vector=skipped" in head and "document=skipped" in head


def test_a_question_with_no_plan_still_records_why_nothing_ran():
    pf = run(routing.plan_retrieval(sig("국내 ETF 중 순자산 상위 3개는?")), _Tools({}))
    assert pf.block == ""
    assert pf.steps and "nothing to prefetch" in pf.steps[0] and "sql_query" in pf.steps[0]


def test_clipped_output_reports_the_true_total_not_the_cap():
    """Reporting the cap as the total is a bug this project has had twice."""
    long = "constituentOf (238 total, 8 examples shown): " + "가나다라, " * 900
    plan = routing.plan_retrieval(sig("삼성전자를 보유한 국내 ETF는?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res(long, evidence=_Ev(238))}))
    assert "238 in total" in pf.block
    assert len(pf.block) < 3000


def test_a_failing_store_is_skipped_and_the_answer_goes_on():
    plan = routing.plan_retrieval(sig(
        "국민성장펀드의 구조와 투자전략 동향은?",
        mentions=[MentionSig("국민성장펀드", resolved=True, table="v_fund_pub")]))
    assert {c.source for c in plan.calls} == {"document", "vector"}
    pf = run(plan, _Tools({"search_documents": RuntimeError("no document store"),
                           "search_products": _Res("- [v_fund_pub] id=1 | 국민성장", evidence=_Ev(1))}))
    assert pf.ran == 1
    assert "국민성장" in pf.block
    assert any("skipped" in s for s in pf.steps)


def test_a_similarly_named_other_product_is_never_injected_as_the_answer():
    """A search result is a ranking, not an identity.

    The top product-view hit for 'Invesco QQQ Trust' was 'Invesco NASDAQ Internet ETF' (measured; up to
    k=10 QQQ's v_etf_gl row never appears).  Harmless as a hint, but injected as a fact it makes another
    product's manager this product's.  With no row left to keep, nothing is injected and the tool loop takes over.
    """
    plan = routing.plan_retrieval(sig(
        "Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘",
        mentions=[MentionSig("Invesco QQQ Trust", resolved=True)]))
    other = "- [v_etf_gl] id=PNQI.O | Invesco NASDAQ Internet ETF | manager_name=Invesco Capital Management LLC"
    pf = run(plan, _Tools({"search_products": _Res(other, evidence=_Ev(5))}))
    assert pf.block == ""
    assert any("no usable rows" in s for s in pf.steps)


def test_an_errored_tool_result_is_not_injected():
    plan = routing.plan_retrieval(sig("야놀자를 보유한 국내 ETF가 있나요?"))
    pf = run(plan, _Tools({"graph_neighbors": _Res("The knowledge graph is not loaded.", is_error=True)}))
    assert pf.block == ""
    assert pf.ran == 0


def test_the_total_budget_stops_the_remaining_calls():
    plan = routing.plan_retrieval(sig(
        "국민성장펀드의 구조와 투자전략 동향은?",
        mentions=[MentionSig("국민성장펀드", resolved=True, table="v_fund_pub")]))
    pf = run(plan, _Tools({"search_documents": _Res("문서", evidence=_Ev(1)),
                           "search_products": _Res("상품", evidence=_Ev(1))}), total_budget=0.0)
    assert pf.ran == 0
    assert any("budget" in s for s in pf.steps)


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
