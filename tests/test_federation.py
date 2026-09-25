"""Federation regression tests (Task 0.3): every source family actually contributes.

The app answers by federating four local source families behind ``ToolRegistry``: SQLite views
(``sql_query``), the rdflib knowledge graph (``graph_neighbors``/``graph_query``), lexical BM25 + LSA
vector search (``search_products``), and reference docs (``search_documents``).  ``Orchestrator.answer``
runs a routing "round 0" prefetch (``agent/routing.py``) before the tool loop and returns an
``AnswerResponse`` whose ``retrieved_context`` cites evidence per source.

These tests run against the REAL built store (``data/store/products.db`` + the vector cache), not a tiny
in-memory fixture, because the point is to prove the real data actually has rows each source can surface
— a tiny store could pass while the real store's federation was silently broken (a different index build,
a missing table, a graph with 0 triples, ...).

Loading the full state (RDB + KG build + vector index) takes about 60-90s, so every test in this file
shares one module-scoped fixture, and the whole module is marked ``slow`` (excluded by
``pytest -m "not slow"``, per the project's global constraints).

Round-0 prefetch evidence is deliberately NOT injected into ``retrieved_context`` (the tool loop usually
repeats the same lookup, and orchestrator.py's comment there explains why double-counting is avoided) —
so the routing-prefetch tests below assert on ``think_trace``'s ``"Round 0 routing:"`` line instead, per
``agent/routing.py``'s ``RoutePlan.summary()``.
"""
from __future__ import annotations

import pytest

from finagent.agent.llm import StubLLM, ToolCall, Turn
from finagent.app_state import AppState
from finagent.config import Settings

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_state():
    """The full application state built from the real store, with a stub (network-free) LLM."""
    settings = Settings(_env_file=None, llm_provider="stub", time_budget_sec=20, max_tool_rounds=4)
    return AppState.load(settings, llm=StubLLM([]))


def _script(real_state, script: list[Turn]) -> StubLLM:
    """Point the shared orchestrator at a fresh scripted LLM for one test (state itself is untouched)."""
    llm = StubLLM(script)
    real_state.orchestrator.llm = llm
    return llm


# ------------------------------------------------------------------ one test per source family
# Each calls the tool directly through ToolRegistry.execute (no LLM involved) and asserts that
# non-empty, correctly-kinded evidence comes back — i.e. the source really has rows to give.

async def test_sql_source_contributes_v_etf_gl(real_state):
    r = await real_state.tools.execute(
        "sql_query", {"sql": "select ticker, pd_nm, is_on_sale from v_etf_gl where is_on_sale=1 limit 5"}
    )
    assert not r.is_error
    assert r.evidence is not None
    assert r.evidence.kind == "sql" and r.evidence.table == "v_etf_gl"
    assert r.evidence.row_count and r.evidence.row_count > 0
    assert r.evidence.refs, "sql_query on v_etf_gl produced no row refs"
    assert "Total rows: 5" in r.content  # real rows came back, not an error/empty message


async def test_graph_source_contributes_samsung_constituents(real_state):
    r = await real_state.tools.execute("graph_neighbors", {"entity": "삼성전자", "relation": "constituentOf"})
    assert not r.is_error
    assert r.evidence is not None and r.evidence.kind == "graph"
    assert r.evidence.row_count and r.evidence.row_count > 0
    assert "constituentOf" in r.content


async def test_lexical_and_vector_source_contributes_hybrid_search(real_state):
    r = await real_state.tools.execute("search_products", {"query": "로봇", "mode": "hybrid"})
    assert not r.is_error
    assert r.evidence is not None and r.evidence.kind == "search"
    assert r.evidence.row_count and r.evidence.row_count > 0
    assert r.evidence.refs, "search_products(mode=hybrid) produced no row refs"


async def test_docs_source_contributes_etf_risk_factors(real_state):
    r = await real_state.tools.execute("search_documents", {"query": "ETF 위험요인"})
    assert not r.is_error
    assert r.evidence is not None and r.evidence.kind == "docs"
    assert r.evidence.row_count and r.evidence.row_count > 0
    assert "위험" in r.content


# ------------------------------------------------------------------ cross-source: one answer, two sources cited
async def test_cross_source_answer_cites_graph_and_sql(real_state):
    """A stub script that calls the graph, then SQL, then final_answer: retrieved_context must cite both."""
    script = [
        Turn("", [ToolCall("c1", "graph_neighbors", {"entity": "삼성전자", "relation": "constituentOf"})], "tool_use"),
        Turn("", [ToolCall(
            "c2", "sql_query",
            {"sql": "select ticker, pd_nm from v_etf_gl where is_on_sale=1 limit 3", "purpose": "example foreign ETFs"},
        )], "tool_use"),
        Turn("", [ToolCall("c3", "final_answer", {
            "status": "answered",
            "answer": "삼성전자를 담고 있는 국내 ETF는 그래프에서 229건 확인되며, 해외 ETF 예시는 다음과 같습니다.",
            "evidence_refs": ["graph constituentOf", "v_etf_gl"],
        })], "tool_use"),
    ]
    _script(real_state, script)
    resp = await real_state.orchestrator.answer("X-FED", "삼성전자를 보유한 국내 ETF는 몇 개이고, 해외 ETF 예시도 알려줘")

    assert "Knowledge graph" in resp.retrieved_context   # graph's evidence label (answer.format_evidence)
    assert "v_etf_gl" in resp.retrieved_context            # sql's evidence names the view
    assert "[Evidence 1]" in resp.retrieved_context and "[Evidence 2]" in resp.retrieved_context


# ------------------------------------------------------------------ routing prefetch: holdings -> graph, descriptive -> docs
# Round-0 prefetch is not in retrieved_context (see module docstring), so these assert on think_trace's
# "Round 0 routing:" line and the source that ran, per RoutePlan.summary() in agent/routing.py.

async def test_routing_prefetch_runs_graph_for_korean_holdings_question(real_state):
    _script(real_state, [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "1개"})], "tool_use")])
    resp = await real_state.orchestrator.answer("H-KO", "삼성전자를 보유한 국내 ETF는 몇 개인가요?")
    assert "Round 0 routing:" in resp.think_trace
    assert "graph=ran(" in resp.think_trace
    assert "graph_neighbors('삼성전자'" in resp.think_trace


async def test_routing_prefetch_runs_graph_for_english_holdings_question(real_state):
    _script(real_state, [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "1 ETF"})], "tool_use")])
    resp = await real_state.orchestrator.answer("H-EN", "Which domestic ETFs hold 삼성전자?")
    assert "Round 0 routing:" in resp.think_trace
    assert "graph=ran(" in resp.think_trace
    assert "graph_neighbors('삼성전자'" in resp.think_trace


async def test_routing_prefetch_runs_docs_for_korean_descriptive_question(real_state):
    _script(real_state, [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "위험요인은 ..."})], "tool_use")])
    resp = await real_state.orchestrator.answer("D-KO", "ETF의 위험요인은 무엇인가요?")
    assert "Round 0 routing:" in resp.think_trace
    assert "document=ran(" in resp.think_trace
    assert "search_documents(" in resp.think_trace


async def test_routing_prefetch_runs_docs_for_english_descriptive_question(real_state):
    _script(real_state, [Turn("", [ToolCall("c", "final_answer", {"status": "answered", "answer": "Risk factors ..."})], "tool_use")])
    resp = await real_state.orchestrator.answer("D-EN", "What are the key risk factors of ETFs?")
    assert "Round 0 routing:" in resp.think_trace
    assert "document=ran(" in resp.think_trace
    assert "search_documents(" in resp.think_trace
