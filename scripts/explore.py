"""Poke every subsystem by hand, with no LLM and no API key.

This is the fastest way to understand (or debug) what the agent sees before it writes an answer.

  python scripts/explore.py health
  python scripts/explore.py sql    "SELECT pd_no, pd_nm, crd_grd FROM v_bond_kr WHERE is_sellable=1 LIMIT 5"
  python scripts/explore.py search "차이나 반도체"            [--mode hybrid|lexical|vector] [--table v_etf_kr]
  python scripts/explore.py entity "KODEX AI로봇"
  python scripts/explore.py graph  "에코프로"                 [--relation hasSubsidiary]
  python scripts/explore.py sparql "SELECT ?n WHERE { ?p a fp:ETF ; fp:name ?n } LIMIT 5"
  python scripts/explore.py docs   "커버드콜 위험"
  python scripts/explore.py card   v_etf_gl
  python scripts/explore.py pre    "에코프로의 자회사를 편입한 ETF 중 순자산이 큰 상품의 위험요인 알려줘"

`pre` prints the deterministic pre-analysis block that gets injected into the LLM's first message —
useful for checking entity resolution and the ontology validators without spending a token.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from finagent.agent.llm import StubLLM  # noqa: E402
from finagent.app_state import AppState  # noqa: E402
from finagent.config import get_settings  # noqa: E402

NEEDS_GRAPH = {"graph", "sparql", "pre", "health"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect the agent's knowledge stores without an LLM.")
    ap.add_argument("command", choices=["health", "sql", "search", "entity", "graph", "sparql", "docs", "card", "pre"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "vector"])
    ap.add_argument("--table", default=None)
    ap.add_argument("--relation", default=None)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--quiet", action="store_true", help="hide the startup log")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    t0 = time.time()
    settings = get_settings()
    st = AppState.load(settings, llm=StubLLM([]), build_graph=a.command in NEEDS_GRAPH)
    if not a.quiet:
        print(f"-- loaded in {time.time() - t0:.0f}s --\n")

    if a.command == "health":
        m = st.rdb.meta()
        for view, info in m.items():
            print(f"{view:12} {info['korean_name']:10} base date {info['base_date']}  {info['row_count']:,} rows")
        print(f"\nlexical index   {len(st.index.docs):,} docs")
        if st.vectors is not None:
            print(f"vector index    {st.vectors.stats.docs:,} docs · {st.vectors.stats.dims} dims · {st.vectors.stats.backend}"
                  f"{' (cached)' if st.vectors.stats.cached else ''}")
        if st.kg is not None:
            print(f"knowledge graph {st.kg.size:,} triples (asserted {st.kg.asserted_triples:,} + inferred {st.kg.inferred_triples:,})")
        print(f"external docs   {len(st.docs)} chunks")
        print(f"ontology        {len(st.ontology.graph):,} triples" if getattr(st.ontology, "graph", None) is not None else "ontology        not loaded")
        return 0

    if not a.arg:
        ap.error(f"'{a.command}' needs an argument")

    async def run() -> int:
        if a.command == "sql":
            r = await st.tools.execute("sql_query", {"sql": a.arg})
        elif a.command == "search":
            r = await st.tools.execute("search_products", {"query": a.arg, "mode": a.mode, "table": a.table, "k": a.k})
        elif a.command == "entity":
            r = await st.tools.execute("resolve_entity", {"mention": a.arg})
        elif a.command == "graph":
            r = await st.tools.execute("graph_neighbors", {"entity": a.arg, "relation": a.relation})
        elif a.command == "sparql":
            r = await st.tools.execute("graph_query", {"sparql": a.arg})
        elif a.command == "docs":
            r = await st.tools.execute("search_documents", {"query": a.arg, "k": a.k})
        elif a.command == "card":
            r = await st.tools.execute("table_info", {"table": a.arg})
        else:  # pre
            print(st.orchestrator.pre_analyze(a.arg).text)
            return 0
        print(("[error] " if r.is_error else "") + r.content)
        if r.evidence is not None:
            e = r.evidence
            print(f"\n-- what gets recorded as evidence --\nkind {e.kind} · table {e.table} · base date {e.base_date} · rows {e.row_count}")
            print(f"executed: {e.detail}")
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
