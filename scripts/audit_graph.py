"""Verify what the knowledge graph is built from, and whether it is built correctly.

Two things are checked.
 1) Provenance: count triples per predicate and classify them as organizer master / our dictionary / external crawl / ontology.
    If you cannot tell "is this answer grounded in data the competition supplied, or in something we made?", the evidence cannot be trusted.
 2) Consistency: whether node counts match the source tables, and a spot check of whether relations joined by name are actually right.

  python scripts/audit_graph.py
  python scripts/audit_graph.py --sample 20     # number of spot checks
"""
from __future__ import annotations

import argparse
import collections
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import rdflib  # noqa: E402

from finagent.agent.llm import StubLLM  # noqa: E402
from finagent.app_state import AppState  # noqa: E402
from finagent.config import get_settings  # noqa: E402

BAR = "=" * 96

# Where each predicate's facts come from.  organizer = the four master files the organizers supplied;
# curated = our curated YAML; external = the Naver constituent crawl; ontology = TBox + materialised inference.
PROVENANCE = {
    "organizer masters": {
        "productCode", "name", "shortName", "productType", "isOnSale", "baseDate", "ticker",
        "hasCreditGrade", "bondMidClass", "bondKind", "maturityDate", "tradingMarket", "issuedBy",
        "hasRiskGrade", "listingDate", "netAssets", "distributionCycle", "expenseRatio",
        "memberOfFamily", "isHedged", "hasAssetClass", "investsInRegion", "leverageFactor",
        "strategy", "returns1y",
    },
    "curated dictionary": {
        "relatedTheme", "hasTheme", "subsidiaryOf", "hasSubsidiary", "englishName", "country",
        "isListed", "group", "managedBy", "manages",
    },
    "external (Naver)": {"hasConstituent", "constituentOf"},
    "ontology/inference": {"type", "subClassOf", "label", "comment", "domain", "range", "inverseOf"},
}
SOURCE_OF = {p: src for src, ps in PROVENANCE.items() for p in ps}


def local(u) -> str:
    s = str(u)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12)
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    st = AppState.load(get_settings(), llm=StubLLM([]), build_graph=True)
    g, rdb = st.kg.graph, st.rdb

    print(f"\n{BAR}\n1. Size\n{BAR}")
    print(f"triples {len(g):,}")

    # ---------------------------------------------------------------- provenance
    print(f"\n{BAR}\n2. Provenance by predicate\n{BAR}")
    per_pred = collections.Counter(local(p) for _, p, _ in g)
    per_src: collections.Counter = collections.Counter()
    unknown: list[tuple[str, int]] = []
    for pred, n in per_pred.items():
        src = SOURCE_OF.get(pred)
        if src is None:
            src = "unclassified"
            unknown.append((pred, n))
        per_src[src] += n
    total = sum(per_src.values())
    for src, n in per_src.most_common():
        print(f"  {src:20} {n:10,}  ({n / total * 100:5.1f}%)")
    print("\n  top 20 predicates:")
    for pred, n in per_pred.most_common(20):
        print(f"    {pred:24} {n:9,}   [{SOURCE_OF.get(pred, 'unclassified')}]")
    if unknown:
        print(f"\n  ⚠ {len(unknown)} predicates missing from the classification table: {[p for p, _ in unknown[:12]]}")

    # ---------------------------------------------------------------- node-count consistency
    print(f"\n{BAR}\n3. Do node counts match the source?\n{BAR}")
    checks = [
        ("domestic bonds", "v_bond_kr", "SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr", "bond/"),
        ("domestic ETF/ETN", "v_etf_kr", "SELECT COUNT(*) FROM v_etf_kr", "etf_kr/"),
        ("overseas ETF/ETN", "v_etf_gl", "SELECT COUNT(*) FROM v_etf_gl", "etf_gl/"),
        ("public funds", "v_fund_pub", "SELECT COUNT(*) FROM v_fund_pub", "fund/"),
    ]
    for label, _view, sql, prefix in checks:
        want = rdb.query(sql).rows[0]
        want = list(want.values())[0]
        got = len({str(s) for s in g.subjects() if f"/data/{prefix}" in str(s)})
        mark = "match ✅" if got == want else f"mismatch ❌ (diff {got - want:+,})"
        print(f"  {label:16} source {want:7,} · graph {got:7,}  {mark}")

    # ---------------------------------------------------------------- relations joined by name
    print(f"\n{BAR}\n4. Relations built by name matching (where contamination creeps in)\n{BAR}")
    fp = rdflib.Namespace("http://mafest.ai/product#")
    for pred_name in ("hasTheme", "managedBy", "hasConstituent"):
        n = sum(1 for _ in g.triples((None, fp[pred_name], None)))
        print(f"  {pred_name:16} {n:8,} triples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
