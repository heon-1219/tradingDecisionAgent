"""지식그래프가 무엇으로 만들어졌는지, 그리고 맞게 만들어졌는지 검증한다.

두 가지를 본다.
 1) 출처(provenance): 술어별 트리플 수를 세고 주최측 마스터 / 우리 사전 / 외부 수집 / 온톨로지로 분류한다.
    "이 답의 근거가 대회가 준 데이터인가, 우리가 만든 것인가"를 구분할 수 없으면 근거를 신뢰할 수 없다.
 2) 정합성: 노드 수가 원천 테이블과 일치하는지, 이름으로 이어붙인 관계가 실제로 맞는지 표본 검사한다.

  python scripts/audit_graph.py
  python scripts/audit_graph.py --sample 20     # 표본 검사 개수
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

# Where each predicate's facts come from.  주최측 = the four master files the organizers supplied;
# 사전 = our curated YAML; 외부 = the Naver constituent crawl; 온톨로지 = TBox + materialised inference.
PROVENANCE = {
    "주최측 마스터": {
        "productCode", "name", "shortName", "productType", "isOnSale", "baseDate", "ticker",
        "hasCreditGrade", "bondMidClass", "bondKind", "maturityDate", "tradingMarket", "issuedBy",
        "hasRiskGrade", "listingDate", "netAssets", "distributionCycle", "expenseRatio",
        "memberOfFamily", "isHedged", "hasAssetClass", "investsInRegion", "leverageFactor",
        "strategy", "returns1y",
    },
    "사전(우리 큐레이션)": {
        "relatedTheme", "hasTheme", "subsidiaryOf", "hasSubsidiary", "englishName", "country",
        "isListed", "group", "managedBy", "manages",
    },
    "외부 수집(네이버)": {"hasConstituent", "constituentOf"},
    "온톨로지/추론": {"type", "subClassOf", "label", "comment", "domain", "range", "inverseOf"},
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

    print(f"\n{BAR}\n1. 규모\n{BAR}")
    print(f"트리플 {len(g):,}")

    # ---------------------------------------------------------------- 출처
    print(f"\n{BAR}\n2. 술어별 출처\n{BAR}")
    per_pred = collections.Counter(local(p) for _, p, _ in g)
    per_src: collections.Counter = collections.Counter()
    unknown: list[tuple[str, int]] = []
    for pred, n in per_pred.items():
        src = SOURCE_OF.get(pred)
        if src is None:
            src = "분류 안 됨"
            unknown.append((pred, n))
        per_src[src] += n
    total = sum(per_src.values())
    for src, n in per_src.most_common():
        print(f"  {src:20} {n:10,}  ({n / total * 100:5.1f}%)")
    print("\n  술어 상위 20:")
    for pred, n in per_pred.most_common(20):
        print(f"    {pred:24} {n:9,}   [{SOURCE_OF.get(pred, '분류 안 됨')}]")
    if unknown:
        print(f"\n  ⚠ 분류표에 없는 술어 {len(unknown)}개: {[p for p, _ in unknown[:12]]}")

    # ---------------------------------------------------------------- 노드 수 정합성
    print(f"\n{BAR}\n3. 노드 수가 원천과 맞는가\n{BAR}")
    checks = [
        ("국내채권", "v_bond_kr", "SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr", "bond/"),
        ("국내ETF/ETN", "v_etf_kr", "SELECT COUNT(*) FROM v_etf_kr", "etf_kr/"),
        ("해외ETF/ETN", "v_etf_gl", "SELECT COUNT(*) FROM v_etf_gl", "etf_gl/"),
        ("공모펀드", "v_fund_pub", "SELECT COUNT(*) FROM v_fund_pub", "fund/"),
    ]
    for label, _view, sql, prefix in checks:
        want = rdb.query(sql).rows[0]
        want = list(want.values())[0]
        got = len({str(s) for s in g.subjects() if f"/data/{prefix}" in str(s)})
        mark = "일치 ✅" if got == want else f"불일치 ❌ (차이 {got - want:+,})"
        print(f"  {label:12} 원천 {want:7,} · 그래프 {got:7,}  {mark}")

    # ---------------------------------------------------------------- 이름으로 이어붙인 관계
    print(f"\n{BAR}\n4. 이름 매칭으로 만들어진 관계 (오염이 생기는 지점)\n{BAR}")
    fp = rdflib.Namespace("http://mafest.ai/product#")
    for pred_name in ("hasTheme", "managedBy", "hasConstituent"):
        n = sum(1 for _ in g.triples((None, fp[pred_name], None)))
        print(f"  {pred_name:16} {n:8,} 트리플")
    return 0


if __name__ == "__main__":
    sys.exit(main())
