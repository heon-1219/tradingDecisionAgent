#!/usr/bin/env python
"""벡터(의미) 색인 감사 — "Vector 레그가 실제로 무엇을 하는가"를 숫자로 답한다.

기술제안서 부록의 "평가 재현 절차"로 쓸 수 있도록, LLM 을 한 번도 호출하지 않고 결정적으로
(고정 시드) 돌아가게 만들었다.  스토어는 읽기 전용으로만 열고, vector_cache.npz 는 서명이
일치할 때만 읽으며 **어떤 경우에도 새로 쓰지 않는다**.

측정 항목
  A. 색인 구성      — 소스별 문서 수, 문서 텍스트 표본, 길이 분포
  B. 검색 품질      — 정답이 자명한 자동 생성 질의로 recall@1/5/10 · MRR
  C. BM25 vs 벡터   — 어휘 단독 / 벡터 단독 / 실제 결합(hybrid_search) 비교, 벡터 구제 사례
  D. 캐시 무결성    — 현재 문서 집합으로 계산한 서명과 캐시 서명의 일치 여부
  E. 도구 호출 실측 — data/eval/*.jsonl 의 think_trace 에서 도구별 호출 횟수
  G. 외부 문서      — search_documents 대상 청크 목록과 주최측 예시 문항 응답 가능성

사용법
  python scripts/audit_vector.py                    # 전 섹션 (약 15분: B/C 의 질의 1,000여 개가 대부분)
  python scripts/audit_vector.py --sections ADEG    # 색인 구성·캐시·도구호출·문서 (약 1분)
  python scripts/audit_vector.py --sections BC      # 검색 품질만 (약 14분)
  python scripts/audit_vector.py --n 40 --seed 0    # 상품군별 질의 수 / 시드 (기본 40, 시드 0)

측정된 기준값(2026-09-06, 문서 75,089개): 어휘색인 구축 9~18초, TF-IDF fit 9.4초,
캐시 로드 1.0초, 벡터 상주 메모리 174MB(matrix 115MB + components 59MB).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
import time
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

from finagent.config import get_settings  # noqa: E402
from finagent.kb.docs import DocStore  # noqa: E402
from finagent.kb.lexicon import Lexicon, norm  # noqa: E402
from finagent.kb.rdb import RDB  # noqa: E402
from finagent.kb.search import INDEX_SPECS, SearchIndex  # noqa: E402
from finagent.kb.vector import DEFAULT_MAX_DF, VectorIndex, hybrid_search  # noqa: E402

BAR = "=" * 100
SUB = "-" * 100

PRODUCT_VIEWS = ["v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub"]
VIEW_LABEL = {
    "v_bond_kr": "국내채권",
    "v_etf_kr": "국내ETF/ETN",
    "v_etf_gl": "해외ETF",
    "v_fund_pub": "공모/사모펀드",
    "v_etf_constituent": "국내ETF 편입종목",
}
# 검색에 쓰는 표면형(코드/티커)이 그 뷰의 색인 text 에 들어 있는가 — INDEX_SPECS 에서 읽어 확인한다.
CODE_COL = {
    "v_bond_kr": "pd_no",
    "v_etf_kr": "pd_ticker",
    "v_etf_gl": "ticker",
    "v_fund_pub": "itm_no",
}


# --------------------------------------------------------------------------- 공통
def pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def table(rows: list[list[str]], head: list[str]) -> str:
    cols = len(head)
    w = [max(len(str(head[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(head[i])) for i in range(cols)]
    out = ["  ".join(str(head[i]).ljust(w[i]) for i in range(cols))]
    out.append("  ".join("-" * w[i] for i in range(cols)))
    for r in rows:
        out.append("  ".join(str(r[i]).ljust(w[i]) for i in range(cols)))
    return "\n".join(out)


def corpus_signature(docs, dims: int, max_df: float) -> str:
    """vector.VectorIndex.build 와 **동일한** 서명 계산 (문서 텍스트까지 해싱)."""
    sep = bytes([31])
    h = hashlib.sha1(f"{len(docs)}|{dims}|{max_df}".encode("utf-8"))
    for d in docs:
        h.update(f"{d.title} {d.text}".encode("utf-8", "ignore"))
        h.update(sep)
    return h.hexdigest()[:16]


def load_indexes(settings, *, allow_slow_rebuild: bool):
    """어휘 색인 + 벡터 색인.  캐시 서명이 일치할 때만 cache_path 를 넘겨 **쓰기를 원천 차단**한다."""
    t0 = time.time()
    rdb = RDB(settings.store_path, row_limit=settings.sql_row_limit, timeout_sec=settings.sql_timeout_sec)
    index = SearchIndex.from_rdb(rdb)
    t_lex = time.time() - t0

    sig = corpus_signature(index.docs, settings.vector_dims, settings.vector_max_df)
    cache_ok, cache_info = inspect_cache(settings, sig, len(index.docs))

    t1 = time.time()
    vectors = None
    if cache_ok:
        vectors = VectorIndex.build(
            index.docs, dims=settings.vector_dims, max_df=settings.vector_max_df,
            cache_path=settings.vector_cache_path, log_fn=lambda *a: None,
        )
    elif allow_slow_rebuild:
        # 서명 불일치 → 캐시를 절대 갱신하지 않도록 cache_path 없이 메모리에서만 다시 만든다.
        vectors = VectorIndex.build(
            index.docs, dims=settings.vector_dims, max_df=settings.vector_max_df,
            cache_path=None, log_fn=lambda *a: None,
        )
    t_vec = time.time() - t1
    return rdb, index, vectors, sig, cache_ok, cache_info, t_lex, t_vec


def inspect_cache(settings, signature: str, n_docs: int):
    p = settings.vector_cache_path
    info: dict = {"path": str(p), "exists": p.exists(), "computed_signature": signature}
    if not p.exists():
        return False, info
    info["size_mb"] = round(p.stat().st_size / 1e6, 1)
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        info["error"] = repr(e)
        return False, info
    info["keys"] = list(z.keys())
    info["cache_signature"] = str(z["signature"])
    info["cache_dims"] = int(z["dims"])
    info["matrix_shape"] = tuple(int(x) for x in z["matrix"].shape)
    info["components_shape"] = tuple(int(x) for x in z["components"].shape)
    info["signature_match"] = info["cache_signature"] == signature
    info["dims_match"] = info["cache_dims"] == settings.vector_dims
    info["rows_match"] = info["matrix_shape"][0] == n_docs
    ok = bool(info["signature_match"] and info["dims_match"] and info["rows_match"])
    return ok, info


# --------------------------------------------------------------------------- A
def section_a(rdb, index, settings, t_lex, t_vec, vectors, seed: int) -> None:
    print(BAR)
    print("A. 색인에 무엇이 들어 있는가")
    print(BAR)
    by_table = collections.Counter(d.table for d in index.docs)
    meta = rdb.meta()
    rows = []
    for view in [*PRODUCT_VIEWS, "v_etf_constituent"]:
        raw = rdb.scalar(f"SELECT COUNT(*) FROM {view}")
        where = INDEX_SPECS[view][4]
        note = f"필터 {where}" if where else "-"
        rows.append([view, VIEW_LABEL[view], f"{raw:,}", f"{by_table.get(view, 0):,}",
                     str(meta.get(view, {}).get("base_date") or "-"), note])
    print(table(rows, ["뷰", "상품군", "원천 행수", "색인 문서수", "기준일", "행 필터"]))
    print(f"\n합계: 색인 문서 {sum(by_table.values()):,}개 / 원천 행 "
          f"{sum(rdb.scalar(f'SELECT COUNT(*) FROM {v}') for v in by_table):,}개")
    print(f"어휘색인 구축 {t_lex:.1f}초 · 벡터색인 적재 {t_vec:.1f}초"
          + (f" (backend={vectors.stats.backend}, dims={vectors.stats.dims}, cached={vectors.stats.cached})"
             if vectors is not None else " (벡터 없음)"))

    print(f"\n{SUB}\n문서 텍스트 표본 (소스별 3건, 임베딩에 실제로 들어가는 문자열)\n{SUB}")
    rng = random.Random(seed)
    for view in [*PRODUCT_VIEWS, "v_etf_constituent"]:
        pool = [d for d in index.docs if d.table == view]
        print(f"\n[{view}] {VIEW_LABEL[view]} — text 컬럼: {', '.join(INDEX_SPECS[view][2])}")
        for d in rng.sample(pool, min(3, len(pool))):
            body = f"{d.title} {d.text}"
            print(f"  id={d.id} | len={len(body)}자")
            print(f"    {body[:300]}{'…' if len(body) > 300 else ''}")

    print(f"\n{SUB}\n문서 길이 분포 (title + text, 문자 수)\n{SUB}")
    rows = []
    short_total = 0
    for view in [*PRODUCT_VIEWS, "v_etf_constituent"]:
        lens = [len(f"{d.title} {d.text}") for d in index.docs if d.table == view]
        lens.sort()
        n = len(lens)
        short = sum(1 for x in lens if x <= 20)
        short_total += short
        rows.append([
            view, f"{n:,}", str(min(lens)), str(lens[n // 10]), str(int(statistics.median(lens))),
            str(lens[int(n * 0.9)]), str(max(lens)), f"{short:,} ({100*short/n:.1f}%)",
        ])
    lens_all = sorted(len(f"{d.title} {d.text}") for d in index.docs)
    n = len(lens_all)
    rows.append(["(전체)", f"{n:,}", str(min(lens_all)), str(lens_all[n // 10]),
                 str(int(statistics.median(lens_all))), str(lens_all[int(n * 0.9)]), str(max(lens_all)),
                 f"{short_total:,} ({100*short_total/n:.1f}%)"])
    print(table(rows, ["뷰", "문서", "최소", "p10", "중앙값", "p90", "최대", "20자 이하"]))

    dup = collections.Counter(f"{d.title} {d.text}" for d in index.docs)
    ndup = sum(c - 1 for c in dup.values() if c > 1)
    print(f"\n완전히 동일한 텍스트를 가진 중복 문서: {ndup:,}개 "
          f"(고유 텍스트 {len(dup):,} / 전체 {len(index.docs):,})")
    top = [(t, c) for t, c in dup.most_common(5)]
    for t, c in top:
        print(f"  x{c:4d}  {t[:90]}")


# --------------------------------------------------------------------------- B/C 질의 생성
_SPLIT = re.compile(r"[\s()\[\]·,/]+")


def abbrev_of(title: str) -> str | None:
    """'TIGER 미국나스닥100' -> '미국나스닥100' (브랜드 토큰 제거), 공백이 없으면 중간 구간을 자른다."""
    s = str(title or "").strip()
    toks = [t for t in _SPLIT.split(s) if t]
    if len(toks) >= 2 and len("".join(toks[1:])) >= 3:
        cand = " ".join(toks[1:])
        return cand if norm(cand) != norm(s) else None
    if len(s) >= 10:
        a = len(s) // 5
        b = max(a + 5, int(len(s) * 0.85))
        cand = s[a:b]
        return cand if norm(cand) != norm(s) and len(cand) >= 4 else None
    return None


def typo_of(s: str, rng: random.Random) -> str | None:
    """한 글자 삭제 또는 인접 두 글자 교환 — 어휘검색이 깨지는 전형적 상황."""
    s = str(s or "").strip()
    if len(s) < 6:
        return None
    i = rng.randrange(2, len(s) - 2)
    if rng.random() < 0.5:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_queries(rdb, index, lexicon: Lexicon, n: int, seed: int) -> dict[str, list[dict]]:
    """정답이 자명한 질의를 데이터에서 자동 생성한다.  각 항목: {query, kind, view, gold(fn), label}"""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = collections.defaultdict(list)

    # ---- 상품 유형 3종: 정식명칭 / 코드·티커 / 부분명(약칭)
    for view in PRODUCT_VIEWS:
        idc, titlec = INDEX_SPECS[view][0], INDEX_SPECS[view][1]
        code = CODE_COL[view]
        cols = list(dict.fromkeys([idc, titlec, code]))
        rows = [dict(zip(cols, r)) for r in
                rdb.fetch_all(f"SELECT {', '.join(cols)} FROM {view} WHERE {titlec} IS NOT NULL AND {titlec} <> ''")]
        # 같은 이름이 여러 상품에 걸치면 '정답 1개'가 성립하지 않으므로 고유 제목만 쓴다.
        by_title = collections.Counter(norm(r[titlec]) for r in rows)
        uniq = [r for r in rows if by_title[norm(r[titlec])] == 1]
        rng.shuffle(uniq)

        def gold_for(gid: str, v=view):
            return lambda h: h.table == v and str(h.id) == str(gid)

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            out["정식명칭"].append({"query": str(r[titlec]), "kind": "정식명칭", "view": view,
                                    "gold": gold_for(r[idc]), "label": str(r[titlec])})
            picked += 1

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            v = r.get(code)
            if v in (None, "") or len(str(v)) < 3:
                continue
            out["코드·티커"].append({"query": str(v), "kind": "코드·티커", "view": view,
                                     "gold": gold_for(r[idc]), "label": f"{v} → {r[titlec]}"})
            picked += 1

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            ab = abbrev_of(str(r[titlec]))
            if not ab:
                continue
            out["부분명(약칭)"].append({"query": ab, "kind": "부분명(약칭)", "view": view,
                                        "gold": gold_for(r[idc]), "label": f"{ab} → {r[titlec]}"})
            picked += 1

    # ---- 편입종목명 → 그 종목을 담은 편입종목 문서
    comps = [c for (c,) in rdb.fetch_all(
        "SELECT DISTINCT company_name FROM v_etf_constituent WHERE is_cash=0 AND company_name IS NOT NULL")]
    rng.shuffle(comps)
    for c in comps[: n * 2]:
        if len(out["편입종목명"]) >= n * 2:
            break
        nm = norm(c)

        def gold_c(h, nm=nm):
            return h.table == "v_etf_constituent" and norm(h.payload.get("company_name")) == nm

        out["편입종목명"].append({"query": str(c), "kind": "편입종목명", "view": "v_etf_constituent",
                                  "gold": gold_c, "label": str(c)})

    # ---- 테마어 → 그 테마를 이름에 드러낸 상품
    # 정답 정의: 상품 제목(정규화)에 그 테마의 별칭이 들어 있는 상품.  ASCII 별칭은 단어 경계로 본다.
    prod_titles = [(i, d) for i, d in enumerate(index.docs) if d.table in PRODUCT_VIEWS]
    ntitles = [(norm(d.title), str(d.title).lower(), d.table, str(d.id)) for _, d in prod_titles]

    def alias_hits(alias: str) -> set[tuple[str, str]]:
        a = str(alias).strip()
        if len(a) < 2:
            return set()
        if all(ord(ch) < 128 for ch in a):
            tail = r"(?![a-z])" if a[-1:].isalpha() else r"(?![a-z0-9])"
            rx = re.compile(r"(?<![a-z0-9])" + r"\s*".join(re.escape(w) for w in a.lower().split()) + tail)
            return {(t, i) for (_, low, t, i) in ntitles if rx.search(low)}
        na = norm(a)
        return {(t, i) for (nt, _, t, i) in ntitles if na in nt}

    for th in lexicon.themes:
        aliases = [th.label_ko, th.label_en, *th.synonyms]
        hit_sets = {a: alias_hits(a) for a in aliases}
        gold_keys: set[tuple[str, str]] = set()
        for s in hit_sets.values():
            gold_keys |= s
        if len(gold_keys) < 3:
            continue  # 이름으로 그 테마임을 알 수 있는 상품이 사실상 없는 테마는 정답 정의가 성립하지 않는다

        def gold_t(h, keys=frozenset(gold_keys)):
            return (h.table, str(h.id)) in keys

        for a in aliases:
            out["테마어"].append({"query": a, "kind": "테마어", "view": "*", "gold": gold_t,
                                  "label": f"{a} (테마 {th.id}, 정답 {len(gold_keys)}건)",
                                  "theme": th.id, "alias_direct": len(hit_sets[a])})
    rng.shuffle(out["테마어"])
    # 별칭 수만큼 늘어나면 수백 개가 되어 재현 실행이 너무 길어진다. 유형별 최소 40개는 유지한다.
    out["테마어"] = out["테마어"][: max(n, 40) * 2]

    # ---- 오타 (정식명칭 변형) — 어휘검색이 실패하는 지점
    for q in list(out["정식명칭"]):
        t = typo_of(q["query"], rng)
        if t:
            out["오타"].append({**q, "query": t, "kind": "오타", "label": f"{t}  (원본: {q['query']})"})

    # ---- 유의어 공백 (표기가 데이터에 아예 없는 테마 별칭) — 의미 검색만 건질 수 있는 지점
    for q in out["테마어"]:
        if q.get("alias_direct", 1) == 0:
            out["유의어공백"].append({**q, "kind": "유의어공백"})

    return out


# --------------------------------------------------------------------------- B/C 실행
def run_modes(index, vectors, query: str, k: int = 10):
    lex = index.search(query, k=max(k, 20))
    vec = vectors.search(query, k=max(k, 20)) if vectors is not None else []
    hyb = hybrid_search(lex, vec, k=k)  # tools._t_search_products 와 동일한 호출
    return lex[:k], vec[:k], hyb


def rank_of(hits, gold) -> int | None:
    for i, h in enumerate(hits, start=1):
        if gold(h):
            return i
    return None


def score(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {
        "n": len(ranks),
        "r1": sum(1 for r in ranks if r == 1) / n,
        "r5": sum(1 for r in ranks if r and r <= 5) / n,
        "r10": sum(1 for r in ranks if r and r <= 10) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
    }


def eval_set(index, vectors, queries: list[dict], k: int = 10):
    res = {"lexical": [], "vector": [], "hybrid": []}
    detail = []
    for q in queries:
        lex, vec, hyb = run_modes(index, vectors, q["query"], k=k)
        rl, rv, rh = rank_of(lex, q["gold"]), rank_of(vec, q["gold"]), rank_of(hyb, q["gold"])
        res["lexical"].append(rl)
        res["vector"].append(rv)
        res["hybrid"].append(rh)
        detail.append({**{kk: q[kk] for kk in ("query", "kind", "view", "label")},
                       "lex": rl, "vec": rv, "hyb": rh,
                       "lex_top": lex[0].title if lex else None,
                       "vec_top": vec[0].title if vec else None})
    return {m: score(v) for m, v in res.items()}, detail


def print_scores(title: str, per_mode: dict) -> None:
    rows = [[m, str(s["n"]), pct(s["r1"]), pct(s["r5"]), pct(s["r10"]), f"{s['mrr']:.3f}"]
            for m, s in per_mode.items()]
    print(f"\n{title}")
    print(table(rows, ["모드", "질의수", "recall@1", "recall@5", "recall@10", "MRR"]))


def section_bc(index, vectors, rdb, lexicon, n: int, seed: int, k: int = 10) -> None:
    print(f"\n{BAR}")
    print("B/C. 검색 품질 — 어휘(BM25) / 벡터(TF-IDF+LSA) / 실제 결합(hybrid_search) 비교")
    print(BAR)
    qs = make_queries(rdb, index, lexicon, n=n, seed=seed)
    print("생성된 질의 (정답은 데이터에서 자동 도출):")
    print(table([[kind, f"{len(v):,}"] for kind, v in qs.items()], ["유형", "질의 수"]))

    all_detail: list[dict] = []
    summary_rows = []
    for kind in ["정식명칭", "코드·티커", "부분명(약칭)", "편입종목명", "테마어", "오타", "유의어공백"]:
        items = qs.get(kind) or []
        if not items:
            continue
        t0 = time.time()
        per_mode, detail = eval_set(index, vectors, items, k=k)
        all_detail += detail
        print(f"\n{SUB}")
        print(f"[{kind}]  질의 {len(items)}개 · {time.time()-t0:.1f}초")
        print(SUB)
        print_scores("", per_mode)
        for m in ("lexical", "vector", "hybrid"):
            s = per_mode[m]
            summary_rows.append([kind, m, str(s["n"]), pct(s["r1"]), pct(s["r5"]), pct(s["r10"]), f"{s['mrr']:.3f}"])

        # 상품군별 분해 (상품 3유형만)
        if kind in ("정식명칭", "코드·티커", "부분명(약칭)"):
            rows = []
            for view in PRODUCT_VIEWS:
                sub = [d for d in detail if d["view"] == view]
                if not sub:
                    continue
                for m, key in (("lexical", "lex"), ("vector", "vec"), ("hybrid", "hyb")):
                    s = score([d[key] for d in sub])
                    rows.append([view, m, str(s["n"]), pct(s["r1"]), pct(s["r10"]), f"{s['mrr']:.3f}"])
            print("\n상품군별 분해")
            print(table(rows, ["뷰", "모드", "질의수", "recall@1", "recall@10", "MRR"]))

    print(f"\n{BAR}")
    print("종합 비교표 (전 유형)")
    print(BAR)
    print(table(summary_rows, ["유형", "모드", "질의수", "recall@1", "recall@5", "recall@10", "MRR"]))

    # ---- 벡터가 실제로 무엇을 더했는가
    print(f"\n{BAR}")
    print("C-핵심. 벡터가 BM25 위에 더한 것 / 뺏은 것")
    print(BAR)
    rescue = [d for d in all_detail if d["lex"] is None and d["vec"] is not None]
    hyb_gain = [d for d in all_detail if d["lex"] is None and d["hyb"] is not None]
    hyb_loss = [d for d in all_detail if d["lex"] is not None and d["hyb"] is None]
    rank_up = [d for d in all_detail if d["lex"] and d["hyb"] and d["hyb"] < d["lex"]]
    rank_down = [d for d in all_detail if d["lex"] and d["hyb"] and d["hyb"] > d["lex"]]
    print(table([
        ["전체 질의", f"{len(all_detail):,}"],
        ["어휘 실패 & 벡터 성공 (top-10)", f"{len(rescue):,}"],
        ["어휘 실패 & 결합 성공 = 실제 구제", f"{len(hyb_gain):,}"],
        ["어휘 성공 & 결합 실패 = 결합이 잃은 것", f"{len(hyb_loss):,}"],
        ["결합이 순위를 올린 질의", f"{len(rank_up):,}"],
        ["결합이 순위를 내린 질의", f"{len(rank_down):,}"],
    ], ["항목", "건수"]))
    if hyb_gain:
        print("\n실제 구제 사례 (어휘 top-10 실패 → 결합 top-10 성공):")
        for d in hyb_gain[:20]:
            print(f"  [{d['kind']}] {d['label'][:70]}  | 벡터순위={d['vec']} 결합순위={d['hyb']} "
                  f"| 어휘 1위='{str(d['lex_top'])[:34]}'")
    else:
        print("\n실제 구제 사례: 0건 — 어휘가 실패한 질의를 결합 결과가 건진 경우가 한 건도 없다.")
    if hyb_loss:
        print("\n결합이 잃은 사례 (어휘 top-10 성공 → 결합 top-10 실패):")
        for d in hyb_loss[:20]:
            print(f"  [{d['kind']}] {d['label'][:70]}  | 어휘순위={d['lex']}")

    # ---- README/제안서가 근거로 드는 개별 프로브를 그대로 재현한다
    print(f"\n{BAR}")
    print("C-2. 문서에 인용된 프로브 재현 (README '벡터 품질 조정' 절의 예시)")
    print(BAR)
    for q in ["space exploration satellite", "China semiconductor ETF", "차이나 반도체",
              "배터리 양극재 관련 상품", "우주항공", "월배당 커버드콜", "야놀자를 보유한 국내 ETF"]:
        lex, vec, hyb = run_modes(index, vectors, q, k=5)
        print(f"\n질의: {q}")
        print("  어휘 :", " | ".join(f"{h.title[:34]}" for h in lex[:5]) or "(0건)")
        print("  벡터 :", " | ".join(f"{h.title[:34]}" for h in vec[:5]) or "(0건)")
        print("  결합 :", " | ".join(f"{h.title[:34]}" for h in hyb[:5]) or "(0건)")


# --------------------------------------------------------------------------- D
def section_d(settings, sig: str, cache_ok: bool, info: dict, index, vectors) -> None:
    print(f"\n{BAR}")
    print("D. 캐시 무결성")
    print(BAR)
    rows = [
        ["캐시 경로", info.get("path", "-")],
        ["존재", str(info.get("exists"))],
        ["크기", f"{info.get('size_mb', '-')} MB"],
        ["저장 키", ", ".join(info.get("keys", []) or [])],
        ["현재 문서집합으로 계산한 서명", sig],
        ["캐시에 저장된 서명", str(info.get("cache_signature", "-"))],
        ["서명 일치", str(info.get("signature_match"))],
        ["dims 일치", f"{info.get('cache_dims')} vs {settings.vector_dims} → {info.get('dims_match')}"],
        ["행수 일치", f"{info.get('matrix_shape', ('-',))[0]} vs {len(index.docs)} → {info.get('rows_match')}"],
        ["matrix / components 형상", f"{info.get('matrix_shape')} / {info.get('components_shape')}"],
        ["종합 판정", "유효 (그대로 사용 가능)" if cache_ok else "낡음 (재빌드 필요)"],
    ]
    print(table(rows, ["항목", "값"]))

    print("\n검증 경로 점검 (vector.VectorIndex.build 의 캐시 수용 조건):")
    for line in [
        "  1) signature == 계산된 서명   — 문서 수 + dims + max_df + **모든 문서 텍스트**의 SHA1",
        "  2) dims 일치",
        "  3) components.shape[1] == 현재 TF-IDF 어휘 폭  (행렬곱 차원 불일치를 막는 안전장치)",
        "  4) matrix.shape[0] == 문서 수",
        "  네 조건 중 하나라도 어긋나면 warning 후 재빌드한다. 조건을 우회해 캐시를 쓰는 경로는 없다",
        "  (kb/vector.py 안에서 np.load 를 호출하는 지점은 이 한 곳뿐).",
    ]:
        print(line)

    if vectors is not None:
        # 캐시가 실제로 살아 있는지: 임의 질의가 예외 없이 정상 차원으로 돌아오는지 확인
        probe = vectors.search("반도체 ETF", k=5)
        print(f"\n실사용 확인: vectors.search('반도체 ETF') → {len(probe)}건, "
              f"backend={vectors.stats.backend}, dims={vectors.stats.dims}, cached={vectors.stats.cached}")
        m = vectors.matrix
        nrm = np.linalg.norm(m, axis=1)
        zero = int((nrm < 1e-6).sum())
        print(f"행렬: {m.shape} {m.dtype} · L2노름 중앙값 {float(np.median(nrm)):.3f} · 영벡터 문서 {zero:,}개 "
              f"({100.0*zero/max(1,m.shape[0]):.2f}%)")
        if zero:
            print("  ⚠ 영벡터 문서는 어떤 질의로도 벡터검색에 걸리지 않는다 (max_df 로 토큰이 전부 잘린 문서).")


# --------------------------------------------------------------------------- E
_ROUND = re.compile(r"라운드\s*\d+:\s*([a-z_]+)\s*\(")
_NOTOOL = re.compile(r"라운드\s*\d+:\s*도구 호출 없이")
KNOWN_TOOLS = ["sql_query", "search_products", "search_documents", "resolve_entity",
               "table_info", "graph_neighbors", "graph_query", "final_answer"]


def parse_traces(path: pathlib.Path):
    recs = []
    if not path.exists():
        return recs
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        calls: list[str] = []
        no_tool = 0
        for line in (r.get("think_trace") or "").splitlines():
            m = _ROUND.search(line)
            if m:
                calls.append(m.group(1))
            elif _NOTOOL.search(line):
                no_tool += 1
        recs.append({"qid": r.get("question_id"), "question": r.get("question"), "calls": calls,
                     "no_tool_rounds": no_tool, "ok": r.get("ok"), "answer": r.get("answer") or "",
                     "context": r.get("retrieved_context") or "", "trace": r.get("think_trace") or "",
                     "seconds": r.get("seconds")})
    return recs


def section_e(root: pathlib.Path) -> None:
    print(f"\n{BAR}")
    print("E. 도구가 실제로 불리는가 — 기록된 실행에서 다시 센 값")
    print(BAR)
    truth = parse_traces(root / "data" / "eval" / "truth_results.jsonl")
    fail = parse_traces(root / "data" / "eval" / "fail_results.jsonl")
    fail_ids = {r["qid"] for r in fail}
    # fail_results 는 truth_results 의 일부 문항을 더 최신으로 다시 돌린 것 → 최신본 우선
    merged = [r for r in truth if r["qid"] not in fail_ids] + fail

    for name, recs in (("truth_results.jsonl (54문항)", truth), ("fail_results.jsonl (재실행)", fail),
                       ("병합 = 문항별 최신 실행", merged)):
        c = collections.Counter(t for r in recs for t in r["calls"])
        rows = [[t, f"{c.get(t, 0):,}", f"{sum(1 for r in recs if t in r['calls']):,}"] for t in KNOWN_TOOLS]
        other = [t for t in c if t not in KNOWN_TOOLS]
        for t in other:
            rows.append([f"{t} (미등록)", f"{c[t]:,}", "-"])
        print(f"\n{SUB}\n{name} — 문항 {len(recs)}개, 도구 호출 {sum(c.values())}회, "
              f"'도구 호출 없이 텍스트 응답' 라운드 {sum(r['no_tool_rounds'] for r in recs)}회\n{SUB}")
        print(table(rows, ["도구", "호출 횟수", "이 도구를 쓴 문항 수"]))

    print(f"\n{SUB}\n벡터 계열 도구 (search_products / search_documents / resolve_entity) 호출 상세\n{SUB}")
    hits = [r for r in merged if any(t in r["calls"] for t in ("search_products", "search_documents", "resolve_entity"))]
    if not hits:
        print("호출 0회 — 54문항 전체에서 한 번도 불리지 않았다.")
    for r in hits:
        print(f"\n  {r['qid']} · ok={r['ok']} · {r['question'][:70]}")
        for line in r["trace"].splitlines():
            if any(t in line for t in ("search_products", "search_documents", "resolve_entity")):
                print(f"    {line.strip()[:150]}")
        print(f"    최종 답변: {r['answer'][:150]}")
        print(f"    근거(retrieved_context)에 검색 근거 포함? "
              f"{'예' if ('search' in r['context'] or '문서' in r['context']) else '아니오'}")

    # 참고: 각 문항이 몇 라운드를 썼는지 (도구를 부를 기회가 있었는가)
    rounds = collections.Counter(len(r["calls"]) for r in merged)
    print(f"\n문항별 도구 호출 횟수 분포: "
          + ", ".join(f"{k}회={v}문항" for k, v in sorted(rounds.items())))


# --------------------------------------------------------------------------- G
ORG_EXAMPLES = [
    "국민성장펀드의 구조와 투자전략 동향",
    "국민성장펀드의 위험요인",
    "ETF 투자 시 유의해야 할 위험요인",
    "괴리율과 추적오차의 차이",
]


def section_g(settings) -> None:
    print(f"\n{BAR}")
    print("G. 외부 문서 저장소 (search_documents 의 대상)")
    print(BAR)
    ds = DocStore.load(settings.docs_dir)
    print(f"경로: {settings.docs_dir}")
    print(f"청크 수: {len(ds)}개 · 원본 파일 {len(set(c.doc_id for c in ds.chunks))}개")
    rows = []
    for i, c in enumerate(ds.chunks):
        rows.append([str(i), c.doc_id[:28], (c.title or "")[:26], (c.heading or "(머리말)")[:40],
                     str(len(c.text)), c.source[:26] or "-", c.as_of or "-"])
    print(table(rows, ["#", "파일", "문서 제목", "섹션(heading)", "글자수", "출처", "기준일"]))

    print(f"\n{SUB}\n주최측 예시 문항으로 검색했을 때 실제로 나오는 것\n{SUB}")
    for q in ORG_EXAMPLES:
        hits = ds.search(q, k=3)
        print(f"\n질의: {q}")
        if not hits:
            print("  (0건)")
            continue
        for c in hits:
            body = c.text.replace("\n", " ")[:180]
            print(f"  - [{c.score:7.2f}] {c.title} › {c.heading}  (출처 {c.source or '-'}, 기준 {c.as_of or '-'})")
            print(f"      {body}…")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="ABCDEG", help="실행할 섹션 문자 (기본 ABCDEG)")
    ap.add_argument("--n", type=int, default=40, help="상품 유형별·상품군별 질의 수 (기본 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--allow-slow-rebuild", action="store_true",
                    help="캐시 서명이 어긋날 때 메모리에서 벡터 색인을 다시 만든다(수 분). 캐시 파일은 어떤 경우에도 쓰지 않는다.")
    args = ap.parse_args()
    sec = set(args.sections.upper())

    root = pathlib.Path(__file__).resolve().parents[1]
    settings = get_settings()
    print(BAR)
    print("fin-agent 벡터 색인 감사 — LLM 미호출 · 스토어 읽기 전용 · 캐시 쓰기 없음")
    print(f"store={settings.store_path}  backend={settings.vector_backend}  dims={settings.vector_dims}  "
          f"max_df={settings.vector_max_df} (기본 {DEFAULT_MAX_DF})  seed={args.seed}  n={args.n}")
    print(BAR)

    need_index = bool(sec & set("ABCD"))
    rdb = index = vectors = None
    sig = ""
    cache_ok = False
    info: dict = {}
    t_lex = t_vec = 0.0
    if need_index:
        rdb, index, vectors, sig, cache_ok, info, t_lex, t_vec = load_indexes(
            settings, allow_slow_rebuild=args.allow_slow_rebuild)
        if vectors is None:
            print("\n⚠ 벡터 색인을 만들지 않았다(캐시 서명 불일치 + --allow-slow-rebuild 미지정). "
                  "B/C 는 어휘 단독 결과만 의미가 있다.\n")

    lexicon = Lexicon.load(settings.enrichment_dir)

    if "A" in sec:
        section_a(rdb, index, settings, t_lex, t_vec, vectors, args.seed)
    if ("B" in sec or "C" in sec) and vectors is not None:
        section_bc(index, vectors, rdb, lexicon, n=args.n, seed=args.seed, k=args.k)
    if "D" in sec:
        section_d(settings, sig, cache_ok, info, index, vectors)
    if "E" in sec:
        section_e(root)
    if "G" in sec:
        section_g(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
