#!/usr/bin/env python
"""Vector (semantic) index audit — answers "what does the vector leg actually do?" with numbers.

Built to run deterministically (fixed seed) without a single LLM call, so that it can serve as the
"evaluation reproduction procedure" in the technical proposal's appendix.  The store is opened read-only,
vector_cache.npz is read only when its signature matches, and **it is never written, under any circumstances**.

What it measures
  A. Index composition  — documents per source, sample document texts, length distribution
  B. Retrieval quality  — recall@1/5/10 · MRR on auto-generated queries whose answer is self-evident
  C. BM25 vs vector     — lexical only / vector only / the actual combination (hybrid_search), cases the vector rescues
  D. Cache integrity    — whether the signature computed from the current document set matches the cached one
  E. Tool-call counts   — measured calls per tool, from the think_trace in data/eval/*.jsonl
  G. External documents — the chunks search_documents searches, and whether the organizers' example questions can be answered

Usage
  python scripts/audit_vector.py                    # all sections (~15 min, mostly the 1,000-odd queries of B/C)
  python scripts/audit_vector.py --sections ADEG    # index composition, cache, tool calls, documents (~1 min)
  python scripts/audit_vector.py --sections BC      # retrieval quality only (~14 min)
  python scripts/audit_vector.py --n 40 --seed 0    # queries per product family / seed (default 40, seed 0)

Measured baselines (2026-09-06, 75,089 documents): lexical index build 9~18 s, TF-IDF fit 9.4 s,
cache load 1.0 s, resident vector memory 174MB (matrix 115MB + components 59MB).
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
    "v_bond_kr": "domestic bonds",
    "v_etf_kr": "domestic ETF/ETN",
    "v_etf_gl": "overseas ETF",
    "v_fund_pub": "public/private funds",
    "v_etf_constituent": "domestic ETF constituents",
}
# Is the surface form used for search (code/ticker) in that view's indexed text? — checked by reading INDEX_SPECS.
CODE_COL = {
    "v_bond_kr": "pd_no",
    "v_etf_kr": "pd_ticker",
    "v_etf_gl": "ticker",
    "v_fund_pub": "itm_no",
}


# --------------------------------------------------------------------------- common
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
    """Signature computed **exactly** as vector.VectorIndex.build does (hashes the document texts too)."""
    sep = bytes([31])
    h = hashlib.sha1(f"{len(docs)}|{dims}|{max_df}".encode("utf-8"))
    for d in docs:
        h.update(f"{d.title} {d.text}".encode("utf-8", "ignore"))
        h.update(sep)
    return h.hexdigest()[:16]


def load_indexes(settings, *, allow_slow_rebuild: bool):
    """Lexical index + vector index.  cache_path is passed only when the cache signature matches, **ruling out writes at the source**."""
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
        # signature mismatch → rebuild in memory only, without cache_path, so the cache is never updated.
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
    print("A. What is in the index")
    print(BAR)
    by_table = collections.Counter(d.table for d in index.docs)
    meta = rdb.meta()
    rows = []
    for view in [*PRODUCT_VIEWS, "v_etf_constituent"]:
        raw = rdb.scalar(f"SELECT COUNT(*) FROM {view}")
        where = INDEX_SPECS[view][4]
        note = f"filter {where}" if where else "-"
        rows.append([view, VIEW_LABEL[view], f"{raw:,}", f"{by_table.get(view, 0):,}",
                     str(meta.get(view, {}).get("base_date") or "-"), note])
    print(table(rows, ["view", "family", "source rows", "indexed docs", "base date", "row filter"]))
    print(f"\ntotal: {sum(by_table.values()):,} indexed docs / "
          f"{sum(rdb.scalar(f'SELECT COUNT(*) FROM {v}') for v in by_table):,} source rows")
    print(f"lexical index build {t_lex:.1f}s · vector index load {t_vec:.1f}s"
          + (f" (backend={vectors.stats.backend}, dims={vectors.stats.dims}, cached={vectors.stats.cached})"
             if vectors is not None else " (no vectors)"))

    print(f"\n{SUB}\nsample document texts (3 per source, the exact strings that go into the embedding)\n{SUB}")
    rng = random.Random(seed)
    for view in [*PRODUCT_VIEWS, "v_etf_constituent"]:
        pool = [d for d in index.docs if d.table == view]
        print(f"\n[{view}] {VIEW_LABEL[view]} — text columns: {', '.join(INDEX_SPECS[view][2])}")
        for d in rng.sample(pool, min(3, len(pool))):
            body = f"{d.title} {d.text}"
            print(f"  id={d.id} | len={len(body)} chars")
            print(f"    {body[:300]}{'…' if len(body) > 300 else ''}")

    print(f"\n{SUB}\ndocument length distribution (title + text, in characters)\n{SUB}")
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
    rows.append(["(all)", f"{n:,}", str(min(lens_all)), str(lens_all[n // 10]),
                 str(int(statistics.median(lens_all))), str(lens_all[int(n * 0.9)]), str(max(lens_all)),
                 f"{short_total:,} ({100*short_total/n:.1f}%)"])
    print(table(rows, ["view", "docs", "min", "p10", "median", "p90", "max", "<=20 chars"]))

    dup = collections.Counter(f"{d.title} {d.text}" for d in index.docs)
    ndup = sum(c - 1 for c in dup.values() if c > 1)
    print(f"\nduplicate documents with exactly identical text: {ndup:,} "
          f"(unique texts {len(dup):,} / total {len(index.docs):,})")
    top = [(t, c) for t, c in dup.most_common(5)]
    for t, c in top:
        print(f"  x{c:4d}  {t[:90]}")


# --------------------------------------------------------------------------- B/C query generation
_SPLIT = re.compile(r"[\s()\[\]·,/]+")


def abbrev_of(title: str) -> str | None:
    """'TIGER 미국나스닥100' -> '미국나스닥100' (brand token dropped); with no spaces, cut out a middle span."""
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
    """Delete one character or swap two adjacent ones — the typical case where lexical search breaks."""
    s = str(s or "").strip()
    if len(s) < 6:
        return None
    i = rng.randrange(2, len(s) - 2)
    if rng.random() < 0.5:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_queries(rdb, index, lexicon: Lexicon, n: int, seed: int) -> dict[str, list[dict]]:
    """Auto-generate queries with a self-evident answer from the data.  Each item: {query, kind, view, gold(fn), label}"""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = collections.defaultdict(list)

    # ---- the three product query types: full name / code/ticker / partial name (abbreviation)
    for view in PRODUCT_VIEWS:
        idc, titlec = INDEX_SPECS[view][0], INDEX_SPECS[view][1]
        code = CODE_COL[view]
        cols = list(dict.fromkeys([idc, titlec, code]))
        rows = [dict(zip(cols, r)) for r in
                rdb.fetch_all(f"SELECT {', '.join(cols)} FROM {view} WHERE {titlec} IS NOT NULL AND {titlec} <> ''")]
        # When one name spans several products there is no 'single right answer', so only unique titles are used.
        by_title = collections.Counter(norm(r[titlec]) for r in rows)
        uniq = [r for r in rows if by_title[norm(r[titlec])] == 1]
        rng.shuffle(uniq)

        def gold_for(gid: str, v=view):
            return lambda h: h.table == v and str(h.id) == str(gid)

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            out["full name"].append({"query": str(r[titlec]), "kind": "full name", "view": view,
                                    "gold": gold_for(r[idc]), "label": str(r[titlec])})
            picked += 1

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            v = r.get(code)
            if v in (None, "") or len(str(v)) < 3:
                continue
            out["code/ticker"].append({"query": str(v), "kind": "code/ticker", "view": view,
                                     "gold": gold_for(r[idc]), "label": f"{v} → {r[titlec]}"})
            picked += 1

        picked = 0
        for r in uniq:
            if picked >= n:
                break
            ab = abbrev_of(str(r[titlec]))
            if not ab:
                continue
            out["partial name (abbrev.)"].append({"query": ab, "kind": "partial name (abbrev.)", "view": view,
                                        "gold": gold_for(r[idc]), "label": f"{ab} → {r[titlec]}"})
            picked += 1

    # ---- constituent name → the constituent documents that hold that stock
    comps = [c for (c,) in rdb.fetch_all(
        "SELECT DISTINCT company_name FROM v_etf_constituent WHERE is_cash=0 AND company_name IS NOT NULL")]
    rng.shuffle(comps)
    for c in comps[: n * 2]:
        if len(out["constituent name"]) >= n * 2:
            break
        nm = norm(c)

        def gold_c(h, nm=nm):
            return h.table == "v_etf_constituent" and norm(h.payload.get("company_name")) == nm

        out["constituent name"].append({"query": str(c), "kind": "constituent name", "view": "v_etf_constituent",
                                  "gold": gold_c, "label": str(c)})

    # ---- theme term → products that show the theme in their name
    # Gold definition: products whose (normalized) title contains one of the theme's aliases.  ASCII aliases match on word boundaries.
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
            continue  # a theme with practically no product recognisable by name has no workable gold definition

        def gold_t(h, keys=frozenset(gold_keys)):
            return (h.table, str(h.id)) in keys

        for a in aliases:
            out["theme term"].append({"query": a, "kind": "theme term", "view": "*", "gold": gold_t,
                                  "label": f"{a} (theme {th.id}, gold {len(gold_keys)})",
                                  "theme": th.id, "alias_direct": len(hit_sets[a])})
    rng.shuffle(out["theme term"])
    # One query per alias would mean hundreds, making the reproduction run far too long. Keep at least 40 per type.
    out["theme term"] = out["theme term"][: max(n, 40) * 2]

    # ---- typos (variants of full names) — where lexical search fails
    for q in list(out["full name"]):
        t = typo_of(q["query"], rng)
        if t:
            out["typo"].append({**q, "query": t, "kind": "typo", "label": f"{t}  (original: {q['query']})"})

    # ---- synonym gap (theme aliases whose spelling never appears in the data) — where only semantic search can help
    for q in out["theme term"]:
        if q.get("alias_direct", 1) == 0:
            out["synonym gap"].append({**q, "kind": "synonym gap"})

    return out


# --------------------------------------------------------------------------- B/C run
def run_modes(index, vectors, query: str, k: int = 10):
    lex = index.search(query, k=max(k, 20))
    vec = vectors.search(query, k=max(k, 20)) if vectors is not None else []
    hyb = hybrid_search(lex, vec, k=k)  # same call as tools._t_search_products
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
    print(table(rows, ["mode", "queries", "recall@1", "recall@5", "recall@10", "MRR"]))


def section_bc(index, vectors, rdb, lexicon, n: int, seed: int, k: int = 10) -> None:
    print(f"\n{BAR}")
    print("B/C. Retrieval quality — lexical (BM25) vs vector (TF-IDF+LSA) vs the actual combination (hybrid_search)")
    print(BAR)
    qs = make_queries(rdb, index, lexicon, n=n, seed=seed)
    print("generated queries (gold answers derived automatically from the data):")
    print(table([[kind, f"{len(v):,}"] for kind, v in qs.items()], ["type", "queries"]))

    all_detail: list[dict] = []
    summary_rows = []
    for kind in ["full name", "code/ticker", "partial name (abbrev.)", "constituent name", "theme term", "typo", "synonym gap"]:
        items = qs.get(kind) or []
        if not items:
            continue
        t0 = time.time()
        per_mode, detail = eval_set(index, vectors, items, k=k)
        all_detail += detail
        print(f"\n{SUB}")
        print(f"[{kind}]  queries {len(items)} · {time.time()-t0:.1f}s")
        print(SUB)
        print_scores("", per_mode)
        for m in ("lexical", "vector", "hybrid"):
            s = per_mode[m]
            summary_rows.append([kind, m, str(s["n"]), pct(s["r1"]), pct(s["r5"]), pct(s["r10"]), f"{s['mrr']:.3f}"])

        # breakdown by product family (the three product query types only)
        if kind in ("full name", "code/ticker", "partial name (abbrev.)"):
            rows = []
            for view in PRODUCT_VIEWS:
                sub = [d for d in detail if d["view"] == view]
                if not sub:
                    continue
                for m, key in (("lexical", "lex"), ("vector", "vec"), ("hybrid", "hyb")):
                    s = score([d[key] for d in sub])
                    rows.append([view, m, str(s["n"]), pct(s["r1"]), pct(s["r10"]), f"{s['mrr']:.3f}"])
            print("\nbreakdown by product family")
            print(table(rows, ["view", "mode", "queries", "recall@1", "recall@10", "MRR"]))

    print(f"\n{BAR}")
    print("Overall comparison (all types)")
    print(BAR)
    print(table(summary_rows, ["type", "mode", "queries", "recall@1", "recall@5", "recall@10", "MRR"]))

    # ---- what the vector actually added
    print(f"\n{BAR}")
    print("C-core. What the vector added on top of BM25 / what it took away")
    print(BAR)
    rescue = [d for d in all_detail if d["lex"] is None and d["vec"] is not None]
    hyb_gain = [d for d in all_detail if d["lex"] is None and d["hyb"] is not None]
    hyb_loss = [d for d in all_detail if d["lex"] is not None and d["hyb"] is None]
    rank_up = [d for d in all_detail if d["lex"] and d["hyb"] and d["hyb"] < d["lex"]]
    rank_down = [d for d in all_detail if d["lex"] and d["hyb"] and d["hyb"] > d["lex"]]
    print(table([
        ["all queries", f"{len(all_detail):,}"],
        ["lexical miss & vector hit (top-10)", f"{len(rescue):,}"],
        ["lexical miss & hybrid hit = actual rescue", f"{len(hyb_gain):,}"],
        ["lexical hit & hybrid miss = lost by the combination", f"{len(hyb_loss):,}"],
        ["queries the combination ranked higher", f"{len(rank_up):,}"],
        ["queries the combination ranked lower", f"{len(rank_down):,}"],
    ], ["item", "count"]))
    if hyb_gain:
        print("\nactual rescues (lexical top-10 miss → hybrid top-10 hit):")
        for d in hyb_gain[:20]:
            print(f"  [{d['kind']}] {d['label'][:70]}  | vector rank={d['vec']} hybrid rank={d['hyb']} "
                  f"| lexical #1='{str(d['lex_top'])[:34]}'")
    else:
        print("\nactual rescues: 0 — the combined result never recovered a single query that lexical search missed.")
    if hyb_loss:
        print("\ncases lost by the combination (lexical top-10 hit → hybrid top-10 miss):")
        for d in hyb_loss[:20]:
            print(f"  [{d['kind']}] {d['label'][:70]}  | lexical rank={d['lex']}")

    # ---- reproduce verbatim the individual probes the README/proposal cite as evidence
    print(f"\n{BAR}")
    print("C-2. Reproducing the probes cited in the docs (examples from the README's 'vector quality tuning' passage)")
    print(BAR)
    for q in ["space exploration satellite", "China semiconductor ETF", "차이나 반도체",
              "배터리 양극재 관련 상품", "우주항공", "월배당 커버드콜", "야놀자를 보유한 국내 ETF"]:
        lex, vec, hyb = run_modes(index, vectors, q, k=5)
        print(f"\nquery: {q}")
        print("  lexical:", " | ".join(f"{h.title[:34]}" for h in lex[:5]) or "(0 hits)")
        print("  vector :", " | ".join(f"{h.title[:34]}" for h in vec[:5]) or "(0 hits)")
        print("  hybrid :", " | ".join(f"{h.title[:34]}" for h in hyb[:5]) or "(0 hits)")


# --------------------------------------------------------------------------- D
def section_d(settings, sig: str, cache_ok: bool, info: dict, index, vectors) -> None:
    print(f"\n{BAR}")
    print("D. Cache integrity")
    print(BAR)
    rows = [
        ["cache path", info.get("path", "-")],
        ["exists", str(info.get("exists"))],
        ["size", f"{info.get('size_mb', '-')} MB"],
        ["stored keys", ", ".join(info.get("keys", []) or [])],
        ["signature computed from the current doc set", sig],
        ["signature stored in the cache", str(info.get("cache_signature", "-"))],
        ["signature match", str(info.get("signature_match"))],
        ["dims match", f"{info.get('cache_dims')} vs {settings.vector_dims} → {info.get('dims_match')}"],
        ["row count match", f"{info.get('matrix_shape', ('-',))[0]} vs {len(index.docs)} → {info.get('rows_match')}"],
        ["matrix / components shape", f"{info.get('matrix_shape')} / {info.get('components_shape')}"],
        ["verdict", "valid (usable as is)" if cache_ok else "stale (rebuild needed)"],
    ]
    print(table(rows, ["item", "value"]))

    print("\nvalidation path check (the conditions under which vector.VectorIndex.build accepts the cache):")
    for line in [
        "  1) signature == computed signature   — SHA1 of doc count + dims + max_df + **every document text**",
        "  2) dims match",
        "  3) components.shape[1] == current TF-IDF vocabulary width  (safeguard against a matmul dimension mismatch)",
        "  4) matrix.shape[0] == doc count",
        "  If any of the four fails, it logs a warning and rebuilds. No path uses the cache while bypassing these checks",
        "  (this is the only place in kb/vector.py that calls np.load).",
    ]:
        print(line)

    if vectors is not None:
        # Is the cache actually live: check that an arbitrary query comes back without exceptions and with the right dimensions
        probe = vectors.search("반도체 ETF", k=5)
        print(f"\nlive check: vectors.search('반도체 ETF') → {len(probe)} hits, "
              f"backend={vectors.stats.backend}, dims={vectors.stats.dims}, cached={vectors.stats.cached}")
        m = vectors.matrix
        nrm = np.linalg.norm(m, axis=1)
        zero = int((nrm < 1e-6).sum())
        print(f"matrix: {m.shape} {m.dtype} · median L2 norm {float(np.median(nrm)):.3f} · zero-vector docs {zero:,} "
              f"({100.0*zero/max(1,m.shape[0]):.2f}%)")
        if zero:
            print("  ⚠ zero-vector documents can never be found by vector search, whatever the query (documents whose tokens were all cut by max_df).")


# --------------------------------------------------------------------------- E
# These match the think_trace round lines recorded in data/eval/*.jsonl: the Korean "라운드 N: tool(...)"
# form from before the agent package was translated (Task 0.1), and the "Round N: tool(...)" form the
# code emits now.
_ROUND = re.compile(r"(?:라운드|Round)\s*\d+:\s*([a-z_]+)\s*\(")
_NOTOOL = re.compile(r"(?:라운드|Round)\s*\d+:\s*(?:도구 호출 없이|text reply without a tool call)")
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
    print("E. Do the tools actually get called? — recounted from recorded runs")
    print(BAR)
    truth = parse_traces(root / "data" / "eval" / "truth_results.jsonl")
    fail = parse_traces(root / "data" / "eval" / "fail_results.jsonl")
    fail_ids = {r["qid"] for r in fail}
    # fail_results re-ran some of the truth_results questions more recently → the latest run wins
    merged = [r for r in truth if r["qid"] not in fail_ids] + fail

    for name, recs in (("truth_results.jsonl (54 questions)", truth), ("fail_results.jsonl (re-run)", fail),
                       ("merged = latest run per question", merged)):
        c = collections.Counter(t for r in recs for t in r["calls"])
        rows = [[t, f"{c.get(t, 0):,}", f"{sum(1 for r in recs if t in r['calls']):,}"] for t in KNOWN_TOOLS]
        other = [t for t in c if t not in KNOWN_TOOLS]
        for t in other:
            rows.append([f"{t} (unregistered)", f"{c[t]:,}", "-"])
        print(f"\n{SUB}\n{name} — questions {len(recs)}, tool calls {sum(c.values())}, "
              f"'text reply without a tool call' rounds {sum(r['no_tool_rounds'] for r in recs)}\n{SUB}")
        print(table(rows, ["tool", "calls", "questions using this tool"]))

    print(f"\n{SUB}\ncall details for the vector-family tools (search_products / search_documents / resolve_entity)\n{SUB}")
    hits = [r for r in merged if any(t in r["calls"] for t in ("search_products", "search_documents", "resolve_entity"))]
    if not hits:
        print("0 calls — never called once across all 54 questions.")
    for r in hits:
        print(f"\n  {r['qid']} · ok={r['ok']} · {r['question'][:70]}")
        for line in r["trace"].splitlines():
            if any(t in line for t in ("search_products", "search_documents", "resolve_entity")):
                print(f"    {line.strip()[:150]}")
        print(f"    final answer: {r['answer'][:150]}")
        print(f"    does the evidence (retrieved_context) include search evidence? "
              f"{'yes' if ('search' in r['context'] or '문서' in r['context'] or 'document' in r['context'].lower()) else 'no'}")

    # For reference: how many rounds each question used (did it even get a chance to call a tool?)
    rounds = collections.Counter(len(r["calls"]) for r in merged)
    print(f"\ndistribution of tool calls per question: "
          + ", ".join(f"{k} calls={v} questions" for k, v in sorted(rounds.items())))


# --------------------------------------------------------------------------- G
ORG_EXAMPLES = [
    "국민성장펀드의 구조와 투자전략 동향",
    "국민성장펀드의 위험요인",
    "ETF 투자 시 유의해야 할 위험요인",
    "괴리율과 추적오차의 차이",
]


def section_g(settings) -> None:
    print(f"\n{BAR}")
    print("G. External document store (what search_documents searches)")
    print(BAR)
    ds = DocStore.load(settings.docs_dir)
    print(f"path: {settings.docs_dir}")
    print(f"chunks: {len(ds)} · source files {len(set(c.doc_id for c in ds.chunks))}")
    rows = []
    for i, c in enumerate(ds.chunks):
        rows.append([str(i), c.doc_id[:28], (c.title or "")[:26], (c.heading or "(preamble)")[:40],
                     str(len(c.text)), c.source[:26] or "-", c.as_of or "-"])
    print(table(rows, ["#", "file", "doc title", "section (heading)", "chars", "source", "as of"]))

    print(f"\n{SUB}\nwhat actually comes back when searching with the organizers' example questions\n{SUB}")
    for q in ORG_EXAMPLES:
        hits = ds.search(q, k=3)
        print(f"\nquery: {q}")
        if not hits:
            print("  (0 hits)")
            continue
        for c in hits:
            body = c.text.replace("\n", " ")[:180]
            print(f"  - [{c.score:7.2f}] {c.title} › {c.heading}  (source {c.source or '-'}, as of {c.as_of or '-'})")
            print(f"      {body}…")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="ABCDEG", help="letters of the sections to run (default ABCDEG)")
    ap.add_argument("--n", type=int, default=40, help="queries per product query type and per product family (default 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--allow-slow-rebuild", action="store_true",
                    help="if the cache signature does not match, rebuild the vector index in memory (takes minutes). The cache file is never written, under any circumstances.")
    args = ap.parse_args()
    sec = set(args.sections.upper())

    root = pathlib.Path(__file__).resolve().parents[1]
    settings = get_settings()
    print(BAR)
    print("fin-agent vector index audit — no LLM calls · store read-only · no cache writes")
    print(f"store={settings.store_path}  backend={settings.vector_backend}  dims={settings.vector_dims}  "
          f"max_df={settings.vector_max_df} (default {DEFAULT_MAX_DF})  seed={args.seed}  n={args.n}")
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
            print("\n⚠ The vector index was not built (cache signature mismatch and no --allow-slow-rebuild). "
                  "Only the lexical-only results of B/C are meaningful.\n")

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
