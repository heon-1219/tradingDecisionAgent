"""Lexical (BM25) product index with Korean character bigrams and exact-title boosts.

No embedding model is required, so the service starts in seconds and behaves deterministically.
The tokenizer emits: lower-cased words, Korean character bigrams (so '로봇액티브' matches '로봇'),
and trigrams for long Latin tokens (so 'semiconductors' matches 'semiconductor').
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9&+\-\.]+|[가-힣]+|[一-鿿]+", re.I)


def tokenize(text: str | None) -> list[str]:
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    toks: list[str] = []
    for w in _WORD.findall(s):
        toks.append(w)
        if re.fullmatch(r"[가-힣]+", w):
            if len(w) >= 2:
                toks += [w[i : i + 2] for i in range(len(w) - 1)]
        elif re.fullmatch(r"[a-z0-9&+\-\.]+", w) and len(w) >= 6:
            toks += [w[i : i + 3] for i in range(len(w) - 2)]
    return toks


def norm_title(s: str | None) -> str:
    return re.sub(r"[\s\-_()\[\]·,./&+]+", "", unicodedata.normalize("NFKC", str(s or "")).lower())


_GENERIC = re.compile(r"(etf|etn|펀드|투자신탁|채권|상품|종목)+$")


def _strip_generic(norm_q: str) -> str:
    """'국민성장펀드' -> '국민성장', 'kodex로봇etf' -> 'kodex로봇' (generic product-type suffixes removed)."""
    return _GENERIC.sub("", norm_q)


@dataclass
class Doc:
    id: str
    table: str
    title: str
    text: str
    payload: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    table: str
    title: str
    score: float
    payload: dict


# view -> (id col, title col, text cols, payload cols)
INDEX_SPECS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "v_bond_kr": (
        "pd_no", "pd_nm",
        ["pd_nm", "pd_abrv_nm", "pd_eng_nm", "issuer_name", "bd_knd", "crd_grd", "std_pd_mcls_nm"],
        ["pd_no", "pd_nm", "pd_exg_mkt", "crd_grd", "is_sellable", "mat_date", "buy_yield", "after_tax_yield", "srfc_irt", "issuer_name", "bd_knd", "pd_risk_nm"],
    ),
    "v_etf_kr": (
        "pd_itm_no", "pd_abrv_nm",
        ["pd_abrv_nm", "pd_nm", "manager_name", "cu_base_index", "ref_base_index", "wu_inv_ast_type", "wu_inv_rgn", "pd_ticker"],
        ["pd_itm_no", "pd_ticker", "pd_abrv_nm", "pd_nm", "manager_name", "product_type", "is_on_sale", "wu_inv_ast_type", "wu_inv_rgn", "pd_net_tamt", "cu_charge_rt", "pd_risk_nm", "listing_date"],
    ),
    "v_etf_gl": (
        "pd_itm_no", "pd_nm",
        ["pd_nm", "ticker", "manager_name", "cu_base_index", "wu_inv_rgn", "wu_inv_ast_type", "cu_strtegy"],
        ["pd_itm_no", "ticker", "pd_nm", "manager_name", "product_type", "wu_inv_ast_type", "wu_inv_rgn", "expense_ratio_pct", "du_last_aum", "cu_base_index", "pd_exg_mkt_cd"],
    ),
    "v_fund_pub": (
        "itm_no", "itm_nm",
        ["itm_nm", "itm_abrv_nm", "manager_name", "or_attr_desc", "zrin_ptn_nm", "bmrk_nm", "fd_ivst_rgn_desc", "zrin_btyp_nm"],
        ["itm_no", "itm_nm", "fund_family", "manager_name", "product_type", "is_on_sale", "or_attr_desc", "zrin_ptn_nm", "bmrk_nm", "fd_nast_suma", "zrin_fd_ivst_risk_grd_nm", "fd_ivst_rgn_desc"],
    ),
}


class SearchIndex:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self._bm25 = BM25Okapi([tokenize(d.text) or ["_"] for d in docs]) if docs else None
        self._norm_titles = [norm_title(d.title) for d in docs]
        self._title_toks = [set(tokenize(d.title)) for d in docs]
        self._tables = np.array([d.table for d in docs]) if docs else np.array([])
        self._by_id: dict[tuple[str, str], Doc] = {(d.table, d.id): d for d in docs}

    @classmethod
    def from_rdb(cls, rdb) -> "SearchIndex":
        docs: list[Doc] = []
        for view, (idc, titlec, textc, payc) in INDEX_SPECS.items():
            cols = list(dict.fromkeys([idc, titlec, *textc, *payc]))
            for row in rdb.fetch_all(f"select {', '.join(cols)} from {view}"):
                r = dict(zip(cols, row))
                if r.get(idc) is None:
                    continue
                text = " ".join(str(r[c]) for c in textc if r.get(c) not in (None, ""))
                docs.append(Doc(str(r[idc]), view, str(r.get(titlec) or ""), text, {c: r.get(c) for c in payc}))
        return cls(docs)

    def get(self, table: str, id_: str) -> Doc | None:
        return self._by_id.get((table, id_))

    def search(self, query: str, k: int = 10, table: str | None = None) -> list[Hit]:
        if not self.docs or not query or not query.strip():
            return []
        qtoks = tokenize(query)
        scores = self._bm25.get_scores(qtoks).astype(float)
        # rank_bm25 can yield negative IDF-floored scores on small corpora; keep only docs sharing a token
        qset = set(qtoks)
        mask = np.fromiter((any(t in df for t in qset) for df in self._bm25.doc_freqs), dtype=bool, count=len(self.docs))
        floor = min(0.0, float(scores.min())) if len(scores) else 0.0
        scores = np.where(mask, scores - floor + 1e-3, 0.0)
        # title-token overlap boost: rewards docs whose *name* shares words with the query
        overlap = np.fromiter((len(qset & ts) for ts in self._title_toks), dtype=float, count=len(self.docs))
        scores = np.where(mask, scores + 4.0 * overlap, scores)
        q = norm_title(query)
        core = _strip_generic(q)
        if len(q) >= 2:
            for i, t in enumerate(self._norm_titles):
                if not t:
                    continue
                if t == q or (core and t == core):
                    scores[i] += 100.0
                elif q in t or t in q:
                    scores[i] += 10.0
                elif core and len(core) >= 2 and core in t:
                    scores[i] += 20.0
        if table is not None:
            scores = np.where(self._tables == table, scores, -1.0)
        k = max(1, min(k, len(self.docs)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(self.docs[i].id, self.docs[i].table, self.docs[i].title, float(scores[i]), self.docs[i].payload)
            for i in top
            if scores[i] > 0
        ]
