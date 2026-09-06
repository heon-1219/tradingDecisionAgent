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


# view -> (id col, title col, text cols, payload cols, row filter or None)
#
# "text cols" is the only thing BM25 actually tokenises, so a column left out of it cannot be matched
# by a word.  The 100%-filled classification columns used to sit in the payload only, which meant
# '변동금리 채권' or '액티브 ETF' — the vocabulary the source itself uses — matched nothing lexically
# even though every row carries the value.  They are indexed now, and the equivalent concept is
# indexed for every view that has it (전략·상품유형·통화·시장구분) rather than for one view only.
INDEX_SPECS: dict[str, tuple[str, str, list[str], list[str], str | None]] = {
    "v_bond_kr": (
        "pd_no", "pd_nm",
        ["pd_nm", "pd_abrv_nm", "pd_eng_nm", "issuer_name", "bd_knd", "crd_grd", "std_pd_mcls_nm",
         "std_pd_scls_nm", "bd_inrt_tcd", "bd_intp_tcd", "bd_ofr_tcd", "pd_exg_mkt", "curr_cd", "product_type"],
        ["pd_no", "pd_nm", "pd_exg_mkt", "crd_grd", "is_sellable", "mat_date", "buy_yield", "after_tax_yield", "srfc_irt", "issuer_name", "bd_knd", "pd_risk_nm"],
        None,
    ),
    "v_etf_kr": (
        "pd_itm_no", "pd_abrv_nm",
        ["pd_abrv_nm", "pd_nm", "manager_name", "cu_base_index", "ref_base_index", "wu_inv_ast_type", "wu_inv_rgn", "pd_ticker",
         "cu_strtegy", "product_type", "pd_dvid_cycl"],
        ["pd_itm_no", "pd_ticker", "pd_abrv_nm", "pd_nm", "manager_name", "product_type", "is_on_sale", "wu_inv_ast_type", "wu_inv_rgn", "pd_net_tamt", "cu_charge_rt", "pd_risk_nm", "listing_date"],
        None,
    ),
    "v_etf_gl": (
        "pd_itm_no", "pd_nm",
        ["pd_nm", "ticker", "manager_name", "cu_base_index", "wu_inv_rgn", "wu_inv_ast_type", "cu_strtegy",
         "product_type", "pd_exg_mkt_cd", "pd_curr_cd"],
        ["pd_itm_no", "ticker", "pd_nm", "manager_name", "product_type", "wu_inv_ast_type", "wu_inv_rgn", "expense_ratio_pct", "du_last_aum", "cu_base_index", "pd_exg_mkt_cd"],
        None,
    ),
    "v_fund_pub": (
        "itm_no", "itm_nm",
        ["itm_nm", "itm_abrv_nm", "manager_name", "or_attr_desc", "zrin_ptn_nm", "bmrk_nm", "fd_ivst_rgn_desc", "zrin_btyp_nm",
         "ovrs_fd_desc", "sale_yn", "product_type", "int_dvd_desc", "curr_cd"],
        ["itm_no", "itm_nm", "fund_family", "manager_name", "product_type", "is_on_sale", "or_attr_desc", "zrin_ptn_nm", "bmrk_nm", "fd_nast_suma", "zrin_fd_ivst_risk_grd_nm", "fd_ivst_rgn_desc"],
        None,
    ),
    # 편입종목.  '삼성전자를 보유한 ETF' 는 SQL 로 풀 수 있지만, 종목명이 어느 색인에도 없으면
    # 그 이름을 검색으로 확인할 방법이 없어 "제공된 데이터로 확인 불가" 로 답하게 된다.
    # is_cash=1 행(885건)은 색인에서 뺀다: 원화현금·설정현금액 같은 현금성 항목이라 기업이 아니고,
    # company_code 가 전부 NULL 이며, 제목이 '현금' 계열 단어라 종목 검색 결과를 밀어낸다.
    # 행 자체는 스토어와 SQL 에 그대로 남아 있으므로 비중 합계는 달라지지 않는다.
    #
    # etf_name 은 payload 에만 두고 text 에서는 뺀다.  편입종목 문서는 21,714건이 각각 소속 ETF 의
    # 이름을 통째로 지니는데, BM25 는 짧은 문서를 선호하므로 ETF 이름의 단어로 검색하면 마스터의
    # ETF 본체보다 그 ETF 를 담은 편입종목 행이 먼저 올라온다.  '액티브' 검색이 상위 20건 중 19건을
    # 편입종목으로 채워 '1Q 200액티브' 같은 실제 상품을 밀어냈다.  text 에서 빼면 0/20 이 되고
    # 상품이 1위로 돌아오며, payload 에 남아 있으므로 결과에는 어느 ETF 의 보유 종목인지 그대로 보인다.
    "v_etf_constituent": (
        "etf_itm_no", "company_name",
        ["company_name", "company_code"],
        ["company_name", "company_code", "etf_itm_no", "etf_name", "weight_pct", "is_cash"],
        "is_cash = 0",
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
        for view, (idc, titlec, textc, payc, where) in INDEX_SPECS.items():
            cols = list(dict.fromkeys([idc, titlec, *textc, *payc]))
            sql = f"select {', '.join(cols)} from {view}" + (f" where {where}" if where else "")
            for row in rdb.fetch_all(sql):
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
