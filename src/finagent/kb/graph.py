"""Knowledge graph (ABox) built at startup from the SQLite store + curated lexicons.

TBox (classes/properties) lives in ontology/*.ttl; this module instantiates products, managers,
issuers, indices, themes, companies and fund families as RDF and exposes SPARQL + neighbor lookups.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import rdflib
from rdflib import RDF, RDFS, Literal, Namespace, URIRef, XSD

FP = Namespace("http://mafest.ai/product#")
D = Namespace("http://mafest.ai/data/")

PREFIXES = (
    "PREFIX fp: <http://mafest.ai/product#>\n"
    "PREFIX d: <http://mafest.ai/data/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)

_CASH_LIKE = {"원화현금", "설정현금액", "현금", "usd", "달러현금", "외화현금", "원화예금"}
_MISSING_INDEX = {"", "index is not provided by management company", "index is not available on lipper database", "none", "nan"}

ASSET_CLASS_CODES = {
    "주식": "Equity", "equity": "Equity", "채권": "Bond", "bond": "Bond", "원자재": "Commodity", "commodity": "Commodity",
    "대체투자": "Alternatives", "alternatives": "Alternatives", "혼합자산": "MixedAssets", "mixed assets": "MixedAssets",
    "단기자금": "MoneyMarket", "money market": "MoneyMarket", "통화": "Currency", "부동산": "RealEstate", "기타": "Other", "other": "Other",
}


def slug(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).strip()
    s = re.sub(r"[^0-9A-Za-z가-힣一-鿿]+", "_", s).strip("_")
    return s[:80] or "unknown"


# --------------------------------------------------------------- group affiliates
# 그룹주 ETF 의 편입종목에서 '같은 기업집단'(계열) 관계를 유도한다.
#
# 왜 데이터로 유도하는가 — companies.yaml 의 손으로 적은 parent 4개(에코프로 3 + SK이노베이션 1)는
#   주최측 예시 문항에 나온 회사만 맞추는 오버피팅이었다. 다른 그룹을 물으면 답변불가가 됐다.
#   운용사가 '이 종목들이 이 그룹이다'라고 상품에 담아둔 사실이 v_etf_constituent 안에 이미 있다.
#
# 왜 '계열사'이지 '자회사'가 아닌가 — 원천에 지분율이 없다. 같은 그룹주 ETF 에 담겼다는 사실은
#   같은 기업집단임을 말할 뿐, 누가 누구를 소유하는지는 말하지 않는다. ontology/common.ttl 의
#   fp:affiliateOf 주석 참조.
#
# 규칙(실측으로 정했다. 17개 그룹주 ETF 전수 확인):
#   1) 편입종목이 기업이 아닌 ETF 는 버린다 — 채권·혼합형은 편입종목이 채권 종목("현대카드914-4",
#      "국고채권03250-2706")이거나 선물("삼성전자선물2609")이다. 이것을 회사로 채택하면 곧 오답이다.
#   2) 그룹명 토큰에 숫자가 들어가면 버린다 — "5대그룹주"(복수 그룹 혼합), "인도시장대표BIG5그룹"
#      (지수명)은 하나의 그룹에 귀속시킬 수 없다.
#   3) 순수형(그룹명 뒤에 가중방식 수식어만 붙는 것: '', '주', '동일가중', '섹터가중', '밸류')은
#      편입종목 전부를 계열사로 채택한다. 이것이 이 방식의 이득이다 — 에스원·제일기획·호텔신라는
#      실제 삼성 계열사인데 이름 접두어로는 절대 잡을 수 없고 ETF 는 잡는다.
#   4) 혼합형(플러스·포커스·대표주 등 '관련주'를 섞는 상품)은 이름에 그룹명 토큰을 포함하는
#      종목만 채택한다. TIGER LG그룹플러스에서 GS·GS건설·GS리테일·LS·휴젤을 떨어뜨리고,
#      WON 두산그룹포커스에서 한전기술·삼성전자·파미셀을 떨어뜨린다.
#   5) 같은 그룹의 ETF 가 여럿이면 합집합.
_GROUP_RX = re.compile(r"([0-9A-Za-z가-힣&.]+)그룹")
# 편입종목이 기업이 아닌 상품(채권 종목·국고채·선물). 회사로 채택하면 안 된다.
_GROUP_NOT_EQUITY = re.compile(r"채권|혼합|국고|통안")
# '그룹' 뒤에 와도 편입 유니버스를 바꾸지 않는 수식어(가중방식·표기 변형)
_GROUP_PURE_SUFFIX = {"", "주", "동일가중", "섹터가중", "밸류", "sw", "주sw", "주동일가중", "주섹터가중"}
# 그룹명 앞에 붙은 투자지역 수식어("인도타타그룹" → 타타그룹)
_GROUP_REGION_PREFIX = ("인도", "미국", "중국", "일본", "국내", "해외", "유럽")
# 같은 회사를 한글/영문 두 표기로 쓰는 그룹 (POSCO홀딩스는 '포스코' 부분문자열로 잡히지 않는다)
_GROUP_TOKEN_ALIASES = {"포스코": ("posco",)}


def parse_group_etf(etf_name: str) -> tuple[str, bool] | None:
    """'…그룹…' ETF 이름 → (그룹명 토큰, 순수형 여부). 계열 유도에 쓸 수 없으면 None.

    >>> parse_group_etf("PLUS 한화그룹주")
    ('한화', True)
    >>> parse_group_etf("TIGER LG그룹플러스")
    ('LG', False)
    >>> parse_group_etf("RISE 삼성그룹Top3채권혼합") is None
    True
    """
    name = unicodedata.normalize("NFKC", str(etf_name or "")).strip()
    if "그룹" not in name or _GROUP_NOT_EQUITY.search(name):
        return None
    m = None
    for m in _GROUP_RX.finditer(name):  # 마지막 '…그룹' 을 그룹명으로 본다
        pass
    if m is None:
        return None
    token, suffix = m.group(1), name[m.end():].strip()
    if re.search(r"\d", token):  # '5대', '인도시장대표BIG5' — 하나의 그룹이 아니다
        return None
    for p in _GROUP_REGION_PREFIX:
        if token.startswith(p) and len(token) - len(p) >= 2:
            token = token[len(p):]
            break
    if len(token) < 2:
        return None
    return token, suffix.lower().replace(" ", "") in _GROUP_PURE_SUFFIX


def _group_token_matches(token: str, company_name: str) -> bool:
    cn = unicodedata.normalize("NFKC", str(company_name or "")).lower()
    return any(t.lower() in cn for t in (token, *_GROUP_TOKEN_ALIASES.get(token, ())))


def derive_group_affiliates(rows) -> dict[str, dict[str, list[str]]]:
    """편입종목 행에서 기업집단을 유도한다.

    ``rows``: (etf_itm_no, etf_name, company_key, company_name) 반복자. ``company_key`` 는
    ``_load_constituents`` 가 회사 IRI 를 만들 때 쓰는 것과 **같은** 식별키여야 한다 — 다른 키를
    쓰면 같은 회사가 그래프에 두 번 생긴다.

    반환: {그룹명 토큰: {company_key: [근거 ETF itm_no, ...]}}
    """
    groups: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for etf_id, etf_name, key, cname in rows:
        parsed = parse_group_etf(etf_name)
        if parsed is None:
            continue
        token, pure = parsed
        if not pure and not _group_token_matches(token, cname):
            continue
        ev = groups[token].setdefault(key, [])
        if etf_id not in ev:
            ev.append(etf_id)
    return dict(groups)


class KG:
    def __init__(self, g: rdflib.Graph):
        self.graph = g
        self.asserted_triples = len(g)
        self.inferred_triples = 0
        self._label_index: dict[str, list[URIRef]] = defaultdict(list)
        self._reindex()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, rdb, lexicon, ontology=None, log=print) -> "KG":
        t0 = time.time()
        g = rdflib.Graph()
        g.bind("fp", FP)
        g.bind("d", D)
        add = g.add
        lit = Literal

        def theme_iri(tid: str) -> URIRef:
            return D[f"theme/{tid}"]

        # --- themes ---------------------------------------------------------
        for t in lexicon.themes:
            ti = theme_iri(t.id)
            add((ti, RDF.type, FP.Theme))
            add((ti, RDFS.label, lit(t.label_ko, lang="ko")))
            add((ti, RDFS.label, lit(t.label_en, lang="en")))
            add((ti, FP.name, lit(t.label_ko)))
            for r in t.related:
                add((ti, FP.relatedTheme, theme_iri(r)))
        # --- companies ------------------------------------------------------
        comp_iri: dict[str, URIRef] = {}
        for c in lexicon.companies:
            ci = D[f"company/{c.id}"]
            comp_iri[c.id] = ci
            add((ci, RDF.type, FP.Company))
            add((ci, FP.name, lit(c.name_ko)))
            add((ci, RDFS.label, lit(c.name_ko, lang="ko")))
            add((ci, RDFS.label, lit(c.name_en, lang="en")))
            add((ci, FP.englishName, lit(c.name_en)))
            for tk in c.tickers:
                add((ci, FP.ticker, lit(tk)))
            if c.country:
                add((ci, FP.country, lit(c.country)))
            if c.not_listed:
                add((ci, FP.isListed, lit(False)))
        for c in lexicon.companies:
            if c.parent and c.parent in comp_iri:
                add((comp_iri[c.id], FP.subsidiaryOf, comp_iri[c.parent]))
                add((comp_iri[c.parent], FP.hasSubsidiary, comp_iri[c.id]))
        # --- managers -------------------------------------------------------
        mgr_iri: dict[str, URIRef] = {}

        def manager(name: str | None) -> URIRef | None:
            if not name:
                return None
            canon = lexicon.manager_canonical(name) or name
            if canon in mgr_iri:
                return mgr_iri[canon]
            mi = D[f"manager/{slug(canon)}"]
            mgr_iri[canon] = mi
            add((mi, RDF.type, FP.ManagementCompany))
            add((mi, FP.name, lit(canon)))
            add((mi, RDFS.label, lit(canon, lang="ko")))
            m = next((m for m in lexicon.managers if m.canonical == canon), None)
            if m and m.group:
                add((mi, FP.group, lit(m.group)))
            return mi

        idx_iri: dict[str, URIRef] = {}

        def index_node(name: str | None) -> URIRef | None:
            if not name or str(name).strip().lower() in _MISSING_INDEX:
                return None
            key = str(name).strip()
            if key in idx_iri:
                return idx_iri[key]
            ii = D[f"index/{slug(key)}"]
            idx_iri[key] = ii
            add((ii, RDF.type, FP.Index))
            add((ii, FP.name, lit(key)))
            add((ii, RDFS.label, lit(key)))
            return ii

        def themes_for(*texts) -> list[URIRef]:
            blob = " ".join(str(t) for t in texts if t)
            return [theme_iri(t.id) for t in lexicon.match_themes(blob, min_len=2)]

        def common(pi: URIRef, cls: URIRef, code, name, short, ptype, on_sale, base):
            add((pi, RDF.type, cls))
            add((pi, FP.productCode, lit(str(code))))
            if name:
                add((pi, FP.name, lit(str(name))))
                add((pi, RDFS.label, lit(str(name), lang="ko")))
            if short and short != name:
                add((pi, FP.shortName, lit(str(short))))
            if ptype:
                add((pi, FP.productType, lit(str(ptype))))
            if on_sale is not None:
                add((pi, FP.isOnSale, lit(bool(on_sale))))
            if base:
                add((pi, FP.baseDate, lit(str(base), datatype=XSD.date)))

        def asset_class(v):
            if v is None:
                return None
            code = ASSET_CLASS_CODES.get(str(v).strip().lower()) or ASSET_CLASS_CODES.get(str(v).strip())
            return FP[f"AssetClass_{code}"] if code else None

        def region(v):
            return FP[f"Region_{slug(v)}"] if v else None

        # --- bonds ----------------------------------------------------------
        n = 0
        for row in rdb.fetch_all(
            "select pd_no, pd_exg_mkt, pd_nm, pd_abrv_nm, product_type, is_sellable, base_date, issuer_name, crd_grd, "
            "std_pd_mcls_nm, bd_knd, pd_risk_gcd, mat_date, srfc_irt from v_bond_kr"
        ):
            pd_no, mkt, name, short, ptype, sell, base, issuer, grade, mcls, knd, risk, mat, coupon = row
            pi = D[f"bond/{pd_no}"]
            common(pi, FP.DomesticBond, pd_no, name, short, ptype, sell, base)
            if issuer:
                ii = D[f"issuer/{slug(issuer)}"]
                if (ii, RDF.type, FP.Issuer) not in g:
                    add((ii, RDF.type, FP.Issuer))
                    add((ii, FP.name, lit(issuer)))
                    add((ii, RDFS.label, lit(issuer, lang="ko")))
                add((pi, FP.issuedBy, ii))
            if grade:
                add((pi, FP.hasCreditGrade, FP[f"CreditGrade_{grade.replace('+', 'plus').replace('-', 'minus')}"]))
            if mcls:
                add((pi, FP.bondMidClass, lit(mcls)))
            if knd:
                add((pi, FP.bondKind, lit(knd)))
            if risk and str(risk).isdigit() and 11 <= int(risk) <= 16:
                add((pi, FP.hasRiskGrade, FP[f"RiskGrade{int(risk) - 10}"]))
            if mat:
                add((pi, FP.maturityDate, lit(mat, datatype=XSD.date)))
            if mkt:
                add((pi, FP.tradingMarket, lit(mkt)))
            n += 1
        log(f"  bonds: {n}")
        # --- domestic ETF/ETN ------------------------------------------------
        n = 0
        for row in rdb.fetch_all(
            "select pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, is_on_sale, base_date, manager_name, "
            "cu_base_index, ref_base_index, wu_inv_ast_type, wu_inv_rgn, risk_grade, cu_strtegy, cu_lev_fector, "
            "listing_date, pd_dvid_cycl, pd_net_tamt from v_etf_kr"
        ):
            code, ticker, short, name, ptype, sale, base, mgr, bidx, ridx, ast, rgn, risk, strat, lev, lst, cyc, aum = row
            pi = D[f"etf_kr/{code}"]
            common(pi, FP.DomesticETN if ptype == "국내ETN" else FP.DomesticETF, code, short or name, name, ptype, sale, base)
            if ticker:
                add((pi, FP.ticker, lit(ticker)))
            mi = manager(mgr)
            if mi is not None:
                add((pi, FP.managedBy, mi))
                add((mi, FP.manages, pi))
            for iname in (bidx, ridx):
                ii = index_node(iname)
                if ii is not None:
                    add((pi, FP.tracksIndex, ii))
            ac = asset_class(ast)
            if ac is not None:
                add((pi, FP.hasAssetClass, ac))
            if rgn:
                add((pi, FP.investsInRegion, region(rgn)))
            if risk:
                add((pi, FP.hasRiskGrade, FP[f"RiskGrade{int(risk)}"]))
            if strat:
                add((pi, FP.strategy, lit(strat)))
            if lev is not None:
                add((pi, FP.leverageFactor, lit(float(lev))))
            if lst:
                add((pi, FP.listingDate, lit(lst, datatype=XSD.date)))
            if cyc and str(cyc).strip():
                add((pi, FP.dividendCycle, FP[f"Cycle_{str(cyc).strip()}"]))
            if aum is not None:
                add((pi, FP.netAssets, lit(float(aum))))
            for ti in themes_for(short, name, bidx, ridx):
                add((pi, FP.hasTheme, ti))
            n += 1
        log(f"  etf_kr: {n}")
        # --- foreign ETF ----------------------------------------------------
        n = 0
        for row in rdb.fetch_all(
            "select pd_itm_no, ticker, pd_nm, product_type, is_on_sale, base_date, manager_name, cu_base_index, "
            "wu_inv_ast_type, wu_inv_rgn, expense_ratio_pct, du_last_aum, cu_lev_fector, cu_inverse_short_yn, listing_date, "
            "cu_strtegy from v_etf_gl"
        ):
            code, ticker, name, ptype, sale, base, mgr, bidx, ast, rgn, er, aum, lev, inv, lst, strat = row
            pi = D[f"etf_gl/{code}"]
            common(pi, FP.ForeignETN if ptype == "해외ETN" else FP.ForeignETF, code, name, ticker, ptype, sale, base)
            if ticker:
                add((pi, FP.ticker, lit(ticker)))
            mi = manager(mgr)
            if mi is not None:
                add((pi, FP.managedBy, mi))
                add((mi, FP.manages, pi))
            ii = index_node(bidx)
            if ii is not None:
                add((pi, FP.tracksIndex, ii))
            ac = asset_class(ast)
            if ac is not None:
                add((pi, FP.hasAssetClass, ac))
            if rgn:
                add((pi, FP.investsInRegion, region(rgn)))
            if er is not None:
                add((pi, FP.expenseRatio, lit(float(er))))
            if aum is not None:
                add((pi, FP.aum, lit(float(aum))))
            if lev is not None:
                add((pi, FP.leverageFactor, lit(float(lev))))
            if inv == "Y":
                add((pi, FP.isInverseOrShort, lit(True)))
            if lst:
                add((pi, FP.listingDate, lit(lst, datatype=XSD.date)))
            for ti in themes_for(name, bidx, strat):
                add((pi, FP.hasTheme, ti))
            n += 1
        log(f"  etf_gl: {n}")
        # --- funds ----------------------------------------------------------
        n = 0
        fam_seen: set[str] = set()
        for row in rdb.fetch_all(
            "select itm_no, itm_nm, itm_abrv_nm, product_type, is_on_sale, base_date, manager_name, fund_family, bmrk_nm, "
            "or_attr_desc, zrin_ptn_nm, fd_ivst_rgn_desc, risk_grade, is_hedged, fd_nast_suma from v_fund_pub"
        ):
            code, name, short, ptype, sale, base, mgr, fam, bmk, attr, zp, rgn, risk, hedged, nav = row
            pi = D[f"fund/{code}"]
            common(pi, FP.PrivateFund if ptype == "사모펀드" else FP.PublicFund, code, name, short, ptype, sale, base)
            add((pi, RDF.type, FP.ShareClass))
            mi = manager(mgr)
            if mi is not None:
                add((pi, FP.managedBy, mi))
                add((mi, FP.manages, pi))
            if fam:
                fi = D[f"family/{slug(fam)}"]
                if fam not in fam_seen:
                    fam_seen.add(fam)
                    add((fi, RDF.type, FP.FundFamily))
                    add((fi, FP.name, lit(fam)))
                    add((fi, RDFS.label, lit(fam, lang="ko")))
                add((pi, FP.memberOfFamily, fi))
            bi = index_node(bmk)
            if bi is not None:
                add((pi, FP.hasBenchmark, bi))
            if attr:
                add((pi, FP.managementAttribute, lit(attr)))
            if zp:
                add((pi, FP.zeroinType, lit(zp)))
            if rgn:
                add((pi, FP.investsInRegion, region(rgn)))
            if risk is not None and str(risk).isdigit():
                add((pi, FP.hasRiskGrade, FP[f"RiskGrade{int(risk)}"]))
            if hedged is not None:
                add((pi, FP.isHedged, lit(bool(hedged))))
            if nav is not None:
                add((pi, FP.netAssets, lit(float(nav))))
            for ti in themes_for(name, bmk, zp):
                add((pi, FP.hasTheme, ti))
            n += 1
        log(f"  funds: {n} ({len(fam_seen)} families)")
        # --- constituents + 기업집단(계열) -----------------------------------
        n, n_groups, n_aff = cls._load_constituents(g, comp_iri, rdb)
        log(f"  constituents: {n} | 기업집단 {n_groups}개 · 계열사 엣지 {n_aff}")
        # --- TBox + inference -----------------------------------------------
        asserted = len(g)
        inferred = cls._materialize(g, ontology, log) if ontology is not None and getattr(ontology, "graph", None) is not None else 0
        log(f"  triples: {len(g):,} (명시 {asserted:,} + 추론 {inferred:,}) | {time.time() - t0:.1f}s")
        kg = cls(g)
        kg.asserted_triples, kg.inferred_triples = asserted, inferred
        return kg

    # --------------------------------------------------------------- inference
    @staticmethod
    def _materialize(g: rdflib.Graph, ontology, log=print) -> int:
        """Merge the TBox and apply the three entailment rules the ontology declares.

        1. type inheritance   x a A, A rdfs:subClassOf B      -> x a B
        2. inverse properties x P y, P owl:inverseOf Q         -> y Q x
        3. domain / range     x P y, P rdfs:domain D / range R -> x a D / y a R

        This is what makes the ontology a reasoning layer rather than a vocabulary: a query for
        `?p a fp:Product` or `?am fp:manages ?p` answers even though only the leaf facts were loaded.
        """
        from rdflib import OWL

        before = len(g)
        for triple in ontology.graph:  # merge TBox (classes, properties, SHACL shapes, enumerations)
            g.add(triple)

        # 1) transitive superclass map
        parents: dict[URIRef, set[URIRef]] = defaultdict(set)
        for c, p in g.subject_objects(RDFS.subClassOf):
            if isinstance(c, URIRef) and isinstance(p, URIRef):
                parents[c].add(p)
        ancestors: dict[URIRef, set[URIRef]] = {}

        def anc(c: URIRef) -> set[URIRef]:
            if c in ancestors:
                return ancestors[c]
            ancestors[c] = set()  # guards cycles
            out: set[URIRef] = set()
            for p in parents.get(c, ()):
                out.add(p)
                out |= anc(p)
            ancestors[c] = out
            return out

        for s, o in list(g.subject_objects(RDF.type)):
            if isinstance(o, URIRef):
                for a in anc(o):
                    g.add((s, RDF.type, a))

        # 2) inverse properties
        inverses: list[tuple[URIRef, URIRef]] = []
        for p, q in g.subject_objects(OWL.inverseOf):
            if isinstance(p, URIRef) and isinstance(q, URIRef):
                inverses += [(p, q), (q, p)]
        for p, q in inverses:
            for s, o in list(g.subject_objects(p)):
                if isinstance(o, URIRef):
                    g.add((o, q, s))

        # 3) domain / range typing (object properties only; literals carry no type)
        used = set(g.predicates(None, None))  # skip TBox properties absent from the data
        for prop, dom in list(g.subject_objects(RDFS.domain)):
            if prop not in used:
                continue
            if not (isinstance(prop, URIRef) and isinstance(dom, URIRef) and str(dom).startswith(str(FP))):
                continue
            for s, _o in list(g.subject_objects(prop)):
                if isinstance(s, URIRef) and str(s).startswith(str(D)):
                    g.add((s, RDF.type, dom))
                    for a in anc(dom):
                        g.add((s, RDF.type, a))
        for prop, rng in list(g.subject_objects(RDFS.range)):
            if prop not in used:
                continue
            if not (isinstance(prop, URIRef) and isinstance(rng, URIRef) and str(rng).startswith(str(FP))):
                continue
            for _s, o in list(g.subject_objects(prop)):
                if isinstance(o, URIRef) and str(o).startswith(str(D)):
                    g.add((o, RDF.type, rng))
                    for a in anc(rng):
                        g.add((o, RDF.type, a))
        return len(g) - before

    @staticmethod
    def _load_constituents(g: rdflib.Graph, comp_iri: dict[str, URIRef], rdb) -> int:
        """Holdings from the store, not from the CSV.

        Reading the same file in two places was how the graph and SQL came to disagree: the ETL kept
        the collector's verdict, the graph re-derived it through a fuzzy name matcher, and only one of
        them was right.  One loader, one table, one answer.

        Identity is the source's own stock code where it exists (79% of rows) and otherwise the name
        the holding was published under — never string similarity.
        """
        try:
            rows = rdb.fetch_all(
                "SELECT etf_itm_no, company_code, company_name, company_id, etf_name FROM etf_constituent WHERE is_cash = 0"
            )
        except Exception:  # noqa: BLE001 - a store built before this table simply has no constituents
            return 0, 0, 0
        n = 0
        group_rows: list[tuple] = []
        for etf_id, code, cname, cid, etf_name in rows:
            if not etf_id or not cname:
                continue
            pi = D[f"etf_kr/{etf_id}"]
            cid = (cid or "").strip() or (f"code_{code}" if code else f"auto_{slug(cname)}")
            ci = comp_iri.get(cid)
            if ci is None:
                ci = D[f"company/{cid}"]
                comp_iri[cid] = ci
                g.add((ci, RDF.type, FP.Company))
                g.add((ci, FP.name, Literal(cname)))
                g.add((ci, RDFS.label, Literal(cname, lang="ko")))
            if code:
                g.add((ci, FP.ticker, Literal(str(code))))
            g.add((pi, FP.hasConstituent, ci))
            g.add((ci, FP.constituentOf, pi))
            group_rows.append((etf_id, etf_name, cid, cname))
            n += 1
        return (n, *KG._load_group_affiliates(g, comp_iri, group_rows))

    @staticmethod
    def _load_group_affiliates(g: rdflib.Graph, comp_iri: dict[str, URIRef], group_rows: list[tuple]) -> tuple[int, int]:
        """Materialize fp:BusinessGroup / fp:affiliateOf from the group-ETF holdings.

        Runs *after* the constituent pass so that ``comp_iri`` already holds every company the
        holdings created — the affiliate edges must land on the same nodes the ETF edges did, or the
        graph gets two 한화에어로스페이스 and every join silently halves.

        fp:affiliateOf is asserted in both directions (it is declared owl:SymmetricProperty, but the
        materializer only closes owl:inverseOf) so a one-hop SPARQL pattern answers from either end.
        """
        n_groups = n_edges = 0
        for token, members in sorted(derive_group_affiliates(group_rows).items()):
            keys = [k for k in members if k in comp_iri]
            if len(keys) < 2:  # 혼자인 '집단'은 계열 관계를 만들지 못한다
                continue
            n_groups += 1
            gi = D[f"group/{slug(token)}"]
            label = f"{token}그룹"
            g.add((gi, RDF.type, FP.BusinessGroup))
            g.add((gi, FP.name, Literal(label)))
            g.add((gi, RDFS.label, Literal(label, lang="ko")))
            for k in keys:
                g.add((comp_iri[k], FP.memberOfGroup, gi))
                g.add((gi, FP.hasGroupMember, comp_iri[k]))
                for etf_id in members[k]:  # 이 계열 관계가 어느 상품에서 나왔는지를 그래프에 남긴다
                    g.add((gi, FP.groupEvidenceProduct, D[f"etf_kr/{etf_id}"]))
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    g.add((comp_iri[a], FP.affiliateOf, comp_iri[b]))
                    g.add((comp_iri[b], FP.affiliateOf, comp_iri[a]))
                    n_edges += 2
        return n_groups, n_edges

    # ------------------------------------------------------------------ query
    def _reindex(self) -> None:
        self._label_index.clear()
        for pred in (FP.name, FP.productCode, FP.ticker, RDFS.label, FP.code):
            for s, o in self.graph.subject_objects(pred):
                self._label_index[_norm(o)].append(s)

    @property
    def size(self) -> int:
        return len(self.graph)

    @staticmethod
    def _uniq(nodes: list[URIRef]) -> list[URIRef]:
        """Distinct nodes, order preserved.

        The label index registers a company under several aliases, so one entity can appear more than
        once for the same IRI.  ``neighbors`` then walks it twice and every count doubles — 삼성전자
        편입 ETF 238개가 476개로, 에코프로비엠 42개가 84개로 보고됐다.
        """
        seen: set[str] = set()
        out: list[URIRef] = []
        for n in nodes:
            if str(n) not in seen:
                seen.add(str(n))
                out.append(n)
        return out

    def find(self, entity: str, limit: int = 5) -> list[URIRef]:
        if entity.startswith("http://"):
            return [URIRef(entity)]
        if entity.startswith("d:") or entity.startswith("fp:"):
            ns = D if entity.startswith("d:") else FP
            return [ns[entity.split(":", 1)[1]]]
        key = _norm(entity)
        exact = self._label_index.get(key, [])
        if exact:
            return self._uniq(exact)[:limit]
        stripped = re.sub(r"(etf|etn|펀드|투자신탁|채권)$", "", key)
        out = []
        for k, v in self._label_index.items():
            if len(key) >= 2 and (key in k or (stripped and len(stripped) >= 2 and stripped in k)):
                out.extend(v)
                if len(self._uniq(out)) >= limit:
                    break
        return self._uniq(out)[:limit]

    def query(self, sparql: str, limit: int = 100) -> list[dict]:
        q = sparql.strip()
        q = re.sub(r"^```(?:sparql)?\s*|\s*```$", "", q, flags=re.I).strip()
        # SPARQL forbids '/' in prefixed local names: rewrite d:etf_kr/KR7... -> <http://mafest.ai/data/etf_kr/KR7...>
        q = re.sub(r"\bd:([A-Za-z0-9_]+/[A-Za-z0-9_.\-가-힣一-鿿]+)", lambda m: f"<{D}{m.group(1)}>", q)
        if "PREFIX" not in q.upper():
            q = PREFIXES + q
        if not re.search(r"\blimit\s+\d+", q, re.I):
            q = f"{q}\nLIMIT {limit}"
        rows = []
        for r in self.graph.query(q):
            rows.append({str(k): self._fmt(v) for k, v in r.asdict().items()})
        return rows

    def _fmt(self, v) -> str:
        if isinstance(v, URIRef):
            s = str(v)
            label = self.graph.value(v, FP.name) or self.graph.value(v, RDFS.label)
            short = s.replace(str(D), "d:").replace(str(FP), "fp:")
            return f"{short} ({label})" if label else short
        return str(v)

    def neighbors(
        self,
        entity: str,
        relation: str | None = None,
        limit: int = 50,
        totals: dict[str, int] | None = None,
    ) -> dict[str, list[dict]]:
        """Relations of ``entity``, at most ``limit`` items per relation.

        Pass ``totals`` to receive the *untruncated, distinct* count per relation.  Callers that
        report a number to the user must use it: answering "몇 개?" with ``len(items)`` silently turns
        the cap into the answer (에코프로비엠 편입 ETF 42개가 60개 상한에 걸려 보고된 사례).

        Targets are counted once per relation even when the entity resolves to several nodes, so a
        product reachable through two aliases is one product, not two.
        """
        out: dict[str, list[dict]] = defaultdict(list)
        tot = totals if totals is not None else {}
        seen: dict[str, set[str]] = defaultdict(set)

        def add(pname: str, target) -> None:
            iri = str(target)
            if iri in seen[pname]:
                return
            seen[pname].add(iri)
            tot[pname] = tot.get(pname, 0) + 1
            if len(out[pname]) < limit:
                out[pname].append({"iri": iri, "label": self._label(target)})

        for node in self.find(entity):
            for p, o in self.graph.predicate_objects(node):
                pname = str(p).split("#")[-1]
                if relation and pname != relation:
                    continue
                if pname in ("type", "label") or pname.startswith("weight_"):
                    continue
                add(pname, o)
            for s, p in self.graph.subject_predicates(node):
                pname = str(p).split("#")[-1] + "_inverse"
                if relation and pname != relation:
                    continue
                add(pname, s)
        return dict(out)

    def subsidiary_candidates(self, entity: str, limit: int = 60) -> dict:
        """'…의 자회사' 질문에 쓸 후보. 지분 기준 자회사가 있으면 그것을, 없으면 계열사를 준다.

        주최측 예시 문항이 '자회사'라는 단어를 쓰므로 이 질문이 답변불가로 떨어지면 안 된다.
        대부분의 경우 자회사 ⊂ 같은 그룹 계열사 − 자기 자신이다 ('에코프로의 자회사' =
        {에코프로, 에코프로비엠, 에코프로머티} − {에코프로}).  fp:affiliateOf 는 자기 자신을 향한
        엣지를 만들지 않으므로 자기 제외는 이미 되어 있다.

        ``kind`` 를 반드시 함께 읽어라: "affiliate" 는 **지분 소유가 확인되지 않은** 관계이므로
        호출자는 '자회사'가 아니라 '계열사'라고 표기해야 한다.
        """
        totals: dict[str, int] = {}
        items = self.neighbors(entity, relation="hasSubsidiary", limit=limit, totals=totals).get("hasSubsidiary", [])
        if items:
            return {"kind": "subsidiary", "items": items, "groups": [], "total": totals.get("hasSubsidiary", len(items))}
        totals = {}
        items = self.neighbors(entity, relation="affiliateOf", limit=limit, totals=totals).get("affiliateOf", [])
        if not items:
            return {"kind": "", "items": [], "groups": [], "total": 0}
        groups = [x["label"] for x in self.neighbors(entity, relation="memberOfGroup", limit=5).get("memberOfGroup", [])]
        return {"kind": "affiliate", "items": items, "groups": groups, "total": totals.get("affiliateOf", len(items))}

    def _label(self, o) -> str:
        if isinstance(o, Literal):
            return str(o)
        v = self.graph.value(o, FP.name) or self.graph.value(o, RDFS.label)
        return str(v) if v is not None else str(o).split("/")[-1].split("#")[-1]

    def export(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(path), format="turtle")


def _norm(s) -> str:
    return re.sub(r"[\s\-_()\[\]·,./&+]+", "", unicodedata.normalize("NFKC", str(s or "")).lower())
