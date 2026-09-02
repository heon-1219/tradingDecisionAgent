"""Knowledge graph (ABox) built at startup from the SQLite store + curated lexicons + constituents.

TBox (classes/properties) lives in ontology/*.ttl; this module instantiates products, managers,
issuers, indices, themes, companies and fund families as RDF and exposes SPARQL + neighbor lookups.
"""
from __future__ import annotations

import csv
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


class KG:
    def __init__(self, g: rdflib.Graph):
        self.graph = g
        self._label_index: dict[str, list[URIRef]] = defaultdict(list)
        self._reindex()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, rdb, lexicon, ontology=None, constituents_dir: Path | None = None, log=print) -> "KG":
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
        # --- constituents ---------------------------------------------------
        n = cls._load_constituents(g, lexicon, comp_iri, constituents_dir) if constituents_dir else 0
        log(f"  constituents: {n} | triples: {len(g):,} | {time.time() - t0:.1f}s")
        return cls(g)

    @staticmethod
    def _load_constituents(g: rdflib.Graph, lexicon, comp_iri: dict[str, URIRef], d: Path) -> int:
        d = Path(d)
        if not d.exists():
            return 0
        n = 0
        table_prefix = {"v_etf_kr": "etf_kr", "v_etf_gl": "etf_gl", "v_fund_pub": "fund", "v_bond_kr": "bond"}
        for f in sorted(d.glob("*.csv")):
            with open(f, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    etf_id, table = (r.get("etf_id") or "").strip(), (r.get("etf_table") or "v_etf_kr").strip()
                    if not etf_id:
                        continue
                    pi = D[f"{table_prefix.get(table, 'etf_kr')}/{etf_id}"]
                    cid = (r.get("company_id") or "").strip()
                    cname = (r.get("company_name") or "").strip()
                    if _norm(cname) in _CASH_LIKE:
                        continue
                    if not cid or cid.startswith("auto_"):
                        c = lexicon.company_by_alias(cname)  # re-resolve with the current lexicon
                        cid = c.id if c else (cid or f"auto_{slug(cname)}")
                    ci = comp_iri.get(cid)
                    if ci is None:
                        ci = D[f"company/{cid}"]
                        comp_iri[cid] = ci
                        g.add((ci, RDF.type, FP.Company))
                        g.add((ci, FP.name, Literal(cname or cid)))
                        g.add((ci, RDFS.label, Literal(cname or cid, lang="ko")))
                    g.add((pi, FP.hasConstituent, ci))
                    g.add((ci, FP.constituentOf, pi))
                    w = (r.get("weight_pct") or "").strip()
                    if w:
                        try:
                            # reified weight kept simple: one statement per (etf, company) pair
                            g.add((pi, FP[f"weight_{slug(cid)}"], Literal(float(w))))
                        except ValueError:
                            pass
                    n += 1
        return n

    # ------------------------------------------------------------------ query
    def _reindex(self) -> None:
        self._label_index.clear()
        for s, o in self.graph.subject_objects(FP.name):
            self._label_index[_norm(o)].append(s)
        for s, o in self.graph.subject_objects(FP.productCode):
            self._label_index[_norm(o)].append(s)
        for s, o in self.graph.subject_objects(FP.ticker):
            self._label_index[_norm(o)].append(s)

    @property
    def size(self) -> int:
        return len(self.graph)

    def find(self, entity: str, limit: int = 5) -> list[URIRef]:
        if entity.startswith("http://"):
            return [URIRef(entity)]
        if entity.startswith("d:") or entity.startswith("fp:"):
            ns = D if entity.startswith("d:") else FP
            return [ns[entity.split(":", 1)[1]]]
        key = _norm(entity)
        exact = self._label_index.get(key, [])
        if exact:
            return exact[:limit]
        stripped = re.sub(r"(etf|etn|펀드|투자신탁|채권)$", "", key)
        out = []
        for k, v in self._label_index.items():
            if len(key) >= 2 and (key in k or (stripped and len(stripped) >= 2 and stripped in k)):
                out.extend(v)
                if len(out) >= limit:
                    break
        return out[:limit]

    def query(self, sparql: str, limit: int = 100) -> list[dict]:
        q = sparql.strip()
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

    def neighbors(self, entity: str, relation: str | None = None, limit: int = 50) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for node in self.find(entity):
            for p, o in self.graph.predicate_objects(node):
                pname = str(p).split("#")[-1]
                if relation and pname != relation:
                    continue
                if pname in ("type", "label") or pname.startswith("weight_"):
                    continue
                if len(out[pname]) < limit:
                    out[pname].append({"iri": str(o), "label": self._label(o)})
            for s, p in self.graph.subject_predicates(node):
                pname = str(p).split("#")[-1] + "_inverse"
                if relation and pname != relation:
                    continue
                if len(out[pname]) < limit:
                    out[pname].append({"iri": str(s), "label": self._label(s)})
        return dict(out)

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
