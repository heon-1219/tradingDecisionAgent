import sqlite3
from pathlib import Path

import pytest

from finagent.kb.graph import KG
from finagent.kb.lexicon import Lexicon
from finagent.kb.rdb import RDB

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_rdb(tmp_path):
    p = tmp_path / "s.db"
    c = sqlite3.connect(p)
    c.execute(
        "create table etf_kr(pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, is_on_sale, base_date, manager_name, "
        "cu_base_index, ref_base_index, wu_inv_ast_type, wu_inv_rgn, risk_grade, cu_strtegy, cu_lev_fector, listing_date, "
        "pd_dvid_cycl, pd_net_tamt)"
    )
    c.execute(
        "insert into etf_kr values ('KR1','305720','KODEX 2차전지산업','삼성 KODEX 2차전지산업증권상장지수투자신탁','국내ETF',1,"
        "'2026-08-21','삼성자산운용','','KRX 2차전지 K-뉴딜지수','주식','국내',1,'실물복제',1.0,'2018-09-11','A',1.0e12)"
    )
    c.execute("create view v_etf_kr as select * from etf_kr")
    c.execute(
        "create view v_bond_kr as select null pd_no, null pd_exg_mkt, null pd_nm, null pd_abrv_nm, null product_type, null is_sellable, "
        "null base_date, null issuer_name, null crd_grd, null std_pd_mcls_nm, null bd_knd, null pd_risk_gcd, null mat_date, null srfc_irt where 0"
    )
    c.execute(
        "create view v_etf_gl as select null pd_itm_no, null ticker, null pd_nm, null product_type, null is_on_sale, null base_date, "
        "null manager_name, null cu_base_index, null wu_inv_ast_type, null wu_inv_rgn, null expense_ratio_pct, null du_last_aum, "
        "null cu_lev_fector, null cu_inverse_short_yn, null listing_date, null cu_strtegy where 0"
    )
    c.execute(
        "create view v_fund_pub as select null itm_no, null itm_nm, null itm_abrv_nm, null product_type, null is_on_sale, null base_date, "
        "null manager_name, null fund_family, null bmrk_nm, null or_attr_desc, null zrin_ptn_nm, null fd_ivst_rgn_desc, null risk_grade, "
        "null is_hedged, null fd_nast_suma where 0"
    )
    c.execute(
        "create table etf_constituent(etf_itm_no, etf_ticker, etf_name, company_code, company_name, "
        "company_id, weight_pct, is_cash, as_of)"
    )
    c.executemany(
        "insert into etf_constituent values (?,?,?,?,?,?,?,?,?)",
        [
            ("KR1", "305720", "KODEX 2차전지산업", "247540", "에코프로비엠", "ecopro_bm", 12.3, 0, "2026-08-21"),
            ("KR1", "305720", "KODEX 2차전지산업", "003670", "포스코퓨처엠", None, 8.0, 0, "2026-08-21"),
            ("KR1", "305720", "KODEX 2차전지산업", None, "듣보잡소재", None, 1.0, 0, "2026-08-21"),
            ("KR1", "305720", "KODEX 2차전지산업", None, "원화현금", None, 3.0, 1, "2026-08-21"),
        ],
    )
    c.execute("create view v_etf_constituent as select * from etf_constituent")
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.commit()
    c.close()
    return RDB(p)


def test_build_query_neighbors(small_rdb, tmp_path):
    kg = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    rows = kg.query("SELECT ?etf WHERE { ?etf fp:hasConstituent ?c . ?c fp:subsidiaryOf ?p . ?p fp:name '에코프로' }")
    assert rows and "etf_kr/KR1" in rows[0]["etf"]
    nb = kg.neighbors("KODEX 2차전지산업")
    assert "hasTheme" in nb and any("battery" in x["iri"] for x in nb["hasTheme"])
    assert "managedBy" in nb and nb["managedBy"][0]["label"] == "삼성자산운용"
    labels = {x["label"] for x in nb["hasConstituent"]}
    assert {"에코프로비엠", "포스코퓨처엠", "듣보잡소재"} <= labels  # by lexicon id, by code, and by name
    assert "원화현금" not in labels  # is_cash rows are holdings but not companies
    inv = kg.neighbors("에코프로", relation="subsidiaryOf_inverse")
    assert {x["label"] for x in inv["subsidiaryOf_inverse"]} >= {"에코프로비엠", "에코프로에이치엔"}
    assert kg.neighbors("305720")["managedBy"]  # ticker lookup
    kg.export(tmp_path / "kg.ttl")
    assert (tmp_path / "kg.ttl").stat().st_size > 1000


def test_query_adds_prefixes_and_limit(small_rdb):
    kg = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    rows = kg.query("SELECT ?p ?n WHERE { ?p a fp:DomesticETF ; fp:name ?n }")
    assert rows[0]["n"] == "KODEX 2차전지산업"


def test_ontology_inference_materializes_supertypes_and_inverses(small_rdb, tmp_path):
    """Axioms + logic rules produce facts nobody asserted (the tech session's inference story)."""
    from finagent.ontology.loader import Ontology

    ont = Ontology.load(ROOT / "ontology")
    kg = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), ont, log=lambda *a: None)
    assert kg.inferred_triples > 0
    # 1) type inheritance: only `a fp:DomesticETF` was asserted
    assert len(kg.query("SELECT ?p WHERE { ?p a fp:DomesticETF }")) == 1
    assert len(kg.query("SELECT ?p WHERE { ?p a fp:ETF }")) == 1
    assert len(kg.query("SELECT ?p WHERE { ?p a fp:Product }")) == 1
    # 2) inverse properties declared in the TBox
    assert kg.query("SELECT ?am WHERE { ?am fp:manages ?p }")
    # 3) the TBox itself is queryable at runtime (ontology as a live layer)
    assert kg.query("SELECT ?c WHERE { ?c rdfs:subClassOf fp:Product }")
    # 4) enumerations are reachable by label/code through find()
    assert kg.neighbors("매우높은위험(1등급)") or kg.neighbors("RiskGrade1")


def test_inference_is_skipped_without_ontology(small_rdb):
    kg = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    assert kg.inferred_triples == 0
    assert len(kg.query("SELECT ?p WHERE { ?p a fp:Product }")) == 0
