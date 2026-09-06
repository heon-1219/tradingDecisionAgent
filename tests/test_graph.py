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


# ---------------------------------------------------------------- group affiliates


def test_parse_group_etf_classifies_the_real_17_group_etfs():
    """이름만으로 순수형/혼합형/제외를 가른다. 값은 스토어의 실제 17개 그룹주 ETF 이름이다."""
    from finagent.kb.graph import parse_group_etf

    assert parse_group_etf("KODEX 삼성그룹") == ("삼성", True)
    assert parse_group_etf("TIGER 삼성그룹") == ("삼성", True)
    assert parse_group_etf("PLUS 한화그룹주") == ("한화", True)
    assert parse_group_etf("KODEX 삼성그룹밸류") == ("삼성", True)
    assert parse_group_etf("ACE 삼성그룹동일가중") == ("삼성", True)
    assert parse_group_etf("ACE 삼성그룹섹터가중") == ("삼성", True)
    assert parse_group_etf("KODEX 인도타타그룹") == ("타타", True)  # 투자지역 수식어는 그룹명이 아니다
    # 혼합형: '관련주'를 섞으므로 이름 토큰으로 걸러야 한다
    assert parse_group_etf("TIGER LG그룹플러스") == ("LG", False)
    assert parse_group_etf("TIGER 현대차그룹플러스") == ("현대차", False)
    assert parse_group_etf("ACE 포스코그룹포커스") == ("포스코", False)
    assert parse_group_etf("BNK 카카오그룹포커스") == ("카카오", False)
    assert parse_group_etf("WON 두산그룹포커스") == ("두산", False)
    assert parse_group_etf("KIWOOM SK그룹대표주") == ("SK", False)
    # 제외: 편입종목이 기업이 아니다(채권 종목·국고채·선물)
    assert parse_group_etf("1Q 현대차그룹채권(A+이상)&국고통안") is None
    assert parse_group_etf("KIWOOM 현대차그룹TOP3채권혼합50") is None
    assert parse_group_etf("RISE 삼성그룹Top3채권혼합") is None
    # 제외: 하나의 그룹에 귀속시킬 수 없다
    assert parse_group_etf("RISE 5대그룹주") is None
    assert parse_group_etf("ACE 인도시장대표BIG5그룹액티브") is None
    assert parse_group_etf("KODEX 2차전지산업") is None


def test_derive_group_affiliates_keeps_related_names_out_of_mixed_etfs():
    """혼합형에서 '관련주'가 계열사로 새어 들어가면 오탐 하나가 오답 하나가 된다."""
    from finagent.kb.graph import derive_group_affiliates

    rows = [
        # 혼합형: TIGER LG그룹플러스 는 GS·LS·휴젤 같은 다른 그룹을 함께 담는다
        ("E1", "TIGER LG그룹플러스", "lg_elec", "LG전자"),
        ("E1", "TIGER LG그룹플러스", "auto_LG", "LG"),
        ("E1", "TIGER LG그룹플러스", "auto_GS건설", "GS건설"),
        ("E1", "TIGER LG그룹플러스", "auto_LS", "LS ELECTRIC"),
        ("E1", "TIGER LG그룹플러스", "auto_휴젤", "휴젤"),
        # 순수형: 편입종목 전부가 계열사다 — 에스원은 '삼성' 접두어가 없지만 실제 삼성 계열사다
        ("E2", "KODEX 삼성그룹", "samsung_elec", "삼성전자"),
        ("E2", "KODEX 삼성그룹", "auto_에스원", "에스원"),
        # 합집합: 같은 그룹의 다른 ETF 가 더하는 종목
        ("E3", "TIGER 삼성그룹", "auto_호텔신라", "호텔신라"),
        # 제외 대상: 국고채·선물을 회사로 채택하면 안 된다
        ("E4", "RISE 삼성그룹Top3채권혼합", "auto_삼성전자선물2609", "삼성전자선물2609"),
    ]
    groups = derive_group_affiliates(rows)
    assert set(groups) == {"LG", "삼성"}
    assert set(groups["LG"]) == {"lg_elec", "auto_LG"}
    assert set(groups["삼성"]) == {"samsung_elec", "auto_에스원", "auto_호텔신라"}
    assert groups["삼성"]["samsung_elec"] == ["E2"]  # 근거 상품이 남는다


@pytest.fixture
def group_rdb(tmp_path):
    """그룹주 ETF 를 담은 스토어. 이름·편입종목은 실제 v_etf_constituent 에서 가져온 값이다."""
    p = tmp_path / "g.db"
    c = sqlite3.connect(p)
    c.execute(
        "create table etf_kr(pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, is_on_sale, base_date, manager_name, "
        "cu_base_index, ref_base_index, wu_inv_ast_type, wu_inv_rgn, risk_grade, cu_strtegy, cu_lev_fector, listing_date, "
        "pd_dvid_cycl, pd_net_tamt)"
    )
    etfs = [  # (id, ticker, 약칭, 순자산)
        ("G1", "0000J0", "PLUS 한화그룹주", 1.143e11),
        ("G2", "138530", "TIGER LG그룹플러스", 8.25e10),
        ("G3", "448630", "RISE 삼성그룹Top3채권혼합", 3.27e11),
        ("G4", "105780", "RISE 5대그룹주", 3.39e10),
        ("G5", "445290", "SOL 조선TOP3플러스", 1.5e12),  # 그룹주가 아닌 일반 ETF
    ]
    for eid, tk, nm, aum in etfs:
        c.execute(
            "insert into etf_kr values (?,?,?,?,'국내ETF',1,'2026-08-21','운용사','','지수','주식','국내',2,'실물복제',1.0,"
            "'2020-01-01','A',?)",
            (eid, tk, nm, nm, aum),
        )
    c.execute("create view v_etf_kr as select * from etf_kr")
    for v, cols in (
        ("v_bond_kr", "pd_no, pd_exg_mkt, pd_nm, pd_abrv_nm, product_type, is_sellable, base_date, issuer_name, crd_grd, "
                      "std_pd_mcls_nm, bd_knd, pd_risk_gcd, mat_date, srfc_irt"),
        ("v_etf_gl", "pd_itm_no, ticker, pd_nm, product_type, is_on_sale, base_date, manager_name, cu_base_index, "
                     "wu_inv_ast_type, wu_inv_rgn, expense_ratio_pct, du_last_aum, cu_lev_fector, cu_inverse_short_yn, "
                     "listing_date, cu_strtegy"),
        ("v_fund_pub", "itm_no, itm_nm, itm_abrv_nm, product_type, is_on_sale, base_date, manager_name, fund_family, bmrk_nm, "
                       "or_attr_desc, zrin_ptn_nm, fd_ivst_rgn_desc, risk_grade, is_hedged, fd_nast_suma"),
    ):
        c.execute(f"create view {v} as select " + ", ".join(f"null {x.strip()}" for x in cols.split(",")) + " where 0")
    c.execute(
        "create table etf_constituent(etf_itm_no, etf_ticker, etf_name, company_code, company_name, "
        "company_id, weight_pct, is_cash, as_of)"
    )
    rows = [
        # 순수형 → 편입종목 전부 채택
        ("G1", "PLUS 한화그룹주", "012450", "한화에어로스페이스", "hanwha_aero", 22.08, 0),
        ("G1", "PLUS 한화그룹주", "042660", "한화오션", "hanwha_ocean", 17.82, 0),
        ("G1", "PLUS 한화그룹주", "272210", "한화시스템", "hanwha_systems", 15.08, 0),
        ("G1", "PLUS 한화그룹주", "000880", "한화", "auto_한화", 7.53, 0),
        ("G1", "PLUS 한화그룹주", None, "원화현금", "auto_원화현금", 0.31, 1),
        # 혼합형 → 그룹명 토큰을 가진 종목만 채택 (GS건설·휴젤은 다른 그룹이다)
        ("G2", "TIGER LG그룹플러스", "066570", "LG전자", "lg_elec", 13.67, 0),
        ("G2", "TIGER LG그룹플러스", "051910", "LG화학", "lg_chem", 16.0, 0),
        ("G2", "TIGER LG그룹플러스", "006360", "GS건설", "auto_GS건설", 7.11, 0),
        ("G2", "TIGER LG그룹플러스", "145020", "휴젤", "auto_휴젤", 0.48, 0),
        # 채권혼합 → 제외 (편입종목이 국고채·선물이다)
        ("G3", "RISE 삼성그룹Top3채권혼합", "005930", "삼성전자", "samsung_elec", 16.82, 0),
        ("G3", "RISE 삼성그룹Top3채권혼합", None, "삼성전자선물2609", "auto_삼성전자선물2609", 0.79, 0),
        # 복수그룹 → 제외
        ("G4", "RISE 5대그룹주", "005930", "삼성전자", "samsung_elec", 10.1, 0),
        ("G4", "RISE 5대그룹주", "000660", "SK하이닉스", "sk_hynix", 9.33, 0),
        # 그룹주가 아닌 ETF 도 계열사를 편입한다 — 예시 문항이 찾아야 하는 상품이다
        ("G5", "SOL 조선TOP3플러스", "042660", "한화오션", "hanwha_ocean", 25.0, 0),
    ]
    c.executemany(
        "insert into etf_constituent values (?,?,?,?,?,?,?,?,'2026-08-21')",
        [(r[0], None, r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows],
    )
    c.execute("create view v_etf_constituent as select * from etf_constituent")
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.commit()
    c.close()
    return RDB(p)


def test_group_etf_holdings_become_affiliate_edges(group_rdb):
    kg = KG.build(group_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)

    def members(group):
        return {r["n"] for r in kg.query(
            "SELECT ?n WHERE { ?g fp:name '%s' ; fp:hasGroupMember ?c . ?c fp:name ?n }" % group)}

    assert members("한화그룹") == {"한화에어로스페이스", "한화오션", "한화시스템", "한화"}  # 순수형: 전부
    assert members("LG그룹") == {"LG전자", "LG화학"}  # 혼합형: GS건설·휴젤은 계열이 아니다
    assert members("삼성그룹") == set()  # 채권혼합·5대그룹주만 있었으므로 그룹 자체가 안 생긴다
    assert not kg.query("SELECT ?c WHERE { ?c fp:affiliateOf ?o . ?c fp:name '삼성전자선물2609' }")

    # 대칭이며 자기 자신을 향하지 않는다
    aff = {r["n"] for r in kg.query(
        "SELECT ?n WHERE { ?a fp:name '한화오션' ; fp:affiliateOf ?b . ?b fp:name ?n }")}
    assert aff == {"한화에어로스페이스", "한화시스템", "한화"}
    assert kg.query("SELECT ?b WHERE { ?a fp:name 'LG화학' ; fp:affiliateOf ?b . ?b fp:name 'LG전자' }")
    assert kg.query("SELECT ?b WHERE { ?a fp:name 'LG전자' ; fp:affiliateOf ?b . ?b fp:name 'LG화학' }")
    # 근거가 그래프에 남는다
    assert kg.query("SELECT ?p WHERE { ?g fp:name '한화그룹' ; fp:groupEvidenceProduct ?p }")


def test_subsidiary_question_falls_back_to_affiliates(group_rdb, small_rdb):
    """'자회사' 질문이 답변불가로 떨어지지 않게 하되, 관계는 '계열사'라고 밝힌다."""
    kg = KG.build(group_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    r = kg.subsidiary_candidates("한화에어로스페이스")
    assert r["kind"] == "affiliate" and r["groups"] == ["한화그룹"]
    assert {x["label"] for x in r["items"]} == {"한화오션", "한화시스템", "한화"}  # 자기 자신 제외

    # 지분 기준 자회사가 있으면 그쪽이 우선이다 (에코프로·SK이노베이션의 손으로 확인한 4개 엣지)
    kg2 = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    r2 = kg2.subsidiary_candidates("에코프로")
    assert r2["kind"] == "subsidiary"
    assert {x["label"] for x in r2["items"]} >= {"에코프로비엠", "에코프로에이치엔"}


def test_affiliate_path_answers_the_worked_example(group_rdb):
    """'한화에어로스페이스의 자회사를 편입한 ETF 중 순자산이 큰 상품' 의 그래프 경로."""
    kg = KG.build(group_rdb, Lexicon.load(ROOT / "data/enrichment"), None, log=lambda *a: None)
    rows = kg.query(
        "SELECT ?etfName ?aum WHERE { ?me fp:name '한화에어로스페이스' ; fp:affiliateOf ?aff . "
        "?etf fp:hasConstituent ?aff ; fp:name ?etfName ; fp:netAssets ?aum } "
        "ORDER BY DESC(xsd:double(?aum))"
    )
    assert rows, "계열사를 편입한 ETF 를 찾지 못했다"
    assert rows[0]["etfName"] == "SOL 조선TOP3플러스"  # 한화오션 편입, 순자산 1.5조로 최대
