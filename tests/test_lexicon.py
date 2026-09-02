from pathlib import Path

from finagent.kb.lexicon import Lexicon

ENRICH = Path(__file__).resolve().parents[1] / "data" / "enrichment"


def test_load_and_match_theme():
    lx = Lexicon.load(ENRICH)
    ids = {t.id for t in lx.match_themes("최근 6개월 동안 우주항공 테마와 연결된 ETF")}
    assert "space_aero" in ids


def test_match_company_alias_and_subsidiaries():
    lx = Lexicon.load(ENRICH)
    comps = lx.match_companies("캠브리콘이 편입된 중국반도체 ETF")
    assert any(c.id == "cambricon" for c in comps)
    subs = {c.id for c in lx.subsidiaries_of("ecopro")}
    assert {"ecopro_bm", "ecopro_hn"} <= subs


def test_manager_for_product_name():
    lx = Lexicon.load(ENRICH)
    assert lx.manager_for_product_name("미래에셋국민참여형국민성장혼합자산투자신탁") == "미래에셋자산운용"
    assert lx.manager_for_product_name("KODEX 로봇액티브") == "삼성자산운용"
    assert lx.manager_for_product_name("알수없는이름") is None


def test_manager_canonical_raw_value():
    lx = Lexicon.load(ENRICH)
    assert lx.manager_canonical("미래에셋증권 주식회사") == "미래에셋증권"
    assert lx.manager_canonical("KB") == "KB자산운용"
