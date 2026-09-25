from pathlib import Path

from finagent.kb.entity import EntityResolver
from finagent.kb.lexicon import Lexicon
from finagent.kb.search import Doc, SearchIndex

ENRICH = Path(__file__).resolve().parents[1] / "data" / "enrichment"


def _idx():
    return SearchIndex(
        [
            Doc("1", "v_etf_kr", "KODEX 로봇액티브", "KODEX 로봇액티브 삼성자산운용", {}),
            Doc("2", "v_etf_kr", "RISE AI&로봇", "RISE AI&로봇 KB자산운용", {}),
            Doc("3", "v_etf_kr", "KODEX 2차전지산업", "KODEX 2차전지산업 삼성자산운용", {}),
        ]
    )


def test_resolver_marks_unknown_product_as_inexact_with_candidates():
    r = EntityResolver(_idx(), Lexicon.load(ENRICH)).resolve("KODEX AI로봇 ETF 정보 알려줘")
    assert not r.exact
    assert r.products[0].title == "KODEX 로봇액티브"
    assert "Similar product candidates" in r.summary()


def test_extract_product_mentions():
    from finagent.kb.entity import extract_product_mentions

    assert extract_product_mentions("KODEX AI로봇 ETF 정보 알려줘") == ["KODEX AI로봇"]
    assert extract_product_mentions("국민성장펀드의 구조와 투자전략 동향 등 찾아서 알려줘") == ["국민성장펀드"]
    assert "TIGER 2차전지테마" in extract_product_mentions("TIGER 2차전지테마의 보수율과 위험을 알려줘")
    assert extract_product_mentions("현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘") == []
    assert "VanEck China Semiconductor ETF" in extract_product_mentions("VanEck China Semiconductor ETF 총보수는?")


def test_resolver_exact_match():
    r = EntityResolver(_idx(), Lexicon.load(ENRICH)).resolve("KODEX 로봇액티브")
    assert r.exact and r.products[0].id == "1"


def test_resolver_company_and_theme():
    r = EntityResolver(_idx(), Lexicon.load(ENRICH)).resolve("에코프로의 자회사를 편입한 2차전지 ETF")
    assert any(c.id == "ecopro" for c in r.companies)
    assert any(t.id == "battery" for t in r.themes)
    r2 = EntityResolver(_idx(), Lexicon.load(ENRICH)).resolve("Kimi 관련 투자상품 있어?")
    assert any(c.id == "moonshot_kimi" and c.not_listed for c in r2.companies)


def test_resolution_flags_nonexistent_product_in_summary():
    """The tool output and the pre-analysis must warn 'does not exist' by the same criterion."""
    lx = Lexicon.load(ENRICH)
    r = EntityResolver(_idx(), lx).resolve("KODEX AI로봇")
    assert not r.resolved and "exists in the data" in r.summary()
    ok = EntityResolver(_idx(), lx).resolve("KODEX 로봇액티브")
    assert ok.resolved and "exists in the data" not in ok.summary()


def test_mention_resolves_tolerates_prefix_and_suffix():
    from finagent.kb.entity import mention_resolves

    assert mention_resolves("미래에셋 TIGER 미국나스닥100", "TIGER 미국나스닥100")
    assert mention_resolves("국민성장펀드", "미래에셋국민참여형국민성장혼합자산투자신탁(사모투자재간접형) 종류C")
    assert not mention_resolves("KODEX AI로봇", "KODEX 로봇액티브")
    assert not mention_resolves("TIGER 양자컴퓨팅커버드콜", "TIGER 미국나스닥100커버드콜")
