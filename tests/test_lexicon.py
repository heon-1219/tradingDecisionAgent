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


# --- span shadowing: a shorter alias must not fire inside a longer, unrelated one ---------------


def test_indonesia_is_not_india():
    lx = Lexicon.load(ENRICH)
    ids = {t.id for t in lx.match_themes("한국투자 ACE MSCI인도네시아증권상장지수투자신탁(주식-파생형)(합성)")}
    assert "indonesia" in ids
    assert "india" not in ids
    # the bare country still resolves on its own
    assert "india" in {t.id for t in lx.match_themes("인도 니프티50 ETF")}


def test_us_treasury_is_not_korea_treasury():
    lx = Lexicon.load(ENRICH)
    for name in (
        "한화 PLUS 미국S&P500미국채혼합50액티브증권상장지수투자신탁(채권혼합-재간접형)",
        "메리츠증권 메리츠 미국채30년 상장지수증권(H) 제37호",
        "삼성 KODEX 미국10년국채액티브증권상장지수투자신탁[채권-재간접형](H)",
        "삼성누버거버먼이머징단기채권증권자투자신탁",  # 신흥국채권 -> '국채' straddles the 신흥국+채권 boundary
        "미래에셋 아시아신흥국채권혼합증권자투자신탁",
    ):
        assert "kr_treasury" not in {t.id for t in lx.match_themes(name)}, name
    # 국고채/국공채/국채선물 still map to the 국고채 (Korea treasury) theme
    for name in ("KB RISE 국채선물10년증권상장지수투자신탁(채권-파생형)", "삼성 KODEX 국고채3년증권상장지수투자신탁"):
        assert "kr_treasury" in {t.id for t in lx.match_themes(name)}, name


def test_us_treasury_still_matches_us_treasury_and_usa():
    lx = Lexicon.load(ENRICH)
    ids = {t.id for t in lx.match_themes("한화 PLUS 미국S&P500미국채혼합50액티브증권상장지수투자신탁(채권혼합-재간접형)")}
    assert {"us_treasury", "usa", "sp500"} <= ids


def test_compound_themes_keep_the_themes_they_contain():
    lx = Lexicon.load(ENRICH)
    assert {"china_semi", "china", "semiconductor"} <= {t.id for t in lx.match_themes("TIGER 차이나반도체FACTSET")}
    assert {"china_semi", "china", "semiconductor"} <= {t.id for t in lx.match_themes("KODEX 중국반도체")}


def test_manager_brand_in_a_product_name_is_not_a_theme():
    lx = Lexicon.load(ENRICH)
    # 메리츠 != 리츠, 한국투자 != 한국
    assert "reits" not in {t.id for t in lx.match_themes("메리츠증권 메리츠 인버스 천연가스 선물 상장지수증권(H) 제44호")}
    assert "korea" not in {t.id for t in lx.match_themes("한국투자 ACE 미국배당귀족증권자투자신탁")}
    # 리츠 outside a brand name still matches
    assert "reits" in {t.id for t in lx.match_themes("미래에셋 TIGER 글로벌리츠부동산자투자신탁")}


def test_new_themes_resolve():
    lx = Lexicon.load(ENRICH)
    cases = {
        "metaverse": "삼성 KODEX 메타버스액티브증권상장지수투자신탁[주식]",
        "group_stock": "우리 WON 두산그룹포커스증권상장지수투자신탁(주식)",
        "silver": "삼성 KODEX 은선물특별자산상장지수투자신탁(은-파생형)(H)",
        "copper": "미래에셋 TIGER 구리실물 특별자산상장지수투자신탁(금속)",
        "agriculture": "삼성 KODEX 3대농산물선물특별자산상장지수투자신탁[농산물-파생형](H)",
        "hydrogen": "KB RISE Fn수소경제테마증권상장지수투자신탁(주식)",
        "construction": "삼성KODEX건설증권상장지수투자신탁[주식]",
        "transport": "삼성KODEX운송증권상장지수투자신탁[주식]",
        "factor": "한화ARIRANG스마트베타Quality증권상장지수투자신탁(주식)",
    }
    for theme_id, name in cases.items():
        assert theme_id in {t.id for t in lx.match_themes(name)}, name
    # '수소' does not fire inside '필수소비재'
    assert "hydrogen" not in {t.id for t in lx.match_themes("KODEX 필수소비재")}


def test_ascii_alias_may_be_followed_by_digits_but_not_absorbed():
    lx = Lexicon.load(ENRICH)
    assert "target_date" in {t.id for t in lx.match_themes("KIWOOM 적격TDF2030액티브증권상장지수투자신탁")}
    assert "mmf" in {t.id for t in lx.match_themes("외환스타신종MMF3")}
    # word boundaries are still respected
    assert "ai" not in {t.id for t in lx.match_themes("Daily Value Fund")}
