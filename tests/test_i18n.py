import pytest

from finagent.i18n import detect_lang, language_name, pick


@pytest.mark.parametrize("question,lang", [
    # plain English and plain Korean
    ("How many AA- or higher KRW bonds can I buy?", "en"),
    ("구매 가능한 원화채권 중 AA- 이상은 몇 개야?", "ko"),
    # English questions that name Korean products or companies stay English
    ("Which ETFs hold 삼성전자?", "en"),
    ("삼성전자 holdings?", "en"),
    ("What is the expense ratio of TIGER 미국S&P500?", "en"),
    ("Compare 미래에셋전략배분TDF2045 and 삼성한국형TDF2045", "en"),
    ("List ETFs holding 삼성전자우(005935)", "en"),
    # Korean questions with Latin brand names, codes or abbreviations stay Korean
    ("TIGER 미국S&P500 총보수는?", "ko"),
    ("KODEX 200 순자산 알려줘", "ko"),
    ("KODEX 200 vs TIGER 200 비교해줘", "ko"),
    ("iShares 운용 ETF 중 총보수 낮은 거", "ko"),
    ("삼성전자를 보유한 ETF top 3", "ko"),
    # Korean grammar wins even when a long English product name outvotes the Korean words
    ("VanEck China Semiconductor ETF의 총보수는 몇 퍼센트인가요?", "ko"),
    ("Invesco QQQ Trust, Series 1의 운용사 이름을 알려줘", "ko"),
    ("VanEck China Semiconductor ETF의 총보수와 기초지수는?", "ko"),
    # ...but a Korean name that merely starts with Latin letters is not a particle
    ("Which ETFs hold SK하이닉스?", "en"),
    ("What does KODEX 차이나H hold?", "en"),
    ("How much LG에너지솔루션 does TIGER 2차전지테마 hold?", "en"),
    # nothing to go on -> English
    ("", "en"),
    (None, "en"),
    ("005930", "en"),
    ("KODEX 200", "en"),
])
def test_detect_lang(question, lang):
    assert detect_lang(question) == lang


def test_pick_and_language_name():
    assert pick("en", "No data.", "데이터 없음") == "No data."
    assert pick("ko", "No data.", "데이터 없음") == "데이터 없음"
    assert language_name("en") == "English" and language_name("ko") == "Korean"
