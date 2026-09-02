from finagent.kb.search import Doc, SearchIndex, tokenize


def test_tokenize_korean_bigrams_and_words():
    toks = tokenize("KODEX 로봇액티브 ETF")
    assert "kodex" in toks and "로봇" in toks and "봇액" in toks
    assert "sem" in tokenize("Semiconductors")


def test_search_ranks_exact_name_first():
    docs = [
        Doc("1", "v_etf_kr", "KODEX 로봇액티브", "KODEX 로봇액티브 삼성자산운용", {}),
        Doc("2", "v_etf_kr", "RISE AI&로봇", "RISE AI&로봇 KB자산운용", {}),
        Doc("3", "v_etf_kr", "TIGER 반도체", "TIGER 반도체 미래에셋", {}),
        Doc("4", "v_etf_gl", "VanEck China Semiconductor ETF", "VanEck China Semiconductor ETF SMHC China Equity", {}),
    ]
    idx = SearchIndex(docs)
    hits = idx.search("KODEX 로봇", k=2)
    assert hits[0].id == "1"
    assert idx.search("반도체", table="v_etf_kr")[0].id == "3"
    assert idx.search("china semiconductor")[0].id == "4"
    assert idx.search("KODEX 로봇액티브")[0].score >= 100


def test_empty_index_and_query():
    assert SearchIndex([]).search("x") == []
    assert SearchIndex([Doc("1", "t", "a", "a", {})]).search("   ") == []
