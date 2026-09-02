from finagent.etl.normalize import fund_family_key, risk_grade_from_name, to_flag, yyyymmdd_to_iso


def test_dates():
    assert yyyymmdd_to_iso(20260821.0) == "2026-08-21"
    assert yyyymmdd_to_iso("20260821") == "2026-08-21"
    assert yyyymmdd_to_iso(None) is None
    assert yyyymmdd_to_iso(float("nan")) is None
    assert yyyymmdd_to_iso(99991231.0) == "9999-12-31"
    assert yyyymmdd_to_iso("") is None


def test_flags():
    assert to_flag("Y") == 1 and to_flag("N") == 0
    assert to_flag(1.0) == 1 and to_flag("0") == 0
    assert to_flag("판매중") == 1 and to_flag("판매완료") == 0
    assert to_flag(None) is None


def test_fund_family_key():
    a = fund_family_key("미래에셋국민참여형국민성장혼합자산투자신탁(사모투자재간접형) 종류C1-e (서민형)")
    b = fund_family_key("미래에셋국민참여형국민성장혼합자산투자신탁(사모투자재간접형) 종류C (일반형)")
    assert a == b
    assert fund_family_key("삼성글로벌 Chat AI 증권자투자신탁H[주식]_Ce") == fund_family_key("삼성글로벌 Chat AI 증권자투자신탁H[주식]_A")
    assert fund_family_key("유리필라델피아반도체인덱스증권자투자신탁H[주식] Class C") == fund_family_key(
        "유리필라델피아반도체인덱스증권자투자신탁H[주식] Class A-e"
    )
    assert fund_family_key("한국투자글로벌AI&반도체TOP10증권자투자신탁H(주식)(C-R)") == fund_family_key(
        "한국투자글로벌AI&반도체TOP10증권자투자신탁H(주식)(A-e)"
    )
    # H vs UH (hedged vs unhedged) are different funds and must NOT collapse
    assert fund_family_key("한국투자글로벌AI&반도체TOP10증권자투자신탁H(주식)(A)") != fund_family_key(
        "한국투자글로벌AI&반도체TOP10증권자투자신탁UH(주식)(A)"
    )


def test_risk_grade_from_name():
    assert risk_grade_from_name("매우높은위험(1등급)") == 1
    assert risk_grade_from_name("해당없음") is None
