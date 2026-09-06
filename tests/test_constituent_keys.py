"""해소된 개체 식별자를 SQL 까지 데려가는 경로의 테스트.

측정된 결함: 지식그래프는 기업 사전의 별칭·티커로 개체를 해소하는데, 모델이 쓰는 SQL 은 그 결과를
받지 못해서 질문의 표면 문자열을 그대로 WHERE 리터럴로 넣고 0건을 받는다.  실측(사전의 기업 97개):
한글명이 편입종목 뷰 표기와 그대로 일치하는 기업 61개, 표기가 어긋나 코드로만 찾아지는 기업 32개,
편입종목 뷰에 행이 아예 없는 기업 4개.  캠브리콘은 뷰에 '한무기'(688256.SS) 로 15행 들어 있다.
"""
import pathlib
import sqlite3

import pytest

from finagent.agent import routing
from finagent.agent.orchestrator import Orchestrator
from finagent.kb.lexicon import Lexicon
from finagent.kb.rdb import RDB

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Tools:
    """Orchestrator 가 매핑을 만들 때 실제로 만지는 것은 ``tools.rdb`` 뿐이다."""

    def __init__(self, rdb):
        self.rdb = rdb


def _orch(rdb, lexicon):
    """스토어와 사전만 붙인 Orchestrator — 매핑과 주입문 형성은 이 둘로 끝난다.

    ToolRegistry·LLM·온톨로지까지 세우지 않는 이유는 새로 넣은 코드가 그것들을 쓰지 않기 때문이다.
    (뷰 이름은 온톨로지가 주지만, 없으면 v_etf_constituent 로 떨어진다 — 그 경로도 함께 확인한다.)
    """
    o = object.__new__(Orchestrator)
    o.tools, o.lexicon, o.ontology = _Tools(rdb), lexicon, None
    o.constituent_keys = o._load_constituent_keys()
    return o


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load(ROOT / "data/enrichment")


_COLS = ["etf_itm_no", "etf_ticker", "etf_name", "company_code", "company_name",
         "company_id", "weight_pct", "is_cash", "as_of"]

# 사전의 한글명과 뷰 표기가 어긋나는 실제 사례들 + 그대로 일치하는 사례 + 현금 행.
_ROWS = [
    ("E1", "1", "ETF1", "688256.SS", "한무기", "cambricon", 5.0, 0, "2026-09-05"),
    ("E2", "2", "ETF2", "688256.SS", "한무기", "cambricon", 4.0, 0, "2026-09-05"),
    ("E1", "1", "ETF1", "035420", "NAVER", "naver", 3.0, 0, "2026-09-05"),
    ("E1", "1", "ETF1", "005930", "삼성전자", "samsung_elec", 9.0, 0, "2026-09-05"),
    ("E1", "1", "ETF1", "247540", "에코프로비엠", "ecopro_bm", 7.0, 0, "2026-09-05"),
    ("E1", "1", "ETF1", "450080", "에코프로머티", "ecopro_materials", 1.0, 0, "2026-09-05"),
    ("E2", "2", "ETF2", "0981.HK", "중국 국제 반도체", "smic", 2.0, 0, "2026-09-05"),
    ("E2", "2", "ETF2", "688981.SS", "중신 국제집성전로제조", "smic", 1.0, 0, "2026-09-05"),
    ("E1", "1", "ETF1", "KRW", "원화현금", "auto_원화현금", 1.0, 1, "2026-09-05"),
]


def _store(dirpath, rows, *, with_company_id=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / "c.db"
    cols = list(_COLS)
    if not with_company_id:
        i = cols.index("company_id")
        cols.pop(i)
        rows = [r[:i] + r[i + 1:] for r in rows]
    c = sqlite3.connect(p)
    c.execute(f"create table etf_constituent({', '.join(cols)})")
    c.executemany(f"insert into etf_constituent values ({','.join('?' * len(cols))})", rows)
    c.execute("create view v_etf_constituent as select * from etf_constituent")
    c.commit()
    c.close()
    return RDB(p)


def test_key_map_is_read_from_the_store_not_guessed(tmp_path, lexicon):
    """매핑은 ETL 이 붙여 둔 company_id 에서 온다 — 사전의 티커로 뷰를 뒤지는 추측이 아니다.

    추측이 왜 안 되는지의 실제 사례: 에어버스의 사전 티커 'AIR' 는 이 데이터에서 AAR 을 가리킨다.
    """
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o.constituent_keys["cambricon"] == [("688256.SS", "한무기", 2)]
    assert o.constituent_keys["naver"] == [("035420", "NAVER", 1)]
    # 코드가 여러 개면 행 수가 많은 것부터
    assert [c for c, _, _ in o.constituent_keys["smic"]] == ["0981.HK", "688981.SS"]
    # is_cash=1 행과 사전에 없는 company_id 는 들어오지 않는다
    assert "auto_원화현금" not in o.constituent_keys


def test_mismatched_korean_name_gets_the_working_key(tmp_path, lexicon):
    """캠브리콘: 그래프는 15건을 찾는데 SQL 은 0건이던 그 문항."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("cambricon"))
    assert "company_code = '688256.SS'" in note      # 쓸 수 있는 키가 들어간다
    assert "'한무기'" in note                         # 뷰의 실제 표기를 사실로 밝힌다
    assert "한글명으로 WHERE 하면 0건" in note          # 하지 말아야 할 것을 명시한다
    assert "\n" not in note                          # 기업 하나당 한 줄을 넘기지 않는다


def test_multiple_codes_become_an_in_list(tmp_path, lexicon):
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("smic"))
    assert "company_code IN ('0981.HK', '688981.SS')" in note


def test_matching_korean_name_is_left_alone(tmp_path, lexicon):
    """한글명이 그대로 통하는 61개 기업의 동작은 바뀌면 안 된다 — 한 글자도 붙이지 않는다."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o._constituent_key_note(lexicon.company("samsung_elec")) == ""
    assert o._constituent_key_note(lexicon.company("ecopro_bm")) == ""


def test_absent_company_states_the_absence_as_a_fact(tmp_path, lexicon):
    """0건을 '내 SQL 이 틀렸나' 가 아니라 '데이터에 없다' 로 읽게 만드는 줄."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("ecopro_hn"))
    assert "0건" in note and "편입종목 데이터에 이 기업이 없다" in note
    assert "company_code" not in note                # 없는 키를 지어내지 않는다


def test_store_without_company_id_degrades_to_previous_behaviour(tmp_path, lexicon):
    """열이 없는 스토어에서는 매핑이 비고, 매핑이 비면 주입문은 예전 그대로다."""
    o = _orch(_store(tmp_path, _ROWS, with_company_id=False), lexicon)
    assert o.constituent_keys == {}
    assert o._constituent_key_note(lexicon.company("cambricon")) == ""
    assert o._constituent_key_note(lexicon.company("ecopro_hn")) == ""   # 없음 단정도 하지 않는다
    assert o._subsidiary_key_line(lexicon.subsidiaries_of("ecopro")) == ""


def test_subsidiary_keys_appear_when_a_name_is_off(tmp_path, lexicon):
    """'에코프로의 자회사를 편입한 ETF' — 모델이 WHERE 에 넣는 것은 이 자회사 이름들이다.

    셋 중 에코프로머티리얼즈는 뷰 표기가 다르고 에코프로에이치엔은 행이 없다.  하나라도 어긋나면
    전부를 싣는다 — 일부만 주면 나머지가 다시 추측 대상이 되기 때문이다.
    """
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    line = o._subsidiary_key_line(lexicon.subsidiaries_of("ecopro"))
    assert "에코프로비엠 → company_code='247540'" in line
    assert "'450080'" in line and "에코프로머티" in line
    assert "에코프로에이치엔" in line and "0건" in line
    assert "\n" not in line


def test_subsidiary_line_is_silent_when_every_name_matches(tmp_path, lexicon):
    """어긋나는 것이 없으면 한 줄도 늘리지 않는다 — 주입은 매 질문마다 나간다."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o._subsidiary_key_line([lexicon.company("ecopro_bm")]) == ""
    assert o._subsidiary_key_line([]) == ""


# ---------------------------------------------------------------- 홀더 추출 (routing.py)
def test_relation_words_are_not_holders():
    """graph_neighbors('자회사', constituentOf) -> 0건 이던 실측 결함."""
    for word, q in (("자회사", "에코프로의 자회사를 편입한 ETF 중 순자산이 큰 상품의 위험요인 알려줘"),
                    ("계열사", "한화 에어로스페이스의 계열사를 편입한 ETF"),
                    ("관련주", "에코프로 관련주를 담은 ETF"),
                    ("모회사", "에코프로비엠의 모회사를 보유한 ETF")):
        assert word not in routing.holder_mentions(q), q


def test_resolved_company_beats_a_surface_token():
    """표면 추출은 동사 앞 한 토큰만 본다 — 해소된 기업이 있으면 그쪽을 먼저 믿는다."""
    plan = routing.plan_retrieval(routing.Signals(
        question="에코프로의 자회사를 편입한 ETF 중 순자산이 큰 상품의 위험요인 알려줘",
        companies=["에코프로"]))
    assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == ["에코프로"]


def test_surface_fallback_survives_for_companies_not_in_the_lexicon():
    """쿠팡·야놀자는 사전에 없지만 정답이 '해당 없음'이고 그것을 아는 곳은 그래프뿐이다."""
    for q, who in (("쿠팡을 편입한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "쿠팡"),
                   ("야놀자를 보유한 국내 ETF가 있나요?", "야놀자")):
        plan = routing.plan_retrieval(routing.Signals(question=q))
        assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == [who], q


def test_named_holder_still_wins_over_another_company_in_the_sentence():
    """문장에 기업이 둘이면 동사 앞의 그 기업이 대상이다 — 순서가 뒤집히면 안 된다."""
    plan = routing.plan_retrieval(routing.Signals(
        question="엔비디아 테마 ETF 중 삼성전자를 편입한 상품", companies=["엔비디아", "삼성전자"]))
    assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == ["삼성전자"]
