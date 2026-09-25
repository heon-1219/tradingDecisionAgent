"""Tests for the path that carries resolved entity identifiers all the way to SQL.

The measured defect: the knowledge graph resolves entities through the company lexicon's aliases and
tickers, but the SQL the model writes never received that result, so it put the question's surface
string straight into a WHERE literal and got 0 rows.  Measured (the lexicon's 97 companies): 61 whose
Korean name matches the holdings view's spelling as is, 32 whose spelling differs and are found only by
code, 4 with no rows in the holdings view at all.  캠브리콘 is in the view as '한무기' (688256.SS), 15 rows.
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
    """All the Orchestrator really touches when building the map is ``tools.rdb``."""

    def __init__(self, rdb):
        self.rdb = rdb


def _orch(rdb, lexicon):
    """An Orchestrator with only the store and the lexicon attached — the map and the injected text need nothing else.

    ToolRegistry, LLM and ontology are not set up because the new code does not use them.
    (The view name comes from the ontology, but without one it falls back to v_etf_constituent — that
    path is checked too.)
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

# Real cases where the lexicon's Korean name and the view's spelling differ + cases that match + a cash row.
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
    """The map comes from the company_id the ETL attached — not from guessing by searching the view with the lexicon's ticker.

    A real case of why guessing fails: Airbus's lexicon ticker 'AIR' points to AAR in this data.
    """
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o.constituent_keys["cambricon"] == [("688256.SS", "한무기", 2)]
    assert o.constituent_keys["naver"] == [("035420", "NAVER", 1)]
    # several codes: the one with the most rows first
    assert [c for c, _, _ in o.constituent_keys["smic"]] == ["0981.HK", "688981.SS"]
    # is_cash=1 rows and company_ids not in the lexicon do not come in
    assert "auto_원화현금" not in o.constituent_keys


def test_mismatched_korean_name_gets_the_working_key(tmp_path, lexicon):
    """캠브리콘: the question where the graph found 15 and SQL found 0."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("cambricon"))
    assert "company_code = '688256.SS'" in note                   # a key that works goes in
    assert "'한무기'" in note                                      # the view's real spelling is stated as a fact
    assert "a WHERE on the Korean name returns 0 rows" in note    # what not to do is stated
    assert "\n" not in note                                       # no more than one line per company


def test_multiple_codes_become_an_in_list(tmp_path, lexicon):
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("smic"))
    assert "company_code IN ('0981.HK', '688981.SS')" in note


def test_matching_korean_name_is_left_alone(tmp_path, lexicon):
    """The behaviour for the 61 companies whose Korean name works as is must not change — not a character is added."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o._constituent_key_note(lexicon.company("samsung_elec")) == ""
    assert o._constituent_key_note(lexicon.company("ecopro_bm")) == ""


def test_absent_company_states_the_absence_as_a_fact(tmp_path, lexicon):
    """The line that makes 0 rows read as 'not in the data' rather than 'is my SQL wrong'."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    note = o._constituent_key_note(lexicon.company("ecopro_hn"))
    assert "0 rows" in note and "the holdings data does not contain this company" in note
    assert "company_code" not in note                # no key is invented


def test_store_without_company_id_degrades_to_previous_behaviour(tmp_path, lexicon):
    """A store without the column gets an empty map, and with an empty map the injected text is as before."""
    o = _orch(_store(tmp_path, _ROWS, with_company_id=False), lexicon)
    assert o.constituent_keys == {}
    assert o._constituent_key_note(lexicon.company("cambricon")) == ""
    assert o._constituent_key_note(lexicon.company("ecopro_hn")) == ""   # no absence is asserted either
    assert o._subsidiary_key_line(lexicon.subsidiaries_of("ecopro")) == ""


def test_subsidiary_keys_appear_when_a_name_is_off(tmp_path, lexicon):
    """'에코프로의 자회사를 편입한 ETF' (ETFs holding 에코프로's subsidiaries) — these subsidiary names are what the model puts in WHERE.

    Of the three, 에코프로머티리얼즈 is spelled differently in the view and 에코프로에이치엔 has no rows.
    If even one is off, all of them are given — giving only some leaves the rest to guesswork again.
    """
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    line = o._subsidiary_key_line(lexicon.subsidiaries_of("ecopro"))
    assert "에코프로비엠 → company_code='247540'" in line
    assert "'450080'" in line and "에코프로머티" in line
    assert "에코프로에이치엔" in line and "0 rows" in line
    assert "\n" not in line


def test_subsidiary_line_is_silent_when_every_name_matches(tmp_path, lexicon):
    """With nothing off, not a single line is added — the injection goes out with every question."""
    o = _orch(_store(tmp_path, _ROWS), lexicon)
    assert o._subsidiary_key_line([lexicon.company("ecopro_bm")]) == ""
    assert o._subsidiary_key_line([]) == ""


# ---------------------------------------------------------------- holder extraction (routing.py)
def test_relation_words_are_not_holders():
    """The measured defect of graph_neighbors('자회사' (subsidiary), constituentOf) -> 0 rows."""
    for word, q in (("자회사", "에코프로의 자회사를 편입한 ETF 중 순자산이 큰 상품의 위험요인 알려줘"),
                    ("계열사", "한화 에어로스페이스의 계열사를 편입한 ETF"),
                    ("관련주", "에코프로 관련주를 담은 ETF"),
                    ("모회사", "에코프로비엠의 모회사를 보유한 ETF")):
        assert word not in routing.holder_mentions(q), q


def test_resolved_company_beats_a_surface_token():
    """Surface extraction looks at one token next to the verb — a resolved company is trusted first."""
    plan = routing.plan_retrieval(routing.Signals(
        question="에코프로의 자회사를 편입한 ETF 중 순자산이 큰 상품의 위험요인 알려줘",
        companies=["에코프로"]))
    assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == ["에코프로"]


def test_surface_fallback_survives_for_companies_not_in_the_lexicon():
    """쿠팡 and 야놀자 are not in the lexicon, yet the answer is 'none', and only the graph knows that."""
    for q, who in (("쿠팡을 편입한 국내 ETF 중 순자산이 가장 큰 3개를 알려줘", "쿠팡"),
                   ("야놀자를 보유한 국내 ETF가 있나요?", "야놀자")):
        plan = routing.plan_retrieval(routing.Signals(question=q))
        assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == [who], q


def test_named_holder_still_wins_over_another_company_in_the_sentence():
    """With two companies in the sentence, the one next to the verb is the target — the order must not flip."""
    plan = routing.plan_retrieval(routing.Signals(
        question="엔비디아 테마 ETF 중 삼성전자를 편입한 상품", companies=["엔비디아", "삼성전자"]))
    assert [c.args["entity"] for c in plan.calls if c.source == "graph"] == ["삼성전자"]
