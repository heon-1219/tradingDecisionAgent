import sqlite3
from pathlib import Path

import pytest

from finagent.etl.build_store import build_store
from finagent.kb.lexicon import Lexicon

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_build_store(tmp_path):
    if not (ROOT / "data" / "raw" / "prbd01n001_data.xlsx").exists():
        pytest.skip("raw data not present")
    counts = build_store(ROOT / "data" / "raw", tmp_path / "p.db", Lexicon.load(ROOT / "data" / "enrichment"), log=lambda *a: None)
    assert counts["bond_kr"] == 21882 and counts["fund_pub"] == 23676
    con = sqlite3.connect(tmp_path / "p.db")
    assert con.execute("select count(*) from v_bond_kr where is_sellable=1").fetchone()[0] == 634
    assert con.execute("select count(*) from v_bond_kr where is_sellable=1 and crd_grd in ('AAA','AA+','AA0','AA-')").fetchone()[0] == 78
    assert con.execute("select count(distinct fund_family) from v_fund_pub where itm_nm like '%국민성장%'").fetchone()[0] == 1
    assert con.execute("select base_date from meta where table_name='etf_gl'").fetchone()[0] == "2026-08-22"
    assert con.execute("select manager_name from v_etf_kr where pd_abrv_nm='KODEX 로봇액티브'").fetchone()[0] == "삼성자산운용"
    assert con.execute("select count(*) from v_etf_kr where pd_ticker like '0%'").fetchone()[0] > 0  # leading zeros kept
    assert con.execute("select mat_date from v_bond_kr where pd_no='KR350105G9C6' limit 1").fetchone()[0] == "2029-12-19"
