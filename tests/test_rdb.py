import sqlite3

import pytest

from finagent.kb.rdb import RDB, SQLGuardError


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.execute("create table bond_kr(pd_no text, crd_grd text, buy_yield real)")
    c.executemany("insert into bond_kr values (?,?,?)", [(f"K{i}", "AAA" if i % 2 else "AA-", i) for i in range(500)])
    c.execute("create view v_bond_kr as select * from bond_kr")
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.execute("insert into meta values ('bond_kr','PRBD01N001','국내채권마스터','2026-08-21',500,'x')")
    c.commit()
    c.close()
    return RDB(p, row_limit=100)


def test_select_with_auto_limit(db):
    r = db.query("SELECT pd_no, crd_grd FROM v_bond_kr WHERE crd_grd='AAA'")
    assert r.row_count == 100 and r.truncated and r.columns == ["pd_no", "crd_grd"]
    assert "LIMIT 100" in r.sql


def test_explicit_small_limit_kept(db):
    r = db.query("select pd_no from v_bond_kr limit 5;")
    assert r.row_count == 5 and not r.truncated


def test_code_fence_stripped(db):
    r = db.query("```sql\nselect count(*) as n from v_bond_kr\n```")
    assert r.rows[0]["n"] == 500


def test_rejects_non_select(db):
    for bad in ["DELETE FROM bond_kr", "SELECT 1; DROP TABLE bond_kr", "PRAGMA table_info(bond_kr)", "ATTACH 'x' AS y", ""]:
        with pytest.raises(SQLGuardError):
            db.query(bad)


def test_sql_error_is_guard_error(db):
    with pytest.raises(SQLGuardError):
        db.query("select nope from v_bond_kr")


def test_meta_and_columns(db):
    assert db.meta()["v_bond_kr"]["base_date"] == "2026-08-21"
    assert "crd_grd" in db.columns("v_bond_kr")
