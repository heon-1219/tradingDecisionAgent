"""ETL: the four Excel product masters -> SQLite store with typed columns, ISO dates, derived
columns and documented views.

Run:  python -m finagent.etl.build_store   (or scripts/build_store.py)
"""
from __future__ import annotations

import csv
import datetime as dt
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from ..kb.lexicon import Lexicon
from .normalize import clean_text, fund_family_key, risk_grade_from_name, to_flag, yyyymmdd_to_iso

# table -> (file stem, source id, korean name, base-date column)
SOURCES: dict[str, tuple[str, str, str, str]] = {
    "bond_kr": ("prbd01n001", "PRBD01N001", "국내채권마스터", "info_base_dt"),
    "etf_kr": ("pref01n001", "PREF01N001", "국내ETF마스터", "du_upt_dt"),
    "etf_gl": ("pref02n001", "PREF02N001", "해외ETF마스터", "du_upt_dt"),
    "fund_pub": ("prfd01n001", "PRFD01N001", "공모펀드마스터", "fd_price_bas_dt"),
}

DATE_COLS: dict[str, list[str]] = {
    "bond_kr": ["info_base_dt", "isu_dt", "mat_dt", "crd_grd_dt", "exg_close_price_base_dt", "pd_std_info_update", "sale_yield_base_dt"],
    "etf_kr": [
        "cu_upt_dt", "du_chas_errt_base_dt", "du_diff_rt_base_dt", "du_nav_base_dt", "du_upt_dt", "du_vlty_base_dt",
        "fn_base_dt", "fn_portfolio_dt", "pd_dvid_base_dt", "pd_dvid_prc_base_dt", "pd_lste_dt", "pd_lstg_dt",
        "ref_base_dt", "wu_upt_dt",
    ],
    "etf_gl": ["cu_upt_dt", "du_clpr_base_dt", "du_nav_base_dt", "du_upt_dt", "pd_lstg_dt", "wu_upt_dt"],
    "fund_pub": ["fd_daily_bas_dt", "fd_last_dstb_actg_bss_dt", "fd_last_dstb_actg_eot_dt", "fd_price_bas_dt"],
}

# numeric-looking codes that must stay text (leading zeros, identifiers)
TEXT_CODE_COLS: dict[str, list[str]] = {
    "bond_kr": ["pd_no", "pd_risk_gcd", "exrt_grte_ern_r_tcd", "info_seq", "curr_cd"],
    "etf_kr": ["pd_itm_no", "pd_itm_no_ma", "pd_ticker", "pd_sale_yn", "pd_sect_cd", "pd_tr_yn", "pd_ric", "pd_isin_cd"],
    "etf_gl": ["pd_itm_no", "pd_itm_no_ma", "pd_lipper_id", "pd_us_cik", "pd_sale_yn", "pd_tr_yn", "pd_exg_mkt_cd", "pd_isin_cd"],
    "fund_pub": [
        "itm_no", "fss_itm_no", "kofia_fd_ccd", "ksd_itm_no", "mtco_itm_no", "or_co_xtn_itt_cd", "trusc_xtn_itt_cd",
        "std_itm_no", "rptt_ksd_itm_no", "zrin_btyp_cd", "zrin_pcd", "zrin_fd_ivst_risk_gcd", "fd_set_pcd",
        "fd_estb_ctry_cd", "pfiv_sale_cntl_tcd", "prfd_attr_cnt", "hdge_fd_yn", "ofsfd_yn", "frc_bpr_itm_yn",
    ],
}


def _read(raw_dir: Path, stem: str, text_cols: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / f"{stem}_data.xlsx", dtype={c: str for c in text_cols})
    for c in text_cols:
        if c in df:
            df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v).removesuffix(".0").strip() or None)
    # blank strings -> NULL everywhere
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "") else v)
    return df


def _derive(table: str, df: pd.DataFrame, lx: Lexicon) -> pd.DataFrame:
    for c in DATE_COLS[table]:
        if c in df:
            df[c] = df[c].map(yyyymmdd_to_iso)
    if table == "bond_kr":
        df["product_type"] = "국내채권"
        # 주최측 공지(2026-08-24): BUYABLE_QUANTITY 값은 무효이며, 상장폐지·리스팅 종료 종목을 제외하면
        # 모두 "구매 가능"으로 간주한다.  채권에서 그 판정은 상환(만기) 도래 여부다.
        df["is_sellable"] = (df["mat_dt"].notna() & (df["mat_dt"] >= df["info_base_dt"])).astype(int)
        # 매수·세후수익률이 실제로 수록된 종목(전체의 약 3%)은 별도 플래그로 남긴다.
        df["has_buy_yield"] = df["buy_yield"].notna().astype(int)
        df["mat_date"], df["isu_date"], df["base_date"] = df["mat_dt"], df["isu_dt"], df["info_base_dt"]
        df["remaining_years"] = (df["remaining_days"] / 365.0).round(2)
        df["issuer_name"] = df["pd_pbcm"].map(clean_text)
        df["pd_nm"] = df["pd_nm"].map(clean_text)
        # 한 종목이 여러 행일 수 있다: 장내/장외 시장 구분 + 장외 판매 LOT(info_seq).
        # 307종목은 LOT 마다 매수수익률·단가가 실제로 다르므로 행을 합치면 안 된다.
        # 대신 종목 수를 세는 질의를 도구 계층에서 잡아준다(agent/tools.py 의 단위 경고).
    elif table == "etf_kr":
        df["product_type"] = df["pd_grp_no"].map(lambda g: "국내ETN" if g == "ETN" else "국내ETF")
        # 상장 종료일이 9999-12-31 이면 상장 유지 중. 거래정지(pd_tr_yn=1)만 제외한다.
        df["is_listed"] = (df["pd_lste_dt"].isna() | (df["pd_lste_dt"] > df["du_upt_dt"])).astype(int)
        df["is_on_sale"] = (df["is_listed"].astype(bool) & (df["pd_tr_yn"].fillna("0") != "1")).astype(int)
        df["manager_name"] = df["cu_fund_mgmt_co"].map(lambda v: (lx.manager_canonical(v) or lx.manager_for_product_name(v) or clean_text(v)) if isinstance(v, str) else None)
        df["listing_date"], df["delisting_date"], df["base_date"] = df["pd_lstg_dt"], df["pd_lste_dt"], df["du_upt_dt"]
        df["risk_grade"] = df["pd_risk_nm"].map(risk_grade_from_name)
        df["pd_abrv_nm"] = df["pd_abrv_nm"].map(clean_text)
        df["pd_nm"] = df["pd_nm"].map(clean_text)
    elif table == "etf_gl":
        df["product_type"] = df["pd_grp_no"].map(lambda g: "해외ETN" if g == "ETN" else "해외ETF")
        df["is_on_sale"] = (df["pd_tr_yn"].fillna("0") != "1").astype(int)
        df["is_listed"] = df["is_on_sale"]
        df["manager_name"] = df["cu_fund_mgmt_co"].map(clean_text)
        df["ticker"] = df["pd_abrv_nm"].map(clean_text)
        df["listing_date"], df["base_date"] = df["pd_lstg_dt"], df["du_upt_dt"]
        df["expense_ratio_pct"] = df["cu_charge_rt"]
        df["pd_nm"] = df["pd_nm"].map(clean_text)
    elif table == "fund_pub":
        df["product_type"] = df["prvo_pbff_desc"].map(lambda v: "공모펀드" if v == "공모" else "사모펀드")
        df["is_on_sale"] = df["sale_yn"].map(to_flag)
        df["itm_nm"] = df["itm_nm"].map(clean_text)
        df["fund_family"] = df["itm_nm"].map(lambda v: fund_family_key(v) if isinstance(v, str) else None)
        # 운용사는 상품명이 아니라 주최측이 준 코드에서 읽는다.  상품명 접두어 매칭은 이름이
        # 운용사 브랜드로 시작할 때만 맞는다: 23,676행 중 17,660행(74.6%)만 풀리고 6,016행이
        # 통째로 NULL 이 된다.  반면 or_co_xtn_itt_cd(운용회사대외기관코드)는 원천이 전 행에
        # 채워 보내주며 275개 코드로 운용사를 구분한다 — 추정할 필요가 없는 값이 이미 있는데
        # 이름을 보고 짐작하고 있었던 것이다.
        # 다만 코드→운용사명 표는 원천에 없으므로 데이터 자체에서 만든다.  이름으로 이미 풀린
        # 행을 코드별로 묶으면 이름이 붙은 코드 89개 중 72개(81%)가 단일 이름으로 수렴하므로,
        # 코드마다 최빈 이름을 그 코드의 정식 명칭으로 삼는다.  동률이면 임의로 고르지 않고
        # 비워 둔다 — 틀린 운용사명은 NULL 보다 나쁘다.  외부 표를 손으로 적지 않고 주최측이
        # 준 값만으로 NULL 6,016행 중 3,938행이 회복된다.
        # 이름으로 풀린 값은 절대 덮어쓰지 않으며, 어느 쪽에서 왔는지는 manager_source 에 남긴다.
        # 파이썬 리스트로 다루는 이유: Series.map 이 돌려준 None 은 pandas 가 str dtype 의
        # NaN 으로 바꾸는데, NaN 은 truthy 라서 `이름 or 코드` 식 폴백이 조용히 전부 '이름'
        # 으로 판정된다.  결측은 반드시 `is None` 으로만 본다.
        by_name = [lx.manager_for_product_name(v) if isinstance(v, str) else None for v in df["itm_nm"]]
        codes = [c if isinstance(c, str) else None for c in df["or_co_xtn_itt_cd"]]
        tally: dict[str, Counter] = defaultdict(Counter)
        for code, name in zip(codes, by_name):
            if code is not None and name is not None:
                tally[code][name] += 1
        code_to_name: dict[str, str] = {}
        for code, cnt in tally.items():
            top = cnt.most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:  # 동률이면 매핑하지 않는다
                code_to_name[code] = top[0][0]
        by_code = [code_to_name.get(c) if c is not None else None for c in codes]
        df["manager_name"] = [n if n is not None else c for n, c in zip(by_name, by_code)]
        df["manager_source"] = [
            "name" if n is not None else ("code" if c is not None else None) for n, c in zip(by_name, by_code)
        ]
        df["base_date"] = df["fd_price_bas_dt"]
        df["risk_grade"] = pd.to_numeric(df["zrin_fd_ivst_risk_gcd"], errors="coerce").astype("Int64")
        df["is_hedged"] = df["exchdg_yn"].map(to_flag)
    return df


VIEWS: dict[str, str] = {
    "v_etf_constituent": (
        "SELECT etf_itm_no, etf_ticker, etf_name, company_code, company_name, weight_pct, is_cash, as_of "
        "FROM etf_constituent"
    ),
    "v_bond_kr": (
        "SELECT pd_no, pd_nm, pd_abrv_nm, pd_eng_nm, product_type, pd_exg_mkt, info_seq, is_sellable, bdbns_abl_chnl_nm, issuer_name, "
        "bd_knd, std_pd_mcls_nm, std_pd_scls_nm, bd_ofr_tcd, bd_inrt_tcd, bd_intp_tcd, crd_grd, crd_grd_dt, pd_risk_nm, "
        "pd_risk_gcd, curr_cd, srfc_irt, isu_date, mat_date, remaining_days, remaining_years, bd_tisu_a, isu_bal_amt, has_buy_yield, "
        "applied_yield, buy_yield, after_tax_yield, pref_tax_yield, avg_annual_tax_yield, corp_pretax_yield, "
        "corp_after_tax_yield, depo_equiv_yield_154, depo_equiv_yield_495, trade_price, buyable_quantity, eval_price, "
        "dirty, dur, cov, exg_close_price, exg_close_yield, exrt_grte_ern_r, exrt_rpy_r, pd_pen_tr_yn, base_date "
        "FROM bond_kr"
    ),
    "v_etf_kr": (
        "SELECT pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, pd_grp_no, is_on_sale, is_listed, pd_sale_yn, pd_tr_yn, "
        "manager_name, cu_fund_mgmt_co, cu_strtegy, cu_lev_fector, cu_base_index, ref_base_index, cu_charge_rt, "
        "cu_charge_etc_rt, wu_inv_ast_type, wu_inv_rgn, ref_ast_type, ref_geo_focus, wu_core_yn, pd_risk_nm, risk_grade, "
        "pd_pen_tr_yn, pd_pen_risk_nm, pd_net_tamt, du_last_aum, du_last_nav, du_clpr, ru_mkt_price, du_bpr, du_diff_rt, "
        "du_chas_errt, du_er_1d, du_er_1m, du_er_3m, du_er_6m, du_er_ytd, du_er_1y, du_vlty_1m, du_vlty_3m, du_vlty_6m, "
        "du_vlty_1y, du_vol_1d, du_val_1d, du_val_1m, pd_dvid_cycl, pd_dvid_yield, pd_divd_amt_pshr, pd_dvid_pay_cnt, "
        "pd_dvid_pay_months, listing_date, delisting_date, pd_lst_stk_cnt, fn_average_coupon, fn_effective_maturity, "
        "base_date FROM etf_kr"
    ),
    "v_etf_gl": (
        "SELECT pd_itm_no, ticker, pd_nm, product_type, pd_grp_no, is_on_sale, is_listed, pd_tr_yn, pd_sale_yn, pd_exg_mkt_cd, pd_mkt_id, pd_curr_cd, "
        "pd_isin_cd, manager_name, cu_base_index, cu_strtegy, expense_ratio_pct, cu_charge_rt, cu_lev_fector, "
        "cu_inverse_short_yn, cu_index_tracking_yn, cu_index_repl_mthd, cu_etn_yn, wu_inv_ast_type, wu_inv_rgn, "
        "wu_core_yn, du_last_aum, du_last_nav, du_clpr, du_opr, du_hpr, du_lpr, ru_mkt_price, du_er_1d, du_vol_1d, "
        "du_val_1d, pd_lst_stk_cnt, listing_date, base_date FROM etf_gl"
    ),
    "v_fund_pub": (
        "SELECT itm_no, itm_nm, itm_abrv_nm, itm_eng_nm, fund_family, product_type, prvo_pbff_desc, is_on_sale, sale_yn, "
        "thco_sale_yn, manager_name, manager_source, or_co_xtn_itt_cd, or_attr_desc, zrin_btyp_nm, zrin_ptn_nm, "
        "zrin_attr_nms, fd_ivst_rgn_desc, "
        "ovrs_fd_desc, bmrk_nm, bmrk_eng_nm, curr_cd, exchdg_yn, is_hedged, hdge_fd_yn, ofsfd_yn, int_dvd_desc, "
        "pers_corp_desc, zrin_fd_ivst_risk_grd_nm, risk_grade, han_clas_nm, han_clas_fee_type, han_clas_sales_channel, "
        "han_clas_policies, or_co_rwrd_r, sale_co_rwrd_r, trusc_rwrd_r, ofwk_trus_rwrd_r, fd_nast_suma, bns_bpr, "
        "fd_wk1_ern_r, fd_mm1_ern_r, fd_mm3_ern_r, fd_mm6_ern_r, fd_yr1_ern_r, fd_mm18_ern_r, fd_yr2_ern_r, fd_yr3_ern_r, "
        "fd_yr5_ern_r, fd_last_dstb_r, zrin_dmst_stk_cmst_rt, zrin_ovrs_stk_cmst_rt, zrin_dmst_bd_cmst_rt, "
        "zrin_ovrs_bd_cmst_rt, zrin_liqt_cmst_rt, zrin_fd_cmst_rt, zrin_etc_ast_cmst_rt, prfd_attr_search_text, "
        "base_date FROM fund_pub"
    ),
}

INDEXES = [
    "bond_kr(pd_no)", "bond_kr(crd_grd)", "bond_kr(is_sellable)", "bond_kr(pd_nm)",
    "etf_kr(pd_itm_no)", "etf_kr(pd_abrv_nm)", "etf_kr(pd_ticker)",
    "etf_gl(pd_itm_no)", "etf_gl(ticker)", "etf_gl(pd_nm)",
    "fund_pub(itm_no)", "fund_pub(fund_family)", "fund_pub(is_on_sale)", "fund_pub(itm_nm)",
    "etf_constituent(etf_itm_no)", "etf_constituent(company_code)", "etf_constituent(company_name)",
]

# Cash and deposit lines are holdings in the source but not companies; flagged rather than dropped so
# the store stays a faithful copy and both SQL and the graph filter on the same column.
_CASH_LIKE = {"원화현금", "설정현금액", "현금", "usd", "달러현금", "외화현금", "원화예금"}


def _load_constituents(con: sqlite3.Connection, path: Path | None, log=print) -> int:
    """ETF holdings into the store, so that 'X를 보유한 ETF 중 순자산 상위 3개' is one SQL join.

    They used to live only in the knowledge graph, which left no way to combine a holding with the
    numbers in the ETF master: the agent tried ``WHERE pd_ticker='005930'`` on the ETF table (the ETF's
    own ticker) and answered "확인할 수 없습니다" while the graph held all 238 rows.  Weights lived as
    ~5,000 distinct ``weight_<company>`` predicates, which is a column pretending to be a vocabulary.
    """
    # Created even when there is no CSV: v_etf_constituent is declared with the other views, and a
    # view over a missing table breaks every query against the store, not just this one.
    con.execute(
        "CREATE TABLE etf_constituent("
        " etf_itm_no TEXT, etf_ticker TEXT, etf_name TEXT, company_code TEXT, company_name TEXT,"
        " company_id TEXT, weight_pct REAL, is_cash INTEGER, as_of TEXT)"
    )
    rows: list[tuple] = []
    for f in sorted(Path(path).glob("*.csv")) if path and Path(path).exists() else []:
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                etf_id = (r.get("etf_id") or "").strip()
                name = (r.get("company_name") or "").strip()
                if not etf_id or not name:
                    continue
                w = (r.get("weight_pct") or "").strip()
                try:
                    weight = float(w) if w else None
                except ValueError:
                    weight = None
                rows.append((
                    etf_id, (r.get("ticker") or "").strip(), (r.get("etf_name") or "").strip(),
                    (r.get("company_code") or "").strip() or None, name,
                    (r.get("company_id") or "").strip() or None, weight,
                    1 if name.replace(" ", "").lower() in _CASH_LIKE else 0,
                    (r.get("as_of") or "").strip() or None,
                ))
    if rows:
        con.executemany("INSERT INTO etf_constituent VALUES (?,?,?,?,?,?,?,?,?)", rows)
    log(f"  etf_constituent: {len(rows):,} rows")
    return len(rows)


def build_store(raw_dir: Path, store_path: Path, lx: Lexicon, log=print,
                constituents_dir: Path | None = None) -> dict[str, int]:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()
    con = sqlite3.connect(store_path)
    con.execute(
        "CREATE TABLE meta(table_name TEXT, source_id TEXT, korean_name TEXT, base_date TEXT, row_count INTEGER, built_at TEXT)"
    )
    counts: dict[str, int] = {}
    for table, (stem, sid, kname, bcol) in SOURCES.items():
        t0 = time.time()
        df = _derive(table, _read(Path(raw_dir), stem, TEXT_CODE_COLS[table]), lx)
        df.to_sql(table, con, index=False)
        base = df[bcol].dropna().mode().iloc[0] if bcol in df and df[bcol].notna().any() else None
        con.execute(
            "INSERT INTO meta VALUES (?,?,?,?,?,?)",
            (table, sid, kname, base, len(df), dt.datetime.now().isoformat(timespec="seconds")),
        )
        counts[table] = len(df)
        log(f"  {table}: {len(df):,} rows (base {base}) in {time.time() - t0:.1f}s")
    n = _load_constituents(con, Path(constituents_dir) if constituents_dir else None, log=log)
    if n:
        counts["etf_constituent"] = n
        as_of = con.execute("SELECT as_of FROM etf_constituent WHERE as_of IS NOT NULL LIMIT 1").fetchone()
        con.execute(
            "INSERT INTO meta VALUES (?,?,?,?,?,?)",
            ("etf_constituent", "naver_finance", "국내ETF편입종목", as_of[0] if as_of else None, n,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
    for name, sql in VIEWS.items():
        con.execute(f"CREATE VIEW {name} AS {sql}")
    for ix in INDEXES:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{ix.replace('(', '_').replace(')', '')} ON {ix}")
    con.commit()
    con.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    from ..config import get_settings

    s = get_settings()
    print(f"Building {s.store_path} from {s.raw_dir} ...")
    counts = build_store(s.raw_dir, s.store_path, Lexicon.load(s.enrichment_dir),
                         constituents_dir=s.constituents_dir)
    print("done:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
