"""Training-data generator for CLOVA Studio tuning.

This is competition-era tooling: it targets CLOVA Studio's tuning-data format and is not used by the Gemini edition.

Key constraint: CLOVA Studio **cannot use tuning and function calling at the same time**
("image input, tuning, function calling, Structured Outputs and reasoning cannot be used together").
So the tool-calling agent loop (HCX-005, function calling) is left as it is, and only the text→text stages that
need no tools are specialised with a tuned model.  This script builds the training sets for those three stages.

  nl2sql  question (+schema)          -> SQL           pure text conversion, no tools
  answer  question + evidence block   -> final answer  learns evidence citation, units, the cannot-confirm style
  intent  question                    -> status + target tables  classifier to prepare for the 5 unanswerable questions

Key idea: no human writes the answers.  Templates are filled with real data values to make the questions, and the
SQL the same template produced is **actually executed**; only pairs that return results are kept (= verified answers).
It is the principle behind `data/eval/groundtruth.jsonl`, run at scale.

  python scripts/build_tuning_data.py --task all --out data/tuning
  python scripts/build_tuning_data.py --task nl2sql --limit 500

Output: CLOVA-format JSONL ({"C_ID","T_ID","Text","Completion","System_Prompt"}) and a 4-column CSV.
Per the docs: UTF-8, at most 8,000 characters per row, at least 100 rows (1,000~100,000 recommended), and at least
200 rows per category for classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from finagent.config import get_settings  # noqa: E402
from finagent.kb.entity import EntityResolver  # noqa: E402
from finagent.kb.lexicon import Lexicon  # noqa: E402
from finagent.kb.rdb import RDB, SQLGuardError  # noqa: E402
from finagent.kb.search import SearchIndex  # noqa: E402

MAX_CHARS = 8000
rng = random.Random(20260904)

SYS_NL2SQL = (
    "너는 미래에셋증권 금융상품 데이터 분석가다. 사용자의 한국어 질문을 SQLite SELECT 문 하나로 변환한다.\n"
    "뷰: v_bond_kr(국내채권 21,882) v_etf_kr(국내ETF/ETN 1,780) v_etf_gl(해외ETF 6,037) v_fund_pub(펀드 23,676)\n"
    "규칙: 구매가능 채권=is_sellable=1(상환 미도래), 매수·세후수익률은 has_buy_yield=1 에만 수록. "
    "국내ETF 판매중=is_on_sale=1, ETF만=pd_grp_no='ETF'. 공모펀드=product_type='공모펀드'. "
    "NULL 은 0 이 아니라 미수록이므로 정렬·집계에서 IS NOT NULL 로 제외한다. "
    "같은 채권이 장내/장외로 중복되므로 DISTINCT 를 쓴다. SQL 만 출력한다."
)
SYS_ANSWER = (
    "너는 미래에셋증권 금융상품 Agent 의 답변 작성자다. 주어진 근거만으로 한국어 답변을 쓴다.\n"
    "규칙: 근거에 없는 수치·상품명을 만들지 않는다. 답변 끝에 참조 테이블 한글명과 데이터 기준일을 밝힌다. "
    "금액은 원/달러, 비율은 % 단위를 붙인다. 값이 없으면 0 이 아니라 '제공되지 않습니다'로 쓴다. "
    "수익률 전망이나 단정적 투자 추천은 하지 않는다. 근거가 없으면 '제공된 데이터로는 확인할 수 없습니다'로 시작한다."
)
SYS_INTENT = (
    "너는 금융상품 질의 분류기다. 질문을 읽고 아래 형식 한 줄로만 답한다.\n"
    "status=answered|cannot_confirm|clarify; tables=v_bond_kr,v_etf_kr,v_etf_gl,v_fund_pub 중 해당하는 것(없으면 none); reason=사유 한 구절\n"
    "제공 데이터는 국내채권·국내ETF·해외ETF·공모펀드 마스터 4종(기준일 2026-08-21/22)이다. "
    "존재하지 않는 값·상품, 데이터 범위 밖 정보, 미래 전망 요구는 cannot_confirm, 조건이 모호하면 clarify 이다."
)


@dataclass
class Row:
    system: str
    text: str
    completion: str


# ------------------------------------------------------------------ helpers
def vals(rdb: RDB, sql: str) -> list:
    try:
        return [r[0] for r in rdb.fetch_all(sql) if r[0] not in (None, "")]
    except SQLGuardError:
        return []


def runs(rdb: RDB, sql: str, need_rows: bool = True) -> list | None:
    """Run the SQL and return its rows if there are any.  This is the 'answer verification' step for training pairs."""
    try:
        rows = rdb.fetch_all(sql if " limit " in sql.lower() else sql + " LIMIT 50")
    except Exception:  # noqa: BLE001
        return None
    if need_rows and not rows:
        return None
    if rows and rows[0][0] in (None, 0) and len(rows) == 1 and len(rows[0]) == 1:
        return None  # a COUNT of 0 has no training value
    return rows


_ENDINGS = [
    ("알려줘", ["알려주세요", "정리해줘", "찾아줘", "말해줘"]),
    ("알려주세요", ["알려줘", "정리해 주세요", "알려주실 수 있나요?"]),
    ("몇 개야?", ["몇 개인가요?", "개수를 알려줘", "총 몇 종목이야?", "몇 건이지?"]),
    ("개수를 알려줘", ["몇 개인가요?", "총 몇 개야?", "건수를 알려주세요"]),
    ("는?", ["는 무엇인가요?", "를 알려줘", "가 궁금해"]),
    ("은?", ["은 무엇인가요?", "을 알려줘", "이 궁금해"]),
    ("비교해줘", ["비교해 주세요", "비교해서 정리해줘"]),
]


def variants(q: str, k: int = 3) -> list[str]:
    """Phrasing variants of one question.  Evaluation queries will be worded differently from our templates, so add variety."""
    out = [q]
    for tail, alts in _ENDINGS:
        if q.endswith(tail):
            out += [q[: -len(tail)] + a for a in alts]
            break
    return out[:k]


PCT_COLS = {"srfc_irt", "buy_yield", "after_tax_yield", "applied_yield", "pref_tax_yield", "corp_pretax_yield",
            "corp_after_tax_yield", "du_er_1d", "du_er_1m", "du_er_3m", "du_er_6m", "du_er_ytd", "du_er_1y",
            "du_vlty_1m", "du_vlty_3m", "du_vlty_6m", "du_vlty_1y", "du_diff_rt", "du_chas_errt", "cu_charge_rt",
            "expense_ratio_pct", "pd_dvid_yield", "fd_yr1_ern_r", "fd_yr2_ern_r", "fd_yr3_ern_r", "fd_yr5_ern_r",
            "fd_mm1_ern_r", "fd_mm3_ern_r", "fd_mm6_ern_r", "수익률1년", "exrt_rpy_r"}
KRW_COLS = {"pd_net_tamt", "du_last_aum", "fd_nast_suma", "bd_tisu_a", "isu_bal_amt", "du_val_1d", "du_val_1m"}
USD_COLS = {"du_last_aum_usd"}
NAME_COLS = ("pd_abrv_nm", "pd_nm", "itm_nm", "상품명", "ticker", "pd_no", "pd_ticker", "itm_no", "pd_itm_no")


def fmt_cell(col: str, v) -> str | None:
    """Render the value in the unit its column name calls for.  A wrong unit contaminates the training label."""
    if v is None:
        return None
    if col in PCT_COLS:
        return f"{float(v):.2f}%".replace(".00%", "%")
    if col in KRW_COLS:
        return money(v)
    if col == "du_last_aum" or col.endswith("_aum"):
        return money(v)
    if col == "remaining_days":
        return f"잔존 {int(v):,}일"
    if col in ("remaining_years",):
        return f"잔존 {float(v):.1f}년"
    if col in ("mat_date", "listing_date", "isu_date"):
        return str(v)
    if col in ("crd_grd", "pd_risk_nm", "manager_name", "pd_dvid_cycl", "구분"):
        return str(v)
    if col in ("pd_lst_stk_cnt", "du_vol_1d"):
        return f"{int(v):,}주"
    if isinstance(v, (int, float)):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def money(v) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "제공되지 않음"
    if n >= 1e12:
        return f"{n / 1e12:.2f}조원"
    if n >= 1e8:
        return f"{n / 1e8:.0f}억원"
    return f"{n:,.0f}원"


# ------------------------------------------------------------------ nl2sql
def gen_nl2sql(rdb: RDB, limit: int) -> list[Row]:
    managers = vals(rdb, "select manager_name from v_etf_kr where is_on_sale=1 group by 1 having count(*)>=5")
    regions = vals(rdb, "select wu_inv_rgn from v_etf_kr group by 1 having count(*)>=8")
    assets = vals(rdb, "select wu_inv_ast_type from v_etf_kr group by 1 having count(*)>=8")
    kinds = vals(rdb, "select trim(bd_knd) from v_bond_kr where is_sellable=1 group by 1 having count(*)>=50")
    issuers = vals(rdb, "select issuer_name from v_bond_kr where is_sellable=1 group by 1 having count(*)>=30")
    attrs = vals(rdb, "select or_attr_desc from v_fund_pub where product_type='공모펀드' and is_on_sale=1 group by 1 having count(*)>=50")
    gl_mgrs = vals(rdb, "select manager_name from v_etf_gl group by 1 having count(*)>=30")
    grades = ["AAA", "AA+", "AA0", "AA-", "A+", "A0", "A-", "BBB+"]
    grade_sets = {"AAA": "'AAA'", "AA+": "'AAA','AA+'", "AA0": "'AAA','AA+','AA0'", "AA-": "'AAA','AA+','AA0','AA-'",
                  "A+": "'AAA','AA+','AA0','AA-','A+'", "A0": "'AAA','AA+','AA0','AA-','A+','A0'",
                  "A-": "'AAA','AA+','AA0','AA-','A+','A0','A-'", "BBB+": "'AAA','AA+','AA0','AA-','A+','A0','A-','BBB+'"}
    ks = [3, 5, 10, 20]

    fams: list[tuple[list[str], str]] = []

    for g in grades:
        s = grade_sets[g]
        fams.append(([f"구매 가능한 원화채권 중 신용등급 {g} 이상인 종목은 몇 개야?",
                      f"신용등급 {g} 이상 원화채권 중 구매 가능한 종목 수를 알려줘",
                      f"현재 살 수 있는 {g} 이상 등급 원화채권 개수는?"],
                     f"SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ({s})"))
    for g in grades[:4]:
        for k in ks:
            fams.append(([f"구매 가능한 신용등급 {g} 이상 채권 중 표면금리가 높은 {k}개를 알려줘",
                          f"{g} 이상 등급 채권 중 표면이자율 상위 {k}종목은?"],
                         f"SELECT DISTINCT pd_no, pd_nm, crd_grd, srfc_irt, mat_date FROM v_bond_kr WHERE is_sellable=1 "
                         f"AND crd_grd IN ({grade_sets[g]}) AND srfc_irt IS NOT NULL ORDER BY srfc_irt DESC LIMIT {k}"))
    for y in ("2026", "2027", "2028", "2029", "2030"):
        fams.append(([f"만기가 {y}년인 구매 가능한 채권은 몇 개야?", f"{y}년에 만기가 도래하는 채권 수를 알려줘"],
                     f"SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1 AND mat_date LIKE '{y}%'"))
    for kind in kinds[:20]:
        fams.append(([f"구매 가능한 {kind} 중 잔존만기가 짧은 5종목을 알려줘", f"{kind} 가운데 만기가 가장 가까운 5개는?"],
                     f"SELECT DISTINCT pd_no, pd_nm, mat_date, remaining_days FROM v_bond_kr WHERE is_sellable=1 "
                     f"AND trim(bd_knd)='{kind}' AND remaining_days IS NOT NULL ORDER BY remaining_days ASC LIMIT 5"))
    for iss in issuers[:20]:
        fams.append(([f"{iss}가 발행한 구매 가능 채권은 몇 개야?", f"{iss} 발행 채권 중 살 수 있는 종목 수는?"],
                     f"SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1 AND issuer_name='{iss}'"))
    fams.append((["매수수익률이 수록된 채권 중 세후 운용수익률이 가장 높은 5종목을 알려줘",
                  "세후수익률 상위 채권 5개를 알려줘"],
                 "SELECT DISTINCT pd_no, pd_nm, crd_grd, buy_yield, after_tax_yield FROM v_bond_kr WHERE has_buy_yield=1 "
                 "AND after_tax_yield IS NOT NULL ORDER BY after_tax_yield DESC LIMIT 5"))

    for m in managers[:30]:
        fams.append(([f"{m}이 운용하는 국내 상장 ETF는 몇 개야?", f"{m} 국내 ETF 개수를 알려줘"],
                     f"SELECT COUNT(*) FROM v_etf_kr WHERE manager_name='{m}' AND pd_grp_no='ETF' AND is_on_sale=1"))
        for k in ks:
            fams.append(([f"{m}의 국내 ETF 중 순자산이 큰 {k}개를 알려줘", f"{m} ETF 순자산 상위 {k}종목은?"],
                         f"SELECT pd_ticker, pd_abrv_nm, pd_net_tamt FROM v_etf_kr WHERE manager_name='{m}' AND pd_grp_no='ETF' "
                         f"AND is_on_sale=1 AND pd_net_tamt IS NOT NULL ORDER BY pd_net_tamt DESC LIMIT {k}"))
    for rgn in regions:
        for k in ks:
            fams.append(([f"{rgn}에 투자하는 국내 ETF 중 1년 수익률 상위 {k}개를 알려줘",
                          f"{rgn} 투자 ETF 최근 1년 수익률 {k}위까지 알려줘"],
                         f"SELECT pd_abrv_nm, manager_name, du_er_1y FROM v_etf_kr WHERE wu_inv_rgn='{rgn}' AND is_on_sale=1 "
                         f"AND pd_grp_no='ETF' AND du_er_1y IS NOT NULL ORDER BY du_er_1y DESC LIMIT {k}"))
        fams.append(([f"{rgn}에 투자하는 국내 ETF는 몇 개야?"],
                     f"SELECT COUNT(*) FROM v_etf_kr WHERE wu_inv_rgn='{rgn}' AND pd_grp_no='ETF' AND is_on_sale=1"))
    for a in assets:
        fams.append(([f"{a} 자산군 국내 ETF 중 순자산 상위 5개를 알려줘"],
                     f"SELECT pd_abrv_nm, manager_name, pd_net_tamt FROM v_etf_kr WHERE wu_inv_ast_type='{a}' AND is_on_sale=1 "
                     f"AND pd_net_tamt IS NOT NULL ORDER BY pd_net_tamt DESC LIMIT 5"))
    for g in (1, 2, 3, 4, 5, 6):
        fams.append(([f"위험등급 {g}등급인 국내 ETF는 몇 개야?", f"{g}등급 위험도 ETF 개수를 알려줘"],
                     f"SELECT COUNT(*) FROM v_etf_kr WHERE risk_grade={g} AND is_on_sale=1"))
    fams.append((["월배당 국내 ETF 중 분배수익률이 높은 5개를 알려줘", "매달 분배금을 주는 ETF 중 분배수익률 상위 5종목은?"],
                 "SELECT pd_abrv_nm, pd_dvid_yield, pd_dvid_cycl FROM v_etf_kr WHERE pd_dvid_cycl='M' AND is_on_sale=1 "
                 "AND pd_dvid_yield IS NOT NULL ORDER BY pd_dvid_yield DESC LIMIT 5"))

    for k in ks:
        fams.append(([f"미국 증시에 상장된 주식형 ETF 중 총보수가 낮은 {k}개를 알려줘",
                      f"해외 주식형 ETF 가운데 보수가 저렴한 {k}종목은?"],
                     f"SELECT ticker, pd_nm, expense_ratio_pct, du_last_aum FROM v_etf_gl WHERE wu_inv_ast_type='Equity' "
                     f"AND is_on_sale=1 AND expense_ratio_pct IS NOT NULL AND expense_ratio_pct>0 "
                     f"ORDER BY expense_ratio_pct ASC, du_last_aum DESC LIMIT {k}"))
        fams.append(([f"해외 ETF 중 운용규모가 큰 {k}개를 알려줘", f"AUM 기준 해외 ETF 상위 {k}종목은?"],
                     f"SELECT ticker, pd_nm, du_last_aum FROM v_etf_gl WHERE du_last_aum IS NOT NULL AND is_on_sale=1 "
                     f"ORDER BY du_last_aum DESC LIMIT {k}"))
    for m in gl_mgrs[:15]:
        fams.append(([f"{m}가 운용하는 해외 ETF 중 운용규모 상위 5개를 알려줘"],
                     f"SELECT ticker, pd_nm, du_last_aum FROM v_etf_gl WHERE manager_name='{m}' AND du_last_aum IS NOT NULL "
                     f"ORDER BY du_last_aum DESC LIMIT 5"))
    fams.append((["레버리지나 인버스가 아닌 해외 ETF 중 운용규모 상위 5개를 알려줘"],
                 "SELECT ticker, pd_nm, du_last_aum FROM v_etf_gl WHERE cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL "
                 "AND du_last_aum IS NOT NULL AND is_on_sale=1 ORDER BY du_last_aum DESC LIMIT 5"))

    for a in attrs:
        for k in ks:
            fams.append(([f"판매중인 공모 {a} 펀드 중 1년 수익률 상위 {k}개를 알려줘",
                          f"{a} 공모펀드 최근 1년 수익률 {k}위까지 알려줘"],
                         f"SELECT itm_nm, fd_yr1_ern_r, fd_nast_suma FROM v_fund_pub WHERE product_type='공모펀드' AND is_on_sale=1 "
                         f"AND or_attr_desc='{a}' AND fd_yr1_ern_r IS NOT NULL ORDER BY fd_yr1_ern_r DESC LIMIT {k}"))
        fams.append(([f"판매중인 공모 {a} 펀드는 몇 개야?"],
                     f"SELECT COUNT(*) FROM v_fund_pub WHERE product_type='공모펀드' AND is_on_sale=1 AND or_attr_desc='{a}'"))
    fams.append((["환헤지형 해외 채권형 공모펀드 중 판매중인 상품은 몇 개야?"],
                 "SELECT COUNT(*) FROM v_fund_pub WHERE is_hedged=1 AND or_attr_desc='채권형' AND ovrs_fd_desc='해외' "
                 "AND is_on_sale=1 AND product_type='공모펀드'"))

    for k in (3, 5, 10):  # cross-table queries — a type the organizers announced
        fams.append(([f"국내 ETF와 공모펀드를 합쳐서 1년 수익률 상위 {k}개를 알려줘",
                      f"ETF와 펀드를 같이 놓고 최근 1년 수익률 {k}위까지 비교해줘"],
                     "SELECT 구분, 상품명, 수익률1년 FROM ("
                     "SELECT '국내ETF' AS 구분, pd_abrv_nm AS 상품명, du_er_1y AS 수익률1년 FROM v_etf_kr "
                     "WHERE is_on_sale=1 AND pd_grp_no='ETF' AND du_er_1y IS NOT NULL "
                     "UNION ALL SELECT '공모펀드', itm_nm, fd_yr1_ern_r FROM v_fund_pub "
                     "WHERE product_type='공모펀드' AND is_on_sale=1 AND fd_yr1_ern_r IS NOT NULL"
                     f") ORDER BY 수익률1년 DESC LIMIT {k}"))

    rows: list[Row] = []
    for phrasings, sql in fams:
        if runs(rdb, sql) is None:
            continue
        for q in phrasings:
            for v in variants(q):
                rows.append(Row(SYS_NL2SQL, v, sql))
                if len(rows) >= limit:
                    return rows
    return rows


# ------------------------------------------------------------------ answer
def gen_answer(rdb: RDB, nl2sql_rows: list[Row], limit: int) -> list[Row]:
    meta = rdb.meta()
    rows: list[Row] = []
    seen: dict[str, int] = {}
    for r in nl2sql_rows:
        # attach up to 3 differently worded questions to the same answer (SQL) to train robustness to phrasing
        if seen.get(r.completion, 0) >= 3:
            continue
        try:
            res = rdb.query(r.completion)
        except Exception:  # noqa: BLE001
            continue
        if not res.rows:
            continue
        seen[r.completion] = seen.get(r.completion, 0) + 1
        view = next((v for v in ("v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub") if v in r.completion), "v_etf_kr")
        m = meta.get(view, {})
        kname, base = m.get("korean_name", view), m.get("base_date", "2026-08-21")
        shown = res.rows[:5]
        evidence = (f"[근거 1] {kname}({view}) · 기준일 {base} · {res.row_count}건\n"
                    f"  sql: {res.sql}\n"
                    f"  결과: {json.dumps(shown, ensure_ascii=False, default=str)}")
        first_col = res.columns[0]
        if first_col.upper().startswith("COUNT") or len(res.columns) == 1 and res.row_count == 1:
            n = list(shown[0].values())[0]
            answer = f"조건에 해당하는 상품은 총 {int(n):,}건입니다. (근거: {kname} · 기준일 {base})"
        else:
            name_col = next((c for c in NAME_COLS if c in res.columns), res.columns[0])
            lines = []
            for i, row in enumerate(shown, 1):
                parts = [fmt_cell(c, row[c]) for c in res.columns if c != name_col]
                parts = [p for p in parts if p]
                lines.append(f"{i}. {row[name_col]}" + (f" — {' · '.join(parts)}" if parts else ""))
            answer = "\n".join(lines) + f"\n\n(근거: {kname} · 기준일 {base} · 총 {res.row_count}건)"
        text = f"질문: {r.text}\n\n근거:\n{evidence}"
        if len(text) + len(answer) + len(SYS_ANSWER) <= MAX_CHARS:
            rows.append(Row(SYS_ANSWER, text, answer))
        if len(rows) >= limit:
            break

    # also train the cannot-confirm style (a behaviour that is scored)
    declines = [
        ("신용등급 AAAA인 채권을 찾아줘", "근거: 없음 (온톨로지 검증 결과 'AAAA'는 존재하지 않는 신용등급)",
         "제공된 데이터로는 확인할 수 없습니다. 신용등급 'AAAA'는 존재하지 않는 값입니다. "
         "유효한 신용등급은 AAA, AA+, AA0, AA-, A+, A0, A-, BBB+ … D 입니다."),
        ("위험등급 8등급 ETF를 알려줘", "근거: 없음 (위험등급은 1~6등급만 존재)",
         "제공된 데이터로는 확인할 수 없습니다. 상품 위험등급은 1등급(매우 높은 위험)부터 6등급(매우 낮은 위험)까지만 존재합니다."),
        ("삼성전자를 보유한 공모펀드를 알려줘", "근거: 없음 (공모펀드 마스터에 편입종목 정보 없음)",
         "제공된 데이터로는 확인할 수 없습니다. 공모펀드 마스터에는 편입종목(구성종목) 정보가 포함되어 있지 않습니다. "
         "국내 ETF의 편입종목은 확인할 수 있으니 필요하시면 말씀해 주세요."),
        ("TIGER 반도체 ETF의 내년 예상 수익률을 알려줘", "근거: 없음 (과거 수익률만 수록, 전망 데이터 없음)",
         "제공된 데이터로는 확인할 수 없습니다. 제공 데이터에는 기준일 시점의 과거 수익률만 있고 미래 전망은 포함되지 않습니다. "
         "과거 1년 수익률이 필요하시면 알려드릴 수 있습니다."),
        ("미래에셋증권 직원 수를 알려줘", "근거: 없음 (상품 마스터 범위 밖)",
         "제공된 데이터로는 확인할 수 없습니다. 제공 데이터는 국내채권·국내ETF·해외ETF·공모펀드 상품 마스터로, 회사 일반 정보는 포함하지 않습니다."),
        ("KODEX AI로봇 ETF 정보를 알려줘", "근거: 상품명 조회 결과 정확 일치 없음 (유사: KODEX 로봇액티브, KODEX 미국휴머노이드로봇)",
         "'KODEX AI로봇'과 정확히 일치하는 상품은 제공된 데이터에서 확인할 수 없습니다. "
         "이름이 비슷한 상품으로는 KODEX 로봇액티브, KODEX 미국휴머노이드로봇 등이 있습니다. 이 중 찾으시는 상품이 있나요?"),
    ]
    for q, ev, ans in declines:
        for _ in range(6):  # the cannot-confirm style needs repeated exposure
            rows.append(Row(SYS_ANSWER, f"질문: {q}\n\n근거:\n{ev}", ans))
    return rows


# ------------------------------------------------------------------ intent
def gen_intent(rdb: RDB, nl2sql_rows: list[Row], lexicon: Lexicon, resolver: EntityResolver, limit: int) -> list[Row]:
    rows: list[Row] = []
    for r in nl2sql_rows:
        view = next((v for v in ("v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub") if v in r.completion), "v_etf_kr")
        tables = "v_etf_kr,v_fund_pub" if "UNION ALL" in r.completion else view
        rows.append(Row(SYS_INTENT, r.text, f"status=answered; tables={tables}; reason=제공 데이터로 조회 가능"))
        if len(rows) >= int(limit * 0.6):
            break

    # Nonexistent product names: swap the theme token of a real product name, then check against the index that it really doesn't exist
    real = [n for n in vals(rdb, "select pd_abrv_nm from v_etf_kr where is_on_sale=1 and pd_abrv_nm is not null") if " " in n]
    swaps = ["양자컴퓨팅", "우주광물", "합성생물학", "핵융합", "수소항공", "심해채굴", "뇌공학", "위성인터넷"]
    made = 0
    for name in rng.sample(real, min(len(real), 400)):
        brand, rest = name.split(" ", 1)
        fake = f"{brand} {rng.choice(swaps)}{rng.choice(['', '액티브', 'TOP10', '커버드콜'])}"
        if resolver.resolve(fake, k=3).resolved:
            continue
        rows.append(Row(SYS_INTENT, f"{fake} ETF 정보를 알려줘",
                        "status=cannot_confirm; tables=none; reason=해당 이름의 상품이 데이터에 존재하지 않음"))
        made += 1
        if made >= 120:
            break

    fixed = [
        ("신용등급 AAAA인 채권 찾아줘", "cannot_confirm", "none", "존재하지 않는 신용등급"),
        ("신용등급 AAAAA 채권 알려줘", "cannot_confirm", "none", "존재하지 않는 신용등급"),
        ("위험등급 7등급 펀드 알려줘", "cannot_confirm", "none", "위험등급은 1~6등급"),
        ("위험등급 9등급 ETF 알려줘", "cannot_confirm", "none", "위험등급은 1~6등급"),
        ("Kimi 관련 투자상품 있어?", "cannot_confirm", "none", "비상장 기업으로 관련 상품 없음"),
        ("삼성전자 현재 주가 알려줘", "cannot_confirm", "none", "개별 주식 시세는 데이터 범위 밖"),
        ("코스피 지수 오늘 얼마야?", "cannot_confirm", "none", "지수 시세는 데이터 범위 밖"),
        ("미래에셋증권 직원 수는?", "cannot_confirm", "none", "회사 일반 정보는 범위 밖"),
        ("내년 금리 전망 알려줘", "cannot_confirm", "none", "전망 데이터 없음"),
        ("TIGER 반도체 ETF 내년 수익률은?", "cannot_confirm", "none", "미래 전망은 생성 금지"),
        ("삼성전자를 보유한 공모펀드 알려줘", "cannot_confirm", "v_fund_pub", "펀드 편입종목 데이터 없음"),
        ("삼성전자를 편입한 해외 ETF 알려줘", "cannot_confirm", "v_etf_gl", "해외ETF 편입종목 데이터 없음"),
        ("제일 좋은 펀드 하나만 골라줘", "clarify", "v_fund_pub", "기준이 모호함(수익률·순자산·위험 중 무엇인지)"),
        ("좋은 ETF 추천해줘", "clarify", "v_etf_kr", "투자 지역·자산군·위험 성향 조건이 없음"),
        ("적당한 채권 알려줘", "clarify", "v_bond_kr", "등급·만기·통화 조건이 없음"),
        ("괜찮은 상품 있어?", "clarify", "none", "상품군과 조건이 모두 불명확"),
        ("수익률 높은 거 알려줘", "clarify", "none", "상품군과 기간이 불명확"),
        ("안전한 상품 추천해줘", "clarify", "none", "'안전'의 기준(위험등급·자산군)이 불명확"),
        ("요즘 인기 있는 ETF 알려줘", "clarify", "v_etf_kr", "'인기'의 기준(거래량·순자산·수익률)이 불명확"),
        ("배당 잘 주는 상품 있어?", "clarify", "none", "상품군과 분배 기준이 불명확"),
        ("돈 많이 버는 펀드 알려줘", "clarify", "v_fund_pub", "기간과 수익률 기준이 불명확"),
        ("초보자한테 맞는 상품 뭐야?", "clarify", "none", "투자 성향·기간·금액 조건이 없음"),
        ("싸고 좋은 ETF 알려줘", "clarify", "v_etf_kr", "'싸다'의 기준(보수·가격)과 '좋다'의 기준이 불명확"),
        ("큰 거 위주로 알려줘", "clarify", "none", "상품군과 '크다'의 기준이 불명확"),
        ("최근에 잘나가는 테마 뭐야?", "clarify", "v_etf_kr", "기간과 판단 기준이 불명확"),
        ("장기투자용 상품 골라줘", "clarify", "none", "기간·자산군·위험 조건이 없음"),
        ("연금으로 살 만한 거 있어?", "clarify", "none", "상품군과 위험 성향 조건이 불명확"),
    ]
    for q, st, tb, why in fixed:
        for _ in range(13):  # classification: 200+ rows per category recommended
            rows.append(Row(SYS_INTENT, q, f"status={st}; tables={tb}; reason={why}"))
    return rows


# ------------------------------------------------------------------ output
def write(rows: list[Row], out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    kept = [r for r in rows if len(r.system) + len(r.text) + len(r.completion) <= MAX_CHARS]
    jsonl = out_dir / f"{name}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            f.write(json.dumps({"C_ID": i, "T_ID": 0, "Text": r.text, "Completion": r.completion,
                                "System_Prompt": r.system}, ensure_ascii=False) + "\n")
    with open(out_dir / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["C_ID", "T_ID", "Text", "Completion"])
        for i, r in enumerate(kept):
            w.writerow([i, 0, r.text, r.completion])
    dropped = len(rows) - len(kept)
    print(f"  {name:8} {len(kept):>6} rows → {jsonl.name}, {name}.csv" + (f"  ({dropped} rows over 8,000 chars dropped)" if dropped else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "nl2sql", "answer", "intent"])
    ap.add_argument("--out", default=str(ROOT / "data/tuning"))
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()

    s = get_settings()
    rdb = RDB(s.store_path)
    out = pathlib.Path(a.out)
    print(f"Generating training data (verification: every SQL is actually executed; only pairs with results are kept)")

    nl = gen_nl2sql(rdb, a.limit)
    if a.task in ("all", "nl2sql"):
        write(nl, out, "nl2sql")
    if a.task in ("all", "answer"):
        write(gen_answer(rdb, nl, a.limit // 2), out, "answer")
    if a.task in ("all", "intent"):
        lx = Lexicon.load(s.enrichment_dir)
        resolver = EntityResolver(SearchIndex.from_rdb(rdb), lx)
        write(gen_intent(rdb, nl, lx, resolver, a.limit), out, "intent")
    print(f"\nCLOVA Studio upload spec: UTF-8, at most 8,000 chars per row, at least 100 rows / 1,000~100,000 recommended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
