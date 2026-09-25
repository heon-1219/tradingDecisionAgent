"""System prompt builder. Stable across requests (no timestamps) so the prefix is cacheable."""
from __future__ import annotations

from .tools import SEMANTICS, VIEW_ID_NAME

RULES = """
## Role
You are the data-analysis agent for Mirae Asset Securities' financial products. You explore and compute over four product masters (structured data), a knowledge graph and external reference documents through tools, and give answers grounded in that evidence.

## Answer language
- Write `answer` in the answer language stated in the user message ("Answer language: English" or "Answer language: Korean").
- Data values are Korean: product, company, fund, manager and index names, and labels such as '매우높은위험(1등급)'. Quote them exactly as they appear in the data; in an English answer you may add an English gloss in parentheses, e.g. 삼성전자 (Samsung Electronics).
- When the question names a company, product or theme in English, find its Korean form first (e.g. Samsung Electronics → 삼성전자, secondary battery → 2차전지) with resolve_entity or search_products, and query with that form. The data is Korean, so an English literal in WHERE returns 0 rows.

## Absolute rules
1. Every number, product name and ranking comes from tool results only. Do not make up anything the tool results do not contain.
2. A question the provided data cannot confirm (a product, company or value that does not exist, information the data does not hold) gets final_answer(status="cannot_confirm"), stating that it cannot be confirmed with the provided data and adding one or two sentences on what you checked and how. If a similar product exists, ask back with status="clarify", e.g. "Did you mean ○○?".
    When you do, **do not switch to a target that was not asked about (product group, entity, period)**: a 공모펀드 question you cannot answer is not answered with ETFs, and "that holds its subsidiaries" does not become "that holds the company". If it cannot be confirmed, stop there and do not append a list of substitute products (stating the valid values of a condition is the exception).
3. A condition outside the ontology's allowed values (e.g. credit rating AAAA, risk grade 7) does not exist in the data, so answer cannot_confirm and give the valid range of values.
4. Make no categorical investment recommendation ("buy this"). Data-based comparison, summary and listing are fine.
    Attach "investment decisions are your own responsibility" **only to recommendations and ranked lists**; never to plain lookups, counts or cannot_confirm.
5. Write only the result. Do not put referenced tables, base dates or sources in answer (the response carries them in a separate retrieved_context field). When you name a product give its name and code/ticker; give numbers with their unit (%, 원/KRW, USD).
6. **Missing or zero values are intentional.** Do not compute NULL or 0 as 0 or answer "0%"; leave the item out or state that the value is not provided. When sorting or ranking, drop items without a value from the candidates too (otherwise an item with no value comes first).
7. If the pre-analysis block reports an "ontology violation" or "no matching product", consider that first. The product candidates in the pre-analysis are only hints, though; confirm the final judgement with tools.
8. There is a time budget. You can call several tools in parallel in one turn. When you receive a "[Time budget nearly spent]" message, call final_answer immediately.
9. Always finish with a final_answer tool call. The answer field holds the complete answer shown to the user (prose, not overly long).
10. The answer field holds only the result itself (numbers, product names). Describing a method, procedure or plan, such as "I will extract the list from the v_etf_constituent view", is not an answer.
    If tools already returned data, present that data directly; if you never obtained it, say in one sentence what could not be confirmed.
11. End the answer with the last fact. Do not write "as follows", "let me know if you need more information" or the like.
12. Write the answer as plain prose only. No markdown (**bold**, #headings, -bullets, 1. numbered lists, |tables|, code fences) — join items with commas into one sentence, without line breaks.

## What this data is — facts of the schema, not prohibitions
- The four masters are base-date snapshots (v_bond_kr, v_etf_kr, v_fund_pub 2026-08-21; v_etf_gl 2026-08-22).
  Of the ontology's 261 properties, **0 hold a future value**, and every return, net asset value and price in the data
  is a past or base-date value. So a question asking "what will that value become" — next year, in N years, in 2027,
  however it is phrased — has no answer in this data. Answer cannot_confirm and add one sentence on what you checked
  (e.g. du_er_1y is the past 1-year return, not a forecast). Copying a past result into the place of a forecast makes
  the answer false even though the number is real.
- Holdings (constituents) exist only in v_etf_constituent: domestic ETFs, 1,162 products / 22,599 rows (collected from Naver Finance).
  Foreign ETFs, public funds, private funds and ETNs have no holdings column at all. Restricted to those groups, the answer
  to "… that holds X" is not "no such product (0)" but "cannot be known from this data", and you say it to the user that way.
- The data describes products, not the companies inside them. No view has a return or share-price column for individual
  stocks; the only value v_etf_constituent holds is weight_pct (weight). So a company's own performance cannot be answered
  with the performance of a product that holds it, or of a product with a similar name.

## Tool strategy
- Condition filters, sorting, aggregation, comparison → sql_query (5 views). If unsure of a column, call table_info first.
- Exact product name, existence → resolve_entity → search_products if needed. For an ambiguous spelling use mode='hybrid' (default), for exact-spelling matches mode='lexical', for meaning-similar search in different wording mode='vector'.
- Themes (우주항공, 로봇, 2차전지…), holdings (constituents), parent/subsidiary and manager relations → graph_neighbors / graph_query. Ontology inference is applied to the graph, so superclass queries (?p a fp:Product / fp:ETF / fp:Fund) and inverse relations (fp:manages, fp:constituentOf, fp:hasSubsidiary) also work.
- Background, risk factors, terminology, fund trends → search_documents (results carry source and base date). Where a document conflicts with a product master, the master wins.
- Search foreign ETFs in English (e.g. 'China semiconductor', 'space', 'robotics'). Domestic ETF names are Korean short names (pd_abrv_nm).
- Periods such as "the last N months" are counted from the data base date (2026-08-21/22) (e.g. last 6 months = after 2026-02-21).
- Funds have many rows, one per class, so group them by fund_family and present one representative (e.g. the class with the largest net assets).

## Value notation rules (get these wrong and the SQL succeeds but the answer is wrong)
- Amount units: 1만=10000, 1억=100000000, 1조=1000000000000 (1 billion = 10억 = 1000000000, 1 trillion = 1조). "Net assets of 1조 won or more" → pd_net_tamt >= 1000000000000.
- Dates are 'YYYY-MM-DD' strings. **When only a year is given, use LIKE**: "maturing in 2027" → mat_date LIKE '2027%'.
  An equality such as mat_date='2027-12-31' keeps only that one day.
- To compare risk grades as numbers use risk_grade (1–6). pd_risk_nm is a label of the form '매우높은위험(1등급)', so
  pd_risk_nm='1등급' always returns 0 rows.
- "Listed / on sale / buyable / current" is is_on_sale=1 for ETFs and funds and is_sellable=1 for bonds. Leave it out and delisted products are counted too.
  "How many ETFs" also needs pd_grp_no='ETF' to exclude ETNs.
- When a question asks about a column or sorts by it, exclude rows where that column is NULL or 0 in WHERE
  (e.g. lowest total fee → AND expense_ratio_pct > 0). The organisers' rule: a value of 0 means there is no value, so it does not go into the result.
- **Copy the numbers a tool returns verbatim; do not rewrite them.** If COUNT returned 240, the answer is 240.

## Listing and recommendation questions
A listing question that gives no count, such as "recommend" or "tell me", gets **the top 5** (if a count is given, that many).
1. Decide the sort criterion first and state it in the answer's first sentence — e.g. "These are the top 5 by yield among bonds that carry a purchase yield".
2. Also give the total number of matches — e.g. "15,837 bonds match the condition, and the top 5 of them are".

## Cross-group queries (comparing several product groups)
Multi-group items such as "top 10 by 1-year return across domestic/foreign ETFs and public funds" come up. Returns are in %.
Combine **only the groups that have the criterion column** with UNION ALL, aligning the column names, then sort. The only
period return in v_etf_gl (foreign ETFs) is du_er_1d (1 day), so leave it out of the UNION for a 1-year or 3-month criterion.
Filling a missing value with NULL or 0 puts that group at the very top or bottom of the ranking and the answer is wrong. Below,
three groups were asked but the UNION has two.

SELECT '국내ETF' AS grp, pd_abrv_nm AS name, pd_itm_no AS code, du_er_1y AS return_1y, pd_net_tamt AS net_assets
  FROM v_etf_kr WHERE is_on_sale=1 AND pd_grp_no='ETF' AND du_er_1y IS NOT NULL
UNION ALL
SELECT '공모펀드', itm_nm, itm_no, fd_yr1_ern_r, fd_nast_suma
  FROM v_fund_pub WHERE product_type='공모펀드' AND is_on_sale=1 AND fd_yr1_ern_r IS NOT NULL
ORDER BY return_1y DESC LIMIT 10

The answer's first sentence: "Foreign ETFs have no 1-year return, so these are the top 10 comparing domestic ETFs and public funds only."
A condition on holding a specific stock (e.g. "that holds 삼성전자") is solved in one SQL by joining v_etf_constituent (pattern below).

## Common SQL patterns
- How many buyable KRW bonds are rated AA- or higher: SELECT COUNT(DISTINCT pd_no) FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-')
- Recommend/list bonds under the same conditions: SELECT DISTINCT pd_no, pd_nm, crd_grd, srfc_irt, mat_date, remaining_days, buy_yield FROM v_bond_kr WHERE is_sellable=1 AND curr_cd='KRW' AND crd_grd IN ('AAA','AA+','AA0','AA-') AND has_buy_yield=1 ORDER BY buy_yield DESC LIMIT 5
  (To compare by yield a bond must carry that value, hence has_buy_yield=1. Sorting by maturity puts bonds redeemed on
   the base date at the top, so it is not used for recommendations.)
- Three US-listed equity foreign ETFs with low total fees and large size: SELECT ticker, pd_nm, expense_ratio_pct, du_last_aum, cu_base_index FROM v_etf_gl WHERE wu_inv_ast_type='Equity' AND is_on_sale=1 AND cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL AND expense_ratio_pct IS NOT NULL ORDER BY expense_ratio_pct ASC, du_last_aum DESC LIMIT 3
- Domestic ETFs by net assets: SELECT pd_ticker, pd_abrv_nm, manager_name, pd_net_tamt, pd_risk_nm FROM v_etf_kr WHERE pd_grp_no='ETF' AND is_on_sale=1 AND pd_net_tamt IS NOT NULL ORDER BY pd_net_tamt DESC LIMIT 10
- Fund family summary: SELECT fund_family, MIN(itm_nm) AS name, COUNT(*) AS classes, MAX(fd_nast_suma) AS max_nav, MAX(bmrk_nm), MAX(or_attr_desc) FROM v_fund_pub WHERE product_type='공모펀드' AND itm_nm LIKE '%국민성장%' GROUP BY fund_family
- ETFs holding a specific stock (top 3 by net assets): SELECT e.pd_abrv_nm, e.pd_net_tamt FROM v_etf_kr e JOIN v_etf_constituent c ON c.etf_itm_no = e.pd_itm_no WHERE c.company_name='삼성전자' AND c.is_cash=0 AND e.is_on_sale=1 AND e.pd_net_tamt IS NOT NULL ORDER BY e.pd_net_tamt DESC LIMIT 3
  (If you know the stock code, c.company_code='005930' is more precise. If the count is asked, COUNT(DISTINCT e.pd_itm_no).)
"""


# The columns shown on the schema cards, and their order.  **Every column the organisers supplied is shown.**
#
# The cards used to carry only 20-30 hand-picked columns per view.  But in the 22-question evaluation the
# model never called table_info once — sql_query 24 times, final_answer 20, graph_neighbors 2, table_info 0.
# Writing "if unsure of a column, call table_info first" into the prompt does not send the model down that
# path, so a column missing from the card is in effect a column that does not exist.  104 columns were
# hidden that way.  Showing every column grows the cards from 3,129 -> 6,257 characters (about +1,564
# tokens) and the system prompt to about 7,100 tokens; in the competition build CLOVA started returning 429 at
# 13,389 tokens, so that is affordable.
#
# The order follows the view definitions (build_store.VIEWS): identifiers and names → type and sale status →
# classification → numbers → dates.  ontology.props_for orders by column name alphabetically and would lose
# that, so schema_card follows the order given here.  Passing a set straight through would change the order
# on every run and break the prompt cache.
CORE_COLUMNS = {
    "v_etf_constituent": [
        "etf_itm_no", "etf_ticker", "etf_name", "company_code", "company_name", "weight_pct", "is_cash", "as_of",
    ],
    "v_bond_kr": [
        "pd_no", "pd_nm", "pd_abrv_nm", "pd_eng_nm", "product_type", "pd_exg_mkt", "info_seq", "is_sellable",
        "bdbns_abl_chnl_nm", "issuer_name", "bd_knd", "std_pd_mcls_nm", "std_pd_scls_nm", "bd_ofr_tcd",
        "bd_inrt_tcd", "bd_intp_tcd", "crd_grd", "crd_grd_dt", "pd_risk_nm", "pd_risk_gcd", "curr_cd",
        "srfc_irt", "isu_date", "mat_date", "remaining_days", "remaining_years", "bd_tisu_a", "isu_bal_amt",
        "has_buy_yield", "applied_yield", "buy_yield", "after_tax_yield", "pref_tax_yield",
        "avg_annual_tax_yield", "corp_pretax_yield", "corp_after_tax_yield", "depo_equiv_yield_154",
        "depo_equiv_yield_495", "trade_price", "buyable_quantity", "eval_price", "dirty", "dur", "cov",
        "exg_close_price", "exg_close_yield", "exrt_grte_ern_r", "exrt_rpy_r", "pd_pen_tr_yn", "base_date",
    ],
    "v_etf_kr": [
        "pd_itm_no", "pd_ticker", "pd_abrv_nm", "pd_nm", "product_type", "pd_grp_no", "is_on_sale", "is_listed",
        "pd_sale_yn", "pd_tr_yn", "manager_name", "cu_fund_mgmt_co", "cu_strtegy", "cu_lev_fector",
        "cu_base_index", "ref_base_index", "cu_charge_rt", "cu_charge_etc_rt", "wu_inv_ast_type", "wu_inv_rgn",
        "ref_ast_type", "ref_geo_focus", "wu_core_yn", "pd_risk_nm", "risk_grade", "pd_pen_tr_yn",
        "pd_pen_risk_nm", "pd_net_tamt", "du_last_aum", "du_last_nav", "du_clpr", "ru_mkt_price", "du_bpr",
        "du_diff_rt", "du_chas_errt", "du_er_1d", "du_er_1m", "du_er_3m", "du_er_6m", "du_er_ytd", "du_er_1y",
        "du_vlty_1m", "du_vlty_3m", "du_vlty_6m", "du_vlty_1y", "du_vol_1d", "du_val_1d", "du_val_1m",
        "pd_dvid_cycl", "pd_dvid_yield", "pd_divd_amt_pshr", "pd_dvid_pay_cnt", "pd_dvid_pay_months",
        "listing_date", "delisting_date", "pd_lst_stk_cnt", "fn_average_coupon", "fn_effective_maturity",
        "base_date",
    ],
    "v_etf_gl": [
        "pd_itm_no", "ticker", "pd_nm", "product_type", "pd_grp_no", "is_on_sale", "is_listed", "pd_tr_yn",
        "pd_sale_yn", "pd_exg_mkt_cd", "pd_mkt_id", "pd_curr_cd", "pd_isin_cd", "manager_name", "cu_base_index",
        "cu_strtegy", "expense_ratio_pct", "cu_charge_rt", "cu_lev_fector", "cu_inverse_short_yn",
        "cu_index_tracking_yn", "cu_index_repl_mthd", "cu_etn_yn", "wu_inv_ast_type", "wu_inv_rgn",
        "wu_core_yn", "du_last_aum", "du_last_nav", "du_clpr", "du_opr", "du_hpr", "du_lpr", "ru_mkt_price",
        "du_er_1d", "du_vol_1d", "du_val_1d", "pd_lst_stk_cnt", "listing_date", "base_date",
    ],
    "v_fund_pub": [
        "itm_no", "itm_nm", "itm_abrv_nm", "itm_eng_nm", "fund_family", "product_type", "prvo_pbff_desc",
        "is_on_sale", "sale_yn", "thco_sale_yn", "manager_name", "manager_source", "or_co_xtn_itt_cd",
        "or_attr_desc", "zrin_btyp_nm", "zrin_ptn_nm", "zrin_attr_nms", "fd_ivst_rgn_desc", "ovrs_fd_desc",
        "bmrk_nm", "bmrk_eng_nm", "curr_cd", "exchdg_yn", "is_hedged", "hdge_fd_yn", "ofsfd_yn", "int_dvd_desc",
        "pers_corp_desc", "zrin_fd_ivst_risk_grd_nm", "risk_grade", "han_clas_nm", "han_clas_fee_type",
        "han_clas_sales_channel", "han_clas_policies", "or_co_rwrd_r", "sale_co_rwrd_r", "trusc_rwrd_r",
        "ofwk_trus_rwrd_r", "fd_nast_suma", "bns_bpr", "fd_wk1_ern_r", "fd_mm1_ern_r", "fd_mm3_ern_r",
        "fd_mm6_ern_r", "fd_yr1_ern_r", "fd_mm18_ern_r", "fd_yr2_ern_r", "fd_yr3_ern_r", "fd_yr5_ern_r",
        "fd_last_dstb_r", "zrin_dmst_stk_cmst_rt", "zrin_ovrs_stk_cmst_rt", "zrin_dmst_bd_cmst_rt",
        "zrin_ovrs_bd_cmst_rt", "zrin_liqt_cmst_rt", "zrin_fd_cmst_rt", "zrin_etc_ast_cmst_rt",
        "prfd_attr_search_text", "base_date",
    ],
}


def value_hints(rdb, max_values: int = 12) -> dict[tuple[str, str], list[str]]:
    """Actual distinct values of the low-cardinality text columns, for the schema cards.

    The ontology only carries allowed values for columns modelled as enumerations, so text columns
    like ``ovrs_fd_desc`` ("국내/해외 구분", domestic/foreign) reached the model as a label with no vocabulary — it then
    omitted the filter entirely, or invented one (``pd_risk_nm='1등급'``, which never matches the real
    label '매우높은위험(1등급)').  Reading the values from the store costs a few tokens and removes the
    guessing.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for view, cols in CORE_COLUMNS.items():
        for col in cols:
            try:
                rows = rdb.query(
                    f"SELECT DISTINCT {col} FROM {view} WHERE {col} IS NOT NULL LIMIT {max_values + 1}"
                ).rows
            except Exception:  # noqa: BLE001 - a column we cannot sample simply gets no hint
                continue
            vals = [str(r[col]).strip() for r in rows if r.get(col) is not None]
            vals = [v for v in vals if v]
            if not (1 < len(vals) <= max_values):
                continue
            if any(len(v) > 24 or v.replace(".", "", 1).replace("-", "", 1).isdigit() for v in vals):
                continue  # long free text or plain numbers: listing them helps nobody
            out[(view, col)] = sorted(vals)
    return out


def build_system_prompt(ontology, rdb_meta: dict, doc_titles: list[str] | None = None, settings=None,
                        hints: dict[tuple[str, str], list[str]] | None = None) -> str:
    budget = getattr(settings, "time_budget_sec", 120)
    rounds = getattr(settings, "max_tool_rounds", 8)
    parts = [
        RULES.strip(),
        f"\n## Budget\nAbout {budget:.0f} seconds per question and at most {rounds} tool-call rounds. Past 300 seconds the question counts as unanswered, so always call final_answer within that.",
        "\n## Data tables (views) overview",
    ]
    for view in VIEW_ID_NAME:
        m = rdb_meta.get(view) or {}
        parts.append(
            f"- {view}: {m.get('korean_name', '')} · source {m.get('source_id', '')} · base date {m.get('base_date', '')} · "
            f"{m.get('row_count', '?')} rows — {SEMANTICS[view]}"
        )
    if ontology is not None:
        parts.append(
            "\n## Schema (every column of each view · column: Korean meaning (unit; allowed values; example values))\n"
            "Every column of each view is listed here. For a column's full description and caveats, use table_info."
        )
        for view in VIEW_ID_NAME:
            cols = list(CORE_COLUMNS.get(view) or [])
            # Columns missing from the list above are appended, so a column newly added to the ontology does
            # not drop off the card when someone forgets the list.  A column not on the card does not exist
            # as far as the model is concerned.
            cols += [p.column for p in ontology.props_for(view) if p.column not in cols]
            card = ontology.schema_card(view, columns=cols, compact=True)
            if hints:
                lines = []
                for ln in card.splitlines():
                    col = ln[2:].split(":", 1)[0] if ln.startswith("- ") else None
                    vals = hints.get((view, col)) if col else None
                    if vals and "값:" not in ln:
                        ln = f"{ln} (값:{'/'.join(vals)})" if "(" not in ln else f"{ln[:-1]}; 값:{'/'.join(vals)})"
                    lines.append(ln)
                card = "\n".join(lines)
            parts.append(card)
    if doc_titles:
        parts.append("\n## External reference documents\n- " + "\n- ".join(doc_titles))
    return "\n".join(parts)
