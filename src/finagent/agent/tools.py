"""Tools exposed to the LLM: typed schemas + executors that also capture evidence."""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass

from ..kb.rdb import SQLGuardError
from ..kb.vector import hybrid_search
from .answer import FINAL_ANSWER_SCHEMA, Evidence
from .llm import ToolSpec

log = logging.getLogger("finagent.tools")

VIEW_ID_NAME = {
    "v_bond_kr": ("pd_no", "pd_nm"),
    "v_etf_kr": ("pd_itm_no", "pd_abrv_nm"),
    "v_etf_gl": ("ticker", "pd_nm"),
    "v_fund_pub": ("itm_no", "itm_nm"),
    "v_etf_constituent": ("etf_itm_no", "company_name"),
}
VIEW_LABEL = {
    "v_bond_kr": "domestic bonds",
    "v_etf_kr": "domestic ETFs/ETNs",
    "v_etf_gl": "foreign ETFs",
    "v_fund_pub": "public/private funds",
    "v_etf_constituent": "domestic ETF holdings",
}
SEMANTICS = {
    "v_bond_kr": "Sellable (buyable) bonds = is_sellable=1 (21,866 bonds not yet redeemed). The buyable_quantity column holds invalid values; do not use it. Only has_buy_yield=1 bonds (634) actually carry a purchase yield and after-tax yield, so a question about yields adds that condition too. crd_grd is NULL for government and public bonds. 'AA- or higher' = crd_grd IN ('AAA','AA+','AA0','AA-'). One bond can have several rows (exchange/OTC + OTC sale LOT info_seq, up to 4 rows), so when asked 'how many', use COUNT(DISTINCT pd_no). Purchase yield and unit price differ per LOT, so the rows themselves are meaningful.",
    "v_etf_kr": "On sale (buyable) = is_on_sale=1 (still listed and not suspended, 1,534). ETFs only = pd_grp_no='ETF' (excludes ETNs). Total fee cu_charge_rt is only partly populated (if NULL, answer 'no data' and do not treat it as 0). Net assets pd_net_tamt (KRW). Listing date listing_date, delisting date delisting_date (9999-12-31 = still listed). Display name pd_abrv_nm. "
                "This view holds 1,235 ETFs and 545 ETNs together — when 'ETF' is asked, always add pd_grp_no='ETF' (without it risk grade 1 returns 594 instead of 223, 2.7 times too many). "
                "Asset class is wu_inv_ast_type: 주식·채권·원자재·대체투자·혼합자산·단기자금·통화·기타. 'ETFs that invest in bonds' = wu_inv_ast_type='채권' (the column prefix is wu_, not cu_). "
                "Distribution frequency is pd_dvid_cycl: 'M'=monthly (monthly dividend), 'Q'=quarterly, 'A'=once a year, 'S'=semi-annual; two spaces and NULL mean no distribution / not recorded. 'Monthly dividend products' = pd_dvid_cycl='M'. pd_dvid_pay_cnt (payments per year 12/4/1/2) carries the same fact.",
    "v_etf_gl": "Every product is US-listed (pd_mkt_id='US'), base date 2026-08-22. Buyable = is_on_sale=1 (not suspended). Equity = wu_inv_ast_type='Equity'. Total fee expense_ratio_pct (%, populated for every product). AUM du_last_aum (USD). Names and strategy (cu_strtegy) are in English → LIKE search with English keywords. "
                "To **count** inverse/short products use cu_inverse_short_yn='Y' (the only values are the string 'Y' (183) and NULL; comparing with the number 1 or IS NOT NULL gives a wrong answer). "
                "cu_lev_fector is the leverage 'multiple' (2.0, 3.0; negative -2.0, -3.0 are inverse multiples), not whether a product is inverse. Use cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL only when excluding both leverage and inverse. "
                "This view also holds 5,972 ETFs and 65 ETNs together, so when 'ETF' is asked add pd_grp_no='ETF'. No holdings data.",
    "v_fund_pub": "Public funds = product_type='공모펀드' (14,716); private funds are mixed in, so always put it in the condition. On sale (buyable) = is_on_sale=1. A fund's classes are separate rows → GROUP BY fund_family and present one representative. "
                  "**When asked 'how many funds', count COUNT(DISTINCT fund_family)** — counting classes (rows) counts the same fund several times (e.g. hedged foreign bond funds on sale are 130 class rows but 63 funds). Use COUNT(*) only when the number of classes itself is asked. "
                  "Domestic/foreign is ovrs_fd_desc: '국내'·'해외'·'국내외혼합'·'해당없음'. 'Foreign funds' = ovrs_fd_desc='해외'. The investment region fd_ivst_rgn_desc (글로벌·아시아·남미/북미…) is a different axis; do not mix the two. "
                  "Currency hedging is is_hedged: 1=hedged (H), 0=unhedged (UH), NULL=not recorded. Management type or_attr_desc: 주식형·채권형·채권혼합·주식혼합·재간접·MMF·파생상품·특별자산·혼합자산 etc. "
                  "The fee columns (or_co_rwrd_r etc.) have an undisclosed unit (do not interpret absolute values; relative comparison only). Returns are populated for only about 30% (NULL means not recorded, not 0). No holdings data.",
    "v_etf_constituent": "Holdings of domestic ETFs (collected from Naver Finance, 1,162 ETFs / 22,599 rows). "
                         "**This view has exactly 8 columns: etf_itm_no, etf_ticker, etf_name, company_code, company_name, weight_pct, is_cash, as_of.** "
                         "Product attributes such as net assets (pd_net_tamt), on-sale status (is_on_sale), ETF/ETN type (pd_grp_no) and total fee are not in this view — put those conditions "
                         "on the product-side alias after joining v_etf_kr: FROM v_etf_constituent c JOIN v_etf_kr e ON e.pd_itm_no = c.etf_itm_no "
                         "WHERE c.company_code = '...' AND e.is_on_sale = 1 ORDER BY e.pd_net_tamt DESC. "
                         "Putting a product-master column directly on the holdings view gives 'no such column', which means the join is missing, not that the data is missing. "
                         "Join on etf_itm_no = v_etf_kr.pd_itm_no. "
                         "A query such as 'ETFs that hold 삼성전자' is solved by joining this view. company_code (stock code, 79% populated) identifies a stock most precisely; "
                         "otherwise use company_name (foreign stocks are written in Korean). 삼성전자 (005930) and 삼성전자우 (005935) are different stocks. "
                         "Cash/deposit rows have is_cash=1, so add is_cash=0 when counting stocks. weight_pct is the weight (%). Foreign ETFs and funds have no holdings data.",
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    evidence: Evidence | None = None


_UNITS = ((10**12, "조"), (10**8, "억"), (10**4, "만"))

# How many labels graph_neighbors actually prints per relation (the total is always reported as is).
#
# Reduced from 40 to 8.  Measured: 삼성전자 (462 relations) dumped 1,501 characters and 미래에셋자산운용
# (9,056) 1,504 characters in one call, and that text stays in the conversation and is re-sent every
# later round.  At 8 they shrink to 455 and 406 characters, 70% and 73% less (우주항공 1,281 -> 252).
#
# Why 8: this list is not sorted.  Questions asking for the "top N" by net assets, return or AUM (T-001,
# T-002, T-007, T-008, T-010, T-017, T-018) can only be solved with SQL anyway, so printing 40 labels does
# not answer them.  What the graph really answers is three things: (a) does the relation exist, (b) how
# many, (c) examples of which ones — and the largest count any evaluation question names is 5.  8 is the
# margin above that.  Holdings now live in v_etf_constituent (SQL), so there is no reason to pull long
# lists from the graph either.
GRAPH_LABELS_SHOWN = 8
# A list this short or shorter is shown in full, uncut.  Cutting to 8 was meant to avoid the latency of big
# lists like 삼성전자 (229), but it hurt small ones: for '캠브리콘이 편입된 중국 반도체 ETF' the graph found
# exactly 15 and showed 8, the model took the list for incomplete, tried to narrow it down again by name,
# ran 8 rounds and died at 119 seconds.  Given the complete list, that search never starts.
GRAPH_LABELS_FULL_UNDER = 25
# How many identifiers go next to the names.  Codes are shorter than labels (about 12 characters) and go
# straight into SQL, so more of them are given.
GRAPH_IDS_SHOWN = 60
# Relations whose target is a 'product' — their IDs join to v_etf_kr.pd_itm_no.
_PRODUCT_RELATIONS = {"hasTheme_inverse", "manages", "tracksIndex_inverse", "constituentOf",
                      "hasConstituent_inverse", "memberOfFamily_inverse", "investsInRegion_inverse",
                      "hasAssetClass_inverse", "hasRiskGrade_inverse"}


def korean_magnitude(v: float) -> str:
    """25474813891225 -> '25조 4,748억'.

    Large amounts arrive as raw integers and the model converts them to Korean units itself, which it
    gets wrong: KODEX 200's net assets came back as '254조 4,818십억' in one run and '25조 4,748십억' in the
    next — the same query, an order of magnitude apart.  Doing the conversion here removes the step
    that was failing.  No currency is named: the magnitude is what gets mistranscribed, and the column
    already says whether it is KRW or USD.
    """
    n = abs(int(v))
    parts, rest = [], n
    for size, name in _UNITS:
        q, rest = divmod(rest, size)
        if q:
            parts.append(f"{q:,}{name}")
        if len(parts) == 2:  # two units is enough to read; more is noise
            break
    if not parts:
        return ""
    return ("-" if v < 0 else "") + " ".join(parts)


def _fmt_val(v, width: int = 70) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) >= 10**8:
        mag = korean_magnitude(v)
        raw = f"{int(v):,}" if float(v).is_integer() else f"{v:,.2f}"
        return f"{raw}({mag})" if mag else raw
    if isinstance(v, float):
        if v.is_integer() and abs(v) >= 1000:
            return f"{int(v):,}"
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
    s = str(v)
    return s if len(s) <= width else s[: width - 1] + "…"


def rows_to_text(cols: list[str], rows: list[dict], max_rows: int = 40) -> str:
    if not rows:
        return "(0 rows)"
    lines = [" | ".join(cols)]
    for r in rows[:max_rows]:
        lines.append(" | ".join(_fmt_val(r.get(c)) for c in cols))
    if len(rows) > max_rows:
        lines.append(f"... (showing {max_rows} of {len(rows)} returned rows)")
    return "\n".join(lines)


class ToolRegistry:
    def __init__(self, rdb, index, resolver, kg, docs, ontology, lexicon, settings, vectors=None):
        self.rdb, self.index, self.resolver, self.kg, self.docs = rdb, index, resolver, kg, docs
        self.ontology, self.lexicon, self.settings = ontology, lexicon, settings
        self.vectors = vectors
        self.meta = rdb.meta() if rdb is not None else {}

    # ------------------------------------------------------------------ specs
    def specs(self) -> list[ToolSpec]:
        tables = "v_bond_kr (domestic bonds) | v_etf_kr (domestic ETFs/ETNs) | v_etf_gl (foreign ETFs) | v_fund_pub (public/private funds)"
        return [
            ToolSpec(
                "table_info",
                f"Returns a table's (view's) column definitions, Korean meanings, allowed values, base date and the rules for judging sellability. Tables: {tables}. Call it before writing SQL when unsure of a column name.",
                {"type": "object", "properties": {"table": {"type": "string", "enum": list(VIEW_ID_NAME)}}, "required": ["table"]},
            ),
            ToolSpec(
                "sql_query",
                "Runs a read-only SQLite SELECT (single statement, at most 200 rows, 5 seconds). Views: v_bond_kr, v_etf_kr, v_etf_gl, v_fund_pub. "
                "Do filtering, sorting, aggregation and comparison with this tool. String matching is LIKE '%keyword%' (English for foreign ETFs). With 0 rows you may relax the conditions and retry.",
                {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SELECT statement to run"},
                        "purpose": {"type": "string", "description": "The purpose of this query (for the trace, one sentence)"},
                    },
                    "required": ["sql"],
                },
            ),
            ToolSpec(
                "search_products",
                "Text search over product names, managers, underlying indices and strategies. mode=hybrid (default) keeps the lexical (BM25) results on top and gives only the bottom ~30% of slots to products that semantic (vector) search alone found. lexical matches the exact spelling; vector finds products close in meaning even when worded differently. "
                "If you know the product name, code or ticker, sql_query is more precise than this tool (measured: official-name recall@1, lexical 93.8% vs vector 33.8%). Use it when you do not know the name and only know the theme, character or strategy (e.g. 우주항공, robot, China semiconductor). For foreign ETFs include English keywords too. Returns: table, id (product code), name, key fields.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "table": {"type": "string", "enum": list(VIEW_ID_NAME), "description": "Restrict to one table (optional)"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                        "mode": {"type": "string", "enum": ["hybrid", "lexical", "vector"], "default": "hybrid"},
                    },
                    "required": ["query"],
                },
            ),
            ToolSpec(
                "resolve_entity",
                "Resolves a proper noun in the question (product, company, manager, theme) against the lexicon and index. Returns whether it matched exactly (exact), similar candidates, a company's parent/subsidiaries and listing status, and theme synonyms. "
                "An English company name resolves to the Korean name used in the data. Call it first when checking whether a product really exists.",
                {"type": "object", "properties": {"mention": {"type": "string"}}, "required": ["mention"]},
            ),
            ToolSpec(
                "graph_neighbors",
                "Looks up the relations of an entity (product name/code/ticker, company, theme, manager) in the knowledge graph. Relations include: managedBy, hasTheme, hasConstituent (holdings), affiliateOf (affiliates), "
                "subsidiaryOf, hasSubsidiary, tracksIndex, hasBenchmark, memberOfFamily, investsInRegion, hasAssetClass, hasRiskGrade, hasTheme_inverse (theme→products). "
                "E.g. graph_neighbors('에코프로','hasSubsidiary'), graph_neighbors('에코프로비엠','constituentOf'), graph_neighbors('우주항공','hasTheme_inverse').",
                {
                    "type": "object",
                    "properties": {"entity": {"type": "string"}, "relation": {"type": "string", "description": "Only this relation (optional)"}},
                    "required": ["entity"],
                },
            ),
            ToolSpec(
                "graph_query",
                "Knowledge-graph SPARQL SELECT. The prefixes fp: (schema) and d: (data) are added automatically. Product IRIs: d:etf_kr/{ISIN}, d:etf_gl/{RIC}, d:bond/{ISIN}, d:fund/{itm_no}. "
                "E.g. SELECT ?etf ?name WHERE { ?etf fp:hasConstituent ?c . ?c fp:subsidiaryOf ?p . ?p fp:name '에코프로' . ?etf fp:name ?name } / "
                "SELECT ?p ?n WHERE { ?p fp:hasTheme d:theme/space_aero ; fp:name ?n ; fp:listingDate ?d . FILTER(?d >= '2026-02-21'^^xsd:date) }",
                {"type": "object", "properties": {"sparql": {"type": "string"}}, "required": ["sparql"]},
            ),
            ToolSpec(
                "search_documents",
                "Searches the external reference documents (markdown: 국민성장펀드 overview, ETF terms and metrics explained, general ETF risk factors). Use it when you need background, risk factors or terminology the product masters do not hold. "
                "Source and base date come with each result; the organisers' data takes precedence.",
                {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 8}}, "required": ["query"]},
            ),
            ToolSpec(
                "final_answer",
                "Submits the final answer and ends. Always call it once, at the end. status: answered (backed by evidence) / cannot_confirm (cannot be confirmed with the provided data) / clarify (ambiguous condition; ask back).",
                FINAL_ANSWER_SCHEMA,
            ),
        ]

    # ------------------------------------------------------------------ exec
    async def execute(self, name: str, args: dict) -> ToolResult:
        try:
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return ToolResult(f"Unknown tool: {name}", True)
            return fn(**(args or {}))
        except TypeError as e:
            return ToolResult(f"Invalid arguments: {e}", True)
        except SQLGuardError as e:
            return ToolResult(str(e), True)
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model
            log.exception("tool %s failed", name)
            return ToolResult(f"Tool error ({type(e).__name__}): {e}", True)

    def _meta_line(self, view: str) -> tuple[str | None, str | None]:
        m = self.meta.get(view) or {}
        return m.get("korean_name"), m.get("base_date")

    def _t_table_info(self, table: str) -> ToolResult:
        if table not in VIEW_ID_NAME:
            return ToolResult(f"Unknown table: {table}. Available: {', '.join(VIEW_ID_NAME)}", True)
        kname, base = self._meta_line(table)
        m = self.meta.get(table) or {}
        card = self.ontology.schema_card(table) if self.ontology is not None else "### " + table + "\n- " + "\n- ".join(self.rdb.columns(table))
        text = f"{card}\nBase date: {base} · rows: {m.get('row_count')}\nRules: {SEMANTICS[table]}"
        return ToolResult(text, False, Evidence("schema", table, kname, base, "table_info", None))

    # Views where one product spans several rows, so the row count and the product count differ.
    #   v_bond_kr  — exchange/OTC + OTC sale LOTs (info_seq): up to 4 rows per bond
    #   v_fund_pub — one fund has a row per class
    ROW_UNIT = {
        "v_bond_kr": ("pd_no", "bonds"),
        "v_fund_pub": ("fund_family", "funds (classes combined)"),
    }
    _COUNT_STAR = re.compile(r"count\s*\(\s*\*\s*\)", re.I)

    def _unit_warning(self, sql: str, view: str | None) -> str:
        """Asking 'how many' with COUNT(*) returns a row count.  Stating the real product count too prevents a wrong answer.

        The prompt says 'use DISTINCT', but the model ignored it and a wrong answer of 16,023 (rows) vs
        15,837 (bonds) actually happened.  Instead of an instruction, the right number rides on the result.
        """
        if not view or view not in self.ROW_UNIT:
            return ""
        if not self._COUNT_STAR.search(sql) or "distinct" in sql.lower():
            return ""
        # Only a query whose whole result is one count can be restated as a different count.  A
        # GROUP BY returns one row per group with the dimension in the first column, and rewriting it
        # produced ValueError("Cannot specify ',' with 's'") when the group key hit the f-string's
        # thousands separator — breaking the very "fund family summary" pattern the prompt teaches.
        if re.search(r"\bgroup\s+by\b", sql, re.I):
            return ""
        key, unit = self.ROW_UNIT[view]
        try:
            alt = self.rdb.query(self._COUNT_STAR.sub(f"COUNT(DISTINCT {key})", sql))
        except SQLGuardError:
            return ""
        if not alt.rows or len(alt.rows) != 1 or len(alt.columns) != 1:
            return ""
        n = list(alt.rows[0].values())[0]
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            return ""
        return (f"\n⚠ The number above is a row count. In {view} one product can span several rows, so "
                f"the actual number of {unit} is {int(n):,}. If the question asks how many, answer {int(n):,}.")

    def _true_total(self, sql: str) -> int | None:
        """How many rows the query really matches, ignoring the row cap.

        The cap exists to bound query cost, but reporting the capped number as the total is the same
        mistake the graph made with its 60-neighbour limit: '200 in total' for a filter that matches 15,837
        products invites the model to answer 200.  A cap may hide rows; it must never change a count.
        """
        inner = re.sub(r"\blimit\s+\d+(\s+offset\s+\d+)?\s*$", "", sql.strip().rstrip(";"), flags=re.I).strip()
        if not inner:
            return None
        try:
            r = self.rdb.query(f"SELECT COUNT(*) AS n FROM ({inner})")
        except SQLGuardError:
            return None
        return int(list(r.rows[0].values())[0]) if r.rows else None

    def _zero_warnings(self, sql: str, columns: list[str], rows: list[dict]) -> str:
        """Attach, right to the result, the two cases where a 0 contaminates the answer.

        The organisers announced that missing values were left as 0 or NULL.  Even with 'do not use 0 as
        a value' in the prompt, the model answered three products with a 0% total fee as the lowest fee
        (MDY is actually 0.24%), and when a mismatched condition spelling made COUNT return 0 it answered
        '0' as is.  A warning at the moment the result arrives is more reliable.
        """
        out = ""
        if len(rows) == 1 and len(columns) == 1:
            v = rows[0].get(columns[0])
            if isinstance(v, (int, float)) and v == 0 and re.search(r"count\s*\(", sql, re.I):
                out += ("\n⚠ 0 rows. A condition value may be spelled differently from the actual data: drop the "
                        "conditions one at a time to find which one produces 0, then answer. Check a column's actual values with table_info.")
        m = re.search(r"order\s+by\s+([A-Za-z_][A-Za-z0-9_]*)\s*(asc)?\b", sql, re.I)
        if m and rows:
            col = m.group(1)
            if col in columns and sum(1 for r in rows if r.get(col) == 0) and (m.group(2) or "").lower() != "desc":
                out += (f"\n⚠ The sort column {col} contains 0s. A 0 means the value was not provided, so it must not be ranked. "
                        f"Add AND {col} > 0 and query again.")
        return out

    # ------------------------------------------- turn SQL errors into messages that teach
    # This lives in the tool layer, not in the guard.  RDB.query is a general accessor shared by the ETL,
    # the evaluation scripts and the internal re-queries of _true_total/_unit_warning just above, so
    # putting LLM-facing guidance into the guard would pollute human-readable logs and the internal retry
    # paths.  The place that knows the view catalogue and whose strings reach the model verbatim is here.
    #
    # Evidence (G-007): "no such column: pd_ticker" says only what is wrong, not what is right.  The model
    # can only guess again, and repeated guessing burns all 8 rounds.
    _NO_COLUMN = re.compile(r"no such column:\s*(?:(?P<alias>[A-Za-z_]\w*)\.)?(?P<col>[A-Za-z_]\w*)", re.I)
    _NO_TABLE = re.compile(r"no such table:\s*(?P<tbl>[A-Za-z_]\w*)", re.I)
    # Attempts to discover the column list through SQL (PRAGMA/sqlite_master) are blocked by the guard —
    # the next round survives only if we name the tool to use instead, not the reason for the block.
    _CATALOG_SQL = re.compile(r"\b(pragma|sqlite_master|sqlite_schema|describe|show\s+columns)\b", re.I)
    # Source-system prefixes.  The model writing 'pd_ticker' when the view only has 'ticker' really happened.
    _SRC_PREFIX = re.compile(r"^(?:pd|cu|du|wu|ru|fn|ref|etf|or|zrin|fd|std|bd|crd)_", re.I)
    _MAX_COLS_SHOWN = 24   # a view narrower than this can be listed in full without hurting the context
    _NEAR_COLS_SHOWN = 10  # a wide view shows only nearby candidates and states the total count

    # Evidence (the 119-second question): right after "SELECT ... du_er_1y FROM v_etf_gl" failed, the model
    # went off track and started writing 'an approach' instead of an answer.  The suggestion above says
    # 'use du_er_1d', but a 1-day return is no substitute for a 1-year return — a close name is not the same
    # meaning.  Saying only what is right, without 'so do this', lets the next round leak into planning.
    # It is attached even when there is a suggestion: du_er_1y was exactly a case where a suggestion came
    # out and still could not be the answer.
    _NO_COLUMN_DIRECTIVE = (
        "\nA similar name does not mean the same meaning (period, unit). Use the column in that view whose meaning matches; if there is none, "
        "do not substitute a value of another period or unit — say in one sentence that the product group was excluded from the comparison "
        "and answer right away with the remaining groups (do not describe the method)."
    )

    def _view_columns(self, view: str) -> list[str]:
        cache = getattr(self, "_colcache", None)
        if cache is None:
            cache = self._colcache = {}
        if view not in cache:
            try:
                cache[view] = list(self.rdb.columns(view))
            except Exception:  # noqa: BLE001 - a failed catalogue lookup must not swallow the error message
                cache[view] = []
        return cache[view]

    @classmethod
    def _core_name(cls, col: str) -> str:
        """The name without its prefix.  'pd_ticker' -> 'ticker', 'etf_itm_no' -> 'itm_no'."""
        return cls._SRC_PREFIX.sub("", col.lower())

    def _column_suggestions(self, wanted: str, cols: list[str], n: int = 4) -> list[str]:
        """The real column names closest to wanted.  Prefix confusion first, then edit distance."""
        want, core = wanted.lower(), self._core_name(wanted)
        out: list[str] = []
        for c in cols:  # a prefix-only difference ranks first (G-007's pd_ticker -> ticker)
            if c.lower() != want and self._core_name(c) == core:
                out.append(c)
        for c in difflib.get_close_matches(wanted, cols, n=n, cutoff=0.6):
            if c not in out and c.lower() != want:
                out.append(c)
        by_core: dict[str, str] = {}
        for c in cols:
            by_core.setdefault(self._core_name(c), c)
        for k in difflib.get_close_matches(core, list(by_core), n=n, cutoff=0.6):
            if by_core[k] not in out and by_core[k].lower() != want:
                out.append(by_core[k])
        return out[:n]

    def _referenced_views(self, sql: str) -> list[str]:
        return [v for v in VIEW_ID_NAME if re.search(rf"\b{v}\b", sql, re.I)]

    def _sql_error_hint(self, sql: str, msg: str) -> str:
        """Append 'what is right' to a failed query.  Empty string when there is nothing to add."""
        m = self._NO_TABLE.search(msg)
        if m:
            bad = m.group("tbl")
            near = difflib.get_close_matches(bad, list(VIEW_ID_NAME), n=1, cutoff=0.5)
            head = f"\nUse {near[0]} instead of '{bad}'. " if near else "\n"
            return (head + "Only these 5 views are available: "
                    + ", ".join(f"{v} ({VIEW_LABEL[v]})" for v in VIEW_ID_NAME)
                    + ". Check column definitions with the table_info tool.")
        m = self._NO_COLUMN.search(msg)
        if m:
            bad, alias = m.group("col"), m.group("alias")
            views = self._referenced_views(sql)
            if alias:  # with an alias, look only at the view it points to — more precise on JOIN queries
                am = re.search(rf"\b(v_\w+)\s+(?:as\s+)?{re.escape(alias)}\b", sql, re.I)
                if am and am.group(1) in VIEW_ID_NAME:
                    views = [am.group(1)]
            if not views:
                return ("\nCannot tell which view the query refers to. Available views: "
                        + ", ".join(VIEW_ID_NAME) + " (column definitions via table_info).")
            best: list[str] = []
            listing: list[str] = []
            for v in views:
                cols = self._view_columns(v)
                if not cols:
                    continue
                sug = self._column_suggestions(bad, cols)
                best += [f"{v}.{c}" for c in sug[:2]]
                if len(cols) <= self._MAX_COLS_SHOWN:
                    listing.append(f"- {v} has {len(cols)} columns: " + ", ".join(cols))
                else:
                    # Listing a wide view in full eats the context.  Give the nearby names + the total only.
                    near = sug or difflib.get_close_matches(bad, cols, n=self._NEAR_COLS_SHOWN, cutoff=0.0)
                    listing.append(
                        f"- {v} has {len(cols)} columns; names close to '{bad}': "
                        + ", ".join(near[: self._NEAR_COLS_SHOWN])
                        + f" (full list: table_info('{v}'))"
                    )
            out = ""
            if best:
                out += "\nUse " + " or ".join(best[:3]) + f" instead of '{bad}'."
            if listing:
                out += "\n" + "\n".join(listing)
            return out + self._NO_COLUMN_DIRECTIVE
        # A catalogue query blocked by the guard: name the tool to use instead, not the reason for the block.
        # (Trying PRAGMA because the column name is unknown is exactly how this dead end arises.)
        if self._CATALOG_SQL.search(sql or ""):
            return "\nLook up column lists, base dates and rules with the table_info tool, not with SQL. E.g. table_info(table='v_etf_kr')."
        return ""

    def _t_sql_query(self, sql: str, purpose: str = "") -> ToolResult:
        try:
            r = self.rdb.query(sql)
        except SQLGuardError as e:
            return ToolResult(str(e) + self._sql_error_hint(sql, str(e)), True)
        view = next((v for v in VIEW_ID_NAME if v in r.sql), None)
        kname, base = self._meta_line(view) if view else (None, None)
        refs = []
        if view:
            idc, namec = VIEW_ID_NAME[view]
            for row in r.rows[:50]:
                cid, cname = row.get(idc), row.get(namec)
                if cid or cname:
                    refs.append(f"{cid or ''} {cname or ''}".strip())
        text = rows_to_text(r.columns, r.rows)
        total = self._true_total(r.sql) if r.truncated else r.row_count
        if r.truncated:
            text += (f"\n{total:,} rows match in total; the above shows {r.row_count} of them."
                     if total is not None else
                     f"\nAt least {r.row_count} rows (the row cap was hit — get the exact number with COUNT).")
            text += " If the question asks how many, answer with this total; if a list is needed, add a sort criterion and take only the top N."
        else:
            text += f"\nTotal rows: {r.row_count}"
        text += self._unit_warning(r.sql, view)
        text += self._zero_warnings(r.sql, r.columns, r.rows)
        return ToolResult(text, False, Evidence("sql", view, kname, base, r.sql, total if total is not None else r.row_count, refs))

    def _t_search_products(self, query: str, table: str | None = None, k: int = 10, mode: str = "hybrid") -> ToolResult:
        k = int(k)
        mode = mode if mode in ("hybrid", "lexical", "vector") else "hybrid"
        if self.vectors is None or len(self.vectors) == 0:
            mode = "lexical"
        if mode == "lexical":
            hits = self.index.search(query, k=k, table=table)
        elif mode == "vector":
            hits = self.vectors.search(query, k=k, table=table)
        else:
            hits = hybrid_search(
                self.index.search(query, k=max(k, 20), table=table),
                self.vectors.search(query, k=max(k, 20), table=table),
                k=k,
            )
        detail = f"search_products('{query}', mode={mode}" + (f", table={table}" if table else "") + ")"
        if not hits:
            return ToolResult("No search results", False, Evidence("search", table, None, None, detail, 0))
        lines, refs = [], []
        for h in hits:
            idc, _ = VIEW_ID_NAME.get(h.table, ("id", "name"))
            key = {k2: v for k2, v in h.payload.items() if v not in (None, "") and k2 not in (idc,)}
            lines.append(f"- [{h.table}] id={h.id} | {h.title} | " + ", ".join(f"{a}={_fmt_val(b, 40)}" for a, b in list(key.items())[:8]))
            refs.append(f"{h.id} {h.title}")
        return ToolResult("\n".join(lines), False, Evidence("search", table, None, None, detail, len(hits), refs))

    def _t_resolve_entity(self, mention: str) -> ToolResult:
        r = self.resolver.resolve(mention, k=5)
        text = r.summary()
        for c in r.companies:
            subs = self.lexicon.subsidiaries_of(c.id)
            if subs:
                text += f"\nSubsidiaries of {c.name_ko} (lexicon): " + ", ".join(s.name_ko for s in subs)
            elif self.kg is not None:
                # Second best when there is no ownership-based subsidiary.  Written as affiliates, not subsidiaries.
                cand = self.kg.subsidiary_candidates(c.name_ko, limit=40)
                if cand["kind"] == "affiliate":
                    text += (
                        f"\nAffiliates of {c.name_ko} (from the holdings of {'/'.join(cand['groups']) or 'group-stock ETFs'}, {cand['total']}): "
                        + ", ".join(x["label"] for x in cand["items"][:20])
                        + " — this means the same business group, not an ownership (subsidiary) relation."
                    )
            if self.kg is not None:
                nb = self.kg.neighbors(c.name_ko, relation="constituentOf", limit=30)
                cons = nb.get("constituentOf", [])
                if cons:
                    text += f"\nProducts holding {c.name_ko} (graph, {len(cons)}): " + ", ".join(x["label"] for x in cons[:15])
                elif not c.not_listed:
                    text += f"\nProducts holding {c.name_ko}: no holdings data"
            if c.note:
                text += f"\nNote: {c.note}"
        refs = [f"{h.id} {h.title}" for h in r.products[:5]]
        return ToolResult(text, False, Evidence("entity", None, None, None, f"resolve_entity('{mention}') exact={r.exact}", None, refs))

    def _t_graph_neighbors(self, entity: str, relation: str | None = None) -> ToolResult:
        if self.kg is None:
            return ToolResult("The knowledge graph is not loaded.", True)
        totals: dict[str, int] = {}
        nb = self.kg.neighbors(entity, relation=relation, limit=60, totals=totals)
        note = ""
        if not nb and relation in ("hasSubsidiary", "subsidiaryOf_inverse"):
            # The organisers' example uses the word 'subsidiary' (자회사).  Ownership-based subsidiary edges
            # exist for only 4 hand-checked cases, so every other company fell to 'cannot answer' (real
            # incident: "한화 에어로스페이스의 자회사를 편입한 ETF").  Answer with affiliates of the same
            # business group instead, but do not swap the relation's name — say what is being returned.
            # An affiliate is 'the same group', not '50%+ ownership'.
            cand = self.kg.subsidiary_candidates(entity, limit=60)
            if cand["kind"] == "affiliate":
                totals, nb = {"affiliateOf": cand["total"]}, {"affiliateOf": cand["items"]}
                note = (
                    f"(The provided data has no ownership-based subsidiaries of '{entity}'. Returning **affiliates** of the same "
                    f"business group ({'/'.join(cand['groups']) or 'group-stock ETFs'}) instead — a relation derived from stocks that "
                    f"group-stock ETFs hold together, not an ownership relation. Call them affiliates (계열사) in the answer and "
                    f"do not write 'X is a subsidiary of Y'.)\n"
                )
        if not nb and relation:
            # An empty *filtered* result is nearly always the wrong direction, not a missing entity:
            # 삼성전자 has no hasConstituent but constituentOf has 238.  Returning the relations it does
            # have — as data, not as advice — answers in this round; a hint alone was ignored and the
            # model replied "확인할 수 없습니다" (cannot be confirmed) while the graph held the answer.
            totals = {}
            nb = self.kg.neighbors(entity, limit=60, totals=totals)
            if nb:
                note = f"(No '{relation}' relation, so all relations of this entity are shown instead)\n"
        if not nb:
            return ToolResult(f"No node or relation found for '{entity}'", False, Evidence("graph", None, None, None, f"graph_neighbors('{entity}', {relation})", 0))
        lines, refs = [], []
        for rel, items in nb.items():
            # report the untruncated total — the model quotes this number verbatim as its answer
            total = totals.get(rel, len(items))
            head = items if total <= GRAPH_LABELS_FULL_UNDER else items[:GRAPH_LABELS_SHOWN]
            more = total > len(head)
            # "Examples", not "top": this list has no order.  The top by net assets or return only comes from SQL.
            shown = f"{total} total" + (f", {len(head)} examples shown" if more else "")
            lines.append(f"{rel} ({shown}): " + ", ".join(x["label"] for x in head) + (" ..." if more else ""))
            # Give the store's identifiers next to the names.  Names alone do not make SQL: for '최근 6개월
            # 우주항공 테마 ETF' the graph found 100 and returned 8 names; the model discarded them, searched
            # the holdings' company names for '우주'/'항공' with LIKE, got 53 unrelated rows and ended with
            # "there is no information linking them to the theme".  With IDs a single WHERE ... IN (...)
            # attaches net assets, returns and total fees.  Same remedy as when the pre-analysis started
            # passing company codes and 캠브리콘, 네이버 and TSMC got solved.
            ids = [x["iri"].rsplit("/", 1)[-1] for x in items if x.get("iri")]
            ids = [i for i in ids if i][:GRAPH_IDS_SHOWN]
            if ids and rel in _PRODUCT_RELATIONS:
                lines.append(
                    f"  └ lookup IDs of the products above ({len(ids)}"
                    + (f" of {total}" if total > len(ids) else "")
                    + "): pd_itm_no IN (" + ", ".join(f"'{i}'" for i in ids) + ")"
                    + " — do not search by name again; query v_etf_kr with these IDs to attach net assets, returns and total fees."
                )
            refs += [x["label"] for x in items[:10]]
        return ToolResult(note + "\n".join(lines), False, Evidence("graph", None, None, None, f"graph_neighbors('{entity}', {relation})", sum(totals.values()) or sum(len(v) for v in nb.values()), refs[:30]))

    def _t_graph_query(self, sparql: str) -> ToolResult:
        if self.kg is None:
            return ToolResult("The knowledge graph is not loaded.", True)
        try:
            rows = self.kg.query(sparql, limit=100)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"SPARQL error: {e}", True)
        if not rows:
            return ToolResult("(0 rows)", False, Evidence("graph", None, None, None, sparql.strip(), 0))
        lines = [" | ".join(f"{k}={v}" for k, v in r.items()) for r in rows[:60]]
        refs = [next(iter(r.values())) for r in rows[:20]]
        return ToolResult("\n".join(lines) + f"\nTotal rows: {len(rows)}", False, Evidence("graph", None, None, None, sparql.strip(), len(rows), refs))

    def _t_search_documents(self, query: str, k: int = 8) -> ToolResult:
        if self.docs is None or len(self.docs) == 0:
            return ToolResult("No external documents", False)
        hits = self.docs.search(query, k=int(k))
        if not hits:
            return ToolResult("No matching documents", False, Evidence("docs", None, None, None, f"search_documents('{query}')", 0))
        lines, refs = [], []
        for c in hits:
            lines.append(f"### {c.title} › {c.heading} (source: {c.source or 'internal notes'}, as of {c.as_of or '-'})\n{c.text}")
            refs.append(f"{c.title} › {c.heading}")
        return ToolResult("\n\n".join(lines), False, Evidence("docs", None, None, hits[0].as_of or None, f"search_documents('{query}')", len(hits), refs))

    def _t_final_answer(self, status: str = "answered", answer: str = "", evidence_refs: list | None = None) -> ToolResult:
        return ToolResult(json.dumps({"status": status, "answer": answer, "evidence_refs": evidence_refs or []}, ensure_ascii=False))
