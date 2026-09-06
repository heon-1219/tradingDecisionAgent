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
    "v_bond_kr": "국내채권",
    "v_etf_kr": "국내ETF/ETN",
    "v_etf_gl": "해외ETF",
    "v_fund_pub": "공모/사모펀드",
    "v_etf_constituent": "국내ETF 편입종목",
}
SEMANTICS = {
    "v_bond_kr": "판매가능(구매가능) 채권 = is_sellable=1 (상환 미도래 종목 21,866건). buyable_quantity 컬럼은 값이 무효이므로 사용 금지. 매수수익률·세후수익률이 실제로 수록된 종목은 has_buy_yield=1 (634건)뿐이므로 수익률을 묻는 질의는 이 조건을 함께 쓴다. crd_grd는 국공채에 NULL. 'AA- 이상' = crd_grd IN ('AAA','AA+','AA0','AA-'). 한 종목이 여러 행일 수 있다(장내/장외 + 장외 판매 LOT info_seq, 최대 4행). 따라서 '몇 개'를 물으면 COUNT(DISTINCT pd_no) 를 쓴다. LOT 마다 매수수익률·단가가 다르므로 행 자체는 의미가 있다.",
    "v_etf_kr": "판매중(구매가능) = is_on_sale=1 (상장 유지 + 거래정지 아님, 1,534건). ETF만 = pd_grp_no='ETF' (ETN 제외). 총보수 cu_charge_rt는 일부만 수록(NULL이면 '데이터 없음'이라고 답하고 0으로 취급하지 않는다). 순자산 pd_net_tamt(KRW). 상장일 listing_date, 상장종료일 delisting_date(9999-12-31=상장 유지). 대표명 pd_abrv_nm.",
    "v_etf_gl": "전 종목 미국 상장(pd_mkt_id='US'), 기준일 2026-08-22. 구매가능 = is_on_sale=1(거래정지 아님). 주식형 = wu_inv_ast_type='Equity'. 총보수 expense_ratio_pct(%, 전 종목 수록). 운용규모 du_last_aum(USD). 이름·전략(cu_strtegy)은 영문 → 영문 키워드로 LIKE 검색. 레버리지/인버스 제외 시 cu_lev_fector IS NULL AND cu_inverse_short_yn IS NULL. 편입종목 데이터 없음.",
    "v_fund_pub": "공모펀드 = product_type='공모펀드'(14,716건), 사모도 섞여 있으므로 반드시 조건에 넣는다. 판매중(구매가능) = is_on_sale=1. 한 펀드의 클래스들이 여러 행 → fund_family로 GROUP BY 하고 대표 1개만 제시. 보수 컬럼(or_co_rwrd_r 등)은 단위 미공개(절대값 해석 금지, 상대 비교만). 수익률은 약 30%만 수록(NULL은 0이 아니라 미수록). 편입종목 데이터 없음.",
    "v_etf_constituent": "국내 ETF의 편입종목(네이버금융 수집, 1,162개 ETF·22,599행). etf_itm_no 로 v_etf_kr.pd_itm_no 와 조인한다. "
                         "'삼성전자를 보유한 ETF' 같은 질의는 이 뷰를 조인해서 푼다. 종목 지정은 company_code(종목코드, 79% 수록)가 가장 정확하고, "
                         "없으면 company_name 을 쓴다(해외 종목은 한글 표기). 삼성전자(005930)와 삼성전자우(005935)는 다른 종목이다. "
                         "현금·예금 행은 is_cash=1 이므로 종목을 셀 때는 is_cash=0 을 건다. weight_pct 는 비중(%). 해외ETF·펀드는 편입종목 데이터가 없다.",
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    evidence: Evidence | None = None


_UNITS = ((10**12, "조"), (10**8, "억"), (10**4, "만"))


def korean_magnitude(v: float) -> str:
    """25474813891225 -> '25조 4,748억'.

    Large amounts arrive as raw integers and the model converts them to Korean units itself, which it
    gets wrong: KODEX 200's 순자산 came back as '254조 4,818십억' in one run and '25조 4,748십억' in the
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
        return "(0건)"
    lines = [" | ".join(cols)]
    for r in rows[:max_rows]:
        lines.append(" | ".join(_fmt_val(r.get(c)) for c in cols))
    if len(rows) > max_rows:
        lines.append(f"... (표시 {max_rows}건 / 반환 {len(rows)}건)")
    return "\n".join(lines)


class ToolRegistry:
    def __init__(self, rdb, index, resolver, kg, docs, ontology, lexicon, settings, vectors=None):
        self.rdb, self.index, self.resolver, self.kg, self.docs = rdb, index, resolver, kg, docs
        self.ontology, self.lexicon, self.settings = ontology, lexicon, settings
        self.vectors = vectors
        self.meta = rdb.meta() if rdb is not None else {}

    # ------------------------------------------------------------------ specs
    def specs(self) -> list[ToolSpec]:
        tables = "v_bond_kr(국내채권) | v_etf_kr(국내ETF/ETN) | v_etf_gl(해외ETF) | v_fund_pub(공모/사모펀드)"
        return [
            ToolSpec(
                "table_info",
                f"테이블(뷰)의 컬럼 정의·한글 의미·허용값·기준일·판매가능 판정 규칙을 반환합니다. 테이블: {tables}. SQL 작성 전에 컬럼명이 불확실하면 호출하세요.",
                {"type": "object", "properties": {"table": {"type": "string", "enum": list(VIEW_ID_NAME)}}, "required": ["table"]},
            ),
            ToolSpec(
                "sql_query",
                "SQLite 읽기 전용 SELECT 를 실행합니다(단일 문장, 최대 200행, 5초). 뷰 이름: v_bond_kr, v_etf_kr, v_etf_gl, v_fund_pub. "
                "필터·정렬·집계·비교는 이 도구로 수행하세요. 문자열 매칭은 LIKE '%키워드%' (해외ETF는 영문). 결과 0건이면 조건을 완화해 재시도할 수 있습니다.",
                {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "실행할 SELECT 문"},
                        "purpose": {"type": "string", "description": "이 질의의 목적(추적용, 한 문장)"},
                    },
                    "required": ["sql"],
                },
            ),
            ToolSpec(
                "search_products",
                "상품명·운용사·기초지수·전략 텍스트 검색. mode=hybrid(기본)는 어휘검색(BM25)과 의미검색(벡터)을 RRF로 결합하고, lexical은 정확한 표기 매칭, vector는 표현이 달라도 의미가 가까운 상품을 찾습니다. "
                "정확한 상품명을 모르거나 테마 키워드(예: 우주항공, robot, China semiconductor)로 후보를 찾을 때 사용. 해외ETF는 영문 키워드도 함께 넣으세요. 반환: 테이블, id(상품코드), 이름, 핵심 필드.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "table": {"type": "string", "enum": list(VIEW_ID_NAME), "description": "특정 테이블로 제한(선택)"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                        "mode": {"type": "string", "enum": ["hybrid", "lexical", "vector"], "default": "hybrid"},
                    },
                    "required": ["query"],
                },
            ),
            ToolSpec(
                "resolve_entity",
                "질문 속 고유명사(상품명, 기업명, 운용사, 테마)를 사전·색인으로 해석합니다. 정확 일치 여부(exact), 유사 후보, 기업의 모회사/자회사·상장 여부, 테마 동의어를 반환합니다. "
                "상품이 실제로 존재하는지 확인할 때 먼저 호출하세요.",
                {"type": "object", "properties": {"mention": {"type": "string"}}, "required": ["mention"]},
            ),
            ToolSpec(
                "graph_neighbors",
                "지식그래프에서 엔티티(상품명/코드/티커/기업명/테마명/운용사명)의 관계를 조회합니다. 관계 예: managedBy, hasTheme, hasConstituent(편입종목), constituentOf_inverse, "
                "subsidiaryOf, hasSubsidiary, tracksIndex, hasBenchmark, memberOfFamily, investsInRegion, hasAssetClass, hasRiskGrade, hasTheme_inverse(테마→상품). "
                "예: graph_neighbors('에코프로','hasSubsidiary'), graph_neighbors('에코프로비엠','constituentOf'), graph_neighbors('우주항공','hasTheme_inverse').",
                {
                    "type": "object",
                    "properties": {"entity": {"type": "string"}, "relation": {"type": "string", "description": "특정 관계만(선택)"}},
                    "required": ["entity"],
                },
            ),
            ToolSpec(
                "graph_query",
                "지식그래프 SPARQL SELECT. 접두사 fp:(스키마), d:(데이터)는 자동 추가. 상품 IRI: d:etf_kr/{ISIN}, d:etf_gl/{RIC}, d:bond/{ISIN}, d:fund/{itm_no}. "
                "예: SELECT ?etf ?name WHERE { ?etf fp:hasConstituent ?c . ?c fp:subsidiaryOf ?p . ?p fp:name '에코프로' . ?etf fp:name ?name } / "
                "SELECT ?p ?n WHERE { ?p fp:hasTheme d:theme/space_aero ; fp:name ?n ; fp:listingDate ?d . FILTER(?d >= '2026-02-21'^^xsd:date) }",
                {"type": "object", "properties": {"sparql": {"type": "string"}}, "required": ["sparql"]},
            ),
            ToolSpec(
                "search_documents",
                "외부 참고 문서(마크다운: 국민성장펀드 개요, ETF 용어·지표 해설, ETF 위험요인 일반)를 검색합니다. 상품 마스터에 없는 배경 설명·위험요인·용어가 필요할 때 사용. "
                "출처·기준일이 함께 반환되며, 주최측 제공 데이터가 우선합니다.",
                {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 4}}, "required": ["query"]},
            ),
            ToolSpec(
                "final_answer",
                "최종 답변을 제출하고 종료합니다. 반드시 마지막에 한 번 호출하세요. status: answered(근거 있음) / cannot_confirm(제공 데이터로 확인 불가) / clarify(조건 모호·역질문).",
                FINAL_ANSWER_SCHEMA,
            ),
        ]

    # ------------------------------------------------------------------ exec
    async def execute(self, name: str, args: dict) -> ToolResult:
        try:
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return ToolResult(f"알 수 없는 도구: {name}", True)
            return fn(**(args or {}))
        except TypeError as e:
            return ToolResult(f"잘못된 인자: {e}", True)
        except SQLGuardError as e:
            return ToolResult(str(e), True)
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model
            log.exception("tool %s failed", name)
            return ToolResult(f"도구 오류({type(e).__name__}): {e}", True)

    def _meta_line(self, view: str) -> tuple[str | None, str | None]:
        m = self.meta.get(view) or {}
        return m.get("korean_name"), m.get("base_date")

    def _t_table_info(self, table: str) -> ToolResult:
        if table not in VIEW_ID_NAME:
            return ToolResult(f"알 수 없는 테이블: {table}. 사용 가능: {', '.join(VIEW_ID_NAME)}", True)
        kname, base = self._meta_line(table)
        m = self.meta.get(table) or {}
        card = self.ontology.schema_card(table) if self.ontology is not None else "### " + table + "\n- " + "\n- ".join(self.rdb.columns(table))
        text = f"{card}\n기준일: {base} · 행수: {m.get('row_count')}\n판정 규칙: {SEMANTICS[table]}"
        return ToolResult(text, False, Evidence("schema", table, kname, base, "table_info", None))

    # 한 상품이 여러 행인 뷰: 행 수와 상품 수가 다르다.
    #   v_bond_kr  — 장내/장외 + 장외 판매 LOT(info_seq) 로 한 종목이 최대 4행
    #   v_fund_pub — 한 펀드가 클래스별로 여러 행
    ROW_UNIT = {
        "v_bond_kr": ("pd_no", "채권 종목"),
        "v_fund_pub": ("fund_family", "펀드(클래스 합산)"),
    }
    _COUNT_STAR = re.compile(r"count\s*\(\s*\*\s*\)", re.I)

    def _unit_warning(self, sql: str, view: str | None) -> str:
        """COUNT(*) 로 '몇 개'를 물으면 행 수가 나온다. 실제 상품 수를 함께 알려줘 오답을 막는다.

        프롬프트에 'DISTINCT 를 쓰라'고 적어두었지만 모델이 무시해 16,023(행) vs 15,837(종목) 오답이
        실제로 나왔다. 지시 대신 결과에 정답을 같이 얹는다.
        """
        if not view or view not in self.ROW_UNIT:
            return ""
        if not self._COUNT_STAR.search(sql) or "distinct" in sql.lower():
            return ""
        # Only a query whose whole result is one count can be restated as a different count.  A
        # GROUP BY returns one row per group with the dimension in the first column, and rewriting it
        # produced ValueError("Cannot specify ',' with 's'") when the group key hit the f-string's
        # thousands separator — breaking the very "펀드 패밀리 요약" pattern the prompt teaches.
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
        return (f"\n⚠ 위 숫자는 '행 수'입니다. {view} 는 한 상품이 여러 행일 수 있어 "
                f"실제 {unit} 수는 {int(n):,}개입니다. 개수를 묻는 질문이라면 {int(n):,}개로 답하세요.")

    def _true_total(self, sql: str) -> int | None:
        """How many rows the query really matches, ignoring the row cap.

        The cap exists to bound query cost, but reporting the capped number as the total is the same
        mistake the graph made with its 60-neighbour limit: '총 200건' for a filter that matches 15,837
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
        """0 이 답을 오염시키는 두 경우를 결과에 바로 붙인다.

        주최측은 값이 없을 때 0 이나 NULL 을 그대로 두었다고 공지했다. 프롬프트에 '0 을 값으로 쓰지 말라'고
        적어도 모델은 총보수 0% 상품 3개를 최저 총보수로 답했고(실제 MDY 는 0.24%), 조건 표기가 어긋나
        COUNT 가 0 으로 나온 것을 그대로 '0개'라고 답했다. 결과가 나온 시점에 경고하는 편이 확실하다.
        """
        out = ""
        if len(rows) == 1 and len(columns) == 1:
            v = rows[0].get(columns[0])
            if isinstance(v, (int, float)) and v == 0 and re.search(r"count\s*\(", sql, re.I):
                out += ("\n⚠ 0건입니다. 조건 값의 표기가 실제 데이터와 다를 수 있으니, 조건을 하나씩 빼서 "
                        "어느 조건이 0을 만드는지 확인한 뒤 답하세요. 컬럼의 실제 값은 table_info 로 확인합니다.")
        m = re.search(r"order\s+by\s+([A-Za-z_][A-Za-z0-9_]*)\s*(asc)?\b", sql, re.I)
        if m and rows:
            col = m.group(1)
            if col in columns and sum(1 for r in rows if r.get(col) == 0) and (m.group(2) or "").lower() != "desc":
                out += (f"\n⚠ 정렬 기준 {col} 에 0 이 섞여 있습니다. 0 은 값이 제공되지 않은 것이므로 순위에 넣으면 안 됩니다. "
                        f"AND {col} > 0 을 추가해 다시 조회하세요.")
        return out

    # ------------------------------------------- SQL 오류를 '가르치는' 메시지로 바꾼다
    # 이 자리는 도구 계층이지 가드가 아니다. RDB.query 는 ETL·평가 스크립트와 바로 위
    # _true_total/_unit_warning 의 내부 재질의까지 함께 쓰는 범용 접근자여서, 가드에 LLM 용
    # 안내문을 넣으면 사람이 읽을 로그와 내부 재시도 경로까지 오염된다. 뷰 카탈로그를 알고
    # 있고 문자열이 그대로 모델에게 전달되는 곳은 이 계층이다.
    #
    # 근거(G-007): "no such column: pd_ticker" 는 무엇이 틀렸는지만 말하고 무엇이 맞는지는
    # 말하지 않는다. 모델은 다시 추측할 수밖에 없고, 추측이 반복되면 8라운드가 사라진다.
    _NO_COLUMN = re.compile(r"no such column:\s*(?:(?P<alias>[A-Za-z_]\w*)\.)?(?P<col>[A-Za-z_]\w*)", re.I)
    _NO_TABLE = re.compile(r"no such table:\s*(?P<tbl>[A-Za-z_]\w*)", re.I)
    # 컬럼 목록을 SQL 로 알아내려는 시도(PRAGMA/sqlite_master)는 가드가 막는다 — 막힌 이유가
    # 아니라 대신 쓸 도구를 알려줘야 다음 라운드가 살아난다.
    _CATALOG_SQL = re.compile(r"\b(pragma|sqlite_master|sqlite_schema|describe|show\s+columns)\b", re.I)
    # 소스 시스템 접두어. 모델이 'pd_ticker' 라고 쓰고 뷰에는 'ticker' 만 있는 혼동이 실제로 났다.
    _SRC_PREFIX = re.compile(r"^(?:pd|cu|du|wu|ru|fn|ref|etf|or|zrin|fd|std|bd|crd)_", re.I)
    _MAX_COLS_SHOWN = 24   # 이보다 좁은 뷰는 전부 나열해도 컨텍스트를 해치지 않는다
    _NEAR_COLS_SHOWN = 10  # 넓은 뷰는 근접 후보만 보이고 전체 개수를 알린다

    def _view_columns(self, view: str) -> list[str]:
        cache = getattr(self, "_colcache", None)
        if cache is None:
            cache = self._colcache = {}
        if view not in cache:
            try:
                cache[view] = list(self.rdb.columns(view))
            except Exception:  # noqa: BLE001 - 카탈로그 조회 실패가 오류 메시지를 삼키면 안 된다
                cache[view] = []
        return cache[view]

    @classmethod
    def _core_name(cls, col: str) -> str:
        """접두어를 뗀 형태. 'pd_ticker' -> 'ticker', 'etf_itm_no' -> 'itm_no'."""
        return cls._SRC_PREFIX.sub("", col.lower())

    def _column_suggestions(self, wanted: str, cols: list[str], n: int = 4) -> list[str]:
        """wanted 와 가장 가까운 실제 컬럼명. 접두어 혼동을 먼저, 그다음 편집거리."""
        want, core = wanted.lower(), self._core_name(wanted)
        out: list[str] = []
        for c in cols:  # 접두어만 다른 경우를 최우선으로 올린다(G-007 의 pd_ticker -> ticker)
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
        """실패한 질의에 '무엇이 맞는지'를 덧붙인다. 더할 게 없으면 빈 문자열."""
        m = self._NO_TABLE.search(msg)
        if m:
            bad = m.group("tbl")
            near = difflib.get_close_matches(bad, list(VIEW_ID_NAME), n=1, cutoff=0.5)
            head = f"\n'{bad}' 대신 {near[0]} 를 쓰세요. " if near else "\n"
            return (head + "사용 가능한 뷰는 다음 5개뿐입니다: "
                    + ", ".join(f"{v}({VIEW_LABEL[v]})" for v in VIEW_ID_NAME)
                    + ". 컬럼 정의는 table_info 도구로 확인합니다.")
        m = self._NO_COLUMN.search(msg)
        if m:
            bad, alias = m.group("col"), m.group("alias")
            views = self._referenced_views(sql)
            if alias:  # 별칭이 붙었으면 그 별칭이 가리키는 뷰만 본다 — JOIN 질의에서 정확도가 올라간다
                am = re.search(rf"\b(v_\w+)\s+(?:as\s+)?{re.escape(alias)}\b", sql, re.I)
                if am and am.group(1) in VIEW_ID_NAME:
                    views = [am.group(1)]
            if not views:
                return ("\n질의가 참조한 뷰를 알 수 없습니다. 사용 가능한 뷰: "
                        + ", ".join(VIEW_ID_NAME) + " (컬럼 정의는 table_info).")
            best: list[str] = []
            listing: list[str] = []
            for v in views:
                cols = self._view_columns(v)
                if not cols:
                    continue
                sug = self._column_suggestions(bad, cols)
                best += [f"{v}.{c}" for c in sug[:2]]
                if len(cols) <= self._MAX_COLS_SHOWN:
                    listing.append(f"- {v} 컬럼 {len(cols)}개: " + ", ".join(cols))
                else:
                    # 넓은 뷰를 통째로 실으면 컨텍스트를 잡아먹는다. 가까운 이름 + 총 개수만 준다.
                    near = sug or difflib.get_close_matches(bad, cols, n=self._NEAR_COLS_SHOWN, cutoff=0.0)
                    listing.append(
                        f"- {v} 컬럼 {len(cols)}개 중 '{bad}' 와 가까운 이름: "
                        + ", ".join(near[: self._NEAR_COLS_SHOWN])
                        + f" (전체 목록은 table_info('{v}'))"
                    )
            out = ""
            if best:
                out += f"\n'{bad}' 대신 " + " 또는 ".join(best[:3]) + " 를 쓰세요."
            if listing:
                out += "\n" + "\n".join(listing)
            return out
        # 가드가 막은 카탈로그 조회: 막힌 이유가 아니라 대신 쓸 도구를 알려준다.
        # (컬럼명을 몰라 PRAGMA 로 알아내려는 시도는 바로 이 막다른 길에서 나온다.)
        if self._CATALOG_SQL.search(sql or ""):
            return "\n컬럼 목록·기준일·판정 규칙은 SQL 이 아니라 table_info 도구로 조회하세요. 예: table_info(table='v_etf_kr')."
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
            text += (f"\n조건에 맞는 행은 총 {total:,}건이며, 위는 그중 {r.row_count}건입니다."
                     if total is not None else
                     f"\n총 {r.row_count}건 이상 (행 수 상한에 걸렸습니다 — 정확한 개수는 COUNT 로 확인하세요)")
            text += " 개수를 묻는 질문이면 이 총계로 답하고, 나열이 필요하면 정렬 기준을 붙여 상위 N개만 뽑으세요."
        else:
            text += f"\n총 {r.row_count}건"
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
            return ToolResult("검색 결과 없음", False, Evidence("search", table, None, None, detail, 0))
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
                text += f"\n{c.name_ko}의 자회사(사전): " + ", ".join(s.name_ko for s in subs)
            if self.kg is not None:
                nb = self.kg.neighbors(c.name_ko, relation="constituentOf", limit=30)
                cons = nb.get("constituentOf", [])
                if cons:
                    text += f"\n{c.name_ko}를 편입한 상품(그래프, {len(cons)}건): " + ", ".join(x["label"] for x in cons[:15])
                elif not c.not_listed:
                    text += f"\n{c.name_ko}를 편입한 상품: 편입 데이터 없음"
            if c.note:
                text += f"\n비고: {c.note}"
        refs = [f"{h.id} {h.title}" for h in r.products[:5]]
        return ToolResult(text, False, Evidence("entity", None, None, None, f"resolve_entity('{mention}') exact={r.exact}", None, refs))

    def _t_graph_neighbors(self, entity: str, relation: str | None = None) -> ToolResult:
        if self.kg is None:
            return ToolResult("지식그래프가 로드되지 않았습니다.", True)
        totals: dict[str, int] = {}
        nb = self.kg.neighbors(entity, relation=relation, limit=60, totals=totals)
        note = ""
        if not nb and relation:
            # An empty *filtered* result is nearly always the wrong direction, not a missing entity:
            # 삼성전자 has no hasConstituent but constituentOf has 238.  Returning the relations it does
            # have — as data, not as advice — answers in this round; a hint alone was ignored and the
            # model replied "확인할 수 없습니다" while the graph held the answer.
            totals = {}
            nb = self.kg.neighbors(entity, limit=60, totals=totals)
            if nb:
                note = f"('{relation}' 관계는 없어 이 엔티티의 전체 관계를 대신 보여줍니다)\n"
        if not nb:
            return ToolResult(f"'{entity}' 에 해당하는 노드/관계 없음", False, Evidence("graph", None, None, None, f"graph_neighbors('{entity}', {relation})", 0))
        lines, refs = [], []
        for rel, items in nb.items():
            # report the untruncated total — the model quotes this number verbatim as its answer
            total = totals.get(rel, len(items))
            shown = f"{total}건" + (f", 아래는 상위 {len(items)}건" if total > len(items) else "")
            lines.append(f"{rel} ({shown}): " + ", ".join(x["label"] for x in items[:40]) + (" ..." if len(items) > 40 else ""))
            refs += [x["label"] for x in items[:10]]
        return ToolResult(note + "\n".join(lines), False, Evidence("graph", None, None, None, f"graph_neighbors('{entity}', {relation})", sum(totals.values()) or sum(len(v) for v in nb.values()), refs[:30]))

    def _t_graph_query(self, sparql: str) -> ToolResult:
        if self.kg is None:
            return ToolResult("지식그래프가 로드되지 않았습니다.", True)
        try:
            rows = self.kg.query(sparql, limit=100)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"SPARQL 오류: {e}", True)
        if not rows:
            return ToolResult("(0건)", False, Evidence("graph", None, None, None, sparql.strip(), 0))
        lines = [" | ".join(f"{k}={v}" for k, v in r.items()) for r in rows[:60]]
        refs = [next(iter(r.values())) for r in rows[:20]]
        return ToolResult("\n".join(lines) + f"\n총 {len(rows)}건", False, Evidence("graph", None, None, None, sparql.strip(), len(rows), refs))

    def _t_search_documents(self, query: str, k: int = 4) -> ToolResult:
        if self.docs is None or len(self.docs) == 0:
            return ToolResult("외부 문서 없음", False)
        hits = self.docs.search(query, k=int(k))
        if not hits:
            return ToolResult("관련 문서 없음", False, Evidence("docs", None, None, None, f"search_documents('{query}')", 0))
        lines, refs = [], []
        for c in hits:
            lines.append(f"### {c.title} › {c.heading} (출처: {c.source or '내부 정리'}, 기준 {c.as_of or '-'})\n{c.text}")
            refs.append(f"{c.title} › {c.heading}")
        return ToolResult("\n\n".join(lines), False, Evidence("docs", None, None, hits[0].as_of or None, f"search_documents('{query}')", len(hits), refs))

    def _t_final_answer(self, status: str = "answered", answer: str = "", evidence_refs: list | None = None) -> ToolResult:
        return ToolResult(json.dumps({"status": status, "answer": answer, "evidence_refs": evidence_refs or []}, ensure_ascii=False))
