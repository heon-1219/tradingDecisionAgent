"""Per-question orchestration: deterministic pre-analysis, tool-calling loop under a time budget,
evidence + trace assembly into the competition response schema."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, field

from ..models import AnswerResponse
from .answer import CANNOT_PREFIX, Evidence, fallback_answer, format_evidence, format_trace, normalize_answer
from .llm import LLMClient, ToolCall
from .tools import ToolRegistry, ToolResult

log = logging.getLogger("finagent.orchestrator")

_GRADE_MENTION = re.compile(r"(?:신용등급|등급)\s*(?:이|가|은|는)?\s*([A-Za-z]{1,5}[+\-0]?)|\b([A-Da-d]{1,5}[+\-0]?)\s*(?:이상|이하|등급)")
_RISK_MENTION = re.compile(r"(?:위험등급|위험\s*)?(\d{1,2})\s*등급")


@dataclass
class PreAnalysis:
    text: str
    invalid: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    # 질문이 특정 상품명을 지목했는데 그런 상품이 없는 경우: (언급, 유사 후보 이름들)
    unresolved: list[tuple[str, list[str]]] = field(default_factory=list)


# 답변이 "그 상품은 없다"는 사실을 인지했는지 판정할 때 쓰는 표현들
_COUNT_QUESTION = re.compile(r"몇\s*(개|건|종목|가지|곳)|개수|건수|수는\??$|수를\s*알려")

# ---------------------------------------------------------------- 미래 예측 요구 감지
# 온톨로지의 속성을 전수 확인한 결과, **미래에 실현될 값을 담은 컬럼은 하나도 없다.** 앞을 보는
# 이름을 가진 유일한 속성은 v_bond_kr.exrt_grte_ern_r(만기보장수익률)인데 이조차 계약으로 확정된
# 값이지 예측이 아니고, mat_date·delisting_date 는 예정된 사실이다. 마스터 테이블은 모두 기준일
# 시점의 스냅샷이다. 따라서 "미래 값"을 묻는 질문은 표현을 바꾼다고 답할 수 있는 것이 아니라
# 스키마상 답이 존재하지 않는 질문이다 — 이것이 이 가드의 근거다.
# (사고 사례 G-019: "내년 예상 수익률" 에 과거 1년 수익률 du_er_1y=260.41 을 예측치로 제시)
_FORECAST_WORD = r"예상|전망|예측|추정"
# '순자산'을 값 표현에 넣은 근거: 이것도 기준일 스냅샷 값이라 미래 시점 값은 존재하지 않는다.
# 목록에 없어서 G-045("2027년 말 예상 순자산 규모")가 가드를 그대로 통과했다.
# '규모'는 값이 아니라 크기를 가리키는 말이라 단독으로 넣으면 안 된다 — 그렇게 넣었더니
# "미래 성장 규모가 큰 테마"라는 정상 질문이 '미래 … 규모'로 걸려 답변이 막혔다. 실측 결과
# G-045 는 '예상 순자산' / '향후 … 순자산' 두 경로 모두로 걸리므로 '규모' 없이도 잡힌다.
# 값 명사에 붙은 '자산 규모'만 남긴다.
_VALUE_WORD = r"수익률|수익|가격|주가|시세|배당|성과|순자산|자산\s*규모"
# '미래' 뒤에 한글이 붙으면 미래 시점이 아니라 이름이다 — 상품명 9,816건이 '미래'로 시작하고
# (미래에셋…, BNK 미래전략기술액티브 …) 운용사 이름 자체가 미래에셋이다. 접미 한글을 배제한다.
# '연말'은 과거일 수도 있어("작년 연말 수익률") 넣지 않는다. 미래를 뜻하는 '올해 말'만 받는다.
_FUTURE_WORD = r"내년|내후년|향후|앞으로|미래(?![가-힣])|장래|다음\s*해|올해\s*말"

_FORWARD_LOOKING = re.compile(
    "|".join(
        (
            # "예상 수익률", "수익률 전망" — 예측어와 값 표현이 붙어 있는 경우
            rf"(?:{_FORECAST_WORD})[^\n]{{0,10}}(?:{_VALUE_WORD})",
            rf"(?:{_VALUE_WORD})[^\n]{{0,10}}(?:{_FORECAST_WORD})",
            # "내년 수익률", "앞으로 주가" — 미래 시점 표현이 값 표현 근처에 있는 경우
            rf"(?:{_FUTURE_WORD})[^\n]{{0,15}}(?:{_VALUE_WORD})",
            rf"(?:{_VALUE_WORD})[^\n]{{0,15}}(?:{_FUTURE_WORD})",
            # 값 표현 없이 등락만 묻는 경우: "오를까요?", "떨어질까"
            r"(?:오를|올라갈|상승할|하락할|떨어질|내려갈|빠질)\s*(?:까|지|것|수|듯|가능성|여지|전망)",
            r"(?:얼마나|얼마|몇\s*%|몇\s*퍼센트)\s*(?:오를|오르|상승|하락|떨어질|내릴)",
        )
    )
)

# 만기보장수익률(v_bond_kr.exrt_grte_ern_r)과 분배/배당 주기는 실제로 데이터에 있는 값이므로
# 미래 예측 질문으로 오인하면 안 된다. 운용사명 '미래 에셋'(띄어 쓴 표기)도 마찬가지다.
# 판정 전에 지워서 값 표현·미래 표현으로 세지 않게 한다.
_FORWARD_EXEMPT = re.compile(r"(?:만기\s*)?보장\s*수익률|배당\s*주기|분배\s*주기|미래\s*에셋")

# ---- 네 자리 연도로 적은 미래 시점
# 미래를 '내년·향후'가 아니라 연도로만 적으면 _FUTURE_WORD 목록에 걸리지 않는다. 실제로 G-045
# ("KODEX 200의 2027년 말 예상 순자산 규모")가 그렇게 빠져나가 현재 순자산을 예측치로 냈다.
# 데이터 기준일의 연도보다 큰 연도는 미래 시점으로 보고 _FUTURE_WORD 자리에 치환한다.
#
# 함정: "만기가 2027년인 채권"(G-002)도 미래 연도를 말하지만 예측이 아니라 **기록된 날짜 필드**를
# 묻는다 — mat_date 는 데이터에 실제로 들어 있으므로 답할 수 있어야 한다. 그래서 연도가 만기·상환·
# 발행·상장 같은 날짜 필드 어휘에 붙어 있으면 미래 시점으로 세지 않는다. 치환한 뒤의 판정은 기존
# 규칙 그대로라, 미래 연도가 값 표현(수익률·순자산 …)과 가까이 있을 때만 예측 질문이 된다.
_DATE_FIELD_WORD = r"만기|상환|만료|발행|상장|설정|기준일|마감|청산|해지|폐지|출시|가입"
_YEAR_MENTION = re.compile(
    rf"(?:(?P<pre>{_DATE_FIELD_WORD})[^\n]{{0,6}})?(?P<year>\d{{4}})\s*년(?:\s*(?P<post>{_DATE_FIELD_WORD}))?"
)
# 스토어 meta 를 못 읽는 경로(모듈 단독 호출·테스트)용 기본값. 실제 판정에서는 Orchestrator 가
# base_date(= v_etf_kr 의 meta 기준일)의 연도를 넘긴다.
_BASE_YEAR_FALLBACK = 2026


def _mark_future_years(text: str, base_year: int) -> str:
    """기준연도보다 뒤인 연도 표기를 미래 시점 표현으로 바꾼다(날짜 필드에 붙은 연도는 제외)."""

    def repl(m: re.Match) -> str:
        if int(m.group("year")) <= base_year or m.group("pre") or m.group("post"):
            return m.group(0)
        return "향후"

    return _YEAR_MENTION.sub(repl, text)


def asks_forward_looking(question: str, base_year: int = _BASE_YEAR_FALLBACK) -> bool:
    """질문이 데이터에 존재하지 않는 '미래 값'을 요구하는가."""
    text = _mark_future_years(_FORWARD_EXEMPT.sub(" ", question or ""), base_year)
    return bool(_FORWARD_LOOKING.search(text))


# ---------------------------------------------------------------- 편입 종목 조회 범위 감지
# 편입(구성) 종목 데이터는 v_etf_constituent 하나뿐이고, 이 뷰는 국내 ETF 1,162 종목 / 22,599행
# (네이버금융 수집)만 담는다. 해외ETF 마스터(v_etf_gl)와 공모펀드(v_fund_pub)에는 보유 종목
# 컬럼 자체가 없다. 그러므로 "X 를 편입한 해외 ETF / 공모펀드"는 질의를 잘 짜면 답이 나오는
# 질문이 아니라 데이터에 그 관계가 없는 질문이다 — 판단이 아니라 스키마의 사실이다.
#   G-040: v_etf_constituent 에서 0건을 받고도 앞 라운드에 남아 있던 국내 ETF 10개를
#          '해외 ETF'라고 나열했다.
#   G-041: search_products 의 이름 유사도 결과를 '그 종목을 보유한 펀드'로 바꿔 읽었다.
#          search_products 는 어휘·의미 유사도만 주고 보유 관계는 전혀 establish 하지 않는다.
_HOLDING_WORD = re.compile(r"보유|편입|담고\s*있|담은|담긴|포함한|포함하고\s*있|포함된|구성\s*종목|들고\s*있")
# 보유 종목 데이터가 없는 상품군. '펀드'는 그 자체로 범위 한정어이므로 공모·사모를 포괄한다.
_NO_HOLDING_SCOPE = re.compile(r"해외\s*상장\s*ETF|해외\s*ETF|미국\s*ETF|공모\s*펀드|사모\s*펀드|ETN|펀드")
# 국내 ETF 를 함께 지목했다면 답할 수 있는 부분이 있으므로 가드를 걸지 않는다. 범위어가 아예 없는
# "에코프로비엠을 편입한 ETF" 도 _NO_HOLDING_SCOPE 에 걸리지 않아 그대로 답변된다.
_HOLDING_SCOPE_OK = re.compile(r"국내\s*(?:상장\s*)?ETF")


def asks_holdings_without_data(question: str) -> bool:
    """'그 기업을 담고 있는 상품'을 묻는데, 지목한 상품군에 편입 데이터가 없는가."""
    q = question or ""
    if not _HOLDING_WORD.search(q) or not _NO_HOLDING_SCOPE.search(q):
        return False
    return not _HOLDING_SCOPE_OK.search(q)


class _Steps(list):
    """The trace list, with a tap.

    ``answer`` records its progress by appending to ``steps`` in 16 places.  Rather than touch all of
    them, the list itself forwards every entry to an optional sink so a caller (the dev console) can
    watch a question being answered live.  The competition endpoint passes no sink and behaves exactly
    as before — observation must not change what is measured.
    """

    def __init__(self, sink=None):
        super().__init__()
        self._sink = sink

    def _emit(self, item) -> None:
        if self._sink:
            try:
                self._sink({"type": "step", "text": str(item)})
            except Exception:  # noqa: BLE001 - a broken console must never break an answer
                self._sink = None

    def append(self, item) -> None:
        super().append(item)
        self._emit(item)

    def extend(self, items) -> None:
        items = list(items)
        super().extend(items)
        for i in items:
            self._emit(i)

    def __iadd__(self, items):
        self.extend(items)
        return self


def _scalar_of(text: str) -> int | None:
    """The single number a COUNT-style query returned, or None if the result isn't one number.

    ``rows_to_text`` renders it as ``cnt`` / ``240`` / ``총 1건``.  HCX has been observed running the
    right SQL, reading 240 and then writing "24개" in the answer, so the server keeps the number the
    tool actually produced and checks the answer against it.
    """
    # The tools append their own advisory lines ("⚠ 위 숫자는 행 수입니다" ...), so match the header /
    # value / count prefix and ignore whatever follows — an exact 3-line check silently gave up on
    # every result that carried a warning.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or "|" in lines[0] or "|" in lines[1] or not lines[2].startswith("총 1건"):
        return None
    val = lines[1].replace(",", "").strip()
    return int(val) if re.fullmatch(r"-?\d+", val) else None


_ACKNOWLEDGED = ("확인할 수 없", "확인되지 않", "존재하지 않", "찾을 수 없", "없습니다만", "혹시", "말씀하시", "유사한", "다른 상품")


# ---------------------------------------------------------------- 같은 호출 반복 감지
# 첫 접근이 실패했을 때 같은 호출을 그대로 다시 보내는 사고가 실제로 났다(G-015: 8라운드 동안
# 서로 다른 질의는 2개뿐이었고, 하나는 매번 'no such column: pd_no' 로 실패, 다른 하나는 매번
# 0건이었다). 루프 안에서 "지금 같은 것을 또 보냈다"고 말해 주는 곳이 없었던 것이 원인이다.
#
# ★ LLM 라운드는 어차피 소비된 뒤다. 여기서 얻는 것은 시간이 아니라 '모델이 자기가 막혔다는
#   사실을 알아차릴 기회'다 — 같은 결과를 말없이 다시 주면 모델은 새 정보를 받았다고 착각하고
#   같은 질의를 또 낸다.
#
# ★ 그래서 호출은 **막지 않고 그대로 실행하고**, 결과 뒤에 경고만 덧붙인다. 처음 판에서는 실행을
#   건너뛰고 캐시를 돌려줬는데, G-040 에서 3·4라운드에 그게 걸리면서 모델이 막다른 길에서
#   빠져나오려던 시도를 잘라 버렸고 결국 앞 라운드에 남아 있던 결과로 답을 지어냈다. 실행을
#   건너뛰어 아끼는 것은 도구 시간 몇 밀리초뿐이고, 그 대가로 탈출구 하나가 사라진다.

_SQL_ARGS = ("sql", "sparql")
# purpose 는 추적용 주석이라 결과를 바꾸지 않는다. 키에 넣으면 문구만 손봐도 '다른 호출'이 되어
# 감지기를 그냥 빠져나간다 — 막으려는 루프가 정확히 그 모양이다.
_KEY_IGNORED_ARGS = ("purpose",)
_SQL_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")


def _norm_sql(s: str) -> str:
    """두 SQL 이 '같은 질의'인지 비교하기 위한 정규화.

    코드펜스와 끝 세미콜론을 떼고, 공백을 접고(연산자·괄호·쉼표 주변 공백은 아예 제거),
    **문자열 리터럴 밖만** 소문자로 내린다. 바꾸는 것은 표기뿐이고 의미는 건드리지 않는다.

    리터럴까지 소문자로 내리면 안 된다 — SQLite 에서 crd_grd='AA-' 와 'aa-' 는 다른 질의라서,
    조건 값을 바꿔 가며 좁혀 가는 정상적인 재시도를 '반복'으로 오인하게 된다.
    """
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", (s or "").strip(), flags=re.I).strip().rstrip(";")
    s = re.sub(r"\s+", " ", s).strip()

    def _plain(seg: str) -> str:
        # 'a = 1' 과 'a=1', 'in (1, 2)' 와 'in(1,2)' 는 같은 질의다. 식별자끼리는 붙지 않는다
        # (공백을 지우는 것은 구두점·연산자에 '붙은' 공백뿐이다).
        return re.sub(r"\s*([^\w\s가-힣])\s*", r"\1", seg.lower())

    out, last = [], 0
    for m in _SQL_LITERAL.finditer(s):
        out.append(_plain(s[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_plain(s[last:]))
    return "".join(out)


def _call_key(call: ToolCall):
    """'같은 호출'의 정의 = 도구 이름 + (purpose 를 뺀) 인자들의 정규화 형태.

    더 느슨하게(예: 모든 인자를 casefold·특수문자 제거) 잡으면 서로 다른 검색어를 한 호출로
    묶어 멀쩡한 시도를 막는다. 오탐 1건의 손해(엉뚱한 이전 결과를 돌려주고 '막혔다'고 말함)가
    미탐 1건의 손해(지금과 동일한 상태)보다 크므로 엄격한 쪽으로 정의했다.
    SQL 만 예외적으로 완화한 이유는 위 _norm_sql 참고 — 대소문자·들여쓰기만 바꿔 같은 질의를
    다시 보내는 것이 실제로 관찰된 형태이기 때문이다.
    """
    if call.name == "final_answer":
        return None  # 종료 호출은 중복 판정 대상이 아니다
    args = call.input or {}
    parts = []
    for k in sorted(args):
        if k in _KEY_IGNORED_ARGS:
            continue
        v = args[k]
        t = json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
        parts.append((k, _norm_sql(t) if k in _SQL_ARGS else re.sub(r"\s+", " ", t).strip()))
    return (call.name, tuple(parts))


def _looks_empty(res: ToolResult) -> bool:
    ev = res.evidence
    if ev is not None and ev.row_count == 0:
        return True
    c = res.content or ""
    return "(0건)" in c or "검색 결과 없음" in c or "관련 문서 없음" in c


def _repeat_note(n: int, res: ToolResult) -> str:
    """반복 호출에 붙이는 안내. 무엇에 막혔는지에 따라 다음 수를 구체적으로 지정한다."""
    if res.is_error:
        nxt = ("위 오류 메시지가 알려준 실제 컬럼·뷰 이름으로 SQL 을 고치세요. "
               "이름이 확실하지 않으면 table_info 로 컬럼 목록부터 확인하세요.")
    elif _looks_empty(res):
        nxt = ("조건을 하나씩 빼면서 어느 조건이 결과를 0건으로 만드는지 찾으세요. "
               "값의 표기가 어긋났을 수 있으니 table_info 로 허용값을, resolve_entity·search_products 로 "
               "실제 표기를 확인한 뒤 다시 거세요.")
    else:
        nxt = "이미 확보한 결과입니다. 이 값을 그대로 쓰고 남은 라운드는 아직 확인하지 않은 부분에 쓰세요."
    return (f"\n\n⚠ 이 호출은 이번 질문에서 {n}번째 동일한 호출입니다. 위 결과는 다시 실행해서 받은 "
            f"것이며 처음과 같습니다 — 같은 입력은 몇 번을 보내도 결과가 달라지지 않습니다. "
            f"다른 접근이 필요합니다: {nxt}")


class Orchestrator:
    def __init__(self, settings, llm: LLMClient, tools: ToolRegistry, resolver, ontology, lexicon, system_prompt: str):
        self.settings, self.llm, self.tools = settings, llm, tools
        self.resolver, self.ontology, self.lexicon = resolver, ontology, lexicon
        self.system_prompt = system_prompt
        self.specs = tools.specs()
        self.base_date = (tools.meta.get("v_etf_kr") or {}).get("base_date") or "2026-08-21"

    @property
    def base_year(self) -> int:
        """데이터 기준연도. '2027년' 같은 표기가 미래인지 판정하는 기준이다(스토어 meta 에서 온다)."""
        m = re.match(r"(\d{4})", self.base_date or "")
        return int(m.group(1)) if m else _BASE_YEAR_FALLBACK

    # ------------------------------------------------------------ pre-analysis
    def pre_analyze(self, question: str) -> PreAnalysis:
        from ..kb.entity import extract_product_mentions

        lines: list[str] = []
        invalid: list[str] = []
        steps: list[str] = []
        mentions = extract_product_mentions(question)
        unresolved: list[tuple[str, list[str]]] = []
        for m in mentions:
            r = self.resolver.resolve(m, k=4)
            resolved = r.resolved  # 판정은 EntityResolver 가 소유한다(도구 출력과 동일 기준)
            tag = "정확 일치" if r.exact else ("유사 상품으로 해석 가능" if resolved else "일치하는 상품 없음")
            lines.append(f"- 상품 언급 '{m}': {tag}. " + ("; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in r.products[:4]) or "후보 없음"))
            if not resolved:
                cands = [h.title for h in r.products[:3]]
                unresolved.append((m, cands))
                lines.append(
                    f"  ⚠ '{m}' 이라는 상품은 데이터에 존재하지 않습니다. 위 후보는 이름이 비슷할 뿐 다른 상품이므로, "
                    f"그 중 하나를 질문의 상품인 것처럼 답하면 오답입니다. 존재하지 않음을 밝히고 후보를 제시하며 되물으세요."
                )
        whole = self.resolver.resolve(question, k=3) if not mentions else None
        comps = self.lexicon.match_companies(question)
        for c in comps:
            subs = self.lexicon.subsidiaries_of(c.id)
            info = f"- 기업 '{c.name_ko}' ({c.name_en}{', 비상장' if c.not_listed else ''})"
            if c.parent:
                p = self.lexicon.company(c.parent)
                info += f", 모회사: {p.name_ko if p else c.parent}"
            if subs:
                info += ", 자회사: " + ", ".join(s.name_ko for s in subs)
            if self.tools.kg is not None:
                cons = self.tools.kg.neighbors(c.name_ko, relation="constituentOf", limit=100).get("constituentOf", [])
                info += f", 편입 상품 {len(cons)}건(그래프)" if cons else ", 편입 상품 없음(그래프)"
            if c.note:
                info += f". 비고: {c.note}"
            lines.append(info)
        themes = self.lexicon.match_themes(question)
        for t in themes:
            cnt = ""
            if self.tools.kg is not None:
                n = len(self.tools.kg.neighbors(f"d:theme/{t.id}", relation="hasTheme_inverse", limit=10000).get("hasTheme_inverse", []))
                cnt = f", 그래프 태깅 상품 {n}건 (graph_neighbors('{t.label_ko}','hasTheme_inverse'))"
            lines.append(f"- 테마 '{t.label_ko}' ({t.label_en}; 동의어 {', '.join(t.synonyms[:6])}){cnt}")
        mgrs = self.lexicon.match_managers(question)
        if mgrs:
            lines.append("- 운용사: " + ", ".join(f"{m.canonical}(manager_name='{m.canonical}')" for m in mgrs))
        # ontology validators
        for m in _GRADE_MENTION.finditer(question):
            g = (m.group(1) or m.group(2) or "").upper()
            if re.fullmatch(r"[A-D]{1,5}[+\-0]?", g) and self.ontology is not None:
                ok, msg = self.ontology.validate_credit_grade(g)
                if not ok:
                    invalid.append(f"신용등급 '{g}'는 존재하지 않는 값입니다. {msg}")
                else:
                    ge = self.ontology.credit_grades_at_or_above(g)
                    lines.append(f"- 신용등급 '{g}' 유효. '{g} 이상' = crd_grd IN ({', '.join(repr(x) for x in ge)})")
        for m in _RISK_MENTION.finditer(question):
            v = m.group(1)
            if self.ontology is not None:
                ok, msg = self.ontology.validate_risk_grade(v)
                if not ok:
                    invalid.append(f"위험등급 '{v}등급'은 존재하지 않습니다. {msg}")
        if re.search(r"최근\s*(\d+)\s*개월", question):
            n = int(re.search(r"최근\s*(\d+)\s*개월", question).group(1))
            base = dt.date.fromisoformat(self.base_date)
            month = base.month - n
            year = base.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            cutoff = base.replace(year=year, month=month, day=min(base.day, 28))
            lines.append(f"- '최근 {n}개월' = {cutoff.isoformat()} ~ {self.base_date} (데이터 기준일 기준)")
        if whole is not None and whole.products and whole.products[0].score >= 60:
            lines.append("- 질문 전체와 유사한 상품: " + "; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in whole.products[:3]))
        if invalid:
            lines.insert(0, "- ⚠ 온톨로지 위반: " + " / ".join(invalid))
        text = "[사전 분석 — 참고용, 최종 판단은 도구로 확인]\n" + ("\n".join(lines) if lines else "- 특기사항 없음")
        steps.append(
            "사전분석: 상품언급 " + (", ".join(mentions) if mentions else "없음")
            + (f"; 기업 {', '.join(c.name_ko for c in comps)}" if comps else "")
            + (f"; 테마 {', '.join(t.label_ko for t in themes)}" if themes else "")
            + (f"; 온톨로지 위반 {len(invalid)}건" if invalid else "")
            + (f"; 미존재 상품 {len(unresolved)}건" if unresolved else "")
        )
        return PreAnalysis(text, invalid, mentions, steps, unresolved)

    # ------------------------------------------------------------ main loop
    async def answer(self, question_id: str, question: str, on_event=None) -> AnswerResponse:
        """Answer one question.

        ``on_event`` is an optional sink for progress events (dev console only).  It never changes the
        answer, the tool results, or a single token sent to the model.
        """
        t0 = time.monotonic()
        budget = float(self.settings.time_budget_sec)
        deadline = t0 + budget
        steps = _Steps(on_event)
        evidences: list[Evidence] = []
        scalars: list[int] = []
        final: dict | None = None

        def emit(kind: str, **data) -> None:
            if on_event:
                try:
                    on_event({"type": kind, **data})
                except Exception:  # noqa: BLE001
                    pass

        emit("start", question_id=question_id, question=question, budget=budget)
        try:
            pre = self.pre_analyze(question)
            steps += pre.steps
        except Exception as e:  # noqa: BLE001
            log.exception("pre-analysis failed")
            pre = PreAnalysis("[사전 분석 실패]", [], [], [f"사전분석 오류: {type(e).__name__}"])
            steps += pre.steps
        emit(
            "pre_analysis",
            text=pre.text,
            mentions=list(pre.mentions),
            unresolved=[m for m, _ in pre.unresolved],
            invalid=list(pre.invalid),
            elapsed=time.monotonic() - t0,
        )

        conv = self.llm.start(self.system_prompt, self.specs, f"질문: {question}\n\n{pre.text}")
        last_text = ""
        nudged = False
        # key -> (지금까지 몇 번째 호출인지, 첫 실행 결과)
        seen_calls: dict = {}
        repeat_nudged = False
        for rnd in range(1, self.settings.max_tool_rounds + 2):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                steps.append(f"라운드 {rnd}: 시간 예산 소진 → 강제 종료")
                break
            try:
                turn = await asyncio.wait_for(conv.step(), timeout=max(3.0, remaining - 1.0))
            except asyncio.TimeoutError:
                steps.append(f"라운드 {rnd}: LLM 응답 대기 시간 초과")
                break
            if turn.stop_reason == "error":
                steps.append(f"라운드 {rnd}: LLM 오류 — {turn.text[:120]}")
                break
            if turn.stop_reason == "refusal":
                steps.append(f"라운드 {rnd}: 모델 응답 거부")
                break
            if turn.text:
                last_text = turn.text
            fa = next((c for c in turn.tool_calls if c.name == "final_answer"), None)
            if fa is not None:
                final = fa.input or {}
                others = [c for c in turn.tool_calls if c.name != "final_answer"]
                steps.append(f"라운드 {rnd}: final_answer(status={final.get('status', '?')})" + (f" + 무시된 도구 {len(others)}개" if others else ""))
                break
            if not turn.tool_calls:
                steps.append(f"라운드 {rnd}: 도구 호출 없이 텍스트 응답")
                if rnd <= self.settings.max_tool_rounds and not nudged:
                    nudged = True
                    conv.add_user("답변을 final_answer 도구로 제출하세요(status 포함).")
                    continue
                break
            if rnd > self.settings.max_tool_rounds:
                steps.append(f"라운드 {rnd}: 최대 라운드 초과")
                break
            # execute all tool calls of this turn concurrently
            for call in turn.tool_calls:
                emit("tool_call", round=rnd, name=call.name, args=call.input or {},
                     elapsed=time.monotonic() - t0)
            tool_t0 = time.monotonic()
            # 이미 낸 적 있는 호출도 그대로 실행하고, 결과 뒤에 '몇 번째 같은 호출인지'만 덧붙인다.
            # 실행을 막으면 도구 시간 몇 밀리초를 아끼는 대신 모델이 막다른 길에서 빠져나올 길을
            # 하나 없앤다(G-040). LLM 라운드는 이 시점에 이미 소비됐다 — 여기서 얻는 것은
            # 시간이 아니라 모델이 '내가 막혔다'는 사실을 알아차릴 기회다(_repeat_note 참고).
            keys = [_call_key(c) for c in turn.tool_calls]
            ordn: list[int] = []                      # 이 질문에서 몇 번째 동일 호출인지(1 = 처음)
            for i, key in enumerate(keys):
                if key is None:                       # final_answer 등 판정 제외 호출
                    ordn.append(1)
                    continue
                # 이전 라운드까지의 횟수 + 이번 턴에서 앞쪽에 있던 같은 호출 수 + 자기 자신
                ordn.append(seen_calls.get(key, 0) + sum(1 for j in range(i) if keys[j] == key) + 1)
            bases = await asyncio.gather(*(self._run_tool(c) for c in turn.tool_calls))
            # 반복분은 evidence 를 떼고 돌려준다 — 근거는 첫 실행 때 이미 기록했고,
            # 다시 넣으면 retrieved_context 에 같은 [근거] 가 중복으로 찍힌다.
            results = [b if n == 1 else ToolResult(b.content + _repeat_note(n, b), b.is_error, None)
                       for b, n in zip(bases, ordn)]
            for key, n in zip(keys, ordn):
                if key is not None:
                    seen_calls[key] = n
            tool_sec = time.monotonic() - tool_t0
            packed = []
            for call, res, ncall in zip(turn.tool_calls, results, ordn):
                desc = self._describe(call)
                ev = res.evidence
                emit(
                    "tool_result", round=rnd, name=call.name, args=call.input or {},
                    ok=not res.is_error, content=res.content, seconds=tool_sec,
                    evidence=None if ev is None else {
                        "kind": ev.kind, "table": ev.korean_name or ev.table,
                        "base_date": ev.base_date, "row_count": ev.row_count,
                        "detail": ev.detail,
                    },
                    elapsed=time.monotonic() - t0,
                )
                if ncall > 1:
                    steps.append(f"라운드 {rnd}: {desc} → 동일 호출 {ncall}번째 — 재실행 + 다른 접근 안내")
                    packed.append((call.id, res.content, res.is_error))
                    continue
                if res.is_error:
                    steps.append(f"라운드 {rnd}: {desc} → 오류: {res.content[:100]}")
                else:
                    n = f", {res.evidence.row_count}건" if res.evidence and res.evidence.row_count is not None else ""
                    steps.append(f"라운드 {rnd}: {desc} → 결과 {len(res.content)}자{n}")
                    if res.evidence is not None and res.evidence.kind != "schema":
                        evidences.append(res.evidence)
                    if call.name == "sql_query":
                        sv = _scalar_of(res.content)
                        if sv is not None:
                            scalars.append(sv)
                packed.append((call.id, res.content, res.is_error))
            conv.add_tool_results(packed)
            # 3번째 동일 호출이면 마무리를 유도한다. 강제 종료가 아니라 안내인 이유:
            # (1) 2번째에 붙인 경고를 보고 방향을 트는 경우가 있고, (2) 키 정규화가 오탐했을 때
            # 문항 하나를 통째로 날리지 않기 위해서다. 질문당 한 번만 보낸다.
            if max(ordn, default=1) >= 3 and not repeat_nudged:
                repeat_nudged = True
                conv.add_user(
                    "[같은 호출 반복] 동일한 도구 호출이 3번 나왔습니다. 이 접근으로는 더 진전이 없습니다. "
                    "조건이나 도구를 바꿔 한 번만 더 시도하고, 그래도 근거를 얻지 못하면 즉시 final_answer 를 "
                    "호출하세요(근거가 없으면 status=cannot_confirm)."
                )
                steps.append(f"라운드 {rnd}: 동일 호출 3회 → 조기 마무리 안내")
            remaining = deadline - time.monotonic()
            if remaining < 14 or rnd == self.settings.max_tool_rounds:
                conv.add_user("[시간 예산 임박] 추가 도구 호출 없이 지금까지 확보한 근거로 즉시 final_answer 를 호출하세요.")
                steps.append(f"라운드 {rnd}: 시간 예산 임박 안내 (잔여 {remaining:.0f}s)")

        # ---------------------------------------------------------- compose
        if final is not None:
            status = str(final.get("status") or "answered")
            if status not in ("answered", "cannot_confirm", "clarify"):
                status = "answered"
            answer = normalize_answer(status, str(final.get("answer") or last_text))
            refs = [str(x) for x in (final.get("evidence_refs") or [])]
        elif last_text.strip():
            status = "answered" if evidences else "cannot_confirm"
            answer = normalize_answer(status, last_text)
            refs = []
            steps.append("final_answer 미호출 → 마지막 텍스트를 답변으로 사용")
        else:
            reason = "시간 초과" if time.monotonic() >= deadline else "LLM 응답 없음"
            status, answer, refs = "cannot_confirm", fallback_answer(reason), []
        if pre.invalid and status == "answered" and not any(k in answer for k in ("확인할 수 없", "존재하지 않")):
            answer = f"{CANNOT_PREFIX} {' '.join(pre.invalid)}\n\n(참고) {answer}"
            status = "cannot_confirm"
            steps.append("온톨로지 위반 조건이 있어 답변을 확인불가로 보정")
        # 존재하지 않는 상품명을 물었는데 모델이 유사 상품으로 그냥 답해버린 경우를 잡는다.
        # 답변불가 문항은 배점이 걸려 있어 프롬프트 준수에만 맡기지 않는다.
        if pre.unresolved and status == "answered" and not any(k in answer for k in _ACKNOWLEDGED):
            names = [m for m, _ in pre.unresolved]
            shown = [c for _, group in pre.unresolved for c in group[:2]]
            asked = (
                f"이름이 비슷한 상품으로는 {', '.join(shown)} 등이 있습니다. 이 중 찾으시는 상품이 있나요?"
                if shown else "찾으시는 상품명을 다시 확인해 주시겠어요?"
            )
            answer = (
                f"'{', '.join(names)}'와(과) 정확히 일치하는 상품은 제공된 데이터에서 확인할 수 없습니다. {asked}"
                f"\n\n(참고: 아래는 유사 상품 기준으로 조회한 내용입니다)\n{answer}"
            )
            status = "clarify"
            steps.append(f"존재하지 않는 상품({', '.join(names)})을 답변에 사용 → 역질문으로 보정")
        # 미래 값을 묻는 질문에 과거 실적 컬럼을 그대로 붙여 답하는 사고를 막는다.
        # 숫자 자체는 도구가 낸 진짜 값이라 숫자 교차검증은 통과한다. 틀린 것은 '라벨'이므로
        # 여기서 상태와 문구를 고쳐 과거 실적임을 분명히 하고 예측으로 읽히지 않게 한다.
        if status == "answered" and asks_forward_looking(question, self.base_year):
            answer = (
                f"{CANNOT_PREFIX} 제공된 데이터는 기준일 {self.base_date} 시점의 스냅샷이며, "
                "예상 수익률·목표가처럼 미래에 실현될 값을 담은 항목이 없습니다. "
                "과거 수익률은 미래 수익을 보장하지 않으므로 예측치로 제시할 수 없습니다."
                f"\n\n(참고) 아래는 예측이 아니라 데이터에 기록된 과거·현재 실적입니다.\n{answer}"
            )
            status = "cannot_confirm"
            steps.append("미래 예측을 요구하는 질문 → 데이터에 예측 컬럼이 없어 확인불가로 보정(과거 실적은 참고로 유지)")
        # 보유/편입 관계는 국내 ETF 에만 있다. 해외ETF·펀드·ETN 으로 범위를 좁힌 편입 질문에
        # 그럴듯한 답을 만들면 국내 ETF 목록을 해외 ETF 라고 내놓거나(G-040), 이름만 비슷한 펀드를
        # '그 종목을 보유 중'이라고 단정하게 된다(G-041). 스키마의 사실을 그대로 말한다.
        if status == "answered" and asks_holdings_without_data(question):
            answer = (
                f"{CANNOT_PREFIX} 편입 종목(구성종목) 데이터는 국내 ETF 1,162 종목(네이버금융 수집)"
                "에만 있습니다. 해외 ETF·공모펀드·사모펀드·ETN 의 보유 종목은 전달받은 데이터에 "
                "없으므로, 특정 기업을 편입했는지 여부와 그 순위는 확인할 수 없습니다."
                f"\n\n(참고) 아래는 편입 관계를 확인한 결과가 아니라 이름·조건으로 조회한 내용입니다.\n{answer}"
            )
            status = "cannot_confirm"
            steps.append("편입 데이터가 없는 상품군(해외ETF·펀드·ETN)에 대한 편입 질문 → 확인불가로 보정")
        # A count question whose single COUNT result never appears in the answer means the model
        # mis-transcribed it.  Only fires when exactly one scalar was produced, so a multi-step
        # calculation is never "corrected" against an intermediate value.
        if status == "answered" and _COUNT_QUESTION.search(question) and len(set(scalars)) == 1:
            want = scalars[0]
            seen = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", answer)}
            if want not in seen:
                answer = f"{answer}\n\n(조회 결과 확인: 해당 조건에 맞는 건수는 {want}건입니다.)"
                steps.append(f"도구가 반환한 {want} 가 답변에 없어 조회 결과를 명시")
        elapsed = time.monotonic() - t0
        steps.append(f"최종: status={status}, 근거 {len(evidences)}건, 소요 {elapsed:.1f}s")
        resp = AnswerResponse(
            question_id=question_id,
            question=question,
            retrieved_context=format_evidence(evidences, refs),
            think_trace=format_trace(steps),
            answer=answer,
        )
        emit("done", status=status, elapsed=elapsed, response=resp.model_dump())
        return resp

    async def _run_tool(self, call: ToolCall):
        return await self.tools.execute(call.name, call.input)

    @staticmethod
    def _describe(call: ToolCall) -> str:
        a = call.input or {}
        key = a.get("sql") or a.get("query") or a.get("mention") or a.get("entity") or a.get("sparql") or a.get("table") or ""
        key = re.sub(r"\s+", " ", str(key))[:140]
        return f"{call.name}({key})"
