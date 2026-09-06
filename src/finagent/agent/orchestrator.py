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
from ..ontology.loader import normalize_text as _nrm
from .answer import CANNOT_PREFIX, Evidence, fallback_answer, format_evidence, format_trace, normalize_answer
from .llm import LLMClient, ToolCall
from .routing import prefetch as route_prefetch
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


# ---------------------------------------------------------------- 상품군 범위 감지 (온톨로지 주도)
# 어느 상품군이 무엇을 갖는가 하는 '사실'은 온톨로지가 갖는다(클래스 계층, fp:hasConstituent 의
# 정의역, 컬럼 속성의 fp:sourceTable). 여기 있는 것은 그 사실이 아니라 **한국어 표기 변형**뿐이다:
# rdfs:label 이 '해외ETF' 라고만 적혀 있어서 '해외 상장 ETF'·'미국 ETF' 라는 말버릇을 못 잡는다.
# 표기가 늘면 이 표에 줄이 늘고, 상품군의 성질이 바뀌면 TTL 이 바뀐다 — 섞이면 안 된다.
_GROUP_ALIASES = {
    "해외상장etf": "ForeignETF", "미국etf": "ForeignETF", "미국상장etf": "ForeignETF",
    "해외상장etn": "ForeignETN", "국내상장etf": "DomesticETF", "국내상장etn": "DomesticETN",
    "공모형펀드": "PublicFund", "사모형펀드": "PrivateFund", "원화채권": "DomesticBond",
}


def product_groups(question: str, ontology) -> list:
    """The product groups (ontology classes) the question restricts itself to, most specific only.

    '해외 ETF' must resolve to ForeignETF and *not* also to the bare 'ETF' that sits inside it — the
    span-containment pass is what keeps '해외 ETF' from inheriting 국내ETF's holdings.
    """
    if ontology is None or getattr(ontology, "graph", None) is None:
        return []
    forms: list[tuple[str, object]] = []
    for ci in ontology.descendants("Product"):
        if ci.label_ko:
            forms.append((_nrm(ci.label_ko), ci))
        if re.fullmatch(r"[A-Z]{2,4}", ci.name):  # fp:ETF / fp:ETN — 로컬명이 곧 한국어 표기다
            forms.append((ci.name.lower(), ci))
    for surf, cname in _GROUP_ALIASES.items():
        ci = ontology.class_info(cname)
        if ci is not None:
            forms.append((surf, ci))
    q = _nrm(question)
    spans: list[tuple[int, int, object]] = []
    for surf, ci in forms:
        if len(surf) < 2:
            continue
        i = q.find(surf)
        while i >= 0:
            spans.append((i, i + len(surf), ci))
            i = q.find(surf, i + 1)
    out, seen = [], set()
    for a, b, ci in sorted(spans):
        if any(x <= a and b <= y and (y - x) > (b - a) for x, y, _ in spans):
            continue  # 더 긴 표기에 포함된 조각('해외ETF' 안의 'ETF')은 버린다
        if ci.iri not in seen:
            seen.add(ci.iri)
            out.append(ci)
    return out


def group_name(ci) -> str:
    """'국내ETF' / '상장지수증권(ETN)' — the acronym classes' Korean labels are the formal names."""
    return f"{ci.label_ko}({ci.name})" if re.fullmatch(r"[A-Z]{2,4}", ci.name) else ci.label_ko


def group_views(groups, ontology) -> dict:
    """{class iri: {view, ...}} — a group's own source view, or its subclasses' when it is abstract."""
    out = {}
    for ci in groups:
        views = {d.source_table for d in ontology.descendants(ci.name) if d.source_table}
        if views:
            out[ci.iri] = views
    return out


# ---------------------------------------------------------------- 정렬 기준 감지
# "…기준", "…이 가장 높은/낮은/큰/작은", "TOP N", "상위/하위 N" 처럼 순위를 요구하는 표현에서
# 기준이 되는 값 표현을 뽑는다. 뽑기만 하고 판단은 온톨로지(resolve_measure)가 한다.
_CRITERION = [
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*기준(?![일가])"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:이|가)\s*가장\s*(?:높은|낮은|큰|작은|많은|적은|빠른|긴|짧은)"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:이|가)\s*(?:높은|낮은|큰|작은|많은|적은)\s*순"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:상위|하위)\s*\d+"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:순으로|순위)"),
]
# 값이 아니라 대상을 가리키는 말. 이것들이 기준으로 잡히면 pd_nm·product_type 같은 문자열 컬럼에
# 걸려 "정렬 기준은 상품명" 같은 헛소리를 주입한다.
_CRIT_STOP = {"상품", "종목", "가지", "개수", "건수", "곳", "것", "기업", "회사", "펀드", "etf", "etn",
              "채권", "상품군", "항목", "기준", "이름", "정보", "데이터", "조건", "구분", "현재", "지금"}
# 기준 표현 앞에 붙는 기간 수식어 — '1년' '5년' '최근 3개월' '연초대비' 는 뒤 명사와 한 덩어리다.
_PERIOD_TOKEN = re.compile(r"^(?:최근)?\s*(?:\d+\s*(?:일|주|주일|개월|달|년)|연초\s*대비|ytd)$", re.I)
# 관형형·용언 어미로 끝나는 말은 값의 이름이 아니라 앞 명사를 꾸미는 말이다. '판매중인 상품 기준으로'
# 의 '판매중인' 이 is_on_sale 을 정렬 기준으로 끌고 왔고, '서로 다른 종목' 의 '다른' 도 그랬다.
# 실제 값 이름(총보수·순자산·수익률·표면금리·분배금·비중·만기·변동성·추적오차)은 이 어미로 끝나지 않는다.
_ADNOMINAL = re.compile(r"(?:인|한|는|된|던|른|은|을|를)$")
_COUNT_ASKED = re.compile(r"(?:top|상위|하위)\s*(\d+)|(\d+)\s*개(?!월)|(\d+)\s*(?:종목|건|가지)", re.I)


def criterion_phrases(question: str) -> list[str]:
    """Candidate measure phrases for a ranking question — [] when the question ranks nothing.

    The regexes capture greedily up to the ranking word, so the capture usually swallows the whole
    clause ('국내 ETF와 공모펀드를 1년 수익률').  Which part of the tail is the measure is not knowable
    from surface form alone: in '1년 수익률' it is the last two tokens, in '순자산 규모' the *first* of
    the two ('규모' alone resolves to a USD AUM column that has nothing to do with a 공모펀드
    question).  So the tail is offered as several candidates and the ontology picks — the caller
    keeps whichever resolves best.  Longest window first, then the individual tokens.
    """
    q = question or ""
    for pat in _CRITERION:
        for m in pat.finditer(q):
            toks = [t.strip("의은는이가을를도와과및,") for t in re.split(r"\s+", m.group(1).strip())]
            toks = [t for t in toks if t][-3:]
            if not toks:
                continue
            cands = [" ".join(toks[-n:]) for n in range(len(toks), 0, -1)] + toks[:-1]
            out, seen = [], set()
            for c in cands:
                n, last = _nrm(c), _nrm(c.split()[-1])
                # 끝 낱말로 판정한다: 한국어는 수식이 앞에 붙으므로 구의 정체는 마지막 명사가 정한다.
                # '판매중인 상품' 은 상품을 가리키는 말이지 값의 이름이 아니다.
                if len(n) < 2 or n in seen or n in _CRIT_STOP or last in _CRIT_STOP or _ADNOMINAL.search(n):
                    continue
                seen.add(n)
                out.append(c)
            if out:
                return out
    return []


_STATUS_RANK = {"resolved": 4, "period_missing": 3, "ambiguous": 2, "weak": 1, "none": 0}


def requested_count(question: str) -> int | None:
    m = _COUNT_ASKED.search(question or "")
    if not m:
        return None
    n = next((g for g in m.groups() if g), None)
    return int(n) if n else None


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


# ------------------------------------------------- 조회 없이 낸 답변을 되돌리는 강제 재시도
# HCX-005 는 도구를 한 번도 부르지 않고 사전학습 지식만으로 답해 버리는 경우가 있다. 같은 질문을
# 5회씩 실측한 결과 그렇게 나온 답은 예외 없이 틀렸다:
#   G-034 — 5/5 가 도구 0회로 3.3초에 끝나고 운용사를 약칭 'Invesco' 로 답했다.
#           DB 의 값은 'Invesco Capital Management LLC' 라 공식 명칭이 아니어서 오답.
#   G-010 — 5회 중 도구를 부른 3회는 정답, 부르지 않은 2회는 DB 에 존재하지도 않는 상품명과
#           수치를 통째로 지어냈다(순수 환각).
# 도구를 부르면 맞고 안 부르면 틀린다는 관찰에 예외가 없었으므로, 조회를 한 번도 하지 않은 채
# 제출된 final_answer 는 받지 않고 한 라운드를 더 돌린다. 채점 기준의 [환각 방지] 항목이
# "데이터에 없는 내용은 추출하지 않음" 을 요구한다.
#
# '조회 도구'의 정의는 final_answer 를 뺀 나머지 전부다. 부정 조건(name != "final_answer")
# 대신 명시적 허용 목록으로 적은 이유는, 모델이 존재하지 않는 이름의 도구를 부른 경우를 조회로
# 세면 안 되기 때문이다 — 그런 호출은 실행되지 않고 "알 수 없는 도구" 오류만 돌아오므로 근거가
# 생기지 않는다. 목록이 실제 레지스트리와 어긋나면 test_lookup_tools_matches_registry 가 잡는다.
#
# table_info 는 값이 아니라 스키마를 돌려주지만 조회로 친다: 기준일·허용값처럼 table_info 만으로
# 답이 되는 문항이 있고, 여기서 빼면 그런 문항에 근거 없이 라운드가 하나 더 붙는다.
LOOKUP_TOOLS = frozenset({
    "sql_query", "table_info", "search_products", "resolve_entity",
    "graph_neighbors", "graph_query", "search_documents",
})

# 강제 재시도 1회의 실측 비용은 LLM 라운드 1회 ≈ 11초다. 다만 되돌린 뒤에는 도구를 실행하는
# 라운드와 final_answer 를 다시 받는 라운드가 더 필요하므로 세 라운드 분의 여유를 요구한다.
# 남은 예산이 이보다 적으면 되돌리지 않고 있는 답을 그대로 받는다 — 근거 없는 답 하나보다
# 응답시간(주최측 권장 60초)을 넘겨 문항을 통째로 잃는 쪽이 나쁘다.
FORCED_LOOKUP_MIN_REMAINING_SEC = 33.0

# final_answer 호출에 되돌려 주는 tool_result. OpenAI/CLOVA 프로토콜은 이 턴의 tool_call 마다
# 결과를 요구하므로, 대화를 이어가려면 반드시 채워야 한다(비우면 다음 요청이 400 으로 떨어진다).
_NO_LOOKUP_TOOL_RESULT = (
    "제출이 거부되었습니다. 이번 질문에서 데이터 조회 도구를 한 번도 호출하지 않았습니다. "
    "조회 결과에 근거하지 않은 답변은 환각으로 처리됩니다. 데이터를 먼저 조회한 뒤 "
    "final_answer 를 다시 호출하세요."
)

_UNRUN_TOOL_RESULT = (
    "이 호출은 같은 턴에 final_answer 가 함께 있어 실행되지 않았습니다. 필요하면 지금 다시 호출하세요."
)

# 재시도 지시는 '사용 시점의 사실'로 준다. 이 프로젝트에서 미리 준 프롬프트 규칙은 무시됐고
# (4/4), 그 순간의 사실로 준 지시는 지켜졌다(5/5). 마지막 문단은 확인불가 문항이 이 재시도
# 때문에 답을 지어내기 시작하는 부작용을 막는다 — 조회 후의 거절은 유지되어야 한다.
_NO_LOOKUP_RETRY = (
    "[답변 전 데이터 조회 필수] 방금 제출한 final_answer 는 받지 않았습니다. 이번 질문에서 "
    "데이터 조회 도구를 한 번도 부르지 않았기 때문입니다.\n"
    "답변 전에 최소 한 번은 데이터를 조회해야 합니다. 상품의 공식 명칭·수치·날짜는 반드시 조회 "
    "결과에서 그대로 가져오세요. 기억에 있는 이름을 쓰면 안 됩니다 — 이 데이터에 실린 정식 명칭은 "
    "흔히 쓰는 약칭과 다르고(예: 운용사명은 법인 정식명입니다), 기억에서 꺼낸 수치는 이 데이터의 "
    "값이 아닙니다.\n"
    "지금 할 일: 관련 뷰를 sql_query 로 조회하세요. 컬럼 이름이 불확실하면 table_info 를, 상품의 "
    "정확한 표기를 모르면 search_products 나 resolve_entity 를 먼저 부르세요. 조회 결과를 확인한 "
    "뒤 다시 final_answer 를 호출하세요.\n"
    "조회 결과가 비어 있거나 근거를 찾지 못했다면 답을 지어내지 마세요. 그럴 때는 "
    "status=cannot_confirm 을 그대로 유지하는 것이 정답입니다. 조회한 뒤의 '제공된 데이터로는 "
    "확인할 수 없습니다' 는 맞는 답이지만, 조회 없이 단정한 답은 틀린 답입니다."
)


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
        # 편입종목을 가진 상품 수. 주입문에 숫자를 적으려면 온톨로지가 아니라 데이터에서 세야 한다
        # (온톨로지는 '어느 상품군이 갖는가'를 알지 '몇 개인가'를 모른다).
        self.holding_product_count = self._count_holding_products()

    def _count_holding_products(self) -> int | None:
        view = self.ontology.holding_view() if self.ontology is not None else None
        if not view or self.tools.rdb is None:
            return None
        try:
            rows = self.tools.rdb.fetch_all(f"SELECT COUNT(DISTINCT etf_itm_no) FROM {view}")
            return int(rows[0][0])
        except Exception:  # noqa: BLE001 - a store without the view simply gets no number
            return None

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
        groups = self._groups(question)
        # 질문이 지목한 상품군 전부가 편입종목 데이터를 갖지 않는가. 이 판정의 근거는 프롬프트의
        # 문장이 아니라 온톨로지다(fp:hasConstituent 의 정의역 = fp:DomesticETF).
        holding_blind = bool(
            groups and _HOLDING_WORD.search(question)
            and self.ontology is not None
            and not any(self.ontology.class_has_holdings(ci.name) for ci in groups)
        )
        holder_label = "/".join(c.label_ko for c in self.ontology.holding_classes()) if self.ontology is not None else ""
        for c in comps:
            subs = self.lexicon.subsidiaries_of(c.id)
            info = f"- 기업 '{c.name_ko}' ({c.name_en}{', 비상장' if c.not_listed else ''})"
            if c.parent:
                p = self.lexicon.company(c.parent)
                info += f", 모회사: {p.name_ko if p else c.parent}"
            if subs:
                info += ", 자회사: " + ", ".join(s.name_ko for s in subs)
            if self.tools.kg is not None and not holding_blind:
                # ★ 건수는 totals 로 받는다. 예전에는 limit=100 으로 자른 목록의 len 을 적어서
                #   삼성전자(실제 238건)가 언제나 '100건' 으로 주입됐다 — 상한이 답이 된 것이다.
                totals: dict[str, int] = {}
                self.tools.kg.neighbors(c.name_ko, relation="constituentOf", limit=1, totals=totals)
                n = totals.get("constituentOf", 0)
                info += f", 편입 {holder_label or '상품'} {n}건(그래프)" if n else ", 편입 상품 없음(그래프)"
            if c.note:
                info += f". 비고: {c.note}"
            lines.append(info)
        if holding_blind:
            lines.append(self._no_holdings_note(groups, holder_label))
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
        crit_phrase, crit = self._criterion_lines(question, groups)
        lines += crit
        if invalid:
            lines.insert(0, "- ⚠ 온톨로지 위반: " + " / ".join(invalid))
        text = "[사전 분석 — 참고용, 최종 판단은 도구로 확인]\n" + ("\n".join(lines) if lines else "- 특기사항 없음")
        steps.append(
            "사전분석: 상품언급 " + (", ".join(mentions) if mentions else "없음")
            + (f"; 기업 {', '.join(c.name_ko for c in comps)}" if comps else "")
            + (f"; 테마 {', '.join(t.label_ko for t in themes)}" if themes else "")
            + (f"; 상품군 {', '.join(group_name(ci) for ci in groups)}" if groups else "")
            + ("; 편입데이터 없는 상품군" if holding_blind else "")
            + (f"; 정렬기준 '{crit_phrase}'" if crit_phrase else "")
            + (f"; 온톨로지 위반 {len(invalid)}건" if invalid else "")
            + (f"; 미존재 상품 {len(unresolved)}건" if unresolved else "")
        )
        return PreAnalysis(text, invalid, mentions, steps, unresolved)

    # ------------------------------------------------------------ pre-analysis helpers
    def _groups(self, question: str) -> list:
        try:
            return product_groups(question, self.ontology)
        except Exception:  # noqa: BLE001 - a hint must never take the answer down with it
            log.exception("product group detection failed")
            return []

    def _no_holdings_note(self, groups, holder_label: str) -> str:
        """The fact to inject instead of a holdings count, when the scope has no holdings data.

        Every number here is read, not remembered: the class from fp:hasConstituent's domain, the
        view from its fp:sourceTable, the product count from that view.
        """
        view = self.ontology.holding_view() or "편입종목 뷰"
        named = ", ".join(f"{group_name(ci)}({', '.join(sorted(v))})" for ci, v in
                          ((ci, group_views([ci], self.ontology).get(ci.iri) or {"해당 뷰 없음"}) for ci in groups))
        cnt = f" {self.holding_product_count:,}종목" if self.holding_product_count else ""
        return (
            f"- ⚠ 편입종목 데이터 범위: {named} 에는 보유·편입 종목 데이터가 없습니다. "
            f"편입종목은 {view} 하나뿐이고 대상은 {holder_label or '국내ETF'}{cnt} 입니다"
            f"(온톨로지 fp:hasConstituent 의 정의역 = {', '.join(c.name for c in self.ontology.holding_classes())}). "
            f"따라서 이 상품군으로 한정된 'X를 보유한 …' 질문은 '0건'이 아니라 **제공된 데이터로 확인할 수 없음**입니다. "
            f"이름이나 테마가 비슷한 상품을 보유 상품인 것처럼 제시하면 오답입니다."
        )

    def _pick_criterion(self, question: str):
        """(phrase, resolution) for the candidate the ontology resolves best, or None."""
        best = None
        for phrase in criterion_phrases(question):
            res = self.ontology.resolve_measure(phrase)
            if res.status == "none":
                continue
            score = max((h.score for h in res.hits + res.other_periods), default=0)
            key = (_STATUS_RANK[res.status], score, len(phrase))
            if best is None or key > best[0]:
                best = (key, phrase, res)
        return None if best is None else (best[1], best[2])

    def _criterion_lines(self, question: str, groups) -> tuple[str | None, list[str]]:
        """(phrase, lines) for the ranking criterion — (None, []) when the question ranks nothing.

        Silence is the default and the common case: 순위를 묻지 않는 질문에는 한 줄도 붙지 않는다.
        """
        if self.ontology is None:
            return None, []
        try:
            picked = self._pick_criterion(question)
            if picked is None:
                return None, []
            phrase, res = picked
            out: list[str] = []
            if res.status == "resolved":
                # 질문이 상품군을 지목했으면 그 뷰의 컬럼을 앞에 둔다. '국내 ETF 중 분배율' 에
                # v_fund_pub 컬럼이 먼저 오면 첫 줄만 읽은 모델이 엉뚱한 뷰로 간다.
                scope = {v for s in group_views(groups, self.ontology).values() for v in s}
                hits = sorted(res.hits, key=lambda h: (h.prop.table not in scope, -h.score)) if scope else res.hits
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko}"
                                 + (f", 단위 {h.prop.unit}" if h.prop.unit else "") + ")" for h in hits[:6])
                out.append(f"- 정렬 기준 '{phrase}' → 온톨로지 라벨 조회 결과 {cols}. 이 컬럼으로 정렬하고 NULL·0 은 후보에서 제외하세요.")
                out += self._missing_in_groups(res, groups)
            elif res.status == "period_missing":
                have = ", ".join(sorted({f"{h.prop.table}.{h.prop.column}({h.prop.label_ko})"
                                         for h in res.other_periods if h.period})[:6])
                out.append(
                    f"- 정렬 기준 '{phrase}': 요청한 기간({res.period})의 컬럼이 온톨로지에 없습니다. "
                    f"같은 개념의 다른 기간 컬럼: {have or '없음'}. 기간을 바꿔 답하려면 그 사실을 답변에 밝히세요."
                )
            elif res.status == "ambiguous":
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko})" for h in res.hits[:10])
                out.append(
                    f"- 정렬 기준 '{phrase}' 는 기간·대상이 지정되지 않아 온톨로지에서 "
                    f"{len({(h.prop.table, h.prop.column) for h in res.hits})}개 컬럼에 대응합니다: {cols}. "
                    f"질문 맥락에 맞는 컬럼을 고르고 그 기준을 답변 첫 문장에 밝히세요. "
                    f"맥락으로도 정해지지 않으면 임의로 고르지 말고 되물으세요."
                )
            elif res.status == "weak":
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko}"
                                 + (f", 단위 {h.prop.unit}" if h.prop.unit else "") + ")" for h in res.hits[:4])
                out.append(
                    f"- 정렬 기준 '{phrase}' 는 온톨로지 라벨과 정확히 일치하는 컬럼이 없습니다. 근접 후보: {cols}. "
                    f"**라벨이 없다고 데이터에 없는 것이 아닙니다** — table_info 로 확인한 뒤 판단하세요."
                )
            if out:
                n = requested_count(question)
                if n is not None:
                    out.append(f"- 요청 개수: {n}개 (LIMIT {n})")
            return (phrase if out else None), out
        except Exception:  # noqa: BLE001
            log.exception("criterion resolution failed")
            return None, []

    def _missing_in_groups(self, res, groups) -> list[str]:
        """Which of the named product groups do *not* carry the resolved criterion.

        Three rungs, strongest first.  An absence is only asserted on the bottom rung, when neither
        a same-concept-other-period column nor a loose label/comment probe turns anything up in that
        view — because a label miss is not proof of absence (v_etf_gl.du_last_aum is labelled 'AUM',
        not '순자산', and reporting 해외ETF as having no 순자산 would be a wrong 'cannot confirm').
        """
        have = set(res.views)
        out: list[str] = []
        for ci in groups:
            for view in sorted(group_views([ci], self.ontology).get(ci.iri, set())):
                if view in have:
                    continue
                other = sorted({f"{h.prop.column}({h.prop.label_ko})" for h in res.other_periods
                                if h.prop.table == view and h.period})
                if other:
                    out.append(
                        f"  ⚠ {group_name(ci)}({view})에는 '{res.period} {res.base}' 컬럼이 없습니다 — 같은 개념은 "
                        f"{', '.join(other[:4])} 뿐입니다. 순위 비교에서 제외하고 그 사실을 답변에 밝히세요(값을 0으로 채우지 마세요)."
                    )
                    continue
                probe = self.ontology.measure_candidates_in(res.base, view)
                if probe:
                    cand = ", ".join(f"{h.prop.column}({h.prop.label_ko})" for h in probe[:3])
                    out.append(
                        f"  · {group_name(ci)}({view})에는 라벨이 같은 컬럼이 없습니다. 관련 후보: {cand}. "
                        f"없다고 단정하지 말고 table_info 로 확인하세요."
                    )
                else:
                    out.append(f"  ⚠ {group_name(ci)}({view})에는 이 기준에 해당하는 컬럼이 없습니다. 순위 비교에서 제외하고 그 사실을 밝히세요.")
        return out

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

        # ---- 라운드 0: 검색 실행 계획(agent/routing.py)
        # 사전분석이 "이 저장소가 필요하다"고 판정한 것만 미리 조회해서, 그 결과를 첫 프롬프트에
        # 사실로 싣는다. 프롬프트로 "그래프를 쓰라"고 지시하는 것과 달리 이것은 지켜진다
        # (이 프로젝트의 측정: 사용 시점의 사실 5/5 준수, 사전 규칙 4/4 무시).
        # 실패하거나 예산을 넘기면 아무것도 싣지 않고 기존 도구 루프가 그대로 이어진다.
        prefetch_block = ""
        try:
            pf = await route_prefetch(
                question, self.tools, resolver=self.resolver, lexicon=self.lexicon,
                ontology=self.ontology, mentions=list(pre.mentions),
                unresolved=[m for m, _ in pre.unresolved],
                forward_looking=asks_forward_looking(question, self.base_year),
            )
            steps += pf.steps
            prefetch_block = pf.block
            # 라운드 0 근거는 retrieved_context 에 넣지 않는다: 도구 루프가 같은 조회를 다시 하는
            # 경우가 대부분이라 [근거] 가 중복으로 찍힌다(근거 개수를 세는 기존 테스트가 이를 잡는다).
            # 라운드 0 이 무엇을 왜 조회했는지는 think_trace 의 '라운드 0' 줄이 그대로 보여준다.
            emit("prefetch", text=pf.block, ran=pf.ran, seconds=pf.seconds,
                 elapsed=time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001 - 사전검색은 최적화일 뿐 의존성이 아니다
            log.exception("retrieval routing failed")
            steps.append(f"라운드 0 라우팅 실패({type(e).__name__}) — 기존 흐름으로 진행")

        conv = self.llm.start(self.system_prompt, self.specs,
                              f"질문: {question}\n\n{pre.text}"
                              + (f"\n\n{prefetch_block}" if prefetch_block else ""))
        last_text = ""
        nudged = False
        # key -> (지금까지 몇 번째 호출인지, 첫 실행 결과)
        seen_calls: dict = {}
        repeat_nudged = False
        # 이번 질의에서 실제로 "실행된" 데이터 조회 호출 수. 같은 턴에 final_answer 와 섞여 나와
        # 실행되지 않은 호출은 세지 않는다 — 결과를 못 봤으면 근거가 없는 것과 같다.
        lookups = 0
        forced_lookup = False                 # 강제 재시도는 질문당 한 번뿐(무한 루프 금지)
        rejected_final: dict | None = None    # 되돌린 답변. 재시도가 빈손으로 끝나면 이것을 쓴다.
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
                remaining = deadline - time.monotonic()
                if lookups == 0 and not forced_lookup:
                    # 되돌릴 여유가 있는지 라운드와 시간을 모두 본다. 재시도가 쓸모 있으려면
                    # 도구를 실행할 라운드가 최소 하나 남아 있어야 한다(rnd+1 <= max_tool_rounds).
                    if rnd >= self.settings.max_tool_rounds or remaining < FORCED_LOOKUP_MIN_REMAINING_SEC:
                        steps.append(
                            f"라운드 {rnd}: 데이터 조회 0회지만 예산 부족(잔여 {remaining:.0f}s, "
                            f"최대 {self.settings.max_tool_rounds}라운드) → 강제 재시도 생략하고 답변 수용"
                        )
                        break
                    forced_lookup = True
                    rejected_final, final = final, None
                    conv.add_tool_results(
                        [(fa.id, _NO_LOOKUP_TOOL_RESULT, True)]
                        + [(c.id, _UNRUN_TOOL_RESULT, True) for c in others]
                    )
                    conv.add_user(_NO_LOOKUP_RETRY)
                    steps.append(f"라운드 {rnd}: 데이터 조회 0회 상태의 final_answer → 제출 거부, 조회 후 재제출 지시")
                    emit("forced_lookup_retry", round=rnd,
                         status=str(rejected_final.get("status") or "?"),
                         elapsed=time.monotonic() - t0)
                    continue
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
            # 여기까지 왔으면 이 턴의 호출은 아래 gather 에서 전부 실행된다 — 실행되는 것만 센다.
            lookups += sum(1 for c in turn.tool_calls if c.name in LOOKUP_TOOLS)
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

        # 되돌린 답변을 잃지 않는다. 재시도 라운드가 LLM 오류·시간 초과·라운드 소진으로 아무것도
        # 내놓지 못하면, 근거는 없어도 원래 있던 답이 빈손보다 낫다.
        if final is None and rejected_final is not None:
            final = rejected_final
            steps.append("강제 재시도 후 final_answer 재호출 없음 → 처음 제출된 답변을 사용")
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
