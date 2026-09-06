"""Round 0: turn the deterministic pre-analysis into a *retrieval plan* and execute it.

Federated query, as the organisers describe it, is not "hit every store on every question" — it is
decomposing the question and routing each piece to the store that can answer it.  This module is the
router.  It decides, from the question text and the pre-analysis signals alone, **which** of the three
non-SQL stores (graph / vector / document) can contribute, runs only those, and hands the rows to the
first prompt as facts.

Why only those three, and why round 0:

* SQL is left alone.  NL2SQL has to be written against an arbitrary question, so it stays in the
  model's tool loop where it belongs.  Nothing here executes SQL.
* Graph / vector / document calls, by contrast, have parameters that are fully determined by the
  question text plus the pre-analysis — so they can be run *before* the model is asked anything.
* The project's measured rule is that **facts given at the point of use are honoured (5/5), while
  prompt rules given in advance are ignored (4/4)**.  "Use the graph" was a rule.  The graph rows
  themselves are a fact.  This module ships the fact.

Everything here is best-effort: a failing prefetch is dropped, an over-budget plan is abandoned, and
the existing tool loop runs exactly as it did before.  The fallback is the unchanged status quo.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger("finagent.routing")

# ---------------------------------------------------------------- budgets
# Total wall clock the whole round-0 plan may consume, and the cap on any single call.  These are
# deliberately small: the point of round 0 is to *remove* a model round (~11s), so it must not add a
# visible cost of its own.  Over budget -> the remaining calls are skipped, never retried.
TOTAL_BUDGET_SEC = 2.5
PER_CALL_BUDGET_SEC = 1.2
# Per-source cap on injected characters.  Truncation is reported together with the exact total, never
# instead of it — reporting a cap as if it were the total has been a real defect here twice.
MAX_CHARS = {"graph": 1200, "vector": 1300, "document": 1400}
# At most this many prefetch calls, whatever the rules say.
MAX_CALLS = 3

# ---------------------------------------------------------------- routing vocabulary
#
# GRAPH.  Only the *inverse* holding direction ("X를 보유한/편입한 상품") is routed, because that is
# the direction whose subject is extractable from surface form with confidence: the holder entity sits
# immediately before the verb.  The forward direction ('그 ETF의 구성종목은') names the product, not the
# company, and resolving it is the product-lookup problem, not a graph one.
_HOLDER_VERB = r"보유|편입|담고\s*있|담은|담긴|들고\s*있|포함한|포함하고\s*있|포함된"
_HOLDING_Q = re.compile(_HOLDER_VERB)
# The holder is the token right before the verb.  The character class stops at whitespace, so the
# capture is one word; trailing case particles are stripped afterwards.
_HOLDER = re.compile(r"([A-Za-z가-힣][A-Za-z0-9가-힣·&\.\-]{1,19})\s*(?:" + _HOLDER_VERB + ")")
_PARTICLE = re.compile(r"(?:을|를|이|가|은|는|와|과|도|의|에)$")
# Words that are never a company: they show up before the verb in scope phrases like
# '판매중인 상품 기준으로 채권에 투자하는' or '국내 ETF가 포함된'.
_NOT_A_HOLDER = {
    "상품", "종목", "펀드", "etf", "etn", "채권", "자산", "것", "기업", "회사", "지수",
    "데이터", "정보", "포트폴리오", "구성", "비중", "현재", "모두", "전체", "다음",
}

# GRAPH (themes).  Only an explicit '테마' question routes to the graph.  Theme aliases match very
# broadly — '국내', '미국', '배당' fire on a third of all questions — so an implicit theme match is not
# evidence that the question is *about* a theme.
_THEME_Q = re.compile(r"테마")
# Geography, index and structure "themes" are scope adjectives, not subjects.  A question is not a
# theme question because it says '미국' or '레버리지'.
_GENERIC_THEMES = {
    "korea", "usa", "china", "global", "japan", "europe", "emerging", "india", "vietnam",
    "indonesia", "kospi200", "kosdaq", "nasdaq100", "sp500", "leverage", "inverse",
}

# DOCUMENT.  Narrative questions only — the ones whose answer is prose that no column holds.  The
# organisers' own example ('국민성장펀드의 구조와 투자전략 동향', '…의 위험요인') is the shape being
# matched.  Deliberately tight: '무엇인가요' and '알려줘' are *not* here, because they end attribute
# lookups ('벤치마크 지수는 무엇인가요') far more often than they open a narrative one.
_DOC_Q = re.compile(
    r"위험\s*요인|리스크\s*요인|유의\s*사항|투자\s*전략|운용\s*전략|투자\s*방식|"
    r"구조와|의\s*구조|구조는|구조를|동향|배경|개요|취지|장단점|장점과\s*단점|"
    r"어떤\s*점|왜\s|무엇을\s*의미|의미하나|뜻인가|용어|설명해|설명\s*해|어떻게\s*운용"
)


@dataclass
class MentionSig:
    """One product name found in the question, as the entity resolver judged it."""

    text: str
    resolved: bool = False
    exact: bool = False
    table: str | None = None


@dataclass
class ThemeSig:
    id: str
    label: str


@dataclass
class Signals:
    """Everything the router reads.  Plain data, so the rules are testable without a store."""

    question: str
    mentions: list[MentionSig] = field(default_factory=list)
    themes: list[ThemeSig] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    holding_blind: bool = False
    forward_looking: bool = False


@dataclass
class RouteCall:
    source: str  # "graph" | "vector" | "document"
    tool: str
    args: dict
    why: str
    caution: str = ""  # extra line injected next to the result
    # The relation the answer actually depends on.  ``graph_neighbors`` falls back to dumping *every*
    # relation of a node when the requested one is empty, which is right for a model that can read the
    # detour but wrong as an injected fact: 야놀자 has no constituentOf and the fallback handed back a
    # 회사채 named 야놀자1-1 under "전체 15건" — for a question whose true answer is "편입 ETF 없음".
    expect_relation: str | None = None
    # Views whose rows cannot hold this question's answer.  v_etf_constituent rows are *other* ETFs'
    # holdings, so on a product-attribute question they crowd out the row that has the answer
    # (measured: QQQ's v_etf_gl row, the one carrying manager_name, was pushed past the cut).
    drop_views: tuple[str, ...] = ()
    # The name the question actually asked about.  A search result is a *ranking*, not an identity:
    # 'Invesco QQQ Trust' 검색의 상품 뷰 최상위는 'Invesco NASDAQ Internet ETF' 였다.  힌트라면 무해하지만
    # 사실로 주입하면 다른 상품의 값을 그 상품의 값으로 만든다.  이름이 실제로 맞는 행만 남긴다.
    match_title: str | None = None


@dataclass
class RoutePlan:
    calls: list[RouteCall] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (source, why not)

    @property
    def sources(self) -> list[str]:
        out: list[str] = []
        for c in self.calls:
            if c.source not in out:
                out.append(c.source)
        return out

    def summary(self) -> str:
        """One line naming every store and why it was or was not chosen — the federation evidence."""
        picked = {c.source: c.why for c in self.calls}
        parts = []
        for src, ko in (("graph", "그래프"), ("vector", "벡터"), ("document", "문서")):
            if src in picked:
                parts.append(f"{ko}=실행({picked[src]})")
            else:
                why = next((w for s, w in self.skipped if s == src), "해당 없음")
                parts.append(f"{ko}=생략({why})")
        return " / ".join(parts)


# ---------------------------------------------------------------- signal extraction
def holder_mentions(question: str) -> list[str]:
    """The entities a '…를 보유/편입한 …' question is asking about, in order of appearance.

    Surface extraction on purpose: '쿠팡'과 '야놀자'는 기업 사전에 없지만 두 문항 모두 정답이
    '해당 없음'이고, 그 사실을 아는 곳은 그래프뿐이다.  사전 매칭만 믿으면 그 두 문항에서 라우터가
    아예 켜지지 않는다.
    """
    out: list[str] = []
    for m in _HOLDER.finditer(question or ""):
        tok = _PARTICLE.sub("", m.group(1).strip())
        if len(tok) < 2 or tok.lower() in _NOT_A_HOLDER:
            continue
        if tok not in out:
            out.append(tok)
    return out


def is_holding_question(question: str) -> bool:
    return bool(_HOLDING_Q.search(question or ""))


def wants_narrative(question: str) -> bool:
    return bool(_DOC_Q.search(question or ""))


def theme_subject(question: str, themes: list[ThemeSig]) -> ThemeSig | None:
    """The theme a question is *about*, or None.  Requires the word 테마 and a non-generic theme."""
    if not _THEME_Q.search(question or ""):
        return None
    real = [t for t in themes if t.id not in _GENERIC_THEMES]
    if not real:
        return None
    # 여러 개면 가장 구체적인 것(라벨이 긴 것)을 고른다: '중국반도체' > '반도체'.
    return max(real, key=lambda t: len(t.label or ""))


# 상품 검색을 붙일 테이블. v_etf_constituent 는 편입종목 뷰라 상품 자체의 속성을 갖지 않는다 —
# 여기로 범위를 좁히면 '운용사'·'총보수' 를 묻는 질문에 종목 행만 돌아온다.
_NOT_A_PRODUCT_VIEW = {"v_etf_constituent"}


def plan_retrieval(sig: Signals) -> RoutePlan:
    """Which stores to hit before the first model turn.  Often none — that is the intended outcome.

    순수 집계·정렬 질문(판매가능 채권 중 AA- 이상, 순자산 상위 3개 …)은 지금도 SQL 한 번으로 풀린다.
    그런 질문에 그래프·벡터·문서를 켜는 것은 라우팅이 아니라 무차별 실행이고, 지연과 토큰만 늘린다.
    """
    plan = RoutePlan()
    q = sig.question or ""

    if sig.forward_looking:
        # 스키마에 미래 시점 값이 없다. 현재 스냅샷을 미리 넣어 주면 그것이 예측치로 둔갑한다.
        for src in ("graph", "vector", "document"):
            plan.skipped.append((src, "미래 시점 값 요구 — 스냅샷에 답이 없어 사전검색 무의미"))
        return plan

    # ---- graph
    if is_holding_question(q):
        if sig.holding_blind:
            plan.skipped.append(("graph", "지목된 상품군에 편입종목 데이터가 없음"))
        else:
            holders = holder_mentions(q)
            # 사전에 있는 기업명이면 그 표준명을 쓴다(그래프 라벨과 같은 표기).
            canon = {c.replace(" ", ""): c for c in sig.companies}
            entity = None
            for h in holders:
                entity = canon.get(h.replace(" ", ""), h)
                break
            if entity is None and sig.companies:
                entity = sig.companies[0]
            if entity is None:
                plan.skipped.append(("graph", "편입 질의지만 대상 기업을 문장에서 특정하지 못함"))
            else:
                plan.calls.append(RouteCall(
                    "graph", "graph_neighbors", {"entity": entity, "relation": "constituentOf"},
                    f"'{entity}' 편입 관계 질의",
                    caution="이 목록은 정렬돼 있지 않습니다. 순자산·수익률 등 순위는 SQL 로 확인하세요.",
                    expect_relation="constituentOf",
                ))
    else:
        plan.skipped.append(("graph", "보유·편입 등 관계를 묻지 않음"))

    theme = theme_subject(q, sig.themes)
    if theme is not None and len(plan.calls) < MAX_CALLS:
        plan.calls.append(RouteCall(
            "graph", "graph_neighbors",
            {"entity": f"d:theme/{theme.id}", "relation": "hasTheme_inverse"},
            f"'{theme.label}' 테마 태깅 상품",
            caution="이 목록은 정렬돼 있지 않습니다. 순위·기간 조건은 SQL 로 확인하세요.",
        ))
        plan.skipped = [(s, w) for s, w in plan.skipped if s != "graph"]

    # ---- document
    if wants_narrative(q):
        plan.calls.append(RouteCall(
            "document", "search_documents", {"query": q, "k": 2}, "서술형 정보 요구(위험요인·전략·구조 등)"))
    else:
        plan.skipped.append(("document", "서술형 설명이 아니라 수치·목록 질의"))

    # ---- vector / lexical product lookup
    #
    # 지목한 상품이 **실재할 때만** 켠다.  실재하지 않는 상품을 지목한 문항(KODEX AI로봇,
    # TIGER 양자컴퓨팅커버드콜, KODEX 우주항공레버리지, 미래에셋탄소중립그린뉴딜)의 정답은
    # '그런 상품 없음'이고, 그 질문에 이름이 비슷한 상품 행을 사실로 넣어 주는 것은 오답 쪽으로
    # 미는 일이다.  사전분석이 이미 후보와 경고를 함께 싣고 있으므로 여기서 더 보탤 것이 없다.
    hit = next((m for m in sig.mentions if m.resolved), None)
    if hit is not None and len(plan.calls) < MAX_CALLS:
        table = hit.table if hit.table not in _NOT_A_PRODUCT_VIEW else None
        args: dict = {"query": hit.text, "k": 3 if table else 5}
        if table:
            args["table"] = table
        plan.calls.append(RouteCall(
            "vector", "search_products", args, f"'{hit.text}' 상품 지목 — 해당 행의 속성이 곧 답",
            drop_views=() if is_holding_question(q) else tuple(_NOT_A_PRODUCT_VIEW),
            match_title=hit.text))
    elif any(sig.mentions):
        plan.skipped.append(("vector", "지목된 상품이 데이터에 없음 — 유사 상품 주입은 오답 유도"))
    else:
        plan.skipped.append(("vector", "특정 상품을 이름으로 지목하지 않음"))

    del plan.calls[MAX_CALLS:]
    return plan


# ---------------------------------------------------------------- execution
@dataclass
class Prefetch:
    """What round 0 produced: the text to inject, the trace lines, and the evidence captured."""

    block: str = ""
    steps: list[str] = field(default_factory=list)
    evidences: list = field(default_factory=list)
    seconds: float = 0.0
    ran: int = 0

    def __bool__(self) -> bool:
        return bool(self.block)


def _relation_guard(call: RouteCall, content: str, total: int | None) -> tuple[str, int | None]:
    """Replace a graph detour with the negative fact the question actually asked for.

    ``graph_neighbors`` answers an empty ``constituentOf`` by returning every *other* relation the node
    has.  As a hint to a model that can read the note, that is helpful.  As an injected fact it is a
    trap: '야놀자를 보유한 국내 ETF가 있나요?' came back with a 회사채 named 야놀자1-1 and a headline count of
    15, when the answer is that no ETF holds it.  So the absence is stated as the absence.
    """
    rel = call.expect_relation
    if not rel or f"{rel} (" in content:
        return content, total
    entity = call.args.get("entity", "")
    return (f"'{entity}' 의 {rel}(편입) 관계는 그래프에 0건입니다. "
            f"(이 이름을 가진 다른 노드가 있어도 편입 관계가 아니므로 답으로 쓸 수 없습니다.)"), 0


_ROW = re.compile(r"^- \[([A-Za-z_0-9]+)\]\s*(?:id=\S*\s*\|)?\s*([^|]*)")


def _filter_rows(call: RouteCall, content: str, total: int | None) -> tuple[str, str]:
    """Keep only the rows that can actually answer *this* question; report what was dropped.

    Two filters, both about identity rather than relevance.  A row from the constituent view is some
    other ETF's holding, and a row whose name is merely similar is a different product — as a ranked
    hint either is fine, as an injected fact both are wrong.
    """
    if not call.drop_views and not call.match_title:
        return content, ""
    try:
        from ..kb.entity import mention_resolves
    except Exception:  # noqa: BLE001
        mention_resolves = None  # type: ignore[assignment]
    keep, dropped = [], 0
    for line in content.splitlines():
        m = _ROW.match(line)
        if m is None:
            keep.append(line)
            continue
        view, title = m.group(1), m.group(2).strip()
        bad_view = view in call.drop_views
        bad_name = bool(call.match_title and mention_resolves is not None
                        and not mention_resolves(call.match_title, title))
        if bad_view or bad_name:
            dropped += 1
            continue
        keep.append(line)
    if not dropped:
        return content, ""
    body = "\n".join(l for l in keep if l.strip())
    n = total if total is not None else "?"
    if not body:
        return "", f"(검색 결과 {n}건 모두 '{call.match_title}' 와 다른 상품이거나 편입종목 뷰 행이라 제외했습니다)"
    return body, f"(검색 결과 {n}건 중 {dropped}건은 다른 상품이거나 편입종목 뷰 행이라 제외했습니다)"


def _clip(text: str, limit: int, total: int | None) -> str:
    """Cut long output, and say what was cut against the *true* total, never against the cap."""
    text = text or ""
    if len(text) <= limit:
        return text
    note = f"\n… (출력 {len(text)}자 중 {limit}자만 표시"
    note += f"; 전체 결과 {total}건" if total is not None else ""
    note += ". 같은 도구를 다시 호출하면 전체를 볼 수 있습니다)"
    return text[:limit] + note


async def _execute(tools, name: str, args: dict):
    """Run one tool off the event loop so the time cap can actually take effect.

    ``ToolRegistry.execute`` is declared async but every executor under it is synchronous, so it never
    yields — ``asyncio.wait_for`` around it cannot interrupt anything, and a slow store would blow the
    budget silently (measured: a 1.29s hybrid product search sailed past a 1.2s cap).  Running it in a
    worker thread makes the cap real: on timeout the answer proceeds without that result.
    """
    return await asyncio.to_thread(lambda: asyncio.run(tools.execute(name, args)))


async def run_plan(plan: RoutePlan, tools, *, total_budget: float = TOTAL_BUDGET_SEC,
                   per_call_budget: float = PER_CALL_BUDGET_SEC) -> Prefetch:
    """Execute the plan against the tool registry.  Never raises; never blocks past the budget."""
    out = Prefetch()
    if not plan.calls:
        out.steps.append("라운드 0 라우팅: 사전검색 없음 — " + plan.summary())
        return out
    t0 = time.monotonic()
    lines: list[str] = []
    for call in plan.calls:
        left = total_budget - (time.monotonic() - t0)
        if left <= 0.05:
            out.steps.append(f"라운드 0: {call.tool} 생략 — 사전검색 시간 예산({total_budget:.1f}s) 소진")
            break
        try:
            res = await asyncio.wait_for(_execute(tools, call.tool, dict(call.args)),
                                         timeout=min(per_call_budget, left))
        except asyncio.TimeoutError:
            out.steps.append(f"라운드 0: {call.tool} 시간 초과 — 건너뜀")
            continue
        except Exception as e:  # noqa: BLE001 - one dead store must not take the answer down
            log.exception("prefetch %s failed", call.tool)
            out.steps.append(f"라운드 0: {call.tool} 오류({type(e).__name__}) — 건너뜀")
            continue
        if res is None or getattr(res, "is_error", False):
            out.steps.append(f"라운드 0: {call.tool} 오류 — 건너뜀")
            continue
        ev = getattr(res, "evidence", None)
        total = getattr(ev, "row_count", None) if ev is not None else None
        content, total = _relation_guard(call, getattr(res, "content", "") or "", total)
        content, drop_note = _filter_rows(call, content, total)
        if not content.strip():
            # 남는 행이 없다 = 이 저장소는 이 질문에 답할 것이 없다. 아무것도 싣지 않고 도구 루프에 맡긴다.
            out.steps.append(f"라운드 0: {_describe(call)} → 사용할 행 없음, 주입하지 않음 {drop_note}".rstrip())
            continue
        body = _clip(content, MAX_CHARS.get(call.source, 1000), total)
        head = f"[{call.source}] {_describe(call)} — 선택 근거: {call.why}"
        if total is not None:
            head += f" / 전체 {total}건"
        if drop_note:
            head += " " + drop_note
        caution = call.caution if total else ""  # 0건에는 '정렬' 주의문이 의미 없다
        lines.append(head + "\n" + body + (f"\n※ {caution}" if caution else ""))
        if ev is not None:
            out.evidences.append(ev)
        out.ran += 1
        n = f", {total}건" if total is not None else ""
        out.steps.append(f"라운드 0: {_describe(call)} → 결과 {len(body)}자{n}")
    out.seconds = time.monotonic() - t0
    out.steps.insert(0, f"라운드 0 라우팅: {plan.summary()}")
    if lines:
        # 머리말은 두 가지를 동시에 말해야 한다. (1) 아래는 이미 실행된 사실이다. (2) 그렇다고
        # 도구를 부르지 말라는 뜻은 아니다 — 조회 0회로 낸 final_answer 는 서버가 되돌리고(약 11초)
        # 정렬·집계·개수는 어차피 SQL 로만 나온다. '반복하지 마세요'만 적으면 그 라운드를 유발한다.
        out.block = (
            "[사전 검색 — 라운드 0에서 서버가 이미 실행한 결과입니다. 아래 행은 조회된 사실이므로 "
            "그대로 근거로 쓰고, 같은 조회를 반복하지 마세요. 다만 수치의 정렬·집계·개수는 이 결과로 "
            "얻을 수 없으므로 sql_query 로 확인한 뒤 답변하세요]\n" + "\n\n".join(lines)
        )
    return out


def _describe(call: RouteCall) -> str:
    a = call.args
    key = a.get("entity") or a.get("query") or ""
    key = re.sub(r"\s+", " ", str(key))[:60]
    extra = f", {a['relation']}" if a.get("relation") else ""
    return f"{call.tool}('{key}'{extra})"


# ---------------------------------------------------------------- orchestrator entry point
def signals_for(question: str, *, resolver=None, lexicon=None, ontology=None,
                holding_blind: bool | None = None, forward_looking: bool = False,
                mentions: list[str] | None = None, unresolved: list[str] | None = None) -> Signals:
    """Collect the router's inputs.

    ``mentions``/``unresolved`` let the caller hand over what the pre-analysis already worked out.
    That path costs nothing; resolving the mentions again here costs 0.2–1.0s per question (measured
    on the 54-question set), which would eat the round this module is meant to save.  The resolver
    fallback exists so the router stays usable — and testable — on its own.
    """
    sig = Signals(question=question or "", forward_looking=forward_looking)
    try:
        if mentions is not None:
            missing = {m for m in (unresolved or [])}
            sig.mentions = [MentionSig(m, m not in missing) for m in mentions]
        else:
            from ..kb.entity import extract_product_mentions

            for m in extract_product_mentions(sig.question):
                r = resolver.resolve(m, k=4) if resolver is not None else None
                top = (r.products[0] if r and r.products else None)
                sig.mentions.append(MentionSig(m, bool(r and r.resolved), bool(r and r.exact),
                                               top.table if top is not None else None))
    except Exception:  # noqa: BLE001
        log.exception("routing: mention extraction failed")
    try:
        if lexicon is not None:
            sig.companies = [c.name_ko for c in lexicon.match_companies(sig.question)]
            sig.themes = [ThemeSig(t.id, t.label_ko) for t in lexicon.match_themes(sig.question)]
    except Exception:  # noqa: BLE001
        log.exception("routing: lexicon match failed")
    if holding_blind is None:
        holding_blind = _holding_blind(sig.question, ontology)
    sig.holding_blind = bool(holding_blind)
    return sig


def _holding_blind(question: str, ontology) -> bool:
    """True when every product group the question names is one that carries no holdings data.

    The judgement belongs to the ontology (fp:hasConstituent's domain), not to a word list here.
    """
    if ontology is None or not is_holding_question(question):
        return False
    try:
        from .orchestrator import product_groups  # local import: orchestrator imports this module

        groups = product_groups(question, ontology)
        return bool(groups) and not any(ontology.class_has_holdings(ci.name) for ci in groups)
    except Exception:  # noqa: BLE001
        log.exception("routing: holding-blind check failed")
        return False


async def prefetch(question: str, tools, *, resolver=None, lexicon=None, ontology=None,
                   holding_blind: bool | None = None, forward_looking: bool = False,
                   mentions: list[str] | None = None, unresolved: list[str] | None = None,
                   total_budget: float = TOTAL_BUDGET_SEC) -> Prefetch:
    """Plan and run round 0 for one question.  The single call the orchestrator makes."""
    try:
        sig = signals_for(question, resolver=resolver, lexicon=lexicon, ontology=ontology,
                          holding_blind=holding_blind, forward_looking=forward_looking,
                          mentions=mentions, unresolved=unresolved)
        plan = plan_retrieval(sig)
    except Exception as e:  # noqa: BLE001 - routing is an optimisation, never a dependency
        log.exception("routing: planning failed")
        return Prefetch(steps=[f"라운드 0 라우팅 실패({type(e).__name__}) — 기존 흐름으로 진행"])
    return await run_plan(plan, tools, total_budget=total_budget)
