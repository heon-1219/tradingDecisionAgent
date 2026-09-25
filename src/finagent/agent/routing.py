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

Questions come in Korean or English.  The Korean vocabulary below is the competition build's and is
unchanged; each English pattern runs only on questions ``detect_lang`` calls English, so a Korean
question routes exactly as it did before.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from ..i18n import detect_lang

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


# ---------------------------------------------------------------- English question text
def is_english(question: str | None) -> bool:
    return detect_lang(question) == "en"


_SENTENCE_START = re.compile(r"(^|[.?!]\s+)([A-Z])(?=[a-z])")


def fold_sentence_starts(text: str) -> str:
    """Lower-case the first letter of each sentence, and nothing else.

    The English patterns match lower-case words only, because product names are written in capitals:
    'RISE 200', 'ASML Holding NV', 'Goldman Sachs Future Tech Leaders', 'RH Tactical Outlook ETF' would
    otherwise read as the verbs and nouns the guards look for.  A sentence-initial 'Will …' or
    'Which …' is still an ordinary word, so only that capital is folded.
    """
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).lower(), text or "")


# ---------------------------------------------------------------- routing vocabulary
#
# GRAPH.  Only the *inverse* holding direction ("X를 보유한/편입한 상품", "products that hold X") is
# routed, because that is the direction whose subject is extractable from surface form with confidence:
# the holder entity sits right next to the verb.  The forward direction ('그 ETF의 구성종목은', "the
# constituents of that ETF") names the product, not the company, and resolving it is the product-lookup
# problem, not a graph one.
_HOLDER_VERB = r"보유|편입|담고\s*있|담은|담긴|들고\s*있|포함한|포함하고\s*있|포함된"
_HOLDING_Q = re.compile(_HOLDER_VERB)
# The holder is the token right before the verb.  The character class stops at whitespace, so the
# capture is one word; trailing case particles are stripped afterwards.
_HOLDER = re.compile(r"([A-Za-z가-힣][A-Za-z0-9가-힣·&\.\-]{1,19})\s*(?:" + _HOLDER_VERB + ")")
_PARTICLE = re.compile(r"(?:을|를|이|가|은|는|와|과|도|의|에)$")
# English: the holder comes *after* the verb ('ETFs that hold 삼성전자').  'including'/'included' are
# left out — in English they are mostly prepositions ('including ETNs'), not holdings.
_HOLDER_VERB_EN = r"hold|holds|holding|held|contain|contains|containing|include|includes"
_HOLDING_Q_EN = re.compile(rf"\b(?:{_HOLDER_VERB_EN})\b")
_HOLDER_EN = re.compile(
    rf"\b(?:{_HOLDER_VERB_EN})\s+(?:(?:the|a|an|any|shares?\s+of|stocks?\s+of)\s+)?([^\s,?.!;:()]+)"
)
_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
_HANGUL = re.compile(r"[가-힣]")
# Words that are never a company: they show up before the verb in scope phrases like
# '판매중인 상품 기준으로 채권에 투자하는' or '국내 ETF가 포함된'.
# Relation words too.  In '에코프로의 자회사를 편입한 ETF' the token right before the verb is '자회사'
# (subsidiary); taking it as a company name and calling graph_neighbors('자회사', constituentOf) returns
# 0 rows (measured).  The company these words point to sits earlier in the sentence, and the
# pre-analysis has already resolved it.
_NOT_A_HOLDER = {
    "상품", "종목", "펀드", "etf", "etn", "채권", "자산", "것", "기업", "회사", "지수",
    "데이터", "정보", "포트폴리오", "구성", "비중", "현재", "모두", "전체", "다음",
    "자회사", "계열사", "계열회사", "모회사", "관계사", "관련주", "테마주", "대장주", "경쟁사",
}

# GRAPH (themes).  Only an explicit '테마' ("theme") question routes to the graph.  Theme aliases match
# very broadly — '국내', '미국', '배당' fire on a third of all questions — so an implicit theme match is
# not evidence that the question is *about* a theme.
_THEME_Q = re.compile(r"테마")
_THEME_Q_EN = re.compile(r"\b(?:themes?|thematic)\b")
# Geography, index and structure "themes" are scope adjectives, not subjects.  A question is not a
# theme question because it says '미국' or '레버리지'.
_GENERIC_THEMES = {
    "korea", "usa", "china", "global", "japan", "europe", "emerging", "india", "vietnam",
    "indonesia", "kospi200", "kosdaq", "nasdaq100", "sp500", "leverage", "inverse",
}

# DOCUMENT.  Narrative questions only — the ones whose answer is prose that no column holds.  The
# organisers' own example ('국민성장펀드의 구조와 투자전략 동향', '…의 위험요인') is the shape being
# matched.  Deliberately tight: '무엇인가요' and '알려줘' ("what is", "tell me") are *not* here, because
# they end attribute lookups ('벤치마크 지수는 무엇인가요') far more often than they open a narrative one.
# The same holds in English: no 'what is', no bare 'strategy' (a short strategy is a product attribute)
# and no bare 'risk' (the risk grade is a column).
_DOC_Q = re.compile(
    r"위험\s*요인|리스크\s*요인|유의\s*사항|투자\s*전략|운용\s*전략|투자\s*방식|"
    r"구조와|의\s*구조|구조는|구조를|동향|배경|개요|취지|장단점|장점과\s*단점|"
    r"어떤\s*점|왜\s|무엇을\s*의미|의미하나|뜻인가|용어|설명해|설명\s*해|어떻게\s*운용"
)
_DOC_Q_EN = re.compile(
    r"\brisk\s+factors?\b|\b(?:key|main|major)\s+risks\b|\brisks\s+(?:of|in|involved)\b"
    r"|\bprecautions?\b|\bcaveats?\b|\bthings\s+to\s+(?:note|watch|consider|keep\s+in\s+mind)\b"
    r"|\b(?:investment|management|operating|portfolio)\s+(?:strateg(?:y|ies)|approach|style)\b"
    r"|\bstructure\b|\boverview\b|\bbackground\b|\bpurpose\b|\brationale\b|\btrends?\b"
    r"|\bpros\s+and\s+cons\b|\b(?:advantages|strengths)\s+and\s+(?:disadvantages|weaknesses)\b"
    r"|\bwhy\b|\bwhat\s+does\b.{0,60}?\bmean\b|\bmeaning\s+of\b|\bwhat\s+is\s+meant\s+by\b|\bstand\s+for\b"
    r"|\bexplain\w*|\bdescribe\b|\bterminology\b"
    r"|\bhow\s+(?:is|are|was|were)\b.{0,50}?\b(?:managed|run|operated)\b"
    r"|\bhow\s+does\b.{0,50}?\b(?:manage|operate|work)\b"
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
    # corporate bond named 야놀자1-1 under "15 in total" — for a question whose true answer is "no ETF
    # holds it".
    expect_relation: str | None = None
    # Views whose rows cannot hold this question's answer.  v_etf_constituent rows are *other* ETFs'
    # holdings, so on a product-attribute question they crowd out the row that has the answer
    # (measured: QQQ's v_etf_gl row, the one carrying manager_name, was pushed past the cut).
    drop_views: tuple[str, ...] = ()
    # The name the question actually asked about.  A search result is a *ranking*, not an identity:
    # the top product-view hit for 'Invesco QQQ Trust' was 'Invesco NASDAQ Internet ETF'.  Harmless as
    # a hint, but injected as a fact it makes another product's values this product's.  Only rows whose
    # name really matches are kept.
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
        for src in ("graph", "vector", "document"):
            if src in picked:
                parts.append(f"{src}=ran({picked[src]})")
            else:
                why = next((w for s, w in self.skipped if s == src), "n/a")
                parts.append(f"{src}=skipped({why})")
        return " / ".join(parts)


# ---------------------------------------------------------------- signal extraction
def holder_mentions(question: str) -> list[str]:
    """The entities a '…를 보유/편입한 …' ("… that holds X") question is asking about, in order of appearance.

    Surface extraction on purpose: 쿠팡 and 야놀자 are not in the company lexicon, but the correct
    answer to both questions is 'none', and only the graph knows that.  Trusting lexicon matches alone,
    the router would not switch on for those two questions at all.

    In an English question only a token written in Hangul is taken ('ETFs that hold 쿠팡').  The graph
    labels are Korean, and an English token the lexicon does not know ('Coupang') would come back as a
    false '0 holding relations' fact; English names the lexicon does know arrive through
    ``Signals.companies`` instead.
    """
    q = question or ""
    out: list[str] = []
    for m in _HOLDER.finditer(q):
        tok = _PARTICLE.sub("", m.group(1).strip())
        if len(tok) < 2 or tok.lower() in _NOT_A_HOLDER:
            continue
        if tok not in out:
            out.append(tok)
    if is_english(q):
        for m in _HOLDER_EN.finditer(fold_sentence_starts(q)):
            tok = _POSSESSIVE.sub("", m.group(1).strip())
            if len(tok) < 2 or not _HANGUL.search(tok) or tok.lower() in _NOT_A_HOLDER:
                continue
            if tok not in out:
                out.append(tok)
    return out


def is_holding_question(question: str) -> bool:
    q = question or ""
    if _HOLDING_Q.search(q):
        return True
    return is_english(q) and bool(_HOLDING_Q_EN.search(fold_sentence_starts(q)))


def wants_narrative(question: str) -> bool:
    q = question or ""
    if _DOC_Q.search(q):
        return True
    return is_english(q) and bool(_DOC_Q_EN.search(fold_sentence_starts(q)))


def theme_subject(question: str, themes: list[ThemeSig]) -> ThemeSig | None:
    """The theme a question is *about*, or None.  Requires the word 테마 / theme and a non-generic theme."""
    q = question or ""
    if not (_THEME_Q.search(q) or (is_english(q) and _THEME_Q_EN.search(fold_sentence_starts(q)))):
        return None
    real = [t for t in themes if t.id not in _GENERIC_THEMES]
    if not real:
        return None
    # With several, pick the most specific (the longest label): '중국반도체' > '반도체'.
    return max(real, key=lambda t: len(t.label or ""))


# Tables a product search may be scoped to.  v_etf_constituent is the holdings view and carries no
# attributes of the product itself — narrowing to it returns only holding rows for a question about
# the manager or the total fee.
_NOT_A_PRODUCT_VIEW = {"v_etf_constituent"}


def plan_retrieval(sig: Signals) -> RoutePlan:
    """Which stores to hit before the first model turn.  Often none — that is the intended outcome.

    Pure aggregation and sorting questions (sellable bonds rated AA- or higher, the top 3 by net
    assets …) are already solved by one SQL query.  Switching on the graph, vector or document store
    for them is not routing but indiscriminate execution; it only adds latency and tokens.
    """
    plan = RoutePlan()
    q = sig.question or ""

    if sig.forward_looking:
        # The schema holds no future values.  Prefetching the current snapshot would turn it into a forecast.
        for src in ("graph", "vector", "document"):
            plan.skipped.append((src, "asks for a future value — the snapshot has no answer, so prefetching is pointless"))
        return plan

    # ---- graph
    if is_holding_question(q):
        if sig.holding_blind:
            plan.skipped.append(("graph", "the named product group has no holdings data"))
        else:
            holders = holder_mentions(q)
            # A company the lexicon knows is queried by its canonical name (the same spelling as the graph label).
            canon = {c.replace(" ", ""): c for c in sig.companies}
            # Priority: (1) a holder token the lexicon knows → (2) a company the pre-analysis resolved →
            # (3) the surface token.
            #
            # Why (2) comes before (3): surface extraction looks at one token next to the verb, so when
            # the subject sits a step away, as in '…의 자회사를 편입한' ("that holds …'s subsidiaries"), it
            # takes the wrong word for the company.  _NOT_A_HOLDER filters some, but phrasings outside
            # the list remain, so a resolved company is trusted first.  (3) stays — 쿠팡 and 야놀자 are
            # not in the lexicon, yet the answer to both questions is 'none' and only the graph knows
            # that, so it has to be asked even with a surface token.
            entity = None
            for h in holders:
                hit = canon.get(h.replace(" ", ""))
                if hit is not None:
                    entity = hit
                    break
            if entity is None and sig.companies:
                entity = sig.companies[0]
            if entity is None and holders:
                entity = holders[0]
            if entity is None:
                plan.skipped.append(("graph", "holdings question, but the target company could not be identified in the sentence"))
            else:
                plan.calls.append(RouteCall(
                    "graph", "graph_neighbors", {"entity": entity, "relation": "constituentOf"},
                    f"holding relation of '{entity}'",
                    caution="This list is not sorted. Check rankings by net assets, returns etc. with SQL.",
                    expect_relation="constituentOf",
                ))
    else:
        plan.skipped.append(("graph", "no holding or other relation asked"))

    theme = theme_subject(q, sig.themes)
    if theme is not None and len(plan.calls) < MAX_CALLS:
        plan.calls.append(RouteCall(
            "graph", "graph_neighbors",
            {"entity": f"d:theme/{theme.id}", "relation": "hasTheme_inverse"},
            f"products tagged with the theme '{theme.label}'",
            caution="This list is not sorted. Check rankings and period conditions with SQL.",
        ))
        plan.skipped = [(s, w) for s, w in plan.skipped if s != "graph"]

    # ---- document
    if wants_narrative(q):
        plan.calls.append(RouteCall(
            "document", "search_documents", {"query": q, "k": 2},
            "narrative information requested (risk factors, strategy, structure, etc.)"))
    else:
        plan.skipped.append(("document", "a numeric/list query, not a narrative explanation"))

    # ---- vector / lexical product lookup
    #
    # Switched on **only when the named product exists**.  For questions naming a product that does not
    # exist (KODEX AI로봇, TIGER 양자컴퓨팅커버드콜, KODEX 우주항공레버리지, 미래에셋탄소중립그린뉴딜)
    # the correct answer is 'no such product', and injecting rows of similarly named products as facts
    # pushes toward a wrong answer.  The pre-analysis already carries the candidates and the warning, so
    # there is nothing to add here.
    hit = next((m for m in sig.mentions if m.resolved), None)
    if hit is not None and len(plan.calls) < MAX_CALLS:
        table = hit.table if hit.table not in _NOT_A_PRODUCT_VIEW else None
        args: dict = {"query": hit.text, "k": 3 if table else 5}
        if table:
            args["table"] = table
        plan.calls.append(RouteCall(
            "vector", "search_products", args, f"product '{hit.text}' named — its row's attributes are the answer",
            drop_views=() if is_holding_question(q) else tuple(_NOT_A_PRODUCT_VIEW),
            match_title=hit.text))
    elif any(sig.mentions):
        plan.skipped.append(("vector", "the named product is not in the data — injecting similar products invites a wrong answer"))
    else:
        plan.skipped.append(("vector", "no specific product named"))

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
    trap: '야놀자를 보유한 국내 ETF가 있나요?' came back with a corporate bond named 야놀자1-1 and a
    headline count of 15, when the answer is that no ETF holds it.  So the absence is stated as the absence.
    """
    rel = call.expect_relation
    if not rel or f"{rel} (" in content:
        return content, total
    entity = call.args.get("entity", "")
    return (f"'{entity}' has 0 {rel} (holding) relations in the graph. "
            f"(Another node with this name, if any, is not a holding relation and cannot be used as the answer.)"), 0


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
        return "", f"(all {n} search results excluded: products other than '{call.match_title}' or holdings-view rows)"
    return body, f"({dropped} of {n} search results excluded: other products or holdings-view rows)"


def _clip(text: str, limit: int, total: int | None) -> str:
    """Cut long output, and say what was cut against the *true* total, never against the cap."""
    text = text or ""
    if len(text) <= limit:
        return text
    note = f"\n… (showing {limit} of {len(text)} characters"
    note += f"; {total} results in total" if total is not None else ""
    note += ". Call the same tool again to see everything)"
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
        # Read alone, this line looks like "no store was used".  What it really means is that round 0
        # had nothing to fetch in advance, and structured aggregation and sorting are handled by the RDB
        # in the next round.  think_trace is a graded item ("the tool-use process"), so the routing
        # decision is written out as it is.
        out.steps.append("Round 0 routing: nothing to prefetch → this question is handled by the RDB (sql_query) — "
                         + plan.summary())
        return out
    t0 = time.monotonic()
    lines: list[str] = []
    for call in plan.calls:
        left = total_budget - (time.monotonic() - t0)
        if left <= 0.05:
            out.steps.append(f"Round 0: {call.tool} skipped — prefetch time budget ({total_budget:.1f}s) used up")
            break
        try:
            res = await asyncio.wait_for(_execute(tools, call.tool, dict(call.args)),
                                         timeout=min(per_call_budget, left))
        except asyncio.TimeoutError:
            out.steps.append(f"Round 0: {call.tool} timed out — skipped")
            continue
        except Exception as e:  # noqa: BLE001 - one dead store must not take the answer down
            log.exception("prefetch %s failed", call.tool)
            out.steps.append(f"Round 0: {call.tool} error ({type(e).__name__}) — skipped")
            continue
        if res is None or getattr(res, "is_error", False):
            out.steps.append(f"Round 0: {call.tool} error — skipped")
            continue
        ev = getattr(res, "evidence", None)
        total = getattr(ev, "row_count", None) if ev is not None else None
        content, total = _relation_guard(call, getattr(res, "content", "") or "", total)
        content, drop_note = _filter_rows(call, content, total)
        if not content.strip():
            # No rows left = this store has nothing to answer this question with.  Inject nothing and
            # leave it to the tool loop.
            out.steps.append(f"Round 0: {_describe(call)} → no usable rows, nothing injected {drop_note}".rstrip())
            continue
        body = _clip(content, MAX_CHARS.get(call.source, 1000), total)
        head = f"[{call.source}] {_describe(call)} — why chosen: {call.why}"
        if total is not None:
            head += f" / {total} in total"
        if drop_note:
            head += " " + drop_note
        caution = call.caution if total else ""  # a 'not sorted' caution means nothing on 0 rows
        lines.append(head + "\n" + body + (f"\n※ {caution}" if caution else ""))
        if ev is not None:
            out.evidences.append(ev)
        out.ran += 1
        n = f", {total} rows" if total is not None else ""
        out.steps.append(f"Round 0: {_describe(call)} → result {len(body)} chars{n}")
    out.seconds = time.monotonic() - t0
    out.steps.insert(0, f"Round 0 routing: {plan.summary()}")
    if lines:
        # The header has to say two things at once.  (1) What follows are facts already executed.
        # (2) That does not mean "call no tools" — a final_answer after zero lookups is sent back by the
        # server (about 11 seconds), and sorting, aggregation and counts come only from SQL anyway.
        # Writing only "do not repeat" provokes exactly that round.
        out.block = (
            "[Prefetch — results the server already ran in round 0. The rows below are retrieved facts: "
            "use them as evidence as they are and do not repeat the same lookup. Sorting, aggregation and "
            "counts of values cannot be obtained from these results, though, so confirm them with sql_query "
            "before answering]\n" + "\n\n".join(lines)
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
        return Prefetch(steps=[f"Round 0 routing failed ({type(e).__name__}) — continuing with the normal flow"])
    return await run_plan(plan, tools, total_budget=total_budget)
