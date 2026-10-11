"""Per-question orchestration: deterministic pre-analysis, tool-calling loop under a time budget,
evidence + trace assembly into the competition response schema.

Questions come in Korean or English.  The answer language is ``detect_lang(question)``; it is stated to
the model next to the question and picks the language of every answer sentence this module writes
itself.  The Korean question-parsing patterns are the competition build's and are unchanged; each
English pattern runs only on questions ``detect_lang`` calls English, so a Korean question takes
exactly the path it took before.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, field

from ..i18n import detect_lang, language_name, pick
from ..models import AnswerResponse
from ..ontology.loader import normalize_text as _nrm
from .answer import Evidence, cannot_prefix, fallback_answer, format_evidence, format_trace, normalize_answer
from .llm import LLMClient, ToolCall
from .routing import fold_sentence_starts, is_english
from .routing import prefetch as route_prefetch
from .tools import ToolRegistry, ToolResult

log = logging.getLogger("finagent.orchestrator")

_GRADE_MENTION = re.compile(r"(?:신용등급|등급)\s*(?:이|가|은|는)?\s*([A-Za-z]{1,5}[+\-0]?)|\b([A-Da-d]{1,5}[+\-0]?)\s*(?:이상|이하|등급)")
_RISK_MENTION = re.compile(r"(?:위험등급|위험\s*)?(\d{1,2})\s*등급")
# English: "credit rating AAAA", "rated AAAA", "AAAA or higher".  The rating itself must be written in
# capitals — a lower-case 'a' is an article, and 'BondBloxx BB Rated USD …' is a product name (the
# keywords are matched in lower case on sentence-folded text, see routing.fold_sentence_starts).
_GRADE_MENTION_EN = re.compile(
    r"\bcredit\s+(?:rating|grade)s?(?:\s+(?:of|is|are|at|least|above|below|over|under|higher|lower|than|equal|to))*"
    r"\s*[=:]?\s*([A-Z]{1,5}[+\-0]?)(?![\w+\-])"
    r"|\brated\s+(?:at\s+)?([A-Z]{1,5}[+\-0]?)(?![\w+\-])"
    r"|\b([A-D]{1,5}[+\-0]?)(?=\s+(?:or|and)\s+(?:higher|above|better|up|lower|below|worse)\b)"
)
# English: "risk grade 7", "risk level of 7", "grade 7", "7th grade".  '… Investment Grade 12% …' is a
# product name, hence lower-case keywords and no percentage right after the number.
_RISK_MENTION_EN = re.compile(
    r"\brisk\s+(?:grade|level|rating|class)\s*(?:of\s+)?(\d{1,2})\b(?!\s*%)"
    r"|\bgrade[\s-]*(\d{1,2})\b(?!\s*%)"
    r"|\b(\d{1,2})(?:st|nd|rd|th)?[\s-]*grade\b"
)
_RECENT_MONTHS = re.compile(r"최근\s*(\d+)\s*개월")
_RECENT_MONTHS_EN = re.compile(r"\b(?:last|past|recent|previous)\s+(\d+)\s+months?\b", re.I)


@dataclass
class PreAnalysis:
    text: str
    invalid: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    # The question named a specific product that does not exist: (mention, names of similar candidates)
    unresolved: list[tuple[str, list[str]]] = field(default_factory=list)
    # The ontology violations in ``invalid``, worded for a Korean answer (``invalid`` is English)
    invalid_ko: list[str] = field(default_factory=list)


# Count questions ("how many").
_COUNT_QUESTION = re.compile(r"몇\s*(개|건|종목|가지|곳)|개수|건수|수는\??$|수를\s*알려")
_COUNT_QUESTION_EN = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\s+of\b|\btotal\s+count\b", re.I)


def is_count_question(question: str) -> bool:
    q = question or ""
    return bool(_COUNT_QUESTION.search(q)) or (is_english(q) and bool(_COUNT_QUESTION_EN.search(q)))


# ---------------------------------------------------------------- detecting requests for future values
# A full check of the ontology's properties found that **not one column holds a value realised in the
# future.**  The only property with a forward-looking name is v_bond_kr.exrt_grte_ern_r (guaranteed
# yield to maturity), and even that is a contractually fixed value, not a forecast; mat_date and
# delisting_date are scheduled facts.  Every master table is a snapshot as of the base date.  So a
# question asking for a "future value" is not one that rewording can make answerable — the schema holds
# no answer to it.  That is the basis of this guard.
# (Incident G-019: for "내년 예상 수익률" (next year's expected return) the past 1-year return
# du_er_1y=260.41 was presented as the forecast.)
_FORECAST_WORD = r"예상|전망|예측|추정"
# Why '순자산' (net assets) is among the value words: it too is a base-date snapshot value, so no future
# value exists.  Missing from the list, G-045 ("2027년 말 예상 순자산 규모", expected net assets at the end
# of 2027) went straight through the guard.
# '규모' (size) points to a magnitude, not a value, so it must not be added alone — added that way, the
# ordinary question "미래 성장 규모가 큰 테마" (themes with large future growth) was caught as
# '미래 … 규모' and its answer blocked.  Measured, G-045 is caught by both '예상 순자산' and
# '향후 … 순자산', so it is caught without '규모'.  Only '자산 규모' attached to a value noun stays.
_VALUE_WORD = r"수익률|수익|가격|주가|시세|배당|성과|순자산|자산\s*규모"
# '미래' (future) followed by Hangul is a name, not a point in the future — 9,816 product names start
# with '미래' (미래에셋…, BNK 미래전략기술액티브 …) and the manager itself is 미래에셋.  A following Hangul
# syllable is excluded.  '연말' (year-end) can be in the past ("작년 연말 수익률", last year-end's return),
# so it is left out; only '올해 말' (the end of this year), which means the future, is accepted.
_FUTURE_WORD = r"내년|내후년|향후|앞으로|미래(?![가-힣])|장래|다음\s*해|올해\s*말"

_FORWARD_LOOKING = re.compile(
    "|".join(
        (
            # "예상 수익률", "수익률 전망" — a forecast word right next to a value word
            rf"(?:{_FORECAST_WORD})[^\n]{{0,10}}(?:{_VALUE_WORD})",
            rf"(?:{_VALUE_WORD})[^\n]{{0,10}}(?:{_FORECAST_WORD})",
            # "내년 수익률", "앞으로 주가" — a future-time word near a value word
            rf"(?:{_FUTURE_WORD})[^\n]{{0,15}}(?:{_VALUE_WORD})",
            rf"(?:{_VALUE_WORD})[^\n]{{0,15}}(?:{_FUTURE_WORD})",
            # rise or fall asked without a value word: "오를까요?" (will it rise?), "떨어질까" (will it fall)
            r"(?:오를|올라갈|상승할|하락할|떨어질|내려갈|빠질)\s*(?:까|지|것|수|듯|가능성|여지|전망)",
            r"(?:얼마나|얼마|몇\s*%|몇\s*퍼센트)\s*(?:오를|오르|상승|하락|떨어질|내릴)",
        )
    )
)

# The guaranteed yield to maturity (v_bond_kr.exrt_grte_ern_r) and the distribution/dividend frequency
# are values the data really holds, so they must not be mistaken for forecast questions.  Nor must the
# manager name '미래 에셋' written with a space.  They are erased before the check so they count neither
# as value words nor as future words.
_FORWARD_EXEMPT = re.compile(r"(?:만기\s*)?보장\s*수익률|배당\s*주기|분배\s*주기|미래\s*에셋")

# ---- a future point written as a four-digit year
# A future written only as a year, not '내년·향후' (next year, from now on), is not caught by _FUTURE_WORD.
# G-045 ("KODEX 200의 2027년 말 예상 순자산 규모") slipped out exactly that way and gave the current net
# assets as the forecast.  A year later than the data's base year is taken as a future point and
# replaced with a _FUTURE_WORD.
#
# The trap: "만기가 2027년인 채권" (bonds maturing in 2027, G-002) also names a future year, but it asks
# about a **recorded date field**, not a forecast — mat_date is in the data, so it must be answerable.
# So a year attached to date-field vocabulary such as maturity, redemption, issue or listing is not
# counted as a future point.  After the replacement the existing rules apply unchanged: only a future
# year close to a value word (return, net assets …) makes a forecast question.
_DATE_FIELD_WORD = r"만기|상환|만료|발행|상장|설정|기준일|마감|청산|해지|폐지|출시|가입"
_YEAR_MENTION = re.compile(
    rf"(?:(?P<pre>{_DATE_FIELD_WORD})[^\n]{{0,6}})?(?P<year>\d{{4}})\s*년(?:\s*(?P<post>{_DATE_FIELD_WORD}))?"
)
# Default for paths that cannot read the store meta (the module on its own, tests).  In real use the
# Orchestrator passes the year of base_date (= v_etf_kr's meta base date).
_BASE_YEAR_FALLBACK = 2026


def _mark_future_years(text: str, base_year: int) -> str:
    """Replace a year after the base year with a future-time word (years attached to date fields excepted)."""

    def repl(m: re.Match) -> str:
        if int(m.group("year")) <= base_year or m.group("pre") or m.group("post"):
            return m.group(0)
        return "향후"

    return _YEAR_MENTION.sub(repl, text)


# ---- English.  The same four families: forecast word + value word, future-time word + value word, a
# future year, and rise/fall asked without a value word.  Matched in lower case on sentence-folded text:
# product names are capitalised ('Goldman Sachs Future Tech Leaders', 'RH Tactical Outlook ETF', 'RISE 200').
_FORECAST_WORD_EN = r"expect(?:ed|s)?|forecast(?:s|ed)?|outlook|predict(?:ed|ion|ions|s)?|projected|projections?|estimated|anticipated"
_VALUE_WORD_EN = (
    r"returns?|yields?|prices?|share\s+prices?|valuations?|dividends?|distributions?|performance|gains?|profits?"
    r"|net\s+assets?|NAV|AUM|assets\s+under\s+management|(?:fund|asset)\s+size"
)
_NUM_EN = r"\d+|two|three|four|five|six|seven|eight|nine|ten"
_FUTURE_WORD_EN = (
    r"next\s+(?:year|month|quarter|week|decade|few\s+years)|the\s+(?:coming|next)\s+(?:years?|months?|quarters?)"
    rf"|in\s+(?:{_NUM_EN})\s+years(?:'|’)?(?:\s+time)?|in\s+(?:a|one)\s+year(?:'s|’s)\s+time"
    rf"|(?:{_NUM_EN}|a|an|one)\s+years?\s+(?:from\s+now|later|ahead)"
    r"|going\s+forward|from\s+now\s+on|future(?!s)|(?:the\s+)?end\s+of\s+this\s+year|later\s+this\s+year"
)
_DIRECTION_EN = (
    r"rise|fall|go\s+(?:up|down)|increase|decrease|drop|climb|gain|decline|rebound|recover|crash|surge|plunge"
    r"|soar|appreciate|depreciate|grow"
)
_FORWARD_LOOKING_EN = re.compile(
    "|".join(
        (
            rf"\b(?:{_FORECAST_WORD_EN})\b[^\n]{{0,30}}?\b(?:{_VALUE_WORD_EN})\b",
            rf"\b(?:{_VALUE_WORD_EN})\b[^\n]{{0,30}}?\b(?:{_FORECAST_WORD_EN})\b",
            rf"\b(?:{_FUTURE_WORD_EN})\b[^\n]{{0,40}}?\b(?:{_VALUE_WORD_EN})\b",
            rf"\b(?:{_VALUE_WORD_EN})\b[^\n]{{0,40}}?\b(?:{_FUTURE_WORD_EN})\b",
            # "will it rise?", "is 삼성전자 likely to fall" — but never "could you …" / "would you …"
            rf"\b(?:will|would|could|might|may|going\s+to|likely\s+to|expected\s+to|set\s+to)\b(?!\s+(?:you|u|I|we)\b)"
            rf"[^\n.?!]{{0,40}}?\b(?:{_DIRECTION_EN})\b",
            rf"\bhow\s+(?:much|far|high|low)\b[^\n.?!]{{0,30}}?\b(?:will|would|could|might|can)\b[^\n.?!]{{0,30}}?"
            rf"\b(?:{_DIRECTION_EN}|return|earn|make)\b",
        )
    )
)
_FORWARD_EXEMPT_EN = re.compile(
    r"\bguaranteed\s+(?:maturity\s+)?(?:yield|return|rate)s?\b|\byield\s+to\s+maturity\b"
    r"|\b(?:dividend|distribution|payout)\s+(?:frequency|cycle|schedule)\b",
    re.I,
)
# A bare year counts as a future point only after a time preposition ('in 2027', 'by the end of 2027'):
# foreign ETF names carry years too ('iShares iBonds Dec 2030 Term Corporate ETF', 'TDF 2045').  Date
# fields nearby ('bonds maturing in 2027', '2027 maturity') are recorded dates, as in Korean.
_YEAR_EN = re.compile(
    r"\b(?:in|by|until|till|through|for|around|during|before|after|of|at)\s+"
    r"(?:the\s+(?:end|start|beginning|middle|close)\s+of\s+)?(?:(?:early|mid|late)[\s-]+)?"
    r"(?P<year>(?:19|20)\d{2})\b(?![\-/.]\d)"
)
_DATE_FIELD_EN = re.compile(
    r"\b(?:matur\w*|redeem\w*|redemption|expir\w*|issu\w*|list(?:ed|ing|s)?|launch\w*|incept\w*|establish\w*"
    r"|settle\w*|base\s+date|deadline|liquidat\w*|terminat\w*|delist\w*|subscri\w*|call(?:ed|able)?)\b",
    re.I,
)


def _mark_future_years_en(text: str, base_year: int) -> str:
    def repl(m: re.Match) -> str:
        if int(m.group("year")) <= base_year:
            return m.group(0)
        if _DATE_FIELD_EN.search(text[max(0, m.start() - 25):m.start()]) or _DATE_FIELD_EN.search(text[m.end():m.end() + 15]):
            return m.group(0)
        return " in the future "

    return _YEAR_EN.sub(repl, text)


def asks_forward_looking(question: str, base_year: int = _BASE_YEAR_FALLBACK) -> bool:
    """Does the question ask for a 'future value' that the data does not hold?"""
    q = question or ""
    text = _mark_future_years(_FORWARD_EXEMPT.sub(" ", q), base_year)
    if _FORWARD_LOOKING.search(text):
        return True
    if not is_english(q):
        return False
    en = _mark_future_years_en(_FORWARD_EXEMPT_EN.sub(" ", fold_sentence_starts(q)), base_year)
    return bool(_FORWARD_LOOKING_EN.search(en))


# ---------------------------------------------------------------- detecting the scope of a holdings lookup
# Holdings (constituent) data exists only in v_etf_constituent, and that view covers only domestic
# ETFs, 1,162 products / 22,599 rows (collected from Naver Finance).  The foreign ETF master (v_etf_gl)
# and public funds (v_fund_pub) have no holdings column at all.  So "foreign ETFs / public funds that
# hold X" is not a question that a well-written query answers; the data simply has no such relation —
# a fact of the schema, not a judgement.
#   G-040: after getting 0 rows from v_etf_constituent, the model listed 10 domestic ETFs left over
#          from an earlier round as 'foreign ETFs'.
#   G-041: it read search_products' name-similarity results as 'funds holding that stock'.
#          search_products gives only lexical/semantic similarity and establishes no holding relation.
_HOLDING_WORD = re.compile(r"보유|편입|담고\s*있|담은|담긴|포함한|포함하고\s*있|포함된|구성\s*종목|들고\s*있")
# Product groups without holdings data.  '펀드' (fund) is a scope word by itself and covers public and private funds.
_NO_HOLDING_SCOPE = re.compile(r"해외\s*상장\s*ETF|해외\s*ETF|미국\s*ETF|공모\s*펀드|사모\s*펀드|ETN|펀드")
# If domestic ETFs are named as well, part of the question is answerable, so the guard does not fire.
# "에코프로비엠을 편입한 ETF" (ETFs holding 에코프로비엠), with no scope word at all, does not hit
# _NO_HOLDING_SCOPE either and is answered normally.
_HOLDING_SCOPE_OK = re.compile(r"국내\s*(?:상장\s*)?ETF")
# English.  Lower-case keywords on sentence-folded text ('ASML Holding NV' is a name); 'including' is a
# preposition far more often than a holding.  'US' must be written in capitals — 'tell us' is not a market.
_HOLDING_WORD_EN = re.compile(
    r"\b(?:hold|holds|holding|holdings|held|contain|contains|containing|include|includes|constituents?"
    r"|portfolio\s+of|in\s+(?:its|their)\s+portfolios?)\b"
)
_NO_HOLDING_SCOPE_EN = re.compile(
    r"\b(?i:overseas|foreign|global|international)(?:[\s-]+(?i:listed))?\s+(?i:etfs?)\b"
    r"|\b(?:U\.?S\.?|(?i:american))(?:[\s-]+(?i:listed))?\s+(?i:etfs?)\b"
    r"|\b(?i:public(?:ly[\s-]+offered)?|mutual|private(?:ly[\s-]+placed)?)\s+(?i:funds?)\b"
    r"|\b(?i:etns?)\b"
    r"|\b(?i:funds?)\b(?!\s+(?i:size|managers?|family|families|flows?))"
)
_HOLDING_SCOPE_OK_EN = re.compile(r"\b(?i:domestic|korean|korea|krx)(?:[\s-]+(?i:listed))?\s+(?i:etfs?)\b")


def mentions_holdings(question: str) -> bool:
    """Does the question use a holding word (보유/편입…, hold/contain/include…)?"""
    q = question or ""
    return bool(_HOLDING_WORD.search(q)) or (is_english(q) and bool(_HOLDING_WORD_EN.search(fold_sentence_starts(q))))


def asks_holdings_without_data(question: str) -> bool:
    """Asks for 'products holding that company', but the named product group has no holdings data?"""
    q = question or ""
    holding, blind, ok = _HOLDING_WORD.search(q), _NO_HOLDING_SCOPE.search(q), _HOLDING_SCOPE_OK.search(q)
    if is_english(q):
        en = fold_sentence_starts(q)
        holding = holding or _HOLDING_WORD_EN.search(en)
        blind = blind or _NO_HOLDING_SCOPE_EN.search(en)
        ok = ok or _HOLDING_SCOPE_OK_EN.search(en)
    return bool(holding and blind) and not ok


# ---------------------------------------------------------------- detecting the product-group scope (ontology-driven)
# The *facts* of which product group has what belong to the ontology (the class hierarchy, the domain of
# fp:hasConstituent, the columns' fp:sourceTable).  What lives here is not those facts but **Korean
# spelling variants** only: rdfs:label says just '해외ETF', so habits of speech like '해외 상장 ETF' and
# '미국 ETF' are not caught.  More spellings add rows to this table; a change in a group's nature
# changes the TTL — the two must not mix.
_GROUP_ALIASES = {
    "해외상장etf": "ForeignETF", "미국etf": "ForeignETF", "미국상장etf": "ForeignETF",
    "해외상장etn": "ForeignETN", "국내상장etf": "DomesticETF", "국내상장etn": "DomesticETN",
    "공모형펀드": "PublicFund", "사모형펀드": "PrivateFund", "원화채권": "DomesticBond",
}
# English questions also match the classes' English labels (label_en: 'Foreign ETF', 'Public Offering
# Fund' …) and these English spelling variants.  'US ETF' is folded to 'US-listed ETF' first, because the
# normalised 'usetf' would also hit 'PLUS ETF' and 'FOCUS ETF'.
_GROUP_ALIASES_EN = {
    "overseasetf": "ForeignETF", "overseaslistedetf": "ForeignETF", "foreignlistedetf": "ForeignETF",
    "globaletf": "ForeignETF", "internationaletf": "ForeignETF", "uslistedetf": "ForeignETF",
    "americanetf": "ForeignETF", "overseasetn": "ForeignETN", "foreignlistedetn": "ForeignETN",
    "koreanetf": "DomesticETF", "koreaetf": "DomesticETF", "korealistedetf": "DomesticETF",
    "domesticlistedetf": "DomesticETF", "krxlistedetf": "DomesticETF", "koreanetn": "DomesticETN",
    "publicfund": "PublicFund", "publiclyofferedfund": "PublicFund", "mutualfund": "PublicFund",
    "privatefund": "PrivateFund", "privatelyplacedfund": "PrivateFund",
    "koreanbond": "DomesticBond", "krwbond": "DomesticBond",
}
_US_ETF = re.compile(r"\bU\.?S\.?(?:[\s-]+(?i:listed))?\s+(?=(?i:etfs?)\b)")


def product_groups(question: str, ontology) -> list:
    """The product groups (ontology classes) the question restricts itself to, most specific only.

    '해외 ETF' must resolve to ForeignETF and *not* also to the bare 'ETF' that sits inside it — the
    span-containment pass is what keeps '해외 ETF' from inheriting 국내ETF's holdings.
    """
    if ontology is None or getattr(ontology, "graph", None) is None:
        return []
    english = is_english(question)
    forms: list[tuple[str, object]] = []
    for ci in ontology.descendants("Product"):
        if ci.label_ko:
            forms.append((_nrm(ci.label_ko), ci))
        if english and getattr(ci, "label_en", ""):
            forms.append((_nrm(ci.label_en), ci))
        if re.fullmatch(r"[A-Z]{2,4}", ci.name):  # fp:ETF / fp:ETN — the local name is the Korean spelling too
            forms.append((ci.name.lower(), ci))
    for surf, cname in list(_GROUP_ALIASES.items()) + (list(_GROUP_ALIASES_EN.items()) if english else []):
        ci = ontology.class_info(cname)
        if ci is not None:
            forms.append((surf, ci))
    q = _nrm(_US_ETF.sub("US-listed ", question) if english else question)
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
            continue  # drop a fragment inside a longer spelling (the 'ETF' inside '해외ETF')
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


# ---------------------------------------------------------------- detecting the ranking criterion
# From expressions that ask for a ranking — "…기준" (by …), "…이 가장 높은/낮은/큰/작은" (the highest/
# lowest/largest/smallest …), "TOP N", "상위/하위 N" (top/bottom N) — extract the value expression the
# ranking is based on.  This only extracts; the ontology (resolve_measure) judges.
_CRITERION = [
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*기준(?![일가])"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:이|가)\s*가장\s*(?:높은|낮은|큰|작은|많은|적은|빠른|긴|짧은)"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:이|가)\s*(?:높은|낮은|큰|작은|많은|적은)\s*순"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:상위|하위)\s*\d+"),
    re.compile(r"([가-힣A-Za-z0-9()%\s]{2,26})\s*(?:순으로|순위)"),
]
# Words that name the target, not the value.  Caught as a criterion they hit string columns such as
# pd_nm or product_type and inject nonsense like "the sort criterion is the product name".
_CRIT_STOP = {"상품", "종목", "가지", "개수", "건수", "곳", "것", "기업", "회사", "펀드", "etf", "etn",
              "채권", "상품군", "항목", "기준", "이름", "정보", "데이터", "조건", "구분", "현재", "지금"}
# Period modifiers in front of a criterion — '1년' (1 year), '5년', '최근 3개월' (last 3 months), '연초대비'
# (year to date) form one unit with the noun after them.
_PERIOD_TOKEN = re.compile(r"^(?:최근)?\s*(?:\d+\s*(?:일|주|주일|개월|달|년)|연초\s*대비|ytd)$", re.I)
# A word ending in an adnominal or verbal ending is not the name of a value but a modifier of the noun
# before it.  The '판매중인' (on sale) of '판매중인 상품 기준으로' dragged in is_on_sale as the sort
# criterion, and the '다른' (other) of '서로 다른 종목' did the same.  Real value names (총보수·순자산·
# 수익률·표면금리·분배금·비중·만기·변동성·추적오차) do not end in these endings.
_ADNOMINAL = re.compile(r"(?:인|한|는|된|던|른|은|을|를)$")
_COUNT_ASKED = re.compile(r"(?:top|상위|하위)\s*(\d+)|(\d+)\s*개(?!월)|(\d+)\s*(?:종목|건|가지)", re.I)

# English: "top 5 … by net assets", "sorted/ranked/ordered by 1-year return", "the highest total fee".
# The phrase runs up to a clause word, so 'by net assets among domestic ETFs' yields 'net assets'.
_CRIT_END_EN = (
    r"(?=\s*(?:[,.?!;:]|$|\b(?:among|in|for|of|with|that|which|from|and|or|as|on|to|at|is|are|was|were"
    r"|has|have|across|between|within|over|than|per|versus|vs|compared|desc|descending|asc|ascending"
    r"|order|first|limit|please)\b))"
)
_CRIT_PHRASE_EN = r"(?:the\s+|their\s+|its\s+)?(?P<m>[A-Za-z0-9()%\-][A-Za-z0-9()%\- ]{1,39}?)"
_CRITERION_EN = [
    re.compile(rf"\b(?:sort(?:ed)?|rank(?:ed)?|order(?:ed)?)\s+by\s+{_CRIT_PHRASE_EN}{_CRIT_END_EN}", re.I),
    re.compile(
        rf"\b(?:top|bottom|highest|lowest|largest|smallest|biggest|best|worst|most|least)\b[^.?!\n]{{0,80}}?"
        rf"\bby\s+{_CRIT_PHRASE_EN}{_CRIT_END_EN}",
        re.I,
    ),
    re.compile(
        rf"\b(?:highest|lowest|largest|smallest|biggest|greatest|longest|shortest|cheapest)\s+{_CRIT_PHRASE_EN}{_CRIT_END_EN}",
        re.I,
    ),
]
_CRIT_STOP_EN = {"product", "products", "fund", "funds", "etf", "etfs", "etn", "etns", "bond", "bonds", "stock",
                 "stocks", "item", "items", "one", "ones", "company", "companies", "name", "names", "number",
                 "count", "data", "value", "values", "share", "shares", "class", "classes"}
# The ontology's measure labels are Korean ('순자산총액', '1년 수익률'), so an English measure phrase is
# looked up through its Korean label words.  A phrase with no known measure word is dropped: a weak
# match on English text is noise ('net assets' half-matches has_buy_yield), not a hint.
_MEASURE_EN = (
    (re.compile(r"\bnet\s+assets?(?:\s+value)?\b|\b(?:fund|asset)\s+size\b", re.I), "순자산"),
    (re.compile(r"\bAUM\b|\bassets\s+under\s+management\b", re.I), "AUM"),
    (re.compile(r"\btotal\s+(?:expense|fee)s?(?:\s+ratio)?\b|\bexpense\s+ratio\b|\bfees?\b", re.I), "총보수"),
    (re.compile(r"\btracking\s+error\b", re.I), "추적오차율"),
    (re.compile(r"\bvolatility\b", re.I), "변동성"),
    (re.compile(r"\b(?:distribution|dividend)\s+(?:yield|rate)\b", re.I), "분배수익률"),
    (re.compile(r"\bcoupon(?:\s+rate)?\b", re.I), "표면이자율"),
    (re.compile(r"\b(?:purchase|buy)\s+yield\b|\byield\s+to\s+maturity\b", re.I), "매수수익률"),
    (re.compile(r"\bafter[\s-]+tax\s+yield\b", re.I), "세후수익률"),
    (re.compile(r"\breturns?\b|\bperformance\b|\byield\b", re.I), "수익률"),
)
_PERIOD_EN = (
    (re.compile(r"\b(\d+)[\s-]*(?:years?|yrs?)\b", re.I), "{}년"),
    (re.compile(r"\b(\d+)[\s-]*months?\b", re.I), "{}개월"),
    (re.compile(r"\b(\d+)[\s-]*weeks?\b", re.I), "{}주"),
    (re.compile(r"\b(\d+)[\s-]*days?\b", re.I), "{}일"),
    (re.compile(r"\bone[\s-]*year\b|\bannual\b", re.I), "1년"),
    (re.compile(r"\bytd\b|\byear[\s-]*to[\s-]*date\b", re.I), "ytd"),
)
_NUMBER_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "twenty": "20"}
_N_EN = r"\d+|" + "|".join(_NUMBER_WORDS)
_COUNT_ASKED_EN = re.compile(
    rf"\b(?:top|bottom|first)\s+({_N_EN})\b"
    rf"|\b({_N_EN})\s+(?:largest|smallest|biggest|highest|lowest|best|worst|cheapest|most|least)\b"
    rf"|\b(\d+)\s+(?:[A-Za-z][A-Za-z-]*\s+){{0,2}}(?:products?|funds?|ETFs?|ETNs?|bonds?|stocks?|items?|names?)\b",
    re.I,
)


def measure_phrase_ko(phrase: str) -> str | None:
    """The Korean label words for an English measure phrase ('1-year return' -> '1년 수익률'), or None."""
    measure = next((ko for pat, ko in _MEASURE_EN if pat.search(phrase)), None)
    if measure is None:
        return None
    period = next((fmt.format(*m.groups()) for pat, fmt in _PERIOD_EN if (m := pat.search(phrase))), None)
    return f"{period} {measure}" if period else measure


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
                # Judged by the last word: Korean modifiers come first, so the last noun decides what the
                # phrase is.  '판매중인 상품' (products on sale) names products, not a value.
                if len(n) < 2 or n in seen or n in _CRIT_STOP or last in _CRIT_STOP or _ADNOMINAL.search(n):
                    continue
                seen.add(n)
                out.append(c)
            if out:
                return out
    if is_english(q):
        for pat in _CRITERION_EN:
            for m in pat.finditer(q):
                toks = [t.strip(",") for t in m.group("m").split() if t.strip(",")][:4]
                if not toks or toks[-1].lower() in _CRIT_STOP_EN:
                    continue
                phrase = " ".join(toks)
                if measure_phrase_ko(phrase) is not None:
                    return [phrase]
    return []


_STATUS_RANK = {"resolved": 4, "period_missing": 3, "ambiguous": 2, "weak": 1, "none": 0}


def requested_count(question: str) -> int | None:
    q = question or ""
    m = _COUNT_ASKED.search(q)
    if not m and is_english(q):
        m = _COUNT_ASKED_EN.search(q)
    if not m:
        return None
    n = next((g for g in m.groups() if g), None)
    return int(_NUMBER_WORDS.get(n.lower(), n)) if n else None


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

    ``rows_to_text`` renders it as ``cnt`` / ``240`` / ``Total rows: 1``.  HCX (the competition build's
    model) has been observed running the right SQL, reading 240 and then writing "24개" in the answer,
    so the server keeps the number the tool actually produced and checks the answer against it.
    """
    # The tools append their own advisory lines ("⚠ The number above is a row count" ...), so match the
    # header / value / count prefix and ignore whatever follows — an exact 3-line check silently gave up
    # on every result that carried a warning.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or "|" in lines[0] or "|" in lines[1] or not re.match(r"Total rows: 1(?!\d)", lines[2]):
        return None
    val = lines[1].replace(",", "").strip()
    return int(val) if re.fullmatch(r"-?\d+", val) else None


# Expressions that show the answer has recognised that "that product does not exist" (or asks back).
_ACKNOWLEDGED = ("확인할 수 없", "확인되지 않", "존재하지 않", "찾을 수 없", "없습니다만", "혹시", "말씀하시", "유사한", "다른 상품")
_ACKNOWLEDGED_EN = re.compile(
    r"\b(?:cannot|can't|can\s+not|could\s+not|couldn't|unable\s+to)\s+(?:be\s+)?"
    r"(?:confirm(?:ed)?|find|found|locate[d]?|identif(?:y|ied))\b"
    r"|\b(?:does|do|did)\s+not\s+exist\b|\bdoesn't\s+exist\b|\bnot\s+found\b|\bno\s+(?:such|matching|exact)\b"
    r"|\bdid\s+you\s+mean\b|\bdo\s+you\s+mean\b|\bare\s+you\s+(?:looking\s+for|referring\s+to)\b|\bby\s+any\s+chance\b"
    r"|\bsimilar(?:ly[\s-]+named)?\b|\b(?:another|a\s+different|other)\s+product\b"
    r"|\b(?:is|are)\s+not\s+(?:available|provided|in\s+the\s+data)\b",
    re.I,
)


def _acknowledged(answer: str) -> bool:
    return any(k in answer for k in _ACKNOWLEDGED) or bool(_ACKNOWLEDGED_EN.search(answer))


# ---------------------------------------------------------------- detecting the same call repeated
# Re-sending the very same call after a first approach failed really happened (G-015: in 8 rounds there
# were only 2 distinct queries; one failed every time with 'no such column: pd_no', the other returned 0
# rows every time).  The cause was that nothing in the loop said "you just sent the same thing again".
#
# ★ The LLM round has been spent by then anyway.  What this buys is not time but 'a chance for the
#   model to notice it is stuck' — handing back the same result silently makes the model believe it got
#   new information, and it sends the same query again.
#
# ★ So the call is **not blocked; it runs as usual** and only a warning is appended to the result.  The
#   first version skipped execution and returned the cached result; in G-040 that kicked in at rounds 3
#   and 4, cut off the model's attempt to get out of the dead end, and it ended up making the answer up
#   from results left over from earlier rounds.  Skipping execution saves only a few milliseconds of tool
#   time, and costs one way out.

_SQL_ARGS = ("sql", "sparql")
# purpose is a comment for the trace and does not change the result.  Put it in the key and merely
# rewording it makes 'a different call' that slips past the detector — which is exactly the shape of the
# loop being prevented.
_KEY_IGNORED_ARGS = ("purpose",)
_SQL_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")


def _norm_sql(s: str) -> str:
    """Normalisation for comparing whether two SQL strings are 'the same query'.

    Strip code fences and the trailing semicolon, collapse whitespace (whitespace around operators,
    parens and commas is removed entirely), and lower-case **only outside string literals**.  Only the
    notation changes; the meaning is not touched.

    Literals must not be lower-cased — in SQLite crd_grd='AA-' and 'aa-' are different queries, and the
    normal retry that narrows down by changing condition values would be mistaken for 'repetition'.
    """
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", (s or "").strip(), flags=re.I).strip().rstrip(";")
    s = re.sub(r"\s+", " ", s).strip()

    def _plain(seg: str) -> str:
        # 'a = 1' and 'a=1', 'in (1, 2)' and 'in(1,2)' are the same query.  Identifiers never merge
        # (only whitespace *attached* to punctuation and operators is removed).
        return re.sub(r"\s*([^\w\s가-힣])\s*", r"\1", seg.lower())

    out, last = [], 0
    for m in _SQL_LITERAL.finditer(s):
        out.append(_plain(s[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_plain(s[last:]))
    return "".join(out)


# ------------------------------------------------- forced retry that sends back an answer given without a lookup
# HCX-005 (the competition build's model) sometimes answered from pre-training knowledge alone, without
# calling a single tool.  Measured over 5 runs of the same questions, answers produced that way were
# wrong without exception:
#   G-034 — 5/5 ended in 3.3 seconds with 0 tools and gave the manager as the short form 'Invesco'.
#           The DB value is 'Invesco Capital Management LLC', so not the official name — wrong.
#   G-010 — of 5 runs, the 3 that called tools were right; the 2 that did not invented a product name
#           and numbers that do not exist in the DB at all (pure hallucination).
# The observation that calling tools is right and not calling them is wrong had no exception, so a
# final_answer submitted without a single lookup is not accepted and one more round runs.  The grading
# criterion [hallucination prevention] requires "does not extract content that is not in the data".
#
# 'Lookup tool' means everything except final_answer.  It is written as an explicit allow-list rather
# than a negative condition (name != "final_answer") because the model calling a tool with a nonexistent
# name must not count as a lookup — such a call is not executed and only an "Unknown tool" error comes
# back, so no evidence arises.  If the list drifts from the real registry,
# test_lookup_tools_matches_registry catches it.
#
# table_info returns a schema, not values, but counts as a lookup: some questions are answered by
# table_info alone (base dates, allowed values), and leaving it out would add an evidence-less round to them.
LOOKUP_TOOLS = frozenset({
    "sql_query", "table_info", "search_products", "resolve_entity",
    "graph_neighbors", "graph_query", "search_documents",
})

# The measured cost of one forced retry is one LLM round ≈ 11 seconds.  After sending it back, though, a
# round to run the tools and a round to receive final_answer again are needed, so three rounds' worth of
# headroom is required.  With less budget left the answer is accepted as is without sending it back —
# losing the whole question by exceeding the response time (the organisers recommend 60 seconds) is worse
# than one answer without evidence.
FORCED_LOOKUP_MIN_REMAINING_SEC = 33.0

# The tool_result returned for the final_answer call.  The OpenAI/CLOVA protocol requires a result for
# every tool_call of the turn, so it must be filled to continue the conversation (left empty, the next
# request fails with 400).
_NO_LOOKUP_TOOL_RESULT = (
    "Submission rejected. No data lookup tool has been called for this question. "
    "An answer not grounded in lookup results counts as a hallucination. Look up the data first, then "
    "call final_answer again."
)

_UNRUN_TOOL_RESULT = (
    "This call was not executed because final_answer came in the same turn. Call it again now if you need it."
)

# The retry instruction is given as 'a fact at the point of use'.  In this project, prompt rules given in
# advance were ignored (4/4) and instructions given as facts of the moment were followed (5/5).  The last
# paragraph prevents the side effect of cannot-confirm questions starting to make up answers because of
# this retry — a refusal after a lookup must stand.
_NO_LOOKUP_RETRY = (
    "[Data lookup required before answering] The final_answer you just submitted was not accepted, because "
    "no data lookup tool has been called for this question.\n"
    "You must look up data at least once before answering. Take official product names, numbers and dates "
    "exactly as the lookup results give them. Do not use names from memory — the official names in this "
    "data differ from the common short forms (e.g. manager names are full legal names), and numbers from "
    "memory are not this data's values.\n"
    "What to do now: query the relevant view with sql_query. If unsure of a column name, call table_info "
    "first; if you do not know a product's exact spelling, call search_products or resolve_entity first. "
    "After checking the lookup results, call final_answer again.\n"
    "If the lookup results are empty or you found no evidence, do not make up an answer. In that case "
    "keeping status=cannot_confirm is the right answer. 'This cannot be confirmed with the provided data' "
    "after a lookup is a correct answer; an assertion made without a lookup is a wrong one."
)


# ---------------------------------------------------------------- final_answer gate
# When the tool loop ends without final_answer, the orchestrator takes the last text the model let out
# as the answer.  7 of the 54 questions took this path and 4 of them were wrong (G-007/G-009/G-025).
# G-009 had already received 3 SQL results and let out "제공된 데이터로는 확인할 수 없습니다" (cannot be
# confirmed with the provided data).
#
# Like the zero-lookup gate (_NO_LOOKUP_RETRY), the instruction carries only 'facts at the point of use'
# — in this project's measurements prompt rules given in advance were ignored 4/4 and instructions given
# as facts of the moment were followed 5/5.  The last paragraph targets the G-025 shape (claiming both
# cannot-confirm and an aggregate result in the same answer, a self-contradiction).
_NO_FINAL_RETRY_HEAD = (
    "[Answer not submitted] final_answer has not been called yet. The text replies so far are not an "
    "answer — an answer is submitted only through the final_answer tool.\n"
)


def _no_final_retry_tail(lang: str) -> str:
    return (
        "What to do now: without further lookups, call final_answer(status, answer) once, using only the "
        "results gathered so far.\n"
        f"Put the answer in complete sentences in {language_name(lang)}. Do not write JSON, code fences or "
        "tool-call syntax inside the answer string — the grader sees it as is.\n"
        "status is answered if you have evidence, cannot_confirm if you looked but found no evidence, and "
        "clarify if the question itself is ambiguous. Pick one — starting with 'this cannot be confirmed' "
        "and then asserting lookup results in the same answer contradicts itself."
    )


def _no_final_retry(lookups: int, evidence_count: int, lang: str = "en") -> str:
    """The instruction to resubmit when final_answer was not called.  Both arguments are 'facts at the point of use'."""
    if lookups <= 0:
        fact = "No data lookup tool has been called for this question either."
    else:
        fact = f"Data lookups run for this question: {lookups}; their results are still in the conversation above."
        if evidence_count:
            fact += f" Recorded as evidence: {evidence_count}."
    return _NO_FINAL_RETRY_HEAD + fact + "\n" + _no_final_retry_tail(lang)


# ------------------------------------------- final_answer argument JSON mixed into the fallback text
# In G-007 the raw {"answer": "...", "status": ...} ended up in the answer field and was shown to the
# grader.  answer.strip_tool_call_syntax only peels leaks of the form `final_answer(answer="…")`, so this
# shape passes straight through.  It is unwrapped here separately.
_FINAL_JSON_KEYS = re.compile(r'"(answer|status)"\s*:\s*')
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}


def _json_string_at(text: str, i: int) -> str | None:
    """If text[i] is an opening double quote, return that JSON string's value (else None).

    json.loads is not used because the real incident was **truncated** JSON — G-007 was cut mid-sentence
    by max_tokens and json.loads cannot open it.  Without a closing quote, whatever is there is salvaged
    up to the end.
    """
    if i >= len(text) or text[i] != '"':
        return None
    buf: list[str] = []
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            esc = text[j + 1: j + 2]
            if not esc:
                break                                    # cut off in the middle of an escape
            if esc == "u":
                hexs = text[j + 2: j + 6]
                if len(hexs) == 4:
                    try:
                        buf.append(chr(int(hexs, 16)))
                        j += 6
                        continue
                    except ValueError:
                        pass
                j += 2
                continue
            buf.append(_JSON_ESCAPES.get(esc, esc))
            j += 2
            continue
        if ch == '"':
            break
        buf.append(ch)
        j += 1
    return "".join(buf)


def unwrap_final_answer_json(text: str) -> tuple[str, str] | None:
    """When final_answer's argument JSON arrives whole in the answer's place, unwrap it into (status, answer).

    Accepted variants: pure JSON, wrapped in a code fence, with explanatory sentences before or after,
    with line breaks escaped as backslash-n, and cut off in the middle.  If it cannot be unwrapped it
    returns None and the caller carries on exactly as now — **it never raises.**  If no status is found it
    returns an empty string and leaves the status decision to the caller's existing logic.
    """
    try:
        if not text or '"answer"' not in text:
            return None
        answer: str | None = None
        status: str | None = None
        for m in _FINAL_JSON_KEYS.finditer(text):
            val = _json_string_at(text, m.end())
            if val is None:
                continue                                 # non-string values (null etc.) are ignored
            if m.group(1) == "answer":
                if answer is None:
                    answer = val
            elif status is None:
                status = val
        if answer is None or not answer.strip():
            return None
        return (status or ""), answer
    except Exception:                                    # noqa: BLE001 - the fallback path must not die
        log.exception("final_answer JSON unwrap failed")
        return None


def _call_key(call: ToolCall):
    """The definition of 'the same call' = tool name + the normalised form of its arguments (purpose excluded).

    A looser definition (e.g. casefolding every argument and removing punctuation) would lump different
    search terms into one call and block legitimate attempts.  One false positive (handing back an
    unrelated earlier result and saying 'you are stuck') costs more than one false negative (the same
    state as now), so the strict side was chosen.  Only SQL is relaxed — see _norm_sql above — because
    re-sending the same query with only case or indentation changed is the form actually observed.
    """
    if call.name == "final_answer":
        return None  # the terminating call is not subject to the duplicate check
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
    return "(0 rows)" in c or "No search results" in c or "No matching documents" in c


def _repeat_note(n: int, res: ToolResult) -> str:
    """The note appended to a repeated call.  The next move is named according to what it got stuck on."""
    if res.is_error:
        nxt = ("Fix the SQL with the real column and view names the error message above gave. "
               "If unsure of a name, check the column list with table_info first.")
    elif _looks_empty(res):
        nxt = ("Drop the conditions one at a time to find which one makes the result 0 rows. "
               "A value may be spelled differently: check the allowed values with table_info and the actual "
               "spelling with resolve_entity or search_products, then apply the condition again.")
    else:
        nxt = "You already have this result. Use these values as they are and spend the remaining rounds on what you have not checked yet."
    return (f"\n\n⚠ This is identical call #{n} for this question. The result above was obtained by running it "
            f"again and is the same as the first time — the same input gives the same result however often it "
            f"is sent. A different approach is needed: {nxt}")


class Orchestrator:
    def __init__(self, settings, llm: LLMClient, tools: ToolRegistry, resolver, ontology, lexicon, system_prompt: str):
        self.settings, self.llm, self.tools = settings, llm, tools
        self.resolver, self.ontology, self.lexicon = resolver, ontology, lexicon
        self.system_prompt = system_prompt
        self.specs = tools.specs()
        self.base_date = (tools.meta.get("v_etf_kr") or {}).get("base_date") or "2026-08-21"
        # The number of products that have holdings.  To put a number in the injected text it has to be
        # counted from the data, not the ontology (the ontology knows 'which product groups have them',
        # not 'how many').
        self.holding_product_count = self._count_holding_products()
        # The lookup keys that actually work in the holdings view.  Some companies' Korean names in the
        # lexicon differ from the source spelling (캠브리콘 -> '한무기', 네이버 -> 'NAVER',
        # 에코프로머티리얼즈 -> '에코프로머티'), so putting the Korean name straight into WHERE returns 0 rows
        # and the model concludes 'not in the data'.  Looking it up per question would add about 12ms per
        # company (company_id has no index), while one GROUP BY at startup measures 16-24ms, so the map is
        # built in advance and a question only reads a dict (microseconds).
        self.constituent_keys = self._load_constituent_keys()

    def _count_holding_products(self) -> int | None:
        view = self.ontology.holding_view() if self.ontology is not None else None
        if not view or self.tools.rdb is None:
            return None
        try:
            rows = self.tools.rdb.fetch_all(f"SELECT COUNT(DISTINCT etf_itm_no) FROM {view}")
            return int(rows[0][0])
        except Exception:  # noqa: BLE001 - a store without the view simply gets no number
            return None

    def _load_constituent_keys(self) -> dict[str, list[tuple[str, str, int]]]:
        """{company id: [(company_code, company_name, row count), ...]} — the holdings view's real spellings.

        The keys are not guessed.  They read ``etf_constituent.company_id``, which the ETL already
        attached from the source's stock codes — the graph's holding edges are built from the same
        column, so the counts of the keys injected here and the graph counts already sent on the same
        line cannot disagree.  (Searching the view by the lexicon's ticker is possible too, but that is
        a guess: Airbus's ticker 'AIR' points to AAR in this data.)

        A store without that column gets an empty map, and with an empty map the injected text is as before.
        """
        if self.tools.rdb is None or self.lexicon is None:
            return {}
        try:
            known = {c.id for c in self.lexicon.companies}
            rows = self.tools.rdb.fetch_all(
                "SELECT company_id, company_code, company_name, COUNT(*) FROM etf_constituent "
                "WHERE is_cash = 0 AND company_id IS NOT NULL AND company_code IS NOT NULL "
                "GROUP BY 1, 2, 3"
            )
        except Exception:  # noqa: BLE001 - without the map no key is attached; the answer is not blocked
            log.exception("constituent key map unavailable - continuing without lookup keys")
            return {}
        out: dict[str, list[tuple[str, str, int]]] = {}
        for cid, code, name, n in rows:
            if cid in known:
                out.setdefault(cid, []).append((str(code), str(name or ""), int(n)))
        for v in out.values():
            v.sort(key=lambda x: -x[2])
        return out

    @property
    def _holding_view(self) -> str:
        view = self.ontology.holding_view() if self.ontology is not None else None
        return view or "v_etf_constituent"

    def _constituent_key_note(self, company) -> str:
        """How to find this company in the holdings view — one line, only facts read from the store.

        A company whose Korean name matches the view's spelling as is (61 of the lexicon's 97) gets
        **nothing attached**.  That path already queries correctly, and there is no reason to touch what
        works.  Only the 32 with a mismatched spelling and the 4 with no rows at all get the fact — the
        latter so the model reads 0 rows as the absence of data, not a faulty query.
        """
        if not self.constituent_keys:
            return ""
        rows = self.constituent_keys.get(company.id) or []
        view = self._holding_view
        if not rows:
            return (f". ⚠ {view} has 0 rows for this company — every lookup key returns 0 rows, and that is "
                    f"not a faulty query but the fact that the holdings data does not contain this company")
        if any(name == company.name_ko for _, name, _ in rows):
            return ""
        shown = rows[:3]
        codes = ", ".join(f"'{c}'" for c, _, _ in shown)
        names = " / ".join(f"'{n}'" for _, n, _ in shown)
        where = f"company_code = {codes}" if len(rows) == 1 else f"company_code IN ({codes})"
        more = f" (top {len(shown)} of {len(rows)} codes shown)" if len(rows) > len(shown) else ""
        return (f". Holdings lookup key: query {view} with {where}{more} "
                f"(this view's company_name value is {names} — a WHERE on the Korean name returns 0 rows)")

    def _subsidiary_key_line(self, subs) -> str:
        """The subsidiaries' lookup keys on one line.  Subsidiary names are not in the question, so they get no line of their own.

        In '에코프로의 자회사를 편입한 ETF' (ETFs holding 에코프로's subsidiaries), what the model really puts
        in WHERE are exactly the names listed on the line above — of the three, one is spelled differently
        in the view (에코프로머티리얼즈 -> '에코프로머티') and one has no rows at all.  Only when at least one
        is off, and then all of them are given (giving only some leaves the rest to guesswork again).
        """
        if not self.constituent_keys or not subs:
            return ""
        parts, off = [], False
        for s in subs[:6]:
            rows = self.constituent_keys.get(s.id) or []
            if not rows:
                parts.append(f"{s.name_ko} → 0 rows (not in the holdings data)")
                off = True
            elif any(n == s.name_ko for _, n, _ in rows):
                parts.append(f"{s.name_ko} → company_code='{rows[0][0]}'")
            else:
                parts.append(f"{s.name_ko} → company_code='{rows[0][0]}' (written '{rows[0][1]}' in the view)")
                off = True
        if not off:
            return ""
        return f"  · Subsidiaries' {self._holding_view} lookup keys: " + ", ".join(parts)

    @property
    def base_year(self) -> int:
        """The data's base year — the reference for judging whether a year like '2027년' is in the future (from the store meta)."""
        m = re.match(r"(\d{4})", self.base_date or "")
        return int(m.group(1)) if m else _BASE_YEAR_FALLBACK

    def _grade_mentions(self, question: str):
        """(credit ratings, risk grades) the question names, Korean and English phrasing."""
        credit = [(m.group(1) or m.group(2) or "").upper() for m in _GRADE_MENTION.finditer(question)]
        risk = [m.group(1) for m in _RISK_MENTION.finditer(question)]
        if is_english(question):
            en = fold_sentence_starts(question)
            for m in _GRADE_MENTION_EN.finditer(en):
                g = next(x for x in m.groups() if x).upper()
                if g not in credit:
                    credit.append(g)
            for m in _RISK_MENTION_EN.finditer(en):
                v = next(x for x in m.groups() if x)
                if v not in risk:
                    risk.append(v)
        return credit, risk

    # ------------------------------------------------------------ pre-analysis
    def pre_analyze(self, question: str) -> PreAnalysis:
        from ..kb.entity import extract_product_mentions

        lines: list[str] = []
        invalid: list[str] = []
        invalid_ko: list[str] = []
        steps: list[str] = []
        mentions = extract_product_mentions(question)
        unresolved: list[tuple[str, list[str]]] = []
        for m in mentions:
            r = self.resolver.resolve(m, k=4)
            resolved = r.resolved  # the EntityResolver owns the judgement (same criterion as the tool output)
            tag = "exact match" if r.exact else ("resolvable as a similar product" if resolved else "no matching product")
            lines.append(f"- Product mention '{m}': {tag}. " + ("; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in r.products[:4]) or "no candidates"))
            if not resolved:
                cands = [h.title for h in r.products[:3]]
                unresolved.append((m, cands))
                lines.append(
                    f"  ⚠ The product '{m}' does not exist in the data. The candidates above only have similar names and are "
                    f"different products, so answering with one of them as if it were the product asked about is wrong. "
                    f"Say that it does not exist, offer the candidates and ask back."
                )
        whole = self.resolver.resolve(question, k=3) if not mentions else None
        comps = self.lexicon.match_companies(question)
        groups = self._groups(question)
        # Does every product group the question names lack holdings data?  The basis of this judgement is
        # the ontology, not a sentence in the prompt (fp:hasConstituent's domain = fp:DomesticETF).
        holding_blind = bool(
            groups and mentions_holdings(question)
            and self.ontology is not None
            and not any(self.ontology.class_has_holdings(ci.name) for ci in groups)
        )
        holder_label = "/".join(c.label_ko for c in self.ontology.holding_classes()) if self.ontology is not None else ""
        for c in comps:
            subs = self.lexicon.subsidiaries_of(c.id)
            info = f"- Company '{c.name_ko}' ({c.name_en}{', unlisted' if c.not_listed else ''})"
            if c.parent:
                p = self.lexicon.company(c.parent)
                info += f", parent: {p.name_ko if p else c.parent}"
            if subs:
                info += ", subsidiaries: " + ", ".join(s.name_ko for s in subs)
            if self.tools.kg is not None and not holding_blind:
                # ★ The count comes from totals.  It used to be len() of a list cut at limit=100, so
                #   삼성전자 (really 238) was always injected as '100' — the cap became the answer.
                totals: dict[str, int] = {}
                self.tools.kg.neighbors(c.name_ko, relation="constituentOf", limit=1, totals=totals)
                n = totals.get("constituentOf", 0)
                info += f", held by {n} {holder_label or 'products'} (graph)" if n else ", held by no product (graph)"
            # The line that carries the resolved entity all the way to SQL.  Given only the label, the model
            # uses the question's surface string as the WHERE literal and gets 0 rows — the root defect
            # this project confirmed again and again.
            if not holding_blind:
                info += self._constituent_key_note(c)
            if c.note:
                info += f". Note: {c.note}"
            lines.append(info)
            if not holding_blind:
                sub_keys = self._subsidiary_key_line(subs)
                if sub_keys:
                    lines.append(sub_keys)
        if holding_blind:
            lines.append(self._no_holdings_note(groups, holder_label))
        themes = self.lexicon.match_themes(question)
        for t in themes:
            cnt = ""
            if self.tools.kg is not None:
                n = len(self.tools.kg.neighbors(f"d:theme/{t.id}", relation="hasTheme_inverse", limit=10000).get("hasTheme_inverse", []))
                cnt = f", {n} products tagged in the graph (graph_neighbors('{t.label_ko}','hasTheme_inverse'))"
            # A warning that "narrowing a theme by name LIKE returns 0 rows" was tried here and removed.  It
            # had no effect on 캠브리콘 (the model still tried name LIKE first) and broke 국민성장펀드 — the
            # path by which that question succeeded was exactly fund_family LIKE '%국민성장%', and the
            # warning blocked it.  Questions where narrowing by theme name is right and wrong are mixed, so
            # neither is forbidden in advance; only the synonyms and the graph count are given as facts.
            lines.append(f"- Theme '{t.label_ko}' ({t.label_en}; synonyms {', '.join(t.synonyms[:6])}){cnt}")
        mgrs = self.lexicon.match_managers(question)
        if mgrs:
            lines.append("- Managers: " + ", ".join(f"{m.canonical}(manager_name='{m.canonical}')" for m in mgrs))
        # ontology validators
        credit, risk = self._grade_mentions(question)
        for g in credit:
            if re.fullmatch(r"[A-D]{1,5}[+\-0]?", g) and self.ontology is not None:
                ok, msg = self.ontology.validate_credit_grade(g)
                if not ok:
                    codes = [e.code for e in self.ontology.credit_grades()]
                    invalid.append(f"Credit rating '{g}' does not exist. {msg}")
                    invalid_ko.append(f"신용등급 '{g}'는 존재하지 않는 값입니다. 유효한 신용등급: {', '.join(codes)}")
                else:
                    ge = self.ontology.credit_grades_at_or_above(g)
                    lines.append(f"- Credit rating '{g}' is valid. '{g} or higher' = crd_grd IN ({', '.join(repr(x) for x in ge)})")
        for v in risk:
            if self.ontology is not None:
                ok, msg = self.ontology.validate_risk_grade(v)
                if not ok:
                    codes = [e.code for e in self.ontology.risk_grades()]
                    rng = f"{codes[0]}~{codes[-1]}" if codes else "?"
                    invalid.append(f"Risk grade '{v}' does not exist. {msg}")
                    invalid_ko.append(f"위험등급 '{v}등급'은 존재하지 않습니다. 위험등급은 {rng}등급만 존재합니다 (1등급이 가장 높은 위험).")
        recent = _RECENT_MONTHS.search(question) or (_RECENT_MONTHS_EN.search(question) if is_english(question) else None)
        if recent:
            n = int(recent.group(1))
            base = dt.date.fromisoformat(self.base_date)
            month = base.month - n
            year = base.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            cutoff = base.replace(year=year, month=month, day=min(base.day, 28))
            lines.append(f"- 'last {n} months' = {cutoff.isoformat()} ~ {self.base_date} (counted from the data base date)")
        if whole is not None and whole.products and whole.products[0].score >= 60:
            lines.append("- Products similar to the whole question: " + "; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in whole.products[:3]))
        crit_phrase, crit = self._criterion_lines(question, groups)
        lines += crit
        if invalid:
            lines.insert(0, "- ⚠ Ontology violation: " + " / ".join(invalid))
        text = "[Pre-analysis — for reference; confirm the final judgement with tools]\n" + ("\n".join(lines) if lines else "- nothing notable")
        steps.append(
            "Pre-analysis: product mentions " + (", ".join(mentions) if mentions else "none")
            + (f"; companies {', '.join(c.name_ko for c in comps)}" if comps else "")
            + (f"; themes {', '.join(t.label_ko for t in themes)}" if themes else "")
            + (f"; product groups {', '.join(group_name(ci) for ci in groups)}" if groups else "")
            + ("; product group without holdings data" if holding_blind else "")
            + (f"; sort criterion '{crit_phrase}'" if crit_phrase else "")
            + (f"; {len(invalid)} ontology violation(s)" if invalid else "")
            + (f"; {len(unresolved)} nonexistent product(s)" if unresolved else "")
        )
        return PreAnalysis(text, invalid, mentions, steps, unresolved, invalid_ko)

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
        view = self.ontology.holding_view() or "the holdings view"
        named = ", ".join(f"{group_name(ci)}({', '.join(sorted(v))})" for ci, v in
                          ((ci, group_views([ci], self.ontology).get(ci.iri) or {"no view"}) for ci in groups))
        cnt = f" ({self.holding_product_count:,} products)" if self.holding_product_count else ""
        return (
            f"- ⚠ Holdings data scope: {named} have no holdings data. "
            f"Holdings exist only in {view}, covering {holder_label or '국내ETF'}{cnt} "
            f"(the domain of the ontology's fp:hasConstituent = {', '.join(c.name for c in self.ontology.holding_classes())}). "
            f"So a '… that holds X' question restricted to this product group is not '0 products' but **cannot be confirmed with the provided data**. "
            f"Presenting products with a similar name or theme as holders is wrong."
        )

    def _pick_criterion(self, question: str):
        """(phrase, resolution) for the candidate the ontology resolves best, or None."""
        best = None
        for phrase in criterion_phrases(question):
            # An English phrase is looked up through its Korean label words; the phrase shown stays as asked.
            res = self.ontology.resolve_measure(measure_phrase_ko(phrase) or phrase if is_english(question) else phrase)
            if res.status == "none":
                continue
            score = max((h.score for h in res.hits + res.other_periods), default=0)
            key = (_STATUS_RANK[res.status], score, len(phrase))
            if best is None or key > best[0]:
                best = (key, phrase, res)
        return None if best is None else (best[1], best[2])

    def _criterion_lines(self, question: str, groups) -> tuple[str | None, list[str]]:
        """(phrase, lines) for the ranking criterion — (None, []) when the question ranks nothing.

        Silence is the default and the common case: a question that asks for no ranking gets not a single line.
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
                # If the question names a product group, that view's columns come first.  With v_fund_pub
                # columns first for '국내 ETF 중 분배율' (distribution rate among domestic ETFs), a model
                # reading only the first line goes to the wrong view.
                scope = {v for s in group_views(groups, self.ontology).values() for v in s}
                hits = sorted(res.hits, key=lambda h: (h.prop.table not in scope, -h.score)) if scope else res.hits
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko}"
                                 + (f", unit {h.prop.unit}" if h.prop.unit else "") + ")" for h in hits[:6])
                out.append(f"- Sort criterion '{phrase}' → ontology label lookup: {cols}. Sort by this column and drop NULL and 0 from the candidates.")
                out += self._missing_in_groups(res, groups)
            elif res.status == "period_missing":
                have = ", ".join(sorted({f"{h.prop.table}.{h.prop.column}({h.prop.label_ko})"
                                         for h in res.other_periods if h.period})[:6])
                out.append(
                    f"- Sort criterion '{phrase}': the ontology has no column for the requested period ({res.period}). "
                    f"Columns of the same concept for other periods: {have or 'none'}. If you answer with another period, say so in the answer."
                )
            elif res.status == "ambiguous":
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko})" for h in res.hits[:10])
                out.append(
                    f"- Sort criterion '{phrase}' names no period or target, so in the ontology it maps to "
                    f"{len({(h.prop.table, h.prop.column) for h in res.hits})} columns: {cols}. "
                    f"Pick the column that fits the question's context and state the criterion in the answer's first sentence. "
                    f"If the context does not settle it either, do not pick one arbitrarily; ask back."
                )
            elif res.status == "weak":
                cols = ", ".join(f"{h.prop.table}.{h.prop.column}({h.prop.label_ko}"
                                 + (f", unit {h.prop.unit}" if h.prop.unit else "") + ")" for h in res.hits[:4])
                out.append(
                    f"- Sort criterion '{phrase}' has no column whose ontology label matches exactly. Close candidates: {cols}. "
                    f"**No label does not mean no data** — check with table_info before judging."
                )
            if out:
                n = requested_count(question)
                if n is not None:
                    out.append(f"- Requested count: {n} (LIMIT {n})")
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
                        f"  ⚠ {group_name(ci)}({view}) has no '{res.period} {res.base}' column — the same concept exists only as "
                        f"{', '.join(other[:4])}. Exclude it from the ranking and say so in the answer (do not fill the value with 0)."
                    )
                    continue
                probe = self.ontology.measure_candidates_in(res.base, view)
                if probe:
                    cand = ", ".join(f"{h.prop.column}({h.prop.label_ko})" for h in probe[:3])
                    out.append(
                        f"  · {group_name(ci)}({view}) has no column with the same label. Related candidates: {cand}. "
                        f"Do not conclude it is missing; check with table_info."
                    )
                else:
                    out.append(f"  ⚠ {group_name(ci)}({view}) has no column for this criterion. Exclude it from the ranking and say so.")
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
        # The answer language, decided once: stated to the model next to the question, and the language of
        # every answer sentence written below.  The system prompt stays identical across requests (cached).
        lang = detect_lang(question)

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
            pre = PreAnalysis("[Pre-analysis failed]", [], [], [f"Pre-analysis error: {type(e).__name__}"])
            steps += pre.steps
        emit(
            "pre_analysis",
            text=pre.text,
            mentions=list(pre.mentions),
            unresolved=[m for m, _ in pre.unresolved],
            invalid=list(pre.invalid),
            elapsed=time.monotonic() - t0,
        )

        # ---- round 0: the retrieval plan (agent/routing.py)
        # Only what the pre-analysis judged "this store is needed" is looked up in advance, and the
        # result goes into the first prompt as facts.  Unlike telling the model "use the graph" in the
        # prompt, this is honoured (this project's measurement: facts at the point of use 5/5 followed,
        # rules in advance 4/4 ignored).  On failure or over budget nothing is injected and the existing
        # tool loop continues as is.
        prefetch_block = ""
        prefetch_ran = 0
        try:
            pf = await route_prefetch(
                question, self.tools, resolver=self.resolver, lexicon=self.lexicon,
                ontology=self.ontology, mentions=list(pre.mentions),
                unresolved=[m for m, _ in pre.unresolved],
                forward_looking=asks_forward_looking(question, self.base_year),
            )
            steps += pf.steps
            prefetch_block = pf.block
            prefetch_ran = int(pf.ran or 0)   # lookups round 0 actually ran
            # Round-0 evidence does not go into retrieved_context: the tool loop usually repeats the same
            # lookup, so [Evidence] would be printed twice (an existing test counting evidence catches
            # that).  What round 0 looked up and why is shown by the 'Round 0' lines of think_trace.
            emit("prefetch", text=pf.block, ran=pf.ran, seconds=pf.seconds,
                 elapsed=time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001 - prefetching is an optimisation, not a dependency
            log.exception("retrieval routing failed")
            steps.append(f"Round 0 routing failed ({type(e).__name__}) — continuing with the normal flow")

        conv = self.llm.start(self.system_prompt, self.specs,
                              f"Question: {question}\nAnswer language: {language_name(lang)}\n\n{pre.text}"
                              + (f"\n\n{prefetch_block}" if prefetch_block else ""))
        last_text = ""
        nudged = False
        # key -> (how many times it has been called so far, result of the first run)
        seen_calls: dict = {}
        repeat_nudged = False
        # Data lookup calls actually *executed* for this question.  Calls that came in the same turn as
        # final_answer and were not executed do not count — a result not seen is the same as no evidence.
        # Lookups that round 0's prefetch actually ran count too.  Excluding them made the gate backfire:
        # for '국민성장펀드의 구조와 투자전략' (the structure and investment strategy of 국민성장펀드) round 0
        # fetched 2 documents and the answer was right, but when the gate sent it back for "0 lookups", the
        # model withdrew the right answer instead of looking up more and apologised "I confirm I did not
        # use the data lookup tools".  The prefetch is a real lookup the server ran deterministically, so
        # it is not a lack of evidence.
        lookups = 0
        forced_lookup = False                 # the forced retry happens at most once per question (no infinite loop)
        rejected_final: dict | None = None    # the answer sent back; used if the retry comes back empty-handed
        # State the final_answer gate below looks at.
        last_turn_ok = False                  # was a normal reply (not an error, refusal or timeout) received even once?
        pending_calls: list[ToolCall] = []    # calls whose results have not been returned yet (the protocol requires them)
        for rnd in range(1, self.settings.max_tool_rounds + 2):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                steps.append(f"Round {rnd}: time budget used up → forced stop")
                break
            try:
                turn = await asyncio.wait_for(conv.step(), timeout=max(3.0, remaining - 1.0))
            except asyncio.TimeoutError:
                steps.append(f"Round {rnd}: timed out waiting for the LLM")
                break
            if turn.stop_reason == "error":
                steps.append(f"Round {rnd}: LLM error — {turn.text[:120]}")
                break
            if turn.stop_reason == "refusal":
                steps.append(f"Round {rnd}: the model refused to answer")
                break
            last_turn_ok = True
            pending_calls = list(turn.tool_calls)
            if turn.text:
                last_text = turn.text
            fa = next((c for c in turn.tool_calls if c.name == "final_answer"), None)
            if fa is not None:
                final = fa.input or {}
                others = [c for c in turn.tool_calls if c.name != "final_answer"]
                steps.append(f"Round {rnd}: final_answer(status={final.get('status', '?')})" + (f" + {len(others)} ignored tool call(s)" if others else ""))
                remaining = deadline - time.monotonic()
                if lookups == 0 and prefetch_ran == 0 and not forced_lookup:
                    # Check both rounds and time for room to send it back.  For the retry to be useful at
                    # least one round to run tools must be left (rnd+1 <= max_tool_rounds).
                    if rnd >= self.settings.max_tool_rounds or remaining < FORCED_LOOKUP_MIN_REMAINING_SEC:
                        steps.append(
                            f"Round {rnd}: 0 data lookups, but not enough budget ({remaining:.0f}s left, "
                            f"at most {self.settings.max_tool_rounds} rounds) → forced retry skipped, answer accepted"
                        )
                        break
                    forced_lookup = True
                    rejected_final, final = final, None
                    conv.add_tool_results(
                        [(fa.id, _NO_LOOKUP_TOOL_RESULT, True)]
                        + [(c.id, _UNRUN_TOOL_RESULT, True) for c in others]
                    )
                    pending_calls = []
                    conv.add_user(_NO_LOOKUP_RETRY)
                    steps.append(f"Round {rnd}: final_answer with 0 data lookups → submission rejected, told to look up and resubmit")
                    emit("forced_lookup_retry", round=rnd,
                         status=str(rejected_final.get("status") or "?"),
                         elapsed=time.monotonic() - t0)
                    continue
                break
            if not turn.tool_calls:
                steps.append(f"Round {rnd}: text reply without a tool call")
                if rnd <= self.settings.max_tool_rounds and not nudged:
                    nudged = True
                    conv.add_user("Submit the answer with the final_answer tool (including status).")
                    continue
                break
            if rnd > self.settings.max_tool_rounds:
                steps.append(f"Round {rnd}: maximum rounds exceeded")
                break
            # Past this point every call of this turn runs in the gather below — count only what runs.
            lookups += sum(1 for c in turn.tool_calls if c.name in LOOKUP_TOOLS)
            # execute all tool calls of this turn concurrently
            for call in turn.tool_calls:
                emit("tool_call", round=rnd, name=call.name, args=call.input or {},
                     elapsed=time.monotonic() - t0)
            tool_t0 = time.monotonic()
            # A call made before still runs as usual; only 'which identical call this is' is appended to the
            # result.  Blocking execution would save a few milliseconds of tool time and remove one way for
            # the model to get out of a dead end (G-040).  The LLM round has been spent at this point — what
            # this buys is not time but a chance for the model to notice 'I am stuck' (see _repeat_note).
            keys = [_call_key(c) for c in turn.tool_calls]
            ordn: list[int] = []                      # which identical call this is for the question (1 = first)
            for i, key in enumerate(keys):
                if key is None:                       # calls exempt from the check, such as final_answer
                    ordn.append(1)
                    continue
                # count up to the previous round + identical calls earlier in this turn + itself
                ordn.append(seen_calls.get(key, 0) + sum(1 for j in range(i) if keys[j] == key) + 1)
            bases = await asyncio.gather(*(self._run_tool(c) for c in turn.tool_calls))
            # Repeats go back without evidence — the evidence was recorded on the first run, and adding it
            # again would print the same [Evidence] twice in retrieved_context.
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
                    steps.append(f"Round {rnd}: {desc} → identical call #{ncall} — re-run + told to change approach")
                    packed.append((call.id, res.content, res.is_error))
                    continue
                if res.is_error:
                    steps.append(f"Round {rnd}: {desc} → error: {res.content[:100]}")
                else:
                    n = f", {res.evidence.row_count} rows" if res.evidence and res.evidence.row_count is not None else ""
                    steps.append(f"Round {rnd}: {desc} → result {len(res.content)} chars{n}")
                    if res.evidence is not None and res.evidence.kind != "schema":
                        evidences.append(res.evidence)
                    if call.name == "sql_query":
                        sv = _scalar_of(res.content)
                        if sv is not None:
                            scalars.append(sv)
                packed.append((call.id, res.content, res.is_error))
            conv.add_tool_results(packed)
            pending_calls = []
            # On the 3rd identical call, steer toward wrapping up.  Guidance, not a forced stop, because
            # (1) the model sometimes changes course after the warning on the 2nd, and (2) a false
            # positive of the key normalisation must not throw away a whole question.  Sent once per question.
            if max(ordn, default=1) >= 3 and not repeat_nudged:
                repeat_nudged = True
                conv.add_user(
                    "[Same call repeated] The same tool call has come 3 times. This approach is making no progress. "
                    "Change the conditions or the tool and try once more; if that still yields no evidence, call "
                    "final_answer immediately (status=cannot_confirm if there is no evidence)."
                )
                steps.append(f"Round {rnd}: identical call 3 times → told to wrap up early")
            remaining = deadline - time.monotonic()
            if remaining < 14 or rnd == self.settings.max_tool_rounds:
                conv.add_user("[Time budget nearly spent] Call final_answer immediately with the evidence gathered so far, without further tool calls.")
                steps.append(f"Round {rnd}: time-budget warning sent ({remaining:.0f}s left)")

        # ---- final_answer gate: if the loop is about to end without final_answer, send it back once.
        # At most one send-back per question (no infinite loop) — if final_answer is still missing the
        # second time, the fallback below takes it as is.  The budget check is the zero-lookup gate's: this
        # send-back costs one more LLM round (measured ≈11 seconds), and losing the whole question by
        # exceeding the response time (the organisers recommend 60 seconds) is worse than one answer
        # without evidence.
        #
        # Not applied when rejected_final exists — the zero-lookup gate has already used a round then, and
        # the answer it sent back is restored as final just below, so the fallback path is never taken.
        # Not applied when last_turn_ok is False (LLM error, refusal, timeout before the first reply) — with
        # no working conversation, one more round only repeats the same failure.
        if final is None and rejected_final is None and last_turn_ok:
            remaining = deadline - time.monotonic()
            if remaining < FORCED_LOOKUP_MIN_REMAINING_SEC:
                steps.append(f"final_answer not called, but not enough budget ({remaining:.0f}s left) → resubmission request skipped")
            else:
                # OpenAI/CLOVA require a result for every assistant tool_call.  Calls left unexecuted
                # because the maximum rounds were exceeded are filled first (left empty, the next request fails with 400).
                if pending_calls:
                    conv.add_tool_results([(c.id, _UNRUN_TOOL_RESULT, True) for c in pending_calls])
                    pending_calls = []
                conv.add_user(_no_final_retry(lookups, len(evidences), lang))
                steps.append(
                    f"final_answer not called → asked to resubmit with the results so far ({lookups} lookups, {len(evidences)} evidence)"
                )
                emit("forced_final_retry", lookups=lookups, evidence=len(evidences),
                     elapsed=time.monotonic() - t0)
                try:
                    rturn = await asyncio.wait_for(conv.step(), timeout=max(3.0, remaining - 1.0))
                except asyncio.TimeoutError:
                    rturn = None
                    steps.append("Resubmission: timed out waiting for the LLM")
                if rturn is not None and rturn.stop_reason in ("error", "refusal"):
                    steps.append(f"Resubmission: LLM {rturn.stop_reason} — {rturn.text[:120]}")
                elif rturn is not None:
                    # The text from the resubmission is newer.  The G-009 incident was 'holding the lookup
                    # results already and letting out cannot-confirm', so switching to the reply given after
                    # being told about the lookups was judged better.
                    if rturn.text:
                        last_text = rturn.text
                    rfa = next((c for c in rturn.tool_calls if c.name == "final_answer"), None)
                    if rfa is not None:
                        final = rfa.input or {}
                        steps.append(f"Resubmission: final_answer(status={final.get('status', '?')}) → adopted")
                    else:
                        steps.append("No final_answer on resubmission either → falling back to the last text")
        # Do not lose the answer that was sent back.  If the retry round produced nothing (LLM error,
        # timeout, rounds used up), the original answer, even without evidence, beats empty hands.
        if final is None and rejected_final is not None:
            final = rejected_final
            steps.append("No final_answer after the forced retry → using the answer first submitted")
        # ---------------------------------------------------------- compose
        if final is not None:
            status = str(final.get("status") or "answered")
            if status not in ("answered", "cannot_confirm", "clarify"):
                status = "answered"
            answer = normalize_answer(status, str(final.get("answer") or last_text), lang)
            refs = [str(x) for x in (final.get("evidence_refs") or [])]
        elif last_text.strip():
            status = "answered" if evidences else "cannot_confirm"
            refs = []
            steps.append("final_answer not called → using the last text as the answer")
            # If that text is final_answer's argument JSON (G-007), do not send it out raw; unwrap it.
            # If it cannot be unwrapped nothing changes.  Either way it goes out through normalize_answer.
            body = last_text
            unwrapped = unwrap_final_answer_json(last_text)
            if unwrapped is not None:
                js, body = unwrapped
                if js in ("answered", "cannot_confirm", "clarify"):
                    status = js
                    steps.append(f"The last text is final_answer argument JSON → read as answer/status={js}")
                else:
                    steps.append("The last text is final_answer argument JSON → only answer read (status from the existing judgement)")
            answer = normalize_answer(status, body, lang)
        else:
            reason = (pick(lang, "time limit exceeded", "시간 초과") if time.monotonic() >= deadline
                      else pick(lang, "no response from the LLM", "LLM 응답 없음"))
            status, answer, refs = "cannot_confirm", fallback_answer(reason, lang), []
        # The old condition skipped this correction entirely when the answer contained "존재하지 않" (does
        # not exist).  But the model writes "존재하지 않습니다" and then lists substitute products — for an
        # 'AAAA bond' question 5 bonds went out with codes and yields attached, status stayed answered, and
        # normalize_answer's substitute-list cut never ran.  The organisers deduct points for generating an
        # answer to an unanswerable question (task description p7).  Once a value-domain violation is
        # confirmed the answer is lowered to cannot-confirm regardless of its wording, so the cut always applies.
        if pre.invalid and status == "answered":
            # The model's text is discarded.  With the refusal prefixed and the rest kept, the model writes
            # "존재하지 않습니다" and then lists substitute products — for the 'AAAA bond' question 2-5 bonds
            # went out with issuer, coupon and maturity.  The cut rule (3+ rows) cannot catch two of them
            # either.  pre.invalid already holds the violation and the list of valid values, so nothing is lost.
            answer = f"{cannot_prefix(lang)} {pick(lang, ' '.join(pre.invalid), ' '.join(pre.invalid_ko))}"
            status = "cannot_confirm"
            steps.append("Ontology-violating condition → answer corrected to cannot_confirm")
        # Catches the model answering with a similar product when the question named a product that does
        # not exist.  Unanswerable questions carry points, so this is not left to prompt compliance alone.
        if pre.unresolved and status == "answered" and not _acknowledged(answer):
            names = [m for m, _ in pre.unresolved]
            shown = [c for _, group in pre.unresolved for c in group[:2]]
            if shown:
                asked = pick(
                    lang,
                    f"Products with similar names include {', '.join(shown)}. Is one of these the product you are looking for?",
                    f"이름이 비슷한 상품으로는 {', '.join(shown)} 등이 있습니다. 이 중 찾으시는 상품이 있나요?",
                )
            else:
                asked = pick(lang, "Could you check the product name you are looking for?",
                             "찾으시는 상품명을 다시 확인해 주시겠어요?")
            answer = pick(
                lang,
                f"A product exactly matching '{', '.join(names)}' cannot be confirmed in the provided data. {asked}"
                f"\n\n(For reference, the following was looked up on the basis of a similar product)\n{answer}",
                f"'{', '.join(names)}'와(과) 정확히 일치하는 상품은 제공된 데이터에서 확인할 수 없습니다. {asked}"
                f"\n\n(참고: 아래는 유사 상품 기준으로 조회한 내용입니다)\n{answer}",
            )
            status = "clarify"
            steps.append(f"Answer used a nonexistent product ({', '.join(names)}) → corrected to a clarifying question")
        # Prevents the incident of answering a question about future values with a past-performance column
        # as is.  The number itself is a real value the tool produced, so the numeric cross-check passes.
        # What is wrong is the 'label', so status and wording are fixed here to make clear it is past
        # performance and keep it from reading as a forecast.
        if status == "answered" and asks_forward_looking(question, self.base_year):
            answer = pick(
                lang,
                f"{cannot_prefix(lang)} The provided data is a snapshot as of the base date {self.base_date} and has no "
                "item holding a value to be realised in the future, such as an expected return or a target price. "
                "Past returns do not guarantee future returns, so they cannot be presented as a forecast."
                f"\n\n(For reference) The following is not a forecast but past and current results recorded in the data.\n{answer}",
                f"{cannot_prefix(lang)} 제공된 데이터는 기준일 {self.base_date} 시점의 스냅샷이며, "
                "예상 수익률·목표가처럼 미래에 실현될 값을 담은 항목이 없습니다. "
                "과거 수익률은 미래 수익을 보장하지 않으므로 예측치로 제시할 수 없습니다."
                f"\n\n(참고) 아래는 예측이 아니라 데이터에 기록된 과거·현재 실적입니다.\n{answer}",
            )
            status = "cannot_confirm"
            steps.append("Question asks for a forecast → no forecast column in the data, corrected to cannot_confirm (past results kept for reference)")
        # Holding relations exist only for domestic ETFs.  Building a plausible answer to a holdings
        # question narrowed to foreign ETFs, funds or ETNs either presents a list of domestic ETFs as foreign
        # ETFs (G-040) or asserts that a fund with a merely similar name 'holds that stock' (G-041).  State
        # the schema's fact as it is.
        # The model's answer is thrown away, not appended.  Even under a "(참고)" (for reference) heading the
        # sentences below are still assertions, and a wrong one really came out: "SK하이닉스를 담고 있는
        # 공모펀드 중 순자산이 가장 큰 상품은 KODEX 200 입니다" — KODEX 200 is an ETF, not a public fund.
        # The organisers state that generating an answer to an unanswerable question costs points, so an
        # answer hidden behind a note carries the same risk.  The refusal above holds all the facts, so
        # nothing is lost.
        if status == "answered" and asks_holdings_without_data(question):
            answer = pick(
                lang,
                f"{cannot_prefix(lang)} Holdings (constituent) data exists only for 1,162 domestic ETFs (collected from "
                "Naver Finance). The holdings of foreign ETFs, public funds, private funds and ETNs are not in the data "
                "provided, so whether they hold a particular company, and how they rank, cannot be confirmed.",
                f"{cannot_prefix(lang)} 편입 종목(구성종목) 데이터는 국내 ETF 1,162 종목(네이버금융 수집)"
                "에만 있습니다. 해외 ETF·공모펀드·사모펀드·ETN 의 보유 종목은 전달받은 데이터에 "
                "없으므로, 특정 기업을 편입했는지 여부와 그 순위는 확인할 수 없습니다.",
            )
            status = "cannot_confirm"
            steps.append("Holdings question on a product group without holdings data (foreign ETFs, funds, ETNs) → corrected to cannot_confirm")
        # A count question whose single COUNT result never appears in the answer means the model
        # mis-transcribed it.  Only fires when exactly one scalar was produced, so a multi-step
        # calculation is never "corrected" against an intermediate value.
        if status == "answered" and is_count_question(question) and len(set(scalars)) == 1:
            want = scalars[0]
            seen = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", answer)}
            if want not in seen:
                answer = pick(
                    lang,
                    f"{answer}\n\n(Checked against the query result: {want} items match the condition.)",
                    f"{answer}\n\n(조회 결과 확인: 해당 조건에 맞는 건수는 {want}건입니다.)",
                )
                steps.append(f"The {want} returned by the tool is missing from the answer → query result stated")
        elapsed = time.monotonic() - t0
        steps.append(f"Final: status={status}, {len(evidences)} evidence, {elapsed:.1f}s")
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
