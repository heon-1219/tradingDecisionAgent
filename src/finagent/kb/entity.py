"""Entity resolution: a free-text mention -> typed candidates (products, companies, managers, themes)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import Company, Lexicon, Manager, Theme
from .search import Hit, SearchIndex, _strip_generic, norm_title, tokenize


def mention_resolves(mention: str, title: str) -> bool:
    """Whether a mention and a candidate product name effectively refer to the same thing.

    'KODEX AI로봇' vs 'KODEX 로봇액티브' do not contain each other, so they do not match (= a product
    that does not exist); '미래에셋 TIGER 미국나스닥100' vs 'TIGER 미국나스닥100' and '국민성장펀드' vs
    '미래에셋국민참여형국민성장혼합자산투자신탁…' count as matches (they differ only by a manager prefix / generic suffix).
    """
    m, t = norm_title(mention), norm_title(title)
    core = _strip_generic(m)
    return bool(t) and (t in m or (len(core) >= 2 and core in t))


@dataclass
class Resolution:
    mention: str
    products: list[Hit] = field(default_factory=list)
    companies: list[Company] = field(default_factory=list)
    managers: list[Manager] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    exact: bool = False

    @property
    def resolved(self) -> bool:
        """Does the mentioned product actually exist?  Looser than ``exact``, but rules out a different product that merely has a similar name."""
        return self.exact or any(mention_resolves(self.mention, h.title) for h in self.products[:3])

    def summary(self, max_products: int = 5) -> str:
        parts = []
        if self.products:
            parts.append(
                ("Exact product match: " if self.exact else "Similar product candidates (not an exact match): ")
                + "; ".join(f"[{h.table}] {h.title} (id={h.id}, score={h.score:.1f})" for h in self.products[:max_products])
            )
            if not self.resolved:
                parts.append(
                    f"⚠ No product named '{self.mention}' exists in the data. The candidates above only have similar "
                    "names and are different products, so answering as if one of them were the product in the question is wrong. "
                    "State that it does not exist, present the candidates and ask the user which one they meant."
                )
        else:
            parts.append("No exact or similar product-name candidates")
        if self.companies:
            parts.append(
                "Companies: "
                + ", ".join(
                    f"{c.name_ko}({c.name_en}{', unlisted' if c.not_listed else ''}{', parent=' + c.parent if c.parent else ''})"
                    for c in self.companies
                )
            )
        if self.managers:
            parts.append("Managers: " + ", ".join(m.canonical for m in self.managers))
        if self.themes:
            parts.append("Themes: " + ", ".join(f"{t.label_ko}/{t.label_en}" for t in self.themes))
        return "\n".join(parts)


_STOP = re.compile(
    r"(etf|etn|펀드|채권|상품|정보|알려\s*줘|알려|주세요|뭐야|있어|있나요|찾아|정리해\s*줘|비교해\s*줘|추천해\s*줘|\?|\s|을|를|의|은|는|이|가)+",
    re.I,
)


_BRANDS = (
    "KODEX|TIGER|RISE|KBSTAR|ACE|KINDEX|SOL|PLUS|ARIRANG|KIWOOM|KOSEF|HANARO|1Q|WON|KoAct|TIME|UNICORN|FOCUS|IBK|BNK|"
    "TIMEFOLIO|삼성|미래에셋|신한|한화|키움|하나|우리|대신|메리츠|KB|iShares|SPDR|Vanguard|Invesco|Global X|KraneShares|"
    "ProShares|Direxion|VanEck|WisdomTree|ARK|Schwab|JPMorgan|First Trust|Xtrackers|Roundhill|Defiance|YieldMax"
)
_BRAND_MENTION = re.compile(rf"\b(?:{_BRANDS})\s+[A-Za-z0-9가-힣&+\-.()]+(?:\s+[A-Za-z0-9가-힣&+\-.()]+){{0,3}}", re.I)
_FUND_MENTION = re.compile(r"[A-Za-z0-9가-힣&+\-.()]{2,}(?:펀드|투자신탁)(?![가-힣]*(?:마스터|데이터))")
_EN_ETF_MENTION = re.compile(r"\b[A-Z][A-Za-z0-9&+\-.]*(?:\s+[A-Za-z0-9&+\-.]+){0,6}\s+ETF\b")
_TRAIL = re.compile(r"(의|은|는|이|가|을|를|에|도|와|과|로|에서)$")
_STOP_WORDS = {"etf", "etn", "정보", "알려줘", "알려", "주세요", "찾아서", "비교해줘", "추천해줘", "정리해줘", "구조와", "총보수는?", "총보수는"}

# Category words, not product names.  "…펀드" matches the fund pattern, so "환헤지형 해외 채권형 공모펀드
# 중 판매중인 상품은 몇 개?" used to extract 공모펀드, fail to resolve it, and get rewritten into a
# clarification ("'공모펀드'와 일치하는 상품이 없습니다") on top of an otherwise correct count.
_GENERIC_FUND_TERMS = {
    "공모펀드", "사모펀드", "공모형펀드", "사모형펀드", "국내펀드", "해외펀드", "주식형펀드", "채권형펀드",
    "혼합형펀드", "혼합자산펀드", "인덱스펀드", "재간접펀드", "머니마켓펀드", "단기금융펀드", "부동산펀드",
    "특별자산펀드", "연금펀드", "적립식펀드", "거치식펀드", "액티브펀드", "패시브펀드", "상장지수펀드",
    "개방형펀드", "폐쇄형펀드", "모펀드", "자펀드", "국내주식형펀드", "해외주식형펀드", "국내채권형펀드",
    "해외채권형펀드", "환헤지펀드", "펀드", "투자신탁",
}


def _is_generic_category(span: str) -> bool:
    return span.replace(" ", "").lower() in _GENERIC_FUND_TERMS


def extract_product_mentions(question: str) -> list[str]:
    """Pull likely product-name spans out of a question (brand + name, ...펀드, English '... ETF')."""
    out: list[str] = []
    for rx in (_EN_ETF_MENTION, _BRAND_MENTION, _FUND_MENTION):
        for m in rx.finditer(question):
            span = m.group(0).strip()
            cut: list[str] = []
            for w in span.split():
                if w.lower() in ("etf", "etn"):
                    if rx is _EN_ETF_MENTION:  # keep the trailing ETF of an English fund name
                        cut.append(w)
                    break
                if w.lower() in _STOP_WORDS:
                    break
                stripped = _TRAIL.sub("", w) if re.search(r"[가-힣]$", w) and len(w) >= 3 else w
                cut.append(stripped)
                if stripped != w:  # a particle ended the product name
                    break
            span = " ".join(cut).strip(" ,.?")
            if _is_generic_category(span):
                continue
            if len(span) >= 2 and span not in out and not any(span in o for o in out):
                out.append(span)
    return out


class EntityResolver:
    def __init__(self, index: SearchIndex, lexicon: Lexicon):
        self.index, self.lexicon = index, lexicon

    def resolve(self, mention: str, k: int = 5, table: str | None = None) -> Resolution:
        r = Resolution(
            mention,
            companies=self.lexicon.match_companies(mention),
            managers=self.lexicon.match_managers(mention),
            themes=self.lexicon.match_themes(mention),
        )
        hits = self.index.search(mention, k=k * 2, table=table)
        # keep only candidates that genuinely share name material with the mention (>= 2 shared tokens),
        # so a company mention like '캠브리콘' does not surface unrelated products via a single bigram
        qtoks = {t for t in tokenize(_STOP.sub(" ", mention)) if len(t) >= 2}
        kept = [h for h in hits if h.score >= 100.0 or len(qtoks & set(tokenize(h.title))) >= 2]
        r.products = kept[:k]
        core = norm_title(_STOP.sub("", mention))
        if r.products:
            top = r.products[0]
            r.exact = top.score >= 100.0 or (len(core) >= 3 and norm_title(top.title) == core)
        return r
