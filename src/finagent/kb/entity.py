"""Entity resolution: a free-text mention -> typed candidates (products, companies, managers, themes)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import Company, Lexicon, Manager, Theme
from .search import Hit, SearchIndex, norm_title, tokenize


@dataclass
class Resolution:
    mention: str
    products: list[Hit] = field(default_factory=list)
    companies: list[Company] = field(default_factory=list)
    managers: list[Manager] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    exact: bool = False

    def summary(self, max_products: int = 5) -> str:
        parts = []
        if self.products:
            parts.append(
                ("정확히 일치하는 상품: " if self.exact else "유사 상품 후보(정확 일치 아님): ")
                + "; ".join(f"[{h.table}] {h.title} (id={h.id}, score={h.score:.1f})" for h in self.products[:max_products])
            )
        else:
            parts.append("상품명 일치/유사 후보 없음")
        if self.companies:
            parts.append(
                "기업: "
                + ", ".join(
                    f"{c.name_ko}({c.name_en}{', 비상장' if c.not_listed else ''}{', 모회사=' + c.parent if c.parent else ''})"
                    for c in self.companies
                )
            )
        if self.managers:
            parts.append("운용사: " + ", ".join(m.canonical for m in self.managers))
        if self.themes:
            parts.append("테마: " + ", ".join(f"{t.label_ko}/{t.label_en}" for t in self.themes))
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
