"""Curated lexicons (managers, themes, companies) loaded from YAML and simple matchers.

These are the hand-maintained "entity resolution" layer: they map the many surface forms found in
product names and user questions (브랜드명, 약칭, 영문명, 한자명 ...) to canonical entities that the
knowledge graph and the agent can reason about.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return re.sub(r"\s+", "", s)


def _is_ascii(s: str) -> bool:
    return all(ord(ch) < 128 for ch in s)


class _Matcher:
    """Korean/CJK aliases match as substrings of the space-stripped text; ASCII aliases match on
    word boundaries of the lower-cased original text (so 'AI' does not hit 'Daily', 'EV' not 'Everything')."""

    def __init__(self, pairs):
        self.sub = []  # (norm alias, obj)
        self.rx = []  # (compiled regex, obj)
        for alias, obj in pairs:
            a = unicodedata.normalize("NFKC", str(alias or "")).strip().lower()
            if not a:
                continue
            if _is_ascii(a):
                pat = r"(?<![a-z0-9])" + r"\s*".join(re.escape(w) for w in a.split()) + r"(?![a-z0-9])"
                self.rx.append((re.compile(pat), obj, a))
            else:
                self.sub.append((norm(a), obj))
        self.sub.sort(key=lambda x: -len(x[0]))

    def match(self, text: str, min_len: int = 2):
        n = norm(text)
        low = unicodedata.normalize("NFKC", str(text or "")).lower()
        seen, out = set(), []
        for a, obj in self.sub:
            if len(a) >= min_len and a in n and obj.id not in seen:
                seen.add(obj.id)
                out.append(obj)
        for rx, obj, a in self.rx:
            if len(a) >= min_len and obj.id not in seen and rx.search(low):
                seen.add(obj.id)
                out.append(obj)
        return out


@dataclass
class Manager:
    id: str
    canonical: str
    group: str | None = None
    aliases: list[str] = field(default_factory=list)
    country: str | None = None


@dataclass
class Theme:
    id: str
    label_ko: str
    label_en: str
    synonyms: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class Company:
    id: str
    name_ko: str
    name_en: str
    aliases: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    parent: str | None = None
    country: str | None = None
    not_listed: bool = False
    note: str | None = None


class Lexicon:
    def __init__(self, managers: list[Manager], themes: list[Theme], companies: list[Company]):
        self.managers, self.themes, self.companies = managers, themes, companies
        self._mgr_alias = sorted(
            ((norm(a), m) for m in managers for a in [m.canonical, *m.aliases] if norm(a)),
            key=lambda x: -len(x[0]),
        )
        self._theme_m = _Matcher((s, t) for t in themes for s in [t.label_ko, t.label_en, *t.synonyms])
        self._comp_m = _Matcher((a, c) for c in companies for a in [c.name_ko, c.name_en, *c.aliases])
        self._mgr_m = _Matcher((a, m) for m in managers for a in [m.canonical, *m.aliases])
        self._comp_alias = sorted(
            ((norm(a), c) for c in companies for a in [c.name_ko, c.name_en, *c.aliases] if norm(a)),
            key=lambda x: -len(x[0]),
        )
        self._company_by_id = {c.id: c for c in companies}
        self._theme_by_id = {t.id: t for t in themes}
        self._manager_by_id = {m.id: m for m in managers}

    @classmethod
    def load(cls, d: Path) -> "Lexicon":
        def rd(name: str):
            p = Path(d) / name
            return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else []

        managers = [Manager(**m) for m in (rd("managers.yaml") or [])]
        themes = [Theme(**t) for t in (rd("themes.yaml") or [])]
        companies = [Company(**{**c, "tickers": [str(x) for x in c.get("tickers", [])]}) for c in (rd("companies.yaml") or [])]
        return cls(managers, themes, companies)

    # --- matching -------------------------------------------------------
    def match_themes(self, text: str, min_len: int = 2) -> list[Theme]:
        return self._theme_m.match(text, min_len)

    def match_companies(self, text: str, min_len: int = 2) -> list[Company]:
        return self._comp_m.match(text, min_len)

    def match_managers(self, text: str, min_len: int = 2) -> list[Manager]:
        return self._mgr_m.match(text, min_len)

    def manager_for_product_name(self, name: str | None) -> str | None:
        """Longest-alias prefix match: '미래에셋국민참여형...' -> 미래에셋자산운용, 'KODEX 로봇' -> 삼성자산운용."""
        n = norm(name)
        if not n:
            return None
        for a, m in self._mgr_alias:  # sorted longest first
            if len(a) >= 2 and n.startswith(a):
                return m.canonical
        return None

    def manager_canonical(self, raw: str | None) -> str | None:
        """Exact alias lookup for a raw manager string (e.g. '미래에셋증권 주식회사')."""
        n = norm(raw)
        if not n:
            return None
        for a, m in self._mgr_alias:
            if a == n:
                return m.canonical
        return None

    # --- relations ------------------------------------------------------
    def subsidiaries_of(self, company_id: str) -> list[Company]:
        return [c for c in self.companies if c.parent == company_id]

    def company(self, company_id: str) -> Company | None:
        return self._company_by_id.get(company_id)

    def theme(self, theme_id: str) -> Theme | None:
        return self._theme_by_id.get(theme_id)

    def manager(self, manager_id: str) -> Manager | None:
        return self._manager_by_id.get(manager_id)

    # ``company_by_alias`` used to live here: an exact-alias lookup with a substring fallback.  It was
    # written for question-side matching, where being forgiving is right, and then reused to identify
    # 15,911 rows of ingested constituent data, where being forgiving is silent corruption — 677 rows
    # landed under the wrong company because a two-letter name is a substring of a long one
    # ("lg" ⊂ "shinhanfinancia|lg|roup").  Constituents now carry the source's stock code, so nothing
    # needs to guess.  Use ``match_companies`` for text from a user: it anchors ASCII aliases on word
    # boundaries and drops one-character ones.
