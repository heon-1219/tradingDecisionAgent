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
        self._theme_syn = sorted(
            ((norm(s), t) for t in themes for s in [t.label_ko, t.label_en, *t.synonyms] if norm(s)),
            key=lambda x: -len(x[0]),
        )
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
        n = norm(text)
        seen: set[str] = set()
        out: list[Theme] = []
        for syn, t in self._theme_syn:
            if len(syn) >= min_len and syn in n and t.id not in seen:
                seen.add(t.id)
                out.append(t)
        return out

    def match_companies(self, text: str, min_len: int = 2) -> list[Company]:
        n = norm(text)
        seen: set[str] = set()
        out: list[Company] = []
        for a, c in self._comp_alias:
            if len(a) >= min_len and a in n and c.id not in seen:
                seen.add(c.id)
                out.append(c)
        return out

    def match_managers(self, text: str, min_len: int = 2) -> list[Manager]:
        n = norm(text)
        seen: set[str] = set()
        out: list[Manager] = []
        for a, m in self._mgr_alias:
            if len(a) >= min_len and a in n and m.id not in seen:
                seen.add(m.id)
                out.append(m)
        return out

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

    def company_by_alias(self, name: str | None) -> Company | None:
        n = norm(name)
        if not n:
            return None
        for a, c in self._comp_alias:
            if a == n:
                return c
        for a, c in self._comp_alias:
            if len(a) >= 3 and (a in n or n in a):
                return c
        return None
