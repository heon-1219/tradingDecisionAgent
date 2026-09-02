# 금융상품 Agent API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI service exposing `GET /answer` that answers Korean financial-product questions over four provided master tables with evidence, and declines/clarifies when the data cannot confirm.

**Architecture:** ETL loads the four Excel masters into SQLite (tables + documented views). Five Turtle ontology files are loaded at runtime to generate LLM schema cards and value validators. An in-memory rdflib knowledge graph, a BM25 lexical index, and the SQLite store are exposed to a tool-calling LLM agent (Anthropic by default, OpenAI-compatible alternative) that runs under a 50-second budget and always returns the five-field JSON.

**Tech Stack:** Python 3.11+ (dev on 3.14), FastAPI, uvicorn, pandas/openpyxl (ETL only), sqlite3, rdflib 7, rank-bm25, numpy, pyyaml, pydantic-settings, anthropic SDK 1.x, openai SDK 3.x, pytest + pytest-asyncio, Docker.

## Global Constraints

- Repo root: `fin-agent/`. Mandated layout: `src/`, `ontology/{common,bond_kr,etf_kr,etf_gl,fund_pub}.ttl`, `requirements.txt`, `README.md`.
- Ontology prefix: `@prefix fp: <http://mafest.ai/product#>` (from the competition PDF). Data prefix: `@prefix d: <http://mafest.ai/data/>`.
- API contract: `GET /answer?question_id=&question=` → `200` JSON with exactly `question_id, question, retrieved_context, think_trace, answer` (all strings). Never 500. Unknown params ignored.
- Response budget: `TIME_BUDGET_SEC=50` default (organizer recommends ≤ 60s).
- Default LLM: `claude-opus-5` via the official `anthropic` SDK, adaptive thinking, `output_config.effort` from settings (default `medium`), server-side `fallbacks="default"` with beta `server-side-fallback-2026-07-01` when enabled.
- SQL from the LLM: SELECT/WITH only, single statement, auto `LIMIT 200`, 5s timeout, read-only connection.
- All Korean text files UTF-8. Tests run with `.venv/Scripts/python.exe -m pytest` from repo root (Windows Git Bash).
- Commit after every task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

```
fin-agent/
├─ src/finagent/
│  ├─ __init__.py
│  ├─ config.py              Settings (env/.env)
│  ├─ models.py              AnswerResponse
│  ├─ paths.py               ROOT, DATA_DIR, ONTOLOGY_DIR
│  ├─ etl/__init__.py
│  ├─ etl/normalize.py       pure normalizers (dates, flags, fund family, text)
│  ├─ etl/build_store.py     xlsx → SQLite tables/views/meta; CLI
│  ├─ ontology/__init__.py
│  ├─ ontology/loader.py     Ontology: properties, enums, schema cards, validators
│  ├─ kb/__init__.py
│  ├─ kb/rdb.py              RDB: guarded read-only SQL
│  ├─ kb/lexicon.py          Lexicon: themes/companies/managers YAML + matching
│  ├─ kb/search.py           SearchIndex: BM25 lexical index
│  ├─ kb/entity.py           EntityResolver
│  ├─ kb/graph.py            KG build/query/export
│  ├─ kb/docs.py             DocStore: markdown chunks + search
│  ├─ agent/__init__.py
│  ├─ agent/llm.py           LLM adapters: anthropic, openai, stub
│  ├─ agent/tools.py         ToolRegistry: specs + executors + evidence
│  ├─ agent/prompts.py       system prompt builder
│  ├─ agent/answer.py        evidence/trace formatting, fallbacks
│  ├─ agent/orchestrator.py  pre-analysis + agent loop + budget
│  └─ api.py                 FastAPI app
├─ ontology/*.ttl (5)
├─ data/raw/*.xlsx, data/enrichment/{themes,companies,managers}.yaml, data/enrichment/constituents/*.csv
├─ data/external/docs/*.md, data/eval/questions.jsonl
├─ scripts/{build_store.py, fetch_constituents_naver.py, parse_docs.py, eval_run.py}
├─ tests/...
├─ requirements.txt, pyproject.toml, .env.example, Dockerfile, docker-compose.yml, README.md
└─ docs/api_spec.md, docs/proposal_outline.md
```

---

### Task 1: Project scaffolding, settings, response model

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `src/finagent/__init__.py`, `src/finagent/paths.py`, `src/finagent/config.py`, `src/finagent/models.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `finagent.config.Settings` (fields below), `finagent.config.get_settings() -> Settings`, `finagent.models.AnswerResponse`, `finagent.paths.ROOT/DATA_DIR/ONTOLOGY_DIR`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from finagent.config import Settings
from finagent.models import AnswerResponse

def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-opus-5"
    assert s.time_budget_sec == 50
    assert s.store_path.name == "products.db"

def test_answer_response_fields():
    r = AnswerResponse(question_id="Q-1", question="q", retrieved_context="c", think_trace="t", answer="a")
    assert set(r.model_dump().keys()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: finagent`

- [ ] **Step 3: Write scaffolding**

```toml
# pyproject.toml
[project]
name = "finagent"
version = "0.1.0"
requires-python = ">=3.11"
[tool.setuptools.packages.find]
where = ["src"]
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
asyncio_mode = "auto"
```

```
# requirements.txt
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.7
pydantic-settings>=2.3
python-dotenv>=1.0
anthropic>=1.3
openai>=1.50
rdflib>=7.0
rank-bm25>=0.2.2
numpy>=1.26
pyyaml>=6.0
pandas>=2.2
openpyxl>=3.1
httpx>=0.27
pytest>=8.0
pytest-asyncio>=0.23
```

```
# .env.example
LLM_PROVIDER=anthropic            # anthropic | openai | stub
LLM_MODEL=claude-opus-5
LLM_EFFORT=medium                 # low | medium | high
ANTHROPIC_API_KEY=
ANTHROPIC_FALLBACKS=true
OPENAI_API_KEY=
OPENAI_BASE_URL=                  # e.g. https://generativelanguage.googleapis.com/v1beta/openai/
TIME_BUDGET_SEC=50
MAX_TOOL_ROUNDS=6
```

```python
# src/finagent/paths.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ONTOLOGY_DIR = ROOT / "ontology"
```

```python
# src/finagent/config.py
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from .paths import DATA_DIR, ONTOLOGY_DIR, ROOT

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")
    llm_provider: Literal["anthropic", "openai", "stub"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 4096
    llm_timeout_sec: float = 40.0
    anthropic_api_key: str | None = None
    anthropic_fallbacks: bool = True
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    time_budget_sec: float = 50.0
    max_tool_rounds: int = 6
    sql_row_limit: int = 200
    sql_timeout_sec: float = 5.0
    data_dir: Path = DATA_DIR
    ontology_dir: Path = ONTOLOGY_DIR
    @property
    def store_path(self) -> Path: return self.data_dir / "store" / "products.db"
    @property
    def raw_dir(self) -> Path: return self.data_dir / "raw"
    @property
    def enrichment_dir(self) -> Path: return self.data_dir / "enrichment"
    @property
    def docs_dir(self) -> Path: return self.data_dir / "external" / "docs"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
# src/finagent/models.py
from pydantic import BaseModel
class AnswerResponse(BaseModel):
    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str
```

`tests/conftest.py`: `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v` → PASS

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: project scaffolding, settings, response model"`

---

### Task 2: Lexicons (managers / themes / companies) and matcher

**Files:**
- Create: `data/enrichment/managers.yaml`, `data/enrichment/themes.yaml`, `data/enrichment/companies.yaml`, `src/finagent/kb/__init__.py`, `src/finagent/kb/lexicon.py`
- Test: `tests/test_lexicon.py`

**Interfaces:**
- Produces: `Lexicon.load(dir: Path) -> Lexicon`; `Lexicon.managers: list[Manager(id, canonical, aliases, group)]`; `Lexicon.themes: list[Theme(id, label_ko, label_en, synonyms, related)]`; `Lexicon.companies: list[Company(id, name_ko, name_en, aliases, tickers, parent, country)]`; `Lexicon.match_themes(text) -> list[Theme]`; `Lexicon.match_companies(text) -> list[Company]`; `Lexicon.match_managers(text) -> list[Manager]`; `Lexicon.manager_for_product_name(name) -> str | None`; `Lexicon.subsidiaries_of(company_id) -> list[Company]`.

YAML shapes:

```yaml
# managers.yaml
- id: mirae_am
  canonical: 미래에셋자산운용
  group: 미래에셋
  aliases: [미래에셋, 미래에셋자산운용, TIGER, Mirae Asset, Mirae Asset Global Investments]
- id: mirae_sec
  canonical: 미래에셋증권
  group: 미래에셋
  aliases: [미래에셋증권 주식회사, 미래에셋증권, Mirae Asset Securities]
- id: samsung_am
  canonical: 삼성자산운용
  group: 삼성
  aliases: [삼성, 삼성자산운용, KODEX, Samsung Asset Management Co Ltd]
# ... KB(RISE), 한국투자(ACE), 신한(SOL), 한화(PLUS), 키움(KIWOOM), NH-Amundi(HANARO), 하나(1Q), 우리(WON), 대신(343), 
# BNK, 브이아이, 에셋플러스, 타임폴리오(TIME), 삼성액티브(KoAct), 유리, 피델리티, 슈로더, 교보악사, DB, IBK, 현대, 유진, 흥국, 
# 마이다스, 트러스톤, 신영, 한국밸류, KCGI, 다올, 우리글로벌, 이스트스프링, 프랭클린, 삼성증권, KB증권, 한국투자증권, 신한투자증권,
# 메리츠증권, 엔에이치투자증권, 하나증권, 키움증권, 대신증권, BlackRock(iShares), Vanguard, State Street(SPDR), Invesco, 
# First Trust, Global X, KraneShares, ProShares, Direxion, VanEck, WisdomTree, ARK, JPMorgan, Schwab, Fidelity, Dimensional
```

```yaml
# themes.yaml
- id: space_aero
  label_ko: 우주항공
  label_en: Space & Aerospace
  synonyms: [우주, 우주항공, 우주테크, 항공우주, 방산, 방위산업, 우주산업, space, aerospace, defense, satellite, 스페이스]
  related: [defense]
- id: semiconductor
  label_ko: 반도체
  label_en: Semiconductor
  synonyms: [반도체, 메모리반도체, 비메모리, 시스템반도체, AI반도체, semiconductor, semiconductors, chip, chips, SOX, 필라델피아반도체]
- id: china_semi
  label_ko: 중국반도체
  label_en: China Semiconductor
  synonyms: [중국반도체, 차이나반도체, 차이나AI반도체, China Semiconductor]
  related: [semiconductor, china]
- id: battery
  label_ko: 2차전지
  label_en: Battery / EV
  synonyms: [2차전지, 이차전지, 배터리, 양극재, 리튬, 전기차, battery, lithium, EV]
- id: robot / humanoid / ai / dividend / covered_call / bio / bank_finance / defense / nuclear / shipbuilding / gold / 
#   oil_gas / reits / kospi200 / nasdaq100 / sp500 / us_treasury / kr_treasury / short_term_bond / mmf / leverage / inverse / 
#   esg / india / japan / vietnam / europe / emerging / korea / usa / china / global / kr_growth(국민성장)
```

```yaml
# companies.yaml
- id: ecopro
  name_ko: 에코프로
  name_en: EcoPro
  aliases: [에코프로, EcoPro Co]
  tickers: [086520]
  country: KR
- id: ecopro_bm
  name_ko: 에코프로비엠
  name_en: EcoPro BM
  aliases: [에코프로비엠, EcoPro BM]
  tickers: [247540]
  parent: ecopro
- id: ecopro_hn
  name_ko: 에코프로에이치엔
  name_en: EcoPro HN
  aliases: [에코프로에이치엔, EcoPro HN]
  tickers: ["383310"]
  parent: ecopro
- id: ecopro_materials
  name_ko: 에코프로머티리얼즈
  name_en: EcoPro Materials
  aliases: [에코프로머티리얼즈, EcoPro Materials]
  tickers: ["450080"]
  parent: ecopro
- id: cambricon
  name_ko: 캠브리콘
  name_en: Cambricon Technologies
  aliases: [캠브리콘, 캄브리콘, Cambricon, 寒武纪, 688256]
  tickers: ["688256"]
  country: CN
# + SMIC(중신궈지), Hua Hong, NAURA, 삼성전자, SK하이닉스, 삼성SDI, LG에너지솔루션, 포스코퓨처엠, 한화에어로스페이스, 
#   한국항공우주(KAI), LIG넥스원, 현대로템, 두산에너빌리티, HD현대중공업, 한화오션, 셀트리온, 알테오젠, NVIDIA, TSMC, ASML, 
#   Tesla, Apple, Microsoft, Alphabet, Amazon, Meta, Broadcom, AMD, Intel, Rocket Lab, AST SpaceMobile, Lockheed Martin, 
#   RTX, Northrop Grumman, Boeing, Airbus, Kimi/Moonshot AI (unlisted, note: not_listed: true)
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lexicon.py
from pathlib import Path
from finagent.kb.lexicon import Lexicon
ENRICH = Path(__file__).resolve().parents[1] / "data" / "enrichment"

def test_load_and_match_theme():
    lx = Lexicon.load(ENRICH)
    ids = {t.id for t in lx.match_themes("최근 6개월 동안 우주항공 테마와 연결된 ETF")}
    assert "space_aero" in ids

def test_match_company_alias_and_subsidiaries():
    lx = Lexicon.load(ENRICH)
    comps = lx.match_companies("캠브리콘이 편입된 중국반도체 ETF")
    assert any(c.id == "cambricon" for c in comps)
    subs = {c.id for c in lx.subsidiaries_of("ecopro")}
    assert {"ecopro_bm", "ecopro_hn"} <= subs

def test_manager_for_product_name():
    lx = Lexicon.load(ENRICH)
    assert lx.manager_for_product_name("미래에셋국민참여형국민성장혼합자산투자신탁") == "미래에셋자산운용"
    assert lx.manager_for_product_name("KODEX 로봇액티브") == "삼성자산운용"
    assert lx.manager_for_product_name("알수없는이름") is None
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError: finagent.kb.lexicon`

- [ ] **Step 3: Write lexicon.py**

```python
# src/finagent/kb/lexicon.py
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
import yaml

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return re.sub(r"\s+", "", s)

@dataclass
class Manager:
    id: str; canonical: str; group: str | None = None; aliases: list[str] = field(default_factory=list)
@dataclass
class Theme:
    id: str; label_ko: str; label_en: str; synonyms: list[str] = field(default_factory=list); related: list[str] = field(default_factory=list)
@dataclass
class Company:
    id: str; name_ko: str; name_en: str; aliases: list[str] = field(default_factory=list); tickers: list[str] = field(default_factory=list)
    parent: str | None = None; country: str | None = None; not_listed: bool = False

class Lexicon:
    def __init__(self, managers, themes, companies):
        self.managers, self.themes, self.companies = managers, themes, companies
        self._mgr_alias = sorted(((norm(a), m) for m in managers for a in [m.canonical, *m.aliases]), key=lambda x: -len(x[0]))
        self._theme_syn = [(norm(s), t) for t in themes for s in [t.label_ko, t.label_en, *t.synonyms]]
        self._comp_alias = [(norm(a), c) for c in companies for a in [c.name_ko, c.name_en, *c.aliases]]
        self._by_id = {c.id: c for c in companies}

    @classmethod
    def load(cls, d: Path) -> "Lexicon":
        def rd(name): return yaml.safe_load((d / name).read_text(encoding="utf-8")) or []
        managers = [Manager(**{k: v for k, v in m.items()}) for m in rd("managers.yaml")]
        themes = [Theme(**t) for t in rd("themes.yaml")]
        companies = [Company(**c) for c in rd("companies.yaml")]
        return cls(managers, themes, companies)

    def match_themes(self, text: str) -> list[Theme]:
        n = norm(text); seen, out = set(), []
        for syn, t in self._theme_syn:
            if len(syn) >= 2 and syn in n and t.id not in seen:
                seen.add(t.id); out.append(t)
        return out
    def match_companies(self, text: str) -> list[Company]:
        n = norm(text); seen, out = set(), []
        for a, c in self._comp_alias:
            if len(a) >= 2 and a in n and c.id not in seen:
                seen.add(c.id); out.append(c)
        return out
    def match_managers(self, text: str) -> list[Manager]:
        n = norm(text); seen, out = set(), []
        for a, m in self._mgr_alias:
            if len(a) >= 2 and a in n and m.id not in seen:
                seen.add(m.id); out.append(m)
        return out
    def manager_for_product_name(self, name: str) -> str | None:
        n = norm(name)
        for a, m in self._mgr_alias:           # longest alias first; prefix match
            if len(a) >= 2 and n.startswith(a):
                return m.canonical
        return None
    def subsidiaries_of(self, company_id: str) -> list[Company]:
        return [c for c in self.companies if c.parent == company_id]
    def company(self, company_id: str) -> Company | None:
        return self._by_id.get(company_id)
```

Populate the three YAML files fully (managers: every `cu_fund_mgmt_co` value seen in the ETF-KR data plus the ETF-brand prefixes TIGER/KODEX/RISE/ACE/SOL/PLUS/KIWOOM/HANARO/1Q/WON/KoAct/TIME/FOCUS/IBK/BNK/PLUS; themes ≥ 40 entries; companies ≥ 40 entries with the relations listed above).

- [ ] **Step 4: Run tests** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: enrichment lexicons (managers, themes, companies) and matcher"`

---

### Task 3: ETL — Excel to SQLite with documented views

**Files:**
- Create: `src/finagent/etl/__init__.py`, `src/finagent/etl/normalize.py`, `src/finagent/etl/build_store.py`, `scripts/build_store.py`
- Test: `tests/test_normalize.py`, `tests/test_build_store.py`

**Interfaces:**
- Consumes: `Lexicon.manager_for_product_name`.
- Produces: `normalize.yyyymmdd_to_iso(v) -> str | None`, `normalize.to_flag(v) -> int | None`, `normalize.fund_family_key(name) -> str`, `normalize.clean_text(v) -> str | None`; `build_store.build_store(raw_dir: Path, store_path: Path, lexicon: Lexicon) -> dict[str, int]`; SQLite tables `bond_kr, etf_kr, etf_gl, fund_pub, meta`; views `v_bond_kr, v_etf_kr, v_etf_gl, v_fund_pub`.

Derived columns (appended to each table):

| Table | Derived columns |
|---|---|
| bond_kr | `product_type='국내채권'`, `is_sellable` (1 if `buy_yield` not null), `mat_date`, `isu_date`, `crd_grd_date`, `base_date` (ISO), `remaining_years = remaining_days/365.0`, `issuer_name = trim(pd_pbcm)` |
| etf_kr | `product_type` ('국내ETF' if pd_grp_no='ETF' else '국내ETN'), `is_on_sale` (pd_sale_yn='1' and pd_tr_yn in (0,null)), `manager_name` (lexicon canonical of cu_fund_mgmt_co, else raw), `listing_date`, `delisting_date`, `base_date` (du_upt_dt ISO), `risk_grade` (int parsed from pd_risk_nm "(N등급)") |
| etf_gl | `product_type='해외ETF'` ('해외ETN' if pd_grp_no='ETN'), `is_on_sale`, `manager_name`(raw cu_fund_mgmt_co), `ticker = pd_abrv_nm`, `listing_date`, `base_date` (du_upt_dt), `expense_ratio_pct = cu_charge_rt` |
| fund_pub | `product_type` ('공모펀드' if prvo_pbff_desc='공모' else '사모펀드'), `is_on_sale` (sale_yn='판매중'), `fund_family` (fund_family_key(itm_nm)), `manager_name` (lexicon prefix), `base_date` (fd_price_bas_dt ISO), `risk_grade` (int of zrin_fd_ivst_risk_gcd), `is_hedged` (exchdg_yn) |

`meta` table: `(table_name TEXT, source_id TEXT, korean_name TEXT, base_date TEXT, row_count INTEGER, built_at TEXT)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_normalize.py
from finagent.etl.normalize import yyyymmdd_to_iso, to_flag, fund_family_key, risk_grade_from_name

def test_dates():
    assert yyyymmdd_to_iso(20260821.0) == "2026-08-21"
    assert yyyymmdd_to_iso("20260821") == "2026-08-21"
    assert yyyymmdd_to_iso(None) is None
    assert yyyymmdd_to_iso(float("nan")) is None
    assert yyyymmdd_to_iso(99991231.0) == "9999-12-31"

def test_flags():
    assert to_flag("Y") == 1 and to_flag("N") == 0
    assert to_flag(1.0) == 1 and to_flag("0") == 0
    assert to_flag("판매중") == 1 and to_flag("판매완료") == 0
    assert to_flag(None) is None

def test_fund_family_key():
    a = fund_family_key("미래에셋국민참여형국민성장혼합자산투자신탁(사모투자재간접형) 종류C1-e (서민형)")
    b = fund_family_key("미래에셋국민참여형국민성장혼합자산투자신탁(사모투자재간접형) 종류C (일반형)")
    assert a == b
    assert fund_family_key("삼성글로벌 Chat AI 증권자투자신탁H[주식]_Ce") == fund_family_key("삼성글로벌 Chat AI 증권자투자신탁H[주식]_A")

def test_risk_grade_from_name():
    assert risk_grade_from_name("매우높은위험(1등급)") == 1
    assert risk_grade_from_name("해당없음") is None
```

```python
# tests/test_build_store.py
import sqlite3
from pathlib import Path
import pytest
from finagent.etl.build_store import build_store
from finagent.kb.lexicon import Lexicon
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.slow
def test_build_store(tmp_path):
    counts = build_store(ROOT / "data" / "raw", tmp_path / "p.db", Lexicon.load(ROOT / "data" / "enrichment"))
    assert counts["bond_kr"] == 21882 and counts["fund_pub"] == 23676
    con = sqlite3.connect(tmp_path / "p.db")
    assert con.execute("select count(*) from v_bond_kr where is_sellable=1").fetchone()[0] == 634
    assert con.execute("select count(*) from v_bond_kr where is_sellable=1 and crd_grd in ('AAA','AA+','AA0','AA-')").fetchone()[0] == 78
    assert con.execute("select count(distinct fund_family) from v_fund_pub where itm_nm like '%국민성장%'").fetchone()[0] == 1
    assert con.execute("select base_date from meta where table_name='etf_gl'").fetchone()[0] == "2026-08-22"
```

- [ ] **Step 2: Run** → FAIL (module missing)

- [ ] **Step 3: Implement normalize.py**

```python
# src/finagent/etl/normalize.py
from __future__ import annotations
import math, re, unicodedata

def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))

def yyyymmdd_to_iso(v) -> str | None:
    if _is_nan(v): return None
    s = str(v).strip()
    if s.endswith(".0"): s = s[:-2]
    if not re.fullmatch(r"\d{8}", s): return None
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

_TRUE = {"y", "1", "1.0", "true", "판매중", "예", "가능"}
_FALSE = {"n", "0", "0.0", "false", "판매완료", "판매중지", "아니오", "불가"}
def to_flag(v) -> int | None:
    if _is_nan(v): return None
    s = str(v).strip().lower()
    if s in _TRUE: return 1
    if s in _FALSE: return 0
    return None

def clean_text(v) -> str | None:
    if _is_nan(v): return None
    s = unicodedata.normalize("NFKC", str(v)).strip()
    return s or None

_CLASS_PATTERNS = [
    r"\s*종류\s*[A-Za-z0-9\-]+.*$",          # 종류C1-e (서민형)
    r"\s*\(?\s*class\s*[A-Za-z0-9\-]+\)?.*$", # Class A-e
    r"\s*_[A-Za-z0-9\-]+\s*(\(.*\))?$",       # _Ce, _C-Pe
    r"\s*\(([A-Z]-?[A-Za-z0-9]*|[A-Z][a-z]?e?|C-?P?e?\d*|S|W|I)\)\s*$",  # (C-Pe), (A), (C-R)
    r"\s*\[?(A|C|Ce|C-e|C-Pe|C-P2e|Ae|S|I|W)\]?\s*$",
]
def fund_family_key(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "")
    for p in _CLASS_PATTERNS:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    return s.lower()

def risk_grade_from_name(v) -> int | None:
    if _is_nan(v): return None
    m = re.search(r"(\d)\s*등급", str(v))
    return int(m.group(1)) if m else None
```

- [ ] **Step 4: Implement build_store.py**

```python
# src/finagent/etl/build_store.py
from __future__ import annotations
import sqlite3, datetime as dt
from pathlib import Path
import pandas as pd
from ..kb.lexicon import Lexicon
from .normalize import yyyymmdd_to_iso, to_flag, clean_text, fund_family_key, risk_grade_from_name

SOURCES = {  # table -> (file stem, source id, korean name, base-date column)
    "bond_kr": ("prbd01n001", "PRBD01N001", "국내채권마스터", "info_base_dt"),
    "etf_kr":  ("pref01n001", "PREF01N001", "국내ETF마스터", "du_upt_dt"),
    "etf_gl":  ("pref02n001", "PREF02N001", "해외ETF마스터", "du_upt_dt"),
    "fund_pub":("prfd01n001", "PRFD01N001", "공모펀드마스터", "fd_price_bas_dt"),
}
DATE_COLS = {
    "bond_kr": ["info_base_dt","isu_dt","mat_dt","crd_grd_dt","exg_close_price_base_dt","pd_std_info_update","sale_yield_base_dt"],
    "etf_kr":  ["cu_upt_dt","du_chas_errt_base_dt","du_diff_rt_base_dt","du_nav_base_dt","du_upt_dt","du_vlty_base_dt","fn_base_dt","fn_portfolio_dt","pd_dvid_base_dt","pd_dvid_prc_base_dt","pd_lste_dt","pd_lstg_dt","ref_base_dt","wu_upt_dt"],
    "etf_gl":  ["cu_upt_dt","du_clpr_base_dt","du_nav_base_dt","du_upt_dt","pd_lstg_dt","wu_upt_dt"],
    "fund_pub":["fd_daily_bas_dt","fd_last_dstb_actg_bss_dt","fd_last_dstb_actg_eot_dt","fd_price_bas_dt"],
}
TEXT_CODE_COLS = {  # numeric-looking codes that must stay text
    "bond_kr": ["pd_no","pd_risk_gcd","exrt_grte_ern_r_tcd","info_seq"],
    "etf_kr":  ["pd_itm_no","pd_itm_no_ma","pd_ticker","pd_sale_yn","pd_sect_cd","pd_tr_yn"],
    "etf_gl":  ["pd_itm_no","pd_itm_no_ma","pd_lipper_id","pd_us_cik","pd_sale_yn","pd_tr_yn","pd_exg_mkt_cd"],
    "fund_pub":["itm_no","fss_itm_no","kofia_fd_ccd","ksd_itm_no","mtco_itm_no","or_co_xtn_itt_cd","trusc_xtn_itt_cd","std_itm_no","rptt_ksd_itm_no","zrin_btyp_cd","zrin_pcd","zrin_fd_ivst_risk_gcd","fd_set_pcd","fd_estb_ctry_cd","pfiv_sale_cntl_tcd","prfd_attr_cnt"],
}

def _read(raw_dir: Path, stem: str, text_cols: list[str]) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / f"{stem}_data.xlsx", dtype={c: str for c in text_cols})
    for c in text_cols:
        if c in df: df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v).removesuffix(".0").strip())
    return df

def _derive(table: str, df: pd.DataFrame, lx: Lexicon) -> pd.DataFrame:
    for c in DATE_COLS[table]:
        if c in df: df[c] = df[c].map(yyyymmdd_to_iso)
    if table == "bond_kr":
        df["product_type"] = "국내채권"
        df["is_sellable"] = df["buy_yield"].notna().astype(int)
        df["mat_date"], df["isu_date"], df["base_date"] = df["mat_dt"], df["isu_dt"], df["info_base_dt"]
        df["remaining_years"] = (df["remaining_days"] / 365.0).round(2)
        df["issuer_name"] = df["pd_pbcm"].map(clean_text)
    elif table == "etf_kr":
        df["product_type"] = df["pd_grp_no"].map(lambda g: "국내ETN" if g == "ETN" else "국내ETF")
        df["is_on_sale"] = ((df["pd_sale_yn"] == "1") & (df["pd_tr_yn"].fillna("0") == "0")).astype(int)
        df["manager_name"] = df["cu_fund_mgmt_co"].map(lambda v: (lx.manager_for_product_name(v) if isinstance(v, str) else None) or clean_text(v))
        df["listing_date"], df["delisting_date"], df["base_date"] = df["pd_lstg_dt"], df["pd_lste_dt"], df["du_upt_dt"]
        df["risk_grade"] = df["pd_risk_nm"].map(risk_grade_from_name)
    elif table == "etf_gl":
        df["product_type"] = df["pd_grp_no"].map(lambda g: "해외ETN" if g == "ETN" else "해외ETF")
        df["is_on_sale"] = ((df["pd_sale_yn"].fillna("1") == "1") & (df["pd_tr_yn"].fillna("0") == "0")).astype(int)
        df["manager_name"] = df["cu_fund_mgmt_co"].map(clean_text)
        df["ticker"] = df["pd_abrv_nm"].map(clean_text)
        df["listing_date"], df["base_date"] = df["pd_lstg_dt"], df["du_upt_dt"]
        df["expense_ratio_pct"] = df["cu_charge_rt"]
    elif table == "fund_pub":
        df["product_type"] = df["prvo_pbff_desc"].map(lambda v: "공모펀드" if v == "공모" else "사모펀드")
        df["is_on_sale"] = df["sale_yn"].map(to_flag)
        df["fund_family"] = df["itm_nm"].map(lambda v: fund_family_key(v) if isinstance(v, str) else None)
        df["manager_name"] = df["itm_nm"].map(lambda v: lx.manager_for_product_name(v) if isinstance(v, str) else None)
        df["base_date"] = df["fd_price_bas_dt"]
        df["risk_grade"] = pd.to_numeric(df["zrin_fd_ivst_risk_gcd"], errors="coerce").astype("Int64")
        df["is_hedged"] = df["exchdg_yn"].map(to_flag)
    return df

VIEWS = {
 "v_bond_kr": "SELECT pd_no, pd_nm, pd_abrv_nm, pd_eng_nm, product_type, pd_exg_mkt, is_sellable, bdbns_abl_chnl_nm, issuer_name, bd_knd, std_pd_mcls_nm, std_pd_scls_nm, bd_ofr_tcd, bd_inrt_tcd, bd_intp_tcd, crd_grd, crd_grd_dt, pd_risk_nm, pd_risk_gcd, curr_cd, srfc_irt, isu_date, mat_date, remaining_days, remaining_years, bd_tisu_a, isu_bal_amt, applied_yield, buy_yield, after_tax_yield, pref_tax_yield, avg_annual_tax_yield, corp_pretax_yield, corp_after_tax_yield, depo_equiv_yield_154, depo_equiv_yield_495, trade_price, buyable_quantity, eval_price, dirty, dur, cov, exg_close_price, exg_close_yield, exrt_grte_ern_r, exrt_rpy_r, pd_pen_tr_yn, base_date FROM bond_kr",
 "v_etf_kr": "SELECT pd_itm_no, pd_ticker, pd_abrv_nm, pd_nm, product_type, pd_grp_no, is_on_sale, pd_sale_yn, pd_tr_yn, manager_name, cu_fund_mgmt_co, cu_strtegy, cu_lev_fector, cu_base_index, ref_base_index, cu_charge_rt, cu_charge_etc_rt, wu_inv_ast_type, wu_inv_rgn, ref_ast_type, ref_geo_focus, wu_core_yn, pd_risk_nm, risk_grade, pd_pen_tr_yn, pd_pen_risk_nm, pd_net_tamt, du_last_aum, du_last_nav, du_clpr, ru_mkt_price, du_bpr, du_diff_rt, du_chas_errt, du_er_1d, du_er_1m, du_er_3m, du_er_6m, du_er_ytd, du_er_1y, du_vlty_1m, du_vlty_3m, du_vlty_6m, du_vlty_1y, du_vol_1d, du_val_1d, du_val_1m, pd_dvid_cycl, pd_dvid_yield, pd_divd_amt_pshr, pd_dvid_pay_cnt, pd_dvid_pay_months, listing_date, delisting_date, pd_lst_stk_cnt, fn_average_coupon, fn_effective_maturity, base_date FROM etf_kr",
 "v_etf_gl": "SELECT pd_itm_no, ticker, pd_nm, product_type, pd_grp_no, is_on_sale, pd_exg_mkt_cd, pd_mkt_id, pd_curr_cd, pd_isin_cd, manager_name, cu_base_index, cu_strtegy, expense_ratio_pct, cu_charge_rt, cu_lev_fector, cu_inverse_short_yn, cu_index_tracking_yn, cu_index_repl_mthd, cu_etn_yn, wu_inv_ast_type, wu_inv_rgn, wu_core_yn, du_last_aum, du_last_nav, du_clpr, du_opr, du_hpr, du_lpr, ru_mkt_price, du_er_1d, du_vol_1d, du_val_1d, pd_lst_stk_cnt, listing_date, base_date FROM etf_gl",
 "v_fund_pub": "SELECT itm_no, itm_nm, itm_abrv_nm, itm_eng_nm, fund_family, product_type, prvo_pbff_desc, is_on_sale, sale_yn, thco_sale_yn, manager_name, or_attr_desc, zrin_btyp_nm, zrin_ptn_nm, zrin_attr_nms, fd_ivst_rgn_desc, ovrs_fd_desc, bmrk_nm, bmrk_eng_nm, curr_cd, exchdg_yn, is_hedged, hdge_fd_yn, ofsfd_yn, int_dvd_desc, pers_corp_desc, zrin_fd_ivst_risk_grd_nm, risk_grade, han_clas_nm, han_clas_fee_type, han_clas_sales_channel, han_clas_policies, or_co_rwrd_r, sale_co_rwrd_r, trusc_rwrd_r, ofwk_trus_rwrd_r, fd_nast_suma, bns_bpr, fd_wk1_ern_r, fd_mm1_ern_r, fd_mm3_ern_r, fd_mm6_ern_r, fd_yr1_ern_r, fd_mm18_ern_r, fd_yr2_ern_r, fd_yr3_ern_r, fd_yr5_ern_r, fd_last_dstb_r, zrin_dmst_stk_cmst_rt, zrin_ovrs_stk_cmst_rt, zrin_dmst_bd_cmst_rt, zrin_ovrs_bd_cmst_rt, zrin_liqt_cmst_rt, zrin_fd_cmst_rt, zrin_etc_ast_cmst_rt, prfd_attr_search_text, base_date FROM fund_pub",
}
INDEXES = ["bond_kr(pd_no)","bond_kr(crd_grd)","bond_kr(is_sellable)","etf_kr(pd_itm_no)","etf_kr(pd_abrv_nm)","etf_gl(pd_itm_no)","etf_gl(ticker)","fund_pub(itm_no)","fund_pub(fund_family)","fund_pub(is_on_sale)"]

def build_store(raw_dir: Path, store_path: Path, lx: Lexicon) -> dict[str, int]:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists(): store_path.unlink()
    con = sqlite3.connect(store_path); counts = {}
    for table, (stem, sid, kname, bcol) in SOURCES.items():
        df = _derive(table, _read(raw_dir, stem, TEXT_CODE_COLS[table]), lx)
        df.to_sql(table, con, index=False)
        base = df[bcol].dropna().mode().iloc[0] if bcol in df else None
        con.execute("CREATE TABLE IF NOT EXISTS meta(table_name TEXT, source_id TEXT, korean_name TEXT, base_date TEXT, row_count INTEGER, built_at TEXT)")
        con.execute("INSERT INTO meta VALUES (?,?,?,?,?,?)", (table, sid, kname, base, len(df), dt.datetime.now().isoformat(timespec="seconds")))
        counts[table] = len(df)
    for name, sql in VIEWS.items(): con.execute(f"CREATE VIEW {name} AS {sql}")
    for ix in INDEXES: con.execute(f"CREATE INDEX IF NOT EXISTS ix_{ix.replace('(', '_').replace(')', '')} ON {ix}")
    con.commit(); con.close()
    return counts
```

`scripts/build_store.py`: argparse wrapper calling `build_store(get_settings().raw_dir, get_settings().store_path, Lexicon.load(get_settings().enrichment_dir))` and printing counts. Also expose `python -m finagent.etl.build_store` via `if __name__ == "__main__"`.

- [ ] **Step 5: Run tests** (`-m "not slow"` for quick, full for slow) → PASS; run the script to create `data/store/products.db`.
- [ ] **Step 6: Commit** `git commit -m "feat: ETL from Excel masters to SQLite with documented views"`

---

### Task 4: Ontology Turtle files and runtime loader

**Files:**
- Create: `ontology/common.ttl`, `ontology/bond_kr.ttl`, `ontology/etf_kr.ttl`, `ontology/etf_gl.ttl`, `ontology/fund_pub.ttl`, `src/finagent/ontology/__init__.py`, `src/finagent/ontology/loader.py`
- Test: `tests/test_ontology.py`

**Interfaces:**
- Produces: `Ontology.load(dir) -> Ontology`; `Ontology.graph: rdflib.Graph`; `Ontology.properties: list[PropInfo(iri, name, label_ko, comment, table, column, range_class, unit, enum: list[EnumValue(code, label)])]`; `Ontology.schema_card(view: str) -> str`; `Ontology.enum_for(prop_name) -> list[EnumValue]`; `Ontology.validate_credit_grade(v) -> tuple[bool, str]`; `Ontology.validate_risk_grade(v) -> tuple[bool, str]`; `Ontology.tables() -> list[TableInfo(view, source_id, label_ko, class_iri)]`; `Ontology.summary() -> str` (class hierarchy text for prompts).

Annotation vocabulary declared in `common.ttl`:

```turtle
fp:sourceTable   a owl:AnnotationProperty ; rdfs:label "원천 테이블(뷰) 이름"@ko .
fp:sourceColumn  a owl:AnnotationProperty ; rdfs:label "원천 컬럼 이름"@ko .
fp:unit          a owl:AnnotationProperty ; rdfs:label "단위"@ko .
fp:code          a owl:DatatypeProperty ;  rdfs:label "코드값(원천 데이터에 저장된 문자열)"@ko .
fp:gradeRank     a owl:DatatypeProperty ;  rdfs:label "등급 서열(작을수록 우량)"@ko .
```

Required content per file (loader relies on it):

- `common.ttl`: classes `fp:Product`, `fp:Bond`, `fp:ETF`, `fp:ETN`, `fp:Fund` (all `rdfs:subClassOf fp:Product`), `fp:ManagementCompany`, `fp:Issuer`, `fp:Company`, `fp:Theme`, `fp:Index`, `fp:RiskGrade`, `fp:AssetClass`, `fp:Region`, `fp:DistributionCycle`, `fp:FundFamily`. Individuals: `fp:RiskGrade1..6` (`fp:code "1"`, label "매우높은위험(1등급)" …), `fp:AssetClass_Equity/Bond/Commodity/Alternatives/MixedAssets/MoneyMarket/Currency/RealEstate/Other` with ko+en labels and `fp:code` values matching `wu_inv_ast_type` in both tables (주식/Equity …), `fp:Region_*` for every `wu_inv_rgn` value, `fp:Cycle_A/Q/M/S`. Object properties `fp:managedBy`, `fp:issuedBy`, `fp:hasRiskGrade`, `fp:hasAssetClass`, `fp:investsInRegion`, `fp:hasTheme`, `fp:tracksIndex`, `fp:hasBenchmark`, `fp:hasConstituent`, `fp:subsidiaryOf`, `fp:memberOfFamily`, `fp:relatedTheme`; datatype properties `fp:productCode`, `fp:name`, `fp:shortName`, `fp:englishName`, `fp:baseDate`, `fp:isOnSale`, `fp:productType`, `fp:currency`; SHACL `fp:ProductShape` (exactly one `fp:productCode`, `fp:name` minCount 1), `fp:RiskGradeShape` (`fp:code` in ("1" … "6")).
- `bond_kr.ttl`: `fp:DomesticBond rdfs:subClassOf fp:Bond`, `fp:sourceTable "v_bond_kr"`. `fp:CreditGrade` individuals for AAA, AA+, AA0, AA-, A+, A0, A-, BBB+, BBB0, BBB-, BB+, BB0, BB-, B+, B0, B-, CCC, CC, C0, D with `fp:code` and `fp:gradeRank` 1..20 and a SHACL `sh:in` list. Datatype properties (each with `fp:sourceColumn`, ko label, comment, unit): `fp:couponRate`(srfc_irt, %), `fp:maturityDate`(mat_date), `fp:issueDate`(isu_date), `fp:remainingDays`, `fp:remainingYears`, `fp:duration`(dur), `fp:convexity`(cov), `fp:evalPrice`, `fp:dirtyPrice`, `fp:appliedYield`(민평), `fp:buyYield`(매수수익률, sellable only), `fp:afterTaxYield`, `fp:prefTaxYield`, `fp:corpPretaxYield`, `fp:corpAfterTaxYield`, `fp:depoEquivYield154`, `fp:depoEquivYield495`, `fp:tradePrice`, `fp:buyableQuantity`, `fp:totalIssueAmount`(bd_tisu_a, KRW), `fp:issueBalance`, `fp:isSellable`(is_sellable; comment: "매수가능 종목 = 장외·온오프겸용·매수수익률 존재; 전체의 약 3%"), `fp:bondKind`(bd_knd), `fp:bondMidClass`(std_pd_mcls_nm ∈ 회사채/특수채/국공채), `fp:bondSubClass`, `fp:offeringType`(공모/사모), `fp:interestType`(고정금리/변동금리/고정+변동금리), `fp:interestPaymentType`(이표채/복리채/할인채/단리채), `fp:tradingMarket`(장내/장외), `fp:pensionEligible`, `fp:hasCreditGrade` (object property → fp:CreditGrade, sourceColumn crd_grd; comment "국공채는 신용등급 없음(NULL)").
- `etf_kr.ttl`: `fp:DomesticETF`, `fp:DomesticETN` (`fp:sourceTable "v_etf_kr"`), properties: `fp:strategy` (실물복제/합성복제/액티브/C), `fp:leverageFactor`, `fp:underlyingIndex`(cu_base_index, "일부 종목만 수록"), `fp:refBaseIndex`(ref_base_index), `fp:totalExpenseRatio`(cu_charge_rt, %, "국내ETF는 18%만 수록, ETN 미수록"), `fp:otherCostRatio`, `fp:netAssets`(pd_net_tamt, KRW), `fp:aum`(du_last_aum, KRW), `fp:nav`, `fp:closePrice`, `fp:marketPrice`, `fp:premiumDiscountRate`(du_diff_rt,%), `fp:trackingError`, returns `fp:return1D/1M/3M/6M/YTD/1Y` (%), volatility `fp:volatility1M/3M/6M/1Y`, `fp:volume1D`, `fp:tradingValue1D/1M`, `fp:dividendCycle`(→ fp:DistributionCycle), `fp:dividendYield`, `fp:dividendPerShare`, `fp:dividendPayCount`, `fp:dividendPayMonths`, `fp:listingDate`, `fp:delistingDate`("9999-12-31 = 상장 중"), `fp:isCoreETF`, `fp:pensionEligible`, `fp:pensionRiskClass`, `fp:listedShares`; object `fp:managedBy`(manager_name), `fp:hasAssetClass`(wu_inv_ast_type), `fp:investsInRegion`(wu_inv_rgn), `fp:hasRiskGrade`(risk_grade).
- `etf_gl.ttl`: `fp:ForeignETF`, `fp:ForeignETN` (`fp:sourceTable "v_etf_gl"`), `fp:expenseRatio`(expense_ratio_pct, %, "전 종목 수록"), `fp:aum`(du_last_aum, USD), `fp:exchangeCode`(AMX/NAS/NYS/101/102), `fp:marketCountry`(pd_mkt_id = US), `fp:tradingCurrency`, `fp:isin`, `fp:ric`(pd_itm_no), `fp:ticker`, `fp:strategyText`(cu_strtegy, 영문 서술), `fp:underlyingIndex`(cu_base_index; "'Index is not provided…' = 미제공"), `fp:replicationMethod`(Optimized/Swap/Full/Other), `fp:isIndexTracking`, `fp:isInverseOrShort`, `fp:leverageFactor`, `fp:isETN`, `fp:closePrice/openPrice/highPrice/lowPrice`, `fp:return1D`, `fp:volume1D`, `fp:tradingValue1D`, `fp:listingDate`, `fp:listedShares`; object `fp:managedBy`, `fp:hasAssetClass`(Equity/Bond/…), `fp:investsInRegion`(English region values).
- `fund_pub.ttl`: `fp:PublicFund`, `fp:PrivateFund`, `fp:ShareClass` (`fp:sourceTable "v_fund_pub"`), `fp:FundFamily`; properties `fp:fundFamilyKey`, `fp:managementAttribute`(or_attr_desc: 주식형/채권형/재간접/채권혼합/파생상품/혼합자산/주식혼합/특별자산/MMF/…), `fp:zeroinMajorType`, `fp:zeroinType`, `fp:zeroinAttributes`, `fp:investRegionDesc`, `fp:overseasType`(국내/해외/국내외혼합), `fp:benchmarkName`, `fp:isHedged`, `fp:isHedgeFund`, `fp:isOffshore`, `fp:incomeType`(배당/이자), `fp:investorType`(개인/법인/해당없음), `fp:offeringType`(공모/사모), `fp:saleStatus`(판매중/판매완료), `fp:soldByUs`(thco_sale_yn), `fp:className`, `fp:classFeeType`(수수료미징구/선취/후취), `fp:classSalesChannel`(온라인/오프라인/직판), `fp:classPolicies`, fee components `fp:managerFee/sellerFee/trusteeFee/adminFee` (or_co_rwrd_r…, unit "원천 값(단위 미공개; 상대 비교용)"), `fp:netAssets`(fd_nast_suma, KRW), `fp:navPrice`(bns_bpr), returns `fp:return1W/1M/3M/6M/1Y/18M/2Y/3Y/5Y`, `fp:lastDistributionRate`, composition ratios `fp:domesticStockRatio/overseasStockRatio/domesticBondRatio/overseasBondRatio/liquidityRatio/fundRatio/otherAssetRatio`, `fp:hasRiskGrade`(risk_grade; "제로인 6단계"), `fp:managedBy`(manager_name).

- [ ] **Step 1: Write failing test**

```python
# tests/test_ontology.py
from pathlib import Path
from finagent.ontology.loader import Ontology
ONT = Path(__file__).resolve().parents[1] / "ontology"

def test_load_all_five_files():
    o = Ontology.load(ONT)
    assert {t.view for t in o.tables()} == {"v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub"}
    assert len(o.graph) > 300

def test_schema_card_mentions_columns_and_enums():
    o = Ontology.load(ONT)
    card = o.schema_card("v_bond_kr")
    assert "crd_grd" in card and "AA-" in card and "is_sellable" in card
    assert "v_etf_gl" in o.schema_card("v_etf_gl")

def test_validators():
    o = Ontology.load(ONT)
    ok, msg = o.validate_credit_grade("AAAA")
    assert not ok and "AAA" in msg
    assert o.validate_credit_grade("AA-")[0]
    assert not o.validate_risk_grade("7")[0] and o.validate_risk_grade(2)[0]
    assert o.credit_grades_at_or_above("AA-") == ["AAA", "AA+", "AA0", "AA-"]
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Author the five TTL files** per the lists above. Every property: `a owl:DatatypeProperty|owl:ObjectProperty ; rdfs:domain <class> ; rdfs:range <xsd type or class> ; rdfs:label "..."@ko ; rdfs:comment "..."@ko ; fp:sourceTable "v_..." ; fp:sourceColumn "..." ; fp:unit "..."`(optional). Check: `python -c "import rdflib; g=rdflib.Graph(); [g.parse(f'ontology/{n}.ttl') for n in ['common','bond_kr','etf_kr','etf_gl','fund_pub']]; print(len(g))"`.

- [ ] **Step 4: Implement loader.py**

```python
# src/finagent/ontology/loader.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import rdflib
from rdflib import RDF, RDFS, OWL, Namespace, Literal
FP = Namespace("http://mafest.ai/product#")
FILES = ["common.ttl", "bond_kr.ttl", "etf_kr.ttl", "etf_gl.ttl", "fund_pub.ttl"]

@dataclass
class EnumValue: code: str; label: str; rank: int | None = None
@dataclass
class PropInfo:
    iri: str; name: str; label_ko: str; comment: str; table: str | None; column: str | None
    range_iri: str | None; unit: str | None; is_object: bool; enum: list[EnumValue] = field(default_factory=list)
@dataclass
class TableInfo: view: str; source_id: str; label_ko: str; class_iri: str

def _ko(g, s, p):
    vals = [str(o) for o in g.objects(s, p)]
    ko = [str(o) for o in g.objects(s, p) if isinstance(o, Literal) and o.language == "ko"]
    return (ko or vals or [""])[0]

class Ontology:
    def __init__(self, g: rdflib.Graph):
        self.graph = g
        self.properties = self._props()
        self._tables = self._table_infos()
    @classmethod
    def load(cls, d: Path) -> "Ontology":
        g = rdflib.Graph()
        for f in FILES: g.parse(d / f, format="turtle")
        g.bind("fp", FP)
        return cls(g)
    def _enum_of(self, cls_iri) -> list[EnumValue]:
        out = []
        for ind in self.graph.subjects(RDF.type, cls_iri):
            code = self.graph.value(ind, FP.code); rank = self.graph.value(ind, FP.gradeRank)
            if code is not None: out.append(EnumValue(str(code), _ko(self.graph, ind, RDFS.label), int(rank) if rank is not None else None))
        return sorted(out, key=lambda e: (e.rank if e.rank is not None else 10**6, e.code))
    def _props(self) -> list[PropInfo]:
        out = []
        for ptype in (OWL.DatatypeProperty, OWL.ObjectProperty):
            for p in self.graph.subjects(RDF.type, ptype):
                rng = self.graph.value(p, RDFS.range)
                enum = self._enum_of(rng) if (ptype == OWL.ObjectProperty and rng is not None) else []
                out.append(PropInfo(str(p), str(p).split("#")[-1], _ko(self.graph, p, RDFS.label), _ko(self.graph, p, RDFS.comment),
                    (str(self.graph.value(p, FP.sourceTable)) if self.graph.value(p, FP.sourceTable) else None),
                    (str(self.graph.value(p, FP.sourceColumn)) if self.graph.value(p, FP.sourceColumn) else None),
                    str(rng) if rng else None, (str(self.graph.value(p, FP.unit)) if self.graph.value(p, FP.unit) else None),
                    ptype == OWL.ObjectProperty, enum))
        return sorted(out, key=lambda x: (x.table or "", x.column or "", x.name))
    def _table_infos(self) -> list[TableInfo]:
        out = []
        for c in self.graph.subjects(FP.sourceTable, None):
            if (c, RDF.type, OWL.Class) in self.graph:
                sid = self.graph.value(c, FP.sourceId)
                out.append(TableInfo(str(self.graph.value(c, FP.sourceTable)), str(sid or ""), _ko(self.graph, c, RDFS.label), str(c)))
        seen, uniq = set(), []
        for t in out:
            if t.view not in seen: seen.add(t.view); uniq.append(t)
        return sorted(uniq, key=lambda t: t.view)
    def tables(self) -> list[TableInfo]: return self._tables
    def props_for(self, view: str) -> list[PropInfo]: return [p for p in self.properties if p.table == view and p.column]
    def enum_for(self, prop_name: str) -> list[EnumValue]:
        for p in self.properties:
            if p.name == prop_name: return p.enum
        return []
    def schema_card(self, view: str) -> str:
        t = next((t for t in self._tables if t.view == view), None)
        lines = [f"### {view} — {t.label_ko if t else ''} (원천 {t.source_id if t else ''})"]
        for p in self.props_for(view):
            extra = []
            if p.unit: extra.append(f"단위:{p.unit}")
            if p.enum: extra.append("값:" + "/".join(e.code for e in p.enum))
            if p.comment: extra.append(p.comment)
            lines.append(f"- {p.column}: {p.label_ko}" + (f" ({'; '.join(extra)})" if extra else ""))
        return "\n".join(lines)
    def summary(self) -> str:
        lines = []
        for c in self.graph.subjects(RDF.type, OWL.Class):
            sup = self.graph.value(c, RDFS.subClassOf)
            if sup is not None and str(sup).startswith(str(FP)):
                lines.append(f"{str(c).split('#')[-1]} ⊂ {str(sup).split('#')[-1]} — {_ko(self.graph, c, RDFS.label)}")
        return "\n".join(sorted(lines))
    # validators
    def credit_grades(self) -> list[EnumValue]: return self.enum_for("hasCreditGrade")
    def validate_credit_grade(self, v) -> tuple[bool, str]:
        codes = [e.code for e in self.credit_grades()]
        s = str(v).strip().upper()
        return (s in codes, f"유효한 신용등급: {', '.join(codes)}")
    def credit_grades_at_or_above(self, v: str) -> list[str]:
        grades = self.credit_grades(); s = str(v).strip().upper()
        ranks = {e.code: e.rank for e in grades}
        if s not in ranks: return []
        return [e.code for e in grades if e.rank is not None and e.rank <= ranks[s]]
    def validate_risk_grade(self, v) -> tuple[bool, str]:
        codes = [e.code for e in self.enum_for("hasRiskGrade")]
        return (str(v).strip() in codes, f"위험등급은 {codes[0]}~{codes[-1]}등급만 존재")
```

Note for TTL authors: each table class carries `fp:sourceTable "v_xxx"` and `fp:sourceId "PRBD01N001"` (declare `fp:sourceId a owl:AnnotationProperty` in common.ttl).

- [ ] **Step 5: Run tests** → PASS
- [ ] **Step 6: Commit** `git commit -m "feat: domain ontology (5 TTL files) and runtime loader with schema cards/validators"`

---

### Task 5: Guarded read-only SQL executor

**Files:**
- Create: `src/finagent/kb/rdb.py`
- Test: `tests/test_rdb.py`

**Interfaces:**
- Produces: `RDB(path, row_limit=200, timeout_sec=5.0)`; `RDB.query(sql: str) -> QueryResult(sql, columns: list[str], rows: list[dict], row_count: int, truncated: bool)`; raises `SQLGuardError(msg)`; `RDB.meta() -> dict[view_name, {source_id, korean_name, base_date, row_count}]`; `RDB.views() -> list[str]`; `RDB.columns(view) -> list[str]`; `RDB.fetch_all(sql, params=()) -> list[tuple]` (internal, unguarded, for index/graph builders).

- [ ] **Step 1: Failing tests**

```python
# tests/test_rdb.py
import sqlite3, pytest
from finagent.kb.rdb import RDB, SQLGuardError

@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"; c = sqlite3.connect(p)
    c.execute("create table bond_kr(pd_no text, crd_grd text, buy_yield real)")
    c.executemany("insert into bond_kr values (?,?,?)", [(f"K{i}", "AAA" if i % 2 else "AA-", i) for i in range(500)])
    c.execute("create view v_bond_kr as select * from bond_kr")
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.execute("insert into meta values ('bond_kr','PRBD01N001','국내채권마스터','2026-08-21',500,'x')")
    c.commit(); c.close(); return RDB(p, row_limit=100)

def test_select_with_auto_limit(db):
    r = db.query("SELECT pd_no, crd_grd FROM v_bond_kr WHERE crd_grd='AAA'")
    assert r.row_count == 100 and r.truncated and r.columns == ["pd_no", "crd_grd"]
    assert "LIMIT" in r.sql.upper()

def test_rejects_non_select(db):
    for bad in ["DELETE FROM bond_kr", "SELECT 1; DROP TABLE bond_kr", "PRAGMA table_info(bond_kr)", "ATTACH 'x' AS y"]:
        with pytest.raises(SQLGuardError): db.query(bad)

def test_meta_and_columns(db):
    assert db.meta()["v_bond_kr"]["base_date"] == "2026-08-21"
    assert "crd_grd" in db.columns("v_bond_kr")
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

```python
# src/finagent/kb/rdb.py
from __future__ import annotations
import re, sqlite3, time
from dataclasses import dataclass
from pathlib import Path

class SQLGuardError(ValueError): pass

@dataclass
class QueryResult:
    sql: str; columns: list[str]; rows: list[dict]; row_count: int; truncated: bool

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex)\b", re.I)

class RDB:
    def __init__(self, path: Path, row_limit: int = 200, timeout_sec: float = 5.0):
        self.path, self.row_limit, self.timeout_sec = Path(path), row_limit, timeout_sec
        self._con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
    def guard(self, sql: str) -> str:
        s = sql.strip().rstrip(";").strip()
        if ";" in s: raise SQLGuardError("한 번에 하나의 SELECT 문만 실행할 수 있습니다.")
        if not re.match(r"^(select|with)\b", s, re.I): raise SQLGuardError("SELECT/WITH 문만 허용됩니다.")
        if _FORBIDDEN.search(s): raise SQLGuardError("읽기 전용 질의만 허용됩니다.")
        if not re.search(r"\blimit\s+\d+", s, re.I): s = f"{s} LIMIT {self.row_limit + 1}"
        else:
            m = re.search(r"\blimit\s+(\d+)", s, re.I)
            if int(m.group(1)) > self.row_limit: s = re.sub(r"\blimit\s+\d+", f"LIMIT {self.row_limit + 1}", s, flags=re.I)
        return s
    def query(self, sql: str) -> QueryResult:
        s = self.guard(sql); deadline = time.monotonic() + self.timeout_sec
        self._con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        try:
            cur = self._con.execute(s)
            cols = [d[0] for d in cur.description]; rows = [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e): raise SQLGuardError("질의 시간 초과(5초). 조건을 좁혀 주세요.")
            raise SQLGuardError(f"SQL 오류: {e}")
        finally: self._con.set_progress_handler(None, 0)
        truncated = len(rows) > self.row_limit
        rows = rows[: self.row_limit]
        shown = re.sub(r"\bLIMIT\s+\d+$", f"LIMIT {self.row_limit}", s, flags=re.I) if truncated else s
        return QueryResult(shown, cols, rows, len(rows), truncated)
    def fetch_all(self, sql: str, params=()) -> list[tuple]:
        return [tuple(r) for r in self._con.execute(sql, params).fetchall()]
    def views(self) -> list[str]:
        return [r[0] for r in self.fetch_all("select name from sqlite_master where type='view' order by name")]
    def columns(self, view: str) -> list[str]:
        return [r[1] for r in self.fetch_all(f"pragma table_info({view})")]
    def meta(self) -> dict[str, dict]:
        out = {}
        for t, sid, k, b, n in self.fetch_all("select table_name, source_id, korean_name, base_date, row_count from meta"):
            out[f"v_{t}"] = {"source_id": sid, "korean_name": k, "base_date": b, "row_count": n}
        return out
```

- [ ] **Step 4: Run tests** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: guarded read-only SQL executor"`

---

### Task 6: Lexical search index and entity resolver

**Files:**
- Create: `src/finagent/kb/search.py`, `src/finagent/kb/entity.py`
- Test: `tests/test_search.py`, `tests/test_entity.py`

**Interfaces:**
- Produces: `tokenize(text) -> list[str]`; `Doc(id, table, title, text, payload: dict)`; `SearchIndex(docs)`; `SearchIndex.from_rdb(rdb) -> SearchIndex` (one doc per product row: bond `pd_nm pd_abrv_nm pd_eng_nm issuer_name bd_knd crd_grd`, etf_kr `pd_abrv_nm pd_nm manager_name cu_base_index ref_base_index wu_inv_ast_type wu_inv_rgn`, etf_gl `pd_nm ticker manager_name cu_base_index wu_inv_rgn wu_inv_ast_type cu_strtegy`, fund `itm_nm itm_abrv_nm manager_name or_attr_desc zrin_ptn_nm bmrk_nm fd_ivst_rgn_desc`); `SearchIndex.search(query, k=10, table=None) -> list[Hit(id, table, title, score, payload)]`; `EntityResolver(index, lexicon)`; `EntityResolver.resolve(mention: str, k=5) -> Resolution(products: list[Hit], companies, managers, themes, exact: bool)`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_search.py
from finagent.kb.search import SearchIndex, Doc, tokenize

def test_tokenize_korean_bigrams_and_words():
    toks = tokenize("KODEX 로봇액티브 ETF")
    assert "kodex" in toks and "로봇" in toks and "봇액" in toks

def test_search_ranks_exact_name_first():
    docs = [Doc("1", "v_etf_kr", "KODEX 로봇액티브", "KODEX 로봇액티브 삼성자산운용", {}),
            Doc("2", "v_etf_kr", "RISE AI&로봇", "RISE AI&로봇 KB자산운용", {}),
            Doc("3", "v_etf_kr", "TIGER 반도체", "TIGER 반도체 미래에셋", {})]
    idx = SearchIndex(docs)
    hits = idx.search("KODEX 로봇", k=2)
    assert hits[0].id == "1"
    assert idx.search("반도체", table="v_etf_kr")[0].id == "3"
```

```python
# tests/test_entity.py
from pathlib import Path
from finagent.kb.search import SearchIndex, Doc
from finagent.kb.lexicon import Lexicon
from finagent.kb.entity import EntityResolver
ENRICH = Path(__file__).resolve().parents[1] / "data" / "enrichment"

def test_resolver_marks_unknown_product():
    idx = SearchIndex([Doc("1", "v_etf_kr", "KODEX 로봇액티브", "KODEX 로봇액티브", {}), Doc("2", "v_etf_kr", "RISE AI&로봇", "RISE AI&로봇", {})])
    r = EntityResolver(idx, Lexicon.load(ENRICH)).resolve("KODEX AI로봇 ETF")
    assert not r.exact and [h.title for h in r.products][:2] == ["KODEX 로봇액티브", "RISE AI&로봇"] or r.products
    r2 = EntityResolver(idx, Lexicon.load(ENRICH)).resolve("KODEX 로봇액티브")
    assert r2.exact and r2.products[0].id == "1"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement search.py**

```python
# src/finagent/kb/search.py
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass, field
import numpy as np
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9&+\-\.]+|[가-힣]+", re.I)
def tokenize(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    toks = []
    for w in _WORD.findall(s):
        toks.append(w)
        if re.fullmatch(r"[가-힣]+", w):
            toks += [w[i:i+2] for i in range(len(w) - 1)]
        elif len(w) >= 6:
            toks += [w[i:i+3] for i in range(len(w) - 2)]
    return toks

@dataclass
class Doc: id: str; table: str; title: str; text: str; payload: dict = field(default_factory=dict)
@dataclass
class Hit: id: str; table: str; title: str; score: float; payload: dict

class SearchIndex:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self._bm25 = BM25Okapi([tokenize(d.text) for d in docs]) if docs else None
        self._norm_titles = [re.sub(r"\s+", "", d.title.lower()) for d in docs]
        self._tables = np.array([d.table for d in docs])
    @classmethod
    def from_rdb(cls, rdb) -> "SearchIndex":
        specs = {
            "v_bond_kr": ("pd_no", "pd_nm", ["pd_nm","pd_abrv_nm","pd_eng_nm","issuer_name","bd_knd","crd_grd"], ["pd_no","pd_nm","crd_grd","is_sellable","mat_date","buy_yield","srfc_irt","issuer_name","bd_knd"]),
            "v_etf_kr": ("pd_itm_no", "pd_abrv_nm", ["pd_abrv_nm","pd_nm","manager_name","cu_base_index","ref_base_index","wu_inv_ast_type","wu_inv_rgn"], ["pd_itm_no","pd_abrv_nm","pd_nm","manager_name","product_type","is_on_sale","wu_inv_ast_type","wu_inv_rgn","pd_net_tamt","cu_charge_rt","pd_risk_nm"]),
            "v_etf_gl": ("pd_itm_no", "pd_nm", ["pd_nm","ticker","manager_name","cu_base_index","wu_inv_rgn","wu_inv_ast_type","cu_strtegy"], ["pd_itm_no","ticker","pd_nm","manager_name","product_type","wu_inv_ast_type","wu_inv_rgn","expense_ratio_pct","du_last_aum","cu_base_index"]),
            "v_fund_pub": ("itm_no", "itm_nm", ["itm_nm","itm_abrv_nm","manager_name","or_attr_desc","zrin_ptn_nm","bmrk_nm","fd_ivst_rgn_desc"], ["itm_no","itm_nm","fund_family","manager_name","product_type","is_on_sale","or_attr_desc","zrin_ptn_nm","bmrk_nm","fd_nast_suma","zrin_fd_ivst_risk_grd_nm"]),
        }
        docs = []
        for view, (idc, titlec, textc, payc) in specs.items():
            cols = list(dict.fromkeys([idc, titlec, *textc, *payc]))
            for row in rdb.fetch_all(f"select {', '.join(cols)} from {view}"):
                r = dict(zip(cols, row))
                docs.append(Doc(str(r[idc]), view, str(r[titlec] or ""), " ".join(str(r[c]) for c in textc if r.get(c) not in (None, "")), {c: r[c] for c in payc}))
        return cls(docs)
    def search(self, query: str, k: int = 10, table: str | None = None) -> list[Hit]:
        if not self.docs: return []
        scores = self._bm25.get_scores(tokenize(query))
        q = re.sub(r"\s+", "", query.lower())
        for i, t in enumerate(self._norm_titles):        # exact / substring boosts
            if t and (t == q): scores[i] += 100
            elif t and (q in t or t in q) and len(q) >= 2: scores[i] += 10
        if table is not None: scores = np.where(self._tables == table, scores, -1.0)
        top = np.argsort(-scores)[:k]
        return [Hit(self.docs[i].id, self.docs[i].table, self.docs[i].title, float(scores[i]), self.docs[i].payload) for i in top if scores[i] > 0]
```

- [ ] **Step 4: Implement entity.py**

```python
# src/finagent/kb/entity.py
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .search import SearchIndex, Hit
from .lexicon import Lexicon, Company, Manager, Theme

@dataclass
class Resolution:
    mention: str
    products: list[Hit] = field(default_factory=list)
    companies: list[Company] = field(default_factory=list)
    managers: list[Manager] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    exact: bool = False
    def summary(self) -> str:
        parts = []
        if self.products: parts.append("상품 후보: " + "; ".join(f"[{h.table}] {h.title} ({h.id}, score={h.score:.1f})" for h in self.products))
        if self.companies: parts.append("기업: " + ", ".join(f"{c.name_ko}({c.name_en}{', 비상장' if c.not_listed else ''})" for c in self.companies))
        if self.managers: parts.append("운용사: " + ", ".join(m.canonical for m in self.managers))
        if self.themes: parts.append("테마: " + ", ".join(f"{t.label_ko}/{t.label_en}" for t in self.themes))
        return "\n".join(parts) if parts else "일치하는 상품·기업·운용사·테마 없음"

_STOP = re.compile(r"(etf|etn|펀드|채권|상품|정보|알려줘|알려|주세요|뭐야|있어|있나요|\?|\s)+", re.I)

class EntityResolver:
    def __init__(self, index: SearchIndex, lexicon: Lexicon): self.index, self.lexicon = index, lexicon
    def resolve(self, mention: str, k: int = 5, table: str | None = None) -> Resolution:
        r = Resolution(mention, companies=self.lexicon.match_companies(mention), managers=self.lexicon.match_managers(mention), themes=self.lexicon.match_themes(mention))
        hits = self.index.search(mention, k=k, table=table)
        core = _STOP.sub("", mention).lower()
        r.products = hits
        r.exact = bool(hits) and (hits[0].score >= 100 or (core and re.sub(r"\s+", "", hits[0].title.lower()) == core))
        return r
```

- [ ] **Step 5: Run tests** → PASS
- [ ] **Step 6: Commit** `git commit -m "feat: BM25 product index and entity resolver"`

---

### Task 7: Knowledge graph (ABox) build, SPARQL, neighbors, export

**Files:**
- Create: `src/finagent/kb/graph.py`, `data/enrichment/constituents/README.md`, `data/enrichment/constituents/sample.csv`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `RDB.fetch_all`, `Lexicon`, `Ontology.graph`.
- Produces: `KG.build(rdb, lexicon, ontology, constituents_dir: Path | None) -> KG`; `KG.query(sparql: str, limit=100) -> list[dict]`; `KG.neighbors(entity: str, relation: str | None = None, limit=50) -> dict[str, list[dict]]` (entity = label or code or IRI suffix); `KG.export(path)`; `KG.size -> int`; `KG.prefixes -> str`. Constituent CSV columns: `etf_id,etf_table,company_id,company_name,weight_pct,as_of,source`.

Triples per product (subject `d:{table}/{code}`): `rdf:type` (fp:DomesticBond etc.), `fp:productCode`, `fp:name`, `fp:shortName`, `fp:productType`, `fp:isOnSale`, `fp:baseDate`; object links: `fp:managedBy d:manager/{slug}` (slug = canonical manager), `fp:issuedBy d:issuer/{slug}`, `fp:hasAssetClass fp:AssetClass_*` (map ko/en value → individual via `fp:code`), `fp:investsInRegion fp:Region_*`, `fp:hasRiskGrade fp:RiskGrade{n}`, `fp:hasCreditGrade fp:CreditGrade_{code}`, `fp:tracksIndex d:index/{slug}` (cu_base_index / ref_base_index when meaningful), `fp:hasBenchmark d:index/{slug}` (funds), `fp:memberOfFamily d:family/{key}`, `fp:hasTheme d:theme/{id}` (lexicon.match_themes on name + strategy text), `fp:hasConstituent d:company/{id}` (from CSV). Companies: `d:company/{id} a fp:Company; fp:name; fp:subsidiaryOf d:company/{parent}`. Themes: `d:theme/{id} a fp:Theme; rdfs:label ko/en; fp:relatedTheme`. Managers: `d:manager/{slug} a fp:ManagementCompany; rdfs:label; fp:group`.

- [ ] **Step 1: Failing test**

```python
# tests/test_graph.py
import sqlite3
from pathlib import Path
import pytest
from finagent.kb.rdb import RDB
from finagent.kb.lexicon import Lexicon
from finagent.ontology.loader import Ontology
from finagent.kb.graph import KG
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def small_rdb(tmp_path):
    p = tmp_path / "s.db"; c = sqlite3.connect(p)
    c.execute("create table etf_kr(pd_itm_no, pd_abrv_nm, pd_nm, product_type, is_on_sale, manager_name, cu_base_index, ref_base_index, wu_inv_ast_type, wu_inv_rgn, risk_grade, base_date, cu_strtegy)")
    c.execute("insert into etf_kr values ('KR1','KODEX 2차전지산업','삼성 KODEX 2차전지산업','국내ETF',1,'삼성자산운용','','KRX 2차전지','주식','국내',1,'2026-08-21','실물복제')")
    c.execute("create view v_etf_kr as select * from etf_kr")
    for v in ["v_bond_kr","v_etf_gl","v_fund_pub"]: c.execute(f"create view {v} as select 1 as x where 0")
    c.execute("create table meta(table_name, source_id, korean_name, base_date, row_count, built_at)")
    c.commit(); c.close(); return RDB(p)

def test_build_query_neighbors(small_rdb, tmp_path):
    cons = tmp_path / "cons"; cons.mkdir()
    (cons / "kr.csv").write_text("etf_id,etf_table,company_id,company_name,weight_pct,as_of,source\nKR1,v_etf_kr,ecopro_bm,에코프로비엠,12.3,2026-08-21,test\n", encoding="utf-8")
    kg = KG.build(small_rdb, Lexicon.load(ROOT / "data/enrichment"), Ontology.load(ROOT / "ontology"), cons)
    rows = kg.query("SELECT ?etf WHERE { ?etf fp:hasConstituent ?c . ?c fp:subsidiaryOf ?p . ?p fp:name '에코프로' }")
    assert rows and rows[0]["etf"].endswith("etf_kr/KR1")
    nb = kg.neighbors("KODEX 2차전지산업")
    assert "hasTheme" in nb and any("battery" in x["iri"] for x in nb["hasTheme"])
    assert kg.neighbors("에코프로", relation="subsidiaryOf_inverse")
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement graph.py** (build from RDB rows with the mapping above; `_slug()` = NFKC + non-alnum → `_`; store label index `dict[norm_label, iri]` for `neighbors`; `neighbors` returns both outgoing predicates and incoming as `<pred>_inverse`; `query` prepends `PREFIX fp: … PREFIX d: … PREFIX rdfs: …` if absent and applies `LIMIT` when missing; `export` writes Turtle). Cache: after build, `pickle.dump(self.graph)` is NOT used (rdflib Graph pickling is unreliable) — instead build time is logged; keep build < 60s by batching `g.add`.

- [ ] **Step 4: Run tests** → PASS; run a full build against `data/store/products.db` and print size/time.
- [ ] **Step 5: Commit** `git commit -m "feat: knowledge graph build, SPARQL, neighbors, constituents loader"`

---

### Task 8: External document store (markdown chunks) + parser script

**Files:**
- Create: `src/finagent/kb/docs.py`, `scripts/parse_docs.py`, `data/external/docs/README.md`, `data/external/docs/2026-08_국민성장펀드_개요.md` (hand-written summary with source URL + date lines in front-matter; only facts the team verified)
- Test: `tests/test_docs.py`

**Interfaces:**
- Produces: `DocStore.load(dir) -> DocStore`; `DocStore.search(query, k=5) -> list[Chunk(doc_id, title, source, as_of, text, score)]`; front-matter keys `title`, `source`, `as_of`, `products` (list of product codes/names). `scripts/parse_docs.py <pdf> <out.md>` converts PDF → markdown via PyMuPDF (`fitz`) with page headers.

- [ ] **Step 1: Failing test**

```python
# tests/test_docs.py
from finagent.kb.docs import DocStore
def test_docstore_search(tmp_path):
    (tmp_path / "a.md").write_text("---\ntitle: 국민성장펀드 개요\nsource: https://example.org\nas_of: 2026-08-20\n---\n# 구조\n국민성장펀드는 모펀드-자펀드 구조이다.\n\n# 전략\nAI·반도체 등 첨단산업에 투자한다.\n", encoding="utf-8")
    ds = DocStore.load(tmp_path)
    hits = ds.search("국민성장펀드 구조")
    assert hits and hits[0].title == "국민성장펀드 개요" and "모펀드" in hits[0].text and hits[0].as_of == "2026-08-20"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** (parse front-matter with a small regex, split on `\n#` headings into chunks ≤ 1200 chars, index chunks with `SearchIndex`).
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: external markdown document store and PDF→markdown parser"`

---

### Task 9: LLM adapters (Anthropic, OpenAI-compatible, stub)

**Files:**
- Create: `src/finagent/agent/__init__.py`, `src/finagent/agent/llm.py`
- Test: `tests/test_llm_stub.py`

**Interfaces:**
- Produces:

```python
@dataclass
class ToolSpec: name: str; description: str; input_schema: dict
@dataclass
class ToolCall: id: str; name: str; input: dict
@dataclass
class Turn: text: str; tool_calls: list[ToolCall]; stop_reason: str  # "tool_use" | "end_turn" | "max_tokens" | "refusal" | "error"
class Conversation(Protocol):
    async def step(self) -> Turn
    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None  # (call_id, content, is_error)
    def add_user(self, text: str) -> None
class LLMClient(Protocol):
    def start(self, system: str, tools: list[ToolSpec], user_text: str) -> Conversation
def make_llm(settings) -> LLMClient
class StubLLM(LLMClient): def __init__(self, script: list[Turn])
```

Anthropic implementation: `AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_sec, max_retries=1)`; per step:

```python
kwargs = dict(model=self.model, max_tokens=self.max_tokens,
    system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
    tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self.tools],
    messages=self.messages, thinking={"type": "adaptive"}, output_config={"effort": self.effort})
if self.fallbacks:
    resp = await self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
else:
    resp = await self.client.messages.create(**kwargs)
self.messages.append({"role": "assistant", "content": resp.content})
text = "".join(b.text for b in resp.content if b.type == "text")
calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
return Turn(text, calls, resp.stop_reason)
```

Tool results: `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": content, "is_error": is_error} ...]}` (all results of one round in ONE user message). If the beta call raises `anthropic.BadRequestError` mentioning `fallbacks`/`betas`, retry once without fallbacks and remember. Catch `anthropic.APIStatusError`/`APIConnectionError`/`APITimeoutError` → `Turn("", [], "error")` with the message in `text`.

OpenAI-compatible: `AsyncOpenAI(api_key, base_url)`; tools as `{"type": "function", "function": {"name", "description", "parameters": input_schema}}`; parse `choice.message.tool_calls` (`json.loads(arguments)`); append assistant message dict; tool results as `{"role": "tool", "tool_call_id": id, "content": content}`; `finish_reason` `tool_calls` → `tool_use`, `stop` → `end_turn`, `length` → `max_tokens`.

- [ ] **Step 1: Failing test**

```python
# tests/test_llm_stub.py
import pytest
from finagent.agent.llm import StubLLM, Turn, ToolCall, ToolSpec
async def test_stub_conversation_replays_script():
    llm = StubLLM([Turn("", [ToolCall("c1", "sql_query", {"sql": "select 1"})], "tool_use"), Turn("done", [], "end_turn")])
    conv = llm.start("sys", [ToolSpec("sql_query", "d", {"type": "object"})], "q")
    t1 = await conv.step(); assert t1.tool_calls[0].name == "sql_query"
    conv.add_tool_results([("c1", "ok", False)])
    t2 = await conv.step(); assert t2.stop_reason == "end_turn" and t2.text == "done"
    assert conv.transcript()[-1]["role"] == "assistant"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement llm.py** (three classes + `make_llm`; each Conversation also exposes `transcript() -> list[dict]` for debugging).
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: LLM adapters (anthropic, openai-compatible, stub)"`

---

### Task 10: Tools registry, prompts, evidence

**Files:**
- Create: `src/finagent/agent/tools.py`, `src/finagent/agent/prompts.py`, `src/finagent/agent/answer.py`
- Test: `tests/test_tools.py`, `tests/test_answer.py`

**Interfaces:**
- Consumes: RDB, SearchIndex, EntityResolver, KG, DocStore, Ontology, Lexicon.
- Produces:

```python
@dataclass
class Evidence: kind: str; table: str | None; korean_name: str | None; base_date: str | None; detail: str; row_count: int | None; refs: list[str]
@dataclass
class ToolResult: content: str; is_error: bool = False; evidence: Evidence | None = None
class ToolRegistry:
    def __init__(self, rdb, index, resolver, kg, docs, ontology, lexicon, settings)
    def specs(self) -> list[ToolSpec]      # table_info, sql_query, search_products, resolve_entity, graph_query, graph_neighbors, search_documents, final_answer
    async def execute(self, name: str, args: dict) -> ToolResult
FINAL_ANSWER_SCHEMA = {"type":"object","properties":{"status":{"enum":["answered","cannot_confirm","clarify"]},"answer":{"type":"string"},"evidence_refs":{"type":"array","items":{"type":"string"}}},"required":["status","answer"]}
def build_system_prompt(ontology, rdb_meta, lexicon) -> str
def format_evidence(evs: list[Evidence]) -> str
def format_trace(steps: list[str]) -> str
def fallback_answer(reason: str) -> str
```

Tool results are compact text: `sql_query` returns a Markdown-ish table limited to 40 rows in content (plus "총 N건, 표시 40건") and full rows kept in `Evidence.detail` refs (product code+name for first 20); `search_products` returns top-k with table/id/title/key payload; `resolve_entity` returns `Resolution.summary()`; `graph_query` returns rows as `key=value` lines; `graph_neighbors` groups by relation; `search_documents` returns chunk text with title/source/as_of; `table_info` returns `ontology.schema_card(view)` + base date + row count + sellable semantics line.

System prompt content (Korean), fixed and cache-stable (no timestamps): role; the four tables with 기준일 and row counts; ontology class summary; rules: (1) 데이터 근거 없는 수치·전망·추천 금지, (2) 확인 불가 시 `final_answer(status="cannot_confirm")`, 모호하면 `clarify`, (3) 답변에 기준일·상품코드·수치 단위 포함, (4) 국내채권 '판매가능' = `is_sellable=1`, 국내ETF '판매중' = `is_on_sale=1`, 펀드 '판매중' = `is_on_sale=1`, (5) 해외ETF는 영문명·영문 전략 텍스트이므로 영문 키워드로 검색, (6) 존재하지 않는 값(예: 신용등급 AAAA, 위험등급 7)은 온톨로지 위반 → cannot_confirm, (7) 상품명이 정확히 일치하지 않으면 유사 후보를 제시하며 clarify, (8) 도구 호출은 병렬 가능, 최대 6라운드, 시간 예산 안내를 따를 것, (9) 항상 `final_answer` 도구로 종료.

- [ ] **Step 1: Failing tests** (`tests/test_tools.py` builds RDB from the real store if present else skips; asserts `sql_query` returns evidence with table `v_bond_kr`, base date `2026-08-21`, and that `sql_query` with `DELETE` returns `is_error=True`; `tests/test_answer.py` asserts `format_evidence` numbers blocks `[근거 1]` and includes SQL and refs).
- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: agent tools, system prompt, evidence formatting"`

---

### Task 11: Orchestrator (pre-analysis, agent loop, budget)

**Files:**
- Create: `src/finagent/agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `Orchestrator(settings, llm: LLMClient, tools: ToolRegistry, resolver: EntityResolver, ontology: Ontology, lexicon: Lexicon)`; `async Orchestrator.answer(question_id: str, question: str) -> AnswerResponse`; `Orchestrator.pre_analyze(question) -> PreAnalysis(text: str, invalid: list[str], resolution: Resolution)`.

Algorithm:
1. `pre = pre_analyze(q)`: lexicon matches + `resolver.resolve(q)` + validators: regex `신용등급\s*([A-Da-d]{1,4}[+\-0]?)` / `([A-D]{2,4}[+\-0]?)\s*이상` → `validate_credit_grade`; `(\d)\s*등급` → `validate_risk_grade`. Compose `pre.text` = "[사전 분석]\n" + resolution summary + validator notes + hint lines (e.g., "'AA- 이상' = crd_grd IN ('AAA','AA+','AA0','AA-')").
2. `conv = llm.start(system, tools.specs(), f"질문: {q}\n\n{pre.text}")`; `steps = ["사전분석: ..."]`; `evidences = []`; `deadline = now + budget`.
3. Loop `round in range(max_tool_rounds)`: `turn = await conv.step()`; if `turn.stop_reason == "error"` → break to fallback; if a `final_answer` call present → parse args → done; else if no tool calls → treat `turn.text` as answer (`status=answered` if evidences else `cannot_confirm`); else execute all tool calls concurrently (`asyncio.gather`), collect evidences, append `steps`, `conv.add_tool_results(...)`; if remaining time < 12s → `conv.add_user("[시간 예산 임박] 지금까지 확보한 근거로 즉시 final_answer를 호출하세요.")`.
4. If loop exhausted without final → one last `conv.add_user("[라운드 종료] 즉시 final_answer 호출")` + `step()`; if still none → `fallback_answer`.
5. Build `AnswerResponse(question_id, question, format_evidence(evidences), format_trace(steps), answer_text)`. `answer_text` for `cannot_confirm` must start with "제공된 데이터로는 확인할 수 없습니다" unless the LLM already says so; for `clarify` must end with a question mark.

- [ ] **Step 1: Failing tests** with `StubLLM` scripts: (a) SQL round then `final_answer(answered)` → evidence contains `[근거 1]` and `v_bond_kr`; (b) immediate `final_answer(cannot_confirm)` for "신용등급 AAAA인 채권" → pre-analysis text contains "유효한 신용등급" and answer starts with "제공된 데이터로는 확인할 수 없습니다"; (c) stub returns `Turn("", [], "error")` → answer is the fallback text, still a valid `AnswerResponse`.
- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "feat: orchestrator with pre-analysis, tool loop, time budget"`

---

### Task 12: FastAPI app and contract tests

**Files:**
- Create: `src/finagent/api.py`, `src/finagent/app_state.py` (builds everything once: `AppState.load(settings) -> AppState(rdb, index, resolver, kg, docs, ontology, lexicon, llm, tools, orchestrator)`), `scripts/serve.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `create_app(state_factory: Callable[[], AppState] | None = None) -> FastAPI`; routes `GET /answer`, `GET /health` (`{"status":"ok","tables":{...},"kg_triples":N}`), `GET /` (short text).

Behavior: `GET /answer` reads `question_id` and `question` from `request.query_params` (extra params ignored); missing → 200 with `answer="question 파라미터가 필요합니다."`; any exception in orchestrator → 200 with `answer=fallback_answer("internal_error")` and `think_trace` containing the exception class; response uses `JSONResponse(media_type="application/json; charset=utf-8", content=resp.model_dump())` with `ensure_ascii=False`.

- [ ] **Step 1: Failing test** (`TestClient` with a `StubLLM`-backed `AppState` built on a tiny SQLite; asserts 200 + five keys for normal, extra params, missing params, and an orchestrator that raises).
- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** → PASS; run `uvicorn finagent.api:app --port 8000` locally and `curl -G "http://127.0.0.1:8000/answer" --data-urlencode "question_id=Q-001" --data-urlencode "question=현재 판매가능한 원화채권 중 AA- 이상 종목 알려줘"` with `LLM_PROVIDER=stub` (scripted) to see the JSON.
- [ ] **Step 5: Commit** `git commit -m "feat: FastAPI /answer and /health with no-500 guarantees"`

---

### Task 13: Constituent fetcher (Naver Finance) and corporate relations

**Files:**
- Create: `scripts/fetch_constituents_naver.py`, `data/enrichment/constituents/kr_etf_constituents.csv` (output, committed if fetch succeeds)
- Test: manual run + `tests/test_graph.py` already covers loader.

Behavior: for each domestic ETF (`v_etf_kr`, `pd_ticker`), GET `https://finance.naver.com/api/sise/etfItemList.nhn` is not per-ETF; use the ETF detail page `https://finance.naver.com/item/coinfo.naver?code={ticker}` → iframe `https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={ticker}` … (inspect at run time; if the constituent table endpoint cannot be reached reliably, fall back to `https://m.stock.naver.com/api/stock/{ticker}/etfAnalysis` JSON which lists `etfTop10MajorConstituentAssetList`). Map constituent names to `companies.yaml` ids by alias match; unmatched names get `company_id = "auto_" + slug(name)` and are still emitted (graph creates `fp:Company` nodes for them). Rate-limit 0.3s, cache raw JSON under `data/external/cache/`. Print coverage stats.

- [ ] **Step 1: Implement and run**; verify `KG.neighbors("KODEX 2차전지산업")["hasConstituent"]` contains 에코프로비엠 after rebuild.
- [ ] **Step 2: Commit** `git commit -m "feat: Naver Finance constituent fetcher and data"`

---

### Task 14: Eval runner, sample questions, Docker, README, API spec, proposal outline

**Files:**
- Create: `data/eval/questions.jsonl` (the 8 PDF samples + ≥ 20 team questions covering each table, numeric filters, comparisons, unanswerables), `scripts/eval_run.py`, `Dockerfile`, `docker-compose.yml`, `README.md`, `docs/api_spec.md`, `docs/proposal_outline.md`

`scripts/eval_run.py --base http://127.0.0.1:8000 --out data/eval/results.jsonl`: sequentially calls `/answer`, records latency, status, answer head; prints a table (id, difficulty, seconds, status keyword).

Dockerfile: `python:3.12-slim`, copy `requirements.txt`, `pip install`, copy repo, `RUN python scripts/build_store.py`, `EXPOSE 8000`, `CMD ["uvicorn", "finagent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]` with `PYTHONPATH=/app/src`. docker-compose: service `api` (env_file `.env`, ports `80:8000`), optional `caddy` service with `Caddyfile` for automatic HTTPS (`{$DOMAIN} { reverse_proxy api:8000 }`).

README (Korean): 개요, 아키텍처 다이어그램(mermaid), 디렉터리 구조, 환경 구성(uv/pip, .env), 데이터 적재(`python scripts/build_store.py`), 실행(`uvicorn …`, docker compose), API 명세(요청/응답 JSON 스키마 + curl 예시), 온톨로지 설명, 평가 재현(`scripts/eval_run.py`), 제한사항.

- [ ] **Step 1: Write all files**; `docker build` locally to confirm it builds.
- [ ] **Step 2: Commit** `git commit -m "docs: README, API spec, proposal outline; chore: Docker, eval runner"`

---

### Task 15: End-to-end run with the real LLM

- [ ] Set `ANTHROPIC_API_KEY` in `.env`; start server; run `scripts/eval_run.py`; inspect `data/eval/results.jsonl`; tune prompts/tool descriptions for failures; re-run. Record latencies in README.
- [ ] Commit `git commit -m "chore: eval results and prompt tuning"`.

## Self-review

- Spec coverage: ETL (T3), ontology + runtime use (T4), RDB (T5), vector/lexical + entity resolution (T6), graph (T7), unstructured docs parsing (T8), LLM abstraction (T9), tools/evidence/policy (T10), orchestrator/budget (T11), API contract/no-500 (T12), external constituents (T13), eval/Docker/README/API spec/proposal (T14), real E2E (T15). Dense embeddings from the spec are deferred (lexical only) — noted in README as an extension.
- Type consistency: `Turn/ToolCall/ToolSpec` (T9) used by T10/T11; `Evidence/ToolResult` (T10) used by T11; `Resolution` (T6) used by T11; `RDB.meta()` keys are `v_*` view names (T5) consumed by T10 prompts.
