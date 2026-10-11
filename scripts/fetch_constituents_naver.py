"""Best-effort fetcher for domestic (KR) ETF constituent holdings from Naver Finance.

Reads the ETF universe from data/raw/pref01n001_data.xlsx (sheet 'data', pd_grp_no == 'ETF'),
fetches constituents per ETF from Naver, maps constituent names/tickers onto
data/enrichment/companies.yaml ids, and writes
data/enrichment/constituents/kr_etf_constituents.csv (+ README.md with coverage stats).

Sources tried per ETF, in order (first non-empty wins):
  1. https://m.stock.naver.com/api/etf/{ticker}/constituent   (JSON, paginated via ?cursor=, 10/page)
  2. https://m.stock.naver.com/api/stock/{ticker}/etfAnalysis  (JSON, etfTop10MajorConstituentAssets)
  3. https://finance.naver.com/item/coinfo.naver?code={ticker} (HTML, EUC-KR) -> wisereport iframe
     https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={ticker} (embedded CU_data JSON)

Every raw HTTP response is cached under data/external/cache/naver_etf/{ticker}.json and replayed
on re-run (no network) unless --no-cache / --retry-failed is given.

Usage (from repo root, Git Bash):
  "$REPO/.venv/Scripts/python.exe" scripts/fetch_constituents_naver.py            # full run
  ... --tickers 305720,0164G0        # subset
  ... --limit 50                     # first N (on-sale rows first)
  ... --retry-failed                 # re-fetch tickers whose cache holds no constituents
  ... --no-verify                    # skip the verification printout
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "pref01n001_data.xlsx"
COMPANIES_YAML = ROOT / "data" / "enrichment" / "companies.yaml"
OUT_DIR = ROOT / "data" / "enrichment" / "constituents"
OUT_CSV = OUT_DIR / "kr_etf_constituents.csv"
OUT_README = OUT_DIR / "README.md"
CACHE_DIR = ROOT / "data" / "external" / "cache" / "naver_etf"
FAILURES_JSON = CACHE_DIR / "_failures.json"

ETF_TABLE = "v_etf_kr"
# company_code is the constituent's own stock code as the source reports it (005930 for 삼성전자 [Samsung
# Electronics], 005935 for 삼성전자우 [its preferred share]).  It used to be dropped, which left the name as the only
# identity and forced downstream code to guess: 'LG' was matched to 신한금융지주 [Shinhan Financial Group] and 'KT'
# to SK텔레콤 [SK Telecom] by substring.  Keep the key the source hands us — it is the one field that cannot be
# reinterpreted.
CSV_HEADER = [
    "etf_id", "etf_table", "ticker", "etf_name", "company_code", "company_id", "company_name",
    "weight_pct", "as_of", "source",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS_M = {"User-Agent": UA, "Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
HEADERS_PC = {"User-Agent": UA, "Referer": "https://finance.naver.com/"}

URL_CONSTITUENT = "https://m.stock.naver.com/api/etf/{ticker}/constituent"
URL_ETF_ANALYSIS = "https://m.stock.naver.com/api/stock/{ticker}/etfAnalysis"
URL_COINFO = "https://finance.naver.com/item/coinfo.naver?code={ticker}"
URL_WISEREPORT = "https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={ticker}"

# Rows skipped only when weight is missing AND the name is one of these (KRW cash, creation cash, USD, futures).
SKIP_NAMES_IF_NO_WEIGHT = {"원화현금", "설정현금액", "USD", "선물"}

# Corporate suffix tokens stripped for the secondary ("core") name match.
CORE_SUFFIX_RE = re.compile(
    r"(?:\s*(?:-a|-h|class\s*a|co\.?,?\s*ltd\.?|co\.?\s*ltd\.?|company|corporation|corp\.?|inc\.?|"
    r"ltd\.?|limited|plc|ag|sa|nv|se|spa|holdings?|group|co\.?))+\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------------
# Normalisation / company mapping
# --------------------------------------------------------------------------------------
def normalize(s: str) -> str:
    """NFKC -> lowercase -> drop whitespace -> strip (주)/주식회사 (the Korean 'Co., Ltd.' markers)."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("(주)", "").replace("주식회사", "")
    return s


def core_name(s: str) -> str:
    """Latin-name normaliser that drops trailing corporate suffixes (Co Ltd, Corp, -A ...)."""
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    prev = None
    while prev != s:
        prev = s
        s = CORE_SUFFIX_RE.sub("", s).strip(" ,.-")
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = re.sub(r"[^0-9a-z가-힣]+", "_", s).strip("_")
    return s or "unknown"


def is_latin(s: str) -> bool:
    return not re.search(r"[가-힣一-鿿]", s or "")


@dataclass
class CompanyIndex:
    by_norm: dict[str, str] = field(default_factory=dict)
    by_ticker: dict[str, str] = field(default_factory=dict)
    by_core: dict[str, str] = field(default_factory=dict)
    long_cores: list[tuple[str, str]] = field(default_factory=list)  # (core, id), len >= 10

    @classmethod
    def load(cls, path: Path) -> "CompanyIndex":
        items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if isinstance(items, dict):
            items = items.get("companies", [])
        idx = cls()
        for it in items:
            cid = str(it.get("id") or "").strip()
            if not cid:
                continue
            names = [it.get("name_ko"), it.get("name_en"), *(it.get("aliases") or [])]
            for n in names:
                if not n:
                    continue
                n = str(n)
                idx.by_norm.setdefault(normalize(n), cid)
                c = core_name(n)
                if c:
                    idx.by_core.setdefault(c, cid)
                    if is_latin(n) and len(c) >= 10:
                        idx.long_cores.append((c, cid))
            for t in it.get("tickers") or []:
                t = str(t).strip()
                if t:
                    idx.by_ticker.setdefault(t, cid)
                    idx.by_ticker.setdefault(t.lstrip("0") or t, cid)
        idx.long_cores.sort(key=lambda x: -len(x[0]))
        return idx

    def match(self, name: str, code: str | None) -> str:
        code = (code or "").strip()
        if code:
            base = code.split(".")[0].strip()
            for cand in (code, base, base.lstrip("0") or base):
                if cand in self.by_ticker:
                    return self.by_ticker[cand]
        n = normalize(name)
        if n in self.by_norm:
            return self.by_norm[n]
        if n in self.by_ticker:
            return self.by_ticker[n]
        c = core_name(name)
        if c and c in self.by_core:
            return self.by_core[c]
        if c and is_latin(name):
            for lc, cid in self.long_cores:
                if c.startswith(lc):
                    return cid
        return "auto_" + slug(name)


# --------------------------------------------------------------------------------------
# HTTP + cache
# --------------------------------------------------------------------------------------
class Fetcher:
    def __init__(self, sleep: float = 0.35, timeout: float = 10.0, retries: int = 2):
        self.sleep = sleep
        self.retries = retries
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.n_requests = 0
        self._last = 0.0

    def close(self) -> None:
        self.client.close()

    def get(self, url: str, headers: dict[str, str], kind: str) -> dict[str, Any]:
        """Returns {url, kind, status, body (json|text|None), error}."""
        rec: dict[str, Any] = {"url": url, "kind": kind, "status": None, "body": None, "error": None}
        for attempt in range(self.retries + 1):
            wait = self.sleep - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.client.get(url, headers=headers)
                self._last = time.monotonic()
                self.n_requests += 1
                rec["status"] = r.status_code
                ctype = r.headers.get("content-type", "")
                if "json" in ctype:
                    try:
                        rec["body"] = r.json()
                    except ValueError:
                        rec["body"] = r.text
                elif "euc-kr" in ctype.lower() or "ks_c_5601" in ctype.lower():
                    rec["body"] = r.content.decode("euc-kr", "replace")
                else:
                    rec["body"] = r.text
                rec["error"] = None
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                return rec
            except (httpx.HTTPError, httpx.InvalidURL) as e:  # noqa: PERF203
                self._last = time.monotonic()
                rec["error"] = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        return rec


def fetch_ticker(fetcher: Fetcher, ticker: str, max_pages: int) -> dict[str, Any]:
    """Runs the source chain for one ticker; returns the cache envelope."""
    env: dict[str, Any] = {
        "ticker": ticker,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "requests": [],
    }
    reqs = env["requests"]

    # 1) paginated constituent endpoint
    url = URL_CONSTITUENT.format(ticker=ticker)
    rec = fetcher.get(url, HEADERS_M, "constituent")
    reqs.append(rec)
    body = rec["body"]
    if isinstance(body, dict) and body.get("result"):
        pages = 1
        while body.get("hasNext") and body.get("nextCursor") and pages < max_pages:
            rec = fetcher.get(f"{url}?cursor={body['nextCursor']}", HEADERS_M, "constituent")
            reqs.append(rec)
            body = rec["body"]
            if not (isinstance(body, dict) and body.get("result")):
                break
            pages += 1
        return env

    # 2) etfAnalysis top-10
    rec = fetcher.get(URL_ETF_ANALYSIS.format(ticker=ticker), HEADERS_M, "etfAnalysis")
    reqs.append(rec)
    body = rec["body"]
    if isinstance(body, dict) and body.get("etfTop10MajorConstituentAssets"):
        return env

    # 3) coinfo HTML -> wisereport iframe
    rec = fetcher.get(URL_COINFO.format(ticker=ticker), HEADERS_PC, "coinfo")
    reqs.append(rec)
    wise_url = URL_WISEREPORT.format(ticker=ticker)
    if isinstance(rec["body"], str):
        m = re.search(r'<iframe[^>]+id="coinfo_cp"[^>]+src="([^"]+)"', rec["body"])
        if m:
            wise_url = m.group(1).replace("&amp;", "&")
    rec = fetcher.get(wise_url, HEADERS_PC, "wisereport")
    reqs.append(rec)
    return env


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
@dataclass
class Holding:
    name: str
    code: str | None
    weight: float | None
    seq: int


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "—", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_date(s: Any) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4})[.\-/]?(\d{2})[.\-/]?(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.fullmatch(r"(\d{2})[.\-/](\d{2})[.\-/](\d{2})", s)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def parse_envelope(env: dict[str, Any]) -> tuple[list[Holding], str | None, str | None, str]:
    """-> (holdings, source_url, as_of, kind). Empty holdings + kind='none' when nothing usable."""
    reqs = env.get("requests") or []

    # 1) constituent pages
    const = [r for r in reqs if r.get("kind") == "constituent" and isinstance(r.get("body"), dict)]
    holdings: list[Holding] = []
    seen: set[tuple[str, str]] = set()
    for r in const:
        for it in r["body"].get("result") or []:
            name = str(it.get("itemName") or "").strip()
            if not name:
                continue
            code = it.get("itemCode") or it.get("reutersCode") or None
            key = (name, str(code or ""))
            if key in seen:
                continue
            seen.add(key)
            holdings.append(Holding(name, code, _to_float(it.get("constituentWeight")), len(holdings)))
    if holdings:
        return holdings, const[0]["url"], None, "constituent"

    # 2) etfAnalysis
    for r in reqs:
        if r.get("kind") == "etfAnalysis" and isinstance(r.get("body"), dict):
            body = r["body"]
            items = body.get("etfTop10MajorConstituentAssets") or []
            for it in items:
                name = str(it.get("itemName") or it.get("stockName") or "").strip()
                if not name:
                    continue
                w = _to_float(it.get("etfWeight", it.get("weight")))
                holdings.append(Holding(name, it.get("itemCode") or None, w, len(holdings)))
            if holdings:
                as_of = _norm_date(body.get("navPerformanceReferenceDate")
                                   or body.get("returnPerformanceReferenceDate"))
                return holdings, r["url"], as_of, "etfAnalysis"

    # 3) wisereport HTML (embedded CU_data JSON, else the CU table)
    for r in reqs:
        if r.get("kind") == "wisereport" and isinstance(r.get("body"), str):
            html = r["body"]
            as_of = None
            m = re.search(r"var\s+CU_data\s*=\s*(\{.*?\});", html, re.S)
            if m:
                try:
                    grid = json.loads(m.group(1)).get("grid_data") or []
                except ValueError:
                    grid = []
                for it in grid:
                    name = str(it.get("STK_NM_KOR") or "").strip()
                    if not name:
                        continue
                    as_of = as_of or _norm_date(it.get("TRD_DT"))
                    holdings.append(Holding(name, None, _to_float(it.get("ETF_WEIGHT")), len(holdings)))
            if not holdings:
                m2 = re.search(r"CU당 구성종목.*?<table.*?</table>", html, re.S)
                if m2:
                    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m2.group(0), re.S):
                        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
                        if len(cells) >= 2 and cells[0] and not cells[0].startswith("종목"):
                            holdings.append(Holding(cells[0], None, _to_float(cells[-1]), len(holdings)))
            if as_of is None:
                m3 = re.search(r"\[기준\s*:\s*([\d.]+)\]", html)
                if m3:
                    as_of = _norm_date(m3.group(1))
            if holdings:
                return holdings, r["url"], as_of, "wisereport"

    return [], None, None, "none"


def select_top(holdings: list[Holding], top_n: int) -> list[Holding]:
    kept = [
        h for h in holdings
        if not (h.weight is None and h.name.strip() in SKIP_NAMES_IF_NO_WEIGHT)
    ]
    kept.sort(key=lambda h: (h.weight is None, -(h.weight or 0.0), h.seq))
    return kept[:top_n]


# --------------------------------------------------------------------------------------
# ETF universe
# --------------------------------------------------------------------------------------
def load_etfs(only_on_sale: bool = False) -> pd.DataFrame:
    df = pd.read_excel(RAW_XLSX, sheet_name="data", dtype=str)
    df = df[df["pd_grp_no"].str.strip() == "ETF"].copy()

    def derive_ticker(row: pd.Series) -> str | None:
        t = row.get("pd_ticker")
        if isinstance(t, str) and len(t.strip()) == 6:
            return t.strip()
        isin = str(row.get("pd_itm_no") or "").strip()
        if len(isin) == 12 and isin.startswith("KR7"):
            return isin[3:9]
        return None

    df["ticker"] = df.apply(derive_ticker, axis=1)
    df["on_sale"] = (df["pd_sale_yn"].fillna("0").str.strip() == "1").astype(int)
    df["_ord"] = range(len(df))
    if only_on_sale:
        df = df[df["on_sale"] == 1]
    df = df.sort_values(["on_sale", "_ord"], ascending=[False, True]).reset_index(drop=True)
    return df[["pd_itm_no", "ticker", "pd_abrv_nm", "on_sale"]]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    etfs = load_etfs(only_on_sale=args.only_on_sale)
    if args.tickers:
        wanted = {t.strip() for t in args.tickers.split(",") if t.strip()}
        etfs = etfs[etfs["ticker"].isin(wanted)]
    if args.limit:
        etfs = etfs.head(args.limit)

    companies = CompanyIndex.load(COMPANIES_YAML)
    fetcher = Fetcher(sleep=args.sleep, timeout=args.timeout, retries=args.retries)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_ok = 0
    n_cache_hits = 0
    kind_counter: Counter[str] = Counter()
    t0 = time.monotonic()
    total = len(etfs)
    print(f"[fetch] {total} ETFs to process (on-sale first). cache={CACHE_DIR}")

    try:
        for i, row in enumerate(etfs.itertuples(index=False), start=1):
            isin, ticker, name = row.pd_itm_no, row.ticker, str(row.pd_abrv_nm or "").strip()
            if not ticker:
                failures.append({"etf_id": isin, "ticker": None, "etf_name": name, "reason": "no ticker"})
                continue

            cache_file = CACHE_DIR / f"{ticker}.json"
            env: dict[str, Any] | None = None
            if cache_file.exists() and not args.no_cache:
                try:
                    env = json.loads(cache_file.read_text(encoding="utf-8"))
                except ValueError:
                    env = None
                if env is not None and args.retry_failed and parse_envelope(env)[3] == "none":
                    env = None
                if env is not None:
                    n_cache_hits += 1
            if env is None:
                env = fetch_ticker(fetcher, ticker, max_pages=args.max_pages)
                cache_file.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")

            holdings, source, as_of, kind = parse_envelope(env)
            kind_counter[kind] += 1
            if not holdings:
                errs = [r.get("error") or f"HTTP {r.get('status')}" for r in env.get("requests", [])]
                failures.append({"etf_id": isin, "ticker": ticker, "etf_name": name,
                                 "reason": "no constituents", "attempts": errs})
            else:
                n_ok += 1
                for h in select_top(holdings, args.top):
                    rows.append({
                        "etf_id": isin,
                        "etf_table": ETF_TABLE,
                        "ticker": ticker,
                        "etf_name": name,
                        "company_code": h.code or "",
                        "company_id": companies.match(h.name, h.code),
                        "company_name": h.name,
                        "weight_pct": "" if h.weight is None else f"{h.weight:.4f}".rstrip("0").rstrip("."),
                        "as_of": as_of or today,
                        "source": source,
                    })

            if i % 100 == 0 or i == total:
                el = time.monotonic() - t0
                print(f"[fetch] {i}/{total} done | ok={n_ok} fail={len(failures)} rows={len(rows)} "
                      f"| requests={fetcher.n_requests} cache_hits={n_cache_hits} | {el:.0f}s")
    finally:
        fetcher.close()

    # ---- write outputs
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    FAILURES_JSON.write_text(json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8")

    stats = {
        "attempted": total,
        "succeeded": n_ok,
        "failed": len(failures),
        "rows": len(rows),
        "requests": fetcher.n_requests,
        "cache_hits": n_cache_hits,
        "by_source": dict(kind_counter),
        "fetched_on": today,
    }
    write_readme(stats)
    print(f"[fetch] wrote {OUT_CSV} ({len(rows)} rows) and {OUT_README}")
    print(f"[fetch] failures: {len(failures)} (see {FAILURES_JSON}); by_source={dict(kind_counter)}")

    if not args.no_verify:
        verify(rows, etfs)


def write_readme(stats: dict[str, Any]) -> None:
    bs = stats["by_source"]
    text = f"""English | [한국어](#한국어)

# KR ETF constituents (Naver Finance, best-effort)

Constituent holdings for domestic (Korean-listed) ETFs, used by the financial-product QA agent to
answer "which ETFs hold company X" / "what does ETF Y hold" questions.

- **File**: `kr_etf_constituents.csv` (UTF-8). Columns: `etf_id,etf_table,ticker,etf_name,company_id,company_name,weight_pct,as_of,source`
  - `etf_id` = ISIN (`pd_itm_no` in `data/raw/pref01n001_data.xlsx`), `etf_table` = `v_etf_kr`, `ticker` = 6-char short code.
  - `company_id` = id from `data/enrichment/companies.yaml` when a ticker/alias matches, else `auto_<slug>`.
  - `weight_pct` = % of NAV (empty when the source has no weight). Top 30 holdings per ETF.
  - `as_of` = date given in the response when present, else the fetch date. The primary endpoint carries no
    portfolio date, so `as_of` is normally the fetch date (Naver's PDF is typically the previous trading day).
  - `source` = URL used. Foreign holdings are given in Naver's Korean transliteration (e.g. 한무기 = Cambricon);
    they are mapped through the `reutersCode` ticker (e.g. `688256.SS`) when the ticker is in companies.yaml.
- **Source**: Naver Finance (`m.stock.naver.com` JSON API; fallback `finance.naver.com` / wisereport iframe).
  Primary endpoint actually used: `https://m.stock.naver.com/api/etf/{{ticker}}/constituent` (paginated, `?cursor=`).
- **Fetched on**: {stats['fetched_on']}
- **Coverage**: ETFs attempted {stats['attempted']}, with >=1 constituent {stats['succeeded']},
  failed/empty {stats['failed']} (mostly delisted or not-on-sale funds), rows {stats['rows']}.
  ETFs by source: constituent={bs.get('constituent', 0)}, etfAnalysis={bs.get('etfAnalysis', 0)},
  wisereport={bs.get('wisereport', 0)}, none={bs.get('none', 0)}. Empty ones are delisted/not-trading funds
  (`pd_sale_yn=0` or `pd_tr_yn=0`) for which Naver returns no portfolio.
- **Re-run** (Git Bash, repo root):
  `"$PWD/.venv/Scripts/python.exe" scripts/fetch_constituents_naver.py`
  Raw responses are cached in `data/external/cache/naver_etf/` (gitignored) and replayed on re-run;
  use `--no-cache` to refresh everything or `--retry-failed` to re-fetch only empty ones.
  Failures are listed in `data/external/cache/naver_etf/_failures.json`.
- **License / ToS**: Naver Finance data is for personal research use only; it is not a licensed
  redistribution feed. Re-fetch before any submission and cite Naver as the source.

---

## 한국어

[English](#kr-etf-constituents-naver-finance-best-effort) | 한국어

# 국내 ETF 편입종목 (네이버금융, best-effort 수집)

국내 상장 ETF의 편입종목(구성종목) 데이터입니다. 금융상품 QA 에이전트가
"기업 X를 편입한 ETF는?" / "ETF Y는 무엇을 담고 있나?" 같은 질문에 답할 때 씁니다.

- **파일**: `kr_etf_constituents.csv` (UTF-8). 컬럼: `etf_id,etf_table,ticker,etf_name,company_id,company_name,weight_pct,as_of,source`
  - `etf_id` = ISIN(`data/raw/pref01n001_data.xlsx`의 `pd_itm_no`), `etf_table` = `v_etf_kr`, `ticker` = 6자리 단축코드.
  - `company_id` = 티커·별칭이 일치하면 `data/enrichment/companies.yaml`의 id, 아니면 `auto_<slug>`.
  - `weight_pct` = 순자산(NAV) 대비 비중 %(원천에 비중이 없으면 빈 값). ETF당 상위 30개 편입종목.
  - `as_of` = 응답에 날짜가 있으면 그 날짜, 없으면 수집일. 주 엔드포인트는 포트폴리오 기준일을
    주지 않으므로 `as_of`는 보통 수집일입니다(네이버의 PDF는 대개 직전 거래일 기준).
  - `source` = 사용한 URL. 해외 편입종목은 네이버의 한글 음차 표기로 나옵니다(예: 한무기 = Cambricon).
    해당 티커가 companies.yaml에 있으면 `reutersCode` 티커(예: `688256.SS`)로 연결합니다.
- **출처**: 네이버금융(`m.stock.naver.com` JSON API; 대체 경로 `finance.naver.com` / wisereport iframe).
  실제로 사용한 주 엔드포인트: `https://m.stock.naver.com/api/etf/{{ticker}}/constituent` (페이지 단위, `?cursor=`).
- **수집일**: {stats['fetched_on']}
- **수록 범위**: 시도한 ETF {stats['attempted']}개, 편입종목 1개 이상 확보 {stats['succeeded']}개,
  실패/빈 결과 {stats['failed']}개(대부분 상장폐지되었거나 판매하지 않는 펀드), 행 {stats['rows']}개.
  원천별 ETF 수: constituent={bs.get('constituent', 0)}, etfAnalysis={bs.get('etfAnalysis', 0)},
  wisereport={bs.get('wisereport', 0)}, none={bs.get('none', 0)}. 빈 결과는 네이버가 포트폴리오를 주지 않는
  상장폐지·거래정지 펀드(`pd_sale_yn=0` 또는 `pd_tr_yn=0`)입니다.
- **다시 수집하기** (Git Bash, 저장소 루트):
  `"$PWD/.venv/Scripts/python.exe" scripts/fetch_constituents_naver.py`
  원본 응답은 `data/external/cache/naver_etf/`(gitignore 대상)에 캐시되고 재실행 시 그대로 재사용됩니다.
  전부 새로 받으려면 `--no-cache`, 빈 결과만 다시 받으려면 `--retry-failed`를 쓰세요.
  실패 목록은 `data/external/cache/naver_etf/_failures.json`에 남습니다.
- **라이선스 / 이용약관**: 네이버금융 데이터는 개인 연구 용도로만 쓸 수 있으며, 재배포가
  허가된 데이터 피드가 아닙니다. 제출 전에 다시 수집하고 출처로 네이버를 명시하세요.
"""
    OUT_README.write_text(text, encoding="utf-8")


def verify(rows: list[dict[str, Any]], etfs: pd.DataFrame) -> None:
    print("\n========== VERIFY ==========")
    by_etf: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_etf.setdefault(r["ticker"], []).append(r)
    print(f"ETFs with >=1 constituent: {len(by_etf)} / {len(etfs)} ; total rows: {len(rows)}")

    def show(ticker: str, n: int = 40) -> list[dict[str, Any]]:
        rs = by_etf.get(ticker, [])
        nm = rs[0]["etf_name"] if rs else etfs.loc[etfs["ticker"] == ticker, "pd_abrv_nm"].head(1).tolist()
        print(f"\n-- {ticker} {nm} ({len(rs)} rows)")
        for r in rs[:n]:
            print(f"   {r['weight_pct']:>7}  {r['company_name']:<36} -> {r['company_id']}")
        return rs

    rs = show("305720", 10)
    print("   contains ecopro_bm:", any(r["company_id"] == "ecopro_bm" for r in rs))

    mask = etfs["pd_abrv_nm"].fillna("").str.contains("차이나") & etfs["pd_abrv_nm"].fillna("").str.contains("반도체")
    for t in etfs.loc[mask, "ticker"].dropna():
        rs = show(t)
        print("   maps to cambricon:", any(r["company_id"] == "cambricon" for r in rs))

    auto = Counter(r["company_name"] for r in rows if r["company_id"].startswith("auto_"))
    print("\n-- 10 most common auto_ company names (rows = ETF appearances):")
    for name, c in auto.most_common(10):
        print(f"   {c:4d}  {name}")
    print("============================")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", help="comma-separated subset of tickers")
    p.add_argument("--limit", type=int, default=0, help="process only the first N ETFs")
    p.add_argument("--only-on-sale", action="store_true", help="skip pd_sale_yn != 1")
    p.add_argument("--top", type=int, default=30, help="max constituents per ETF")
    p.add_argument("--max-pages", type=int, default=3, help="constituent pages (10 rows each)")
    p.add_argument("--sleep", type=float, default=0.35, help="seconds between requests")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    p.add_argument("--retry-failed", action="store_true", help="re-fetch cached tickers with no constituents")
    p.add_argument("--no-verify", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
