# KR ETF constituents (Naver Finance, best-effort)

Constituent holdings for domestic (Korean-listed) ETFs, used by the financial-product QA agent to
answer "which ETFs hold company X" / "what does ETF Y hold" questions.

- **File**: `kr_etf_constituents.csv` (UTF-8). Columns: `etf_id,etf_table,ticker,etf_name,company_id,company_name,weight_pct,as_of,source`
  - `etf_id` = ISIN (`pd_itm_no` in `data/raw/pref01n001_data.xlsx`), `etf_table` = `v_etf_kr`, `ticker` = 6-char short code.
  - `company_id` = id from `data/enrichment/companies.yaml` when a ticker/alias matches, else `auto_<slug>`.
  - `weight_pct` = % of NAV (empty when the source has no weight). Top 30 holdings per ETF.
  - `as_of` = date given in the response when present, else the fetch date. `source` = URL used.
- **Source**: Naver Finance (`m.stock.naver.com` JSON API; fallback `finance.naver.com` / wisereport iframe).
  Primary endpoint actually used: `https://m.stock.naver.com/api/etf/{ticker}/constituent` (paginated, `?cursor=`).
- **Fetched on**: 2026-09-03
- **Coverage**: ETFs attempted 1235, with >=1 constituent 1162,
  failed/empty 73 (mostly delisted or not-on-sale funds), rows 22599.
  Rows by source: constituent=1162, etfAnalysis=0,
  wisereport=0, none=73.
- **Re-run** (Git Bash, repo root):
  `"$PWD/.venv/Scripts/python.exe" scripts/fetch_constituents_naver.py`
  Raw responses are cached in `data/external/cache/naver_etf/` (gitignored) and replayed on re-run;
  use `--no-cache` to refresh everything or `--retry-failed` to re-fetch only empty ones.
  Failures are listed in `data/external/cache/naver_etf/_failures.json`.
- **License / ToS**: Naver Finance data is for personal research use only; it is not a licensed
  redistribution feed. Re-fetch before any submission and cite Naver as the source.
