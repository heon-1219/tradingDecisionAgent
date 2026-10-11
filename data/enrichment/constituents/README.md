English | [한국어](#한국어)

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
  Primary endpoint actually used: `https://m.stock.naver.com/api/etf/{ticker}/constituent` (paginated, `?cursor=`).
- **Fetched on**: 2026-09-05
- **Coverage**: ETFs attempted 1235, with >=1 constituent 1162,
  failed/empty 73 (mostly delisted or not-on-sale funds), rows 22599.
  ETFs by source: constituent=1162, etfAnalysis=0,
  wisereport=0, none=73. Empty ones are delisted/not-trading funds
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
  실제로 사용한 주 엔드포인트: `https://m.stock.naver.com/api/etf/{ticker}/constituent` (페이지 단위, `?cursor=`).
- **수집일**: 2026-09-05
- **수록 범위**: 시도한 ETF 1235개, 편입종목 1개 이상 확보 1162개,
  실패/빈 결과 73개(대부분 상장폐지되었거나 판매하지 않는 펀드), 행 22599개.
  원천별 ETF 수: constituent=1162, etfAnalysis=0,
  wisereport=0, none=73. 빈 결과는 네이버가 포트폴리오를 주지 않는
  상장폐지·거래정지 펀드(`pd_sale_yn=0` 또는 `pd_tr_yn=0`)입니다.
- **다시 수집하기** (Git Bash, 저장소 루트):
  `"$PWD/.venv/Scripts/python.exe" scripts/fetch_constituents_naver.py`
  원본 응답은 `data/external/cache/naver_etf/`(gitignore 대상)에 캐시되고 재실행 시 그대로 재사용됩니다.
  전부 새로 받으려면 `--no-cache`, 빈 결과만 다시 받으려면 `--retry-failed`를 쓰세요.
  실패 목록은 `data/external/cache/naver_etf/_failures.json`에 남습니다.
- **라이선스 / 이용약관**: 네이버금융 데이터는 개인 연구 용도로만 쓸 수 있으며, 재배포가
  허가된 데이터 피드가 아닙니다. 제출 전에 다시 수집하고 출처로 네이버를 명시하세요.
