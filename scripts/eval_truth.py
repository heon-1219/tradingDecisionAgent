"""Score answers against a computed answer key.

The organizers never published an evaluation question set or answers ("예시질문(참고용 질의set)은 본
자료에 포함되지 않습니다"), so there is nothing official to grade against.  This builds the next best
thing: questions whose correct answer is *derivable from the delivered data itself* — a count, a name,
a top-N list — computed by SQL or by a graph lookup at scoring time.  That gives a real accuracy
number instead of a subjective read of the answers.

  python scripts/eval_truth.py --compute                      # 정답 키만 계산해서 출력 (LLM 불필요)
  python scripts/eval_truth.py --base http://127.0.0.1:8000   # 서버에 물어보고 채점

Checks are deliberately loose about formatting: numbers match with or without thousands separators,
names match after whitespace/case normalisation, and an "cannot_confirm" item passes when the answer
declines rather than inventing a result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from finagent.config import get_settings  # noqa: E402
from finagent.kb.graph import KG  # noqa: E402
from finagent.kb.lexicon import Lexicon  # noqa: E402
from finagent.kb.rdb import RDB  # noqa: E402

# "없습니다" 같은 흔한 종결어미는 넣지 않는다 — 정상 답변에도 자주 나와 오탐이 된다.
DECLINE = ("확인할 수 없", "확인되지 않", "제공되지 않", "존재하지 않", "찾을 수 없", "데이터에 없")
CLARIFY = ("혹시", "말씀하시", "찾으시는", "다시 확인")
# 시스템 장애로 생성된 대체 답변. 문구상 '확인할 수 없습니다'로 시작하므로 그냥 두면
# 확인불가 문항이 전부 통과한 것처럼 보인다 — 실제로는 모델이 판단조차 못 한 것이다.
ERROR_MARKERS = ("시스템 처리 중 문제가 발생", "<요청 실패", "내부 오류")


def is_infra_error(answer: str, trace: str = "") -> bool:
    return any(m in (answer or "") for m in ERROR_MARKERS) or "LLM 오류" in (trace or "")


def norm(s: str) -> str:
    return re.sub(r"[\s,·・\-_()\[\]]+", "", unicodedata.normalize("NFKC", str(s or "")).lower())


def compute_truth(item: dict, rdb: RDB, kg: KG | None) -> list[str]:
    """Return the list of strings the answer must contain."""
    kind = item.get("truth_kind")
    if kind in ("graph_count", "graph_join"):
        if kg is None:
            return []
        g = item["truth_graph"]
        hits = kg.neighbors(g["entity"], relation=g["relation"], limit=5000).get(g["relation"], [])
        if kind == "graph_count":
            return [str(len(hits))]
        ids = [h["iri"].rsplit("/", 1)[-1] for h in hits]
        if not ids:
            return []
        sql = item["truth_sql"].replace("{ids}", ",".join(f"'{i}'" for i in ids))
        return [str(r[0]) for r in rdb.fetch_all(sql)]
    if item.get("truth_sql"):
        return [str(r[0]) for r in rdb.fetch_all(item["truth_sql"])]
    return []


def score(item: dict, answer: str, truth: list[str], trace: str = "") -> tuple[bool, str]:
    check = item.get("check")
    if is_infra_error(answer, trace):
        return False, "시스템 오류 — 모델이 답하지 못함(채점 불가)"
    a_norm = norm(answer)
    if check == "cannot_confirm":
        declined = any(d in answer for d in DECLINE)
        asked_back = any(c in answer for c in CLARIFY) and "?" in answer
        ok = declined or asked_back
        return ok, "확인불가 명시" if declined else ("역질문" if asked_back else "없음 — 답을 지어냈을 가능성")
    if not truth:
        return False, "정답 키 계산 실패"
    if check == "number":
        want = truth[0]
        try:
            num = float(want)
            cands = {f"{num:,.0f}", f"{num:.0f}", f"{num:g}", f"{num:,.2f}".rstrip("0").rstrip(".")}
        except ValueError:
            cands = {want}
        digits = {norm(c) for c in cands}
        ok = any(d and d in a_norm for d in digits)
        return ok, f"기대값 {want} " + ("포함" if ok else "미포함")
    if check == "text":
        ok = norm(truth[0])[:14] in a_norm
        return ok, f"'{truth[0][:28]}' " + ("포함" if ok else "미포함")
    if check == "text_all":
        found = [t for t in truth if norm(t)[:14] in a_norm]
        return len(found) == len(truth), f"{len(found)}/{len(truth)}개 포함"
    return False, f"알 수 없는 check: {check}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=str(ROOT / "data/eval/groundtruth.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/truth_results.jsonl"))
    ap.add_argument("--compute", action="store_true", help="정답 키만 계산하고 종료 (서버·LLM 불필요)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="문항 사이 대기(초). CLOVA 테스트앱은 요청 빈도 제한이 낮아 429 가 날 수 있다.")
    a = ap.parse_args()

    items = [json.loads(l) for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    s = get_settings()
    rdb = RDB(s.store_path)
    needs_graph = any(i.get("truth_kind") for i in items)
    kg = KG.build(rdb, Lexicon.load(s.enrichment_dir), None, log=lambda *x: None) if needs_graph else None

    truths = {i["question_id"]: compute_truth(i, rdb, kg) for i in items}

    if a.compute:
        print(f"{'ID':8} {'난이도':4} 정답 키")
        print("-" * 92)
        for i in items:
            t = truths[i["question_id"]]
            shown = "(확인불가로 답해야 정답)" if i.get("check") == "cannot_confirm" else " | ".join(x[:44] for x in t) or "계산 실패"
            print(f"{i['question_id']:8} {i.get('difficulty','-'):4} {shown}")
        print(f"\n{len(items)}문항 · 정답 키 계산 완료. 서버를 띄운 뒤 --compute 없이 실행하면 채점합니다.")
        return 0

    rows, passed = [], 0
    with httpx.Client(timeout=a.timeout) as client, open(a.out, "w", encoding="utf-8") as fh:
        for i in items:
            t0 = time.time()
            try:
                r = client.get(f"{a.base}/answer", params={"question_id": i["question_id"], "question": i["question"]})
                body = r.json()
            except Exception as e:  # noqa: BLE001
                body = {"answer": f"<요청 실패: {e}>", "retrieved_context": "", "think_trace": ""}
            sec = time.time() - t0
            ok, why = score(i, body.get("answer", ""), truths[i["question_id"]], body.get("think_trace", ""))
            passed += ok
            rows.append({**i, "seconds": round(sec, 1), "ok": ok, "why": why, "truth": truths[i["question_id"]],
                         "answer": body.get("answer", ""), "retrieved_context": body.get("retrieved_context", ""),
                         "think_trace": body.get("think_trace", "")})
            fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{'O' if ok else 'X'} {i['question_id']:8} {i.get('difficulty','-'):4} {sec:6.1f}s  {why}")
            if a.delay:
                time.sleep(a.delay)
    n = len(rows)
    slow = sum(1 for r in rows if r["seconds"] >= 300)
    errs = sum(1 for r in rows if is_infra_error(r["answer"], r["think_trace"]))
    graded = n - errs
    print(f"\n정답률 {passed}/{n} ({passed / n * 100:.0f}%) · 평균 {sum(r['seconds'] for r in rows) / n:.1f}초 · 300초 초과 {slow}건")
    if errs:
        pct = f"{passed / graded * 100:.0f}%" if graded else "측정 불가"
        print(f"⚠ 시스템 오류 {errs}건(LLM 429·타임아웃 등)은 모델 실력과 무관합니다. "
              f"실제로 채점된 문항은 {graded}건, 그 기준 정답률은 {pct} 입니다. "
              f"오류를 먼저 없앤 뒤 다시 측정하세요.")
    print(f"상세 결과 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
