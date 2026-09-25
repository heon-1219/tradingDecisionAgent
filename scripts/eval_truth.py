"""Score answers against a computed answer key.

The organizers never published an evaluation question set or answers ("the sample questions (reference
query set) are not included in this material"), so there is nothing official to grade against.  This builds the next best
thing: questions whose correct answer is *derivable from the delivered data itself* — a count, a name,
a top-N list — computed by SQL or by a graph lookup at scoring time.  That gives a real accuracy
number instead of a subjective read of the answers.

  python scripts/eval_truth.py --compute                      # compute and print the answer key only (no LLM needed)
  python scripts/eval_truth.py --base http://127.0.0.1:8000   # ask the server and grade the answers

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

# Adding "없습니다" ("there is none") as-is would count normal answers such as "값이 없는 종목은 제외했습니다"
# ("items with no value were excluded") as cannot-confirm.  Leaving it out instead grades an answer that correctly
# denies existence, like "Kimi 관련 투자상품은 없습니다" ("there are no Kimi-related products"), as wrong
# (G-018 was actually graded that way), so a negation counts only when its subject is a product or the data.
DECLINE = ("확인할 수 없", "확인되지 않", "제공되지 않", "존재하지 않", "찾을 수 없", "데이터에 없")
DECLINE_RX = re.compile(r"(상품|종목|펀드|ETF|ETN|채권|데이터|정보|결과)[^.\n]{0,12}?없(습니다|음|어요)")
CLARIFY = ("혹시", "말씀하시", "찾으시는", "다시 확인")
# Fallback answers produced by a system failure.  Their wording starts with '확인할 수 없습니다' ("cannot be
# confirmed"), so left alone every cannot-confirm item would look like a pass — when the model never even got to judge.
ERROR_MARKERS = ("시스템 처리 중 문제가 발생", "<요청 실패", "내부 오류")


def is_infra_error(answer: str, trace: str = "") -> bool:
    """Did the run fail before the model could answer?

    Judged on the answer, not the trace.  An LLM error in a later round does not undo an answer the
    model already produced: G-021 replied "KODEX AI로봇 … 해당 상품은 존재하지 않습니다" — correct — and
    was thrown out as an infrastructure failure because a subsequent round hit a connection error.
    A run that truly failed carries the server's fallback wording, which ERROR_MARKERS matches.
    """
    return any(m in (answer or "") for m in ERROR_MARKERS)


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
        return False, "system error — the model did not answer (cannot be graded)"
    a_norm = norm(answer)
    if check == "cannot_confirm":
        declined = any(d in answer for d in DECLINE) or bool(DECLINE_RX.search(answer))
        asked_back = any(c in answer for c in CLARIFY) and "?" in answer
        ok = declined or asked_back
        return ok, "explicit cannot-confirm" if declined else ("asked back" if asked_back else "neither — the answer may be made up")
    if not truth:
        return False, "answer key computation failed"
    if check == "number":
        want = truth[0]
        try:
            num = float(want)
            # Offering a 'rounded integer' as a candidate for a decimal expected value breaks the grading.
            # G-008's answer is 0.65, but f"{0.65:,.0f}" is "1", so every answer containing the digit 1 even
            # once passed regardless of content (a wrong answer like "총보수는 0.12%입니다" also counted as correct
            # because it contains a '1').  Integer-form candidates are made only when the expected value really
            # is an integer; for decimals only notations that preserve the value are used.  (Always adding ".2f"
            # would turn 0.001 into "0" — the same trap.)
            cands = {f"{num:g}"}
            if num.is_integer():
                cands |= {f"{num:,.0f}", f"{num:.0f}"}
            else:
                two = f"{num:,.2f}"
                if float(two.replace(",", "")) == num:
                    cands.add(two)
        except ValueError:
            cands = {want}
        digits = {norm(c) for c in cands}
        ok = any(d and d in a_norm for d in digits)
        return ok, f"expected {want} " + ("found" if ok else "not found")
    if check == "text":
        ok = norm(truth[0])[:14] in a_norm
        return ok, f"'{truth[0][:28]}' " + ("found" if ok else "not found")
    if check == "text_all":
        found = [t for t in truth if norm(t)[:14] in a_norm]
        return len(found) == len(truth), f"{len(found)}/{len(truth)} found"
    return False, f"unknown check: {check}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=str(ROOT / "data/eval/groundtruth.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/truth_results.jsonl"))
    ap.add_argument("--compute", action="store_true", help="compute the answer key and exit (no server or LLM needed)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between questions. Free-tier API keys have low rate limits and can return 429.")
    a = ap.parse_args()

    items = [json.loads(l) for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    s = get_settings()
    rdb = RDB(s.store_path)
    needs_graph = any(i.get("truth_kind") for i in items)
    kg = KG.build(rdb, Lexicon.load(s.enrichment_dir), None, log=lambda *x: None) if needs_graph else None

    truths = {i["question_id"]: compute_truth(i, rdb, kg) for i in items}

    if a.compute:
        print(f"{'ID':8} {'lvl':4} answer key")
        print("-" * 92)
        for i in items:
            t = truths[i["question_id"]]
            shown = "(correct only if it answers cannot-confirm)" if i.get("check") == "cannot_confirm" else " | ".join(x[:44] for x in t) or "computation failed"
            print(f"{i['question_id']:8} {i.get('difficulty','-'):4} {shown}")
        print(f"\n{len(items)} questions · answer key computed. Start the server and run without --compute to grade.")
        return 0

    rows, passed = [], 0
    with httpx.Client(timeout=a.timeout) as client, open(a.out, "w", encoding="utf-8") as fh:
        for i in items:
            t0 = time.time()
            try:
                r = client.get(f"{a.base}/answer", params={"question_id": i["question_id"], "question": i["question"]})
                body = r.json()
            except Exception as e:  # noqa: BLE001
                # "<요청 실패" ("request failed") stays Korean: ERROR_MARKERS matches on it
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
    print(f"\naccuracy {passed}/{n} ({passed / n * 100:.0f}%) · avg {sum(r['seconds'] for r in rows) / n:.1f}s · over 300s: {slow}")
    if errs:
        pct = f"{passed / graded * 100:.0f}%" if graded else "not measurable"
        print(f"⚠ {errs} system errors (LLM 429, timeouts, etc.) have nothing to do with the model's ability. "
              f"{graded} questions were actually graded; accuracy on those is {pct}. "
              f"Get rid of the errors first, then measure again.")
    print(f"detailed results → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
