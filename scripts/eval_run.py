"""Run the evaluation question set against a running server and record latency + answers.

Usage:
  python scripts/eval_run.py --base http://127.0.0.1:8000 [--questions data/eval/questions.jsonl]
                             [--out data/eval/results.jsonl] [--ids S-001,U-001] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def classify(answer: str) -> str:
    a = answer or ""
    if "확인할 수 없" in a or "확인되지 않" in a:
        return "cannot_confirm"
    if a.rstrip().endswith(("?", "？")) or "말씀하시" in a:
        return "clarify"
    return "answered"


def main() -> int:
    try:  # Windows consoles default to cp949; keep Korean and symbols printable
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default=str(ROOT / "data/eval/questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/results.jsonl"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        keep = set(args.ids.split(","))
        qs = [q for q in qs if q["question_id"] in keep]
    if args.limit:
        qs = qs[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=args.timeout) as client, open(out, "w", encoding="utf-8") as fh:
        for q in qs:
            t0 = time.time()
            try:
                r = client.get(f"{args.base}/answer", params={"question_id": q["question_id"], "question": q["question"]})
                body = r.json()
                status_code = r.status_code
            except Exception as e:  # noqa: BLE001
                body = {"answer": f"<request failed: {e}>", "retrieved_context": "", "think_trace": ""}
                status_code = 0
            sec = time.time() - t0
            got = classify(body.get("answer", ""))
            ok = (got == q.get("expect")) if q.get("expect") else None
            rec = {**q, "http": status_code, "seconds": round(sec, 1), "got": got, "match": ok, "answer": body.get("answer", ""),
                   "retrieved_context": body.get("retrieved_context", ""), "think_trace": body.get("think_trace", "")}
            rows.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            flag = "✓" if ok else ("✗" if ok is False else "·")
            print(f"{flag} {q['question_id']:6} {q.get('difficulty', '-'):2} {sec:5.1f}s {got:14} | {body.get('answer', '')[:70].replace(chr(10), ' ')}")
    n = len(rows)
    if n:
        matched = sum(1 for r in rows if r["match"])
        avg = sum(r["seconds"] for r in rows) / n
        over = sum(1 for r in rows if r["seconds"] > 60)
        print(f"\n{n} questions · status match {matched}/{n} · avg {avg:.1f}s · >60s: {over} · results → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
