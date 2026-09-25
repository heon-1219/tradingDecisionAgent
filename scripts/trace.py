"""Lay out, step by step, everything one question goes through on its way to an answer.

`think_trace` is the summary sent to the grading server, so it leaves out the arguments passed to tools and
the raw text the tools returned.  This script opens up everything in between, so you can see which value
came out at which step and what it means.

  python scripts/trace.py "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘"
  python scripts/trace.py --id G-015 --full        # show tool output in full, untruncated
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from finagent.agent.llm import make_llm  # noqa: E402
from finagent.app_state import AppState  # noqa: E402
from finagent.config import get_settings  # noqa: E402

BAR = "=" * 96
SUB = "-" * 96


def head(n: int, title: str) -> None:
    print(f"\n{BAR}\n[step {n}] {title}\n{BAR}")


def clip(s: str, limit: int | None) -> str:
    s = str(s)
    if limit is None or len(s) <= limit:
        return s
    return s[:limit] + f"\n… (showing {limit:,} of {len(s):,} chars — use --full to see everything)"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="the question (if omitted, taken from the eval set via --id)")
    ap.add_argument("--id", default=None, help="question_id in groundtruth.jsonl")
    ap.add_argument("--full", action="store_true", help="do not truncate tool output")
    a = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    question, qid, truth = a.question, a.id or "T-001", None
    if a.id:
        for line in (root / "data/eval/groundtruth.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["question_id"] == a.id:
                question, truth = d["question"], d
                break
    if not question:
        print("Give a question or pass --id."); return 2
    limit = None if a.full else 1400

    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    print(f"question  {question}")
    print(f"model     {s.llm_provider}/{s.llm_model} · budget {s.time_budget_sec:.0f}s · max {s.max_tool_rounds} rounds")
    if truth:
        print(f"truth key difficulty={truth.get('difficulty')} · {truth.get('check')} · {truth.get('note') or ''}")

    t0 = time.time()
    st = AppState.load(s, llm=make_llm(s))
    ndocs = getattr(st.index, "size", None) or len(getattr(st.index, "docs", []) or [])
    print(f"loaded    {time.time() - t0:.1f}s · lexical index {ndocs:,} docs · vector {len(st.vectors) if st.vectors else 0:,} docs · graph {len(st.kg.graph):,} triples")

    # --- wrap the tool layer so every call and its full result is visible ------------------------
    calls: list[dict] = []
    orig = st.tools.execute

    async def spy(name, args):
        c0 = time.time()
        res = await orig(name, args)
        calls.append({
            "name": name, "args": args, "sec": time.time() - c0,
            "error": res.is_error, "content": res.content,
            "evidence": res.evidence,
        })
        return res

    st.tools.execute = spy  # type: ignore[method-assign]

    head(1, "Pre-analysis — what is computed deterministically before any LLM call")
    pre = st.orchestrator.pre_analyze(question)
    if pre is None:  # if the name differs, fall back to the trace of the actual run
        print("(the pre-analysis function cannot be called directly; check the trace after the run instead)")
    else:
        print(f"name candidates    : {pre.mentions or 'none'}")
        print(f"unresolved products: {[m for m, _ in pre.unresolved] or 'none'}")
        print(f"ontology violations: {pre.invalid or 'none'}")
        print(f"\nblock injected into the prompt:\n{SUB}\n{clip(pre.text, limit)}\n{SUB}")

    head(2, "Agent run — the stretch where the LLM picks and calls tools")
    t1 = time.time()
    resp = await st.orchestrator.answer(qid, question)
    elapsed = time.time() - t1

    for i, c in enumerate(calls, 1):
        ev = c["evidence"]
        print(f"\n{SUB}\ntool {i}. {c['name']}  ({c['sec']:.2f}s){'  ← error' if c['error'] else ''}")
        print(f"args: {json.dumps(c['args'], ensure_ascii=False)}")
        if ev is not None:
            print(f"evidence: kind={ev.kind} table={ev.korean_name or ev.table or '-'} base_date={ev.base_date or '-'} rows={ev.row_count}")
        print(f"returned:\n{clip(c['content'], limit)}")
    if not calls:
        print("(no tool was called)")

    head(3, "Final response — the 5 fields the grading server receives")
    m = re.search(r"status=(\w+)", resp.think_trace)
    print(f"status            : {m.group(1) if m else '(not in trace)'}   ← internal state, not one of the 5 response fields")
    print(f"elapsed           : {elapsed:.1f}s")
    print(f"\nanswer:\n{SUB}\n{resp.answer}\n{SUB}")
    print(f"\nthink_trace:\n{SUB}\n{resp.think_trace}\n{SUB}")
    print(f"\nretrieved_context:\n{SUB}\n{clip(resp.retrieved_context, limit)}\n{SUB}")

    if truth:
        head(4, "Check against the answer key")
        print(f"expected : {truth.get('expect')}")
        print(f"criterion: {truth.get('check')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
