"""한 문항이 답까지 가는 과정을 단계별로 전부 펼쳐 보여준다.

`think_trace` 는 채점 서버에 보내는 요약본이라 도구에 넘긴 인자와 도구가 돌려준 원문이 빠져 있다.
이 스크립트는 그 사이를 다 열어서, 어느 단계에서 무슨 값이 나왔고 그게 무슨 뜻인지 확인할 수 있게 한다.

  python scripts/trace.py "삼성전자를 보유한 국내 ETF 중 순자산이 가장 큰 상품 3개를 알려줘"
  python scripts/trace.py --id G-015 --full        # 도구 출력을 자르지 않고 전부
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
    print(f"\n{BAR}\n[단계 {n}] {title}\n{BAR}")


def clip(s: str, limit: int | None) -> str:
    s = str(s)
    if limit is None or len(s) <= limit:
        return s
    return s[:limit] + f"\n… (총 {len(s):,}자 중 {limit:,}자만 표시 — 전부 보려면 --full)"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="질문 (생략하면 --id 로 평가셋에서 가져옴)")
    ap.add_argument("--id", default=None, help="groundtruth.jsonl 의 question_id")
    ap.add_argument("--full", action="store_true", help="도구 출력을 자르지 않는다")
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
        print("질문을 주거나 --id 를 지정하세요."); return 2
    limit = None if a.full else 1400

    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    print(f"질문   {question}")
    print(f"모델   {s.llm_provider}/{s.llm_model} · 예산 {s.time_budget_sec:.0f}초 · 최대 {s.max_tool_rounds}라운드")
    if truth:
        print(f"정답키 난이도={truth.get('difficulty')} · {truth.get('check')} · {truth.get('note') or ''}")

    t0 = time.time()
    st = AppState.load(s, llm=make_llm(s))
    ndocs = getattr(st.index, "size", None) or len(getattr(st.index, "docs", []) or [])
    print(f"적재   {time.time() - t0:.1f}초 · 어휘색인 {ndocs:,}문서 · 벡터 {len(st.vectors) if st.vectors else 0:,}문서 · 그래프 {len(st.kg.graph):,}트리플")

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

    head(1, "사전 분석 — LLM 호출 전에 결정적으로 계산하는 것")
    pre = st.orchestrator.pre_analyze(question)
    if pre is None:  # 이름이 다르면 실제 실행 결과의 trace 로 대체
        print("(사전분석 함수를 직접 부를 수 없어 실행 후 trace 로 확인합니다)")
    else:
        print(f"상품명 후보 : {pre.mentions or '없음'}")
        print(f"미해결 상품 : {[m for m, _ in pre.unresolved] or '없음'}")
        print(f"온톨로지 위반: {pre.invalid or '없음'}")
        print(f"\n프롬프트에 주입되는 블록:\n{SUB}\n{clip(pre.text, limit)}\n{SUB}")

    head(2, "에이전트 실행 — LLM 이 도구를 고르고 부르는 구간")
    t1 = time.time()
    resp = await st.orchestrator.answer(qid, question)
    elapsed = time.time() - t1

    for i, c in enumerate(calls, 1):
        ev = c["evidence"]
        print(f"\n{SUB}\n도구 {i}. {c['name']}  ({c['sec']:.2f}초){'  ← 오류' if c['error'] else ''}")
        print(f"인자: {json.dumps(c['args'], ensure_ascii=False)}")
        if ev is not None:
            print(f"근거: 종류={ev.kind} 테이블={ev.korean_name or ev.table or '-'} 기준일={ev.base_date or '-'} 행수={ev.row_count}")
        print(f"돌려준 값:\n{clip(c['content'], limit)}")
    if not calls:
        print("(도구를 한 번도 부르지 않았습니다)")

    head(3, "최종 응답 — 채점 서버가 받는 5개 필드")
    m = re.search(r"status=(\w+)", resp.think_trace)
    print(f"status            : {m.group(1) if m else '(trace 에 없음)'}   ← 응답 5필드에는 없고 내부 상태입니다")
    print(f"소요               : {elapsed:.1f}초")
    print(f"\nanswer:\n{SUB}\n{resp.answer}\n{SUB}")
    print(f"\nthink_trace:\n{SUB}\n{resp.think_trace}\n{SUB}")
    print(f"\nretrieved_context:\n{SUB}\n{clip(resp.retrieved_context, limit)}\n{SUB}")

    if truth:
        head(4, "정답 대조")
        print(f"기대: {truth.get('expect')}")
        print(f"판정 기준: {truth.get('check')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
