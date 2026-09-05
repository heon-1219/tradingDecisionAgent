"""LLM 연결을 한 번의 호출로 점검한다 — 서버·색인 없이 5초면 끝난다.

서비스앱 키로 바꿨을 때, 모델명을 바꿨을 때, 429 가 계속 날 때 가장 먼저 돌릴 것.
확인하는 것: 인증 / 모델 접근 / **function calling 동작 여부** / 응답 지연 / 429 여부.

  python scripts/check_llm.py
  python scripts/check_llm.py --model HCX-005 --repeat 5   # 연속 호출로 요청 빈도 제한 확인
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from finagent.agent.llm import ToolSpec, make_llm, rate_state  # noqa: E402
from finagent.config import get_settings  # noqa: E402

PING_TOOL = ToolSpec(
    "get_bond_count",
    "조건에 맞는 국내 채권 개수를 반환합니다. 개수를 묻는 질문에는 반드시 이 도구를 사용하세요.",
    {
        "type": "object",
        "properties": {"grade": {"type": "string", "description": "신용등급 (예: AA-)"}},
        "required": ["grade"],
    },
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="설정 대신 사용할 모델명")
    ap.add_argument("--repeat", type=int, default=1, help="연속 호출 횟수(요청 빈도 제한 확인용)")
    a = ap.parse_args()

    s = get_settings()
    if a.model:
        s = s.model_copy(update={"llm_model": a.model})
    key = s.clova_api_key or s.openai_api_key or s.anthropic_api_key
    base = s.clova_base_url if s.llm_provider == "clova" else (s.openai_base_url or "(제공자 기본값)")

    print(f"제공자   {s.llm_provider}")
    print(f"모델     {s.llm_model}")
    print(f"엔드포인트 {base}")
    print(f"API 키   {'설정됨 (' + key[:6] + '…' + key[-4:] + ')' if key else '❌ 없음 — .env 를 확인하세요'}")
    print(f"max_tokens {s.llm_max_tokens}" + ("  ⚠ CLOVA function calling 은 1024 이상 필요" if s.llm_max_tokens < 1024 else ""))
    if not key and s.llm_provider != "stub":
        return 2
    print("-" * 68)

    llm = make_llm(s)
    ok = rate_limited = 0
    for i in range(1, a.repeat + 1):
        conv = llm.start(
            "너는 금융상품 분석가다. 개수를 묻는 질문에는 반드시 get_bond_count 도구를 사용한다.",
            [PING_TOOL],
            "신용등급 AA- 이상인 채권은 몇 개야?",
        )
        t0 = time.time()
        turn = await conv.step()
        sec = time.time() - t0

        if turn.stop_reason == "error":
            msg = turn.text
            rate_limited += "429" in msg or "RateLimit" in msg
            print(f"{i}. ❌ {sec:.1f}s  {msg[:150]}")
            if "401" in msg or "Authentication" in msg:
                print("     → 키가 거부되었습니다. 서비스앱 키를 CLOVA_API_KEY 에 'Bearer ' 없이 넣었는지 확인하세요.")
            elif "404" in msg or "NotFound" in msg:
                print(f"     → 모델명 '{s.llm_model}' 또는 엔드포인트를 확인하세요.")
            elif "429" in msg:
                print("     → 요청 빈도 제한. 서비스앱 한도를 확인하거나 호출 간격을 늘리세요.")
            continue

        if turn.tool_calls:
            call = turn.tool_calls[0]
            print(f"{i}. ✅ {sec:.1f}s  function calling 동작 — {call.name}({call.input})")
            ok += 1
        else:
            print(f"{i}. ⚠ {sec:.1f}s  도구를 부르지 않고 텍스트로만 답했습니다: {turn.text[:90]!r}")
            print("     → max_tokens(≥1024), tools 전달, 모델의 function calling 지원 여부를 확인하세요.")

    print("-" * 68)
    toks, reqs, reset = rate_state.get("tokens"), rate_state.get("requests"), rate_state.get("reset_s")
    if toks is not None or reqs is not None:
        print(f"남은 한도  요청 {reqs if reqs is not None else '?'} 건 · 토큰 {toks if toks is not None else '?'} "
              f"· 리셋까지 {reset if reset is not None else '?'}초")
        if toks is not None:
            tier = "서비스앱으로 보입니다" if toks >= 100000 else ("테스트앱으로 보입니다" if toks <= 60000 else "판단 보류")
            print(f"           → TPM 규모상 {tier} (테스트앱 3만~6만 / 서비스앱 12만~30만)")
    else:
        print("남은 한도  헤더가 오지 않았습니다(x-ratelimit-*). 한도는 콘솔에서 확인하세요.")
    if ok == a.repeat:
        print(f"정상: {ok}/{a.repeat} 호출에서 도구 호출 확인. 서버를 띄우고 eval_truth 를 돌려도 됩니다.")
    else:
        print(f"주의: 도구 호출 성공 {ok}/{a.repeat}" + (f", 429 {rate_limited}건" if rate_limited else ""))
        print("이 상태로는 Agent 가 데이터를 조회하지 못해 대부분의 문항을 '확인 불가'로 답합니다.")
    return 0 if ok == a.repeat else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
