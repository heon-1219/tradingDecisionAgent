"""Check the LLM connection with a single call - no server or index needed, done in a few seconds.

Run it first after changing the key or the model, or whenever calls keep failing.
It checks: authentication / model access / **whether function calling works** / latency / 429s.

  python scripts/check_llm.py
  python scripts/check_llm.py --model gemini-3.8-flash --repeat 5   # back-to-back calls to probe rate limits
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from dotenv import dotenv_values  # noqa: E402

from finagent.agent.llm import ToolSpec, make_llm, rate_state  # noqa: E402
from finagent.config import get_settings  # noqa: E402
from finagent.paths import ROOT  # noqa: E402

PING_TOOL = ToolSpec(
    "get_bond_count",
    "Returns the number of domestic bonds that match a condition. Always use this tool for questions that ask how many.",
    {
        "type": "object",
        "properties": {"grade": {"type": "string", "description": "Credit rating (e.g. AA-)"}},
        "required": ["grade"],
    },
)


def key_and_endpoint(s) -> tuple[str | None, str, str]:
    """The configured key, where it came from, and the endpoint for the active provider."""
    if s.llm_provider == "gemini":
        # .env wins when it has a key; the environment only fills in what .env leaves out.
        names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        env_file = ROOT / ".env"
        in_file = {k.upper() for k, v in dotenv_values(env_file).items() if v} if env_file.is_file() else set()
        if any(n in in_file for n in names):
            source = ".env"
        else:
            env = next((v for v in names if os.environ.get(v)), None)
            source = f"environment variable {env} (.env has no key)" if env else ".env"
        return s.gemini_api_key, source, s.gemini_base_url
    if s.llm_provider == "anthropic":
        return s.anthropic_api_key, "ANTHROPIC_API_KEY", "(Anthropic default)"
    if s.llm_provider == "openai":
        return s.openai_api_key, "OPENAI_API_KEY", s.openai_base_url or "(OpenAI default)"
    return None, "-", "-"


def hint(msg: str, model: str) -> str | None:
    low = msg.lower()
    if any(s in low for s in ("valid api key", "api key not valid", "api_key_invalid", "authentication")) or "401" in msg:
        return ("The key was rejected. Create a new one at https://aistudio.google.com/apikey and put it "
                "in .env (a key in .env is used ahead of one exported in your shell).")
    if "403" in msg or "permission" in low:
        return "The key is valid but not allowed to call this API (restricted key or disabled Generative Language API)."
    if "404" in msg or "notfound" in low:
        return f"Check the model name '{model}' (see https://ai.google.dev/gemini-api/docs/models) and the endpoint."
    if "429" in msg or "ratelimit" in low or "resource_exhausted" in low:
        return "Rate limited. Free-tier keys allow only a few requests per minute; wait, or raise REQUEST_MIN_INTERVAL_SEC."
    if "503" in msg or "overloaded" in low or "unavailable" in low:
        return "The model is overloaded right now; try again shortly."
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name to use instead of the configured one")
    ap.add_argument("--repeat", type=int, default=1, help="number of back-to-back calls (to probe rate limits)")
    a = ap.parse_args()

    s = get_settings()
    if a.model:
        s = s.model_copy(update={"llm_model": a.model})
    key, source, endpoint = key_and_endpoint(s)

    print(f"provider   {s.llm_provider}")
    print(f"model      {s.llm_model}")
    print(f"endpoint   {endpoint}")
    print(f"API key    {'set (…' + key[-4:] + ') from ' + source if key else 'MISSING - add it to .env (see .env.example)'}")
    print(f"effort     {s.llm_effort}   max_tokens {s.llm_max_tokens}")
    if not key and s.llm_provider != "stub":
        return 2
    print("-" * 68)

    llm = make_llm(s)
    ok = rate_limited = 0
    for i in range(1, a.repeat + 1):
        conv = llm.start(
            "You are a financial-product analyst. For questions that ask how many, always use the get_bond_count tool.",
            [PING_TOOL],
            "How many bonds are rated AA- or higher?",
        )
        t0 = time.time()
        turn = await conv.step()
        sec = time.time() - t0

        if turn.stop_reason == "error":
            msg = turn.text
            rate_limited += "429" in msg or "RateLimit" in msg
            print(f"{i}. FAIL {sec:.1f}s  {msg[:180]}")
            h = hint(msg, s.llm_model)
            if h:
                print(f"     -> {h}")
            continue

        if turn.tool_calls:
            call = turn.tool_calls[0]
            print(f"{i}. OK   {sec:.1f}s  function calling works - {call.name}({call.input})")
            ok += 1
        else:
            print(f"{i}. WARN {sec:.1f}s  answered in text without calling the tool: {turn.text[:90]!r}")
            print("     -> check that tools are passed and that the model supports function calling.")

    print("-" * 68)
    toks, reqs, reset = rate_state.get("tokens"), rate_state.get("requests"), rate_state.get("reset_s")
    if toks is not None or reqs is not None:
        print(f"quota left  requests {reqs if reqs is not None else '?'} · tokens {toks if toks is not None else '?'} "
              f"· resets in {reset if reset is not None else '?'}s")
    else:
        print("quota left  not reported in response headers (Gemini shows quotas in Google AI Studio).")
    if ok == a.repeat:
        print(f"Healthy: tool call confirmed in {ok}/{a.repeat} calls. You can start the server and run the evaluation.")
    else:
        print(f"Warning: tool calls succeeded in {ok}/{a.repeat}" + (f", 429s: {rate_limited}" if rate_limited else ""))
        print("In this state the agent cannot query the data and will answer most questions with 'cannot confirm'.")
    return 0 if ok == a.repeat else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
