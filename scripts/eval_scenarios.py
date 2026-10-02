"""Scenario evaluation for the chat/trading agent: run trading directions, measure the proposals.

Each scenario in `data/eval/scenarios.jsonl` is one user message (plus an optional mandate saved
first, and optional web pages for the fake HTTP). The script drives the real `ChatAgent` for one
turn per scenario in a fresh session, collects the events, and scores them (see `_eval_metrics.py`).
It only ever asks for PROPOSALS: nothing here confirms a ticket or places, cancels or suspends
anything, and the run fails if the fake broker saw a submit.

  python scripts/eval_scenarios.py                       # scripted DemoLLM over the fake world, offline
  python scripts/eval_scenarios.py --llm live --scenario etf_core --scenario options_out_of_scope --delay 20
  python scripts/eval_scenarios.py --llm live --resume --limit 3     # the next three that are not done yet

--llm demo|live     demo (default) is the deterministic DemoLLM: it checks the pipeline, not a model.
                    live is the configured model, `make_llm(get_settings())`.
--scenario ID       run only these (repeatable).   --limit N   run at most N.
--delay SEC         pause between scenarios (default 15 for live, 0 for demo).
--out PATH          results file (default data/eval/scenario_results.jsonl for live,
                    data/eval/scenario_results_demo.jsonl for demo).
--resume            skip the ids that already have an `ok` line for the same llm and model.
--real-products     use the real product database and Q&A stack (slow to load) instead of the tiny one.
--live-broker       read the real Alpaca PAPER account and market data instead of the fake broker.
                    Reads only: the broker is wrapped so that order writes raise.
--time-budget SEC   seconds one turn may take (default: the configured CHAT_TIME_BUDGET_SEC, 90).
                    A rate-limited key spends much of a turn waiting on 429 retries; a larger budget
                    lets such a turn finish. The value in effect is recorded on every line.

Results are one JSON line per scenario, keyed by (id, llm, model): a run replaces the lines it
re-ran and keeps the others, so partial runs add up. A scenario whose turn ended in a model error
(a free-tier 429 or 503 shows up as an `error` event) is recorded with status `llm_error`, left out
of the averages and retried by `--resume`; one that ran out of the turn budget is `budget_exhausted`
and treated the same way. The 429s and 5xx that the model client retried inside a step are counted
per line as `provider_retries`. A live scenario costs roughly 4-8 model requests; the Gemini free
tier allows about 20 a day and 5 a minute, so run a few scenarios at a time and `--resume`. On such
a key a turn with more than 5 model steps has to wait out 429s: give it room with `--delay 65
--time-budget 240`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", pathlib.Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _eval_metrics import retries_text, scenario_metrics, summary_table, ticket_summary  # noqa: E402
from _fakeworld import DemoLLM, FakeBroker, build_fake_world, build_products, release_sqlite  # noqa: E402
from finagent.agent.llm import make_llm  # noqa: E402
from finagent.app_state import ChatRuntime  # noqa: E402
from finagent.chat.agent import fallback_text  # noqa: E402
from finagent.config import get_settings  # noqa: E402
from finagent.trading.broker import AlpacaPaper  # noqa: E402
from finagent.trading.models import Mandate  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "data" / "eval" / "scenarios.jsonl"
DEFAULT_OUT = {"live": ROOT / "data" / "eval" / "scenario_results.jsonl",
               "demo": ROOT / "data" / "eval" / "scenario_results_demo.jsonl"}
LIVE_REQUESTS_PER_SCENARIO = (4, 8)
_SCENARIO_KEYS = {"id", "lang", "message", "category", "expect"}
# The agent's fallback answer when the turn budget ran out and the model gave no text.
_BUDGET_PREFIXES = tuple(fallback_text(lang, []).split(".")[0] for lang in ("en", "ko"))
_ERROR_KINDS = {
    "RateLimitError": "rate_limited_429", "InternalServerError": "server_error_5xx", "APITimeoutError": "timeout",
    "APIConnectionError": "connection", "AuthenticationError": "auth", "PermissionDeniedError": "auth",
    "NotFoundError": "model_not_found", "BadRequestError": "bad_request", "GEMINI_API_KEY": "missing_key",
}
_LLM_KEY_FIELDS = {"gemini": ("gemini_api_key", "GEMINI_API_KEY"), "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
                   "openai": ("openai_api_key", "OPENAI_API_KEY")}


class UsageError(SystemExit):
    """A request the script cannot run (unknown scenario, missing keys); `main` prints it and returns 2."""


# --- counting model calls ---------------------------------------------------------------------------------


def error_kind(text: str) -> str:
    """A short label for a failed model step. Only the exception class is read from the step's
    error text; the provider's message (which can quote request details) is never kept."""
    match = re.match(r"LLM call error: (\w+)", text or "")
    return _ERROR_KINDS.get(match[1] if match else "", "error_turn")


class _CountingConversation:
    def __init__(self, owner: "CountingLLM", inner):
        self._owner, self._inner = owner, inner

    async def step(self):
        self._owner.calls += 1
        try:
            turn = await self._inner.step()
        except asyncio.CancelledError:
            self._owner.errors.append("cancelled_at_time_budget")
            raise
        except Exception as exc:  # noqa: BLE001 - recorded by type, then left to the agent
            self._owner.errors.append(type(exc).__name__)
            raise
        if turn.stop_reason == "error":
            self._owner.errors.append(error_kind(turn.text))
        return turn

    def add_tool_results(self, results) -> None:
        self._inner.add_tool_results(results)

    def add_user(self, text: str) -> None:
        self._inner.add_user(text)

    def add_assistant(self, text: str) -> None:
        self._inner.add_assistant(text)

    def transcript(self):
        return self._inner.transcript()


class CountingLLM:
    """A thin `LLMClient` wrapper: `calls` counts every `conv.step`, `errors` lists the failed ones."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "llm")
        self.model = getattr(inner, "model", None)
        self.calls = 0
        self.errors: list[str] = []

    def start(self, system, tools, user_text):
        return _CountingConversation(self, self.inner.start(system, tools, user_text))


class ProviderRetryLog(logging.Handler):
    """Counts the model client's retry warnings (logger `finagent.llm`): a 429 or a 5xx that the
    client retried inside one step never reaches the event stream, but it is why a turn ran out of
    its budget. Only the kind is counted; the messages are not kept."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()  # also right when a record factory has pre-formatted it
        except Exception:  # noqa: BLE001 - a malformed record is not a retry
            return
        if "retrying" not in message:
            return
        if message.startswith("429"):
            kind = "rate_limited_429"
        elif message.startswith("InternalServerError"):
            kind = "server_error_5xx"
        else:
            kind = "timeout_or_connection"
        self.counts[kind] = self.counts.get(kind, 0) + 1


# --- the live paper account, reads only -------------------------------------------------------------------


class ReadOnlyBroker:
    """Wraps a broker so that only its reads go through. Order writes raise `PermissionError`, and
    any other attribute does not exist: this script proposes, it never trades."""

    READS = frozenset({"account", "positions", "orders", "asset", "clock", "get_order", "get_order_by_client_id",
                       "aclose"})
    WRITES = frozenset({"submit_order", "cancel_order", "set_suspend_trade"})

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name: str):
        if name in ReadOnlyBroker.READS:
            return getattr(self._inner, name)
        if name in ReadOnlyBroker.WRITES:
            async def blocked(*args, **kwargs):
                raise PermissionError(f"eval_scenarios is read-only: {name} is blocked (proposals only, never orders)")
            return blocked
        raise AttributeError(name)


@dataclass
class LiveBrokerWorld:
    """The production composition over the real paper account, read-only, with scratch state."""

    runtime: ChatRuntime
    registry: Any = None

    @property
    def broker(self):
        return self.runtime.broker

    @property
    def sessions(self):
        return self.runtime.sessions

    @property
    def agent(self):
        return self.runtime.agent

    async def close(self) -> None:
        await self.runtime.shutdown()
        release_sqlite(self.runtime.db, getattr(self.registry, "rdb", None))


def build_live_broker_world(*, llm, tmp_dir: Path, settings, real_products: bool) -> LiveBrokerWorld:
    """A `ChatRuntime` on the user's settings (paper keys, research keys) whose broker is wrapped in
    `ReadOnlyBroker`. The app DB and the env file live under `tmp_dir`, never in `data/store/`."""
    scratch = settings.model_copy(update={"app_db_path": tmp_dir / "app.db"})
    env_path = tmp_dir / ".env"
    env_path.write_text("# eval_scenarios scratch file\n", encoding="utf-8")
    registry, orchestrator = build_products(tmp_dir, llm=llm, real_products=real_products)
    runtime = ChatRuntime(
        scratch, llm=llm, registry=registry, orchestrator=orchestrator, env_path=env_path,
        broker_factory=lambda key_id, secret: ReadOnlyBroker(AlpacaPaper(key_id, secret)),
    )
    if not isinstance(runtime.broker, ReadOnlyBroker):
        raise RuntimeError("the live broker must be wrapped read-only")
    return LiveBrokerWorld(runtime=runtime, registry=registry)


# --- files ------------------------------------------------------------------------------------------------


def load_scenarios(path: Path | str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        missing = _SCENARIO_KEYS - set(row)
        if missing:
            raise ValueError(f"scenario {row.get('id', '?')!r} lacks {', '.join(sorted(missing))}")
        if row["id"] in seen:
            raise ValueError(f"duplicate scenario id {row['id']!r}")
        seen.add(row["id"])
    return rows


def load_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict) -> tuple:
    return row.get("id"), row.get("llm"), row.get("model")


def merge_results(existing: list[dict], new: list[dict], order: list[str]) -> list[dict]:
    """`existing` with the lines for the same (id, llm, model) replaced by `new`, in scenario order."""
    fresh = {_key(r) for r in new}
    rows = [r for r in existing if _key(r) not in fresh] + list(new)
    position = {sid: i for i, sid in enumerate(order)}
    return sorted(rows, key=lambda r: (position.get(r.get("id"), len(order)), str(r.get("llm")), str(r.get("model"))))


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    scratch = path.with_name(path.name + ".tmp")
    try:
        scratch.write_text(text, encoding="utf-8", newline="\n")
        os.replace(scratch, path)
    except OSError:  # e.g. a sync client holding the file: write in place instead
        path.write_text(text, encoding="utf-8", newline="\n")
        scratch.unlink(missing_ok=True)


# --- one scenario -----------------------------------------------------------------------------------------


async def run_scenario(world, counter: CountingLLM, scenario: dict, labels: dict,
                       retry_log: ProviderRetryLog | None = None) -> dict:
    """One fresh session, the optional mandate saved first, one turn; returns the result line."""
    base = {"id": scenario["id"], "category": scenario["category"], "lang": scenario["lang"], **labels,
            "time_budget_sec": float(world.runtime.settings.chat_time_budget_sec),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if labels["broker"] == "live" and scenario.get("web_pages"):
        return {**base, "status": "skipped",
                "note": "needs the fake web pages, which --live-broker (real HTTP) does not serve"}
    session = world.sessions.create(title=scenario["id"], lang=scenario["lang"])
    if scenario.get("mandate"):
        world.sessions.save_mandate(session["id"], Mandate.model_validate(scenario["mandate"]))
    events: list[dict] = []
    calls_before, errors_before = counter.calls, len(counter.errors)
    retries_before = dict(retry_log.counts) if retry_log is not None else {}
    answer = await world.agent.run_turn(session["id"], scenario["message"], events.append)
    errors = counter.errors[errors_before:]
    retries_now = retry_log.counts if retry_log is not None else {}
    retries = {kind: count - retries_before.get(kind, 0) for kind, count in retries_now.items()
               if count > retries_before.get(kind, 0)}
    row = dict(base, status="ok")
    if any(e.get("type") == "error" for e in events):
        row.update(status="llm_error", error_kind=errors[-1] if errors else "error_event")
    elif answer.startswith(_BUDGET_PREFIXES):
        row.update(status="budget_exhausted", error_kind=errors[-1] if errors else "turn_budget")
    if errors:
        row["llm_errors"] = errors
    if retries:
        row["provider_retries"] = retries
    proposals = [e["ticket"] for e in events if e.get("type") == "data-trade-proposal"]
    row["metrics"] = scenario_metrics(scenario, events, answer, llm_calls=counter.calls - calls_before)
    row["answer"] = answer
    row["ticket"] = ticket_summary(proposals[-1] if proposals else None)
    return row


def _progress(index: int, total: int, row: dict) -> str:
    m = row.get("metrics")
    if not m:
        return f"[{index}/{total}] {row['id']}: {row['status']} ({row.get('note', '')})"
    verdict = "PASS" if m["expect_pass"] else "FAIL"
    if row["status"] != "ok":
        verdict = row.get("error_kind") or row["status"]
        if retries_text(row):
            verdict += f" (retries: {retries_text(row)})"
    return (f"[{index}/{total}] {row['id']}: {row['status']}, ticket {m['tickets_proposed']}, "
            f"legs a/c/r {m['legs_accepted']}/{m['legs_clamped']}/{m['legs_rejected']}, "
            f"{m['llm_calls']} LLM call(s), {(m['elapsed_ms'] or 0) / 1000:.1f}s, {verdict}")


# --- the run ----------------------------------------------------------------------------------------------


@dataclass
class Report:
    out: Path
    rows: list[dict] = field(default_factory=list)  # every line in `out` for this llm and model
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # done before (--resume)
    broker_submits: int | None = None  # submits the fake broker saw (must be 0); None with --live-broker
    table: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scenario evaluation for the chat/trading agent (proposals only).")
    ap.add_argument("--llm", choices=["demo", "live"], default="demo")
    ap.add_argument("--limit", type=int, default=0, help="run at most N scenarios")
    ap.add_argument("--scenario", action="append", default=[], metavar="ID", help="run only this id (repeatable)")
    ap.add_argument("--delay", type=float, default=None, help="seconds between scenarios (default 15 live, 0 demo)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="skip ids already done for the same llm and model")
    ap.add_argument("--real-products", action="store_true", help="load the real product database (slow)")
    ap.add_argument("--live-broker", action="store_true", help="read the real paper account (never places orders)")
    ap.add_argument("--time-budget", type=float, default=None, metavar="SEC",
                    help="seconds one turn may take (default: the configured CHAT_TIME_BUDGET_SEC, 90)")
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.delay is None:
        args.delay = 15.0 if args.llm == "live" else 0.0
    if args.out is None:
        args.out = DEFAULT_OUT[args.llm]
    return args


def _select(scenarios: list[dict], ids: list[str]) -> list[dict]:
    if not ids:
        return list(scenarios)
    known = {s["id"] for s in scenarios}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise UsageError(f"unknown scenario id(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    return [s for s in scenarios if s["id"] in ids]


def _resolve_llm(args) -> tuple[Any, str, str, Any]:
    """(the model client, provider label, model name, the settings it came from or None)."""
    if args.llm == "demo":
        llm = DemoLLM()
        return llm, "demo", llm.model, None
    settings = get_settings()
    provider = settings.llm_provider
    if provider not in _LLM_KEY_FIELDS:
        raise UsageError(f"--llm live needs a real model, but LLM_PROVIDER is {provider!r}. Nothing was run.")
    key_field, env_name = _LLM_KEY_FIELDS[provider]
    if not getattr(settings, key_field, None):
        raise UsageError(f"--llm live needs {env_name} in .env for the {provider} provider. Nothing was run.")
    return make_llm(settings), provider, settings.llm_model, settings


async def run_async(args) -> Report:
    scenarios = load_scenarios(args.scenarios)
    selected = _select(scenarios, args.scenario)
    inner, provider, model, llm_settings = _resolve_llm(args)
    broker_settings = None
    if args.live_broker:
        broker_settings = get_settings()
        if not (broker_settings.alpaca_api_key_id and broker_settings.alpaca_api_secret_key):
            raise UsageError("--live-broker needs ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys) in .env. "
                             "Nothing was run.")
    out = Path(args.out)
    existing = load_results(out)
    done = {r["id"] for r in existing
            if args.resume and r.get("llm") == args.llm and r.get("model") == model and r.get("status") == "ok"}
    report = Report(out=out, skipped=[s["id"] for s in selected if s["id"] in done])
    todo = [s for s in selected if s["id"] not in done]
    if args.limit > 0:
        todo = todo[: args.limit]
    labels = {"llm": args.llm, "provider": provider, "model": model,
              "broker": "live" if args.live_broker else "fake", "products": "real" if args.real_products else "fake"}
    if report.skipped:
        print(f"Resume: {len(report.skipped)} scenario(s) already done for {args.llm}/{model}: {', '.join(report.skipped)}")
    if args.llm == "live" and todo:
        low, high = (n * len(todo) for n in LIVE_REQUESTS_PER_SCENARIO)
        print(f"Live run: {len(todo)} scenario(s) with {provider}/{model}, {args.delay:g}s apart.")
        print(f"Estimated model requests: {low}-{high} ({LIVE_REQUESTS_PER_SCENARIO[0]}-{LIVE_REQUESTS_PER_SCENARIO[1]} "
              "per scenario; retries after a 429 and --real-products add more). The Gemini free tier allows about "
              "20 a day and 5 a minute: run a few at a time and continue with --resume. If turns end as "
              "budget_exhausted after 429 retries, try --delay 65 --time-budget 240.")

    new_rows: list[dict] = []
    if todo:
        counter = CountingLLM(inner)
        retry_log = ProviderRetryLog()
        llm_logger = logging.getLogger("finagent.llm")
        llm_logger.addHandler(retry_log)
        tmp_dir = Path(tempfile.mkdtemp(prefix="finagent-eval-"))
        try:
            if args.live_broker:
                if args.time_budget:
                    broker_settings = broker_settings.model_copy(update={"chat_time_budget_sec": args.time_budget})
                world = build_live_broker_world(llm=counter, tmp_dir=tmp_dir, settings=broker_settings,
                                                real_products=args.real_products)
            else:
                pages = {url: text for s in scenarios for url, text in (s.get("web_pages") or {}).items()}
                budget = {name: getattr(llm_settings, name) for name in
                          ("chat_time_budget_sec", "chat_max_rounds", "ticket_ttl_sec", "llm_timeout_sec")
                          } if llm_settings is not None else {}
                if args.time_budget:
                    budget["chat_time_budget_sec"] = args.time_budget
                world = build_fake_world(llm=counter, tmp_dir=tmp_dir, real_products=args.real_products,
                                         web_pages=pages, settings_over=budget)
            try:
                for index, scenario in enumerate(todo, 1):
                    if index > 1 and args.delay > 0:
                        await asyncio.sleep(args.delay)
                    row = await run_scenario(world, counter, scenario, labels, retry_log)
                    new_rows.append(row)
                    write_results(out, merge_results(existing, new_rows, [s["id"] for s in scenarios]))
                    print(_progress(index, len(todo), row), flush=True)
            finally:
                if isinstance(world.broker, FakeBroker):
                    report.broker_submits = world.broker.submits
                await world.close()
        finally:
            llm_logger.removeHandler(retry_log)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if report.broker_submits:
            raise RuntimeError(f"eval_scenarios must never submit orders, but the fake broker saw "
                               f"{report.broker_submits} submit(s)")
    else:
        print("Nothing to run.")

    report.ran = [r["id"] for r in new_rows]
    merged = merge_results(existing, new_rows, [s["id"] for s in scenarios])
    report.rows = [r for r in merged if r.get("llm") == args.llm and r.get("model") == model]
    report.table = summary_table(report.rows)
    print(f"\nResults for {args.llm}/{model} in {out} ({len(report.ran)} run now, {len(report.rows)} line(s) in all):")
    print(report.table)
    return report


def run(args) -> Report:
    return asyncio.run(run_async(args))


def main(argv: list[str] | None = None) -> int:
    try:  # Windows consoles default to cp949; keep Korean and symbols printable
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        run(args)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
