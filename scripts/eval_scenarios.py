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
--resume            skip the ids that already have an `ok` line for the same configuration.
--real-products     use the real product database and Q&A stack (slow to load) instead of the tiny one.
--live-broker       read the real Alpaca PAPER account and market data instead of the fake broker.
                    Reads only: the broker is wrapped so that order writes raise. Its result lines
                    quote the real paper account (balances, positions, prices) in `answer` and
                    `ticket`: think before committing or sharing such a file.
--time-budget SEC   seconds one turn may take (default: the configured turn budget). It sets every
                    turn-budget setting the backend has. A rate-limited key spends much of a turn
                    waiting on 429 retries; a larger budget lets such a turn finish.

Results are one JSON line per scenario, keyed by (id, llm, model, broker, products, turn budget, and
the turn mode when the backend reports one): a run replaces the lines it re-ran with the same key
and keeps all others, so partial runs add up and a `--live-broker` or `--real-products` run never
overwrites or skips fake-world lines. The summary shows the lines of the current configuration only.
A scenario whose turn ended in a model error (a free-tier 429 or 503 shows up as an `error` event)
is recorded with status `llm_error`, left out of the averages and retried by `--resume`; one whose
turn budget ran out before the model wrote anything is `budget_exhausted` and treated the same way
(a turn that was cut at its budget but still got an answer is `ok`, with `"budget_exhausted": true`). The 429s and 5xx that the
model client retried inside a step are counted per line as `provider_retries`. The demo model's
token counts are character-based estimates and are marked `"estimated": true`. A live scenario costs
roughly 4-8 model requests; the Gemini free tier allows about 20 a day and 5 a minute, so run a few
scenarios at a time and `--resume`, e.g. with `--delay 65`.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
_HERE = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# Run as a script, Python puts this directory first on sys.path. Keep it last instead, so that a
# file here (scripts/trace.py) can never shadow a standard-library module.
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or os.curdir)) != _HERE] + [_HERE]

from _eval_metrics import retries_text, scenario_metrics, summary_table, ticket_summary  # noqa: E402
from _eval_support import (  # noqa: E402,F401 - re-exported: the runner's helpers under one name
    CountingLLM,
    ProviderRetryLog,
    ReadOnlyBroker,
    config_key,
    error_kind,
    load_results,
    load_scenarios,
    merge_results,
    same_config,
    write_results,
)
from _fakeworld import DemoLLM, FakeBroker, build_fake_world, build_products, release_sqlite  # noqa: E402
from finagent.agent.llm import make_llm  # noqa: E402
from finagent.app_state import ChatRuntime  # noqa: E402
from finagent.chat.agent import fallback_text  # noqa: E402
from finagent.config import Settings, get_settings  # noqa: E402
from finagent.trading.broker import AlpacaPaper  # noqa: E402
from finagent.trading.models import Mandate  # noqa: E402

log = logging.getLogger("eval_scenarios")

DEFAULT_SCENARIOS = ROOT / "data" / "eval" / "scenarios.jsonl"
DEFAULT_OUT = {"live": ROOT / "data" / "eval" / "scenario_results.jsonl",
               "demo": ROOT / "data" / "eval" / "scenario_results_demo.jsonl"}
LIVE_REQUESTS_PER_SCENARIO = (4, 8)


def _fallback_prefixes() -> tuple[str, ...]:
    """How the agent's own fallback answers begin (the turn budget ran out and the model gave no
    text): one per language, and per mode on a backend that has modes."""
    texts = [fallback_text(lang, []) for lang in ("en", "ko")]
    try:
        texts += [fallback_text(lang, [], "shallow") for lang in ("en", "ko")]
    except TypeError:  # the older backend: one fallback text
        pass
    return tuple(dict.fromkeys(text.split(".")[0] for text in texts))


_BUDGET_PREFIXES = _fallback_prefixes()
_LLM_KEY_FIELDS = {"gemini": ("gemini_api_key", "GEMINI_API_KEY"), "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
                   "openai": ("openai_api_key", "OPENAI_API_KEY")}
# Non-secret settings that decide a turn's limits; a live run copies those that exist into the fake world.
_LIMIT_FIELDS = ("llm_provider", "gemini_paid", "chat_time_budget_sec", "chat_max_rounds", "shallow_time_budget_sec",
                 "shallow_max_rounds", "ticket_ttl_sec", "llm_timeout_sec")


class UsageError(SystemExit):
    """A request the script cannot run (unknown scenario, missing keys); `main` prints it and returns 2."""


# --- turn limits: tolerant of the backend with and without `chat_limits` ---------------------------------


def _chat_limits_fn():
    """The backend's `chat_limits(settings)` if it has one (`finagent.chat.limits` or `finagent.config`)."""
    for name in ("finagent.chat.limits", "finagent.config"):
        try:
            fn = getattr(importlib.import_module(name), "chat_limits", None)
        except Exception:  # noqa: BLE001 - an absent or broken module means "no such function"
            continue
        if callable(fn):
            return fn
    return None


def effective_limits(settings) -> dict:
    """`{"time_budget_sec", "max_rounds", "mode"}` a turn runs under with these settings: from the
    backend's `chat_limits(settings)` when there is one (an object or a dict; `mode` may be an enum),
    else from `chat_time_budget_sec` / `chat_max_rounds`, with `mode` None."""
    fn = _chat_limits_fn()
    limits = None
    if fn is not None:
        try:
            limits = fn(settings)
        except Exception as exc:  # noqa: BLE001 - e.g. a different signature: fall back to the settings
            log.warning("chat_limits not usable (%s); reading the turn limits from the settings", type(exc).__name__)

    def read(*names: str):
        for name in names:
            value = limits.get(name) if isinstance(limits, dict) else getattr(limits, name, None)
            if value is not None:
                return value
        return None

    budget, rounds, mode = read("time_budget_sec", "time_budget"), read("max_rounds", "rounds"), read("mode")
    if budget is None or rounds is None:  # no function, or a shape this script cannot read
        budget, rounds, mode = settings.chat_time_budget_sec, settings.chat_max_rounds, None
    mode = getattr(mode, "value", mode)
    return {"time_budget_sec": float(budget), "max_rounds": int(rounds), "mode": str(mode) if mode is not None else None}


def budget_overrides(settings_cls, seconds: float) -> dict:
    """`--time-budget` for every turn-budget setting this backend has (`chat_time_budget_sec`,
    `shallow_time_budget_sec`, ...), but not `time_budget_sec`, which is the product Q&A agent's."""
    return {name: float(seconds) for name in getattr(settings_cls, "model_fields", {}) if name.endswith("_time_budget_sec")}


def world_overrides(llm_settings, time_budget: float | None) -> dict:
    """`Settings` overrides for the fake world: the limit-related settings of the live model's
    configuration (so a live run gets the limits the real app would apply; no key is copied), then
    `--time-budget`."""
    over = {name: getattr(llm_settings, name) for name in _LIMIT_FIELDS
            if llm_settings is not None and hasattr(llm_settings, name)}
    if time_budget:
        over.update(budget_overrides(Settings, time_budget))
    return over


# --- the live paper account, reads only -------------------------------------------------------------------


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


async def read_account(world) -> dict | None:
    """Equity and the shares held per symbol, read from the broker before the turns: what the
    account-based expectations compare against. None if the broker cannot be read."""
    try:
        account, positions = await world.broker.account(), await world.broker.positions()
    except Exception as exc:  # noqa: BLE001 - the scenarios still run; those expectations then fail loudly
        log.warning("account not readable (%s); account-based expectations cannot be checked", type(exc).__name__)
        return None
    return {"equity": str(account.equity), "positions": {s: str(p.qty_available) for s, p in positions.items()}}


# --- one scenario -----------------------------------------------------------------------------------------


async def run_scenario(world, counter: CountingLLM, scenario: dict, labels: dict,
                       retry_log: ProviderRetryLog | None = None, account: dict | None = None) -> dict:
    """One fresh session, the optional mandate saved first, one turn; returns the result line.
    `labels` is the run's configuration (llm, provider, model, broker, products, the expected turn
    limits); what the `finish` event reports about the turn's mode and limits replaces the expectation."""
    base = {"id": scenario["id"], "category": scenario["category"], "lang": scenario["lang"],
            **{k: v for k, v in labels.items() if v is not None},
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
    finish = next((e for e in reversed(events) if e.get("type") == "finish"), {})
    if finish.get("mode") is not None:
        row["mode"] = str(finish["mode"])
    if finish.get("max_rounds") is not None:
        row["max_rounds"] = int(finish["max_rounds"])
    if finish.get("time_budget_sec") is not None:
        row["time_budget_sec"] = float(finish["time_budget_sec"])
    if any(e.get("type") == "error" for e in events):
        row.update(status="llm_error", error_kind=errors[-1] if errors else "error_event")
    elif answer.startswith(_BUDGET_PREFIXES):  # the agent's fallback: the model never answered
        row.update(status="budget_exhausted", error_kind=errors[-1] if errors else "turn_budget")
    elif finish.get("budget_exhausted") is True:  # nudged at the budget, but the model wrote an answer: scored
        row["budget_exhausted"] = True
    if errors:
        row["llm_errors"] = errors
    if retries:
        row["provider_retries"] = retries
    proposals = [e["ticket"] for e in events if e.get("type") == "data-trade-proposal"]
    metrics = scenario_metrics(scenario, events, answer, llm_calls=counter.calls - calls_before, account=account)
    if labels.get("provider") == "demo":  # DemoLLM counts characters, not tokens
        metrics["usage"] = {"estimated": True, **metrics["usage"]}
    row["metrics"] = metrics
    row["answer"] = answer
    row["ticket"] = ticket_summary(proposals[-1] if proposals else None)
    return row


def _progress(index: int, total: int, row: dict) -> str:
    m = row.get("metrics")
    if not m:
        return f"[{index}/{total}] {row['id']}: {row['status']} ({row.get('note', '')})"
    verdict = ("PASS" if m["expect_pass"] else "FAIL") + (" (ended on its budget)" if row.get("budget_exhausted") else "")
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
    rows: list[dict] = field(default_factory=list)  # every line in `out` for this configuration
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
    ap.add_argument("--resume", action="store_true", help="skip ids already done for the same configuration")
    ap.add_argument("--real-products", action="store_true", help="load the real product database (slow)")
    ap.add_argument("--live-broker", action="store_true",
                    help="read the real paper account (never places orders); its result lines quote that account")
    ap.add_argument("--time-budget", type=float, default=None, metavar="SEC",
                    help="seconds one turn may take (default: the configured turn budget)")
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
    order = [s["id"] for s in scenarios]
    selected = _select(scenarios, args.scenario)
    inner, provider, model, llm_settings = _resolve_llm(args)
    broker_settings, overrides = None, {}
    if args.live_broker:
        broker_settings = get_settings()
        if not (broker_settings.alpaca_api_key_id and broker_settings.alpaca_api_secret_key):
            raise UsageError("--live-broker needs ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys) in .env. "
                             "Nothing was run.")
        if args.time_budget:
            broker_settings = broker_settings.model_copy(update=budget_overrides(type(broker_settings), args.time_budget))
        limits = effective_limits(broker_settings)
    else:
        overrides = world_overrides(llm_settings, args.time_budget)
        limits = effective_limits(Settings(_env_file=None, **{"llm_provider": "stub", **overrides}))
    labels = {"llm": args.llm, "provider": provider, "model": model,
              "broker": "live" if args.live_broker else "fake", "products": "real" if args.real_products else "fake",
              **limits}
    out = Path(args.out)
    existing = load_results(out)
    done = {r["id"] for r in existing if args.resume and r.get("status") == "ok" and same_config(r, labels)}
    report = Report(out=out, skipped=[s["id"] for s in selected if s["id"] in done])
    todo = [s for s in selected if s["id"] not in done]
    if args.limit > 0:
        todo = todo[: args.limit]
    config = (f"{args.llm}/{model}, broker {labels['broker']}, products {labels['products']}, "
              f"turn budget {limits['time_budget_sec']:g}s, {limits['max_rounds']} rounds"
              + (f", mode {limits['mode']}" if limits["mode"] else ""))
    if report.skipped:
        print(f"Resume: {len(report.skipped)} scenario(s) already done for this configuration: {', '.join(report.skipped)}")
    if args.llm == "live" and todo:
        low, high = (n * len(todo) for n in LIVE_REQUESTS_PER_SCENARIO)
        print(f"Live run: {len(todo)} scenario(s) with {provider}/{model}, {args.delay:g}s apart.")
        print(f"Estimated model requests: {low}-{high} ({LIVE_REQUESTS_PER_SCENARIO[0]}-{LIVE_REQUESTS_PER_SCENARIO[1]} "
              "per scenario; retries after a 429 and --real-products add more). The Gemini free tier allows about "
              "20 a day and 5 a minute: run a few at a time and continue with --resume. If turns end as "
              "budget_exhausted after 429 retries, try --delay 65 and a larger --time-budget.")

    new_rows: list[dict] = []
    if todo:
        counter = CountingLLM(inner)
        retry_log = ProviderRetryLog()
        llm_logger = logging.getLogger("finagent.llm")
        llm_logger.addHandler(retry_log)
        tmp_dir = Path(tempfile.mkdtemp(prefix="finagent-eval-"))
        try:
            if args.live_broker:
                world = build_live_broker_world(llm=counter, tmp_dir=tmp_dir, settings=broker_settings,
                                                real_products=args.real_products)
            else:
                pages = {url: text for s in scenarios for url, text in (s.get("web_pages") or {}).items()}
                world = build_fake_world(llm=counter, tmp_dir=tmp_dir, real_products=args.real_products,
                                         web_pages=pages, settings_over=overrides)
            try:
                account = await read_account(world)
                for index, scenario in enumerate(todo, 1):
                    if index > 1 and args.delay > 0:
                        await asyncio.sleep(args.delay)
                    row = await run_scenario(world, counter, scenario, labels, retry_log, account)
                    new_rows.append(row)
                    write_results(out, merge_results(existing, new_rows, order))
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
    # the mode a turn reported is the configuration's mode from here on (the old backend reports none)
    shown = {**labels, "mode": next((r["mode"] for r in new_rows if r.get("mode")), labels["mode"])}
    report.rows = [r for r in merge_results(existing, new_rows, order) if same_config(r, shown)]
    report.table = summary_table(report.rows)
    print(f"\nResults for {config} in {out} ({len(report.ran)} run now, {len(report.rows)} line(s) in all):")
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
