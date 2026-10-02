"""Check the Alpaca PAPER connection - read-only by default, a few seconds, no server needed.

  python scripts/check_alpaca.py                     # read-only: account, clock, SPY snapshot, latest news
  python scripts/check_alpaca.py --place-test-order  # ALSO places and cancels one tiny paper order

Default mode never writes anything. It runs four independent steps, each with its latency and a
checkmark or cross: the account (status, equity, buying power, and whether Alpaca says it is a paper
account), the market clock, one IEX snapshot for SPY (bid, ask, last, timestamp) and the latest 3 news
items. A failing step does not stop the others.

Exit codes
  0  every step passed
  1  at least one step failed (or the test order ended uncertain / unconfirmed)
  2  the Alpaca keys are not configured (the output says exactly what to add to .env)
  3  --place-test-order was refused (no confirmation, not paper, ...): nothing was placed

--place-test-order submits a 1-share DAY limit BUY of SPY at 50% of the last price (rounded down to
cents, so it cannot fill), prints its status, cancels it, and polls up to ~10 s for the final status.
  * Confirmation rule: on a terminal you must type PLACE exactly, always - even with --yes-i-am-sure,
    which changes nothing there. If stdin is NOT a terminal the script refuses, unless --yes-i-am-sure
    is passed, which then stands in for the typed word. (Git Bash/mintty reports no terminal for native
    Python: use PowerShell, cmd or winpty instead.) The controlling agent never passes --yes-i-am-sure
    without the user's explicit go-ahead.
  * It refuses unless the broker's host constant is the paper host and the account can trade. If Alpaca
    says the account is not a paper account (its number lacks the "PA" prefix) it refuses; if that is
    unknown it says so and relies on the paper host.
  * If the submit's outcome is uncertain (timeout, 5xx, unreadable reply) it NEVER resubmits: it looks
    the order up by client_order_id, reports, and cancels it if it turns out to be live.
  * Ctrl-C after the submit prints where the possibly open order is, then exits.

Everything printed has control characters stripped, keys are never printed (only "set (...last4)"), and
any key value that shows up inside an error message is replaced by ***.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from decimal import ROUND_DOWN, Decimal
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import httpx  # noqa: E402

import _checkio  # noqa: E402
from _checkio import Outcome, Reporter, strip_controls  # noqa: E402
from finagent.config import get_settings  # noqa: E402
from finagent.research.evidence import EvidenceStore, sanitize_untrusted  # noqa: E402
from finagent.research.http import HttpClient  # noqa: E402
from finagent.store.appdb import AppDB  # noqa: E402
from finagent.trading import broker as broker_mod  # noqa: E402
from finagent.trading.broker import AlpacaPaper, BrokerError, OrderUncertain  # noqa: E402
from finagent.trading.marketdata import AlpacaData  # noqa: E402
from finagent.trading.models import OrderLeg  # noqa: E402

PAPER_KEYS_URL = "https://app.alpaca.markets/paper/dashboard/overview"
PAPER_HOST = "paper-api.alpaca.markets"
CONFIRM_WORD = "PLACE"
SYMBOL = "SPY"
NEWS_COUNT = 3
POLL_TIMEOUT_SEC = 10.0
POLL_INTERVAL_SEC = 0.5
# Statuses after which an order can no longer change (so polling can stop).
TERMINAL_STATUSES = frozenset({"canceled", "cancelled", "expired", "filled", "rejected", "done_for_day", "replaced"})

EXIT_OK, EXIT_STEP_FAILED, EXIT_NO_KEYS, EXIT_REFUSED = 0, 1, 2, 3

_CENT = Decimal("0.01")
_HALF = Decimal("0.5")


# --------------------------------------------------------------------------------- output helpers


def hint_for(exc: BaseException) -> str | None:
    status = getattr(exc, "status", None)
    if status in (401, 403):
        return (f"Alpaca rejected the request. Use PAPER keys (create them at {PAPER_KEYS_URL}); live keys do not "
                "work against the paper host.")
    if status == 429:
        return "Rate limited. Wait a minute and run the check again."
    if status == 0 or isinstance(exc, (httpx.RequestError, TimeoutError, OSError)):
        return "Could not reach Alpaca (network, proxy or firewall?). Nothing was changed."
    return None


def clean(text: Any, limit: int) -> str:
    """Terminal-safe, truncated copy of externally sourced text (headlines and the like)."""
    return strip_controls(sanitize_untrusted(str(text or ""), max_chars=limit))


async def run_step(rep: Reporter, label: str, fn: Callable[[], Awaitable[Outcome]]) -> bool:
    return await _checkio.run_step(rep, label, fn, hint_for)


# --------------------------------------------------------------------------------- read-only steps


async def check_account(broker) -> Outcome:
    acct = await broker.account()
    problems = [name for name in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
                if getattr(acct, name, False)]
    paper = getattr(acct, "paper", None)
    lines = [
        f"status        {acct.status}",
        f"equity        {acct.equity}",
        f"buying power  {acct.buying_power}",
        f"cash          {acct.cash}",
        "paper account  " + ("verified (Alpaca reports a paper account)" if paper is True else
                             "NOT a paper account" if paper is False else
                             "unknown (Alpaca reported no account number)"),
    ]
    if paper is False:
        return Outcome(False, "Alpaca says this is NOT a paper account", lines)
    if problems:
        return Outcome(False, f"account is {acct.status} but flagged: {', '.join(problems)}", lines)
    if str(acct.status).upper() != "ACTIVE":
        return Outcome(False, f"account status is {acct.status}, expected ACTIVE", lines)
    return Outcome(True, str(acct.status), lines)


async def check_clock(broker) -> Outcome:
    clock = await broker.clock()
    lines = [
        f"market        {'OPEN' if clock.is_open else 'CLOSED'}",
        f"now           {clock.timestamp}",
        f"next open     {clock.next_open}",
        f"next close    {clock.next_close}",
    ]
    return Outcome(True, "open" if clock.is_open else "closed", lines)


async def check_snapshot(data) -> Outcome:
    quotes = await data.snapshots([SYMBOL])
    quote = quotes.get(SYMBOL)
    if quote is None:
        return Outcome(False, f"no {SYMBOL} snapshot came back from the IEX feed")

    def show(v: Any) -> str:
        return "n/a" if v is None else str(v)

    lines = [f"{SYMBOL} ({quote.feed})  bid {show(quote.bid)}  ask {show(quote.ask)}  last {show(quote.last)}",
             f"timestamp     {show(quote.as_of)}"]
    return Outcome(True, f"{SYMBOL} last {show(quote.last)}", lines)


async def check_news(data, evidence: EvidenceStore) -> Outcome:
    result = await data.news("alpaca-check", evidence, symbols=None, limit=NEWS_COUNT)
    sources = list(result.sources)[:NEWS_COUNT]
    if not sources:
        return Outcome(True, "0 news items", ["(Alpaca returned no news)"])
    lines = []
    for src in sources:
        when = clean(src.published_at, 25) or "no time"
        who = clean(src.publisher, 20) or "unknown source"
        lines.append(f"{when}  {who}  {clean(src.title, 100)}")
    return Outcome(True, f"{len(sources)} news items", lines)


async def run_readonly(broker, data, rep: Reporter, evidence: EvidenceStore) -> bool:
    """The four read-only steps. Returns True only if every one passed."""
    results = [
        await run_step(rep, "Account", lambda: check_account(broker)),
        await run_step(rep, "Market clock", lambda: check_clock(broker)),
        await run_step(rep, f"{SYMBOL} snapshot (IEX)", lambda: check_snapshot(data)),
        await run_step(rep, f"Latest {NEWS_COUNT} news items", lambda: check_news(data, evidence)),
    ]
    return all(results)


# --------------------------------------------------------------------------------- the test order


def half_price_limit(last: Decimal) -> Decimal:
    """50% of `last`, rounded DOWN to cents. Decimal only: money is never float."""
    if not isinstance(last, Decimal):
        raise TypeError(f"price must be a Decimal, got {type(last).__name__}")
    return (last * _HALF).quantize(_CENT, rounding=ROUND_DOWN)


def new_client_order_id() -> str:
    # [A-Za-z0-9._:-]{1,128} is what AlpacaPaper accepts.
    return f"check-alpaca-{uuid.uuid4().hex}"


def host_refusal() -> str | None:
    """None when the broker module's host constant is the paper host over plain https."""
    parts = urlsplit(str(getattr(broker_mod, "PAPER_BASE", "")))
    if parts.scheme != "https" or parts.hostname != PAPER_HOST or parts.port not in (None, 443):
        return (f"the broker's host constant is {broker_mod.PAPER_BASE!r}, not the paper host "
                f"https://{PAPER_HOST}")
    return None


def account_refusal(account: Any) -> str | None:
    """A reason not to trade on this account, or None. `paper` is Alpaca's own answer (from the account
    number's "PA" prefix, set by `AlpacaPaper.account()`): False refuses, None is handled by the caller
    (it says so and relies on the paper host)."""
    if getattr(account, "paper", None) is False:
        return "Alpaca says this is not a paper account (its number lacks the paper prefix)"
    if getattr(account, "trading_blocked", False) or getattr(account, "account_blocked", False):
        return "the account is blocked from trading"
    if getattr(account, "trade_suspended_by_user", False):
        return "trading is suspended on the account (the kill switch is on)"
    if str(getattr(account, "status", "")).upper() != "ACTIVE":
        return f"the account status is {getattr(account, 'status', None)!r}, not ACTIVE"
    return None


def refuse(rep: Reporter, reason: str, *more: str) -> int:
    rep.say(f"    REFUSED: {reason}")
    for line in more:
        rep.say(f"    {line}")
    rep.say("    Nothing was placed.")
    return EXIT_REFUSED


def report_fill(rep: Reporter, seconds: float, order_id: str) -> None:
    rep.fail(seconds, "the test order FILLED: the paper account now holds 1 share of SPY",
             f"order id {order_id}. Sell it from {PAPER_KEYS_URL} if you do not want it.")


async def cancel_and_confirm(
    broker, rep: Reporter, order_id: str, *, sleep, clock, timeout: float, interval: float
) -> bool:
    """Cancel, then poll `get_order` until the status is final or the budget is spent. True only for a
    confirmed cancel."""
    rep.begin("Cancel the test order and confirm its final status")
    t0 = _checkio.perf()
    cancel_error: BaseException | None = None
    try:
        await broker.cancel_order(order_id)
    except Exception as exc:  # noqa: BLE001 - never crash with a live order outstanding
        cancel_error = exc
        rep.detail(f"cancel request failed: {rep.describe(exc)}")

    status: str | None = None
    poll_error: BaseException | None = None
    deadline = clock() + timeout
    while True:
        try:
            snap = await broker.get_order(order_id)
            status, poll_error = snap.status, None
        except Exception as exc:  # noqa: BLE001
            poll_error = exc
        if status is not None and strip_controls(status).lower() in TERMINAL_STATUSES:
            break
        if clock() >= deadline:
            break
        await sleep(interval)

    seconds = _checkio.perf() - t0
    if poll_error is not None and status is None:
        rep.fail(seconds, f"could not read the order's status: {rep.describe(poll_error)}",
                 f"order id {order_id}: check {PAPER_KEYS_URL} and cancel it by hand if it is still open.")
        return False
    final = strip_controls(status or "").lower()
    if final in {"canceled", "cancelled"}:
        rep.ok(seconds, f"final status: {status}")
        return True
    if final == "filled":
        report_fill(rep, seconds, order_id)
        return False
    if final in TERMINAL_STATUSES:
        rep.fail(seconds, f"the order ended as {status}, not canceled", f"order id {order_id}")
        return False
    reason = f"cancel not confirmed within {timeout:.0f}s; last status {status}"
    if cancel_error is not None:
        reason += f" (cancel request error: {rep.describe(cancel_error, 120)})"
    rep.fail(seconds, reason, f"order id {order_id}: cancel it at {PAPER_KEYS_URL} if it is still open.")
    return False


async def place_test_order(
    broker,
    data,
    rep: Reporter,
    *,
    interactive: bool,
    assume_yes: bool,
    read_line: Callable[[str], str] = input,
    sleep=asyncio.sleep,
    clock=time.monotonic,
    poll_timeout: float = POLL_TIMEOUT_SEC,
    poll_interval: float = POLL_INTERVAL_SEC,
    make_client_order_id: Callable[[], str] = new_client_order_id,
) -> int:
    """The only write path in this script. Returns an exit code (0 ok, 1 failed/uncertain, 3 refused).

    Confirmation: on a terminal (`interactive`) the typed word is always required, `assume_yes` or not;
    without a terminal the order is placed only when `assume_yes` is set."""
    rep.say()
    rep.say(f"== Test order: BUY 1 {SYMBOL}, DAY limit at 50% of the last price (PAPER; cancelled right away) ==")

    if not interactive and not assume_yes:
        return refuse(
            rep, "stdin is not a terminal, so the confirmation word cannot be typed.",
            "In Git Bash (mintty) native Python reports no terminal even though you can type: run this from "
            "PowerShell or cmd,",
            "or prefix the command with winpty (winpty python scripts/check_alpaca.py --place-test-order).",
            "Last resort, only with the user's explicit go-ahead: --yes-i-am-sure replaces the typed word "
            "when there is no terminal.",
        )
    reason = host_refusal()
    if reason:
        return refuse(rep, reason)

    try:
        account = await broker.account()
    except Exception as exc:  # noqa: BLE001
        rep.say(f"    ✗ could not read the account, so no order was placed: {rep.describe(exc)}")
        hint = hint_for(exc)
        if hint:
            rep.say(f"      -> {hint}")
        rep.failed += 1
        return EXIT_STEP_FAILED
    reason = account_refusal(account)
    if reason:
        return refuse(rep, reason)

    try:
        quote = (await data.snapshots([SYMBOL])).get(SYMBOL)
    except Exception as exc:  # noqa: BLE001
        rep.say(f"    ✗ could not read the {SYMBOL} price, so no order was placed: {rep.describe(exc)}")
        rep.failed += 1
        return EXIT_STEP_FAILED
    last = quote.last if quote is not None else None
    if last is None or last <= 0:
        return refuse(rep, f"there is no usable last price for {SYMBOL} (market data returned {last!r})")
    limit = half_price_limit(last)
    if limit <= 0:
        return refuse(rep, f"50% of {last} rounds down to {limit}, which is not a valid limit price")
    leg = OrderLeg(symbol=SYMBOL, side="buy", qty=Decimal(1), type="limit", time_in_force="day", limit_price=limit)

    rep.say(f"    Plan: BUY 1 {SYMBOL}  limit {limit}  (50% of last {last}, rounded down to cents)  DAY")
    sizes = f"equity {account.equity}, buying power {account.buying_power}"
    if getattr(account, "paper", None) is True:
        rep.say(f"    Account: PAPER (verified), {sizes}")
    else:
        rep.say(f"    Account: {sizes}")
        rep.say("    Note: paper status unknown; relying on the paper host")
    if interactive:
        try:
            typed = read_line(f"    Type {CONFIRM_WORD} to submit this paper order (anything else cancels): ")
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if typed != CONFIRM_WORD:
            return refuse(rep, f"{CONFIRM_WORD} was not typed exactly")
    else:
        rep.say("    stdin is not a terminal: --yes-i-am-sure given, so the typed confirmation is skipped")

    cid = make_client_order_id()
    order = None
    try:
        rep.begin(f"Submit the test order (client_order_id {cid})")
        t0 = _checkio.perf()
        try:
            order = await broker.submit_order(leg, cid)
        except OrderUncertain as exc:
            rep.fail(_checkio.perf() - t0, f"submit outcome is UNCERTAIN: {rep.describe(exc)}")
            await reconcile_uncertain(broker, rep, cid, sleep=sleep, clock=clock,
                                      timeout=poll_timeout, interval=poll_interval)
            return EXIT_STEP_FAILED  # an uncertain outcome is never reported as a pass
        except BrokerError as exc:
            rep.fail(_checkio.perf() - t0, f"Alpaca rejected the order: {rep.describe(exc)}", hint_for(exc))
            return EXIT_STEP_FAILED
        rep.detail(f"order id {order.id}")
        rep.ok(_checkio.perf() - t0, f"status: {order.status}  ({order.side} {order.qty} {order.symbol} "
                                      f"{order.type} {order.limit_price} {order.time_in_force})")

        cancelled = await cancel_and_confirm(broker, rep, order.id, sleep=sleep, clock=clock,
                                             timeout=poll_timeout, interval=poll_interval)
        return EXIT_OK if cancelled else EXIT_STEP_FAILED
    except BaseException:  # Ctrl-C, task cancellation, anything: say where an order may still be, then go on
        where = (f"id {order.id}, client_order_id {cid}" if order is not None
                 else f"client_order_id {cid} (the submit was in flight)")
        rep.say(f"    Interrupted: an open order may remain: {where}. Cancel it at {PAPER_KEYS_URL}")
        raise


async def reconcile_uncertain(broker, rep: Reporter, cid: str, *, sleep, clock, timeout: float, interval: float) -> None:
    """After an uncertain submit: look the order up by client_order_id. Never resubmits."""
    rep.detail("NOT resubmitting. Looking the order up by its client_order_id instead.")
    t0 = _checkio.perf()
    try:
        found = await broker.get_order_by_client_id(cid)
    except BrokerError as exc:
        rep.detail(f"The lookup failed too ({rep.describe(exc)}), so the state is unknown. "
                   f"Check {PAPER_KEYS_URL} for an order with client_order_id {cid} and cancel it if it is open.")
        return
    if found is None:
        rep.detail(f"No order with client_order_id {cid} exists at Alpaca (yet). Nothing was resubmitted. "
                   f"If one shows up later, cancel it at {PAPER_KEYS_URL}.")
        return
    rep.detail(f"Found it: id {found.id}, status {found.status}")
    status = strip_controls(found.status).lower()
    if status == "filled":
        report_fill(rep, _checkio.perf() - t0, found.id)
        return
    if status in TERMINAL_STATUSES:
        return
    await cancel_and_confirm(broker, rep, found.id, sleep=sleep, clock=clock, timeout=timeout, interval=interval)


# --------------------------------------------------------------------------------- entry point


def key_status(value: str | None) -> str:
    return f"set (…{value[-4:]})" if value else "MISSING"


def print_missing_keys(rep: Reporter, key_id: str | None, secret: str | None) -> None:
    rep.say(f"  ALPACA_API_KEY_ID       {key_status(key_id)}")
    rep.say(f"  ALPACA_API_SECRET_KEY   {key_status(secret)}")
    rep.say()
    rep.say("The Alpaca paper keys are not (both) configured. Add these two lines to .env in the project root:")
    rep.say()
    rep.say("  ALPACA_API_KEY_ID=<your paper key id>")
    rep.say("  ALPACA_API_SECRET_KEY=<your paper secret key>")
    rep.say()
    rep.say(f"Create PAPER keys at {PAPER_KEYS_URL} (use the paper account, then Generate New Keys).")
    rep.say("Then run this script again. Nothing was sent to Alpaca.")


def build_clients(settings) -> tuple[Any, Any, list[Callable[[], Awaitable[None]]]]:
    """The real broker and market-data client. `AlpacaPaper` is never given a client of our own."""
    broker = AlpacaPaper(settings.alpaca_api_key_id, settings.alpaca_api_secret_key)
    http = HttpClient()
    data = AlpacaData(settings.alpaca_api_key_id, settings.alpaca_api_secret_key, http)
    return broker, data, [broker.aclose, http.aclose]


async def _amain(a, settings, rep: Reporter, build, read_line, interactive, sleep, clock) -> int:
    broker, data, closers = build(settings)
    evidence = EvidenceStore(AppDB(":memory:"))  # news needs an evidence store; nothing persistent is touched
    try:
        readonly_ok = await run_readonly(broker, data, rep, evidence)
        order_rc = EXIT_OK
        if a.place_test_order:
            order_rc = await place_test_order(broker, data, rep, interactive=interactive, assume_yes=a.yes_i_am_sure,
                                              read_line=read_line, sleep=sleep, clock=clock)
        rep.say("-" * 68)
        rep.say(f"{rep.passed} step(s) passed, {rep.failed} failed.")
        if order_rc == EXIT_REFUSED:
            return EXIT_REFUSED
        return EXIT_OK if readonly_ok and order_rc == EXIT_OK else EXIT_STEP_FAILED
    finally:
        for close in closers:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass


def main(
    argv: list[str] | None = None,
    *,
    settings=None,
    build=build_clients,
    read_line: Callable[[str], str] = input,
    interactive: bool | None = None,
    sleep=asyncio.sleep,
    clock=time.monotonic,
) -> int:
    # allow_abbrev=False: "--p --y" must be an argument error, never a silent --place-test-order --yes-i-am-sure.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--place-test-order", action="store_true",
                    help="after the read-only checks, place a 1-share SPY limit buy at 50%% of the last price "
                         "on the PAPER account, then cancel it (on a terminal you must type PLACE)")
    ap.add_argument("--yes-i-am-sure", action="store_true",
                    help="lets --place-test-order run when stdin is NOT a terminal (it stands in for the typed "
                         "PLACE); on a terminal PLACE is still required. The controller never passes this "
                         "without the user's go-ahead")
    a = ap.parse_args(argv)

    s = settings if settings is not None else get_settings()
    key_id, secret = s.alpaca_api_key_id, s.alpaca_api_secret_key
    rep = Reporter([key_id, secret])
    rep.say(f"Alpaca PAPER check   host {getattr(broker_mod, 'PAPER_BASE', '?')}")
    if not (key_id and secret):
        print_missing_keys(rep, key_id, secret)
        return EXIT_NO_KEYS
    rep.say(f"  ALPACA_API_KEY_ID       {key_status(key_id)}")
    rep.say(f"  ALPACA_API_SECRET_KEY   {key_status(secret)}")
    if a.place_test_order:
        rep.say("  mode: read-only checks, then ONE paper test order (needs confirmation)")
    else:
        rep.say("  mode: read-only (no orders are placed)")
        if a.yes_i_am_sure:
            rep.say("  note: --yes-i-am-sure does nothing without --place-test-order")
    rep.say("-" * 68)

    if interactive is None:
        interactive = bool(sys.stdin and sys.stdin.isatty())
    return asyncio.run(_amain(a, s, rep, build, read_line, interactive, sleep, clock))


if __name__ == "__main__":
    sys.exit(main())
