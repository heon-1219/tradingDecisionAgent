"""Offline tests for `scripts/check_alpaca.py` and `scripts/check_research.py`.

The scripts are not a package, so they are loaded by path (the way `test_eval_truth_markers.py` does).
Everything runs against `fakes_trading.FakeBroker` / `FakeData` and small research fakes: no network,
no real `.env` (settings are passed in), no real app DB. An autouse fixture makes any accidental
HTTP or DNS call fail loudly instead of reaching the network.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import re
import socket
import sys
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from fakes_trading import FakeBroker, FakeData, make_account, make_clock, make_quote
from finagent.research.evidence import EvidenceStore, ResearchResult
from finagent.research.grounding import GroundedAnswer, GroundedWeb
from finagent.research.http import HttpError
from finagent.research.edgar import Edgar
from finagent.research.websearch import WebSearch
from finagent.store.appdb import AppDB
from finagent.trading import broker as broker_mod
from finagent.trading.broker import BrokerError, OrderUncertain
from finagent.trading.marketdata import AlpacaData

D = Decimal
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass field resolution needs the module registered before exec
    spec.loader.exec_module(mod)
    return mod


check_alpaca = _load("check_alpaca")
check_research = _load("check_research")
checkio = sys.modules["_checkio"]  # the shared output module both scripts import

KEY_ID = "PKTESTKEYID0000AAAA"
SECRET = "sEcReTvAlUe-0123456789-zzzz"
CONFIRM_URL = "https://app.alpaca.markets/paper/dashboard/overview"
LATENCY_OK_RE = re.compile(r"✓ \d+\.\d{2}s")
# ESC (CSI/OSC), BEL, C1 (0x80-0x9f, e.g. 0x9b = CSI) and DEL: nothing a terminal would act on
TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")
HOSTILE = "\x1b[2J\x1b]0;pwned\x07 visible-text \x9b31m end"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real HTTP/DNS use inside these tests is a bug in the test or the script."""
    async def _blocked(self, request):
        raise AssertionError(f"network call attempted: {request.method} {request.url}")

    def _blocked_dns(*a, **k):
        raise AssertionError(f"DNS lookup attempted: {a[:1]}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_dns)


# --------------------------------------------------------------------------- shared fakes


class StubHttp:
    """Stands in for `HttpClient` under `AlpacaData.news`: canned JSON, records the requests."""

    def __init__(self, payload):
        self.payload = payload
        self.requests: list[tuple[str, dict | None]] = []

    async def get_json(self, url, *, params=None, headers=None, timeout=20.0):
        self.requests.append((url, params))
        return self.payload


NEWS_PAYLOAD = {
    "news": [
        {"headline": "Fed holds rates steady", "summary": "s1", "source": "benzinga",
         "created_at": "2026-09-28T13:00:00Z", "url": "https://example.com/1"},
        {"headline": "<b>Chip stocks</b> rally<script>alert(1)</script>\x1b[31m red", "summary": "s2",
         "source": "reuters", "created_at": "2026-09-28T12:00:00Z", "url": "https://example.com/2"},
        {"headline": "H" * 400, "summary": "s3", "source": "wsj",
         "created_at": "2026-09-28T11:00:00Z", "url": "https://example.com/3"},
    ]
}


class NewsData(FakeData):
    """`FakeData` plus the real `AlpacaData.news` over a stub HTTP client (so the real
    sanitizing/wrapping path runs)."""

    news = AlpacaData.news

    def __init__(self, quotes=None, payload=NEWS_PAYLOAD):
        super().__init__(quotes if quotes is not None else {"SPY": make_quote("SPY", "512.37")})
        self._http = StubHttp(payload)
        self._headers = {}


class FailingNewsData(NewsData):
    async def news(self, *a, **k):
        raise HttpError(500, "boom")


class FailingSnapshotData(NewsData):
    async def snapshots(self, symbols):
        raise HttpError(403, "forbidden")


def alpaca_settings(**over):
    base = dict(alpaca_api_key_id=KEY_ID, alpaca_api_secret_key=SECRET, tavily_api_key=None,
                sec_user_agent=None, gemini_api_key=None, grounding_model="gemini-test")
    base.update(over)
    return SimpleNamespace(**base)


def run_main(broker, data, argv=(), *, settings=None, read_line=None, interactive=True):
    """Drive `check_alpaca.main` with fakes; `read_line` defaults to a function that must not be called."""
    def _no_prompt(prompt=""):
        raise AssertionError("the script prompted for input unexpectedly")

    async def _no_sleep(_):
        return None

    return check_alpaca.main(
        list(argv),
        settings=settings or alpaca_settings(),
        build=lambda s: (broker, data, []),
        read_line=read_line or _no_prompt,
        interactive=interactive,
        sleep=_no_sleep,
    )


# --------------------------------------------------------------------------- check_alpaca: read-only


def test_readonly_flow_prints_the_four_sections_and_exits_zero(capsys):
    broker, data = FakeBroker(), NewsData()
    rc = run_main(broker, data)
    out = capsys.readouterr().out

    assert rc == 0
    # section 1: account, Decimal strings
    assert "ACTIVE" in out and "100000" in out and "50000" in out
    # section 2: clock
    assert "market clock" in out.lower() and "next open" in out.lower()
    # section 3: SPY IEX snapshot (bid / ask / last / timestamp)
    assert "SPY" in out
    # section 4: three news items, sanitized
    assert "Fed holds rates steady" in out and "benzinga" in out
    assert "<script>" not in out and "<b>" not in out and "\x1b" not in out
    assert "H" * 150 not in out  # headline truncated
    assert out.count("✓") == 4 and "✗" not in out
    assert len(LATENCY_OK_RE.findall(out)) == 4  # every step prints its latency, e.g. "✓ 0.00s"
    # strictly read-only: only reads reached the broker, nothing was submitted or cancelled
    assert set(broker.calls) == {"account", "clock"}
    assert broker.submits == 0 and broker.cancelled == []
    assert data.calls == [["SPY"]]
    # the news request asked for 3 items
    assert data._http.requests[0][1]["limit"] == 3


def test_readonly_snapshot_shows_bid_ask_last_and_timestamp(capsys):
    data = NewsData(quotes={"SPY": make_quote("SPY", "512.37", "0.02")})
    run_main(FakeBroker(), data)
    out = capsys.readouterr().out
    assert "512.35" in out and "512.39" in out and "512.37" in out
    assert "2026-09-28T14:00:00Z" in out


async def test_each_step_reports_its_measured_latency(monkeypatch, capsys):
    ticks = iter(x * 0.25 for x in range(1000))  # every perf-counter read advances 0.25 s
    monkeypatch.setattr(checkio, "perf", lambda: next(ticks))
    rep = check_alpaca.Reporter()
    ok = await check_alpaca.run_readonly(FakeBroker(), NewsData(), rep, EvidenceStore(AppDB(":memory:")))
    out = capsys.readouterr().out
    assert ok is True
    ok_lines = [ln for ln in out.splitlines() if "✓" in ln]
    assert len(ok_lines) == 4
    assert all(re.fullmatch(r"\s+✓ 0\.25s  \S.*", ln) for ln in ok_lines), ok_lines


async def test_a_failed_step_reports_its_latency_and_reason(monkeypatch, capsys):
    ticks = iter(x * 0.5 for x in range(1000))
    monkeypatch.setattr(checkio, "perf", lambda: next(ticks))
    broker = FakeBroker()
    broker.account_error = BrokerError(500, None, "server exploded")
    rep = check_alpaca.Reporter()
    ok = await check_alpaca.run_readonly(broker, NewsData(), rep, EvidenceStore(AppDB(":memory:")))
    out = capsys.readouterr().out
    assert ok is False
    assert re.search(r"✗ 0\.50s  BrokerError: .*server exploded", out)


def test_a_failing_step_shows_a_cross_and_exit_one_but_later_steps_still_run(capsys):
    broker = FakeBroker()
    broker.account_error = BrokerError(401, None, "unauthorized")
    data = NewsData()
    rc = run_main(broker, data)
    out = capsys.readouterr().out

    assert rc == 1
    assert out.count("✗") == 1 and out.count("✓") == 3
    assert "401" in out and "unauthorized" in out
    assert "PAPER" in out  # the hint points at paper keys
    assert "clock" in broker.calls  # later steps still ran
    assert data.calls == [["SPY"]]
    assert "Fed holds rates steady" in out


def test_failures_in_every_data_step_are_isolated(capsys):
    rc = run_main(FakeBroker(), FailingSnapshotData())
    out = capsys.readouterr().out
    assert rc == 1 and out.count("✗") == 1 and out.count("✓") == 3 and "403" in out

    rc = run_main(FakeBroker(), FailingNewsData())
    out = capsys.readouterr().out
    assert rc == 1 and out.count("✗") == 1 and out.count("✓") == 3


def test_a_snapshot_without_spy_is_a_failed_step(capsys):
    rc = run_main(FakeBroker(), NewsData(quotes={}))
    out = capsys.readouterr().out
    assert rc == 1 and "SPY" in out and out.count("✗") == 1


def test_a_blocked_account_is_a_failed_account_step(capsys):
    broker = FakeBroker(account=make_account(trading_blocked=True))
    rc = run_main(broker, NewsData())
    out = capsys.readouterr().out
    assert rc == 1 and "trading_blocked" in out


def test_missing_keys_exit_two_with_instructions_and_no_secret(capsys):
    built = []
    for settings in (
        alpaca_settings(alpaca_api_key_id=KEY_ID, alpaca_api_secret_key=None),
        alpaca_settings(alpaca_api_key_id=None, alpaca_api_secret_key=SECRET),
        alpaca_settings(alpaca_api_key_id=None, alpaca_api_secret_key=None),
    ):
        rc = check_alpaca.main([], settings=settings, build=lambda s: built.append(s) or (None, None, []))
        out = capsys.readouterr().out
        assert rc == 2
        assert "ALPACA_API_KEY_ID" in out and "ALPACA_API_SECRET_KEY" in out
        assert ".env" in out and CONFIRM_URL in out
        assert SECRET not in out and KEY_ID not in out  # never the key itself
    assert built == []  # no client was even built


def test_missing_keys_show_only_the_last_four_of_a_key_that_is_set(capsys):
    check_alpaca.main([], settings=alpaca_settings(alpaca_api_secret_key=None),
                      build=lambda s: (None, None, []))
    out = capsys.readouterr().out
    assert "set (…AAAA)" in out and KEY_ID not in out
    assert "MISSING" in out


def test_secrets_inside_an_error_message_are_scrubbed(capsys):
    broker = FakeBroker()
    broker.account_error = BrokerError(401, None, f"bad credentials {KEY_ID} / {SECRET}")
    rc = run_main(broker, NewsData())
    out = capsys.readouterr().out
    assert rc == 1
    assert SECRET not in out and KEY_ID not in out and "***" in out


# --------------------------------------------------------------------------- check_alpaca: test order


def order_data(last="512.37"):
    return NewsData(quotes={"SPY": make_quote("SPY", last)})


def cancelling(broker: FakeBroker, statuses=("canceled",)):
    """A submit action that places the order and scripts the statuses `get_order` will report."""
    def action(b, leg, cid):
        order = b.place(leg, cid)
        b.status_script[order.id] = list(statuses)
        return order
    return action


def place(broker, data, *, typed="PLACE", extra=(), interactive=True):
    prompts: list[str] = []

    def read_line(prompt=""):
        prompts.append(prompt)
        # nothing may have been submitted by the time the user is asked
        assert broker.submits == 0
        return typed

    rc = run_main(broker, data, ["--place-test-order", *extra], read_line=read_line, interactive=interactive)
    return rc, prompts


def test_place_test_order_submits_one_half_price_limit_and_cancels(capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker)]
    rc, prompts = place(broker, order_data("512.37"))
    out = capsys.readouterr().out

    assert rc == 0
    assert len(prompts) == 1 and "PLACE" in prompts[0]
    assert broker.submits == 1
    leg, cid = broker.submitted[0]
    assert (leg.symbol, leg.side, leg.qty, leg.type, leg.time_in_force) == ("SPY", "buy", D(1), "limit", "day")
    # 50% of 512.37 is 256.185: rounded DOWN to cents (half-up would give 256.19)
    assert leg.limit_price == D("256.18") and isinstance(leg.limit_price, Decimal)
    assert leg.notional is None
    broker_mod._validate_client_order_id(cid)  # a format AlpacaPaper accepts
    order = next(iter(broker.orders_by_id.values()))
    assert broker.cancelled == [order.id]
    assert "accepted" in out and "canceled" in out
    assert out.count("✗") == 0
    assert broker.calls.count("submit_order") == 1 and broker.reconcile_calls == []


@pytest.mark.parametrize("last,expected", [
    ("100.03", "50.01"),    # 50.015 -> 50.01
    ("100", "50.00"),
    ("0.05", "0.02"),       # 0.025 -> 0.02
    ("1234.56", "617.28"),
])
def test_half_price_is_rounded_down_with_decimal(last, expected, capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker)]
    rc, _ = place(broker, order_data(last))
    assert rc == 0
    assert broker.submitted[0][0].limit_price == D(expected)


def test_each_run_uses_a_fresh_client_order_id():
    ids = []
    for _ in range(2):
        broker = FakeBroker()
        broker.submit_actions = [cancelling(broker)]
        place(broker, order_data())
        ids.append(broker.submitted[0][1])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("typed", ["", "place", "yes", "PLACE ", " PLACE", "PLACE!", "y", "PLACEPLACE"])
def test_without_the_exact_typed_confirmation_nothing_is_placed(typed, capsys):
    broker = FakeBroker()
    rc, prompts = place(broker, order_data(), typed=typed)
    out = capsys.readouterr().out
    assert rc == 3
    assert len(prompts) == 1
    assert broker.submits == 0 and "submit_order" not in broker.calls
    assert broker.cancelled == []
    assert "nothing was placed" in out.lower()


def test_eof_on_stdin_places_nothing(capsys):
    broker = FakeBroker()

    def eof(prompt=""):
        raise EOFError

    rc = run_main(broker, order_data(), ["--place-test-order"], read_line=eof, interactive=True)
    assert rc == 3 and broker.submits == 0


def test_non_interactive_stdin_refuses_without_the_flag(capsys):
    broker = FakeBroker()
    rc = run_main(broker, order_data(), ["--place-test-order"], interactive=False)  # prompting would assert
    out = capsys.readouterr().out
    assert rc == 3 and broker.submits == 0
    assert "terminal" in out.lower() and "--yes-i-am-sure" in out
    # Git Bash/mintty reports no TTY for native Python: the fix comes first, the flag is the last resort
    assert "PowerShell" in out and "winpty" in out and "Git Bash" in out
    assert out.index("PowerShell") < out.index("winpty") < out.index("--yes-i-am-sure")
    assert "last resort" in out.lower()


def test_non_interactive_stdin_with_the_flag_places_one_order(capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker)]
    rc = run_main(broker, order_data(), ["--place-test-order", "--yes-i-am-sure"], interactive=False)
    assert rc == 0 and broker.submits == 1 and len(broker.cancelled) == 1


def test_the_flag_alone_never_places_anything(capsys):
    broker = FakeBroker()
    rc = run_main(broker, order_data(), ["--yes-i-am-sure"], interactive=False)
    assert rc == 0 and broker.submits == 0
    assert "--place-test-order" in capsys.readouterr().out


def test_the_secret_key_never_shows_even_its_last_four(capsys):
    # Final review A-m7: only the key ID's last 4 characters are shown; the secret is just "set".
    rc = run_main(FakeBroker(), NewsData())
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALPACA_API_KEY_ID       set (…AAAA)" in out
    assert "ALPACA_API_SECRET_KEY   set\n" in out
    assert SECRET[-4:] not in out and SECRET not in out


def test_the_test_order_is_refused_while_the_kill_switch_is_on(capsys):
    # Final review A-m7: TRADING_ENABLED=false (the app's kill switch) stops this script too.
    broker = FakeBroker()
    rc = run_main(broker, order_data(), ["--place-test-order"],
                  settings=alpaca_settings(trading_enabled=False))  # read_line must not be called
    out = capsys.readouterr().out
    assert rc == 3 and broker.submits == 0
    assert "TRADING_ENABLED is false" in out and "Nothing was placed." in out
    assert out.count("✓") == 4  # the read-only checks still ran


def test_refuses_when_the_broker_host_constant_is_not_paper(monkeypatch, capsys):
    monkeypatch.setattr(broker_mod, "PAPER_BASE", "https://api.alpaca.markets")
    broker = FakeBroker()
    rc, prompts = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 3 and broker.submits == 0 and prompts == []
    assert "paper" in out.lower() and "refus" in out.lower()


def test_refuses_when_alpaca_says_the_account_is_not_paper(capsys):
    broker = FakeBroker(account=make_account(paper=False))
    rc, prompts = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 3 and broker.submits == 0 and prompts == []
    assert "not a paper account" in out


def test_paper_status_unknown_continues_relying_on_the_paper_host(capsys):
    broker = FakeBroker(account=make_account())  # paper=None: the account number was not reported
    broker.submit_actions = [cancelling(broker)]
    rc, prompts = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 0 and broker.submits == 1 and len(prompts) == 1
    assert "paper status unknown; relying on the paper host" in out
    assert "PAPER (verified)" not in out


def test_a_verified_paper_account_is_stated_in_the_plan(capsys):
    broker = FakeBroker(account=make_account(paper=True))
    broker.submit_actions = [cancelling(broker)]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 0
    assert "Account: PAPER (verified)" in out and "unknown" not in out


def test_the_readonly_account_step_shows_the_paper_flag(capsys):
    for account, expected, rc in ((make_account(paper=True), "verified", 0),
                                  (make_account(), "unknown", 0),
                                  (make_account(paper=False), "NOT", 1)):
        assert run_main(FakeBroker(account=account), NewsData()) == rc
        out = capsys.readouterr().out
        assert re.search(rf"paper account\s+.*{expected}", out), out


@pytest.mark.parametrize("over", [{"status": "ACCOUNT_UPDATED"}, {"trading_blocked": True},
                                   {"account_blocked": True}, {"trade_suspended_by_user": True}])
def test_refuses_when_the_account_cannot_trade(over, capsys):
    broker = FakeBroker(account=make_account(**over))
    rc, prompts = place(broker, order_data())
    assert rc == 3 and broker.submits == 0 and prompts == []


def test_refuses_without_a_last_price(capsys):
    quote = make_quote("SPY", "500").model_copy(update={"last": None})
    broker = FakeBroker()
    rc, prompts = place(broker, NewsData(quotes={"SPY": quote}))
    assert rc == 3 and broker.submits == 0 and prompts == []
    assert "last price" in capsys.readouterr().out.lower()


def test_a_broker_rejection_is_reported_and_not_retried(capsys):
    broker = FakeBroker()
    broker.submit_actions = [BrokerError(403, 40310000, "insufficient buying power")]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 1 and broker.submits == 1
    assert "insufficient buying power" in out and "✗" in out
    assert broker.reconcile_calls == [] and broker.cancelled == []


def test_uncertain_submit_reconciles_cancels_and_never_resubmits(capsys):
    broker = FakeBroker()

    def placed_then_lost(b, leg, cid):
        cancelling(b)(b, leg, cid)
        raise OrderUncertain(0, None, "ReadTimeout: timed out")

    broker.submit_actions = [placed_then_lost]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out

    assert rc == 1  # an uncertain outcome is never reported as a pass
    assert broker.submits == 1, "an uncertain submit must never be retried"
    cid = broker.submitted[0][1]
    assert broker.reconcile_calls == [cid]
    assert len(broker.cancelled) == 1  # the order that did reach Alpaca is cleaned up
    assert "UNCERTAIN" in out and "resubmit" in out.lower()


def test_uncertain_submit_with_no_order_found_reports_and_stops(capsys):
    broker = FakeBroker()
    broker.submit_actions = ["timeout"]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 1 and broker.submits == 1
    assert broker.reconcile_calls == [broker.submitted[0][1]]
    assert broker.cancelled == []
    assert "no order" in out.lower() and CONFIRM_URL in out


def test_uncertain_submit_with_a_failing_reconcile_says_the_state_is_unknown(capsys):
    broker = FakeBroker()
    broker.submit_actions = ["placed_then_timeout"]
    broker.reconcile_error = BrokerError(0, None, "ConnectError")
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 1 and broker.submits == 1
    assert "unknown" in out.lower() and broker.submitted[0][1] in out


def test_a_cancel_that_never_confirms_is_reported_after_the_poll_budget(capsys):
    broker = FakeBroker()  # get_order keeps answering "accepted": the cancel is never confirmed
    t = [0.0]
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)
        t[0] += s

    rc = check_alpaca.main(
        ["--place-test-order"], settings=alpaca_settings(),
        build=lambda s: (broker, order_data(), []),
        read_line=lambda p="": "PLACE", interactive=True, sleep=sleep, clock=lambda: t[0],
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert broker.submits == 1 and len(broker.cancelled) == 1
    assert 8 <= sum(slept) <= 12  # polled for about 10 seconds, no more
    assert "not confirmed" in out.lower() and "accepted" in out and CONFIRM_URL in out


def test_pending_cancel_is_polled_until_the_final_status(capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker, ("pending_cancel", "pending_cancel", "canceled"))]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 0 and "canceled" in out and len(broker.get_order_calls) == 3


def test_an_order_that_fills_is_flagged_loudly(capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker, ("filled",))]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 1 and "FILLED" in out and "✗" in out


def test_a_failed_cancel_request_is_reported_with_the_order_id(capsys):
    broker = FakeBroker()
    holder = {}

    def action(b, leg, cid):
        order = b.place(leg, cid)
        holder["id"] = order.id
        b.cancel_errors[order.id] = BrokerError(422, 42210000, "order is not cancelable")
        b.status_script[order.id] = ["filled"]
        return order

    broker.submit_actions = [action]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 1 and "not cancelable" in out and holder["id"] in out


def test_readonly_failure_still_lets_the_order_flow_refuse_safely(capsys):
    broker = FakeBroker()
    broker.account_error = BrokerError(401, None, "unauthorized")
    rc, prompts = place(broker, order_data())
    assert rc == 1 and broker.submits == 0 and prompts == []


# --------------------------------------------------------------------------- fix round 1: check_alpaca


@pytest.mark.parametrize("argv", [
    ["--p", "--y"],
    ["--place", "--yes"],
    ["--place-test", "--yes-i-am-sure"],
    ["--place-test-order", "--yes"],
])
def test_abbreviated_flags_are_argument_errors_and_place_nothing(argv, capsys):
    broker = FakeBroker()
    with pytest.raises(SystemExit) as exc:
        run_main(broker, order_data(), argv, interactive=False)
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert broker.calls == [] and broker.submits == 0  # not even a read: the parse failed first


def test_a_secret_straddling_the_truncation_limit_is_still_blanked(capsys):
    broker = FakeBroker()
    # "BrokerError: Alpaca broker error 500: " is 38 characters, so the secret starts at ~208 and
    # crosses the 220-character cut: truncating before scrubbing would leave a readable fragment.
    broker.account_error = BrokerError(500, None, "x" * 170 + SECRET + " tail")
    rc = run_main(broker, NewsData())
    out = capsys.readouterr().out
    assert rc == 1
    assert SECRET[:8] not in out and SECRET[-8:] not in out and KEY_ID[:8] not in out
    assert "***" in out


def test_control_characters_in_statuses_and_clock_never_reach_the_terminal(capsys):
    broker = FakeBroker(account=make_account(status="ACTIVE" + HOSTILE),
                        clock=make_clock().model_copy(update={"timestamp": HOSTILE, "next_open": HOSTILE,
                                                              "next_close": HOSTILE}))
    run_main(broker, NewsData())
    out = capsys.readouterr().out
    assert TERMINAL_CONTROL_RE.search(out) is None
    assert "visible-text" in out  # the readable part survives


def test_control_characters_in_an_order_status_never_reach_the_terminal(capsys):
    broker = FakeBroker()

    def hostile_status(b, leg, cid):
        order = b.place(leg, cid, status="accepted" + HOSTILE)
        b.status_script[order.id] = ["canceled"]
        return order

    broker.submit_actions = [hostile_status]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    assert rc == 0 and TERMINAL_CONTROL_RE.search(out) is None and "visible-text" in out


def test_control_characters_in_an_error_message_never_reach_the_terminal(capsys):
    broker = FakeBroker()
    broker.account_error = BrokerError(502, None, "bad gateway " + HOSTILE)
    run_main(broker, NewsData())
    out = capsys.readouterr().out
    assert TERMINAL_CONTROL_RE.search(out) is None and "bad gateway" in out and "visible-text" in out


def test_yes_i_am_sure_on_a_terminal_still_requires_the_typed_word(capsys):
    broker = FakeBroker()
    rc, prompts = place(broker, order_data(), typed="yes", extra=["--yes-i-am-sure"], interactive=True)
    out = capsys.readouterr().out
    assert rc == 3 and len(prompts) == 1 and broker.submits == 0
    assert "nothing was placed" in out.lower()


def test_yes_i_am_sure_on_a_terminal_with_the_word_places_one_order(capsys):
    broker = FakeBroker()
    broker.submit_actions = [cancelling(broker)]
    rc, prompts = place(broker, order_data(), typed="PLACE", extra=["--yes-i-am-sure"], interactive=True)
    assert rc == 0 and len(prompts) == 1 and broker.submits == 1


def test_ctrl_c_at_the_confirmation_prompt_places_nothing(capsys):
    broker = FakeBroker()

    def interrupted(prompt=""):
        raise KeyboardInterrupt

    rc = run_main(broker, order_data(), ["--place-test-order"], read_line=interrupted, interactive=True)
    assert rc == 3 and broker.submits == 0
    assert "nothing was placed" in capsys.readouterr().out.lower()


def test_ctrl_c_after_the_submit_says_an_open_order_may_remain_and_re_raises(capsys):
    broker = FakeBroker()
    holder = {}

    def action(b, leg, cid):
        order = b.place(leg, cid)
        holder["id"] = order.id
        return order

    broker.submit_actions = [action]
    broker.get_order_error = KeyboardInterrupt()  # the user hits Ctrl-C while the cancel is being polled
    with pytest.raises(KeyboardInterrupt):
        place(broker, order_data())
    out = capsys.readouterr().out
    assert "an open order may remain" in out
    assert holder["id"] in out and broker.submitted[0][1] in out and CONFIRM_URL in out
    assert broker.submits == 1


async def test_task_cancellation_after_the_submit_is_handled_the_same_way(capsys):
    broker = FakeBroker()
    holder = {}

    def action(b, leg, cid):
        order = b.place(leg, cid)
        holder["id"] = order.id
        return order

    broker.submit_actions = [action]
    broker.get_order_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await check_alpaca.place_test_order(broker, order_data(), check_alpaca.Reporter(), interactive=True,
                                            assume_yes=False, read_line=lambda p="": "PLACE")
    out = capsys.readouterr().out
    assert "an open order may remain" in out and holder["id"] in out and CONFIRM_URL in out


def test_ctrl_c_while_the_submit_is_in_flight_names_the_client_order_id(capsys):
    broker = FakeBroker()
    broker.submit_actions = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        place(broker, order_data())
    out = capsys.readouterr().out
    assert "an open order may remain" in out and broker.submitted[0][1] in out and CONFIRM_URL in out
    assert broker.submits == 1


def test_a_reconciled_order_that_filled_is_flagged_as_loudly_as_a_normal_fill(capsys):
    broker = FakeBroker()

    def filled_then_lost(b, leg, cid):
        b.place(leg, cid, status="filled")
        raise OrderUncertain(0, None, "ReadTimeout: timed out")

    broker.submit_actions = [filled_then_lost]
    rc, _ = place(broker, order_data())
    out = capsys.readouterr().out
    order_id = next(iter(broker.orders_by_id))
    assert rc == 1 and broker.submits == 1 and broker.cancelled == []
    assert "the test order FILLED: the paper account now holds 1 share of SPY" in out
    assert order_id in out and CONFIRM_URL in out
    assert out.count("✗") == 2  # the uncertain submit and the fill

    # the same wording as the normal path (cancel poll sees "filled")
    normal = FakeBroker()
    normal.submit_actions = [cancelling(normal, ("filled",))]
    place(normal, order_data())
    assert "the test order FILLED: the paper account now holds 1 share of SPY" in capsys.readouterr().out


# --------------------------------------------------------------------------- check_research


class FakeEdgar:
    def __init__(self, evidence, *, error=False, empty=False):
        self.evidence, self.error, self.empty, self.calls = evidence, error, empty, []

    async def financial_facts(self, session_id, ticker, concepts, periods=8):
        self.calls.append((session_id, ticker, list(concepts)))
        if self.error:
            return ResearchResult(text="Could not fetch SEC EDGAR XBRL company facts.", sources=[], is_error=True)
        if self.empty:
            return ResearchResult(text="Revenues: no data found.", sources=[])
        src = self.evidence.add(session_id, connector="edgar", title="Revenues", url="https://sec.example/f",
                                publisher="SEC EDGAR XBRL", tier=1, published_at="2025-09-27",
                                snippet="Revenues: FY2025 val=1", untrusted=False)
        return ResearchResult(text=f"[{src.id}] Revenues FY2025FY end=2025-09-27 val=391035000000 USD", sources=[src])


class FakeWeb:
    def __init__(self, evidence, *, kind, error=None, long=False, body=None):
        self.evidence, self.kind, self.error, self.long, self.body = evidence, kind, error, long, body
        self.calls: list[tuple] = []

    def _result(self, session_id, url, n):
        sources, lines = [], []
        for i in range(n):
            body = self.body or (("x" * 400) if self.long else f"snippet {i} " + "ab" * 10)
            src = self.evidence.add(session_id, connector="websearch", title=f"t{i}", url=f"{url}/{i}",
                                    publisher="example.com", tier=3, published_at=None, snippet=body,
                                    untrusted=True)
            sources.append(src)
            lines.append(f'[{src.id}] <untrusted source="{src.id}">{body}</untrusted>')
        return ResearchResult(text="\n".join(lines), sources=sources)

    async def search(self, session_id, query, **kw):
        self.calls.append(("search", query))
        if self.error:
            raise self.error
        return self._result(session_id, "https://found.example", 3)

    async def fetch(self, session_id, url, **kw):
        self.calls.append(("fetch", url))
        if self.error:
            raise self.error
        return self._result(session_id, url, 1)


class FakeGrounded:
    def __init__(self, *, error=None, text=None, queries=None):
        self.error, self.questions, self.text, self.queries = error, [], text, queries

    async def answer(self, question):
        self.questions.append(question)
        return GroundedAnswer(text=self.text or "Apple's ticker is AAPL. " * 20,
                              sources=[{"title": "t", "uri": "https://g.example"}],
                              search_suggestions_html=None, queries=self.queries or ["apple ticker"],
                              error=self.error)


def research_settings(**over):
    base = dict(sec_user_agent=None, tavily_api_key=None, gemini_api_key=None, grounding_model="gemini-test",
                app_db_path=pathlib.Path("must-not-be-used") / "app.db")
    base.update(over)
    return SimpleNamespace(**base)


def run_research(make_conns, argv=(), *, settings=None, tmp_root=None):
    seen = {}

    def factory(s, http, evidence, with_grounding):
        seen["with_grounding"] = with_grounding
        seen["evidence"] = evidence
        return make_conns(evidence, with_grounding)

    rc = check_research.main(list(argv), settings=settings or research_settings(),
                             connectors_factory=factory, tmp_root=tmp_root)
    return rc, seen


def conns(evidence, with_grounding=False, **over):
    base = dict(
        edgar=None, tavily=None, jina=FakeWeb(evidence, kind="jina"), grounded=None,
        skips={"edgar": "SEC_USER_AGENT is not set", "tavily": "TAVILY_API_KEY is not set",
               "grounding": "pass --grounding to run it"},
    )
    base.update(over)
    return check_research.Connectors(**base)


def test_skipped_steps_are_reported_and_do_not_fail(capsys, tmp_path):
    rc, _ = run_research(lambda ev, g: conns(ev), tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("SKIPPED") == 3
    assert "SEC_USER_AGENT" in out and "TAVILY_API_KEY" in out and "--grounding" in out
    assert out.count("✓") == 1 and "✗" not in out  # only Jina ran


def test_all_steps_pass_with_counts_evidence_ids_and_120_char_previews(capsys, tmp_path):
    holder = {}

    def make(ev, g):
        holder["conns"] = conns(ev, edgar=FakeEdgar(ev), tavily=FakeWeb(ev, kind="tavily", long=True),
                                jina=FakeWeb(ev, kind="jina", long=True), grounded=FakeGrounded())
        return holder["conns"]

    rc, seen = run_research(make, ["--grounding"], settings=research_settings(), tmp_root=tmp_path)
    out = capsys.readouterr().out

    assert rc == 0 and out.count("✓") == 4 and "✗" not in out and "SKIPPED" not in out
    assert "1 record" in out and "3 records" in out
    # evidence ids: edgar E1, tavily E2-E4, jina E5
    for eid in ("E1", "E2", "E4", "E5"):
        assert eid in out
    # the first 120 characters of each wrapped result, and no more
    wrapped = f'[E2] <untrusted source="E2">{"x" * 400}</untrusted>'
    assert wrapped[:120] in out and wrapped[:121] not in out
    # EDGAR was asked for AAPL revenue; grounding was asked exactly once
    assert holder["conns"].edgar.calls[0][1] == "AAPL" and "Revenues" in holder["conns"].edgar.calls[0][2]
    assert len(holder["conns"].grounded.questions) == 1
    assert seen["with_grounding"] is True


def test_grounding_is_not_called_without_the_flag(capsys, tmp_path):
    grounded = FakeGrounded()
    rc, seen = run_research(lambda ev, g: conns(ev, grounded=grounded if g else None), tmp_root=tmp_path)
    assert rc == 0 and seen["with_grounding"] is False and grounded.questions == []
    assert "--grounding" in capsys.readouterr().out


def test_failed_attempts_exit_one_but_the_remaining_steps_still_run(capsys, tmp_path):
    jina_holder = {}

    def make(ev, g):
        jina_holder["jina"] = FakeWeb(ev, kind="jina")
        return conns(ev, edgar=FakeEdgar(ev, error=True),
                     tavily=FakeWeb(ev, kind="tavily", error=HttpError(429, "rate limited")),
                     jina=jina_holder["jina"], grounded=FakeGrounded(error="Gemini grounding request failed (HTTP 429)."))

    rc, _ = run_research(make, ["--grounding"], tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("✗") == 3 and out.count("✓") == 1
    assert "429" in out and "Could not fetch SEC EDGAR" in out
    assert jina_holder["jina"].calls == [("fetch", "https://example.com/")]


def test_an_edgar_answer_with_no_records_counts_as_a_failure(capsys, tmp_path):
    rc, _ = run_research(lambda ev, g: conns(ev, edgar=FakeEdgar(ev, empty=True)), tmp_root=tmp_path)
    assert rc == 1 and "0 records" in capsys.readouterr().out


def test_the_temp_db_is_used_and_removed_and_the_real_db_is_never_touched(capsys, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    real = tmp_path / "real" / "app.db"
    during = {}

    def make(ev, g):
        during["entries"] = [(e.is_dir(), (e / "check.db").exists()) for e in scratch.iterdir()]
        during["rows"] = ev.list("live-check")
        ev.add("live-check", connector="x", title="t", url="u", publisher=None, tier=3, published_at=None,
               snippet="s", untrusted=True)
        return conns(ev)

    rc, _ = run_research(make, settings=research_settings(app_db_path=real), tmp_root=scratch)
    capsys.readouterr()
    assert rc == 0
    assert during["entries"] == [(True, True)]  # a temp dir with the temp DB existed during the run
    assert list(scratch.iterdir()) == []  # ...and is gone afterwards
    assert not real.parent.exists()


def test_the_temp_db_is_removed_even_when_a_step_blows_up(capsys, tmp_path):
    def make(ev, g):
        return conns(ev, jina=FakeWeb(ev, kind="jina", error=RuntimeError("kaboom")))

    rc, _ = run_research(make, tmp_root=tmp_path)
    assert rc == 1 and list(tmp_path.iterdir()) == []
    assert "kaboom" in capsys.readouterr().out


def test_no_secret_reaches_the_output(capsys, tmp_path):
    secrets = dict(tavily_api_key="tvly-SECRET-abcdef123456", gemini_api_key="AIzaSECRETgeminiKEY9876",
                   sec_user_agent="Jane Doe jane@example.com")

    def make(ev, g):
        return conns(ev, tavily=FakeWeb(ev, kind="tavily", error=HttpError(401, f"bad key {secrets['tavily_api_key']}")),
                     grounded=FakeGrounded(error=f"failed with {secrets['gemini_api_key']}"))

    rc, _ = run_research(make, ["--grounding"], settings=research_settings(**secrets), tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 1
    assert secrets["tavily_api_key"] not in out and secrets["gemini_api_key"] not in out
    assert "***" in out


@pytest.mark.parametrize("argv", [["--ground"], ["--groundin"], ["--g"]])
def test_research_abbreviated_flags_are_argument_errors(argv, capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_research(lambda ev, g: conns(ev), argv, tmp_root=tmp_path)
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # it failed before any temp DB was made


def test_terminal_escapes_in_search_and_fetch_previews_are_stripped(capsys, tmp_path):
    def make(ev, g):
        return conns(ev, tavily=FakeWeb(ev, kind="tavily", body=HOSTILE), jina=FakeWeb(ev, kind="jina", body=HOSTILE))

    rc, _ = run_research(make, tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert TERMINAL_CONTROL_RE.search(out) is None
    assert out.count("visible-text") == 4  # 3 search results + 1 fetch: the readable part survives


def test_terminal_escapes_in_the_grounding_preview_and_queries_are_stripped(capsys, tmp_path):
    def make(ev, g):
        return conns(ev, grounded=FakeGrounded(text=HOSTILE, queries=[HOSTILE, "second " + HOSTILE]))

    rc, _ = run_research(make, ["--grounding"], tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0 and TERMINAL_CONTROL_RE.search(out) is None
    assert out.count("visible-text") >= 3  # the answer preview and both queries


def test_terminal_escapes_in_error_messages_and_edgar_text_are_stripped(capsys, tmp_path):
    def make(ev, g):
        edgar = FakeEdgar(ev)
        return conns(ev, edgar=edgar, tavily=FakeWeb(ev, kind="tavily", error=HttpError(500, HOSTILE)),
                     grounded=FakeGrounded(error=HOSTILE))

    rc, _ = run_research(make, ["--grounding"], tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 1 and TERMINAL_CONTROL_RE.search(out) is None and out.count("visible-text") == 2


def test_research_secret_straddling_the_truncation_limit_is_still_blanked(capsys, tmp_path):
    key = "tvly-SECRET-abcdef123456"
    # "RuntimeError: " is 14 characters; the key starts at ~199 and crosses the 220-character cut
    # (an HttpError would not do: it cuts its own body at 200 characters, before we ever see it)
    err = RuntimeError("y" * 185 + key + " tail")

    def make(ev, g):
        return conns(ev, tavily=FakeWeb(ev, kind="tavily", error=err))

    rc, _ = run_research(make, settings=research_settings(tavily_api_key=key), tmp_root=tmp_path)
    out = capsys.readouterr().out
    assert rc == 1 and key[:8] not in out and key[-8:] not in out and "***" in out


# --- the real build_connectors through main(): grounding gating --------------------------------------

GEMINI_KEY = "AIzaSy-FAKE-gemini-key-0000"
GROUNDED_PAYLOAD = {"candidates": [{
    "content": {"parts": [{"text": "Apple trades as AAPL on Nasdaq. " + HOSTILE}]},
    "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://g.example/a", "title": "A"}}],
                          "webSearchQueries": ["apple ticker " + HOSTILE]},
}]}


class StubResearchHttp:
    """Replaces `HttpClient` inside `check_research.main`: records every request, no network."""

    def __init__(self, post_error=None):
        self.gets: list[str] = []
        self.posts: list[tuple[str, dict, dict]] = []
        self.post_error = post_error
        self.closed = False

    async def get_text(self, url, *, params=None, headers=None, timeout=20.0):
        self.gets.append(url)
        return "# Example Domain\n\nThis domain is for illustrative examples. " + HOSTILE

    async def post_json(self, url, *, json=None, headers=None, timeout=20.0):
        self.posts.append((url, json, headers))
        if self.post_error is not None:
            raise self.post_error
        return GROUNDED_PAYLOAD

    async def aclose(self):
        self.closed = True


def run_real_main(monkeypatch, argv, tmp_path, *, gemini_key=GEMINI_KEY, post_error=None):
    stub = StubResearchHttp(post_error=post_error)
    monkeypatch.setattr(check_research, "HttpClient", lambda: stub)
    rc = check_research.main(list(argv), settings=research_settings(gemini_api_key=gemini_key), tmp_root=tmp_path)
    return rc, stub


def test_grounding_is_not_called_without_the_flag_even_with_a_gemini_key(monkeypatch, capsys, tmp_path):
    rc, stub = run_real_main(monkeypatch, [], tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert stub.posts == []  # no Gemini request, no quota spent
    assert stub.gets == ["https://r.jina.ai/https://example.com/"]
    assert "Gemini grounding" in out and "--grounding" in out and out.count("SKIPPED") == 3
    assert stub.closed is True and list(tmp_path.iterdir()) == []


def test_grounding_with_the_flag_makes_exactly_one_gemini_request(monkeypatch, capsys, tmp_path):
    rc, stub = run_real_main(monkeypatch, ["--grounding"], tmp_path)
    out = capsys.readouterr().out
    assert rc == 0 and len(stub.posts) == 1
    url, body, headers = stub.posts[0]
    assert url.startswith("https://generativelanguage.googleapis.com/") and "gemini-test" in url
    assert body["tools"] == [{"google_search": {}}] and headers == {"x-goog-api-key": GEMINI_KEY}
    assert "1 source" in out and "apple ticker" in out
    assert GEMINI_KEY not in out and TERMINAL_CONTROL_RE.search(out) is None
    assert out.count("SKIPPED") == 2  # EDGAR and Tavily only


def test_grounding_flag_without_a_gemini_key_is_skipped_not_called(monkeypatch, capsys, tmp_path):
    rc, stub = run_real_main(monkeypatch, ["--grounding"], tmp_path, gemini_key=None)
    out = capsys.readouterr().out
    assert rc == 0 and stub.posts == []
    assert "GEMINI_API_KEY" in out and out.count("SKIPPED") == 3


def test_a_grounding_http_error_is_a_failed_step_with_the_friendly_message(monkeypatch, capsys, tmp_path):
    rc, stub = run_real_main(monkeypatch, ["--grounding"], tmp_path, post_error=HttpError(429, "quota " + GEMINI_KEY))
    out = capsys.readouterr().out
    assert rc == 1 and len(stub.posts) == 1
    assert "billing" in out and "429" in out and GEMINI_KEY not in out
    assert list(tmp_path.iterdir()) == []


# --- build_connectors: the real classes, wired offline -------------------------------------------


def fresh_evidence():
    return EvidenceStore(AppDB(":memory:"))


def test_build_connectors_skips_what_has_no_key():
    c = check_research.build_connectors(research_settings(), httpx_free_http(), fresh_evidence(), with_grounding=False)
    assert c.edgar is None and c.tavily is None and c.grounded is None
    assert isinstance(c.jina, WebSearch)
    assert "SEC_USER_AGENT" in c.skips["edgar"] and "TAVILY_API_KEY" in c.skips["tavily"]
    assert "--grounding" in c.skips["grounding"]


def test_build_connectors_builds_the_real_classes_when_keys_exist():
    s = research_settings(sec_user_agent="Jane jane@example.com", tavily_api_key="tvly-x", gemini_api_key="g-key")
    c = check_research.build_connectors(s, httpx_free_http(), fresh_evidence(), with_grounding=True)
    assert isinstance(c.edgar, Edgar) and isinstance(c.tavily, WebSearch) and isinstance(c.grounded, GroundedWeb)
    c2 = check_research.build_connectors(s, httpx_free_http(), fresh_evidence(), with_grounding=False)
    assert c2.grounded is None and "--grounding" in c2.skips["grounding"]
    c3 = check_research.build_connectors(research_settings(), httpx_free_http(), fresh_evidence(), with_grounding=True)
    assert c3.grounded is None and "GEMINI_API_KEY" in c3.skips["grounding"]


class RecordingHttp:
    """Stands in for `HttpClient`: records what `WebSearch.fetch` asks for."""

    def __init__(self):
        self.gets: list[str] = []
        self.posts: list[str] = []

    async def get_text(self, url, *, params=None, headers=None, timeout=20.0):
        self.gets.append(url)
        return "# Example Domain\n\nThis domain is for use in illustrative examples."

    async def post_json(self, url, *, json=None, headers=None, timeout=20.0):
        self.posts.append(url)
        raise AssertionError("the Jina step must not use Tavily")


def httpx_free_http():
    return RecordingHttp()


async def test_the_fetch_step_uses_jina_even_when_a_tavily_key_is_configured():
    http = RecordingHttp()
    ev = fresh_evidence()
    s = research_settings(tavily_api_key="tvly-configured")
    c = check_research.build_connectors(s, http, ev, with_grounding=False)
    rep = check_research.Reporter()
    result = await check_research.check_jina(c, rep)
    assert result is True
    assert http.gets == ["https://r.jina.ai/https://example.com/"] and http.posts == []


# --------------------------------------------------------------------------- the scripts' own surface


def test_scripts_document_the_safety_flags():
    doc = (check_alpaca.__doc__ or "")
    assert "--yes-i-am-sure" in doc and "never" in doc.lower()
    # the rule: on a terminal PLACE is always required; the flag only matters without a terminal
    assert "even with --yes-i-am-sure" in " ".join(doc.split())
    assert check_alpaca.CONFIRM_WORD == "PLACE"


def test_the_order_leg_helper_is_pure_decimal():
    assert check_alpaca.half_price_limit(D("512.37")) == D("256.18")
    assert check_alpaca.half_price_limit(D("0.01")) == D("0.00")  # caller must refuse a zero limit
    with pytest.raises(TypeError):
        check_alpaca.half_price_limit(512.37)  # floats are never accepted for money
