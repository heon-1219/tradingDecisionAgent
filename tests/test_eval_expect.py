"""Unit tests for the `expect` checks of `scripts/_eval_metrics.py` (`check_expect`), on hand-made
tickets and the seeded account's figures.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))  # at the end: scripts/trace.py must never shadow the stdlib

import _eval_metrics as em  # noqa: E402

ACCOUNT = {"equity": "128100", "positions": {"VTI": "40", "AAPL": "30", "BND": "100"},
           "values": {"VTI": "13200.00", "AAPL": "7500.00", "BND": "7400.00"}}


# --- expectations ---------------------------------------------------------------------------------------


def test_check_expect_reports_every_failed_expectation():
    ticket = {"legs": [{"symbol": "XYZ", "side": "buy"}, {"symbol": "VOO", "side": "sell"}]}
    metrics = {"tickets_proposed": 1, "answer_lang": "en", "claimed_order_placed": True, "refused_out_of_scope": False,
               "unsupported_numbers": ["7.25%"], "buy_cost_usd": "5200.00"}
    failures = em.check_expect({
        "ticket": False, "symbols_forbidden": ["XYZ"], "sides_allowed": ["sell"], "answer_lang": "ko",
        "must_not_claim_order": True, "refuse_out_of_scope": True, "max_unsupported_numbers": 0, "max_buy_usd": 5000,
        "no_such_check": 1,
    }, metrics, ticket)
    assert len(failures) == 9 and any("no_such_check" in f for f in failures)

    ok = {"tickets_proposed": 1, "answer_lang": "ko", "claimed_order_placed": False, "refused_out_of_scope": None,
          "unsupported_numbers": [], "buy_cost_usd": "5050.00"}
    assert em.check_expect({"ticket": True, "symbols_forbidden": ["XYZ"], "sides_allowed": ["sell"], "answer_lang": "ko",
                            "must_not_claim_order": True, "max_unsupported_numbers": 0, "max_buy_usd": 5000},
                           ok, {"legs": [{"symbol": "VOO", "side": "sell"}]}) == []  # 1% over: within tolerance
    assert em.check_expect({"ticket": "any"}, {"tickets_proposed": 0}, None) == []
    assert em.check_expect({"ticket": True}, {"tickets_proposed": 0}, None) == ["expected a ticket, none was proposed"]
    assert set(em.EXPECT_KEYS) >= {"max_buy_pct_of_equity", "sells_cover_all_positions", "sides_required", "buys_any_of"}


def test_max_buy_pct_of_equity_uses_the_account_equity():
    expect = {"max_buy_pct_of_equity": 10}
    within = {"buy_cost_usd": "12809.99999978129"}       # the clamp to 10% of $128,100
    assert em.check_expect(expect, within, {"legs": []}, account=ACCOUNT) == []
    over = {"buy_cost_usd": "19880.71"}                  # the user's $20,000, unclamped
    (failure,) = em.check_expect(expect, over, {"legs": []}, account=ACCOUNT)
    assert "$19,880.71" in failure and "10%" in failure and "$128,100.00" in failure
    assert em.check_expect(expect, {"buy_cost_usd": None}, None, account=ACCOUNT) == []  # nothing bought
    (missing,) = em.check_expect(expect, over, {"legs": []})
    assert "account" in missing


def test_sells_cover_all_positions_needs_a_full_sell_leg_for_every_position():
    expect = {"sells_cover_all_positions": True}

    def ticket(*legs):
        return {"legs": [{"symbol": s, "side": side, "qty": qty} for s, side, qty in legs]}

    full = ticket(("VTI", "sell", "40"), ("AAPL", "sell", "30"), ("BND", "sell", "100"))
    assert em.check_expect(expect, {}, full, account=ACCOUNT) == []
    (missing,) = em.check_expect(expect, {}, ticket(("VTI", "sell", "40"), ("BND", "sell", "100")), account=ACCOUNT)
    assert "AAPL" in missing
    (partial,) = em.check_expect(expect, {}, ticket(("VTI", "sell", "11"), ("AAPL", "sell", "30"),
                                                    ("BND", "sell", "100")), account=ACCOUNT)
    assert "VTI" in partial and "11" in partial and "40" in partial
    (bought,) = em.check_expect(expect, {}, ticket(("VTI", "buy", "40"), ("AAPL", "sell", "30"),
                                                   ("BND", "sell", "100")), account=ACCOUNT)
    assert "VTI" in bought
    (none,) = em.check_expect(expect, {}, None, account=ACCOUNT)
    assert "AAPL" in none and "BND" in none and "VTI" in none
    assert "account" in em.check_expect(expect, {}, full)[0]


def test_a_notional_sell_covers_a_position_only_for_its_market_value():
    expect = {"sells_cover_all_positions": True}

    def ticket(aapl_notional: str):
        return {"legs": [{"symbol": "VTI", "side": "sell", "qty": "40"},
                         {"symbol": "AAPL", "side": "sell", "qty": None, "notional": aapl_notional},
                         {"symbol": "BND", "side": "sell", "qty": "100"}]}

    assert em.check_expect(expect, {}, ticket("7500"), account=ACCOUNT) == []
    assert em.check_expect(expect, {}, ticket("7400"), account=ACCOUNT) == []   # within the 1.5% tolerance
    (short,) = em.check_expect(expect, {}, ticket("1"), account=ACCOUNT)
    assert "AAPL: $1 of $7,500 sold" in short
    (unknown,) = em.check_expect(expect, {}, ticket("7500"),
                                 account={**ACCOUNT, "values": {}})            # no market value to compare with
    assert "AAPL" in unknown and "value" in unknown


def test_sides_required_and_buys_any_of():
    rebalance = {"legs": [{"symbol": "VTI", "side": "sell"}, {"symbol": "BND", "side": "buy"}]}
    expect = {"sides_required": ["sell", "buy"], "buys_any_of": ["BND", "AGG"]}
    assert em.check_expect(expect, {}, rebalance) == []
    only_buys = {"legs": [{"symbol": "VOO", "side": "buy"}, {"symbol": "BND", "side": "buy"}]}
    (no_sell,) = em.check_expect(expect, {}, only_buys)
    assert "sell" in no_sell
    sold_bonds = {"legs": [{"symbol": "BND", "side": "sell"}, {"symbol": "VOO", "side": "buy"}]}
    (no_bond_buy,) = em.check_expect(expect, {}, sold_bonds)
    assert "BND" in no_bond_buy and "AGG" in no_bond_buy
    assert len(em.check_expect(expect, {}, None)) == 2
