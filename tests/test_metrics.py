"""Tests for `finagent.trading.metrics` - deterministic price statistics from daily bars.

Expected values for the 6-point hand-computed series are worked out with plain `Decimal`
arithmetic in this file's module-level constants (not by calling the functions under test), then
compared to the module's output within a tight tolerance - full bit-for-bit equality is not
asserted for the irrational-looking ones (division doesn't terminate in base 10 for most of these
ratios), since summation order inside the implementation is an implementation detail, not part of
the contract.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finagent.trading.metrics import (
    annualized_volatility,
    correlation,
    max_drawdown,
    simple_returns,
    sma,
    summarize,
    total_return,
)
from finagent.trading.models import Bar

TOL = Decimal("1e-18")


def _close(a: Decimal, b: Decimal, tol: Decimal = TOL) -> bool:
    return abs(a - b) <= tol


def _bar(t: str, c: Decimal) -> Bar:
    """A minimal bar for metrics tests: o/h/l/v don't matter to any function under test."""
    return Bar(t=t, o=c, h=c, l=c, c=c, v=Decimal("1000"))


def _bars_from_closes(closes: list[Decimal], start: datetime | None = None) -> list[Bar]:
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        _bar((start + timedelta(days=i)).isoformat(), Decimal(c))
        for i, c in enumerate(closes)
    ]


# --- hand-computed 6-point series --------------------------------------------------
# closes: 100, 102, 101, 105, 103, 108
SIX = [Decimal(x) for x in [100, 102, 101, 105, 103, 108]]

# r_i = c_i / c_{i-1} - 1, computed by hand (plain fraction arithmetic):
#   102/100 - 1 = 0.02
#   101/102 - 1 = -0.0098039215686274509803921569 (approx, non-terminating)
#   105/101 - 1 = 0.0396039603960396039603960396 (approx, non-terminating, repeating)
#   103/105 - 1 = -0.0190476190476190476190476190 (approx, non-terminating, repeating)
#   108/103 - 1 = 0.0485436893203883495145631068 (approx, non-terminating)
EXPECTED_RETURNS = [
    Decimal("0.02"),
    Decimal("-0.00980392156862745098039215686"),
    Decimal("0.0396039603960396039603960396"),
    Decimal("-0.0190476190476190476190476190"),
    Decimal("0.0485436893203883495145631068"),
]
# total_return = last/first - 1 = 108/100 - 1 = 0.08 exactly
EXPECTED_TOTAL_RETURN = Decimal("0.08")
# sample mean and sample (n-1) variance of the 5 returns above, then sqrt * sqrt(252):
EXPECTED_MEAN = Decimal("0.01585922182003629097510387422")
EXPECTED_SAMPLE_VAR = Decimal("0.0008815793844267254470922269025")
EXPECTED_ANNUALIZED_VOL = Decimal("0.4713364030875769441253739299")


def test_simple_returns_hand_computed():
    got = simple_returns(SIX)
    assert len(got) == 5
    for actual, expected in zip(got, EXPECTED_RETURNS):
        assert _close(actual, expected, Decimal("1e-15")), f"{actual} != {expected}"


def test_simple_returns_skips_zero_previous_close():
    closes = [Decimal("0"), Decimal("10"), Decimal("20")]
    got = simple_returns(closes)
    # first pair (0 -> 10) skipped since prev == 0; second pair kept.
    assert got == [Decimal("1")]


def test_total_return_hand_computed():
    assert total_return(SIX) == EXPECTED_TOTAL_RETURN


def test_total_return_fewer_than_two_points():
    assert total_return([]) == Decimal("0")
    assert total_return([Decimal("100")]) == Decimal("0")


def test_annualized_volatility_hand_computed():
    got = annualized_volatility(SIX)
    assert _close(got, EXPECTED_ANNUALIZED_VOL, Decimal("1e-15")), f"{got} != {EXPECTED_ANNUALIZED_VOL}"


def test_annualized_volatility_fewer_than_two_returns():
    assert annualized_volatility([]) == Decimal("0")
    assert annualized_volatility([Decimal("100")]) == Decimal("0")
    # exactly one return (two points) -> sample variance needs n-1 >= 1, but the brief calls for
    # "fewer than 2 returns" to yield 0; one return alone has undefined (n-1=0) sample variance.
    assert annualized_volatility([Decimal("100"), Decimal("110")]) == Decimal("0")


# --- max_drawdown -------------------------------------------------------------------


def test_max_drawdown_brief_example():
    closes = [Decimal(x) for x in [100, 120, 90, 130]]
    assert max_drawdown(closes) == Decimal("-0.25")


def test_max_drawdown_never_below_peak_is_zero():
    closes = [Decimal(x) for x in [100, 110, 120, 130]]
    assert max_drawdown(closes) == Decimal("0")


def test_max_drawdown_empty():
    assert max_drawdown([]) == Decimal("0")


# --- sma ------------------------------------------------------------------------


def test_sma_none_when_fewer_than_n_points():
    closes = [Decimal(x) for x in [1, 2, 3]]
    assert sma(closes, 5) is None


def test_sma_mean_of_last_n():
    closes = [Decimal(x) for x in [1, 2, 3, 4, 5]]
    assert sma(closes, 3) == Decimal("4")  # mean(3,4,5)


def test_sma_exact_n_points():
    closes = [Decimal(x) for x in [10, 20]]
    assert sma(closes, 2) == Decimal("15")


# --- correlation ------------------------------------------------------------------


def test_correlation_identical_series_is_one():
    a = [Decimal(x) for x in [1, 2, 3, 4, 5]]
    got = correlation(a, list(a))
    assert got is not None
    assert _close(got, Decimal("1"), Decimal("1e-12"))


def test_correlation_inverse_series_is_negative_one():
    a = [Decimal(x) for x in [1, 2, 3, 4, 5]]
    b = [Decimal(x) for x in [5, 4, 3, 2, 1]]
    got = correlation(a, b)
    assert got is not None
    assert _close(got, Decimal("-1"), Decimal("1e-12"))


def test_correlation_none_when_fewer_than_three_points():
    a = [Decimal(x) for x in [1, 2]]
    b = [Decimal(x) for x in [1, 2]]
    assert correlation(a, b) is None


def test_correlation_none_on_zero_variance():
    a = [Decimal("5"), Decimal("5"), Decimal("5")]
    b = [Decimal(x) for x in [1, 2, 3]]
    assert correlation(a, b) is None


def test_correlation_uses_overlapping_tail_of_shorter_length():
    # a has 6 points, b has 4: correlation should use the last 4 of each.
    a = [Decimal(x) for x in [999, 999, 1, 2, 3, 4]]
    b = [Decimal(x) for x in [1, 2, 3, 4]]
    got = correlation(a, b)
    assert got is not None
    assert _close(got, Decimal("1"), Decimal("1e-12"))


# --- float rejection ----------------------------------------------------------------


def test_simple_returns_rejects_float():
    with pytest.raises(TypeError):
        simple_returns([Decimal("1"), 2.0])


def test_total_return_rejects_float():
    with pytest.raises(TypeError):
        total_return([1.0, 2.0])


def test_annualized_volatility_rejects_float():
    with pytest.raises(TypeError):
        annualized_volatility([1.0, 2.0, 3.0])


def test_max_drawdown_rejects_float():
    with pytest.raises(TypeError):
        max_drawdown([1.0, 2.0])


def test_sma_rejects_float():
    with pytest.raises(TypeError):
        sma([1.0, 2.0, 3.0], 2)


def test_correlation_rejects_float():
    with pytest.raises(TypeError):
        correlation([1.0, 2.0, 3.0], [1, 2, 3])


# --- summarize --------------------------------------------------------------------

EXPECTED_SUMMARIZE_KEYS = {
    "last",
    "return_1m",
    "return_3m",
    "return_1y",
    "vol_1y",
    "max_dd_1y",
    "sma_50",
    "sma_200",
    "as_of",
    "bars",
}


def test_summarize_key_set_and_string_types():
    closes = [Decimal(100) + i for i in range(260)]
    bars = _bars_from_closes(closes)
    got = summarize(bars)
    assert set(got.keys()) == EXPECTED_SUMMARIZE_KEYS
    for key, value in got.items():
        assert value is None or isinstance(value, str), f"{key} -> {value!r} is not str/None"
    # with 260 bars, every window is fully populated (>= 252 + 1 points for 1y, >= 200 for sma_200)
    for key in EXPECTED_SUMMARIZE_KEYS:
        assert got[key] is not None, f"{key} unexpectedly None"
    assert got["bars"] == "260"
    assert got["last"] == format(Decimal(100 + 259).quantize(Decimal("0.01")), "f")


def test_summarize_short_history_returns_none_for_unmet_windows():
    closes = [Decimal(x) for x in [100, 101, 102, 103, 105]]
    bars = _bars_from_closes(closes)
    got = summarize(bars)
    assert set(got.keys()) == EXPECTED_SUMMARIZE_KEYS
    # only 5 points: sma_50/sma_200 need more history than we have.
    assert got["sma_50"] is None
    assert got["sma_200"] is None
    # but return/vol/drawdown windows fall back to whatever history is available (>= 2 points).
    assert got["return_1m"] is not None
    assert got["return_3m"] is not None
    assert got["return_1y"] is not None
    assert got["vol_1y"] is not None
    assert got["max_dd_1y"] is not None
    assert got["bars"] == "5"
    assert got["as_of"] == bars[-1].t


def test_summarize_single_bar_all_return_metrics_none():
    bars = _bars_from_closes([Decimal("100")])
    got = summarize(bars)
    assert got["return_1m"] is None
    assert got["return_3m"] is None
    assert got["return_1y"] is None
    assert got["vol_1y"] is None
    assert got["max_dd_1y"] is None
    assert got["sma_50"] is None
    assert got["sma_200"] is None
    assert got["last"] == format(Decimal("100.00"), "f")
    assert got["bars"] == "1"


def test_summarize_sorts_bars_by_timestamp():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # deliberately out of order
    bars = [
        _bar((start + timedelta(days=2)).isoformat(), Decimal("103")),
        _bar((start + timedelta(days=0)).isoformat(), Decimal("100")),
        _bar((start + timedelta(days=1)).isoformat(), Decimal("101")),
    ]
    got = summarize(bars)
    assert got["last"] == format(Decimal("103.00"), "f")
    assert got["as_of"] == (start + timedelta(days=2)).isoformat()
