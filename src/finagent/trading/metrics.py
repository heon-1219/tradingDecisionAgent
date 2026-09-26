"""Deterministic price statistics, computed from daily `Bar` history.

The chat agent must never compute a number itself - it calls a tool, and this module is the
tool's arithmetic. Every function here is pure and stdlib-only (`decimal`; no `math`, no
third-party numeric library), so a reviewer can read the exact formula being executed instead of
trusting an LLM's mental math.

Money/price rule (see the project's global constraints): `Decimal`, never `float`. `float` inputs
are rejected outright with `TypeError` rather than silently accepted - a caller that passes a
`float` closes list has already lost precision before this module ever sees it, and accepting it
quietly would let that imprecision leak into a supposedly-exact computation. `int` and `Decimal`
are both accepted.

Local decimal context: every arithmetic operation in this module runs inside
`with localcontext(_CTX):`, a copy of the ambient context at import time with `prec=28`.
`localcontext` installs a *copy* of `_CTX` as the thread's current context for the duration of the
`with` block and restores the previous context on exit, so nothing here ever mutates the global
decimal context that other modules (or the interactive interpreter) may be relying on.
"""
from __future__ import annotations

from decimal import Decimal, getcontext, localcontext
from typing import Sequence

from finagent.trading.models import Bar

_CTX = getcontext().copy()
_CTX.prec = 28

# Trading-day windows used by `summarize`, expressed as "bars back" from the last close - e.g.
# 1m = 21 means the return window spans the last close and the close 21 bars before it (22 points).
_WINDOW_1M = 21
_WINDOW_3M = 63
_WINDOW_1Y = 252


def _reject_float(values: Sequence[object]) -> None:
    for v in values:
        if isinstance(v, float):
            raise TypeError(
                "float is not allowed for price/return data in trading.metrics; use Decimal or int."
            )


def simple_returns(closes: Sequence[Decimal]) -> list[Decimal]:
    """r_i = c_i / c_{i-1} - 1 for each consecutive pair. A pair whose previous close is 0 is
    skipped (a 0 previous close makes the return undefined, not infinite)."""
    _reject_float(closes)
    with localcontext(_CTX):
        out: list[Decimal] = []
        for i in range(1, len(closes)):
            prev = Decimal(closes[i - 1])
            if prev == 0:
                continue
            cur = Decimal(closes[i])
            out.append(cur / prev - 1)
        return out


def total_return(closes: Sequence[Decimal]) -> Decimal:
    """last/first - 1 over the whole series. `Decimal("0")` for fewer than 2 points, or when the
    first close is 0 (undefined, not infinite)."""
    _reject_float(closes)
    if len(closes) < 2:
        return Decimal("0")
    with localcontext(_CTX):
        first = Decimal(closes[0])
        if first == 0:
            return Decimal("0")
        last = Decimal(closes[-1])
        return last / first - 1


def annualized_volatility(closes: Sequence[Decimal], periods_per_year: int = 252) -> Decimal:
    """Sample standard deviation (n-1) of `simple_returns(closes)`, scaled by
    `sqrt(periods_per_year)`. `Decimal("0")` if there are fewer than 2 returns (sample variance is
    undefined with fewer than 2 data points)."""
    _reject_float(closes)
    with localcontext(_CTX):
        returns = simple_returns(closes)
        n = len(returns)
        if n < 2:
            return Decimal("0")
        mean = sum(returns) / Decimal(n)
        variance = sum((r - mean) ** 2 for r in returns) / Decimal(n - 1)
        return variance.sqrt() * Decimal(periods_per_year).sqrt()


def max_drawdown(closes: Sequence[Decimal]) -> Decimal:
    """The most negative (price / running peak - 1) seen across the series, as a negative
    fraction (or `Decimal("0")` if the price never drops below a prior peak, or the series is
    empty)."""
    _reject_float(closes)
    if not closes:
        return Decimal("0")
    with localcontext(_CTX):
        peak = Decimal(closes[0])
        worst = Decimal("0")
        for c in closes:
            price = Decimal(c)
            if price > peak:
                peak = price
            if peak == 0:
                continue
            drawdown = price / peak - 1
            if drawdown < worst:
                worst = drawdown
        return worst


def sma(closes: Sequence[Decimal], n: int) -> Decimal | None:
    """Mean of the last `n` closes, or `None` if there are fewer than `n` points."""
    _reject_float(closes)
    if len(closes) < n:
        return None
    with localcontext(_CTX):
        tail = closes[-n:]
        return sum((Decimal(c) for c in tail), start=Decimal("0")) / Decimal(n)


def correlation(a: Sequence[Decimal], b: Sequence[Decimal]) -> Decimal | None:
    """Pearson correlation over the overlapping tail of `a` and `b` (the last `min(len(a),
    len(b))` points of each). `None` if that overlap is fewer than 3 points, or either series has
    zero variance over the overlap (correlation is undefined, not infinite, against a constant
    series)."""
    _reject_float(a)
    _reject_float(b)
    n = min(len(a), len(b))
    if n < 3:
        return None
    with localcontext(_CTX):
        a_tail = [Decimal(x) for x in a[-n:]]
        b_tail = [Decimal(x) for x in b[-n:]]
        mean_a = sum(a_tail) / Decimal(n)
        mean_b = sum(b_tail) / Decimal(n)
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a_tail, b_tail))
        var_a = sum((x - mean_a) ** 2 for x in a_tail)
        var_b = sum((y - mean_b) ** 2 for y in b_tail)
        if var_a == 0 or var_b == 0:
            return None
        denom = (var_a * var_b).sqrt()
        return cov / denom


def _window_slice(closes: list[Decimal], bars_back: int) -> list[Decimal]:
    """The last `bars_back + 1` closes, or the whole series if shorter - "use the available
    history if shorter" from the brief. Python's own negative-slice clipping does this for free:
    `closes[-(bars_back + 1):]` clips to `closes[0:]` once `bars_back + 1` exceeds `len(closes)`."""
    return closes[-(bars_back + 1):]


def _fmt(value: Decimal | None, places: int) -> str | None:
    if value is None:
        return None
    with localcontext(_CTX):
        quantum = Decimal(1).scaleb(-places)
        return format(value.quantize(quantum), "f")


def summarize(bars: list[Bar]) -> dict:
    """Deterministic price-statistics summary for a chat tool: every value is a formatted string
    (or `None`), never a bare number, so the numbers displayed to the user are exactly the digits
    this function produced - no LLM re-formatting in between.

    Bars are sorted by `t` first (callers may hand this an unsorted page of history). Windows are
    counted in trading days (bars), not calendar days: 1m = 21, 3m = 63, 1y = 252 bars back from
    the last close. A window with fewer than 2 available points (including an empty `bars` list)
    yields `None` for every metric derived from it, rather than falling back to a partial-history
    computation that would be misleadingly presented as a full window.

    Returns, volatility and drawdown are formatted to 4 decimal places as fractions (e.g.
    "0.0500" for 5%); prices (`last`, `sma_50`, `sma_200`) to 2 decimal places. `as_of` and `bars`
    are extra, ungated keys: the last bar's timestamp and the bar count, respectively.
    """
    sorted_bars = sorted(bars, key=lambda b: b.t)
    closes = [b.c for b in sorted_bars]

    if not closes:
        return {
            "last": None,
            "return_1m": None,
            "return_3m": None,
            "return_1y": None,
            "vol_1y": None,
            "max_dd_1y": None,
            "sma_50": None,
            "sma_200": None,
            "as_of": None,
            "bars": str(len(sorted_bars)),
        }

    def _windowed_return(bars_back: int) -> Decimal | None:
        window = _window_slice(closes, bars_back)
        return total_return(window) if len(window) >= 2 else None

    window_1y = _window_slice(closes, _WINDOW_1Y)
    vol_1y = annualized_volatility(window_1y) if len(window_1y) >= 2 else None
    max_dd_1y = max_drawdown(window_1y) if len(window_1y) >= 2 else None

    return {
        "last": _fmt(closes[-1], 2),
        "return_1m": _fmt(_windowed_return(_WINDOW_1M), 4),
        "return_3m": _fmt(_windowed_return(_WINDOW_3M), 4),
        "return_1y": _fmt(_windowed_return(_WINDOW_1Y), 4),
        "vol_1y": _fmt(vol_1y, 4),
        "max_dd_1y": _fmt(max_dd_1y, 4),
        "sma_50": _fmt(sma(closes, 50), 2),
        "sma_200": _fmt(sma(closes, 200), 2),
        "as_of": sorted_bars[-1].t,
        "bars": str(len(sorted_bars)),
    }
