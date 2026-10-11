"""Numeric grounding check: which numbers in an answer appear nowhere in the evidence?

The chat agent's rule is that every number in an answer comes from a tool output or from the user.
`unsupported_numbers(answer, corpus)` enforces it after the fact: `corpus` is the text of every tool
output in the turn plus the user's own message and the mandate JSON, and whatever comes back is
flagged in the UI (it is never silently shown as a fact, and never blocks the answer).

Both functions are pure and linear-time. The corpus may be ~200k characters, so the corpus is
scanned once into sorted lists of `Decimal` values and each answer number is a couple of bisects.
Money and ratios never touch `float`.

What counts as a number (`extract_numbers`)
    optional sign, optional `$`, digits with thousands separators and decimals (`.5` too), then an
    optional unit:
    * percent: `%`, ` percent`, ` pct`;
    * scale: `B`/`M`/`K`/`T` (uppercase, and lowercase `k`), `bn`/`mn`, the words billion, million,
      thousand and trillion, and the Korean 만/억/조 - `m` alone is minutes, so it is not a scale;
    * currency: a `$`, or a currency word after the number (` USD`, ` dollars`, 달러, 원), which may
      follow a scale (`1.5 trillion dollars`, `3조원`).
    ISO dates (`2026-09-25`, also inside timestamps) are one token, not year, month and day.
    Skipped: evidence ids (`E12` - any digit run glued to a letter), list markers (`1.`, `2)` at the
    start of a line), times of day (`14:30`), the digits of `제3조` (article 3: a Korean unit right
    after a number preceded by 제 is not an amount; nor is one followed by a Hangul syllable other than
    원, as in `3만큼`) and absurdly long digit runs (ids, hashes).

When is a number supported (`unsupported_numbers`)
    Every token is a percent token, a currency token or a plain token, and values are compared by
    absolute value:
    * a percent answer (`3.4%`) is supported by corpus percent tokens (compared as percent), by
      plain corpus numbers below 1 read as a fraction (0.034 -> 3.4%), and by a `..._pct` /
      `percent...` key with a number in the corpus (`"max_position_pct": "20"`); it is not supported
      by plain numbers of 1 or more, nor by currency or scaled numbers;
    * a currency answer (`$1.2B`) and a plain answer (`101.3`) are supported by corpus currency
      tokens and plain tokens, scaled by their own suffix, but never by percent tokens. The bare
      mantissa of a scaled corpus number (997.35 of `$997.35B`) supports nothing on its own: `$997.35`
      is not supported by `$997.35B`, while `$997.35 billion` and `997.35B` are.
    * precision: the answer is compared at the decimals it displays, in its own unit. `$1.2B` is
      supported by any corpus value v with round(v / 1e9, 1) == 1.2. With at least one displayed
      decimal, half-up, half-even and truncation are all accepted, which makes the accepted corpus
      values the interval [x - u/2, x + u) around the answer value x (u is one unit of its last
      displayed digit). With no decimals only rounding is accepted (`12%` is not supported by 12.7%):
      [x - 1/2, x + 1/2), plus x + 1/2 itself when x is even. An answer can therefore not claim more
      precision than the data has (`101.2575` is not supported by `101.25`).
    * dates are supported by the same `YYYY-MM-DD` anywhere in the corpus (also inside a timestamp).
      So that "September 25, 2026" and `09/25/2026` pass, a corpus date also offers its day and its
      month to a bare integer from 1 to 31 (no `$`, `%`, separator or decimal), and its year to a
      bare four-digit integer (`2026`, not `$2,026` or `2,026`).
    * exempt: bare integers with |n| <= 10 and no unit ("top 3", "2 legs").

Limits
    This is a heuristic that flags likely-unsupported numbers; it is advisory and not a guarantee.
    * Low-precision numbers are weakly checked against a large corpus: a bare integer such as `150`
      is supported by any corpus value that rounds to 150, and a bare day, month or year by any
      corpus date. The unit classes keep percents from being backed by unrelated prices and
      volumes, but plain integers are not protected that way.
    * A ratio of 1 or more (a return of 1.5 for 150%) is not read as a fraction.
    * Numbers the model computes itself (a cost, a difference, "about 3.4% above the average") are
      flagged by design; the prompt tells it to quote tool-computed values instead.
    * Korean compounds such as `1조 2,000억` are read as two separate amounts, and a unit followed by
      a particle (`3억이`) is not read as an amount.
    * Digits inside identifiers, `S&P 500`, `3M` (the company, read as 3 million) and a day of the
      month written as a word ("Sep 25" when the corpus has no such date) can be flagged or
      supported by accident.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from decimal import Context, Decimal
from typing import NamedTuple

# Every token we parse is at most _MAX_MANTISSA_CHARS long, so 120 digits keep all arithmetic exact.
_CTX = Context(prec=120)
_MAX_MANTISSA_CHARS = 40

# Powers of ten for each scale suffix (lower-cased). `%` and currency words are not scales.
_SCALE_EXP = {
    "k": 3, "thousand": 3,
    "m": 6, "mn": 6, "million": 6,
    "b": 9, "bn": 9, "billion": 9,
    "t": 12, "trillion": 12,
    "만": 4, "억": 8, "조": 12,  # 만 (10^4), 억 (10^8), 조 (10^12)
}
_KOREAN_SCALES = frozenset("만억조")
_ARTICLE = "제"  # 제: "제3조" is "article 3", not 3 x 10^12

# One pass over the text. Alternatives are tried left to right at every position:
#   marker  a list marker at the start of a line ("1." / "2)") - consumed, never emitted
#   date    YYYY-MM-DD, with an optional ISO time tail that is consumed but not emitted
#   (time)  a time of day such as 9:30 or 16:00:00 - consumed, never emitted
#   num     the number itself: sign, $, mantissa, then either a percent unit or an optional scale
#           followed by an optional currency word
# `(?<![A-Za-z0-9])` in front of a number rejects digits glued to a letter (E12, SMA50, Q3) and, because
# it also rejects a digit, keeps the scan from restarting inside a digit run. There are no nested
# unbounded quantifiers: the only repetition of a repetition is `(?:,\d{3})+`, whose body is fixed-width.
_TOKEN_RE = re.compile(
    r"""
      (?P<marker>^[ \t]*\d{1,3}[.)](?=[ \t]))
    | (?P<date>(?<![A-Za-z0-9])\d{4}-\d{2}-\d{2})(?!\d)
      (?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+\-]\d{2}(?::?\d{2})?)?)?
    | (?<![A-Za-z0-9:])\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?!\d)(?:Z|[+\-]\d{2}:?\d{2})?
    | (?<![A-Za-z0-9])
      (?P<num>
        (?P<sign>[+\-−])?
        (?P<usd>\$)?
        (?P<mant>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|(?<!\.)\.\d+)
        (?:
            (?P<pct>%|[ ]?(?i:percent|pct)(?![A-Za-z]))
          |
            (?P<scale>
                [ ]?(?i:trillion|billion|million|thousand)(?![A-Za-z])
              | [ ]?(?i:bn|mn)(?![A-Za-z])
              | [BMKTk](?![A-Za-z])
              | [만억조](?:(?=원)|(?![가-힣]))
            )?
            (?P<cur>[ ]?(?i:usd|dollars?)(?![A-Za-z])|[ ]?(?:달러|원))?
        )
      )
    """,
    re.VERBOSE | re.MULTILINE,
)

# A number that follows a percent-valued key: "max_position_pct": "20", price_band_pct = 5, percent_change: 2.5.
# The gaps are bounded, so a long run of "pct" cannot make the scan quadratic.
_PCT_KEY_RE = re.compile(
    r"""(?i:pct|percent)[A-Za-z_]{0,20}["']?[ \t]{0,4}[:=][ \t]{0,4}["']?(\d{1,12}(?:\.\d{1,12})?)"""
)


class _Num(NamedTuple):
    mant: Decimal      # absolute mantissa
    decimals: int      # decimals displayed in the mantissa
    kind: str          # "pct", "cur" or "plain"
    exp: int | None    # power of ten of the scale suffix, None when unscaled
    bare_int: bool     # digits only: no separator, no decimal point
    ndigits: int


def _scan(text: str):
    """Yield (kind, token, match, trimmed) for every date and number in `text`, in order."""
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "date":
            yield "date", m["date"], m, False
        elif kind == "num" and len(m["mant"]) <= _MAX_MANTISSA_CHARS:
            scale = m["scale"]
            if scale in _KOREAN_SCALES and m.start() > 0 and text[m.start() - 1] == _ARTICLE:
                # 제3조: the digits are an article number, not an amount - keep them, drop the unit
                yield "num", m.group()[: m.start("scale") - m.start()], m, True
            else:
                yield "num", m.group(), m, False


def extract_numbers(text: str) -> list[str]:
    """Number tokens as they appear in `text`, in order, without duplicates.

    Evidence ids (`[E12]`), list markers, times of day and the parts of a date are not numbers.
    """
    out: dict[str, None] = {}
    for _, token, _m, _trimmed in _scan(text or ""):
        out.setdefault(token)
    return list(out)


def _parse(m: re.Match, trimmed: bool) -> _Num:
    text = m["mant"]
    digits = text.replace(",", "")
    dot = digits.find(".")
    pct = cur = False
    exp = None
    if not trimmed:
        if m["pct"]:
            pct = True
        else:
            if m["scale"]:
                exp = _SCALE_EXP[m["scale"].strip().lower()]
            cur = bool(m["cur"])
    cur = cur or bool(m["usd"])
    return _Num(
        Decimal(digits),
        len(digits) - dot - 1 if dot >= 0 else 0,
        "pct" if pct else "cur" if cur else "plain",
        exp,
        "," not in text and dot < 0,
        len(digits),
    )


class _Index(NamedTuple):
    nonpct: list[Decimal]   # plain and currency values, scaled (support plain and currency answers)
    pct: list[Decimal]      # percent-token values, in percent units
    frac: list[Decimal]     # plain unscaled values below 1, times 100 (support percent answers)
    dates: set[str]
    years: set[int]
    day_month: set[int]


def _index_corpus(corpus: list[str]) -> _Index:
    """Every value the corpus offers, by unit class, and the dates it mentions."""
    nonpct: set[Decimal] = set()
    pct: set[Decimal] = set()
    frac: set[Decimal] = set()
    dates: set[str] = set()
    years: set[int] = set()
    day_month: set[int] = set()
    seen: set[str] = set()
    for text in corpus:
        if not text:
            continue
        for kind, token, m, trimmed in _scan(text):
            if token in seen:
                continue
            seen.add(token)
            if kind == "date":
                dates.add(token)
                years.add(int(token[:4]))
                day_month.update((int(token[5:7]), int(token[8:10])))
                continue
            n = _parse(m, trimmed)
            if n.kind == "pct":
                pct.add(n.mant)
                continue
            nonpct.add(n.mant.scaleb(n.exp, context=_CTX) if n.exp else n.mant)
            if n.kind == "plain" and n.exp is None and n.mant < 1:
                frac.add(n.mant.scaleb(2, context=_CTX))
        for key in _PCT_KEY_RE.finditer(text):
            pct.add(Decimal(key[1]))
    return _Index(sorted(nonpct), sorted(pct), sorted(frac), dates, years, day_month)


def _accepts(values: list[Decimal], n: _Num, exp: int) -> bool:
    """Does some value round to the answer's mantissa at the decimals it displays (scaled by 10^exp)?"""
    half = Decimal(5).scaleb(-n.decimals - 1, context=_CTX)
    lo = _CTX.subtract(n.mant, half)
    # Truncation is only accepted when at least one decimal is shown; "12" is not a truncation of 12.7.
    hi = _CTX.add(n.mant, Decimal(1).scaleb(-n.decimals, context=_CTX) if n.decimals else half)
    low, high = lo.scaleb(exp, context=_CTX), hi.scaleb(exp, context=_CTX)
    i = bisect_left(values, low)
    if i < len(values) and values[i] < high:
        return True
    if not n.decimals and int(n.mant) % 2 == 0:  # half-even: 12.5 rounds to 12
        j = bisect_left(values, high)
        return j < len(values) and values[j] == high
    return False


def _num_supported(n: _Num, ix: _Index) -> bool:
    if n.kind == "plain" and n.exp is None and n.decimals == 0:
        if n.mant <= 10:
            return True  # "top 3", "2 legs"
        if n.bare_int:
            if n.mant <= 31 and int(n.mant) in ix.day_month:
                return True  # "September 25, 2026" against a corpus 2026-09-25
            if n.ndigits == 4 and int(n.mant) in ix.years:
                return True
    if n.kind == "pct":
        return _accepts(ix.pct, n, 0) or _accepts(ix.frac, n, 0)
    return _accepts(ix.nonpct, n, n.exp or 0)


def unsupported_numbers(answer: str, corpus: list[str]) -> list[str]:
    """Numbers in `answer` that no text in `corpus` supports, in answer order, without duplicates."""
    ix = _index_corpus(corpus)
    out: dict[str, None] = {}
    seen: set[str] = set()
    for kind, token, m, trimmed in _scan(answer or ""):
        if token in seen:
            continue
        seen.add(token)
        supported = token in ix.dates if kind == "date" else _num_supported(_parse(m, trimmed), ix)
        if not supported:
            out[token] = None
    return list(out)
