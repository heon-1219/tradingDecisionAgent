"""Numeric grounding check: which numbers in an answer appear nowhere in the evidence?

The chat agent's rule is that every number in an answer comes from a tool output or from the user.
`unsupported_numbers(answer, corpus)` enforces it after the fact: `corpus` is the text of every tool
output in the turn plus the user's own message and the mandate JSON, and whatever comes back is
flagged in the UI (it is never silently shown as a fact, and never blocks the answer).

Both functions are pure and linear-time. The corpus may be ~200k characters, so the corpus is
scanned once into a sorted list of `Decimal` values and each answer number is one bisect. Money and
ratios never touch `float`.

What counts as a number (`extract_numbers`)
    optional sign, optional `$`, digits with thousands separators and decimals, then an optional
    suffix: `%`, `B`/`M`/`K`/`T`, `bn`/`mn`, the words billion/million/thousand/trillion, or the
    Korean units 만/억/조 (Korean answers write $997.35B as 9,973억). ISO dates (`2026-09-25`, also
    inside timestamps) are one token, not year, month and day.
    Skipped: evidence ids (`E12` - any digit run glued to a letter), list markers (`1.`, `2)` at the
    start of a line), times of day (`14:30`) and absurdly long digit runs (ids, hashes).

When is a number supported (`unsupported_numbers`)
    Values are compared by absolute value, so "down 3.4%" is supported by `-0.034`.
    * A corpus `%` token offers p and p/100; a scaled token (`$997.35B`) offers the scaled value and
      its bare mantissa (997.35); every date offers its year. Everything else offers itself.
    * An answer `3.4%` may match a corpus percent (3.4) or a corpus fraction (0.034); an answer
      `$1.2B` matches corpus values in billions; a bare number matches a bare corpus number.
    * Precision: the answer is compared at the decimals it displays, in its own unit. `$1.2B` is
      supported by any corpus value v with round(v / 1e9, 1) == 1.2. Half-up, half-even and
      truncation are all accepted, which makes the accepted corpus values one interval
      [x - u/2, x + u) around the answer value x, with u = one unit of its last displayed digit.
      An answer can therefore not claim more precision than the data has (`101.2575` is not
      supported by `101.25`).
    * Dates are supported by the same `YYYY-MM-DD` anywhere in the corpus.
    * Exempt: bare integers with |n| <= 10 and no unit ("top 3", "2 legs").
"""
from __future__ import annotations

import re
from bisect import bisect_left
from decimal import Context, Decimal

# Every token we parse is at most _MAX_MANTISSA_CHARS long, so 120 digits keep all arithmetic exact.
_CTX = Context(prec=120)
_MAX_MANTISSA_CHARS = 40

# Powers of ten for each suffix (lower-cased). `%` is handled separately.
_SCALE_EXP = {
    "k": 3, "thousand": 3,
    "m": 6, "mn": 6, "million": 6,
    "b": 9, "bn": 9, "billion": 9,
    "t": 12, "trillion": 12,
    "만": 4, "억": 8, "조": 12,  # 만 (10^4), 억 (10^8), 조 (10^12)
}

# One pass over the text. Alternatives are tried left to right at every position:
#   marker  a list marker at the start of a line ("1." / "2)") - consumed, never emitted
#   date    YYYY-MM-DD, with an optional ISO time tail that is consumed but not emitted
#   (time)  a time of day such as 9:30 or 16:00:00 - consumed, never emitted
#   num     the number itself
# `(?<![A-Za-z0-9])` in front of a number rejects digits glued to a letter (E12, SMA50, Q3) and,
# because it also rejects a digit, keeps the scan from restarting inside a digit run - that is what
# makes the scan linear. There are no nested unbounded quantifiers: the only repetition inside
# repetition is `(?:,\d{3})+`, whose body is fixed-width.
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
        (?P<mant>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?)
        (?P<suf>
            %
          | (?i:bn|mn)(?![A-Za-z])
          | [BMKT](?![A-Za-z])
          | [ ]?(?i:trillion|billion|million|thousand)(?![A-Za-z])
          | [만억조]
        )?
      )
    """,
    re.VERBOSE | re.MULTILINE,
)


def _scan(text: str):
    """Yield (kind, token, match) for every date and number in `text`, in order."""
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "date":
            yield "date", m["date"], m
        elif kind == "num" and len(m["mant"]) <= _MAX_MANTISSA_CHARS:
            yield "num", m.group(), m


def extract_numbers(text: str) -> list[str]:
    """Number tokens as they appear in `text`, in order, without duplicates.

    Evidence ids (`[E12]`), list markers, times of day and the parts of a date are not numbers.
    """
    out: dict[str, None] = {}
    for _, token, _m in _scan(text or ""):
        out.setdefault(token)
    return list(out)


def _mantissa(m: re.Match) -> tuple[Decimal, int]:
    """(absolute mantissa, number of displayed decimals) of a `num` match."""
    digits = m["mant"].replace(",", "")
    dot = digits.find(".")
    return Decimal(digits), (len(digits) - dot - 1 if dot >= 0 else 0)


def _suffix(m: re.Match) -> str:
    return (m["suf"] or "").strip().lower()


def _index_corpus(corpus: list[str]) -> tuple[list[Decimal], set[str]]:
    """Sorted list of every value the corpus offers, and the set of dates it mentions."""
    values: set[Decimal] = set()
    dates: set[str] = set()
    seen: set[str] = set()
    for text in corpus:
        if not text:
            continue
        for kind, token, m in _scan(text):
            if token in seen:
                continue
            seen.add(token)
            if kind == "date":
                dates.add(token)
                values.add(Decimal(token[:4]))
                continue
            mant, _ = _mantissa(m)
            suffix = _suffix(m)
            values.add(mant)
            if suffix == "%":
                values.add(mant.scaleb(-2, context=_CTX))
            elif suffix:
                values.add(mant.scaleb(_SCALE_EXP[suffix], context=_CTX))
    return sorted(values), dates


def _num_supported(m: re.Match, values: list[Decimal]) -> bool:
    mant, decimals = _mantissa(m)
    suffix = _suffix(m)
    if not suffix and not m["usd"] and decimals == 0 and mant <= 10:
        return True  # "top 3", "2 legs"
    unit = Decimal(1).scaleb(-decimals, context=_CTX)
    lo = _CTX.subtract(mant, Decimal(5).scaleb(-decimals - 1, context=_CTX))
    hi = _CTX.add(mant, unit)
    if suffix == "%":
        exps = (0, -2)  # 3.4% is a corpus 3.4 (percent) or a corpus 0.034 (fraction)
    else:
        exps = (_SCALE_EXP.get(suffix, 0),)
    for exp in exps:
        low, high = lo.scaleb(exp, context=_CTX), hi.scaleb(exp, context=_CTX)
        i = bisect_left(values, low)
        if i < len(values) and values[i] < high:
            return True
    return False


def unsupported_numbers(answer: str, corpus: list[str]) -> list[str]:
    """Numbers in `answer` that no text in `corpus` supports, in answer order, without duplicates."""
    values, dates = _index_corpus(corpus)
    out: dict[str, None] = {}
    seen: set[str] = set()
    for kind, token, m in _scan(answer or ""):
        if token in seen:
            continue
        seen.add(token)
        supported = token in dates if kind == "date" else _num_supported(m, values)
        if not supported:
            out[token] = None
    return list(out)
