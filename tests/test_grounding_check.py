"""Tests for `finagent.chat.grounding_check`: number extraction and numeric grounding.

Both functions are pure (strings in, strings out), so there is no fixture world here - just text.
The corpus in real use is the tool outputs of one turn plus the user's message and the mandate JSON;
the tests use small stand-ins for those, in the same number formats the toolbox prints
(`$1,234.56`, `$101.2575`, `+3.40%`, `$997.35B`) and raw fractions (`0.034`).
"""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from finagent.chat.grounding_check import extract_numbers, unsupported_numbers
from finagent.chat.toolbox import _compact_usd, _pct_from_fraction, _price, _usd

# --------------------------------------------------------------------------- extract_numbers


def test_extract_covers_the_documented_forms():
    text = "Price $1,234.56, up 12.5%, cap 1.2B, change -3.4 on 2026-09-25 with 150 shares."
    assert extract_numbers(text) == ["$1,234.56", "12.5%", "1.2B", "-3.4", "2026-09-25", "150"]


def test_extract_ignores_evidence_ids():
    assert extract_numbers("Apple beat estimates [E12] and guided up [E3, E4][E5].") == []
    assert extract_numbers("[E12] revenue rose 15%") == ["15%"]
    assert extract_numbers('<untrusted source="E7,E8">up 2.5%</untrusted>') == ["2.5%"]


def test_iso_date_is_one_token_not_year_month_day():
    assert extract_numbers("As of 2026-09-25 close") == ["2026-09-25"]


def test_iso_timestamp_yields_just_the_date():
    text = "quote at 2026-09-25T14:30:00Z and 2026-09-26T09:31:02.123456+00:00"
    assert extract_numbers(text) == ["2026-09-25", "2026-09-26"]


def test_time_of_day_is_not_a_number():
    assert extract_numbers("Opens 9:30 ET, closes 16:00:00.") == []
    assert extract_numbers("On 2026-09-25 14:30:00 the price was 101.25") == ["2026-09-25", "101.25"]


@pytest.mark.parametrize("token", [
    "$997.35B", "$1.20M", "$12.50K", "$3.10T",
    "1.2bn", "3mn", "2.5 billion", "4 million", "10 thousand", "1.1 trillion", "7 Million",
    "9,973억", "3만", "2조",
    "+3.40%", "-1.25%", "12.5%", "-$1,234.56", "−3.4%", "$1,234,567",
])
def test_extract_keeps_the_unit_inside_the_token(token):
    assert extract_numbers(f"x {token} y") == [token]


def test_letters_after_a_number_are_not_units():
    # lowercase m is minutes here, "Bonds" is a word, "MB" is megabytes, "-day" is a suffix word
    assert extract_numbers("15m bars, 5 Bonds, 3MB, 50-day") == ["15", "5", "3", "50"]


def test_digits_glued_to_letters_are_skipped():
    assert extract_numbers("SMA50 H2O Q3 FY2025 E12 x5") == []
    # an underscore is not a letter: the toolbox prints sma_50
    assert extract_numbers("sma_50: 101.25") == ["50", "101.25"]


def test_list_markers_are_ignored():
    text = "1. Buy the dip\n2) Wait for 3.4% pullback\n   3. Size at $500\n10. Done"
    assert extract_numbers(text) == ["3.4%", "$500"]


def test_a_decimal_at_the_start_of_a_line_is_not_a_list_marker():
    assert extract_numbers("3.4% was the move\n2.5 is the ratio") == ["3.4%", "2.5"]


def test_sentence_final_period_is_not_part_of_the_number():
    assert extract_numbers("It closed at 101.") == ["101"]


def test_ranges_split_into_two_unsigned_numbers():
    assert extract_numbers("range 10-15% and 100-day") == ["10", "15%", "100"]


def test_signed_numbers_keep_their_sign():
    assert extract_numbers("(-3.4%) and +2 and -7") == ["-3.4%", "+2", "-7"]


def test_results_are_deduplicated_in_order():
    assert extract_numbers("5% then 12.5% then 5% again, 12.5%") == ["5%", "12.5%"]


def test_empty_and_numberless_text():
    assert extract_numbers("") == []
    assert extract_numbers("no figures at all") == []


def test_absurdly_long_digit_runs_are_dropped_not_parsed():
    assert extract_numbers("id " + "7" * 60 + " and 42") == ["42"]


# --------------------------------------------------------------------------- unsupported_numbers: the brief's cases


def test_price_and_change_supported_by_price_and_fraction():
    assert unsupported_numbers("$101.25 (up 3.4%)", ["price 101.25 … change 0.034"]) == []


def test_percent_unsupported_when_the_corpus_only_has_a_different_number():
    assert unsupported_numbers("about 12%", ["ratio 10.9"]) == ["12%"]


def test_evidence_ids_are_never_flagged():
    assert unsupported_numbers("Growth was strong [E12].", ["nothing numeric here"]) == []


# --------------------------------------------------------------------------- normalization rules


def test_thousands_separators_and_dollar_sign_are_normalized():
    assert unsupported_numbers("$1,234.56", ["1234.56"]) == []
    assert unsupported_numbers("1234.56", ["$1,234.56"]) == []
    assert unsupported_numbers("$50", ["50"]) == []
    assert unsupported_numbers("50", ["$50.00"]) == []


@pytest.mark.parametrize("corpus_text", [
    "change 0.034",         # raw fraction
    "0.0340",               # fraction with trailing zero
    "+3.40%",               # the toolbox's own format
    "3.4%",
    "moved 3.4 points",     # already in percent units, no sign
])
def test_percent_matches_fraction_or_percent_forms(corpus_text):
    assert unsupported_numbers("up 3.4%", [corpus_text]) == []


@pytest.mark.parametrize("corpus_text", ["34", "0.34", "0.0034", "nothing"])
def test_percent_does_not_match_other_magnitudes(corpus_text):
    assert unsupported_numbers("up 3.4%", [corpus_text]) == ["3.4%"]


@pytest.mark.parametrize("answer,corpus_text", [
    ("$1.2B", "1,234,567,890"),
    ("$1.2B", "$1.20B"),
    ("$1.2B", "$1,200.00M"),
    ("1.2 billion", "$1.2B"),
    ("1.2bn", "$1.2B"),
    ("12.5K", "12,500"),
    ("$12.50K", "12500"),
    ("$3.1T", "$3.10T"),
    ("$3.1T", "3,100,000,000,000"),
    ("2.5M", "2,500,000"),
    ("2.5 million", "2500000"),
    ("52.3M", "volume 52,341,900"),
    ("9,973억", "$997.35B"),        # Korean unit: 9,973 x 10^8
    ("$1,250", "$1.25K"),
])
def test_scale_suffixes_multiply(answer, corpus_text):
    assert unsupported_numbers(answer, [corpus_text]) == []


@pytest.mark.parametrize("answer,corpus_text", [
    ("$1.2B", "$1.2M"),
    ("$1.2B", "1.2"),                   # a bare 1.2 is not 1.2 billion
    ("$1.3B", "1,234,567,890"),
    ("12.5K", "12,500,000"),
    ("$3.1T", "$3.1B"),
])
def test_scale_mismatches_are_flagged(answer, corpus_text):
    assert unsupported_numbers(answer, [corpus_text]) == [answer]


def test_scaled_corpus_value_also_offers_its_bare_mantissa():
    assert unsupported_numbers("997.35", ["market cap $997.35B"]) == []
    assert unsupported_numbers("3.4", ["moved by $3.4B"]) == []


def test_comparison_is_on_absolute_values():
    assert unsupported_numbers("down 3.4%", ["change -0.034"]) == []
    assert unsupported_numbers("-3.4", ["3.4"]) == []
    assert unsupported_numbers("3.4", ["-3.4"]) == []
    assert unsupported_numbers("-$50.00", ["$50.00"]) == []
    assert unsupported_numbers("fell 1.25%", ["-1.25%"]) == []


# --------------------------------------------------------------------------- rounding to the displayed precision


def test_precision_of_the_answer_decides_how_much_rounding_is_allowed():
    assert unsupported_numbers("12%", ["10.9"]) == ["12%"]
    assert unsupported_numbers("11%", ["10.9"]) == []       # 10.9 rounds to 11
    assert unsupported_numbers("$101.25", ["$101.2575"]) == []
    assert unsupported_numbers("$101.26", ["$101.2575"]) == []


@pytest.mark.parametrize("answer", ["101.3", "101.2"])
def test_half_up_half_even_and_truncation_are_all_accepted(answer):
    # 101.25 rounds to 101.3 (half-up), 101.2 (half-even) and truncates to 101.2
    assert unsupported_numbers(answer, ["close 101.25"]) == []


@pytest.mark.parametrize("answer", ["101.4", "101.1"])
def test_one_digit_further_away_is_flagged(answer):
    assert unsupported_numbers(answer, ["close 101.25"]) == [answer]


def test_truncation_of_a_longer_value_is_accepted():
    assert unsupported_numbers("101.2", ["101.29"]) == []
    assert unsupported_numbers("101.3", ["101.29"]) == []


def test_answer_cannot_claim_more_precision_than_the_data():
    assert unsupported_numbers("101.2575", ["101.25"]) == ["101.2575"]
    assert unsupported_numbers("3.45%", ["0.034"]) == ["3.45%"]
    assert unsupported_numbers("3.4%", ["0.0345"]) == []


def test_rounding_is_decimal_not_float():
    # float(2.675) is 2.67499999...; the decimal 2.675 rounds half-up to 2.68
    assert unsupported_numbers("2.68", ["2.675"]) == []
    assert unsupported_numbers("2.7", ["2.65"]) == []
    assert unsupported_numbers("2.5", ["2.65"]) == ["2.5"]


def test_scale_precision_applies_in_the_answers_own_unit():
    assert unsupported_numbers("$1.2B", ["$1,150,000,000"]) == []       # 1.15 rounds to 1.2
    assert unsupported_numbers("$1.2B", ["$1,149,999,999"]) == ["$1.2B"]
    assert unsupported_numbers("$1.20B", ["$1,234,567,890"]) == ["$1.20B"]
    assert unsupported_numbers("$1B", ["$1,234,567,890"]) == []


# --------------------------------------------------------------------------- dates


def test_a_date_is_supported_by_the_same_date_anywhere_in_the_corpus():
    assert unsupported_numbers("Filed 2026-09-25", ["published 2026-09-25"]) == []


def test_a_date_inside_an_iso_timestamp_counts():
    corpus = ['{"as_of": "2026-09-25T14:30:00.123Z"}']
    assert unsupported_numbers("Data as of 2026-09-25", corpus) == []
    assert unsupported_numbers("Data as of 2026-09-26", corpus) == ["2026-09-26"]


def test_a_date_is_not_supported_by_its_loose_number_parts():
    assert unsupported_numbers("2026-09-25", ["2026 25 9"]) == ["2026-09-25"]


def test_corpus_dates_do_not_leak_day_or_month_numbers():
    assert unsupported_numbers("25", ["2026-09-25"]) == ["25"]


def test_a_bare_year_is_supported_by_a_corpus_date_of_that_year():
    assert unsupported_numbers("earnings in 2026", ["as of 2026-09-25"]) == []
    assert unsupported_numbers("earnings in 2025", ["as of 2026-09-25"]) == ["2025"]


# --------------------------------------------------------------------------- exemptions


def test_small_bare_integers_are_exempt():
    assert unsupported_numbers("top 3 picks, 2 legs, 10 shares, -4 days, 0 left", []) == []


@pytest.mark.parametrize("token", ["11", "10.0", "5.5", "$5", "5%", "5B", "+11"])
def test_small_looking_numbers_with_units_or_decimals_are_not_exempt(token):
    assert unsupported_numbers(f"about {token} now", []) == [token]


def test_list_markers_are_not_flagged():
    assert unsupported_numbers("1. First\n2. Second\n3. Third 45%", []) == ["45%"]


# --------------------------------------------------------------------------- shape of the result


def test_result_is_in_answer_order_and_deduplicated():
    assert unsupported_numbers("12% then 99 then 12% and 99", []) == ["12%", "99"]


def test_empty_inputs():
    assert unsupported_numbers("", ["101.25"]) == []
    assert unsupported_numbers("no figures", ["101.25"]) == []
    assert unsupported_numbers("$101.25", []) == ["$101.25"]
    assert unsupported_numbers("$101.25", [""]) == ["$101.25"]


def test_every_text_in_the_corpus_counts():
    corpus = ["[E1] price 101.25", "[E2] qty 55", "user says: budget $5,000"]
    assert unsupported_numbers("$101.25, 55 shares, budget $5,000", corpus) == []


def test_mandate_json_supports_its_own_limits():
    mandate = '{"budget_usd": "5000", "max_position_pct": "20", "deny_symbols": ["TSLA"]}'
    assert unsupported_numbers("Budget $5,000 with a 20% position cap", [mandate]) == []
    assert unsupported_numbers("Budget $6,000", [mandate]) == ["$6,000"]


def test_quotes_of_numbers_printed_by_the_toolbox_formatters_are_supported():
    corpus = [
        f"last {_price(Decimal('101.2575'))}, change {_pct_from_fraction(Decimal('0.034'))}, "
        f"cap {_compact_usd(Decimal('997350000000'))}, equity {_usd(Decimal('1234.5'))}, "
        f"drop {_pct_from_fraction(Decimal('-0.0125'))}",
    ]
    assert corpus[0] == "last $101.2575, change +3.40%, cap $997.35B, equity $1,234.50, drop -1.25%"
    quotes = ["$101.2575", "$101.26", "$101.3", "3.4%", "+3.40%", "$997.35B", "$997.4B", "$1.0T",
              "$1,234.50", "$1,234.5", "down 1.25%"]
    for quote in quotes:
        assert unsupported_numbers(f"It shows {quote} today.", corpus) == [], quote
    for wrong in ["$101.30", "3.5%", "$997.45B", "$1,236", "1.4%"]:
        assert unsupported_numbers(f"It shows {wrong} today.", corpus) == [wrong], wrong


def test_untrusted_wrapper_text_still_counts_as_corpus():
    corpus = ['<untrusted source="E7">Revenue rose 8.5% to $2.1B</untrusted>']
    assert unsupported_numbers("Revenue grew 8.5% to $2.1B [E7]", corpus) == []


# --------------------------------------------------------------------------- performance (linear-time regexes)

_ADVERSARIAL = {
    "comma_ones": "1," * 100_000,
    "long_digit_run": "9" * 200_000,
    "digit_run_then_percent": "9" * 199_999 + "%",
    "dollar_run": "$" * 200_000,
    "sign_run": "-" * 100_000 + "+" * 100_000,
    "dot_run": "." * 200_000,
    "dotted_ones": "1." * 100_000,
    "grouped_thousands": "1,111" * 40_000,
    "broken_groups": "1,1111," * 28_000,
    "spaces_then_marker": " " * 199_990 + "1)",
    "newlines": "\n" * 200_000,
    "indented_lines": ("   " * 4 + "\n") * 15_000,
    "spaced_ones": "1 " * 100_000,
    "letter_glued": "a1" * 100_000,
    "evidence_ids": "[E12]" * 40_000,
    "dates": "2026-09-25" * 20_000,
    "timestamps": "2026-09-25T14:30:00Z " * 9_500,
    "times": "12:30 " * 33_000,
    "billion_words": "1 billion " * 20_000,
    "near_miss_words": "1 billio " * 22_000,
    "near_miss_dates": "2026-09-2 " * 20_000,
    "signed_ones": "-1 +1 " * 33_000,
}


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL))
def test_extract_numbers_is_fast_on_adversarial_200k_input(name):
    text = _ADVERSARIAL[name]
    assert len(text) >= 190_000
    started = time.perf_counter()
    extract_numbers(text)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("name", ["comma_ones", "long_digit_run", "grouped_thousands", "broken_groups", "spaced_ones"])
def test_unsupported_numbers_is_fast_on_adversarial_corpus(name):
    started = time.perf_counter()
    result = unsupported_numbers("about $101.25 and 12%", [_ADVERSARIAL[name]])
    assert time.perf_counter() - started < 1.0
    assert result == ["$101.25", "12%"]


def _corpus_line(i: int) -> str:
    return (
        f"- SYM{i % 50} ${100 + i * 0.37:.2f} ({(i % 200 - 100) / 1000:+.2f}%) vol {i * 7919 % 9_999_999:,} "
        f"mcap ${i % 900 + 100}.{i % 100:02d}B as of 2026-09-{1 + i % 28:02d}T14:30:00Z [E{i % 40}]\n"
    )


def test_unsupported_numbers_is_fast_on_a_200k_corpus_and_a_5k_answer():
    corpus = ["".join(_corpus_line(i) for i in range(2_400))]
    assert 190_000 <= len(corpus[0]) <= 260_000
    supported = [f"${100 + i * 0.37:.2f}" for i in range(0, 2_400, 75)]
    answer = ("; ".join(f"SYM{i} closed at {p} on 2026-09-{1 + i % 28:02d}" for i, p in enumerate(supported)) + ". ")
    answer = (answer * 6)[:5_000] + " Also $99999.99 and 4.4% and 2027-01-01."
    started = time.perf_counter()
    flagged = unsupported_numbers(answer, corpus)
    assert time.perf_counter() - started < 1.0
    assert "$99999.99" in flagged and "4.4%" in flagged and "2027-01-01" in flagged
    assert not [t for t in flagged if t in supported]


def test_unsupported_numbers_is_fast_on_a_dense_distinct_number_corpus():
    corpus = [" ".join(str(i) for i in range(1_000, 36_000))]
    assert len(corpus[0]) >= 170_000
    answer = " ".join(str(i) for i in range(1_000, 2_000)) + " 35999 123456789 36000"
    started = time.perf_counter()
    flagged = unsupported_numbers(answer, corpus)
    assert time.perf_counter() - started < 1.0
    assert flagged == ["123456789", "36000"]
