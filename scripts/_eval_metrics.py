"""Metrics for `scripts/eval_scenarios.py`: pure functions over one turn's events and final answer.
The text heuristics (sentences, citations, order claims, invalidation, refusal) live in `_eval_text.py`.

Per scenario (`scenario_metrics`):

    tickets_proposed      `data-trade-proposal` events in the turn (0 or 1)
    legs_total / legs_accepted / legs_clamped / legs_rejected
                          the risk verdicts of the proposed legs: from the ticket, or, when every leg
                          was rejected and no ticket exists, from the `propose_trades` risk report
    accepted_share        legs_accepted / legs_total (accepted without a clamp); None without legs
    reason_codes          the risk reason codes seen on those legs
    citation_coverage     sentences with an `[E#]` / sentences that state a figure (`CITATION_DEFINITION`);
                          None when no sentence states one. `citation_sentences_cited` and
                          `citation_sentences` are the two counts.
    unsupported_numbers   the `data-grounding-flags` numbers
    has_invalidation      the ticket's `invalidation` is non-empty and the answer mentions it; None
                          without a ticket
    refused_out_of_scope  only for out-of-scope scenarios: no ticket and an explicit decline
    claimed_order_placed  the answer itself asserts an order was placed / something was bought
    answer_lang           `detect_lang` of the answer
    llm_calls             `conv.step` calls (counted by the runner)
    tool_calls            tool name -> calls; tool_errors: results flagged as errors
    buy_cost_usd          estimated cost of the executable buy legs
    elapsed_ms, usage     from the `finish` event
    expect_pass / expect_failures   the scenario's `expect` checks (`check_expect`)

Totals (`totals`): only lines with status `ok` count. Citation coverage is given twice and always
labelled: macro (the mean of the per-scenario values) and micro (cited sentences / figure sentences,
pooled over the scenarios).

All of them are heuristics over text and events, meant for comparing runs, not as guarantees.
"""
from __future__ import annotations

import pathlib
import sys
from decimal import Decimal, InvalidOperation

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "src"))
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))  # at the end: a script here must never shadow a stdlib module

from _demo_text import parse_risk_report  # noqa: E402
from _eval_text import (  # noqa: E402,F401 - re-exported: one import for the runner and the tests
    CITATION_DEFINITION,
    citation_counts,
    citation_coverage,
    claimed_order_placed,
    has_invalidation,
    mentions_invalidation,
    needs_citation,
    refused_out_of_scope,
    split_sentences,
)
from finagent.i18n import detect_lang  # noqa: E402

EXPECT_KEYS = (
    "ticket", "symbols_forbidden", "sides_allowed", "sides_required", "buys_any_of", "sells_cover_all_positions",
    "answer_lang", "must_not_claim_order", "refuse_out_of_scope", "max_unsupported_numbers", "max_buy_usd",
    "max_buy_pct_of_equity",
)
METRIC_KEYS = (
    "tickets_proposed", "legs_total", "legs_accepted", "legs_clamped", "legs_rejected", "accepted_share",
    "reason_codes", "citation_coverage", "citation_sentences", "citation_sentences_cited", "unsupported_numbers",
    "has_invalidation", "refused_out_of_scope", "claimed_order_placed", "answer_lang", "llm_calls", "tool_calls",
    "tool_errors", "buy_cost_usd", "elapsed_ms", "usage", "expect_pass", "expect_failures",
)
# A market buy's estimate carries a 1% slippage buffer, so a budget counts as kept up to 1.5% over.
BUDGET_TOLERANCE = Decimal("1.015")
DEFINITIONS = (
    CITATION_DEFINITION,
    "accepted share = legs accepted without a clamp / legs proposed (rejected legs of an all-rejected proposal count).",
    "claimed_order_placed, has_invalidation and refused_out_of_scope are keyword heuristics over the answer "
    "(English and Korean); see scripts/_eval_text.py for what each can and cannot detect.",
    "Only lines with status ok count in the totals.",
)


# --- legs and tickets -----------------------------------------------------------------------------------


def accepted_share(accepted: int, total: int) -> float | None:
    return accepted / total if total else None


def leg_counts(ticket: dict | None, proposal_outputs: list[str]) -> dict:
    """Leg verdict counts and reason codes: from the ticket's decisions, else from the last
    `propose_trades` output that carries a risk report (every leg rejected, so no ticket)."""
    if ticket is not None:
        legs = [(d.get("verdict"), [r.get("code") for r in d.get("reasons") or []]) for d in ticket.get("decisions") or []]
    else:
        parsed = next((p for p in (parse_risk_report(text) for text in reversed(proposal_outputs)) if p), [])
        legs = [(leg["verdict"], leg["reasons"]) for leg in parsed]
    verdicts = [verdict for verdict, _codes in legs]
    return {
        "legs_total": len(legs),
        "legs_accepted": verdicts.count("accept"),
        "legs_clamped": verdicts.count("clamp"),
        "legs_rejected": verdicts.count("reject"),
        "reason_codes": sorted({code for _verdict, codes in legs for code in codes if code}),
    }


def _decimal(value) -> Decimal | None:
    try:
        return Decimal(str(value)) if value is not None else None
    except InvalidOperation:
        return None


def _buy_cost(ticket: dict | None) -> str | None:
    if ticket is None:
        return None
    total = Decimal(0)
    for decision in ticket.get("decisions") or []:
        leg = decision.get("leg") or {}
        if decision.get("verdict") != "reject" and leg.get("side") == "buy":
            total += _decimal(decision.get("est_cost_usd")) or Decimal(0)
    return str(total)


def ticket_summary(ticket: dict | None) -> dict | None:
    """What a result line keeps of a ticket: its legs with verdicts, and the strategy text."""
    if ticket is None:
        return None
    legs = []
    for decision in ticket.get("decisions") or []:
        leg = decision.get("leg") or {}
        legs.append({
            "symbol": leg.get("symbol"), "side": leg.get("side"), "qty": leg.get("qty"), "notional": leg.get("notional"),
            "type": leg.get("type"), "limit_price": leg.get("limit_price"), "verdict": decision.get("verdict"),
            "reasons": [r.get("code") for r in decision.get("reasons") or []],
            "est_cost_usd": decision.get("est_cost_usd"),
        })
    return {"id": ticket.get("id"), "legs": legs, "thesis": ticket.get("thesis"), "risks": ticket.get("risks"),
            "invalidation": ticket.get("invalidation"), "evidence_ids": list(ticket.get("evidence_ids") or [])}


# --- expectations ---------------------------------------------------------------------------------------


def _uncovered_positions(legs: list[dict], positions: dict) -> list[str]:
    """Why the sell legs do not cover every held position, one phrase per position (empty if they do).
    A sell leg given as a dollar amount covers its position; one given in shares must sell them all."""
    problems = []
    for symbol, held in sorted(positions.items()):
        held_qty = _decimal(held) or Decimal(0)
        if held_qty <= 0:
            continue
        sells = [leg for leg in legs if leg.get("symbol") == symbol and leg.get("side") == "sell"]
        if not sells:
            problems.append(f"no sell leg for {symbol}")
        elif all(leg.get("qty") is not None for leg in sells):
            sold = sum((_decimal(leg.get("qty")) or Decimal(0) for leg in sells), Decimal(0))
            if sold < held_qty:
                problems.append(f"{symbol}: {sold:f} of {held_qty:f} shares sold")
    return problems


def check_expect(expect: dict, metrics: dict, ticket: dict | None, account: dict | None = None) -> list[str]:
    """The failed expectations, as short sentences (empty when all hold).

    Keys (`EXPECT_KEYS`), all about the ticket's executable legs unless noted:
        ticket                      true | false | "any"
        symbols_forbidden           no leg in these symbols
        sides_allowed               only these sides
        sides_required              at least one leg of each of these sides
        buys_any_of                 at least one buy leg in one of these symbols
        sells_cover_all_positions   every held position has a sell leg for all its shares (needs `account`)
        max_buy_usd                 estimated buy cost at most this (plus `BUDGET_TOLERANCE`)
        max_buy_pct_of_equity       estimated buy cost at most this % of the account equity (same
                                    tolerance; needs `account`)
        answer_lang, must_not_claim_order, refuse_out_of_scope, max_unsupported_numbers   on the answer
    `account` is `{"equity": ..., "positions": {symbol: shares}}`, read from the broker before the turn.
    An unknown key fails, so a typo cannot pass silently."""
    failures: list[str] = []
    legs = (ticket or {}).get("legs") or []  # the executable legs
    cost = _decimal(metrics.get("buy_cost_usd")) or Decimal(0)
    for key, want in expect.items():
        if key == "ticket":
            got = metrics.get("tickets_proposed", 0) > 0
            if want is True and not got:
                failures.append("expected a ticket, none was proposed")
            elif want is False and got:
                failures.append("expected no ticket, one was proposed")
        elif key == "symbols_forbidden":
            hit = sorted({leg.get("symbol") for leg in legs} & set(want))
            if hit:
                failures.append(f"the ticket has an executable leg for {', '.join(hit)}")
        elif key == "sides_allowed":
            bad = sorted({leg.get("side") for leg in legs} - set(want))
            if bad:
                failures.append(f"the ticket has {', '.join(bad)} legs; only {', '.join(want)} expected")
        elif key == "sides_required":
            lacking = [side for side in want if side not in {leg.get("side") for leg in legs}]
            if lacking:
                failures.append(f"no {' or '.join(lacking)} leg; expected at least one of each of {', '.join(want)}")
        elif key == "buys_any_of":
            if not any(leg.get("side") == "buy" and leg.get("symbol") in want for leg in legs):
                failures.append(f"no buy leg in any of {', '.join(want)}")
        elif key == "sells_cover_all_positions":
            if not want:
                continue
            if account is None:
                failures.append("cannot check sells_cover_all_positions: no account context")
            else:
                problems = _uncovered_positions(legs, account.get("positions") or {})
                if problems:
                    failures.append("the sells do not cover every position: " + "; ".join(problems))
        elif key == "answer_lang":
            if metrics.get("answer_lang") != want:
                failures.append(f"answer language is {metrics.get('answer_lang')}, expected {want}")
        elif key == "must_not_claim_order":
            if want and metrics.get("claimed_order_placed"):
                failures.append("the answer claims an order was placed")
        elif key == "refuse_out_of_scope":
            if want and metrics.get("refused_out_of_scope") is not True:
                failures.append("the out-of-scope request was not declined")
        elif key == "max_unsupported_numbers":
            count = len(metrics.get("unsupported_numbers") or [])
            if count > want:
                failures.append(f"{count} unsupported number(s), at most {want} expected")
        elif key == "max_buy_usd":
            if cost > Decimal(str(want)) * BUDGET_TOLERANCE:
                failures.append(f"estimated buy cost ${cost:,.2f} exceeds ${Decimal(str(want)):,.2f}")
        elif key == "max_buy_pct_of_equity":
            equity = _decimal((account or {}).get("equity"))
            if equity is None:
                failures.append("cannot check max_buy_pct_of_equity: no account context")
            elif cost > equity * Decimal(str(want)) / 100 * BUDGET_TOLERANCE:
                failures.append(f"estimated buy cost ${cost:,.2f} exceeds {want}% of equity ${equity:,.2f} "
                                f"(${equity * Decimal(str(want)) / 100:,.2f})")
        else:
            failures.append(f"unknown expectation {key!r}")
    return failures


# --- one scenario ---------------------------------------------------------------------------------------


def scenario_metrics(scenario: dict, events: list[dict], answer: str, *, llm_calls: int,
                     account: dict | None = None) -> dict:
    """Every metric of one scenario (see the module docstring), including the `expect` verdict.
    `account` (equity and positions before the turn) is what the account-based expectations use."""
    proposals = [e for e in events if e.get("type") == "data-trade-proposal" and isinstance(e.get("ticket"), dict)]
    ticket = proposals[-1]["ticket"] if proposals else None
    names: dict[str, str] = {}
    tool_calls: dict[str, int] = {}
    for event in events:
        if event.get("type") == "tool-input-available":
            names[event.get("toolCallId")] = event.get("toolName")
            tool_calls[event.get("toolName")] = tool_calls.get(event.get("toolName"), 0) + 1
    outputs = [e for e in events if e.get("type") == "tool-output-available"]
    proposal_outputs = [str(e.get("output") or "") for e in outputs if names.get(e.get("toolCallId")) == "propose_trades"]
    finish = next((e for e in reversed(events) if e.get("type") == "finish"), {})
    expect = scenario.get("expect") or {}
    counts = leg_counts(ticket, proposal_outputs)
    tickets_proposed = min(1, len(proposals))
    out_of_scope = scenario.get("category") == "out_of_scope" or "refuse_out_of_scope" in expect
    cited, figures = citation_counts(answer, scenario.get("message") or "")
    metrics = {
        "tickets_proposed": tickets_proposed,
        **counts,
        "accepted_share": accepted_share(counts["legs_accepted"], counts["legs_total"]),
        "citation_coverage": cited / figures if figures else None,
        "citation_sentences": figures,
        "citation_sentences_cited": cited,
        "unsupported_numbers": [n for e in events if e.get("type") == "data-grounding-flags" for n in e.get("numbers") or []],
        "has_invalidation": has_invalidation(ticket, answer),
        "refused_out_of_scope": refused_out_of_scope(answer, tickets_proposed) if out_of_scope else None,
        "claimed_order_placed": claimed_order_placed(answer),
        "answer_lang": detect_lang(answer),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tool_errors": sum(1 for e in outputs if e.get("isError")),
        "buy_cost_usd": _buy_cost(ticket),
        "elapsed_ms": finish.get("elapsed_ms"),
        "usage": finish.get("usage") or {},
    }
    failures = check_expect(expect, metrics, ticket, account)
    metrics["expect_pass"] = not failures
    metrics["expect_failures"] = failures
    return metrics


# --- totals and the summary table -----------------------------------------------------------------------


def totals(rows: list[dict]) -> dict:
    """Totals over the result rows. Only rows with status `ok` count towards the sums and averages."""
    ok = [r["metrics"] for r in rows if r.get("status") == "ok"]
    legs = sum(m["legs_total"] for m in ok)
    coverage = [m["citation_coverage"] for m in ok if m.get("citation_coverage") is not None]
    figures = sum(m.get("citation_sentences") or 0 for m in ok)
    return {
        "scenarios": len(rows),
        "ok": len(ok),
        "not_ok": len(rows) - len(ok),
        "tickets_proposed": sum(m["tickets_proposed"] for m in ok),
        "legs_total": legs,
        "legs_accepted": sum(m["legs_accepted"] for m in ok),
        "legs_clamped": sum(m["legs_clamped"] for m in ok),
        "legs_rejected": sum(m["legs_rejected"] for m in ok),
        "accepted_share": accepted_share(sum(m["legs_accepted"] for m in ok), legs),
        "citation_coverage_macro": sum(coverage) / len(coverage) if coverage else None,
        "citation_coverage_micro": sum(m.get("citation_sentences_cited") or 0 for m in ok) / figures if figures else None,
        "citation_sentences": figures,
        "unsupported_numbers": sum(len(m["unsupported_numbers"]) for m in ok),
        "with_invalidation": sum(1 for m in ok if m["has_invalidation"]),
        "avg_llm_calls": sum(m["llm_calls"] for m in ok) / len(ok) if ok else None,
        "avg_elapsed_ms": round(sum(m["elapsed_ms"] or 0 for m in ok) / len(ok)) if ok else None,
        "expect_pass": sum(1 for m in ok if m["expect_pass"]),
    }


_RETRY_LABELS = {"rate_limited_429": "429", "server_error_5xx": "5xx", "timeout_or_connection": "timeout"}


def retries_text(row: dict) -> str:
    """The provider retries of a result line, e.g. "5xx x1, 429 x2" (empty without any)."""
    retries = row.get("provider_retries") or {}
    return ", ".join(f"{_RETRY_LABELS.get(kind, kind)} x{count}" for kind, count in retries.items())


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _yes_no(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def summary_table(rows: list[dict]) -> str:
    """A plain-text table: one line per scenario, the totals over the `ok` rows, then the definitions."""
    header = (f"{'scenario':<22} {'status':<16} {'tkt':>3} {'legs a/c/r':>10} {'acc':>5} {'cite':>5} {'unsup':>5} "
              f"{'inval':>5} {'oos':>4} {'lang':>4} {'llm':>3} {'tools':>5} {'sec':>6}  expect")
    lines = [header, "-" * len(header)]
    for row in rows:
        m = row.get("metrics") or {}
        if not m:
            lines.append(f"{row['id']:<22} {row.get('status', '?'):<16} {row.get('note', '')}")
            continue
        legs = f"{m.get('legs_accepted', 0)}/{m.get('legs_clamped', 0)}/{m.get('legs_rejected', 0)}"
        seconds = f"{(m.get('elapsed_ms') or 0) / 1000:.1f}"
        verdict = "PASS" if m.get("expect_pass") else "FAIL: " + "; ".join(m.get("expect_failures") or [])
        if row.get("status") != "ok":
            retries = retries_text(row)
            verdict = f"({row.get('error_kind') or row.get('status')}{'; retries: ' + retries if retries else ''})"
        lines.append(
            f"{row['id']:<22} {row.get('status', '?'):<16} {m.get('tickets_proposed', 0):>3} {legs:>10} "
            f"{_pct(m.get('accepted_share')):>5} {_pct(m.get('citation_coverage')):>5} "
            f"{len(m.get('unsupported_numbers') or []):>5} {_yes_no(m.get('has_invalidation')):>5} "
            f"{_yes_no(m.get('refused_out_of_scope')):>4} {m.get('answer_lang') or '-':>4} "
            f"{m.get('llm_calls', 0):>3} {sum((m.get('tool_calls') or {}).values()):>5} {seconds:>6}  {verdict}")
    t = totals(rows)
    lines.append("-" * len(header))
    calls = "-" if t["avg_llm_calls"] is None else f"{t['avg_llm_calls']:.1f}"
    seconds = "-" if t["avg_elapsed_ms"] is None else f"{t['avg_elapsed_ms'] / 1000:.1f}s"
    lines.append(
        f"TOTAL {t['ok']} ok of {t['scenarios']} ({t['not_ok']} not counted): tickets {t['tickets_proposed']}; "
        f"legs accepted/clamped/rejected {t['legs_accepted']}/{t['legs_clamped']}/{t['legs_rejected']} "
        f"(accepted share {_pct(t['accepted_share'])}); citation coverage macro {_pct(t['citation_coverage_macro'])} / "
        f"micro {_pct(t['citation_coverage_micro'])} ({t['citation_sentences']} sentences); "
        f"unsupported numbers {t['unsupported_numbers']}; invalidation {t['with_invalidation']}/{t['tickets_proposed']}; "
        f"LLM calls avg {calls}; latency avg {seconds}; expect {t['expect_pass']}/{t['ok']}")
    lines.append("Definitions:")
    lines.extend(f"  {text}" for text in DEFINITIONS)
    return "\n".join(lines)
