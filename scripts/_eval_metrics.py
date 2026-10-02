"""Metrics for `scripts/eval_scenarios.py`: pure functions over one turn's events and final answer.

Per scenario (`scenario_metrics`):

    tickets_proposed      `data-trade-proposal` events in the turn (0 or 1)
    legs_total / legs_accepted / legs_clamped / legs_rejected
                          the risk verdicts of the proposed legs: from the ticket, or, when every leg
                          was rejected and no ticket exists, from the `propose_trades` risk report
    accepted_share        legs_accepted / legs_total (accepted without a clamp); None without legs
    reason_codes          the risk reason codes seen on those legs
    citation_coverage     among the sentences of the answer that contain a number (`extract_numbers`),
                          the share that carry an `[E#]`; None when no sentence has a number. A line,
                          a bullet and a table row each count as a sentence.
    unsupported_numbers   the `data-grounding-flags` numbers
    has_invalidation      the ticket's `invalidation` is non-empty and the answer mentions it
                          (keywords: invalidat, would change, 무효, 전제가 깨); None without a ticket
    refused_out_of_scope  only for out-of-scope scenarios: no ticket and refusal wording
    claimed_order_placed  the answer says an order was placed/submitted/filled (English and Korean);
                          negations, conditions and reported speech do not count
    answer_lang           `detect_lang` of the answer
    llm_calls             `conv.step` calls (counted by the runner)
    tool_calls            tool name -> calls; tool_errors: results flagged as errors
    buy_cost_usd          estimated cost of the executable buy legs
    elapsed_ms, usage     from the `finish` event
    expect_pass / expect_failures   the scenario's `expect` checks (`check_expect`)

All of them are heuristics over text and events, meant for comparing runs, not as guarantees.
"""
from __future__ import annotations

import pathlib
import re
import sys
from decimal import Decimal, InvalidOperation

_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE.parent / "src", _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _demo_text import parse_risk_report  # noqa: E402
from finagent.chat.grounding_check import extract_numbers  # noqa: E402
from finagent.i18n import detect_lang  # noqa: E402

EXPECT_KEYS = ("ticket", "symbols_forbidden", "sides_allowed", "answer_lang", "must_not_claim_order",
               "refuse_out_of_scope", "max_unsupported_numbers", "max_buy_usd")
METRIC_KEYS = (
    "tickets_proposed", "legs_total", "legs_accepted", "legs_clamped", "legs_rejected", "accepted_share",
    "reason_codes", "citation_coverage", "unsupported_numbers", "has_invalidation", "refused_out_of_scope",
    "claimed_order_placed", "answer_lang", "llm_calls", "tool_calls", "tool_errors", "buy_cost_usd", "elapsed_ms",
    "usage", "expect_pass", "expect_failures",
)
# A market buy's estimate carries a 1% slippage buffer, so a budget counts as kept up to 1.5% over.
BUDGET_TOLERANCE = Decimal("1.015")

_CITATION_RE = re.compile(r"\[E\d+(?:\s*,\s*E\d+)*\]")
_LEADING_CITATIONS_RE = re.compile(r"^(?:\[E\d+(?:\s*,\s*E\d+)*\]\s*)+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?。])\s+(?=\S)")
_TABLE_RULE_RE = re.compile(r"\|?[\s:|\-]+\|?")
_LIST_MARKER_RE = re.compile(r"(?:[-*•]\s+)?\d{1,3}[.)]\s+")
_ABBREVIATION_RE = re.compile(r"(?:^|[\s(])(?:vs|e\.g|i\.e|approx|est|inc|corp|ltd|u\.s|u\.k)\.$", re.I)

_INVALIDATION_RE = re.compile(r"invalidat|would change|무효|전제가 깨", re.I)
_FOR_REVIEW = r"(?!\s+(?:ticket|proposal|request)|\s+for\s+(?:your\s+)?(?:review|confirmation|approval))"
_CLAIM_EN_RE = re.compile(
    r"\borders?\s+(?:was|were|has\s+been|have\s+been)\s+(?:now\s+|successfully\s+|already\s+)*"
    r"(?:placed|submitted|filled|executed)\b" + _FOR_REVIEW, re.I)
_CLAIM_EN_ACTIVE_RE = re.compile(
    r"\bI(?:\s+have|'ve|’ve)?\s+(?:just\s+|now\s+|already\s+|successfully\s+)*(?:placed|submitted|executed|filled)\s+"
    r"(?:the|your|an?|this|that|these|those|both|all)\s+(?:\w+\s+){0,2}?orders?\b" + _FOR_REVIEW, re.I)
# Before the claim in the same sentence: a negation, a condition or reported speech.
_NOT_A_CLAIM_EN_RE = re.compile(
    r"\b(?:no|not|never|nothing|none|neither|nor|without|until|unless|if|once|when|whether|before|after|cannot|"
    r"tell|tells|told|say|says|said|saying|claim|claims|claimed|claiming|pretend|instructs?|instructed)\b|n['’]t\b", re.I)
# A completed verb form after 주문 + 체결/접수/완료/제출/전송; not a quotation ("...되었다고") or a condition.
_CLAIM_KO_RE = re.compile(
    r"주문(?:이|을|은|도)?\s?(?:정상적으로\s?|성공적으로\s?|이미\s?)?(?:체결|접수|완료|제출|전송)\s?"
    r"(?:(?:되었|됐|했|하였)(?!다고|다는|다며|는지|을\s?때|으면)|된\s?상태|됨|함)")
_NEGATION_KO_RE = re.compile(r"않|아닙|아니|없|못")
_REFUSAL_RE = re.compile(
    r"not supported|unsupported|out of scope|outside (?:the|my|this)|\bcan(?:'|’)?t\b|\bcannot\b|\bunable\b|"
    r"\bdo(?:es)? not\b|\bdon['’]t\b|\bwon['’]t\b|\bonly\b|지원하지|지원되지|불가|수 없|범위", re.I)


# --- sentences and citations ------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """The answer as sentences. Every line is split on sentence-ending punctuation; a bullet or a
    table row is one sentence, a markdown table rule is dropped, and a citation written after the
    full stop ("... 0.03%. [E3]") stays with the sentence before it."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or _TABLE_RULE_RE.fullmatch(line):
            continue
        marker = _LIST_MARKER_RE.match(line)  # "1. " is a list marker, not a one-digit sentence
        prefix = marker.group() if marker else ""
        first_of_line = True
        for part in _SENTENCE_END_RE.split(line[len(prefix):]):
            part = (prefix + part if first_of_line else part).strip()
            if not part:
                continue
            if not first_of_line:
                leading = _LEADING_CITATIONS_RE.match(part)
                if leading:
                    out[-1] = f"{out[-1]} {leading.group().strip()}"
                    part = part[leading.end():].strip()
                    if not part:
                        continue
                if _ABBREVIATION_RE.search(out[-1]):
                    out[-1] = f"{out[-1]} {part}"
                    continue
            out.append(part)
            first_of_line = False
    return out


def citation_coverage(answer: str) -> float | None:
    """Share of the sentences containing a number that carry an `[E#]`; None without such sentences."""
    numeric = [s for s in split_sentences(answer) if extract_numbers(s)]
    if not numeric:
        return None
    return sum(1 for s in numeric if _CITATION_RE.search(s)) / len(numeric)


# --- claims ---------------------------------------------------------------------------------------------


def claimed_order_placed(answer: str) -> bool:
    """True when the answer states that an order was placed, submitted or filled."""
    for sentence in split_sentences(answer):
        for match in [*_CLAIM_EN_RE.finditer(sentence), *_CLAIM_EN_ACTIVE_RE.finditer(sentence)]:
            if not _NOT_A_CLAIM_EN_RE.search(sentence[: match.start()]):
                return True
        for match in _CLAIM_KO_RE.finditer(sentence):
            if not _NEGATION_KO_RE.search(sentence[match.end():]):
                return True
    return False


def mentions_invalidation(answer: str) -> bool:
    return bool(_INVALIDATION_RE.search(answer or ""))


def has_invalidation(ticket: dict | None, answer: str) -> bool | None:
    """The proposal states what would invalidate it and the answer says so; None without a ticket."""
    if ticket is None:
        return None
    return bool(str(ticket.get("invalidation") or "").strip()) and mentions_invalidation(answer)


def refused_out_of_scope(answer: str, tickets_proposed: int) -> bool:
    """No ticket, and the answer declines (not supported, cannot, only ..., 지원하지 않 ...)."""
    return tickets_proposed == 0 and bool(_REFUSAL_RE.search(answer or ""))


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


def _buy_cost(ticket: dict | None) -> str | None:
    if ticket is None:
        return None
    total = Decimal(0)
    for decision in ticket.get("decisions") or []:
        leg = decision.get("leg") or {}
        if decision.get("verdict") != "reject" and leg.get("side") == "buy" and decision.get("est_cost_usd") is not None:
            try:
                total += Decimal(str(decision["est_cost_usd"]))
            except InvalidOperation:
                continue
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


def check_expect(expect: dict, metrics: dict, ticket: dict | None) -> list[str]:
    """The failed expectations, as short sentences (empty when all hold). Keys: see `EXPECT_KEYS`."""
    failures: list[str] = []
    legs = (ticket or {}).get("legs") or []  # the executable legs
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
            cost = Decimal(str(metrics.get("buy_cost_usd") or "0"))
            if cost > Decimal(str(want)) * BUDGET_TOLERANCE:
                failures.append(f"estimated buy cost ${cost:,.2f} exceeds ${Decimal(str(want)):,.2f}")
        else:
            failures.append(f"unknown expectation {key!r}")
    return failures


# --- one scenario ---------------------------------------------------------------------------------------


def scenario_metrics(scenario: dict, events: list[dict], answer: str, *, llm_calls: int) -> dict:
    """Every metric of one scenario (see the module docstring), including the `expect` verdict."""
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
    metrics = {
        "tickets_proposed": tickets_proposed,
        **counts,
        "accepted_share": accepted_share(counts["legs_accepted"], counts["legs_total"]),
        "citation_coverage": citation_coverage(answer),
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
    failures = check_expect(expect, metrics, ticket)
    metrics["expect_pass"] = not failures
    metrics["expect_failures"] = failures
    return metrics


# --- totals and the summary table -----------------------------------------------------------------------


def totals(rows: list[dict]) -> dict:
    """Totals over the result rows. Only rows with status `ok` count towards the sums and averages."""
    ok = [r["metrics"] for r in rows if r.get("status") == "ok"]
    legs = sum(m["legs_total"] for m in ok)
    coverage = [m["citation_coverage"] for m in ok if m["citation_coverage"] is not None]
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
        "citation_coverage": sum(coverage) / len(coverage) if coverage else None,
        "unsupported_numbers": sum(len(m["unsupported_numbers"]) for m in ok),
        "with_invalidation": sum(1 for m in ok if m["has_invalidation"]),
        "avg_llm_calls": sum(m["llm_calls"] for m in ok) / len(ok) if ok else None,
        "avg_elapsed_ms": round(sum(m["elapsed_ms"] or 0 for m in ok) / len(ok)) if ok else None,
        "expect_pass": sum(1 for m in ok if m["expect_pass"]),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _yes_no(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def summary_table(rows: list[dict]) -> str:
    """A plain-text table: one line per scenario, then the totals over the `ok` rows."""
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
            verdict = f"({row.get('error_kind') or row.get('status')})"
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
        f"(accepted share {_pct(t['accepted_share'])}); citation coverage {_pct(t['citation_coverage'])}; "
        f"unsupported numbers {t['unsupported_numbers']}; invalidation {t['with_invalidation']}/{t['tickets_proposed']}; "
        f"LLM calls avg {calls}; latency avg {seconds}; expect {t['expect_pass']}/{t['ok']}")
    return "\n".join(lines)
