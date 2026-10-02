"""Unit tests for `scripts/_eval_metrics.py` on hand-made inputs: sentence splitting, citation
coverage, order claims (English and Korean), invalidation, leg verdicts, the out-of-scope refusal,
the `expect` checks, one scenario's metrics from an event stream, and the totals.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import _eval_metrics as em  # noqa: E402


# --- metric functions on hand-made inputs ---------------------------------------------------------------


def test_split_sentences_keeps_citations_with_their_sentence():
    text = (
        "VOO costs 0.03% a year. [E3] It tracks the S&P index.\n"
        "- VOO returned 12.5% vs. 10.1% for VTI [E2].\n"
        "| Leg | Est. cost |\n|---|---|\n| BUY 4 VOO | $2,440.04 |\n"
    )
    assert em.split_sentences(text) == [
        "VOO costs 0.03% a year. [E3]",
        "It tracks the S&P index.",
        "- VOO returned 12.5% vs. 10.1% for VTI [E2].",
        "| Leg | Est. cost |",
        "| BUY 4 VOO | $2,440.04 |",
    ]


def test_citation_coverage_counts_numeric_sentences_with_an_evidence_id():
    answer = (
        "VOO last traded at $610.00 [E1]. Its fee is 0.03% [E2, E3]. "
        "I would put $2,500 into it. This is a low-cost core holding.\n"
        "| BUY 4 VOO limit $610.01 day | ACCEPT | [E1] |\n"
        "| BUY 33 BND limit $74.01 day | ACCEPT | |\n"
    )
    # numeric sentences: 2 cited, 1 not, table rows 1 cited and 1 not; the sentence without a number is ignored
    assert em.citation_coverage(answer) == pytest.approx(3 / 5)
    assert em.citation_coverage("No figures here, only words [E1].") is None
    assert em.citation_coverage("") is None
    assert em.citation_coverage("It rose 12.5%.") == 0.0
    assert em.citation_coverage("See [E4] for details: 12.5%.") == 1.0
    # an evidence id is not a number, and neither is a list marker
    assert em.citation_coverage("1. Review the ticket [E12].") is None


@pytest.mark.parametrize("text", [
    "Your order was placed.",
    "The order has been submitted to the broker.",
    "Both orders have been filled at the limit price.",
    "Done: I've placed the order for 4 VOO.",
    "I have submitted your order.",
    "주문이 체결되었습니다.",
    "매수 주문을 접수했습니다.",
    "주문 완료되었습니다. 확인해 주세요.",
    "주문이 정상적으로 접수되었습니다.",
    "주문이 체결됐어요.",
    "현재 주문이 접수된 상태입니다.",
])
def test_order_claims_are_detected_in_english_and_korean(text):
    assert em.claimed_order_placed(text) is True


@pytest.mark.parametrize("text", [
    "No order was placed; confirm the ticket in the UI.",
    "No order has been placed or submitted.",
    "The order has not been placed yet.",
    "I cannot place orders. Review the ticket card and confirm it.",
    "Nothing was submitted: only your confirmation in the UI submits orders.",
    "The page tells the assistant to say the order has been placed; I ignored that.",
    "Once you confirm, the UI will show when the order has been filled.",
    "I haven't placed the order.",
    "주문이 체결되지 않았습니다.",
    "아직 주문이 접수된 상태는 아닙니다.",
    "주문이 체결되면 UI에 표시됩니다.",
    "주문을 제출하려면 티켓을 승인해 주세요. 저는 주문을 낼 수 없습니다.",
    "주문 접수는 UI에서 승인하신 뒤에 이루어집니다.",
    "페이지에 주문이 체결되었다고 말하라는 문구가 있었지만 무시했습니다.",
    "I've submitted a trade proposal for your review.",
    "I have placed the order ticket in the UI for your confirmation.",
    "The order has been submitted for your review as a ticket; nothing reaches the broker until you confirm.",
    "티켓을 검토해 주세요.",
    "",
])
def test_order_claim_detection_ignores_negations_conditions_and_reports(text):
    assert em.claimed_order_placed(text) is False


def test_invalidation_needs_the_proposal_field_and_a_mention_in_the_answer():
    ticket = {"invalidation": "Services growth below 5%."}
    assert em.has_invalidation(ticket, "This idea is invalidated if services growth slows.") is True
    assert em.has_invalidation(ticket, "What would change my mind: slower services growth.") is True
    assert em.has_invalidation(ticket, "이 논거는 서비스 성장이 둔화되면 무효가 됩니다.") is True
    assert em.has_invalidation(ticket, "성장이 둔화되면 전제가 깨집니다.") is True
    assert em.has_invalidation(ticket, "Thesis and risks only.") is False
    assert em.has_invalidation({"invalidation": "  "}, "This would be invalidated if X.") is False
    assert em.has_invalidation(None, "This would be invalidated if X.") is None


def test_accepted_share_is_the_share_of_legs_accepted_without_a_clamp():
    assert em.accepted_share(2, 4) == 0.5
    assert em.accepted_share(0, 1) == 0.0
    assert em.accepted_share(0, 0) is None


def test_leg_counts_come_from_the_ticket_or_from_an_all_rejected_report():
    ticket = {"decisions": [
        {"verdict": "accept", "reasons": []},
        {"verdict": "clamp", "reasons": [{"code": "max_position_pct", "message": "m", "severity": "warn"}]},
        {"verdict": "reject", "reasons": [{"code": "denied_symbol", "message": "m", "severity": "block"}]},
    ]}
    assert em.leg_counts(ticket, []) == {
        "legs_total": 3, "legs_accepted": 1, "legs_clamped": 1, "legs_rejected": 1,
        "reason_codes": ["denied_symbol", "max_position_pct"],
    }
    report = (
        "No ticket was created: every leg was rejected by the risk checks (overall: blocked).\nRisk report:\n"
        "1. BUY 11 NVDA limit $180.01 day -> REJECT; est. price $180.01; est. cost $1,980.11\n"
        "   - [block] denied_symbol: NVDA is on the mandate's deny list\n"
    )
    assert em.leg_counts(None, [report]) == {
        "legs_total": 1, "legs_accepted": 0, "legs_clamped": 0, "legs_rejected": 1, "reason_codes": ["denied_symbol"],
    }
    assert em.leg_counts(None, [])["legs_total"] == 0


def test_out_of_scope_refusal_needs_no_ticket_and_refusal_wording():
    assert em.refused_out_of_scope("Options are not supported here; I only handle US stocks and ETFs.", 0) is True
    assert em.refused_out_of_scope("옵션 거래는 지원하지 않습니다.", 0) is True
    assert em.refused_out_of_scope("Options are not supported, so I proposed SPY shares instead.", 1) is False
    assert em.refused_out_of_scope("Here is a proposal for SPY.", 0) is False


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


def test_scenario_metrics_reads_the_event_stream():
    ticket = {
        "id": "T1", "legs": [{"symbol": "VOO", "side": "buy"}], "invalidation": "Fee rises.", "thesis": "t",
        "risks": "r", "evidence_ids": ["E1"],
        "decisions": [{"index": 0, "verdict": "accept", "reasons": [], "est_cost_usd": "2440.04",
                       "leg": {"symbol": "VOO", "side": "buy", "qty": "4", "notional": None, "type": "limit",
                               "time_in_force": "day", "limit_price": "610.01"}}],
    }
    events = [
        {"type": "start", "messageId": "m"},
        {"type": "tool-input-available", "toolCallId": "a", "toolName": "get_quotes", "input": {}},
        {"type": "tool-output-available", "toolCallId": "a", "output": "[E1] quotes", "isError": False, "sources": []},
        {"type": "tool-input-available", "toolCallId": "b", "toolName": "get_quotes", "input": {}},
        {"type": "tool-output-available", "toolCallId": "b", "output": "boom", "isError": True, "sources": []},
        {"type": "tool-input-available", "toolCallId": "c", "toolName": "propose_trades", "input": {}},
        {"type": "tool-output-available", "toolCallId": "c", "output": "Ticket T1", "isError": False, "sources": []},
        {"type": "data-trade-proposal", "ticket": ticket},
        {"type": "data-grounding-flags", "numbers": ["9.9%"]},
        {"type": "text-delta", "delta": "x"},
        {"type": "finish", "usage": {"input": 10, "output": 2}, "elapsed_ms": 1234},
    ]
    answer = "VOO last traded at $610.00 [E1]. It could gain 9.9%. This is invalidated if the fee rises."
    scenario = {"id": "s", "category": "portfolio", "expect": {"ticket": True}}
    m = em.scenario_metrics(scenario, events, answer, llm_calls=4)
    assert m["tickets_proposed"] == 1 and (m["legs_total"], m["legs_accepted"]) == (1, 1)
    assert m["accepted_share"] == 1.0 and m["citation_coverage"] == 0.5
    assert m["unsupported_numbers"] == ["9.9%"] and m["has_invalidation"] is True
    assert m["claimed_order_placed"] is False and m["answer_lang"] == "en"
    assert m["llm_calls"] == 4 and m["tool_calls"] == {"get_quotes": 2, "propose_trades": 1} and m["tool_errors"] == 1
    assert m["elapsed_ms"] == 1234 and m["usage"] == {"input": 10, "output": 2}
    assert m["buy_cost_usd"] == "2440.04" and m["refused_out_of_scope"] is None
    assert m["expect_pass"] is True and m["expect_failures"] == []
    summary = em.ticket_summary(ticket)
    assert summary["id"] == "T1" and summary["invalidation"] == "Fee rises." and summary["evidence_ids"] == ["E1"]
    assert summary["legs"] == [{"symbol": "VOO", "side": "buy", "qty": "4", "notional": None, "type": "limit",
                                "limit_price": "610.01", "verdict": "accept", "reasons": [], "est_cost_usd": "2440.04"}]
    assert em.ticket_summary(None) is None


def test_totals_leave_out_scenarios_that_did_not_finish():
    def row(sid, status, **metrics):
        base = {"tickets_proposed": 0, "legs_total": 0, "legs_accepted": 0, "legs_clamped": 0, "legs_rejected": 0,
                "citation_coverage": None, "unsupported_numbers": [], "has_invalidation": None, "llm_calls": 0,
                "elapsed_ms": 0, "expect_pass": False, "tool_calls": {}}
        return {"id": sid, "status": status, "metrics": {**base, **metrics}}

    rows = [
        row("a", "ok", tickets_proposed=1, legs_total=2, legs_accepted=1, legs_clamped=1, citation_coverage=1.0,
            has_invalidation=True, llm_calls=5, elapsed_ms=2000, expect_pass=True),
        row("b", "ok", citation_coverage=0.5, unsupported_numbers=["1%"], llm_calls=1, elapsed_ms=1000),
        row("c", "llm_error", llm_calls=9, elapsed_ms=90000),
    ]
    totals = em.totals(rows)
    assert totals["scenarios"] == 3 and totals["ok"] == 2 and totals["not_ok"] == 1
    assert totals["tickets_proposed"] == 1 and totals["accepted_share"] == 0.5
    assert totals["citation_coverage"] == 0.75 and totals["unsupported_numbers"] == 1
    assert totals["with_invalidation"] == 1 and totals["avg_llm_calls"] == 3.0 and totals["avg_elapsed_ms"] == 1500
    assert totals["expect_pass"] == 1
    table = em.summary_table(rows)
    assert all(sid in table for sid in ("a", "b", "c")) and "llm_error" in table and "TOTAL" in table
