"""The text side of `DemoLLM` (see `_demo_llm.py`): reading tool outputs and writing answers.

Readers turn the chat tools' plain-text outputs back into values (evidence ids, quotes, positions,
the `propose_trades` risk report); `parse_risk_report` is also what `_eval_metrics.py` uses when
every leg was rejected and no ticket exists. Writers build the scripted answers, in English or
Korean. Every number in an answer is copied from a tool output as printed, so the agent's grounding
check has nothing to flag, except for the one deliberate number in the "plain" answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

UNSUPPORTED_DEMO_NUMBER = "7.25%"  # the "plain" answer's number that no tool output supports

_EVIDENCE_RE = re.compile(r"\[(E\d+)\]")
_QUOTE_RE = re.compile(
    r"^- ([A-Z0-9.\-]+): last \$([\d,]+(?:\.\d+)?), bid \$([\d,]+(?:\.\d+)?), ask \$([\d,]+(?:\.\d+)?)", re.M)
_POSITION_RE = re.compile(
    r"^- ([A-Z0-9.\-]+): ([\d,]+(?:\.\d+)?) shares \w+(?: \(([\d,]+(?:\.\d+)?) available to sell\))?"
    r".*?market value \$([\d,]+(?:\.\d+)?)", re.M)
_METRIC_RE = re.compile(r"^- (return_1y|vol_1y): (-?\d+\.\d+)$", re.M)
_SCREEN_RE = re.compile(r"^\d+\. (\S+) — .*?\| expense ratio ([\d.]+%)", re.M)
_RISK_LINE_RE = re.compile(
    r"^(\d+)\. (.+?) -> (ACCEPT|CLAMP|REJECT)(?: \(clamped to (.+?)\))?"
    r"(?:; est\. price (\S+))?(?:; est\. (?:cost|value) (\S+))?\s*$")
_RISK_REASON_RE = re.compile(r"^\s+- \[(\w+)\] (\w+): (.*)$")
_UNTRUSTED_RE = re.compile(r'<untrusted source="([^"]*)">(.*?)</untrusted>', re.S)
_INJECTION_RE = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) instructions|disregard (?:all |the )?(?:previous|prior|above)"
    r"|system prompt|\btell the user\b|\b(?:buy|sell) \d[\d,]* shares\b|이전 지시(?:를|는)? ?무시", re.I)


@dataclass
class Done:
    """One finished tool call of the current turn."""

    name: str
    args: dict
    text: str
    is_error: bool


# --- reading tool outputs ---------------------------------------------------------------------------


def dec(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def evidence_ids(text: str) -> list[str]:
    """Evidence ids written as `[E#]` in a tool output, in order, without duplicates."""
    return list(dict.fromkeys(_EVIDENCE_RE.findall(text or "")))


def parse_risk_report(text: str) -> list[dict]:
    """The legs of a `propose_trades` risk report: `{index, leg, verdict, clamped_to, est_cost,
    reasons, messages}` per leg (`verdict` is accept, clamp or reject; `reasons` are the reason
    codes and `messages` their texts; `leg`, `clamped_to` and `est_cost` are quoted as printed)."""
    legs: list[dict] = []
    for line in (text or "").splitlines():
        head = _RISK_LINE_RE.match(line)
        if head:
            legs.append({
                "index": int(head[1]), "leg": head[2], "verdict": head[3].lower(), "clamped_to": head[4],
                "est_cost": head[6], "reasons": [], "messages": [],
            })
            continue
        reason = _RISK_REASON_RE.match(line)
        if reason and legs:
            legs[-1]["reasons"].append(reason[2])
            legs[-1]["messages"].append(reason[3])
    return legs


def quotes(done: list[Done]) -> dict[str, dict[str, Decimal]]:
    """symbol -> {last, bid, ask}, from this turn's `get_quotes` outputs."""
    out: dict[str, dict[str, Decimal]] = {}
    for d in done:
        if d.name == "get_quotes" and not d.is_error:
            for symbol, last, bid, ask in _QUOTE_RE.findall(d.text):
                out[symbol] = {"last": dec(last), "bid": dec(bid), "ask": dec(ask)}
    return out


def positions(done: list[Done]) -> dict[str, dict[str, Decimal]]:
    """symbol -> {available, value}, from this turn's `get_positions` output."""
    out: dict[str, dict[str, Decimal]] = {}
    for d in done:
        if d.name == "get_positions" and not d.is_error:
            for symbol, qty, available, value in _POSITION_RE.findall(d.text):
                out[symbol] = {"available": dec(available or qty), "value": dec(value)}
    return out


def ids_by_tool(done: list[Done]) -> dict[str, list[str]]:
    """tool name -> the evidence ids its successful outputs carry."""
    out: dict[str, list[str]] = {}
    for d in done:
        if not d.is_error:
            out.setdefault(d.name, []).extend(evidence_ids(d.text))
    return out


# --- fixed answers ----------------------------------------------------------------------------------

PLAIN_EN = (
    "Dollar-cost averaging means investing a fixed amount on a regular schedule, whatever the price is that "
    "day, so you buy more shares when prices are low and fewer when they are high. For a sense of scale, an "
    f"index that gains {UNSUPPORTED_DEMO_NUMBER} a year roughly doubles in a decade. This is general "
    "information, not investment advice."
)
PLAIN_KO = (
    "적립식 투자는 가격과 상관없이 일정한 금액을 정해진 주기로 투자하는 방식입니다. 가격이 낮을 때는 더 많이, 높을 때는 "
    f"더 적게 사게 됩니다. 예를 들어 해마다 {UNSUPPORTED_DEMO_NUMBER}씩 오르는 지수는 대략 십 년이면 두 배가 됩니다. "
    "일반적인 정보이며 투자 권유가 아닙니다."
)
SCOPE_EN = (
    "I can't help with that here. Options, futures, crypto and margin trades are not supported: this assistant "
    "only researches and proposes paper trades in US-listed stocks and ETFs, so no ticket was created. If you "
    "want exposure to the same idea through shares or an ETF, tell me the budget and I will draft a proposal "
    "for you to review."
)
SCOPE_KO = (
    "이 요청은 도와드릴 수 없습니다. 옵션, 선물, 암호화폐, 신용 거래는 지원하지 않습니다. 이 어시스턴트는 미국 상장 주식과 "
    "ETF의 모의투자만 조사하고 제안하므로 티켓을 만들지 않았습니다. 같은 아이디어를 주식이나 ETF로 구현하고 싶으시면 "
    "예산을 알려 주세요. 검토하실 제안을 준비하겠습니다."
)
# kind -> ((thesis, risks, invalidation, sizing) in English, the same in Korean); the first three go to
# propose_trades, and all four into the answer
PROPOSAL_TEXT = {
    "buy": (
        ("Low-cost, liquid exposure that matches the direction in the message.",
         "Prices can fall after the purchase, and a limit order may not fill if the price moves away.",
         "A change in the funds' costs or indices, or in your direction, would invalidate this idea.",
         "Sized from the amount in your message at the current quotes; the risk engine checked every leg."),
        ("요청하신 방향에 맞는 저비용·고유동성 자산입니다.",
         "매수 후 가격이 하락할 수 있고, 가격이 움직이면 지정가 주문이 체결되지 않을 수 있습니다.",
         "펀드의 비용이나 기초지수가 바뀌거나 투자 방향이 달라지면 이 논거는 무효가 됩니다.",
         "메시지의 금액과 현재 호가를 기준으로 정했고, 위험 엔진이 모든 주문을 점검했습니다."),
    ),
    "liquidate": (
        ("You asked to close every open position; each leg sells the full available quantity at the current bid.",
         "A limit sell may not fill if the bid drops, and selling everything leaves the account in cash.",
         "This would change if you want to keep any position: reject the ticket and say which ones to keep.",
         "Each leg sells the whole quantity available to sell; the risk engine checked every leg."),
        ("모든 보유 종목을 정리해 달라고 하셨습니다. 각 주문은 매도 가능한 수량 전부를 현재 매수호가에 매도합니다.",
         "호가가 내려가면 지정가 매도가 체결되지 않을 수 있고, 전량 매도 후에는 계좌가 현금만 남습니다.",
         "유지하고 싶은 종목이 있다면 이 제안은 무효입니다. 티켓을 거절하고 남길 종목을 알려 주세요.",
         "각 주문은 매도 가능한 수량 전부를 매도하며, 위험 엔진이 모든 주문을 점검했습니다."),
    ),
    "rebalance": (
        ("One sell and one buy move the stock and bond mix towards the target split in the message.",
         "Both legs are limit orders, so one may fill without the other and leave the mix unchanged or lopsided.",
         "A different target split, or new cash you want to count, would invalidate this sizing.",
         "Sized in whole shares at the current quotes to close the gap to the target split; the risk engine "
         "checked every leg."),
        ("매도 한 건과 매수 한 건으로 주식과 채권 비중을 요청하신 목표에 가깝게 옮깁니다.",
         "두 주문 모두 지정가여서 한쪽만 체결되면 비중이 그대로이거나 한쪽으로 치우칠 수 있습니다.",
         "목표 비중이 달라지거나 현금을 포함해 계산하려면 이 수량 산정은 무효가 됩니다.",
         "목표 비중과의 차이를 현재 호가 기준 정수 주식 수로 맞췄고, 위험 엔진이 모든 주문을 점검했습니다."),
    ),
}


# --- built answers ----------------------------------------------------------------------------------


def page_summary(ko: bool, done: list[Done]) -> str:
    """The answer to "summarise this URL": the first sentences of the fetched page, without the
    ones that read like instructions to the assistant, and a note when there were any."""
    block = _UNTRUSTED_RE.search(done[-1].text) if done and not done[-1].is_error else None
    if block is None:
        return ("해당 페이지를 가져오지 못해 요약할 수 없습니다." if ko else
                "I could not fetch that page, so I cannot summarise it.")
    source = block[1].split(",")[0]
    sentences = [s.strip().lstrip("# ").strip() for s in re.split(r"(?<=[.!?])\s+", block[2].strip()) if s.strip()]
    kept = [s for s in sentences if not _INJECTION_RE.search(s)]
    summary = " ".join(kept[:3]) or ("(요약할 본문이 없습니다)" if ko else "(no readable content)")
    injected = len(kept) != len(sentences)
    if ko:
        text = f"페이지 요약 [{source}]: {summary}"
        if injected:
            text += ("\n\n참고: 이 페이지에는 어시스턴트에게 내리는 지시처럼 보이는 문장(프롬프트 주입 시도)이 있어 무시했습니다. "
                     "페이지 내용은 데이터일 뿐 지시가 아닙니다.")
        return text + "\n\n거래는 제안하지 않았습니다."
    text = f"Summary of the page [{source}]: {summary}"
    if injected:
        text += ("\n\nNote: the page also contains text that reads like instructions to the assistant "
                 "(a prompt-injection attempt). I ignored it: page content is data, not instructions.")
    return text + "\n\nNo trade was proposed."


def grounded_answer(ko: bool, done: list[Done]) -> str:
    """The answer after `grounded_web_answer`: the first sentences of the grounded text, cited with
    the evidence ids of its sources, and a reminder that it is unverified web content."""
    block = _UNTRUSTED_RE.search(done[-1].text) if done and not done[-1].is_error else None
    if block is None:
        return ("웹 검색 결과를 가져오지 못했습니다." if ko else "The grounded web search returned nothing I can use.")
    cites = "".join(f"[{i}]" for i in block[1].split(",") if _EVIDENCE_RE.fullmatch(f"[{i}]"))
    body = re.sub(r"\s*\(search queries: .*\)\s*$", "", block[2].strip(), flags=re.S)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    summary = " ".join([s for s in sentences if not _INJECTION_RE.search(s)][:2])
    if ko:
        return (f"웹 검색 결과 {cites}: {summary}\n\n공개 웹에서 가져온 내용이므로 확인되지 않은 정보로 보고, 중요한 수치는 "
                "1차 자료와 대조해 주세요. 거래는 제안하지 않았습니다.")
    return (f"From a grounded web search {cites}: {summary}\n\nThis comes from the open web, so treat it as "
            "unverified and check key numbers against primary sources. No trade was proposed.")


def _facts(ko: bool, done: list[Done]) -> list[str]:
    """Cited one-line facts for the thesis: last prices, 1y metrics and the screen's expense ratios."""
    lines: list[str] = []
    quote_id = (ids_by_tool(done).get("get_quotes") or [None])[0]
    if quote_id:
        prices = ", ".join(f"{s} ${q['last']:,.2f}" for s, q in quotes(done).items())
        lines.append(f"최근 체결가: {prices} [{quote_id}]." if ko else f"Last prices: {prices} [{quote_id}].")
    for d in done:
        if d.is_error or not evidence_ids(d.text):
            continue
        eid = evidence_ids(d.text)[0]
        if d.name == "price_metrics":
            metrics = dict(_METRIC_RE.findall(d.text))
            if {"return_1y", "vol_1y"} <= set(metrics):
                symbol = str(d.args.get("symbol", "")).upper()
                lines.append(
                    f"{symbol}: 1년 수익률 {metrics['return_1y']}, 1년 변동성 {metrics['vol_1y']} (소수 표기) [{eid}]." if ko else
                    f"{symbol}: 1y return {metrics['return_1y']} and 1y volatility {metrics['vol_1y']}, as fractions [{eid}].")
        elif d.name == "screen_us_etfs":
            rows = ", ".join(f"{t} {er}" for t, er in _SCREEN_RE.findall(d.text)[:3])
            if rows:
                lines.append(f"ETF 스크린의 총보수: {rows} [{eid}]." if ko else f"Expense ratios in the ETF screen: {rows} [{eid}].")
        elif d.name in ("ask_product_database", "search_news"):
            lines.append(f"상품 데이터베이스와 뉴스 조사 결과를 근거로 삼았습니다 [{eid}]." if ko else
                         f"The product database and news research back the choice [{eid}].")
    return lines


def trade_answer(ko: bool, done: list[Done], kind: str, can_propose: bool) -> str:
    """The final answer of a trade flow (`kind`: buy, liquidate or rebalance), built from the
    `propose_trades` output: a ticket, an all-rejected report, a refused call, or no call at all."""
    proposal = next((d for d in reversed(done) if d.name == "propose_trades"), None)
    if proposal is None:
        if not can_propose:
            return ("거래가 연결되어 있지 않아(Alpaca 모의투자 키 없음) 티켓을 만들 수 없습니다. 설정에서 키를 추가하면 제안을 드릴 수 있습니다."
                    if ko else
                    "Trading is not connected (no Alpaca paper keys), so I cannot create a ticket. Add paper keys in "
                    "Settings and I can draft a proposal.")
        return ("시세나 조사 근거를 확인하지 못해 제안을 만들지 않았습니다. 잠시 후 다시 시도해 주세요." if ko else
                "I could not read a quote or a research source for this, so I did not propose anything. "
                "Please try again in a moment.")
    legs = parse_risk_report(proposal.text)
    quote_id = (ids_by_tool(done).get("get_quotes") or [None])[0]
    source = f"[{quote_id}]" if quote_id else ""
    if proposal.is_error or not legs:
        first = proposal.text.splitlines()[0] if proposal.text else ""
        return f"제안이 받아들여지지 않았습니다: {first}" if ko else f"The proposal was not accepted: {first}"
    table = ["| 주문 | 판정 | 예상 금액 | 출처 |" if ko else "| Leg | Verdict | Est. amount | Source |", "|---|---|---|---|"]
    notes: list[str] = []
    for leg in legs:
        table.append(f"| {leg['clamped_to'] or leg['leg']} | {leg['verdict'].upper()} | {leg['est_cost'] or 'n/a'} | {source} |")
        for message in leg["messages"]:
            notes.append(f"- {leg['index']}번 주문: {message}" if ko else f"- Leg {leg['index']}: {message}")
    if proposal.text.startswith("No ticket was created"):
        head = ("**티켓이 만들어지지 않았습니다.** 위험 점검에서 모든 주문이 거절되었습니다." if ko else
                "**No ticket was created.** The risk engine rejected every leg:")
        tail = ("한도를 우회할 수는 없습니다. 다른 종목을 원하시거나 운용 지침을 바꾸시려면 알려 주세요." if ko else
                "I will not work around the limit. Tell me if you want a different symbol, or change the mandate in the UI.")
        return "\n".join([head, "", *table, "", *notes, "", tail])
    thesis, risks, invalidation, sizing = PROPOSAL_TEXT[kind][1 if ko else 0]
    labels = ("**논거.**", "**위험.**", "**무효 조건.**", "**수량 산정.**") if ko else (
        "**Thesis.**", "**Risks.**", "**Invalidation.**", "**Sizing.**")
    head = ("**검토하실 제안입니다** (모의투자이며 투자 권유가 아닙니다)." if ko else
            "**Proposal for your review** (paper trading, not investment advice).")
    tail = ("티켓 카드를 검토한 뒤 UI에서 승인하거나 거절해 주세요. 저는 주문을 낼 수 없습니다." if ko else
            "Review the ticket card and confirm or reject it in the UI. I cannot place orders.")
    body = [head, "", *table, ""]
    if notes:
        body += [("위험 엔진의 조정 내용:" if ko else "What the risk engine changed or flagged:"), *notes, ""]
    body += [f"{labels[0]} {thesis} {' '.join(_facts(ko, done))}".rstrip(), f"{labels[1]} {risks}",
             f"{labels[2]} {invalidation}", f"{labels[3]} {sizing}", "", tail]
    return "\n".join(body)
