"""`DemoLLM`: a deterministic, offline stand-in for the chat model (no network, no key).

It is an `LLMClient` whose conversation decides every step from the last user message and the tool
results of the current turn, so it drives the real `ChatAgent` and `ChatToolbox` end to end: the
demo server (fake broker) and the offline run of `scripts/eval_scenarios.py` both use it. It is a
script, not a model: its answers show what the pipeline does, not how well a model reasons.

What the last user message triggers, first match wins:

    contains "error"         a simulated step error (`stop_reason="error"`)
    starts with "Modify:"    a new proposal with the legs written in the message
                             ("Modify: buy 3 VOO limit 610; sell 2 AAPL")
    contains "mandate"       `draft_mandate` (budget and max position % read from the message)
    contains "plain"         a text-only answer with one deliberately unsupported number
    contains "grounded"      `grounded_web_answer` (when the session offers it), then a cited summary
    contains a URL           `fetch_source`, then a summary that leaves out injected instructions
    options, futures, ...    a text-only refusal: out of scope
    "sell everything"        account, quotes, then sells of every open position
    "rebalance"              account, quotes, a bond screen, then one sell and one buy
    anything else            `get_account` + `get_positions`, then `get_quotes` + `price_metrics`,
                             then one research tool (`screen_us_etfs`, else `ask_product_database`,
                             else `search_news`), then `propose_trades` (1-2 legs sized from the
                             dollar amount in the message, $2,000 without one), then the answer

A Korean message is answered in Korean. The first tool round also carries interim text, which the
agent emits as `reasoning-delta`. Every number in an answer is copied from a tool output, and
`propose_trades` is only called with evidence ids (`E\\d+`) read from this turn's tool results.
A tool the session does not offer is skipped, and a conversation started without the chat tools
(the product Q&A orchestrator) gets one plain-text reply. The answer texts and the tool-output
readers live in `_demo_text.py`.
"""
from __future__ import annotations

import pathlib
import re
import sys
from decimal import ROUND_FLOOR, Decimal

_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE.parent / "src", _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _demo_text import (  # noqa: E402,F401 - the readers are re-exported for the tests and the eval
    PLAIN_EN,
    PLAIN_KO,
    PROPOSAL_TEXT,
    SCOPE_EN,
    SCOPE_KO,
    UNSUPPORTED_DEMO_NUMBER,
    Done,
    dec,
    evidence_ids,
    grounded_answer,
    ids_by_tool,
    page_summary,
    parse_risk_report,
    positions,
    quotes,
    trade_answer,
)
from finagent.agent.llm import ToolCall, ToolSpec, Turn  # noqa: E402
from finagent.i18n import detect_lang  # noqa: E402

DEFAULT_AMOUNT = Decimal("2000")
BOND_ETFS = frozenset({"BND", "AGG"})
KNOWN_TICKERS = frozenset({
    "VOO", "VTI", "SPY", "QQQ", "SCHD", "VYM", "BND", "AGG", "XLK", "XLE", "SMH", "GLD",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "XYZ",
})
RESEARCH_TOOLS = ("screen_us_etfs", "ask_product_database", "search_news")
_NUDGE_PREFIX = "Budget reached:"

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:달러|dollars?\b|USD\b)", re.I)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_SPLIT_RE = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b")
_MODIFY_LEG_RE = re.compile(
    r"\b(buy|sell)\s+(\d+(?:\.\d+)?)\s+(?:shares?\s+(?:of\s+)?)?([A-Za-z][A-Za-z0-9.\-]{0,9})"
    r"(?:[^;,\n]*?(?:limit|@|at)\s*\$?(\d+(?:\.\d+)?))?", re.I)
_OUT_OF_SCOPE_RE = re.compile(
    r"\b(?:call|put)s?\s+options?\b|\boptions?\s+(?:contract|chain|trad)|\b(?:calls|puts)\b|\bfutures\b"
    r"|\bcrypto|\bbitcoin\b|\bforex\b|옵션|선물|코인|암호화폐", re.I)
_SELL_ALL_RE = re.compile(
    r"\bsell\s+(?:everything|all)\b|\bliquidate\b|\bclose\s+(?:all|every)\b|전부\s?매도|모두\s?매도|전량\s?매도", re.I)
_REBALANCE_RE = re.compile(r"\brebalanc|리밸런싱|재조정", re.I)
_THEMES = (
    (re.compile(r"dividend|income|배당", re.I), ("SCHD", "VYM"), {"keyword": "Dividend", "limit": 5}),
    (re.compile(r"tech|semiconductor|기술|반도체", re.I), ("XLK",), {"keyword": "Technology", "limit": 5}),
    (re.compile(r"\bbonds?\b|채권", re.I), ("BND",), {"asset_type": "Bond", "limit": 5}),
    (re.compile(r"\benergy\b|에너지", re.I), ("XLE",), {"keyword": "Energy", "limit": 5}),
)
_CORE = (("VOO", "BND"), {"sort_by": "expense_ratio", "limit": 5})


# --- reading the user message -----------------------------------------------------------------------


def _amount(message: str) -> Decimal | None:
    match = _AMOUNT_RE.search(message)
    return dec(match[1] or match[2]) if match else None


def _split_user_text(raw: str) -> tuple[str, bool]:
    """The user's own message (without the lines the agent appends) and whether to answer in Korean."""
    cut = raw.rfind("\n\nToday: ")
    if cut < 0:
        return raw, detect_lang(raw) == "ko"
    message, context = raw[:cut], raw[cut:]
    if "Answer language: Korean" in context:
        return message, True
    if "Answer language: English" in context:
        return message, False
    return message, detect_lang(message) == "ko"


def _pick_symbols(message: str) -> tuple[tuple[str, ...], dict]:
    """The symbols to trade and the ETF screen to run: tickers named in the message, else a theme
    (dividend, technology, bonds, energy), else a stock-and-bond core."""
    tickers = tuple(dict.fromkeys(t for t in re.findall(r"\b[A-Z]{1,5}\b", message) if t in KNOWN_TICKERS))
    for pattern, symbols, screen in _THEMES:
        if pattern.search(message):
            return (tickers[:2] or symbols), dict(screen)
    if tickers:
        return tickers[:2], {"sort_by": "aum", "limit": 5}
    return _CORE[0], dict(_CORE[1])


# --- legs -------------------------------------------------------------------------------------------


def _limit_leg(symbol: str, side: str, qty: Decimal, price: Decimal) -> dict:
    return {"symbol": symbol, "side": side, "qty": format(qty.normalize(), "f"), "type": "limit",
            "time_in_force": "day", "limit_price": format(price, "f")}


def _buy_legs(symbols: tuple[str, ...], amount: Decimal, quoted: dict) -> list[dict]:
    """Limit buys at the ask, `amount` split evenly: whole shares, or thousandths below one share."""
    priced = [s for s in symbols if s in quoted]
    legs = []
    for symbol in priced:
        ask = quoted[symbol]["ask"]
        qty = (amount / len(priced) / ask).to_integral_value(rounding=ROUND_FLOOR)
        if qty < 1:
            qty = (amount / len(priced) / ask).quantize(Decimal("0.001"), rounding=ROUND_FLOOR)
        if qty > 0:
            legs.append(_limit_leg(symbol, "buy", qty, ask))
    return legs


def _rebalance_legs(message: str, held: dict, quoted: dict) -> list[dict]:
    """One sell and one buy that move the stock/bond mix towards the target in the message
    (60/40 without one): trim the largest stock position and add to BND, or the reverse."""
    split = _SPLIT_RE.search(message)
    bond_share = Decimal(split[2]) / (Decimal(split[1]) + Decimal(split[2])) if split else Decimal("0.4")
    stocks = {s: p for s, p in held.items() if s not in BOND_ETFS}
    bonds = sum((p["value"] for s, p in held.items() if s in BOND_ETFS), Decimal(0))
    total = bonds + sum((p["value"] for p in stocks.values()), Decimal(0))
    shift = total * bond_share - bonds  # dollars to move from stocks into bonds
    if not stocks or "BND" not in quoted or shift == 0:
        return []
    largest = max(stocks, key=lambda s: stocks[s]["value"])
    sell, buy = (largest, "BND") if shift > 0 else ("BND", largest)
    if sell not in held or sell not in quoted or buy not in quoted:
        return []
    amount = abs(shift)
    sell_qty = min((amount / quoted[sell]["bid"]).to_integral_value(rounding=ROUND_FLOOR), held[sell]["available"])
    buy_qty = (amount / quoted[buy]["ask"]).to_integral_value(rounding=ROUND_FLOOR)
    if sell_qty < 1 or buy_qty < 1:
        return []
    return [_limit_leg(sell, "sell", sell_qty, quoted[sell]["bid"]), _limit_leg(buy, "buy", buy_qty, quoted[buy]["ask"])]


# --- the conversation -------------------------------------------------------------------------------


class DemoConversation:
    def __init__(self, tools: list[ToolSpec], user_text: str):
        self._tools = frozenset(t.name for t in tools)
        self._chat = "draft_mandate" in self._tools  # the chat tool set always has it
        self._messages: list[dict] = [{"role": "user", "content": user_text}]
        self._calls = 0

    # the Conversation protocol
    async def step(self) -> Turn:
        turn = self._decide()
        self._messages.append({"role": "assistant", "content": turn.text, "tool_calls": list(turn.tool_calls)})
        return turn

    def add_tool_results(self, results) -> None:
        self._messages.append({"role": "tool", "results": [(cid, text, bool(err)) for cid, text, err in results]})

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text, "tool_calls": []})

    def transcript(self) -> list[dict]:
        return self._messages

    # helpers
    def _has(self, name: str) -> bool:
        return name in self._tools

    def _say(self, text: str) -> Turn:
        return Turn(text, [], "end_turn", self._usage(text))

    def _call(self, calls: list[tuple[str, dict]], text: str = "") -> Turn:
        out = []
        for name, args in calls:
            self._calls += 1
            out.append(ToolCall(f"demo-{self._calls}", name, args))
        return Turn(text, out, "tool_use", self._usage(text))

    def _usage(self, text: str) -> dict:
        seen = sum(len(str(m.get("content") or m.get("results") or "")) for m in self._messages)
        return {"input": seen // 4, "output": max(1, len(text) // 4)}

    def _done_since(self, index: int) -> list[Done]:
        names: dict[str, tuple[str, dict]] = {}
        done: list[Done] = []
        for message in self._messages[index + 1:]:
            if message["role"] == "assistant":
                names.update({c.id: (c.name, c.input) for c in message["tool_calls"]})
            elif message["role"] == "tool":
                for cid, text, err in message["results"]:
                    name, args = names.get(cid, ("?", {}))
                    done.append(Done(name, args, text, err))
        return done

    # --- deciding ---------------------------------------------------------------------------------

    def _decide(self) -> Turn:
        if not self._chat:
            return self._say("The demo model does not answer product-database questions; use a real model for that.")
        users = [i for i, m in enumerate(self._messages) if m["role"] == "user"]
        index = users[-1]
        raw = self._messages[index]["content"]
        if raw.startswith(_NUDGE_PREFIX):
            _, ko = _split_user_text(self._messages[users[-2]]["content"]) if len(users) > 1 else ("", False)
            return self._say("답변을 마무리할 예산이 부족했습니다. 범위를 좁혀 다시 요청해 주세요." if ko else
                             "I ran out of budget before finishing. Please ask again with a narrower request.")
        message, ko = _split_user_text(raw)
        done = self._done_since(index)
        low = message.lower()
        if "error" in low:
            return Turn("simulated model error", [], "error")
        if message.lstrip().lower().startswith("modify:"):
            return self._trade_flow(message, ko, done, modify=True)
        if "mandate" in low:
            return self._mandate_flow(message, ko, done)
        if "plain" in low:
            return self._say(PLAIN_KO if ko else PLAIN_EN)
        if "grounded" in low:
            return self._grounded_flow(message, ko, done)
        url = _URL_RE.search(message)
        if url:
            if self._has("fetch_source") and not done:
                return self._call([("fetch_source", {"url": url.group().rstrip(".,;")})])
            return self._say(page_summary(ko, done))
        if _OUT_OF_SCOPE_RE.search(message):
            return self._say(SCOPE_KO if ko else SCOPE_EN)
        if _SELL_ALL_RE.search(message):
            return self._liquidate_flow(ko, done)
        if _REBALANCE_RE.search(message):
            return self._rebalance_flow(message, ko, done)
        return self._trade_flow(message, ko, done, modify=False)

    def _account_round(self, ko: bool, called: set[str]) -> Turn | None:
        tools = [t for t in ("get_account", "get_positions") if self._has(t)]
        if not tools or called & set(tools):
            return None
        return self._call([(t, {}) for t in tools],
                          "먼저 모의투자 계좌와 보유 종목을 확인하겠습니다." if ko else
                          "Checking the paper account and its positions before sizing anything.")

    def _quote_round(self, called: set[str], symbols: list[str], metrics_for: str | None) -> Turn | None:
        if not self._has("get_quotes") or "get_quotes" in called:
            return None
        calls: list[tuple[str, dict]] = [("get_quotes", {"symbols": symbols[:10]})]
        if metrics_for and self._has("price_metrics"):
            calls.append(("price_metrics", {"symbol": metrics_for}))
        return self._call(calls)

    def _research_round(self, called: set[str], screen_args: dict, symbols: tuple[str, ...]) -> Turn | None:
        if called & set(RESEARCH_TOOLS):
            return None
        if self._has("screen_us_etfs"):
            return self._call([("screen_us_etfs", screen_args)])
        if self._has("ask_product_database"):
            return self._call([("ask_product_database", {"question": f"Expense ratio and size of {', '.join(symbols)}?"})])
        if self._has("search_news"):
            return self._call([("search_news", {"symbols": list(symbols), "limit": 3})])
        return None

    def _propose(self, legs: list[dict], done: list[Done], ko: bool, kind: str) -> Turn | None:
        """The `propose_trades` call, or None when it was already made, is not offered, there is
        nothing to propose, or no evidence id was read from this turn's tool results (a buy also
        needs a research id)."""
        if not self._has("propose_trades") or any(d.name == "propose_trades" for d in done):
            return None
        ids = ids_by_tool(done)
        research = [i for tool in RESEARCH_TOOLS for i in ids.get(tool, [])[:1]]
        market = [i for tool in ("get_quotes", "price_metrics") for i in ids.get(tool, [])[:1]]
        cited = [*research[:1], *market]
        if not legs or not cited or (any(leg["side"] == "buy" for leg in legs) and not research):
            return None
        thesis, risks, invalidation, _sizing = PROPOSAL_TEXT[kind][1 if ko else 0]
        return self._call([("propose_trades", {
            "legs": legs, "thesis": f"{thesis} [{cited[0]}]", "risks": risks, "invalidation": invalidation,
            "evidence_ids": cited,
        })])

    def _trade_flow(self, message: str, ko: bool, done: list[Done], *, modify: bool) -> Turn:
        called = {d.name for d in done}
        wanted = [(m[1].lower(), m[2], m[3].upper(), m[4]) for m in _MODIFY_LEG_RE.finditer(message)] if modify else []
        symbols, screen_args = _pick_symbols(message)
        if wanted:
            symbols = tuple(dict.fromkeys(w[2] for w in wanted))
        step = (self._account_round(ko, called) or self._quote_round(called, list(symbols), symbols[0])
                or self._research_round(called, screen_args, symbols))
        if step:
            return step
        quoted = quotes(done)
        if wanted:
            legs = []
            for side, qty, symbol, limit in wanted:
                price = limit or quoted.get(symbol, {}).get("ask" if side == "buy" else "bid")
                if price:
                    legs.append(_limit_leg(symbol, side, Decimal(qty), Decimal(str(price))))
        else:
            legs = _buy_legs(symbols, _amount(message) or DEFAULT_AMOUNT, quoted)
        return self._propose(legs, done, ko, "buy") or self._say(trade_answer(ko, done, "buy", self._has("propose_trades")))

    def _liquidate_flow(self, ko: bool, done: list[Done]) -> Turn:
        called = {d.name for d in done}
        step = self._account_round(ko, called)
        if step:
            return step
        held = positions(done)
        if not held:
            return self._say("모의투자 계좌에 보유 종목이 없어 매도할 것이 없습니다." if ko else
                             "There are no open positions in the paper account, so there is nothing to sell.")
        step = self._quote_round(called, list(held), None)
        if step:
            return step
        quoted = quotes(done)
        legs = [_limit_leg(s, "sell", p["available"], quoted[s]["bid"]) for s, p in held.items()
                if s in quoted and p["available"] > 0]
        return (self._propose(legs, done, ko, "liquidate")
                or self._say(trade_answer(ko, done, "liquidate", self._has("propose_trades"))))

    def _rebalance_flow(self, message: str, ko: bool, done: list[Done]) -> Turn:
        called = {d.name for d in done}
        step = self._account_round(ko, called)
        if step:
            return step
        held = positions(done)
        if not held:
            return self._say("보유 종목이 없어 리밸런싱할 것이 없습니다." if ko else
                             "There are no open positions to rebalance.")
        step = (self._quote_round(called, list(dict.fromkeys([*held, "BND"])), "BND")
                or self._research_round(called, {"asset_type": "Bond", "limit": 5}, ("BND",)))
        if step:
            return step
        legs = _rebalance_legs(message, held, quotes(done))
        return (self._propose(legs, done, ko, "rebalance")
                or self._say(trade_answer(ko, done, "rebalance", self._has("propose_trades"))))

    def _grounded_flow(self, message: str, ko: bool, done: list[Done]) -> Turn:
        if not self._has("grounded_web_answer"):
            return self._say("이 세션에서는 grounded 웹 검색을 사용할 수 없습니다." if ko else
                             "Grounded web search is not available in this session.")
        if not done:
            return self._call([("grounded_web_answer", {"question": " ".join(message.split())[:500]})])
        return self._say(grounded_answer(ko, done))

    def _mandate_flow(self, message: str, ko: bool, done: list[Done]) -> Turn:
        if self._has("draft_mandate") and not done:
            args: dict = {"direction_text": " ".join(message.split())[:300]}
            amount, pct = _amount(message), _PCT_RE.search(message)
            if amount is not None:
                args["budget_usd"] = str(amount)
            if pct:
                args["max_position_pct"] = pct[1]
            return self._call([("draft_mandate", args)])
        if done and done[-1].is_error:
            return self._say("운용 지침 초안을 만들지 못했습니다. 조건을 다시 알려 주세요." if ko else
                             "I could not draft the mandate from that message. Please restate the limits.")
        return self._say(
            "말씀하신 내용으로 운용 지침 초안을 만들었습니다. 지침 카드에서 내용을 검토한 뒤 저장해 주세요. 저장하기 전에는 적용되지 않습니다."
            if ko else
            "I drafted a mandate from your message. Review it in the mandate card and save it in the UI; "
            "it is not in effect until you save it.")


class DemoLLM:
    """The scripted model; see the module docstring. `started` counts conversations."""

    name = "demo"
    model = "demo-scripted"

    def __init__(self) -> None:
        self.started = 0

    def start(self, system: str, tools: list[ToolSpec], user_text: str) -> DemoConversation:
        self.started += 1
        return DemoConversation(tools, user_text)
