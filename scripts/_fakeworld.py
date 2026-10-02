"""A complete offline world for the chat/trading stack: a real `ChatRuntime` over fakes.

`build_fake_world(llm=..., tmp_dir=...)` wires the production composition (`finagent.app_state.
ChatRuntime`: sessions, evidence, tickets, ledger, executor, toolbox, agent) to

- `tests.fakes_trading.FakeBroker` and a `FakeData` (here `DemoData`, which adds bars, news and
  movers) through the runtime's `broker_factory` / `data_factory`. The Alpaca key settings hold the
  dummy strings "demo-key-id" / "demo-secret-key" so that the runtime builds a broker at all, and
  the helper refuses to return a world whose broker is not the `FakeBroker`: `AlpacaPaper` is never
  constructed here;
- an `HttpClient` over `httpx.MockTransport` that serves only the `web_pages` it was given (to the
  Jina Reader and Tavily extract/search paths) and refuses everything else, so there is no network;
- a small product database: a temp SQLite `v_etf_gl` with 12 US ETF rows (so `screen_us_etfs`
  works), a registry and a Q&A orchestrator over it. `real_products=True` loads the real QA stack
  instead (`AppState.load`, about 75 s; it reads `data/store/`).

All state (the app DB, the env file, the product DB) lives under `tmp_dir`; the real `.env` and
`data/store/` are never written. Prices, positions, bars and news are made-up demo values.

Used by `scripts/eval_scenarios.py` and by the demo server. `DemoLLM` (see `_demo_llm.py`) is the
scripted model that goes with it.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import re
import sqlite3
import sys
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", pathlib.Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(ROOT) not in sys.path:  # for `tests.fakes_trading`
    sys.path.append(str(ROOT))

import httpx  # noqa: E402

from _demo_llm import DemoLLM  # noqa: E402,F401 - re-exported: the scripted model for this world
from finagent.agent.tools import ToolResult  # noqa: E402
from finagent.app_state import ChatRuntime  # noqa: E402
from finagent.config import Settings  # noqa: E402
from finagent.kb.rdb import RDB  # noqa: E402
from finagent.models import AnswerResponse  # noqa: E402
from finagent.research.evidence import ResearchResult, sanitize_untrusted, wrap_untrusted  # noqa: E402
from finagent.research.http import HttpClient  # noqa: E402
from finagent.trading.models import Bar, ClockInfo, PositionSnapshot  # noqa: E402
from tests.fakes_trading import FakeBroker, FakeData, make_account, make_asset, make_quote  # noqa: E402

DEMO_KEY_ID = "demo-key-id"  # not secrets: they only make the runtime build a (fake) broker
DEMO_SECRET = "demo-secret-key"
ETF_BASE_DATE = "2026-08-22"

# symbol -> (demo last price, name). Made-up prices, not market data.
UNIVERSE: dict[str, tuple[str, str]] = {
    "VOO": ("610.00", "Vanguard S&P 500 ETF"),
    "VTI": ("330.00", "Vanguard Total Stock Market ETF"),
    "SPY": ("665.00", "SPDR S&P 500 ETF Trust"),
    "QQQ": ("590.00", "Invesco QQQ Trust"),
    "SCHD": ("28.00", "Schwab U.S. Dividend Equity ETF"),
    "VYM": ("140.00", "Vanguard High Dividend Yield ETF"),
    "BND": ("74.00", "Vanguard Total Bond Market ETF"),
    "AGG": ("100.00", "iShares Core U.S. Aggregate Bond ETF"),
    "XLK": ("280.00", "Technology Select Sector SPDR Fund"),
    "XLE": ("90.00", "Energy Select Sector SPDR Fund"),
    "SMH": ("330.00", "VanEck Semiconductor ETF"),
    "GLD": ("350.00", "SPDR Gold Shares"),
    "AAPL": ("250.00", "Apple Inc."),
    "MSFT": ("510.00", "Microsoft Corporation"),
    "NVDA": ("180.00", "NVIDIA Corporation"),
    "GOOGL": ("240.00", "Alphabet Inc."),
    "AMZN": ("230.00", "Amazon.com, Inc."),
    "XYZ": ("70.00", "Block, Inc."),
}
_BOND_LIKE = frozenset({"BND", "AGG"})
SEED_CASH = Decimal("100000")
SEED_POSITIONS = (("VTI", "40", "300.00"), ("AAPL", "30", "220.00"), ("BND", "100", "73.00"))  # qty, avg entry

# ticker, name, expense ratio %, AUM USD, asset type, index, strategy: demo values in the shape of v_etf_gl
ETF_ROWS = (
    ("VOO", "Vanguard S&P 500 ETF", 0.03, 760e9, "Equity", "S&P 500 TR", "Tracks large-cap US stocks"),
    ("SPY", "SPDR S&P 500 ETF Trust", 0.0945, 680e9, "Equity", "S&P 500 TR", "Tracks large-cap US stocks"),
    ("VTI", "Vanguard Total Stock Market ETF", 0.03, 540e9, "Equity", "CRSP US Total Market TR", "Tracks the whole US stock market"),
    ("QQQ", "Invesco QQQ Trust, Series 1", 0.2, 385e9, "Equity", "NASDAQ 100 TR", "Tracks the largest Nasdaq non-financial stocks"),
    ("BND", "Vanguard Total Bond Market ETF", 0.03, 140e9, "Bond", "Bloomberg US Aggregate Float Adjusted TR", "Tracks investment-grade US bonds"),
    ("AGG", "iShares Core U.S. Aggregate Bond ETF", 0.03, 130e9, "Bond", "Bloomberg US Aggregate Bond TR", "Tracks investment-grade US bonds"),
    ("GLD", "SPDR Gold Shares", 0.4, 110e9, "Commodity", "LBMA Gold Price PM", "Holds physical gold"),
    ("XLK", "Technology Select Sector SPDR Fund", 0.08, 90e9, "Equity", "Technology Select Sector TR", "Tracks S&P 500 technology companies"),
    ("SCHD", "Schwab U.S. Dividend Equity ETF", 0.06, 71e9, "Equity", "Dow Jones US Dividend 100 TR", "Tracks quality dividend payers"),
    ("VYM", "Vanguard High Dividend Yield ETF", 0.06, 66e9, "Equity", "FTSE High Dividend Yield TR", "Tracks high dividend yield stocks"),
    ("SMH", "VanEck Semiconductor ETF", 0.35, 30e9, "Equity", "MVIS US Listed Semiconductor 25 TR", "Tracks semiconductor companies"),
    ("XLE", "Energy Select Sector SPDR Fund", 0.08, 27e9, "Equity", "Energy Select Sector TR", "Tracks S&P 500 energy companies"),
)
_NEWS = (
    ("{name} draws steady inflows as investors favour low-cost funds",
     "Demo newswire: flows into {symbol} stayed positive this week while trading volume was close to its recent average."),
    ("Commentators weigh the outlook for {symbol}",
     "Demo newswire: views on the near-term outlook for {name} are split. This is placeholder text, not real news."),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- market data --------------------------------------------------------------------------------------


def demo_bars(symbol: str, last: Decimal, end_day: date, count: int = 520) -> list[Bar]:
    """`count` daily bars ending on `end_day` (weekdays only) whose last close is `last`: a seeded
    random walk with a fixed average drift, so the same symbol and day always give the same series
    and the returns stay in a plausible range."""
    rng = random.Random(zlib.crc32(symbol.encode("utf-8")))
    calm = symbol in _BOND_LIKE
    drift = (0.00005 if calm else 0.0002) + rng.random() * (0.0001 if calm else 0.0004)
    vol = (0.002 if calm else 0.006) + rng.random() * (0.001 if calm else 0.008)
    noise = [rng.gauss(0.0, vol) for _ in range(count - 1)]
    mean = sum(noise) / len(noise)
    level, path = 1.0, [1.0]
    for step in noise:
        level *= math.exp(drift + step - mean)
        path.append(level)
    days: list[date] = []
    day = end_day
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()
    scale = float(last) / path[-1]
    closes = [Decimal(f"{value * scale:.2f}") for value in path[:-1]] + [last]
    bars, previous = [], closes[0]
    for when, close in zip(days, closes):
        bars.append(Bar(t=f"{when.isoformat()}T04:00:00Z", o=previous, h=max(previous, close),
                        l=min(previous, close), c=close, v=Decimal(1_000_000)))
        previous = close
    return bars


class DemoData(FakeData):
    """`FakeData` plus what the chat tools also need: fresh quote timestamps, deterministic daily
    bars, placeholder news and movers. No network."""

    def __init__(self, quotes=None, now=_utcnow):
        super().__init__(quotes)
        self._now = now

    async def snapshots(self, symbols):
        found = await super().snapshots(symbols)
        stamp = self._now().strftime("%Y-%m-%dT%H:%M:%SZ")
        return {s: q.model_copy(update={"as_of": stamp}) for s, q in found.items()}

    async def bars(self, symbol, timeframe="1Day", start=None, end=None, limit=500):
        quote = self.quotes.get(str(symbol).upper())
        if quote is None or quote.last is None:
            return []
        end_day = self._now().date() - timedelta(days=1)
        while end_day.weekday() >= 5:
            end_day -= timedelta(days=1)
        bars = demo_bars(quote.symbol, quote.last, end_day)
        if start:
            bars = [b for b in bars if b.t[:10] >= str(start)[:10]]
        return bars[:limit]

    async def news(self, session_id, evidence, symbols=None, start=None, limit=10):
        today = self._now().date().isoformat()
        sources, lines = [], []
        for symbol in (symbols or ["VOO", "QQQ", "BND"]):
            name = UNIVERSE.get(symbol, ("", symbol))[1]
            for i, (headline, summary) in enumerate(_NEWS):
                if len(lines) >= limit:
                    break
                title = sanitize_untrusted(headline.format(symbol=symbol, name=name), max_chars=200)
                body = sanitize_untrusted(summary.format(symbol=symbol, name=name), max_chars=1000)
                source = evidence.add(
                    session_id, connector="news", title=title,
                    url=f"https://news.example.com/{symbol.lower()}/{i + 1}", publisher="Demo Newswire", tier=2,
                    published_at=today, snippet=f"{title} {body}", untrusted=True,
                )
                sources.append(source)
                lines.append(f"[{source.id}] {wrap_untrusted(source.id, f'{title} — {body} (Demo Newswire, {today})')}")
        if not lines:
            return ResearchResult(text="No news found.", sources=[])
        return ResearchResult(text="\n".join(lines), sources=sources)

    async def movers(self, top=10):
        return {
            "gainers": [{"symbol": "NVDA", "percent_change": 3.2, "change": 5.58, "price": 180.0},
                        {"symbol": "XLK", "percent_change": 1.4, "change": 3.87, "price": 280.0}][:top],
            "losers": [{"symbol": "XLE", "percent_change": -1.9, "change": -1.74, "price": 90.0}][:top],
            "market_type": "stocks",
            "last_updated": self._now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    async def most_actives(self, top=10):
        return [{"symbol": "NVDA", "volume": 182_000_000, "trade_count": 1_250_000},
                {"symbol": "AAPL", "volume": 54_000_000, "trade_count": 610_000},
                {"symbol": "SPY", "volume": 48_000_000, "trade_count": 520_000}][:top]


def seeded_broker(now: datetime) -> FakeBroker:
    """A paper account with $100,000 cash and three positions (VTI, AAPL, BND), market open."""
    positions = {}
    for symbol, qty, entry in SEED_POSITIONS:
        q, avg, price = Decimal(qty), Decimal(entry), Decimal(UNIVERSE[symbol][0])
        positions[symbol] = PositionSnapshot(
            symbol=symbol, qty=q, qty_available=q, avg_entry_price=avg, current_price=price, market_value=q * price,
            unrealized_pl=q * (price - avg), unrealized_plpc=((price - avg) / avg).quantize(Decimal("0.0001")),
            side="long",
        )
    equity = SEED_CASH + sum((p.market_value for p in positions.values()), Decimal(0))
    account = make_account(equity=equity, cash=SEED_CASH, buying_power=SEED_CASH,
                           non_marginable_buying_power=SEED_CASH, last_equity=equity)
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    clock = ClockInfo(is_open=True, timestamp=now.strftime(stamp),
                      next_close=(now + timedelta(hours=4)).strftime(stamp),
                      next_open=(now + timedelta(hours=21)).strftime(stamp))
    return FakeBroker(account=account, positions=positions, assets={s: make_asset(s) for s in UNIVERSE},
                      clock=clock)


# --- HTTP: only the given pages -------------------------------------------------------------------------


class FakeWeb:
    """The fake world's whole internet. `pages` maps a URL to its HTML/markdown; they are served to
    the Jina Reader path (`GET https://r.jina.ai/<url>`) and to Tavily extract and search. A page that
    is not there is a 404, any other host is refused with `httpx.ConnectError`; both are listed in
    `refused`, and every request in `requests`, as (method, url)."""

    def __init__(self, pages: dict[str, str] | None = None):
        self.pages = dict(pages or {})
        self.requests: list[tuple[str, str]] = []
        self.refused: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append((request.method, url))
        host, path = request.url.host, request.url.path
        if host == "r.jina.ai" and request.method == "GET":
            target = url.split("r.jina.ai/", 1)[1]
            if target in self.pages:
                return httpx.Response(200, text=self.pages[target])
        elif host == "api.tavily.com" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            if path == "/extract":
                target = (body.get("urls") or [""])[0]
                if target in self.pages:
                    return httpx.Response(200, json={"results": [{"url": target, "raw_content": self.pages[target]}]})
            elif path == "/search":
                words = set(re.findall(r"\w{4,}", str(body.get("query", "")).lower()))
                hits = [{"title": u, "url": u, "content": text[:1000]} for u, text in self.pages.items()
                        if words & set(re.findall(r"\w{4,}", text.lower()))]
                return httpx.Response(200, json={"results": hits[: int(body.get("max_results") or 5)]})
        else:
            self.refused.append((request.method, url))
            raise httpx.ConnectError("fake world: no network", request=request)
        self.refused.append((request.method, url))
        return httpx.Response(404, text="not found (fake world serves only the pages it was given)")

    def client(self) -> HttpClient:
        return HttpClient(client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)), rates={})


# --- products -------------------------------------------------------------------------------------------


def build_etf_db(path: Path) -> RDB:
    """A tiny product store with `v_etf_gl` (12 demo US ETF rows) and its `meta` row."""
    con = sqlite3.connect(path)
    con.execute(
        "create table v_etf_gl (ticker text, pd_nm text, expense_ratio_pct real, du_last_aum real, "
        "wu_inv_ast_type text, cu_base_index text, cu_strtegy text, is_on_sale integer, base_date text)")
    con.executemany(f"insert into v_etf_gl values (?,?,?,?,?,?,?,1,'{ETF_BASE_DATE}')", ETF_ROWS)
    con.execute("create table meta (table_name, source_id, korean_name, base_date, row_count, built_at)")
    con.execute("insert into meta values ('etf_gl','DEMO','해외 ETF',?,?,?)", (ETF_BASE_DATE, len(ETF_ROWS), ETF_BASE_DATE))
    con.commit()
    con.close()
    return RDB(path)


def _etf_matches(rdb: RDB, text: str, limit: int = 5) -> list[tuple]:
    """Rows whose ticker is named in `text`, else rows sharing a word with it, else the largest."""
    rows = rdb.fetch_all(
        "select ticker, pd_nm, expense_ratio_pct, du_last_aum, cu_base_index, cu_strtegy from v_etf_gl "
        "order by du_last_aum desc")
    named = [r for r in rows if re.search(rf"\b{re.escape(r[0])}\b", text)]
    if named:
        return named[:limit]
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    shared = [r for r in rows if words & set(re.findall(r"[a-z]{4,}", f"{r[1]} {r[4]} {r[5]}".lower()))]
    return (shared or rows)[:limit]


def _etf_line(row: tuple) -> str:
    ticker, name, er, aum, index, _strategy = row
    return f"{ticker}: {name}; expense ratio {er:g}%; AUM ${aum / 1e9:.2f}B; index {index}"


class DemoRegistry:
    """Duck-typed `ToolRegistry` over the tiny store: `rdb`, `meta` and the two knowledge tools."""

    def __init__(self, rdb: RDB):
        self.rdb = rdb
        self.meta = rdb.meta()

    async def execute(self, name: str, args: dict) -> ToolResult:
        if name == "search_products":
            rows = _etf_matches(self.rdb, str(args.get("query", "")), int(args.get("k") or 5))
            return ToolResult("\n".join(f"- [v_etf_gl] id={_etf_line(r)}" for r in rows))
        if name == "graph_neighbors":
            return ToolResult(f"No graph relations for {str(args.get('entity', ''))[:80]!r} in the demo product database.")
        return ToolResult(f"Unknown tool: {name}", is_error=True)


class DemoOrchestrator:
    """Duck-typed product Q&A agent: answers from the tiny ETF table, without a model."""

    def __init__(self, rdb: RDB):
        self._rdb = rdb

    async def answer(self, question_id: str, question: str, on_event=None) -> AnswerResponse:
        lines = "\n".join(_etf_line(r) for r in _etf_matches(self._rdb, question))
        return AnswerResponse(
            question_id=question_id, question=question, think_trace="demo product database lookup",
            retrieved_context=f"[Evidence] sql: v_etf_gl (base date {ETF_BASE_DATE})\n{lines}",
            answer=f"Matches in the demo ETF master (base date {ETF_BASE_DATE}):\n{lines}",
        )


def build_products(tmp_dir: Path | str, *, llm, real_products: bool = False) -> tuple[Any, Any]:
    """(registry, orchestrator): the tiny demo store under `tmp_dir`, or the real QA stack.

    The real stack is `AppState.load` on a copy of the settings without Alpaca keys and with its
    app DB under `tmp_dir`, so the chat runtime it also builds has no broker and writes nothing
    under `data/store/`. Its `llm` answers `ask_product_database` questions."""
    tmp = Path(tmp_dir)
    if not real_products:
        rdb = build_etf_db(tmp / "products.db")
        return DemoRegistry(rdb), DemoOrchestrator(rdb)
    from finagent.app_state import AppState
    from finagent.config import get_settings

    qa_settings = get_settings().model_copy(update={
        "alpaca_api_key_id": None, "alpaca_api_secret_key": None, "app_db_path": tmp / "qa_app.db",
    })
    state = AppState.load(qa_settings, llm=llm)
    return state.tools, state.orchestrator


# --- the world ------------------------------------------------------------------------------------------


@dataclass
class FakeWorld:
    runtime: ChatRuntime
    settings: Settings
    broker: FakeBroker
    data: DemoData
    web: FakeWeb
    tmp_dir: Path
    registry: Any = None
    orchestrator: Any = None

    @property
    def sessions(self):
        return self.runtime.sessions

    @property
    def tickets(self):
        return self.runtime.tickets

    @property
    def evidence(self):
        return self.runtime.evidence

    @property
    def agent(self):
        return self.runtime.agent

    @property
    def refused(self) -> list[tuple[str, str]]:
        """HTTP requests the fake transport refused, as (method, url)."""
        return self.web.refused

    async def close(self) -> None:
        """Shut the runtime down and release the SQLite files, so `tmp_dir` can be removed."""
        await self.runtime.shutdown()
        for owner in (self.runtime.db, getattr(self.registry, "rdb", None)):
            connection = getattr(owner, "_con", None)
            try:
                if connection is not None:
                    connection.close()
            except Exception:  # noqa: BLE001 - best effort
                pass


def build_fake_world(*, llm, tmp_dir, real_products: bool = False, web_pages: dict[str, str] | None = None,
                     settings_over: dict | None = None) -> FakeWorld:
    """Build the offline world; see the module docstring. `settings_over` overrides `Settings`
    fields (e.g. `chat_max_rounds`); state paths must stay under `tmp_dir`."""
    tmp = Path(tmp_dir).resolve()
    if tmp == ROOT or ROOT / "data" in (tmp, *tmp.parents):
        raise ValueError("fake-world state must live in a scratch directory, never under data/ or the repo root")
    tmp.mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = dict(
        llm_provider="stub", llm_model=str(getattr(llm, "model", None) or getattr(llm, "name", "demo")),
        alpaca_api_key_id=DEMO_KEY_ID, alpaca_api_secret_key=DEMO_SECRET,
        tavily_api_key=None, gemini_api_key=None, anthropic_api_key=None, openai_api_key=None,
        sec_user_agent=None, trading_enabled=True, app_db_path=tmp / "app.db", api_allow_remote=False,
    )
    fields.update(settings_over or {})
    settings = Settings(_env_file=None, **fields)
    if tmp not in Path(settings.app_db_path).resolve().parents:
        raise ValueError("app_db_path must stay under tmp_dir")
    env_path = tmp / ".env"
    env_path.write_text(
        "# fake world: dummy values, no real keys\n"
        f"ALPACA_API_KEY_ID={DEMO_KEY_ID}\nALPACA_API_SECRET_KEY={DEMO_SECRET}\nTRADING_ENABLED=true\n",
        encoding="utf-8")

    now = _utcnow()
    broker = seeded_broker(now)
    data = DemoData({symbol: make_quote(symbol, price) for symbol, (price, _name) in UNIVERSE.items()})
    web = FakeWeb(web_pages)
    registry, orchestrator = build_products(tmp, llm=llm, real_products=real_products)
    runtime = ChatRuntime(
        settings, llm=llm, registry=registry, orchestrator=orchestrator, env_path=env_path,
        broker_factory=lambda key_id, secret: broker,
        data_factory=lambda key_id, secret, http: data,
        http_factory=web.client,
    )
    if not isinstance(runtime.broker, FakeBroker):  # a hard guard, not an assert (asserts vanish under -O)
        raise RuntimeError(
            f"the fake world must run on FakeBroker, got {type(runtime.broker).__name__}: keep both Alpaca "
            "key settings non-empty (they are dummy strings here)")
    return FakeWorld(runtime=runtime, settings=settings, broker=broker, data=data, web=web, tmp_dir=tmp,
                     registry=registry, orchestrator=orchestrator)
