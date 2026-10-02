"""Check the research connectors with one live call each - no server, no app DB, a few seconds.

  python scripts/check_research.py              # EDGAR, Tavily, Jina (grounding is skipped)
  python scripts/check_research.py --grounding  # ALSO one Gemini grounding call

One call per connector, each with its latency and the number of records it returned:
  * EDGAR      AAPL XBRL company facts, concept Revenues (and the newer revenue concept, since Apple
               switched). SKIPPED when SEC_USER_AGENT is not set.
  * Tavily     one web search. SKIPPED when TAVILY_API_KEY is not set.
  * Jina       fetch https://example.com/ through r.jina.ai (no key). Always keyless on purpose, so it
               tests Jina even when a Tavily key is configured.
  * Grounding  one Gemini google_search question, ONLY with --grounding: every call spends Gemini
               free-tier quota (about 20 requests per day). SKIPPED without a Gemini key.

The evidence store is a temporary database in a temporary directory that is deleted at the end; the
app's own data/store/app.db is never opened. For each step it prints the evidence ids created and the
first 120 characters of each wrapped result. Keys are never printed (and any key value found inside an
error message is replaced by ***).

Exit code: 0 if no attempted step failed (SKIPPED is not a failure), otherwise 1.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import httpx  # noqa: E402

from finagent.config import get_settings  # noqa: E402
from finagent.research.edgar import Edgar  # noqa: E402
from finagent.research.evidence import EvidenceStore, ResearchResult  # noqa: E402
from finagent.research.grounding import GroundedWeb  # noqa: E402
from finagent.research.http import HttpClient  # noqa: E402
from finagent.research.websearch import WebSearch  # noqa: E402
from finagent.store.appdb import AppDB  # noqa: E402

SESSION = "live-check"
EDGAR_TICKER = "AAPL"
# Apple reports revenue under the newer concept now; the classic one is what the brief names.
EDGAR_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
SEARCH_QUERY = "Apple Inc. latest annual report 10-K revenue"
FETCH_URL = "https://example.com/"
GROUNDING_QUESTION = "What is Apple Inc.'s stock ticker symbol and on which exchange does it trade?"
PREVIEW_CHARS = 120
MAX_PREVIEWS = 5


# --------------------------------------------------------------------------------- output helpers


class Reporter:
    """Everything printed goes through `say`, which blanks any known secret first."""

    def __init__(self, secrets: Iterable[str | None] = ()):
        self._secrets = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)
        self.step = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text

    def say(self, text: str = "") -> None:
        print(self.scrub(str(text)))

    def begin(self, label: str) -> None:
        self.step += 1
        self.say(f"[{self.step}] {label}")

    def detail(self, line: str) -> None:
        self.say(f"    {line}")

    def ok(self, seconds: float, summary: str = "") -> None:
        self.passed += 1
        self.say(f"    ✓ {seconds:.2f}s  {summary}".rstrip())

    def fail(self, seconds: float, reason: str, hint: str | None = None) -> None:
        self.failed += 1
        self.say(f"    ✗ {seconds:.2f}s  {reason}")
        if hint:
            self.say(f"      -> {hint}")

    def skip(self, label: str, reason: str) -> None:
        self.step += 1
        self.skipped += 1
        self.say(f"[{self.step}] {label}")
        self.say(f"    - SKIPPED: {reason}")


@dataclass
class Outcome:
    ok: bool
    summary: str
    lines: list[str] = field(default_factory=list)


def describe(exc: BaseException, limit: int = 220) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:limit]


def hint_for(exc: BaseException) -> str | None:
    status = getattr(exc, "status", None)
    if status in (401, 403):
        return "The service rejected the credentials or the User-Agent; check the matching value in .env."
    if status == 429:
        return "Rate limited. Wait a bit and run the check again."
    if isinstance(exc, (httpx.RequestError, TimeoutError, OSError)):
        return "Could not reach the service (network, proxy or firewall?)."
    return None


def preview(line: str) -> str:
    return " ".join(line.split())[:PREVIEW_CHARS]


async def run_step(rep: Reporter, label: str, fn: Callable[[], Awaitable[Outcome]]) -> bool:
    rep.begin(label)
    t0 = time.perf_counter()
    try:
        outcome = await fn()
    except Exception as exc:  # noqa: BLE001 - one failing connector must never stop the others
        rep.fail(time.perf_counter() - t0, describe(exc), hint_for(exc))
        return False
    seconds = time.perf_counter() - t0
    for line in outcome.lines:
        rep.detail(line)
    if outcome.ok:
        rep.ok(seconds, outcome.summary)
    else:
        rep.fail(seconds, outcome.summary)
    return outcome.ok


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def research_outcome(result: ResearchResult) -> Outcome:
    """A `ResearchResult` as a step outcome: ids of the evidence created, then the first 120 characters
    of each wrapped result line. It passes only if it is not an error and has at least one record."""
    n = len(result.sources)
    if result.is_error:
        return Outcome(False, " ".join(result.text.split())[:200] or "connector reported an error")
    lines = []
    if result.sources:
        lines.append("evidence: " + ", ".join(s.id for s in result.sources))
    text_lines = [ln for ln in result.text.splitlines() if ln.strip()]
    lines += [f"{preview(ln)}" for ln in text_lines[:MAX_PREVIEWS]]
    if n == 0:
        return Outcome(False, "0 records", lines or [preview(result.text)])
    return Outcome(True, plural(n, "record"), lines)


# --------------------------------------------------------------------------------- connectors


@dataclass
class Connectors:
    """What the steps use. A connector that is None is SKIPPED, with the reason in `skips`."""

    edgar: Any | None = None
    tavily: Any | None = None
    jina: Any = None
    grounded: Any | None = None
    skips: dict[str, str] = field(default_factory=dict)


def build_connectors(settings, http, evidence: EvidenceStore, with_grounding: bool) -> Connectors:
    """Wire the real connectors the way `ChatRuntime._build` does (same constructors), minus the broker."""
    skips: dict[str, str] = {}
    ua = getattr(settings, "sec_user_agent", None)
    edgar = Edgar(http, evidence, ua) if ua else None
    if edgar is None:
        skips["edgar"] = ("SEC_USER_AGENT is not set. Add a line like SEC_USER_AGENT=Your Name you@example.com "
                          "to .env (SEC blocks anonymous clients).")

    tavily_key = getattr(settings, "tavily_api_key", None)
    tavily = WebSearch(http, evidence, tavily_key) if tavily_key else None
    if tavily is None:
        skips["tavily"] = "TAVILY_API_KEY is not set. Add it to .env (free key at https://tavily.com)."

    # Keyless on purpose: with a Tavily key set, WebSearch.fetch would try Tavily extract first and the
    # Jina fallback would never be exercised.
    jina = WebSearch(http, evidence, None)

    grounded = None
    gemini_key = getattr(settings, "gemini_api_key", None)
    if not with_grounding:
        skips["grounding"] = ("pass --grounding to run it (each call spends Gemini free-tier quota, "
                              "about 20 requests per day).")
    elif not gemini_key:
        skips["grounding"] = "GEMINI_API_KEY is not set. Add it to .env (https://aistudio.google.com/apikey)."
    else:
        grounded = GroundedWeb(http, gemini_key, settings.grounding_model)
    return Connectors(edgar=edgar, tavily=tavily, jina=jina, grounded=grounded, skips=skips)


# --------------------------------------------------------------------------------- steps


async def check_edgar(conns: Connectors, rep: Reporter) -> bool | None:
    label = f"EDGAR: {EDGAR_TICKER} company facts ({EDGAR_CONCEPTS[0]})"
    if conns.edgar is None:
        rep.skip(label, conns.skips.get("edgar", "not configured"))
        return None

    async def go() -> Outcome:
        return research_outcome(await conns.edgar.financial_facts(SESSION, EDGAR_TICKER, EDGAR_CONCEPTS, periods=4))

    return await run_step(rep, label, go)


async def check_tavily(conns: Connectors, rep: Reporter) -> bool | None:
    label = "Tavily: web search"
    if conns.tavily is None:
        rep.skip(label, conns.skips.get("tavily", "not configured"))
        return None

    async def go() -> Outcome:
        return research_outcome(await conns.tavily.search(SESSION, SEARCH_QUERY, max_results=3))

    return await run_step(rep, label, go)


async def check_jina(conns: Connectors, rep: Reporter) -> bool | None:
    async def go() -> Outcome:
        return research_outcome(await conns.jina.fetch(SESSION, FETCH_URL, max_chars=2000))

    return await run_step(rep, f"Jina Reader: fetch {FETCH_URL}", go)


async def check_grounding(conns: Connectors, rep: Reporter) -> bool | None:
    label = "Gemini grounding: one google_search question"
    if conns.grounded is None:
        rep.skip(label, conns.skips.get("grounding", "not configured"))
        return None

    async def go() -> Outcome:
        answer = await conns.grounded.answer(GROUNDING_QUESTION)
        if answer.error:
            return Outcome(False, answer.error)
        lines = ["evidence: none by design (grounded answers are never stored or cached)"]
        if answer.queries:
            lines.append("searches: " + "; ".join(answer.queries[:3]))
        if answer.text.strip():
            lines.append(preview(answer.text))
        if not answer.text.strip():
            return Outcome(False, "the model returned no text", lines)
        return Outcome(True, plural(len(answer.sources), "source"), lines)

    return await run_step(rep, label, go)


async def run_checks(conns: Connectors, rep: Reporter) -> int:
    """Every step runs even if an earlier one failed. Returns the exit code."""
    results = [
        await check_edgar(conns, rep),
        await check_tavily(conns, rep),
        await check_jina(conns, rep),
        await check_grounding(conns, rep),
    ]
    rep.say("-" * 68)
    rep.say(f"{rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped.")
    return 1 if any(r is False for r in results) else 0


# --------------------------------------------------------------------------------- entry point


def dispose_temp_db(db: AppDB | None, directory: pathlib.Path, rep: Reporter) -> None:
    """Close the temp DB's connection (AppDB has no close(); an open handle blocks deletion on Windows),
    then delete the whole temp directory."""
    con = getattr(db, "_con", None)
    if con is not None:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(directory, ignore_errors=True)
    if directory.exists():
        rep.say(f"warning: could not remove the temporary directory {directory}; delete it by hand.")


async def _amain(a, settings, rep: Reporter, connectors_factory, db: AppDB) -> int:
    http = HttpClient()
    try:
        evidence = EvidenceStore(db)
        conns = connectors_factory(settings, http, evidence, a.grounding)
        return await run_checks(conns, rep)
    finally:
        try:
            await http.aclose()
        except Exception:  # noqa: BLE001
            pass


def main(
    argv: list[str] | None = None,
    *,
    settings=None,
    connectors_factory=build_connectors,
    tmp_root: pathlib.Path | str | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounding", action="store_true",
                    help="also make one Gemini grounding call (spends free-tier quota: about 20 requests/day)")
    a = ap.parse_args(argv)

    s = settings if settings is not None else get_settings()
    rep = Reporter([getattr(s, n, None) for n in ("tavily_api_key", "gemini_api_key", "sec_user_agent")])
    rep.say("Research connectors check")
    rep.say(f"  SEC_USER_AGENT   {'set' if getattr(s, 'sec_user_agent', None) else 'MISSING'}")
    rep.say(f"  TAVILY_API_KEY   {'set' if getattr(s, 'tavily_api_key', None) else 'MISSING'}")
    rep.say(f"  grounding        {'requested (spends Gemini quota)' if a.grounding else 'off (pass --grounding)'}")
    rep.say("-" * 68)

    directory = pathlib.Path(tempfile.mkdtemp(prefix="finagent-check-", dir=str(tmp_root) if tmp_root else None))
    db: AppDB | None = None
    try:
        db = AppDB(directory / "check.db")  # temporary: never data/store/app.db
        return asyncio.run(_amain(a, s, rep, connectors_factory, db))
    finally:
        dispose_temp_db(db, directory, rep)


if __name__ == "__main__":
    sys.exit(main())
