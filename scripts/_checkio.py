"""Shared console output for the live check scripts (`check_alpaca.py`, `check_research.py`).

ONE place decides what may reach the terminal. Everything either script prints goes through
`Reporter.say`, which (in this order)

1. strips control characters: newlines/tabs become a space, and ESC, BEL, the rest of C0, DEL and the C1
   range (0x80-0x9f, which includes the 8-bit CSI/OSC introducers) are removed, as are Unicode bidi
   overrides. External text (web pages, news, model answers) can carry terminal escape sequences that
   `sanitize_untrusted` does not touch; without this a fetched page could clear the screen or retitle
   the window;
2. replaces every known secret value with ``***``.

Stripping comes first so a secret with a control character slipped into it is still recognised.
`Reporter.describe` does the same two steps on an exception BEFORE truncating, so a secret that straddles
the length limit cannot leave a readable fragment behind.
"""
from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

_LINE_BREAKS_RE = re.compile(r"[\n\r\t\x0b\x0c\x85]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# Unicode bidi overrides/isolates (U+202A-202E, U+2066-2069) can reorder what a terminal shows
_BIDI_RE = re.compile("[" + "".join(chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))) + "]")

# Read at call time so tests can swap the clock.
perf = time.perf_counter


def strip_controls(text: object) -> str:
    """`text` as one terminal-safe line: line breaks and tabs -> a space, other control characters gone."""
    s = _LINE_BREAKS_RE.sub(" ", str(text))
    s = _CONTROL_RE.sub("", s)
    return _BIDI_RE.sub("", s)


@dataclass
class Outcome:
    ok: bool
    summary: str
    lines: list[str] = field(default_factory=list)


class Reporter:
    """Numbered steps with a latency and a check or cross; counts passed, failed and skipped."""

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

    def line(self, text: object = "") -> str:
        """The terminal-safe, secret-free form of `text` (what `say` prints)."""
        return self.scrub(strip_controls(text))

    def say(self, text: object = "") -> None:
        print(self.line(text))

    def cut(self, text: object, limit: int) -> str:
        """`text` as one safe line, cut to `limit`: controls stripped and secrets blanked first, THEN cut."""
        return " ".join(self.line(text).split())[:limit]

    def describe(self, exc: BaseException, limit: int = 220) -> str:
        """`ExcType: message` for an exception, safe to print."""
        return self.cut(f"{type(exc).__name__}: {exc}", limit)

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
        self.say(f"    ✗ {seconds:.2f}s  {self.cut(reason, 300)}")
        if hint:
            self.say(f"      -> {hint}")

    def skip(self, label: str, reason: str) -> None:
        self.step += 1
        self.skipped += 1
        self.say(f"[{self.step}] {label}")
        self.say(f"    - SKIPPED: {reason}")


async def run_step(
    rep: Reporter,
    label: str,
    fn: Callable[[], Awaitable[Outcome]],
    hint_for: Callable[[BaseException], str | None] = lambda exc: None,
) -> bool:
    """Run one check: its latency, a check or cross with a short reason, never raises for an `Exception`
    (one failing step must not stop the others)."""
    rep.begin(label)
    t0 = perf()
    try:
        outcome = await fn()
    except Exception as exc:  # noqa: BLE001
        rep.fail(perf() - t0, rep.describe(exc), hint_for(exc))
        return False
    seconds = perf() - t0
    for line in outcome.lines:
        rep.detail(line)
    if outcome.ok:
        rep.ok(seconds, outcome.summary)
    else:
        rep.fail(seconds, outcome.summary)
    return outcome.ok
