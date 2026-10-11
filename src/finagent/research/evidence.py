"""Per-session evidence store and untrusted-text handling for research connectors.

Every connector result (EDGAR filings, web search snippets, grounding output) becomes a
`SourceRecord` here before it reaches the model, numbered `E1`, `E2`, ... per session so the
chat layer and the model can both refer to "E3" unambiguously. External text is never trusted
per the project's global constraints: `sanitize_untrusted` strips it down to plain text and
`wrap_untrusted` marks the boundary so the model can tell it apart from the system/user turns. A
wrapped body can neither end its block nor open a new one, nor pose as a source: the three forms of
"<" (ASCII "<", fullwidth "＜" U+FF1C, small "﹤" U+FE64) are escaped, invisible
(default-ignorable) code points are removed, and an evidence label such as "[E2]" becomes
"(E2 cited in source text)".
`EvidenceStore` also doubles as a generic TTL cache (`http_cache`) for connector responses.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any

from ..store.appdb import AppDB

# --- untrusted text -----------------------------------------------------------
#
# strip_hidden_markup reduces untrusted HTML / markdown / plain text to the text a person would
# actually see on the rendered page, so text hidden from the human - a classic prompt-injection
# vector - never reaches the model. It is one linear pass: the stdlib html.parser tokenizer drives
# a small open-element stack that follows the parts of the HTML tree-construction rules deciding
# where a hidden element ends. The (lower-cased) element categories those rules need:

# Void elements have no content and no end tag. They are never pushed on the stack, so a hidden
# condition on one (`<br hidden>`, `<input type="text" hidden>`, `<img hidden>`) hides only the
# element itself, never the text after it. Includes obsolete elements browsers still parse as void.
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
    "source", "track", "wbr", "basefont", "bgsound", "frame", "keygen",
})
# Elements whose content browsers never render: suppressed exactly like a hidden element.
_NOT_RENDERED = frozenset({
    "script", "style", "template", "noscript", "iframe", "noembed", "noframes", "datalist",
})
# Raw-text elements (content is text, not markup, up to the matching end tag), extending the
# script/style pair html.parser already knows. noscript is raw text because browsers run scripts.
_RAW_TEXT_ELEMENTS = ("script", "style", "noscript", "iframe", "noembed", "noframes", "xmp")
# SVG / MathML roots: inside them (only) `<x/>` really is an empty element.
_FOREIGN_ROOTS = frozenset({"svg", "math"})
# The HTML "special" category: an end tag for a non-special element (e.g. </span>) is ignored
# while a special element (e.g. <div>) is open above it.
_SPECIAL = frozenset({
    "address", "applet", "area", "article", "aside", "base", "basefont", "bgsound", "blockquote",
    "body", "br", "button", "caption", "center", "col", "colgroup", "dd", "details", "dir", "div",
    "dl", "dt", "embed", "fieldset", "figcaption", "figure", "footer", "form", "frame",
    "frameset", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html",
    "iframe", "img", "input", "keygen", "li", "link", "listing", "main", "marquee", "menu",
    "meta", "nav", "noembed", "noframes", "noscript", "object", "ol", "p", "param", "plaintext",
    "pre", "script", "search", "section", "select", "source", "style", "summary", "table",
    "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "title", "tr", "track", "ul",
    "wbr", "xmp", "mi", "mo", "mn", "ms", "mtext", "annotation-xml", "foreignobject", "desc",
})
# "Has an element in scope" boundaries: an end tag for a special element (e.g. </div>) is
# ignored while one of these is open above it. </p> also stops at <button>, </li> at <ol>/<ul>.
_SCOPE_BOUNDARIES = frozenset({
    "applet", "caption", "html", "table", "td", "th", "marquee", "object", "template",
    "mi", "mo", "mn", "ms", "mtext", "annotation-xml", "foreignobject", "desc", "title",
})
# </table>, </tr>, </td>, ... use "table scope" instead (they also close any open rows/cells).
_TABLE_END_TAGS = frozenset({"table", "tbody", "thead", "tfoot", "tr", "td", "th", "caption"})
_TABLE_SCOPE_BOUNDARIES = frozenset({"html", "table", "template"})
# Formatting elements: when one is closed implicitly (by an ancestor's end tag), browsers re-open
# a copy of it - same attributes, so still hidden - around the text that follows, until its own
# end tag or until the enclosing table cell / caption / object / marquee / applet / template
# ("marker") closes.
_FORMATTING = frozenset({
    "a", "b", "big", "code", "em", "font", "i", "nobr", "s", "small", "strike", "strong", "tt",
    "u",
})
_MARKERS = frozenset({"applet", "caption", "marquee", "object", "td", "th", "template"})
# Implied end tags on a start tag, applied only to the element directly on top of the stack: a
# subset of what browsers close, so it can only ever end a hidden element later than a browser
# would (over-suppress), never earlier (leak). Start tags that close a <p> on top of the stack
# (not <table>: in quirks mode browsers keep the <p> open):
_CLOSES_P = frozenset({
    "address", "article", "aside", "blockquote", "center", "details", "dialog", "dir", "div",
    "dl", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
    "h6", "header", "hgroup", "listing", "main", "menu", "nav", "ol", "p", "pre", "search",
    "section", "summary", "ul", "li", "dd", "dt", "xmp",
})
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_CELLS = frozenset({"td", "th"})
_ROW_GROUPS = frozenset({"tbody", "thead", "tfoot"})
# tag -> groups of tags; for each group in order, pop the top of the stack if it is in the group.
_IMPLIED_CLOSES: dict[str, tuple[frozenset[str], ...]] = {
    "li": (frozenset({"li"}),),
    "dd": (frozenset({"dd", "dt"}),),
    "dt": (frozenset({"dd", "dt"}),),
    **{h: (_HEADINGS,) for h in _HEADINGS},
    "td": (_CELLS,),
    "th": (_CELLS,),
    "tr": (_CELLS, frozenset({"tr"})),
    **{g: (_CELLS, frozenset({"tr"}), _ROW_GROUPS) for g in _ROW_GROUPS},
    "option": (frozenset({"option"}),),
    "optgroup": (frozenset({"option"}), frozenset({"optgroup"})),
}
# Block boundaries become newlines (so paragraphs / list items / rows stay apart for
# collapse_whitespace and for callers that split on blank lines); table cells are separated by a
# space; inline elements add nothing, as when rendered ("Hel<b>lo</b>" -> "Hello").
_BLOCK_ELEMENTS = frozenset({
    "address", "article", "aside", "blockquote", "body", "caption", "center", "dd", "details",
    "dialog", "dir", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "html", "legend", "li",
    "listing", "main", "menu", "nav", "ol", "optgroup", "option", "p", "pre", "search",
    "section", "select", "summary", "table", "tbody", "textarea", "tfoot", "thead", "title",
    "tr", "ul", "xmp",
})
_TAG_START_RE = re.compile(r"</?[A-Za-z]")
_CSS_UNIT_CHARS = "abcdefghijklmnopqrstuvwxyz%"
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_WS_RE = re.compile(r"\s+")


def _strip_css_comments(css: str) -> str:
    """Replace /* ... */ comments with a space (an unterminated one runs to the end). Uses
    str.find in a loop rather than a lazy regex so it stays linear on a comment-bomb style value."""
    if "/*" not in css:
        return css
    parts: list[str] = []
    i = 0
    while True:
        start = css.find("/*", i)
        if start < 0:
            parts.append(css[i:])
            break
        parts.append(css[i:start])
        parts.append(" ")
        end = css.find("*/", start + 2)
        if end < 0:
            break
        i = end + 2
    return "".join(parts)


def _css_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _style_hides(style: str) -> bool:
    """True if an inline style hides the element: display:none, visibility:hidden|collapse,
    opacity <= 0 (0, 0.0, 0%, ...), or a zero font-size in any unit (0, 0px, 0.0em, 0%, ...).
    Split/partition only - no regex - so cost is linear in the attribute's length."""
    for declaration in _strip_css_comments(style).lower().split(";"):
        prop, colon, value = declaration.partition(":")
        if not colon:
            continue
        prop = prop.strip()
        value = value.split("!", 1)[0].strip()  # drop "!important"
        if prop == "display":
            if value == "none":
                return True
        elif prop == "visibility":
            if value in ("hidden", "collapse"):
                return True
        elif prop == "opacity":
            number = _css_number(value.rstrip("%"))
            if number is not None and number <= 0:  # opacity clamps: negative is fully transparent
                return True
        elif prop == "font-size":
            number = _css_number(value.rstrip(_CSS_UNIT_CHARS))
            if number == 0:  # a negative font-size is invalid CSS and is ignored
                return True
    return False


def _attrs_hide(attrs: list[tuple[str, str | None]]) -> bool:
    """Hidden conditions on an element's attributes (names arrive lower-cased, values with
    character references already decoded): the boolean `hidden` attribute (any value, e.g.
    "until-found"), aria-hidden="true", or a hiding inline style. `class="hidden"` and friends are
    left alone - stylesheets are not evaluated."""
    for name, value in attrs:
        if name == "hidden":
            return True
        if value is None:
            continue
        if name == "aria-hidden":
            if value.strip().lower() == "true":
                return True
        elif name == "style":
            if _style_hides(value):
                return True
    return False


def _boundary(tag: str) -> str:
    if tag in _BLOCK_ELEMENTS:
        return "\n"
    if tag in _CELLS:
        return " "
    return ""


class _VisibleTextParser(HTMLParser):
    """Collects the text a browser would render, dropping hidden elements, comments and
    never-rendered content.

    Work per token is O(1) amortized: every element is pushed and popped at most once, and each
    "does the browser honor this end tag" check is an index comparison against values cached on
    the stack entry, never a walk of the stack. Text is emitted only while no hidden element is
    open (and no implicitly-closed hidden formatting element is waiting to be re-opened).
    """

    CDATA_CONTENT_ELEMENTS = _RAW_TEXT_ELEMENTS

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._saw_markup = False
        # Open-element stack. Entry: (tag, hidden, then - for the nearest special / scope-boundary
        # / button-scope-boundary / list-scope-boundary / table-scope-boundary element at or below
        # this entry - its stack index, or -1).
        self._stack: list[tuple[str, bool, int, int, int, int, int]] = []
        self._open_at: dict[str, list[int]] = {}  # tag -> stack indices of its open elements
        self._hidden_open = 0  # hidden or never-rendered entries currently on the stack
        self._foreign_open = 0  # open <svg>/<math> elements
        # One frame per open marker element (plus one for the document): tag -> number of hidden
        # formatting elements closed implicitly inside it, which the browser re-opens (hidden)
        # around the text that follows.
        self._reopen: list[dict[str, int]] = [{}]
        self._reopen_total: list[int] = [0]

    # -- output --------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        if text and not self._hidden_open and not self._reopen_total[-1]:
            self._out.append(text)

    def partial_text(self) -> str:
        return "".join(self._out)

    def finish(self) -> str:
        """Settle whatever the tokenizer could not complete before the end of the input (it holds
        an unfinished construct back in `rawdata`, waiting for more data), then return the text.

        Deliberately not `close()`: its end-of-input handling varies across CPython releases.
        Everything here is dropped while a hidden element is still open - including when the
        input was cut off mid-element, e.g. by an upstream size bound (that is how a hidden block
        straddling the cut stays hidden). Otherwise:
        - text held back for a possibly-split character reference is emitted, decoded;
        - an unterminated comment / `<!...>` declaration / `<?...>` instruction is dropped, as
          browsers treat it as a comment running to the end;
        - an unterminated tag (`<x ...` with no `>` after it) is dropped, as browsers do, if the
          input has contained any markup; in markup-free text (plain text / markdown) with no
          `>` anywhere after it, it cannot be a tag, so it is kept as the prose it is
          ("... where P<E ratios ...").
        """
        rest = self.rawdata
        if rest:
            if self.cdata_elem:
                self._emit(rest)  # unterminated raw text of a rendered element, e.g. <xmp>
            elif rest[0] != "<":
                self._emit(unescape(rest))
            elif len(rest) == 1:
                self._emit(rest)
            elif _TAG_START_RE.match(rest) and not self._saw_markup and ">" not in rest:
                self._emit(unescape(rest))
        return "".join(self._out)

    # -- stack ---------------------------------------------------------------------

    def _push(self, tag: str, hidden: bool) -> None:
        stack = self._stack
        i = len(stack)
        if stack:
            _, _, special, scope, button, listing, table = stack[-1]
        else:
            special = scope = button = listing = table = -1
        is_boundary = tag in _SCOPE_BOUNDARIES
        stack.append((
            tag,
            hidden,
            i if tag in _SPECIAL else special,
            i if is_boundary else scope,
            i if is_boundary or tag == "button" else button,
            i if is_boundary or tag in ("ol", "ul") else listing,
            i if tag in _TABLE_SCOPE_BOUNDARIES else table,
        ))
        self._open_at.setdefault(tag, []).append(i)
        if hidden:
            self._hidden_open += 1
        if tag in _MARKERS:
            self._reopen.append({})
            self._reopen_total.append(0)
        if tag in _FOREIGN_ROOTS:
            self._foreign_open += 1

    def _pop_through(self, index: int) -> None:
        """Pop the stack down to and including the entry at `index`."""
        stack = self._stack
        while len(stack) > index:
            tag, hidden = stack.pop()[:2]
            self._open_at[tag].pop()
            if tag in _MARKERS:
                self._reopen.pop()
                self._reopen_total.pop()
            if tag in _FOREIGN_ROOTS:
                self._foreign_open -= 1
            if hidden:
                self._hidden_open -= 1
                if tag in _FORMATTING and len(stack) > index:
                    # Closed implicitly (not by its own end tag): the browser re-opens it.
                    frame = self._reopen[-1]
                    frame[tag] = frame.get(tag, 0) + 1
                    self._reopen_total[-1] += 1

    def _close_implied(self, tag: str) -> None:
        stack = self._stack
        if tag in _CLOSES_P and stack and stack[-1][0] == "p":
            self._pop_through(len(stack) - 1)
        for group in _IMPLIED_CLOSES.get(tag, ()):
            if stack and stack[-1][0] in group:
                self._pop_through(len(stack) - 1)

    def _end_tag_honored(self, tag: str, index: int) -> bool:
        """Whether a browser closes the open `tag` element at stack `index` for `</tag>`."""
        top = self._stack[-1]
        if tag == "template":
            return True
        if tag == "form":  # browsers take <form> off the stack without closing what is inside it
            return index == len(self._stack) - 1
        if tag in _TABLE_END_TAGS:
            return top[6] <= index
        if tag in _SPECIAL:
            boundary = top[4] if tag == "p" else top[5] if tag == "li" else top[3]
            return boundary <= index
        return top[2] <= index  # no special element open above it

    # -- tokenizer callbacks -------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._saw_markup = True
        if tag in _VOID_ELEMENTS:
            if tag in ("br", "hr"):
                self._emit("\n")
            return
        self._close_implied(tag)
        hidden = (
            tag in _NOT_RENDERED
            or _attrs_hide(attrs)
            or (tag == "dialog" and all(name != "open" for name, _ in attrs))
        )
        self._emit(_boundary(tag))
        self._push(tag, hidden)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_ELEMENTS:
            self.handle_starttag(tag, attrs)
        elif self._foreign_open or tag in _FOREIGN_ROOTS:
            self._saw_markup = True  # a self-closed SVG/MathML element really is empty
        else:
            # On any other HTML element browsers ignore the "/": `<div hidden/>text</div>` hides
            # "text" exactly like `<div hidden>text</div>`, so treat it as a plain start tag.
            self.handle_starttag(tag, attrs)
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.set_cdata_mode(tag)

    def handle_endtag(self, tag: str) -> None:
        self._saw_markup = True
        if tag in ("body", "html") or tag in _VOID_ELEMENTS:
            return  # browsers never pop elements for these
        open_at = self._open_at.get(tag)
        if not open_at:
            frame = self._reopen[-1]
            if frame.get(tag):  # closes a re-opened hidden formatting element
                frame[tag] -= 1
                self._reopen_total[-1] -= 1
            return  # stray end tag: ignored
        index = open_at[-1]
        if not self._end_tag_honored(tag, index):
            return
        self._pop_through(index)
        self._emit(_boundary(tag))

    def handle_data(self, data: str) -> None:
        self._emit(data)

    def handle_comment(self, data: str) -> None:
        self._saw_markup = True

    def handle_decl(self, decl: str) -> None:
        self._saw_markup = True

    def handle_pi(self, data: str) -> None:
        self._saw_markup = True

    def unknown_decl(self, data: str) -> None:
        self._saw_markup = True


def strip_hidden_markup(text: str) -> str:
    """Reduce untrusted text to what a person would see rendered, without collapsing whitespace
    or truncating.

    Drops hidden elements with everything inside them (inline style display:none,
    visibility:hidden, opacity:0, font-size:0; the `hidden` attribute; aria-hidden="true"),
    content browsers never render (script, style, template, noscript, iframe, a closed dialog,
    ...), comments, all tags, and zero-width characters; decodes character references. Block
    boundaries become newlines. Plain text and markdown pass through unchanged: a "<" that does
    not start a tag ("a < b", "x<5") is text.

    One linear pass (stdlib html.parser plus an O(1)-per-token open-element stack; see
    `_VisibleTextParser`), so cost stays proportional to input size on adversarial pages. How a
    hidden element ends follows browser tree construction closely enough to neither leak nor
    over-delete on real pages:
    - void elements (`<br hidden>`, `<input type="text" hidden>`, `<img hidden>`) hide nothing
      after them; `<x/>` on a non-void HTML element is an ordinary start tag, as in browsers;
    - nesting and mis-nesting are resolved like the browser does (an end tag closes the matching
      open element and everything above it, unless the browser would ignore that end tag);
    - an element still open at the end of `text` hides everything to the end. So a hidden
      element cut off by an upstream size bound (e.g. websearch.RAW_PAGE_MAX_CHARS) never leaks
      its partial payload, and it is safe to cut the *result* anywhere afterwards.

    Markup is recognized even inside markdown code samples: a literal `<!-- ... -->` there is
    dropped as a comment, and an unterminated `<!--` drops everything after it, exactly as a
    browser would hide it. This errs toward suppression: a code sample is not worth a hole.

    Split out of `sanitize_untrusted` so a caller that selects or truncates a bounded excerpt
    (by paragraph, or to a char budget) can strip hidden content *first*, then cut; cutting
    first could slice a hidden element in two and expose its payload.
    """
    s = text or ""
    if "<" in s:
        parser = _VisibleTextParser()
        try:
            parser.feed(s)
        except AssertionError:
            # Older CPython tokenizers raise on some malformed declarations: fail closed, keeping
            # only the text already extracted before the malformed construct.
            s = parser.partial_text()
        else:
            s = parser.finish()
    elif "&" in s:
        s = unescape(s)
    return _ZERO_WIDTH_RE.sub("", s)


def collapse_whitespace(text: str, max_chars: int = 4000) -> str:
    """Collapse whitespace runs and cap length.

    Companion to `strip_hidden_markup`: `sanitize_untrusted` is just `collapse_whitespace(
    strip_hidden_markup(text), max_chars)`. Exposed separately so a caller that already ran
    `strip_hidden_markup` on the full text (e.g. before selecting/cutting an excerpt of it) can
    finish the job on that excerpt without re-running the stripping pass over text that's
    already clean of markup.
    """
    s = _WS_RE.sub(" ", text or "").strip()
    return s[:max_chars].rstrip()


def sanitize_untrusted(text: str, max_chars: int = 4000) -> str:
    """Strip markup, hidden/zero-width text and comments out of externally-sourced text."""
    return collapse_whitespace(strip_hidden_markup(text), max_chars)


# Unicode default-ignorable code points (DerivedCoreProperties.txt, Default_Ignorable_Code_Point)
# and the C0/C1 controls other than tab and newline. All render as nothing, so inside a word the
# model reads whole they hide what it is ("</untr<U+00AD>usted>"), and the bidi controls reorder
# what a person sees. None belongs in text the model reads as data.
_INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f"
    "\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]"
)
# The three forms of "<" that are escaped, and only these three: the ASCII sign, the fullwidth "＜"
# (U+FF1C) and the small "﹤" (U+FE64). Sanitized text carries no markup, so in a wrapped body each
# one becomes U+2039 and no tag (a closer for this block, an opener for a forged one) can be
# spelled there.
_LT_LOOKALIKES_RE = re.compile("[<\uff1c\ufe64]")
_LT_ESCAPE = "\u2039"
# An evidence label like the tools' own "[E12]", also spaced, lower-case or fullwidth ("［Ｅ１２］").
_EVIDENCE_LABEL_RE = re.compile("[\\[\uff3b]\\s*[Ee\uff25\uff45]\\s*([0-9\uff10-\uff19]{1,9})\\s*[\\]\uff3d]")
_FULLWIDTH_DIGITS = str.maketrans("\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19", "0123456789")


def _neutralize_label(match: re.Match) -> str:
    return f"(E{match.group(1).translate(_FULLWIDTH_DIGITS)} cited in source text)"


def neutralize_untrusted_body(text: str) -> str:
    """What `wrap_untrusted` puts inside a block: invisible code points removed, then evidence
    labels neutralized (a page cannot pose as the tools' own "[E2]" source), then each of the
    three forms of "<" (U+003C, U+FF1C, U+FE64) escaped to U+2039 (a page cannot close its block
    or open a forged one)."""
    body = _INVISIBLE_RE.sub("", text or "")
    body = _EVIDENCE_LABEL_RE.sub(_neutralize_label, body)
    return _LT_LOOKALIKES_RE.sub(_LT_ESCAPE, body)


def wrap_untrusted(source_id: str, text: str) -> str:
    """Wrap sanitized external text as `<untrusted source="...">...</untrusted>`; the body goes
    through `neutralize_untrusted_body`, so the block's own tags are the only ones in it."""
    return f'<untrusted source="{source_id}">{neutralize_untrusted_body(text)}</untrusted>'


# --- records --------------------------------------------------------------------


@dataclass
class SourceRecord:
    id: str
    session_id: str
    connector: str
    title: str
    url: str | None
    publisher: str | None
    tier: int
    published_at: str | None
    retrieved_at: str
    snippet: str
    content_sha256: str
    untrusted: bool


@dataclass
class ResearchResult:
    text: str
    sources: list[SourceRecord]
    is_error: bool = False


def _row_to_source(row: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        id=row["id"],
        session_id=row["session_id"],
        connector=row["connector"],
        title=row["title"],
        url=row["url"],
        publisher=row["publisher"],
        tier=row["tier"],
        published_at=row["published_at"],
        retrieved_at=row["retrieved_at"],
        snippet=row["snippet"],
        content_sha256=row["content_sha256"],
        untrusted=bool(row["untrusted"]),
    )


class EvidenceStore:
    """Per-session evidence (backed by `AppDB.evidence`) plus a TTL cache (`AppDB.http_cache`)."""

    def __init__(self, db: AppDB, now: Callable[[], datetime] | None = None):
        self._db = db
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()  # serializes the count-then-insert numbering in add()

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def add(
        self,
        session_id: str,
        *,
        connector: str,
        title: str,
        url: str | None,
        publisher: str | None,
        tier: int,
        published_at: str | None,
        snippet: str,
        untrusted: bool,
    ) -> SourceRecord:
        content_sha256 = hashlib.sha256((snippet or "").encode("utf-8")).hexdigest()
        dedupe_url = url or ""
        with self._lock:
            existing = self._db.query(
                "SELECT * FROM evidence WHERE session_id=? AND COALESCE(url,'')=? AND content_sha256=?",
                (session_id, dedupe_url, content_sha256),
            )
            if existing:
                return _row_to_source(existing[0])
            count = self._db.query(
                "SELECT COUNT(*) AS n FROM evidence WHERE session_id=?", (session_id,)
            )[0]["n"]
            source_id = f"E{count + 1}"
            retrieved_at = self._timestamp()
            self._db.execute(
                "INSERT INTO evidence (session_id, id, connector, title, url, publisher, tier, "
                "published_at, retrieved_at, snippet, content_sha256, untrusted) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    source_id,
                    connector,
                    title,
                    url,
                    publisher,
                    tier,
                    published_at,
                    retrieved_at,
                    snippet,
                    content_sha256,
                    int(untrusted),
                ),
            )
        return SourceRecord(
            id=source_id,
            session_id=session_id,
            connector=connector,
            title=title,
            url=url,
            publisher=publisher,
            tier=tier,
            published_at=published_at,
            retrieved_at=retrieved_at,
            snippet=snippet,
            content_sha256=content_sha256,
            untrusted=untrusted,
        )

    def get(self, session_id: str, source_id: str) -> SourceRecord | None:
        rows = self._db.query(
            "SELECT * FROM evidence WHERE session_id=? AND id=?", (session_id, source_id)
        )
        return _row_to_source(rows[0]) if rows else None

    def list(self, session_id: str) -> list[SourceRecord]:
        rows = self._db.query(
            "SELECT * FROM evidence WHERE session_id=? ORDER BY rowid", (session_id,)
        )
        return [_row_to_source(r) for r in rows]

    def cache_get(self, key: str, ttl_sec: float) -> Any | None:
        rows = self._db.query("SELECT payload, stored_at FROM http_cache WHERE key=?", (key,))
        if not rows:
            return None
        stored_at = datetime.fromisoformat(rows[0]["stored_at"])
        age = (self._now() - stored_at).total_seconds()
        if age > ttl_sec:
            return None
        return json.loads(rows[0]["payload"])

    def cache_put(self, key: str, payload: Any) -> None:
        self._db.execute(
            "INSERT INTO http_cache (key, payload, stored_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, stored_at=excluded.stored_at",
            (key, json.dumps(payload), self._timestamp()),
        )
