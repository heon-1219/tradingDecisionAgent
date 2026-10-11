"""Evidence / trace formatting and answer-policy helpers (pure functions)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..i18n import pick

# The sentence a cannot-confirm answer starts with, in each answer language.
CANNOT_PREFIX_EN = "This cannot be confirmed with the provided data."
CANNOT_PREFIX_KO = "제공된 데이터로는 확인할 수 없습니다."


def cannot_prefix(lang: str = "en") -> str:
    return pick(lang, CANNOT_PREFIX_EN, CANNOT_PREFIX_KO)


FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "cannot_confirm", "clarify"],
            "description": "answered = answered from the supporting data, cannot_confirm = cannot be confirmed with the provided data, clarify = the condition is ambiguous, so ask back",
        },
        "answer": {"type": "string", "description": "The final answer shown to the user, in the answer language stated in the user message. Include the supporting numbers, product codes and base date"},
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The key evidence used in the answer (table names, product codes/names, document titles)",
        },
    },
    "required": ["status", "answer"],
}


@dataclass
class Evidence:
    kind: str  # sql | search | graph | docs | entity | schema
    table: str | None
    korean_name: str | None
    base_date: str | None
    detail: str
    row_count: int | None = None
    refs: list[str] = field(default_factory=list)


def format_evidence(evs: list[Evidence], extra_refs: list[str] | None = None) -> str:
    if not evs and not extra_refs:
        return "No data referenced"
    blocks = []
    for i, e in enumerate(evs, 1):
        head = f"[Evidence {i}] "
        if e.table:
            head += f"{e.korean_name or ''}({e.table})".strip()
        else:
            head += {"docs": "External documents", "graph": "Knowledge graph", "search": "Product search",
                     "entity": "Entity resolution", "schema": "Schema"}.get(e.kind, e.kind)
        if e.base_date:
            head += f" · base date {e.base_date}"
        if e.row_count is not None:
            head += f" · {e.row_count} rows"
        lines = [head, f"  {e.kind}: {e.detail}"]
        if e.refs:
            uniq = list(dict.fromkeys(r for r in e.refs if r))
            shown = uniq[:20]
            lines.append("  Refs: " + " | ".join(shown) + (f" | ... and {len(uniq) - 20} more" if len(uniq) > 20 else ""))
        blocks.append("\n".join(lines))
    if extra_refs:
        blocks.append("[Answer evidence summary] " + " | ".join(extra_refs))
    return "\n".join(blocks)


def format_trace(steps: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else "-"


def fallback_answer(reason: str, lang: str = "en") -> str:
    return pick(
        lang,
        f"{CANNOT_PREFIX_EN} A system problem prevented an answer from being generated ({reason}). "
        "Please rephrase the question more specifically and try again.",
        f"{CANNOT_PREFIX_KO} 시스템 처리 중 문제가 발생하여 답변을 생성하지 못했습니다({reason}). "
        "질문을 조금 더 구체적으로 바꾸어 다시 시도해 주세요.",
    )


# --------------------------------------------------------- tool-call syntax leaking into the answer
# The model really did write out the final_answer call as text instead of *calling* it.  6 of the 54
# recorded answers did, and the user saw it as is:
#     final_answer(status="completed", answer="… 총 22개입니다.")
#     제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", answer="…")
#     final_answer 도구를 사용하여 답변을 제출하겠습니다:\n```plaintext\nfinal_answer(…)\n```
#
# Deleting it would lose the whole answer — the real answer text sits inside answer="…".  So only the
# shell is peeled off and the body inside is lifted out (unwrap).  With a lower output cap the call can
# be cut off before it closes, so unfinished fragments with no closing paren or quote are handled the
# same way.
_LEAK_CALL = re.compile(r"final_answer\s*\(", re.I)
_LEAK_ANSWER_KW = re.compile(r"\banswer\s*=\s*(\"|')", re.I)
_LEAK_STATUS_KW = re.compile(r"""^\W*\bstatus\s*=\s*["']?\w*["']?\s*,?\s*""", re.I)
_LEAK_MARKERS = ('final_answer', 'status="', "status='", 'answer="', "answer='")
_ESCAPES = {"n": "\n", "t": "\t", "r": "", "\\": "\\", '"': '"', "'": "'"}


def _unescape(s: str) -> str:
    """Source-level escapes such as \\n and \\" become real characters.  Unknown escapes are left as written."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append(_ESCAPES[nxt] if nxt in _ESCAPES else s[i:i + 2])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _call_span(text: str, open_paren: int) -> tuple[int, str]:
    """Find the closing paren matching the opening one (parens inside quotes do not count).

    Returns (end of the call [exclusive], argument string).  If it never closes, the arguments run to the end.
    """
    i, depth, quote = open_paren + 1, 1, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1, text[open_paren + 1:i]
        i += 1
    return len(text), text[open_paren + 1:]


def _quoted_value(src: str, at: int, quote: str) -> str:
    """The body of a string literal from position at up to the closing quote (or the end)."""
    buf, i = [], at
    while i < len(src):
        if src[i] == "\\" and i + 1 < len(src):
            buf.append(src[i:i + 2])
            i += 2
            continue
        if src[i] == quote:
            break
        buf.append(src[i])
        i += 1
    return _unescape("".join(buf))


def strip_tool_call_syntax(text: str) -> str:
    """Peel a final_answer call that leaked into the answer text and keep only the body inside."""
    if not text or not any(m in text for m in _LEAK_MARKERS):
        return text
    out = text
    for _ in range(4):  # nested and repeated leaks happen, but never an infinite loop
        m = _LEAK_CALL.search(out)
        if not m:
            break
        end, args = _call_span(out, m.end() - 1)
        am = _LEAK_ANSWER_KW.search(args)
        inner = _quoted_value(args, am.end(), am.group(1)) if am else ""
        out = out[:m.start()] + inner + out[end:]
    # Only argument fragments left, without the paren (truncated output): drop status=… and unwrap answer="…".
    if any(m in out for m in _LEAK_MARKERS[1:]):
        stripped = _LEAK_STATUS_KW.sub("", out.lstrip(), count=1)
        am = _LEAK_ANSWER_KW.search(stripped)
        if am and stripped[:am.start()].strip(" \t\r\n,\"'") == "":
            out = _quoted_value(stripped, am.end(), am.group(1))
    # What shell remains — lines that merely mention the tool name, and the code fence around the call.
    lines = [ln for ln in out.splitlines() if "final_answer" not in ln]
    if any(ln.strip() for ln in lines):
        out = "\n".join(lines)
    body = out.strip()
    if body.startswith("```") and body.endswith("```"):
        inner_lines = body.splitlines()[1:-1]
        if inner_lines:
            out = "\n".join(inner_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    for prefix in (CANNOT_PREFIX_KO, CANNOT_PREFIX_EN):
        dup = f"{prefix} {prefix}"
        while dup in out:
            out = out.replace(dup, prefix)
    return out


# ------------------------------------------------------- closing lines that carry no information
# 8 of the 54 recorded answers ended with a sentence like "추가적인 정보가 필요하시면 …" ("if you need
# more information …").  That sentence adds nothing to the answer and costs generation time (measured:
# a fixed 2.7 seconds + 63 tokens/second).
#
# What must not be deleted matters more.  A sentence is kept if any of these hits:
#   * a sentence with a number or % — numbers are the body of the answer.
#   * a caveat such as "투자 판단은 본인 책임" (investment decisions are your own responsibility) —
#     deleting it deletes the disclosure (trimming that wording is prompt rule 4's job; a sentence
#     already produced is not cut here).
#   * a sentence stating a cannot-confirm or exclusion fact, such as "…없습니다/않습니다/제외".
# What remains is a pure "ask if you need more", and only then is it deleted.  At most the last 2
# sentences are examined.  The English patterns apply the same three tests to English sentences.
_OFFER_TRIGGER = re.compile(r"추가(?:적인)?\s*(?:정보|문의|사항|질문)|더\s*궁금|도움이\s*필요")
_OFFER_CLOSE = re.compile(r"문의|말씀|연락|알려\s*주|권장|바랍니다|주세요|주십시오|드립니다")
_OFFER_KEEP = re.compile(r"[0-9%]|책임|제외|없습니다|않습니다|아닙니다|못했습니다")
_OFFER_TRIGGER_EN = re.compile(
    r"\b(?:let\s+me\s+know|feel\s+free|(?:if|should)\s+you\s+(?:have|need|want|would\s+like)"
    r"|(?:any\s+)?(?:further|more|other|additional)\s+(?:questions?|information|info|details|help|assistance)"
    r"|anything\s+else|happy\s+to\s+help|glad\s+to\s+help|do\s+not\s+hesitate|don't\s+hesitate)\b",
    re.I,
)
_OFFER_CLOSE_EN = re.compile(
    r"\b(?:let\s+me\s+know|feel\s+free|ask|contact|reach\s+out|hesitate|happy\s+to|glad\s+to"
    r"|recommend(?:ed)?|please|help|assist)\b",
    re.I,
)
_OFFER_KEEP_EN = re.compile(
    r"\b(?:responsib\w*|exclud\w*|cannot|(?:is|are|was|were|does|did|could|can|has|have)\s*n(?:o|')t"
    r"|not\s+(?:available|provided|found|included|listed|recorded|confirmed))\b"
    r"|\bno\s+(?:data|value|such|matching|records?)\b",
    re.I,
)


def _is_closing_offer(sent: str) -> bool:
    ko = _OFFER_TRIGGER.search(sent) and _OFFER_CLOSE.search(sent)
    en = _OFFER_TRIGGER_EN.search(sent) and _OFFER_CLOSE_EN.search(sent)
    return bool(ko or en) and not (_OFFER_KEEP.search(sent) or _OFFER_KEEP_EN.search(sent))


def _last_sentence(text: str) -> tuple[int, str]:
    """(start of the sentence, the sentence).  The nearest . ! ? or newline from the end is the boundary."""
    body = text.rstrip()
    i = len(body) - 1
    while i >= 0 and body[i] in ".!?\"')]”’":
        i -= 1
    j = i
    while j >= 0 and body[j] not in ".!?\n":
        j -= 1
    return j + 1, body[j + 1:]


def strip_closing_offer(text: str) -> str:
    """Delete a trailing 'feel free to ask' sentence.  Sentences mixed with facts or caveats are not touched."""
    for _ in range(2):
        if not text.strip():
            break
        start, sent = _last_sentence(text)
        if not _is_closing_offer(sent):
            break
        text = text[:start].rstrip()
    return text


# ----------------------------------------------------------------- markdown → prose
# The answer in the organisers' sample response is one line of plain text ("총보수 0.03%인 A …").  Our
# answers came out with bold and numbered lists mixed in (13 of the 54 recorded).  The rule is in the
# prompt, but it is not trusted alone — in this project, prompt rules given in advance have a record of
# being ignored.
#
# Only the **syntax** is stripped; the content is not touched.  Numbers like 0.03% and 1,436.24, column
# names like du_er_1y and an asterisk mid-sentence are not markdown and stay.  Lists are not deleted: the
# markers are removed and the items joined with commas into one sentence (losing an item loses answer).
_MD_FENCE = re.compile(r"^\s*(?:```|~~~)")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+")
_MD_RULE = re.compile(r"^[\s\-*_=|:]{3,}$")            # ---, ***, |---|:--:| (table separator)
_MD_ITEM = re.compile(r"^\s*(?:[-*•·]|\(?\d{1,3}[.)])\s+")
_MD_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_MD_BOLD = re.compile(r"\*\*(\S(?:[^*\n]*\S)?)\*\*")
_MD_BOLD_U = re.compile(r"__(\S(?:[^_\n]*\S)?)__")
_MD_ITALIC = re.compile(r"(?<![\w*])\*(\S(?:[^*\n]*\S)?)\*(?![\w*])")
_MD_CODE = re.compile(r"`([^`\n]+)`")
_SENT_END = ".!?。？！"
_CLOSERS = ")]}”’\"'』」"


def _ends_sentence(s: str) -> bool:
    """Has the sentence ended?  Closing parens and quotes are skipped first ("…(11조)" has not ended)."""
    t = s.rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1]
    return bool(t) and t[-1] in _SENT_END


def _item_body(line: str) -> str:
    """A list item's body without the marker.  The line as is if it is not an item."""
    m = _MD_ITEM.match(line)
    return line[m.end():].strip() if m else line.strip()


def _joined_item(s: str) -> str:
    """One item to join with commas.  Only the final period is dropped, so it never reads 'A입니다., B'."""
    s = s.strip()
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1].rstrip()
    return s


def strip_markdown(text: str) -> str:
    """Strip markdown syntax and turn line-by-line lists into comma-joined prose."""
    if not text or not text.strip():
        return text
    kinds: list[tuple[str, str]] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if _MD_FENCE.match(line):
            continue                                   # drop only the fence itself; the body inside stays
        if not line.strip():
            kinds.append(("blank", ""))
            continue
        if _MD_RULE.match(line.strip()):
            continue                                   # horizontal rules and table separators carry nothing
        if _MD_TABLE_ROW.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            kinds.append(("item", " ".join(c for c in cells if c)))
            continue
        if _MD_ITEM.match(line):
            kinds.append(("item", _item_body(line)))
            continue
        kinds.append(("text", _MD_HEADER.sub("", line).strip()))
    # A blank line between items belongs to the list (the model often spaces items apart).
    cleaned: list[tuple[str, str]] = []
    for i, (k, c) in enumerate(kinds):
        if k == "blank":
            prev = next((kinds[j][0] for j in range(i - 1, -1, -1) if kinds[j][0] != "blank"), None)
            nxt = next((kinds[j][0] for j in range(i + 1, len(kinds)) if kinds[j][0] != "blank"), None)
            if prev == "item" and nxt == "item":
                continue
        cleaned.append((k, c))

    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        body = ", ".join(run)
        if body and not _ends_sentence(body):
            body += "."
        while out and not out[-1].strip():
            out.pop()                                  # absorb the blank line right before the list
        if out:
            prev = out[-1].rstrip()
            if prev.endswith((":", "：")):              # "…다음과 같습니다:" / "…as follows:" → ends with "."
                prev = prev[:-1].rstrip()
                if prev and not _ends_sentence(prev):
                    prev += "."
            out[-1] = f"{prev} {body}".strip()
        else:
            out.append(body)
        run.clear()

    for k, c in cleaned:
        if k == "item":
            if c:
                run.append(_joined_item(c))
            continue
        flush()
        out.append(c)
    flush()

    res = "\n".join(out)
    res = _MD_BOLD.sub(r"\1", res)
    res = _MD_BOLD_U.sub(r"\1", res)
    res = _MD_ITALIC.sub(r"\1", res)
    res = _MD_CODE.sub(r"\1", res)
    res = res.replace("**", "")                        # unpaired emphasis (truncated output)
    return re.sub(r"\n{3,}", "\n\n", res).strip()


# ------------------------------------------------- the 'substitute product list' after a refusal
# Real incident: for "한화 에어로스페이스의 **자회사를 편입한** ETF 중 …" ("ETFs holding Hanwha Aerospace's
# subsidiaries") the model said "찾을 수 없습니다" (cannot be found), then went on "그러나 한화
# 에어로스페이스를 편입한 ETF 목록을 제공할 수 있습니다" ("but I can list ETFs holding Hanwha Aerospace")
# and listed 40.  A substitute answer that switched to an entity nobody asked about (subsidiary → the
# company itself).  Of the 54 recorded, G-020 has the same shape (public fund → domestic ETFs).
#
# The difference from what must not be deleted is the whole of this function:
#   keep   — "…유효한 신용등급은 AAA, AA+, AA0, … 입니다."  the condition's valid values, inline, in one sentence.
#   delete — "…목록은 다음과 같습니다:\n1. 아이엠에셋 200 (KR70007N0000)\n2. …"  products, one per line.
# So it fires only when (1) at least 3 line-broken items run consecutively and (2) those items look like
# products — a code or ticker, a number with a unit, a product-family word.  An inline enumeration is never
# touched, however many items it lists.  It runs **before** markdown stripping, while the line structure
# is still there.  The English patterns mirror the Korean ones for English answers.
_ALT_CODE = re.compile(r"\b(?:KR[0-9A-Z]{9,11}|[0-9]{6})\b")
_ALT_NUM_UNIT = re.compile(r"\d[\d,.]*\s*(?:조|억|만원|원|%|퍼센트|개|건|USD|달러)")
_ALT_NUM_UNIT_EN = re.compile(r"\d[\d,.]*\s*(?:trillion|billion|million|won|KRW)\b", re.I)
_ALT_PRODUCT = re.compile(
    r"KODEX|TIGER|RISE|ACE|PLUS|SOL|KIWOOM|HANARO|ARIRANG|KoAct|FOCUS|SPDR|iShares|Vanguard|Invesco"
    r"|ETF|ETN|펀드|투자신탁|채권|Fund|Trust|Shares|Index"
)
_ALT_PIVOT = re.compile(r"그러나|하지만|다만|대신|참고로|아래|다음은|덧붙|한편")
_ALT_PIVOT_EN = re.compile(
    r"\b(?:however|but|instead|alternatively|for\s+reference|below|in\s+addition|additionally|meanwhile"
    r"|that\s+said|note\s+that)\b",
    re.I,
)
_ALT_OFFER = re.compile(r"다음과 같|목록|나열|제공(할 수 있|해 ?드리|드릴)|알려\s*드리|소개|참고하")
_ALT_OFFER_EN = re.compile(
    r"\b(?:here\s+(?:is|are)|as\s+follows|the\s+following|list(?:ed|ing)?|can\s+(?:provide|offer|show|share)"
    r"|(?:would|could)\s+(?:provide|show)|let\s+me\s+(?:show|list)|see\s+below)\b",
    re.I,
)
_REFUSAL = re.compile(r"확인할 수 없|확인되지 않|찾을 수 없|제공되지 않|존재하지 않|알 수 없|없습니다|않습니다")
_REFUSAL_EN = re.compile(
    r"\b(?:cannot|can't|can\s+not|could\s+not|couldn't|unable\s+to)\s+(?:be\s+)?"
    r"(?:confirm(?:ed)?|find|found|verif(?:y|ied)|determin(?:e|ed)|identif(?:y|ied)|know|known)\b"
    r"|\b(?:is|are|was|were)\s+not\s+(?:available|provided|included|found|listed|recorded|known)\b"
    r"|\b(?:does|do)\s+not\s+exist\b|\bthere\s+(?:is|are)\s+no\b|\bno\s+(?:such|matching)\b"
    r"|\bnot\s+in\s+the\s+(?:provided\s+)?data\b",
    re.I,
)


def _refuses(s: str) -> bool:
    return bool(_REFUSAL.search(s) or _REFUSAL_EN.search(s))


def _looks_like_product_row(line: str) -> bool:
    """Does this one line look like 'one product'?  A refusal sentence is never an item, however short."""
    s = line.strip()
    if not (3 <= len(s) <= 160):
        return False
    body = _item_body(s)
    if len(body) < 3 or _refuses(body):
        return False
    if not (_ALT_CODE.search(body) or _ALT_NUM_UNIT.search(body) or _ALT_NUM_UNIT_EN.search(body)
            or _ALT_PRODUCT.search(body)):
        return False
    return bool(_MD_ITEM.match(s)) or len(body) <= 60   # without a marker, only a 'short line' counts as an item


def _drop_offer_tail(head: str) -> str:
    """Also cut the sentence that introduced the list ("…목록은 다음과 같습니다:" / "…are as follows:")."""
    for _ in range(3):
        body = head.rstrip()
        if not body:
            break
        start, sent = _last_sentence(body)
        s = sent.strip()
        if not s:
            break
        colon = s.endswith((":", "："))
        pivot = _ALT_PIVOT.search(s) or _ALT_PIVOT_EN.search(s)
        offer = _ALT_OFFER.search(s) or _ALT_OFFER_EN.search(s)
        if not (colon or (pivot and offer)):
            break
        if _refuses(s) and not colon:
            break                                      # never cut the refusal sentence itself
        head = body[:start].rstrip()
    return head


def strip_alternative_listing(text: str) -> str:
    """Cut the substitute product list appended after a cannot-confirm (cannot_confirm only)."""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    flags = [_looks_like_product_row(ln) for ln in lines]
    start, i = None, 0
    while i < len(lines):
        if not flags[i]:
            i += 1
            continue
        j, cnt = i, 0
        while j < len(lines) and (flags[j] or not lines[j].strip()):
            cnt += 1 if flags[j] else 0
            j += 1
        if cnt >= 3:
            start = i
            break
        i = j
    if start is None:
        return text
    head = _drop_offer_tail("\n".join(lines[:start]).rstrip())
    # A refusal must survive the cut.  If none does, the list was the whole answer, and then it is left
    # alone (dropping the entire answer here would be the bigger loss).
    if len(head.strip()) < 10 or not _refuses(head):
        return text
    return head


# Does the text already say it cannot be confirmed?  Then no prefix is added.  The Korean check matches
# only '확인할 수 없'/'확인되지 않' (cannot be / is not confirmed), and the English one the same family.
_CONFIRM_NEG_EN = re.compile(
    r"\b(?:cannot|can't|can\s+not|could\s+not|couldn't|unable\s+to)\s+(?:be\s+)?confirm(?:ed)?\b"
    r"|\bnot\s+(?:been\s+)?confirmed\b",
    re.I,
)


def normalize_answer(status: str, text: str, lang: str = "en") -> str:
    text = strip_closing_offer(strip_tool_call_syntax((text or "").strip()).strip()).strip()
    if status == "cannot_confirm":
        text = strip_alternative_listing(text)
    text = strip_markdown(text)
    if status == "cannot_confirm":
        if "확인할 수 없" not in text and "확인되지 않" not in text and not _CONFIRM_NEG_EN.search(text):
            text = f"{cannot_prefix(lang)} {text}".strip()
    elif status == "clarify":
        if not text.rstrip().endswith(("?", "？")):
            text = text.rstrip() + pick(lang, " Could you tell me which condition you mean?",
                                        " 어떤 조건을 말씀하시는지 알려주시겠어요?")
    return text or cannot_prefix(lang)
