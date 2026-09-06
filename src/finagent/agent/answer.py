"""Evidence / trace formatting and answer-policy helpers (pure functions)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CANNOT_PREFIX = "제공된 데이터로는 확인할 수 없습니다."

FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "cannot_confirm", "clarify"],
            "description": "answered=근거 데이터로 답변함, cannot_confirm=제공 데이터로 확인 불가, clarify=조건이 모호하여 역질문",
        },
        "answer": {"type": "string", "description": "사용자에게 보여줄 최종 답변(한국어). 근거 수치·상품코드·기준일 포함"},
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "답변에 사용한 핵심 근거(테이블명, 상품코드/명, 문서 제목)",
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
        return "참조한 데이터 없음"
    blocks = []
    for i, e in enumerate(evs, 1):
        head = f"[근거 {i}] "
        if e.table:
            head += f"{e.korean_name or ''}({e.table})".strip()
        else:
            head += {"docs": "외부 문서", "graph": "지식그래프", "search": "상품 검색", "entity": "엔티티 해석", "schema": "스키마"}.get(e.kind, e.kind)
        if e.base_date:
            head += f" · 기준일 {e.base_date}"
        if e.row_count is not None:
            head += f" · {e.row_count}건"
        lines = [head, f"  {e.kind}: {e.detail}"]
        if e.refs:
            uniq = list(dict.fromkeys(r for r in e.refs if r))
            shown = uniq[:20]
            lines.append("  참조: " + " | ".join(shown) + (f" | ... 외 {len(uniq) - 20}건" if len(uniq) > 20 else ""))
        blocks.append("\n".join(lines))
    if extra_refs:
        blocks.append("[답변 근거 요약] " + " | ".join(extra_refs))
    return "\n".join(blocks)


def format_trace(steps: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else "-"


def fallback_answer(reason: str) -> str:
    return (
        f"{CANNOT_PREFIX} 시스템 처리 중 문제가 발생하여 답변을 생성하지 못했습니다({reason}). "
        "질문을 조금 더 구체적으로 바꾸어 다시 시도해 주세요."
    )


# --------------------------------------------------------- 도구 호출 구문이 답변 본문에 새는 것
# 모델이 final_answer 를 '호출'하는 대신 그 호출 구문을 글자로 써서 보내는 일이 실제로 있었다.
# 기록된 54개 답변 중 6개가 그랬고, 사용자에게 그대로 노출됐다:
#     final_answer(status="completed", answer="… 총 22개입니다.")
#     제공된 데이터로는 확인할 수 없습니다. final_answer(status="cannot_confirm", answer="…")
#     final_answer 도구를 사용하여 답변을 제출하겠습니다:\n```plaintext\nfinal_answer(…)\n```
#
# 지우기만 하면 답이 통째로 사라진다 — 진짜 답변 문장은 answer="…" 안에 들어 있다. 그래서
# 껍데기만 벗기고 안의 본문을 꺼내 올린다(unwrap). 출력 상한을 낮추면 이 구문이 닫히기 전에
# 잘릴 수 있으므로, 닫는 괄호·따옴표가 없는 미완성 조각도 같은 방식으로 처리한다.
_LEAK_CALL = re.compile(r"final_answer\s*\(", re.I)
_LEAK_ANSWER_KW = re.compile(r"\banswer\s*=\s*(\"|')", re.I)
_LEAK_STATUS_KW = re.compile(r"""^\W*\bstatus\s*=\s*["']?\w*["']?\s*,?\s*""", re.I)
_LEAK_MARKERS = ('final_answer', 'status="', "status='", 'answer="', "answer='")
_ESCAPES = {"n": "\n", "t": "\t", "r": "", "\\": "\\", '"': '"', "'": "'"}


def _unescape(s: str) -> str:
    """\\n·\\" 같은 소스 수준 이스케이프를 실제 문자로. 모르는 이스케이프는 원문 그대로 둔다."""
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
    """여는 괄호에 대응하는 닫는 괄호를 찾는다(따옴표 안의 괄호는 세지 않는다).

    반환: (호출이 끝나는 위치[배타적], 인자 문자열). 닫히지 않았으면 문자열 끝까지가 인자다.
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
    """at 위치부터 닫는 따옴표(없으면 끝)까지의 문자열 리터럴 본문."""
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
    """답변 본문에 새어 나온 final_answer 호출 구문을 벗기고 안의 본문만 남긴다."""
    if not text or not any(m in text for m in _LEAK_MARKERS):
        return text
    out = text
    for _ in range(4):  # 중첩·반복 누출도 있으나 무한 루프는 만들지 않는다
        m = _LEAK_CALL.search(out)
        if not m:
            break
        end, args = _call_span(out, m.end() - 1)
        am = _LEAK_ANSWER_KW.search(args)
        inner = _quoted_value(args, am.end(), am.group(1)) if am else ""
        out = out[:m.start()] + inner + out[end:]
    # 괄호 없이 인자 조각만 남은 경우(잘린 출력): status=… 를 떼고 answer="…" 를 벗긴다.
    if any(m in out for m in _LEAK_MARKERS[1:]):
        stripped = _LEAK_STATUS_KW.sub("", out.lstrip(), count=1)
        am = _LEAK_ANSWER_KW.search(stripped)
        if am and stripped[:am.start()].strip(" \t\r\n,\"'") == "":
            out = _quoted_value(stripped, am.end(), am.group(1))
    # 남은 껍데기 — 도구 이름만 언급하는 안내 줄과, 그 호출을 감싸고 있던 코드펜스.
    lines = [ln for ln in out.splitlines() if "final_answer" not in ln]
    if any(ln.strip() for ln in lines):
        out = "\n".join(lines)
    body = out.strip()
    if body.startswith("```") and body.endswith("```"):
        inner_lines = body.splitlines()[1:-1]
        if inner_lines:
            out = "\n".join(inner_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    dup = f"{CANNOT_PREFIX} {CANNOT_PREFIX}"
    while dup in out:
        out = out.replace(dup, CANNOT_PREFIX)
    return out


# ------------------------------------------------------- 정보가 0인 맺음말
# 기록된 54개 답변 중 8개가 "추가적인 정보가 필요하시면 …" 류의 문장으로 끝났다. 그 문장은 답에
# 아무것도 더하지 않으면서 생성 시간을 쓴다(측정: 고정 2.7초 + 63토큰/초).
#
# 무엇을 지우면 안 되는가가 더 중요하다. 다음 중 하나라도 걸리면 남긴다:
#   * 숫자·% 가 있는 문장 — 수치는 답의 본체다.
#   * "투자 판단은 본인 책임" 류의 유의사항 — 지우면 고지가 사라진다(그 문구를 줄이는 일은
#     프롬프트 규칙 4가 맡는다. 이미 나온 문장을 여기서 잘라내지는 않는다).
#   * "…없습니다/않습니다/제외" 처럼 확인 불가·제외 사실을 말하는 문장.
# 남는 것은 순수한 "더 필요하면 물어보세요"뿐이고, 그때만 지운다. 맨 뒤 문장만 최대 2개까지 본다.
_OFFER_TRIGGER = re.compile(r"추가(?:적인)?\s*(?:정보|문의|사항|질문)|더\s*궁금|도움이\s*필요")
_OFFER_CLOSE = re.compile(r"문의|말씀|연락|알려\s*주|권장|바랍니다|주세요|주십시오|드립니다")
_OFFER_KEEP = re.compile(r"[0-9%]|책임|제외|없습니다|않습니다|아닙니다|못했습니다")


def _last_sentence(text: str) -> tuple[int, str]:
    """(문장 시작 위치, 문장). 뒤에서부터 가장 가까운 . ! ? 줄바꿈 을 경계로 삼는다."""
    body = text.rstrip()
    i = len(body) - 1
    while i >= 0 and body[i] in ".!?\"')]”’":
        i -= 1
    j = i
    while j >= 0 and body[j] not in ".!?\n":
        j -= 1
    return j + 1, body[j + 1:]


def strip_closing_offer(text: str) -> str:
    """맨 끝의 '추가 문의 바랍니다' 류 문장을 지운다. 사실·유의사항이 섞인 문장은 건드리지 않는다."""
    for _ in range(2):
        if not text.strip():
            break
        start, sent = _last_sentence(text)
        if not (_OFFER_TRIGGER.search(sent) and _OFFER_CLOSE.search(sent)) or _OFFER_KEEP.search(sent):
            break
        text = text[:start].rstrip()
    return text


# ----------------------------------------------------------------- 마크다운 → 줄글
# 주최측 응답 예시의 answer 는 평문 한 줄이다("총보수 0.03%인 A …"). 우리 답변은 굵게·번호 목록이
# 섞여 나왔다(기록된 54개 중 13개). 프롬프트에 규칙을 넣되, 그것만 믿지 않는다 —
# 이 프로젝트에서 '미리 준 프롬프트 규칙'은 지켜지지 않은 전례가 있다.
#
# 벗기는 것은 **문법**뿐이고 내용은 건드리지 않는다. 0.03%·1,436.24 같은 숫자, du_er_1y 같은
# 컬럼명, 문장 중간의 별표는 마크다운이 아니므로 그대로 둔다. 목록은 지우는 것이 아니라
# 마커만 떼고 쉼표로 이어 한 문장으로 만든다(항목 자체가 사라지면 답이 사라진다).
_MD_FENCE = re.compile(r"^\s*(?:```|~~~)")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+")
_MD_RULE = re.compile(r"^[\s\-*_=|:]{3,}$")            # ---, ***, |---|:--:| (표 구분선)
_MD_ITEM = re.compile(r"^\s*(?:[-*•·]|\(?\d{1,3}[.)])\s+")
_MD_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_MD_BOLD = re.compile(r"\*\*(\S(?:[^*\n]*\S)?)\*\*")
_MD_BOLD_U = re.compile(r"__(\S(?:[^_\n]*\S)?)__")
_MD_ITALIC = re.compile(r"(?<![\w*])\*(\S(?:[^*\n]*\S)?)\*(?![\w*])")
_MD_CODE = re.compile(r"`([^`\n]+)`")
_SENT_END = ".!?。？！"
_CLOSERS = ")]}”’\"'』」"


def _ends_sentence(s: str) -> bool:
    """문장이 끝났는가. 닫는 괄호·따옴표는 넘겨보고 판정한다("…(11조)" 는 끝나지 않았다)."""
    t = s.rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1]
    return bool(t) and t[-1] in _SENT_END


def _item_body(line: str) -> str:
    """목록 항목에서 마커를 뗀 본문. 항목이 아니면 원문 그대로."""
    m = _MD_ITEM.match(line)
    return line[m.end():].strip() if m else line.strip()


def _joined_item(s: str) -> str:
    """쉼표로 이어 붙일 항목 하나. 끝의 마침표만 떼어 'A입니다., B' 가 되지 않게 한다."""
    s = s.strip()
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1].rstrip()
    return s


def strip_markdown(text: str) -> str:
    """마크다운 문법을 벗기고, 줄 단위 목록을 쉼표로 이은 줄글로 만든다."""
    if not text or not text.strip():
        return text
    kinds: list[tuple[str, str]] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if _MD_FENCE.match(line):
            continue                                   # 코드펜스 자체만 제거하고 안의 본문은 남긴다
        if not line.strip():
            kinds.append(("blank", ""))
            continue
        if _MD_RULE.match(line.strip()):
            continue                                   # 수평선·표 구분선은 내용이 없다
        if _MD_TABLE_ROW.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            kinds.append(("item", " ".join(c for c in cells if c)))
            continue
        if _MD_ITEM.match(line):
            kinds.append(("item", _item_body(line)))
            continue
        kinds.append(("text", _MD_HEADER.sub("", line).strip()))
    # 항목과 항목 사이의 빈 줄은 목록의 일부다(모델이 항목 사이를 띄우는 일이 잦다).
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
            out.pop()                                  # 목록 바로 앞의 빈 줄은 흡수한다
        if out:
            prev = out[-1].rstrip()
            if prev.endswith((":", "：")):              # "…다음과 같습니다:" → "…다음과 같습니다."
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
    res = res.replace("**", "")                        # 짝이 맞지 않는 강조(잘린 출력)
    return re.sub(r"\n{3,}", "\n\n", res).strip()


# ------------------------------------------------- 거절한 뒤에 붙는 '대체 상품 목록'
# 실제 사고: "한화 에어로스페이스의 **자회사를 편입한** ETF 중 …" 에 대해 "찾을 수 없습니다" 라고
# 해놓고, 그 뒤에 "그러나 한화 에어로스페이스를 편입한 ETF 목록을 제공할 수 있습니다" 하며 40개를
# 나열했다. 묻지 않은 개체(자회사 → 회사 본체)로 갈아탄 대체 답변이다. 기록된 54개 중 G-020 이
# 같은 모양이다(공모펀드 → 국내 ETF).
#
# 지우면 안 되는 쪽과의 차이가 이 함수의 전부다:
#   살린다 — "…유효한 신용등급은 AAA, AA+, AA0, … 입니다."  조건의 유효값을 한 문장 안에 인라인으로.
#   지운다 — "…목록은 다음과 같습니다:\n1. 아이엠에셋 200 (KR70007N0000)\n2. …"  상품을 행 단위로.
# 그래서 판정은 (1) 줄 단위로 끊긴 항목이 3개 이상 연속하고, (2) 그 항목들이 상품처럼 생겼을 때
# — 코드·티커, 단위 붙은 수치, 상품군 낱말 — 에만 걸린다. 인라인 나열은 몇 개가 나열되든 손대지
# 않는다. 마크다운 제거보다 **먼저** 돌려서 줄 구조가 남아 있을 때 본다.
_ALT_CODE = re.compile(r"\b(?:KR[0-9A-Z]{9,11}|[0-9]{6})\b")
_ALT_NUM_UNIT = re.compile(r"\d[\d,.]*\s*(?:조|억|만원|원|%|퍼센트|개|건|USD|달러)")
_ALT_PRODUCT = re.compile(
    r"KODEX|TIGER|RISE|ACE|PLUS|SOL|KIWOOM|HANARO|ARIRANG|KoAct|FOCUS|SPDR|iShares|Vanguard|Invesco"
    r"|ETF|ETN|펀드|투자신탁|채권|Fund|Trust|Shares|Index"
)
_ALT_PIVOT = re.compile(r"그러나|하지만|다만|대신|참고로|아래|다음은|덧붙|한편")
_ALT_OFFER = re.compile(r"다음과 같|목록|나열|제공(할 수 있|해 ?드리|드릴)|알려\s*드리|소개|참고하")
_REFUSAL = re.compile(r"확인할 수 없|확인되지 않|찾을 수 없|제공되지 않|존재하지 않|알 수 없|없습니다|않습니다")


def _looks_like_product_row(line: str) -> bool:
    """이 줄 하나가 '상품 한 건'으로 보이는가. 거절 문장은 아무리 짧아도 항목이 아니다."""
    s = line.strip()
    if not (3 <= len(s) <= 160):
        return False
    body = _item_body(s)
    if len(body) < 3 or _REFUSAL.search(body):
        return False
    if not (_ALT_CODE.search(body) or _ALT_NUM_UNIT.search(body) or _ALT_PRODUCT.search(body)):
        return False
    return bool(_MD_ITEM.match(s)) or len(body) <= 60   # 마커가 없으면 '짧은 행'일 때만 항목으로 본다


def _drop_offer_tail(head: str) -> str:
    """목록을 소개하던 문장("…목록은 다음과 같습니다:")까지 함께 떼어낸다."""
    for _ in range(3):
        body = head.rstrip()
        if not body:
            break
        start, sent = _last_sentence(body)
        s = sent.strip()
        if not s:
            break
        colon = s.endswith((":", "："))
        if not (colon or (_ALT_PIVOT.search(s) and _ALT_OFFER.search(s))):
            break
        if _REFUSAL.search(s) and not colon:
            break                                      # 거절 문장 자체는 절대 떼지 않는다
        head = body[:start].rstrip()
    return head


def strip_alternative_listing(text: str) -> str:
    """확인 불가로 답한 뒤 이어 붙인 대체 상품 목록을 잘라낸다(cannot_confirm 전용)."""
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
    # 잘라낸 뒤에도 거절이 남아 있어야 한다. 남지 않으면 목록이 답의 전부였다는 뜻이고,
    # 그때는 손대지 않는다(여기서 답을 통째로 날리는 것이 더 큰 손해다).
    if len(head.strip()) < 10 or not _REFUSAL.search(head):
        return text
    return head


def normalize_answer(status: str, text: str) -> str:
    text = strip_closing_offer(strip_tool_call_syntax((text or "").strip()).strip()).strip()
    if status == "cannot_confirm":
        text = strip_alternative_listing(text)
    text = strip_markdown(text)
    if status == "cannot_confirm":
        if "확인할 수 없" not in text and "확인되지 않" not in text:
            text = f"{CANNOT_PREFIX} {text}".strip()
    elif status == "clarify":
        if not text.rstrip().endswith(("?", "？")):
            text = text.rstrip() + " 어떤 조건을 말씀하시는지 알려주시겠어요?"
    return text or CANNOT_PREFIX
