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


def normalize_answer(status: str, text: str) -> str:
    text = strip_closing_offer(strip_tool_call_syntax((text or "").strip()).strip()).strip()
    if status == "cannot_confirm":
        if "확인할 수 없" not in text and "확인되지 않" not in text:
            text = f"{CANNOT_PREFIX} {text}".strip()
    elif status == "clarify":
        if not text.rstrip().endswith(("?", "？")):
            text = text.rstrip() + " 어떤 조건을 말씀하시는지 알려주시겠어요?"
    return text or CANNOT_PREFIX
