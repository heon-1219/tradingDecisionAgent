"""Evidence / trace formatting and answer-policy helpers (pure functions)."""
from __future__ import annotations

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


def normalize_answer(status: str, text: str) -> str:
    text = (text or "").strip()
    if status == "cannot_confirm":
        if "확인할 수 없" not in text and "확인되지 않" not in text:
            text = f"{CANNOT_PREFIX} {text}".strip()
    elif status == "clarify":
        if not text.rstrip().endswith(("?", "？")):
            text = text.rstrip() + " 어떤 조건을 말씀하시는지 알려주시겠어요?"
    return text or CANNOT_PREFIX
