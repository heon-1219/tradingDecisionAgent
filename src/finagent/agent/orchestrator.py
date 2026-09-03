"""Per-question orchestration: deterministic pre-analysis, tool-calling loop under a time budget,
evidence + trace assembly into the competition response schema."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, field

from ..models import AnswerResponse
from .answer import CANNOT_PREFIX, Evidence, fallback_answer, format_evidence, format_trace, normalize_answer
from .llm import LLMClient, ToolCall
from .tools import ToolRegistry

log = logging.getLogger("finagent.orchestrator")

_GRADE_MENTION = re.compile(r"(?:신용등급|등급)\s*(?:이|가|은|는)?\s*([A-Za-z]{1,5}[+\-0]?)|\b([A-Da-d]{1,5}[+\-0]?)\s*(?:이상|이하|등급)")
_RISK_MENTION = re.compile(r"(?:위험등급|위험\s*)?(\d{1,2})\s*등급")


@dataclass
class PreAnalysis:
    text: str
    invalid: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    # 질문이 특정 상품명을 지목했는데 그런 상품이 없는 경우: (언급, 유사 후보 이름들)
    unresolved: list[tuple[str, list[str]]] = field(default_factory=list)


# 답변이 "그 상품은 없다"는 사실을 인지했는지 판정할 때 쓰는 표현들
_ACKNOWLEDGED = ("확인할 수 없", "확인되지 않", "존재하지 않", "찾을 수 없", "없습니다만", "혹시", "말씀하시", "유사한", "다른 상품")


def _mention_resolved(mention: str, title: str) -> bool:
    """언급과 후보 상품명이 사실상 같은 것을 가리키는지.

    'KODEX AI로봇' vs 'KODEX 로봇액티브' 는 서로 포함하지 않으므로 불일치(→ 존재하지 않는 상품),
    '미래에셋 TIGER 미국나스닥100' vs 'TIGER 미국나스닥100' 이나 '국민성장펀드' vs
    '미래에셋국민참여형국민성장혼합자산투자신탁…' 은 일치로 본다(접두 운용사·일반 접미사 차이일 뿐).
    """
    from ..kb.search import _strip_generic, norm_title

    m, t = norm_title(mention), norm_title(title)
    core = _strip_generic(m)
    return bool(t) and (t in m or (len(core) >= 2 and core in t))


class Orchestrator:
    def __init__(self, settings, llm: LLMClient, tools: ToolRegistry, resolver, ontology, lexicon, system_prompt: str):
        self.settings, self.llm, self.tools = settings, llm, tools
        self.resolver, self.ontology, self.lexicon = resolver, ontology, lexicon
        self.system_prompt = system_prompt
        self.specs = tools.specs()
        self.base_date = (tools.meta.get("v_etf_kr") or {}).get("base_date") or "2026-08-21"

    # ------------------------------------------------------------ pre-analysis
    def pre_analyze(self, question: str) -> PreAnalysis:
        from ..kb.entity import extract_product_mentions

        lines: list[str] = []
        invalid: list[str] = []
        steps: list[str] = []
        mentions = extract_product_mentions(question)
        unresolved: list[tuple[str, list[str]]] = []
        for m in mentions:
            r = self.resolver.resolve(m, k=4)
            resolved = r.exact or any(_mention_resolved(m, h.title) for h in r.products[:3])
            tag = "정확 일치" if r.exact else ("유사 상품으로 해석 가능" if resolved else "일치하는 상품 없음")
            lines.append(f"- 상품 언급 '{m}': {tag}. " + ("; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in r.products[:4]) or "후보 없음"))
            if not resolved:
                cands = [h.title for h in r.products[:3]]
                unresolved.append((m, cands))
                lines.append(
                    f"  ⚠ '{m}' 이라는 상품은 데이터에 존재하지 않습니다. 위 후보는 이름이 비슷할 뿐 다른 상품이므로, "
                    f"그 중 하나를 질문의 상품인 것처럼 답하면 오답입니다. 존재하지 않음을 밝히고 후보를 제시하며 되물으세요."
                )
        whole = self.resolver.resolve(question, k=3) if not mentions else None
        comps = self.lexicon.match_companies(question)
        for c in comps:
            subs = self.lexicon.subsidiaries_of(c.id)
            info = f"- 기업 '{c.name_ko}' ({c.name_en}{', 비상장' if c.not_listed else ''})"
            if c.parent:
                p = self.lexicon.company(c.parent)
                info += f", 모회사: {p.name_ko if p else c.parent}"
            if subs:
                info += ", 자회사: " + ", ".join(s.name_ko for s in subs)
            if self.tools.kg is not None:
                cons = self.tools.kg.neighbors(c.name_ko, relation="constituentOf", limit=100).get("constituentOf", [])
                info += f", 편입 상품 {len(cons)}건(그래프)" if cons else ", 편입 상품 없음(그래프)"
            if c.note:
                info += f". 비고: {c.note}"
            lines.append(info)
        themes = self.lexicon.match_themes(question)
        for t in themes:
            cnt = ""
            if self.tools.kg is not None:
                n = len(self.tools.kg.neighbors(f"d:theme/{t.id}", relation="hasTheme_inverse", limit=10000).get("hasTheme_inverse", []))
                cnt = f", 그래프 태깅 상품 {n}건 (graph_neighbors('{t.label_ko}','hasTheme_inverse'))"
            lines.append(f"- 테마 '{t.label_ko}' ({t.label_en}; 동의어 {', '.join(t.synonyms[:6])}){cnt}")
        mgrs = self.lexicon.match_managers(question)
        if mgrs:
            lines.append("- 운용사: " + ", ".join(f"{m.canonical}(manager_name='{m.canonical}')" for m in mgrs))
        # ontology validators
        for m in _GRADE_MENTION.finditer(question):
            g = (m.group(1) or m.group(2) or "").upper()
            if re.fullmatch(r"[A-D]{1,5}[+\-0]?", g) and self.ontology is not None:
                ok, msg = self.ontology.validate_credit_grade(g)
                if not ok:
                    invalid.append(f"신용등급 '{g}'는 존재하지 않는 값입니다. {msg}")
                else:
                    ge = self.ontology.credit_grades_at_or_above(g)
                    lines.append(f"- 신용등급 '{g}' 유효. '{g} 이상' = crd_grd IN ({', '.join(repr(x) for x in ge)})")
        for m in _RISK_MENTION.finditer(question):
            v = m.group(1)
            if self.ontology is not None:
                ok, msg = self.ontology.validate_risk_grade(v)
                if not ok:
                    invalid.append(f"위험등급 '{v}등급'은 존재하지 않습니다. {msg}")
        if re.search(r"최근\s*(\d+)\s*개월", question):
            n = int(re.search(r"최근\s*(\d+)\s*개월", question).group(1))
            base = dt.date.fromisoformat(self.base_date)
            month = base.month - n
            year = base.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            cutoff = base.replace(year=year, month=month, day=min(base.day, 28))
            lines.append(f"- '최근 {n}개월' = {cutoff.isoformat()} ~ {self.base_date} (데이터 기준일 기준)")
        if whole is not None and whole.products and whole.products[0].score >= 60:
            lines.append("- 질문 전체와 유사한 상품: " + "; ".join(f"[{h.table}] {h.title} (id={h.id})" for h in whole.products[:3]))
        if invalid:
            lines.insert(0, "- ⚠ 온톨로지 위반: " + " / ".join(invalid))
        text = "[사전 분석 — 참고용, 최종 판단은 도구로 확인]\n" + ("\n".join(lines) if lines else "- 특기사항 없음")
        steps.append(
            "사전분석: 상품언급 " + (", ".join(mentions) if mentions else "없음")
            + (f"; 기업 {', '.join(c.name_ko for c in comps)}" if comps else "")
            + (f"; 테마 {', '.join(t.label_ko for t in themes)}" if themes else "")
            + (f"; 온톨로지 위반 {len(invalid)}건" if invalid else "")
            + (f"; 미존재 상품 {len(unresolved)}건" if unresolved else "")
        )
        return PreAnalysis(text, invalid, mentions, steps, unresolved)

    # ------------------------------------------------------------ main loop
    async def answer(self, question_id: str, question: str) -> AnswerResponse:
        t0 = time.monotonic()
        budget = float(self.settings.time_budget_sec)
        deadline = t0 + budget
        steps: list[str] = []
        evidences: list[Evidence] = []
        final: dict | None = None

        try:
            pre = self.pre_analyze(question)
            steps += pre.steps
        except Exception as e:  # noqa: BLE001
            log.exception("pre-analysis failed")
            pre = PreAnalysis("[사전 분석 실패]", [], [], [f"사전분석 오류: {type(e).__name__}"])
            steps += pre.steps

        conv = self.llm.start(self.system_prompt, self.specs, f"질문: {question}\n\n{pre.text}")
        last_text = ""
        nudged = False
        for rnd in range(1, self.settings.max_tool_rounds + 2):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                steps.append(f"라운드 {rnd}: 시간 예산 소진 → 강제 종료")
                break
            try:
                turn = await asyncio.wait_for(conv.step(), timeout=max(3.0, remaining - 1.0))
            except asyncio.TimeoutError:
                steps.append(f"라운드 {rnd}: LLM 응답 대기 시간 초과")
                break
            if turn.stop_reason == "error":
                steps.append(f"라운드 {rnd}: LLM 오류 — {turn.text[:120]}")
                break
            if turn.stop_reason == "refusal":
                steps.append(f"라운드 {rnd}: 모델 응답 거부")
                break
            if turn.text:
                last_text = turn.text
            fa = next((c for c in turn.tool_calls if c.name == "final_answer"), None)
            if fa is not None:
                final = fa.input or {}
                others = [c for c in turn.tool_calls if c.name != "final_answer"]
                steps.append(f"라운드 {rnd}: final_answer(status={final.get('status', '?')})" + (f" + 무시된 도구 {len(others)}개" if others else ""))
                break
            if not turn.tool_calls:
                steps.append(f"라운드 {rnd}: 도구 호출 없이 텍스트 응답")
                if rnd <= self.settings.max_tool_rounds and not nudged:
                    nudged = True
                    conv.add_user("답변을 final_answer 도구로 제출하세요(status 포함).")
                    continue
                break
            if rnd > self.settings.max_tool_rounds:
                steps.append(f"라운드 {rnd}: 최대 라운드 초과")
                break
            # execute all tool calls of this turn concurrently
            results = await asyncio.gather(*(self._run_tool(c) for c in turn.tool_calls))
            packed = []
            for call, res in zip(turn.tool_calls, results):
                desc = self._describe(call)
                if res.is_error:
                    steps.append(f"라운드 {rnd}: {desc} → 오류: {res.content[:100]}")
                else:
                    n = f", {res.evidence.row_count}건" if res.evidence and res.evidence.row_count is not None else ""
                    steps.append(f"라운드 {rnd}: {desc} → 결과 {len(res.content)}자{n}")
                    if res.evidence is not None and res.evidence.kind != "schema":
                        evidences.append(res.evidence)
                packed.append((call.id, res.content, res.is_error))
            conv.add_tool_results(packed)
            remaining = deadline - time.monotonic()
            if remaining < 14 or rnd == self.settings.max_tool_rounds:
                conv.add_user("[시간 예산 임박] 추가 도구 호출 없이 지금까지 확보한 근거로 즉시 final_answer 를 호출하세요.")
                steps.append(f"라운드 {rnd}: 시간 예산 임박 안내 (잔여 {remaining:.0f}s)")

        # ---------------------------------------------------------- compose
        if final is not None:
            status = str(final.get("status") or "answered")
            if status not in ("answered", "cannot_confirm", "clarify"):
                status = "answered"
            answer = normalize_answer(status, str(final.get("answer") or last_text))
            refs = [str(x) for x in (final.get("evidence_refs") or [])]
        elif last_text.strip():
            status = "answered" if evidences else "cannot_confirm"
            answer = normalize_answer(status, last_text)
            refs = []
            steps.append("final_answer 미호출 → 마지막 텍스트를 답변으로 사용")
        else:
            reason = "시간 초과" if time.monotonic() >= deadline else "LLM 응답 없음"
            status, answer, refs = "cannot_confirm", fallback_answer(reason), []
        if pre.invalid and status == "answered" and not any(k in answer for k in ("확인할 수 없", "존재하지 않")):
            answer = f"{CANNOT_PREFIX} {' '.join(pre.invalid)}\n\n(참고) {answer}"
            status = "cannot_confirm"
            steps.append("온톨로지 위반 조건이 있어 답변을 확인불가로 보정")
        # 존재하지 않는 상품명을 물었는데 모델이 유사 상품으로 그냥 답해버린 경우를 잡는다.
        # 답변불가 문항은 배점이 걸려 있어 프롬프트 준수에만 맡기지 않는다.
        if pre.unresolved and status == "answered" and not any(k in answer for k in _ACKNOWLEDGED):
            names = [m for m, _ in pre.unresolved]
            shown = [c for _, group in pre.unresolved for c in group[:2]]
            asked = (
                f"이름이 비슷한 상품으로는 {', '.join(shown)} 등이 있습니다. 이 중 찾으시는 상품이 있나요?"
                if shown else "찾으시는 상품명을 다시 확인해 주시겠어요?"
            )
            answer = (
                f"'{', '.join(names)}'와(과) 정확히 일치하는 상품은 제공된 데이터에서 확인할 수 없습니다. {asked}"
                f"\n\n(참고: 아래는 유사 상품 기준으로 조회한 내용입니다)\n{answer}"
            )
            status = "clarify"
            steps.append(f"존재하지 않는 상품({', '.join(names)})을 답변에 사용 → 역질문으로 보정")
        elapsed = time.monotonic() - t0
        steps.append(f"최종: status={status}, 근거 {len(evidences)}건, 소요 {elapsed:.1f}s")
        return AnswerResponse(
            question_id=question_id,
            question=question,
            retrieved_context=format_evidence(evidences, refs),
            think_trace=format_trace(steps),
            answer=answer,
        )

    async def _run_tool(self, call: ToolCall):
        return await self.tools.execute(call.name, call.input)

    @staticmethod
    def _describe(call: ToolCall) -> str:
        a = call.input or {}
        key = a.get("sql") or a.get("query") or a.get("mention") or a.get("entity") or a.get("sparql") or a.get("table") or ""
        key = re.sub(r"\s+", " ", str(key))[:140]
        return f"{call.name}({key})"
