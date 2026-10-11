"""The eval/audit scripts' Korean-only markers must also accept the English strings the (now bilingual)
agent emits for English questions, while the Korean markers keep matching the recorded runs from before
the translation (Task 0.1). These scripts are not a package, so they are loaded here the way a
command-line invocation would load them, straight off their file path."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass field resolution needs the module registered before exec
    spec.loader.exec_module(mod)
    return mod


eval_truth = _load("eval_truth")
eval_run = _load("eval_run")
audit_vector = _load("audit_vector")


# --------------------------------------------------------------------------- eval_truth.py
def test_decline_matches_korean_and_english():
    assert any(d in "제공된 데이터로는 확인할 수 없습니다." for d in eval_truth.DECLINE)
    assert any(d in "This cannot be confirmed with the provided data." for d in eval_truth.DECLINE)


def test_decline_rx_matches_korean_and_english():
    assert eval_truth.DECLINE_RX.search("문의하신 조건에 맞는 상품이 없습니다.")
    assert eval_truth.DECLINE_RX.search("No matching product was found in the data.")


def test_clarify_matches_korean_and_english():
    assert any(c in "혹시 KODEX 로봇액티브를 말씀하시나요?" for c in eval_truth.CLARIFY)
    assert any(c in "Could you tell me which condition you mean?" for c in eval_truth.CLARIFY)


def test_error_markers_and_is_infra_error_match_korean_and_english():
    ko = "제공된 데이터로는 확인할 수 없습니다. 시스템 처리 중 문제가 발생하여 답변을 생성하지 못했습니다(timeout)."
    en = ("This cannot be confirmed with the provided data. A system problem prevented an answer from "
          "being generated (timeout). Please rephrase the question more specifically and try again.")
    assert eval_truth.is_infra_error(ko)
    assert eval_truth.is_infra_error(en)
    # the script's own "the request itself failed" placeholder stays Korean and must keep matching
    assert eval_truth.is_infra_error("<요청 실패: ConnectionError>")


def test_score_cannot_confirm_passes_for_korean_and_english_declines():
    item = {"check": "cannot_confirm"}
    ok_ko, _ = eval_truth.score(item, "제공된 데이터로는 확인할 수 없습니다. 신용등급 'AAAA'는 존재하지 않는 값입니다.", [])
    ok_en, _ = eval_truth.score(item, "This cannot be confirmed with the provided data. 'AAAA' does not exist.", [])
    assert ok_ko and ok_en


# --------------------------------------------------------------------------- eval_run.py
def test_classify_matches_korean_and_english():
    assert eval_run.classify("제공된 데이터로는 확인할 수 없습니다.") == "cannot_confirm"
    assert eval_run.classify("This cannot be confirmed with the provided data.") == "cannot_confirm"
    assert eval_run.classify("혹시 KODEX 로봇액티브를 말씀하시나요") == "clarify"
    assert eval_run.classify("Could you tell me which condition you mean?") == "clarify"
    assert eval_run.classify("15,837건입니다.") == "answered"


# --------------------------------------------------------------------------- audit_vector.py
def test_round_regex_matches_korean_and_english():
    assert audit_vector._ROUND.search("1. 라운드 1: sql_query(...) → 결과 120자") .group(1) == "sql_query"
    assert audit_vector._ROUND.search("2. Round 1: sql_query(...) → result 120 chars, 3 rows").group(1) == "sql_query"


def test_notool_regex_matches_korean_and_english():
    assert audit_vector._NOTOOL.search("3. 라운드 2: 도구 호출 없이 텍스트로 응답")
    assert audit_vector._NOTOOL.search("3. Round 2: text reply without a tool call")


def test_document_evidence_check_matches_korean_and_english_context():
    def has_doc_evidence(context: str) -> bool:
        return "search" in context or "문서" in context or "document" in context.lower()

    assert has_doc_evidence("[Evidence 1] 외부 문서(국민성장펀드 안내서) · base date 2026-08-01")
    assert has_doc_evidence("[Evidence 1] External documents · base date 2026-08-01")
