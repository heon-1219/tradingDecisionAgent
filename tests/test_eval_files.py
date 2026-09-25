"""data/eval/groundtruth_en.jsonl must be the English mirror of data/eval/groundtruth.jsonl: same ids,
same order, identical grading fields, only `question` translated. The English text is lifted verbatim
from web/console.html's SET_GT.en batch preset (the console's own English eval-set loader), so this file
also guards against the two ever drifting apart.

`eval_truth.py` is not a package, so it is loaded the way tests/test_eval_truth_markers.py already does:
straight off its file path, as a command-line invocation would.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

from finagent.i18n import detect_lang

ROOT = pathlib.Path(__file__).resolve().parents[1]
GT_KO = ROOT / "data/eval/groundtruth.jsonl"
GT_EN = ROOT / "data/eval/groundtruth_en.jsonl"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eval_truth = _load("eval_truth")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_files_exist_with_54_questions():
    ko = _read_jsonl(GT_KO)
    en = _read_jsonl(GT_EN)
    assert len(ko) == 54
    assert len(en) == 54


def test_same_ids_in_same_order():
    ko = _read_jsonl(GT_KO)
    en = _read_jsonl(GT_EN)
    assert [i["question_id"] for i in ko] == [i["question_id"] for i in en]


def test_grading_fields_identical_between_ko_and_en():
    ko = _read_jsonl(GT_KO)
    en = _read_jsonl(GT_EN)
    for ko_item, en_item in zip(ko, en):
        qid = ko_item["question_id"]
        for field in ("truth_sql", "check", "expect", "category", "difficulty", "truth_kind", "truth_graph"):
            assert ko_item.get(field) == en_item.get(field), f"{qid}: {field} differs"


def test_en_question_replaced_and_non_empty():
    ko = _read_jsonl(GT_KO)
    en = _read_jsonl(GT_EN)
    for ko_item, en_item in zip(ko, en):
        assert en_item["question"].strip()


def test_en_questions_detected_as_english():
    en = _read_jsonl(GT_EN)
    for item in en:
        assert detect_lang(item["question"]) == "en", f"{item['question_id']}: {item['question']!r}"


def test_en_questions_match_console_html_set_gt():
    """The English text must be lifted verbatim from web/console.html's SET_GT.en, not retyped."""
    html = (ROOT / "web/console.html").read_text(encoding="utf-8")
    start = html.index("const SET_GT = {")
    end = html.index("\n};", start)
    block = html[start:end]
    en_start = block.index("en: [")
    en_end = block.index("\n  ],", en_start)
    en_block = block[en_start:en_end]
    pairs = re_findall_pairs(en_block)
    console_by_id = {qid: text for qid, text in pairs}
    assert len(console_by_id) == 54

    en = _read_jsonl(GT_EN)
    for item in en:
        assert item["question"] == console_by_id[item["question_id"]]


def re_findall_pairs(block: str) -> list[tuple[str, str]]:
    import re
    return re.findall(r'\["(G-\d+)",\s*"((?:[^"\\]|\\.)*)"\]', block)


def test_score_treats_english_decline_like_korean_for_cannot_confirm():
    """Reuses the loading approach of tests/test_eval_truth_markers.py: an expect='cannot_confirm' item
    from the answer key must pass score() whether the model declined in Korean or in English."""
    item = next(i for i in _read_jsonl(GT_KO) if i.get("expect") == "cannot_confirm")
    assert item.get("check") == "cannot_confirm"

    ko_answer = "제공된 데이터로는 확인할 수 없습니다. 문의하신 조건에 맞는 상품이 없습니다."
    en_answer = "This cannot be confirmed with the provided data. No matching product was found in the data."

    ok_ko, _ = eval_truth.score(item, ko_answer, [])
    ok_en, _ = eval_truth.score(item, en_answer, [])
    assert ok_ko
    assert ok_en
