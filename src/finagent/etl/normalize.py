"""Pure value normalizers used by the ETL (no pandas dependency so they are trivially testable)."""
from __future__ import annotations

import math
import re
import unicodedata


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def yyyymmdd_to_iso(v) -> str | None:
    """20260821.0 / 20260821 / '20260821' -> '2026-08-21'; anything else -> None."""
    if _is_nan(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not re.fullmatch(r"\d{8}", s):
        return None
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


_TRUE = {"y", "1", "1.0", "true", "판매중", "예", "가능"}
_FALSE = {"n", "0", "0.0", "false", "판매완료", "판매중지", "아니오", "불가"}


def to_flag(v) -> int | None:
    if _is_nan(v):
        return None
    s = str(v).strip().lower()
    if s in _TRUE:
        return 1
    if s in _FALSE:
        return 0
    return None


def clean_text(v) -> str | None:
    if _is_nan(v):
        return None
    s = unicodedata.normalize("NFKC", str(v)).strip()
    return s or None


# Share-class suffix patterns seen in prfd01n001.itm_nm, applied in order.
_CLASS_PATTERNS = [
    r"\s*종류\s*[A-Za-z0-9\-]+.*$",                       # 종류C1-e (서민형)
    r"\s*\(?\s*class\s*[A-Za-z0-9\-]+\)?.*$",              # Class A-e / (Class C)
    r"\s*_[A-Za-z0-9\-]+\s*(\(.*\))?\s*$",                 # _Ce, _C-Pe, _Cpe(퇴직연금)
    r"\s*\((?:[A-Z]{1,2}-?[A-Za-z0-9]{0,4}|C-?P?e?\d*)\)\s*$",  # (C-Pe), (A), (C-R), (Ae), (S)
    r"\s*\[(?:A|C|Ce|C-e|C-Pe|C-P2e|Ae|S|I|W|P)\]\s*$",
    r"\s+(?:A|C|Ce|C-e|C-Pe|C-P2e|Ae|S|I|W|P|CP|Cw|Cf|C-W)\s*(?:\(.*\))?$",  # trailing bare class token
]


def fund_family_key(name: str | None) -> str:
    """Strip share-class suffixes so classes of the same fund collapse to one key."""
    s = unicodedata.normalize("NFKC", name or "")
    for p in _CLASS_PATTERNS:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def risk_grade_from_name(v) -> int | None:
    if _is_nan(v):
        return None
    m = re.search(r"(\d)\s*등급", str(v))
    return int(m.group(1)) if m else None
