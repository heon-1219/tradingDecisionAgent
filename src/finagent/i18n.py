"""Language of user-facing text: English by default, Korean when the question is written in Korean.

Product and company names in the data are Korean, so English questions often contain Hangul
("Which ETFs hold 삼성전자?"), and Korean questions often contain Latin brand names ("TIGER 미국S&P500
총보수는?"). Presence of Hangul therefore decides nothing. Instead, pure-Hangul tokens vote Korean with
their syllable count, Hangul-free tokens vote English with their lowercase-letter count, and mixed-script
tokens (product names such as 미국S&P500 or 삼성전자우(005935)) and all-caps names (TIGER, ETF) do not
vote. A tie, including an empty question, is English.

Korean grammar overrides the vote: a particle glued to a Latin name ("VanEck China Semiconductor ETF의
총보수는…", "Series 1의 운용사") or a Korean sentence ending ("…인가요?", "…알려줘") means the question is
Korean however many English words the product name contributes.
"""
from __future__ import annotations

import re
from typing import Literal

Lang = Literal["en", "ko"]

_HANGUL = re.compile(r"[가-힣]")
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
_LOWER = re.compile(r"[a-z]")
# A Korean particle directly after a Latin/digit name, ending the token ("ETF의", "QQQ는", "1의").
_PARTICLE_ON_NAME = re.compile(
    r"[A-Za-z0-9)](?:의|은|는|이|가|을|를|에|와|과|로|으로|도|에서|만|보다|까지|부터|랑|하고)[?.!,]*$")
# A Korean sentence ending on the last Hangul word ("몇 개인가요?", "알려줘", "있나요", "뭐야?").
_SENTENCE_END = re.compile(r"(?:요|까|죠|줘|니다|냐|야|는지|을까|인가)[?.!~]*$")


def detect_lang(text: str | None) -> Lang:
    toks = (text or "").split()
    hangul_words = [t for t in toks if _HANGUL.search(t) and not _LATIN_OR_DIGIT.search(t)]
    if any(_PARTICLE_ON_NAME.search(t) for t in toks) or (hangul_words and _SENTENCE_END.search(hangul_words[-1])):
        return "ko"
    ko = en = 0
    for tok in toks:
        hangul = _HANGUL.findall(tok)
        if hangul and not _LATIN_OR_DIGIT.search(tok):
            ko += len(hangul)
        elif not hangul:
            en += len(_LOWER.findall(tok))
    return "ko" if ko > en else "en"


def pick(lang: Lang, en: str, ko: str) -> str:
    """The English or Korean variant of a user-facing message."""
    return ko if lang == "ko" else en


def language_name(lang: Lang) -> str:
    """How the prompt names the answer language."""
    return "Korean" if lang == "ko" else "English"
