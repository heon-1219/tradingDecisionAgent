"""Offline stand-ins for the model providers, for the fake world and the demo server.

`ChatRuntime` checks a model key with one live call (`llm_admin.validate_llm`), lists a key's models
(`llm_admin.fetch_models`) and builds a provider client (`agent.llm.make_llm`) when the provider, the
model or the key changes. `DemoLLMAdmin` replaces all three, so switching the provider in the demo's
Settings never reaches a network:

- the key check answers "ok" for `DEMO_VALID_KEY` (and for the fake world's own dummy Gemini key),
  "invalid_key" for `DEMO_INVALID_KEY`, and "unreachable" for anything else (no provider can be
  reached from the demo, so a real key is neither accepted nor called invalid);
- the model list is a fixed list per provider, for the same keys (the invalid key and an unknown key
  raise what a provider SDK would, so the API classifies them the same way);
- the client factory records `(provider, model)` and always returns a fresh scripted `DemoLLM`.

`GROUNDED_SAMPLE` is the canned Gemini `generateContent` answer with Google Search grounding
metadata that the fake transport serves when the fake world is built with `grounded=True`.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "src"))
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))  # at the end: scripts/trace.py must never shadow the stdlib `trace`

from _demo_llm import DemoLLM  # noqa: E402

DEMO_VALID_KEY = "demo-valid-key"  # test values only: the demo accepts this key for any provider
DEMO_INVALID_KEY = "demo-invalid-key"  # ... and rejects this one
DEMO_GEMINI_KEY = "demo-gemini-key"  # the fake world's own dummy key (it only switches grounding on)

KEY_FIELDS = {"gemini": "gemini_api_key", "openai": "openai_api_key", "anthropic": "anthropic_api_key"}

# id, label, created (epoch seconds): made-up lists in the shape `llm_admin.fetch_models` returns.
DEMO_MODELS: dict[str, list[dict]] = {
    "gemini": [
        {"id": "gemini-3.8-flash", "label": None, "created": 0.0},
        {"id": "gemini-3.8-pro", "label": None, "created": 0.0},
        {"id": "gemini-embedding-002", "label": None, "created": 0.0},  # filtered out by the API
    ],
    "openai": [
        {"id": "gpt-6", "label": None, "created": 1790000000.0},
        {"id": "gpt-6-mini", "label": None, "created": 1789000000.0},
        {"id": "text-embedding-4", "label": None, "created": 1780000000.0},  # filtered out by the API
    ],
    "anthropic": [
        {"id": "claude-opus-5-5", "label": "Claude Opus 5.5", "created": 1790000000.0},
        {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "created": 1785000000.0},
    ],
}


class DemoProviderError(Exception):
    """What a provider SDK raises, reduced to the `status_code` that `llm_admin` classifies."""

    def __init__(self, status_code: int):
        super().__init__(f"demo provider error {status_code}")
        self.status_code = status_code


def _status_for(key: Any) -> str:
    if key in (DEMO_VALID_KEY, DEMO_GEMINI_KEY):
        return "ok"
    if key == DEMO_INVALID_KEY:
        return "invalid_key"
    return "unreachable"


class DemoLLMAdmin:
    """The three injectable model-provider functions of `ChatRuntime`; see the module docstring.
    `built` lists `(provider, model)` for every client the runtime asked for, `checked` and `listed`
    the providers that were validated and listed."""

    def __init__(self) -> None:
        self.built: list[tuple[str, str]] = []
        self.checked: list[str] = []
        self.listed: list[str] = []

    async def validate(self, settings: Any) -> str:
        provider = getattr(settings, "llm_provider", None)
        self.checked.append(str(provider))
        field = KEY_FIELDS.get(provider)
        return "ok" if field is None else _status_for(getattr(settings, field, None))

    async def list_models(self, provider: str, api_key: str, settings: Any) -> list[dict]:
        self.listed.append(provider)
        status = _status_for(api_key)
        if status == "invalid_key":
            raise DemoProviderError(401)
        if status != "ok":
            raise DemoProviderError(503)
        return [dict(model) for model in DEMO_MODELS.get(provider, [])]

    def build(self, settings: Any) -> DemoLLM:
        self.built.append((str(getattr(settings, "llm_provider", None)), str(getattr(settings, "llm_model", None))))
        return DemoLLM()

    def runtime_kwargs(self) -> dict:
        """The `ChatRuntime` keyword arguments that put these fakes in place."""
        return {"llm_validator": self.validate, "llm_models": self.list_models, "llm_factory": self.build}


# A canned Gemini `generateContent` answer with Google Search grounding metadata, in the shape
# `research/grounding.py` parses. The search-suggestions HTML is a small made-up sample: a style
# block and plain links without `target`, like the real `searchEntryPoint.renderedContent`.
GROUNDED_SAMPLE: dict = {
    "candidates": [{
        "content": {"role": "model", "parts": [{"text": (
            "An S&P 500 index fund holds the stocks of the index in the same proportions, so its return follows "
            "the index before costs. Funds that track the same index differ mainly in their expense ratio and in "
            "how closely they follow the index. This is demo text served by the fake world, not a web result."
        )}]},
        "groundingMetadata": {
            "webSearchQueries": ["how does an S&P 500 index fund work", "index fund expense ratio"],
            "groundingChunks": [
                {"web": {"uri": "https://www.investor.gov/introduction-investing/investing-basics/glossary/index-fund",
                         "title": "investor.gov"}},
                {"web": {"uri": "https://funds.example.com/guide/index-funds", "title": "funds.example.com"}},
            ],
            "searchEntryPoint": {"renderedContent": (
                "<style>.demo-sg{display:flex;align-items:center;gap:8px;padding:8px 10px;font:13px/1.4 Arial,sans-serif;"
                "color:#1f1f1f;background:#fafafa;border:1px solid #d2d2d2;border-radius:8px;overflow-x:auto;"
                "white-space:nowrap}.demo-sg b{color:#5e5e5e;font-weight:500}.demo-sg a{padding:5px 12px;"
                "border:1px solid #d2d2d2;border-radius:16px;background:#fff;color:#0b57d0;text-decoration:none}"
                "@media (prefers-color-scheme:dark){.demo-sg{color:#e3e3e3;background:#1f1f1f;border-color:#444}"
                ".demo-sg a{background:#2c2c2c;border-color:#3c4043;color:#a8c7fa}}</style>"
                '<div class="demo-sg"><b>Search (demo)</b>'
                '<a href="https://www.google.com/search?q=how+does+an+S%26P+500+index+fund+work">how does an S&amp;P 500 '
                "index fund work</a>"
                '<a href="https://www.google.com/search?q=index+fund+expense+ratio">index fund expense ratio</a></div>'
            )},
        },
    }],
}
