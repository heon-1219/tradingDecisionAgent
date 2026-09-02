"""Builds every runtime component once (startup) and hands them to the API."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .agent.llm import LLMClient, make_llm
from .agent.orchestrator import Orchestrator
from .agent.prompts import build_system_prompt
from .agent.tools import ToolRegistry
from .config import Settings
from .kb.docs import DocStore
from .kb.entity import EntityResolver
from .kb.graph import KG
from .kb.lexicon import Lexicon
from .kb.rdb import RDB
from .kb.search import SearchIndex
from .ontology.loader import FallbackValidators, Ontology

log = logging.getLogger("finagent.app")


@dataclass
class AppState:
    settings: Settings
    rdb: RDB
    index: SearchIndex
    resolver: EntityResolver
    kg: KG | None
    docs: DocStore
    ontology: Ontology | None
    lexicon: Lexicon
    llm: LLMClient
    tools: ToolRegistry
    orchestrator: Orchestrator
    system_prompt: str

    @classmethod
    def load(cls, settings: Settings, llm: LLMClient | None = None, build_graph: bool = True) -> "AppState":
        t0 = time.time()
        rdb = RDB(settings.store_path, row_limit=settings.sql_row_limit, timeout_sec=settings.sql_timeout_sec)
        lexicon = Lexicon.load(settings.enrichment_dir)
        ontology: Ontology | None = None
        try:
            ontology = Ontology.load(settings.ontology_dir)
            log.info("ontology: %d triples", len(ontology.graph))
        except Exception as e:  # noqa: BLE001
            log.warning("ontology not loaded (%s); using fallback validators", e)
            ontology = FallbackValidators()  # type: ignore[assignment]
        index = SearchIndex.from_rdb(rdb)
        log.info("index: %d docs", len(index.docs))
        resolver = EntityResolver(index, lexicon)
        kg: KG | None = None
        if build_graph:
            kg = KG.build(rdb, lexicon, ontology, settings.constituents_dir, log=log.info)
        docs = DocStore.load(settings.docs_dir)
        llm = llm or make_llm(settings)
        tools = ToolRegistry(rdb, index, resolver, kg, docs, ontology, lexicon, settings)
        system_prompt = build_system_prompt(ontology, rdb.meta(), docs.titles())
        orch = Orchestrator(settings, llm, tools, resolver, ontology, lexicon, system_prompt)
        log.info("app state ready in %.1fs (llm=%s/%s)", time.time() - t0, llm.name, settings.llm_model)
        return cls(settings, rdb, index, resolver, kg, docs, ontology, lexicon, llm, tools, orch, system_prompt)
