"""FastAPI application: GET /answer (competition contract), GET /health.

Guarantees: always HTTP 200 with the five string fields; unknown query params ignored.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .agent.answer import fallback_answer
from .config import get_settings
from .models import AnswerResponse

log = logging.getLogger("finagent.api")


def _json(resp: AnswerResponse) -> JSONResponse:
    return JSONResponse(
        content=json.loads(json.dumps(resp.model_dump(), ensure_ascii=False)),
        media_type="application/json; charset=utf-8",
    )


def create_app(state_factory: Callable | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state_factory is not None:
            app.state.fin = state_factory()
        else:
            from .app_state import AppState

            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
            app.state.fin = AppState.load(get_settings())
        yield

    app = FastAPI(title="금융상품 Agent API", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=PlainTextResponse)
    async def root():
        return "금융상품 Agent API — GET /answer?question_id=Q-001&question=..."

    @app.get("/health")
    async def health():
        st = getattr(app.state, "fin", None)
        if st is None:
            return {"status": "starting"}
        return {
            "status": "ok",
            "tables": st.rdb.meta(),
            "index_docs": len(st.index.docs),
            "kg_triples": st.kg.size if st.kg is not None else 0,
            "docs_chunks": len(st.docs),
            "llm": f"{st.llm.name}/{st.settings.llm_model}",
        }

    @app.get("/answer")
    async def answer(request: Request):
        qp = request.query_params
        question_id = (qp.get("question_id") or "").strip()
        question = (qp.get("question") or "").strip()
        if not question:
            return _json(
                AnswerResponse(
                    question_id=question_id, question=question, retrieved_context="",
                    think_trace="입력 검증: question 파라미터 누락",
                    answer="question 파라미터가 필요합니다. 예: /answer?question_id=Q-001&question=질의문",
                )
            )
        t0 = time.time()
        try:
            resp = await app.state.fin.orchestrator.answer(question_id, question)
        except Exception as e:  # noqa: BLE001 - never 500
            log.exception("answer failed for %s", question_id)
            resp = AnswerResponse(
                question_id=question_id, question=question, retrieved_context="",
                think_trace=f"내부 오류: {type(e).__name__}: {str(e)[:200]}",
                answer=fallback_answer("internal_error"),
            )
        log.info("answered %s in %.1fs: %s", question_id, time.time() - t0, resp.answer[:80].replace("\n", " "))
        return _json(resp)

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run("finagent.api:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
