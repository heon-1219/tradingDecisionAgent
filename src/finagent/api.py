"""FastAPI application: GET /answer (competition contract), GET /health, and the chat app.

Guarantees for /answer: always HTTP 200 with the five string fields; unknown query params ignored.

The paper-trading chat app lives under `/api/*` (`api_chat.router`), guarded by
`api_security.LocalApiGuard` (loopback clients, a Host allowlist, anti-CSRF header). Its web UI is
served at `/` and `/app/*` from `web/app/` once `web/app/index.html` exists; until then `/` keeps
its plain-text line. `/answer`, `/answer/stream`, `/health` and `/ui` are untouched by the guard.
`main()` binds 127.0.0.1 unless `FINAGENT_HOST` says otherwise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import api_chat
from .agent.answer import fallback_answer
from .api_security import LocalApiGuard
from .config import get_settings
from .i18n import detect_lang
from .models import AnswerResponse
from .paths import ROOT

log = logging.getLogger("finagent.api")

WEB_APP_DIR = ROOT / "web" / "app"
ROOT_TEXT = "Financial Product Agent API — GET /answer?question_id=Q-001&question=...  ·  test console: /ui"


def _json(resp: AnswerResponse) -> JSONResponse:
    return JSONResponse(
        content=json.loads(json.dumps(resp.model_dump(), ensure_ascii=False)),
        media_type="application/json; charset=utf-8",
    )


def create_app(state_factory: Callable | None = None, *, web_app_dir: Path | None = None) -> FastAPI:
    web_dir = Path(web_app_dir) if web_app_dir is not None else WEB_APP_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state_factory is not None:
            app.state.fin = state_factory()
        else:
            from .app_state import AppState

            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
            app.state.fin = AppState.load(get_settings())
        chat = getattr(app.state.fin, "chat", None)
        if chat is not None:
            try:
                await chat.startup()
            except Exception as e:  # noqa: BLE001 - /answer must start regardless
                log.warning("chat runtime startup failed (%s)", type(e).__name__)
        try:
            yield
        finally:
            if chat is not None:
                await chat.shutdown()

    app = FastAPI(title="Financial Product Agent API", version="0.1.0", lifespan=lifespan)

    def allow_remote() -> bool:
        settings = getattr(getattr(app.state, "fin", None), "settings", None)
        return bool(getattr(settings, "api_allow_remote", False))

    app.add_middleware(LocalApiGuard, allow_remote=allow_remote)
    app.include_router(api_chat.router)
    if (web_dir / "index.html").is_file():
        app.mount("/app", StaticFiles(directory=web_dir), name="webapp")

    @app.get("/")
    async def root():
        index = web_dir / "index.html"
        if index.is_file():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return PlainTextResponse(ROOT_TEXT)

    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        """Local test console.  Served from the API itself so the page and /answer share an origin —
        opening the file directly would put it on file:// and the browser would block the call."""
        page = Path(__file__).resolve().parents[2] / "web" / "console.html"
        if not page.exists():
            return HTMLResponse("<p>web/console.html is missing.</p>", status_code=404)
        return HTMLResponse(page.read_text(encoding="utf-8"))

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
            "kg_asserted": getattr(st.kg, "asserted_triples", 0) if st.kg is not None else 0,
            "kg_inferred": getattr(st.kg, "inferred_triples", 0) if st.kg is not None else 0,
            "vector": (
                {"backend": st.vectors.stats.backend, "docs": st.vectors.stats.docs, "dims": st.vectors.stats.dims}
                if getattr(st, "vectors", None) is not None else None
            ),
            "docs_chunks": len(st.docs),
            "llm": f"{st.llm.name}/{st.settings.llm_model}",
        }

    @app.get("/answer")
    async def answer(request: Request):
        qp = request.query_params
        question_id = (qp.get("question_id") or "").strip()
        question = (qp.get("question") or "").strip()
        if not question:
            # No question, no language to detect: the reply is English (the English edition's default).
            return _json(
                AnswerResponse(
                    question_id=question_id, question=question, retrieved_context="",
                    think_trace="Input validation: question parameter missing",
                    answer="The question parameter is required. E.g. /answer?question_id=Q-001&question=your+question",
                )
            )
        t0 = time.time()
        try:
            resp = await app.state.fin.orchestrator.answer(question_id, question)
        except Exception as e:  # noqa: BLE001 - never 500
            log.exception("answer failed for %s", question_id)
            resp = AnswerResponse(
                question_id=question_id, question=question, retrieved_context="",
                think_trace=f"Internal error: {type(e).__name__}: {str(e)[:200]}",
                answer=fallback_answer("internal_error", detect_lang(question)),
            )
        log.info("answered %s in %.1fs: %s", question_id, time.time() - t0, resp.answer[:80].replace("\n", " "))
        return _json(resp)

    @app.get("/answer/stream")
    async def answer_stream(request: Request):
        """Same answer, streamed step by step as Server-Sent Events — development console only.

        Deliberately a separate route: the graded ``/answer`` contract (HTTP 200, five string fields)
        must not grow a second code path.  The event sink only observes; it never reaches the model.
        """
        qp = request.query_params
        question_id = (qp.get("question_id") or "T-001").strip()
        question = (qp.get("question") or "").strip()

        async def gen():
            if not question:
                yield _sse({"type": "error", "message": "The question parameter is required."})
                return
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def sink(ev: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            task = asyncio.create_task(
                app.state.fin.orchestrator.answer(question_id, question, on_event=sink)
            )
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # keeps proxies and the browser from closing an idle stream
                    continue
                yield _sse(ev)
            try:
                await task
            except Exception as e:  # noqa: BLE001 - report, never 500
                log.exception("stream answer failed for %s", question_id)
                yield _sse({"type": "error", "message": f"{type(e).__name__}: {str(e)[:200]}"})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


app = create_app()


def main():
    import uvicorn

    # Loopback by default: /api/* can place paper orders. Set FINAGENT_HOST=0.0.0.0 to expose
    # /answer; the /api/* guard still refuses non-loopback clients (see api_security).
    # proxy_headers=False: the guard's client address must be the socket peer, never a value an
    # X-Forwarded-For header supplied.
    host = os.environ.get("FINAGENT_HOST", "127.0.0.1")
    uvicorn.run("finagent.api:app", host=host, port=8000, workers=1, proxy_headers=False)


if __name__ == "__main__":
    main()
