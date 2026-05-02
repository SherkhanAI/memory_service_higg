from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import health, recall, search, sessions, turns, users
from .config import settings
from .db import close_pool, init_pool, run_migrations
from .services.openrouter import close_client

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    await run_migrations()
    log.info(
        "memory-service ready: embed=%s/%s extract=%s/%s rerank=%s/%s",
        settings.embedding_provider, settings.embedding_model,
        settings.extraction_provider, settings.extraction_model,
        settings.reranker_provider, settings.reranker_model,
    )
    try:
        yield
    finally:
        await close_client()
        await close_pool()


app = FastAPI(
    title="memory-service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(turns.router)
app.include_router(recall.router)
app.include_router(search.router)
app.include_router(users.router)
app.include_router(sessions.router)


@app.exception_handler(RequestValidationError)
async def _on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # 422 by default; ТЗ требует «не падать» на malformed input — это и так 4xx, не 5xx.
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def _on_unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )
