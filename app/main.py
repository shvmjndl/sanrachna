from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    compare,
    contradictions,
    costs,
    documents,
    eval,
    extract,
    families,
    graph,
    health,
    ingest,
    query,
    search,
    templates,
    translate,
    webhooks,
)
from app.core.logging import setup_logging
from app.db.init_db import init_db
from app.db.session import async_session, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title="Sanrachna",
    description="Indic-first document-to-knowledge API",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/v1", tags=["ingest"])
app.include_router(documents.router, prefix="/v1", tags=["documents"])
app.include_router(search.router, prefix="/v1", tags=["search"])
app.include_router(graph.router, prefix="/v1", tags=["graph"])
app.include_router(query.router, prefix="/v1", tags=["agent"])
app.include_router(extract.router, prefix="/v1", tags=["extract"])
app.include_router(compare.router, prefix="/v1", tags=["compare"])
app.include_router(contradictions.router, prefix="/v1", tags=["contradictions"])
app.include_router(templates.router, prefix="/v1", tags=["templates"])
app.include_router(translate.router, prefix="/v1", tags=["translate"])
app.include_router(webhooks.router, prefix="/v1", tags=["webhooks"])
app.include_router(eval.router, prefix="/v1", tags=["eval"])
app.include_router(families.router, prefix="/v1", tags=["families"])
app.include_router(costs.router, prefix="/v1", tags=["costs"])


async def get_db_session():
    async with async_session() as session:
        yield session
