from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import AgentQueryRequest, AgentQueryResponse
from pipeline.indexing import HybridIndexer
from services.agent import DocumentAgent
from services.entity_resolution import EntityResolver
from services.graph_service import GraphService
from services.llm import LLMService

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)


def _build_agent():
    cost_tracker = CostTracker(SyncSessionLocal, log)
    llm = LLMService(settings, cost_tracker)
    indexer = HybridIndexer(settings, llm, SyncSessionLocal, cost_tracker)
    graph_svc = GraphService(SyncSessionLocal)
    entity_resolver = EntityResolver(SyncSessionLocal)
    agent = DocumentAgent(
        settings=settings,
        llm=llm,
        indexer=indexer,
        graph=graph_svc,
        entity_resolver=entity_resolver,
        session_factory=SyncSessionLocal,
    )
    return agent


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(req: AgentQueryRequest):
    agent = _build_agent()
    return await agent.run(
        query=req.query,
        document_id=req.document_id,
        corpus_id=req.corpus_id,
        top_k=req.top_k,
        max_steps=req.max_steps,
    )


@router.post("/query/stream")
async def agent_query_stream(req: AgentQueryRequest):
    agent = _build_agent()

    async def event_generator():
        async for event in agent.run_stream(
            query=req.query,
            document_id=req.document_id,
            corpus_id=req.corpus_id,
            top_k=req.top_k,
            max_steps=req.max_steps,
        ):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
