from fastapi import APIRouter

from app.core.config import get_settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import SearchRequest, SearchResponse
from pipeline.indexing import HybridIndexer
from services.confidence import assess_evidence
from services.graph_service import GraphService
from services.llm import LLMService
from services.multilingual import MultilingualQueryService
from services.scope import ScopeResolver

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)


@router.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    cost_tracker = CostTracker(SyncSessionLocal, log, buffer_writes=False)
    llm = LLMService(settings, cost_tracker)
    indexer = HybridIndexer(settings, llm, SyncSessionLocal, cost_tracker)
    graph_svc = GraphService(SyncSessionLocal)

    scope = ScopeResolver(SyncSessionLocal).resolve(
        document_id=req.document_id,
        corpus_id=req.corpus_id,
    )

    lang = MultilingualQueryService(settings)
    lang_meta = lang.preprocess(req.query)
    search_query = lang_meta["search_query"]

    results = await indexer.search(
        query=search_query,
        top_k=req.top_k,
        document_id=scope.document_id,
        corpus_id=scope.corpus_id,
        hybrid=req.hybrid,
        rerank=req.rerank,
        expand_parent=req.expand_parent,
    )

    entities, _, graph = graph_svc.lookup_entities(
        search_query, corpus_id=scope.corpus_id, document_id=scope.document_id
    )
    graph_context = GraphService.format_graph_context(entities, graph) if entities else None

    evidence = assess_evidence(results) if results else None

    return SearchResponse(
        query=req.query,
        results=results,
        total=len(results),
        graph_context=graph_context,
        evidence=evidence,
    )
