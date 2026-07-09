from fastapi import APIRouter

from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import GraphTraverseRequest, GraphTraverseResponse, KnowledgeGraph
from services.entity_resolution import EntityResolver
from services.graph_service import GraphService

router = APIRouter()


@router.get("/graph/{corpus_id}", response_model=KnowledgeGraph)
async def get_corpus_graph(corpus_id: str):
    graph_svc = GraphService(SyncSessionLocal)
    return graph_svc.get_corpus_graph(corpus_id)


@router.get("/graph/entities/search")
async def search_entities(q: str, corpus_id: str | None = None, limit: int = 10):
    resolver = EntityResolver(SyncSessionLocal)
    return resolver.search_entities(q, corpus_id=corpus_id, limit=limit)


@router.post("/graph/traverse", response_model=GraphTraverseResponse)
async def traverse_graph(req: GraphTraverseRequest):
    graph_svc = GraphService(SyncSessionLocal)
    result = graph_svc.traverse(
        req.start_entity_query,
        max_hops=req.max_hops,
        corpus_id=req.corpus_id,
        document_id=req.document_id,
        predicate_filter=req.predicate_filter,
    )
    return GraphTraverseResponse(**result)
