from fastapi import APIRouter, Query

from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import DocumentFamilyResponse
from services.families import DocumentFamilyService

router = APIRouter()


@router.get("/families", response_model=DocumentFamilyResponse)
async def get_document_family(
    document_id: str | None = None,
    corpus_id: str | None = None,
    min_similarity: float = Query(default=0.55, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
):
    service = DocumentFamilyService(SyncSessionLocal)
    return service.find_family(
        document_id=document_id,
        corpus_id=corpus_id,
        min_similarity=min_similarity,
        limit=limit,
    )
