from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.init_db import SyncSessionLocal
from app.db.models import Document
from app.db.session import async_session
from app.schemas.knowledge import HighlightItem, HighlightResponse, KnowledgeObject
from services.document_store import DocumentStore

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)


@router.get("/documents/{document_id}", response_model=KnowledgeObject)
async def get_document(document_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()

    if not doc or not doc.knowledge_object:
        raise HTTPException(status_code=404, detail="Document not found")

    return KnowledgeObject(**doc.knowledge_object)


@router.get("/documents")
async def list_documents(corpus_id: str | None = None, limit: int = 50):
    async with async_session() as session:
        query = select(Document).limit(limit)
        if corpus_id:
            query = query.where(Document.corpus_id == corpus_id)
        result = await session.execute(query)
        docs = result.scalars().all()

    return [
        {
            "document_id": d.id,
            "filename": d.filename,
            "status": d.status,
            "language": d.language,
            "corpus_id": d.corpus_id,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.get("/documents/{document_id}/highlights", response_model=HighlightResponse)
async def get_document_highlights(
    document_id: str,
    chunk_ids: str = Query(..., description="Comma-separated chunk IDs"),
):
    async with async_session() as session:
        result = await session.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()

    if not doc or not doc.knowledge_object:
        raise HTTPException(status_code=404, detail="Document not found")

    ko = KnowledgeObject(**doc.knowledge_object)
    wanted = {cid.strip() for cid in chunk_ids.split(",") if cid.strip()}
    chunk_map = {c.id: c for c in ko.chunks}

    highlights: list[HighlightItem] = []
    for cid in wanted:
        chunk = chunk_map.get(cid)
        if not chunk:
            continue
        highlights.append(
            HighlightItem(
                chunk_id=cid,
                page=chunk.page,
                text=chunk.text[:1000],
                bbox=chunk.bbox,
            )
        )

    return HighlightResponse(
        document_id=document_id,
        highlights=highlights,
        file_path=doc.file_path,
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    store = DocumentStore(settings, SyncSessionLocal)
    result = await store.delete_document(document_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="Document not found")
    return result
