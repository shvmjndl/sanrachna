"""Document lifecycle operations — delete from all indexes and storage."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, select

from app.core.config import Settings
from app.db.models import Document
from app.db.session import async_session
from pipeline.indexing import HybridIndexer
from services.bm25 import BM25Indexer
from services.entity_resolution import EntityResolver
from services.llm import LLMService


class DocumentStore:
    def __init__(self, settings: Settings, sync_session_factory):
        self._settings = settings
        self._sync_session_factory = sync_session_factory
        self._bm25 = BM25Indexer(sync_session_factory)
        self._entities = EntityResolver(sync_session_factory)
        self._indexer = HybridIndexer(
            settings,
            LLMService(settings, None),
            sync_session_factory,
            None,
        )

    async def delete_document(self, document_id: str) -> dict:
        async with async_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if not doc:
                return {"deleted": False, "document_id": document_id}

            file_path = doc.file_path
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.commit()

        bm25_deleted = self._bm25.delete_document(document_id)
        self._indexer.delete_document(document_id)
        entities_updated = self._entities.remove_document(document_id)
        file_deleted = self._delete_file(file_path)

        return {
            "deleted": True,
            "document_id": document_id,
            "bm25_chunks_removed": bm25_deleted,
            "vector_index_cleared": True,
            "canonical_entities_updated": entities_updated,
            "file_deleted": file_deleted,
        }

    def _delete_file(self, file_path: str | None) -> bool:
        if not file_path:
            return False
        path = Path(file_path)
        if not path.exists():
            return False
        try:
            path.unlink()
            parent = path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            return True
        except OSError:
            return False
