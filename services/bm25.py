"""PostgreSQL BM25-style full-text search index."""

from __future__ import annotations

from sqlalchemy import Column, Float, Index, Integer, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Session

from app.db.session import SyncBase


class ChunkIndex(SyncBase):
    __tablename__ = "chunk_index"

    id = Column(String(64), primary_key=True)
    chunk_id = Column(String(64), index=True, nullable=False)
    document_id = Column(String(64), index=True, nullable=False)
    corpus_id = Column(String(64), index=True, nullable=True)
    text = Column(Text, nullable=False)
    parent_text = Column(Text, nullable=True)
    page = Column(Integer, nullable=True)
    section = Column(String(256), nullable=True)
    confidence = Column(Float, nullable=True, default=1.0)
    bbox = Column(JSONB, nullable=True)
    search_vector = Column(TSVECTOR)

    __table_args__ = (
        Index("ix_chunk_search_vector", "search_vector", postgresql_using="gin"),
    )


class BM25Indexer:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def index_chunks(
        self,
        chunks: list,
        document_id: str,
        corpus_id: str | None = None,
    ) -> int:
        with self._session_factory() as session:
            for chunk in chunks:
                entry = ChunkIndex(
                    id=f"idx_{chunk.id}",
                    chunk_id=chunk.id,
                    document_id=document_id,
                    corpus_id=corpus_id,
                    text=chunk.text,
                    parent_text=getattr(chunk, "parent_text", None),
                    page=chunk.page,
                    section=chunk.section,
                    confidence=getattr(chunk, "confidence", 1.0),
                    bbox=chunk.bbox.model_dump() if getattr(chunk, "bbox", None) else None,
                )
                session.merge(entry)

            session.execute(
                text("""
                    UPDATE chunk_index
                    SET search_vector = to_tsvector('simple', coalesce(text, '') || ' ' || coalesce(parent_text, ''))
                    WHERE document_id = :doc_id
                """),
                {"doc_id": document_id},
            )
            session.commit()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._session_factory() as session:
            result = session.execute(
                text("DELETE FROM chunk_index WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )
            session.commit()
            return result.rowcount or 0

    def search(
        self,
        query: str,
        top_k: int = 20,
        document_id: str | None = None,
        corpus_id: str | None = None,
    ) -> list[dict]:
        ts_query = func.plainto_tsquery("simple", query)

        with self._session_factory() as session:
            rank = func.ts_rank(ChunkIndex.search_vector, ts_query).label("score")
            stmt = (
                select(
                    ChunkIndex.chunk_id,
                    ChunkIndex.document_id,
                    ChunkIndex.text,
                    ChunkIndex.parent_text,
                    ChunkIndex.page,
                    ChunkIndex.section,
                    ChunkIndex.confidence,
                    ChunkIndex.bbox,
                    rank,
                )
                .where(ChunkIndex.search_vector.op("@@")(ts_query))
                .order_by(rank.desc())
                .limit(top_k)
            )

            if document_id:
                stmt = stmt.where(ChunkIndex.document_id == document_id)
            if corpus_id:
                stmt = stmt.where(ChunkIndex.corpus_id == corpus_id)

            rows = session.execute(stmt).all()

        return [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "parent_text": r.parent_text,
                "page": r.page,
                "section": r.section,
                "confidence": float(r.confidence or 1.0),
                "bbox": r.bbox,
                "score": float(r.score),
                "source": "bm25",
            }
            for r in rows
        ]
