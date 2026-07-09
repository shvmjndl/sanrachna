"""Cross-document entity resolution and canonical linking."""

from __future__ import annotations

import re
import uuid

from sqlalchemy import Column, Float, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.db.session import SyncBase
from app.schemas.knowledge import Entity


class CanonicalEntity(SyncBase):
    __tablename__ = "canonical_entities"

    id = Column(String(64), primary_key=True)
    canonical_text = Column(String(512), index=True, nullable=False)
    entity_type = Column(String(64), index=True, nullable=False)
    normalized = Column(String(512), index=True, nullable=False)
    variants = Column(JSONB, default=list)
    document_ids = Column(JSONB, default=list)
    corpus_id = Column(String(64), index=True, nullable=True)
    confidence = Column(Float, default=1.0)


def _normalize(text: str, entity_type: str) -> str:
    t = text.strip()
    if entity_type in ("GSTIN", "PAN", "IFSC", "CIN"):
        return re.sub(r"\s+", "", t).upper()
    return re.sub(r"\s+", " ", t).lower()


class EntityResolver:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def resolve(
        self,
        entities: list[Entity],
        document_id: str,
        corpus_id: str | None = None,
    ) -> list[Entity]:
        resolved: list[Entity] = []

        with self._session_factory() as session:
            for ent in entities:
                norm = _normalize(ent.normalized or ent.text, ent.type)
                existing = session.execute(
                    select(CanonicalEntity).where(
                        CanonicalEntity.normalized == norm,
                        CanonicalEntity.entity_type == ent.type,
                        *(
                            [CanonicalEntity.corpus_id == corpus_id]
                            if corpus_id
                            else []
                        ),
                    )
                ).scalar_one_or_none()

                if existing:
                    variants = existing.variants or []
                    if ent.text not in variants:
                        variants.append(ent.text)
                    doc_ids = existing.document_ids or []
                    if document_id not in doc_ids:
                        doc_ids.append(document_id)
                    existing.variants = variants
                    existing.document_ids = doc_ids
                    canonical_id = existing.id
                else:
                    canonical_id = f"cent_{uuid.uuid4().hex[:10]}"
                    session.add(
                        CanonicalEntity(
                            id=canonical_id,
                            canonical_text=ent.text,
                            entity_type=ent.type,
                            normalized=norm,
                            variants=[ent.text],
                            document_ids=[document_id],
                            corpus_id=corpus_id,
                            confidence=ent.confidence,
                        )
                    )

                resolved.append(
                    ent.model_copy(
                        update={
                            "id": canonical_id,
                            "canonical_id": canonical_id,
                            "normalized": norm,
                        }
                    )
                )

            session.commit()

        return resolved

    def search_entities(
        self,
        query: str,
        corpus_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        q = query.lower().strip()
        with self._session_factory() as session:
            stmt = select(CanonicalEntity).limit(limit * 3)
            if corpus_id:
                stmt = stmt.where(CanonicalEntity.corpus_id == corpus_id)
            all_ents = session.execute(stmt).scalars().all()

        matches = []
        for ent in all_ents:
            if (
                q in ent.normalized
                or q in ent.canonical_text.lower()
                or any(q in v.lower() for v in (ent.variants or []))
            ):
                matches.append(
                    {
                        "id": ent.id,
                        "type": ent.entity_type,
                        "text": ent.canonical_text,
                        "normalized": ent.normalized,
                        "variants": ent.variants,
                        "document_ids": ent.document_ids,
                    }
                )
            if len(matches) >= limit:
                break
        return matches

    def remove_document(self, document_id: str) -> int:
        """Remove document references from canonical entities; delete orphans."""
        removed = 0
        with self._session_factory() as session:
            entities = session.execute(select(CanonicalEntity)).scalars().all()
            for ent in entities:
                doc_ids = ent.document_ids or []
                if document_id not in doc_ids:
                    continue
                doc_ids = [d for d in doc_ids if d != document_id]
                if doc_ids:
                    ent.document_ids = doc_ids
                else:
                    session.delete(ent)
                    removed += 1
            session.commit()
        return removed
