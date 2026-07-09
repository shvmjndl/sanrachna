"""Cross-document family grouping via structural signal similarity."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import Document
from app.schemas.knowledge import DocumentFamilyMember, DocumentFamilyResponse, KnowledgeObject


def _signal_vector(ko: KnowledgeObject) -> dict[str, float]:
    family = ko.source.family_signals.get("family", {})
    quality = ko.source.family_signals.get("quality", {})
    return {
        "numeric_ratio": float(family.get("numeric_ratio", 0.0)),
        "record_chunks": float(family.get("record_chunks", 0.0)),
        "table_rows": float(family.get("table_rows", 0.0)),
        "bbox_ratio": float(family.get("bbox_ratio", 0.0)),
        "label_value_lines": float(family.get("label_value_lines", 0.0)),
        "avg_chunk_len": float(family.get("avg_chunk_len", 0.0)),
        "quality_score": float(ko.metadata.quality_score or quality.get("text_density", 0.5)),
    }


def _similarity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = 0.0
    den = 0.0
    for key in keys:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)
        num += min(av, bv)
        den += max(av, bv)
    return round(num / den if den else 0.0, 3)


class DocumentFamilyService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def find_family(
        self,
        *,
        document_id: str | None = None,
        corpus_id: str | None = None,
        min_similarity: float = 0.55,
        limit: int = 20,
    ) -> DocumentFamilyResponse:
        with self._session_factory() as session:
            stmt = select(Document)
            if corpus_id:
                stmt = stmt.where(Document.corpus_id == corpus_id)
            docs = session.execute(stmt).scalars().all()

        members: list[tuple[Document, KnowledgeObject, dict[str, float]]] = []
        for doc in docs:
            if not doc.knowledge_object:
                continue
            ko = KnowledgeObject(**doc.knowledge_object)
            members.append((doc, ko, _signal_vector(ko)))

        if not members:
            return DocumentFamilyResponse(family="unknown", members=[], signals={})

        anchor_doc = None
        anchor_ko = None
        anchor_vec: dict[str, float] = {}
        if document_id:
            for doc, ko, vec in members:
                if doc.id == document_id:
                    anchor_doc, anchor_ko, anchor_vec = doc, ko, vec
                    break
        if anchor_doc is None:
            anchor_doc, anchor_ko, anchor_vec = members[0]

        family_label = anchor_ko.source.document_family or "mixed_general"
        scored: list[DocumentFamilyMember] = []
        for doc, ko, vec in members:
            sim = 1.0 if doc.id == anchor_doc.id else _similarity(anchor_vec, vec)
            if doc.id != anchor_doc.id and sim < min_similarity:
                continue
            scored.append(
                DocumentFamilyMember(
                    document_id=doc.id,
                    filename=doc.filename,
                    corpus_id=doc.corpus_id,
                    document_family=ko.source.document_family,
                    similarity=sim,
                    shared_signals={
                        "quality_score": ko.metadata.quality_score,
                        **ko.source.family_signals.get("family", {}),
                    },
                )
            )

        scored.sort(key=lambda m: m.similarity, reverse=True)
        return DocumentFamilyResponse(
            anchor_document_id=anchor_doc.id,
            family=family_label,
            members=scored[:limit],
            signals={
                "anchor": anchor_vec,
                "family_label": family_label,
            },
        )
