"""Resolve query/search scope from optional hints and indexed documents."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import desc, func, select

from app.db.models import Document


@dataclass
class ResolvedScope:
    document_id: str | None = None
    corpus_id: str | None = None
    mode: str = "global"


class ScopeResolver:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def resolve(
        self,
        document_id: str | None = None,
        corpus_id: str | None = None,
    ) -> ResolvedScope:
        with self._session_factory() as session:
            if document_id:
                doc = session.execute(
                    select(Document).where(Document.id == document_id)
                ).scalar_one_or_none()
                if doc:
                    return ResolvedScope(
                        document_id=doc.id,
                        corpus_id=doc.corpus_id,
                        mode="document",
                    )

            if corpus_id:
                docs = session.execute(
                    select(Document)
                    .where(Document.corpus_id == corpus_id)
                    .order_by(desc(Document.created_at))
                ).scalars().all()
                if docs:
                    if len(docs) == 1:
                        return ResolvedScope(
                            document_id=docs[0].id,
                            corpus_id=corpus_id,
                            mode="document",
                        )
                    return ResolvedScope(corpus_id=corpus_id, mode="corpus")

            total = session.execute(select(func.count(Document.id))).scalar_one() or 0
            if total == 1:
                doc = session.execute(select(Document)).scalar_one()
                return ResolvedScope(
                    document_id=doc.id,
                    corpus_id=doc.corpus_id,
                    mode="single_corpus",
                )

            latest = session.execute(
                select(Document).order_by(desc(Document.created_at)).limit(1)
            ).scalar_one_or_none()
            if latest and total > 1:
                return ResolvedScope(
                    document_id=latest.id,
                    corpus_id=latest.corpus_id,
                    mode="latest",
                )

            return ResolvedScope(mode="global")
