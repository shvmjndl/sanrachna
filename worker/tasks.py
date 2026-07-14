"""Celery tasks for async document processing."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger, setup_logging
from app.db.init_db import SyncSessionLocal, init_db
from app.db.models import Document, IngestJob
from app.schemas.knowledge import JobStatus
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.router import collect_files
from services.document_store import DocumentStore
from services.versioning import file_content_hash, resolve_versioning
from services.job_control import is_cancelled
from services.webhooks import WebhookService
from worker.celery_app import celery_app

log = get_logger(__name__)
settings = get_settings()


@celery_app.task(bind=True, name="worker.process_ingest_job", max_retries=2)
def process_ingest_job(self, job_id: str) -> dict:
    setup_logging()
    init_db()

    with SyncSessionLocal() as session:
        job = session.execute(
            select(IngestJob).where(IngestJob.id == uuid.UUID(job_id))
        ).scalar_one_or_none()

        if not job:
            return {"error": "Job not found"}

        if is_cancelled(job):
            return {"cancelled": True}

        job.status = JobStatus.PROCESSING.value
        session.commit()

        try:
            result = asyncio.run(_process_job(job, session))
            session.refresh(job)
            if is_cancelled(job):
                return {"cancelled": True}
            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.documents_processed = result["processed"]
            job.documents_total = result["total"]
            session.commit()
            return result
        except Exception as exc:
            log.exception("job_failed", job_id=job_id, error=str(exc))
            job.status = JobStatus.FAILED.value
            job.error = str(exc)
            session.commit()
            raise


async def _process_job(job: IngestJob, session) -> dict:
    cost_tracker = CostTracker(SyncSessionLocal, log)
    orchestrator = PipelineOrchestrator(settings, cost_tracker, SyncSessionLocal)

    try:
        root = Path(job.file_path)
        files = collect_files(root)
        job.documents_total = len(files)
        session.commit()

        processed = 0
        document_ids: list[str] = []
        store = DocumentStore(settings, SyncSessionLocal)
        for i, file_path in enumerate(files):
            session.refresh(job)
            if is_cancelled(job):
                return {
                    "total": len(files),
                    "processed": processed,
                    "corpus_id": job.corpus_id,
                    "document_ids": document_ids,
                    "cancelled": True,
                }
            options = job.options or {}
            doc_id, version, supersedes_id = resolve_versioning(session, file_path, options)
            if supersedes_id and doc_id:
                await store.delete_document(supersedes_id)

            ko = await orchestrator.process_file(
                file_path,
                language=job.language,
                job_id=str(job.id),
                corpus_id=job.corpus_id,
                options=options,
                document_id=doc_id,
                supersedes_id=supersedes_id,
            )

            content_hash = file_content_hash(file_path)
            doc = Document(
                id=ko.document_id,
                job_id=str(job.id),
                corpus_id=job.corpus_id,
                filename=file_path.name,
                file_path=str(file_path),
                mime_type=ko.source.mime_type,
                language=job.language,
                status="completed",
                knowledge_object=ko.model_dump(mode="json"),
                version=version,
                content_hash=content_hash,
                supersedes_id=supersedes_id if supersedes_id != ko.document_id else None,
            )
            session.merge(doc)
            document_ids.append(ko.document_id)
            processed += 1
            job.documents_processed = processed
            job.progress = round((processed / max(len(files), 1)) * 100, 1)
            session.commit()

            log.info(
                "document_processed",
                job_id=str(job.id),
                document_id=ko.document_id,
                chunks=len(ko.chunks),
                entities=len(ko.entities),
            )

        result = {
            "total": len(files),
            "processed": processed,
            "corpus_id": job.corpus_id,
            "document_ids": document_ids,
        }
        WebhookService(SyncSessionLocal).emit(
            "ingest.completed",
            {"job_id": str(job.id), "corpus_id": job.corpus_id, "document_ids": document_ids},
        )
        return result
    finally:
        await orchestrator.aclose()
