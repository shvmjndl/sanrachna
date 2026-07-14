from __future__ import annotations

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, UploadFile
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Document, IngestJob
from app.db.session import async_session
from app.schemas.knowledge import (
    IngestJobListResponse,
    IngestJobSummary,
    IngestResponse,
    JobCancelAllResponse,
    JobCancelResponse,
    JobStatus,
    JobStatusResponse,
)
from services.job_control import cancel_active_jobs, cancel_job_record
from worker.tasks import process_ingest_job

router = APIRouter()
settings = get_settings()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    language: str = Form(default="en-IN"),
    enable_entities: str = Form(default="true"),
    enable_graph: str = Form(default="true"),
    enable_captions: str = Form(default="true"),
    auto_detect_language: str = Form(default="true"),
    document_id: str = Form(default=""),
    supersedes_id: str = Form(default=""),
    incremental: str = Form(default="true"),
):
    job_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    options = {
        "enable_entities": enable_entities.lower() == "true",
        "enable_graph": enable_graph.lower() == "true",
        "enable_captions": enable_captions.lower() == "true",
        "auto_detect_language": auto_detect_language.lower() == "true",
        "incremental": incremental.lower() == "true",
    }
    if document_id:
        options["document_id"] = document_id
    if supersedes_id:
        options["supersedes_id"] = supersedes_id

    async with async_session() as session:
        job = IngestJob(
            id=uuid.UUID(job_id),
            status=JobStatus.PENDING.value,
            filename=file.filename,
            file_path=str(dest),
            language=language,
            options=options,
            corpus_id=f"corp_{job_id[:12]}",
        )
        session.add(job)
        await session.commit()

    process_ingest_job.delay(job_id)

    return IngestResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Document queued for processing",
    )


@router.get("/jobs", response_model=IngestJobListResponse)
async def list_jobs(
    limit: int = 50,
    status: str | None = None,
):
    async with async_session() as session:
        query = select(IngestJob).order_by(IngestJob.created_at.desc()).limit(min(limit, 200))
        if status:
            query = query.where(IngestJob.status == status)
        result = await session.execute(query)
        jobs = result.scalars().all()

    summaries = [
        IngestJobSummary(
            job_id=str(j.id),
            status=JobStatus(j.status),
            filename=j.filename,
            progress=j.progress,
            documents_total=j.documents_total,
            documents_processed=j.documents_processed,
            error=j.error,
            corpus_id=j.corpus_id,
            created_at=j.created_at,
        )
        for j in jobs
    ]
    active = sum(
        1 for j in summaries if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    return IngestJobListResponse(jobs=summaries, total=len(summaries), active_count=active)


@router.post("/jobs/cancel-active", response_model=JobCancelAllResponse)
async def cancel_active():
    from app.db.init_db import SyncSessionLocal

    with SyncSessionLocal() as session:
        result = cancel_active_jobs(session)
    return JobCancelAllResponse(**result)


@router.post("/jobs/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(job_id: str):
    from app.db.init_db import SyncSessionLocal

    with SyncSessionLocal() as session:
        result = cancel_job_record(session, job_id)
    return JobCancelResponse(**result)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    from app.schemas.knowledge import CorpusSummary

    async with async_session() as session:
        result = await session.execute(select(IngestJob).where(IngestJob.id == uuid.UUID(job_id)))
        job = result.scalar_one_or_none()

        if not job:
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.FAILED,
                error="Job not found",
            )

        summary = None
        if job.status == JobStatus.COMPLETED.value:
            doc_rows = await session.execute(
                select(Document.id).where(Document.corpus_id == job.corpus_id)
            )
            document_ids = [row[0] for row in doc_rows.all()]

            summary = CorpusSummary(
                corpus_id=job.corpus_id or "",
                documents=job.documents_processed,
                document_ids=document_ids,
                status=JobStatus.COMPLETED,
                created_at=job.created_at,
            )

        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(job.status),
            progress=job.progress,
            documents_total=job.documents_total,
            documents_processed=job.documents_processed,
            error=job.error,
            result=summary,
        )
