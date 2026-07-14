"""Ingest job queue control — list, cancel, purge pending tasks."""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import IngestJob
from app.schemas.knowledge import JobStatus

CANCELLED_MSG = "Cancelled by user"


def is_cancelled(job: IngestJob) -> bool:
    return job.status == JobStatus.FAILED.value and (job.error or "").startswith("Cancelled")


def remove_job_from_queue(job_id: str) -> int:
    """Remove pending Celery tasks for job_id from the broker queue."""
    settings = get_settings()
    client = redis.from_url(settings.celery_broker_url)
    removed = 0
    try:
        raw_items = client.lrange("celery", 0, -1)
        for raw in raw_items:
            try:
                payload = json.loads(raw)
                body = base64.b64decode(payload.get("body", "")).decode("utf-8", errors="replace")
                if job_id in body:
                    client.lrem("celery", 1, raw)
                    removed += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    finally:
        client.close()
    return removed


def cancel_job_record(session: Session, job_id: str) -> dict[str, Any]:
    """Mark a job cancelled and drop it from the Redis queue if still pending."""
    job = session.execute(
        select(IngestJob).where(IngestJob.id == uuid.UUID(job_id))
    ).scalar_one_or_none()
    if not job:
        return {"job_id": job_id, "cancelled": False, "message": "Job not found", "removed_from_queue": 0}

    if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
        if is_cancelled(job):
            return {
                "job_id": job_id,
                "cancelled": True,
                "message": "Already cancelled",
                "removed_from_queue": 0,
            }
        return {
            "job_id": job_id,
            "cancelled": False,
            "message": f"Job already {job.status}",
            "removed_from_queue": 0,
        }

    removed = 0
    if job.status == JobStatus.PENDING.value:
        removed = remove_job_from_queue(job_id)

    job.status = JobStatus.FAILED.value
    job.error = CANCELLED_MSG
    session.commit()

    note = "removed from queue" if removed else "marked cancelled (worker may finish current step)"
    return {
        "job_id": job_id,
        "cancelled": True,
        "message": note,
        "removed_from_queue": removed,
    }


def cancel_active_jobs(session: Session) -> dict[str, Any]:
    """Cancel every pending/processing ingest job."""
    rows = session.execute(
        select(IngestJob).where(
            IngestJob.status.in_([JobStatus.PENDING.value, JobStatus.PROCESSING.value])
        )
    ).scalars().all()

    cancelled: list[str] = []
    removed_total = 0
    for job in rows:
        result = cancel_job_record(session, str(job.id))
        if result["cancelled"]:
            cancelled.append(str(job.id))
            removed_total += int(result.get("removed_from_queue") or 0)

    return {
        "cancelled_count": len(cancelled),
        "job_ids": cancelled,
        "removed_from_queue": removed_total,
    }
