"""Document versioning for incremental re-ingest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import select

from app.db.models import Document


def file_content_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_versioning(session, file_path: Path, options: dict) -> tuple[str | None, int, str | None]:
    """Return (document_id, version, supersedes_id) for incremental ingest."""
    explicit_id = options.get("document_id")
    supersedes_id = options.get("supersedes_id")

    if supersedes_id:
        old = session.execute(select(Document).where(Document.id == supersedes_id)).scalar_one_or_none()
        if old:
            new_version = (old.version or 1) + 1
            return supersedes_id, new_version, supersedes_id

    content_hash = file_content_hash(file_path)
    existing = session.execute(
        select(Document).where(Document.content_hash == content_hash).order_by(Document.created_at.desc())
    ).scalar_one_or_none()
    if existing and options.get("incremental", True):
        return existing.id, (existing.version or 1) + 1, existing.id

    return explicit_id, 1, None
