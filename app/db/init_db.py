from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.cost_tracker import CostLog  # noqa: F401
from app.db import models  # noqa: F401
from app.db.session import Base, SyncBase
from services.bm25 import ChunkIndex  # noqa: F401
from services.entity_resolution import CanonicalEntity  # noqa: F401

settings = get_settings()

sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=sync_engine)


def init_db() -> None:
    Base.metadata.create_all(bind=sync_engine)
    SyncBase.metadata.create_all(bind=sync_engine)
    _ensure_chunk_index_columns()
    _ensure_document_columns()


def _ensure_document_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(sync_engine)
    if "documents" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("documents")}
    statements = []
    if "version" not in columns:
        statements.append("ALTER TABLE documents ADD COLUMN version INTEGER DEFAULT 1")
    if "content_hash" not in columns:
        statements.append("ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64)")
    if "supersedes_id" not in columns:
        statements.append("ALTER TABLE documents ADD COLUMN supersedes_id VARCHAR(64)")
    with sync_engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_chunk_index_columns() -> None:
    """Lightweight migration for additive BM25 columns."""
    from sqlalchemy import inspect, text

    inspector = inspect(sync_engine)
    if "chunk_index" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("chunk_index")}
    statements = []
    if "confidence" not in columns:
        statements.append("ALTER TABLE chunk_index ADD COLUMN confidence DOUBLE PRECISION DEFAULT 1.0")
    if "bbox" not in columns:
        statements.append("ALTER TABLE chunk_index ADD COLUMN bbox JSONB")

    with sync_engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
