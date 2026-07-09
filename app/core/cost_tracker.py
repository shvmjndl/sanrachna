"""Cost tracking for Sarvam pages and LiteLLM token usage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import SyncBase


class CostLog(SyncBase):
    __tablename__ = "cost_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    document_id: Mapped[str | None] = mapped_column(String(64), index=True)
    service: Mapped[str] = mapped_column(String(32), index=True)  # sarvam | litellm
    operation: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    pages: Mapped[int | None] = mapped_column(Integer)
    cost_inr: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class CostTracker:
    """Records per-operation costs to DB and structured logs."""

    def __init__(self, session_factory, logger):
        self._session_factory = session_factory
        self._log = logger

    def log_sarvam(
        self,
        *,
        job_id: str,
        document_id: str | None,
        pages: int,
        cost_per_page: float,
        operation: str = "document_digitization",
        extra: dict | None = None,
    ) -> float:
        cost_inr = pages * cost_per_page
        self._persist(
            job_id=job_id,
            document_id=document_id,
            service="sarvam",
            operation=operation,
            pages=pages,
            cost_inr=cost_inr,
            metadata=extra,
        )
        self._log.info(
            "sarvam_cost",
            job_id=job_id,
            document_id=document_id,
            pages=pages,
            cost_inr=cost_inr,
            operation=operation,
        )
        return cost_inr

    def log_litellm(
        self,
        *,
        job_id: str | None,
        document_id: str | None,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
        extra: dict | None = None,
    ) -> None:
        self._persist(
            job_id=job_id,
            document_id=document_id,
            service="litellm",
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=extra,
        )
        self._log.info(
            "litellm_cost",
            job_id=job_id,
            document_id=document_id,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def _persist(self, **kwargs) -> None:
        meta = kwargs.pop("metadata", None)
        with self._session_factory() as session:
            entry = CostLog(
                job_id=kwargs.get("job_id"),
                document_id=kwargs.get("document_id"),
                service=kwargs["service"],
                operation=kwargs["operation"],
                model=kwargs.get("model"),
                input_tokens=kwargs.get("input_tokens"),
                output_tokens=kwargs.get("output_tokens"),
                pages=kwargs.get("pages"),
                cost_inr=kwargs.get("cost_inr"),
                cost_usd=kwargs.get("cost_usd"),
                metadata_=meta,
            )
            session.add(entry)
            session.commit()
