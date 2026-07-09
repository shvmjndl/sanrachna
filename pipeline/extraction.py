"""Schema-on-read field extraction over retrieved evidence."""

from __future__ import annotations

import json
import time

from app.schemas.knowledge import (
    ExtractRequest,
    ExtractResponse,
    ExtractedField,
    FieldSpec,
    KnowledgeObject,
)
from pipeline.indexing import HybridIndexer
from services.llm import LLMService
from services.scope import ScopeResolver


class SchemaExtractor:
    def __init__(self, llm: LLMService, indexer: HybridIndexer, session_factory):
        self._llm = llm
        self._indexer = indexer
        self._scope = ScopeResolver(session_factory)
        self._session = session_factory

    async def extract(self, req: ExtractRequest) -> ExtractResponse:
        start = time.monotonic()
        scope = self._scope.resolve(document_id=req.document_id, corpus_id=req.corpus_id)
        document_id = scope.document_id

        query = " ".join(f"{f.name}: {f.description}" for f in req.fields)
        hits = await self._indexer.search(
            query=query,
            top_k=req.top_k,
            document_id=scope.document_id,
            corpus_id=scope.corpus_id,
            hybrid=True,
            rerank=True,
            expand_parent=False,
        )

        ko = None
        if document_id:
            ko = self._load_knowledge_object(document_id)

        evidence = self._build_evidence(hits, ko)
        fields = await self._extract_fields(req.fields, evidence, hits, req.include_sources)

        return ExtractResponse(
            document_id=document_id,
            fields=fields,
            processing_ms=int((time.monotonic() - start) * 1000),
        )

    def _load_knowledge_object(self, document_id: str) -> KnowledgeObject | None:
        from sqlalchemy import select

        from app.db.models import Document

        with self._session() as session:
            row = session.execute(select(Document).where(Document.id == document_id)).scalar_one_or_none()
            if not row or not row.knowledge_object:
                return None
            return KnowledgeObject(**row.knowledge_object)

    def _build_evidence(self, hits, ko: KnowledgeObject | None) -> str:
        parts: list[str] = []
        for hit in hits[:8]:
            parts.append(f"[{hit.chunk_id}] page={hit.page}\n{hit.text[:1200]}")

        if ko and ko.tables:
            for table in ko.tables[:5]:
                parts.append(
                    f"[table {table.id}] headers={table.headers}\n"
                    + "\n".join(" | ".join(row) for row in table.rows[:20])
                )
        return "\n\n".join(parts)

    async def _extract_fields(
        self,
        field_specs: list[FieldSpec],
        evidence: str,
        hits,
        include_sources: bool,
    ) -> list[ExtractedField]:
        schema_desc = json.dumps(
            [{"name": f.name, "description": f.description, "type": f.type} for f in field_specs],
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""Extract fields from the evidence. Return ONLY JSON:
{{
  "fields": [
    {{
      "name": "...",
      "value": ...,
      "confidence": 0.0-1.0,
      "source_chunk_ids": ["..."],
      "reasoning": "short reason"
    }}
  ]
}}

Rules:
- Use only values explicitly supported by the evidence.
- If a field is missing, set value to null and confidence <= 0.3.
- Match row labels to field descriptions structurally, not by hardcoded document type.
- Prefer structured table rows when available.

Fields to extract:
{schema_desc}

Evidence:
{evidence[:12000]}"""

        raw, _ = await self._llm.chat(
            messages=[
                {"role": "system", "content": "You extract structured data from document evidence."},
                {"role": "user", "content": prompt},
            ],
            operation="schema_extract",
            response_format={"type": "json_object"},
        )

        try:
            parsed = json.loads(raw)
            rows = parsed.get("fields", parsed if isinstance(parsed, list) else [])
        except json.JSONDecodeError:
            rows = []

        hit_pages = {h.chunk_id: h.page for h in hits}
        extracted: list[ExtractedField] = []
        seen = set()

        for row in rows:
            name = row.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            chunk_ids = row.get("source_chunk_ids") or []
            if not include_sources:
                chunk_ids = []
            extracted.append(
                ExtractedField(
                    name=name,
                    value=row.get("value"),
                    confidence=float(row.get("confidence", 0.0)),
                    source_chunk_ids=chunk_ids,
                    source_pages=sorted({hit_pages[c] for c in chunk_ids if c in hit_pages and hit_pages[c]}),
                    reasoning=row.get("reasoning"),
                )
            )

        for spec in field_specs:
            if spec.name not in seen:
                extracted.append(ExtractedField(name=spec.name, value=None, confidence=0.0))

        return extracted
