"""India-specific entity extraction via regex + LiteLLM."""

from __future__ import annotations

import asyncio
import json
import re
import uuid

from app.schemas.knowledge import Entity
from services.llm import LLMService

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("EMAIL", re.compile(r"\b[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}\b")),
    ("UPI_ID", re.compile(r"\b[\w.\-]+@(?:ybl|oksbi|paytm|axl|ibl|upi)\b", re.I)),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("CIN", re.compile(r"\b[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")),
    (
        "MONEY",
        re.compile(
            r"(?:₹|Rs\.?|INR|\$|USD|EUR|GBP|£|€)\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{1,2})?\s*(?:₹|Rs\.?|INR|USD|EUR|GBP)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "REFERENCE_ID",
        re.compile(
            r"\b(?:ref(?:erence)?|id|code|no|number)\s*[:#]?\s*([A-Z0-9][\w\-/.]{2,})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "INDIAN_DATE",
        re.compile(
            r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)[a-zA-Z\u0900-\u097F]*\s+\d{2,4})\b"
        ),
    ),
    ("PINCODE", re.compile(r"\b[1-9]\d{5}\b")),
]


def extract_regex_entities(text: str, source_chunk: str | None = None) -> list[Entity]:
    entities: list[Entity] = []
    seen: set[str] = set()
    for entity_type, pattern in PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            key = f"{entity_type}:{value}"
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    id=f"ent_{uuid.uuid4().hex[:8]}",
                    type=entity_type,
                    text=value.strip(),
                    normalized=value.strip().upper() if entity_type in ("GSTIN", "PAN", "IFSC", "CIN") else value.strip(),
                    confidence=0.95,
                    source_chunk=source_chunk,
                )
            )
    return entities


async def extract_llm_entities(
    llm: LLMService,
    text: str,
    language: str,
    source_chunk: str | None = None,
    job_id: str | None = None,
    document_id: str | None = None,
) -> list[Entity]:
    prompt = f"""Extract named entities from this {language} document text.
Return JSON array with objects: {{"type": "ORG|PERSON|LOCATION|DATE|AMOUNT|ACT_SECTION", "text": "...", "normalized": "..."}}
Only return the JSON array, no explanation.

Text:
{text[:3000]}"""

    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            operation="entity_extraction",
            job_id=job_id,
            document_id=document_id,
        )
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return [
            Entity(
                id=f"ent_{uuid.uuid4().hex[:8]}",
                type=item.get("type", "UNKNOWN"),
                text=item.get("text", ""),
                normalized=item.get("normalized"),
                confidence=0.8,
                source_chunk=source_chunk,
            )
            for item in parsed
            if item.get("text")
        ]
    except (json.JSONDecodeError, KeyError, IndexError):
        return []


async def extract_entities(
    llm: LLMService,
    chunks: list,
    language: str,
    use_llm: bool = True,
    job_id: str | None = None,
    document_id: str | None = None,
    *,
    concurrency: int = 1,
) -> list[Entity]:
    """Extract entities with one LLM call per chunk (unchanged prompts for accuracy)."""
    all_entities: list[Entity] = []
    llm_chunks = []

    for chunk in chunks:
        all_entities.extend(extract_regex_entities(chunk.text, chunk.id))
        if use_llm and len(chunk.text) > 50:
            llm_chunks.append(chunk)

    if not llm_chunks:
        return _dedupe(all_entities)

    workers = max(1, min(concurrency, len(llm_chunks)))
    if workers == 1:
        for chunk in llm_chunks:
            all_entities.extend(
                await extract_llm_entities(
                    llm, chunk.text, language, chunk.id, job_id=job_id, document_id=document_id
                )
            )
        return _dedupe(all_entities)

    sem = asyncio.Semaphore(workers)

    async def one(chunk) -> list[Entity]:
        async with sem:
            return await extract_llm_entities(
                llm, chunk.text, language, chunk.id, job_id=job_id, document_id=document_id
            )

    for result in await asyncio.gather(*[one(chunk) for chunk in llm_chunks]):
        all_entities.extend(result)

    return _dedupe(all_entities)


def _dedupe(entities: list[Entity]) -> list[Entity]:
    seen: dict[str, Entity] = {}
    for ent in entities:
        key = f"{ent.type}:{ent.normalized or ent.text}".lower()
        if key not in seen or ent.confidence > seen[key].confidence:
            seen[key] = ent
    return list(seen.values())
