"""Generic document family classification from layout/structure signals."""

from __future__ import annotations

import re
from collections import Counter

from app.schemas.knowledge import Chunk, ChunkType, Entity, KnowledgeObject, Table

_NUMERIC_DENSE = re.compile(r"[\d,₹$%.]+")
_LABEL_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9\s/&().-]{1,60}:\s*\S", re.M)


def classify_document(
    markdown: str,
    chunks: list[Chunk],
    tables: list[Table],
    entities: list[Entity],
    pages: int,
) -> tuple[str, dict]:
    """Return (family_label, signals) using structural heuristics only."""
    text = markdown or ""
    child_chunks = [c for c in chunks if c.chunk_type == ChunkType.CHILD]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    numeric_lines = sum(1 for ln in lines if len(_NUMERIC_DENSE.findall(ln)) >= 2)
    numeric_ratio = numeric_lines / max(1, len(lines))
    record_chunks = sum(1 for c in child_chunks if c.section == "record")
    table_rows = sum(len(t.rows) for t in tables)
    bbox_ratio = sum(1 for c in child_chunks if c.bbox) / max(1, len(child_chunks))
    entity_types = Counter(e.type for e in entities)
    label_value_lines = len(_LABEL_VALUE.findall(text))
    avg_chunk_len = (
        sum(len(c.text) for c in child_chunks) / len(child_chunks) if child_chunks else 0
    )

    signals = {
        "pages": pages,
        "child_chunks": len(child_chunks),
        "numeric_ratio": round(numeric_ratio, 3),
        "record_chunks": record_chunks,
        "table_rows": table_rows,
        "bbox_ratio": round(bbox_ratio, 3),
        "entity_types": dict(entity_types),
        "label_value_lines": label_value_lines,
        "avg_chunk_len": round(avg_chunk_len, 1),
    }

    if record_chunks >= 2 or table_rows >= 6 or numeric_ratio >= 0.45:
        family = "tabular_records"
    elif label_value_lines >= 8 and numeric_ratio >= 0.25:
        family = "key_value_form"
    elif avg_chunk_len >= 500 and numeric_ratio < 0.2:
        family = "narrative_longform"
    elif pages <= 1 and len(child_chunks) <= 3 and len(text) < 800:
        family = "sparse_short"
    elif bbox_ratio >= 0.5 and numeric_ratio < 0.3:
        family = "layout_rich"
    else:
        family = "mixed_general"

    return family, signals


def classify_knowledge_object(ko: KnowledgeObject) -> KnowledgeObject:
    family, signals = classify_document(
        ko.markdown,
        ko.chunks,
        ko.tables,
        ko.entities,
        ko.metadata.pages,
    )
    ko.source.document_family = family
    ko.source.family_signals = {**ko.source.family_signals, "family": signals}
    return ko
