"""Compute document quality scores from structural signals — no doc-type keywords."""

from __future__ import annotations

import re

from app.schemas.knowledge import Chunk, ChunkType, KnowledgeObject, Table

_OCR_FALLBACK = re.compile(r"\[(?:No extractable text|Image file|Unsupported|scanned document)", re.I)


def score_document(
    markdown: str,
    chunks: list[Chunk],
    tables: list[Table],
    pages: int,
) -> tuple[float, dict[str, float]]:
    """Return quality score in [0,1] and component breakdown."""
    text = markdown or ""
    text_len = len(text.strip())
    child_chunks = [c for c in chunks if c.chunk_type == ChunkType.CHILD]

    text_density = min(1.0, text_len / max(1, pages * 500))
    chunk_coverage = min(1.0, len(child_chunks) / max(1, pages * 2))
    table_signal = min(1.0, len(tables) / 3)
    bbox_signal = min(
        1.0,
        sum(1 for c in child_chunks if c.bbox) / max(1, len(child_chunks)),
    )
    avg_chunk_conf = (
        sum(c.confidence for c in child_chunks) / len(child_chunks) if child_chunks else 0.5
    )

    ocr_penalty = 0.35 if _OCR_FALLBACK.search(text) else 0.0
    sparse_penalty = 0.25 if text_len < 120 and pages > 0 else 0.0

    raw = (
        0.30 * text_density
        + 0.20 * chunk_coverage
        + 0.15 * table_signal
        + 0.15 * bbox_signal
        + 0.20 * avg_chunk_conf
        - ocr_penalty
        - sparse_penalty
    )
    score = round(max(0.0, min(1.0, raw)), 3)

    return score, {
        "text_density": round(text_density, 3),
        "chunk_coverage": round(chunk_coverage, 3),
        "table_signal": round(table_signal, 3),
        "bbox_signal": round(bbox_signal, 3),
        "avg_chunk_confidence": round(avg_chunk_conf, 3),
        "ocr_penalty": ocr_penalty,
        "sparse_penalty": sparse_penalty,
    }


def apply_quality_to_chunks(chunks: list[Chunk], quality_score: float) -> list[Chunk]:
    """Scale chunk confidence by document quality for retrieval ranking."""
    factor = 0.65 + 0.35 * quality_score
    for chunk in chunks:
        chunk.confidence = round(min(1.0, chunk.confidence * factor), 3)
    return chunks


def enrich_knowledge_object(ko: KnowledgeObject) -> KnowledgeObject:
    score, breakdown = score_document(
        ko.markdown,
        ko.chunks,
        ko.tables,
        ko.metadata.pages,
    )
    ko.metadata.quality_score = score
    ko.source.family_signals = {**ko.source.family_signals, "quality": breakdown}
    ko.chunks = apply_quality_to_chunks(ko.chunks, score)
    return ko
