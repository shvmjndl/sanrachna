"""Layout-aware parent-child chunking for Indic documents."""

from __future__ import annotations

import re
import uuid

from app.core.config import Settings
from app.schemas.knowledge import BBox, Chunk, ChunkType, ProvenanceInfo
from pipeline.record_detection import group_records

INDIC_SENTENCE_END = re.compile(r"(?<=[।.!?])\s+")
SECTION_PATTERN = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_markdown(
    markdown: str,
    settings: Settings,
    structured: dict | None = None,
) -> list[Chunk]:
    """Return parent + child chunks. Children are indexed; parents provide context."""
    if structured and "pages" in structured:
        flat = _chunk_from_structure(structured, settings)
        if flat:
            return _wrap_as_parent_child(flat, settings)

    sections = _split_sections(markdown)
    all_chunks: list[Chunk] = []

    for section_name, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        parent = Chunk(
            id=parent_id,
            text=section_text,
            section=section_name,
            token_count=_estimate_tokens(section_text),
            chunk_type=ChunkType.PARENT,
        )
        all_chunks.append(parent)

        child_size = settings.child_chunk_size * 4
        overlap = settings.chunk_overlap * 4
        all_chunks.extend(_split_into_children(section_text, section_name, parent_id, child_size, overlap))

    return all_chunks


def _split_into_children(
    section_text: str,
    section_name: str | None,
    parent_id: str,
    child_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split section into multiple child chunks — critical for search granularity."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]

    # Multi-paragraph: one child per paragraph (or group small ones)
    if len(paragraphs) > 1:
        chunks: list[Chunk] = []
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) > child_size and buf:
                chunks.append(_make_child(buf, section_name, parent_id, section_text))
                buf = para
            else:
                buf = f"{buf}\n\n{para}".strip() if buf else para
        if buf:
            chunks.append(_make_child(buf, section_name, parent_id, section_text))
        if len(chunks) > 1:
            return chunks

    # Dense single-block: prefer intact label+value records from PDF tables
    lines = [ln.strip() for ln in section_text.split("\n") if ln.strip()]
    records = group_records(lines)
    if len(records) >= 2:
        return [
            _make_child("\n".join(record), section_name, parent_id, section_text)
            for record in records
        ]

    if len(lines) > 6:
        chunks = []
        group_size = 5
        for i in range(0, len(lines), group_size):
            group = "\n".join(lines[i : i + group_size])
            chunks.append(_make_child(group, section_name, parent_id, section_text))
        return chunks

    if len(section_text) <= child_size:
        return [_make_child(section_text, section_name, parent_id, section_text)]

    # Fall back to sentence splitting
    chunks = []
    sentences = INDIC_SENTENCE_END.split(section_text)
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) > child_size and buf:
            chunks.append(_make_child(buf, section_name, parent_id, section_text))
            buf = (buf[-overlap:] if overlap else "") + sent
        else:
            buf += (" " if buf else "") + sent
    if buf.strip():
        chunks.append(_make_child(buf, section_name, parent_id, section_text))
    return chunks


def _make_child(
    text: str,
    section: str | None,
    parent_id: str,
    parent_text: str,
    *,
    page: int | None = None,
    bbox: BBox | None = None,
    provenance: ProvenanceInfo | None = None,
    confidence: float = 1.0,
) -> Chunk:
    return Chunk(
        id=f"child_{uuid.uuid4().hex[:8]}",
        text=text.strip(),
        page=page,
        section=section,
        token_count=_estimate_tokens(text),
        parent_id=parent_id,
        parent_text=parent_text,
        chunk_type=ChunkType.CHILD,
        bbox=bbox,
        provenance=provenance,
        confidence=confidence,
    )


def get_indexable_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Only child chunks go into the search index."""
    children = [c for c in chunks if c.chunk_type == ChunkType.CHILD]
    return children if children else chunks


def _wrap_as_parent_child(flat: list[Chunk], settings: Settings) -> list[Chunk]:
    """Group flat structure chunks into parent-child hierarchy by section/page."""
    groups: dict[str, list[Chunk]] = {}
    for c in flat:
        key = f"{c.page}:{c.section}"
        groups.setdefault(key, []).append(c)

    result: list[Chunk] = []
    for group_chunks in groups.values():
        combined = "\n\n".join(c.text for c in group_chunks)
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        result.append(
            Chunk(
                id=parent_id,
                text=combined,
                page=group_chunks[0].page,
                section=group_chunks[0].section,
                token_count=_estimate_tokens(combined),
                chunk_type=ChunkType.PARENT,
                bbox=group_chunks[0].bbox,
                provenance=group_chunks[0].provenance,
                confidence=group_chunks[0].confidence,
            )
        )
        child_size = settings.child_chunk_size * 4
        overlap = settings.chunk_overlap * 4
        result.extend(
            _split_into_children(
                combined,
                group_chunks[0].section,
                parent_id,
                child_size,
                overlap,
            )
        )
    return result


def _split_sections(markdown: str) -> list[tuple[str | None, str]]:
    matches = list(SECTION_PATTERN.finditer(markdown))
    if not matches:
        return [(None, markdown)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, markdown[: matches[0].start()]))

    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((name, markdown[start:end]))

    return sections


def _chunk_from_structure(structured: dict, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_data in structured.get("pages", []):
        if isinstance(page_data, dict) and "pages" in page_data:
            for inner in page_data.get("pages", []):
                chunks.extend(_blocks_to_chunks(inner))
        else:
            chunks.extend(_blocks_to_chunks(page_data))
    return chunks


def _blocks_to_chunks(page_data: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    page_num = page_data.get("page_number") or page_data.get("page")
    blocks = page_data.get("blocks", page_data.get("content", []))

    if isinstance(blocks, list):
        for block in blocks:
            text = block.get("text", "") if isinstance(block, dict) else str(block)
            if not text.strip():
                continue
            tag = block.get("layout_tag", "paragraph") if isinstance(block, dict) else "paragraph"
            bbox = block.get("bbox") if isinstance(block, dict) else None
            provenance = block.get("provenance") if isinstance(block, dict) else None
            confidence = block.get("confidence", 1.0) if isinstance(block, dict) else 1.0
            chunks.append(
                Chunk(
                    id=f"chunk_{uuid.uuid4().hex[:8]}",
                    text=text.strip(),
                    page=page_num,
                    section=tag,
                    token_count=_estimate_tokens(text),
                    chunk_type=ChunkType.CHILD,
                    bbox=bbox,
                    provenance=provenance,
                    confidence=confidence,
                )
            )
    return chunks
