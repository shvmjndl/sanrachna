"""Image captioning and table extraction enrichment."""

from __future__ import annotations

import re
import uuid

from app.schemas.knowledge import ImageAsset, Table
from pipeline.record_detection import filter_coherent_rows, group_records
from services.llm import LLMService

TABLE_PATTERN = re.compile(r"(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)")
COLUMN_SPLIT = re.compile(r"\t| {2,}")


def extract_tables(markdown: str) -> list[Table]:
    tables: list[Table] = []
    for i, match in enumerate(TABLE_PATTERN.finditer(markdown)):
        raw = match.group(1)
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in lines[2:]
        ]
        tables.append(
            Table(id=f"table_{i + 1:02d}", headers=headers, rows=rows, raw_markdown=raw)
        )

    tables.extend(_extract_column_aligned_tables(markdown, offset=len(tables)))
    tables.extend(_extract_vertical_record_tables(markdown, offset=len(tables)))
    return tables


def table_row_to_text(headers: list[str], row: list[str]) -> str:
    """Flatten a table row into a self-contained searchable line."""
    if not row:
        return ""
    if headers and len(headers) == len(row):
        return " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
    if headers and len(headers) < len(row):
        pairs = list(zip(headers, row[: len(headers)]))
        extras = row[len(headers) :]
        text = " | ".join(f"{h}: {v}" for h, v in pairs if v)
        if extras:
            text += " | " + " | ".join(extras)
        return text
    return " | ".join(row)


def tables_to_search_chunks(tables: list[Table]) -> list[str]:
    """Convert structured tables into atomic search texts — one row per chunk."""
    chunks: list[str] = []
    for table in tables:
        for row in table.rows:
            text = table_row_to_text(table.headers, row)
            if text.strip():
                chunks.append(text)
    return chunks


def _split_columns(line: str) -> list[str]:
    return [part.strip() for part in COLUMN_SPLIT.split(line.strip()) if part.strip()]


def _extract_column_aligned_tables(markdown: str, offset: int = 0) -> list[Table]:
    """Detect tables from whitespace- or tab-aligned column blocks."""
    lines = [line.rstrip() for line in markdown.split("\n") if line.strip()]
    tables: list[Table] = []
    i = 0

    while i < len(lines):
        first_cols = _split_columns(lines[i])
        if len(first_cols) < 2:
            i += 1
            continue

        col_count = len(first_cols)
        block_rows = [first_cols]
        j = i + 1

        while j < len(lines):
            cols = _split_columns(lines[j])
            if len(cols) < 2:
                break
            if abs(len(cols) - col_count) <= 1:
                if len(cols) < col_count:
                    cols.extend([""] * (col_count - len(cols)))
                elif len(cols) > col_count:
                    cols = cols[:col_count]
                block_rows.append(cols)
                j += 1
            else:
                break

        if len(block_rows) >= 3:
            tables.append(
                Table(
                    id=f"table_{offset + len(tables) + 1:02d}",
                    headers=block_rows[0],
                    rows=block_rows[1:],
                    raw_markdown="\n".join(lines[i:j]),
                )
            )
            i = j
        else:
            i += 1

    return tables


def _extract_vertical_record_tables(markdown: str, offset: int = 0) -> list[Table]:
    """Parse vertically stacked label+value sequences from PDF text extraction."""
    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    records = group_records(lines)
    if len(records) < 2:
        return []

    max_width = max(len(record) for record in records)
    if max_width < 2:
        return []

    headers = [f"col_{i + 1}" for i in range(max_width)]
    normalized_rows = [record + [""] * (max_width - len(record)) for record in records]
    filtered_rows = filter_coherent_rows(normalized_rows)
    if not filtered_rows:
        return []

    return [
        Table(
            id=f"table_{offset + 1:02d}",
            headers=headers,
            rows=filtered_rows,
            raw_markdown="\n\n".join("\n".join(cell for cell in row if cell) for row in filtered_rows),
        )
    ]


async def caption_images(
    llm: LLMService,
    images: list[ImageAsset],
    context: str,
    job_id: str | None = None,
    document_id: str | None = None,
    *,
    concurrency: int = 4,
) -> list[ImageAsset]:
    import asyncio

    sem = asyncio.Semaphore(max(1, concurrency))

    async def caption_one(img: ImageAsset) -> ImageAsset:
        if img.caption:
            return img
        ocr = img.ocr_text or ""
        prompt = f"""Generate a concise caption for this document image.
OCR text: {ocr}
Surrounding context: {context[:500]}
Return only the caption text."""
        async with sem:
            try:
                caption, _ = await llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    operation="image_captioning",
                    job_id=job_id,
                    document_id=document_id,
                )
                return img.model_copy(update={"caption": caption.strip()})
            except Exception:
                return img

    return list(await asyncio.gather(*[caption_one(img) for img in images]))


def extract_image_refs(markdown: str, structured: dict | None = None) -> list[ImageAsset]:
    images: list[ImageAsset] = []
    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    for i, match in enumerate(img_pattern.finditer(markdown)):
        images.append(
            ImageAsset(
                id=f"img_{i + 1:02d}",
                path=match.group(2),
                caption=match.group(1) or None,
            )
        )

    if structured:
        for page_data in structured.get("pages", []):
            if not isinstance(page_data, dict):
                continue
            for block in page_data.get("blocks", []):
                if isinstance(block, dict) and block.get("layout_tag") == "figure":
                    images.append(
                        ImageAsset(
                            id=f"img_{uuid.uuid4().hex[:8]}",
                            ocr_text=block.get("text", ""),
                            page=page_data.get("page_number"),
                        )
                    )
    return images
