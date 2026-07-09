"""Local digitization for English docs — no Sarvam required.

Works for text-based PDFs, TXT, MD, DOCX, PPTX, EML.
Scanned/image-only PDFs need Sarvam or another OCR provider.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz

from pipeline.converters import convert_to_markdown
from pipeline.provenance import normalize_block, normalize_structured

_LINE_GROUP = 5


class LocalDigitizer:
    def process_file(self, file_path: Path) -> tuple[str, dict, int]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext in (".txt", ".md", ".csv", ".eml", ".docx", ".pptx"):
            md = convert_to_markdown(file_path)
            return md, {"provider": "local", "format": ext}, 1
        if ext in (".png", ".jpg", ".jpeg"):
            return (
                f"[Image file: {file_path.name} — local mode cannot OCR images. "
                "Use Sarvam for scanned documents or upload a text-based PDF.]",
                {"provider": "local", "format": ext},
                1,
            )

        return f"[Unsupported in local mode: {ext}]", {"provider": "local"}, 0

    def _extract_pdf(self, pdf_path: Path) -> tuple[str, dict, int]:
        doc = fitz.open(pdf_path)
        pages_md: list[str] = []
        structured_pages: list[dict] = []

        for i, page in enumerate(doc):
            page_num = i + 1
            page_rect = page.rect
            page_width = float(page_rect.width)
            page_height = float(page_rect.height)

            blocks = self._extract_blocks(page, page_width, page_height)
            text = "\n\n".join(b["text"] for b in blocks if b["text"]).strip()
            pages_md.append(
                f"## Page {page_num}\n\n{text}" if text else f"## Page {page_num}\n\n[No extractable text]"
            )
            structured_pages.append(
                {
                    "page_number": page_num,
                    "width": page_width,
                    "height": page_height,
                    "blocks": blocks,
                }
            )

        doc.close()
        markdown = "\n\n---\n\n".join(pages_md)
        page_count = len(structured_pages)

        structured = normalize_structured(
            {
                "provider": "local",
                "pages": structured_pages,
                "extraction": "pymupdf",
            },
            source="local",
        )

        if not markdown.strip() or all("[No extractable text]" in p for p in pages_md):
            markdown = (
                f"[PDF {pdf_path.name} has no extractable text — likely a scanned document. "
                "Set SARVAM_API_KEY for OCR, or upload a text-based PDF.]"
            )

        return markdown, structured, page_count

    def _extract_blocks(self, page, page_width: float, page_height: float) -> list[dict]:
        blocks: list[dict] = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            lines: list[str] = []
            x0 = y0 = x1 = y1 = None
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if line_text:
                    lines.append(line_text)
                for span in spans:
                    bbox = span.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    sx0, sy0, sx1, sy1 = [float(v) for v in bbox]
                    x0 = sx0 if x0 is None else min(x0, sx0)
                    y0 = sy0 if y0 is None else min(y0, sy0)
                    x1 = sx1 if x1 is None else max(x1, sx1)
                    y1 = sy1 if y1 is None else max(y1, sy1)

            text = "\n".join(lines).strip()
            if not text:
                continue

            raw = {
                "text": text,
                "layout_tag": "paragraph",
                "bbox": [x0, y0, x1, y1] if None not in (x0, y0, x1, y1) else None,
                "id": f"blk_{uuid.uuid4().hex[:8]}",
                "confidence": 0.97,
            }
            blocks.append(
                normalize_block(raw, page_width=page_width, page_height=page_height, source="local")
            )

        if blocks:
            return blocks

        text = page.get_text("text").strip()
        return self._text_to_blocks(text, page_width, page_height) if text else []

    def _text_to_blocks(self, text: str, page_width: float, page_height: float) -> list[dict]:
        """Fallback when dict extraction yields no blocks."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        source_blocks: list[dict]
        if len(paragraphs) > 1:
            source_blocks = [{"layout_tag": "paragraph", "text": p} for p in paragraphs]
        else:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if len(lines) <= _LINE_GROUP:
                source_blocks = [{"layout_tag": "paragraph", "text": text}]
            else:
                source_blocks = [
                    {"layout_tag": "paragraph", "text": "\n".join(lines[i : i + _LINE_GROUP])}
                    for i in range(0, len(lines), _LINE_GROUP)
                ]

        return [
            normalize_block(b, page_width=page_width, page_height=page_height, source="local")
            for b in source_blocks
            if b.get("text")
        ]
