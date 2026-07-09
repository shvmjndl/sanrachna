"""Normalize bbox/provenance from heterogeneous digitizer outputs."""

from __future__ import annotations

import uuid
from typing import Any

from app.schemas.knowledge import BBox, ProvenanceInfo


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bbox(raw: Any, page_width: float | None = None, page_height: float | None = None) -> BBox | None:
    if not raw:
        return None

    if isinstance(raw, BBox):
        return raw

    if isinstance(raw, dict):
        keys = raw.keys()
        if {"x0", "y0", "x1", "y1"}.issubset(keys):
            coords = [_as_float(raw[k]) for k in ("x0", "y0", "x1", "y1")]
        elif {"left", "top", "right", "bottom"}.issubset(keys):
            coords = [
                _as_float(raw["left"]),
                _as_float(raw["top"]),
                _as_float(raw["right"]),
                _as_float(raw["bottom"]),
            ]
        elif {"x", "y", "width", "height"}.issubset(keys):
            x = _as_float(raw["x"]) or 0.0
            y = _as_float(raw["y"]) or 0.0
            w = _as_float(raw["width"]) or 0.0
            h = _as_float(raw["height"]) or 0.0
            coords = [x, y, x + w, y + h]
        elif len(raw.get("bbox", [])) == 4:
            coords = [_as_float(v) for v in raw["bbox"]]
        elif len(raw.get("coordinates", [])) == 4:
            coords = [_as_float(v) for v in raw["coordinates"]]
        else:
            return None

        if any(c is None for c in coords):
            return None
        return BBox(
            x0=coords[0],
            y0=coords[1],
            x1=coords[2],
            y1=coords[3],
            page_width=page_width,
            page_height=page_height,
        )

    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        coords = [_as_float(v) for v in raw]
        if any(c is None for c in coords):
            return None
        return BBox(
            x0=coords[0],
            y0=coords[1],
            x1=coords[2],
            y1=coords[3],
            page_width=page_width,
            page_height=page_height,
        )

    return None


def normalize_block(
    block: dict,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    source: str = "unknown",
) -> dict:
    text = (block.get("text") or block.get("content") or "").strip()
    layout_tag = block.get("layout_tag") or block.get("type") or block.get("label") or "paragraph"
    block_id = block.get("id") or block.get("block_id") or f"blk_{uuid.uuid4().hex[:8]}"

    bbox_raw = (
        block.get("bbox")
        or block.get("bounding_box")
        or block.get("boundingBox")
        or block.get("coordinates")
        or block.get("rect")
    )
    bbox = parse_bbox(bbox_raw, page_width, page_height)

    confidence = _as_float(block.get("confidence"))
    if confidence is None:
        confidence = 0.95 if bbox else 0.85

    return {
        "text": text,
        "layout_tag": layout_tag,
        "block_id": block_id,
        "bbox": bbox,
        "confidence": min(1.0, max(0.0, confidence)),
        "provenance": ProvenanceInfo(
            source=source,
            block_id=block_id,
            layout_tag=layout_tag,
        ),
    }


def normalize_structured(structured: dict, source: str = "unknown") -> dict:
    """Normalize pages/blocks from local or Sarvam JSON into a common shape."""
    if not structured:
        return structured

    pages_out: list[dict] = []
    pages = structured.get("pages", [])

    for page_data in pages:
        if not isinstance(page_data, dict):
            continue

        if "pages" in page_data:
            for inner in page_data.get("pages", []):
                pages_out.append(_normalize_page(inner, source))
            continue

        pages_out.append(_normalize_page(page_data, source))

    structured["pages"] = pages_out
    structured["provider"] = structured.get("provider") or source
    return structured


def _normalize_page(page_data: dict, source: str) -> dict:
    page_num = page_data.get("page_number") or page_data.get("page")
    page_width = _as_float(page_data.get("width") or page_data.get("page_width"))
    page_height = _as_float(page_data.get("height") or page_data.get("page_height"))

    raw_blocks = page_data.get("blocks") or page_data.get("content") or []
    blocks: list[dict] = []
    for block in raw_blocks:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                blocks.append(
                    normalize_block({"text": text}, page_width=page_width, page_height=page_height, source=source)
                )
            continue
        normalized = normalize_block(
            block,
            page_width=page_width,
            page_height=page_height,
            source=source,
        )
        if normalized["text"]:
            blocks.append(normalized)

    return {
        "page_number": page_num,
        "width": page_width,
        "height": page_height,
        "blocks": blocks,
    }
