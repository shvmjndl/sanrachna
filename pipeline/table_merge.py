"""Cross-page table merge — structural continuation detection, no doc-type keywords."""

from __future__ import annotations

import re

from app.schemas.knowledge import Table

_HEADER_REPEAT = re.compile(r"^(description|item|particulars|sr\.?\s*no|s\.?\s*no\.?|date|amount|qty|quantity|rate|total)\b", re.I)


def _normalize_header(headers: list[str]) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in headers if h.strip())


def _headers_compatible(a: list[str], b: list[str]) -> bool:
    na, nb = _normalize_header(a), _normalize_header(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    overlap = len(set(na) & set(nb))
    return overlap >= max(2, min(len(na), len(nb)) // 2)


def _is_continuation_row(headers: list[str], row: list[str]) -> bool:
    if not row:
        return False
    first = (row[0] or "").strip()
    if not first:
        return True
    if _HEADER_REPEAT.match(first):
        return False
    numeric_cells = sum(1 for c in row[1:] if re.search(r"\d", c or ""))
    return numeric_cells >= 1 or len(first) < 60


def merge_cross_page_tables(tables: list[Table]) -> list[Table]:
    if len(tables) < 2:
        return tables

    merged: list[Table] = []
    for table in tables:
        if not merged:
            merged.append(table)
            continue

        prev = merged[-1]
        if _headers_compatible(prev.headers, table.headers):
            new_rows = list(prev.rows)
            for row in table.rows:
                if row == table.headers:
                    continue
                if _is_continuation_row(prev.headers, row):
                    new_rows.append(row)
            prev.rows = new_rows
            if table.page and prev.page:
                prev.page_end = table.page
            elif table.page:
                prev.page_end = table.page
            if table.raw_markdown:
                prev.raw_markdown = (prev.raw_markdown or "") + "\n" + table.raw_markdown
            continue

        merged.append(table)

    for i, t in enumerate(merged):
        if t.page_end and t.page and t.page_end > t.page:
            t.caption = t.caption or f"table_{i + 1}_p{t.page}-{t.page_end}"
    return merged
