"""Structure-based record detection for vertically extracted document text.

No document-type keywords — uses only layout signals (label length, value shape,
field-count consistency) that apply to any tabular or key-value content.
"""

from __future__ import annotations

import re

_NUMERIC_VALUE = re.compile(r"^[\d,]+(?:\.\d+)?%?$")
_LIST_MARKER = re.compile(r"^\d{1,3}\.$")
_MAX_LABEL_WORDS = 10


def is_record_label(line: str) -> bool:
    """Short, text-heavy line that likely heads a structured record."""
    text = line.strip()
    if not text or _NUMERIC_VALUE.match(text):
        return False
    if text.startswith(":") and sum(c.isalpha() for c in text) < 4:
        return False
    words = text.split()
    if len(words) > _MAX_LABEL_WORDS:
        return False
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    return letters >= 3 and letters > digits


def is_record_value(line: str) -> bool:
    """Compact value line that belongs with a preceding record label."""
    text = line.strip()
    if not text or _LIST_MARKER.match(text):
        return False
    if _NUMERIC_VALUE.match(text):
        return True
    return len(text) <= 20 and any(c.isdigit() for c in text) and not is_record_label(text)


def group_records(lines: list[str]) -> list[list[str]]:
    """Group lines into label + value record rows."""
    records: list[list[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if is_record_label(line) and i + 1 < len(lines) and is_record_value(lines[i + 1]):
            block = [line]
            j = i + 1
            while j < len(lines) and is_record_value(lines[j]) and not is_record_label(lines[j]):
                block.append(lines[j].strip())
                j += 1
            if len(block) >= 2:
                records.append(block)
                i = j
                continue
        i += 1
    return records


def _has_short_values(row: list[str]) -> bool:
    """Detail rows typically mix compact codes/rates with longer values."""
    return any(1 <= len(cell.strip()) <= 7 for cell in row[1:] if cell.strip())


def filter_coherent_rows(rows: list[list[str]], min_rows: int = 2) -> list[list[str]]:
    """Keep the row cluster with consistent width and the strongest detail signal."""
    if len(rows) < min_rows:
        return []

    clusters: dict[int, list[list[str]]] = {}
    for row in rows:
        width = sum(1 for cell in row if cell.strip())
        clusters.setdefault(width, []).append(row)

    best: list[list[str]] = []
    best_score = -1.0
    for width, group in clusters.items():
        if width < 2 or len(group) < min_rows:
            continue
        detail_ratio = sum(1 for row in group if _has_short_values(row)) / len(group)
        score = len(group) + detail_ratio
        if score > best_score:
            best_score = score
            best = group

    if len(best) < min_rows:
        return []

    return [
        row
        for row in best
        if sum(1 for cell in row if cell.strip()) >= max(2, len(best[0]) - 1)
        and len(row[0].split()) <= _MAX_LABEL_WORDS
    ]
