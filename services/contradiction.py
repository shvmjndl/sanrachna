"""Contradiction detection across documents and within corpus."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from app.db.models import Document
from services.compare import CompareService


_MONEY = re.compile(r"[\d,]+(?:\.\d+)?")


class ContradictionService:
    def __init__(self, session_factory):
        self._session = session_factory
        self._compare = CompareService(session_factory)

    def detect_between(self, document_id_a: str, document_id_b: str) -> dict[str, Any]:
        comparison = self._compare.compare(document_id_a, document_id_b)
        if comparison.get("error"):
            return comparison

        contradictions: list[dict] = []

        for item in comparison.get("shared_entities", []):
            if item.get("changed"):
                contradictions.append(
                    {
                        "type": "entity_value_mismatch",
                        "field": item["key"],
                        "entity_type": item["type"],
                        "value_a": item["text_a"],
                        "value_b": item["text_b"],
                        "severity": "high" if item["type"] in ("MONEY", "GSTIN", "PAN") else "medium",
                    }
                )

        for diff in comparison.get("field_diffs", []):
            contradictions.append(
                {
                    "type": "label_value_mismatch",
                    "field": diff["label"],
                    "value_a": diff["value_a"],
                    "value_b": diff["value_b"],
                    "severity": self._money_severity(diff["value_a"], diff["value_b"]),
                }
            )

        for td in comparison.get("table_diffs", []):
            if td.get("rows_only_in_a") and td.get("rows_only_in_b"):
                contradictions.append(
                    {
                        "type": "table_row_divergence",
                        "table_a": td["table_a"],
                        "table_b": td["table_b"],
                        "headers": td["headers"],
                        "severity": "medium",
                    }
                )

        return {
            "document_id_a": document_id_a,
            "document_id_b": document_id_b,
            "contradiction_count": len(contradictions),
            "contradictions": contradictions,
            "comparison_summary": comparison.get("summary"),
        }

    def detect_in_corpus(self, corpus_id: str, limit: int = 10) -> dict[str, Any]:
        with self._session() as session:
            docs = session.execute(
                select(Document).where(Document.corpus_id == corpus_id)
            ).scalars().all()

        doc_ids = [d.id for d in docs if d.knowledge_object]
        pairs: list[dict] = []
        for i, a in enumerate(doc_ids):
            for b in doc_ids[i + 1 : i + 4]:
                result = self.detect_between(a, b)
                if result.get("contradiction_count", 0) > 0:
                    pairs.append(result)
                if len(pairs) >= limit:
                    break
            if len(pairs) >= limit:
                break

        return {"corpus_id": corpus_id, "pairs_checked": len(doc_ids), "conflicts": pairs}

    @staticmethod
    def _money_severity(a: str, b: str) -> str:
        nums_a = [float(x.replace(",", "")) for x in _MONEY.findall(a) if x.replace(",", "").replace(".", "").isdigit()]
        nums_b = [float(x.replace(",", "")) for x in _MONEY.findall(b) if x.replace(",", "").replace(".", "").isdigit()]
        if nums_a and nums_b and nums_a[0] != nums_b[0]:
            return "high"
        return "medium"
