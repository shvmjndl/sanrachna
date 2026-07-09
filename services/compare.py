"""Document comparison service — entities, tables, field diffs."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from app.db.models import Document
from app.schemas.knowledge import KnowledgeObject


class CompareService:
    def __init__(self, session_factory):
        self._session = session_factory

    def compare(self, document_id_a: str, document_id_b: str) -> dict[str, Any]:
        ko_a = self._load_ko(document_id_a)
        ko_b = self._load_ko(document_id_b)
        if not ko_a or not ko_b:
            return {"error": "One or both documents not found"}

        ents_a = {self._ent_key(e): e for e in ko_a.entities}
        ents_b = {self._ent_key(e): e for e in ko_b.entities}

        shared, only_a, only_b = [], [], []
        for key, ent in ents_a.items():
            if key in ents_b:
                shared.append(
                    {
                        "key": key,
                        "type": ent.type,
                        "text_a": ent.text,
                        "text_b": ents_b[key].text,
                        "changed": ent.text != ents_b[key].text,
                    }
                )
            else:
                only_a.append({"key": key, "type": ent.type, "text": ent.text})

        for key, ent in ents_b.items():
            if key not in ents_a:
                only_b.append({"key": key, "type": ent.type, "text": ent.text})

        table_diffs = self._diff_tables(ko_a.tables, ko_b.tables)
        field_diffs = self._diff_common_fields(ko_a, ko_b)

        return {
            "document_id_a": document_id_a,
            "document_id_b": document_id_b,
            "shared_entities": shared[:50],
            "only_in_a": only_a[:30],
            "only_in_b": only_b[:30],
            "table_diffs": table_diffs,
            "field_diffs": field_diffs,
            "summary": {
                "shared_entity_count": len(shared),
                "only_a_count": len(only_a),
                "only_b_count": len(only_b),
                "changed_entities": sum(1 for s in shared if s["changed"]),
                "tables_a": len(ko_a.tables),
                "tables_b": len(ko_b.tables),
            },
        }

    def _load_ko(self, document_id: str) -> KnowledgeObject | None:
        with self._session() as session:
            doc = session.execute(select(Document).where(Document.id == document_id)).scalar_one_or_none()
            if not doc or not doc.knowledge_object:
                return None
            return KnowledgeObject(**doc.knowledge_object)

    @staticmethod
    def _ent_key(ent) -> str:
        return (ent.normalized or ent.text or "").strip().lower()

    def _diff_tables(self, tables_a, tables_b) -> list[dict]:
        diffs = []
        for i, ta in enumerate(tables_a[:10]):
            for j, tb in enumerate(tables_b[:10]):
                if ta.headers and tb.headers and ta.headers == tb.headers:
                    rows_a = {" | ".join(r) for r in ta.rows}
                    rows_b = {" | ".join(r) for r in tb.rows}
                    diffs.append(
                        {
                            "table_a": ta.id,
                            "table_b": tb.id,
                            "headers": ta.headers,
                            "rows_only_in_a": list(rows_a - rows_b)[:20],
                            "rows_only_in_b": list(rows_b - rows_a)[:20],
                            "shared_rows": len(rows_a & rows_b),
                        }
                    )
        return diffs[:15]

    def _diff_common_fields(self, ko_a: KnowledgeObject, ko_b: KnowledgeObject) -> list[dict]:
        """Extract label:value pairs generically and diff overlapping labels."""
        fields_a = self._extract_label_values(ko_a.markdown)
        fields_b = self._extract_label_values(ko_b.markdown)
        diffs = []
        for label in set(fields_a) & set(fields_b):
            if fields_a[label] != fields_b[label]:
                diffs.append(
                    {
                        "label": label,
                        "value_a": fields_a[label],
                        "value_b": fields_b[label],
                    }
                )
        return diffs[:40]

    @staticmethod
    def _extract_label_values(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^([A-Za-z][A-Za-z0-9\s/&().-]{1,50}):\s*(.+)$", line.strip())
            if m:
                out[m.group(1).strip().lower()] = m.group(2).strip()
        return out
