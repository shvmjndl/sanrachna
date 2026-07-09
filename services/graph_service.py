"""Knowledge graph query service for retrieval-time graph traversal."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import Document
from app.schemas.knowledge import Entity, GraphEdge, GraphNode, KnowledgeGraph, Relationship


class GraphService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def get_document_graph(self, document_id: str) -> KnowledgeGraph:
        with self._session_factory() as session:
            doc = session.execute(
                select(Document).where(Document.id == document_id)
            ).scalar_one_or_none()

        if not doc or not doc.knowledge_object:
            return KnowledgeGraph()

        kg = doc.knowledge_object.get("knowledge_graph", {})
        return KnowledgeGraph(**kg) if kg else KnowledgeGraph()

    def get_corpus_graph(self, corpus_id: str) -> KnowledgeGraph:
        with self._session_factory() as session:
            docs = session.execute(
                select(Document).where(Document.corpus_id == corpus_id)
            ).scalars().all()

        all_nodes: dict[str, GraphNode] = {}
        all_edges: list[GraphEdge] = []

        for doc in docs:
            if not doc.knowledge_object:
                continue
            kg = doc.knowledge_object.get("knowledge_graph", {})
            for node in kg.get("nodes", []):
                all_nodes[node["id"]] = GraphNode(**node)
            for edge in kg.get("edges", []):
                all_edges.append(GraphEdge(**edge))

        return KnowledgeGraph(nodes=list(all_nodes.values()), edges=all_edges)

    def lookup_entities(
        self,
        query: str,
        corpus_id: str | None = None,
        document_id: str | None = None,
        limit: int = 10,
    ) -> tuple[list[Entity], list[Relationship], KnowledgeGraph]:
        """Find entities matching query and expand their neighborhood."""
        q = query.lower()
        matched_entities: list[Entity] = []
        matched_rels: list[Relationship] = []

        with self._session_factory() as session:
            stmt = select(Document)
            if document_id:
                stmt = stmt.where(Document.id == document_id)
            elif corpus_id:
                stmt = stmt.where(Document.corpus_id == corpus_id)
            docs = session.execute(stmt).scalars().all()

        entity_map: dict[str, Entity] = {}
        all_rels: list[Relationship] = []

        for doc in docs:
            if not doc.knowledge_object:
                continue
            for e in doc.knowledge_object.get("entities", []):
                ent = Entity(**e)
                entity_map[ent.id] = ent
                text_match = (
                    q in ent.text.lower()
                    or q in (ent.normalized or "").lower()
                )
                if text_match:
                    matched_entities.append(ent)
            for r in doc.knowledge_object.get("relationships", []):
                all_rels.append(Relationship(**r))

        matched_ids = {e.id for e in matched_entities}
        for rel in all_rels:
            if rel.subject in matched_ids or rel.object in matched_ids:
                matched_rels.append(rel)
                matched_ids.add(rel.subject)
                matched_ids.add(rel.object)

        nodes = [
            GraphNode(id=eid, label=entity_map[eid].text, type=entity_map[eid].type)
            for eid in matched_ids
            if eid in entity_map
        ][:limit]
        edges = [
            GraphEdge(source=r.subject, target=r.object, predicate=r.predicate, confidence=r.confidence)
            for r in matched_rels
        ][:limit * 2]

        return matched_entities[:limit], matched_rels[:limit], KnowledgeGraph(nodes=nodes, edges=edges)

    def search_tables(
        self,
        query: str,
        corpus_id: str | None = None,
        document_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        q = query.lower()
        results: list[dict] = []

        with self._session_factory() as session:
            stmt = select(Document)
            if document_id:
                stmt = stmt.where(Document.id == document_id)
            elif corpus_id:
                stmt = stmt.where(Document.corpus_id == corpus_id)
            docs = session.execute(stmt).scalars().all()

        for doc in docs:
            if not doc.knowledge_object:
                continue
            for table in doc.knowledge_object.get("tables", []):
                searchable = " ".join(
                    [table.get("caption") or ""]
                    + table.get("headers", [])
                    + [" ".join(row) for row in table.get("rows", [])]
                ).lower()
                if any(term in searchable for term in q.split()):
                    results.append(
                        {
                            "document_id": doc.id,
                            "table_id": table.get("id"),
                            "caption": table.get("caption"),
                            "headers": table.get("headers"),
                            "rows": table.get("rows", [])[:5],
                            "raw_markdown": table.get("raw_markdown"),
                        }
                    )
            if len(results) >= limit:
                break

        return results[:limit]

    def traverse(
        self,
        start_entity_query: str,
        *,
        max_hops: int = 3,
        corpus_id: str | None = None,
        document_id: str | None = None,
        predicate_filter: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Multi-hop BFS from entities matching start_entity_query."""
        entities, rels, _ = self.lookup_entities(
            start_entity_query,
            corpus_id=corpus_id,
            document_id=document_id,
            limit=20,
        )
        if not entities:
            return {"start_entities": [], "paths": [], "nodes": [], "edges": []}

        with self._session_factory() as session:
            stmt = select(Document)
            if document_id:
                stmt = stmt.where(Document.id == document_id)
            elif corpus_id:
                stmt = stmt.where(Document.corpus_id == corpus_id)
            docs = session.execute(stmt).scalars().all()

        entity_map: dict[str, Entity] = {}
        adjacency: dict[str, list[Relationship]] = {}
        for doc in docs:
            if not doc.knowledge_object:
                continue
            for e in doc.knowledge_object.get("entities", []):
                ent = Entity(**e)
                entity_map[ent.id] = ent
            for r in doc.knowledge_object.get("relationships", []):
                rel = Relationship(**r)
                if predicate_filter and predicate_filter.lower() not in rel.predicate.lower():
                    continue
                adjacency.setdefault(rel.subject, []).append(rel)
                adjacency.setdefault(rel.object, []).append(rel)

        start_ids = {e.id for e in entities}
        visited = set(start_ids)
        frontier = list(start_ids)
        paths: list[list[dict]] = []
        collected_edges: list[GraphEdge] = []
        collected_nodes: dict[str, GraphNode] = {}

        for hop in range(max_hops):
            next_frontier: list[str] = []
            for node_id in frontier:
                for rel in adjacency.get(node_id, []):
                    other = rel.object if rel.subject == node_id else rel.subject
                    edge = GraphEdge(
                        source=rel.subject,
                        target=rel.object,
                        predicate=rel.predicate,
                        confidence=rel.confidence,
                    )
                    collected_edges.append(edge)
                    paths.append(
                        [
                            {"entity_id": rel.subject, "predicate": rel.predicate, "entity_id_to": rel.object},
                        ]
                    )
                    if other in entity_map:
                        collected_nodes[other] = GraphNode(
                            id=other,
                            label=entity_map[other].text,
                            type=entity_map[other].type,
                        )
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break

        for eid in visited:
            if eid in entity_map:
                collected_nodes[eid] = GraphNode(
                    id=eid, label=entity_map[eid].text, type=entity_map[eid].type
                )

        return {
            "start_entities": entities,
            "paths": paths[:limit],
            "nodes": list(collected_nodes.values())[:limit],
            "edges": collected_edges[: limit * 2],
        }

    @staticmethod
    def format_graph_context(entities: list[Entity], graph: KnowledgeGraph) -> str:
        if not entities and not graph.edges:
            return ""

        lines = ["## Knowledge Graph Context"]
        for ent in entities:
            lines.append(f"- [{ent.type}] {ent.text} (id={ent.id})")
        for edge in graph.edges:
            lines.append(f"- ({edge.source}) --[{edge.predicate}]--> ({edge.target})")
        return "\n".join(lines)
