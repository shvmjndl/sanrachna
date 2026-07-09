"""Knowledge graph builder from entities and relationships."""

from __future__ import annotations

import json
import uuid

from app.schemas.knowledge import Entity, GraphEdge, GraphNode, KnowledgeGraph, Relationship
from services.llm import LLMService


def build_graph_from_entities(entities: list[Entity]) -> KnowledgeGraph:
    nodes = [
        GraphNode(id=ent.id, label=ent.normalized or ent.text, type=ent.type)
        for ent in entities
    ]
    return KnowledgeGraph(nodes=nodes, edges=[])


async def extract_relationships(
    llm: LLMService,
    entities: list[Entity],
    text: str,
    job_id: str | None = None,
    document_id: str | None = None,
) -> tuple[list[Relationship], KnowledgeGraph]:
    if len(entities) < 2:
        graph = build_graph_from_entities(entities)
        return [], graph

    entity_summary = "\n".join(f"- {e.id}: {e.type} = {e.text}" for e in entities[:30])
    prompt = f"""Given these entities and document text, extract relationships.
Return JSON: {{"relationships": [{{"subject": "ent_id", "predicate": "RELATION_TYPE", "object": "ent_id", "confidence": 0.0-1.0}}]}}

Entities:
{entity_summary}

Text (excerpt):
{text[:2000]}"""

    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            operation="relationship_extraction",
            job_id=job_id,
            document_id=document_id,
        )
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        rels = [
            Relationship(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                confidence=r.get("confidence", 0.7),
            )
            for r in parsed.get("relationships", [])
            if r.get("subject") and r.get("object")
        ]
    except (json.JSONDecodeError, KeyError):
        rels = []

    nodes = [GraphNode(id=e.id, label=e.normalized or e.text, type=e.type) for e in entities]
    edges = [
        GraphEdge(
            source=r.subject,
            target=r.object,
            predicate=r.predicate,
            confidence=r.confidence,
        )
        for r in rels
    ]
    return rels, KnowledgeGraph(nodes=nodes, edges=edges)


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    node_map: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for graph in graphs:
        for node in graph.nodes:
            if node.id not in node_map:
                node_map[node.id] = node
        edges.extend(graph.edges)

    return KnowledgeGraph(nodes=list(node_map.values()), edges=edges)
